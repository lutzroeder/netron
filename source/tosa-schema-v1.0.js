
export const tosa = {};

tosa.DType = {
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

tosa.ResizeMode = {
    UNKNOWN: 0,
    NEAREST: 1,
    BILINEAR: 2
};

tosa.NanPropagationMode = {
    UNKNOWN: 0,
    PROPAGATE: 1,
    IGNORE: 2
};

tosa.RoundingMode = {
    UNKNOWN: 0,
    SINGLE_ROUND: 1,
    INEXACT_ROUND: 2,
    DOUBLE_ROUND: 3
};

tosa.Op = {
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

tosa.Attribute = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tosa.ArgMaxAttribute.decode(reader, position);
            case 2: return tosa.AvgPool2dAttribute.decode(reader, position);
            case 3: return tosa.Conv2dAttribute.decode(reader, position);
            case 4: return tosa.Conv3dAttribute.decode(reader, position);
            case 5: return tosa.DepthwiseConv2dAttribute.decode(reader, position);
            case 6: return tosa.FFT2dAttribute.decode(reader, position);
            case 7: return tosa.MatMulAttribute.decode(reader, position);
            case 8: return tosa.MaxPool2dAttribute.decode(reader, position);
            case 9: return tosa.RFFT2dAttribute.decode(reader, position);
            case 10: return tosa.TransposeConv2dAttribute.decode(reader, position);
            case 11: return tosa.ClampAttribute.decode(reader, position);
            case 12: return tosa.ErfAttribute.decode(reader, position);
            case 13: return tosa.SigmoidAttribute.decode(reader, position);
            case 14: return tosa.TanhAttribute.decode(reader, position);
            case 15: return tosa.AddAttribute.decode(reader, position);
            case 16: return tosa.ArithmeticRightShiftAttribute.decode(reader, position);
            case 17: return tosa.BitwiseAndAttribute.decode(reader, position);
            case 18: return tosa.BitwiseOrAttribute.decode(reader, position);
            case 19: return tosa.BitwiseXorAttribute.decode(reader, position);
            case 20: return tosa.IntDivAttribute.decode(reader, position);
            case 21: return tosa.LogicalAndAttribute.decode(reader, position);
            case 22: return tosa.LogicalLeftShiftAttribute.decode(reader, position);
            case 23: return tosa.LogicalRightShiftAttribute.decode(reader, position);
            case 24: return tosa.LogicalOrAttribute.decode(reader, position);
            case 25: return tosa.LogicalXorAttribute.decode(reader, position);
            case 26: return tosa.MaximumAttribute.decode(reader, position);
            case 27: return tosa.MinimumAttribute.decode(reader, position);
            case 28: return tosa.MulAttribute.decode(reader, position);
            case 29: return tosa.PowAttribute.decode(reader, position);
            case 30: return tosa.SubAttribute.decode(reader, position);
            case 31: return tosa.TableAttribute.decode(reader, position);
            case 32: return tosa.AbsAttribute.decode(reader, position);
            case 33: return tosa.BitwiseNotAttribute.decode(reader, position);
            case 34: return tosa.CeilAttribute.decode(reader, position);
            case 35: return tosa.ClzAttribute.decode(reader, position);
            case 36: return tosa.CosAttribute.decode(reader, position);
            case 37: return tosa.ExpAttribute.decode(reader, position);
            case 38: return tosa.FloorAttribute.decode(reader, position);
            case 39: return tosa.LogAttribute.decode(reader, position);
            case 40: return tosa.LogicalNotAttribute.decode(reader, position);
            case 41: return tosa.NegateAttribute.decode(reader, position);
            case 42: return tosa.ReciprocalAttribute.decode(reader, position);
            case 43: return tosa.RsqrtAttribute.decode(reader, position);
            case 44: return tosa.SinAttribute.decode(reader, position);
            case 45: return tosa.SelectAttribute.decode(reader, position);
            case 46: return tosa.EqualAttribute.decode(reader, position);
            case 47: return tosa.GreaterAttribute.decode(reader, position);
            case 48: return tosa.GreaterEqualAttribute.decode(reader, position);
            case 49: return tosa.ReduceAllAttribute.decode(reader, position);
            case 50: return tosa.ReduceAnyAttribute.decode(reader, position);
            case 51: return tosa.ReduceMaxAttribute.decode(reader, position);
            case 52: return tosa.ReduceMinAttribute.decode(reader, position);
            case 53: return tosa.ReduceProductAttribute.decode(reader, position);
            case 54: return tosa.ReduceSumAttribute.decode(reader, position);
            case 55: return tosa.ConcatAttribute.decode(reader, position);
            case 56: return tosa.PadAttribute.decode(reader, position);
            case 57: return tosa.ReshapeAttribute.decode(reader, position);
            case 58: return tosa.ReverseAttribute.decode(reader, position);
            case 59: return tosa.SliceAttribute.decode(reader, position);
            case 60: return tosa.TileAttribute.decode(reader, position);
            case 61: return tosa.TransposeAttribute.decode(reader, position);
            case 62: return tosa.GatherAttribute.decode(reader, position);
            case 63: return tosa.ScatterAttribute.decode(reader, position);
            case 64: return tosa.ResizeAttribute.decode(reader, position);
            case 65: return tosa.CastAttribute.decode(reader, position);
            case 66: return tosa.RescaleAttribute.decode(reader, position);
            case 67: return tosa.ConstAttribute.decode(reader, position);
            case 68: return tosa.IdentityAttribute.decode(reader, position);
            case 69: return tosa.CustomAttribute.decode(reader, position);
            case 70: return tosa.CondIfAttribute.decode(reader, position);
            case 71: return tosa.WhileLoopAttribute.decode(reader, position);
            case 72: return tosa.VariableAttribute.decode(reader, position);
            case 73: return tosa.VariableWriteAttribute.decode(reader, position);
            case 74: return tosa.VariableReadAttribute.decode(reader, position);
            case 75: return tosa.ConstShapeAttribute.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'ArgMaxAttribute': return tosa.ArgMaxAttribute.decodeText(reader, json);
            case 'AvgPool2dAttribute': return tosa.AvgPool2dAttribute.decodeText(reader, json);
            case 'Conv2dAttribute': return tosa.Conv2dAttribute.decodeText(reader, json);
            case 'Conv3dAttribute': return tosa.Conv3dAttribute.decodeText(reader, json);
            case 'DepthwiseConv2dAttribute': return tosa.DepthwiseConv2dAttribute.decodeText(reader, json);
            case 'FFT2dAttribute': return tosa.FFT2dAttribute.decodeText(reader, json);
            case 'MatMulAttribute': return tosa.MatMulAttribute.decodeText(reader, json);
            case 'MaxPool2dAttribute': return tosa.MaxPool2dAttribute.decodeText(reader, json);
            case 'RFFT2dAttribute': return tosa.RFFT2dAttribute.decodeText(reader, json);
            case 'TransposeConv2dAttribute': return tosa.TransposeConv2dAttribute.decodeText(reader, json);
            case 'ClampAttribute': return tosa.ClampAttribute.decodeText(reader, json);
            case 'ErfAttribute': return tosa.ErfAttribute.decodeText(reader, json);
            case 'SigmoidAttribute': return tosa.SigmoidAttribute.decodeText(reader, json);
            case 'TanhAttribute': return tosa.TanhAttribute.decodeText(reader, json);
            case 'AddAttribute': return tosa.AddAttribute.decodeText(reader, json);
            case 'ArithmeticRightShiftAttribute': return tosa.ArithmeticRightShiftAttribute.decodeText(reader, json);
            case 'BitwiseAndAttribute': return tosa.BitwiseAndAttribute.decodeText(reader, json);
            case 'BitwiseOrAttribute': return tosa.BitwiseOrAttribute.decodeText(reader, json);
            case 'BitwiseXorAttribute': return tosa.BitwiseXorAttribute.decodeText(reader, json);
            case 'IntDivAttribute': return tosa.IntDivAttribute.decodeText(reader, json);
            case 'LogicalAndAttribute': return tosa.LogicalAndAttribute.decodeText(reader, json);
            case 'LogicalLeftShiftAttribute': return tosa.LogicalLeftShiftAttribute.decodeText(reader, json);
            case 'LogicalRightShiftAttribute': return tosa.LogicalRightShiftAttribute.decodeText(reader, json);
            case 'LogicalOrAttribute': return tosa.LogicalOrAttribute.decodeText(reader, json);
            case 'LogicalXorAttribute': return tosa.LogicalXorAttribute.decodeText(reader, json);
            case 'MaximumAttribute': return tosa.MaximumAttribute.decodeText(reader, json);
            case 'MinimumAttribute': return tosa.MinimumAttribute.decodeText(reader, json);
            case 'MulAttribute': return tosa.MulAttribute.decodeText(reader, json);
            case 'PowAttribute': return tosa.PowAttribute.decodeText(reader, json);
            case 'SubAttribute': return tosa.SubAttribute.decodeText(reader, json);
            case 'TableAttribute': return tosa.TableAttribute.decodeText(reader, json);
            case 'AbsAttribute': return tosa.AbsAttribute.decodeText(reader, json);
            case 'BitwiseNotAttribute': return tosa.BitwiseNotAttribute.decodeText(reader, json);
            case 'CeilAttribute': return tosa.CeilAttribute.decodeText(reader, json);
            case 'ClzAttribute': return tosa.ClzAttribute.decodeText(reader, json);
            case 'CosAttribute': return tosa.CosAttribute.decodeText(reader, json);
            case 'ExpAttribute': return tosa.ExpAttribute.decodeText(reader, json);
            case 'FloorAttribute': return tosa.FloorAttribute.decodeText(reader, json);
            case 'LogAttribute': return tosa.LogAttribute.decodeText(reader, json);
            case 'LogicalNotAttribute': return tosa.LogicalNotAttribute.decodeText(reader, json);
            case 'NegateAttribute': return tosa.NegateAttribute.decodeText(reader, json);
            case 'ReciprocalAttribute': return tosa.ReciprocalAttribute.decodeText(reader, json);
            case 'RsqrtAttribute': return tosa.RsqrtAttribute.decodeText(reader, json);
            case 'SinAttribute': return tosa.SinAttribute.decodeText(reader, json);
            case 'SelectAttribute': return tosa.SelectAttribute.decodeText(reader, json);
            case 'EqualAttribute': return tosa.EqualAttribute.decodeText(reader, json);
            case 'GreaterAttribute': return tosa.GreaterAttribute.decodeText(reader, json);
            case 'GreaterEqualAttribute': return tosa.GreaterEqualAttribute.decodeText(reader, json);
            case 'ReduceAllAttribute': return tosa.ReduceAllAttribute.decodeText(reader, json);
            case 'ReduceAnyAttribute': return tosa.ReduceAnyAttribute.decodeText(reader, json);
            case 'ReduceMaxAttribute': return tosa.ReduceMaxAttribute.decodeText(reader, json);
            case 'ReduceMinAttribute': return tosa.ReduceMinAttribute.decodeText(reader, json);
            case 'ReduceProductAttribute': return tosa.ReduceProductAttribute.decodeText(reader, json);
            case 'ReduceSumAttribute': return tosa.ReduceSumAttribute.decodeText(reader, json);
            case 'ConcatAttribute': return tosa.ConcatAttribute.decodeText(reader, json);
            case 'PadAttribute': return tosa.PadAttribute.decodeText(reader, json);
            case 'ReshapeAttribute': return tosa.ReshapeAttribute.decodeText(reader, json);
            case 'ReverseAttribute': return tosa.ReverseAttribute.decodeText(reader, json);
            case 'SliceAttribute': return tosa.SliceAttribute.decodeText(reader, json);
            case 'TileAttribute': return tosa.TileAttribute.decodeText(reader, json);
            case 'TransposeAttribute': return tosa.TransposeAttribute.decodeText(reader, json);
            case 'GatherAttribute': return tosa.GatherAttribute.decodeText(reader, json);
            case 'ScatterAttribute': return tosa.ScatterAttribute.decodeText(reader, json);
            case 'ResizeAttribute': return tosa.ResizeAttribute.decodeText(reader, json);
            case 'CastAttribute': return tosa.CastAttribute.decodeText(reader, json);
            case 'RescaleAttribute': return tosa.RescaleAttribute.decodeText(reader, json);
            case 'ConstAttribute': return tosa.ConstAttribute.decodeText(reader, json);
            case 'IdentityAttribute': return tosa.IdentityAttribute.decodeText(reader, json);
            case 'CustomAttribute': return tosa.CustomAttribute.decodeText(reader, json);
            case 'CondIfAttribute': return tosa.CondIfAttribute.decodeText(reader, json);
            case 'WhileLoopAttribute': return tosa.WhileLoopAttribute.decodeText(reader, json);
            case 'VariableAttribute': return tosa.VariableAttribute.decodeText(reader, json);
            case 'VariableWriteAttribute': return tosa.VariableWriteAttribute.decodeText(reader, json);
            case 'VariableReadAttribute': return tosa.VariableReadAttribute.decodeText(reader, json);
            case 'ConstShapeAttribute': return tosa.ConstShapeAttribute.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tosa.ArgMaxAttribute = class ArgMaxAttribute {

    static decode(reader, position) {
        const $ = new tosa.ArgMaxAttribute();
        $.axis = reader.int32_(position, 4, 0);
        $.nan_mode = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ArgMaxAttribute();
        $.axis = reader.value(json.axis, 0);
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.AvgPool2dAttribute = class AvgPool2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.AvgPool2dAttribute();
        $.kernel = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.pad = reader.array(position, 8, Int32Array);
        $.acc_type = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.AvgPool2dAttribute();
        $.kernel = reader.array(json.kernel, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.pad = reader.array(json.pad, Int32Array);
        $.acc_type = tosa.DType[json.acc_type];
        return $;
    }
};

tosa.Conv2dAttribute = class Conv2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.Conv2dAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.local_bound = reader.bool_(position, 10, false);
        $.acc_type = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.Conv2dAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.DType[json.acc_type];
        return $;
    }
};

tosa.Conv3dAttribute = class Conv3dAttribute {

    static decode(reader, position) {
        const $ = new tosa.Conv3dAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.local_bound = reader.bool_(position, 10, false);
        $.acc_type = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.Conv3dAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.DType[json.acc_type];
        return $;
    }
};

tosa.DepthwiseConv2dAttribute = class DepthwiseConv2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.DepthwiseConv2dAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.local_bound = reader.bool_(position, 10, false);
        $.acc_type = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.DepthwiseConv2dAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.DType[json.acc_type];
        return $;
    }
};

tosa.FFT2dAttribute = class FFT2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.FFT2dAttribute();
        $.inverse = reader.bool_(position, 4, false);
        $.local_bound = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.FFT2dAttribute();
        $.inverse = reader.value(json.inverse, false);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.MatMulAttribute = class MatMulAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.MatMulAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.MatMulAttribute();
        return $;
    }
};

tosa.MaxPool2dAttribute = class MaxPool2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.MaxPool2dAttribute();
        $.kernel = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.pad = reader.array(position, 8, Int32Array);
        $.nan_mode = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.MaxPool2dAttribute();
        $.kernel = reader.array(json.kernel, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.pad = reader.array(json.pad, Int32Array);
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.RFFT2dAttribute = class RFFT2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.RFFT2dAttribute();
        $.local_bound = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.RFFT2dAttribute();
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.TransposeConv2dAttribute = class TransposeConv2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.TransposeConv2dAttribute();
        $.out_pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.local_bound = reader.bool_(position, 8, false);
        $.acc_type = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TransposeConv2dAttribute();
        $.out_pad = reader.array(json.out_pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.DType[json.acc_type];
        return $;
    }
};

tosa.ClampAttribute = class ClampAttribute {

    static decode(reader, position) {
        const $ = new tosa.ClampAttribute();
        $.min_val = reader.array(position, 4, Uint8Array);
        $.max_val = reader.array(position, 6, Uint8Array);
        $.nan_mode = reader.uint32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ClampAttribute();
        $.min_val = reader.array(json.min_val, Uint8Array);
        $.max_val = reader.array(json.max_val, Uint8Array);
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.ErfAttribute = class ErfAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ErfAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ErfAttribute();
        return $;
    }
};

tosa.SigmoidAttribute = class SigmoidAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.SigmoidAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.SigmoidAttribute();
        return $;
    }
};

tosa.TanhAttribute = class TanhAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.TanhAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.TanhAttribute();
        return $;
    }
};

tosa.AddAttribute = class AddAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.AddAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.AddAttribute();
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

tosa.BitwiseAndAttribute = class BitwiseAndAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.BitwiseAndAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.BitwiseAndAttribute();
        return $;
    }
};

tosa.BitwiseOrAttribute = class BitwiseOrAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.BitwiseOrAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.BitwiseOrAttribute();
        return $;
    }
};

tosa.BitwiseXorAttribute = class BitwiseXorAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.BitwiseXorAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.BitwiseXorAttribute();
        return $;
    }
};

tosa.IntDivAttribute = class IntDivAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.IntDivAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.IntDivAttribute();
        return $;
    }
};

tosa.LogicalAndAttribute = class LogicalAndAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogicalAndAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogicalAndAttribute();
        return $;
    }
};

tosa.LogicalLeftShiftAttribute = class LogicalLeftShiftAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogicalLeftShiftAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogicalLeftShiftAttribute();
        return $;
    }
};

tosa.LogicalRightShiftAttribute = class LogicalRightShiftAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogicalRightShiftAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogicalRightShiftAttribute();
        return $;
    }
};

tosa.LogicalOrAttribute = class LogicalOrAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogicalOrAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogicalOrAttribute();
        return $;
    }
};

tosa.LogicalXorAttribute = class LogicalXorAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogicalXorAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogicalXorAttribute();
        return $;
    }
};

tosa.MaximumAttribute = class MaximumAttribute {

    static decode(reader, position) {
        const $ = new tosa.MaximumAttribute();
        $.nan_mode = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.MaximumAttribute();
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.MinimumAttribute = class MinimumAttribute {

    static decode(reader, position) {
        const $ = new tosa.MinimumAttribute();
        $.nan_mode = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.MinimumAttribute();
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.MulAttribute = class MulAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.MulAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.MulAttribute();
        return $;
    }
};

tosa.PowAttribute = class PowAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.PowAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.PowAttribute();
        return $;
    }
};

tosa.SubAttribute = class SubAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.SubAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.SubAttribute();
        return $;
    }
};

tosa.TableAttribute = class TableAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.TableAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.TableAttribute();
        return $;
    }
};

tosa.AbsAttribute = class AbsAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.AbsAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.AbsAttribute();
        return $;
    }
};

tosa.BitwiseNotAttribute = class BitwiseNotAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.BitwiseNotAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.BitwiseNotAttribute();
        return $;
    }
};

tosa.CeilAttribute = class CeilAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.CeilAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.CeilAttribute();
        return $;
    }
};

tosa.ClzAttribute = class ClzAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ClzAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ClzAttribute();
        return $;
    }
};

tosa.CosAttribute = class CosAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.CosAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.CosAttribute();
        return $;
    }
};

tosa.ExpAttribute = class ExpAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ExpAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ExpAttribute();
        return $;
    }
};

tosa.FloorAttribute = class FloorAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.FloorAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.FloorAttribute();
        return $;
    }
};

tosa.LogAttribute = class LogAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogAttribute();
        return $;
    }
};

tosa.LogicalNotAttribute = class LogicalNotAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.LogicalNotAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.LogicalNotAttribute();
        return $;
    }
};

tosa.NegateAttribute = class NegateAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.NegateAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.NegateAttribute();
        return $;
    }
};

tosa.ReciprocalAttribute = class ReciprocalAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ReciprocalAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ReciprocalAttribute();
        return $;
    }
};

tosa.RsqrtAttribute = class RsqrtAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.RsqrtAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.RsqrtAttribute();
        return $;
    }
};

tosa.SinAttribute = class SinAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.SinAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.SinAttribute();
        return $;
    }
};

tosa.SelectAttribute = class SelectAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.SelectAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.SelectAttribute();
        return $;
    }
};

tosa.EqualAttribute = class EqualAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.EqualAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.EqualAttribute();
        return $;
    }
};

tosa.GreaterAttribute = class GreaterAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.GreaterAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.GreaterAttribute();
        return $;
    }
};

tosa.GreaterEqualAttribute = class GreaterEqualAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.GreaterEqualAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.GreaterEqualAttribute();
        return $;
    }
};

tosa.ReduceAllAttribute = class ReduceAllAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReduceAllAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReduceAllAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.ReduceAnyAttribute = class ReduceAnyAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReduceAnyAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReduceAnyAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.ReduceMaxAttribute = class ReduceMaxAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReduceMaxAttribute();
        $.axis = reader.int32_(position, 4, 0);
        $.nan_mode = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReduceMaxAttribute();
        $.axis = reader.value(json.axis, 0);
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.ReduceMinAttribute = class ReduceMinAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReduceMinAttribute();
        $.axis = reader.int32_(position, 4, 0);
        $.nan_mode = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReduceMinAttribute();
        $.axis = reader.value(json.axis, 0);
        $.nan_mode = tosa.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.ReduceProductAttribute = class ReduceProductAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReduceProductAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReduceProductAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.ReduceSumAttribute = class ReduceSumAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReduceSumAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReduceSumAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.ConcatAttribute = class ConcatAttribute {

    static decode(reader, position) {
        const $ = new tosa.ConcatAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ConcatAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.PadAttribute = class PadAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.PadAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.PadAttribute();
        return $;
    }
};

tosa.ReshapeAttribute = class ReshapeAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ReshapeAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ReshapeAttribute();
        return $;
    }
};

tosa.ReverseAttribute = class ReverseAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReverseAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReverseAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.SliceAttribute = class SliceAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.SliceAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.SliceAttribute();
        return $;
    }
};

tosa.TileAttribute = class TileAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.TileAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.TileAttribute();
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

tosa.GatherAttribute = class GatherAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.GatherAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.GatherAttribute();
        return $;
    }
};

tosa.ScatterAttribute = class ScatterAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ScatterAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ScatterAttribute();
        return $;
    }
};

tosa.ResizeAttribute = class ResizeAttribute {

    static decode(reader, position) {
        const $ = new tosa.ResizeAttribute();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ResizeAttribute();
        $.mode = tosa.ResizeMode[json.mode];
        return $;
    }
};

tosa.CastAttribute = class CastAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.CastAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.CastAttribute();
        return $;
    }
};

tosa.RescaleAttribute = class RescaleAttribute {

    static decode(reader, position) {
        const $ = new tosa.RescaleAttribute();
        $.scale32 = reader.bool_(position, 4, false);
        $.rounding_mode = reader.uint32_(position, 6, 0);
        $.per_channel = reader.bool_(position, 8, false);
        $.input_unsigned = reader.bool_(position, 10, false);
        $.output_unsigned = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.RescaleAttribute();
        $.scale32 = reader.value(json.scale32, false);
        $.rounding_mode = tosa.RoundingMode[json.rounding_mode];
        $.per_channel = reader.value(json.per_channel, false);
        $.input_unsigned = reader.value(json.input_unsigned, false);
        $.output_unsigned = reader.value(json.output_unsigned, false);
        return $;
    }
};

tosa.ConstAttribute = class ConstAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ConstAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ConstAttribute();
        return $;
    }
};

tosa.IdentityAttribute = class IdentityAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.IdentityAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.IdentityAttribute();
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

tosa.CondIfAttribute = class CondIfAttribute {

    static decode(reader, position) {
        const $ = new tosa.CondIfAttribute();
        $.then_graph = reader.string_(position, 4, null);
        $.else_graph = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.CondIfAttribute();
        $.then_graph = reader.value(json.then_graph, null);
        $.else_graph = reader.value(json.else_graph, null);
        return $;
    }
};

tosa.WhileLoopAttribute = class WhileLoopAttribute {

    static decode(reader, position) {
        const $ = new tosa.WhileLoopAttribute();
        $.cond_graph = reader.string_(position, 4, null);
        $.body_graph = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.WhileLoopAttribute();
        $.cond_graph = reader.value(json.cond_graph, null);
        $.body_graph = reader.value(json.body_graph, null);
        return $;
    }
};

tosa.VariableAttribute = class VariableAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.VariableAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.VariableAttribute();
        return $;
    }
};

tosa.VariableWriteAttribute = class VariableWriteAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.VariableWriteAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.VariableWriteAttribute();
        return $;
    }
};

tosa.VariableReadAttribute = class VariableReadAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.VariableReadAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.VariableReadAttribute();
        return $;
    }
};

tosa.ConstShapeAttribute = class ConstShapeAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.ConstShapeAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.ConstShapeAttribute();
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

tosa.TosaShape = class TosaShape {

    static decode(reader, position) {
        const $ = new tosa.TosaShape();
        $.name = reader.string_(position, 4, null);
        $.rank = reader.uint32_(position, 6, 0);
        $.data = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaShape();
        $.name = reader.value(json.name, null);
        $.rank = reader.value(json.rank, 0);
        $.data = reader.array(json.data, Uint8Array);
        return $;
    }
};

tosa.OpLocation = class OpLocation {

    static decode(reader, position) {
        const $ = new tosa.OpLocation();
        $.text = reader.string_(position, 4, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.OpLocation();
        $.text = reader.value(json.text, null);
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
        $.location = reader.table(position, 14, tosa.OpLocation);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaOperator();
        $.op = tosa.Op[json.op];
        $.attribute = tosa.Attribute.decodeText(reader, json.attribute, json.attribute_type);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        $.location = reader.object(json.location, tosa.OpLocation);
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
        $.shapes = reader.tables(position, 14, tosa.TosaShape);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaBasicBlock();
        $.name = reader.value(json.name, null);
        $.operators = reader.objects(json.operators, tosa.TosaOperator);
        $.tensors = reader.objects(json.tensors, tosa.TosaTensor);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        $.shapes = reader.objects(json.shapes, tosa.TosaShape);
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
