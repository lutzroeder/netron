var $root = flatbuffers.get('wnnx');

$root.wnn = $root.wnn || {};

$root.wnn.DataType = {
    DT_INVALID: 0,
    DT_FLOAT: 1,
    DT_DOUBLE: 2,
    DT_INT32: 3,
    DT_UINT8: 4,
    DT_INT16: 5,
    DT_INT8: 6,
    DT_STRING: 7,
    DT_COMPLEX64: 8,
    DT_INT64: 9,
    DT_BOOL: 10,
    DT_QINT8: 11,
    DT_QUINT8: 12,
    DT_QINT32: 13,
    DT_BFLOAT16: 14,
    DT_QINT16: 15,
    DT_QUINT16: 16,
    DT_UINT16: 17,
    DT_COMPLEX128: 18,
    DT_FLOAT16: 19,
    DT_TFLOAT32: 20
};

$root.wnn.DeviceType = {
    kX86: 0,
    kARM: 1,
    kOPENCL: 2,
    kMETAL: 3,
    kCUDA: 4,
    kDSP: 5,
    kATLAS: 6,
    kHUAWEI_NPU: 7,
    kRK_NPU: 8,
    kAPPLE_NPU: 9,
    kCPU: 10
};

$root.wnn.QuantType = {
    no_quant: 0,
    int8: 1,
    sparse_quant: 2,
    fp16: 3,
    bfp16: 4,
    weight_int8: 5,
    int4: 6
};

$root.wnn.Quant = class Quant {

    static decode(reader, position) {
        const $ = new $root.wnn.Quant();
        $.buffer = reader.typedArray(position, 4, Int8Array);
        $.alpha = reader.typedArray(position, 6, Float32Array);
        $.quant_type = reader.int8_(position, 8, 0);
        $.use_int32 = reader.bool_(position, 10, false);
        $.quant_scale = reader.float32_(position, 12, 0);
        $.scale_in = reader.float32_(position, 14, 0);
        $.scale_out = reader.float32_(position, 16, 0);
        $.a_max = reader.int32_(position, 18, 0);
        $.a_min = reader.int32_(position, 20, 0);
        $.read_type = reader.int32_(position, 22, 0);
        $.has_scale_int = reader.bool_(position, 24, false);
        return $;
    }
};

$root.wnn.WNN_DATA_FORMAT = {
    NCHW: 0,
    NHWC: 1,
    NC4HW4: 2,
    NHWC4: 3,
    UNKNOWN: 4
};

$root.wnn.Blob = class Blob {

    static decode(reader, position) {
        const $ = new $root.wnn.Blob();
        $.dims = reader.typedArray(position, 4, Int32Array);
        $.dataformat = reader.int8_(position, 6, 0);
        $.dtype = reader.int32_(position, 8, 1);
        $.device = reader.int32_(position, 10, 0);
        $.uint8s = reader.typedArray(position, 12, Uint8Array);
        $.int8s = reader.typedArray(position, 14, Uint8Array);
        $.int32s = reader.typedArray(position, 16, Int32Array);
        $.int64s = reader.int64s_(position, 18);
        $.float32s = reader.typedArray(position, 20, Float32Array);
        $.strings = reader.strings_(position, 22);
        return $;
    }
};

$root.wnn.ListValue = class ListValue {

    static decode(reader, position) {
        const $ = new $root.wnn.ListValue();
        $.s = reader.strings_(position, 4);
        $.i = reader.typedArray(position, 6, Int32Array);
        $.f = reader.typedArray(position, 8, Float32Array);
        $.b = reader.bools_(position, 10);
        $.type = reader.typedArray(position, 12, Int32Array);
        return $;
    }
};

$root.wnn.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new $root.wnn.Attribute();
        $.s = reader.string_(position, 4, null);
        $.i = reader.int32_(position, 6, 0);
        $.b = reader.bool_(position, 8, false);
        $.key = reader.string_(position, 10, null);
        $.type = reader.int32_(position, 12, 0);
        $.f = reader.float32_(position, 14, 0);
        $.blob = reader.table(position, 16, $root.wnn.Blob.decode);
        $.list = reader.table(position, 18, $root.wnn.ListValue.decode);
        $.func = reader.table(position, 20, $root.wnn.NamedAttrList.decode);
        $.shape = reader.typedArray(position, 22, Int32Array);
        $.data = reader.typedArray(position, 24, Int8Array);
        return $;
    }
};

$root.wnn.NamedAttrList = class NamedAttrList {

    static decode(reader, position) {
        const $ = new $root.wnn.NamedAttrList();
        $.name = reader.string_(position, 4, null);
        $.attr = reader.tableArray(position, 6, $root.wnn.Attribute.decode);
        return $;
    }
};

$root.wnn.OpType = {
    argmax: 0,
    argmin: 1,
    const: 2,
    conv1d: 3,
    conv2d: 4,
    conv3d: 5,
    pool2d: 6,
    pool3d: 7,
    adaptive_avg_pool2d: 8,
    batchnorm: 9,
    layernorm: 10,
    relu: 11,
    relu6: 12,
    elu: 13,
    prelu: 14,
    leakyrelu: 15,
    tanh: 16,
    silu: 17,
    mish: 18,
    hardswish: 19,
    hardsigmoid: 20,
    sigmoid: 21,
    fc: 22,
    flatten: 23,
    matmul: 24,
    fc_share: 25,
    lstm: 26,
    onehot: 27,
    transpose: 28,
    gather: 29,
    split: 30,
    concat: 31,
    activation: 32,
    binary_op: 33,
    fill: 34,
    pad: 35,
    reshape: 36,
    instancenorm: 37,
    conv_depthwise: 38,
    quantized_avgpool: 39,
    quantized_concat: 40,
    quantized_matmul: 41,
    quantized_relu: 42,
    quantized_relu6: 43,
    quantized_softmax: 44,
    roipooling: 45,
    roialign: 46,
    unary: 47,
    unary_square: 48,
    unary_sqrt: 49,
    binary: 50,
    binary_add: 51,
    binary_mul: 52,
    binary_div: 53,
    binary_sub: 54,
    softmax: 55,
    scatternd: 56,
    gathernd: 57,
    nms: 58,
    input: 59,
    output: 60,
    extra: 61,
    eltwise: 62,
    reduction: 63,
    expand_dims: 64,
    normalize: 65,
    unsupported: 66,
    film_lpn: 67,
    cubic: 68
};

$root.wnn.PadMode = {
    CAFFE: 0,
    VALID: 1,
    SAME: 2
};

$root.wnn.Conv2DCommon = class Conv2DCommon {

    static decode(reader, position) {
        const $ = new $root.wnn.Conv2DCommon();
        $.pad_x = reader.int32_(position, 4, 0);
        $.pad_y = reader.int32_(position, 6, 0);
        $.kernel_x = reader.int32_(position, 8, 1);
        $.kernel_y = reader.int32_(position, 10, 1);
        $.stride_x = reader.int32_(position, 12, 1);
        $.stride_y = reader.int32_(position, 14, 1);
        $.dilate_x = reader.int32_(position, 16, 1);
        $.dilate_y = reader.int32_(position, 18, 1);
        $.padmode = reader.int8_(position, 20, 2);
        $.group = reader.int32_(position, 22, 1);
        $.output_count = reader.int32_(position, 24, 0);
        $.input_count = reader.int32_(position, 26, 0);
        $.sparse_output_count = reader.int32_(position, 28, 0);
        $.in_channels = reader.int32_(position, 30, 0);
        $.out_channels = reader.int32_(position, 32, 0);
        $.relu = reader.bool_(position, 34, false);
        $.relu6 = reader.bool_(position, 36, false);
        $.pads = reader.typedArray(position, 38, Int32Array);
        $.out_pads = reader.typedArray(position, 40, Int32Array);
        $.has_outputshape = reader.bool_(position, 42, false);
        return $;
    }
};

$root.wnn.ActivationType = {
    RELU: 0,
    RELU6: 1,
    LEAKY_RELU: 2,
    ELU: 3,
    TANH: 4,
    PRELU: 5,
    MISH: 6,
    SWISH: 7
};

$root.wnn.Conv2D = class Conv2D {

    static decode(reader, position) {
        const $ = new $root.wnn.Conv2D();
        $.common = reader.table(position, 4, $root.wnn.Conv2DCommon.decode);
        $.weight = reader.typedArray(position, 6, Float32Array);
        $.bias = reader.typedArray(position, 8, Float32Array);
        $.has_act = reader.bool_(position, 10, false);
        $.act_type = reader.int32_(position, 12, 0);
        $.input_shape = reader.typedArray(position, 14, Int32Array);
        $.quant_param = reader.table(position, 16, $root.wnn.Quant.decode);
        return $;
    }
};

$root.wnn.PoolType = {
    MAXPOOL: 0,
    AVEPOOL: 1
};

$root.wnn.PoolPadType = {
    CAFFE: 0,
    VALID: 1,
    SAME: 2
};

$root.wnn.AvgPoolCountType = {
    DEFAULT: 0,
    INCLUDE_PADDING: 1,
    EXCLUDE_PADDING: 2
};

$root.wnn.Pool = class Pool {

    static decode(reader, position) {
        const $ = new $root.wnn.Pool();
        $.pad_x = reader.int32_(position, 4, 0);
        $.pad_y = reader.int32_(position, 6, 0);
        $.is_global = reader.bool_(position, 8, false);
        $.kernel_x = reader.int32_(position, 10, 0);
        $.kernel_y = reader.int32_(position, 12, 0);
        $.stride_x = reader.int32_(position, 14, 0);
        $.stride_y = reader.int32_(position, 16, 0);
        $.type = reader.int8_(position, 18, 0);
        $.pad_type = reader.int8_(position, 20, 0);
        $.data_type = reader.int32_(position, 22, 1);
        $.ceil_model = reader.bool_(position, 24, true);
        $.pads = reader.typedArray(position, 26, Int32Array);
        $.count_type = reader.int8_(position, 28, 0);
        $.in_channels = reader.int32_(position, 30, 0);
        $.in_width = reader.int32_(position, 32, 0);
        $.in_height = reader.int32_(position, 34, 0);
        $.is_adaptive = reader.bool_(position, 36, false);
        $.out_height = reader.int32_(position, 38, 0);
        $.out_width = reader.int32_(position, 40, 0);
        return $;
    }
};

$root.wnn.AdaptiveAvgPool2D = class AdaptiveAvgPool2D {

    static decode(reader, position) {
        const $ = new $root.wnn.AdaptiveAvgPool2D();
        $.out_h = reader.int32_(position, 4, 0);
        $.out_w = reader.int32_(position, 6, 0);
        $.data_type = reader.int32_(position, 8, 1);
        return $;
    }
};

$root.wnn.LayerNorm = class LayerNorm {

    static decode(reader, position) {
        const $ = new $root.wnn.LayerNorm();
        $.axis = reader.typedArray(position, 4, Int32Array);
        $.epsilon = reader.float32_(position, 6, 0);
        $.gamma = reader.typedArray(position, 8, Float32Array);
        $.beta = reader.typedArray(position, 10, Float32Array);
        $.group = reader.int32_(position, 12, 1);
        $.elmentwise_affine = reader.bool_(position, 14, true);
        return $;
    }
};

$root.wnn.BatchNorm = class BatchNorm {

    static decode(reader, position) {
        const $ = new $root.wnn.BatchNorm();
        $.channels = reader.int32_(position, 4, 0);
        $.slope_data = reader.typedArray(position, 6, Float32Array);
        $.mean_data = reader.typedArray(position, 8, Float32Array);
        $.var_data = reader.typedArray(position, 10, Float32Array);
        $.bias_data = reader.typedArray(position, 12, Float32Array);
        $.a_data = reader.typedArray(position, 14, Float32Array);
        $.b_data = reader.typedArray(position, 16, Float32Array);
        $.epsilon = reader.float32_(position, 18, 0.001);
        return $;
    }
};

$root.wnn.Relu = class Relu {

    static decode(reader, position) {
        const $ = new $root.wnn.Relu();
        $.slope = reader.float32_(position, 4, 0);
        return $;
    }
};

$root.wnn.Relu6 = class Relu6 {

    static decode(reader, position) {
        const $ = new $root.wnn.Relu6();
        $.min_value = reader.float32_(position, 4, 0);
        $.max_value = reader.float32_(position, 6, 6);
        return $;
    }
};

$root.wnn.Softmax = class Softmax {

    static decode(reader, position) {
        const $ = new $root.wnn.Softmax();
        $.dim = reader.int32_(position, 4, 1);
        return $;
    }
};

$root.wnn.PRelu = class PRelu {

    static decode(reader, position) {
        const $ = new $root.wnn.PRelu();
        $.slope_count = reader.int32_(position, 4, 0);
        $.slope = reader.typedArray(position, 6, Float32Array);
        return $;
    }
};

$root.wnn.ELU = class ELU {

    static decode(reader, position) {
        const $ = new $root.wnn.ELU();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }
};

$root.wnn.LRN = class LRN {

    static decode(reader, position) {
        const $ = new $root.wnn.LRN();
        $.region_type = reader.int32_(position, 4, 0);
        $.local_size = reader.int32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        $.bias = reader.float32_(position, 12, 1);
        return $;
    }
};

$root.wnn.FC = class FC {

    static decode(reader, position) {
        const $ = new $root.wnn.FC();
        $.in_features = reader.int32_(position, 4, 0);
        $.out_features = reader.int32_(position, 6, 0);
        $.weight_size = reader.int32_(position, 8, 0);
        $.weights = reader.typedArray(position, 10, Float32Array);
        $.bias = reader.typedArray(position, 12, Float32Array);
        $.axis = reader.int32_(position, 14, 0);
        $.transpose = reader.bool_(position, 16, false);
        $.has_act = reader.bool_(position, 18, false);
        $.act_type = reader.int32_(position, 20, 0);
        $.act_params = reader.typedArray(position, 22, Float32Array);
        $.quant_param = reader.table(position, 24, $root.wnn.Quant.decode);
        return $;
    }
};

$root.wnn.Input = class Input {

    static decode(reader, position) {
        const $ = new $root.wnn.Input();
        $.dims = reader.typedArray(position, 4, Int32Array);
        $.dtype = reader.int32_(position, 6, 1);
        $.dformat = reader.int8_(position, 8, 0);
        return $;
    }
};

$root.wnn.ArgMax = class ArgMax {

    static decode(reader, position) {
        const $ = new $root.wnn.ArgMax();
        $.out_max_val = reader.int32_(position, 4, 0);
        $.top_k = reader.int32_(position, 6, 0);
        $.axis = reader.int32_(position, 8, 0);
        $.softmax_thresh = reader.int32_(position, 10, 0);
        return $;
    }
};

$root.wnn.BinaryOperation = {
    ADD: 0,
    SUB: 1,
    MUL: 2,
    DIV: 3,
    MAX_TEMP: 4,
    MIN_TEMP: 5,
    POW: 6,
    REALDIV: 7,
    MINIMUM: 8,
    MAXIMUM: 9,
    GREATER: 10,
    GREATER_EQUAL: 11,
    LESS: 12,
    FLOORDIV: 13,
    SquaredDifference: 14,
    EQUAL: 15,
    LESS_EQUAL: 16,
    FLOORMOD: 17,
    MOD: 19,
    ATAN2: 20,
    LOGICALOR: 21,
    NOTEQUAL: 22,
    BITWISE_AND: 23,
    BITWISE_OR: 24,
    BITWISE_XOR: 25,
    LOGICALXOR: 26,
    LEFTSHIFT: 27,
    RIGHTSHIFT: 28,
    RSUB: 29
};

$root.wnn.Binary = class Binary {

    static decode(reader, position) {
        const $ = new $root.wnn.Binary();
        $.operation_type = reader.int8_(position, 4, 0);
        $.dtype = reader.int32_(position, 6, 1);
        return $;
    }
};

$root.wnn.UnaryOperation = {
    ABS: 0,
    NEG: 1,
    FLOOR: 2,
    CEIL: 3,
    SQUARE: 4,
    SQRT: 5,
    RSQRT: 6,
    EXP: 7,
    LOG: 8,
    SIN: 9,
    COS: 10,
    TAN: 11,
    ASIN: 12,
    ACOS: 13,
    ATAN: 14,
    RECIPROCAL: 15,
    LOG1P: 16,
    BNLL: 17,
    ACOSH: 18,
    SINH: 19,
    ASINH: 20,
    ATANH: 21,
    SIGN: 22,
    ROUND: 23,
    COSH: 24,
    ERF: 25,
    ERFC: 26,
    ERFINV: 27,
    EXPM1: 28,
    SIGMOID: 29,
    TANH: 30,
    HARDSWISH: 31,
    GELU: 32,
    GELU_STANDARD: 33,
    NOT: 34,
    BOOL: 35
};

$root.wnn.Unary = class Unary {

    static decode(reader, position) {
        const $ = new $root.wnn.Unary();
        $.operation_type = reader.int32_(position, 4, 0);
        $.dtype = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.wnn.EltwiseType = {
    PROD: 0,
    SUM: 1,
    MAXIUM: 2,
    SUB: 3,
    SOFTMAX: 4
};

$root.wnn.Eltwise = class Eltwise {

    static decode(reader, position) {
        const $ = new $root.wnn.Eltwise();
        $.type = reader.int8_(position, 4, 0);
        $.coeff = reader.typedArray(position, 6, Float32Array);
        return $;
    }
};

$root.wnn.ReductionType = {
    SUM: 0,
    ASUM: 1,
    SUMSQ: 2,
    MEAN: 3,
    MAXIMUM: 4,
    MINIMUM: 5,
    PROD: 6,
    ANY: 7,
    ALL: 8
};

$root.wnn.Reduction = class Reduction {

    static decode(reader, position) {
        const $ = new $root.wnn.Reduction();
        $.operation = reader.int8_(position, 4, 0);
        $.dim = reader.typedArray(position, 6, Int32Array);
        $.coeff = reader.float32_(position, 8, 0);
        $.keep_dims = reader.bool_(position, 10, false);
        $.dtype = reader.int32_(position, 12, 1);
        return $;
    }
};

$root.wnn.Squeeze = class Squeeze {

    static decode(reader, position) {
        const $ = new $root.wnn.Squeeze();
        $.squeeze_dims = reader.typedArray(position, 4, Int32Array);
        return $;
    }
};

$root.wnn.Gather = class Gather {

    static decode(reader, position) {
        const $ = new $root.wnn.Gather();
        $.indices_dtype = reader.int32_(position, 4, 0);
        $.dtype = reader.int32_(position, 6, 0);
        $.validateindices = reader.bool_(position, 8, false);
        $.axis = reader.int32_(position, 10, 0);
        return $;
    }
};

$root.wnn.ExpandDims = class ExpandDims {

    static decode(reader, position) {
        const $ = new $root.wnn.ExpandDims();
        $.dtype = reader.int32_(position, 4, 0);
        $.dim_dtype = reader.int32_(position, 6, 0);
        $.axis = reader.int32_(position, 8, 0);
        return $;
    }
};

$root.wnn.Flatten = class Flatten {

    static decode(reader, position) {
        const $ = new $root.wnn.Flatten();
        $.start_dim = reader.int32_(position, 4, 0);
        $.end_dim = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.wnn.ImageFormatType = {
    RGBA: 0,
    RGB: 1,
    BGR: 2,
    GRAY: 3,
    YUV: 4,
    HSV: 5
};

$root.wnn.FilterType = {
    NEAREST: 0,
    BILINEAR: 1,
    BICUBIC: 2
};

$root.wnn.Normalize = class Normalize {

    static decode(reader, position) {
        const $ = new $root.wnn.Normalize();
        $.means = reader.typedArray(position, 4, Float32Array);
        $.stds = reader.typedArray(position, 6, Float32Array);
        $.denormalize = reader.bool_(position, 8, false);
        $.epsilon = reader.float32_(position, 10, 0.00001);
        return $;
    }
};

$root.wnn.FilmLPN = class FilmLPN {

    static decode(reader, position) {
        const $ = new $root.wnn.FilmLPN();
        $.in_features = reader.int32_(position, 4, 0);
        $.out_features = reader.int32_(position, 6, 0);
        $.input_n = reader.int32_(position, 8, 1);
        $.weight_size = reader.int32_(position, 10, 0);
        $.weights = reader.typedArray(position, 12, Float32Array);
        $.bias = reader.typedArray(position, 14, Float32Array);
        $.bias_size = reader.int32_(position, 16, 0);
        $.quant_param = reader.table(position, 18, $root.wnn.Quant.decode);
        return $;
    }
};

$root.wnn.Cubic = class Cubic {

    static decode(reader, position) {
        const $ = new $root.wnn.Cubic();
        $.in_features = reader.int32_(position, 4, 0);
        $.out_features = reader.int32_(position, 6, 0);
        $.input_n = reader.int32_(position, 8, 1);
        $.merge_matmul = reader.bool_(position, 10, true);
        $.weight_size = reader.int32_(position, 12, 0);
        $.weight = reader.typedArray(position, 14, Float32Array);
        $.bias = reader.typedArray(position, 16, Float32Array);
        $.bias_size = reader.int32_(position, 18, 0);
        $.quant_param = reader.table(position, 20, $root.wnn.Quant.decode);
        return $;
    }
};

$root.wnn.ModelSource = {
    TORCH: 0,
    TENSORFLOW: 1,
    ONNX: 2,
    TFLITE: 3
};

$root.wnn.OpParameter = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.wnn.Conv2DCommon.decode(reader, position);
            case 2: return $root.wnn.Conv2D.decode(reader, position);
            case 3: return $root.wnn.Pool.decode(reader, position);
            case 4: return $root.wnn.AdaptiveAvgPool2D.decode(reader, position);
            case 5: return $root.wnn.LayerNorm.decode(reader, position);
            case 6: return $root.wnn.BatchNorm.decode(reader, position);
            case 7: return $root.wnn.Relu.decode(reader, position);
            case 8: return $root.wnn.Relu6.decode(reader, position);
            case 9: return $root.wnn.PRelu.decode(reader, position);
            case 10: return $root.wnn.ELU.decode(reader, position);
            case 11: return $root.wnn.LRN.decode(reader, position);
            case 12: return $root.wnn.Softmax.decode(reader, position);
            case 13: return $root.wnn.Input.decode(reader, position);
            case 14: return $root.wnn.Extra.decode(reader, position);
            case 15: return $root.wnn.FC.decode(reader, position);
            case 16: return $root.wnn.ArgMax.decode(reader, position);
            case 17: return $root.wnn.Binary.decode(reader, position);
            case 18: return $root.wnn.Unary.decode(reader, position);
            case 19: return $root.wnn.Eltwise.decode(reader, position);
            case 20: return $root.wnn.Reduction.decode(reader, position);
            case 21: return $root.wnn.Squeeze.decode(reader, position);
            case 22: return $root.wnn.Gather.decode(reader, position);
            case 23: return $root.wnn.ExpandDims.decode(reader, position);
            case 24: return $root.wnn.Normalize.decode(reader, position);
            case 25: return $root.wnn.Flatten.decode(reader, position);
            case 26: return $root.wnn.Blob.decode(reader, position);
            case 27: return $root.wnn.FilmLPN.decode(reader, position);
            case 28: return $root.wnn.Cubic.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Conv2DCommon': return $root.wnn.Conv2DCommon.decodeText(reader, json);
            case 'Conv2D': return $root.wnn.Conv2D.decodeText(reader, json);
            case 'Pool': return $root.wnn.Pool.decodeText(reader, json);
            case 'AdaptiveAvgPool2D': return $root.wnn.AdaptiveAvgPool2D.decodeText(reader, json);
            case 'LayerNorm': return $root.wnn.LayerNorm.decodeText(reader, json);
            case 'BatchNorm': return $root.wnn.BatchNorm.decodeText(reader, json);
            case 'Relu': return $root.wnn.Relu.decodeText(reader, json);
            case 'Relu6': return $root.wnn.Relu6.decodeText(reader, json);
            case 'PRelu': return $root.wnn.PRelu.decodeText(reader, json);
            case 'ELU': return $root.wnn.ELU.decodeText(reader, json);
            case 'LRN': return $root.wnn.LRN.decodeText(reader, json);
            case 'Softmax': return $root.wnn.Softmax.decodeText(reader, json);
            case 'Input': return $root.wnn.Input.decodeText(reader, json);
            case 'Extra': return $root.wnn.Extra.decodeText(reader, json);
            case 'FC': return $root.wnn.FC.decodeText(reader, json);
            case 'ArgMax': return $root.wnn.ArgMax.decodeText(reader, json);
            case 'Binary': return $root.wnn.Binary.decodeText(reader, json);
            case 'Unary': return $root.wnn.Unary.decodeText(reader, json);
            case 'Eltwise': return $root.wnn.Eltwise.decodeText(reader, json);
            case 'Reduction': return $root.wnn.Reduction.decodeText(reader, json);
            case 'Squeeze': return $root.wnn.Squeeze.decodeText(reader, json);
            case 'Gather': return $root.wnn.Gather.decodeText(reader, json);
            case 'ExpandDims': return $root.wnn.ExpandDims.decodeText(reader, json);
            case 'Normalize': return $root.wnn.Normalize.decodeText(reader, json);
            case 'Flatten': return $root.wnn.Flatten.decodeText(reader, json);
            case 'Blob': return $root.wnn.Blob.decodeText(reader, json);
            case 'FilmLPN': return $root.wnn.FilmLPN.decodeText(reader, json);
            case 'Cubic': return $root.wnn.Cubic.decodeText(reader, json);
            default: return undefined;
        }
    }
};

$root.wnn.Dims = class Dims {

    static decode(reader, position) {
        const $ = new $root.wnn.Dims();
        $.shape = reader.typedArray(position, 4, Int32Array);
        $.total_size = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.wnn.Op = class Op {

    static decode(reader, position) {
        const $ = new $root.wnn.Op();
        $.input_indexes = reader.typedArray(position, 4, Int32Array);
        $.output_indexes = reader.typedArray(position, 6, Int32Array);
        $.input_names = reader.strings_(position, 8);
        $.output_names = reader.strings_(position, 10);
        $.input_shapes = reader.tableArray(position, 12, $root.wnn.Dims.decode);
        $.output_shapes = reader.tableArray(position, 14, $root.wnn.Dims.decode);
        $.is_static_shape = reader.bool_(position, 16, false);
        $.param = reader.union(position, 18, $root.wnn.OpParameter.decode);
        $.name = reader.string_(position, 22, null);
        $.type = reader.int32_(position, 24, 0);
        return $;
    }
};

$root.wnn.Extra = class Extra {

    static decode(reader, position) {
        const $ = new $root.wnn.Extra();
        $.type = reader.string_(position, 4, null);
        $.engine = reader.string_(position, 6, null);
        $.info = reader.typedArray(position, 8, Int8Array);
        $.attr = reader.tableArray(position, 10, $root.wnn.Attribute.decode);
        return $;
    }
};

$root.wnn.SubGraph = class SubGraph {

    static decode(reader, position) {
        const $ = new $root.wnn.SubGraph();
        $.name = reader.string_(position, 4, null);
        $.inputs = reader.typedArray(position, 6, Int32Array);
        $.outputs = reader.typedArray(position, 8, Int32Array);
        $.tensors = reader.strings_(position, 10);
        $.nodes = reader.tableArray(position, 12, $root.wnn.Op.decode);
        return $;
    }
};

$root.wnn.TensorQuantInfo = class TensorQuantInfo {

    static decode(reader, position) {
        const $ = new $root.wnn.TensorQuantInfo();
        $.scale = reader.float32_(position, 4, 0);
        $.zero = reader.float32_(position, 6, 0);
        $.min = reader.float32_(position, 8, -128);
        $.max = reader.float32_(position, 10, 127);
        $.type = reader.int32_(position, 12, 0);
        return $;
    }
};

$root.wnn.TensorDescribe = class TensorDescribe {

    static decode(reader, position) {
        const $ = new $root.wnn.TensorDescribe();
        $.blob = reader.table(position, 4, $root.wnn.Blob.decode);
        $.index = reader.int32_(position, 6, 0);
        $.name = reader.string_(position, 8, null);
        $.quant_info = reader.table(position, 10, $root.wnn.TensorQuantInfo.decode);
        return $;
    }
};

$root.wnn.Graph = class Graph {

    static create(reader) {
        return $root.wnn.Graph.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.wnn.Graph();
        $.desc = reader.string_(position, 4, null);
        $.usage = reader.string_(position, 6, null);
        $.vendor = reader.string_(position, 8, null);
        $.version = reader.string_(position, 10, null);
        $.extra_tensor_describe = reader.tableArray(position, 12, $root.wnn.TensorDescribe.decode);
        $.oplists = reader.tableArray(position, 14, $root.wnn.Op.decode);
        $.output_names = reader.strings_(position, 16);
        $.input_names = reader.strings_(position, 18);
        $.model_source = reader.int8_(position, 20, 0);
        $.tensor_names = reader.strings_(position, 22);
        $.tensor_number = reader.int32_(position, 24, 0);
        $.subgraph = reader.table(position, 26, $root.wnn.SubGraph.decode);
        $.uuid = reader.string_(position, 28, null);
        $.password = reader.string_(position, 30, null);
        return $;
    }
};
