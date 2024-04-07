
export const mgb = {};

mgb.serialization = mgb.serialization || {};

mgb.serialization.fbs = mgb.serialization.fbs || {};

mgb.serialization.fbs.DTypeEnum = {
    Float32: 0,
    Uint8: 1,
    Int8: 2,
    Int16: 3,
    Int32: 4,
    IntB1: 5,
    IntB2: 6,
    IntB4: 7,
    Byte: 8,
    Float16: 9,
    UintB4: 10,
    Quantized8Asymm: 11,
    QuantizedS32: 12,
    QuantizedS8: 13,
    Quantized4Asymm: 14,
    QuantizedS4: 15,
    QuantizedS16: 16,
    BFloat16: 17,
    Bool: 18,
    Uint16: 19,
    QuantizedS1: 20
};

mgb.serialization.fbs.LinearQuantizationParam = class LinearQuantizationParam {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.LinearQuantizationParam();
        $.scale = reader.float32_(position, 4, 0);
        $.zero_point = reader.uint8_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.DTypeParam = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return mgb.serialization.fbs.LinearQuantizationParam.decode(reader, position);
            default: return undefined;
        }
    }
};

mgb.serialization.fbs.DType = class DType {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.DType();
        $.type = reader.int8_(position, 4, 0);
        $.param = reader.union(position, 6, mgb.serialization.fbs.DTypeParam);
        return $;
    }
};

mgb.serialization = mgb.serialization || {};

mgb.serialization.fbs = mgb.serialization.fbs || {};

mgb.serialization.fbs.param = mgb.serialization.fbs.param || {};

mgb.serialization.fbs.param.ArgsortOrder = {
    ASCENDING: 0,
    DESCENDING: 1
};

mgb.serialization.fbs.param.BNFwdMode = {
    TRAINING: 0,
    INFERENCE: 1
};

mgb.serialization.fbs.param.BNParamDim = {
    DIM_11HW: 0,
    DIM_1CHW: 1,
    DIM_1C11: 2,
    DIM_111C: 3
};

mgb.serialization.fbs.param.CondTakeMode = {
    EQ: 0,
    NEQ: 1,
    LT: 2,
    LEQ: 3,
    GT: 4,
    GEQ: 5
};

mgb.serialization.fbs.param.Conv3DBiasNonlineMode = {
    IDENTITY: 0,
    RELU: 1,
    SIGMOID: 2
};

mgb.serialization.fbs.param.ConvBiasV0NonlineMode = {
    IDENTITY: 0,
    RELU: 1,
    SIGMOID: 2,
    H_SWISH: 3
};

mgb.serialization.fbs.param.ConvPoolingMethod = {
    WITH_TEXTURE_OBJ: 0,
    WITH_SHARED_MEM: 1
};

mgb.serialization.fbs.param.ConvPoolingNonlineMode = {
    IDENTITY: 0,
    RELU: 1,
    SIGMOID: 2
};

mgb.serialization.fbs.param.ConvPoolingPoolMode = {
    AVERAGE: 0,
    MAX_: 1
};

mgb.serialization.fbs.param.ConvolutionFormat = {
    NCHW: 0,
    NHWC: 1,
    NHWCD4: 2,
    NCHW4: 3,
    NCHW8: 4,
    NCHW32: 5,
    NCHW88: 6,
    NCHW44: 7,
    NCHW44_DOT: 8,
    NCHW4_NCHW32: 9,
    NCHW32_NCHW4: 10,
    NCHW4_NCHW: 11,
    NHWC_NCHW: 12,
    NHWC_NCHW4_IC_SMALL: 13,
    NCHW_NCHW4_IC_SMALL: 14,
    CHWN4: 15,
    NCHW64: 16,
    NCHW4_NHWC: 17
};

mgb.serialization.fbs.param.Convolution3DDataType = {
    FLOAT: 0,
    FLOAT_IO16xC32: 1
};

mgb.serialization.fbs.param.Convolution3DFormat = {
    NCDHW: 0,
    NDHWC: 1
};

mgb.serialization.fbs.param.Convolution3DMode = {
    CROSS_CORRELATION: 0,
    CONVOLUTION: 1
};

mgb.serialization.fbs.param.Convolution3DSparse = {
    DENSE: 0,
    GROUP: 1
};

mgb.serialization.fbs.param.ConvolutionV0DataType = {
    FLOAT: 0,
    INT8x8x16: 1,
    INT8x8x32: 2,
    FLOAT_IO16xC32: 3,
    QUINT8x8x32: 4,
    INT8x8xX: 5,
    QUINT4x4x32: 6
};

mgb.serialization.fbs.param.ConvolutionV0Format = {
    NCHW: 0,
    NHWC: 1,
    NHWCD4: 2,
    NCHW4: 3,
    NCHW8: 4,
    NCHW32: 5,
    NCHW88: 6,
    NCHW44: 7,
    NCHW44_DOT: 8,
    NCHW_WINOGRAD: 9,
    NCHW88_WINOGRAD: 10,
    NCHW44_WINOGRAD: 11,
    NCHW4_NCHW32: 12,
    NCHW32_NCHW4: 13,
    NCHW4_NCHW: 14,
    NHWC_NCHW: 15,
    NHWC_NCHW4_IC_SMALL: 16,
    NCHW_NCHW4_IC_SMALL: 17,
    CHWN4: 18,
    NCHW4_NHWC: 19
};

mgb.serialization.fbs.param.ConvolutionV0Mode = {
    CROSS_CORRELATION: 0,
    CONVOLUTION: 1
};

mgb.serialization.fbs.param.ConvolutionV0Sparse = {
    DENSE: 0,
    GROUP: 1
};

mgb.serialization.fbs.param.ConvolutionV1ComputeMode = {
    DEFAULT: 0,
    FLOAT32: 1
};

mgb.serialization.fbs.param.CvtColorMode = {
    RGB2GRAY: 0,
    RGB2YUV: 1,
    YUV2RGB: 2,
    GRAY2RGB: 3,
    RGBA2RGB: 4,
    RGBA2BGR: 5,
    RGBA2GRAY: 6,
    RGB2BGR: 7,
    BGR2GRAY: 8,
    BGR2RGB: 9,
    YUV2GRAY_NV21: 10,
    YUV2RGB_NV21: 11,
    YUV2BGR_NV21: 12,
    YUV2GRAY_NV12: 13,
    YUV2RGB_NV12: 14,
    YUV2BGR_NV12: 15,
    YUV2GRAY_YV12: 16,
    YUV2RGB_YV12: 17,
    YUV2BGR_YV12: 18,
    YUV2GRAY_YU12: 19,
    YUV2RGB_YU12: 20,
    YUV2BGR_YU12: 21,
    YCrCb2RGB: 22,
    YCrCb2BGR: 23,
    BT601_YUV2RGB_NV21: 24,
    BT601_YUV2BGR_NV21: 25,
    BT601_YUV2RGB_NV12: 26,
    BT601_YUV2BGR_NV12: 27,
    BT601_YUV2RGB_YV12: 28,
    BT601_YUV2BGR_YV12: 29,
    BT601_YUV2RGB_YU12: 30,
    BT601_YUV2BGR_YU12: 31
};

mgb.serialization.fbs.param.DctChannelSelectV0FastImpl = {
    NONE: 0,
    FIX_32_MASK: 1
};

mgb.serialization.fbs.param.ElemwiseMode = {
    RELU: 0,
    ABS: 1,
    ACOS: 2,
    ASIN: 3,
    CEIL: 4,
    COS: 5,
    EXP: 6,
    EXPM1: 7,
    FLOOR: 8,
    LOG: 9,
    LOG1P: 10,
    NEGATE: 11,
    SIGMOID: 12,
    SIN: 13,
    TANH: 14,
    ABS_GRAD: 15,
    ADD: 16,
    FLOOR_DIV: 17,
    MAX_: 18,
    MIN_: 19,
    MOD: 20,
    MUL: 21,
    POW: 22,
    SIGMOID_GRAD: 23,
    SUB: 24,
    SWITCH_GT0: 25,
    TANH_GRAD: 26,
    TRUE_DIV: 27,
    LOG_SUM_EXP: 28,
    LT: 29,
    LEQ: 30,
    EQ: 31,
    SHL: 32,
    SHR: 33,
    COND_LEQ_MOV: 34,
    FUSE_MUL_ADD3: 35,
    FUSE_MUL_ADD4: 36,
    FUSE_ADD_RELU: 37,
    FUSE_ADD_SIGMOID: 38,
    FUSE_ADD_TANH: 39,
    FAST_TANH: 40,
    FAST_TANH_GRAD: 41,
    ROUND: 42,
    RMULH: 43,
    ATAN2: 44,
    ERF: 45,
    ERFINV: 46,
    ERFC: 47,
    ERFCINV: 48,
    H_SWISH: 49,
    H_SWISH_GRAD: 50,
    FUSE_ADD_H_SWISH: 51,
    NOT: 52,
    AND: 53,
    OR: 54,
    XOR: 55,
    SILU: 56,
    SILU_GRAD: 57,
    GELU: 58,
    GELU_GRAD: 59
};

mgb.serialization.fbs.param.ElemwiseMultiTypeMode = {
    FUSE_MUL_ADD3_INT16x32x32x32: 0,
    FUSE_MUL_ADD3_IXxF32xF32xI8: 1,
    ROUND_SHR_SATURATE_IXxI8xI8: 2,
    FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8: 3,
    FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8: 4,
    ROUND_SHR_SATURATE_IXxI8xI16: 5,
    QADD: 6,
    QFUSE_ADD_RELU: 7,
    QMUL: 8,
    QMIN: 9,
    QMAX: 10,
    QSUB: 11,
    QTRUE_DIV: 12,
    QFUSE_ADD_SIGMOID: 13,
    QFUSE_ADD_TANH: 14,
    QRELU: 15,
    QABS: 16,
    QSIGMOID: 17,
    QEXP: 18,
    QTANH: 19,
    QFUSE_MUL_ADD3: 20,
    QFAST_TANH: 21,
    QNEGATE: 22,
    QACOS: 23,
    QASIN: 24,
    QCEIL: 25,
    QCOS: 26,
    QEXPM1: 27,
    QFLOOR: 28,
    QLOG: 29,
    QLOG1P: 30,
    QSIN: 31,
    QROUND: 32,
    QERF: 33,
    QERFINV: 34,
    QERFC: 35,
    QERFCINV: 36,
    QABS_GRAD: 37,
    QFLOOR_DIV: 38,
    QMOD: 39,
    QSIGMOID_GRAD: 40,
    QSWITCH_GT0: 41,
    QTANH_GRAD: 42,
    QLT: 43,
    QLEQ: 44,
    QEQ: 45,
    QPOW: 46,
    QLOG_SUM_EXP: 47,
    QFAST_TANH_GRAD: 48,
    QATAN2: 49,
    QCOND_LEQ_MOV: 50,
    QH_SWISH: 51,
    QFUSE_ADD_H_SWISH: 52,
    QH_SWISH_GRAD: 53,
    FUSE_MUL_ADD3_INT16xF32xF32xF32: 54,
    MUL_INT16xF32xF32: 55,
    FUSE_MUL_ADD3_UINT8xF32xF32xF32: 56
};

mgb.serialization.fbs.param.MatrixMulFormat = {
    DEFAULT: 0,
    MK4: 1,
    MK8: 2,
    MK4_DOT: 3
};

mgb.serialization.fbs.param.MatrixMulV0DataType = {
    FLOAT: 0,
    INT8x8x16: 1,
    INT8x8x32: 2,
    FLOAT_IO16xC32: 3,
    QUINT8x8x32: 4,
    QUINT4x4x32: 5
};

mgb.serialization.fbs.param.MatrixMulV1ComputeMode = {
    DEFAULT: 0,
    FLOAT32: 1
};

mgb.serialization.fbs.param.PaddingPaddingMode = {
    REPLICATE: 0,
    REFLECT: 1,
    CONSTANT: 2
};

mgb.serialization.fbs.param.PoolingV0Mode = {
    MAX_: 0,
    AVERAGE: 1,
    AVERAGE_COUNT_EXCLUDE_PADDING: 2
};

mgb.serialization.fbs.param.RNNCellNonlineMode = {
    IDENTITY: 0,
    RELU: 1,
    TANH: 2
};

mgb.serialization.fbs.param.ROIAlignV0Mode = {
    MAX_: 0,
    AVERAGE: 1
};

mgb.serialization.fbs.param.ROIPoolingMode = {
    MAX_: 0,
    AVERAGE: 1
};

mgb.serialization.fbs.param.ReduceDataType = {
    DEFAULT: 0,
    FLOAT_IO16xC32: 1,
    FLOAT_O32xC32: 2,
    FLOAT_O16xC32: 3,
    QUINT_I8xO32: 4,
    QINT_I8xO32: 5
};

mgb.serialization.fbs.param.ReduceMode = {
    SUM: 0,
    SUM_SQR: 1,
    PRODUCT: 2,
    MIN_: 3,
    MAX_: 4,
    MEAN: 5
};

mgb.serialization.fbs.param.ReduceV0Mode = {
    SUM: 0,
    SUM_SQR: 1,
    PRODUCT: 2,
    MIN_: 3,
    MAX_: 4
};

mgb.serialization.fbs.param.ReduceV1DataType = {
    DEFAULT: 0,
    FLOAT_IO16xC32: 1,
    FLOAT_O32xC32: 2,
    FLOAT_O16xC32: 3,
    QUINT_I8xO32: 4,
    QINT_I8xO32: 5
};

mgb.serialization.fbs.param.ReduceV1Mode = {
    SUM: 0,
    SUM_SQR: 1,
    PRODUCT: 2,
    MIN_: 3,
    MAX_: 4,
    MEAN: 5
};

mgb.serialization.fbs.param.RelayoutFormatV0Mode = {
    NHWC_NHWCD4: 0,
    NHWCD4_NHWC: 1,
    NHWC_NHWCD4I: 2,
    NCHW_NHWCD4: 3,
    NCHW_NHWCD4I: 4,
    NHWCD4I_NCHW: 5,
    NHWCD4_NCHW: 6,
    INTER_WEIGHT_DENSE: 7,
    INTER_WEIGHT_DENSEI: 8,
    INTER_WEIGHT_GROUP: 9,
    INTER_WEIGHT_GROUPI: 10,
    INTER_WEIGHT_CHAN: 11,
    INTER_WEIGHT_CHANI: 12,
    INTER_WEIGHT_DENSEI_DOT: 13,
    INTER_WEIGHT_GROUPI_DOT: 14,
    NCHW4_CHWN4: 15,
    CHWN4_NCHW4: 16,
    NCHW_NCHW88_CONV_DENSE_WEIGHT: 17,
    NCHW_NCHW88_CONV_CHAN_WEIGHT: 18,
    NCHW_NCHW88_CONV_GROUP_WEIGHT: 19,
    NCHW_NCHW88: 20,
    NCHW88_NCHW: 21,
    NCHW_NCHW4_IC_SMALL: 22,
    NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT: 23,
    NCHW_NCHW4: 24,
    NCHW4_NCHW: 25,
    NCHW_NCHW4_WEIGHT: 26,
    NCHW_NCHW64: 27,
    NCHW64_NCHW: 28,
    NCHW_NHWC: 29,
    NHWC_NCHW: 30,
    NHWCD4I_NHWC: 31
};

mgb.serialization.fbs.param.SeparableConvBorderMode = {
    BORDER_REPLICATE: 0,
    BORDER_REFLECT: 1,
    BORDER_REFLECT_101: 2,
    BORDER_WRAP: 3,
    BORDER_CONSTANT: 4,
    BORDER_TRANSPARENT: 5,
    BORDER_ISOLATED: 6
};

mgb.serialization.fbs.param.SeparableConv3DBorderMode = {
    BORDER_REPLICATE: 0,
    BORDER_REFLECT: 1,
    BORDER_REFLECT_101: 2,
    BORDER_WRAP: 3,
    BORDER_CONSTANT: 4,
    BORDER_TRANSPARENT: 5,
    BORDER_ISOLATED: 6
};

mgb.serialization.fbs.param.SpatialTfGridGeneratorMode = {
    AFFINE: 0
};

mgb.serialization.fbs.param.SpatialTfSamplerMode = {
    BILINEAR: 0
};

mgb.serialization.fbs.param.TopKMode = {
    KTH_ONLY: 0,
    VALUE_IDX_NOSORT: 1,
    VALUE_IDX_SORTED: 2
};

mgb.serialization.fbs.param.WarpPerspectiveV1BorderMode = {
    REPLICATE: 0,
    REFLECT: 1,
    REFLECT_101: 2,
    WRAP: 3,
    CONSTANT: 4,
    TRANSPARENT: 5,
    ISOLATED: 6
};

mgb.serialization.fbs.param.WarpPerspectiveV1InterpolationMode = {
    NEAREST: 0,
    LINEAR: 1,
    AREA: 2,
    CUBIC: 3,
    LANCZOS4: 4
};

mgb.serialization.fbs.param.Empty = class Empty {

    static decode(/* reader, position */) {
        const $ = new mgb.serialization.fbs.param.Empty();
        return $;
    }
};

mgb.serialization.fbs.param.Axis = class Axis {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Axis();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ConvolutionV0 = class ConvolutionV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvolutionV0();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 1);
        $.stride_w = reader.uint32_(position, 12, 1);
        $.dilate_h = reader.uint32_(position, 14, 1);
        $.dilate_w = reader.uint32_(position, 16, 1);
        $.data_type = reader.uint32_(position, 18, 0);
        $.sparse = reader.uint32_(position, 20, 0);
        $.format = reader.uint32_(position, 22, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ConvolutionV1 = class ConvolutionV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvolutionV1();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 1);
        $.stride_w = reader.uint32_(position, 12, 1);
        $.dilate_h = reader.uint32_(position, 14, 1);
        $.dilate_w = reader.uint32_(position, 16, 1);
        $.sparse = reader.uint32_(position, 18, 0);
        $.format = reader.uint32_(position, 20, 0);
        $.compute_mode = reader.uint32_(position, 22, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Convolution = class Convolution {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Convolution();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 1);
        $.stride_w = reader.uint32_(position, 12, 1);
        $.dilate_h = reader.uint32_(position, 14, 1);
        $.dilate_w = reader.uint32_(position, 16, 1);
        $.sparse = reader.uint32_(position, 18, 0);
        $.format = reader.uint32_(position, 20, 0);
        $.compute_mode = reader.uint32_(position, 22, 0);
        return $;
    }
};

mgb.serialization.fbs.param.MaskPropagate = class MaskPropagate {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MaskPropagate();
        $.pad_h = reader.uint32_(position, 4, 0);
        $.pad_w = reader.uint32_(position, 6, 0);
        $.stride_h = reader.uint32_(position, 8, 1);
        $.stride_w = reader.uint32_(position, 10, 1);
        $.kernel_h = reader.uint32_(position, 12, 1);
        $.kernel_w = reader.uint32_(position, 14, 1);
        $.dilate_h = reader.uint32_(position, 16, 1);
        $.dilate_w = reader.uint32_(position, 18, 1);
        return $;
    }
};

mgb.serialization.fbs.param.ConvPooling = class ConvPooling {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvPooling();
        $.method = reader.uint32_(position, 4, 0);
        $.convMode = reader.uint32_(position, 6, 0);
        $.poolMode = reader.uint32_(position, 8, 0);
        $.nonlineMode = reader.uint32_(position, 10, 0);
        $.pool_shape_h = reader.uint32_(position, 12, 1);
        $.pool_shape_w = reader.uint32_(position, 14, 1);
        $.pool_stride_h = reader.uint32_(position, 16, 1);
        $.pool_stride_w = reader.uint32_(position, 18, 1);
        $.pool_pad_h = reader.uint32_(position, 20, 0);
        $.pool_pad_w = reader.uint32_(position, 22, 0);
        $.conv_stride_h = reader.uint32_(position, 24, 1);
        $.conv_stride_w = reader.uint32_(position, 26, 1);
        $.conv_pad_h = reader.uint32_(position, 28, 0);
        $.conv_pad_w = reader.uint32_(position, 30, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ConvBiasV0 = class ConvBiasV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvBiasV0();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.pad_h = reader.uint32_(position, 8, 0);
        $.pad_w = reader.uint32_(position, 10, 0);
        $.stride_h = reader.uint32_(position, 12, 1);
        $.stride_w = reader.uint32_(position, 14, 1);
        return $;
    }
};

mgb.serialization.fbs.param.ConvBiasV1 = class ConvBiasV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvBiasV1();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.data_type = reader.uint32_(position, 8, 0);
        $.sparse = reader.uint32_(position, 10, 0);
        $.format = reader.uint32_(position, 12, 0);
        $.pad_h = reader.uint32_(position, 14, 0);
        $.pad_w = reader.uint32_(position, 16, 0);
        $.stride_h = reader.uint32_(position, 18, 1);
        $.stride_w = reader.uint32_(position, 20, 1);
        $.dilate_h = reader.uint32_(position, 22, 1);
        $.dilate_w = reader.uint32_(position, 24, 1);
        return $;
    }
};

mgb.serialization.fbs.param.ConvBiasV2 = class ConvBiasV2 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvBiasV2();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.sparse = reader.uint32_(position, 8, 0);
        $.format = reader.uint32_(position, 10, 0);
        $.pad_h = reader.uint32_(position, 12, 0);
        $.pad_w = reader.uint32_(position, 14, 0);
        $.stride_h = reader.uint32_(position, 16, 1);
        $.stride_w = reader.uint32_(position, 18, 1);
        $.dilate_h = reader.uint32_(position, 20, 1);
        $.dilate_w = reader.uint32_(position, 22, 1);
        $.compute_mode = reader.uint32_(position, 24, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ConvBiasV3 = class ConvBiasV3 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvBiasV3();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.sparse = reader.uint32_(position, 8, 0);
        $.format = reader.uint32_(position, 10, 0);
        $.pad_h = reader.uint32_(position, 12, 0);
        $.pad_w = reader.uint32_(position, 14, 0);
        $.stride_h = reader.uint32_(position, 16, 1);
        $.stride_w = reader.uint32_(position, 18, 1);
        $.dilate_h = reader.uint32_(position, 20, 1);
        $.dilate_w = reader.uint32_(position, 22, 1);
        $.output_block_size = reader.uint32_(position, 24, 0);
        $.compute_mode = reader.uint32_(position, 26, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ConvBias = class ConvBias {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ConvBias();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.sparse = reader.uint32_(position, 8, 0);
        $.format = reader.uint32_(position, 10, 0);
        $.pad_h = reader.uint32_(position, 12, 0);
        $.pad_w = reader.uint32_(position, 14, 0);
        $.stride_h = reader.uint32_(position, 16, 1);
        $.stride_w = reader.uint32_(position, 18, 1);
        $.dilate_h = reader.uint32_(position, 20, 1);
        $.dilate_w = reader.uint32_(position, 22, 1);
        $.compute_mode = reader.uint32_(position, 24, 0);
        return $;
    }
};

mgb.serialization.fbs.param.SeparableConv = class SeparableConv {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SeparableConv();
        $.mode = reader.uint32_(position, 4, 0);
        $.borderMode = reader.uint32_(position, 6, 0);
        $.is_symm_kernel = reader.bool_(position, 8, true);
        $.pad_h = reader.uint32_(position, 10, 0);
        $.pad_w = reader.uint32_(position, 12, 0);
        $.stride_h = reader.uint32_(position, 14, 1);
        $.stride_w = reader.uint32_(position, 16, 1);
        $.ksize_h = reader.uint32_(position, 18, 3);
        $.ksize_w = reader.uint32_(position, 20, 3);
        $.anchor_h = reader.uint32_(position, 22, 1);
        $.anchor_w = reader.uint32_(position, 24, 1);
        return $;
    }
};

mgb.serialization.fbs.param.Images2Neibs = class Images2Neibs {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Images2Neibs();
        $.pad_h = reader.uint32_(position, 4, 0);
        $.pad_w = reader.uint32_(position, 6, 0);
        $.stride_h = reader.uint32_(position, 8, 1);
        $.stride_w = reader.uint32_(position, 10, 1);
        $.dilate_h = reader.uint32_(position, 12, 1);
        $.dilate_w = reader.uint32_(position, 14, 1);
        $.window_h = reader.uint32_(position, 16, 3);
        $.window_w = reader.uint32_(position, 18, 3);
        return $;
    }
};

mgb.serialization.fbs.param.SlidingWindowTranspose = class SlidingWindowTranspose {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SlidingWindowTranspose();
        $.out_h = reader.uint32_(position, 4, 0);
        $.out_w = reader.uint32_(position, 6, 0);
        $.pad_h = reader.uint32_(position, 8, 0);
        $.pad_w = reader.uint32_(position, 10, 0);
        $.stride_h = reader.uint32_(position, 12, 1);
        $.stride_w = reader.uint32_(position, 14, 1);
        $.dilate_h = reader.uint32_(position, 16, 1);
        $.dilate_w = reader.uint32_(position, 18, 1);
        $.window_h = reader.uint32_(position, 20, 3);
        $.window_w = reader.uint32_(position, 22, 3);
        return $;
    }
};

mgb.serialization.fbs.param.PoolingV0 = class PoolingV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.PoolingV0();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 2);
        $.stride_w = reader.uint32_(position, 12, 2);
        $.window_h = reader.uint32_(position, 14, 2);
        $.window_w = reader.uint32_(position, 16, 2);
        $.format = reader.uint32_(position, 18, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Pooling = class Pooling {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Pooling();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 2);
        $.stride_w = reader.uint32_(position, 12, 2);
        $.window_h = reader.uint32_(position, 14, 2);
        $.window_w = reader.uint32_(position, 16, 2);
        $.format = reader.uint32_(position, 18, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Softmax = class Softmax {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Softmax();
        $.axis = reader.int32_(position, 4, -1);
        return $;
    }
};

mgb.serialization.fbs.param.AdaptivePoolingV0 = class AdaptivePoolingV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.AdaptivePoolingV0();
        $.mode = reader.uint32_(position, 4, 0);
        $.format = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.AdaptivePooling = class AdaptivePooling {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.AdaptivePooling();
        $.mode = reader.uint32_(position, 4, 0);
        $.format = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.LRN = class LRN {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LRN();
        $.n = reader.uint32_(position, 4, 5);
        $.k = reader.float32_(position, 6, 2);
        $.alpha = reader.float32_(position, 8, 0.0001);
        $.beta = reader.float32_(position, 10, 0.75);
        return $;
    }
};

mgb.serialization.fbs.param.BN = class BN {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.BN();
        $.param_dim = reader.uint32_(position, 4, 0);
        $.fwd_mode = reader.uint32_(position, 6, 0);
        $.epsilon = reader.float64_(position, 8, 0.0001);
        $.avg_factor = reader.float64_(position, 10, 1);
        $.scale = reader.float32_(position, 12, 1);
        $.bias = reader.float32_(position, 14, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ROIPooling = class ROIPooling {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ROIPooling();
        $.mode = reader.uint32_(position, 4, 0);
        $.scale = reader.float32_(position, 6, 1);
        return $;
    }
};

mgb.serialization.fbs.param.WarpPerspectiveV1 = class WarpPerspectiveV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.WarpPerspectiveV1();
        $.imode = reader.uint32_(position, 4, 1);
        $.bmode = reader.uint32_(position, 6, 0);
        $.format = reader.uint32_(position, 8, 0);
        $.border_val = reader.float32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.WarpPerspective = class WarpPerspective {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.WarpPerspective();
        $.imode = reader.uint32_(position, 4, 1);
        $.bmode = reader.uint32_(position, 6, 0);
        $.format = reader.uint32_(position, 8, 0);
        $.border_val = reader.float32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.SpatialTfGridGenerator = class SpatialTfGridGenerator {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SpatialTfGridGenerator();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.SpatialTfSampler = class SpatialTfSampler {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SpatialTfSampler();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.AddUpdate = class AddUpdate {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.AddUpdate();
        $.alpha = reader.float32_(position, 4, 1);
        $.beta = reader.float32_(position, 6, 1);
        $.bias = reader.float32_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Elemwise = class Elemwise {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Elemwise();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ElemwiseMultiType = class ElemwiseMultiType {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ElemwiseMultiType();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.PowC = class PowC {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.PowC();
        $.exp = reader.float32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.DctChannelSelectV0 = class DctChannelSelectV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.DctChannelSelectV0();
        $.format = reader.uint32_(position, 4, 0);
        $.fastImpl = reader.uint32_(position, 6, 0);
        $.dct_block_size = reader.int32_(position, 8, 8);
        return $;
    }
};

mgb.serialization.fbs.param.DctChannelSelect = class DctChannelSelect {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.DctChannelSelect();
        $.format = reader.uint32_(position, 4, 0);
        $.fastImpl = reader.uint32_(position, 6, 0);
        $.dct_block_size = reader.int32_(position, 8, 8);
        return $;
    }
};

mgb.serialization.fbs.param.MatrixMulV0 = class MatrixMulV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MatrixMulV0();
        $.transposeA = reader.bool_(position, 4, false);
        $.transposeB = reader.bool_(position, 6, false);
        $.data_type = reader.uint32_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.MatrixMulV1 = class MatrixMulV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MatrixMulV1();
        $.transposeA = reader.bool_(position, 4, false);
        $.transposeB = reader.bool_(position, 6, false);
        $.compute_mode = reader.uint32_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.MatrixMul = class MatrixMul {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MatrixMul();
        $.transposeA = reader.bool_(position, 4, false);
        $.transposeB = reader.bool_(position, 6, false);
        $.compute_mode = reader.uint32_(position, 8, 0);
        $.format = reader.uint32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.SVD = class SVD {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SVD();
        $.full_matrices = reader.bool_(position, 4, false);
        $.compute_uv = reader.bool_(position, 6, true);
        return $;
    }
};

mgb.serialization.fbs.param.ReduceV0 = class ReduceV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ReduceV0();
        $.mode = reader.uint32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, -1);
        return $;
    }
};

mgb.serialization.fbs.param.ReduceV1 = class ReduceV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ReduceV1();
        $.mode = reader.uint32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, -1);
        $.data_type = reader.uint32_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Reduce = class Reduce {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Reduce();
        $.mode = reader.uint32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 2147483647);
        $.data_type = reader.uint32_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.CumsumV0 = class CumsumV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CumsumV0();
        $.axis = reader.int32_(position, 4, -1);
        $.exclusive = reader.bool_(position, 6, true);
        $.reverse = reader.bool_(position, 8, false);
        return $;
    }
};

mgb.serialization.fbs.param.Cumsum = class Cumsum {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Cumsum();
        $.axis = reader.int32_(position, 4, 2147483647);
        $.exclusive = reader.bool_(position, 6, true);
        $.reverse = reader.bool_(position, 8, false);
        return $;
    }
};

mgb.serialization.fbs.param.CondTake = class CondTake {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CondTake();
        $.mode = reader.uint32_(position, 4, 0);
        $.val = reader.float32_(position, 6, 0);
        $.eps = reader.float32_(position, 8, 0.000001);
        return $;
    }
};

mgb.serialization.fbs.param.Argsort = class Argsort {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Argsort();
        $.order = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.IndexingRemap = class IndexingRemap {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.IndexingRemap();
        $.is_non_overlapping = reader.bool_(position, 4, false);
        return $;
    }
};

mgb.serialization.fbs.param.Sleep = class Sleep {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Sleep();
        $.time = reader.float32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Linspace = class Linspace {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Linspace();
        $.endpoint = reader.bool_(position, 4, true);
        return $;
    }
};

mgb.serialization.fbs.param.LinspaceFull = class LinspaceFull {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LinspaceFull();
        $.start = reader.float64_(position, 4, 0);
        $.stop = reader.float64_(position, 6, 1);
        $.endpoint = reader.bool_(position, 8, true);
        return $;
    }
};

mgb.serialization.fbs.param.Eye = class Eye {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Eye();
        $.k = reader.int32_(position, 4, 0);
        $.dtype = reader.int8_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Diag = class Diag {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Diag();
        $.k = reader.int32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.UniformRNGV0 = class UniformRNGV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.UniformRNGV0();
        $.seed = reader.uint64_(position, 4, 0n);
        return $;
    }
};

mgb.serialization.fbs.param.UniformRNG = class UniformRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.UniformRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        $.dtype = reader.int8_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.GaussianRNGV0 = class GaussianRNGV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.GaussianRNGV0();
        $.seed = reader.uint64_(position, 4, 0n);
        $.mean = reader.float32_(position, 6, 0);
        $.std = reader.float32_(position, 8, 1);
        return $;
    }
};

mgb.serialization.fbs.param.GaussianRNG = class GaussianRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.GaussianRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        $.mean = reader.float32_(position, 6, 0);
        $.std = reader.float32_(position, 8, 1);
        $.dtype = reader.int8_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.GammaRNG = class GammaRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.GammaRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        return $;
    }
};

mgb.serialization.fbs.param.BetaRNG = class BetaRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.BetaRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        return $;
    }
};

mgb.serialization.fbs.param.PoissonRNG = class PoissonRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.PoissonRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        return $;
    }
};

mgb.serialization.fbs.param.MultinomialRNG = class MultinomialRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MultinomialRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        $.num_samples = reader.uint64_(position, 6, 1n);
        $.replacement = reader.bool_(position, 8, false);
        return $;
    }
};

mgb.serialization.fbs.param.PermutationRNG = class PermutationRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.PermutationRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        $.dtype = reader.int8_(position, 6, 4);
        return $;
    }
};

mgb.serialization.fbs.param.ShuffleRNG = class ShuffleRNG {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ShuffleRNG();
        $.seed = reader.uint64_(position, 4, 0n);
        return $;
    }
};

mgb.serialization.fbs.param.Flip = class Flip {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Flip();
        $.vertical = reader.bool_(position, 4, false);
        $.horizontal = reader.bool_(position, 6, false);
        return $;
    }
};

mgb.serialization.fbs.param.Rotate = class Rotate {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Rotate();
        $.clockwise = reader.bool_(position, 4, true);
        return $;
    }
};

mgb.serialization.fbs.param.ROICopy = class ROICopy {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ROICopy();
        $.row_from = reader.uint32_(position, 4, 0);
        $.row_to = reader.uint32_(position, 6, 0);
        $.col_from = reader.uint32_(position, 8, 0);
        $.col_to = reader.uint32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.CvtColor = class CvtColor {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CvtColor();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.WarpAffineV0 = class WarpAffineV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.WarpAffineV0();
        $.imode = reader.uint32_(position, 4, 1);
        $.border_mode = reader.uint32_(position, 6, 0);
        $.border_val = reader.float32_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.WarpAffineV1 = class WarpAffineV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.WarpAffineV1();
        $.imode = reader.uint32_(position, 4, 1);
        $.border_mode = reader.uint32_(position, 6, 0);
        $.border_val = reader.float32_(position, 8, 0);
        $.format = reader.uint32_(position, 10, 1);
        return $;
    }
};

mgb.serialization.fbs.param.WarpAffine = class WarpAffine {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.WarpAffine();
        $.imode = reader.uint32_(position, 4, 1);
        $.border_mode = reader.uint32_(position, 6, 0);
        $.border_val = reader.float32_(position, 8, 0);
        $.format = reader.uint32_(position, 10, 1);
        return $;
    }
};

mgb.serialization.fbs.param.GaussianBlur = class GaussianBlur {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.GaussianBlur();
        $.border_mode = reader.uint32_(position, 4, 0);
        $.kernel_height = reader.uint32_(position, 6, 0);
        $.kernel_width = reader.uint32_(position, 8, 0);
        $.sigma_x = reader.float32_(position, 10, 0);
        $.sigma_y = reader.float32_(position, 12, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ResizeV0 = class ResizeV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ResizeV0();
        $.imode = reader.uint32_(position, 4, 1);
        return $;
    }
};

mgb.serialization.fbs.param.ResizeV1 = class ResizeV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ResizeV1();
        $.imode = reader.uint32_(position, 4, 1);
        $.format = reader.uint32_(position, 6, 1);
        return $;
    }
};

mgb.serialization.fbs.param.Resize = class Resize {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Resize();
        $.imode = reader.uint32_(position, 4, 1);
        $.format = reader.uint32_(position, 6, 1);
        return $;
    }
};

mgb.serialization.fbs.param.RemapV0 = class RemapV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.RemapV0();
        $.imode = reader.uint32_(position, 4, 1);
        $.border_type = reader.uint32_(position, 6, 0);
        $.format = reader.uint32_(position, 8, 1);
        $.scalar = reader.float32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Remap = class Remap {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Remap();
        $.imode = reader.uint32_(position, 4, 1);
        $.border_type = reader.uint32_(position, 6, 0);
        $.format = reader.uint32_(position, 8, 1);
        $.scalar = reader.float32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Convolution3D = class Convolution3D {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Convolution3D();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_d = reader.uint32_(position, 6, 0);
        $.pad_h = reader.uint32_(position, 8, 0);
        $.pad_w = reader.uint32_(position, 10, 0);
        $.stride_d = reader.uint32_(position, 12, 1);
        $.stride_h = reader.uint32_(position, 14, 1);
        $.stride_w = reader.uint32_(position, 16, 1);
        $.dilate_d = reader.uint32_(position, 18, 1);
        $.dilate_h = reader.uint32_(position, 20, 1);
        $.dilate_w = reader.uint32_(position, 22, 1);
        $.sparse = reader.uint32_(position, 24, 0);
        $.data_type = reader.uint32_(position, 26, 0);
        $.format = reader.uint32_(position, 28, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Conv3DBias = class Conv3DBias {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Conv3DBias();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.pad_d = reader.uint32_(position, 8, 0);
        $.pad_h = reader.uint32_(position, 10, 0);
        $.pad_w = reader.uint32_(position, 12, 0);
        $.stride_d = reader.uint32_(position, 14, 1);
        $.stride_h = reader.uint32_(position, 16, 1);
        $.stride_w = reader.uint32_(position, 18, 0);
        return $;
    }
};

mgb.serialization.fbs.param.SeparableConv3D = class SeparableConv3D {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SeparableConv3D();
        $.mode = reader.uint32_(position, 4, 0);
        $.borderMode = reader.uint32_(position, 6, 0);
        $.is_symm_kernel = reader.bool_(position, 8, true);
        $.pad_d = reader.uint32_(position, 10, 0);
        $.pad_h = reader.uint32_(position, 12, 0);
        $.pad_w = reader.uint32_(position, 14, 0);
        $.stride_d = reader.uint32_(position, 16, 0);
        $.stride_h = reader.uint32_(position, 18, 1);
        $.stride_w = reader.uint32_(position, 20, 1);
        $.ksize_d = reader.uint32_(position, 22, 0);
        $.ksize_h = reader.uint32_(position, 24, 3);
        $.ksize_w = reader.uint32_(position, 26, 3);
        $.anchor_d = reader.uint32_(position, 28, 0);
        $.anchor_h = reader.uint32_(position, 30, 1);
        $.anchor_w = reader.uint32_(position, 32, 1);
        return $;
    }
};

mgb.serialization.fbs.param.TopK = class TopK {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.TopK();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.RelayoutFormatV0 = class RelayoutFormatV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.RelayoutFormatV0();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.RelayoutFormat = class RelayoutFormat {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.RelayoutFormat();
        $.mode = reader.uint32_(position, 4, 0);
        $.oc = reader.uint32_(position, 6, 0);
        $.group = reader.uint32_(position, 8, 1);
        return $;
    }
};

mgb.serialization.fbs.param.SeparableFilterV0 = class SeparableFilterV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SeparableFilterV0();
        $.format = reader.uint32_(position, 4, 0);
        $.borderMode = reader.uint32_(position, 6, 0);
        $.is_symm_kernel = reader.bool_(position, 8, true);
        $.ksize_h = reader.uint32_(position, 10, 3);
        $.ksize_w = reader.uint32_(position, 12, 3);
        $.anchor_h = reader.uint32_(position, 14, 1);
        $.anchor_w = reader.uint32_(position, 16, 1);
        return $;
    }
};

mgb.serialization.fbs.param.SeparableFilter = class SeparableFilter {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.SeparableFilter();
        $.format = reader.uint32_(position, 4, 0);
        $.borderMode = reader.uint32_(position, 6, 0);
        $.is_symm_kernel = reader.bool_(position, 8, true);
        $.ksize_h = reader.uint32_(position, 10, 3);
        $.ksize_w = reader.uint32_(position, 12, 3);
        $.anchor_h = reader.uint32_(position, 14, 1);
        $.anchor_w = reader.uint32_(position, 16, 1);
        return $;
    }
};

mgb.serialization.fbs.param.LocalShareV0 = class LocalShareV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LocalShareV0();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 1);
        $.stride_w = reader.uint32_(position, 12, 1);
        $.dilate_h = reader.uint32_(position, 14, 1);
        $.dilate_w = reader.uint32_(position, 16, 1);
        $.spatial_groups_h = reader.uint32_(position, 18, 1);
        $.spatial_groups_w = reader.uint32_(position, 20, 1);
        $.sparse = reader.uint32_(position, 22, 0);
        $.format = reader.uint32_(position, 24, 0);
        $.computeMode = reader.uint32_(position, 26, 0);
        return $;
    }
};

mgb.serialization.fbs.param.LocalShare = class LocalShare {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LocalShare();
        $.mode = reader.uint32_(position, 4, 0);
        $.pad_h = reader.uint32_(position, 6, 0);
        $.pad_w = reader.uint32_(position, 8, 0);
        $.stride_h = reader.uint32_(position, 10, 1);
        $.stride_w = reader.uint32_(position, 12, 1);
        $.dilate_h = reader.uint32_(position, 14, 1);
        $.dilate_w = reader.uint32_(position, 16, 1);
        $.spatial_groups_h = reader.uint32_(position, 18, 1);
        $.spatial_groups_w = reader.uint32_(position, 20, 1);
        $.sparse = reader.uint32_(position, 22, 0);
        $.format = reader.uint32_(position, 24, 0);
        $.computeMode = reader.uint32_(position, 26, 0);
        return $;
    }
};

mgb.serialization.fbs.param.ROIAlignV0 = class ROIAlignV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ROIAlignV0();
        $.mode = reader.uint32_(position, 4, 0);
        $.format = reader.uint32_(position, 6, 0);
        $.spatial_scale = reader.float32_(position, 8, 1);
        $.offset = reader.float32_(position, 10, 0);
        $.pooled_height = reader.uint32_(position, 12, 1);
        $.pooled_width = reader.uint32_(position, 14, 1);
        $.sample_height = reader.uint32_(position, 16, 2);
        $.sample_width = reader.uint32_(position, 18, 2);
        return $;
    }
};

mgb.serialization.fbs.param.ROIAlign = class ROIAlign {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ROIAlign();
        $.mode = reader.uint32_(position, 4, 0);
        $.format = reader.uint32_(position, 6, 0);
        $.spatial_scale = reader.float32_(position, 8, 1);
        $.offset = reader.float32_(position, 10, 0);
        $.pooled_height = reader.uint32_(position, 12, 1);
        $.pooled_width = reader.uint32_(position, 14, 1);
        $.sample_height = reader.uint32_(position, 16, 2);
        $.sample_width = reader.uint32_(position, 18, 2);
        return $;
    }
};

mgb.serialization.fbs.param.Correlation = class Correlation {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Correlation();
        $.format = reader.uint32_(position, 4, 0);
        $.kernel_size = reader.uint32_(position, 6, 1);
        $.max_displacement = reader.uint32_(position, 8, 1);
        $.stride1 = reader.uint32_(position, 10, 1);
        $.stride2 = reader.uint32_(position, 12, 1);
        $.pad_size = reader.uint32_(position, 14, 0);
        $.is_multiply = reader.bool_(position, 16, true);
        return $;
    }
};

mgb.serialization.fbs.param.DeformablePSROIPooling = class DeformablePSROIPooling {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.DeformablePSROIPooling();
        $.no_trans = reader.bool_(position, 4, true);
        $.spatial_scale = reader.float32_(position, 6, 1);
        $.trans_std = reader.float32_(position, 8, 1);
        $.pooled_h = reader.uint32_(position, 10, 1);
        $.pooled_w = reader.uint32_(position, 12, 1);
        $.part_size = reader.uint32_(position, 14, 1);
        $.sample_per_part = reader.uint32_(position, 16, 1);
        return $;
    }
};

mgb.serialization.fbs.param.BatchConvBiasV0 = class BatchConvBiasV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.BatchConvBiasV0();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.pad_h = reader.uint32_(position, 8, 0);
        $.pad_w = reader.uint32_(position, 10, 0);
        $.stride_h = reader.uint32_(position, 12, 1);
        $.stride_w = reader.uint32_(position, 14, 1);
        $.dilate_h = reader.uint32_(position, 16, 1);
        $.dilate_w = reader.uint32_(position, 18, 1);
        $.sparse = reader.uint32_(position, 20, 0);
        $.format = reader.uint32_(position, 22, 0);
        $.compute_mode = reader.uint32_(position, 24, 0);
        return $;
    }
};

mgb.serialization.fbs.param.BatchConvBias = class BatchConvBias {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.BatchConvBias();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        $.mode = reader.uint32_(position, 6, 0);
        $.pad_h = reader.uint32_(position, 8, 0);
        $.pad_w = reader.uint32_(position, 10, 0);
        $.stride_h = reader.uint32_(position, 12, 1);
        $.stride_w = reader.uint32_(position, 14, 1);
        $.dilate_h = reader.uint32_(position, 16, 1);
        $.dilate_w = reader.uint32_(position, 18, 1);
        $.sparse = reader.uint32_(position, 20, 0);
        $.format = reader.uint32_(position, 22, 0);
        $.compute_mode = reader.uint32_(position, 24, 0);
        return $;
    }
};

mgb.serialization.fbs.param.FakeQuant = class FakeQuant {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.FakeQuant();
        $.qmin = reader.int32_(position, 4, -2147483648);
        $.qmax = reader.int32_(position, 6, 2147483647);
        return $;
    }
};

mgb.serialization.fbs.param.TQT = class TQT {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.TQT();
        $.qmin = reader.int32_(position, 4, -2147483648);
        $.qmax = reader.int32_(position, 6, 2147483647);
        return $;
    }
};

mgb.serialization.fbs.param.LSQ = class LSQ {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LSQ();
        $.qmin = reader.int32_(position, 4, -2147483648);
        $.qmax = reader.int32_(position, 6, 2147483647);
        return $;
    }
};

mgb.serialization.fbs.param.Fill = class Fill {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Fill();
        $.value = reader.float32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.CheckNonFinite = class CheckNonFinite {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CheckNonFinite();
        $.scale = reader.float32_(position, 4, 1);
        return $;
    }
};

mgb.serialization.fbs.param.Padding = class Padding {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Padding();
        $.front_offset_dim0 = reader.uint32_(position, 4, 0);
        $.front_offset_dim1 = reader.uint32_(position, 6, 0);
        $.front_offset_dim2 = reader.uint32_(position, 8, 0);
        $.front_offset_dim3 = reader.uint32_(position, 10, 0);
        $.front_offset_dim4 = reader.uint32_(position, 12, 0);
        $.front_offset_dim5 = reader.uint32_(position, 14, 0);
        $.front_offset_dim6 = reader.uint32_(position, 16, 0);
        $.back_offset_dim0 = reader.uint32_(position, 18, 0);
        $.back_offset_dim1 = reader.uint32_(position, 20, 0);
        $.back_offset_dim2 = reader.uint32_(position, 22, 0);
        $.back_offset_dim3 = reader.uint32_(position, 24, 0);
        $.back_offset_dim4 = reader.uint32_(position, 26, 0);
        $.back_offset_dim5 = reader.uint32_(position, 28, 0);
        $.back_offset_dim6 = reader.uint32_(position, 30, 0);
        $.padding_val = reader.float32_(position, 32, 0);
        $.padding_mode = reader.uint32_(position, 34, 2);
        return $;
    }
};

mgb.serialization.fbs.param.LayerNorm = class LayerNorm {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LayerNorm();
        $.affine = reader.bool_(position, 4, true);
        $.eps = reader.float32_(position, 6, 0.00001);
        $.normalized_dim = reader.uint64_(position, 8, 1n);
        $.normalized_size = reader.uint64_(position, 10, 1n);
        return $;
    }
};

mgb.serialization.fbs.param.GroupNorm = class GroupNorm {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.GroupNorm();
        $.affine = reader.bool_(position, 4, true);
        $.eps = reader.float32_(position, 6, 0.00001);
        $.group = reader.uint32_(position, 8, 1);
        $.format = reader.uint32_(position, 10, 0);
        return $;
    }
};

mgb.serialization.fbs.param.Dropout = class Dropout {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Dropout();
        $.drop_prob = reader.float32_(position, 4, 0);
        $.seed = reader.uint64_(position, 6, 0n);
        return $;
    }
};

mgb.serialization.fbs.param.RNNCell = class RNNCell {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.RNNCell();
        $.nonlineMode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.RNN = class RNN {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.RNN();
        $.num_layers = reader.uint32_(position, 4, 1);
        $.bidirectional = reader.bool_(position, 6, false);
        $.bias = reader.bool_(position, 8, true);
        $.hidden_size = reader.uint32_(position, 10, 128);
        $.dropout = reader.float32_(position, 12, 0);
        $.nonlineMode = reader.uint32_(position, 14, 0);
        $.fwd_mode = reader.uint32_(position, 16, 0);
        return $;
    }
};

mgb.serialization.fbs.param.LSTM = class LSTM {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.LSTM();
        $.num_layers = reader.uint32_(position, 4, 1);
        $.bidirectional = reader.bool_(position, 6, false);
        $.bias = reader.bool_(position, 8, true);
        $.hidden_size = reader.uint32_(position, 10, 128);
        $.proj_size = reader.uint32_(position, 12, 0);
        $.dropout = reader.float32_(position, 14, 0);
        $.fwd_mode = reader.uint32_(position, 16, 0);
        return $;
    }
};

mgb.serialization.fbs.param.CollectiveCommMode = {
    REDUCE_SUM: 0,
    BROADCAST: 1,
    ALL_GATHER: 2,
    REDUCE_SCATTER_SUM: 3,
    ALL_REDUCE_SUM: 4,
    ALL_REDUCE_MAX: 5,
    ALL_REDUCE_MIN: 6,
    ALL_REDUCE_PROD: 7,
    GATHER: 8,
    SCATTER: 9,
    ALL_TO_ALL: 10
};

mgb.serialization.fbs.param.CondExecMarkGradMode = {
    SUM: 0,
    SUM_COND_OUT: 1
};

mgb.serialization.fbs.param.CondExecMarkStaticInfer = {
    SHAPE_VALUE: 0,
    SHAPE_ONLY: 1,
    NONE: 2
};

mgb.serialization.fbs.param.CondExecMergeMode = {
    EXACT_ONE: 0,
    EXACT_ONE_SAME_SHAPE: 1,
    SUM: 2,
    SUM_COND_OUT: 3
};

mgb.serialization.fbs.param.CondExecPredMode = {
    CASE: 0,
    CASE_FALLBACK: 1,
    PIECEWISE: 2
};

mgb.serialization.fbs.param.CondExecPredLogicalMode = {
    OR: 0,
    AND: 1,
    XOR: 2,
    NOR: 3,
    NAND: 4,
    XNOR: 5
};

mgb.serialization.fbs.param.ExecutionPolicyStrategy = {
    HEURISTIC: 0,
    PROFILE: 1,
    REPRODUCIBLE: 2,
    OPTIMIZED: 3
};

mgb.serialization.fbs.param.ExecutionPolicyV0Strategy = {
    HEURISTIC: 0,
    HEURISTIC_REPRODUCIBLE: 1,
    PROFILE: 2,
    PROFILE_REPRODUCIBLE: 3,
    PROFILE_HEURISTIC: 4
};

mgb.serialization.fbs.param.DType = class DType {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.DType();
        $.dtype = reader.int8_(position, 4, 8);
        return $;
    }
};

mgb.serialization.fbs.param.PersistentOutputStorage = class PersistentOutputStorage {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.PersistentOutputStorage();
        $.share_key = reader.int32_(position, 4, -1);
        return $;
    }
};

mgb.serialization.fbs.param.OptionalAxis = class OptionalAxis {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.OptionalAxis();
        $.axis = reader.int32_(position, 4, -1);
        return $;
    }
};

mgb.serialization.fbs.param.OptionalAxisV1 = class OptionalAxisV1 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.OptionalAxisV1();
        $.axis = reader.int32_(position, 4, 7);
        return $;
    }
};

mgb.serialization.fbs.param.ExecutionPolicyV0 = class ExecutionPolicyV0 {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ExecutionPolicyV0();
        $.strategy = reader.uint32_(position, 4, 0);
        $.workspace_limit = reader.uint64_(position, 6, 18446744073709552000n);
        return $;
    }
};

mgb.serialization.fbs.param.ExecutionPolicy = class ExecutionPolicy {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.ExecutionPolicy();
        $.strategy = reader.uint32_(position, 4, 1);
        $.workspace_limit = reader.uint64_(position, 6, 18446744073709552000n);
        return $;
    }
};

mgb.serialization.fbs.param.AssertEqual = class AssertEqual {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.AssertEqual();
        $.maxerr = reader.float32_(position, 4, 0.0001);
        $.verbose = reader.bool_(position, 6, false);
        return $;
    }
};

mgb.serialization.fbs.param.FpgaConv = class FpgaConv {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.FpgaConv();
        $.need_output_quantize = reader.bool_(position, 4, false);
        $.need_output_threshold = reader.bool_(position, 6, false);
        $.stride = reader.int32_(position, 8, 1);
        $.input_bit_width = reader.int32_(position, 10, 2);
        $.output_bit_width = reader.int32_(position, 12, 2);
        $.weight_bit_width = reader.int32_(position, 14, 2);
        $.thres0 = reader.int32_(position, 16, 0);
        $.thres1 = reader.int32_(position, 18, 1);
        $.unpool_size = reader.uint32_(position, 20, 4);
        $.direct_size = reader.uint32_(position, 22, 4);
        return $;
    }
};

mgb.serialization.fbs.param.CollectiveComm = class CollectiveComm {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CollectiveComm();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.FakeSerializedDType = class FakeSerializedDType {

    static decode(/* reader, position */) {
        const $ = new mgb.serialization.fbs.param.FakeSerializedDType();
        return $;
    }
};

mgb.serialization.fbs.param.CondExecPred = class CondExecPred {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CondExecPred();
        $.mode = reader.uint32_(position, 4, 0);
        $.eps = reader.float32_(position, 6, 0.0001);
        return $;
    }
};

mgb.serialization.fbs.param.CondExecPredLogical = class CondExecPredLogical {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CondExecPredLogical();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.param.CondExecMark = class CondExecMark {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CondExecMark();
        $.grad_mode = reader.uint32_(position, 4, 0);
        $.static_infer = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.CondExecMerge = class CondExecMerge {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.CondExecMerge();
        $.nr_output = reader.uint32_(position, 4, 1);
        $.mode = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.NvOf = class NvOf {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.NvOf();
        $.precision = reader.uint32_(position, 4, 1);
        return $;
    }
};

mgb.serialization.fbs.param.PersistentDTypeScalar = class PersistentDTypeScalar {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.PersistentDTypeScalar();
        $.dtype = reader.int8(position + 0);
        $.storage = undefined; // not implemented
        return $;
    }
};

mgb.serialization.fbs.param.MGBAddUpdate = class MGBAddUpdate {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MGBAddUpdate();
        $.alpha = reader.struct(position, 4, mgb.serialization.fbs.param.PersistentDTypeScalar);
        $.beta = reader.struct(position, 6, mgb.serialization.fbs.param.PersistentDTypeScalar);
        $.bias = reader.struct(position, 8, mgb.serialization.fbs.param.PersistentDTypeScalar);
        return $;
    }
};

mgb.serialization.fbs.param.Host2DeviceCopy = class Host2DeviceCopy {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Host2DeviceCopy();
        $.enable_value_infer = reader.bool_(position, 4, true);
        $.dump_default_value = reader.bool_(position, 6, false);
        $.allow_cpu_mem_fwd = reader.bool_(position, 8, true);
        return $;
    }
};

mgb.serialization.fbs.param.Dimshuffle = class Dimshuffle {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.Dimshuffle();
        $.pattern = reader.array(position, 4, Int32Array);
        $.ndim = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.param.AxisDescMethod = {
    ADD_1: 0,
    REMOVE: 1
};

mgb.serialization.fbs.param.AxisDesc = class AxisDesc {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.AxisDesc();
        $.method = reader.int8(position + 0);
        $.axis = reader.int32(position + 4);
        return $;
    }
};

mgb.serialization.fbs.param.AxisAddRemove = class AxisAddRemove {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.AxisAddRemove();
        $.desc = reader.structs(position, 4, mgb.serialization.fbs.param.AxisDesc);
        return $;
    }
};

mgb.serialization.fbs.param.MGBSleep = class MGBSleep {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.MGBSleep();
        $.device = reader.bool_(position, 4, true);
        $.host = reader.bool_(position, 6, false);
        $.seconds = reader.float64_(position, 8, 0);
        return $;
    }
};

mgb.serialization.fbs.param.IndexDescMaskItem = class IndexDescMaskItem {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.IndexDescMaskItem();
        $.axis = reader.int8(position + 0);
        $.begin = reader.bool(position + 1);
        $.end = reader.bool(position + 2);
        $.step = reader.bool(position + 3);
        $.idx = reader.bool(position + 4);
        return $;
    }
};

mgb.serialization.fbs.param.IndexDescMaskDump = class IndexDescMaskDump {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.IndexDescMaskDump();
        $.items = reader.structs(position, 4, mgb.serialization.fbs.param.IndexDescMaskItem);
        return $;
    }
};

mgb.serialization.fbs.param.NMSKeep = class NMSKeep {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.param.NMSKeep();
        $.iou_thresh = reader.float32_(position, 4, 0);
        $.max_output = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization = mgb.serialization || {};

mgb.serialization.fbs = mgb.serialization.fbs || {};

mgb.serialization.fbs.v2 = mgb.serialization.fbs.v2 || {};

mgb.serialization.fbs.v2.CompNode = class CompNode {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.CompNode();
        $.logical_locator = reader.string_(position, 4, null);
        return $;
    }
};

mgb.serialization.fbs.v2.DefaultTensorFormat = class DefaultTensorFormat {

    static decode(/* reader, position */) {
        const $ = new mgb.serialization.fbs.v2.DefaultTensorFormat();
        return $;
    }
};

mgb.serialization.fbs.v2.Image2DPackedTensorFormat = class Image2DPackedTensorFormat {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.Image2DPackedTensorFormat();
        $.align_axis = reader.uint8_(position, 4, 0);
        return $;
    }
};

mgb.serialization.fbs.v2.LowbitsAlignedTensorFormat = class LowbitsAlignedTensorFormat {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.LowbitsAlignedTensorFormat();
        $.size_nbits = reader.uint8_(position, 4, 0);
        $.align_size_in_bits = reader.uint8_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.v2.TensorFormat = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return mgb.serialization.fbs.v2.DefaultTensorFormat.decode(reader, position);
            case 2: return mgb.serialization.fbs.v2.Image2DPackedTensorFormat.decode(reader, position);
            case 3: return mgb.serialization.fbs.v2.LowbitsAlignedTensorFormat.decode(reader, position);
            default: return undefined;
        }
    }
};

mgb.serialization.fbs.v2.Blob = class Blob {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.Blob();
        $.data = reader.array(position, 4, Uint8Array);
        return $;
    }
};

mgb.serialization.fbs.v2.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.Tensor();
        $.name = reader.string_(position, 4, null);
        $.shape = reader.array(position, 6, Uint32Array);
        $.comp_node = reader.table(position, 8, mgb.serialization.fbs.v2.CompNode);
        $.dtype = reader.table(position, 10, mgb.serialization.fbs.DType);
        $.format = reader.union(position, 12, mgb.serialization.fbs.v2.TensorFormat);
        $.data = reader.array(position, 16, Uint8Array);
        return $;
    }
};

mgb.serialization.fbs.v2.Reserved0 = class Reserved0 {

    static decode(/* reader, position */) {
        const $ = new mgb.serialization.fbs.v2.Reserved0();
        return $;
    }
};

mgb.serialization.fbs.v2.DeprecatedParam = class DeprecatedParam {

    static decode(/* reader, position */) {
        const $ = new mgb.serialization.fbs.v2.DeprecatedParam();
        return $;
    }
};

mgb.serialization.fbs.v2.OperatorParam = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return mgb.serialization.fbs.param.Empty.decode(reader, position);
            case 2: return mgb.serialization.fbs.param.Axis.decode(reader, position);
            case 3: return mgb.serialization.fbs.param.Convolution.decode(reader, position);
            case 4: return mgb.serialization.fbs.param.MaskPropagate.decode(reader, position);
            case 5: return mgb.serialization.fbs.param.ConvPooling.decode(reader, position);
            case 6: return mgb.serialization.fbs.param.ConvBias.decode(reader, position);
            case 7: return mgb.serialization.fbs.param.SeparableConv.decode(reader, position);
            case 8: return mgb.serialization.fbs.param.Images2Neibs.decode(reader, position);
            case 9: return mgb.serialization.fbs.param.Pooling.decode(reader, position);
            case 10: return mgb.serialization.fbs.param.LRN.decode(reader, position);
            case 11: return mgb.serialization.fbs.param.BN.decode(reader, position);
            case 12: return mgb.serialization.fbs.param.ROIPooling.decode(reader, position);
            case 13: return mgb.serialization.fbs.param.WarpPerspective.decode(reader, position);
            case 14: return mgb.serialization.fbs.param.SpatialTfGridGenerator.decode(reader, position);
            case 15: return mgb.serialization.fbs.param.SpatialTfSampler.decode(reader, position);
            case 16: return mgb.serialization.fbs.param.MGBAddUpdate.decode(reader, position);
            case 17: return mgb.serialization.fbs.param.Elemwise.decode(reader, position);
            case 18: return mgb.serialization.fbs.param.ElemwiseMultiType.decode(reader, position);
            case 19: return mgb.serialization.fbs.param.PowC.decode(reader, position);
            case 20: return mgb.serialization.fbs.param.MatrixMul.decode(reader, position);
            case 21: return mgb.serialization.fbs.v2.DeprecatedParam.decode(reader, position);
            case 22: return mgb.serialization.fbs.param.SVD.decode(reader, position);
            case 23: return mgb.serialization.fbs.param.Reduce.decode(reader, position);
            case 24: return mgb.serialization.fbs.param.Cumsum.decode(reader, position);
            case 25: return mgb.serialization.fbs.param.CondTake.decode(reader, position);
            case 26: return mgb.serialization.fbs.param.Argsort.decode(reader, position);
            case 27: return mgb.serialization.fbs.param.IndexingRemap.decode(reader, position);
            case 28: return mgb.serialization.fbs.param.MGBSleep.decode(reader, position);
            case 29: return mgb.serialization.fbs.param.Linspace.decode(reader, position);
            case 30: return mgb.serialization.fbs.param.LinspaceFull.decode(reader, position);
            case 31: return mgb.serialization.fbs.param.Eye.decode(reader, position);
            case 32: return mgb.serialization.fbs.param.UniformRNG.decode(reader, position);
            case 33: return mgb.serialization.fbs.param.GaussianRNG.decode(reader, position);
            case 34: return mgb.serialization.fbs.param.Flip.decode(reader, position);
            case 35: return mgb.serialization.fbs.param.Rotate.decode(reader, position);
            case 36: return mgb.serialization.fbs.param.ROICopy.decode(reader, position);
            case 37: return mgb.serialization.fbs.param.CvtColor.decode(reader, position);
            case 38: return mgb.serialization.fbs.param.WarpAffine.decode(reader, position);
            case 39: return mgb.serialization.fbs.param.GaussianBlur.decode(reader, position);
            case 40: return mgb.serialization.fbs.param.Resize.decode(reader, position);
            case 41: return mgb.serialization.fbs.param.Convolution3D.decode(reader, position);
            case 42: return mgb.serialization.fbs.param.Conv3DBias.decode(reader, position);
            case 43: return mgb.serialization.fbs.param.SeparableConv3D.decode(reader, position);
            case 44: return mgb.serialization.fbs.param.TopK.decode(reader, position);
            case 45: return mgb.serialization.fbs.param.RelayoutFormat.decode(reader, position);
            case 46: return mgb.serialization.fbs.param.SeparableFilter.decode(reader, position);
            case 47: return mgb.serialization.fbs.param.LocalShare.decode(reader, position);
            case 48: return mgb.serialization.fbs.param.ROIAlign.decode(reader, position);
            case 49: return mgb.serialization.fbs.param.DeformablePSROIPooling.decode(reader, position);
            case 50: return mgb.serialization.fbs.param.BatchConvBias.decode(reader, position);
            case 51: return mgb.serialization.fbs.param.DType.decode(reader, position);
            case 52: return mgb.serialization.fbs.param.PersistentOutputStorage.decode(reader, position);
            case 53: return mgb.serialization.fbs.param.OptionalAxis.decode(reader, position);
            case 54: return mgb.serialization.fbs.param.OptionalAxisV1.decode(reader, position);
            case 55: return mgb.serialization.fbs.param.ExecutionPolicy.decode(reader, position);
            case 56: return mgb.serialization.fbs.param.AssertEqual.decode(reader, position);
            case 57: return mgb.serialization.fbs.param.FpgaConv.decode(reader, position);
            case 58: return mgb.serialization.fbs.param.CollectiveComm.decode(reader, position);
            case 59: return mgb.serialization.fbs.param.CondExecPred.decode(reader, position);
            case 60: return mgb.serialization.fbs.param.CondExecPredLogical.decode(reader, position);
            case 61: return mgb.serialization.fbs.param.CondExecMark.decode(reader, position);
            case 62: return mgb.serialization.fbs.param.CondExecMerge.decode(reader, position);
            case 63: return mgb.serialization.fbs.param.Host2DeviceCopy.decode(reader, position);
            case 64: return mgb.serialization.fbs.param.Dimshuffle.decode(reader, position);
            case 65: return mgb.serialization.fbs.param.AxisAddRemove.decode(reader, position);
            case 66: return mgb.serialization.fbs.param.IndexDescMaskDump.decode(reader, position);
            case 67: return mgb.serialization.fbs.DType.decode(reader, position);
            case 68: return mgb.serialization.fbs.param.Remap.decode(reader, position);
            case 69: return mgb.serialization.fbs.param.NMSKeep.decode(reader, position);
            case 70: return mgb.serialization.fbs.param.AdaptivePooling.decode(reader, position);
            case 71: return mgb.serialization.fbs.param.NvOf.decode(reader, position);
            case 72: return mgb.serialization.fbs.param.DctChannelSelect.decode(reader, position);
            case 73: return mgb.serialization.fbs.param.FakeQuant.decode(reader, position);
            case 74: return mgb.serialization.fbs.param.TQT.decode(reader, position);
            case 75: return mgb.serialization.fbs.param.Correlation.decode(reader, position);
            case 76: return mgb.serialization.fbs.param.LSQ.decode(reader, position);
            case 77: return mgb.serialization.fbs.param.GammaRNG.decode(reader, position);
            case 78: return mgb.serialization.fbs.param.PoissonRNG.decode(reader, position);
            case 79: return mgb.serialization.fbs.param.PermutationRNG.decode(reader, position);
            case 80: return mgb.serialization.fbs.param.BetaRNG.decode(reader, position);
            case 81: return mgb.serialization.fbs.param.SlidingWindowTranspose.decode(reader, position);
            case 82: return mgb.serialization.fbs.param.Padding.decode(reader, position);
            case 83: return mgb.serialization.fbs.param.ShuffleRNG.decode(reader, position);
            case 84: return mgb.serialization.fbs.param.CheckNonFinite.decode(reader, position);
            case 85: return mgb.serialization.fbs.param.LayerNorm.decode(reader, position);
            case 86: return mgb.serialization.fbs.param.Dropout.decode(reader, position);
            case 87: return mgb.serialization.fbs.param.RNNCell.decode(reader, position);
            case 88: return mgb.serialization.fbs.param.RNN.decode(reader, position);
            case 89: return mgb.serialization.fbs.param.LSTM.decode(reader, position);
            case 90: return mgb.serialization.fbs.param.Softmax.decode(reader, position);
            case 91: return mgb.serialization.fbs.param.Diag.decode(reader, position);
            case 92: return mgb.serialization.fbs.param.GroupNorm.decode(reader, position);
            case 93: return mgb.serialization.fbs.param.Fill.decode(reader, position);
            default: return undefined;
        }
    }
};

mgb.serialization.fbs.v2.Operator = class Operator {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.Operator();
        $.type = reader.string_(position, 4, null);
        $.type_id = reader.uint64_(position, 6, 0n);
        $.name = reader.string_(position, 8, null);
        $.param = reader.union(position, 10, mgb.serialization.fbs.v2.OperatorParam);
        $.additional_params = reader.unions(position, 14, mgb.serialization.fbs.v2.OperatorParam);
        $.inputs = reader.array(position, 18, Uint32Array);
        $.outputs = reader.array(position, 20, Uint32Array);
        $.comp_node = reader.tables(position, 22, mgb.serialization.fbs.v2.CompNode);
        $.output_dtype = reader.table(position, 24, mgb.serialization.fbs.DType);
        $.tensors = reader.tables(position, 26, mgb.serialization.fbs.v2.Tensor);
        $.opr_version = reader.uint32_(position, 28, 0);
        $.priority = reader.int32_(position, 30, 0);
        $.custom_data = reader.tables(position, 32, mgb.serialization.fbs.v2.Blob);
        return $;
    }
};

mgb.serialization.fbs.v2.Metadata = class Metadata {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.Metadata();
        $.is_valid = reader.bool_(position, 4, false);
        $.graph_modified = reader.bool_(position, 6, false);
        $.optimize_options = reader.uint64_(position, 8, 0n);
        $.user_info = reader.string_(position, 10, null);
        return $;
    }
};

mgb.serialization.fbs.v2.MiddleTensor = class MiddleTensor {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.MiddleTensor();
        $.name = reader.string_(position, 4, null);
        $.shape = reader.array(position, 6, Uint32Array);
        $.comp_node = reader.table(position, 8, mgb.serialization.fbs.v2.CompNode);
        $.dtype = reader.table(position, 10, mgb.serialization.fbs.DType);
        $.format = reader.union(position, 12, mgb.serialization.fbs.v2.TensorFormat);
        return $;
    }
};

mgb.serialization.fbs.v2.OutputVar = class OutputVar {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.OutputVar();
        $.compact_id = reader.uint32_(position, 4, 0);
        $.original_id = reader.uint32_(position, 6, 0);
        return $;
    }
};

mgb.serialization.fbs.v2.OutputAlias = class OutputAlias {

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.OutputAlias();
        $.id = reader.uint32_(position, 4, 0);
        $.name = reader.string_(position, 6, null);
        return $;
    }
};

mgb.serialization.fbs.v2.Model = class Model {

    static identifier(reader) {
        return reader.identifier === 'mge2';
    }

    static create(reader) {
        return mgb.serialization.fbs.v2.Model.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new mgb.serialization.fbs.v2.Model();
        $.mge_version = reader.uint32_(position, 4, 0);
        $.model_version = reader.uint32_(position, 6, 0);
        $.oprs = reader.tables(position, 8, mgb.serialization.fbs.v2.Operator);
        $.middle_tensors = reader.tables(position, 10, mgb.serialization.fbs.v2.MiddleTensor);
        $.output_vars_idx = reader.tables(position, 12, mgb.serialization.fbs.v2.OutputVar);
        $.output_alias = reader.tables(position, 14, mgb.serialization.fbs.v2.OutputAlias);
        $.nr_shared_tensor = reader.uint32_(position, 16, 0);
        $.metadata = reader.table(position, 18, mgb.serialization.fbs.v2.Metadata);
        return $;
    }
};
