
export const MNN = {};

MNN.NetSource = {
    CAFFE: 0,
    TENSORFLOW: 1,
    TFLITE: 2,
    ONNX: 3,
    TORCH: 4
};

MNN.DataType = {
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
    DT_HALF: 19,
    DT_RESOURCE: 20,
    DT_VARIANT: 21
};

MNN.MNN_DATA_FORMAT = {
    NCHW: 0,
    NHWC: 1,
    NC4HW4: 2,
    NHWC4: 3,
    UNKNOWN: 4
};

MNN.Blob = class Blob {

    static decode(reader, position) {
        const $ = new MNN.Blob();
        $.dims = reader.array(position, 4, Int32Array);
        $.dataFormat = reader.int8_(position, 6, 0);
        $.dataType = reader.int32_(position, 8, 1);
        $.uint8s = reader.array(position, 10, Uint8Array);
        $.int8s = reader.array(position, 12, Int8Array);
        $.int32s = reader.array(position, 14, Int32Array);
        $.int64s = reader.int64s_(position, 16);
        $.float32s = reader.array(position, 18, Float32Array);
        $.strings = reader.strings_(position, 20);
        $.external = reader.int64s_(position, 22);
        return $;
    }
};

MNN.ListValue = class ListValue {

    static decode(reader, position) {
        const $ = new MNN.ListValue();
        $.s = reader.strings_(position, 4);
        $.i = reader.array(position, 6, Int32Array);
        $.f = reader.array(position, 8, Float32Array);
        $.b = reader.bools_(position, 10);
        $.type = reader.array(position, 12, Int32Array);
        return $;
    }
};

MNN.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new MNN.Attribute();
        $.s = reader.string_(position, 4, null);
        $.i = reader.int32_(position, 6, 0);
        $.b = reader.bool_(position, 8, false);
        $.key = reader.string_(position, 10, null);
        $.type = reader.int32_(position, 12, 0);
        $.f = reader.float32_(position, 14, 0);
        $.tensor = reader.table(position, 16, MNN.Blob);
        $.list = reader.table(position, 18, MNN.ListValue);
        $.func = reader.table(position, 20, MNN.NamedAttrList);
        return $;
    }
};

MNN.NamedAttrList = class NamedAttrList {

    static decode(reader, position) {
        const $ = new MNN.NamedAttrList();
        $.name = reader.string_(position, 4, null);
        $.attr = reader.tables(position, 6, MNN.Attribute);
        return $;
    }
};

MNN.PadMode = {
    CAFFE: 0,
    VALID: 1,
    SAME: 2
};

MNN.Convolution2DCommon = class Convolution2DCommon {

    static decode(reader, position) {
        const $ = new MNN.Convolution2DCommon();
        $.padX = reader.int32_(position, 4, 0);
        $.padY = reader.int32_(position, 6, 0);
        $.kernelX = reader.int32_(position, 8, 1);
        $.kernelY = reader.int32_(position, 10, 1);
        $.strideX = reader.int32_(position, 12, 1);
        $.strideY = reader.int32_(position, 14, 1);
        $.dilateX = reader.int32_(position, 16, 1);
        $.dilateY = reader.int32_(position, 18, 1);
        $.padMode = reader.int8_(position, 20, 0);
        $.group = reader.int32_(position, 22, 1);
        $.outputCount = reader.int32_(position, 24, 0);
        $.inputCount = reader.int32_(position, 26, 0);
        $.relu = reader.bool_(position, 28, false);
        $.relu6 = reader.bool_(position, 30, false);
        $.pads = reader.array(position, 32, Int32Array);
        $.outPads = reader.array(position, 34, Int32Array);
        $.hasOutputShape = reader.bool_(position, 36, false);
        return $;
    }
};

MNN.Convolution3DCommon = class Convolution3DCommon {

    static decode(reader, position) {
        const $ = new MNN.Convolution3DCommon();
        $.dilates = reader.array(position, 4, Int32Array);
        $.strides = reader.array(position, 6, Int32Array);
        $.kernels = reader.array(position, 8, Int32Array);
        $.pads = reader.array(position, 10, Int32Array);
        $.padMode = reader.int8_(position, 12, 0);
        $.inputCount = reader.int32_(position, 14, 0);
        $.outputCount = reader.int32_(position, 16, 0);
        $.relu = reader.bool_(position, 18, false);
        $.relu6 = reader.bool_(position, 20, false);
        $.group = reader.int32_(position, 22, 1);
        $.outPads = reader.array(position, 24, Int32Array);
        $.hasOutputShape = reader.bool_(position, 26, false);
        return $;
    }
};

MNN.SparseAlgo = {
    RANDOM: 0,
    SIMD_OC: 1
};

MNN.SparseCommon = class SparseCommon {

    static decode(reader, position) {
        const $ = new MNN.SparseCommon();
        $.method = reader.int8_(position, 4, 0);
        $.args = reader.tables(position, 6, MNN.Attribute);
        return $;
    }
};

MNN.IDSTQuan = class IDSTQuan {

    static decode(reader, position) {
        const $ = new MNN.IDSTQuan();
        $.buffer = reader.array(position, 4, Int8Array);
        $.alpha = reader.array(position, 6, Float32Array);
        $.type = reader.int32_(position, 8, 0);
        $.useInt32 = reader.bool_(position, 10, false);
        $.quantScale = reader.float32_(position, 12, 0);
        $.scaleIn = reader.float32_(position, 14, 0);
        $.scaleOut = reader.float32_(position, 16, 0);
        $.aMax = reader.int32_(position, 18, 0);
        $.aMin = reader.int32_(position, 20, 0);
        $.readType = reader.int32_(position, 22, 0);
        $.has_scaleInt = reader.bool_(position, 24, false);
        $.shapeInt32 = reader.bool_(position, 26, false);
        $.weightSize = reader.uint32_(position, 28, 0);
        $.index = reader.array(position, 30, Uint32Array);
        return $;
    }
};

MNN.QuantizeAlgo = {
    DEFAULT: 0,
    OVERFLOW_AWARE: 1,
    WINOGRAD_AWARE: 2
};

MNN.QuantizedFloatParam = class QuantizedFloatParam {

    static decode(reader, position) {
        const $ = new MNN.QuantizedFloatParam();
        $.weight = reader.array(position, 4, Int8Array);
        $.bias = reader.array(position, 6, Int32Array);
        $.scale = reader.array(position, 8, Float32Array);
        $.tensorScale = reader.array(position, 10, Float32Array);
        $.method = reader.int8_(position, 12, 0);
        $.nbits = reader.int32_(position, 14, 8);
        $.zeroPoint = reader.int8_(position, 16, 0);
        $.outputZeroPoint = reader.int8_(position, 18, 0);
        $.clampMin = reader.int8_(position, 20, -128);
        $.clampMax = reader.int8_(position, 22, 127);
        $.winogradAttr = reader.array(position, 24, Int32Array);
        $.outputDataType = reader.int32_(position, 26, 6);
        return $;
    }
};

MNN.Convolution2D = class Convolution2D {

    static decode(reader, position) {
        const $ = new MNN.Convolution2D();
        $.common = reader.table(position, 4, MNN.Convolution2DCommon);
        $.weight = reader.array(position, 6, Float32Array);
        $.bias = reader.array(position, 8, Float32Array);
        $.quanParameter = reader.table(position, 10, MNN.IDSTQuan);
        $.symmetricQuan = reader.table(position, 12, MNN.QuantizedFloatParam);
        $.sparseParameter = reader.table(position, 14, MNN.SparseCommon);
        $.external = reader.int64s_(position, 16);
        return $;
    }
};

MNN.Convolution3D = class Convolution3D {

    static decode(reader, position) {
        const $ = new MNN.Convolution3D();
        $.common = reader.table(position, 4, MNN.Convolution3DCommon);
        $.weight = reader.array(position, 6, Float32Array);
        $.bias = reader.array(position, 8, Float32Array);
        $.external = reader.int64s_(position, 10);
        return $;
    }
};

MNN.InnerProduct = class InnerProduct {

    static decode(reader, position) {
        const $ = new MNN.InnerProduct();
        $.outputCount = reader.int32_(position, 4, 0);
        $.biasTerm = reader.int32_(position, 6, 0);
        $.weightSize = reader.int32_(position, 8, 0);
        $.weight = reader.array(position, 10, Float32Array);
        $.bias = reader.array(position, 12, Float32Array);
        $.axis = reader.int32_(position, 14, 0);
        $.transpose = reader.bool_(position, 16, false);
        $.quanParameter = reader.table(position, 18, MNN.IDSTQuan);
        return $;
    }
};

MNN.PoolType = {
    MAXPOOL: 0,
    AVEPOOL: 1
};

MNN.PoolPadType = {
    CAFFE: 0,
    VALID: 1,
    SAME: 2
};

MNN.AvgPoolCountType = {
    DEFAULT: 0,
    INCLUDE_PADDING: 1,
    EXCLUDE_PADDING: 2
};

MNN.Pool = class Pool {

    static decode(reader, position) {
        const $ = new MNN.Pool();
        $.padX = reader.int32_(position, 4, 0);
        $.padY = reader.int32_(position, 6, 0);
        $.isGlobal = reader.bool_(position, 8, false);
        $.kernelX = reader.int32_(position, 10, 0);
        $.kernelY = reader.int32_(position, 12, 0);
        $.strideX = reader.int32_(position, 14, 0);
        $.strideY = reader.int32_(position, 16, 0);
        $.type = reader.int8_(position, 18, 0);
        $.padType = reader.int8_(position, 20, 0);
        $.dataType = reader.int32_(position, 22, 1);
        $.ceilModel = reader.bool_(position, 24, true);
        $.pads = reader.array(position, 26, Int32Array);
        $.countType = reader.int8_(position, 28, 0);
        return $;
    }
};

MNN.Pool3D = class Pool3D {

    static decode(reader, position) {
        const $ = new MNN.Pool3D();
        $.strides = reader.array(position, 4, Int32Array);
        $.kernels = reader.array(position, 6, Int32Array);
        $.pads = reader.array(position, 8, Int32Array);
        $.type = reader.int8_(position, 10, 0);
        $.padType = reader.int8_(position, 12, 0);
        $.isGlobal = reader.bool_(position, 14, false);
        return $;
    }
};

MNN.Relu = class Relu {

    static decode(reader, position) {
        const $ = new MNN.Relu();
        $.slope = reader.float32_(position, 4, 0);
        return $;
    }
};

MNN.Relu6 = class Relu6 {

    static decode(reader, position) {
        const $ = new MNN.Relu6();
        $.minValue = reader.float32_(position, 4, 0);
        $.maxValue = reader.float32_(position, 6, 6);
        return $;
    }
};

MNN.PRelu = class PRelu {

    static decode(reader, position) {
        const $ = new MNN.PRelu();
        $.slopeCount = reader.int32_(position, 4, 0);
        $.slope = reader.array(position, 6, Float32Array);
        return $;
    }
};

MNN.ELU = class ELU {

    static decode(reader, position) {
        const $ = new MNN.ELU();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }
};

MNN.LRN = class LRN {

    static decode(reader, position) {
        const $ = new MNN.LRN();
        $.regionType = reader.int32_(position, 4, 0);
        $.localSize = reader.int32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        $.bias = reader.float32_(position, 12, 1);
        return $;
    }
};

MNN.ArgMax = class ArgMax {

    static decode(reader, position) {
        const $ = new MNN.ArgMax();
        $.outMaxVal = reader.int32_(position, 4, 0);
        $.topK = reader.int32_(position, 6, 0);
        $.axis = reader.int32_(position, 8, 0);
        $.softmaxThreshold = reader.int32_(position, 10, 0);
        return $;
    }
};

MNN.Axis = class Axis {

    static decode(reader, position) {
        const $ = new MNN.Axis();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.Input = class Input {

    static decode(reader, position) {
        const $ = new MNN.Input();
        $.dims = reader.array(position, 4, Int32Array);
        $.dtype = reader.int32_(position, 6, 1);
        $.dformat = reader.int8_(position, 8, 2);
        return $;
    }
};

MNN.LSTM = class LSTM {

    static decode(reader, position) {
        const $ = new MNN.LSTM();
        $.outputCount = reader.int32_(position, 4, 0);
        $.weightSize = reader.int32_(position, 6, 0);
        $.clippingThreshold = reader.float32_(position, 8, 0);
        $.weightI = reader.table(position, 10, MNN.Blob);
        $.weightH = reader.table(position, 12, MNN.Blob);
        $.bias = reader.table(position, 14, MNN.Blob);
        $.weightIQ = reader.table(position, 16, MNN.Blob);
        $.weightIA = reader.table(position, 18, MNN.Blob);
        $.quantScale = reader.float32_(position, 20, 0);
        return $;
    }
};

MNN.Slice = class Slice {

    static decode(reader, position) {
        const $ = new MNN.Slice();
        $.axis = reader.int32_(position, 4, 0);
        $.slicePoints = reader.array(position, 6, Int32Array);
        $.sourceType = reader.int8_(position, 8, 0);
        return $;
    }
};

MNN.BatchNorm = class BatchNorm {

    static decode(reader, position) {
        const $ = new MNN.BatchNorm();
        $.channels = reader.int32_(position, 4, 0);
        $.slopeData = reader.array(position, 6, Float32Array);
        $.meanData = reader.array(position, 8, Float32Array);
        $.varData = reader.array(position, 10, Float32Array);
        $.biasData = reader.array(position, 12, Float32Array);
        $.Adata = reader.array(position, 14, Float32Array);
        $.Bdata = reader.array(position, 16, Float32Array);
        $.epsilon = reader.float32_(position, 18, 0.001);
        return $;
    }
};

MNN.Scale = class Scale {

    static decode(reader, position) {
        const $ = new MNN.Scale();
        $.channels = reader.int32_(position, 4, 0);
        $.scaleData = reader.array(position, 6, Float32Array);
        $.biasData = reader.array(position, 8, Float32Array);
        $.external = reader.int64s_(position, 10);
        return $;
    }
};

MNN.EltwiseType = {
    PROD: 0,
    SUM: 1,
    MAXIMUM: 2,
    SUB: 3
};

MNN.Eltwise = class Eltwise {

    static decode(reader, position) {
        const $ = new MNN.Eltwise();
        $.type = reader.int8_(position, 4, 0);
        $.coeff = reader.array(position, 6, Float32Array);
        return $;
    }
};

MNN.Flatten = class Flatten {

    static decode(reader, position) {
        const $ = new MNN.Flatten();
        $.axis = reader.int32_(position, 4, 0);
        $.endAxis = reader.int32_(position, 6, 0);
        return $;
    }
};

MNN.Permute = class Permute {

    static decode(reader, position) {
        const $ = new MNN.Permute();
        $.dims = reader.array(position, 4, Int32Array);
        return $;
    }
};

MNN.Reshape = class Reshape {

    static decode(reader, position) {
        const $ = new MNN.Reshape();
        $.dims = reader.array(position, 4, Int32Array);
        $.dimType = reader.int8_(position, 6, 0);
        return $;
    }
};

MNN.DetectionOutput = class DetectionOutput {

    static decode(reader, position) {
        const $ = new MNN.DetectionOutput();
        $.classCount = reader.int32_(position, 4, 0);
        $.nmsThresholdold = reader.float32_(position, 6, 0);
        $.nmsTopK = reader.int32_(position, 8, 0);
        $.keepTopK = reader.int32_(position, 10, 0);
        $.confidenceThreshold = reader.float32_(position, 12, 0);
        $.shareLocation = reader.int32_(position, 14, 0);
        $.backgroundLable = reader.int32_(position, 16, 0);
        $.varianceEncodedTarget = reader.int32_(position, 18, 0);
        $.codeType = reader.int32_(position, 20, 0);
        $.objectnessScore = reader.float32_(position, 22, 0.01);
        return $;
    }
};

MNN.RoiParameters = class RoiParameters {

    static decode(reader, position) {
        const $ = new MNN.RoiParameters();
        $.pooledWidth = reader.int32_(position, 4, 0);
        $.pooledHeight = reader.int32_(position, 6, 0);
        $.spatialScale = reader.float32_(position, 8, 0);
        $.samplingRatio = reader.int32_(position, 10, -1);
        $.aligned = reader.bool_(position, 12, false);
        $.poolType = reader.int8_(position, 14, 1);
        $.outputGrad = reader.bool_(position, 16, false);
        return $;
    }
};

MNN.Proposal = class Proposal {

    static decode(reader, position) {
        const $ = new MNN.Proposal();
        $.featStride = reader.int32_(position, 4, 0);
        $.baseSize = reader.int32_(position, 6, 0);
        $.preNmsTopN = reader.int32_(position, 8, 0);
        $.afterNmsTopN = reader.int32_(position, 10, 0);
        $.nmsThreshold = reader.float32_(position, 12, 0);
        $.minSize = reader.int32_(position, 14, 0);
        $.ratios = reader.table(position, 16, MNN.Blob);
        $.scales = reader.table(position, 18, MNN.Blob);
        $.anchors = reader.table(position, 20, MNN.Blob);
        return $;
    }
};

MNN.CoordinateTransformationMode = {
    NotSet: 0,
    AlignCorners: 1,
    HalfPixels: 2,
    PytorchHalfPixels: 3,
    Asymmetric: 4,
    TensorflowHalfPixels: 5,
    TensorflowCropAndResize: 6
};

MNN.Interp = class Interp {

    static decode(reader, position) {
        const $ = new MNN.Interp();
        $.widthScale = reader.float32_(position, 4, 0);
        $.heightScale = reader.float32_(position, 6, 0);
        $.outputWidth = reader.int32_(position, 8, 0);
        $.outputHeight = reader.int32_(position, 10, 0);
        $.resizeType = reader.int32_(position, 12, 0);
        $.alignCorners = reader.bool_(position, 14, false);
        $.halfPixelCenters = reader.bool_(position, 16, false);
        $.widthOffset = reader.float32_(position, 18, 0);
        $.heightOffset = reader.float32_(position, 20, 0);
        $.cubicCoeffA = reader.float32_(position, 22, -0.75);
        $.ctm = reader.int8_(position, 24, 0);
        $.depthScale = reader.float32_(position, 26, 0);
        $.outputDepth = reader.int32_(position, 28, 0);
        $.depthOffset = reader.float32_(position, 30, 0);
        return $;
    }
};

MNN.Resize = class Resize {

    static decode(reader, position) {
        const $ = new MNN.Resize();
        $.xScale = reader.float32_(position, 4, 0);
        $.yScale = reader.float32_(position, 6, 0);
        return $;
    }
};

MNN.PriorBox = class PriorBox {

    static decode(reader, position) {
        const $ = new MNN.PriorBox();
        $.minSizes = reader.array(position, 4, Float32Array);
        $.maxSizes = reader.array(position, 6, Float32Array);
        $.aspectRatios = reader.array(position, 8, Float32Array);
        $.variances = reader.array(position, 10, Float32Array);
        $.flip = reader.bool_(position, 12, false);
        $.clip = reader.bool_(position, 14, false);
        $.imageWidth = reader.int32_(position, 16, 0);
        $.imageHeight = reader.int32_(position, 18, 0);
        $.stepWidth = reader.int32_(position, 20, 0);
        $.stepHeight = reader.int32_(position, 22, 0);
        $.offset = reader.float32_(position, 24, 0);
        return $;
    }
};

MNN.Normalize = class Normalize {

    static decode(reader, position) {
        const $ = new MNN.Normalize();
        $.acrossSpatial = reader.int32_(position, 4, 0);
        $.channelShared = reader.int32_(position, 6, 0);
        $.eps = reader.float32_(position, 8, 0);
        $.scale = reader.array(position, 10, Float32Array);
        return $;
    }
};

MNN.EltwiseInt8 = class EltwiseInt8 {

    static decode(reader, position) {
        const $ = new MNN.EltwiseInt8();
        $.type = reader.int8_(position, 4, 0);
        $.inputQuan0 = reader.table(position, 6, MNN.QuantizedFloatParam);
        $.inputQuan1 = reader.table(position, 8, MNN.QuantizedFloatParam);
        $.outputQuan = reader.table(position, 10, MNN.QuantizedFloatParam);
        return $;
    }
};

MNN.CumSum = class CumSum {

    static decode(reader, position) {
        const $ = new MNN.CumSum();
        $.exclusive = reader.bool_(position, 4, false);
        $.reverse = reader.bool_(position, 6, false);
        return $;
    }
};

MNN.BinaryOpOperation = {
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
    RIGHTSHIFT: 28
};

MNN.BinaryOp = class BinaryOp {

    static decode(reader, position) {
        const $ = new MNN.BinaryOp();
        $.opType = reader.int32_(position, 4, 0);
        $.T = reader.int32_(position, 6, 1);
        $.activationType = reader.int32_(position, 8, 0);
        return $;
    }
};

MNN.PackParam = class PackParam {

    static decode(reader, position) {
        const $ = new MNN.PackParam();
        $.dataType = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }
};

MNN.StridedSliceParam = class StridedSliceParam {

    static decode(reader, position) {
        const $ = new MNN.StridedSliceParam();
        $.Index = reader.int32_(position, 4, 0);
        $.T = reader.int32_(position, 6, 0);
        $.beginMask = reader.int32_(position, 8, 0);
        $.endMask = reader.int32_(position, 10, 0);
        $.ellipsisMask = reader.int32_(position, 12, 0);
        $.newAxisMask = reader.int32_(position, 14, 0);
        $.shrinkAxisMask = reader.int32_(position, 16, 0);
        $.fromType = reader.int32_(position, 18, 0);
        return $;
    }
};

MNN.SqueezeParam = class SqueezeParam {

    static decode(reader, position) {
        const $ = new MNN.SqueezeParam();
        $.squeezeDims = reader.array(position, 4, Int32Array);
        return $;
    }
};

MNN.CastParam = class CastParam {

    static decode(reader, position) {
        const $ = new MNN.CastParam();
        $.srcT = reader.int32_(position, 4, 0);
        $.dstT = reader.int32_(position, 6, 0);
        return $;
    }
};

MNN.ReductionType = {
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

MNN.ReductionParam = class ReductionParam {

    static decode(reader, position) {
        const $ = new MNN.ReductionParam();
        $.operation = reader.int8_(position, 4, 0);
        $.dim = reader.array(position, 6, Int32Array);
        $.coeff = reader.float32_(position, 8, 0);
        $.keepDims = reader.bool_(position, 10, false);
        $.dType = reader.int32_(position, 12, 1);
        return $;
    }
};

MNN.Gather = class Gather {

    static decode(reader, position) {
        const $ = new MNN.Gather();
        $.Tindices = reader.int32_(position, 4, 0);
        $.Tparams = reader.int32_(position, 6, 0);
        $.validateIndices = reader.bool_(position, 8, false);
        $.axis = reader.int32_(position, 10, 0);
        return $;
    }
};

MNN.ExpandDims = class ExpandDims {

    static decode(reader, position) {
        const $ = new MNN.ExpandDims();
        $.T = reader.int32_(position, 4, 0);
        $.Tdim = reader.int32_(position, 6, 0);
        $.axis = reader.int32_(position, 8, 0);
        return $;
    }
};

MNN.Selu = class Selu {

    static decode(reader, position) {
        const $ = new MNN.Selu();
        $.scale = reader.float32_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        return $;
    }
};

MNN.AsString = class AsString {

    static decode(reader, position) {
        const $ = new MNN.AsString();
        $.T = reader.int32_(position, 4, 0);
        $.precision = reader.int32_(position, 6, 0);
        $.scientific = reader.bool_(position, 8, false);
        $.shortest = reader.bool_(position, 10, false);
        $.width = reader.int32_(position, 12, 0);
        $.fillString = reader.string_(position, 14, null);
        return $;
    }
};

MNN.ReduceJoin = class ReduceJoin {

    static decode(reader, position) {
        const $ = new MNN.ReduceJoin();
        $.keepDims = reader.bool_(position, 4, false);
        $.separator = reader.string_(position, 6, null);
        return $;
    }
};

MNN.UnaryOpOperation = {
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
    GELU_STANDARD: 33
};

MNN.UnaryOp = class UnaryOp {

    static decode(reader, position) {
        const $ = new MNN.UnaryOp();
        $.opType = reader.int32_(position, 4, 0);
        $.T = reader.int32_(position, 6, 0);
        $.tableInt8 = reader.array(position, 8, Int8Array);
        return $;
    }
};

MNN.TopKV2 = class TopKV2 {

    static decode(reader, position) {
        const $ = new MNN.TopKV2();
        $.T = reader.int32_(position, 4, 1);
        $.sorted = reader.bool_(position, 6, false);
        $.largest = reader.bool_(position, 8, true);
        return $;
    }
};

MNN.CropAndResizeMethod = {
    BILINEAR: 0,
    NEAREST: 1
};

MNN.CropAndResize = class CropAndResize {

    static decode(reader, position) {
        const $ = new MNN.CropAndResize();
        $.extrapolationValue = reader.float32_(position, 4, 0);
        $.method = reader.int8_(position, 6, 0);
        return $;
    }
};

MNN.Fill = class Fill {

    static decode(/* reader, position */) {
        const $ = new MNN.Fill();
        return $;
    }
};

MNN.GatherV2 = class GatherV2 {

    static decode(reader, position) {
        const $ = new MNN.GatherV2();
        $.Taxis = reader.int32_(position, 4, 0);
        $.Tindices = reader.int32_(position, 6, 0);
        $.Tparams = reader.int32_(position, 8, 0);
        return $;
    }
};

MNN.NonMaxSuppressionV2 = class NonMaxSuppressionV2 {

    static decode(/* reader, position */) {
        const $ = new MNN.NonMaxSuppressionV2();
        return $;
    }
};

MNN.Range = class Range {

    static decode(reader, position) {
        const $ = new MNN.Range();
        $.Tidx = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.Rank = class Rank {

    static decode(/* reader, position */) {
        const $ = new MNN.Rank();
        return $;
    }
};

MNN.Size = class Size {

    static decode(reader, position) {
        const $ = new MNN.Size();
        $.outputDataType = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.Transpose = class Transpose {

    static decode(reader, position) {
        const $ = new MNN.Transpose();
        $.Tperm = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.SliceTf = class SliceTf {

    static decode(reader, position) {
        const $ = new MNN.SliceTf();
        $.T = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.QuantizeMaxMin = class QuantizeMaxMin {

    static decode(reader, position) {
        const $ = new MNN.QuantizeMaxMin();
        $.T = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.Crop = class Crop {

    static decode(reader, position) {
        const $ = new MNN.Crop();
        $.axis = reader.int32_(position, 4, 2);
        $.offset = reader.array(position, 6, Int32Array);
        return $;
    }
};

MNN.SpaceBatch = class SpaceBatch {

    static decode(reader, position) {
        const $ = new MNN.SpaceBatch();
        $.blockShape = reader.table(position, 4, MNN.Blob);
        $.padding = reader.table(position, 6, MNN.Blob);
        return $;
    }
};

MNN.MatMul = class MatMul {

    static decode(reader, position) {
        const $ = new MNN.MatMul();
        $.T = reader.int32_(position, 4, 0);
        $.transposeA = reader.bool_(position, 6, false);
        $.transposeB = reader.bool_(position, 8, false);
        $.weight = reader.array(position, 10, Float32Array);
        $.bias = reader.array(position, 12, Float32Array);
        return $;
    }
};

MNN.MomentsParam = class MomentsParam {

    static decode(reader, position) {
        const $ = new MNN.MomentsParam();
        $.dim = reader.array(position, 4, Int32Array);
        $.keepDims = reader.bool_(position, 6, true);
        $.dType = reader.int32_(position, 8, 1);
        return $;
    }
};

MNN.RNNParam = class RNNParam {

    static decode(reader, position) {
        const $ = new MNN.RNNParam();
        $.numUnits = reader.int32_(position, 4, 0);
        $.isBidirectionalRNN = reader.bool_(position, 6, false);
        $.linearBeforeReset = reader.bool_(position, 8, false);
        $.keepAllOutputs = reader.bool_(position, 10, false);
        $.fwGateWeight = reader.table(position, 12, MNN.Blob);
        $.fwGateBias = reader.table(position, 14, MNN.Blob);
        $.fwCandidateWeight = reader.table(position, 16, MNN.Blob);
        $.fwCandidateBias = reader.table(position, 18, MNN.Blob);
        $.fwRecurrentBias = reader.table(position, 20, MNN.Blob);
        $.bwGateWeight = reader.table(position, 22, MNN.Blob);
        $.bwGateBias = reader.table(position, 24, MNN.Blob);
        $.bwCandidateWeight = reader.table(position, 26, MNN.Blob);
        $.bwCandidateBias = reader.table(position, 28, MNN.Blob);
        $.bwRecurrentBias = reader.table(position, 30, MNN.Blob);
        return $;
    }
};

MNN.BatchMatMulParam = class BatchMatMulParam {

    static decode(reader, position) {
        const $ = new MNN.BatchMatMulParam();
        $.adjX = reader.bool_(position, 4, false);
        $.adjY = reader.bool_(position, 6, false);
        return $;
    }
};

MNN.DepthToSpaceMode = {
    DCR: 0,
    CRD: 1
};

MNN.DepthSpaceParam = class DepthSpaceParam {

    static decode(reader, position) {
        const $ = new MNN.DepthSpaceParam();
        $.blockSize = reader.int32_(position, 4, 0);
        $.mode = reader.int8_(position, 6, 0);
        return $;
    }
};

MNN.ReverseSequenceParam = class ReverseSequenceParam {

    static decode(reader, position) {
        const $ = new MNN.ReverseSequenceParam();
        $.batchDim = reader.int32_(position, 4, 0);
        $.seqDim = reader.int32_(position, 6, 0);
        return $;
    }
};

MNN.DetectionPostProcessParam = class DetectionPostProcessParam {

    static decode(reader, position) {
        const $ = new MNN.DetectionPostProcessParam();
        $.maxDetections = reader.int32_(position, 4, 0);
        $.maxClassesPerDetection = reader.int32_(position, 6, 0);
        $.detectionsPerClass = reader.int32_(position, 8, 0);
        $.nmsScoreThreshold = reader.float32_(position, 10, 0);
        $.iouThreshold = reader.float32_(position, 12, 0);
        $.numClasses = reader.int32_(position, 14, 0);
        $.useRegularNMS = reader.bool_(position, 16, false);
        $.centerSizeEncoding = reader.array(position, 18, Float32Array);
        return $;
    }
};

MNN.OneHotParam = class OneHotParam {

    static decode(reader, position) {
        const $ = new MNN.OneHotParam();
        $.dType = reader.int32_(position, 4, 1);
        $.axis = reader.int32_(position, 6, -1);
        return $;
    }
};

MNN.PadValueMode = {
    CONSTANT: 0,
    REFLECT: 1,
    SYMMETRIC: 2,
    EDGE: 3
};

MNN.PadParam = class PadParam {

    static decode(reader, position) {
        const $ = new MNN.PadParam();
        $.mode = reader.int8_(position, 4, 0);
        return $;
    }
};

MNN.LayerNorm = class LayerNorm {

    static decode(reader, position) {
        const $ = new MNN.LayerNorm();
        $.axis = reader.array(position, 4, Int32Array);
        $.epsilon = reader.float32_(position, 6, 0);
        $.gamma = reader.array(position, 8, Float32Array);
        $.beta = reader.array(position, 10, Float32Array);
        $.group = reader.int32_(position, 12, 1);
        $.external = reader.int64s_(position, 14);
        $.useRMSNorm = reader.bool_(position, 16, false);
        return $;
    }
};

MNN.GroupNorm = class GroupNorm {

    static decode(reader, position) {
        const $ = new MNN.GroupNorm();
        $.axis = reader.int32_(position, 4, 0);
        $.epsilon = reader.float32_(position, 6, 0);
        $.gamma = reader.array(position, 8, Float32Array);
        $.beta = reader.array(position, 10, Float32Array);
        $.group = reader.int32_(position, 12, 1);
        $.bSwish = reader.int32_(position, 14, 0);
        $.external = reader.int64s_(position, 16);
        return $;
    }
};

MNN.RandomUniform = class RandomUniform {

    static decode(reader, position) {
        const $ = new MNN.RandomUniform();
        $.seed = reader.int32_(position, 4, 0);
        $.seed2 = reader.int32_(position, 6, 0);
        $.type = reader.int32_(position, 8, 1);
        $.low = reader.float32_(position, 10, 0);
        $.high = reader.float32_(position, 12, 1);
        return $;
    }
};

MNN.TensorArray = class TensorArray {

    static decode(reader, position) {
        const $ = new MNN.TensorArray();
        $.dynamic_size = reader.bool_(position, 4, false);
        $.identical_element_shapes = reader.bool_(position, 6, false);
        $.element_shape = reader.array(position, 8, Int32Array);
        $.T = reader.int32_(position, 10, 1);
        $.axis = reader.int32_(position, 12, 0);
        $.keepdims = reader.bool_(position, 14, true);
        $.new_axis = reader.bool_(position, 16, false);
        return $;
    }
};

MNN.LSTMBlockCell = class LSTMBlockCell {

    static decode(reader, position) {
        const $ = new MNN.LSTMBlockCell();
        $.cell_clip = reader.float32_(position, 4, 3);
        $.forget_bias = reader.float32_(position, 6, 1);
        $.use_peephole = reader.bool_(position, 8, false);
        return $;
    }
};

MNN.FusedActivation = {
    kTfLiteActNone: 0,
    kTfLiteActRelu: 1,
    kTfLiteActRelu1: 2,
    kTfLiteActRelu6: 3,
    kTfLiteActTanh: 4,
    kTfLiteActSignBit: 5,
    kTfLiteActSigmoid: 6
};

MNN.QuantizedParam = class QuantizedParam {

    static decode(reader, position) {
        const $ = new MNN.QuantizedParam();
        $.zeroPoint = reader.int32_(position, 4, 0);
        $.scale = reader.float32_(position, 6, 0);
        return $;
    }
};

MNN.QuantizedAdd = class QuantizedAdd {

    static decode(reader, position) {
        const $ = new MNN.QuantizedAdd();
        $.activationType = reader.int8_(position, 4, 0);
        $.input1QuantizedParam = reader.table(position, 6, MNN.QuantizedParam);
        $.input2QuantizedParam = reader.table(position, 8, MNN.QuantizedParam);
        $.outputQuantizedParam = reader.table(position, 10, MNN.QuantizedParam);
        return $;
    }
};

MNN.ModeFormat = {
    TENSORFLOW: 0,
    TFLITE: 1
};

MNN.QuantizeMode = {
    MIN_COMBINED: 0,
    MIN_FIRST: 1,
    SCALED: 2
};

MNN.Dequantize = class Dequantize {

    static decode(reader, position) {
        const $ = new MNN.Dequantize();
        $.inputQuantizedParam = reader.table(position, 4, MNN.QuantizedParam);
        $.mode = reader.int8_(position, 6, 0);
        $.modelFormat = reader.int8_(position, 8, 0);
        $.type = reader.int32_(position, 10, 0);
        return $;
    }
};

MNN.QuantizedAvgPool = class QuantizedAvgPool {

    static decode(reader, position) {
        const $ = new MNN.QuantizedAvgPool();
        $.kernelX = reader.int32_(position, 4, 0);
        $.kernelY = reader.int32_(position, 6, 0);
        $.modelFormat = reader.int8_(position, 8, 0);
        $.outputActivationMax = reader.int32_(position, 10, 0);
        $.outputActivationMin = reader.int32_(position, 12, 0);
        $.padType = reader.int8_(position, 14, 0);
        $.padX = reader.int32_(position, 16, 0);
        $.padY = reader.int32_(position, 18, 0);
        $.strideX = reader.int32_(position, 20, 0);
        $.strideY = reader.int32_(position, 22, 0);
        $.type = reader.int32_(position, 24, 0);
        return $;
    }
};

MNN.QuantizedBiasAdd = class QuantizedBiasAdd {

    static decode(reader, position) {
        const $ = new MNN.QuantizedBiasAdd();
        $.bias = reader.array(position, 4, Int32Array);
        $.inputType = reader.int32_(position, 6, 0);
        $.max = reader.int32_(position, 8, 0);
        $.min = reader.int32_(position, 10, 0);
        $.outputType = reader.int32_(position, 12, 0);
        return $;
    }
};

MNN.QuantizedConcat = class QuantizedConcat {

    static decode(reader, position) {
        const $ = new MNN.QuantizedConcat();
        $.activationType = reader.int8_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        $.inputScale = reader.array(position, 8, Float32Array);
        $.inputZeroPoint = reader.array(position, 10, Int32Array);
        $.outputQuantizedParam = reader.table(position, 12, MNN.QuantizedParam);
        return $;
    }
};

MNN.QuantizedLogistic = class QuantizedLogistic {

    static decode(reader, position) {
        const $ = new MNN.QuantizedLogistic();
        $.inputQuantizedParam = reader.table(position, 4, MNN.QuantizedParam);
        $.outputQuantizedParam = reader.table(position, 6, MNN.QuantizedParam);
        return $;
    }
};

MNN.QuantizedMatMul = class QuantizedMatMul {

    static decode(reader, position) {
        const $ = new MNN.QuantizedMatMul();
        $.transposeA = reader.bool_(position, 4, false);
        $.transposeB = reader.bool_(position, 6, false);
        return $;
    }
};

MNN.QuantizedMaxPool = class QuantizedMaxPool {

    static decode(reader, position) {
        const $ = new MNN.QuantizedMaxPool();
        $.kernelX = reader.int32_(position, 4, 0);
        $.kernelY = reader.int32_(position, 6, 0);
        $.modelFormat = reader.int8_(position, 8, 0);
        $.outputActivationMax = reader.int32_(position, 10, 0);
        $.outputActivationMin = reader.int32_(position, 12, 0);
        $.padType = reader.int8_(position, 14, 0);
        $.padX = reader.int32_(position, 16, 0);
        $.padY = reader.int32_(position, 18, 0);
        $.strideX = reader.int32_(position, 20, 0);
        $.strideY = reader.int32_(position, 22, 0);
        $.type = reader.int32_(position, 24, 0);
        return $;
    }
};

MNN.QuantizedRelu = class QuantizedRelu {

    static decode(reader, position) {
        const $ = new MNN.QuantizedRelu();
        $.type = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.QuantizedRelu6 = class QuantizedRelu6 {

    static decode(reader, position) {
        const $ = new MNN.QuantizedRelu6();
        $.type = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.QuantizedReshape = class QuantizedReshape {

    static decode(reader, position) {
        const $ = new MNN.QuantizedReshape();
        $.dims = reader.array(position, 4, Int32Array);
        $.modelFormat = reader.int8_(position, 6, 0);
        return $;
    }
};

MNN.QuantizedSoftmax = class QuantizedSoftmax {

    static decode(reader, position) {
        const $ = new MNN.QuantizedSoftmax();
        $.beta = reader.float32_(position, 4, 0);
        $.inputScale = reader.float32_(position, 6, 0);
        return $;
    }
};

MNN.QuantizeRoundMode = {
    HALF_AWAY_FROM_ZERO: 0,
    HALF_TO_EVEN: 1
};

MNN.QuantizeV2 = class QuantizeV2 {

    static decode(reader, position) {
        const $ = new MNN.QuantizeV2();
        $.type = reader.int32_(position, 4, 0);
        $.mode = reader.int8_(position, 6, 0);
        $.roundMode = reader.int8_(position, 8, 0);
        return $;
    }
};

MNN.RequantizationRange = class RequantizationRange {

    static decode(/* reader, position */) {
        const $ = new MNN.RequantizationRange();
        return $;
    }
};

MNN.Requantize = class Requantize {

    static decode(/* reader, position */) {
        const $ = new MNN.Requantize();
        return $;
    }
};

MNN.TfQuantizedConv2D = class TfQuantizedConv2D {

    static decode(reader, position) {
        const $ = new MNN.TfQuantizedConv2D();
        $.bias = reader.array(position, 4, Int32Array);
        $.biasflag = reader.bool_(position, 6, false);
        $.common = reader.table(position, 8, MNN.Convolution2DCommon);
        $.weight = reader.array(position, 10, Uint8Array);
        $.activationType = reader.int8_(position, 12, 0);
        $.multiplier = reader.int32_(position, 14, 0);
        $.outMax = reader.int32_(position, 16, 0);
        $.outMin = reader.int32_(position, 18, 0);
        $.shift = reader.int32_(position, 20, 0);
        $.biasQuantizedParam = reader.table(position, 22, MNN.QuantizedParam);
        $.depthMultiplier = reader.int32_(position, 24, 0);
        $.filterQuantizedParam = reader.table(position, 26, MNN.QuantizedParam);
        $.inputQuantizedParam = reader.table(position, 28, MNN.QuantizedParam);
        $.modelFormat = reader.int8_(position, 30, 0);
        $.outputQuantizedParam = reader.table(position, 32, MNN.QuantizedParam);
        return $;
    }
};

MNN.ExtraInfo = class ExtraInfo {

    static decode(reader, position) {
        const $ = new MNN.ExtraInfo();
        $.buffer = reader.array(position, 4, Int8Array);
        $.name = reader.string_(position, 6, null);
        $.version = reader.string_(position, 8, null);
        return $;
    }
};

MNN.TensorConvertInfo = class TensorConvertInfo {

    static decode(reader, position) {
        const $ = new MNN.TensorConvertInfo();
        $.source = reader.int8_(position, 4, 0);
        $.dest = reader.int8_(position, 6, 0);
        return $;
    }
};

MNN.SampleMode = {
    BILINEAR: 0,
    NEAREST: 1
};

MNN.BorderMode = {
    ZEROS: 0,
    CLAMP: 1,
    REFLECTION: 2,
    CUBE: 3
};

MNN.GridSample = class GridSample {

    static decode(reader, position) {
        const $ = new MNN.GridSample();
        $.mode = reader.int8_(position, 4, 0);
        $.paddingMode = reader.int8_(position, 6, 0);
        $.alignCorners = reader.bool_(position, 8, false);
        $.backward = reader.bool_(position, 10, false);
        return $;
    }
};

MNN.ImageFormatType = {
    RGBA: 0,
    RGB: 1,
    BGR: 2,
    GRAY: 3,
    BGRA: 4,
    YCrCb: 5,
    YUV: 6,
    HSV: 7,
    XYZ: 8,
    BGR555: 9,
    BGR565: 10,
    YUV_NV21: 11,
    YUV_NV12: 12,
    YUV_I420: 13,
    HSV_FULL: 14
};

MNN.FilterType = {
    NEAREST: 0,
    BILINEAR: 1,
    BICUBIC: 2
};

MNN.WrapType = {
    CLAMP_TO_EDGE: 0,
    ZERO: 1,
    REPEAT: 2
};

MNN.ImageProcessParam = class ImageProcessParam {

    static decode(reader, position) {
        const $ = new MNN.ImageProcessParam();
        $.filterType = reader.int8_(position, 4, 0);
        $.sourceFormat = reader.int32_(position, 6, 0);
        $.destFormat = reader.int32_(position, 8, 0);
        $.wrap = reader.int8_(position, 10, 0);
        $.mean = reader.array(position, 12, Float32Array);
        $.normal = reader.array(position, 14, Float32Array);
        $.transform = reader.array(position, 16, Float32Array);
        $.paddingValue = reader.int8_(position, 18, 0);
        $.shape = reader.array(position, 20, Int32Array);
        $.outputType = reader.int32_(position, 22, 0);
        $.draw = reader.bool_(position, 24, false);
        return $;
    }
};

MNN.OpType = {
    AbsVal: 0,
    QuantizedAdd: 1,
    ArgMax: 2,
    AsString: 3,
    InstanceNorm: 4,
    BatchToSpaceND: 5,
    Copy: 6,
    BinaryOp: 7,
    Bnll: 8,
    Cast: 9,
    Concat: 10,
    Const: 11,
    Convolution: 12,
    ConvolutionDepthwise: 13,
    Crop: 14,
    CropAndResize: 15,
    ImageProcess: 16,
    Deconvolution: 17,
    DeconvolutionDepthwise: 18,
    Dequantize: 19,
    DetectionOutput: 20,
    Dropout: 21,
    Eltwise: 22,
    ELU: 23,
    Unique: 24,
    Exp: 25,
    ExpandDims: 26,
    Fill: 27,
    Flatten: 28,
    Im2Col: 29,
    Gather: 30,
    GatherV2: 31,
    Im2Seq: 32,
    InnerProduct: 33,
    Input: 34,
    Interp: 35,
    Log: 36,
    LRN: 37,
    LSTM: 38,
    MatMul: 39,
    MVN: 40,
    NonMaxSuppression: 41,
    NonMaxSuppressionV2: 42,
    Normalize: 43,
    Pack: 44,
    Padding: 45,
    Permute: 46,
    Pooling: 47,
    Power: 48,
    PReLU: 49,
    PriorBox: 50,
    Proposal: 51,
    QuantizedAvgPool: 52,
    QuantizedBiasAdd: 53,
    QuantizedConcat: 54,
    QuantizedDepthwiseConv2D: 55,
    QuantizedLogistic: 56,
    RasterAndInterpolate: 57,
    QuantizedMaxPool: 58,
    Texture: 59,
    RasterDiff: 60,
    QuantizedReshape: 61,
    QuantizedSoftmax: 62,
    QuantizeMaxMin: 63,
    QuantizeV2: 64,
    Range: 65,
    Rank: 66,
    ReduceJoin: 67,
    Reduction: 68,
    ReLU: 69,
    ReLU6: 70,
    RequantizationRange: 71,
    Requantize: 72,
    Reshape: 73,
    Resize: 74,
    RNN: 75,
    ROIPooling: 76,
    Scale: 77,
    Selu: 78,
    Seq2Out: 79,
    Shape: 80,
    Sigmoid: 81,
    Size: 82,
    Slice: 83,
    SliceTf: 84,
    Softmax: 85,
    SpaceToBatchND: 86,
    SpatialProduct: 87,
    Col2Im: 88,
    Segment: 89,
    Squeeze: 90,
    StridedSlice: 91,
    StringJoin: 92,
    StringSplit: 93,
    StringToNumber: 94,
    TanH: 95,
    TfQuantizedConv2D: 96,
    Threshold: 97,
    Tile: 98,
    TopKV2: 99,
    Transpose: 100,
    UnaryOp: 101,
    Unpack: 102,
    Where: 103,
    Moments: 104,
    RNNSequenceGRU: 105,
    BatchMatMul: 106,
    Unsqueeze: 107,
    CosineSimilarity: 108,
    DepthToSpace: 109,
    SpaceToDepth: 110,
    ReverseSequence: 111,
    Pooling3D: 112,
    Convolution3D: 113,
    MatrixBandPart: 114,
    GatherND: 115,
    DetectionPostProcess: 116,
    UnravelIndex: 117,
    ScatterNd: 118,
    OneHot: 119,
    BroadcastTo: 120,
    Dilation2D: 121,
    Interp3D: 122,
    Raster: 128,
    ConvertTensor: 129,
    ArgMin: 130,
    LinSpace: 131,
    RandomUniform: 132,
    TensorArray: 133,
    TensorArraySize: 134,
    TensorArrayRead: 135,
    TensorArrayWrite: 136,
    TensorArrayGather: 137,
    TensorArrayScatter: 138,
    TensorArraySplit: 139,
    TensorArrayConcat: 140,
    LSTMBlockCell: 141,
    Reverse: 142,
    ROIAlign: 143,
    RandomNormal: 144,
    TensorArrayInsert: 145,
    TensorArrayErase: 146,
    EyeLike: 147,
    CumSum: 148,
    Det: 149,
    CumProd: 150,
    ScatterElements: 151,
    GatherElements: 152,
    Svd: 153,
    Histogram: 154,
    DynamicQuant: 155,
    Plugin: 256,
    Select: 257,
    ZerosLike: 258,
    Broastcast: 259,
    SetDiff1D: 260,
    ReluGrad: 261,
    Identity: 262,
    PoolGrad: 263,
    SoftmaxGrad: 264,
    Conv2DBackPropFilter: 265,
    TrainableParam: 266,
    BatchNorm: 267,
    ConvTranspose3D: 268,
    ZeroGrad: 269,
    FmhaV2: 300,
    Fmhca: 301,
    SeqLen2Spatial: 302,
    SplitGeLU: 303,
    GroupNorm: 304,
    Extra: 512,
    ConvInt8: 513,
    Int8ToFloat: 514,
    DepthwiseConvInt8: 515,
    PoolInt8: 516,
    FloatToInt8: 517,
    EltwiseInt8: 518,
    While: 600,
    If: 601,
    LayerNorm: 603,
    GridSample: 604
};

MNN.Plugin = class Plugin {

    static decode(reader, position) {
        const $ = new MNN.Plugin();
        $.type = reader.string_(position, 4, null);
        $.attr = reader.tables(position, 6, MNN.Attribute);
        return $;
    }
};

MNN.Extra = class Extra {

    static decode(reader, position) {
        const $ = new MNN.Extra();
        $.type = reader.string_(position, 4, null);
        $.engine = reader.string_(position, 6, null);
        $.info = reader.array(position, 8, Int8Array);
        $.attr = reader.tables(position, 10, MNN.Attribute);
        $.vector = reader.bool_(position, 12, false);
        return $;
    }
};

MNN.StringVec = class StringVec {

    static decode(reader, position) {
        const $ = new MNN.StringVec();
        $.data = reader.strings_(position, 4);
        return $;
    }
};

MNN.FmhaV2Param = class FmhaV2Param {

    static decode(reader, position) {
        const $ = new MNN.FmhaV2Param();
        $.heads = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.FmhcaParam = class FmhcaParam {

    static decode(reader, position) {
        const $ = new MNN.FmhcaParam();
        $.heads = reader.int32_(position, 4, 0);
        return $;
    }
};

MNN.WhileParam = class WhileParam {

    static decode(reader, position) {
        const $ = new MNN.WhileParam();
        $.cond_graph = reader.string_(position, 4, null);
        $.body_graph = reader.string_(position, 6, null);
        $.aliases_inputs = reader.tables(position, 8, MNN.StringVec);
        $.aliases_outputs = reader.strings_(position, 10);
        $.aliases_updates = reader.tables(position, 12, MNN.StringVec);
        return $;
    }
};

MNN.IfParam = class IfParam {

    static decode(reader, position) {
        const $ = new MNN.IfParam();
        $.then_graph = reader.string_(position, 4, null);
        $.else_graph = reader.string_(position, 6, null);
        $.aliases_inputs = reader.tables(position, 8, MNN.StringVec);
        $.aliases_outputs = reader.tables(position, 10, MNN.StringVec);
        return $;
    }
};

MNN.RegionCommand = class RegionCommand {

    static decode(reader, position) {
        const $ = new MNN.RegionCommand();
        $.op = reader.table(position, 4, MNN.Op);
        $.steps = reader.array(position, 6, Int32Array);
        $.size = reader.array(position, 8, Int32Array);
        $.indexes = reader.array(position, 10, Int32Array);
        $.view = reader.tables(position, 12, MNN.View);
        $.fuse = reader.int32_(position, 14, -1);
        $.iterIndexes = reader.array(position, 16, Int32Array);
        return $;
    }
};

MNN.LoopParam = class LoopParam {

    static decode(reader, position) {
        const $ = new MNN.LoopParam();
        $.tensorNumber = reader.int32_(position, 4, 0);
        $.outputIndexes = reader.array(position, 6, Int32Array);
        $.inputIndexes = reader.array(position, 8, Int32Array);
        $.extraTensorInfos = reader.tables(position, 10, MNN.TensorDescribe);
        $.parallel = reader.bool_(position, 12, true);
        $.loopNumber = reader.int32_(position, 14, 0);
        $.commands = reader.tables(position, 16, MNN.RegionCommand);
        $.initCommand = reader.tables(position, 18, MNN.RegionCommand);
        return $;
    }
};

MNN.OpParameter = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return MNN.QuantizedAdd.decode(reader, position);
            case 2: return MNN.ArgMax.decode(reader, position);
            case 3: return MNN.AsString.decode(reader, position);
            case 4: return MNN.Axis.decode(reader, position);
            case 5: return MNN.BatchNorm.decode(reader, position);
            case 6: return MNN.BinaryOp.decode(reader, position);
            case 7: return MNN.Blob.decode(reader, position);
            case 8: return MNN.CastParam.decode(reader, position);
            case 9: return MNN.Convolution2D.decode(reader, position);
            case 10: return MNN.Crop.decode(reader, position);
            case 11: return MNN.CropAndResize.decode(reader, position);
            case 12: return MNN.Dequantize.decode(reader, position);
            case 13: return MNN.DetectionOutput.decode(reader, position);
            case 14: return MNN.Eltwise.decode(reader, position);
            case 15: return MNN.ExpandDims.decode(reader, position);
            case 16: return MNN.Fill.decode(reader, position);
            case 17: return MNN.Flatten.decode(reader, position);
            case 18: return MNN.Gather.decode(reader, position);
            case 19: return MNN.GatherV2.decode(reader, position);
            case 20: return MNN.InnerProduct.decode(reader, position);
            case 21: return MNN.Input.decode(reader, position);
            case 22: return MNN.Interp.decode(reader, position);
            case 23: return MNN.LRN.decode(reader, position);
            case 24: return MNN.LSTM.decode(reader, position);
            case 25: return MNN.MatMul.decode(reader, position);
            case 26: return MNN.NonMaxSuppressionV2.decode(reader, position);
            case 27: return MNN.Normalize.decode(reader, position);
            case 28: return MNN.PackParam.decode(reader, position);
            case 29: return MNN.Permute.decode(reader, position);
            case 30: return MNN.Plugin.decode(reader, position);
            case 31: return MNN.Pool.decode(reader, position);
            case 32: return MNN.PRelu.decode(reader, position);
            case 33: return MNN.PriorBox.decode(reader, position);
            case 34: return MNN.Proposal.decode(reader, position);
            case 35: return MNN.QuantizedAvgPool.decode(reader, position);
            case 36: return MNN.QuantizedBiasAdd.decode(reader, position);
            case 37: return MNN.QuantizedConcat.decode(reader, position);
            case 38: return MNN.QuantizedLogistic.decode(reader, position);
            case 39: return MNN.QuantizedMatMul.decode(reader, position);
            case 40: return MNN.QuantizedMaxPool.decode(reader, position);
            case 41: return MNN.QuantizedRelu.decode(reader, position);
            case 42: return MNN.QuantizedRelu6.decode(reader, position);
            case 43: return MNN.QuantizedReshape.decode(reader, position);
            case 44: return MNN.QuantizedSoftmax.decode(reader, position);
            case 45: return MNN.QuantizeMaxMin.decode(reader, position);
            case 46: return MNN.QuantizeV2.decode(reader, position);
            case 47: return MNN.Range.decode(reader, position);
            case 48: return MNN.Rank.decode(reader, position);
            case 49: return MNN.ReduceJoin.decode(reader, position);
            case 50: return MNN.ReductionParam.decode(reader, position);
            case 51: return MNN.Relu.decode(reader, position);
            case 52: return MNN.Relu6.decode(reader, position);
            case 53: return MNN.RequantizationRange.decode(reader, position);
            case 54: return MNN.Requantize.decode(reader, position);
            case 55: return MNN.Reshape.decode(reader, position);
            case 56: return MNN.Resize.decode(reader, position);
            case 57: return MNN.RoiParameters.decode(reader, position);
            case 58: return MNN.Scale.decode(reader, position);
            case 59: return MNN.Selu.decode(reader, position);
            case 60: return MNN.Size.decode(reader, position);
            case 61: return MNN.Slice.decode(reader, position);
            case 62: return MNN.SliceTf.decode(reader, position);
            case 63: return MNN.SpaceBatch.decode(reader, position);
            case 64: return MNN.SqueezeParam.decode(reader, position);
            case 65: return MNN.StridedSliceParam.decode(reader, position);
            case 66: return MNN.TensorConvertInfo.decode(reader, position);
            case 67: return MNN.TfQuantizedConv2D.decode(reader, position);
            case 68: return MNN.TopKV2.decode(reader, position);
            case 69: return MNN.Transpose.decode(reader, position);
            case 70: return MNN.UnaryOp.decode(reader, position);
            case 71: return MNN.MomentsParam.decode(reader, position);
            case 72: return MNN.RNNParam.decode(reader, position);
            case 73: return MNN.BatchMatMulParam.decode(reader, position);
            case 74: return MNN.QuantizedFloatParam.decode(reader, position);
            case 75: return MNN.DepthSpaceParam.decode(reader, position);
            case 76: return MNN.EltwiseInt8.decode(reader, position);
            case 77: return MNN.ReverseSequenceParam.decode(reader, position);
            case 78: return MNN.Extra.decode(reader, position);
            case 79: return MNN.Pool3D.decode(reader, position);
            case 80: return MNN.Convolution3D.decode(reader, position);
            case 81: return MNN.ELU.decode(reader, position);
            case 82: return MNN.DetectionPostProcessParam.decode(reader, position);
            case 83: return MNN.OneHotParam.decode(reader, position);
            case 84: return MNN.PadParam.decode(reader, position);
            case 85: return MNN.WhileParam.decode(reader, position);
            case 86: return MNN.IfParam.decode(reader, position);
            case 87: return MNN.RandomUniform.decode(reader, position);
            case 88: return MNN.LayerNorm.decode(reader, position);
            case 89: return MNN.TensorArray.decode(reader, position);
            case 90: return MNN.LSTMBlockCell.decode(reader, position);
            case 91: return MNN.GridSample.decode(reader, position);
            case 92: return MNN.LoopParam.decode(reader, position);
            case 93: return MNN.ImageProcessParam.decode(reader, position);
            case 94: return MNN.CumSum.decode(reader, position);
            case 95: return MNN.GroupNorm.decode(reader, position);
            case 96: return MNN.FmhaV2Param.decode(reader, position);
            case 97: return MNN.FmhcaParam.decode(reader, position);
            default: return undefined;
        }
    }
};

MNN.Op = class Op {

    static decode(reader, position) {
        const $ = new MNN.Op();
        $.inputIndexes = reader.array(position, 4, Int32Array);
        $.main = reader.union(position, 6, MNN.OpParameter);
        $.name = reader.string_(position, 10, null);
        $.outputIndexes = reader.array(position, 12, Int32Array);
        $.type = reader.int32_(position, 14, 0);
        $.defaultDimentionFormat = reader.int8_(position, 16, 1);
        $.externalPath = reader.string_(position, 18, null);
        return $;
    }
};

MNN.View = class View {

    static decode(reader, position) {
        const $ = new MNN.View();
        $.offset = reader.int32_(position, 4, 0);
        $.stride = reader.array(position, 6, Int32Array);
        return $;
    }
};

MNN.Region = class Region {

    static decode(reader, position) {
        const $ = new MNN.Region();
        $.src = reader.table(position, 4, MNN.View);
        $.dst = reader.table(position, 6, MNN.View);
        $.size = reader.array(position, 8, Int32Array);
        $.origin = reader.int32_(position, 10, 0);
        return $;
    }
};

MNN.TensorDescribe = class TensorDescribe {

    static decode(reader, position) {
        const $ = new MNN.TensorDescribe();
        $.blob = reader.table(position, 4, MNN.Blob);
        $.index = reader.int32_(position, 6, 0);
        $.name = reader.string_(position, 8, null);
        $.regions = reader.tables(position, 10, MNN.Region);
        $.quantInfo = reader.table(position, 12, MNN.TensorQuantInfo);
        return $;
    }
};

MNN.ForwardType = {
    CPU: 0,
    METAL: 1,
    OPENCL: 2,
    OPENGLES: 3,
    VULKAN: 4
};

MNN.Usage = {
    INFERENCE: 0,
    TRAIN: 1,
    INFERENCE_STATIC: 2
};

MNN.SubGraphProto = class SubGraphProto {

    static decode(reader, position) {
        const $ = new MNN.SubGraphProto();
        $.name = reader.string_(position, 4, null);
        $.inputs = reader.array(position, 6, Int32Array);
        $.outputs = reader.array(position, 8, Int32Array);
        $.tensors = reader.strings_(position, 10);
        $.nodes = reader.tables(position, 12, MNN.Op);
        $.extraTensorDescribe = reader.tables(position, 14, MNN.TensorDescribe);
        return $;
    }
};

MNN.TensorQuantInfo = class TensorQuantInfo {

    static decode(reader, position) {
        const $ = new MNN.TensorQuantInfo();
        $.scale = reader.float32_(position, 4, 0);
        $.zero = reader.float32_(position, 6, 0);
        $.min = reader.float32_(position, 8, -128);
        $.max = reader.float32_(position, 10, 127);
        $.type = reader.int32_(position, 12, 0);
        return $;
    }
};

MNN.Net = class Net {

    static create(reader) {
        return MNN.Net.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new MNN.Net();
        $.bizCode = reader.string_(position, 4, null);
        $.extraTensorDescribe = reader.tables(position, 6, MNN.TensorDescribe);
        $.extraInfo = reader.table(position, 8, MNN.ExtraInfo);
        $.oplists = reader.tables(position, 10, MNN.Op);
        $.outputName = reader.strings_(position, 12);
        $.preferForwardType = reader.int8_(position, 14, 0);
        $.sourceType = reader.int8_(position, 16, 0);
        $.tensorName = reader.strings_(position, 18);
        $.tensorNumber = reader.int32_(position, 20, 0);
        $.usage = reader.int8_(position, 22, 0);
        $.subgraphs = reader.tables(position, 24, MNN.SubGraphProto);
        $.mnn_uuid = reader.string_(position, 26, null);
        return $;
    }
};
