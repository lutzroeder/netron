var $root = flatbuffers.get('mnn');

$root.MNN = $root.MNN || {};

$root.MNN.NetSource = {
    CAFFE: 0,
    TENSORFLOW: 1,
    TFLITE: 2,
    ONNX: 3,
    TORCH: 4
};

$root.MNN.DataType = {
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

$root.MNN.MNN_DATA_FORMAT = {
    NCHW: 0,
    NHWC: 1,
    NC4HW4: 2,
    NHWC4: 3,
    UNKNOWN: 4
};

$root.MNN.Blob = class Blob {

    static decode(reader, position) {
        const $ = new $root.MNN.Blob();
        $.dims = reader.typedArray(position, 4, Int32Array);
        $.dataFormat = reader.int8_(position, 6, 0);
        $.dataType = reader.int32_(position, 8, 1);
        $.uint8s = reader.typedArray(position, 10, Uint8Array);
        $.int8s = reader.typedArray(position, 12, Int8Array);
        $.int32s = reader.typedArray(position, 14, Int32Array);
        $.int64s = reader.int64s_(position, 16);
        $.float32s = reader.typedArray(position, 18, Float32Array);
        $.strings = reader.strings_(position, 20);
        $.external = reader.int64s_(position, 22);
        return $;
    }
};

$root.MNN.ListValue = class ListValue {

    static decode(reader, position) {
        const $ = new $root.MNN.ListValue();
        $.s = reader.strings_(position, 4);
        $.i = reader.typedArray(position, 6, Int32Array);
        $.f = reader.typedArray(position, 8, Float32Array);
        $.b = reader.bools_(position, 10);
        $.type = reader.typedArray(position, 12, Int32Array);
        return $;
    }
};

$root.MNN.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new $root.MNN.Attribute();
        $.s = reader.string_(position, 4, null);
        $.i = reader.int32_(position, 6, 0);
        $.b = reader.bool_(position, 8, false);
        $.key = reader.string_(position, 10, null);
        $.type = reader.int32_(position, 12, 0);
        $.f = reader.float32_(position, 14, 0);
        $.tensor = reader.table(position, 16, $root.MNN.Blob.decode);
        $.list = reader.table(position, 18, $root.MNN.ListValue.decode);
        $.func = reader.table(position, 20, $root.MNN.NamedAttrList.decode);
        return $;
    }
};

$root.MNN.NamedAttrList = class NamedAttrList {

    static decode(reader, position) {
        const $ = new $root.MNN.NamedAttrList();
        $.name = reader.string_(position, 4, null);
        $.attr = reader.tableArray(position, 6, $root.MNN.Attribute.decode);
        return $;
    }
};

$root.MNN.PadMode = {
    CAFFE: 0,
    VALID: 1,
    SAME: 2
};

$root.MNN.Convolution2DCommon = class Convolution2DCommon {

    static decode(reader, position) {
        const $ = new $root.MNN.Convolution2DCommon();
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
        $.pads = reader.typedArray(position, 32, Int32Array);
        $.outPads = reader.typedArray(position, 34, Int32Array);
        $.hasOutputShape = reader.bool_(position, 36, false);
        return $;
    }
};

$root.MNN.Convolution3DCommon = class Convolution3DCommon {

    static decode(reader, position) {
        const $ = new $root.MNN.Convolution3DCommon();
        $.dilates = reader.typedArray(position, 4, Int32Array);
        $.strides = reader.typedArray(position, 6, Int32Array);
        $.kernels = reader.typedArray(position, 8, Int32Array);
        $.pads = reader.typedArray(position, 10, Int32Array);
        $.padMode = reader.int8_(position, 12, 0);
        $.inputCount = reader.int32_(position, 14, 0);
        $.outputCount = reader.int32_(position, 16, 0);
        $.relu = reader.bool_(position, 18, false);
        $.relu6 = reader.bool_(position, 20, false);
        $.group = reader.int32_(position, 22, 1);
        $.outPads = reader.typedArray(position, 24, Int32Array);
        $.hasOutputShape = reader.bool_(position, 26, false);
        return $;
    }
};

$root.MNN.SparseAlgo = {
    RANDOM: 0,
    SIMD_OC: 1
};

$root.MNN.SparseCommon = class SparseCommon {

    static decode(reader, position) {
        const $ = new $root.MNN.SparseCommon();
        $.method = reader.int8_(position, 4, 0);
        $.args = reader.tableArray(position, 6, $root.MNN.Attribute.decode);
        return $;
    }
};

$root.MNN.IDSTQuan = class IDSTQuan {

    static decode(reader, position) {
        const $ = new $root.MNN.IDSTQuan();
        $.buffer = reader.typedArray(position, 4, Int8Array);
        $.alpha = reader.typedArray(position, 6, Float32Array);
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
        $.index = reader.typedArray(position, 30, Uint32Array);
        return $;
    }
};

$root.MNN.QuantizeAlgo = {
    DEFAULT: 0,
    OVERFLOW_AWARE: 1,
    WINOGRAD_AWARE: 2
};

$root.MNN.QuantizedFloatParam = class QuantizedFloatParam {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedFloatParam();
        $.weight = reader.typedArray(position, 4, Int8Array);
        $.bias = reader.typedArray(position, 6, Int32Array);
        $.scale = reader.typedArray(position, 8, Float32Array);
        $.tensorScale = reader.typedArray(position, 10, Float32Array);
        $.method = reader.int8_(position, 12, 0);
        $.nbits = reader.int32_(position, 14, 8);
        $.zeroPoint = reader.int8_(position, 16, 0);
        $.outputZeroPoint = reader.int8_(position, 18, 0);
        $.clampMin = reader.int8_(position, 20, -128);
        $.clampMax = reader.int8_(position, 22, 127);
        $.winogradAttr = reader.typedArray(position, 24, Int32Array);
        return $;
    }
};

$root.MNN.Convolution2D = class Convolution2D {

    static decode(reader, position) {
        const $ = new $root.MNN.Convolution2D();
        $.common = reader.table(position, 4, $root.MNN.Convolution2DCommon.decode);
        $.weight = reader.typedArray(position, 6, Float32Array);
        $.bias = reader.typedArray(position, 8, Float32Array);
        $.quanParameter = reader.table(position, 10, $root.MNN.IDSTQuan.decode);
        $.symmetricQuan = reader.table(position, 12, $root.MNN.QuantizedFloatParam.decode);
        $.sparseParameter = reader.table(position, 14, $root.MNN.SparseCommon.decode);
        $.external = reader.int64s_(position, 16);
        return $;
    }
};

$root.MNN.Convolution3D = class Convolution3D {

    static decode(reader, position) {
        const $ = new $root.MNN.Convolution3D();
        $.common = reader.table(position, 4, $root.MNN.Convolution3DCommon.decode);
        $.weight = reader.typedArray(position, 6, Float32Array);
        $.bias = reader.typedArray(position, 8, Float32Array);
        $.external = reader.int64s_(position, 10);
        return $;
    }
};

$root.MNN.InnerProduct = class InnerProduct {

    static decode(reader, position) {
        const $ = new $root.MNN.InnerProduct();
        $.outputCount = reader.int32_(position, 4, 0);
        $.biasTerm = reader.int32_(position, 6, 0);
        $.weightSize = reader.int32_(position, 8, 0);
        $.weight = reader.typedArray(position, 10, Float32Array);
        $.bias = reader.typedArray(position, 12, Float32Array);
        $.axis = reader.int32_(position, 14, 0);
        $.transpose = reader.bool_(position, 16, false);
        $.quanParameter = reader.table(position, 18, $root.MNN.IDSTQuan.decode);
        return $;
    }
};

$root.MNN.PoolType = {
    MAXPOOL: 0,
    AVEPOOL: 1
};

$root.MNN.PoolPadType = {
    CAFFE: 0,
    VALID: 1,
    SAME: 2
};

$root.MNN.AvgPoolCountType = {
    DEFAULT: 0,
    INCLUDE_PADDING: 1,
    EXCLUDE_PADDING: 2
};

$root.MNN.Pool = class Pool {

    static decode(reader, position) {
        const $ = new $root.MNN.Pool();
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
        $.pads = reader.typedArray(position, 26, Int32Array);
        $.countType = reader.int8_(position, 28, 0);
        return $;
    }
};

$root.MNN.Pool3D = class Pool3D {

    static decode(reader, position) {
        const $ = new $root.MNN.Pool3D();
        $.strides = reader.typedArray(position, 4, Int32Array);
        $.kernels = reader.typedArray(position, 6, Int32Array);
        $.pads = reader.typedArray(position, 8, Int32Array);
        $.type = reader.int8_(position, 10, 0);
        $.padType = reader.int8_(position, 12, 0);
        $.isGlobal = reader.bool_(position, 14, false);
        return $;
    }
};

$root.MNN.Relu = class Relu {

    static decode(reader, position) {
        const $ = new $root.MNN.Relu();
        $.slope = reader.float32_(position, 4, 0);
        return $;
    }
};

$root.MNN.Relu6 = class Relu6 {

    static decode(reader, position) {
        const $ = new $root.MNN.Relu6();
        $.minValue = reader.float32_(position, 4, 0);
        $.maxValue = reader.float32_(position, 6, 6);
        return $;
    }
};

$root.MNN.PRelu = class PRelu {

    static decode(reader, position) {
        const $ = new $root.MNN.PRelu();
        $.slopeCount = reader.int32_(position, 4, 0);
        $.slope = reader.typedArray(position, 6, Float32Array);
        return $;
    }
};

$root.MNN.ELU = class ELU {

    static decode(reader, position) {
        const $ = new $root.MNN.ELU();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }
};

$root.MNN.LRN = class LRN {

    static decode(reader, position) {
        const $ = new $root.MNN.LRN();
        $.regionType = reader.int32_(position, 4, 0);
        $.localSize = reader.int32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        $.bias = reader.float32_(position, 12, 1);
        return $;
    }
};

$root.MNN.ArgMax = class ArgMax {

    static decode(reader, position) {
        const $ = new $root.MNN.ArgMax();
        $.outMaxVal = reader.int32_(position, 4, 0);
        $.topK = reader.int32_(position, 6, 0);
        $.axis = reader.int32_(position, 8, 0);
        $.softmaxThreshold = reader.int32_(position, 10, 0);
        return $;
    }
};

$root.MNN.Axis = class Axis {

    static decode(reader, position) {
        const $ = new $root.MNN.Axis();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.Input = class Input {

    static decode(reader, position) {
        const $ = new $root.MNN.Input();
        $.dims = reader.typedArray(position, 4, Int32Array);
        $.dtype = reader.int32_(position, 6, 1);
        $.dformat = reader.int8_(position, 8, 2);
        return $;
    }
};

$root.MNN.LSTM = class LSTM {

    static decode(reader, position) {
        const $ = new $root.MNN.LSTM();
        $.outputCount = reader.int32_(position, 4, 0);
        $.weightSize = reader.int32_(position, 6, 0);
        $.clippingThreshold = reader.float32_(position, 8, 0);
        $.weightI = reader.table(position, 10, $root.MNN.Blob.decode);
        $.weightH = reader.table(position, 12, $root.MNN.Blob.decode);
        $.bias = reader.table(position, 14, $root.MNN.Blob.decode);
        $.weightIQ = reader.table(position, 16, $root.MNN.Blob.decode);
        $.weightIA = reader.table(position, 18, $root.MNN.Blob.decode);
        $.quantScale = reader.float32_(position, 20, 0);
        return $;
    }
};

$root.MNN.Slice = class Slice {

    static decode(reader, position) {
        const $ = new $root.MNN.Slice();
        $.axis = reader.int32_(position, 4, 0);
        $.slicePoints = reader.typedArray(position, 6, Int32Array);
        $.sourceType = reader.int8_(position, 8, 0);
        return $;
    }
};

$root.MNN.BatchNorm = class BatchNorm {

    static decode(reader, position) {
        const $ = new $root.MNN.BatchNorm();
        $.channels = reader.int32_(position, 4, 0);
        $.slopeData = reader.typedArray(position, 6, Float32Array);
        $.meanData = reader.typedArray(position, 8, Float32Array);
        $.varData = reader.typedArray(position, 10, Float32Array);
        $.biasData = reader.typedArray(position, 12, Float32Array);
        $.Adata = reader.typedArray(position, 14, Float32Array);
        $.Bdata = reader.typedArray(position, 16, Float32Array);
        $.epsilon = reader.float32_(position, 18, 0.001);
        return $;
    }
};

$root.MNN.Scale = class Scale {

    static decode(reader, position) {
        const $ = new $root.MNN.Scale();
        $.channels = reader.int32_(position, 4, 0);
        $.scaleData = reader.typedArray(position, 6, Float32Array);
        $.biasData = reader.typedArray(position, 8, Float32Array);
        $.external = reader.int64s_(position, 10);
        return $;
    }
};

$root.MNN.EltwiseType = {
    PROD: 0,
    SUM: 1,
    MAXIMUM: 2,
    SUB: 3
};

$root.MNN.Eltwise = class Eltwise {

    static decode(reader, position) {
        const $ = new $root.MNN.Eltwise();
        $.type = reader.int8_(position, 4, 0);
        $.coeff = reader.typedArray(position, 6, Float32Array);
        return $;
    }
};

$root.MNN.Flatten = class Flatten {

    static decode(reader, position) {
        const $ = new $root.MNN.Flatten();
        $.axis = reader.int32_(position, 4, 0);
        $.endAxis = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.MNN.Permute = class Permute {

    static decode(reader, position) {
        const $ = new $root.MNN.Permute();
        $.dims = reader.typedArray(position, 4, Int32Array);
        return $;
    }
};

$root.MNN.Reshape = class Reshape {

    static decode(reader, position) {
        const $ = new $root.MNN.Reshape();
        $.dims = reader.typedArray(position, 4, Int32Array);
        $.dimType = reader.int8_(position, 6, 0);
        return $;
    }
};

$root.MNN.DetectionOutput = class DetectionOutput {

    static decode(reader, position) {
        const $ = new $root.MNN.DetectionOutput();
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

$root.MNN.RoiParameters = class RoiParameters {

    static decode(reader, position) {
        const $ = new $root.MNN.RoiParameters();
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

$root.MNN.Proposal = class Proposal {

    static decode(reader, position) {
        const $ = new $root.MNN.Proposal();
        $.featStride = reader.int32_(position, 4, 0);
        $.baseSize = reader.int32_(position, 6, 0);
        $.preNmsTopN = reader.int32_(position, 8, 0);
        $.afterNmsTopN = reader.int32_(position, 10, 0);
        $.nmsThreshold = reader.float32_(position, 12, 0);
        $.minSize = reader.int32_(position, 14, 0);
        $.ratios = reader.table(position, 16, $root.MNN.Blob.decode);
        $.scales = reader.table(position, 18, $root.MNN.Blob.decode);
        $.anchors = reader.table(position, 20, $root.MNN.Blob.decode);
        return $;
    }
};

$root.MNN.CoordinateTransformationMode = {
    NotSet: 0,
    AlignCorners: 1,
    HalfPixels: 2,
    PytorchHalfPixels: 3,
    Asymmetric: 4,
    TensorflowHalfPixels: 5,
    TensorflowCropAndResize: 6
};

$root.MNN.Interp = class Interp {

    static decode(reader, position) {
        const $ = new $root.MNN.Interp();
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

$root.MNN.Resize = class Resize {

    static decode(reader, position) {
        const $ = new $root.MNN.Resize();
        $.xScale = reader.float32_(position, 4, 0);
        $.yScale = reader.float32_(position, 6, 0);
        return $;
    }
};

$root.MNN.PriorBox = class PriorBox {

    static decode(reader, position) {
        const $ = new $root.MNN.PriorBox();
        $.minSizes = reader.typedArray(position, 4, Float32Array);
        $.maxSizes = reader.typedArray(position, 6, Float32Array);
        $.aspectRatios = reader.typedArray(position, 8, Float32Array);
        $.variances = reader.typedArray(position, 10, Float32Array);
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

$root.MNN.Normalize = class Normalize {

    static decode(reader, position) {
        const $ = new $root.MNN.Normalize();
        $.acrossSpatial = reader.int32_(position, 4, 0);
        $.channelShared = reader.int32_(position, 6, 0);
        $.eps = reader.float32_(position, 8, 0);
        $.scale = reader.typedArray(position, 10, Float32Array);
        return $;
    }
};

$root.MNN.EltwiseInt8 = class EltwiseInt8 {

    static decode(reader, position) {
        const $ = new $root.MNN.EltwiseInt8();
        $.type = reader.int8_(position, 4, 0);
        $.inputQuan0 = reader.table(position, 6, $root.MNN.QuantizedFloatParam.decode);
        $.inputQuan1 = reader.table(position, 8, $root.MNN.QuantizedFloatParam.decode);
        $.outputQuan = reader.table(position, 10, $root.MNN.QuantizedFloatParam.decode);
        return $;
    }
};

$root.MNN.CumSum = class CumSum {

    static decode(reader, position) {
        const $ = new $root.MNN.CumSum();
        $.exclusive = reader.bool_(position, 4, false);
        $.reverse = reader.bool_(position, 6, false);
        return $;
    }
};

$root.MNN.BinaryOpOperation = {
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

$root.MNN.BinaryOp = class BinaryOp {

    static decode(reader, position) {
        const $ = new $root.MNN.BinaryOp();
        $.opType = reader.int32_(position, 4, 0);
        $.T = reader.int32_(position, 6, 1);
        $.activationType = reader.int32_(position, 8, 0);
        return $;
    }
};

$root.MNN.PackParam = class PackParam {

    static decode(reader, position) {
        const $ = new $root.MNN.PackParam();
        $.dataType = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.MNN.StridedSliceParam = class StridedSliceParam {

    static decode(reader, position) {
        const $ = new $root.MNN.StridedSliceParam();
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

$root.MNN.SqueezeParam = class SqueezeParam {

    static decode(reader, position) {
        const $ = new $root.MNN.SqueezeParam();
        $.squeezeDims = reader.typedArray(position, 4, Int32Array);
        return $;
    }
};

$root.MNN.CastParam = class CastParam {

    static decode(reader, position) {
        const $ = new $root.MNN.CastParam();
        $.srcT = reader.int32_(position, 4, 0);
        $.dstT = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.MNN.ReductionType = {
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

$root.MNN.ReductionParam = class ReductionParam {

    static decode(reader, position) {
        const $ = new $root.MNN.ReductionParam();
        $.operation = reader.int8_(position, 4, 0);
        $.dim = reader.typedArray(position, 6, Int32Array);
        $.coeff = reader.float32_(position, 8, 0);
        $.keepDims = reader.bool_(position, 10, false);
        $.dType = reader.int32_(position, 12, 1);
        return $;
    }
};

$root.MNN.Gather = class Gather {

    static decode(reader, position) {
        const $ = new $root.MNN.Gather();
        $.Tindices = reader.int32_(position, 4, 0);
        $.Tparams = reader.int32_(position, 6, 0);
        $.validateIndices = reader.bool_(position, 8, false);
        $.axis = reader.int32_(position, 10, 0);
        return $;
    }
};

$root.MNN.ExpandDims = class ExpandDims {

    static decode(reader, position) {
        const $ = new $root.MNN.ExpandDims();
        $.T = reader.int32_(position, 4, 0);
        $.Tdim = reader.int32_(position, 6, 0);
        $.axis = reader.int32_(position, 8, 0);
        return $;
    }
};

$root.MNN.Selu = class Selu {

    static decode(reader, position) {
        const $ = new $root.MNN.Selu();
        $.scale = reader.float32_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        return $;
    }
};

$root.MNN.AsString = class AsString {

    static decode(reader, position) {
        const $ = new $root.MNN.AsString();
        $.T = reader.int32_(position, 4, 0);
        $.precision = reader.int32_(position, 6, 0);
        $.scientific = reader.bool_(position, 8, false);
        $.shortest = reader.bool_(position, 10, false);
        $.width = reader.int32_(position, 12, 0);
        $.fillString = reader.string_(position, 14, null);
        return $;
    }
};

$root.MNN.ReduceJoin = class ReduceJoin {

    static decode(reader, position) {
        const $ = new $root.MNN.ReduceJoin();
        $.keepDims = reader.bool_(position, 4, false);
        $.separator = reader.string_(position, 6, null);
        return $;
    }
};

$root.MNN.UnaryOpOperation = {
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

$root.MNN.UnaryOp = class UnaryOp {

    static decode(reader, position) {
        const $ = new $root.MNN.UnaryOp();
        $.opType = reader.int32_(position, 4, 0);
        $.T = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.MNN.TopKV2 = class TopKV2 {

    static decode(reader, position) {
        const $ = new $root.MNN.TopKV2();
        $.T = reader.int32_(position, 4, 1);
        $.sorted = reader.bool_(position, 6, false);
        $.largest = reader.bool_(position, 8, true);
        return $;
    }
};

$root.MNN.CropAndResizeMethod = {
    BILINEAR: 0,
    NEAREST: 1
};

$root.MNN.CropAndResize = class CropAndResize {

    static decode(reader, position) {
        const $ = new $root.MNN.CropAndResize();
        $.extrapolationValue = reader.float32_(position, 4, 0);
        $.method = reader.int8_(position, 6, 0);
        return $;
    }
};

$root.MNN.Fill = class Fill {

    static decode(/* reader, position */) {
        const $ = new $root.MNN.Fill();
        return $;
    }
};

$root.MNN.GatherV2 = class GatherV2 {

    static decode(reader, position) {
        const $ = new $root.MNN.GatherV2();
        $.Taxis = reader.int32_(position, 4, 0);
        $.Tindices = reader.int32_(position, 6, 0);
        $.Tparams = reader.int32_(position, 8, 0);
        return $;
    }
};

$root.MNN.NonMaxSuppressionV2 = class NonMaxSuppressionV2 {

    static decode(/* reader, position */) {
        const $ = new $root.MNN.NonMaxSuppressionV2();
        return $;
    }
};

$root.MNN.Range = class Range {

    static decode(reader, position) {
        const $ = new $root.MNN.Range();
        $.Tidx = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.Rank = class Rank {

    static decode(/* reader, position */) {
        const $ = new $root.MNN.Rank();
        return $;
    }
};

$root.MNN.Size = class Size {

    static decode(reader, position) {
        const $ = new $root.MNN.Size();
        $.outputDataType = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.Transpose = class Transpose {

    static decode(reader, position) {
        const $ = new $root.MNN.Transpose();
        $.Tperm = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.SliceTf = class SliceTf {

    static decode(reader, position) {
        const $ = new $root.MNN.SliceTf();
        $.T = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.QuantizeMaxMin = class QuantizeMaxMin {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizeMaxMin();
        $.T = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.Crop = class Crop {

    static decode(reader, position) {
        const $ = new $root.MNN.Crop();
        $.axis = reader.int32_(position, 4, 2);
        $.offset = reader.typedArray(position, 6, Int32Array);
        return $;
    }
};

$root.MNN.SpaceBatch = class SpaceBatch {

    static decode(reader, position) {
        const $ = new $root.MNN.SpaceBatch();
        $.blockShape = reader.table(position, 4, $root.MNN.Blob.decode);
        $.padding = reader.table(position, 6, $root.MNN.Blob.decode);
        return $;
    }
};

$root.MNN.MatMul = class MatMul {

    static decode(reader, position) {
        const $ = new $root.MNN.MatMul();
        $.T = reader.int32_(position, 4, 0);
        $.transposeA = reader.bool_(position, 6, false);
        $.transposeB = reader.bool_(position, 8, false);
        $.weight = reader.typedArray(position, 10, Float32Array);
        $.bias = reader.typedArray(position, 12, Float32Array);
        return $;
    }
};

$root.MNN.MomentsParam = class MomentsParam {

    static decode(reader, position) {
        const $ = new $root.MNN.MomentsParam();
        $.dim = reader.typedArray(position, 4, Int32Array);
        $.keepDims = reader.bool_(position, 6, true);
        $.dType = reader.int32_(position, 8, 1);
        return $;
    }
};

$root.MNN.RNNParam = class RNNParam {

    static decode(reader, position) {
        const $ = new $root.MNN.RNNParam();
        $.numUnits = reader.int32_(position, 4, 0);
        $.isBidirectionalRNN = reader.bool_(position, 6, false);
        $.linearBeforeReset = reader.bool_(position, 8, false);
        $.keepAllOutputs = reader.bool_(position, 10, false);
        $.fwGateWeight = reader.table(position, 12, $root.MNN.Blob.decode);
        $.fwGateBias = reader.table(position, 14, $root.MNN.Blob.decode);
        $.fwCandidateWeight = reader.table(position, 16, $root.MNN.Blob.decode);
        $.fwCandidateBias = reader.table(position, 18, $root.MNN.Blob.decode);
        $.fwRecurrentBias = reader.table(position, 20, $root.MNN.Blob.decode);
        $.bwGateWeight = reader.table(position, 22, $root.MNN.Blob.decode);
        $.bwGateBias = reader.table(position, 24, $root.MNN.Blob.decode);
        $.bwCandidateWeight = reader.table(position, 26, $root.MNN.Blob.decode);
        $.bwCandidateBias = reader.table(position, 28, $root.MNN.Blob.decode);
        $.bwRecurrentBias = reader.table(position, 30, $root.MNN.Blob.decode);
        return $;
    }
};

$root.MNN.BatchMatMulParam = class BatchMatMulParam {

    static decode(reader, position) {
        const $ = new $root.MNN.BatchMatMulParam();
        $.adjX = reader.bool_(position, 4, false);
        $.adjY = reader.bool_(position, 6, false);
        return $;
    }
};

$root.MNN.DepthToSpaceMode = {
    DCR: 0,
    CRD: 1
};

$root.MNN.DepthSpaceParam = class DepthSpaceParam {

    static decode(reader, position) {
        const $ = new $root.MNN.DepthSpaceParam();
        $.blockSize = reader.int32_(position, 4, 0);
        $.mode = reader.int8_(position, 6, 0);
        return $;
    }
};

$root.MNN.ReverseSequenceParam = class ReverseSequenceParam {

    static decode(reader, position) {
        const $ = new $root.MNN.ReverseSequenceParam();
        $.batchDim = reader.int32_(position, 4, 0);
        $.seqDim = reader.int32_(position, 6, 0);
        return $;
    }
};

$root.MNN.DetectionPostProcessParam = class DetectionPostProcessParam {

    static decode(reader, position) {
        const $ = new $root.MNN.DetectionPostProcessParam();
        $.maxDetections = reader.int32_(position, 4, 0);
        $.maxClassesPerDetection = reader.int32_(position, 6, 0);
        $.detectionsPerClass = reader.int32_(position, 8, 0);
        $.nmsScoreThreshold = reader.float32_(position, 10, 0);
        $.iouThreshold = reader.float32_(position, 12, 0);
        $.numClasses = reader.int32_(position, 14, 0);
        $.useRegularNMS = reader.bool_(position, 16, false);
        $.centerSizeEncoding = reader.typedArray(position, 18, Float32Array);
        return $;
    }
};

$root.MNN.OneHotParam = class OneHotParam {

    static decode(reader, position) {
        const $ = new $root.MNN.OneHotParam();
        $.dType = reader.int32_(position, 4, 1);
        $.axis = reader.int32_(position, 6, -1);
        return $;
    }
};

$root.MNN.PadValueMode = {
    CONSTANT: 0,
    REFLECT: 1,
    SYMMETRIC: 2,
    EDGE: 3
};

$root.MNN.PadParam = class PadParam {

    static decode(reader, position) {
        const $ = new $root.MNN.PadParam();
        $.mode = reader.int8_(position, 4, 0);
        return $;
    }
};

$root.MNN.LayerNorm = class LayerNorm {

    static decode(reader, position) {
        const $ = new $root.MNN.LayerNorm();
        $.axis = reader.typedArray(position, 4, Int32Array);
        $.epsilon = reader.float32_(position, 6, 0);
        $.gamma = reader.typedArray(position, 8, Float32Array);
        $.beta = reader.typedArray(position, 10, Float32Array);
        $.group = reader.int32_(position, 12, 1);
        $.external = reader.int64s_(position, 14);
        return $;
    }
};

$root.MNN.RandomUniform = class RandomUniform {

    static decode(reader, position) {
        const $ = new $root.MNN.RandomUniform();
        $.seed = reader.int32_(position, 4, 0);
        $.seed2 = reader.int32_(position, 6, 0);
        $.type = reader.int32_(position, 8, 1);
        $.low = reader.float32_(position, 10, 0);
        $.high = reader.float32_(position, 12, 1);
        return $;
    }
};

$root.MNN.TensorArray = class TensorArray {

    static decode(reader, position) {
        const $ = new $root.MNN.TensorArray();
        $.dynamic_size = reader.bool_(position, 4, false);
        $.identical_element_shapes = reader.bool_(position, 6, false);
        $.element_shape = reader.typedArray(position, 8, Int32Array);
        $.T = reader.int32_(position, 10, 1);
        $.axis = reader.int32_(position, 12, 0);
        $.keepdims = reader.bool_(position, 14, true);
        $.new_axis = reader.bool_(position, 16, false);
        return $;
    }
};

$root.MNN.LSTMBlockCell = class LSTMBlockCell {

    static decode(reader, position) {
        const $ = new $root.MNN.LSTMBlockCell();
        $.cell_clip = reader.float32_(position, 4, 3);
        $.forget_bias = reader.float32_(position, 6, 1);
        $.use_peephole = reader.bool_(position, 8, false);
        return $;
    }
};

$root.MNN.FusedActivation = {
    kTfLiteActNone: 0,
    kTfLiteActRelu: 1,
    kTfLiteActRelu1: 2,
    kTfLiteActRelu6: 3,
    kTfLiteActTanh: 4,
    kTfLiteActSignBit: 5,
    kTfLiteActSigmoid: 6
};

$root.MNN.QuantizedParam = class QuantizedParam {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedParam();
        $.zeroPoint = reader.int32_(position, 4, 0);
        $.scale = reader.float32_(position, 6, 0);
        return $;
    }
};

$root.MNN.QuantizedAdd = class QuantizedAdd {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedAdd();
        $.activationType = reader.int8_(position, 4, 0);
        $.input1QuantizedParam = reader.table(position, 6, $root.MNN.QuantizedParam.decode);
        $.input2QuantizedParam = reader.table(position, 8, $root.MNN.QuantizedParam.decode);
        $.outputQuantizedParam = reader.table(position, 10, $root.MNN.QuantizedParam.decode);
        return $;
    }
};

$root.MNN.ModeFormat = {
    TENSORFLOW: 0,
    TFLITE: 1
};

$root.MNN.QuantizeMode = {
    MIN_COMBINED: 0,
    MIN_FIRST: 1,
    SCALED: 2
};

$root.MNN.Dequantize = class Dequantize {

    static decode(reader, position) {
        const $ = new $root.MNN.Dequantize();
        $.inputQuantizedParam = reader.table(position, 4, $root.MNN.QuantizedParam.decode);
        $.mode = reader.int8_(position, 6, 0);
        $.modelFormat = reader.int8_(position, 8, 0);
        $.type = reader.int32_(position, 10, 0);
        return $;
    }
};

$root.MNN.QuantizedAvgPool = class QuantizedAvgPool {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedAvgPool();
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

$root.MNN.QuantizedBiasAdd = class QuantizedBiasAdd {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedBiasAdd();
        $.bias = reader.typedArray(position, 4, Int32Array);
        $.inputType = reader.int32_(position, 6, 0);
        $.max = reader.int32_(position, 8, 0);
        $.min = reader.int32_(position, 10, 0);
        $.outputType = reader.int32_(position, 12, 0);
        return $;
    }
};

$root.MNN.QuantizedConcat = class QuantizedConcat {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedConcat();
        $.activationType = reader.int8_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        $.inputScale = reader.typedArray(position, 8, Float32Array);
        $.inputZeroPoint = reader.typedArray(position, 10, Int32Array);
        $.outputQuantizedParam = reader.table(position, 12, $root.MNN.QuantizedParam.decode);
        return $;
    }
};

$root.MNN.QuantizedLogistic = class QuantizedLogistic {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedLogistic();
        $.inputQuantizedParam = reader.table(position, 4, $root.MNN.QuantizedParam.decode);
        $.outputQuantizedParam = reader.table(position, 6, $root.MNN.QuantizedParam.decode);
        return $;
    }
};

$root.MNN.QuantizedMatMul = class QuantizedMatMul {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedMatMul();
        $.transposeA = reader.bool_(position, 4, false);
        $.transposeB = reader.bool_(position, 6, false);
        return $;
    }
};

$root.MNN.QuantizedMaxPool = class QuantizedMaxPool {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedMaxPool();
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

$root.MNN.QuantizedRelu = class QuantizedRelu {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedRelu();
        $.type = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.QuantizedRelu6 = class QuantizedRelu6 {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedRelu6();
        $.type = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.MNN.QuantizedReshape = class QuantizedReshape {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedReshape();
        $.dims = reader.typedArray(position, 4, Int32Array);
        $.modelFormat = reader.int8_(position, 6, 0);
        return $;
    }
};

$root.MNN.QuantizedSoftmax = class QuantizedSoftmax {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizedSoftmax();
        $.beta = reader.float32_(position, 4, 0);
        $.inputScale = reader.float32_(position, 6, 0);
        return $;
    }
};

$root.MNN.QuantizeRoundMode = {
    HALF_AWAY_FROM_ZERO: 0,
    HALF_TO_EVEN: 1
};

$root.MNN.QuantizeV2 = class QuantizeV2 {

    static decode(reader, position) {
        const $ = new $root.MNN.QuantizeV2();
        $.type = reader.int32_(position, 4, 0);
        $.mode = reader.int8_(position, 6, 0);
        $.roundMode = reader.int8_(position, 8, 0);
        return $;
    }
};

$root.MNN.RequantizationRange = class RequantizationRange {

    static decode(/* reader, position */) {
        const $ = new $root.MNN.RequantizationRange();
        return $;
    }
};

$root.MNN.Requantize = class Requantize {

    static decode(/* reader, position */) {
        const $ = new $root.MNN.Requantize();
        return $;
    }
};

$root.MNN.TfQuantizedConv2D = class TfQuantizedConv2D {

    static decode(reader, position) {
        const $ = new $root.MNN.TfQuantizedConv2D();
        $.bias = reader.typedArray(position, 4, Int32Array);
        $.biasflag = reader.bool_(position, 6, false);
        $.common = reader.table(position, 8, $root.MNN.Convolution2DCommon.decode);
        $.weight = reader.typedArray(position, 10, Uint8Array);
        $.activationType = reader.int8_(position, 12, 0);
        $.multiplier = reader.int32_(position, 14, 0);
        $.outMax = reader.int32_(position, 16, 0);
        $.outMin = reader.int32_(position, 18, 0);
        $.shift = reader.int32_(position, 20, 0);
        $.biasQuantizedParam = reader.table(position, 22, $root.MNN.QuantizedParam.decode);
        $.depthMultiplier = reader.int32_(position, 24, 0);
        $.filterQuantizedParam = reader.table(position, 26, $root.MNN.QuantizedParam.decode);
        $.inputQuantizedParam = reader.table(position, 28, $root.MNN.QuantizedParam.decode);
        $.modelFormat = reader.int8_(position, 30, 0);
        $.outputQuantizedParam = reader.table(position, 32, $root.MNN.QuantizedParam.decode);
        return $;
    }
};

$root.MNN.ExtraInfo = class ExtraInfo {

    static decode(reader, position) {
        const $ = new $root.MNN.ExtraInfo();
        $.buffer = reader.typedArray(position, 4, Int8Array);
        $.name = reader.string_(position, 6, null);
        $.version = reader.string_(position, 8, null);
        return $;
    }
};

$root.MNN.TensorConvertInfo = class TensorConvertInfo {

    static decode(reader, position) {
        const $ = new $root.MNN.TensorConvertInfo();
        $.source = reader.int8_(position, 4, 0);
        $.dest = reader.int8_(position, 6, 0);
        return $;
    }
};

$root.MNN.SampleMode = {
    BILINEAR: 0,
    NEAREST: 1
};

$root.MNN.BorderMode = {
    ZEROS: 0,
    CLAMP: 1,
    REFLECTION: 2
};

$root.MNN.GridSample = class GridSample {

    static decode(reader, position) {
        const $ = new $root.MNN.GridSample();
        $.mode = reader.int8_(position, 4, 0);
        $.paddingMode = reader.int8_(position, 6, 0);
        $.alignCorners = reader.bool_(position, 8, false);
        return $;
    }
};

$root.MNN.ImageFormatType = {
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

$root.MNN.FilterType = {
    NEAREST: 0,
    BILINEAR: 1,
    BICUBIC: 2
};

$root.MNN.WrapType = {
    CLAMP_TO_EDGE: 0,
    ZERO: 1,
    REPEAT: 2
};

$root.MNN.ImageProcessParam = class ImageProcessParam {

    static decode(reader, position) {
        const $ = new $root.MNN.ImageProcessParam();
        $.filterType = reader.int8_(position, 4, 0);
        $.sourceFormat = reader.int32_(position, 6, 0);
        $.destFormat = reader.int32_(position, 8, 0);
        $.wrap = reader.int8_(position, 10, 0);
        $.mean = reader.typedArray(position, 12, Float32Array);
        $.normal = reader.typedArray(position, 14, Float32Array);
        $.transform = reader.typedArray(position, 16, Float32Array);
        $.paddingValue = reader.int8_(position, 18, 0);
        $.shape = reader.typedArray(position, 20, Int32Array);
        $.outputType = reader.int32_(position, 22, 0);
        $.draw = reader.bool_(position, 24, false);
        return $;
    }
};

$root.MNN.OpType = {
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
    QuantizedMatMul: 57,
    QuantizedMaxPool: 58,
    QuantizedRelu: 59,
    QuantizedRelu6: 60,
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

$root.MNN.Plugin = class Plugin {

    static decode(reader, position) {
        const $ = new $root.MNN.Plugin();
        $.type = reader.string_(position, 4, null);
        $.attr = reader.tableArray(position, 6, $root.MNN.Attribute.decode);
        return $;
    }
};

$root.MNN.Extra = class Extra {

    static decode(reader, position) {
        const $ = new $root.MNN.Extra();
        $.type = reader.string_(position, 4, null);
        $.engine = reader.string_(position, 6, null);
        $.info = reader.typedArray(position, 8, Int8Array);
        $.attr = reader.tableArray(position, 10, $root.MNN.Attribute.decode);
        $.vector = reader.bool_(position, 12, false);
        return $;
    }
};

$root.MNN.StringVec = class StringVec {

    static decode(reader, position) {
        const $ = new $root.MNN.StringVec();
        $.data = reader.strings_(position, 4);
        return $;
    }
};

$root.MNN.WhileParam = class WhileParam {

    static decode(reader, position) {
        const $ = new $root.MNN.WhileParam();
        $.cond_graph = reader.string_(position, 4, null);
        $.body_graph = reader.string_(position, 6, null);
        $.aliases_inputs = reader.tableArray(position, 8, $root.MNN.StringVec.decode);
        $.aliases_outputs = reader.strings_(position, 10);
        $.aliases_updates = reader.tableArray(position, 12, $root.MNN.StringVec.decode);
        return $;
    }
};

$root.MNN.IfParam = class IfParam {

    static decode(reader, position) {
        const $ = new $root.MNN.IfParam();
        $.then_graph = reader.string_(position, 4, null);
        $.else_graph = reader.string_(position, 6, null);
        $.aliases_inputs = reader.tableArray(position, 8, $root.MNN.StringVec.decode);
        $.aliases_outputs = reader.tableArray(position, 10, $root.MNN.StringVec.decode);
        return $;
    }
};

$root.MNN.RegionCommand = class RegionCommand {

    static decode(reader, position) {
        const $ = new $root.MNN.RegionCommand();
        $.op = reader.table(position, 4, $root.MNN.Op.decode);
        $.steps = reader.typedArray(position, 6, Int32Array);
        $.size = reader.typedArray(position, 8, Int32Array);
        $.indexes = reader.typedArray(position, 10, Int32Array);
        $.view = reader.tableArray(position, 12, $root.MNN.View.decode);
        $.fuse = reader.int32_(position, 14, -1);
        $.iterIndexes = reader.typedArray(position, 16, Int32Array);
        return $;
    }
};

$root.MNN.LoopParam = class LoopParam {

    static decode(reader, position) {
        const $ = new $root.MNN.LoopParam();
        $.tensorNumber = reader.int32_(position, 4, 0);
        $.outputIndexes = reader.typedArray(position, 6, Int32Array);
        $.inputIndexes = reader.typedArray(position, 8, Int32Array);
        $.extraTensorInfos = reader.tableArray(position, 10, $root.MNN.TensorDescribe.decode);
        $.parallel = reader.bool_(position, 12, true);
        $.loopNumber = reader.int32_(position, 14, 0);
        $.commands = reader.tableArray(position, 16, $root.MNN.RegionCommand.decode);
        $.initCommand = reader.tableArray(position, 18, $root.MNN.RegionCommand.decode);
        return $;
    }
};

$root.MNN.OpParameter = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.MNN.QuantizedAdd.decode(reader, position);
            case 2: return $root.MNN.ArgMax.decode(reader, position);
            case 3: return $root.MNN.AsString.decode(reader, position);
            case 4: return $root.MNN.Axis.decode(reader, position);
            case 5: return $root.MNN.BatchNorm.decode(reader, position);
            case 6: return $root.MNN.BinaryOp.decode(reader, position);
            case 7: return $root.MNN.Blob.decode(reader, position);
            case 8: return $root.MNN.CastParam.decode(reader, position);
            case 9: return $root.MNN.Convolution2D.decode(reader, position);
            case 10: return $root.MNN.Crop.decode(reader, position);
            case 11: return $root.MNN.CropAndResize.decode(reader, position);
            case 12: return $root.MNN.Dequantize.decode(reader, position);
            case 13: return $root.MNN.DetectionOutput.decode(reader, position);
            case 14: return $root.MNN.Eltwise.decode(reader, position);
            case 15: return $root.MNN.ExpandDims.decode(reader, position);
            case 16: return $root.MNN.Fill.decode(reader, position);
            case 17: return $root.MNN.Flatten.decode(reader, position);
            case 18: return $root.MNN.Gather.decode(reader, position);
            case 19: return $root.MNN.GatherV2.decode(reader, position);
            case 20: return $root.MNN.InnerProduct.decode(reader, position);
            case 21: return $root.MNN.Input.decode(reader, position);
            case 22: return $root.MNN.Interp.decode(reader, position);
            case 23: return $root.MNN.LRN.decode(reader, position);
            case 24: return $root.MNN.LSTM.decode(reader, position);
            case 25: return $root.MNN.MatMul.decode(reader, position);
            case 26: return $root.MNN.NonMaxSuppressionV2.decode(reader, position);
            case 27: return $root.MNN.Normalize.decode(reader, position);
            case 28: return $root.MNN.PackParam.decode(reader, position);
            case 29: return $root.MNN.Permute.decode(reader, position);
            case 30: return $root.MNN.Plugin.decode(reader, position);
            case 31: return $root.MNN.Pool.decode(reader, position);
            case 32: return $root.MNN.PRelu.decode(reader, position);
            case 33: return $root.MNN.PriorBox.decode(reader, position);
            case 34: return $root.MNN.Proposal.decode(reader, position);
            case 35: return $root.MNN.QuantizedAvgPool.decode(reader, position);
            case 36: return $root.MNN.QuantizedBiasAdd.decode(reader, position);
            case 37: return $root.MNN.QuantizedConcat.decode(reader, position);
            case 38: return $root.MNN.QuantizedLogistic.decode(reader, position);
            case 39: return $root.MNN.QuantizedMatMul.decode(reader, position);
            case 40: return $root.MNN.QuantizedMaxPool.decode(reader, position);
            case 41: return $root.MNN.QuantizedRelu.decode(reader, position);
            case 42: return $root.MNN.QuantizedRelu6.decode(reader, position);
            case 43: return $root.MNN.QuantizedReshape.decode(reader, position);
            case 44: return $root.MNN.QuantizedSoftmax.decode(reader, position);
            case 45: return $root.MNN.QuantizeMaxMin.decode(reader, position);
            case 46: return $root.MNN.QuantizeV2.decode(reader, position);
            case 47: return $root.MNN.Range.decode(reader, position);
            case 48: return $root.MNN.Rank.decode(reader, position);
            case 49: return $root.MNN.ReduceJoin.decode(reader, position);
            case 50: return $root.MNN.ReductionParam.decode(reader, position);
            case 51: return $root.MNN.Relu.decode(reader, position);
            case 52: return $root.MNN.Relu6.decode(reader, position);
            case 53: return $root.MNN.RequantizationRange.decode(reader, position);
            case 54: return $root.MNN.Requantize.decode(reader, position);
            case 55: return $root.MNN.Reshape.decode(reader, position);
            case 56: return $root.MNN.Resize.decode(reader, position);
            case 57: return $root.MNN.RoiParameters.decode(reader, position);
            case 58: return $root.MNN.Scale.decode(reader, position);
            case 59: return $root.MNN.Selu.decode(reader, position);
            case 60: return $root.MNN.Size.decode(reader, position);
            case 61: return $root.MNN.Slice.decode(reader, position);
            case 62: return $root.MNN.SliceTf.decode(reader, position);
            case 63: return $root.MNN.SpaceBatch.decode(reader, position);
            case 64: return $root.MNN.SqueezeParam.decode(reader, position);
            case 65: return $root.MNN.StridedSliceParam.decode(reader, position);
            case 66: return $root.MNN.TensorConvertInfo.decode(reader, position);
            case 67: return $root.MNN.TfQuantizedConv2D.decode(reader, position);
            case 68: return $root.MNN.TopKV2.decode(reader, position);
            case 69: return $root.MNN.Transpose.decode(reader, position);
            case 70: return $root.MNN.UnaryOp.decode(reader, position);
            case 71: return $root.MNN.MomentsParam.decode(reader, position);
            case 72: return $root.MNN.RNNParam.decode(reader, position);
            case 73: return $root.MNN.BatchMatMulParam.decode(reader, position);
            case 74: return $root.MNN.QuantizedFloatParam.decode(reader, position);
            case 75: return $root.MNN.DepthSpaceParam.decode(reader, position);
            case 76: return $root.MNN.EltwiseInt8.decode(reader, position);
            case 77: return $root.MNN.ReverseSequenceParam.decode(reader, position);
            case 78: return $root.MNN.Extra.decode(reader, position);
            case 79: return $root.MNN.Pool3D.decode(reader, position);
            case 80: return $root.MNN.Convolution3D.decode(reader, position);
            case 81: return $root.MNN.ELU.decode(reader, position);
            case 82: return $root.MNN.DetectionPostProcessParam.decode(reader, position);
            case 83: return $root.MNN.OneHotParam.decode(reader, position);
            case 84: return $root.MNN.PadParam.decode(reader, position);
            case 85: return $root.MNN.WhileParam.decode(reader, position);
            case 86: return $root.MNN.IfParam.decode(reader, position);
            case 87: return $root.MNN.RandomUniform.decode(reader, position);
            case 88: return $root.MNN.LayerNorm.decode(reader, position);
            case 89: return $root.MNN.TensorArray.decode(reader, position);
            case 90: return $root.MNN.LSTMBlockCell.decode(reader, position);
            case 91: return $root.MNN.GridSample.decode(reader, position);
            case 92: return $root.MNN.LoopParam.decode(reader, position);
            case 93: return $root.MNN.ImageProcessParam.decode(reader, position);
            case 94: return $root.MNN.CumSum.decode(reader, position);
            default: return undefined;
        }
    }
};

$root.MNN.Op = class Op {

    static decode(reader, position) {
        const $ = new $root.MNN.Op();
        $.inputIndexes = reader.typedArray(position, 4, Int32Array);
        $.main = reader.union(position, 6, $root.MNN.OpParameter.decode);
        $.name = reader.string_(position, 10, null);
        $.outputIndexes = reader.typedArray(position, 12, Int32Array);
        $.type = reader.int32_(position, 14, 0);
        $.defaultDimentionFormat = reader.int8_(position, 16, 1);
        return $;
    }
};

$root.MNN.View = class View {

    static decode(reader, position) {
        const $ = new $root.MNN.View();
        $.offset = reader.int32_(position, 4, 0);
        $.stride = reader.typedArray(position, 6, Int32Array);
        return $;
    }
};

$root.MNN.Region = class Region {

    static decode(reader, position) {
        const $ = new $root.MNN.Region();
        $.src = reader.table(position, 4, $root.MNN.View.decode);
        $.dst = reader.table(position, 6, $root.MNN.View.decode);
        $.size = reader.typedArray(position, 8, Int32Array);
        $.origin = reader.int32_(position, 10, 0);
        return $;
    }
};

$root.MNN.TensorDescribe = class TensorDescribe {

    static decode(reader, position) {
        const $ = new $root.MNN.TensorDescribe();
        $.blob = reader.table(position, 4, $root.MNN.Blob.decode);
        $.index = reader.int32_(position, 6, 0);
        $.name = reader.string_(position, 8, null);
        $.regions = reader.tableArray(position, 10, $root.MNN.Region.decode);
        $.quantInfo = reader.table(position, 12, $root.MNN.TensorQuantInfo.decode);
        return $;
    }
};

$root.MNN.ForwardType = {
    CPU: 0,
    METAL: 1,
    OPENCL: 2,
    OPENGLES: 3,
    VULKAN: 4
};

$root.MNN.Usage = {
    INFERENCE: 0,
    TRAIN: 1,
    INFERENCE_STATIC: 2
};

$root.MNN.SubGraphProto = class SubGraphProto {

    static decode(reader, position) {
        const $ = new $root.MNN.SubGraphProto();
        $.name = reader.string_(position, 4, null);
        $.inputs = reader.typedArray(position, 6, Int32Array);
        $.outputs = reader.typedArray(position, 8, Int32Array);
        $.tensors = reader.strings_(position, 10);
        $.nodes = reader.tableArray(position, 12, $root.MNN.Op.decode);
        $.extraTensorDescribe = reader.tableArray(position, 14, $root.MNN.TensorDescribe.decode);
        return $;
    }
};

$root.MNN.TensorQuantInfo = class TensorQuantInfo {

    static decode(reader, position) {
        const $ = new $root.MNN.TensorQuantInfo();
        $.scale = reader.float32_(position, 4, 0);
        $.zero = reader.float32_(position, 6, 0);
        $.min = reader.float32_(position, 8, -128);
        $.max = reader.float32_(position, 10, 127);
        $.type = reader.int32_(position, 12, 0);
        return $;
    }
};

$root.MNN.Net = class Net {

    static create(reader) {
        return $root.MNN.Net.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.MNN.Net();
        $.bizCode = reader.string_(position, 4, null);
        $.extraTensorDescribe = reader.tableArray(position, 6, $root.MNN.TensorDescribe.decode);
        $.extraInfo = reader.table(position, 8, $root.MNN.ExtraInfo.decode);
        $.oplists = reader.tableArray(position, 10, $root.MNN.Op.decode);
        $.outputName = reader.strings_(position, 12);
        $.preferForwardType = reader.int8_(position, 14, 0);
        $.sourceType = reader.int8_(position, 16, 0);
        $.tensorName = reader.strings_(position, 18);
        $.tensorNumber = reader.int32_(position, 20, 0);
        $.usage = reader.int8_(position, 22, 0);
        $.subgraphs = reader.tableArray(position, 24, $root.MNN.SubGraphProto.decode);
        $.mnn_uuid = reader.string_(position, 26, null);
        return $;
    }
};
