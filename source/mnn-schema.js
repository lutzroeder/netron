
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

    static decodeText(reader, json) {
        const $ = new MNN.Blob();
        $.dims = reader.array(json.dims, Int32Array);
        $.dataFormat = MNN.MNN_DATA_FORMAT[json.dataFormat];
        $.dataType = MNN.DataType[json.dataType];
        $.uint8s = reader.array(json.uint8s, Uint8Array);
        $.int8s = reader.array(json.int8s, Int8Array);
        $.int32s = reader.array(json.int32s, Int32Array);
        $.int64s = reader.array(json.int64s);
        $.float32s = reader.array(json.float32s, Float32Array);
        $.strings = reader.array(json.strings);
        $.external = reader.array(json.external);
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

    static decodeText(reader, json) {
        const $ = new MNN.ListValue();
        $.s = reader.array(json.s);
        $.i = reader.array(json.i, Int32Array);
        $.f = reader.array(json.f, Float32Array);
        $.b = reader.array(json.b);
        $.type = reader.objects(json.type, MNN.DataType);
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

    static decodeText(reader, json) {
        const $ = new MNN.Attribute();
        $.s = reader.value(json.s, null);
        $.i = reader.value(json.i, 0);
        $.b = reader.value(json.b, false);
        $.key = reader.value(json.key, null);
        $.type = MNN.DataType[json.type];
        $.f = reader.value(json.f, 0);
        $.tensor = reader.object(json.tensor, MNN.Blob);
        $.list = reader.object(json.list, MNN.ListValue);
        $.func = reader.object(json.func, MNN.NamedAttrList);
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

    static decodeText(reader, json) {
        const $ = new MNN.NamedAttrList();
        $.name = reader.value(json.name, null);
        $.attr = reader.objects(json.attr, MNN.Attribute);
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

    static decodeText(reader, json) {
        const $ = new MNN.Convolution2DCommon();
        $.padX = reader.value(json.padX, 0);
        $.padY = reader.value(json.padY, 0);
        $.kernelX = reader.value(json.kernelX, 1);
        $.kernelY = reader.value(json.kernelY, 1);
        $.strideX = reader.value(json.strideX, 1);
        $.strideY = reader.value(json.strideY, 1);
        $.dilateX = reader.value(json.dilateX, 1);
        $.dilateY = reader.value(json.dilateY, 1);
        $.padMode = MNN.PadMode[json.padMode];
        $.group = reader.value(json.group, 1);
        $.outputCount = reader.value(json.outputCount, 0);
        $.inputCount = reader.value(json.inputCount, 0);
        $.relu = reader.value(json.relu, false);
        $.relu6 = reader.value(json.relu6, false);
        $.pads = reader.array(json.pads, Int32Array);
        $.outPads = reader.array(json.outPads, Int32Array);
        $.hasOutputShape = reader.value(json.hasOutputShape, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.Convolution3DCommon();
        $.dilates = reader.array(json.dilates, Int32Array);
        $.strides = reader.array(json.strides, Int32Array);
        $.kernels = reader.array(json.kernels, Int32Array);
        $.pads = reader.array(json.pads, Int32Array);
        $.padMode = MNN.PadMode[json.padMode];
        $.inputCount = reader.value(json.inputCount, 0);
        $.outputCount = reader.value(json.outputCount, 0);
        $.relu = reader.value(json.relu, false);
        $.relu6 = reader.value(json.relu6, false);
        $.group = reader.value(json.group, 1);
        $.outPads = reader.array(json.outPads, Int32Array);
        $.hasOutputShape = reader.value(json.hasOutputShape, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.SparseCommon();
        $.method = MNN.SparseAlgo[json.method];
        $.args = reader.objects(json.args, MNN.Attribute);
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

    static decodeText(reader, json) {
        const $ = new MNN.IDSTQuan();
        $.buffer = reader.array(json.buffer, Int8Array);
        $.alpha = reader.array(json.alpha, Float32Array);
        $.type = reader.value(json.type, 0);
        $.useInt32 = reader.value(json.useInt32, false);
        $.quantScale = reader.value(json.quantScale, 0);
        $.scaleIn = reader.value(json.scaleIn, 0);
        $.scaleOut = reader.value(json.scaleOut, 0);
        $.aMax = reader.value(json.aMax, 0);
        $.aMin = reader.value(json.aMin, 0);
        $.readType = reader.value(json.readType, 0);
        $.has_scaleInt = reader.value(json.has_scaleInt, false);
        $.shapeInt32 = reader.value(json.shapeInt32, false);
        $.weightSize = reader.value(json.weightSize, 0);
        $.index = reader.array(json.index, Uint32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedFloatParam();
        $.weight = reader.array(json.weight, Int8Array);
        $.bias = reader.array(json.bias, Int32Array);
        $.scale = reader.array(json.scale, Float32Array);
        $.tensorScale = reader.array(json.tensorScale, Float32Array);
        $.method = MNN.QuantizeAlgo[json.method];
        $.nbits = reader.value(json.nbits, 8);
        $.zeroPoint = reader.value(json.zeroPoint, 0);
        $.outputZeroPoint = reader.value(json.outputZeroPoint, 0);
        $.clampMin = reader.value(json.clampMin, -128);
        $.clampMax = reader.value(json.clampMax, 127);
        $.winogradAttr = reader.array(json.winogradAttr, Int32Array);
        $.outputDataType = MNN.DataType[json.outputDataType];
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

    static decodeText(reader, json) {
        const $ = new MNN.Convolution2D();
        $.common = reader.object(json.common, MNN.Convolution2DCommon);
        $.weight = reader.array(json.weight, Float32Array);
        $.bias = reader.array(json.bias, Float32Array);
        $.quanParameter = reader.object(json.quanParameter, MNN.IDSTQuan);
        $.symmetricQuan = reader.object(json.symmetricQuan, MNN.QuantizedFloatParam);
        $.sparseParameter = reader.object(json.sparseParameter, MNN.SparseCommon);
        $.external = reader.array(json.external);
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

    static decodeText(reader, json) {
        const $ = new MNN.Convolution3D();
        $.common = reader.object(json.common, MNN.Convolution3DCommon);
        $.weight = reader.array(json.weight, Float32Array);
        $.bias = reader.array(json.bias, Float32Array);
        $.external = reader.array(json.external);
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

    static decodeText(reader, json) {
        const $ = new MNN.InnerProduct();
        $.outputCount = reader.value(json.outputCount, 0);
        $.biasTerm = reader.value(json.biasTerm, 0);
        $.weightSize = reader.value(json.weightSize, 0);
        $.weight = reader.array(json.weight, Float32Array);
        $.bias = reader.array(json.bias, Float32Array);
        $.axis = reader.value(json.axis, 0);
        $.transpose = reader.value(json.transpose, false);
        $.quanParameter = reader.object(json.quanParameter, MNN.IDSTQuan);
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

    static decodeText(reader, json) {
        const $ = new MNN.Pool();
        $.padX = reader.value(json.padX, 0);
        $.padY = reader.value(json.padY, 0);
        $.isGlobal = reader.value(json.isGlobal, false);
        $.kernelX = reader.value(json.kernelX, 0);
        $.kernelY = reader.value(json.kernelY, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.type = MNN.PoolType[json.type];
        $.padType = MNN.PoolPadType[json.padType];
        $.dataType = MNN.DataType[json.dataType];
        $.ceilModel = reader.value(json.ceilModel, true);
        $.pads = reader.array(json.pads, Int32Array);
        $.countType = MNN.AvgPoolCountType[json.countType];
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

    static decodeText(reader, json) {
        const $ = new MNN.Pool3D();
        $.strides = reader.array(json.strides, Int32Array);
        $.kernels = reader.array(json.kernels, Int32Array);
        $.pads = reader.array(json.pads, Int32Array);
        $.type = MNN.PoolType[json.type];
        $.padType = MNN.PoolPadType[json.padType];
        $.isGlobal = reader.value(json.isGlobal, false);
        return $;
    }
};

MNN.Relu = class Relu {

    static decode(reader, position) {
        const $ = new MNN.Relu();
        $.slope = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.Relu();
        $.slope = reader.value(json.slope, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.Relu6();
        $.minValue = reader.value(json.minValue, 0);
        $.maxValue = reader.value(json.maxValue, 6);
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

    static decodeText(reader, json) {
        const $ = new MNN.PRelu();
        $.slopeCount = reader.value(json.slopeCount, 0);
        $.slope = reader.array(json.slope, Float32Array);
        return $;
    }
};

MNN.ELU = class ELU {

    static decode(reader, position) {
        const $ = new MNN.ELU();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.ELU();
        $.alpha = reader.value(json.alpha, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.LRN();
        $.regionType = reader.value(json.regionType, 0);
        $.localSize = reader.value(json.localSize, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
        $.bias = reader.value(json.bias, 1);
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

    static decodeText(reader, json) {
        const $ = new MNN.ArgMax();
        $.outMaxVal = reader.value(json.outMaxVal, 0);
        $.topK = reader.value(json.topK, 0);
        $.axis = reader.value(json.axis, 0);
        $.softmaxThreshold = reader.value(json.softmaxThreshold, 0);
        return $;
    }
};

MNN.Axis = class Axis {

    static decode(reader, position) {
        const $ = new MNN.Axis();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.Axis();
        $.axis = reader.value(json.axis, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.Input();
        $.dims = reader.array(json.dims, Int32Array);
        $.dtype = MNN.DataType[json.dtype];
        $.dformat = MNN.MNN_DATA_FORMAT[json.dformat];
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

    static decodeText(reader, json) {
        const $ = new MNN.LSTM();
        $.outputCount = reader.value(json.outputCount, 0);
        $.weightSize = reader.value(json.weightSize, 0);
        $.clippingThreshold = reader.value(json.clippingThreshold, 0);
        $.weightI = reader.object(json.weightI, MNN.Blob);
        $.weightH = reader.object(json.weightH, MNN.Blob);
        $.bias = reader.object(json.bias, MNN.Blob);
        $.weightIQ = reader.object(json.weightIQ, MNN.Blob);
        $.weightIA = reader.object(json.weightIA, MNN.Blob);
        $.quantScale = reader.value(json.quantScale, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.Slice();
        $.axis = reader.value(json.axis, 0);
        $.slicePoints = reader.array(json.slicePoints, Int32Array);
        $.sourceType = MNN.NetSource[json.sourceType];
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

    static decodeText(reader, json) {
        const $ = new MNN.BatchNorm();
        $.channels = reader.value(json.channels, 0);
        $.slopeData = reader.array(json.slopeData, Float32Array);
        $.meanData = reader.array(json.meanData, Float32Array);
        $.varData = reader.array(json.varData, Float32Array);
        $.biasData = reader.array(json.biasData, Float32Array);
        $.Adata = reader.array(json.Adata, Float32Array);
        $.Bdata = reader.array(json.Bdata, Float32Array);
        $.epsilon = reader.value(json.epsilon, 0.001);
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

    static decodeText(reader, json) {
        const $ = new MNN.Scale();
        $.channels = reader.value(json.channels, 0);
        $.scaleData = reader.array(json.scaleData, Float32Array);
        $.biasData = reader.array(json.biasData, Float32Array);
        $.external = reader.array(json.external);
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

    static decodeText(reader, json) {
        const $ = new MNN.Eltwise();
        $.type = MNN.EltwiseType[json.type];
        $.coeff = reader.array(json.coeff, Float32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.Flatten();
        $.axis = reader.value(json.axis, 0);
        $.endAxis = reader.value(json.endAxis, 0);
        return $;
    }
};

MNN.Permute = class Permute {

    static decode(reader, position) {
        const $ = new MNN.Permute();
        $.dims = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.Permute();
        $.dims = reader.array(json.dims, Int32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.Reshape();
        $.dims = reader.array(json.dims, Int32Array);
        $.dimType = MNN.MNN_DATA_FORMAT[json.dimType];
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

    static decodeText(reader, json) {
        const $ = new MNN.DetectionOutput();
        $.classCount = reader.value(json.classCount, 0);
        $.nmsThresholdold = reader.value(json.nmsThresholdold, 0);
        $.nmsTopK = reader.value(json.nmsTopK, 0);
        $.keepTopK = reader.value(json.keepTopK, 0);
        $.confidenceThreshold = reader.value(json.confidenceThreshold, 0);
        $.shareLocation = reader.value(json.shareLocation, 0);
        $.backgroundLable = reader.value(json.backgroundLable, 0);
        $.varianceEncodedTarget = reader.value(json.varianceEncodedTarget, 0);
        $.codeType = reader.value(json.codeType, 0);
        $.objectnessScore = reader.value(json.objectnessScore, 0.01);
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

    static decodeText(reader, json) {
        const $ = new MNN.RoiParameters();
        $.pooledWidth = reader.value(json.pooledWidth, 0);
        $.pooledHeight = reader.value(json.pooledHeight, 0);
        $.spatialScale = reader.value(json.spatialScale, 0);
        $.samplingRatio = reader.value(json.samplingRatio, -1);
        $.aligned = reader.value(json.aligned, false);
        $.poolType = MNN.PoolType[json.poolType];
        $.outputGrad = reader.value(json.outputGrad, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.Proposal();
        $.featStride = reader.value(json.featStride, 0);
        $.baseSize = reader.value(json.baseSize, 0);
        $.preNmsTopN = reader.value(json.preNmsTopN, 0);
        $.afterNmsTopN = reader.value(json.afterNmsTopN, 0);
        $.nmsThreshold = reader.value(json.nmsThreshold, 0);
        $.minSize = reader.value(json.minSize, 0);
        $.ratios = reader.object(json.ratios, MNN.Blob);
        $.scales = reader.object(json.scales, MNN.Blob);
        $.anchors = reader.object(json.anchors, MNN.Blob);
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

    static decodeText(reader, json) {
        const $ = new MNN.Interp();
        $.widthScale = reader.value(json.widthScale, 0);
        $.heightScale = reader.value(json.heightScale, 0);
        $.outputWidth = reader.value(json.outputWidth, 0);
        $.outputHeight = reader.value(json.outputHeight, 0);
        $.resizeType = reader.value(json.resizeType, 0);
        $.alignCorners = reader.value(json.alignCorners, false);
        $.halfPixelCenters = reader.value(json.halfPixelCenters, false);
        $.widthOffset = reader.value(json.widthOffset, 0);
        $.heightOffset = reader.value(json.heightOffset, 0);
        $.cubicCoeffA = reader.value(json.cubicCoeffA, -0.75);
        $.ctm = MNN.CoordinateTransformationMode[json.ctm];
        $.depthScale = reader.value(json.depthScale, 0);
        $.outputDepth = reader.value(json.outputDepth, 0);
        $.depthOffset = reader.value(json.depthOffset, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.Resize();
        $.xScale = reader.value(json.xScale, 0);
        $.yScale = reader.value(json.yScale, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.PriorBox();
        $.minSizes = reader.array(json.minSizes, Float32Array);
        $.maxSizes = reader.array(json.maxSizes, Float32Array);
        $.aspectRatios = reader.array(json.aspectRatios, Float32Array);
        $.variances = reader.array(json.variances, Float32Array);
        $.flip = reader.value(json.flip, false);
        $.clip = reader.value(json.clip, false);
        $.imageWidth = reader.value(json.imageWidth, 0);
        $.imageHeight = reader.value(json.imageHeight, 0);
        $.stepWidth = reader.value(json.stepWidth, 0);
        $.stepHeight = reader.value(json.stepHeight, 0);
        $.offset = reader.value(json.offset, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.Normalize();
        $.acrossSpatial = reader.value(json.acrossSpatial, 0);
        $.channelShared = reader.value(json.channelShared, 0);
        $.eps = reader.value(json.eps, 0);
        $.scale = reader.array(json.scale, Float32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.EltwiseInt8();
        $.type = MNN.EltwiseType[json.type];
        $.inputQuan0 = reader.object(json.inputQuan0, MNN.QuantizedFloatParam);
        $.inputQuan1 = reader.object(json.inputQuan1, MNN.QuantizedFloatParam);
        $.outputQuan = reader.object(json.outputQuan, MNN.QuantizedFloatParam);
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

    static decodeText(reader, json) {
        const $ = new MNN.CumSum();
        $.exclusive = reader.value(json.exclusive, false);
        $.reverse = reader.value(json.reverse, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.BinaryOp();
        $.opType = reader.value(json.opType, 0);
        $.T = MNN.DataType[json.T];
        $.activationType = reader.value(json.activationType, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.PackParam();
        $.dataType = MNN.DataType[json.dataType];
        $.axis = reader.value(json.axis, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.StridedSliceParam();
        $.Index = MNN.DataType[json.Index];
        $.T = MNN.DataType[json.T];
        $.beginMask = reader.value(json.beginMask, 0);
        $.endMask = reader.value(json.endMask, 0);
        $.ellipsisMask = reader.value(json.ellipsisMask, 0);
        $.newAxisMask = reader.value(json.newAxisMask, 0);
        $.shrinkAxisMask = reader.value(json.shrinkAxisMask, 0);
        $.fromType = reader.value(json.fromType, 0);
        return $;
    }
};

MNN.SqueezeParam = class SqueezeParam {

    static decode(reader, position) {
        const $ = new MNN.SqueezeParam();
        $.squeezeDims = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.SqueezeParam();
        $.squeezeDims = reader.array(json.squeezeDims, Int32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.CastParam();
        $.srcT = MNN.DataType[json.srcT];
        $.dstT = MNN.DataType[json.dstT];
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

    static decodeText(reader, json) {
        const $ = new MNN.ReductionParam();
        $.operation = MNN.ReductionType[json.operation];
        $.dim = reader.array(json.dim, Int32Array);
        $.coeff = reader.value(json.coeff, 0);
        $.keepDims = reader.value(json.keepDims, false);
        $.dType = MNN.DataType[json.dType];
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

    static decodeText(reader, json) {
        const $ = new MNN.Gather();
        $.Tindices = MNN.DataType[json.Tindices];
        $.Tparams = MNN.DataType[json.Tparams];
        $.validateIndices = reader.value(json.validateIndices, false);
        $.axis = reader.value(json.axis, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.ExpandDims();
        $.T = MNN.DataType[json.T];
        $.Tdim = MNN.DataType[json.Tdim];
        $.axis = reader.value(json.axis, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.Selu();
        $.scale = reader.value(json.scale, 0);
        $.alpha = reader.value(json.alpha, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.AsString();
        $.T = MNN.DataType[json.T];
        $.precision = reader.value(json.precision, 0);
        $.scientific = reader.value(json.scientific, false);
        $.shortest = reader.value(json.shortest, false);
        $.width = reader.value(json.width, 0);
        $.fillString = reader.value(json.fillString, null);
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

    static decodeText(reader, json) {
        const $ = new MNN.ReduceJoin();
        $.keepDims = reader.value(json.keepDims, false);
        $.separator = reader.value(json.separator, null);
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

    static decodeText(reader, json) {
        const $ = new MNN.UnaryOp();
        $.opType = MNN.UnaryOpOperation[json.opType];
        $.T = MNN.DataType[json.T];
        $.tableInt8 = reader.array(json.tableInt8, Int8Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.TopKV2();
        $.T = MNN.DataType[json.T];
        $.sorted = reader.value(json.sorted, false);
        $.largest = reader.value(json.largest, true);
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

    static decodeText(reader, json) {
        const $ = new MNN.CropAndResize();
        $.extrapolationValue = reader.value(json.extrapolationValue, 0);
        $.method = MNN.CropAndResizeMethod[json.method];
        return $;
    }
};

MNN.Fill = class Fill {

    static decode(/* reader, position */) {
        const $ = new MNN.Fill();
        return $;
    }

    static decodeText(/* reader, json */) {
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

    static decodeText(reader, json) {
        const $ = new MNN.GatherV2();
        $.Taxis = MNN.DataType[json.Taxis];
        $.Tindices = MNN.DataType[json.Tindices];
        $.Tparams = MNN.DataType[json.Tparams];
        return $;
    }
};

MNN.NonMaxSuppressionV2 = class NonMaxSuppressionV2 {

    static decode(/* reader, position */) {
        const $ = new MNN.NonMaxSuppressionV2();
        return $;
    }

    static decodeText(/* reader, json */) {
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

    static decodeText(reader, json) {
        const $ = new MNN.Range();
        $.Tidx = MNN.DataType[json.Tidx];
        return $;
    }
};

MNN.Rank = class Rank {

    static decode(/* reader, position */) {
        const $ = new MNN.Rank();
        return $;
    }

    static decodeText(/* reader, json */) {
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

    static decodeText(reader, json) {
        const $ = new MNN.Size();
        $.outputDataType = MNN.DataType[json.outputDataType];
        return $;
    }
};

MNN.Transpose = class Transpose {

    static decode(reader, position) {
        const $ = new MNN.Transpose();
        $.Tperm = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.Transpose();
        $.Tperm = MNN.DataType[json.Tperm];
        return $;
    }
};

MNN.SliceTf = class SliceTf {

    static decode(reader, position) {
        const $ = new MNN.SliceTf();
        $.T = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.SliceTf();
        $.T = MNN.DataType[json.T];
        return $;
    }
};

MNN.QuantizeMaxMin = class QuantizeMaxMin {

    static decode(reader, position) {
        const $ = new MNN.QuantizeMaxMin();
        $.T = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.QuantizeMaxMin();
        $.T = MNN.DataType[json.T];
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

    static decodeText(reader, json) {
        const $ = new MNN.Crop();
        $.axis = reader.value(json.axis, 2);
        $.offset = reader.array(json.offset, Int32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.SpaceBatch();
        $.blockShape = reader.object(json.blockShape, MNN.Blob);
        $.padding = reader.object(json.padding, MNN.Blob);
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

    static decodeText(reader, json) {
        const $ = new MNN.MatMul();
        $.T = MNN.DataType[json.T];
        $.transposeA = reader.value(json.transposeA, false);
        $.transposeB = reader.value(json.transposeB, false);
        $.weight = reader.array(json.weight, Float32Array);
        $.bias = reader.array(json.bias, Float32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.MomentsParam();
        $.dim = reader.array(json.dim, Int32Array);
        $.keepDims = reader.value(json.keepDims, true);
        $.dType = MNN.DataType[json.dType];
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

    static decodeText(reader, json) {
        const $ = new MNN.RNNParam();
        $.numUnits = reader.value(json.numUnits, 0);
        $.isBidirectionalRNN = reader.value(json.isBidirectionalRNN, false);
        $.linearBeforeReset = reader.value(json.linearBeforeReset, false);
        $.keepAllOutputs = reader.value(json.keepAllOutputs, false);
        $.fwGateWeight = reader.object(json.fwGateWeight, MNN.Blob);
        $.fwGateBias = reader.object(json.fwGateBias, MNN.Blob);
        $.fwCandidateWeight = reader.object(json.fwCandidateWeight, MNN.Blob);
        $.fwCandidateBias = reader.object(json.fwCandidateBias, MNN.Blob);
        $.fwRecurrentBias = reader.object(json.fwRecurrentBias, MNN.Blob);
        $.bwGateWeight = reader.object(json.bwGateWeight, MNN.Blob);
        $.bwGateBias = reader.object(json.bwGateBias, MNN.Blob);
        $.bwCandidateWeight = reader.object(json.bwCandidateWeight, MNN.Blob);
        $.bwCandidateBias = reader.object(json.bwCandidateBias, MNN.Blob);
        $.bwRecurrentBias = reader.object(json.bwRecurrentBias, MNN.Blob);
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

    static decodeText(reader, json) {
        const $ = new MNN.BatchMatMulParam();
        $.adjX = reader.value(json.adjX, false);
        $.adjY = reader.value(json.adjY, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.DepthSpaceParam();
        $.blockSize = reader.value(json.blockSize, 0);
        $.mode = MNN.DepthToSpaceMode[json.mode];
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

    static decodeText(reader, json) {
        const $ = new MNN.ReverseSequenceParam();
        $.batchDim = reader.value(json.batchDim, 0);
        $.seqDim = reader.value(json.seqDim, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.DetectionPostProcessParam();
        $.maxDetections = reader.value(json.maxDetections, 0);
        $.maxClassesPerDetection = reader.value(json.maxClassesPerDetection, 0);
        $.detectionsPerClass = reader.value(json.detectionsPerClass, 0);
        $.nmsScoreThreshold = reader.value(json.nmsScoreThreshold, 0);
        $.iouThreshold = reader.value(json.iouThreshold, 0);
        $.numClasses = reader.value(json.numClasses, 0);
        $.useRegularNMS = reader.value(json.useRegularNMS, false);
        $.centerSizeEncoding = reader.array(json.centerSizeEncoding, Float32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.OneHotParam();
        $.dType = MNN.DataType[json.dType];
        $.axis = reader.value(json.axis, -1);
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

    static decodeText(reader, json) {
        const $ = new MNN.PadParam();
        $.mode = MNN.PadValueMode[json.mode];
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

    static decodeText(reader, json) {
        const $ = new MNN.LayerNorm();
        $.axis = reader.array(json.axis, Int32Array);
        $.epsilon = reader.value(json.epsilon, 0);
        $.gamma = reader.array(json.gamma, Float32Array);
        $.beta = reader.array(json.beta, Float32Array);
        $.group = reader.value(json.group, 1);
        $.external = reader.array(json.external);
        $.useRMSNorm = reader.value(json.useRMSNorm, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.GroupNorm();
        $.axis = reader.value(json.axis, 0);
        $.epsilon = reader.value(json.epsilon, 0);
        $.gamma = reader.array(json.gamma, Float32Array);
        $.beta = reader.array(json.beta, Float32Array);
        $.group = reader.value(json.group, 1);
        $.bSwish = reader.value(json.bSwish, 0);
        $.external = reader.array(json.external);
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

    static decodeText(reader, json) {
        const $ = new MNN.RandomUniform();
        $.seed = reader.value(json.seed, 0);
        $.seed2 = reader.value(json.seed2, 0);
        $.type = MNN.DataType[json.type];
        $.low = reader.value(json.low, 0);
        $.high = reader.value(json.high, 1);
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

    static decodeText(reader, json) {
        const $ = new MNN.TensorArray();
        $.dynamic_size = reader.value(json.dynamic_size, false);
        $.identical_element_shapes = reader.value(json.identical_element_shapes, false);
        $.element_shape = reader.array(json.element_shape, Int32Array);
        $.T = MNN.DataType[json.T];
        $.axis = reader.value(json.axis, 0);
        $.keepdims = reader.value(json.keepdims, true);
        $.new_axis = reader.value(json.new_axis, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.LSTMBlockCell();
        $.cell_clip = reader.value(json.cell_clip, 3);
        $.forget_bias = reader.value(json.forget_bias, 1);
        $.use_peephole = reader.value(json.use_peephole, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedParam();
        $.zeroPoint = reader.value(json.zeroPoint, 0);
        $.scale = reader.value(json.scale, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedAdd();
        $.activationType = MNN.FusedActivation[json.activationType];
        $.input1QuantizedParam = reader.object(json.input1QuantizedParam, MNN.QuantizedParam);
        $.input2QuantizedParam = reader.object(json.input2QuantizedParam, MNN.QuantizedParam);
        $.outputQuantizedParam = reader.object(json.outputQuantizedParam, MNN.QuantizedParam);
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

    static decodeText(reader, json) {
        const $ = new MNN.Dequantize();
        $.inputQuantizedParam = reader.object(json.inputQuantizedParam, MNN.QuantizedParam);
        $.mode = MNN.QuantizeMode[json.mode];
        $.modelFormat = MNN.ModeFormat[json.modelFormat];
        $.type = MNN.DataType[json.type];
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedAvgPool();
        $.kernelX = reader.value(json.kernelX, 0);
        $.kernelY = reader.value(json.kernelY, 0);
        $.modelFormat = MNN.ModeFormat[json.modelFormat];
        $.outputActivationMax = reader.value(json.outputActivationMax, 0);
        $.outputActivationMin = reader.value(json.outputActivationMin, 0);
        $.padType = MNN.PoolPadType[json.padType];
        $.padX = reader.value(json.padX, 0);
        $.padY = reader.value(json.padY, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.type = MNN.DataType[json.type];
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedBiasAdd();
        $.bias = reader.array(json.bias, Int32Array);
        $.inputType = MNN.DataType[json.inputType];
        $.max = reader.value(json.max, 0);
        $.min = reader.value(json.min, 0);
        $.outputType = MNN.DataType[json.outputType];
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedConcat();
        $.activationType = MNN.FusedActivation[json.activationType];
        $.axis = reader.value(json.axis, 0);
        $.inputScale = reader.array(json.inputScale, Float32Array);
        $.inputZeroPoint = reader.array(json.inputZeroPoint, Int32Array);
        $.outputQuantizedParam = reader.object(json.outputQuantizedParam, MNN.QuantizedParam);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedLogistic();
        $.inputQuantizedParam = reader.object(json.inputQuantizedParam, MNN.QuantizedParam);
        $.outputQuantizedParam = reader.object(json.outputQuantizedParam, MNN.QuantizedParam);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedMatMul();
        $.transposeA = reader.value(json.transposeA, false);
        $.transposeB = reader.value(json.transposeB, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedMaxPool();
        $.kernelX = reader.value(json.kernelX, 0);
        $.kernelY = reader.value(json.kernelY, 0);
        $.modelFormat = MNN.ModeFormat[json.modelFormat];
        $.outputActivationMax = reader.value(json.outputActivationMax, 0);
        $.outputActivationMin = reader.value(json.outputActivationMin, 0);
        $.padType = MNN.PoolPadType[json.padType];
        $.padX = reader.value(json.padX, 0);
        $.padY = reader.value(json.padY, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.type = MNN.DataType[json.type];
        return $;
    }
};

MNN.QuantizedRelu = class QuantizedRelu {

    static decode(reader, position) {
        const $ = new MNN.QuantizedRelu();
        $.type = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedRelu();
        $.type = MNN.DataType[json.type];
        return $;
    }
};

MNN.QuantizedRelu6 = class QuantizedRelu6 {

    static decode(reader, position) {
        const $ = new MNN.QuantizedRelu6();
        $.type = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedRelu6();
        $.type = MNN.DataType[json.type];
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedReshape();
        $.dims = reader.array(json.dims, Int32Array);
        $.modelFormat = MNN.ModeFormat[json.modelFormat];
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizedSoftmax();
        $.beta = reader.value(json.beta, 0);
        $.inputScale = reader.value(json.inputScale, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.QuantizeV2();
        $.type = MNN.DataType[json.type];
        $.mode = MNN.QuantizeMode[json.mode];
        $.roundMode = MNN.QuantizeRoundMode[json.roundMode];
        return $;
    }
};

MNN.RequantizationRange = class RequantizationRange {

    static decode(/* reader, position */) {
        const $ = new MNN.RequantizationRange();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new MNN.RequantizationRange();
        return $;
    }
};

MNN.Requantize = class Requantize {

    static decode(/* reader, position */) {
        const $ = new MNN.Requantize();
        return $;
    }

    static decodeText(/* reader, json */) {
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

    static decodeText(reader, json) {
        const $ = new MNN.TfQuantizedConv2D();
        $.bias = reader.array(json.bias, Int32Array);
        $.biasflag = reader.value(json.biasflag, false);
        $.common = reader.object(json.common, MNN.Convolution2DCommon);
        $.weight = reader.array(json.weight, Uint8Array);
        $.activationType = MNN.FusedActivation[json.activationType];
        $.multiplier = reader.value(json.multiplier, 0);
        $.outMax = reader.value(json.outMax, 0);
        $.outMin = reader.value(json.outMin, 0);
        $.shift = reader.value(json.shift, 0);
        $.biasQuantizedParam = reader.object(json.biasQuantizedParam, MNN.QuantizedParam);
        $.depthMultiplier = reader.value(json.depthMultiplier, 0);
        $.filterQuantizedParam = reader.object(json.filterQuantizedParam, MNN.QuantizedParam);
        $.inputQuantizedParam = reader.object(json.inputQuantizedParam, MNN.QuantizedParam);
        $.modelFormat = MNN.ModeFormat[json.modelFormat];
        $.outputQuantizedParam = reader.object(json.outputQuantizedParam, MNN.QuantizedParam);
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

    static decodeText(reader, json) {
        const $ = new MNN.ExtraInfo();
        $.buffer = reader.array(json.buffer, Int8Array);
        $.name = reader.value(json.name, null);
        $.version = reader.value(json.version, null);
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

    static decodeText(reader, json) {
        const $ = new MNN.TensorConvertInfo();
        $.source = MNN.MNN_DATA_FORMAT[json.source];
        $.dest = MNN.MNN_DATA_FORMAT[json.dest];
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

    static decodeText(reader, json) {
        const $ = new MNN.GridSample();
        $.mode = MNN.SampleMode[json.mode];
        $.paddingMode = MNN.BorderMode[json.paddingMode];
        $.alignCorners = reader.value(json.alignCorners, false);
        $.backward = reader.value(json.backward, false);
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

    static decodeText(reader, json) {
        const $ = new MNN.ImageProcessParam();
        $.filterType = MNN.FilterType[json.filterType];
        $.sourceFormat = MNN.ImageFormatType[json.sourceFormat];
        $.destFormat = MNN.ImageFormatType[json.destFormat];
        $.wrap = MNN.WrapType[json.wrap];
        $.mean = reader.array(json.mean, Float32Array);
        $.normal = reader.array(json.normal, Float32Array);
        $.transform = reader.array(json.transform, Float32Array);
        $.paddingValue = reader.value(json.paddingValue, 0);
        $.shape = reader.array(json.shape, Int32Array);
        $.outputType = MNN.DataType[json.outputType];
        $.draw = reader.value(json.draw, false);
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
    Attention: 299,
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

    static decodeText(reader, json) {
        const $ = new MNN.Plugin();
        $.type = reader.value(json.type, null);
        $.attr = reader.objects(json.attr, MNN.Attribute);
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

    static decodeText(reader, json) {
        const $ = new MNN.Extra();
        $.type = reader.value(json.type, null);
        $.engine = reader.value(json.engine, null);
        $.info = reader.array(json.info, Int8Array);
        $.attr = reader.objects(json.attr, MNN.Attribute);
        $.vector = reader.value(json.vector, false);
        return $;
    }
};

MNN.StringVec = class StringVec {

    static decode(reader, position) {
        const $ = new MNN.StringVec();
        $.data = reader.strings_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.StringVec();
        $.data = reader.array(json.data);
        return $;
    }
};

MNN.AttentionParam = class AttentionParam {

    static decode(reader, position) {
        const $ = new MNN.AttentionParam();
        $.kv_cache = reader.bool_(position, 4, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.AttentionParam();
        $.kv_cache = reader.value(json.kv_cache, true);
        return $;
    }
};

MNN.FmhaV2Param = class FmhaV2Param {

    static decode(reader, position) {
        const $ = new MNN.FmhaV2Param();
        $.heads = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.FmhaV2Param();
        $.heads = reader.value(json.heads, 0);
        return $;
    }
};

MNN.FmhcaParam = class FmhcaParam {

    static decode(reader, position) {
        const $ = new MNN.FmhcaParam();
        $.heads = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new MNN.FmhcaParam();
        $.heads = reader.value(json.heads, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.WhileParam();
        $.cond_graph = reader.value(json.cond_graph, null);
        $.body_graph = reader.value(json.body_graph, null);
        $.aliases_inputs = reader.objects(json.aliases_inputs, MNN.StringVec);
        $.aliases_outputs = reader.array(json.aliases_outputs);
        $.aliases_updates = reader.objects(json.aliases_updates, MNN.StringVec);
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

    static decodeText(reader, json) {
        const $ = new MNN.IfParam();
        $.then_graph = reader.value(json.then_graph, null);
        $.else_graph = reader.value(json.else_graph, null);
        $.aliases_inputs = reader.objects(json.aliases_inputs, MNN.StringVec);
        $.aliases_outputs = reader.objects(json.aliases_outputs, MNN.StringVec);
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

    static decodeText(reader, json) {
        const $ = new MNN.RegionCommand();
        $.op = reader.object(json.op, MNN.Op);
        $.steps = reader.array(json.steps, Int32Array);
        $.size = reader.array(json.size, Int32Array);
        $.indexes = reader.array(json.indexes, Int32Array);
        $.view = reader.objects(json.view, MNN.View);
        $.fuse = reader.value(json.fuse, -1);
        $.iterIndexes = reader.array(json.iterIndexes, Int32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.LoopParam();
        $.tensorNumber = reader.value(json.tensorNumber, 0);
        $.outputIndexes = reader.array(json.outputIndexes, Int32Array);
        $.inputIndexes = reader.array(json.inputIndexes, Int32Array);
        $.extraTensorInfos = reader.objects(json.extraTensorInfos, MNN.TensorDescribe);
        $.parallel = reader.value(json.parallel, true);
        $.loopNumber = reader.value(json.loopNumber, 0);
        $.commands = reader.objects(json.commands, MNN.RegionCommand);
        $.initCommand = reader.objects(json.initCommand, MNN.RegionCommand);
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
            case 98: return MNN.AttentionParam.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'QuantizedAdd': return MNN.QuantizedAdd.decodeText(reader, json);
            case 'ArgMax': return MNN.ArgMax.decodeText(reader, json);
            case 'AsString': return MNN.AsString.decodeText(reader, json);
            case 'Axis': return MNN.Axis.decodeText(reader, json);
            case 'BatchNorm': return MNN.BatchNorm.decodeText(reader, json);
            case 'BinaryOp': return MNN.BinaryOp.decodeText(reader, json);
            case 'Blob': return MNN.Blob.decodeText(reader, json);
            case 'CastParam': return MNN.CastParam.decodeText(reader, json);
            case 'Convolution2D': return MNN.Convolution2D.decodeText(reader, json);
            case 'Crop': return MNN.Crop.decodeText(reader, json);
            case 'CropAndResize': return MNN.CropAndResize.decodeText(reader, json);
            case 'Dequantize': return MNN.Dequantize.decodeText(reader, json);
            case 'DetectionOutput': return MNN.DetectionOutput.decodeText(reader, json);
            case 'Eltwise': return MNN.Eltwise.decodeText(reader, json);
            case 'ExpandDims': return MNN.ExpandDims.decodeText(reader, json);
            case 'Fill': return MNN.Fill.decodeText(reader, json);
            case 'Flatten': return MNN.Flatten.decodeText(reader, json);
            case 'Gather': return MNN.Gather.decodeText(reader, json);
            case 'GatherV2': return MNN.GatherV2.decodeText(reader, json);
            case 'InnerProduct': return MNN.InnerProduct.decodeText(reader, json);
            case 'Input': return MNN.Input.decodeText(reader, json);
            case 'Interp': return MNN.Interp.decodeText(reader, json);
            case 'LRN': return MNN.LRN.decodeText(reader, json);
            case 'LSTM': return MNN.LSTM.decodeText(reader, json);
            case 'MatMul': return MNN.MatMul.decodeText(reader, json);
            case 'NonMaxSuppressionV2': return MNN.NonMaxSuppressionV2.decodeText(reader, json);
            case 'Normalize': return MNN.Normalize.decodeText(reader, json);
            case 'PackParam': return MNN.PackParam.decodeText(reader, json);
            case 'Permute': return MNN.Permute.decodeText(reader, json);
            case 'Plugin': return MNN.Plugin.decodeText(reader, json);
            case 'Pool': return MNN.Pool.decodeText(reader, json);
            case 'PRelu': return MNN.PRelu.decodeText(reader, json);
            case 'PriorBox': return MNN.PriorBox.decodeText(reader, json);
            case 'Proposal': return MNN.Proposal.decodeText(reader, json);
            case 'QuantizedAvgPool': return MNN.QuantizedAvgPool.decodeText(reader, json);
            case 'QuantizedBiasAdd': return MNN.QuantizedBiasAdd.decodeText(reader, json);
            case 'QuantizedConcat': return MNN.QuantizedConcat.decodeText(reader, json);
            case 'QuantizedLogistic': return MNN.QuantizedLogistic.decodeText(reader, json);
            case 'QuantizedMatMul': return MNN.QuantizedMatMul.decodeText(reader, json);
            case 'QuantizedMaxPool': return MNN.QuantizedMaxPool.decodeText(reader, json);
            case 'QuantizedRelu': return MNN.QuantizedRelu.decodeText(reader, json);
            case 'QuantizedRelu6': return MNN.QuantizedRelu6.decodeText(reader, json);
            case 'QuantizedReshape': return MNN.QuantizedReshape.decodeText(reader, json);
            case 'QuantizedSoftmax': return MNN.QuantizedSoftmax.decodeText(reader, json);
            case 'QuantizeMaxMin': return MNN.QuantizeMaxMin.decodeText(reader, json);
            case 'QuantizeV2': return MNN.QuantizeV2.decodeText(reader, json);
            case 'Range': return MNN.Range.decodeText(reader, json);
            case 'Rank': return MNN.Rank.decodeText(reader, json);
            case 'ReduceJoin': return MNN.ReduceJoin.decodeText(reader, json);
            case 'ReductionParam': return MNN.ReductionParam.decodeText(reader, json);
            case 'Relu': return MNN.Relu.decodeText(reader, json);
            case 'Relu6': return MNN.Relu6.decodeText(reader, json);
            case 'RequantizationRange': return MNN.RequantizationRange.decodeText(reader, json);
            case 'Requantize': return MNN.Requantize.decodeText(reader, json);
            case 'Reshape': return MNN.Reshape.decodeText(reader, json);
            case 'Resize': return MNN.Resize.decodeText(reader, json);
            case 'RoiParameters': return MNN.RoiParameters.decodeText(reader, json);
            case 'Scale': return MNN.Scale.decodeText(reader, json);
            case 'Selu': return MNN.Selu.decodeText(reader, json);
            case 'Size': return MNN.Size.decodeText(reader, json);
            case 'Slice': return MNN.Slice.decodeText(reader, json);
            case 'SliceTf': return MNN.SliceTf.decodeText(reader, json);
            case 'SpaceBatch': return MNN.SpaceBatch.decodeText(reader, json);
            case 'SqueezeParam': return MNN.SqueezeParam.decodeText(reader, json);
            case 'StridedSliceParam': return MNN.StridedSliceParam.decodeText(reader, json);
            case 'TensorConvertInfo': return MNN.TensorConvertInfo.decodeText(reader, json);
            case 'TfQuantizedConv2D': return MNN.TfQuantizedConv2D.decodeText(reader, json);
            case 'TopKV2': return MNN.TopKV2.decodeText(reader, json);
            case 'Transpose': return MNN.Transpose.decodeText(reader, json);
            case 'UnaryOp': return MNN.UnaryOp.decodeText(reader, json);
            case 'MomentsParam': return MNN.MomentsParam.decodeText(reader, json);
            case 'RNNParam': return MNN.RNNParam.decodeText(reader, json);
            case 'BatchMatMulParam': return MNN.BatchMatMulParam.decodeText(reader, json);
            case 'QuantizedFloatParam': return MNN.QuantizedFloatParam.decodeText(reader, json);
            case 'DepthSpaceParam': return MNN.DepthSpaceParam.decodeText(reader, json);
            case 'EltwiseInt8': return MNN.EltwiseInt8.decodeText(reader, json);
            case 'ReverseSequenceParam': return MNN.ReverseSequenceParam.decodeText(reader, json);
            case 'Extra': return MNN.Extra.decodeText(reader, json);
            case 'Pool3D': return MNN.Pool3D.decodeText(reader, json);
            case 'Convolution3D': return MNN.Convolution3D.decodeText(reader, json);
            case 'ELU': return MNN.ELU.decodeText(reader, json);
            case 'DetectionPostProcessParam': return MNN.DetectionPostProcessParam.decodeText(reader, json);
            case 'OneHotParam': return MNN.OneHotParam.decodeText(reader, json);
            case 'PadParam': return MNN.PadParam.decodeText(reader, json);
            case 'WhileParam': return MNN.WhileParam.decodeText(reader, json);
            case 'IfParam': return MNN.IfParam.decodeText(reader, json);
            case 'RandomUniform': return MNN.RandomUniform.decodeText(reader, json);
            case 'LayerNorm': return MNN.LayerNorm.decodeText(reader, json);
            case 'TensorArray': return MNN.TensorArray.decodeText(reader, json);
            case 'LSTMBlockCell': return MNN.LSTMBlockCell.decodeText(reader, json);
            case 'GridSample': return MNN.GridSample.decodeText(reader, json);
            case 'LoopParam': return MNN.LoopParam.decodeText(reader, json);
            case 'ImageProcessParam': return MNN.ImageProcessParam.decodeText(reader, json);
            case 'CumSum': return MNN.CumSum.decodeText(reader, json);
            case 'GroupNorm': return MNN.GroupNorm.decodeText(reader, json);
            case 'FmhaV2Param': return MNN.FmhaV2Param.decodeText(reader, json);
            case 'FmhcaParam': return MNN.FmhcaParam.decodeText(reader, json);
            case 'AttentionParam': return MNN.AttentionParam.decodeText(reader, json);
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

    static decodeText(reader, json) {
        const $ = new MNN.Op();
        $.inputIndexes = reader.array(json.inputIndexes, Int32Array);
        $.main = MNN.OpParameter.decodeText(reader, json.main, json.main_type);
        $.name = reader.value(json.name, null);
        $.outputIndexes = reader.array(json.outputIndexes, Int32Array);
        $.type = MNN.OpType[json.type];
        $.defaultDimentionFormat = MNN.MNN_DATA_FORMAT[json.defaultDimentionFormat];
        $.externalPath = reader.value(json.externalPath, null);
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

    static decodeText(reader, json) {
        const $ = new MNN.View();
        $.offset = reader.value(json.offset, 0);
        $.stride = reader.array(json.stride, Int32Array);
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

    static decodeText(reader, json) {
        const $ = new MNN.Region();
        $.src = reader.object(json.src, MNN.View);
        $.dst = reader.object(json.dst, MNN.View);
        $.size = reader.array(json.size, Int32Array);
        $.origin = reader.value(json.origin, 0);
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

    static decodeText(reader, json) {
        const $ = new MNN.TensorDescribe();
        $.blob = reader.object(json.blob, MNN.Blob);
        $.index = reader.value(json.index, 0);
        $.name = reader.value(json.name, null);
        $.regions = reader.objects(json.regions, MNN.Region);
        $.quantInfo = reader.object(json.quantInfo, MNN.TensorQuantInfo);
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

    static decodeText(reader, json) {
        const $ = new MNN.SubGraphProto();
        $.name = reader.value(json.name, null);
        $.inputs = reader.array(json.inputs, Int32Array);
        $.outputs = reader.array(json.outputs, Int32Array);
        $.tensors = reader.array(json.tensors);
        $.nodes = reader.objects(json.nodes, MNN.Op);
        $.extraTensorDescribe = reader.objects(json.extraTensorDescribe, MNN.TensorDescribe);
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

    static decodeText(reader, json) {
        const $ = new MNN.TensorQuantInfo();
        $.scale = reader.value(json.scale, 0);
        $.zero = reader.value(json.zero, 0);
        $.min = reader.value(json.min, -128);
        $.max = reader.value(json.max, 127);
        $.type = MNN.DataType[json.type];
        return $;
    }
};

MNN.Net = class Net {

    static create(reader) {
        return MNN.Net.decode(reader, reader.root);
    }

    static createText(reader) {
        return MNN.Net.decodeText(reader, reader.root);
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

    static decodeText(reader, json) {
        const $ = new MNN.Net();
        $.bizCode = reader.value(json.bizCode, null);
        $.extraTensorDescribe = reader.objects(json.extraTensorDescribe, MNN.TensorDescribe);
        $.extraInfo = reader.object(json.extraInfo, MNN.ExtraInfo);
        $.oplists = reader.objects(json.oplists, MNN.Op);
        $.outputName = reader.array(json.outputName);
        $.preferForwardType = MNN.ForwardType[json.preferForwardType];
        $.sourceType = MNN.NetSource[json.sourceType];
        $.tensorName = reader.array(json.tensorName);
        $.tensorNumber = reader.value(json.tensorNumber, 0);
        $.usage = MNN.Usage[json.usage];
        $.subgraphs = reader.objects(json.subgraphs, MNN.SubGraphProto);
        $.mnn_uuid = reader.value(json.mnn_uuid, null);
        return $;
    }
};
