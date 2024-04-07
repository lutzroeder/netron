
export const armnnSerializer = {};

armnnSerializer.ActivationFunction = {
    Sigmoid: 0,
    TanH: 1,
    Linear: 2,
    ReLu: 3,
    BoundedReLu: 4,
    SoftReLu: 5,
    LeakyReLu: 6,
    Abs: 7,
    Sqrt: 8,
    Square: 9,
    Elu: 10,
    HardSwish: 11
};

armnnSerializer.ArgMinMaxFunction = {
    Min: 0,
    Max: 1
};

armnnSerializer.DataType = {
    Float16: 0,
    Float32: 1,
    QuantisedAsymm8: 2,
    Signed32: 3,
    Boolean: 4,
    QuantisedSymm16: 5,
    QAsymmU8: 6,
    QSymmS16: 7,
    QAsymmS8: 8,
    QSymmS8: 9
};

armnnSerializer.DataLayout = {
    NHWC: 0,
    NCHW: 1
};

armnnSerializer.ResizeMethod = {
    NearestNeighbor: 0,
    Bilinear: 1
};

armnnSerializer.TensorInfo = class TensorInfo {

    static decode(reader, position) {
        const $ = new armnnSerializer.TensorInfo();
        $.dimensions = reader.array(position, 4, Uint32Array);
        $.dataType = reader.int8_(position, 6, 0);
        $.quantizationScale = reader.float32_(position, 8, 1);
        $.quantizationOffset = reader.int32_(position, 10, 0);
        $.quantizationScales = reader.array(position, 12, Float32Array);
        $.quantizationDim = reader.uint32_(position, 14, 0);
        $.dimensionality = reader.uint32_(position, 16, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.TensorInfo();
        $.dimensions = reader.array(json.dimensions, Uint32Array);
        $.dataType = armnnSerializer.DataType[json.dataType];
        $.quantizationScale = reader.value(json.quantizationScale, 1);
        $.quantizationOffset = reader.value(json.quantizationOffset, 0);
        $.quantizationScales = reader.array(json.quantizationScales, Float32Array);
        $.quantizationDim = reader.value(json.quantizationDim, 0);
        $.dimensionality = reader.value(json.dimensionality, 1);
        return $;
    }
};

armnnSerializer.Connection = class Connection {

    static decode(reader, position) {
        const $ = new armnnSerializer.Connection();
        $.sourceLayerIndex = reader.uint32(position + 0);
        $.outputSlotIndex = reader.uint32(position + 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.Connection();
        $.sourceLayerIndex = json.sourceLayerIndex;
        $.outputSlotIndex = json.outputSlotIndex;
        return $;
    }
};

armnnSerializer.ByteData = class ByteData {

    static decode(reader, position) {
        const $ = new armnnSerializer.ByteData();
        $.data = reader.array(position, 4, Int8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ByteData();
        $.data = reader.array(json.data, Int8Array);
        return $;
    }
};

armnnSerializer.ShortData = class ShortData {

    static decode(reader, position) {
        const $ = new armnnSerializer.ShortData();
        $.data = reader.array(position, 4, Int16Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ShortData();
        $.data = reader.array(json.data, Int16Array);
        return $;
    }
};

armnnSerializer.IntData = class IntData {

    static decode(reader, position) {
        const $ = new armnnSerializer.IntData();
        $.data = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.IntData();
        $.data = reader.array(json.data, Int32Array);
        return $;
    }
};

armnnSerializer.LongData = class LongData {

    static decode(reader, position) {
        const $ = new armnnSerializer.LongData();
        $.data = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LongData();
        $.data = reader.array(json.data);
        return $;
    }
};

armnnSerializer.ConstTensorData = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return armnnSerializer.ByteData.decode(reader, position);
            case 2: return armnnSerializer.ShortData.decode(reader, position);
            case 3: return armnnSerializer.IntData.decode(reader, position);
            case 4: return armnnSerializer.LongData.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'ByteData': return armnnSerializer.ByteData.decodeText(reader, json);
            case 'ShortData': return armnnSerializer.ShortData.decodeText(reader, json);
            case 'IntData': return armnnSerializer.IntData.decodeText(reader, json);
            case 'LongData': return armnnSerializer.LongData.decodeText(reader, json);
            default: return undefined;
        }
    }
};

armnnSerializer.ConstTensor = class ConstTensor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ConstTensor();
        $.info = reader.table(position, 4, armnnSerializer.TensorInfo);
        $.data = reader.union(position, 6, armnnSerializer.ConstTensorData);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ConstTensor();
        $.info = reader.object(json.info, armnnSerializer.TensorInfo);
        $.data = armnnSerializer.ConstTensorData.decodeText(reader, json.data, json.data_type);
        return $;
    }
};

armnnSerializer.InputSlot = class InputSlot {

    static decode(reader, position) {
        const $ = new armnnSerializer.InputSlot();
        $.index = reader.uint32_(position, 4, 0);
        $.connection = reader.struct(position, 6, armnnSerializer.Connection);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.InputSlot();
        $.index = reader.value(json.index, 0);
        $.connection = reader.object(json.connection, armnnSerializer.Connection);
        return $;
    }
};

armnnSerializer.OutputSlot = class OutputSlot {

    static decode(reader, position) {
        const $ = new armnnSerializer.OutputSlot();
        $.index = reader.uint32_(position, 4, 0);
        $.tensorInfo = reader.table(position, 6, armnnSerializer.TensorInfo);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.OutputSlot();
        $.index = reader.value(json.index, 0);
        $.tensorInfo = reader.object(json.tensorInfo, armnnSerializer.TensorInfo);
        return $;
    }
};

armnnSerializer.LayerType = {
    Addition: 0,
    Input: 1,
    Multiplication: 2,
    Output: 3,
    Pooling2d: 4,
    Reshape: 5,
    Softmax: 6,
    Convolution2d: 7,
    DepthwiseConvolution2d: 8,
    Activation: 9,
    Permute: 10,
    FullyConnected: 11,
    Constant: 12,
    SpaceToBatchNd: 13,
    BatchToSpaceNd: 14,
    Division: 15,
    Minimum: 16,
    Equal: 17,
    Maximum: 18,
    Normalization: 19,
    Pad: 20,
    Rsqrt: 21,
    Floor: 22,
    BatchNormalization: 23,
    Greater: 24,
    ResizeBilinear: 25,
    Subtraction: 26,
    StridedSlice: 27,
    Gather: 28,
    Mean: 29,
    Merger: 30,
    L2Normalization: 31,
    Splitter: 32,
    DetectionPostProcess: 33,
    Lstm: 34,
    Quantize: 35,
    Dequantize: 36,
    Merge: 37,
    Switch: 38,
    Concat: 39,
    SpaceToDepth: 40,
    Prelu: 41,
    TransposeConvolution2d: 42,
    Resize: 43,
    Stack: 44,
    QuantizedLstm: 45,
    Abs: 46,
    ArgMinMax: 47,
    Slice: 48,
    DepthToSpace: 49,
    InstanceNormalization: 50,
    LogSoftmax: 51,
    Comparison: 52,
    StandIn: 53,
    ElementwiseUnary: 54,
    Transpose: 55,
    QLstm: 56,
    Fill: 57,
    Rank: 58
};

armnnSerializer.LayerBase = class LayerBase {

    static decode(reader, position) {
        const $ = new armnnSerializer.LayerBase();
        $.index = reader.uint32_(position, 4, 0);
        $.layerName = reader.string_(position, 6, null);
        $.layerType = reader.uint32_(position, 8, 0);
        $.inputSlots = reader.tables(position, 10, armnnSerializer.InputSlot);
        $.outputSlots = reader.tables(position, 12, armnnSerializer.OutputSlot);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LayerBase();
        $.index = reader.value(json.index, 0);
        $.layerName = reader.value(json.layerName, null);
        $.layerType = armnnSerializer.LayerType[json.layerType];
        $.inputSlots = reader.objects(json.inputSlots, armnnSerializer.InputSlot);
        $.outputSlots = reader.objects(json.outputSlots, armnnSerializer.OutputSlot);
        return $;
    }
};

armnnSerializer.BindableLayerBase = class BindableLayerBase {

    static decode(reader, position) {
        const $ = new armnnSerializer.BindableLayerBase();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.layerBindingId = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.BindableLayerBase();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.layerBindingId = reader.value(json.layerBindingId, 0);
        return $;
    }
};

armnnSerializer.AbsLayer = class AbsLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.AbsLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.AbsLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.ActivationLayer = class ActivationLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ActivationLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ActivationDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ActivationLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ActivationDescriptor);
        return $;
    }
};

armnnSerializer.ActivationDescriptor = class ActivationDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ActivationDescriptor();
        $.activationFunction = reader.int8_(position, 4, 0);
        $.a = reader.float32_(position, 6, 0);
        $.b = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ActivationDescriptor();
        $.activationFunction = armnnSerializer.ActivationFunction[json.activationFunction];
        $.a = reader.value(json.a, 0);
        $.b = reader.value(json.b, 0);
        return $;
    }
};

armnnSerializer.AdditionLayer = class AdditionLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.AdditionLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.AdditionLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.ArgMinMaxLayer = class ArgMinMaxLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ArgMinMaxLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ArgMinMaxDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ArgMinMaxLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ArgMinMaxDescriptor);
        return $;
    }
};

armnnSerializer.ArgMinMaxDescriptor = class ArgMinMaxDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ArgMinMaxDescriptor();
        $.argMinMaxFunction = reader.int8_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ArgMinMaxDescriptor();
        $.argMinMaxFunction = armnnSerializer.ArgMinMaxFunction[json.argMinMaxFunction];
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

armnnSerializer.ComparisonOperation = {
    Equal: 0,
    Greater: 1,
    GreaterOrEqual: 2,
    Less: 3,
    LessOrEqual: 4,
    NotEqual: 5
};

armnnSerializer.ComparisonDescriptor = class ComparisonDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ComparisonDescriptor();
        $.operation = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ComparisonDescriptor();
        $.operation = armnnSerializer.ComparisonOperation[json.operation];
        return $;
    }
};

armnnSerializer.ComparisonLayer = class ComparisonLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ComparisonLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ComparisonDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ComparisonLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ComparisonDescriptor);
        return $;
    }
};

armnnSerializer.ConstantLayer = class ConstantLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ConstantLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.input = reader.table(position, 6, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ConstantLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.input = reader.object(json.input, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.Convolution2dLayer = class Convolution2dLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.Convolution2dLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.Convolution2dDescriptor);
        $.weights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.biases = reader.table(position, 10, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.Convolution2dLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.Convolution2dDescriptor);
        $.weights = reader.object(json.weights, armnnSerializer.ConstTensor);
        $.biases = reader.object(json.biases, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.Convolution2dDescriptor = class Convolution2dDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.Convolution2dDescriptor();
        $.padLeft = reader.uint32_(position, 4, 0);
        $.padRight = reader.uint32_(position, 6, 0);
        $.padTop = reader.uint32_(position, 8, 0);
        $.padBottom = reader.uint32_(position, 10, 0);
        $.strideX = reader.uint32_(position, 12, 0);
        $.strideY = reader.uint32_(position, 14, 0);
        $.dilationX = reader.uint32_(position, 16, 1);
        $.dilationY = reader.uint32_(position, 18, 1);
        $.biasEnabled = reader.bool_(position, 20, false);
        $.dataLayout = reader.int8_(position, 22, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.Convolution2dDescriptor();
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.padTop = reader.value(json.padTop, 0);
        $.padBottom = reader.value(json.padBottom, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.dilationX = reader.value(json.dilationX, 1);
        $.dilationY = reader.value(json.dilationY, 1);
        $.biasEnabled = reader.value(json.biasEnabled, false);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.DepthToSpaceLayer = class DepthToSpaceLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.DepthToSpaceLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.DepthToSpaceDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DepthToSpaceLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.DepthToSpaceDescriptor);
        return $;
    }
};

armnnSerializer.DepthToSpaceDescriptor = class DepthToSpaceDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.DepthToSpaceDescriptor();
        $.blockSize = reader.uint32_(position, 4, 0);
        $.dataLayout = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DepthToSpaceDescriptor();
        $.blockSize = reader.value(json.blockSize, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.DivisionLayer = class DivisionLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.DivisionLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DivisionLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.UnaryOperation = {
    Abs: 0,
    Rsqrt: 1,
    Sqrt: 2,
    Exp: 3,
    Neg: 4
};

armnnSerializer.ElementwiseUnaryDescriptor = class ElementwiseUnaryDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ElementwiseUnaryDescriptor();
        $.operation = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ElementwiseUnaryDescriptor();
        $.operation = armnnSerializer.UnaryOperation[json.operation];
        return $;
    }
};

armnnSerializer.ElementwiseUnaryLayer = class ElementwiseUnaryLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ElementwiseUnaryLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ElementwiseUnaryDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ElementwiseUnaryLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ElementwiseUnaryDescriptor);
        return $;
    }
};

armnnSerializer.EqualLayer = class EqualLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.EqualLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.EqualLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.FillLayer = class FillLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.FillLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.FillDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.FillLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.FillDescriptor);
        return $;
    }
};

armnnSerializer.FillDescriptor = class FillDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.FillDescriptor();
        $.value = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.FillDescriptor();
        $.value = reader.value(json.value, 0);
        return $;
    }
};

armnnSerializer.FloorLayer = class FloorLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.FloorLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.FloorLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.FullyConnectedLayer = class FullyConnectedLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.FullyConnectedLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.FullyConnectedDescriptor);
        $.weights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.biases = reader.table(position, 10, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.FullyConnectedLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.FullyConnectedDescriptor);
        $.weights = reader.object(json.weights, armnnSerializer.ConstTensor);
        $.biases = reader.object(json.biases, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.FullyConnectedDescriptor = class FullyConnectedDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.FullyConnectedDescriptor();
        $.biasEnabled = reader.bool_(position, 4, false);
        $.transposeWeightsMatrix = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.FullyConnectedDescriptor();
        $.biasEnabled = reader.value(json.biasEnabled, false);
        $.transposeWeightsMatrix = reader.value(json.transposeWeightsMatrix, false);
        return $;
    }
};

armnnSerializer.GatherLayer = class GatherLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.GatherLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.GatherDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.GatherLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.GatherDescriptor);
        return $;
    }
};

armnnSerializer.GatherDescriptor = class GatherDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.GatherDescriptor();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.GatherDescriptor();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

armnnSerializer.GreaterLayer = class GreaterLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.GreaterLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.GreaterLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.InputLayer = class InputLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.InputLayer();
        $.base = reader.table(position, 4, armnnSerializer.BindableLayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.InputLayer();
        $.base = reader.object(json.base, armnnSerializer.BindableLayerBase);
        return $;
    }
};

armnnSerializer.InstanceNormalizationLayer = class InstanceNormalizationLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.InstanceNormalizationLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.InstanceNormalizationDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.InstanceNormalizationLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.InstanceNormalizationDescriptor);
        return $;
    }
};

armnnSerializer.InstanceNormalizationDescriptor = class InstanceNormalizationDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.InstanceNormalizationDescriptor();
        $.gamma = reader.float32_(position, 4, 0);
        $.beta = reader.float32_(position, 6, 0);
        $.eps = reader.float32_(position, 8, 0);
        $.dataLayout = reader.int8_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.InstanceNormalizationDescriptor();
        $.gamma = reader.value(json.gamma, 0);
        $.beta = reader.value(json.beta, 0);
        $.eps = reader.value(json.eps, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.LogSoftmaxLayer = class LogSoftmaxLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.LogSoftmaxLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.LogSoftmaxDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LogSoftmaxLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.LogSoftmaxDescriptor);
        return $;
    }
};

armnnSerializer.LogSoftmaxDescriptor = class LogSoftmaxDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.LogSoftmaxDescriptor();
        $.beta = reader.float32_(position, 4, 1);
        $.axis = reader.int32_(position, 6, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LogSoftmaxDescriptor();
        $.beta = reader.value(json.beta, 1);
        $.axis = reader.value(json.axis, -1);
        return $;
    }
};

armnnSerializer.L2NormalizationLayer = class L2NormalizationLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.L2NormalizationLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.L2NormalizationDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.L2NormalizationLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.L2NormalizationDescriptor);
        return $;
    }
};

armnnSerializer.L2NormalizationDescriptor = class L2NormalizationDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.L2NormalizationDescriptor();
        $.dataLayout = reader.int8_(position, 4, 1);
        $.eps = reader.float32_(position, 6, 1e-12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.L2NormalizationDescriptor();
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        $.eps = reader.value(json.eps, 1e-12);
        return $;
    }
};

armnnSerializer.MinimumLayer = class MinimumLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.MinimumLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MinimumLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.MaximumLayer = class MaximumLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.MaximumLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MaximumLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.MultiplicationLayer = class MultiplicationLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.MultiplicationLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MultiplicationLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.Pooling2dLayer = class Pooling2dLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.Pooling2dLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.Pooling2dDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.Pooling2dLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.Pooling2dDescriptor);
        return $;
    }
};

armnnSerializer.PoolingAlgorithm = {
    Max: 0,
    Average: 1,
    L2: 2
};

armnnSerializer.OutputShapeRounding = {
    Floor: 0,
    Ceiling: 1
};

armnnSerializer.PaddingMethod = {
    IgnoreValue: 0,
    Exclude: 1
};

armnnSerializer.Pooling2dDescriptor = class Pooling2dDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.Pooling2dDescriptor();
        $.poolType = reader.int8_(position, 4, 0);
        $.padLeft = reader.uint32_(position, 6, 0);
        $.padRight = reader.uint32_(position, 8, 0);
        $.padTop = reader.uint32_(position, 10, 0);
        $.padBottom = reader.uint32_(position, 12, 0);
        $.poolWidth = reader.uint32_(position, 14, 0);
        $.poolHeight = reader.uint32_(position, 16, 0);
        $.strideX = reader.uint32_(position, 18, 0);
        $.strideY = reader.uint32_(position, 20, 0);
        $.outputShapeRounding = reader.int8_(position, 22, 0);
        $.paddingMethod = reader.int8_(position, 24, 0);
        $.dataLayout = reader.int8_(position, 26, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.Pooling2dDescriptor();
        $.poolType = armnnSerializer.PoolingAlgorithm[json.poolType];
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.padTop = reader.value(json.padTop, 0);
        $.padBottom = reader.value(json.padBottom, 0);
        $.poolWidth = reader.value(json.poolWidth, 0);
        $.poolHeight = reader.value(json.poolHeight, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.outputShapeRounding = armnnSerializer.OutputShapeRounding[json.outputShapeRounding];
        $.paddingMethod = armnnSerializer.PaddingMethod[json.paddingMethod];
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.QuantizeLayer = class QuantizeLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.QuantizeLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.QuantizeLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.SoftmaxLayer = class SoftmaxLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SoftmaxLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.SoftmaxDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SoftmaxLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.SoftmaxDescriptor);
        return $;
    }
};

armnnSerializer.SoftmaxDescriptor = class SoftmaxDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.SoftmaxDescriptor();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SoftmaxDescriptor();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

armnnSerializer.DepthwiseConvolution2dLayer = class DepthwiseConvolution2dLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.DepthwiseConvolution2dLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.DepthwiseConvolution2dDescriptor);
        $.weights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.biases = reader.table(position, 10, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DepthwiseConvolution2dLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.DepthwiseConvolution2dDescriptor);
        $.weights = reader.object(json.weights, armnnSerializer.ConstTensor);
        $.biases = reader.object(json.biases, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.DepthwiseConvolution2dDescriptor = class DepthwiseConvolution2dDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.DepthwiseConvolution2dDescriptor();
        $.padLeft = reader.uint32_(position, 4, 0);
        $.padRight = reader.uint32_(position, 6, 0);
        $.padTop = reader.uint32_(position, 8, 0);
        $.padBottom = reader.uint32_(position, 10, 0);
        $.strideX = reader.uint32_(position, 12, 0);
        $.strideY = reader.uint32_(position, 14, 0);
        $.dilationX = reader.uint32_(position, 16, 1);
        $.dilationY = reader.uint32_(position, 18, 1);
        $.biasEnabled = reader.bool_(position, 20, false);
        $.dataLayout = reader.int8_(position, 22, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DepthwiseConvolution2dDescriptor();
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.padTop = reader.value(json.padTop, 0);
        $.padBottom = reader.value(json.padBottom, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.dilationX = reader.value(json.dilationX, 1);
        $.dilationY = reader.value(json.dilationY, 1);
        $.biasEnabled = reader.value(json.biasEnabled, false);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.OutputLayer = class OutputLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.OutputLayer();
        $.base = reader.table(position, 4, armnnSerializer.BindableLayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.OutputLayer();
        $.base = reader.object(json.base, armnnSerializer.BindableLayerBase);
        return $;
    }
};

armnnSerializer.ReshapeLayer = class ReshapeLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ReshapeLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ReshapeDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ReshapeLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ReshapeDescriptor);
        return $;
    }
};

armnnSerializer.ReshapeDescriptor = class ReshapeDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ReshapeDescriptor();
        $.targetShape = reader.array(position, 4, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ReshapeDescriptor();
        $.targetShape = reader.array(json.targetShape, Uint32Array);
        return $;
    }
};

armnnSerializer.PermuteLayer = class PermuteLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.PermuteLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.PermuteDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.PermuteLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.PermuteDescriptor);
        return $;
    }
};

armnnSerializer.PermuteDescriptor = class PermuteDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.PermuteDescriptor();
        $.dimMappings = reader.array(position, 4, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.PermuteDescriptor();
        $.dimMappings = reader.array(json.dimMappings, Uint32Array);
        return $;
    }
};

armnnSerializer.SpaceToBatchNdLayer = class SpaceToBatchNdLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SpaceToBatchNdLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.SpaceToBatchNdDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SpaceToBatchNdLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.SpaceToBatchNdDescriptor);
        return $;
    }
};

armnnSerializer.SpaceToBatchNdDescriptor = class SpaceToBatchNdDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.SpaceToBatchNdDescriptor();
        $.blockShape = reader.array(position, 4, Uint32Array);
        $.padList = reader.array(position, 6, Uint32Array);
        $.dataLayout = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SpaceToBatchNdDescriptor();
        $.blockShape = reader.array(json.blockShape, Uint32Array);
        $.padList = reader.array(json.padList, Uint32Array);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.SpaceToDepthLayer = class SpaceToDepthLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SpaceToDepthLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.SpaceToDepthDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SpaceToDepthLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.SpaceToDepthDescriptor);
        return $;
    }
};

armnnSerializer.SpaceToDepthDescriptor = class SpaceToDepthDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.SpaceToDepthDescriptor();
        $.blockSize = reader.uint32_(position, 4, 0);
        $.dataLayout = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SpaceToDepthDescriptor();
        $.blockSize = reader.value(json.blockSize, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.SubtractionLayer = class SubtractionLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SubtractionLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SubtractionLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.BatchToSpaceNdLayer = class BatchToSpaceNdLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.BatchToSpaceNdLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.BatchToSpaceNdDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.BatchToSpaceNdLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.BatchToSpaceNdDescriptor);
        return $;
    }
};

armnnSerializer.BatchToSpaceNdDescriptor = class BatchToSpaceNdDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.BatchToSpaceNdDescriptor();
        $.blockShape = reader.array(position, 4, Uint32Array);
        $.crops = reader.array(position, 6, Uint32Array);
        $.dataLayout = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.BatchToSpaceNdDescriptor();
        $.blockShape = reader.array(json.blockShape, Uint32Array);
        $.crops = reader.array(json.crops, Uint32Array);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.NormalizationAlgorithmChannel = {
    Across: 0,
    Within: 1
};

armnnSerializer.NormalizationAlgorithmMethod = {
    LocalBrightness: 0,
    LocalContrast: 1
};

armnnSerializer.NormalizationLayer = class NormalizationLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.NormalizationLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.NormalizationDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.NormalizationLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.NormalizationDescriptor);
        return $;
    }
};

armnnSerializer.NormalizationDescriptor = class NormalizationDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.NormalizationDescriptor();
        $.normChannelType = reader.int8_(position, 4, 0);
        $.normMethodType = reader.int8_(position, 6, 0);
        $.normSize = reader.uint32_(position, 8, 0);
        $.alpha = reader.float32_(position, 10, 0);
        $.beta = reader.float32_(position, 12, 0);
        $.k = reader.float32_(position, 14, 0);
        $.dataLayout = reader.int8_(position, 16, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.NormalizationDescriptor();
        $.normChannelType = armnnSerializer.NormalizationAlgorithmChannel[json.normChannelType];
        $.normMethodType = armnnSerializer.NormalizationAlgorithmMethod[json.normMethodType];
        $.normSize = reader.value(json.normSize, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
        $.k = reader.value(json.k, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.MeanLayer = class MeanLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.MeanLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.MeanDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MeanLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.MeanDescriptor);
        return $;
    }
};

armnnSerializer.MeanDescriptor = class MeanDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.MeanDescriptor();
        $.axis = reader.array(position, 4, Uint32Array);
        $.keepDims = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MeanDescriptor();
        $.axis = reader.array(json.axis, Uint32Array);
        $.keepDims = reader.value(json.keepDims, false);
        return $;
    }
};

armnnSerializer.PadLayer = class PadLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.PadLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.PadDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.PadLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.PadDescriptor);
        return $;
    }
};

armnnSerializer.PadDescriptor = class PadDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.PadDescriptor();
        $.padList = reader.array(position, 4, Uint32Array);
        $.padValue = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.PadDescriptor();
        $.padList = reader.array(json.padList, Uint32Array);
        $.padValue = reader.value(json.padValue, 0);
        return $;
    }
};

armnnSerializer.RsqrtLayer = class RsqrtLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.RsqrtLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.RsqrtLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.BatchNormalizationLayer = class BatchNormalizationLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.BatchNormalizationLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.BatchNormalizationDescriptor);
        $.mean = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.variance = reader.table(position, 10, armnnSerializer.ConstTensor);
        $.beta = reader.table(position, 12, armnnSerializer.ConstTensor);
        $.gamma = reader.table(position, 14, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.BatchNormalizationLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.BatchNormalizationDescriptor);
        $.mean = reader.object(json.mean, armnnSerializer.ConstTensor);
        $.variance = reader.object(json.variance, armnnSerializer.ConstTensor);
        $.beta = reader.object(json.beta, armnnSerializer.ConstTensor);
        $.gamma = reader.object(json.gamma, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.BatchNormalizationDescriptor = class BatchNormalizationDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.BatchNormalizationDescriptor();
        $.eps = reader.float32_(position, 4, 0);
        $.dataLayout = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.BatchNormalizationDescriptor();
        $.eps = reader.value(json.eps, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.ResizeBilinearLayer = class ResizeBilinearLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ResizeBilinearLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ResizeBilinearDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ResizeBilinearLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ResizeBilinearDescriptor);
        return $;
    }
};

armnnSerializer.ResizeBilinearDescriptor = class ResizeBilinearDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ResizeBilinearDescriptor();
        $.targetWidth = reader.uint32_(position, 4, 0);
        $.targetHeight = reader.uint32_(position, 6, 0);
        $.dataLayout = reader.int8_(position, 8, 0);
        $.alignCorners = reader.bool_(position, 10, false);
        $.halfPixelCenters = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ResizeBilinearDescriptor();
        $.targetWidth = reader.value(json.targetWidth, 0);
        $.targetHeight = reader.value(json.targetHeight, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        $.alignCorners = reader.value(json.alignCorners, false);
        $.halfPixelCenters = reader.value(json.halfPixelCenters, false);
        return $;
    }
};

armnnSerializer.SliceLayer = class SliceLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SliceLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.SliceDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SliceLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.SliceDescriptor);
        return $;
    }
};

armnnSerializer.SliceDescriptor = class SliceDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.SliceDescriptor();
        $.begin = reader.array(position, 4, Uint32Array);
        $.size = reader.array(position, 6, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SliceDescriptor();
        $.begin = reader.array(json.begin, Uint32Array);
        $.size = reader.array(json.size, Uint32Array);
        return $;
    }
};

armnnSerializer.StridedSliceLayer = class StridedSliceLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.StridedSliceLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.StridedSliceDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.StridedSliceLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.StridedSliceDescriptor);
        return $;
    }
};

armnnSerializer.StridedSliceDescriptor = class StridedSliceDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.StridedSliceDescriptor();
        $.begin = reader.array(position, 4, Int32Array);
        $.end = reader.array(position, 6, Int32Array);
        $.stride = reader.array(position, 8, Int32Array);
        $.beginMask = reader.int32_(position, 10, 0);
        $.endMask = reader.int32_(position, 12, 0);
        $.shrinkAxisMask = reader.int32_(position, 14, 0);
        $.ellipsisMask = reader.int32_(position, 16, 0);
        $.newAxisMask = reader.int32_(position, 18, 0);
        $.dataLayout = reader.int8_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.StridedSliceDescriptor();
        $.begin = reader.array(json.begin, Int32Array);
        $.end = reader.array(json.end, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.beginMask = reader.value(json.beginMask, 0);
        $.endMask = reader.value(json.endMask, 0);
        $.shrinkAxisMask = reader.value(json.shrinkAxisMask, 0);
        $.ellipsisMask = reader.value(json.ellipsisMask, 0);
        $.newAxisMask = reader.value(json.newAxisMask, 0);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.ConcatLayer = class ConcatLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ConcatLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.OriginsDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ConcatLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.OriginsDescriptor);
        return $;
    }
};

armnnSerializer.MergerLayer = class MergerLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.MergerLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.OriginsDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MergerLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.OriginsDescriptor);
        return $;
    }
};

armnnSerializer.UintVector = class UintVector {

    static decode(reader, position) {
        const $ = new armnnSerializer.UintVector();
        $.data = reader.array(position, 4, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.UintVector();
        $.data = reader.array(json.data, Uint32Array);
        return $;
    }
};

armnnSerializer.OriginsDescriptor = class OriginsDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.OriginsDescriptor();
        $.concatAxis = reader.uint32_(position, 4, 0);
        $.numViews = reader.uint32_(position, 6, 0);
        $.numDimensions = reader.uint32_(position, 8, 0);
        $.viewOrigins = reader.tables(position, 10, armnnSerializer.UintVector);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.OriginsDescriptor();
        $.concatAxis = reader.value(json.concatAxis, 0);
        $.numViews = reader.value(json.numViews, 0);
        $.numDimensions = reader.value(json.numDimensions, 0);
        $.viewOrigins = reader.objects(json.viewOrigins, armnnSerializer.UintVector);
        return $;
    }
};

armnnSerializer.ViewsDescriptor = class ViewsDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ViewsDescriptor();
        $.origins = reader.table(position, 4, armnnSerializer.OriginsDescriptor);
        $.viewSizes = reader.tables(position, 6, armnnSerializer.UintVector);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ViewsDescriptor();
        $.origins = reader.object(json.origins, armnnSerializer.OriginsDescriptor);
        $.viewSizes = reader.objects(json.viewSizes, armnnSerializer.UintVector);
        return $;
    }
};

armnnSerializer.SplitterLayer = class SplitterLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SplitterLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ViewsDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SplitterLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ViewsDescriptor);
        return $;
    }
};

armnnSerializer.DetectionPostProcessLayer = class DetectionPostProcessLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.DetectionPostProcessLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.DetectionPostProcessDescriptor);
        $.anchors = reader.table(position, 8, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DetectionPostProcessLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.DetectionPostProcessDescriptor);
        $.anchors = reader.object(json.anchors, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.DetectionPostProcessDescriptor = class DetectionPostProcessDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.DetectionPostProcessDescriptor();
        $.maxDetections = reader.uint32_(position, 4, 0);
        $.maxClassesPerDetection = reader.uint32_(position, 6, 0);
        $.detectionsPerClass = reader.uint32_(position, 8, 0);
        $.nmsScoreThreshold = reader.float32_(position, 10, 0);
        $.nmsIouThreshold = reader.float32_(position, 12, 0);
        $.numClasses = reader.uint32_(position, 14, 0);
        $.useRegularNms = reader.bool_(position, 16, false);
        $.scaleX = reader.float32_(position, 18, 0);
        $.scaleY = reader.float32_(position, 20, 0);
        $.scaleW = reader.float32_(position, 22, 0);
        $.scaleH = reader.float32_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DetectionPostProcessDescriptor();
        $.maxDetections = reader.value(json.maxDetections, 0);
        $.maxClassesPerDetection = reader.value(json.maxClassesPerDetection, 0);
        $.detectionsPerClass = reader.value(json.detectionsPerClass, 0);
        $.nmsScoreThreshold = reader.value(json.nmsScoreThreshold, 0);
        $.nmsIouThreshold = reader.value(json.nmsIouThreshold, 0);
        $.numClasses = reader.value(json.numClasses, 0);
        $.useRegularNms = reader.value(json.useRegularNms, false);
        $.scaleX = reader.value(json.scaleX, 0);
        $.scaleY = reader.value(json.scaleY, 0);
        $.scaleW = reader.value(json.scaleW, 0);
        $.scaleH = reader.value(json.scaleH, 0);
        return $;
    }
};

armnnSerializer.LstmInputParams = class LstmInputParams {

    static decode(reader, position) {
        const $ = new armnnSerializer.LstmInputParams();
        $.inputToForgetWeights = reader.table(position, 4, armnnSerializer.ConstTensor);
        $.inputToCellWeights = reader.table(position, 6, armnnSerializer.ConstTensor);
        $.inputToOutputWeights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.recurrentToForgetWeights = reader.table(position, 10, armnnSerializer.ConstTensor);
        $.recurrentToCellWeights = reader.table(position, 12, armnnSerializer.ConstTensor);
        $.recurrentToOutputWeights = reader.table(position, 14, armnnSerializer.ConstTensor);
        $.forgetGateBias = reader.table(position, 16, armnnSerializer.ConstTensor);
        $.cellBias = reader.table(position, 18, armnnSerializer.ConstTensor);
        $.outputGateBias = reader.table(position, 20, armnnSerializer.ConstTensor);
        $.inputToInputWeights = reader.table(position, 22, armnnSerializer.ConstTensor);
        $.recurrentToInputWeights = reader.table(position, 24, armnnSerializer.ConstTensor);
        $.cellToInputWeights = reader.table(position, 26, armnnSerializer.ConstTensor);
        $.inputGateBias = reader.table(position, 28, armnnSerializer.ConstTensor);
        $.projectionWeights = reader.table(position, 30, armnnSerializer.ConstTensor);
        $.projectionBias = reader.table(position, 32, armnnSerializer.ConstTensor);
        $.cellToForgetWeights = reader.table(position, 34, armnnSerializer.ConstTensor);
        $.cellToOutputWeights = reader.table(position, 36, armnnSerializer.ConstTensor);
        $.inputLayerNormWeights = reader.table(position, 38, armnnSerializer.ConstTensor);
        $.forgetLayerNormWeights = reader.table(position, 40, armnnSerializer.ConstTensor);
        $.cellLayerNormWeights = reader.table(position, 42, armnnSerializer.ConstTensor);
        $.outputLayerNormWeights = reader.table(position, 44, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LstmInputParams();
        $.inputToForgetWeights = reader.object(json.inputToForgetWeights, armnnSerializer.ConstTensor);
        $.inputToCellWeights = reader.object(json.inputToCellWeights, armnnSerializer.ConstTensor);
        $.inputToOutputWeights = reader.object(json.inputToOutputWeights, armnnSerializer.ConstTensor);
        $.recurrentToForgetWeights = reader.object(json.recurrentToForgetWeights, armnnSerializer.ConstTensor);
        $.recurrentToCellWeights = reader.object(json.recurrentToCellWeights, armnnSerializer.ConstTensor);
        $.recurrentToOutputWeights = reader.object(json.recurrentToOutputWeights, armnnSerializer.ConstTensor);
        $.forgetGateBias = reader.object(json.forgetGateBias, armnnSerializer.ConstTensor);
        $.cellBias = reader.object(json.cellBias, armnnSerializer.ConstTensor);
        $.outputGateBias = reader.object(json.outputGateBias, armnnSerializer.ConstTensor);
        $.inputToInputWeights = reader.object(json.inputToInputWeights, armnnSerializer.ConstTensor);
        $.recurrentToInputWeights = reader.object(json.recurrentToInputWeights, armnnSerializer.ConstTensor);
        $.cellToInputWeights = reader.object(json.cellToInputWeights, armnnSerializer.ConstTensor);
        $.inputGateBias = reader.object(json.inputGateBias, armnnSerializer.ConstTensor);
        $.projectionWeights = reader.object(json.projectionWeights, armnnSerializer.ConstTensor);
        $.projectionBias = reader.object(json.projectionBias, armnnSerializer.ConstTensor);
        $.cellToForgetWeights = reader.object(json.cellToForgetWeights, armnnSerializer.ConstTensor);
        $.cellToOutputWeights = reader.object(json.cellToOutputWeights, armnnSerializer.ConstTensor);
        $.inputLayerNormWeights = reader.object(json.inputLayerNormWeights, armnnSerializer.ConstTensor);
        $.forgetLayerNormWeights = reader.object(json.forgetLayerNormWeights, armnnSerializer.ConstTensor);
        $.cellLayerNormWeights = reader.object(json.cellLayerNormWeights, armnnSerializer.ConstTensor);
        $.outputLayerNormWeights = reader.object(json.outputLayerNormWeights, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.LstmDescriptor = class LstmDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.LstmDescriptor();
        $.activationFunc = reader.uint32_(position, 4, 0);
        $.clippingThresCell = reader.float32_(position, 6, 0);
        $.clippingThresProj = reader.float32_(position, 8, 0);
        $.cifgEnabled = reader.bool_(position, 10, true);
        $.peepholeEnabled = reader.bool_(position, 12, false);
        $.projectionEnabled = reader.bool_(position, 14, false);
        $.layerNormEnabled = reader.bool_(position, 16, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LstmDescriptor();
        $.activationFunc = reader.value(json.activationFunc, 0);
        $.clippingThresCell = reader.value(json.clippingThresCell, 0);
        $.clippingThresProj = reader.value(json.clippingThresProj, 0);
        $.cifgEnabled = reader.value(json.cifgEnabled, true);
        $.peepholeEnabled = reader.value(json.peepholeEnabled, false);
        $.projectionEnabled = reader.value(json.projectionEnabled, false);
        $.layerNormEnabled = reader.value(json.layerNormEnabled, false);
        return $;
    }
};

armnnSerializer.LstmLayer = class LstmLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.LstmLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.LstmDescriptor);
        $.inputParams = reader.table(position, 8, armnnSerializer.LstmInputParams);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.LstmLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.LstmDescriptor);
        $.inputParams = reader.object(json.inputParams, armnnSerializer.LstmInputParams);
        return $;
    }
};

armnnSerializer.QLstmInputParams = class QLstmInputParams {

    static decode(reader, position) {
        const $ = new armnnSerializer.QLstmInputParams();
        $.inputToForgetWeights = reader.table(position, 4, armnnSerializer.ConstTensor);
        $.inputToCellWeights = reader.table(position, 6, armnnSerializer.ConstTensor);
        $.inputToOutputWeights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.recurrentToForgetWeights = reader.table(position, 10, armnnSerializer.ConstTensor);
        $.recurrentToCellWeights = reader.table(position, 12, armnnSerializer.ConstTensor);
        $.recurrentToOutputWeights = reader.table(position, 14, armnnSerializer.ConstTensor);
        $.forgetGateBias = reader.table(position, 16, armnnSerializer.ConstTensor);
        $.cellBias = reader.table(position, 18, armnnSerializer.ConstTensor);
        $.outputGateBias = reader.table(position, 20, armnnSerializer.ConstTensor);
        $.inputToInputWeights = reader.table(position, 22, armnnSerializer.ConstTensor);
        $.recurrentToInputWeights = reader.table(position, 24, armnnSerializer.ConstTensor);
        $.inputGateBias = reader.table(position, 26, armnnSerializer.ConstTensor);
        $.projectionWeights = reader.table(position, 28, armnnSerializer.ConstTensor);
        $.projectionBias = reader.table(position, 30, armnnSerializer.ConstTensor);
        $.cellToInputWeights = reader.table(position, 32, armnnSerializer.ConstTensor);
        $.cellToForgetWeights = reader.table(position, 34, armnnSerializer.ConstTensor);
        $.cellToOutputWeights = reader.table(position, 36, armnnSerializer.ConstTensor);
        $.inputLayerNormWeights = reader.table(position, 38, armnnSerializer.ConstTensor);
        $.forgetLayerNormWeights = reader.table(position, 40, armnnSerializer.ConstTensor);
        $.cellLayerNormWeights = reader.table(position, 42, armnnSerializer.ConstTensor);
        $.outputLayerNormWeights = reader.table(position, 44, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.QLstmInputParams();
        $.inputToForgetWeights = reader.object(json.inputToForgetWeights, armnnSerializer.ConstTensor);
        $.inputToCellWeights = reader.object(json.inputToCellWeights, armnnSerializer.ConstTensor);
        $.inputToOutputWeights = reader.object(json.inputToOutputWeights, armnnSerializer.ConstTensor);
        $.recurrentToForgetWeights = reader.object(json.recurrentToForgetWeights, armnnSerializer.ConstTensor);
        $.recurrentToCellWeights = reader.object(json.recurrentToCellWeights, armnnSerializer.ConstTensor);
        $.recurrentToOutputWeights = reader.object(json.recurrentToOutputWeights, armnnSerializer.ConstTensor);
        $.forgetGateBias = reader.object(json.forgetGateBias, armnnSerializer.ConstTensor);
        $.cellBias = reader.object(json.cellBias, armnnSerializer.ConstTensor);
        $.outputGateBias = reader.object(json.outputGateBias, armnnSerializer.ConstTensor);
        $.inputToInputWeights = reader.object(json.inputToInputWeights, armnnSerializer.ConstTensor);
        $.recurrentToInputWeights = reader.object(json.recurrentToInputWeights, armnnSerializer.ConstTensor);
        $.inputGateBias = reader.object(json.inputGateBias, armnnSerializer.ConstTensor);
        $.projectionWeights = reader.object(json.projectionWeights, armnnSerializer.ConstTensor);
        $.projectionBias = reader.object(json.projectionBias, armnnSerializer.ConstTensor);
        $.cellToInputWeights = reader.object(json.cellToInputWeights, armnnSerializer.ConstTensor);
        $.cellToForgetWeights = reader.object(json.cellToForgetWeights, armnnSerializer.ConstTensor);
        $.cellToOutputWeights = reader.object(json.cellToOutputWeights, armnnSerializer.ConstTensor);
        $.inputLayerNormWeights = reader.object(json.inputLayerNormWeights, armnnSerializer.ConstTensor);
        $.forgetLayerNormWeights = reader.object(json.forgetLayerNormWeights, armnnSerializer.ConstTensor);
        $.cellLayerNormWeights = reader.object(json.cellLayerNormWeights, armnnSerializer.ConstTensor);
        $.outputLayerNormWeights = reader.object(json.outputLayerNormWeights, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.QLstmDescriptor = class QLstmDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.QLstmDescriptor();
        $.cifgEnabled = reader.bool_(position, 4, true);
        $.peepholeEnabled = reader.bool_(position, 6, false);
        $.projectionEnabled = reader.bool_(position, 8, false);
        $.layerNormEnabled = reader.bool_(position, 10, false);
        $.cellClip = reader.float32_(position, 12, 0);
        $.projectionClip = reader.float32_(position, 14, 0);
        $.inputIntermediateScale = reader.float32_(position, 16, 0);
        $.forgetIntermediateScale = reader.float32_(position, 18, 0);
        $.cellIntermediateScale = reader.float32_(position, 20, 0);
        $.outputIntermediateScale = reader.float32_(position, 22, 0);
        $.hiddenStateZeroPoint = reader.int32_(position, 24, 0);
        $.hiddenStateScale = reader.float32_(position, 26, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.QLstmDescriptor();
        $.cifgEnabled = reader.value(json.cifgEnabled, true);
        $.peepholeEnabled = reader.value(json.peepholeEnabled, false);
        $.projectionEnabled = reader.value(json.projectionEnabled, false);
        $.layerNormEnabled = reader.value(json.layerNormEnabled, false);
        $.cellClip = reader.value(json.cellClip, 0);
        $.projectionClip = reader.value(json.projectionClip, 0);
        $.inputIntermediateScale = reader.value(json.inputIntermediateScale, 0);
        $.forgetIntermediateScale = reader.value(json.forgetIntermediateScale, 0);
        $.cellIntermediateScale = reader.value(json.cellIntermediateScale, 0);
        $.outputIntermediateScale = reader.value(json.outputIntermediateScale, 0);
        $.hiddenStateZeroPoint = reader.value(json.hiddenStateZeroPoint, 0);
        $.hiddenStateScale = reader.value(json.hiddenStateScale, 0);
        return $;
    }
};

armnnSerializer.QLstmLayer = class QLstmLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.QLstmLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.QLstmDescriptor);
        $.inputParams = reader.table(position, 8, armnnSerializer.QLstmInputParams);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.QLstmLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.QLstmDescriptor);
        $.inputParams = reader.object(json.inputParams, armnnSerializer.QLstmInputParams);
        return $;
    }
};

armnnSerializer.QuantizedLstmInputParams = class QuantizedLstmInputParams {

    static decode(reader, position) {
        const $ = new armnnSerializer.QuantizedLstmInputParams();
        $.inputToInputWeights = reader.table(position, 4, armnnSerializer.ConstTensor);
        $.inputToForgetWeights = reader.table(position, 6, armnnSerializer.ConstTensor);
        $.inputToCellWeights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.inputToOutputWeights = reader.table(position, 10, armnnSerializer.ConstTensor);
        $.recurrentToInputWeights = reader.table(position, 12, armnnSerializer.ConstTensor);
        $.recurrentToForgetWeights = reader.table(position, 14, armnnSerializer.ConstTensor);
        $.recurrentToCellWeights = reader.table(position, 16, armnnSerializer.ConstTensor);
        $.recurrentToOutputWeights = reader.table(position, 18, armnnSerializer.ConstTensor);
        $.inputGateBias = reader.table(position, 20, armnnSerializer.ConstTensor);
        $.forgetGateBias = reader.table(position, 22, armnnSerializer.ConstTensor);
        $.cellBias = reader.table(position, 24, armnnSerializer.ConstTensor);
        $.outputGateBias = reader.table(position, 26, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.QuantizedLstmInputParams();
        $.inputToInputWeights = reader.object(json.inputToInputWeights, armnnSerializer.ConstTensor);
        $.inputToForgetWeights = reader.object(json.inputToForgetWeights, armnnSerializer.ConstTensor);
        $.inputToCellWeights = reader.object(json.inputToCellWeights, armnnSerializer.ConstTensor);
        $.inputToOutputWeights = reader.object(json.inputToOutputWeights, armnnSerializer.ConstTensor);
        $.recurrentToInputWeights = reader.object(json.recurrentToInputWeights, armnnSerializer.ConstTensor);
        $.recurrentToForgetWeights = reader.object(json.recurrentToForgetWeights, armnnSerializer.ConstTensor);
        $.recurrentToCellWeights = reader.object(json.recurrentToCellWeights, armnnSerializer.ConstTensor);
        $.recurrentToOutputWeights = reader.object(json.recurrentToOutputWeights, armnnSerializer.ConstTensor);
        $.inputGateBias = reader.object(json.inputGateBias, armnnSerializer.ConstTensor);
        $.forgetGateBias = reader.object(json.forgetGateBias, armnnSerializer.ConstTensor);
        $.cellBias = reader.object(json.cellBias, armnnSerializer.ConstTensor);
        $.outputGateBias = reader.object(json.outputGateBias, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.QuantizedLstmLayer = class QuantizedLstmLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.QuantizedLstmLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.inputParams = reader.table(position, 6, armnnSerializer.QuantizedLstmInputParams);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.QuantizedLstmLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.inputParams = reader.object(json.inputParams, armnnSerializer.QuantizedLstmInputParams);
        return $;
    }
};

armnnSerializer.DequantizeLayer = class DequantizeLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.DequantizeLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.DequantizeLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.MergeLayer = class MergeLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.MergeLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.MergeLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.SwitchLayer = class SwitchLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.SwitchLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SwitchLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.PreluLayer = class PreluLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.PreluLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.PreluLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.TransposeConvolution2dLayer = class TransposeConvolution2dLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.TransposeConvolution2dLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.TransposeConvolution2dDescriptor);
        $.weights = reader.table(position, 8, armnnSerializer.ConstTensor);
        $.biases = reader.table(position, 10, armnnSerializer.ConstTensor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.TransposeConvolution2dLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.TransposeConvolution2dDescriptor);
        $.weights = reader.object(json.weights, armnnSerializer.ConstTensor);
        $.biases = reader.object(json.biases, armnnSerializer.ConstTensor);
        return $;
    }
};

armnnSerializer.TransposeConvolution2dDescriptor = class TransposeConvolution2dDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.TransposeConvolution2dDescriptor();
        $.padLeft = reader.uint32_(position, 4, 0);
        $.padRight = reader.uint32_(position, 6, 0);
        $.padTop = reader.uint32_(position, 8, 0);
        $.padBottom = reader.uint32_(position, 10, 0);
        $.strideX = reader.uint32_(position, 12, 0);
        $.strideY = reader.uint32_(position, 14, 0);
        $.biasEnabled = reader.bool_(position, 16, false);
        $.dataLayout = reader.int8_(position, 18, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.TransposeConvolution2dDescriptor();
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.padTop = reader.value(json.padTop, 0);
        $.padBottom = reader.value(json.padBottom, 0);
        $.strideX = reader.value(json.strideX, 0);
        $.strideY = reader.value(json.strideY, 0);
        $.biasEnabled = reader.value(json.biasEnabled, false);
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        return $;
    }
};

armnnSerializer.TransposeLayer = class TransposeLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.TransposeLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.TransposeDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.TransposeLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.TransposeDescriptor);
        return $;
    }
};

armnnSerializer.TransposeDescriptor = class TransposeDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.TransposeDescriptor();
        $.dimMappings = reader.array(position, 4, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.TransposeDescriptor();
        $.dimMappings = reader.array(json.dimMappings, Uint32Array);
        return $;
    }
};

armnnSerializer.ResizeLayer = class ResizeLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.ResizeLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.ResizeDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ResizeLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.ResizeDescriptor);
        return $;
    }
};

armnnSerializer.ResizeDescriptor = class ResizeDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.ResizeDescriptor();
        $.targetHeight = reader.uint32_(position, 4, 0);
        $.targetWidth = reader.uint32_(position, 6, 0);
        $.method = reader.int8_(position, 8, 0);
        $.dataLayout = reader.int8_(position, 10, 0);
        $.alignCorners = reader.bool_(position, 12, false);
        $.halfPixelCenters = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.ResizeDescriptor();
        $.targetHeight = reader.value(json.targetHeight, 0);
        $.targetWidth = reader.value(json.targetWidth, 0);
        $.method = armnnSerializer.ResizeMethod[json.method];
        $.dataLayout = armnnSerializer.DataLayout[json.dataLayout];
        $.alignCorners = reader.value(json.alignCorners, false);
        $.halfPixelCenters = reader.value(json.halfPixelCenters, false);
        return $;
    }
};

armnnSerializer.StackLayer = class StackLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.StackLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.StackDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.StackLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.StackDescriptor);
        return $;
    }
};

armnnSerializer.StackDescriptor = class StackDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.StackDescriptor();
        $.axis = reader.uint32_(position, 4, 0);
        $.numInputs = reader.uint32_(position, 6, 0);
        $.inputShape = reader.array(position, 8, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.StackDescriptor();
        $.axis = reader.value(json.axis, 0);
        $.numInputs = reader.value(json.numInputs, 0);
        $.inputShape = reader.array(json.inputShape, Uint32Array);
        return $;
    }
};

armnnSerializer.StandInDescriptor = class StandInDescriptor {

    static decode(reader, position) {
        const $ = new armnnSerializer.StandInDescriptor();
        $.numInputs = reader.uint32_(position, 4, 0);
        $.numOutputs = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.StandInDescriptor();
        $.numInputs = reader.value(json.numInputs, 0);
        $.numOutputs = reader.value(json.numOutputs, 0);
        return $;
    }
};

armnnSerializer.StandInLayer = class StandInLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.StandInLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        $.descriptor = reader.table(position, 6, armnnSerializer.StandInDescriptor);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.StandInLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        $.descriptor = reader.object(json.descriptor, armnnSerializer.StandInDescriptor);
        return $;
    }
};

armnnSerializer.RankLayer = class RankLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.RankLayer();
        $.base = reader.table(position, 4, armnnSerializer.LayerBase);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.RankLayer();
        $.base = reader.object(json.base, armnnSerializer.LayerBase);
        return $;
    }
};

armnnSerializer.Layer = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return armnnSerializer.ActivationLayer.decode(reader, position);
            case 2: return armnnSerializer.AdditionLayer.decode(reader, position);
            case 3: return armnnSerializer.BatchToSpaceNdLayer.decode(reader, position);
            case 4: return armnnSerializer.BatchNormalizationLayer.decode(reader, position);
            case 5: return armnnSerializer.ConstantLayer.decode(reader, position);
            case 6: return armnnSerializer.Convolution2dLayer.decode(reader, position);
            case 7: return armnnSerializer.DepthwiseConvolution2dLayer.decode(reader, position);
            case 8: return armnnSerializer.FullyConnectedLayer.decode(reader, position);
            case 9: return armnnSerializer.InputLayer.decode(reader, position);
            case 10: return armnnSerializer.MultiplicationLayer.decode(reader, position);
            case 11: return armnnSerializer.OutputLayer.decode(reader, position);
            case 12: return armnnSerializer.PermuteLayer.decode(reader, position);
            case 13: return armnnSerializer.Pooling2dLayer.decode(reader, position);
            case 14: return armnnSerializer.ReshapeLayer.decode(reader, position);
            case 15: return armnnSerializer.SoftmaxLayer.decode(reader, position);
            case 16: return armnnSerializer.SpaceToBatchNdLayer.decode(reader, position);
            case 17: return armnnSerializer.DivisionLayer.decode(reader, position);
            case 18: return armnnSerializer.MinimumLayer.decode(reader, position);
            case 19: return armnnSerializer.EqualLayer.decode(reader, position);
            case 20: return armnnSerializer.MaximumLayer.decode(reader, position);
            case 21: return armnnSerializer.NormalizationLayer.decode(reader, position);
            case 22: return armnnSerializer.PadLayer.decode(reader, position);
            case 23: return armnnSerializer.RsqrtLayer.decode(reader, position);
            case 24: return armnnSerializer.FloorLayer.decode(reader, position);
            case 25: return armnnSerializer.GreaterLayer.decode(reader, position);
            case 26: return armnnSerializer.ResizeBilinearLayer.decode(reader, position);
            case 27: return armnnSerializer.SubtractionLayer.decode(reader, position);
            case 28: return armnnSerializer.StridedSliceLayer.decode(reader, position);
            case 29: return armnnSerializer.GatherLayer.decode(reader, position);
            case 30: return armnnSerializer.MeanLayer.decode(reader, position);
            case 31: return armnnSerializer.MergerLayer.decode(reader, position);
            case 32: return armnnSerializer.L2NormalizationLayer.decode(reader, position);
            case 33: return armnnSerializer.SplitterLayer.decode(reader, position);
            case 34: return armnnSerializer.DetectionPostProcessLayer.decode(reader, position);
            case 35: return armnnSerializer.LstmLayer.decode(reader, position);
            case 36: return armnnSerializer.QuantizedLstmLayer.decode(reader, position);
            case 37: return armnnSerializer.QuantizeLayer.decode(reader, position);
            case 38: return armnnSerializer.DequantizeLayer.decode(reader, position);
            case 39: return armnnSerializer.MergeLayer.decode(reader, position);
            case 40: return armnnSerializer.SwitchLayer.decode(reader, position);
            case 41: return armnnSerializer.ConcatLayer.decode(reader, position);
            case 42: return armnnSerializer.SpaceToDepthLayer.decode(reader, position);
            case 43: return armnnSerializer.PreluLayer.decode(reader, position);
            case 44: return armnnSerializer.TransposeConvolution2dLayer.decode(reader, position);
            case 45: return armnnSerializer.ResizeLayer.decode(reader, position);
            case 46: return armnnSerializer.StackLayer.decode(reader, position);
            case 47: return armnnSerializer.AbsLayer.decode(reader, position);
            case 48: return armnnSerializer.ArgMinMaxLayer.decode(reader, position);
            case 49: return armnnSerializer.SliceLayer.decode(reader, position);
            case 50: return armnnSerializer.DepthToSpaceLayer.decode(reader, position);
            case 51: return armnnSerializer.InstanceNormalizationLayer.decode(reader, position);
            case 52: return armnnSerializer.LogSoftmaxLayer.decode(reader, position);
            case 53: return armnnSerializer.ComparisonLayer.decode(reader, position);
            case 54: return armnnSerializer.StandInLayer.decode(reader, position);
            case 55: return armnnSerializer.ElementwiseUnaryLayer.decode(reader, position);
            case 56: return armnnSerializer.TransposeLayer.decode(reader, position);
            case 57: return armnnSerializer.QLstmLayer.decode(reader, position);
            case 58: return armnnSerializer.FillLayer.decode(reader, position);
            case 59: return armnnSerializer.RankLayer.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'ActivationLayer': return armnnSerializer.ActivationLayer.decodeText(reader, json);
            case 'AdditionLayer': return armnnSerializer.AdditionLayer.decodeText(reader, json);
            case 'BatchToSpaceNdLayer': return armnnSerializer.BatchToSpaceNdLayer.decodeText(reader, json);
            case 'BatchNormalizationLayer': return armnnSerializer.BatchNormalizationLayer.decodeText(reader, json);
            case 'ConstantLayer': return armnnSerializer.ConstantLayer.decodeText(reader, json);
            case 'Convolution2dLayer': return armnnSerializer.Convolution2dLayer.decodeText(reader, json);
            case 'DepthwiseConvolution2dLayer': return armnnSerializer.DepthwiseConvolution2dLayer.decodeText(reader, json);
            case 'FullyConnectedLayer': return armnnSerializer.FullyConnectedLayer.decodeText(reader, json);
            case 'InputLayer': return armnnSerializer.InputLayer.decodeText(reader, json);
            case 'MultiplicationLayer': return armnnSerializer.MultiplicationLayer.decodeText(reader, json);
            case 'OutputLayer': return armnnSerializer.OutputLayer.decodeText(reader, json);
            case 'PermuteLayer': return armnnSerializer.PermuteLayer.decodeText(reader, json);
            case 'Pooling2dLayer': return armnnSerializer.Pooling2dLayer.decodeText(reader, json);
            case 'ReshapeLayer': return armnnSerializer.ReshapeLayer.decodeText(reader, json);
            case 'SoftmaxLayer': return armnnSerializer.SoftmaxLayer.decodeText(reader, json);
            case 'SpaceToBatchNdLayer': return armnnSerializer.SpaceToBatchNdLayer.decodeText(reader, json);
            case 'DivisionLayer': return armnnSerializer.DivisionLayer.decodeText(reader, json);
            case 'MinimumLayer': return armnnSerializer.MinimumLayer.decodeText(reader, json);
            case 'EqualLayer': return armnnSerializer.EqualLayer.decodeText(reader, json);
            case 'MaximumLayer': return armnnSerializer.MaximumLayer.decodeText(reader, json);
            case 'NormalizationLayer': return armnnSerializer.NormalizationLayer.decodeText(reader, json);
            case 'PadLayer': return armnnSerializer.PadLayer.decodeText(reader, json);
            case 'RsqrtLayer': return armnnSerializer.RsqrtLayer.decodeText(reader, json);
            case 'FloorLayer': return armnnSerializer.FloorLayer.decodeText(reader, json);
            case 'GreaterLayer': return armnnSerializer.GreaterLayer.decodeText(reader, json);
            case 'ResizeBilinearLayer': return armnnSerializer.ResizeBilinearLayer.decodeText(reader, json);
            case 'SubtractionLayer': return armnnSerializer.SubtractionLayer.decodeText(reader, json);
            case 'StridedSliceLayer': return armnnSerializer.StridedSliceLayer.decodeText(reader, json);
            case 'GatherLayer': return armnnSerializer.GatherLayer.decodeText(reader, json);
            case 'MeanLayer': return armnnSerializer.MeanLayer.decodeText(reader, json);
            case 'MergerLayer': return armnnSerializer.MergerLayer.decodeText(reader, json);
            case 'L2NormalizationLayer': return armnnSerializer.L2NormalizationLayer.decodeText(reader, json);
            case 'SplitterLayer': return armnnSerializer.SplitterLayer.decodeText(reader, json);
            case 'DetectionPostProcessLayer': return armnnSerializer.DetectionPostProcessLayer.decodeText(reader, json);
            case 'LstmLayer': return armnnSerializer.LstmLayer.decodeText(reader, json);
            case 'QuantizedLstmLayer': return armnnSerializer.QuantizedLstmLayer.decodeText(reader, json);
            case 'QuantizeLayer': return armnnSerializer.QuantizeLayer.decodeText(reader, json);
            case 'DequantizeLayer': return armnnSerializer.DequantizeLayer.decodeText(reader, json);
            case 'MergeLayer': return armnnSerializer.MergeLayer.decodeText(reader, json);
            case 'SwitchLayer': return armnnSerializer.SwitchLayer.decodeText(reader, json);
            case 'ConcatLayer': return armnnSerializer.ConcatLayer.decodeText(reader, json);
            case 'SpaceToDepthLayer': return armnnSerializer.SpaceToDepthLayer.decodeText(reader, json);
            case 'PreluLayer': return armnnSerializer.PreluLayer.decodeText(reader, json);
            case 'TransposeConvolution2dLayer': return armnnSerializer.TransposeConvolution2dLayer.decodeText(reader, json);
            case 'ResizeLayer': return armnnSerializer.ResizeLayer.decodeText(reader, json);
            case 'StackLayer': return armnnSerializer.StackLayer.decodeText(reader, json);
            case 'AbsLayer': return armnnSerializer.AbsLayer.decodeText(reader, json);
            case 'ArgMinMaxLayer': return armnnSerializer.ArgMinMaxLayer.decodeText(reader, json);
            case 'SliceLayer': return armnnSerializer.SliceLayer.decodeText(reader, json);
            case 'DepthToSpaceLayer': return armnnSerializer.DepthToSpaceLayer.decodeText(reader, json);
            case 'InstanceNormalizationLayer': return armnnSerializer.InstanceNormalizationLayer.decodeText(reader, json);
            case 'LogSoftmaxLayer': return armnnSerializer.LogSoftmaxLayer.decodeText(reader, json);
            case 'ComparisonLayer': return armnnSerializer.ComparisonLayer.decodeText(reader, json);
            case 'StandInLayer': return armnnSerializer.StandInLayer.decodeText(reader, json);
            case 'ElementwiseUnaryLayer': return armnnSerializer.ElementwiseUnaryLayer.decodeText(reader, json);
            case 'TransposeLayer': return armnnSerializer.TransposeLayer.decodeText(reader, json);
            case 'QLstmLayer': return armnnSerializer.QLstmLayer.decodeText(reader, json);
            case 'FillLayer': return armnnSerializer.FillLayer.decodeText(reader, json);
            case 'RankLayer': return armnnSerializer.RankLayer.decodeText(reader, json);
            default: return undefined;
        }
    }
};

armnnSerializer.AnyLayer = class AnyLayer {

    static decode(reader, position) {
        const $ = new armnnSerializer.AnyLayer();
        $.layer = reader.union(position, 4, armnnSerializer.Layer);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.AnyLayer();
        $.layer = armnnSerializer.Layer.decodeText(reader, json.layer, json.layer_type);
        return $;
    }
};

armnnSerializer.FeatureCompatibilityVersions = class FeatureCompatibilityVersions {

    static decode(reader, position) {
        const $ = new armnnSerializer.FeatureCompatibilityVersions();
        $.bindingIdsScheme = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.FeatureCompatibilityVersions();
        $.bindingIdsScheme = reader.value(json.bindingIdsScheme, 0);
        return $;
    }
};

armnnSerializer.SerializedGraph = class SerializedGraph {

    static identifier(reader) {
        return reader.identifier === 'ARMN';
    }

    static create(reader) {
        return armnnSerializer.SerializedGraph.decode(reader, reader.root);
    }

    static createText(reader) {
        return armnnSerializer.SerializedGraph.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new armnnSerializer.SerializedGraph();
        $.layers = reader.tables(position, 4, armnnSerializer.AnyLayer);
        $.inputIds = reader.array(position, 6, Int32Array);
        $.outputIds = reader.array(position, 8, Int32Array);
        $.featureVersions = reader.table(position, 10, armnnSerializer.FeatureCompatibilityVersions);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new armnnSerializer.SerializedGraph();
        $.layers = reader.objects(json.layers, armnnSerializer.AnyLayer);
        $.inputIds = reader.array(json.inputIds, Int32Array);
        $.outputIds = reader.array(json.outputIds, Int32Array);
        $.featureVersions = reader.object(json.featureVersions, armnnSerializer.FeatureCompatibilityVersions);
        return $;
    }
};
