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
    kAPPLE_NPU: 9
};

$root.wnn.WNN_DATA_FORMAT = {
    NCHW: 0,
    NHWC: 1,
    NC4HW4: 2,
    NHWC4: 3,
    UNKNOWN: 4
};

$root.wnn.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.wnn.Tensor();
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
        $.tensor = reader.table(position, 16, $root.wnn.Tensor.decode);
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
        $.relu = reader.bool_(position, 28, false);
        $.relu6 = reader.bool_(position, 30, false);
        $.pads = reader.typedArray(position, 32, Int32Array);
        $.out_pads = reader.typedArray(position, 34, Int32Array);
        $.has_outputshape = reader.bool_(position, 36, false);
        return $;
    }
};

$root.wnn.Conv2D = class Conv2D {

    static decode(reader, position) {
        const $ = new $root.wnn.Conv2D();
        $.common = reader.table(position, 4, $root.wnn.Conv2DCommon.decode);
        $.weight = reader.typedArray(position, 6, Float32Array);
        $.bias = reader.typedArray(position, 8, Float32Array);
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

$root.wnn.ModelSource = {
    TORCH: 0,
    TENSORFLOW: 1,
    ONNX: 2,
    TFLITE: 3
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
    fc: 16,
    matmul: 17,
    fc_share: 18,
    lstm: 19,
    onehot: 20,
    transpose: 21,
    gather: 22,
    split: 23,
    concat: 24,
    activation: 25,
    binary_op: 26,
    reduce: 27,
    fill: 28,
    pad: 29,
    reshape: 30,
    instancenorm: 31,
    conv_depthwise: 32,
    quantized_avgpool: 33,
    quantized_concat: 34,
    quantized_matmul: 35,
    quantized_relu: 36,
    quantized_relu6: 37,
    quantized_softmax: 38,
    roipooling: 39,
    roialign: 40,
    scatternd: 41,
    gathernd: 42,
    nms: 43,
    input: 44,
    output: 45,
    extra: 46,
    unsupported: 47
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
            case 12: return $root.wnn.Input.decode(reader, position);
            case 13: return $root.wnn.Extra.decode(reader, position);
            case 14: return $root.wnn.ArgMax.decode(reader, position);
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
            case 'Input': return $root.wnn.Input.decodeText(reader, json);
            case 'Extra': return $root.wnn.Extra.decodeText(reader, json);
            case 'ArgMax': return $root.wnn.ArgMax.decodeText(reader, json);
            default: return undefined;
        }
    }
};

$root.wnn.Op = class Op {

    static decode(reader, position) {
        const $ = new $root.wnn.Op();
        $.input_indexes = reader.typedArray(position, 4, Int32Array);
        $.output_indexes = reader.typedArray(position, 6, Int32Array);
        $.input_names = reader.strings_(position, 8);
        $.output_names = reader.strings_(position, 10);
        $.param = reader.union(position, 12, $root.wnn.OpParameter.decode);
        $.name = reader.string_(position, 16, null);
        $.type = reader.int32_(position, 18, 0);
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
        $.oplists = reader.tableArray(position, 12, $root.wnn.Op.decode);
        $.output_names = reader.strings_(position, 14);
        $.input_names = reader.strings_(position, 16);
        $.model_source = reader.int8_(position, 18, 0);
        $.tensor_names = reader.strings_(position, 20);
        $.tensor_number = reader.int32_(position, 22, 0);
        $.subgraph = reader.table(position, 24, $root.wnn.SubGraph.decode);
        $.uuid = reader.string_(position, 26, null);
        $.password = reader.string_(position, 28, null);
        return $;
    }
};
