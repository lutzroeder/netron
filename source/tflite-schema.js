
export const tflite = {};

tflite.TensorType = {
    FLOAT32: 0,
    FLOAT16: 1,
    INT32: 2,
    UINT8: 3,
    INT64: 4,
    STRING: 5,
    BOOL: 6,
    INT16: 7,
    COMPLEX64: 8,
    INT8: 9,
    FLOAT64: 10,
    COMPLEX128: 11,
    UINT64: 12,
    RESOURCE: 13,
    VARIANT: 14,
    UINT32: 15,
    UINT16: 16,
    INT4: 17,
    BFLOAT16: 18
};

tflite.CustomQuantization = class CustomQuantization {

    static decode(reader, position) {
        const $ = new tflite.CustomQuantization();
        $.custom = reader.array(position, 4, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.CustomQuantization();
        $.custom = reader.array(json.custom, Uint8Array);
        return $;
    }
};

tflite.QuantizationDetails = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tflite.CustomQuantization.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'CustomQuantization': return tflite.CustomQuantization.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tflite.QuantizationParameters = class QuantizationParameters {

    static decode(reader, position) {
        const $ = new tflite.QuantizationParameters();
        $.min = reader.array(position, 4, Float32Array);
        $.max = reader.array(position, 6, Float32Array);
        $.scale = reader.array(position, 8, Float32Array);
        $.zero_point = reader.int64s_(position, 10);
        $.details = reader.union(position, 12, tflite.QuantizationDetails);
        $.quantized_dimension = reader.int32_(position, 16, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.QuantizationParameters();
        $.min = reader.array(json.min, Float32Array);
        $.max = reader.array(json.max, Float32Array);
        $.scale = reader.array(json.scale, Float32Array);
        $.zero_point = reader.array(json.zero_point);
        $.details = tflite.QuantizationDetails.decodeText(reader, json.details, json.details_type);
        $.quantized_dimension = reader.value(json.quantized_dimension, 0);
        return $;
    }
};

tflite.DimensionType = {
    DENSE: 0,
    SPARSE_CSR: 1
};

tflite.Int32Vector = class Int32Vector {

    static decode(reader, position) {
        const $ = new tflite.Int32Vector();
        $.values = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Int32Vector();
        $.values = reader.array(json.values, Int32Array);
        return $;
    }
};

tflite.Uint16Vector = class Uint16Vector {

    static decode(reader, position) {
        const $ = new tflite.Uint16Vector();
        $.values = reader.array(position, 4, Uint16Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Uint16Vector();
        $.values = reader.array(json.values, Uint16Array);
        return $;
    }
};

tflite.Uint8Vector = class Uint8Vector {

    static decode(reader, position) {
        const $ = new tflite.Uint8Vector();
        $.values = reader.array(position, 4, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Uint8Vector();
        $.values = reader.array(json.values, Uint8Array);
        return $;
    }
};

tflite.SparseIndexVector = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tflite.Int32Vector.decode(reader, position);
            case 2: return tflite.Uint16Vector.decode(reader, position);
            case 3: return tflite.Uint8Vector.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Int32Vector': return tflite.Int32Vector.decodeText(reader, json);
            case 'Uint16Vector': return tflite.Uint16Vector.decodeText(reader, json);
            case 'Uint8Vector': return tflite.Uint8Vector.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tflite.DimensionMetadata = class DimensionMetadata {

    static decode(reader, position) {
        const $ = new tflite.DimensionMetadata();
        $.format = reader.int8_(position, 4, 0);
        $.dense_size = reader.int32_(position, 6, 0);
        $.array_segments = reader.union(position, 8, tflite.SparseIndexVector);
        $.array_indices = reader.union(position, 12, tflite.SparseIndexVector);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.DimensionMetadata();
        $.format = tflite.DimensionType[json.format];
        $.dense_size = reader.value(json.dense_size, 0);
        $.array_segments = tflite.SparseIndexVector.decodeText(reader, json.array_segments, json.array_segments_type);
        $.array_indices = tflite.SparseIndexVector.decodeText(reader, json.array_indices, json.array_indices_type);
        return $;
    }
};

tflite.SparsityParameters = class SparsityParameters {

    static decode(reader, position) {
        const $ = new tflite.SparsityParameters();
        $.traversal_order = reader.array(position, 4, Int32Array);
        $.block_map = reader.array(position, 6, Int32Array);
        $.dim_metadata = reader.tables(position, 8, tflite.DimensionMetadata);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SparsityParameters();
        $.traversal_order = reader.array(json.traversal_order, Int32Array);
        $.block_map = reader.array(json.block_map, Int32Array);
        $.dim_metadata = reader.objects(json.dim_metadata, tflite.DimensionMetadata);
        return $;
    }
};

tflite.VariantSubType = class VariantSubType {

    static decode(reader, position) {
        const $ = new tflite.VariantSubType();
        $.shape = reader.array(position, 4, Int32Array);
        $.type = reader.int8_(position, 6, 0);
        $.has_rank = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.VariantSubType();
        $.shape = reader.array(json.shape, Int32Array);
        $.type = tflite.TensorType[json.type];
        $.has_rank = reader.value(json.has_rank, false);
        return $;
    }
};

tflite.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new tflite.Tensor();
        $.shape = reader.array(position, 4, Int32Array);
        $.type = reader.int8_(position, 6, 0);
        $.buffer = reader.uint32_(position, 8, 0);
        $.name = reader.string_(position, 10, null);
        $.quantization = reader.table(position, 12, tflite.QuantizationParameters);
        $.is_variable = reader.bool_(position, 14, false);
        $.sparsity = reader.table(position, 16, tflite.SparsityParameters);
        $.shape_signature = reader.array(position, 18, Int32Array);
        $.has_rank = reader.bool_(position, 20, false);
        $.variant_tensors = reader.tables(position, 22, tflite.VariantSubType);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Tensor();
        $.shape = reader.array(json.shape, Int32Array);
        $.type = tflite.TensorType[json.type];
        $.buffer = reader.value(json.buffer, 0);
        $.name = reader.value(json.name, null);
        $.quantization = reader.object(json.quantization, tflite.QuantizationParameters);
        $.is_variable = reader.value(json.is_variable, false);
        $.sparsity = reader.object(json.sparsity, tflite.SparsityParameters);
        $.shape_signature = reader.array(json.shape_signature, Int32Array);
        $.has_rank = reader.value(json.has_rank, false);
        $.variant_tensors = reader.objects(json.variant_tensors, tflite.VariantSubType);
        return $;
    }
};

tflite.BuiltinOperator = {
    ADD: 0,
    AVERAGE_POOL_2D: 1,
    CONCATENATION: 2,
    CONV_2D: 3,
    DEPTHWISE_CONV_2D: 4,
    DEPTH_TO_SPACE: 5,
    DEQUANTIZE: 6,
    EMBEDDING_LOOKUP: 7,
    FLOOR: 8,
    FULLY_CONNECTED: 9,
    HASHTABLE_LOOKUP: 10,
    L2_NORMALIZATION: 11,
    L2_POOL_2D: 12,
    LOCAL_RESPONSE_NORMALIZATION: 13,
    LOGISTIC: 14,
    LSH_PROJECTION: 15,
    LSTM: 16,
    MAX_POOL_2D: 17,
    MUL: 18,
    RELU: 19,
    RELU_N1_TO_1: 20,
    RELU6: 21,
    RESHAPE: 22,
    RESIZE_BILINEAR: 23,
    RNN: 24,
    SOFTMAX: 25,
    SPACE_TO_DEPTH: 26,
    SVDF: 27,
    TANH: 28,
    CONCAT_EMBEDDINGS: 29,
    SKIP_GRAM: 30,
    CALL: 31,
    CUSTOM: 32,
    EMBEDDING_LOOKUP_SPARSE: 33,
    PAD: 34,
    UNIDIRECTIONAL_SEQUENCE_RNN: 35,
    GATHER: 36,
    BATCH_TO_SPACE_ND: 37,
    SPACE_TO_BATCH_ND: 38,
    TRANSPOSE: 39,
    MEAN: 40,
    SUB: 41,
    DIV: 42,
    SQUEEZE: 43,
    UNIDIRECTIONAL_SEQUENCE_LSTM: 44,
    STRIDED_SLICE: 45,
    BIDIRECTIONAL_SEQUENCE_RNN: 46,
    EXP: 47,
    TOPK_V2: 48,
    SPLIT: 49,
    LOG_SOFTMAX: 50,
    DELEGATE: 51,
    BIDIRECTIONAL_SEQUENCE_LSTM: 52,
    CAST: 53,
    PRELU: 54,
    MAXIMUM: 55,
    ARG_MAX: 56,
    MINIMUM: 57,
    LESS: 58,
    NEG: 59,
    PADV2: 60,
    GREATER: 61,
    GREATER_EQUAL: 62,
    LESS_EQUAL: 63,
    SELECT: 64,
    SLICE: 65,
    SIN: 66,
    TRANSPOSE_CONV: 67,
    SPARSE_TO_DENSE: 68,
    TILE: 69,
    EXPAND_DIMS: 70,
    EQUAL: 71,
    NOT_EQUAL: 72,
    LOG: 73,
    SUM: 74,
    SQRT: 75,
    RSQRT: 76,
    SHAPE: 77,
    POW: 78,
    ARG_MIN: 79,
    FAKE_QUANT: 80,
    REDUCE_PROD: 81,
    REDUCE_MAX: 82,
    PACK: 83,
    LOGICAL_OR: 84,
    ONE_HOT: 85,
    LOGICAL_AND: 86,
    LOGICAL_NOT: 87,
    UNPACK: 88,
    REDUCE_MIN: 89,
    FLOOR_DIV: 90,
    REDUCE_ANY: 91,
    SQUARE: 92,
    ZEROS_LIKE: 93,
    FILL: 94,
    FLOOR_MOD: 95,
    RANGE: 96,
    RESIZE_NEAREST_NEIGHBOR: 97,
    LEAKY_RELU: 98,
    SQUARED_DIFFERENCE: 99,
    MIRROR_PAD: 100,
    ABS: 101,
    SPLIT_V: 102,
    UNIQUE: 103,
    CEIL: 104,
    REVERSE_V2: 105,
    ADD_N: 106,
    GATHER_ND: 107,
    COS: 108,
    WHERE: 109,
    RANK: 110,
    ELU: 111,
    REVERSE_SEQUENCE: 112,
    MATRIX_DIAG: 113,
    QUANTIZE: 114,
    MATRIX_SET_DIAG: 115,
    ROUND: 116,
    HARD_SWISH: 117,
    IF: 118,
    WHILE: 119,
    NON_MAX_SUPPRESSION_V4: 120,
    NON_MAX_SUPPRESSION_V5: 121,
    SCATTER_ND: 122,
    SELECT_V2: 123,
    DENSIFY: 124,
    SEGMENT_SUM: 125,
    BATCH_MATMUL: 126,
    PLACEHOLDER_FOR_GREATER_OP_CODES: 127,
    CUMSUM: 128,
    CALL_ONCE: 129,
    BROADCAST_TO: 130,
    RFFT2D: 131,
    CONV_3D: 132,
    IMAG: 133,
    REAL: 134,
    COMPLEX_ABS: 135,
    HASHTABLE: 136,
    HASHTABLE_FIND: 137,
    HASHTABLE_IMPORT: 138,
    HASHTABLE_SIZE: 139,
    REDUCE_ALL: 140,
    CONV_3D_TRANSPOSE: 141,
    VAR_HANDLE: 142,
    READ_VARIABLE: 143,
    ASSIGN_VARIABLE: 144,
    BROADCAST_ARGS: 145,
    RANDOM_STANDARD_NORMAL: 146,
    BUCKETIZE: 147,
    RANDOM_UNIFORM: 148,
    MULTINOMIAL: 149,
    GELU: 150,
    DYNAMIC_UPDATE_SLICE: 151,
    RELU_0_TO_1: 152,
    UNSORTED_SEGMENT_PROD: 153,
    UNSORTED_SEGMENT_MAX: 154,
    UNSORTED_SEGMENT_SUM: 155,
    ATAN2: 156,
    UNSORTED_SEGMENT_MIN: 157,
    SIGN: 158,
    BITCAST: 159,
    BITWISE_XOR: 160,
    RIGHT_SHIFT: 161,
    STABLEHLO_LOGISTIC: 162,
    STABLEHLO_ADD: 163,
    STABLEHLO_DIVIDE: 164,
    STABLEHLO_MULTIPLY: 165,
    STABLEHLO_MAXIMUM: 166,
    STABLEHLO_RESHAPE: 167,
    STABLEHLO_CLAMP: 168,
    STABLEHLO_CONCATENATE: 169,
    STABLEHLO_BROADCAST_IN_DIM: 170,
    STABLEHLO_CONVOLUTION: 171,
    STABLEHLO_SLICE: 172,
    STABLEHLO_CUSTOM_CALL: 173,
    STABLEHLO_REDUCE: 174,
    STABLEHLO_ABS: 175,
    STABLEHLO_AND: 176,
    STABLEHLO_COSINE: 177,
    STABLEHLO_EXPONENTIAL: 178,
    STABLEHLO_FLOOR: 179,
    STABLEHLO_LOG: 180,
    STABLEHLO_MINIMUM: 181,
    STABLEHLO_NEGATE: 182,
    STABLEHLO_OR: 183,
    STABLEHLO_POWER: 184,
    STABLEHLO_REMAINDER: 185,
    STABLEHLO_RSQRT: 186,
    STABLEHLO_SELECT: 187,
    STABLEHLO_SUBTRACT: 188,
    STABLEHLO_TANH: 189,
    STABLEHLO_SCATTER: 190,
    STABLEHLO_COMPARE: 191,
    STABLEHLO_CONVERT: 192,
    STABLEHLO_DYNAMIC_SLICE: 193,
    STABLEHLO_DYNAMIC_UPDATE_SLICE: 194,
    STABLEHLO_PAD: 195,
    STABLEHLO_IOTA: 196,
    STABLEHLO_DOT_GENERAL: 197,
    STABLEHLO_REDUCE_WINDOW: 198,
    STABLEHLO_SORT: 199,
    STABLEHLO_WHILE: 200,
    STABLEHLO_GATHER: 201,
    STABLEHLO_TRANSPOSE: 202,
    DILATE: 203,
    STABLEHLO_RNG_BIT_GENERATOR: 204,
    REDUCE_WINDOW: 205,
    STABLEHLO_COMPOSITE: 206
};

tflite.BuiltinOptions = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tflite.Conv2DOptions.decode(reader, position);
            case 2: return tflite.DepthwiseConv2DOptions.decode(reader, position);
            case 3: return tflite.ConcatEmbeddingsOptions.decode(reader, position);
            case 4: return tflite.LSHProjectionOptions.decode(reader, position);
            case 5: return tflite.Pool2DOptions.decode(reader, position);
            case 6: return tflite.SVDFOptions.decode(reader, position);
            case 7: return tflite.RNNOptions.decode(reader, position);
            case 8: return tflite.FullyConnectedOptions.decode(reader, position);
            case 9: return tflite.SoftmaxOptions.decode(reader, position);
            case 10: return tflite.ConcatenationOptions.decode(reader, position);
            case 11: return tflite.AddOptions.decode(reader, position);
            case 12: return tflite.L2NormOptions.decode(reader, position);
            case 13: return tflite.LocalResponseNormalizationOptions.decode(reader, position);
            case 14: return tflite.LSTMOptions.decode(reader, position);
            case 15: return tflite.ResizeBilinearOptions.decode(reader, position);
            case 16: return tflite.CallOptions.decode(reader, position);
            case 17: return tflite.ReshapeOptions.decode(reader, position);
            case 18: return tflite.SkipGramOptions.decode(reader, position);
            case 19: return tflite.SpaceToDepthOptions.decode(reader, position);
            case 20: return tflite.EmbeddingLookupSparseOptions.decode(reader, position);
            case 21: return tflite.MulOptions.decode(reader, position);
            case 22: return tflite.PadOptions.decode(reader, position);
            case 23: return tflite.GatherOptions.decode(reader, position);
            case 24: return tflite.BatchToSpaceNDOptions.decode(reader, position);
            case 25: return tflite.SpaceToBatchNDOptions.decode(reader, position);
            case 26: return tflite.TransposeOptions.decode(reader, position);
            case 27: return tflite.ReducerOptions.decode(reader, position);
            case 28: return tflite.SubOptions.decode(reader, position);
            case 29: return tflite.DivOptions.decode(reader, position);
            case 30: return tflite.SqueezeOptions.decode(reader, position);
            case 31: return tflite.SequenceRNNOptions.decode(reader, position);
            case 32: return tflite.StridedSliceOptions.decode(reader, position);
            case 33: return tflite.ExpOptions.decode(reader, position);
            case 34: return tflite.TopKV2Options.decode(reader, position);
            case 35: return tflite.SplitOptions.decode(reader, position);
            case 36: return tflite.LogSoftmaxOptions.decode(reader, position);
            case 37: return tflite.CastOptions.decode(reader, position);
            case 38: return tflite.DequantizeOptions.decode(reader, position);
            case 39: return tflite.MaximumMinimumOptions.decode(reader, position);
            case 40: return tflite.ArgMaxOptions.decode(reader, position);
            case 41: return tflite.LessOptions.decode(reader, position);
            case 42: return tflite.NegOptions.decode(reader, position);
            case 43: return tflite.PadV2Options.decode(reader, position);
            case 44: return tflite.GreaterOptions.decode(reader, position);
            case 45: return tflite.GreaterEqualOptions.decode(reader, position);
            case 46: return tflite.LessEqualOptions.decode(reader, position);
            case 47: return tflite.SelectOptions.decode(reader, position);
            case 48: return tflite.SliceOptions.decode(reader, position);
            case 49: return tflite.TransposeConvOptions.decode(reader, position);
            case 50: return tflite.SparseToDenseOptions.decode(reader, position);
            case 51: return tflite.TileOptions.decode(reader, position);
            case 52: return tflite.ExpandDimsOptions.decode(reader, position);
            case 53: return tflite.EqualOptions.decode(reader, position);
            case 54: return tflite.NotEqualOptions.decode(reader, position);
            case 55: return tflite.ShapeOptions.decode(reader, position);
            case 56: return tflite.PowOptions.decode(reader, position);
            case 57: return tflite.ArgMinOptions.decode(reader, position);
            case 58: return tflite.FakeQuantOptions.decode(reader, position);
            case 59: return tflite.PackOptions.decode(reader, position);
            case 60: return tflite.LogicalOrOptions.decode(reader, position);
            case 61: return tflite.OneHotOptions.decode(reader, position);
            case 62: return tflite.LogicalAndOptions.decode(reader, position);
            case 63: return tflite.LogicalNotOptions.decode(reader, position);
            case 64: return tflite.UnpackOptions.decode(reader, position);
            case 65: return tflite.FloorDivOptions.decode(reader, position);
            case 66: return tflite.SquareOptions.decode(reader, position);
            case 67: return tflite.ZerosLikeOptions.decode(reader, position);
            case 68: return tflite.FillOptions.decode(reader, position);
            case 69: return tflite.BidirectionalSequenceLSTMOptions.decode(reader, position);
            case 70: return tflite.BidirectionalSequenceRNNOptions.decode(reader, position);
            case 71: return tflite.UnidirectionalSequenceLSTMOptions.decode(reader, position);
            case 72: return tflite.FloorModOptions.decode(reader, position);
            case 73: return tflite.RangeOptions.decode(reader, position);
            case 74: return tflite.ResizeNearestNeighborOptions.decode(reader, position);
            case 75: return tflite.LeakyReluOptions.decode(reader, position);
            case 76: return tflite.SquaredDifferenceOptions.decode(reader, position);
            case 77: return tflite.MirrorPadOptions.decode(reader, position);
            case 78: return tflite.AbsOptions.decode(reader, position);
            case 79: return tflite.SplitVOptions.decode(reader, position);
            case 80: return tflite.UniqueOptions.decode(reader, position);
            case 81: return tflite.ReverseV2Options.decode(reader, position);
            case 82: return tflite.AddNOptions.decode(reader, position);
            case 83: return tflite.GatherNdOptions.decode(reader, position);
            case 84: return tflite.CosOptions.decode(reader, position);
            case 85: return tflite.WhereOptions.decode(reader, position);
            case 86: return tflite.RankOptions.decode(reader, position);
            case 87: return tflite.ReverseSequenceOptions.decode(reader, position);
            case 88: return tflite.MatrixDiagOptions.decode(reader, position);
            case 89: return tflite.QuantizeOptions.decode(reader, position);
            case 90: return tflite.MatrixSetDiagOptions.decode(reader, position);
            case 91: return tflite.HardSwishOptions.decode(reader, position);
            case 92: return tflite.IfOptions.decode(reader, position);
            case 93: return tflite.WhileOptions.decode(reader, position);
            case 94: return tflite.DepthToSpaceOptions.decode(reader, position);
            case 95: return tflite.NonMaxSuppressionV4Options.decode(reader, position);
            case 96: return tflite.NonMaxSuppressionV5Options.decode(reader, position);
            case 97: return tflite.ScatterNdOptions.decode(reader, position);
            case 98: return tflite.SelectV2Options.decode(reader, position);
            case 99: return tflite.DensifyOptions.decode(reader, position);
            case 100: return tflite.SegmentSumOptions.decode(reader, position);
            case 101: return tflite.BatchMatMulOptions.decode(reader, position);
            case 102: return tflite.CumsumOptions.decode(reader, position);
            case 103: return tflite.CallOnceOptions.decode(reader, position);
            case 104: return tflite.BroadcastToOptions.decode(reader, position);
            case 105: return tflite.Rfft2dOptions.decode(reader, position);
            case 106: return tflite.Conv3DOptions.decode(reader, position);
            case 107: return tflite.HashtableOptions.decode(reader, position);
            case 108: return tflite.HashtableFindOptions.decode(reader, position);
            case 109: return tflite.HashtableImportOptions.decode(reader, position);
            case 110: return tflite.HashtableSizeOptions.decode(reader, position);
            case 111: return tflite.VarHandleOptions.decode(reader, position);
            case 112: return tflite.ReadVariableOptions.decode(reader, position);
            case 113: return tflite.AssignVariableOptions.decode(reader, position);
            case 114: return tflite.RandomOptions.decode(reader, position);
            case 115: return tflite.BucketizeOptions.decode(reader, position);
            case 116: return tflite.GeluOptions.decode(reader, position);
            case 117: return tflite.DynamicUpdateSliceOptions.decode(reader, position);
            case 118: return tflite.UnsortedSegmentProdOptions.decode(reader, position);
            case 119: return tflite.UnsortedSegmentMaxOptions.decode(reader, position);
            case 120: return tflite.UnsortedSegmentMinOptions.decode(reader, position);
            case 121: return tflite.UnsortedSegmentSumOptions.decode(reader, position);
            case 122: return tflite.ATan2Options.decode(reader, position);
            case 123: return tflite.SignOptions.decode(reader, position);
            case 124: return tflite.BitcastOptions.decode(reader, position);
            case 125: return tflite.BitwiseXorOptions.decode(reader, position);
            case 126: return tflite.RightShiftOptions.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Conv2DOptions': return tflite.Conv2DOptions.decodeText(reader, json);
            case 'DepthwiseConv2DOptions': return tflite.DepthwiseConv2DOptions.decodeText(reader, json);
            case 'ConcatEmbeddingsOptions': return tflite.ConcatEmbeddingsOptions.decodeText(reader, json);
            case 'LSHProjectionOptions': return tflite.LSHProjectionOptions.decodeText(reader, json);
            case 'Pool2DOptions': return tflite.Pool2DOptions.decodeText(reader, json);
            case 'SVDFOptions': return tflite.SVDFOptions.decodeText(reader, json);
            case 'RNNOptions': return tflite.RNNOptions.decodeText(reader, json);
            case 'FullyConnectedOptions': return tflite.FullyConnectedOptions.decodeText(reader, json);
            case 'SoftmaxOptions': return tflite.SoftmaxOptions.decodeText(reader, json);
            case 'ConcatenationOptions': return tflite.ConcatenationOptions.decodeText(reader, json);
            case 'AddOptions': return tflite.AddOptions.decodeText(reader, json);
            case 'L2NormOptions': return tflite.L2NormOptions.decodeText(reader, json);
            case 'LocalResponseNormalizationOptions': return tflite.LocalResponseNormalizationOptions.decodeText(reader, json);
            case 'LSTMOptions': return tflite.LSTMOptions.decodeText(reader, json);
            case 'ResizeBilinearOptions': return tflite.ResizeBilinearOptions.decodeText(reader, json);
            case 'CallOptions': return tflite.CallOptions.decodeText(reader, json);
            case 'ReshapeOptions': return tflite.ReshapeOptions.decodeText(reader, json);
            case 'SkipGramOptions': return tflite.SkipGramOptions.decodeText(reader, json);
            case 'SpaceToDepthOptions': return tflite.SpaceToDepthOptions.decodeText(reader, json);
            case 'EmbeddingLookupSparseOptions': return tflite.EmbeddingLookupSparseOptions.decodeText(reader, json);
            case 'MulOptions': return tflite.MulOptions.decodeText(reader, json);
            case 'PadOptions': return tflite.PadOptions.decodeText(reader, json);
            case 'GatherOptions': return tflite.GatherOptions.decodeText(reader, json);
            case 'BatchToSpaceNDOptions': return tflite.BatchToSpaceNDOptions.decodeText(reader, json);
            case 'SpaceToBatchNDOptions': return tflite.SpaceToBatchNDOptions.decodeText(reader, json);
            case 'TransposeOptions': return tflite.TransposeOptions.decodeText(reader, json);
            case 'ReducerOptions': return tflite.ReducerOptions.decodeText(reader, json);
            case 'SubOptions': return tflite.SubOptions.decodeText(reader, json);
            case 'DivOptions': return tflite.DivOptions.decodeText(reader, json);
            case 'SqueezeOptions': return tflite.SqueezeOptions.decodeText(reader, json);
            case 'SequenceRNNOptions': return tflite.SequenceRNNOptions.decodeText(reader, json);
            case 'StridedSliceOptions': return tflite.StridedSliceOptions.decodeText(reader, json);
            case 'ExpOptions': return tflite.ExpOptions.decodeText(reader, json);
            case 'TopKV2Options': return tflite.TopKV2Options.decodeText(reader, json);
            case 'SplitOptions': return tflite.SplitOptions.decodeText(reader, json);
            case 'LogSoftmaxOptions': return tflite.LogSoftmaxOptions.decodeText(reader, json);
            case 'CastOptions': return tflite.CastOptions.decodeText(reader, json);
            case 'DequantizeOptions': return tflite.DequantizeOptions.decodeText(reader, json);
            case 'MaximumMinimumOptions': return tflite.MaximumMinimumOptions.decodeText(reader, json);
            case 'ArgMaxOptions': return tflite.ArgMaxOptions.decodeText(reader, json);
            case 'LessOptions': return tflite.LessOptions.decodeText(reader, json);
            case 'NegOptions': return tflite.NegOptions.decodeText(reader, json);
            case 'PadV2Options': return tflite.PadV2Options.decodeText(reader, json);
            case 'GreaterOptions': return tflite.GreaterOptions.decodeText(reader, json);
            case 'GreaterEqualOptions': return tflite.GreaterEqualOptions.decodeText(reader, json);
            case 'LessEqualOptions': return tflite.LessEqualOptions.decodeText(reader, json);
            case 'SelectOptions': return tflite.SelectOptions.decodeText(reader, json);
            case 'SliceOptions': return tflite.SliceOptions.decodeText(reader, json);
            case 'TransposeConvOptions': return tflite.TransposeConvOptions.decodeText(reader, json);
            case 'SparseToDenseOptions': return tflite.SparseToDenseOptions.decodeText(reader, json);
            case 'TileOptions': return tflite.TileOptions.decodeText(reader, json);
            case 'ExpandDimsOptions': return tflite.ExpandDimsOptions.decodeText(reader, json);
            case 'EqualOptions': return tflite.EqualOptions.decodeText(reader, json);
            case 'NotEqualOptions': return tflite.NotEqualOptions.decodeText(reader, json);
            case 'ShapeOptions': return tflite.ShapeOptions.decodeText(reader, json);
            case 'PowOptions': return tflite.PowOptions.decodeText(reader, json);
            case 'ArgMinOptions': return tflite.ArgMinOptions.decodeText(reader, json);
            case 'FakeQuantOptions': return tflite.FakeQuantOptions.decodeText(reader, json);
            case 'PackOptions': return tflite.PackOptions.decodeText(reader, json);
            case 'LogicalOrOptions': return tflite.LogicalOrOptions.decodeText(reader, json);
            case 'OneHotOptions': return tflite.OneHotOptions.decodeText(reader, json);
            case 'LogicalAndOptions': return tflite.LogicalAndOptions.decodeText(reader, json);
            case 'LogicalNotOptions': return tflite.LogicalNotOptions.decodeText(reader, json);
            case 'UnpackOptions': return tflite.UnpackOptions.decodeText(reader, json);
            case 'FloorDivOptions': return tflite.FloorDivOptions.decodeText(reader, json);
            case 'SquareOptions': return tflite.SquareOptions.decodeText(reader, json);
            case 'ZerosLikeOptions': return tflite.ZerosLikeOptions.decodeText(reader, json);
            case 'FillOptions': return tflite.FillOptions.decodeText(reader, json);
            case 'BidirectionalSequenceLSTMOptions': return tflite.BidirectionalSequenceLSTMOptions.decodeText(reader, json);
            case 'BidirectionalSequenceRNNOptions': return tflite.BidirectionalSequenceRNNOptions.decodeText(reader, json);
            case 'UnidirectionalSequenceLSTMOptions': return tflite.UnidirectionalSequenceLSTMOptions.decodeText(reader, json);
            case 'FloorModOptions': return tflite.FloorModOptions.decodeText(reader, json);
            case 'RangeOptions': return tflite.RangeOptions.decodeText(reader, json);
            case 'ResizeNearestNeighborOptions': return tflite.ResizeNearestNeighborOptions.decodeText(reader, json);
            case 'LeakyReluOptions': return tflite.LeakyReluOptions.decodeText(reader, json);
            case 'SquaredDifferenceOptions': return tflite.SquaredDifferenceOptions.decodeText(reader, json);
            case 'MirrorPadOptions': return tflite.MirrorPadOptions.decodeText(reader, json);
            case 'AbsOptions': return tflite.AbsOptions.decodeText(reader, json);
            case 'SplitVOptions': return tflite.SplitVOptions.decodeText(reader, json);
            case 'UniqueOptions': return tflite.UniqueOptions.decodeText(reader, json);
            case 'ReverseV2Options': return tflite.ReverseV2Options.decodeText(reader, json);
            case 'AddNOptions': return tflite.AddNOptions.decodeText(reader, json);
            case 'GatherNdOptions': return tflite.GatherNdOptions.decodeText(reader, json);
            case 'CosOptions': return tflite.CosOptions.decodeText(reader, json);
            case 'WhereOptions': return tflite.WhereOptions.decodeText(reader, json);
            case 'RankOptions': return tflite.RankOptions.decodeText(reader, json);
            case 'ReverseSequenceOptions': return tflite.ReverseSequenceOptions.decodeText(reader, json);
            case 'MatrixDiagOptions': return tflite.MatrixDiagOptions.decodeText(reader, json);
            case 'QuantizeOptions': return tflite.QuantizeOptions.decodeText(reader, json);
            case 'MatrixSetDiagOptions': return tflite.MatrixSetDiagOptions.decodeText(reader, json);
            case 'HardSwishOptions': return tflite.HardSwishOptions.decodeText(reader, json);
            case 'IfOptions': return tflite.IfOptions.decodeText(reader, json);
            case 'WhileOptions': return tflite.WhileOptions.decodeText(reader, json);
            case 'DepthToSpaceOptions': return tflite.DepthToSpaceOptions.decodeText(reader, json);
            case 'NonMaxSuppressionV4Options': return tflite.NonMaxSuppressionV4Options.decodeText(reader, json);
            case 'NonMaxSuppressionV5Options': return tflite.NonMaxSuppressionV5Options.decodeText(reader, json);
            case 'ScatterNdOptions': return tflite.ScatterNdOptions.decodeText(reader, json);
            case 'SelectV2Options': return tflite.SelectV2Options.decodeText(reader, json);
            case 'DensifyOptions': return tflite.DensifyOptions.decodeText(reader, json);
            case 'SegmentSumOptions': return tflite.SegmentSumOptions.decodeText(reader, json);
            case 'BatchMatMulOptions': return tflite.BatchMatMulOptions.decodeText(reader, json);
            case 'CumsumOptions': return tflite.CumsumOptions.decodeText(reader, json);
            case 'CallOnceOptions': return tflite.CallOnceOptions.decodeText(reader, json);
            case 'BroadcastToOptions': return tflite.BroadcastToOptions.decodeText(reader, json);
            case 'Rfft2dOptions': return tflite.Rfft2dOptions.decodeText(reader, json);
            case 'Conv3DOptions': return tflite.Conv3DOptions.decodeText(reader, json);
            case 'HashtableOptions': return tflite.HashtableOptions.decodeText(reader, json);
            case 'HashtableFindOptions': return tflite.HashtableFindOptions.decodeText(reader, json);
            case 'HashtableImportOptions': return tflite.HashtableImportOptions.decodeText(reader, json);
            case 'HashtableSizeOptions': return tflite.HashtableSizeOptions.decodeText(reader, json);
            case 'VarHandleOptions': return tflite.VarHandleOptions.decodeText(reader, json);
            case 'ReadVariableOptions': return tflite.ReadVariableOptions.decodeText(reader, json);
            case 'AssignVariableOptions': return tflite.AssignVariableOptions.decodeText(reader, json);
            case 'RandomOptions': return tflite.RandomOptions.decodeText(reader, json);
            case 'BucketizeOptions': return tflite.BucketizeOptions.decodeText(reader, json);
            case 'GeluOptions': return tflite.GeluOptions.decodeText(reader, json);
            case 'DynamicUpdateSliceOptions': return tflite.DynamicUpdateSliceOptions.decodeText(reader, json);
            case 'UnsortedSegmentProdOptions': return tflite.UnsortedSegmentProdOptions.decodeText(reader, json);
            case 'UnsortedSegmentMaxOptions': return tflite.UnsortedSegmentMaxOptions.decodeText(reader, json);
            case 'UnsortedSegmentMinOptions': return tflite.UnsortedSegmentMinOptions.decodeText(reader, json);
            case 'UnsortedSegmentSumOptions': return tflite.UnsortedSegmentSumOptions.decodeText(reader, json);
            case 'ATan2Options': return tflite.ATan2Options.decodeText(reader, json);
            case 'SignOptions': return tflite.SignOptions.decodeText(reader, json);
            case 'BitcastOptions': return tflite.BitcastOptions.decodeText(reader, json);
            case 'BitwiseXorOptions': return tflite.BitwiseXorOptions.decodeText(reader, json);
            case 'RightShiftOptions': return tflite.RightShiftOptions.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tflite.BuiltinOptions2 = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tflite.StablehloConcatenateOptions.decode(reader, position);
            case 2: return tflite.StablehloBroadcastInDimOptions.decode(reader, position);
            case 3: return tflite.StablehloSliceOptions.decode(reader, position);
            case 4: return tflite.StablehloConvolutionOptions.decode(reader, position);
            case 5: return tflite.StablehloCustomCallOptions.decode(reader, position);
            case 6: return tflite.StablehloReduceOptions.decode(reader, position);
            case 7: return tflite.StablehloScatterOptions.decode(reader, position);
            case 8: return tflite.StablehloCompareOptions.decode(reader, position);
            case 9: return tflite.StablehloDynamicSliceOptions.decode(reader, position);
            case 10: return tflite.StablehloPadOptions.decode(reader, position);
            case 11: return tflite.StablehloIotaOptions.decode(reader, position);
            case 12: return tflite.StablehloDotGeneralOptions.decode(reader, position);
            case 13: return tflite.StablehloReduceWindowOptions.decode(reader, position);
            case 14: return tflite.StablehloSortOptions.decode(reader, position);
            case 15: return tflite.StablehloWhileOptions.decode(reader, position);
            case 16: return tflite.StablehloGatherOptions.decode(reader, position);
            case 17: return tflite.StablehloTransposeOptions.decode(reader, position);
            case 18: return tflite.DilateOptions.decode(reader, position);
            case 19: return tflite.StablehloRngBitGeneratorOptions.decode(reader, position);
            case 20: return tflite.ReduceWindowOptions.decode(reader, position);
            case 21: return tflite.StableHLOCompositeOptions.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'StablehloConcatenateOptions': return tflite.StablehloConcatenateOptions.decodeText(reader, json);
            case 'StablehloBroadcastInDimOptions': return tflite.StablehloBroadcastInDimOptions.decodeText(reader, json);
            case 'StablehloSliceOptions': return tflite.StablehloSliceOptions.decodeText(reader, json);
            case 'StablehloConvolutionOptions': return tflite.StablehloConvolutionOptions.decodeText(reader, json);
            case 'StablehloCustomCallOptions': return tflite.StablehloCustomCallOptions.decodeText(reader, json);
            case 'StablehloReduceOptions': return tflite.StablehloReduceOptions.decodeText(reader, json);
            case 'StablehloScatterOptions': return tflite.StablehloScatterOptions.decodeText(reader, json);
            case 'StablehloCompareOptions': return tflite.StablehloCompareOptions.decodeText(reader, json);
            case 'StablehloDynamicSliceOptions': return tflite.StablehloDynamicSliceOptions.decodeText(reader, json);
            case 'StablehloPadOptions': return tflite.StablehloPadOptions.decodeText(reader, json);
            case 'StablehloIotaOptions': return tflite.StablehloIotaOptions.decodeText(reader, json);
            case 'StablehloDotGeneralOptions': return tflite.StablehloDotGeneralOptions.decodeText(reader, json);
            case 'StablehloReduceWindowOptions': return tflite.StablehloReduceWindowOptions.decodeText(reader, json);
            case 'StablehloSortOptions': return tflite.StablehloSortOptions.decodeText(reader, json);
            case 'StablehloWhileOptions': return tflite.StablehloWhileOptions.decodeText(reader, json);
            case 'StablehloGatherOptions': return tflite.StablehloGatherOptions.decodeText(reader, json);
            case 'StablehloTransposeOptions': return tflite.StablehloTransposeOptions.decodeText(reader, json);
            case 'DilateOptions': return tflite.DilateOptions.decodeText(reader, json);
            case 'StablehloRngBitGeneratorOptions': return tflite.StablehloRngBitGeneratorOptions.decodeText(reader, json);
            case 'ReduceWindowOptions': return tflite.ReduceWindowOptions.decodeText(reader, json);
            case 'StableHLOCompositeOptions': return tflite.StableHLOCompositeOptions.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tflite.StablehloGatherOptions = class StablehloGatherOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloGatherOptions();
        $.offset_dims = reader.int64s_(position, 4);
        $.collapsed_slice_dims = reader.int64s_(position, 6);
        $.start_index_map = reader.int64s_(position, 8);
        $.index_vector_dim = reader.int64_(position, 10, 0n);
        $.slice_sizes = reader.int64s_(position, 12);
        $.indices_are_sorted = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloGatherOptions();
        $.offset_dims = reader.array(json.offset_dims);
        $.collapsed_slice_dims = reader.array(json.collapsed_slice_dims);
        $.start_index_map = reader.array(json.start_index_map);
        $.index_vector_dim = reader.int64(json.index_vector_dim, 0n);
        $.slice_sizes = reader.array(json.slice_sizes);
        $.indices_are_sorted = reader.value(json.indices_are_sorted, false);
        return $;
    }
};

tflite.StablehloTransposeOptions = class StablehloTransposeOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloTransposeOptions();
        $.permutation = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloTransposeOptions();
        $.permutation = reader.array(json.permutation);
        return $;
    }
};

tflite.StablehloPrecisionConfig = {
    DEFAULT: 0,
    HIGH: 1,
    HIGHEST: 2
};

tflite.StablehloDotGeneralOptions = class StablehloDotGeneralOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloDotGeneralOptions();
        $.lhs_batching_dimensions = reader.int64s_(position, 4);
        $.rhs_batching_dimensions = reader.int64s_(position, 6);
        $.lhs_contracting_dimensions = reader.int64s_(position, 8);
        $.rhs_contracting_dimensions = reader.int64s_(position, 10);
        $.precision_config = reader.array(position, 12, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloDotGeneralOptions();
        $.lhs_batching_dimensions = reader.array(json.lhs_batching_dimensions);
        $.rhs_batching_dimensions = reader.array(json.rhs_batching_dimensions);
        $.lhs_contracting_dimensions = reader.array(json.lhs_contracting_dimensions);
        $.rhs_contracting_dimensions = reader.array(json.rhs_contracting_dimensions);
        $.precision_config = reader.objects(json.precision_config, tflite.StablehloPrecisionConfig);
        return $;
    }
};

tflite.StablehloReduceWindowOptions = class StablehloReduceWindowOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloReduceWindowOptions();
        $.window_dimensions = reader.int64s_(position, 4);
        $.window_strides = reader.int64s_(position, 6);
        $.base_dilations = reader.int64s_(position, 8);
        $.window_dilations = reader.int64s_(position, 10);
        $.padding = reader.int64s_(position, 12);
        $.body_subgraph_index = reader.int32_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloReduceWindowOptions();
        $.window_dimensions = reader.array(json.window_dimensions);
        $.window_strides = reader.array(json.window_strides);
        $.base_dilations = reader.array(json.base_dilations);
        $.window_dilations = reader.array(json.window_dilations);
        $.padding = reader.array(json.padding);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

tflite.StablehloWhileOptions = class StablehloWhileOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloWhileOptions();
        $.cond_subgraph_index = reader.int32_(position, 4, 0);
        $.body_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloWhileOptions();
        $.cond_subgraph_index = reader.value(json.cond_subgraph_index, 0);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

tflite.StablehloSortOptions = class StablehloSortOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloSortOptions();
        $.dimension = reader.int64_(position, 4, 0n);
        $.is_stable = reader.bool_(position, 6, false);
        $.comparator_subgraph_index = reader.int32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloSortOptions();
        $.dimension = reader.int64(json.dimension, 0n);
        $.is_stable = reader.value(json.is_stable, false);
        $.comparator_subgraph_index = reader.value(json.comparator_subgraph_index, 0);
        return $;
    }
};

tflite.StablehloConcatenateOptions = class StablehloConcatenateOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloConcatenateOptions();
        $.dimension = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloConcatenateOptions();
        $.dimension = reader.int64(json.dimension, 0n);
        return $;
    }
};

tflite.StablehloBroadcastInDimOptions = class StablehloBroadcastInDimOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloBroadcastInDimOptions();
        $.broadcast_dimensions = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloBroadcastInDimOptions();
        $.broadcast_dimensions = reader.array(json.broadcast_dimensions);
        return $;
    }
};

tflite.StablehloComparisonDirection = {
    STABLEHLO_COMPARISON_DIRECTION_EQ: 0,
    STABLEHLO_COMPARISON_DIRECTION_NE: 1,
    STABLEHLO_COMPARISON_DIRECTION_GE: 2,
    STABLEHLO_COMPARISON_DIRECTION_GT: 3,
    STABLEHLO_COMPARISON_DIRECTION_LE: 4,
    STABLEHLO_COMPARISON_DIRECTION_LT: 5
};

tflite.StablehloComparisonType = {
    STABLEHLO_COMPARISON_TYPE_NOTYPE: 0,
    STABLEHLO_COMPARISON_TYPE_FLOAT: 1,
    STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER: 2,
    STABLEHLO_COMPARISON_TYPE_SIGNED: 3,
    STABLEHLO_COMPARISON_TYPE_UNSIGNED: 4
};

tflite.StablehloCompareOptions = class StablehloCompareOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloCompareOptions();
        $.comparison_direction = reader.uint32_(position, 4, 0);
        $.compare_type = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloCompareOptions();
        $.comparison_direction = tflite.StablehloComparisonDirection[json.comparison_direction];
        $.compare_type = tflite.StablehloComparisonType[json.compare_type];
        return $;
    }
};

tflite.StablehloDynamicSliceOptions = class StablehloDynamicSliceOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloDynamicSliceOptions();
        $.slice_sizes = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloDynamicSliceOptions();
        $.slice_sizes = reader.array(json.slice_sizes);
        return $;
    }
};

tflite.StablehloPadOptions = class StablehloPadOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloPadOptions();
        $.edge_padding_low = reader.int64s_(position, 4);
        $.edge_padding_high = reader.int64s_(position, 6);
        $.interior_padding = reader.int64s_(position, 8);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloPadOptions();
        $.edge_padding_low = reader.array(json.edge_padding_low);
        $.edge_padding_high = reader.array(json.edge_padding_high);
        $.interior_padding = reader.array(json.interior_padding);
        return $;
    }
};

tflite.StablehloIotaOptions = class StablehloIotaOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloIotaOptions();
        $.iota_dimension = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloIotaOptions();
        $.iota_dimension = reader.int64(json.iota_dimension, 0n);
        return $;
    }
};

tflite.StablehloCustomCallOptions = class StablehloCustomCallOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloCustomCallOptions();
        $.call_target_name = reader.string_(position, 4, null);
        $.has_side_effect = reader.bool_(position, 6, false);
        $.backend_config = reader.string_(position, 8, null);
        $.api_version = reader.int32_(position, 10, 0);
        $.called_computations = reader.array(position, 12, Int32Array);
        $.custom_attributes = reader.array(position, 14, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloCustomCallOptions();
        $.call_target_name = reader.value(json.call_target_name, null);
        $.has_side_effect = reader.value(json.has_side_effect, false);
        $.backend_config = reader.value(json.backend_config, null);
        $.api_version = reader.value(json.api_version, 0);
        $.called_computations = reader.array(json.called_computations, Int32Array);
        $.custom_attributes = reader.array(json.custom_attributes, Uint8Array);
        return $;
    }
};

tflite.StablehloReduceOptions = class StablehloReduceOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloReduceOptions();
        $.dimensions = reader.int64s_(position, 4);
        $.body_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloReduceOptions();
        $.dimensions = reader.array(json.dimensions);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

tflite.StablehloSliceOptions = class StablehloSliceOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloSliceOptions();
        $.start_indices = reader.int64s_(position, 4);
        $.limit_indices = reader.int64s_(position, 6);
        $.strides = reader.int64s_(position, 8);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloSliceOptions();
        $.start_indices = reader.array(json.start_indices);
        $.limit_indices = reader.array(json.limit_indices);
        $.strides = reader.array(json.strides);
        return $;
    }
};

tflite.StablehloConvolutionOptions = class StablehloConvolutionOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloConvolutionOptions();
        $.window_strides = reader.int64s_(position, 4);
        $.padding = reader.int64s_(position, 6);
        $.lhs_dilation = reader.int64s_(position, 8);
        $.rhs_dilation = reader.int64s_(position, 10);
        $.window_reversal = reader.bools_(position, 12);
        $.input_batch_dimension = reader.int64_(position, 14, 0n);
        $.input_feature_dimension = reader.int64_(position, 16, 0n);
        $.input_spatial_dimensions = reader.int64s_(position, 18);
        $.kernel_input_feature_dimension = reader.int64_(position, 20, 0n);
        $.kernel_output_feature_dimension = reader.int64_(position, 22, 0n);
        $.kernel_spatial_dimensions = reader.int64s_(position, 24);
        $.output_batch_dimension = reader.int64_(position, 26, 0n);
        $.output_feature_dimension = reader.int64_(position, 28, 0n);
        $.output_spatial_dimensions = reader.int64s_(position, 30);
        $.feature_group_count = reader.int64_(position, 32, 0n);
        $.batch_group_count = reader.int64_(position, 34, 0n);
        $.precision_config = reader.array(position, 36, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloConvolutionOptions();
        $.window_strides = reader.array(json.window_strides);
        $.padding = reader.array(json.padding);
        $.lhs_dilation = reader.array(json.lhs_dilation);
        $.rhs_dilation = reader.array(json.rhs_dilation);
        $.window_reversal = reader.array(json.window_reversal);
        $.input_batch_dimension = reader.int64(json.input_batch_dimension, 0n);
        $.input_feature_dimension = reader.int64(json.input_feature_dimension, 0n);
        $.input_spatial_dimensions = reader.array(json.input_spatial_dimensions);
        $.kernel_input_feature_dimension = reader.int64(json.kernel_input_feature_dimension, 0n);
        $.kernel_output_feature_dimension = reader.int64(json.kernel_output_feature_dimension, 0n);
        $.kernel_spatial_dimensions = reader.array(json.kernel_spatial_dimensions);
        $.output_batch_dimension = reader.int64(json.output_batch_dimension, 0n);
        $.output_feature_dimension = reader.int64(json.output_feature_dimension, 0n);
        $.output_spatial_dimensions = reader.array(json.output_spatial_dimensions);
        $.feature_group_count = reader.int64(json.feature_group_count, 0n);
        $.batch_group_count = reader.int64(json.batch_group_count, 0n);
        $.precision_config = reader.objects(json.precision_config, tflite.StablehloPrecisionConfig);
        return $;
    }
};

tflite.StablehloScatterOptions = class StablehloScatterOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloScatterOptions();
        $.indices_are_sorted = reader.bool_(position, 4, false);
        $.update_window_dims = reader.int64s_(position, 6);
        $.inserted_window_dims = reader.int64s_(position, 8);
        $.scatter_dims_to_operand_dims = reader.int64s_(position, 10);
        $.index_vector_dim = reader.int64_(position, 12, 0n);
        $.unique_indices = reader.bool_(position, 14, false);
        $.update_computation_subgraph_index = reader.int32_(position, 16, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloScatterOptions();
        $.indices_are_sorted = reader.value(json.indices_are_sorted, false);
        $.update_window_dims = reader.array(json.update_window_dims);
        $.inserted_window_dims = reader.array(json.inserted_window_dims);
        $.scatter_dims_to_operand_dims = reader.array(json.scatter_dims_to_operand_dims);
        $.index_vector_dim = reader.int64(json.index_vector_dim, 0n);
        $.unique_indices = reader.value(json.unique_indices, false);
        $.update_computation_subgraph_index = reader.value(json.update_computation_subgraph_index, 0);
        return $;
    }
};

tflite.RngAlgorithm = {
    DEFAULT: 0,
    PHILOX: 1,
    THREEFRY: 2
};

tflite.StablehloRngBitGeneratorOptions = class StablehloRngBitGeneratorOptions {

    static decode(reader, position) {
        const $ = new tflite.StablehloRngBitGeneratorOptions();
        $.algorithm = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StablehloRngBitGeneratorOptions();
        $.algorithm = tflite.RngAlgorithm[json.algorithm];
        return $;
    }
};

tflite.Padding = {
    SAME: 0,
    VALID: 1
};

tflite.ActivationFunctionType = {
    NONE: 0,
    RELU: 1,
    RELU_N1_TO_1: 2,
    RELU6: 3,
    TANH: 4,
    SIGN_BIT: 5
};

tflite.Conv2DOptions = class Conv2DOptions {

    static decode(reader, position) {
        const $ = new tflite.Conv2DOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_w = reader.int32_(position, 6, 0);
        $.stride_h = reader.int32_(position, 8, 0);
        $.fused_activation_function = reader.int8_(position, 10, 0);
        $.dilation_w_factor = reader.int32_(position, 12, 1);
        $.dilation_h_factor = reader.int32_(position, 14, 1);
        $.quantized_bias_type = reader.int8_(position, 16, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Conv2DOptions();
        $.padding = tflite.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.dilation_w_factor = reader.value(json.dilation_w_factor, 1);
        $.dilation_h_factor = reader.value(json.dilation_h_factor, 1);
        $.quantized_bias_type = tflite.TensorType[json.quantized_bias_type];
        return $;
    }
};

tflite.Conv3DOptions = class Conv3DOptions {

    static decode(reader, position) {
        const $ = new tflite.Conv3DOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_d = reader.int32_(position, 6, 0);
        $.stride_w = reader.int32_(position, 8, 0);
        $.stride_h = reader.int32_(position, 10, 0);
        $.fused_activation_function = reader.int8_(position, 12, 0);
        $.dilation_d_factor = reader.int32_(position, 14, 1);
        $.dilation_w_factor = reader.int32_(position, 16, 1);
        $.dilation_h_factor = reader.int32_(position, 18, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Conv3DOptions();
        $.padding = tflite.Padding[json.padding];
        $.stride_d = reader.value(json.stride_d, 0);
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.dilation_d_factor = reader.value(json.dilation_d_factor, 1);
        $.dilation_w_factor = reader.value(json.dilation_w_factor, 1);
        $.dilation_h_factor = reader.value(json.dilation_h_factor, 1);
        return $;
    }
};

tflite.Pool2DOptions = class Pool2DOptions {

    static decode(reader, position) {
        const $ = new tflite.Pool2DOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_w = reader.int32_(position, 6, 0);
        $.stride_h = reader.int32_(position, 8, 0);
        $.filter_width = reader.int32_(position, 10, 0);
        $.filter_height = reader.int32_(position, 12, 0);
        $.fused_activation_function = reader.int8_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Pool2DOptions();
        $.padding = tflite.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.filter_width = reader.value(json.filter_width, 0);
        $.filter_height = reader.value(json.filter_height, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

tflite.DepthwiseConv2DOptions = class DepthwiseConv2DOptions {

    static decode(reader, position) {
        const $ = new tflite.DepthwiseConv2DOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_w = reader.int32_(position, 6, 0);
        $.stride_h = reader.int32_(position, 8, 0);
        $.depth_multiplier = reader.int32_(position, 10, 0);
        $.fused_activation_function = reader.int8_(position, 12, 0);
        $.dilation_w_factor = reader.int32_(position, 14, 1);
        $.dilation_h_factor = reader.int32_(position, 16, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.DepthwiseConv2DOptions();
        $.padding = tflite.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.depth_multiplier = reader.value(json.depth_multiplier, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.dilation_w_factor = reader.value(json.dilation_w_factor, 1);
        $.dilation_h_factor = reader.value(json.dilation_h_factor, 1);
        return $;
    }
};

tflite.ConcatEmbeddingsOptions = class ConcatEmbeddingsOptions {

    static decode(reader, position) {
        const $ = new tflite.ConcatEmbeddingsOptions();
        $.num_channels = reader.int32_(position, 4, 0);
        $.num_columns_per_channel = reader.array(position, 6, Int32Array);
        $.embedding_dim_per_channel = reader.array(position, 8, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ConcatEmbeddingsOptions();
        $.num_channels = reader.value(json.num_channels, 0);
        $.num_columns_per_channel = reader.array(json.num_columns_per_channel, Int32Array);
        $.embedding_dim_per_channel = reader.array(json.embedding_dim_per_channel, Int32Array);
        return $;
    }
};

tflite.LSHProjectionType = {
    UNKNOWN: 0,
    SPARSE: 1,
    DENSE: 2
};

tflite.LSHProjectionOptions = class LSHProjectionOptions {

    static decode(reader, position) {
        const $ = new tflite.LSHProjectionOptions();
        $.type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.LSHProjectionOptions();
        $.type = tflite.LSHProjectionType[json.type];
        return $;
    }
};

tflite.SVDFOptions = class SVDFOptions {

    static decode(reader, position) {
        const $ = new tflite.SVDFOptions();
        $.rank = reader.int32_(position, 4, 0);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SVDFOptions();
        $.rank = reader.value(json.rank, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.RNNOptions = class RNNOptions {

    static decode(reader, position) {
        const $ = new tflite.RNNOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.RNNOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.SequenceRNNOptions = class SequenceRNNOptions {

    static decode(reader, position) {
        const $ = new tflite.SequenceRNNOptions();
        $.time_major = reader.bool_(position, 4, false);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SequenceRNNOptions();
        $.time_major = reader.value(json.time_major, false);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.BidirectionalSequenceRNNOptions = class BidirectionalSequenceRNNOptions {

    static decode(reader, position) {
        const $ = new tflite.BidirectionalSequenceRNNOptions();
        $.time_major = reader.bool_(position, 4, false);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        $.merge_outputs = reader.bool_(position, 8, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.BidirectionalSequenceRNNOptions();
        $.time_major = reader.value(json.time_major, false);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.merge_outputs = reader.value(json.merge_outputs, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.FullyConnectedOptionsWeightsFormat = {
    DEFAULT: 0,
    SHUFFLED4x16INT8: 1
};

tflite.FullyConnectedOptions = class FullyConnectedOptions {

    static decode(reader, position) {
        const $ = new tflite.FullyConnectedOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.weights_format = reader.int8_(position, 6, 0);
        $.keep_num_dims = reader.bool_(position, 8, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 10, false);
        $.quantized_bias_type = reader.int8_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.FullyConnectedOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.weights_format = tflite.FullyConnectedOptionsWeightsFormat[json.weights_format];
        $.keep_num_dims = reader.value(json.keep_num_dims, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        $.quantized_bias_type = tflite.TensorType[json.quantized_bias_type];
        return $;
    }
};

tflite.SoftmaxOptions = class SoftmaxOptions {

    static decode(reader, position) {
        const $ = new tflite.SoftmaxOptions();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SoftmaxOptions();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

tflite.ConcatenationOptions = class ConcatenationOptions {

    static decode(reader, position) {
        const $ = new tflite.ConcatenationOptions();
        $.axis = reader.int32_(position, 4, 0);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ConcatenationOptions();
        $.axis = reader.value(json.axis, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

tflite.AddOptions = class AddOptions {

    static decode(reader, position) {
        const $ = new tflite.AddOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.pot_scale_int16 = reader.bool_(position, 6, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.AddOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.pot_scale_int16 = reader.value(json.pot_scale_int16, true);
        return $;
    }
};

tflite.MulOptions = class MulOptions {

    static decode(reader, position) {
        const $ = new tflite.MulOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.MulOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

tflite.L2NormOptions = class L2NormOptions {

    static decode(reader, position) {
        const $ = new tflite.L2NormOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.L2NormOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

tflite.LocalResponseNormalizationOptions = class LocalResponseNormalizationOptions {

    static decode(reader, position) {
        const $ = new tflite.LocalResponseNormalizationOptions();
        $.radius = reader.int32_(position, 4, 0);
        $.bias = reader.float32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.LocalResponseNormalizationOptions();
        $.radius = reader.value(json.radius, 0);
        $.bias = reader.value(json.bias, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

tflite.LSTMKernelType = {
    FULL: 0,
    BASIC: 1
};

tflite.LSTMOptions = class LSTMOptions {

    static decode(reader, position) {
        const $ = new tflite.LSTMOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.cell_clip = reader.float32_(position, 6, 0);
        $.proj_clip = reader.float32_(position, 8, 0);
        $.kernel_type = reader.int8_(position, 10, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.LSTMOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.cell_clip = reader.value(json.cell_clip, 0);
        $.proj_clip = reader.value(json.proj_clip, 0);
        $.kernel_type = tflite.LSTMKernelType[json.kernel_type];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.UnidirectionalSequenceLSTMOptions = class UnidirectionalSequenceLSTMOptions {

    static decode(reader, position) {
        const $ = new tflite.UnidirectionalSequenceLSTMOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.cell_clip = reader.float32_(position, 6, 0);
        $.proj_clip = reader.float32_(position, 8, 0);
        $.time_major = reader.bool_(position, 10, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 12, false);
        $.diagonal_recurrent_tensors = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.UnidirectionalSequenceLSTMOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.cell_clip = reader.value(json.cell_clip, 0);
        $.proj_clip = reader.value(json.proj_clip, 0);
        $.time_major = reader.value(json.time_major, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        $.diagonal_recurrent_tensors = reader.value(json.diagonal_recurrent_tensors, false);
        return $;
    }
};

tflite.BidirectionalSequenceLSTMOptions = class BidirectionalSequenceLSTMOptions {

    static decode(reader, position) {
        const $ = new tflite.BidirectionalSequenceLSTMOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.cell_clip = reader.float32_(position, 6, 0);
        $.proj_clip = reader.float32_(position, 8, 0);
        $.merge_outputs = reader.bool_(position, 10, false);
        $.time_major = reader.bool_(position, 12, true);
        $.asymmetric_quantize_inputs = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.BidirectionalSequenceLSTMOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.cell_clip = reader.value(json.cell_clip, 0);
        $.proj_clip = reader.value(json.proj_clip, 0);
        $.merge_outputs = reader.value(json.merge_outputs, false);
        $.time_major = reader.value(json.time_major, true);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.ResizeBilinearOptions = class ResizeBilinearOptions {

    static decode(reader, position) {
        const $ = new tflite.ResizeBilinearOptions();
        $.new_height = reader.int32_(position, 4, 0);
        $.new_width = reader.int32_(position, 6, 0);
        $.align_corners = reader.bool_(position, 8, false);
        $.half_pixel_centers = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ResizeBilinearOptions();
        $.new_height = reader.value(json.new_height, 0);
        $.new_width = reader.value(json.new_width, 0);
        $.align_corners = reader.value(json.align_corners, false);
        $.half_pixel_centers = reader.value(json.half_pixel_centers, false);
        return $;
    }
};

tflite.ResizeNearestNeighborOptions = class ResizeNearestNeighborOptions {

    static decode(reader, position) {
        const $ = new tflite.ResizeNearestNeighborOptions();
        $.align_corners = reader.bool_(position, 4, false);
        $.half_pixel_centers = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ResizeNearestNeighborOptions();
        $.align_corners = reader.value(json.align_corners, false);
        $.half_pixel_centers = reader.value(json.half_pixel_centers, false);
        return $;
    }
};

tflite.CallOptions = class CallOptions {

    static decode(reader, position) {
        const $ = new tflite.CallOptions();
        $.subgraph = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.CallOptions();
        $.subgraph = reader.value(json.subgraph, 0);
        return $;
    }
};

tflite.PadOptions = class PadOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.PadOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.PadOptions();
        return $;
    }
};

tflite.PadV2Options = class PadV2Options {

    static decode(/* reader, position */) {
        const $ = new tflite.PadV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.PadV2Options();
        return $;
    }
};

tflite.ReshapeOptions = class ReshapeOptions {

    static decode(reader, position) {
        const $ = new tflite.ReshapeOptions();
        $.new_shape = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ReshapeOptions();
        $.new_shape = reader.array(json.new_shape, Int32Array);
        return $;
    }
};

tflite.SpaceToBatchNDOptions = class SpaceToBatchNDOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SpaceToBatchNDOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SpaceToBatchNDOptions();
        return $;
    }
};

tflite.BatchToSpaceNDOptions = class BatchToSpaceNDOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.BatchToSpaceNDOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.BatchToSpaceNDOptions();
        return $;
    }
};

tflite.SkipGramOptions = class SkipGramOptions {

    static decode(reader, position) {
        const $ = new tflite.SkipGramOptions();
        $.ngram_size = reader.int32_(position, 4, 0);
        $.max_skip_size = reader.int32_(position, 6, 0);
        $.include_all_ngrams = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SkipGramOptions();
        $.ngram_size = reader.value(json.ngram_size, 0);
        $.max_skip_size = reader.value(json.max_skip_size, 0);
        $.include_all_ngrams = reader.value(json.include_all_ngrams, false);
        return $;
    }
};

tflite.SpaceToDepthOptions = class SpaceToDepthOptions {

    static decode(reader, position) {
        const $ = new tflite.SpaceToDepthOptions();
        $.block_size = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SpaceToDepthOptions();
        $.block_size = reader.value(json.block_size, 0);
        return $;
    }
};

tflite.DepthToSpaceOptions = class DepthToSpaceOptions {

    static decode(reader, position) {
        const $ = new tflite.DepthToSpaceOptions();
        $.block_size = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.DepthToSpaceOptions();
        $.block_size = reader.value(json.block_size, 0);
        return $;
    }
};

tflite.SubOptions = class SubOptions {

    static decode(reader, position) {
        const $ = new tflite.SubOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.pot_scale_int16 = reader.bool_(position, 6, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SubOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.pot_scale_int16 = reader.value(json.pot_scale_int16, true);
        return $;
    }
};

tflite.DivOptions = class DivOptions {

    static decode(reader, position) {
        const $ = new tflite.DivOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.DivOptions();
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

tflite.TopKV2Options = class TopKV2Options {

    static decode(/* reader, position */) {
        const $ = new tflite.TopKV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.TopKV2Options();
        return $;
    }
};

tflite.CombinerType = {
    SUM: 0,
    MEAN: 1,
    SQRTN: 2
};

tflite.EmbeddingLookupSparseOptions = class EmbeddingLookupSparseOptions {

    static decode(reader, position) {
        const $ = new tflite.EmbeddingLookupSparseOptions();
        $.combiner = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.EmbeddingLookupSparseOptions();
        $.combiner = tflite.CombinerType[json.combiner];
        return $;
    }
};

tflite.GatherOptions = class GatherOptions {

    static decode(reader, position) {
        const $ = new tflite.GatherOptions();
        $.axis = reader.int32_(position, 4, 0);
        $.batch_dims = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.GatherOptions();
        $.axis = reader.value(json.axis, 0);
        $.batch_dims = reader.value(json.batch_dims, 0);
        return $;
    }
};

tflite.TransposeOptions = class TransposeOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.TransposeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.TransposeOptions();
        return $;
    }
};

tflite.ExpOptions = class ExpOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.ExpOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ExpOptions();
        return $;
    }
};

tflite.CosOptions = class CosOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.CosOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.CosOptions();
        return $;
    }
};

tflite.ReducerOptions = class ReducerOptions {

    static decode(reader, position) {
        const $ = new tflite.ReducerOptions();
        $.keep_dims = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ReducerOptions();
        $.keep_dims = reader.value(json.keep_dims, false);
        return $;
    }
};

tflite.SqueezeOptions = class SqueezeOptions {

    static decode(reader, position) {
        const $ = new tflite.SqueezeOptions();
        $.squeeze_dims = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SqueezeOptions();
        $.squeeze_dims = reader.array(json.squeeze_dims, Int32Array);
        return $;
    }
};

tflite.SplitOptions = class SplitOptions {

    static decode(reader, position) {
        const $ = new tflite.SplitOptions();
        $.num_splits = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SplitOptions();
        $.num_splits = reader.value(json.num_splits, 0);
        return $;
    }
};

tflite.SplitVOptions = class SplitVOptions {

    static decode(reader, position) {
        const $ = new tflite.SplitVOptions();
        $.num_splits = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SplitVOptions();
        $.num_splits = reader.value(json.num_splits, 0);
        return $;
    }
};

tflite.StridedSliceOptions = class StridedSliceOptions {

    static decode(reader, position) {
        const $ = new tflite.StridedSliceOptions();
        $.begin_mask = reader.int32_(position, 4, 0);
        $.end_mask = reader.int32_(position, 6, 0);
        $.ellipsis_mask = reader.int32_(position, 8, 0);
        $.new_axis_mask = reader.int32_(position, 10, 0);
        $.shrink_axis_mask = reader.int32_(position, 12, 0);
        $.offset = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StridedSliceOptions();
        $.begin_mask = reader.value(json.begin_mask, 0);
        $.end_mask = reader.value(json.end_mask, 0);
        $.ellipsis_mask = reader.value(json.ellipsis_mask, 0);
        $.new_axis_mask = reader.value(json.new_axis_mask, 0);
        $.shrink_axis_mask = reader.value(json.shrink_axis_mask, 0);
        $.offset = reader.value(json.offset, false);
        return $;
    }
};

tflite.LogSoftmaxOptions = class LogSoftmaxOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.LogSoftmaxOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.LogSoftmaxOptions();
        return $;
    }
};

tflite.CastOptions = class CastOptions {

    static decode(reader, position) {
        const $ = new tflite.CastOptions();
        $.in_data_type = reader.int8_(position, 4, 0);
        $.out_data_type = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.CastOptions();
        $.in_data_type = tflite.TensorType[json.in_data_type];
        $.out_data_type = tflite.TensorType[json.out_data_type];
        return $;
    }
};

tflite.DequantizeOptions = class DequantizeOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.DequantizeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.DequantizeOptions();
        return $;
    }
};

tflite.MaximumMinimumOptions = class MaximumMinimumOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.MaximumMinimumOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.MaximumMinimumOptions();
        return $;
    }
};

tflite.TileOptions = class TileOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.TileOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.TileOptions();
        return $;
    }
};

tflite.ArgMaxOptions = class ArgMaxOptions {

    static decode(reader, position) {
        const $ = new tflite.ArgMaxOptions();
        $.output_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ArgMaxOptions();
        $.output_type = tflite.TensorType[json.output_type];
        return $;
    }
};

tflite.ArgMinOptions = class ArgMinOptions {

    static decode(reader, position) {
        const $ = new tflite.ArgMinOptions();
        $.output_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ArgMinOptions();
        $.output_type = tflite.TensorType[json.output_type];
        return $;
    }
};

tflite.GreaterOptions = class GreaterOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.GreaterOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.GreaterOptions();
        return $;
    }
};

tflite.GreaterEqualOptions = class GreaterEqualOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.GreaterEqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.GreaterEqualOptions();
        return $;
    }
};

tflite.LessOptions = class LessOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.LessOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.LessOptions();
        return $;
    }
};

tflite.LessEqualOptions = class LessEqualOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.LessEqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.LessEqualOptions();
        return $;
    }
};

tflite.NegOptions = class NegOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.NegOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.NegOptions();
        return $;
    }
};

tflite.SelectOptions = class SelectOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SelectOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SelectOptions();
        return $;
    }
};

tflite.SliceOptions = class SliceOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SliceOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SliceOptions();
        return $;
    }
};

tflite.TransposeConvOptions = class TransposeConvOptions {

    static decode(reader, position) {
        const $ = new tflite.TransposeConvOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_w = reader.int32_(position, 6, 0);
        $.stride_h = reader.int32_(position, 8, 0);
        $.fused_activation_function = reader.int8_(position, 10, 0);
        $.quantized_bias_type = reader.int8_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.TransposeConvOptions();
        $.padding = tflite.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.fused_activation_function = tflite.ActivationFunctionType[json.fused_activation_function];
        $.quantized_bias_type = tflite.TensorType[json.quantized_bias_type];
        return $;
    }
};

tflite.ExpandDimsOptions = class ExpandDimsOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.ExpandDimsOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ExpandDimsOptions();
        return $;
    }
};

tflite.SparseToDenseOptions = class SparseToDenseOptions {

    static decode(reader, position) {
        const $ = new tflite.SparseToDenseOptions();
        $.validate_indices = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SparseToDenseOptions();
        $.validate_indices = reader.value(json.validate_indices, false);
        return $;
    }
};

tflite.EqualOptions = class EqualOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.EqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.EqualOptions();
        return $;
    }
};

tflite.NotEqualOptions = class NotEqualOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.NotEqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.NotEqualOptions();
        return $;
    }
};

tflite.ShapeOptions = class ShapeOptions {

    static decode(reader, position) {
        const $ = new tflite.ShapeOptions();
        $.out_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ShapeOptions();
        $.out_type = tflite.TensorType[json.out_type];
        return $;
    }
};

tflite.RankOptions = class RankOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.RankOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.RankOptions();
        return $;
    }
};

tflite.PowOptions = class PowOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.PowOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.PowOptions();
        return $;
    }
};

tflite.FakeQuantOptions = class FakeQuantOptions {

    static decode(reader, position) {
        const $ = new tflite.FakeQuantOptions();
        $.min = reader.float32_(position, 4, 0);
        $.max = reader.float32_(position, 6, 0);
        $.num_bits = reader.int32_(position, 8, 0);
        $.narrow_range = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.FakeQuantOptions();
        $.min = reader.value(json.min, 0);
        $.max = reader.value(json.max, 0);
        $.num_bits = reader.value(json.num_bits, 0);
        $.narrow_range = reader.value(json.narrow_range, false);
        return $;
    }
};

tflite.PackOptions = class PackOptions {

    static decode(reader, position) {
        const $ = new tflite.PackOptions();
        $.values_count = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.PackOptions();
        $.values_count = reader.value(json.values_count, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tflite.LogicalOrOptions = class LogicalOrOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.LogicalOrOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.LogicalOrOptions();
        return $;
    }
};

tflite.OneHotOptions = class OneHotOptions {

    static decode(reader, position) {
        const $ = new tflite.OneHotOptions();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.OneHotOptions();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tflite.AbsOptions = class AbsOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.AbsOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.AbsOptions();
        return $;
    }
};

tflite.HardSwishOptions = class HardSwishOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.HardSwishOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.HardSwishOptions();
        return $;
    }
};

tflite.LogicalAndOptions = class LogicalAndOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.LogicalAndOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.LogicalAndOptions();
        return $;
    }
};

tflite.LogicalNotOptions = class LogicalNotOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.LogicalNotOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.LogicalNotOptions();
        return $;
    }
};

tflite.UnpackOptions = class UnpackOptions {

    static decode(reader, position) {
        const $ = new tflite.UnpackOptions();
        $.num = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.UnpackOptions();
        $.num = reader.value(json.num, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tflite.FloorDivOptions = class FloorDivOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.FloorDivOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.FloorDivOptions();
        return $;
    }
};

tflite.SquareOptions = class SquareOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SquareOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SquareOptions();
        return $;
    }
};

tflite.ZerosLikeOptions = class ZerosLikeOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.ZerosLikeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ZerosLikeOptions();
        return $;
    }
};

tflite.FillOptions = class FillOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.FillOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.FillOptions();
        return $;
    }
};

tflite.FloorModOptions = class FloorModOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.FloorModOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.FloorModOptions();
        return $;
    }
};

tflite.RangeOptions = class RangeOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.RangeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.RangeOptions();
        return $;
    }
};

tflite.LeakyReluOptions = class LeakyReluOptions {

    static decode(reader, position) {
        const $ = new tflite.LeakyReluOptions();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.LeakyReluOptions();
        $.alpha = reader.value(json.alpha, 0);
        return $;
    }
};

tflite.SquaredDifferenceOptions = class SquaredDifferenceOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SquaredDifferenceOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SquaredDifferenceOptions();
        return $;
    }
};

tflite.MirrorPadMode = {
    REFLECT: 0,
    SYMMETRIC: 1
};

tflite.MirrorPadOptions = class MirrorPadOptions {

    static decode(reader, position) {
        const $ = new tflite.MirrorPadOptions();
        $.mode = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.MirrorPadOptions();
        $.mode = tflite.MirrorPadMode[json.mode];
        return $;
    }
};

tflite.UniqueOptions = class UniqueOptions {

    static decode(reader, position) {
        const $ = new tflite.UniqueOptions();
        $.idx_out_type = reader.int8_(position, 4, 2);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.UniqueOptions();
        $.idx_out_type = tflite.TensorType[json.idx_out_type];
        return $;
    }
};

tflite.ReverseV2Options = class ReverseV2Options {

    static decode(/* reader, position */) {
        const $ = new tflite.ReverseV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ReverseV2Options();
        return $;
    }
};

tflite.AddNOptions = class AddNOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.AddNOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.AddNOptions();
        return $;
    }
};

tflite.GatherNdOptions = class GatherNdOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.GatherNdOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.GatherNdOptions();
        return $;
    }
};

tflite.WhereOptions = class WhereOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.WhereOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.WhereOptions();
        return $;
    }
};

tflite.ReverseSequenceOptions = class ReverseSequenceOptions {

    static decode(reader, position) {
        const $ = new tflite.ReverseSequenceOptions();
        $.seq_dim = reader.int32_(position, 4, 0);
        $.batch_dim = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ReverseSequenceOptions();
        $.seq_dim = reader.value(json.seq_dim, 0);
        $.batch_dim = reader.value(json.batch_dim, 0);
        return $;
    }
};

tflite.MatrixDiagOptions = class MatrixDiagOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.MatrixDiagOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.MatrixDiagOptions();
        return $;
    }
};

tflite.QuantizeOptions = class QuantizeOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.QuantizeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.QuantizeOptions();
        return $;
    }
};

tflite.MatrixSetDiagOptions = class MatrixSetDiagOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.MatrixSetDiagOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.MatrixSetDiagOptions();
        return $;
    }
};

tflite.IfOptions = class IfOptions {

    static decode(reader, position) {
        const $ = new tflite.IfOptions();
        $.then_subgraph_index = reader.int32_(position, 4, 0);
        $.else_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.IfOptions();
        $.then_subgraph_index = reader.value(json.then_subgraph_index, 0);
        $.else_subgraph_index = reader.value(json.else_subgraph_index, 0);
        return $;
    }
};

tflite.CallOnceOptions = class CallOnceOptions {

    static decode(reader, position) {
        const $ = new tflite.CallOnceOptions();
        $.init_subgraph_index = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.CallOnceOptions();
        $.init_subgraph_index = reader.value(json.init_subgraph_index, 0);
        return $;
    }
};

tflite.WhileOptions = class WhileOptions {

    static decode(reader, position) {
        const $ = new tflite.WhileOptions();
        $.cond_subgraph_index = reader.int32_(position, 4, 0);
        $.body_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.WhileOptions();
        $.cond_subgraph_index = reader.value(json.cond_subgraph_index, 0);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

tflite.NonMaxSuppressionV4Options = class NonMaxSuppressionV4Options {

    static decode(/* reader, position */) {
        const $ = new tflite.NonMaxSuppressionV4Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.NonMaxSuppressionV4Options();
        return $;
    }
};

tflite.NonMaxSuppressionV5Options = class NonMaxSuppressionV5Options {

    static decode(/* reader, position */) {
        const $ = new tflite.NonMaxSuppressionV5Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.NonMaxSuppressionV5Options();
        return $;
    }
};

tflite.ScatterNdOptions = class ScatterNdOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.ScatterNdOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ScatterNdOptions();
        return $;
    }
};

tflite.SelectV2Options = class SelectV2Options {

    static decode(/* reader, position */) {
        const $ = new tflite.SelectV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SelectV2Options();
        return $;
    }
};

tflite.DensifyOptions = class DensifyOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.DensifyOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.DensifyOptions();
        return $;
    }
};

tflite.SegmentSumOptions = class SegmentSumOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SegmentSumOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SegmentSumOptions();
        return $;
    }
};

tflite.BatchMatMulOptions = class BatchMatMulOptions {

    static decode(reader, position) {
        const $ = new tflite.BatchMatMulOptions();
        $.adj_x = reader.bool_(position, 4, false);
        $.adj_y = reader.bool_(position, 6, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.BatchMatMulOptions();
        $.adj_x = reader.value(json.adj_x, false);
        $.adj_y = reader.value(json.adj_y, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

tflite.CumsumOptions = class CumsumOptions {

    static decode(reader, position) {
        const $ = new tflite.CumsumOptions();
        $.exclusive = reader.bool_(position, 4, false);
        $.reverse = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.CumsumOptions();
        $.exclusive = reader.value(json.exclusive, false);
        $.reverse = reader.value(json.reverse, false);
        return $;
    }
};

tflite.BroadcastToOptions = class BroadcastToOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.BroadcastToOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.BroadcastToOptions();
        return $;
    }
};

tflite.Rfft2dOptions = class Rfft2dOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.Rfft2dOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.Rfft2dOptions();
        return $;
    }
};

tflite.HashtableOptions = class HashtableOptions {

    static decode(reader, position) {
        const $ = new tflite.HashtableOptions();
        $.table_id = reader.int32_(position, 4, 0);
        $.key_dtype = reader.int8_(position, 6, 0);
        $.value_dtype = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.HashtableOptions();
        $.table_id = reader.value(json.table_id, 0);
        $.key_dtype = tflite.TensorType[json.key_dtype];
        $.value_dtype = tflite.TensorType[json.value_dtype];
        return $;
    }
};

tflite.HashtableFindOptions = class HashtableFindOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.HashtableFindOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.HashtableFindOptions();
        return $;
    }
};

tflite.HashtableImportOptions = class HashtableImportOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.HashtableImportOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.HashtableImportOptions();
        return $;
    }
};

tflite.HashtableSizeOptions = class HashtableSizeOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.HashtableSizeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.HashtableSizeOptions();
        return $;
    }
};

tflite.VarHandleOptions = class VarHandleOptions {

    static decode(reader, position) {
        const $ = new tflite.VarHandleOptions();
        $.container = reader.string_(position, 4, null);
        $.shared_name = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.VarHandleOptions();
        $.container = reader.value(json.container, null);
        $.shared_name = reader.value(json.shared_name, null);
        return $;
    }
};

tflite.ReadVariableOptions = class ReadVariableOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.ReadVariableOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ReadVariableOptions();
        return $;
    }
};

tflite.AssignVariableOptions = class AssignVariableOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.AssignVariableOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.AssignVariableOptions();
        return $;
    }
};

tflite.RandomOptions = class RandomOptions {

    static decode(reader, position) {
        const $ = new tflite.RandomOptions();
        $.seed = reader.int64_(position, 4, 0n);
        $.seed2 = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.RandomOptions();
        $.seed = reader.int64(json.seed, 0n);
        $.seed2 = reader.int64(json.seed2, 0n);
        return $;
    }
};

tflite.BucketizeOptions = class BucketizeOptions {

    static decode(reader, position) {
        const $ = new tflite.BucketizeOptions();
        $.boundaries = reader.array(position, 4, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.BucketizeOptions();
        $.boundaries = reader.array(json.boundaries, Float32Array);
        return $;
    }
};

tflite.GeluOptions = class GeluOptions {

    static decode(reader, position) {
        const $ = new tflite.GeluOptions();
        $.approximate = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.GeluOptions();
        $.approximate = reader.value(json.approximate, false);
        return $;
    }
};

tflite.DynamicUpdateSliceOptions = class DynamicUpdateSliceOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.DynamicUpdateSliceOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.DynamicUpdateSliceOptions();
        return $;
    }
};

tflite.UnsortedSegmentProdOptions = class UnsortedSegmentProdOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.UnsortedSegmentProdOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.UnsortedSegmentProdOptions();
        return $;
    }
};

tflite.UnsortedSegmentMaxOptions = class UnsortedSegmentMaxOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.UnsortedSegmentMaxOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.UnsortedSegmentMaxOptions();
        return $;
    }
};

tflite.UnsortedSegmentSumOptions = class UnsortedSegmentSumOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.UnsortedSegmentSumOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.UnsortedSegmentSumOptions();
        return $;
    }
};

tflite.ATan2Options = class ATan2Options {

    static decode(/* reader, position */) {
        const $ = new tflite.ATan2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.ATan2Options();
        return $;
    }
};

tflite.UnsortedSegmentMinOptions = class UnsortedSegmentMinOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.UnsortedSegmentMinOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.UnsortedSegmentMinOptions();
        return $;
    }
};

tflite.SignOptions = class SignOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.SignOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.SignOptions();
        return $;
    }
};

tflite.BitcastOptions = class BitcastOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.BitcastOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.BitcastOptions();
        return $;
    }
};

tflite.BitwiseXorOptions = class BitwiseXorOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.BitwiseXorOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.BitwiseXorOptions();
        return $;
    }
};

tflite.RightShiftOptions = class RightShiftOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.RightShiftOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.RightShiftOptions();
        return $;
    }
};

tflite.DilateOptions = class DilateOptions {

    static decode(/* reader, position */) {
        const $ = new tflite.DilateOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.DilateOptions();
        return $;
    }
};

tflite.ReduceWindowFunction = {
    UNSUPPORTED: 0,
    ADD: 1,
    MUL: 2,
    MINIMUM: 3,
    MAXIMUM: 4,
    ALL: 5,
    ANY: 6
};

tflite.ReduceWindowOptions = class ReduceWindowOptions {

    static decode(reader, position) {
        const $ = new tflite.ReduceWindowOptions();
        $.reduce_function = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ReduceWindowOptions();
        $.reduce_function = tflite.ReduceWindowFunction[json.reduce_function];
        return $;
    }
};

tflite.OperatorCode = class OperatorCode {

    static decode(reader, position) {
        const $ = new tflite.OperatorCode();
        $.deprecated_builtin_code = reader.int8_(position, 4, 0);
        $.custom_code = reader.string_(position, 6, null);
        $.version = reader.int32_(position, 8, 1);
        $.builtin_code = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.OperatorCode();
        $.deprecated_builtin_code = reader.value(json.deprecated_builtin_code, 0);
        $.custom_code = reader.value(json.custom_code, null);
        $.version = reader.value(json.version, 1);
        $.builtin_code = tflite.BuiltinOperator[json.builtin_code];
        return $;
    }
};

tflite.CustomOptionsFormat = {
    FLEXBUFFERS: 0
};

tflite.StableHLOCompositeOptions = class StableHLOCompositeOptions {

    static decode(reader, position) {
        const $ = new tflite.StableHLOCompositeOptions();
        $.name = reader.string_(position, 4, null);
        $.decomposition_subgraph_index = reader.int32_(position, 6, 0);
        $.composite_attributes = reader.array(position, 8, Uint8Array);
        $.composite_attributes_format = reader.int8_(position, 10, 0);
        $.version = reader.int32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.StableHLOCompositeOptions();
        $.name = reader.value(json.name, null);
        $.decomposition_subgraph_index = reader.value(json.decomposition_subgraph_index, 0);
        $.composite_attributes = reader.array(json.composite_attributes, Uint8Array);
        $.composite_attributes_format = tflite.CustomOptionsFormat[json.composite_attributes_format];
        $.version = reader.value(json.version, 0);
        return $;
    }
};

tflite.Operator = class Operator {

    static decode(reader, position) {
        const $ = new tflite.Operator();
        $.opcode_index = reader.uint32_(position, 4, 0);
        $.inputs = reader.array(position, 6, Int32Array);
        $.outputs = reader.array(position, 8, Int32Array);
        $.builtin_options = reader.union(position, 10, tflite.BuiltinOptions);
        $.custom_options = reader.array(position, 14, Uint8Array);
        $.custom_options_format = reader.int8_(position, 16, 0);
        $.mutating_variable_inputs = reader.bools_(position, 18);
        $.intermediates = reader.array(position, 20, Int32Array);
        $.large_custom_options_offset = reader.uint64_(position, 22, 0n);
        $.large_custom_options_size = reader.uint64_(position, 24, 0n);
        $.builtin_options_2 = reader.union(position, 26, tflite.BuiltinOptions2);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Operator();
        $.opcode_index = reader.value(json.opcode_index, 0);
        $.inputs = reader.array(json.inputs, Int32Array);
        $.outputs = reader.array(json.outputs, Int32Array);
        $.builtin_options = tflite.BuiltinOptions.decodeText(reader, json.builtin_options, json.builtin_options_type);
        $.custom_options = reader.array(json.custom_options, Uint8Array);
        $.custom_options_format = tflite.CustomOptionsFormat[json.custom_options_format];
        $.mutating_variable_inputs = reader.array(json.mutating_variable_inputs);
        $.intermediates = reader.array(json.intermediates, Int32Array);
        $.large_custom_options_offset = reader.uint64(json.large_custom_options_offset, 0n);
        $.large_custom_options_size = reader.uint64(json.large_custom_options_size, 0n);
        $.builtin_options_2 = tflite.BuiltinOptions2.decodeText(reader, json.builtin_options_2, json.builtin_options_2_type);
        return $;
    }
};

tflite.SubGraph = class SubGraph {

    static decode(reader, position) {
        const $ = new tflite.SubGraph();
        $.tensors = reader.tables(position, 4, tflite.Tensor);
        $.inputs = reader.array(position, 6, Int32Array);
        $.outputs = reader.array(position, 8, Int32Array);
        $.operators = reader.tables(position, 10, tflite.Operator);
        $.name = reader.string_(position, 12, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SubGraph();
        $.tensors = reader.objects(json.tensors, tflite.Tensor);
        $.inputs = reader.array(json.inputs, Int32Array);
        $.outputs = reader.array(json.outputs, Int32Array);
        $.operators = reader.objects(json.operators, tflite.Operator);
        $.name = reader.value(json.name, null);
        return $;
    }
};

tflite.Buffer = class Buffer {

    static decode(reader, position) {
        const $ = new tflite.Buffer();
        $.data = reader.array(position, 4, Uint8Array);
        $.offset = reader.uint64_(position, 6, 0n);
        $.size = reader.uint64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Buffer();
        $.data = reader.array(json.data, Uint8Array);
        $.offset = reader.uint64(json.offset, 0n);
        $.size = reader.uint64(json.size, 0n);
        return $;
    }
};

tflite.Metadata = class Metadata {

    static decode(reader, position) {
        const $ = new tflite.Metadata();
        $.name = reader.string_(position, 4, null);
        $.buffer = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Metadata();
        $.name = reader.value(json.name, null);
        $.buffer = reader.value(json.buffer, 0);
        return $;
    }
};

tflite.TensorMap = class TensorMap {

    static decode(reader, position) {
        const $ = new tflite.TensorMap();
        $.name = reader.string_(position, 4, null);
        $.tensor_index = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.TensorMap();
        $.name = reader.value(json.name, null);
        $.tensor_index = reader.value(json.tensor_index, 0);
        return $;
    }
};

tflite.SignatureDef = class SignatureDef {

    static decode(reader, position) {
        const $ = new tflite.SignatureDef();
        $.inputs = reader.tables(position, 4, tflite.TensorMap);
        $.outputs = reader.tables(position, 6, tflite.TensorMap);
        $.signature_key = reader.string_(position, 8, null);
        $.deprecated_tag = reader.string_(position, 10, null);
        $.subgraph_index = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SignatureDef();
        $.inputs = reader.objects(json.inputs, tflite.TensorMap);
        $.outputs = reader.objects(json.outputs, tflite.TensorMap);
        $.signature_key = reader.value(json.signature_key, null);
        $.deprecated_tag = reader.value(json.deprecated_tag, null);
        $.subgraph_index = reader.value(json.subgraph_index, 0);
        return $;
    }
};

tflite.Model = class Model {

    static identifier(reader) {
        return reader.identifier === 'TFL3';
    }

    static create(reader) {
        return tflite.Model.decode(reader, reader.root);
    }

    static createText(reader) {
        return tflite.Model.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new tflite.Model();
        $.version = reader.uint32_(position, 4, 0);
        $.operator_codes = reader.tables(position, 6, tflite.OperatorCode);
        $.subgraphs = reader.tables(position, 8, tflite.SubGraph);
        $.description = reader.string_(position, 10, null);
        $.buffers = reader.tables(position, 12, tflite.Buffer);
        $.metadata_buffer = reader.array(position, 14, Int32Array);
        $.metadata = reader.tables(position, 16, tflite.Metadata);
        $.signature_defs = reader.tables(position, 18, tflite.SignatureDef);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Model();
        $.version = reader.value(json.version, 0);
        $.operator_codes = reader.objects(json.operator_codes, tflite.OperatorCode);
        $.subgraphs = reader.objects(json.subgraphs, tflite.SubGraph);
        $.description = reader.value(json.description, null);
        $.buffers = reader.objects(json.buffers, tflite.Buffer);
        $.metadata_buffer = reader.array(json.metadata_buffer, Int32Array);
        $.metadata = reader.objects(json.metadata, tflite.Metadata);
        $.signature_defs = reader.objects(json.signature_defs, tflite.SignatureDef);
        return $;
    }
};

tflite.AssociatedFileType = {
    UNKNOWN: 0,
    DESCRIPTIONS: 1,
    TENSOR_AXIS_LABELS: 2,
    TENSOR_VALUE_LABELS: 3,
    TENSOR_AXIS_SCORE_CALIBRATION: 4,
    VOCABULARY: 5,
    SCANN_INDEX_FILE: 6
};

tflite.AssociatedFile = class AssociatedFile {

    static decode(reader, position) {
        const $ = new tflite.AssociatedFile();
        $.name = reader.string_(position, 4, null);
        $.description = reader.string_(position, 6, null);
        $.type = reader.int8_(position, 8, 0);
        $.locale = reader.string_(position, 10, null);
        $.version = reader.string_(position, 12, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.AssociatedFile();
        $.name = reader.value(json.name, null);
        $.description = reader.value(json.description, null);
        $.type = tflite.AssociatedFileType[json.type];
        $.locale = reader.value(json.locale, null);
        $.version = reader.value(json.version, null);
        return $;
    }
};

tflite.FeatureProperties = class FeatureProperties {

    static decode(/* reader, position */) {
        const $ = new tflite.FeatureProperties();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tflite.FeatureProperties();
        return $;
    }
};

tflite.ColorSpaceType = {
    UNKNOWN: 0,
    RGB: 1,
    GRAYSCALE: 2
};

tflite.ImageSize = class ImageSize {

    static decode(reader, position) {
        const $ = new tflite.ImageSize();
        $.width = reader.uint32_(position, 4, 0);
        $.height = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ImageSize();
        $.width = reader.value(json.width, 0);
        $.height = reader.value(json.height, 0);
        return $;
    }
};

tflite.ImageProperties = class ImageProperties {

    static decode(reader, position) {
        const $ = new tflite.ImageProperties();
        $.color_space = reader.int8_(position, 4, 0);
        $.default_size = reader.table(position, 6, tflite.ImageSize);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ImageProperties();
        $.color_space = tflite.ColorSpaceType[json.color_space];
        $.default_size = reader.object(json.default_size, tflite.ImageSize);
        return $;
    }
};

tflite.BoundingBoxType = {
    UNKNOWN: 0,
    BOUNDARIES: 1,
    UPPER_LEFT: 2,
    CENTER: 3
};

tflite.AudioProperties = class AudioProperties {

    static decode(reader, position) {
        const $ = new tflite.AudioProperties();
        $.sample_rate = reader.uint32_(position, 4, 0);
        $.channels = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.AudioProperties();
        $.sample_rate = reader.value(json.sample_rate, 0);
        $.channels = reader.value(json.channels, 0);
        return $;
    }
};

tflite.CoordinateType = {
    RATIO: 0,
    PIXEL: 1
};

tflite.BoundingBoxProperties = class BoundingBoxProperties {

    static decode(reader, position) {
        const $ = new tflite.BoundingBoxProperties();
        $.index = reader.array(position, 4, Uint32Array);
        $.type = reader.int8_(position, 6, 0);
        $.coordinate_type = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.BoundingBoxProperties();
        $.index = reader.array(json.index, Uint32Array);
        $.type = tflite.BoundingBoxType[json.type];
        $.coordinate_type = tflite.CoordinateType[json.coordinate_type];
        return $;
    }
};

tflite.ContentProperties = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tflite.FeatureProperties.decode(reader, position);
            case 2: return tflite.ImageProperties.decode(reader, position);
            case 3: return tflite.BoundingBoxProperties.decode(reader, position);
            case 4: return tflite.AudioProperties.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'FeatureProperties': return tflite.FeatureProperties.decodeText(reader, json);
            case 'ImageProperties': return tflite.ImageProperties.decodeText(reader, json);
            case 'BoundingBoxProperties': return tflite.BoundingBoxProperties.decodeText(reader, json);
            case 'AudioProperties': return tflite.AudioProperties.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tflite.ValueRange = class ValueRange {

    static decode(reader, position) {
        const $ = new tflite.ValueRange();
        $.min = reader.int32_(position, 4, 0);
        $.max = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ValueRange();
        $.min = reader.value(json.min, 0);
        $.max = reader.value(json.max, 0);
        return $;
    }
};

tflite.Content = class Content {

    static decode(reader, position) {
        const $ = new tflite.Content();
        $.content_properties = reader.union(position, 4, tflite.ContentProperties);
        $.range = reader.table(position, 8, tflite.ValueRange);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Content();
        $.content_properties = tflite.ContentProperties.decodeText(reader, json.content_properties, json.content_properties_type);
        $.range = reader.object(json.range, tflite.ValueRange);
        return $;
    }
};

tflite.NormalizationOptions = class NormalizationOptions {

    static decode(reader, position) {
        const $ = new tflite.NormalizationOptions();
        $.mean = reader.array(position, 4, Float32Array);
        $.std = reader.array(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.NormalizationOptions();
        $.mean = reader.array(json.mean, Float32Array);
        $.std = reader.array(json.std, Float32Array);
        return $;
    }
};

tflite.ScoreTransformationType = {
    IDENTITY: 0,
    LOG: 1,
    INVERSE_LOGISTIC: 2
};

tflite.ScoreCalibrationOptions = class ScoreCalibrationOptions {

    static decode(reader, position) {
        const $ = new tflite.ScoreCalibrationOptions();
        $.score_transformation = reader.int8_(position, 4, 0);
        $.default_score = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ScoreCalibrationOptions();
        $.score_transformation = tflite.ScoreTransformationType[json.score_transformation];
        $.default_score = reader.value(json.default_score, 0);
        return $;
    }
};

tflite.ScoreThresholdingOptions = class ScoreThresholdingOptions {

    static decode(reader, position) {
        const $ = new tflite.ScoreThresholdingOptions();
        $.global_score_threshold = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ScoreThresholdingOptions();
        $.global_score_threshold = reader.value(json.global_score_threshold, 0);
        return $;
    }
};

tflite.BertTokenizerOptions = class BertTokenizerOptions {

    static decode(reader, position) {
        const $ = new tflite.BertTokenizerOptions();
        $.vocab_file = reader.tables(position, 4, tflite.AssociatedFile);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.BertTokenizerOptions();
        $.vocab_file = reader.objects(json.vocab_file, tflite.AssociatedFile);
        return $;
    }
};

tflite.SentencePieceTokenizerOptions = class SentencePieceTokenizerOptions {

    static decode(reader, position) {
        const $ = new tflite.SentencePieceTokenizerOptions();
        $.sentencePiece_model = reader.tables(position, 4, tflite.AssociatedFile);
        $.vocab_file = reader.tables(position, 6, tflite.AssociatedFile);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SentencePieceTokenizerOptions();
        $.sentencePiece_model = reader.objects(json.sentencePiece_model, tflite.AssociatedFile);
        $.vocab_file = reader.objects(json.vocab_file, tflite.AssociatedFile);
        return $;
    }
};

tflite.RegexTokenizerOptions = class RegexTokenizerOptions {

    static decode(reader, position) {
        const $ = new tflite.RegexTokenizerOptions();
        $.delim_regex_pattern = reader.string_(position, 4, null);
        $.vocab_file = reader.tables(position, 6, tflite.AssociatedFile);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.RegexTokenizerOptions();
        $.delim_regex_pattern = reader.value(json.delim_regex_pattern, null);
        $.vocab_file = reader.objects(json.vocab_file, tflite.AssociatedFile);
        return $;
    }
};

tflite.ProcessUnitOptions = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tflite.NormalizationOptions.decode(reader, position);
            case 2: return tflite.ScoreCalibrationOptions.decode(reader, position);
            case 3: return tflite.ScoreThresholdingOptions.decode(reader, position);
            case 4: return tflite.BertTokenizerOptions.decode(reader, position);
            case 5: return tflite.SentencePieceTokenizerOptions.decode(reader, position);
            case 6: return tflite.RegexTokenizerOptions.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'NormalizationOptions': return tflite.NormalizationOptions.decodeText(reader, json);
            case 'ScoreCalibrationOptions': return tflite.ScoreCalibrationOptions.decodeText(reader, json);
            case 'ScoreThresholdingOptions': return tflite.ScoreThresholdingOptions.decodeText(reader, json);
            case 'BertTokenizerOptions': return tflite.BertTokenizerOptions.decodeText(reader, json);
            case 'SentencePieceTokenizerOptions': return tflite.SentencePieceTokenizerOptions.decodeText(reader, json);
            case 'RegexTokenizerOptions': return tflite.RegexTokenizerOptions.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tflite.ProcessUnit = class ProcessUnit {

    static decode(reader, position) {
        const $ = new tflite.ProcessUnit();
        $.options = reader.union(position, 4, tflite.ProcessUnitOptions);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ProcessUnit();
        $.options = tflite.ProcessUnitOptions.decodeText(reader, json.options, json.options_type);
        return $;
    }
};

tflite.Stats = class Stats {

    static decode(reader, position) {
        const $ = new tflite.Stats();
        $.max = reader.array(position, 4, Float32Array);
        $.min = reader.array(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.Stats();
        $.max = reader.array(json.max, Float32Array);
        $.min = reader.array(json.min, Float32Array);
        return $;
    }
};

tflite.TensorGroup = class TensorGroup {

    static decode(reader, position) {
        const $ = new tflite.TensorGroup();
        $.name = reader.string_(position, 4, null);
        $.tensor_names = reader.strings_(position, 6);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.TensorGroup();
        $.name = reader.value(json.name, null);
        $.tensor_names = reader.array(json.tensor_names);
        return $;
    }
};

tflite.TensorMetadata = class TensorMetadata {

    static decode(reader, position) {
        const $ = new tflite.TensorMetadata();
        $.name = reader.string_(position, 4, null);
        $.description = reader.string_(position, 6, null);
        $.dimension_names = reader.strings_(position, 8);
        $.content = reader.table(position, 10, tflite.Content);
        $.process_units = reader.tables(position, 12, tflite.ProcessUnit);
        $.stats = reader.table(position, 14, tflite.Stats);
        $.associated_files = reader.tables(position, 16, tflite.AssociatedFile);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.TensorMetadata();
        $.name = reader.value(json.name, null);
        $.description = reader.value(json.description, null);
        $.dimension_names = reader.array(json.dimension_names);
        $.content = reader.object(json.content, tflite.Content);
        $.process_units = reader.objects(json.process_units, tflite.ProcessUnit);
        $.stats = reader.object(json.stats, tflite.Stats);
        $.associated_files = reader.objects(json.associated_files, tflite.AssociatedFile);
        return $;
    }
};

tflite.CustomMetadata = class CustomMetadata {

    static decode(reader, position) {
        const $ = new tflite.CustomMetadata();
        $.name = reader.string_(position, 4, null);
        $.data = reader.array(position, 6, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.CustomMetadata();
        $.name = reader.value(json.name, null);
        $.data = reader.array(json.data, Uint8Array);
        return $;
    }
};

tflite.SubGraphMetadata = class SubGraphMetadata {

    static decode(reader, position) {
        const $ = new tflite.SubGraphMetadata();
        $.name = reader.string_(position, 4, null);
        $.description = reader.string_(position, 6, null);
        $.input_tensor_metadata = reader.tables(position, 8, tflite.TensorMetadata);
        $.output_tensor_metadata = reader.tables(position, 10, tflite.TensorMetadata);
        $.associated_files = reader.tables(position, 12, tflite.AssociatedFile);
        $.input_process_units = reader.tables(position, 14, tflite.ProcessUnit);
        $.output_process_units = reader.tables(position, 16, tflite.ProcessUnit);
        $.input_tensor_groups = reader.tables(position, 18, tflite.TensorGroup);
        $.output_tensor_groups = reader.tables(position, 20, tflite.TensorGroup);
        $.custom_metadata = reader.tables(position, 22, tflite.CustomMetadata);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.SubGraphMetadata();
        $.name = reader.value(json.name, null);
        $.description = reader.value(json.description, null);
        $.input_tensor_metadata = reader.objects(json.input_tensor_metadata, tflite.TensorMetadata);
        $.output_tensor_metadata = reader.objects(json.output_tensor_metadata, tflite.TensorMetadata);
        $.associated_files = reader.objects(json.associated_files, tflite.AssociatedFile);
        $.input_process_units = reader.objects(json.input_process_units, tflite.ProcessUnit);
        $.output_process_units = reader.objects(json.output_process_units, tflite.ProcessUnit);
        $.input_tensor_groups = reader.objects(json.input_tensor_groups, tflite.TensorGroup);
        $.output_tensor_groups = reader.objects(json.output_tensor_groups, tflite.TensorGroup);
        $.custom_metadata = reader.objects(json.custom_metadata, tflite.CustomMetadata);
        return $;
    }
};

tflite.ModelMetadata = class ModelMetadata {

    static identifier(reader) {
        return reader.identifier === 'M001';
    }

    static create(reader) {
        return tflite.ModelMetadata.decode(reader, reader.root);
    }

    static createText(reader) {
        return tflite.ModelMetadata.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new tflite.ModelMetadata();
        $.name = reader.string_(position, 4, null);
        $.description = reader.string_(position, 6, null);
        $.version = reader.string_(position, 8, null);
        $.subgraph_metadata = reader.tables(position, 10, tflite.SubGraphMetadata);
        $.author = reader.string_(position, 12, null);
        $.license = reader.string_(position, 14, null);
        $.associated_files = reader.tables(position, 16, tflite.AssociatedFile);
        $.min_parser_version = reader.string_(position, 18, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tflite.ModelMetadata();
        $.name = reader.value(json.name, null);
        $.description = reader.value(json.description, null);
        $.version = reader.value(json.version, null);
        $.subgraph_metadata = reader.objects(json.subgraph_metadata, tflite.SubGraphMetadata);
        $.author = reader.value(json.author, null);
        $.license = reader.value(json.license, null);
        $.associated_files = reader.objects(json.associated_files, tflite.AssociatedFile);
        $.min_parser_version = reader.value(json.min_parser_version, null);
        return $;
    }
};
