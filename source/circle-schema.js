
export const circle = {};

circle.TensorType = {
    UINT4: -1,
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
    INT4: 17
};

circle.CustomQuantization = class CustomQuantization {

    static decode(reader, position) {
        const $ = new circle.CustomQuantization();
        $.custom = reader.array(position, 4, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.CustomQuantization();
        $.custom = reader.array(json.custom, Uint8Array);
        return $;
    }
};

circle.QuantizationDetails = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return circle.CustomQuantization.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'CustomQuantization': return circle.CustomQuantization.decodeText(reader, json);
            default: return undefined;
        }
    }
};

circle.QuantizationParameters = class QuantizationParameters {

    static decode(reader, position) {
        const $ = new circle.QuantizationParameters();
        $.min = reader.array(position, 4, Float32Array);
        $.max = reader.array(position, 6, Float32Array);
        $.scale = reader.array(position, 8, Float32Array);
        $.zero_point = reader.int64s_(position, 10);
        $.details = reader.union(position, 12, circle.QuantizationDetails);
        $.quantized_dimension = reader.int32_(position, 16, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.QuantizationParameters();
        $.min = reader.array(json.min, Float32Array);
        $.max = reader.array(json.max, Float32Array);
        $.scale = reader.array(json.scale, Float32Array);
        $.zero_point = reader.array(json.zero_point);
        $.details = circle.QuantizationDetails.decodeText(reader, json.details, json.details_type);
        $.quantized_dimension = reader.value(json.quantized_dimension, 0);
        return $;
    }
};

circle.DimensionType = {
    DENSE: 0,
    SPARSE_CSR: 1
};

circle.Int32Vector = class Int32Vector {

    static decode(reader, position) {
        const $ = new circle.Int32Vector();
        $.values = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Int32Vector();
        $.values = reader.array(json.values, Int32Array);
        return $;
    }
};

circle.Uint16Vector = class Uint16Vector {

    static decode(reader, position) {
        const $ = new circle.Uint16Vector();
        $.values = reader.array(position, 4, Uint16Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Uint16Vector();
        $.values = reader.array(json.values, Uint16Array);
        return $;
    }
};

circle.Uint8Vector = class Uint8Vector {

    static decode(reader, position) {
        const $ = new circle.Uint8Vector();
        $.values = reader.array(position, 4, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Uint8Vector();
        $.values = reader.array(json.values, Uint8Array);
        return $;
    }
};

circle.SparseIndexVector = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return circle.Int32Vector.decode(reader, position);
            case 2: return circle.Uint16Vector.decode(reader, position);
            case 3: return circle.Uint8Vector.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Int32Vector': return circle.Int32Vector.decodeText(reader, json);
            case 'Uint16Vector': return circle.Uint16Vector.decodeText(reader, json);
            case 'Uint8Vector': return circle.Uint8Vector.decodeText(reader, json);
            default: return undefined;
        }
    }
};

circle.DimensionMetadata = class DimensionMetadata {

    static decode(reader, position) {
        const $ = new circle.DimensionMetadata();
        $.format = reader.int8_(position, 4, 0);
        $.dense_size = reader.int32_(position, 6, 0);
        $.array_segments = reader.union(position, 8, circle.SparseIndexVector);
        $.array_indices = reader.union(position, 12, circle.SparseIndexVector);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.DimensionMetadata();
        $.format = circle.DimensionType[json.format];
        $.dense_size = reader.value(json.dense_size, 0);
        $.array_segments = circle.SparseIndexVector.decodeText(reader, json.array_segments, json.array_segments_type);
        $.array_indices = circle.SparseIndexVector.decodeText(reader, json.array_indices, json.array_indices_type);
        return $;
    }
};

circle.SparsityParameters = class SparsityParameters {

    static decode(reader, position) {
        const $ = new circle.SparsityParameters();
        $.traversal_order = reader.array(position, 4, Int32Array);
        $.block_map = reader.array(position, 6, Int32Array);
        $.dim_metadata = reader.tables(position, 8, circle.DimensionMetadata);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SparsityParameters();
        $.traversal_order = reader.array(json.traversal_order, Int32Array);
        $.block_map = reader.array(json.block_map, Int32Array);
        $.dim_metadata = reader.objects(json.dim_metadata, circle.DimensionMetadata);
        return $;
    }
};

circle.VariantSubType = class VariantSubType {

    static decode(reader, position) {
        const $ = new circle.VariantSubType();
        $.shape = reader.array(position, 4, Int32Array);
        $.type = reader.int8_(position, 6, 0);
        $.has_rank = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.VariantSubType();
        $.shape = reader.array(json.shape, Int32Array);
        $.type = circle.TensorType[json.type];
        $.has_rank = reader.value(json.has_rank, false);
        return $;
    }
};

circle.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new circle.Tensor();
        $.shape = reader.array(position, 4, Int32Array);
        $.type = reader.int8_(position, 6, 0);
        $.buffer = reader.uint32_(position, 8, 0);
        $.name = reader.string_(position, 10, null);
        $.quantization = reader.table(position, 12, circle.QuantizationParameters);
        $.is_variable = reader.bool_(position, 14, false);
        $.sparsity = reader.table(position, 16, circle.SparsityParameters);
        $.shape_signature = reader.array(position, 18, Int32Array);
        $.has_rank = reader.bool_(position, 20, false);
        $.variant_tensors = reader.tables(position, 22, circle.VariantSubType);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Tensor();
        $.shape = reader.array(json.shape, Int32Array);
        $.type = circle.TensorType[json.type];
        $.buffer = reader.value(json.buffer, 0);
        $.name = reader.value(json.name, null);
        $.quantization = reader.object(json.quantization, circle.QuantizationParameters);
        $.is_variable = reader.value(json.is_variable, false);
        $.sparsity = reader.object(json.sparsity, circle.SparsityParameters);
        $.shape_signature = reader.array(json.shape_signature, Int32Array);
        $.has_rank = reader.value(json.has_rank, false);
        $.variant_tensors = reader.objects(json.variant_tensors, circle.VariantSubType);
        return $;
    }
};

circle.BuiltinOperator = {
    GRU: -5,
    BCQ_GATHER: -4,
    BCQ_FULLY_CONNECTED: -3,
    INSTANCE_NORM: -2,
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
    REDUCE_WINDOW: 205
};

circle.BuiltinOptions = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return circle.Conv2DOptions.decode(reader, position);
            case 2: return circle.DepthwiseConv2DOptions.decode(reader, position);
            case 3: return circle.ConcatEmbeddingsOptions.decode(reader, position);
            case 4: return circle.LSHProjectionOptions.decode(reader, position);
            case 5: return circle.Pool2DOptions.decode(reader, position);
            case 6: return circle.SVDFOptions.decode(reader, position);
            case 7: return circle.RNNOptions.decode(reader, position);
            case 8: return circle.FullyConnectedOptions.decode(reader, position);
            case 9: return circle.SoftmaxOptions.decode(reader, position);
            case 10: return circle.ConcatenationOptions.decode(reader, position);
            case 11: return circle.AddOptions.decode(reader, position);
            case 12: return circle.L2NormOptions.decode(reader, position);
            case 13: return circle.LocalResponseNormalizationOptions.decode(reader, position);
            case 14: return circle.LSTMOptions.decode(reader, position);
            case 15: return circle.ResizeBilinearOptions.decode(reader, position);
            case 16: return circle.CallOptions.decode(reader, position);
            case 17: return circle.ReshapeOptions.decode(reader, position);
            case 18: return circle.SkipGramOptions.decode(reader, position);
            case 19: return circle.SpaceToDepthOptions.decode(reader, position);
            case 20: return circle.EmbeddingLookupSparseOptions.decode(reader, position);
            case 21: return circle.MulOptions.decode(reader, position);
            case 22: return circle.PadOptions.decode(reader, position);
            case 23: return circle.GatherOptions.decode(reader, position);
            case 24: return circle.BatchToSpaceNDOptions.decode(reader, position);
            case 25: return circle.SpaceToBatchNDOptions.decode(reader, position);
            case 26: return circle.TransposeOptions.decode(reader, position);
            case 27: return circle.ReducerOptions.decode(reader, position);
            case 28: return circle.SubOptions.decode(reader, position);
            case 29: return circle.DivOptions.decode(reader, position);
            case 30: return circle.SqueezeOptions.decode(reader, position);
            case 31: return circle.SequenceRNNOptions.decode(reader, position);
            case 32: return circle.StridedSliceOptions.decode(reader, position);
            case 33: return circle.ExpOptions.decode(reader, position);
            case 34: return circle.TopKV2Options.decode(reader, position);
            case 35: return circle.SplitOptions.decode(reader, position);
            case 36: return circle.LogSoftmaxOptions.decode(reader, position);
            case 37: return circle.CastOptions.decode(reader, position);
            case 38: return circle.DequantizeOptions.decode(reader, position);
            case 39: return circle.MaximumMinimumOptions.decode(reader, position);
            case 40: return circle.ArgMaxOptions.decode(reader, position);
            case 41: return circle.LessOptions.decode(reader, position);
            case 42: return circle.NegOptions.decode(reader, position);
            case 43: return circle.PadV2Options.decode(reader, position);
            case 44: return circle.GreaterOptions.decode(reader, position);
            case 45: return circle.GreaterEqualOptions.decode(reader, position);
            case 46: return circle.LessEqualOptions.decode(reader, position);
            case 47: return circle.SelectOptions.decode(reader, position);
            case 48: return circle.SliceOptions.decode(reader, position);
            case 49: return circle.TransposeConvOptions.decode(reader, position);
            case 50: return circle.SparseToDenseOptions.decode(reader, position);
            case 51: return circle.TileOptions.decode(reader, position);
            case 52: return circle.ExpandDimsOptions.decode(reader, position);
            case 53: return circle.EqualOptions.decode(reader, position);
            case 54: return circle.NotEqualOptions.decode(reader, position);
            case 55: return circle.ShapeOptions.decode(reader, position);
            case 56: return circle.PowOptions.decode(reader, position);
            case 57: return circle.ArgMinOptions.decode(reader, position);
            case 58: return circle.FakeQuantOptions.decode(reader, position);
            case 59: return circle.PackOptions.decode(reader, position);
            case 60: return circle.LogicalOrOptions.decode(reader, position);
            case 61: return circle.OneHotOptions.decode(reader, position);
            case 62: return circle.LogicalAndOptions.decode(reader, position);
            case 63: return circle.LogicalNotOptions.decode(reader, position);
            case 64: return circle.UnpackOptions.decode(reader, position);
            case 65: return circle.FloorDivOptions.decode(reader, position);
            case 66: return circle.SquareOptions.decode(reader, position);
            case 67: return circle.ZerosLikeOptions.decode(reader, position);
            case 68: return circle.FillOptions.decode(reader, position);
            case 69: return circle.BidirectionalSequenceLSTMOptions.decode(reader, position);
            case 70: return circle.BidirectionalSequenceRNNOptions.decode(reader, position);
            case 71: return circle.UnidirectionalSequenceLSTMOptions.decode(reader, position);
            case 72: return circle.FloorModOptions.decode(reader, position);
            case 73: return circle.RangeOptions.decode(reader, position);
            case 74: return circle.ResizeNearestNeighborOptions.decode(reader, position);
            case 75: return circle.LeakyReluOptions.decode(reader, position);
            case 76: return circle.SquaredDifferenceOptions.decode(reader, position);
            case 77: return circle.MirrorPadOptions.decode(reader, position);
            case 78: return circle.AbsOptions.decode(reader, position);
            case 79: return circle.SplitVOptions.decode(reader, position);
            case 80: return circle.UniqueOptions.decode(reader, position);
            case 81: return circle.ReverseV2Options.decode(reader, position);
            case 82: return circle.AddNOptions.decode(reader, position);
            case 83: return circle.GatherNdOptions.decode(reader, position);
            case 84: return circle.CosOptions.decode(reader, position);
            case 85: return circle.WhereOptions.decode(reader, position);
            case 86: return circle.RankOptions.decode(reader, position);
            case 87: return circle.ReverseSequenceOptions.decode(reader, position);
            case 88: return circle.MatrixDiagOptions.decode(reader, position);
            case 89: return circle.QuantizeOptions.decode(reader, position);
            case 90: return circle.MatrixSetDiagOptions.decode(reader, position);
            case 91: return circle.HardSwishOptions.decode(reader, position);
            case 92: return circle.IfOptions.decode(reader, position);
            case 93: return circle.WhileOptions.decode(reader, position);
            case 94: return circle.DepthToSpaceOptions.decode(reader, position);
            case 95: return circle.NonMaxSuppressionV4Options.decode(reader, position);
            case 96: return circle.NonMaxSuppressionV5Options.decode(reader, position);
            case 97: return circle.ScatterNdOptions.decode(reader, position);
            case 98: return circle.SelectV2Options.decode(reader, position);
            case 99: return circle.DensifyOptions.decode(reader, position);
            case 100: return circle.SegmentSumOptions.decode(reader, position);
            case 101: return circle.BatchMatMulOptions.decode(reader, position);
            case 102: return circle.CumsumOptions.decode(reader, position);
            case 103: return circle.CallOnceOptions.decode(reader, position);
            case 104: return circle.BroadcastToOptions.decode(reader, position);
            case 105: return circle.Rfft2dOptions.decode(reader, position);
            case 106: return circle.Conv3DOptions.decode(reader, position);
            case 107: return circle.HashtableOptions.decode(reader, position);
            case 108: return circle.HashtableFindOptions.decode(reader, position);
            case 109: return circle.HashtableImportOptions.decode(reader, position);
            case 110: return circle.HashtableSizeOptions.decode(reader, position);
            case 111: return circle.VarHandleOptions.decode(reader, position);
            case 112: return circle.ReadVariableOptions.decode(reader, position);
            case 113: return circle.AssignVariableOptions.decode(reader, position);
            case 114: return circle.RandomOptions.decode(reader, position);
            case 115: return circle.BucketizeOptions.decode(reader, position);
            case 116: return circle.GeluOptions.decode(reader, position);
            case 117: return circle.DynamicUpdateSliceOptions.decode(reader, position);
            case 118: return circle.UnsortedSegmentProdOptions.decode(reader, position);
            case 119: return circle.UnsortedSegmentMaxOptions.decode(reader, position);
            case 120: return circle.UnsortedSegmentMinOptions.decode(reader, position);
            case 121: return circle.UnsortedSegmentSumOptions.decode(reader, position);
            case 122: return circle.ATan2Options.decode(reader, position);
            case 123: return circle.SignOptions.decode(reader, position);
            case 124: return circle.BitcastOptions.decode(reader, position);
            case 125: return circle.BitwiseXorOptions.decode(reader, position);
            case 126: return circle.RightShiftOptions.decode(reader, position);
            case 251: return circle.GRUOptions.decode(reader, position);
            case 252: return circle.BCQGatherOptions.decode(reader, position);
            case 253: return circle.BCQFullyConnectedOptions.decode(reader, position);
            case 254: return circle.InstanceNormOptions.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Conv2DOptions': return circle.Conv2DOptions.decodeText(reader, json);
            case 'DepthwiseConv2DOptions': return circle.DepthwiseConv2DOptions.decodeText(reader, json);
            case 'ConcatEmbeddingsOptions': return circle.ConcatEmbeddingsOptions.decodeText(reader, json);
            case 'LSHProjectionOptions': return circle.LSHProjectionOptions.decodeText(reader, json);
            case 'Pool2DOptions': return circle.Pool2DOptions.decodeText(reader, json);
            case 'SVDFOptions': return circle.SVDFOptions.decodeText(reader, json);
            case 'RNNOptions': return circle.RNNOptions.decodeText(reader, json);
            case 'FullyConnectedOptions': return circle.FullyConnectedOptions.decodeText(reader, json);
            case 'SoftmaxOptions': return circle.SoftmaxOptions.decodeText(reader, json);
            case 'ConcatenationOptions': return circle.ConcatenationOptions.decodeText(reader, json);
            case 'AddOptions': return circle.AddOptions.decodeText(reader, json);
            case 'L2NormOptions': return circle.L2NormOptions.decodeText(reader, json);
            case 'LocalResponseNormalizationOptions': return circle.LocalResponseNormalizationOptions.decodeText(reader, json);
            case 'LSTMOptions': return circle.LSTMOptions.decodeText(reader, json);
            case 'ResizeBilinearOptions': return circle.ResizeBilinearOptions.decodeText(reader, json);
            case 'CallOptions': return circle.CallOptions.decodeText(reader, json);
            case 'ReshapeOptions': return circle.ReshapeOptions.decodeText(reader, json);
            case 'SkipGramOptions': return circle.SkipGramOptions.decodeText(reader, json);
            case 'SpaceToDepthOptions': return circle.SpaceToDepthOptions.decodeText(reader, json);
            case 'EmbeddingLookupSparseOptions': return circle.EmbeddingLookupSparseOptions.decodeText(reader, json);
            case 'MulOptions': return circle.MulOptions.decodeText(reader, json);
            case 'PadOptions': return circle.PadOptions.decodeText(reader, json);
            case 'GatherOptions': return circle.GatherOptions.decodeText(reader, json);
            case 'BatchToSpaceNDOptions': return circle.BatchToSpaceNDOptions.decodeText(reader, json);
            case 'SpaceToBatchNDOptions': return circle.SpaceToBatchNDOptions.decodeText(reader, json);
            case 'TransposeOptions': return circle.TransposeOptions.decodeText(reader, json);
            case 'ReducerOptions': return circle.ReducerOptions.decodeText(reader, json);
            case 'SubOptions': return circle.SubOptions.decodeText(reader, json);
            case 'DivOptions': return circle.DivOptions.decodeText(reader, json);
            case 'SqueezeOptions': return circle.SqueezeOptions.decodeText(reader, json);
            case 'SequenceRNNOptions': return circle.SequenceRNNOptions.decodeText(reader, json);
            case 'StridedSliceOptions': return circle.StridedSliceOptions.decodeText(reader, json);
            case 'ExpOptions': return circle.ExpOptions.decodeText(reader, json);
            case 'TopKV2Options': return circle.TopKV2Options.decodeText(reader, json);
            case 'SplitOptions': return circle.SplitOptions.decodeText(reader, json);
            case 'LogSoftmaxOptions': return circle.LogSoftmaxOptions.decodeText(reader, json);
            case 'CastOptions': return circle.CastOptions.decodeText(reader, json);
            case 'DequantizeOptions': return circle.DequantizeOptions.decodeText(reader, json);
            case 'MaximumMinimumOptions': return circle.MaximumMinimumOptions.decodeText(reader, json);
            case 'ArgMaxOptions': return circle.ArgMaxOptions.decodeText(reader, json);
            case 'LessOptions': return circle.LessOptions.decodeText(reader, json);
            case 'NegOptions': return circle.NegOptions.decodeText(reader, json);
            case 'PadV2Options': return circle.PadV2Options.decodeText(reader, json);
            case 'GreaterOptions': return circle.GreaterOptions.decodeText(reader, json);
            case 'GreaterEqualOptions': return circle.GreaterEqualOptions.decodeText(reader, json);
            case 'LessEqualOptions': return circle.LessEqualOptions.decodeText(reader, json);
            case 'SelectOptions': return circle.SelectOptions.decodeText(reader, json);
            case 'SliceOptions': return circle.SliceOptions.decodeText(reader, json);
            case 'TransposeConvOptions': return circle.TransposeConvOptions.decodeText(reader, json);
            case 'SparseToDenseOptions': return circle.SparseToDenseOptions.decodeText(reader, json);
            case 'TileOptions': return circle.TileOptions.decodeText(reader, json);
            case 'ExpandDimsOptions': return circle.ExpandDimsOptions.decodeText(reader, json);
            case 'EqualOptions': return circle.EqualOptions.decodeText(reader, json);
            case 'NotEqualOptions': return circle.NotEqualOptions.decodeText(reader, json);
            case 'ShapeOptions': return circle.ShapeOptions.decodeText(reader, json);
            case 'PowOptions': return circle.PowOptions.decodeText(reader, json);
            case 'ArgMinOptions': return circle.ArgMinOptions.decodeText(reader, json);
            case 'FakeQuantOptions': return circle.FakeQuantOptions.decodeText(reader, json);
            case 'PackOptions': return circle.PackOptions.decodeText(reader, json);
            case 'LogicalOrOptions': return circle.LogicalOrOptions.decodeText(reader, json);
            case 'OneHotOptions': return circle.OneHotOptions.decodeText(reader, json);
            case 'LogicalAndOptions': return circle.LogicalAndOptions.decodeText(reader, json);
            case 'LogicalNotOptions': return circle.LogicalNotOptions.decodeText(reader, json);
            case 'UnpackOptions': return circle.UnpackOptions.decodeText(reader, json);
            case 'FloorDivOptions': return circle.FloorDivOptions.decodeText(reader, json);
            case 'SquareOptions': return circle.SquareOptions.decodeText(reader, json);
            case 'ZerosLikeOptions': return circle.ZerosLikeOptions.decodeText(reader, json);
            case 'FillOptions': return circle.FillOptions.decodeText(reader, json);
            case 'BidirectionalSequenceLSTMOptions': return circle.BidirectionalSequenceLSTMOptions.decodeText(reader, json);
            case 'BidirectionalSequenceRNNOptions': return circle.BidirectionalSequenceRNNOptions.decodeText(reader, json);
            case 'UnidirectionalSequenceLSTMOptions': return circle.UnidirectionalSequenceLSTMOptions.decodeText(reader, json);
            case 'FloorModOptions': return circle.FloorModOptions.decodeText(reader, json);
            case 'RangeOptions': return circle.RangeOptions.decodeText(reader, json);
            case 'ResizeNearestNeighborOptions': return circle.ResizeNearestNeighborOptions.decodeText(reader, json);
            case 'LeakyReluOptions': return circle.LeakyReluOptions.decodeText(reader, json);
            case 'SquaredDifferenceOptions': return circle.SquaredDifferenceOptions.decodeText(reader, json);
            case 'MirrorPadOptions': return circle.MirrorPadOptions.decodeText(reader, json);
            case 'AbsOptions': return circle.AbsOptions.decodeText(reader, json);
            case 'SplitVOptions': return circle.SplitVOptions.decodeText(reader, json);
            case 'UniqueOptions': return circle.UniqueOptions.decodeText(reader, json);
            case 'ReverseV2Options': return circle.ReverseV2Options.decodeText(reader, json);
            case 'AddNOptions': return circle.AddNOptions.decodeText(reader, json);
            case 'GatherNdOptions': return circle.GatherNdOptions.decodeText(reader, json);
            case 'CosOptions': return circle.CosOptions.decodeText(reader, json);
            case 'WhereOptions': return circle.WhereOptions.decodeText(reader, json);
            case 'RankOptions': return circle.RankOptions.decodeText(reader, json);
            case 'ReverseSequenceOptions': return circle.ReverseSequenceOptions.decodeText(reader, json);
            case 'MatrixDiagOptions': return circle.MatrixDiagOptions.decodeText(reader, json);
            case 'QuantizeOptions': return circle.QuantizeOptions.decodeText(reader, json);
            case 'MatrixSetDiagOptions': return circle.MatrixSetDiagOptions.decodeText(reader, json);
            case 'HardSwishOptions': return circle.HardSwishOptions.decodeText(reader, json);
            case 'IfOptions': return circle.IfOptions.decodeText(reader, json);
            case 'WhileOptions': return circle.WhileOptions.decodeText(reader, json);
            case 'DepthToSpaceOptions': return circle.DepthToSpaceOptions.decodeText(reader, json);
            case 'NonMaxSuppressionV4Options': return circle.NonMaxSuppressionV4Options.decodeText(reader, json);
            case 'NonMaxSuppressionV5Options': return circle.NonMaxSuppressionV5Options.decodeText(reader, json);
            case 'ScatterNdOptions': return circle.ScatterNdOptions.decodeText(reader, json);
            case 'SelectV2Options': return circle.SelectV2Options.decodeText(reader, json);
            case 'DensifyOptions': return circle.DensifyOptions.decodeText(reader, json);
            case 'SegmentSumOptions': return circle.SegmentSumOptions.decodeText(reader, json);
            case 'BatchMatMulOptions': return circle.BatchMatMulOptions.decodeText(reader, json);
            case 'CumsumOptions': return circle.CumsumOptions.decodeText(reader, json);
            case 'CallOnceOptions': return circle.CallOnceOptions.decodeText(reader, json);
            case 'BroadcastToOptions': return circle.BroadcastToOptions.decodeText(reader, json);
            case 'Rfft2dOptions': return circle.Rfft2dOptions.decodeText(reader, json);
            case 'Conv3DOptions': return circle.Conv3DOptions.decodeText(reader, json);
            case 'HashtableOptions': return circle.HashtableOptions.decodeText(reader, json);
            case 'HashtableFindOptions': return circle.HashtableFindOptions.decodeText(reader, json);
            case 'HashtableImportOptions': return circle.HashtableImportOptions.decodeText(reader, json);
            case 'HashtableSizeOptions': return circle.HashtableSizeOptions.decodeText(reader, json);
            case 'VarHandleOptions': return circle.VarHandleOptions.decodeText(reader, json);
            case 'ReadVariableOptions': return circle.ReadVariableOptions.decodeText(reader, json);
            case 'AssignVariableOptions': return circle.AssignVariableOptions.decodeText(reader, json);
            case 'RandomOptions': return circle.RandomOptions.decodeText(reader, json);
            case 'BucketizeOptions': return circle.BucketizeOptions.decodeText(reader, json);
            case 'GeluOptions': return circle.GeluOptions.decodeText(reader, json);
            case 'DynamicUpdateSliceOptions': return circle.DynamicUpdateSliceOptions.decodeText(reader, json);
            case 'UnsortedSegmentProdOptions': return circle.UnsortedSegmentProdOptions.decodeText(reader, json);
            case 'UnsortedSegmentMaxOptions': return circle.UnsortedSegmentMaxOptions.decodeText(reader, json);
            case 'UnsortedSegmentMinOptions': return circle.UnsortedSegmentMinOptions.decodeText(reader, json);
            case 'UnsortedSegmentSumOptions': return circle.UnsortedSegmentSumOptions.decodeText(reader, json);
            case 'ATan2Options': return circle.ATan2Options.decodeText(reader, json);
            case 'SignOptions': return circle.SignOptions.decodeText(reader, json);
            case 'BitcastOptions': return circle.BitcastOptions.decodeText(reader, json);
            case 'BitwiseXorOptions': return circle.BitwiseXorOptions.decodeText(reader, json);
            case 'RightShiftOptions': return circle.RightShiftOptions.decodeText(reader, json);
            case 'GRUOptions': return circle.GRUOptions.decodeText(reader, json);
            case 'BCQGatherOptions': return circle.BCQGatherOptions.decodeText(reader, json);
            case 'BCQFullyConnectedOptions': return circle.BCQFullyConnectedOptions.decodeText(reader, json);
            case 'InstanceNormOptions': return circle.InstanceNormOptions.decodeText(reader, json);
            default: return undefined;
        }
    }
};

circle.BuiltinOptions2 = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return circle.StablehloConcatenateOptions.decode(reader, position);
            case 2: return circle.StablehloBroadcastInDimOptions.decode(reader, position);
            case 3: return circle.StablehloSliceOptions.decode(reader, position);
            case 4: return circle.StablehloConvolutionOptions.decode(reader, position);
            case 5: return circle.StablehloCustomCallOptions.decode(reader, position);
            case 6: return circle.StablehloReduceOptions.decode(reader, position);
            case 7: return circle.StablehloScatterOptions.decode(reader, position);
            case 8: return circle.StablehloCompareOptions.decode(reader, position);
            case 9: return circle.StablehloDynamicSliceOptions.decode(reader, position);
            case 10: return circle.StablehloPadOptions.decode(reader, position);
            case 11: return circle.StablehloIotaOptions.decode(reader, position);
            case 12: return circle.StablehloDotGeneralOptions.decode(reader, position);
            case 13: return circle.StablehloReduceWindowOptions.decode(reader, position);
            case 14: return circle.StablehloSortOptions.decode(reader, position);
            case 15: return circle.StablehloWhileOptions.decode(reader, position);
            case 16: return circle.StablehloGatherOptions.decode(reader, position);
            case 17: return circle.StablehloTransposeOptions.decode(reader, position);
            case 18: return circle.DilateOptions.decode(reader, position);
            case 19: return circle.StablehloRngBitGeneratorOptions.decode(reader, position);
            case 20: return circle.ReduceWindowOptions.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'StablehloConcatenateOptions': return circle.StablehloConcatenateOptions.decodeText(reader, json);
            case 'StablehloBroadcastInDimOptions': return circle.StablehloBroadcastInDimOptions.decodeText(reader, json);
            case 'StablehloSliceOptions': return circle.StablehloSliceOptions.decodeText(reader, json);
            case 'StablehloConvolutionOptions': return circle.StablehloConvolutionOptions.decodeText(reader, json);
            case 'StablehloCustomCallOptions': return circle.StablehloCustomCallOptions.decodeText(reader, json);
            case 'StablehloReduceOptions': return circle.StablehloReduceOptions.decodeText(reader, json);
            case 'StablehloScatterOptions': return circle.StablehloScatterOptions.decodeText(reader, json);
            case 'StablehloCompareOptions': return circle.StablehloCompareOptions.decodeText(reader, json);
            case 'StablehloDynamicSliceOptions': return circle.StablehloDynamicSliceOptions.decodeText(reader, json);
            case 'StablehloPadOptions': return circle.StablehloPadOptions.decodeText(reader, json);
            case 'StablehloIotaOptions': return circle.StablehloIotaOptions.decodeText(reader, json);
            case 'StablehloDotGeneralOptions': return circle.StablehloDotGeneralOptions.decodeText(reader, json);
            case 'StablehloReduceWindowOptions': return circle.StablehloReduceWindowOptions.decodeText(reader, json);
            case 'StablehloSortOptions': return circle.StablehloSortOptions.decodeText(reader, json);
            case 'StablehloWhileOptions': return circle.StablehloWhileOptions.decodeText(reader, json);
            case 'StablehloGatherOptions': return circle.StablehloGatherOptions.decodeText(reader, json);
            case 'StablehloTransposeOptions': return circle.StablehloTransposeOptions.decodeText(reader, json);
            case 'DilateOptions': return circle.DilateOptions.decodeText(reader, json);
            case 'StablehloRngBitGeneratorOptions': return circle.StablehloRngBitGeneratorOptions.decodeText(reader, json);
            case 'ReduceWindowOptions': return circle.ReduceWindowOptions.decodeText(reader, json);
            default: return undefined;
        }
    }
};

circle.StablehloGatherOptions = class StablehloGatherOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloGatherOptions();
        $.offset_dims = reader.int64s_(position, 4);
        $.collapsed_slice_dims = reader.int64s_(position, 6);
        $.start_index_map = reader.int64s_(position, 8);
        $.index_vector_dim = reader.int64_(position, 10, 0n);
        $.slice_sizes = reader.int64s_(position, 12);
        $.indices_are_sorted = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloGatherOptions();
        $.offset_dims = reader.array(json.offset_dims);
        $.collapsed_slice_dims = reader.array(json.collapsed_slice_dims);
        $.start_index_map = reader.array(json.start_index_map);
        $.index_vector_dim = reader.int64(json.index_vector_dim, 0n);
        $.slice_sizes = reader.array(json.slice_sizes);
        $.indices_are_sorted = reader.value(json.indices_are_sorted, false);
        return $;
    }
};

circle.StablehloTransposeOptions = class StablehloTransposeOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloTransposeOptions();
        $.permutation = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloTransposeOptions();
        $.permutation = reader.array(json.permutation);
        return $;
    }
};

circle.StablehloPrecisionConfig = {
    DEFAULT: 0,
    HIGH: 1,
    HIGHEST: 2
};

circle.StablehloDotGeneralOptions = class StablehloDotGeneralOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloDotGeneralOptions();
        $.lhs_batching_dimensions = reader.int64s_(position, 4);
        $.rhs_batching_dimensions = reader.int64s_(position, 6);
        $.lhs_contracting_dimensions = reader.int64s_(position, 8);
        $.rhs_contracting_dimensions = reader.int64s_(position, 10);
        $.precision_config = reader.array(position, 12, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloDotGeneralOptions();
        $.lhs_batching_dimensions = reader.array(json.lhs_batching_dimensions);
        $.rhs_batching_dimensions = reader.array(json.rhs_batching_dimensions);
        $.lhs_contracting_dimensions = reader.array(json.lhs_contracting_dimensions);
        $.rhs_contracting_dimensions = reader.array(json.rhs_contracting_dimensions);
        $.precision_config = reader.objects(json.precision_config, circle.StablehloPrecisionConfig);
        return $;
    }
};

circle.StablehloReduceWindowOptions = class StablehloReduceWindowOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloReduceWindowOptions();
        $.window_dimensions = reader.int64s_(position, 4);
        $.window_strides = reader.int64s_(position, 6);
        $.base_dilations = reader.int64s_(position, 8);
        $.window_dilations = reader.int64s_(position, 10);
        $.padding = reader.int64s_(position, 12);
        $.body_subgraph_index = reader.int32_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloReduceWindowOptions();
        $.window_dimensions = reader.array(json.window_dimensions);
        $.window_strides = reader.array(json.window_strides);
        $.base_dilations = reader.array(json.base_dilations);
        $.window_dilations = reader.array(json.window_dilations);
        $.padding = reader.array(json.padding);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

circle.StablehloWhileOptions = class StablehloWhileOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloWhileOptions();
        $.cond_subgraph_index = reader.int32_(position, 4, 0);
        $.body_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloWhileOptions();
        $.cond_subgraph_index = reader.value(json.cond_subgraph_index, 0);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

circle.StablehloSortOptions = class StablehloSortOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloSortOptions();
        $.dimension = reader.int64_(position, 4, 0n);
        $.is_stable = reader.bool_(position, 6, false);
        $.comparator_subgraph_index = reader.int32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloSortOptions();
        $.dimension = reader.int64(json.dimension, 0n);
        $.is_stable = reader.value(json.is_stable, false);
        $.comparator_subgraph_index = reader.value(json.comparator_subgraph_index, 0);
        return $;
    }
};

circle.StablehloConcatenateOptions = class StablehloConcatenateOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloConcatenateOptions();
        $.dimension = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloConcatenateOptions();
        $.dimension = reader.int64(json.dimension, 0n);
        return $;
    }
};

circle.StablehloBroadcastInDimOptions = class StablehloBroadcastInDimOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloBroadcastInDimOptions();
        $.broadcast_dimensions = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloBroadcastInDimOptions();
        $.broadcast_dimensions = reader.array(json.broadcast_dimensions);
        return $;
    }
};

circle.StablehloComparisonDirection = {
    STABLEHLO_COMPARISON_DIRECTION_EQ: 0,
    STABLEHLO_COMPARISON_DIRECTION_NE: 1,
    STABLEHLO_COMPARISON_DIRECTION_GE: 2,
    STABLEHLO_COMPARISON_DIRECTION_GT: 3,
    STABLEHLO_COMPARISON_DIRECTION_LE: 4,
    STABLEHLO_COMPARISON_DIRECTION_LT: 5
};

circle.StablehloComparisonType = {
    STABLEHLO_COMPARISON_TYPE_NOTYPE: 0,
    STABLEHLO_COMPARISON_TYPE_FLOAT: 1,
    STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER: 2,
    STABLEHLO_COMPARISON_TYPE_SIGNED: 3,
    STABLEHLO_COMPARISON_TYPE_UNSIGNED: 4
};

circle.StablehloCompareOptions = class StablehloCompareOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloCompareOptions();
        $.comparison_direction = reader.uint32_(position, 4, 0);
        $.compare_type = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloCompareOptions();
        $.comparison_direction = circle.StablehloComparisonDirection[json.comparison_direction];
        $.compare_type = circle.StablehloComparisonType[json.compare_type];
        return $;
    }
};

circle.StablehloDynamicSliceOptions = class StablehloDynamicSliceOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloDynamicSliceOptions();
        $.slice_sizes = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloDynamicSliceOptions();
        $.slice_sizes = reader.array(json.slice_sizes);
        return $;
    }
};

circle.StablehloPadOptions = class StablehloPadOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloPadOptions();
        $.edge_padding_low = reader.int64s_(position, 4);
        $.edge_padding_high = reader.int64s_(position, 6);
        $.interior_padding = reader.int64s_(position, 8);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloPadOptions();
        $.edge_padding_low = reader.array(json.edge_padding_low);
        $.edge_padding_high = reader.array(json.edge_padding_high);
        $.interior_padding = reader.array(json.interior_padding);
        return $;
    }
};

circle.StablehloIotaOptions = class StablehloIotaOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloIotaOptions();
        $.iota_dimension = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloIotaOptions();
        $.iota_dimension = reader.int64(json.iota_dimension, 0n);
        return $;
    }
};

circle.StablehloCustomCallOptions = class StablehloCustomCallOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloCustomCallOptions();
        $.call_target_name = reader.string_(position, 4, null);
        $.has_side_effect = reader.bool_(position, 6, false);
        $.backend_config = reader.string_(position, 8, null);
        $.api_version = reader.int32_(position, 10, 0);
        $.called_computations = reader.array(position, 12, Int32Array);
        $.custom_attributes = reader.array(position, 14, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloCustomCallOptions();
        $.call_target_name = reader.value(json.call_target_name, null);
        $.has_side_effect = reader.value(json.has_side_effect, false);
        $.backend_config = reader.value(json.backend_config, null);
        $.api_version = reader.value(json.api_version, 0);
        $.called_computations = reader.array(json.called_computations, Int32Array);
        $.custom_attributes = reader.array(json.custom_attributes, Uint8Array);
        return $;
    }
};

circle.StablehloReduceOptions = class StablehloReduceOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloReduceOptions();
        $.dimensions = reader.int64s_(position, 4);
        $.body_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloReduceOptions();
        $.dimensions = reader.array(json.dimensions);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

circle.StablehloSliceOptions = class StablehloSliceOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloSliceOptions();
        $.start_indices = reader.int64s_(position, 4);
        $.limit_indices = reader.int64s_(position, 6);
        $.strides = reader.int64s_(position, 8);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloSliceOptions();
        $.start_indices = reader.array(json.start_indices);
        $.limit_indices = reader.array(json.limit_indices);
        $.strides = reader.array(json.strides);
        return $;
    }
};

circle.StablehloConvolutionOptions = class StablehloConvolutionOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloConvolutionOptions();
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
        const $ = new circle.StablehloConvolutionOptions();
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
        $.precision_config = reader.objects(json.precision_config, circle.StablehloPrecisionConfig);
        return $;
    }
};

circle.StablehloScatterOptions = class StablehloScatterOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloScatterOptions();
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
        const $ = new circle.StablehloScatterOptions();
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

circle.RngAlgorithm = {
    DEFAULT: 0,
    PHILOX: 1,
    THREEFRY: 2
};

circle.StablehloRngBitGeneratorOptions = class StablehloRngBitGeneratorOptions {

    static decode(reader, position) {
        const $ = new circle.StablehloRngBitGeneratorOptions();
        $.algorithm = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StablehloRngBitGeneratorOptions();
        $.algorithm = circle.RngAlgorithm[json.algorithm];
        return $;
    }
};

circle.Padding = {
    SAME: 0,
    VALID: 1
};

circle.ActivationFunctionType = {
    NONE: 0,
    RELU: 1,
    RELU_N1_TO_1: 2,
    RELU6: 3,
    TANH: 4,
    SIGN_BIT: 5
};

circle.Conv2DOptions = class Conv2DOptions {

    static decode(reader, position) {
        const $ = new circle.Conv2DOptions();
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
        const $ = new circle.Conv2DOptions();
        $.padding = circle.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.dilation_w_factor = reader.value(json.dilation_w_factor, 1);
        $.dilation_h_factor = reader.value(json.dilation_h_factor, 1);
        $.quantized_bias_type = circle.TensorType[json.quantized_bias_type];
        return $;
    }
};

circle.Conv3DOptions = class Conv3DOptions {

    static decode(reader, position) {
        const $ = new circle.Conv3DOptions();
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
        const $ = new circle.Conv3DOptions();
        $.padding = circle.Padding[json.padding];
        $.stride_d = reader.value(json.stride_d, 0);
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.dilation_d_factor = reader.value(json.dilation_d_factor, 1);
        $.dilation_w_factor = reader.value(json.dilation_w_factor, 1);
        $.dilation_h_factor = reader.value(json.dilation_h_factor, 1);
        return $;
    }
};

circle.Pool2DOptions = class Pool2DOptions {

    static decode(reader, position) {
        const $ = new circle.Pool2DOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_w = reader.int32_(position, 6, 0);
        $.stride_h = reader.int32_(position, 8, 0);
        $.filter_width = reader.int32_(position, 10, 0);
        $.filter_height = reader.int32_(position, 12, 0);
        $.fused_activation_function = reader.int8_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Pool2DOptions();
        $.padding = circle.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.filter_width = reader.value(json.filter_width, 0);
        $.filter_height = reader.value(json.filter_height, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.DepthwiseConv2DOptions = class DepthwiseConv2DOptions {

    static decode(reader, position) {
        const $ = new circle.DepthwiseConv2DOptions();
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
        const $ = new circle.DepthwiseConv2DOptions();
        $.padding = circle.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.depth_multiplier = reader.value(json.depth_multiplier, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.dilation_w_factor = reader.value(json.dilation_w_factor, 1);
        $.dilation_h_factor = reader.value(json.dilation_h_factor, 1);
        return $;
    }
};

circle.ConcatEmbeddingsOptions = class ConcatEmbeddingsOptions {

    static decode(reader, position) {
        const $ = new circle.ConcatEmbeddingsOptions();
        $.num_channels = reader.int32_(position, 4, 0);
        $.num_columns_per_channel = reader.array(position, 6, Int32Array);
        $.embedding_dim_per_channel = reader.array(position, 8, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ConcatEmbeddingsOptions();
        $.num_channels = reader.value(json.num_channels, 0);
        $.num_columns_per_channel = reader.array(json.num_columns_per_channel, Int32Array);
        $.embedding_dim_per_channel = reader.array(json.embedding_dim_per_channel, Int32Array);
        return $;
    }
};

circle.LSHProjectionType = {
    UNKNOWN: 0,
    SPARSE: 1,
    DENSE: 2
};

circle.LSHProjectionOptions = class LSHProjectionOptions {

    static decode(reader, position) {
        const $ = new circle.LSHProjectionOptions();
        $.type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.LSHProjectionOptions();
        $.type = circle.LSHProjectionType[json.type];
        return $;
    }
};

circle.SVDFOptions = class SVDFOptions {

    static decode(reader, position) {
        const $ = new circle.SVDFOptions();
        $.rank = reader.int32_(position, 4, 0);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SVDFOptions();
        $.rank = reader.value(json.rank, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.RNNOptions = class RNNOptions {

    static decode(reader, position) {
        const $ = new circle.RNNOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.RNNOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.SequenceRNNOptions = class SequenceRNNOptions {

    static decode(reader, position) {
        const $ = new circle.SequenceRNNOptions();
        $.time_major = reader.bool_(position, 4, false);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SequenceRNNOptions();
        $.time_major = reader.value(json.time_major, false);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.BidirectionalSequenceRNNOptions = class BidirectionalSequenceRNNOptions {

    static decode(reader, position) {
        const $ = new circle.BidirectionalSequenceRNNOptions();
        $.time_major = reader.bool_(position, 4, false);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        $.merge_outputs = reader.bool_(position, 8, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.BidirectionalSequenceRNNOptions();
        $.time_major = reader.value(json.time_major, false);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.merge_outputs = reader.value(json.merge_outputs, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.FullyConnectedOptionsWeightsFormat = {
    DEFAULT: 0,
    SHUFFLED4x16INT8: 1,
    SHUFFLED16x1FLOAT32: 127
};

circle.FullyConnectedOptions = class FullyConnectedOptions {

    static decode(reader, position) {
        const $ = new circle.FullyConnectedOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.weights_format = reader.int8_(position, 6, 0);
        $.keep_num_dims = reader.bool_(position, 8, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 10, false);
        $.quantized_bias_type = reader.int8_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.FullyConnectedOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.weights_format = circle.FullyConnectedOptionsWeightsFormat[json.weights_format];
        $.keep_num_dims = reader.value(json.keep_num_dims, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        $.quantized_bias_type = circle.TensorType[json.quantized_bias_type];
        return $;
    }
};

circle.SoftmaxOptions = class SoftmaxOptions {

    static decode(reader, position) {
        const $ = new circle.SoftmaxOptions();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SoftmaxOptions();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

circle.ConcatenationOptions = class ConcatenationOptions {

    static decode(reader, position) {
        const $ = new circle.ConcatenationOptions();
        $.axis = reader.int32_(position, 4, 0);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ConcatenationOptions();
        $.axis = reader.value(json.axis, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.AddOptions = class AddOptions {

    static decode(reader, position) {
        const $ = new circle.AddOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.pot_scale_int16 = reader.bool_(position, 6, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.AddOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.pot_scale_int16 = reader.value(json.pot_scale_int16, true);
        return $;
    }
};

circle.MulOptions = class MulOptions {

    static decode(reader, position) {
        const $ = new circle.MulOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.MulOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.L2NormOptions = class L2NormOptions {

    static decode(reader, position) {
        const $ = new circle.L2NormOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.L2NormOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.LocalResponseNormalizationOptions = class LocalResponseNormalizationOptions {

    static decode(reader, position) {
        const $ = new circle.LocalResponseNormalizationOptions();
        $.radius = reader.int32_(position, 4, 0);
        $.bias = reader.float32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.LocalResponseNormalizationOptions();
        $.radius = reader.value(json.radius, 0);
        $.bias = reader.value(json.bias, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

circle.LSTMKernelType = {
    FULL: 0,
    BASIC: 1
};

circle.LSTMOptions = class LSTMOptions {

    static decode(reader, position) {
        const $ = new circle.LSTMOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.cell_clip = reader.float32_(position, 6, 0);
        $.proj_clip = reader.float32_(position, 8, 0);
        $.kernel_type = reader.int8_(position, 10, 0);
        $.asymmetric_quantize_inputs = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.LSTMOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.cell_clip = reader.value(json.cell_clip, 0);
        $.proj_clip = reader.value(json.proj_clip, 0);
        $.kernel_type = circle.LSTMKernelType[json.kernel_type];
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.UnidirectionalSequenceLSTMOptions = class UnidirectionalSequenceLSTMOptions {

    static decode(reader, position) {
        const $ = new circle.UnidirectionalSequenceLSTMOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.cell_clip = reader.float32_(position, 6, 0);
        $.proj_clip = reader.float32_(position, 8, 0);
        $.time_major = reader.bool_(position, 10, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 12, false);
        $.diagonal_recurrent_tensors = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.UnidirectionalSequenceLSTMOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.cell_clip = reader.value(json.cell_clip, 0);
        $.proj_clip = reader.value(json.proj_clip, 0);
        $.time_major = reader.value(json.time_major, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        $.diagonal_recurrent_tensors = reader.value(json.diagonal_recurrent_tensors, false);
        return $;
    }
};

circle.BidirectionalSequenceLSTMOptions = class BidirectionalSequenceLSTMOptions {

    static decode(reader, position) {
        const $ = new circle.BidirectionalSequenceLSTMOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.cell_clip = reader.float32_(position, 6, 0);
        $.proj_clip = reader.float32_(position, 8, 0);
        $.merge_outputs = reader.bool_(position, 10, false);
        $.time_major = reader.bool_(position, 12, true);
        $.asymmetric_quantize_inputs = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.BidirectionalSequenceLSTMOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.cell_clip = reader.value(json.cell_clip, 0);
        $.proj_clip = reader.value(json.proj_clip, 0);
        $.merge_outputs = reader.value(json.merge_outputs, false);
        $.time_major = reader.value(json.time_major, true);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.ResizeBilinearOptions = class ResizeBilinearOptions {

    static decode(reader, position) {
        const $ = new circle.ResizeBilinearOptions();
        $.new_height = reader.int32_(position, 4, 0);
        $.new_width = reader.int32_(position, 6, 0);
        $.align_corners = reader.bool_(position, 8, false);
        $.half_pixel_centers = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ResizeBilinearOptions();
        $.new_height = reader.value(json.new_height, 0);
        $.new_width = reader.value(json.new_width, 0);
        $.align_corners = reader.value(json.align_corners, false);
        $.half_pixel_centers = reader.value(json.half_pixel_centers, false);
        return $;
    }
};

circle.ResizeNearestNeighborOptions = class ResizeNearestNeighborOptions {

    static decode(reader, position) {
        const $ = new circle.ResizeNearestNeighborOptions();
        $.align_corners = reader.bool_(position, 4, false);
        $.half_pixel_centers = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ResizeNearestNeighborOptions();
        $.align_corners = reader.value(json.align_corners, false);
        $.half_pixel_centers = reader.value(json.half_pixel_centers, false);
        return $;
    }
};

circle.CallOptions = class CallOptions {

    static decode(reader, position) {
        const $ = new circle.CallOptions();
        $.subgraph = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.CallOptions();
        $.subgraph = reader.value(json.subgraph, 0);
        return $;
    }
};

circle.PadOptions = class PadOptions {

    static decode(/* reader, position */) {
        const $ = new circle.PadOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.PadOptions();
        return $;
    }
};

circle.PadV2Options = class PadV2Options {

    static decode(/* reader, position */) {
        const $ = new circle.PadV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.PadV2Options();
        return $;
    }
};

circle.ReshapeOptions = class ReshapeOptions {

    static decode(reader, position) {
        const $ = new circle.ReshapeOptions();
        $.new_shape = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ReshapeOptions();
        $.new_shape = reader.array(json.new_shape, Int32Array);
        return $;
    }
};

circle.SpaceToBatchNDOptions = class SpaceToBatchNDOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SpaceToBatchNDOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SpaceToBatchNDOptions();
        return $;
    }
};

circle.BatchToSpaceNDOptions = class BatchToSpaceNDOptions {

    static decode(/* reader, position */) {
        const $ = new circle.BatchToSpaceNDOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.BatchToSpaceNDOptions();
        return $;
    }
};

circle.SkipGramOptions = class SkipGramOptions {

    static decode(reader, position) {
        const $ = new circle.SkipGramOptions();
        $.ngram_size = reader.int32_(position, 4, 0);
        $.max_skip_size = reader.int32_(position, 6, 0);
        $.include_all_ngrams = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SkipGramOptions();
        $.ngram_size = reader.value(json.ngram_size, 0);
        $.max_skip_size = reader.value(json.max_skip_size, 0);
        $.include_all_ngrams = reader.value(json.include_all_ngrams, false);
        return $;
    }
};

circle.SpaceToDepthOptions = class SpaceToDepthOptions {

    static decode(reader, position) {
        const $ = new circle.SpaceToDepthOptions();
        $.block_size = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SpaceToDepthOptions();
        $.block_size = reader.value(json.block_size, 0);
        return $;
    }
};

circle.DepthToSpaceOptions = class DepthToSpaceOptions {

    static decode(reader, position) {
        const $ = new circle.DepthToSpaceOptions();
        $.block_size = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.DepthToSpaceOptions();
        $.block_size = reader.value(json.block_size, 0);
        return $;
    }
};

circle.SubOptions = class SubOptions {

    static decode(reader, position) {
        const $ = new circle.SubOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.pot_scale_int16 = reader.bool_(position, 6, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SubOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.pot_scale_int16 = reader.value(json.pot_scale_int16, true);
        return $;
    }
};

circle.DivOptions = class DivOptions {

    static decode(reader, position) {
        const $ = new circle.DivOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.DivOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.TopKV2Options = class TopKV2Options {

    static decode(/* reader, position */) {
        const $ = new circle.TopKV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.TopKV2Options();
        return $;
    }
};

circle.CombinerType = {
    SUM: 0,
    MEAN: 1,
    SQRTN: 2
};

circle.EmbeddingLookupSparseOptions = class EmbeddingLookupSparseOptions {

    static decode(reader, position) {
        const $ = new circle.EmbeddingLookupSparseOptions();
        $.combiner = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.EmbeddingLookupSparseOptions();
        $.combiner = circle.CombinerType[json.combiner];
        return $;
    }
};

circle.GatherOptions = class GatherOptions {

    static decode(reader, position) {
        const $ = new circle.GatherOptions();
        $.axis = reader.int32_(position, 4, 0);
        $.batch_dims = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.GatherOptions();
        $.axis = reader.value(json.axis, 0);
        $.batch_dims = reader.value(json.batch_dims, 0);
        return $;
    }
};

circle.TransposeOptions = class TransposeOptions {

    static decode(/* reader, position */) {
        const $ = new circle.TransposeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.TransposeOptions();
        return $;
    }
};

circle.ExpOptions = class ExpOptions {

    static decode(/* reader, position */) {
        const $ = new circle.ExpOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ExpOptions();
        return $;
    }
};

circle.CosOptions = class CosOptions {

    static decode(/* reader, position */) {
        const $ = new circle.CosOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.CosOptions();
        return $;
    }
};

circle.ReducerOptions = class ReducerOptions {

    static decode(reader, position) {
        const $ = new circle.ReducerOptions();
        $.keep_dims = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ReducerOptions();
        $.keep_dims = reader.value(json.keep_dims, false);
        return $;
    }
};

circle.SqueezeOptions = class SqueezeOptions {

    static decode(reader, position) {
        const $ = new circle.SqueezeOptions();
        $.squeeze_dims = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SqueezeOptions();
        $.squeeze_dims = reader.array(json.squeeze_dims, Int32Array);
        return $;
    }
};

circle.SplitOptions = class SplitOptions {

    static decode(reader, position) {
        const $ = new circle.SplitOptions();
        $.num_splits = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SplitOptions();
        $.num_splits = reader.value(json.num_splits, 0);
        return $;
    }
};

circle.SplitVOptions = class SplitVOptions {

    static decode(reader, position) {
        const $ = new circle.SplitVOptions();
        $.num_splits = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SplitVOptions();
        $.num_splits = reader.value(json.num_splits, 0);
        return $;
    }
};

circle.StridedSliceOptions = class StridedSliceOptions {

    static decode(reader, position) {
        const $ = new circle.StridedSliceOptions();
        $.begin_mask = reader.int32_(position, 4, 0);
        $.end_mask = reader.int32_(position, 6, 0);
        $.ellipsis_mask = reader.int32_(position, 8, 0);
        $.new_axis_mask = reader.int32_(position, 10, 0);
        $.shrink_axis_mask = reader.int32_(position, 12, 0);
        $.offset = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.StridedSliceOptions();
        $.begin_mask = reader.value(json.begin_mask, 0);
        $.end_mask = reader.value(json.end_mask, 0);
        $.ellipsis_mask = reader.value(json.ellipsis_mask, 0);
        $.new_axis_mask = reader.value(json.new_axis_mask, 0);
        $.shrink_axis_mask = reader.value(json.shrink_axis_mask, 0);
        $.offset = reader.value(json.offset, false);
        return $;
    }
};

circle.LogSoftmaxOptions = class LogSoftmaxOptions {

    static decode(/* reader, position */) {
        const $ = new circle.LogSoftmaxOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.LogSoftmaxOptions();
        return $;
    }
};

circle.CastOptions = class CastOptions {

    static decode(reader, position) {
        const $ = new circle.CastOptions();
        $.in_data_type = reader.int8_(position, 4, 0);
        $.out_data_type = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.CastOptions();
        $.in_data_type = circle.TensorType[json.in_data_type];
        $.out_data_type = circle.TensorType[json.out_data_type];
        return $;
    }
};

circle.DequantizeOptions = class DequantizeOptions {

    static decode(/* reader, position */) {
        const $ = new circle.DequantizeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.DequantizeOptions();
        return $;
    }
};

circle.MaximumMinimumOptions = class MaximumMinimumOptions {

    static decode(/* reader, position */) {
        const $ = new circle.MaximumMinimumOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.MaximumMinimumOptions();
        return $;
    }
};

circle.TileOptions = class TileOptions {

    static decode(/* reader, position */) {
        const $ = new circle.TileOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.TileOptions();
        return $;
    }
};

circle.ArgMaxOptions = class ArgMaxOptions {

    static decode(reader, position) {
        const $ = new circle.ArgMaxOptions();
        $.output_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ArgMaxOptions();
        $.output_type = circle.TensorType[json.output_type];
        return $;
    }
};

circle.ArgMinOptions = class ArgMinOptions {

    static decode(reader, position) {
        const $ = new circle.ArgMinOptions();
        $.output_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ArgMinOptions();
        $.output_type = circle.TensorType[json.output_type];
        return $;
    }
};

circle.GreaterOptions = class GreaterOptions {

    static decode(/* reader, position */) {
        const $ = new circle.GreaterOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.GreaterOptions();
        return $;
    }
};

circle.GreaterEqualOptions = class GreaterEqualOptions {

    static decode(/* reader, position */) {
        const $ = new circle.GreaterEqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.GreaterEqualOptions();
        return $;
    }
};

circle.LessOptions = class LessOptions {

    static decode(/* reader, position */) {
        const $ = new circle.LessOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.LessOptions();
        return $;
    }
};

circle.LessEqualOptions = class LessEqualOptions {

    static decode(/* reader, position */) {
        const $ = new circle.LessEqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.LessEqualOptions();
        return $;
    }
};

circle.NegOptions = class NegOptions {

    static decode(/* reader, position */) {
        const $ = new circle.NegOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.NegOptions();
        return $;
    }
};

circle.SelectOptions = class SelectOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SelectOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SelectOptions();
        return $;
    }
};

circle.SliceOptions = class SliceOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SliceOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SliceOptions();
        return $;
    }
};

circle.TransposeConvOptions = class TransposeConvOptions {

    static decode(reader, position) {
        const $ = new circle.TransposeConvOptions();
        $.padding = reader.int8_(position, 4, 0);
        $.stride_w = reader.int32_(position, 6, 0);
        $.stride_h = reader.int32_(position, 8, 0);
        $.fused_activation_function = reader.int8_(position, 10, 0);
        $.quantized_bias_type = reader.int8_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.TransposeConvOptions();
        $.padding = circle.Padding[json.padding];
        $.stride_w = reader.value(json.stride_w, 0);
        $.stride_h = reader.value(json.stride_h, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.quantized_bias_type = circle.TensorType[json.quantized_bias_type];
        return $;
    }
};

circle.ExpandDimsOptions = class ExpandDimsOptions {

    static decode(/* reader, position */) {
        const $ = new circle.ExpandDimsOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ExpandDimsOptions();
        return $;
    }
};

circle.SparseToDenseOptions = class SparseToDenseOptions {

    static decode(reader, position) {
        const $ = new circle.SparseToDenseOptions();
        $.validate_indices = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SparseToDenseOptions();
        $.validate_indices = reader.value(json.validate_indices, false);
        return $;
    }
};

circle.EqualOptions = class EqualOptions {

    static decode(/* reader, position */) {
        const $ = new circle.EqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.EqualOptions();
        return $;
    }
};

circle.NotEqualOptions = class NotEqualOptions {

    static decode(/* reader, position */) {
        const $ = new circle.NotEqualOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.NotEqualOptions();
        return $;
    }
};

circle.ShapeOptions = class ShapeOptions {

    static decode(reader, position) {
        const $ = new circle.ShapeOptions();
        $.out_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ShapeOptions();
        $.out_type = circle.TensorType[json.out_type];
        return $;
    }
};

circle.RankOptions = class RankOptions {

    static decode(/* reader, position */) {
        const $ = new circle.RankOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.RankOptions();
        return $;
    }
};

circle.PowOptions = class PowOptions {

    static decode(/* reader, position */) {
        const $ = new circle.PowOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.PowOptions();
        return $;
    }
};

circle.FakeQuantOptions = class FakeQuantOptions {

    static decode(reader, position) {
        const $ = new circle.FakeQuantOptions();
        $.min = reader.float32_(position, 4, 0);
        $.max = reader.float32_(position, 6, 0);
        $.num_bits = reader.int32_(position, 8, 0);
        $.narrow_range = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.FakeQuantOptions();
        $.min = reader.value(json.min, 0);
        $.max = reader.value(json.max, 0);
        $.num_bits = reader.value(json.num_bits, 0);
        $.narrow_range = reader.value(json.narrow_range, false);
        return $;
    }
};

circle.PackOptions = class PackOptions {

    static decode(reader, position) {
        const $ = new circle.PackOptions();
        $.values_count = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.PackOptions();
        $.values_count = reader.value(json.values_count, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

circle.LogicalOrOptions = class LogicalOrOptions {

    static decode(/* reader, position */) {
        const $ = new circle.LogicalOrOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.LogicalOrOptions();
        return $;
    }
};

circle.OneHotOptions = class OneHotOptions {

    static decode(reader, position) {
        const $ = new circle.OneHotOptions();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.OneHotOptions();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

circle.AbsOptions = class AbsOptions {

    static decode(/* reader, position */) {
        const $ = new circle.AbsOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.AbsOptions();
        return $;
    }
};

circle.HardSwishOptions = class HardSwishOptions {

    static decode(/* reader, position */) {
        const $ = new circle.HardSwishOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.HardSwishOptions();
        return $;
    }
};

circle.LogicalAndOptions = class LogicalAndOptions {

    static decode(/* reader, position */) {
        const $ = new circle.LogicalAndOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.LogicalAndOptions();
        return $;
    }
};

circle.LogicalNotOptions = class LogicalNotOptions {

    static decode(/* reader, position */) {
        const $ = new circle.LogicalNotOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.LogicalNotOptions();
        return $;
    }
};

circle.UnpackOptions = class UnpackOptions {

    static decode(reader, position) {
        const $ = new circle.UnpackOptions();
        $.num = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.UnpackOptions();
        $.num = reader.value(json.num, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

circle.FloorDivOptions = class FloorDivOptions {

    static decode(/* reader, position */) {
        const $ = new circle.FloorDivOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.FloorDivOptions();
        return $;
    }
};

circle.SquareOptions = class SquareOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SquareOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SquareOptions();
        return $;
    }
};

circle.ZerosLikeOptions = class ZerosLikeOptions {

    static decode(/* reader, position */) {
        const $ = new circle.ZerosLikeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ZerosLikeOptions();
        return $;
    }
};

circle.FillOptions = class FillOptions {

    static decode(/* reader, position */) {
        const $ = new circle.FillOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.FillOptions();
        return $;
    }
};

circle.FloorModOptions = class FloorModOptions {

    static decode(/* reader, position */) {
        const $ = new circle.FloorModOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.FloorModOptions();
        return $;
    }
};

circle.RangeOptions = class RangeOptions {

    static decode(/* reader, position */) {
        const $ = new circle.RangeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.RangeOptions();
        return $;
    }
};

circle.LeakyReluOptions = class LeakyReluOptions {

    static decode(reader, position) {
        const $ = new circle.LeakyReluOptions();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.LeakyReluOptions();
        $.alpha = reader.value(json.alpha, 0);
        return $;
    }
};

circle.SquaredDifferenceOptions = class SquaredDifferenceOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SquaredDifferenceOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SquaredDifferenceOptions();
        return $;
    }
};

circle.MirrorPadMode = {
    REFLECT: 0,
    SYMMETRIC: 1
};

circle.MirrorPadOptions = class MirrorPadOptions {

    static decode(reader, position) {
        const $ = new circle.MirrorPadOptions();
        $.mode = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.MirrorPadOptions();
        $.mode = circle.MirrorPadMode[json.mode];
        return $;
    }
};

circle.UniqueOptions = class UniqueOptions {

    static decode(reader, position) {
        const $ = new circle.UniqueOptions();
        $.idx_out_type = reader.int8_(position, 4, 2);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.UniqueOptions();
        $.idx_out_type = circle.TensorType[json.idx_out_type];
        return $;
    }
};

circle.ReverseV2Options = class ReverseV2Options {

    static decode(/* reader, position */) {
        const $ = new circle.ReverseV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ReverseV2Options();
        return $;
    }
};

circle.AddNOptions = class AddNOptions {

    static decode(/* reader, position */) {
        const $ = new circle.AddNOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.AddNOptions();
        return $;
    }
};

circle.GatherNdOptions = class GatherNdOptions {

    static decode(/* reader, position */) {
        const $ = new circle.GatherNdOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.GatherNdOptions();
        return $;
    }
};

circle.WhereOptions = class WhereOptions {

    static decode(/* reader, position */) {
        const $ = new circle.WhereOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.WhereOptions();
        return $;
    }
};

circle.ReverseSequenceOptions = class ReverseSequenceOptions {

    static decode(reader, position) {
        const $ = new circle.ReverseSequenceOptions();
        $.seq_dim = reader.int32_(position, 4, 0);
        $.batch_dim = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ReverseSequenceOptions();
        $.seq_dim = reader.value(json.seq_dim, 0);
        $.batch_dim = reader.value(json.batch_dim, 0);
        return $;
    }
};

circle.MatrixDiagOptions = class MatrixDiagOptions {

    static decode(/* reader, position */) {
        const $ = new circle.MatrixDiagOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.MatrixDiagOptions();
        return $;
    }
};

circle.QuantizeOptions = class QuantizeOptions {

    static decode(/* reader, position */) {
        const $ = new circle.QuantizeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.QuantizeOptions();
        return $;
    }
};

circle.MatrixSetDiagOptions = class MatrixSetDiagOptions {

    static decode(/* reader, position */) {
        const $ = new circle.MatrixSetDiagOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.MatrixSetDiagOptions();
        return $;
    }
};

circle.IfOptions = class IfOptions {

    static decode(reader, position) {
        const $ = new circle.IfOptions();
        $.then_subgraph_index = reader.int32_(position, 4, 0);
        $.else_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.IfOptions();
        $.then_subgraph_index = reader.value(json.then_subgraph_index, 0);
        $.else_subgraph_index = reader.value(json.else_subgraph_index, 0);
        return $;
    }
};

circle.CallOnceOptions = class CallOnceOptions {

    static decode(reader, position) {
        const $ = new circle.CallOnceOptions();
        $.init_subgraph_index = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.CallOnceOptions();
        $.init_subgraph_index = reader.value(json.init_subgraph_index, 0);
        return $;
    }
};

circle.WhileOptions = class WhileOptions {

    static decode(reader, position) {
        const $ = new circle.WhileOptions();
        $.cond_subgraph_index = reader.int32_(position, 4, 0);
        $.body_subgraph_index = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.WhileOptions();
        $.cond_subgraph_index = reader.value(json.cond_subgraph_index, 0);
        $.body_subgraph_index = reader.value(json.body_subgraph_index, 0);
        return $;
    }
};

circle.NonMaxSuppressionV4Options = class NonMaxSuppressionV4Options {

    static decode(/* reader, position */) {
        const $ = new circle.NonMaxSuppressionV4Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.NonMaxSuppressionV4Options();
        return $;
    }
};

circle.NonMaxSuppressionV5Options = class NonMaxSuppressionV5Options {

    static decode(/* reader, position */) {
        const $ = new circle.NonMaxSuppressionV5Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.NonMaxSuppressionV5Options();
        return $;
    }
};

circle.ScatterNdOptions = class ScatterNdOptions {

    static decode(/* reader, position */) {
        const $ = new circle.ScatterNdOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ScatterNdOptions();
        return $;
    }
};

circle.SelectV2Options = class SelectV2Options {

    static decode(/* reader, position */) {
        const $ = new circle.SelectV2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SelectV2Options();
        return $;
    }
};

circle.DensifyOptions = class DensifyOptions {

    static decode(/* reader, position */) {
        const $ = new circle.DensifyOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.DensifyOptions();
        return $;
    }
};

circle.SegmentSumOptions = class SegmentSumOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SegmentSumOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SegmentSumOptions();
        return $;
    }
};

circle.BatchMatMulOptions = class BatchMatMulOptions {

    static decode(reader, position) {
        const $ = new circle.BatchMatMulOptions();
        $.adjoint_lhs = reader.bool_(position, 4, false);
        $.adjoint_rhs = reader.bool_(position, 6, false);
        $.asymmetric_quantize_inputs = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.BatchMatMulOptions();
        $.adjoint_lhs = reader.value(json.adjoint_lhs, false);
        $.adjoint_rhs = reader.value(json.adjoint_rhs, false);
        $.asymmetric_quantize_inputs = reader.value(json.asymmetric_quantize_inputs, false);
        return $;
    }
};

circle.CumsumOptions = class CumsumOptions {

    static decode(reader, position) {
        const $ = new circle.CumsumOptions();
        $.exclusive = reader.bool_(position, 4, false);
        $.reverse = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.CumsumOptions();
        $.exclusive = reader.value(json.exclusive, false);
        $.reverse = reader.value(json.reverse, false);
        return $;
    }
};

circle.BroadcastToOptions = class BroadcastToOptions {

    static decode(/* reader, position */) {
        const $ = new circle.BroadcastToOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.BroadcastToOptions();
        return $;
    }
};

circle.Rfft2dOptions = class Rfft2dOptions {

    static decode(/* reader, position */) {
        const $ = new circle.Rfft2dOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.Rfft2dOptions();
        return $;
    }
};

circle.HashtableOptions = class HashtableOptions {

    static decode(reader, position) {
        const $ = new circle.HashtableOptions();
        $.table_id = reader.int32_(position, 4, 0);
        $.key_dtype = reader.int8_(position, 6, 0);
        $.value_dtype = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.HashtableOptions();
        $.table_id = reader.value(json.table_id, 0);
        $.key_dtype = circle.TensorType[json.key_dtype];
        $.value_dtype = circle.TensorType[json.value_dtype];
        return $;
    }
};

circle.HashtableFindOptions = class HashtableFindOptions {

    static decode(/* reader, position */) {
        const $ = new circle.HashtableFindOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.HashtableFindOptions();
        return $;
    }
};

circle.HashtableImportOptions = class HashtableImportOptions {

    static decode(/* reader, position */) {
        const $ = new circle.HashtableImportOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.HashtableImportOptions();
        return $;
    }
};

circle.HashtableSizeOptions = class HashtableSizeOptions {

    static decode(/* reader, position */) {
        const $ = new circle.HashtableSizeOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.HashtableSizeOptions();
        return $;
    }
};

circle.VarHandleOptions = class VarHandleOptions {

    static decode(reader, position) {
        const $ = new circle.VarHandleOptions();
        $.container = reader.string_(position, 4, null);
        $.shared_name = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.VarHandleOptions();
        $.container = reader.value(json.container, null);
        $.shared_name = reader.value(json.shared_name, null);
        return $;
    }
};

circle.ReadVariableOptions = class ReadVariableOptions {

    static decode(/* reader, position */) {
        const $ = new circle.ReadVariableOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ReadVariableOptions();
        return $;
    }
};

circle.AssignVariableOptions = class AssignVariableOptions {

    static decode(/* reader, position */) {
        const $ = new circle.AssignVariableOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.AssignVariableOptions();
        return $;
    }
};

circle.RandomOptions = class RandomOptions {

    static decode(reader, position) {
        const $ = new circle.RandomOptions();
        $.seed = reader.int64_(position, 4, 0n);
        $.seed2 = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.RandomOptions();
        $.seed = reader.int64(json.seed, 0n);
        $.seed2 = reader.int64(json.seed2, 0n);
        return $;
    }
};

circle.BucketizeOptions = class BucketizeOptions {

    static decode(reader, position) {
        const $ = new circle.BucketizeOptions();
        $.boundaries = reader.array(position, 4, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.BucketizeOptions();
        $.boundaries = reader.array(json.boundaries, Float32Array);
        return $;
    }
};

circle.GeluOptions = class GeluOptions {

    static decode(reader, position) {
        const $ = new circle.GeluOptions();
        $.approximate = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.GeluOptions();
        $.approximate = reader.value(json.approximate, false);
        return $;
    }
};

circle.DynamicUpdateSliceOptions = class DynamicUpdateSliceOptions {

    static decode(/* reader, position */) {
        const $ = new circle.DynamicUpdateSliceOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.DynamicUpdateSliceOptions();
        return $;
    }
};

circle.UnsortedSegmentProdOptions = class UnsortedSegmentProdOptions {

    static decode(/* reader, position */) {
        const $ = new circle.UnsortedSegmentProdOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.UnsortedSegmentProdOptions();
        return $;
    }
};

circle.UnsortedSegmentMaxOptions = class UnsortedSegmentMaxOptions {

    static decode(/* reader, position */) {
        const $ = new circle.UnsortedSegmentMaxOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.UnsortedSegmentMaxOptions();
        return $;
    }
};

circle.UnsortedSegmentSumOptions = class UnsortedSegmentSumOptions {

    static decode(/* reader, position */) {
        const $ = new circle.UnsortedSegmentSumOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.UnsortedSegmentSumOptions();
        return $;
    }
};

circle.ATan2Options = class ATan2Options {

    static decode(/* reader, position */) {
        const $ = new circle.ATan2Options();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.ATan2Options();
        return $;
    }
};

circle.UnsortedSegmentMinOptions = class UnsortedSegmentMinOptions {

    static decode(/* reader, position */) {
        const $ = new circle.UnsortedSegmentMinOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.UnsortedSegmentMinOptions();
        return $;
    }
};

circle.SignOptions = class SignOptions {

    static decode(/* reader, position */) {
        const $ = new circle.SignOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.SignOptions();
        return $;
    }
};

circle.BitcastOptions = class BitcastOptions {

    static decode(/* reader, position */) {
        const $ = new circle.BitcastOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.BitcastOptions();
        return $;
    }
};

circle.BitwiseXorOptions = class BitwiseXorOptions {

    static decode(/* reader, position */) {
        const $ = new circle.BitwiseXorOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.BitwiseXorOptions();
        return $;
    }
};

circle.RightShiftOptions = class RightShiftOptions {

    static decode(/* reader, position */) {
        const $ = new circle.RightShiftOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.RightShiftOptions();
        return $;
    }
};

circle.DilateOptions = class DilateOptions {

    static decode(/* reader, position */) {
        const $ = new circle.DilateOptions();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new circle.DilateOptions();
        return $;
    }
};

circle.ReduceWindowFunction = {
    UNSUPPORTED: 0,
    ADD: 1,
    MUL: 2,
    MINIMUM: 3,
    MAXIMUM: 4,
    ALL: 5,
    ANY: 6
};

circle.ReduceWindowOptions = class ReduceWindowOptions {

    static decode(reader, position) {
        const $ = new circle.ReduceWindowOptions();
        $.reduce_function = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.ReduceWindowOptions();
        $.reduce_function = circle.ReduceWindowFunction[json.reduce_function];
        return $;
    }
};

circle.GRUOptions = class GRUOptions {

    static decode(reader, position) {
        const $ = new circle.GRUOptions();
        $.fused_activation_function = reader.int8_(position, 4, 0);
        $.return_sequences = reader.bool_(position, 6, false);
        $.time_major = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.GRUOptions();
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        $.return_sequences = reader.value(json.return_sequences, false);
        $.time_major = reader.value(json.time_major, false);
        return $;
    }
};

circle.BCQGatherOptions = class BCQGatherOptions {

    static decode(reader, position) {
        const $ = new circle.BCQGatherOptions();
        $.input_hidden_size = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.BCQGatherOptions();
        $.input_hidden_size = reader.value(json.input_hidden_size, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

circle.BCQFullyConnectedOptions = class BCQFullyConnectedOptions {

    static decode(reader, position) {
        const $ = new circle.BCQFullyConnectedOptions();
        $.weights_hidden_size = reader.int32_(position, 4, 0);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.BCQFullyConnectedOptions();
        $.weights_hidden_size = reader.value(json.weights_hidden_size, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.InstanceNormOptions = class InstanceNormOptions {

    static decode(reader, position) {
        const $ = new circle.InstanceNormOptions();
        $.epsilon = reader.float32_(position, 4, 0);
        $.fused_activation_function = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.InstanceNormOptions();
        $.epsilon = reader.value(json.epsilon, 0);
        $.fused_activation_function = circle.ActivationFunctionType[json.fused_activation_function];
        return $;
    }
};

circle.OperatorCode = class OperatorCode {

    static decode(reader, position) {
        const $ = new circle.OperatorCode();
        $.deprecated_builtin_code = reader.int8_(position, 4, 0);
        $.custom_code = reader.string_(position, 6, null);
        $.version = reader.int32_(position, 8, 1);
        $.builtin_code = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.OperatorCode();
        $.deprecated_builtin_code = reader.value(json.deprecated_builtin_code, 0);
        $.custom_code = reader.value(json.custom_code, null);
        $.version = reader.value(json.version, 1);
        $.builtin_code = circle.BuiltinOperator[json.builtin_code];
        return $;
    }
};

circle.CustomOptionsFormat = {
    FLEXBUFFERS: 0
};

circle.DataFormat = {
    CHANNELS_LAST: 0,
    CHANNELS_FIRST: 1
};

circle.Operator = class Operator {

    static decode(reader, position) {
        const $ = new circle.Operator();
        $.opcode_index = reader.uint32_(position, 4, 0);
        $.inputs = reader.array(position, 6, Int32Array);
        $.outputs = reader.array(position, 8, Int32Array);
        $.builtin_options = reader.union(position, 10, circle.BuiltinOptions);
        $.custom_options = reader.array(position, 14, Uint8Array);
        $.custom_options_format = reader.int8_(position, 16, 0);
        $.mutating_variable_inputs = reader.bools_(position, 18);
        $.intermediates = reader.array(position, 20, Int32Array);
        $.large_custom_options_offset = reader.uint64_(position, 22, 0n);
        $.large_custom_options_size = reader.uint64_(position, 24, 0n);
        $.builtin_options_2 = reader.union(position, 26, circle.BuiltinOptions2);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Operator();
        $.opcode_index = reader.value(json.opcode_index, 0);
        $.inputs = reader.array(json.inputs, Int32Array);
        $.outputs = reader.array(json.outputs, Int32Array);
        $.builtin_options = circle.BuiltinOptions.decodeText(reader, json.builtin_options, json.builtin_options_type);
        $.custom_options = reader.array(json.custom_options, Uint8Array);
        $.custom_options_format = circle.CustomOptionsFormat[json.custom_options_format];
        $.mutating_variable_inputs = reader.array(json.mutating_variable_inputs);
        $.intermediates = reader.array(json.intermediates, Int32Array);
        $.large_custom_options_offset = reader.uint64(json.large_custom_options_offset, 0n);
        $.large_custom_options_size = reader.uint64(json.large_custom_options_size, 0n);
        $.builtin_options_2 = circle.BuiltinOptions2.decodeText(reader, json.builtin_options_2, json.builtin_options_2_type);
        return $;
    }
};

circle.SubGraph = class SubGraph {

    static decode(reader, position) {
        const $ = new circle.SubGraph();
        $.tensors = reader.tables(position, 4, circle.Tensor);
        $.inputs = reader.array(position, 6, Int32Array);
        $.outputs = reader.array(position, 8, Int32Array);
        $.operators = reader.tables(position, 10, circle.Operator);
        $.name = reader.string_(position, 12, null);
        $.deprecated_data_format = reader.int8_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SubGraph();
        $.tensors = reader.objects(json.tensors, circle.Tensor);
        $.inputs = reader.array(json.inputs, Int32Array);
        $.outputs = reader.array(json.outputs, Int32Array);
        $.operators = reader.objects(json.operators, circle.Operator);
        $.name = reader.value(json.name, null);
        $.deprecated_data_format = circle.DataFormat[json.deprecated_data_format];
        return $;
    }
};

circle.Buffer = class Buffer {

    static decode(reader, position) {
        const $ = new circle.Buffer();
        $.data = reader.array(position, 4, Uint8Array);
        $.offset = reader.uint64_(position, 6, 0n);
        $.size = reader.uint64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Buffer();
        $.data = reader.array(json.data, Uint8Array);
        $.offset = reader.uint64(json.offset, 0n);
        $.size = reader.uint64(json.size, 0n);
        return $;
    }
};

circle.Metadata = class Metadata {

    static decode(reader, position) {
        const $ = new circle.Metadata();
        $.name = reader.string_(position, 4, null);
        $.buffer = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Metadata();
        $.name = reader.value(json.name, null);
        $.buffer = reader.value(json.buffer, 0);
        return $;
    }
};

circle.TensorMap = class TensorMap {

    static decode(reader, position) {
        const $ = new circle.TensorMap();
        $.name = reader.string_(position, 4, null);
        $.tensor_index = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.TensorMap();
        $.name = reader.value(json.name, null);
        $.tensor_index = reader.value(json.tensor_index, 0);
        return $;
    }
};

circle.SignatureDef = class SignatureDef {

    static decode(reader, position) {
        const $ = new circle.SignatureDef();
        $.inputs = reader.tables(position, 4, circle.TensorMap);
        $.outputs = reader.tables(position, 6, circle.TensorMap);
        $.signature_key = reader.string_(position, 8, null);
        $.deprecated_tag = reader.string_(position, 10, null);
        $.subgraph_index = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.SignatureDef();
        $.inputs = reader.objects(json.inputs, circle.TensorMap);
        $.outputs = reader.objects(json.outputs, circle.TensorMap);
        $.signature_key = reader.value(json.signature_key, null);
        $.deprecated_tag = reader.value(json.deprecated_tag, null);
        $.subgraph_index = reader.value(json.subgraph_index, 0);
        return $;
    }
};

circle.Model = class Model {

    static identifier(reader) {
        return reader.identifier === 'CIR0';
    }

    static create(reader) {
        return circle.Model.decode(reader, reader.root);
    }

    static createText(reader) {
        return circle.Model.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new circle.Model();
        $.version = reader.uint32_(position, 4, 0);
        $.operator_codes = reader.tables(position, 6, circle.OperatorCode);
        $.subgraphs = reader.tables(position, 8, circle.SubGraph);
        $.description = reader.string_(position, 10, null);
        $.buffers = reader.tables(position, 12, circle.Buffer);
        $.metadata_buffer = reader.array(position, 14, Int32Array);
        $.metadata = reader.tables(position, 16, circle.Metadata);
        $.signature_defs = reader.tables(position, 18, circle.SignatureDef);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new circle.Model();
        $.version = reader.value(json.version, 0);
        $.operator_codes = reader.objects(json.operator_codes, circle.OperatorCode);
        $.subgraphs = reader.objects(json.subgraphs, circle.SubGraph);
        $.description = reader.value(json.description, null);
        $.buffers = reader.objects(json.buffers, circle.Buffer);
        $.metadata_buffer = reader.array(json.metadata_buffer, Int32Array);
        $.metadata = reader.objects(json.metadata, circle.Metadata);
        $.signature_defs = reader.objects(json.signature_defs, circle.SignatureDef);
        return $;
    }
};
