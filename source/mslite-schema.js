var $root = flatbuffers.get('mslite');

$root.mindspore = $root.mindspore || {};

$root.mindspore.schema = $root.mindspore.schema || {};

$root.mindspore.schema.ResizeMethod = {
    UNKNOWN: -1,
    LINEAR: 0,
    NEAREST: 1,
    CUBIC: 2
};

$root.mindspore.schema.CoordinateTransformMode = {
    ASYMMETRIC: 0,
    ALIGN_CORNERS: 1,
    HALF_PIXEL: 2
};

$root.mindspore.schema.NearestMode = {
    NORMAL: 0,
    ROUND_HALF_DOWN: 1,
    ROUND_HALF_UP: 2,
    FLOOR: 3,
    CEIL: 4
};

$root.mindspore.schema.Format = {
    NCHW: 0,
    NHWC: 1,
    NHWC4: 2,
    HWKC: 3,
    HWCK: 4,
    KCHW: 5,
    CKHW: 6,
    KHWC: 7,
    CHWK: 8,
    HW: 9,
    HW4: 10,
    NC: 11,
    NC4: 12,
    NC4HW4: 13,
    NUM_OF_FORMAT: 14,
    NCDHW: 15,
    NWC: 16,
    NCW: 17,
    NC8HW8: 18
};

$root.mindspore.schema.ActivationType = {
    NO_ACTIVATION: 0,
    RELU: 1,
    SIGMOID: 2,
    RELU6: 3,
    ELU: 4,
    LEAKY_RELU: 5,
    ABS: 6,
    RELU1: 7,
    SOFTSIGN: 8,
    SOFTPLUS: 9,
    TANH: 10,
    SELU: 11,
    HSWISH: 12,
    HSIGMOID: 13,
    THRESHOLDRELU: 14,
    LINEAR: 15,
    HARD_TANH: 16,
    SIGN: 17,
    SWISH: 18,
    GELU: 19,
    UNKNOWN: 20
};

$root.mindspore.schema.ReduceMode = {
    ReduceMean: 0,
    ReduceMax: 1,
    ReduceMin: 2,
    ReduceProd: 3,
    ReduceSum: 4,
    ReduceSumSquare: 5,
    ReduceASum: 6,
    ReduceAll: 7,
    ReduceL2: 8
};

$root.mindspore.schema.PoolMode = {
    MAX_POOLING: 0,
    MEAN_POOLING: 1
};

$root.mindspore.schema.EltwiseMode = {
    PROD: 0,
    SUM: 1,
    MAXIMUM: 2,
    UNKNOWN: 3
};

$root.mindspore.schema.PadMode = {
    PAD: 0,
    SAME: 1,
    VALID: 2
};

$root.mindspore.schema.RoundMode = {
    FLOOR: 0,
    CEIL: 1
};

$root.mindspore.schema.PaddingMode = {
    CONSTANT: 0,
    REFLECT: 1,
    SYMMETRIC: 2,
    MODE_RESERVED: 3
};

$root.mindspore.schema.LshProjectionType = {
    UNKNOWN: 0,
    SPARSE: 1,
    DENSE: 2
};

$root.mindspore.schema.Reduction = {
    REDUCTION_SUM: 0,
    MEAN: 1,
    NONE: 2
};

$root.mindspore.schema.Vec = class Vec {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Vec();
        $.data = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Vec();
        $.data = reader.array(json.data);
        return $;
    }
};

$root.mindspore.schema.Vec2D = class Vec2D {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Vec2D();
        $.data = reader.tableArray(position, 4, $root.mindspore.schema.Vec.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Vec2D();
        $.data = reader.objectArray(json.data, $root.mindspore.schema.Vec.decodeText);
        return $;
    }
};

$root.mindspore.schema.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Attribute();
        $.name = reader.string_(position, 4, null);
        $.data = reader.typedArray(position, 6, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Attribute();
        $.name = reader.value(json.name, null);
        $.data = reader.typedArray(json.data, Uint8Array);
        return $;
    }
};

$root.mindspore.schema.PrimitiveType = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.mindspore.schema.Abs.decode(reader, position);
            case 2: return $root.mindspore.schema.Activation.decode(reader, position);
            case 3: return $root.mindspore.schema.ActivationGrad.decode(reader, position);
            case 4: return $root.mindspore.schema.Adam.decode(reader, position);
            case 5: return $root.mindspore.schema.AddFusion.decode(reader, position);
            case 6: return $root.mindspore.schema.AdderFusion.decode(reader, position);
            case 7: return $root.mindspore.schema.AddGrad.decode(reader, position);
            case 8: return $root.mindspore.schema.AddN.decode(reader, position);
            case 9: return $root.mindspore.schema.All.decode(reader, position);
            case 10: return $root.mindspore.schema.ApplyMomentum.decode(reader, position);
            case 11: return $root.mindspore.schema.ArgMaxFusion.decode(reader, position);
            case 12: return $root.mindspore.schema.ArgMinFusion.decode(reader, position);
            case 13: return $root.mindspore.schema.Assert.decode(reader, position);
            case 14: return $root.mindspore.schema.Assign.decode(reader, position);
            case 15: return $root.mindspore.schema.AssignAdd.decode(reader, position);
            case 16: return $root.mindspore.schema.AudioSpectrogram.decode(reader, position);
            case 17: return $root.mindspore.schema.AvgPoolFusion.decode(reader, position);
            case 18: return $root.mindspore.schema.AvgPoolGrad.decode(reader, position);
            case 19: return $root.mindspore.schema.BatchNorm.decode(reader, position);
            case 20: return $root.mindspore.schema.BatchNormGrad.decode(reader, position);
            case 21: return $root.mindspore.schema.BatchToSpace.decode(reader, position);
            case 22: return $root.mindspore.schema.BatchToSpaceND.decode(reader, position);
            case 23: return $root.mindspore.schema.BiasAdd.decode(reader, position);
            case 24: return $root.mindspore.schema.BinaryCrossEntropy.decode(reader, position);
            case 25: return $root.mindspore.schema.BinaryCrossEntropyGrad.decode(reader, position);
            case 26: return $root.mindspore.schema.BiasAddGrad.decode(reader, position);
            case 27: return $root.mindspore.schema.BroadcastTo.decode(reader, position);
            case 28: return $root.mindspore.schema.Cast.decode(reader, position);
            case 29: return $root.mindspore.schema.Ceil.decode(reader, position);
            case 30: return $root.mindspore.schema.Clip.decode(reader, position);
            case 31: return $root.mindspore.schema.Concat.decode(reader, position);
            case 32: return $root.mindspore.schema.Attention.decode(reader, position);
            case 33: return $root.mindspore.schema.Conv2DBackpropFilterFusion.decode(reader, position);
            case 34: return $root.mindspore.schema.Conv2DBackpropInputFusion.decode(reader, position);
            case 35: return $root.mindspore.schema.Conv2DFusion.decode(reader, position);
            case 36: return $root.mindspore.schema.Conv2dTransposeFusion.decode(reader, position);
            case 37: return $root.mindspore.schema.Cos.decode(reader, position);
            case 38: return $root.mindspore.schema.ConstantOfShape.decode(reader, position);
            case 39: return $root.mindspore.schema.Crop.decode(reader, position);
            case 40: return $root.mindspore.schema.CustomExtractFeatures.decode(reader, position);
            case 41: return $root.mindspore.schema.CustomNormalize.decode(reader, position);
            case 42: return $root.mindspore.schema.CustomPredict.decode(reader, position);
            case 43: return $root.mindspore.schema.DeConv2DGradFilter.decode(reader, position);
            case 44: return $root.mindspore.schema.Depend.decode(reader, position);
            case 45: return $root.mindspore.schema.DepthToSpace.decode(reader, position);
            case 46: return $root.mindspore.schema.DetectionPostProcess.decode(reader, position);
            case 47: return $root.mindspore.schema.DivFusion.decode(reader, position);
            case 48: return $root.mindspore.schema.DivGrad.decode(reader, position);
            case 49: return $root.mindspore.schema.Dropout.decode(reader, position);
            case 50: return $root.mindspore.schema.DropoutGrad.decode(reader, position);
            case 51: return $root.mindspore.schema.Elu.decode(reader, position);
            case 52: return $root.mindspore.schema.Eltwise.decode(reader, position);
            case 53: return $root.mindspore.schema.Equal.decode(reader, position);
            case 54: return $root.mindspore.schema.EmbeddingLookupFusion.decode(reader, position);
            case 55: return $root.mindspore.schema.ExpFusion.decode(reader, position);
            case 56: return $root.mindspore.schema.ExpandDims.decode(reader, position);
            case 57: return $root.mindspore.schema.FakeQuantWithMinMaxVars.decode(reader, position);
            case 58: return $root.mindspore.schema.FakeQuantWithMinMaxVarsPerChannel.decode(reader, position);
            case 59: return $root.mindspore.schema.FftReal.decode(reader, position);
            case 60: return $root.mindspore.schema.FftImag.decode(reader, position);
            case 61: return $root.mindspore.schema.Flatten.decode(reader, position);
            case 62: return $root.mindspore.schema.FlattenGrad.decode(reader, position);
            case 63: return $root.mindspore.schema.Floor.decode(reader, position);
            case 64: return $root.mindspore.schema.FloorDiv.decode(reader, position);
            case 65: return $root.mindspore.schema.FloorMod.decode(reader, position);
            case 66: return $root.mindspore.schema.Fill.decode(reader, position);
            case 67: return $root.mindspore.schema.FullConnection.decode(reader, position);
            case 68: return $root.mindspore.schema.FusedBatchNorm.decode(reader, position);
            case 69: return $root.mindspore.schema.Gather.decode(reader, position);
            case 70: return $root.mindspore.schema.GatherNd.decode(reader, position);
            case 71: return $root.mindspore.schema.Greater.decode(reader, position);
            case 72: return $root.mindspore.schema.GreaterEqual.decode(reader, position);
            case 73: return $root.mindspore.schema.HashtableLookup.decode(reader, position);
            case 74: return $root.mindspore.schema.InstanceNorm.decode(reader, position);
            case 75: return $root.mindspore.schema.LayerNormFusion.decode(reader, position);
            case 76: return $root.mindspore.schema.LeakyRelu.decode(reader, position);
            case 77: return $root.mindspore.schema.Less.decode(reader, position);
            case 78: return $root.mindspore.schema.LessEqual.decode(reader, position);
            case 79: return $root.mindspore.schema.Log.decode(reader, position);
            case 80: return $root.mindspore.schema.LogGrad.decode(reader, position);
            case 81: return $root.mindspore.schema.LogicalAnd.decode(reader, position);
            case 82: return $root.mindspore.schema.LogicalNot.decode(reader, position);
            case 83: return $root.mindspore.schema.LogicalOr.decode(reader, position);
            case 84: return $root.mindspore.schema.LpNormalization.decode(reader, position);
            case 85: return $root.mindspore.schema.LRN.decode(reader, position);
            case 86: return $root.mindspore.schema.LshProjection.decode(reader, position);
            case 87: return $root.mindspore.schema.LSTM.decode(reader, position);
            case 88: return $root.mindspore.schema.L2NormalizeFusion.decode(reader, position);
            case 89: return $root.mindspore.schema.MatMulFusion.decode(reader, position);
            case 90: return $root.mindspore.schema.Maximum.decode(reader, position);
            case 91: return $root.mindspore.schema.MaximumGrad.decode(reader, position);
            case 92: return $root.mindspore.schema.MaxPoolFusion.decode(reader, position);
            case 93: return $root.mindspore.schema.MaxPoolGrad.decode(reader, position);
            case 94: return $root.mindspore.schema.SwitchLayer.decode(reader, position);
            case 95: return $root.mindspore.schema.Mfcc.decode(reader, position);
            case 96: return $root.mindspore.schema.Minimum.decode(reader, position);
            case 97: return $root.mindspore.schema.MinimumGrad.decode(reader, position);
            case 98: return $root.mindspore.schema.Mod.decode(reader, position);
            case 99: return $root.mindspore.schema.MulFusion.decode(reader, position);
            case 100: return $root.mindspore.schema.MulGrad.decode(reader, position);
            case 101: return $root.mindspore.schema.Neg.decode(reader, position);
            case 102: return $root.mindspore.schema.NegGrad.decode(reader, position);
            case 103: return $root.mindspore.schema.NotEqual.decode(reader, position);
            case 104: return $root.mindspore.schema.NonMaxSuppression.decode(reader, position);
            case 105: return $root.mindspore.schema.OneHot.decode(reader, position);
            case 106: return $root.mindspore.schema.OnesLike.decode(reader, position);
            case 107: return $root.mindspore.schema.PadFusion.decode(reader, position);
            case 108: return $root.mindspore.schema.PartialFusion.decode(reader, position);
            case 109: return $root.mindspore.schema.PowerGrad.decode(reader, position);
            case 110: return $root.mindspore.schema.PowFusion.decode(reader, position);
            case 111: return $root.mindspore.schema.PriorBox.decode(reader, position);
            case 112: return $root.mindspore.schema.PReLUFusion.decode(reader, position);
            case 113: return $root.mindspore.schema.QuantDTypeCast.decode(reader, position);
            case 114: return $root.mindspore.schema.Rank.decode(reader, position);
            case 115: return $root.mindspore.schema.Range.decode(reader, position);
            case 116: return $root.mindspore.schema.Reciprocal.decode(reader, position);
            case 117: return $root.mindspore.schema.RealDiv.decode(reader, position);
            case 118: return $root.mindspore.schema.ReduceFusion.decode(reader, position);
            case 119: return $root.mindspore.schema.Reshape.decode(reader, position);
            case 120: return $root.mindspore.schema.Resize.decode(reader, position);
            case 121: return $root.mindspore.schema.ReverseSequence.decode(reader, position);
            case 122: return $root.mindspore.schema.ReverseV2.decode(reader, position);
            case 123: return $root.mindspore.schema.Rfft.decode(reader, position);
            case 124: return $root.mindspore.schema.ROIPooling.decode(reader, position);
            case 125: return $root.mindspore.schema.Round.decode(reader, position);
            case 126: return $root.mindspore.schema.Rsqrt.decode(reader, position);
            case 127: return $root.mindspore.schema.ScaleFusion.decode(reader, position);
            case 128: return $root.mindspore.schema.ScatterNd.decode(reader, position);
            case 129: return $root.mindspore.schema.SGD.decode(reader, position);
            case 130: return $root.mindspore.schema.Shape.decode(reader, position);
            case 131: return $root.mindspore.schema.SigmoidCrossEntropyWithLogits.decode(reader, position);
            case 132: return $root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad.decode(reader, position);
            case 133: return $root.mindspore.schema.Sin.decode(reader, position);
            case 134: return $root.mindspore.schema.SkipGram.decode(reader, position);
            case 135: return $root.mindspore.schema.SliceFusion.decode(reader, position);
            case 136: return $root.mindspore.schema.SmoothL1Loss.decode(reader, position);
            case 137: return $root.mindspore.schema.SmoothL1LossGrad.decode(reader, position);
            case 138: return $root.mindspore.schema.Softmax.decode(reader, position);
            case 139: return $root.mindspore.schema.SoftmaxCrossEntropyWithLogits.decode(reader, position);
            case 140: return $root.mindspore.schema.SpaceToBatch.decode(reader, position);
            case 141: return $root.mindspore.schema.SpaceToBatchND.decode(reader, position);
            case 142: return $root.mindspore.schema.SpaceToDepth.decode(reader, position);
            case 143: return $root.mindspore.schema.SparseSoftmaxCrossEntropyWithLogits.decode(reader, position);
            case 144: return $root.mindspore.schema.SparseToDense.decode(reader, position);
            case 145: return $root.mindspore.schema.Split.decode(reader, position);
            case 146: return $root.mindspore.schema.Sqrt.decode(reader, position);
            case 147: return $root.mindspore.schema.Squeeze.decode(reader, position);
            case 148: return $root.mindspore.schema.Square.decode(reader, position);
            case 149: return $root.mindspore.schema.SquaredDifference.decode(reader, position);
            case 150: return $root.mindspore.schema.Stack.decode(reader, position);
            case 151: return $root.mindspore.schema.StridedSlice.decode(reader, position);
            case 152: return $root.mindspore.schema.SubFusion.decode(reader, position);
            case 153: return $root.mindspore.schema.SubGrad.decode(reader, position);
            case 154: return $root.mindspore.schema.Switch.decode(reader, position);
            case 155: return $root.mindspore.schema.TensorListFromTensor.decode(reader, position);
            case 156: return $root.mindspore.schema.TensorListGetItem.decode(reader, position);
            case 157: return $root.mindspore.schema.TensorListReserve.decode(reader, position);
            case 158: return $root.mindspore.schema.TensorListSetItem.decode(reader, position);
            case 159: return $root.mindspore.schema.TensorListStack.decode(reader, position);
            case 160: return $root.mindspore.schema.TileFusion.decode(reader, position);
            case 161: return $root.mindspore.schema.TopKFusion.decode(reader, position);
            case 162: return $root.mindspore.schema.Transpose.decode(reader, position);
            case 163: return $root.mindspore.schema.Unique.decode(reader, position);
            case 164: return $root.mindspore.schema.UnsortedSegmentSum.decode(reader, position);
            case 165: return $root.mindspore.schema.Unsqueeze.decode(reader, position);
            case 166: return $root.mindspore.schema.Unstack.decode(reader, position);
            case 167: return $root.mindspore.schema.LSTMGrad.decode(reader, position);
            case 168: return $root.mindspore.schema.Where.decode(reader, position);
            case 169: return $root.mindspore.schema.ZerosLike.decode(reader, position);
            case 170: return $root.mindspore.schema.Select.decode(reader, position);
            case 171: return $root.mindspore.schema.ScatterNdUpdate.decode(reader, position);
            case 172: return $root.mindspore.schema.GRU.decode(reader, position);
            case 173: return $root.mindspore.schema.NonZero.decode(reader, position);
            case 174: return $root.mindspore.schema.InvertPermutation.decode(reader, position);
            case 175: return $root.mindspore.schema.Size.decode(reader, position);
            case 176: return $root.mindspore.schema.RandomStandardNormal.decode(reader, position);
            case 177: return $root.mindspore.schema.CropAndResize.decode(reader, position);
            case 178: return $root.mindspore.schema.Erf.decode(reader, position);
            case 179: return $root.mindspore.schema.StridedSliceGrad.decode(reader, position);
            case 180: return $root.mindspore.schema.IsFinite.decode(reader, position);
            case 181: return $root.mindspore.schema.LinSpace.decode(reader, position);
            case 182: return $root.mindspore.schema.UniformReal.decode(reader, position);
            case 183: return $root.mindspore.schema.AbsGrad.decode(reader, position);
            case 184: return $root.mindspore.schema.RsqrtGrad.decode(reader, position);
            case 185: return $root.mindspore.schema.SqrtGrad.decode(reader, position);
            case 186: return $root.mindspore.schema.LayerNormGrad.decode(reader, position);
            case 187: return $root.mindspore.schema.ResizeGrad.decode(reader, position);
            case 188: return $root.mindspore.schema.Splice.decode(reader, position);
            case 189: return $root.mindspore.schema.LogSoftmax.decode(reader, position);
            case 190: return $root.mindspore.schema.Call.decode(reader, position);
            case 191: return $root.mindspore.schema.Custom.decode(reader, position);
            case 192: return $root.mindspore.schema.CumSum.decode(reader, position);
            case 193: return $root.mindspore.schema.SplitWithOverlap.decode(reader, position);
            case 194: return $root.mindspore.schema.GenOP.decode(reader, position);
            case 195: return $root.mindspore.schema.RaggedRange.decode(reader, position);
            case 196: return $root.mindspore.schema.GLU.decode(reader, position);
            case 197: return $root.mindspore.schema.TensorArray.decode(reader, position);
            case 198: return $root.mindspore.schema.TensorArrayRead.decode(reader, position);
            case 199: return $root.mindspore.schema.TensorArrayWrite.decode(reader, position);
            case 200: return $root.mindspore.schema.Affine.decode(reader, position);
            case 201: return $root.mindspore.schema.AllGather.decode(reader, position);
            case 202: return $root.mindspore.schema.ReduceScatter.decode(reader, position);
            case 203: return $root.mindspore.schema.DynamicQuant.decode(reader, position);
            case 204: return $root.mindspore.schema.LSTMGradData.decode(reader, position);
            case 205: return $root.mindspore.schema.LSTMGradWeight.decode(reader, position);
            case 206: return $root.mindspore.schema.RandomNormal.decode(reader, position);
            case 207: return $root.mindspore.schema.NLLLoss.decode(reader, position);
            case 208: return $root.mindspore.schema.NLLLossGrad.decode(reader, position);
            case 209: return $root.mindspore.schema.FormatTranspose.decode(reader, position);
            case 210: return $root.mindspore.schema.GatherD.decode(reader, position);
            case 211: return $root.mindspore.schema.GroupNormFusion.decode(reader, position);
            case 212: return $root.mindspore.schema.Log1p.decode(reader, position);
            case 213: return $root.mindspore.schema.TensorScatterAdd.decode(reader, position);
            case 214: return $root.mindspore.schema.SparseFillEmptyRows.decode(reader, position);
            case 215: return $root.mindspore.schema.SparseReshape.decode(reader, position);
            case 216: return $root.mindspore.schema.SparseSegmentSum.decode(reader, position);
            case 217: return $root.mindspore.schema.ScatterElements.decode(reader, position);
            case 218: return $root.mindspore.schema.Triu.decode(reader, position);
            case 219: return $root.mindspore.schema.Tril.decode(reader, position);
            case 220: return $root.mindspore.schema.AdamWeightDecay.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Abs': return $root.mindspore.schema.Abs.decodeText(reader, json);
            case 'Activation': return $root.mindspore.schema.Activation.decodeText(reader, json);
            case 'ActivationGrad': return $root.mindspore.schema.ActivationGrad.decodeText(reader, json);
            case 'Adam': return $root.mindspore.schema.Adam.decodeText(reader, json);
            case 'AddFusion': return $root.mindspore.schema.AddFusion.decodeText(reader, json);
            case 'AdderFusion': return $root.mindspore.schema.AdderFusion.decodeText(reader, json);
            case 'AddGrad': return $root.mindspore.schema.AddGrad.decodeText(reader, json);
            case 'AddN': return $root.mindspore.schema.AddN.decodeText(reader, json);
            case 'All': return $root.mindspore.schema.All.decodeText(reader, json);
            case 'ApplyMomentum': return $root.mindspore.schema.ApplyMomentum.decodeText(reader, json);
            case 'ArgMaxFusion': return $root.mindspore.schema.ArgMaxFusion.decodeText(reader, json);
            case 'ArgMinFusion': return $root.mindspore.schema.ArgMinFusion.decodeText(reader, json);
            case 'Assert': return $root.mindspore.schema.Assert.decodeText(reader, json);
            case 'Assign': return $root.mindspore.schema.Assign.decodeText(reader, json);
            case 'AssignAdd': return $root.mindspore.schema.AssignAdd.decodeText(reader, json);
            case 'AudioSpectrogram': return $root.mindspore.schema.AudioSpectrogram.decodeText(reader, json);
            case 'AvgPoolFusion': return $root.mindspore.schema.AvgPoolFusion.decodeText(reader, json);
            case 'AvgPoolGrad': return $root.mindspore.schema.AvgPoolGrad.decodeText(reader, json);
            case 'BatchNorm': return $root.mindspore.schema.BatchNorm.decodeText(reader, json);
            case 'BatchNormGrad': return $root.mindspore.schema.BatchNormGrad.decodeText(reader, json);
            case 'BatchToSpace': return $root.mindspore.schema.BatchToSpace.decodeText(reader, json);
            case 'BatchToSpaceND': return $root.mindspore.schema.BatchToSpaceND.decodeText(reader, json);
            case 'BiasAdd': return $root.mindspore.schema.BiasAdd.decodeText(reader, json);
            case 'BinaryCrossEntropy': return $root.mindspore.schema.BinaryCrossEntropy.decodeText(reader, json);
            case 'BinaryCrossEntropyGrad': return $root.mindspore.schema.BinaryCrossEntropyGrad.decodeText(reader, json);
            case 'BiasAddGrad': return $root.mindspore.schema.BiasAddGrad.decodeText(reader, json);
            case 'BroadcastTo': return $root.mindspore.schema.BroadcastTo.decodeText(reader, json);
            case 'Cast': return $root.mindspore.schema.Cast.decodeText(reader, json);
            case 'Ceil': return $root.mindspore.schema.Ceil.decodeText(reader, json);
            case 'Clip': return $root.mindspore.schema.Clip.decodeText(reader, json);
            case 'Concat': return $root.mindspore.schema.Concat.decodeText(reader, json);
            case 'Attention': return $root.mindspore.schema.Attention.decodeText(reader, json);
            case 'Conv2DBackpropFilterFusion': return $root.mindspore.schema.Conv2DBackpropFilterFusion.decodeText(reader, json);
            case 'Conv2DBackpropInputFusion': return $root.mindspore.schema.Conv2DBackpropInputFusion.decodeText(reader, json);
            case 'Conv2DFusion': return $root.mindspore.schema.Conv2DFusion.decodeText(reader, json);
            case 'Conv2dTransposeFusion': return $root.mindspore.schema.Conv2dTransposeFusion.decodeText(reader, json);
            case 'Cos': return $root.mindspore.schema.Cos.decodeText(reader, json);
            case 'ConstantOfShape': return $root.mindspore.schema.ConstantOfShape.decodeText(reader, json);
            case 'Crop': return $root.mindspore.schema.Crop.decodeText(reader, json);
            case 'CustomExtractFeatures': return $root.mindspore.schema.CustomExtractFeatures.decodeText(reader, json);
            case 'CustomNormalize': return $root.mindspore.schema.CustomNormalize.decodeText(reader, json);
            case 'CustomPredict': return $root.mindspore.schema.CustomPredict.decodeText(reader, json);
            case 'DeConv2DGradFilter': return $root.mindspore.schema.DeConv2DGradFilter.decodeText(reader, json);
            case 'Depend': return $root.mindspore.schema.Depend.decodeText(reader, json);
            case 'DepthToSpace': return $root.mindspore.schema.DepthToSpace.decodeText(reader, json);
            case 'DetectionPostProcess': return $root.mindspore.schema.DetectionPostProcess.decodeText(reader, json);
            case 'DivFusion': return $root.mindspore.schema.DivFusion.decodeText(reader, json);
            case 'DivGrad': return $root.mindspore.schema.DivGrad.decodeText(reader, json);
            case 'Dropout': return $root.mindspore.schema.Dropout.decodeText(reader, json);
            case 'DropoutGrad': return $root.mindspore.schema.DropoutGrad.decodeText(reader, json);
            case 'Elu': return $root.mindspore.schema.Elu.decodeText(reader, json);
            case 'Eltwise': return $root.mindspore.schema.Eltwise.decodeText(reader, json);
            case 'Equal': return $root.mindspore.schema.Equal.decodeText(reader, json);
            case 'EmbeddingLookupFusion': return $root.mindspore.schema.EmbeddingLookupFusion.decodeText(reader, json);
            case 'ExpFusion': return $root.mindspore.schema.ExpFusion.decodeText(reader, json);
            case 'ExpandDims': return $root.mindspore.schema.ExpandDims.decodeText(reader, json);
            case 'FakeQuantWithMinMaxVars': return $root.mindspore.schema.FakeQuantWithMinMaxVars.decodeText(reader, json);
            case 'FakeQuantWithMinMaxVarsPerChannel': return $root.mindspore.schema.FakeQuantWithMinMaxVarsPerChannel.decodeText(reader, json);
            case 'FftReal': return $root.mindspore.schema.FftReal.decodeText(reader, json);
            case 'FftImag': return $root.mindspore.schema.FftImag.decodeText(reader, json);
            case 'Flatten': return $root.mindspore.schema.Flatten.decodeText(reader, json);
            case 'FlattenGrad': return $root.mindspore.schema.FlattenGrad.decodeText(reader, json);
            case 'Floor': return $root.mindspore.schema.Floor.decodeText(reader, json);
            case 'FloorDiv': return $root.mindspore.schema.FloorDiv.decodeText(reader, json);
            case 'FloorMod': return $root.mindspore.schema.FloorMod.decodeText(reader, json);
            case 'Fill': return $root.mindspore.schema.Fill.decodeText(reader, json);
            case 'FullConnection': return $root.mindspore.schema.FullConnection.decodeText(reader, json);
            case 'FusedBatchNorm': return $root.mindspore.schema.FusedBatchNorm.decodeText(reader, json);
            case 'Gather': return $root.mindspore.schema.Gather.decodeText(reader, json);
            case 'GatherNd': return $root.mindspore.schema.GatherNd.decodeText(reader, json);
            case 'Greater': return $root.mindspore.schema.Greater.decodeText(reader, json);
            case 'GreaterEqual': return $root.mindspore.schema.GreaterEqual.decodeText(reader, json);
            case 'HashtableLookup': return $root.mindspore.schema.HashtableLookup.decodeText(reader, json);
            case 'InstanceNorm': return $root.mindspore.schema.InstanceNorm.decodeText(reader, json);
            case 'LayerNormFusion': return $root.mindspore.schema.LayerNormFusion.decodeText(reader, json);
            case 'LeakyRelu': return $root.mindspore.schema.LeakyRelu.decodeText(reader, json);
            case 'Less': return $root.mindspore.schema.Less.decodeText(reader, json);
            case 'LessEqual': return $root.mindspore.schema.LessEqual.decodeText(reader, json);
            case 'Log': return $root.mindspore.schema.Log.decodeText(reader, json);
            case 'LogGrad': return $root.mindspore.schema.LogGrad.decodeText(reader, json);
            case 'LogicalAnd': return $root.mindspore.schema.LogicalAnd.decodeText(reader, json);
            case 'LogicalNot': return $root.mindspore.schema.LogicalNot.decodeText(reader, json);
            case 'LogicalOr': return $root.mindspore.schema.LogicalOr.decodeText(reader, json);
            case 'LpNormalization': return $root.mindspore.schema.LpNormalization.decodeText(reader, json);
            case 'LRN': return $root.mindspore.schema.LRN.decodeText(reader, json);
            case 'LshProjection': return $root.mindspore.schema.LshProjection.decodeText(reader, json);
            case 'LSTM': return $root.mindspore.schema.LSTM.decodeText(reader, json);
            case 'L2NormalizeFusion': return $root.mindspore.schema.L2NormalizeFusion.decodeText(reader, json);
            case 'MatMulFusion': return $root.mindspore.schema.MatMulFusion.decodeText(reader, json);
            case 'Maximum': return $root.mindspore.schema.Maximum.decodeText(reader, json);
            case 'MaximumGrad': return $root.mindspore.schema.MaximumGrad.decodeText(reader, json);
            case 'MaxPoolFusion': return $root.mindspore.schema.MaxPoolFusion.decodeText(reader, json);
            case 'MaxPoolGrad': return $root.mindspore.schema.MaxPoolGrad.decodeText(reader, json);
            case 'SwitchLayer': return $root.mindspore.schema.SwitchLayer.decodeText(reader, json);
            case 'Mfcc': return $root.mindspore.schema.Mfcc.decodeText(reader, json);
            case 'Minimum': return $root.mindspore.schema.Minimum.decodeText(reader, json);
            case 'MinimumGrad': return $root.mindspore.schema.MinimumGrad.decodeText(reader, json);
            case 'Mod': return $root.mindspore.schema.Mod.decodeText(reader, json);
            case 'MulFusion': return $root.mindspore.schema.MulFusion.decodeText(reader, json);
            case 'MulGrad': return $root.mindspore.schema.MulGrad.decodeText(reader, json);
            case 'Neg': return $root.mindspore.schema.Neg.decodeText(reader, json);
            case 'NegGrad': return $root.mindspore.schema.NegGrad.decodeText(reader, json);
            case 'NotEqual': return $root.mindspore.schema.NotEqual.decodeText(reader, json);
            case 'NonMaxSuppression': return $root.mindspore.schema.NonMaxSuppression.decodeText(reader, json);
            case 'OneHot': return $root.mindspore.schema.OneHot.decodeText(reader, json);
            case 'OnesLike': return $root.mindspore.schema.OnesLike.decodeText(reader, json);
            case 'PadFusion': return $root.mindspore.schema.PadFusion.decodeText(reader, json);
            case 'PartialFusion': return $root.mindspore.schema.PartialFusion.decodeText(reader, json);
            case 'PowerGrad': return $root.mindspore.schema.PowerGrad.decodeText(reader, json);
            case 'PowFusion': return $root.mindspore.schema.PowFusion.decodeText(reader, json);
            case 'PriorBox': return $root.mindspore.schema.PriorBox.decodeText(reader, json);
            case 'PReLUFusion': return $root.mindspore.schema.PReLUFusion.decodeText(reader, json);
            case 'QuantDTypeCast': return $root.mindspore.schema.QuantDTypeCast.decodeText(reader, json);
            case 'Rank': return $root.mindspore.schema.Rank.decodeText(reader, json);
            case 'Range': return $root.mindspore.schema.Range.decodeText(reader, json);
            case 'Reciprocal': return $root.mindspore.schema.Reciprocal.decodeText(reader, json);
            case 'RealDiv': return $root.mindspore.schema.RealDiv.decodeText(reader, json);
            case 'ReduceFusion': return $root.mindspore.schema.ReduceFusion.decodeText(reader, json);
            case 'Reshape': return $root.mindspore.schema.Reshape.decodeText(reader, json);
            case 'Resize': return $root.mindspore.schema.Resize.decodeText(reader, json);
            case 'ReverseSequence': return $root.mindspore.schema.ReverseSequence.decodeText(reader, json);
            case 'ReverseV2': return $root.mindspore.schema.ReverseV2.decodeText(reader, json);
            case 'Rfft': return $root.mindspore.schema.Rfft.decodeText(reader, json);
            case 'ROIPooling': return $root.mindspore.schema.ROIPooling.decodeText(reader, json);
            case 'Round': return $root.mindspore.schema.Round.decodeText(reader, json);
            case 'Rsqrt': return $root.mindspore.schema.Rsqrt.decodeText(reader, json);
            case 'ScaleFusion': return $root.mindspore.schema.ScaleFusion.decodeText(reader, json);
            case 'ScatterNd': return $root.mindspore.schema.ScatterNd.decodeText(reader, json);
            case 'SGD': return $root.mindspore.schema.SGD.decodeText(reader, json);
            case 'Shape': return $root.mindspore.schema.Shape.decodeText(reader, json);
            case 'SigmoidCrossEntropyWithLogits': return $root.mindspore.schema.SigmoidCrossEntropyWithLogits.decodeText(reader, json);
            case 'SigmoidCrossEntropyWithLogitsGrad': return $root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad.decodeText(reader, json);
            case 'Sin': return $root.mindspore.schema.Sin.decodeText(reader, json);
            case 'SkipGram': return $root.mindspore.schema.SkipGram.decodeText(reader, json);
            case 'SliceFusion': return $root.mindspore.schema.SliceFusion.decodeText(reader, json);
            case 'SmoothL1Loss': return $root.mindspore.schema.SmoothL1Loss.decodeText(reader, json);
            case 'SmoothL1LossGrad': return $root.mindspore.schema.SmoothL1LossGrad.decodeText(reader, json);
            case 'Softmax': return $root.mindspore.schema.Softmax.decodeText(reader, json);
            case 'SoftmaxCrossEntropyWithLogits': return $root.mindspore.schema.SoftmaxCrossEntropyWithLogits.decodeText(reader, json);
            case 'SpaceToBatch': return $root.mindspore.schema.SpaceToBatch.decodeText(reader, json);
            case 'SpaceToBatchND': return $root.mindspore.schema.SpaceToBatchND.decodeText(reader, json);
            case 'SpaceToDepth': return $root.mindspore.schema.SpaceToDepth.decodeText(reader, json);
            case 'SparseSoftmaxCrossEntropyWithLogits': return $root.mindspore.schema.SparseSoftmaxCrossEntropyWithLogits.decodeText(reader, json);
            case 'SparseToDense': return $root.mindspore.schema.SparseToDense.decodeText(reader, json);
            case 'Split': return $root.mindspore.schema.Split.decodeText(reader, json);
            case 'Sqrt': return $root.mindspore.schema.Sqrt.decodeText(reader, json);
            case 'Squeeze': return $root.mindspore.schema.Squeeze.decodeText(reader, json);
            case 'Square': return $root.mindspore.schema.Square.decodeText(reader, json);
            case 'SquaredDifference': return $root.mindspore.schema.SquaredDifference.decodeText(reader, json);
            case 'Stack': return $root.mindspore.schema.Stack.decodeText(reader, json);
            case 'StridedSlice': return $root.mindspore.schema.StridedSlice.decodeText(reader, json);
            case 'SubFusion': return $root.mindspore.schema.SubFusion.decodeText(reader, json);
            case 'SubGrad': return $root.mindspore.schema.SubGrad.decodeText(reader, json);
            case 'Switch': return $root.mindspore.schema.Switch.decodeText(reader, json);
            case 'TensorListFromTensor': return $root.mindspore.schema.TensorListFromTensor.decodeText(reader, json);
            case 'TensorListGetItem': return $root.mindspore.schema.TensorListGetItem.decodeText(reader, json);
            case 'TensorListReserve': return $root.mindspore.schema.TensorListReserve.decodeText(reader, json);
            case 'TensorListSetItem': return $root.mindspore.schema.TensorListSetItem.decodeText(reader, json);
            case 'TensorListStack': return $root.mindspore.schema.TensorListStack.decodeText(reader, json);
            case 'TileFusion': return $root.mindspore.schema.TileFusion.decodeText(reader, json);
            case 'TopKFusion': return $root.mindspore.schema.TopKFusion.decodeText(reader, json);
            case 'Transpose': return $root.mindspore.schema.Transpose.decodeText(reader, json);
            case 'Unique': return $root.mindspore.schema.Unique.decodeText(reader, json);
            case 'UnsortedSegmentSum': return $root.mindspore.schema.UnsortedSegmentSum.decodeText(reader, json);
            case 'Unsqueeze': return $root.mindspore.schema.Unsqueeze.decodeText(reader, json);
            case 'Unstack': return $root.mindspore.schema.Unstack.decodeText(reader, json);
            case 'LSTMGrad': return $root.mindspore.schema.LSTMGrad.decodeText(reader, json);
            case 'Where': return $root.mindspore.schema.Where.decodeText(reader, json);
            case 'ZerosLike': return $root.mindspore.schema.ZerosLike.decodeText(reader, json);
            case 'Select': return $root.mindspore.schema.Select.decodeText(reader, json);
            case 'ScatterNdUpdate': return $root.mindspore.schema.ScatterNdUpdate.decodeText(reader, json);
            case 'GRU': return $root.mindspore.schema.GRU.decodeText(reader, json);
            case 'NonZero': return $root.mindspore.schema.NonZero.decodeText(reader, json);
            case 'InvertPermutation': return $root.mindspore.schema.InvertPermutation.decodeText(reader, json);
            case 'Size': return $root.mindspore.schema.Size.decodeText(reader, json);
            case 'RandomStandardNormal': return $root.mindspore.schema.RandomStandardNormal.decodeText(reader, json);
            case 'CropAndResize': return $root.mindspore.schema.CropAndResize.decodeText(reader, json);
            case 'Erf': return $root.mindspore.schema.Erf.decodeText(reader, json);
            case 'StridedSliceGrad': return $root.mindspore.schema.StridedSliceGrad.decodeText(reader, json);
            case 'IsFinite': return $root.mindspore.schema.IsFinite.decodeText(reader, json);
            case 'LinSpace': return $root.mindspore.schema.LinSpace.decodeText(reader, json);
            case 'UniformReal': return $root.mindspore.schema.UniformReal.decodeText(reader, json);
            case 'AbsGrad': return $root.mindspore.schema.AbsGrad.decodeText(reader, json);
            case 'RsqrtGrad': return $root.mindspore.schema.RsqrtGrad.decodeText(reader, json);
            case 'SqrtGrad': return $root.mindspore.schema.SqrtGrad.decodeText(reader, json);
            case 'LayerNormGrad': return $root.mindspore.schema.LayerNormGrad.decodeText(reader, json);
            case 'ResizeGrad': return $root.mindspore.schema.ResizeGrad.decodeText(reader, json);
            case 'Splice': return $root.mindspore.schema.Splice.decodeText(reader, json);
            case 'LogSoftmax': return $root.mindspore.schema.LogSoftmax.decodeText(reader, json);
            case 'Call': return $root.mindspore.schema.Call.decodeText(reader, json);
            case 'Custom': return $root.mindspore.schema.Custom.decodeText(reader, json);
            case 'CumSum': return $root.mindspore.schema.CumSum.decodeText(reader, json);
            case 'SplitWithOverlap': return $root.mindspore.schema.SplitWithOverlap.decodeText(reader, json);
            case 'GenOP': return $root.mindspore.schema.GenOP.decodeText(reader, json);
            case 'RaggedRange': return $root.mindspore.schema.RaggedRange.decodeText(reader, json);
            case 'GLU': return $root.mindspore.schema.GLU.decodeText(reader, json);
            case 'TensorArray': return $root.mindspore.schema.TensorArray.decodeText(reader, json);
            case 'TensorArrayRead': return $root.mindspore.schema.TensorArrayRead.decodeText(reader, json);
            case 'TensorArrayWrite': return $root.mindspore.schema.TensorArrayWrite.decodeText(reader, json);
            case 'Affine': return $root.mindspore.schema.Affine.decodeText(reader, json);
            case 'AllGather': return $root.mindspore.schema.AllGather.decodeText(reader, json);
            case 'ReduceScatter': return $root.mindspore.schema.ReduceScatter.decodeText(reader, json);
            case 'DynamicQuant': return $root.mindspore.schema.DynamicQuant.decodeText(reader, json);
            case 'LSTMGradData': return $root.mindspore.schema.LSTMGradData.decodeText(reader, json);
            case 'LSTMGradWeight': return $root.mindspore.schema.LSTMGradWeight.decodeText(reader, json);
            case 'RandomNormal': return $root.mindspore.schema.RandomNormal.decodeText(reader, json);
            case 'NLLLoss': return $root.mindspore.schema.NLLLoss.decodeText(reader, json);
            case 'NLLLossGrad': return $root.mindspore.schema.NLLLossGrad.decodeText(reader, json);
            case 'FormatTranspose': return $root.mindspore.schema.FormatTranspose.decodeText(reader, json);
            case 'GatherD': return $root.mindspore.schema.GatherD.decodeText(reader, json);
            case 'GroupNormFusion': return $root.mindspore.schema.GroupNormFusion.decodeText(reader, json);
            case 'Log1p': return $root.mindspore.schema.Log1p.decodeText(reader, json);
            case 'TensorScatterAdd': return $root.mindspore.schema.TensorScatterAdd.decodeText(reader, json);
            case 'SparseFillEmptyRows': return $root.mindspore.schema.SparseFillEmptyRows.decodeText(reader, json);
            case 'SparseReshape': return $root.mindspore.schema.SparseReshape.decodeText(reader, json);
            case 'SparseSegmentSum': return $root.mindspore.schema.SparseSegmentSum.decodeText(reader, json);
            case 'ScatterElements': return $root.mindspore.schema.ScatterElements.decodeText(reader, json);
            case 'Triu': return $root.mindspore.schema.Triu.decodeText(reader, json);
            case 'Tril': return $root.mindspore.schema.Tril.decodeText(reader, json);
            case 'AdamWeightDecay': return $root.mindspore.schema.AdamWeightDecay.decodeText(reader, json);
            default: return undefined;
        }
    }
};

$root.mindspore.schema.Abs = class Abs {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Abs();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Abs();
        return $;
    }
};

$root.mindspore.schema.Activation = class Activation {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Activation();
        $.activation_type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        $.min_val = reader.float32_(position, 8, 0);
        $.max_val = reader.float32_(position, 10, 0);
        $.approximate = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Activation();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        $.alpha = reader.value(json.alpha, 0);
        $.min_val = reader.value(json.min_val, 0);
        $.max_val = reader.value(json.max_val, 0);
        $.approximate = reader.value(json.approximate, false);
        return $;
    }
};

$root.mindspore.schema.ActivationGrad = class ActivationGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ActivationGrad();
        $.activation_type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ActivationGrad();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        $.alpha = reader.value(json.alpha, 0);
        return $;
    }
};

$root.mindspore.schema.Adam = class Adam {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Adam();
        $.use_locking = reader.bool_(position, 4, false);
        $.use_nesterov = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Adam();
        $.use_locking = reader.value(json.use_locking, false);
        $.use_nesterov = reader.value(json.use_nesterov, false);
        return $;
    }
};

$root.mindspore.schema.AddFusion = class AddFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AddFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AddFusion();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.AdderFusion = class AdderFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AdderFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad_list = reader.int64s_(position, 14);
        $.group = reader.int64_(position, 16, 0);
        $.in_channel = reader.int64_(position, 18, 0);
        $.out_channel = reader.int64_(position, 20, 0);
        $.activation_type = reader.int8_(position, 22, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AdderFusion();
        $.format = $root.mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.group = reader.value(json.group, 0);
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.AddGrad = class AddGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.AddGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.AddGrad();
        return $;
    }
};

$root.mindspore.schema.AddN = class AddN {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.AddN();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.AddN();
        return $;
    }
};

$root.mindspore.schema.All = class All {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.All();
        $.keep_dims = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.All();
        $.keep_dims = reader.value(json.keep_dims, 0);
        return $;
    }
};

$root.mindspore.schema.ApplyMomentum = class ApplyMomentum {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ApplyMomentum();
        $.use_nesterov = reader.bool_(position, 4, false);
        $.use_locking = reader.bool_(position, 6, false);
        $.gradient_scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ApplyMomentum();
        $.use_nesterov = reader.value(json.use_nesterov, false);
        $.use_locking = reader.value(json.use_locking, false);
        $.gradient_scale = reader.value(json.gradient_scale, 0);
        return $;
    }
};

$root.mindspore.schema.ArgMaxFusion = class ArgMaxFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ArgMaxFusion();
        $.axis = reader.int64_(position, 4, 0);
        $.top_k = reader.int64_(position, 6, 1);
        $.keep_dims = reader.bool_(position, 8, false);
        $.out_max_value = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ArgMaxFusion();
        $.axis = reader.value(json.axis, 0);
        $.top_k = reader.value(json.top_k, 1);
        $.keep_dims = reader.value(json.keep_dims, false);
        $.out_max_value = reader.value(json.out_max_value, false);
        return $;
    }
};

$root.mindspore.schema.ArgMinFusion = class ArgMinFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ArgMinFusion();
        $.axis = reader.int64_(position, 4, 0);
        $.top_k = reader.int64_(position, 6, 0);
        $.keep_dims = reader.bool_(position, 8, false);
        $.out_max_value = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ArgMinFusion();
        $.axis = reader.value(json.axis, 0);
        $.top_k = reader.value(json.top_k, 0);
        $.keep_dims = reader.value(json.keep_dims, false);
        $.out_max_value = reader.value(json.out_max_value, false);
        return $;
    }
};

$root.mindspore.schema.Assert = class Assert {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Assert();
        $.summarize = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Assert();
        $.summarize = reader.value(json.summarize, 0);
        return $;
    }
};

$root.mindspore.schema.Assign = class Assign {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Assign();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Assign();
        return $;
    }
};

$root.mindspore.schema.AssignAdd = class AssignAdd {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.AssignAdd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.AssignAdd();
        return $;
    }
};

$root.mindspore.schema.AudioSpectrogram = class AudioSpectrogram {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AudioSpectrogram();
        $.window_size = reader.int64_(position, 4, 0);
        $.stride = reader.int64_(position, 6, 0);
        $.mag_square = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AudioSpectrogram();
        $.window_size = reader.value(json.window_size, 0);
        $.stride = reader.value(json.stride, 0);
        $.mag_square = reader.value(json.mag_square, false);
        return $;
    }
};

$root.mindspore.schema.AvgPoolFusion = class AvgPoolFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AvgPoolFusion();
        $.kernel_size = reader.int64s_(position, 4);
        $.strides = reader.int64s_(position, 6);
        $.pad = reader.int64s_(position, 8);
        $.pad_mode = reader.int8_(position, 10, 0);
        $.round_mode = reader.int8_(position, 12, 0);
        $.format = reader.int32_(position, 14, 0);
        $.global = reader.bool_(position, 16, false);
        $.activation_type = reader.int8_(position, 18, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AvgPoolFusion();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad = reader.array(json.pad);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.round_mode = $root.mindspore.schema.RoundMode[json.round_mode];
        $.format = $root.mindspore.schema.Format[json.format];
        $.global = reader.value(json.global, false);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.AvgPoolGrad = class AvgPoolGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AvgPoolGrad();
        $.kernel_size = reader.int64s_(position, 4);
        $.strides = reader.int64s_(position, 6);
        $.pad_mode = reader.int8_(position, 8, 0);
        $.format = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AvgPoolGrad();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.format = $root.mindspore.schema.Format[json.format];
        return $;
    }
};

$root.mindspore.schema.BatchNorm = class BatchNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchNorm();
        $.epsilon = reader.float32_(position, 4, 0);
        $.format = reader.int32_(position, 6, 0);
        $.is_training = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchNorm();
        $.epsilon = reader.value(json.epsilon, 0);
        $.format = $root.mindspore.schema.Format[json.format];
        $.is_training = reader.value(json.is_training, false);
        return $;
    }
};

$root.mindspore.schema.BatchNormGrad = class BatchNormGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchNormGrad();
        $.epsilon = reader.float32_(position, 4, 0);
        $.is_training = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchNormGrad();
        $.epsilon = reader.value(json.epsilon, 0);
        $.is_training = reader.value(json.is_training, false);
        return $;
    }
};

$root.mindspore.schema.BatchToSpace = class BatchToSpace {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchToSpace();
        $.block_size = reader.int64s_(position, 4);
        $.crops = reader.table(position, 6, $root.mindspore.schema.Vec2D.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchToSpace();
        $.block_size = reader.array(json.block_size);
        $.crops = reader.object(json.crops, $root.mindspore.schema.Vec2D.decodeText);
        return $;
    }
};

$root.mindspore.schema.BatchToSpaceND = class BatchToSpaceND {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchToSpaceND();
        $.block_shape = reader.int64s_(position, 4);
        $.crops = reader.table(position, 6, $root.mindspore.schema.Vec2D.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchToSpaceND();
        $.block_shape = reader.array(json.block_shape);
        $.crops = reader.object(json.crops, $root.mindspore.schema.Vec2D.decodeText);
        return $;
    }
};

$root.mindspore.schema.BiasAdd = class BiasAdd {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BiasAdd();
        $.format = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BiasAdd();
        $.format = $root.mindspore.schema.Format[json.format];
        return $;
    }
};

$root.mindspore.schema.BinaryCrossEntropy = class BinaryCrossEntropy {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropy();
        $.reduction = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropy();
        $.reduction = $root.mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

$root.mindspore.schema.BinaryCrossEntropyGrad = class BinaryCrossEntropyGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropyGrad();
        $.reduction = reader.int8_(position, 4, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropyGrad();
        $.reduction = $root.mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

$root.mindspore.schema.BiasAddGrad = class BiasAddGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.BiasAddGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.BiasAddGrad();
        return $;
    }
};

$root.mindspore.schema.BroadcastTo = class BroadcastTo {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BroadcastTo();
        $.shape = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BroadcastTo();
        $.shape = reader.array(json.shape);
        return $;
    }
};

$root.mindspore.schema.Cast = class Cast {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Cast();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Cast();
        return $;
    }
};

$root.mindspore.schema.Ceil = class Ceil {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Ceil();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Ceil();
        return $;
    }
};

$root.mindspore.schema.Clip = class Clip {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Clip();
        $.max = reader.float32_(position, 4, 0);
        $.min = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Clip();
        $.max = reader.value(json.max, 0);
        $.min = reader.value(json.min, 0);
        return $;
    }
};

$root.mindspore.schema.Concat = class Concat {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Concat();
        $.axis = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Concat();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.Attention = class Attention {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Attention();
        $.head_num = reader.int64_(position, 4, 0);
        $.head_size = reader.int64_(position, 6, 0);
        $.cross = reader.bool_(position, 8, false);
        $.scale = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Attention();
        $.head_num = reader.value(json.head_num, 0);
        $.head_size = reader.value(json.head_size, 0);
        $.cross = reader.value(json.cross, false);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

$root.mindspore.schema.Conv2DBackpropFilterFusion = class Conv2DBackpropFilterFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2DBackpropFilterFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad_list = reader.int64s_(position, 14);
        $.mode = reader.int64_(position, 16, 0);
        $.group = reader.int64_(position, 18, 0);
        $.in_channel = reader.int64_(position, 20, 0);
        $.out_channel = reader.int64_(position, 22, 0);
        $.activation_type = reader.int8_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2DBackpropFilterFusion();
        $.format = $root.mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.value(json.mode, 0);
        $.group = reader.value(json.group, 0);
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.Conv2DBackpropInputFusion = class Conv2DBackpropInputFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2DBackpropInputFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad = reader.int64s_(position, 14);
        $.pad_list = reader.int64s_(position, 16);
        $.mode = reader.int64_(position, 18, 0);
        $.group = reader.int64_(position, 20, 0);
        $.in_channel = reader.int64_(position, 22, 0);
        $.out_channel = reader.int64_(position, 24, 0);
        $.activation_type = reader.int8_(position, 26, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2DBackpropInputFusion();
        $.format = $root.mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad = reader.array(json.pad);
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.value(json.mode, 0);
        $.group = reader.value(json.group, 0);
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.Conv2DFusion = class Conv2DFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2DFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad_list = reader.int64s_(position, 14);
        $.mode = reader.int64_(position, 16, 0);
        $.group = reader.int64_(position, 18, 0);
        $.in_channel = reader.int64_(position, 20, 0);
        $.out_channel = reader.int64_(position, 22, 0);
        $.activation_type = reader.int8_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2DFusion();
        $.format = $root.mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.value(json.mode, 0);
        $.group = reader.value(json.group, 0);
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.Conv2dTransposeFusion = class Conv2dTransposeFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2dTransposeFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad = reader.int64s_(position, 14);
        $.pad_list = reader.int64s_(position, 16);
        $.mode = reader.int64_(position, 18, 0);
        $.group = reader.int64_(position, 20, 0);
        $.in_channel = reader.int64_(position, 22, 0);
        $.out_channel = reader.int64_(position, 24, 0);
        $.activation_type = reader.int8_(position, 26, 0);
        $.output_paddings = reader.int64s_(position, 28);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2dTransposeFusion();
        $.format = $root.mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad = reader.array(json.pad);
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.value(json.mode, 0);
        $.group = reader.value(json.group, 0);
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        $.output_paddings = reader.array(json.output_paddings);
        return $;
    }
};

$root.mindspore.schema.Cos = class Cos {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Cos();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Cos();
        return $;
    }
};

$root.mindspore.schema.ConstantOfShape = class ConstantOfShape {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ConstantOfShape();
        $.data_type = reader.int64_(position, 4, 0);
        $.value = reader.typedArray(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ConstantOfShape();
        $.data_type = reader.value(json.data_type, 0);
        $.value = reader.typedArray(json.value, Float32Array);
        return $;
    }
};

$root.mindspore.schema.Crop = class Crop {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Crop();
        $.axis = reader.int64_(position, 4, 0);
        $.offsets = reader.int64s_(position, 6);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Crop();
        $.axis = reader.value(json.axis, 0);
        $.offsets = reader.array(json.offsets);
        return $;
    }
};

$root.mindspore.schema.CustomExtractFeatures = class CustomExtractFeatures {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.CustomExtractFeatures();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.CustomExtractFeatures();
        return $;
    }
};

$root.mindspore.schema.CustomNormalize = class CustomNormalize {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.CustomNormalize();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.CustomNormalize();
        return $;
    }
};

$root.mindspore.schema.CustomPredict = class CustomPredict {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.CustomPredict();
        $.output_num = reader.int64_(position, 4, 0);
        $.weight_threshold = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.CustomPredict();
        $.output_num = reader.value(json.output_num, 0);
        $.weight_threshold = reader.value(json.weight_threshold, 0);
        return $;
    }
};

$root.mindspore.schema.DeConv2DGradFilter = class DeConv2DGradFilter {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DeConv2DGradFilter();
        $.in_channel = reader.int64_(position, 4, 0);
        $.out_channel = reader.int64_(position, 6, 0);
        $.kernel_size = reader.int64s_(position, 8);
        $.pad_mode = reader.int8_(position, 10, 0);
        $.pad_list = reader.int64s_(position, 12);
        $.stride = reader.int64s_(position, 14);
        $.dilation = reader.int64s_(position, 16);
        $.group = reader.int64_(position, 18, 0);
        $.format = reader.int32_(position, 20, 0);
        $.activation_type = reader.int8_(position, 22, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DeConv2DGradFilter();
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.kernel_size = reader.array(json.kernel_size);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.group = reader.value(json.group, 0);
        $.format = $root.mindspore.schema.Format[json.format];
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.Depend = class Depend {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Depend();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Depend();
        return $;
    }
};

$root.mindspore.schema.DepthToSpace = class DepthToSpace {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DepthToSpace();
        $.block_size = reader.int64_(position, 4, 0);
        $.format = reader.int32_(position, 6, 0);
        $.mode = reader.string_(position, 8, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DepthToSpace();
        $.block_size = reader.value(json.block_size, 0);
        $.format = $root.mindspore.schema.Format[json.format];
        $.mode = reader.value(json.mode, null);
        return $;
    }
};

$root.mindspore.schema.DetectionPostProcess = class DetectionPostProcess {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DetectionPostProcess();
        $.format = reader.int32_(position, 4, 0);
        $.input_size = reader.int64_(position, 6, 0);
        $.scale = reader.typedArray(position, 8, Float32Array);
        $.nms_iou_threshold = reader.float32_(position, 10, 0);
        $.nms_score_threshold = reader.float32_(position, 12, 0);
        $.max_detections = reader.int64_(position, 14, 0);
        $.detections_per_class = reader.int64_(position, 16, 0);
        $.max_classes_per_detection = reader.int64_(position, 18, 0);
        $.num_classes = reader.int64_(position, 20, 0);
        $.use_regular_nms = reader.bool_(position, 22, false);
        $.out_quantized = reader.bool_(position, 24, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DetectionPostProcess();
        $.format = $root.mindspore.schema.Format[json.format];
        $.input_size = reader.value(json.input_size, 0);
        $.scale = reader.typedArray(json.scale, Float32Array);
        $.nms_iou_threshold = reader.value(json.nms_iou_threshold, 0);
        $.nms_score_threshold = reader.value(json.nms_score_threshold, 0);
        $.max_detections = reader.value(json.max_detections, 0);
        $.detections_per_class = reader.value(json.detections_per_class, 0);
        $.max_classes_per_detection = reader.value(json.max_classes_per_detection, 0);
        $.num_classes = reader.value(json.num_classes, 0);
        $.use_regular_nms = reader.value(json.use_regular_nms, false);
        $.out_quantized = reader.value(json.out_quantized, false);
        return $;
    }
};

$root.mindspore.schema.DivFusion = class DivFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DivFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DivFusion();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.DivGrad = class DivGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.DivGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.DivGrad();
        return $;
    }
};

$root.mindspore.schema.Dropout = class Dropout {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Dropout();
        $.keep_prob = reader.float32_(position, 4, 0.5);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Dropout();
        $.keep_prob = reader.value(json.keep_prob, 0.5);
        return $;
    }
};

$root.mindspore.schema.DropoutGrad = class DropoutGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DropoutGrad();
        $.keep_prob = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DropoutGrad();
        $.keep_prob = reader.value(json.keep_prob, 0);
        return $;
    }
};

$root.mindspore.schema.Elu = class Elu {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Elu();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Elu();
        $.alpha = reader.value(json.alpha, 0);
        return $;
    }
};

$root.mindspore.schema.Eltwise = class Eltwise {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Eltwise();
        $.mode = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Eltwise();
        $.mode = $root.mindspore.schema.EltwiseMode[json.mode];
        return $;
    }
};

$root.mindspore.schema.Equal = class Equal {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Equal();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Equal();
        return $;
    }
};

$root.mindspore.schema.EmbeddingLookupFusion = class EmbeddingLookupFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.EmbeddingLookupFusion();
        $.max_norm = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.EmbeddingLookupFusion();
        $.max_norm = reader.value(json.max_norm, 0);
        return $;
    }
};

$root.mindspore.schema.ExpFusion = class ExpFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ExpFusion();
        $.base = reader.float32_(position, 4, -1);
        $.scale = reader.float32_(position, 6, 1);
        $.shift = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ExpFusion();
        $.base = reader.value(json.base, -1);
        $.scale = reader.value(json.scale, 1);
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

$root.mindspore.schema.ExpandDims = class ExpandDims {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.ExpandDims();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.ExpandDims();
        return $;
    }
};

$root.mindspore.schema.FakeQuantWithMinMaxVars = class FakeQuantWithMinMaxVars {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxVars();
        $.num_bits = reader.int64_(position, 4, 0);
        $.narrow_range = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxVars();
        $.num_bits = reader.value(json.num_bits, 0);
        $.narrow_range = reader.value(json.narrow_range, false);
        return $;
    }
};

$root.mindspore.schema.FakeQuantWithMinMaxVarsPerChannel = class FakeQuantWithMinMaxVarsPerChannel {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxVarsPerChannel();
        $.num_bits = reader.int64_(position, 4, 0);
        $.narrow_range = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxVarsPerChannel();
        $.num_bits = reader.value(json.num_bits, 0);
        $.narrow_range = reader.value(json.narrow_range, false);
        return $;
    }
};

$root.mindspore.schema.FftReal = class FftReal {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FftReal();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FftReal();
        return $;
    }
};

$root.mindspore.schema.FftImag = class FftImag {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FftImag();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FftImag();
        return $;
    }
};

$root.mindspore.schema.Flatten = class Flatten {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Flatten();
        $.axis = reader.int64_(position, 4, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Flatten();
        $.axis = reader.value(json.axis, 1);
        return $;
    }
};

$root.mindspore.schema.FlattenGrad = class FlattenGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FlattenGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FlattenGrad();
        return $;
    }
};

$root.mindspore.schema.Floor = class Floor {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Floor();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Floor();
        return $;
    }
};

$root.mindspore.schema.FloorDiv = class FloorDiv {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FloorDiv();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FloorDiv();
        return $;
    }
};

$root.mindspore.schema.FloorMod = class FloorMod {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FloorMod();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FloorMod();
        return $;
    }
};

$root.mindspore.schema.Fill = class Fill {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Fill();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Fill();
        return $;
    }
};

$root.mindspore.schema.FullConnection = class FullConnection {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FullConnection();
        $.has_bias = reader.bool_(position, 4, false);
        $.use_axis = reader.bool_(position, 6, false);
        $.axis = reader.int64_(position, 8, 0);
        $.activation_type = reader.int8_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FullConnection();
        $.has_bias = reader.value(json.has_bias, false);
        $.use_axis = reader.value(json.use_axis, false);
        $.axis = reader.value(json.axis, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.FusedBatchNorm = class FusedBatchNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FusedBatchNorm();
        $.epsilon = reader.float32_(position, 4, 0.0001);
        $.momentum = reader.float32_(position, 6, 0.9);
        $.mode = reader.int64_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FusedBatchNorm();
        $.epsilon = reader.value(json.epsilon, 0.0001);
        $.momentum = reader.value(json.momentum, 0.9);
        $.mode = reader.value(json.mode, 0);
        return $;
    }
};

$root.mindspore.schema.Gather = class Gather {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Gather();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Gather();
        return $;
    }
};

$root.mindspore.schema.GatherNd = class GatherNd {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.GatherNd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.GatherNd();
        return $;
    }
};

$root.mindspore.schema.Greater = class Greater {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Greater();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Greater();
        return $;
    }
};

$root.mindspore.schema.GreaterEqual = class GreaterEqual {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.GreaterEqual();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.GreaterEqual();
        return $;
    }
};

$root.mindspore.schema.HashtableLookup = class HashtableLookup {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.HashtableLookup();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.HashtableLookup();
        return $;
    }
};

$root.mindspore.schema.InstanceNorm = class InstanceNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.InstanceNorm();
        $.epsilon = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.InstanceNorm();
        $.epsilon = reader.value(json.epsilon, 0);
        return $;
    }
};

$root.mindspore.schema.LayerNormFusion = class LayerNormFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LayerNormFusion();
        $.begin_norm_axis = reader.int64_(position, 4, 0);
        $.epsilon = reader.float32_(position, 6, 0.00001);
        $.elementwise_affine = reader.bool_(position, 8, false);
        $.begin_params_axis = reader.int64_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LayerNormFusion();
        $.begin_norm_axis = reader.value(json.begin_norm_axis, 0);
        $.epsilon = reader.value(json.epsilon, 0.00001);
        $.elementwise_affine = reader.value(json.elementwise_affine, false);
        $.begin_params_axis = reader.value(json.begin_params_axis, 0);
        return $;
    }
};

$root.mindspore.schema.LeakyRelu = class LeakyRelu {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LeakyRelu();
        $.negative_slope = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LeakyRelu();
        $.negative_slope = reader.value(json.negative_slope, 0);
        return $;
    }
};

$root.mindspore.schema.Less = class Less {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Less();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Less();
        return $;
    }
};

$root.mindspore.schema.LessEqual = class LessEqual {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LessEqual();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LessEqual();
        return $;
    }
};

$root.mindspore.schema.Log = class Log {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Log();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Log();
        return $;
    }
};

$root.mindspore.schema.LogGrad = class LogGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LogGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LogGrad();
        return $;
    }
};

$root.mindspore.schema.LogicalAnd = class LogicalAnd {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LogicalAnd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LogicalAnd();
        return $;
    }
};

$root.mindspore.schema.LogicalNot = class LogicalNot {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LogicalNot();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LogicalNot();
        return $;
    }
};

$root.mindspore.schema.LogicalOr = class LogicalOr {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LogicalOr();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LogicalOr();
        return $;
    }
};

$root.mindspore.schema.LpNormalization = class LpNormalization {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LpNormalization();
        $.axis = reader.int64_(position, 4, 0);
        $.p = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LpNormalization();
        $.axis = reader.value(json.axis, 0);
        $.p = reader.value(json.p, 0);
        return $;
    }
};

$root.mindspore.schema.LRN = class LRN {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LRN();
        $.depth_radius = reader.int64_(position, 4, 0);
        $.bias = reader.float32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        $.norm_region = reader.string_(position, 12, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LRN();
        $.depth_radius = reader.value(json.depth_radius, 0);
        $.bias = reader.value(json.bias, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
        $.norm_region = reader.value(json.norm_region, null);
        return $;
    }
};

$root.mindspore.schema.LshProjection = class LshProjection {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LshProjection();
        $.type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LshProjection();
        $.type = $root.mindspore.schema.LshProjectionType[json.type];
        return $;
    }
};

$root.mindspore.schema.LSTM = class LSTM {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LSTM();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0);
        $.hidden_size = reader.int64_(position, 10, 0);
        $.num_layers = reader.int64_(position, 12, 0);
        $.num_directions = reader.int64_(position, 14, 0);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LSTM();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.value(json.input_size, 0);
        $.hidden_size = reader.value(json.hidden_size, 0);
        $.num_layers = reader.value(json.num_layers, 0);
        $.num_directions = reader.value(json.num_directions, 0);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

$root.mindspore.schema.LSTMGrad = class LSTMGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LSTMGrad();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0);
        $.hidden_size = reader.int64_(position, 10, 0);
        $.num_layers = reader.int64_(position, 12, 0);
        $.num_directions = reader.int64_(position, 14, 0);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LSTMGrad();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.value(json.input_size, 0);
        $.hidden_size = reader.value(json.hidden_size, 0);
        $.num_layers = reader.value(json.num_layers, 0);
        $.num_directions = reader.value(json.num_directions, 0);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

$root.mindspore.schema.L2NormalizeFusion = class L2NormalizeFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.L2NormalizeFusion();
        $.axis = reader.int64s_(position, 4);
        $.epsilon = reader.float32_(position, 6, 0);
        $.activation_type = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.L2NormalizeFusion();
        $.axis = reader.array(json.axis);
        $.epsilon = reader.value(json.epsilon, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.MatMulFusion = class MatMulFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MatMulFusion();
        $.transpose_a = reader.bool_(position, 4, false);
        $.transpose_b = reader.bool_(position, 6, false);
        $.activation_type = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MatMulFusion();
        $.transpose_a = reader.value(json.transpose_a, false);
        $.transpose_b = reader.value(json.transpose_b, false);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.Maximum = class Maximum {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Maximum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Maximum();
        return $;
    }
};

$root.mindspore.schema.MaximumGrad = class MaximumGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MaximumGrad();
        $.grad_x = reader.bool_(position, 4, false);
        $.grad_y = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MaximumGrad();
        $.grad_x = reader.value(json.grad_x, false);
        $.grad_y = reader.value(json.grad_y, false);
        return $;
    }
};

$root.mindspore.schema.MaxPoolFusion = class MaxPoolFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MaxPoolFusion();
        $.kernel_size = reader.int64s_(position, 4);
        $.strides = reader.int64s_(position, 6);
        $.pad = reader.int64s_(position, 8);
        $.pad_mode = reader.int8_(position, 10, 0);
        $.round_mode = reader.int8_(position, 12, 0);
        $.format = reader.int32_(position, 14, 0);
        $.global = reader.bool_(position, 16, false);
        $.activation_type = reader.int8_(position, 18, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MaxPoolFusion();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad = reader.array(json.pad);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.round_mode = $root.mindspore.schema.RoundMode[json.round_mode];
        $.format = $root.mindspore.schema.Format[json.format];
        $.global = reader.value(json.global, false);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.MaxPoolGrad = class MaxPoolGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MaxPoolGrad();
        $.kernel_size = reader.int64s_(position, 4);
        $.strides = reader.int64s_(position, 6);
        $.pad_mode = reader.int8_(position, 8, 0);
        $.format = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MaxPoolGrad();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.format = $root.mindspore.schema.Format[json.format];
        return $;
    }
};

$root.mindspore.schema.SwitchLayer = class SwitchLayer {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SwitchLayer();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SwitchLayer();
        return $;
    }
};

$root.mindspore.schema.Mfcc = class Mfcc {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Mfcc();
        $.freq_upper_limit = reader.float32_(position, 4, 0);
        $.freq_lower_limit = reader.float32_(position, 6, 0);
        $.filter_bank_channel_num = reader.int64_(position, 8, 0);
        $.dct_coeff_num = reader.int64_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Mfcc();
        $.freq_upper_limit = reader.value(json.freq_upper_limit, 0);
        $.freq_lower_limit = reader.value(json.freq_lower_limit, 0);
        $.filter_bank_channel_num = reader.value(json.filter_bank_channel_num, 0);
        $.dct_coeff_num = reader.value(json.dct_coeff_num, 0);
        return $;
    }
};

$root.mindspore.schema.Minimum = class Minimum {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Minimum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Minimum();
        return $;
    }
};

$root.mindspore.schema.MinimumGrad = class MinimumGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MinimumGrad();
        $.grad_x = reader.bool_(position, 4, false);
        $.grad_y = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MinimumGrad();
        $.grad_x = reader.value(json.grad_x, false);
        $.grad_y = reader.value(json.grad_y, false);
        return $;
    }
};

$root.mindspore.schema.Mod = class Mod {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Mod();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Mod();
        return $;
    }
};

$root.mindspore.schema.MulFusion = class MulFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MulFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MulFusion();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.MulGrad = class MulGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.MulGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.MulGrad();
        return $;
    }
};

$root.mindspore.schema.Neg = class Neg {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Neg();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Neg();
        return $;
    }
};

$root.mindspore.schema.NegGrad = class NegGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.NegGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.NegGrad();
        return $;
    }
};

$root.mindspore.schema.NotEqual = class NotEqual {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.NotEqual();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.NotEqual();
        return $;
    }
};

$root.mindspore.schema.NonMaxSuppression = class NonMaxSuppression {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.NonMaxSuppression();
        $.center_point_box = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.NonMaxSuppression();
        $.center_point_box = reader.value(json.center_point_box, 0);
        return $;
    }
};

$root.mindspore.schema.OneHot = class OneHot {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.OneHot();
        $.axis = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.OneHot();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.OnesLike = class OnesLike {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.OnesLike();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.OnesLike();
        return $;
    }
};

$root.mindspore.schema.PadFusion = class PadFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PadFusion();
        $.paddings = reader.table(position, 4, $root.mindspore.schema.Vec2D.decode);
        $.padding_mode = reader.int8_(position, 6, 0);
        $.constant_value = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PadFusion();
        $.paddings = reader.object(json.paddings, $root.mindspore.schema.Vec2D.decodeText);
        $.padding_mode = $root.mindspore.schema.PaddingMode[json.padding_mode];
        $.constant_value = reader.value(json.constant_value, 0);
        return $;
    }
};

$root.mindspore.schema.PartialFusion = class PartialFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PartialFusion();
        $.sub_graph_index = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PartialFusion();
        $.sub_graph_index = reader.value(json.sub_graph_index, 0);
        return $;
    }
};

$root.mindspore.schema.PowerGrad = class PowerGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PowerGrad();
        $.power = reader.float32_(position, 4, 0);
        $.scale = reader.float32_(position, 6, 0);
        $.shift = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PowerGrad();
        $.power = reader.value(json.power, 0);
        $.scale = reader.value(json.scale, 0);
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

$root.mindspore.schema.PowFusion = class PowFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PowFusion();
        $.scale = reader.float32_(position, 4, 1);
        $.shift = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PowFusion();
        $.scale = reader.value(json.scale, 1);
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

$root.mindspore.schema.PriorBox = class PriorBox {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PriorBox();
        $.min_sizes = reader.int64s_(position, 4);
        $.max_sizes = reader.int64s_(position, 6);
        $.aspect_ratios = reader.typedArray(position, 8, Float32Array);
        $.variances = reader.typedArray(position, 10, Float32Array);
        $.image_size_w = reader.int64_(position, 12, 0);
        $.image_size_h = reader.int64_(position, 14, 0);
        $.step_w = reader.float32_(position, 16, 0);
        $.step_h = reader.float32_(position, 18, 0);
        $.clip = reader.bool_(position, 20, false);
        $.flip = reader.bool_(position, 22, false);
        $.offset = reader.float32_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PriorBox();
        $.min_sizes = reader.array(json.min_sizes);
        $.max_sizes = reader.array(json.max_sizes);
        $.aspect_ratios = reader.typedArray(json.aspect_ratios, Float32Array);
        $.variances = reader.typedArray(json.variances, Float32Array);
        $.image_size_w = reader.value(json.image_size_w, 0);
        $.image_size_h = reader.value(json.image_size_h, 0);
        $.step_w = reader.value(json.step_w, 0);
        $.step_h = reader.value(json.step_h, 0);
        $.clip = reader.value(json.clip, false);
        $.flip = reader.value(json.flip, false);
        $.offset = reader.value(json.offset, 0);
        return $;
    }
};

$root.mindspore.schema.PReLUFusion = class PReLUFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PReLUFusion();
        $.channel_shared = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PReLUFusion();
        $.channel_shared = reader.value(json.channel_shared, false);
        return $;
    }
};

$root.mindspore.schema.Rank = class Rank {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Rank();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Rank();
        return $;
    }
};

$root.mindspore.schema.Range = class Range {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Range();
        $.d_type = reader.int64_(position, 4, 0);
        $.start = reader.int64_(position, 6, 0);
        $.limit = reader.int64_(position, 8, 0);
        $.delta = reader.int64_(position, 10, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Range();
        $.d_type = reader.value(json.d_type, 0);
        $.start = reader.value(json.start, 0);
        $.limit = reader.value(json.limit, 0);
        $.delta = reader.value(json.delta, 1);
        return $;
    }
};

$root.mindspore.schema.Reciprocal = class Reciprocal {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Reciprocal();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Reciprocal();
        return $;
    }
};

$root.mindspore.schema.RealDiv = class RealDiv {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.RealDiv();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.RealDiv();
        return $;
    }
};

$root.mindspore.schema.ReduceFusion = class ReduceFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ReduceFusion();
        $.keep_dims = reader.bool_(position, 4, false);
        $.mode = reader.int8_(position, 6, 0);
        $.reduce_to_end = reader.bool_(position, 8, false);
        $.coeff = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ReduceFusion();
        $.keep_dims = reader.value(json.keep_dims, false);
        $.mode = $root.mindspore.schema.ReduceMode[json.mode];
        $.reduce_to_end = reader.value(json.reduce_to_end, false);
        $.coeff = reader.value(json.coeff, 0);
        return $;
    }
};

$root.mindspore.schema.Reshape = class Reshape {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Reshape();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Reshape();
        return $;
    }
};

$root.mindspore.schema.Resize = class Resize {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Resize();
        $.format = reader.int32_(position, 4, 0);
        $.method = reader.int8_(position, 6, 0);
        $.new_height = reader.int64_(position, 8, 0);
        $.new_width = reader.int64_(position, 10, 0);
        $.preserve_aspect_ratio = reader.bool_(position, 12, false);
        $.coordinate_transform_mode = reader.int8_(position, 14, 0);
        $.cubic_coeff = reader.float32_(position, 16, 0);
        $.exclude_outside = reader.int64_(position, 18, 0);
        $.extrapolation_value = reader.float32_(position, 20, 0);
        $.nearest_mode = reader.int8_(position, 22, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Resize();
        $.format = $root.mindspore.schema.Format[json.format];
        $.method = $root.mindspore.schema.ResizeMethod[json.method];
        $.new_height = reader.value(json.new_height, 0);
        $.new_width = reader.value(json.new_width, 0);
        $.preserve_aspect_ratio = reader.value(json.preserve_aspect_ratio, false);
        $.coordinate_transform_mode = $root.mindspore.schema.CoordinateTransformMode[json.coordinate_transform_mode];
        $.cubic_coeff = reader.value(json.cubic_coeff, 0);
        $.exclude_outside = reader.value(json.exclude_outside, 0);
        $.extrapolation_value = reader.value(json.extrapolation_value, 0);
        $.nearest_mode = $root.mindspore.schema.NearestMode[json.nearest_mode];
        return $;
    }
};

$root.mindspore.schema.ReverseSequence = class ReverseSequence {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ReverseSequence();
        $.seq_dim = reader.int64_(position, 4, 0);
        $.batch_dim = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ReverseSequence();
        $.seq_dim = reader.value(json.seq_dim, 0);
        $.batch_dim = reader.value(json.batch_dim, 0);
        return $;
    }
};

$root.mindspore.schema.ReverseV2 = class ReverseV2 {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ReverseV2();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ReverseV2();
        $.axis = reader.array(json.axis);
        return $;
    }
};

$root.mindspore.schema.Rfft = class Rfft {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Rfft();
        $.fft_length = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Rfft();
        $.fft_length = reader.value(json.fft_length, 0);
        return $;
    }
};

$root.mindspore.schema.ROIPooling = class ROIPooling {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ROIPooling();
        $.pooled_h = reader.int64_(position, 4, 0);
        $.pooled_w = reader.int64_(position, 6, 0);
        $.scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ROIPooling();
        $.pooled_h = reader.value(json.pooled_h, 0);
        $.pooled_w = reader.value(json.pooled_w, 0);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

$root.mindspore.schema.Round = class Round {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Round();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Round();
        return $;
    }
};

$root.mindspore.schema.Rsqrt = class Rsqrt {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Rsqrt();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Rsqrt();
        return $;
    }
};

$root.mindspore.schema.QuantDTypeCast = class QuantDTypeCast {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.QuantDTypeCast();
        $.src_t = reader.int64_(position, 4, 0);
        $.dst_t = reader.int64_(position, 6, 0);
        $.axis = reader.int64_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.QuantDTypeCast();
        $.src_t = reader.value(json.src_t, 0);
        $.dst_t = reader.value(json.dst_t, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.ScaleFusion = class ScaleFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ScaleFusion();
        $.axis = reader.int64_(position, 4, 0);
        $.activation_type = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ScaleFusion();
        $.axis = reader.value(json.axis, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.ScatterNd = class ScatterNd {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.ScatterNd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.ScatterNd();
        return $;
    }
};

$root.mindspore.schema.SGD = class SGD {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SGD();
        $.nesterov = reader.bool_(position, 4, false);
        $.dampening = reader.float32_(position, 6, 0);
        $.weight_decay = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SGD();
        $.nesterov = reader.value(json.nesterov, false);
        $.dampening = reader.value(json.dampening, 0);
        $.weight_decay = reader.value(json.weight_decay, 0);
        return $;
    }
};

$root.mindspore.schema.Shape = class Shape {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Shape();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Shape();
        return $;
    }
};

$root.mindspore.schema.SigmoidCrossEntropyWithLogits = class SigmoidCrossEntropyWithLogits {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SigmoidCrossEntropyWithLogits();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SigmoidCrossEntropyWithLogits();
        return $;
    }
};

$root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad = class SigmoidCrossEntropyWithLogitsGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad();
        return $;
    }
};

$root.mindspore.schema.Sin = class Sin {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Sin();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Sin();
        return $;
    }
};

$root.mindspore.schema.SkipGram = class SkipGram {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SkipGram();
        $.include_all_grams = reader.bool_(position, 4, false);
        $.max_skip_size = reader.int64_(position, 6, 0);
        $.ngram_size = reader.int64_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SkipGram();
        $.include_all_grams = reader.value(json.include_all_grams, false);
        $.max_skip_size = reader.value(json.max_skip_size, 0);
        $.ngram_size = reader.value(json.ngram_size, 0);
        return $;
    }
};

$root.mindspore.schema.SliceFusion = class SliceFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SliceFusion();
        $.axes = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SliceFusion();
        $.axes = reader.array(json.axes);
        return $;
    }
};

$root.mindspore.schema.SmoothL1Loss = class SmoothL1Loss {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SmoothL1Loss();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SmoothL1Loss();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

$root.mindspore.schema.SmoothL1LossGrad = class SmoothL1LossGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SmoothL1LossGrad();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SmoothL1LossGrad();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

$root.mindspore.schema.Softmax = class Softmax {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Softmax();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Softmax();
        $.axis = reader.array(json.axis);
        return $;
    }
};

$root.mindspore.schema.SoftmaxCrossEntropyWithLogits = class SoftmaxCrossEntropyWithLogits {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SoftmaxCrossEntropyWithLogits();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SoftmaxCrossEntropyWithLogits();
        return $;
    }
};

$root.mindspore.schema.SpaceToBatch = class SpaceToBatch {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SpaceToBatch();
        $.block_size = reader.int64s_(position, 4);
        $.paddings = reader.table(position, 6, $root.mindspore.schema.Vec2D.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SpaceToBatch();
        $.block_size = reader.array(json.block_size);
        $.paddings = reader.object(json.paddings, $root.mindspore.schema.Vec2D.decodeText);
        return $;
    }
};

$root.mindspore.schema.SpaceToBatchND = class SpaceToBatchND {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SpaceToBatchND();
        $.block_shape = reader.int64s_(position, 4);
        $.paddings = reader.table(position, 6, $root.mindspore.schema.Vec2D.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SpaceToBatchND();
        $.block_shape = reader.array(json.block_shape);
        $.paddings = reader.object(json.paddings, $root.mindspore.schema.Vec2D.decodeText);
        return $;
    }
};

$root.mindspore.schema.SpaceToDepth = class SpaceToDepth {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SpaceToDepth();
        $.block_size = reader.int64_(position, 4, 0);
        $.format = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SpaceToDepth();
        $.block_size = reader.value(json.block_size, 0);
        $.format = $root.mindspore.schema.Format[json.format];
        return $;
    }
};

$root.mindspore.schema.SparseSoftmaxCrossEntropyWithLogits = class SparseSoftmaxCrossEntropyWithLogits {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SparseSoftmaxCrossEntropyWithLogits();
        $.is_grad = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SparseSoftmaxCrossEntropyWithLogits();
        $.is_grad = reader.value(json.is_grad, false);
        return $;
    }
};

$root.mindspore.schema.SparseToDense = class SparseToDense {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SparseToDense();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SparseToDense();
        return $;
    }
};

$root.mindspore.schema.Split = class Split {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Split();
        $.output_num = reader.int64_(position, 4, 0);
        $.size_splits = reader.int64s_(position, 6);
        $.axis = reader.int64_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Split();
        $.output_num = reader.value(json.output_num, 0);
        $.size_splits = reader.array(json.size_splits);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.Sqrt = class Sqrt {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Sqrt();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Sqrt();
        return $;
    }
};

$root.mindspore.schema.Squeeze = class Squeeze {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Squeeze();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Squeeze();
        $.axis = reader.array(json.axis);
        return $;
    }
};

$root.mindspore.schema.Square = class Square {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Square();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Square();
        return $;
    }
};

$root.mindspore.schema.SquaredDifference = class SquaredDifference {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SquaredDifference();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SquaredDifference();
        return $;
    }
};

$root.mindspore.schema.Stack = class Stack {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Stack();
        $.axis = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Stack();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.StridedSlice = class StridedSlice {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.StridedSlice();
        $.begin_mask = reader.int64_(position, 4, 0);
        $.end_mask = reader.int64_(position, 6, 0);
        $.ellipsis_mask = reader.int64_(position, 8, 0);
        $.new_axis_mask = reader.int64_(position, 10, 0);
        $.shrink_axis_mask = reader.int64_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.StridedSlice();
        $.begin_mask = reader.value(json.begin_mask, 0);
        $.end_mask = reader.value(json.end_mask, 0);
        $.ellipsis_mask = reader.value(json.ellipsis_mask, 0);
        $.new_axis_mask = reader.value(json.new_axis_mask, 0);
        $.shrink_axis_mask = reader.value(json.shrink_axis_mask, 0);
        return $;
    }
};

$root.mindspore.schema.SubFusion = class SubFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SubFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SubFusion();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

$root.mindspore.schema.SubGrad = class SubGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SubGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SubGrad();
        return $;
    }
};

$root.mindspore.schema.Switch = class Switch {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Switch();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Switch();
        return $;
    }
};

$root.mindspore.schema.TensorListFromTensor = class TensorListFromTensor {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListFromTensor();
        $.element_dtype = reader.int64_(position, 4, 0);
        $.shape_type = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListFromTensor();
        $.element_dtype = reader.value(json.element_dtype, 0);
        $.shape_type = reader.value(json.shape_type, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListGetItem = class TensorListGetItem {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListGetItem();
        $.element_dtype = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListGetItem();
        $.element_dtype = reader.value(json.element_dtype, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListReserve = class TensorListReserve {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListReserve();
        $.element_dtype = reader.int64_(position, 4, 0);
        $.shape_type = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListReserve();
        $.element_dtype = reader.value(json.element_dtype, 0);
        $.shape_type = reader.value(json.shape_type, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListSetItem = class TensorListSetItem {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListSetItem();
        $.element_dtype = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListSetItem();
        $.element_dtype = reader.value(json.element_dtype, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListStack = class TensorListStack {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListStack();
        $.num_elements = reader.int64_(position, 4, 0);
        $.element_dtype = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListStack();
        $.num_elements = reader.value(json.num_elements, 0);
        $.element_dtype = reader.value(json.element_dtype, 0);
        return $;
    }
};

$root.mindspore.schema.TileFusion = class TileFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TileFusion();
        $.dims = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TileFusion();
        $.dims = reader.array(json.dims);
        return $;
    }
};

$root.mindspore.schema.TopKFusion = class TopKFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TopKFusion();
        $.sorted = reader.bool_(position, 4, true);
        $.axis = reader.int64_(position, 6, 0);
        $.largest = reader.int64_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TopKFusion();
        $.sorted = reader.value(json.sorted, true);
        $.axis = reader.value(json.axis, 0);
        $.largest = reader.value(json.largest, 0);
        return $;
    }
};

$root.mindspore.schema.Transpose = class Transpose {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Transpose();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Transpose();
        return $;
    }
};

$root.mindspore.schema.Unique = class Unique {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Unique();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Unique();
        return $;
    }
};

$root.mindspore.schema.UnsortedSegmentSum = class UnsortedSegmentSum {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.UnsortedSegmentSum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.UnsortedSegmentSum();
        return $;
    }
};

$root.mindspore.schema.Unsqueeze = class Unsqueeze {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Unsqueeze();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Unsqueeze();
        $.axis = reader.array(json.axis);
        return $;
    }
};

$root.mindspore.schema.Unstack = class Unstack {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Unstack();
        $.axis = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Unstack();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.Where = class Where {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Where();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Where();
        return $;
    }
};

$root.mindspore.schema.ZerosLike = class ZerosLike {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.ZerosLike();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.ZerosLike();
        return $;
    }
};

$root.mindspore.schema.Select = class Select {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Select();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Select();
        return $;
    }
};

$root.mindspore.schema.GRU = class GRU {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GRU();
        $.bidirectional = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GRU();
        $.bidirectional = reader.value(json.bidirectional, false);
        return $;
    }
};

$root.mindspore.schema.NonZero = class NonZero {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.NonZero();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.NonZero();
        return $;
    }
};

$root.mindspore.schema.InvertPermutation = class InvertPermutation {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.InvertPermutation();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.InvertPermutation();
        return $;
    }
};

$root.mindspore.schema.Size = class Size {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Size();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Size();
        return $;
    }
};

$root.mindspore.schema.RandomStandardNormal = class RandomStandardNormal {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.RandomStandardNormal();
        $.seed = reader.int64_(position, 4, 0);
        $.seed2 = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.RandomStandardNormal();
        $.seed = reader.value(json.seed, 0);
        $.seed2 = reader.value(json.seed2, 0);
        return $;
    }
};

$root.mindspore.schema.CropAndResize = class CropAndResize {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.CropAndResize();
        $.method = reader.int8_(position, 4, 0);
        $.extrapolation_value = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.CropAndResize();
        $.method = $root.mindspore.schema.ResizeMethod[json.method];
        $.extrapolation_value = reader.value(json.extrapolation_value, 0);
        return $;
    }
};

$root.mindspore.schema.Erf = class Erf {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Erf();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Erf();
        return $;
    }
};

$root.mindspore.schema.StridedSliceGrad = class StridedSliceGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.StridedSliceGrad();
        $.begin_mask = reader.int64_(position, 4, 0);
        $.end_mask = reader.int64_(position, 6, 0);
        $.ellipsis_mask = reader.int64_(position, 8, 0);
        $.new_axis_mask = reader.int64_(position, 10, 0);
        $.shrink_axis_mask = reader.int64_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.StridedSliceGrad();
        $.begin_mask = reader.value(json.begin_mask, 0);
        $.end_mask = reader.value(json.end_mask, 0);
        $.ellipsis_mask = reader.value(json.ellipsis_mask, 0);
        $.new_axis_mask = reader.value(json.new_axis_mask, 0);
        $.shrink_axis_mask = reader.value(json.shrink_axis_mask, 0);
        return $;
    }
};

$root.mindspore.schema.IsFinite = class IsFinite {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.IsFinite();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.IsFinite();
        return $;
    }
};

$root.mindspore.schema.LinSpace = class LinSpace {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LinSpace();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LinSpace();
        return $;
    }
};

$root.mindspore.schema.UniformReal = class UniformReal {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.UniformReal();
        $.seed = reader.int64_(position, 4, 0);
        $.seed2 = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.UniformReal();
        $.seed = reader.value(json.seed, 0);
        $.seed2 = reader.value(json.seed2, 0);
        return $;
    }
};

$root.mindspore.schema.AbsGrad = class AbsGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.AbsGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.AbsGrad();
        return $;
    }
};

$root.mindspore.schema.RsqrtGrad = class RsqrtGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.RsqrtGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.RsqrtGrad();
        return $;
    }
};

$root.mindspore.schema.SqrtGrad = class SqrtGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SqrtGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SqrtGrad();
        return $;
    }
};

$root.mindspore.schema.LayerNormGrad = class LayerNormGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LayerNormGrad();
        $.begin_norm_axis = reader.int64_(position, 4, 0);
        $.begin_params_axis = reader.int64_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LayerNormGrad();
        $.begin_norm_axis = reader.value(json.begin_norm_axis, 0);
        $.begin_params_axis = reader.value(json.begin_params_axis, 0);
        return $;
    }
};

$root.mindspore.schema.ResizeGrad = class ResizeGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ResizeGrad();
        $.method = reader.int8_(position, 4, 0);
        $.align_corners = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ResizeGrad();
        $.method = $root.mindspore.schema.ResizeMethod[json.method];
        $.align_corners = reader.value(json.align_corners, false);
        return $;
    }
};

$root.mindspore.schema.Splice = class Splice {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Splice();
        $.context = reader.int64s_(position, 4);
        $.forward_indexes = reader.int64s_(position, 6);
        $.output_dim = reader.int64_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Splice();
        $.context = reader.array(json.context);
        $.forward_indexes = reader.array(json.forward_indexes);
        $.output_dim = reader.value(json.output_dim, 0);
        return $;
    }
};

$root.mindspore.schema.LogSoftmax = class LogSoftmax {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LogSoftmax();
        $.axis = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LogSoftmax();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.Call = class Call {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Call();
        $.is_tail_call = reader.bool_(position, 4, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Call();
        $.is_tail_call = reader.value(json.is_tail_call, true);
        return $;
    }
};

$root.mindspore.schema.CumSum = class CumSum {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.CumSum();
        $.exclusive = reader.bool_(position, 4, false);
        $.reverse = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.CumSum();
        $.exclusive = reader.value(json.exclusive, false);
        $.reverse = reader.value(json.reverse, false);
        return $;
    }
};

$root.mindspore.schema.Custom = class Custom {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Custom();
        $.type = reader.string_(position, 4, null);
        $.attr = reader.tableArray(position, 6, $root.mindspore.schema.Attribute.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Custom();
        $.type = reader.value(json.type, null);
        $.attr = reader.objectArray(json.attr, $root.mindspore.schema.Attribute.decodeText);
        return $;
    }
};

$root.mindspore.schema.SplitWithOverlap = class SplitWithOverlap {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SplitWithOverlap();
        $.split_dim = reader.int64_(position, 4, 0);
        $.number_split = reader.int64_(position, 6, 0);
        $.ratio = reader.int64s_(position, 8);
        $.extend_top = reader.int64s_(position, 10);
        $.extend_bottom = reader.int64s_(position, 12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SplitWithOverlap();
        $.split_dim = reader.value(json.split_dim, 0);
        $.number_split = reader.value(json.number_split, 0);
        $.ratio = reader.array(json.ratio);
        $.extend_top = reader.array(json.extend_top);
        $.extend_bottom = reader.array(json.extend_bottom);
        return $;
    }
};

$root.mindspore.schema.GenOP = class GenOP {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GenOP();
        $.activation_type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        $.min_val = reader.float32_(position, 8, 0);
        $.max_val = reader.float32_(position, 10, 0);
        $.is_training = reader.bool_(position, 12, false);
        $.format = reader.int32_(position, 14, 0);
        $.kernel_size = reader.int64s_(position, 16);
        $.stride = reader.int64s_(position, 18);
        $.dilation = reader.int64s_(position, 20);
        $.pad_mode = reader.int8_(position, 22, 0);
        $.pad_list = reader.int64s_(position, 24);
        $.mode = reader.int64_(position, 26, 0);
        $.group = reader.int64_(position, 28, 0);
        $.in_channel = reader.int64_(position, 30, 0);
        $.out_channel = reader.int64_(position, 32, 0);
        $.eltwise_mode = reader.int8_(position, 34, 0);
        $.has_bias = reader.bool_(position, 36, false);
        $.use_axis = reader.bool_(position, 38, false);
        $.axis = reader.int64_(position, 40, 0);
        $.epsilon = reader.float32_(position, 42, 0.0001);
        $.momentum = reader.float32_(position, 44, 0.9);
        $.transpose_a = reader.bool_(position, 46, false);
        $.transpose_b = reader.bool_(position, 48, false);
        $.pad = reader.int64s_(position, 50);
        $.round_mode = reader.int8_(position, 52, 0);
        $.global = reader.bool_(position, 54, false);
        $.channel_shared = reader.bool_(position, 56, false);
        $.axes = reader.int64s_(position, 58);
        $.keep_dims = reader.bool_(position, 60, false);
        $.reduce_mode = reader.int8_(position, 62, 0);
        $.reduce_to_end = reader.bool_(position, 64, false);
        $.coeff = reader.float32_(position, 66, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GenOP();
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        $.alpha = reader.value(json.alpha, 0);
        $.min_val = reader.value(json.min_val, 0);
        $.max_val = reader.value(json.max_val, 0);
        $.is_training = reader.value(json.is_training, false);
        $.format = $root.mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = $root.mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.value(json.mode, 0);
        $.group = reader.value(json.group, 0);
        $.in_channel = reader.value(json.in_channel, 0);
        $.out_channel = reader.value(json.out_channel, 0);
        $.eltwise_mode = $root.mindspore.schema.EltwiseMode[json.eltwise_mode];
        $.has_bias = reader.value(json.has_bias, false);
        $.use_axis = reader.value(json.use_axis, false);
        $.axis = reader.value(json.axis, 0);
        $.epsilon = reader.value(json.epsilon, 0.0001);
        $.momentum = reader.value(json.momentum, 0.9);
        $.transpose_a = reader.value(json.transpose_a, false);
        $.transpose_b = reader.value(json.transpose_b, false);
        $.pad = reader.array(json.pad);
        $.round_mode = $root.mindspore.schema.RoundMode[json.round_mode];
        $.global = reader.value(json.global, false);
        $.channel_shared = reader.value(json.channel_shared, false);
        $.axes = reader.array(json.axes);
        $.keep_dims = reader.value(json.keep_dims, false);
        $.reduce_mode = $root.mindspore.schema.ReduceMode[json.reduce_mode];
        $.reduce_to_end = reader.value(json.reduce_to_end, false);
        $.coeff = reader.value(json.coeff, 0);
        return $;
    }
};

$root.mindspore.schema.RaggedRange = class RaggedRange {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.RaggedRange();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.RaggedRange();
        return $;
    }
};

$root.mindspore.schema.GLU = class GLU {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GLU();
        $.axis = reader.int64_(position, 4, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GLU();
        $.axis = reader.value(json.axis, -1);
        return $;
    }
};

$root.mindspore.schema.TensorArray = class TensorArray {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorArray();
        $.dynamic_size = reader.bool_(position, 4, false);
        $.identical_element_shapes = reader.bool_(position, 6, false);
        $.element_shape = reader.typedArray(position, 8, Int32Array);
        $.data_type = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorArray();
        $.dynamic_size = reader.value(json.dynamic_size, false);
        $.identical_element_shapes = reader.value(json.identical_element_shapes, false);
        $.element_shape = reader.typedArray(json.element_shape, Int32Array);
        $.data_type = reader.value(json.data_type, 0);
        return $;
    }
};

$root.mindspore.schema.TensorArrayRead = class TensorArrayRead {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.TensorArrayRead();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.TensorArrayRead();
        return $;
    }
};

$root.mindspore.schema.TensorArrayWrite = class TensorArrayWrite {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.TensorArrayWrite();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.TensorArrayWrite();
        return $;
    }
};

$root.mindspore.schema.Affine = class Affine {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Affine();
        $.context = reader.int64s_(position, 4);
        $.output_dim = reader.int64_(position, 6, 0);
        $.activation_type = reader.int8_(position, 8, 0);
        $.transpose_a = reader.bool_(position, 10, false);
        $.transpose_b = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Affine();
        $.context = reader.array(json.context);
        $.output_dim = reader.value(json.output_dim, 0);
        $.activation_type = $root.mindspore.schema.ActivationType[json.activation_type];
        $.transpose_a = reader.value(json.transpose_a, false);
        $.transpose_b = reader.value(json.transpose_b, false);
        return $;
    }
};

$root.mindspore.schema.ScatterNdUpdate = class ScatterNdUpdate {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.ScatterNdUpdate();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.ScatterNdUpdate();
        return $;
    }
};

$root.mindspore.schema.AllGather = class AllGather {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AllGather();
        $.group = reader.string_(position, 4, null);
        $.rank_size = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AllGather();
        $.group = reader.value(json.group, null);
        $.rank_size = reader.value(json.rank_size, 0);
        return $;
    }
};

$root.mindspore.schema.ReduceScatter = class ReduceScatter {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ReduceScatter();
        $.group = reader.string_(position, 4, null);
        $.mode = reader.int8_(position, 6, 0);
        $.rank_size = reader.int32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ReduceScatter();
        $.group = reader.value(json.group, null);
        $.mode = $root.mindspore.schema.ReduceMode[json.mode];
        $.rank_size = reader.value(json.rank_size, 0);
        return $;
    }
};

$root.mindspore.schema.DynamicQuant = class DynamicQuant {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DynamicQuant();
        $.symmetric = reader.bool_(position, 4, false);
        $.dst_type = reader.int64_(position, 6, 32);
        $.activation_channel = reader.bool_(position, 8, false);
        $.prefer_axis = reader.int64_(position, 10, 0);
        $.transpose = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DynamicQuant();
        $.symmetric = reader.value(json.symmetric, false);
        $.dst_type = reader.value(json.dst_type, 32);
        $.activation_channel = reader.value(json.activation_channel, false);
        $.prefer_axis = reader.value(json.prefer_axis, 0);
        $.transpose = reader.value(json.transpose, false);
        return $;
    }
};

$root.mindspore.schema.LSTMGradData = class LSTMGradData {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LSTMGradData();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0);
        $.hidden_size = reader.int64_(position, 10, 0);
        $.num_layers = reader.int64_(position, 12, 0);
        $.num_directions = reader.int64_(position, 14, 0);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LSTMGradData();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.value(json.input_size, 0);
        $.hidden_size = reader.value(json.hidden_size, 0);
        $.num_layers = reader.value(json.num_layers, 0);
        $.num_directions = reader.value(json.num_directions, 0);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

$root.mindspore.schema.LSTMGradWeight = class LSTMGradWeight {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LSTMGradWeight();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0);
        $.hidden_size = reader.int64_(position, 10, 0);
        $.num_layers = reader.int64_(position, 12, 0);
        $.num_directions = reader.int64_(position, 14, 0);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LSTMGradWeight();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.value(json.input_size, 0);
        $.hidden_size = reader.value(json.hidden_size, 0);
        $.num_layers = reader.value(json.num_layers, 0);
        $.num_directions = reader.value(json.num_directions, 0);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

$root.mindspore.schema.RandomNormal = class RandomNormal {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.RandomNormal();
        $.seed = reader.float32_(position, 4, 0);
        $.mean = reader.float32_(position, 6, 0);
        $.scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.RandomNormal();
        $.seed = reader.value(json.seed, 0);
        $.mean = reader.value(json.mean, 0);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

$root.mindspore.schema.NLLLoss = class NLLLoss {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.NLLLoss();
        $.reduction = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.NLLLoss();
        $.reduction = $root.mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

$root.mindspore.schema.NLLLossGrad = class NLLLossGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.NLLLossGrad();
        $.reduction = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.NLLLossGrad();
        $.reduction = $root.mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

$root.mindspore.schema.FormatTranspose = class FormatTranspose {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FormatTranspose();
        $.src_format = reader.int32_(position, 4, 1);
        $.dst_format = reader.int32_(position, 6, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FormatTranspose();
        $.src_format = $root.mindspore.schema.Format[json.src_format];
        $.dst_format = $root.mindspore.schema.Format[json.dst_format];
        return $;
    }
};

$root.mindspore.schema.GatherD = class GatherD {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.GatherD();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.GatherD();
        return $;
    }
};

$root.mindspore.schema.GroupNormFusion = class GroupNormFusion {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GroupNormFusion();
        $.num_groups = reader.int64_(position, 4, 0);
        $.epsilon = reader.float32_(position, 6, 0.00001);
        $.affine = reader.bool_(position, 8, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GroupNormFusion();
        $.num_groups = reader.value(json.num_groups, 0);
        $.epsilon = reader.value(json.epsilon, 0.00001);
        $.affine = reader.value(json.affine, true);
        return $;
    }
};

$root.mindspore.schema.Log1p = class Log1p {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Log1p();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Log1p();
        return $;
    }
};

$root.mindspore.schema.TensorScatterAdd = class TensorScatterAdd {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.TensorScatterAdd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.TensorScatterAdd();
        return $;
    }
};

$root.mindspore.schema.SparseFillEmptyRows = class SparseFillEmptyRows {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SparseFillEmptyRows();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SparseFillEmptyRows();
        return $;
    }
};

$root.mindspore.schema.SparseReshape = class SparseReshape {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SparseReshape();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SparseReshape();
        return $;
    }
};

$root.mindspore.schema.SparseSegmentSum = class SparseSegmentSum {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SparseSegmentSum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SparseSegmentSum();
        return $;
    }
};

$root.mindspore.schema.ScatterElements = class ScatterElements {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ScatterElements();
        $.axis = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ScatterElements();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.Triu = class Triu {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Triu();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Triu();
        return $;
    }
};

$root.mindspore.schema.Tril = class Tril {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Tril();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Tril();
        return $;
    }
};

$root.mindspore.schema.AdamWeightDecay = class AdamWeightDecay {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AdamWeightDecay();
        $.use_locking = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AdamWeightDecay();
        $.use_locking = reader.value(json.use_locking, false);
        return $;
    }
};

$root.mindspore.schema.QuantParam = class QuantParam {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.QuantParam();
        $.scale = reader.float64_(position, 4, 1);
        $.zeroPoint = reader.int32_(position, 6, 0);
        $.min = reader.float64_(position, 8, 0);
        $.max = reader.float64_(position, 10, 0);
        $.narrowRange = reader.bool_(position, 12, true);
        $.numBits = reader.int32_(position, 14, 8);
        $.inited = reader.bool_(position, 16, false);
        $.varCorr = reader.float32_(position, 18, 1);
        $.meanCorr = reader.float32_(position, 20, 0);
        $.dstDtype = reader.int32_(position, 22, 32);
        $.roundType = reader.int32_(position, 24, 1);
        $.multiplier = reader.int32_(position, 26, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.QuantParam();
        $.scale = reader.value(json.scale, 1);
        $.zeroPoint = reader.value(json.zeroPoint, 0);
        $.min = reader.value(json.min, 0);
        $.max = reader.value(json.max, 0);
        $.narrowRange = reader.value(json.narrowRange, true);
        $.numBits = reader.value(json.numBits, 8);
        $.inited = reader.value(json.inited, false);
        $.varCorr = reader.value(json.varCorr, 1);
        $.meanCorr = reader.value(json.meanCorr, 0);
        $.dstDtype = reader.value(json.dstDtype, 32);
        $.roundType = reader.value(json.roundType, 1);
        $.multiplier = reader.value(json.multiplier, 1);
        return $;
    }
};

$root.mindspore.schema.WeightQuantCompressType = {
    NONE: 0,
    INDEXING: 1,
    SPARSE: 2,
    FSE: 3,
    BITPACKING: 4,
    FSE_INT: 5,
    FSE_INFER: 6
};

$root.mindspore.schema.ExternalData = class ExternalData {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ExternalData();
        $.checkSum = reader.string_(position, 4, null);
        $.location = reader.string_(position, 6, null);
        $.offset = reader.int64_(position, 8, 0);
        $.length = reader.int64_(position, 10, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ExternalData();
        $.checkSum = reader.value(json.checkSum, null);
        $.location = reader.value(json.location, null);
        $.offset = reader.value(json.offset, 0);
        $.length = reader.value(json.length, -1);
        return $;
    }
};

$root.mindspore.schema.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Tensor();
        $.nodeType = reader.int32_(position, 4, 0);
        $.dataType = reader.int32_(position, 6, 0);
        $.dims = reader.typedArray(position, 8, Int32Array);
        $.format = reader.int32_(position, 10, 0);
        $.refCount = reader.int32_(position, 12, 0);
        $.offset = reader.int32_(position, 14, 0);
        $.data = reader.typedArray(position, 16, Uint8Array);
        $.quantParams = reader.tableArray(position, 18, $root.mindspore.schema.QuantParam.decode);
        $.quantClusters = reader.typedArray(position, 20, Float32Array);
        $.name = reader.string_(position, 22, null);
        $.enableHuffmanCode = reader.bool_(position, 24, false);
        $.weightQuantCompressType = reader.int32_(position, 26, 0);
        $.externalData = reader.tableArray(position, 28, $root.mindspore.schema.ExternalData.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Tensor();
        $.nodeType = reader.value(json.nodeType, 0);
        $.dataType = reader.value(json.dataType, 0);
        $.dims = reader.typedArray(json.dims, Int32Array);
        $.format = $root.mindspore.schema.Format[json.format];
        $.refCount = reader.value(json.refCount, 0);
        $.offset = reader.value(json.offset, 0);
        $.data = reader.typedArray(json.data, Uint8Array);
        $.quantParams = reader.objectArray(json.quantParams, $root.mindspore.schema.QuantParam.decodeText);
        $.quantClusters = reader.typedArray(json.quantClusters, Float32Array);
        $.name = reader.value(json.name, null);
        $.enableHuffmanCode = reader.value(json.enableHuffmanCode, false);
        $.weightQuantCompressType = $root.mindspore.schema.WeightQuantCompressType[json.weightQuantCompressType];
        $.externalData = reader.objectArray(json.externalData, $root.mindspore.schema.ExternalData.decodeText);
        return $;
    }
};

$root.mindspore.schema.QuantType = {
    QUANT_NONE: 0,
    AwareTraining: 1,
    WeightQuant: 2,
    PostTraining: 3,
    QUANT_WEIGHT: 4,
    QUANT_ALL: 5,
    QUANT_DYNAMIC: 6
};

$root.mindspore.schema.Primitive = class Primitive {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Primitive();
        $.value = reader.union(position, 4, $root.mindspore.schema.PrimitiveType.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Primitive();
        $.value = $root.mindspore.schema.PrimitiveType.decodeText(reader, json.value, json.value_type);
        return $;
    }
};

$root.mindspore.schema.CNode = class CNode {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.CNode();
        $.name = reader.string_(position, 4, null);
        $.nodeType = reader.int32_(position, 6, 0);
        $.primitive = reader.table(position, 8, $root.mindspore.schema.Primitive.decode);
        $.inputIndex = reader.typedArray(position, 10, Uint32Array);
        $.outputIndex = reader.typedArray(position, 12, Uint32Array);
        $.quantType = reader.int32_(position, 14, 0);
        $.deviceType = reader.int32_(position, 16, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.CNode();
        $.name = reader.value(json.name, null);
        $.nodeType = reader.value(json.nodeType, 0);
        $.primitive = reader.object(json.primitive, $root.mindspore.schema.Primitive.decodeText);
        $.inputIndex = reader.typedArray(json.inputIndex, Uint32Array);
        $.outputIndex = reader.typedArray(json.outputIndex, Uint32Array);
        $.quantType = $root.mindspore.schema.QuantType[json.quantType];
        $.deviceType = reader.value(json.deviceType, -1);
        return $;
    }
};

$root.mindspore.schema.SubGraph = class SubGraph {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SubGraph();
        $.name = reader.string_(position, 4, null);
        $.inputIndices = reader.typedArray(position, 6, Uint32Array);
        $.outputIndices = reader.typedArray(position, 8, Uint32Array);
        $.nodeIndices = reader.typedArray(position, 10, Uint32Array);
        $.tensorIndices = reader.typedArray(position, 12, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SubGraph();
        $.name = reader.value(json.name, null);
        $.inputIndices = reader.typedArray(json.inputIndices, Uint32Array);
        $.outputIndices = reader.typedArray(json.outputIndices, Uint32Array);
        $.nodeIndices = reader.typedArray(json.nodeIndices, Uint32Array);
        $.tensorIndices = reader.typedArray(json.tensorIndices, Uint32Array);
        return $;
    }
};

$root.mindspore.schema.MetaGraph = class MetaGraph {

    static identifier(reader) {
        return reader.identifier === 'MSL2';
    }

    static create(reader) {
        return $root.mindspore.schema.MetaGraph.decode(reader, reader.root);
    }

    static createText(reader) {
        return $root.mindspore.schema.MetaGraph.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MetaGraph();
        $.name = reader.string_(position, 4, null);
        $.version = reader.string_(position, 6, null);
        $.fmkType = reader.int32_(position, 8, 0);
        $.inputIndex = reader.typedArray(position, 10, Uint32Array);
        $.outputIndex = reader.typedArray(position, 12, Uint32Array);
        $.mempoolSize = reader.uint32_(position, 14, 0);
        $.nodes = reader.tableArray(position, 16, $root.mindspore.schema.CNode.decode);
        $.allTensors = reader.tableArray(position, 18, $root.mindspore.schema.Tensor.decode);
        $.subGraph = reader.tableArray(position, 20, $root.mindspore.schema.SubGraph.decode);
        $.obfuscate = reader.bool_(position, 22, false);
        $.obfMetaData = reader.typedArray(position, 24, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MetaGraph();
        $.name = reader.value(json.name, null);
        $.version = reader.value(json.version, null);
        $.fmkType = reader.value(json.fmkType, 0);
        $.inputIndex = reader.typedArray(json.inputIndex, Uint32Array);
        $.outputIndex = reader.typedArray(json.outputIndex, Uint32Array);
        $.mempoolSize = reader.value(json.mempoolSize, 0);
        $.nodes = reader.objectArray(json.nodes, $root.mindspore.schema.CNode.decodeText);
        $.allTensors = reader.objectArray(json.allTensors, $root.mindspore.schema.Tensor.decodeText);
        $.subGraph = reader.objectArray(json.subGraph, $root.mindspore.schema.SubGraph.decodeText);
        $.obfuscate = reader.value(json.obfuscate, false);
        $.obfMetaData = reader.typedArray(json.obfMetaData, Uint8Array);
        return $;
    }
};
