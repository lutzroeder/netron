
export const mindspore = {};

mindspore.schema = mindspore.schema || {};

mindspore.schema.ResizeMethod = {
    UNKNOWN: -1,
    LINEAR: 0,
    NEAREST: 1,
    CUBIC: 2
};

mindspore.schema.CoordinateTransformMode = {
    ASYMMETRIC: 0,
    ALIGN_CORNERS: 1,
    HALF_PIXEL: 2
};

mindspore.schema.NearestMode = {
    NORMAL: 0,
    ROUND_HALF_DOWN: 1,
    ROUND_HALF_UP: 2,
    FLOOR: 3,
    CEIL: 4
};

mindspore.schema.Format = {
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

mindspore.schema.ActivationType = {
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
    FAST_GELU: 20,
    UNKNOWN: 21
};

mindspore.schema.ReduceMode = {
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

mindspore.schema.PoolMode = {
    MAX_POOLING: 0,
    MEAN_POOLING: 1
};

mindspore.schema.EltwiseMode = {
    PROD: 0,
    SUM: 1,
    MAXIMUM: 2,
    UNKNOWN: 3
};

mindspore.schema.PadMode = {
    PAD: 0,
    SAME: 1,
    VALID: 2
};

mindspore.schema.RoundMode = {
    FLOOR: 0,
    CEIL: 1
};

mindspore.schema.PaddingMode = {
    CONSTANT: 0,
    REFLECT: 1,
    SYMMETRIC: 2,
    MODE_RESERVED: 3
};

mindspore.schema.LshProjectionType = {
    UNKNOWN: 0,
    SPARSE: 1,
    DENSE: 2
};

mindspore.schema.Reduction = {
    REDUCTION_SUM: 0,
    MEAN: 1,
    NONE: 2
};

mindspore.schema.Vec = class Vec {

    static decode(reader, position) {
        const $ = new mindspore.schema.Vec();
        $.data = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Vec();
        $.data = reader.array(json.data);
        return $;
    }
};

mindspore.schema.Vec2D = class Vec2D {

    static decode(reader, position) {
        const $ = new mindspore.schema.Vec2D();
        $.data = reader.tables(position, 4, mindspore.schema.Vec);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Vec2D();
        $.data = reader.objects(json.data, mindspore.schema.Vec);
        return $;
    }
};

mindspore.schema.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new mindspore.schema.Attribute();
        $.name = reader.string_(position, 4, null);
        $.data = reader.array(position, 6, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Attribute();
        $.name = reader.value(json.name, null);
        $.data = reader.array(json.data, Uint8Array);
        return $;
    }
};

mindspore.schema.PrimitiveType = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return mindspore.schema.Abs.decode(reader, position);
            case 2: return mindspore.schema.Activation.decode(reader, position);
            case 3: return mindspore.schema.ActivationGrad.decode(reader, position);
            case 4: return mindspore.schema.Adam.decode(reader, position);
            case 5: return mindspore.schema.AddFusion.decode(reader, position);
            case 6: return mindspore.schema.AdderFusion.decode(reader, position);
            case 7: return mindspore.schema.AddGrad.decode(reader, position);
            case 8: return mindspore.schema.AddN.decode(reader, position);
            case 9: return mindspore.schema.All.decode(reader, position);
            case 10: return mindspore.schema.ApplyMomentum.decode(reader, position);
            case 11: return mindspore.schema.ArgMaxFusion.decode(reader, position);
            case 12: return mindspore.schema.ArgMinFusion.decode(reader, position);
            case 13: return mindspore.schema.Assert.decode(reader, position);
            case 14: return mindspore.schema.Assign.decode(reader, position);
            case 15: return mindspore.schema.AssignAdd.decode(reader, position);
            case 16: return mindspore.schema.AudioSpectrogram.decode(reader, position);
            case 17: return mindspore.schema.AvgPoolFusion.decode(reader, position);
            case 18: return mindspore.schema.AvgPoolGrad.decode(reader, position);
            case 19: return mindspore.schema.BatchNorm.decode(reader, position);
            case 20: return mindspore.schema.BatchNormGrad.decode(reader, position);
            case 21: return mindspore.schema.BatchToSpace.decode(reader, position);
            case 22: return mindspore.schema.BatchToSpaceND.decode(reader, position);
            case 23: return mindspore.schema.BiasAdd.decode(reader, position);
            case 24: return mindspore.schema.BinaryCrossEntropy.decode(reader, position);
            case 25: return mindspore.schema.BinaryCrossEntropyGrad.decode(reader, position);
            case 26: return mindspore.schema.BiasAddGrad.decode(reader, position);
            case 27: return mindspore.schema.BroadcastTo.decode(reader, position);
            case 28: return mindspore.schema.Cast.decode(reader, position);
            case 29: return mindspore.schema.Ceil.decode(reader, position);
            case 30: return mindspore.schema.Clip.decode(reader, position);
            case 31: return mindspore.schema.Concat.decode(reader, position);
            case 32: return mindspore.schema.Attention.decode(reader, position);
            case 33: return mindspore.schema.Conv2DBackpropFilterFusion.decode(reader, position);
            case 34: return mindspore.schema.Conv2DBackpropInputFusion.decode(reader, position);
            case 35: return mindspore.schema.Conv2DFusion.decode(reader, position);
            case 36: return mindspore.schema.Conv2dTransposeFusion.decode(reader, position);
            case 37: return mindspore.schema.Cos.decode(reader, position);
            case 38: return mindspore.schema.ConstantOfShape.decode(reader, position);
            case 39: return mindspore.schema.Crop.decode(reader, position);
            case 40: return mindspore.schema.CustomExtractFeatures.decode(reader, position);
            case 41: return mindspore.schema.CustomNormalize.decode(reader, position);
            case 42: return mindspore.schema.CustomPredict.decode(reader, position);
            case 43: return mindspore.schema.DeConv2DGradFilter.decode(reader, position);
            case 44: return mindspore.schema.Depend.decode(reader, position);
            case 45: return mindspore.schema.DepthToSpace.decode(reader, position);
            case 46: return mindspore.schema.DetectionPostProcess.decode(reader, position);
            case 47: return mindspore.schema.DivFusion.decode(reader, position);
            case 48: return mindspore.schema.DivGrad.decode(reader, position);
            case 49: return mindspore.schema.Dropout.decode(reader, position);
            case 50: return mindspore.schema.DropoutGrad.decode(reader, position);
            case 51: return mindspore.schema.Elu.decode(reader, position);
            case 52: return mindspore.schema.Eltwise.decode(reader, position);
            case 53: return mindspore.schema.Equal.decode(reader, position);
            case 54: return mindspore.schema.EmbeddingLookupFusion.decode(reader, position);
            case 55: return mindspore.schema.ExpFusion.decode(reader, position);
            case 56: return mindspore.schema.ExpandDims.decode(reader, position);
            case 57: return mindspore.schema.FakeQuantWithMinMaxVars.decode(reader, position);
            case 58: return mindspore.schema.FakeQuantWithMinMaxVarsPerChannel.decode(reader, position);
            case 59: return mindspore.schema.FftReal.decode(reader, position);
            case 60: return mindspore.schema.FftImag.decode(reader, position);
            case 61: return mindspore.schema.Flatten.decode(reader, position);
            case 62: return mindspore.schema.FlattenGrad.decode(reader, position);
            case 63: return mindspore.schema.Floor.decode(reader, position);
            case 64: return mindspore.schema.FloorDiv.decode(reader, position);
            case 65: return mindspore.schema.FloorMod.decode(reader, position);
            case 66: return mindspore.schema.Fill.decode(reader, position);
            case 67: return mindspore.schema.FullConnection.decode(reader, position);
            case 68: return mindspore.schema.FusedBatchNorm.decode(reader, position);
            case 69: return mindspore.schema.Gather.decode(reader, position);
            case 70: return mindspore.schema.GatherNd.decode(reader, position);
            case 71: return mindspore.schema.Greater.decode(reader, position);
            case 72: return mindspore.schema.GreaterEqual.decode(reader, position);
            case 73: return mindspore.schema.HashtableLookup.decode(reader, position);
            case 74: return mindspore.schema.InstanceNorm.decode(reader, position);
            case 75: return mindspore.schema.LayerNormFusion.decode(reader, position);
            case 76: return mindspore.schema.LeakyRelu.decode(reader, position);
            case 77: return mindspore.schema.Less.decode(reader, position);
            case 78: return mindspore.schema.LessEqual.decode(reader, position);
            case 79: return mindspore.schema.Log.decode(reader, position);
            case 80: return mindspore.schema.LogGrad.decode(reader, position);
            case 81: return mindspore.schema.LogicalAnd.decode(reader, position);
            case 82: return mindspore.schema.LogicalNot.decode(reader, position);
            case 83: return mindspore.schema.LogicalOr.decode(reader, position);
            case 84: return mindspore.schema.LpNormalization.decode(reader, position);
            case 85: return mindspore.schema.LRN.decode(reader, position);
            case 86: return mindspore.schema.LshProjection.decode(reader, position);
            case 87: return mindspore.schema.LSTM.decode(reader, position);
            case 88: return mindspore.schema.L2NormalizeFusion.decode(reader, position);
            case 89: return mindspore.schema.MatMulFusion.decode(reader, position);
            case 90: return mindspore.schema.Maximum.decode(reader, position);
            case 91: return mindspore.schema.MaximumGrad.decode(reader, position);
            case 92: return mindspore.schema.MaxPoolFusion.decode(reader, position);
            case 93: return mindspore.schema.MaxPoolGrad.decode(reader, position);
            case 94: return mindspore.schema.SwitchLayer.decode(reader, position);
            case 95: return mindspore.schema.Mfcc.decode(reader, position);
            case 96: return mindspore.schema.Minimum.decode(reader, position);
            case 97: return mindspore.schema.MinimumGrad.decode(reader, position);
            case 98: return mindspore.schema.Mod.decode(reader, position);
            case 99: return mindspore.schema.MulFusion.decode(reader, position);
            case 100: return mindspore.schema.MulGrad.decode(reader, position);
            case 101: return mindspore.schema.Neg.decode(reader, position);
            case 102: return mindspore.schema.NegGrad.decode(reader, position);
            case 103: return mindspore.schema.NotEqual.decode(reader, position);
            case 104: return mindspore.schema.NonMaxSuppression.decode(reader, position);
            case 105: return mindspore.schema.OneHot.decode(reader, position);
            case 106: return mindspore.schema.OnesLike.decode(reader, position);
            case 107: return mindspore.schema.PadFusion.decode(reader, position);
            case 108: return mindspore.schema.PartialFusion.decode(reader, position);
            case 109: return mindspore.schema.PowerGrad.decode(reader, position);
            case 110: return mindspore.schema.PowFusion.decode(reader, position);
            case 111: return mindspore.schema.PriorBox.decode(reader, position);
            case 112: return mindspore.schema.PReLUFusion.decode(reader, position);
            case 113: return mindspore.schema.QuantDTypeCast.decode(reader, position);
            case 114: return mindspore.schema.Rank.decode(reader, position);
            case 115: return mindspore.schema.Range.decode(reader, position);
            case 116: return mindspore.schema.Reciprocal.decode(reader, position);
            case 117: return mindspore.schema.RealDiv.decode(reader, position);
            case 118: return mindspore.schema.ReduceFusion.decode(reader, position);
            case 119: return mindspore.schema.Reshape.decode(reader, position);
            case 120: return mindspore.schema.Resize.decode(reader, position);
            case 121: return mindspore.schema.ReverseSequence.decode(reader, position);
            case 122: return mindspore.schema.ReverseV2.decode(reader, position);
            case 123: return mindspore.schema.Rfft.decode(reader, position);
            case 124: return mindspore.schema.ROIPooling.decode(reader, position);
            case 125: return mindspore.schema.Round.decode(reader, position);
            case 126: return mindspore.schema.Rsqrt.decode(reader, position);
            case 127: return mindspore.schema.ScaleFusion.decode(reader, position);
            case 128: return mindspore.schema.ScatterNd.decode(reader, position);
            case 129: return mindspore.schema.SGD.decode(reader, position);
            case 130: return mindspore.schema.Shape.decode(reader, position);
            case 131: return mindspore.schema.SigmoidCrossEntropyWithLogits.decode(reader, position);
            case 132: return mindspore.schema.SigmoidCrossEntropyWithLogitsGrad.decode(reader, position);
            case 133: return mindspore.schema.Sin.decode(reader, position);
            case 134: return mindspore.schema.SkipGram.decode(reader, position);
            case 135: return mindspore.schema.SliceFusion.decode(reader, position);
            case 136: return mindspore.schema.SmoothL1Loss.decode(reader, position);
            case 137: return mindspore.schema.SmoothL1LossGrad.decode(reader, position);
            case 138: return mindspore.schema.Softmax.decode(reader, position);
            case 139: return mindspore.schema.SoftmaxCrossEntropyWithLogits.decode(reader, position);
            case 140: return mindspore.schema.SpaceToBatch.decode(reader, position);
            case 141: return mindspore.schema.SpaceToBatchND.decode(reader, position);
            case 142: return mindspore.schema.SpaceToDepth.decode(reader, position);
            case 143: return mindspore.schema.SparseSoftmaxCrossEntropyWithLogits.decode(reader, position);
            case 144: return mindspore.schema.SparseToDense.decode(reader, position);
            case 145: return mindspore.schema.Split.decode(reader, position);
            case 146: return mindspore.schema.Sqrt.decode(reader, position);
            case 147: return mindspore.schema.Squeeze.decode(reader, position);
            case 148: return mindspore.schema.Square.decode(reader, position);
            case 149: return mindspore.schema.SquaredDifference.decode(reader, position);
            case 150: return mindspore.schema.Stack.decode(reader, position);
            case 151: return mindspore.schema.StridedSlice.decode(reader, position);
            case 152: return mindspore.schema.SubFusion.decode(reader, position);
            case 153: return mindspore.schema.SubGrad.decode(reader, position);
            case 154: return mindspore.schema.Switch.decode(reader, position);
            case 155: return mindspore.schema.TensorListFromTensor.decode(reader, position);
            case 156: return mindspore.schema.TensorListGetItem.decode(reader, position);
            case 157: return mindspore.schema.TensorListReserve.decode(reader, position);
            case 158: return mindspore.schema.TensorListSetItem.decode(reader, position);
            case 159: return mindspore.schema.TensorListStack.decode(reader, position);
            case 160: return mindspore.schema.TileFusion.decode(reader, position);
            case 161: return mindspore.schema.TopKFusion.decode(reader, position);
            case 162: return mindspore.schema.Transpose.decode(reader, position);
            case 163: return mindspore.schema.Unique.decode(reader, position);
            case 164: return mindspore.schema.UnsortedSegmentSum.decode(reader, position);
            case 165: return mindspore.schema.Unsqueeze.decode(reader, position);
            case 166: return mindspore.schema.Unstack.decode(reader, position);
            case 167: return mindspore.schema.LSTMGrad.decode(reader, position);
            case 168: return mindspore.schema.Where.decode(reader, position);
            case 169: return mindspore.schema.ZerosLike.decode(reader, position);
            case 170: return mindspore.schema.Select.decode(reader, position);
            case 171: return mindspore.schema.ScatterNdUpdate.decode(reader, position);
            case 172: return mindspore.schema.GRU.decode(reader, position);
            case 173: return mindspore.schema.NonZero.decode(reader, position);
            case 174: return mindspore.schema.InvertPermutation.decode(reader, position);
            case 175: return mindspore.schema.Size.decode(reader, position);
            case 176: return mindspore.schema.RandomStandardNormal.decode(reader, position);
            case 177: return mindspore.schema.CropAndResize.decode(reader, position);
            case 178: return mindspore.schema.Erf.decode(reader, position);
            case 179: return mindspore.schema.StridedSliceGrad.decode(reader, position);
            case 180: return mindspore.schema.IsFinite.decode(reader, position);
            case 181: return mindspore.schema.LinSpace.decode(reader, position);
            case 182: return mindspore.schema.UniformReal.decode(reader, position);
            case 183: return mindspore.schema.AbsGrad.decode(reader, position);
            case 184: return mindspore.schema.RsqrtGrad.decode(reader, position);
            case 185: return mindspore.schema.SqrtGrad.decode(reader, position);
            case 186: return mindspore.schema.LayerNormGrad.decode(reader, position);
            case 187: return mindspore.schema.ResizeGrad.decode(reader, position);
            case 188: return mindspore.schema.Splice.decode(reader, position);
            case 189: return mindspore.schema.LogSoftmax.decode(reader, position);
            case 190: return mindspore.schema.Call.decode(reader, position);
            case 191: return mindspore.schema.Custom.decode(reader, position);
            case 192: return mindspore.schema.CumSum.decode(reader, position);
            case 193: return mindspore.schema.SplitWithOverlap.decode(reader, position);
            case 194: return mindspore.schema.GenOP.decode(reader, position);
            case 195: return mindspore.schema.RaggedRange.decode(reader, position);
            case 196: return mindspore.schema.GLU.decode(reader, position);
            case 197: return mindspore.schema.TensorArray.decode(reader, position);
            case 198: return mindspore.schema.TensorArrayRead.decode(reader, position);
            case 199: return mindspore.schema.TensorArrayWrite.decode(reader, position);
            case 200: return mindspore.schema.Affine.decode(reader, position);
            case 201: return mindspore.schema.AllGather.decode(reader, position);
            case 202: return mindspore.schema.ReduceScatter.decode(reader, position);
            case 203: return mindspore.schema.DynamicQuant.decode(reader, position);
            case 204: return mindspore.schema.LSTMGradData.decode(reader, position);
            case 205: return mindspore.schema.LSTMGradWeight.decode(reader, position);
            case 206: return mindspore.schema.RandomNormal.decode(reader, position);
            case 207: return mindspore.schema.NLLLoss.decode(reader, position);
            case 208: return mindspore.schema.NLLLossGrad.decode(reader, position);
            case 209: return mindspore.schema.FormatTranspose.decode(reader, position);
            case 210: return mindspore.schema.GatherD.decode(reader, position);
            case 211: return mindspore.schema.GroupNormFusion.decode(reader, position);
            case 212: return mindspore.schema.Log1p.decode(reader, position);
            case 213: return mindspore.schema.TensorScatterAdd.decode(reader, position);
            case 214: return mindspore.schema.SparseFillEmptyRows.decode(reader, position);
            case 215: return mindspore.schema.SparseReshape.decode(reader, position);
            case 216: return mindspore.schema.SparseSegmentSum.decode(reader, position);
            case 217: return mindspore.schema.ScatterElements.decode(reader, position);
            case 218: return mindspore.schema.Triu.decode(reader, position);
            case 219: return mindspore.schema.Tril.decode(reader, position);
            case 220: return mindspore.schema.AdamWeightDecay.decode(reader, position);
            case 221: return mindspore.schema.FillV2.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Abs': return mindspore.schema.Abs.decodeText(reader, json);
            case 'Activation': return mindspore.schema.Activation.decodeText(reader, json);
            case 'ActivationGrad': return mindspore.schema.ActivationGrad.decodeText(reader, json);
            case 'Adam': return mindspore.schema.Adam.decodeText(reader, json);
            case 'AddFusion': return mindspore.schema.AddFusion.decodeText(reader, json);
            case 'AdderFusion': return mindspore.schema.AdderFusion.decodeText(reader, json);
            case 'AddGrad': return mindspore.schema.AddGrad.decodeText(reader, json);
            case 'AddN': return mindspore.schema.AddN.decodeText(reader, json);
            case 'All': return mindspore.schema.All.decodeText(reader, json);
            case 'ApplyMomentum': return mindspore.schema.ApplyMomentum.decodeText(reader, json);
            case 'ArgMaxFusion': return mindspore.schema.ArgMaxFusion.decodeText(reader, json);
            case 'ArgMinFusion': return mindspore.schema.ArgMinFusion.decodeText(reader, json);
            case 'Assert': return mindspore.schema.Assert.decodeText(reader, json);
            case 'Assign': return mindspore.schema.Assign.decodeText(reader, json);
            case 'AssignAdd': return mindspore.schema.AssignAdd.decodeText(reader, json);
            case 'AudioSpectrogram': return mindspore.schema.AudioSpectrogram.decodeText(reader, json);
            case 'AvgPoolFusion': return mindspore.schema.AvgPoolFusion.decodeText(reader, json);
            case 'AvgPoolGrad': return mindspore.schema.AvgPoolGrad.decodeText(reader, json);
            case 'BatchNorm': return mindspore.schema.BatchNorm.decodeText(reader, json);
            case 'BatchNormGrad': return mindspore.schema.BatchNormGrad.decodeText(reader, json);
            case 'BatchToSpace': return mindspore.schema.BatchToSpace.decodeText(reader, json);
            case 'BatchToSpaceND': return mindspore.schema.BatchToSpaceND.decodeText(reader, json);
            case 'BiasAdd': return mindspore.schema.BiasAdd.decodeText(reader, json);
            case 'BinaryCrossEntropy': return mindspore.schema.BinaryCrossEntropy.decodeText(reader, json);
            case 'BinaryCrossEntropyGrad': return mindspore.schema.BinaryCrossEntropyGrad.decodeText(reader, json);
            case 'BiasAddGrad': return mindspore.schema.BiasAddGrad.decodeText(reader, json);
            case 'BroadcastTo': return mindspore.schema.BroadcastTo.decodeText(reader, json);
            case 'Cast': return mindspore.schema.Cast.decodeText(reader, json);
            case 'Ceil': return mindspore.schema.Ceil.decodeText(reader, json);
            case 'Clip': return mindspore.schema.Clip.decodeText(reader, json);
            case 'Concat': return mindspore.schema.Concat.decodeText(reader, json);
            case 'Attention': return mindspore.schema.Attention.decodeText(reader, json);
            case 'Conv2DBackpropFilterFusion': return mindspore.schema.Conv2DBackpropFilterFusion.decodeText(reader, json);
            case 'Conv2DBackpropInputFusion': return mindspore.schema.Conv2DBackpropInputFusion.decodeText(reader, json);
            case 'Conv2DFusion': return mindspore.schema.Conv2DFusion.decodeText(reader, json);
            case 'Conv2dTransposeFusion': return mindspore.schema.Conv2dTransposeFusion.decodeText(reader, json);
            case 'Cos': return mindspore.schema.Cos.decodeText(reader, json);
            case 'ConstantOfShape': return mindspore.schema.ConstantOfShape.decodeText(reader, json);
            case 'Crop': return mindspore.schema.Crop.decodeText(reader, json);
            case 'CustomExtractFeatures': return mindspore.schema.CustomExtractFeatures.decodeText(reader, json);
            case 'CustomNormalize': return mindspore.schema.CustomNormalize.decodeText(reader, json);
            case 'CustomPredict': return mindspore.schema.CustomPredict.decodeText(reader, json);
            case 'DeConv2DGradFilter': return mindspore.schema.DeConv2DGradFilter.decodeText(reader, json);
            case 'Depend': return mindspore.schema.Depend.decodeText(reader, json);
            case 'DepthToSpace': return mindspore.schema.DepthToSpace.decodeText(reader, json);
            case 'DetectionPostProcess': return mindspore.schema.DetectionPostProcess.decodeText(reader, json);
            case 'DivFusion': return mindspore.schema.DivFusion.decodeText(reader, json);
            case 'DivGrad': return mindspore.schema.DivGrad.decodeText(reader, json);
            case 'Dropout': return mindspore.schema.Dropout.decodeText(reader, json);
            case 'DropoutGrad': return mindspore.schema.DropoutGrad.decodeText(reader, json);
            case 'Elu': return mindspore.schema.Elu.decodeText(reader, json);
            case 'Eltwise': return mindspore.schema.Eltwise.decodeText(reader, json);
            case 'Equal': return mindspore.schema.Equal.decodeText(reader, json);
            case 'EmbeddingLookupFusion': return mindspore.schema.EmbeddingLookupFusion.decodeText(reader, json);
            case 'ExpFusion': return mindspore.schema.ExpFusion.decodeText(reader, json);
            case 'ExpandDims': return mindspore.schema.ExpandDims.decodeText(reader, json);
            case 'FakeQuantWithMinMaxVars': return mindspore.schema.FakeQuantWithMinMaxVars.decodeText(reader, json);
            case 'FakeQuantWithMinMaxVarsPerChannel': return mindspore.schema.FakeQuantWithMinMaxVarsPerChannel.decodeText(reader, json);
            case 'FftReal': return mindspore.schema.FftReal.decodeText(reader, json);
            case 'FftImag': return mindspore.schema.FftImag.decodeText(reader, json);
            case 'Flatten': return mindspore.schema.Flatten.decodeText(reader, json);
            case 'FlattenGrad': return mindspore.schema.FlattenGrad.decodeText(reader, json);
            case 'Floor': return mindspore.schema.Floor.decodeText(reader, json);
            case 'FloorDiv': return mindspore.schema.FloorDiv.decodeText(reader, json);
            case 'FloorMod': return mindspore.schema.FloorMod.decodeText(reader, json);
            case 'Fill': return mindspore.schema.Fill.decodeText(reader, json);
            case 'FullConnection': return mindspore.schema.FullConnection.decodeText(reader, json);
            case 'FusedBatchNorm': return mindspore.schema.FusedBatchNorm.decodeText(reader, json);
            case 'Gather': return mindspore.schema.Gather.decodeText(reader, json);
            case 'GatherNd': return mindspore.schema.GatherNd.decodeText(reader, json);
            case 'Greater': return mindspore.schema.Greater.decodeText(reader, json);
            case 'GreaterEqual': return mindspore.schema.GreaterEqual.decodeText(reader, json);
            case 'HashtableLookup': return mindspore.schema.HashtableLookup.decodeText(reader, json);
            case 'InstanceNorm': return mindspore.schema.InstanceNorm.decodeText(reader, json);
            case 'LayerNormFusion': return mindspore.schema.LayerNormFusion.decodeText(reader, json);
            case 'LeakyRelu': return mindspore.schema.LeakyRelu.decodeText(reader, json);
            case 'Less': return mindspore.schema.Less.decodeText(reader, json);
            case 'LessEqual': return mindspore.schema.LessEqual.decodeText(reader, json);
            case 'Log': return mindspore.schema.Log.decodeText(reader, json);
            case 'LogGrad': return mindspore.schema.LogGrad.decodeText(reader, json);
            case 'LogicalAnd': return mindspore.schema.LogicalAnd.decodeText(reader, json);
            case 'LogicalNot': return mindspore.schema.LogicalNot.decodeText(reader, json);
            case 'LogicalOr': return mindspore.schema.LogicalOr.decodeText(reader, json);
            case 'LpNormalization': return mindspore.schema.LpNormalization.decodeText(reader, json);
            case 'LRN': return mindspore.schema.LRN.decodeText(reader, json);
            case 'LshProjection': return mindspore.schema.LshProjection.decodeText(reader, json);
            case 'LSTM': return mindspore.schema.LSTM.decodeText(reader, json);
            case 'L2NormalizeFusion': return mindspore.schema.L2NormalizeFusion.decodeText(reader, json);
            case 'MatMulFusion': return mindspore.schema.MatMulFusion.decodeText(reader, json);
            case 'Maximum': return mindspore.schema.Maximum.decodeText(reader, json);
            case 'MaximumGrad': return mindspore.schema.MaximumGrad.decodeText(reader, json);
            case 'MaxPoolFusion': return mindspore.schema.MaxPoolFusion.decodeText(reader, json);
            case 'MaxPoolGrad': return mindspore.schema.MaxPoolGrad.decodeText(reader, json);
            case 'SwitchLayer': return mindspore.schema.SwitchLayer.decodeText(reader, json);
            case 'Mfcc': return mindspore.schema.Mfcc.decodeText(reader, json);
            case 'Minimum': return mindspore.schema.Minimum.decodeText(reader, json);
            case 'MinimumGrad': return mindspore.schema.MinimumGrad.decodeText(reader, json);
            case 'Mod': return mindspore.schema.Mod.decodeText(reader, json);
            case 'MulFusion': return mindspore.schema.MulFusion.decodeText(reader, json);
            case 'MulGrad': return mindspore.schema.MulGrad.decodeText(reader, json);
            case 'Neg': return mindspore.schema.Neg.decodeText(reader, json);
            case 'NegGrad': return mindspore.schema.NegGrad.decodeText(reader, json);
            case 'NotEqual': return mindspore.schema.NotEqual.decodeText(reader, json);
            case 'NonMaxSuppression': return mindspore.schema.NonMaxSuppression.decodeText(reader, json);
            case 'OneHot': return mindspore.schema.OneHot.decodeText(reader, json);
            case 'OnesLike': return mindspore.schema.OnesLike.decodeText(reader, json);
            case 'PadFusion': return mindspore.schema.PadFusion.decodeText(reader, json);
            case 'PartialFusion': return mindspore.schema.PartialFusion.decodeText(reader, json);
            case 'PowerGrad': return mindspore.schema.PowerGrad.decodeText(reader, json);
            case 'PowFusion': return mindspore.schema.PowFusion.decodeText(reader, json);
            case 'PriorBox': return mindspore.schema.PriorBox.decodeText(reader, json);
            case 'PReLUFusion': return mindspore.schema.PReLUFusion.decodeText(reader, json);
            case 'QuantDTypeCast': return mindspore.schema.QuantDTypeCast.decodeText(reader, json);
            case 'Rank': return mindspore.schema.Rank.decodeText(reader, json);
            case 'Range': return mindspore.schema.Range.decodeText(reader, json);
            case 'Reciprocal': return mindspore.schema.Reciprocal.decodeText(reader, json);
            case 'RealDiv': return mindspore.schema.RealDiv.decodeText(reader, json);
            case 'ReduceFusion': return mindspore.schema.ReduceFusion.decodeText(reader, json);
            case 'Reshape': return mindspore.schema.Reshape.decodeText(reader, json);
            case 'Resize': return mindspore.schema.Resize.decodeText(reader, json);
            case 'ReverseSequence': return mindspore.schema.ReverseSequence.decodeText(reader, json);
            case 'ReverseV2': return mindspore.schema.ReverseV2.decodeText(reader, json);
            case 'Rfft': return mindspore.schema.Rfft.decodeText(reader, json);
            case 'ROIPooling': return mindspore.schema.ROIPooling.decodeText(reader, json);
            case 'Round': return mindspore.schema.Round.decodeText(reader, json);
            case 'Rsqrt': return mindspore.schema.Rsqrt.decodeText(reader, json);
            case 'ScaleFusion': return mindspore.schema.ScaleFusion.decodeText(reader, json);
            case 'ScatterNd': return mindspore.schema.ScatterNd.decodeText(reader, json);
            case 'SGD': return mindspore.schema.SGD.decodeText(reader, json);
            case 'Shape': return mindspore.schema.Shape.decodeText(reader, json);
            case 'SigmoidCrossEntropyWithLogits': return mindspore.schema.SigmoidCrossEntropyWithLogits.decodeText(reader, json);
            case 'SigmoidCrossEntropyWithLogitsGrad': return mindspore.schema.SigmoidCrossEntropyWithLogitsGrad.decodeText(reader, json);
            case 'Sin': return mindspore.schema.Sin.decodeText(reader, json);
            case 'SkipGram': return mindspore.schema.SkipGram.decodeText(reader, json);
            case 'SliceFusion': return mindspore.schema.SliceFusion.decodeText(reader, json);
            case 'SmoothL1Loss': return mindspore.schema.SmoothL1Loss.decodeText(reader, json);
            case 'SmoothL1LossGrad': return mindspore.schema.SmoothL1LossGrad.decodeText(reader, json);
            case 'Softmax': return mindspore.schema.Softmax.decodeText(reader, json);
            case 'SoftmaxCrossEntropyWithLogits': return mindspore.schema.SoftmaxCrossEntropyWithLogits.decodeText(reader, json);
            case 'SpaceToBatch': return mindspore.schema.SpaceToBatch.decodeText(reader, json);
            case 'SpaceToBatchND': return mindspore.schema.SpaceToBatchND.decodeText(reader, json);
            case 'SpaceToDepth': return mindspore.schema.SpaceToDepth.decodeText(reader, json);
            case 'SparseSoftmaxCrossEntropyWithLogits': return mindspore.schema.SparseSoftmaxCrossEntropyWithLogits.decodeText(reader, json);
            case 'SparseToDense': return mindspore.schema.SparseToDense.decodeText(reader, json);
            case 'Split': return mindspore.schema.Split.decodeText(reader, json);
            case 'Sqrt': return mindspore.schema.Sqrt.decodeText(reader, json);
            case 'Squeeze': return mindspore.schema.Squeeze.decodeText(reader, json);
            case 'Square': return mindspore.schema.Square.decodeText(reader, json);
            case 'SquaredDifference': return mindspore.schema.SquaredDifference.decodeText(reader, json);
            case 'Stack': return mindspore.schema.Stack.decodeText(reader, json);
            case 'StridedSlice': return mindspore.schema.StridedSlice.decodeText(reader, json);
            case 'SubFusion': return mindspore.schema.SubFusion.decodeText(reader, json);
            case 'SubGrad': return mindspore.schema.SubGrad.decodeText(reader, json);
            case 'Switch': return mindspore.schema.Switch.decodeText(reader, json);
            case 'TensorListFromTensor': return mindspore.schema.TensorListFromTensor.decodeText(reader, json);
            case 'TensorListGetItem': return mindspore.schema.TensorListGetItem.decodeText(reader, json);
            case 'TensorListReserve': return mindspore.schema.TensorListReserve.decodeText(reader, json);
            case 'TensorListSetItem': return mindspore.schema.TensorListSetItem.decodeText(reader, json);
            case 'TensorListStack': return mindspore.schema.TensorListStack.decodeText(reader, json);
            case 'TileFusion': return mindspore.schema.TileFusion.decodeText(reader, json);
            case 'TopKFusion': return mindspore.schema.TopKFusion.decodeText(reader, json);
            case 'Transpose': return mindspore.schema.Transpose.decodeText(reader, json);
            case 'Unique': return mindspore.schema.Unique.decodeText(reader, json);
            case 'UnsortedSegmentSum': return mindspore.schema.UnsortedSegmentSum.decodeText(reader, json);
            case 'Unsqueeze': return mindspore.schema.Unsqueeze.decodeText(reader, json);
            case 'Unstack': return mindspore.schema.Unstack.decodeText(reader, json);
            case 'LSTMGrad': return mindspore.schema.LSTMGrad.decodeText(reader, json);
            case 'Where': return mindspore.schema.Where.decodeText(reader, json);
            case 'ZerosLike': return mindspore.schema.ZerosLike.decodeText(reader, json);
            case 'Select': return mindspore.schema.Select.decodeText(reader, json);
            case 'ScatterNdUpdate': return mindspore.schema.ScatterNdUpdate.decodeText(reader, json);
            case 'GRU': return mindspore.schema.GRU.decodeText(reader, json);
            case 'NonZero': return mindspore.schema.NonZero.decodeText(reader, json);
            case 'InvertPermutation': return mindspore.schema.InvertPermutation.decodeText(reader, json);
            case 'Size': return mindspore.schema.Size.decodeText(reader, json);
            case 'RandomStandardNormal': return mindspore.schema.RandomStandardNormal.decodeText(reader, json);
            case 'CropAndResize': return mindspore.schema.CropAndResize.decodeText(reader, json);
            case 'Erf': return mindspore.schema.Erf.decodeText(reader, json);
            case 'StridedSliceGrad': return mindspore.schema.StridedSliceGrad.decodeText(reader, json);
            case 'IsFinite': return mindspore.schema.IsFinite.decodeText(reader, json);
            case 'LinSpace': return mindspore.schema.LinSpace.decodeText(reader, json);
            case 'UniformReal': return mindspore.schema.UniformReal.decodeText(reader, json);
            case 'AbsGrad': return mindspore.schema.AbsGrad.decodeText(reader, json);
            case 'RsqrtGrad': return mindspore.schema.RsqrtGrad.decodeText(reader, json);
            case 'SqrtGrad': return mindspore.schema.SqrtGrad.decodeText(reader, json);
            case 'LayerNormGrad': return mindspore.schema.LayerNormGrad.decodeText(reader, json);
            case 'ResizeGrad': return mindspore.schema.ResizeGrad.decodeText(reader, json);
            case 'Splice': return mindspore.schema.Splice.decodeText(reader, json);
            case 'LogSoftmax': return mindspore.schema.LogSoftmax.decodeText(reader, json);
            case 'Call': return mindspore.schema.Call.decodeText(reader, json);
            case 'Custom': return mindspore.schema.Custom.decodeText(reader, json);
            case 'CumSum': return mindspore.schema.CumSum.decodeText(reader, json);
            case 'SplitWithOverlap': return mindspore.schema.SplitWithOverlap.decodeText(reader, json);
            case 'GenOP': return mindspore.schema.GenOP.decodeText(reader, json);
            case 'RaggedRange': return mindspore.schema.RaggedRange.decodeText(reader, json);
            case 'GLU': return mindspore.schema.GLU.decodeText(reader, json);
            case 'TensorArray': return mindspore.schema.TensorArray.decodeText(reader, json);
            case 'TensorArrayRead': return mindspore.schema.TensorArrayRead.decodeText(reader, json);
            case 'TensorArrayWrite': return mindspore.schema.TensorArrayWrite.decodeText(reader, json);
            case 'Affine': return mindspore.schema.Affine.decodeText(reader, json);
            case 'AllGather': return mindspore.schema.AllGather.decodeText(reader, json);
            case 'ReduceScatter': return mindspore.schema.ReduceScatter.decodeText(reader, json);
            case 'DynamicQuant': return mindspore.schema.DynamicQuant.decodeText(reader, json);
            case 'LSTMGradData': return mindspore.schema.LSTMGradData.decodeText(reader, json);
            case 'LSTMGradWeight': return mindspore.schema.LSTMGradWeight.decodeText(reader, json);
            case 'RandomNormal': return mindspore.schema.RandomNormal.decodeText(reader, json);
            case 'NLLLoss': return mindspore.schema.NLLLoss.decodeText(reader, json);
            case 'NLLLossGrad': return mindspore.schema.NLLLossGrad.decodeText(reader, json);
            case 'FormatTranspose': return mindspore.schema.FormatTranspose.decodeText(reader, json);
            case 'GatherD': return mindspore.schema.GatherD.decodeText(reader, json);
            case 'GroupNormFusion': return mindspore.schema.GroupNormFusion.decodeText(reader, json);
            case 'Log1p': return mindspore.schema.Log1p.decodeText(reader, json);
            case 'TensorScatterAdd': return mindspore.schema.TensorScatterAdd.decodeText(reader, json);
            case 'SparseFillEmptyRows': return mindspore.schema.SparseFillEmptyRows.decodeText(reader, json);
            case 'SparseReshape': return mindspore.schema.SparseReshape.decodeText(reader, json);
            case 'SparseSegmentSum': return mindspore.schema.SparseSegmentSum.decodeText(reader, json);
            case 'ScatterElements': return mindspore.schema.ScatterElements.decodeText(reader, json);
            case 'Triu': return mindspore.schema.Triu.decodeText(reader, json);
            case 'Tril': return mindspore.schema.Tril.decodeText(reader, json);
            case 'AdamWeightDecay': return mindspore.schema.AdamWeightDecay.decodeText(reader, json);
            case 'FillV2': return mindspore.schema.FillV2.decodeText(reader, json);
            default: return undefined;
        }
    }
};

mindspore.schema.Abs = class Abs {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Abs();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Abs();
        return $;
    }
};

mindspore.schema.Activation = class Activation {

    static decode(reader, position) {
        const $ = new mindspore.schema.Activation();
        $.activation_type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        $.min_val = reader.float32_(position, 8, 0);
        $.max_val = reader.float32_(position, 10, 0);
        $.approximate = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Activation();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        $.alpha = reader.value(json.alpha, 0);
        $.min_val = reader.value(json.min_val, 0);
        $.max_val = reader.value(json.max_val, 0);
        $.approximate = reader.value(json.approximate, false);
        return $;
    }
};

mindspore.schema.ActivationGrad = class ActivationGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.ActivationGrad();
        $.activation_type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ActivationGrad();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        $.alpha = reader.value(json.alpha, 0);
        return $;
    }
};

mindspore.schema.Adam = class Adam {

    static decode(reader, position) {
        const $ = new mindspore.schema.Adam();
        $.use_locking = reader.bool_(position, 4, false);
        $.use_nesterov = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Adam();
        $.use_locking = reader.value(json.use_locking, false);
        $.use_nesterov = reader.value(json.use_nesterov, false);
        return $;
    }
};

mindspore.schema.AddFusion = class AddFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.AddFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.AddFusion();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.AdderFusion = class AdderFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.AdderFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad_list = reader.int64s_(position, 14);
        $.group = reader.int64_(position, 16, 0n);
        $.in_channel = reader.int64_(position, 18, 0n);
        $.out_channel = reader.int64_(position, 20, 0n);
        $.activation_type = reader.int8_(position, 22, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.AdderFusion();
        $.format = mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.group = reader.int64(json.group, 0n);
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.AddGrad = class AddGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.AddGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.AddGrad();
        return $;
    }
};

mindspore.schema.AddN = class AddN {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.AddN();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.AddN();
        return $;
    }
};

mindspore.schema.All = class All {

    static decode(reader, position) {
        const $ = new mindspore.schema.All();
        $.keep_dims = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.All();
        $.keep_dims = reader.int64(json.keep_dims, 0n);
        return $;
    }
};

mindspore.schema.ApplyMomentum = class ApplyMomentum {

    static decode(reader, position) {
        const $ = new mindspore.schema.ApplyMomentum();
        $.use_nesterov = reader.bool_(position, 4, false);
        $.use_locking = reader.bool_(position, 6, false);
        $.gradient_scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ApplyMomentum();
        $.use_nesterov = reader.value(json.use_nesterov, false);
        $.use_locking = reader.value(json.use_locking, false);
        $.gradient_scale = reader.value(json.gradient_scale, 0);
        return $;
    }
};

mindspore.schema.ArgMaxFusion = class ArgMaxFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.ArgMaxFusion();
        $.axis = reader.int64_(position, 4, 0n);
        $.top_k = reader.int64_(position, 6, 1n);
        $.keep_dims = reader.bool_(position, 8, false);
        $.out_max_value = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ArgMaxFusion();
        $.axis = reader.int64(json.axis, 0n);
        $.top_k = reader.int64(json.top_k, 1n);
        $.keep_dims = reader.value(json.keep_dims, false);
        $.out_max_value = reader.value(json.out_max_value, false);
        return $;
    }
};

mindspore.schema.ArgMinFusion = class ArgMinFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.ArgMinFusion();
        $.axis = reader.int64_(position, 4, 0n);
        $.top_k = reader.int64_(position, 6, 0n);
        $.keep_dims = reader.bool_(position, 8, false);
        $.out_max_value = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ArgMinFusion();
        $.axis = reader.int64(json.axis, 0n);
        $.top_k = reader.int64(json.top_k, 0n);
        $.keep_dims = reader.value(json.keep_dims, false);
        $.out_max_value = reader.value(json.out_max_value, false);
        return $;
    }
};

mindspore.schema.Assert = class Assert {

    static decode(reader, position) {
        const $ = new mindspore.schema.Assert();
        $.summarize = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Assert();
        $.summarize = reader.int64(json.summarize, 0n);
        return $;
    }
};

mindspore.schema.Assign = class Assign {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Assign();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Assign();
        return $;
    }
};

mindspore.schema.AssignAdd = class AssignAdd {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.AssignAdd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.AssignAdd();
        return $;
    }
};

mindspore.schema.AudioSpectrogram = class AudioSpectrogram {

    static decode(reader, position) {
        const $ = new mindspore.schema.AudioSpectrogram();
        $.window_size = reader.int64_(position, 4, 0n);
        $.stride = reader.int64_(position, 6, 0n);
        $.mag_square = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.AudioSpectrogram();
        $.window_size = reader.int64(json.window_size, 0n);
        $.stride = reader.int64(json.stride, 0n);
        $.mag_square = reader.value(json.mag_square, false);
        return $;
    }
};

mindspore.schema.AvgPoolFusion = class AvgPoolFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.AvgPoolFusion();
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
        const $ = new mindspore.schema.AvgPoolFusion();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad = reader.array(json.pad);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.round_mode = mindspore.schema.RoundMode[json.round_mode];
        $.format = mindspore.schema.Format[json.format];
        $.global = reader.value(json.global, false);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.AvgPoolGrad = class AvgPoolGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.AvgPoolGrad();
        $.kernel_size = reader.int64s_(position, 4);
        $.strides = reader.int64s_(position, 6);
        $.pad_mode = reader.int8_(position, 8, 0);
        $.format = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.AvgPoolGrad();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.format = mindspore.schema.Format[json.format];
        return $;
    }
};

mindspore.schema.BatchNorm = class BatchNorm {

    static decode(reader, position) {
        const $ = new mindspore.schema.BatchNorm();
        $.epsilon = reader.float32_(position, 4, 0);
        $.format = reader.int32_(position, 6, 0);
        $.is_training = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BatchNorm();
        $.epsilon = reader.value(json.epsilon, 0);
        $.format = mindspore.schema.Format[json.format];
        $.is_training = reader.value(json.is_training, false);
        return $;
    }
};

mindspore.schema.BatchNormGrad = class BatchNormGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.BatchNormGrad();
        $.epsilon = reader.float32_(position, 4, 0);
        $.is_training = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BatchNormGrad();
        $.epsilon = reader.value(json.epsilon, 0);
        $.is_training = reader.value(json.is_training, false);
        return $;
    }
};

mindspore.schema.BatchToSpace = class BatchToSpace {

    static decode(reader, position) {
        const $ = new mindspore.schema.BatchToSpace();
        $.block_size = reader.int64s_(position, 4);
        $.crops = reader.table(position, 6, mindspore.schema.Vec2D);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BatchToSpace();
        $.block_size = reader.array(json.block_size);
        $.crops = reader.object(json.crops, mindspore.schema.Vec2D);
        return $;
    }
};

mindspore.schema.BatchToSpaceND = class BatchToSpaceND {

    static decode(reader, position) {
        const $ = new mindspore.schema.BatchToSpaceND();
        $.block_shape = reader.int64s_(position, 4);
        $.crops = reader.table(position, 6, mindspore.schema.Vec2D);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BatchToSpaceND();
        $.block_shape = reader.array(json.block_shape);
        $.crops = reader.object(json.crops, mindspore.schema.Vec2D);
        return $;
    }
};

mindspore.schema.BiasAdd = class BiasAdd {

    static decode(reader, position) {
        const $ = new mindspore.schema.BiasAdd();
        $.format = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BiasAdd();
        $.format = mindspore.schema.Format[json.format];
        return $;
    }
};

mindspore.schema.BinaryCrossEntropy = class BinaryCrossEntropy {

    static decode(reader, position) {
        const $ = new mindspore.schema.BinaryCrossEntropy();
        $.reduction = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BinaryCrossEntropy();
        $.reduction = mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

mindspore.schema.BinaryCrossEntropyGrad = class BinaryCrossEntropyGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.BinaryCrossEntropyGrad();
        $.reduction = reader.int8_(position, 4, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BinaryCrossEntropyGrad();
        $.reduction = mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

mindspore.schema.BiasAddGrad = class BiasAddGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.BiasAddGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.BiasAddGrad();
        return $;
    }
};

mindspore.schema.BroadcastTo = class BroadcastTo {

    static decode(reader, position) {
        const $ = new mindspore.schema.BroadcastTo();
        $.shape = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.BroadcastTo();
        $.shape = reader.array(json.shape);
        return $;
    }
};

mindspore.schema.Cast = class Cast {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Cast();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Cast();
        return $;
    }
};

mindspore.schema.Ceil = class Ceil {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Ceil();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Ceil();
        return $;
    }
};

mindspore.schema.Clip = class Clip {

    static decode(reader, position) {
        const $ = new mindspore.schema.Clip();
        $.max = reader.float32_(position, 4, 0);
        $.min = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Clip();
        $.max = reader.value(json.max, 0);
        $.min = reader.value(json.min, 0);
        return $;
    }
};

mindspore.schema.Concat = class Concat {

    static decode(reader, position) {
        const $ = new mindspore.schema.Concat();
        $.axis = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Concat();
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.Attention = class Attention {

    static decode(reader, position) {
        const $ = new mindspore.schema.Attention();
        $.head_num = reader.int64_(position, 4, 0n);
        $.head_size = reader.int64_(position, 6, 0n);
        $.cross = reader.bool_(position, 8, false);
        $.scale = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Attention();
        $.head_num = reader.int64(json.head_num, 0n);
        $.head_size = reader.int64(json.head_size, 0n);
        $.cross = reader.value(json.cross, false);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

mindspore.schema.Conv2DBackpropFilterFusion = class Conv2DBackpropFilterFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.Conv2DBackpropFilterFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad_list = reader.int64s_(position, 14);
        $.mode = reader.int64_(position, 16, 0n);
        $.group = reader.int64_(position, 18, 0n);
        $.in_channel = reader.int64_(position, 20, 0n);
        $.out_channel = reader.int64_(position, 22, 0n);
        $.activation_type = reader.int8_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Conv2DBackpropFilterFusion();
        $.format = mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.int64(json.mode, 0n);
        $.group = reader.int64(json.group, 0n);
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.Conv2DBackpropInputFusion = class Conv2DBackpropInputFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.Conv2DBackpropInputFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad = reader.int64s_(position, 14);
        $.pad_list = reader.int64s_(position, 16);
        $.mode = reader.int64_(position, 18, 0n);
        $.group = reader.int64_(position, 20, 0n);
        $.in_channel = reader.int64_(position, 22, 0n);
        $.out_channel = reader.int64_(position, 24, 0n);
        $.activation_type = reader.int8_(position, 26, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Conv2DBackpropInputFusion();
        $.format = mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad = reader.array(json.pad);
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.int64(json.mode, 0n);
        $.group = reader.int64(json.group, 0n);
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.Conv2DFusion = class Conv2DFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.Conv2DFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad_list = reader.int64s_(position, 14);
        $.mode = reader.int64_(position, 16, 0n);
        $.group = reader.int64_(position, 18, 0n);
        $.in_channel = reader.int64_(position, 20, 0n);
        $.out_channel = reader.int64_(position, 22, 0n);
        $.activation_type = reader.int8_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Conv2DFusion();
        $.format = mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.int64(json.mode, 0n);
        $.group = reader.int64(json.group, 0n);
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.Conv2dTransposeFusion = class Conv2dTransposeFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.Conv2dTransposeFusion();
        $.format = reader.int32_(position, 4, 0);
        $.kernel_size = reader.int64s_(position, 6);
        $.stride = reader.int64s_(position, 8);
        $.dilation = reader.int64s_(position, 10);
        $.pad_mode = reader.int8_(position, 12, 0);
        $.pad = reader.int64s_(position, 14);
        $.pad_list = reader.int64s_(position, 16);
        $.mode = reader.int64_(position, 18, 0n);
        $.group = reader.int64_(position, 20, 0n);
        $.in_channel = reader.int64_(position, 22, 0n);
        $.out_channel = reader.int64_(position, 24, 0n);
        $.activation_type = reader.int8_(position, 26, 0);
        $.output_paddings = reader.int64s_(position, 28);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Conv2dTransposeFusion();
        $.format = mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad = reader.array(json.pad);
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.int64(json.mode, 0n);
        $.group = reader.int64(json.group, 0n);
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        $.output_paddings = reader.array(json.output_paddings);
        return $;
    }
};

mindspore.schema.Cos = class Cos {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Cos();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Cos();
        return $;
    }
};

mindspore.schema.ConstantOfShape = class ConstantOfShape {

    static decode(reader, position) {
        const $ = new mindspore.schema.ConstantOfShape();
        $.data_type = reader.int64_(position, 4, 0n);
        $.value = reader.array(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ConstantOfShape();
        $.data_type = reader.int64(json.data_type, 0n);
        $.value = reader.array(json.value, Float32Array);
        return $;
    }
};

mindspore.schema.Crop = class Crop {

    static decode(reader, position) {
        const $ = new mindspore.schema.Crop();
        $.axis = reader.int64_(position, 4, 0n);
        $.offsets = reader.int64s_(position, 6);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Crop();
        $.axis = reader.int64(json.axis, 0n);
        $.offsets = reader.array(json.offsets);
        return $;
    }
};

mindspore.schema.CustomExtractFeatures = class CustomExtractFeatures {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.CustomExtractFeatures();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.CustomExtractFeatures();
        return $;
    }
};

mindspore.schema.CustomNormalize = class CustomNormalize {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.CustomNormalize();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.CustomNormalize();
        return $;
    }
};

mindspore.schema.CustomPredict = class CustomPredict {

    static decode(reader, position) {
        const $ = new mindspore.schema.CustomPredict();
        $.output_num = reader.int64_(position, 4, 0n);
        $.weight_threshold = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.CustomPredict();
        $.output_num = reader.int64(json.output_num, 0n);
        $.weight_threshold = reader.value(json.weight_threshold, 0);
        return $;
    }
};

mindspore.schema.DeConv2DGradFilter = class DeConv2DGradFilter {

    static decode(reader, position) {
        const $ = new mindspore.schema.DeConv2DGradFilter();
        $.in_channel = reader.int64_(position, 4, 0n);
        $.out_channel = reader.int64_(position, 6, 0n);
        $.kernel_size = reader.int64s_(position, 8);
        $.pad_mode = reader.int8_(position, 10, 0);
        $.pad_list = reader.int64s_(position, 12);
        $.stride = reader.int64s_(position, 14);
        $.dilation = reader.int64s_(position, 16);
        $.group = reader.int64_(position, 18, 0n);
        $.format = reader.int32_(position, 20, 0);
        $.activation_type = reader.int8_(position, 22, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.DeConv2DGradFilter();
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.kernel_size = reader.array(json.kernel_size);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.group = reader.int64(json.group, 0n);
        $.format = mindspore.schema.Format[json.format];
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.Depend = class Depend {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Depend();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Depend();
        return $;
    }
};

mindspore.schema.DepthToSpace = class DepthToSpace {

    static decode(reader, position) {
        const $ = new mindspore.schema.DepthToSpace();
        $.block_size = reader.int64_(position, 4, 0n);
        $.format = reader.int32_(position, 6, 0);
        $.mode = reader.string_(position, 8, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.DepthToSpace();
        $.block_size = reader.int64(json.block_size, 0n);
        $.format = mindspore.schema.Format[json.format];
        $.mode = reader.value(json.mode, null);
        return $;
    }
};

mindspore.schema.DetectionPostProcess = class DetectionPostProcess {

    static decode(reader, position) {
        const $ = new mindspore.schema.DetectionPostProcess();
        $.format = reader.int32_(position, 4, 0);
        $.input_size = reader.int64_(position, 6, 0n);
        $.scale = reader.array(position, 8, Float32Array);
        $.nms_iou_threshold = reader.float32_(position, 10, 0);
        $.nms_score_threshold = reader.float32_(position, 12, 0);
        $.max_detections = reader.int64_(position, 14, 0n);
        $.detections_per_class = reader.int64_(position, 16, 0n);
        $.max_classes_per_detection = reader.int64_(position, 18, 0n);
        $.num_classes = reader.int64_(position, 20, 0n);
        $.use_regular_nms = reader.bool_(position, 22, false);
        $.out_quantized = reader.bool_(position, 24, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.DetectionPostProcess();
        $.format = mindspore.schema.Format[json.format];
        $.input_size = reader.int64(json.input_size, 0n);
        $.scale = reader.array(json.scale, Float32Array);
        $.nms_iou_threshold = reader.value(json.nms_iou_threshold, 0);
        $.nms_score_threshold = reader.value(json.nms_score_threshold, 0);
        $.max_detections = reader.int64(json.max_detections, 0n);
        $.detections_per_class = reader.int64(json.detections_per_class, 0n);
        $.max_classes_per_detection = reader.int64(json.max_classes_per_detection, 0n);
        $.num_classes = reader.int64(json.num_classes, 0n);
        $.use_regular_nms = reader.value(json.use_regular_nms, false);
        $.out_quantized = reader.value(json.out_quantized, false);
        return $;
    }
};

mindspore.schema.DivFusion = class DivFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.DivFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.DivFusion();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.DivGrad = class DivGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.DivGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.DivGrad();
        return $;
    }
};

mindspore.schema.Dropout = class Dropout {

    static decode(reader, position) {
        const $ = new mindspore.schema.Dropout();
        $.keep_prob = reader.float32_(position, 4, 0.5);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Dropout();
        $.keep_prob = reader.value(json.keep_prob, 0.5);
        return $;
    }
};

mindspore.schema.DropoutGrad = class DropoutGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.DropoutGrad();
        $.keep_prob = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.DropoutGrad();
        $.keep_prob = reader.value(json.keep_prob, 0);
        return $;
    }
};

mindspore.schema.Elu = class Elu {

    static decode(reader, position) {
        const $ = new mindspore.schema.Elu();
        $.alpha = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Elu();
        $.alpha = reader.value(json.alpha, 0);
        return $;
    }
};

mindspore.schema.Eltwise = class Eltwise {

    static decode(reader, position) {
        const $ = new mindspore.schema.Eltwise();
        $.mode = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Eltwise();
        $.mode = mindspore.schema.EltwiseMode[json.mode];
        return $;
    }
};

mindspore.schema.Equal = class Equal {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Equal();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Equal();
        return $;
    }
};

mindspore.schema.EmbeddingLookupFusion = class EmbeddingLookupFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.EmbeddingLookupFusion();
        $.max_norm = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.EmbeddingLookupFusion();
        $.max_norm = reader.value(json.max_norm, 0);
        return $;
    }
};

mindspore.schema.ExpFusion = class ExpFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.ExpFusion();
        $.base = reader.float32_(position, 4, -1);
        $.scale = reader.float32_(position, 6, 1);
        $.shift = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ExpFusion();
        $.base = reader.value(json.base, -1);
        $.scale = reader.value(json.scale, 1);
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

mindspore.schema.ExpandDims = class ExpandDims {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.ExpandDims();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.ExpandDims();
        return $;
    }
};

mindspore.schema.FakeQuantWithMinMaxVars = class FakeQuantWithMinMaxVars {

    static decode(reader, position) {
        const $ = new mindspore.schema.FakeQuantWithMinMaxVars();
        $.num_bits = reader.int64_(position, 4, 0n);
        $.narrow_range = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.FakeQuantWithMinMaxVars();
        $.num_bits = reader.int64(json.num_bits, 0n);
        $.narrow_range = reader.value(json.narrow_range, false);
        return $;
    }
};

mindspore.schema.FakeQuantWithMinMaxVarsPerChannel = class FakeQuantWithMinMaxVarsPerChannel {

    static decode(reader, position) {
        const $ = new mindspore.schema.FakeQuantWithMinMaxVarsPerChannel();
        $.num_bits = reader.int64_(position, 4, 0n);
        $.narrow_range = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.FakeQuantWithMinMaxVarsPerChannel();
        $.num_bits = reader.int64(json.num_bits, 0n);
        $.narrow_range = reader.value(json.narrow_range, false);
        return $;
    }
};

mindspore.schema.FftReal = class FftReal {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.FftReal();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.FftReal();
        return $;
    }
};

mindspore.schema.FftImag = class FftImag {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.FftImag();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.FftImag();
        return $;
    }
};

mindspore.schema.Flatten = class Flatten {

    static decode(reader, position) {
        const $ = new mindspore.schema.Flatten();
        $.axis = reader.int64_(position, 4, 1n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Flatten();
        $.axis = reader.int64(json.axis, 1n);
        return $;
    }
};

mindspore.schema.FlattenGrad = class FlattenGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.FlattenGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.FlattenGrad();
        return $;
    }
};

mindspore.schema.Floor = class Floor {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Floor();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Floor();
        return $;
    }
};

mindspore.schema.FloorDiv = class FloorDiv {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.FloorDiv();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.FloorDiv();
        return $;
    }
};

mindspore.schema.FloorMod = class FloorMod {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.FloorMod();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.FloorMod();
        return $;
    }
};

mindspore.schema.Fill = class Fill {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Fill();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Fill();
        return $;
    }
};

mindspore.schema.FullConnection = class FullConnection {

    static decode(reader, position) {
        const $ = new mindspore.schema.FullConnection();
        $.has_bias = reader.bool_(position, 4, false);
        $.use_axis = reader.bool_(position, 6, false);
        $.axis = reader.int64_(position, 8, 0n);
        $.activation_type = reader.int8_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.FullConnection();
        $.has_bias = reader.value(json.has_bias, false);
        $.use_axis = reader.value(json.use_axis, false);
        $.axis = reader.int64(json.axis, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.FusedBatchNorm = class FusedBatchNorm {

    static decode(reader, position) {
        const $ = new mindspore.schema.FusedBatchNorm();
        $.epsilon = reader.float32_(position, 4, 0.0001);
        $.momentum = reader.float32_(position, 6, 0.9);
        $.mode = reader.int64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.FusedBatchNorm();
        $.epsilon = reader.value(json.epsilon, 0.0001);
        $.momentum = reader.value(json.momentum, 0.9);
        $.mode = reader.int64(json.mode, 0n);
        return $;
    }
};

mindspore.schema.Gather = class Gather {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Gather();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Gather();
        return $;
    }
};

mindspore.schema.GatherNd = class GatherNd {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.GatherNd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.GatherNd();
        return $;
    }
};

mindspore.schema.Greater = class Greater {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Greater();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Greater();
        return $;
    }
};

mindspore.schema.GreaterEqual = class GreaterEqual {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.GreaterEqual();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.GreaterEqual();
        return $;
    }
};

mindspore.schema.HashtableLookup = class HashtableLookup {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.HashtableLookup();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.HashtableLookup();
        return $;
    }
};

mindspore.schema.InstanceNorm = class InstanceNorm {

    static decode(reader, position) {
        const $ = new mindspore.schema.InstanceNorm();
        $.epsilon = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.InstanceNorm();
        $.epsilon = reader.value(json.epsilon, 0);
        return $;
    }
};

mindspore.schema.LayerNormFusion = class LayerNormFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.LayerNormFusion();
        $.begin_norm_axis = reader.int64_(position, 4, 0n);
        $.epsilon = reader.float32_(position, 6, 0.00001);
        $.elementwise_affine = reader.bool_(position, 8, false);
        $.begin_params_axis = reader.int64_(position, 10, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LayerNormFusion();
        $.begin_norm_axis = reader.int64(json.begin_norm_axis, 0n);
        $.epsilon = reader.value(json.epsilon, 0.00001);
        $.elementwise_affine = reader.value(json.elementwise_affine, false);
        $.begin_params_axis = reader.int64(json.begin_params_axis, 0n);
        return $;
    }
};

mindspore.schema.LeakyRelu = class LeakyRelu {

    static decode(reader, position) {
        const $ = new mindspore.schema.LeakyRelu();
        $.negative_slope = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LeakyRelu();
        $.negative_slope = reader.value(json.negative_slope, 0);
        return $;
    }
};

mindspore.schema.Less = class Less {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Less();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Less();
        return $;
    }
};

mindspore.schema.LessEqual = class LessEqual {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.LessEqual();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.LessEqual();
        return $;
    }
};

mindspore.schema.Log = class Log {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Log();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Log();
        return $;
    }
};

mindspore.schema.LogGrad = class LogGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.LogGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.LogGrad();
        return $;
    }
};

mindspore.schema.LogicalAnd = class LogicalAnd {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.LogicalAnd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.LogicalAnd();
        return $;
    }
};

mindspore.schema.LogicalNot = class LogicalNot {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.LogicalNot();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.LogicalNot();
        return $;
    }
};

mindspore.schema.LogicalOr = class LogicalOr {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.LogicalOr();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.LogicalOr();
        return $;
    }
};

mindspore.schema.LpNormalization = class LpNormalization {

    static decode(reader, position) {
        const $ = new mindspore.schema.LpNormalization();
        $.axis = reader.int64_(position, 4, 0n);
        $.p = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LpNormalization();
        $.axis = reader.int64(json.axis, 0n);
        $.p = reader.int64(json.p, 0n);
        return $;
    }
};

mindspore.schema.LRN = class LRN {

    static decode(reader, position) {
        const $ = new mindspore.schema.LRN();
        $.depth_radius = reader.int64_(position, 4, 0n);
        $.bias = reader.float32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        $.norm_region = reader.string_(position, 12, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LRN();
        $.depth_radius = reader.int64(json.depth_radius, 0n);
        $.bias = reader.value(json.bias, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
        $.norm_region = reader.value(json.norm_region, null);
        return $;
    }
};

mindspore.schema.LshProjection = class LshProjection {

    static decode(reader, position) {
        const $ = new mindspore.schema.LshProjection();
        $.type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LshProjection();
        $.type = mindspore.schema.LshProjectionType[json.type];
        return $;
    }
};

mindspore.schema.LSTM = class LSTM {

    static decode(reader, position) {
        const $ = new mindspore.schema.LSTM();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0n);
        $.hidden_size = reader.int64_(position, 10, 0n);
        $.num_layers = reader.int64_(position, 12, 0n);
        $.num_directions = reader.int64_(position, 14, 0n);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        $.proj_size = reader.int64_(position, 22, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LSTM();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.int64(json.input_size, 0n);
        $.hidden_size = reader.int64(json.hidden_size, 0n);
        $.num_layers = reader.int64(json.num_layers, 0n);
        $.num_directions = reader.int64(json.num_directions, 0n);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        $.proj_size = reader.int64(json.proj_size, 0n);
        return $;
    }
};

mindspore.schema.LSTMGrad = class LSTMGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.LSTMGrad();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0n);
        $.hidden_size = reader.int64_(position, 10, 0n);
        $.num_layers = reader.int64_(position, 12, 0n);
        $.num_directions = reader.int64_(position, 14, 0n);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LSTMGrad();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.int64(json.input_size, 0n);
        $.hidden_size = reader.int64(json.hidden_size, 0n);
        $.num_layers = reader.int64(json.num_layers, 0n);
        $.num_directions = reader.int64(json.num_directions, 0n);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

mindspore.schema.L2NormalizeFusion = class L2NormalizeFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.L2NormalizeFusion();
        $.axis = reader.int64s_(position, 4);
        $.epsilon = reader.float32_(position, 6, 0);
        $.activation_type = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.L2NormalizeFusion();
        $.axis = reader.array(json.axis);
        $.epsilon = reader.value(json.epsilon, 0);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.MatMulFusion = class MatMulFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.MatMulFusion();
        $.transpose_a = reader.bool_(position, 4, false);
        $.transpose_b = reader.bool_(position, 6, false);
        $.activation_type = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.MatMulFusion();
        $.transpose_a = reader.value(json.transpose_a, false);
        $.transpose_b = reader.value(json.transpose_b, false);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.Maximum = class Maximum {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Maximum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Maximum();
        return $;
    }
};

mindspore.schema.MaximumGrad = class MaximumGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.MaximumGrad();
        $.grad_x = reader.bool_(position, 4, false);
        $.grad_y = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.MaximumGrad();
        $.grad_x = reader.value(json.grad_x, false);
        $.grad_y = reader.value(json.grad_y, false);
        return $;
    }
};

mindspore.schema.MaxPoolFusion = class MaxPoolFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.MaxPoolFusion();
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
        const $ = new mindspore.schema.MaxPoolFusion();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad = reader.array(json.pad);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.round_mode = mindspore.schema.RoundMode[json.round_mode];
        $.format = mindspore.schema.Format[json.format];
        $.global = reader.value(json.global, false);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.MaxPoolGrad = class MaxPoolGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.MaxPoolGrad();
        $.kernel_size = reader.int64s_(position, 4);
        $.strides = reader.int64s_(position, 6);
        $.pad_mode = reader.int8_(position, 8, 0);
        $.format = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.MaxPoolGrad();
        $.kernel_size = reader.array(json.kernel_size);
        $.strides = reader.array(json.strides);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.format = mindspore.schema.Format[json.format];
        return $;
    }
};

mindspore.schema.SwitchLayer = class SwitchLayer {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SwitchLayer();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SwitchLayer();
        return $;
    }
};

mindspore.schema.Mfcc = class Mfcc {

    static decode(reader, position) {
        const $ = new mindspore.schema.Mfcc();
        $.freq_upper_limit = reader.float32_(position, 4, 0);
        $.freq_lower_limit = reader.float32_(position, 6, 0);
        $.filter_bank_channel_num = reader.int64_(position, 8, 0n);
        $.dct_coeff_num = reader.int64_(position, 10, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Mfcc();
        $.freq_upper_limit = reader.value(json.freq_upper_limit, 0);
        $.freq_lower_limit = reader.value(json.freq_lower_limit, 0);
        $.filter_bank_channel_num = reader.int64(json.filter_bank_channel_num, 0n);
        $.dct_coeff_num = reader.int64(json.dct_coeff_num, 0n);
        return $;
    }
};

mindspore.schema.Minimum = class Minimum {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Minimum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Minimum();
        return $;
    }
};

mindspore.schema.MinimumGrad = class MinimumGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.MinimumGrad();
        $.grad_x = reader.bool_(position, 4, false);
        $.grad_y = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.MinimumGrad();
        $.grad_x = reader.value(json.grad_x, false);
        $.grad_y = reader.value(json.grad_y, false);
        return $;
    }
};

mindspore.schema.Mod = class Mod {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Mod();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Mod();
        return $;
    }
};

mindspore.schema.MulFusion = class MulFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.MulFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.MulFusion();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.MulGrad = class MulGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.MulGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.MulGrad();
        return $;
    }
};

mindspore.schema.Neg = class Neg {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Neg();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Neg();
        return $;
    }
};

mindspore.schema.NegGrad = class NegGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.NegGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.NegGrad();
        return $;
    }
};

mindspore.schema.NotEqual = class NotEqual {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.NotEqual();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.NotEqual();
        return $;
    }
};

mindspore.schema.NonMaxSuppression = class NonMaxSuppression {

    static decode(reader, position) {
        const $ = new mindspore.schema.NonMaxSuppression();
        $.center_point_box = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.NonMaxSuppression();
        $.center_point_box = reader.int64(json.center_point_box, 0n);
        return $;
    }
};

mindspore.schema.OneHot = class OneHot {

    static decode(reader, position) {
        const $ = new mindspore.schema.OneHot();
        $.axis = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.OneHot();
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.OnesLike = class OnesLike {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.OnesLike();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.OnesLike();
        return $;
    }
};

mindspore.schema.PadFusion = class PadFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.PadFusion();
        $.paddings = reader.table(position, 4, mindspore.schema.Vec2D);
        $.padding_mode = reader.int8_(position, 6, 0);
        $.constant_value = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.PadFusion();
        $.paddings = reader.object(json.paddings, mindspore.schema.Vec2D);
        $.padding_mode = mindspore.schema.PaddingMode[json.padding_mode];
        $.constant_value = reader.value(json.constant_value, 0);
        return $;
    }
};

mindspore.schema.PartialFusion = class PartialFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.PartialFusion();
        $.sub_graph_index = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.PartialFusion();
        $.sub_graph_index = reader.int64(json.sub_graph_index, 0n);
        return $;
    }
};

mindspore.schema.PowerGrad = class PowerGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.PowerGrad();
        $.power = reader.float32_(position, 4, 0);
        $.scale = reader.float32_(position, 6, 0);
        $.shift = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.PowerGrad();
        $.power = reader.value(json.power, 0);
        $.scale = reader.value(json.scale, 0);
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

mindspore.schema.PowFusion = class PowFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.PowFusion();
        $.scale = reader.float32_(position, 4, 1);
        $.shift = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.PowFusion();
        $.scale = reader.value(json.scale, 1);
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

mindspore.schema.PriorBox = class PriorBox {

    static decode(reader, position) {
        const $ = new mindspore.schema.PriorBox();
        $.min_sizes = reader.int64s_(position, 4);
        $.max_sizes = reader.int64s_(position, 6);
        $.aspect_ratios = reader.array(position, 8, Float32Array);
        $.variances = reader.array(position, 10, Float32Array);
        $.image_size_w = reader.int64_(position, 12, 0n);
        $.image_size_h = reader.int64_(position, 14, 0n);
        $.step_w = reader.float32_(position, 16, 0);
        $.step_h = reader.float32_(position, 18, 0);
        $.clip = reader.bool_(position, 20, false);
        $.flip = reader.bool_(position, 22, false);
        $.offset = reader.float32_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.PriorBox();
        $.min_sizes = reader.array(json.min_sizes);
        $.max_sizes = reader.array(json.max_sizes);
        $.aspect_ratios = reader.array(json.aspect_ratios, Float32Array);
        $.variances = reader.array(json.variances, Float32Array);
        $.image_size_w = reader.int64(json.image_size_w, 0n);
        $.image_size_h = reader.int64(json.image_size_h, 0n);
        $.step_w = reader.value(json.step_w, 0);
        $.step_h = reader.value(json.step_h, 0);
        $.clip = reader.value(json.clip, false);
        $.flip = reader.value(json.flip, false);
        $.offset = reader.value(json.offset, 0);
        return $;
    }
};

mindspore.schema.PReLUFusion = class PReLUFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.PReLUFusion();
        $.channel_shared = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.PReLUFusion();
        $.channel_shared = reader.value(json.channel_shared, false);
        return $;
    }
};

mindspore.schema.Rank = class Rank {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Rank();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Rank();
        return $;
    }
};

mindspore.schema.Range = class Range {

    static decode(reader, position) {
        const $ = new mindspore.schema.Range();
        $.d_type = reader.int64_(position, 4, 0n);
        $.start = reader.int64_(position, 6, 0n);
        $.limit = reader.int64_(position, 8, 0n);
        $.delta = reader.int64_(position, 10, 1n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Range();
        $.d_type = reader.int64(json.d_type, 0n);
        $.start = reader.int64(json.start, 0n);
        $.limit = reader.int64(json.limit, 0n);
        $.delta = reader.int64(json.delta, 1n);
        return $;
    }
};

mindspore.schema.Reciprocal = class Reciprocal {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Reciprocal();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Reciprocal();
        return $;
    }
};

mindspore.schema.RealDiv = class RealDiv {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.RealDiv();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.RealDiv();
        return $;
    }
};

mindspore.schema.ReduceFusion = class ReduceFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.ReduceFusion();
        $.keep_dims = reader.bool_(position, 4, false);
        $.mode = reader.int8_(position, 6, 0);
        $.reduce_to_end = reader.bool_(position, 8, false);
        $.coeff = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ReduceFusion();
        $.keep_dims = reader.value(json.keep_dims, false);
        $.mode = mindspore.schema.ReduceMode[json.mode];
        $.reduce_to_end = reader.value(json.reduce_to_end, false);
        $.coeff = reader.value(json.coeff, 0);
        return $;
    }
};

mindspore.schema.Reshape = class Reshape {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Reshape();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Reshape();
        return $;
    }
};

mindspore.schema.Resize = class Resize {

    static decode(reader, position) {
        const $ = new mindspore.schema.Resize();
        $.format = reader.int32_(position, 4, 0);
        $.method = reader.int8_(position, 6, 0);
        $.new_height = reader.int64_(position, 8, 0n);
        $.new_width = reader.int64_(position, 10, 0n);
        $.preserve_aspect_ratio = reader.bool_(position, 12, false);
        $.coordinate_transform_mode = reader.int8_(position, 14, 0);
        $.cubic_coeff = reader.float32_(position, 16, 0);
        $.exclude_outside = reader.int64_(position, 18, 0n);
        $.extrapolation_value = reader.float32_(position, 20, 0);
        $.nearest_mode = reader.int8_(position, 22, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Resize();
        $.format = mindspore.schema.Format[json.format];
        $.method = mindspore.schema.ResizeMethod[json.method];
        $.new_height = reader.int64(json.new_height, 0n);
        $.new_width = reader.int64(json.new_width, 0n);
        $.preserve_aspect_ratio = reader.value(json.preserve_aspect_ratio, false);
        $.coordinate_transform_mode = mindspore.schema.CoordinateTransformMode[json.coordinate_transform_mode];
        $.cubic_coeff = reader.value(json.cubic_coeff, 0);
        $.exclude_outside = reader.int64(json.exclude_outside, 0n);
        $.extrapolation_value = reader.value(json.extrapolation_value, 0);
        $.nearest_mode = mindspore.schema.NearestMode[json.nearest_mode];
        return $;
    }
};

mindspore.schema.ReverseSequence = class ReverseSequence {

    static decode(reader, position) {
        const $ = new mindspore.schema.ReverseSequence();
        $.seq_dim = reader.int64_(position, 4, 0n);
        $.batch_dim = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ReverseSequence();
        $.seq_dim = reader.int64(json.seq_dim, 0n);
        $.batch_dim = reader.int64(json.batch_dim, 0n);
        return $;
    }
};

mindspore.schema.ReverseV2 = class ReverseV2 {

    static decode(reader, position) {
        const $ = new mindspore.schema.ReverseV2();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ReverseV2();
        $.axis = reader.array(json.axis);
        return $;
    }
};

mindspore.schema.Rfft = class Rfft {

    static decode(reader, position) {
        const $ = new mindspore.schema.Rfft();
        $.fft_length = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Rfft();
        $.fft_length = reader.int64(json.fft_length, 0n);
        return $;
    }
};

mindspore.schema.ROIPooling = class ROIPooling {

    static decode(reader, position) {
        const $ = new mindspore.schema.ROIPooling();
        $.pooled_h = reader.int64_(position, 4, 0n);
        $.pooled_w = reader.int64_(position, 6, 0n);
        $.scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ROIPooling();
        $.pooled_h = reader.int64(json.pooled_h, 0n);
        $.pooled_w = reader.int64(json.pooled_w, 0n);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

mindspore.schema.Round = class Round {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Round();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Round();
        return $;
    }
};

mindspore.schema.Rsqrt = class Rsqrt {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Rsqrt();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Rsqrt();
        return $;
    }
};

mindspore.schema.QuantDTypeCast = class QuantDTypeCast {

    static decode(reader, position) {
        const $ = new mindspore.schema.QuantDTypeCast();
        $.src_t = reader.int64_(position, 4, 0n);
        $.dst_t = reader.int64_(position, 6, 0n);
        $.axis = reader.int64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.QuantDTypeCast();
        $.src_t = reader.int64(json.src_t, 0n);
        $.dst_t = reader.int64(json.dst_t, 0n);
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.ScaleFusion = class ScaleFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.ScaleFusion();
        $.axis = reader.int64_(position, 4, 0n);
        $.activation_type = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ScaleFusion();
        $.axis = reader.int64(json.axis, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.ScatterNd = class ScatterNd {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.ScatterNd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.ScatterNd();
        return $;
    }
};

mindspore.schema.SGD = class SGD {

    static decode(reader, position) {
        const $ = new mindspore.schema.SGD();
        $.nesterov = reader.bool_(position, 4, false);
        $.dampening = reader.float32_(position, 6, 0);
        $.weight_decay = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SGD();
        $.nesterov = reader.value(json.nesterov, false);
        $.dampening = reader.value(json.dampening, 0);
        $.weight_decay = reader.value(json.weight_decay, 0);
        return $;
    }
};

mindspore.schema.Shape = class Shape {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Shape();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Shape();
        return $;
    }
};

mindspore.schema.SigmoidCrossEntropyWithLogits = class SigmoidCrossEntropyWithLogits {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SigmoidCrossEntropyWithLogits();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SigmoidCrossEntropyWithLogits();
        return $;
    }
};

mindspore.schema.SigmoidCrossEntropyWithLogitsGrad = class SigmoidCrossEntropyWithLogitsGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SigmoidCrossEntropyWithLogitsGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SigmoidCrossEntropyWithLogitsGrad();
        return $;
    }
};

mindspore.schema.Sin = class Sin {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Sin();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Sin();
        return $;
    }
};

mindspore.schema.SkipGram = class SkipGram {

    static decode(reader, position) {
        const $ = new mindspore.schema.SkipGram();
        $.include_all_grams = reader.bool_(position, 4, false);
        $.max_skip_size = reader.int64_(position, 6, 0n);
        $.ngram_size = reader.int64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SkipGram();
        $.include_all_grams = reader.value(json.include_all_grams, false);
        $.max_skip_size = reader.int64(json.max_skip_size, 0n);
        $.ngram_size = reader.int64(json.ngram_size, 0n);
        return $;
    }
};

mindspore.schema.SliceFusion = class SliceFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.SliceFusion();
        $.axes = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SliceFusion();
        $.axes = reader.array(json.axes);
        return $;
    }
};

mindspore.schema.SmoothL1Loss = class SmoothL1Loss {

    static decode(reader, position) {
        const $ = new mindspore.schema.SmoothL1Loss();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SmoothL1Loss();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

mindspore.schema.SmoothL1LossGrad = class SmoothL1LossGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.SmoothL1LossGrad();
        $.beta = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SmoothL1LossGrad();
        $.beta = reader.value(json.beta, 0);
        return $;
    }
};

mindspore.schema.Softmax = class Softmax {

    static decode(reader, position) {
        const $ = new mindspore.schema.Softmax();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Softmax();
        $.axis = reader.array(json.axis);
        return $;
    }
};

mindspore.schema.SoftmaxCrossEntropyWithLogits = class SoftmaxCrossEntropyWithLogits {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SoftmaxCrossEntropyWithLogits();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SoftmaxCrossEntropyWithLogits();
        return $;
    }
};

mindspore.schema.SpaceToBatch = class SpaceToBatch {

    static decode(reader, position) {
        const $ = new mindspore.schema.SpaceToBatch();
        $.block_size = reader.int64s_(position, 4);
        $.paddings = reader.table(position, 6, mindspore.schema.Vec2D);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SpaceToBatch();
        $.block_size = reader.array(json.block_size);
        $.paddings = reader.object(json.paddings, mindspore.schema.Vec2D);
        return $;
    }
};

mindspore.schema.SpaceToBatchND = class SpaceToBatchND {

    static decode(reader, position) {
        const $ = new mindspore.schema.SpaceToBatchND();
        $.block_shape = reader.int64s_(position, 4);
        $.paddings = reader.table(position, 6, mindspore.schema.Vec2D);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SpaceToBatchND();
        $.block_shape = reader.array(json.block_shape);
        $.paddings = reader.object(json.paddings, mindspore.schema.Vec2D);
        return $;
    }
};

mindspore.schema.SpaceToDepth = class SpaceToDepth {

    static decode(reader, position) {
        const $ = new mindspore.schema.SpaceToDepth();
        $.block_size = reader.int64_(position, 4, 0n);
        $.format = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SpaceToDepth();
        $.block_size = reader.int64(json.block_size, 0n);
        $.format = mindspore.schema.Format[json.format];
        return $;
    }
};

mindspore.schema.SparseSoftmaxCrossEntropyWithLogits = class SparseSoftmaxCrossEntropyWithLogits {

    static decode(reader, position) {
        const $ = new mindspore.schema.SparseSoftmaxCrossEntropyWithLogits();
        $.is_grad = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SparseSoftmaxCrossEntropyWithLogits();
        $.is_grad = reader.value(json.is_grad, false);
        return $;
    }
};

mindspore.schema.SparseToDense = class SparseToDense {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SparseToDense();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SparseToDense();
        return $;
    }
};

mindspore.schema.Split = class Split {

    static decode(reader, position) {
        const $ = new mindspore.schema.Split();
        $.output_num = reader.int64_(position, 4, 0n);
        $.size_splits = reader.int64s_(position, 6);
        $.axis = reader.int64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Split();
        $.output_num = reader.int64(json.output_num, 0n);
        $.size_splits = reader.array(json.size_splits);
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.Sqrt = class Sqrt {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Sqrt();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Sqrt();
        return $;
    }
};

mindspore.schema.Squeeze = class Squeeze {

    static decode(reader, position) {
        const $ = new mindspore.schema.Squeeze();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Squeeze();
        $.axis = reader.array(json.axis);
        return $;
    }
};

mindspore.schema.Square = class Square {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Square();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Square();
        return $;
    }
};

mindspore.schema.SquaredDifference = class SquaredDifference {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SquaredDifference();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SquaredDifference();
        return $;
    }
};

mindspore.schema.Stack = class Stack {

    static decode(reader, position) {
        const $ = new mindspore.schema.Stack();
        $.axis = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Stack();
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.StridedSlice = class StridedSlice {

    static decode(reader, position) {
        const $ = new mindspore.schema.StridedSlice();
        $.begin_mask = reader.int64_(position, 4, 0n);
        $.end_mask = reader.int64_(position, 6, 0n);
        $.ellipsis_mask = reader.int64_(position, 8, 0n);
        $.new_axis_mask = reader.int64_(position, 10, 0n);
        $.shrink_axis_mask = reader.int64_(position, 12, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.StridedSlice();
        $.begin_mask = reader.int64(json.begin_mask, 0n);
        $.end_mask = reader.int64(json.end_mask, 0n);
        $.ellipsis_mask = reader.int64(json.ellipsis_mask, 0n);
        $.new_axis_mask = reader.int64(json.new_axis_mask, 0n);
        $.shrink_axis_mask = reader.int64(json.shrink_axis_mask, 0n);
        return $;
    }
};

mindspore.schema.SubFusion = class SubFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.SubFusion();
        $.activation_type = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SubFusion();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        return $;
    }
};

mindspore.schema.SubGrad = class SubGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SubGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SubGrad();
        return $;
    }
};

mindspore.schema.Switch = class Switch {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Switch();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Switch();
        return $;
    }
};

mindspore.schema.TensorListFromTensor = class TensorListFromTensor {

    static decode(reader, position) {
        const $ = new mindspore.schema.TensorListFromTensor();
        $.element_dtype = reader.int64_(position, 4, 0n);
        $.shape_type = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TensorListFromTensor();
        $.element_dtype = reader.int64(json.element_dtype, 0n);
        $.shape_type = reader.int64(json.shape_type, 0n);
        return $;
    }
};

mindspore.schema.TensorListGetItem = class TensorListGetItem {

    static decode(reader, position) {
        const $ = new mindspore.schema.TensorListGetItem();
        $.element_dtype = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TensorListGetItem();
        $.element_dtype = reader.int64(json.element_dtype, 0n);
        return $;
    }
};

mindspore.schema.TensorListReserve = class TensorListReserve {

    static decode(reader, position) {
        const $ = new mindspore.schema.TensorListReserve();
        $.element_dtype = reader.int64_(position, 4, 0n);
        $.shape_type = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TensorListReserve();
        $.element_dtype = reader.int64(json.element_dtype, 0n);
        $.shape_type = reader.int64(json.shape_type, 0n);
        return $;
    }
};

mindspore.schema.TensorListSetItem = class TensorListSetItem {

    static decode(reader, position) {
        const $ = new mindspore.schema.TensorListSetItem();
        $.element_dtype = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TensorListSetItem();
        $.element_dtype = reader.int64(json.element_dtype, 0n);
        return $;
    }
};

mindspore.schema.TensorListStack = class TensorListStack {

    static decode(reader, position) {
        const $ = new mindspore.schema.TensorListStack();
        $.num_elements = reader.int64_(position, 4, 0n);
        $.element_dtype = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TensorListStack();
        $.num_elements = reader.int64(json.num_elements, 0n);
        $.element_dtype = reader.int64(json.element_dtype, 0n);
        return $;
    }
};

mindspore.schema.TileFusion = class TileFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.TileFusion();
        $.dims = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TileFusion();
        $.dims = reader.array(json.dims);
        return $;
    }
};

mindspore.schema.TopKFusion = class TopKFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.TopKFusion();
        $.sorted = reader.bool_(position, 4, true);
        $.axis = reader.int64_(position, 6, 0n);
        $.largest = reader.int64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TopKFusion();
        $.sorted = reader.value(json.sorted, true);
        $.axis = reader.int64(json.axis, 0n);
        $.largest = reader.int64(json.largest, 0n);
        return $;
    }
};

mindspore.schema.Transpose = class Transpose {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Transpose();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Transpose();
        return $;
    }
};

mindspore.schema.Unique = class Unique {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Unique();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Unique();
        return $;
    }
};

mindspore.schema.UnsortedSegmentSum = class UnsortedSegmentSum {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.UnsortedSegmentSum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.UnsortedSegmentSum();
        return $;
    }
};

mindspore.schema.Unsqueeze = class Unsqueeze {

    static decode(reader, position) {
        const $ = new mindspore.schema.Unsqueeze();
        $.axis = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Unsqueeze();
        $.axis = reader.array(json.axis);
        return $;
    }
};

mindspore.schema.Unstack = class Unstack {

    static decode(reader, position) {
        const $ = new mindspore.schema.Unstack();
        $.axis = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Unstack();
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.Where = class Where {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Where();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Where();
        return $;
    }
};

mindspore.schema.ZerosLike = class ZerosLike {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.ZerosLike();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.ZerosLike();
        return $;
    }
};

mindspore.schema.Select = class Select {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Select();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Select();
        return $;
    }
};

mindspore.schema.GRU = class GRU {

    static decode(reader, position) {
        const $ = new mindspore.schema.GRU();
        $.bidirectional = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.GRU();
        $.bidirectional = reader.value(json.bidirectional, false);
        return $;
    }
};

mindspore.schema.NonZero = class NonZero {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.NonZero();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.NonZero();
        return $;
    }
};

mindspore.schema.InvertPermutation = class InvertPermutation {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.InvertPermutation();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.InvertPermutation();
        return $;
    }
};

mindspore.schema.Size = class Size {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Size();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Size();
        return $;
    }
};

mindspore.schema.RandomStandardNormal = class RandomStandardNormal {

    static decode(reader, position) {
        const $ = new mindspore.schema.RandomStandardNormal();
        $.seed = reader.int64_(position, 4, 0n);
        $.seed2 = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.RandomStandardNormal();
        $.seed = reader.int64(json.seed, 0n);
        $.seed2 = reader.int64(json.seed2, 0n);
        return $;
    }
};

mindspore.schema.CropAndResize = class CropAndResize {

    static decode(reader, position) {
        const $ = new mindspore.schema.CropAndResize();
        $.method = reader.int8_(position, 4, 0);
        $.extrapolation_value = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.CropAndResize();
        $.method = mindspore.schema.ResizeMethod[json.method];
        $.extrapolation_value = reader.value(json.extrapolation_value, 0);
        return $;
    }
};

mindspore.schema.Erf = class Erf {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Erf();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Erf();
        return $;
    }
};

mindspore.schema.StridedSliceGrad = class StridedSliceGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.StridedSliceGrad();
        $.begin_mask = reader.int64_(position, 4, 0n);
        $.end_mask = reader.int64_(position, 6, 0n);
        $.ellipsis_mask = reader.int64_(position, 8, 0n);
        $.new_axis_mask = reader.int64_(position, 10, 0n);
        $.shrink_axis_mask = reader.int64_(position, 12, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.StridedSliceGrad();
        $.begin_mask = reader.int64(json.begin_mask, 0n);
        $.end_mask = reader.int64(json.end_mask, 0n);
        $.ellipsis_mask = reader.int64(json.ellipsis_mask, 0n);
        $.new_axis_mask = reader.int64(json.new_axis_mask, 0n);
        $.shrink_axis_mask = reader.int64(json.shrink_axis_mask, 0n);
        return $;
    }
};

mindspore.schema.IsFinite = class IsFinite {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.IsFinite();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.IsFinite();
        return $;
    }
};

mindspore.schema.LinSpace = class LinSpace {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.LinSpace();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.LinSpace();
        return $;
    }
};

mindspore.schema.UniformReal = class UniformReal {

    static decode(reader, position) {
        const $ = new mindspore.schema.UniformReal();
        $.seed = reader.int64_(position, 4, 0n);
        $.seed2 = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.UniformReal();
        $.seed = reader.int64(json.seed, 0n);
        $.seed2 = reader.int64(json.seed2, 0n);
        return $;
    }
};

mindspore.schema.AbsGrad = class AbsGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.AbsGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.AbsGrad();
        return $;
    }
};

mindspore.schema.RsqrtGrad = class RsqrtGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.RsqrtGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.RsqrtGrad();
        return $;
    }
};

mindspore.schema.SqrtGrad = class SqrtGrad {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SqrtGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SqrtGrad();
        return $;
    }
};

mindspore.schema.LayerNormGrad = class LayerNormGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.LayerNormGrad();
        $.begin_norm_axis = reader.int64_(position, 4, 0n);
        $.begin_params_axis = reader.int64_(position, 6, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LayerNormGrad();
        $.begin_norm_axis = reader.int64(json.begin_norm_axis, 0n);
        $.begin_params_axis = reader.int64(json.begin_params_axis, 0n);
        return $;
    }
};

mindspore.schema.ResizeGrad = class ResizeGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.ResizeGrad();
        $.method = reader.int8_(position, 4, 0);
        $.align_corners = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ResizeGrad();
        $.method = mindspore.schema.ResizeMethod[json.method];
        $.align_corners = reader.value(json.align_corners, false);
        return $;
    }
};

mindspore.schema.Splice = class Splice {

    static decode(reader, position) {
        const $ = new mindspore.schema.Splice();
        $.context = reader.int64s_(position, 4);
        $.forward_indexes = reader.int64s_(position, 6);
        $.output_dim = reader.int64_(position, 8, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Splice();
        $.context = reader.array(json.context);
        $.forward_indexes = reader.array(json.forward_indexes);
        $.output_dim = reader.int64(json.output_dim, 0n);
        return $;
    }
};

mindspore.schema.LogSoftmax = class LogSoftmax {

    static decode(reader, position) {
        const $ = new mindspore.schema.LogSoftmax();
        $.axis = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LogSoftmax();
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.Call = class Call {

    static decode(reader, position) {
        const $ = new mindspore.schema.Call();
        $.is_tail_call = reader.bool_(position, 4, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Call();
        $.is_tail_call = reader.value(json.is_tail_call, true);
        return $;
    }
};

mindspore.schema.CumSum = class CumSum {

    static decode(reader, position) {
        const $ = new mindspore.schema.CumSum();
        $.exclusive = reader.bool_(position, 4, false);
        $.reverse = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.CumSum();
        $.exclusive = reader.value(json.exclusive, false);
        $.reverse = reader.value(json.reverse, false);
        return $;
    }
};

mindspore.schema.Custom = class Custom {

    static decode(reader, position) {
        const $ = new mindspore.schema.Custom();
        $.type = reader.string_(position, 4, null);
        $.attr = reader.tables(position, 6, mindspore.schema.Attribute);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Custom();
        $.type = reader.value(json.type, null);
        $.attr = reader.objects(json.attr, mindspore.schema.Attribute);
        return $;
    }
};

mindspore.schema.SplitWithOverlap = class SplitWithOverlap {

    static decode(reader, position) {
        const $ = new mindspore.schema.SplitWithOverlap();
        $.split_dim = reader.int64_(position, 4, 0n);
        $.number_split = reader.int64_(position, 6, 0n);
        $.ratio = reader.int64s_(position, 8);
        $.extend_top = reader.int64s_(position, 10);
        $.extend_bottom = reader.int64s_(position, 12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SplitWithOverlap();
        $.split_dim = reader.int64(json.split_dim, 0n);
        $.number_split = reader.int64(json.number_split, 0n);
        $.ratio = reader.array(json.ratio);
        $.extend_top = reader.array(json.extend_top);
        $.extend_bottom = reader.array(json.extend_bottom);
        return $;
    }
};

mindspore.schema.GenOP = class GenOP {

    static decode(reader, position) {
        const $ = new mindspore.schema.GenOP();
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
        $.mode = reader.int64_(position, 26, 0n);
        $.group = reader.int64_(position, 28, 0n);
        $.in_channel = reader.int64_(position, 30, 0n);
        $.out_channel = reader.int64_(position, 32, 0n);
        $.eltwise_mode = reader.int8_(position, 34, 0);
        $.has_bias = reader.bool_(position, 36, false);
        $.use_axis = reader.bool_(position, 38, false);
        $.axis = reader.int64_(position, 40, 0n);
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
        const $ = new mindspore.schema.GenOP();
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        $.alpha = reader.value(json.alpha, 0);
        $.min_val = reader.value(json.min_val, 0);
        $.max_val = reader.value(json.max_val, 0);
        $.is_training = reader.value(json.is_training, false);
        $.format = mindspore.schema.Format[json.format];
        $.kernel_size = reader.array(json.kernel_size);
        $.stride = reader.array(json.stride);
        $.dilation = reader.array(json.dilation);
        $.pad_mode = mindspore.schema.PadMode[json.pad_mode];
        $.pad_list = reader.array(json.pad_list);
        $.mode = reader.int64(json.mode, 0n);
        $.group = reader.int64(json.group, 0n);
        $.in_channel = reader.int64(json.in_channel, 0n);
        $.out_channel = reader.int64(json.out_channel, 0n);
        $.eltwise_mode = mindspore.schema.EltwiseMode[json.eltwise_mode];
        $.has_bias = reader.value(json.has_bias, false);
        $.use_axis = reader.value(json.use_axis, false);
        $.axis = reader.int64(json.axis, 0n);
        $.epsilon = reader.value(json.epsilon, 0.0001);
        $.momentum = reader.value(json.momentum, 0.9);
        $.transpose_a = reader.value(json.transpose_a, false);
        $.transpose_b = reader.value(json.transpose_b, false);
        $.pad = reader.array(json.pad);
        $.round_mode = mindspore.schema.RoundMode[json.round_mode];
        $.global = reader.value(json.global, false);
        $.channel_shared = reader.value(json.channel_shared, false);
        $.axes = reader.array(json.axes);
        $.keep_dims = reader.value(json.keep_dims, false);
        $.reduce_mode = mindspore.schema.ReduceMode[json.reduce_mode];
        $.reduce_to_end = reader.value(json.reduce_to_end, false);
        $.coeff = reader.value(json.coeff, 0);
        return $;
    }
};

mindspore.schema.RaggedRange = class RaggedRange {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.RaggedRange();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.RaggedRange();
        return $;
    }
};

mindspore.schema.GLU = class GLU {

    static decode(reader, position) {
        const $ = new mindspore.schema.GLU();
        $.axis = reader.int64_(position, 4, -1n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.GLU();
        $.axis = reader.int64(json.axis, -1n);
        return $;
    }
};

mindspore.schema.TensorArray = class TensorArray {

    static decode(reader, position) {
        const $ = new mindspore.schema.TensorArray();
        $.dynamic_size = reader.bool_(position, 4, false);
        $.identical_element_shapes = reader.bool_(position, 6, false);
        $.element_shape = reader.array(position, 8, Int32Array);
        $.data_type = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.TensorArray();
        $.dynamic_size = reader.value(json.dynamic_size, false);
        $.identical_element_shapes = reader.value(json.identical_element_shapes, false);
        $.element_shape = reader.array(json.element_shape, Int32Array);
        $.data_type = reader.value(json.data_type, 0);
        return $;
    }
};

mindspore.schema.TensorArrayRead = class TensorArrayRead {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.TensorArrayRead();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.TensorArrayRead();
        return $;
    }
};

mindspore.schema.TensorArrayWrite = class TensorArrayWrite {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.TensorArrayWrite();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.TensorArrayWrite();
        return $;
    }
};

mindspore.schema.Affine = class Affine {

    static decode(reader, position) {
        const $ = new mindspore.schema.Affine();
        $.context = reader.int64s_(position, 4);
        $.output_dim = reader.int64_(position, 6, 0n);
        $.activation_type = reader.int8_(position, 8, 0);
        $.transpose_a = reader.bool_(position, 10, false);
        $.transpose_b = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Affine();
        $.context = reader.array(json.context);
        $.output_dim = reader.int64(json.output_dim, 0n);
        $.activation_type = mindspore.schema.ActivationType[json.activation_type];
        $.transpose_a = reader.value(json.transpose_a, false);
        $.transpose_b = reader.value(json.transpose_b, false);
        return $;
    }
};

mindspore.schema.ScatterNdUpdate = class ScatterNdUpdate {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.ScatterNdUpdate();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.ScatterNdUpdate();
        return $;
    }
};

mindspore.schema.AllGather = class AllGather {

    static decode(reader, position) {
        const $ = new mindspore.schema.AllGather();
        $.group = reader.string_(position, 4, null);
        $.rank_size = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.AllGather();
        $.group = reader.value(json.group, null);
        $.rank_size = reader.value(json.rank_size, 0);
        return $;
    }
};

mindspore.schema.ReduceScatter = class ReduceScatter {

    static decode(reader, position) {
        const $ = new mindspore.schema.ReduceScatter();
        $.group = reader.string_(position, 4, null);
        $.mode = reader.int8_(position, 6, 0);
        $.rank_size = reader.int32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ReduceScatter();
        $.group = reader.value(json.group, null);
        $.mode = mindspore.schema.ReduceMode[json.mode];
        $.rank_size = reader.value(json.rank_size, 0);
        return $;
    }
};

mindspore.schema.DynamicQuant = class DynamicQuant {

    static decode(reader, position) {
        const $ = new mindspore.schema.DynamicQuant();
        $.symmetric = reader.bool_(position, 4, false);
        $.dst_type = reader.int64_(position, 6, 32n);
        $.activation_channel = reader.bool_(position, 8, false);
        $.prefer_axis = reader.int64_(position, 10, 0n);
        $.transpose = reader.bool_(position, 12, false);
        $.prefer_axes = reader.array(position, 14, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.DynamicQuant();
        $.symmetric = reader.value(json.symmetric, false);
        $.dst_type = reader.int64(json.dst_type, 32n);
        $.activation_channel = reader.value(json.activation_channel, false);
        $.prefer_axis = reader.int64(json.prefer_axis, 0n);
        $.transpose = reader.value(json.transpose, false);
        $.prefer_axes = reader.array(json.prefer_axes, Int32Array);
        return $;
    }
};

mindspore.schema.LSTMGradData = class LSTMGradData {

    static decode(reader, position) {
        const $ = new mindspore.schema.LSTMGradData();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0n);
        $.hidden_size = reader.int64_(position, 10, 0n);
        $.num_layers = reader.int64_(position, 12, 0n);
        $.num_directions = reader.int64_(position, 14, 0n);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LSTMGradData();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.int64(json.input_size, 0n);
        $.hidden_size = reader.int64(json.hidden_size, 0n);
        $.num_layers = reader.int64(json.num_layers, 0n);
        $.num_directions = reader.int64(json.num_directions, 0n);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

mindspore.schema.LSTMGradWeight = class LSTMGradWeight {

    static decode(reader, position) {
        const $ = new mindspore.schema.LSTMGradWeight();
        $.bidirectional = reader.bool_(position, 4, false);
        $.has_bias = reader.bool_(position, 6, false);
        $.input_size = reader.int64_(position, 8, 0n);
        $.hidden_size = reader.int64_(position, 10, 0n);
        $.num_layers = reader.int64_(position, 12, 0n);
        $.num_directions = reader.int64_(position, 14, 0n);
        $.dropout = reader.float32_(position, 16, 0);
        $.zoneout_cell = reader.float32_(position, 18, 0);
        $.zoneout_hidden = reader.float32_(position, 20, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.LSTMGradWeight();
        $.bidirectional = reader.value(json.bidirectional, false);
        $.has_bias = reader.value(json.has_bias, false);
        $.input_size = reader.int64(json.input_size, 0n);
        $.hidden_size = reader.int64(json.hidden_size, 0n);
        $.num_layers = reader.int64(json.num_layers, 0n);
        $.num_directions = reader.int64(json.num_directions, 0n);
        $.dropout = reader.value(json.dropout, 0);
        $.zoneout_cell = reader.value(json.zoneout_cell, 0);
        $.zoneout_hidden = reader.value(json.zoneout_hidden, 0);
        return $;
    }
};

mindspore.schema.RandomNormal = class RandomNormal {

    static decode(reader, position) {
        const $ = new mindspore.schema.RandomNormal();
        $.seed = reader.float32_(position, 4, 0);
        $.mean = reader.float32_(position, 6, 0);
        $.scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.RandomNormal();
        $.seed = reader.value(json.seed, 0);
        $.mean = reader.value(json.mean, 0);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

mindspore.schema.NLLLoss = class NLLLoss {

    static decode(reader, position) {
        const $ = new mindspore.schema.NLLLoss();
        $.reduction = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.NLLLoss();
        $.reduction = mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

mindspore.schema.NLLLossGrad = class NLLLossGrad {

    static decode(reader, position) {
        const $ = new mindspore.schema.NLLLossGrad();
        $.reduction = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.NLLLossGrad();
        $.reduction = mindspore.schema.Reduction[json.reduction];
        return $;
    }
};

mindspore.schema.FormatTranspose = class FormatTranspose {

    static decode(reader, position) {
        const $ = new mindspore.schema.FormatTranspose();
        $.src_format = reader.int32_(position, 4, 1);
        $.dst_format = reader.int32_(position, 6, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.FormatTranspose();
        $.src_format = mindspore.schema.Format[json.src_format];
        $.dst_format = mindspore.schema.Format[json.dst_format];
        return $;
    }
};

mindspore.schema.GatherD = class GatherD {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.GatherD();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.GatherD();
        return $;
    }
};

mindspore.schema.GroupNormFusion = class GroupNormFusion {

    static decode(reader, position) {
        const $ = new mindspore.schema.GroupNormFusion();
        $.num_groups = reader.int64_(position, 4, 0n);
        $.epsilon = reader.float32_(position, 6, 0.00001);
        $.affine = reader.bool_(position, 8, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.GroupNormFusion();
        $.num_groups = reader.int64(json.num_groups, 0n);
        $.epsilon = reader.value(json.epsilon, 0.00001);
        $.affine = reader.value(json.affine, true);
        return $;
    }
};

mindspore.schema.Log1p = class Log1p {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Log1p();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Log1p();
        return $;
    }
};

mindspore.schema.TensorScatterAdd = class TensorScatterAdd {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.TensorScatterAdd();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.TensorScatterAdd();
        return $;
    }
};

mindspore.schema.SparseFillEmptyRows = class SparseFillEmptyRows {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SparseFillEmptyRows();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SparseFillEmptyRows();
        return $;
    }
};

mindspore.schema.SparseReshape = class SparseReshape {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SparseReshape();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SparseReshape();
        return $;
    }
};

mindspore.schema.SparseSegmentSum = class SparseSegmentSum {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.SparseSegmentSum();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.SparseSegmentSum();
        return $;
    }
};

mindspore.schema.ScatterElements = class ScatterElements {

    static decode(reader, position) {
        const $ = new mindspore.schema.ScatterElements();
        $.axis = reader.int64_(position, 4, 0n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ScatterElements();
        $.axis = reader.int64(json.axis, 0n);
        return $;
    }
};

mindspore.schema.Triu = class Triu {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Triu();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Triu();
        return $;
    }
};

mindspore.schema.Tril = class Tril {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.Tril();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.Tril();
        return $;
    }
};

mindspore.schema.AdamWeightDecay = class AdamWeightDecay {

    static decode(reader, position) {
        const $ = new mindspore.schema.AdamWeightDecay();
        $.use_locking = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.AdamWeightDecay();
        $.use_locking = reader.value(json.use_locking, false);
        return $;
    }
};

mindspore.schema.FillV2 = class FillV2 {

    static decode(/* reader, position */) {
        const $ = new mindspore.schema.FillV2();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new mindspore.schema.FillV2();
        return $;
    }
};

mindspore.schema.QuantParam = class QuantParam {

    static decode(reader, position) {
        const $ = new mindspore.schema.QuantParam();
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
        const $ = new mindspore.schema.QuantParam();
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

mindspore.schema.WeightQuantCompressType = {
    NONE: 0,
    INDEXING: 1,
    SPARSE: 2,
    FSE: 3,
    BITPACKING: 4,
    FSE_INT: 5,
    FSE_INFER: 6
};

mindspore.schema.ExternalData = class ExternalData {

    static decode(reader, position) {
        const $ = new mindspore.schema.ExternalData();
        $.checkSum = reader.string_(position, 4, null);
        $.location = reader.string_(position, 6, null);
        $.offset = reader.int64_(position, 8, 0n);
        $.length = reader.int64_(position, 10, -1n);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.ExternalData();
        $.checkSum = reader.value(json.checkSum, null);
        $.location = reader.value(json.location, null);
        $.offset = reader.int64(json.offset, 0n);
        $.length = reader.int64(json.length, -1n);
        return $;
    }
};

mindspore.schema.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new mindspore.schema.Tensor();
        $.nodeType = reader.int32_(position, 4, 0);
        $.dataType = reader.int32_(position, 6, 0);
        $.dims = reader.array(position, 8, Int32Array);
        $.format = reader.int32_(position, 10, 0);
        $.refCount = reader.int32_(position, 12, 0);
        $.offset = reader.int32_(position, 14, 0);
        $.data = reader.array(position, 16, Uint8Array);
        $.quantParams = reader.tables(position, 18, mindspore.schema.QuantParam);
        $.quantClusters = reader.array(position, 20, Float32Array);
        $.name = reader.string_(position, 22, null);
        $.enableHuffmanCode = reader.bool_(position, 24, false);
        $.weightQuantCompressType = reader.int32_(position, 26, 0);
        $.externalData = reader.tables(position, 28, mindspore.schema.ExternalData);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Tensor();
        $.nodeType = reader.value(json.nodeType, 0);
        $.dataType = reader.value(json.dataType, 0);
        $.dims = reader.array(json.dims, Int32Array);
        $.format = mindspore.schema.Format[json.format];
        $.refCount = reader.value(json.refCount, 0);
        $.offset = reader.value(json.offset, 0);
        $.data = reader.array(json.data, Uint8Array);
        $.quantParams = reader.objects(json.quantParams, mindspore.schema.QuantParam);
        $.quantClusters = reader.array(json.quantClusters, Float32Array);
        $.name = reader.value(json.name, null);
        $.enableHuffmanCode = reader.value(json.enableHuffmanCode, false);
        $.weightQuantCompressType = mindspore.schema.WeightQuantCompressType[json.weightQuantCompressType];
        $.externalData = reader.objects(json.externalData, mindspore.schema.ExternalData);
        return $;
    }
};

mindspore.schema.QuantType = {
    QUANT_NONE: 0,
    AwareTraining: 1,
    WeightQuant: 2,
    PostTraining: 3,
    QUANT_WEIGHT: 4,
    QUANT_ALL: 5,
    QUANT_DYNAMIC: 6
};

mindspore.schema.Primitive = class Primitive {

    static decode(reader, position) {
        const $ = new mindspore.schema.Primitive();
        $.value = reader.union(position, 4, mindspore.schema.PrimitiveType);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.Primitive();
        $.value = mindspore.schema.PrimitiveType.decodeText(reader, json.value, json.value_type);
        return $;
    }
};

mindspore.schema.CNode = class CNode {

    static decode(reader, position) {
        const $ = new mindspore.schema.CNode();
        $.name = reader.string_(position, 4, null);
        $.nodeType = reader.int32_(position, 6, 0);
        $.primitive = reader.table(position, 8, mindspore.schema.Primitive);
        $.inputIndex = reader.array(position, 10, Uint32Array);
        $.outputIndex = reader.array(position, 12, Uint32Array);
        $.quantType = reader.int32_(position, 14, 0);
        $.deviceType = reader.int32_(position, 16, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.CNode();
        $.name = reader.value(json.name, null);
        $.nodeType = reader.value(json.nodeType, 0);
        $.primitive = reader.object(json.primitive, mindspore.schema.Primitive);
        $.inputIndex = reader.array(json.inputIndex, Uint32Array);
        $.outputIndex = reader.array(json.outputIndex, Uint32Array);
        $.quantType = mindspore.schema.QuantType[json.quantType];
        $.deviceType = reader.value(json.deviceType, -1);
        return $;
    }
};

mindspore.schema.SubGraph = class SubGraph {

    static decode(reader, position) {
        const $ = new mindspore.schema.SubGraph();
        $.name = reader.string_(position, 4, null);
        $.inputIndices = reader.array(position, 6, Uint32Array);
        $.outputIndices = reader.array(position, 8, Uint32Array);
        $.nodeIndices = reader.array(position, 10, Uint32Array);
        $.tensorIndices = reader.array(position, 12, Uint32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.SubGraph();
        $.name = reader.value(json.name, null);
        $.inputIndices = reader.array(json.inputIndices, Uint32Array);
        $.outputIndices = reader.array(json.outputIndices, Uint32Array);
        $.nodeIndices = reader.array(json.nodeIndices, Uint32Array);
        $.tensorIndices = reader.array(json.tensorIndices, Uint32Array);
        return $;
    }
};

mindspore.schema.MetaGraph = class MetaGraph {

    static identifier(reader) {
        return reader.identifier === 'MSL2';
    }

    static create(reader) {
        return mindspore.schema.MetaGraph.decode(reader, reader.root);
    }

    static createText(reader) {
        return mindspore.schema.MetaGraph.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new mindspore.schema.MetaGraph();
        $.name = reader.string_(position, 4, null);
        $.version = reader.string_(position, 6, null);
        $.fmkType = reader.int32_(position, 8, 0);
        $.inputIndex = reader.array(position, 10, Uint32Array);
        $.outputIndex = reader.array(position, 12, Uint32Array);
        $.mempoolSize = reader.uint32_(position, 14, 0);
        $.nodes = reader.tables(position, 16, mindspore.schema.CNode);
        $.allTensors = reader.tables(position, 18, mindspore.schema.Tensor);
        $.subGraph = reader.tables(position, 20, mindspore.schema.SubGraph);
        $.obfuscate = reader.bool_(position, 22, false);
        $.obfMetaData = reader.array(position, 24, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new mindspore.schema.MetaGraph();
        $.name = reader.value(json.name, null);
        $.version = reader.value(json.version, null);
        $.fmkType = reader.value(json.fmkType, 0);
        $.inputIndex = reader.array(json.inputIndex, Uint32Array);
        $.outputIndex = reader.array(json.outputIndex, Uint32Array);
        $.mempoolSize = reader.value(json.mempoolSize, 0);
        $.nodes = reader.objects(json.nodes, mindspore.schema.CNode);
        $.allTensors = reader.objects(json.allTensors, mindspore.schema.Tensor);
        $.subGraph = reader.objects(json.subGraph, mindspore.schema.SubGraph);
        $.obfuscate = reader.value(json.obfuscate, false);
        $.obfMetaData = reader.array(json.obfMetaData, Uint8Array);
        return $;
    }
};
