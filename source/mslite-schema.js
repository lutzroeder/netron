var $root = flatbuffers.get('mslite');

$root.mindspore = $root.mindspore || {};

$root.mindspore.schema = $root.mindspore.schema || {};

$root.mindspore.schema.NodeType = {
    ValueNode: 0,
    Parameter: 1,
    CNode: 2
};

$root.mindspore.schema.QuantParam = class QuantParam {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.QuantParam();
        $.scale = reader.float64_(position, 4, 0);
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
        $.scale = reader.value(json.scale, 0);
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

$root.mindspore.schema.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Tensor();
        $.nodeType = reader.int32_(position, 4, 0);
        $.dataType = reader.int32_(position, 6, 0);
        $.dims = reader.typedArray(position, 8, Int32Array);
        $.format = reader.int32_(position, 10, undefined);
        $.refCount = reader.int32_(position, 12, 0);
        $.offset = reader.int32_(position, 14, 0);
        $.data = reader.typedArray(position, 16, Uint8Array);
        $.quantParams = reader.tableArray(position, 18, $root.mindspore.schema.QuantParam.decode);
        $.quantClusters = reader.typedArray(position, 20, Float32Array);
        $.name = reader.string_(position, 22, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Tensor();
        $.nodeType = $root.mindspore.schema.NodeType[json.nodeType];
        $.dataType = reader.value(json.dataType, 0);
        $.dims = reader.typedArray(json.dims, Int32Array);
        $.format = $root.mindspore.schema.Format[json.format];
        $.refCount = reader.value(json.refCount, 0);
        $.offset = reader.value(json.offset, 0);
        $.data = reader.typedArray(json.data, Uint8Array);
        $.quantParams = reader.objectArray(json.quantParams, $root.mindspore.schema.QuantParam.decodeText);
        $.quantClusters = reader.typedArray(json.quantClusters, Float32Array);
        $.name = reader.value(json.name, null);
        return $;
    }
};

$root.mindspore.schema.PrimitiveType = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.mindspore.schema.Concat.decode(reader, position);
            case 2: return $root.mindspore.schema.SoftMax.decode(reader, position);
            case 3: return $root.mindspore.schema.Activation.decode(reader, position);
            case 4: return $root.mindspore.schema.Conv2D.decode(reader, position);
            case 5: return $root.mindspore.schema.FusedBatchNorm.decode(reader, position);
            case 6: return $root.mindspore.schema.BatchNorm.decode(reader, position);
            case 7: return $root.mindspore.schema.BiasAdd.decode(reader, position);
            case 8: return $root.mindspore.schema.Pooling.decode(reader, position);
            case 9: return $root.mindspore.schema.ROIPooling.decode(reader, position);
            case 10: return $root.mindspore.schema.DepthwiseConv2D.decode(reader, position);
            case 11: return $root.mindspore.schema.DeDepthwiseConv2D.decode(reader, position);
            case 12: return $root.mindspore.schema.Resize.decode(reader, position);
            case 13: return $root.mindspore.schema.DetectionPostProcess.decode(reader, position);
            case 14: return $root.mindspore.schema.FullConnection.decode(reader, position);
            case 15: return $root.mindspore.schema.Mean.decode(reader, position);
            case 16: return $root.mindspore.schema.DeConv2D.decode(reader, position);
            case 17: return $root.mindspore.schema.Scale.decode(reader, position);
            case 18: return $root.mindspore.schema.Reshape.decode(reader, position);
            case 19: return $root.mindspore.schema.Eltwise.decode(reader, position);
            case 20: return $root.mindspore.schema.NetOutput.decode(reader, position);
            case 21: return $root.mindspore.schema.Add.decode(reader, position);
            case 22: return $root.mindspore.schema.Sub.decode(reader, position);
            case 23: return $root.mindspore.schema.MatMul.decode(reader, position);
            case 24: return $root.mindspore.schema.StridedSlice.decode(reader, position);
            case 25: return $root.mindspore.schema.Power.decode(reader, position);
            case 26: return $root.mindspore.schema.Slice.decode(reader, position);
            case 27: return $root.mindspore.schema.Stack.decode(reader, position);
            case 28: return $root.mindspore.schema.Mul.decode(reader, position);
            case 29: return $root.mindspore.schema.RealDiv.decode(reader, position);
            case 30: return $root.mindspore.schema.Pad.decode(reader, position);
            case 31: return $root.mindspore.schema.Maximum.decode(reader, position);
            case 32: return $root.mindspore.schema.Minimum.decode(reader, position);
            case 33: return $root.mindspore.schema.PReLU.decode(reader, position);
            case 34: return $root.mindspore.schema.LeakyReLU.decode(reader, position);
            case 35: return $root.mindspore.schema.ArgMax.decode(reader, position);
            case 36: return $root.mindspore.schema.ArgMin.decode(reader, position);
            case 37: return $root.mindspore.schema.Exp.decode(reader, position);
            case 38: return $root.mindspore.schema.Crop.decode(reader, position);
            case 39: return $root.mindspore.schema.Range.decode(reader, position);
            case 40: return $root.mindspore.schema.Rsqrt.decode(reader, position);
            case 41: return $root.mindspore.schema.ExpandDims.decode(reader, position);
            case 42: return $root.mindspore.schema.Tile.decode(reader, position);
            case 43: return $root.mindspore.schema.Cast.decode(reader, position);
            case 44: return $root.mindspore.schema.Shape.decode(reader, position);
            case 45: return $root.mindspore.schema.Nchw2Nhwc.decode(reader, position);
            case 46: return $root.mindspore.schema.Nhwc2Nchw.decode(reader, position);
            case 47: return $root.mindspore.schema.QuantDTypeCast.decode(reader, position);
            case 48: return $root.mindspore.schema.Split.decode(reader, position);
            case 49: return $root.mindspore.schema.Permute.decode(reader, position);
            case 50: return $root.mindspore.schema.FakeQuantWithMinMaxVars.decode(reader, position);
            case 51: return $root.mindspore.schema.Equal.decode(reader, position);
            case 52: return $root.mindspore.schema.Less.decode(reader, position);
            case 53: return $root.mindspore.schema.Greater.decode(reader, position);
            case 54: return $root.mindspore.schema.NotEqual.decode(reader, position);
            case 55: return $root.mindspore.schema.LessEqual.decode(reader, position);
            case 56: return $root.mindspore.schema.GreaterEqual.decode(reader, position);
            case 57: return $root.mindspore.schema.Min.decode(reader, position);
            case 58: return $root.mindspore.schema.Floor.decode(reader, position);
            case 59: return $root.mindspore.schema.Abs.decode(reader, position);
            case 60: return $root.mindspore.schema.Neg.decode(reader, position);
            case 61: return $root.mindspore.schema.Cos.decode(reader, position);
            case 62: return $root.mindspore.schema.Sin.decode(reader, position);
            case 63: return $root.mindspore.schema.Sqrt.decode(reader, position);
            case 64: return $root.mindspore.schema.Square.decode(reader, position);
            case 65: return $root.mindspore.schema.Constant.decode(reader, position);
            case 66: return $root.mindspore.schema.Log.decode(reader, position);
            case 67: return $root.mindspore.schema.Tan.decode(reader, position);
            case 68: return $root.mindspore.schema.Atan.decode(reader, position);
            case 69: return $root.mindspore.schema.Asin.decode(reader, position);
            case 70: return $root.mindspore.schema.Clip.decode(reader, position);
            case 71: return $root.mindspore.schema.Transpose.decode(reader, position);
            case 72: return $root.mindspore.schema.Squeeze.decode(reader, position);
            case 73: return $root.mindspore.schema.Unsqueeze.decode(reader, position);
            case 74: return $root.mindspore.schema.Upsample.decode(reader, position);
            case 75: return $root.mindspore.schema.Dropout.decode(reader, position);
            case 76: return $root.mindspore.schema.Broadcast.decode(reader, position);
            case 77: return $root.mindspore.schema.BroadcastTo.decode(reader, position);
            case 78: return $root.mindspore.schema.Lrn.decode(reader, position);
            case 79: return $root.mindspore.schema.ZerosLike.decode(reader, position);
            case 80: return $root.mindspore.schema.TopK.decode(reader, position);
            case 81: return $root.mindspore.schema.SpaceToDepth.decode(reader, position);
            case 82: return $root.mindspore.schema.SpaceToBatch.decode(reader, position);
            case 83: return $root.mindspore.schema.SparseToDense.decode(reader, position);
            case 84: return $root.mindspore.schema.ReverseSequence.decode(reader, position);
            case 85: return $root.mindspore.schema.Rank.decode(reader, position);
            case 86: return $root.mindspore.schema.Gather.decode(reader, position);
            case 87: return $root.mindspore.schema.GatherNd.decode(reader, position);
            case 88: return $root.mindspore.schema.Fill.decode(reader, position);
            case 89: return $root.mindspore.schema.Elu.decode(reader, position);
            case 90: return $root.mindspore.schema.DepthToSpace.decode(reader, position);
            case 91: return $root.mindspore.schema.BatchToSpace.decode(reader, position);
            case 92: return $root.mindspore.schema.AddN.decode(reader, position);
            case 93: return $root.mindspore.schema.Ceil.decode(reader, position);
            case 94: return $root.mindspore.schema.EmbeddingLookup.decode(reader, position);
            case 95: return $root.mindspore.schema.EmbeddingLookupSparse.decode(reader, position);
            case 96: return $root.mindspore.schema.FloorDiv.decode(reader, position);
            case 97: return $root.mindspore.schema.FloorMod.decode(reader, position);
            case 98: return $root.mindspore.schema.L2Norm.decode(reader, position);
            case 99: return $root.mindspore.schema.LocalResponseNormalization.decode(reader, position);
            case 100: return $root.mindspore.schema.MatrixDiag.decode(reader, position);
            case 101: return $root.mindspore.schema.Reduce.decode(reader, position);
            case 102: return $root.mindspore.schema.Reverse.decode(reader, position);
            case 103: return $root.mindspore.schema.Round.decode(reader, position);
            case 104: return $root.mindspore.schema.Select.decode(reader, position);
            case 105: return $root.mindspore.schema.Scatter.decode(reader, position);
            case 106: return $root.mindspore.schema.ScatterND.decode(reader, position);
            case 107: return $root.mindspore.schema.ConstantOfShape.decode(reader, position);
            case 108: return $root.mindspore.schema.Unique.decode(reader, position);
            case 109: return $root.mindspore.schema.Unstack.decode(reader, position);
            case 110: return $root.mindspore.schema.LogicalAnd.decode(reader, position);
            case 111: return $root.mindspore.schema.LogicalOr.decode(reader, position);
            case 112: return $root.mindspore.schema.LogicalXor.decode(reader, position);
            case 113: return $root.mindspore.schema.LogicalNot.decode(reader, position);
            case 114: return $root.mindspore.schema.OnnxInt8Quantize.decode(reader, position);
            case 115: return $root.mindspore.schema.OnnxInt8Dequantize.decode(reader, position);
            case 116: return $root.mindspore.schema.FakeQuantWithMinMax.decode(reader, position);
            case 117: return $root.mindspore.schema.FakeQuantWithMinMaxPerChannel.decode(reader, position);
            case 118: return $root.mindspore.schema.BatchNormFold.decode(reader, position);
            case 119: return $root.mindspore.schema.MulFold.decode(reader, position);
            case 120: return $root.mindspore.schema.AddFold.decode(reader, position);
            case 121: return $root.mindspore.schema.SquaredDifference.decode(reader, position);
            case 122: return $root.mindspore.schema.Flatten.decode(reader, position);
            case 123: return $root.mindspore.schema.FlattenGrad.decode(reader, position);
            case 124: return $root.mindspore.schema.TupleGetItem.decode(reader, position);
            case 125: return $root.mindspore.schema.Div.decode(reader, position);
            case 126: return $root.mindspore.schema.Where.decode(reader, position);
            case 127: return $root.mindspore.schema.OneHot.decode(reader, position);
            case 128: return $root.mindspore.schema.Lstm.decode(reader, position);
            case 129: return $root.mindspore.schema.Conv2DGradFilter.decode(reader, position);
            case 130: return $root.mindspore.schema.Conv2DGradInput.decode(reader, position);
            case 131: return $root.mindspore.schema.PoolingGrad.decode(reader, position);
            case 132: return $root.mindspore.schema.BNGrad.decode(reader, position);
            case 133: return $root.mindspore.schema.Assign.decode(reader, position);
            case 134: return $root.mindspore.schema.ApplyMomentum.decode(reader, position);
            case 135: return $root.mindspore.schema.BiasGrad.decode(reader, position);
            case 136: return $root.mindspore.schema.SoftmaxCrossEntropy.decode(reader, position);
            case 137: return $root.mindspore.schema.AddGrad.decode(reader, position);
            case 138: return $root.mindspore.schema.SubGrad.decode(reader, position);
            case 139: return $root.mindspore.schema.MulGrad.decode(reader, position);
            case 140: return $root.mindspore.schema.DivGrad.decode(reader, position);
            case 141: return $root.mindspore.schema.PowerGrad.decode(reader, position);
            case 142: return $root.mindspore.schema.ActivationGrad.decode(reader, position);
            case 143: return $root.mindspore.schema.PriorBox.decode(reader, position);
            case 144: return $root.mindspore.schema.SpaceToBatchND.decode(reader, position);
            case 145: return $root.mindspore.schema.Depend.decode(reader, position);
            case 146: return $root.mindspore.schema.Return.decode(reader, position);
            case 147: return $root.mindspore.schema.MakeTuple.decode(reader, position);
            case 148: return $root.mindspore.schema.ToFormat.decode(reader, position);
            case 149: return $root.mindspore.schema.Proposal.decode(reader, position);
            case 150: return $root.mindspore.schema.Custom.decode(reader, position);
            case 151: return $root.mindspore.schema.BlackBox.decode(reader, position);
            case 152: return $root.mindspore.schema.NegGrad.decode(reader, position);
            case 153: return $root.mindspore.schema.LogGrad.decode(reader, position);
            case 154: return $root.mindspore.schema.BatchToSpaceND.decode(reader, position);
            case 155: return $root.mindspore.schema.LshProjection.decode(reader, position);
            case 156: return $root.mindspore.schema.HashtableLookup.decode(reader, position);
            case 157: return $root.mindspore.schema.SkipGram.decode(reader, position);
            case 158: return $root.mindspore.schema.DeConv2DGradFilter.decode(reader, position);
            case 159: return $root.mindspore.schema.CustomPredict.decode(reader, position);
            case 160: return $root.mindspore.schema.CustomNormalize.decode(reader, position);
            case 161: return $root.mindspore.schema.CustomExtractFeatures.decode(reader, position);
            case 162: return $root.mindspore.schema.AudioSpectrogram.decode(reader, position);
            case 163: return $root.mindspore.schema.Mfcc.decode(reader, position);
            case 164: return $root.mindspore.schema.Rfft.decode(reader, position);
            case 165: return $root.mindspore.schema.FftReal.decode(reader, position);
            case 166: return $root.mindspore.schema.FftImag.decode(reader, position);
            case 167: return $root.mindspore.schema.Sgd.decode(reader, position);
            case 168: return $root.mindspore.schema.Adam.decode(reader, position);
            case 169: return $root.mindspore.schema.GroupConv2DGradInput.decode(reader, position);
            case 170: return $root.mindspore.schema.Loop.decode(reader, position);
            case 171: return $root.mindspore.schema.NonMaxSuppression.decode(reader, position);
            case 172: return $root.mindspore.schema.InstanceNorm.decode(reader, position);
            case 173: return $root.mindspore.schema.Identity.decode(reader, position);
            case 174: return $root.mindspore.schema.LayerNorm.decode(reader, position);
            case 175: return $root.mindspore.schema.While.decode(reader, position);
            case 176: return $root.mindspore.schema.ControlDepend.decode(reader, position);
            case 177: return $root.mindspore.schema.UnsortedSegmentSum.decode(reader, position);
            case 178: return $root.mindspore.schema.AssignAdd.decode(reader, position);
            case 179: return $root.mindspore.schema.OnesLike.decode(reader, position);
            case 180: return $root.mindspore.schema.BinaryCrossEntropyGrad.decode(reader, position);
            case 181: return $root.mindspore.schema.BinaryCrossEntropy.decode(reader, position);
            case 182: return $root.mindspore.schema.LpNormalization.decode(reader, position);
            case 183: return $root.mindspore.schema.DropoutGrad.decode(reader, position);
            case 184: return $root.mindspore.schema.MaximumGrad.decode(reader, position);
            case 185: return $root.mindspore.schema.MinimumGrad.decode(reader, position);
            case 186: return $root.mindspore.schema.Switch.decode(reader, position);
            case 187: return $root.mindspore.schema.Partial.decode(reader, position);
            case 188: return $root.mindspore.schema.TensorListFromTensor.decode(reader, position);
            case 189: return $root.mindspore.schema.TensorListStack.decode(reader, position);
            case 190: return $root.mindspore.schema.TensorListGetItem.decode(reader, position);
            case 191: return $root.mindspore.schema.TensorListSetItem.decode(reader, position);
            case 192: return $root.mindspore.schema.TensorListReserve.decode(reader, position);
            case 193: return $root.mindspore.schema.All.decode(reader, position);
            case 194: return $root.mindspore.schema.Assert.decode(reader, position);
            case 195: return $root.mindspore.schema.Adder.decode(reader, position);
            case 196: return $root.mindspore.schema.SparseSoftmaxCrossEntropy.decode(reader, position);
            case 197: return $root.mindspore.schema.SmoothL1Loss.decode(reader, position);
            case 198: return $root.mindspore.schema.SmoothL1LossGrad.decode(reader, position);
            case 199: return $root.mindspore.schema.SigmoidCrossEntropyWithLogits.decode(reader, position);
            case 200: return $root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad.decode(reader, position);
            case 201: return $root.mindspore.schema.Reciprocal.decode(reader, position);
            case 202: return $root.mindspore.schema.Merge.decode(reader, position);
            case 203: return $root.mindspore.schema.Mod.decode(reader, position);
            case 204: return $root.mindspore.schema.If.decode(reader, position);
            case 205: return $root.mindspore.schema.GeLU.decode(reader, position);
            case 206: return $root.mindspore.schema.Gru.decode(reader, position);
            case 207: return $root.mindspore.schema.NonZero.decode(reader, position);
            case 208: return $root.mindspore.schema.InvertPermutation.decode(reader, position);
            case 209: return $root.mindspore.schema.Size.decode(reader, position);
            case 210: return $root.mindspore.schema.RandomStandardNormal.decode(reader, position);
            case 211: return $root.mindspore.schema.CropAndResize.decode(reader, position);
        }
        return undefined;
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'Concat': return $root.mindspore.schema.Concat.decodeText(reader, json);
            case 'SoftMax': return $root.mindspore.schema.SoftMax.decodeText(reader, json);
            case 'Activation': return $root.mindspore.schema.Activation.decodeText(reader, json);
            case 'Conv2D': return $root.mindspore.schema.Conv2D.decodeText(reader, json);
            case 'FusedBatchNorm': return $root.mindspore.schema.FusedBatchNorm.decodeText(reader, json);
            case 'BatchNorm': return $root.mindspore.schema.BatchNorm.decodeText(reader, json);
            case 'BiasAdd': return $root.mindspore.schema.BiasAdd.decodeText(reader, json);
            case 'Pooling': return $root.mindspore.schema.Pooling.decodeText(reader, json);
            case 'ROIPooling': return $root.mindspore.schema.ROIPooling.decodeText(reader, json);
            case 'DepthwiseConv2D': return $root.mindspore.schema.DepthwiseConv2D.decodeText(reader, json);
            case 'DeDepthwiseConv2D': return $root.mindspore.schema.DeDepthwiseConv2D.decodeText(reader, json);
            case 'Resize': return $root.mindspore.schema.Resize.decodeText(reader, json);
            case 'DetectionPostProcess': return $root.mindspore.schema.DetectionPostProcess.decodeText(reader, json);
            case 'FullConnection': return $root.mindspore.schema.FullConnection.decodeText(reader, json);
            case 'Mean': return $root.mindspore.schema.Mean.decodeText(reader, json);
            case 'DeConv2D': return $root.mindspore.schema.DeConv2D.decodeText(reader, json);
            case 'Scale': return $root.mindspore.schema.Scale.decodeText(reader, json);
            case 'Reshape': return $root.mindspore.schema.Reshape.decodeText(reader, json);
            case 'Eltwise': return $root.mindspore.schema.Eltwise.decodeText(reader, json);
            case 'NetOutput': return $root.mindspore.schema.NetOutput.decodeText(reader, json);
            case 'Add': return $root.mindspore.schema.Add.decodeText(reader, json);
            case 'Sub': return $root.mindspore.schema.Sub.decodeText(reader, json);
            case 'MatMul': return $root.mindspore.schema.MatMul.decodeText(reader, json);
            case 'StridedSlice': return $root.mindspore.schema.StridedSlice.decodeText(reader, json);
            case 'Power': return $root.mindspore.schema.Power.decodeText(reader, json);
            case 'Slice': return $root.mindspore.schema.Slice.decodeText(reader, json);
            case 'Stack': return $root.mindspore.schema.Stack.decodeText(reader, json);
            case 'Mul': return $root.mindspore.schema.Mul.decodeText(reader, json);
            case 'RealDiv': return $root.mindspore.schema.RealDiv.decodeText(reader, json);
            case 'Pad': return $root.mindspore.schema.Pad.decodeText(reader, json);
            case 'Maximum': return $root.mindspore.schema.Maximum.decodeText(reader, json);
            case 'Minimum': return $root.mindspore.schema.Minimum.decodeText(reader, json);
            case 'PReLU': return $root.mindspore.schema.PReLU.decodeText(reader, json);
            case 'LeakyReLU': return $root.mindspore.schema.LeakyReLU.decodeText(reader, json);
            case 'ArgMax': return $root.mindspore.schema.ArgMax.decodeText(reader, json);
            case 'ArgMin': return $root.mindspore.schema.ArgMin.decodeText(reader, json);
            case 'Exp': return $root.mindspore.schema.Exp.decodeText(reader, json);
            case 'Crop': return $root.mindspore.schema.Crop.decodeText(reader, json);
            case 'Range': return $root.mindspore.schema.Range.decodeText(reader, json);
            case 'Rsqrt': return $root.mindspore.schema.Rsqrt.decodeText(reader, json);
            case 'ExpandDims': return $root.mindspore.schema.ExpandDims.decodeText(reader, json);
            case 'Tile': return $root.mindspore.schema.Tile.decodeText(reader, json);
            case 'Cast': return $root.mindspore.schema.Cast.decodeText(reader, json);
            case 'Shape': return $root.mindspore.schema.Shape.decodeText(reader, json);
            case 'Nchw2Nhwc': return $root.mindspore.schema.Nchw2Nhwc.decodeText(reader, json);
            case 'Nhwc2Nchw': return $root.mindspore.schema.Nhwc2Nchw.decodeText(reader, json);
            case 'QuantDTypeCast': return $root.mindspore.schema.QuantDTypeCast.decodeText(reader, json);
            case 'Split': return $root.mindspore.schema.Split.decodeText(reader, json);
            case 'Permute': return $root.mindspore.schema.Permute.decodeText(reader, json);
            case 'FakeQuantWithMinMaxVars': return $root.mindspore.schema.FakeQuantWithMinMaxVars.decodeText(reader, json);
            case 'Equal': return $root.mindspore.schema.Equal.decodeText(reader, json);
            case 'Less': return $root.mindspore.schema.Less.decodeText(reader, json);
            case 'Greater': return $root.mindspore.schema.Greater.decodeText(reader, json);
            case 'NotEqual': return $root.mindspore.schema.NotEqual.decodeText(reader, json);
            case 'LessEqual': return $root.mindspore.schema.LessEqual.decodeText(reader, json);
            case 'GreaterEqual': return $root.mindspore.schema.GreaterEqual.decodeText(reader, json);
            case 'Min': return $root.mindspore.schema.Min.decodeText(reader, json);
            case 'Floor': return $root.mindspore.schema.Floor.decodeText(reader, json);
            case 'Abs': return $root.mindspore.schema.Abs.decodeText(reader, json);
            case 'Neg': return $root.mindspore.schema.Neg.decodeText(reader, json);
            case 'Cos': return $root.mindspore.schema.Cos.decodeText(reader, json);
            case 'Sin': return $root.mindspore.schema.Sin.decodeText(reader, json);
            case 'Sqrt': return $root.mindspore.schema.Sqrt.decodeText(reader, json);
            case 'Square': return $root.mindspore.schema.Square.decodeText(reader, json);
            case 'Constant': return $root.mindspore.schema.Constant.decodeText(reader, json);
            case 'Log': return $root.mindspore.schema.Log.decodeText(reader, json);
            case 'Tan': return $root.mindspore.schema.Tan.decodeText(reader, json);
            case 'Atan': return $root.mindspore.schema.Atan.decodeText(reader, json);
            case 'Asin': return $root.mindspore.schema.Asin.decodeText(reader, json);
            case 'Clip': return $root.mindspore.schema.Clip.decodeText(reader, json);
            case 'Transpose': return $root.mindspore.schema.Transpose.decodeText(reader, json);
            case 'Squeeze': return $root.mindspore.schema.Squeeze.decodeText(reader, json);
            case 'Unsqueeze': return $root.mindspore.schema.Unsqueeze.decodeText(reader, json);
            case 'Upsample': return $root.mindspore.schema.Upsample.decodeText(reader, json);
            case 'Dropout': return $root.mindspore.schema.Dropout.decodeText(reader, json);
            case 'Broadcast': return $root.mindspore.schema.Broadcast.decodeText(reader, json);
            case 'BroadcastTo': return $root.mindspore.schema.BroadcastTo.decodeText(reader, json);
            case 'Lrn': return $root.mindspore.schema.Lrn.decodeText(reader, json);
            case 'ZerosLike': return $root.mindspore.schema.ZerosLike.decodeText(reader, json);
            case 'TopK': return $root.mindspore.schema.TopK.decodeText(reader, json);
            case 'SpaceToDepth': return $root.mindspore.schema.SpaceToDepth.decodeText(reader, json);
            case 'SpaceToBatch': return $root.mindspore.schema.SpaceToBatch.decodeText(reader, json);
            case 'SparseToDense': return $root.mindspore.schema.SparseToDense.decodeText(reader, json);
            case 'ReverseSequence': return $root.mindspore.schema.ReverseSequence.decodeText(reader, json);
            case 'Rank': return $root.mindspore.schema.Rank.decodeText(reader, json);
            case 'Gather': return $root.mindspore.schema.Gather.decodeText(reader, json);
            case 'GatherNd': return $root.mindspore.schema.GatherNd.decodeText(reader, json);
            case 'Fill': return $root.mindspore.schema.Fill.decodeText(reader, json);
            case 'Elu': return $root.mindspore.schema.Elu.decodeText(reader, json);
            case 'DepthToSpace': return $root.mindspore.schema.DepthToSpace.decodeText(reader, json);
            case 'BatchToSpace': return $root.mindspore.schema.BatchToSpace.decodeText(reader, json);
            case 'AddN': return $root.mindspore.schema.AddN.decodeText(reader, json);
            case 'Ceil': return $root.mindspore.schema.Ceil.decodeText(reader, json);
            case 'EmbeddingLookup': return $root.mindspore.schema.EmbeddingLookup.decodeText(reader, json);
            case 'EmbeddingLookupSparse': return $root.mindspore.schema.EmbeddingLookupSparse.decodeText(reader, json);
            case 'FloorDiv': return $root.mindspore.schema.FloorDiv.decodeText(reader, json);
            case 'FloorMod': return $root.mindspore.schema.FloorMod.decodeText(reader, json);
            case 'L2Norm': return $root.mindspore.schema.L2Norm.decodeText(reader, json);
            case 'LocalResponseNormalization': return $root.mindspore.schema.LocalResponseNormalization.decodeText(reader, json);
            case 'MatrixDiag': return $root.mindspore.schema.MatrixDiag.decodeText(reader, json);
            case 'Reduce': return $root.mindspore.schema.Reduce.decodeText(reader, json);
            case 'Reverse': return $root.mindspore.schema.Reverse.decodeText(reader, json);
            case 'Round': return $root.mindspore.schema.Round.decodeText(reader, json);
            case 'Select': return $root.mindspore.schema.Select.decodeText(reader, json);
            case 'Scatter': return $root.mindspore.schema.Scatter.decodeText(reader, json);
            case 'ScatterND': return $root.mindspore.schema.ScatterND.decodeText(reader, json);
            case 'ConstantOfShape': return $root.mindspore.schema.ConstantOfShape.decodeText(reader, json);
            case 'Unique': return $root.mindspore.schema.Unique.decodeText(reader, json);
            case 'Unstack': return $root.mindspore.schema.Unstack.decodeText(reader, json);
            case 'LogicalAnd': return $root.mindspore.schema.LogicalAnd.decodeText(reader, json);
            case 'LogicalOr': return $root.mindspore.schema.LogicalOr.decodeText(reader, json);
            case 'LogicalXor': return $root.mindspore.schema.LogicalXor.decodeText(reader, json);
            case 'LogicalNot': return $root.mindspore.schema.LogicalNot.decodeText(reader, json);
            case 'OnnxInt8Quantize': return $root.mindspore.schema.OnnxInt8Quantize.decodeText(reader, json);
            case 'OnnxInt8Dequantize': return $root.mindspore.schema.OnnxInt8Dequantize.decodeText(reader, json);
            case 'FakeQuantWithMinMax': return $root.mindspore.schema.FakeQuantWithMinMax.decodeText(reader, json);
            case 'FakeQuantWithMinMaxPerChannel': return $root.mindspore.schema.FakeQuantWithMinMaxPerChannel.decodeText(reader, json);
            case 'BatchNormFold': return $root.mindspore.schema.BatchNormFold.decodeText(reader, json);
            case 'MulFold': return $root.mindspore.schema.MulFold.decodeText(reader, json);
            case 'AddFold': return $root.mindspore.schema.AddFold.decodeText(reader, json);
            case 'SquaredDifference': return $root.mindspore.schema.SquaredDifference.decodeText(reader, json);
            case 'Flatten': return $root.mindspore.schema.Flatten.decodeText(reader, json);
            case 'FlattenGrad': return $root.mindspore.schema.FlattenGrad.decodeText(reader, json);
            case 'TupleGetItem': return $root.mindspore.schema.TupleGetItem.decodeText(reader, json);
            case 'Div': return $root.mindspore.schema.Div.decodeText(reader, json);
            case 'Where': return $root.mindspore.schema.Where.decodeText(reader, json);
            case 'OneHot': return $root.mindspore.schema.OneHot.decodeText(reader, json);
            case 'Lstm': return $root.mindspore.schema.Lstm.decodeText(reader, json);
            case 'Conv2DGradFilter': return $root.mindspore.schema.Conv2DGradFilter.decodeText(reader, json);
            case 'Conv2DGradInput': return $root.mindspore.schema.Conv2DGradInput.decodeText(reader, json);
            case 'PoolingGrad': return $root.mindspore.schema.PoolingGrad.decodeText(reader, json);
            case 'BNGrad': return $root.mindspore.schema.BNGrad.decodeText(reader, json);
            case 'Assign': return $root.mindspore.schema.Assign.decodeText(reader, json);
            case 'ApplyMomentum': return $root.mindspore.schema.ApplyMomentum.decodeText(reader, json);
            case 'BiasGrad': return $root.mindspore.schema.BiasGrad.decodeText(reader, json);
            case 'SoftmaxCrossEntropy': return $root.mindspore.schema.SoftmaxCrossEntropy.decodeText(reader, json);
            case 'AddGrad': return $root.mindspore.schema.AddGrad.decodeText(reader, json);
            case 'SubGrad': return $root.mindspore.schema.SubGrad.decodeText(reader, json);
            case 'MulGrad': return $root.mindspore.schema.MulGrad.decodeText(reader, json);
            case 'DivGrad': return $root.mindspore.schema.DivGrad.decodeText(reader, json);
            case 'PowerGrad': return $root.mindspore.schema.PowerGrad.decodeText(reader, json);
            case 'ActivationGrad': return $root.mindspore.schema.ActivationGrad.decodeText(reader, json);
            case 'PriorBox': return $root.mindspore.schema.PriorBox.decodeText(reader, json);
            case 'SpaceToBatchND': return $root.mindspore.schema.SpaceToBatchND.decodeText(reader, json);
            case 'Depend': return $root.mindspore.schema.Depend.decodeText(reader, json);
            case 'Return': return $root.mindspore.schema.Return.decodeText(reader, json);
            case 'MakeTuple': return $root.mindspore.schema.MakeTuple.decodeText(reader, json);
            case 'ToFormat': return $root.mindspore.schema.ToFormat.decodeText(reader, json);
            case 'Proposal': return $root.mindspore.schema.Proposal.decodeText(reader, json);
            case 'Custom': return $root.mindspore.schema.Custom.decodeText(reader, json);
            case 'BlackBox': return $root.mindspore.schema.BlackBox.decodeText(reader, json);
            case 'NegGrad': return $root.mindspore.schema.NegGrad.decodeText(reader, json);
            case 'LogGrad': return $root.mindspore.schema.LogGrad.decodeText(reader, json);
            case 'BatchToSpaceND': return $root.mindspore.schema.BatchToSpaceND.decodeText(reader, json);
            case 'LshProjection': return $root.mindspore.schema.LshProjection.decodeText(reader, json);
            case 'HashtableLookup': return $root.mindspore.schema.HashtableLookup.decodeText(reader, json);
            case 'SkipGram': return $root.mindspore.schema.SkipGram.decodeText(reader, json);
            case 'DeConv2DGradFilter': return $root.mindspore.schema.DeConv2DGradFilter.decodeText(reader, json);
            case 'CustomPredict': return $root.mindspore.schema.CustomPredict.decodeText(reader, json);
            case 'CustomNormalize': return $root.mindspore.schema.CustomNormalize.decodeText(reader, json);
            case 'CustomExtractFeatures': return $root.mindspore.schema.CustomExtractFeatures.decodeText(reader, json);
            case 'AudioSpectrogram': return $root.mindspore.schema.AudioSpectrogram.decodeText(reader, json);
            case 'Mfcc': return $root.mindspore.schema.Mfcc.decodeText(reader, json);
            case 'Rfft': return $root.mindspore.schema.Rfft.decodeText(reader, json);
            case 'FftReal': return $root.mindspore.schema.FftReal.decodeText(reader, json);
            case 'FftImag': return $root.mindspore.schema.FftImag.decodeText(reader, json);
            case 'Sgd': return $root.mindspore.schema.Sgd.decodeText(reader, json);
            case 'Adam': return $root.mindspore.schema.Adam.decodeText(reader, json);
            case 'GroupConv2DGradInput': return $root.mindspore.schema.GroupConv2DGradInput.decodeText(reader, json);
            case 'Loop': return $root.mindspore.schema.Loop.decodeText(reader, json);
            case 'NonMaxSuppression': return $root.mindspore.schema.NonMaxSuppression.decodeText(reader, json);
            case 'InstanceNorm': return $root.mindspore.schema.InstanceNorm.decodeText(reader, json);
            case 'Identity': return $root.mindspore.schema.Identity.decodeText(reader, json);
            case 'LayerNorm': return $root.mindspore.schema.LayerNorm.decodeText(reader, json);
            case 'While': return $root.mindspore.schema.While.decodeText(reader, json);
            case 'ControlDepend': return $root.mindspore.schema.ControlDepend.decodeText(reader, json);
            case 'UnsortedSegmentSum': return $root.mindspore.schema.UnsortedSegmentSum.decodeText(reader, json);
            case 'AssignAdd': return $root.mindspore.schema.AssignAdd.decodeText(reader, json);
            case 'OnesLike': return $root.mindspore.schema.OnesLike.decodeText(reader, json);
            case 'BinaryCrossEntropyGrad': return $root.mindspore.schema.BinaryCrossEntropyGrad.decodeText(reader, json);
            case 'BinaryCrossEntropy': return $root.mindspore.schema.BinaryCrossEntropy.decodeText(reader, json);
            case 'LpNormalization': return $root.mindspore.schema.LpNormalization.decodeText(reader, json);
            case 'DropoutGrad': return $root.mindspore.schema.DropoutGrad.decodeText(reader, json);
            case 'MaximumGrad': return $root.mindspore.schema.MaximumGrad.decodeText(reader, json);
            case 'MinimumGrad': return $root.mindspore.schema.MinimumGrad.decodeText(reader, json);
            case 'Switch': return $root.mindspore.schema.Switch.decodeText(reader, json);
            case 'Partial': return $root.mindspore.schema.Partial.decodeText(reader, json);
            case 'TensorListFromTensor': return $root.mindspore.schema.TensorListFromTensor.decodeText(reader, json);
            case 'TensorListStack': return $root.mindspore.schema.TensorListStack.decodeText(reader, json);
            case 'TensorListGetItem': return $root.mindspore.schema.TensorListGetItem.decodeText(reader, json);
            case 'TensorListSetItem': return $root.mindspore.schema.TensorListSetItem.decodeText(reader, json);
            case 'TensorListReserve': return $root.mindspore.schema.TensorListReserve.decodeText(reader, json);
            case 'All': return $root.mindspore.schema.All.decodeText(reader, json);
            case 'Assert': return $root.mindspore.schema.Assert.decodeText(reader, json);
            case 'Adder': return $root.mindspore.schema.Adder.decodeText(reader, json);
            case 'SparseSoftmaxCrossEntropy': return $root.mindspore.schema.SparseSoftmaxCrossEntropy.decodeText(reader, json);
            case 'SmoothL1Loss': return $root.mindspore.schema.SmoothL1Loss.decodeText(reader, json);
            case 'SmoothL1LossGrad': return $root.mindspore.schema.SmoothL1LossGrad.decodeText(reader, json);
            case 'SigmoidCrossEntropyWithLogits': return $root.mindspore.schema.SigmoidCrossEntropyWithLogits.decodeText(reader, json);
            case 'SigmoidCrossEntropyWithLogitsGrad': return $root.mindspore.schema.SigmoidCrossEntropyWithLogitsGrad.decodeText(reader, json);
            case 'Reciprocal': return $root.mindspore.schema.Reciprocal.decodeText(reader, json);
            case 'Merge': return $root.mindspore.schema.Merge.decodeText(reader, json);
            case 'Mod': return $root.mindspore.schema.Mod.decodeText(reader, json);
            case 'If': return $root.mindspore.schema.If.decodeText(reader, json);
            case 'GeLU': return $root.mindspore.schema.GeLU.decodeText(reader, json);
            case 'Gru': return $root.mindspore.schema.Gru.decodeText(reader, json);
            case 'NonZero': return $root.mindspore.schema.NonZero.decodeText(reader, json);
            case 'InvertPermutation': return $root.mindspore.schema.InvertPermutation.decodeText(reader, json);
            case 'Size': return $root.mindspore.schema.Size.decodeText(reader, json);
            case 'RandomStandardNormal': return $root.mindspore.schema.RandomStandardNormal.decodeText(reader, json);
            case 'CropAndResize': return $root.mindspore.schema.CropAndResize.decodeText(reader, json);
        }
        return undefined;
    }
};

$root.mindspore.schema.QuantType = {
    QUANT_NONE: 0,
    AwareTraining: 1,
    WeightQuant: 2,
    PostTraining: 3
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
        $.nodeType = reader.int32_(position, 6, 2);
        $.primitive = reader.table(position, 8, $root.mindspore.schema.Primitive.decode);
        $.inputIndex = reader.typedArray(position, 10, Uint32Array);
        $.outputIndex = reader.typedArray(position, 12, Uint32Array);
        $.quantType = reader.int32_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.CNode();
        $.name = reader.value(json.name, null);
        $.nodeType = $root.mindspore.schema.NodeType[json.nodeType];
        $.primitive = reader.object(json.primitive, $root.mindspore.schema.Primitive.decodeText);
        $.inputIndex = reader.typedArray(json.inputIndex, Uint32Array);
        $.outputIndex = reader.typedArray(json.outputIndex, Uint32Array);
        $.quantType = $root.mindspore.schema.QuantType[json.quantType];
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
        return reader.identifier('MSL1');
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
        return $;
    }
};

$root.mindspore.schema.ResizeMethod = {
    UNKNOWN: -1,
    LINEAR: 0,
    NEAREST: 1,
    CUBIC: 2
};

$root.mindspore.schema.CoordinateTransformMode = {
    COMMON: 0,
    HALF_PIXEL: 1,
    PYTORCH_HALF_PIXEL: 2,
    TF_HALF_PIXEL: 3,
    TF_CROP_AND_RESIZE: 4,
    ALIGN_CORNERS: 5,
    ASYMMETRIC: 6,
    ALIGN_CORNERS_WITH_HALF_PIEXL: 7
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
    NC4HW4: 100,
    NUM_OF_FORMAT: 101
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
    UNKNOWN: 19
};

$root.mindspore.schema.ActivationGradType = {
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
    UNKNOWN: 16
};

$root.mindspore.schema.ReduceType = {
    REDUCE_MAX: 0,
    REDUCE_MEAN: 1,
    REDUCE_ALL: 2,
    REDUCE_ANY: 3,
    REDUCE_LOG_SUM_EXP: 4,
    REDUCE_PROD: 5,
    REDUCE_SUM: 6,
    UNKNOWN: 7
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
    NOTSET: 0,
    SAME_UPPER: 1,
    VALID: 2,
    CAFFE: 4,
    SAME_LOWER: 5
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

$root.mindspore.schema.Pad = class Pad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Pad();
        $.paddings = reader.typedArray(position, 4, Int32Array);
        $.paddingMode = reader.int8_(position, 6, 0);
        $.constantValue = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Pad();
        $.paddings = reader.typedArray(json.paddings, Int32Array);
        $.paddingMode = $root.mindspore.schema.PaddingMode[json.paddingMode];
        $.constantValue = reader.value(json.constantValue, 0);
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

$root.mindspore.schema.Flatten = class Flatten {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Flatten();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Flatten();
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

$root.mindspore.schema.Concat = class Concat {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Concat();
        $.axis = reader.int32_(position, 4, 0);
        $.n = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Concat();
        $.axis = reader.value(json.axis, 0);
        $.n = reader.value(json.n, 0);
        return $;
    }
};

$root.mindspore.schema.SoftMax = class SoftMax {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SoftMax();
        $.axis = reader.int32_(position, 4, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SoftMax();
        $.axis = reader.value(json.axis, -1);
        return $;
    }
};

$root.mindspore.schema.Activation = class Activation {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Activation();
        $.type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0.2);
        $.min_val = reader.float32_(position, 8, -1);
        $.max_val = reader.float32_(position, 10, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Activation();
        $.type = $root.mindspore.schema.ActivationType[json.type];
        $.alpha = reader.value(json.alpha, 0.2);
        $.min_val = reader.value(json.min_val, -1);
        $.max_val = reader.value(json.max_val, 1);
        return $;
    }
};

$root.mindspore.schema.ActivationGrad = class ActivationGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ActivationGrad();
        $.type = reader.int8_(position, 4, 0);
        $.alpha = reader.float32_(position, 6, 0.2);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ActivationGrad();
        $.type = $root.mindspore.schema.ActivationType[json.type];
        $.alpha = reader.value(json.alpha, 0.2);
        return $;
    }
};

$root.mindspore.schema.Conv2D = class Conv2D {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2D();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.activationType = reader.int8_(position, 36, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2D();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Adder = class Adder {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Adder();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.activationType = reader.int8_(position, 36, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Adder();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Conv2DGradFilter = class Conv2DGradFilter {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2DGradFilter();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.filter_shape = reader.typedArray(position, 36, Int32Array);
        $.activationType = reader.int8_(position, 38, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2DGradFilter();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.filter_shape = reader.typedArray(json.filter_shape, Int32Array);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Conv2DGradInput = class Conv2DGradInput {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Conv2DGradInput();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.input_shape = reader.typedArray(position, 36, Int32Array);
        $.activationType = reader.int8_(position, 38, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Conv2DGradInput();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.input_shape = reader.typedArray(json.input_shape, Int32Array);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.GroupConv2DGradInput = class GroupConv2DGradInput {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GroupConv2DGradInput();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.input_shape = reader.typedArray(position, 36, Int32Array);
        $.activationType = reader.int8_(position, 38, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GroupConv2DGradInput();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.input_shape = reader.typedArray(json.input_shape, Int32Array);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.FusedBatchNorm = class FusedBatchNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FusedBatchNorm();
        $.epsilon = reader.float32_(position, 4, 0.00001);
        $.momentum = reader.float32_(position, 6, 0.9);
        $.spatial = reader.int32_(position, 8, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FusedBatchNorm();
        $.epsilon = reader.value(json.epsilon, 0.00001);
        $.momentum = reader.value(json.momentum, 0.9);
        $.spatial = reader.value(json.spatial, 1);
        return $;
    }
};

$root.mindspore.schema.BatchNorm = class BatchNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchNorm();
        $.epsilon = reader.float32_(position, 4, 0.00001);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchNorm();
        $.epsilon = reader.value(json.epsilon, 0.00001);
        return $;
    }
};

$root.mindspore.schema.BiasGrad = class BiasGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.BiasGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.BiasGrad();
        return $;
    }
};

$root.mindspore.schema.SoftmaxCrossEntropy = class SoftmaxCrossEntropy {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.SoftmaxCrossEntropy();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.SoftmaxCrossEntropy();
        return $;
    }
};

$root.mindspore.schema.SparseSoftmaxCrossEntropy = class SparseSoftmaxCrossEntropy {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SparseSoftmaxCrossEntropy();
        $.isGrad = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SparseSoftmaxCrossEntropy();
        $.isGrad = reader.value(json.isGrad, false);
        return $;
    }
};

$root.mindspore.schema.make_tuple = class make_tuple {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.make_tuple();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.make_tuple();
        return $;
    }
};

$root.mindspore.schema.PoolingGrad = class PoolingGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PoolingGrad();
        $.format = reader.int32_(position, 4, 0);
        $.poolingMode = reader.int8_(position, 6, 0);
        $.global = reader.bool_(position, 8, false);
        $.windowW = reader.int32_(position, 10, 0);
        $.windowH = reader.int32_(position, 12, 0);
        $.strideW = reader.int32_(position, 14, 0);
        $.strideH = reader.int32_(position, 16, 0);
        $.padMode = reader.int8_(position, 18, 0);
        $.padUp = reader.int32_(position, 20, 0);
        $.padDown = reader.int32_(position, 22, 0);
        $.padLeft = reader.int32_(position, 24, 0);
        $.padRight = reader.int32_(position, 26, 0);
        $.roundMode = reader.int8_(position, 28, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PoolingGrad();
        $.format = $root.mindspore.schema.Format[json.format];
        $.poolingMode = $root.mindspore.schema.PoolMode[json.poolingMode];
        $.global = reader.value(json.global, false);
        $.windowW = reader.value(json.windowW, 0);
        $.windowH = reader.value(json.windowH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.roundMode = $root.mindspore.schema.RoundMode[json.roundMode];
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

$root.mindspore.schema.ConstantOfShape = class ConstantOfShape {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ConstantOfShape();
        $.dataType = reader.int32_(position, 4, 0);
        $.value = reader.typedArray(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ConstantOfShape();
        $.dataType = reader.value(json.dataType, 0);
        $.value = reader.typedArray(json.value, Float32Array);
        return $;
    }
};

$root.mindspore.schema.Nchw2Nhwc = class Nchw2Nhwc {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Nchw2Nhwc();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Nchw2Nhwc();
        return $;
    }
};

$root.mindspore.schema.Nhwc2Nchw = class Nhwc2Nchw {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Nhwc2Nchw();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Nhwc2Nchw();
        return $;
    }
};

$root.mindspore.schema.FakeQuantWithMinMaxVars = class FakeQuantWithMinMaxVars {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxVars();
        $.narrowRange = reader.bool_(position, 4, false);
        $.numBits = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxVars();
        $.narrowRange = reader.value(json.narrowRange, false);
        $.numBits = reader.value(json.numBits, 0);
        return $;
    }
};

$root.mindspore.schema.BiasAdd = class BiasAdd {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BiasAdd();
        $.axis = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BiasAdd();
        $.axis = reader.typedArray(json.axis, Int32Array);
        return $;
    }
};

$root.mindspore.schema.ROIPooling = class ROIPooling {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ROIPooling();
        $.pooledH = reader.int32_(position, 4, 0);
        $.pooledW = reader.int32_(position, 6, 0);
        $.scale = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ROIPooling();
        $.pooledH = reader.value(json.pooledH, 0);
        $.pooledW = reader.value(json.pooledW, 0);
        $.scale = reader.value(json.scale, 0);
        return $;
    }
};

$root.mindspore.schema.Pooling = class Pooling {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Pooling();
        $.format = reader.int32_(position, 4, 0);
        $.poolingMode = reader.int8_(position, 6, 0);
        $.global = reader.bool_(position, 8, false);
        $.windowW = reader.int32_(position, 10, 0);
        $.windowH = reader.int32_(position, 12, 0);
        $.strideW = reader.int32_(position, 14, 0);
        $.strideH = reader.int32_(position, 16, 0);
        $.padMode = reader.int8_(position, 18, 0);
        $.padUp = reader.int32_(position, 20, 0);
        $.padDown = reader.int32_(position, 22, 0);
        $.padLeft = reader.int32_(position, 24, 0);
        $.padRight = reader.int32_(position, 26, 0);
        $.roundMode = reader.int8_(position, 28, 0);
        $.activationType = reader.int8_(position, 30, 0);
        $.avgMode = reader.int32_(position, 32, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Pooling();
        $.format = $root.mindspore.schema.Format[json.format];
        $.poolingMode = $root.mindspore.schema.PoolMode[json.poolingMode];
        $.global = reader.value(json.global, false);
        $.windowW = reader.value(json.windowW, 0);
        $.windowH = reader.value(json.windowH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.roundMode = $root.mindspore.schema.RoundMode[json.roundMode];
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        $.avgMode = reader.value(json.avgMode, 0);
        return $;
    }
};

$root.mindspore.schema.DepthwiseConv2D = class DepthwiseConv2D {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DepthwiseConv2D();
        $.format = reader.int32_(position, 4, 0);
        $.channelIn = reader.int32_(position, 6, 0);
        $.channelMultiplier = reader.int32_(position, 8, 0);
        $.kernelW = reader.int32_(position, 10, 0);
        $.kernelH = reader.int32_(position, 12, 0);
        $.strideW = reader.int32_(position, 14, 0);
        $.strideH = reader.int32_(position, 16, 0);
        $.padMode = reader.int8_(position, 18, 0);
        $.padUp = reader.int32_(position, 20, 0);
        $.padDown = reader.int32_(position, 22, 0);
        $.padLeft = reader.int32_(position, 24, 0);
        $.padRight = reader.int32_(position, 26, 0);
        $.dilateW = reader.int32_(position, 28, 0);
        $.dilateH = reader.int32_(position, 30, 0);
        $.hasBias = reader.bool_(position, 32, false);
        $.activationType = reader.int8_(position, 34, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DepthwiseConv2D();
        $.format = $root.mindspore.schema.Format[json.format];
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelMultiplier = reader.value(json.channelMultiplier, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.DeDepthwiseConv2D = class DeDepthwiseConv2D {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DeDepthwiseConv2D();
        $.format = reader.int32_(position, 4, 0);
        $.channelIn = reader.int32_(position, 6, 0);
        $.channelMultiplier = reader.int32_(position, 8, 0);
        $.kernelW = reader.int32_(position, 10, 0);
        $.kernelH = reader.int32_(position, 12, 0);
        $.strideW = reader.int32_(position, 14, 0);
        $.strideH = reader.int32_(position, 16, 0);
        $.padMode = reader.int8_(position, 18, 0);
        $.padUp = reader.int32_(position, 20, 0);
        $.padDown = reader.int32_(position, 22, 0);
        $.padLeft = reader.int32_(position, 24, 0);
        $.padRight = reader.int32_(position, 26, 0);
        $.dilateW = reader.int32_(position, 28, 0);
        $.dilateH = reader.int32_(position, 30, 0);
        $.hasBias = reader.bool_(position, 32, false);
        $.activationType = reader.int8_(position, 34, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DeDepthwiseConv2D();
        $.format = $root.mindspore.schema.Format[json.format];
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelMultiplier = reader.value(json.channelMultiplier, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Resize = class Resize {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Resize();
        $.format = reader.int32_(position, 4, 0);
        $.method = reader.int8_(position, 6, 0);
        $.newHeight = reader.int64_(position, 8, 0);
        $.newWidth = reader.int64_(position, 10, 0);
        $.alignCorners = reader.bool_(position, 12, false);
        $.preserveAspectRatio = reader.bool_(position, 14, false);
        $.coordinateTransformMode = reader.int8_(position, 16, 0);
        $.cubicCoeff = reader.float32_(position, 18, 0);
        $.excludeOutside = reader.int32_(position, 20, 0);
        $.extrapolationValue = reader.float32_(position, 22, 0);
        $.nearestMode = reader.int8_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Resize();
        $.format = $root.mindspore.schema.Format[json.format];
        $.method = $root.mindspore.schema.ResizeMethod[json.method];
        $.newHeight = reader.value(json.newHeight, 0);
        $.newWidth = reader.value(json.newWidth, 0);
        $.alignCorners = reader.value(json.alignCorners, false);
        $.preserveAspectRatio = reader.value(json.preserveAspectRatio, false);
        $.coordinateTransformMode = $root.mindspore.schema.CoordinateTransformMode[json.coordinateTransformMode];
        $.cubicCoeff = reader.value(json.cubicCoeff, 0);
        $.excludeOutside = reader.value(json.excludeOutside, 0);
        $.extrapolationValue = reader.value(json.extrapolationValue, 0);
        $.nearestMode = $root.mindspore.schema.NearestMode[json.nearestMode];
        return $;
    }
};

$root.mindspore.schema.DetectionPostProcess = class DetectionPostProcess {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DetectionPostProcess();
        $.format = reader.int32_(position, 4, 0);
        $.inputSize = reader.int32_(position, 6, 0);
        $.hScale = reader.float32_(position, 8, 0);
        $.wScale = reader.float32_(position, 10, 0);
        $.xScale = reader.float32_(position, 12, 0);
        $.yScale = reader.float32_(position, 14, 0);
        $.NmsIouThreshold = reader.float32_(position, 16, 0);
        $.NmsScoreThreshold = reader.float32_(position, 18, 0);
        $.MaxDetections = reader.int64_(position, 20, 0);
        $.DetectionsPerClass = reader.int64_(position, 22, 0);
        $.MaxClassesPerDetection = reader.int64_(position, 24, 0);
        $.NumClasses = reader.int64_(position, 26, 0);
        $.UseRegularNms = reader.bool_(position, 28, false);
        $.OutQuantized = reader.bool_(position, 30, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DetectionPostProcess();
        $.format = $root.mindspore.schema.Format[json.format];
        $.inputSize = reader.value(json.inputSize, 0);
        $.hScale = reader.value(json.hScale, 0);
        $.wScale = reader.value(json.wScale, 0);
        $.xScale = reader.value(json.xScale, 0);
        $.yScale = reader.value(json.yScale, 0);
        $.NmsIouThreshold = reader.value(json.NmsIouThreshold, 0);
        $.NmsScoreThreshold = reader.value(json.NmsScoreThreshold, 0);
        $.MaxDetections = reader.value(json.MaxDetections, 0);
        $.DetectionsPerClass = reader.value(json.DetectionsPerClass, 0);
        $.MaxClassesPerDetection = reader.value(json.MaxClassesPerDetection, 0);
        $.NumClasses = reader.value(json.NumClasses, 0);
        $.UseRegularNms = reader.value(json.UseRegularNms, false);
        $.OutQuantized = reader.value(json.OutQuantized, false);
        return $;
    }
};

$root.mindspore.schema.FullConnection = class FullConnection {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.FullConnection();
        $.hasBias = reader.bool_(position, 4, false);
        $.axis = reader.int32_(position, 6, 0);
        $.useAxis = reader.bool_(position, 8, false);
        $.activationType = reader.int8_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.FullConnection();
        $.hasBias = reader.value(json.hasBias, false);
        $.axis = reader.value(json.axis, 0);
        $.useAxis = reader.value(json.useAxis, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Mean = class Mean {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Mean();
        $.axis = reader.typedArray(position, 4, Int32Array);
        $.keepDims = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Mean();
        $.axis = reader.typedArray(json.axis, Int32Array);
        $.keepDims = reader.value(json.keepDims, false);
        return $;
    }
};

$root.mindspore.schema.DeConv2D = class DeConv2D {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DeConv2D();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.activationType = reader.int8_(position, 36, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DeConv2D();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.DeConv2DGradFilter = class DeConv2DGradFilter {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DeConv2DGradFilter();
        $.format = reader.int32_(position, 4, 0);
        $.group = reader.int32_(position, 6, 0);
        $.channelIn = reader.int32_(position, 8, 0);
        $.channelOut = reader.int32_(position, 10, 0);
        $.kernelW = reader.int32_(position, 12, 0);
        $.kernelH = reader.int32_(position, 14, 0);
        $.strideW = reader.int32_(position, 16, 0);
        $.strideH = reader.int32_(position, 18, 0);
        $.padMode = reader.int8_(position, 20, 0);
        $.padUp = reader.int32_(position, 22, 0);
        $.padDown = reader.int32_(position, 24, 0);
        $.padLeft = reader.int32_(position, 26, 0);
        $.padRight = reader.int32_(position, 28, 0);
        $.dilateW = reader.int32_(position, 30, 0);
        $.dilateH = reader.int32_(position, 32, 0);
        $.hasBias = reader.bool_(position, 34, false);
        $.activationType = reader.int8_(position, 36, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DeConv2DGradFilter();
        $.format = $root.mindspore.schema.Format[json.format];
        $.group = reader.value(json.group, 0);
        $.channelIn = reader.value(json.channelIn, 0);
        $.channelOut = reader.value(json.channelOut, 0);
        $.kernelW = reader.value(json.kernelW, 0);
        $.kernelH = reader.value(json.kernelH, 0);
        $.strideW = reader.value(json.strideW, 0);
        $.strideH = reader.value(json.strideH, 0);
        $.padMode = $root.mindspore.schema.PadMode[json.padMode];
        $.padUp = reader.value(json.padUp, 0);
        $.padDown = reader.value(json.padDown, 0);
        $.padLeft = reader.value(json.padLeft, 0);
        $.padRight = reader.value(json.padRight, 0);
        $.dilateW = reader.value(json.dilateW, 0);
        $.dilateH = reader.value(json.dilateH, 0);
        $.hasBias = reader.value(json.hasBias, false);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.BNGrad = class BNGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BNGrad();
        $.eps = reader.float32_(position, 4, 0);
        $.momentum = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BNGrad();
        $.eps = reader.value(json.eps, 0);
        $.momentum = reader.value(json.momentum, 0);
        return $;
    }
};

$root.mindspore.schema.Scale = class Scale {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Scale();
        $.axis = reader.int32_(position, 4, 0);
        $.activationType = reader.int8_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Scale();
        $.axis = reader.value(json.axis, 0);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
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

$root.mindspore.schema.Add = class Add {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Add();
        $.activationType = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Add();
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Sub = class Sub {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Sub();
        $.activationType = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Sub();
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Mul = class Mul {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Mul();
        $.activationType = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Mul();
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
        return $;
    }
};

$root.mindspore.schema.Div = class Div {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Div();
        $.activationType = reader.int8_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Div();
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
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

$root.mindspore.schema.Min = class Min {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Min();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Min();
        return $;
    }
};

$root.mindspore.schema.Slice = class Slice {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Slice();
        $.format = reader.int32_(position, 4, 0);
        $.axes = reader.typedArray(position, 6, Int32Array);
        $.begin = reader.typedArray(position, 8, Int32Array);
        $.size = reader.typedArray(position, 10, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Slice();
        $.format = $root.mindspore.schema.Format[json.format];
        $.axes = reader.typedArray(json.axes, Int32Array);
        $.begin = reader.typedArray(json.begin, Int32Array);
        $.size = reader.typedArray(json.size, Int32Array);
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

$root.mindspore.schema.Exp = class Exp {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Exp();
        $.base = reader.float32_(position, 4, -1);
        $.scale = reader.float32_(position, 6, 1);
        $.shift = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Exp();
        $.base = reader.value(json.base, -1);
        $.scale = reader.value(json.scale, 1);
        $.shift = reader.value(json.shift, 0);
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

$root.mindspore.schema.Tan = class Tan {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Tan();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Tan();
        return $;
    }
};

$root.mindspore.schema.Atan = class Atan {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Atan();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Atan();
        return $;
    }
};

$root.mindspore.schema.Asin = class Asin {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Asin();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Asin();
        return $;
    }
};

$root.mindspore.schema.Reshape = class Reshape {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Reshape();
        $.format = reader.int32_(position, 4, 0);
        $.shape = reader.int64s_(position, 6);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Reshape();
        $.format = $root.mindspore.schema.Format[json.format];
        $.shape = reader.array(json.shape);
        return $;
    }
};

$root.mindspore.schema.Power = class Power {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Power();
        $.power = reader.float32_(position, 4, 0);
        $.scale = reader.float32_(position, 6, 0);
        $.shift = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Power();
        $.power = reader.value(json.power, 0);
        $.scale = reader.value(json.scale, 0);
        $.shift = reader.value(json.shift, 0);
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

$root.mindspore.schema.ArgMax = class ArgMax {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ArgMax();
        $.axis = reader.int32_(position, 4, 0);
        $.outMaxValue = reader.bool_(position, 6, false);
        $.topK = reader.int32_(position, 8, 1);
        $.keepDims = reader.bool_(position, 10, false);
        $.axisType = reader.int32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ArgMax();
        $.axis = reader.value(json.axis, 0);
        $.outMaxValue = reader.value(json.outMaxValue, false);
        $.topK = reader.value(json.topK, 1);
        $.keepDims = reader.value(json.keepDims, false);
        $.axisType = reader.value(json.axisType, 0);
        return $;
    }
};

$root.mindspore.schema.ArgMin = class ArgMin {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ArgMin();
        $.axis = reader.int32_(position, 4, 0);
        $.outMaxValue = reader.bool_(position, 6, false);
        $.topK = reader.int32_(position, 8, 1);
        $.keepDims = reader.bool_(position, 10, false);
        $.axisType = reader.int32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ArgMin();
        $.axis = reader.value(json.axis, 0);
        $.outMaxValue = reader.value(json.outMaxValue, false);
        $.topK = reader.value(json.topK, 1);
        $.keepDims = reader.value(json.keepDims, false);
        $.axisType = reader.value(json.axisType, 0);
        return $;
    }
};

$root.mindspore.schema.NetOutput = class NetOutput {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.NetOutput();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.NetOutput();
        return $;
    }
};

$root.mindspore.schema.MatMul = class MatMul {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MatMul();
        $.broadcast = reader.bool_(position, 4, false);
        $.transposeA = reader.bool_(position, 6, false);
        $.transposeB = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MatMul();
        $.broadcast = reader.value(json.broadcast, false);
        $.transposeA = reader.value(json.transposeA, false);
        $.transposeB = reader.value(json.transposeB, false);
        return $;
    }
};

$root.mindspore.schema.PReLU = class PReLU {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PReLU();
        $.channelShared = reader.bool_(position, 4, false);
        $.slope = reader.typedArray(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PReLU();
        $.channelShared = reader.value(json.channelShared, false);
        $.slope = reader.typedArray(json.slope, Float32Array);
        return $;
    }
};

$root.mindspore.schema.LeakyReLU = class LeakyReLU {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LeakyReLU();
        $.negativeSlope = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LeakyReLU();
        $.negativeSlope = reader.value(json.negativeSlope, 0);
        return $;
    }
};

$root.mindspore.schema.StridedSlice = class StridedSlice {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.StridedSlice();
        $.beginMask = reader.int32_(position, 4, 0);
        $.endMask = reader.int32_(position, 6, 0);
        $.ellipsisMask = reader.int32_(position, 8, 0);
        $.newAxisMask = reader.int32_(position, 10, 0);
        $.shrinkAxisMask = reader.int32_(position, 12, 0);
        $.begin = reader.typedArray(position, 14, Int32Array);
        $.end = reader.typedArray(position, 16, Int32Array);
        $.stride = reader.typedArray(position, 18, Int32Array);
        $.isScale = reader.typedArray(position, 20, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.StridedSlice();
        $.beginMask = reader.value(json.beginMask, 0);
        $.endMask = reader.value(json.endMask, 0);
        $.ellipsisMask = reader.value(json.ellipsisMask, 0);
        $.newAxisMask = reader.value(json.newAxisMask, 0);
        $.shrinkAxisMask = reader.value(json.shrinkAxisMask, 0);
        $.begin = reader.typedArray(json.begin, Int32Array);
        $.end = reader.typedArray(json.end, Int32Array);
        $.stride = reader.typedArray(json.stride, Int32Array);
        $.isScale = reader.typedArray(json.isScale, Int32Array);
        return $;
    }
};

$root.mindspore.schema.Stack = class Stack {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Stack();
        $.axis = reader.int32_(position, 4, 0);
        $.n = reader.int32_(position, 6, 0);
        $.isScale = reader.typedArray(position, 8, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Stack();
        $.axis = reader.value(json.axis, 0);
        $.n = reader.value(json.n, 0);
        $.isScale = reader.typedArray(json.isScale, Int32Array);
        return $;
    }
};

$root.mindspore.schema.Range = class Range {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Range();
        $.dType = reader.int32_(position, 4, 0);
        $.start = reader.int32_(position, 6, 0);
        $.limit = reader.int32_(position, 8, 0);
        $.delta = reader.int32_(position, 10, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Range();
        $.dType = reader.value(json.dType, 0);
        $.start = reader.value(json.start, 0);
        $.limit = reader.value(json.limit, 0);
        $.delta = reader.value(json.delta, 1);
        return $;
    }
};

$root.mindspore.schema.ExpandDims = class ExpandDims {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ExpandDims();
        $.dim = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ExpandDims();
        $.dim = reader.value(json.dim, 0);
        return $;
    }
};

$root.mindspore.schema.Tile = class Tile {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Tile();
        $.multiples = reader.typedArray(position, 4, Int32Array);
        $.dims = reader.typedArray(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Tile();
        $.multiples = reader.typedArray(json.multiples, Int32Array);
        $.dims = reader.typedArray(json.dims, Int32Array);
        return $;
    }
};

$root.mindspore.schema.Cast = class Cast {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Cast();
        $.srcT = reader.int32_(position, 4, 0);
        $.dstT = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Cast();
        $.srcT = reader.value(json.srcT, 0);
        $.dstT = reader.value(json.dstT, 0);
        return $;
    }
};

$root.mindspore.schema.QuantDTypeCast = class QuantDTypeCast {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.QuantDTypeCast();
        $.srcT = reader.int32_(position, 4, 0);
        $.dstT = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.QuantDTypeCast();
        $.srcT = reader.value(json.srcT, 0);
        $.dstT = reader.value(json.dstT, 0);
        return $;
    }
};

$root.mindspore.schema.Split = class Split {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Split();
        $.numberSplit = reader.int32_(position, 4, 0);
        $.sizeSplits = reader.typedArray(position, 6, Int32Array);
        $.splitDim = reader.int32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Split();
        $.numberSplit = reader.value(json.numberSplit, 0);
        $.sizeSplits = reader.typedArray(json.sizeSplits, Int32Array);
        $.splitDim = reader.value(json.splitDim, 0);
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

$root.mindspore.schema.Permute = class Permute {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Permute();
        $.order = reader.int64s_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Permute();
        $.order = reader.array(json.order);
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

$root.mindspore.schema.Constant = class Constant {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Constant();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Constant();
        return $;
    }
};

$root.mindspore.schema.Elu = class Elu {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Elu();
        $.alpha = reader.float32_(position, 4, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Elu();
        $.alpha = reader.value(json.alpha, 1);
        return $;
    }
};

$root.mindspore.schema.Broadcast = class Broadcast {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Broadcast();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Broadcast();
        return $;
    }
};

$root.mindspore.schema.BroadcastTo = class BroadcastTo {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BroadcastTo();
        $.dst_shape = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BroadcastTo();
        $.dst_shape = reader.typedArray(json.dst_shape, Int32Array);
        return $;
    }
};

$root.mindspore.schema.Lrn = class Lrn {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Lrn();
        $.alpha = reader.float32_(position, 4, 0.0001);
        $.beta = reader.float32_(position, 6, 0.75);
        $.bias = reader.float32_(position, 8, 1);
        $.size = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Lrn();
        $.alpha = reader.value(json.alpha, 0.0001);
        $.beta = reader.value(json.beta, 0.75);
        $.bias = reader.value(json.bias, 1);
        $.size = reader.value(json.size, 0);
        return $;
    }
};

$root.mindspore.schema.ReduceMode = {
    ReduceMean: 0,
    ReduceMax: 1,
    ReduceMin: 2,
    ReduceProd: 3,
    ReduceSum: 4,
    ReduceSumSquare: 5,
    ReduceASum: 6,
    ReduceAll: 7
};

$root.mindspore.schema.Reduce = class Reduce {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Reduce();
        $.axes = reader.typedArray(position, 4, Int32Array);
        $.keepDims = reader.int32_(position, 6, 0);
        $.mode = reader.int8_(position, 8, 0);
        $.reduceToEnd = reader.bool_(position, 10, false);
        $.coeff = reader.float32_(position, 12, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Reduce();
        $.axes = reader.typedArray(json.axes, Int32Array);
        $.keepDims = reader.value(json.keepDims, 0);
        $.mode = $root.mindspore.schema.ReduceMode[json.mode];
        $.reduceToEnd = reader.value(json.reduceToEnd, false);
        $.coeff = reader.value(json.coeff, 1);
        return $;
    }
};

$root.mindspore.schema.Transpose = class Transpose {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Transpose();
        $.perm = reader.typedArray(position, 4, Int32Array);
        $.conjugate = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Transpose();
        $.perm = reader.typedArray(json.perm, Int32Array);
        $.conjugate = reader.value(json.conjugate, false);
        return $;
    }
};

$root.mindspore.schema.Squeeze = class Squeeze {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Squeeze();
        $.axis = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Squeeze();
        $.axis = reader.typedArray(json.axis, Int32Array);
        return $;
    }
};

$root.mindspore.schema.Unsqueeze = class Unsqueeze {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Unsqueeze();
        $.axis = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Unsqueeze();
        $.axis = reader.typedArray(json.axis, Int32Array);
        return $;
    }
};

$root.mindspore.schema.Upsample = class Upsample {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Upsample();
        $.mode = reader.string_(position, 4, null);
        $.scales = reader.typedArray(position, 6, Float32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Upsample();
        $.mode = reader.value(json.mode, null);
        $.scales = reader.typedArray(json.scales, Float32Array);
        return $;
    }
};

$root.mindspore.schema.Dropout = class Dropout {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Dropout();
        $.ratio = reader.float32_(position, 4, 0.5);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Dropout();
        $.ratio = reader.value(json.ratio, 0.5);
        return $;
    }
};

$root.mindspore.schema.LocalResponseNormalization = class LocalResponseNormalization {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LocalResponseNormalization();
        $.depth_radius = reader.int32_(position, 4, 0);
        $.bias = reader.float32_(position, 6, 0);
        $.alpha = reader.float32_(position, 8, 0);
        $.beta = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LocalResponseNormalization();
        $.depth_radius = reader.value(json.depth_radius, 0);
        $.bias = reader.value(json.bias, 0);
        $.alpha = reader.value(json.alpha, 0);
        $.beta = reader.value(json.beta, 0);
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

$root.mindspore.schema.TopK = class TopK {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TopK();
        $.k = reader.int32_(position, 4, 0);
        $.sorted = reader.bool_(position, 6, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TopK();
        $.k = reader.value(json.k, 0);
        $.sorted = reader.value(json.sorted, true);
        return $;
    }
};

$root.mindspore.schema.SpaceToDepth = class SpaceToDepth {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SpaceToDepth();
        $.blockSize = reader.int32_(position, 4, 0);
        $.format = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SpaceToDepth();
        $.blockSize = reader.value(json.blockSize, 0);
        $.format = $root.mindspore.schema.Format[json.format];
        return $;
    }
};

$root.mindspore.schema.SpaceToBatch = class SpaceToBatch {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SpaceToBatch();
        $.blockShape = reader.typedArray(position, 4, Int32Array);
        $.paddings = reader.typedArray(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SpaceToBatch();
        $.blockShape = reader.typedArray(json.blockShape, Int32Array);
        $.paddings = reader.typedArray(json.paddings, Int32Array);
        return $;
    }
};

$root.mindspore.schema.SparseToDense = class SparseToDense {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SparseToDense();
        $.validateIndices = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SparseToDense();
        $.validateIndices = reader.value(json.validateIndices, false);
        return $;
    }
};

$root.mindspore.schema.ReverseSequence = class ReverseSequence {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ReverseSequence();
        $.seqAxis = reader.int32_(position, 4, 0);
        $.batchAxis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ReverseSequence();
        $.seqAxis = reader.value(json.seqAxis, 0);
        $.batchAxis = reader.value(json.batchAxis, 0);
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

$root.mindspore.schema.Gather = class Gather {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Gather();
        $.axis = reader.int32_(position, 4, 0);
        $.batchDims = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Gather();
        $.axis = reader.value(json.axis, 0);
        $.batchDims = reader.value(json.batchDims, 0);
        return $;
    }
};

$root.mindspore.schema.GatherNd = class GatherNd {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GatherNd();
        $.batchDims = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GatherNd();
        $.batchDims = reader.value(json.batchDims, 0);
        return $;
    }
};

$root.mindspore.schema.Fill = class Fill {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Fill();
        $.dims = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Fill();
        $.dims = reader.typedArray(json.dims, Int32Array);
        return $;
    }
};

$root.mindspore.schema.DepthToSpace = class DepthToSpace {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DepthToSpace();
        $.blockSize = reader.int32_(position, 4, 0);
        $.format = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DepthToSpace();
        $.blockSize = reader.value(json.blockSize, 0);
        $.format = $root.mindspore.schema.Format[json.format];
        return $;
    }
};

$root.mindspore.schema.BatchToSpace = class BatchToSpace {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchToSpace();
        $.blockShape = reader.typedArray(position, 4, Int32Array);
        $.crops = reader.typedArray(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchToSpace();
        $.blockShape = reader.typedArray(json.blockShape, Int32Array);
        $.crops = reader.typedArray(json.crops, Int32Array);
        return $;
    }
};

$root.mindspore.schema.BatchToSpaceND = class BatchToSpaceND {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BatchToSpaceND();
        $.blockShape = reader.typedArray(position, 4, Int32Array);
        $.crops = reader.typedArray(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BatchToSpaceND();
        $.blockShape = reader.typedArray(json.blockShape, Int32Array);
        $.crops = reader.typedArray(json.crops, Int32Array);
        return $;
    }
};

$root.mindspore.schema.AddN = class AddN {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AddN();
        $.N = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AddN();
        $.N = reader.value(json.N, 0);
        return $;
    }
};

$root.mindspore.schema.EmbeddingLookup = class EmbeddingLookup {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.EmbeddingLookup();
        $.maxNorm = reader.float32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.EmbeddingLookup();
        $.maxNorm = reader.value(json.maxNorm, 0);
        return $;
    }
};

$root.mindspore.schema.EmbeddingLookupSparse = class EmbeddingLookupSparse {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.EmbeddingLookupSparse();
        $.spIds = reader.typedArray(position, 4, Int32Array);
        $.spWeights = reader.typedArray(position, 6, Float32Array);
        $.maxNortm = reader.float32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.EmbeddingLookupSparse();
        $.spIds = reader.typedArray(json.spIds, Int32Array);
        $.spWeights = reader.typedArray(json.spWeights, Float32Array);
        $.maxNortm = reader.value(json.maxNortm, 0);
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

$root.mindspore.schema.L2Norm = class L2Norm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.L2Norm();
        $.axis = reader.typedArray(position, 4, Int32Array);
        $.epsilon = reader.float32_(position, 6, 0);
        $.activationType = reader.int8_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.L2Norm();
        $.axis = reader.typedArray(json.axis, Int32Array);
        $.epsilon = reader.value(json.epsilon, 0);
        $.activationType = $root.mindspore.schema.ActivationType[json.activationType];
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

$root.mindspore.schema.LogicalXor = class LogicalXor {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.LogicalXor();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.LogicalXor();
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

$root.mindspore.schema.MatrixDiag = class MatrixDiag {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.MatrixDiag();
        $.k = reader.int32_(position, 4, 0);
        $.numRows = reader.int32_(position, 6, 0);
        $.numCols = reader.int32_(position, 8, 0);
        $.paddingValue = reader.float32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.MatrixDiag();
        $.k = reader.value(json.k, 0);
        $.numRows = reader.value(json.numRows, 0);
        $.numCols = reader.value(json.numCols, 0);
        $.paddingValue = reader.value(json.paddingValue, 0);
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

$root.mindspore.schema.TfReduce = class TfReduce {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TfReduce();
        $.type = reader.int8_(position, 4, 7);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TfReduce();
        $.type = $root.mindspore.schema.ReduceType[json.type];
        return $;
    }
};

$root.mindspore.schema.Reverse = class Reverse {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Reverse();
        $.axis = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Reverse();
        $.axis = reader.typedArray(json.axis, Int32Array);
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

$root.mindspore.schema.Scatter = class Scatter {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Scatter();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Scatter();
        return $;
    }
};

$root.mindspore.schema.ScatterND = class ScatterND {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.ScatterND();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.ScatterND();
        return $;
    }
};

$root.mindspore.schema.Unique = class Unique {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Unique();
        $.outType = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Unique();
        $.outType = reader.value(json.outType, 0);
        return $;
    }
};

$root.mindspore.schema.Unstack = class Unstack {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Unstack();
        $.num = reader.int32_(position, 4, 0);
        $.axis = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Unstack();
        $.num = reader.value(json.num, 0);
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.OnnxInt8Quantize = class OnnxInt8Quantize {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.OnnxInt8Quantize();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.OnnxInt8Quantize();
        return $;
    }
};

$root.mindspore.schema.OnnxInt8Dequantize = class OnnxInt8Dequantize {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.OnnxInt8Dequantize();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.OnnxInt8Dequantize();
        return $;
    }
};

$root.mindspore.schema.FakeQuantWithMinMax = class FakeQuantWithMinMax {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMax();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMax();
        return $;
    }
};

$root.mindspore.schema.FakeQuantWithMinMaxPerChannel = class FakeQuantWithMinMaxPerChannel {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxPerChannel();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.FakeQuantWithMinMaxPerChannel();
        return $;
    }
};

$root.mindspore.schema.BatchNormFold = class BatchNormFold {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.BatchNormFold();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.BatchNormFold();
        return $;
    }
};

$root.mindspore.schema.MulFold = class MulFold {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.MulFold();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.MulFold();
        return $;
    }
};

$root.mindspore.schema.AddFold = class AddFold {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.AddFold();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.AddFold();
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

$root.mindspore.schema.TupleGetItem = class TupleGetItem {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.TupleGetItem();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.TupleGetItem();
        return $;
    }
};

$root.mindspore.schema.ApplyMomentum = class ApplyMomentum {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ApplyMomentum();
        $.gradientScale = reader.float32_(position, 4, 0);
        $.useNesterov = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ApplyMomentum();
        $.gradientScale = reader.value(json.gradientScale, 0);
        $.useNesterov = reader.value(json.useNesterov, false);
        return $;
    }
};

$root.mindspore.schema.Sgd = class Sgd {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Sgd();
        $.weightDecay = reader.float32_(position, 4, 0);
        $.dampening = reader.float32_(position, 6, 0);
        $.useNesterov = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Sgd();
        $.weightDecay = reader.value(json.weightDecay, 0);
        $.dampening = reader.value(json.dampening, 0);
        $.useNesterov = reader.value(json.useNesterov, false);
        return $;
    }
};

$root.mindspore.schema.Adam = class Adam {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Adam();
        $.useNesterov = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Adam();
        $.useNesterov = reader.value(json.useNesterov, false);
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

$root.mindspore.schema.Where = class Where {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Where();
        $.condition = reader.bools_(position, 4);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Where();
        $.condition = reader.array(json.condition);
        return $;
    }
};

$root.mindspore.schema.OneHot = class OneHot {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.OneHot();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.OneHot();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

$root.mindspore.schema.Lstm = class Lstm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Lstm();
        $.bidirection = reader.bool_(position, 4, false);
        $.smooth = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Lstm();
        $.bidirection = reader.value(json.bidirection, false);
        $.smooth = reader.value(json.smooth, 0);
        return $;
    }
};

$root.mindspore.schema.Gru = class Gru {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Gru();
        $.bidirection = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Gru();
        $.bidirection = reader.value(json.bidirection, false);
        return $;
    }
};

$root.mindspore.schema.PriorBox = class PriorBox {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.PriorBox();
        $.min_sizes = reader.typedArray(position, 4, Int32Array);
        $.max_sizes = reader.typedArray(position, 6, Int32Array);
        $.aspect_ratios = reader.typedArray(position, 8, Float32Array);
        $.variances = reader.typedArray(position, 10, Float32Array);
        $.image_size_w = reader.int32_(position, 12, 0);
        $.image_size_h = reader.int32_(position, 14, 0);
        $.step_w = reader.float32_(position, 16, 0);
        $.step_h = reader.float32_(position, 18, 0);
        $.clip = reader.bool_(position, 20, true);
        $.flip = reader.bool_(position, 22, true);
        $.offset = reader.float32_(position, 24, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.PriorBox();
        $.min_sizes = reader.typedArray(json.min_sizes, Int32Array);
        $.max_sizes = reader.typedArray(json.max_sizes, Int32Array);
        $.aspect_ratios = reader.typedArray(json.aspect_ratios, Float32Array);
        $.variances = reader.typedArray(json.variances, Float32Array);
        $.image_size_w = reader.value(json.image_size_w, 0);
        $.image_size_h = reader.value(json.image_size_h, 0);
        $.step_w = reader.value(json.step_w, 0);
        $.step_h = reader.value(json.step_h, 0);
        $.clip = reader.value(json.clip, true);
        $.flip = reader.value(json.flip, true);
        $.offset = reader.value(json.offset, 0);
        return $;
    }
};

$root.mindspore.schema.SpaceToBatchND = class SpaceToBatchND {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SpaceToBatchND();
        $.blockShape = reader.typedArray(position, 4, Int32Array);
        $.paddings = reader.typedArray(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SpaceToBatchND();
        $.blockShape = reader.typedArray(json.blockShape, Int32Array);
        $.paddings = reader.typedArray(json.paddings, Int32Array);
        return $;
    }
};

$root.mindspore.schema.MakeTuple = class MakeTuple {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.MakeTuple();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.MakeTuple();
        return $;
    }
};

$root.mindspore.schema.ToFormat = class ToFormat {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.ToFormat();
        $.srcT = reader.int32_(position, 4, 0);
        $.dstT = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.ToFormat();
        $.srcT = reader.value(json.srcT, 0);
        $.dstT = reader.value(json.dstT, 0);
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

$root.mindspore.schema.ControlDepend = class ControlDepend {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.ControlDepend();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.ControlDepend();
        return $;
    }
};

$root.mindspore.schema.Return = class Return {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Return();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Return();
        return $;
    }
};

$root.mindspore.schema.Proposal = class Proposal {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Proposal();
        $.feat_stride = reader.float32_(position, 4, 0);
        $.base_size = reader.float32_(position, 6, 0);
        $.min_size = reader.float32_(position, 8, 0);
        $.ratio = reader.typedArray(position, 10, Float32Array);
        $.scale = reader.typedArray(position, 12, Float32Array);
        $.pre_nms_topn = reader.int32_(position, 14, 0);
        $.post_nms_topn = reader.int32_(position, 16, 0);
        $.nms_thresh = reader.float32_(position, 18, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Proposal();
        $.feat_stride = reader.value(json.feat_stride, 0);
        $.base_size = reader.value(json.base_size, 0);
        $.min_size = reader.value(json.min_size, 0);
        $.ratio = reader.typedArray(json.ratio, Float32Array);
        $.scale = reader.typedArray(json.scale, Float32Array);
        $.pre_nms_topn = reader.value(json.pre_nms_topn, 0);
        $.post_nms_topn = reader.value(json.post_nms_topn, 0);
        $.nms_thresh = reader.value(json.nms_thresh, 0);
        return $;
    }
};

$root.mindspore.schema.Custom = class Custom {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Custom();
        $.custom = reader.typedArray(position, 4, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Custom();
        $.custom = reader.typedArray(json.custom, Uint8Array);
        return $;
    }
};

$root.mindspore.schema.BlackBox = class BlackBox {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BlackBox();
        $.id = reader.string_(position, 4, null);
        $.size = reader.int32_(position, 6, 0);
        $.address = reader.typedArray(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BlackBox();
        $.id = reader.value(json.id, null);
        $.size = reader.value(json.size, 0);
        $.address = reader.typedArray(json.address, Uint8Array);
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

$root.mindspore.schema.SkipGram = class SkipGram {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.SkipGram();
        $.includeAllGrams = reader.bool_(position, 4, false);
        $.maxSkipSize = reader.int32_(position, 6, 0);
        $.ngramSize = reader.int32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.SkipGram();
        $.includeAllGrams = reader.value(json.includeAllGrams, false);
        $.maxSkipSize = reader.value(json.maxSkipSize, 0);
        $.ngramSize = reader.value(json.ngramSize, 0);
        return $;
    }
};

$root.mindspore.schema.CustomPredict = class CustomPredict {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.CustomPredict();
        $.outputNum = reader.int32_(position, 4, 0);
        $.weightThreshold = reader.float32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.CustomPredict();
        $.outputNum = reader.value(json.outputNum, 0);
        $.weightThreshold = reader.value(json.weightThreshold, 0);
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

$root.mindspore.schema.AudioSpectrogram = class AudioSpectrogram {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.AudioSpectrogram();
        $.windowSize = reader.int32_(position, 4, 0);
        $.stride = reader.int32_(position, 6, 0);
        $.magSquare = reader.bool_(position, 8, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.AudioSpectrogram();
        $.windowSize = reader.value(json.windowSize, 0);
        $.stride = reader.value(json.stride, 0);
        $.magSquare = reader.value(json.magSquare, false);
        return $;
    }
};

$root.mindspore.schema.Mfcc = class Mfcc {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Mfcc();
        $.freqUpperLimit = reader.float32_(position, 4, 0);
        $.freqLowerLimit = reader.float32_(position, 6, 0);
        $.filterBankChannelNum = reader.int32_(position, 8, 0);
        $.dctCoeffNum = reader.int32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Mfcc();
        $.freqUpperLimit = reader.value(json.freqUpperLimit, 0);
        $.freqLowerLimit = reader.value(json.freqLowerLimit, 0);
        $.filterBankChannelNum = reader.value(json.filterBankChannelNum, 0);
        $.dctCoeffNum = reader.value(json.dctCoeffNum, 0);
        return $;
    }
};

$root.mindspore.schema.Rfft = class Rfft {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Rfft();
        $.fftLength = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Rfft();
        $.fftLength = reader.value(json.fftLength, 0);
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

$root.mindspore.schema.DropoutGrad = class DropoutGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.DropoutGrad();
        $.ratio = reader.float32_(position, 4, 0.5);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.DropoutGrad();
        $.ratio = reader.value(json.ratio, 0.5);
        return $;
    }
};

$root.mindspore.schema.MaximumGrad = class MaximumGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.MaximumGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.MaximumGrad();
        return $;
    }
};

$root.mindspore.schema.MinimumGrad = class MinimumGrad {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.MinimumGrad();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.MinimumGrad();
        return $;
    }
};

$root.mindspore.schema.NonMaxSuppression = class NonMaxSuppression {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.NonMaxSuppression();
        $.centerPointBox = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.NonMaxSuppression();
        $.centerPointBox = reader.value(json.centerPointBox, 0);
        return $;
    }
};

$root.mindspore.schema.InstanceNorm = class InstanceNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.InstanceNorm();
        $.epsilon = reader.float32_(position, 4, 0.00001);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.InstanceNorm();
        $.epsilon = reader.value(json.epsilon, 0.00001);
        return $;
    }
};

$root.mindspore.schema.Loop = class Loop {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Loop();
        $.subGraphIndex = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Loop();
        $.subGraphIndex = reader.value(json.subGraphIndex, 0);
        return $;
    }
};

$root.mindspore.schema.Identity = class Identity {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Identity();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Identity();
        return $;
    }
};

$root.mindspore.schema.LayerNorm = class LayerNorm {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LayerNorm();
        $.begin_norm_axis = reader.int32_(position, 4, 0);
        $.begin_params_axis = reader.int32_(position, 6, 0);
        $.epsilon = reader.float32_(position, 8, 0.00001);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LayerNorm();
        $.begin_norm_axis = reader.value(json.begin_norm_axis, 0);
        $.begin_params_axis = reader.value(json.begin_params_axis, 0);
        $.epsilon = reader.value(json.epsilon, 0.00001);
        return $;
    }
};

$root.mindspore.schema.While = class While {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.While();
        $.condSubgraphIndex = reader.int32_(position, 4, 0);
        $.bodySubgraphIndex = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.While();
        $.condSubgraphIndex = reader.value(json.condSubgraphIndex, 0);
        $.bodySubgraphIndex = reader.value(json.bodySubgraphIndex, 0);
        return $;
    }
};

$root.mindspore.schema.If = class If {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.If();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.If();
        return $;
    }
};

$root.mindspore.schema.UnsortedSegmentSum = class UnsortedSegmentSum {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.UnsortedSegmentSum();
        $.numSegments = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.UnsortedSegmentSum();
        $.numSegments = reader.value(json.numSegments, 0);
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

$root.mindspore.schema.BinaryCrossEntropy = class BinaryCrossEntropy {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropy();
        $.reduction = reader.int32_(position, 4, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropy();
        $.reduction = reader.value(json.reduction, 1);
        return $;
    }
};

$root.mindspore.schema.BinaryCrossEntropyGrad = class BinaryCrossEntropyGrad {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropyGrad();
        $.reduction = reader.int32_(position, 4, 1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.BinaryCrossEntropyGrad();
        $.reduction = reader.value(json.reduction, 1);
        return $;
    }
};

$root.mindspore.schema.LpNormalization = class LpNormalization {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.LpNormalization();
        $.axis = reader.int32_(position, 4, 0);
        $.p = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.LpNormalization();
        $.axis = reader.value(json.axis, 0);
        $.p = reader.value(json.p, 0);
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

$root.mindspore.schema.Partial = class Partial {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Partial();
        $.subGraphIndex = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Partial();
        $.subGraphIndex = reader.value(json.subGraphIndex, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListFromTensor = class TensorListFromTensor {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListFromTensor();
        $.elementDType = reader.int32_(position, 4, 0);
        $.shapeType = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListFromTensor();
        $.elementDType = reader.value(json.elementDType, 0);
        $.shapeType = reader.value(json.shapeType, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListStack = class TensorListStack {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListStack();
        $.numElements = reader.int32_(position, 4, 0);
        $.elementDType = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListStack();
        $.numElements = reader.value(json.numElements, 0);
        $.elementDType = reader.value(json.elementDType, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListGetItem = class TensorListGetItem {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListGetItem();
        $.elementDType = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListGetItem();
        $.elementDType = reader.value(json.elementDType, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListSetItem = class TensorListSetItem {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListSetItem();
        $.elementDType = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListSetItem();
        $.elementDType = reader.value(json.elementDType, 0);
        return $;
    }
};

$root.mindspore.schema.TensorListReserve = class TensorListReserve {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.TensorListReserve();
        $.elementDType = reader.int32_(position, 4, 0);
        $.shapeType = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.TensorListReserve();
        $.elementDType = reader.value(json.elementDType, 0);
        $.shapeType = reader.value(json.shapeType, 0);
        return $;
    }
};

$root.mindspore.schema.All = class All {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.All();
        $.keepDims = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.All();
        $.keepDims = reader.value(json.keepDims, 0);
        return $;
    }
};

$root.mindspore.schema.Assert = class Assert {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.Assert();
        $.summarize = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.Assert();
        $.summarize = reader.value(json.summarize, 0);
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

$root.mindspore.schema.Merge = class Merge {

    static decode(/* reader, position */) {
        const $ = new $root.mindspore.schema.Merge();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new $root.mindspore.schema.Merge();
        return $;
    }
};

$root.mindspore.schema.GeLU = class GeLU {

    static decode(reader, position) {
        const $ = new $root.mindspore.schema.GeLU();
        $.approximate = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.mindspore.schema.GeLU();
        $.approximate = reader.value(json.approximate, false);
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
        $.seed = reader.int32_(position, 4, 0);
        $.seed2 = reader.int32_(position, 6, 0);
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
