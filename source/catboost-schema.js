
export const NCatBoostFbs = {};

NCatBoostFbs.TGuid = class TGuid {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TGuid();
        $.dw0 = reader.uint32(position + 0);
        $.dw1 = reader.uint32(position + 4);
        $.dw2 = reader.uint32(position + 8);
        $.dw3 = reader.uint32(position + 12);
        return $;
    }
};

NCatBoostFbs.ENanValueTreatment = {
    AsIs: 0, '0': 'AsIs',
    AsFalse: 1, '1': 'AsFalse',
    AsTrue: 2, '2': 'AsTrue'
};

NCatBoostFbs.TFloatFeature = class TFloatFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TFloatFeature();
        $.HasNans = reader.bool_(position, 4, false);
        $.Index = reader.int32_(position, 6, -1);
        $.FlatIndex = reader.int32_(position, 8, -1);
        $.Borders = reader.array(position, 10, Float32Array);
        $.FeatureId = reader.string_(position, 12, null);
        $.NanValueTreatment = reader.int8_(position, 14, 0);
        return $;
    }
};

NCatBoostFbs.TCatFeature = class TCatFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TCatFeature();
        $.Index = reader.int32_(position, 4, -1);
        $.FlatIndex = reader.int32_(position, 6, -1);
        $.FeatureId = reader.string_(position, 8, null);
        $.UsedInModel = reader.bool_(position, 10, true);
        return $;
    }
};

NCatBoostFbs.TTextFeature = class TTextFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TTextFeature();
        $.Index = reader.int32_(position, 4, -1);
        $.FlatIndex = reader.int32_(position, 6, -1);
        $.FeatureId = reader.string_(position, 8, null);
        $.UsedInModel = reader.bool_(position, 10, true);
        return $;
    }
};

NCatBoostFbs.TEmbeddingFeature = class TEmbeddingFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TEmbeddingFeature();
        $.Index = reader.int32_(position, 4, -1);
        $.FlatIndex = reader.int32_(position, 6, -1);
        $.FeatureId = reader.string_(position, 8, null);
        $.Dimension = reader.int32_(position, 10, 0);
        $.UsedInModel = reader.bool_(position, 12, true);
        return $;
    }
};

NCatBoostFbs.ESourceFeatureType = {
    Text: 0, '0': 'Text',
    Embedding: 1, '1': 'Embedding'
};

NCatBoostFbs.TEstimatedFeature = class TEstimatedFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TEstimatedFeature();
        $.SourceFeatureIndex = reader.int32_(position, 4, -1);
        $.CalcerId = reader.struct(position, 6, NCatBoostFbs.TGuid);
        $.LocalIndex = reader.int32_(position, 8, -1);
        $.Borders = reader.array(position, 10, Float32Array);
        $.SourceFeatureType = reader.int8_(position, 12, 0);
        return $;
    }
};

NCatBoostFbs.TOneHotFeature = class TOneHotFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TOneHotFeature();
        $.Index = reader.int32_(position, 4, -1);
        $.Values = reader.array(position, 6, Int32Array);
        $.StringValues = reader.strings_(position, 8);
        return $;
    }
};

NCatBoostFbs.TFloatSplit = class TFloatSplit {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TFloatSplit();
        $.Index = reader.int32(position + 0);
        $.Border = reader.float32(position + 4);
        return $;
    }
};

NCatBoostFbs.TOneHotSplit = class TOneHotSplit {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TOneHotSplit();
        $.Index = reader.int32(position + 0);
        $.Value = reader.int32(position + 4);
        return $;
    }
};

NCatBoostFbs.TFeatureCombination = class TFeatureCombination {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TFeatureCombination();
        $.CatFeatures = reader.array(position, 4, Int32Array);
        $.FloatSplits = reader.structs(position, 6, NCatBoostFbs.TFloatSplit, 8);
        $.OneHotSplits = reader.structs(position, 8, NCatBoostFbs.TOneHotSplit, 8);
        return $;
    }
};

NCatBoostFbs.ECtrType = {
    Borders: 0, '0': 'Borders',
    Buckets: 1, '1': 'Buckets',
    BinarizedTargetMeanValue: 2, '2': 'BinarizedTargetMeanValue',
    FloatTargetMeanValue: 3, '3': 'FloatTargetMeanValue',
    Counter: 4, '4': 'Counter',
    FeatureFreq: 5, '5': 'FeatureFreq'
};

NCatBoostFbs.TModelCtrBase = class TModelCtrBase {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TModelCtrBase();
        $.FeatureCombination = reader.table(position, 4, NCatBoostFbs.TFeatureCombination);
        $.CtrType = reader.int8_(position, 6, 0);
        $.TargetBorderClassifierIdx = reader.int32_(position, 8, 0);
        return $;
    }
};

NCatBoostFbs.TModelCtr = class TModelCtr {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TModelCtr();
        $.Base = reader.table(position, 4, NCatBoostFbs.TModelCtrBase);
        $.TargetBorderIdx = reader.int32_(position, 6, 0);
        $.PriorNum = reader.float32_(position, 8, 0);
        $.PriorDenom = reader.float32_(position, 10, 1);
        $.Shift = reader.float32_(position, 12, 0);
        $.Scale = reader.float32_(position, 14, 1);
        return $;
    }
};

NCatBoostFbs.TCtrFeature = class TCtrFeature {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TCtrFeature();
        $.Ctr = reader.table(position, 4, NCatBoostFbs.TModelCtr);
        $.Borders = reader.array(position, 6, Float32Array);
        return $;
    }
};

NCatBoostFbs.TCtrValueTable = class TCtrValueTable {

    static create(reader) {
        return NCatBoostFbs.TCtrValueTable.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TCtrValueTable();
        $.ModelCtrBase = reader.table(position, 4, NCatBoostFbs.TModelCtrBase);
        $.IndexHashRaw = reader.array(position, 6, Uint8Array);
        $.CTRBlob = reader.array(position, 8, Uint8Array);
        $.CounterDenominator = reader.int32_(position, 10, 0);
        $.TargetClassesCount = reader.int32_(position, 12, 0);
        return $;
    }
};

NCatBoostFbs.TKeyValue = class TKeyValue {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TKeyValue();
        $.Key = reader.string_(position, 4, null);
        $.Value = reader.string_(position, 6, null);
        return $;
    }
};

NCatBoostFbs.TNonSymmetricTreeStepNode = class TNonSymmetricTreeStepNode {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TNonSymmetricTreeStepNode();
        $.LeftSubtreeDiff = reader.uint16(position + 0);
        $.RightSubtreeDiff = reader.uint16(position + 2);
        return $;
    }
};

NCatBoostFbs.TRepackedBin = class TRepackedBin {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TRepackedBin();
        $.FeatureIndex = reader.uint16(position + 0);
        $.XorMask = reader.uint8(position + 2);
        $.SplitIdx = reader.uint8(position + 3);
        return $;
    }
};

NCatBoostFbs.TModelTrees = class TModelTrees {

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TModelTrees();
        $.ApproxDimension = reader.int32_(position, 4, 0);
        $.TreeSplits = reader.array(position, 6, Int32Array);
        $.TreeSizes = reader.array(position, 8, Int32Array);
        $.TreeStartOffsets = reader.array(position, 10, Int32Array);
        $.CatFeatures = reader.tables(position, 12, NCatBoostFbs.TCatFeature);
        $.FloatFeatures = reader.tables(position, 14, NCatBoostFbs.TFloatFeature);
        $.OneHotFeatures = reader.tables(position, 16, NCatBoostFbs.TOneHotFeature);
        $.CtrFeatures = reader.tables(position, 18, NCatBoostFbs.TCtrFeature);
        $.LeafValues = reader.array(position, 20, Float64Array);
        $.LeafWeights = reader.array(position, 22, Float64Array);
        $.NonSymmetricStepNodes = reader.structs(position, 24, NCatBoostFbs.TNonSymmetricTreeStepNode, 4);
        $.NonSymmetricNodeIdToLeafId = reader.array(position, 26, Uint32Array);
        $.TextFeatures = reader.tables(position, 28, NCatBoostFbs.TTextFeature);
        $.EstimatedFeatures = reader.tables(position, 30, NCatBoostFbs.TEstimatedFeature);
        $.Scale = reader.float64_(position, 32, 1);
        $.Bias = reader.float64_(position, 34, 0);
        $.MultiBias = reader.array(position, 36, Float64Array);
        $.RepackedBins = reader.structs(position, 38, NCatBoostFbs.TRepackedBin, 4);
        $.EmbeddingFeatures = reader.tables(position, 40, NCatBoostFbs.TEmbeddingFeature);
        return $;
    }
};

NCatBoostFbs.TModelCore = class TModelCore {

    static create(reader) {
        return NCatBoostFbs.TModelCore.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new NCatBoostFbs.TModelCore();
        $.FormatVersion = reader.string_(position, 4, null);
        $.ModelTrees = reader.table(position, 6, NCatBoostFbs.TModelTrees);
        $.InfoMap = reader.tables(position, 8, NCatBoostFbs.TKeyValue);
        $.ModelPartIds = reader.strings_(position, 10);
        return $;
    }
};
