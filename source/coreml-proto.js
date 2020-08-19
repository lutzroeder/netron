var $root = protobuf.get('coreml');

$root.CoreML = {};

$root.CoreML.Specification = {};

$root.CoreML.Specification.Pipeline = class Pipeline {

    constructor() {
        this.models = [];
        this.names = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Pipeline();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.models.push($root.CoreML.Specification.Model.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.names.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PipelineClassifier = class PipelineClassifier {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PipelineClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pipeline = $root.CoreML.Specification.Pipeline.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PipelineClassifier.prototype.pipeline = null;

$root.CoreML.Specification.PipelineRegressor = class PipelineRegressor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PipelineRegressor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pipeline = $root.CoreML.Specification.Pipeline.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PipelineRegressor.prototype.pipeline = null;

$root.CoreML.Specification.FeatureDescription = class FeatureDescription {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FeatureDescription();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shortDescription = reader.string();
                    break;
                case 3:
                    message.type = $root.CoreML.Specification.FeatureType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FeatureDescription.prototype.name = "";
$root.CoreML.Specification.FeatureDescription.prototype.shortDescription = "";
$root.CoreML.Specification.FeatureDescription.prototype.type = null;

$root.CoreML.Specification.Metadata = class Metadata {

    constructor() {
        this.userDefined = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Metadata();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shortDescription = reader.string();
                    break;
                case 2:
                    message.versionString = reader.string();
                    break;
                case 3:
                    message.author = reader.string();
                    break;
                case 4:
                    message.license = reader.string();
                    break;
                case 100:
                    reader.entry(message.userDefined, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Metadata.prototype.shortDescription = "";
$root.CoreML.Specification.Metadata.prototype.versionString = "";
$root.CoreML.Specification.Metadata.prototype.author = "";
$root.CoreML.Specification.Metadata.prototype.license = "";

$root.CoreML.Specification.ModelDescription = class ModelDescription {

    constructor() {
        this.input = [];
        this.output = [];
        this.trainingInput = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ModelDescription();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input.push($root.CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.output.push($root.CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.predictedFeatureName = reader.string();
                    break;
                case 12:
                    message.predictedProbabilitiesName = reader.string();
                    break;
                case 50:
                    message.trainingInput.push($root.CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.metadata = $root.CoreML.Specification.Metadata.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ModelDescription.prototype.predictedFeatureName = "";
$root.CoreML.Specification.ModelDescription.prototype.predictedProbabilitiesName = "";
$root.CoreML.Specification.ModelDescription.prototype.metadata = null;

$root.CoreML.Specification.SerializedModel = class SerializedModel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SerializedModel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.identifier = reader.string();
                    break;
                case 2:
                    message.model = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SerializedModel.prototype.identifier = "";
$root.CoreML.Specification.SerializedModel.prototype.model = new Uint8Array([]);

$root.CoreML.Specification.Model = class Model {

    constructor() {
    }

    get Type() {
        $root.CoreML.Specification.Model.TypeSet = $root.CoreML.Specification.Model.TypeSet || new Set([ "pipelineClassifier", "pipelineRegressor", "pipeline", "glmRegressor", "supportVectorRegressor", "treeEnsembleRegressor", "neuralNetworkRegressor", "bayesianProbitRegressor", "glmClassifier", "supportVectorClassifier", "treeEnsembleClassifier", "neuralNetworkClassifier", "kNearestNeighborsClassifier", "neuralNetwork", "itemSimilarityRecommender", "customModel", "linkedModel", "oneHotEncoder", "imputer", "featureVectorizer", "dictVectorizer", "scaler", "categoricalMapping", "normalizer", "arrayFeatureExtractor", "nonMaximumSuppression", "identity", "textClassifier", "wordTagger", "visionFeaturePrint", "soundAnalysisPreprocessing", "gazetteer", "wordEmbedding", "serializedModel"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Model.TypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Model();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.specificationVersion = reader.int32();
                    break;
                case 2:
                    message.description = $root.CoreML.Specification.ModelDescription.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.isUpdatable = reader.bool();
                    break;
                case 200:
                    message.pipelineClassifier = $root.CoreML.Specification.PipelineClassifier.decode(reader, reader.uint32());
                    break;
                case 201:
                    message.pipelineRegressor = $root.CoreML.Specification.PipelineRegressor.decode(reader, reader.uint32());
                    break;
                case 202:
                    message.pipeline = $root.CoreML.Specification.Pipeline.decode(reader, reader.uint32());
                    break;
                case 300:
                    message.glmRegressor = $root.CoreML.Specification.GLMRegressor.decode(reader, reader.uint32());
                    break;
                case 301:
                    message.supportVectorRegressor = $root.CoreML.Specification.SupportVectorRegressor.decode(reader, reader.uint32());
                    break;
                case 302:
                    message.treeEnsembleRegressor = $root.CoreML.Specification.TreeEnsembleRegressor.decode(reader, reader.uint32());
                    break;
                case 303:
                    message.neuralNetworkRegressor = $root.CoreML.Specification.NeuralNetworkRegressor.decode(reader, reader.uint32());
                    break;
                case 304:
                    message.bayesianProbitRegressor = $root.CoreML.Specification.BayesianProbitRegressor.decode(reader, reader.uint32());
                    break;
                case 400:
                    message.glmClassifier = $root.CoreML.Specification.GLMClassifier.decode(reader, reader.uint32());
                    break;
                case 401:
                    message.supportVectorClassifier = $root.CoreML.Specification.SupportVectorClassifier.decode(reader, reader.uint32());
                    break;
                case 402:
                    message.treeEnsembleClassifier = $root.CoreML.Specification.TreeEnsembleClassifier.decode(reader, reader.uint32());
                    break;
                case 403:
                    message.neuralNetworkClassifier = $root.CoreML.Specification.NeuralNetworkClassifier.decode(reader, reader.uint32());
                    break;
                case 404:
                    message.kNearestNeighborsClassifier = $root.CoreML.Specification.KNearestNeighborsClassifier.decode(reader, reader.uint32());
                    break;
                case 500:
                    message.neuralNetwork = $root.CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                case 501:
                    message.itemSimilarityRecommender = $root.CoreML.Specification.ItemSimilarityRecommender.decode(reader, reader.uint32());
                    break;
                case 555:
                    message.customModel = $root.CoreML.Specification.CustomModel.decode(reader, reader.uint32());
                    break;
                case 556:
                    message.linkedModel = $root.CoreML.Specification.LinkedModel.decode(reader, reader.uint32());
                    break;
                case 600:
                    message.oneHotEncoder = $root.CoreML.Specification.OneHotEncoder.decode(reader, reader.uint32());
                    break;
                case 601:
                    message.imputer = $root.CoreML.Specification.Imputer.decode(reader, reader.uint32());
                    break;
                case 602:
                    message.featureVectorizer = $root.CoreML.Specification.FeatureVectorizer.decode(reader, reader.uint32());
                    break;
                case 603:
                    message.dictVectorizer = $root.CoreML.Specification.DictVectorizer.decode(reader, reader.uint32());
                    break;
                case 604:
                    message.scaler = $root.CoreML.Specification.Scaler.decode(reader, reader.uint32());
                    break;
                case 606:
                    message.categoricalMapping = $root.CoreML.Specification.CategoricalMapping.decode(reader, reader.uint32());
                    break;
                case 607:
                    message.normalizer = $root.CoreML.Specification.Normalizer.decode(reader, reader.uint32());
                    break;
                case 609:
                    message.arrayFeatureExtractor = $root.CoreML.Specification.ArrayFeatureExtractor.decode(reader, reader.uint32());
                    break;
                case 610:
                    message.nonMaximumSuppression = $root.CoreML.Specification.NonMaximumSuppression.decode(reader, reader.uint32());
                    break;
                case 900:
                    message.identity = $root.CoreML.Specification.Identity.decode(reader, reader.uint32());
                    break;
                case 2000:
                    message.textClassifier = $root.CoreML.Specification.CoreMLModels.TextClassifier.decode(reader, reader.uint32());
                    break;
                case 2001:
                    message.wordTagger = $root.CoreML.Specification.CoreMLModels.WordTagger.decode(reader, reader.uint32());
                    break;
                case 2002:
                    message.visionFeaturePrint = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.decode(reader, reader.uint32());
                    break;
                case 2003:
                    message.soundAnalysisPreprocessing = $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.decode(reader, reader.uint32());
                    break;
                case 2004:
                    message.gazetteer = $root.CoreML.Specification.CoreMLModels.Gazetteer.decode(reader, reader.uint32());
                    break;
                case 2005:
                    message.wordEmbedding = $root.CoreML.Specification.CoreMLModels.WordEmbedding.decode(reader, reader.uint32());
                    break;
                case 3000:
                    message.serializedModel = $root.CoreML.Specification.SerializedModel.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Model.prototype.specificationVersion = 0;
$root.CoreML.Specification.Model.prototype.description = null;
$root.CoreML.Specification.Model.prototype.isUpdatable = false;

$root.CoreML.Specification.CoreMLModels = {};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint = class VisionFeaturePrint {

    constructor() {
    }

    get VisionFeaturePrintType() {
        $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.VisionFeaturePrintTypeSet = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.VisionFeaturePrintTypeSet || new Set([ "scene", "objects"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.VisionFeaturePrintTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.scene = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.objects = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene = class Scene {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.prototype.version = 0;

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion = {
    "SCENE_VERSION_INVALID": 0,
    "SCENE_VERSION_1": 1
};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects = class Objects {

    constructor() {
        this.output = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int32();
                    break;
                case 100:
                    message.output.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.prototype.version = 0;

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.ObjectsVersion = {
    "OBJECTS_VERSION_INVALID": 0,
    "OBJECTS_VERSION_1": 1
};

$root.CoreML.Specification.CoreMLModels.TextClassifier = class TextClassifier {

    constructor() {
    }

    get ClassLabels() {
        $root.CoreML.Specification.CoreMLModels.TextClassifier.ClassLabelsSet = $root.CoreML.Specification.CoreMLModels.TextClassifier.ClassLabelsSet || new Set([ "stringClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CoreMLModels.TextClassifier.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.TextClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.revision = reader.uint32();
                    break;
                case 10:
                    message.language = reader.string();
                    break;
                case 100:
                    message.modelParameterData = reader.bytes();
                    break;
                case 200:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.TextClassifier.prototype.revision = 0;
$root.CoreML.Specification.CoreMLModels.TextClassifier.prototype.language = "";
$root.CoreML.Specification.CoreMLModels.TextClassifier.prototype.modelParameterData = new Uint8Array([]);

$root.CoreML.Specification.CoreMLModels.WordTagger = class WordTagger {

    constructor() {
    }

    get Tags() {
        $root.CoreML.Specification.CoreMLModels.WordTagger.TagsSet = $root.CoreML.Specification.CoreMLModels.WordTagger.TagsSet || new Set([ "stringTags"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CoreMLModels.WordTagger.TagsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.WordTagger();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.revision = reader.uint32();
                    break;
                case 10:
                    message.language = reader.string();
                    break;
                case 20:
                    message.tokensOutputFeatureName = reader.string();
                    break;
                case 21:
                    message.tokenTagsOutputFeatureName = reader.string();
                    break;
                case 22:
                    message.tokenLocationsOutputFeatureName = reader.string();
                    break;
                case 23:
                    message.tokenLengthsOutputFeatureName = reader.string();
                    break;
                case 100:
                    message.modelParameterData = reader.bytes();
                    break;
                case 200:
                    message.stringTags = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.revision = 0;
$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.language = "";
$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.tokensOutputFeatureName = "";
$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.tokenTagsOutputFeatureName = "";
$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.tokenLocationsOutputFeatureName = "";
$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.tokenLengthsOutputFeatureName = "";
$root.CoreML.Specification.CoreMLModels.WordTagger.prototype.modelParameterData = new Uint8Array([]);

$root.CoreML.Specification.CoreMLModels.Gazetteer = class Gazetteer {

    constructor() {
    }

    get ClassLabels() {
        $root.CoreML.Specification.CoreMLModels.Gazetteer.ClassLabelsSet = $root.CoreML.Specification.CoreMLModels.Gazetteer.ClassLabelsSet || new Set([ "stringClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CoreMLModels.Gazetteer.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.Gazetteer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.revision = reader.uint32();
                    break;
                case 10:
                    message.language = reader.string();
                    break;
                case 100:
                    message.modelParameterData = reader.bytes();
                    break;
                case 200:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.Gazetteer.prototype.revision = 0;
$root.CoreML.Specification.CoreMLModels.Gazetteer.prototype.language = "";
$root.CoreML.Specification.CoreMLModels.Gazetteer.prototype.modelParameterData = new Uint8Array([]);

$root.CoreML.Specification.CoreMLModels.WordEmbedding = class WordEmbedding {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.WordEmbedding();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.revision = reader.uint32();
                    break;
                case 10:
                    message.language = reader.string();
                    break;
                case 100:
                    message.modelParameterData = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.WordEmbedding.prototype.revision = 0;
$root.CoreML.Specification.CoreMLModels.WordEmbedding.prototype.language = "";
$root.CoreML.Specification.CoreMLModels.WordEmbedding.prototype.modelParameterData = new Uint8Array([]);

$root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing = class SoundAnalysisPreprocessing {

    constructor() {
    }

    get SoundAnalysisPreprocessingType() {
        $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.SoundAnalysisPreprocessingTypeSet = $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.SoundAnalysisPreprocessingTypeSet || new Set([ "vggish"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.SoundAnalysisPreprocessingTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.vggish = $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish = class Vggish {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.StringToInt64Map = class StringToInt64Map {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringToInt64Map();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.map, () => reader.string(), () => reader.int64());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64ToStringMap = class Int64ToStringMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64ToStringMap();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.map, () => reader.int64(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.StringToDoubleMap = class StringToDoubleMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringToDoubleMap();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.map, () => reader.string(), () => reader.double());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64ToDoubleMap = class Int64ToDoubleMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64ToDoubleMap();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.map, () => reader.int64(), () => reader.double());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.StringVector = class StringVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringVector();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vector.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64Vector = class Int64Vector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Vector();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vector = reader.array(message.vector, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FloatVector = class FloatVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FloatVector();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vector = reader.floats(message.vector, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DoubleVector = class DoubleVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleVector();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vector = reader.doubles(message.vector, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64Range = class Int64Range {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Range();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.minValue = reader.int64();
                    break;
                case 2:
                    message.maxValue = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64Range.prototype.minValue = protobuf.Int64.create(0);
$root.CoreML.Specification.Int64Range.prototype.maxValue = protobuf.Int64.create(0);

$root.CoreML.Specification.Int64Set = class Int64Set {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Set();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = reader.array(message.values, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DoubleRange = class DoubleRange {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleRange();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.minValue = reader.double();
                    break;
                case 2:
                    message.maxValue = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DoubleRange.prototype.minValue = 0;
$root.CoreML.Specification.DoubleRange.prototype.maxValue = 0;

$root.CoreML.Specification.Int64FeatureType = class Int64FeatureType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64FeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DoubleFeatureType = class DoubleFeatureType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleFeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.StringFeatureType = class StringFeatureType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringFeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SizeRange = class SizeRange {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SizeRange();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lowerBound = reader.uint64();
                    break;
                case 2:
                    message.upperBound = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SizeRange.prototype.lowerBound = protobuf.Uint64.create(0);
$root.CoreML.Specification.SizeRange.prototype.upperBound = protobuf.Int64.create(0);

$root.CoreML.Specification.ImageFeatureType = class ImageFeatureType {

    constructor() {
    }

    get SizeFlexibility() {
        $root.CoreML.Specification.ImageFeatureType.SizeFlexibilitySet = $root.CoreML.Specification.ImageFeatureType.SizeFlexibilitySet || new Set([ "enumeratedSizes", "imageSizeRange"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.ImageFeatureType.SizeFlexibilitySet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ImageFeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.width = reader.int64();
                    break;
                case 2:
                    message.height = reader.int64();
                    break;
                case 21:
                    message.enumeratedSizes = $root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.imageSizeRange = $root.CoreML.Specification.ImageFeatureType.ImageSizeRange.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.colorSpace = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ImageFeatureType.prototype.width = protobuf.Int64.create(0);
$root.CoreML.Specification.ImageFeatureType.prototype.height = protobuf.Int64.create(0);
$root.CoreML.Specification.ImageFeatureType.prototype.colorSpace = 0;

$root.CoreML.Specification.ImageFeatureType.ColorSpace = {
    "INVALID_COLOR_SPACE": 0,
    "GRAYSCALE": 10,
    "RGB": 20,
    "BGR": 30
};

$root.CoreML.Specification.ImageFeatureType.ImageSize = class ImageSize {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ImageFeatureType.ImageSize();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.width = reader.uint64();
                    break;
                case 2:
                    message.height = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ImageFeatureType.ImageSize.prototype.width = protobuf.Uint64.create(0);
$root.CoreML.Specification.ImageFeatureType.ImageSize.prototype.height = protobuf.Uint64.create(0);

$root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes = class EnumeratedImageSizes {

    constructor() {
        this.sizes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sizes.push($root.CoreML.Specification.ImageFeatureType.ImageSize.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ImageFeatureType.ImageSizeRange = class ImageSizeRange {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ImageFeatureType.ImageSizeRange();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.widthRange = $root.CoreML.Specification.SizeRange.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.heightRange = $root.CoreML.Specification.SizeRange.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ImageFeatureType.ImageSizeRange.prototype.widthRange = null;
$root.CoreML.Specification.ImageFeatureType.ImageSizeRange.prototype.heightRange = null;

$root.CoreML.Specification.ArrayFeatureType = class ArrayFeatureType {

    constructor() {
        this.shape = [];
    }

    get ShapeFlexibility() {
        $root.CoreML.Specification.ArrayFeatureType.ShapeFlexibilitySet = $root.CoreML.Specification.ArrayFeatureType.ShapeFlexibilitySet || new Set([ "enumeratedShapes", "shapeRange"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.ArrayFeatureType.ShapeFlexibilitySet.has(key) && this[key] != null);
    }

    get defaultOptionalValue() {
        $root.CoreML.Specification.ArrayFeatureType.defaultOptionalValueSet = $root.CoreML.Specification.ArrayFeatureType.defaultOptionalValueSet || new Set([ "intDefaultValue", "floatDefaultValue", "doubleDefaultValue"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.ArrayFeatureType.defaultOptionalValueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.int64(), tag);
                    break;
                case 2:
                    message.dataType = reader.int32();
                    break;
                case 21:
                    message.enumeratedShapes = $root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.shapeRange = $root.CoreML.Specification.ArrayFeatureType.ShapeRange.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.intDefaultValue = reader.int32();
                    break;
                case 51:
                    message.floatDefaultValue = reader.float();
                    break;
                case 61:
                    message.doubleDefaultValue = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ArrayFeatureType.prototype.dataType = 0;

$root.CoreML.Specification.ArrayFeatureType.ArrayDataType = {
    "INVALID_ARRAY_DATA_TYPE": 0,
    "FLOAT32": 65568,
    "DOUBLE": 65600,
    "INT32": 131104
};

$root.CoreML.Specification.ArrayFeatureType.Shape = class Shape {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.Shape();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes = class EnumeratedShapes {

    constructor() {
        this.shapes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shapes.push($root.CoreML.Specification.ArrayFeatureType.Shape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ArrayFeatureType.ShapeRange = class ShapeRange {

    constructor() {
        this.sizeRanges = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.ShapeRange();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sizeRanges.push($root.CoreML.Specification.SizeRange.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DictionaryFeatureType = class DictionaryFeatureType {

    constructor() {
    }

    get KeyType() {
        $root.CoreML.Specification.DictionaryFeatureType.KeyTypeSet = $root.CoreML.Specification.DictionaryFeatureType.KeyTypeSet || new Set([ "int64KeyType", "stringKeyType"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.DictionaryFeatureType.KeyTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DictionaryFeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.int64KeyType = $root.CoreML.Specification.Int64FeatureType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stringKeyType = $root.CoreML.Specification.StringFeatureType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SequenceFeatureType = class SequenceFeatureType {

    constructor() {
    }

    get Type() {
        $root.CoreML.Specification.SequenceFeatureType.TypeSet = $root.CoreML.Specification.SequenceFeatureType.TypeSet || new Set([ "int64Type", "stringType"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.SequenceFeatureType.TypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SequenceFeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.int64Type = $root.CoreML.Specification.Int64FeatureType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stringType = $root.CoreML.Specification.StringFeatureType.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.sizeRange = $root.CoreML.Specification.SizeRange.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SequenceFeatureType.prototype.sizeRange = null;

$root.CoreML.Specification.FeatureType = class FeatureType {

    constructor() {
    }

    get Type() {
        $root.CoreML.Specification.FeatureType.TypeSet = $root.CoreML.Specification.FeatureType.TypeSet || new Set([ "int64Type", "doubleType", "stringType", "imageType", "multiArrayType", "dictionaryType", "sequenceType"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.FeatureType.TypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FeatureType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.int64Type = $root.CoreML.Specification.Int64FeatureType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.doubleType = $root.CoreML.Specification.DoubleFeatureType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stringType = $root.CoreML.Specification.StringFeatureType.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.imageType = $root.CoreML.Specification.ImageFeatureType.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.multiArrayType = $root.CoreML.Specification.ArrayFeatureType.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.dictionaryType = $root.CoreML.Specification.DictionaryFeatureType.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.sequenceType = $root.CoreML.Specification.SequenceFeatureType.decode(reader, reader.uint32());
                    break;
                case 1000:
                    message.isOptional = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FeatureType.prototype.isOptional = false;

$root.CoreML.Specification.ArrayFeatureExtractor = class ArrayFeatureExtractor {

    constructor() {
        this.extractIndex = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureExtractor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.extractIndex = reader.array(message.extractIndex, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BayesianProbitRegressor = class BayesianProbitRegressor {

    constructor() {
        this.features = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numberOfFeatures = reader.uint32();
                    break;
                case 2:
                    message.bias = $root.CoreML.Specification.BayesianProbitRegressor.Gaussian.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.features.push($root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.regressionInputFeatureName = reader.string();
                    break;
                case 11:
                    message.optimismInputFeatureName = reader.string();
                    break;
                case 12:
                    message.samplingScaleInputFeatureName = reader.string();
                    break;
                case 13:
                    message.samplingTruncationInputFeatureName = reader.string();
                    break;
                case 20:
                    message.meanOutputFeatureName = reader.string();
                    break;
                case 21:
                    message.varianceOutputFeatureName = reader.string();
                    break;
                case 22:
                    message.pessimisticProbabilityOutputFeatureName = reader.string();
                    break;
                case 23:
                    message.sampledProbabilityOutputFeatureName = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BayesianProbitRegressor.prototype.numberOfFeatures = 0;
$root.CoreML.Specification.BayesianProbitRegressor.prototype.bias = null;
$root.CoreML.Specification.BayesianProbitRegressor.prototype.regressionInputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.optimismInputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.samplingScaleInputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.samplingTruncationInputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.meanOutputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.varianceOutputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.pessimisticProbabilityOutputFeatureName = "";
$root.CoreML.Specification.BayesianProbitRegressor.prototype.sampledProbabilityOutputFeatureName = "";

$root.CoreML.Specification.BayesianProbitRegressor.Gaussian = class Gaussian {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.Gaussian();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mean = reader.double();
                    break;
                case 2:
                    message.precision = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BayesianProbitRegressor.Gaussian.prototype.mean = 0;
$root.CoreML.Specification.BayesianProbitRegressor.Gaussian.prototype.precision = 0;

$root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight = class FeatureValueWeight {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.featureValue = reader.uint32();
                    break;
                case 2:
                    message.featureWeight = $root.CoreML.Specification.BayesianProbitRegressor.Gaussian.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.prototype.featureValue = 0;
$root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.prototype.featureWeight = null;

$root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight = class FeatureWeight {

    constructor() {
        this.weights = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.featureId = reader.uint32();
                    break;
                case 2:
                    message.weights.push($root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight.prototype.featureId = 0;

$root.CoreML.Specification.CategoricalMapping = class CategoricalMapping {

    constructor() {
    }

    get MappingType() {
        $root.CoreML.Specification.CategoricalMapping.MappingTypeSet = $root.CoreML.Specification.CategoricalMapping.MappingTypeSet || new Set([ "stringToInt64Map", "int64ToStringMap"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CategoricalMapping.MappingTypeSet.has(key) && this[key] != null);
    }

    get ValueOnUnknown() {
        $root.CoreML.Specification.CategoricalMapping.ValueOnUnknownSet = $root.CoreML.Specification.CategoricalMapping.ValueOnUnknownSet || new Set([ "strValue", "int64Value"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CategoricalMapping.ValueOnUnknownSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CategoricalMapping();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.stringToInt64Map = $root.CoreML.Specification.StringToInt64Map.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.int64ToStringMap = $root.CoreML.Specification.Int64ToStringMap.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.strValue = reader.string();
                    break;
                case 102:
                    message.int64Value = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CustomModel = class CustomModel {

    constructor() {
        this.parameters = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CustomModel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.className = reader.string();
                    break;
                case 30:
                    reader.entry(message.parameters, () => reader.string(), () => $root.CoreML.Specification.CustomModel.CustomModelParamValue.decode(reader, reader.uint32()));
                    break;
                case 40:
                    message.description = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CustomModel.prototype.className = "";
$root.CoreML.Specification.CustomModel.prototype.description = "";

$root.CoreML.Specification.CustomModel.CustomModelParamValue = class CustomModelParamValue {

    constructor() {
    }

    get value() {
        $root.CoreML.Specification.CustomModel.CustomModelParamValue.valueSet = $root.CoreML.Specification.CustomModel.CustomModelParamValue.valueSet || new Set([ "doubleValue", "stringValue", "intValue", "longValue", "boolValue", "bytesValue"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CustomModel.CustomModelParamValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CustomModel.CustomModelParamValue();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.doubleValue = reader.double();
                    break;
                case 20:
                    message.stringValue = reader.string();
                    break;
                case 30:
                    message.intValue = reader.int32();
                    break;
                case 40:
                    message.longValue = reader.int64();
                    break;
                case 50:
                    message.boolValue = reader.bool();
                    break;
                case 60:
                    message.bytesValue = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DictVectorizer = class DictVectorizer {

    constructor() {
    }

    get Map() {
        $root.CoreML.Specification.DictVectorizer.MapSet = $root.CoreML.Specification.DictVectorizer.MapSet || new Set([ "stringToIndex", "int64ToIndex"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.DictVectorizer.MapSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DictVectorizer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.stringToIndex = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.int64ToIndex = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FeatureVectorizer = class FeatureVectorizer {

    constructor() {
        this.inputList = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FeatureVectorizer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputList.push($root.CoreML.Specification.FeatureVectorizer.InputColumn.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FeatureVectorizer.InputColumn = class InputColumn {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FeatureVectorizer.InputColumn();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputColumn = reader.string();
                    break;
                case 2:
                    message.inputDimensions = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FeatureVectorizer.InputColumn.prototype.inputColumn = "";
$root.CoreML.Specification.FeatureVectorizer.InputColumn.prototype.inputDimensions = protobuf.Uint64.create(0);

$root.CoreML.Specification.GLMRegressor = class GLMRegressor {

    constructor() {
        this.weights = [];
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GLMRegressor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.weights.push($root.CoreML.Specification.GLMRegressor.DoubleArray.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.offset = reader.doubles(message.offset, tag);
                    break;
                case 3:
                    message.postEvaluationTransform = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GLMRegressor.prototype.postEvaluationTransform = 0;

$root.CoreML.Specification.GLMRegressor.DoubleArray = class DoubleArray {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GLMRegressor.DoubleArray();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.doubles(message.value, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GLMRegressor.PostEvaluationTransform = {
    "NoTransform": 0,
    "Logit": 1,
    "Probit": 2
};

$root.CoreML.Specification.GLMClassifier = class GLMClassifier {

    constructor() {
        this.weights = [];
        this.offset = [];
    }

    get ClassLabels() {
        $root.CoreML.Specification.GLMClassifier.ClassLabelsSet = $root.CoreML.Specification.GLMClassifier.ClassLabelsSet || new Set([ "stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.GLMClassifier.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GLMClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.weights.push($root.CoreML.Specification.GLMClassifier.DoubleArray.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.offset = reader.doubles(message.offset, tag);
                    break;
                case 3:
                    message.postEvaluationTransform = reader.int32();
                    break;
                case 4:
                    message.classEncoding = reader.int32();
                    break;
                case 100:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GLMClassifier.prototype.postEvaluationTransform = 0;
$root.CoreML.Specification.GLMClassifier.prototype.classEncoding = 0;

$root.CoreML.Specification.GLMClassifier.DoubleArray = class DoubleArray {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GLMClassifier.DoubleArray();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.doubles(message.value, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GLMClassifier.PostEvaluationTransform = {
    "Logit": 0,
    "Probit": 1
};

$root.CoreML.Specification.GLMClassifier.ClassEncoding = {
    "ReferenceClass": 0,
    "OneVsRest": 1
};

$root.CoreML.Specification.KNearestNeighborsClassifier = class KNearestNeighborsClassifier {

    constructor() {
    }

    get ClassLabels() {
        $root.CoreML.Specification.KNearestNeighborsClassifier.ClassLabelsSet = $root.CoreML.Specification.KNearestNeighborsClassifier.ClassLabelsSet || new Set([ "stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.KNearestNeighborsClassifier.ClassLabelsSet.has(key) && this[key] != null);
    }

    get DefaultClassLabel() {
        $root.CoreML.Specification.KNearestNeighborsClassifier.DefaultClassLabelSet = $root.CoreML.Specification.KNearestNeighborsClassifier.DefaultClassLabelSet || new Set([ "defaultStringLabel", "defaultInt64Label"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.KNearestNeighborsClassifier.DefaultClassLabelSet.has(key) && this[key] != null);
    }

    get WeightingScheme() {
        $root.CoreML.Specification.KNearestNeighborsClassifier.WeightingSchemeSet = $root.CoreML.Specification.KNearestNeighborsClassifier.WeightingSchemeSet || new Set([ "uniformWeighting", "inverseDistanceWeighting"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.KNearestNeighborsClassifier.WeightingSchemeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.KNearestNeighborsClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nearestNeighborsIndex = $root.CoreML.Specification.NearestNeighborsIndex.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.numberOfNeighbors = $root.CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.defaultStringLabel = reader.string();
                    break;
                case 111:
                    message.defaultInt64Label = reader.int64();
                    break;
                case 200:
                    message.uniformWeighting = $root.CoreML.Specification.UniformWeighting.decode(reader, reader.uint32());
                    break;
                case 210:
                    message.inverseDistanceWeighting = $root.CoreML.Specification.InverseDistanceWeighting.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.KNearestNeighborsClassifier.prototype.nearestNeighborsIndex = null;
$root.CoreML.Specification.KNearestNeighborsClassifier.prototype.numberOfNeighbors = null;

$root.CoreML.Specification.NearestNeighborsIndex = class NearestNeighborsIndex {

    constructor() {
        this.floatSamples = [];
    }

    get IndexType() {
        $root.CoreML.Specification.NearestNeighborsIndex.IndexTypeSet = $root.CoreML.Specification.NearestNeighborsIndex.IndexTypeSet || new Set([ "linearIndex", "singleKdTreeIndex"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NearestNeighborsIndex.IndexTypeSet.has(key) && this[key] != null);
    }

    get DistanceFunction() {
        $root.CoreML.Specification.NearestNeighborsIndex.DistanceFunctionSet = $root.CoreML.Specification.NearestNeighborsIndex.DistanceFunctionSet || new Set([ "squaredEuclideanDistance"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NearestNeighborsIndex.DistanceFunctionSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NearestNeighborsIndex();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numberOfDimensions = reader.int32();
                    break;
                case 2:
                    message.floatSamples.push($root.CoreML.Specification.FloatVector.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.linearIndex = $root.CoreML.Specification.LinearIndex.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.singleKdTreeIndex = $root.CoreML.Specification.SingleKdTreeIndex.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.squaredEuclideanDistance = $root.CoreML.Specification.SquaredEuclideanDistance.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NearestNeighborsIndex.prototype.numberOfDimensions = 0;

$root.CoreML.Specification.UniformWeighting = class UniformWeighting {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UniformWeighting();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.InverseDistanceWeighting = class InverseDistanceWeighting {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.InverseDistanceWeighting();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LinearIndex = class LinearIndex {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinearIndex();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SingleKdTreeIndex = class SingleKdTreeIndex {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SingleKdTreeIndex();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.leafSize = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SingleKdTreeIndex.prototype.leafSize = 0;

$root.CoreML.Specification.SquaredEuclideanDistance = class SquaredEuclideanDistance {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SquaredEuclideanDistance();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64Parameter = class Int64Parameter {

    constructor() {
    }

    get AllowedValues() {
        $root.CoreML.Specification.Int64Parameter.AllowedValuesSet = $root.CoreML.Specification.Int64Parameter.AllowedValuesSet || new Set([ "range", "set"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Int64Parameter.AllowedValuesSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Parameter();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.defaultValue = reader.int64();
                    break;
                case 10:
                    message.range = $root.CoreML.Specification.Int64Range.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.set = $root.CoreML.Specification.Int64Set.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Int64Parameter.prototype.defaultValue = protobuf.Int64.create(0);

$root.CoreML.Specification.DoubleParameter = class DoubleParameter {

    constructor() {
    }

    get AllowedValues() {
        $root.CoreML.Specification.DoubleParameter.AllowedValuesSet = $root.CoreML.Specification.DoubleParameter.AllowedValuesSet || new Set([ "range"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.DoubleParameter.AllowedValuesSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleParameter();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.defaultValue = reader.double();
                    break;
                case 10:
                    message.range = $root.CoreML.Specification.DoubleRange.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DoubleParameter.prototype.defaultValue = 0;

$root.CoreML.Specification.StringParameter = class StringParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringParameter();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.defaultValue = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.StringParameter.prototype.defaultValue = "";

$root.CoreML.Specification.BoolParameter = class BoolParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BoolParameter();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.defaultValue = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BoolParameter.prototype.defaultValue = false;

$root.CoreML.Specification.Identity = class Identity {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Identity();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Imputer = class Imputer {

    constructor() {
    }

    get ImputedValue() {
        $root.CoreML.Specification.Imputer.ImputedValueSet = $root.CoreML.Specification.Imputer.ImputedValueSet || new Set([ "imputedDoubleValue", "imputedInt64Value", "imputedStringValue", "imputedDoubleArray", "imputedInt64Array", "imputedStringDictionary", "imputedInt64Dictionary"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Imputer.ImputedValueSet.has(key) && this[key] != null);
    }

    get ReplaceValue() {
        $root.CoreML.Specification.Imputer.ReplaceValueSet = $root.CoreML.Specification.Imputer.ReplaceValueSet || new Set([ "replaceDoubleValue", "replaceInt64Value", "replaceStringValue"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Imputer.ReplaceValueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Imputer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.imputedDoubleValue = reader.double();
                    break;
                case 2:
                    message.imputedInt64Value = reader.int64();
                    break;
                case 3:
                    message.imputedStringValue = reader.string();
                    break;
                case 4:
                    message.imputedDoubleArray = $root.CoreML.Specification.DoubleVector.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.imputedInt64Array = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.imputedStringDictionary = $root.CoreML.Specification.StringToDoubleMap.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.imputedInt64Dictionary = $root.CoreML.Specification.Int64ToDoubleMap.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.replaceDoubleValue = reader.double();
                    break;
                case 12:
                    message.replaceInt64Value = reader.int64();
                    break;
                case 13:
                    message.replaceStringValue = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkMultiArrayShapeMapping = {
    "RANK5_ARRAY_MAPPING": 0,
    "EXACT_ARRAY_MAPPING": 1
};

$root.CoreML.Specification.NeuralNetworkImageShapeMapping = {
    "RANK5_IMAGE_MAPPING": 0,
    "RANK4_IMAGE_MAPPING": 1
};

$root.CoreML.Specification.NeuralNetwork = class NeuralNetwork {

    constructor() {
        this.layers = [];
        this.preprocessing = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetwork();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.preprocessing.push($root.CoreML.Specification.NeuralNetworkPreprocessing.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.arrayInputShapeMapping = reader.int32();
                    break;
                case 6:
                    message.imageInputShapeMapping = reader.int32();
                    break;
                case 10:
                    message.updateParams = $root.CoreML.Specification.NetworkUpdateParameters.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetwork.prototype.arrayInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetwork.prototype.imageInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetwork.prototype.updateParams = null;

$root.CoreML.Specification.NeuralNetworkImageScaler = class NeuralNetworkImageScaler {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkImageScaler();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.channelScale = reader.float();
                    break;
                case 20:
                    message.blueBias = reader.float();
                    break;
                case 21:
                    message.greenBias = reader.float();
                    break;
                case 22:
                    message.redBias = reader.float();
                    break;
                case 30:
                    message.grayBias = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkImageScaler.prototype.channelScale = 0;
$root.CoreML.Specification.NeuralNetworkImageScaler.prototype.blueBias = 0;
$root.CoreML.Specification.NeuralNetworkImageScaler.prototype.greenBias = 0;
$root.CoreML.Specification.NeuralNetworkImageScaler.prototype.redBias = 0;
$root.CoreML.Specification.NeuralNetworkImageScaler.prototype.grayBias = 0;

$root.CoreML.Specification.NeuralNetworkMeanImage = class NeuralNetworkMeanImage {

    constructor() {
        this.meanImage = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkMeanImage();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meanImage = reader.floats(message.meanImage, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkPreprocessing = class NeuralNetworkPreprocessing {

    constructor() {
    }

    get preprocessor() {
        $root.CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet = $root.CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet || new Set([ "scaler", "meanImage"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkPreprocessing();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.featureName = reader.string();
                    break;
                case 10:
                    message.scaler = $root.CoreML.Specification.NeuralNetworkImageScaler.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.meanImage = $root.CoreML.Specification.NeuralNetworkMeanImage.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkPreprocessing.prototype.featureName = "";

$root.CoreML.Specification.ActivationReLU = class ActivationReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationReLU();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationLeakyReLU = class ActivationLeakyReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationLeakyReLU();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationLeakyReLU.prototype.alpha = 0;

$root.CoreML.Specification.ActivationTanh = class ActivationTanh {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationTanh();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationScaledTanh = class ActivationScaledTanh {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationScaledTanh();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationScaledTanh.prototype.alpha = 0;
$root.CoreML.Specification.ActivationScaledTanh.prototype.beta = 0;

$root.CoreML.Specification.ActivationSigmoid = class ActivationSigmoid {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSigmoid();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationLinear = class ActivationLinear {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationLinear();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationLinear.prototype.alpha = 0;
$root.CoreML.Specification.ActivationLinear.prototype.beta = 0;

$root.CoreML.Specification.ActivationSigmoidHard = class ActivationSigmoidHard {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSigmoidHard();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationSigmoidHard.prototype.alpha = 0;
$root.CoreML.Specification.ActivationSigmoidHard.prototype.beta = 0;

$root.CoreML.Specification.ActivationPReLU = class ActivationPReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationPReLU();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationPReLU.prototype.alpha = null;

$root.CoreML.Specification.ActivationELU = class ActivationELU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationELU();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationELU.prototype.alpha = 0;

$root.CoreML.Specification.ActivationThresholdedReLU = class ActivationThresholdedReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationThresholdedReLU();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationThresholdedReLU.prototype.alpha = 0;

$root.CoreML.Specification.ActivationSoftsign = class ActivationSoftsign {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSoftsign();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationSoftplus = class ActivationSoftplus {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSoftplus();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationParametricSoftplus = class ActivationParametricSoftplus {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationParametricSoftplus();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.beta = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ActivationParametricSoftplus.prototype.alpha = null;
$root.CoreML.Specification.ActivationParametricSoftplus.prototype.beta = null;

$root.CoreML.Specification.ActivationParams = class ActivationParams {

    constructor() {
    }

    get NonlinearityType() {
        $root.CoreML.Specification.ActivationParams.NonlinearityTypeSet = $root.CoreML.Specification.ActivationParams.NonlinearityTypeSet || new Set([ "linear", "ReLU", "leakyReLU", "thresholdedReLU", "PReLU", "tanh", "scaledTanh", "sigmoid", "sigmoidHard", "ELU", "softsign", "softplus", "parametricSoftplus"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.ActivationParams.NonlinearityTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 5:
                    message.linear = $root.CoreML.Specification.ActivationLinear.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.ReLU = $root.CoreML.Specification.ActivationReLU.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.leakyReLU = $root.CoreML.Specification.ActivationLeakyReLU.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.thresholdedReLU = $root.CoreML.Specification.ActivationThresholdedReLU.decode(reader, reader.uint32());
                    break;
                case 25:
                    message.PReLU = $root.CoreML.Specification.ActivationPReLU.decode(reader, reader.uint32());
                    break;
                case 30:
                    message.tanh = $root.CoreML.Specification.ActivationTanh.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.scaledTanh = $root.CoreML.Specification.ActivationScaledTanh.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.sigmoid = $root.CoreML.Specification.ActivationSigmoid.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.sigmoidHard = $root.CoreML.Specification.ActivationSigmoidHard.decode(reader, reader.uint32());
                    break;
                case 50:
                    message.ELU = $root.CoreML.Specification.ActivationELU.decode(reader, reader.uint32());
                    break;
                case 60:
                    message.softsign = $root.CoreML.Specification.ActivationSoftsign.decode(reader, reader.uint32());
                    break;
                case 70:
                    message.softplus = $root.CoreML.Specification.ActivationSoftplus.decode(reader, reader.uint32());
                    break;
                case 71:
                    message.parametricSoftplus = $root.CoreML.Specification.ActivationParametricSoftplus.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Tensor = class Tensor {

    constructor() {
        this.dimValue = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Tensor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.rank = reader.uint32();
                    break;
                case 2:
                    message.dimValue = reader.array(message.dimValue, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Tensor.prototype.rank = 0;

$root.CoreML.Specification.NeuralNetworkLayer = class NeuralNetworkLayer {

    constructor() {
        this.input = [];
        this.output = [];
        this.inputTensor = [];
        this.outputTensor = [];
    }

    get layer() {
        $root.CoreML.Specification.NeuralNetworkLayer.layerSet = $root.CoreML.Specification.NeuralNetworkLayer.layerSet || new Set([ "convolution", "pooling", "activation", "innerProduct", "embedding", "batchnorm", "mvn", "l2normalize", "softmax", "lrn", "crop", "padding", "upsample", "resizeBilinear", "cropResize", "unary", "add", "multiply", "average", "scale", "bias", "max", "min", "dot", "reduce", "loadConstant", "reshape", "flatten", "permute", "concat", "split", "sequenceRepeat", "reorganizeData", "slice", "simpleRecurrent", "gru", "uniDirectionalLSTM", "biDirectionalLSTM", "custom", "copy", "branch", "loop", "loopBreak", "loopContinue", "rangeStatic", "rangeDynamic", "clip", "ceil", "floor", "sign", "round", "exp2", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "gelu", "equal", "notEqual", "lessThan", "lessEqual", "greaterThan", "greaterEqual", "logicalOr", "logicalXor", "logicalNot", "logicalAnd", "modBroadcastable", "minBroadcastable", "maxBroadcastable", "addBroadcastable", "powBroadcastable", "divideBroadcastable", "floorDivBroadcastable", "multiplyBroadcastable", "subtractBroadcastable", "tile", "stack", "gather", "scatter", "gatherND", "scatterND", "softmaxND", "gatherAlongAxis", "scatterAlongAxis", "reverse", "reverseSeq", "splitND", "concatND", "transpose", "sliceStatic", "sliceDynamic", "slidingWindows", "topK", "argMin", "argMax", "embeddingND", "batchedMatmul", "getShape", "loadConstantND", "fillLike", "fillStatic", "fillDynamic", "broadcastToLike", "broadcastToStatic", "broadcastToDynamic", "squeeze", "expandDims", "flattenTo2D", "reshapeLike", "reshapeStatic", "reshapeDynamic", "rankPreservingReshape", "constantPad", "randomNormalLike", "randomNormalStatic", "randomNormalDynamic", "randomUniformLike", "randomUniformStatic", "randomUniformDynamic", "randomBernoulliLike", "randomBernoulliStatic", "randomBernoulliDynamic", "categoricalDistribution", "reduceL1", "reduceL2", "reduceMax", "reduceMin", "reduceSum", "reduceProd", "reduceMean", "reduceLogSum", "reduceSumSquare", "reduceLogSumExp", "whereNonZero", "matrixBandPart", "lowerTriangular", "upperTriangular", "whereBroadcastable", "layerNormalization", "NonMaximumSuppression", "oneHot", "cumSum", "clampedReLU", "argSort", "pooling3d", "globalPooling3d", "sliceBySize", "convolution3d"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NeuralNetworkLayer.layerSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkLayer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.input.push(reader.string());
                    break;
                case 3:
                    message.output.push(reader.string());
                    break;
                case 4:
                    message.inputTensor.push($root.CoreML.Specification.Tensor.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.outputTensor.push($root.CoreML.Specification.Tensor.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.isUpdatable = reader.bool();
                    break;
                case 100:
                    message.convolution = $root.CoreML.Specification.ConvolutionLayerParams.decode(reader, reader.uint32());
                    break;
                case 120:
                    message.pooling = $root.CoreML.Specification.PoolingLayerParams.decode(reader, reader.uint32());
                    break;
                case 130:
                    message.activation = $root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32());
                    break;
                case 140:
                    message.innerProduct = $root.CoreML.Specification.InnerProductLayerParams.decode(reader, reader.uint32());
                    break;
                case 150:
                    message.embedding = $root.CoreML.Specification.EmbeddingLayerParams.decode(reader, reader.uint32());
                    break;
                case 160:
                    message.batchnorm = $root.CoreML.Specification.BatchnormLayerParams.decode(reader, reader.uint32());
                    break;
                case 165:
                    message.mvn = $root.CoreML.Specification.MeanVarianceNormalizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 170:
                    message.l2normalize = $root.CoreML.Specification.L2NormalizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 175:
                    message.softmax = $root.CoreML.Specification.SoftmaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 180:
                    message.lrn = $root.CoreML.Specification.LRNLayerParams.decode(reader, reader.uint32());
                    break;
                case 190:
                    message.crop = $root.CoreML.Specification.CropLayerParams.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.padding = $root.CoreML.Specification.PaddingLayerParams.decode(reader, reader.uint32());
                    break;
                case 210:
                    message.upsample = $root.CoreML.Specification.UpsampleLayerParams.decode(reader, reader.uint32());
                    break;
                case 211:
                    message.resizeBilinear = $root.CoreML.Specification.ResizeBilinearLayerParams.decode(reader, reader.uint32());
                    break;
                case 212:
                    message.cropResize = $root.CoreML.Specification.CropResizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 220:
                    message.unary = $root.CoreML.Specification.UnaryFunctionLayerParams.decode(reader, reader.uint32());
                    break;
                case 230:
                    message.add = $root.CoreML.Specification.AddLayerParams.decode(reader, reader.uint32());
                    break;
                case 231:
                    message.multiply = $root.CoreML.Specification.MultiplyLayerParams.decode(reader, reader.uint32());
                    break;
                case 240:
                    message.average = $root.CoreML.Specification.AverageLayerParams.decode(reader, reader.uint32());
                    break;
                case 245:
                    message.scale = $root.CoreML.Specification.ScaleLayerParams.decode(reader, reader.uint32());
                    break;
                case 250:
                    message.bias = $root.CoreML.Specification.BiasLayerParams.decode(reader, reader.uint32());
                    break;
                case 260:
                    message.max = $root.CoreML.Specification.MaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 261:
                    message.min = $root.CoreML.Specification.MinLayerParams.decode(reader, reader.uint32());
                    break;
                case 270:
                    message.dot = $root.CoreML.Specification.DotProductLayerParams.decode(reader, reader.uint32());
                    break;
                case 280:
                    message.reduce = $root.CoreML.Specification.ReduceLayerParams.decode(reader, reader.uint32());
                    break;
                case 290:
                    message.loadConstant = $root.CoreML.Specification.LoadConstantLayerParams.decode(reader, reader.uint32());
                    break;
                case 300:
                    message.reshape = $root.CoreML.Specification.ReshapeLayerParams.decode(reader, reader.uint32());
                    break;
                case 301:
                    message.flatten = $root.CoreML.Specification.FlattenLayerParams.decode(reader, reader.uint32());
                    break;
                case 310:
                    message.permute = $root.CoreML.Specification.PermuteLayerParams.decode(reader, reader.uint32());
                    break;
                case 320:
                    message.concat = $root.CoreML.Specification.ConcatLayerParams.decode(reader, reader.uint32());
                    break;
                case 330:
                    message.split = $root.CoreML.Specification.SplitLayerParams.decode(reader, reader.uint32());
                    break;
                case 340:
                    message.sequenceRepeat = $root.CoreML.Specification.SequenceRepeatLayerParams.decode(reader, reader.uint32());
                    break;
                case 345:
                    message.reorganizeData = $root.CoreML.Specification.ReorganizeDataLayerParams.decode(reader, reader.uint32());
                    break;
                case 350:
                    message.slice = $root.CoreML.Specification.SliceLayerParams.decode(reader, reader.uint32());
                    break;
                case 400:
                    message.simpleRecurrent = $root.CoreML.Specification.SimpleRecurrentLayerParams.decode(reader, reader.uint32());
                    break;
                case 410:
                    message.gru = $root.CoreML.Specification.GRULayerParams.decode(reader, reader.uint32());
                    break;
                case 420:
                    message.uniDirectionalLSTM = $root.CoreML.Specification.UniDirectionalLSTMLayerParams.decode(reader, reader.uint32());
                    break;
                case 430:
                    message.biDirectionalLSTM = $root.CoreML.Specification.BiDirectionalLSTMLayerParams.decode(reader, reader.uint32());
                    break;
                case 500:
                    message.custom = $root.CoreML.Specification.CustomLayerParams.decode(reader, reader.uint32());
                    break;
                case 600:
                    message.copy = $root.CoreML.Specification.CopyLayerParams.decode(reader, reader.uint32());
                    break;
                case 605:
                    message.branch = $root.CoreML.Specification.BranchLayerParams.decode(reader, reader.uint32());
                    break;
                case 615:
                    message.loop = $root.CoreML.Specification.LoopLayerParams.decode(reader, reader.uint32());
                    break;
                case 620:
                    message.loopBreak = $root.CoreML.Specification.LoopBreakLayerParams.decode(reader, reader.uint32());
                    break;
                case 625:
                    message.loopContinue = $root.CoreML.Specification.LoopContinueLayerParams.decode(reader, reader.uint32());
                    break;
                case 635:
                    message.rangeStatic = $root.CoreML.Specification.RangeStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 640:
                    message.rangeDynamic = $root.CoreML.Specification.RangeDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 660:
                    message.clip = $root.CoreML.Specification.ClipLayerParams.decode(reader, reader.uint32());
                    break;
                case 665:
                    message.ceil = $root.CoreML.Specification.CeilLayerParams.decode(reader, reader.uint32());
                    break;
                case 670:
                    message.floor = $root.CoreML.Specification.FloorLayerParams.decode(reader, reader.uint32());
                    break;
                case 680:
                    message.sign = $root.CoreML.Specification.SignLayerParams.decode(reader, reader.uint32());
                    break;
                case 685:
                    message.round = $root.CoreML.Specification.RoundLayerParams.decode(reader, reader.uint32());
                    break;
                case 700:
                    message.exp2 = $root.CoreML.Specification.Exp2LayerParams.decode(reader, reader.uint32());
                    break;
                case 710:
                    message.sin = $root.CoreML.Specification.SinLayerParams.decode(reader, reader.uint32());
                    break;
                case 715:
                    message.cos = $root.CoreML.Specification.CosLayerParams.decode(reader, reader.uint32());
                    break;
                case 720:
                    message.tan = $root.CoreML.Specification.TanLayerParams.decode(reader, reader.uint32());
                    break;
                case 730:
                    message.asin = $root.CoreML.Specification.AsinLayerParams.decode(reader, reader.uint32());
                    break;
                case 735:
                    message.acos = $root.CoreML.Specification.AcosLayerParams.decode(reader, reader.uint32());
                    break;
                case 740:
                    message.atan = $root.CoreML.Specification.AtanLayerParams.decode(reader, reader.uint32());
                    break;
                case 750:
                    message.sinh = $root.CoreML.Specification.SinhLayerParams.decode(reader, reader.uint32());
                    break;
                case 755:
                    message.cosh = $root.CoreML.Specification.CoshLayerParams.decode(reader, reader.uint32());
                    break;
                case 760:
                    message.tanh = $root.CoreML.Specification.TanhLayerParams.decode(reader, reader.uint32());
                    break;
                case 770:
                    message.asinh = $root.CoreML.Specification.AsinhLayerParams.decode(reader, reader.uint32());
                    break;
                case 775:
                    message.acosh = $root.CoreML.Specification.AcoshLayerParams.decode(reader, reader.uint32());
                    break;
                case 780:
                    message.atanh = $root.CoreML.Specification.AtanhLayerParams.decode(reader, reader.uint32());
                    break;
                case 790:
                    message.erf = $root.CoreML.Specification.ErfLayerParams.decode(reader, reader.uint32());
                    break;
                case 795:
                    message.gelu = $root.CoreML.Specification.GeluLayerParams.decode(reader, reader.uint32());
                    break;
                case 815:
                    message.equal = $root.CoreML.Specification.EqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 820:
                    message.notEqual = $root.CoreML.Specification.NotEqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 825:
                    message.lessThan = $root.CoreML.Specification.LessThanLayerParams.decode(reader, reader.uint32());
                    break;
                case 827:
                    message.lessEqual = $root.CoreML.Specification.LessEqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 830:
                    message.greaterThan = $root.CoreML.Specification.GreaterThanLayerParams.decode(reader, reader.uint32());
                    break;
                case 832:
                    message.greaterEqual = $root.CoreML.Specification.GreaterEqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 840:
                    message.logicalOr = $root.CoreML.Specification.LogicalOrLayerParams.decode(reader, reader.uint32());
                    break;
                case 845:
                    message.logicalXor = $root.CoreML.Specification.LogicalXorLayerParams.decode(reader, reader.uint32());
                    break;
                case 850:
                    message.logicalNot = $root.CoreML.Specification.LogicalNotLayerParams.decode(reader, reader.uint32());
                    break;
                case 855:
                    message.logicalAnd = $root.CoreML.Specification.LogicalAndLayerParams.decode(reader, reader.uint32());
                    break;
                case 865:
                    message.modBroadcastable = $root.CoreML.Specification.ModBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 870:
                    message.minBroadcastable = $root.CoreML.Specification.MinBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 875:
                    message.maxBroadcastable = $root.CoreML.Specification.MaxBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 880:
                    message.addBroadcastable = $root.CoreML.Specification.AddBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 885:
                    message.powBroadcastable = $root.CoreML.Specification.PowBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 890:
                    message.divideBroadcastable = $root.CoreML.Specification.DivideBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 895:
                    message.floorDivBroadcastable = $root.CoreML.Specification.FloorDivBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 900:
                    message.multiplyBroadcastable = $root.CoreML.Specification.MultiplyBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 905:
                    message.subtractBroadcastable = $root.CoreML.Specification.SubtractBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 920:
                    message.tile = $root.CoreML.Specification.TileLayerParams.decode(reader, reader.uint32());
                    break;
                case 925:
                    message.stack = $root.CoreML.Specification.StackLayerParams.decode(reader, reader.uint32());
                    break;
                case 930:
                    message.gather = $root.CoreML.Specification.GatherLayerParams.decode(reader, reader.uint32());
                    break;
                case 935:
                    message.scatter = $root.CoreML.Specification.ScatterLayerParams.decode(reader, reader.uint32());
                    break;
                case 940:
                    message.gatherND = $root.CoreML.Specification.GatherNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 945:
                    message.scatterND = $root.CoreML.Specification.ScatterNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 950:
                    message.softmaxND = $root.CoreML.Specification.SoftmaxNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 952:
                    message.gatherAlongAxis = $root.CoreML.Specification.GatherAlongAxisLayerParams.decode(reader, reader.uint32());
                    break;
                case 954:
                    message.scatterAlongAxis = $root.CoreML.Specification.ScatterAlongAxisLayerParams.decode(reader, reader.uint32());
                    break;
                case 960:
                    message.reverse = $root.CoreML.Specification.ReverseLayerParams.decode(reader, reader.uint32());
                    break;
                case 965:
                    message.reverseSeq = $root.CoreML.Specification.ReverseSeqLayerParams.decode(reader, reader.uint32());
                    break;
                case 975:
                    message.splitND = $root.CoreML.Specification.SplitNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 980:
                    message.concatND = $root.CoreML.Specification.ConcatNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 985:
                    message.transpose = $root.CoreML.Specification.TransposeLayerParams.decode(reader, reader.uint32());
                    break;
                case 995:
                    message.sliceStatic = $root.CoreML.Specification.SliceStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1000:
                    message.sliceDynamic = $root.CoreML.Specification.SliceDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1005:
                    message.slidingWindows = $root.CoreML.Specification.SlidingWindowsLayerParams.decode(reader, reader.uint32());
                    break;
                case 1015:
                    message.topK = $root.CoreML.Specification.TopKLayerParams.decode(reader, reader.uint32());
                    break;
                case 1020:
                    message.argMin = $root.CoreML.Specification.ArgMinLayerParams.decode(reader, reader.uint32());
                    break;
                case 1025:
                    message.argMax = $root.CoreML.Specification.ArgMaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 1040:
                    message.embeddingND = $root.CoreML.Specification.EmbeddingNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 1045:
                    message.batchedMatmul = $root.CoreML.Specification.BatchedMatMulLayerParams.decode(reader, reader.uint32());
                    break;
                case 1065:
                    message.getShape = $root.CoreML.Specification.GetShapeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1070:
                    message.loadConstantND = $root.CoreML.Specification.LoadConstantNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 1080:
                    message.fillLike = $root.CoreML.Specification.FillLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1085:
                    message.fillStatic = $root.CoreML.Specification.FillStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1090:
                    message.fillDynamic = $root.CoreML.Specification.FillDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1100:
                    message.broadcastToLike = $root.CoreML.Specification.BroadcastToLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1105:
                    message.broadcastToStatic = $root.CoreML.Specification.BroadcastToStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1110:
                    message.broadcastToDynamic = $root.CoreML.Specification.BroadcastToDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1120:
                    message.squeeze = $root.CoreML.Specification.SqueezeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1125:
                    message.expandDims = $root.CoreML.Specification.ExpandDimsLayerParams.decode(reader, reader.uint32());
                    break;
                case 1130:
                    message.flattenTo2D = $root.CoreML.Specification.FlattenTo2DLayerParams.decode(reader, reader.uint32());
                    break;
                case 1135:
                    message.reshapeLike = $root.CoreML.Specification.ReshapeLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1140:
                    message.reshapeStatic = $root.CoreML.Specification.ReshapeStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1145:
                    message.reshapeDynamic = $root.CoreML.Specification.ReshapeDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1150:
                    message.rankPreservingReshape = $root.CoreML.Specification.RankPreservingReshapeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1155:
                    message.constantPad = $root.CoreML.Specification.ConstantPaddingLayerParams.decode(reader, reader.uint32());
                    break;
                case 1170:
                    message.randomNormalLike = $root.CoreML.Specification.RandomNormalLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1175:
                    message.randomNormalStatic = $root.CoreML.Specification.RandomNormalStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1180:
                    message.randomNormalDynamic = $root.CoreML.Specification.RandomNormalDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1190:
                    message.randomUniformLike = $root.CoreML.Specification.RandomUniformLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1195:
                    message.randomUniformStatic = $root.CoreML.Specification.RandomUniformStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1200:
                    message.randomUniformDynamic = $root.CoreML.Specification.RandomUniformDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1210:
                    message.randomBernoulliLike = $root.CoreML.Specification.RandomBernoulliLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1215:
                    message.randomBernoulliStatic = $root.CoreML.Specification.RandomBernoulliStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1220:
                    message.randomBernoulliDynamic = $root.CoreML.Specification.RandomBernoulliDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1230:
                    message.categoricalDistribution = $root.CoreML.Specification.CategoricalDistributionLayerParams.decode(reader, reader.uint32());
                    break;
                case 1250:
                    message.reduceL1 = $root.CoreML.Specification.ReduceL1LayerParams.decode(reader, reader.uint32());
                    break;
                case 1255:
                    message.reduceL2 = $root.CoreML.Specification.ReduceL2LayerParams.decode(reader, reader.uint32());
                    break;
                case 1260:
                    message.reduceMax = $root.CoreML.Specification.ReduceMaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 1265:
                    message.reduceMin = $root.CoreML.Specification.ReduceMinLayerParams.decode(reader, reader.uint32());
                    break;
                case 1270:
                    message.reduceSum = $root.CoreML.Specification.ReduceSumLayerParams.decode(reader, reader.uint32());
                    break;
                case 1275:
                    message.reduceProd = $root.CoreML.Specification.ReduceProdLayerParams.decode(reader, reader.uint32());
                    break;
                case 1280:
                    message.reduceMean = $root.CoreML.Specification.ReduceMeanLayerParams.decode(reader, reader.uint32());
                    break;
                case 1285:
                    message.reduceLogSum = $root.CoreML.Specification.ReduceLogSumLayerParams.decode(reader, reader.uint32());
                    break;
                case 1290:
                    message.reduceSumSquare = $root.CoreML.Specification.ReduceSumSquareLayerParams.decode(reader, reader.uint32());
                    break;
                case 1295:
                    message.reduceLogSumExp = $root.CoreML.Specification.ReduceLogSumExpLayerParams.decode(reader, reader.uint32());
                    break;
                case 1313:
                    message.whereNonZero = $root.CoreML.Specification.WhereNonZeroLayerParams.decode(reader, reader.uint32());
                    break;
                case 1315:
                    message.matrixBandPart = $root.CoreML.Specification.MatrixBandPartLayerParams.decode(reader, reader.uint32());
                    break;
                case 1320:
                    message.lowerTriangular = $root.CoreML.Specification.LowerTriangularLayerParams.decode(reader, reader.uint32());
                    break;
                case 1325:
                    message.upperTriangular = $root.CoreML.Specification.UpperTriangularLayerParams.decode(reader, reader.uint32());
                    break;
                case 1330:
                    message.whereBroadcastable = $root.CoreML.Specification.WhereBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 1350:
                    message.layerNormalization = $root.CoreML.Specification.LayerNormalizationLayerParams.decode(reader, reader.uint32());
                    break;
                case 1400:
                    message.NonMaximumSuppression = $root.CoreML.Specification.NonMaximumSuppressionLayerParams.decode(reader, reader.uint32());
                    break;
                case 1450:
                    message.oneHot = $root.CoreML.Specification.OneHotLayerParams.decode(reader, reader.uint32());
                    break;
                case 1455:
                    message.cumSum = $root.CoreML.Specification.CumSumLayerParams.decode(reader, reader.uint32());
                    break;
                case 1460:
                    message.clampedReLU = $root.CoreML.Specification.ClampedReLULayerParams.decode(reader, reader.uint32());
                    break;
                case 1461:
                    message.argSort = $root.CoreML.Specification.ArgSortLayerParams.decode(reader, reader.uint32());
                    break;
                case 1465:
                    message.pooling3d = $root.CoreML.Specification.Pooling3DLayerParams.decode(reader, reader.uint32());
                    break;
                case 1466:
                    message.globalPooling3d = $root.CoreML.Specification.GlobalPooling3DLayerParams.decode(reader, reader.uint32());
                    break;
                case 1470:
                    message.sliceBySize = $root.CoreML.Specification.SliceBySizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1471:
                    message.convolution3d = $root.CoreML.Specification.Convolution3DLayerParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkLayer.prototype.name = "";
$root.CoreML.Specification.NeuralNetworkLayer.prototype.isUpdatable = false;

$root.CoreML.Specification.BranchLayerParams = class BranchLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BranchLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ifBranch = $root.CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.elseBranch = $root.CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BranchLayerParams.prototype.ifBranch = null;
$root.CoreML.Specification.BranchLayerParams.prototype.elseBranch = null;

$root.CoreML.Specification.LoopLayerParams = class LoopLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoopLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.maxLoopIterations = reader.uint64();
                    break;
                case 2:
                    message.conditionVar = reader.string();
                    break;
                case 3:
                    message.conditionNetwork = $root.CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bodyNetwork = $root.CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LoopLayerParams.prototype.maxLoopIterations = protobuf.Uint64.create(0);
$root.CoreML.Specification.LoopLayerParams.prototype.conditionVar = "";
$root.CoreML.Specification.LoopLayerParams.prototype.conditionNetwork = null;
$root.CoreML.Specification.LoopLayerParams.prototype.bodyNetwork = null;

$root.CoreML.Specification.LoopBreakLayerParams = class LoopBreakLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoopBreakLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LoopContinueLayerParams = class LoopContinueLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoopContinueLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CopyLayerParams = class CopyLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CopyLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GreaterThanLayerParams = class GreaterThanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GreaterThanLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GreaterThanLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.GreaterEqualLayerParams = class GreaterEqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GreaterEqualLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GreaterEqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.LessThanLayerParams = class LessThanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LessThanLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LessThanLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.LessEqualLayerParams = class LessEqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LessEqualLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LessEqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.EqualLayerParams = class EqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.EqualLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.EqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.NotEqualLayerParams = class NotEqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NotEqualLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NotEqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.LogicalAndLayerParams = class LogicalAndLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalAndLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LogicalOrLayerParams = class LogicalOrLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalOrLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LogicalXorLayerParams = class LogicalXorLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalXorLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LogicalNotLayerParams = class LogicalNotLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalNotLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BorderAmounts = class BorderAmounts {

    constructor() {
        this.borderAmounts = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BorderAmounts();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.borderAmounts.push($root.CoreML.Specification.BorderAmounts.EdgeSizes.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BorderAmounts.EdgeSizes = class EdgeSizes {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BorderAmounts.EdgeSizes();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.startEdgeSize = reader.uint64();
                    break;
                case 2:
                    message.endEdgeSize = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BorderAmounts.EdgeSizes.prototype.startEdgeSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.BorderAmounts.EdgeSizes.prototype.endEdgeSize = protobuf.Uint64.create(0);

$root.CoreML.Specification.ValidPadding = class ValidPadding {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ValidPadding();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.paddingAmounts = $root.CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ValidPadding.prototype.paddingAmounts = null;

$root.CoreML.Specification.SamePadding = class SamePadding {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SamePadding();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.asymmetryMode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SamePadding.prototype.asymmetryMode = 0;

$root.CoreML.Specification.SamePadding.SamePaddingMode = {
    "BOTTOM_RIGHT_HEAVY": 0,
    "TOP_LEFT_HEAVY": 1
};

$root.CoreML.Specification.SamplingMode = class SamplingMode {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SamplingMode();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.samplingMethod = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SamplingMode.prototype.samplingMethod = 0;

$root.CoreML.Specification.SamplingMode.Method = {
    "STRICT_ALIGN_ENDPOINTS_MODE": 0,
    "ALIGN_ENDPOINTS_MODE": 1,
    "UPSAMPLE_MODE": 2,
    "ROI_ALIGN_MODE": 3
};

$root.CoreML.Specification.BoxCoordinatesMode = class BoxCoordinatesMode {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BoxCoordinatesMode();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.boxMode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BoxCoordinatesMode.prototype.boxMode = 0;

$root.CoreML.Specification.BoxCoordinatesMode.Coordinates = {
    "CORNERS_HEIGHT_FIRST": 0,
    "CORNERS_WIDTH_FIRST": 1,
    "CENTER_SIZE_HEIGHT_FIRST": 2,
    "CENTER_SIZE_WIDTH_FIRST": 3
};

$root.CoreML.Specification.WeightParams = class WeightParams {

    constructor() {
        this.floatValue = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.WeightParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.floatValue = reader.floats(message.floatValue, tag);
                    break;
                case 2:
                    message.float16Value = reader.bytes();
                    break;
                case 30:
                    message.rawValue = reader.bytes();
                    break;
                case 31:
                    message.int8RawValue = reader.bytes();
                    break;
                case 40:
                    message.quantization = $root.CoreML.Specification.QuantizationParams.decode(reader, reader.uint32());
                    break;
                case 50:
                    message.isUpdatable = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.WeightParams.prototype.float16Value = new Uint8Array([]);
$root.CoreML.Specification.WeightParams.prototype.rawValue = new Uint8Array([]);
$root.CoreML.Specification.WeightParams.prototype.int8RawValue = new Uint8Array([]);
$root.CoreML.Specification.WeightParams.prototype.quantization = null;
$root.CoreML.Specification.WeightParams.prototype.isUpdatable = false;

$root.CoreML.Specification.QuantizationParams = class QuantizationParams {

    constructor() {
    }

    get QuantizationType() {
        $root.CoreML.Specification.QuantizationParams.QuantizationTypeSet = $root.CoreML.Specification.QuantizationParams.QuantizationTypeSet || new Set([ "linearQuantization", "lookupTableQuantization"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.QuantizationParams.QuantizationTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.QuantizationParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numberOfBits = reader.uint64();
                    break;
                case 101:
                    message.linearQuantization = $root.CoreML.Specification.LinearQuantizationParams.decode(reader, reader.uint32());
                    break;
                case 102:
                    message.lookupTableQuantization = $root.CoreML.Specification.LookUpTableQuantizationParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.QuantizationParams.prototype.numberOfBits = protobuf.Uint64.create(0);

$root.CoreML.Specification.LinearQuantizationParams = class LinearQuantizationParams {

    constructor() {
        this.scale = [];
        this.bias = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinearQuantizationParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.scale = reader.floats(message.scale, tag);
                    break;
                case 2:
                    message.bias = reader.floats(message.bias, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LookUpTableQuantizationParams = class LookUpTableQuantizationParams {

    constructor() {
        this.floatValue = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LookUpTableQuantizationParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.floatValue = reader.floats(message.floatValue, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConvolutionLayerParams = class ConvolutionLayerParams {

    constructor() {
        this.kernelSize = [];
        this.stride = [];
        this.dilationFactor = [];
        this.outputShape = [];
    }

    get ConvolutionPaddingType() {
        $root.CoreML.Specification.ConvolutionLayerParams.ConvolutionPaddingTypeSet = $root.CoreML.Specification.ConvolutionLayerParams.ConvolutionPaddingTypeSet || new Set([ "valid", "same"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.ConvolutionLayerParams.ConvolutionPaddingTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ConvolutionLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.outputChannels = reader.uint64();
                    break;
                case 2:
                    message.kernelChannels = reader.uint64();
                    break;
                case 10:
                    message.nGroups = reader.uint64();
                    break;
                case 20:
                    message.kernelSize = reader.array(message.kernelSize, () => reader.uint64(), tag);
                    break;
                case 30:
                    message.stride = reader.array(message.stride, () => reader.uint64(), tag);
                    break;
                case 40:
                    message.dilationFactor = reader.array(message.dilationFactor, () => reader.uint64(), tag);
                    break;
                case 50:
                    message.valid = $root.CoreML.Specification.ValidPadding.decode(reader, reader.uint32());
                    break;
                case 51:
                    message.same = $root.CoreML.Specification.SamePadding.decode(reader, reader.uint32());
                    break;
                case 60:
                    message.isDeconvolution = reader.bool();
                    break;
                case 70:
                    message.hasBias = reader.bool();
                    break;
                case 90:
                    message.weights = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 91:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.outputShape = reader.array(message.outputShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConvolutionLayerParams.prototype.outputChannels = protobuf.Uint64.create(0);
$root.CoreML.Specification.ConvolutionLayerParams.prototype.kernelChannels = protobuf.Uint64.create(0);
$root.CoreML.Specification.ConvolutionLayerParams.prototype.nGroups = protobuf.Uint64.create(0);
$root.CoreML.Specification.ConvolutionLayerParams.prototype.isDeconvolution = false;
$root.CoreML.Specification.ConvolutionLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.ConvolutionLayerParams.prototype.weights = null;
$root.CoreML.Specification.ConvolutionLayerParams.prototype.bias = null;

$root.CoreML.Specification.Convolution3DLayerParams = class Convolution3DLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Convolution3DLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.outputChannels = reader.int32();
                    break;
                case 2:
                    message.inputChannels = reader.int32();
                    break;
                case 10:
                    message.nGroups = reader.int32();
                    break;
                case 20:
                    message.kernelDepth = reader.int32();
                    break;
                case 21:
                    message.kernelHeight = reader.int32();
                    break;
                case 22:
                    message.kernelWidth = reader.int32();
                    break;
                case 31:
                    message.strideDepth = reader.int32();
                    break;
                case 32:
                    message.strideHeight = reader.int32();
                    break;
                case 33:
                    message.strideWidth = reader.int32();
                    break;
                case 40:
                    message.dilationDepth = reader.int32();
                    break;
                case 41:
                    message.dilationHeight = reader.int32();
                    break;
                case 42:
                    message.dilationWidth = reader.int32();
                    break;
                case 50:
                    message.hasBias = reader.bool();
                    break;
                case 60:
                    message.weights = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 61:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 70:
                    message.paddingType = reader.int32();
                    break;
                case 80:
                    message.customPaddingFront = reader.int32();
                    break;
                case 81:
                    message.customPaddingBack = reader.int32();
                    break;
                case 82:
                    message.customPaddingTop = reader.int32();
                    break;
                case 83:
                    message.customPaddingBottom = reader.int32();
                    break;
                case 84:
                    message.customPaddingLeft = reader.int32();
                    break;
                case 85:
                    message.customPaddingRight = reader.int32();
                    break;
                case 86:
                    message.isDeconvolution = reader.bool();
                    break;
                case 87:
                    message.outputShape = reader.array(message.outputShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Convolution3DLayerParams.prototype.outputChannels = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.inputChannels = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.nGroups = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.kernelDepth = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.kernelHeight = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.kernelWidth = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.strideDepth = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.strideHeight = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.strideWidth = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.dilationDepth = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.dilationHeight = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.dilationWidth = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.weights = null;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.bias = null;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.paddingType = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingFront = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingBack = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingTop = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingBottom = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingLeft = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingRight = 0;
$root.CoreML.Specification.Convolution3DLayerParams.prototype.isDeconvolution = false;

$root.CoreML.Specification.Convolution3DLayerParams.PaddingType = {
    "CUSTOM": 0,
    "VALID": 1,
    "SAME": 2
};

$root.CoreML.Specification.InnerProductLayerParams = class InnerProductLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.InnerProductLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputChannels = reader.uint64();
                    break;
                case 2:
                    message.outputChannels = reader.uint64();
                    break;
                case 10:
                    message.hasBias = reader.bool();
                    break;
                case 20:
                    message.weights = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.int8DynamicQuantize = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.InnerProductLayerParams.prototype.inputChannels = protobuf.Uint64.create(0);
$root.CoreML.Specification.InnerProductLayerParams.prototype.outputChannels = protobuf.Uint64.create(0);
$root.CoreML.Specification.InnerProductLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.InnerProductLayerParams.prototype.weights = null;
$root.CoreML.Specification.InnerProductLayerParams.prototype.bias = null;
$root.CoreML.Specification.InnerProductLayerParams.prototype.int8DynamicQuantize = false;

$root.CoreML.Specification.EmbeddingLayerParams = class EmbeddingLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.EmbeddingLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputDim = reader.uint64();
                    break;
                case 2:
                    message.outputChannels = reader.uint64();
                    break;
                case 10:
                    message.hasBias = reader.bool();
                    break;
                case 20:
                    message.weights = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.EmbeddingLayerParams.prototype.inputDim = protobuf.Uint64.create(0);
$root.CoreML.Specification.EmbeddingLayerParams.prototype.outputChannels = protobuf.Uint64.create(0);
$root.CoreML.Specification.EmbeddingLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.EmbeddingLayerParams.prototype.weights = null;
$root.CoreML.Specification.EmbeddingLayerParams.prototype.bias = null;

$root.CoreML.Specification.EmbeddingNDLayerParams = class EmbeddingNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.EmbeddingNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vocabSize = reader.uint64();
                    break;
                case 2:
                    message.embeddingSize = reader.uint64();
                    break;
                case 3:
                    message.hasBias = reader.bool();
                    break;
                case 20:
                    message.weights = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.EmbeddingNDLayerParams.prototype.vocabSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.EmbeddingNDLayerParams.prototype.embeddingSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.EmbeddingNDLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.EmbeddingNDLayerParams.prototype.weights = null;
$root.CoreML.Specification.EmbeddingNDLayerParams.prototype.bias = null;

$root.CoreML.Specification.BatchnormLayerParams = class BatchnormLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BatchnormLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.channels = reader.uint64();
                    break;
                case 5:
                    message.computeMeanVar = reader.bool();
                    break;
                case 6:
                    message.instanceNormalization = reader.bool();
                    break;
                case 10:
                    message.epsilon = reader.float();
                    break;
                case 15:
                    message.gamma = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.beta = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 17:
                    message.mean = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 18:
                    message.variance = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BatchnormLayerParams.prototype.channels = protobuf.Uint64.create(0);
$root.CoreML.Specification.BatchnormLayerParams.prototype.computeMeanVar = false;
$root.CoreML.Specification.BatchnormLayerParams.prototype.instanceNormalization = false;
$root.CoreML.Specification.BatchnormLayerParams.prototype.epsilon = 0;
$root.CoreML.Specification.BatchnormLayerParams.prototype.gamma = null;
$root.CoreML.Specification.BatchnormLayerParams.prototype.beta = null;
$root.CoreML.Specification.BatchnormLayerParams.prototype.mean = null;
$root.CoreML.Specification.BatchnormLayerParams.prototype.variance = null;

$root.CoreML.Specification.PoolingLayerParams = class PoolingLayerParams {

    constructor() {
        this.kernelSize = [];
        this.stride = [];
    }

    get PoolingPaddingType() {
        $root.CoreML.Specification.PoolingLayerParams.PoolingPaddingTypeSet = $root.CoreML.Specification.PoolingLayerParams.PoolingPaddingTypeSet || new Set([ "valid", "same", "includeLastPixel"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.PoolingLayerParams.PoolingPaddingTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PoolingLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 10:
                    message.kernelSize = reader.array(message.kernelSize, () => reader.uint64(), tag);
                    break;
                case 20:
                    message.stride = reader.array(message.stride, () => reader.uint64(), tag);
                    break;
                case 30:
                    message.valid = $root.CoreML.Specification.ValidPadding.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.same = $root.CoreML.Specification.SamePadding.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.includeLastPixel = $root.CoreML.Specification.PoolingLayerParams.ValidCompletePadding.decode(reader, reader.uint32());
                    break;
                case 50:
                    message.avgPoolExcludePadding = reader.bool();
                    break;
                case 60:
                    message.globalPooling = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PoolingLayerParams.prototype.type = 0;
$root.CoreML.Specification.PoolingLayerParams.prototype.avgPoolExcludePadding = false;
$root.CoreML.Specification.PoolingLayerParams.prototype.globalPooling = false;

$root.CoreML.Specification.PoolingLayerParams.PoolingType = {
    "MAX": 0,
    "AVERAGE": 1,
    "L2": 2
};

$root.CoreML.Specification.PoolingLayerParams.ValidCompletePadding = class ValidCompletePadding {

    constructor() {
        this.paddingAmounts = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PoolingLayerParams.ValidCompletePadding();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.paddingAmounts = reader.array(message.paddingAmounts, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Pooling3DLayerParams = class Pooling3DLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Pooling3DLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.kernelDepth = reader.int32();
                    break;
                case 3:
                    message.kernelHeight = reader.int32();
                    break;
                case 4:
                    message.kernelWidth = reader.int32();
                    break;
                case 5:
                    message.strideDepth = reader.int32();
                    break;
                case 6:
                    message.strideHeight = reader.int32();
                    break;
                case 7:
                    message.strideWidth = reader.int32();
                    break;
                case 15:
                    message.paddingType = reader.int32();
                    break;
                case 8:
                    message.customPaddingFront = reader.int32();
                    break;
                case 9:
                    message.customPaddingBack = reader.int32();
                    break;
                case 10:
                    message.customPaddingTop = reader.int32();
                    break;
                case 11:
                    message.customPaddingBottom = reader.int32();
                    break;
                case 12:
                    message.customPaddingLeft = reader.int32();
                    break;
                case 13:
                    message.customPaddingRight = reader.int32();
                    break;
                case 14:
                    message.countExcludePadding = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Pooling3DLayerParams.prototype.type = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.kernelDepth = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.kernelHeight = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.kernelWidth = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.strideDepth = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.strideHeight = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.strideWidth = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.paddingType = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingFront = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingBack = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingTop = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingBottom = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingLeft = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingRight = 0;
$root.CoreML.Specification.Pooling3DLayerParams.prototype.countExcludePadding = false;

$root.CoreML.Specification.Pooling3DLayerParams.PoolingType3D = {
    "MAX": 0,
    "AVERAGE": 1
};

$root.CoreML.Specification.Pooling3DLayerParams.Pooling3DPaddingType = {
    "CUSTOM": 0,
    "VALID": 1,
    "SAME": 2
};

$root.CoreML.Specification.GlobalPooling3DLayerParams = class GlobalPooling3DLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GlobalPooling3DLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GlobalPooling3DLayerParams.prototype.type = 0;

$root.CoreML.Specification.GlobalPooling3DLayerParams.GlobalPoolingType3D = {
    "MAX": 0,
    "AVERAGE": 1
};

$root.CoreML.Specification.PaddingLayerParams = class PaddingLayerParams {

    constructor() {
    }

    get PaddingType() {
        $root.CoreML.Specification.PaddingLayerParams.PaddingTypeSet = $root.CoreML.Specification.PaddingLayerParams.PaddingTypeSet || new Set([ "constant", "reflection", "replication"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.PaddingLayerParams.PaddingTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PaddingLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.constant = $root.CoreML.Specification.PaddingLayerParams.PaddingConstant.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.reflection = $root.CoreML.Specification.PaddingLayerParams.PaddingReflection.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.replication = $root.CoreML.Specification.PaddingLayerParams.PaddingReplication.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.paddingAmounts = $root.CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PaddingLayerParams.prototype.paddingAmounts = null;

$root.CoreML.Specification.PaddingLayerParams.PaddingConstant = class PaddingConstant {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingConstant();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PaddingLayerParams.PaddingConstant.prototype.value = 0;

$root.CoreML.Specification.PaddingLayerParams.PaddingReflection = class PaddingReflection {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReflection();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PaddingLayerParams.PaddingReplication = class PaddingReplication {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReplication();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConcatLayerParams = class ConcatLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ConcatLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 100:
                    message.sequenceConcat = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConcatLayerParams.prototype.sequenceConcat = false;

$root.CoreML.Specification.LRNLayerParams = class LRNLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LRNLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta = reader.float();
                    break;
                case 3:
                    message.localSize = reader.uint64();
                    break;
                case 4:
                    message.k = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LRNLayerParams.prototype.alpha = 0;
$root.CoreML.Specification.LRNLayerParams.prototype.beta = 0;
$root.CoreML.Specification.LRNLayerParams.prototype.localSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.LRNLayerParams.prototype.k = 0;

$root.CoreML.Specification.SoftmaxLayerParams = class SoftmaxLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SoftmaxLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SplitLayerParams = class SplitLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SplitLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nOutputs = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SplitLayerParams.prototype.nOutputs = protobuf.Uint64.create(0);

$root.CoreML.Specification.AddLayerParams = class AddLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AddLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AddLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.MultiplyLayerParams = class MultiplyLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MultiplyLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MultiplyLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.UnaryFunctionLayerParams = class UnaryFunctionLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UnaryFunctionLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.alpha = reader.float();
                    break;
                case 3:
                    message.epsilon = reader.float();
                    break;
                case 4:
                    message.shift = reader.float();
                    break;
                case 5:
                    message.scale = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.UnaryFunctionLayerParams.prototype.type = 0;
$root.CoreML.Specification.UnaryFunctionLayerParams.prototype.alpha = 0;
$root.CoreML.Specification.UnaryFunctionLayerParams.prototype.epsilon = 0;
$root.CoreML.Specification.UnaryFunctionLayerParams.prototype.shift = 0;
$root.CoreML.Specification.UnaryFunctionLayerParams.prototype.scale = 0;

$root.CoreML.Specification.UnaryFunctionLayerParams.Operation = {
    "SQRT": 0,
    "RSQRT": 1,
    "INVERSE": 2,
    "POWER": 3,
    "EXP": 4,
    "LOG": 5,
    "ABS": 6,
    "THRESHOLD": 7
};

$root.CoreML.Specification.UpsampleLayerParams = class UpsampleLayerParams {

    constructor() {
        this.scalingFactor = [];
        this.fractionalScalingFactor = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UpsampleLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.scalingFactor = reader.array(message.scalingFactor, () => reader.uint64(), tag);
                    break;
                case 7:
                    message.fractionalScalingFactor = reader.floats(message.fractionalScalingFactor, tag);
                    break;
                case 5:
                    message.mode = reader.int32();
                    break;
                case 6:
                    message.linearUpsampleMode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.UpsampleLayerParams.prototype.mode = 0;
$root.CoreML.Specification.UpsampleLayerParams.prototype.linearUpsampleMode = 0;

$root.CoreML.Specification.UpsampleLayerParams.InterpolationMode = {
    "NN": 0,
    "BILINEAR": 1
};

$root.CoreML.Specification.UpsampleLayerParams.LinearUpsampleMode = {
    "DEFAULT": 0,
    "ALIGN_CORNERS_TRUE": 1,
    "ALIGN_CORNERS_FALSE": 2
};

$root.CoreML.Specification.ResizeBilinearLayerParams = class ResizeBilinearLayerParams {

    constructor() {
        this.targetSize = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ResizeBilinearLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetSize = reader.array(message.targetSize, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.mode = $root.CoreML.Specification.SamplingMode.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ResizeBilinearLayerParams.prototype.mode = null;

$root.CoreML.Specification.CropResizeLayerParams = class CropResizeLayerParams {

    constructor() {
        this.targetSize = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CropResizeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetSize = reader.array(message.targetSize, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.normalizedCoordinates = reader.bool();
                    break;
                case 3:
                    message.mode = $root.CoreML.Specification.SamplingMode.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.boxIndicesMode = $root.CoreML.Specification.BoxCoordinatesMode.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.spatialScale = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CropResizeLayerParams.prototype.normalizedCoordinates = false;
$root.CoreML.Specification.CropResizeLayerParams.prototype.mode = null;
$root.CoreML.Specification.CropResizeLayerParams.prototype.boxIndicesMode = null;
$root.CoreML.Specification.CropResizeLayerParams.prototype.spatialScale = 0;

$root.CoreML.Specification.BiasLayerParams = class BiasLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BiasLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BiasLayerParams.prototype.bias = null;

$root.CoreML.Specification.ScaleLayerParams = class ScaleLayerParams {

    constructor() {
        this.shapeScale = [];
        this.shapeBias = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ScaleLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shapeScale = reader.array(message.shapeScale, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.scale = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.hasBias = reader.bool();
                    break;
                case 4:
                    message.shapeBias = reader.array(message.shapeBias, () => reader.uint64(), tag);
                    break;
                case 5:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ScaleLayerParams.prototype.scale = null;
$root.CoreML.Specification.ScaleLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.ScaleLayerParams.prototype.bias = null;

$root.CoreML.Specification.LoadConstantLayerParams = class LoadConstantLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoadConstantLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.data = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LoadConstantLayerParams.prototype.data = null;

$root.CoreML.Specification.L2NormalizeLayerParams = class L2NormalizeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.L2NormalizeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.epsilon = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.L2NormalizeLayerParams.prototype.epsilon = 0;

$root.CoreML.Specification.FlattenLayerParams = class FlattenLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FlattenLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FlattenLayerParams.prototype.mode = 0;

$root.CoreML.Specification.FlattenLayerParams.FlattenOrder = {
    "CHANNEL_FIRST": 0,
    "CHANNEL_LAST": 1
};

$root.CoreML.Specification.ReshapeLayerParams = class ReshapeLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetShape = reader.array(message.targetShape, () => reader.int64(), tag);
                    break;
                case 2:
                    message.mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReshapeLayerParams.prototype.mode = 0;

$root.CoreML.Specification.ReshapeLayerParams.ReshapeOrder = {
    "CHANNEL_FIRST": 0,
    "CHANNEL_LAST": 1
};

$root.CoreML.Specification.PermuteLayerParams = class PermuteLayerParams {

    constructor() {
        this.axis = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PermuteLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.array(message.axis, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReorganizeDataLayerParams = class ReorganizeDataLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReorganizeDataLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mode = reader.int32();
                    break;
                case 2:
                    message.blockSize = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReorganizeDataLayerParams.prototype.mode = 0;
$root.CoreML.Specification.ReorganizeDataLayerParams.prototype.blockSize = protobuf.Uint64.create(0);

$root.CoreML.Specification.ReorganizeDataLayerParams.ReorganizationType = {
    "SPACE_TO_DEPTH": 0,
    "DEPTH_TO_SPACE": 1,
    "PIXEL_SHUFFLE": 2
};

$root.CoreML.Specification.SliceLayerParams = class SliceLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SliceLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.startIndex = reader.int64();
                    break;
                case 2:
                    message.endIndex = reader.int64();
                    break;
                case 3:
                    message.stride = reader.uint64();
                    break;
                case 4:
                    message.axis = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SliceLayerParams.prototype.startIndex = protobuf.Int64.create(0);
$root.CoreML.Specification.SliceLayerParams.prototype.endIndex = protobuf.Int64.create(0);
$root.CoreML.Specification.SliceLayerParams.prototype.stride = protobuf.Uint64.create(0);
$root.CoreML.Specification.SliceLayerParams.prototype.axis = 0;

$root.CoreML.Specification.SliceLayerParams.SliceAxis = {
    "CHANNEL_AXIS": 0,
    "HEIGHT_AXIS": 1,
    "WIDTH_AXIS": 2
};

$root.CoreML.Specification.ReduceLayerParams = class ReduceLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mode = reader.int32();
                    break;
                case 2:
                    message.epsilon = reader.float();
                    break;
                case 3:
                    message.axis = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceLayerParams.prototype.mode = 0;
$root.CoreML.Specification.ReduceLayerParams.prototype.epsilon = 0;
$root.CoreML.Specification.ReduceLayerParams.prototype.axis = 0;

$root.CoreML.Specification.ReduceLayerParams.ReduceOperation = {
    "SUM": 0,
    "AVG": 1,
    "PROD": 2,
    "LOGSUM": 3,
    "SUMSQUARE": 4,
    "L1": 5,
    "L2": 6,
    "MAX": 7,
    "MIN": 8,
    "ARGMAX": 9
};

$root.CoreML.Specification.ReduceLayerParams.ReduceAxis = {
    "CHW": 0,
    "HW": 1,
    "C": 2,
    "H": 3,
    "W": 4
};

$root.CoreML.Specification.CropLayerParams = class CropLayerParams {

    constructor() {
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CropLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.cropAmounts = $root.CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.offset = reader.array(message.offset, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CropLayerParams.prototype.cropAmounts = null;

$root.CoreML.Specification.AverageLayerParams = class AverageLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AverageLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MaxLayerParams = class MaxLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MaxLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MinLayerParams = class MinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MinLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DotProductLayerParams = class DotProductLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DotProductLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.cosineSimilarity = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DotProductLayerParams.prototype.cosineSimilarity = false;

$root.CoreML.Specification.MeanVarianceNormalizeLayerParams = class MeanVarianceNormalizeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MeanVarianceNormalizeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.acrossChannels = reader.bool();
                    break;
                case 2:
                    message.normalizeVariance = reader.bool();
                    break;
                case 3:
                    message.epsilon = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.acrossChannels = false;
$root.CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.normalizeVariance = false;
$root.CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.epsilon = 0;

$root.CoreML.Specification.SequenceRepeatLayerParams = class SequenceRepeatLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SequenceRepeatLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nRepetitions = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SequenceRepeatLayerParams.prototype.nRepetitions = protobuf.Uint64.create(0);

$root.CoreML.Specification.SimpleRecurrentLayerParams = class SimpleRecurrentLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SimpleRecurrentLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activation = $root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.sequenceOutput = reader.bool();
                    break;
                case 20:
                    message.hasBiasVector = reader.bool();
                    break;
                case 30:
                    message.weightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.recursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.biasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.reverseInput = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.inputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.outputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.activation = null;
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.sequenceOutput = false;
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.hasBiasVector = false;
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.weightMatrix = null;
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.recursionMatrix = null;
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.biasVector = null;
$root.CoreML.Specification.SimpleRecurrentLayerParams.prototype.reverseInput = false;

$root.CoreML.Specification.GRULayerParams = class GRULayerParams {

    constructor() {
        this.activations = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GRULayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activations.push($root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.sequenceOutput = reader.bool();
                    break;
                case 20:
                    message.hasBiasVectors = reader.bool();
                    break;
                case 30:
                    message.updateGateWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.resetGateWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.outputGateWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 50:
                    message.updateGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 51:
                    message.resetGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 52:
                    message.outputGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 70:
                    message.updateGateBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 71:
                    message.resetGateBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 72:
                    message.outputGateBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.reverseInput = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GRULayerParams.prototype.inputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.GRULayerParams.prototype.outputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.GRULayerParams.prototype.sequenceOutput = false;
$root.CoreML.Specification.GRULayerParams.prototype.hasBiasVectors = false;
$root.CoreML.Specification.GRULayerParams.prototype.updateGateWeightMatrix = null;
$root.CoreML.Specification.GRULayerParams.prototype.resetGateWeightMatrix = null;
$root.CoreML.Specification.GRULayerParams.prototype.outputGateWeightMatrix = null;
$root.CoreML.Specification.GRULayerParams.prototype.updateGateRecursionMatrix = null;
$root.CoreML.Specification.GRULayerParams.prototype.resetGateRecursionMatrix = null;
$root.CoreML.Specification.GRULayerParams.prototype.outputGateRecursionMatrix = null;
$root.CoreML.Specification.GRULayerParams.prototype.updateGateBiasVector = null;
$root.CoreML.Specification.GRULayerParams.prototype.resetGateBiasVector = null;
$root.CoreML.Specification.GRULayerParams.prototype.outputGateBiasVector = null;
$root.CoreML.Specification.GRULayerParams.prototype.reverseInput = false;

$root.CoreML.Specification.LSTMParams = class LSTMParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LSTMParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.sequenceOutput = reader.bool();
                    break;
                case 20:
                    message.hasBiasVectors = reader.bool();
                    break;
                case 30:
                    message.forgetBias = reader.bool();
                    break;
                case 40:
                    message.hasPeepholeVectors = reader.bool();
                    break;
                case 50:
                    message.coupledInputAndForgetGate = reader.bool();
                    break;
                case 60:
                    message.cellClipThreshold = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LSTMParams.prototype.sequenceOutput = false;
$root.CoreML.Specification.LSTMParams.prototype.hasBiasVectors = false;
$root.CoreML.Specification.LSTMParams.prototype.forgetBias = false;
$root.CoreML.Specification.LSTMParams.prototype.hasPeepholeVectors = false;
$root.CoreML.Specification.LSTMParams.prototype.coupledInputAndForgetGate = false;
$root.CoreML.Specification.LSTMParams.prototype.cellClipThreshold = 0;

$root.CoreML.Specification.LSTMWeightParams = class LSTMWeightParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LSTMWeightParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputGateWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.forgetGateWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.blockInputWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.outputGateWeightMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.inputGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.forgetGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.blockInputRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 23:
                    message.outputGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.inputGateBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.forgetGateBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 42:
                    message.blockInputBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 43:
                    message.outputGateBiasVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 60:
                    message.inputGatePeepholeVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 61:
                    message.forgetGatePeepholeVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 62:
                    message.outputGatePeepholeVector = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LSTMWeightParams.prototype.inputGateWeightMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.forgetGateWeightMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.blockInputWeightMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.outputGateWeightMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.inputGateRecursionMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.forgetGateRecursionMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.blockInputRecursionMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.outputGateRecursionMatrix = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.inputGateBiasVector = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.forgetGateBiasVector = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.blockInputBiasVector = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.outputGateBiasVector = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.inputGatePeepholeVector = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.forgetGatePeepholeVector = null;
$root.CoreML.Specification.LSTMWeightParams.prototype.outputGatePeepholeVector = null;

$root.CoreML.Specification.UniDirectionalLSTMLayerParams = class UniDirectionalLSTMLayerParams {

    constructor() {
        this.activations = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UniDirectionalLSTMLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activations.push($root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.params = $root.CoreML.Specification.LSTMParams.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.weightParams = $root.CoreML.Specification.LSTMWeightParams.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.reverseInput = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.inputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.outputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.params = null;
$root.CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.weightParams = null;
$root.CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.reverseInput = false;

$root.CoreML.Specification.BiDirectionalLSTMLayerParams = class BiDirectionalLSTMLayerParams {

    constructor() {
        this.activationsForwardLSTM = [];
        this.activationsBackwardLSTM = [];
        this.weightParams = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BiDirectionalLSTMLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activationsForwardLSTM.push($root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.activationsBackwardLSTM.push($root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.params = $root.CoreML.Specification.LSTMParams.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.weightParams.push($root.CoreML.Specification.LSTMWeightParams.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BiDirectionalLSTMLayerParams.prototype.inputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.BiDirectionalLSTMLayerParams.prototype.outputVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.BiDirectionalLSTMLayerParams.prototype.params = null;

$root.CoreML.Specification.CustomLayerParams = class CustomLayerParams {

    constructor() {
        this.weights = [];
        this.parameters = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CustomLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.className = reader.string();
                    break;
                case 20:
                    message.weights.push($root.CoreML.Specification.WeightParams.decode(reader, reader.uint32()));
                    break;
                case 30:
                    reader.entry(message.parameters, () => reader.string(), () => $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.decode(reader, reader.uint32()));
                    break;
                case 40:
                    message.description = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CustomLayerParams.prototype.className = "";
$root.CoreML.Specification.CustomLayerParams.prototype.description = "";

$root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue = class CustomLayerParamValue {

    constructor() {
    }

    get value() {
        $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.valueSet = $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.valueSet || new Set([ "doubleValue", "stringValue", "intValue", "longValue", "boolValue"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.doubleValue = reader.double();
                    break;
                case 20:
                    message.stringValue = reader.string();
                    break;
                case 30:
                    message.intValue = reader.int32();
                    break;
                case 40:
                    message.longValue = reader.int64();
                    break;
                case 50:
                    message.boolValue = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TransposeLayerParams = class TransposeLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TransposeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BatchedMatMulLayerParams = class BatchedMatMulLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BatchedMatMulLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.transposeA = reader.bool();
                    break;
                case 2:
                    message.transposeB = reader.bool();
                    break;
                case 5:
                    message.weightMatrixFirstDimension = reader.uint64();
                    break;
                case 6:
                    message.weightMatrixSecondDimension = reader.uint64();
                    break;
                case 7:
                    message.hasBias = reader.bool();
                    break;
                case 8:
                    message.weights = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.bias = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.int8DynamicQuantize = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.transposeA = false;
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.transposeB = false;
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.weightMatrixFirstDimension = protobuf.Uint64.create(0);
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.weightMatrixSecondDimension = protobuf.Uint64.create(0);
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.hasBias = false;
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.weights = null;
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.bias = null;
$root.CoreML.Specification.BatchedMatMulLayerParams.prototype.int8DynamicQuantize = false;

$root.CoreML.Specification.ConcatNDLayerParams = class ConcatNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ConcatNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.interleave = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConcatNDLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ConcatNDLayerParams.prototype.interleave = false;

$root.CoreML.Specification.SoftmaxNDLayerParams = class SoftmaxNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SoftmaxNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SoftmaxNDLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ReverseLayerParams = class ReverseLayerParams {

    constructor() {
        this.reverseDim = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReverseLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.reverseDim = reader.array(message.reverseDim, () => reader.bool(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReverseSeqLayerParams = class ReverseSeqLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReverseSeqLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.batchAxis = reader.int64();
                    break;
                case 2:
                    message.sequenceAxis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReverseSeqLayerParams.prototype.batchAxis = protobuf.Int64.create(0);
$root.CoreML.Specification.ReverseSeqLayerParams.prototype.sequenceAxis = protobuf.Int64.create(0);

$root.CoreML.Specification.LoadConstantNDLayerParams = class LoadConstantNDLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoadConstantNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.data = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LoadConstantNDLayerParams.prototype.data = null;

$root.CoreML.Specification.FillLikeLayerParams = class FillLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FillLikeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FillLikeLayerParams.prototype.value = 0;

$root.CoreML.Specification.FillStaticLayerParams = class FillStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FillStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.float();
                    break;
                case 2:
                    message.targetShape = reader.array(message.targetShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FillStaticLayerParams.prototype.value = 0;

$root.CoreML.Specification.FillDynamicLayerParams = class FillDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FillDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FillDynamicLayerParams.prototype.value = 0;

$root.CoreML.Specification.WhereBroadcastableLayerParams = class WhereBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.WhereBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SinLayerParams = class SinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SinLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CosLayerParams = class CosLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CosLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TanLayerParams = class TanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TanLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AsinLayerParams = class AsinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AsinLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AcosLayerParams = class AcosLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AcosLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AtanLayerParams = class AtanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AtanLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SinhLayerParams = class SinhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SinhLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoshLayerParams = class CoshLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoshLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TanhLayerParams = class TanhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TanhLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AsinhLayerParams = class AsinhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AsinhLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AcoshLayerParams = class AcoshLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AcoshLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AtanhLayerParams = class AtanhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AtanhLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PowBroadcastableLayerParams = class PowBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PowBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Exp2LayerParams = class Exp2LayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Exp2LayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.WhereNonZeroLayerParams = class WhereNonZeroLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.WhereNonZeroLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MatrixBandPartLayerParams = class MatrixBandPartLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MatrixBandPartLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numLower = reader.int64();
                    break;
                case 2:
                    message.numUpper = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MatrixBandPartLayerParams.prototype.numLower = protobuf.Int64.create(0);
$root.CoreML.Specification.MatrixBandPartLayerParams.prototype.numUpper = protobuf.Int64.create(0);

$root.CoreML.Specification.UpperTriangularLayerParams = class UpperTriangularLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UpperTriangularLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.UpperTriangularLayerParams.prototype.k = protobuf.Int64.create(0);

$root.CoreML.Specification.LowerTriangularLayerParams = class LowerTriangularLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LowerTriangularLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LowerTriangularLayerParams.prototype.k = protobuf.Int64.create(0);

$root.CoreML.Specification.BroadcastToLikeLayerParams = class BroadcastToLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BroadcastToLikeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BroadcastToStaticLayerParams = class BroadcastToStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BroadcastToStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetShape = reader.array(message.targetShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.BroadcastToDynamicLayerParams = class BroadcastToDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BroadcastToDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AddBroadcastableLayerParams = class AddBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AddBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MaxBroadcastableLayerParams = class MaxBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MaxBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MinBroadcastableLayerParams = class MinBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MinBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ModBroadcastableLayerParams = class ModBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ModBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FloorDivBroadcastableLayerParams = class FloorDivBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FloorDivBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SubtractBroadcastableLayerParams = class SubtractBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SubtractBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MultiplyBroadcastableLayerParams = class MultiplyBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MultiplyBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DivideBroadcastableLayerParams = class DivideBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DivideBroadcastableLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GatherLayerParams = class GatherLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GatherLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GatherLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ScatterMode = {
    "SCATTER_UPDATE": 0,
    "SCATTER_ADD": 1,
    "SCATTER_SUB": 2,
    "SCATTER_MUL": 3,
    "SCATTER_DIV": 4,
    "SCATTER_MAX": 5,
    "SCATTER_MIN": 6
};

$root.CoreML.Specification.ScatterLayerParams = class ScatterLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ScatterLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ScatterLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ScatterLayerParams.prototype.mode = 0;

$root.CoreML.Specification.GatherNDLayerParams = class GatherNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GatherNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ScatterNDLayerParams = class ScatterNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ScatterNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ScatterNDLayerParams.prototype.mode = 0;

$root.CoreML.Specification.GatherAlongAxisLayerParams = class GatherAlongAxisLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GatherAlongAxisLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GatherAlongAxisLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ScatterAlongAxisLayerParams = class ScatterAlongAxisLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ScatterAlongAxisLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ScatterAlongAxisLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ScatterAlongAxisLayerParams.prototype.mode = 0;

$root.CoreML.Specification.StackLayerParams = class StackLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StackLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.StackLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.RankPreservingReshapeLayerParams = class RankPreservingReshapeLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RankPreservingReshapeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetShape = reader.array(message.targetShape, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConstantPaddingLayerParams = class ConstantPaddingLayerParams {

    constructor() {
        this.padAmounts = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ConstantPaddingLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.float();
                    break;
                case 2:
                    message.padAmounts = reader.array(message.padAmounts, () => reader.uint64(), tag);
                    break;
                case 3:
                    message.padToGivenOutputSizeMode = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ConstantPaddingLayerParams.prototype.value = 0;
$root.CoreML.Specification.ConstantPaddingLayerParams.prototype.padToGivenOutputSizeMode = false;

$root.CoreML.Specification.RandomNormalLikeLayerParams = class RandomNormalLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomNormalLikeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.mean = reader.float();
                    break;
                case 3:
                    message.stdDev = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomNormalLikeLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomNormalLikeLayerParams.prototype.mean = 0;
$root.CoreML.Specification.RandomNormalLikeLayerParams.prototype.stdDev = 0;

$root.CoreML.Specification.RandomNormalStaticLayerParams = class RandomNormalStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomNormalStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.mean = reader.float();
                    break;
                case 3:
                    message.stdDev = reader.float();
                    break;
                case 4:
                    message.outputShape = reader.array(message.outputShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomNormalStaticLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomNormalStaticLayerParams.prototype.mean = 0;
$root.CoreML.Specification.RandomNormalStaticLayerParams.prototype.stdDev = 0;

$root.CoreML.Specification.RandomNormalDynamicLayerParams = class RandomNormalDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomNormalDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.mean = reader.float();
                    break;
                case 3:
                    message.stdDev = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomNormalDynamicLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomNormalDynamicLayerParams.prototype.mean = 0;
$root.CoreML.Specification.RandomNormalDynamicLayerParams.prototype.stdDev = 0;

$root.CoreML.Specification.RandomUniformLikeLayerParams = class RandomUniformLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomUniformLikeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.minVal = reader.float();
                    break;
                case 3:
                    message.maxVal = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomUniformLikeLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomUniformLikeLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.RandomUniformLikeLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.RandomUniformStaticLayerParams = class RandomUniformStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomUniformStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.minVal = reader.float();
                    break;
                case 3:
                    message.maxVal = reader.float();
                    break;
                case 4:
                    message.outputShape = reader.array(message.outputShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomUniformStaticLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomUniformStaticLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.RandomUniformStaticLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.RandomUniformDynamicLayerParams = class RandomUniformDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomUniformDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.minVal = reader.float();
                    break;
                case 3:
                    message.maxVal = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomUniformDynamicLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomUniformDynamicLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.RandomUniformDynamicLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.RandomBernoulliLikeLayerParams = class RandomBernoulliLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomBernoulliLikeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.prob = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomBernoulliLikeLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomBernoulliLikeLayerParams.prototype.prob = 0;

$root.CoreML.Specification.RandomBernoulliStaticLayerParams = class RandomBernoulliStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomBernoulliStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.prob = reader.float();
                    break;
                case 3:
                    message.outputShape = reader.array(message.outputShape, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomBernoulliStaticLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomBernoulliStaticLayerParams.prototype.prob = 0;

$root.CoreML.Specification.RandomBernoulliDynamicLayerParams = class RandomBernoulliDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomBernoulliDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.prob = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RandomBernoulliDynamicLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomBernoulliDynamicLayerParams.prototype.prob = 0;

$root.CoreML.Specification.CategoricalDistributionLayerParams = class CategoricalDistributionLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CategoricalDistributionLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.seed = reader.int64();
                    break;
                case 2:
                    message.numSamples = reader.int64();
                    break;
                case 3:
                    message.isLogits = reader.bool();
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.temperature = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CategoricalDistributionLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.CategoricalDistributionLayerParams.prototype.numSamples = protobuf.Int64.create(0);
$root.CoreML.Specification.CategoricalDistributionLayerParams.prototype.isLogits = false;
$root.CoreML.Specification.CategoricalDistributionLayerParams.prototype.eps = 0;
$root.CoreML.Specification.CategoricalDistributionLayerParams.prototype.temperature = 0;

$root.CoreML.Specification.ReduceL1LayerParams = class ReduceL1LayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceL1LayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceL1LayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceL1LayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceL2LayerParams = class ReduceL2LayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceL2LayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceL2LayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceL2LayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceMaxLayerParams = class ReduceMaxLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceMaxLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceMaxLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceMaxLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceMinLayerParams = class ReduceMinLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceMinLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceMinLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceMinLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceSumLayerParams = class ReduceSumLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceSumLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceSumLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceSumLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceProdLayerParams = class ReduceProdLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceProdLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceProdLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceProdLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceMeanLayerParams = class ReduceMeanLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceMeanLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceMeanLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceMeanLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceLogSumLayerParams = class ReduceLogSumLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceLogSumLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceLogSumLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceLogSumLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceSumSquareLayerParams = class ReduceSumSquareLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceSumSquareLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceSumSquareLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceSumSquareLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceLogSumExpLayerParams = class ReduceLogSumExpLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceLogSumExpLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keepDims = reader.bool();
                    break;
                case 3:
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReduceLogSumExpLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceLogSumExpLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ExpandDimsLayerParams = class ExpandDimsLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ExpandDimsLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FlattenTo2DLayerParams = class FlattenTo2DLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FlattenTo2DLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FlattenTo2DLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ReshapeStaticLayerParams = class ReshapeStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetShape = reader.array(message.targetShape, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReshapeLikeLayerParams = class ReshapeLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeLikeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ReshapeDynamicLayerParams = class ReshapeDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SqueezeLayerParams = class SqueezeLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SqueezeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.squeezeAll = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SqueezeLayerParams.prototype.squeezeAll = false;

$root.CoreML.Specification.TopKLayerParams = class TopKLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TopKLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.K = reader.uint64();
                    break;
                case 3:
                    message.useBottomK = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TopKLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.TopKLayerParams.prototype.K = protobuf.Uint64.create(0);
$root.CoreML.Specification.TopKLayerParams.prototype.useBottomK = false;

$root.CoreML.Specification.ArgMaxLayerParams = class ArgMaxLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArgMaxLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.removeDim = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ArgMaxLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ArgMaxLayerParams.prototype.removeDim = false;

$root.CoreML.Specification.ArgMinLayerParams = class ArgMinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArgMinLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.removeDim = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ArgMinLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ArgMinLayerParams.prototype.removeDim = false;

$root.CoreML.Specification.SplitNDLayerParams = class SplitNDLayerParams {

    constructor() {
        this.splitSizes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SplitNDLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.numSplits = reader.uint64();
                    break;
                case 3:
                    message.splitSizes = reader.array(message.splitSizes, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SplitNDLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.SplitNDLayerParams.prototype.numSplits = protobuf.Uint64.create(0);

$root.CoreML.Specification.CeilLayerParams = class CeilLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CeilLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RoundLayerParams = class RoundLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RoundLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.FloorLayerParams = class FloorLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FloorLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SignLayerParams = class SignLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SignLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ClipLayerParams = class ClipLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ClipLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.minVal = reader.float();
                    break;
                case 2:
                    message.maxVal = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ClipLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.ClipLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.SliceStaticLayerParams = class SliceStaticLayerParams {

    constructor() {
        this.beginIds = [];
        this.beginMasks = [];
        this.endIds = [];
        this.endMasks = [];
        this.strides = [];
        this.squeezeMasks = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SliceStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.beginIds = reader.array(message.beginIds, () => reader.int64(), tag);
                    break;
                case 2:
                    message.beginMasks = reader.array(message.beginMasks, () => reader.bool(), tag);
                    break;
                case 3:
                    message.endIds = reader.array(message.endIds, () => reader.int64(), tag);
                    break;
                case 4:
                    message.endMasks = reader.array(message.endMasks, () => reader.bool(), tag);
                    break;
                case 5:
                    message.strides = reader.array(message.strides, () => reader.int64(), tag);
                    break;
                case 6:
                    message.squeezeMasks = reader.array(message.squeezeMasks, () => reader.bool(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SliceDynamicLayerParams = class SliceDynamicLayerParams {

    constructor() {
        this.beginMasks = [];
        this.endIds = [];
        this.endMasks = [];
        this.strides = [];
        this.squeezeMasks = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SliceDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.beginMasks = reader.array(message.beginMasks, () => reader.bool(), tag);
                    break;
                case 3:
                    message.endIds = reader.array(message.endIds, () => reader.int64(), tag);
                    break;
                case 4:
                    message.endMasks = reader.array(message.endMasks, () => reader.bool(), tag);
                    break;
                case 5:
                    message.strides = reader.array(message.strides, () => reader.int64(), tag);
                    break;
                case 6:
                    message.squeezeMasks = reader.array(message.squeezeMasks, () => reader.bool(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TileLayerParams = class TileLayerParams {

    constructor() {
        this.reps = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TileLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.reps = reader.array(message.reps, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GetShapeLayerParams = class GetShapeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GetShapeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ErfLayerParams = class ErfLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ErfLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GeluLayerParams = class GeluLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GeluLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.GeluLayerParams.prototype.mode = 0;

$root.CoreML.Specification.GeluLayerParams.GeluMode = {
    "EXACT": 0,
    "TANH_APPROXIMATION": 1,
    "SIGMOID_APPROXIMATION": 2
};

$root.CoreML.Specification.RangeStaticLayerParams = class RangeStaticLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RangeStaticLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.endValue = reader.float();
                    break;
                case 2:
                    message.startValue = reader.float();
                    break;
                case 3:
                    message.stepSizeValue = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RangeStaticLayerParams.prototype.endValue = 0;
$root.CoreML.Specification.RangeStaticLayerParams.prototype.startValue = 0;
$root.CoreML.Specification.RangeStaticLayerParams.prototype.stepSizeValue = 0;

$root.CoreML.Specification.RangeDynamicLayerParams = class RangeDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RangeDynamicLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.startValue = reader.float();
                    break;
                case 3:
                    message.stepSizeValue = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RangeDynamicLayerParams.prototype.startValue = 0;
$root.CoreML.Specification.RangeDynamicLayerParams.prototype.stepSizeValue = 0;

$root.CoreML.Specification.SlidingWindowsLayerParams = class SlidingWindowsLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SlidingWindowsLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.windowSize = reader.uint64();
                    break;
                case 3:
                    message.step = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SlidingWindowsLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.SlidingWindowsLayerParams.prototype.windowSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.SlidingWindowsLayerParams.prototype.step = protobuf.Uint64.create(0);

$root.CoreML.Specification.LayerNormalizationLayerParams = class LayerNormalizationLayerParams {

    constructor() {
        this.normalizedShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LayerNormalizationLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.normalizedShape = reader.array(message.normalizedShape, () => reader.int64(), tag);
                    break;
                case 2:
                    message.eps = reader.float();
                    break;
                case 3:
                    message.gamma = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.beta = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LayerNormalizationLayerParams.prototype.eps = 0;
$root.CoreML.Specification.LayerNormalizationLayerParams.prototype.gamma = null;
$root.CoreML.Specification.LayerNormalizationLayerParams.prototype.beta = null;

$root.CoreML.Specification.NonMaximumSuppressionLayerParams = class NonMaximumSuppressionLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NonMaximumSuppressionLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.iouThreshold = reader.float();
                    break;
                case 2:
                    message.scoreThreshold = reader.float();
                    break;
                case 3:
                    message.maxBoxes = reader.uint64();
                    break;
                case 4:
                    message.perClassSuppression = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.iouThreshold = 0;
$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.scoreThreshold = 0;
$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.maxBoxes = protobuf.Uint64.create(0);
$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.perClassSuppression = false;

$root.CoreML.Specification.ClampedReLULayerParams = class ClampedReLULayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ClampedReLULayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ClampedReLULayerParams.prototype.alpha = 0;
$root.CoreML.Specification.ClampedReLULayerParams.prototype.beta = 0;

$root.CoreML.Specification.ArgSortLayerParams = class ArgSortLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArgSortLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.descending = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ArgSortLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ArgSortLayerParams.prototype.descending = false;

$root.CoreML.Specification.SliceBySizeLayerParams = class SliceBySizeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SliceBySizeLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.size = reader.int64();
                    break;
                case 3:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SliceBySizeLayerParams.prototype.size = protobuf.Int64.create(0);
$root.CoreML.Specification.SliceBySizeLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.NeuralNetworkClassifier = class NeuralNetworkClassifier {

    constructor() {
        this.layers = [];
        this.preprocessing = [];
    }

    get ClassLabels() {
        $root.CoreML.Specification.NeuralNetworkClassifier.ClassLabelsSet = $root.CoreML.Specification.NeuralNetworkClassifier.ClassLabelsSet || new Set([ "stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NeuralNetworkClassifier.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.preprocessing.push($root.CoreML.Specification.NeuralNetworkPreprocessing.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.arrayInputShapeMapping = reader.int32();
                    break;
                case 6:
                    message.imageInputShapeMapping = reader.int32();
                    break;
                case 10:
                    message.updateParams = $root.CoreML.Specification.NetworkUpdateParameters.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.labelProbabilityLayerName = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkClassifier.prototype.arrayInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetworkClassifier.prototype.imageInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetworkClassifier.prototype.updateParams = null;
$root.CoreML.Specification.NeuralNetworkClassifier.prototype.labelProbabilityLayerName = "";

$root.CoreML.Specification.OneHotLayerParams = class OneHotLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.OneHotLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.oneHotVectorSize = reader.uint64();
                    break;
                case 2:
                    message.axis = reader.int64();
                    break;
                case 3:
                    message.onValue = reader.float();
                    break;
                case 4:
                    message.offValue = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.OneHotLayerParams.prototype.oneHotVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.OneHotLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.OneHotLayerParams.prototype.onValue = 0;
$root.CoreML.Specification.OneHotLayerParams.prototype.offValue = 0;

$root.CoreML.Specification.CumSumLayerParams = class CumSumLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CumSumLayerParams();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.excludeFinalSum = reader.bool();
                    break;
                case 3:
                    message.reverse = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CumSumLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.CumSumLayerParams.prototype.excludeFinalSum = false;
$root.CoreML.Specification.CumSumLayerParams.prototype.reverse = false;

$root.CoreML.Specification.NeuralNetworkRegressor = class NeuralNetworkRegressor {

    constructor() {
        this.layers = [];
        this.preprocessing = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkRegressor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.preprocessing.push($root.CoreML.Specification.NeuralNetworkPreprocessing.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.arrayInputShapeMapping = reader.int32();
                    break;
                case 6:
                    message.imageInputShapeMapping = reader.int32();
                    break;
                case 10:
                    message.updateParams = $root.CoreML.Specification.NetworkUpdateParameters.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NeuralNetworkRegressor.prototype.arrayInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetworkRegressor.prototype.imageInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetworkRegressor.prototype.updateParams = null;

$root.CoreML.Specification.NetworkUpdateParameters = class NetworkUpdateParameters {

    constructor() {
        this.lossLayers = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NetworkUpdateParameters();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lossLayers.push($root.CoreML.Specification.LossLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.optimizer = $root.CoreML.Specification.Optimizer.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.epochs = $root.CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.shuffle = $root.CoreML.Specification.BoolParameter.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.seed = $root.CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NetworkUpdateParameters.prototype.optimizer = null;
$root.CoreML.Specification.NetworkUpdateParameters.prototype.epochs = null;
$root.CoreML.Specification.NetworkUpdateParameters.prototype.shuffle = null;
$root.CoreML.Specification.NetworkUpdateParameters.prototype.seed = null;

$root.CoreML.Specification.LossLayer = class LossLayer {

    constructor() {
    }

    get LossLayerType() {
        $root.CoreML.Specification.LossLayer.LossLayerTypeSet = $root.CoreML.Specification.LossLayer.LossLayerTypeSet || new Set([ "categoricalCrossEntropyLossLayer", "meanSquaredErrorLossLayer"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.LossLayer.LossLayerTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LossLayer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 10:
                    message.categoricalCrossEntropyLossLayer = $root.CoreML.Specification.CategoricalCrossEntropyLossLayer.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.meanSquaredErrorLossLayer = $root.CoreML.Specification.MeanSquaredErrorLossLayer.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LossLayer.prototype.name = "";

$root.CoreML.Specification.CategoricalCrossEntropyLossLayer = class CategoricalCrossEntropyLossLayer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CategoricalCrossEntropyLossLayer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input = reader.string();
                    break;
                case 2:
                    message.target = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CategoricalCrossEntropyLossLayer.prototype.input = "";
$root.CoreML.Specification.CategoricalCrossEntropyLossLayer.prototype.target = "";

$root.CoreML.Specification.MeanSquaredErrorLossLayer = class MeanSquaredErrorLossLayer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MeanSquaredErrorLossLayer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input = reader.string();
                    break;
                case 2:
                    message.target = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MeanSquaredErrorLossLayer.prototype.input = "";
$root.CoreML.Specification.MeanSquaredErrorLossLayer.prototype.target = "";

$root.CoreML.Specification.Optimizer = class Optimizer {

    constructor() {
    }

    get OptimizerType() {
        $root.CoreML.Specification.Optimizer.OptimizerTypeSet = $root.CoreML.Specification.Optimizer.OptimizerTypeSet || new Set([ "sgdOptimizer", "adamOptimizer"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Optimizer.OptimizerTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Optimizer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.sgdOptimizer = $root.CoreML.Specification.SGDOptimizer.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.adamOptimizer = $root.CoreML.Specification.AdamOptimizer.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SGDOptimizer = class SGDOptimizer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SGDOptimizer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.learningRate = $root.CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.miniBatchSize = $root.CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.momentum = $root.CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SGDOptimizer.prototype.learningRate = null;
$root.CoreML.Specification.SGDOptimizer.prototype.miniBatchSize = null;
$root.CoreML.Specification.SGDOptimizer.prototype.momentum = null;

$root.CoreML.Specification.AdamOptimizer = class AdamOptimizer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AdamOptimizer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.learningRate = $root.CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.miniBatchSize = $root.CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.beta1 = $root.CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.beta2 = $root.CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.eps = $root.CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.AdamOptimizer.prototype.learningRate = null;
$root.CoreML.Specification.AdamOptimizer.prototype.miniBatchSize = null;
$root.CoreML.Specification.AdamOptimizer.prototype.beta1 = null;
$root.CoreML.Specification.AdamOptimizer.prototype.beta2 = null;
$root.CoreML.Specification.AdamOptimizer.prototype.eps = null;

$root.CoreML.Specification.Normalizer = class Normalizer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Normalizer();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.normType = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Normalizer.prototype.normType = 0;

$root.CoreML.Specification.Normalizer.NormType = {
    "LMax": 0,
    "L1": 1,
    "L2": 2
};

$root.CoreML.Specification.OneHotEncoder = class OneHotEncoder {

    constructor() {
    }

    get CategoryType() {
        $root.CoreML.Specification.OneHotEncoder.CategoryTypeSet = $root.CoreML.Specification.OneHotEncoder.CategoryTypeSet || new Set([ "stringCategories", "int64Categories"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.OneHotEncoder.CategoryTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.OneHotEncoder();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.stringCategories = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.int64Categories = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.outputSparse = reader.bool();
                    break;
                case 11:
                    message.handleUnknown = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.OneHotEncoder.prototype.outputSparse = false;
$root.CoreML.Specification.OneHotEncoder.prototype.handleUnknown = 0;

$root.CoreML.Specification.OneHotEncoder.HandleUnknown = {
    "ErrorOnUnknown": 0,
    "IgnoreUnknown": 1
};

$root.CoreML.Specification.Scaler = class Scaler {

    constructor() {
        this.shiftValue = [];
        this.scaleValue = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Scaler();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shiftValue = reader.doubles(message.shiftValue, tag);
                    break;
                case 2:
                    message.scaleValue = reader.doubles(message.scaleValue, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NonMaximumSuppression = class NonMaximumSuppression {

    constructor() {
    }

    get SuppressionMethod() {
        $root.CoreML.Specification.NonMaximumSuppression.SuppressionMethodSet = $root.CoreML.Specification.NonMaximumSuppression.SuppressionMethodSet || new Set([ "pickTop"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NonMaximumSuppression.SuppressionMethodSet.has(key) && this[key] != null);
    }

    get ClassLabels() {
        $root.CoreML.Specification.NonMaximumSuppression.ClassLabelsSet = $root.CoreML.Specification.NonMaximumSuppression.ClassLabelsSet || new Set([ "stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NonMaximumSuppression.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NonMaximumSuppression();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pickTop = $root.CoreML.Specification.NonMaximumSuppression.PickTop.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.iouThreshold = reader.double();
                    break;
                case 111:
                    message.confidenceThreshold = reader.double();
                    break;
                case 200:
                    message.confidenceInputFeatureName = reader.string();
                    break;
                case 201:
                    message.coordinatesInputFeatureName = reader.string();
                    break;
                case 202:
                    message.iouThresholdInputFeatureName = reader.string();
                    break;
                case 203:
                    message.confidenceThresholdInputFeatureName = reader.string();
                    break;
                case 210:
                    message.confidenceOutputFeatureName = reader.string();
                    break;
                case 211:
                    message.coordinatesOutputFeatureName = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NonMaximumSuppression.prototype.iouThreshold = 0;
$root.CoreML.Specification.NonMaximumSuppression.prototype.confidenceThreshold = 0;
$root.CoreML.Specification.NonMaximumSuppression.prototype.confidenceInputFeatureName = "";
$root.CoreML.Specification.NonMaximumSuppression.prototype.coordinatesInputFeatureName = "";
$root.CoreML.Specification.NonMaximumSuppression.prototype.iouThresholdInputFeatureName = "";
$root.CoreML.Specification.NonMaximumSuppression.prototype.confidenceThresholdInputFeatureName = "";
$root.CoreML.Specification.NonMaximumSuppression.prototype.confidenceOutputFeatureName = "";
$root.CoreML.Specification.NonMaximumSuppression.prototype.coordinatesOutputFeatureName = "";

$root.CoreML.Specification.NonMaximumSuppression.PickTop = class PickTop {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NonMaximumSuppression.PickTop();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.perClass = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.NonMaximumSuppression.PickTop.prototype.perClass = false;

$root.CoreML.Specification.LinearKernel = class LinearKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinearKernel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RBFKernel = class RBFKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RBFKernel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.gamma = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.RBFKernel.prototype.gamma = 0;

$root.CoreML.Specification.PolyKernel = class PolyKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PolyKernel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.degree = reader.int32();
                    break;
                case 2:
                    message.c = reader.double();
                    break;
                case 3:
                    message.gamma = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PolyKernel.prototype.degree = 0;
$root.CoreML.Specification.PolyKernel.prototype.c = 0;
$root.CoreML.Specification.PolyKernel.prototype.gamma = 0;

$root.CoreML.Specification.SigmoidKernel = class SigmoidKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SigmoidKernel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.gamma = reader.double();
                    break;
                case 2:
                    message.c = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SigmoidKernel.prototype.gamma = 0;
$root.CoreML.Specification.SigmoidKernel.prototype.c = 0;

$root.CoreML.Specification.Kernel = class Kernel {

    constructor() {
    }

    get kernel() {
        $root.CoreML.Specification.Kernel.kernelSet = $root.CoreML.Specification.Kernel.kernelSet || new Set([ "linearKernel", "rbfKernel", "polyKernel", "sigmoidKernel"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Kernel.kernelSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Kernel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.linearKernel = $root.CoreML.Specification.LinearKernel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.rbfKernel = $root.CoreML.Specification.RBFKernel.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.polyKernel = $root.CoreML.Specification.PolyKernel.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.sigmoidKernel = $root.CoreML.Specification.SigmoidKernel.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SparseNode = class SparseNode {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SparseNode();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.index = reader.int32();
                    break;
                case 2:
                    message.value = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SparseNode.prototype.index = 0;
$root.CoreML.Specification.SparseNode.prototype.value = 0;

$root.CoreML.Specification.SparseVector = class SparseVector {

    constructor() {
        this.nodes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SparseVector();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push($root.CoreML.Specification.SparseNode.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SparseSupportVectors = class SparseSupportVectors {

    constructor() {
        this.vectors = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SparseSupportVectors();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vectors.push($root.CoreML.Specification.SparseVector.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DenseVector = class DenseVector {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DenseVector();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = reader.doubles(message.values, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.DenseSupportVectors = class DenseSupportVectors {

    constructor() {
        this.vectors = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DenseSupportVectors();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vectors.push($root.CoreML.Specification.DenseVector.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.Coefficients = class Coefficients {

    constructor() {
        this.alpha = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Coefficients();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.doubles(message.alpha, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SupportVectorRegressor = class SupportVectorRegressor {

    constructor() {
    }

    get supportVectors() {
        $root.CoreML.Specification.SupportVectorRegressor.supportVectorsSet = $root.CoreML.Specification.SupportVectorRegressor.supportVectorsSet || new Set([ "sparseSupportVectors", "denseSupportVectors"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.SupportVectorRegressor.supportVectorsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SupportVectorRegressor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.CoreML.Specification.Kernel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.sparseSupportVectors = $root.CoreML.Specification.SparseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.denseSupportVectors = $root.CoreML.Specification.DenseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.coefficients = $root.CoreML.Specification.Coefficients.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.rho = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SupportVectorRegressor.prototype.kernel = null;
$root.CoreML.Specification.SupportVectorRegressor.prototype.coefficients = null;
$root.CoreML.Specification.SupportVectorRegressor.prototype.rho = 0;

$root.CoreML.Specification.SupportVectorClassifier = class SupportVectorClassifier {

    constructor() {
        this.numberOfSupportVectorsPerClass = [];
        this.coefficients = [];
        this.rho = [];
        this.probA = [];
        this.probB = [];
    }

    get supportVectors() {
        $root.CoreML.Specification.SupportVectorClassifier.supportVectorsSet = $root.CoreML.Specification.SupportVectorClassifier.supportVectorsSet || new Set([ "sparseSupportVectors", "denseSupportVectors"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.SupportVectorClassifier.supportVectorsSet.has(key) && this[key] != null);
    }

    get ClassLabels() {
        $root.CoreML.Specification.SupportVectorClassifier.ClassLabelsSet = $root.CoreML.Specification.SupportVectorClassifier.ClassLabelsSet || new Set([ "stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.SupportVectorClassifier.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SupportVectorClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.CoreML.Specification.Kernel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.numberOfSupportVectorsPerClass = reader.array(message.numberOfSupportVectorsPerClass, () => reader.int32(), tag);
                    break;
                case 3:
                    message.sparseSupportVectors = $root.CoreML.Specification.SparseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.denseSupportVectors = $root.CoreML.Specification.DenseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.coefficients.push($root.CoreML.Specification.Coefficients.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.rho = reader.doubles(message.rho, tag);
                    break;
                case 7:
                    message.probA = reader.doubles(message.probA, tag);
                    break;
                case 8:
                    message.probB = reader.doubles(message.probB, tag);
                    break;
                case 100:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.SupportVectorClassifier.prototype.kernel = null;

$root.CoreML.Specification.TreeEnsemblePostEvaluationTransform = {
    "NoTransform": 0,
    "Classification_SoftMax": 1,
    "Regression_Logistic": 2,
    "Classification_SoftMaxWithZeroClassReference": 3
};

$root.CoreML.Specification.TreeEnsembleParameters = class TreeEnsembleParameters {

    constructor() {
        this.nodes = [];
        this.basePredictionValue = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TreeEnsembleParameters();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push($root.CoreML.Specification.TreeEnsembleParameters.TreeNode.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.numPredictionDimensions = reader.uint64();
                    break;
                case 3:
                    message.basePredictionValue = reader.doubles(message.basePredictionValue, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TreeEnsembleParameters.prototype.numPredictionDimensions = protobuf.Uint64.create(0);

$root.CoreML.Specification.TreeEnsembleParameters.TreeNode = class TreeNode {

    constructor() {
        this.evaluationInfo = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.treeId = reader.uint64();
                    break;
                case 2:
                    message.nodeId = reader.uint64();
                    break;
                case 3:
                    message.nodeBehavior = reader.int32();
                    break;
                case 10:
                    message.branchFeatureIndex = reader.uint64();
                    break;
                case 11:
                    message.branchFeatureValue = reader.double();
                    break;
                case 12:
                    message.trueChildNodeId = reader.uint64();
                    break;
                case 13:
                    message.falseChildNodeId = reader.uint64();
                    break;
                case 14:
                    message.missingValueTracksTrueChild = reader.bool();
                    break;
                case 20:
                    message.evaluationInfo.push($root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.decode(reader, reader.uint32()));
                    break;
                case 30:
                    message.relativeHitRate = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.treeId = protobuf.Uint64.create(0);
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.nodeId = protobuf.Uint64.create(0);
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.nodeBehavior = 0;
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.branchFeatureIndex = protobuf.Uint64.create(0);
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.branchFeatureValue = 0;
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.trueChildNodeId = protobuf.Uint64.create(0);
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.falseChildNodeId = protobuf.Uint64.create(0);
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.missingValueTracksTrueChild = false;
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.relativeHitRate = 0;

$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior = {
    "BranchOnValueLessThanEqual": 0,
    "BranchOnValueLessThan": 1,
    "BranchOnValueGreaterThanEqual": 2,
    "BranchOnValueGreaterThan": 3,
    "BranchOnValueEqual": 4,
    "BranchOnValueNotEqual": 5,
    "LeafNode": 6
};

$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo = class EvaluationInfo {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.evaluationIndex = reader.uint64();
                    break;
                case 2:
                    message.evaluationValue = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.prototype.evaluationIndex = protobuf.Uint64.create(0);
$root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.prototype.evaluationValue = 0;

$root.CoreML.Specification.TreeEnsembleClassifier = class TreeEnsembleClassifier {

    constructor() {
    }

    get ClassLabels() {
        $root.CoreML.Specification.TreeEnsembleClassifier.ClassLabelsSet = $root.CoreML.Specification.TreeEnsembleClassifier.ClassLabelsSet || new Set([ "stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.TreeEnsembleClassifier.ClassLabelsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TreeEnsembleClassifier();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.treeEnsemble = $root.CoreML.Specification.TreeEnsembleParameters.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.postEvaluationTransform = reader.int32();
                    break;
                case 100:
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TreeEnsembleClassifier.prototype.treeEnsemble = null;
$root.CoreML.Specification.TreeEnsembleClassifier.prototype.postEvaluationTransform = 0;

$root.CoreML.Specification.TreeEnsembleRegressor = class TreeEnsembleRegressor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TreeEnsembleRegressor();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.treeEnsemble = $root.CoreML.Specification.TreeEnsembleParameters.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.postEvaluationTransform = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.TreeEnsembleRegressor.prototype.treeEnsemble = null;
$root.CoreML.Specification.TreeEnsembleRegressor.prototype.postEvaluationTransform = 0;

$root.CoreML.Specification.ItemSimilarityRecommender = class ItemSimilarityRecommender {

    constructor() {
        this.itemItemSimilarities = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.itemItemSimilarities.push($root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.itemStringIds = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.itemInt64Ids = $root.CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.itemInputFeatureName = reader.string();
                    break;
                case 11:
                    message.numRecommendationsInputFeatureName = reader.string();
                    break;
                case 12:
                    message.itemRestrictionInputFeatureName = reader.string();
                    break;
                case 13:
                    message.itemExclusionInputFeatureName = reader.string();
                    break;
                case 20:
                    message.recommendedItemListOutputFeatureName = reader.string();
                    break;
                case 21:
                    message.recommendedItemScoreOutputFeatureName = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ItemSimilarityRecommender.prototype.itemStringIds = null;
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.itemInt64Ids = null;
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.itemInputFeatureName = "";
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.numRecommendationsInputFeatureName = "";
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.itemRestrictionInputFeatureName = "";
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.itemExclusionInputFeatureName = "";
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.recommendedItemListOutputFeatureName = "";
$root.CoreML.Specification.ItemSimilarityRecommender.prototype.recommendedItemScoreOutputFeatureName = "";

$root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem = class ConnectedItem {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.itemId = reader.uint64();
                    break;
                case 2:
                    message.similarityScore = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.prototype.itemId = protobuf.Uint64.create(0);
$root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.prototype.similarityScore = 0;

$root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems = class SimilarItems {

    constructor() {
        this.similarItemList = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.itemId = reader.uint64();
                    break;
                case 2:
                    message.similarItemList.push($root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.itemScoreAdjustment = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems.prototype.itemId = protobuf.Uint64.create(0);
$root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems.prototype.itemScoreAdjustment = 0;

$root.CoreML.Specification.LinkedModel = class LinkedModel {

    constructor() {
    }

    get LinkType() {
        $root.CoreML.Specification.LinkedModel.LinkTypeSet = $root.CoreML.Specification.LinkedModel.LinkTypeSet || new Set([ "linkedModelFile"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.LinkedModel.LinkTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinkedModel();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.linkedModelFile = $root.CoreML.Specification.LinkedModelFile.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LinkedModelFile = class LinkedModelFile {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinkedModelFile();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.linkedModelFileName = $root.CoreML.Specification.StringParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.linkedModelSearchPath = $root.CoreML.Specification.StringParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LinkedModelFile.prototype.linkedModelFileName = null;
$root.CoreML.Specification.LinkedModelFile.prototype.linkedModelSearchPath = null;
