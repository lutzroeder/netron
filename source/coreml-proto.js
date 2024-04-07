
export const CoreML = {};

CoreML.Specification = {};

CoreML.Specification.Pipeline = class Pipeline {

    constructor() {
        this.models = [];
        this.names = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Pipeline();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.models.push(CoreML.Specification.Model.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Pipeline();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "models":
                    message.models.push(CoreML.Specification.Model.decodeText(reader));
                    break;
                case "names":
                    reader.array(message.names, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PipelineClassifier = class PipelineClassifier {

    static decode(reader, length) {
        const message = new CoreML.Specification.PipelineClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pipeline = CoreML.Specification.Pipeline.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PipelineClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pipeline":
                    message.pipeline = CoreML.Specification.Pipeline.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PipelineClassifier.prototype.pipeline = null;

CoreML.Specification.PipelineRegressor = class PipelineRegressor {

    static decode(reader, length) {
        const message = new CoreML.Specification.PipelineRegressor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pipeline = CoreML.Specification.Pipeline.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PipelineRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pipeline":
                    message.pipeline = CoreML.Specification.Pipeline.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PipelineRegressor.prototype.pipeline = null;

CoreML.Specification.FeatureDescription = class FeatureDescription {

    static decode(reader, length) {
        const message = new CoreML.Specification.FeatureDescription();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shortDescription = reader.string();
                    break;
                case 3:
                    message.type = CoreML.Specification.FeatureType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.FeatureDescription();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shortDescription":
                    message.shortDescription = reader.string();
                    break;
                case "type":
                    message.type = CoreML.Specification.FeatureType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FeatureDescription.prototype.name = "";
CoreML.Specification.FeatureDescription.prototype.shortDescription = "";
CoreML.Specification.FeatureDescription.prototype.type = null;

CoreML.Specification.Metadata = class Metadata {

    constructor() {
        this.userDefined = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Metadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Metadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shortDescription":
                    message.shortDescription = reader.string();
                    break;
                case "versionString":
                    message.versionString = reader.string();
                    break;
                case "author":
                    message.author = reader.string();
                    break;
                case "license":
                    message.license = reader.string();
                    break;
                case "userDefined":
                    reader.entry(message.userDefined, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Metadata.prototype.shortDescription = "";
CoreML.Specification.Metadata.prototype.versionString = "";
CoreML.Specification.Metadata.prototype.author = "";
CoreML.Specification.Metadata.prototype.license = "";

CoreML.Specification.ModelDescription = class ModelDescription {

    constructor() {
        this.input = [];
        this.output = [];
        this.trainingInput = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ModelDescription();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input.push(CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.output.push(CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.predictedFeatureName = reader.string();
                    break;
                case 12:
                    message.predictedProbabilitiesName = reader.string();
                    break;
                case 50:
                    message.trainingInput.push(CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.metadata = CoreML.Specification.Metadata.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ModelDescription();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    message.input.push(CoreML.Specification.FeatureDescription.decodeText(reader));
                    break;
                case "output":
                    message.output.push(CoreML.Specification.FeatureDescription.decodeText(reader));
                    break;
                case "predictedFeatureName":
                    message.predictedFeatureName = reader.string();
                    break;
                case "predictedProbabilitiesName":
                    message.predictedProbabilitiesName = reader.string();
                    break;
                case "trainingInput":
                    message.trainingInput.push(CoreML.Specification.FeatureDescription.decodeText(reader));
                    break;
                case "metadata":
                    message.metadata = CoreML.Specification.Metadata.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ModelDescription.prototype.predictedFeatureName = "";
CoreML.Specification.ModelDescription.prototype.predictedProbabilitiesName = "";
CoreML.Specification.ModelDescription.prototype.metadata = null;

CoreML.Specification.SerializedModel = class SerializedModel {

    static decode(reader, length) {
        const message = new CoreML.Specification.SerializedModel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SerializedModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "identifier":
                    message.identifier = reader.string();
                    break;
                case "model":
                    message.model = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SerializedModel.prototype.identifier = "";
CoreML.Specification.SerializedModel.prototype.model = new Uint8Array([]);

CoreML.Specification.Model = class Model {

    get Type() {
        CoreML.Specification.Model.TypeSet = CoreML.Specification.Model.TypeSet || new Set(["pipelineClassifier", "pipelineRegressor", "pipeline", "glmRegressor", "supportVectorRegressor", "treeEnsembleRegressor", "neuralNetworkRegressor", "bayesianProbitRegressor", "glmClassifier", "supportVectorClassifier", "treeEnsembleClassifier", "neuralNetworkClassifier", "kNearestNeighborsClassifier", "neuralNetwork", "itemSimilarityRecommender", "mlProgram", "customModel", "linkedModel", "classConfidenceThresholding", "oneHotEncoder", "imputer", "featureVectorizer", "dictVectorizer", "scaler", "categoricalMapping", "normalizer", "arrayFeatureExtractor", "nonMaximumSuppression", "identity", "textClassifier", "wordTagger", "visionFeaturePrint", "soundAnalysisPreprocessing", "gazetteer", "wordEmbedding", "audioFeaturePrint", "serializedModel"]);
        return Object.keys(this).find((key) => CoreML.Specification.Model.TypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Model();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.specificationVersion = reader.int32();
                    break;
                case 2:
                    message.description = CoreML.Specification.ModelDescription.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.isUpdatable = reader.bool();
                    break;
                case 200:
                    message.pipelineClassifier = CoreML.Specification.PipelineClassifier.decode(reader, reader.uint32());
                    break;
                case 201:
                    message.pipelineRegressor = CoreML.Specification.PipelineRegressor.decode(reader, reader.uint32());
                    break;
                case 202:
                    message.pipeline = CoreML.Specification.Pipeline.decode(reader, reader.uint32());
                    break;
                case 300:
                    message.glmRegressor = CoreML.Specification.GLMRegressor.decode(reader, reader.uint32());
                    break;
                case 301:
                    message.supportVectorRegressor = CoreML.Specification.SupportVectorRegressor.decode(reader, reader.uint32());
                    break;
                case 302:
                    message.treeEnsembleRegressor = CoreML.Specification.TreeEnsembleRegressor.decode(reader, reader.uint32());
                    break;
                case 303:
                    message.neuralNetworkRegressor = CoreML.Specification.NeuralNetworkRegressor.decode(reader, reader.uint32());
                    break;
                case 304:
                    message.bayesianProbitRegressor = CoreML.Specification.BayesianProbitRegressor.decode(reader, reader.uint32());
                    break;
                case 400:
                    message.glmClassifier = CoreML.Specification.GLMClassifier.decode(reader, reader.uint32());
                    break;
                case 401:
                    message.supportVectorClassifier = CoreML.Specification.SupportVectorClassifier.decode(reader, reader.uint32());
                    break;
                case 402:
                    message.treeEnsembleClassifier = CoreML.Specification.TreeEnsembleClassifier.decode(reader, reader.uint32());
                    break;
                case 403:
                    message.neuralNetworkClassifier = CoreML.Specification.NeuralNetworkClassifier.decode(reader, reader.uint32());
                    break;
                case 404:
                    message.kNearestNeighborsClassifier = CoreML.Specification.KNearestNeighborsClassifier.decode(reader, reader.uint32());
                    break;
                case 500:
                    message.neuralNetwork = CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                case 501:
                    message.itemSimilarityRecommender = CoreML.Specification.ItemSimilarityRecommender.decode(reader, reader.uint32());
                    break;
                case 502:
                    message.mlProgram = CoreML.Specification.MILSpec.Program.decode(reader, reader.uint32());
                    break;
                case 555:
                    message.customModel = CoreML.Specification.CustomModel.decode(reader, reader.uint32());
                    break;
                case 556:
                    message.linkedModel = CoreML.Specification.LinkedModel.decode(reader, reader.uint32());
                    break;
                case 560:
                    message.classConfidenceThresholding = CoreML.Specification.ClassConfidenceThresholding.decode(reader, reader.uint32());
                    break;
                case 600:
                    message.oneHotEncoder = CoreML.Specification.OneHotEncoder.decode(reader, reader.uint32());
                    break;
                case 601:
                    message.imputer = CoreML.Specification.Imputer.decode(reader, reader.uint32());
                    break;
                case 602:
                    message.featureVectorizer = CoreML.Specification.FeatureVectorizer.decode(reader, reader.uint32());
                    break;
                case 603:
                    message.dictVectorizer = CoreML.Specification.DictVectorizer.decode(reader, reader.uint32());
                    break;
                case 604:
                    message.scaler = CoreML.Specification.Scaler.decode(reader, reader.uint32());
                    break;
                case 606:
                    message.categoricalMapping = CoreML.Specification.CategoricalMapping.decode(reader, reader.uint32());
                    break;
                case 607:
                    message.normalizer = CoreML.Specification.Normalizer.decode(reader, reader.uint32());
                    break;
                case 609:
                    message.arrayFeatureExtractor = CoreML.Specification.ArrayFeatureExtractor.decode(reader, reader.uint32());
                    break;
                case 610:
                    message.nonMaximumSuppression = CoreML.Specification.NonMaximumSuppression.decode(reader, reader.uint32());
                    break;
                case 900:
                    message.identity = CoreML.Specification.Identity.decode(reader, reader.uint32());
                    break;
                case 2000:
                    message.textClassifier = CoreML.Specification.CoreMLModels.TextClassifier.decode(reader, reader.uint32());
                    break;
                case 2001:
                    message.wordTagger = CoreML.Specification.CoreMLModels.WordTagger.decode(reader, reader.uint32());
                    break;
                case 2002:
                    message.visionFeaturePrint = CoreML.Specification.CoreMLModels.VisionFeaturePrint.decode(reader, reader.uint32());
                    break;
                case 2003:
                    message.soundAnalysisPreprocessing = CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.decode(reader, reader.uint32());
                    break;
                case 2004:
                    message.gazetteer = CoreML.Specification.CoreMLModels.Gazetteer.decode(reader, reader.uint32());
                    break;
                case 2005:
                    message.wordEmbedding = CoreML.Specification.CoreMLModels.WordEmbedding.decode(reader, reader.uint32());
                    break;
                case 2006:
                    message.audioFeaturePrint = CoreML.Specification.CoreMLModels.AudioFeaturePrint.decode(reader, reader.uint32());
                    break;
                case 3000:
                    message.serializedModel = CoreML.Specification.SerializedModel.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Model();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "specificationVersion":
                    message.specificationVersion = reader.int32();
                    break;
                case "description":
                    message.description = CoreML.Specification.ModelDescription.decodeText(reader);
                    break;
                case "isUpdatable":
                    message.isUpdatable = reader.bool();
                    break;
                case "pipelineClassifier":
                    message.pipelineClassifier = CoreML.Specification.PipelineClassifier.decodeText(reader);
                    break;
                case "pipelineRegressor":
                    message.pipelineRegressor = CoreML.Specification.PipelineRegressor.decodeText(reader);
                    break;
                case "pipeline":
                    message.pipeline = CoreML.Specification.Pipeline.decodeText(reader);
                    break;
                case "glmRegressor":
                    message.glmRegressor = CoreML.Specification.GLMRegressor.decodeText(reader);
                    break;
                case "supportVectorRegressor":
                    message.supportVectorRegressor = CoreML.Specification.SupportVectorRegressor.decodeText(reader);
                    break;
                case "treeEnsembleRegressor":
                    message.treeEnsembleRegressor = CoreML.Specification.TreeEnsembleRegressor.decodeText(reader);
                    break;
                case "neuralNetworkRegressor":
                    message.neuralNetworkRegressor = CoreML.Specification.NeuralNetworkRegressor.decodeText(reader);
                    break;
                case "bayesianProbitRegressor":
                    message.bayesianProbitRegressor = CoreML.Specification.BayesianProbitRegressor.decodeText(reader);
                    break;
                case "glmClassifier":
                    message.glmClassifier = CoreML.Specification.GLMClassifier.decodeText(reader);
                    break;
                case "supportVectorClassifier":
                    message.supportVectorClassifier = CoreML.Specification.SupportVectorClassifier.decodeText(reader);
                    break;
                case "treeEnsembleClassifier":
                    message.treeEnsembleClassifier = CoreML.Specification.TreeEnsembleClassifier.decodeText(reader);
                    break;
                case "neuralNetworkClassifier":
                    message.neuralNetworkClassifier = CoreML.Specification.NeuralNetworkClassifier.decodeText(reader);
                    break;
                case "kNearestNeighborsClassifier":
                    message.kNearestNeighborsClassifier = CoreML.Specification.KNearestNeighborsClassifier.decodeText(reader);
                    break;
                case "neuralNetwork":
                    message.neuralNetwork = CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                case "itemSimilarityRecommender":
                    message.itemSimilarityRecommender = CoreML.Specification.ItemSimilarityRecommender.decodeText(reader);
                    break;
                case "mlProgram":
                    message.mlProgram = CoreML.Specification.MILSpec.Program.decodeText(reader);
                    break;
                case "customModel":
                    message.customModel = CoreML.Specification.CustomModel.decodeText(reader);
                    break;
                case "linkedModel":
                    message.linkedModel = CoreML.Specification.LinkedModel.decodeText(reader);
                    break;
                case "classConfidenceThresholding":
                    message.classConfidenceThresholding = CoreML.Specification.ClassConfidenceThresholding.decodeText(reader);
                    break;
                case "oneHotEncoder":
                    message.oneHotEncoder = CoreML.Specification.OneHotEncoder.decodeText(reader);
                    break;
                case "imputer":
                    message.imputer = CoreML.Specification.Imputer.decodeText(reader);
                    break;
                case "featureVectorizer":
                    message.featureVectorizer = CoreML.Specification.FeatureVectorizer.decodeText(reader);
                    break;
                case "dictVectorizer":
                    message.dictVectorizer = CoreML.Specification.DictVectorizer.decodeText(reader);
                    break;
                case "scaler":
                    message.scaler = CoreML.Specification.Scaler.decodeText(reader);
                    break;
                case "categoricalMapping":
                    message.categoricalMapping = CoreML.Specification.CategoricalMapping.decodeText(reader);
                    break;
                case "normalizer":
                    message.normalizer = CoreML.Specification.Normalizer.decodeText(reader);
                    break;
                case "arrayFeatureExtractor":
                    message.arrayFeatureExtractor = CoreML.Specification.ArrayFeatureExtractor.decodeText(reader);
                    break;
                case "nonMaximumSuppression":
                    message.nonMaximumSuppression = CoreML.Specification.NonMaximumSuppression.decodeText(reader);
                    break;
                case "identity":
                    message.identity = CoreML.Specification.Identity.decodeText(reader);
                    break;
                case "textClassifier":
                    message.textClassifier = CoreML.Specification.CoreMLModels.TextClassifier.decodeText(reader);
                    break;
                case "wordTagger":
                    message.wordTagger = CoreML.Specification.CoreMLModels.WordTagger.decodeText(reader);
                    break;
                case "visionFeaturePrint":
                    message.visionFeaturePrint = CoreML.Specification.CoreMLModels.VisionFeaturePrint.decodeText(reader);
                    break;
                case "soundAnalysisPreprocessing":
                    message.soundAnalysisPreprocessing = CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.decodeText(reader);
                    break;
                case "gazetteer":
                    message.gazetteer = CoreML.Specification.CoreMLModels.Gazetteer.decodeText(reader);
                    break;
                case "wordEmbedding":
                    message.wordEmbedding = CoreML.Specification.CoreMLModels.WordEmbedding.decodeText(reader);
                    break;
                case "audioFeaturePrint":
                    message.audioFeaturePrint = CoreML.Specification.CoreMLModels.AudioFeaturePrint.decodeText(reader);
                    break;
                case "serializedModel":
                    message.serializedModel = CoreML.Specification.SerializedModel.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Model.prototype.specificationVersion = 0;
CoreML.Specification.Model.prototype.description = null;
CoreML.Specification.Model.prototype.isUpdatable = false;

CoreML.Specification.CoreMLModels = {};

CoreML.Specification.CoreMLModels.VisionFeaturePrint = class VisionFeaturePrint {

    get VisionFeaturePrintType() {
        CoreML.Specification.CoreMLModels.VisionFeaturePrint.VisionFeaturePrintTypeSet = CoreML.Specification.CoreMLModels.VisionFeaturePrint.VisionFeaturePrintTypeSet || new Set(["scene", "objects"]);
        return Object.keys(this).find((key) => CoreML.Specification.CoreMLModels.VisionFeaturePrint.VisionFeaturePrintTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.VisionFeaturePrint();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.scene = CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.objects = CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.VisionFeaturePrint();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "scene":
                    message.scene = CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.decodeText(reader);
                    break;
                case "objects":
                    message.objects = CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene = class Scene {

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.enum(CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.prototype.version = 0;

CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion = {
    "SCENE_VERSION_INVALID": 0,
    "SCENE_VERSION_1": 1,
    "SCENE_VERSION_2": 2
};

CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects = class Objects {

    constructor() {
        this.output = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.enum(CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.ObjectsVersion);
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.prototype.version = 0;

CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.ObjectsVersion = {
    "OBJECTS_VERSION_INVALID": 0,
    "OBJECTS_VERSION_1": 1
};

CoreML.Specification.CoreMLModels.AudioFeaturePrint = class AudioFeaturePrint {

    get AudioFeaturePrintType() {
        CoreML.Specification.CoreMLModels.AudioFeaturePrint.AudioFeaturePrintTypeSet = CoreML.Specification.CoreMLModels.AudioFeaturePrint.AudioFeaturePrintTypeSet || new Set(["sound"]);
        return Object.keys(this).find((key) => CoreML.Specification.CoreMLModels.AudioFeaturePrint.AudioFeaturePrintTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.AudioFeaturePrint();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.sound = CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.AudioFeaturePrint();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sound":
                    message.sound = CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound = class Sound {

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.enum(CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.SoundVersion);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.prototype.version = 0;

CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.SoundVersion = {
    "SOUND_VERSION_INVALID": 0,
    "SOUND_VERSION_1": 1
};

CoreML.Specification.CoreMLModels.TextClassifier = class TextClassifier {

    get ClassLabels() {
        CoreML.Specification.CoreMLModels.TextClassifier.ClassLabelsSet = CoreML.Specification.CoreMLModels.TextClassifier.ClassLabelsSet || new Set(["stringClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.CoreMLModels.TextClassifier.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.TextClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.TextClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "revision":
                    message.revision = reader.uint32();
                    break;
                case "language":
                    message.language = reader.string();
                    break;
                case "modelParameterData":
                    message.modelParameterData = reader.bytes();
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.TextClassifier.prototype.revision = 0;
CoreML.Specification.CoreMLModels.TextClassifier.prototype.language = "";
CoreML.Specification.CoreMLModels.TextClassifier.prototype.modelParameterData = new Uint8Array([]);

CoreML.Specification.CoreMLModels.WordTagger = class WordTagger {

    get Tags() {
        CoreML.Specification.CoreMLModels.WordTagger.TagsSet = CoreML.Specification.CoreMLModels.WordTagger.TagsSet || new Set(["stringTags"]);
        return Object.keys(this).find((key) => CoreML.Specification.CoreMLModels.WordTagger.TagsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.WordTagger();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.stringTags = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.WordTagger();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "revision":
                    message.revision = reader.uint32();
                    break;
                case "language":
                    message.language = reader.string();
                    break;
                case "tokensOutputFeatureName":
                    message.tokensOutputFeatureName = reader.string();
                    break;
                case "tokenTagsOutputFeatureName":
                    message.tokenTagsOutputFeatureName = reader.string();
                    break;
                case "tokenLocationsOutputFeatureName":
                    message.tokenLocationsOutputFeatureName = reader.string();
                    break;
                case "tokenLengthsOutputFeatureName":
                    message.tokenLengthsOutputFeatureName = reader.string();
                    break;
                case "modelParameterData":
                    message.modelParameterData = reader.bytes();
                    break;
                case "stringTags":
                    message.stringTags = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.WordTagger.prototype.revision = 0;
CoreML.Specification.CoreMLModels.WordTagger.prototype.language = "";
CoreML.Specification.CoreMLModels.WordTagger.prototype.tokensOutputFeatureName = "";
CoreML.Specification.CoreMLModels.WordTagger.prototype.tokenTagsOutputFeatureName = "";
CoreML.Specification.CoreMLModels.WordTagger.prototype.tokenLocationsOutputFeatureName = "";
CoreML.Specification.CoreMLModels.WordTagger.prototype.tokenLengthsOutputFeatureName = "";
CoreML.Specification.CoreMLModels.WordTagger.prototype.modelParameterData = new Uint8Array([]);

CoreML.Specification.CoreMLModels.Gazetteer = class Gazetteer {

    get ClassLabels() {
        CoreML.Specification.CoreMLModels.Gazetteer.ClassLabelsSet = CoreML.Specification.CoreMLModels.Gazetteer.ClassLabelsSet || new Set(["stringClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.CoreMLModels.Gazetteer.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.Gazetteer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.Gazetteer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "revision":
                    message.revision = reader.uint32();
                    break;
                case "language":
                    message.language = reader.string();
                    break;
                case "modelParameterData":
                    message.modelParameterData = reader.bytes();
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.Gazetteer.prototype.revision = 0;
CoreML.Specification.CoreMLModels.Gazetteer.prototype.language = "";
CoreML.Specification.CoreMLModels.Gazetteer.prototype.modelParameterData = new Uint8Array([]);

CoreML.Specification.CoreMLModels.WordEmbedding = class WordEmbedding {

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.WordEmbedding();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.WordEmbedding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "revision":
                    message.revision = reader.uint32();
                    break;
                case "language":
                    message.language = reader.string();
                    break;
                case "modelParameterData":
                    message.modelParameterData = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.WordEmbedding.prototype.revision = 0;
CoreML.Specification.CoreMLModels.WordEmbedding.prototype.language = "";
CoreML.Specification.CoreMLModels.WordEmbedding.prototype.modelParameterData = new Uint8Array([]);

CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing = class SoundAnalysisPreprocessing {

    get SoundAnalysisPreprocessingType() {
        CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.SoundAnalysisPreprocessingTypeSet = CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.SoundAnalysisPreprocessingTypeSet || new Set(["vggish"]);
        return Object.keys(this).find((key) => CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.SoundAnalysisPreprocessingTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.vggish = CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vggish":
                    message.vggish = CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish = class Vggish {

    static decode(reader, length) {
        const message = new CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.StringToInt64Map = class StringToInt64Map {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.StringToInt64Map();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.StringToInt64Map();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "map":
                    reader.entry(message.map, () => reader.string(), () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64ToStringMap = class Int64ToStringMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64ToStringMap();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64ToStringMap();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "map":
                    reader.entry(message.map, () => reader.int64(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.StringToDoubleMap = class StringToDoubleMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.StringToDoubleMap();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.StringToDoubleMap();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "map":
                    reader.entry(message.map, () => reader.string(), () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64ToDoubleMap = class Int64ToDoubleMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64ToDoubleMap();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64ToDoubleMap();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "map":
                    reader.entry(message.map, () => reader.int64(), () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.StringVector = class StringVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.StringVector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.StringVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vector":
                    reader.array(message.vector, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64Vector = class Int64Vector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64Vector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64Vector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vector":
                    reader.array(message.vector, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FloatVector = class FloatVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.FloatVector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FloatVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vector":
                    reader.array(message.vector, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DoubleVector = class DoubleVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.DoubleVector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.DoubleVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vector":
                    reader.array(message.vector, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64Range = class Int64Range {

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64Range();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64Range();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "minValue":
                    message.minValue = reader.int64();
                    break;
                case "maxValue":
                    message.maxValue = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64Range.prototype.minValue = 0n;
CoreML.Specification.Int64Range.prototype.maxValue = 0n;

CoreML.Specification.Int64Set = class Int64Set {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64Set();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64Set();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DoubleRange = class DoubleRange {

    static decode(reader, length) {
        const message = new CoreML.Specification.DoubleRange();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.DoubleRange();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "minValue":
                    message.minValue = reader.double();
                    break;
                case "maxValue":
                    message.maxValue = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DoubleRange.prototype.minValue = 0;
CoreML.Specification.DoubleRange.prototype.maxValue = 0;

CoreML.Specification.PrecisionRecallCurve = class PrecisionRecallCurve {

    static decode(reader, length) {
        const message = new CoreML.Specification.PrecisionRecallCurve();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.precisionValues = CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.precisionConfidenceThresholds = CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.recallValues = CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.recallConfidenceThresholds = CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PrecisionRecallCurve();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "precisionValues":
                    message.precisionValues = CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                case "precisionConfidenceThresholds":
                    message.precisionConfidenceThresholds = CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                case "recallValues":
                    message.recallValues = CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                case "recallConfidenceThresholds":
                    message.recallConfidenceThresholds = CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PrecisionRecallCurve.prototype.precisionValues = null;
CoreML.Specification.PrecisionRecallCurve.prototype.precisionConfidenceThresholds = null;
CoreML.Specification.PrecisionRecallCurve.prototype.recallValues = null;
CoreML.Specification.PrecisionRecallCurve.prototype.recallConfidenceThresholds = null;

CoreML.Specification.Int64FeatureType = class Int64FeatureType {

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64FeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64FeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DoubleFeatureType = class DoubleFeatureType {

    static decode(reader, length) {
        const message = new CoreML.Specification.DoubleFeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.DoubleFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.StringFeatureType = class StringFeatureType {

    static decode(reader, length) {
        const message = new CoreML.Specification.StringFeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.StringFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SizeRange = class SizeRange {

    static decode(reader, length) {
        const message = new CoreML.Specification.SizeRange();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SizeRange();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lowerBound":
                    message.lowerBound = reader.uint64();
                    break;
                case "upperBound":
                    message.upperBound = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SizeRange.prototype.lowerBound = 0n;
CoreML.Specification.SizeRange.prototype.upperBound = 0n;

CoreML.Specification.ImageFeatureType = class ImageFeatureType {

    get SizeFlexibility() {
        CoreML.Specification.ImageFeatureType.SizeFlexibilitySet = CoreML.Specification.ImageFeatureType.SizeFlexibilitySet || new Set(["enumeratedSizes", "imageSizeRange"]);
        return Object.keys(this).find((key) => CoreML.Specification.ImageFeatureType.SizeFlexibilitySet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ImageFeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.width = reader.int64();
                    break;
                case 2:
                    message.height = reader.int64();
                    break;
                case 21:
                    message.enumeratedSizes = CoreML.Specification.ImageFeatureType.EnumeratedImageSizes.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.imageSizeRange = CoreML.Specification.ImageFeatureType.ImageSizeRange.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ImageFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "width":
                    message.width = reader.int64();
                    break;
                case "height":
                    message.height = reader.int64();
                    break;
                case "enumeratedSizes":
                    message.enumeratedSizes = CoreML.Specification.ImageFeatureType.EnumeratedImageSizes.decodeText(reader);
                    break;
                case "imageSizeRange":
                    message.imageSizeRange = CoreML.Specification.ImageFeatureType.ImageSizeRange.decodeText(reader);
                    break;
                case "colorSpace":
                    message.colorSpace = reader.enum(CoreML.Specification.ImageFeatureType.ColorSpace);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ImageFeatureType.prototype.width = 0n;
CoreML.Specification.ImageFeatureType.prototype.height = 0n;
CoreML.Specification.ImageFeatureType.prototype.colorSpace = 0;

CoreML.Specification.ImageFeatureType.ColorSpace = {
    "INVALID_COLOR_SPACE": 0,
    "GRAYSCALE": 10,
    "RGB": 20,
    "BGR": 30,
    "GRAYSCALE_FLOAT16": 40
};

CoreML.Specification.ImageFeatureType.ImageSize = class ImageSize {

    static decode(reader, length) {
        const message = new CoreML.Specification.ImageFeatureType.ImageSize();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ImageFeatureType.ImageSize();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "width":
                    message.width = reader.uint64();
                    break;
                case "height":
                    message.height = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ImageFeatureType.ImageSize.prototype.width = 0n;
CoreML.Specification.ImageFeatureType.ImageSize.prototype.height = 0n;

CoreML.Specification.ImageFeatureType.EnumeratedImageSizes = class EnumeratedImageSizes {

    constructor() {
        this.sizes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ImageFeatureType.EnumeratedImageSizes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sizes.push(CoreML.Specification.ImageFeatureType.ImageSize.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ImageFeatureType.EnumeratedImageSizes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sizes":
                    message.sizes.push(CoreML.Specification.ImageFeatureType.ImageSize.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ImageFeatureType.ImageSizeRange = class ImageSizeRange {

    static decode(reader, length) {
        const message = new CoreML.Specification.ImageFeatureType.ImageSizeRange();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.widthRange = CoreML.Specification.SizeRange.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.heightRange = CoreML.Specification.SizeRange.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ImageFeatureType.ImageSizeRange();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "widthRange":
                    message.widthRange = CoreML.Specification.SizeRange.decodeText(reader);
                    break;
                case "heightRange":
                    message.heightRange = CoreML.Specification.SizeRange.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ImageFeatureType.ImageSizeRange.prototype.widthRange = null;
CoreML.Specification.ImageFeatureType.ImageSizeRange.prototype.heightRange = null;

CoreML.Specification.ArrayFeatureType = class ArrayFeatureType {

    constructor() {
        this.shape = [];
    }

    get ShapeFlexibility() {
        CoreML.Specification.ArrayFeatureType.ShapeFlexibilitySet = CoreML.Specification.ArrayFeatureType.ShapeFlexibilitySet || new Set(["enumeratedShapes", "shapeRange"]);
        return Object.keys(this).find((key) => CoreML.Specification.ArrayFeatureType.ShapeFlexibilitySet.has(key) && this[key] !== null);
    }

    get defaultOptionalValue() {
        CoreML.Specification.ArrayFeatureType.defaultOptionalValueSet = CoreML.Specification.ArrayFeatureType.defaultOptionalValueSet || new Set(["intDefaultValue", "floatDefaultValue", "doubleDefaultValue"]);
        return Object.keys(this).find((key) => CoreML.Specification.ArrayFeatureType.defaultOptionalValueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ArrayFeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.int64(), tag);
                    break;
                case 2:
                    message.dataType = reader.int32();
                    break;
                case 21:
                    message.enumeratedShapes = CoreML.Specification.ArrayFeatureType.EnumeratedShapes.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.shapeRange = CoreML.Specification.ArrayFeatureType.ShapeRange.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ArrayFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.int64());
                    break;
                case "dataType":
                    message.dataType = reader.enum(CoreML.Specification.ArrayFeatureType.ArrayDataType);
                    break;
                case "enumeratedShapes":
                    message.enumeratedShapes = CoreML.Specification.ArrayFeatureType.EnumeratedShapes.decodeText(reader);
                    break;
                case "shapeRange":
                    message.shapeRange = CoreML.Specification.ArrayFeatureType.ShapeRange.decodeText(reader);
                    break;
                case "intDefaultValue":
                    message.intDefaultValue = reader.int32();
                    break;
                case "floatDefaultValue":
                    message.floatDefaultValue = reader.float();
                    break;
                case "doubleDefaultValue":
                    message.doubleDefaultValue = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ArrayFeatureType.prototype.dataType = 0;

CoreML.Specification.ArrayFeatureType.ArrayDataType = {
    "INVALID_ARRAY_DATA_TYPE": 0,
    "FLOAT32": 65568,
    "DOUBLE": 65600,
    "INT32": 131104,
    "FLOAT16": 65552
};

CoreML.Specification.ArrayFeatureType.Shape = class Shape {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ArrayFeatureType.Shape();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ArrayFeatureType.Shape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ArrayFeatureType.EnumeratedShapes = class EnumeratedShapes {

    constructor() {
        this.shapes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ArrayFeatureType.EnumeratedShapes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shapes.push(CoreML.Specification.ArrayFeatureType.Shape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ArrayFeatureType.EnumeratedShapes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shapes":
                    message.shapes.push(CoreML.Specification.ArrayFeatureType.Shape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ArrayFeatureType.ShapeRange = class ShapeRange {

    constructor() {
        this.sizeRanges = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ArrayFeatureType.ShapeRange();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sizeRanges.push(CoreML.Specification.SizeRange.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ArrayFeatureType.ShapeRange();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sizeRanges":
                    message.sizeRanges.push(CoreML.Specification.SizeRange.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DictionaryFeatureType = class DictionaryFeatureType {

    get KeyType() {
        CoreML.Specification.DictionaryFeatureType.KeyTypeSet = CoreML.Specification.DictionaryFeatureType.KeyTypeSet || new Set(["int64KeyType", "stringKeyType"]);
        return Object.keys(this).find((key) => CoreML.Specification.DictionaryFeatureType.KeyTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.DictionaryFeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.int64KeyType = CoreML.Specification.Int64FeatureType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stringKeyType = CoreML.Specification.StringFeatureType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.DictionaryFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "int64KeyType":
                    message.int64KeyType = CoreML.Specification.Int64FeatureType.decodeText(reader);
                    break;
                case "stringKeyType":
                    message.stringKeyType = CoreML.Specification.StringFeatureType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SequenceFeatureType = class SequenceFeatureType {

    get Type() {
        CoreML.Specification.SequenceFeatureType.TypeSet = CoreML.Specification.SequenceFeatureType.TypeSet || new Set(["int64Type", "stringType"]);
        return Object.keys(this).find((key) => CoreML.Specification.SequenceFeatureType.TypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SequenceFeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.int64Type = CoreML.Specification.Int64FeatureType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stringType = CoreML.Specification.StringFeatureType.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.sizeRange = CoreML.Specification.SizeRange.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SequenceFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "int64Type":
                    message.int64Type = CoreML.Specification.Int64FeatureType.decodeText(reader);
                    break;
                case "stringType":
                    message.stringType = CoreML.Specification.StringFeatureType.decodeText(reader);
                    break;
                case "sizeRange":
                    message.sizeRange = CoreML.Specification.SizeRange.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SequenceFeatureType.prototype.sizeRange = null;

CoreML.Specification.FeatureType = class FeatureType {

    get Type() {
        CoreML.Specification.FeatureType.TypeSet = CoreML.Specification.FeatureType.TypeSet || new Set(["int64Type", "doubleType", "stringType", "imageType", "multiArrayType", "dictionaryType", "sequenceType"]);
        return Object.keys(this).find((key) => CoreML.Specification.FeatureType.TypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.FeatureType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.int64Type = CoreML.Specification.Int64FeatureType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.doubleType = CoreML.Specification.DoubleFeatureType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stringType = CoreML.Specification.StringFeatureType.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.imageType = CoreML.Specification.ImageFeatureType.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.multiArrayType = CoreML.Specification.ArrayFeatureType.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.dictionaryType = CoreML.Specification.DictionaryFeatureType.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.sequenceType = CoreML.Specification.SequenceFeatureType.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "int64Type":
                    message.int64Type = CoreML.Specification.Int64FeatureType.decodeText(reader);
                    break;
                case "doubleType":
                    message.doubleType = CoreML.Specification.DoubleFeatureType.decodeText(reader);
                    break;
                case "stringType":
                    message.stringType = CoreML.Specification.StringFeatureType.decodeText(reader);
                    break;
                case "imageType":
                    message.imageType = CoreML.Specification.ImageFeatureType.decodeText(reader);
                    break;
                case "multiArrayType":
                    message.multiArrayType = CoreML.Specification.ArrayFeatureType.decodeText(reader);
                    break;
                case "dictionaryType":
                    message.dictionaryType = CoreML.Specification.DictionaryFeatureType.decodeText(reader);
                    break;
                case "sequenceType":
                    message.sequenceType = CoreML.Specification.SequenceFeatureType.decodeText(reader);
                    break;
                case "isOptional":
                    message.isOptional = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FeatureType.prototype.isOptional = false;

CoreML.Specification.ArrayFeatureExtractor = class ArrayFeatureExtractor {

    constructor() {
        this.extractIndex = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ArrayFeatureExtractor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ArrayFeatureExtractor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "extractIndex":
                    reader.array(message.extractIndex, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BayesianProbitRegressor = class BayesianProbitRegressor {

    constructor() {
        this.features = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.BayesianProbitRegressor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numberOfFeatures = reader.uint32();
                    break;
                case 2:
                    message.bias = CoreML.Specification.BayesianProbitRegressor.Gaussian.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.features.push(CoreML.Specification.BayesianProbitRegressor.FeatureWeight.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BayesianProbitRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numberOfFeatures":
                    message.numberOfFeatures = reader.uint32();
                    break;
                case "bias":
                    message.bias = CoreML.Specification.BayesianProbitRegressor.Gaussian.decodeText(reader);
                    break;
                case "features":
                    message.features.push(CoreML.Specification.BayesianProbitRegressor.FeatureWeight.decodeText(reader));
                    break;
                case "regressionInputFeatureName":
                    message.regressionInputFeatureName = reader.string();
                    break;
                case "optimismInputFeatureName":
                    message.optimismInputFeatureName = reader.string();
                    break;
                case "samplingScaleInputFeatureName":
                    message.samplingScaleInputFeatureName = reader.string();
                    break;
                case "samplingTruncationInputFeatureName":
                    message.samplingTruncationInputFeatureName = reader.string();
                    break;
                case "meanOutputFeatureName":
                    message.meanOutputFeatureName = reader.string();
                    break;
                case "varianceOutputFeatureName":
                    message.varianceOutputFeatureName = reader.string();
                    break;
                case "pessimisticProbabilityOutputFeatureName":
                    message.pessimisticProbabilityOutputFeatureName = reader.string();
                    break;
                case "sampledProbabilityOutputFeatureName":
                    message.sampledProbabilityOutputFeatureName = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BayesianProbitRegressor.prototype.numberOfFeatures = 0;
CoreML.Specification.BayesianProbitRegressor.prototype.bias = null;
CoreML.Specification.BayesianProbitRegressor.prototype.regressionInputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.optimismInputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.samplingScaleInputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.samplingTruncationInputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.meanOutputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.varianceOutputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.pessimisticProbabilityOutputFeatureName = "";
CoreML.Specification.BayesianProbitRegressor.prototype.sampledProbabilityOutputFeatureName = "";

CoreML.Specification.BayesianProbitRegressor.Gaussian = class Gaussian {

    static decode(reader, length) {
        const message = new CoreML.Specification.BayesianProbitRegressor.Gaussian();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BayesianProbitRegressor.Gaussian();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mean":
                    message.mean = reader.double();
                    break;
                case "precision":
                    message.precision = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BayesianProbitRegressor.Gaussian.prototype.mean = 0;
CoreML.Specification.BayesianProbitRegressor.Gaussian.prototype.precision = 0;

CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight = class FeatureValueWeight {

    static decode(reader, length) {
        const message = new CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.featureValue = reader.uint32();
                    break;
                case 2:
                    message.featureWeight = CoreML.Specification.BayesianProbitRegressor.Gaussian.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "featureValue":
                    message.featureValue = reader.uint32();
                    break;
                case "featureWeight":
                    message.featureWeight = CoreML.Specification.BayesianProbitRegressor.Gaussian.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.prototype.featureValue = 0;
CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.prototype.featureWeight = null;

CoreML.Specification.BayesianProbitRegressor.FeatureWeight = class FeatureWeight {

    constructor() {
        this.weights = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.BayesianProbitRegressor.FeatureWeight();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.featureId = reader.uint32();
                    break;
                case 2:
                    message.weights.push(CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BayesianProbitRegressor.FeatureWeight();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "featureId":
                    message.featureId = reader.uint32();
                    break;
                case "weights":
                    message.weights.push(CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BayesianProbitRegressor.FeatureWeight.prototype.featureId = 0;

CoreML.Specification.CategoricalMapping = class CategoricalMapping {

    get MappingType() {
        CoreML.Specification.CategoricalMapping.MappingTypeSet = CoreML.Specification.CategoricalMapping.MappingTypeSet || new Set(["stringToInt64Map", "int64ToStringMap"]);
        return Object.keys(this).find((key) => CoreML.Specification.CategoricalMapping.MappingTypeSet.has(key) && this[key] !== null);
    }

    get ValueOnUnknown() {
        CoreML.Specification.CategoricalMapping.ValueOnUnknownSet = CoreML.Specification.CategoricalMapping.ValueOnUnknownSet || new Set(["strValue", "int64Value"]);
        return Object.keys(this).find((key) => CoreML.Specification.CategoricalMapping.ValueOnUnknownSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CategoricalMapping();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.stringToInt64Map = CoreML.Specification.StringToInt64Map.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.int64ToStringMap = CoreML.Specification.Int64ToStringMap.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CategoricalMapping();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "stringToInt64Map":
                    message.stringToInt64Map = CoreML.Specification.StringToInt64Map.decodeText(reader);
                    break;
                case "int64ToStringMap":
                    message.int64ToStringMap = CoreML.Specification.Int64ToStringMap.decodeText(reader);
                    break;
                case "strValue":
                    message.strValue = reader.string();
                    break;
                case "int64Value":
                    message.int64Value = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CustomModel = class CustomModel {

    constructor() {
        this.parameters = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CustomModel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.className = reader.string();
                    break;
                case 30:
                    reader.entry(message.parameters, () => reader.string(), () => CoreML.Specification.CustomModel.CustomModelParamValue.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CustomModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "className":
                    message.className = reader.string();
                    break;
                case "parameters":
                    reader.entry(message.parameters, () => reader.string(), () => CoreML.Specification.CustomModel.CustomModelParamValue.decodeText(reader));
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CustomModel.prototype.className = "";
CoreML.Specification.CustomModel.prototype.description = "";

CoreML.Specification.CustomModel.CustomModelParamValue = class CustomModelParamValue {

    get value() {
        CoreML.Specification.CustomModel.CustomModelParamValue.valueSet = CoreML.Specification.CustomModel.CustomModelParamValue.valueSet || new Set(["doubleValue", "stringValue", "intValue", "longValue", "boolValue", "bytesValue"]);
        return Object.keys(this).find((key) => CoreML.Specification.CustomModel.CustomModelParamValue.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CustomModel.CustomModelParamValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CustomModel.CustomModelParamValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "doubleValue":
                    message.doubleValue = reader.double();
                    break;
                case "stringValue":
                    message.stringValue = reader.string();
                    break;
                case "intValue":
                    message.intValue = reader.int32();
                    break;
                case "longValue":
                    message.longValue = reader.int64();
                    break;
                case "boolValue":
                    message.boolValue = reader.bool();
                    break;
                case "bytesValue":
                    message.bytesValue = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DictVectorizer = class DictVectorizer {

    get Map() {
        CoreML.Specification.DictVectorizer.MapSet = CoreML.Specification.DictVectorizer.MapSet || new Set(["stringToIndex", "int64ToIndex"]);
        return Object.keys(this).find((key) => CoreML.Specification.DictVectorizer.MapSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.DictVectorizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.stringToIndex = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.int64ToIndex = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.DictVectorizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "stringToIndex":
                    message.stringToIndex = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ToIndex":
                    message.int64ToIndex = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FeatureVectorizer = class FeatureVectorizer {

    constructor() {
        this.inputList = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.FeatureVectorizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputList.push(CoreML.Specification.FeatureVectorizer.InputColumn.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.FeatureVectorizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputList":
                    message.inputList.push(CoreML.Specification.FeatureVectorizer.InputColumn.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FeatureVectorizer.InputColumn = class InputColumn {

    static decode(reader, length) {
        const message = new CoreML.Specification.FeatureVectorizer.InputColumn();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FeatureVectorizer.InputColumn();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputColumn":
                    message.inputColumn = reader.string();
                    break;
                case "inputDimensions":
                    message.inputDimensions = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FeatureVectorizer.InputColumn.prototype.inputColumn = "";
CoreML.Specification.FeatureVectorizer.InputColumn.prototype.inputDimensions = 0n;

CoreML.Specification.GLMRegressor = class GLMRegressor {

    constructor() {
        this.weights = [];
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.GLMRegressor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.weights.push(CoreML.Specification.GLMRegressor.DoubleArray.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GLMRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "weights":
                    message.weights.push(CoreML.Specification.GLMRegressor.DoubleArray.decodeText(reader));
                    break;
                case "offset":
                    reader.array(message.offset, () => reader.double());
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum(CoreML.Specification.GLMRegressor.PostEvaluationTransform);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GLMRegressor.prototype.postEvaluationTransform = 0;

CoreML.Specification.GLMRegressor.DoubleArray = class DoubleArray {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.GLMRegressor.DoubleArray();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GLMRegressor.DoubleArray();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GLMRegressor.PostEvaluationTransform = {
    "NoTransform": 0,
    "Logit": 1,
    "Probit": 2
};

CoreML.Specification.GLMClassifier = class GLMClassifier {

    constructor() {
        this.weights = [];
        this.offset = [];
    }

    get ClassLabels() {
        CoreML.Specification.GLMClassifier.ClassLabelsSet = CoreML.Specification.GLMClassifier.ClassLabelsSet || new Set(["stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.GLMClassifier.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.GLMClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.weights.push(CoreML.Specification.GLMClassifier.DoubleArray.decode(reader, reader.uint32()));
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
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.GLMClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "weights":
                    message.weights.push(CoreML.Specification.GLMClassifier.DoubleArray.decodeText(reader));
                    break;
                case "offset":
                    reader.array(message.offset, () => reader.double());
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum(CoreML.Specification.GLMClassifier.PostEvaluationTransform);
                    break;
                case "classEncoding":
                    message.classEncoding = reader.enum(CoreML.Specification.GLMClassifier.ClassEncoding);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GLMClassifier.prototype.postEvaluationTransform = 0;
CoreML.Specification.GLMClassifier.prototype.classEncoding = 0;

CoreML.Specification.GLMClassifier.DoubleArray = class DoubleArray {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.GLMClassifier.DoubleArray();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GLMClassifier.DoubleArray();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GLMClassifier.PostEvaluationTransform = {
    "Logit": 0,
    "Probit": 1
};

CoreML.Specification.GLMClassifier.ClassEncoding = {
    "ReferenceClass": 0,
    "OneVsRest": 1
};

CoreML.Specification.KNearestNeighborsClassifier = class KNearestNeighborsClassifier {

    get ClassLabels() {
        CoreML.Specification.KNearestNeighborsClassifier.ClassLabelsSet = CoreML.Specification.KNearestNeighborsClassifier.ClassLabelsSet || new Set(["stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.KNearestNeighborsClassifier.ClassLabelsSet.has(key) && this[key] !== null);
    }

    get DefaultClassLabel() {
        CoreML.Specification.KNearestNeighborsClassifier.DefaultClassLabelSet = CoreML.Specification.KNearestNeighborsClassifier.DefaultClassLabelSet || new Set(["defaultStringLabel", "defaultInt64Label"]);
        return Object.keys(this).find((key) => CoreML.Specification.KNearestNeighborsClassifier.DefaultClassLabelSet.has(key) && this[key] !== null);
    }

    get WeightingScheme() {
        CoreML.Specification.KNearestNeighborsClassifier.WeightingSchemeSet = CoreML.Specification.KNearestNeighborsClassifier.WeightingSchemeSet || new Set(["uniformWeighting", "inverseDistanceWeighting"]);
        return Object.keys(this).find((key) => CoreML.Specification.KNearestNeighborsClassifier.WeightingSchemeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.KNearestNeighborsClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nearestNeighborsIndex = CoreML.Specification.NearestNeighborsIndex.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.numberOfNeighbors = CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.defaultStringLabel = reader.string();
                    break;
                case 111:
                    message.defaultInt64Label = reader.int64();
                    break;
                case 200:
                    message.uniformWeighting = CoreML.Specification.UniformWeighting.decode(reader, reader.uint32());
                    break;
                case 210:
                    message.inverseDistanceWeighting = CoreML.Specification.InverseDistanceWeighting.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.KNearestNeighborsClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nearestNeighborsIndex":
                    message.nearestNeighborsIndex = CoreML.Specification.NearestNeighborsIndex.decodeText(reader);
                    break;
                case "numberOfNeighbors":
                    message.numberOfNeighbors = CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "defaultStringLabel":
                    message.defaultStringLabel = reader.string();
                    break;
                case "defaultInt64Label":
                    message.defaultInt64Label = reader.int64();
                    break;
                case "uniformWeighting":
                    message.uniformWeighting = CoreML.Specification.UniformWeighting.decodeText(reader);
                    break;
                case "inverseDistanceWeighting":
                    message.inverseDistanceWeighting = CoreML.Specification.InverseDistanceWeighting.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.KNearestNeighborsClassifier.prototype.nearestNeighborsIndex = null;
CoreML.Specification.KNearestNeighborsClassifier.prototype.numberOfNeighbors = null;

CoreML.Specification.NearestNeighborsIndex = class NearestNeighborsIndex {

    constructor() {
        this.floatSamples = [];
    }

    get IndexType() {
        CoreML.Specification.NearestNeighborsIndex.IndexTypeSet = CoreML.Specification.NearestNeighborsIndex.IndexTypeSet || new Set(["linearIndex", "singleKdTreeIndex"]);
        return Object.keys(this).find((key) => CoreML.Specification.NearestNeighborsIndex.IndexTypeSet.has(key) && this[key] !== null);
    }

    get DistanceFunction() {
        CoreML.Specification.NearestNeighborsIndex.DistanceFunctionSet = CoreML.Specification.NearestNeighborsIndex.DistanceFunctionSet || new Set(["squaredEuclideanDistance"]);
        return Object.keys(this).find((key) => CoreML.Specification.NearestNeighborsIndex.DistanceFunctionSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NearestNeighborsIndex();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numberOfDimensions = reader.int32();
                    break;
                case 2:
                    message.floatSamples.push(CoreML.Specification.FloatVector.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.linearIndex = CoreML.Specification.LinearIndex.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.singleKdTreeIndex = CoreML.Specification.SingleKdTreeIndex.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.squaredEuclideanDistance = CoreML.Specification.SquaredEuclideanDistance.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.NearestNeighborsIndex();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numberOfDimensions":
                    message.numberOfDimensions = reader.int32();
                    break;
                case "floatSamples":
                    message.floatSamples.push(CoreML.Specification.FloatVector.decodeText(reader));
                    break;
                case "linearIndex":
                    message.linearIndex = CoreML.Specification.LinearIndex.decodeText(reader);
                    break;
                case "singleKdTreeIndex":
                    message.singleKdTreeIndex = CoreML.Specification.SingleKdTreeIndex.decodeText(reader);
                    break;
                case "squaredEuclideanDistance":
                    message.squaredEuclideanDistance = CoreML.Specification.SquaredEuclideanDistance.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NearestNeighborsIndex.prototype.numberOfDimensions = 0;

CoreML.Specification.UniformWeighting = class UniformWeighting {

    static decode(reader, length) {
        const message = new CoreML.Specification.UniformWeighting();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.UniformWeighting();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.InverseDistanceWeighting = class InverseDistanceWeighting {

    static decode(reader, length) {
        const message = new CoreML.Specification.InverseDistanceWeighting();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.InverseDistanceWeighting();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LinearIndex = class LinearIndex {

    static decode(reader, length) {
        const message = new CoreML.Specification.LinearIndex();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LinearIndex();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SingleKdTreeIndex = class SingleKdTreeIndex {

    static decode(reader, length) {
        const message = new CoreML.Specification.SingleKdTreeIndex();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SingleKdTreeIndex();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "leafSize":
                    message.leafSize = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SingleKdTreeIndex.prototype.leafSize = 0;

CoreML.Specification.SquaredEuclideanDistance = class SquaredEuclideanDistance {

    static decode(reader, length) {
        const message = new CoreML.Specification.SquaredEuclideanDistance();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SquaredEuclideanDistance();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64Parameter = class Int64Parameter {

    get AllowedValues() {
        CoreML.Specification.Int64Parameter.AllowedValuesSet = CoreML.Specification.Int64Parameter.AllowedValuesSet || new Set(["range", "set"]);
        return Object.keys(this).find((key) => CoreML.Specification.Int64Parameter.AllowedValuesSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Int64Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.defaultValue = reader.int64();
                    break;
                case 10:
                    message.range = CoreML.Specification.Int64Range.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.set = CoreML.Specification.Int64Set.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Int64Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "defaultValue":
                    message.defaultValue = reader.int64();
                    break;
                case "range":
                    message.range = CoreML.Specification.Int64Range.decodeText(reader);
                    break;
                case "set":
                    message.set = CoreML.Specification.Int64Set.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Int64Parameter.prototype.defaultValue = 0n;

CoreML.Specification.DoubleParameter = class DoubleParameter {

    get AllowedValues() {
        CoreML.Specification.DoubleParameter.AllowedValuesSet = CoreML.Specification.DoubleParameter.AllowedValuesSet || new Set(["range"]);
        return Object.keys(this).find((key) => CoreML.Specification.DoubleParameter.AllowedValuesSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.DoubleParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.defaultValue = reader.double();
                    break;
                case 10:
                    message.range = CoreML.Specification.DoubleRange.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.DoubleParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "defaultValue":
                    message.defaultValue = reader.double();
                    break;
                case "range":
                    message.range = CoreML.Specification.DoubleRange.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DoubleParameter.prototype.defaultValue = 0;

CoreML.Specification.StringParameter = class StringParameter {

    static decode(reader, length) {
        const message = new CoreML.Specification.StringParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.StringParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "defaultValue":
                    message.defaultValue = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.StringParameter.prototype.defaultValue = "";

CoreML.Specification.BoolParameter = class BoolParameter {

    static decode(reader, length) {
        const message = new CoreML.Specification.BoolParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BoolParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "defaultValue":
                    message.defaultValue = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BoolParameter.prototype.defaultValue = false;

CoreML.Specification.Identity = class Identity {

    static decode(reader, length) {
        const message = new CoreML.Specification.Identity();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Identity();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Imputer = class Imputer {

    get ImputedValue() {
        CoreML.Specification.Imputer.ImputedValueSet = CoreML.Specification.Imputer.ImputedValueSet || new Set(["imputedDoubleValue", "imputedInt64Value", "imputedStringValue", "imputedDoubleArray", "imputedInt64Array", "imputedStringDictionary", "imputedInt64Dictionary"]);
        return Object.keys(this).find((key) => CoreML.Specification.Imputer.ImputedValueSet.has(key) && this[key] !== null);
    }

    get ReplaceValue() {
        CoreML.Specification.Imputer.ReplaceValueSet = CoreML.Specification.Imputer.ReplaceValueSet || new Set(["replaceDoubleValue", "replaceInt64Value", "replaceStringValue"]);
        return Object.keys(this).find((key) => CoreML.Specification.Imputer.ReplaceValueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Imputer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.imputedDoubleArray = CoreML.Specification.DoubleVector.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.imputedInt64Array = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.imputedStringDictionary = CoreML.Specification.StringToDoubleMap.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.imputedInt64Dictionary = CoreML.Specification.Int64ToDoubleMap.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Imputer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "imputedDoubleValue":
                    message.imputedDoubleValue = reader.double();
                    break;
                case "imputedInt64Value":
                    message.imputedInt64Value = reader.int64();
                    break;
                case "imputedStringValue":
                    message.imputedStringValue = reader.string();
                    break;
                case "imputedDoubleArray":
                    message.imputedDoubleArray = CoreML.Specification.DoubleVector.decodeText(reader);
                    break;
                case "imputedInt64Array":
                    message.imputedInt64Array = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "imputedStringDictionary":
                    message.imputedStringDictionary = CoreML.Specification.StringToDoubleMap.decodeText(reader);
                    break;
                case "imputedInt64Dictionary":
                    message.imputedInt64Dictionary = CoreML.Specification.Int64ToDoubleMap.decodeText(reader);
                    break;
                case "replaceDoubleValue":
                    message.replaceDoubleValue = reader.double();
                    break;
                case "replaceInt64Value":
                    message.replaceInt64Value = reader.int64();
                    break;
                case "replaceStringValue":
                    message.replaceStringValue = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec = {};

CoreML.Specification.MILSpec.Program = class Program {

    constructor() {
        this.functions = {};
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Program();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int64();
                    break;
                case 2:
                    reader.entry(message.functions, () => reader.string(), () => CoreML.Specification.MILSpec.Function.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.docString = reader.string();
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Program();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.int64();
                    break;
                case "functions":
                    reader.entry(message.functions, () => reader.string(), () => CoreML.Specification.MILSpec.Function.decodeText(reader));
                    break;
                case "docString":
                    message.docString = reader.string();
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Program.prototype.version = 0n;
CoreML.Specification.MILSpec.Program.prototype.docString = "";

CoreML.Specification.MILSpec.Function = class Function {

    constructor() {
        this.inputs = [];
        this.block_specializations = {};
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Function();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputs.push(CoreML.Specification.MILSpec.NamedValueType.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.opset = reader.string();
                    break;
                case 3:
                    reader.entry(message.block_specializations, () => reader.string(), () => CoreML.Specification.MILSpec.Block.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Function();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputs":
                    message.inputs.push(CoreML.Specification.MILSpec.NamedValueType.decodeText(reader));
                    break;
                case "opset":
                    message.opset = reader.string();
                    break;
                case "block_specializations":
                    reader.entry(message.block_specializations, () => reader.string(), () => CoreML.Specification.MILSpec.Block.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Function.prototype.opset = "";

CoreML.Specification.MILSpec.Block = class Block {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.operations = [];
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Block();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputs.push(CoreML.Specification.MILSpec.NamedValueType.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.outputs.push(reader.string());
                    break;
                case 3:
                    message.operations.push(CoreML.Specification.MILSpec.Operation.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Block();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputs":
                    message.inputs.push(CoreML.Specification.MILSpec.NamedValueType.decodeText(reader));
                    break;
                case "outputs":
                    reader.array(message.outputs, () => reader.string());
                    break;
                case "operations":
                    message.operations.push(CoreML.Specification.MILSpec.Operation.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Argument = class Argument {

    constructor() {
        this.arguments = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Argument();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.arguments.push(CoreML.Specification.MILSpec.Argument.Binding.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Argument();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "arguments":
                    message.arguments.push(CoreML.Specification.MILSpec.Argument.Binding.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Argument.Binding = class Binding {

    get binding() {
        CoreML.Specification.MILSpec.Argument.Binding.bindingSet = CoreML.Specification.MILSpec.Argument.Binding.bindingSet || new Set(["name", "value"]);
        return Object.keys(this).find((key) => CoreML.Specification.MILSpec.Argument.Binding.bindingSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Argument.Binding();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.value = CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Argument.Binding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "value":
                    message.value = CoreML.Specification.MILSpec.Value.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Operation = class Operation {

    constructor() {
        this.inputs = {};
        this.outputs = [];
        this.blocks = [];
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Operation();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.string();
                    break;
                case 2:
                    reader.entry(message.inputs, () => reader.string(), () => CoreML.Specification.MILSpec.Argument.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.outputs.push(CoreML.Specification.MILSpec.NamedValueType.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.blocks.push(CoreML.Specification.MILSpec.Block.decode(reader, reader.uint32()));
                    break;
                case 5:
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Operation();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "inputs":
                    reader.entry(message.inputs, () => reader.string(), () => CoreML.Specification.MILSpec.Argument.decodeText(reader));
                    break;
                case "outputs":
                    message.outputs.push(CoreML.Specification.MILSpec.NamedValueType.decodeText(reader));
                    break;
                case "blocks":
                    message.blocks.push(CoreML.Specification.MILSpec.Block.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Operation.prototype.type = "";

CoreML.Specification.MILSpec.NamedValueType = class NamedValueType {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.NamedValueType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.NamedValueType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.NamedValueType.prototype.name = "";
CoreML.Specification.MILSpec.NamedValueType.prototype.type = null;

CoreML.Specification.MILSpec.ValueType = class ValueType {

    get type() {
        CoreML.Specification.MILSpec.ValueType.typeSet = CoreML.Specification.MILSpec.ValueType.typeSet || new Set(["tensorType", "listType", "tupleType", "dictionaryType"]);
        return Object.keys(this).find((key) => CoreML.Specification.MILSpec.ValueType.typeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.ValueType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensorType = CoreML.Specification.MILSpec.TensorType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.listType = CoreML.Specification.MILSpec.ListType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.tupleType = CoreML.Specification.MILSpec.TupleType.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dictionaryType = CoreML.Specification.MILSpec.DictionaryType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.ValueType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensorType":
                    message.tensorType = CoreML.Specification.MILSpec.TensorType.decodeText(reader);
                    break;
                case "listType":
                    message.listType = CoreML.Specification.MILSpec.ListType.decodeText(reader);
                    break;
                case "tupleType":
                    message.tupleType = CoreML.Specification.MILSpec.TupleType.decodeText(reader);
                    break;
                case "dictionaryType":
                    message.dictionaryType = CoreML.Specification.MILSpec.DictionaryType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.DataType = {
    "UNUSED_TYPE": 0,
    "BOOL": 1,
    "STRING": 2,
    "FLOAT16": 10,
    "FLOAT32": 11,
    "FLOAT64": 12,
    "BFLOAT16": 13,
    "INT8": 21,
    "INT16": 22,
    "INT32": 23,
    "INT64": 24,
    "UINT8": 31,
    "UINT16": 32,
    "UINT32": 33,
    "UINT64": 34
};

CoreML.Specification.MILSpec.TensorType = class TensorType {

    constructor() {
        this.dimensions = [];
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dataType = reader.int32();
                    break;
                case 2:
                    message.rank = reader.int64();
                    break;
                case 3:
                    message.dimensions.push(CoreML.Specification.MILSpec.Dimension.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dataType":
                    message.dataType = reader.enum(CoreML.Specification.MILSpec.DataType);
                    break;
                case "rank":
                    message.rank = reader.int64();
                    break;
                case "dimensions":
                    message.dimensions.push(CoreML.Specification.MILSpec.Dimension.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorType.prototype.dataType = 0;
CoreML.Specification.MILSpec.TensorType.prototype.rank = 0n;

CoreML.Specification.MILSpec.TupleType = class TupleType {

    constructor() {
        this.types = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TupleType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.types.push(CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TupleType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "types":
                    message.types.push(CoreML.Specification.MILSpec.ValueType.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.ListType = class ListType {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.ListType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.length = CoreML.Specification.MILSpec.Dimension.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.ListType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                case "length":
                    message.length = CoreML.Specification.MILSpec.Dimension.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.ListType.prototype.type = null;
CoreML.Specification.MILSpec.ListType.prototype.length = null;

CoreML.Specification.MILSpec.DictionaryType = class DictionaryType {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.DictionaryType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.keyType = CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.valueType = CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.DictionaryType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "keyType":
                    message.keyType = CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                case "valueType":
                    message.valueType = CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.DictionaryType.prototype.keyType = null;
CoreML.Specification.MILSpec.DictionaryType.prototype.valueType = null;

CoreML.Specification.MILSpec.Dimension = class Dimension {

    get dimension() {
        CoreML.Specification.MILSpec.Dimension.dimensionSet = CoreML.Specification.MILSpec.Dimension.dimensionSet || new Set(["constant", "unknown"]);
        return Object.keys(this).find((key) => CoreML.Specification.MILSpec.Dimension.dimensionSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Dimension();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.constant = CoreML.Specification.MILSpec.Dimension.ConstantDimension.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.unknown = CoreML.Specification.MILSpec.Dimension.UnknownDimension.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Dimension();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "constant":
                    message.constant = CoreML.Specification.MILSpec.Dimension.ConstantDimension.decodeText(reader);
                    break;
                case "unknown":
                    message.unknown = CoreML.Specification.MILSpec.Dimension.UnknownDimension.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Dimension.ConstantDimension = class ConstantDimension {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Dimension.ConstantDimension();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.size = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Dimension.ConstantDimension();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    message.size = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Dimension.ConstantDimension.prototype.size = 0n;

CoreML.Specification.MILSpec.Dimension.UnknownDimension = class UnknownDimension {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Dimension.UnknownDimension();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variadic = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Dimension.UnknownDimension();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variadic":
                    message.variadic = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Dimension.UnknownDimension.prototype.variadic = false;

CoreML.Specification.MILSpec.Value = class Value {

    get value() {
        CoreML.Specification.MILSpec.Value.valueSet = CoreML.Specification.MILSpec.Value.valueSet || new Set(["immediateValue", "blobFileValue"]);
        return Object.keys(this).find((key) => CoreML.Specification.MILSpec.Value.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Value();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.docString = reader.string();
                    break;
                case 2:
                    message.type = CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.immediateValue = CoreML.Specification.MILSpec.Value.ImmediateValue.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.blobFileValue = CoreML.Specification.MILSpec.Value.BlobFileValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Value();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "docString":
                    message.docString = reader.string();
                    break;
                case "type":
                    message.type = CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                case "immediateValue":
                    message.immediateValue = CoreML.Specification.MILSpec.Value.ImmediateValue.decodeText(reader);
                    break;
                case "blobFileValue":
                    message.blobFileValue = CoreML.Specification.MILSpec.Value.BlobFileValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Value.prototype.docString = "";
CoreML.Specification.MILSpec.Value.prototype.type = null;

CoreML.Specification.MILSpec.Value.ImmediateValue = class ImmediateValue {

    get value() {
        CoreML.Specification.MILSpec.Value.ImmediateValue.valueSet = CoreML.Specification.MILSpec.Value.ImmediateValue.valueSet || new Set(["tensor", "tuple", "list", "dictionary"]);
        return Object.keys(this).find((key) => CoreML.Specification.MILSpec.Value.ImmediateValue.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Value.ImmediateValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor = CoreML.Specification.MILSpec.TensorValue.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.tuple = CoreML.Specification.MILSpec.TupleValue.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.list = CoreML.Specification.MILSpec.ListValue.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dictionary = CoreML.Specification.MILSpec.DictionaryValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Value.ImmediateValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor = CoreML.Specification.MILSpec.TensorValue.decodeText(reader);
                    break;
                case "tuple":
                    message.tuple = CoreML.Specification.MILSpec.TupleValue.decodeText(reader);
                    break;
                case "list":
                    message.list = CoreML.Specification.MILSpec.ListValue.decodeText(reader);
                    break;
                case "dictionary":
                    message.dictionary = CoreML.Specification.MILSpec.DictionaryValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Value.BlobFileValue = class BlobFileValue {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.Value.BlobFileValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.fileName = reader.string();
                    break;
                case 2:
                    message.offset = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.Value.BlobFileValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "fileName":
                    message.fileName = reader.string();
                    break;
                case "offset":
                    message.offset = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.Value.BlobFileValue.prototype.fileName = "";
CoreML.Specification.MILSpec.Value.BlobFileValue.prototype.offset = 0n;

CoreML.Specification.MILSpec.TensorValue = class TensorValue {

    get value() {
        CoreML.Specification.MILSpec.TensorValue.valueSet = CoreML.Specification.MILSpec.TensorValue.valueSet || new Set(["floats", "ints", "bools", "strings", "longInts", "doubles", "bytes"]);
        return Object.keys(this).find((key) => CoreML.Specification.MILSpec.TensorValue.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.floats = CoreML.Specification.MILSpec.TensorValue.RepeatedFloats.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.ints = CoreML.Specification.MILSpec.TensorValue.RepeatedInts.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.bools = CoreML.Specification.MILSpec.TensorValue.RepeatedBools.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.strings = CoreML.Specification.MILSpec.TensorValue.RepeatedStrings.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.longInts = CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.doubles = CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.bytes = CoreML.Specification.MILSpec.TensorValue.RepeatedBytes.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "floats":
                    message.floats = CoreML.Specification.MILSpec.TensorValue.RepeatedFloats.decodeText(reader);
                    break;
                case "ints":
                    message.ints = CoreML.Specification.MILSpec.TensorValue.RepeatedInts.decodeText(reader);
                    break;
                case "bools":
                    message.bools = CoreML.Specification.MILSpec.TensorValue.RepeatedBools.decodeText(reader);
                    break;
                case "strings":
                    message.strings = CoreML.Specification.MILSpec.TensorValue.RepeatedStrings.decodeText(reader);
                    break;
                case "longInts":
                    message.longInts = CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts.decodeText(reader);
                    break;
                case "doubles":
                    message.doubles = CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles.decodeText(reader);
                    break;
                case "bytes":
                    message.bytes = CoreML.Specification.MILSpec.TensorValue.RepeatedBytes.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedFloats = class RepeatedFloats {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedFloats();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = reader.floats(message.values, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedFloats();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles = class RepeatedDoubles {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedInts = class RepeatedInts {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedInts();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = reader.array(message.values, () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedInts();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.int32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts = class RepeatedLongInts {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedBools = class RepeatedBools {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedBools();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = reader.array(message.values, () => reader.bool(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedBools();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.bool());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedStrings = class RepeatedStrings {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedStrings();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedStrings();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedBytes = class RepeatedBytes {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedBytes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TensorValue.RepeatedBytes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.TensorValue.RepeatedBytes.prototype.values = new Uint8Array([]);

CoreML.Specification.MILSpec.TupleValue = class TupleValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.TupleValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push(CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.TupleValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push(CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.ListValue = class ListValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.ListValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push(CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push(CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.DictionaryValue = class DictionaryValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.DictionaryValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push(CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.DictionaryValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push(CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair = class KeyValuePair {

    static decode(reader, length) {
        const message = new CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.value = CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = CoreML.Specification.MILSpec.Value.decodeText(reader);
                    break;
                case "value":
                    message.value = CoreML.Specification.MILSpec.Value.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.prototype.key = null;
CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.prototype.value = null;

CoreML.Specification.NeuralNetworkMultiArrayShapeMapping = {
    "RANK5_ARRAY_MAPPING": 0,
    "EXACT_ARRAY_MAPPING": 1
};

CoreML.Specification.NeuralNetworkImageShapeMapping = {
    "RANK5_IMAGE_MAPPING": 0,
    "RANK4_IMAGE_MAPPING": 1
};

CoreML.Specification.NeuralNetwork = class NeuralNetwork {

    constructor() {
        this.layers = [];
        this.preprocessing = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetwork();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layers.push(CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.preprocessing.push(CoreML.Specification.NeuralNetworkPreprocessing.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.arrayInputShapeMapping = reader.int32();
                    break;
                case 6:
                    message.imageInputShapeMapping = reader.int32();
                    break;
                case 10:
                    message.updateParams = CoreML.Specification.NetworkUpdateParameters.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetwork();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layers":
                    message.layers.push(CoreML.Specification.NeuralNetworkLayer.decodeText(reader));
                    break;
                case "preprocessing":
                    message.preprocessing.push(CoreML.Specification.NeuralNetworkPreprocessing.decodeText(reader));
                    break;
                case "arrayInputShapeMapping":
                    message.arrayInputShapeMapping = reader.enum(CoreML.Specification.NeuralNetworkMultiArrayShapeMapping);
                    break;
                case "imageInputShapeMapping":
                    message.imageInputShapeMapping = reader.enum(CoreML.Specification.NeuralNetworkImageShapeMapping);
                    break;
                case "updateParams":
                    message.updateParams = CoreML.Specification.NetworkUpdateParameters.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetwork.prototype.arrayInputShapeMapping = 0;
CoreML.Specification.NeuralNetwork.prototype.imageInputShapeMapping = 0;
CoreML.Specification.NeuralNetwork.prototype.updateParams = null;

CoreML.Specification.NeuralNetworkImageScaler = class NeuralNetworkImageScaler {

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetworkImageScaler();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetworkImageScaler();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "channelScale":
                    message.channelScale = reader.float();
                    break;
                case "blueBias":
                    message.blueBias = reader.float();
                    break;
                case "greenBias":
                    message.greenBias = reader.float();
                    break;
                case "redBias":
                    message.redBias = reader.float();
                    break;
                case "grayBias":
                    message.grayBias = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetworkImageScaler.prototype.channelScale = 0;
CoreML.Specification.NeuralNetworkImageScaler.prototype.blueBias = 0;
CoreML.Specification.NeuralNetworkImageScaler.prototype.greenBias = 0;
CoreML.Specification.NeuralNetworkImageScaler.prototype.redBias = 0;
CoreML.Specification.NeuralNetworkImageScaler.prototype.grayBias = 0;

CoreML.Specification.NeuralNetworkMeanImage = class NeuralNetworkMeanImage {

    constructor() {
        this.meanImage = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetworkMeanImage();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetworkMeanImage();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meanImage":
                    reader.array(message.meanImage, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetworkPreprocessing = class NeuralNetworkPreprocessing {

    get preprocessor() {
        CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet = CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet || new Set(["scaler", "meanImage"]);
        return Object.keys(this).find((key) => CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetworkPreprocessing();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.featureName = reader.string();
                    break;
                case 10:
                    message.scaler = CoreML.Specification.NeuralNetworkImageScaler.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.meanImage = CoreML.Specification.NeuralNetworkMeanImage.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetworkPreprocessing();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "featureName":
                    message.featureName = reader.string();
                    break;
                case "scaler":
                    message.scaler = CoreML.Specification.NeuralNetworkImageScaler.decodeText(reader);
                    break;
                case "meanImage":
                    message.meanImage = CoreML.Specification.NeuralNetworkMeanImage.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetworkPreprocessing.prototype.featureName = "";

CoreML.Specification.ActivationReLU = class ActivationReLU {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationReLU();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationReLU();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationLeakyReLU = class ActivationLeakyReLU {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationLeakyReLU();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationLeakyReLU();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationLeakyReLU.prototype.alpha = 0;

CoreML.Specification.ActivationTanh = class ActivationTanh {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationTanh();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationTanh();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationScaledTanh = class ActivationScaledTanh {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationScaledTanh();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationScaledTanh();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationScaledTanh.prototype.alpha = 0;
CoreML.Specification.ActivationScaledTanh.prototype.beta = 0;

CoreML.Specification.ActivationSigmoid = class ActivationSigmoid {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationSigmoid();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationSigmoid();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationLinear = class ActivationLinear {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationLinear();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationLinear();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationLinear.prototype.alpha = 0;
CoreML.Specification.ActivationLinear.prototype.beta = 0;

CoreML.Specification.ActivationSigmoidHard = class ActivationSigmoidHard {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationSigmoidHard();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationSigmoidHard();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationSigmoidHard.prototype.alpha = 0;
CoreML.Specification.ActivationSigmoidHard.prototype.beta = 0;

CoreML.Specification.ActivationPReLU = class ActivationPReLU {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationPReLU();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationPReLU();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationPReLU.prototype.alpha = null;

CoreML.Specification.ActivationELU = class ActivationELU {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationELU();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationELU();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationELU.prototype.alpha = 0;

CoreML.Specification.ActivationThresholdedReLU = class ActivationThresholdedReLU {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationThresholdedReLU();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationThresholdedReLU();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationThresholdedReLU.prototype.alpha = 0;

CoreML.Specification.ActivationSoftsign = class ActivationSoftsign {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationSoftsign();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationSoftsign();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationSoftplus = class ActivationSoftplus {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationSoftplus();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationSoftplus();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationParametricSoftplus = class ActivationParametricSoftplus {

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationParametricSoftplus();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.beta = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationParametricSoftplus();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "beta":
                    message.beta = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ActivationParametricSoftplus.prototype.alpha = null;
CoreML.Specification.ActivationParametricSoftplus.prototype.beta = null;

CoreML.Specification.ActivationParams = class ActivationParams {

    get NonlinearityType() {
        CoreML.Specification.ActivationParams.NonlinearityTypeSet = CoreML.Specification.ActivationParams.NonlinearityTypeSet || new Set(["linear", "ReLU", "leakyReLU", "thresholdedReLU", "PReLU", "tanh", "scaledTanh", "sigmoid", "sigmoidHard", "ELU", "softsign", "softplus", "parametricSoftplus"]);
        return Object.keys(this).find((key) => CoreML.Specification.ActivationParams.NonlinearityTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ActivationParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 5:
                    message.linear = CoreML.Specification.ActivationLinear.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.ReLU = CoreML.Specification.ActivationReLU.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.leakyReLU = CoreML.Specification.ActivationLeakyReLU.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.thresholdedReLU = CoreML.Specification.ActivationThresholdedReLU.decode(reader, reader.uint32());
                    break;
                case 25:
                    message.PReLU = CoreML.Specification.ActivationPReLU.decode(reader, reader.uint32());
                    break;
                case 30:
                    message.tanh = CoreML.Specification.ActivationTanh.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.scaledTanh = CoreML.Specification.ActivationScaledTanh.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.sigmoid = CoreML.Specification.ActivationSigmoid.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.sigmoidHard = CoreML.Specification.ActivationSigmoidHard.decode(reader, reader.uint32());
                    break;
                case 50:
                    message.ELU = CoreML.Specification.ActivationELU.decode(reader, reader.uint32());
                    break;
                case 60:
                    message.softsign = CoreML.Specification.ActivationSoftsign.decode(reader, reader.uint32());
                    break;
                case 70:
                    message.softplus = CoreML.Specification.ActivationSoftplus.decode(reader, reader.uint32());
                    break;
                case 71:
                    message.parametricSoftplus = CoreML.Specification.ActivationParametricSoftplus.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ActivationParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linear":
                    message.linear = CoreML.Specification.ActivationLinear.decodeText(reader);
                    break;
                case "ReLU":
                    message.ReLU = CoreML.Specification.ActivationReLU.decodeText(reader);
                    break;
                case "leakyReLU":
                    message.leakyReLU = CoreML.Specification.ActivationLeakyReLU.decodeText(reader);
                    break;
                case "thresholdedReLU":
                    message.thresholdedReLU = CoreML.Specification.ActivationThresholdedReLU.decodeText(reader);
                    break;
                case "PReLU":
                    message.PReLU = CoreML.Specification.ActivationPReLU.decodeText(reader);
                    break;
                case "tanh":
                    message.tanh = CoreML.Specification.ActivationTanh.decodeText(reader);
                    break;
                case "scaledTanh":
                    message.scaledTanh = CoreML.Specification.ActivationScaledTanh.decodeText(reader);
                    break;
                case "sigmoid":
                    message.sigmoid = CoreML.Specification.ActivationSigmoid.decodeText(reader);
                    break;
                case "sigmoidHard":
                    message.sigmoidHard = CoreML.Specification.ActivationSigmoidHard.decodeText(reader);
                    break;
                case "ELU":
                    message.ELU = CoreML.Specification.ActivationELU.decodeText(reader);
                    break;
                case "softsign":
                    message.softsign = CoreML.Specification.ActivationSoftsign.decodeText(reader);
                    break;
                case "softplus":
                    message.softplus = CoreML.Specification.ActivationSoftplus.decodeText(reader);
                    break;
                case "parametricSoftplus":
                    message.parametricSoftplus = CoreML.Specification.ActivationParametricSoftplus.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Tensor = class Tensor {

    constructor() {
        this.dimValue = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Tensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Tensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "rank":
                    message.rank = reader.uint32();
                    break;
                case "dimValue":
                    reader.array(message.dimValue, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Tensor.prototype.rank = 0;

CoreML.Specification.NeuralNetworkLayer = class NeuralNetworkLayer {

    constructor() {
        this.input = [];
        this.output = [];
        this.inputTensor = [];
        this.outputTensor = [];
    }

    get layer() {
        CoreML.Specification.NeuralNetworkLayer.layerSet = CoreML.Specification.NeuralNetworkLayer.layerSet || new Set(["convolution", "pooling", "activation", "innerProduct", "embedding", "batchnorm", "mvn", "l2normalize", "softmax", "lrn", "crop", "padding", "upsample", "resizeBilinear", "cropResize", "unary", "add", "multiply", "average", "scale", "bias", "max", "min", "dot", "reduce", "loadConstant", "reshape", "flatten", "permute", "concat", "split", "sequenceRepeat", "reorganizeData", "slice", "simpleRecurrent", "gru", "uniDirectionalLSTM", "biDirectionalLSTM", "custom", "copy", "branch", "loop", "loopBreak", "loopContinue", "rangeStatic", "rangeDynamic", "clip", "ceil", "floor", "sign", "round", "exp2", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "gelu", "equal", "notEqual", "lessThan", "lessEqual", "greaterThan", "greaterEqual", "logicalOr", "logicalXor", "logicalNot", "logicalAnd", "modBroadcastable", "minBroadcastable", "maxBroadcastable", "addBroadcastable", "powBroadcastable", "divideBroadcastable", "floorDivBroadcastable", "multiplyBroadcastable", "subtractBroadcastable", "tile", "stack", "gather", "scatter", "gatherND", "scatterND", "softmaxND", "gatherAlongAxis", "scatterAlongAxis", "reverse", "reverseSeq", "splitND", "concatND", "transpose", "sliceStatic", "sliceDynamic", "slidingWindows", "topK", "argMin", "argMax", "embeddingND", "batchedMatmul", "getShape", "loadConstantND", "fillLike", "fillStatic", "fillDynamic", "broadcastToLike", "broadcastToStatic", "broadcastToDynamic", "squeeze", "expandDims", "flattenTo2D", "reshapeLike", "reshapeStatic", "reshapeDynamic", "rankPreservingReshape", "constantPad", "randomNormalLike", "randomNormalStatic", "randomNormalDynamic", "randomUniformLike", "randomUniformStatic", "randomUniformDynamic", "randomBernoulliLike", "randomBernoulliStatic", "randomBernoulliDynamic", "categoricalDistribution", "reduceL1", "reduceL2", "reduceMax", "reduceMin", "reduceSum", "reduceProd", "reduceMean", "reduceLogSum", "reduceSumSquare", "reduceLogSumExp", "whereNonZero", "matrixBandPart", "lowerTriangular", "upperTriangular", "whereBroadcastable", "layerNormalization", "NonMaximumSuppression", "oneHot", "cumSum", "clampedReLU", "argSort", "pooling3d", "globalPooling3d", "sliceBySize", "convolution3d"]);
        return Object.keys(this).find((key) => CoreML.Specification.NeuralNetworkLayer.layerSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetworkLayer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.inputTensor.push(CoreML.Specification.Tensor.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.outputTensor.push(CoreML.Specification.Tensor.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.isUpdatable = reader.bool();
                    break;
                case 100:
                    message.convolution = CoreML.Specification.ConvolutionLayerParams.decode(reader, reader.uint32());
                    break;
                case 120:
                    message.pooling = CoreML.Specification.PoolingLayerParams.decode(reader, reader.uint32());
                    break;
                case 130:
                    message.activation = CoreML.Specification.ActivationParams.decode(reader, reader.uint32());
                    break;
                case 140:
                    message.innerProduct = CoreML.Specification.InnerProductLayerParams.decode(reader, reader.uint32());
                    break;
                case 150:
                    message.embedding = CoreML.Specification.EmbeddingLayerParams.decode(reader, reader.uint32());
                    break;
                case 160:
                    message.batchnorm = CoreML.Specification.BatchnormLayerParams.decode(reader, reader.uint32());
                    break;
                case 165:
                    message.mvn = CoreML.Specification.MeanVarianceNormalizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 170:
                    message.l2normalize = CoreML.Specification.L2NormalizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 175:
                    message.softmax = CoreML.Specification.SoftmaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 180:
                    message.lrn = CoreML.Specification.LRNLayerParams.decode(reader, reader.uint32());
                    break;
                case 190:
                    message.crop = CoreML.Specification.CropLayerParams.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.padding = CoreML.Specification.PaddingLayerParams.decode(reader, reader.uint32());
                    break;
                case 210:
                    message.upsample = CoreML.Specification.UpsampleLayerParams.decode(reader, reader.uint32());
                    break;
                case 211:
                    message.resizeBilinear = CoreML.Specification.ResizeBilinearLayerParams.decode(reader, reader.uint32());
                    break;
                case 212:
                    message.cropResize = CoreML.Specification.CropResizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 220:
                    message.unary = CoreML.Specification.UnaryFunctionLayerParams.decode(reader, reader.uint32());
                    break;
                case 230:
                    message.add = CoreML.Specification.AddLayerParams.decode(reader, reader.uint32());
                    break;
                case 231:
                    message.multiply = CoreML.Specification.MultiplyLayerParams.decode(reader, reader.uint32());
                    break;
                case 240:
                    message.average = CoreML.Specification.AverageLayerParams.decode(reader, reader.uint32());
                    break;
                case 245:
                    message.scale = CoreML.Specification.ScaleLayerParams.decode(reader, reader.uint32());
                    break;
                case 250:
                    message.bias = CoreML.Specification.BiasLayerParams.decode(reader, reader.uint32());
                    break;
                case 260:
                    message.max = CoreML.Specification.MaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 261:
                    message.min = CoreML.Specification.MinLayerParams.decode(reader, reader.uint32());
                    break;
                case 270:
                    message.dot = CoreML.Specification.DotProductLayerParams.decode(reader, reader.uint32());
                    break;
                case 280:
                    message.reduce = CoreML.Specification.ReduceLayerParams.decode(reader, reader.uint32());
                    break;
                case 290:
                    message.loadConstant = CoreML.Specification.LoadConstantLayerParams.decode(reader, reader.uint32());
                    break;
                case 300:
                    message.reshape = CoreML.Specification.ReshapeLayerParams.decode(reader, reader.uint32());
                    break;
                case 301:
                    message.flatten = CoreML.Specification.FlattenLayerParams.decode(reader, reader.uint32());
                    break;
                case 310:
                    message.permute = CoreML.Specification.PermuteLayerParams.decode(reader, reader.uint32());
                    break;
                case 320:
                    message.concat = CoreML.Specification.ConcatLayerParams.decode(reader, reader.uint32());
                    break;
                case 330:
                    message.split = CoreML.Specification.SplitLayerParams.decode(reader, reader.uint32());
                    break;
                case 340:
                    message.sequenceRepeat = CoreML.Specification.SequenceRepeatLayerParams.decode(reader, reader.uint32());
                    break;
                case 345:
                    message.reorganizeData = CoreML.Specification.ReorganizeDataLayerParams.decode(reader, reader.uint32());
                    break;
                case 350:
                    message.slice = CoreML.Specification.SliceLayerParams.decode(reader, reader.uint32());
                    break;
                case 400:
                    message.simpleRecurrent = CoreML.Specification.SimpleRecurrentLayerParams.decode(reader, reader.uint32());
                    break;
                case 410:
                    message.gru = CoreML.Specification.GRULayerParams.decode(reader, reader.uint32());
                    break;
                case 420:
                    message.uniDirectionalLSTM = CoreML.Specification.UniDirectionalLSTMLayerParams.decode(reader, reader.uint32());
                    break;
                case 430:
                    message.biDirectionalLSTM = CoreML.Specification.BiDirectionalLSTMLayerParams.decode(reader, reader.uint32());
                    break;
                case 500:
                    message.custom = CoreML.Specification.CustomLayerParams.decode(reader, reader.uint32());
                    break;
                case 600:
                    message.copy = CoreML.Specification.CopyLayerParams.decode(reader, reader.uint32());
                    break;
                case 605:
                    message.branch = CoreML.Specification.BranchLayerParams.decode(reader, reader.uint32());
                    break;
                case 615:
                    message.loop = CoreML.Specification.LoopLayerParams.decode(reader, reader.uint32());
                    break;
                case 620:
                    message.loopBreak = CoreML.Specification.LoopBreakLayerParams.decode(reader, reader.uint32());
                    break;
                case 625:
                    message.loopContinue = CoreML.Specification.LoopContinueLayerParams.decode(reader, reader.uint32());
                    break;
                case 635:
                    message.rangeStatic = CoreML.Specification.RangeStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 640:
                    message.rangeDynamic = CoreML.Specification.RangeDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 660:
                    message.clip = CoreML.Specification.ClipLayerParams.decode(reader, reader.uint32());
                    break;
                case 665:
                    message.ceil = CoreML.Specification.CeilLayerParams.decode(reader, reader.uint32());
                    break;
                case 670:
                    message.floor = CoreML.Specification.FloorLayerParams.decode(reader, reader.uint32());
                    break;
                case 680:
                    message.sign = CoreML.Specification.SignLayerParams.decode(reader, reader.uint32());
                    break;
                case 685:
                    message.round = CoreML.Specification.RoundLayerParams.decode(reader, reader.uint32());
                    break;
                case 700:
                    message.exp2 = CoreML.Specification.Exp2LayerParams.decode(reader, reader.uint32());
                    break;
                case 710:
                    message.sin = CoreML.Specification.SinLayerParams.decode(reader, reader.uint32());
                    break;
                case 715:
                    message.cos = CoreML.Specification.CosLayerParams.decode(reader, reader.uint32());
                    break;
                case 720:
                    message.tan = CoreML.Specification.TanLayerParams.decode(reader, reader.uint32());
                    break;
                case 730:
                    message.asin = CoreML.Specification.AsinLayerParams.decode(reader, reader.uint32());
                    break;
                case 735:
                    message.acos = CoreML.Specification.AcosLayerParams.decode(reader, reader.uint32());
                    break;
                case 740:
                    message.atan = CoreML.Specification.AtanLayerParams.decode(reader, reader.uint32());
                    break;
                case 750:
                    message.sinh = CoreML.Specification.SinhLayerParams.decode(reader, reader.uint32());
                    break;
                case 755:
                    message.cosh = CoreML.Specification.CoshLayerParams.decode(reader, reader.uint32());
                    break;
                case 760:
                    message.tanh = CoreML.Specification.TanhLayerParams.decode(reader, reader.uint32());
                    break;
                case 770:
                    message.asinh = CoreML.Specification.AsinhLayerParams.decode(reader, reader.uint32());
                    break;
                case 775:
                    message.acosh = CoreML.Specification.AcoshLayerParams.decode(reader, reader.uint32());
                    break;
                case 780:
                    message.atanh = CoreML.Specification.AtanhLayerParams.decode(reader, reader.uint32());
                    break;
                case 790:
                    message.erf = CoreML.Specification.ErfLayerParams.decode(reader, reader.uint32());
                    break;
                case 795:
                    message.gelu = CoreML.Specification.GeluLayerParams.decode(reader, reader.uint32());
                    break;
                case 815:
                    message.equal = CoreML.Specification.EqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 820:
                    message.notEqual = CoreML.Specification.NotEqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 825:
                    message.lessThan = CoreML.Specification.LessThanLayerParams.decode(reader, reader.uint32());
                    break;
                case 827:
                    message.lessEqual = CoreML.Specification.LessEqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 830:
                    message.greaterThan = CoreML.Specification.GreaterThanLayerParams.decode(reader, reader.uint32());
                    break;
                case 832:
                    message.greaterEqual = CoreML.Specification.GreaterEqualLayerParams.decode(reader, reader.uint32());
                    break;
                case 840:
                    message.logicalOr = CoreML.Specification.LogicalOrLayerParams.decode(reader, reader.uint32());
                    break;
                case 845:
                    message.logicalXor = CoreML.Specification.LogicalXorLayerParams.decode(reader, reader.uint32());
                    break;
                case 850:
                    message.logicalNot = CoreML.Specification.LogicalNotLayerParams.decode(reader, reader.uint32());
                    break;
                case 855:
                    message.logicalAnd = CoreML.Specification.LogicalAndLayerParams.decode(reader, reader.uint32());
                    break;
                case 865:
                    message.modBroadcastable = CoreML.Specification.ModBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 870:
                    message.minBroadcastable = CoreML.Specification.MinBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 875:
                    message.maxBroadcastable = CoreML.Specification.MaxBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 880:
                    message.addBroadcastable = CoreML.Specification.AddBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 885:
                    message.powBroadcastable = CoreML.Specification.PowBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 890:
                    message.divideBroadcastable = CoreML.Specification.DivideBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 895:
                    message.floorDivBroadcastable = CoreML.Specification.FloorDivBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 900:
                    message.multiplyBroadcastable = CoreML.Specification.MultiplyBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 905:
                    message.subtractBroadcastable = CoreML.Specification.SubtractBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 920:
                    message.tile = CoreML.Specification.TileLayerParams.decode(reader, reader.uint32());
                    break;
                case 925:
                    message.stack = CoreML.Specification.StackLayerParams.decode(reader, reader.uint32());
                    break;
                case 930:
                    message.gather = CoreML.Specification.GatherLayerParams.decode(reader, reader.uint32());
                    break;
                case 935:
                    message.scatter = CoreML.Specification.ScatterLayerParams.decode(reader, reader.uint32());
                    break;
                case 940:
                    message.gatherND = CoreML.Specification.GatherNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 945:
                    message.scatterND = CoreML.Specification.ScatterNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 950:
                    message.softmaxND = CoreML.Specification.SoftmaxNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 952:
                    message.gatherAlongAxis = CoreML.Specification.GatherAlongAxisLayerParams.decode(reader, reader.uint32());
                    break;
                case 954:
                    message.scatterAlongAxis = CoreML.Specification.ScatterAlongAxisLayerParams.decode(reader, reader.uint32());
                    break;
                case 960:
                    message.reverse = CoreML.Specification.ReverseLayerParams.decode(reader, reader.uint32());
                    break;
                case 965:
                    message.reverseSeq = CoreML.Specification.ReverseSeqLayerParams.decode(reader, reader.uint32());
                    break;
                case 975:
                    message.splitND = CoreML.Specification.SplitNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 980:
                    message.concatND = CoreML.Specification.ConcatNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 985:
                    message.transpose = CoreML.Specification.TransposeLayerParams.decode(reader, reader.uint32());
                    break;
                case 995:
                    message.sliceStatic = CoreML.Specification.SliceStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1000:
                    message.sliceDynamic = CoreML.Specification.SliceDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1005:
                    message.slidingWindows = CoreML.Specification.SlidingWindowsLayerParams.decode(reader, reader.uint32());
                    break;
                case 1015:
                    message.topK = CoreML.Specification.TopKLayerParams.decode(reader, reader.uint32());
                    break;
                case 1020:
                    message.argMin = CoreML.Specification.ArgMinLayerParams.decode(reader, reader.uint32());
                    break;
                case 1025:
                    message.argMax = CoreML.Specification.ArgMaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 1040:
                    message.embeddingND = CoreML.Specification.EmbeddingNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 1045:
                    message.batchedMatmul = CoreML.Specification.BatchedMatMulLayerParams.decode(reader, reader.uint32());
                    break;
                case 1065:
                    message.getShape = CoreML.Specification.GetShapeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1070:
                    message.loadConstantND = CoreML.Specification.LoadConstantNDLayerParams.decode(reader, reader.uint32());
                    break;
                case 1080:
                    message.fillLike = CoreML.Specification.FillLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1085:
                    message.fillStatic = CoreML.Specification.FillStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1090:
                    message.fillDynamic = CoreML.Specification.FillDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1100:
                    message.broadcastToLike = CoreML.Specification.BroadcastToLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1105:
                    message.broadcastToStatic = CoreML.Specification.BroadcastToStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1110:
                    message.broadcastToDynamic = CoreML.Specification.BroadcastToDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1120:
                    message.squeeze = CoreML.Specification.SqueezeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1125:
                    message.expandDims = CoreML.Specification.ExpandDimsLayerParams.decode(reader, reader.uint32());
                    break;
                case 1130:
                    message.flattenTo2D = CoreML.Specification.FlattenTo2DLayerParams.decode(reader, reader.uint32());
                    break;
                case 1135:
                    message.reshapeLike = CoreML.Specification.ReshapeLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1140:
                    message.reshapeStatic = CoreML.Specification.ReshapeStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1145:
                    message.reshapeDynamic = CoreML.Specification.ReshapeDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1150:
                    message.rankPreservingReshape = CoreML.Specification.RankPreservingReshapeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1155:
                    message.constantPad = CoreML.Specification.ConstantPaddingLayerParams.decode(reader, reader.uint32());
                    break;
                case 1170:
                    message.randomNormalLike = CoreML.Specification.RandomNormalLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1175:
                    message.randomNormalStatic = CoreML.Specification.RandomNormalStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1180:
                    message.randomNormalDynamic = CoreML.Specification.RandomNormalDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1190:
                    message.randomUniformLike = CoreML.Specification.RandomUniformLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1195:
                    message.randomUniformStatic = CoreML.Specification.RandomUniformStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1200:
                    message.randomUniformDynamic = CoreML.Specification.RandomUniformDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1210:
                    message.randomBernoulliLike = CoreML.Specification.RandomBernoulliLikeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1215:
                    message.randomBernoulliStatic = CoreML.Specification.RandomBernoulliStaticLayerParams.decode(reader, reader.uint32());
                    break;
                case 1220:
                    message.randomBernoulliDynamic = CoreML.Specification.RandomBernoulliDynamicLayerParams.decode(reader, reader.uint32());
                    break;
                case 1230:
                    message.categoricalDistribution = CoreML.Specification.CategoricalDistributionLayerParams.decode(reader, reader.uint32());
                    break;
                case 1250:
                    message.reduceL1 = CoreML.Specification.ReduceL1LayerParams.decode(reader, reader.uint32());
                    break;
                case 1255:
                    message.reduceL2 = CoreML.Specification.ReduceL2LayerParams.decode(reader, reader.uint32());
                    break;
                case 1260:
                    message.reduceMax = CoreML.Specification.ReduceMaxLayerParams.decode(reader, reader.uint32());
                    break;
                case 1265:
                    message.reduceMin = CoreML.Specification.ReduceMinLayerParams.decode(reader, reader.uint32());
                    break;
                case 1270:
                    message.reduceSum = CoreML.Specification.ReduceSumLayerParams.decode(reader, reader.uint32());
                    break;
                case 1275:
                    message.reduceProd = CoreML.Specification.ReduceProdLayerParams.decode(reader, reader.uint32());
                    break;
                case 1280:
                    message.reduceMean = CoreML.Specification.ReduceMeanLayerParams.decode(reader, reader.uint32());
                    break;
                case 1285:
                    message.reduceLogSum = CoreML.Specification.ReduceLogSumLayerParams.decode(reader, reader.uint32());
                    break;
                case 1290:
                    message.reduceSumSquare = CoreML.Specification.ReduceSumSquareLayerParams.decode(reader, reader.uint32());
                    break;
                case 1295:
                    message.reduceLogSumExp = CoreML.Specification.ReduceLogSumExpLayerParams.decode(reader, reader.uint32());
                    break;
                case 1313:
                    message.whereNonZero = CoreML.Specification.WhereNonZeroLayerParams.decode(reader, reader.uint32());
                    break;
                case 1315:
                    message.matrixBandPart = CoreML.Specification.MatrixBandPartLayerParams.decode(reader, reader.uint32());
                    break;
                case 1320:
                    message.lowerTriangular = CoreML.Specification.LowerTriangularLayerParams.decode(reader, reader.uint32());
                    break;
                case 1325:
                    message.upperTriangular = CoreML.Specification.UpperTriangularLayerParams.decode(reader, reader.uint32());
                    break;
                case 1330:
                    message.whereBroadcastable = CoreML.Specification.WhereBroadcastableLayerParams.decode(reader, reader.uint32());
                    break;
                case 1350:
                    message.layerNormalization = CoreML.Specification.LayerNormalizationLayerParams.decode(reader, reader.uint32());
                    break;
                case 1400:
                    message.NonMaximumSuppression = CoreML.Specification.NonMaximumSuppressionLayerParams.decode(reader, reader.uint32());
                    break;
                case 1450:
                    message.oneHot = CoreML.Specification.OneHotLayerParams.decode(reader, reader.uint32());
                    break;
                case 1455:
                    message.cumSum = CoreML.Specification.CumSumLayerParams.decode(reader, reader.uint32());
                    break;
                case 1460:
                    message.clampedReLU = CoreML.Specification.ClampedReLULayerParams.decode(reader, reader.uint32());
                    break;
                case 1461:
                    message.argSort = CoreML.Specification.ArgSortLayerParams.decode(reader, reader.uint32());
                    break;
                case 1465:
                    message.pooling3d = CoreML.Specification.Pooling3DLayerParams.decode(reader, reader.uint32());
                    break;
                case 1466:
                    message.globalPooling3d = CoreML.Specification.GlobalPooling3DLayerParams.decode(reader, reader.uint32());
                    break;
                case 1470:
                    message.sliceBySize = CoreML.Specification.SliceBySizeLayerParams.decode(reader, reader.uint32());
                    break;
                case 1471:
                    message.convolution3d = CoreML.Specification.Convolution3DLayerParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetworkLayer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                case "inputTensor":
                    message.inputTensor.push(CoreML.Specification.Tensor.decodeText(reader));
                    break;
                case "outputTensor":
                    message.outputTensor.push(CoreML.Specification.Tensor.decodeText(reader));
                    break;
                case "isUpdatable":
                    message.isUpdatable = reader.bool();
                    break;
                case "convolution":
                    message.convolution = CoreML.Specification.ConvolutionLayerParams.decodeText(reader);
                    break;
                case "pooling":
                    message.pooling = CoreML.Specification.PoolingLayerParams.decodeText(reader);
                    break;
                case "activation":
                    message.activation = CoreML.Specification.ActivationParams.decodeText(reader);
                    break;
                case "innerProduct":
                    message.innerProduct = CoreML.Specification.InnerProductLayerParams.decodeText(reader);
                    break;
                case "embedding":
                    message.embedding = CoreML.Specification.EmbeddingLayerParams.decodeText(reader);
                    break;
                case "batchnorm":
                    message.batchnorm = CoreML.Specification.BatchnormLayerParams.decodeText(reader);
                    break;
                case "mvn":
                    message.mvn = CoreML.Specification.MeanVarianceNormalizeLayerParams.decodeText(reader);
                    break;
                case "l2normalize":
                    message.l2normalize = CoreML.Specification.L2NormalizeLayerParams.decodeText(reader);
                    break;
                case "softmax":
                    message.softmax = CoreML.Specification.SoftmaxLayerParams.decodeText(reader);
                    break;
                case "lrn":
                    message.lrn = CoreML.Specification.LRNLayerParams.decodeText(reader);
                    break;
                case "crop":
                    message.crop = CoreML.Specification.CropLayerParams.decodeText(reader);
                    break;
                case "padding":
                    message.padding = CoreML.Specification.PaddingLayerParams.decodeText(reader);
                    break;
                case "upsample":
                    message.upsample = CoreML.Specification.UpsampleLayerParams.decodeText(reader);
                    break;
                case "resizeBilinear":
                    message.resizeBilinear = CoreML.Specification.ResizeBilinearLayerParams.decodeText(reader);
                    break;
                case "cropResize":
                    message.cropResize = CoreML.Specification.CropResizeLayerParams.decodeText(reader);
                    break;
                case "unary":
                    message.unary = CoreML.Specification.UnaryFunctionLayerParams.decodeText(reader);
                    break;
                case "add":
                    message.add = CoreML.Specification.AddLayerParams.decodeText(reader);
                    break;
                case "multiply":
                    message.multiply = CoreML.Specification.MultiplyLayerParams.decodeText(reader);
                    break;
                case "average":
                    message.average = CoreML.Specification.AverageLayerParams.decodeText(reader);
                    break;
                case "scale":
                    message.scale = CoreML.Specification.ScaleLayerParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.BiasLayerParams.decodeText(reader);
                    break;
                case "max":
                    message.max = CoreML.Specification.MaxLayerParams.decodeText(reader);
                    break;
                case "min":
                    message.min = CoreML.Specification.MinLayerParams.decodeText(reader);
                    break;
                case "dot":
                    message.dot = CoreML.Specification.DotProductLayerParams.decodeText(reader);
                    break;
                case "reduce":
                    message.reduce = CoreML.Specification.ReduceLayerParams.decodeText(reader);
                    break;
                case "loadConstant":
                    message.loadConstant = CoreML.Specification.LoadConstantLayerParams.decodeText(reader);
                    break;
                case "reshape":
                    message.reshape = CoreML.Specification.ReshapeLayerParams.decodeText(reader);
                    break;
                case "flatten":
                    message.flatten = CoreML.Specification.FlattenLayerParams.decodeText(reader);
                    break;
                case "permute":
                    message.permute = CoreML.Specification.PermuteLayerParams.decodeText(reader);
                    break;
                case "concat":
                    message.concat = CoreML.Specification.ConcatLayerParams.decodeText(reader);
                    break;
                case "split":
                    message.split = CoreML.Specification.SplitLayerParams.decodeText(reader);
                    break;
                case "sequenceRepeat":
                    message.sequenceRepeat = CoreML.Specification.SequenceRepeatLayerParams.decodeText(reader);
                    break;
                case "reorganizeData":
                    message.reorganizeData = CoreML.Specification.ReorganizeDataLayerParams.decodeText(reader);
                    break;
                case "slice":
                    message.slice = CoreML.Specification.SliceLayerParams.decodeText(reader);
                    break;
                case "simpleRecurrent":
                    message.simpleRecurrent = CoreML.Specification.SimpleRecurrentLayerParams.decodeText(reader);
                    break;
                case "gru":
                    message.gru = CoreML.Specification.GRULayerParams.decodeText(reader);
                    break;
                case "uniDirectionalLSTM":
                    message.uniDirectionalLSTM = CoreML.Specification.UniDirectionalLSTMLayerParams.decodeText(reader);
                    break;
                case "biDirectionalLSTM":
                    message.biDirectionalLSTM = CoreML.Specification.BiDirectionalLSTMLayerParams.decodeText(reader);
                    break;
                case "custom":
                    message.custom = CoreML.Specification.CustomLayerParams.decodeText(reader);
                    break;
                case "copy":
                    message.copy = CoreML.Specification.CopyLayerParams.decodeText(reader);
                    break;
                case "branch":
                    message.branch = CoreML.Specification.BranchLayerParams.decodeText(reader);
                    break;
                case "loop":
                    message.loop = CoreML.Specification.LoopLayerParams.decodeText(reader);
                    break;
                case "loopBreak":
                    message.loopBreak = CoreML.Specification.LoopBreakLayerParams.decodeText(reader);
                    break;
                case "loopContinue":
                    message.loopContinue = CoreML.Specification.LoopContinueLayerParams.decodeText(reader);
                    break;
                case "rangeStatic":
                    message.rangeStatic = CoreML.Specification.RangeStaticLayerParams.decodeText(reader);
                    break;
                case "rangeDynamic":
                    message.rangeDynamic = CoreML.Specification.RangeDynamicLayerParams.decodeText(reader);
                    break;
                case "clip":
                    message.clip = CoreML.Specification.ClipLayerParams.decodeText(reader);
                    break;
                case "ceil":
                    message.ceil = CoreML.Specification.CeilLayerParams.decodeText(reader);
                    break;
                case "floor":
                    message.floor = CoreML.Specification.FloorLayerParams.decodeText(reader);
                    break;
                case "sign":
                    message.sign = CoreML.Specification.SignLayerParams.decodeText(reader);
                    break;
                case "round":
                    message.round = CoreML.Specification.RoundLayerParams.decodeText(reader);
                    break;
                case "exp2":
                    message.exp2 = CoreML.Specification.Exp2LayerParams.decodeText(reader);
                    break;
                case "sin":
                    message.sin = CoreML.Specification.SinLayerParams.decodeText(reader);
                    break;
                case "cos":
                    message.cos = CoreML.Specification.CosLayerParams.decodeText(reader);
                    break;
                case "tan":
                    message.tan = CoreML.Specification.TanLayerParams.decodeText(reader);
                    break;
                case "asin":
                    message.asin = CoreML.Specification.AsinLayerParams.decodeText(reader);
                    break;
                case "acos":
                    message.acos = CoreML.Specification.AcosLayerParams.decodeText(reader);
                    break;
                case "atan":
                    message.atan = CoreML.Specification.AtanLayerParams.decodeText(reader);
                    break;
                case "sinh":
                    message.sinh = CoreML.Specification.SinhLayerParams.decodeText(reader);
                    break;
                case "cosh":
                    message.cosh = CoreML.Specification.CoshLayerParams.decodeText(reader);
                    break;
                case "tanh":
                    message.tanh = CoreML.Specification.TanhLayerParams.decodeText(reader);
                    break;
                case "asinh":
                    message.asinh = CoreML.Specification.AsinhLayerParams.decodeText(reader);
                    break;
                case "acosh":
                    message.acosh = CoreML.Specification.AcoshLayerParams.decodeText(reader);
                    break;
                case "atanh":
                    message.atanh = CoreML.Specification.AtanhLayerParams.decodeText(reader);
                    break;
                case "erf":
                    message.erf = CoreML.Specification.ErfLayerParams.decodeText(reader);
                    break;
                case "gelu":
                    message.gelu = CoreML.Specification.GeluLayerParams.decodeText(reader);
                    break;
                case "equal":
                    message.equal = CoreML.Specification.EqualLayerParams.decodeText(reader);
                    break;
                case "notEqual":
                    message.notEqual = CoreML.Specification.NotEqualLayerParams.decodeText(reader);
                    break;
                case "lessThan":
                    message.lessThan = CoreML.Specification.LessThanLayerParams.decodeText(reader);
                    break;
                case "lessEqual":
                    message.lessEqual = CoreML.Specification.LessEqualLayerParams.decodeText(reader);
                    break;
                case "greaterThan":
                    message.greaterThan = CoreML.Specification.GreaterThanLayerParams.decodeText(reader);
                    break;
                case "greaterEqual":
                    message.greaterEqual = CoreML.Specification.GreaterEqualLayerParams.decodeText(reader);
                    break;
                case "logicalOr":
                    message.logicalOr = CoreML.Specification.LogicalOrLayerParams.decodeText(reader);
                    break;
                case "logicalXor":
                    message.logicalXor = CoreML.Specification.LogicalXorLayerParams.decodeText(reader);
                    break;
                case "logicalNot":
                    message.logicalNot = CoreML.Specification.LogicalNotLayerParams.decodeText(reader);
                    break;
                case "logicalAnd":
                    message.logicalAnd = CoreML.Specification.LogicalAndLayerParams.decodeText(reader);
                    break;
                case "modBroadcastable":
                    message.modBroadcastable = CoreML.Specification.ModBroadcastableLayerParams.decodeText(reader);
                    break;
                case "minBroadcastable":
                    message.minBroadcastable = CoreML.Specification.MinBroadcastableLayerParams.decodeText(reader);
                    break;
                case "maxBroadcastable":
                    message.maxBroadcastable = CoreML.Specification.MaxBroadcastableLayerParams.decodeText(reader);
                    break;
                case "addBroadcastable":
                    message.addBroadcastable = CoreML.Specification.AddBroadcastableLayerParams.decodeText(reader);
                    break;
                case "powBroadcastable":
                    message.powBroadcastable = CoreML.Specification.PowBroadcastableLayerParams.decodeText(reader);
                    break;
                case "divideBroadcastable":
                    message.divideBroadcastable = CoreML.Specification.DivideBroadcastableLayerParams.decodeText(reader);
                    break;
                case "floorDivBroadcastable":
                    message.floorDivBroadcastable = CoreML.Specification.FloorDivBroadcastableLayerParams.decodeText(reader);
                    break;
                case "multiplyBroadcastable":
                    message.multiplyBroadcastable = CoreML.Specification.MultiplyBroadcastableLayerParams.decodeText(reader);
                    break;
                case "subtractBroadcastable":
                    message.subtractBroadcastable = CoreML.Specification.SubtractBroadcastableLayerParams.decodeText(reader);
                    break;
                case "tile":
                    message.tile = CoreML.Specification.TileLayerParams.decodeText(reader);
                    break;
                case "stack":
                    message.stack = CoreML.Specification.StackLayerParams.decodeText(reader);
                    break;
                case "gather":
                    message.gather = CoreML.Specification.GatherLayerParams.decodeText(reader);
                    break;
                case "scatter":
                    message.scatter = CoreML.Specification.ScatterLayerParams.decodeText(reader);
                    break;
                case "gatherND":
                    message.gatherND = CoreML.Specification.GatherNDLayerParams.decodeText(reader);
                    break;
                case "scatterND":
                    message.scatterND = CoreML.Specification.ScatterNDLayerParams.decodeText(reader);
                    break;
                case "softmaxND":
                    message.softmaxND = CoreML.Specification.SoftmaxNDLayerParams.decodeText(reader);
                    break;
                case "gatherAlongAxis":
                    message.gatherAlongAxis = CoreML.Specification.GatherAlongAxisLayerParams.decodeText(reader);
                    break;
                case "scatterAlongAxis":
                    message.scatterAlongAxis = CoreML.Specification.ScatterAlongAxisLayerParams.decodeText(reader);
                    break;
                case "reverse":
                    message.reverse = CoreML.Specification.ReverseLayerParams.decodeText(reader);
                    break;
                case "reverseSeq":
                    message.reverseSeq = CoreML.Specification.ReverseSeqLayerParams.decodeText(reader);
                    break;
                case "splitND":
                    message.splitND = CoreML.Specification.SplitNDLayerParams.decodeText(reader);
                    break;
                case "concatND":
                    message.concatND = CoreML.Specification.ConcatNDLayerParams.decodeText(reader);
                    break;
                case "transpose":
                    message.transpose = CoreML.Specification.TransposeLayerParams.decodeText(reader);
                    break;
                case "sliceStatic":
                    message.sliceStatic = CoreML.Specification.SliceStaticLayerParams.decodeText(reader);
                    break;
                case "sliceDynamic":
                    message.sliceDynamic = CoreML.Specification.SliceDynamicLayerParams.decodeText(reader);
                    break;
                case "slidingWindows":
                    message.slidingWindows = CoreML.Specification.SlidingWindowsLayerParams.decodeText(reader);
                    break;
                case "topK":
                    message.topK = CoreML.Specification.TopKLayerParams.decodeText(reader);
                    break;
                case "argMin":
                    message.argMin = CoreML.Specification.ArgMinLayerParams.decodeText(reader);
                    break;
                case "argMax":
                    message.argMax = CoreML.Specification.ArgMaxLayerParams.decodeText(reader);
                    break;
                case "embeddingND":
                    message.embeddingND = CoreML.Specification.EmbeddingNDLayerParams.decodeText(reader);
                    break;
                case "batchedMatmul":
                    message.batchedMatmul = CoreML.Specification.BatchedMatMulLayerParams.decodeText(reader);
                    break;
                case "getShape":
                    message.getShape = CoreML.Specification.GetShapeLayerParams.decodeText(reader);
                    break;
                case "loadConstantND":
                    message.loadConstantND = CoreML.Specification.LoadConstantNDLayerParams.decodeText(reader);
                    break;
                case "fillLike":
                    message.fillLike = CoreML.Specification.FillLikeLayerParams.decodeText(reader);
                    break;
                case "fillStatic":
                    message.fillStatic = CoreML.Specification.FillStaticLayerParams.decodeText(reader);
                    break;
                case "fillDynamic":
                    message.fillDynamic = CoreML.Specification.FillDynamicLayerParams.decodeText(reader);
                    break;
                case "broadcastToLike":
                    message.broadcastToLike = CoreML.Specification.BroadcastToLikeLayerParams.decodeText(reader);
                    break;
                case "broadcastToStatic":
                    message.broadcastToStatic = CoreML.Specification.BroadcastToStaticLayerParams.decodeText(reader);
                    break;
                case "broadcastToDynamic":
                    message.broadcastToDynamic = CoreML.Specification.BroadcastToDynamicLayerParams.decodeText(reader);
                    break;
                case "squeeze":
                    message.squeeze = CoreML.Specification.SqueezeLayerParams.decodeText(reader);
                    break;
                case "expandDims":
                    message.expandDims = CoreML.Specification.ExpandDimsLayerParams.decodeText(reader);
                    break;
                case "flattenTo2D":
                    message.flattenTo2D = CoreML.Specification.FlattenTo2DLayerParams.decodeText(reader);
                    break;
                case "reshapeLike":
                    message.reshapeLike = CoreML.Specification.ReshapeLikeLayerParams.decodeText(reader);
                    break;
                case "reshapeStatic":
                    message.reshapeStatic = CoreML.Specification.ReshapeStaticLayerParams.decodeText(reader);
                    break;
                case "reshapeDynamic":
                    message.reshapeDynamic = CoreML.Specification.ReshapeDynamicLayerParams.decodeText(reader);
                    break;
                case "rankPreservingReshape":
                    message.rankPreservingReshape = CoreML.Specification.RankPreservingReshapeLayerParams.decodeText(reader);
                    break;
                case "constantPad":
                    message.constantPad = CoreML.Specification.ConstantPaddingLayerParams.decodeText(reader);
                    break;
                case "randomNormalLike":
                    message.randomNormalLike = CoreML.Specification.RandomNormalLikeLayerParams.decodeText(reader);
                    break;
                case "randomNormalStatic":
                    message.randomNormalStatic = CoreML.Specification.RandomNormalStaticLayerParams.decodeText(reader);
                    break;
                case "randomNormalDynamic":
                    message.randomNormalDynamic = CoreML.Specification.RandomNormalDynamicLayerParams.decodeText(reader);
                    break;
                case "randomUniformLike":
                    message.randomUniformLike = CoreML.Specification.RandomUniformLikeLayerParams.decodeText(reader);
                    break;
                case "randomUniformStatic":
                    message.randomUniformStatic = CoreML.Specification.RandomUniformStaticLayerParams.decodeText(reader);
                    break;
                case "randomUniformDynamic":
                    message.randomUniformDynamic = CoreML.Specification.RandomUniformDynamicLayerParams.decodeText(reader);
                    break;
                case "randomBernoulliLike":
                    message.randomBernoulliLike = CoreML.Specification.RandomBernoulliLikeLayerParams.decodeText(reader);
                    break;
                case "randomBernoulliStatic":
                    message.randomBernoulliStatic = CoreML.Specification.RandomBernoulliStaticLayerParams.decodeText(reader);
                    break;
                case "randomBernoulliDynamic":
                    message.randomBernoulliDynamic = CoreML.Specification.RandomBernoulliDynamicLayerParams.decodeText(reader);
                    break;
                case "categoricalDistribution":
                    message.categoricalDistribution = CoreML.Specification.CategoricalDistributionLayerParams.decodeText(reader);
                    break;
                case "reduceL1":
                    message.reduceL1 = CoreML.Specification.ReduceL1LayerParams.decodeText(reader);
                    break;
                case "reduceL2":
                    message.reduceL2 = CoreML.Specification.ReduceL2LayerParams.decodeText(reader);
                    break;
                case "reduceMax":
                    message.reduceMax = CoreML.Specification.ReduceMaxLayerParams.decodeText(reader);
                    break;
                case "reduceMin":
                    message.reduceMin = CoreML.Specification.ReduceMinLayerParams.decodeText(reader);
                    break;
                case "reduceSum":
                    message.reduceSum = CoreML.Specification.ReduceSumLayerParams.decodeText(reader);
                    break;
                case "reduceProd":
                    message.reduceProd = CoreML.Specification.ReduceProdLayerParams.decodeText(reader);
                    break;
                case "reduceMean":
                    message.reduceMean = CoreML.Specification.ReduceMeanLayerParams.decodeText(reader);
                    break;
                case "reduceLogSum":
                    message.reduceLogSum = CoreML.Specification.ReduceLogSumLayerParams.decodeText(reader);
                    break;
                case "reduceSumSquare":
                    message.reduceSumSquare = CoreML.Specification.ReduceSumSquareLayerParams.decodeText(reader);
                    break;
                case "reduceLogSumExp":
                    message.reduceLogSumExp = CoreML.Specification.ReduceLogSumExpLayerParams.decodeText(reader);
                    break;
                case "whereNonZero":
                    message.whereNonZero = CoreML.Specification.WhereNonZeroLayerParams.decodeText(reader);
                    break;
                case "matrixBandPart":
                    message.matrixBandPart = CoreML.Specification.MatrixBandPartLayerParams.decodeText(reader);
                    break;
                case "lowerTriangular":
                    message.lowerTriangular = CoreML.Specification.LowerTriangularLayerParams.decodeText(reader);
                    break;
                case "upperTriangular":
                    message.upperTriangular = CoreML.Specification.UpperTriangularLayerParams.decodeText(reader);
                    break;
                case "whereBroadcastable":
                    message.whereBroadcastable = CoreML.Specification.WhereBroadcastableLayerParams.decodeText(reader);
                    break;
                case "layerNormalization":
                    message.layerNormalization = CoreML.Specification.LayerNormalizationLayerParams.decodeText(reader);
                    break;
                case "NonMaximumSuppression":
                    message.NonMaximumSuppression = CoreML.Specification.NonMaximumSuppressionLayerParams.decodeText(reader);
                    break;
                case "oneHot":
                    message.oneHot = CoreML.Specification.OneHotLayerParams.decodeText(reader);
                    break;
                case "cumSum":
                    message.cumSum = CoreML.Specification.CumSumLayerParams.decodeText(reader);
                    break;
                case "clampedReLU":
                    message.clampedReLU = CoreML.Specification.ClampedReLULayerParams.decodeText(reader);
                    break;
                case "argSort":
                    message.argSort = CoreML.Specification.ArgSortLayerParams.decodeText(reader);
                    break;
                case "pooling3d":
                    message.pooling3d = CoreML.Specification.Pooling3DLayerParams.decodeText(reader);
                    break;
                case "globalPooling3d":
                    message.globalPooling3d = CoreML.Specification.GlobalPooling3DLayerParams.decodeText(reader);
                    break;
                case "sliceBySize":
                    message.sliceBySize = CoreML.Specification.SliceBySizeLayerParams.decodeText(reader);
                    break;
                case "convolution3d":
                    message.convolution3d = CoreML.Specification.Convolution3DLayerParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetworkLayer.prototype.name = "";
CoreML.Specification.NeuralNetworkLayer.prototype.isUpdatable = false;

CoreML.Specification.BranchLayerParams = class BranchLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.BranchLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ifBranch = CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.elseBranch = CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BranchLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ifBranch":
                    message.ifBranch = CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                case "elseBranch":
                    message.elseBranch = CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BranchLayerParams.prototype.ifBranch = null;
CoreML.Specification.BranchLayerParams.prototype.elseBranch = null;

CoreML.Specification.LoopLayerParams = class LoopLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LoopLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.maxLoopIterations = reader.uint64();
                    break;
                case 2:
                    message.conditionVar = reader.string();
                    break;
                case 3:
                    message.conditionNetwork = CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bodyNetwork = CoreML.Specification.NeuralNetwork.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LoopLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "maxLoopIterations":
                    message.maxLoopIterations = reader.uint64();
                    break;
                case "conditionVar":
                    message.conditionVar = reader.string();
                    break;
                case "conditionNetwork":
                    message.conditionNetwork = CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                case "bodyNetwork":
                    message.bodyNetwork = CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LoopLayerParams.prototype.maxLoopIterations = 0n;
CoreML.Specification.LoopLayerParams.prototype.conditionVar = "";
CoreML.Specification.LoopLayerParams.prototype.conditionNetwork = null;
CoreML.Specification.LoopLayerParams.prototype.bodyNetwork = null;

CoreML.Specification.LoopBreakLayerParams = class LoopBreakLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LoopBreakLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LoopBreakLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LoopContinueLayerParams = class LoopContinueLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LoopContinueLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LoopContinueLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CopyLayerParams = class CopyLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.CopyLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CopyLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GreaterThanLayerParams = class GreaterThanLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GreaterThanLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GreaterThanLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GreaterThanLayerParams.prototype.alpha = 0;

CoreML.Specification.GreaterEqualLayerParams = class GreaterEqualLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GreaterEqualLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GreaterEqualLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GreaterEqualLayerParams.prototype.alpha = 0;

CoreML.Specification.LessThanLayerParams = class LessThanLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LessThanLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LessThanLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LessThanLayerParams.prototype.alpha = 0;

CoreML.Specification.LessEqualLayerParams = class LessEqualLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LessEqualLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LessEqualLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LessEqualLayerParams.prototype.alpha = 0;

CoreML.Specification.EqualLayerParams = class EqualLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.EqualLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.EqualLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.EqualLayerParams.prototype.alpha = 0;

CoreML.Specification.NotEqualLayerParams = class NotEqualLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.NotEqualLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NotEqualLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NotEqualLayerParams.prototype.alpha = 0;

CoreML.Specification.LogicalAndLayerParams = class LogicalAndLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LogicalAndLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LogicalAndLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LogicalOrLayerParams = class LogicalOrLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LogicalOrLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LogicalOrLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LogicalXorLayerParams = class LogicalXorLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LogicalXorLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LogicalXorLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LogicalNotLayerParams = class LogicalNotLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LogicalNotLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LogicalNotLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BorderAmounts = class BorderAmounts {

    constructor() {
        this.borderAmounts = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.BorderAmounts();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.borderAmounts.push(CoreML.Specification.BorderAmounts.EdgeSizes.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BorderAmounts();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "borderAmounts":
                    message.borderAmounts.push(CoreML.Specification.BorderAmounts.EdgeSizes.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BorderAmounts.EdgeSizes = class EdgeSizes {

    static decode(reader, length) {
        const message = new CoreML.Specification.BorderAmounts.EdgeSizes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BorderAmounts.EdgeSizes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "startEdgeSize":
                    message.startEdgeSize = reader.uint64();
                    break;
                case "endEdgeSize":
                    message.endEdgeSize = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BorderAmounts.EdgeSizes.prototype.startEdgeSize = 0n;
CoreML.Specification.BorderAmounts.EdgeSizes.prototype.endEdgeSize = 0n;

CoreML.Specification.ValidPadding = class ValidPadding {

    static decode(reader, length) {
        const message = new CoreML.Specification.ValidPadding();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.paddingAmounts = CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ValidPadding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "paddingAmounts":
                    message.paddingAmounts = CoreML.Specification.BorderAmounts.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ValidPadding.prototype.paddingAmounts = null;

CoreML.Specification.SamePadding = class SamePadding {

    static decode(reader, length) {
        const message = new CoreML.Specification.SamePadding();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SamePadding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "asymmetryMode":
                    message.asymmetryMode = reader.enum(CoreML.Specification.SamePadding.SamePaddingMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SamePadding.prototype.asymmetryMode = 0;

CoreML.Specification.SamePadding.SamePaddingMode = {
    "BOTTOM_RIGHT_HEAVY": 0,
    "TOP_LEFT_HEAVY": 1
};

CoreML.Specification.SamplingMode = class SamplingMode {

    static decode(reader, length) {
        const message = new CoreML.Specification.SamplingMode();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SamplingMode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "samplingMethod":
                    message.samplingMethod = reader.enum(CoreML.Specification.SamplingMode.Method);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SamplingMode.prototype.samplingMethod = 0;

CoreML.Specification.SamplingMode.Method = {
    "STRICT_ALIGN_ENDPOINTS_MODE": 0,
    "ALIGN_ENDPOINTS_MODE": 1,
    "UPSAMPLE_MODE": 2,
    "ROI_ALIGN_MODE": 3
};

CoreML.Specification.BoxCoordinatesMode = class BoxCoordinatesMode {

    static decode(reader, length) {
        const message = new CoreML.Specification.BoxCoordinatesMode();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BoxCoordinatesMode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "boxMode":
                    message.boxMode = reader.enum(CoreML.Specification.BoxCoordinatesMode.Coordinates);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BoxCoordinatesMode.prototype.boxMode = 0;

CoreML.Specification.BoxCoordinatesMode.Coordinates = {
    "CORNERS_HEIGHT_FIRST": 0,
    "CORNERS_WIDTH_FIRST": 1,
    "CENTER_SIZE_HEIGHT_FIRST": 2,
    "CENTER_SIZE_WIDTH_FIRST": 3
};

CoreML.Specification.WeightParams = class WeightParams {

    constructor() {
        this.floatValue = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.WeightParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.quantization = CoreML.Specification.QuantizationParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.WeightParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "floatValue":
                    reader.array(message.floatValue, () => reader.float());
                    break;
                case "float16Value":
                    message.float16Value = reader.bytes();
                    break;
                case "rawValue":
                    message.rawValue = reader.bytes();
                    break;
                case "int8RawValue":
                    message.int8RawValue = reader.bytes();
                    break;
                case "quantization":
                    message.quantization = CoreML.Specification.QuantizationParams.decodeText(reader);
                    break;
                case "isUpdatable":
                    message.isUpdatable = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.WeightParams.prototype.float16Value = new Uint8Array([]);
CoreML.Specification.WeightParams.prototype.rawValue = new Uint8Array([]);
CoreML.Specification.WeightParams.prototype.int8RawValue = new Uint8Array([]);
CoreML.Specification.WeightParams.prototype.quantization = null;
CoreML.Specification.WeightParams.prototype.isUpdatable = false;

CoreML.Specification.QuantizationParams = class QuantizationParams {

    get QuantizationType() {
        CoreML.Specification.QuantizationParams.QuantizationTypeSet = CoreML.Specification.QuantizationParams.QuantizationTypeSet || new Set(["linearQuantization", "lookupTableQuantization"]);
        return Object.keys(this).find((key) => CoreML.Specification.QuantizationParams.QuantizationTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.QuantizationParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.numberOfBits = reader.uint64();
                    break;
                case 101:
                    message.linearQuantization = CoreML.Specification.LinearQuantizationParams.decode(reader, reader.uint32());
                    break;
                case 102:
                    message.lookupTableQuantization = CoreML.Specification.LookUpTableQuantizationParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.QuantizationParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numberOfBits":
                    message.numberOfBits = reader.uint64();
                    break;
                case "linearQuantization":
                    message.linearQuantization = CoreML.Specification.LinearQuantizationParams.decodeText(reader);
                    break;
                case "lookupTableQuantization":
                    message.lookupTableQuantization = CoreML.Specification.LookUpTableQuantizationParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.QuantizationParams.prototype.numberOfBits = 0n;

CoreML.Specification.LinearQuantizationParams = class LinearQuantizationParams {

    constructor() {
        this.scale = [];
        this.bias = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LinearQuantizationParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LinearQuantizationParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "scale":
                    reader.array(message.scale, () => reader.float());
                    break;
                case "bias":
                    reader.array(message.bias, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LookUpTableQuantizationParams = class LookUpTableQuantizationParams {

    constructor() {
        this.floatValue = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LookUpTableQuantizationParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LookUpTableQuantizationParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "floatValue":
                    reader.array(message.floatValue, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConvolutionLayerParams = class ConvolutionLayerParams {

    constructor() {
        this.kernelSize = [];
        this.stride = [];
        this.dilationFactor = [];
        this.outputShape = [];
    }

    get ConvolutionPaddingType() {
        CoreML.Specification.ConvolutionLayerParams.ConvolutionPaddingTypeSet = CoreML.Specification.ConvolutionLayerParams.ConvolutionPaddingTypeSet || new Set(["valid", "same"]);
        return Object.keys(this).find((key) => CoreML.Specification.ConvolutionLayerParams.ConvolutionPaddingTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ConvolutionLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.valid = CoreML.Specification.ValidPadding.decode(reader, reader.uint32());
                    break;
                case 51:
                    message.same = CoreML.Specification.SamePadding.decode(reader, reader.uint32());
                    break;
                case 60:
                    message.isDeconvolution = reader.bool();
                    break;
                case 70:
                    message.hasBias = reader.bool();
                    break;
                case 90:
                    message.weights = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 91:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ConvolutionLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "outputChannels":
                    message.outputChannels = reader.uint64();
                    break;
                case "kernelChannels":
                    message.kernelChannels = reader.uint64();
                    break;
                case "nGroups":
                    message.nGroups = reader.uint64();
                    break;
                case "kernelSize":
                    reader.array(message.kernelSize, () => reader.uint64());
                    break;
                case "stride":
                    reader.array(message.stride, () => reader.uint64());
                    break;
                case "dilationFactor":
                    reader.array(message.dilationFactor, () => reader.uint64());
                    break;
                case "valid":
                    message.valid = CoreML.Specification.ValidPadding.decodeText(reader);
                    break;
                case "same":
                    message.same = CoreML.Specification.SamePadding.decodeText(reader);
                    break;
                case "isDeconvolution":
                    message.isDeconvolution = reader.bool();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputShape":
                    reader.array(message.outputShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConvolutionLayerParams.prototype.outputChannels = 0n;
CoreML.Specification.ConvolutionLayerParams.prototype.kernelChannels = 0n;
CoreML.Specification.ConvolutionLayerParams.prototype.nGroups = 0n;
CoreML.Specification.ConvolutionLayerParams.prototype.isDeconvolution = false;
CoreML.Specification.ConvolutionLayerParams.prototype.hasBias = false;
CoreML.Specification.ConvolutionLayerParams.prototype.weights = null;
CoreML.Specification.ConvolutionLayerParams.prototype.bias = null;

CoreML.Specification.Convolution3DLayerParams = class Convolution3DLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Convolution3DLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.weights = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 61:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Convolution3DLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "outputChannels":
                    message.outputChannels = reader.int32();
                    break;
                case "inputChannels":
                    message.inputChannels = reader.int32();
                    break;
                case "nGroups":
                    message.nGroups = reader.int32();
                    break;
                case "kernelDepth":
                    message.kernelDepth = reader.int32();
                    break;
                case "kernelHeight":
                    message.kernelHeight = reader.int32();
                    break;
                case "kernelWidth":
                    message.kernelWidth = reader.int32();
                    break;
                case "strideDepth":
                    message.strideDepth = reader.int32();
                    break;
                case "strideHeight":
                    message.strideHeight = reader.int32();
                    break;
                case "strideWidth":
                    message.strideWidth = reader.int32();
                    break;
                case "dilationDepth":
                    message.dilationDepth = reader.int32();
                    break;
                case "dilationHeight":
                    message.dilationHeight = reader.int32();
                    break;
                case "dilationWidth":
                    message.dilationWidth = reader.int32();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "paddingType":
                    message.paddingType = reader.enum(CoreML.Specification.Convolution3DLayerParams.PaddingType);
                    break;
                case "customPaddingFront":
                    message.customPaddingFront = reader.int32();
                    break;
                case "customPaddingBack":
                    message.customPaddingBack = reader.int32();
                    break;
                case "customPaddingTop":
                    message.customPaddingTop = reader.int32();
                    break;
                case "customPaddingBottom":
                    message.customPaddingBottom = reader.int32();
                    break;
                case "customPaddingLeft":
                    message.customPaddingLeft = reader.int32();
                    break;
                case "customPaddingRight":
                    message.customPaddingRight = reader.int32();
                    break;
                case "isDeconvolution":
                    message.isDeconvolution = reader.bool();
                    break;
                case "outputShape":
                    reader.array(message.outputShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Convolution3DLayerParams.prototype.outputChannels = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.inputChannels = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.nGroups = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.kernelDepth = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.kernelHeight = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.kernelWidth = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.strideDepth = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.strideHeight = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.strideWidth = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.dilationDepth = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.dilationHeight = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.dilationWidth = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.hasBias = false;
CoreML.Specification.Convolution3DLayerParams.prototype.weights = null;
CoreML.Specification.Convolution3DLayerParams.prototype.bias = null;
CoreML.Specification.Convolution3DLayerParams.prototype.paddingType = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingFront = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingBack = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingTop = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingBottom = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingLeft = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.customPaddingRight = 0;
CoreML.Specification.Convolution3DLayerParams.prototype.isDeconvolution = false;

CoreML.Specification.Convolution3DLayerParams.PaddingType = {
    "CUSTOM": 0,
    "VALID": 1,
    "SAME": 2
};

CoreML.Specification.InnerProductLayerParams = class InnerProductLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.InnerProductLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.weights = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.InnerProductLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputChannels":
                    message.inputChannels = reader.uint64();
                    break;
                case "outputChannels":
                    message.outputChannels = reader.uint64();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "int8DynamicQuantize":
                    message.int8DynamicQuantize = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.InnerProductLayerParams.prototype.inputChannels = 0n;
CoreML.Specification.InnerProductLayerParams.prototype.outputChannels = 0n;
CoreML.Specification.InnerProductLayerParams.prototype.hasBias = false;
CoreML.Specification.InnerProductLayerParams.prototype.weights = null;
CoreML.Specification.InnerProductLayerParams.prototype.bias = null;
CoreML.Specification.InnerProductLayerParams.prototype.int8DynamicQuantize = false;

CoreML.Specification.EmbeddingLayerParams = class EmbeddingLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.EmbeddingLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.weights = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.EmbeddingLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputDim":
                    message.inputDim = reader.uint64();
                    break;
                case "outputChannels":
                    message.outputChannels = reader.uint64();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.EmbeddingLayerParams.prototype.inputDim = 0n;
CoreML.Specification.EmbeddingLayerParams.prototype.outputChannels = 0n;
CoreML.Specification.EmbeddingLayerParams.prototype.hasBias = false;
CoreML.Specification.EmbeddingLayerParams.prototype.weights = null;
CoreML.Specification.EmbeddingLayerParams.prototype.bias = null;

CoreML.Specification.EmbeddingNDLayerParams = class EmbeddingNDLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.EmbeddingNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.weights = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.EmbeddingNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vocabSize":
                    message.vocabSize = reader.uint64();
                    break;
                case "embeddingSize":
                    message.embeddingSize = reader.uint64();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.EmbeddingNDLayerParams.prototype.vocabSize = 0n;
CoreML.Specification.EmbeddingNDLayerParams.prototype.embeddingSize = 0n;
CoreML.Specification.EmbeddingNDLayerParams.prototype.hasBias = false;
CoreML.Specification.EmbeddingNDLayerParams.prototype.weights = null;
CoreML.Specification.EmbeddingNDLayerParams.prototype.bias = null;

CoreML.Specification.BatchnormLayerParams = class BatchnormLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.BatchnormLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.gamma = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.beta = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 17:
                    message.mean = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 18:
                    message.variance = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BatchnormLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "channels":
                    message.channels = reader.uint64();
                    break;
                case "computeMeanVar":
                    message.computeMeanVar = reader.bool();
                    break;
                case "instanceNormalization":
                    message.instanceNormalization = reader.bool();
                    break;
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                case "gamma":
                    message.gamma = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "beta":
                    message.beta = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "mean":
                    message.mean = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "variance":
                    message.variance = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BatchnormLayerParams.prototype.channels = 0n;
CoreML.Specification.BatchnormLayerParams.prototype.computeMeanVar = false;
CoreML.Specification.BatchnormLayerParams.prototype.instanceNormalization = false;
CoreML.Specification.BatchnormLayerParams.prototype.epsilon = 0;
CoreML.Specification.BatchnormLayerParams.prototype.gamma = null;
CoreML.Specification.BatchnormLayerParams.prototype.beta = null;
CoreML.Specification.BatchnormLayerParams.prototype.mean = null;
CoreML.Specification.BatchnormLayerParams.prototype.variance = null;

CoreML.Specification.PoolingLayerParams = class PoolingLayerParams {

    constructor() {
        this.kernelSize = [];
        this.stride = [];
    }

    get PoolingPaddingType() {
        CoreML.Specification.PoolingLayerParams.PoolingPaddingTypeSet = CoreML.Specification.PoolingLayerParams.PoolingPaddingTypeSet || new Set(["valid", "same", "includeLastPixel"]);
        return Object.keys(this).find((key) => CoreML.Specification.PoolingLayerParams.PoolingPaddingTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.PoolingLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.valid = CoreML.Specification.ValidPadding.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.same = CoreML.Specification.SamePadding.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.includeLastPixel = CoreML.Specification.PoolingLayerParams.ValidCompletePadding.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.PoolingLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(CoreML.Specification.PoolingLayerParams.PoolingType);
                    break;
                case "kernelSize":
                    reader.array(message.kernelSize, () => reader.uint64());
                    break;
                case "stride":
                    reader.array(message.stride, () => reader.uint64());
                    break;
                case "valid":
                    message.valid = CoreML.Specification.ValidPadding.decodeText(reader);
                    break;
                case "same":
                    message.same = CoreML.Specification.SamePadding.decodeText(reader);
                    break;
                case "includeLastPixel":
                    message.includeLastPixel = CoreML.Specification.PoolingLayerParams.ValidCompletePadding.decodeText(reader);
                    break;
                case "avgPoolExcludePadding":
                    message.avgPoolExcludePadding = reader.bool();
                    break;
                case "globalPooling":
                    message.globalPooling = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PoolingLayerParams.prototype.type = 0;
CoreML.Specification.PoolingLayerParams.prototype.avgPoolExcludePadding = false;
CoreML.Specification.PoolingLayerParams.prototype.globalPooling = false;

CoreML.Specification.PoolingLayerParams.PoolingType = {
    "MAX": 0,
    "AVERAGE": 1,
    "L2": 2
};

CoreML.Specification.PoolingLayerParams.ValidCompletePadding = class ValidCompletePadding {

    constructor() {
        this.paddingAmounts = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.PoolingLayerParams.ValidCompletePadding();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.PoolingLayerParams.ValidCompletePadding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "paddingAmounts":
                    reader.array(message.paddingAmounts, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Pooling3DLayerParams = class Pooling3DLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.Pooling3DLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Pooling3DLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(CoreML.Specification.Pooling3DLayerParams.PoolingType3D);
                    break;
                case "kernelDepth":
                    message.kernelDepth = reader.int32();
                    break;
                case "kernelHeight":
                    message.kernelHeight = reader.int32();
                    break;
                case "kernelWidth":
                    message.kernelWidth = reader.int32();
                    break;
                case "strideDepth":
                    message.strideDepth = reader.int32();
                    break;
                case "strideHeight":
                    message.strideHeight = reader.int32();
                    break;
                case "strideWidth":
                    message.strideWidth = reader.int32();
                    break;
                case "paddingType":
                    message.paddingType = reader.enum(CoreML.Specification.Pooling3DLayerParams.Pooling3DPaddingType);
                    break;
                case "customPaddingFront":
                    message.customPaddingFront = reader.int32();
                    break;
                case "customPaddingBack":
                    message.customPaddingBack = reader.int32();
                    break;
                case "customPaddingTop":
                    message.customPaddingTop = reader.int32();
                    break;
                case "customPaddingBottom":
                    message.customPaddingBottom = reader.int32();
                    break;
                case "customPaddingLeft":
                    message.customPaddingLeft = reader.int32();
                    break;
                case "customPaddingRight":
                    message.customPaddingRight = reader.int32();
                    break;
                case "countExcludePadding":
                    message.countExcludePadding = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Pooling3DLayerParams.prototype.type = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.kernelDepth = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.kernelHeight = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.kernelWidth = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.strideDepth = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.strideHeight = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.strideWidth = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.paddingType = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingFront = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingBack = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingTop = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingBottom = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingLeft = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.customPaddingRight = 0;
CoreML.Specification.Pooling3DLayerParams.prototype.countExcludePadding = false;

CoreML.Specification.Pooling3DLayerParams.PoolingType3D = {
    "MAX": 0,
    "AVERAGE": 1
};

CoreML.Specification.Pooling3DLayerParams.Pooling3DPaddingType = {
    "CUSTOM": 0,
    "VALID": 1,
    "SAME": 2
};

CoreML.Specification.GlobalPooling3DLayerParams = class GlobalPooling3DLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GlobalPooling3DLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GlobalPooling3DLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(CoreML.Specification.GlobalPooling3DLayerParams.GlobalPoolingType3D);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GlobalPooling3DLayerParams.prototype.type = 0;

CoreML.Specification.GlobalPooling3DLayerParams.GlobalPoolingType3D = {
    "MAX": 0,
    "AVERAGE": 1
};

CoreML.Specification.PaddingLayerParams = class PaddingLayerParams {

    get PaddingType() {
        CoreML.Specification.PaddingLayerParams.PaddingTypeSet = CoreML.Specification.PaddingLayerParams.PaddingTypeSet || new Set(["constant", "reflection", "replication"]);
        return Object.keys(this).find((key) => CoreML.Specification.PaddingLayerParams.PaddingTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.PaddingLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.constant = CoreML.Specification.PaddingLayerParams.PaddingConstant.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.reflection = CoreML.Specification.PaddingLayerParams.PaddingReflection.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.replication = CoreML.Specification.PaddingLayerParams.PaddingReplication.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.paddingAmounts = CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PaddingLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "constant":
                    message.constant = CoreML.Specification.PaddingLayerParams.PaddingConstant.decodeText(reader);
                    break;
                case "reflection":
                    message.reflection = CoreML.Specification.PaddingLayerParams.PaddingReflection.decodeText(reader);
                    break;
                case "replication":
                    message.replication = CoreML.Specification.PaddingLayerParams.PaddingReplication.decodeText(reader);
                    break;
                case "paddingAmounts":
                    message.paddingAmounts = CoreML.Specification.BorderAmounts.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PaddingLayerParams.prototype.paddingAmounts = null;

CoreML.Specification.PaddingLayerParams.PaddingConstant = class PaddingConstant {

    static decode(reader, length) {
        const message = new CoreML.Specification.PaddingLayerParams.PaddingConstant();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.PaddingLayerParams.PaddingConstant();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PaddingLayerParams.PaddingConstant.prototype.value = 0;

CoreML.Specification.PaddingLayerParams.PaddingReflection = class PaddingReflection {

    static decode(reader, length) {
        const message = new CoreML.Specification.PaddingLayerParams.PaddingReflection();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PaddingLayerParams.PaddingReflection();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PaddingLayerParams.PaddingReplication = class PaddingReplication {

    static decode(reader, length) {
        const message = new CoreML.Specification.PaddingLayerParams.PaddingReplication();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PaddingLayerParams.PaddingReplication();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConcatLayerParams = class ConcatLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ConcatLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ConcatLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sequenceConcat":
                    message.sequenceConcat = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConcatLayerParams.prototype.sequenceConcat = false;

CoreML.Specification.LRNLayerParams = class LRNLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LRNLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LRNLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                case "localSize":
                    message.localSize = reader.uint64();
                    break;
                case "k":
                    message.k = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LRNLayerParams.prototype.alpha = 0;
CoreML.Specification.LRNLayerParams.prototype.beta = 0;
CoreML.Specification.LRNLayerParams.prototype.localSize = 0n;
CoreML.Specification.LRNLayerParams.prototype.k = 0;

CoreML.Specification.SoftmaxLayerParams = class SoftmaxLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SoftmaxLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SoftmaxLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SplitLayerParams = class SplitLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SplitLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SplitLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nOutputs":
                    message.nOutputs = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SplitLayerParams.prototype.nOutputs = 0n;

CoreML.Specification.AddLayerParams = class AddLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AddLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.AddLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AddLayerParams.prototype.alpha = 0;

CoreML.Specification.MultiplyLayerParams = class MultiplyLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MultiplyLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.MultiplyLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MultiplyLayerParams.prototype.alpha = 0;

CoreML.Specification.UnaryFunctionLayerParams = class UnaryFunctionLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.UnaryFunctionLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.UnaryFunctionLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(CoreML.Specification.UnaryFunctionLayerParams.Operation);
                    break;
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                case "shift":
                    message.shift = reader.float();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.UnaryFunctionLayerParams.prototype.type = 0;
CoreML.Specification.UnaryFunctionLayerParams.prototype.alpha = 0;
CoreML.Specification.UnaryFunctionLayerParams.prototype.epsilon = 0;
CoreML.Specification.UnaryFunctionLayerParams.prototype.shift = 0;
CoreML.Specification.UnaryFunctionLayerParams.prototype.scale = 0;

CoreML.Specification.UnaryFunctionLayerParams.Operation = {
    "SQRT": 0,
    "RSQRT": 1,
    "INVERSE": 2,
    "POWER": 3,
    "EXP": 4,
    "LOG": 5,
    "ABS": 6,
    "THRESHOLD": 7
};

CoreML.Specification.UpsampleLayerParams = class UpsampleLayerParams {

    constructor() {
        this.scalingFactor = [];
        this.fractionalScalingFactor = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.UpsampleLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.UpsampleLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "scalingFactor":
                    reader.array(message.scalingFactor, () => reader.uint64());
                    break;
                case "fractionalScalingFactor":
                    reader.array(message.fractionalScalingFactor, () => reader.float());
                    break;
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.UpsampleLayerParams.InterpolationMode);
                    break;
                case "linearUpsampleMode":
                    message.linearUpsampleMode = reader.enum(CoreML.Specification.UpsampleLayerParams.LinearUpsampleMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.UpsampleLayerParams.prototype.mode = 0;
CoreML.Specification.UpsampleLayerParams.prototype.linearUpsampleMode = 0;

CoreML.Specification.UpsampleLayerParams.InterpolationMode = {
    "NN": 0,
    "BILINEAR": 1
};

CoreML.Specification.UpsampleLayerParams.LinearUpsampleMode = {
    "DEFAULT": 0,
    "ALIGN_CORNERS_TRUE": 1,
    "ALIGN_CORNERS_FALSE": 2
};

CoreML.Specification.ResizeBilinearLayerParams = class ResizeBilinearLayerParams {

    constructor() {
        this.targetSize = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ResizeBilinearLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetSize = reader.array(message.targetSize, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.mode = CoreML.Specification.SamplingMode.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ResizeBilinearLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetSize":
                    reader.array(message.targetSize, () => reader.uint64());
                    break;
                case "mode":
                    message.mode = CoreML.Specification.SamplingMode.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ResizeBilinearLayerParams.prototype.mode = null;

CoreML.Specification.CropResizeLayerParams = class CropResizeLayerParams {

    constructor() {
        this.targetSize = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CropResizeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.targetSize = reader.array(message.targetSize, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.normalizedCoordinates = reader.bool();
                    break;
                case 3:
                    message.mode = CoreML.Specification.SamplingMode.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.boxIndicesMode = CoreML.Specification.BoxCoordinatesMode.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CropResizeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetSize":
                    reader.array(message.targetSize, () => reader.uint64());
                    break;
                case "normalizedCoordinates":
                    message.normalizedCoordinates = reader.bool();
                    break;
                case "mode":
                    message.mode = CoreML.Specification.SamplingMode.decodeText(reader);
                    break;
                case "boxIndicesMode":
                    message.boxIndicesMode = CoreML.Specification.BoxCoordinatesMode.decodeText(reader);
                    break;
                case "spatialScale":
                    message.spatialScale = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CropResizeLayerParams.prototype.normalizedCoordinates = false;
CoreML.Specification.CropResizeLayerParams.prototype.mode = null;
CoreML.Specification.CropResizeLayerParams.prototype.boxIndicesMode = null;
CoreML.Specification.CropResizeLayerParams.prototype.spatialScale = 0;

CoreML.Specification.BiasLayerParams = class BiasLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.BiasLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BiasLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.uint64());
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BiasLayerParams.prototype.bias = null;

CoreML.Specification.ScaleLayerParams = class ScaleLayerParams {

    constructor() {
        this.shapeScale = [];
        this.shapeBias = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ScaleLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shapeScale = reader.array(message.shapeScale, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.scale = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.hasBias = reader.bool();
                    break;
                case 4:
                    message.shapeBias = reader.array(message.shapeBias, () => reader.uint64(), tag);
                    break;
                case 5:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ScaleLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shapeScale":
                    reader.array(message.shapeScale, () => reader.uint64());
                    break;
                case "scale":
                    message.scale = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "shapeBias":
                    reader.array(message.shapeBias, () => reader.uint64());
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ScaleLayerParams.prototype.scale = null;
CoreML.Specification.ScaleLayerParams.prototype.hasBias = false;
CoreML.Specification.ScaleLayerParams.prototype.bias = null;

CoreML.Specification.LoadConstantLayerParams = class LoadConstantLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LoadConstantLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.data = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LoadConstantLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.uint64());
                    break;
                case "data":
                    message.data = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LoadConstantLayerParams.prototype.data = null;

CoreML.Specification.L2NormalizeLayerParams = class L2NormalizeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.L2NormalizeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.L2NormalizeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.L2NormalizeLayerParams.prototype.epsilon = 0;

CoreML.Specification.FlattenLayerParams = class FlattenLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.FlattenLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FlattenLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.FlattenLayerParams.FlattenOrder);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FlattenLayerParams.prototype.mode = 0;

CoreML.Specification.FlattenLayerParams.FlattenOrder = {
    "CHANNEL_FIRST": 0,
    "CHANNEL_LAST": 1
};

CoreML.Specification.ReshapeLayerParams = class ReshapeLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReshapeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReshapeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetShape":
                    reader.array(message.targetShape, () => reader.int64());
                    break;
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.ReshapeLayerParams.ReshapeOrder);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReshapeLayerParams.prototype.mode = 0;

CoreML.Specification.ReshapeLayerParams.ReshapeOrder = {
    "CHANNEL_FIRST": 0,
    "CHANNEL_LAST": 1
};

CoreML.Specification.PermuteLayerParams = class PermuteLayerParams {

    constructor() {
        this.axis = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.PermuteLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.PermuteLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    reader.array(message.axis, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReorganizeDataLayerParams = class ReorganizeDataLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ReorganizeDataLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReorganizeDataLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.ReorganizeDataLayerParams.ReorganizationType);
                    break;
                case "blockSize":
                    message.blockSize = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReorganizeDataLayerParams.prototype.mode = 0;
CoreML.Specification.ReorganizeDataLayerParams.prototype.blockSize = 0n;

CoreML.Specification.ReorganizeDataLayerParams.ReorganizationType = {
    "SPACE_TO_DEPTH": 0,
    "DEPTH_TO_SPACE": 1,
    "PIXEL_SHUFFLE": 2
};

CoreML.Specification.SliceLayerParams = class SliceLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SliceLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SliceLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "startIndex":
                    message.startIndex = reader.int64();
                    break;
                case "endIndex":
                    message.endIndex = reader.int64();
                    break;
                case "stride":
                    message.stride = reader.uint64();
                    break;
                case "axis":
                    message.axis = reader.enum(CoreML.Specification.SliceLayerParams.SliceAxis);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SliceLayerParams.prototype.startIndex = 0n;
CoreML.Specification.SliceLayerParams.prototype.endIndex = 0n;
CoreML.Specification.SliceLayerParams.prototype.stride = 0n;
CoreML.Specification.SliceLayerParams.prototype.axis = 0;

CoreML.Specification.SliceLayerParams.SliceAxis = {
    "CHANNEL_AXIS": 0,
    "HEIGHT_AXIS": 1,
    "WIDTH_AXIS": 2
};

CoreML.Specification.ReduceLayerParams = class ReduceLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.ReduceLayerParams.ReduceOperation);
                    break;
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                case "axis":
                    message.axis = reader.enum(CoreML.Specification.ReduceLayerParams.ReduceAxis);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceLayerParams.prototype.mode = 0;
CoreML.Specification.ReduceLayerParams.prototype.epsilon = 0;
CoreML.Specification.ReduceLayerParams.prototype.axis = 0;

CoreML.Specification.ReduceLayerParams.ReduceOperation = {
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

CoreML.Specification.ReduceLayerParams.ReduceAxis = {
    "CHW": 0,
    "HW": 1,
    "C": 2,
    "H": 3,
    "W": 4
};

CoreML.Specification.CropLayerParams = class CropLayerParams {

    constructor() {
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CropLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.cropAmounts = CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CropLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "cropAmounts":
                    message.cropAmounts = CoreML.Specification.BorderAmounts.decodeText(reader);
                    break;
                case "offset":
                    reader.array(message.offset, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CropLayerParams.prototype.cropAmounts = null;

CoreML.Specification.AverageLayerParams = class AverageLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AverageLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AverageLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MaxLayerParams = class MaxLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MaxLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MaxLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MinLayerParams = class MinLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MinLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MinLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DotProductLayerParams = class DotProductLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.DotProductLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.DotProductLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "cosineSimilarity":
                    message.cosineSimilarity = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DotProductLayerParams.prototype.cosineSimilarity = false;

CoreML.Specification.MeanVarianceNormalizeLayerParams = class MeanVarianceNormalizeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MeanVarianceNormalizeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.MeanVarianceNormalizeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "acrossChannels":
                    message.acrossChannels = reader.bool();
                    break;
                case "normalizeVariance":
                    message.normalizeVariance = reader.bool();
                    break;
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.acrossChannels = false;
CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.normalizeVariance = false;
CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.epsilon = 0;

CoreML.Specification.SequenceRepeatLayerParams = class SequenceRepeatLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SequenceRepeatLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SequenceRepeatLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nRepetitions":
                    message.nRepetitions = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SequenceRepeatLayerParams.prototype.nRepetitions = 0n;

CoreML.Specification.SimpleRecurrentLayerParams = class SimpleRecurrentLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SimpleRecurrentLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activation = CoreML.Specification.ActivationParams.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.sequenceOutput = reader.bool();
                    break;
                case 20:
                    message.hasBiasVector = reader.bool();
                    break;
                case 30:
                    message.weightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.recursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.biasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SimpleRecurrentLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputVectorSize":
                    message.inputVectorSize = reader.uint64();
                    break;
                case "outputVectorSize":
                    message.outputVectorSize = reader.uint64();
                    break;
                case "activation":
                    message.activation = CoreML.Specification.ActivationParams.decodeText(reader);
                    break;
                case "sequenceOutput":
                    message.sequenceOutput = reader.bool();
                    break;
                case "hasBiasVector":
                    message.hasBiasVector = reader.bool();
                    break;
                case "weightMatrix":
                    message.weightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "recursionMatrix":
                    message.recursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "biasVector":
                    message.biasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "reverseInput":
                    message.reverseInput = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SimpleRecurrentLayerParams.prototype.inputVectorSize = 0n;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.outputVectorSize = 0n;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.activation = null;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.sequenceOutput = false;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.hasBiasVector = false;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.weightMatrix = null;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.recursionMatrix = null;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.biasVector = null;
CoreML.Specification.SimpleRecurrentLayerParams.prototype.reverseInput = false;

CoreML.Specification.GRULayerParams = class GRULayerParams {

    constructor() {
        this.activations = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.GRULayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activations.push(CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.sequenceOutput = reader.bool();
                    break;
                case 20:
                    message.hasBiasVectors = reader.bool();
                    break;
                case 30:
                    message.updateGateWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.resetGateWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.outputGateWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 50:
                    message.updateGateRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 51:
                    message.resetGateRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 52:
                    message.outputGateRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 70:
                    message.updateGateBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 71:
                    message.resetGateBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 72:
                    message.outputGateBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GRULayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputVectorSize":
                    message.inputVectorSize = reader.uint64();
                    break;
                case "outputVectorSize":
                    message.outputVectorSize = reader.uint64();
                    break;
                case "activations":
                    message.activations.push(CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "sequenceOutput":
                    message.sequenceOutput = reader.bool();
                    break;
                case "hasBiasVectors":
                    message.hasBiasVectors = reader.bool();
                    break;
                case "updateGateWeightMatrix":
                    message.updateGateWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "resetGateWeightMatrix":
                    message.resetGateWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateWeightMatrix":
                    message.outputGateWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "updateGateRecursionMatrix":
                    message.updateGateRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "resetGateRecursionMatrix":
                    message.resetGateRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateRecursionMatrix":
                    message.outputGateRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "updateGateBiasVector":
                    message.updateGateBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "resetGateBiasVector":
                    message.resetGateBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateBiasVector":
                    message.outputGateBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "reverseInput":
                    message.reverseInput = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GRULayerParams.prototype.inputVectorSize = 0n;
CoreML.Specification.GRULayerParams.prototype.outputVectorSize = 0n;
CoreML.Specification.GRULayerParams.prototype.sequenceOutput = false;
CoreML.Specification.GRULayerParams.prototype.hasBiasVectors = false;
CoreML.Specification.GRULayerParams.prototype.updateGateWeightMatrix = null;
CoreML.Specification.GRULayerParams.prototype.resetGateWeightMatrix = null;
CoreML.Specification.GRULayerParams.prototype.outputGateWeightMatrix = null;
CoreML.Specification.GRULayerParams.prototype.updateGateRecursionMatrix = null;
CoreML.Specification.GRULayerParams.prototype.resetGateRecursionMatrix = null;
CoreML.Specification.GRULayerParams.prototype.outputGateRecursionMatrix = null;
CoreML.Specification.GRULayerParams.prototype.updateGateBiasVector = null;
CoreML.Specification.GRULayerParams.prototype.resetGateBiasVector = null;
CoreML.Specification.GRULayerParams.prototype.outputGateBiasVector = null;
CoreML.Specification.GRULayerParams.prototype.reverseInput = false;

CoreML.Specification.LSTMParams = class LSTMParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LSTMParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LSTMParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sequenceOutput":
                    message.sequenceOutput = reader.bool();
                    break;
                case "hasBiasVectors":
                    message.hasBiasVectors = reader.bool();
                    break;
                case "forgetBias":
                    message.forgetBias = reader.bool();
                    break;
                case "hasPeepholeVectors":
                    message.hasPeepholeVectors = reader.bool();
                    break;
                case "coupledInputAndForgetGate":
                    message.coupledInputAndForgetGate = reader.bool();
                    break;
                case "cellClipThreshold":
                    message.cellClipThreshold = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LSTMParams.prototype.sequenceOutput = false;
CoreML.Specification.LSTMParams.prototype.hasBiasVectors = false;
CoreML.Specification.LSTMParams.prototype.forgetBias = false;
CoreML.Specification.LSTMParams.prototype.hasPeepholeVectors = false;
CoreML.Specification.LSTMParams.prototype.coupledInputAndForgetGate = false;
CoreML.Specification.LSTMParams.prototype.cellClipThreshold = 0;

CoreML.Specification.LSTMWeightParams = class LSTMWeightParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LSTMWeightParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputGateWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.forgetGateWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.blockInputWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.outputGateWeightMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.inputGateRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.forgetGateRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.blockInputRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 23:
                    message.outputGateRecursionMatrix = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.inputGateBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.forgetGateBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 42:
                    message.blockInputBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 43:
                    message.outputGateBiasVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 60:
                    message.inputGatePeepholeVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 61:
                    message.forgetGatePeepholeVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 62:
                    message.outputGatePeepholeVector = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LSTMWeightParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputGateWeightMatrix":
                    message.inputGateWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGateWeightMatrix":
                    message.forgetGateWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "blockInputWeightMatrix":
                    message.blockInputWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateWeightMatrix":
                    message.outputGateWeightMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "inputGateRecursionMatrix":
                    message.inputGateRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGateRecursionMatrix":
                    message.forgetGateRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "blockInputRecursionMatrix":
                    message.blockInputRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateRecursionMatrix":
                    message.outputGateRecursionMatrix = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "inputGateBiasVector":
                    message.inputGateBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGateBiasVector":
                    message.forgetGateBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "blockInputBiasVector":
                    message.blockInputBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateBiasVector":
                    message.outputGateBiasVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "inputGatePeepholeVector":
                    message.inputGatePeepholeVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGatePeepholeVector":
                    message.forgetGatePeepholeVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGatePeepholeVector":
                    message.outputGatePeepholeVector = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LSTMWeightParams.prototype.inputGateWeightMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.forgetGateWeightMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.blockInputWeightMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.outputGateWeightMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.inputGateRecursionMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.forgetGateRecursionMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.blockInputRecursionMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.outputGateRecursionMatrix = null;
CoreML.Specification.LSTMWeightParams.prototype.inputGateBiasVector = null;
CoreML.Specification.LSTMWeightParams.prototype.forgetGateBiasVector = null;
CoreML.Specification.LSTMWeightParams.prototype.blockInputBiasVector = null;
CoreML.Specification.LSTMWeightParams.prototype.outputGateBiasVector = null;
CoreML.Specification.LSTMWeightParams.prototype.inputGatePeepholeVector = null;
CoreML.Specification.LSTMWeightParams.prototype.forgetGatePeepholeVector = null;
CoreML.Specification.LSTMWeightParams.prototype.outputGatePeepholeVector = null;

CoreML.Specification.UniDirectionalLSTMLayerParams = class UniDirectionalLSTMLayerParams {

    constructor() {
        this.activations = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.UniDirectionalLSTMLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activations.push(CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.params = CoreML.Specification.LSTMParams.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.weightParams = CoreML.Specification.LSTMWeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.UniDirectionalLSTMLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputVectorSize":
                    message.inputVectorSize = reader.uint64();
                    break;
                case "outputVectorSize":
                    message.outputVectorSize = reader.uint64();
                    break;
                case "activations":
                    message.activations.push(CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "params":
                    message.params = CoreML.Specification.LSTMParams.decodeText(reader);
                    break;
                case "weightParams":
                    message.weightParams = CoreML.Specification.LSTMWeightParams.decodeText(reader);
                    break;
                case "reverseInput":
                    message.reverseInput = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.inputVectorSize = 0n;
CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.outputVectorSize = 0n;
CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.params = null;
CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.weightParams = null;
CoreML.Specification.UniDirectionalLSTMLayerParams.prototype.reverseInput = false;

CoreML.Specification.BiDirectionalLSTMLayerParams = class BiDirectionalLSTMLayerParams {

    constructor() {
        this.activationsForwardLSTM = [];
        this.activationsBackwardLSTM = [];
        this.weightParams = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.BiDirectionalLSTMLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputVectorSize = reader.uint64();
                    break;
                case 2:
                    message.outputVectorSize = reader.uint64();
                    break;
                case 10:
                    message.activationsForwardLSTM.push(CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.activationsBackwardLSTM.push(CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.params = CoreML.Specification.LSTMParams.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.weightParams.push(CoreML.Specification.LSTMWeightParams.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BiDirectionalLSTMLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputVectorSize":
                    message.inputVectorSize = reader.uint64();
                    break;
                case "outputVectorSize":
                    message.outputVectorSize = reader.uint64();
                    break;
                case "activationsForwardLSTM":
                    message.activationsForwardLSTM.push(CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "activationsBackwardLSTM":
                    message.activationsBackwardLSTM.push(CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "params":
                    message.params = CoreML.Specification.LSTMParams.decodeText(reader);
                    break;
                case "weightParams":
                    message.weightParams.push(CoreML.Specification.LSTMWeightParams.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BiDirectionalLSTMLayerParams.prototype.inputVectorSize = 0n;
CoreML.Specification.BiDirectionalLSTMLayerParams.prototype.outputVectorSize = 0n;
CoreML.Specification.BiDirectionalLSTMLayerParams.prototype.params = null;

CoreML.Specification.CustomLayerParams = class CustomLayerParams {

    constructor() {
        this.weights = [];
        this.parameters = {};
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CustomLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.className = reader.string();
                    break;
                case 20:
                    message.weights.push(CoreML.Specification.WeightParams.decode(reader, reader.uint32()));
                    break;
                case 30:
                    reader.entry(message.parameters, () => reader.string(), () => CoreML.Specification.CustomLayerParams.CustomLayerParamValue.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CustomLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "className":
                    message.className = reader.string();
                    break;
                case "weights":
                    message.weights.push(CoreML.Specification.WeightParams.decodeText(reader));
                    break;
                case "parameters":
                    reader.entry(message.parameters, () => reader.string(), () => CoreML.Specification.CustomLayerParams.CustomLayerParamValue.decodeText(reader));
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CustomLayerParams.prototype.className = "";
CoreML.Specification.CustomLayerParams.prototype.description = "";

CoreML.Specification.CustomLayerParams.CustomLayerParamValue = class CustomLayerParamValue {

    get value() {
        CoreML.Specification.CustomLayerParams.CustomLayerParamValue.valueSet = CoreML.Specification.CustomLayerParams.CustomLayerParamValue.valueSet || new Set(["doubleValue", "stringValue", "intValue", "longValue", "boolValue"]);
        return Object.keys(this).find((key) => CoreML.Specification.CustomLayerParams.CustomLayerParamValue.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.CustomLayerParams.CustomLayerParamValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CustomLayerParams.CustomLayerParamValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "doubleValue":
                    message.doubleValue = reader.double();
                    break;
                case "stringValue":
                    message.stringValue = reader.string();
                    break;
                case "intValue":
                    message.intValue = reader.int32();
                    break;
                case "longValue":
                    message.longValue = reader.int64();
                    break;
                case "boolValue":
                    message.boolValue = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TransposeLayerParams = class TransposeLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.TransposeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TransposeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BatchedMatMulLayerParams = class BatchedMatMulLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.BatchedMatMulLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.weights = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.bias = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BatchedMatMulLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "transposeA":
                    message.transposeA = reader.bool();
                    break;
                case "transposeB":
                    message.transposeB = reader.bool();
                    break;
                case "weightMatrixFirstDimension":
                    message.weightMatrixFirstDimension = reader.uint64();
                    break;
                case "weightMatrixSecondDimension":
                    message.weightMatrixSecondDimension = reader.uint64();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "int8DynamicQuantize":
                    message.int8DynamicQuantize = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BatchedMatMulLayerParams.prototype.transposeA = false;
CoreML.Specification.BatchedMatMulLayerParams.prototype.transposeB = false;
CoreML.Specification.BatchedMatMulLayerParams.prototype.weightMatrixFirstDimension = 0n;
CoreML.Specification.BatchedMatMulLayerParams.prototype.weightMatrixSecondDimension = 0n;
CoreML.Specification.BatchedMatMulLayerParams.prototype.hasBias = false;
CoreML.Specification.BatchedMatMulLayerParams.prototype.weights = null;
CoreML.Specification.BatchedMatMulLayerParams.prototype.bias = null;
CoreML.Specification.BatchedMatMulLayerParams.prototype.int8DynamicQuantize = false;

CoreML.Specification.ConcatNDLayerParams = class ConcatNDLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ConcatNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ConcatNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "interleave":
                    message.interleave = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConcatNDLayerParams.prototype.axis = 0n;
CoreML.Specification.ConcatNDLayerParams.prototype.interleave = false;

CoreML.Specification.SoftmaxNDLayerParams = class SoftmaxNDLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SoftmaxNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SoftmaxNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SoftmaxNDLayerParams.prototype.axis = 0n;

CoreML.Specification.ReverseLayerParams = class ReverseLayerParams {

    constructor() {
        this.reverseDim = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReverseLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReverseLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "reverseDim":
                    reader.array(message.reverseDim, () => reader.bool());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReverseSeqLayerParams = class ReverseSeqLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ReverseSeqLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReverseSeqLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "batchAxis":
                    message.batchAxis = reader.int64();
                    break;
                case "sequenceAxis":
                    message.sequenceAxis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReverseSeqLayerParams.prototype.batchAxis = 0n;
CoreML.Specification.ReverseSeqLayerParams.prototype.sequenceAxis = 0n;

CoreML.Specification.LoadConstantNDLayerParams = class LoadConstantNDLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LoadConstantNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.uint64(), tag);
                    break;
                case 2:
                    message.data = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LoadConstantNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.uint64());
                    break;
                case "data":
                    message.data = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LoadConstantNDLayerParams.prototype.data = null;

CoreML.Specification.FillLikeLayerParams = class FillLikeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.FillLikeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FillLikeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FillLikeLayerParams.prototype.value = 0;

CoreML.Specification.FillStaticLayerParams = class FillStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.FillStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FillStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.float();
                    break;
                case "targetShape":
                    reader.array(message.targetShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FillStaticLayerParams.prototype.value = 0;

CoreML.Specification.FillDynamicLayerParams = class FillDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.FillDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FillDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FillDynamicLayerParams.prototype.value = 0;

CoreML.Specification.WhereBroadcastableLayerParams = class WhereBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.WhereBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.WhereBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SinLayerParams = class SinLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SinLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SinLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CosLayerParams = class CosLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.CosLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CosLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TanLayerParams = class TanLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.TanLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.TanLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AsinLayerParams = class AsinLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AsinLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AsinLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AcosLayerParams = class AcosLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AcosLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AcosLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AtanLayerParams = class AtanLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AtanLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AtanLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SinhLayerParams = class SinhLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SinhLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SinhLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CoshLayerParams = class CoshLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.CoshLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CoshLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TanhLayerParams = class TanhLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.TanhLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.TanhLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AsinhLayerParams = class AsinhLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AsinhLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AsinhLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AcoshLayerParams = class AcoshLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AcoshLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AcoshLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AtanhLayerParams = class AtanhLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AtanhLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AtanhLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PowBroadcastableLayerParams = class PowBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.PowBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.PowBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Exp2LayerParams = class Exp2LayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.Exp2LayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Exp2LayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.WhereNonZeroLayerParams = class WhereNonZeroLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.WhereNonZeroLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.WhereNonZeroLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MatrixBandPartLayerParams = class MatrixBandPartLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MatrixBandPartLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.MatrixBandPartLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numLower":
                    message.numLower = reader.int64();
                    break;
                case "numUpper":
                    message.numUpper = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MatrixBandPartLayerParams.prototype.numLower = 0n;
CoreML.Specification.MatrixBandPartLayerParams.prototype.numUpper = 0n;

CoreML.Specification.UpperTriangularLayerParams = class UpperTriangularLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.UpperTriangularLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.UpperTriangularLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.UpperTriangularLayerParams.prototype.k = 0n;

CoreML.Specification.LowerTriangularLayerParams = class LowerTriangularLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.LowerTriangularLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.LowerTriangularLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LowerTriangularLayerParams.prototype.k = 0n;

CoreML.Specification.BroadcastToLikeLayerParams = class BroadcastToLikeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.BroadcastToLikeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BroadcastToLikeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BroadcastToStaticLayerParams = class BroadcastToStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.BroadcastToStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.BroadcastToStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetShape":
                    reader.array(message.targetShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.BroadcastToDynamicLayerParams = class BroadcastToDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.BroadcastToDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.BroadcastToDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AddBroadcastableLayerParams = class AddBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.AddBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AddBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MaxBroadcastableLayerParams = class MaxBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MaxBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MaxBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MinBroadcastableLayerParams = class MinBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MinBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MinBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ModBroadcastableLayerParams = class ModBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ModBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ModBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FloorDivBroadcastableLayerParams = class FloorDivBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.FloorDivBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.FloorDivBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SubtractBroadcastableLayerParams = class SubtractBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SubtractBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SubtractBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MultiplyBroadcastableLayerParams = class MultiplyBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.MultiplyBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.MultiplyBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DivideBroadcastableLayerParams = class DivideBroadcastableLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.DivideBroadcastableLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.DivideBroadcastableLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GatherLayerParams = class GatherLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GatherLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GatherLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GatherLayerParams.prototype.axis = 0n;

CoreML.Specification.ScatterMode = {
    "SCATTER_UPDATE": 0,
    "SCATTER_ADD": 1,
    "SCATTER_SUB": 2,
    "SCATTER_MUL": 3,
    "SCATTER_DIV": 4,
    "SCATTER_MAX": 5,
    "SCATTER_MIN": 6
};

CoreML.Specification.ScatterLayerParams = class ScatterLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ScatterLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ScatterLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.ScatterMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ScatterLayerParams.prototype.axis = 0n;
CoreML.Specification.ScatterLayerParams.prototype.mode = 0;

CoreML.Specification.GatherNDLayerParams = class GatherNDLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GatherNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.GatherNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ScatterNDLayerParams = class ScatterNDLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ScatterNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ScatterNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.ScatterMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ScatterNDLayerParams.prototype.mode = 0;

CoreML.Specification.GatherAlongAxisLayerParams = class GatherAlongAxisLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GatherAlongAxisLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GatherAlongAxisLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GatherAlongAxisLayerParams.prototype.axis = 0n;

CoreML.Specification.ScatterAlongAxisLayerParams = class ScatterAlongAxisLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ScatterAlongAxisLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ScatterAlongAxisLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.ScatterMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ScatterAlongAxisLayerParams.prototype.axis = 0n;
CoreML.Specification.ScatterAlongAxisLayerParams.prototype.mode = 0;

CoreML.Specification.StackLayerParams = class StackLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.StackLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.StackLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.StackLayerParams.prototype.axis = 0n;

CoreML.Specification.RankPreservingReshapeLayerParams = class RankPreservingReshapeLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.RankPreservingReshapeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RankPreservingReshapeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetShape":
                    reader.array(message.targetShape, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConstantPaddingLayerParams = class ConstantPaddingLayerParams {

    constructor() {
        this.padAmounts = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ConstantPaddingLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ConstantPaddingLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.float();
                    break;
                case "padAmounts":
                    reader.array(message.padAmounts, () => reader.uint64());
                    break;
                case "padToGivenOutputSizeMode":
                    message.padToGivenOutputSizeMode = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ConstantPaddingLayerParams.prototype.value = 0;
CoreML.Specification.ConstantPaddingLayerParams.prototype.padToGivenOutputSizeMode = false;

CoreML.Specification.RandomNormalLikeLayerParams = class RandomNormalLikeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomNormalLikeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomNormalLikeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "mean":
                    message.mean = reader.float();
                    break;
                case "stdDev":
                    message.stdDev = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomNormalLikeLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomNormalLikeLayerParams.prototype.mean = 0;
CoreML.Specification.RandomNormalLikeLayerParams.prototype.stdDev = 0;

CoreML.Specification.RandomNormalStaticLayerParams = class RandomNormalStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomNormalStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomNormalStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "mean":
                    message.mean = reader.float();
                    break;
                case "stdDev":
                    message.stdDev = reader.float();
                    break;
                case "outputShape":
                    reader.array(message.outputShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomNormalStaticLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomNormalStaticLayerParams.prototype.mean = 0;
CoreML.Specification.RandomNormalStaticLayerParams.prototype.stdDev = 0;

CoreML.Specification.RandomNormalDynamicLayerParams = class RandomNormalDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomNormalDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomNormalDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "mean":
                    message.mean = reader.float();
                    break;
                case "stdDev":
                    message.stdDev = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomNormalDynamicLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomNormalDynamicLayerParams.prototype.mean = 0;
CoreML.Specification.RandomNormalDynamicLayerParams.prototype.stdDev = 0;

CoreML.Specification.RandomUniformLikeLayerParams = class RandomUniformLikeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomUniformLikeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomUniformLikeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "minVal":
                    message.minVal = reader.float();
                    break;
                case "maxVal":
                    message.maxVal = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomUniformLikeLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomUniformLikeLayerParams.prototype.minVal = 0;
CoreML.Specification.RandomUniformLikeLayerParams.prototype.maxVal = 0;

CoreML.Specification.RandomUniformStaticLayerParams = class RandomUniformStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomUniformStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomUniformStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "minVal":
                    message.minVal = reader.float();
                    break;
                case "maxVal":
                    message.maxVal = reader.float();
                    break;
                case "outputShape":
                    reader.array(message.outputShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomUniformStaticLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomUniformStaticLayerParams.prototype.minVal = 0;
CoreML.Specification.RandomUniformStaticLayerParams.prototype.maxVal = 0;

CoreML.Specification.RandomUniformDynamicLayerParams = class RandomUniformDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomUniformDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomUniformDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "minVal":
                    message.minVal = reader.float();
                    break;
                case "maxVal":
                    message.maxVal = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomUniformDynamicLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomUniformDynamicLayerParams.prototype.minVal = 0;
CoreML.Specification.RandomUniformDynamicLayerParams.prototype.maxVal = 0;

CoreML.Specification.RandomBernoulliLikeLayerParams = class RandomBernoulliLikeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomBernoulliLikeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomBernoulliLikeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "prob":
                    message.prob = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomBernoulliLikeLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomBernoulliLikeLayerParams.prototype.prob = 0;

CoreML.Specification.RandomBernoulliStaticLayerParams = class RandomBernoulliStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomBernoulliStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomBernoulliStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "prob":
                    message.prob = reader.float();
                    break;
                case "outputShape":
                    reader.array(message.outputShape, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomBernoulliStaticLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomBernoulliStaticLayerParams.prototype.prob = 0;

CoreML.Specification.RandomBernoulliDynamicLayerParams = class RandomBernoulliDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RandomBernoulliDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RandomBernoulliDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "prob":
                    message.prob = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RandomBernoulliDynamicLayerParams.prototype.seed = 0n;
CoreML.Specification.RandomBernoulliDynamicLayerParams.prototype.prob = 0;

CoreML.Specification.CategoricalDistributionLayerParams = class CategoricalDistributionLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.CategoricalDistributionLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CategoricalDistributionLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "numSamples":
                    message.numSamples = reader.int64();
                    break;
                case "isLogits":
                    message.isLogits = reader.bool();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "temperature":
                    message.temperature = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CategoricalDistributionLayerParams.prototype.seed = 0n;
CoreML.Specification.CategoricalDistributionLayerParams.prototype.numSamples = 0n;
CoreML.Specification.CategoricalDistributionLayerParams.prototype.isLogits = false;
CoreML.Specification.CategoricalDistributionLayerParams.prototype.eps = 0;
CoreML.Specification.CategoricalDistributionLayerParams.prototype.temperature = 0;

CoreML.Specification.ReduceL1LayerParams = class ReduceL1LayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceL1LayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceL1LayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceL1LayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceL1LayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceL2LayerParams = class ReduceL2LayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceL2LayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceL2LayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceL2LayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceL2LayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceMaxLayerParams = class ReduceMaxLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceMaxLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceMaxLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceMaxLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceMaxLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceMinLayerParams = class ReduceMinLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceMinLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceMinLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceMinLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceMinLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceSumLayerParams = class ReduceSumLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceSumLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceSumLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceSumLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceSumLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceProdLayerParams = class ReduceProdLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceProdLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceProdLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceProdLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceProdLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceMeanLayerParams = class ReduceMeanLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceMeanLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceMeanLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceMeanLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceMeanLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceLogSumLayerParams = class ReduceLogSumLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceLogSumLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceLogSumLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceLogSumLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceLogSumLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceSumSquareLayerParams = class ReduceSumSquareLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceSumSquareLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceSumSquareLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceSumSquareLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceSumSquareLayerParams.prototype.reduceAll = false;

CoreML.Specification.ReduceLogSumExpLayerParams = class ReduceLogSumExpLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReduceLogSumExpLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReduceLogSumExpLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "keepDims":
                    message.keepDims = reader.bool();
                    break;
                case "reduceAll":
                    message.reduceAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReduceLogSumExpLayerParams.prototype.keepDims = false;
CoreML.Specification.ReduceLogSumExpLayerParams.prototype.reduceAll = false;

CoreML.Specification.ExpandDimsLayerParams = class ExpandDimsLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ExpandDimsLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ExpandDimsLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FlattenTo2DLayerParams = class FlattenTo2DLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.FlattenTo2DLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.FlattenTo2DLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FlattenTo2DLayerParams.prototype.axis = 0n;

CoreML.Specification.ReshapeStaticLayerParams = class ReshapeStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ReshapeStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ReshapeStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetShape":
                    reader.array(message.targetShape, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReshapeLikeLayerParams = class ReshapeLikeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ReshapeLikeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ReshapeLikeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ReshapeDynamicLayerParams = class ReshapeDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ReshapeDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ReshapeDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SqueezeLayerParams = class SqueezeLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SqueezeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SqueezeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
                    break;
                case "squeezeAll":
                    message.squeezeAll = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SqueezeLayerParams.prototype.squeezeAll = false;

CoreML.Specification.TopKLayerParams = class TopKLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.TopKLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TopKLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "K":
                    message.K = reader.uint64();
                    break;
                case "useBottomK":
                    message.useBottomK = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TopKLayerParams.prototype.axis = 0n;
CoreML.Specification.TopKLayerParams.prototype.K = 0n;
CoreML.Specification.TopKLayerParams.prototype.useBottomK = false;

CoreML.Specification.ArgMaxLayerParams = class ArgMaxLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ArgMaxLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ArgMaxLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "removeDim":
                    message.removeDim = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ArgMaxLayerParams.prototype.axis = 0n;
CoreML.Specification.ArgMaxLayerParams.prototype.removeDim = false;

CoreML.Specification.ArgMinLayerParams = class ArgMinLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ArgMinLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ArgMinLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "removeDim":
                    message.removeDim = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ArgMinLayerParams.prototype.axis = 0n;
CoreML.Specification.ArgMinLayerParams.prototype.removeDim = false;

CoreML.Specification.SplitNDLayerParams = class SplitNDLayerParams {

    constructor() {
        this.splitSizes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SplitNDLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SplitNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "numSplits":
                    message.numSplits = reader.uint64();
                    break;
                case "splitSizes":
                    reader.array(message.splitSizes, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SplitNDLayerParams.prototype.axis = 0n;
CoreML.Specification.SplitNDLayerParams.prototype.numSplits = 0n;

CoreML.Specification.CeilLayerParams = class CeilLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.CeilLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.CeilLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RoundLayerParams = class RoundLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RoundLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.RoundLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.FloorLayerParams = class FloorLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.FloorLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.FloorLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SignLayerParams = class SignLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SignLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SignLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ClipLayerParams = class ClipLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ClipLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ClipLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "minVal":
                    message.minVal = reader.float();
                    break;
                case "maxVal":
                    message.maxVal = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ClipLayerParams.prototype.minVal = 0;
CoreML.Specification.ClipLayerParams.prototype.maxVal = 0;

CoreML.Specification.SliceStaticLayerParams = class SliceStaticLayerParams {

    constructor() {
        this.beginIds = [];
        this.beginMasks = [];
        this.endIds = [];
        this.endMasks = [];
        this.strides = [];
        this.squeezeMasks = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SliceStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SliceStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "beginIds":
                    reader.array(message.beginIds, () => reader.int64());
                    break;
                case "beginMasks":
                    reader.array(message.beginMasks, () => reader.bool());
                    break;
                case "endIds":
                    reader.array(message.endIds, () => reader.int64());
                    break;
                case "endMasks":
                    reader.array(message.endMasks, () => reader.bool());
                    break;
                case "strides":
                    reader.array(message.strides, () => reader.int64());
                    break;
                case "squeezeMasks":
                    reader.array(message.squeezeMasks, () => reader.bool());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SliceDynamicLayerParams = class SliceDynamicLayerParams {

    constructor() {
        this.beginMasks = [];
        this.endIds = [];
        this.endMasks = [];
        this.strides = [];
        this.squeezeMasks = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SliceDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SliceDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "beginMasks":
                    reader.array(message.beginMasks, () => reader.bool());
                    break;
                case "endIds":
                    reader.array(message.endIds, () => reader.int64());
                    break;
                case "endMasks":
                    reader.array(message.endMasks, () => reader.bool());
                    break;
                case "strides":
                    reader.array(message.strides, () => reader.int64());
                    break;
                case "squeezeMasks":
                    reader.array(message.squeezeMasks, () => reader.bool());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TileLayerParams = class TileLayerParams {

    constructor() {
        this.reps = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.TileLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TileLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "reps":
                    reader.array(message.reps, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GetShapeLayerParams = class GetShapeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GetShapeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.GetShapeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ErfLayerParams = class ErfLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ErfLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ErfLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GeluLayerParams = class GeluLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.GeluLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.GeluLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum(CoreML.Specification.GeluLayerParams.GeluMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.GeluLayerParams.prototype.mode = 0;

CoreML.Specification.GeluLayerParams.GeluMode = {
    "EXACT": 0,
    "TANH_APPROXIMATION": 1,
    "SIGMOID_APPROXIMATION": 2
};

CoreML.Specification.RangeStaticLayerParams = class RangeStaticLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RangeStaticLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RangeStaticLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "endValue":
                    message.endValue = reader.float();
                    break;
                case "startValue":
                    message.startValue = reader.float();
                    break;
                case "stepSizeValue":
                    message.stepSizeValue = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RangeStaticLayerParams.prototype.endValue = 0;
CoreML.Specification.RangeStaticLayerParams.prototype.startValue = 0;
CoreML.Specification.RangeStaticLayerParams.prototype.stepSizeValue = 0;

CoreML.Specification.RangeDynamicLayerParams = class RangeDynamicLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.RangeDynamicLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RangeDynamicLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "startValue":
                    message.startValue = reader.float();
                    break;
                case "stepSizeValue":
                    message.stepSizeValue = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RangeDynamicLayerParams.prototype.startValue = 0;
CoreML.Specification.RangeDynamicLayerParams.prototype.stepSizeValue = 0;

CoreML.Specification.SlidingWindowsLayerParams = class SlidingWindowsLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SlidingWindowsLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SlidingWindowsLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "windowSize":
                    message.windowSize = reader.uint64();
                    break;
                case "step":
                    message.step = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SlidingWindowsLayerParams.prototype.axis = 0n;
CoreML.Specification.SlidingWindowsLayerParams.prototype.windowSize = 0n;
CoreML.Specification.SlidingWindowsLayerParams.prototype.step = 0n;

CoreML.Specification.LayerNormalizationLayerParams = class LayerNormalizationLayerParams {

    constructor() {
        this.normalizedShape = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LayerNormalizationLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.normalizedShape = reader.array(message.normalizedShape, () => reader.int64(), tag);
                    break;
                case 2:
                    message.eps = reader.float();
                    break;
                case 3:
                    message.gamma = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.beta = CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LayerNormalizationLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "normalizedShape":
                    reader.array(message.normalizedShape, () => reader.int64());
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "gamma":
                    message.gamma = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "beta":
                    message.beta = CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LayerNormalizationLayerParams.prototype.eps = 0;
CoreML.Specification.LayerNormalizationLayerParams.prototype.gamma = null;
CoreML.Specification.LayerNormalizationLayerParams.prototype.beta = null;

CoreML.Specification.NonMaximumSuppressionLayerParams = class NonMaximumSuppressionLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.NonMaximumSuppressionLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NonMaximumSuppressionLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "iouThreshold":
                    message.iouThreshold = reader.float();
                    break;
                case "scoreThreshold":
                    message.scoreThreshold = reader.float();
                    break;
                case "maxBoxes":
                    message.maxBoxes = reader.uint64();
                    break;
                case "perClassSuppression":
                    message.perClassSuppression = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.iouThreshold = 0;
CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.scoreThreshold = 0;
CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.maxBoxes = 0n;
CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.perClassSuppression = false;

CoreML.Specification.ClampedReLULayerParams = class ClampedReLULayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ClampedReLULayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ClampedReLULayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ClampedReLULayerParams.prototype.alpha = 0;
CoreML.Specification.ClampedReLULayerParams.prototype.beta = 0;

CoreML.Specification.ArgSortLayerParams = class ArgSortLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.ArgSortLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ArgSortLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "descending":
                    message.descending = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ArgSortLayerParams.prototype.axis = 0n;
CoreML.Specification.ArgSortLayerParams.prototype.descending = false;

CoreML.Specification.SliceBySizeLayerParams = class SliceBySizeLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.SliceBySizeLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SliceBySizeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    message.size = reader.int64();
                    break;
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SliceBySizeLayerParams.prototype.size = 0n;
CoreML.Specification.SliceBySizeLayerParams.prototype.axis = 0n;

CoreML.Specification.NeuralNetworkClassifier = class NeuralNetworkClassifier {

    constructor() {
        this.layers = [];
        this.preprocessing = [];
    }

    get ClassLabels() {
        CoreML.Specification.NeuralNetworkClassifier.ClassLabelsSet = CoreML.Specification.NeuralNetworkClassifier.ClassLabelsSet || new Set(["stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.NeuralNetworkClassifier.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetworkClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layers.push(CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.preprocessing.push(CoreML.Specification.NeuralNetworkPreprocessing.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.arrayInputShapeMapping = reader.int32();
                    break;
                case 6:
                    message.imageInputShapeMapping = reader.int32();
                    break;
                case 10:
                    message.updateParams = CoreML.Specification.NetworkUpdateParameters.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetworkClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layers":
                    message.layers.push(CoreML.Specification.NeuralNetworkLayer.decodeText(reader));
                    break;
                case "preprocessing":
                    message.preprocessing.push(CoreML.Specification.NeuralNetworkPreprocessing.decodeText(reader));
                    break;
                case "arrayInputShapeMapping":
                    message.arrayInputShapeMapping = reader.enum(CoreML.Specification.NeuralNetworkMultiArrayShapeMapping);
                    break;
                case "imageInputShapeMapping":
                    message.imageInputShapeMapping = reader.enum(CoreML.Specification.NeuralNetworkImageShapeMapping);
                    break;
                case "updateParams":
                    message.updateParams = CoreML.Specification.NetworkUpdateParameters.decodeText(reader);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "labelProbabilityLayerName":
                    message.labelProbabilityLayerName = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetworkClassifier.prototype.arrayInputShapeMapping = 0;
CoreML.Specification.NeuralNetworkClassifier.prototype.imageInputShapeMapping = 0;
CoreML.Specification.NeuralNetworkClassifier.prototype.updateParams = null;
CoreML.Specification.NeuralNetworkClassifier.prototype.labelProbabilityLayerName = "";

CoreML.Specification.OneHotLayerParams = class OneHotLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.OneHotLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.OneHotLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "oneHotVectorSize":
                    message.oneHotVectorSize = reader.uint64();
                    break;
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "onValue":
                    message.onValue = reader.float();
                    break;
                case "offValue":
                    message.offValue = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.OneHotLayerParams.prototype.oneHotVectorSize = 0n;
CoreML.Specification.OneHotLayerParams.prototype.axis = 0n;
CoreML.Specification.OneHotLayerParams.prototype.onValue = 0;
CoreML.Specification.OneHotLayerParams.prototype.offValue = 0;

CoreML.Specification.CumSumLayerParams = class CumSumLayerParams {

    static decode(reader, length) {
        const message = new CoreML.Specification.CumSumLayerParams();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CumSumLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "excludeFinalSum":
                    message.excludeFinalSum = reader.bool();
                    break;
                case "reverse":
                    message.reverse = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CumSumLayerParams.prototype.axis = 0n;
CoreML.Specification.CumSumLayerParams.prototype.excludeFinalSum = false;
CoreML.Specification.CumSumLayerParams.prototype.reverse = false;

CoreML.Specification.NeuralNetworkRegressor = class NeuralNetworkRegressor {

    constructor() {
        this.layers = [];
        this.preprocessing = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NeuralNetworkRegressor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layers.push(CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.preprocessing.push(CoreML.Specification.NeuralNetworkPreprocessing.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.arrayInputShapeMapping = reader.int32();
                    break;
                case 6:
                    message.imageInputShapeMapping = reader.int32();
                    break;
                case 10:
                    message.updateParams = CoreML.Specification.NetworkUpdateParameters.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.NeuralNetworkRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layers":
                    message.layers.push(CoreML.Specification.NeuralNetworkLayer.decodeText(reader));
                    break;
                case "preprocessing":
                    message.preprocessing.push(CoreML.Specification.NeuralNetworkPreprocessing.decodeText(reader));
                    break;
                case "arrayInputShapeMapping":
                    message.arrayInputShapeMapping = reader.enum(CoreML.Specification.NeuralNetworkMultiArrayShapeMapping);
                    break;
                case "imageInputShapeMapping":
                    message.imageInputShapeMapping = reader.enum(CoreML.Specification.NeuralNetworkImageShapeMapping);
                    break;
                case "updateParams":
                    message.updateParams = CoreML.Specification.NetworkUpdateParameters.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NeuralNetworkRegressor.prototype.arrayInputShapeMapping = 0;
CoreML.Specification.NeuralNetworkRegressor.prototype.imageInputShapeMapping = 0;
CoreML.Specification.NeuralNetworkRegressor.prototype.updateParams = null;

CoreML.Specification.NetworkUpdateParameters = class NetworkUpdateParameters {

    constructor() {
        this.lossLayers = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NetworkUpdateParameters();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lossLayers.push(CoreML.Specification.LossLayer.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.optimizer = CoreML.Specification.Optimizer.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.epochs = CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.shuffle = CoreML.Specification.BoolParameter.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.seed = CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.NetworkUpdateParameters();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lossLayers":
                    message.lossLayers.push(CoreML.Specification.LossLayer.decodeText(reader));
                    break;
                case "optimizer":
                    message.optimizer = CoreML.Specification.Optimizer.decodeText(reader);
                    break;
                case "epochs":
                    message.epochs = CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "shuffle":
                    message.shuffle = CoreML.Specification.BoolParameter.decodeText(reader);
                    break;
                case "seed":
                    message.seed = CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NetworkUpdateParameters.prototype.optimizer = null;
CoreML.Specification.NetworkUpdateParameters.prototype.epochs = null;
CoreML.Specification.NetworkUpdateParameters.prototype.shuffle = null;
CoreML.Specification.NetworkUpdateParameters.prototype.seed = null;

CoreML.Specification.LossLayer = class LossLayer {

    get LossLayerType() {
        CoreML.Specification.LossLayer.LossLayerTypeSet = CoreML.Specification.LossLayer.LossLayerTypeSet || new Set(["categoricalCrossEntropyLossLayer", "meanSquaredErrorLossLayer"]);
        return Object.keys(this).find((key) => CoreML.Specification.LossLayer.LossLayerTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LossLayer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 10:
                    message.categoricalCrossEntropyLossLayer = CoreML.Specification.CategoricalCrossEntropyLossLayer.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.meanSquaredErrorLossLayer = CoreML.Specification.MeanSquaredErrorLossLayer.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LossLayer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "categoricalCrossEntropyLossLayer":
                    message.categoricalCrossEntropyLossLayer = CoreML.Specification.CategoricalCrossEntropyLossLayer.decodeText(reader);
                    break;
                case "meanSquaredErrorLossLayer":
                    message.meanSquaredErrorLossLayer = CoreML.Specification.MeanSquaredErrorLossLayer.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LossLayer.prototype.name = "";

CoreML.Specification.CategoricalCrossEntropyLossLayer = class CategoricalCrossEntropyLossLayer {

    static decode(reader, length) {
        const message = new CoreML.Specification.CategoricalCrossEntropyLossLayer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.CategoricalCrossEntropyLossLayer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    message.input = reader.string();
                    break;
                case "target":
                    message.target = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.CategoricalCrossEntropyLossLayer.prototype.input = "";
CoreML.Specification.CategoricalCrossEntropyLossLayer.prototype.target = "";

CoreML.Specification.MeanSquaredErrorLossLayer = class MeanSquaredErrorLossLayer {

    static decode(reader, length) {
        const message = new CoreML.Specification.MeanSquaredErrorLossLayer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.MeanSquaredErrorLossLayer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    message.input = reader.string();
                    break;
                case "target":
                    message.target = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.MeanSquaredErrorLossLayer.prototype.input = "";
CoreML.Specification.MeanSquaredErrorLossLayer.prototype.target = "";

CoreML.Specification.Optimizer = class Optimizer {

    get OptimizerType() {
        CoreML.Specification.Optimizer.OptimizerTypeSet = CoreML.Specification.Optimizer.OptimizerTypeSet || new Set(["sgdOptimizer", "adamOptimizer"]);
        return Object.keys(this).find((key) => CoreML.Specification.Optimizer.OptimizerTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Optimizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 10:
                    message.sgdOptimizer = CoreML.Specification.SGDOptimizer.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.adamOptimizer = CoreML.Specification.AdamOptimizer.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Optimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sgdOptimizer":
                    message.sgdOptimizer = CoreML.Specification.SGDOptimizer.decodeText(reader);
                    break;
                case "adamOptimizer":
                    message.adamOptimizer = CoreML.Specification.AdamOptimizer.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SGDOptimizer = class SGDOptimizer {

    static decode(reader, length) {
        const message = new CoreML.Specification.SGDOptimizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.learningRate = CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.miniBatchSize = CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.momentum = CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SGDOptimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "learningRate":
                    message.learningRate = CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "miniBatchSize":
                    message.miniBatchSize = CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "momentum":
                    message.momentum = CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SGDOptimizer.prototype.learningRate = null;
CoreML.Specification.SGDOptimizer.prototype.miniBatchSize = null;
CoreML.Specification.SGDOptimizer.prototype.momentum = null;

CoreML.Specification.AdamOptimizer = class AdamOptimizer {

    static decode(reader, length) {
        const message = new CoreML.Specification.AdamOptimizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.learningRate = CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.miniBatchSize = CoreML.Specification.Int64Parameter.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.beta1 = CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.beta2 = CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.eps = CoreML.Specification.DoubleParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.AdamOptimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "learningRate":
                    message.learningRate = CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "miniBatchSize":
                    message.miniBatchSize = CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "beta1":
                    message.beta1 = CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "beta2":
                    message.beta2 = CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "eps":
                    message.eps = CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.AdamOptimizer.prototype.learningRate = null;
CoreML.Specification.AdamOptimizer.prototype.miniBatchSize = null;
CoreML.Specification.AdamOptimizer.prototype.beta1 = null;
CoreML.Specification.AdamOptimizer.prototype.beta2 = null;
CoreML.Specification.AdamOptimizer.prototype.eps = null;

CoreML.Specification.Normalizer = class Normalizer {

    static decode(reader, length) {
        const message = new CoreML.Specification.Normalizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Normalizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "normType":
                    message.normType = reader.enum(CoreML.Specification.Normalizer.NormType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Normalizer.prototype.normType = 0;

CoreML.Specification.Normalizer.NormType = {
    "LMax": 0,
    "L1": 1,
    "L2": 2
};

CoreML.Specification.OneHotEncoder = class OneHotEncoder {

    get CategoryType() {
        CoreML.Specification.OneHotEncoder.CategoryTypeSet = CoreML.Specification.OneHotEncoder.CategoryTypeSet || new Set(["stringCategories", "int64Categories"]);
        return Object.keys(this).find((key) => CoreML.Specification.OneHotEncoder.CategoryTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.OneHotEncoder();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.stringCategories = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.int64Categories = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.OneHotEncoder();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "stringCategories":
                    message.stringCategories = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64Categories":
                    message.int64Categories = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "outputSparse":
                    message.outputSparse = reader.bool();
                    break;
                case "handleUnknown":
                    message.handleUnknown = reader.enum(CoreML.Specification.OneHotEncoder.HandleUnknown);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.OneHotEncoder.prototype.outputSparse = false;
CoreML.Specification.OneHotEncoder.prototype.handleUnknown = 0;

CoreML.Specification.OneHotEncoder.HandleUnknown = {
    "ErrorOnUnknown": 0,
    "IgnoreUnknown": 1
};

CoreML.Specification.Scaler = class Scaler {

    constructor() {
        this.shiftValue = [];
        this.scaleValue = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Scaler();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Scaler();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shiftValue":
                    reader.array(message.shiftValue, () => reader.double());
                    break;
                case "scaleValue":
                    reader.array(message.scaleValue, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NonMaximumSuppression = class NonMaximumSuppression {

    get SuppressionMethod() {
        CoreML.Specification.NonMaximumSuppression.SuppressionMethodSet = CoreML.Specification.NonMaximumSuppression.SuppressionMethodSet || new Set(["pickTop"]);
        return Object.keys(this).find((key) => CoreML.Specification.NonMaximumSuppression.SuppressionMethodSet.has(key) && this[key] !== null);
    }

    get ClassLabels() {
        CoreML.Specification.NonMaximumSuppression.ClassLabelsSet = CoreML.Specification.NonMaximumSuppression.ClassLabelsSet || new Set(["stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.NonMaximumSuppression.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.NonMaximumSuppression();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pickTop = CoreML.Specification.NonMaximumSuppression.PickTop.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NonMaximumSuppression();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pickTop":
                    message.pickTop = CoreML.Specification.NonMaximumSuppression.PickTop.decodeText(reader);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "iouThreshold":
                    message.iouThreshold = reader.double();
                    break;
                case "confidenceThreshold":
                    message.confidenceThreshold = reader.double();
                    break;
                case "confidenceInputFeatureName":
                    message.confidenceInputFeatureName = reader.string();
                    break;
                case "coordinatesInputFeatureName":
                    message.coordinatesInputFeatureName = reader.string();
                    break;
                case "iouThresholdInputFeatureName":
                    message.iouThresholdInputFeatureName = reader.string();
                    break;
                case "confidenceThresholdInputFeatureName":
                    message.confidenceThresholdInputFeatureName = reader.string();
                    break;
                case "confidenceOutputFeatureName":
                    message.confidenceOutputFeatureName = reader.string();
                    break;
                case "coordinatesOutputFeatureName":
                    message.coordinatesOutputFeatureName = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NonMaximumSuppression.prototype.iouThreshold = 0;
CoreML.Specification.NonMaximumSuppression.prototype.confidenceThreshold = 0;
CoreML.Specification.NonMaximumSuppression.prototype.confidenceInputFeatureName = "";
CoreML.Specification.NonMaximumSuppression.prototype.coordinatesInputFeatureName = "";
CoreML.Specification.NonMaximumSuppression.prototype.iouThresholdInputFeatureName = "";
CoreML.Specification.NonMaximumSuppression.prototype.confidenceThresholdInputFeatureName = "";
CoreML.Specification.NonMaximumSuppression.prototype.confidenceOutputFeatureName = "";
CoreML.Specification.NonMaximumSuppression.prototype.coordinatesOutputFeatureName = "";

CoreML.Specification.NonMaximumSuppression.PickTop = class PickTop {

    static decode(reader, length) {
        const message = new CoreML.Specification.NonMaximumSuppression.PickTop();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.NonMaximumSuppression.PickTop();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "perClass":
                    message.perClass = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.NonMaximumSuppression.PickTop.prototype.perClass = false;

CoreML.Specification.LinearKernel = class LinearKernel {

    static decode(reader, length) {
        const message = new CoreML.Specification.LinearKernel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LinearKernel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RBFKernel = class RBFKernel {

    static decode(reader, length) {
        const message = new CoreML.Specification.RBFKernel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.RBFKernel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "gamma":
                    message.gamma = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.RBFKernel.prototype.gamma = 0;

CoreML.Specification.PolyKernel = class PolyKernel {

    static decode(reader, length) {
        const message = new CoreML.Specification.PolyKernel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.PolyKernel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "degree":
                    message.degree = reader.int32();
                    break;
                case "c":
                    message.c = reader.double();
                    break;
                case "gamma":
                    message.gamma = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.PolyKernel.prototype.degree = 0;
CoreML.Specification.PolyKernel.prototype.c = 0;
CoreML.Specification.PolyKernel.prototype.gamma = 0;

CoreML.Specification.SigmoidKernel = class SigmoidKernel {

    static decode(reader, length) {
        const message = new CoreML.Specification.SigmoidKernel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SigmoidKernel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "gamma":
                    message.gamma = reader.double();
                    break;
                case "c":
                    message.c = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SigmoidKernel.prototype.gamma = 0;
CoreML.Specification.SigmoidKernel.prototype.c = 0;

CoreML.Specification.Kernel = class Kernel {

    get kernel() {
        CoreML.Specification.Kernel.kernelSet = CoreML.Specification.Kernel.kernelSet || new Set(["linearKernel", "rbfKernel", "polyKernel", "sigmoidKernel"]);
        return Object.keys(this).find((key) => CoreML.Specification.Kernel.kernelSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Kernel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.linearKernel = CoreML.Specification.LinearKernel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.rbfKernel = CoreML.Specification.RBFKernel.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.polyKernel = CoreML.Specification.PolyKernel.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.sigmoidKernel = CoreML.Specification.SigmoidKernel.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.Kernel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linearKernel":
                    message.linearKernel = CoreML.Specification.LinearKernel.decodeText(reader);
                    break;
                case "rbfKernel":
                    message.rbfKernel = CoreML.Specification.RBFKernel.decodeText(reader);
                    break;
                case "polyKernel":
                    message.polyKernel = CoreML.Specification.PolyKernel.decodeText(reader);
                    break;
                case "sigmoidKernel":
                    message.sigmoidKernel = CoreML.Specification.SigmoidKernel.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SparseNode = class SparseNode {

    static decode(reader, length) {
        const message = new CoreML.Specification.SparseNode();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SparseNode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "index":
                    message.index = reader.int32();
                    break;
                case "value":
                    message.value = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SparseNode.prototype.index = 0;
CoreML.Specification.SparseNode.prototype.value = 0;

CoreML.Specification.SparseVector = class SparseVector {

    constructor() {
        this.nodes = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SparseVector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push(CoreML.Specification.SparseNode.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SparseVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push(CoreML.Specification.SparseNode.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SparseSupportVectors = class SparseSupportVectors {

    constructor() {
        this.vectors = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SparseSupportVectors();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vectors.push(CoreML.Specification.SparseVector.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SparseSupportVectors();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vectors":
                    message.vectors.push(CoreML.Specification.SparseVector.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DenseVector = class DenseVector {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.DenseVector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.DenseVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    reader.array(message.values, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.DenseSupportVectors = class DenseSupportVectors {

    constructor() {
        this.vectors = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.DenseSupportVectors();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.vectors.push(CoreML.Specification.DenseVector.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.DenseSupportVectors();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vectors":
                    message.vectors.push(CoreML.Specification.DenseVector.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.Coefficients = class Coefficients {

    constructor() {
        this.alpha = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.Coefficients();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.Coefficients();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    reader.array(message.alpha, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SupportVectorRegressor = class SupportVectorRegressor {

    get supportVectors() {
        CoreML.Specification.SupportVectorRegressor.supportVectorsSet = CoreML.Specification.SupportVectorRegressor.supportVectorsSet || new Set(["sparseSupportVectors", "denseSupportVectors"]);
        return Object.keys(this).find((key) => CoreML.Specification.SupportVectorRegressor.supportVectorsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SupportVectorRegressor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = CoreML.Specification.Kernel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.sparseSupportVectors = CoreML.Specification.SparseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.denseSupportVectors = CoreML.Specification.DenseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.coefficients = CoreML.Specification.Coefficients.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.SupportVectorRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = CoreML.Specification.Kernel.decodeText(reader);
                    break;
                case "sparseSupportVectors":
                    message.sparseSupportVectors = CoreML.Specification.SparseSupportVectors.decodeText(reader);
                    break;
                case "denseSupportVectors":
                    message.denseSupportVectors = CoreML.Specification.DenseSupportVectors.decodeText(reader);
                    break;
                case "coefficients":
                    message.coefficients = CoreML.Specification.Coefficients.decodeText(reader);
                    break;
                case "rho":
                    message.rho = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SupportVectorRegressor.prototype.kernel = null;
CoreML.Specification.SupportVectorRegressor.prototype.coefficients = null;
CoreML.Specification.SupportVectorRegressor.prototype.rho = 0;

CoreML.Specification.SupportVectorClassifier = class SupportVectorClassifier {

    constructor() {
        this.numberOfSupportVectorsPerClass = [];
        this.coefficients = [];
        this.rho = [];
        this.probA = [];
        this.probB = [];
    }

    get supportVectors() {
        CoreML.Specification.SupportVectorClassifier.supportVectorsSet = CoreML.Specification.SupportVectorClassifier.supportVectorsSet || new Set(["sparseSupportVectors", "denseSupportVectors"]);
        return Object.keys(this).find((key) => CoreML.Specification.SupportVectorClassifier.supportVectorsSet.has(key) && this[key] !== null);
    }

    get ClassLabels() {
        CoreML.Specification.SupportVectorClassifier.ClassLabelsSet = CoreML.Specification.SupportVectorClassifier.ClassLabelsSet || new Set(["stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.SupportVectorClassifier.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.SupportVectorClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = CoreML.Specification.Kernel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.numberOfSupportVectorsPerClass = reader.array(message.numberOfSupportVectorsPerClass, () => reader.int32(), tag);
                    break;
                case 3:
                    message.sparseSupportVectors = CoreML.Specification.SparseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.denseSupportVectors = CoreML.Specification.DenseSupportVectors.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.coefficients.push(CoreML.Specification.Coefficients.decode(reader, reader.uint32()));
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
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.SupportVectorClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = CoreML.Specification.Kernel.decodeText(reader);
                    break;
                case "numberOfSupportVectorsPerClass":
                    reader.array(message.numberOfSupportVectorsPerClass, () => reader.int32());
                    break;
                case "sparseSupportVectors":
                    message.sparseSupportVectors = CoreML.Specification.SparseSupportVectors.decodeText(reader);
                    break;
                case "denseSupportVectors":
                    message.denseSupportVectors = CoreML.Specification.DenseSupportVectors.decodeText(reader);
                    break;
                case "coefficients":
                    message.coefficients.push(CoreML.Specification.Coefficients.decodeText(reader));
                    break;
                case "rho":
                    reader.array(message.rho, () => reader.double());
                    break;
                case "probA":
                    reader.array(message.probA, () => reader.double());
                    break;
                case "probB":
                    reader.array(message.probB, () => reader.double());
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.SupportVectorClassifier.prototype.kernel = null;

CoreML.Specification.TreeEnsemblePostEvaluationTransform = {
    "NoTransform": 0,
    "Classification_SoftMax": 1,
    "Regression_Logistic": 2,
    "Classification_SoftMaxWithZeroClassReference": 3
};

CoreML.Specification.TreeEnsembleParameters = class TreeEnsembleParameters {

    constructor() {
        this.nodes = [];
        this.basePredictionValue = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.TreeEnsembleParameters();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push(CoreML.Specification.TreeEnsembleParameters.TreeNode.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TreeEnsembleParameters();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push(CoreML.Specification.TreeEnsembleParameters.TreeNode.decodeText(reader));
                    break;
                case "numPredictionDimensions":
                    message.numPredictionDimensions = reader.uint64();
                    break;
                case "basePredictionValue":
                    reader.array(message.basePredictionValue, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TreeEnsembleParameters.prototype.numPredictionDimensions = 0n;

CoreML.Specification.TreeEnsembleParameters.TreeNode = class TreeNode {

    constructor() {
        this.evaluationInfo = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.TreeEnsembleParameters.TreeNode();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.evaluationInfo.push(CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TreeEnsembleParameters.TreeNode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "treeId":
                    message.treeId = reader.uint64();
                    break;
                case "nodeId":
                    message.nodeId = reader.uint64();
                    break;
                case "nodeBehavior":
                    message.nodeBehavior = reader.enum(CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior);
                    break;
                case "branchFeatureIndex":
                    message.branchFeatureIndex = reader.uint64();
                    break;
                case "branchFeatureValue":
                    message.branchFeatureValue = reader.double();
                    break;
                case "trueChildNodeId":
                    message.trueChildNodeId = reader.uint64();
                    break;
                case "falseChildNodeId":
                    message.falseChildNodeId = reader.uint64();
                    break;
                case "missingValueTracksTrueChild":
                    message.missingValueTracksTrueChild = reader.bool();
                    break;
                case "evaluationInfo":
                    message.evaluationInfo.push(CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.decodeText(reader));
                    break;
                case "relativeHitRate":
                    message.relativeHitRate = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.treeId = 0n;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.nodeId = 0n;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.nodeBehavior = 0;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.branchFeatureIndex = 0n;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.branchFeatureValue = 0;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.trueChildNodeId = 0n;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.falseChildNodeId = 0n;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.missingValueTracksTrueChild = false;
CoreML.Specification.TreeEnsembleParameters.TreeNode.prototype.relativeHitRate = 0;

CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior = {
    "BranchOnValueLessThanEqual": 0,
    "BranchOnValueLessThan": 1,
    "BranchOnValueGreaterThanEqual": 2,
    "BranchOnValueGreaterThan": 3,
    "BranchOnValueEqual": 4,
    "BranchOnValueNotEqual": 5,
    "LeafNode": 6
};

CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo = class EvaluationInfo {

    static decode(reader, length) {
        const message = new CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "evaluationIndex":
                    message.evaluationIndex = reader.uint64();
                    break;
                case "evaluationValue":
                    message.evaluationValue = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.prototype.evaluationIndex = 0n;
CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.prototype.evaluationValue = 0;

CoreML.Specification.TreeEnsembleClassifier = class TreeEnsembleClassifier {

    get ClassLabels() {
        CoreML.Specification.TreeEnsembleClassifier.ClassLabelsSet = CoreML.Specification.TreeEnsembleClassifier.ClassLabelsSet || new Set(["stringClassLabels", "int64ClassLabels"]);
        return Object.keys(this).find((key) => CoreML.Specification.TreeEnsembleClassifier.ClassLabelsSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.TreeEnsembleClassifier();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.treeEnsemble = CoreML.Specification.TreeEnsembleParameters.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.postEvaluationTransform = reader.int32();
                    break;
                case 100:
                    message.stringClassLabels = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.TreeEnsembleClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "treeEnsemble":
                    message.treeEnsemble = CoreML.Specification.TreeEnsembleParameters.decodeText(reader);
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum(CoreML.Specification.TreeEnsemblePostEvaluationTransform);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TreeEnsembleClassifier.prototype.treeEnsemble = null;
CoreML.Specification.TreeEnsembleClassifier.prototype.postEvaluationTransform = 0;

CoreML.Specification.TreeEnsembleRegressor = class TreeEnsembleRegressor {

    static decode(reader, length) {
        const message = new CoreML.Specification.TreeEnsembleRegressor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.treeEnsemble = CoreML.Specification.TreeEnsembleParameters.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.TreeEnsembleRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "treeEnsemble":
                    message.treeEnsemble = CoreML.Specification.TreeEnsembleParameters.decodeText(reader);
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum(CoreML.Specification.TreeEnsemblePostEvaluationTransform);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.TreeEnsembleRegressor.prototype.treeEnsemble = null;
CoreML.Specification.TreeEnsembleRegressor.prototype.postEvaluationTransform = 0;

CoreML.Specification.ItemSimilarityRecommender = class ItemSimilarityRecommender {

    constructor() {
        this.itemItemSimilarities = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ItemSimilarityRecommender();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.itemItemSimilarities.push(CoreML.Specification.ItemSimilarityRecommender.SimilarItems.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.itemStringIds = CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.itemInt64Ids = CoreML.Specification.Int64Vector.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ItemSimilarityRecommender();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "itemItemSimilarities":
                    message.itemItemSimilarities.push(CoreML.Specification.ItemSimilarityRecommender.SimilarItems.decodeText(reader));
                    break;
                case "itemStringIds":
                    message.itemStringIds = CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "itemInt64Ids":
                    message.itemInt64Ids = CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "itemInputFeatureName":
                    message.itemInputFeatureName = reader.string();
                    break;
                case "numRecommendationsInputFeatureName":
                    message.numRecommendationsInputFeatureName = reader.string();
                    break;
                case "itemRestrictionInputFeatureName":
                    message.itemRestrictionInputFeatureName = reader.string();
                    break;
                case "itemExclusionInputFeatureName":
                    message.itemExclusionInputFeatureName = reader.string();
                    break;
                case "recommendedItemListOutputFeatureName":
                    message.recommendedItemListOutputFeatureName = reader.string();
                    break;
                case "recommendedItemScoreOutputFeatureName":
                    message.recommendedItemScoreOutputFeatureName = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ItemSimilarityRecommender.prototype.itemStringIds = null;
CoreML.Specification.ItemSimilarityRecommender.prototype.itemInt64Ids = null;
CoreML.Specification.ItemSimilarityRecommender.prototype.itemInputFeatureName = "";
CoreML.Specification.ItemSimilarityRecommender.prototype.numRecommendationsInputFeatureName = "";
CoreML.Specification.ItemSimilarityRecommender.prototype.itemRestrictionInputFeatureName = "";
CoreML.Specification.ItemSimilarityRecommender.prototype.itemExclusionInputFeatureName = "";
CoreML.Specification.ItemSimilarityRecommender.prototype.recommendedItemListOutputFeatureName = "";
CoreML.Specification.ItemSimilarityRecommender.prototype.recommendedItemScoreOutputFeatureName = "";

CoreML.Specification.ItemSimilarityRecommender.ConnectedItem = class ConnectedItem {

    static decode(reader, length) {
        const message = new CoreML.Specification.ItemSimilarityRecommender.ConnectedItem();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ItemSimilarityRecommender.ConnectedItem();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "itemId":
                    message.itemId = reader.uint64();
                    break;
                case "similarityScore":
                    message.similarityScore = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.prototype.itemId = 0n;
CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.prototype.similarityScore = 0;

CoreML.Specification.ItemSimilarityRecommender.SimilarItems = class SimilarItems {

    constructor() {
        this.similarItemList = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ItemSimilarityRecommender.SimilarItems();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.itemId = reader.uint64();
                    break;
                case 2:
                    message.similarItemList.push(CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new CoreML.Specification.ItemSimilarityRecommender.SimilarItems();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "itemId":
                    message.itemId = reader.uint64();
                    break;
                case "similarItemList":
                    message.similarItemList.push(CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.decodeText(reader));
                    break;
                case "itemScoreAdjustment":
                    message.itemScoreAdjustment = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.ItemSimilarityRecommender.SimilarItems.prototype.itemId = 0n;
CoreML.Specification.ItemSimilarityRecommender.SimilarItems.prototype.itemScoreAdjustment = 0;

CoreML.Specification.LinkedModel = class LinkedModel {

    get LinkType() {
        CoreML.Specification.LinkedModel.LinkTypeSet = CoreML.Specification.LinkedModel.LinkTypeSet || new Set(["linkedModelFile"]);
        return Object.keys(this).find((key) => CoreML.Specification.LinkedModel.LinkTypeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.LinkedModel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.linkedModelFile = CoreML.Specification.LinkedModelFile.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LinkedModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linkedModelFile":
                    message.linkedModelFile = CoreML.Specification.LinkedModelFile.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LinkedModelFile = class LinkedModelFile {

    static decode(reader, length) {
        const message = new CoreML.Specification.LinkedModelFile();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.linkedModelFileName = CoreML.Specification.StringParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.linkedModelSearchPath = CoreML.Specification.StringParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.LinkedModelFile();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linkedModelFileName":
                    message.linkedModelFileName = CoreML.Specification.StringParameter.decodeText(reader);
                    break;
                case "linkedModelSearchPath":
                    message.linkedModelSearchPath = CoreML.Specification.StringParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

CoreML.Specification.LinkedModelFile.prototype.linkedModelFileName = null;
CoreML.Specification.LinkedModelFile.prototype.linkedModelSearchPath = null;

CoreML.Specification.ClassConfidenceThresholding = class ClassConfidenceThresholding {

    constructor() {
        this.precisionRecallCurves = [];
    }

    static decode(reader, length) {
        const message = new CoreML.Specification.ClassConfidenceThresholding();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 100:
                    message.precisionRecallCurves.push(CoreML.Specification.PrecisionRecallCurve.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new CoreML.Specification.ClassConfidenceThresholding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "precisionRecallCurves":
                    message.precisionRecallCurves.push(CoreML.Specification.PrecisionRecallCurve.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};
