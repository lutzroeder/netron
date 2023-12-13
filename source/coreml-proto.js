
import * as protobuf from './protobuf.js';

const $root = protobuf.get('coreml');

$root.CoreML = {};

$root.CoreML.Specification = {};

$root.CoreML.Specification.Pipeline = class Pipeline {

    constructor() {
        this.models = [];
        this.names = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Pipeline();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Pipeline();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "models":
                    message.models.push($root.CoreML.Specification.Model.decodeText(reader));
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

$root.CoreML.Specification.PipelineClassifier = class PipelineClassifier {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PipelineClassifier();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.PipelineClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pipeline":
                    message.pipeline = $root.CoreML.Specification.Pipeline.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.PipelineRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pipeline":
                    message.pipeline = $root.CoreML.Specification.Pipeline.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.type = $root.CoreML.Specification.FeatureType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.FeatureDescription();
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
                    message.type = $root.CoreML.Specification.FeatureType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Metadata();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ModelDescription();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    message.input.push($root.CoreML.Specification.FeatureDescription.decodeText(reader));
                    break;
                case "output":
                    message.output.push($root.CoreML.Specification.FeatureDescription.decodeText(reader));
                    break;
                case "predictedFeatureName":
                    message.predictedFeatureName = reader.string();
                    break;
                case "predictedProbabilitiesName":
                    message.predictedProbabilitiesName = reader.string();
                    break;
                case "trainingInput":
                    message.trainingInput.push($root.CoreML.Specification.FeatureDescription.decodeText(reader));
                    break;
                case "metadata":
                    message.metadata = $root.CoreML.Specification.Metadata.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SerializedModel();
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

$root.CoreML.Specification.SerializedModel.prototype.identifier = "";
$root.CoreML.Specification.SerializedModel.prototype.model = new Uint8Array([]);

$root.CoreML.Specification.Model = class Model {

    constructor() {
    }

    get Type() {
        $root.CoreML.Specification.Model.TypeSet = $root.CoreML.Specification.Model.TypeSet || new Set([ "pipelineClassifier", "pipelineRegressor", "pipeline", "glmRegressor", "supportVectorRegressor", "treeEnsembleRegressor", "neuralNetworkRegressor", "bayesianProbitRegressor", "glmClassifier", "supportVectorClassifier", "treeEnsembleClassifier", "neuralNetworkClassifier", "kNearestNeighborsClassifier", "neuralNetwork", "itemSimilarityRecommender", "mlProgram", "customModel", "linkedModel", "classConfidenceThresholding", "oneHotEncoder", "imputer", "featureVectorizer", "dictVectorizer", "scaler", "categoricalMapping", "normalizer", "arrayFeatureExtractor", "nonMaximumSuppression", "identity", "textClassifier", "wordTagger", "visionFeaturePrint", "soundAnalysisPreprocessing", "gazetteer", "wordEmbedding", "audioFeaturePrint", "serializedModel"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Model.TypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Model();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                case 502:
                    message.mlProgram = $root.CoreML.Specification.MILSpec.Program.decode(reader, reader.uint32());
                    break;
                case 555:
                    message.customModel = $root.CoreML.Specification.CustomModel.decode(reader, reader.uint32());
                    break;
                case 556:
                    message.linkedModel = $root.CoreML.Specification.LinkedModel.decode(reader, reader.uint32());
                    break;
                case 560:
                    message.classConfidenceThresholding = $root.CoreML.Specification.ClassConfidenceThresholding.decode(reader, reader.uint32());
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
                case 2006:
                    message.audioFeaturePrint = $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Model();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "specificationVersion":
                    message.specificationVersion = reader.int32();
                    break;
                case "description":
                    message.description = $root.CoreML.Specification.ModelDescription.decodeText(reader);
                    break;
                case "isUpdatable":
                    message.isUpdatable = reader.bool();
                    break;
                case "pipelineClassifier":
                    message.pipelineClassifier = $root.CoreML.Specification.PipelineClassifier.decodeText(reader);
                    break;
                case "pipelineRegressor":
                    message.pipelineRegressor = $root.CoreML.Specification.PipelineRegressor.decodeText(reader);
                    break;
                case "pipeline":
                    message.pipeline = $root.CoreML.Specification.Pipeline.decodeText(reader);
                    break;
                case "glmRegressor":
                    message.glmRegressor = $root.CoreML.Specification.GLMRegressor.decodeText(reader);
                    break;
                case "supportVectorRegressor":
                    message.supportVectorRegressor = $root.CoreML.Specification.SupportVectorRegressor.decodeText(reader);
                    break;
                case "treeEnsembleRegressor":
                    message.treeEnsembleRegressor = $root.CoreML.Specification.TreeEnsembleRegressor.decodeText(reader);
                    break;
                case "neuralNetworkRegressor":
                    message.neuralNetworkRegressor = $root.CoreML.Specification.NeuralNetworkRegressor.decodeText(reader);
                    break;
                case "bayesianProbitRegressor":
                    message.bayesianProbitRegressor = $root.CoreML.Specification.BayesianProbitRegressor.decodeText(reader);
                    break;
                case "glmClassifier":
                    message.glmClassifier = $root.CoreML.Specification.GLMClassifier.decodeText(reader);
                    break;
                case "supportVectorClassifier":
                    message.supportVectorClassifier = $root.CoreML.Specification.SupportVectorClassifier.decodeText(reader);
                    break;
                case "treeEnsembleClassifier":
                    message.treeEnsembleClassifier = $root.CoreML.Specification.TreeEnsembleClassifier.decodeText(reader);
                    break;
                case "neuralNetworkClassifier":
                    message.neuralNetworkClassifier = $root.CoreML.Specification.NeuralNetworkClassifier.decodeText(reader);
                    break;
                case "kNearestNeighborsClassifier":
                    message.kNearestNeighborsClassifier = $root.CoreML.Specification.KNearestNeighborsClassifier.decodeText(reader);
                    break;
                case "neuralNetwork":
                    message.neuralNetwork = $root.CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                case "itemSimilarityRecommender":
                    message.itemSimilarityRecommender = $root.CoreML.Specification.ItemSimilarityRecommender.decodeText(reader);
                    break;
                case "mlProgram":
                    message.mlProgram = $root.CoreML.Specification.MILSpec.Program.decodeText(reader);
                    break;
                case "customModel":
                    message.customModel = $root.CoreML.Specification.CustomModel.decodeText(reader);
                    break;
                case "linkedModel":
                    message.linkedModel = $root.CoreML.Specification.LinkedModel.decodeText(reader);
                    break;
                case "classConfidenceThresholding":
                    message.classConfidenceThresholding = $root.CoreML.Specification.ClassConfidenceThresholding.decodeText(reader);
                    break;
                case "oneHotEncoder":
                    message.oneHotEncoder = $root.CoreML.Specification.OneHotEncoder.decodeText(reader);
                    break;
                case "imputer":
                    message.imputer = $root.CoreML.Specification.Imputer.decodeText(reader);
                    break;
                case "featureVectorizer":
                    message.featureVectorizer = $root.CoreML.Specification.FeatureVectorizer.decodeText(reader);
                    break;
                case "dictVectorizer":
                    message.dictVectorizer = $root.CoreML.Specification.DictVectorizer.decodeText(reader);
                    break;
                case "scaler":
                    message.scaler = $root.CoreML.Specification.Scaler.decodeText(reader);
                    break;
                case "categoricalMapping":
                    message.categoricalMapping = $root.CoreML.Specification.CategoricalMapping.decodeText(reader);
                    break;
                case "normalizer":
                    message.normalizer = $root.CoreML.Specification.Normalizer.decodeText(reader);
                    break;
                case "arrayFeatureExtractor":
                    message.arrayFeatureExtractor = $root.CoreML.Specification.ArrayFeatureExtractor.decodeText(reader);
                    break;
                case "nonMaximumSuppression":
                    message.nonMaximumSuppression = $root.CoreML.Specification.NonMaximumSuppression.decodeText(reader);
                    break;
                case "identity":
                    message.identity = $root.CoreML.Specification.Identity.decodeText(reader);
                    break;
                case "textClassifier":
                    message.textClassifier = $root.CoreML.Specification.CoreMLModels.TextClassifier.decodeText(reader);
                    break;
                case "wordTagger":
                    message.wordTagger = $root.CoreML.Specification.CoreMLModels.WordTagger.decodeText(reader);
                    break;
                case "visionFeaturePrint":
                    message.visionFeaturePrint = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.decodeText(reader);
                    break;
                case "soundAnalysisPreprocessing":
                    message.soundAnalysisPreprocessing = $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.decodeText(reader);
                    break;
                case "gazetteer":
                    message.gazetteer = $root.CoreML.Specification.CoreMLModels.Gazetteer.decodeText(reader);
                    break;
                case "wordEmbedding":
                    message.wordEmbedding = $root.CoreML.Specification.CoreMLModels.WordEmbedding.decodeText(reader);
                    break;
                case "audioFeaturePrint":
                    message.audioFeaturePrint = $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.decodeText(reader);
                    break;
                case "serializedModel":
                    message.serializedModel = $root.CoreML.Specification.SerializedModel.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "scene":
                    message.scene = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.decodeText(reader);
                    break;
                case "objects":
                    message.objects = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.enum($root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.prototype.version = 0;

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.SceneVersion = {
    "SCENE_VERSION_INVALID": 0,
    "SCENE_VERSION_1": 1,
    "SCENE_VERSION_2": 2
};

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects = class Objects {

    constructor() {
        this.output = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.enum($root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.ObjectsVersion);
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

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.prototype.version = 0;

$root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Objects.ObjectsVersion = {
    "OBJECTS_VERSION_INVALID": 0,
    "OBJECTS_VERSION_1": 1
};

$root.CoreML.Specification.CoreMLModels.AudioFeaturePrint = class AudioFeaturePrint {

    constructor() {
    }

    get AudioFeaturePrintType() {
        $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.AudioFeaturePrintTypeSet = $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.AudioFeaturePrintTypeSet || new Set([ "sound"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.AudioFeaturePrintTypeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.sound = $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sound":
                    message.sound = $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound = class Sound {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.enum($root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.SoundVersion);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.prototype.version = 0;

$root.CoreML.Specification.CoreMLModels.AudioFeaturePrint.Sound.SoundVersion = {
    "SOUND_VERSION_INVALID": 0,
    "SOUND_VERSION_1": 1
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CoreMLModels.TextClassifier();
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
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.stringTags = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CoreMLModels.WordTagger();
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
                    message.stringTags = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CoreMLModels.Gazetteer();
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
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CoreMLModels.WordEmbedding();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vggish":
                    message.vggish = $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish();
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

$root.CoreML.Specification.StringToInt64Map = class StringToInt64Map {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringToInt64Map();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.StringToInt64Map();
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

$root.CoreML.Specification.Int64ToStringMap = class Int64ToStringMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64ToStringMap();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Int64ToStringMap();
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

$root.CoreML.Specification.StringToDoubleMap = class StringToDoubleMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringToDoubleMap();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.StringToDoubleMap();
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

$root.CoreML.Specification.Int64ToDoubleMap = class Int64ToDoubleMap {

    constructor() {
        this.map = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64ToDoubleMap();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Int64ToDoubleMap();
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

$root.CoreML.Specification.StringVector = class StringVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringVector();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.StringVector();
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

$root.CoreML.Specification.Int64Vector = class Int64Vector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Vector();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Int64Vector();
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

$root.CoreML.Specification.FloatVector = class FloatVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FloatVector();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FloatVector();
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

$root.CoreML.Specification.DoubleVector = class DoubleVector {

    constructor() {
        this.vector = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleVector();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.DoubleVector();
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

$root.CoreML.Specification.Int64Range = class Int64Range {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Range();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Int64Range();
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

$root.CoreML.Specification.Int64Range.prototype.minValue = protobuf.Int64.create(0);
$root.CoreML.Specification.Int64Range.prototype.maxValue = protobuf.Int64.create(0);

$root.CoreML.Specification.Int64Set = class Int64Set {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Set();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Int64Set();
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

$root.CoreML.Specification.DoubleRange = class DoubleRange {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleRange();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.DoubleRange();
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

$root.CoreML.Specification.DoubleRange.prototype.minValue = 0;
$root.CoreML.Specification.DoubleRange.prototype.maxValue = 0;

$root.CoreML.Specification.PrecisionRecallCurve = class PrecisionRecallCurve {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PrecisionRecallCurve();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.precisionValues = $root.CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.precisionConfidenceThresholds = $root.CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.recallValues = $root.CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.recallConfidenceThresholds = $root.CoreML.Specification.FloatVector.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.PrecisionRecallCurve();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "precisionValues":
                    message.precisionValues = $root.CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                case "precisionConfidenceThresholds":
                    message.precisionConfidenceThresholds = $root.CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                case "recallValues":
                    message.recallValues = $root.CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                case "recallConfidenceThresholds":
                    message.recallConfidenceThresholds = $root.CoreML.Specification.FloatVector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.PrecisionRecallCurve.prototype.precisionValues = null;
$root.CoreML.Specification.PrecisionRecallCurve.prototype.precisionConfidenceThresholds = null;
$root.CoreML.Specification.PrecisionRecallCurve.prototype.recallValues = null;
$root.CoreML.Specification.PrecisionRecallCurve.prototype.recallConfidenceThresholds = null;

$root.CoreML.Specification.Int64FeatureType = class Int64FeatureType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64FeatureType();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Int64FeatureType();
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

$root.CoreML.Specification.DoubleFeatureType = class DoubleFeatureType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DoubleFeatureType();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.DoubleFeatureType();
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

$root.CoreML.Specification.StringFeatureType = class StringFeatureType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.StringFeatureType();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.StringFeatureType();
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

$root.CoreML.Specification.SizeRange = class SizeRange {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SizeRange();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SizeRange();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ImageFeatureType();
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
                    message.enumeratedSizes = $root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes.decodeText(reader);
                    break;
                case "imageSizeRange":
                    message.imageSizeRange = $root.CoreML.Specification.ImageFeatureType.ImageSizeRange.decodeText(reader);
                    break;
                case "colorSpace":
                    message.colorSpace = reader.enum($root.CoreML.Specification.ImageFeatureType.ColorSpace);
                    break;
                default:
                    reader.field(tag, message);
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
    "BGR": 30,
    "GRAYSCALE_FLOAT16": 40
};

$root.CoreML.Specification.ImageFeatureType.ImageSize = class ImageSize {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ImageFeatureType.ImageSize();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ImageFeatureType.ImageSize();
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

$root.CoreML.Specification.ImageFeatureType.ImageSize.prototype.width = protobuf.Uint64.create(0);
$root.CoreML.Specification.ImageFeatureType.ImageSize.prototype.height = protobuf.Uint64.create(0);

$root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes = class EnumeratedImageSizes {

    constructor() {
        this.sizes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sizes":
                    message.sizes.push($root.CoreML.Specification.ImageFeatureType.ImageSize.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ImageFeatureType.ImageSizeRange();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "widthRange":
                    message.widthRange = $root.CoreML.Specification.SizeRange.decodeText(reader);
                    break;
                case "heightRange":
                    message.heightRange = $root.CoreML.Specification.SizeRange.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ArrayFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.int64());
                    break;
                case "dataType":
                    message.dataType = reader.enum($root.CoreML.Specification.ArrayFeatureType.ArrayDataType);
                    break;
                case "enumeratedShapes":
                    message.enumeratedShapes = $root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes.decodeText(reader);
                    break;
                case "shapeRange":
                    message.shapeRange = $root.CoreML.Specification.ArrayFeatureType.ShapeRange.decodeText(reader);
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

$root.CoreML.Specification.ArrayFeatureType.prototype.dataType = 0;

$root.CoreML.Specification.ArrayFeatureType.ArrayDataType = {
    "INVALID_ARRAY_DATA_TYPE": 0,
    "FLOAT32": 65568,
    "DOUBLE": 65600,
    "INT32": 131104,
    "FLOAT16": 65552
};

$root.CoreML.Specification.ArrayFeatureType.Shape = class Shape {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.Shape();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ArrayFeatureType.Shape();
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

$root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes = class EnumeratedShapes {

    constructor() {
        this.shapes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shapes":
                    message.shapes.push($root.CoreML.Specification.ArrayFeatureType.Shape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ArrayFeatureType.ShapeRange();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sizeRanges":
                    message.sizeRanges.push($root.CoreML.Specification.SizeRange.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.DictionaryFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "int64KeyType":
                    message.int64KeyType = $root.CoreML.Specification.Int64FeatureType.decodeText(reader);
                    break;
                case "stringKeyType":
                    message.stringKeyType = $root.CoreML.Specification.StringFeatureType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SequenceFeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "int64Type":
                    message.int64Type = $root.CoreML.Specification.Int64FeatureType.decodeText(reader);
                    break;
                case "stringType":
                    message.stringType = $root.CoreML.Specification.StringFeatureType.decodeText(reader);
                    break;
                case "sizeRange":
                    message.sizeRange = $root.CoreML.Specification.SizeRange.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.FeatureType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "int64Type":
                    message.int64Type = $root.CoreML.Specification.Int64FeatureType.decodeText(reader);
                    break;
                case "doubleType":
                    message.doubleType = $root.CoreML.Specification.DoubleFeatureType.decodeText(reader);
                    break;
                case "stringType":
                    message.stringType = $root.CoreML.Specification.StringFeatureType.decodeText(reader);
                    break;
                case "imageType":
                    message.imageType = $root.CoreML.Specification.ImageFeatureType.decodeText(reader);
                    break;
                case "multiArrayType":
                    message.multiArrayType = $root.CoreML.Specification.ArrayFeatureType.decodeText(reader);
                    break;
                case "dictionaryType":
                    message.dictionaryType = $root.CoreML.Specification.DictionaryFeatureType.decodeText(reader);
                    break;
                case "sequenceType":
                    message.sequenceType = $root.CoreML.Specification.SequenceFeatureType.decodeText(reader);
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

$root.CoreML.Specification.FeatureType.prototype.isOptional = false;

$root.CoreML.Specification.ArrayFeatureExtractor = class ArrayFeatureExtractor {

    constructor() {
        this.extractIndex = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArrayFeatureExtractor();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ArrayFeatureExtractor();
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

$root.CoreML.Specification.BayesianProbitRegressor = class BayesianProbitRegressor {

    constructor() {
        this.features = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numberOfFeatures":
                    message.numberOfFeatures = reader.uint32();
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.BayesianProbitRegressor.Gaussian.decodeText(reader);
                    break;
                case "features":
                    message.features.push($root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight.decodeText(reader));
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.Gaussian();
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

$root.CoreML.Specification.BayesianProbitRegressor.Gaussian.prototype.mean = 0;
$root.CoreML.Specification.BayesianProbitRegressor.Gaussian.prototype.precision = 0;

$root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight = class FeatureValueWeight {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "featureValue":
                    message.featureValue = reader.uint32();
                    break;
                case "featureWeight":
                    message.featureWeight = $root.CoreML.Specification.BayesianProbitRegressor.Gaussian.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "featureId":
                    message.featureId = reader.uint32();
                    break;
                case "weights":
                    message.weights.push($root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CategoricalMapping();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "stringToInt64Map":
                    message.stringToInt64Map = $root.CoreML.Specification.StringToInt64Map.decodeText(reader);
                    break;
                case "int64ToStringMap":
                    message.int64ToStringMap = $root.CoreML.Specification.Int64ToStringMap.decodeText(reader);
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

$root.CoreML.Specification.CustomModel = class CustomModel {

    constructor() {
        this.parameters = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CustomModel();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CustomModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "className":
                    message.className = reader.string();
                    break;
                case "parameters":
                    reader.entry(message.parameters, () => reader.string(), () => $root.CoreML.Specification.CustomModel.CustomModelParamValue.decodeText(reader));
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CustomModel.CustomModelParamValue();
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

$root.CoreML.Specification.DictVectorizer = class DictVectorizer {

    constructor() {
    }

    get Map() {
        $root.CoreML.Specification.DictVectorizer.MapSet = $root.CoreML.Specification.DictVectorizer.MapSet || new Set([ "stringToIndex", "int64ToIndex"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.DictVectorizer.MapSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DictVectorizer();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.DictVectorizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "stringToIndex":
                    message.stringToIndex = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ToIndex":
                    message.int64ToIndex = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.FeatureVectorizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputList":
                    message.inputList.push($root.CoreML.Specification.FeatureVectorizer.InputColumn.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FeatureVectorizer.InputColumn();
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

$root.CoreML.Specification.FeatureVectorizer.InputColumn.prototype.inputColumn = "";
$root.CoreML.Specification.FeatureVectorizer.InputColumn.prototype.inputDimensions = protobuf.Uint64.create(0);

$root.CoreML.Specification.GLMRegressor = class GLMRegressor {

    constructor() {
        this.weights = [];
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GLMRegressor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.GLMRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "weights":
                    message.weights.push($root.CoreML.Specification.GLMRegressor.DoubleArray.decodeText(reader));
                    break;
                case "offset":
                    reader.array(message.offset, () => reader.double());
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum($root.CoreML.Specification.GLMRegressor.PostEvaluationTransform);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GLMRegressor.DoubleArray();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.GLMClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "weights":
                    message.weights.push($root.CoreML.Specification.GLMClassifier.DoubleArray.decodeText(reader));
                    break;
                case "offset":
                    reader.array(message.offset, () => reader.double());
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum($root.CoreML.Specification.GLMClassifier.PostEvaluationTransform);
                    break;
                case "classEncoding":
                    message.classEncoding = reader.enum($root.CoreML.Specification.GLMClassifier.ClassEncoding);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GLMClassifier.DoubleArray();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.KNearestNeighborsClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nearestNeighborsIndex":
                    message.nearestNeighborsIndex = $root.CoreML.Specification.NearestNeighborsIndex.decodeText(reader);
                    break;
                case "numberOfNeighbors":
                    message.numberOfNeighbors = $root.CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "defaultStringLabel":
                    message.defaultStringLabel = reader.string();
                    break;
                case "defaultInt64Label":
                    message.defaultInt64Label = reader.int64();
                    break;
                case "uniformWeighting":
                    message.uniformWeighting = $root.CoreML.Specification.UniformWeighting.decodeText(reader);
                    break;
                case "inverseDistanceWeighting":
                    message.inverseDistanceWeighting = $root.CoreML.Specification.InverseDistanceWeighting.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NearestNeighborsIndex();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numberOfDimensions":
                    message.numberOfDimensions = reader.int32();
                    break;
                case "floatSamples":
                    message.floatSamples.push($root.CoreML.Specification.FloatVector.decodeText(reader));
                    break;
                case "linearIndex":
                    message.linearIndex = $root.CoreML.Specification.LinearIndex.decodeText(reader);
                    break;
                case "singleKdTreeIndex":
                    message.singleKdTreeIndex = $root.CoreML.Specification.SingleKdTreeIndex.decodeText(reader);
                    break;
                case "squaredEuclideanDistance":
                    message.squaredEuclideanDistance = $root.CoreML.Specification.SquaredEuclideanDistance.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.UniformWeighting();
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

$root.CoreML.Specification.InverseDistanceWeighting = class InverseDistanceWeighting {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.InverseDistanceWeighting();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.InverseDistanceWeighting();
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

$root.CoreML.Specification.LinearIndex = class LinearIndex {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinearIndex();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LinearIndex();
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

$root.CoreML.Specification.SingleKdTreeIndex = class SingleKdTreeIndex {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SingleKdTreeIndex();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SingleKdTreeIndex();
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

$root.CoreML.Specification.SingleKdTreeIndex.prototype.leafSize = 0;

$root.CoreML.Specification.SquaredEuclideanDistance = class SquaredEuclideanDistance {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SquaredEuclideanDistance();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SquaredEuclideanDistance();
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

$root.CoreML.Specification.Int64Parameter = class Int64Parameter {

    constructor() {
    }

    get AllowedValues() {
        $root.CoreML.Specification.Int64Parameter.AllowedValuesSet = $root.CoreML.Specification.Int64Parameter.AllowedValuesSet || new Set([ "range", "set"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.Int64Parameter.AllowedValuesSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Int64Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Int64Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "defaultValue":
                    message.defaultValue = reader.int64();
                    break;
                case "range":
                    message.range = $root.CoreML.Specification.Int64Range.decodeText(reader);
                    break;
                case "set":
                    message.set = $root.CoreML.Specification.Int64Set.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.DoubleParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "defaultValue":
                    message.defaultValue = reader.double();
                    break;
                case "range":
                    message.range = $root.CoreML.Specification.DoubleRange.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.StringParameter();
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

$root.CoreML.Specification.StringParameter.prototype.defaultValue = "";

$root.CoreML.Specification.BoolParameter = class BoolParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BoolParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BoolParameter();
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

$root.CoreML.Specification.BoolParameter.prototype.defaultValue = false;

$root.CoreML.Specification.Identity = class Identity {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Identity();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Identity();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Imputer();
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
                    message.imputedDoubleArray = $root.CoreML.Specification.DoubleVector.decodeText(reader);
                    break;
                case "imputedInt64Array":
                    message.imputedInt64Array = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "imputedStringDictionary":
                    message.imputedStringDictionary = $root.CoreML.Specification.StringToDoubleMap.decodeText(reader);
                    break;
                case "imputedInt64Dictionary":
                    message.imputedInt64Dictionary = $root.CoreML.Specification.Int64ToDoubleMap.decodeText(reader);
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

$root.CoreML.Specification.MILSpec = {};

$root.CoreML.Specification.MILSpec.Program = class Program {

    constructor() {
        this.functions = {};
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Program();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int64();
                    break;
                case 2:
                    reader.entry(message.functions, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Function.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.docString = reader.string();
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Program();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.int64();
                    break;
                case "functions":
                    reader.entry(message.functions, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Function.decodeText(reader));
                    break;
                case "docString":
                    message.docString = reader.string();
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Program.prototype.version = protobuf.Int64.create(0);
$root.CoreML.Specification.MILSpec.Program.prototype.docString = "";

$root.CoreML.Specification.MILSpec.Function = class Function {

    constructor() {
        this.inputs = [];
        this.block_specializations = {};
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Function();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputs.push($root.CoreML.Specification.MILSpec.NamedValueType.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.opset = reader.string();
                    break;
                case 3:
                    reader.entry(message.block_specializations, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Block.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Function();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputs":
                    message.inputs.push($root.CoreML.Specification.MILSpec.NamedValueType.decodeText(reader));
                    break;
                case "opset":
                    message.opset = reader.string();
                    break;
                case "block_specializations":
                    reader.entry(message.block_specializations, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Block.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Function.prototype.opset = "";

$root.CoreML.Specification.MILSpec.Block = class Block {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.operations = [];
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Block();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inputs.push($root.CoreML.Specification.MILSpec.NamedValueType.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.outputs.push(reader.string());
                    break;
                case 3:
                    message.operations.push($root.CoreML.Specification.MILSpec.Operation.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Block();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputs":
                    message.inputs.push($root.CoreML.Specification.MILSpec.NamedValueType.decodeText(reader));
                    break;
                case "outputs":
                    reader.array(message.outputs, () => reader.string());
                    break;
                case "operations":
                    message.operations.push($root.CoreML.Specification.MILSpec.Operation.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Argument = class Argument {

    constructor() {
        this["arguments"] = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Argument();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message["arguments"].push($root.CoreML.Specification.MILSpec.Argument.Binding.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Argument();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "arguments":
                    message["arguments"].push($root.CoreML.Specification.MILSpec.Argument.Binding.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Argument.Binding = class Binding {

    constructor() {
    }

    get binding() {
        $root.CoreML.Specification.MILSpec.Argument.Binding.bindingSet = $root.CoreML.Specification.MILSpec.Argument.Binding.bindingSet || new Set([ "name", "value"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.MILSpec.Argument.Binding.bindingSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Argument.Binding();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.value = $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Argument.Binding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "value":
                    message.value = $root.CoreML.Specification.MILSpec.Value.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Operation = class Operation {

    constructor() {
        this.inputs = {};
        this.outputs = [];
        this.blocks = [];
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Operation();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.string();
                    break;
                case 2:
                    reader.entry(message.inputs, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Argument.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.outputs.push($root.CoreML.Specification.MILSpec.NamedValueType.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.blocks.push($root.CoreML.Specification.MILSpec.Block.decode(reader, reader.uint32()));
                    break;
                case 5:
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Operation();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "inputs":
                    reader.entry(message.inputs, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Argument.decodeText(reader));
                    break;
                case "outputs":
                    message.outputs.push($root.CoreML.Specification.MILSpec.NamedValueType.decodeText(reader));
                    break;
                case "blocks":
                    message.blocks.push($root.CoreML.Specification.MILSpec.Block.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Operation.prototype.type = "";

$root.CoreML.Specification.MILSpec.NamedValueType = class NamedValueType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.NamedValueType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = $root.CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.NamedValueType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = $root.CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.NamedValueType.prototype.name = "";
$root.CoreML.Specification.MILSpec.NamedValueType.prototype.type = null;

$root.CoreML.Specification.MILSpec.ValueType = class ValueType {

    constructor() {
    }

    get type() {
        $root.CoreML.Specification.MILSpec.ValueType.typeSet = $root.CoreML.Specification.MILSpec.ValueType.typeSet || new Set([ "tensorType", "listType", "tupleType", "dictionaryType"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.MILSpec.ValueType.typeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.ValueType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensorType = $root.CoreML.Specification.MILSpec.TensorType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.listType = $root.CoreML.Specification.MILSpec.ListType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.tupleType = $root.CoreML.Specification.MILSpec.TupleType.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dictionaryType = $root.CoreML.Specification.MILSpec.DictionaryType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.ValueType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensorType":
                    message.tensorType = $root.CoreML.Specification.MILSpec.TensorType.decodeText(reader);
                    break;
                case "listType":
                    message.listType = $root.CoreML.Specification.MILSpec.ListType.decodeText(reader);
                    break;
                case "tupleType":
                    message.tupleType = $root.CoreML.Specification.MILSpec.TupleType.decodeText(reader);
                    break;
                case "dictionaryType":
                    message.dictionaryType = $root.CoreML.Specification.MILSpec.DictionaryType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.DataType = {
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

$root.CoreML.Specification.MILSpec.TensorType = class TensorType {

    constructor() {
        this.dimensions = [];
        this.attributes = {};
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorType();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.dimensions.push($root.CoreML.Specification.MILSpec.Dimension.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.TensorType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dataType":
                    message.dataType = reader.enum($root.CoreML.Specification.MILSpec.DataType);
                    break;
                case "rank":
                    message.rank = reader.int64();
                    break;
                case "dimensions":
                    message.dimensions.push($root.CoreML.Specification.MILSpec.Dimension.decodeText(reader));
                    break;
                case "attributes":
                    reader.entry(message.attributes, () => reader.string(), () => $root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.TensorType.prototype.dataType = 0;
$root.CoreML.Specification.MILSpec.TensorType.prototype.rank = protobuf.Int64.create(0);

$root.CoreML.Specification.MILSpec.TupleType = class TupleType {

    constructor() {
        this.types = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TupleType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.types.push($root.CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.TupleType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "types":
                    message.types.push($root.CoreML.Specification.MILSpec.ValueType.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.ListType = class ListType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.ListType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = $root.CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.length = $root.CoreML.Specification.MILSpec.Dimension.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.ListType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = $root.CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                case "length":
                    message.length = $root.CoreML.Specification.MILSpec.Dimension.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.ListType.prototype.type = null;
$root.CoreML.Specification.MILSpec.ListType.prototype.length = null;

$root.CoreML.Specification.MILSpec.DictionaryType = class DictionaryType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.DictionaryType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.keyType = $root.CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.valueType = $root.CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.DictionaryType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "keyType":
                    message.keyType = $root.CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                case "valueType":
                    message.valueType = $root.CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.DictionaryType.prototype.keyType = null;
$root.CoreML.Specification.MILSpec.DictionaryType.prototype.valueType = null;

$root.CoreML.Specification.MILSpec.Dimension = class Dimension {

    constructor() {
    }

    get dimension() {
        $root.CoreML.Specification.MILSpec.Dimension.dimensionSet = $root.CoreML.Specification.MILSpec.Dimension.dimensionSet || new Set([ "constant", "unknown"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.MILSpec.Dimension.dimensionSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Dimension();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.constant = $root.CoreML.Specification.MILSpec.Dimension.ConstantDimension.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.unknown = $root.CoreML.Specification.MILSpec.Dimension.UnknownDimension.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Dimension();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "constant":
                    message.constant = $root.CoreML.Specification.MILSpec.Dimension.ConstantDimension.decodeText(reader);
                    break;
                case "unknown":
                    message.unknown = $root.CoreML.Specification.MILSpec.Dimension.UnknownDimension.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Dimension.ConstantDimension = class ConstantDimension {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Dimension.ConstantDimension();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.Dimension.ConstantDimension();
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

$root.CoreML.Specification.MILSpec.Dimension.ConstantDimension.prototype.size = protobuf.Uint64.create(0);

$root.CoreML.Specification.MILSpec.Dimension.UnknownDimension = class UnknownDimension {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Dimension.UnknownDimension();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.Dimension.UnknownDimension();
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

$root.CoreML.Specification.MILSpec.Dimension.UnknownDimension.prototype.variadic = false;

$root.CoreML.Specification.MILSpec.Value = class Value {

    constructor() {
    }

    get value() {
        $root.CoreML.Specification.MILSpec.Value.valueSet = $root.CoreML.Specification.MILSpec.Value.valueSet || new Set([ "immediateValue", "blobFileValue"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.MILSpec.Value.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Value();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.docString = reader.string();
                    break;
                case 2:
                    message.type = $root.CoreML.Specification.MILSpec.ValueType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.immediateValue = $root.CoreML.Specification.MILSpec.Value.ImmediateValue.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.blobFileValue = $root.CoreML.Specification.MILSpec.Value.BlobFileValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Value();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "docString":
                    message.docString = reader.string();
                    break;
                case "type":
                    message.type = $root.CoreML.Specification.MILSpec.ValueType.decodeText(reader);
                    break;
                case "immediateValue":
                    message.immediateValue = $root.CoreML.Specification.MILSpec.Value.ImmediateValue.decodeText(reader);
                    break;
                case "blobFileValue":
                    message.blobFileValue = $root.CoreML.Specification.MILSpec.Value.BlobFileValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Value.prototype.docString = "";
$root.CoreML.Specification.MILSpec.Value.prototype.type = null;

$root.CoreML.Specification.MILSpec.Value.ImmediateValue = class ImmediateValue {

    constructor() {
    }

    get value() {
        $root.CoreML.Specification.MILSpec.Value.ImmediateValue.valueSet = $root.CoreML.Specification.MILSpec.Value.ImmediateValue.valueSet || new Set([ "tensor", "tuple", "list", "dictionary"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.MILSpec.Value.ImmediateValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Value.ImmediateValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor = $root.CoreML.Specification.MILSpec.TensorValue.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.tuple = $root.CoreML.Specification.MILSpec.TupleValue.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.list = $root.CoreML.Specification.MILSpec.ListValue.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dictionary = $root.CoreML.Specification.MILSpec.DictionaryValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.Value.ImmediateValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor = $root.CoreML.Specification.MILSpec.TensorValue.decodeText(reader);
                    break;
                case "tuple":
                    message.tuple = $root.CoreML.Specification.MILSpec.TupleValue.decodeText(reader);
                    break;
                case "list":
                    message.list = $root.CoreML.Specification.MILSpec.ListValue.decodeText(reader);
                    break;
                case "dictionary":
                    message.dictionary = $root.CoreML.Specification.MILSpec.DictionaryValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.Value.BlobFileValue = class BlobFileValue {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.Value.BlobFileValue();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.Value.BlobFileValue();
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

$root.CoreML.Specification.MILSpec.Value.BlobFileValue.prototype.fileName = "";
$root.CoreML.Specification.MILSpec.Value.BlobFileValue.prototype.offset = protobuf.Uint64.create(0);

$root.CoreML.Specification.MILSpec.TensorValue = class TensorValue {

    constructor() {
    }

    get value() {
        $root.CoreML.Specification.MILSpec.TensorValue.valueSet = $root.CoreML.Specification.MILSpec.TensorValue.valueSet || new Set([ "floats", "ints", "bools", "strings", "longInts", "doubles", "bytes"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.MILSpec.TensorValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.floats = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedFloats.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.ints = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedInts.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.bools = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBools.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.strings = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedStrings.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.longInts = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.doubles = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.bytes = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBytes.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "floats":
                    message.floats = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedFloats.decodeText(reader);
                    break;
                case "ints":
                    message.ints = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedInts.decodeText(reader);
                    break;
                case "bools":
                    message.bools = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBools.decodeText(reader);
                    break;
                case "strings":
                    message.strings = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedStrings.decodeText(reader);
                    break;
                case "longInts":
                    message.longInts = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts.decodeText(reader);
                    break;
                case "doubles":
                    message.doubles = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles.decodeText(reader);
                    break;
                case "bytes":
                    message.bytes = $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBytes.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedFloats = class RepeatedFloats {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedFloats();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedFloats();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles = class RepeatedDoubles {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedDoubles();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedInts = class RepeatedInts {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedInts();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedInts();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts = class RepeatedLongInts {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedLongInts();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedBools = class RepeatedBools {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBools();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBools();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedStrings = class RepeatedStrings {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedStrings();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedStrings();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedBytes = class RepeatedBytes {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBytes();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MILSpec.TensorValue.RepeatedBytes();
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

$root.CoreML.Specification.MILSpec.TensorValue.RepeatedBytes.prototype.values = new Uint8Array([]);

$root.CoreML.Specification.MILSpec.TupleValue = class TupleValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.TupleValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push($root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.TupleValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push($root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.ListValue = class ListValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.ListValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push($root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push($root.CoreML.Specification.MILSpec.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.DictionaryValue = class DictionaryValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.DictionaryValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push($root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.DictionaryValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push($root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair = class KeyValuePair {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.value = $root.CoreML.Specification.MILSpec.Value.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = $root.CoreML.Specification.MILSpec.Value.decodeText(reader);
                    break;
                case "value":
                    message.value = $root.CoreML.Specification.MILSpec.Value.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.prototype.key = null;
$root.CoreML.Specification.MILSpec.DictionaryValue.KeyValuePair.prototype.value = null;

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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NeuralNetwork();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layers":
                    message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decodeText(reader));
                    break;
                case "preprocessing":
                    message.preprocessing.push($root.CoreML.Specification.NeuralNetworkPreprocessing.decodeText(reader));
                    break;
                case "arrayInputShapeMapping":
                    message.arrayInputShapeMapping = reader.enum($root.CoreML.Specification.NeuralNetworkMultiArrayShapeMapping);
                    break;
                case "imageInputShapeMapping":
                    message.imageInputShapeMapping = reader.enum($root.CoreML.Specification.NeuralNetworkImageShapeMapping);
                    break;
                case "updateParams":
                    message.updateParams = $root.CoreML.Specification.NetworkUpdateParameters.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.NeuralNetworkImageScaler();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.NeuralNetworkMeanImage();
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

$root.CoreML.Specification.NeuralNetworkPreprocessing = class NeuralNetworkPreprocessing {

    constructor() {
    }

    get preprocessor() {
        $root.CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet = $root.CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet || new Set([ "scaler", "meanImage"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.NeuralNetworkPreprocessing.preprocessorSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NeuralNetworkPreprocessing();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NeuralNetworkPreprocessing();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "featureName":
                    message.featureName = reader.string();
                    break;
                case "scaler":
                    message.scaler = $root.CoreML.Specification.NeuralNetworkImageScaler.decodeText(reader);
                    break;
                case "meanImage":
                    message.meanImage = $root.CoreML.Specification.NeuralNetworkMeanImage.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationReLU();
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

$root.CoreML.Specification.ActivationLeakyReLU = class ActivationLeakyReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationLeakyReLU();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationLeakyReLU();
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

$root.CoreML.Specification.ActivationLeakyReLU.prototype.alpha = 0;

$root.CoreML.Specification.ActivationTanh = class ActivationTanh {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationTanh();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationTanh();
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

$root.CoreML.Specification.ActivationScaledTanh = class ActivationScaledTanh {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationScaledTanh();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationScaledTanh();
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

$root.CoreML.Specification.ActivationScaledTanh.prototype.alpha = 0;
$root.CoreML.Specification.ActivationScaledTanh.prototype.beta = 0;

$root.CoreML.Specification.ActivationSigmoid = class ActivationSigmoid {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSigmoid();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationSigmoid();
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

$root.CoreML.Specification.ActivationLinear = class ActivationLinear {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationLinear();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationLinear();
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

$root.CoreML.Specification.ActivationLinear.prototype.alpha = 0;
$root.CoreML.Specification.ActivationLinear.prototype.beta = 0;

$root.CoreML.Specification.ActivationSigmoidHard = class ActivationSigmoidHard {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSigmoidHard();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationSigmoidHard();
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

$root.CoreML.Specification.ActivationSigmoidHard.prototype.alpha = 0;
$root.CoreML.Specification.ActivationSigmoidHard.prototype.beta = 0;

$root.CoreML.Specification.ActivationPReLU = class ActivationPReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationPReLU();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ActivationPReLU();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationELU();
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

$root.CoreML.Specification.ActivationELU.prototype.alpha = 0;

$root.CoreML.Specification.ActivationThresholdedReLU = class ActivationThresholdedReLU {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationThresholdedReLU();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationThresholdedReLU();
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

$root.CoreML.Specification.ActivationThresholdedReLU.prototype.alpha = 0;

$root.CoreML.Specification.ActivationSoftsign = class ActivationSoftsign {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSoftsign();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationSoftsign();
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

$root.CoreML.Specification.ActivationSoftplus = class ActivationSoftplus {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationSoftplus();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ActivationSoftplus();
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

$root.CoreML.Specification.ActivationParametricSoftplus = class ActivationParametricSoftplus {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ActivationParametricSoftplus();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ActivationParametricSoftplus();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "beta":
                    message.beta = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ActivationParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linear":
                    message.linear = $root.CoreML.Specification.ActivationLinear.decodeText(reader);
                    break;
                case "ReLU":
                    message.ReLU = $root.CoreML.Specification.ActivationReLU.decodeText(reader);
                    break;
                case "leakyReLU":
                    message.leakyReLU = $root.CoreML.Specification.ActivationLeakyReLU.decodeText(reader);
                    break;
                case "thresholdedReLU":
                    message.thresholdedReLU = $root.CoreML.Specification.ActivationThresholdedReLU.decodeText(reader);
                    break;
                case "PReLU":
                    message.PReLU = $root.CoreML.Specification.ActivationPReLU.decodeText(reader);
                    break;
                case "tanh":
                    message.tanh = $root.CoreML.Specification.ActivationTanh.decodeText(reader);
                    break;
                case "scaledTanh":
                    message.scaledTanh = $root.CoreML.Specification.ActivationScaledTanh.decodeText(reader);
                    break;
                case "sigmoid":
                    message.sigmoid = $root.CoreML.Specification.ActivationSigmoid.decodeText(reader);
                    break;
                case "sigmoidHard":
                    message.sigmoidHard = $root.CoreML.Specification.ActivationSigmoidHard.decodeText(reader);
                    break;
                case "ELU":
                    message.ELU = $root.CoreML.Specification.ActivationELU.decodeText(reader);
                    break;
                case "softsign":
                    message.softsign = $root.CoreML.Specification.ActivationSoftsign.decodeText(reader);
                    break;
                case "softplus":
                    message.softplus = $root.CoreML.Specification.ActivationSoftplus.decodeText(reader);
                    break;
                case "parametricSoftplus":
                    message.parametricSoftplus = $root.CoreML.Specification.ActivationParametricSoftplus.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Tensor();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NeuralNetworkLayer();
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
                    message.inputTensor.push($root.CoreML.Specification.Tensor.decodeText(reader));
                    break;
                case "outputTensor":
                    message.outputTensor.push($root.CoreML.Specification.Tensor.decodeText(reader));
                    break;
                case "isUpdatable":
                    message.isUpdatable = reader.bool();
                    break;
                case "convolution":
                    message.convolution = $root.CoreML.Specification.ConvolutionLayerParams.decodeText(reader);
                    break;
                case "pooling":
                    message.pooling = $root.CoreML.Specification.PoolingLayerParams.decodeText(reader);
                    break;
                case "activation":
                    message.activation = $root.CoreML.Specification.ActivationParams.decodeText(reader);
                    break;
                case "innerProduct":
                    message.innerProduct = $root.CoreML.Specification.InnerProductLayerParams.decodeText(reader);
                    break;
                case "embedding":
                    message.embedding = $root.CoreML.Specification.EmbeddingLayerParams.decodeText(reader);
                    break;
                case "batchnorm":
                    message.batchnorm = $root.CoreML.Specification.BatchnormLayerParams.decodeText(reader);
                    break;
                case "mvn":
                    message.mvn = $root.CoreML.Specification.MeanVarianceNormalizeLayerParams.decodeText(reader);
                    break;
                case "l2normalize":
                    message.l2normalize = $root.CoreML.Specification.L2NormalizeLayerParams.decodeText(reader);
                    break;
                case "softmax":
                    message.softmax = $root.CoreML.Specification.SoftmaxLayerParams.decodeText(reader);
                    break;
                case "lrn":
                    message.lrn = $root.CoreML.Specification.LRNLayerParams.decodeText(reader);
                    break;
                case "crop":
                    message.crop = $root.CoreML.Specification.CropLayerParams.decodeText(reader);
                    break;
                case "padding":
                    message.padding = $root.CoreML.Specification.PaddingLayerParams.decodeText(reader);
                    break;
                case "upsample":
                    message.upsample = $root.CoreML.Specification.UpsampleLayerParams.decodeText(reader);
                    break;
                case "resizeBilinear":
                    message.resizeBilinear = $root.CoreML.Specification.ResizeBilinearLayerParams.decodeText(reader);
                    break;
                case "cropResize":
                    message.cropResize = $root.CoreML.Specification.CropResizeLayerParams.decodeText(reader);
                    break;
                case "unary":
                    message.unary = $root.CoreML.Specification.UnaryFunctionLayerParams.decodeText(reader);
                    break;
                case "add":
                    message.add = $root.CoreML.Specification.AddLayerParams.decodeText(reader);
                    break;
                case "multiply":
                    message.multiply = $root.CoreML.Specification.MultiplyLayerParams.decodeText(reader);
                    break;
                case "average":
                    message.average = $root.CoreML.Specification.AverageLayerParams.decodeText(reader);
                    break;
                case "scale":
                    message.scale = $root.CoreML.Specification.ScaleLayerParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.BiasLayerParams.decodeText(reader);
                    break;
                case "max":
                    message.max = $root.CoreML.Specification.MaxLayerParams.decodeText(reader);
                    break;
                case "min":
                    message.min = $root.CoreML.Specification.MinLayerParams.decodeText(reader);
                    break;
                case "dot":
                    message.dot = $root.CoreML.Specification.DotProductLayerParams.decodeText(reader);
                    break;
                case "reduce":
                    message.reduce = $root.CoreML.Specification.ReduceLayerParams.decodeText(reader);
                    break;
                case "loadConstant":
                    message.loadConstant = $root.CoreML.Specification.LoadConstantLayerParams.decodeText(reader);
                    break;
                case "reshape":
                    message.reshape = $root.CoreML.Specification.ReshapeLayerParams.decodeText(reader);
                    break;
                case "flatten":
                    message.flatten = $root.CoreML.Specification.FlattenLayerParams.decodeText(reader);
                    break;
                case "permute":
                    message.permute = $root.CoreML.Specification.PermuteLayerParams.decodeText(reader);
                    break;
                case "concat":
                    message.concat = $root.CoreML.Specification.ConcatLayerParams.decodeText(reader);
                    break;
                case "split":
                    message.split = $root.CoreML.Specification.SplitLayerParams.decodeText(reader);
                    break;
                case "sequenceRepeat":
                    message.sequenceRepeat = $root.CoreML.Specification.SequenceRepeatLayerParams.decodeText(reader);
                    break;
                case "reorganizeData":
                    message.reorganizeData = $root.CoreML.Specification.ReorganizeDataLayerParams.decodeText(reader);
                    break;
                case "slice":
                    message.slice = $root.CoreML.Specification.SliceLayerParams.decodeText(reader);
                    break;
                case "simpleRecurrent":
                    message.simpleRecurrent = $root.CoreML.Specification.SimpleRecurrentLayerParams.decodeText(reader);
                    break;
                case "gru":
                    message.gru = $root.CoreML.Specification.GRULayerParams.decodeText(reader);
                    break;
                case "uniDirectionalLSTM":
                    message.uniDirectionalLSTM = $root.CoreML.Specification.UniDirectionalLSTMLayerParams.decodeText(reader);
                    break;
                case "biDirectionalLSTM":
                    message.biDirectionalLSTM = $root.CoreML.Specification.BiDirectionalLSTMLayerParams.decodeText(reader);
                    break;
                case "custom":
                    message.custom = $root.CoreML.Specification.CustomLayerParams.decodeText(reader);
                    break;
                case "copy":
                    message.copy = $root.CoreML.Specification.CopyLayerParams.decodeText(reader);
                    break;
                case "branch":
                    message.branch = $root.CoreML.Specification.BranchLayerParams.decodeText(reader);
                    break;
                case "loop":
                    message.loop = $root.CoreML.Specification.LoopLayerParams.decodeText(reader);
                    break;
                case "loopBreak":
                    message.loopBreak = $root.CoreML.Specification.LoopBreakLayerParams.decodeText(reader);
                    break;
                case "loopContinue":
                    message.loopContinue = $root.CoreML.Specification.LoopContinueLayerParams.decodeText(reader);
                    break;
                case "rangeStatic":
                    message.rangeStatic = $root.CoreML.Specification.RangeStaticLayerParams.decodeText(reader);
                    break;
                case "rangeDynamic":
                    message.rangeDynamic = $root.CoreML.Specification.RangeDynamicLayerParams.decodeText(reader);
                    break;
                case "clip":
                    message.clip = $root.CoreML.Specification.ClipLayerParams.decodeText(reader);
                    break;
                case "ceil":
                    message.ceil = $root.CoreML.Specification.CeilLayerParams.decodeText(reader);
                    break;
                case "floor":
                    message.floor = $root.CoreML.Specification.FloorLayerParams.decodeText(reader);
                    break;
                case "sign":
                    message.sign = $root.CoreML.Specification.SignLayerParams.decodeText(reader);
                    break;
                case "round":
                    message.round = $root.CoreML.Specification.RoundLayerParams.decodeText(reader);
                    break;
                case "exp2":
                    message.exp2 = $root.CoreML.Specification.Exp2LayerParams.decodeText(reader);
                    break;
                case "sin":
                    message.sin = $root.CoreML.Specification.SinLayerParams.decodeText(reader);
                    break;
                case "cos":
                    message.cos = $root.CoreML.Specification.CosLayerParams.decodeText(reader);
                    break;
                case "tan":
                    message.tan = $root.CoreML.Specification.TanLayerParams.decodeText(reader);
                    break;
                case "asin":
                    message.asin = $root.CoreML.Specification.AsinLayerParams.decodeText(reader);
                    break;
                case "acos":
                    message.acos = $root.CoreML.Specification.AcosLayerParams.decodeText(reader);
                    break;
                case "atan":
                    message.atan = $root.CoreML.Specification.AtanLayerParams.decodeText(reader);
                    break;
                case "sinh":
                    message.sinh = $root.CoreML.Specification.SinhLayerParams.decodeText(reader);
                    break;
                case "cosh":
                    message.cosh = $root.CoreML.Specification.CoshLayerParams.decodeText(reader);
                    break;
                case "tanh":
                    message.tanh = $root.CoreML.Specification.TanhLayerParams.decodeText(reader);
                    break;
                case "asinh":
                    message.asinh = $root.CoreML.Specification.AsinhLayerParams.decodeText(reader);
                    break;
                case "acosh":
                    message.acosh = $root.CoreML.Specification.AcoshLayerParams.decodeText(reader);
                    break;
                case "atanh":
                    message.atanh = $root.CoreML.Specification.AtanhLayerParams.decodeText(reader);
                    break;
                case "erf":
                    message.erf = $root.CoreML.Specification.ErfLayerParams.decodeText(reader);
                    break;
                case "gelu":
                    message.gelu = $root.CoreML.Specification.GeluLayerParams.decodeText(reader);
                    break;
                case "equal":
                    message.equal = $root.CoreML.Specification.EqualLayerParams.decodeText(reader);
                    break;
                case "notEqual":
                    message.notEqual = $root.CoreML.Specification.NotEqualLayerParams.decodeText(reader);
                    break;
                case "lessThan":
                    message.lessThan = $root.CoreML.Specification.LessThanLayerParams.decodeText(reader);
                    break;
                case "lessEqual":
                    message.lessEqual = $root.CoreML.Specification.LessEqualLayerParams.decodeText(reader);
                    break;
                case "greaterThan":
                    message.greaterThan = $root.CoreML.Specification.GreaterThanLayerParams.decodeText(reader);
                    break;
                case "greaterEqual":
                    message.greaterEqual = $root.CoreML.Specification.GreaterEqualLayerParams.decodeText(reader);
                    break;
                case "logicalOr":
                    message.logicalOr = $root.CoreML.Specification.LogicalOrLayerParams.decodeText(reader);
                    break;
                case "logicalXor":
                    message.logicalXor = $root.CoreML.Specification.LogicalXorLayerParams.decodeText(reader);
                    break;
                case "logicalNot":
                    message.logicalNot = $root.CoreML.Specification.LogicalNotLayerParams.decodeText(reader);
                    break;
                case "logicalAnd":
                    message.logicalAnd = $root.CoreML.Specification.LogicalAndLayerParams.decodeText(reader);
                    break;
                case "modBroadcastable":
                    message.modBroadcastable = $root.CoreML.Specification.ModBroadcastableLayerParams.decodeText(reader);
                    break;
                case "minBroadcastable":
                    message.minBroadcastable = $root.CoreML.Specification.MinBroadcastableLayerParams.decodeText(reader);
                    break;
                case "maxBroadcastable":
                    message.maxBroadcastable = $root.CoreML.Specification.MaxBroadcastableLayerParams.decodeText(reader);
                    break;
                case "addBroadcastable":
                    message.addBroadcastable = $root.CoreML.Specification.AddBroadcastableLayerParams.decodeText(reader);
                    break;
                case "powBroadcastable":
                    message.powBroadcastable = $root.CoreML.Specification.PowBroadcastableLayerParams.decodeText(reader);
                    break;
                case "divideBroadcastable":
                    message.divideBroadcastable = $root.CoreML.Specification.DivideBroadcastableLayerParams.decodeText(reader);
                    break;
                case "floorDivBroadcastable":
                    message.floorDivBroadcastable = $root.CoreML.Specification.FloorDivBroadcastableLayerParams.decodeText(reader);
                    break;
                case "multiplyBroadcastable":
                    message.multiplyBroadcastable = $root.CoreML.Specification.MultiplyBroadcastableLayerParams.decodeText(reader);
                    break;
                case "subtractBroadcastable":
                    message.subtractBroadcastable = $root.CoreML.Specification.SubtractBroadcastableLayerParams.decodeText(reader);
                    break;
                case "tile":
                    message.tile = $root.CoreML.Specification.TileLayerParams.decodeText(reader);
                    break;
                case "stack":
                    message.stack = $root.CoreML.Specification.StackLayerParams.decodeText(reader);
                    break;
                case "gather":
                    message.gather = $root.CoreML.Specification.GatherLayerParams.decodeText(reader);
                    break;
                case "scatter":
                    message.scatter = $root.CoreML.Specification.ScatterLayerParams.decodeText(reader);
                    break;
                case "gatherND":
                    message.gatherND = $root.CoreML.Specification.GatherNDLayerParams.decodeText(reader);
                    break;
                case "scatterND":
                    message.scatterND = $root.CoreML.Specification.ScatterNDLayerParams.decodeText(reader);
                    break;
                case "softmaxND":
                    message.softmaxND = $root.CoreML.Specification.SoftmaxNDLayerParams.decodeText(reader);
                    break;
                case "gatherAlongAxis":
                    message.gatherAlongAxis = $root.CoreML.Specification.GatherAlongAxisLayerParams.decodeText(reader);
                    break;
                case "scatterAlongAxis":
                    message.scatterAlongAxis = $root.CoreML.Specification.ScatterAlongAxisLayerParams.decodeText(reader);
                    break;
                case "reverse":
                    message.reverse = $root.CoreML.Specification.ReverseLayerParams.decodeText(reader);
                    break;
                case "reverseSeq":
                    message.reverseSeq = $root.CoreML.Specification.ReverseSeqLayerParams.decodeText(reader);
                    break;
                case "splitND":
                    message.splitND = $root.CoreML.Specification.SplitNDLayerParams.decodeText(reader);
                    break;
                case "concatND":
                    message.concatND = $root.CoreML.Specification.ConcatNDLayerParams.decodeText(reader);
                    break;
                case "transpose":
                    message.transpose = $root.CoreML.Specification.TransposeLayerParams.decodeText(reader);
                    break;
                case "sliceStatic":
                    message.sliceStatic = $root.CoreML.Specification.SliceStaticLayerParams.decodeText(reader);
                    break;
                case "sliceDynamic":
                    message.sliceDynamic = $root.CoreML.Specification.SliceDynamicLayerParams.decodeText(reader);
                    break;
                case "slidingWindows":
                    message.slidingWindows = $root.CoreML.Specification.SlidingWindowsLayerParams.decodeText(reader);
                    break;
                case "topK":
                    message.topK = $root.CoreML.Specification.TopKLayerParams.decodeText(reader);
                    break;
                case "argMin":
                    message.argMin = $root.CoreML.Specification.ArgMinLayerParams.decodeText(reader);
                    break;
                case "argMax":
                    message.argMax = $root.CoreML.Specification.ArgMaxLayerParams.decodeText(reader);
                    break;
                case "embeddingND":
                    message.embeddingND = $root.CoreML.Specification.EmbeddingNDLayerParams.decodeText(reader);
                    break;
                case "batchedMatmul":
                    message.batchedMatmul = $root.CoreML.Specification.BatchedMatMulLayerParams.decodeText(reader);
                    break;
                case "getShape":
                    message.getShape = $root.CoreML.Specification.GetShapeLayerParams.decodeText(reader);
                    break;
                case "loadConstantND":
                    message.loadConstantND = $root.CoreML.Specification.LoadConstantNDLayerParams.decodeText(reader);
                    break;
                case "fillLike":
                    message.fillLike = $root.CoreML.Specification.FillLikeLayerParams.decodeText(reader);
                    break;
                case "fillStatic":
                    message.fillStatic = $root.CoreML.Specification.FillStaticLayerParams.decodeText(reader);
                    break;
                case "fillDynamic":
                    message.fillDynamic = $root.CoreML.Specification.FillDynamicLayerParams.decodeText(reader);
                    break;
                case "broadcastToLike":
                    message.broadcastToLike = $root.CoreML.Specification.BroadcastToLikeLayerParams.decodeText(reader);
                    break;
                case "broadcastToStatic":
                    message.broadcastToStatic = $root.CoreML.Specification.BroadcastToStaticLayerParams.decodeText(reader);
                    break;
                case "broadcastToDynamic":
                    message.broadcastToDynamic = $root.CoreML.Specification.BroadcastToDynamicLayerParams.decodeText(reader);
                    break;
                case "squeeze":
                    message.squeeze = $root.CoreML.Specification.SqueezeLayerParams.decodeText(reader);
                    break;
                case "expandDims":
                    message.expandDims = $root.CoreML.Specification.ExpandDimsLayerParams.decodeText(reader);
                    break;
                case "flattenTo2D":
                    message.flattenTo2D = $root.CoreML.Specification.FlattenTo2DLayerParams.decodeText(reader);
                    break;
                case "reshapeLike":
                    message.reshapeLike = $root.CoreML.Specification.ReshapeLikeLayerParams.decodeText(reader);
                    break;
                case "reshapeStatic":
                    message.reshapeStatic = $root.CoreML.Specification.ReshapeStaticLayerParams.decodeText(reader);
                    break;
                case "reshapeDynamic":
                    message.reshapeDynamic = $root.CoreML.Specification.ReshapeDynamicLayerParams.decodeText(reader);
                    break;
                case "rankPreservingReshape":
                    message.rankPreservingReshape = $root.CoreML.Specification.RankPreservingReshapeLayerParams.decodeText(reader);
                    break;
                case "constantPad":
                    message.constantPad = $root.CoreML.Specification.ConstantPaddingLayerParams.decodeText(reader);
                    break;
                case "randomNormalLike":
                    message.randomNormalLike = $root.CoreML.Specification.RandomNormalLikeLayerParams.decodeText(reader);
                    break;
                case "randomNormalStatic":
                    message.randomNormalStatic = $root.CoreML.Specification.RandomNormalStaticLayerParams.decodeText(reader);
                    break;
                case "randomNormalDynamic":
                    message.randomNormalDynamic = $root.CoreML.Specification.RandomNormalDynamicLayerParams.decodeText(reader);
                    break;
                case "randomUniformLike":
                    message.randomUniformLike = $root.CoreML.Specification.RandomUniformLikeLayerParams.decodeText(reader);
                    break;
                case "randomUniformStatic":
                    message.randomUniformStatic = $root.CoreML.Specification.RandomUniformStaticLayerParams.decodeText(reader);
                    break;
                case "randomUniformDynamic":
                    message.randomUniformDynamic = $root.CoreML.Specification.RandomUniformDynamicLayerParams.decodeText(reader);
                    break;
                case "randomBernoulliLike":
                    message.randomBernoulliLike = $root.CoreML.Specification.RandomBernoulliLikeLayerParams.decodeText(reader);
                    break;
                case "randomBernoulliStatic":
                    message.randomBernoulliStatic = $root.CoreML.Specification.RandomBernoulliStaticLayerParams.decodeText(reader);
                    break;
                case "randomBernoulliDynamic":
                    message.randomBernoulliDynamic = $root.CoreML.Specification.RandomBernoulliDynamicLayerParams.decodeText(reader);
                    break;
                case "categoricalDistribution":
                    message.categoricalDistribution = $root.CoreML.Specification.CategoricalDistributionLayerParams.decodeText(reader);
                    break;
                case "reduceL1":
                    message.reduceL1 = $root.CoreML.Specification.ReduceL1LayerParams.decodeText(reader);
                    break;
                case "reduceL2":
                    message.reduceL2 = $root.CoreML.Specification.ReduceL2LayerParams.decodeText(reader);
                    break;
                case "reduceMax":
                    message.reduceMax = $root.CoreML.Specification.ReduceMaxLayerParams.decodeText(reader);
                    break;
                case "reduceMin":
                    message.reduceMin = $root.CoreML.Specification.ReduceMinLayerParams.decodeText(reader);
                    break;
                case "reduceSum":
                    message.reduceSum = $root.CoreML.Specification.ReduceSumLayerParams.decodeText(reader);
                    break;
                case "reduceProd":
                    message.reduceProd = $root.CoreML.Specification.ReduceProdLayerParams.decodeText(reader);
                    break;
                case "reduceMean":
                    message.reduceMean = $root.CoreML.Specification.ReduceMeanLayerParams.decodeText(reader);
                    break;
                case "reduceLogSum":
                    message.reduceLogSum = $root.CoreML.Specification.ReduceLogSumLayerParams.decodeText(reader);
                    break;
                case "reduceSumSquare":
                    message.reduceSumSquare = $root.CoreML.Specification.ReduceSumSquareLayerParams.decodeText(reader);
                    break;
                case "reduceLogSumExp":
                    message.reduceLogSumExp = $root.CoreML.Specification.ReduceLogSumExpLayerParams.decodeText(reader);
                    break;
                case "whereNonZero":
                    message.whereNonZero = $root.CoreML.Specification.WhereNonZeroLayerParams.decodeText(reader);
                    break;
                case "matrixBandPart":
                    message.matrixBandPart = $root.CoreML.Specification.MatrixBandPartLayerParams.decodeText(reader);
                    break;
                case "lowerTriangular":
                    message.lowerTriangular = $root.CoreML.Specification.LowerTriangularLayerParams.decodeText(reader);
                    break;
                case "upperTriangular":
                    message.upperTriangular = $root.CoreML.Specification.UpperTriangularLayerParams.decodeText(reader);
                    break;
                case "whereBroadcastable":
                    message.whereBroadcastable = $root.CoreML.Specification.WhereBroadcastableLayerParams.decodeText(reader);
                    break;
                case "layerNormalization":
                    message.layerNormalization = $root.CoreML.Specification.LayerNormalizationLayerParams.decodeText(reader);
                    break;
                case "NonMaximumSuppression":
                    message.NonMaximumSuppression = $root.CoreML.Specification.NonMaximumSuppressionLayerParams.decodeText(reader);
                    break;
                case "oneHot":
                    message.oneHot = $root.CoreML.Specification.OneHotLayerParams.decodeText(reader);
                    break;
                case "cumSum":
                    message.cumSum = $root.CoreML.Specification.CumSumLayerParams.decodeText(reader);
                    break;
                case "clampedReLU":
                    message.clampedReLU = $root.CoreML.Specification.ClampedReLULayerParams.decodeText(reader);
                    break;
                case "argSort":
                    message.argSort = $root.CoreML.Specification.ArgSortLayerParams.decodeText(reader);
                    break;
                case "pooling3d":
                    message.pooling3d = $root.CoreML.Specification.Pooling3DLayerParams.decodeText(reader);
                    break;
                case "globalPooling3d":
                    message.globalPooling3d = $root.CoreML.Specification.GlobalPooling3DLayerParams.decodeText(reader);
                    break;
                case "sliceBySize":
                    message.sliceBySize = $root.CoreML.Specification.SliceBySizeLayerParams.decodeText(reader);
                    break;
                case "convolution3d":
                    message.convolution3d = $root.CoreML.Specification.Convolution3DLayerParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BranchLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ifBranch":
                    message.ifBranch = $root.CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                case "elseBranch":
                    message.elseBranch = $root.CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LoopLayerParams();
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
                    message.conditionNetwork = $root.CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                case "bodyNetwork":
                    message.bodyNetwork = $root.CoreML.Specification.NeuralNetwork.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LoopBreakLayerParams();
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

$root.CoreML.Specification.LoopContinueLayerParams = class LoopContinueLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoopContinueLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LoopContinueLayerParams();
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

$root.CoreML.Specification.CopyLayerParams = class CopyLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CopyLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CopyLayerParams();
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

$root.CoreML.Specification.GreaterThanLayerParams = class GreaterThanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GreaterThanLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GreaterThanLayerParams();
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

$root.CoreML.Specification.GreaterThanLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.GreaterEqualLayerParams = class GreaterEqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GreaterEqualLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GreaterEqualLayerParams();
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

$root.CoreML.Specification.GreaterEqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.LessThanLayerParams = class LessThanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LessThanLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LessThanLayerParams();
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

$root.CoreML.Specification.LessThanLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.LessEqualLayerParams = class LessEqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LessEqualLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LessEqualLayerParams();
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

$root.CoreML.Specification.LessEqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.EqualLayerParams = class EqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.EqualLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.EqualLayerParams();
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

$root.CoreML.Specification.EqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.NotEqualLayerParams = class NotEqualLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.NotEqualLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.NotEqualLayerParams();
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

$root.CoreML.Specification.NotEqualLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.LogicalAndLayerParams = class LogicalAndLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalAndLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LogicalAndLayerParams();
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

$root.CoreML.Specification.LogicalOrLayerParams = class LogicalOrLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalOrLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LogicalOrLayerParams();
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

$root.CoreML.Specification.LogicalXorLayerParams = class LogicalXorLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalXorLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LogicalXorLayerParams();
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

$root.CoreML.Specification.LogicalNotLayerParams = class LogicalNotLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LogicalNotLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LogicalNotLayerParams();
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

$root.CoreML.Specification.BorderAmounts = class BorderAmounts {

    constructor() {
        this.borderAmounts = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BorderAmounts();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BorderAmounts();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "borderAmounts":
                    message.borderAmounts.push($root.CoreML.Specification.BorderAmounts.EdgeSizes.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BorderAmounts.EdgeSizes();
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

$root.CoreML.Specification.BorderAmounts.EdgeSizes.prototype.startEdgeSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.BorderAmounts.EdgeSizes.prototype.endEdgeSize = protobuf.Uint64.create(0);

$root.CoreML.Specification.ValidPadding = class ValidPadding {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ValidPadding();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ValidPadding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "paddingAmounts":
                    message.paddingAmounts = $root.CoreML.Specification.BorderAmounts.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SamePadding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "asymmetryMode":
                    message.asymmetryMode = reader.enum($root.CoreML.Specification.SamePadding.SamePaddingMode);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SamplingMode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "samplingMethod":
                    message.samplingMethod = reader.enum($root.CoreML.Specification.SamplingMode.Method);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BoxCoordinatesMode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "boxMode":
                    message.boxMode = reader.enum($root.CoreML.Specification.BoxCoordinatesMode.Coordinates);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.WeightParams();
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
                    message.quantization = $root.CoreML.Specification.QuantizationParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.QuantizationParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "numberOfBits":
                    message.numberOfBits = reader.uint64();
                    break;
                case "linearQuantization":
                    message.linearQuantization = $root.CoreML.Specification.LinearQuantizationParams.decodeText(reader);
                    break;
                case "lookupTableQuantization":
                    message.lookupTableQuantization = $root.CoreML.Specification.LookUpTableQuantizationParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LinearQuantizationParams();
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

$root.CoreML.Specification.LookUpTableQuantizationParams = class LookUpTableQuantizationParams {

    constructor() {
        this.floatValue = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LookUpTableQuantizationParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LookUpTableQuantizationParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ConvolutionLayerParams();
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
                    message.valid = $root.CoreML.Specification.ValidPadding.decodeText(reader);
                    break;
                case "same":
                    message.same = $root.CoreML.Specification.SamePadding.decodeText(reader);
                    break;
                case "isDeconvolution":
                    message.isDeconvolution = reader.bool();
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "weights":
                    message.weights = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Convolution3DLayerParams();
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
                    message.weights = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "paddingType":
                    message.paddingType = reader.enum($root.CoreML.Specification.Convolution3DLayerParams.PaddingType);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.InnerProductLayerParams();
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
                    message.weights = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.EmbeddingLayerParams();
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
                    message.weights = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.EmbeddingNDLayerParams();
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
                    message.weights = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BatchnormLayerParams();
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
                    message.gamma = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "beta":
                    message.beta = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "mean":
                    message.mean = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "variance":
                    message.variance = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.PoolingLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum($root.CoreML.Specification.PoolingLayerParams.PoolingType);
                    break;
                case "kernelSize":
                    reader.array(message.kernelSize, () => reader.uint64());
                    break;
                case "stride":
                    reader.array(message.stride, () => reader.uint64());
                    break;
                case "valid":
                    message.valid = $root.CoreML.Specification.ValidPadding.decodeText(reader);
                    break;
                case "same":
                    message.same = $root.CoreML.Specification.SamePadding.decodeText(reader);
                    break;
                case "includeLastPixel":
                    message.includeLastPixel = $root.CoreML.Specification.PoolingLayerParams.ValidCompletePadding.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PoolingLayerParams.ValidCompletePadding();
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

$root.CoreML.Specification.Pooling3DLayerParams = class Pooling3DLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Pooling3DLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Pooling3DLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum($root.CoreML.Specification.Pooling3DLayerParams.PoolingType3D);
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
                    message.paddingType = reader.enum($root.CoreML.Specification.Pooling3DLayerParams.Pooling3DPaddingType);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GlobalPooling3DLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum($root.CoreML.Specification.GlobalPooling3DLayerParams.GlobalPoolingType3D);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.PaddingLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "constant":
                    message.constant = $root.CoreML.Specification.PaddingLayerParams.PaddingConstant.decodeText(reader);
                    break;
                case "reflection":
                    message.reflection = $root.CoreML.Specification.PaddingLayerParams.PaddingReflection.decodeText(reader);
                    break;
                case "replication":
                    message.replication = $root.CoreML.Specification.PaddingLayerParams.PaddingReplication.decodeText(reader);
                    break;
                case "paddingAmounts":
                    message.paddingAmounts = $root.CoreML.Specification.BorderAmounts.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingConstant();
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

$root.CoreML.Specification.PaddingLayerParams.PaddingConstant.prototype.value = 0;

$root.CoreML.Specification.PaddingLayerParams.PaddingReflection = class PaddingReflection {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReflection();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReflection();
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

$root.CoreML.Specification.PaddingLayerParams.PaddingReplication = class PaddingReplication {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReplication();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReplication();
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

$root.CoreML.Specification.ConcatLayerParams = class ConcatLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ConcatLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ConcatLayerParams();
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

$root.CoreML.Specification.ConcatLayerParams.prototype.sequenceConcat = false;

$root.CoreML.Specification.LRNLayerParams = class LRNLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LRNLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LRNLayerParams();
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

$root.CoreML.Specification.LRNLayerParams.prototype.alpha = 0;
$root.CoreML.Specification.LRNLayerParams.prototype.beta = 0;
$root.CoreML.Specification.LRNLayerParams.prototype.localSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.LRNLayerParams.prototype.k = 0;

$root.CoreML.Specification.SoftmaxLayerParams = class SoftmaxLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SoftmaxLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SoftmaxLayerParams();
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

$root.CoreML.Specification.SplitLayerParams = class SplitLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SplitLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SplitLayerParams();
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

$root.CoreML.Specification.SplitLayerParams.prototype.nOutputs = protobuf.Uint64.create(0);

$root.CoreML.Specification.AddLayerParams = class AddLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AddLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AddLayerParams();
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

$root.CoreML.Specification.AddLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.MultiplyLayerParams = class MultiplyLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MultiplyLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MultiplyLayerParams();
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

$root.CoreML.Specification.MultiplyLayerParams.prototype.alpha = 0;

$root.CoreML.Specification.UnaryFunctionLayerParams = class UnaryFunctionLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UnaryFunctionLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.UnaryFunctionLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum($root.CoreML.Specification.UnaryFunctionLayerParams.Operation);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.UpsampleLayerParams();
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
                    message.mode = reader.enum($root.CoreML.Specification.UpsampleLayerParams.InterpolationMode);
                    break;
                case "linearUpsampleMode":
                    message.linearUpsampleMode = reader.enum($root.CoreML.Specification.UpsampleLayerParams.LinearUpsampleMode);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ResizeBilinearLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetSize":
                    reader.array(message.targetSize, () => reader.uint64());
                    break;
                case "mode":
                    message.mode = $root.CoreML.Specification.SamplingMode.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CropResizeLayerParams();
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
                    message.mode = $root.CoreML.Specification.SamplingMode.decodeText(reader);
                    break;
                case "boxIndicesMode":
                    message.boxIndicesMode = $root.CoreML.Specification.BoxCoordinatesMode.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BiasLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.uint64());
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ScaleLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shapeScale":
                    reader.array(message.shapeScale, () => reader.uint64());
                    break;
                case "scale":
                    message.scale = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "hasBias":
                    message.hasBias = reader.bool();
                    break;
                case "shapeBias":
                    reader.array(message.shapeBias, () => reader.uint64());
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LoadConstantLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.uint64());
                    break;
                case "data":
                    message.data = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.L2NormalizeLayerParams();
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

$root.CoreML.Specification.L2NormalizeLayerParams.prototype.epsilon = 0;

$root.CoreML.Specification.FlattenLayerParams = class FlattenLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FlattenLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FlattenLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.FlattenLayerParams.FlattenOrder);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReshapeLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "targetShape":
                    reader.array(message.targetShape, () => reader.int64());
                    break;
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.ReshapeLayerParams.ReshapeOrder);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PermuteLayerParams();
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

$root.CoreML.Specification.ReorganizeDataLayerParams = class ReorganizeDataLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReorganizeDataLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReorganizeDataLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.ReorganizeDataLayerParams.ReorganizationType);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SliceLayerParams();
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
                    message.axis = reader.enum($root.CoreML.Specification.SliceLayerParams.SliceAxis);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.ReduceLayerParams.ReduceOperation);
                    break;
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                case "axis":
                    message.axis = reader.enum($root.CoreML.Specification.ReduceLayerParams.ReduceAxis);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CropLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "cropAmounts":
                    message.cropAmounts = $root.CoreML.Specification.BorderAmounts.decodeText(reader);
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

$root.CoreML.Specification.CropLayerParams.prototype.cropAmounts = null;

$root.CoreML.Specification.AverageLayerParams = class AverageLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AverageLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AverageLayerParams();
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

$root.CoreML.Specification.MaxLayerParams = class MaxLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MaxLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MaxLayerParams();
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

$root.CoreML.Specification.MinLayerParams = class MinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MinLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MinLayerParams();
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

$root.CoreML.Specification.DotProductLayerParams = class DotProductLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DotProductLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.DotProductLayerParams();
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

$root.CoreML.Specification.DotProductLayerParams.prototype.cosineSimilarity = false;

$root.CoreML.Specification.MeanVarianceNormalizeLayerParams = class MeanVarianceNormalizeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MeanVarianceNormalizeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MeanVarianceNormalizeLayerParams();
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

$root.CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.acrossChannels = false;
$root.CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.normalizeVariance = false;
$root.CoreML.Specification.MeanVarianceNormalizeLayerParams.prototype.epsilon = 0;

$root.CoreML.Specification.SequenceRepeatLayerParams = class SequenceRepeatLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SequenceRepeatLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SequenceRepeatLayerParams();
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

$root.CoreML.Specification.SequenceRepeatLayerParams.prototype.nRepetitions = protobuf.Uint64.create(0);

$root.CoreML.Specification.SimpleRecurrentLayerParams = class SimpleRecurrentLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SimpleRecurrentLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SimpleRecurrentLayerParams();
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
                    message.activation = $root.CoreML.Specification.ActivationParams.decodeText(reader);
                    break;
                case "sequenceOutput":
                    message.sequenceOutput = reader.bool();
                    break;
                case "hasBiasVector":
                    message.hasBiasVector = reader.bool();
                    break;
                case "weightMatrix":
                    message.weightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "recursionMatrix":
                    message.recursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "biasVector":
                    message.biasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.GRULayerParams();
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
                    message.activations.push($root.CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "sequenceOutput":
                    message.sequenceOutput = reader.bool();
                    break;
                case "hasBiasVectors":
                    message.hasBiasVectors = reader.bool();
                    break;
                case "updateGateWeightMatrix":
                    message.updateGateWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "resetGateWeightMatrix":
                    message.resetGateWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateWeightMatrix":
                    message.outputGateWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "updateGateRecursionMatrix":
                    message.updateGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "resetGateRecursionMatrix":
                    message.resetGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateRecursionMatrix":
                    message.outputGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "updateGateBiasVector":
                    message.updateGateBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "resetGateBiasVector":
                    message.resetGateBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateBiasVector":
                    message.outputGateBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LSTMParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LSTMWeightParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputGateWeightMatrix":
                    message.inputGateWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGateWeightMatrix":
                    message.forgetGateWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "blockInputWeightMatrix":
                    message.blockInputWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateWeightMatrix":
                    message.outputGateWeightMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "inputGateRecursionMatrix":
                    message.inputGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGateRecursionMatrix":
                    message.forgetGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "blockInputRecursionMatrix":
                    message.blockInputRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateRecursionMatrix":
                    message.outputGateRecursionMatrix = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "inputGateBiasVector":
                    message.inputGateBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGateBiasVector":
                    message.forgetGateBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "blockInputBiasVector":
                    message.blockInputBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGateBiasVector":
                    message.outputGateBiasVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "inputGatePeepholeVector":
                    message.inputGatePeepholeVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "forgetGatePeepholeVector":
                    message.forgetGatePeepholeVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "outputGatePeepholeVector":
                    message.outputGatePeepholeVector = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.UniDirectionalLSTMLayerParams();
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
                    message.activations.push($root.CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "params":
                    message.params = $root.CoreML.Specification.LSTMParams.decodeText(reader);
                    break;
                case "weightParams":
                    message.weightParams = $root.CoreML.Specification.LSTMWeightParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BiDirectionalLSTMLayerParams();
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
                    message.activationsForwardLSTM.push($root.CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "activationsBackwardLSTM":
                    message.activationsBackwardLSTM.push($root.CoreML.Specification.ActivationParams.decodeText(reader));
                    break;
                case "params":
                    message.params = $root.CoreML.Specification.LSTMParams.decodeText(reader);
                    break;
                case "weightParams":
                    message.weightParams.push($root.CoreML.Specification.LSTMWeightParams.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.CustomLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "className":
                    message.className = reader.string();
                    break;
                case "weights":
                    message.weights.push($root.CoreML.Specification.WeightParams.decodeText(reader));
                    break;
                case "parameters":
                    reader.entry(message.parameters, () => reader.string(), () => $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.decodeText(reader));
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue();
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

$root.CoreML.Specification.TransposeLayerParams = class TransposeLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TransposeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.TransposeLayerParams();
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

$root.CoreML.Specification.BatchedMatMulLayerParams = class BatchedMatMulLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BatchedMatMulLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.BatchedMatMulLayerParams();
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
                    message.weights = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "bias":
                    message.bias = $root.CoreML.Specification.WeightParams.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ConcatNDLayerParams();
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

$root.CoreML.Specification.ConcatNDLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ConcatNDLayerParams.prototype.interleave = false;

$root.CoreML.Specification.SoftmaxNDLayerParams = class SoftmaxNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SoftmaxNDLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SoftmaxNDLayerParams();
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

$root.CoreML.Specification.SoftmaxNDLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ReverseLayerParams = class ReverseLayerParams {

    constructor() {
        this.reverseDim = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReverseLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReverseLayerParams();
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

$root.CoreML.Specification.ReverseSeqLayerParams = class ReverseSeqLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReverseSeqLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReverseSeqLayerParams();
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

$root.CoreML.Specification.ReverseSeqLayerParams.prototype.batchAxis = protobuf.Int64.create(0);
$root.CoreML.Specification.ReverseSeqLayerParams.prototype.sequenceAxis = protobuf.Int64.create(0);

$root.CoreML.Specification.LoadConstantNDLayerParams = class LoadConstantNDLayerParams {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LoadConstantNDLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LoadConstantNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.uint64());
                    break;
                case "data":
                    message.data = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FillLikeLayerParams();
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

$root.CoreML.Specification.FillLikeLayerParams.prototype.value = 0;

$root.CoreML.Specification.FillStaticLayerParams = class FillStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FillStaticLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FillStaticLayerParams();
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

$root.CoreML.Specification.FillStaticLayerParams.prototype.value = 0;

$root.CoreML.Specification.FillDynamicLayerParams = class FillDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FillDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FillDynamicLayerParams();
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

$root.CoreML.Specification.FillDynamicLayerParams.prototype.value = 0;

$root.CoreML.Specification.WhereBroadcastableLayerParams = class WhereBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.WhereBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.WhereBroadcastableLayerParams();
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

$root.CoreML.Specification.SinLayerParams = class SinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SinLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SinLayerParams();
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

$root.CoreML.Specification.CosLayerParams = class CosLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CosLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CosLayerParams();
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

$root.CoreML.Specification.TanLayerParams = class TanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TanLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.TanLayerParams();
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

$root.CoreML.Specification.AsinLayerParams = class AsinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AsinLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AsinLayerParams();
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

$root.CoreML.Specification.AcosLayerParams = class AcosLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AcosLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AcosLayerParams();
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

$root.CoreML.Specification.AtanLayerParams = class AtanLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AtanLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AtanLayerParams();
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

$root.CoreML.Specification.SinhLayerParams = class SinhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SinhLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SinhLayerParams();
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

$root.CoreML.Specification.CoshLayerParams = class CoshLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CoshLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CoshLayerParams();
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

$root.CoreML.Specification.TanhLayerParams = class TanhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TanhLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.TanhLayerParams();
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

$root.CoreML.Specification.AsinhLayerParams = class AsinhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AsinhLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AsinhLayerParams();
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

$root.CoreML.Specification.AcoshLayerParams = class AcoshLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AcoshLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AcoshLayerParams();
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

$root.CoreML.Specification.AtanhLayerParams = class AtanhLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AtanhLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AtanhLayerParams();
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

$root.CoreML.Specification.PowBroadcastableLayerParams = class PowBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PowBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PowBroadcastableLayerParams();
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

$root.CoreML.Specification.Exp2LayerParams = class Exp2LayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.Exp2LayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Exp2LayerParams();
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

$root.CoreML.Specification.WhereNonZeroLayerParams = class WhereNonZeroLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.WhereNonZeroLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.WhereNonZeroLayerParams();
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

$root.CoreML.Specification.MatrixBandPartLayerParams = class MatrixBandPartLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MatrixBandPartLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MatrixBandPartLayerParams();
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

$root.CoreML.Specification.MatrixBandPartLayerParams.prototype.numLower = protobuf.Int64.create(0);
$root.CoreML.Specification.MatrixBandPartLayerParams.prototype.numUpper = protobuf.Int64.create(0);

$root.CoreML.Specification.UpperTriangularLayerParams = class UpperTriangularLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.UpperTriangularLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.UpperTriangularLayerParams();
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

$root.CoreML.Specification.UpperTriangularLayerParams.prototype.k = protobuf.Int64.create(0);

$root.CoreML.Specification.LowerTriangularLayerParams = class LowerTriangularLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LowerTriangularLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LowerTriangularLayerParams();
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

$root.CoreML.Specification.LowerTriangularLayerParams.prototype.k = protobuf.Int64.create(0);

$root.CoreML.Specification.BroadcastToLikeLayerParams = class BroadcastToLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BroadcastToLikeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BroadcastToLikeLayerParams();
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

$root.CoreML.Specification.BroadcastToStaticLayerParams = class BroadcastToStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BroadcastToStaticLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BroadcastToStaticLayerParams();
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

$root.CoreML.Specification.BroadcastToDynamicLayerParams = class BroadcastToDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.BroadcastToDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.BroadcastToDynamicLayerParams();
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

$root.CoreML.Specification.AddBroadcastableLayerParams = class AddBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.AddBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.AddBroadcastableLayerParams();
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

$root.CoreML.Specification.MaxBroadcastableLayerParams = class MaxBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MaxBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MaxBroadcastableLayerParams();
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

$root.CoreML.Specification.MinBroadcastableLayerParams = class MinBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MinBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MinBroadcastableLayerParams();
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

$root.CoreML.Specification.ModBroadcastableLayerParams = class ModBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ModBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ModBroadcastableLayerParams();
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

$root.CoreML.Specification.FloorDivBroadcastableLayerParams = class FloorDivBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FloorDivBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FloorDivBroadcastableLayerParams();
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

$root.CoreML.Specification.SubtractBroadcastableLayerParams = class SubtractBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SubtractBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SubtractBroadcastableLayerParams();
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

$root.CoreML.Specification.MultiplyBroadcastableLayerParams = class MultiplyBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MultiplyBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MultiplyBroadcastableLayerParams();
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

$root.CoreML.Specification.DivideBroadcastableLayerParams = class DivideBroadcastableLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DivideBroadcastableLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.DivideBroadcastableLayerParams();
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

$root.CoreML.Specification.GatherLayerParams = class GatherLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GatherLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GatherLayerParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ScatterLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.ScatterMode);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GatherNDLayerParams();
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

$root.CoreML.Specification.ScatterNDLayerParams = class ScatterNDLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ScatterNDLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ScatterNDLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.ScatterMode);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GatherAlongAxisLayerParams();
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

$root.CoreML.Specification.GatherAlongAxisLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ScatterAlongAxisLayerParams = class ScatterAlongAxisLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ScatterAlongAxisLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ScatterAlongAxisLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.ScatterMode);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.StackLayerParams();
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

$root.CoreML.Specification.StackLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.RankPreservingReshapeLayerParams = class RankPreservingReshapeLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RankPreservingReshapeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RankPreservingReshapeLayerParams();
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

$root.CoreML.Specification.ConstantPaddingLayerParams = class ConstantPaddingLayerParams {

    constructor() {
        this.padAmounts = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ConstantPaddingLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ConstantPaddingLayerParams();
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

$root.CoreML.Specification.ConstantPaddingLayerParams.prototype.value = 0;
$root.CoreML.Specification.ConstantPaddingLayerParams.prototype.padToGivenOutputSizeMode = false;

$root.CoreML.Specification.RandomNormalLikeLayerParams = class RandomNormalLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomNormalLikeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomNormalLikeLayerParams();
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

$root.CoreML.Specification.RandomNormalLikeLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomNormalLikeLayerParams.prototype.mean = 0;
$root.CoreML.Specification.RandomNormalLikeLayerParams.prototype.stdDev = 0;

$root.CoreML.Specification.RandomNormalStaticLayerParams = class RandomNormalStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomNormalStaticLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomNormalStaticLayerParams();
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

$root.CoreML.Specification.RandomNormalStaticLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomNormalStaticLayerParams.prototype.mean = 0;
$root.CoreML.Specification.RandomNormalStaticLayerParams.prototype.stdDev = 0;

$root.CoreML.Specification.RandomNormalDynamicLayerParams = class RandomNormalDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomNormalDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomNormalDynamicLayerParams();
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

$root.CoreML.Specification.RandomNormalDynamicLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomNormalDynamicLayerParams.prototype.mean = 0;
$root.CoreML.Specification.RandomNormalDynamicLayerParams.prototype.stdDev = 0;

$root.CoreML.Specification.RandomUniformLikeLayerParams = class RandomUniformLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomUniformLikeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomUniformLikeLayerParams();
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

$root.CoreML.Specification.RandomUniformLikeLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomUniformLikeLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.RandomUniformLikeLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.RandomUniformStaticLayerParams = class RandomUniformStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomUniformStaticLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomUniformStaticLayerParams();
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

$root.CoreML.Specification.RandomUniformStaticLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomUniformStaticLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.RandomUniformStaticLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.RandomUniformDynamicLayerParams = class RandomUniformDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomUniformDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomUniformDynamicLayerParams();
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

$root.CoreML.Specification.RandomUniformDynamicLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomUniformDynamicLayerParams.prototype.minVal = 0;
$root.CoreML.Specification.RandomUniformDynamicLayerParams.prototype.maxVal = 0;

$root.CoreML.Specification.RandomBernoulliLikeLayerParams = class RandomBernoulliLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomBernoulliLikeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomBernoulliLikeLayerParams();
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

$root.CoreML.Specification.RandomBernoulliLikeLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomBernoulliLikeLayerParams.prototype.prob = 0;

$root.CoreML.Specification.RandomBernoulliStaticLayerParams = class RandomBernoulliStaticLayerParams {

    constructor() {
        this.outputShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomBernoulliStaticLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomBernoulliStaticLayerParams();
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

$root.CoreML.Specification.RandomBernoulliStaticLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomBernoulliStaticLayerParams.prototype.prob = 0;

$root.CoreML.Specification.RandomBernoulliDynamicLayerParams = class RandomBernoulliDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RandomBernoulliDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RandomBernoulliDynamicLayerParams();
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

$root.CoreML.Specification.RandomBernoulliDynamicLayerParams.prototype.seed = protobuf.Int64.create(0);
$root.CoreML.Specification.RandomBernoulliDynamicLayerParams.prototype.prob = 0;

$root.CoreML.Specification.CategoricalDistributionLayerParams = class CategoricalDistributionLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CategoricalDistributionLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CategoricalDistributionLayerParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceL1LayerParams();
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

$root.CoreML.Specification.ReduceL1LayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceL1LayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceL2LayerParams = class ReduceL2LayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceL2LayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceL2LayerParams();
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

$root.CoreML.Specification.ReduceL2LayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceL2LayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceMaxLayerParams = class ReduceMaxLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceMaxLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceMaxLayerParams();
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

$root.CoreML.Specification.ReduceMaxLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceMaxLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceMinLayerParams = class ReduceMinLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceMinLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceMinLayerParams();
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

$root.CoreML.Specification.ReduceMinLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceMinLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceSumLayerParams = class ReduceSumLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceSumLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceSumLayerParams();
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

$root.CoreML.Specification.ReduceSumLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceSumLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceProdLayerParams = class ReduceProdLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceProdLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceProdLayerParams();
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

$root.CoreML.Specification.ReduceProdLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceProdLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceMeanLayerParams = class ReduceMeanLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceMeanLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceMeanLayerParams();
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

$root.CoreML.Specification.ReduceMeanLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceMeanLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceLogSumLayerParams = class ReduceLogSumLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceLogSumLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceLogSumLayerParams();
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

$root.CoreML.Specification.ReduceLogSumLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceLogSumLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceSumSquareLayerParams = class ReduceSumSquareLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceSumSquareLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceSumSquareLayerParams();
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

$root.CoreML.Specification.ReduceSumSquareLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceSumSquareLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ReduceLogSumExpLayerParams = class ReduceLogSumExpLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReduceLogSumExpLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReduceLogSumExpLayerParams();
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

$root.CoreML.Specification.ReduceLogSumExpLayerParams.prototype.keepDims = false;
$root.CoreML.Specification.ReduceLogSumExpLayerParams.prototype.reduceAll = false;

$root.CoreML.Specification.ExpandDimsLayerParams = class ExpandDimsLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ExpandDimsLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ExpandDimsLayerParams();
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

$root.CoreML.Specification.FlattenTo2DLayerParams = class FlattenTo2DLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FlattenTo2DLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FlattenTo2DLayerParams();
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

$root.CoreML.Specification.FlattenTo2DLayerParams.prototype.axis = protobuf.Int64.create(0);

$root.CoreML.Specification.ReshapeStaticLayerParams = class ReshapeStaticLayerParams {

    constructor() {
        this.targetShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeStaticLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReshapeStaticLayerParams();
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

$root.CoreML.Specification.ReshapeLikeLayerParams = class ReshapeLikeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeLikeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReshapeLikeLayerParams();
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

$root.CoreML.Specification.ReshapeDynamicLayerParams = class ReshapeDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ReshapeDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ReshapeDynamicLayerParams();
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

$root.CoreML.Specification.SqueezeLayerParams = class SqueezeLayerParams {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SqueezeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SqueezeLayerParams();
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

$root.CoreML.Specification.SqueezeLayerParams.prototype.squeezeAll = false;

$root.CoreML.Specification.TopKLayerParams = class TopKLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TopKLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.TopKLayerParams();
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

$root.CoreML.Specification.TopKLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.TopKLayerParams.prototype.K = protobuf.Uint64.create(0);
$root.CoreML.Specification.TopKLayerParams.prototype.useBottomK = false;

$root.CoreML.Specification.ArgMaxLayerParams = class ArgMaxLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArgMaxLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ArgMaxLayerParams();
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

$root.CoreML.Specification.ArgMaxLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ArgMaxLayerParams.prototype.removeDim = false;

$root.CoreML.Specification.ArgMinLayerParams = class ArgMinLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArgMinLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ArgMinLayerParams();
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

$root.CoreML.Specification.ArgMinLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ArgMinLayerParams.prototype.removeDim = false;

$root.CoreML.Specification.SplitNDLayerParams = class SplitNDLayerParams {

    constructor() {
        this.splitSizes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SplitNDLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SplitNDLayerParams();
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

$root.CoreML.Specification.SplitNDLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.SplitNDLayerParams.prototype.numSplits = protobuf.Uint64.create(0);

$root.CoreML.Specification.CeilLayerParams = class CeilLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CeilLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CeilLayerParams();
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

$root.CoreML.Specification.RoundLayerParams = class RoundLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RoundLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RoundLayerParams();
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

$root.CoreML.Specification.FloorLayerParams = class FloorLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.FloorLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.FloorLayerParams();
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

$root.CoreML.Specification.SignLayerParams = class SignLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SignLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SignLayerParams();
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

$root.CoreML.Specification.ClipLayerParams = class ClipLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ClipLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ClipLayerParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SliceStaticLayerParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SliceDynamicLayerParams();
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

$root.CoreML.Specification.TileLayerParams = class TileLayerParams {

    constructor() {
        this.reps = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TileLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.TileLayerParams();
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

$root.CoreML.Specification.GetShapeLayerParams = class GetShapeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GetShapeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GetShapeLayerParams();
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

$root.CoreML.Specification.ErfLayerParams = class ErfLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ErfLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ErfLayerParams();
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

$root.CoreML.Specification.GeluLayerParams = class GeluLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.GeluLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.GeluLayerParams();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.enum($root.CoreML.Specification.GeluLayerParams.GeluMode);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RangeStaticLayerParams();
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

$root.CoreML.Specification.RangeStaticLayerParams.prototype.endValue = 0;
$root.CoreML.Specification.RangeStaticLayerParams.prototype.startValue = 0;
$root.CoreML.Specification.RangeStaticLayerParams.prototype.stepSizeValue = 0;

$root.CoreML.Specification.RangeDynamicLayerParams = class RangeDynamicLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RangeDynamicLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RangeDynamicLayerParams();
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

$root.CoreML.Specification.RangeDynamicLayerParams.prototype.startValue = 0;
$root.CoreML.Specification.RangeDynamicLayerParams.prototype.stepSizeValue = 0;

$root.CoreML.Specification.SlidingWindowsLayerParams = class SlidingWindowsLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SlidingWindowsLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SlidingWindowsLayerParams();
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

$root.CoreML.Specification.SlidingWindowsLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.SlidingWindowsLayerParams.prototype.windowSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.SlidingWindowsLayerParams.prototype.step = protobuf.Uint64.create(0);

$root.CoreML.Specification.LayerNormalizationLayerParams = class LayerNormalizationLayerParams {

    constructor() {
        this.normalizedShape = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LayerNormalizationLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LayerNormalizationLayerParams();
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
                    message.gamma = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                case "beta":
                    message.beta = $root.CoreML.Specification.WeightParams.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.NonMaximumSuppressionLayerParams();
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

$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.iouThreshold = 0;
$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.scoreThreshold = 0;
$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.maxBoxes = protobuf.Uint64.create(0);
$root.CoreML.Specification.NonMaximumSuppressionLayerParams.prototype.perClassSuppression = false;

$root.CoreML.Specification.ClampedReLULayerParams = class ClampedReLULayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ClampedReLULayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ClampedReLULayerParams();
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

$root.CoreML.Specification.ClampedReLULayerParams.prototype.alpha = 0;
$root.CoreML.Specification.ClampedReLULayerParams.prototype.beta = 0;

$root.CoreML.Specification.ArgSortLayerParams = class ArgSortLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ArgSortLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ArgSortLayerParams();
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

$root.CoreML.Specification.ArgSortLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.ArgSortLayerParams.prototype.descending = false;

$root.CoreML.Specification.SliceBySizeLayerParams = class SliceBySizeLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SliceBySizeLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SliceBySizeLayerParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NeuralNetworkClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layers":
                    message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decodeText(reader));
                    break;
                case "preprocessing":
                    message.preprocessing.push($root.CoreML.Specification.NeuralNetworkPreprocessing.decodeText(reader));
                    break;
                case "arrayInputShapeMapping":
                    message.arrayInputShapeMapping = reader.enum($root.CoreML.Specification.NeuralNetworkMultiArrayShapeMapping);
                    break;
                case "imageInputShapeMapping":
                    message.imageInputShapeMapping = reader.enum($root.CoreML.Specification.NeuralNetworkImageShapeMapping);
                    break;
                case "updateParams":
                    message.updateParams = $root.CoreML.Specification.NetworkUpdateParameters.decodeText(reader);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decodeText(reader);
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

$root.CoreML.Specification.NeuralNetworkClassifier.prototype.arrayInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetworkClassifier.prototype.imageInputShapeMapping = 0;
$root.CoreML.Specification.NeuralNetworkClassifier.prototype.updateParams = null;
$root.CoreML.Specification.NeuralNetworkClassifier.prototype.labelProbabilityLayerName = "";

$root.CoreML.Specification.OneHotLayerParams = class OneHotLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.OneHotLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.OneHotLayerParams();
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

$root.CoreML.Specification.OneHotLayerParams.prototype.oneHotVectorSize = protobuf.Uint64.create(0);
$root.CoreML.Specification.OneHotLayerParams.prototype.axis = protobuf.Int64.create(0);
$root.CoreML.Specification.OneHotLayerParams.prototype.onValue = 0;
$root.CoreML.Specification.OneHotLayerParams.prototype.offValue = 0;

$root.CoreML.Specification.CumSumLayerParams = class CumSumLayerParams {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.CumSumLayerParams();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CumSumLayerParams();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NeuralNetworkRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layers":
                    message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decodeText(reader));
                    break;
                case "preprocessing":
                    message.preprocessing.push($root.CoreML.Specification.NeuralNetworkPreprocessing.decodeText(reader));
                    break;
                case "arrayInputShapeMapping":
                    message.arrayInputShapeMapping = reader.enum($root.CoreML.Specification.NeuralNetworkMultiArrayShapeMapping);
                    break;
                case "imageInputShapeMapping":
                    message.imageInputShapeMapping = reader.enum($root.CoreML.Specification.NeuralNetworkImageShapeMapping);
                    break;
                case "updateParams":
                    message.updateParams = $root.CoreML.Specification.NetworkUpdateParameters.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NetworkUpdateParameters();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lossLayers":
                    message.lossLayers.push($root.CoreML.Specification.LossLayer.decodeText(reader));
                    break;
                case "optimizer":
                    message.optimizer = $root.CoreML.Specification.Optimizer.decodeText(reader);
                    break;
                case "epochs":
                    message.epochs = $root.CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "shuffle":
                    message.shuffle = $root.CoreML.Specification.BoolParameter.decodeText(reader);
                    break;
                case "seed":
                    message.seed = $root.CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LossLayer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "categoricalCrossEntropyLossLayer":
                    message.categoricalCrossEntropyLossLayer = $root.CoreML.Specification.CategoricalCrossEntropyLossLayer.decodeText(reader);
                    break;
                case "meanSquaredErrorLossLayer":
                    message.meanSquaredErrorLossLayer = $root.CoreML.Specification.MeanSquaredErrorLossLayer.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.CategoricalCrossEntropyLossLayer();
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

$root.CoreML.Specification.CategoricalCrossEntropyLossLayer.prototype.input = "";
$root.CoreML.Specification.CategoricalCrossEntropyLossLayer.prototype.target = "";

$root.CoreML.Specification.MeanSquaredErrorLossLayer = class MeanSquaredErrorLossLayer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.MeanSquaredErrorLossLayer();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.MeanSquaredErrorLossLayer();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Optimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sgdOptimizer":
                    message.sgdOptimizer = $root.CoreML.Specification.SGDOptimizer.decodeText(reader);
                    break;
                case "adamOptimizer":
                    message.adamOptimizer = $root.CoreML.Specification.AdamOptimizer.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SGDOptimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "learningRate":
                    message.learningRate = $root.CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "miniBatchSize":
                    message.miniBatchSize = $root.CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "momentum":
                    message.momentum = $root.CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.AdamOptimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "learningRate":
                    message.learningRate = $root.CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "miniBatchSize":
                    message.miniBatchSize = $root.CoreML.Specification.Int64Parameter.decodeText(reader);
                    break;
                case "beta1":
                    message.beta1 = $root.CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "beta2":
                    message.beta2 = $root.CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                case "eps":
                    message.eps = $root.CoreML.Specification.DoubleParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Normalizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "normType":
                    message.normType = reader.enum($root.CoreML.Specification.Normalizer.NormType);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.OneHotEncoder();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "stringCategories":
                    message.stringCategories = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64Categories":
                    message.int64Categories = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                case "outputSparse":
                    message.outputSparse = reader.bool();
                    break;
                case "handleUnknown":
                    message.handleUnknown = reader.enum($root.CoreML.Specification.OneHotEncoder.HandleUnknown);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Scaler();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.NonMaximumSuppression();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pickTop":
                    message.pickTop = $root.CoreML.Specification.NonMaximumSuppression.PickTop.decodeText(reader);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.NonMaximumSuppression.PickTop();
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

$root.CoreML.Specification.NonMaximumSuppression.PickTop.prototype.perClass = false;

$root.CoreML.Specification.LinearKernel = class LinearKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.LinearKernel();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.LinearKernel();
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

$root.CoreML.Specification.RBFKernel = class RBFKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.RBFKernel();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.RBFKernel();
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

$root.CoreML.Specification.RBFKernel.prototype.gamma = 0;

$root.CoreML.Specification.PolyKernel = class PolyKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.PolyKernel();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.PolyKernel();
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

$root.CoreML.Specification.PolyKernel.prototype.degree = 0;
$root.CoreML.Specification.PolyKernel.prototype.c = 0;
$root.CoreML.Specification.PolyKernel.prototype.gamma = 0;

$root.CoreML.Specification.SigmoidKernel = class SigmoidKernel {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SigmoidKernel();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SigmoidKernel();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.Kernel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linearKernel":
                    message.linearKernel = $root.CoreML.Specification.LinearKernel.decodeText(reader);
                    break;
                case "rbfKernel":
                    message.rbfKernel = $root.CoreML.Specification.RBFKernel.decodeText(reader);
                    break;
                case "polyKernel":
                    message.polyKernel = $root.CoreML.Specification.PolyKernel.decodeText(reader);
                    break;
                case "sigmoidKernel":
                    message.sigmoidKernel = $root.CoreML.Specification.SigmoidKernel.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.SparseNode();
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

$root.CoreML.Specification.SparseNode.prototype.index = 0;
$root.CoreML.Specification.SparseNode.prototype.value = 0;

$root.CoreML.Specification.SparseVector = class SparseVector {

    constructor() {
        this.nodes = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SparseVector();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SparseVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push($root.CoreML.Specification.SparseNode.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SparseSupportVectors();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vectors":
                    message.vectors.push($root.CoreML.Specification.SparseVector.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.DenseVector();
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

$root.CoreML.Specification.DenseSupportVectors = class DenseSupportVectors {

    constructor() {
        this.vectors = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.DenseSupportVectors();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.DenseSupportVectors();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "vectors":
                    message.vectors.push($root.CoreML.Specification.DenseVector.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.Coefficients();
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

$root.CoreML.Specification.SupportVectorRegressor = class SupportVectorRegressor {

    constructor() {
    }

    get supportVectors() {
        $root.CoreML.Specification.SupportVectorRegressor.supportVectorsSet = $root.CoreML.Specification.SupportVectorRegressor.supportVectorsSet || new Set([ "sparseSupportVectors", "denseSupportVectors"]);
        return Object.keys(this).find((key) => $root.CoreML.Specification.SupportVectorRegressor.supportVectorsSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.SupportVectorRegressor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SupportVectorRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.CoreML.Specification.Kernel.decodeText(reader);
                    break;
                case "sparseSupportVectors":
                    message.sparseSupportVectors = $root.CoreML.Specification.SparseSupportVectors.decodeText(reader);
                    break;
                case "denseSupportVectors":
                    message.denseSupportVectors = $root.CoreML.Specification.DenseSupportVectors.decodeText(reader);
                    break;
                case "coefficients":
                    message.coefficients = $root.CoreML.Specification.Coefficients.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.SupportVectorClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.CoreML.Specification.Kernel.decodeText(reader);
                    break;
                case "numberOfSupportVectorsPerClass":
                    reader.array(message.numberOfSupportVectorsPerClass, () => reader.int32());
                    break;
                case "sparseSupportVectors":
                    message.sparseSupportVectors = $root.CoreML.Specification.SparseSupportVectors.decodeText(reader);
                    break;
                case "denseSupportVectors":
                    message.denseSupportVectors = $root.CoreML.Specification.DenseSupportVectors.decodeText(reader);
                    break;
                case "coefficients":
                    message.coefficients.push($root.CoreML.Specification.Coefficients.decodeText(reader));
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
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.TreeEnsembleParameters();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push($root.CoreML.Specification.TreeEnsembleParameters.TreeNode.decodeText(reader));
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

$root.CoreML.Specification.TreeEnsembleParameters.prototype.numPredictionDimensions = protobuf.Uint64.create(0);

$root.CoreML.Specification.TreeEnsembleParameters.TreeNode = class TreeNode {

    constructor() {
        this.evaluationInfo = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode();
        const end = length !== undefined ? reader.position + length : reader.length;
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode();
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
                    message.nodeBehavior = reader.enum($root.CoreML.Specification.TreeEnsembleParameters.TreeNode.TreeNodeBehavior);
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
                    message.evaluationInfo.push($root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo.decodeText(reader));
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo();
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.TreeEnsembleClassifier();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "treeEnsemble":
                    message.treeEnsemble = $root.CoreML.Specification.TreeEnsembleParameters.decodeText(reader);
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum($root.CoreML.Specification.TreeEnsemblePostEvaluationTransform);
                    break;
                case "stringClassLabels":
                    message.stringClassLabels = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "int64ClassLabels":
                    message.int64ClassLabels = $root.CoreML.Specification.Int64Vector.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.TreeEnsembleRegressor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "treeEnsemble":
                    message.treeEnsemble = $root.CoreML.Specification.TreeEnsembleParameters.decodeText(reader);
                    break;
                case "postEvaluationTransform":
                    message.postEvaluationTransform = reader.enum($root.CoreML.Specification.TreeEnsemblePostEvaluationTransform);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "itemItemSimilarities":
                    message.itemItemSimilarities.push($root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems.decodeText(reader));
                    break;
                case "itemStringIds":
                    message.itemStringIds = $root.CoreML.Specification.StringVector.decodeText(reader);
                    break;
                case "itemInt64Ids":
                    message.itemInt64Ids = $root.CoreML.Specification.Int64Vector.decodeText(reader);
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
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem();
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

$root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.prototype.itemId = protobuf.Uint64.create(0);
$root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.prototype.similarityScore = 0;

$root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems = class SimilarItems {

    constructor() {
        this.similarItemList = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "itemId":
                    message.itemId = reader.uint64();
                    break;
                case "similarItemList":
                    message.similarItemList.push($root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem.decodeText(reader));
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LinkedModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linkedModelFile":
                    message.linkedModelFile = $root.CoreML.Specification.LinkedModelFile.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.LinkedModelFile();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "linkedModelFileName":
                    message.linkedModelFileName = $root.CoreML.Specification.StringParameter.decodeText(reader);
                    break;
                case "linkedModelSearchPath":
                    message.linkedModelSearchPath = $root.CoreML.Specification.StringParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.CoreML.Specification.LinkedModelFile.prototype.linkedModelFileName = null;
$root.CoreML.Specification.LinkedModelFile.prototype.linkedModelSearchPath = null;

$root.CoreML.Specification.ClassConfidenceThresholding = class ClassConfidenceThresholding {

    constructor() {
        this.precisionRecallCurves = [];
    }

    static decode(reader, length) {
        const message = new $root.CoreML.Specification.ClassConfidenceThresholding();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 100:
                    message.precisionRecallCurves.push($root.CoreML.Specification.PrecisionRecallCurve.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.CoreML.Specification.ClassConfidenceThresholding();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "precisionRecallCurves":
                    message.precisionRecallCurves.push($root.CoreML.Specification.PrecisionRecallCurve.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};
