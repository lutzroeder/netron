(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('coreml');

    $root.CoreML = (function() {

        const CoreML = {};

        CoreML.Specification = (function() {

            const Specification = {};

            Specification.Pipeline = (function() {

                function Pipeline() {
                    this.models = [];
                    this.names = [];
                }

                Pipeline.prototype.models = [];
                Pipeline.prototype.names = [];

                Pipeline.decode = function (reader, length) {
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
                };

                return Pipeline;
            })();

            Specification.PipelineClassifier = (function() {

                function PipelineClassifier() {
                }

                PipelineClassifier.prototype.pipeline = null;

                PipelineClassifier.decode = function (reader, length) {
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
                };

                return PipelineClassifier;
            })();

            Specification.PipelineRegressor = (function() {

                function PipelineRegressor() {
                }

                PipelineRegressor.prototype.pipeline = null;

                PipelineRegressor.decode = function (reader, length) {
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
                };

                return PipelineRegressor;
            })();

            Specification.FeatureDescription = (function() {

                function FeatureDescription() {
                }

                FeatureDescription.prototype.name = "";
                FeatureDescription.prototype.shortDescription = "";
                FeatureDescription.prototype.type = null;

                FeatureDescription.decode = function (reader, length) {
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
                };

                return FeatureDescription;
            })();

            Specification.Metadata = (function() {

                function Metadata() {
                    this.userDefined = {};
                }

                Metadata.prototype.shortDescription = "";
                Metadata.prototype.versionString = "";
                Metadata.prototype.author = "";
                Metadata.prototype.license = "";
                Metadata.prototype.userDefined = {};

                Metadata.decode = function (reader, length) {
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
                                reader.pair(message.userDefined, () => reader.string(), () => reader.string());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                return Metadata;
            })();

            Specification.ModelDescription = (function() {

                function ModelDescription() {
                    this.input = [];
                    this.output = [];
                    this.trainingInput = [];
                }

                ModelDescription.prototype.input = [];
                ModelDescription.prototype.output = [];
                ModelDescription.prototype.predictedFeatureName = "";
                ModelDescription.prototype.predictedProbabilitiesName = "";
                ModelDescription.prototype.trainingInput = [];
                ModelDescription.prototype.metadata = null;

                ModelDescription.decode = function (reader, length) {
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
                };

                return ModelDescription;
            })();

            Specification.SerializedModel = (function() {

                function SerializedModel() {
                }

                SerializedModel.prototype.identifier = "";
                SerializedModel.prototype.model = new Uint8Array([]);

                SerializedModel.decode = function (reader, length) {
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
                };

                return SerializedModel;
            })();

            Specification.Model = (function() {

                function Model() {
                }

                Model.prototype.specificationVersion = 0;
                Model.prototype.description = null;
                Model.prototype.isUpdatable = false;
                Model.prototype.pipelineClassifier = null;
                Model.prototype.pipelineRegressor = null;
                Model.prototype.pipeline = null;
                Model.prototype.glmRegressor = null;
                Model.prototype.supportVectorRegressor = null;
                Model.prototype.treeEnsembleRegressor = null;
                Model.prototype.neuralNetworkRegressor = null;
                Model.prototype.bayesianProbitRegressor = null;
                Model.prototype.glmClassifier = null;
                Model.prototype.supportVectorClassifier = null;
                Model.prototype.treeEnsembleClassifier = null;
                Model.prototype.neuralNetworkClassifier = null;
                Model.prototype.kNearestNeighborsClassifier = null;
                Model.prototype.neuralNetwork = null;
                Model.prototype.itemSimilarityRecommender = null;
                Model.prototype.customModel = null;
                Model.prototype.linkedModel = null;
                Model.prototype.oneHotEncoder = null;
                Model.prototype.imputer = null;
                Model.prototype.featureVectorizer = null;
                Model.prototype.dictVectorizer = null;
                Model.prototype.scaler = null;
                Model.prototype.categoricalMapping = null;
                Model.prototype.normalizer = null;
                Model.prototype.arrayFeatureExtractor = null;
                Model.prototype.nonMaximumSuppression = null;
                Model.prototype.identity = null;
                Model.prototype.textClassifier = null;
                Model.prototype.wordTagger = null;
                Model.prototype.visionFeaturePrint = null;
                Model.prototype.soundAnalysisPreprocessing = null;
                Model.prototype.gazetteer = null;
                Model.prototype.wordEmbedding = null;
                Model.prototype.serializedModel = null;

                const TypeSet = new Set([ "pipelineClassifier", "pipelineRegressor", "pipeline", "glmRegressor", "supportVectorRegressor", "treeEnsembleRegressor", "neuralNetworkRegressor", "bayesianProbitRegressor", "glmClassifier", "supportVectorClassifier", "treeEnsembleClassifier", "neuralNetworkClassifier", "kNearestNeighborsClassifier", "neuralNetwork", "itemSimilarityRecommender", "customModel", "linkedModel", "oneHotEncoder", "imputer", "featureVectorizer", "dictVectorizer", "scaler", "categoricalMapping", "normalizer", "arrayFeatureExtractor", "nonMaximumSuppression", "identity", "textClassifier", "wordTagger", "visionFeaturePrint", "soundAnalysisPreprocessing", "gazetteer", "wordEmbedding", "serializedModel"]);
                Object.defineProperty(Model.prototype, "Type", {
                    get: function() { return Object.keys(this).find((key) => TypeSet.has(key) && this[key] != null); }
                });

                Model.decode = function (reader, length) {
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
                };

                return Model;
            })();

            Specification.CoreMLModels = (function() {

                const CoreMLModels = {};

                CoreMLModels.VisionFeaturePrint = (function() {

                    function VisionFeaturePrint() {
                    }

                    VisionFeaturePrint.prototype.scene = null;
                    VisionFeaturePrint.prototype.object = null;

                    const VisionFeaturePrintTypeSet = new Set([ "scene", "object"]);
                    Object.defineProperty(VisionFeaturePrint.prototype, "VisionFeaturePrintType", {
                        get: function() { return Object.keys(this).find((key) => VisionFeaturePrintTypeSet.has(key) && this[key] != null); }
                    });

                    VisionFeaturePrint.decode = function (reader, length) {
                        const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint();
                        const end = reader.next(length);
                        while (reader.end(end)) {
                            const tag = reader.uint32();
                            switch (tag >>> 3) {
                                case 20:
                                    message.scene = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.decode(reader, reader.uint32());
                                    break;
                                case 21:
                                    message.object = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Object.decode(reader, reader.uint32());
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                            }
                        }
                        return message;
                    };

                    VisionFeaturePrint.Scene = (function() {

                        function Scene() {
                        }

                        Scene.prototype.version = 0;

                        Scene.decode = function (reader, length) {
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
                        };

                        Scene.SceneVersion = (function() {
                            const values = {};
                            values["SCENE_VERSION_INVALID"] = 0;
                            values["SCENE_VERSION_1"] = 1;
                            return values;
                        })();

                        return Scene;
                    })();

                    VisionFeaturePrint.Object = (function() {

                        function Object() {
                            this.output = [];
                        }

                        Object.prototype.version = 0;
                        Object.prototype.output = [];

                        Object.decode = function (reader, length) {
                            const message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Object();
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
                        };

                        Object.ObjectVersion = (function() {
                            const values = {};
                            values["OBJECT_VERSION_INVALID"] = 0;
                            values["OBJECT_VERSION_1"] = 1;
                            return values;
                        })();

                        return Object;
                    })();

                    return VisionFeaturePrint;
                })();

                CoreMLModels.TextClassifier = (function() {

                    function TextClassifier() {
                    }

                    TextClassifier.prototype.revision = 0;
                    TextClassifier.prototype.language = "";
                    TextClassifier.prototype.modelParameterData = new Uint8Array([]);
                    TextClassifier.prototype.stringClassLabels = null;

                    const ClassLabelsSet = new Set([ "stringClassLabels"]);
                    Object.defineProperty(TextClassifier.prototype, "ClassLabels", {
                        get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                    });

                    TextClassifier.decode = function (reader, length) {
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
                    };

                    return TextClassifier;
                })();

                CoreMLModels.WordTagger = (function() {

                    function WordTagger() {
                    }

                    WordTagger.prototype.revision = 0;
                    WordTagger.prototype.language = "";
                    WordTagger.prototype.tokensOutputFeatureName = "";
                    WordTagger.prototype.tokenTagsOutputFeatureName = "";
                    WordTagger.prototype.tokenLocationsOutputFeatureName = "";
                    WordTagger.prototype.tokenLengthsOutputFeatureName = "";
                    WordTagger.prototype.modelParameterData = new Uint8Array([]);
                    WordTagger.prototype.stringTags = null;

                    const TagsSet = new Set([ "stringTags"]);
                    Object.defineProperty(WordTagger.prototype, "Tags", {
                        get: function() { return Object.keys(this).find((key) => TagsSet.has(key) && this[key] != null); }
                    });

                    WordTagger.decode = function (reader, length) {
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
                    };

                    return WordTagger;
                })();

                CoreMLModels.Gazetteer = (function() {

                    function Gazetteer() {
                    }

                    Gazetteer.prototype.revision = 0;
                    Gazetteer.prototype.language = "";
                    Gazetteer.prototype.modelParameterData = new Uint8Array([]);
                    Gazetteer.prototype.stringClassLabels = null;

                    const ClassLabelsSet = new Set([ "stringClassLabels"]);
                    Object.defineProperty(Gazetteer.prototype, "ClassLabels", {
                        get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                    });

                    Gazetteer.decode = function (reader, length) {
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
                    };

                    return Gazetteer;
                })();

                CoreMLModels.WordEmbedding = (function() {

                    function WordEmbedding() {
                    }

                    WordEmbedding.prototype.revision = 0;
                    WordEmbedding.prototype.language = "";
                    WordEmbedding.prototype.modelParameterData = new Uint8Array([]);

                    WordEmbedding.decode = function (reader, length) {
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
                    };

                    return WordEmbedding;
                })();

                CoreMLModels.SoundAnalysisPreprocessing = (function() {

                    function SoundAnalysisPreprocessing() {
                    }

                    SoundAnalysisPreprocessing.prototype.vggish = null;

                    const SoundAnalysisPreprocessingTypeSet = new Set([ "vggish"]);
                    Object.defineProperty(SoundAnalysisPreprocessing.prototype, "SoundAnalysisPreprocessingType", {
                        get: function() { return Object.keys(this).find((key) => SoundAnalysisPreprocessingTypeSet.has(key) && this[key] != null); }
                    });

                    SoundAnalysisPreprocessing.decode = function (reader, length) {
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
                    };

                    SoundAnalysisPreprocessing.Vggish = (function() {

                        function Vggish() {
                        }

                        Vggish.decode = function (reader, length) {
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
                        };

                        return Vggish;
                    })();

                    return SoundAnalysisPreprocessing;
                })();

                return CoreMLModels;
            })();

            Specification.StringToInt64Map = (function() {

                function StringToInt64Map() {
                    this.map = {};
                }

                StringToInt64Map.prototype.map = {};

                StringToInt64Map.decode = function (reader, length) {
                    const message = new $root.CoreML.Specification.StringToInt64Map();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                reader.pair(message.map, () => reader.string(), () => reader.int64());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                return StringToInt64Map;
            })();

            Specification.Int64ToStringMap = (function() {

                function Int64ToStringMap() {
                    this.map = {};
                }

                Int64ToStringMap.prototype.map = {};

                Int64ToStringMap.decode = function (reader, length) {
                    const message = new $root.CoreML.Specification.Int64ToStringMap();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                reader.pair(message.map, () => reader.int64(), () => reader.string());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                return Int64ToStringMap;
            })();

            Specification.StringToDoubleMap = (function() {

                function StringToDoubleMap() {
                    this.map = {};
                }

                StringToDoubleMap.prototype.map = {};

                StringToDoubleMap.decode = function (reader, length) {
                    const message = new $root.CoreML.Specification.StringToDoubleMap();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                reader.pair(message.map, () => reader.string(), () => reader.double());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                return StringToDoubleMap;
            })();

            Specification.Int64ToDoubleMap = (function() {

                function Int64ToDoubleMap() {
                    this.map = {};
                }

                Int64ToDoubleMap.prototype.map = {};

                Int64ToDoubleMap.decode = function (reader, length) {
                    const message = new $root.CoreML.Specification.Int64ToDoubleMap();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                reader.pair(message.map, () => reader.int64(), () => reader.double());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                return Int64ToDoubleMap;
            })();

            Specification.StringVector = (function() {

                function StringVector() {
                    this.vector = [];
                }

                StringVector.prototype.vector = [];

                StringVector.decode = function (reader, length) {
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
                };

                return StringVector;
            })();

            Specification.Int64Vector = (function() {

                function Int64Vector() {
                    this.vector = [];
                }

                Int64Vector.prototype.vector = [];

                Int64Vector.decode = function (reader, length) {
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
                };

                return Int64Vector;
            })();

            Specification.FloatVector = (function() {

                function FloatVector() {
                    this.vector = [];
                }

                FloatVector.prototype.vector = [];

                FloatVector.decode = function (reader, length) {
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
                };

                return FloatVector;
            })();

            Specification.DoubleVector = (function() {

                function DoubleVector() {
                    this.vector = [];
                }

                DoubleVector.prototype.vector = [];

                DoubleVector.decode = function (reader, length) {
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
                };

                return DoubleVector;
            })();

            Specification.Int64Range = (function() {

                function Int64Range() {
                }

                Int64Range.prototype.minValue = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Int64Range.prototype.maxValue = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                Int64Range.decode = function (reader, length) {
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
                };

                return Int64Range;
            })();

            Specification.Int64Set = (function() {

                function Int64Set() {
                    this.values = [];
                }

                Int64Set.prototype.values = [];

                Int64Set.decode = function (reader, length) {
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
                };

                return Int64Set;
            })();

            Specification.DoubleRange = (function() {

                function DoubleRange() {
                }

                DoubleRange.prototype.minValue = 0;
                DoubleRange.prototype.maxValue = 0;

                DoubleRange.decode = function (reader, length) {
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
                };

                return DoubleRange;
            })();

            Specification.Int64FeatureType = (function() {

                function Int64FeatureType() {
                }

                Int64FeatureType.decode = function (reader, length) {
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
                };

                return Int64FeatureType;
            })();

            Specification.DoubleFeatureType = (function() {

                function DoubleFeatureType() {
                }

                DoubleFeatureType.decode = function (reader, length) {
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
                };

                return DoubleFeatureType;
            })();

            Specification.StringFeatureType = (function() {

                function StringFeatureType() {
                }

                StringFeatureType.decode = function (reader, length) {
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
                };

                return StringFeatureType;
            })();

            Specification.SizeRange = (function() {

                function SizeRange() {
                }

                SizeRange.prototype.lowerBound = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                SizeRange.prototype.upperBound = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                SizeRange.decode = function (reader, length) {
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
                };

                return SizeRange;
            })();

            Specification.ImageFeatureType = (function() {

                function ImageFeatureType() {
                }

                ImageFeatureType.prototype.width = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ImageFeatureType.prototype.height = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ImageFeatureType.prototype.enumeratedSizes = null;
                ImageFeatureType.prototype.imageSizeRange = null;
                ImageFeatureType.prototype.colorSpace = 0;

                const SizeFlexibilitySet = new Set([ "enumeratedSizes", "imageSizeRange"]);
                Object.defineProperty(ImageFeatureType.prototype, "SizeFlexibility", {
                    get: function() { return Object.keys(this).find((key) => SizeFlexibilitySet.has(key) && this[key] != null); }
                });

                ImageFeatureType.decode = function (reader, length) {
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
                };

                ImageFeatureType.ColorSpace = (function() {
                    const values = {};
                    values["INVALID_COLOR_SPACE"] = 0;
                    values["GRAYSCALE"] = 10;
                    values["RGB"] = 20;
                    values["BGR"] = 30;
                    return values;
                })();

                ImageFeatureType.ImageSize = (function() {

                    function ImageSize() {
                    }

                    ImageSize.prototype.width = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    ImageSize.prototype.height = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                    ImageSize.decode = function (reader, length) {
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
                    };

                    return ImageSize;
                })();

                ImageFeatureType.EnumeratedImageSizes = (function() {

                    function EnumeratedImageSizes() {
                        this.sizes = [];
                    }

                    EnumeratedImageSizes.prototype.sizes = [];

                    EnumeratedImageSizes.decode = function (reader, length) {
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
                    };

                    return EnumeratedImageSizes;
                })();

                ImageFeatureType.ImageSizeRange = (function() {

                    function ImageSizeRange() {
                    }

                    ImageSizeRange.prototype.widthRange = null;
                    ImageSizeRange.prototype.heightRange = null;

                    ImageSizeRange.decode = function (reader, length) {
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
                    };

                    return ImageSizeRange;
                })();

                return ImageFeatureType;
            })();

            Specification.ArrayFeatureType = (function() {

                function ArrayFeatureType() {
                    this.shape = [];
                }

                ArrayFeatureType.prototype.shape = [];
                ArrayFeatureType.prototype.dataType = 0;
                ArrayFeatureType.prototype.enumeratedShapes = null;
                ArrayFeatureType.prototype.shapeRange = null;
                ArrayFeatureType.prototype.intDefaultValue = 0;
                ArrayFeatureType.prototype.floatDefaultValue = 0;
                ArrayFeatureType.prototype.doubleDefaultValue = 0;

                const ShapeFlexibilitySet = new Set([ "enumeratedShapes", "shapeRange"]);
                Object.defineProperty(ArrayFeatureType.prototype, "ShapeFlexibility", {
                    get: function() { return Object.keys(this).find((key) => ShapeFlexibilitySet.has(key) && this[key] != null); }
                });

                const defaultOptionalValueSet = new Set([ "intDefaultValue", "floatDefaultValue", "doubleDefaultValue"]);
                Object.defineProperty(ArrayFeatureType.prototype, "defaultOptionalValue", {
                    get: function() { return Object.keys(this).find((key) => defaultOptionalValueSet.has(key) && this[key] != null); }
                });

                ArrayFeatureType.decode = function (reader, length) {
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
                };

                ArrayFeatureType.ArrayDataType = (function() {
                    const values = {};
                    values["INVALID_ARRAY_DATA_TYPE"] = 0;
                    values["FLOAT32"] = 65568;
                    values["DOUBLE"] = 65600;
                    values["INT32"] = 131104;
                    return values;
                })();

                ArrayFeatureType.Shape = (function() {

                    function Shape() {
                        this.shape = [];
                    }

                    Shape.prototype.shape = [];

                    Shape.decode = function (reader, length) {
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
                    };

                    return Shape;
                })();

                ArrayFeatureType.EnumeratedShapes = (function() {

                    function EnumeratedShapes() {
                        this.shapes = [];
                    }

                    EnumeratedShapes.prototype.shapes = [];

                    EnumeratedShapes.decode = function (reader, length) {
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
                    };

                    return EnumeratedShapes;
                })();

                ArrayFeatureType.ShapeRange = (function() {

                    function ShapeRange() {
                        this.sizeRanges = [];
                    }

                    ShapeRange.prototype.sizeRanges = [];

                    ShapeRange.decode = function (reader, length) {
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
                    };

                    return ShapeRange;
                })();

                return ArrayFeatureType;
            })();

            Specification.DictionaryFeatureType = (function() {

                function DictionaryFeatureType() {
                }

                DictionaryFeatureType.prototype.int64KeyType = null;
                DictionaryFeatureType.prototype.stringKeyType = null;

                const KeyTypeSet = new Set([ "int64KeyType", "stringKeyType"]);
                Object.defineProperty(DictionaryFeatureType.prototype, "KeyType", {
                    get: function() { return Object.keys(this).find((key) => KeyTypeSet.has(key) && this[key] != null); }
                });

                DictionaryFeatureType.decode = function (reader, length) {
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
                };

                return DictionaryFeatureType;
            })();

            Specification.SequenceFeatureType = (function() {

                function SequenceFeatureType() {
                }

                SequenceFeatureType.prototype.int64Type = null;
                SequenceFeatureType.prototype.stringType = null;
                SequenceFeatureType.prototype.sizeRange = null;

                const TypeSet = new Set([ "int64Type", "stringType"]);
                Object.defineProperty(SequenceFeatureType.prototype, "Type", {
                    get: function() { return Object.keys(this).find((key) => TypeSet.has(key) && this[key] != null); }
                });

                SequenceFeatureType.decode = function (reader, length) {
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
                };

                return SequenceFeatureType;
            })();

            Specification.FeatureType = (function() {

                function FeatureType() {
                }

                FeatureType.prototype.int64Type = null;
                FeatureType.prototype.doubleType = null;
                FeatureType.prototype.stringType = null;
                FeatureType.prototype.imageType = null;
                FeatureType.prototype.multiArrayType = null;
                FeatureType.prototype.dictionaryType = null;
                FeatureType.prototype.sequenceType = null;
                FeatureType.prototype.isOptional = false;

                const TypeSet = new Set([ "int64Type", "doubleType", "stringType", "imageType", "multiArrayType", "dictionaryType", "sequenceType"]);
                Object.defineProperty(FeatureType.prototype, "Type", {
                    get: function() { return Object.keys(this).find((key) => TypeSet.has(key) && this[key] != null); }
                });

                FeatureType.decode = function (reader, length) {
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
                };

                return FeatureType;
            })();

            Specification.ArrayFeatureExtractor = (function() {

                function ArrayFeatureExtractor() {
                    this.extractIndex = [];
                }

                ArrayFeatureExtractor.prototype.extractIndex = [];

                ArrayFeatureExtractor.decode = function (reader, length) {
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
                };

                return ArrayFeatureExtractor;
            })();

            Specification.BayesianProbitRegressor = (function() {

                function BayesianProbitRegressor() {
                    this.features = [];
                }

                BayesianProbitRegressor.prototype.numberOfFeatures = 0;
                BayesianProbitRegressor.prototype.bias = null;
                BayesianProbitRegressor.prototype.features = [];
                BayesianProbitRegressor.prototype.regressionInputFeatureName = "";
                BayesianProbitRegressor.prototype.optimismInputFeatureName = "";
                BayesianProbitRegressor.prototype.samplingScaleInputFeatureName = "";
                BayesianProbitRegressor.prototype.samplingTruncationInputFeatureName = "";
                BayesianProbitRegressor.prototype.meanOutputFeatureName = "";
                BayesianProbitRegressor.prototype.varianceOutputFeatureName = "";
                BayesianProbitRegressor.prototype.pessimisticProbabilityOutputFeatureName = "";
                BayesianProbitRegressor.prototype.sampledProbabilityOutputFeatureName = "";

                BayesianProbitRegressor.decode = function (reader, length) {
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
                };

                BayesianProbitRegressor.Gaussian = (function() {

                    function Gaussian() {
                    }

                    Gaussian.prototype.mean = 0;
                    Gaussian.prototype.precision = 0;

                    Gaussian.decode = function (reader, length) {
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
                    };

                    return Gaussian;
                })();

                BayesianProbitRegressor.FeatureValueWeight = (function() {

                    function FeatureValueWeight() {
                    }

                    FeatureValueWeight.prototype.featureValue = 0;
                    FeatureValueWeight.prototype.featureWeight = null;

                    FeatureValueWeight.decode = function (reader, length) {
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
                    };

                    return FeatureValueWeight;
                })();

                BayesianProbitRegressor.FeatureWeight = (function() {

                    function FeatureWeight() {
                        this.weights = [];
                    }

                    FeatureWeight.prototype.featureId = 0;
                    FeatureWeight.prototype.weights = [];

                    FeatureWeight.decode = function (reader, length) {
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
                    };

                    return FeatureWeight;
                })();

                return BayesianProbitRegressor;
            })();

            Specification.CategoricalMapping = (function() {

                function CategoricalMapping() {
                }

                CategoricalMapping.prototype.stringToInt64Map = null;
                CategoricalMapping.prototype.int64ToStringMap = null;
                CategoricalMapping.prototype.strValue = "";
                CategoricalMapping.prototype.int64Value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                const MappingTypeSet = new Set([ "stringToInt64Map", "int64ToStringMap"]);
                Object.defineProperty(CategoricalMapping.prototype, "MappingType", {
                    get: function() { return Object.keys(this).find((key) => MappingTypeSet.has(key) && this[key] != null); }
                });

                const ValueOnUnknownSet = new Set([ "strValue", "int64Value"]);
                Object.defineProperty(CategoricalMapping.prototype, "ValueOnUnknown", {
                    get: function() { return Object.keys(this).find((key) => ValueOnUnknownSet.has(key) && this[key] != null); }
                });

                CategoricalMapping.decode = function (reader, length) {
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
                };

                return CategoricalMapping;
            })();

            Specification.CustomModel = (function() {

                function CustomModel() {
                    this.parameters = {};
                }

                CustomModel.prototype.className = "";
                CustomModel.prototype.parameters = {};
                CustomModel.prototype.description = "";

                CustomModel.decode = function (reader, length) {
                    const message = new $root.CoreML.Specification.CustomModel();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 10:
                                message.className = reader.string();
                                break;
                            case 30:
                                reader.pair(message.parameters, () => reader.string(), () => $root.CoreML.Specification.CustomModel.CustomModelParamValue.decode(reader, reader.uint32()));
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
                };

                CustomModel.CustomModelParamValue = (function() {

                    function CustomModelParamValue() {
                    }

                    CustomModelParamValue.prototype.doubleValue = 0;
                    CustomModelParamValue.prototype.stringValue = "";
                    CustomModelParamValue.prototype.intValue = 0;
                    CustomModelParamValue.prototype.longValue = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                    CustomModelParamValue.prototype.boolValue = false;
                    CustomModelParamValue.prototype.bytesValue = new Uint8Array([]);

                    const valueSet = new Set([ "doubleValue", "stringValue", "intValue", "longValue", "boolValue", "bytesValue"]);
                    Object.defineProperty(CustomModelParamValue.prototype, "value", {
                        get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
                    });

                    CustomModelParamValue.decode = function (reader, length) {
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
                    };

                    return CustomModelParamValue;
                })();

                return CustomModel;
            })();

            Specification.DictVectorizer = (function() {

                function DictVectorizer() {
                }

                DictVectorizer.prototype.stringToIndex = null;
                DictVectorizer.prototype.int64ToIndex = null;

                const MapSet = new Set([ "stringToIndex", "int64ToIndex"]);
                Object.defineProperty(DictVectorizer.prototype, "Map", {
                    get: function() { return Object.keys(this).find((key) => MapSet.has(key) && this[key] != null); }
                });

                DictVectorizer.decode = function (reader, length) {
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
                };

                return DictVectorizer;
            })();

            Specification.FeatureVectorizer = (function() {

                function FeatureVectorizer() {
                    this.inputList = [];
                }

                FeatureVectorizer.prototype.inputList = [];

                FeatureVectorizer.decode = function (reader, length) {
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
                };

                FeatureVectorizer.InputColumn = (function() {

                    function InputColumn() {
                    }

                    InputColumn.prototype.inputColumn = "";
                    InputColumn.prototype.inputDimensions = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                    InputColumn.decode = function (reader, length) {
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
                    };

                    return InputColumn;
                })();

                return FeatureVectorizer;
            })();

            Specification.GLMRegressor = (function() {

                function GLMRegressor() {
                    this.weights = [];
                    this.offset = [];
                }

                GLMRegressor.prototype.weights = [];
                GLMRegressor.prototype.offset = [];
                GLMRegressor.prototype.postEvaluationTransform = 0;

                GLMRegressor.decode = function (reader, length) {
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
                };

                GLMRegressor.DoubleArray = (function() {

                    function DoubleArray() {
                        this.value = [];
                    }

                    DoubleArray.prototype.value = [];

                    DoubleArray.decode = function (reader, length) {
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
                    };

                    return DoubleArray;
                })();

                GLMRegressor.PostEvaluationTransform = (function() {
                    const values = {};
                    values["NoTransform"] = 0;
                    values["Logit"] = 1;
                    values["Probit"] = 2;
                    return values;
                })();

                return GLMRegressor;
            })();

            Specification.GLMClassifier = (function() {

                function GLMClassifier() {
                    this.weights = [];
                    this.offset = [];
                }

                GLMClassifier.prototype.weights = [];
                GLMClassifier.prototype.offset = [];
                GLMClassifier.prototype.postEvaluationTransform = 0;
                GLMClassifier.prototype.classEncoding = 0;
                GLMClassifier.prototype.stringClassLabels = null;
                GLMClassifier.prototype.int64ClassLabels = null;

                const ClassLabelsSet = new Set([ "stringClassLabels", "int64ClassLabels"]);
                Object.defineProperty(GLMClassifier.prototype, "ClassLabels", {
                    get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                });

                GLMClassifier.decode = function (reader, length) {
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
                };

                GLMClassifier.DoubleArray = (function() {

                    function DoubleArray() {
                        this.value = [];
                    }

                    DoubleArray.prototype.value = [];

                    DoubleArray.decode = function (reader, length) {
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
                    };

                    return DoubleArray;
                })();

                GLMClassifier.PostEvaluationTransform = (function() {
                    const values = {};
                    values["Logit"] = 0;
                    values["Probit"] = 1;
                    return values;
                })();

                GLMClassifier.ClassEncoding = (function() {
                    const values = {};
                    values["ReferenceClass"] = 0;
                    values["OneVsRest"] = 1;
                    return values;
                })();

                return GLMClassifier;
            })();

            Specification.KNearestNeighborsClassifier = (function() {

                function KNearestNeighborsClassifier() {
                }

                KNearestNeighborsClassifier.prototype.nearestNeighborsIndex = null;
                KNearestNeighborsClassifier.prototype.numberOfNeighbors = null;
                KNearestNeighborsClassifier.prototype.stringClassLabels = null;
                KNearestNeighborsClassifier.prototype.int64ClassLabels = null;
                KNearestNeighborsClassifier.prototype.defaultStringLabel = "";
                KNearestNeighborsClassifier.prototype.defaultInt64Label = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                KNearestNeighborsClassifier.prototype.uniformWeighting = null;
                KNearestNeighborsClassifier.prototype.inverseDistanceWeighting = null;

                const ClassLabelsSet = new Set([ "stringClassLabels", "int64ClassLabels"]);
                Object.defineProperty(KNearestNeighborsClassifier.prototype, "ClassLabels", {
                    get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                });

                const DefaultClassLabelSet = new Set([ "defaultStringLabel", "defaultInt64Label"]);
                Object.defineProperty(KNearestNeighborsClassifier.prototype, "DefaultClassLabel", {
                    get: function() { return Object.keys(this).find((key) => DefaultClassLabelSet.has(key) && this[key] != null); }
                });

                const WeightingSchemeSet = new Set([ "uniformWeighting", "inverseDistanceWeighting"]);
                Object.defineProperty(KNearestNeighborsClassifier.prototype, "WeightingScheme", {
                    get: function() { return Object.keys(this).find((key) => WeightingSchemeSet.has(key) && this[key] != null); }
                });

                KNearestNeighborsClassifier.decode = function (reader, length) {
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
                };

                return KNearestNeighborsClassifier;
            })();

            Specification.NearestNeighborsIndex = (function() {

                function NearestNeighborsIndex() {
                    this.floatSamples = [];
                }

                NearestNeighborsIndex.prototype.numberOfDimensions = 0;
                NearestNeighborsIndex.prototype.floatSamples = [];
                NearestNeighborsIndex.prototype.linearIndex = null;
                NearestNeighborsIndex.prototype.singleKdTreeIndex = null;
                NearestNeighborsIndex.prototype.squaredEuclideanDistance = null;

                const IndexTypeSet = new Set([ "linearIndex", "singleKdTreeIndex"]);
                Object.defineProperty(NearestNeighborsIndex.prototype, "IndexType", {
                    get: function() { return Object.keys(this).find((key) => IndexTypeSet.has(key) && this[key] != null); }
                });

                const DistanceFunctionSet = new Set([ "squaredEuclideanDistance"]);
                Object.defineProperty(NearestNeighborsIndex.prototype, "DistanceFunction", {
                    get: function() { return Object.keys(this).find((key) => DistanceFunctionSet.has(key) && this[key] != null); }
                });

                NearestNeighborsIndex.decode = function (reader, length) {
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
                };

                return NearestNeighborsIndex;
            })();

            Specification.UniformWeighting = (function() {

                function UniformWeighting() {
                }

                UniformWeighting.decode = function (reader, length) {
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
                };

                return UniformWeighting;
            })();

            Specification.InverseDistanceWeighting = (function() {

                function InverseDistanceWeighting() {
                }

                InverseDistanceWeighting.decode = function (reader, length) {
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
                };

                return InverseDistanceWeighting;
            })();

            Specification.LinearIndex = (function() {

                function LinearIndex() {
                }

                LinearIndex.decode = function (reader, length) {
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
                };

                return LinearIndex;
            })();

            Specification.SingleKdTreeIndex = (function() {

                function SingleKdTreeIndex() {
                }

                SingleKdTreeIndex.prototype.leafSize = 0;

                SingleKdTreeIndex.decode = function (reader, length) {
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
                };

                return SingleKdTreeIndex;
            })();

            Specification.SquaredEuclideanDistance = (function() {

                function SquaredEuclideanDistance() {
                }

                SquaredEuclideanDistance.decode = function (reader, length) {
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
                };

                return SquaredEuclideanDistance;
            })();

            Specification.Int64Parameter = (function() {

                function Int64Parameter() {
                }

                Int64Parameter.prototype.defaultValue = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Int64Parameter.prototype.range = null;
                Int64Parameter.prototype.set = null;

                const AllowedValuesSet = new Set([ "range", "set"]);
                Object.defineProperty(Int64Parameter.prototype, "AllowedValues", {
                    get: function() { return Object.keys(this).find((key) => AllowedValuesSet.has(key) && this[key] != null); }
                });

                Int64Parameter.decode = function (reader, length) {
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
                };

                return Int64Parameter;
            })();

            Specification.DoubleParameter = (function() {

                function DoubleParameter() {
                }

                DoubleParameter.prototype.defaultValue = 0;
                DoubleParameter.prototype.range = null;

                const AllowedValuesSet = new Set([ "range"]);
                Object.defineProperty(DoubleParameter.prototype, "AllowedValues", {
                    get: function() { return Object.keys(this).find((key) => AllowedValuesSet.has(key) && this[key] != null); }
                });

                DoubleParameter.decode = function (reader, length) {
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
                };

                return DoubleParameter;
            })();

            Specification.StringParameter = (function() {

                function StringParameter() {
                }

                StringParameter.prototype.defaultValue = "";

                StringParameter.decode = function (reader, length) {
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
                };

                return StringParameter;
            })();

            Specification.BoolParameter = (function() {

                function BoolParameter() {
                }

                BoolParameter.prototype.defaultValue = false;

                BoolParameter.decode = function (reader, length) {
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
                };

                return BoolParameter;
            })();

            Specification.Identity = (function() {

                function Identity() {
                }

                Identity.decode = function (reader, length) {
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
                };

                return Identity;
            })();

            Specification.Imputer = (function() {

                function Imputer() {
                }

                Imputer.prototype.imputedDoubleValue = 0;
                Imputer.prototype.imputedInt64Value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Imputer.prototype.imputedStringValue = "";
                Imputer.prototype.imputedDoubleArray = null;
                Imputer.prototype.imputedInt64Array = null;
                Imputer.prototype.imputedStringDictionary = null;
                Imputer.prototype.imputedInt64Dictionary = null;
                Imputer.prototype.replaceDoubleValue = 0;
                Imputer.prototype.replaceInt64Value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Imputer.prototype.replaceStringValue = "";

                const ImputedValueSet = new Set([ "imputedDoubleValue", "imputedInt64Value", "imputedStringValue", "imputedDoubleArray", "imputedInt64Array", "imputedStringDictionary", "imputedInt64Dictionary"]);
                Object.defineProperty(Imputer.prototype, "ImputedValue", {
                    get: function() { return Object.keys(this).find((key) => ImputedValueSet.has(key) && this[key] != null); }
                });

                const ReplaceValueSet = new Set([ "replaceDoubleValue", "replaceInt64Value", "replaceStringValue"]);
                Object.defineProperty(Imputer.prototype, "ReplaceValue", {
                    get: function() { return Object.keys(this).find((key) => ReplaceValueSet.has(key) && this[key] != null); }
                });

                Imputer.decode = function (reader, length) {
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
                };

                return Imputer;
            })();

            Specification.NeuralNetworkMultiArrayShapeMapping = (function() {
                const values = {};
                values["RANK5_ARRAY_MAPPING"] = 0;
                values["EXACT_ARRAY_MAPPING"] = 1;
                return values;
            })();

            Specification.NeuralNetworkImageShapeMapping = (function() {
                const values = {};
                values["RANK5_IMAGE_MAPPING"] = 0;
                values["RANK4_IMAGE_MAPPING"] = 1;
                return values;
            })();

            Specification.NeuralNetwork = (function() {

                function NeuralNetwork() {
                    this.layers = [];
                    this.preprocessing = [];
                }

                NeuralNetwork.prototype.layers = [];
                NeuralNetwork.prototype.preprocessing = [];
                NeuralNetwork.prototype.arrayInputShapeMapping = 0;
                NeuralNetwork.prototype.imageInputShapeMapping = 0;
                NeuralNetwork.prototype.updateParams = null;

                NeuralNetwork.decode = function (reader, length) {
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
                };

                return NeuralNetwork;
            })();

            Specification.NeuralNetworkImageScaler = (function() {

                function NeuralNetworkImageScaler() {
                }

                NeuralNetworkImageScaler.prototype.channelScale = 0;
                NeuralNetworkImageScaler.prototype.blueBias = 0;
                NeuralNetworkImageScaler.prototype.greenBias = 0;
                NeuralNetworkImageScaler.prototype.redBias = 0;
                NeuralNetworkImageScaler.prototype.grayBias = 0;

                NeuralNetworkImageScaler.decode = function (reader, length) {
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
                };

                return NeuralNetworkImageScaler;
            })();

            Specification.NeuralNetworkMeanImage = (function() {

                function NeuralNetworkMeanImage() {
                    this.meanImage = [];
                }

                NeuralNetworkMeanImage.prototype.meanImage = [];

                NeuralNetworkMeanImage.decode = function (reader, length) {
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
                };

                return NeuralNetworkMeanImage;
            })();

            Specification.NeuralNetworkPreprocessing = (function() {

                function NeuralNetworkPreprocessing() {
                }

                NeuralNetworkPreprocessing.prototype.featureName = "";
                NeuralNetworkPreprocessing.prototype.scaler = null;
                NeuralNetworkPreprocessing.prototype.meanImage = null;

                const preprocessorSet = new Set([ "scaler", "meanImage"]);
                Object.defineProperty(NeuralNetworkPreprocessing.prototype, "preprocessor", {
                    get: function() { return Object.keys(this).find((key) => preprocessorSet.has(key) && this[key] != null); }
                });

                NeuralNetworkPreprocessing.decode = function (reader, length) {
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
                };

                return NeuralNetworkPreprocessing;
            })();

            Specification.ActivationReLU = (function() {

                function ActivationReLU() {
                }

                ActivationReLU.decode = function (reader, length) {
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
                };

                return ActivationReLU;
            })();

            Specification.ActivationLeakyReLU = (function() {

                function ActivationLeakyReLU() {
                }

                ActivationLeakyReLU.prototype.alpha = 0;

                ActivationLeakyReLU.decode = function (reader, length) {
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
                };

                return ActivationLeakyReLU;
            })();

            Specification.ActivationTanh = (function() {

                function ActivationTanh() {
                }

                ActivationTanh.decode = function (reader, length) {
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
                };

                return ActivationTanh;
            })();

            Specification.ActivationScaledTanh = (function() {

                function ActivationScaledTanh() {
                }

                ActivationScaledTanh.prototype.alpha = 0;
                ActivationScaledTanh.prototype.beta = 0;

                ActivationScaledTanh.decode = function (reader, length) {
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
                };

                return ActivationScaledTanh;
            })();

            Specification.ActivationSigmoid = (function() {

                function ActivationSigmoid() {
                }

                ActivationSigmoid.decode = function (reader, length) {
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
                };

                return ActivationSigmoid;
            })();

            Specification.ActivationLinear = (function() {

                function ActivationLinear() {
                }

                ActivationLinear.prototype.alpha = 0;
                ActivationLinear.prototype.beta = 0;

                ActivationLinear.decode = function (reader, length) {
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
                };

                return ActivationLinear;
            })();

            Specification.ActivationSigmoidHard = (function() {

                function ActivationSigmoidHard() {
                }

                ActivationSigmoidHard.prototype.alpha = 0;
                ActivationSigmoidHard.prototype.beta = 0;

                ActivationSigmoidHard.decode = function (reader, length) {
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
                };

                return ActivationSigmoidHard;
            })();

            Specification.ActivationPReLU = (function() {

                function ActivationPReLU() {
                }

                ActivationPReLU.prototype.alpha = null;

                ActivationPReLU.decode = function (reader, length) {
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
                };

                return ActivationPReLU;
            })();

            Specification.ActivationELU = (function() {

                function ActivationELU() {
                }

                ActivationELU.prototype.alpha = 0;

                ActivationELU.decode = function (reader, length) {
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
                };

                return ActivationELU;
            })();

            Specification.ActivationThresholdedReLU = (function() {

                function ActivationThresholdedReLU() {
                }

                ActivationThresholdedReLU.prototype.alpha = 0;

                ActivationThresholdedReLU.decode = function (reader, length) {
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
                };

                return ActivationThresholdedReLU;
            })();

            Specification.ActivationSoftsign = (function() {

                function ActivationSoftsign() {
                }

                ActivationSoftsign.decode = function (reader, length) {
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
                };

                return ActivationSoftsign;
            })();

            Specification.ActivationSoftplus = (function() {

                function ActivationSoftplus() {
                }

                ActivationSoftplus.decode = function (reader, length) {
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
                };

                return ActivationSoftplus;
            })();

            Specification.ActivationParametricSoftplus = (function() {

                function ActivationParametricSoftplus() {
                }

                ActivationParametricSoftplus.prototype.alpha = null;
                ActivationParametricSoftplus.prototype.beta = null;

                ActivationParametricSoftplus.decode = function (reader, length) {
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
                };

                return ActivationParametricSoftplus;
            })();

            Specification.ActivationParams = (function() {

                function ActivationParams() {
                }

                ActivationParams.prototype.linear = null;
                ActivationParams.prototype.ReLU = null;
                ActivationParams.prototype.leakyReLU = null;
                ActivationParams.prototype.thresholdedReLU = null;
                ActivationParams.prototype.PReLU = null;
                ActivationParams.prototype.tanh = null;
                ActivationParams.prototype.scaledTanh = null;
                ActivationParams.prototype.sigmoid = null;
                ActivationParams.prototype.sigmoidHard = null;
                ActivationParams.prototype.ELU = null;
                ActivationParams.prototype.softsign = null;
                ActivationParams.prototype.softplus = null;
                ActivationParams.prototype.parametricSoftplus = null;

                const NonlinearityTypeSet = new Set([ "linear", "ReLU", "leakyReLU", "thresholdedReLU", "PReLU", "tanh", "scaledTanh", "sigmoid", "sigmoidHard", "ELU", "softsign", "softplus", "parametricSoftplus"]);
                Object.defineProperty(ActivationParams.prototype, "NonlinearityType", {
                    get: function() { return Object.keys(this).find((key) => NonlinearityTypeSet.has(key) && this[key] != null); }
                });

                ActivationParams.decode = function (reader, length) {
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
                };

                return ActivationParams;
            })();

            Specification.Tensor = (function() {

                function Tensor() {
                    this.dimValue = [];
                }

                Tensor.prototype.rank = 0;
                Tensor.prototype.dimValue = [];

                Tensor.decode = function (reader, length) {
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
                };

                return Tensor;
            })();

            Specification.NeuralNetworkLayer = (function() {

                function NeuralNetworkLayer() {
                    this.input = [];
                    this.output = [];
                    this.inputTensor = [];
                    this.outputTensor = [];
                }

                NeuralNetworkLayer.prototype.name = "";
                NeuralNetworkLayer.prototype.input = [];
                NeuralNetworkLayer.prototype.output = [];
                NeuralNetworkLayer.prototype.inputTensor = [];
                NeuralNetworkLayer.prototype.outputTensor = [];
                NeuralNetworkLayer.prototype.isUpdatable = false;
                NeuralNetworkLayer.prototype.convolution = null;
                NeuralNetworkLayer.prototype.pooling = null;
                NeuralNetworkLayer.prototype.activation = null;
                NeuralNetworkLayer.prototype.innerProduct = null;
                NeuralNetworkLayer.prototype.embedding = null;
                NeuralNetworkLayer.prototype.batchnorm = null;
                NeuralNetworkLayer.prototype.mvn = null;
                NeuralNetworkLayer.prototype.l2normalize = null;
                NeuralNetworkLayer.prototype.softmax = null;
                NeuralNetworkLayer.prototype.lrn = null;
                NeuralNetworkLayer.prototype.crop = null;
                NeuralNetworkLayer.prototype.padding = null;
                NeuralNetworkLayer.prototype.upsample = null;
                NeuralNetworkLayer.prototype.resizeBilinear = null;
                NeuralNetworkLayer.prototype.cropResize = null;
                NeuralNetworkLayer.prototype.unary = null;
                NeuralNetworkLayer.prototype.add = null;
                NeuralNetworkLayer.prototype.multiply = null;
                NeuralNetworkLayer.prototype.average = null;
                NeuralNetworkLayer.prototype.scale = null;
                NeuralNetworkLayer.prototype.bias = null;
                NeuralNetworkLayer.prototype.max = null;
                NeuralNetworkLayer.prototype.min = null;
                NeuralNetworkLayer.prototype.dot = null;
                NeuralNetworkLayer.prototype.reduce = null;
                NeuralNetworkLayer.prototype.loadConstant = null;
                NeuralNetworkLayer.prototype.reshape = null;
                NeuralNetworkLayer.prototype.flatten = null;
                NeuralNetworkLayer.prototype.permute = null;
                NeuralNetworkLayer.prototype.concat = null;
                NeuralNetworkLayer.prototype.split = null;
                NeuralNetworkLayer.prototype.sequenceRepeat = null;
                NeuralNetworkLayer.prototype.reorganizeData = null;
                NeuralNetworkLayer.prototype.slice = null;
                NeuralNetworkLayer.prototype.simpleRecurrent = null;
                NeuralNetworkLayer.prototype.gru = null;
                NeuralNetworkLayer.prototype.uniDirectionalLSTM = null;
                NeuralNetworkLayer.prototype.biDirectionalLSTM = null;
                NeuralNetworkLayer.prototype.custom = null;
                NeuralNetworkLayer.prototype.copy = null;
                NeuralNetworkLayer.prototype.branch = null;
                NeuralNetworkLayer.prototype.loop = null;
                NeuralNetworkLayer.prototype.loopBreak = null;
                NeuralNetworkLayer.prototype.loopContinue = null;
                NeuralNetworkLayer.prototype.rangeStatic = null;
                NeuralNetworkLayer.prototype.rangeDynamic = null;
                NeuralNetworkLayer.prototype.clip = null;
                NeuralNetworkLayer.prototype.ceil = null;
                NeuralNetworkLayer.prototype.floor = null;
                NeuralNetworkLayer.prototype.sign = null;
                NeuralNetworkLayer.prototype.round = null;
                NeuralNetworkLayer.prototype.exp2 = null;
                NeuralNetworkLayer.prototype.sin = null;
                NeuralNetworkLayer.prototype.cos = null;
                NeuralNetworkLayer.prototype.tan = null;
                NeuralNetworkLayer.prototype.asin = null;
                NeuralNetworkLayer.prototype.acos = null;
                NeuralNetworkLayer.prototype.atan = null;
                NeuralNetworkLayer.prototype.sinh = null;
                NeuralNetworkLayer.prototype.cosh = null;
                NeuralNetworkLayer.prototype.tanh = null;
                NeuralNetworkLayer.prototype.asinh = null;
                NeuralNetworkLayer.prototype.acosh = null;
                NeuralNetworkLayer.prototype.atanh = null;
                NeuralNetworkLayer.prototype.erf = null;
                NeuralNetworkLayer.prototype.gelu = null;
                NeuralNetworkLayer.prototype.equal = null;
                NeuralNetworkLayer.prototype.notEqual = null;
                NeuralNetworkLayer.prototype.lessThan = null;
                NeuralNetworkLayer.prototype.lessEqual = null;
                NeuralNetworkLayer.prototype.greaterThan = null;
                NeuralNetworkLayer.prototype.greaterEqual = null;
                NeuralNetworkLayer.prototype.logicalOr = null;
                NeuralNetworkLayer.prototype.logicalXor = null;
                NeuralNetworkLayer.prototype.logicalNot = null;
                NeuralNetworkLayer.prototype.logicalAnd = null;
                NeuralNetworkLayer.prototype.modBroadcastable = null;
                NeuralNetworkLayer.prototype.minBroadcastable = null;
                NeuralNetworkLayer.prototype.maxBroadcastable = null;
                NeuralNetworkLayer.prototype.addBroadcastable = null;
                NeuralNetworkLayer.prototype.powBroadcastable = null;
                NeuralNetworkLayer.prototype.divideBroadcastable = null;
                NeuralNetworkLayer.prototype.floorDivBroadcastable = null;
                NeuralNetworkLayer.prototype.multiplyBroadcastable = null;
                NeuralNetworkLayer.prototype.subtractBroadcastable = null;
                NeuralNetworkLayer.prototype.tile = null;
                NeuralNetworkLayer.prototype.stack = null;
                NeuralNetworkLayer.prototype.gather = null;
                NeuralNetworkLayer.prototype.scatter = null;
                NeuralNetworkLayer.prototype.gatherND = null;
                NeuralNetworkLayer.prototype.scatterND = null;
                NeuralNetworkLayer.prototype.softmaxND = null;
                NeuralNetworkLayer.prototype.gatherAlongAxis = null;
                NeuralNetworkLayer.prototype.scatterAlongAxis = null;
                NeuralNetworkLayer.prototype.reverse = null;
                NeuralNetworkLayer.prototype.reverseSeq = null;
                NeuralNetworkLayer.prototype.splitND = null;
                NeuralNetworkLayer.prototype.concatND = null;
                NeuralNetworkLayer.prototype.transpose = null;
                NeuralNetworkLayer.prototype.sliceStatic = null;
                NeuralNetworkLayer.prototype.sliceDynamic = null;
                NeuralNetworkLayer.prototype.slidingWindows = null;
                NeuralNetworkLayer.prototype.topK = null;
                NeuralNetworkLayer.prototype.argMin = null;
                NeuralNetworkLayer.prototype.argMax = null;
                NeuralNetworkLayer.prototype.embeddingND = null;
                NeuralNetworkLayer.prototype.batchedMatmul = null;
                NeuralNetworkLayer.prototype.getShape = null;
                NeuralNetworkLayer.prototype.loadConstantND = null;
                NeuralNetworkLayer.prototype.fillLike = null;
                NeuralNetworkLayer.prototype.fillStatic = null;
                NeuralNetworkLayer.prototype.fillDynamic = null;
                NeuralNetworkLayer.prototype.broadcastToLike = null;
                NeuralNetworkLayer.prototype.broadcastToStatic = null;
                NeuralNetworkLayer.prototype.broadcastToDynamic = null;
                NeuralNetworkLayer.prototype.squeeze = null;
                NeuralNetworkLayer.prototype.expandDims = null;
                NeuralNetworkLayer.prototype.flattenTo2D = null;
                NeuralNetworkLayer.prototype.reshapeLike = null;
                NeuralNetworkLayer.prototype.reshapeStatic = null;
                NeuralNetworkLayer.prototype.reshapeDynamic = null;
                NeuralNetworkLayer.prototype.rankPreservingReshape = null;
                NeuralNetworkLayer.prototype.constantPad = null;
                NeuralNetworkLayer.prototype.randomNormalLike = null;
                NeuralNetworkLayer.prototype.randomNormalStatic = null;
                NeuralNetworkLayer.prototype.randomNormalDynamic = null;
                NeuralNetworkLayer.prototype.randomUniformLike = null;
                NeuralNetworkLayer.prototype.randomUniformStatic = null;
                NeuralNetworkLayer.prototype.randomUniformDynamic = null;
                NeuralNetworkLayer.prototype.randomBernoulliLike = null;
                NeuralNetworkLayer.prototype.randomBernoulliStatic = null;
                NeuralNetworkLayer.prototype.randomBernoulliDynamic = null;
                NeuralNetworkLayer.prototype.categoricalDistribution = null;
                NeuralNetworkLayer.prototype.reduceL1 = null;
                NeuralNetworkLayer.prototype.reduceL2 = null;
                NeuralNetworkLayer.prototype.reduceMax = null;
                NeuralNetworkLayer.prototype.reduceMin = null;
                NeuralNetworkLayer.prototype.reduceSum = null;
                NeuralNetworkLayer.prototype.reduceProd = null;
                NeuralNetworkLayer.prototype.reduceMean = null;
                NeuralNetworkLayer.prototype.reduceLogSum = null;
                NeuralNetworkLayer.prototype.reduceSumSquare = null;
                NeuralNetworkLayer.prototype.reduceLogSumExp = null;
                NeuralNetworkLayer.prototype.whereNonZero = null;
                NeuralNetworkLayer.prototype.matrixBandPart = null;
                NeuralNetworkLayer.prototype.lowerTriangular = null;
                NeuralNetworkLayer.prototype.upperTriangular = null;
                NeuralNetworkLayer.prototype.whereBroadcastable = null;
                NeuralNetworkLayer.prototype.layerNormalization = null;
                NeuralNetworkLayer.prototype.NonMaximumSuppression = null;
                NeuralNetworkLayer.prototype.oneHot = null;
                NeuralNetworkLayer.prototype.cumSum = null;
                NeuralNetworkLayer.prototype.clampedReLU = null;
                NeuralNetworkLayer.prototype.argSort = null;
                NeuralNetworkLayer.prototype.pooling3d = null;
                NeuralNetworkLayer.prototype.globalPooling3d = null;
                NeuralNetworkLayer.prototype.sliceBySize = null;
                NeuralNetworkLayer.prototype.convolution3d = null;

                const layerSet = new Set([ "convolution", "pooling", "activation", "innerProduct", "embedding", "batchnorm", "mvn", "l2normalize", "softmax", "lrn", "crop", "padding", "upsample", "resizeBilinear", "cropResize", "unary", "add", "multiply", "average", "scale", "bias", "max", "min", "dot", "reduce", "loadConstant", "reshape", "flatten", "permute", "concat", "split", "sequenceRepeat", "reorganizeData", "slice", "simpleRecurrent", "gru", "uniDirectionalLSTM", "biDirectionalLSTM", "custom", "copy", "branch", "loop", "loopBreak", "loopContinue", "rangeStatic", "rangeDynamic", "clip", "ceil", "floor", "sign", "round", "exp2", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "gelu", "equal", "notEqual", "lessThan", "lessEqual", "greaterThan", "greaterEqual", "logicalOr", "logicalXor", "logicalNot", "logicalAnd", "modBroadcastable", "minBroadcastable", "maxBroadcastable", "addBroadcastable", "powBroadcastable", "divideBroadcastable", "floorDivBroadcastable", "multiplyBroadcastable", "subtractBroadcastable", "tile", "stack", "gather", "scatter", "gatherND", "scatterND", "softmaxND", "gatherAlongAxis", "scatterAlongAxis", "reverse", "reverseSeq", "splitND", "concatND", "transpose", "sliceStatic", "sliceDynamic", "slidingWindows", "topK", "argMin", "argMax", "embeddingND", "batchedMatmul", "getShape", "loadConstantND", "fillLike", "fillStatic", "fillDynamic", "broadcastToLike", "broadcastToStatic", "broadcastToDynamic", "squeeze", "expandDims", "flattenTo2D", "reshapeLike", "reshapeStatic", "reshapeDynamic", "rankPreservingReshape", "constantPad", "randomNormalLike", "randomNormalStatic", "randomNormalDynamic", "randomUniformLike", "randomUniformStatic", "randomUniformDynamic", "randomBernoulliLike", "randomBernoulliStatic", "randomBernoulliDynamic", "categoricalDistribution", "reduceL1", "reduceL2", "reduceMax", "reduceMin", "reduceSum", "reduceProd", "reduceMean", "reduceLogSum", "reduceSumSquare", "reduceLogSumExp", "whereNonZero", "matrixBandPart", "lowerTriangular", "upperTriangular", "whereBroadcastable", "layerNormalization", "NonMaximumSuppression", "oneHot", "cumSum", "clampedReLU", "argSort", "pooling3d", "globalPooling3d", "sliceBySize", "convolution3d"]);
                Object.defineProperty(NeuralNetworkLayer.prototype, "layer", {
                    get: function() { return Object.keys(this).find((key) => layerSet.has(key) && this[key] != null); }
                });

                NeuralNetworkLayer.decode = function (reader, length) {
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
                };

                return NeuralNetworkLayer;
            })();

            Specification.BranchLayerParams = (function() {

                function BranchLayerParams() {
                }

                BranchLayerParams.prototype.ifBranch = null;
                BranchLayerParams.prototype.elseBranch = null;

                BranchLayerParams.decode = function (reader, length) {
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
                };

                return BranchLayerParams;
            })();

            Specification.LoopLayerParams = (function() {

                function LoopLayerParams() {
                }

                LoopLayerParams.prototype.maxLoopIterations = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                LoopLayerParams.prototype.conditionVar = "";
                LoopLayerParams.prototype.conditionNetwork = null;
                LoopLayerParams.prototype.bodyNetwork = null;

                LoopLayerParams.decode = function (reader, length) {
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
                };

                return LoopLayerParams;
            })();

            Specification.LoopBreakLayerParams = (function() {

                function LoopBreakLayerParams() {
                }

                LoopBreakLayerParams.decode = function (reader, length) {
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
                };

                return LoopBreakLayerParams;
            })();

            Specification.LoopContinueLayerParams = (function() {

                function LoopContinueLayerParams() {
                }

                LoopContinueLayerParams.decode = function (reader, length) {
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
                };

                return LoopContinueLayerParams;
            })();

            Specification.CopyLayerParams = (function() {

                function CopyLayerParams() {
                }

                CopyLayerParams.decode = function (reader, length) {
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
                };

                return CopyLayerParams;
            })();

            Specification.GreaterThanLayerParams = (function() {

                function GreaterThanLayerParams() {
                }

                GreaterThanLayerParams.prototype.alpha = 0;

                GreaterThanLayerParams.decode = function (reader, length) {
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
                };

                return GreaterThanLayerParams;
            })();

            Specification.GreaterEqualLayerParams = (function() {

                function GreaterEqualLayerParams() {
                }

                GreaterEqualLayerParams.prototype.alpha = 0;

                GreaterEqualLayerParams.decode = function (reader, length) {
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
                };

                return GreaterEqualLayerParams;
            })();

            Specification.LessThanLayerParams = (function() {

                function LessThanLayerParams() {
                }

                LessThanLayerParams.prototype.alpha = 0;

                LessThanLayerParams.decode = function (reader, length) {
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
                };

                return LessThanLayerParams;
            })();

            Specification.LessEqualLayerParams = (function() {

                function LessEqualLayerParams() {
                }

                LessEqualLayerParams.prototype.alpha = 0;

                LessEqualLayerParams.decode = function (reader, length) {
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
                };

                return LessEqualLayerParams;
            })();

            Specification.EqualLayerParams = (function() {

                function EqualLayerParams() {
                }

                EqualLayerParams.prototype.alpha = 0;

                EqualLayerParams.decode = function (reader, length) {
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
                };

                return EqualLayerParams;
            })();

            Specification.NotEqualLayerParams = (function() {

                function NotEqualLayerParams() {
                }

                NotEqualLayerParams.prototype.alpha = 0;

                NotEqualLayerParams.decode = function (reader, length) {
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
                };

                return NotEqualLayerParams;
            })();

            Specification.LogicalAndLayerParams = (function() {

                function LogicalAndLayerParams() {
                }

                LogicalAndLayerParams.decode = function (reader, length) {
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
                };

                return LogicalAndLayerParams;
            })();

            Specification.LogicalOrLayerParams = (function() {

                function LogicalOrLayerParams() {
                }

                LogicalOrLayerParams.decode = function (reader, length) {
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
                };

                return LogicalOrLayerParams;
            })();

            Specification.LogicalXorLayerParams = (function() {

                function LogicalXorLayerParams() {
                }

                LogicalXorLayerParams.decode = function (reader, length) {
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
                };

                return LogicalXorLayerParams;
            })();

            Specification.LogicalNotLayerParams = (function() {

                function LogicalNotLayerParams() {
                }

                LogicalNotLayerParams.decode = function (reader, length) {
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
                };

                return LogicalNotLayerParams;
            })();

            Specification.BorderAmounts = (function() {

                function BorderAmounts() {
                    this.borderAmounts = [];
                }

                BorderAmounts.prototype.borderAmounts = [];

                BorderAmounts.decode = function (reader, length) {
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
                };

                BorderAmounts.EdgeSizes = (function() {

                    function EdgeSizes() {
                    }

                    EdgeSizes.prototype.startEdgeSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    EdgeSizes.prototype.endEdgeSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                    EdgeSizes.decode = function (reader, length) {
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
                    };

                    return EdgeSizes;
                })();

                return BorderAmounts;
            })();

            Specification.ValidPadding = (function() {

                function ValidPadding() {
                }

                ValidPadding.prototype.paddingAmounts = null;

                ValidPadding.decode = function (reader, length) {
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
                };

                return ValidPadding;
            })();

            Specification.SamePadding = (function() {

                function SamePadding() {
                }

                SamePadding.prototype.asymmetryMode = 0;

                SamePadding.decode = function (reader, length) {
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
                };

                SamePadding.SamePaddingMode = (function() {
                    const values = {};
                    values["BOTTOM_RIGHT_HEAVY"] = 0;
                    values["TOP_LEFT_HEAVY"] = 1;
                    return values;
                })();

                return SamePadding;
            })();

            Specification.SamplingMode = (function() {

                function SamplingMode() {
                }

                SamplingMode.prototype.samplingMethod = 0;

                SamplingMode.decode = function (reader, length) {
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
                };

                SamplingMode.Method = (function() {
                    const values = {};
                    values["STRICT_ALIGN_ENDPOINTS_MODE"] = 0;
                    values["ALIGN_ENDPOINTS_MODE"] = 1;
                    values["UPSAMPLE_MODE"] = 2;
                    values["ROI_ALIGN_MODE"] = 3;
                    return values;
                })();

                return SamplingMode;
            })();

            Specification.BoxCoordinatesMode = (function() {

                function BoxCoordinatesMode() {
                }

                BoxCoordinatesMode.prototype.boxMode = 0;

                BoxCoordinatesMode.decode = function (reader, length) {
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
                };

                BoxCoordinatesMode.Coordinates = (function() {
                    const values = {};
                    values["CORNERS_HEIGHT_FIRST"] = 0;
                    values["CORNERS_WIDTH_FIRST"] = 1;
                    values["CENTER_SIZE_HEIGHT_FIRST"] = 2;
                    values["CENTER_SIZE_WIDTH_FIRST"] = 3;
                    return values;
                })();

                return BoxCoordinatesMode;
            })();

            Specification.WeightParams = (function() {

                function WeightParams() {
                    this.floatValue = [];
                }

                WeightParams.prototype.floatValue = [];
                WeightParams.prototype.float16Value = new Uint8Array([]);
                WeightParams.prototype.rawValue = new Uint8Array([]);
                WeightParams.prototype.int8RawValue = new Uint8Array([]);
                WeightParams.prototype.quantization = null;
                WeightParams.prototype.isUpdatable = false;

                WeightParams.decode = function (reader, length) {
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
                };

                return WeightParams;
            })();

            Specification.QuantizationParams = (function() {

                function QuantizationParams() {
                }

                QuantizationParams.prototype.numberOfBits = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                QuantizationParams.prototype.linearQuantization = null;
                QuantizationParams.prototype.lookupTableQuantization = null;

                const QuantizationTypeSet = new Set([ "linearQuantization", "lookupTableQuantization"]);
                Object.defineProperty(QuantizationParams.prototype, "QuantizationType", {
                    get: function() { return Object.keys(this).find((key) => QuantizationTypeSet.has(key) && this[key] != null); }
                });

                QuantizationParams.decode = function (reader, length) {
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
                };

                return QuantizationParams;
            })();

            Specification.LinearQuantizationParams = (function() {

                function LinearQuantizationParams() {
                    this.scale = [];
                    this.bias = [];
                }

                LinearQuantizationParams.prototype.scale = [];
                LinearQuantizationParams.prototype.bias = [];

                LinearQuantizationParams.decode = function (reader, length) {
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
                };

                return LinearQuantizationParams;
            })();

            Specification.LookUpTableQuantizationParams = (function() {

                function LookUpTableQuantizationParams() {
                    this.floatValue = [];
                }

                LookUpTableQuantizationParams.prototype.floatValue = [];

                LookUpTableQuantizationParams.decode = function (reader, length) {
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
                };

                return LookUpTableQuantizationParams;
            })();

            Specification.ConvolutionLayerParams = (function() {

                function ConvolutionLayerParams() {
                    this.kernelSize = [];
                    this.stride = [];
                    this.dilationFactor = [];
                    this.outputShape = [];
                }

                ConvolutionLayerParams.prototype.outputChannels = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                ConvolutionLayerParams.prototype.kernelChannels = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                ConvolutionLayerParams.prototype.nGroups = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                ConvolutionLayerParams.prototype.kernelSize = [];
                ConvolutionLayerParams.prototype.stride = [];
                ConvolutionLayerParams.prototype.dilationFactor = [];
                ConvolutionLayerParams.prototype.valid = null;
                ConvolutionLayerParams.prototype.same = null;
                ConvolutionLayerParams.prototype.isDeconvolution = false;
                ConvolutionLayerParams.prototype.hasBias = false;
                ConvolutionLayerParams.prototype.weights = null;
                ConvolutionLayerParams.prototype.bias = null;
                ConvolutionLayerParams.prototype.outputShape = [];

                const ConvolutionPaddingTypeSet = new Set([ "valid", "same"]);
                Object.defineProperty(ConvolutionLayerParams.prototype, "ConvolutionPaddingType", {
                    get: function() { return Object.keys(this).find((key) => ConvolutionPaddingTypeSet.has(key) && this[key] != null); }
                });

                ConvolutionLayerParams.decode = function (reader, length) {
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
                };

                return ConvolutionLayerParams;
            })();

            Specification.Convolution3DLayerParams = (function() {

                function Convolution3DLayerParams() {
                    this.outputShape = [];
                }

                Convolution3DLayerParams.prototype.outputChannels = 0;
                Convolution3DLayerParams.prototype.inputChannels = 0;
                Convolution3DLayerParams.prototype.nGroups = 0;
                Convolution3DLayerParams.prototype.kernelDepth = 0;
                Convolution3DLayerParams.prototype.kernelHeight = 0;
                Convolution3DLayerParams.prototype.kernelWidth = 0;
                Convolution3DLayerParams.prototype.strideDepth = 0;
                Convolution3DLayerParams.prototype.strideHeight = 0;
                Convolution3DLayerParams.prototype.strideWidth = 0;
                Convolution3DLayerParams.prototype.dilationDepth = 0;
                Convolution3DLayerParams.prototype.dilationHeight = 0;
                Convolution3DLayerParams.prototype.dilationWidth = 0;
                Convolution3DLayerParams.prototype.hasBias = false;
                Convolution3DLayerParams.prototype.weights = null;
                Convolution3DLayerParams.prototype.bias = null;
                Convolution3DLayerParams.prototype.paddingType = 0;
                Convolution3DLayerParams.prototype.customPaddingFront = 0;
                Convolution3DLayerParams.prototype.customPaddingBack = 0;
                Convolution3DLayerParams.prototype.customPaddingTop = 0;
                Convolution3DLayerParams.prototype.customPaddingBottom = 0;
                Convolution3DLayerParams.prototype.customPaddingLeft = 0;
                Convolution3DLayerParams.prototype.customPaddingRight = 0;
                Convolution3DLayerParams.prototype.isDeconvolution = false;
                Convolution3DLayerParams.prototype.outputShape = [];

                Convolution3DLayerParams.decode = function (reader, length) {
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
                };

                Convolution3DLayerParams.PaddingType = (function() {
                    const values = {};
                    values["CUSTOM"] = 0;
                    values["VALID"] = 1;
                    values["SAME"] = 2;
                    return values;
                })();

                return Convolution3DLayerParams;
            })();

            Specification.InnerProductLayerParams = (function() {

                function InnerProductLayerParams() {
                }

                InnerProductLayerParams.prototype.inputChannels = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                InnerProductLayerParams.prototype.outputChannels = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                InnerProductLayerParams.prototype.hasBias = false;
                InnerProductLayerParams.prototype.weights = null;
                InnerProductLayerParams.prototype.bias = null;
                InnerProductLayerParams.prototype.int8DynamicQuantize = false;

                InnerProductLayerParams.decode = function (reader, length) {
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
                };

                return InnerProductLayerParams;
            })();

            Specification.EmbeddingLayerParams = (function() {

                function EmbeddingLayerParams() {
                }

                EmbeddingLayerParams.prototype.inputDim = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                EmbeddingLayerParams.prototype.outputChannels = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                EmbeddingLayerParams.prototype.hasBias = false;
                EmbeddingLayerParams.prototype.weights = null;
                EmbeddingLayerParams.prototype.bias = null;

                EmbeddingLayerParams.decode = function (reader, length) {
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
                };

                return EmbeddingLayerParams;
            })();

            Specification.EmbeddingNDLayerParams = (function() {

                function EmbeddingNDLayerParams() {
                }

                EmbeddingNDLayerParams.prototype.vocabSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                EmbeddingNDLayerParams.prototype.embeddingSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                EmbeddingNDLayerParams.prototype.hasBias = false;
                EmbeddingNDLayerParams.prototype.weights = null;
                EmbeddingNDLayerParams.prototype.bias = null;

                EmbeddingNDLayerParams.decode = function (reader, length) {
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
                };

                return EmbeddingNDLayerParams;
            })();

            Specification.BatchnormLayerParams = (function() {

                function BatchnormLayerParams() {
                }

                BatchnormLayerParams.prototype.channels = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                BatchnormLayerParams.prototype.computeMeanVar = false;
                BatchnormLayerParams.prototype.instanceNormalization = false;
                BatchnormLayerParams.prototype.epsilon = 0;
                BatchnormLayerParams.prototype.gamma = null;
                BatchnormLayerParams.prototype.beta = null;
                BatchnormLayerParams.prototype.mean = null;
                BatchnormLayerParams.prototype.variance = null;

                BatchnormLayerParams.decode = function (reader, length) {
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
                };

                return BatchnormLayerParams;
            })();

            Specification.PoolingLayerParams = (function() {

                function PoolingLayerParams() {
                    this.kernelSize = [];
                    this.stride = [];
                }

                PoolingLayerParams.prototype.type = 0;
                PoolingLayerParams.prototype.kernelSize = [];
                PoolingLayerParams.prototype.stride = [];
                PoolingLayerParams.prototype.valid = null;
                PoolingLayerParams.prototype.same = null;
                PoolingLayerParams.prototype.includeLastPixel = null;
                PoolingLayerParams.prototype.avgPoolExcludePadding = false;
                PoolingLayerParams.prototype.globalPooling = false;

                const PoolingPaddingTypeSet = new Set([ "valid", "same", "includeLastPixel"]);
                Object.defineProperty(PoolingLayerParams.prototype, "PoolingPaddingType", {
                    get: function() { return Object.keys(this).find((key) => PoolingPaddingTypeSet.has(key) && this[key] != null); }
                });

                PoolingLayerParams.decode = function (reader, length) {
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
                };

                PoolingLayerParams.PoolingType = (function() {
                    const values = {};
                    values["MAX"] = 0;
                    values["AVERAGE"] = 1;
                    values["L2"] = 2;
                    return values;
                })();

                PoolingLayerParams.ValidCompletePadding = (function() {

                    function ValidCompletePadding() {
                        this.paddingAmounts = [];
                    }

                    ValidCompletePadding.prototype.paddingAmounts = [];

                    ValidCompletePadding.decode = function (reader, length) {
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
                    };

                    return ValidCompletePadding;
                })();

                return PoolingLayerParams;
            })();

            Specification.Pooling3DLayerParams = (function() {

                function Pooling3DLayerParams() {
                }

                Pooling3DLayerParams.prototype.type = 0;
                Pooling3DLayerParams.prototype.kernelDepth = 0;
                Pooling3DLayerParams.prototype.kernelHeight = 0;
                Pooling3DLayerParams.prototype.kernelWidth = 0;
                Pooling3DLayerParams.prototype.strideDepth = 0;
                Pooling3DLayerParams.prototype.strideHeight = 0;
                Pooling3DLayerParams.prototype.strideWidth = 0;
                Pooling3DLayerParams.prototype.paddingType = 0;
                Pooling3DLayerParams.prototype.customPaddingFront = 0;
                Pooling3DLayerParams.prototype.customPaddingBack = 0;
                Pooling3DLayerParams.prototype.customPaddingTop = 0;
                Pooling3DLayerParams.prototype.customPaddingBottom = 0;
                Pooling3DLayerParams.prototype.customPaddingLeft = 0;
                Pooling3DLayerParams.prototype.customPaddingRight = 0;
                Pooling3DLayerParams.prototype.countExcludePadding = false;

                Pooling3DLayerParams.decode = function (reader, length) {
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
                };

                Pooling3DLayerParams.PoolingType3D = (function() {
                    const values = {};
                    values["MAX"] = 0;
                    values["AVERAGE"] = 1;
                    return values;
                })();

                Pooling3DLayerParams.Pooling3DPaddingType = (function() {
                    const values = {};
                    values["CUSTOM"] = 0;
                    values["VALID"] = 1;
                    values["SAME"] = 2;
                    return values;
                })();

                return Pooling3DLayerParams;
            })();

            Specification.GlobalPooling3DLayerParams = (function() {

                function GlobalPooling3DLayerParams() {
                }

                GlobalPooling3DLayerParams.prototype.type = 0;

                GlobalPooling3DLayerParams.decode = function (reader, length) {
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
                };

                GlobalPooling3DLayerParams.GlobalPoolingType3D = (function() {
                    const values = {};
                    values["MAX"] = 0;
                    values["AVERAGE"] = 1;
                    return values;
                })();

                return GlobalPooling3DLayerParams;
            })();

            Specification.PaddingLayerParams = (function() {

                function PaddingLayerParams() {
                }

                PaddingLayerParams.prototype.constant = null;
                PaddingLayerParams.prototype.reflection = null;
                PaddingLayerParams.prototype.replication = null;
                PaddingLayerParams.prototype.paddingAmounts = null;

                const PaddingTypeSet = new Set([ "constant", "reflection", "replication"]);
                Object.defineProperty(PaddingLayerParams.prototype, "PaddingType", {
                    get: function() { return Object.keys(this).find((key) => PaddingTypeSet.has(key) && this[key] != null); }
                });

                PaddingLayerParams.decode = function (reader, length) {
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
                };

                PaddingLayerParams.PaddingConstant = (function() {

                    function PaddingConstant() {
                    }

                    PaddingConstant.prototype.value = 0;

                    PaddingConstant.decode = function (reader, length) {
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
                    };

                    return PaddingConstant;
                })();

                PaddingLayerParams.PaddingReflection = (function() {

                    function PaddingReflection() {
                    }

                    PaddingReflection.decode = function (reader, length) {
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
                    };

                    return PaddingReflection;
                })();

                PaddingLayerParams.PaddingReplication = (function() {

                    function PaddingReplication() {
                    }

                    PaddingReplication.decode = function (reader, length) {
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
                    };

                    return PaddingReplication;
                })();

                return PaddingLayerParams;
            })();

            Specification.ConcatLayerParams = (function() {

                function ConcatLayerParams() {
                }

                ConcatLayerParams.prototype.sequenceConcat = false;

                ConcatLayerParams.decode = function (reader, length) {
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
                };

                return ConcatLayerParams;
            })();

            Specification.LRNLayerParams = (function() {

                function LRNLayerParams() {
                }

                LRNLayerParams.prototype.alpha = 0;
                LRNLayerParams.prototype.beta = 0;
                LRNLayerParams.prototype.localSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                LRNLayerParams.prototype.k = 0;

                LRNLayerParams.decode = function (reader, length) {
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
                };

                return LRNLayerParams;
            })();

            Specification.SoftmaxLayerParams = (function() {

                function SoftmaxLayerParams() {
                }

                SoftmaxLayerParams.decode = function (reader, length) {
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
                };

                return SoftmaxLayerParams;
            })();

            Specification.SplitLayerParams = (function() {

                function SplitLayerParams() {
                }

                SplitLayerParams.prototype.nOutputs = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                SplitLayerParams.decode = function (reader, length) {
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
                };

                return SplitLayerParams;
            })();

            Specification.AddLayerParams = (function() {

                function AddLayerParams() {
                }

                AddLayerParams.prototype.alpha = 0;

                AddLayerParams.decode = function (reader, length) {
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
                };

                return AddLayerParams;
            })();

            Specification.MultiplyLayerParams = (function() {

                function MultiplyLayerParams() {
                }

                MultiplyLayerParams.prototype.alpha = 0;

                MultiplyLayerParams.decode = function (reader, length) {
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
                };

                return MultiplyLayerParams;
            })();

            Specification.UnaryFunctionLayerParams = (function() {

                function UnaryFunctionLayerParams() {
                }

                UnaryFunctionLayerParams.prototype.type = 0;
                UnaryFunctionLayerParams.prototype.alpha = 0;
                UnaryFunctionLayerParams.prototype.epsilon = 0;
                UnaryFunctionLayerParams.prototype.shift = 0;
                UnaryFunctionLayerParams.prototype.scale = 0;

                UnaryFunctionLayerParams.decode = function (reader, length) {
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
                };

                UnaryFunctionLayerParams.Operation = (function() {
                    const values = {};
                    values["SQRT"] = 0;
                    values["RSQRT"] = 1;
                    values["INVERSE"] = 2;
                    values["POWER"] = 3;
                    values["EXP"] = 4;
                    values["LOG"] = 5;
                    values["ABS"] = 6;
                    values["THRESHOLD"] = 7;
                    return values;
                })();

                return UnaryFunctionLayerParams;
            })();

            Specification.UpsampleLayerParams = (function() {

                function UpsampleLayerParams() {
                    this.scalingFactor = [];
                    this.fractionalScalingFactor = [];
                }

                UpsampleLayerParams.prototype.scalingFactor = [];
                UpsampleLayerParams.prototype.fractionalScalingFactor = [];
                UpsampleLayerParams.prototype.mode = 0;
                UpsampleLayerParams.prototype.linearUpsampleMode = 0;

                UpsampleLayerParams.decode = function (reader, length) {
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
                };

                UpsampleLayerParams.InterpolationMode = (function() {
                    const values = {};
                    values["NN"] = 0;
                    values["BILINEAR"] = 1;
                    return values;
                })();

                UpsampleLayerParams.LinearUpsampleMode = (function() {
                    const values = {};
                    values["DEFAULT"] = 0;
                    values["ALIGN_CORNERS_TRUE"] = 1;
                    values["ALIGN_CORNERS_FALSE"] = 2;
                    return values;
                })();

                return UpsampleLayerParams;
            })();

            Specification.ResizeBilinearLayerParams = (function() {

                function ResizeBilinearLayerParams() {
                    this.targetSize = [];
                }

                ResizeBilinearLayerParams.prototype.targetSize = [];
                ResizeBilinearLayerParams.prototype.mode = null;

                ResizeBilinearLayerParams.decode = function (reader, length) {
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
                };

                return ResizeBilinearLayerParams;
            })();

            Specification.CropResizeLayerParams = (function() {

                function CropResizeLayerParams() {
                    this.targetSize = [];
                }

                CropResizeLayerParams.prototype.targetSize = [];
                CropResizeLayerParams.prototype.normalizedCoordinates = false;
                CropResizeLayerParams.prototype.mode = null;
                CropResizeLayerParams.prototype.boxIndicesMode = null;
                CropResizeLayerParams.prototype.spatialScale = 0;

                CropResizeLayerParams.decode = function (reader, length) {
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
                };

                return CropResizeLayerParams;
            })();

            Specification.BiasLayerParams = (function() {

                function BiasLayerParams() {
                    this.shape = [];
                }

                BiasLayerParams.prototype.shape = [];
                BiasLayerParams.prototype.bias = null;

                BiasLayerParams.decode = function (reader, length) {
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
                };

                return BiasLayerParams;
            })();

            Specification.ScaleLayerParams = (function() {

                function ScaleLayerParams() {
                    this.shapeScale = [];
                    this.shapeBias = [];
                }

                ScaleLayerParams.prototype.shapeScale = [];
                ScaleLayerParams.prototype.scale = null;
                ScaleLayerParams.prototype.hasBias = false;
                ScaleLayerParams.prototype.shapeBias = [];
                ScaleLayerParams.prototype.bias = null;

                ScaleLayerParams.decode = function (reader, length) {
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
                };

                return ScaleLayerParams;
            })();

            Specification.LoadConstantLayerParams = (function() {

                function LoadConstantLayerParams() {
                    this.shape = [];
                }

                LoadConstantLayerParams.prototype.shape = [];
                LoadConstantLayerParams.prototype.data = null;

                LoadConstantLayerParams.decode = function (reader, length) {
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
                };

                return LoadConstantLayerParams;
            })();

            Specification.L2NormalizeLayerParams = (function() {

                function L2NormalizeLayerParams() {
                }

                L2NormalizeLayerParams.prototype.epsilon = 0;

                L2NormalizeLayerParams.decode = function (reader, length) {
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
                };

                return L2NormalizeLayerParams;
            })();

            Specification.FlattenLayerParams = (function() {

                function FlattenLayerParams() {
                }

                FlattenLayerParams.prototype.mode = 0;

                FlattenLayerParams.decode = function (reader, length) {
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
                };

                FlattenLayerParams.FlattenOrder = (function() {
                    const values = {};
                    values["CHANNEL_FIRST"] = 0;
                    values["CHANNEL_LAST"] = 1;
                    return values;
                })();

                return FlattenLayerParams;
            })();

            Specification.ReshapeLayerParams = (function() {

                function ReshapeLayerParams() {
                    this.targetShape = [];
                }

                ReshapeLayerParams.prototype.targetShape = [];
                ReshapeLayerParams.prototype.mode = 0;

                ReshapeLayerParams.decode = function (reader, length) {
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
                };

                ReshapeLayerParams.ReshapeOrder = (function() {
                    const values = {};
                    values["CHANNEL_FIRST"] = 0;
                    values["CHANNEL_LAST"] = 1;
                    return values;
                })();

                return ReshapeLayerParams;
            })();

            Specification.PermuteLayerParams = (function() {

                function PermuteLayerParams() {
                    this.axis = [];
                }

                PermuteLayerParams.prototype.axis = [];

                PermuteLayerParams.decode = function (reader, length) {
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
                };

                return PermuteLayerParams;
            })();

            Specification.ReorganizeDataLayerParams = (function() {

                function ReorganizeDataLayerParams() {
                }

                ReorganizeDataLayerParams.prototype.mode = 0;
                ReorganizeDataLayerParams.prototype.blockSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                ReorganizeDataLayerParams.decode = function (reader, length) {
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
                };

                ReorganizeDataLayerParams.ReorganizationType = (function() {
                    const values = {};
                    values["SPACE_TO_DEPTH"] = 0;
                    values["DEPTH_TO_SPACE"] = 1;
                    values["PIXEL_SHUFFLE"] = 2;
                    return values;
                })();

                return ReorganizeDataLayerParams;
            })();

            Specification.SliceLayerParams = (function() {

                function SliceLayerParams() {
                }

                SliceLayerParams.prototype.startIndex = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                SliceLayerParams.prototype.endIndex = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                SliceLayerParams.prototype.stride = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                SliceLayerParams.prototype.axis = 0;

                SliceLayerParams.decode = function (reader, length) {
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
                };

                SliceLayerParams.SliceAxis = (function() {
                    const values = {};
                    values["CHANNEL_AXIS"] = 0;
                    values["HEIGHT_AXIS"] = 1;
                    values["WIDTH_AXIS"] = 2;
                    return values;
                })();

                return SliceLayerParams;
            })();

            Specification.ReduceLayerParams = (function() {

                function ReduceLayerParams() {
                }

                ReduceLayerParams.prototype.mode = 0;
                ReduceLayerParams.prototype.epsilon = 0;
                ReduceLayerParams.prototype.axis = 0;

                ReduceLayerParams.decode = function (reader, length) {
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
                };

                ReduceLayerParams.ReduceOperation = (function() {
                    const values = {};
                    values["SUM"] = 0;
                    values["AVG"] = 1;
                    values["PROD"] = 2;
                    values["LOGSUM"] = 3;
                    values["SUMSQUARE"] = 4;
                    values["L1"] = 5;
                    values["L2"] = 6;
                    values["MAX"] = 7;
                    values["MIN"] = 8;
                    values["ARGMAX"] = 9;
                    return values;
                })();

                ReduceLayerParams.ReduceAxis = (function() {
                    const values = {};
                    values["CHW"] = 0;
                    values["HW"] = 1;
                    values["C"] = 2;
                    values["H"] = 3;
                    values["W"] = 4;
                    return values;
                })();

                return ReduceLayerParams;
            })();

            Specification.CropLayerParams = (function() {

                function CropLayerParams() {
                    this.offset = [];
                }

                CropLayerParams.prototype.cropAmounts = null;
                CropLayerParams.prototype.offset = [];

                CropLayerParams.decode = function (reader, length) {
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
                };

                return CropLayerParams;
            })();

            Specification.AverageLayerParams = (function() {

                function AverageLayerParams() {
                }

                AverageLayerParams.decode = function (reader, length) {
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
                };

                return AverageLayerParams;
            })();

            Specification.MaxLayerParams = (function() {

                function MaxLayerParams() {
                }

                MaxLayerParams.decode = function (reader, length) {
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
                };

                return MaxLayerParams;
            })();

            Specification.MinLayerParams = (function() {

                function MinLayerParams() {
                }

                MinLayerParams.decode = function (reader, length) {
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
                };

                return MinLayerParams;
            })();

            Specification.DotProductLayerParams = (function() {

                function DotProductLayerParams() {
                }

                DotProductLayerParams.prototype.cosineSimilarity = false;

                DotProductLayerParams.decode = function (reader, length) {
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
                };

                return DotProductLayerParams;
            })();

            Specification.MeanVarianceNormalizeLayerParams = (function() {

                function MeanVarianceNormalizeLayerParams() {
                }

                MeanVarianceNormalizeLayerParams.prototype.acrossChannels = false;
                MeanVarianceNormalizeLayerParams.prototype.normalizeVariance = false;
                MeanVarianceNormalizeLayerParams.prototype.epsilon = 0;

                MeanVarianceNormalizeLayerParams.decode = function (reader, length) {
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
                };

                return MeanVarianceNormalizeLayerParams;
            })();

            Specification.SequenceRepeatLayerParams = (function() {

                function SequenceRepeatLayerParams() {
                }

                SequenceRepeatLayerParams.prototype.nRepetitions = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                SequenceRepeatLayerParams.decode = function (reader, length) {
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
                };

                return SequenceRepeatLayerParams;
            })();

            Specification.SimpleRecurrentLayerParams = (function() {

                function SimpleRecurrentLayerParams() {
                }

                SimpleRecurrentLayerParams.prototype.inputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                SimpleRecurrentLayerParams.prototype.outputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                SimpleRecurrentLayerParams.prototype.activation = null;
                SimpleRecurrentLayerParams.prototype.sequenceOutput = false;
                SimpleRecurrentLayerParams.prototype.hasBiasVector = false;
                SimpleRecurrentLayerParams.prototype.weightMatrix = null;
                SimpleRecurrentLayerParams.prototype.recursionMatrix = null;
                SimpleRecurrentLayerParams.prototype.biasVector = null;
                SimpleRecurrentLayerParams.prototype.reverseInput = false;

                SimpleRecurrentLayerParams.decode = function (reader, length) {
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
                };

                return SimpleRecurrentLayerParams;
            })();

            Specification.GRULayerParams = (function() {

                function GRULayerParams() {
                    this.activations = [];
                }

                GRULayerParams.prototype.inputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                GRULayerParams.prototype.outputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                GRULayerParams.prototype.activations = [];
                GRULayerParams.prototype.sequenceOutput = false;
                GRULayerParams.prototype.hasBiasVectors = false;
                GRULayerParams.prototype.updateGateWeightMatrix = null;
                GRULayerParams.prototype.resetGateWeightMatrix = null;
                GRULayerParams.prototype.outputGateWeightMatrix = null;
                GRULayerParams.prototype.updateGateRecursionMatrix = null;
                GRULayerParams.prototype.resetGateRecursionMatrix = null;
                GRULayerParams.prototype.outputGateRecursionMatrix = null;
                GRULayerParams.prototype.updateGateBiasVector = null;
                GRULayerParams.prototype.resetGateBiasVector = null;
                GRULayerParams.prototype.outputGateBiasVector = null;
                GRULayerParams.prototype.reverseInput = false;

                GRULayerParams.decode = function (reader, length) {
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
                };

                return GRULayerParams;
            })();

            Specification.LSTMParams = (function() {

                function LSTMParams() {
                }

                LSTMParams.prototype.sequenceOutput = false;
                LSTMParams.prototype.hasBiasVectors = false;
                LSTMParams.prototype.forgetBias = false;
                LSTMParams.prototype.hasPeepholeVectors = false;
                LSTMParams.prototype.coupledInputAndForgetGate = false;
                LSTMParams.prototype.cellClipThreshold = 0;

                LSTMParams.decode = function (reader, length) {
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
                };

                return LSTMParams;
            })();

            Specification.LSTMWeightParams = (function() {

                function LSTMWeightParams() {
                }

                LSTMWeightParams.prototype.inputGateWeightMatrix = null;
                LSTMWeightParams.prototype.forgetGateWeightMatrix = null;
                LSTMWeightParams.prototype.blockInputWeightMatrix = null;
                LSTMWeightParams.prototype.outputGateWeightMatrix = null;
                LSTMWeightParams.prototype.inputGateRecursionMatrix = null;
                LSTMWeightParams.prototype.forgetGateRecursionMatrix = null;
                LSTMWeightParams.prototype.blockInputRecursionMatrix = null;
                LSTMWeightParams.prototype.outputGateRecursionMatrix = null;
                LSTMWeightParams.prototype.inputGateBiasVector = null;
                LSTMWeightParams.prototype.forgetGateBiasVector = null;
                LSTMWeightParams.prototype.blockInputBiasVector = null;
                LSTMWeightParams.prototype.outputGateBiasVector = null;
                LSTMWeightParams.prototype.inputGatePeepholeVector = null;
                LSTMWeightParams.prototype.forgetGatePeepholeVector = null;
                LSTMWeightParams.prototype.outputGatePeepholeVector = null;

                LSTMWeightParams.decode = function (reader, length) {
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
                };

                return LSTMWeightParams;
            })();

            Specification.UniDirectionalLSTMLayerParams = (function() {

                function UniDirectionalLSTMLayerParams() {
                    this.activations = [];
                }

                UniDirectionalLSTMLayerParams.prototype.inputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                UniDirectionalLSTMLayerParams.prototype.outputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                UniDirectionalLSTMLayerParams.prototype.activations = [];
                UniDirectionalLSTMLayerParams.prototype.params = null;
                UniDirectionalLSTMLayerParams.prototype.weightParams = null;
                UniDirectionalLSTMLayerParams.prototype.reverseInput = false;

                UniDirectionalLSTMLayerParams.decode = function (reader, length) {
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
                };

                return UniDirectionalLSTMLayerParams;
            })();

            Specification.BiDirectionalLSTMLayerParams = (function() {

                function BiDirectionalLSTMLayerParams() {
                    this.activationsForwardLSTM = [];
                    this.activationsBackwardLSTM = [];
                    this.weightParams = [];
                }

                BiDirectionalLSTMLayerParams.prototype.inputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                BiDirectionalLSTMLayerParams.prototype.outputVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                BiDirectionalLSTMLayerParams.prototype.activationsForwardLSTM = [];
                BiDirectionalLSTMLayerParams.prototype.activationsBackwardLSTM = [];
                BiDirectionalLSTMLayerParams.prototype.params = null;
                BiDirectionalLSTMLayerParams.prototype.weightParams = [];

                BiDirectionalLSTMLayerParams.decode = function (reader, length) {
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
                };

                return BiDirectionalLSTMLayerParams;
            })();

            Specification.CustomLayerParams = (function() {

                function CustomLayerParams() {
                    this.weights = [];
                    this.parameters = {};
                }

                CustomLayerParams.prototype.className = "";
                CustomLayerParams.prototype.weights = [];
                CustomLayerParams.prototype.parameters = {};
                CustomLayerParams.prototype.description = "";

                CustomLayerParams.decode = function (reader, length) {
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
                                reader.pair(message.parameters, () => reader.string(), () => $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.decode(reader, reader.uint32()));
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
                };

                CustomLayerParams.CustomLayerParamValue = (function() {

                    function CustomLayerParamValue() {
                    }

                    CustomLayerParamValue.prototype.doubleValue = 0;
                    CustomLayerParamValue.prototype.stringValue = "";
                    CustomLayerParamValue.prototype.intValue = 0;
                    CustomLayerParamValue.prototype.longValue = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                    CustomLayerParamValue.prototype.boolValue = false;

                    const valueSet = new Set([ "doubleValue", "stringValue", "intValue", "longValue", "boolValue"]);
                    Object.defineProperty(CustomLayerParamValue.prototype, "value", {
                        get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
                    });

                    CustomLayerParamValue.decode = function (reader, length) {
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
                    };

                    return CustomLayerParamValue;
                })();

                return CustomLayerParams;
            })();

            Specification.TransposeLayerParams = (function() {

                function TransposeLayerParams() {
                    this.axes = [];
                }

                TransposeLayerParams.prototype.axes = [];

                TransposeLayerParams.decode = function (reader, length) {
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
                };

                return TransposeLayerParams;
            })();

            Specification.BatchedMatMulLayerParams = (function() {

                function BatchedMatMulLayerParams() {
                }

                BatchedMatMulLayerParams.prototype.transposeA = false;
                BatchedMatMulLayerParams.prototype.transposeB = false;
                BatchedMatMulLayerParams.prototype.weightMatrixFirstDimension = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                BatchedMatMulLayerParams.prototype.weightMatrixSecondDimension = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                BatchedMatMulLayerParams.prototype.hasBias = false;
                BatchedMatMulLayerParams.prototype.weights = null;
                BatchedMatMulLayerParams.prototype.bias = null;
                BatchedMatMulLayerParams.prototype.int8DynamicQuantize = false;

                BatchedMatMulLayerParams.decode = function (reader, length) {
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
                };

                return BatchedMatMulLayerParams;
            })();

            Specification.ConcatNDLayerParams = (function() {

                function ConcatNDLayerParams() {
                }

                ConcatNDLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                ConcatNDLayerParams.decode = function (reader, length) {
                    const message = new $root.CoreML.Specification.ConcatNDLayerParams();
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
                };

                return ConcatNDLayerParams;
            })();

            Specification.SoftmaxNDLayerParams = (function() {

                function SoftmaxNDLayerParams() {
                }

                SoftmaxNDLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                SoftmaxNDLayerParams.decode = function (reader, length) {
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
                };

                return SoftmaxNDLayerParams;
            })();

            Specification.ReverseLayerParams = (function() {

                function ReverseLayerParams() {
                    this.reverseDim = [];
                }

                ReverseLayerParams.prototype.reverseDim = [];

                ReverseLayerParams.decode = function (reader, length) {
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
                };

                return ReverseLayerParams;
            })();

            Specification.ReverseSeqLayerParams = (function() {

                function ReverseSeqLayerParams() {
                }

                ReverseSeqLayerParams.prototype.batchAxis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ReverseSeqLayerParams.prototype.sequenceAxis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                ReverseSeqLayerParams.decode = function (reader, length) {
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
                };

                return ReverseSeqLayerParams;
            })();

            Specification.LoadConstantNDLayerParams = (function() {

                function LoadConstantNDLayerParams() {
                    this.shape = [];
                }

                LoadConstantNDLayerParams.prototype.shape = [];
                LoadConstantNDLayerParams.prototype.data = null;

                LoadConstantNDLayerParams.decode = function (reader, length) {
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
                };

                return LoadConstantNDLayerParams;
            })();

            Specification.FillLikeLayerParams = (function() {

                function FillLikeLayerParams() {
                }

                FillLikeLayerParams.prototype.value = 0;

                FillLikeLayerParams.decode = function (reader, length) {
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
                };

                return FillLikeLayerParams;
            })();

            Specification.FillStaticLayerParams = (function() {

                function FillStaticLayerParams() {
                    this.targetShape = [];
                }

                FillStaticLayerParams.prototype.value = 0;
                FillStaticLayerParams.prototype.targetShape = [];

                FillStaticLayerParams.decode = function (reader, length) {
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
                };

                return FillStaticLayerParams;
            })();

            Specification.FillDynamicLayerParams = (function() {

                function FillDynamicLayerParams() {
                }

                FillDynamicLayerParams.prototype.value = 0;

                FillDynamicLayerParams.decode = function (reader, length) {
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
                };

                return FillDynamicLayerParams;
            })();

            Specification.WhereBroadcastableLayerParams = (function() {

                function WhereBroadcastableLayerParams() {
                }

                WhereBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return WhereBroadcastableLayerParams;
            })();

            Specification.SinLayerParams = (function() {

                function SinLayerParams() {
                }

                SinLayerParams.decode = function (reader, length) {
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
                };

                return SinLayerParams;
            })();

            Specification.CosLayerParams = (function() {

                function CosLayerParams() {
                }

                CosLayerParams.decode = function (reader, length) {
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
                };

                return CosLayerParams;
            })();

            Specification.TanLayerParams = (function() {

                function TanLayerParams() {
                }

                TanLayerParams.decode = function (reader, length) {
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
                };

                return TanLayerParams;
            })();

            Specification.AsinLayerParams = (function() {

                function AsinLayerParams() {
                }

                AsinLayerParams.decode = function (reader, length) {
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
                };

                return AsinLayerParams;
            })();

            Specification.AcosLayerParams = (function() {

                function AcosLayerParams() {
                }

                AcosLayerParams.decode = function (reader, length) {
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
                };

                return AcosLayerParams;
            })();

            Specification.AtanLayerParams = (function() {

                function AtanLayerParams() {
                }

                AtanLayerParams.decode = function (reader, length) {
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
                };

                return AtanLayerParams;
            })();

            Specification.SinhLayerParams = (function() {

                function SinhLayerParams() {
                }

                SinhLayerParams.decode = function (reader, length) {
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
                };

                return SinhLayerParams;
            })();

            Specification.CoshLayerParams = (function() {

                function CoshLayerParams() {
                }

                CoshLayerParams.decode = function (reader, length) {
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
                };

                return CoshLayerParams;
            })();

            Specification.TanhLayerParams = (function() {

                function TanhLayerParams() {
                }

                TanhLayerParams.decode = function (reader, length) {
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
                };

                return TanhLayerParams;
            })();

            Specification.AsinhLayerParams = (function() {

                function AsinhLayerParams() {
                }

                AsinhLayerParams.decode = function (reader, length) {
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
                };

                return AsinhLayerParams;
            })();

            Specification.AcoshLayerParams = (function() {

                function AcoshLayerParams() {
                }

                AcoshLayerParams.decode = function (reader, length) {
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
                };

                return AcoshLayerParams;
            })();

            Specification.AtanhLayerParams = (function() {

                function AtanhLayerParams() {
                }

                AtanhLayerParams.decode = function (reader, length) {
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
                };

                return AtanhLayerParams;
            })();

            Specification.PowBroadcastableLayerParams = (function() {

                function PowBroadcastableLayerParams() {
                }

                PowBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return PowBroadcastableLayerParams;
            })();

            Specification.Exp2LayerParams = (function() {

                function Exp2LayerParams() {
                }

                Exp2LayerParams.decode = function (reader, length) {
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
                };

                return Exp2LayerParams;
            })();

            Specification.WhereNonZeroLayerParams = (function() {

                function WhereNonZeroLayerParams() {
                }

                WhereNonZeroLayerParams.decode = function (reader, length) {
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
                };

                return WhereNonZeroLayerParams;
            })();

            Specification.MatrixBandPartLayerParams = (function() {

                function MatrixBandPartLayerParams() {
                }

                MatrixBandPartLayerParams.prototype.numLower = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                MatrixBandPartLayerParams.prototype.numUpper = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                MatrixBandPartLayerParams.decode = function (reader, length) {
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
                };

                return MatrixBandPartLayerParams;
            })();

            Specification.UpperTriangularLayerParams = (function() {

                function UpperTriangularLayerParams() {
                }

                UpperTriangularLayerParams.prototype.k = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                UpperTriangularLayerParams.decode = function (reader, length) {
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
                };

                return UpperTriangularLayerParams;
            })();

            Specification.LowerTriangularLayerParams = (function() {

                function LowerTriangularLayerParams() {
                }

                LowerTriangularLayerParams.prototype.k = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                LowerTriangularLayerParams.decode = function (reader, length) {
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
                };

                return LowerTriangularLayerParams;
            })();

            Specification.BroadcastToLikeLayerParams = (function() {

                function BroadcastToLikeLayerParams() {
                }

                BroadcastToLikeLayerParams.decode = function (reader, length) {
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
                };

                return BroadcastToLikeLayerParams;
            })();

            Specification.BroadcastToStaticLayerParams = (function() {

                function BroadcastToStaticLayerParams() {
                    this.targetShape = [];
                }

                BroadcastToStaticLayerParams.prototype.targetShape = [];

                BroadcastToStaticLayerParams.decode = function (reader, length) {
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
                };

                return BroadcastToStaticLayerParams;
            })();

            Specification.BroadcastToDynamicLayerParams = (function() {

                function BroadcastToDynamicLayerParams() {
                }

                BroadcastToDynamicLayerParams.decode = function (reader, length) {
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
                };

                return BroadcastToDynamicLayerParams;
            })();

            Specification.AddBroadcastableLayerParams = (function() {

                function AddBroadcastableLayerParams() {
                }

                AddBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return AddBroadcastableLayerParams;
            })();

            Specification.MaxBroadcastableLayerParams = (function() {

                function MaxBroadcastableLayerParams() {
                }

                MaxBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return MaxBroadcastableLayerParams;
            })();

            Specification.MinBroadcastableLayerParams = (function() {

                function MinBroadcastableLayerParams() {
                }

                MinBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return MinBroadcastableLayerParams;
            })();

            Specification.ModBroadcastableLayerParams = (function() {

                function ModBroadcastableLayerParams() {
                }

                ModBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return ModBroadcastableLayerParams;
            })();

            Specification.FloorDivBroadcastableLayerParams = (function() {

                function FloorDivBroadcastableLayerParams() {
                }

                FloorDivBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return FloorDivBroadcastableLayerParams;
            })();

            Specification.SubtractBroadcastableLayerParams = (function() {

                function SubtractBroadcastableLayerParams() {
                }

                SubtractBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return SubtractBroadcastableLayerParams;
            })();

            Specification.MultiplyBroadcastableLayerParams = (function() {

                function MultiplyBroadcastableLayerParams() {
                }

                MultiplyBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return MultiplyBroadcastableLayerParams;
            })();

            Specification.DivideBroadcastableLayerParams = (function() {

                function DivideBroadcastableLayerParams() {
                }

                DivideBroadcastableLayerParams.decode = function (reader, length) {
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
                };

                return DivideBroadcastableLayerParams;
            })();

            Specification.GatherLayerParams = (function() {

                function GatherLayerParams() {
                }

                GatherLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                GatherLayerParams.decode = function (reader, length) {
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
                };

                return GatherLayerParams;
            })();

            Specification.ScatterMode = (function() {
                const values = {};
                values["SCATTER_UPDATE"] = 0;
                values["SCATTER_ADD"] = 1;
                values["SCATTER_SUB"] = 2;
                values["SCATTER_MUL"] = 3;
                values["SCATTER_DIV"] = 4;
                values["SCATTER_MAX"] = 5;
                values["SCATTER_MIN"] = 6;
                return values;
            })();

            Specification.ScatterLayerParams = (function() {

                function ScatterLayerParams() {
                }

                ScatterLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ScatterLayerParams.prototype.mode = 0;

                ScatterLayerParams.decode = function (reader, length) {
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
                };

                return ScatterLayerParams;
            })();

            Specification.GatherNDLayerParams = (function() {

                function GatherNDLayerParams() {
                }

                GatherNDLayerParams.decode = function (reader, length) {
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
                };

                return GatherNDLayerParams;
            })();

            Specification.ScatterNDLayerParams = (function() {

                function ScatterNDLayerParams() {
                }

                ScatterNDLayerParams.prototype.mode = 0;

                ScatterNDLayerParams.decode = function (reader, length) {
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
                };

                return ScatterNDLayerParams;
            })();

            Specification.GatherAlongAxisLayerParams = (function() {

                function GatherAlongAxisLayerParams() {
                }

                GatherAlongAxisLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                GatherAlongAxisLayerParams.decode = function (reader, length) {
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
                };

                return GatherAlongAxisLayerParams;
            })();

            Specification.ScatterAlongAxisLayerParams = (function() {

                function ScatterAlongAxisLayerParams() {
                }

                ScatterAlongAxisLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ScatterAlongAxisLayerParams.prototype.mode = 0;

                ScatterAlongAxisLayerParams.decode = function (reader, length) {
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
                };

                return ScatterAlongAxisLayerParams;
            })();

            Specification.StackLayerParams = (function() {

                function StackLayerParams() {
                }

                StackLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                StackLayerParams.decode = function (reader, length) {
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
                };

                return StackLayerParams;
            })();

            Specification.RankPreservingReshapeLayerParams = (function() {

                function RankPreservingReshapeLayerParams() {
                    this.targetShape = [];
                }

                RankPreservingReshapeLayerParams.prototype.targetShape = [];

                RankPreservingReshapeLayerParams.decode = function (reader, length) {
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
                };

                return RankPreservingReshapeLayerParams;
            })();

            Specification.ConstantPaddingLayerParams = (function() {

                function ConstantPaddingLayerParams() {
                    this.padAmounts = [];
                }

                ConstantPaddingLayerParams.prototype.value = 0;
                ConstantPaddingLayerParams.prototype.padAmounts = [];
                ConstantPaddingLayerParams.prototype.padToGivenOutputSizeMode = false;

                ConstantPaddingLayerParams.decode = function (reader, length) {
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
                };

                return ConstantPaddingLayerParams;
            })();

            Specification.RandomNormalLikeLayerParams = (function() {

                function RandomNormalLikeLayerParams() {
                }

                RandomNormalLikeLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomNormalLikeLayerParams.prototype.mean = 0;
                RandomNormalLikeLayerParams.prototype.stdDev = 0;

                RandomNormalLikeLayerParams.decode = function (reader, length) {
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
                };

                return RandomNormalLikeLayerParams;
            })();

            Specification.RandomNormalStaticLayerParams = (function() {

                function RandomNormalStaticLayerParams() {
                    this.outputShape = [];
                }

                RandomNormalStaticLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomNormalStaticLayerParams.prototype.mean = 0;
                RandomNormalStaticLayerParams.prototype.stdDev = 0;
                RandomNormalStaticLayerParams.prototype.outputShape = [];

                RandomNormalStaticLayerParams.decode = function (reader, length) {
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
                };

                return RandomNormalStaticLayerParams;
            })();

            Specification.RandomNormalDynamicLayerParams = (function() {

                function RandomNormalDynamicLayerParams() {
                }

                RandomNormalDynamicLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomNormalDynamicLayerParams.prototype.mean = 0;
                RandomNormalDynamicLayerParams.prototype.stdDev = 0;

                RandomNormalDynamicLayerParams.decode = function (reader, length) {
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
                };

                return RandomNormalDynamicLayerParams;
            })();

            Specification.RandomUniformLikeLayerParams = (function() {

                function RandomUniformLikeLayerParams() {
                }

                RandomUniformLikeLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomUniformLikeLayerParams.prototype.minVal = 0;
                RandomUniformLikeLayerParams.prototype.maxVal = 0;

                RandomUniformLikeLayerParams.decode = function (reader, length) {
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
                };

                return RandomUniformLikeLayerParams;
            })();

            Specification.RandomUniformStaticLayerParams = (function() {

                function RandomUniformStaticLayerParams() {
                    this.outputShape = [];
                }

                RandomUniformStaticLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomUniformStaticLayerParams.prototype.minVal = 0;
                RandomUniformStaticLayerParams.prototype.maxVal = 0;
                RandomUniformStaticLayerParams.prototype.outputShape = [];

                RandomUniformStaticLayerParams.decode = function (reader, length) {
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
                };

                return RandomUniformStaticLayerParams;
            })();

            Specification.RandomUniformDynamicLayerParams = (function() {

                function RandomUniformDynamicLayerParams() {
                }

                RandomUniformDynamicLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomUniformDynamicLayerParams.prototype.minVal = 0;
                RandomUniformDynamicLayerParams.prototype.maxVal = 0;

                RandomUniformDynamicLayerParams.decode = function (reader, length) {
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
                };

                return RandomUniformDynamicLayerParams;
            })();

            Specification.RandomBernoulliLikeLayerParams = (function() {

                function RandomBernoulliLikeLayerParams() {
                }

                RandomBernoulliLikeLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomBernoulliLikeLayerParams.prototype.prob = 0;

                RandomBernoulliLikeLayerParams.decode = function (reader, length) {
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
                };

                return RandomBernoulliLikeLayerParams;
            })();

            Specification.RandomBernoulliStaticLayerParams = (function() {

                function RandomBernoulliStaticLayerParams() {
                    this.outputShape = [];
                }

                RandomBernoulliStaticLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomBernoulliStaticLayerParams.prototype.prob = 0;
                RandomBernoulliStaticLayerParams.prototype.outputShape = [];

                RandomBernoulliStaticLayerParams.decode = function (reader, length) {
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
                };

                return RandomBernoulliStaticLayerParams;
            })();

            Specification.RandomBernoulliDynamicLayerParams = (function() {

                function RandomBernoulliDynamicLayerParams() {
                }

                RandomBernoulliDynamicLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                RandomBernoulliDynamicLayerParams.prototype.prob = 0;

                RandomBernoulliDynamicLayerParams.decode = function (reader, length) {
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
                };

                return RandomBernoulliDynamicLayerParams;
            })();

            Specification.CategoricalDistributionLayerParams = (function() {

                function CategoricalDistributionLayerParams() {
                }

                CategoricalDistributionLayerParams.prototype.seed = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                CategoricalDistributionLayerParams.prototype.numSamples = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                CategoricalDistributionLayerParams.prototype.isLogits = false;
                CategoricalDistributionLayerParams.prototype.eps = 0;
                CategoricalDistributionLayerParams.prototype.temperature = 0;

                CategoricalDistributionLayerParams.decode = function (reader, length) {
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
                };

                return CategoricalDistributionLayerParams;
            })();

            Specification.ReduceL1LayerParams = (function() {

                function ReduceL1LayerParams() {
                    this.axes = [];
                }

                ReduceL1LayerParams.prototype.axes = [];
                ReduceL1LayerParams.prototype.keepDims = false;
                ReduceL1LayerParams.prototype.reduceAll = false;

                ReduceL1LayerParams.decode = function (reader, length) {
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
                };

                return ReduceL1LayerParams;
            })();

            Specification.ReduceL2LayerParams = (function() {

                function ReduceL2LayerParams() {
                    this.axes = [];
                }

                ReduceL2LayerParams.prototype.axes = [];
                ReduceL2LayerParams.prototype.keepDims = false;
                ReduceL2LayerParams.prototype.reduceAll = false;

                ReduceL2LayerParams.decode = function (reader, length) {
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
                };

                return ReduceL2LayerParams;
            })();

            Specification.ReduceMaxLayerParams = (function() {

                function ReduceMaxLayerParams() {
                    this.axes = [];
                }

                ReduceMaxLayerParams.prototype.axes = [];
                ReduceMaxLayerParams.prototype.keepDims = false;
                ReduceMaxLayerParams.prototype.reduceAll = false;

                ReduceMaxLayerParams.decode = function (reader, length) {
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
                };

                return ReduceMaxLayerParams;
            })();

            Specification.ReduceMinLayerParams = (function() {

                function ReduceMinLayerParams() {
                    this.axes = [];
                }

                ReduceMinLayerParams.prototype.axes = [];
                ReduceMinLayerParams.prototype.keepDims = false;
                ReduceMinLayerParams.prototype.reduceAll = false;

                ReduceMinLayerParams.decode = function (reader, length) {
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
                };

                return ReduceMinLayerParams;
            })();

            Specification.ReduceSumLayerParams = (function() {

                function ReduceSumLayerParams() {
                    this.axes = [];
                }

                ReduceSumLayerParams.prototype.axes = [];
                ReduceSumLayerParams.prototype.keepDims = false;
                ReduceSumLayerParams.prototype.reduceAll = false;

                ReduceSumLayerParams.decode = function (reader, length) {
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
                };

                return ReduceSumLayerParams;
            })();

            Specification.ReduceProdLayerParams = (function() {

                function ReduceProdLayerParams() {
                    this.axes = [];
                }

                ReduceProdLayerParams.prototype.axes = [];
                ReduceProdLayerParams.prototype.keepDims = false;
                ReduceProdLayerParams.prototype.reduceAll = false;

                ReduceProdLayerParams.decode = function (reader, length) {
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
                };

                return ReduceProdLayerParams;
            })();

            Specification.ReduceMeanLayerParams = (function() {

                function ReduceMeanLayerParams() {
                    this.axes = [];
                }

                ReduceMeanLayerParams.prototype.axes = [];
                ReduceMeanLayerParams.prototype.keepDims = false;
                ReduceMeanLayerParams.prototype.reduceAll = false;

                ReduceMeanLayerParams.decode = function (reader, length) {
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
                };

                return ReduceMeanLayerParams;
            })();

            Specification.ReduceLogSumLayerParams = (function() {

                function ReduceLogSumLayerParams() {
                    this.axes = [];
                }

                ReduceLogSumLayerParams.prototype.axes = [];
                ReduceLogSumLayerParams.prototype.keepDims = false;
                ReduceLogSumLayerParams.prototype.reduceAll = false;

                ReduceLogSumLayerParams.decode = function (reader, length) {
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
                };

                return ReduceLogSumLayerParams;
            })();

            Specification.ReduceSumSquareLayerParams = (function() {

                function ReduceSumSquareLayerParams() {
                    this.axes = [];
                }

                ReduceSumSquareLayerParams.prototype.axes = [];
                ReduceSumSquareLayerParams.prototype.keepDims = false;
                ReduceSumSquareLayerParams.prototype.reduceAll = false;

                ReduceSumSquareLayerParams.decode = function (reader, length) {
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
                };

                return ReduceSumSquareLayerParams;
            })();

            Specification.ReduceLogSumExpLayerParams = (function() {

                function ReduceLogSumExpLayerParams() {
                    this.axes = [];
                }

                ReduceLogSumExpLayerParams.prototype.axes = [];
                ReduceLogSumExpLayerParams.prototype.keepDims = false;
                ReduceLogSumExpLayerParams.prototype.reduceAll = false;

                ReduceLogSumExpLayerParams.decode = function (reader, length) {
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
                };

                return ReduceLogSumExpLayerParams;
            })();

            Specification.ExpandDimsLayerParams = (function() {

                function ExpandDimsLayerParams() {
                    this.axes = [];
                }

                ExpandDimsLayerParams.prototype.axes = [];

                ExpandDimsLayerParams.decode = function (reader, length) {
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
                };

                return ExpandDimsLayerParams;
            })();

            Specification.FlattenTo2DLayerParams = (function() {

                function FlattenTo2DLayerParams() {
                }

                FlattenTo2DLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                FlattenTo2DLayerParams.decode = function (reader, length) {
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
                };

                return FlattenTo2DLayerParams;
            })();

            Specification.ReshapeStaticLayerParams = (function() {

                function ReshapeStaticLayerParams() {
                    this.targetShape = [];
                }

                ReshapeStaticLayerParams.prototype.targetShape = [];

                ReshapeStaticLayerParams.decode = function (reader, length) {
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
                };

                return ReshapeStaticLayerParams;
            })();

            Specification.ReshapeLikeLayerParams = (function() {

                function ReshapeLikeLayerParams() {
                }

                ReshapeLikeLayerParams.decode = function (reader, length) {
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
                };

                return ReshapeLikeLayerParams;
            })();

            Specification.ReshapeDynamicLayerParams = (function() {

                function ReshapeDynamicLayerParams() {
                }

                ReshapeDynamicLayerParams.decode = function (reader, length) {
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
                };

                return ReshapeDynamicLayerParams;
            })();

            Specification.SqueezeLayerParams = (function() {

                function SqueezeLayerParams() {
                    this.axes = [];
                }

                SqueezeLayerParams.prototype.axes = [];
                SqueezeLayerParams.prototype.squeezeAll = false;

                SqueezeLayerParams.decode = function (reader, length) {
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
                };

                return SqueezeLayerParams;
            })();

            Specification.TopKLayerParams = (function() {

                function TopKLayerParams() {
                }

                TopKLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                TopKLayerParams.prototype.K = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                TopKLayerParams.prototype.useBottomK = false;

                TopKLayerParams.decode = function (reader, length) {
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
                };

                return TopKLayerParams;
            })();

            Specification.ArgMaxLayerParams = (function() {

                function ArgMaxLayerParams() {
                }

                ArgMaxLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ArgMaxLayerParams.prototype.removeDim = false;

                ArgMaxLayerParams.decode = function (reader, length) {
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
                };

                return ArgMaxLayerParams;
            })();

            Specification.ArgMinLayerParams = (function() {

                function ArgMinLayerParams() {
                }

                ArgMinLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ArgMinLayerParams.prototype.removeDim = false;

                ArgMinLayerParams.decode = function (reader, length) {
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
                };

                return ArgMinLayerParams;
            })();

            Specification.SplitNDLayerParams = (function() {

                function SplitNDLayerParams() {
                    this.splitSizes = [];
                }

                SplitNDLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                SplitNDLayerParams.prototype.numSplits = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                SplitNDLayerParams.prototype.splitSizes = [];

                SplitNDLayerParams.decode = function (reader, length) {
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
                };

                return SplitNDLayerParams;
            })();

            Specification.CeilLayerParams = (function() {

                function CeilLayerParams() {
                }

                CeilLayerParams.decode = function (reader, length) {
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
                };

                return CeilLayerParams;
            })();

            Specification.RoundLayerParams = (function() {

                function RoundLayerParams() {
                }

                RoundLayerParams.decode = function (reader, length) {
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
                };

                return RoundLayerParams;
            })();

            Specification.FloorLayerParams = (function() {

                function FloorLayerParams() {
                }

                FloorLayerParams.decode = function (reader, length) {
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
                };

                return FloorLayerParams;
            })();

            Specification.SignLayerParams = (function() {

                function SignLayerParams() {
                }

                SignLayerParams.decode = function (reader, length) {
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
                };

                return SignLayerParams;
            })();

            Specification.ClipLayerParams = (function() {

                function ClipLayerParams() {
                }

                ClipLayerParams.prototype.minVal = 0;
                ClipLayerParams.prototype.maxVal = 0;

                ClipLayerParams.decode = function (reader, length) {
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
                };

                return ClipLayerParams;
            })();

            Specification.SliceStaticLayerParams = (function() {

                function SliceStaticLayerParams() {
                    this.beginIds = [];
                    this.beginMasks = [];
                    this.endIds = [];
                    this.endMasks = [];
                    this.strides = [];
                    this.squeezeMasks = [];
                }

                SliceStaticLayerParams.prototype.beginIds = [];
                SliceStaticLayerParams.prototype.beginMasks = [];
                SliceStaticLayerParams.prototype.endIds = [];
                SliceStaticLayerParams.prototype.endMasks = [];
                SliceStaticLayerParams.prototype.strides = [];
                SliceStaticLayerParams.prototype.squeezeMasks = [];

                SliceStaticLayerParams.decode = function (reader, length) {
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
                };

                return SliceStaticLayerParams;
            })();

            Specification.SliceDynamicLayerParams = (function() {

                function SliceDynamicLayerParams() {
                    this.beginMasks = [];
                    this.endIds = [];
                    this.endMasks = [];
                    this.strides = [];
                    this.squeezeMasks = [];
                }

                SliceDynamicLayerParams.prototype.beginMasks = [];
                SliceDynamicLayerParams.prototype.endIds = [];
                SliceDynamicLayerParams.prototype.endMasks = [];
                SliceDynamicLayerParams.prototype.strides = [];
                SliceDynamicLayerParams.prototype.squeezeMasks = [];

                SliceDynamicLayerParams.decode = function (reader, length) {
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
                };

                return SliceDynamicLayerParams;
            })();

            Specification.TileLayerParams = (function() {

                function TileLayerParams() {
                    this.reps = [];
                }

                TileLayerParams.prototype.reps = [];

                TileLayerParams.decode = function (reader, length) {
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
                };

                return TileLayerParams;
            })();

            Specification.GetShapeLayerParams = (function() {

                function GetShapeLayerParams() {
                }

                GetShapeLayerParams.decode = function (reader, length) {
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
                };

                return GetShapeLayerParams;
            })();

            Specification.ErfLayerParams = (function() {

                function ErfLayerParams() {
                }

                ErfLayerParams.decode = function (reader, length) {
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
                };

                return ErfLayerParams;
            })();

            Specification.GeluLayerParams = (function() {

                function GeluLayerParams() {
                }

                GeluLayerParams.prototype.mode = 0;

                GeluLayerParams.decode = function (reader, length) {
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
                };

                GeluLayerParams.GeluMode = (function() {
                    const values = {};
                    values["EXACT"] = 0;
                    values["TANH_APPROXIMATION"] = 1;
                    values["SIGMOID_APPROXIMATION"] = 2;
                    return values;
                })();

                return GeluLayerParams;
            })();

            Specification.RangeStaticLayerParams = (function() {

                function RangeStaticLayerParams() {
                }

                RangeStaticLayerParams.prototype.endValue = 0;
                RangeStaticLayerParams.prototype.startValue = 0;
                RangeStaticLayerParams.prototype.stepSizeValue = 0;

                RangeStaticLayerParams.decode = function (reader, length) {
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
                };

                return RangeStaticLayerParams;
            })();

            Specification.RangeDynamicLayerParams = (function() {

                function RangeDynamicLayerParams() {
                }

                RangeDynamicLayerParams.prototype.startValue = 0;
                RangeDynamicLayerParams.prototype.stepSizeValue = 0;

                RangeDynamicLayerParams.decode = function (reader, length) {
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
                };

                return RangeDynamicLayerParams;
            })();

            Specification.SlidingWindowsLayerParams = (function() {

                function SlidingWindowsLayerParams() {
                }

                SlidingWindowsLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                SlidingWindowsLayerParams.prototype.windowSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                SlidingWindowsLayerParams.prototype.step = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;

                SlidingWindowsLayerParams.decode = function (reader, length) {
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
                };

                return SlidingWindowsLayerParams;
            })();

            Specification.LayerNormalizationLayerParams = (function() {

                function LayerNormalizationLayerParams() {
                    this.normalizedShape = [];
                }

                LayerNormalizationLayerParams.prototype.normalizedShape = [];
                LayerNormalizationLayerParams.prototype.eps = 0;
                LayerNormalizationLayerParams.prototype.gamma = null;
                LayerNormalizationLayerParams.prototype.beta = null;

                LayerNormalizationLayerParams.decode = function (reader, length) {
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
                };

                return LayerNormalizationLayerParams;
            })();

            Specification.NonMaximumSuppressionLayerParams = (function() {

                function NonMaximumSuppressionLayerParams() {
                }

                NonMaximumSuppressionLayerParams.prototype.iouThreshold = 0;
                NonMaximumSuppressionLayerParams.prototype.scoreThreshold = 0;
                NonMaximumSuppressionLayerParams.prototype.maxBoxes = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                NonMaximumSuppressionLayerParams.prototype.perClassSuppression = false;

                NonMaximumSuppressionLayerParams.decode = function (reader, length) {
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
                };

                return NonMaximumSuppressionLayerParams;
            })();

            Specification.ClampedReLULayerParams = (function() {

                function ClampedReLULayerParams() {
                }

                ClampedReLULayerParams.prototype.alpha = 0;
                ClampedReLULayerParams.prototype.beta = 0;

                ClampedReLULayerParams.decode = function (reader, length) {
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
                };

                return ClampedReLULayerParams;
            })();

            Specification.ArgSortLayerParams = (function() {

                function ArgSortLayerParams() {
                }

                ArgSortLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                ArgSortLayerParams.prototype.descending = false;

                ArgSortLayerParams.decode = function (reader, length) {
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
                };

                return ArgSortLayerParams;
            })();

            Specification.SliceBySizeLayerParams = (function() {

                function SliceBySizeLayerParams() {
                }

                SliceBySizeLayerParams.prototype.size = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                SliceBySizeLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                SliceBySizeLayerParams.decode = function (reader, length) {
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
                };

                return SliceBySizeLayerParams;
            })();

            Specification.NeuralNetworkClassifier = (function() {

                function NeuralNetworkClassifier() {
                    this.layers = [];
                    this.preprocessing = [];
                }

                NeuralNetworkClassifier.prototype.layers = [];
                NeuralNetworkClassifier.prototype.preprocessing = [];
                NeuralNetworkClassifier.prototype.arrayInputShapeMapping = 0;
                NeuralNetworkClassifier.prototype.imageInputShapeMapping = 0;
                NeuralNetworkClassifier.prototype.updateParams = null;
                NeuralNetworkClassifier.prototype.stringClassLabels = null;
                NeuralNetworkClassifier.prototype.int64ClassLabels = null;
                NeuralNetworkClassifier.prototype.labelProbabilityLayerName = "";

                const ClassLabelsSet = new Set([ "stringClassLabels", "int64ClassLabels"]);
                Object.defineProperty(NeuralNetworkClassifier.prototype, "ClassLabels", {
                    get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                });

                NeuralNetworkClassifier.decode = function (reader, length) {
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
                };

                return NeuralNetworkClassifier;
            })();

            Specification.OneHotLayerParams = (function() {

                function OneHotLayerParams() {
                }

                OneHotLayerParams.prototype.oneHotVectorSize = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                OneHotLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                OneHotLayerParams.prototype.onValue = 0;
                OneHotLayerParams.prototype.offValue = 0;

                OneHotLayerParams.decode = function (reader, length) {
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
                };

                return OneHotLayerParams;
            })();

            Specification.CumSumLayerParams = (function() {

                function CumSumLayerParams() {
                }

                CumSumLayerParams.prototype.axis = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                CumSumLayerParams.prototype.excludeFinalSum = false;
                CumSumLayerParams.prototype.reverse = false;

                CumSumLayerParams.decode = function (reader, length) {
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
                };

                return CumSumLayerParams;
            })();

            Specification.NeuralNetworkRegressor = (function() {

                function NeuralNetworkRegressor() {
                    this.layers = [];
                    this.preprocessing = [];
                }

                NeuralNetworkRegressor.prototype.layers = [];
                NeuralNetworkRegressor.prototype.preprocessing = [];
                NeuralNetworkRegressor.prototype.arrayInputShapeMapping = 0;
                NeuralNetworkRegressor.prototype.imageInputShapeMapping = 0;
                NeuralNetworkRegressor.prototype.updateParams = null;

                NeuralNetworkRegressor.decode = function (reader, length) {
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
                };

                return NeuralNetworkRegressor;
            })();

            Specification.NetworkUpdateParameters = (function() {

                function NetworkUpdateParameters() {
                    this.lossLayers = [];
                }

                NetworkUpdateParameters.prototype.lossLayers = [];
                NetworkUpdateParameters.prototype.optimizer = null;
                NetworkUpdateParameters.prototype.epochs = null;
                NetworkUpdateParameters.prototype.shuffle = null;
                NetworkUpdateParameters.prototype.seed = null;

                NetworkUpdateParameters.decode = function (reader, length) {
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
                };

                return NetworkUpdateParameters;
            })();

            Specification.LossLayer = (function() {

                function LossLayer() {
                }

                LossLayer.prototype.name = "";
                LossLayer.prototype.categoricalCrossEntropyLossLayer = null;
                LossLayer.prototype.meanSquaredErrorLossLayer = null;

                const LossLayerTypeSet = new Set([ "categoricalCrossEntropyLossLayer", "meanSquaredErrorLossLayer"]);
                Object.defineProperty(LossLayer.prototype, "LossLayerType", {
                    get: function() { return Object.keys(this).find((key) => LossLayerTypeSet.has(key) && this[key] != null); }
                });

                LossLayer.decode = function (reader, length) {
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
                };

                return LossLayer;
            })();

            Specification.CategoricalCrossEntropyLossLayer = (function() {

                function CategoricalCrossEntropyLossLayer() {
                }

                CategoricalCrossEntropyLossLayer.prototype.input = "";
                CategoricalCrossEntropyLossLayer.prototype.target = "";

                CategoricalCrossEntropyLossLayer.decode = function (reader, length) {
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
                };

                return CategoricalCrossEntropyLossLayer;
            })();

            Specification.MeanSquaredErrorLossLayer = (function() {

                function MeanSquaredErrorLossLayer() {
                }

                MeanSquaredErrorLossLayer.prototype.input = "";
                MeanSquaredErrorLossLayer.prototype.target = "";

                MeanSquaredErrorLossLayer.decode = function (reader, length) {
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
                };

                return MeanSquaredErrorLossLayer;
            })();

            Specification.Optimizer = (function() {

                function Optimizer() {
                }

                Optimizer.prototype.sgdOptimizer = null;
                Optimizer.prototype.adamOptimizer = null;

                const OptimizerTypeSet = new Set([ "sgdOptimizer", "adamOptimizer"]);
                Object.defineProperty(Optimizer.prototype, "OptimizerType", {
                    get: function() { return Object.keys(this).find((key) => OptimizerTypeSet.has(key) && this[key] != null); }
                });

                Optimizer.decode = function (reader, length) {
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
                };

                return Optimizer;
            })();

            Specification.SGDOptimizer = (function() {

                function SGDOptimizer() {
                }

                SGDOptimizer.prototype.learningRate = null;
                SGDOptimizer.prototype.miniBatchSize = null;
                SGDOptimizer.prototype.momentum = null;

                SGDOptimizer.decode = function (reader, length) {
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
                };

                return SGDOptimizer;
            })();

            Specification.AdamOptimizer = (function() {

                function AdamOptimizer() {
                }

                AdamOptimizer.prototype.learningRate = null;
                AdamOptimizer.prototype.miniBatchSize = null;
                AdamOptimizer.prototype.beta1 = null;
                AdamOptimizer.prototype.beta2 = null;
                AdamOptimizer.prototype.eps = null;

                AdamOptimizer.decode = function (reader, length) {
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
                };

                return AdamOptimizer;
            })();

            Specification.Normalizer = (function() {

                function Normalizer() {
                }

                Normalizer.prototype.normType = 0;

                Normalizer.decode = function (reader, length) {
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
                };

                Normalizer.NormType = (function() {
                    const values = {};
                    values["LMax"] = 0;
                    values["L1"] = 1;
                    values["L2"] = 2;
                    return values;
                })();

                return Normalizer;
            })();

            Specification.OneHotEncoder = (function() {

                function OneHotEncoder() {
                }

                OneHotEncoder.prototype.stringCategories = null;
                OneHotEncoder.prototype.int64Categories = null;
                OneHotEncoder.prototype.outputSparse = false;
                OneHotEncoder.prototype.handleUnknown = 0;

                const CategoryTypeSet = new Set([ "stringCategories", "int64Categories"]);
                Object.defineProperty(OneHotEncoder.prototype, "CategoryType", {
                    get: function() { return Object.keys(this).find((key) => CategoryTypeSet.has(key) && this[key] != null); }
                });

                OneHotEncoder.decode = function (reader, length) {
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
                };

                OneHotEncoder.HandleUnknown = (function() {
                    const values = {};
                    values["ErrorOnUnknown"] = 0;
                    values["IgnoreUnknown"] = 1;
                    return values;
                })();

                return OneHotEncoder;
            })();

            Specification.Scaler = (function() {

                function Scaler() {
                    this.shiftValue = [];
                    this.scaleValue = [];
                }

                Scaler.prototype.shiftValue = [];
                Scaler.prototype.scaleValue = [];

                Scaler.decode = function (reader, length) {
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
                };

                return Scaler;
            })();

            Specification.NonMaximumSuppression = (function() {

                function NonMaximumSuppression() {
                }

                NonMaximumSuppression.prototype.pickTop = null;
                NonMaximumSuppression.prototype.stringClassLabels = null;
                NonMaximumSuppression.prototype.int64ClassLabels = null;
                NonMaximumSuppression.prototype.iouThreshold = 0;
                NonMaximumSuppression.prototype.confidenceThreshold = 0;
                NonMaximumSuppression.prototype.confidenceInputFeatureName = "";
                NonMaximumSuppression.prototype.coordinatesInputFeatureName = "";
                NonMaximumSuppression.prototype.iouThresholdInputFeatureName = "";
                NonMaximumSuppression.prototype.confidenceThresholdInputFeatureName = "";
                NonMaximumSuppression.prototype.confidenceOutputFeatureName = "";
                NonMaximumSuppression.prototype.coordinatesOutputFeatureName = "";

                const SuppressionMethodSet = new Set([ "pickTop"]);
                Object.defineProperty(NonMaximumSuppression.prototype, "SuppressionMethod", {
                    get: function() { return Object.keys(this).find((key) => SuppressionMethodSet.has(key) && this[key] != null); }
                });

                const ClassLabelsSet = new Set([ "stringClassLabels", "int64ClassLabels"]);
                Object.defineProperty(NonMaximumSuppression.prototype, "ClassLabels", {
                    get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                });

                NonMaximumSuppression.decode = function (reader, length) {
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
                };

                NonMaximumSuppression.PickTop = (function() {

                    function PickTop() {
                    }

                    PickTop.prototype.perClass = false;

                    PickTop.decode = function (reader, length) {
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
                    };

                    return PickTop;
                })();

                return NonMaximumSuppression;
            })();

            Specification.LinearKernel = (function() {

                function LinearKernel() {
                }

                LinearKernel.decode = function (reader, length) {
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
                };

                return LinearKernel;
            })();

            Specification.RBFKernel = (function() {

                function RBFKernel() {
                }

                RBFKernel.prototype.gamma = 0;

                RBFKernel.decode = function (reader, length) {
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
                };

                return RBFKernel;
            })();

            Specification.PolyKernel = (function() {

                function PolyKernel() {
                }

                PolyKernel.prototype.degree = 0;
                PolyKernel.prototype.c = 0;
                PolyKernel.prototype.gamma = 0;

                PolyKernel.decode = function (reader, length) {
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
                };

                return PolyKernel;
            })();

            Specification.SigmoidKernel = (function() {

                function SigmoidKernel() {
                }

                SigmoidKernel.prototype.gamma = 0;
                SigmoidKernel.prototype.c = 0;

                SigmoidKernel.decode = function (reader, length) {
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
                };

                return SigmoidKernel;
            })();

            Specification.Kernel = (function() {

                function Kernel() {
                }

                Kernel.prototype.linearKernel = null;
                Kernel.prototype.rbfKernel = null;
                Kernel.prototype.polyKernel = null;
                Kernel.prototype.sigmoidKernel = null;

                const kernelSet = new Set([ "linearKernel", "rbfKernel", "polyKernel", "sigmoidKernel"]);
                Object.defineProperty(Kernel.prototype, "kernel", {
                    get: function() { return Object.keys(this).find((key) => kernelSet.has(key) && this[key] != null); }
                });

                Kernel.decode = function (reader, length) {
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
                };

                return Kernel;
            })();

            Specification.SparseNode = (function() {

                function SparseNode() {
                }

                SparseNode.prototype.index = 0;
                SparseNode.prototype.value = 0;

                SparseNode.decode = function (reader, length) {
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
                };

                return SparseNode;
            })();

            Specification.SparseVector = (function() {

                function SparseVector() {
                    this.nodes = [];
                }

                SparseVector.prototype.nodes = [];

                SparseVector.decode = function (reader, length) {
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
                };

                return SparseVector;
            })();

            Specification.SparseSupportVectors = (function() {

                function SparseSupportVectors() {
                    this.vectors = [];
                }

                SparseSupportVectors.prototype.vectors = [];

                SparseSupportVectors.decode = function (reader, length) {
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
                };

                return SparseSupportVectors;
            })();

            Specification.DenseVector = (function() {

                function DenseVector() {
                    this.values = [];
                }

                DenseVector.prototype.values = [];

                DenseVector.decode = function (reader, length) {
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
                };

                return DenseVector;
            })();

            Specification.DenseSupportVectors = (function() {

                function DenseSupportVectors() {
                    this.vectors = [];
                }

                DenseSupportVectors.prototype.vectors = [];

                DenseSupportVectors.decode = function (reader, length) {
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
                };

                return DenseSupportVectors;
            })();

            Specification.Coefficients = (function() {

                function Coefficients() {
                    this.alpha = [];
                }

                Coefficients.prototype.alpha = [];

                Coefficients.decode = function (reader, length) {
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
                };

                return Coefficients;
            })();

            Specification.SupportVectorRegressor = (function() {

                function SupportVectorRegressor() {
                }

                SupportVectorRegressor.prototype.kernel = null;
                SupportVectorRegressor.prototype.sparseSupportVectors = null;
                SupportVectorRegressor.prototype.denseSupportVectors = null;
                SupportVectorRegressor.prototype.coefficients = null;
                SupportVectorRegressor.prototype.rho = 0;

                const supportVectorsSet = new Set([ "sparseSupportVectors", "denseSupportVectors"]);
                Object.defineProperty(SupportVectorRegressor.prototype, "supportVectors", {
                    get: function() { return Object.keys(this).find((key) => supportVectorsSet.has(key) && this[key] != null); }
                });

                SupportVectorRegressor.decode = function (reader, length) {
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
                };

                return SupportVectorRegressor;
            })();

            Specification.SupportVectorClassifier = (function() {

                function SupportVectorClassifier() {
                    this.numberOfSupportVectorsPerClass = [];
                    this.coefficients = [];
                    this.rho = [];
                    this.probA = [];
                    this.probB = [];
                }

                SupportVectorClassifier.prototype.kernel = null;
                SupportVectorClassifier.prototype.numberOfSupportVectorsPerClass = [];
                SupportVectorClassifier.prototype.sparseSupportVectors = null;
                SupportVectorClassifier.prototype.denseSupportVectors = null;
                SupportVectorClassifier.prototype.coefficients = [];
                SupportVectorClassifier.prototype.rho = [];
                SupportVectorClassifier.prototype.probA = [];
                SupportVectorClassifier.prototype.probB = [];
                SupportVectorClassifier.prototype.stringClassLabels = null;
                SupportVectorClassifier.prototype.int64ClassLabels = null;

                const supportVectorsSet = new Set([ "sparseSupportVectors", "denseSupportVectors"]);
                Object.defineProperty(SupportVectorClassifier.prototype, "supportVectors", {
                    get: function() { return Object.keys(this).find((key) => supportVectorsSet.has(key) && this[key] != null); }
                });

                const ClassLabelsSet = new Set([ "stringClassLabels", "int64ClassLabels"]);
                Object.defineProperty(SupportVectorClassifier.prototype, "ClassLabels", {
                    get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                });

                SupportVectorClassifier.decode = function (reader, length) {
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
                };

                return SupportVectorClassifier;
            })();

            Specification.TreeEnsemblePostEvaluationTransform = (function() {
                const values = {};
                values["NoTransform"] = 0;
                values["Classification_SoftMax"] = 1;
                values["Regression_Logistic"] = 2;
                values["Classification_SoftMaxWithZeroClassReference"] = 3;
                return values;
            })();

            Specification.TreeEnsembleParameters = (function() {

                function TreeEnsembleParameters() {
                    this.nodes = [];
                    this.basePredictionValue = [];
                }

                TreeEnsembleParameters.prototype.nodes = [];
                TreeEnsembleParameters.prototype.numPredictionDimensions = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                TreeEnsembleParameters.prototype.basePredictionValue = [];

                TreeEnsembleParameters.decode = function (reader, length) {
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
                };

                TreeEnsembleParameters.TreeNode = (function() {

                    function TreeNode() {
                        this.evaluationInfo = [];
                    }

                    TreeNode.prototype.treeId = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    TreeNode.prototype.nodeId = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    TreeNode.prototype.nodeBehavior = 0;
                    TreeNode.prototype.branchFeatureIndex = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    TreeNode.prototype.branchFeatureValue = 0;
                    TreeNode.prototype.trueChildNodeId = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    TreeNode.prototype.falseChildNodeId = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    TreeNode.prototype.missingValueTracksTrueChild = false;
                    TreeNode.prototype.evaluationInfo = [];
                    TreeNode.prototype.relativeHitRate = 0;

                    TreeNode.decode = function (reader, length) {
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
                    };

                    TreeNode.TreeNodeBehavior = (function() {
                        const values = {};
                        values["BranchOnValueLessThanEqual"] = 0;
                        values["BranchOnValueLessThan"] = 1;
                        values["BranchOnValueGreaterThanEqual"] = 2;
                        values["BranchOnValueGreaterThan"] = 3;
                        values["BranchOnValueEqual"] = 4;
                        values["BranchOnValueNotEqual"] = 5;
                        values["LeafNode"] = 6;
                        return values;
                    })();

                    TreeNode.EvaluationInfo = (function() {

                        function EvaluationInfo() {
                        }

                        EvaluationInfo.prototype.evaluationIndex = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                        EvaluationInfo.prototype.evaluationValue = 0;

                        EvaluationInfo.decode = function (reader, length) {
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
                        };

                        return EvaluationInfo;
                    })();

                    return TreeNode;
                })();

                return TreeEnsembleParameters;
            })();

            Specification.TreeEnsembleClassifier = (function() {

                function TreeEnsembleClassifier() {
                }

                TreeEnsembleClassifier.prototype.treeEnsemble = null;
                TreeEnsembleClassifier.prototype.postEvaluationTransform = 0;
                TreeEnsembleClassifier.prototype.stringClassLabels = null;
                TreeEnsembleClassifier.prototype.int64ClassLabels = null;

                const ClassLabelsSet = new Set([ "stringClassLabels", "int64ClassLabels"]);
                Object.defineProperty(TreeEnsembleClassifier.prototype, "ClassLabels", {
                    get: function() { return Object.keys(this).find((key) => ClassLabelsSet.has(key) && this[key] != null); }
                });

                TreeEnsembleClassifier.decode = function (reader, length) {
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
                };

                return TreeEnsembleClassifier;
            })();

            Specification.TreeEnsembleRegressor = (function() {

                function TreeEnsembleRegressor() {
                }

                TreeEnsembleRegressor.prototype.treeEnsemble = null;
                TreeEnsembleRegressor.prototype.postEvaluationTransform = 0;

                TreeEnsembleRegressor.decode = function (reader, length) {
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
                };

                return TreeEnsembleRegressor;
            })();

            Specification.ItemSimilarityRecommender = (function() {

                function ItemSimilarityRecommender() {
                    this.itemItemSimilarities = [];
                }

                ItemSimilarityRecommender.prototype.itemItemSimilarities = [];
                ItemSimilarityRecommender.prototype.itemStringIds = null;
                ItemSimilarityRecommender.prototype.itemInt64Ids = null;
                ItemSimilarityRecommender.prototype.itemInputFeatureName = "";
                ItemSimilarityRecommender.prototype.numRecommendationsInputFeatureName = "";
                ItemSimilarityRecommender.prototype.itemRestrictionInputFeatureName = "";
                ItemSimilarityRecommender.prototype.itemExclusionInputFeatureName = "";
                ItemSimilarityRecommender.prototype.recommendedItemListOutputFeatureName = "";
                ItemSimilarityRecommender.prototype.recommendedItemScoreOutputFeatureName = "";

                ItemSimilarityRecommender.decode = function (reader, length) {
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
                };

                ItemSimilarityRecommender.ConnectedItem = (function() {

                    function ConnectedItem() {
                    }

                    ConnectedItem.prototype.itemId = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    ConnectedItem.prototype.similarityScore = 0;

                    ConnectedItem.decode = function (reader, length) {
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
                    };

                    return ConnectedItem;
                })();

                ItemSimilarityRecommender.SimilarItems = (function() {

                    function SimilarItems() {
                        this.similarItemList = [];
                    }

                    SimilarItems.prototype.itemId = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                    SimilarItems.prototype.similarItemList = [];
                    SimilarItems.prototype.itemScoreAdjustment = 0;

                    SimilarItems.decode = function (reader, length) {
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
                    };

                    return SimilarItems;
                })();

                return ItemSimilarityRecommender;
            })();

            Specification.LinkedModel = (function() {

                function LinkedModel() {
                }

                LinkedModel.prototype.linkedModelFile = null;

                const LinkTypeSet = new Set([ "linkedModelFile"]);
                Object.defineProperty(LinkedModel.prototype, "LinkType", {
                    get: function() { return Object.keys(this).find((key) => LinkTypeSet.has(key) && this[key] != null); }
                });

                LinkedModel.decode = function (reader, length) {
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
                };

                return LinkedModel;
            })();

            Specification.LinkedModelFile = (function() {

                function LinkedModelFile() {
                }

                LinkedModelFile.prototype.linkedModelFileName = null;
                LinkedModelFile.prototype.linkedModelSearchPath = null;

                LinkedModelFile.decode = function (reader, length) {
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
                };

                return LinkedModelFile;
            })();

            return Specification;
        })();

        return CoreML;
    })();
    return $root;
})(protobuf);
