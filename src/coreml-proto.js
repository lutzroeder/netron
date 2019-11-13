/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.coreml || ($protobuf.roots.coreml = {});
    
    $root.CoreML = (function() {
    
        var CoreML = {};
    
        CoreML.Specification = (function() {
    
            var Specification = {};
    
            Specification.Pipeline = (function() {
    
                function Pipeline(properties) {
                    this.models = [];
                    this.names = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Pipeline.prototype.models = $util.emptyArray;
                Pipeline.prototype.names = $util.emptyArray;
    
                Pipeline.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Pipeline();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.models && message.models.length))
                                message.models = [];
                            message.models.push($root.CoreML.Specification.Model.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.names && message.names.length))
                                message.names = [];
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
    
                function PipelineClassifier(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PipelineClassifier.prototype.pipeline = null;
    
                PipelineClassifier.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PipelineClassifier();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function PipelineRegressor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PipelineRegressor.prototype.pipeline = null;
    
                PipelineRegressor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PipelineRegressor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FeatureDescription(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FeatureDescription.prototype.name = "";
                FeatureDescription.prototype.shortDescription = "";
                FeatureDescription.prototype.type = null;
    
                FeatureDescription.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FeatureDescription();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Metadata(properties) {
                    this.userDefined = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Metadata.prototype.shortDescription = "";
                Metadata.prototype.versionString = "";
                Metadata.prototype.author = "";
                Metadata.prototype.license = "";
                Metadata.prototype.userDefined = $util.emptyObject;
    
                Metadata.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Metadata(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                            reader.skip().pos++;
                            if (message.userDefined === $util.emptyObject)
                                message.userDefined = {};
                            key = reader.string();
                            reader.pos++;
                            message.userDefined[key] = reader.string();
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
    
                function ModelDescription(properties) {
                    this.input = [];
                    this.output = [];
                    this.trainingInput = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ModelDescription.prototype.input = $util.emptyArray;
                ModelDescription.prototype.output = $util.emptyArray;
                ModelDescription.prototype.predictedFeatureName = "";
                ModelDescription.prototype.predictedProbabilitiesName = "";
                ModelDescription.prototype.trainingInput = $util.emptyArray;
                ModelDescription.prototype.metadata = null;
    
                ModelDescription.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ModelDescription();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.input && message.input.length))
                                message.input = [];
                            message.input.push($root.CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                            break;
                        case 10:
                            if (!(message.output && message.output.length))
                                message.output = [];
                            message.output.push($root.CoreML.Specification.FeatureDescription.decode(reader, reader.uint32()));
                            break;
                        case 11:
                            message.predictedFeatureName = reader.string();
                            break;
                        case 12:
                            message.predictedProbabilitiesName = reader.string();
                            break;
                        case 50:
                            if (!(message.trainingInput && message.trainingInput.length))
                                message.trainingInput = [];
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
    
            Specification.Model = (function() {
    
                function Model(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
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
    
                var $oneOfFields;
    
                Object.defineProperty(Model.prototype, "Type", {
                    get: $util.oneOfGetter($oneOfFields = ["pipelineClassifier", "pipelineRegressor", "pipeline", "glmRegressor", "supportVectorRegressor", "treeEnsembleRegressor", "neuralNetworkRegressor", "bayesianProbitRegressor", "glmClassifier", "supportVectorClassifier", "treeEnsembleClassifier", "neuralNetworkClassifier", "kNearestNeighborsClassifier", "neuralNetwork", "itemSimilarityRecommender", "customModel", "linkedModel", "oneHotEncoder", "imputer", "featureVectorizer", "dictVectorizer", "scaler", "categoricalMapping", "normalizer", "arrayFeatureExtractor", "nonMaximumSuppression", "identity", "textClassifier", "wordTagger", "visionFeaturePrint", "soundAnalysisPreprocessing", "gazetteer", "wordEmbedding"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Model.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Model();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                var CoreMLModels = {};
    
                CoreMLModels.VisionFeaturePrint = (function() {
    
                    function VisionFeaturePrint(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    VisionFeaturePrint.prototype.scene = null;
    
                    var $oneOfFields;
    
                    Object.defineProperty(VisionFeaturePrint.prototype, "VisionFeaturePrintType", {
                        get: $util.oneOfGetter($oneOfFields = ["scene"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    VisionFeaturePrint.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 20:
                                message.scene = $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                            }
                        }
                        return message;
                    };
    
                    VisionFeaturePrint.Scene = (function() {
    
                        function Scene(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Scene.prototype.version = 0;
    
                        Scene.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.VisionFeaturePrint.Scene();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "SCENE_VERSION_INVALID"] = 0;
                            values[valuesById[1] = "SCENE_VERSION_1"] = 1;
                            return values;
                        })();
    
                        return Scene;
                    })();
    
                    return VisionFeaturePrint;
                })();
    
                CoreMLModels.TextClassifier = (function() {
    
                    function TextClassifier(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    TextClassifier.prototype.revision = 0;
                    TextClassifier.prototype.language = "";
                    TextClassifier.prototype.modelParameterData = $util.newBuffer([]);
                    TextClassifier.prototype.stringClassLabels = null;
    
                    var $oneOfFields;
    
                    Object.defineProperty(TextClassifier.prototype, "ClassLabels", {
                        get: $util.oneOfGetter($oneOfFields = ["stringClassLabels"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    TextClassifier.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.TextClassifier();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function WordTagger(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    WordTagger.prototype.revision = 0;
                    WordTagger.prototype.language = "";
                    WordTagger.prototype.tokensOutputFeatureName = "";
                    WordTagger.prototype.tokenTagsOutputFeatureName = "";
                    WordTagger.prototype.tokenLocationsOutputFeatureName = "";
                    WordTagger.prototype.tokenLengthsOutputFeatureName = "";
                    WordTagger.prototype.modelParameterData = $util.newBuffer([]);
                    WordTagger.prototype.stringTags = null;
    
                    var $oneOfFields;
    
                    Object.defineProperty(WordTagger.prototype, "Tags", {
                        get: $util.oneOfGetter($oneOfFields = ["stringTags"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    WordTagger.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.WordTagger();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function Gazetteer(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    Gazetteer.prototype.revision = 0;
                    Gazetteer.prototype.language = "";
                    Gazetteer.prototype.modelParameterData = $util.newBuffer([]);
                    Gazetteer.prototype.stringClassLabels = null;
    
                    var $oneOfFields;
    
                    Object.defineProperty(Gazetteer.prototype, "ClassLabels", {
                        get: $util.oneOfGetter($oneOfFields = ["stringClassLabels"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    Gazetteer.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.Gazetteer();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function WordEmbedding(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    WordEmbedding.prototype.revision = 0;
                    WordEmbedding.prototype.language = "";
                    WordEmbedding.prototype.modelParameterData = $util.newBuffer([]);
    
                    WordEmbedding.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.WordEmbedding();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function SoundAnalysisPreprocessing(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    SoundAnalysisPreprocessing.prototype.vggish = null;
    
                    var $oneOfFields;
    
                    Object.defineProperty(SoundAnalysisPreprocessing.prototype, "SoundAnalysisPreprocessingType", {
                        get: $util.oneOfGetter($oneOfFields = ["vggish"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    SoundAnalysisPreprocessing.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                        function Vggish(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Vggish.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoreMLModels.SoundAnalysisPreprocessing.Vggish();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
    
                function StringToInt64Map(properties) {
                    this.map = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                StringToInt64Map.prototype.map = $util.emptyObject;
    
                StringToInt64Map.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.StringToInt64Map(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            reader.skip().pos++;
                            if (message.map === $util.emptyObject)
                                message.map = {};
                            key = reader.string();
                            reader.pos++;
                            message.map[key] = reader.int64();
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
    
                function Int64ToStringMap(properties) {
                    this.map = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64ToStringMap.prototype.map = $util.emptyObject;
    
                Int64ToStringMap.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64ToStringMap(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            reader.skip().pos++;
                            if (message.map === $util.emptyObject)
                                message.map = {};
                            key = reader.int64();
                            reader.pos++;
                            message.map[typeof key === "object" ? $util.longToHash(key) : key] = reader.string();
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
    
                function StringToDoubleMap(properties) {
                    this.map = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                StringToDoubleMap.prototype.map = $util.emptyObject;
    
                StringToDoubleMap.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.StringToDoubleMap(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            reader.skip().pos++;
                            if (message.map === $util.emptyObject)
                                message.map = {};
                            key = reader.string();
                            reader.pos++;
                            message.map[key] = reader.double();
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
    
                function Int64ToDoubleMap(properties) {
                    this.map = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64ToDoubleMap.prototype.map = $util.emptyObject;
    
                Int64ToDoubleMap.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64ToDoubleMap(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            reader.skip().pos++;
                            if (message.map === $util.emptyObject)
                                message.map = {};
                            key = reader.int64();
                            reader.pos++;
                            message.map[typeof key === "object" ? $util.longToHash(key) : key] = reader.double();
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
    
                function StringVector(properties) {
                    this.vector = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                StringVector.prototype.vector = $util.emptyArray;
    
                StringVector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.StringVector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.vector && message.vector.length))
                                message.vector = [];
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
    
                function Int64Vector(properties) {
                    this.vector = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64Vector.prototype.vector = $util.emptyArray;
    
                Int64Vector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64Vector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.vector && message.vector.length))
                                message.vector = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.vector.push(reader.int64());
                            } else
                                message.vector.push(reader.int64());
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
    
                function FloatVector(properties) {
                    this.vector = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FloatVector.prototype.vector = $util.emptyArray;
    
                FloatVector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FloatVector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.vector && message.vector.length))
                                message.vector = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.vector.push(reader.float());
                            } else
                                message.vector.push(reader.float());
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
    
                function DoubleVector(properties) {
                    this.vector = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DoubleVector.prototype.vector = $util.emptyArray;
    
                DoubleVector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DoubleVector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.vector && message.vector.length))
                                message.vector = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.vector.push(reader.double());
                            } else
                                message.vector.push(reader.double());
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
    
                function Int64Range(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64Range.prototype.minValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Int64Range.prototype.maxValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                Int64Range.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64Range();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Int64Set(properties) {
                    this.values = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64Set.prototype.values = $util.emptyArray;
    
                Int64Set.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64Set();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.values && message.values.length))
                                message.values = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.values.push(reader.int64());
                            } else
                                message.values.push(reader.int64());
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
    
                function DoubleRange(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DoubleRange.prototype.minValue = 0;
                DoubleRange.prototype.maxValue = 0;
    
                DoubleRange.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DoubleRange();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Int64FeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64FeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64FeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function DoubleFeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DoubleFeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DoubleFeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function StringFeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                StringFeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.StringFeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SizeRange(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SizeRange.prototype.lowerBound = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                SizeRange.prototype.upperBound = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                SizeRange.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SizeRange();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ImageFeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ImageFeatureType.prototype.width = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ImageFeatureType.prototype.height = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ImageFeatureType.prototype.enumeratedSizes = null;
                ImageFeatureType.prototype.imageSizeRange = null;
                ImageFeatureType.prototype.colorSpace = 0;
    
                var $oneOfFields;
    
                Object.defineProperty(ImageFeatureType.prototype, "SizeFlexibility", {
                    get: $util.oneOfGetter($oneOfFields = ["enumeratedSizes", "imageSizeRange"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                ImageFeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ImageFeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "INVALID_COLOR_SPACE"] = 0;
                    values[valuesById[10] = "GRAYSCALE"] = 10;
                    values[valuesById[20] = "RGB"] = 20;
                    values[valuesById[30] = "BGR"] = 30;
                    return values;
                })();
    
                ImageFeatureType.ImageSize = (function() {
    
                    function ImageSize(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ImageSize.prototype.width = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    ImageSize.prototype.height = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                    ImageSize.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ImageFeatureType.ImageSize();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function EnumeratedImageSizes(properties) {
                        this.sizes = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    EnumeratedImageSizes.prototype.sizes = $util.emptyArray;
    
                    EnumeratedImageSizes.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ImageFeatureType.EnumeratedImageSizes();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.sizes && message.sizes.length))
                                    message.sizes = [];
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
    
                    function ImageSizeRange(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ImageSizeRange.prototype.widthRange = null;
                    ImageSizeRange.prototype.heightRange = null;
    
                    ImageSizeRange.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ImageFeatureType.ImageSizeRange();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function ArrayFeatureType(properties) {
                    this.shape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ArrayFeatureType.prototype.shape = $util.emptyArray;
                ArrayFeatureType.prototype.dataType = 0;
                ArrayFeatureType.prototype.enumeratedShapes = null;
                ArrayFeatureType.prototype.shapeRange = null;
    
                var $oneOfFields;
    
                Object.defineProperty(ArrayFeatureType.prototype, "ShapeFlexibility", {
                    get: $util.oneOfGetter($oneOfFields = ["enumeratedShapes", "shapeRange"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                ArrayFeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArrayFeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shape && message.shape.length))
                                message.shape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shape.push(reader.int64());
                            } else
                                message.shape.push(reader.int64());
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
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                ArrayFeatureType.ArrayDataType = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "INVALID_ARRAY_DATA_TYPE"] = 0;
                    values[valuesById[65568] = "FLOAT32"] = 65568;
                    values[valuesById[65600] = "DOUBLE"] = 65600;
                    values[valuesById[131104] = "INT32"] = 131104;
                    return values;
                })();
    
                ArrayFeatureType.Shape = (function() {
    
                    function Shape(properties) {
                        this.shape = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    Shape.prototype.shape = $util.emptyArray;
    
                    Shape.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArrayFeatureType.Shape();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.shape && message.shape.length))
                                    message.shape = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    while (reader.pos < end2)
                                        message.shape.push(reader.int64());
                                } else
                                    message.shape.push(reader.int64());
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
    
                    function EnumeratedShapes(properties) {
                        this.shapes = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    EnumeratedShapes.prototype.shapes = $util.emptyArray;
    
                    EnumeratedShapes.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArrayFeatureType.EnumeratedShapes();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.shapes && message.shapes.length))
                                    message.shapes = [];
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
    
                    function ShapeRange(properties) {
                        this.sizeRanges = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ShapeRange.prototype.sizeRanges = $util.emptyArray;
    
                    ShapeRange.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArrayFeatureType.ShapeRange();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.sizeRanges && message.sizeRanges.length))
                                    message.sizeRanges = [];
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
    
                function DictionaryFeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DictionaryFeatureType.prototype.int64KeyType = null;
                DictionaryFeatureType.prototype.stringKeyType = null;
    
                var $oneOfFields;
    
                Object.defineProperty(DictionaryFeatureType.prototype, "KeyType", {
                    get: $util.oneOfGetter($oneOfFields = ["int64KeyType", "stringKeyType"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                DictionaryFeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DictionaryFeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SequenceFeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SequenceFeatureType.prototype.int64Type = null;
                SequenceFeatureType.prototype.stringType = null;
                SequenceFeatureType.prototype.sizeRange = null;
    
                var $oneOfFields;
    
                Object.defineProperty(SequenceFeatureType.prototype, "Type", {
                    get: $util.oneOfGetter($oneOfFields = ["int64Type", "stringType"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                SequenceFeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SequenceFeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FeatureType(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FeatureType.prototype.int64Type = null;
                FeatureType.prototype.doubleType = null;
                FeatureType.prototype.stringType = null;
                FeatureType.prototype.imageType = null;
                FeatureType.prototype.multiArrayType = null;
                FeatureType.prototype.dictionaryType = null;
                FeatureType.prototype.sequenceType = null;
                FeatureType.prototype.isOptional = false;
    
                var $oneOfFields;
    
                Object.defineProperty(FeatureType.prototype, "Type", {
                    get: $util.oneOfGetter($oneOfFields = ["int64Type", "doubleType", "stringType", "imageType", "multiArrayType", "dictionaryType", "sequenceType"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                FeatureType.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FeatureType();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ArrayFeatureExtractor(properties) {
                    this.extractIndex = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ArrayFeatureExtractor.prototype.extractIndex = $util.emptyArray;
    
                ArrayFeatureExtractor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArrayFeatureExtractor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.extractIndex && message.extractIndex.length))
                                message.extractIndex = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.extractIndex.push(reader.uint64());
                            } else
                                message.extractIndex.push(reader.uint64());
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
    
                function BayesianProbitRegressor(properties) {
                    this.features = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BayesianProbitRegressor.prototype.numberOfFeatures = 0;
                BayesianProbitRegressor.prototype.bias = null;
                BayesianProbitRegressor.prototype.features = $util.emptyArray;
                BayesianProbitRegressor.prototype.regressionInputFeatureName = "";
                BayesianProbitRegressor.prototype.optimismInputFeatureName = "";
                BayesianProbitRegressor.prototype.samplingScaleInputFeatureName = "";
                BayesianProbitRegressor.prototype.samplingTruncationInputFeatureName = "";
                BayesianProbitRegressor.prototype.meanOutputFeatureName = "";
                BayesianProbitRegressor.prototype.varianceOutputFeatureName = "";
                BayesianProbitRegressor.prototype.pessimisticProbabilityOutputFeatureName = "";
                BayesianProbitRegressor.prototype.sampledProbabilityOutputFeatureName = "";
    
                BayesianProbitRegressor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BayesianProbitRegressor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.numberOfFeatures = reader.uint32();
                            break;
                        case 2:
                            message.bias = $root.CoreML.Specification.BayesianProbitRegressor.Gaussian.decode(reader, reader.uint32());
                            break;
                        case 3:
                            if (!(message.features && message.features.length))
                                message.features = [];
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
    
                    function Gaussian(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    Gaussian.prototype.mean = 0;
                    Gaussian.prototype.precision = 0;
    
                    Gaussian.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BayesianProbitRegressor.Gaussian();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function FeatureValueWeight(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    FeatureValueWeight.prototype.featureValue = 0;
                    FeatureValueWeight.prototype.featureWeight = null;
    
                    FeatureValueWeight.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureValueWeight();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function FeatureWeight(properties) {
                        this.weights = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    FeatureWeight.prototype.featureId = 0;
                    FeatureWeight.prototype.weights = $util.emptyArray;
    
                    FeatureWeight.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BayesianProbitRegressor.FeatureWeight();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                message.featureId = reader.uint32();
                                break;
                            case 2:
                                if (!(message.weights && message.weights.length))
                                    message.weights = [];
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
    
                function CategoricalMapping(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CategoricalMapping.prototype.stringToInt64Map = null;
                CategoricalMapping.prototype.int64ToStringMap = null;
                CategoricalMapping.prototype.strValue = "";
                CategoricalMapping.prototype.int64Value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                var $oneOfFields;
    
                Object.defineProperty(CategoricalMapping.prototype, "MappingType", {
                    get: $util.oneOfGetter($oneOfFields = ["stringToInt64Map", "int64ToStringMap"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(CategoricalMapping.prototype, "ValueOnUnknown", {
                    get: $util.oneOfGetter($oneOfFields = ["strValue", "int64Value"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                CategoricalMapping.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CategoricalMapping();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function CustomModel(properties) {
                    this.parameters = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CustomModel.prototype.className = "";
                CustomModel.prototype.parameters = $util.emptyObject;
                CustomModel.prototype.description = "";
    
                CustomModel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CustomModel(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 10:
                            message.className = reader.string();
                            break;
                        case 30:
                            reader.skip().pos++;
                            if (message.parameters === $util.emptyObject)
                                message.parameters = {};
                            key = reader.string();
                            reader.pos++;
                            message.parameters[key] = $root.CoreML.Specification.CustomModel.CustomModelParamValue.decode(reader, reader.uint32());
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
    
                    function CustomModelParamValue(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    CustomModelParamValue.prototype.doubleValue = 0;
                    CustomModelParamValue.prototype.stringValue = "";
                    CustomModelParamValue.prototype.intValue = 0;
                    CustomModelParamValue.prototype.longValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                    CustomModelParamValue.prototype.boolValue = false;
                    CustomModelParamValue.prototype.bytesValue = $util.newBuffer([]);
    
                    var $oneOfFields;
    
                    Object.defineProperty(CustomModelParamValue.prototype, "value", {
                        get: $util.oneOfGetter($oneOfFields = ["doubleValue", "stringValue", "intValue", "longValue", "boolValue", "bytesValue"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    CustomModelParamValue.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CustomModel.CustomModelParamValue();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function DictVectorizer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DictVectorizer.prototype.stringToIndex = null;
                DictVectorizer.prototype.int64ToIndex = null;
    
                var $oneOfFields;
    
                Object.defineProperty(DictVectorizer.prototype, "Map", {
                    get: $util.oneOfGetter($oneOfFields = ["stringToIndex", "int64ToIndex"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                DictVectorizer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DictVectorizer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FeatureVectorizer(properties) {
                    this.inputList = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FeatureVectorizer.prototype.inputList = $util.emptyArray;
    
                FeatureVectorizer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FeatureVectorizer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.inputList && message.inputList.length))
                                message.inputList = [];
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
    
                    function InputColumn(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    InputColumn.prototype.inputColumn = "";
                    InputColumn.prototype.inputDimensions = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                    InputColumn.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FeatureVectorizer.InputColumn();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function GLMRegressor(properties) {
                    this.weights = [];
                    this.offset = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GLMRegressor.prototype.weights = $util.emptyArray;
                GLMRegressor.prototype.offset = $util.emptyArray;
                GLMRegressor.prototype.postEvaluationTransform = 0;
    
                GLMRegressor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GLMRegressor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.weights && message.weights.length))
                                message.weights = [];
                            message.weights.push($root.CoreML.Specification.GLMRegressor.DoubleArray.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.offset && message.offset.length))
                                message.offset = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.offset.push(reader.double());
                            } else
                                message.offset.push(reader.double());
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
    
                    function DoubleArray(properties) {
                        this.value = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    DoubleArray.prototype.value = $util.emptyArray;
    
                    DoubleArray.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GLMRegressor.DoubleArray();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.value && message.value.length))
                                    message.value = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    while (reader.pos < end2)
                                        message.value.push(reader.double());
                                } else
                                    message.value.push(reader.double());
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "NoTransform"] = 0;
                    values[valuesById[1] = "Logit"] = 1;
                    values[valuesById[2] = "Probit"] = 2;
                    return values;
                })();
    
                return GLMRegressor;
            })();
    
            Specification.GLMClassifier = (function() {
    
                function GLMClassifier(properties) {
                    this.weights = [];
                    this.offset = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GLMClassifier.prototype.weights = $util.emptyArray;
                GLMClassifier.prototype.offset = $util.emptyArray;
                GLMClassifier.prototype.postEvaluationTransform = 0;
                GLMClassifier.prototype.classEncoding = 0;
                GLMClassifier.prototype.stringClassLabels = null;
                GLMClassifier.prototype.int64ClassLabels = null;
    
                var $oneOfFields;
    
                Object.defineProperty(GLMClassifier.prototype, "ClassLabels", {
                    get: $util.oneOfGetter($oneOfFields = ["stringClassLabels", "int64ClassLabels"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                GLMClassifier.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GLMClassifier();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.weights && message.weights.length))
                                message.weights = [];
                            message.weights.push($root.CoreML.Specification.GLMClassifier.DoubleArray.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.offset && message.offset.length))
                                message.offset = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.offset.push(reader.double());
                            } else
                                message.offset.push(reader.double());
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
    
                    function DoubleArray(properties) {
                        this.value = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    DoubleArray.prototype.value = $util.emptyArray;
    
                    DoubleArray.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GLMClassifier.DoubleArray();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.value && message.value.length))
                                    message.value = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    while (reader.pos < end2)
                                        message.value.push(reader.double());
                                } else
                                    message.value.push(reader.double());
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "Logit"] = 0;
                    values[valuesById[1] = "Probit"] = 1;
                    return values;
                })();
    
                GLMClassifier.ClassEncoding = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "ReferenceClass"] = 0;
                    values[valuesById[1] = "OneVsRest"] = 1;
                    return values;
                })();
    
                return GLMClassifier;
            })();
    
            Specification.KNearestNeighborsClassifier = (function() {
    
                function KNearestNeighborsClassifier(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                KNearestNeighborsClassifier.prototype.nearestNeighborsIndex = null;
                KNearestNeighborsClassifier.prototype.numberOfNeighbors = null;
                KNearestNeighborsClassifier.prototype.stringClassLabels = null;
                KNearestNeighborsClassifier.prototype.int64ClassLabels = null;
                KNearestNeighborsClassifier.prototype.defaultStringLabel = "";
                KNearestNeighborsClassifier.prototype.defaultInt64Label = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                KNearestNeighborsClassifier.prototype.uniformWeighting = null;
                KNearestNeighborsClassifier.prototype.inverseDistanceWeighting = null;
    
                var $oneOfFields;
    
                Object.defineProperty(KNearestNeighborsClassifier.prototype, "ClassLabels", {
                    get: $util.oneOfGetter($oneOfFields = ["stringClassLabels", "int64ClassLabels"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(KNearestNeighborsClassifier.prototype, "DefaultClassLabel", {
                    get: $util.oneOfGetter($oneOfFields = ["defaultStringLabel", "defaultInt64Label"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(KNearestNeighborsClassifier.prototype, "WeightingScheme", {
                    get: $util.oneOfGetter($oneOfFields = ["uniformWeighting", "inverseDistanceWeighting"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                KNearestNeighborsClassifier.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.KNearestNeighborsClassifier();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function NearestNeighborsIndex(properties) {
                    this.floatSamples = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NearestNeighborsIndex.prototype.numberOfDimensions = 0;
                NearestNeighborsIndex.prototype.floatSamples = $util.emptyArray;
                NearestNeighborsIndex.prototype.linearIndex = null;
                NearestNeighborsIndex.prototype.singleKdTreeIndex = null;
                NearestNeighborsIndex.prototype.squaredEuclideanDistance = null;
    
                var $oneOfFields;
    
                Object.defineProperty(NearestNeighborsIndex.prototype, "IndexType", {
                    get: $util.oneOfGetter($oneOfFields = ["linearIndex", "singleKdTreeIndex"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(NearestNeighborsIndex.prototype, "DistanceFunction", {
                    get: $util.oneOfGetter($oneOfFields = ["squaredEuclideanDistance"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                NearestNeighborsIndex.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NearestNeighborsIndex();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.numberOfDimensions = reader.int32();
                            break;
                        case 2:
                            if (!(message.floatSamples && message.floatSamples.length))
                                message.floatSamples = [];
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
    
                function UniformWeighting(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                UniformWeighting.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.UniformWeighting();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function InverseDistanceWeighting(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                InverseDistanceWeighting.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.InverseDistanceWeighting();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LinearIndex(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LinearIndex.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LinearIndex();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SingleKdTreeIndex(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SingleKdTreeIndex.prototype.leafSize = 0;
    
                SingleKdTreeIndex.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SingleKdTreeIndex();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SquaredEuclideanDistance(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SquaredEuclideanDistance.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SquaredEuclideanDistance();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Int64Parameter(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Int64Parameter.prototype.defaultValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Int64Parameter.prototype.range = null;
                Int64Parameter.prototype.set = null;
    
                var $oneOfFields;
    
                Object.defineProperty(Int64Parameter.prototype, "AllowedValues", {
                    get: $util.oneOfGetter($oneOfFields = ["range", "set"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Int64Parameter.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Int64Parameter();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function DoubleParameter(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DoubleParameter.prototype.defaultValue = 0;
                DoubleParameter.prototype.range = null;
    
                var $oneOfFields;
    
                Object.defineProperty(DoubleParameter.prototype, "AllowedValues", {
                    get: $util.oneOfGetter($oneOfFields = ["range"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                DoubleParameter.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DoubleParameter();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function StringParameter(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                StringParameter.prototype.defaultValue = "";
    
                StringParameter.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.StringParameter();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function BoolParameter(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BoolParameter.prototype.defaultValue = false;
    
                BoolParameter.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BoolParameter();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Identity(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Identity.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Identity();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Imputer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Imputer.prototype.imputedDoubleValue = 0;
                Imputer.prototype.imputedInt64Value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Imputer.prototype.imputedStringValue = "";
                Imputer.prototype.imputedDoubleArray = null;
                Imputer.prototype.imputedInt64Array = null;
                Imputer.prototype.imputedStringDictionary = null;
                Imputer.prototype.imputedInt64Dictionary = null;
                Imputer.prototype.replaceDoubleValue = 0;
                Imputer.prototype.replaceInt64Value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Imputer.prototype.replaceStringValue = "";
    
                var $oneOfFields;
    
                Object.defineProperty(Imputer.prototype, "ImputedValue", {
                    get: $util.oneOfGetter($oneOfFields = ["imputedDoubleValue", "imputedInt64Value", "imputedStringValue", "imputedDoubleArray", "imputedInt64Array", "imputedStringDictionary", "imputedInt64Dictionary"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(Imputer.prototype, "ReplaceValue", {
                    get: $util.oneOfGetter($oneOfFields = ["replaceDoubleValue", "replaceInt64Value", "replaceStringValue"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Imputer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Imputer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "RANK5_ARRAY_MAPPING"] = 0;
                values[valuesById[1] = "EXACT_ARRAY_MAPPING"] = 1;
                return values;
            })();
    
            Specification.NeuralNetworkImageShapeMapping = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "RANK5_IMAGE_MAPPING"] = 0;
                values[valuesById[1] = "RANK4_IMAGE_MAPPING"] = 1;
                return values;
            })();
    
            Specification.NeuralNetwork = (function() {
    
                function NeuralNetwork(properties) {
                    this.layers = [];
                    this.preprocessing = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetwork.prototype.layers = $util.emptyArray;
                NeuralNetwork.prototype.preprocessing = $util.emptyArray;
                NeuralNetwork.prototype.arrayInputShapeMapping = 0;
                NeuralNetwork.prototype.imageInputShapeMapping = 0;
                NeuralNetwork.prototype.updateParams = null;
    
                NeuralNetwork.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetwork();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.layers && message.layers.length))
                                message.layers = [];
                            message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.preprocessing && message.preprocessing.length))
                                message.preprocessing = [];
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
    
                function NeuralNetworkImageScaler(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetworkImageScaler.prototype.channelScale = 0;
                NeuralNetworkImageScaler.prototype.blueBias = 0;
                NeuralNetworkImageScaler.prototype.greenBias = 0;
                NeuralNetworkImageScaler.prototype.redBias = 0;
                NeuralNetworkImageScaler.prototype.grayBias = 0;
    
                NeuralNetworkImageScaler.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetworkImageScaler();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function NeuralNetworkMeanImage(properties) {
                    this.meanImage = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetworkMeanImage.prototype.meanImage = $util.emptyArray;
    
                NeuralNetworkMeanImage.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetworkMeanImage();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.meanImage && message.meanImage.length))
                                message.meanImage = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.meanImage.push(reader.float());
                            } else
                                message.meanImage.push(reader.float());
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
    
                function NeuralNetworkPreprocessing(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetworkPreprocessing.prototype.featureName = "";
                NeuralNetworkPreprocessing.prototype.scaler = null;
                NeuralNetworkPreprocessing.prototype.meanImage = null;
    
                var $oneOfFields;
    
                Object.defineProperty(NeuralNetworkPreprocessing.prototype, "preprocessor", {
                    get: $util.oneOfGetter($oneOfFields = ["scaler", "meanImage"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                NeuralNetworkPreprocessing.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetworkPreprocessing();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationReLU(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationReLU.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationReLU();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationLeakyReLU(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationLeakyReLU.prototype.alpha = 0;
    
                ActivationLeakyReLU.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationLeakyReLU();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationTanh(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationTanh.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationTanh();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationScaledTanh(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationScaledTanh.prototype.alpha = 0;
                ActivationScaledTanh.prototype.beta = 0;
    
                ActivationScaledTanh.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationScaledTanh();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationSigmoid(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationSigmoid.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationSigmoid();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationLinear(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationLinear.prototype.alpha = 0;
                ActivationLinear.prototype.beta = 0;
    
                ActivationLinear.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationLinear();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationSigmoidHard(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationSigmoidHard.prototype.alpha = 0;
                ActivationSigmoidHard.prototype.beta = 0;
    
                ActivationSigmoidHard.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationSigmoidHard();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationPReLU(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationPReLU.prototype.alpha = null;
    
                ActivationPReLU.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationPReLU();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationELU(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationELU.prototype.alpha = 0;
    
                ActivationELU.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationELU();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationThresholdedReLU(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationThresholdedReLU.prototype.alpha = 0;
    
                ActivationThresholdedReLU.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationThresholdedReLU();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationSoftsign(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationSoftsign.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationSoftsign();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationSoftplus(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationSoftplus.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationSoftplus();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationParametricSoftplus(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ActivationParametricSoftplus.prototype.alpha = null;
                ActivationParametricSoftplus.prototype.beta = null;
    
                ActivationParametricSoftplus.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationParametricSoftplus();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ActivationParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
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
    
                var $oneOfFields;
    
                Object.defineProperty(ActivationParams.prototype, "NonlinearityType", {
                    get: $util.oneOfGetter($oneOfFields = ["linear", "ReLU", "leakyReLU", "thresholdedReLU", "PReLU", "tanh", "scaledTanh", "sigmoid", "sigmoidHard", "ELU", "softsign", "softplus", "parametricSoftplus"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                ActivationParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ActivationParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Tensor(properties) {
                    this.dimValue = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Tensor.prototype.rank = 0;
                Tensor.prototype.dimValue = $util.emptyArray;
    
                Tensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Tensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.rank = reader.uint32();
                            break;
                        case 2:
                            if (!(message.dimValue && message.dimValue.length))
                                message.dimValue = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.dimValue.push(reader.int64());
                            } else
                                message.dimValue.push(reader.int64());
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
    
                function NeuralNetworkLayer(properties) {
                    this.input = [];
                    this.output = [];
                    this.inputTensor = [];
                    this.outputTensor = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetworkLayer.prototype.name = "";
                NeuralNetworkLayer.prototype.input = $util.emptyArray;
                NeuralNetworkLayer.prototype.output = $util.emptyArray;
                NeuralNetworkLayer.prototype.inputTensor = $util.emptyArray;
                NeuralNetworkLayer.prototype.outputTensor = $util.emptyArray;
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
    
                var $oneOfFields;
    
                Object.defineProperty(NeuralNetworkLayer.prototype, "layer", {
                    get: $util.oneOfGetter($oneOfFields = ["convolution", "pooling", "activation", "innerProduct", "embedding", "batchnorm", "mvn", "l2normalize", "softmax", "lrn", "crop", "padding", "upsample", "resizeBilinear", "cropResize", "unary", "add", "multiply", "average", "scale", "bias", "max", "min", "dot", "reduce", "loadConstant", "reshape", "flatten", "permute", "concat", "split", "sequenceRepeat", "reorganizeData", "slice", "simpleRecurrent", "gru", "uniDirectionalLSTM", "biDirectionalLSTM", "custom", "copy", "branch", "loop", "loopBreak", "loopContinue", "rangeStatic", "rangeDynamic", "clip", "ceil", "floor", "sign", "round", "exp2", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "gelu", "equal", "notEqual", "lessThan", "lessEqual", "greaterThan", "greaterEqual", "logicalOr", "logicalXor", "logicalNot", "logicalAnd", "modBroadcastable", "minBroadcastable", "maxBroadcastable", "addBroadcastable", "powBroadcastable", "divideBroadcastable", "floorDivBroadcastable", "multiplyBroadcastable", "subtractBroadcastable", "tile", "stack", "gather", "scatter", "gatherND", "scatterND", "softmaxND", "gatherAlongAxis", "scatterAlongAxis", "reverse", "reverseSeq", "splitND", "concatND", "transpose", "sliceStatic", "sliceDynamic", "slidingWindows", "topK", "argMin", "argMax", "embeddingND", "batchedMatmul", "getShape", "loadConstantND", "fillLike", "fillStatic", "fillDynamic", "broadcastToLike", "broadcastToStatic", "broadcastToDynamic", "squeeze", "expandDims", "flattenTo2D", "reshapeLike", "reshapeStatic", "reshapeDynamic", "rankPreservingReshape", "constantPad", "randomNormalLike", "randomNormalStatic", "randomNormalDynamic", "randomUniformLike", "randomUniformStatic", "randomUniformDynamic", "randomBernoulliLike", "randomBernoulliStatic", "randomBernoulliDynamic", "categoricalDistribution", "reduceL1", "reduceL2", "reduceMax", "reduceMin", "reduceSum", "reduceProd", "reduceMean", "reduceLogSum", "reduceSumSquare", "reduceLogSumExp", "whereNonZero", "matrixBandPart", "lowerTriangular", "upperTriangular", "whereBroadcastable", "layerNormalization", "NonMaximumSuppression"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                NeuralNetworkLayer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetworkLayer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.name = reader.string();
                            break;
                        case 2:
                            if (!(message.input && message.input.length))
                                message.input = [];
                            message.input.push(reader.string());
                            break;
                        case 3:
                            if (!(message.output && message.output.length))
                                message.output = [];
                            message.output.push(reader.string());
                            break;
                        case 4:
                            if (!(message.inputTensor && message.inputTensor.length))
                                message.inputTensor = [];
                            message.inputTensor.push($root.CoreML.Specification.Tensor.decode(reader, reader.uint32()));
                            break;
                        case 5:
                            if (!(message.outputTensor && message.outputTensor.length))
                                message.outputTensor = [];
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
    
                function BranchLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BranchLayerParams.prototype.ifBranch = null;
                BranchLayerParams.prototype.elseBranch = null;
    
                BranchLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BranchLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LoopLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LoopLayerParams.prototype.maxLoopIterations = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                LoopLayerParams.prototype.conditionVar = "";
                LoopLayerParams.prototype.conditionNetwork = null;
                LoopLayerParams.prototype.bodyNetwork = null;
    
                LoopLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LoopLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LoopBreakLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LoopBreakLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LoopBreakLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LoopContinueLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LoopContinueLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LoopContinueLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function CopyLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CopyLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CopyLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GreaterThanLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GreaterThanLayerParams.prototype.alpha = 0;
    
                GreaterThanLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GreaterThanLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GreaterEqualLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GreaterEqualLayerParams.prototype.alpha = 0;
    
                GreaterEqualLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GreaterEqualLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LessThanLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LessThanLayerParams.prototype.alpha = 0;
    
                LessThanLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LessThanLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LessEqualLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LessEqualLayerParams.prototype.alpha = 0;
    
                LessEqualLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LessEqualLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function EqualLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                EqualLayerParams.prototype.alpha = 0;
    
                EqualLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.EqualLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function NotEqualLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NotEqualLayerParams.prototype.alpha = 0;
    
                NotEqualLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NotEqualLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LogicalAndLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LogicalAndLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LogicalAndLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LogicalOrLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LogicalOrLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LogicalOrLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LogicalXorLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LogicalXorLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LogicalXorLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LogicalNotLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LogicalNotLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LogicalNotLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function BorderAmounts(properties) {
                    this.borderAmounts = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BorderAmounts.prototype.borderAmounts = $util.emptyArray;
    
                BorderAmounts.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BorderAmounts();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 10:
                            if (!(message.borderAmounts && message.borderAmounts.length))
                                message.borderAmounts = [];
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
    
                    function EdgeSizes(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    EdgeSizes.prototype.startEdgeSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    EdgeSizes.prototype.endEdgeSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                    EdgeSizes.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BorderAmounts.EdgeSizes();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function ValidPadding(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ValidPadding.prototype.paddingAmounts = null;
    
                ValidPadding.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ValidPadding();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SamePadding(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SamePadding.prototype.asymmetryMode = 0;
    
                SamePadding.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SamePadding();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "BOTTOM_RIGHT_HEAVY"] = 0;
                    values[valuesById[1] = "TOP_LEFT_HEAVY"] = 1;
                    return values;
                })();
    
                return SamePadding;
            })();
    
            Specification.SamplingMode = (function() {
    
                function SamplingMode(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SamplingMode.prototype.samplingMethod = 0;
    
                SamplingMode.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SamplingMode();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "STRICT_ALIGN_ENDPOINTS_MODE"] = 0;
                    values[valuesById[1] = "ALIGN_ENDPOINTS_MODE"] = 1;
                    values[valuesById[2] = "UPSAMPLE_MODE"] = 2;
                    values[valuesById[3] = "ROI_ALIGN_MODE"] = 3;
                    return values;
                })();
    
                return SamplingMode;
            })();
    
            Specification.BoxCoordinatesMode = (function() {
    
                function BoxCoordinatesMode(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BoxCoordinatesMode.prototype.boxMode = 0;
    
                BoxCoordinatesMode.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BoxCoordinatesMode();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "CORNERS_HEIGHT_FIRST"] = 0;
                    values[valuesById[1] = "CORNERS_WIDTH_FIRST"] = 1;
                    values[valuesById[2] = "CENTER_SIZE_HEIGHT_FIRST"] = 2;
                    values[valuesById[3] = "CENTER_SIZE_WIDTH_FIRST"] = 3;
                    return values;
                })();
    
                return BoxCoordinatesMode;
            })();
    
            Specification.WeightParams = (function() {
    
                function WeightParams(properties) {
                    this.floatValue = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                WeightParams.prototype.floatValue = $util.emptyArray;
                WeightParams.prototype.float16Value = $util.newBuffer([]);
                WeightParams.prototype.rawValue = $util.newBuffer([]);
                WeightParams.prototype.quantization = null;
                WeightParams.prototype.isUpdatable = false;
    
                WeightParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.WeightParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.floatValue && message.floatValue.length))
                                message.floatValue = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                if (message.floatValue.length == 0 && (end2 - reader.pos) > 1048576) {
                                    var floatValueLength = end2 - reader.pos;
                                    var floatValueView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, floatValueLength);
                                    floatValueLength = floatValueLength >>> 2;
                                    var floatValue = new Float32Array(floatValueLength);
                                    for (var i = 0; i < floatValueLength; i++) {
                                        floatValue[i] = floatValueView.getFloat32(i << 2, true);
                                    }
                                    message.floatValue = floatValue;
                                    reader.pos = end2;
                                }
                                else {
                                    while (reader.pos < end2)
                                        message.floatValue.push(reader.float());
                                }
                            } else
                                message.floatValue.push(reader.float());
                            break;
                        case 2:
                            message.float16Value = reader.bytes();
                            break;
                        case 30:
                            message.rawValue = reader.bytes();
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
    
                function QuantizationParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                QuantizationParams.prototype.numberOfBits = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                QuantizationParams.prototype.linearQuantization = null;
                QuantizationParams.prototype.lookupTableQuantization = null;
    
                var $oneOfFields;
    
                Object.defineProperty(QuantizationParams.prototype, "QuantizationType", {
                    get: $util.oneOfGetter($oneOfFields = ["linearQuantization", "lookupTableQuantization"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                QuantizationParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.QuantizationParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LinearQuantizationParams(properties) {
                    this.scale = [];
                    this.bias = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LinearQuantizationParams.prototype.scale = $util.emptyArray;
                LinearQuantizationParams.prototype.bias = $util.emptyArray;
    
                LinearQuantizationParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LinearQuantizationParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.scale && message.scale.length))
                                message.scale = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.scale.push(reader.float());
                            } else
                                message.scale.push(reader.float());
                            break;
                        case 2:
                            if (!(message.bias && message.bias.length))
                                message.bias = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.bias.push(reader.float());
                            } else
                                message.bias.push(reader.float());
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
    
                function LookUpTableQuantizationParams(properties) {
                    this.floatValue = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LookUpTableQuantizationParams.prototype.floatValue = $util.emptyArray;
    
                LookUpTableQuantizationParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LookUpTableQuantizationParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.floatValue && message.floatValue.length))
                                message.floatValue = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                if (message.floatValue.length == 0 && (end2 - reader.pos) > 1048576) {
                                    var floatValueLength = end2 - reader.pos;
                                    var floatValueView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, floatValueLength);
                                    floatValueLength = floatValueLength >>> 2;
                                    var floatValue = new Float32Array(floatValueLength);
                                    for (var i = 0; i < floatValueLength; i++) {
                                        floatValue[i] = floatValueView.getFloat32(i << 2, true);
                                    }
                                    message.floatValue = floatValue;
                                    reader.pos = end2;
                                }
                                else {
                                    while (reader.pos < end2)
                                        message.floatValue.push(reader.float());
                                }
                            } else
                                message.floatValue.push(reader.float());
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
    
                function ConvolutionLayerParams(properties) {
                    this.kernelSize = [];
                    this.stride = [];
                    this.dilationFactor = [];
                    this.outputShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ConvolutionLayerParams.prototype.outputChannels = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                ConvolutionLayerParams.prototype.kernelChannels = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                ConvolutionLayerParams.prototype.nGroups = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                ConvolutionLayerParams.prototype.kernelSize = $util.emptyArray;
                ConvolutionLayerParams.prototype.stride = $util.emptyArray;
                ConvolutionLayerParams.prototype.dilationFactor = $util.emptyArray;
                ConvolutionLayerParams.prototype.valid = null;
                ConvolutionLayerParams.prototype.same = null;
                ConvolutionLayerParams.prototype.isDeconvolution = false;
                ConvolutionLayerParams.prototype.hasBias = false;
                ConvolutionLayerParams.prototype.weights = null;
                ConvolutionLayerParams.prototype.bias = null;
                ConvolutionLayerParams.prototype.outputShape = $util.emptyArray;
    
                var $oneOfFields;
    
                Object.defineProperty(ConvolutionLayerParams.prototype, "ConvolutionPaddingType", {
                    get: $util.oneOfGetter($oneOfFields = ["valid", "same"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                ConvolutionLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ConvolutionLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                            if (!(message.kernelSize && message.kernelSize.length))
                                message.kernelSize = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.kernelSize.push(reader.uint64());
                            } else
                                message.kernelSize.push(reader.uint64());
                            break;
                        case 30:
                            if (!(message.stride && message.stride.length))
                                message.stride = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.stride.push(reader.uint64());
                            } else
                                message.stride.push(reader.uint64());
                            break;
                        case 40:
                            if (!(message.dilationFactor && message.dilationFactor.length))
                                message.dilationFactor = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.dilationFactor.push(reader.uint64());
                            } else
                                message.dilationFactor.push(reader.uint64());
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
                            if (!(message.outputShape && message.outputShape.length))
                                message.outputShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.outputShape.push(reader.uint64());
                            } else
                                message.outputShape.push(reader.uint64());
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
    
            Specification.InnerProductLayerParams = (function() {
    
                function InnerProductLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                InnerProductLayerParams.prototype.inputChannels = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                InnerProductLayerParams.prototype.outputChannels = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                InnerProductLayerParams.prototype.hasBias = false;
                InnerProductLayerParams.prototype.weights = null;
                InnerProductLayerParams.prototype.bias = null;
    
                InnerProductLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.InnerProductLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function EmbeddingLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                EmbeddingLayerParams.prototype.inputDim = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                EmbeddingLayerParams.prototype.outputChannels = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                EmbeddingLayerParams.prototype.hasBias = false;
                EmbeddingLayerParams.prototype.weights = null;
                EmbeddingLayerParams.prototype.bias = null;
    
                EmbeddingLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.EmbeddingLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function EmbeddingNDLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                EmbeddingNDLayerParams.prototype.vocabSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                EmbeddingNDLayerParams.prototype.embeddingSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                EmbeddingNDLayerParams.prototype.hasBias = false;
                EmbeddingNDLayerParams.prototype.weights = null;
                EmbeddingNDLayerParams.prototype.bias = null;
    
                EmbeddingNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.EmbeddingNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function BatchnormLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BatchnormLayerParams.prototype.channels = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                BatchnormLayerParams.prototype.computeMeanVar = false;
                BatchnormLayerParams.prototype.instanceNormalization = false;
                BatchnormLayerParams.prototype.epsilon = 0;
                BatchnormLayerParams.prototype.gamma = null;
                BatchnormLayerParams.prototype.beta = null;
                BatchnormLayerParams.prototype.mean = null;
                BatchnormLayerParams.prototype.variance = null;
    
                BatchnormLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BatchnormLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function PoolingLayerParams(properties) {
                    this.kernelSize = [];
                    this.stride = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PoolingLayerParams.prototype.type = 0;
                PoolingLayerParams.prototype.kernelSize = $util.emptyArray;
                PoolingLayerParams.prototype.stride = $util.emptyArray;
                PoolingLayerParams.prototype.valid = null;
                PoolingLayerParams.prototype.same = null;
                PoolingLayerParams.prototype.includeLastPixel = null;
                PoolingLayerParams.prototype.avgPoolExcludePadding = false;
                PoolingLayerParams.prototype.globalPooling = false;
    
                var $oneOfFields;
    
                Object.defineProperty(PoolingLayerParams.prototype, "PoolingPaddingType", {
                    get: $util.oneOfGetter($oneOfFields = ["valid", "same", "includeLastPixel"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                PoolingLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PoolingLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.type = reader.int32();
                            break;
                        case 10:
                            if (!(message.kernelSize && message.kernelSize.length))
                                message.kernelSize = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.kernelSize.push(reader.uint64());
                            } else
                                message.kernelSize.push(reader.uint64());
                            break;
                        case 20:
                            if (!(message.stride && message.stride.length))
                                message.stride = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.stride.push(reader.uint64());
                            } else
                                message.stride.push(reader.uint64());
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "MAX"] = 0;
                    values[valuesById[1] = "AVERAGE"] = 1;
                    values[valuesById[2] = "L2"] = 2;
                    return values;
                })();
    
                PoolingLayerParams.ValidCompletePadding = (function() {
    
                    function ValidCompletePadding(properties) {
                        this.paddingAmounts = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ValidCompletePadding.prototype.paddingAmounts = $util.emptyArray;
    
                    ValidCompletePadding.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PoolingLayerParams.ValidCompletePadding();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 10:
                                if (!(message.paddingAmounts && message.paddingAmounts.length))
                                    message.paddingAmounts = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    while (reader.pos < end2)
                                        message.paddingAmounts.push(reader.uint64());
                                } else
                                    message.paddingAmounts.push(reader.uint64());
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
    
            Specification.PaddingLayerParams = (function() {
    
                function PaddingLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PaddingLayerParams.prototype.constant = null;
                PaddingLayerParams.prototype.reflection = null;
                PaddingLayerParams.prototype.replication = null;
                PaddingLayerParams.prototype.paddingAmounts = null;
    
                var $oneOfFields;
    
                Object.defineProperty(PaddingLayerParams.prototype, "PaddingType", {
                    get: $util.oneOfGetter($oneOfFields = ["constant", "reflection", "replication"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                PaddingLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PaddingLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                    function PaddingConstant(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    PaddingConstant.prototype.value = 0;
    
                    PaddingConstant.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PaddingLayerParams.PaddingConstant();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function PaddingReflection(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    PaddingReflection.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReflection();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function PaddingReplication(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    PaddingReplication.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PaddingLayerParams.PaddingReplication();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function ConcatLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ConcatLayerParams.prototype.sequenceConcat = false;
    
                ConcatLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ConcatLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LRNLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LRNLayerParams.prototype.alpha = 0;
                LRNLayerParams.prototype.beta = 0;
                LRNLayerParams.prototype.localSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                LRNLayerParams.prototype.k = 0;
    
                LRNLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LRNLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SoftmaxLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SoftmaxLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SoftmaxLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SplitLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SplitLayerParams.prototype.nOutputs = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                SplitLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SplitLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AddLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AddLayerParams.prototype.alpha = 0;
    
                AddLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AddLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MultiplyLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MultiplyLayerParams.prototype.alpha = 0;
    
                MultiplyLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MultiplyLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function UnaryFunctionLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                UnaryFunctionLayerParams.prototype.type = 0;
                UnaryFunctionLayerParams.prototype.alpha = 0;
                UnaryFunctionLayerParams.prototype.epsilon = 0;
                UnaryFunctionLayerParams.prototype.shift = 0;
                UnaryFunctionLayerParams.prototype.scale = 0;
    
                UnaryFunctionLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.UnaryFunctionLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "SQRT"] = 0;
                    values[valuesById[1] = "RSQRT"] = 1;
                    values[valuesById[2] = "INVERSE"] = 2;
                    values[valuesById[3] = "POWER"] = 3;
                    values[valuesById[4] = "EXP"] = 4;
                    values[valuesById[5] = "LOG"] = 5;
                    values[valuesById[6] = "ABS"] = 6;
                    values[valuesById[7] = "THRESHOLD"] = 7;
                    return values;
                })();
    
                return UnaryFunctionLayerParams;
            })();
    
            Specification.UpsampleLayerParams = (function() {
    
                function UpsampleLayerParams(properties) {
                    this.scalingFactor = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                UpsampleLayerParams.prototype.scalingFactor = $util.emptyArray;
                UpsampleLayerParams.prototype.mode = 0;
    
                UpsampleLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.UpsampleLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.scalingFactor && message.scalingFactor.length))
                                message.scalingFactor = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.scalingFactor.push(reader.uint64());
                            } else
                                message.scalingFactor.push(reader.uint64());
                            break;
                        case 5:
                            message.mode = reader.int32();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                UpsampleLayerParams.InterpolationMode = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "NN"] = 0;
                    values[valuesById[1] = "BILINEAR"] = 1;
                    return values;
                })();
    
                return UpsampleLayerParams;
            })();
    
            Specification.ResizeBilinearLayerParams = (function() {
    
                function ResizeBilinearLayerParams(properties) {
                    this.targetSize = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ResizeBilinearLayerParams.prototype.targetSize = $util.emptyArray;
                ResizeBilinearLayerParams.prototype.mode = null;
    
                ResizeBilinearLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ResizeBilinearLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.targetSize && message.targetSize.length))
                                message.targetSize = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetSize.push(reader.uint64());
                            } else
                                message.targetSize.push(reader.uint64());
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
    
                function CropResizeLayerParams(properties) {
                    this.targetSize = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CropResizeLayerParams.prototype.targetSize = $util.emptyArray;
                CropResizeLayerParams.prototype.normalizedCoordinates = false;
                CropResizeLayerParams.prototype.mode = null;
                CropResizeLayerParams.prototype.boxIndicesMode = null;
                CropResizeLayerParams.prototype.spatialScale = 0;
    
                CropResizeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CropResizeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.targetSize && message.targetSize.length))
                                message.targetSize = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetSize.push(reader.uint64());
                            } else
                                message.targetSize.push(reader.uint64());
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
    
                function BiasLayerParams(properties) {
                    this.shape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BiasLayerParams.prototype.shape = $util.emptyArray;
                BiasLayerParams.prototype.bias = null;
    
                BiasLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BiasLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shape && message.shape.length))
                                message.shape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shape.push(reader.uint64());
                            } else
                                message.shape.push(reader.uint64());
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
    
                function ScaleLayerParams(properties) {
                    this.shapeScale = [];
                    this.shapeBias = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ScaleLayerParams.prototype.shapeScale = $util.emptyArray;
                ScaleLayerParams.prototype.scale = null;
                ScaleLayerParams.prototype.hasBias = false;
                ScaleLayerParams.prototype.shapeBias = $util.emptyArray;
                ScaleLayerParams.prototype.bias = null;
    
                ScaleLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ScaleLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shapeScale && message.shapeScale.length))
                                message.shapeScale = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shapeScale.push(reader.uint64());
                            } else
                                message.shapeScale.push(reader.uint64());
                            break;
                        case 2:
                            message.scale = $root.CoreML.Specification.WeightParams.decode(reader, reader.uint32());
                            break;
                        case 3:
                            message.hasBias = reader.bool();
                            break;
                        case 4:
                            if (!(message.shapeBias && message.shapeBias.length))
                                message.shapeBias = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shapeBias.push(reader.uint64());
                            } else
                                message.shapeBias.push(reader.uint64());
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
    
                function LoadConstantLayerParams(properties) {
                    this.shape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LoadConstantLayerParams.prototype.shape = $util.emptyArray;
                LoadConstantLayerParams.prototype.data = null;
    
                LoadConstantLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LoadConstantLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shape && message.shape.length))
                                message.shape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shape.push(reader.uint64());
                            } else
                                message.shape.push(reader.uint64());
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
    
                function L2NormalizeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                L2NormalizeLayerParams.prototype.epsilon = 0;
    
                L2NormalizeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.L2NormalizeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FlattenLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FlattenLayerParams.prototype.mode = 0;
    
                FlattenLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FlattenLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "CHANNEL_FIRST"] = 0;
                    values[valuesById[1] = "CHANNEL_LAST"] = 1;
                    return values;
                })();
    
                return FlattenLayerParams;
            })();
    
            Specification.ReshapeLayerParams = (function() {
    
                function ReshapeLayerParams(properties) {
                    this.targetShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReshapeLayerParams.prototype.targetShape = $util.emptyArray;
                ReshapeLayerParams.prototype.mode = 0;
    
                ReshapeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReshapeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.targetShape && message.targetShape.length))
                                message.targetShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetShape.push(reader.int64());
                            } else
                                message.targetShape.push(reader.int64());
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "CHANNEL_FIRST"] = 0;
                    values[valuesById[1] = "CHANNEL_LAST"] = 1;
                    return values;
                })();
    
                return ReshapeLayerParams;
            })();
    
            Specification.PermuteLayerParams = (function() {
    
                function PermuteLayerParams(properties) {
                    this.axis = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PermuteLayerParams.prototype.axis = $util.emptyArray;
    
                PermuteLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PermuteLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axis && message.axis.length))
                                message.axis = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axis.push(reader.uint64());
                            } else
                                message.axis.push(reader.uint64());
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
    
                function ReorganizeDataLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReorganizeDataLayerParams.prototype.mode = 0;
                ReorganizeDataLayerParams.prototype.blockSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                ReorganizeDataLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReorganizeDataLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "SPACE_TO_DEPTH"] = 0;
                    values[valuesById[1] = "DEPTH_TO_SPACE"] = 1;
                    return values;
                })();
    
                return ReorganizeDataLayerParams;
            })();
    
            Specification.SliceLayerParams = (function() {
    
                function SliceLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SliceLayerParams.prototype.startIndex = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                SliceLayerParams.prototype.endIndex = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                SliceLayerParams.prototype.stride = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                SliceLayerParams.prototype.axis = 0;
    
                SliceLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SliceLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "CHANNEL_AXIS"] = 0;
                    values[valuesById[1] = "HEIGHT_AXIS"] = 1;
                    values[valuesById[2] = "WIDTH_AXIS"] = 2;
                    return values;
                })();
    
                return SliceLayerParams;
            })();
    
            Specification.ReduceLayerParams = (function() {
    
                function ReduceLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceLayerParams.prototype.mode = 0;
                ReduceLayerParams.prototype.epsilon = 0;
                ReduceLayerParams.prototype.axis = 0;
    
                ReduceLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "SUM"] = 0;
                    values[valuesById[1] = "AVG"] = 1;
                    values[valuesById[2] = "PROD"] = 2;
                    values[valuesById[3] = "LOGSUM"] = 3;
                    values[valuesById[4] = "SUMSQUARE"] = 4;
                    values[valuesById[5] = "L1"] = 5;
                    values[valuesById[6] = "L2"] = 6;
                    values[valuesById[7] = "MAX"] = 7;
                    values[valuesById[8] = "MIN"] = 8;
                    values[valuesById[9] = "ARGMAX"] = 9;
                    return values;
                })();
    
                ReduceLayerParams.ReduceAxis = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "CHW"] = 0;
                    values[valuesById[1] = "HW"] = 1;
                    values[valuesById[2] = "C"] = 2;
                    values[valuesById[3] = "H"] = 3;
                    values[valuesById[4] = "W"] = 4;
                    return values;
                })();
    
                return ReduceLayerParams;
            })();
    
            Specification.CropLayerParams = (function() {
    
                function CropLayerParams(properties) {
                    this.offset = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CropLayerParams.prototype.cropAmounts = null;
                CropLayerParams.prototype.offset = $util.emptyArray;
    
                CropLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CropLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.cropAmounts = $root.CoreML.Specification.BorderAmounts.decode(reader, reader.uint32());
                            break;
                        case 5:
                            if (!(message.offset && message.offset.length))
                                message.offset = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.offset.push(reader.uint64());
                            } else
                                message.offset.push(reader.uint64());
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
    
                function AverageLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AverageLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AverageLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MaxLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MaxLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MaxLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MinLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MinLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MinLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function DotProductLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DotProductLayerParams.prototype.cosineSimilarity = false;
    
                DotProductLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DotProductLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MeanVarianceNormalizeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MeanVarianceNormalizeLayerParams.prototype.acrossChannels = false;
                MeanVarianceNormalizeLayerParams.prototype.normalizeVariance = false;
                MeanVarianceNormalizeLayerParams.prototype.epsilon = 0;
    
                MeanVarianceNormalizeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MeanVarianceNormalizeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SequenceRepeatLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SequenceRepeatLayerParams.prototype.nRepetitions = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                SequenceRepeatLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SequenceRepeatLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SimpleRecurrentLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SimpleRecurrentLayerParams.prototype.inputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                SimpleRecurrentLayerParams.prototype.outputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                SimpleRecurrentLayerParams.prototype.activation = null;
                SimpleRecurrentLayerParams.prototype.sequenceOutput = false;
                SimpleRecurrentLayerParams.prototype.hasBiasVector = false;
                SimpleRecurrentLayerParams.prototype.weightMatrix = null;
                SimpleRecurrentLayerParams.prototype.recursionMatrix = null;
                SimpleRecurrentLayerParams.prototype.biasVector = null;
                SimpleRecurrentLayerParams.prototype.reverseInput = false;
    
                SimpleRecurrentLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SimpleRecurrentLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GRULayerParams(properties) {
                    this.activations = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GRULayerParams.prototype.inputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                GRULayerParams.prototype.outputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                GRULayerParams.prototype.activations = $util.emptyArray;
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
    
                GRULayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GRULayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.inputVectorSize = reader.uint64();
                            break;
                        case 2:
                            message.outputVectorSize = reader.uint64();
                            break;
                        case 10:
                            if (!(message.activations && message.activations.length))
                                message.activations = [];
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
    
                function LSTMParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LSTMParams.prototype.sequenceOutput = false;
                LSTMParams.prototype.hasBiasVectors = false;
                LSTMParams.prototype.forgetBias = false;
                LSTMParams.prototype.hasPeepholeVectors = false;
                LSTMParams.prototype.coupledInputAndForgetGate = false;
                LSTMParams.prototype.cellClipThreshold = 0;
    
                LSTMParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LSTMParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LSTMWeightParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
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
    
                LSTMWeightParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LSTMWeightParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function UniDirectionalLSTMLayerParams(properties) {
                    this.activations = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                UniDirectionalLSTMLayerParams.prototype.inputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                UniDirectionalLSTMLayerParams.prototype.outputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                UniDirectionalLSTMLayerParams.prototype.activations = $util.emptyArray;
                UniDirectionalLSTMLayerParams.prototype.params = null;
                UniDirectionalLSTMLayerParams.prototype.weightParams = null;
                UniDirectionalLSTMLayerParams.prototype.reverseInput = false;
    
                UniDirectionalLSTMLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.UniDirectionalLSTMLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.inputVectorSize = reader.uint64();
                            break;
                        case 2:
                            message.outputVectorSize = reader.uint64();
                            break;
                        case 10:
                            if (!(message.activations && message.activations.length))
                                message.activations = [];
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
    
                function BiDirectionalLSTMLayerParams(properties) {
                    this.activationsForwardLSTM = [];
                    this.activationsBackwardLSTM = [];
                    this.weightParams = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BiDirectionalLSTMLayerParams.prototype.inputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                BiDirectionalLSTMLayerParams.prototype.outputVectorSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                BiDirectionalLSTMLayerParams.prototype.activationsForwardLSTM = $util.emptyArray;
                BiDirectionalLSTMLayerParams.prototype.activationsBackwardLSTM = $util.emptyArray;
                BiDirectionalLSTMLayerParams.prototype.params = null;
                BiDirectionalLSTMLayerParams.prototype.weightParams = $util.emptyArray;
    
                BiDirectionalLSTMLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BiDirectionalLSTMLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.inputVectorSize = reader.uint64();
                            break;
                        case 2:
                            message.outputVectorSize = reader.uint64();
                            break;
                        case 10:
                            if (!(message.activationsForwardLSTM && message.activationsForwardLSTM.length))
                                message.activationsForwardLSTM = [];
                            message.activationsForwardLSTM.push($root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                            break;
                        case 11:
                            if (!(message.activationsBackwardLSTM && message.activationsBackwardLSTM.length))
                                message.activationsBackwardLSTM = [];
                            message.activationsBackwardLSTM.push($root.CoreML.Specification.ActivationParams.decode(reader, reader.uint32()));
                            break;
                        case 15:
                            message.params = $root.CoreML.Specification.LSTMParams.decode(reader, reader.uint32());
                            break;
                        case 20:
                            if (!(message.weightParams && message.weightParams.length))
                                message.weightParams = [];
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
    
                function CustomLayerParams(properties) {
                    this.weights = [];
                    this.parameters = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CustomLayerParams.prototype.className = "";
                CustomLayerParams.prototype.weights = $util.emptyArray;
                CustomLayerParams.prototype.parameters = $util.emptyObject;
                CustomLayerParams.prototype.description = "";
    
                CustomLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CustomLayerParams(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 10:
                            message.className = reader.string();
                            break;
                        case 20:
                            if (!(message.weights && message.weights.length))
                                message.weights = [];
                            message.weights.push($root.CoreML.Specification.WeightParams.decode(reader, reader.uint32()));
                            break;
                        case 30:
                            reader.skip().pos++;
                            if (message.parameters === $util.emptyObject)
                                message.parameters = {};
                            key = reader.string();
                            reader.pos++;
                            message.parameters[key] = $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue.decode(reader, reader.uint32());
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
    
                    function CustomLayerParamValue(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    CustomLayerParamValue.prototype.doubleValue = 0;
                    CustomLayerParamValue.prototype.stringValue = "";
                    CustomLayerParamValue.prototype.intValue = 0;
                    CustomLayerParamValue.prototype.longValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                    CustomLayerParamValue.prototype.boolValue = false;
    
                    var $oneOfFields;
    
                    Object.defineProperty(CustomLayerParamValue.prototype, "value", {
                        get: $util.oneOfGetter($oneOfFields = ["doubleValue", "stringValue", "intValue", "longValue", "boolValue"]),
                        set: $util.oneOfSetter($oneOfFields)
                    });
    
                    CustomLayerParamValue.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CustomLayerParams.CustomLayerParamValue();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function TransposeLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TransposeLayerParams.prototype.axes = $util.emptyArray;
    
                TransposeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TransposeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.uint64());
                            } else
                                message.axes.push(reader.uint64());
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
    
                function BatchedMatMulLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BatchedMatMulLayerParams.prototype.transposeA = false;
                BatchedMatMulLayerParams.prototype.transposeB = false;
                BatchedMatMulLayerParams.prototype.weightMatrixFirstDimension = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                BatchedMatMulLayerParams.prototype.weightMatrixSecondDimension = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                BatchedMatMulLayerParams.prototype.hasBias = false;
                BatchedMatMulLayerParams.prototype.weights = null;
                BatchedMatMulLayerParams.prototype.bias = null;
    
                BatchedMatMulLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BatchedMatMulLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ConcatNDLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ConcatNDLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                ConcatNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ConcatNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SoftmaxNDLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SoftmaxNDLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                SoftmaxNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SoftmaxNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ReverseLayerParams(properties) {
                    this.reverseDim = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReverseLayerParams.prototype.reverseDim = $util.emptyArray;
    
                ReverseLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReverseLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.reverseDim && message.reverseDim.length))
                                message.reverseDim = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.reverseDim.push(reader.bool());
                            } else
                                message.reverseDim.push(reader.bool());
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
    
                function ReverseSeqLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReverseSeqLayerParams.prototype.batchAxis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ReverseSeqLayerParams.prototype.sequenceAxis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                ReverseSeqLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReverseSeqLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LoadConstantNDLayerParams(properties) {
                    this.shape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LoadConstantNDLayerParams.prototype.shape = $util.emptyArray;
                LoadConstantNDLayerParams.prototype.data = null;
    
                LoadConstantNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LoadConstantNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shape && message.shape.length))
                                message.shape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shape.push(reader.uint64());
                            } else
                                message.shape.push(reader.uint64());
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
    
                function FillLikeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FillLikeLayerParams.prototype.value = 0;
    
                FillLikeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FillLikeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FillStaticLayerParams(properties) {
                    this.targetShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FillStaticLayerParams.prototype.value = 0;
                FillStaticLayerParams.prototype.targetShape = $util.emptyArray;
    
                FillStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FillStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.value = reader.float();
                            break;
                        case 2:
                            if (!(message.targetShape && message.targetShape.length))
                                message.targetShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetShape.push(reader.uint64());
                            } else
                                message.targetShape.push(reader.uint64());
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
    
                function FillDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FillDynamicLayerParams.prototype.value = 0;
    
                FillDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FillDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function WhereBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                WhereBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.WhereBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SinLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SinLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SinLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function CosLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CosLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CosLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function TanLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TanLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TanLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AsinLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AsinLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AsinLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AcosLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AcosLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AcosLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AtanLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AtanLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AtanLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SinhLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SinhLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SinhLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function CoshLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CoshLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CoshLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function TanhLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TanhLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TanhLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AsinhLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AsinhLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AsinhLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AcoshLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AcoshLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AcoshLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AtanhLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AtanhLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AtanhLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function PowBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PowBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PowBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Exp2LayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Exp2LayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Exp2LayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function WhereNonZeroLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                WhereNonZeroLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.WhereNonZeroLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MatrixBandPartLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MatrixBandPartLayerParams.prototype.numLower = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                MatrixBandPartLayerParams.prototype.numUpper = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                MatrixBandPartLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MatrixBandPartLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function UpperTriangularLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                UpperTriangularLayerParams.prototype.k = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                UpperTriangularLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.UpperTriangularLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LowerTriangularLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LowerTriangularLayerParams.prototype.k = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                LowerTriangularLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LowerTriangularLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function BroadcastToLikeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BroadcastToLikeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BroadcastToLikeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function BroadcastToStaticLayerParams(properties) {
                    this.targetShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BroadcastToStaticLayerParams.prototype.targetShape = $util.emptyArray;
    
                BroadcastToStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BroadcastToStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.targetShape && message.targetShape.length))
                                message.targetShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetShape.push(reader.uint64());
                            } else
                                message.targetShape.push(reader.uint64());
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
    
                function BroadcastToDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                BroadcastToDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.BroadcastToDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AddBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AddBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AddBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MaxBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MaxBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MaxBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MinBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MinBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MinBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ModBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ModBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ModBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FloorDivBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FloorDivBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FloorDivBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SubtractBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SubtractBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SubtractBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MultiplyBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MultiplyBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MultiplyBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function DivideBroadcastableLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DivideBroadcastableLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DivideBroadcastableLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GatherLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GatherLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                GatherLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GatherLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "SCATTER_UPDATE"] = 0;
                values[valuesById[1] = "SCATTER_ADD"] = 1;
                values[valuesById[2] = "SCATTER_SUB"] = 2;
                values[valuesById[3] = "SCATTER_MUL"] = 3;
                values[valuesById[4] = "SCATTER_DIV"] = 4;
                values[valuesById[5] = "SCATTER_MAX"] = 5;
                values[valuesById[6] = "SCATTER_MIN"] = 6;
                return values;
            })();
    
            Specification.ScatterLayerParams = (function() {
    
                function ScatterLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ScatterLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ScatterLayerParams.prototype.mode = 0;
    
                ScatterLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ScatterLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GatherNDLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GatherNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GatherNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ScatterNDLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ScatterNDLayerParams.prototype.mode = 0;
    
                ScatterNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ScatterNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GatherAlongAxisLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GatherAlongAxisLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                GatherAlongAxisLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GatherAlongAxisLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ScatterAlongAxisLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ScatterAlongAxisLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ScatterAlongAxisLayerParams.prototype.mode = 0;
    
                ScatterAlongAxisLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ScatterAlongAxisLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function StackLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                StackLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                StackLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.StackLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RankPreservingReshapeLayerParams(properties) {
                    this.targetShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RankPreservingReshapeLayerParams.prototype.targetShape = $util.emptyArray;
    
                RankPreservingReshapeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RankPreservingReshapeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.targetShape && message.targetShape.length))
                                message.targetShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetShape.push(reader.int64());
                            } else
                                message.targetShape.push(reader.int64());
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
    
                function ConstantPaddingLayerParams(properties) {
                    this.padAmounts = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ConstantPaddingLayerParams.prototype.value = 0;
                ConstantPaddingLayerParams.prototype.padAmounts = $util.emptyArray;
                ConstantPaddingLayerParams.prototype.padToGivenOutputSizeMode = false;
    
                ConstantPaddingLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ConstantPaddingLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.value = reader.float();
                            break;
                        case 2:
                            if (!(message.padAmounts && message.padAmounts.length))
                                message.padAmounts = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.padAmounts.push(reader.uint64());
                            } else
                                message.padAmounts.push(reader.uint64());
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
    
                function RandomNormalLikeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomNormalLikeLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomNormalLikeLayerParams.prototype.mean = 0;
                RandomNormalLikeLayerParams.prototype.stdDev = 0;
    
                RandomNormalLikeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomNormalLikeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RandomNormalStaticLayerParams(properties) {
                    this.outputShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomNormalStaticLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomNormalStaticLayerParams.prototype.mean = 0;
                RandomNormalStaticLayerParams.prototype.stdDev = 0;
                RandomNormalStaticLayerParams.prototype.outputShape = $util.emptyArray;
    
                RandomNormalStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomNormalStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                            if (!(message.outputShape && message.outputShape.length))
                                message.outputShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.outputShape.push(reader.uint64());
                            } else
                                message.outputShape.push(reader.uint64());
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
    
                function RandomNormalDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomNormalDynamicLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomNormalDynamicLayerParams.prototype.mean = 0;
                RandomNormalDynamicLayerParams.prototype.stdDev = 0;
    
                RandomNormalDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomNormalDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RandomUniformLikeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomUniformLikeLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomUniformLikeLayerParams.prototype.minVal = 0;
                RandomUniformLikeLayerParams.prototype.maxVal = 0;
    
                RandomUniformLikeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomUniformLikeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RandomUniformStaticLayerParams(properties) {
                    this.outputShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomUniformStaticLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomUniformStaticLayerParams.prototype.minVal = 0;
                RandomUniformStaticLayerParams.prototype.maxVal = 0;
                RandomUniformStaticLayerParams.prototype.outputShape = $util.emptyArray;
    
                RandomUniformStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomUniformStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                            if (!(message.outputShape && message.outputShape.length))
                                message.outputShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.outputShape.push(reader.uint64());
                            } else
                                message.outputShape.push(reader.uint64());
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
    
                function RandomUniformDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomUniformDynamicLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomUniformDynamicLayerParams.prototype.minVal = 0;
                RandomUniformDynamicLayerParams.prototype.maxVal = 0;
    
                RandomUniformDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomUniformDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RandomBernoulliLikeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomBernoulliLikeLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomBernoulliLikeLayerParams.prototype.prob = 0;
    
                RandomBernoulliLikeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomBernoulliLikeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RandomBernoulliStaticLayerParams(properties) {
                    this.outputShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomBernoulliStaticLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomBernoulliStaticLayerParams.prototype.prob = 0;
                RandomBernoulliStaticLayerParams.prototype.outputShape = $util.emptyArray;
    
                RandomBernoulliStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomBernoulliStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.seed = reader.int64();
                            break;
                        case 2:
                            message.prob = reader.float();
                            break;
                        case 3:
                            if (!(message.outputShape && message.outputShape.length))
                                message.outputShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.outputShape.push(reader.uint64());
                            } else
                                message.outputShape.push(reader.uint64());
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
    
                function RandomBernoulliDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RandomBernoulliDynamicLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                RandomBernoulliDynamicLayerParams.prototype.prob = 0;
    
                RandomBernoulliDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RandomBernoulliDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function CategoricalDistributionLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CategoricalDistributionLayerParams.prototype.seed = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                CategoricalDistributionLayerParams.prototype.numSamples = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                CategoricalDistributionLayerParams.prototype.isLogits = false;
                CategoricalDistributionLayerParams.prototype.eps = 0;
                CategoricalDistributionLayerParams.prototype.temperature = 0;
    
                CategoricalDistributionLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CategoricalDistributionLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ReduceL1LayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceL1LayerParams.prototype.axes = $util.emptyArray;
                ReduceL1LayerParams.prototype.keepDims = false;
                ReduceL1LayerParams.prototype.reduceAll = false;
    
                ReduceL1LayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceL1LayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceL2LayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceL2LayerParams.prototype.axes = $util.emptyArray;
                ReduceL2LayerParams.prototype.keepDims = false;
                ReduceL2LayerParams.prototype.reduceAll = false;
    
                ReduceL2LayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceL2LayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceMaxLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceMaxLayerParams.prototype.axes = $util.emptyArray;
                ReduceMaxLayerParams.prototype.keepDims = false;
                ReduceMaxLayerParams.prototype.reduceAll = false;
    
                ReduceMaxLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceMaxLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceMinLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceMinLayerParams.prototype.axes = $util.emptyArray;
                ReduceMinLayerParams.prototype.keepDims = false;
                ReduceMinLayerParams.prototype.reduceAll = false;
    
                ReduceMinLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceMinLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceSumLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceSumLayerParams.prototype.axes = $util.emptyArray;
                ReduceSumLayerParams.prototype.keepDims = false;
                ReduceSumLayerParams.prototype.reduceAll = false;
    
                ReduceSumLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceSumLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceProdLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceProdLayerParams.prototype.axes = $util.emptyArray;
                ReduceProdLayerParams.prototype.keepDims = false;
                ReduceProdLayerParams.prototype.reduceAll = false;
    
                ReduceProdLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceProdLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceMeanLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceMeanLayerParams.prototype.axes = $util.emptyArray;
                ReduceMeanLayerParams.prototype.keepDims = false;
                ReduceMeanLayerParams.prototype.reduceAll = false;
    
                ReduceMeanLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceMeanLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceLogSumLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceLogSumLayerParams.prototype.axes = $util.emptyArray;
                ReduceLogSumLayerParams.prototype.keepDims = false;
                ReduceLogSumLayerParams.prototype.reduceAll = false;
    
                ReduceLogSumLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceLogSumLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceSumSquareLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceSumSquareLayerParams.prototype.axes = $util.emptyArray;
                ReduceSumSquareLayerParams.prototype.keepDims = false;
                ReduceSumSquareLayerParams.prototype.reduceAll = false;
    
                ReduceSumSquareLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceSumSquareLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ReduceLogSumExpLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReduceLogSumExpLayerParams.prototype.axes = $util.emptyArray;
                ReduceLogSumExpLayerParams.prototype.keepDims = false;
                ReduceLogSumExpLayerParams.prototype.reduceAll = false;
    
                ReduceLogSumExpLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReduceLogSumExpLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function ExpandDimsLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ExpandDimsLayerParams.prototype.axes = $util.emptyArray;
    
                ExpandDimsLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ExpandDimsLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function FlattenTo2DLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FlattenTo2DLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                FlattenTo2DLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FlattenTo2DLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ReshapeStaticLayerParams(properties) {
                    this.targetShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReshapeStaticLayerParams.prototype.targetShape = $util.emptyArray;
    
                ReshapeStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReshapeStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.targetShape && message.targetShape.length))
                                message.targetShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.targetShape.push(reader.int64());
                            } else
                                message.targetShape.push(reader.int64());
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
    
                function ReshapeLikeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReshapeLikeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReshapeLikeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ReshapeDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ReshapeDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ReshapeDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SqueezeLayerParams(properties) {
                    this.axes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SqueezeLayerParams.prototype.axes = $util.emptyArray;
                SqueezeLayerParams.prototype.squeezeAll = false;
    
                SqueezeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SqueezeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.axes && message.axes.length))
                                message.axes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.axes.push(reader.int64());
                            } else
                                message.axes.push(reader.int64());
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
    
                function TopKLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TopKLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                TopKLayerParams.prototype.K = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                TopKLayerParams.prototype.useBottomK = false;
    
                TopKLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TopKLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ArgMaxLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ArgMaxLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ArgMaxLayerParams.prototype.removeDim = false;
    
                ArgMaxLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArgMaxLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ArgMinLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ArgMinLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                ArgMinLayerParams.prototype.removeDim = false;
    
                ArgMinLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ArgMinLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SplitNDLayerParams(properties) {
                    this.splitSizes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SplitNDLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                SplitNDLayerParams.prototype.numSplits = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                SplitNDLayerParams.prototype.splitSizes = $util.emptyArray;
    
                SplitNDLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SplitNDLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.axis = reader.int64();
                            break;
                        case 2:
                            message.numSplits = reader.uint64();
                            break;
                        case 3:
                            if (!(message.splitSizes && message.splitSizes.length))
                                message.splitSizes = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.splitSizes.push(reader.uint64());
                            } else
                                message.splitSizes.push(reader.uint64());
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
    
                function CeilLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CeilLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CeilLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RoundLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RoundLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RoundLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function FloorLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                FloorLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.FloorLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SignLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SignLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SignLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ClipLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ClipLayerParams.prototype.minVal = 0;
                ClipLayerParams.prototype.maxVal = 0;
    
                ClipLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ClipLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SliceStaticLayerParams(properties) {
                    this.beginIds = [];
                    this.beginMasks = [];
                    this.endIds = [];
                    this.endMasks = [];
                    this.strides = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SliceStaticLayerParams.prototype.beginIds = $util.emptyArray;
                SliceStaticLayerParams.prototype.beginMasks = $util.emptyArray;
                SliceStaticLayerParams.prototype.endIds = $util.emptyArray;
                SliceStaticLayerParams.prototype.endMasks = $util.emptyArray;
                SliceStaticLayerParams.prototype.strides = $util.emptyArray;
    
                SliceStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SliceStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.beginIds && message.beginIds.length))
                                message.beginIds = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.beginIds.push(reader.int64());
                            } else
                                message.beginIds.push(reader.int64());
                            break;
                        case 2:
                            if (!(message.beginMasks && message.beginMasks.length))
                                message.beginMasks = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.beginMasks.push(reader.bool());
                            } else
                                message.beginMasks.push(reader.bool());
                            break;
                        case 3:
                            if (!(message.endIds && message.endIds.length))
                                message.endIds = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.endIds.push(reader.int64());
                            } else
                                message.endIds.push(reader.int64());
                            break;
                        case 4:
                            if (!(message.endMasks && message.endMasks.length))
                                message.endMasks = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.endMasks.push(reader.bool());
                            } else
                                message.endMasks.push(reader.bool());
                            break;
                        case 5:
                            if (!(message.strides && message.strides.length))
                                message.strides = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.strides.push(reader.int64());
                            } else
                                message.strides.push(reader.int64());
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
    
                function SliceDynamicLayerParams(properties) {
                    this.beginMasks = [];
                    this.endIds = [];
                    this.endMasks = [];
                    this.strides = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SliceDynamicLayerParams.prototype.beginMasks = $util.emptyArray;
                SliceDynamicLayerParams.prototype.endIds = $util.emptyArray;
                SliceDynamicLayerParams.prototype.endMasks = $util.emptyArray;
                SliceDynamicLayerParams.prototype.strides = $util.emptyArray;
    
                SliceDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SliceDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 2:
                            if (!(message.beginMasks && message.beginMasks.length))
                                message.beginMasks = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.beginMasks.push(reader.bool());
                            } else
                                message.beginMasks.push(reader.bool());
                            break;
                        case 3:
                            if (!(message.endIds && message.endIds.length))
                                message.endIds = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.endIds.push(reader.int64());
                            } else
                                message.endIds.push(reader.int64());
                            break;
                        case 4:
                            if (!(message.endMasks && message.endMasks.length))
                                message.endMasks = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.endMasks.push(reader.bool());
                            } else
                                message.endMasks.push(reader.bool());
                            break;
                        case 5:
                            if (!(message.strides && message.strides.length))
                                message.strides = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.strides.push(reader.int64());
                            } else
                                message.strides.push(reader.int64());
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
    
                function TileLayerParams(properties) {
                    this.reps = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TileLayerParams.prototype.reps = $util.emptyArray;
    
                TileLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TileLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.reps && message.reps.length))
                                message.reps = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.reps.push(reader.uint64());
                            } else
                                message.reps.push(reader.uint64());
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
    
                function GetShapeLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GetShapeLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GetShapeLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ErfLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ErfLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ErfLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function GeluLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                GeluLayerParams.prototype.mode = 0;
    
                GeluLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.GeluLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "EXACT"] = 0;
                    values[valuesById[1] = "TANH_APPROXIMATION"] = 1;
                    values[valuesById[2] = "SIGMOID_APPROXIMATION"] = 2;
                    return values;
                })();
    
                return GeluLayerParams;
            })();
    
            Specification.RangeStaticLayerParams = (function() {
    
                function RangeStaticLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RangeStaticLayerParams.prototype.endValue = 0;
                RangeStaticLayerParams.prototype.startValue = 0;
                RangeStaticLayerParams.prototype.stepSizeValue = 0;
    
                RangeStaticLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RangeStaticLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RangeDynamicLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RangeDynamicLayerParams.prototype.startValue = 0;
                RangeDynamicLayerParams.prototype.stepSizeValue = 0;
    
                RangeDynamicLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RangeDynamicLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SlidingWindowsLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SlidingWindowsLayerParams.prototype.axis = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                SlidingWindowsLayerParams.prototype.windowSize = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                SlidingWindowsLayerParams.prototype.step = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
    
                SlidingWindowsLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SlidingWindowsLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LayerNormalizationLayerParams(properties) {
                    this.normalizedShape = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LayerNormalizationLayerParams.prototype.normalizedShape = $util.emptyArray;
                LayerNormalizationLayerParams.prototype.eps = 0;
                LayerNormalizationLayerParams.prototype.gamma = null;
                LayerNormalizationLayerParams.prototype.beta = null;
    
                LayerNormalizationLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LayerNormalizationLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.normalizedShape && message.normalizedShape.length))
                                message.normalizedShape = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.normalizedShape.push(reader.int64());
                            } else
                                message.normalizedShape.push(reader.int64());
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
    
                function NonMaximumSuppressionLayerParams(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NonMaximumSuppressionLayerParams.prototype.iouThreshold = 0;
                NonMaximumSuppressionLayerParams.prototype.scoreThreshold = 0;
                NonMaximumSuppressionLayerParams.prototype.maxBoxes = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                NonMaximumSuppressionLayerParams.prototype.perClassSuppression = false;
    
                NonMaximumSuppressionLayerParams.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NonMaximumSuppressionLayerParams();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
            Specification.NeuralNetworkClassifier = (function() {
    
                function NeuralNetworkClassifier(properties) {
                    this.layers = [];
                    this.preprocessing = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetworkClassifier.prototype.layers = $util.emptyArray;
                NeuralNetworkClassifier.prototype.preprocessing = $util.emptyArray;
                NeuralNetworkClassifier.prototype.arrayInputShapeMapping = 0;
                NeuralNetworkClassifier.prototype.imageInputShapeMapping = 0;
                NeuralNetworkClassifier.prototype.updateParams = null;
                NeuralNetworkClassifier.prototype.stringClassLabels = null;
                NeuralNetworkClassifier.prototype.int64ClassLabels = null;
                NeuralNetworkClassifier.prototype.labelProbabilityLayerName = "";
    
                var $oneOfFields;
    
                Object.defineProperty(NeuralNetworkClassifier.prototype, "ClassLabels", {
                    get: $util.oneOfGetter($oneOfFields = ["stringClassLabels", "int64ClassLabels"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                NeuralNetworkClassifier.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetworkClassifier();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.layers && message.layers.length))
                                message.layers = [];
                            message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.preprocessing && message.preprocessing.length))
                                message.preprocessing = [];
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
    
            Specification.NeuralNetworkRegressor = (function() {
    
                function NeuralNetworkRegressor(properties) {
                    this.layers = [];
                    this.preprocessing = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NeuralNetworkRegressor.prototype.layers = $util.emptyArray;
                NeuralNetworkRegressor.prototype.preprocessing = $util.emptyArray;
                NeuralNetworkRegressor.prototype.arrayInputShapeMapping = 0;
                NeuralNetworkRegressor.prototype.imageInputShapeMapping = 0;
                NeuralNetworkRegressor.prototype.updateParams = null;
    
                NeuralNetworkRegressor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NeuralNetworkRegressor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.layers && message.layers.length))
                                message.layers = [];
                            message.layers.push($root.CoreML.Specification.NeuralNetworkLayer.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            if (!(message.preprocessing && message.preprocessing.length))
                                message.preprocessing = [];
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
    
                function NetworkUpdateParameters(properties) {
                    this.lossLayers = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NetworkUpdateParameters.prototype.lossLayers = $util.emptyArray;
                NetworkUpdateParameters.prototype.optimizer = null;
                NetworkUpdateParameters.prototype.epochs = null;
                NetworkUpdateParameters.prototype.shuffle = null;
                NetworkUpdateParameters.prototype.seed = null;
    
                NetworkUpdateParameters.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NetworkUpdateParameters();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.lossLayers && message.lossLayers.length))
                                message.lossLayers = [];
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
    
                function LossLayer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LossLayer.prototype.name = "";
                LossLayer.prototype.categoricalCrossEntropyLossLayer = null;
                LossLayer.prototype.meanSquaredErrorLossLayer = null;
    
                var $oneOfFields;
    
                Object.defineProperty(LossLayer.prototype, "LossLayerType", {
                    get: $util.oneOfGetter($oneOfFields = ["categoricalCrossEntropyLossLayer", "meanSquaredErrorLossLayer"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                LossLayer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LossLayer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function CategoricalCrossEntropyLossLayer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                CategoricalCrossEntropyLossLayer.prototype.input = "";
                CategoricalCrossEntropyLossLayer.prototype.target = "";
    
                CategoricalCrossEntropyLossLayer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.CategoricalCrossEntropyLossLayer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function MeanSquaredErrorLossLayer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                MeanSquaredErrorLossLayer.prototype.input = "";
                MeanSquaredErrorLossLayer.prototype.target = "";
    
                MeanSquaredErrorLossLayer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.MeanSquaredErrorLossLayer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Optimizer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Optimizer.prototype.sgdOptimizer = null;
                Optimizer.prototype.adamOptimizer = null;
    
                var $oneOfFields;
    
                Object.defineProperty(Optimizer.prototype, "OptimizerType", {
                    get: $util.oneOfGetter($oneOfFields = ["sgdOptimizer", "adamOptimizer"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Optimizer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Optimizer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SGDOptimizer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SGDOptimizer.prototype.learningRate = null;
                SGDOptimizer.prototype.miniBatchSize = null;
                SGDOptimizer.prototype.momentum = null;
    
                SGDOptimizer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SGDOptimizer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function AdamOptimizer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                AdamOptimizer.prototype.learningRate = null;
                AdamOptimizer.prototype.miniBatchSize = null;
                AdamOptimizer.prototype.beta1 = null;
                AdamOptimizer.prototype.beta2 = null;
                AdamOptimizer.prototype.eps = null;
    
                AdamOptimizer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.AdamOptimizer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Normalizer(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Normalizer.prototype.normType = 0;
    
                Normalizer.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Normalizer();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "LMax"] = 0;
                    values[valuesById[1] = "L1"] = 1;
                    values[valuesById[2] = "L2"] = 2;
                    return values;
                })();
    
                return Normalizer;
            })();
    
            Specification.OneHotEncoder = (function() {
    
                function OneHotEncoder(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                OneHotEncoder.prototype.stringCategories = null;
                OneHotEncoder.prototype.int64Categories = null;
                OneHotEncoder.prototype.outputSparse = false;
                OneHotEncoder.prototype.handleUnknown = 0;
    
                var $oneOfFields;
    
                Object.defineProperty(OneHotEncoder.prototype, "CategoryType", {
                    get: $util.oneOfGetter($oneOfFields = ["stringCategories", "int64Categories"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                OneHotEncoder.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.OneHotEncoder();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "ErrorOnUnknown"] = 0;
                    values[valuesById[1] = "IgnoreUnknown"] = 1;
                    return values;
                })();
    
                return OneHotEncoder;
            })();
    
            Specification.Scaler = (function() {
    
                function Scaler(properties) {
                    this.shiftValue = [];
                    this.scaleValue = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Scaler.prototype.shiftValue = $util.emptyArray;
                Scaler.prototype.scaleValue = $util.emptyArray;
    
                Scaler.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Scaler();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shiftValue && message.shiftValue.length))
                                message.shiftValue = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shiftValue.push(reader.double());
                            } else
                                message.shiftValue.push(reader.double());
                            break;
                        case 2:
                            if (!(message.scaleValue && message.scaleValue.length))
                                message.scaleValue = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.scaleValue.push(reader.double());
                            } else
                                message.scaleValue.push(reader.double());
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
    
                function NonMaximumSuppression(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
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
    
                var $oneOfFields;
    
                Object.defineProperty(NonMaximumSuppression.prototype, "SuppressionMethod", {
                    get: $util.oneOfGetter($oneOfFields = ["pickTop"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(NonMaximumSuppression.prototype, "ClassLabels", {
                    get: $util.oneOfGetter($oneOfFields = ["stringClassLabels", "int64ClassLabels"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                NonMaximumSuppression.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NonMaximumSuppression();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                    function PickTop(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    PickTop.prototype.perClass = false;
    
                    PickTop.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.NonMaximumSuppression.PickTop();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                function LinearKernel(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LinearKernel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LinearKernel();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function RBFKernel(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                RBFKernel.prototype.gamma = 0;
    
                RBFKernel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.RBFKernel();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function PolyKernel(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                PolyKernel.prototype.degree = 0;
                PolyKernel.prototype.c = 0;
                PolyKernel.prototype.gamma = 0;
    
                PolyKernel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.PolyKernel();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SigmoidKernel(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SigmoidKernel.prototype.gamma = 0;
                SigmoidKernel.prototype.c = 0;
    
                SigmoidKernel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SigmoidKernel();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function Kernel(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Kernel.prototype.linearKernel = null;
                Kernel.prototype.rbfKernel = null;
                Kernel.prototype.polyKernel = null;
                Kernel.prototype.sigmoidKernel = null;
    
                var $oneOfFields;
    
                Object.defineProperty(Kernel.prototype, "kernel", {
                    get: $util.oneOfGetter($oneOfFields = ["linearKernel", "rbfKernel", "polyKernel", "sigmoidKernel"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Kernel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Kernel();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SparseNode(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SparseNode.prototype.index = 0;
                SparseNode.prototype.value = 0;
    
                SparseNode.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SparseNode();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SparseVector(properties) {
                    this.nodes = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SparseVector.prototype.nodes = $util.emptyArray;
    
                SparseVector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SparseVector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.nodes && message.nodes.length))
                                message.nodes = [];
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
    
                function SparseSupportVectors(properties) {
                    this.vectors = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SparseSupportVectors.prototype.vectors = $util.emptyArray;
    
                SparseSupportVectors.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SparseSupportVectors();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.vectors && message.vectors.length))
                                message.vectors = [];
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
    
                function DenseVector(properties) {
                    this.values = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DenseVector.prototype.values = $util.emptyArray;
    
                DenseVector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DenseVector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.values && message.values.length))
                                message.values = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.values.push(reader.double());
                            } else
                                message.values.push(reader.double());
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
    
                function DenseSupportVectors(properties) {
                    this.vectors = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DenseSupportVectors.prototype.vectors = $util.emptyArray;
    
                DenseSupportVectors.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.DenseSupportVectors();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.vectors && message.vectors.length))
                                message.vectors = [];
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
    
                function Coefficients(properties) {
                    this.alpha = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Coefficients.prototype.alpha = $util.emptyArray;
    
                Coefficients.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.Coefficients();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.alpha && message.alpha.length))
                                message.alpha = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.alpha.push(reader.double());
                            } else
                                message.alpha.push(reader.double());
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
    
                function SupportVectorRegressor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SupportVectorRegressor.prototype.kernel = null;
                SupportVectorRegressor.prototype.sparseSupportVectors = null;
                SupportVectorRegressor.prototype.denseSupportVectors = null;
                SupportVectorRegressor.prototype.coefficients = null;
                SupportVectorRegressor.prototype.rho = 0;
    
                var $oneOfFields;
    
                Object.defineProperty(SupportVectorRegressor.prototype, "supportVectors", {
                    get: $util.oneOfGetter($oneOfFields = ["sparseSupportVectors", "denseSupportVectors"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                SupportVectorRegressor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SupportVectorRegressor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function SupportVectorClassifier(properties) {
                    this.numberOfSupportVectorsPerClass = [];
                    this.coefficients = [];
                    this.rho = [];
                    this.probA = [];
                    this.probB = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SupportVectorClassifier.prototype.kernel = null;
                SupportVectorClassifier.prototype.numberOfSupportVectorsPerClass = $util.emptyArray;
                SupportVectorClassifier.prototype.sparseSupportVectors = null;
                SupportVectorClassifier.prototype.denseSupportVectors = null;
                SupportVectorClassifier.prototype.coefficients = $util.emptyArray;
                SupportVectorClassifier.prototype.rho = $util.emptyArray;
                SupportVectorClassifier.prototype.probA = $util.emptyArray;
                SupportVectorClassifier.prototype.probB = $util.emptyArray;
                SupportVectorClassifier.prototype.stringClassLabels = null;
                SupportVectorClassifier.prototype.int64ClassLabels = null;
    
                var $oneOfFields;
    
                Object.defineProperty(SupportVectorClassifier.prototype, "supportVectors", {
                    get: $util.oneOfGetter($oneOfFields = ["sparseSupportVectors", "denseSupportVectors"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Object.defineProperty(SupportVectorClassifier.prototype, "ClassLabels", {
                    get: $util.oneOfGetter($oneOfFields = ["stringClassLabels", "int64ClassLabels"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                SupportVectorClassifier.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.SupportVectorClassifier();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.kernel = $root.CoreML.Specification.Kernel.decode(reader, reader.uint32());
                            break;
                        case 2:
                            if (!(message.numberOfSupportVectorsPerClass && message.numberOfSupportVectorsPerClass.length))
                                message.numberOfSupportVectorsPerClass = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.numberOfSupportVectorsPerClass.push(reader.int32());
                            } else
                                message.numberOfSupportVectorsPerClass.push(reader.int32());
                            break;
                        case 3:
                            message.sparseSupportVectors = $root.CoreML.Specification.SparseSupportVectors.decode(reader, reader.uint32());
                            break;
                        case 4:
                            message.denseSupportVectors = $root.CoreML.Specification.DenseSupportVectors.decode(reader, reader.uint32());
                            break;
                        case 5:
                            if (!(message.coefficients && message.coefficients.length))
                                message.coefficients = [];
                            message.coefficients.push($root.CoreML.Specification.Coefficients.decode(reader, reader.uint32()));
                            break;
                        case 6:
                            if (!(message.rho && message.rho.length))
                                message.rho = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.rho.push(reader.double());
                            } else
                                message.rho.push(reader.double());
                            break;
                        case 7:
                            if (!(message.probA && message.probA.length))
                                message.probA = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.probA.push(reader.double());
                            } else
                                message.probA.push(reader.double());
                            break;
                        case 8:
                            if (!(message.probB && message.probB.length))
                                message.probB = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.probB.push(reader.double());
                            } else
                                message.probB.push(reader.double());
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "NoTransform"] = 0;
                values[valuesById[1] = "Classification_SoftMax"] = 1;
                values[valuesById[2] = "Regression_Logistic"] = 2;
                values[valuesById[3] = "Classification_SoftMaxWithZeroClassReference"] = 3;
                return values;
            })();
    
            Specification.TreeEnsembleParameters = (function() {
    
                function TreeEnsembleParameters(properties) {
                    this.nodes = [];
                    this.basePredictionValue = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TreeEnsembleParameters.prototype.nodes = $util.emptyArray;
                TreeEnsembleParameters.prototype.numPredictionDimensions = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                TreeEnsembleParameters.prototype.basePredictionValue = $util.emptyArray;
    
                TreeEnsembleParameters.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TreeEnsembleParameters();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.nodes && message.nodes.length))
                                message.nodes = [];
                            message.nodes.push($root.CoreML.Specification.TreeEnsembleParameters.TreeNode.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            message.numPredictionDimensions = reader.uint64();
                            break;
                        case 3:
                            if (!(message.basePredictionValue && message.basePredictionValue.length))
                                message.basePredictionValue = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.basePredictionValue.push(reader.double());
                            } else
                                message.basePredictionValue.push(reader.double());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                TreeEnsembleParameters.TreeNode = (function() {
    
                    function TreeNode(properties) {
                        this.evaluationInfo = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    TreeNode.prototype.treeId = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    TreeNode.prototype.nodeId = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    TreeNode.prototype.nodeBehavior = 0;
                    TreeNode.prototype.branchFeatureIndex = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    TreeNode.prototype.branchFeatureValue = 0;
                    TreeNode.prototype.trueChildNodeId = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    TreeNode.prototype.falseChildNodeId = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    TreeNode.prototype.missingValueTracksTrueChild = false;
                    TreeNode.prototype.evaluationInfo = $util.emptyArray;
                    TreeNode.prototype.relativeHitRate = 0;
    
                    TreeNode.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
                                if (!(message.evaluationInfo && message.evaluationInfo.length))
                                    message.evaluationInfo = [];
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
                        var valuesById = {}, values = Object.create(valuesById);
                        values[valuesById[0] = "BranchOnValueLessThanEqual"] = 0;
                        values[valuesById[1] = "BranchOnValueLessThan"] = 1;
                        values[valuesById[2] = "BranchOnValueGreaterThanEqual"] = 2;
                        values[valuesById[3] = "BranchOnValueGreaterThan"] = 3;
                        values[valuesById[4] = "BranchOnValueEqual"] = 4;
                        values[valuesById[5] = "BranchOnValueNotEqual"] = 5;
                        values[valuesById[6] = "LeafNode"] = 6;
                        return values;
                    })();
    
                    TreeNode.EvaluationInfo = (function() {
    
                        function EvaluationInfo(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        EvaluationInfo.prototype.evaluationIndex = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                        EvaluationInfo.prototype.evaluationValue = 0;
    
                        EvaluationInfo.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TreeEnsembleParameters.TreeNode.EvaluationInfo();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
    
                function TreeEnsembleClassifier(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TreeEnsembleClassifier.prototype.treeEnsemble = null;
                TreeEnsembleClassifier.prototype.postEvaluationTransform = 0;
                TreeEnsembleClassifier.prototype.stringClassLabels = null;
                TreeEnsembleClassifier.prototype.int64ClassLabels = null;
    
                var $oneOfFields;
    
                Object.defineProperty(TreeEnsembleClassifier.prototype, "ClassLabels", {
                    get: $util.oneOfGetter($oneOfFields = ["stringClassLabels", "int64ClassLabels"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                TreeEnsembleClassifier.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TreeEnsembleClassifier();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function TreeEnsembleRegressor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                TreeEnsembleRegressor.prototype.treeEnsemble = null;
                TreeEnsembleRegressor.prototype.postEvaluationTransform = 0;
    
                TreeEnsembleRegressor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.TreeEnsembleRegressor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function ItemSimilarityRecommender(properties) {
                    this.itemItemSimilarities = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                ItemSimilarityRecommender.prototype.itemItemSimilarities = $util.emptyArray;
                ItemSimilarityRecommender.prototype.itemStringIds = null;
                ItemSimilarityRecommender.prototype.itemInt64Ids = null;
                ItemSimilarityRecommender.prototype.itemInputFeatureName = "";
                ItemSimilarityRecommender.prototype.numRecommendationsInputFeatureName = "";
                ItemSimilarityRecommender.prototype.itemRestrictionInputFeatureName = "";
                ItemSimilarityRecommender.prototype.itemExclusionInputFeatureName = "";
                ItemSimilarityRecommender.prototype.recommendedItemListOutputFeatureName = "";
                ItemSimilarityRecommender.prototype.recommendedItemScoreOutputFeatureName = "";
    
                ItemSimilarityRecommender.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ItemSimilarityRecommender();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.itemItemSimilarities && message.itemItemSimilarities.length))
                                message.itemItemSimilarities = [];
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
    
                    function ConnectedItem(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ConnectedItem.prototype.itemId = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    ConnectedItem.prototype.similarityScore = 0;
    
                    ConnectedItem.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ItemSimilarityRecommender.ConnectedItem();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
    
                    function SimilarItems(properties) {
                        this.similarItemList = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    SimilarItems.prototype.itemId = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                    SimilarItems.prototype.similarItemList = $util.emptyArray;
                    SimilarItems.prototype.itemScoreAdjustment = 0;
    
                    SimilarItems.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.ItemSimilarityRecommender.SimilarItems();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                message.itemId = reader.uint64();
                                break;
                            case 2:
                                if (!(message.similarItemList && message.similarItemList.length))
                                    message.similarItemList = [];
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
    
                function LinkedModel(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LinkedModel.prototype.linkedModelFile = null;
    
                var $oneOfFields;
    
                Object.defineProperty(LinkedModel.prototype, "LinkType", {
                    get: $util.oneOfGetter($oneOfFields = ["linkedModelFile"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                LinkedModel.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LinkedModel();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
    
                function LinkedModelFile(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                LinkedModelFile.prototype.linkedModelFileName = null;
                LinkedModelFile.prototype.linkedModelSearchPath = null;
    
                LinkedModelFile.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CoreML.Specification.LinkedModelFile();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
