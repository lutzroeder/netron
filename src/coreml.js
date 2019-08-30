/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var coreml = coreml || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var marked = marked || require('marked');

coreml.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        return extension == 'mlmodel';
    }

    open(context, host) { 
        return host.require('./coreml-proto').then(() => {
            var identifier = context.identifier;
            var decodedBuffer = null;
            try {
                coreml.proto = protobuf.roots.coreml.CoreML.Specification;
                decodedBuffer = coreml.proto.Model.decode(context.buffer);
            }
            catch (error) {
                throw new coreml.Error("File format is not coreml.Model (" + error.message + ") in '" + identifier + "'.");
            }
            return coreml.Metadata.open(host).then((metadata) => {
                try {
                    return new coreml.Model(metadata, decodedBuffer);
                }
                catch (error) {
                    host.exception(error, false);
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new coreml.Error(message + " in '" + identifier + "'.");
                }
            });
        });
    }
};

coreml.Model = class {

    constructor(metadata, model) {
        this._specificationVersion = model.specificationVersion;
        this._graphs = [ new coreml.Graph(metadata, model) ];
        if (model.description && model.description.metadata) {
            var properties = model.description.metadata;
            if (properties.versionString) {
                this._version = properties.versionString;
            }
            if (properties.author) {
                this._author = properties.author;
            }
            if (properties.shortDescription) {
                this._description = properties.shortDescription;
            }
            if (properties.license) {
                this._license = properties.license;
            }
            if (metadata.userDefined && Object.keys(properties.userDefined).length > 0) {
                /* empty */
            }
        }
    }

    get format() {
        return 'Core ML v' + this._specificationVersion.toString();
    }

    get version() {
        return this._version || null;
    }

    get description() {
        return this._description || null;
    }

    get author() {
        return this._author || null;
    }

    get license() {
        return this._license || null;
    }

    get graphs() {
        return this._graphs;
    }
};

coreml.Graph = class {

    constructor(metadata, model)
    {
        this._metadata = metadata;
        this._description = model.description;
        this._groups = false; 
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        if (this._description) {
            this._inputs = this._description.input.map((input) => {
                var argument = new coreml.Argument(input.name, coreml.Graph._formatFeatureType(input.type), input.shortDescription, null);
                return new coreml.Parameter(input.name, true, [ argument ]);
            });

            this._outputs = this._description.output.map((output) => {
                var argument = new coreml.Argument(output.name, coreml.Graph._formatFeatureType(output.type), output.shortDescription, null);
                return new coreml.Parameter(output.name, true, [ argument ]);
            });
        }

        this._type = this._loadModel(model, {}, '');
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    get groups() {
        return this._groups;
    }

    _updateOutput(name, newName) {
        for (var node of this._nodes) {
            node._outputs = node._outputs.map((output) => (output != name) ? output : newName);
        }
        return newName;
    }

    _updateClassifierOutput(group, classifier) {
        var labelProbabilityLayerName = classifier.labelProbabilityLayerName;
        if (!labelProbabilityLayerName && this._nodes.length > 0) {
            labelProbabilityLayerName = this._nodes.slice(-1).pop()._outputs[0];
        }
        var predictedFeatureName = this._description.predictedFeatureName;
        var predictedProbabilitiesName = this._description.predictedProbabilitiesName;
        if ((predictedFeatureName || predictedProbabilitiesName) && labelProbabilityLayerName && classifier.ClassLabels) {
            predictedFeatureName = predictedFeatureName ? predictedFeatureName : '?';
            predictedProbabilitiesName = predictedProbabilitiesName ? predictedProbabilitiesName : '?';
            var labelProbabilityInput = this._updateOutput(labelProbabilityLayerName, labelProbabilityLayerName + ':labelProbabilityLayerName');
            var operator = classifier.ClassLabels;
            this._nodes.push(new coreml.Node(this._metadata, this._group, operator, null, classifier[operator], [ labelProbabilityInput ], [ predictedProbabilitiesName, predictedFeatureName ]));
        }
    }

    _updatePreprocessing(scope, group, preprocessing) {
        if (preprocessing && preprocessing.length > 0) {
            var preprocessingInput = this._description.input[0].name;
            var inputNodes = [];
            for (var node of this._nodes) {
                if (node._inputs.some((input => input == preprocessingInput))) {
                    inputNodes.push(node);
                }
            }
            var preprocessorOutput = preprocessingInput;
            var preprocessorIndex = 0;
            for (var p of preprocessing) {
                var input = p.featureName ? p.featureName : preprocessorOutput;
                preprocessorOutput = preprocessingInput + ':' + preprocessorIndex.toString();
                this._createNode(scope, group, p.preprocessor, null, p[p.preprocessor], [ input ], [ preprocessorOutput ]);
                preprocessorIndex++;
            }
            for (var inputNode of inputNodes) {
                inputNode._inputs = inputNode._inputs.map((input) => (input != preprocessingInput) ? input : preprocessorOutput);
            }
        }
    }

    _loadModel(model, scope, group) {
        var i;
        this._groups = this._groups | (group.length > 0 ? true : false);
        var layer;
        if (model.neuralNetworkClassifier) {
            var neuralNetworkClassifier = model.neuralNetworkClassifier;
            for (layer of neuralNetworkClassifier.layers) {
                this._createNode(scope, group, layer.layer, layer.name, layer[layer.layer], layer.input, layer.output);
            }
            this._updateClassifierOutput(group, neuralNetworkClassifier);
            this._updatePreprocessing(scope, group, neuralNetworkClassifier.preprocessing);
            return 'Neural Network Classifier';
        }
        else if (model.neuralNetwork) {
            var neuralNetwork = model.neuralNetwork;
            for (layer of neuralNetwork.layers) {
                this._createNode(scope, group, layer.layer, layer.name, layer[layer.layer], layer.input, layer.output);
            }
            this._updatePreprocessing(scope, group, neuralNetwork.preprocessing);
            return 'Neural Network';
        }
        else if (model.neuralNetworkRegressor) {
            var neuralNetworkRegressor = model.neuralNetworkRegressor;
            for (layer of neuralNetworkRegressor.layers) {
                this._createNode(scope, group, layer.layer, layer.name, layer[layer.layer], layer.input, layer.output);
            }
            this._updatePreprocessing(scope, group, neuralNetworkRegressor);
            return 'Neural Network Regressor';
        }
        else if (model.pipeline) {
            for (i = 0; i < model.pipeline.models.length; i++) {
                this._loadModel(model.pipeline.models[i], scope, (group ? (group + '/') : '') + 'pipeline[' + i.toString() + ']');
            }
            return 'Pipeline';
        }
        else if (model.pipelineClassifier) {
            for (i = 0; i < model.pipelineClassifier.pipeline.models.length; i++) {
                this._loadModel(model.pipelineClassifier.pipeline.models[i], scope, (group ? (group + '/') : '') + 'pipelineClassifier[' + i.toString() + ']');
            }
            return 'Pipeline Classifier';
        }
        else if (model.pipelineRegressor) {
            for (i = 0; i < model.pipelineRegressor.pipeline.models.length; i++) {
                this._loadModel(model.pipelineRegressor.pipeline.models[i], scope, (group ? (group + '/') : '') + 'pipelineRegressor[' + i.toString() + ']');
            }
            return 'Pipeline Regressor';
        }
        else if (model.glmClassifier) {
            this._createNode(scope, group, 'glmClassifier', null, 
                {
                    classEncoding: model.glmClassifier.classEncoding, 
                    offset: model.glmClassifier.offset, 
                    weights: model.glmClassifier.weights 
                }, 
                [ model.description.input[0].name ],
                [ model.description.predictedProbabilitiesName ]);
            this._updateClassifierOutput(group, model.glmClassifier);
            return 'Generalized Linear Classifier';
        }
        else if (model.glmRegressor) {
            this._createNode(scope, group, 'glmRegressor', null, 
                model.glmRegressor,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Generalized Linear Regressor';
        }
        else if (model.dictVectorizer) {
            this._createNode(scope, group, 'dictVectorizer', null, model.dictVectorizer,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Dictionary Vectorizer';
        }
        else if (model.featureVectorizer) {
            this._createNode(scope, group, 'featureVectorizer', null, model.featureVectorizer, 
                coreml.Graph._formatFeatureDescriptionList(model.description.input),
                [ model.description.output[0].name ]);
            return 'Feature Vectorizer';
        }
        else if (model.treeEnsembleClassifier) {
            this._createNode(scope, group, 'treeEnsembleClassifier', null, model.treeEnsembleClassifier.treeEnsemble, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            this._updateClassifierOutput(group, model.treeEnsembleClassifier);
            return 'Tree Ensemble Classifier';
        }
        else if (model.treeEnsembleRegressor) {
            this._createNode(scope, group, 'treeEnsembleRegressor', null, model.treeEnsembleRegressor.treeEnsemble, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Tree Ensemble Regressor';
        }
        else if (model.supportVectorClassifier) {
            this._createNode(scope, group, 'supportVectorClassifier', null, 
                {   
                    coefficients: model.supportVectorClassifier.coefficients, 
                    denseSupportVectors: model.supportVectorClassifier.denseSupportVectors,
                    kernel: model.supportVectorClassifier.kernel,
                    numberOfSupportVectorsPerClass: model.supportVectorClassifier.numberOfSupportVectorsPerClass,
                    probA: model.supportVectorClassifier.probA,
                    probB: model.supportVectorClassifier.probB,
                    rho: model.supportVectorClassifier.rho,
                    supportVectors: model.supportVectorClassifier.supportVectors 
                }, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            this._updateClassifierOutput(group, model.supportVectorClassifier);
            return 'Support Vector Classifier';
        }
        else if (model.supportVectorRegressor) {
            this._createNode(scope, group, 'supportVectorRegressor', null, 
                {
                    coefficients: model.supportVectorRegressor.coefficients, 
                    kernel: model.supportVectorRegressor.kernel,
                    rho: model.supportVectorRegressor.rho,
                    supportVectors: model.supportVectorRegressor.supportVectors 
                }, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Support Vector Regressor';
        }
        else if (model.arrayFeatureExtractor) {
            this._createNode(scope, group, 'arrayFeatureExtractor', null, 
                { extractIndex: model.arrayFeatureExtractor.extractIndex },
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Array Feature Extractor';
        }
        else if (model.oneHotEncoder) {
            var categoryType = model.oneHotEncoder.CategoryType;
            var oneHotEncoderParams = { outputSparse: model.oneHotEncoder.outputSparse };
            oneHotEncoderParams[categoryType] = model.oneHotEncoder[categoryType];
            this._createNode(scope, group, 'oneHotEncoder', null, 
                oneHotEncoderParams,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'One Hot Encoder';
        }
        else if (model.imputer) {
            var imputedValue = model.imputer.ImputedValue;
            var replaceValue = model.imputer.ReplaceValue;
            var imputerParams = {};
            imputerParams[imputedValue] = model.imputer[imputedValue];
            imputerParams[replaceValue] = model.imputer[replaceValue];
            this._createNode(scope, group, 'oneHotEncoder', null, 
                imputerParams,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Imputer';
            
        }
        else if (model.normalizer) {
            this._createNode(scope, group, 'normalizer', null, 
                model.normalizer,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Normalizer';
        }
        else if (model.wordTagger) {
            this._createNode(scope, group, 'wordTagger', null, 
                model.wordTagger,
                [ model.description.input[0].name ],
                [ 
                    model.wordTagger.tokensOutputFeatureName,
                    model.wordTagger.tokenTagsOutputFeatureName,
                    model.wordTagger.tokenLocationsOutputFeatureName,
                    model.wordTagger.tokenLengthsOutputFeatureName 
                ]);
            return 'Word Tagger';
        }
        else if (model.textClassifier) {
            this._createNode(scope, group, 'textClassifier', null, 
                model.textClassifier,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Text Classifier';
        }
        else if (model.nonMaximumSuppression) {
            var nonMaximumSuppressionParams = { 
                pickTop: model.nonMaximumSuppression.pickTop,
                stringClassLabels: model.nonMaximumSuppression.stringClassLabels,
                iouThreshold: model.nonMaximumSuppression.iouThreshold, 
                confidenceThreshold: model.nonMaximumSuppression.confidenceThreshold
            };
            this._createNode(scope, group, 'nonMaximumSuppression', null, 
                nonMaximumSuppressionParams,
                [
                    model.nonMaximumSuppression.confidenceInputFeatureName,
                    model.nonMaximumSuppression.coordinatesInputFeatureName,
                    model.nonMaximumSuppression.iouThresholdInputFeatureName,
                    model.nonMaximumSuppression.confidenceThresholdInputFeatureName,
                ],
                [ 
                    model.nonMaximumSuppression.confidenceOutputFeatureName,
                    model.nonMaximumSuppression.coordinatesOutputFeatureName
                ]);
            return 'Non Maximum Suppression';
        }
        else if (model.visionFeaturePrint) {
            var visionFeaturePrintParams = {
                scene: model.visionFeaturePrint.scene
            }
            this._createNode(scope, group, 'visionFeaturePrint', null,
                visionFeaturePrintParams,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Vision Feature Print';
        }
        else if (model.soundAnalysisPreprocessing) {
            this._createNode(scope, group, 'soundAnalysisPreprocessing', null,
                model.soundAnalysisPreprocessing,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'Sound Analysis Preprocessing';
        }
        else if (model.kNearestNeighborsClassifier) {
            this._createNode(scope, group, 'kNearestNeighborsClassifier', null,
                model.kNearestNeighborsClassifier,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            this._updateClassifierOutput(group, model.kNearestNeighborsClassifier);
            return 'kNearestNeighborsClassifier';
        }
        else if (model.customModel) {
            this._createNode(scope, group, 'customModel', null,
                { className: model.customModel.className, parameters: model.customModel.parameters },
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            return 'customModel';
        }
        throw new coreml.Error("Unknown model type '" + JSON.stringify(Object.keys(model)) + "'.");
    }

    _createNode(scope, group, operator, name, data, inputs, outputs) {
        inputs = inputs.map((input) => scope[input] ? scope[input].argument : input);
        outputs = outputs.map((output) => {
            if (scope[output]) {
                scope[output].counter++;
                var next = output + '\n' + scope[output].counter.toString(); // custom argument id
                scope[output].argument = next;
                return next;
            }
            scope[output] = {
                argument: output,
                counter: 0
            };
            return output;
        });

        var node = new coreml.Node(this._metadata, group, operator, name, data, inputs, outputs);
        this._nodes.push(node);
        return node;
    }

    static _formatFeatureType(type) {
        var result = '?';
        if (type) {
            switch (type.Type) {
                case 'multiArrayType':
                    var shape = new coreml.TensorShape([]);
                    if (type.multiArrayType.shape && type.multiArrayType.shape.length > 0) {
                        shape = new coreml.TensorShape(type.multiArrayType.shape);
                    }
                    var dataType = '?';
                    switch (type.multiArrayType.dataType) {
                        case coreml.proto.ArrayFeatureType.ArrayDataType.FLOAT32:
                            dataType = 'float32';
                            break;
                        case coreml.proto.ArrayFeatureType.ArrayDataType.INT32:
                            dataType = 'int32';
                            break;
                        case coreml.proto.ArrayFeatureType.ArrayDataType.DOUBLE:
                            dataType = 'float64';
                            break;
                    }
                    result = new coreml.TensorType(dataType, shape);
                    break;
                case 'stringType':
                    result = new coreml.TensorType('string');
                    break;
                case 'doubleType':
                    result = new coreml.TensorType('float64');
                    break;
                case 'int64Type':
                    result = new coreml.TensorType('int64');
                    break;
                case 'dictionaryType':
                    result = new coreml.MapType(type.dictionaryType.KeyType.replace('KeyType', ''), 'float64');
                    break;
                case 'imageType':
                    result = new coreml.ImageType(type.imageType.colorSpace, type.imageType.width, type.imageType.height);
                    break;
            }
            if (type.isOptional) {
                result = new coreml.OptionalType(result);
            }
        }
        return result;
    }

    static _formatFeatureDescriptionList(list) {
        return list.map((item) => item.name);
    }
};

coreml.Parameter = class {
    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

coreml.Argument = class {

    constructor(id, type, description, initializer) {
        this._id = id;
        this._type = type;
        this._description = description || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get description() {
        return this._description;
    }

    get quantization() {
        if (this._initializer) {
            return this._initializer.quantization;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

coreml.Node = class {

    constructor(metadata, group, operator, name, data, inputs, outputs) {
        this._metadata = metadata;
        if (group) {
            this._group = group;
        }
        this._operator = operator;
        this._name = name || '';
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        this._initializers = [];
        if (data) {
            var initializerMap = this._initialize(data);
            for (var key of Object.keys(data)) {
                if (!initializerMap[key]) {
                    this._attributes.push(new coreml.Attribute(this._metadata, this.operator, key, data[key]));
                }
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get category() {
        var schema = this._metadata.getSchema(this.operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        var schema = this._metadata.getSchema(this.operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this.operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (var attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (var input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get group() {
        return this._group ? this._group : null;
    }

    get inputs() {
        var inputs = this._metadata.getInputs(this._operator, this._inputs).map((input) => {
            return new coreml.Parameter(input.name, true, input.arguments.map((argument) => {
                return new coreml.Argument(argument.id, argument.type, null, null);
            }));
        });
        return inputs.concat(this._initializers);
    }

    get outputs() {
        return this._outputs.map((output, index) => {
            var name = this._metadata.getOutputName(this._operator, index);
            return new coreml.Parameter(name, true, [ new coreml.Argument(output, null, null, null) ]);
        });
    }

    get attributes() {
        return this._attributes;
    }

    _initialize(data) {
        switch (this._operator) {
            case 'convolution':
                var weightsShape = [ data.outputChannels, data.kernelChannels, data.kernelSize[0], data.kernelSize[1] ];
                if (data.isDeconvolution) {
                    weightsShape[0] = data.kernelChannels;
                    weightsShape[1] = Math.floor(data.outputChannels / (data.nGroups != 0 ? data.nGroups : 1));
                }    
                this._initializer('Weights', 'weights', weightsShape, data.weights);
                if (data.hasBias) {
                    this._initializer('Weights', 'bias', [ data.outputChannels ], data.bias);
                }

                return { 'weights': true, 'bias': data.hasBias };
            case 'innerProduct':
                this._initializer('Weights', 'weights', [ data.outputChannels, data.inputChannels ], data.weights);
                if (data.hasBias) {
                    this._initializer('Weights', 'bias', [ data.outputChannels ], data.bias);
                }
                return { 'weights': true, 'bias': data.hasBias };
            case 'batchnorm':
                this._initializer('Weights', 'gamma', [ data.channels ], data.gamma);
                this._initializer('Weights', 'beta', [ data.channels ], data.beta);
                if (data.mean) {
                    this._initializer('Weights', 'mean', [ data.channels ], data.mean);
                }
                if (data.variance) {
                    this._initializer('Weights', 'variance', [ data.channels ], data.variance);
                }
                return { 'gamma': true, 'beta': true, 'mean': true, 'variance': true };
            case 'embedding':
                this._initializer('Weights', 'weights', [ data.inputDim, data.outputChannels ], data.weights);
                return { 'weights': true };
            case 'loadConstant':    
                this._initializer('Weights', 'data', data.shape, data.data);
                return { 'data': true };
            case 'scale':
                this._initializer('Weights', 'scale', data.shapeScale, data.scale);
                if (data.hasBias) {
                    this._initializer('Weights', 'bias', data.shapeBias, data.bias);
                }
                return { 'scale': true, 'bias': data.hasBias };
            case 'bias':
                this._initializer('Weights', 'bias', data.shape, data.bias);
                return { 'bias': true };
            case 'simpleRecurrent':
                this._initializer('Weights', 'weights', [ data.outputVectorSize, data.inputVectorSize ], data.weightMatrix);
                this._initializer('Weights', 'recurrent', [ data.outputVectorSize, data.inputVectorSize ], data.recursionMatrix);
                if (data.hasBiasVectors) {
                    this._initializer('Weights', 'bias', [ data.outputVectorSize ], data.biasVector);
                }
                return { 'weightMatrix': true, 'recursionMatrix': true, 'biasVector': data.hasBiasVectors };
            case 'gru':
                var recursionMatrixShape = [ data.outputVectorSize, data.outputVectorSize ];
                var weightMatrixShape = [ data.outputVectorSize, data.inputVectorSize ];
                var biasVectorShape = [ data.outputVectorSize ];
                this._initializer('Weights', 'updateGateWeightMatrix', weightMatrixShape, data.updateGateWeightMatrix);
                this._initializer('Weights', 'resetGateWeightMatrix', weightMatrixShape, data.resetGateWeightMatrix);
                this._initializer('Weights', 'outputGateWeightMatrix', weightMatrixShape, data.outputGateWeightMatrix);
                this._initializer('Weights', 'updateGateRecursionMatrix', recursionMatrixShape, data.updateGateRecursionMatrix);
                this._initializer('Weights', 'resetGateRecursionMatrix', recursionMatrixShape, data.resetGateRecursionMatrix);
                this._initializer('Weights', 'outputGateRecursionMatrix', recursionMatrixShape, data.outputGateRecursionMatrix);
                if (data.hasBiasVectors) {
                    this._initializer('Weights', 'updateGateBiasVector', biasVectorShape, data.updateGateBiasVector);
                    this._initializer('Weights', 'resetGateBiasVector', biasVectorShape, data.resetGateBiasVector);
                    this._initializer('Weights', 'outputGateBiasVector', biasVectorShape, data.outputGateBiasVector);
                }  
                return {
                    'updateGateWeightMatrix': true, 'resetGateWeightMatrix': true, 'outputGateWeightMatrix': true, 
                    'updateGateRecursionMatrix': true, 'resetGateRecursionMatrix': true, 'outputGateRecursionMatrix': true,
                    'updateGateBiasVector': data.hasBiasVectors, 'resetGateBiasVector': data.hasBiasVectors, 'outputGateBiasVector': data.hasBiasVectors };
            case 'uniDirectionalLSTM':
            case 'biDirectionalLSTM':
                var count = (this._operator == 'uniDirectionalLSTM') ? 1 : 2;
                var matrixShape = [ data.outputVectorSize, data.inputVectorSize ];
                var vectorShape = [ data.outputVectorSize ];
                for (var i = 0; i < count; i++) {
                    var weights = count == 1 ? data.weightParams : data.weightParams[i];
                    var suffix = (i == 0) ? '' : '_rev';
                    this._initializer('Weights', 'inputGateWeightMatrix' + suffix, matrixShape, weights.inputGateWeightMatrix);
                    this._initializer('Weights', 'forgetGateWeightMatrix' + suffix, matrixShape, weights.forgetGateWeightMatrix);
                    this._initializer('Weights', 'blockInputWeightMatrix' + suffix, matrixShape, weights.blockInputWeightMatrix);
                    this._initializer('Weights', 'outputGateWeightMatrix' + suffix, matrixShape, weights.outputGateWeightMatrix);
                    this._initializer('Weights', 'inputGateRecursionMatrix' + suffix, matrixShape, weights.inputGateRecursionMatrix);
                    this._initializer('Weights', 'forgetGateRecursionMatrix' + suffix, matrixShape,weights.forgetGateRecursionMatrix);
                    this._initializer('Weights', 'blockInputRecursionMatrix' + suffix, matrixShape, weights.blockInputRecursionMatrix);
                    this._initializer('Weights', 'outputGateRecursionMatrix' + suffix, matrixShape, weights.outputGateRecursionMatrix);
                    if (data.params.hasBiasVectors) {
                        this._initializer('Weights', 'inputGateBiasVector' + suffix, vectorShape, weights.inputGateBiasVector);
                        this._initializer('Weights', 'forgetGateBiasVector' + suffix, vectorShape, weights.forgetGateBiasVector);
                        this._initializer('Weights', 'blockInputBiasVector' + suffix, vectorShape, weights.blockInputBiasVector);
                        this._initializer('Weights', 'outputGateBiasVector' + suffix, vectorShape, weights.outputGateBiasVector);
                    }
                    if (data.params.hasPeepholeVectors) {
                        this._initializer('Weights', 'inputGatePeepholeVector' + suffix, vectorShape, weights.inputGatePeepholeVector);
                        this._initializer('Weights', 'forgetGatePeepholeVector' + suffix, vectorShape, weights.forgetGatePeepholeVector);
                        this._initializer('Weights', 'outputGatePeepholeVector' + suffix, vectorShape, weights.outputGatePeepholeVector);
                    }
                }
                return { 'weightParams': true };
            case 'dictVectorizer':
                data.stringToIndex = this._convertVector(data.stringToIndex);
                return {};
            case 'wordTagger':
                data.modelParameterData = Array.from(data.modelParameterData);
                data.stringTags = this._convertVector(data.stringTags);
                return { tokensOutputFeatureName: true, tokenTagsOutputFeatureName: true, tokenLengthsOutputFeatureName: true, tokenLocationsOutputFeatureName: true };
            case 'textClassifier':
                data.modelParameterData = Array.from(data.modelParameterData);
                data.stringClassLabels = this._convertVector(data.stringClassLabels);
                return {};
            case 'nonMaximumSuppression':
                data.stringClassLabels = this._convertVector(data.stringClassLabels);
                return {};
        }
        return {};
    }

    _convertVector(value) {
        if (value && Object.keys(value).length == 1 && value.vector) {
            return value.vector;
        }
        return value;
    }

    _initializer(kind, name, shape, data) {
        var initializer = new coreml.Tensor(kind, name, shape, data);
        var argument = new coreml.Argument('', null, null, initializer);
        var visible = true;
        var schema = this._metadata.getInputSchema(this._operator, name);
        if (schema && Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
            visible = false;
        }
        this._initializers.push(new coreml.Parameter(name, visible, [ argument ]));
    }
};

coreml.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;
        var schema = metadata.getAttributeSchema(operator, this._name);
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
            }
            if (this._type && coreml.proto) {
                var type = coreml.proto;
                var parts = this._type.split('.');
                while (type && parts.length > 0) {
                    type = type[parts.shift()];
                }
                if (type && type[this._value]) {
                    this._value = type[this.value];
                }
            }

            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (Array.isArray(value)) {
                    value = value.map((item) => {
                        if (item && long.Long.isLong(item)) {
                            return item.toNumber();
                        }
                        return item;
                    });
                }
                if (JSON.stringify(schema.default) == JSON.stringify(value)) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

coreml.Tensor = class {

    constructor(kind, name, shape, data) {
        this._kind = kind;
        this._name = name;
        this._data = null;
        var dataType = '?';
        if (data) {
            if (data.floatValue && data.floatValue.length > 0) {
                this._data = data.floatValue;
                dataType = 'float32';
            }
            else if (data.float16Value && data.float16Value.length > 0) {
                this._data = data.float16Value; // byte[]
                dataType = 'float16';
            }
            else if (data.rawValue && data.rawValue.length > 0) {
                if (data.quantization) {
                    this._data = data.rawValue;
                    dataType = 'uint' + data.quantization.numberOfBits.toString();
                }
                else {
                    shape = [];
                }
            }
        }

        this._quantization = data.quantization || null;
        this._type = new coreml.TensorType(dataType, new coreml.TensorShape(shape));
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        if (this._quantization) {
            if (this._quantization.lookupTableQuantization && 
                this._quantization.lookupTableQuantization.floatValue &&
                this._quantization.lookupTableQuantization.floatValue.length > 0) {
                var map = [];
                for (var key of Object.keys(this._quantization.lookupTableQuantization.floatValue)) {
                    map.push(key.toString() + ' = ' + this._quantization.lookupTableQuantization.floatValue[key].toString());
                }
                return map.join('; ');
            }
            return '?'; 
        }
        return null;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
 
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (context.dataType) {
            case 'float32':
                context.data = this._data;
                break;
            case 'float16':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                if (this._quantization) {
                    context.dataType = 'quantization';
                    context.bits = long.Long.isLong(this._quantization.numberOfBits) ? this._quantization.numberOfBits.toNumber() : this._quantization.numberOfBits;
                    context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                }
                else {
                    context.state = 'Tensor data type is not implemented.';
                }
                break;
        }

        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float32':
                        results.push(this._data[context.index]);
                        context.index++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        break;
                    case 'quantization':
                        results.push(context.data.getBits(context.index, context.bits));
                        context.index++;
                        break;

                }
                context.count++;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
    }
};

coreml.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape || new coreml.TensorShape([]);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

coreml.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

coreml.MapType = class {

    constructor(keyType, valueType) {
        this._keyType = keyType;
        this._valueType = valueType;
    }

    get keyType() {
        return this._keyType;
    }

    get valueType() {
        return this._valueType;
    }

    toString() {
        return 'map<' + this._keyType + ',' + this._valueType.toString() + '>';
    }
};

coreml.ImageType = class {

    constructor(colorSpace, width, height) {
        this._colorSpace = '?';
        switch (colorSpace) {
            case coreml.proto.ImageFeatureType.ColorSpace.GRAYSCALE:
                this._colorSpace = 'Grayscale';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.RGB:
                this._colorSpace = 'RGB';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.BGR:
                this._colorSpace = 'BGR';
                break;
        }
        this._width = width;
        this._height = height;
    }

    toString() {
        return 'image<' + this._colorSpace + ',' + this._width. toString() + 'x' + this._height.toString() + '>';
    }
};

coreml.OptionalType = class {

    constructor(type) {
        this._type = type;
    }

    toString() {
        return this._type.toString() + '?';
    }
};

coreml.Metadata = class {

    static open(host) {
        if (coreml.Metadata._metadata) {
            return Promise.resolve(coreml.Metadata._metadata);
        }
        return host.request(null, 'coreml-metadata.json', 'utf-8').then((data) => {
            coreml.Metadata._metadata = new coreml.Metadata(data);
            return coreml.Metadata._metadata;
        }).catch(() => {
            coreml.Metadata._metadata = new coreml.Metadata(null);
            return coreml.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        this._inputCache = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }

    getInputSchema(operator, name) {
        var map = this._inputCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.inputs && schema.inputs.length > 0) {
                for (var input of schema.inputs) {
                    map[input.name] = input;
                }
            }
            this._inputCache[operator] = map;
        }
        return map[name] || null;
    }

    getInputs(operator, inputs) {
        var results = [];
        var schema = this._map[operator];
        var index = 0;
        while (index < inputs.length) {
            var result = { arguments: [] };
            var count = 1;
            var name = null;
            if (schema && schema.inputs) {
                if (index < schema.inputs.length) {
                    var input = schema.inputs[index];
                    name = input.name;
                    if (schema.inputs[index].option == 'variadic') {
                        count = inputs.length - index;
                    }
                }
            }
            else {
                if (index == 0) {
                    name = 'input';
                }
            }
            result.name = name ? name : '(' + index.toString() + ')';
            var array = inputs.slice(index, index + count);
            for (var j = 0; j < array.length; j++) {
                result.arguments.push({ id: array[j] });
            }
            index += count;
            results.push(result);
        }
        return results;
    }

    getOutputName(operator, index) {
        var schema = this._map[operator];
        if (schema) {
            var outputs = schema.outputs;
            if (outputs && index < outputs.length) {
                var output = outputs[index];
                if (output) {
                    var name = output.name;
                    if (name) {
                        return name;
                    }
                } 
            }
        }
        if (index == 0) {
            return 'output';
        }
        return '(' + index.toString() + ')';
    }
};

coreml.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Core ML model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = coreml.ModelFactory;
}