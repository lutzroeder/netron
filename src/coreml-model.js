/*jshint esversion: 6 */

var coreml = null;

class CoreMLModelFactory {

    match(context, host) {
        var extension = context.identifier.split('.').pop();
        return extension == 'mlmodel';
    }

    open(context, host, callback) { 
        host.require('coreml', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var decodedBuffer = null;
            try {
                coreml = protobuf.roots.coreml.CoreML.Specification;
                decodedBuffer = coreml.Model.decode(context.buffer);
            }
            catch (error) {
                callback(new CoreMLError('File format is not coreml.Model (' + error.message + ').'), null);
                return;
            }
            CoreMLOperatorMetadata.open(host, (err, metadata) => {
                try {
                    var model = new CoreMLModel(decodedBuffer);
                    callback(null, model);
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new CoreMLError(error.message), null);
                    return;
                }
            });
        });
    }
}

class CoreMLModel {

    constructor(model) {
        this._specificationVersion = model.specificationVersion;
        this._graphs = [ new CoreMLGraph(model) ];
        if (model.description && model.description.metadata) {
            var metadata = model.description.metadata;
            if (metadata.versionString) {
                this._version = metadata.versionString;
            }
            if (metadata.author) {
                this._author = metadata.author;
            }
            if (metadata.shortDescription) {
                this._description = metadata.shortDescription;
            }
            if (metadata.license) {
                this._license = metadata.license;
            }
            if (metadata.userDefined && Object.keys(metadata.userDefined).length > 0) {
                debugger;
            }
        }
    }

    get format() {
        return 'CoreML v' + this._specificationVersion.toString();
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
}

class CoreMLGraph {

    constructor(model)
    {
        this._description = model.description;
        this._groups = false; 
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._operators = {};

        if (this._description) {
            this._inputs = this._description.input.map((input) => {
                var connection = new CoreMLConnection(input.name, CoreMLGraph._formatFeatureType(input.type), input.shortDescription, null);
                return new CoreMLArgument(input.name, true, [ connection ]);
            });

            this._outputs = this._description.output.map((output) => {
                var connection = new CoreMLConnection(output.name, CoreMLGraph._formatFeatureType(output.type), output.shortDescription, null);
                return new CoreMLArgument(output.name, true, [ connection ]);
            });
        }

        this._type = this._loadModel(model, {}, '');
    }

    get operators() {
        return this._operators;
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
        this._nodes.forEach((node) => {
            node._outputs = node._outputs.map((output) => (output != name) ? output : newName);
        });
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
            this._operators[operator] = (this._operators[operator] || 0) + 1;
            this._nodes.push(new CoreMLNode(group, operator, null, classifier[operator], [ labelProbabilityInput ], [ predictedProbabilitiesName, predictedFeatureName ]));
        }
    }

    _loadModel(model, scope, group) {
        this._groups = this._groups | (group.length > 0 ? true : false);
        if (model.neuralNetworkClassifier) {
            var neuralNetworkClassifier = model.neuralNetworkClassifier;
            neuralNetworkClassifier.layers.forEach((layer) => {
                var operator = layer.layer;
                this._createNode(scope, group, operator, layer.name, layer[operator], layer.input, layer.output);
            });
            this._updateClassifierOutput(group, neuralNetworkClassifier);
            if (neuralNetworkClassifier.preprocessing && neuralNetworkClassifier.preprocessing.length > 0) {               
                var preprocessingInput = this._description.input[0].name;
                var inputNodes = [];
                this._nodes.forEach((node) => {
                    if (node._inputs.some((input => input == preprocessingInput))) {
                        inputNodes.push(node);
                    }
                });
                var preprocessorOutput = preprocessingInput;
                var preprocessorIndex = 0;
                var nodes = [];
                neuralNetworkClassifier.preprocessing.forEach((preprocessing) => {
                    var operator = preprocessing.preprocessor;
                    var input = preprocessing.featureName ? preprocessing.featureName : preprocessorOutput;
                    preprocessorOutput = preprocessingInput + ':' + preprocessorIndex.toString();
                    this._createNode(scope, group, operator, null, preprocessing[operator], [ input ], [ preprocessorOutput ]);
                    preprocessorIndex++;
                });
                inputNodes.forEach((node) => {
                    node._inputs = node._inputs.map((input) => (input != preprocessingInput) ? input : preprocessorOutput);
                });
            }
            return 'Neural Network Classifier';
        }
        else if (model.neuralNetwork) {
            model.neuralNetwork.layers.forEach((layer) => {
                var operator = layer.layer;
                this._createNode(scope, group, operator, layer.name, layer[operator], layer.input, layer.output);
            });
            return 'Neural Network';
        }
        else if (model.neuralNetworkRegressor) {
            model.neuralNetworkRegressor.layers.forEach((layer) => {
                var operator = layer.layer;
                this._createNode(scope, group, operator, layer.name, layer[operator], layer.input, layer.output);
            });
            return 'Neural Network Regressor';
        }
        else if (model.pipeline) {
            model.pipeline.models.forEach((subModel) => {
                this._loadModel(subModel, scope, (group ? (group + '/') : '') + 'pipeline');
            });
            return 'Pipeline';
        }
        else if (model.pipelineClassifier) {
            model.pipelineClassifier.pipeline.models.forEach((subModel) => {
                this._loadModel(subModel, scope, (group ? (group + '/') : '') + 'pipelineClassifier');
            });
            return 'Pipeline Classifier';
        }
        else if (model.pipelineRegressor) {
            model.pipelineRegressor.pipeline.models.forEach((subModel) => {
                this._loadModel(subModel, scope, (group ? (group + '/') : '') + 'pipelineRegressor');
            });
            return 'Pipeline Regressor';
        }
        else if (model.glmClassifier) {
            this._createNode(scope, group, 'glmClassifier', null, 
                { classEncoding: model.glmClassifier.classEncoding, 
                  offset: model.glmClassifier.offset, 
                  weights: model.glmClassifier.weights }, 
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
            CoreMLGraph.formatFeatureDescriptionList(model.description.input),
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
                { coefficients: model.supportVectorClassifier.coefficients, 
                  denseSupportVectors: model.supportVectorClassifier.denseSupportVectors,
                  kernel: model.supportVectorClassifier.kernel,
                  numberOfSupportVectorsPerClass: model.supportVectorClassifier.numberOfSupportVectorsPerClass,
                  probA: model.supportVectorClassifier.probA,
                  probB: model.supportVectorClassifier.probB,
                  rho: model.supportVectorClassifier.rho,
                  supportVectors: model.supportVectorClassifier.supportVectors }, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]);
            this._updateClassifierOutput(group, model.supportVectorClassifier);
            return 'Support Vector Classifier';            
        }
        else if (model.supportVectorRegressor) {
            this._createNode(scope, group, 'supportVectorRegressor', null, 
                { coefficients: model.supportVectorRegressor.coefficients, 
                  kernel: model.supportVectorRegressor.kernel,
                  rho: model.supportVectorRegressor.rho,
                  supportVectors: model.supportVectorRegressor.supportVectors }, 
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
        }
        return 'Unknown';
    }

    _createNode(scope, group, operator, name, data, inputs, outputs) {
        this._operators[operator] = (this._operators[operator] || 0) + 1;
        inputs = inputs.map((input) => scope[input] ? scope[input].connection : input);
        outputs = outputs.map((output) => {
            if (scope[output]) {
                scope[output].counter++;
                var next = output + '\n' + scope[output].counter.toString(); // custom connection id
                scope[output].connection = next;
                return next;
            }
            scope[output] = {
                connection: output,
                counter: 0
            };
            return output;
        });

        var node = new CoreMLNode(group, operator, name, data, inputs, outputs);
        this._nodes.push(node);
        return node;
    }

    static _formatFeatureType(type) {
        var result = '';
        switch (type.Type) {
            case 'multiArrayType':
                result = CoreMLGraph.formatArrayDataType(type.multiArrayType.dataType);
                if (type.multiArrayType.shape && type.multiArrayType.shape.length > 0) {
                    result += '[' + type.multiArrayType.shape.map(dimension => dimension.toString()).join(',') + ']';
                }
                break;
            case 'imageType':
                result = 'image(' + CoreMLGraph.formatColorSpace(type.imageType.colorSpace) + ',' + type.imageType.width.toString() + 'x' + type.imageType.height.toString() + ')';
                break;
            case 'dictionaryType':
                result = 'map<' + type.dictionaryType.KeyType.replace('KeyType', '') + ',float64>';
                break;
            case 'stringType':
                result = 'string';
                break;
            case 'doubleType':
                result = 'float64';
                break;
            case 'int64Type':
                result = 'int64';
                break;
        }
        if (type.isOptional) {
            result += '?';
        }
        return result;
    }

    static formatArrayDataType(dataType) {
        switch (dataType) {
            case coreml.ArrayFeatureType.ArrayDataType.FLOAT32:
                return 'float32';
            case coreml.ArrayFeatureType.ArrayDataType.INT32:
                return 'int32';
            case coreml.ArrayFeatureType.ArrayDataType.DOUBLE:
                return 'float64';
        }
        return '?';
    }

    static formatColorSpace(colorSpace) {
        switch (colorSpace) {
            case coreml.ImageFeatureType.ColorSpace.GRAYSCALE:
                return 'Grayscale';
            case coreml.ImageFeatureType.ColorSpace.RGB:
                return 'RGB';
            case coreml.ImageFeatureType.ColorSpace.BGR:
                return 'BGR';
        }
        return '?';
    }

    static formatFeatureDescriptionList(list) {
        return list.map((item) => item.name);
    }
}

class CoreMLArgument {
    constructor(name, visible, connections) {
        this._name = name;
        this._visible = visible;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get connections() {
        return this._connections;
    }
}

class CoreMLConnection {
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

    get initializer() {
        return this._initializer;
    }
}

class CoreMLNode {

    constructor(group, operator, name, data, inputs, outputs) {
        if (group) {
            this._group = group;
        }
        this._operator = operator;
        this._name = name;
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        this._initializers = [];
        if (data) {
            var initializerMap = this.initializer(data);
            Object.keys(data).forEach((key) => {
                if (!initializerMap[key]) {
                    this._attributes.push(new CoreMLAttribute(this.operator, key, data[key]));
                }
            });
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get category() {
        return CoreMLOperatorMetadata.operatorMetadata.getOperatorCategory(this.operator);
    }

    get group() {
        return this._group ? this._group : null;
    }

    get documentation() {
        return CoreMLOperatorMetadata.operatorMetadata.getOperatorDocumentation(this.operator);
    }

    get inputs() {
        var inputs = CoreMLOperatorMetadata.operatorMetadata.getInputs(this._operator, this._inputs).map((input) => {
            return new CoreMLArgument(input.name, true, input.connections.map((connection) => {
                return new CoreMLConnection(connection.id, connection.type, null, null);
            }));
        });
        this._initializers.forEach((initializer) => {
            var connection = new CoreMLConnection(null, null, null, initializer);
            var visible = CoreMLOperatorMetadata.operatorMetadata.getInputVisible(this._operator, initializer.name);
            inputs.push(new CoreMLArgument(initializer.name, visible, [ connection ]));
        });
        return inputs;
    }

    get outputs() {
        return this._outputs.map((output, index) => {
            var name = CoreMLOperatorMetadata.operatorMetadata.getOutputName(this._operator, index);
            return new CoreMLArgument(name, true, [ new CoreMLConnection(output, null, null, null) ]);
        });
    }

    get attributes() {
        return this._attributes;
    }

    initializer(data) {
        switch (this._operator) {
            case 'convolution':
                var weightsShape = [ data.outputChannels, data.kernelChannels, data.kernelSize[0], data.kernelSize[1] ];
                if (data.isDeconvolution) {
                    weightsShape[0] = data.kernelChannels;
                    weightsShape[1] = Math.floor(data.outputChannels / (data.nGroups != 0 ? data.nGroups : 1));
                }    
                this._initializers.push(new CoreMLTensor('Weights', 'weights', weightsShape, data.weights));
                if (data.hasBias) {
                    this._initializers.push(new CoreMLTensor('Weights', 'bias', [ data.outputChannels ], data.bias));
                }
                return { 'weights': true, 'bias': data.hasBias };
            case 'innerProduct':
                this._initializers.push(new CoreMLTensor('Weights', 'weights', [ data.outputChannels, data.inputChannels ], data.weights));
                if (data.hasBias) {
                    this._initializers.push(new CoreMLTensor('Weights', 'bias', [ data.outputChannels ], data.bias));
                }
                return { 'weights': true, 'bias': data.hasBias };
            case 'batchnorm':
                this._initializers.push(new CoreMLTensor('Weights', 'gamma', [ data.channels ], data.gamma));
                this._initializers.push(new CoreMLTensor('Weights', 'beta', [ data.channels ], data.beta));
                if (data.mean) {
                    this._initializers.push(new CoreMLTensor('Weights', 'mean', [ data.channels ], data.mean));
                }
                if (data.variance) {
                    this._initializers.push(new CoreMLTensor('Weights', 'variance', [ data.channels ], data.variance));
                }
                return { 'gamma': true, 'beta': true, 'mean': true, 'variance': true };
            case 'embedding':
                this._initializers.push(new CoreMLTensor('Weights', 'weights', [ data.inputDim, data.outputChannels ], data.weights));
                return { 'weights': true };
            case 'loadConstant':    
                this._initializers.push(new CoreMLTensor('Weights', 'data', data.shape, data.data));            
                return { 'data': true };
            case 'scale':
                this._initializers.push(new CoreMLTensor('Weights', 'scale', data.shapeScale, data.scale));
                if (data.hasBias) {
                    this._initializers.push(new CoreMLTensor('Weights', 'bias', data.shapeBias, data.bias));
                }
                return { 'scale': true, 'bias': data.hasBias };
            case 'bias':
                this._initializers.push(new CoreMLTensor('Weights', 'bias', data.shapeBias, data.bias));
                return { 'bias': true };
            case 'simpleRecurrentLayer':
                this._initializers.push(new CoreMLTensor('Weights', 'weights', null, data.weightMatrix));
                this._initializers.push(new CoreMLTensor('Weights', 'recurrent', null, data.recursionMatrix));
                if (data.hasBiasVectors) {
                    this._initializers.push(new CoreMLTensor('Weights', 'bias', null, data.biasVector));
                }
                return { 'weightMatrix': true, 'recursionMatrix': true, 'biasVector': data.hasBiasVectors };
            case 'gru':
                this._initializers.push(new CoreMLTensor('Weights', 'updateGateWeightMatrix', null, data.updateGateWeightMatrix));
                this._initializers.push(new CoreMLTensor('Weights', 'resetGateWeightMatrix', null, data.resetGateWeightMatrix));
                this._initializers.push(new CoreMLTensor('Weights', 'outputGateWeightMatrix', null, data.outputGateWeightMatrix));
                this._initializers.push(new CoreMLTensor('Weights', 'updateGateRecursionMatrix', null, data.updateGateRecursionMatrix));
                this._initializers.push(new CoreMLTensor('Weights', 'resetGateRecursionMatrix', null, data.resetGateRecursionMatrix));
                this._initializers.push(new CoreMLTensor('Weights', 'outputGateRecursionMatrix', null, data.outputGateRecursionMatrix));
                if (data.hasBiasVectors) {
                    this._initializers.push(new CoreMLTensor('Weights', 'updateGateBiasVector', null, data.updateGateBiasVector));
                    this._initializers.push(new CoreMLTensor('Weights', 'resetGateBiasVector', null, data.resetGateBiasVector));
                    this._initializers.push(new CoreMLTensor('Weights', 'outputGateBiasVector', null, data.outputGateBiasVector));
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
                    this._initializers.push(new CoreMLTensor('Weights', 'inputGateWeightMatrix' + suffix, matrixShape, weights.inputGateWeightMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'forgetGateWeightMatrix' + suffix, matrixShape, weights.forgetGateWeightMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'blockInputWeightMatrix' + suffix, matrixShape, weights.blockInputWeightMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'outputGateWeightMatrix' + suffix, matrixShape, weights.outputGateWeightMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'inputGateRecursionMatrix' + suffix, matrixShape, weights.inputGateRecursionMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'forgetGateRecursionMatrix' + suffix, matrixShape,weights.forgetGateRecursionMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'blockInputRecursionMatrix' + suffix, matrixShape, weights.blockInputRecursionMatrix));
                    this._initializers.push(new CoreMLTensor('Weights', 'outputGateRecursionMatrix' + suffix, matrixShape, weights.outputGateRecursionMatrix));
                    if (data.params.hasBiasVectors) {
                        this._initializers.push(new CoreMLTensor('Weights', 'inputGateBiasVector' + suffix, vectorShape, weights.inputGateBiasVector));
                        this._initializers.push(new CoreMLTensor('Weights', 'forgetGateBiasVector' + suffix, vectorShape, weights.forgetGateBiasVector));
                        this._initializers.push(new CoreMLTensor('Weights', 'blockInputBiasVector' + suffix, vectorShape, weights.blockInputBiasVector));
                        this._initializers.push(new CoreMLTensor('Weights', 'outputGateBiasVector' + suffix, vectorShape, weights.outputGateBiasVector));
                    }
                    if (data.params.hasPeepholeVectors) {
                        this._initializers.push(new CoreMLTensor('Weights', 'inputGatePeepholeVector' + suffix, vectorShape, weights.inputGatePeepholeVector));
                        this._initializers.push(new CoreMLTensor('Weights', 'forgetGatePeepholeVector' + suffix, vectorShape, weights.forgetGatePeepholeVector));
                        this._initializers.push(new CoreMLTensor('Weights', 'outputGatePeepholeVector' + suffix, vectorShape, weights.outputGatePeepholeVector));
                    }
                }
                return { 'weightParams': true };
        }
        return {};
    }
}

class CoreMLAttribute {

    constructor(operator, name, value) {
        this._name = name;
        this._value = value;
        if (!CoreMLOperatorMetadata.operatorMetadata.getAttributeVisible(operator, this._name, this._value)) {
            this._visible = false;
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
}

class CoreMLTensor {

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
                this._data = data.float16Value;
                dataType = 'float16';
            }
            else if (data.rawValue && data.rawValue.length > 0) {
                this._data = null;
                dataType = 'byte';
                shape = [];
            }
        }

        this._type = new CoreMLTensorType(dataType, shape);
    }

    get id() {
        return this._name;
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
        context.data = this._data;
        context.dataType = this._type.dataType;
        context.shape = this._type.shape;
 
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.shape[dimension];
        if (dimension == context.shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float32':
                        results.push(this._data[context.index]);
                        break;
                    case 'float16':
                        var value = this._data[context.index] | (this._data[context.index + 1] << 8);
                        results.push(CoreMLTensor._decodeFloat16(value));
                        context.index += 2;
                        break;
                }
                context.index++;
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

    static _decodeFloat16(value) {
        var s = (value & 0x8000) >> 15;
        var e = (value & 0x7C00) >> 10;
        var f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
    }
}

class CoreMLTensorType {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }

}

class CoreMLOperatorMetadata 
{

    static open(host, callback) {
        if (CoreMLOperatorMetadata.operatorMetadata) {
            callback(null, CoreMLOperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'coreml-metadata.json', 'utf-8', (err, data) => {
                CoreMLOperatorMetadata.operatorMetadata = new CoreMLOperatorMetadata(data);
                callback(null, CoreMLOperatorMetadata.operatorMetadata);
            });
        }    
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema)
                    {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                    }
                });
            }
        }
    }

    getOperatorCategory(operator) {
        var schema = this._map[operator];
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    getInputs(operator, inputs) {
        var results = [];
        var schema = this._map[operator];
        var index = 0;
        while (index < inputs.length) {
            var result = { connections: [] };
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
                result.connections.push({ id: array[j] });
            }
            index += count;
            results.push(result);
        }
        return results;
    }

    getInputVisible(operator, name) {
        var schema = this._map[operator];
        if (schema && schema.inputs) {
            if (!schema.inputsMap) {
                schema.inputsMap = {};
                schema.inputs.forEach((input) => {
                    schema.inputsMap[input.name] = input;
                });
            }
            var input = schema.inputsMap[name];
            if (input && input.hasOwnProperty('visible')) {
                return input.visible;
            }
        }
        return true;
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

    getAttributeVisible(operator, name, value) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributesMap) {
                schema.attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributesMap[attribute.name] = attribute;
                });
            }
            var attribute = schema.attributesMap[name];

            if (attribute) {
                if (attribute.hasOwnProperty('visible')) {
                    return attribute.visible;
                }
                if (attribute.hasOwnProperty('default')) {
                    if (Array.isArray(value)) {
                        value = value.map((item) => {
                            if (item && item.__isLong__) {
                                return item.toNumber();
                            }
                            return item;
                        });
                    }
                    return JSON.stringify(attribute.default) != JSON.stringify(value);
                 }
            }
        }
        return true;
    }

    getOperatorDocumentation(operator) {
        var schema = this._map[operator];
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            return schema;
        }
        return '';
    }
}

class CoreMLError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading CoreML model.';
    }
}