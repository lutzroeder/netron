/*jshint esversion: 6 */

// Experimental

var coreml = null;

class CoreMLModel {

    static open(buffer, identifier, host, callback) { 
        host.import('/coreml.js', (err) => {
            if (err) {
                callback(err, null);
            }
            else {
                coreml = protobuf.roots.coreml.CoreML.Specification;
                CoreMLModel.create(buffer, identifier, host, (err, model) => {
                    callback(err, model);
                });
            }
        });
    }

    static create(buffer, identifier, host, callback) {
        try {
            var decodedBuffer = coreml.Model.decode(buffer);
            var model = new CoreMLModel(decodedBuffer, identifier);
            CoreMLOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(err, null);
        }
    }

    constructor(model, identifier) {
        this._specificationVersion = model.specificationVersion;
        this._description = model.description;
        this._graphs = [ new CoreMLGraph(model) ];
    }

    get properties() {
        var results = [];

        results.push({ name: 'Format', value: 'CoreML v' + this._specificationVersion.toString() });

        if (this._description && this._description.metadata) {
            var metadata = this._description.metadata;
            if (metadata.versionString) {
                results.push({ name: 'Version', value: metadata.versionString });
            }
            if (metadata.author) {
                results.push({ name: 'Author', value: metadata.author });
            }
            if (metadata.shortDescription) {
                results.push({ name: 'Description', value: metadata.shortDescription });
            }
            if (metadata.license) {
                results.push({ name: 'License', value: metadata.license });
            }
            if (metadata.userDefined && Object.keys(metadata.userDefined).length > 0) {
                debugger;
            }
        }

        return results;
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

        this._inputs = this._description.input.map((input) => {
            return {
                id: input.name,
                name: input.name,
                description: input.shortDescription,
                type: CoreMLGraph.formatFeatureType(input.type) 
            };
        });

        this._outputs = this._description.output.map((output) => {
            return {
                id: output.name,
                name: output.name,
                description: output.shortDescription,
                type: CoreMLGraph.formatFeatureType(output.type) 
            };
        });

        this._nodes = [];
        this._type = this.loadModel(model, '');
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

    updateInput(name, newName) {
        this._nodes.forEach((node) => {
            node._inputs = node._inputs.map((input) => (input != name) ? input : newName);
        });
        return newName;
    }

    updateOutput(name, newName) {
        this._nodes.forEach((node) => {
            node._outputs = node._outputs.map((output) => (output != name) ? output : newName);
        });
        return newName;
    }

    updateClassifierOutput(group, classifier) {
        var labelProbabilityLayerName = classifier.labelProbabilityLayerName;
        if (!labelProbabilityLayerName && this._nodes.length > 0) {
            labelProbabilityLayerName = this._nodes.slice(-1).pop()._outputs[0];
        }
        var predictedFeatureName = this._description.predictedFeatureName;
        var predictedProbabilitiesName = this._description.predictedProbabilitiesName;
        if (predictedFeatureName && predictedProbabilitiesName && labelProbabilityLayerName && classifier.ClassLabels) {
            var labelProbabilityInput = this.updateOutput(labelProbabilityLayerName, labelProbabilityLayerName + ':labelProbabilityLayerName');
            var operator = classifier.ClassLabels;
            this._nodes.push(new CoreMLNode(group, operator, null, classifier[operator], [ labelProbabilityInput ], [ predictedProbabilitiesName, predictedFeatureName ]));
        }
    }

    loadModel(model, group) {
        this._groups = this._groups | (group.length > 0 ? true : false);
        if (model.neuralNetworkClassifier) {
            var neuralNetworkClassifier = model.neuralNetworkClassifier;
            neuralNetworkClassifier.layers.forEach((layer) => {
                var operator = layer.layer;
                this._nodes.push(new CoreMLNode(group, operator, layer.name, layer[operator], layer.input, layer.output));
            });
            this.updateClassifierOutput(group, neuralNetworkClassifier);
            if (neuralNetworkClassifier.preprocessing && neuralNetworkClassifier.preprocessing.length > 0) {               
                var preprocessingInput = this._description.input[0].name;
                var preprocessorOutput = preprocessingInput;
                var preprocessorIndex = 0;
                var nodes = [];
                neuralNetworkClassifier.preprocessing.forEach((preprocessing) => {
                    var operator = preprocessing.preprocessor;
                    var input = preprocessing.featureName ? preprocessing.featureName : preprocessorOutput;
                    preprocessorOutput = preprocessingInput + ':' + preprocessorIndex.toString();
                    nodes.push(new CoreMLNode(group, operator, null, preprocessing[operator], [ input ], [ preprocessorOutput ]));
                    preprocessorIndex++;
                });
                this.updateInput(preprocessingInput, preprocessorOutput);
                nodes.forEach((node) => {
                    this._nodes.push(node);
                });
            }
            return 'Neural Network Classifier';
        }
        else if (model.neuralNetwork) {
            model.neuralNetwork.layers.forEach((layer) => {
                var operator = layer.layer;
                this._nodes.push(new CoreMLNode(group, operator, layer.name, layer[operator], layer.input, layer.output));
            });
            return 'Neural Network';
        }
        else if (model.neuralNetworkRegressor) {
            model.neuralNetworkRegressor.layers.forEach((layer) => {
                var operator = layer.layer;
                this._nodes.push(new CoreMLNode(group, operator, layer.name, layer[operator], layer.input, layer.output));
            });
            return 'Neural Network Regressor';
        }
        else if (model.pipeline) {
            model.pipeline.models.forEach((subModel) => {
                this.loadModel(subModel, (group ? (group + '/') : '') + 'pipeline');
            });
            return 'Pipeline';
        }
        else if (model.pipelineClassifier) {
            model.pipelineClassifier.pipeline.models.forEach((subModel) => {
                this.loadModel(subModel, (group ? (group + '/') : '') + 'pipelineClassifier');
            });
            return 'Pipeline Classifier';
        }
        else if (model.pipelineRegressor) {
            model.pipelineRegressor.pipeline.models.forEach((subModel) => {
                this.loadModel(subModel, (group ? (group + '/') : '') + 'pipelineRegressor');
            });
            return 'Pipeline Regressor';
        }
        else if (model.glmClassifier) {
            this._nodes.push(new CoreMLNode(group, 'glmClassifier', null, 
                { classEncoding: model.glmClassifier.classEncoding, 
                  offset: model.glmClassifier.offset, 
                  weights: model.glmClassifier.weights }, 
                [ model.description.input[0].name ],
                [ model.description.predictedProbabilitiesName ]));
            this.updateClassifierOutput(group, model.glmClassifier);
            return 'Generalized Linear Classifier';
        }
        else if (model.dictVectorizer) {
            this._nodes.push(new CoreMLNode(group, 'dictVectorizer', null, model.dictVectorizer,
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]));
            return 'Dictionary Vectorizer';
        }
        else if (model.featureVectorizer) {
            this._nodes.push(new CoreMLNode(group, 'featureVectorizer', null, model.featureVectorizer, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]));
            return 'Feature Vectorizer';
        }
        else if (model.treeEnsembleClassifier) {
            this._nodes.push(new CoreMLNode(group, 'treeEnsembleClassifier', null, model.treeEnsembleClassifier.treeEnsemble, 
                [ model.description.input[0].name ],
                [ model.description.output[0].name ]));
            this.updateClassifierOutput(group, model.treeEnsembleClassifier);
            return 'Tree Ensemble Classifier';
        }
        return 'Unknown';
    }

    static formatFeatureType(type) {
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
                result = 'map<' + type.dictionaryType.KeyType.replace('KeyType', '') + ',double>';
                break;
            case 'stringType':
                result = 'string';
                break;
            case 'doubleType':
                result = 'double';
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
                return 'double';
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
                    this._attributes.push(new CoreMLAttribute(this, key, data[key]));
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
        var results = [];
        CoreMLOperatorMetadata.operatorMetadata.getInputs(this._operator, this._inputs).forEach((input) => {
            results.push(input);
        });
        this._initializers.forEach((initializer) => {
            var input = {
                name: initializer.name,
                connections: [ { 
                    id: initializer.id, 
                    type: initializer.type,
                    initializer: initializer, } ]
            };
            if (CoreMLOperatorMetadata.operatorMetadata.getInputHidden(this._operator, initializer.name)) {
                input.hidden = true;
            }
            results.push(input);
        });
        return results;
    }

    get outputs() {
        var results = [];
        this._outputs.forEach((output, index) => {
            results.push({
                name: CoreMLOperatorMetadata.operatorMetadata.getOutputName(this._operator, index),
                connections: [ { id: output } ]
            });
        });
        return results;
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
                this._initializers.push(new CoreMLTensor('weights', weightsShape, data.weights));
                if (data.hasBias) {
                    this._initializers.push(new CoreMLTensor('bias', [ data.bias.floatValue.length ], data.bias));
                }
                return { 'weights': true, 'bias': data.hasBias };
            case 'innerProduct':
                this._initializers.push(new CoreMLTensor('weights', [ data.outputChannels, data.inputChannels ], data.weights));
                if (data.hasBias) {
                    this._initializers.push(new CoreMLTensor('bias', [ data.outputChannels ], data.bias));
                }
                return { 'weights': true, 'bias': data.hasBias };
            case 'batchnorm':
                this._initializers.push(new CoreMLTensor('gamma', [ data.channels ], data.gamma));
                this._initializers.push(new CoreMLTensor('beta', [ data.channels ], data.beta));
                if (data.mean) {
                    this._initializers.push(new CoreMLTensor('mean', [ data.channels ], data.mean));
                }
                if (data.variance) {
                    this._initializers.push(new CoreMLTensor('variance', [ data.channels ], data.variance));
                }
                return { 'gamma': true, 'beta': true, 'mean': true, 'variance': true };
            case 'embedding':
                this._initializers.push(new CoreMLTensor('weights', [ data.inputDim, data.outputChannels ], data.weights));
                return { 'weights': true };
            case 'loadConstant':    
                this._initializers.push(new CoreMLTensor('data', data.shape, data.data));            
                return { 'data': true };
            case 'scale':
                this._initializers.push(new CoreMLTensor('scale', data.shapeScale, data.scale));
                if (data.hasBias) {
                    this._initializers.push(new CoreMLTensor('bias', data.shapeBias, data.bias));
                }
                return { 'scale': true, 'bias': data.hasBias };
            case 'bias':
                this._initializers.push(new CoreMLTensor('bias', data.shapeBias, data.bias));
                return { 'bias': true };
            case 'simpleRecurrentLayer':
                this._initializers.push(new CoreMLTensor('weights', null, data.weightMatrix));
                this._initializers.push(new CoreMLTensor('recurrent', null, data.recursionMatrix));
                if (data.hasBiasVectors) {
                    this._initializers.push(new CoreMLTensor('bias', null, data.biasVector));
                }
                return { 'weightMatrix': true, 'recursionMatrix': true, 'biasVector': data.hasBiasVectors };
            case 'gru':
                this._initializers.push(new CoreMLTensor('updateGateWeightMatrix', null, data.updateGateWeightMatrix));
                this._initializers.push(new CoreMLTensor('resetGateWeightMatrix', null, data.resetGateWeightMatrix));
                this._initializers.push(new CoreMLTensor('outputGateWeightMatrix', null, data.outputGateWeightMatrix));
                this._initializers.push(new CoreMLTensor('updateGateRecursionMatrix', null, data.updateGateRecursionMatrix));
                this._initializers.push(new CoreMLTensor('resetGateRecursionMatrix', null, data.resetGateRecursionMatrix));
                this._initializers.push(new CoreMLTensor('outputGateRecursionMatrix', null, data.outputGateRecursionMatrix));
                if (data.hasBiasVectors) {
                    this._initializers.push(new CoreMLTensor('updateGateBiasVector', null, data.updateGateBiasVector));
                    this._initializers.push(new CoreMLTensor('resetGateBiasVector', null, data.resetGateBiasVector));
                    this._initializers.push(new CoreMLTensor('outputGateBiasVector', null, data.outputGateBiasVector));
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
                    this._initializers.push(new CoreMLTensor('inputGateWeightMatrix' + suffix, matrixShape, weights.inputGateWeightMatrix));
                    this._initializers.push(new CoreMLTensor('forgetGateWeightMatrix' + suffix, matrixShape, weights.forgetGateWeightMatrix));
                    this._initializers.push(new CoreMLTensor('blockInputWeightMatrix' + suffix, matrixShape, weights.blockInputWeightMatrix));
                    this._initializers.push(new CoreMLTensor('outputGateWeightMatrix' + suffix, matrixShape, weights.outputGateWeightMatrix));
                    this._initializers.push(new CoreMLTensor('inputGateRecursionMatrix' + suffix, matrixShape, weights.inputGateRecursionMatrix));
                    this._initializers.push(new CoreMLTensor('forgetGateRecursionMatrix' + suffix, matrixShape,weights.forgetGateRecursionMatrix));
                    this._initializers.push(new CoreMLTensor('blockInputRecursionMatrix' + suffix, matrixShape, weights.blockInputRecursionMatrix));
                    this._initializers.push(new CoreMLTensor('outputGateRecursionMatrix' + suffix, matrixShape, weights.outputGateRecursionMatrix));
                    if (data.params.hasBiasVectors) {
                        this._initializers.push(new CoreMLTensor('inputGateBiasVector' + suffix, vectorShape, weights.inputGateBiasVector));
                        this._initializers.push(new CoreMLTensor('forgetGateBiasVector' + suffix, vectorShape, weights.forgetGateBiasVector));
                        this._initializers.push(new CoreMLTensor('blockInputBiasVector' + suffix, vectorShape, weights.blockInputBiasVector));
                        this._initializers.push(new CoreMLTensor('outputGateBiasVector' + suffix, vectorShape, weights.outputGateBiasVector));
                    }
                    if (data.params.hasPeepholeVectors) {
                        this._initializers.push(new CoreMLTensor('inputGatePeepholeVector' + suffix, vectorShape, weights.inputGatePeepholeVector));
                        this._initializers.push(new CoreMLTensor('forgetGatePeepholeVector' + suffix, vectorShape, weights.forgetGatePeepholeVector));
                        this._initializers.push(new CoreMLTensor('outputGatePeepholeVector' + suffix, vectorShape, weights.outputGatePeepholeVector));
                    }
                }
                return { 'weightParams': true };
        }
        return {};
    }
}

class CoreMLAttribute {

    constructor(owner, name, value, hidden) {
        this._owner = owner;
        this._name = name;
        this._value = value;
        if (hidden) {
            this._hidden = hidden;
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        if (Array.isArray(this._value)) {
            return this._value.map((item) => JSON.stringify(item)).join(', ');
        }
        return JSON.stringify(this._value);
    }

    get hidden() {
        return CoreMLOperatorMetadata.operatorMetadata.getAttributeHidden(this._owner.operator, this._name);
    }
}

class CoreMLTensor {

    constructor(name, shape, data) {
        this._name = name;
        this._shape = shape;
        this._type = null;
        this._data = null;
        if (data) {
            if (data.floatValue && data.floatValue.length > 0) {
                this._data = data.floatValue;
                this._type = 'float';
            }
            else if (data.float16Value && data.float16Value.length > 0) {
                this._data = data.float16Value;
                this._type = 'float16';
            }
            else if (data.rawValue && data.rawValue.length > 0) {
                this._data = null;
                this._type = 'byte';
                this._shape = [];
            }
        }
    }

    get id() {
        return this._name;
    }

    get name() {
        return this._name;
    }

    get title() {
        return 'Initializer';
    }

    get type() {
        if (this._type && this._shape) {
            return this._type + '[' + this._shape.join(',') + ']';
        }
        return '?';
    }

    get value() {
        if (this._data) {
            this._index = 0;
            this._count = 0;
            var result = this.read(0);
            delete this._index;
            delete this._count;
            return JSON.stringify(result, null, 4);
        }
        return '?';
    }

    read(dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                results.push(this._data[this._index]);
                this._index++;
                this._count++;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                results.push(this.read(dimension + 1));
            }
        }
        return results;
    }
}

class CoreMLOperatorMetadata 
{

    static open(host, callback) {
        if (CoreMLOperatorMetadata.operatorMetadata) {
            callback(null, CoreMLOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/coreml-operator.json', (err, data) => {
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

    getInputHidden(operator, name) {
        var schema = this._map[operator];
        if (schema && schema.inputs) {
            if (!schema.inputsMap) {
                schema.inputsMap = {};
                schema.inputs.forEach((input) => {
                    schema.inputsMap[input.name] = input;
                });
            }
            var input = schema.inputsMap[name];
            if (input) {
                return input.hidden;
            }
        }
        return false;
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

    getAttributeHidden(operator, name) {
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
                return attribute.hidden;
            }
        }
        return false;
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
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return '';
    }

}
