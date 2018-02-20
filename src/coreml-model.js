/*jshint esversion: 6 */

// Experimental

var coreml = null;

class CoreMLModel {

    static open(buffer, identifier, host, callback) { 
        host.import('/coreml.js', (err) => {
            if (err) {
                callback(new Error('Unsupported file extension \'.mlmodel\'.'), null);
                // callback(err, null);
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
        this._graphs = [ new CoreMLGraph(model, identifier) ];
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

    constructor(model, identifier)
    {
        this._name = identifier;
        this._description = model.description;

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
        if (model.neuralNetworkClassifier) {
            this._type = "Neural Network Classifier";
            var neuralNetworkClassifier = model.neuralNetworkClassifier;
            neuralNetworkClassifier.layers.forEach((layer) => {
                var operator = layer.layer;
                this._nodes.push(new CoreMLNode(operator, layer.name, layer[operator], layer.input, layer.output));
            });
            this.updateClassifierOutput(neuralNetworkClassifier);
            if (neuralNetworkClassifier.preprocessing && neuralNetworkClassifier.preprocessing.length > 0) {               
                var preprocessingInput = this._description.input[0].name;
                var preprocessorOutput = preprocessingInput;
                var preprocessorIndex = 0;
                var nodes = [];
                neuralNetworkClassifier.preprocessing.forEach((preprocessing) => {
                    var operator = preprocessing.preprocessor;
                    var input = preprocessing.featureName ? preprocessing.featureName : preprocessorOutput;
                    preprocessorOutput = preprocessingInput + ':' + preprocessorIndex.toString();
                    nodes.push(new CoreMLNode(operator, null, preprocessing[operator], [ input ], [ preprocessorOutput ]));
                    preprocessorIndex++;
                });
                this.updateInput(preprocessingInput, preprocessorOutput);
                nodes.forEach((node) => {
                    this._nodes.push(node);
                });
            }
        }
        else if (model.neuralNetwork) {
            this._type = "Neural Network";
            model.neuralNetwork.layers.forEach((layer) => {
                var operator = layer.layer;
                this._nodes.push(new CoreMLNode(operator, layer.name, layer[operator], layer.input, layer.output));
            });
        }
        else if (model.pipelineClassifier) {
            this._type = "Pipeline Classifier";
            this._nodes.push(new CoreMLNode('pipelineClassifier', null, model.pipelineClassifier, 
                this._description.input.map((input) => input.name), 
                this._description.output.map((output) => output.name)));
            this.updateClassifierOutput(model.pipelineClassifier);
            /*
            model.pipelineClassifier.pipeline.models.forEach((subModel, index) => {
                var buffer = coreml.Model.encode(subModel).finish();
                require('fs').writeFileSync(require('os').homedir + '/' + identifier + '_' + index.toString(), buffer);
                console.log();
            });
            */
            debugger;
        }
        else if (model.pipeline) {
            this._type = "Pipeline";
            this._nodes.push(new CoreMLNode('pipeline', null, model.pipeline, 
                this._description.input.map((input) => input.name), 
                this._description.output.map((output) => output.name)));
            /*
            model.pipeline.models.forEach((subModel, index) => {
                var buffer = coreml.Model.encode(subModel).finish();
                require('fs').writeFileSync(require('os').homedir + '/' + identifier + '_' + index.toString(), buffer);
            });
            */
            debugger;
        }
        else if (model.glmClassifier) {
            this._type = "Generalized Linear Classifier";
            this._nodes.push(new CoreMLNode('glmClassifier', null, 
                { classEncoding: model.glmClassifier.classEncoding, 
                  offset: model.glmClassifier.offset, 
                  weights: model.glmClassifier.weights }, 
                [ this._description.input[0].name ],
                [ this._description.predictedProbabilitiesName ]));
            this.updateClassifierOutput(model.glmClassifier);
        }
        else if (model.dictVectorizer) {
            this._type = "Dictionary Vectorizer";
            this._nodes.push(new CoreMLNode('dictVectorizer', null, model.dictVectorizer,
                [ this._description.input[0].name ],
                [ this._description.output[0].name ]));
            debugger;
        }
        else if (model.featureVectorizer) {
            this._type = "Feature Vectorizer";
            this._nodes.push(new CoreMLNode('featureVectorizer', null, model.featureVectorizer, 
                [ this._description.input[0].name ],
                [ this._description.output[0].name ]));
            debugger;
        }
        else if (model.treeEnsembleClassifier) {
            this._type = "Tree Ensemble Classifier";
            this._nodes.push(new CoreMLNode('treeEnsembleClassifier', null, model.treeEnsembleClassifier.treeEnsemble, 
                [ this._description.input[0].name ],
                [ this._description.output[0].name ]));
            this.updateClassifierOutput(model.treeEnsembleClassifier);
            debugger;          
        }
        else {
            debugger;
        }
    }

    get name() {
        return this._name;
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

    updateClassifierOutput(classifier) {
        var labelProbabilityLayerName = classifier.labelProbabilityLayerName;
        if (!labelProbabilityLayerName && this._nodes.length > 0) {
            labelProbabilityLayerName = this._nodes.slice(-1).pop()._outputs[0];
        }
        var predictedFeatureName = this._description.predictedFeatureName;
        var predictedProbabilitiesName = this._description.predictedProbabilitiesName;
        if (predictedFeatureName && predictedProbabilitiesName && labelProbabilityLayerName && classifier.ClassLabels) {
            var labelProbabilityInput = this.updateOutput(labelProbabilityLayerName, labelProbabilityLayerName + ':labelProbabilityLayerName');
            var operator = classifier.ClassLabels;
            this._nodes.push(new CoreMLNode(operator, null, classifier[operator], [ labelProbabilityInput ], [ predictedProbabilitiesName, predictedFeatureName ]));
        }
    }

    static formatFeatureType(type) {
        var result = '';
        switch (type.Type) {
            case 'multiArrayType':
                result = CoreMLGraph.formatArrayDataType(type.multiArrayType.dataType);
                if (type.multiArrayType.shape && type.multiArrayType.shape.length > 0) {
                    result += '[' + type.multiArrayType.shape.map(dimension => dimension.toString()).join(',') + ']';
                }
                return result;
            case 'imageType':
                return 'image(' + CoreMLGraph.formatColorSpace(type.imageType.colorSpace) + ',' + type.imageType.width.toString() + 'x' + type.imageType.height.toString() + ')';
            case 'dictionaryType':
                return 'map<' + type.dictionaryType.KeyType.replace('KeyType', '') + ',double>';
            case 'stringType':
                return 'string';
            case 'doubleType':
                return 'double';
            case 'int64Type':
                return 'int64';
        }
        debugger;
        return '?';
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

    constructor(operator, name, data, inputs, outputs) {
        this._operator = operator;
        this._name = name;
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        this._initializers = [];
        if (data) {
            Object.keys(data).forEach((key) => {
                this.initialize(key, data[key]);
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

    initialize(name, value) {
        switch (this._operator) {
            case 'glmClassifier':
                if (name == 'weights') {
                    this._initializers.push(new CoreMLTensor(name, value));
                    return;
                }
                break;
            case 'convolution':
            case 'innerProduct':
            case 'embedding':
            case 'batchnorm':
            case 'bias':
            case 'scale':
            case 'loadConstant':    
            case 'simpleRecurrentLayer':
            case 'gru':
                if (value instanceof coreml.WeightParams) {
                    this._initializers.push(new CoreMLTensor(name, value));
                    return;
                }
                break;
            case 'uniDirectionalLSTM':
                if (value instanceof coreml.LSTMWeightParams) {
                    Object.keys(value).forEach((key) => {
                        this._initializers.push(new CoreMLTensor(key, value));
                    });
                    return;
                }
                break;
        }

        this._attributes.push(new CoreMLAttribute(this, name, value));
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

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get id() {
        return this._name;
    }

    get name() {
        return this._name;
    }

    get value() {
        return JSON.stringify(this._value);
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
            schema = Object.assign({}, schema);
            schema.name = operator;
            schema.description = this.markdown(schema.description);
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = this.markdown(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = this.markdown(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = this.markdown(output.description);
                    }
                });
            }
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return "";
    }

    markdown(text) {
        if (text) {
            text = text.replace(/\`\`(.*?)\`\`/gm, (match, content) => '<code>' + content + '</code>');
            text = text.replace(/\`(.*?)\`/gm, (match, content) => '<code>' + content + '</code>');                
        }
        return text;
    }

}
