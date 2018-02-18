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
            var labelProbabilityLayerName = neuralNetworkClassifier.labelProbabilityLayerName;
            if (!labelProbabilityLayerName && neuralNetworkClassifier.layers.length > 0) {
                labelProbabilityLayerName = neuralNetworkClassifier.layers.slice(-1).pop().output[0];
            }
            var predictedFeatureName = this._description.predictedFeatureName;
            var predictedProbabilitiesName = this._description.predictedProbabilitiesName;
            if (predictedFeatureName && predictedProbabilitiesName && labelProbabilityLayerName && neuralNetworkClassifier.ClassLabels) {
                var labelProbabilityInput = this.updateOutput(labelProbabilityLayerName, labelProbabilityLayerName + ':labelProbabilityLayerName');
                var operator = neuralNetworkClassifier.ClassLabels;
                this._nodes.push(new CoreMLNode(operator, null, neuralNetworkClassifier[operator], [ labelProbabilityInput ], [ predictedProbabilitiesName, predictedFeatureName ]));
            }
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
            debugger;
        }
        else if (model.glmClassifier) {
            this._type = "Generalized Linear Classifier";
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
        this._initializer = [];
        if (data) {
            Object.keys(data).forEach((key) => {
                var value = data[key];
                if (value instanceof coreml.WeightParams) {
                    this._initializer.push({
                        name: key,
                        value: value
                    });
                }
                else {
                    this._attributes.push(new CoreMLAttribute(key, value));
                }
            });
        }
    }

    get operator() {
        return this._operator;
    }

    get category() {
        return CoreMLOperatorMetadata.operatorMetadata.getOperatorCategory(this.operator);
    }
    
    get name() {
        return this._name;
    }

    get inputs() {
        var results = [];
        this._inputs.forEach((input, index) => {
            results.push({
                name: '(' + index.toString() + ')',
                connections: [ { id: input } ]
            });
        });
        this._initializer.forEach((initializer) => {
            results.push({
                name: initializer.name,
                connections: [ { 
                    id: initializer, 
                    initializer: initializer.value } ]
            });
        });
        return results;
    }

    get outputs() {
        var results = [];
        this._outputs.forEach((output, index) => {
            results.push({
                name: '(' + index.toString() + ')',
                connections: [ { id: output } ]
            });
        });
        return results;
    }

    get attributes() {
        return this._attributes;
    }

}

class CoreMLAttribute {

    constructor(name, value) {
        this._name = name;
        this._value = value;
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
}
