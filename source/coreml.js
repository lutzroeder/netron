/* jshint esversion: 6 */

var coreml = coreml || {};
var json = json || require('./json');
var protobuf = protobuf || require('./protobuf');

coreml.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(1) === 0 && tags.get(2) === 2) {
            return 'coreml.pb';
        }
        const stream = context.stream;
        const identifier = context.identifier.toLowerCase();
        const extension = identifier.split('.').pop().toLowerCase();
        switch (identifier) {
            case 'manifest.json': {
                const obj = context.open('json');
                if (obj && obj.rootModelIdentifier && obj.itemInfoEntries) {
                    const entries = Object.keys(obj.itemInfoEntries).map((key) => obj.itemInfoEntries[key]);
                    if (entries.filter((entry) => entry.path.toLowerCase().endsWith('.mlmodel').length === 1)){
                        return 'coreml.manifest';
                    }
                }
                break;
            }
            case 'metadata.json': {
                const obj = context.open('json');
                if (obj && obj.rootModelIdentifier && obj.itemInfoEntries) {
                    return 'coreml.metadata';
                }
                break;
            }
            case 'featuredescriptions.json': {
                const obj = context.open('json');
                if (obj && (obj.Inputs || obj.Outputs)) {
                    return 'coreml.featuredescriptions';
                }
                break;
            }
        }
        if (extension === 'bin' && stream.length > 16) {
            const buffer = stream.peek(Math.min(256, stream.length));
            for (let i = 0; i < buffer.length - 4; i++) {
                const signature = (buffer[i] | buffer[i + 1] << 8 | buffer[i + 2] << 16 | buffer [i + 3] << 24) >>> 0;
                if (signature === 0xdeadbeef) {
                    return 'coreml.weights';
                }
            }
        }
        return undefined;
    }

    open(context, match) {
        return context.require('./coreml-proto').then(() => {
            return coreml.Metadata.open(context).then((metadata) => {
                const openModel = (stream, context, path, format) => {
                    let model = null;
                    try {
                        coreml.proto = protobuf.get('coreml').CoreML.Specification;
                        const reader = protobuf.BinaryReader.open(stream);
                        model = coreml.proto.Model.decode(reader);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new coreml.Error('File format is not coreml.Model (' + message.replace(/\.$/, '') + ').');
                    }
                    const weightPaths = new Set();
                    const walkProgram = (program) => {
                        for (const entry of Object.entries(program.functions)) {
                            const func = entry[1];
                            for (const entry of Object.entries(func.block_specializations)) {
                                const block = entry[1];
                                for (const operation of block.operations) {
                                    for (const entry of Object.entries(operation.attributes)) {
                                        const value = entry[1];
                                        if (value.blobFileValue && value.blobFileValue.fileName) {
                                            weightPaths.add(value.blobFileValue.fileName);
                                        }
                                    }
                                }
                            }
                        }
                    };
                    const walkModel = (model) => {
                        if (model.mlProgram) {
                            walkProgram(model.mlProgram);
                        }
                        if (model.pipeline && model.pipeline.models) {
                            for (const node of model.pipeline.models) {
                                walkModel(node);
                            }
                        }
                        if (model.pipelineClassifier && model.pipelineClassifier.pipeline && model.pipelineClassifier.pipeline.models) {
                            for (const node of model.pipelineClassifier.pipeline.models) {
                                walkModel(node);
                            }
                        }
                        if (model.pipelineRegressor && model.pipelineRegressor.pipeline && model.pipelineRegressor.pipeline.models) {
                            for (const node of model.pipelineRegressor.pipeline.models) {
                                walkModel(node);
                            }
                        }
                    };
                    walkModel(model);
                    if (weightPaths.size > 0) {
                        const items = path.split('/');
                        items.pop();
                        const folder = items.join('/');
                        const keys = Array.from(weightPaths);
                        const paths = keys.map((path) => {
                            const items = path.split('/');
                            if (items[0] === '@model_path') {
                                items[0] = folder;
                            }
                            return items.join('/');
                        });
                        const promises = paths.map((path) => context.request(path, null));
                        return Promise.all(promises).then((streams) => {
                            const weights = new Map();
                            for (let i = 0; i < keys.length; i++) {
                                weights.set(keys[i], streams[i]);
                            }
                            return new coreml.Model(metadata, format, model, weights);
                        }).catch((/* err */) => {
                            return new coreml.Model(metadata, format, model, new Map());
                        });
                    }
                    return new coreml.Model(metadata, format, model, new Map());
                };
                const openManifest = (obj, context, path) => {
                    const entries = Object.keys(obj.itemInfoEntries).map((key) => obj.itemInfoEntries[key]);
                    const entry = entries.filter((entry) => entry.path.toLowerCase().endsWith('.mlmodel'))[0];
                    const file = path + 'Data/' + entry.path;
                    return context.request(file, null).then((stream) => {
                        return openModel(stream, context, file, 'Core ML Package');
                    });
                };
                const openManifestStream = (context, path) => {
                    return context.request(path + 'Manifest.json', null).then((stream) => {
                        const reader = json.TextReader.open(stream);
                        const obj = reader.read();
                        return openManifest(obj, context, path);
                    });
                };
                switch (match) {
                    case 'coreml.pb': {
                        return openModel(context.stream, context, context.identifier);
                    }
                    case 'coreml.manifest': {
                        const obj = context.open('json');
                        return openManifest(obj, context, '');
                    }
                    case 'coreml.featuredescriptions':
                    case 'coreml.metadata': {
                        return openManifestStream(context, '../../');
                    }
                    case 'coreml.weights': {
                        return openManifestStream(context, '../../../');
                    }
                    default: {
                        throw new coreml.Error("Unknown Core ML format '" + match + "'.");
                    }
                }
            });
        });
    }
};

coreml.Model = class {

    constructor(metadata, format, model, weights) {
        this._format = (format || 'Core ML') + ' v' + model.specificationVersion.toString();
        this._graphs = [ new coreml.Graph(metadata, model, weights) ];
        if (model.description && model.description.metadata) {
            const properties = model.description.metadata;
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
        return this._format;
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

    constructor(metadata, model, weights) {
        this._metadata = metadata;
        this._description = model.description;
        this._groups = false;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        if (this._description) {
            this._inputs = this._description.input.map((input) => {
                const argument = new coreml.Argument(input.name, coreml.Utility.featureType(input.type), input.shortDescription, null);
                return new coreml.Parameter(input.name, true, [ argument ]);
            });

            this._outputs = this._description.output.map((output) => {
                const argument = new coreml.Argument(output.name, coreml.Utility.featureType(output.type), output.shortDescription, null);
                return new coreml.Parameter(output.name, true, [ argument ]);
            });
        }

        this._type = this._loadModel(model, {}, '', weights);
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
        for (const node of this._nodes) {
            for (const output of node.outputs) {
                for (const argument of output.arguments) {
                    if (argument.name === name) {
                        argument.name = newName;
                    }
                }
            }
        }
        return newName;
    }

    _updateClassifierOutput(group, classifier) {
        let labelProbabilityLayerName = classifier.labelProbabilityLayerName;
        if (!labelProbabilityLayerName && this._nodes.length > 0) {
            const node = this._nodes.slice(-1).pop();
            if (node && node.outputs.length == 1 && node.outputs[0].arguments.length == 1) {
                labelProbabilityLayerName = node.outputs[0].arguments[0].name;
            }
        }
        let predictedFeatureName = this._description.predictedFeatureName;
        let predictedProbabilitiesName = this._description.predictedProbabilitiesName;
        if ((predictedFeatureName || predictedProbabilitiesName) && labelProbabilityLayerName && classifier.ClassLabels) {
            predictedFeatureName = predictedFeatureName ? predictedFeatureName : '?';
            predictedProbabilitiesName = predictedProbabilitiesName ? predictedProbabilitiesName : '?';
            const labelProbabilityInput = this._updateOutput(labelProbabilityLayerName, labelProbabilityLayerName + ':labelProbabilityLayerName');
            const type = classifier.ClassLabels;
            const inputs = [
                new coreml.Parameter('input', true, [ new coreml.Argument(labelProbabilityInput) ])
            ];
            const outputs = [
                new coreml.Parameter('probabilities', true, [ new coreml.Argument(predictedProbabilitiesName) ]),
                new coreml.Parameter('feature', true, [ new coreml.Argument(predictedFeatureName) ])
            ];
            const node = new coreml.Node(this._metadata, this._group, type, null, '', classifier[type], inputs, outputs);
            this._nodes.push(node);
        }
    }

    _updatePreprocessing(scope, group, preprocessing) {
        if (preprocessing && preprocessing.length > 0) {
            const preprocessingInput = this._description.input[0].name;
            const inputNodes = [];
            for (const node of this._nodes) {
                if (node.inputs.some((input) => input.arguments.some((arg) => arg.name == preprocessingInput))) {
                    inputNodes.push(node);
                }
            }
            let preprocessorOutput = preprocessingInput;
            let preprocessorIndex = 0;
            for (const p of preprocessing) {
                const input = p.featureName ? p.featureName : preprocessorOutput;
                preprocessorOutput = preprocessingInput + ':' + preprocessorIndex.toString();
                this._createNode(scope, group, p.preprocessor, null, '', p[p.preprocessor], [ input ], [ preprocessorOutput ]);
                preprocessorIndex++;
            }
            for (const node of inputNodes) {
                for (const input of node.inputs) {
                    for (const arg of input.arguments) {
                        if (arg.name === preprocessingInput) {
                            arg.name = preprocessorOutput;
                        }
                    }
                }
            }
        }
    }

    _loadModel(model, scope, group, weights) {
        this._groups = this._groups | (group.length > 0 ? true : false);
        const description = model && model.description && model.description.metadata && model.description.metadata.shortDescription ? model.description.metadata.shortDescription : '';
        switch (model.Type) {
            case 'neuralNetworkClassifier': {
                const neuralNetworkClassifier = model.neuralNetworkClassifier;
                for (const layer of neuralNetworkClassifier.layers) {
                    this._createNode(scope, group, layer.layer, layer.name, description, layer[layer.layer], layer.input, layer.output);
                }
                this._updateClassifierOutput(group, neuralNetworkClassifier);
                this._updatePreprocessing(scope, group, neuralNetworkClassifier.preprocessing);
                return 'Neural Network Classifier';
            }
            case 'neuralNetwork': {
                const neuralNetwork = model.neuralNetwork;
                for (const layer of neuralNetwork.layers) {
                    this._createNode(scope, group, layer.layer, layer.name, description, layer[layer.layer], layer.input, layer.output);
                }
                this._updatePreprocessing(scope, group, neuralNetwork.preprocessing);
                return 'Neural Network';
            }
            case 'neuralNetworkRegressor': {
                const neuralNetworkRegressor = model.neuralNetworkRegressor;
                for (const layer of neuralNetworkRegressor.layers) {
                    this._createNode(scope, group, layer.layer, layer.name, description, layer[layer.layer], layer.input, layer.output);
                }
                this._updatePreprocessing(scope, group, neuralNetworkRegressor);
                return 'Neural Network Regressor';
            }
            case 'pipeline': {
                for (let i = 0; i < model.pipeline.models.length; i++) {
                    this._loadModel(model.pipeline.models[i], scope, (group ? (group + '/') : '') + 'pipeline[' + i.toString() + ']');
                }
                return 'Pipeline';
            }
            case 'pipelineClassifier': {
                for (let i = 0; i < model.pipelineClassifier.pipeline.models.length; i++) {
                    this._loadModel(model.pipelineClassifier.pipeline.models[i], scope, (group ? (group + '/') : '') + 'pipelineClassifier[' + i.toString() + ']');
                }
                return 'Pipeline Classifier';
            }
            case 'pipelineRegressor': {
                for (let i = 0; i < model.pipelineRegressor.pipeline.models.length; i++) {
                    this._loadModel(model.pipelineRegressor.pipeline.models[i], scope, (group ? (group + '/') : '') + 'pipelineRegressor[' + i.toString() + ']');
                }
                return 'Pipeline Regressor';
            }
            case 'glmClassifier': {
                this._createNode(scope, group, 'glmClassifier', null, description,
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
            case 'glmRegressor': {
                this._createNode(scope, group, 'glmRegressor', null, description,
                    model.glmRegressor,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Generalized Linear Regressor';
            }
            case 'dictVectorizer': {
                this._createNode(scope, group, 'dictVectorizer', null, description,
                    model.dictVectorizer,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Dictionary Vectorizer';
            }
            case 'featureVectorizer': {
                this._createNode(scope, group, 'featureVectorizer', null, description,
                    model.featureVectorizer,
                    coreml.Graph._formatFeatureDescriptionList(model.description.input),
                    [ model.description.output[0].name ]);
                return 'Feature Vectorizer';
            }
            case 'treeEnsembleClassifier': {
                this._createNode(scope, group, 'treeEnsembleClassifier', null, description,
                    model.treeEnsembleClassifier.treeEnsemble,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                this._updateClassifierOutput(group, model.treeEnsembleClassifier);
                return 'Tree Ensemble Classifier';
            }
            case 'treeEnsembleRegressor': {
                this._createNode(scope, group, 'treeEnsembleRegressor', null, description,
                    model.treeEnsembleRegressor.treeEnsemble,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Tree Ensemble Regressor';
            }
            case 'supportVectorClassifier': {
                this._createNode(scope, group, 'supportVectorClassifier', null, description,
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
            case 'supportVectorRegressor': {
                this._createNode(scope, group, 'supportVectorRegressor', null, description,
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
            case 'arrayFeatureExtractor': {
                this._createNode(scope, group, 'arrayFeatureExtractor', null, description,
                    { extractIndex: model.arrayFeatureExtractor.extractIndex },
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Array Feature Extractor';
            }
            case 'oneHotEncoder': {
                const categoryType = model.oneHotEncoder.CategoryType;
                const oneHotEncoderParams = { outputSparse: model.oneHotEncoder.outputSparse };
                oneHotEncoderParams[categoryType] = model.oneHotEncoder[categoryType];
                this._createNode(scope, group, 'oneHotEncoder', null, description,
                    oneHotEncoderParams,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'One Hot Encoder';
            }
            case 'imputer': {
                const imputedValue = model.imputer.ImputedValue;
                const replaceValue = model.imputer.ReplaceValue;
                const imputerParams = {};
                imputerParams[imputedValue] = model.imputer[imputedValue];
                imputerParams[replaceValue] = model.imputer[replaceValue];
                this._createNode(scope, group, 'oneHotEncoder', null, description,
                    imputerParams,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Imputer';
            }
            case 'normalizer': {
                this._createNode(scope, group, 'normalizer', null, description,
                    model.normalizer,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Normalizer';
            }
            case 'wordTagger': {
                this._createNode(scope, group, 'wordTagger', null, description,
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
            case 'textClassifier': {
                this._createNode(scope, group, 'textClassifier', null, description,
                    model.textClassifier,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Text Classifier';
            }
            case 'nonMaximumSuppression': {
                const nonMaximumSuppressionParams = {
                    pickTop: model.nonMaximumSuppression.pickTop,
                    stringClassLabels: model.nonMaximumSuppression.stringClassLabels,
                    iouThreshold: model.nonMaximumSuppression.iouThreshold,
                    confidenceThreshold: model.nonMaximumSuppression.confidenceThreshold
                };
                this._createNode(scope, group, 'nonMaximumSuppression', null, description,
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
            case 'visionFeaturePrint': {
                const visionFeaturePrintParams = {
                    scene: model.visionFeaturePrint.scene
                };
                this._createNode(scope, group, 'visionFeaturePrint', null, description,
                    visionFeaturePrintParams,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Vision Feature Print';
            }
            case 'soundAnalysisPreprocessing': {
                this._createNode(scope, group, 'soundAnalysisPreprocessing', null, description,
                    model.soundAnalysisPreprocessing,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Sound Analysis Preprocessing';
            }
            case 'kNearestNeighborsClassifier': {
                this._createNode(scope, group, 'kNearestNeighborsClassifier', null, description,
                    model.kNearestNeighborsClassifier,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                this._updateClassifierOutput(group, model.kNearestNeighborsClassifier);
                return 'Nearest Neighbors Classifier';
            }
            case 'itemSimilarityRecommender': {
                this._createNode(scope, group, 'itemSimilarityRecommender', null, description,
                    {
                        itemStringIds: model.itemSimilarityRecommender.itemStringIds.vector,
                        itemItemSimilarities: model.itemSimilarityRecommender.itemItemSimilarities
                    },
                    model.description.input.map((feature) => feature.name),
                    model.description.output.map((feature) => feature.name));
                return 'Item Similarity Recommender';
            }
            case 'linkedModel': {
                this._createNode(scope, group, 'linkedModel', null, description,
                    model.linkedModel.linkedModelFile,
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'Linked Model';
            }
            case 'customModel': {
                this._createNode(scope, group, 'customModel', null, description,
                    { className: model.customModel.className, parameters: model.customModel.parameters },
                    [ model.description.input[0].name ],
                    [ model.description.output[0].name ]);
                return 'customModel';
            }
            case 'mlProgram': {
                return this._loadProgram(model.mlProgram, scope, group, weights);
            }
        }
        throw new coreml.Error("Unknown model type '" + JSON.stringify(Object.keys(model)) + "'.");
    }

    _loadProgram(program, scope, group, weights) {
        // TODO: need to handle functions other than main?
        const main = program.functions.main;
        // TODO: need to handle more than one block specialization?
        const block = main.block_specializations.CoreML5;

        const convertValue = (value) => {
            switch (value.value) {
                case 'immediateValue': {
                    const tensor = value.immediateValue.tensor;
                    let values = null;
                    switch (tensor.value) {
                        case 'ints':
                            values = tensor.ints.values;
                            break;
                        case 'strings':
                            values = tensor.strings.values;
                            break;
                        case 'bools':
                            values = tensor.bools.values;
                            break;
                        case 'floats':
                            values = tensor.floats.values;
                            break;
                        case 'bytes':
                            values = tensor.bytes.values;
                            break;
                        default:
                            throw new coreml.Error("Unsupported tensor value '" + tensor.value + "'.");
                    }
                    return values;
                }
                case 'blobFileValue': {
                    const type = coreml.Utility.valueType(value.type);
                    const blob = value.blobFileValue;
                    const offset = blob.offset.toNumber();
                    const file = blob.fileName;
                    let data = null;
                    const stream = weights.get(file);
                    if (stream) {
                        stream.seek(offset);
                        const buffer = stream.read(32);
                        const reader = new coreml.BinaryReader(buffer);
                        const signature = reader.uint32();
                        if (signature == 0xdeadbeef) {
                            reader.uint32(); // dataType
                            const size = reader.uint64();
                            stream.seek(reader.uint64());
                            const length = (type.shape.dimensions || []).reduce((a, b) => a * b, 1);
                            switch (type.dataType) {
                                case 'float32': {
                                    const buffer = stream.read(size);
                                    data = new Float32Array(buffer.buffer, buffer.byteOffset, length).slice();
                                    break;
                                }
                                case 'float16': {
                                    data = stream.read(size);
                                    break;
                                }
                                default:
                                    throw new coreml.Error("Unsupported blob data type '" + type.dataType + "'.");
                            }
                        }
                    }
                    return new coreml.Tensor('Blob', type, data);
                }
            }
            throw new coreml.Error("Unsupported value '" + value.value + "'.");
        };

        const args = new Map();
        const arg = (name) => {
            if (!args.has(name)) {
                args.set(name, { name: name, to: [], from: [] });
            }
            return args.get(name);
        };

        const operations = block.operations.map((op) => {
            const operation = {
                type: op.type,
                attributes: {}
            };
            for (const entry of Object.entries(op.attributes)) {
                const key = entry[0];
                const value = entry[1];
                operation.attributes[key] = convertValue(value);
            }
            operation.inputs = Object.entries(op.inputs).map((entry) => {
                const key = entry[0];
                const input = entry[1];
                const args = input.arguments.map((argument) => {
                    if (argument.name) {
                        const value = arg(argument.name);
                        value.to.push(operation);
                        return value;
                    }
                    return { value: argument.value };
                });
                return {
                    name: key,
                    arguments: args
                };
            });
            operation.outputs = op.outputs.map((output) => {
                const value = arg(output.name);
                value.type = coreml.Utility.valueType(output.type);
                value.from.push(operation);
                return {
                    name: 'output',
                    arguments: [ value ]
                };
            });
            return operation;
        });

        for (const op of operations) {
            if (op.type === 'const' && op.inputs.length === 0 &&
                op.outputs.length === 1 && op.outputs[0].arguments.length === 1) {
                const argument = op.outputs[0].arguments[0];
                if (op.attributes && op.attributes.val) {
                    const type = argument.type;
                    const data = op.attributes.val;
                    if (data instanceof Uint8Array && data.length === 2 &&
                        type.dataType === 'float16' && type.shape.dimensions.length === 0) {
                        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                        argument.value = view.getFloat16(0, true);
                    }
                    else {
                        argument.value = data;
                    }
                    op.delete = true;
                }
            }
        }

        for (const op of operations) {
            if (op.delete) {
                continue;
            }
            op.inputs = op.inputs.filter((input) => {
                if (input.arguments.length !== 1) {
                    return true;
                }
                const argument = input.arguments[0];
                if (argument.value === undefined || argument.value instanceof coreml.Tensor) {
                    return true;
                }
                op.attributes[input.name] = argument.value;
                return false;
            });
        }

        const tensors = new Map();
        const tensor = (arg) => {
            if (!tensors.has(arg.name)) {
                tensors.set(arg.name, new coreml.Argument(arg.name, arg.type, null, arg.value));
            }
            return tensors.get(arg.name);
        };

        for (const op of operations) {
            if (op.delete) {
                continue;
            }
            op.inputs = op.inputs.map((input) => new coreml.Parameter(input.name, true, input.arguments.map((argument) => tensor(argument))));
            op.outputs = op.outputs.map((output) => new coreml.Parameter(output.name, true, output.arguments.map((argument) => tensor(argument))));
        }

        for (const op of operations.filter((op) => !op.delete)) {
            const type = 'program:' + op.type;
            const metadata = this._metadata.type(type);
            if (metadata && Array.isArray(metadata.inputs)) {
                let index = 1;
                const map = new Map(metadata.inputs.map((input) => [ input.name, index++ ]));
                op.inputs.sort((a, b) => (map.get(a.name) || map.size) - (map.get(b.name) || map.size));
            }
            const node = new coreml.Node(this._metadata, group, type, null, null, op.attributes, op.inputs, op.outputs);
            this._nodes.push(node);
        }

        return 'ML Program';
    }

    _createNode(scope, group, type, name, description, data, inputs, outputs, outputTypes) {
        inputs = inputs.map((input) => scope[input] ? scope[input].argument : input);
        outputs = outputs.map((output) => {
            if (scope[output]) {
                scope[output].counter++;
                const next = output + '\n' + scope[output].counter.toString(); // custom argument id
                scope[output].argument = next;
                return next;
            }
            scope[output] = {
                argument: output,
                counter: 0
            };
            return output;
        });

        const initializers = [];
        const attributes = {};
        if (data) {
            const map = this._initialize(type, data, initializers);
            for (const key of Object.keys(data)) {
                if (map[key]) {
                    continue;
                }
                attributes[key] = data[key];
            }
        }
        const inputParameters = this._metadata.getInputs(type, inputs).map((input) => {
            return new coreml.Parameter(input.name, true, input.arguments.map((argument) => {
                return new coreml.Argument(argument.name, argument.type, null, null);
            }));
        });
        inputParameters.push(...initializers);
        const outputParameters = outputs.map((output, index) => {
            const name = this._metadata.getOutputName(type, index);
            const outputType = outputTypes ? outputTypes[index] : null;
            return new coreml.Parameter(name, true, [ new coreml.Argument(output, outputType, null, null) ]);
        });

        const node = new coreml.Node(this._metadata, group, type, name, description, attributes, inputParameters, outputParameters);
        this._nodes.push(node);
        return node;
    }

    _initializer(type, initializers, kind, name, shape, data) {
        let dataType = '?';
        let quantization = null;
        let values = null;
        if (data) {
            if (data.floatValue && data.floatValue.length > 0) {
                values = data.floatValue;
                dataType = 'float32';
            }
            else if (data.float16Value && data.float16Value.length > 0) {
                values = data.float16Value; // byte[]
                dataType = 'float16';
            }
            else if (data.rawValue && data.rawValue.length > 0) {
                if (data.quantization) {
                    values = data.rawValue;
                    dataType = 'uint' + data.quantization.numberOfBits.toString();
                }
                else {
                    shape = [];
                }
            }
            quantization = data.quantization || null;
        }
        const tensorType = new coreml.TensorType(dataType, new coreml.TensorShape(shape));
        const tensor = new coreml.Tensor(kind, tensorType, values, quantization);
        const argument = new coreml.Argument('', null, null, tensor);
        const visible = this._metadata.visible(type, name);
        initializers.push(new coreml.Parameter(name, visible, [ argument ]));
    }

    _initialize(type, data, initializers) {
        switch (type) {
            case 'convolution': {
                const weightsShape = [ data.outputChannels, data.kernelChannels, data.kernelSize[0], data.kernelSize[1] ];
                if (data.isDeconvolution) {
                    weightsShape[0] = data.kernelChannels;
                    weightsShape[1] = Math.floor(data.outputChannels / (data.nGroups != 0 ? data.nGroups : 1));
                }
                this._initializer(type, initializers, 'Weights', 'weights', weightsShape, data.weights);
                if (data.hasBias) {
                    this._initializer(type, initializers, 'Weights', 'bias', [ data.outputChannels ], data.bias);
                }
                return { 'weights': true, 'bias': data.hasBias };
            }
            case 'innerProduct':
                this._initializer(type, initializers, 'Weights', 'weights', [ data.outputChannels, data.inputChannels ], data.weights);
                if (data.hasBias) {
                    this._initializer(type, initializers, 'Weights', 'bias', [ data.outputChannels ], data.bias);
                }
                return { 'weights': true, 'bias': data.hasBias };
            case 'batchnorm':
                this._initializer(type, initializers, 'Weights', 'gamma', [ data.channels ], data.gamma);
                this._initializer(type, initializers, 'Weights', 'beta', [ data.channels ], data.beta);
                if (data.mean) {
                    this._initializer(type, initializers, 'Weights', 'mean', [ data.channels ], data.mean);
                }
                if (data.variance) {
                    this._initializer(type, initializers, 'Weights', 'variance', [ data.channels ], data.variance);
                }
                return { 'gamma': true, 'beta': true, 'mean': true, 'variance': true };
            case 'embedding':
                this._initializer(type, initializers, 'Weights', 'weights', [ data.inputDim, data.outputChannels ], data.weights);
                return { 'weights': true };
            case 'loadConstant':
            case 'loadConstantND':
                this._initializer(type, initializers, 'Weights', 'data', data.shape, data.data);
                return { 'data': true };
            case 'scale':
                this._initializer(type, initializers, 'Weights', 'scale', data.shapeScale, data.scale);
                if (data.hasBias) {
                    this._initializer(type, initializers, 'Weights', 'bias', data.shapeBias, data.bias);
                }
                return { 'scale': true, 'bias': data.hasBias };
            case 'bias':
                this._initializer(type, initializers, 'Weights', 'bias', data.shape, data.bias);
                return { 'bias': true };
            case 'simpleRecurrent':
                this._initializer(type, initializers, 'Weights', 'weights', [ data.outputVectorSize, data.inputVectorSize ], data.weightMatrix);
                this._initializer(type, initializers, 'Weights', 'recurrent', [ data.outputVectorSize, data.inputVectorSize ], data.recursionMatrix);
                if (data.hasBiasVectors) {
                    this._initializer(type, initializers, 'Weights', 'bias', [ data.outputVectorSize ], data.biasVector);
                }
                return { 'weightMatrix': true, 'recursionMatrix': true, 'biasVector': data.hasBiasVectors };
            case 'gru': {
                const recursionMatrixShape = [ data.outputVectorSize, data.outputVectorSize ];
                const weightMatrixShape = [ data.outputVectorSize, data.inputVectorSize ];
                const biasVectorShape = [ data.outputVectorSize ];
                this._initializer(type, initializers, 'Weights', 'updateGateWeightMatrix', weightMatrixShape, data.updateGateWeightMatrix);
                this._initializer(type, initializers, 'Weights', 'resetGateWeightMatrix', weightMatrixShape, data.resetGateWeightMatrix);
                this._initializer(type, initializers, 'Weights', 'outputGateWeightMatrix', weightMatrixShape, data.outputGateWeightMatrix);
                this._initializer(type, initializers, 'Weights', 'updateGateRecursionMatrix', recursionMatrixShape, data.updateGateRecursionMatrix);
                this._initializer(type, initializers, 'Weights', 'resetGateRecursionMatrix', recursionMatrixShape, data.resetGateRecursionMatrix);
                this._initializer(type, initializers, 'Weights', 'outputGateRecursionMatrix', recursionMatrixShape, data.outputGateRecursionMatrix);
                if (data.hasBiasVectors) {
                    this._initializer(type, initializers, 'Weights', 'updateGateBiasVector', biasVectorShape, data.updateGateBiasVector);
                    this._initializer(type, initializers, 'Weights', 'resetGateBiasVector', biasVectorShape, data.resetGateBiasVector);
                    this._initializer(type, initializers, 'Weights', 'outputGateBiasVector', biasVectorShape, data.outputGateBiasVector);
                }
                return {
                    'updateGateWeightMatrix': true, 'resetGateWeightMatrix': true, 'outputGateWeightMatrix': true,
                    'updateGateRecursionMatrix': true, 'resetGateRecursionMatrix': true, 'outputGateRecursionMatrix': true,
                    'updateGateBiasVector': data.hasBiasVectors, 'resetGateBiasVector': data.hasBiasVectors, 'outputGateBiasVector': data.hasBiasVectors
                };
            }
            case 'uniDirectionalLSTM':
            case 'biDirectionalLSTM': {
                const count = (type == 'uniDirectionalLSTM') ? 1 : 2;
                const matrixShape = [ data.outputVectorSize, data.inputVectorSize ];
                const vectorShape = [ data.outputVectorSize ];
                for (let i = 0; i < count; i++) {
                    const weights = count == 1 ? data.weightParams : data.weightParams[i];
                    const suffix = (i == 0) ? '' : '_rev';
                    this._initializer(type, initializers, 'Weights', 'inputGateWeightMatrix' + suffix, matrixShape, weights.inputGateWeightMatrix);
                    this._initializer(type, initializers, 'Weights', 'forgetGateWeightMatrix' + suffix, matrixShape, weights.forgetGateWeightMatrix);
                    this._initializer(type, initializers, 'Weights', 'blockInputWeightMatrix' + suffix, matrixShape, weights.blockInputWeightMatrix);
                    this._initializer(type, initializers, 'Weights', 'outputGateWeightMatrix' + suffix, matrixShape, weights.outputGateWeightMatrix);
                    this._initializer(type, initializers, 'Weights', 'inputGateRecursionMatrix' + suffix, matrixShape, weights.inputGateRecursionMatrix);
                    this._initializer(type, initializers, 'Weights', 'forgetGateRecursionMatrix' + suffix, matrixShape,weights.forgetGateRecursionMatrix);
                    this._initializer(type, initializers, 'Weights', 'blockInputRecursionMatrix' + suffix, matrixShape, weights.blockInputRecursionMatrix);
                    this._initializer(type, initializers, 'Weights', 'outputGateRecursionMatrix' + suffix, matrixShape, weights.outputGateRecursionMatrix);
                    if (data.params.hasBiasVectors) {
                        this._initializer(type, initializers, 'Weights', 'inputGateBiasVector' + suffix, vectorShape, weights.inputGateBiasVector);
                        this._initializer(type, initializers, 'Weights', 'forgetGateBiasVector' + suffix, vectorShape, weights.forgetGateBiasVector);
                        this._initializer(type, initializers, 'Weights', 'blockInputBiasVector' + suffix, vectorShape, weights.blockInputBiasVector);
                        this._initializer(type, initializers, 'Weights', 'outputGateBiasVector' + suffix, vectorShape, weights.outputGateBiasVector);
                    }
                    if (data.params.hasPeepholeVectors) {
                        this._initializer(type, initializers, 'Weights', 'inputGatePeepholeVector' + suffix, vectorShape, weights.inputGatePeepholeVector);
                        this._initializer(type, initializers, 'Weights', 'forgetGatePeepholeVector' + suffix, vectorShape, weights.forgetGatePeepholeVector);
                        this._initializer(type, initializers, 'Weights', 'outputGatePeepholeVector' + suffix, vectorShape, weights.outputGatePeepholeVector);
                    }
                }
                return { 'weightParams': true };
            }
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

    constructor(name, type, description, initializer) {
        if (typeof name !== 'string') {
            throw new coreml.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._description = description || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    set name(value) {
        this._name = value;
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

    constructor(metadata, group, type, name, description, attributes, inputs, outputs) {
        if (!type) {
            throw new Error('Undefined node type.');
        }
        if (group) {
            this._group = group;
        }
        this._type = Object.assign({}, metadata.type(type) || { name: type });
        this._type.name = type.split(':').pop();
        this._name = name || '';
        this._description = description || '';
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        if (attributes) {
            for (const key of Object.keys(attributes)) {
                const schema = metadata.attribute(type, key);
                const value = attributes[key];
                const attribute = new coreml.Attribute(schema, key, value);
                this._attributes.push(attribute);
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get metadata() {
        return this._metadata;
    }

    get group() {
        return this._group ? this._group : null;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }
};

coreml.Attribute = class {

    constructor(schema, name, value) {
        this._name = name;
        this._value = value;
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
            }
            if (this._type && coreml.proto) {
                this._value = coreml.Utility.enum(this._type, this._value);
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (Array.isArray(value)) {
                    value = value.map((item) => item.toNumber());
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

    constructor(kind, type, data, quantization) {
        this._kind = kind;
        this._type = type;
        this._data = data;
        this._quantization = quantization;
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
                const map = [];
                for (const key of Object.keys(this._quantization.lookupTableQuantization.floatValue)) {
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
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
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
                    context.bits = this._quantization.numberOfBits.toNumber();
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
        const results = [];
        const size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
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
            for (let j = 0; j < size; j++) {
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

coreml.ListType = class {

    constructor(elementType) {
        this._elementType = elementType;
    }

    toString() {
        return 'list<' + this._elementType.toString() + '>';
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

coreml.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    skip(offset) {
        const position = this._position;
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        return position;
    }

    uint32() {
        const position = this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    uint64() {
        const position = this.skip(8);
        return this._dataView.getUint64(position, true).toNumber();
    }
};

coreml.Utility = class {

    static enum(name, value) {
        let type = coreml.proto;
        const parts = name.split('.');
        while (type && parts.length > 0) {
            type = type[parts.shift()];
        }
        if (type) {
            coreml.Utility._enumKeyMap = coreml.Utility._enumKeyMap || new Map();
            if (!coreml.Utility._enumKeyMap.has(name)) {
                const map = new Map(Object.entries(type).map((pair) => [ pair[1], pair[0] ]));
                coreml.Utility._enumKeyMap.set(name, map);
            }
            const map = coreml.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }

    static featureType(type) {
        let result = '?';
        if (type) {
            switch (type.Type) {
                case 'multiArrayType': {
                    let shape = new coreml.TensorShape([]);
                    if (type.multiArrayType.shape && type.multiArrayType.shape.length > 0) {
                        shape = new coreml.TensorShape(type.multiArrayType.shape);
                    }
                    let dataType = '?';
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
                }
                case 'stringType': {
                    result = new coreml.TensorType('string');
                    break;
                }
                case 'doubleType': {
                    result = new coreml.TensorType('float64');
                    break;
                }
                case 'int64Type': {
                    result = new coreml.TensorType('int64');
                    break;
                }
                case 'dictionaryType': {
                    result = new coreml.MapType(type.dictionaryType.KeyType.replace('KeyType', ''), 'float64');
                    break;
                }
                case 'imageType': {
                    result = new coreml.ImageType(type.imageType.colorSpace, type.imageType.width, type.imageType.height);
                    break;
                }
            }
            if (type.isOptional) {
                result = new coreml.OptionalType(result);
            }
        }
        return result;
    }

    static tensorType(type) {
        if (!coreml.Utility._dataTypes) {
            coreml.Utility._dataTypes = new Map();
            const DataType = coreml.proto.MILSpec.DataType;
            for (const pair of Object.entries(DataType)) {
                if (pair[0] === 'UNUSED_TYPE') {
                    continue;
                }
                const name = pair[0] === 'bool' ? 'boolean' : pair[0].toLowerCase();
                coreml.Utility._dataTypes.set(pair[1], name);
            }
        }
        const shape = (type.dimensions.map(dim => dim.constant ? dim.constant.size : '?'));
        const dataType = coreml.Utility._dataTypes.get(type.dataType);
        if (dataType === null) {
            throw new coreml.Error("Unsupported data type '" + type.dataType + "'.");
        }
        return new coreml.TensorType(dataType, new coreml.TensorShape(shape));
    }

    static valueType(type) {
        switch (type.type) {
            case 'tensorType':
                return coreml.Utility.tensorType(type.tensorType);
            case 'listType':
                return new coreml.ListType(coreml.Utility.tensorType(type.listType.tensorType));
            default:
                throw new coreml.Error("Unsupported value type '" + type.type + "'.");
        }
    }
};

coreml.Metadata = class {

    static open(context) {
        if (coreml.Metadata._metadata) {
            return Promise.resolve(coreml.Metadata._metadata);
        }
        return context.request('coreml-metadata.json', 'utf-8', null).then((data) => {
            coreml.Metadata._metadata = new coreml.Metadata(data);
            return coreml.Metadata._metadata;
        }).catch(() => {
            coreml.Metadata._metadata = new coreml.Metadata(null);
            return coreml.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        this._inputCache = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributeCache.has(key)) {
            this._attributeCache.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.attributes) && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributeCache.get(key);
    }

    visible(type, name) {
        const key = type + ':' + name;
        if (!this._inputCache.has(key)) {
            this._inputCache.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.inputs) && metadata.inputs.length > 0) {
                for (const input of metadata.inputs) {
                    this._inputCache.set(type + ':' + input.name, input);
                }
            }
        }
        const input = this._inputCache.get(key);
        if (input) {
            return input.visible === false ? false : true;
        }
        return true;
    }

    getInputs(type, inputs) {
        const results = [];
        const schema = this._map.get(type);
        let index = 0;
        while (index < inputs.length) {
            const result = { arguments: [] };
            let count = 1;
            let name = null;
            if (schema && schema.inputs) {
                if (index < schema.inputs.length) {
                    const input = schema.inputs[index];
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
            const array = inputs.slice(index, index + count);
            for (let j = 0; j < array.length; j++) {
                result.arguments.push({ name: array[j] });
            }
            index += count;
            results.push(result);
        }
        return results;
    }

    getOutputName(type, index) {
        const schema = this._map.get(type);
        if (schema) {
            const outputs = schema.outputs;
            if (outputs && index < outputs.length) {
                const output = outputs[index];
                if (output) {
                    const name = output.name;
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