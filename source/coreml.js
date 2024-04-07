
import * as base from './base.js';

const coreml = {};

coreml.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const identifier = context.identifier.toLowerCase();
        const extension = identifier.split('.').pop().toLowerCase();
        const tags = context.tags('pb');
        if (tags.get(1) === 0 && tags.get(2) === 2) {
            const match = (key) =>
                (key >= 200 && key < 220) || (key >= 300 && key < 320) || (key >= 400 && key < 420) ||
                (key >= 500 && key < 520) || (key >= 550 && key < 560) || (key >= 600 && key < 620) ||
                (key === 900) ||
                (key >= 2000 && key < 2010) || (key === 3000);
            if (extension === 'pb' && Array.from(tags.keys()).every((key) => !match(key))) {
                return;
            }
            context.type = 'coreml.pb';
            return;
        }
        if (extension === 'pbtxt') {
            const tags = context.tags('pbtxt');
            if (tags.has('specificationVersion') && tags.has('description')) {
                context.type = 'coreml.pbtxt';
                return;
            }
        }
        if (identifier === 'manifest.json') {
            const obj = context.peek('json');
            if (obj && obj.rootModelIdentifier && obj.itemInfoEntries) {
                const entries = Object.keys(obj.itemInfoEntries).map((key) => obj.itemInfoEntries[key]);
                if (entries.filter((entry) => entry.path.toLowerCase().endsWith('.mlmodel').length === 1)) {
                    context.type = 'coreml.manifest';
                    return;
                }
            }
        }
        if (identifier === 'metadata.json') {
            const obj = context.peek('json');
            if (obj && obj.rootModelIdentifier && obj.itemInfoEntries) {
                context.type = 'coreml.metadata';
                return;
            }
        }
        if (identifier === 'featuredescriptions.json') {
            const obj = context.peek('json');
            if (obj && (obj.Inputs || obj.Outputs)) {
                context.type = 'coreml.featuredescriptions';
                return;
            }
        }
        if (extension === 'bin' && stream.length > 16) {
            const buffer = stream.peek(Math.min(256, stream.length));
            for (let i = 0; i < buffer.length - 4; i++) {
                const signature = (buffer[i] | buffer[i + 1] << 8 | buffer[i + 2] << 16 | buffer [i + 3] << 24) >>> 0;
                if (signature === 0xdeadbeef) {
                    context.type = 'coreml.weights';
                    return;
                }
            }
        }
    }

    async open(context) {
        coreml.proto = await context.require('./coreml-proto');
        coreml.proto = coreml.proto.CoreML.Specification;
        const metadata = await context.metadata('coreml-metadata.json');
        const openBinary = async (content, context, path, format) => {
            let model = null;
            try {
                const reader = content.read('protobuf.binary');
                model = coreml.proto.Model.decode(reader);
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new coreml.Error(`File format is not coreml.Model (${message.replace(/\.$/, '')}).`);
            }
            const weightPaths = new Set();
            const walkProgram = (program) => {
                for (const func of Object.values(program.functions)) {
                    for (const block of Object.values(func.block_specializations)) {
                        for (const operation of block.operations) {
                            for (const value of Object.values(operation.attributes)) {
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
            const weights = new Map();
            if (weightPaths.size > 0) {
                const folder = path.replace(/\/[^/]*$/, '');
                const keys = Array.from(weightPaths);
                const paths = keys.map((path) => path.replace(/^@model_path\//, `${folder}/`));
                try {
                    const contexts = await Promise.all(paths.map((path) => context.fetch(path)));
                    for (let i = 0; i < keys.length; i++) {
                        weights.set(keys[i], contexts[i].stream);
                    }
                } catch {
                    // continue regardless of error
                }
            }
            return new coreml.Model(metadata, format, model, weights);
        };
        const openText = async (context) => {
            let model = null;
            try {
                const reader = context.read('protobuf.text');
                model = coreml.proto.Model.decodeText(reader);
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new coreml.Error(`File format is not coreml.Model (${message.replace(/\.$/, '')}).`);
            }
            const weights = new Map();
            return new coreml.Model(metadata, null, model, weights);
        };
        const openManifest = async (obj, context, path) => {
            const entries = Object.values(obj.itemInfoEntries).filter((entry) => entry.path.toLowerCase().endsWith('.mlmodel'));
            if (entries.length !== 1) {
                throw new coreml.Error('Manifest does not contain Core ML model.');
            }
            const name = `${path}Data/${entries[0].path}`;
            const content = await context.fetch(name);
            return openBinary(content, context, name, 'Core ML Package');
        };
        const openManifestStream = async (context, path) => {
            const name = `${path}Manifest.json`;
            const content = await context.fetch(name);
            const obj = content.read('json');
            return openManifest(obj, context, path);
        };
        switch (context.type) {
            case 'coreml.pb': {
                return openBinary(context, context, context.identifier);
            }
            case 'coreml.pbtxt': {
                return openText(context, context, context.identifier);
            }
            case 'coreml.manifest': {
                const obj = context.peek('json');
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
                throw new coreml.Error(`Unsupported Core ML format '${context.type}'.`);
            }
        }
    }
};

coreml.Model = class {

    constructor(metadata, format, model, weights) {
        this.format = `${format || 'Core ML'} v${model.specificationVersion}`;
        this.metadata = [];
        const context = new coreml.Context(metadata, model, weights);
        const graph = new coreml.Graph(context);
        this.graphs = [graph];
        if (model.description && model.description.metadata) {
            const properties = model.description.metadata;
            if (properties.versionString) {
                this.version = properties.versionString;
            }
            if (properties.shortDescription) {
                this.description = properties.shortDescription;
            }
            if (properties.author) {
                this.metadata.push(new coreml.Argument('author', properties.author));
            }
            if (properties.license) {
                this.metadata.push(new coreml.Argument('license', properties.license));
            }
            if (metadata.userDefined && Object.keys(properties.userDefined).length > 0) {
                /* empty */
            }
        }
    }
};

coreml.Graph = class {

    constructor(context) {
        this.name = '';
        this.type = context.type;
        this.groups = context.groups;
        for (const value of context.values.values()) {
            const name = value.name;
            const type = value.type;
            const description = value.description;
            const initializer = value.initializer;
            if (!value.obj) {
                value.obj = new coreml.Value(name, type, description, initializer);
            }
        }
        this.inputs = context.inputs.map((argument) => {
            const values = argument.value.map((value) => value.obj);
            return new coreml.Argument(argument.name, values, argument.visible);
        });
        this.outputs = context.outputs.map((argument) => {
            const values = argument.value.map((value) => value.obj);
            return new coreml.Argument(argument.name, values, argument.visible);
        });
        for (const obj of context.nodes) {
            const attributes = obj.attributes;
            switch (obj.type) {
                case 'loop':
                    attributes.conditionNetwork = new coreml.Graph(attributes.conditionNetwork);
                    attributes.bodyNetwork = new coreml.Graph(attributes.bodyNetwork);
                    break;
                case 'branch':
                    attributes.ifBranch = new coreml.Graph(attributes.ifBranch);
                    attributes.elseBranch = new coreml.Graph(attributes.elseBranch);
                    break;
                default:
                    break;
            }
        }
        this.nodes = context.nodes.map((obj) => new coreml.Node(context, obj));
    }
};

coreml.Argument = class {

    constructor(name, value, visible) {
        this.name = name;
        this.value = value;
        this.visible = visible !== false;
    }
};

coreml.Value = class {

    constructor(name, type, description, initializer) {
        if (typeof name !== 'string') {
            throw new coreml.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.description = description || null;
        this.initializer = initializer || null;
        this.quantization = initializer ? initializer.quantization : null;
    }
};

coreml.Node = class {

    constructor(context, obj) {
        if (!obj.type) {
            throw new Error('Undefined node type.');
        }
        if (obj.group) {
            this.group = obj.group || null;
        }
        const type = context.metadata.type(obj.type);
        this.type = type ? { ...type } : { name: obj.type };
        this.type.name = obj.type.split(':').pop();
        this.name = obj.name || '';
        this.description = obj.description || '';
        this.inputs = (obj.inputs || []).map((argument) => {
            const values = argument.value.map((value) => value.obj);
            return new coreml.Argument(argument.name, values, argument.visible);
        });
        this.outputs = (obj.outputs || []).map((argument) => {
            const values = argument.value.map((value) => value.obj);
            return new coreml.Argument(argument.name, values, argument.visible);
        });
        this.attributes = Object.entries(obj.attributes).map(([name, value]) => {
            const metadata = context.metadata.attribute(obj.type, name);
            return new coreml.Attribute(metadata, name, value);
        });
    }
};

coreml.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.value = value;
        if (this.value instanceof coreml.Tensor) {
            this.type = 'tensor';
        }
        if (metadata) {
            if (metadata.type) {
                this.type = metadata.type;
            }
            if (this.type && coreml.proto) {
                this.value = coreml.Utility.enum(this.type, this.value);
            }
            if (metadata.visible === false) {
                this.visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (Array.isArray(value)) {
                    value = value.map((item) => Number(item));
                }
                if (typeof value === 'bigint') {
                    value = Number(value);
                }
                if (JSON.stringify(metadata.default) === JSON.stringify(value)) {
                    this.visible = false;
                }
            }
        }
        if (this.value instanceof coreml.Graph) {
            this.type = 'graph';
        }
    }
};

coreml.Tensor = class {

    constructor(type, values, quantization, category) {
        this.type = type;
        this.values = values;
        this.category = category;
        this._quantization = quantization;
        if (type.dataType === 'float32') {
            this.encoding = '|';
        } else if ((type.dataType.startsWith('uint') && type.dataType.length === 5) ||
                   (type.dataType.startsWith('int')  && type.dataType.length === 4)) {
            this.encoding = '>';
        } else {
            this.encoding = '<';
        }
        if (quantization &&
            quantization.linearQuantization &&
            Array.isArray(quantization.linearQuantization.scale) &&
            Array.isArray(quantization.linearQuantization.bias)) {
            this.quantization = {
                type: 'linear',
                scale: quantization.linearQuantization.scale,
                bias: quantization.linearQuantization.bias
            };
        }
        if (quantization &&
            quantization.lookupTableQuantization &&
            quantization.lookupTableQuantization.floatValue &&
            quantization.lookupTableQuantization.floatValue.length > 0) {
            this.quantization = {
                type: 'lookup',
                value: quantization.lookupTableQuantization.floatValue
            };
        }
    }
};

coreml.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape || new coreml.TensorShape([]);
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

coreml.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions) &&
            this.dimensions.length === obj.dimensions.length &&
            obj.dimensions.every((value, index) => this.dimensions[index] === value);
    }

    toString() {
        return Array.isArray(this.dimensions) && this.dimensions.length > 0 ?
            `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]` : '';
    }
};

coreml.ListType = class {

    constructor(elementType) {
        this.elementType = elementType;
    }

    equals(obj) {
        return obj instanceof coreml.ListType && this.elementType.equals(obj.elementType);
    }

    toString() {
        return `list<${this.elementType}>`;
    }
};

coreml.MapType = class {

    constructor(keyType, valueType) {
        this.keyType = keyType;
        this.valueType = valueType;
    }

    equals(obj) {
        return obj instanceof coreml.MapType && this.keyType.equals(obj.keyType) && this.valueType.equals(obj.valueType);
    }

    toString() {
        return `map<${this.keyType},${this.valueType}>`;
    }
};

coreml.SequenceType = class {

    constructor(type) {
        this.type = type;
    }

    equals(obj) {
        return obj instanceof coreml.SequenceType && this.type.equals(obj.type);
    }

    toString() {
        return `sequence<${this.type}>`;
    }
};

coreml.ImageType = class {

    constructor(colorSpace, width, height) {
        this.width = width;
        this.height = height;
        switch (colorSpace) {
            case coreml.proto.ImageFeatureType.ColorSpace.GRAYSCALE:
                this.colorSpace = 'grayscale';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.RGB:
                this.colorSpace = 'RGB';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.BGR:
                this.colorSpace = 'BGR';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.GRAYSCALE_FLOAT16:
                this.colorSpace = 'grayscale:float16';
                break;
            default:
                throw new coreml.Error(`Unsupported image color space '${colorSpace}'.`);
        }
    }

    equals(obj) {
        return obj instanceof coreml.ImageType && this.width === obj.width && this.height === obj.height && this.colorSpace === obj.colorSpace;
    }

    toString() {
        return `image<${this.colorSpace},${this.width. toString()}x${this.height}>`;
    }
};

coreml.OptionalType = class {

    constructor(type) {
        this.type = type;
    }

    equals(obj) {
        return obj instanceof coreml.OptionalType && this.type.equals(obj.type);
    }

    toString() {
        return `optional<${this.type}>`;
    }
};

coreml.Context = class {

    constructor(metadata, model, weights, values) {
        this.metadata = metadata;
        this.weights = weights;
        this.values = values || new Map();
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        if (model) {
            const description = model.description;
            const inputs = description && Array.isArray(description.input) ? description.input : [];
            for (const description of inputs) {
                const value = this.output(description.name);
                this.update(value, description);
                this.inputs.push({ name: description.name, visible: true, value: [value] });
            }
            this.type = this.model(model, '', description);
            const outputs = description && Array.isArray(description.output) ? description.output : [];
            for (const description of outputs) {
                const value = this.input(description.name);
                this.update(value, description);
                this.outputs.push({ name: description.name, visible: true, value: [value] });
            }
        }
    }

    context() {
        return new coreml.Context(this.metadata, null, this.weights, this.values);
    }

    network(obj) {
        const context = this.context();
        for (const layer of obj.layers) {
            const type = layer.layer;
            context.node(context.groups, type, layer.name, '', layer[type], layer.input, layer.output, layer.inputTensor, layer.outputTensor);
        }
        context.updatePreprocessing('', obj.preprocessing, null);
        context.type = 'Neural Network';
        return context;
    }

    input(name) {
        if (!this.values.has(name)) {
            this.values.set(name, { counter: 0, name, to: [], from: [] });
        }
        return this.values.get(name);
    }

    output(name) {
        if (this.values.has(name)) {
            const value = { ...this.values.get(name) };
            value.counter++;
            value.name = `${name}|${value.counter}`; // custom argument id
            this.values.set(name, value);
            this.values.set(value.name, value);
        } else {
            const value = { counter: 0, name, to: [], from: [] };
            this.values.set(name, value);
            const key = `${name}|${value.counter}`;
            this.values.set(key, value);
        }
        return this.values.get(name);
    }

    update(value, description) {
        if (!value.type) {
            value.type = coreml.Utility.featureType(description.type);
        }
        if (!value.description && description.shortDescription) {
            value.description = description.shortDescription;
        }
    }

    node(group, type, name, description, data, inputs, outputs, inputTensors, outputTensors) {
        const obj = {
            group,
            type,
            name,
            description,
            attributes: {},
            inputs: [],
            outputs: []
        };
        inputs = inputs.map((input, index) => {
            const value = this.input(input);
            if (!value.type && inputTensors && index < inputTensors.length) {
                const tensor = inputTensors[index];
                const shape = tensor && tensor.dimValue ? new coreml.TensorShape(tensor.dimValue) : null;
                value.type = new coreml.TensorType('?', shape);
            }
            return value;
        });
        outputs = outputs.map((output, index) => {
            const value = this.output(output);
            if (!value.type && outputTensors && index < outputTensors.length) {
                const tensor = outputTensors[index];
                const shape = tensor && tensor.dimValue ? new coreml.TensorShape(tensor.dimValue) : null;
                value.type = new coreml.TensorType('?', shape);
            }
            return value;
        });
        const initializers = [];
        const initializer = (type, name, shape, data) => {
            let dataType = '?';
            let quantization = null;
            let values = null;
            if (data) {
                if (data.floatValue && data.floatValue.length > 0) {
                    values = data.floatValue;
                    dataType = 'float32';
                } else if (data.float16Value && data.float16Value.length > 0) {
                    values = data.float16Value; // byte[]
                    dataType = 'float16';
                } else if (data.rawValue && data.rawValue.length > 0) {
                    if (data.quantization) {
                        values = data.rawValue;
                        dataType = `uint${data.quantization.numberOfBits}`;
                    } else {
                        shape = [];
                    }
                }
                quantization = data.quantization || null;
            }
            const tensorType = new coreml.TensorType(dataType, new coreml.TensorShape(shape));
            const tensor = new coreml.Tensor(tensorType, values, quantization, 'Weights');
            const input = this.metadata.input(type, name);
            const visible = input && input.visible === false ? false : true;
            const value = { obj: new coreml.Value('', null, null, tensor) };
            initializers.push({ name, visible, value: [value] });
        };
        const vector = (value) => {
            return (value && Object.keys(value).length === 1 && value.vector) ? value.vector : value;
        };
        const weights = (type, data) => {
            switch (type) {
                case 'convolution': {
                    const weightsShape = [data.outputChannels, data.kernelChannels, data.kernelSize[0], data.kernelSize[1]];
                    if (data.isDeconvolution) {
                        weightsShape[0] = data.kernelChannels;
                        weightsShape[1] = Math.floor(Number(data.outputChannels / (data.nGroups === 0 ? 1 : data.nGroups)));
                    }
                    initializer(type, 'weights', weightsShape, data.weights);
                    if (data.hasBias) {
                        initializer(type, 'bias', [data.outputChannels], data.bias);
                    }
                    return { 'weights': true, 'bias': data.hasBias };
                }
                case 'innerProduct':
                    initializer(type, 'weights', [data.outputChannels, data.inputChannels], data.weights);
                    if (data.hasBias) {
                        initializer(type, 'bias', [data.outputChannels], data.bias);
                    }
                    return { 'weights': true, 'bias': data.hasBias };
                case 'batchnorm':
                    initializer(type, 'gamma', [data.channels], data.gamma);
                    initializer(type, 'beta', [data.channels], data.beta);
                    if (data.mean) {
                        initializer(type, 'mean', [data.channels], data.mean);
                    }
                    if (data.variance) {
                        initializer(type, 'variance', [data.channels], data.variance);
                    }
                    return { 'gamma': true, 'beta': true, 'mean': true, 'variance': true };
                case 'embedding':
                    initializer(type, 'weights', [data.inputDim, data.outputChannels], data.weights);
                    return { 'weights': true };
                case 'loadConstant':
                case 'loadConstantND':
                    initializer(type, 'data', data.shape, data.data);
                    return { 'data': true };
                case 'scale':
                    initializer(type, 'scale', data.shapeScale, data.scale);
                    if (data.hasBias) {
                        initializer(type, 'bias', data.shapeBias, data.bias);
                    }
                    return { 'scale': true, 'bias': data.hasBias };
                case 'bias':
                    initializer(type, 'bias', data.shape, data.bias);
                    return { 'bias': true };
                case 'simpleRecurrent':
                    initializer(type, 'weights', [data.outputVectorSize, data.inputVectorSize], data.weightMatrix);
                    initializer(type, 'recurrent', [data.outputVectorSize, data.inputVectorSize], data.recursionMatrix);
                    if (data.hasBiasVectors) {
                        initializer(type, 'bias', [data.outputVectorSize], data.biasVector);
                    }
                    return { 'weightMatrix': true, 'recursionMatrix': true, 'biasVector': data.hasBiasVectors };
                case 'gru': {
                    const recursionMatrixShape = [data.outputVectorSize, data.outputVectorSize];
                    const weightMatrixShape = [data.outputVectorSize, data.inputVectorSize];
                    const biasVectorShape = [data.outputVectorSize];
                    initializer(type, 'updateGateWeightMatrix', weightMatrixShape, data.updateGateWeightMatrix);
                    initializer(type, 'resetGateWeightMatrix', weightMatrixShape, data.resetGateWeightMatrix);
                    initializer(type, 'outputGateWeightMatrix', weightMatrixShape, data.outputGateWeightMatrix);
                    initializer(type, 'updateGateRecursionMatrix', recursionMatrixShape, data.updateGateRecursionMatrix);
                    initializer(type, 'resetGateRecursionMatrix', recursionMatrixShape, data.resetGateRecursionMatrix);
                    initializer(type, 'outputGateRecursionMatrix', recursionMatrixShape, data.outputGateRecursionMatrix);
                    if (data.hasBiasVectors) {
                        initializer(type, 'updateGateBiasVector', biasVectorShape, data.updateGateBiasVector);
                        initializer(type, 'resetGateBiasVector', biasVectorShape, data.resetGateBiasVector);
                        initializer(type, 'outputGateBiasVector', biasVectorShape, data.outputGateBiasVector);
                    }
                    return {
                        'updateGateWeightMatrix': true, 'resetGateWeightMatrix': true, 'outputGateWeightMatrix': true,
                        'updateGateRecursionMatrix': true, 'resetGateRecursionMatrix': true, 'outputGateRecursionMatrix': true,
                        'updateGateBiasVector': data.hasBiasVectors, 'resetGateBiasVector': data.hasBiasVectors, 'outputGateBiasVector': data.hasBiasVectors
                    };
                }
                case 'uniDirectionalLSTM':
                case 'biDirectionalLSTM': {
                    const count = (type === 'uniDirectionalLSTM') ? 1 : 2;
                    const h = data.outputVectorSize;
                    const x = data.inputVectorSize;
                    for (let i = 0; i < count; i++) {
                        const weights = count === 1 ? data.weightParams : data.weightParams[i];
                        const suffix = (i === 0) ? '' : '_rev';
                        initializer(type, `inputGateWeightMatrix${suffix}`, [h,x], weights.inputGateWeightMatrix);
                        initializer(type, `forgetGateWeightMatrix${suffix}`, [h,x], weights.forgetGateWeightMatrix);
                        initializer(type, `blockInputWeightMatrix${suffix}`, [h,x], weights.blockInputWeightMatrix);
                        initializer(type, `outputGateWeightMatrix${suffix}`, [h,x], weights.outputGateWeightMatrix);
                        initializer(type, `inputGateRecursionMatrix${suffix}`, [h,h], weights.inputGateRecursionMatrix);
                        initializer(type, `forgetGateRecursionMatrix${suffix}`, [h,h],weights.forgetGateRecursionMatrix);
                        initializer(type, `blockInputRecursionMatrix${suffix}`, [h,h], weights.blockInputRecursionMatrix);
                        initializer(type, `outputGateRecursionMatrix${suffix}`, [h,h], weights.outputGateRecursionMatrix);
                        if (data.params.hasBiasVectors) {
                            initializer(type, `inputGateBiasVector${suffix}`, [h], weights.inputGateBiasVector);
                            initializer(type, `forgetGateBiasVector${suffix}`, [h], weights.forgetGateBiasVector);
                            initializer(type, `blockInputBiasVector${suffix}`, [h], weights.blockInputBiasVector);
                            initializer(type, `outputGateBiasVector${suffix}`, [h], weights.outputGateBiasVector);
                        }
                        if (data.params.hasPeepholeVectors) {
                            initializer(type, `inputGatePeepholeVector${suffix}`, [h], weights.inputGatePeepholeVector);
                            initializer(type, `forgetGatePeepholeVector${suffix}`, [h], weights.forgetGatePeepholeVector);
                            initializer(type, `outputGatePeepholeVector${suffix}`, [h], weights.outputGatePeepholeVector);
                        }
                    }
                    return { 'weightParams': true };
                }
                case 'dictVectorizer':
                    data.stringToIndex = vector(data.stringToIndex);
                    return {};
                case 'wordTagger':
                    data.modelParameterData = Array.from(data.modelParameterData);
                    data.stringTags = vector(data.stringTags);
                    return { tokensOutputFeatureName: true, tokenTagsOutputFeatureName: true, tokenLengthsOutputFeatureName: true, tokenLocationsOutputFeatureName: true };
                case 'textClassifier':
                    data.modelParameterData = Array.from(data.modelParameterData);
                    data.stringClassLabels = vector(data.stringClassLabels);
                    return {};
                case 'nonMaximumSuppression':
                    data.stringClassLabels = vector(data.stringClassLabels);
                    return {};
                default:
                    return {};
            }
        };
        if (data) {
            const attributes = obj.attributes;
            const map = weights(type, data, initializers);
            for (const [name, value] of Object.entries(data)) {
                if (!map[name]) {
                    attributes[name] = value;
                }
            }
            switch (obj.type) {
                case 'loop':
                    attributes.bodyNetwork = this.network(attributes.bodyNetwork);
                    attributes.conditionNetwork = this.network(attributes.conditionNetwork);
                    break;
                case 'branch':
                    attributes.ifBranch = this.network(attributes.ifBranch);
                    attributes.elseBranch = this.network(attributes.elseBranch);
                    break;
                default:
                    break;
            }
        }
        const metadata = this.metadata.type(type);
        for (let i = 0; i < inputs.length;) {
            const input = metadata && metadata.inputs && i < metadata.inputs.length ? metadata.inputs[i] : { name: i === 0 ? 'input' : i.toString() };
            const count = input.type === 'Tensor[]' ? inputs.length - i : 1;
            const values = inputs.slice(i, i + count);
            obj.inputs.push({ name: input.name, visible: true, value: values });
            i += count;
        }
        obj.inputs.push(...initializers);
        for (let i = 0; i < outputs.length;) {
            const output = metadata && metadata.outputs && i < metadata.outputs.length ? metadata.outputs[i] : { name: i === 0 ? 'output' : i.toString() };
            const count = output.type === 'Tensor[]' ? outputs.length - i : 1;
            const args = outputs.slice(i, i + count);
            obj.outputs.push({ name: output.name, visible: true, value: args });
            i += count;
        }
        this.nodes.push(obj);
        return obj;
    }

    model(model, group, description) {
        this.groups |= group.length > 0;
        const shortDescription = model && model.description && model.description.metadata && model.description.metadata.shortDescription ? model.description.metadata.shortDescription : '';
        switch (model.Type) {
            case 'neuralNetworkClassifier': {
                const neuralNetworkClassifier = model.neuralNetworkClassifier;
                for (const layer of neuralNetworkClassifier.layers) {
                    const type = layer.layer;
                    this.node(group, type, layer.name, group === '' ? '' : shortDescription, layer[type], layer.input, layer.output, layer.inputTensor, layer.outputTensor);
                }
                this.updateClassifierOutput(group, neuralNetworkClassifier, description);
                this.updatePreprocessing(group, neuralNetworkClassifier.preprocessing, description);
                return 'Neural Network Classifier';
            }
            case 'neuralNetwork': {
                const neuralNetwork = model.neuralNetwork;
                for (const layer of neuralNetwork.layers) {
                    this.node(group, layer.layer, layer.name, group === '' ? '' : shortDescription, layer[layer.layer], layer.input, layer.output, layer.inputTensor, layer.outputTensor);
                }
                this.updatePreprocessing(group, neuralNetwork.preprocessing, description);
                return 'Neural Network';
            }
            case 'neuralNetworkRegressor': {
                const neuralNetworkRegressor = model.neuralNetworkRegressor;
                for (const layer of neuralNetworkRegressor.layers) {
                    this.node(group, layer.layer, layer.name, shortDescription, layer[layer.layer], layer.input, layer.output);
                }
                this.updatePreprocessing(group, neuralNetworkRegressor, description);
                return 'Neural Network Regressor';
            }
            case 'pipeline': {
                for (let i = 0; i < model.pipeline.models.length; i++) {
                    this.model(model.pipeline.models[i], `${group ? (`${group}/`) : ''}pipeline[${i}]`, description);
                }
                return 'Pipeline';
            }
            case 'pipelineClassifier': {
                for (let i = 0; i < model.pipelineClassifier.pipeline.models.length; i++) {
                    this.model(model.pipelineClassifier.pipeline.models[i], `${group ? (`${group}/`) : ''}pipelineClassifier[${i}]`, description);
                }
                return 'Pipeline Classifier';
            }
            case 'pipelineRegressor': {
                for (let i = 0; i < model.pipelineRegressor.pipeline.models.length; i++) {
                    this.model(model.pipelineRegressor.pipeline.models[i], `${group ? (`${group}/`) : ''}pipelineRegressor[${i}]`, description);
                }
                return 'Pipeline Regressor';
            }
            case 'glmClassifier': {
                this.node(group, 'glmClassifier', null, shortDescription,
                    {
                        classEncoding: model.glmClassifier.classEncoding,
                        offset: model.glmClassifier.offset,
                        weights: model.glmClassifier.weights
                    },
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                this.updateClassifierOutput(group, model.glmClassifier, description);
                return 'Generalized Linear Classifier';
            }
            case 'glmRegressor': {
                this.node(group, 'glmRegressor', null, shortDescription,
                    model.glmRegressor,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Generalized Linear Regressor';
            }
            case 'treeEnsembleClassifier': {
                this.node(group, 'treeEnsembleClassifier', null, shortDescription,
                    model.treeEnsembleClassifier.treeEnsemble,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                this.updateClassifierOutput(group, model.treeEnsembleClassifier, description);
                return 'Tree Ensemble Classifier';
            }
            case 'treeEnsembleRegressor': {
                this.node(group, 'treeEnsembleRegressor', null, shortDescription,
                    model.treeEnsembleRegressor.treeEnsemble,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Tree Ensemble Regressor';
            }
            case 'supportVectorClassifier': {
                this.node(group, 'supportVectorClassifier', null, shortDescription,
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
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                this.updateClassifierOutput(group, model.supportVectorClassifier, description);
                return 'Support Vector Classifier';
            }
            case 'supportVectorRegressor': {
                this.node(group, 'supportVectorRegressor', null, shortDescription,
                    {
                        coefficients: model.supportVectorRegressor.coefficients,
                        kernel: model.supportVectorRegressor.kernel,
                        rho: model.supportVectorRegressor.rho,
                        supportVectors: model.supportVectorRegressor.supportVectors
                    },
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Support Vector Regressor';
            }
            case 'oneHotEncoder': {
                const categoryType = model.oneHotEncoder.CategoryType;
                const oneHotEncoderParams = { outputSparse: model.oneHotEncoder.outputSparse };
                oneHotEncoderParams[categoryType] = model.oneHotEncoder[categoryType];
                this.node(group, 'oneHotEncoder', null, shortDescription,
                    oneHotEncoderParams,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'One Hot Encoder';
            }
            case 'imputer': {
                const imputedValue = model.imputer.ImputedValue;
                const replaceValue = model.imputer.ReplaceValue;
                const imputerParams = {};
                imputerParams[imputedValue] = model.imputer[imputedValue];
                imputerParams[replaceValue] = model.imputer[replaceValue];
                this.node(group, 'oneHotEncoder', null, shortDescription,
                    imputerParams,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Imputer';
            }
            case 'featureVectorizer': {
                this.node(group, 'featureVectorizer', null, shortDescription,
                    model.featureVectorizer,
                    model.description.input.map((item) => item.name),
                    [model.description.output[0].name]);
                return 'Feature Vectorizer';
            }
            case 'dictVectorizer': {
                this.node(group, 'dictVectorizer', null, shortDescription,
                    model.dictVectorizer,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Dictionary Vectorizer';
            }
            case 'scaler': {
                this.node(group, 'scaler', null, shortDescription,
                    model.scaler,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Scaler';
            }
            case 'categoricalMapping': {
                this.node(group, 'categoricalMapping', null, shortDescription,
                    model.categoricalMapping,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Categorical Mapping';
            }
            case 'normalizer': {
                this.node(group, 'normalizer', null, shortDescription,
                    model.normalizer,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Normalizer';
            }
            case 'arrayFeatureExtractor': {
                this.node(group, 'arrayFeatureExtractor', null, shortDescription,
                    { extractIndex: model.arrayFeatureExtractor.extractIndex },
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Array Feature Extractor';
            }
            case 'nonMaximumSuppression': {
                const nonMaximumSuppressionParams = {
                    pickTop: model.nonMaximumSuppression.pickTop,
                    stringClassLabels: model.nonMaximumSuppression.stringClassLabels,
                    iouThreshold: model.nonMaximumSuppression.iouThreshold,
                    confidenceThreshold: model.nonMaximumSuppression.confidenceThreshold
                };
                this.node(group, 'nonMaximumSuppression', null, shortDescription,
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
            case 'wordTagger': {
                this.node(group, 'wordTagger', null, shortDescription,
                    model.wordTagger,
                    [model.description.input[0].name],
                    [
                        model.wordTagger.tokensOutputFeatureName,
                        model.wordTagger.tokenTagsOutputFeatureName,
                        model.wordTagger.tokenLocationsOutputFeatureName,
                        model.wordTagger.tokenLengthsOutputFeatureName
                    ]);
                return 'Word Tagger';
            }
            case 'textClassifier': {
                this.node(group, 'textClassifier', null, shortDescription,
                    model.textClassifier,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Text Classifier';
            }
            case 'visionFeaturePrint': {
                const visionFeaturePrintParams = {
                    scene: model.visionFeaturePrint.scene
                };
                this.node(group, 'visionFeaturePrint', null, shortDescription,
                    visionFeaturePrintParams,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Vision Feature Print';
            }
            case 'soundAnalysisPreprocessing': {
                this.node(group, 'soundAnalysisPreprocessing', null, shortDescription,
                    model.soundAnalysisPreprocessing,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Sound Analysis Preprocessing';
            }
            case 'kNearestNeighborsClassifier': {
                this.node(group, 'kNearestNeighborsClassifier', null, shortDescription,
                    model.kNearestNeighborsClassifier,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                this.updateClassifierOutput(group, model.kNearestNeighborsClassifier, description);
                return 'Nearest Neighbors Classifier';
            }
            case 'itemSimilarityRecommender': {
                this.node(group, 'itemSimilarityRecommender', null, shortDescription,
                    {
                        itemStringIds: model.itemSimilarityRecommender.itemStringIds.vector,
                        itemItemSimilarities: model.itemSimilarityRecommender.itemItemSimilarities
                    },
                    model.description.input.map((feature) => feature.name),
                    model.description.output.map((feature) => feature.name));
                return 'Item Similarity Recommender';
            }
            case 'audioFeaturePrint': {
                this.node(group, 'audioFeaturePrint', null, shortDescription,
                    model.audioFeaturePrint,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Audio Feature Print';
            }
            case 'linkedModel': {
                this.node(group, 'linkedModel', null, shortDescription,
                    model.linkedModel.linkedModelFile,
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'Linked Model';
            }
            case 'customModel': {
                this.node(group, 'customModel', null, shortDescription,
                    { className: model.customModel.className, parameters: model.customModel.parameters },
                    [model.description.input[0].name],
                    [model.description.output[0].name]);
                return 'customModel';
            }
            case 'mlProgram': {
                return this.program(model.mlProgram, group);
            }
            default: {
                throw new coreml.Error(`Unsupported model type '${JSON.stringify(Object.keys(model))}'.`);
            }
        }
    }

    updateClassifierOutput(group, classifier, description) {
        let labelProbabilityLayerName = classifier.labelProbabilityLayerName;
        if (!labelProbabilityLayerName && this.nodes.length > 0) {
            const node = this.nodes.slice(-1).pop();
            if (node && node.outputs.length === 1 && node.outputs[0].value.length === 1) {
                labelProbabilityLayerName = node.outputs[0].value[0].name;
            }
        }
        let predictedFeatureName = description.predictedFeatureName;
        let predictedProbabilitiesName = description.predictedProbabilitiesName;
        if ((predictedFeatureName || predictedProbabilitiesName) && labelProbabilityLayerName && classifier.ClassLabels) {
            predictedFeatureName = predictedFeatureName ? predictedFeatureName : '?';
            predictedProbabilitiesName = predictedProbabilitiesName ? predictedProbabilitiesName : '?';
            const labelProbabilityInput = `${labelProbabilityLayerName}:labelProbabilityLayerName`;
            const values = new Set();
            for (const node of this.nodes) {
                for (const output of node.outputs) {
                    for (const value of output.value) {
                        if (value.name === labelProbabilityLayerName) {
                            value.name = labelProbabilityInput;
                            values.add(value);
                        }
                    }
                }
            }
            this.values.set(labelProbabilityInput, this.values.get(labelProbabilityLayerName));
            this.values.delete(labelProbabilityLayerName);
            const type = classifier.ClassLabels;
            const node = {
                // group: this._group,
                type,
                name: null,
                description: '',
                attributes: classifier[type] || {}
            };
            node.inputs = [
                { name: 'input', visible: true, value: Array.from(values) }
            ];
            node.outputs = [
                { name: 'probabilities', visible: true, value: [this.output(predictedProbabilitiesName)] },
                { name: 'feature', visible: true, value: [this.output(predictedFeatureName)] }
            ];
            this.nodes.push(node);
        }
    }

    updatePreprocessing(group, preprocessings, description) {
        if (preprocessings && preprocessings.length > 0) {
            const preprocessingInput = description.input[0].name;
            const inputNodes = [];
            for (const node of this.nodes) {
                if (node.inputs.some((input) => Array.isArray(input.value) && input.value.some((arg) => arg.name === preprocessingInput))) {
                    inputNodes.push(node);
                }
            }
            let currentOutput = preprocessingInput;
            let preprocessorOutput = null;
            let preprocessorIndex = 0;
            for (const preprocessing of preprocessings) {
                const input = preprocessing.featureName ? preprocessing.featureName : currentOutput;
                currentOutput = `${preprocessingInput}:${preprocessorIndex}`;
                const preprocessor = preprocessing.preprocessor;
                const node = this.node(group, preprocessor, null, '', preprocessing[preprocessor], [input], [currentOutput]);
                /* eslint-disable prefer-destructuring */
                preprocessorOutput = node.outputs[0].value[0];
                /* eslint-enable prefer-destructuring */
                preprocessorIndex++;
            }
            for (const node of inputNodes) {
                for (const input of node.inputs) {
                    if (Array.isArray(input.value)) {
                        for (let i = 0; i < input.value.length; i++) {
                            if (input.value[i].name === preprocessingInput) {
                                input.value[i] = preprocessorOutput;
                            }
                        }
                    }
                }
            }
        }
    }

    program(program, group) {
        // need to handle functions other than main?
        const main = program.functions.main;
        // need to handle more than one block specialization?
        const block_specializations = main.block_specializations;
        const key = Object.keys(block_specializations).filter((key) => key.startsWith('CoreML')).shift();
        const block = block_specializations[key];
        const convertValue = (value) => {
            switch (value.value) {
                case 'immediateValue': {
                    const tensor = value.immediateValue.tensor;
                    const type = coreml.Utility.valueType(value.type);
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
                            throw new coreml.Error(`Unsupported tensor value '${tensor.value}'.`);
                    }
                    if (type.shape.dimensions.length === 0) {
                        [values] = values;
                    }
                    return values;
                }
                case 'blobFileValue': {
                    const type = coreml.Utility.valueType(value.type);
                    const blob = value.blobFileValue;
                    const offset = Number(blob.offset);
                    const file = blob.fileName;
                    let data = null;
                    const stream = this.weights.get(file);
                    if (stream) {
                        stream.seek(offset);
                        const buffer = stream.read(32);
                        const reader = base.BinaryReader.open(buffer);
                        const signature = reader.uint32();
                        if (signature === 0xdeadbeef) {
                            reader.uint32(); // dataType
                            const size = reader.uint64().toNumber();
                            const offset = reader.uint64().toNumber();
                            stream.seek(offset);
                            const length = (type.shape.dimensions || []).reduce((a, b) => a * b, 1);
                            switch (type.dataType) {
                                case 'float32': {
                                    const buffer = stream.read(size);
                                    data = new Float32Array(buffer.buffer, buffer.byteOffset, length).slice();
                                    break;
                                }
                                case 'float16':
                                case 'int8':
                                case 'uint8': {
                                    data = stream.read(size);
                                    break;
                                }
                                default:
                                    throw new coreml.Error(`Unsupported blob data type '${type.dataType}'.`);
                            }
                        }
                    }
                    return new coreml.Tensor(type, data, null, 'Blob');
                }
                default: {
                    throw new coreml.Error(`Unsupported value '${value.value}'.`);
                }
            }
        };
        const operations = block.operations.map((op) => {
            const operation = {
                type: op.type,
                attributes: {}
            };
            for (const [key, value] of Object.entries(op.attributes)) {
                operation.attributes[key] = convertValue(value);
            }
            operation.inputs = Object.entries(op.inputs).map(([name, input]) => {
                const args = input.arguments.map((argument) => {
                    if (argument.name) {
                        const value = this.input(argument.name);
                        value.to.push(operation);
                        return value;
                    }
                    return { value: argument.value };
                });
                return { name, value: args };
            });
            operation.outputs = op.outputs.map((output) => {
                const value = this.input(output.name);
                value.type = coreml.Utility.valueType(output.type);
                value.from.push(operation);
                return { name: 'output', value: [value] };
            });
            return operation;
        });
        for (const op of operations) {
            if (op.type === 'const' && op.inputs.length === 0 &&
                op.outputs.length === 1 && op.outputs[0].value.length === 1) {
                /* eslint-disable prefer-destructuring */
                const value = op.outputs[0].value[0];
                /* eslint-enable prefer-destructuring */
                if (op.attributes && op.attributes.val) {
                    const type = value.type;
                    const data = op.attributes.val;
                    if (data instanceof Uint8Array && data.length === 2 &&
                        type.dataType === 'float16' && type.shape.dimensions.length === 0) {
                        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                        value.value = view.getFloat16(0, true);
                    } else {
                        value.value = data;
                    }
                    value.const = true;
                    op.delete = true;
                }
            }
        }
        for (const op of operations) {
            for (const input of op.inputs) {
                if (input.value.length > 1 && input.value.some((argument) => argument.const)) {
                    if (!input.value.every((argument) => argument.value instanceof coreml.Tensor)) {
                        for (const value of input.value) {
                            for (const from of value.from) {
                                from.delete = false;
                            }
                            delete value.value;
                        }
                    }
                }
            }
        }
        for (const op of operations.filter((op) => !op.delete)) {
            op.inputs = op.inputs.filter((input) => {
                if (input.value.every((value) => value.value === undefined || value.value instanceof coreml.Tensor)) {
                    return true;
                }
                op.attributes[input.name] = input.value.length === 1 ?
                    input.value[0].value :
                    input.value.map((argument) => argument.value[0]);
                return false;
            });
        }
        const mapValue = (name, value) => {
            if (value.value instanceof coreml.Tensor) {
                value.initializer = value.value;
                delete value.value;
            }
            if (!this.values.has(name)) {
                this.values.set(name, value);
            } else if ((value.type && !value.type.equals(this.values.get(name).type)) ||
                       (value.initializer && value.initializer !== this.values.get(name).initializer)) {
                throw new coreml.Error(`Duplicate value '${name}'.`);
            }
            return this.values.get(name);
        };
        for (const op of operations.filter((op) => !op.delete)) {
            for (const argument of op.inputs) {
                for (const value of argument.value) {
                    mapValue(value.name, value);
                }
            }
            for (const argument of op.outputs) {
                for (const value of argument.value) {
                    mapValue(value.name, value);
                }
            }
        }
        for (const op of operations.filter((op) => !op.delete)) {
            op.group = group;
            op.type = `program:${op.type}`;
            const metadata = this.metadata.type(op.type);
            if (metadata && Array.isArray(metadata.inputs)) {
                const map = new Map(metadata.inputs.map((input, index) => [input.name, index + 1]));
                op.inputs.sort((a, b) => (map.get(a.name) || map.size) - (map.get(b.name) || map.size));
            }
            this.nodes.push(op);
        }
        return 'ML Program';
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
                const map = new Map(Object.entries(type).map(([key, value]) => [value, key]));
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
                        shape = new coreml.TensorShape(type.multiArrayType.shape.map((dim) => Number(dim)));
                    }
                    let dataType = '';
                    const ArrayDataType = coreml.proto.ArrayFeatureType.ArrayDataType;
                    switch (type.multiArrayType.dataType) {
                        case ArrayDataType.INVALID_ARRAY_DATA_TYPE:
                            dataType = '?';
                            break;
                        case ArrayDataType.FLOAT16:
                            dataType = 'float16';
                            break;
                        case ArrayDataType.FLOAT32:
                            dataType = 'float32';
                            break;
                        case ArrayDataType.DOUBLE:
                            dataType = 'float64';
                            break;
                        case ArrayDataType.INT32:
                            dataType = 'int32';
                            break;
                        default:
                            throw new coreml.Error(`Unsupported array data type '${type.multiArrayType.dataType}'.`);
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
                case 'sequenceType': {
                    result = new coreml.SequenceType(coreml.Utility.featureType(type[type.Type]));
                    break;
                }
                case 'imageType': {
                    result = new coreml.ImageType(type.imageType.colorSpace, type.imageType.width, type.imageType.height);
                    break;
                }
                default: {
                    throw new coreml.Error(`Unsupported feature type '${type.Type}'.`);
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
            coreml.Utility._dataTypes = new Map(Object.entries(coreml.proto.MILSpec.DataType).map((([key, value]) => [value, key.toLowerCase()])));
            coreml.Utility._dataTypes.delete(0);
            coreml.Utility._dataTypes.set(1, 'boolean');
        }
        const shape = type.dimensions.map((dim) => dim.constant ? dim.constant.size : '?');
        const dataType = coreml.Utility._dataTypes.get(type.dataType);
        if (!dataType) {
            throw new coreml.Error(`Unsupported data type '${type.dataType}'.`);
        }
        return new coreml.TensorType(dataType, new coreml.TensorShape(shape));
    }

    static valueType(type) {
        switch (type.type) {
            case 'tensorType':
                return coreml.Utility.tensorType(type.tensorType);
            case 'listType':
                return new coreml.ListType(coreml.Utility.valueType(type.listType.type));
            case 'dictionaryType':
                return new coreml.MapType(coreml.Utility.valueType(type.dictionaryType.keyType), coreml.Utility.valueType(type.dictionaryType.valueType));
            default:
                throw new coreml.Error(`Unsupported value type '${type.type}'.`);
        }
    }
};

coreml.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Core ML model.';
    }
};

export const ModelFactory = coreml.ModelFactory;
