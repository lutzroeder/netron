
var coreml = {};
var base = require('./base');
var json = require('./json');
var protobuf = require('./protobuf');

coreml.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const identifier = context.identifier.toLowerCase();
        const extension = identifier.split('.').pop().toLowerCase();
        const tags = context.tags('pb');
        if (tags.get(1) === 0 && tags.get(2) === 2) {
            if (extension === 'pb') {
                const tags = context.tags('pb+');
                const keys = Object.keys(tags).map((key) => parseInt(key, 10));
                const match = (key) =>
                    (key >= 200 && key < 220) ||
                    (key >= 300 && key < 320) ||
                    (key >= 400 && key < 420) ||
                    (key >= 500 && key < 520) ||
                    (key >= 550 && key < 560) ||
                    (key >= 600 && key < 620) ||
                    (key === 900) ||
                    (key >= 2000 && key < 2010) ||
                    (key === 3000);
                if (!keys.some((key) => match(key))) {
                    return null;
                }
            }
            return 'coreml.pb';
        }
        if (identifier === 'manifest.json') {
            const obj = context.open('json');
            if (obj && obj.rootModelIdentifier && obj.itemInfoEntries) {
                const entries = Object.keys(obj.itemInfoEntries).map((key) => obj.itemInfoEntries[key]);
                if (entries.filter((entry) => entry.path.toLowerCase().endsWith('.mlmodel').length === 1)) {
                    return 'coreml.manifest';
                }
            }
        }
        if (identifier === 'metadata.json') {
            const obj = context.open('json');
            if (obj && obj.rootModelIdentifier && obj.itemInfoEntries) {
                return 'coreml.metadata';
            }
        }
        if (identifier === 'featuredescriptions.json') {
            const obj = context.open('json');
            if (obj && (obj.Inputs || obj.Outputs)) {
                return 'coreml.featuredescriptions';
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

    async open(context, target) {
        await context.require('./coreml-proto');
        const metadata = await context.metadata('coreml-metadata.json');
        const openModel = async (stream, context, path, format) => {
            let model = null;
            try {
                coreml.proto = protobuf.get('coreml').CoreML.Specification;
                const reader = protobuf.BinaryReader.open(stream);
                model = coreml.proto.Model.decode(reader);
            } catch (error) {
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
                const folder = path.replace(/\/[^/]*$/, '');
                const keys = Array.from(weightPaths);
                const paths = keys.map((path) => path.replace(/^@model_path\//, folder + '/'));
                try {
                    const streams = await Promise.all(paths.map((path) => context.request(path, null)));
                    const weights = new Map();
                    for (let i = 0; i < keys.length; i++) {
                        weights.set(keys[i], streams[i]);
                    }
                    return new coreml.Model(metadata, format, model, weights);
                } catch (error) {
                    return new coreml.Model(metadata, format, model, new Map());
                }
            }
            return new coreml.Model(metadata, format, model, new Map());
        };
        const openManifest = async (obj, context, path) => {
            const entries = Object.values(obj.itemInfoEntries).filter((entry) => entry.path.toLowerCase().endsWith('.mlmodel'));
            if (entries.length !== 1) {
                throw new coreml.Error('Manifest does not contain Core ML model.');
            }
            const file = path + 'Data/' + entries[0].path;
            const stream = await context.request(file, null);
            return openModel(stream, context, file, 'Core ML Package');
        };
        const openManifestStream = async (context, path) => {
            const stream = await context.request(path + 'Manifest.json', null);
            const reader = json.TextReader.open(stream);
            const obj = reader.read();
            return openManifest(obj, context, path);
        };
        switch (target) {
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
                throw new coreml.Error("Unsupported Core ML format '" + target + "'.");
            }
        }
    }
};

coreml.Model = class {

    constructor(metadata, format, model, weights) {
        this._format = (format || 'Core ML') + ' v' + model.specificationVersion.toString();
        this._metadata = [];
        this._graphs = [ new coreml.Graph(metadata, model, weights) ];
        if (model.description && model.description.metadata) {
            const properties = model.description.metadata;
            if (properties.versionString) {
                this._version = properties.versionString;
            }
            if (properties.shortDescription) {
                this._description = properties.shortDescription;
            }
            if (properties.author) {
                this._metadata.push({ name: 'author', value: properties.author });
            }
            if (properties.license) {
                this._metadata.push({ name: 'license', value: properties.license });
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

    get metadata() {
        return this._metadata;
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
        const loadProgram = (program, _, group, weights) => {
            // TODO: need to handle functions other than main?
            const main = program.functions.main;
            // TODO: need to handle more than one block specialization?
            const block = main.block_specializations.CoreML5 || main.block_specializations.CoreML6;
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
                            const reader = new base.BinaryReader(buffer);
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
                                    case 'float16':
                                    case 'uint8': {
                                        data = stream.read(size);
                                        break;
                                    }
                                    default:
                                        throw new coreml.Error("Unsupported blob data type '" + type.dataType + "'.");
                                }
                            }
                        }
                        return new coreml.Tensor(type, data, null, 'Blob');
                    }
                    default: {
                        throw new coreml.Error("Unsupported value '" + value.value + "'.");
                    }
                }
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
                    operation.attributes[key] = convertValue(entry[1]);
                }
                operation.inputs = Object.entries(op.inputs).map((entry) => {
                    const args = entry[1].arguments.map((argument) => {
                        if (argument.name) {
                            const value = arg(argument.name);
                            value.to.push(operation);
                            return value;
                        }
                        return { value: argument.value };
                    });
                    return { name: entry[0], arguments: args };
                });
                operation.outputs = op.outputs.map((output) => {
                    const value = arg(output.name);
                    value.type = coreml.Utility.valueType(output.type);
                    value.from.push(operation);
                    return { name: 'output', arguments: [ value ] };
                });
                return operation;
            });
            for (const op of operations) {
                if (op.type === 'const' && op.inputs.length === 0 &&
                    op.outputs.length === 1 && op.outputs[0].arguments.length === 1) {
                    const value = op.outputs[0].arguments[0];
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
                    if (input.arguments.length > 1 && input.arguments.some((argument) => argument.const)) {
                        if (!input.arguments.every((argument) => argument.value instanceof coreml.Tensor)) {
                            for (const argument of input.arguments) {
                                for (const from of argument.from) {
                                    from.delete = false;
                                }
                                delete argument.value;
                            }
                        }
                    }
                }
            }
            for (const op of operations.filter((op) => !op.delete)) {
                op.inputs = op.inputs.filter((input) => {
                    if (input.arguments.every((argument) => argument.value === undefined || argument.value instanceof coreml.Tensor)) {
                        return true;
                    }
                    op.attributes[input.name] = input.arguments.length === 1 ?
                        input.arguments[0].value :
                        input.arguments.map((argument) => argument.value[0]);
                    return false;
                });
            }
            const values = new Map();
            const value = (name, type, description, tensor) => {
                if (!values.has(name)) {
                    values.set(name, new coreml.Value(name, type, description, tensor));
                } else if ((type && !type.equals(values.get(name).type)) ||
                           (tensor && tensor !== values.get(name).initializer) || description) {
                    throw new coreml.Error("Duplicate value '" + name + "'.");
                }
                return values.get(name);
            };
            for (const op of operations.filter((op) => !op.delete)) {
                op.inputs = op.inputs.map((input) => new coreml.Argument(input.name, true, input.arguments.map((v) => value(v.name, v.type, null, v.value))));
                op.outputs = op.outputs.map((output) => new coreml.Argument(output.name, true, output.arguments.map((v) => value(v.name, v.type, null, v.value))));
            }
            for (const op of operations.filter((op) => !op.delete)) {
                op.group = group;
                op.type = 'program:' + op.type;
                const metadata = this._metadata.type(op.type);
                if (metadata && Array.isArray(metadata.inputs)) {
                    const map = new Map(metadata.inputs.map((input, index) => [ input.name, index + 1 ]));
                    op.inputs.sort((a, b) => (map.get(a.name) || map.size) - (map.get(b.name) || map.size));
                }
                const node = new coreml.Node(this._metadata, op);
                this._nodes.push(node);
            }
            return 'ML Program';
        };
        const createNode = (values, group, type, name, description, data, inputs, outputs, inputTensors, outputTensors) => {
            const obj = {
                group: group,
                type: type,
                name: name,
                description: description,
                attributes: {},
                inputs: [],
                outputs: []
            };
            inputs = inputs.map((input, index) => {
                const value = values.input(input);
                if (!value.type && inputTensors && index < inputTensors.length) {
                    const tensor = inputTensors[index];
                    const shape = tensor && tensor.dimValue ? new coreml.TensorShape(tensor.dimValue) : null;
                    value.type = new coreml.TensorType('?', shape);
                }
                return value;
            });
            outputs = outputs.map((output, index) => {
                const value = values.output(output);
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
                            dataType = 'uint' + data.quantization.numberOfBits.toString();
                        } else {
                            shape = [];
                        }
                    }
                    quantization = data.quantization || null;
                }
                const tensorType = new coreml.TensorType(dataType, new coreml.TensorShape(shape));
                const tensor = new coreml.Tensor(tensorType, values, quantization, 'Weights');
                const value = new coreml.Value('', null, null, tensor);
                const input = this._metadata.input(type, name);
                const visible = input && input.visible === false ? false : true;
                const argument = new coreml.Argument(name, visible, [ value ]);
                initializers.push(argument);
            };
            const vector = (value) => {
                return (value && Object.keys(value).length == 1 && value.vector) ? value.vector : value;
            };
            const weights = (type, data) => {
                switch (type) {
                    case 'convolution': {
                        const weightsShape = [ data.outputChannels, data.kernelChannels, data.kernelSize[0], data.kernelSize[1] ];
                        if (data.isDeconvolution) {
                            weightsShape[0] = data.kernelChannels;
                            weightsShape[1] = Math.floor(data.outputChannels / (data.nGroups != 0 ? data.nGroups : 1));
                        }
                        initializer(type, 'weights', weightsShape, data.weights);
                        if (data.hasBias) {
                            initializer(type, 'bias', [ data.outputChannels ], data.bias);
                        }
                        return { 'weights': true, 'bias': data.hasBias };
                    }
                    case 'innerProduct':
                        initializer(type, 'weights', [ data.outputChannels, data.inputChannels ], data.weights);
                        if (data.hasBias) {
                            initializer(type, 'bias', [ data.outputChannels ], data.bias);
                        }
                        return { 'weights': true, 'bias': data.hasBias };
                    case 'batchnorm':
                        initializer(type, 'gamma', [ data.channels ], data.gamma);
                        initializer(type, 'beta', [ data.channels ], data.beta);
                        if (data.mean) {
                            initializer(type, 'mean', [ data.channels ], data.mean);
                        }
                        if (data.variance) {
                            initializer(type, 'variance', [ data.channels ], data.variance);
                        }
                        return { 'gamma': true, 'beta': true, 'mean': true, 'variance': true };
                    case 'embedding':
                        initializer(type, 'weights', [ data.inputDim, data.outputChannels ], data.weights);
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
                        initializer(type, 'weights', [ data.outputVectorSize, data.inputVectorSize ], data.weightMatrix);
                        initializer(type, 'recurrent', [ data.outputVectorSize, data.inputVectorSize ], data.recursionMatrix);
                        if (data.hasBiasVectors) {
                            initializer(type, 'bias', [ data.outputVectorSize ], data.biasVector);
                        }
                        return { 'weightMatrix': true, 'recursionMatrix': true, 'biasVector': data.hasBiasVectors };
                    case 'gru': {
                        const recursionMatrixShape = [ data.outputVectorSize, data.outputVectorSize ];
                        const weightMatrixShape = [ data.outputVectorSize, data.inputVectorSize ];
                        const biasVectorShape = [ data.outputVectorSize ];
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
                        const count = (type == 'uniDirectionalLSTM') ? 1 : 2;
                        const h = data.outputVectorSize;
                        const x = data.inputVectorSize;
                        for (let i = 0; i < count; i++) {
                            const weights = count == 1 ? data.weightParams : data.weightParams[i];
                            const suffix = (i == 0) ? '' : '_rev';
                            initializer(type, 'inputGateWeightMatrix' + suffix, [h,x], weights.inputGateWeightMatrix);
                            initializer(type, 'forgetGateWeightMatrix' + suffix, [h,x], weights.forgetGateWeightMatrix);
                            initializer(type, 'blockInputWeightMatrix' + suffix, [h,x], weights.blockInputWeightMatrix);
                            initializer(type, 'outputGateWeightMatrix' + suffix, [h,x], weights.outputGateWeightMatrix);
                            initializer(type, 'inputGateRecursionMatrix' + suffix, [h,h], weights.inputGateRecursionMatrix);
                            initializer(type, 'forgetGateRecursionMatrix' + suffix, [h,h],weights.forgetGateRecursionMatrix);
                            initializer(type, 'blockInputRecursionMatrix' + suffix, [h,h], weights.blockInputRecursionMatrix);
                            initializer(type, 'outputGateRecursionMatrix' + suffix, [h,h], weights.outputGateRecursionMatrix);
                            if (data.params.hasBiasVectors) {
                                initializer(type, 'inputGateBiasVector' + suffix, [h], weights.inputGateBiasVector);
                                initializer(type, 'forgetGateBiasVector' + suffix, [h], weights.forgetGateBiasVector);
                                initializer(type, 'blockInputBiasVector' + suffix, [h], weights.blockInputBiasVector);
                                initializer(type, 'outputGateBiasVector' + suffix, [h], weights.outputGateBiasVector);
                            }
                            if (data.params.hasPeepholeVectors) {
                                initializer(type, 'inputGatePeepholeVector' + suffix, [h], weights.inputGatePeepholeVector);
                                initializer(type, 'forgetGatePeepholeVector' + suffix, [h], weights.forgetGatePeepholeVector);
                                initializer(type, 'outputGatePeepholeVector' + suffix, [h], weights.outputGatePeepholeVector);
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
                const map = weights(type, data, initializers);
                for (const entry of Object.entries(data)) {
                    const name = entry[0];
                    if (!map[name]) {
                        obj.attributes[name] = entry[1];
                    }
                }
            }
            const metadata = this._metadata.type(type);
            for (let i = 0; i < inputs.length;) {
                const input = metadata && metadata.inputs && i < metadata.inputs.length ? metadata.inputs[i] : { name: i === 0 ? 'input' : i.toString() };
                const count = input.type === 'Tensor[]' ? inputs.length - i : 1;
                const values = inputs.slice(i, i + count);
                obj.inputs.push(new coreml.Argument(input.name, true, values));
                i += count;
            }
            obj.inputs.push(...initializers);
            for (let i = 0; i < outputs.length;) {
                const output = metadata && metadata.outputs && i < metadata.outputs.length ? metadata.outputs[i] : { name: i === 0 ? 'output' : i.toString() };
                const count = output.type === 'Tensor[]' ? outputs.length - i : 1;
                const args = outputs.slice(i, i + count);
                obj.outputs.push(new coreml.Argument(output.name, true, args));
                i += count;
            }
            const node = new coreml.Node(this._metadata, obj);
            this._nodes.push(node);
            return node;
        };
        const updateClassifierOutput = (args, group, classifier) => {
            let labelProbabilityLayerName = classifier.labelProbabilityLayerName;
            if (!labelProbabilityLayerName && this._nodes.length > 0) {
                const node = this._nodes.slice(-1).pop();
                if (node && node.outputs.length == 1 && node.outputs[0].value.length == 1) {
                    labelProbabilityLayerName = node.outputs[0].value[0].name;
                }
            }
            let predictedFeatureName = this._description.predictedFeatureName;
            let predictedProbabilitiesName = this._description.predictedProbabilitiesName;
            if ((predictedFeatureName || predictedProbabilitiesName) && labelProbabilityLayerName && classifier.ClassLabels) {
                predictedFeatureName = predictedFeatureName ? predictedFeatureName : '?';
                predictedProbabilitiesName = predictedProbabilitiesName ? predictedProbabilitiesName : '?';
                const labelProbabilityInput = labelProbabilityLayerName + ':labelProbabilityLayerName';
                for (const node of this._nodes) {
                    for (const output of node.outputs) {
                        for (const value of output.value) {
                            if (value.name === labelProbabilityLayerName) {
                                value.name = labelProbabilityInput;
                            }
                        }
                    }
                }
                const type = classifier.ClassLabels;
                const inputs = [
                    new coreml.Argument('input', true, [ new coreml.Value(labelProbabilityInput) ])
                ];
                const outputs = [
                    new coreml.Argument('probabilities', true, [ args.output(predictedProbabilitiesName) ]),
                    new coreml.Argument('feature', true, [ args.output(predictedFeatureName) ])
                ];
                const node = new coreml.Node(this._metadata, {
                    group: this._group,
                    type: type,
                    name: null,
                    description: '',
                    attributes: classifier[type] || {},
                    inputs: inputs,
                    outputs: outputs
                });
                this._nodes.push(node);
            }
        };
        const updatePreprocessing = (args, group, preprocessing) => {
            if (preprocessing && preprocessing.length > 0) {
                const preprocessingInput = this._description.input[0].name;
                const inputNodes = [];
                for (const node of this._nodes) {
                    if (node.inputs.some((input) => input.value.some((arg) => arg.name == preprocessingInput))) {
                        inputNodes.push(node);
                    }
                }
                let currentOutput = preprocessingInput;
                let preprocessorOutput = null;
                let preprocessorIndex = 0;
                for (const p of preprocessing) {
                    const input = p.featureName ? p.featureName : currentOutput;
                    currentOutput = preprocessingInput + ':' + preprocessorIndex.toString();
                    const node = createNode(values, group, p.preprocessor, null, '', p[p.preprocessor], [ input ], [ currentOutput ]);
                    preprocessorOutput = node.outputs[0].value[0];
                    preprocessorIndex++;
                }
                for (const node of inputNodes) {
                    for (const input of node.inputs) {
                        for (let i = 0; i < input.value.length; i++) {
                            if (input.value[i].name === preprocessingInput) {
                                input.value[i] = preprocessorOutput;
                            }
                        }
                    }
                }
            }
        };
        const loadModel = (model, values, group, weights) => {
            this._groups = this._groups | (group.length > 0 ? true : false);
            const description = model && model.description && model.description.metadata && model.description.metadata.shortDescription ? model.description.metadata.shortDescription : '';
            switch (model.Type) {
                case 'neuralNetworkClassifier': {
                    const neuralNetworkClassifier = model.neuralNetworkClassifier;
                    for (const layer of neuralNetworkClassifier.layers) {
                        createNode(values, group, layer.layer, layer.name, group === '' ? '' : description, layer[layer.layer], layer.input, layer.output, layer.inputTensor, layer.outputTensor);
                    }
                    updateClassifierOutput(values, group, neuralNetworkClassifier);
                    updatePreprocessing(values, group, neuralNetworkClassifier.preprocessing);
                    return 'Neural Network Classifier';
                }
                case 'neuralNetwork': {
                    const neuralNetwork = model.neuralNetwork;
                    for (const layer of neuralNetwork.layers) {
                        createNode(values, group, layer.layer, layer.name, group === '' ? '' : description, layer[layer.layer], layer.input, layer.output, layer.inputTensor, layer.outputTensor);
                    }
                    updatePreprocessing(values, group, neuralNetwork.preprocessing);
                    return 'Neural Network';
                }
                case 'neuralNetworkRegressor': {
                    const neuralNetworkRegressor = model.neuralNetworkRegressor;
                    for (const layer of neuralNetworkRegressor.layers) {
                        createNode(values, group, layer.layer, layer.name, description, layer[layer.layer], layer.input, layer.output);
                    }
                    updatePreprocessing(values, group, neuralNetworkRegressor);
                    return 'Neural Network Regressor';
                }
                case 'pipeline': {
                    for (let i = 0; i < model.pipeline.models.length; i++) {
                        loadModel(model.pipeline.models[i], values, (group ? (group + '/') : '') + 'pipeline[' + i.toString() + ']');
                    }
                    return 'Pipeline';
                }
                case 'pipelineClassifier': {
                    for (let i = 0; i < model.pipelineClassifier.pipeline.models.length; i++) {
                        loadModel(model.pipelineClassifier.pipeline.models[i], values, (group ? (group + '/') : '') + 'pipelineClassifier[' + i.toString() + ']');
                    }
                    return 'Pipeline Classifier';
                }
                case 'pipelineRegressor': {
                    for (let i = 0; i < model.pipelineRegressor.pipeline.models.length; i++) {
                        loadModel(model.pipelineRegressor.pipeline.models[i], values, (group ? (group + '/') : '') + 'pipelineRegressor[' + i.toString() + ']');
                    }
                    return 'Pipeline Regressor';
                }
                case 'glmClassifier': {
                    createNode(values, group, 'glmClassifier', null, description,
                        {
                            classEncoding: model.glmClassifier.classEncoding,
                            offset: model.glmClassifier.offset,
                            weights: model.glmClassifier.weights
                        },
                        [ model.description.input[0].name ],
                        [ model.description.predictedProbabilitiesName ]);
                    updateClassifierOutput(values, group, model.glmClassifier);
                    return 'Generalized Linear Classifier';
                }
                case 'glmRegressor': {
                    createNode(values, group, 'glmRegressor', null, description,
                        model.glmRegressor,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Generalized Linear Regressor';
                }
                case 'treeEnsembleClassifier': {
                    createNode(values, group, 'treeEnsembleClassifier', null, description,
                        model.treeEnsembleClassifier.treeEnsemble,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    updateClassifierOutput(values, group, model.treeEnsembleClassifier);
                    return 'Tree Ensemble Classifier';
                }
                case 'treeEnsembleRegressor': {
                    createNode(values, group, 'treeEnsembleRegressor', null, description,
                        model.treeEnsembleRegressor.treeEnsemble,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Tree Ensemble Regressor';
                }
                case 'supportVectorClassifier': {
                    createNode(values, group, 'supportVectorClassifier', null, description,
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
                    updateClassifierOutput(values, group, model.supportVectorClassifier);
                    return 'Support Vector Classifier';
                }
                case 'supportVectorRegressor': {
                    createNode(values, group, 'supportVectorRegressor', null, description,
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
                case 'oneHotEncoder': {
                    const categoryType = model.oneHotEncoder.CategoryType;
                    const oneHotEncoderParams = { outputSparse: model.oneHotEncoder.outputSparse };
                    oneHotEncoderParams[categoryType] = model.oneHotEncoder[categoryType];
                    createNode(values, group, 'oneHotEncoder', null, description,
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
                    createNode(values, group, 'oneHotEncoder', null, description,
                        imputerParams,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Imputer';
                }
                case 'featureVectorizer': {
                    createNode(values, group, 'featureVectorizer', null, description,
                        model.featureVectorizer,
                        model.description.input.map((item) => item.name),
                        [ model.description.output[0].name ]);
                    return 'Feature Vectorizer';
                }
                case 'dictVectorizer': {
                    createNode(values, group, 'dictVectorizer', null, description,
                        model.dictVectorizer,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Dictionary Vectorizer';
                }
                case 'scaler': {
                    createNode(values, group, 'scaler', null, description,
                        model.scaler,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Scaler';
                }
                case 'categoricalMapping': {
                    createNode(values, group, 'categoricalMapping', null, description,
                        model.categoricalMapping,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Categorical Mapping';
                }
                case 'normalizer': {
                    createNode(values, group, 'normalizer', null, description,
                        model.normalizer,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Normalizer';
                }
                case 'arrayFeatureExtractor': {
                    createNode(values, group, 'arrayFeatureExtractor', null, description,
                        { extractIndex: model.arrayFeatureExtractor.extractIndex },
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Array Feature Extractor';
                }
                case 'nonMaximumSuppression': {
                    const nonMaximumSuppressionParams = {
                        pickTop: model.nonMaximumSuppression.pickTop,
                        stringClassLabels: model.nonMaximumSuppression.stringClassLabels,
                        iouThreshold: model.nonMaximumSuppression.iouThreshold,
                        confidenceThreshold: model.nonMaximumSuppression.confidenceThreshold
                    };
                    createNode(values, group, 'nonMaximumSuppression', null, description,
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
                    createNode(values, group, 'wordTagger', null, description,
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
                    createNode(values, group, 'textClassifier', null, description,
                        model.textClassifier,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Text Classifier';
                }
                case 'visionFeaturePrint': {
                    const visionFeaturePrintParams = {
                        scene: model.visionFeaturePrint.scene
                    };
                    createNode(values, group, 'visionFeaturePrint', null, description,
                        visionFeaturePrintParams,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Vision Feature Print';
                }
                case 'soundAnalysisPreprocessing': {
                    createNode(values, group, 'soundAnalysisPreprocessing', null, description,
                        model.soundAnalysisPreprocessing,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Sound Analysis Preprocessing';
                }
                case 'kNearestNeighborsClassifier': {
                    createNode(values, group, 'kNearestNeighborsClassifier', null, description,
                        model.kNearestNeighborsClassifier,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    updateClassifierOutput(values, group, model.kNearestNeighborsClassifier);
                    return 'Nearest Neighbors Classifier';
                }
                case 'itemSimilarityRecommender': {
                    createNode(values, group, 'itemSimilarityRecommender', null, description,
                        {
                            itemStringIds: model.itemSimilarityRecommender.itemStringIds.vector,
                            itemItemSimilarities: model.itemSimilarityRecommender.itemItemSimilarities
                        },
                        model.description.input.map((feature) => feature.name),
                        model.description.output.map((feature) => feature.name));
                    return 'Item Similarity Recommender';
                }
                case 'audioFeaturePrint': {
                    createNode(values, group, 'audioFeaturePrint', null, description,
                        model.audioFeaturePrint,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Audio Feature Print';
                }
                case 'linkedModel': {
                    createNode(values, group, 'linkedModel', null, description,
                        model.linkedModel.linkedModelFile,
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'Linked Model';
                }
                case 'customModel': {
                    createNode(values, group, 'customModel', null, description,
                        { className: model.customModel.className, parameters: model.customModel.parameters },
                        [ model.description.input[0].name ],
                        [ model.description.output[0].name ]);
                    return 'customModel';
                }
                case 'mlProgram': {
                    return loadProgram(model.mlProgram, values, group, weights);
                }
                default: {
                    throw new coreml.Error("Unsupported model type '" + JSON.stringify(Object.keys(model)) + "'.");
                }
            }
        };
        const values = new Map();
        values.input = (name) => {
            if (!values.has(name)) {
                values.set(name, { counter: 0, value: new coreml.Value(name) });
            }
            return values.get(name).value;
        };
        values.output = (name) => {
            if (!values.has(name)) {
                const entry = { counter: 0, value: new coreml.Value(name) };
                values.set(name, entry);
            } else {
                const entry = values.get(name);
                entry.counter++;
                const next = name + '\n' + entry.counter.toString(); // custom argument id
                entry.value = new coreml.Value(next);
            }
            return values.get(name).value;
        };
        const update = (value, description) => {
            if (!value.type) {
                value.type = coreml.Utility.featureType(description.type);
            }
            if (!value.description && description.shortDescription) {
                value.description = description.shortDescription;
            }
            return value;
        };
        if (this._description) {
            this._inputs = this._description.input.map((input) => {
                const value = values.output(input.name);
                update(value, input);
                return new coreml.Argument(input.name, true, [ value ]);
            });
        }
        this._type = loadModel(model, values, '', weights);
        if (this._description) {
            this._outputs = this._description.output.map((output) => {
                const value = values.input(output.name);
                update(value, output);
                return new coreml.Argument(output.name, true, [ value ]);
            });
        }
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
};

coreml.Argument = class {

    constructor(name, visible, value) {
        this._name = name;
        this._visible = visible;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get value() {
        return this._value;
    }
};

coreml.Value = class {

    constructor(name, type, description, initializer) {
        if (typeof name !== 'string') {
            throw new coreml.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
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

    set type(value) {
        this._type = value;
    }

    get description() {
        return this._description;
    }

    set description(value) {
        this._description = value;
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

    constructor(metadata, obj) {
        if (!obj.type) {
            throw new Error('Undefined node type.');
        }
        if (obj.group) {
            this._group = obj.group;
        }
        this._type = Object.assign({}, metadata.type(obj.type) || { name: obj.type });
        this._type.name = obj.type.split(':').pop();
        this._name = obj.name || '';
        this._description = obj.description || '';
        this._inputs = obj.inputs || [];
        this._outputs = obj.outputs || [];
        this._attributes = [];
        for (const entry of Object.entries(obj.attributes)) {
            const attribute = new coreml.Attribute(metadata.attribute(obj.type, entry[0]), entry[0], entry[1]);
            this._attributes.push(attribute);
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

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        if (this._value instanceof coreml.Tensor) {
            this._type = 'tensor';
        }
        if (metadata) {
            if (metadata.type) {
                this._type = metadata.type;
            }
            if (this._type && coreml.proto) {
                this._value = coreml.Utility.enum(this._type, this._value);
            }
            if (metadata.visible === false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (Array.isArray(value)) {
                    value = value.map((item) => item.toNumber());
                }
                if (JSON.stringify(metadata.default) == JSON.stringify(value)) {
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

    constructor(type, data, quantization, category) {
        this._type = type;
        this._data = data;
        this._quantization = quantization;
        this._category = category;
    }

    get category() {
        return this._category;
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

    get layout() {
        switch (this._type.dataType) {
            case 'float32': return '|';
            default: return '<';
        }
    }

    get values() {
        return this._data;
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

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

coreml.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions.map((dim) => typeof dim === 'string' || Number.isInteger(dim) ? dim : dim.toNumber());
    }

    get dimensions() {
        return this._dimensions;
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) && Array.isArray(this._dimensions) &&
            this._dimensions.length === obj.dimensions.length &&
            obj.dimensions.every((value, index) => this._dimensions[index] === value);
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

    get elementType() {
        return this._elementType;
    }

    equals(obj) {
        return obj instanceof coreml.ListType && this.elementType.equals(obj.elementType);
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

coreml.SequenceType = class {

    constructor(type) {
        this._type = type;
    }

    get type() {
        return this._type;
    }

    toString() {
        return 'sequence<' + this._type + '>';
    }
};

coreml.ImageType = class {

    constructor(colorSpace, width, height) {
        this._width = width;
        this._height = height;
        switch (colorSpace) {
            case coreml.proto.ImageFeatureType.ColorSpace.GRAYSCALE:
                this._colorSpace = 'grayscale';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.RGB:
                this._colorSpace = 'RGB';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.BGR:
                this._colorSpace = 'BGR';
                break;
            case coreml.proto.ImageFeatureType.ColorSpace.GRAYSCALE_FLOAT16:
                this._colorSpace = 'grayscale:float16';
                break;
            default:
                throw new coreml.Error("Unsupported image color space '" + colorSpace + "'.");
        }
    }

    toString() {
        return 'image<' + this._colorSpace + ',' + this._width. toString() + 'x' + this._height.toString() + '>';
    }
};

coreml.OptionalType = class {

    constructor(type) {
        this._type = type;
    }

    get type() {
        return this._type;
    }

    toString() {
        return 'optional<' + this._type.toString() + '>';
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
                        shape = new coreml.TensorShape(type.multiArrayType.shape.map((dim) => dim.toNumber()));
                    }
                    let dataType;
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
                            throw new coreml.Error("Unsupported array data type '" + type.multiArrayType.dataType + "'.");
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
                    throw new coreml.Error("Unsupported feature type '" + type.Type + "'.");
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
            coreml.Utility._dataTypes = new Map(Object.entries(coreml.proto.MILSpec.DataType).map((entry => [entry[1], entry[0].toLowerCase()])));
            coreml.Utility._dataTypes.delete(0);
            coreml.Utility._dataTypes.set(1, 'boolean');
        }
        const shape = type.dimensions.map(dim => dim.constant ? dim.constant.size : '?');
        const dataType = coreml.Utility._dataTypes.get(type.dataType);
        if (!dataType) {
            throw new coreml.Error("Unsupported data type '" + type.dataType + "'.");
        }
        return new coreml.TensorType(dataType, new coreml.TensorShape(shape));
    }

    static valueType(type) {
        switch (type.type) {
            case 'tensorType':
                return coreml.Utility.tensorType(type.tensorType);
            case 'listType':
                return new coreml.ListType(coreml.Utility.valueType(type.listType.type));
            default:
                throw new coreml.Error("Unsupported value type '" + type.type + "'.");
        }
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