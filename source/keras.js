/* jshint esversion: 6 */

var keras = keras || {};
var json = json || require('./json');

keras.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A ];
        if (stream.length > signature.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            return 'keras.h5';
        }
        const obj = context.open('json');
        if (obj) {
            if (obj.mxnet_version || (obj.nodes && obj.arg_nodes && obj.heads)) {
                return undefined;
            }
            if (obj.modelTopology && (obj.format === 'layers-model' || obj.modelTopology.class_name || obj.modelTopology.model_config)) {
                return 'keras.json.tfjs';
            }
            if (obj.model_config || (obj.class_name && obj.config)) {
                return 'keras.json';
            }
            if (Array.isArray(obj) && obj.every((item) => item.weights && item.paths)) {
                return 'keras.json.tfjs.weights';
            }
        }
        return undefined;
    }

    open(context, match) {
        const openModel = (format, producer, backend, config, weights) => {
            return keras.Metadata.open(context).then((metadata) => {
                return new keras.Model(metadata, format, producer, backend, config, weights);
            });
        };
        const openShards = (manifests, shards) => {
            const weights = new keras.Weights();
            const dtype_size_map = new Map([ [ 'float16', 2 ], [ 'float32', 4 ], [ 'float64', 8 ], [ 'int8', 1 ], [ 'int16', 2 ], [ 'int32', 4 ], [ 'int64', 8 ], [ 'uint8', 1 ], [ 'uint16', 2 ], [ 'uint32', 4 ], [ 'uint64', 8 ] ]);
            for (const manifest of manifests) {
                let buffer = null;
                if (Array.isArray(manifest.paths) && manifest.paths.length > 0 && manifest.paths.every((path) => shards.has(path))) {
                    const list = manifest.paths.map((path) => shards.get(path));
                    const size = list.reduce((a, b) => a + b.length, 0);
                    buffer = new Uint8Array(size);
                    let offset = 0;
                    for (const item of list) {
                        buffer.set(item, offset);
                        offset += item.length;
                    }
                }
                let offset = 0;
                for (const weight of manifest.weights) {
                    const dtype = weight.quantization && weight.quantization.dtype ? weight.quantization.dtype : weight.dtype;
                    if (!dtype_size_map.has(dtype)) {
                        throw new keras.Error("Unknown weight data type size '" + dtype + "'.");
                    }
                    const itemsize = dtype_size_map.get(dtype);
                    const size = weight.shape.reduce((a, b) => a * b, 1);
                    const length = itemsize * size;
                    const data = buffer ? buffer.slice(offset, offset + length) : null;
                    weights.add(weight.identifier, new keras.Tensor(weight.name, weight.shape, dtype, weight.quantization, true, data));
                    offset += length;
                }
            }
            return Promise.resolve(weights);
        };
        const openManifests = (manifests) => {
            const shards = new Map();
            for (const manifest of manifests) {
                for (const path of manifest.paths) {
                    if (!shards.has(path)) {
                        shards.set(path, context.request(path, null));
                    }
                }
            }
            const promises = shards.values();
            return Promise.all(promises).then((streams) => {
                for (const key of shards.keys()) {
                    shards.set(key, streams.shift().peek());
                }
                return openShards(manifests, shards);
            }).catch(() => {
                shards.clear();
                return openShards(manifests, shards);
            });
        };
        const stream = context.stream;
        switch (match) {
            case 'keras.h5': {
                return context.require('./hdf5').then((hdf5) => {
                    const find_root_group = (file) => {
                        const root_group = file.rootGroup;
                        const kerasmodel = root_group.group('model/kerasmodel');
                        if (kerasmodel && kerasmodel.attributes.has('model_config')) {
                            return kerasmodel;
                        }
                        return root_group;
                    };
                    const read_model_config = (group) => {
                        if (group.attributes.has('model_config')) {
                            const buffer = group.attributes.get('model_config');
                            const reader = json.TextReader.open(buffer);
                            if (reader) {
                                return reader.read();
                            }
                        }
                        return null;
                    };
                    const load_attributes_from_hdf5_group = (group, name) => {
                        if (group.attributes.has(name)) {
                            return group.attributes.get(name);
                        }
                        if (group.attributes.has(name + '0')) {
                            let index = 0;
                            let value = [];
                            while (group.attributes.has(name + index.toString())) {
                                const chunk = group.attributes.get(name + index.toString());
                                value = value.concat(chunk);
                                index++;
                            }
                            return value;
                        }
                        return null;
                    };
                    const weights = new keras.Weights();
                    const file = hdf5.File.open(stream);
                    const root_group = find_root_group(file);
                    const model_config = read_model_config(root_group);
                    if (model_config) {
                        const backend = root_group.attributes.get('backend') || '';
                        const version = root_group.attributes.get('keras_version') || '';
                        const format = 'Keras' + (version ? ' v' + version : '');
                        const model_weights_group = root_group.group('model_weights');
                        if (model_weights_group) {
                            const layer_names = load_attributes_from_hdf5_group(model_weights_group, 'layer_names');
                            for (const layer_name of layer_names) {
                                const layer_weights = model_weights_group.group(layer_name);
                                if (layer_weights) {
                                    const weight_names = load_attributes_from_hdf5_group(layer_weights, 'weight_names');
                                    if (Array.isArray(weight_names) && weight_names.length > 0) {
                                        for (const weight_name of weight_names) {
                                            const weight = layer_weights.group(weight_name);
                                            if (weight && weight.value) {
                                                const variable = weight.value;
                                                const tensor = new keras.Tensor(weight_name, variable.shape, variable.type, null, variable.littleEndian, variable.data);
                                                weights.add(layer_name, tensor);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (!model_config) {
                            throw new keras.Error("'model_config' is not present.");
                        }
                        if (!model_config.class_name) {
                            throw new keras.Error("'class_name' is not present.");
                        }
                        return openModel(format, '', backend, model_config, weights);
                    }
                    const layer_names = load_attributes_from_hdf5_group(root_group, 'layer_names');
                    if (layer_names && Array.isArray(layer_names)) {
                        const version = root_group.attributes.get('keras_version') || '';
                        const format = 'Keras Weights' + (version ? ' v' + version : '');
                        const backend = root_group.attributes.get('backend') || '';
                        for (const layer_name of layer_names) {
                            const layer_weights = root_group.group(layer_name);
                            if (layer_weights) {
                                const weight_names = load_attributes_from_hdf5_group(layer_weights, 'weight_names');
                                if (Array.isArray(weight_names) && weight_names.length > 0) {
                                    for (const weight_name of weight_names) {
                                        const weight = layer_weights.group(weight_name);
                                        if (weight && weight.value) {
                                            const variable = weight.value;
                                            const components = weight_name.split('/');
                                            components.pop();
                                            const name = (components.length == 0 || components[0] !== layer_name) ? [ layer_name ].concat(components).join('/') : components.join('/');
                                            const tensor = new keras.Tensor(weight_name, variable.shape, variable.type, null, variable.littleEndian, variable.data);
                                            weights.add(name, tensor);
                                        }
                                    }
                                }
                            }
                        }
                        return openModel(format, '', backend, null, weights);
                    }
                    else {
                        const rootKeys = new Set(root_group.attributes.keys());
                        rootKeys.delete('nb_layers');
                        if (rootKeys.size > 0 || root_group.value !== null) {
                            throw new keras.Error('File format is not HDF5 Weights');
                        }
                        let format = 'HDF5 Weights';
                        let weights_group = root_group;
                        if (root_group.attributes.size === 0 && root_group.value === null && root_group.groups.size == 1) {
                            const group = root_group.groups.values().next().value;
                            if (group.attributes.size === 0 && group.value === null) {
                                weights_group = group;
                            }
                        }
                        const tensorKeys = new Set([ 'name', 'shape', 'quantization' ]);
                        const groups = Array.from(weights_group.groups.values());
                        if (groups.every((group) => group.attributes.size === 0 && group.groups.length == 0 && group.value !== null)) {
                            for (const group of groups) {
                                const variable = group.value;
                                const tensor = new keras.Tensor(group.name, variable.shape, variable.type, null, variable.littleEndian, variable.type === 'string' ? variable.value : variable.data);
                                weights.add('', tensor);
                            }
                            return openModel(format, '', '', null, weights);
                        }
                        if (groups.every((group) => group.value === null && Array.from(group.attributes.keys()).filter((key) => !tensorKeys.has(key)).length === 0 && Array.from(group.groups.values()).every((variable) => Object.keys(variable.attributes).length === 0 && variable.value !== null))) {
                            for (const group of groups) {
                                const moduleName = group.attributes.has('name') ? group.attributes.get('name') : group.name;
                                for (const variableGroup of group.groups.values()) {
                                    if (variableGroup.attributes.size !== 0 || variableGroup.groups.size !== 0) {
                                        throw new keras.Error('Variable format is not HDF5 Weights');
                                    }
                                    const variable = variableGroup.value;
                                    if (!variable) {
                                        throw new keras.Error('Variable value is not HDF5 Weights');
                                    }
                                    const name = moduleName ? [ moduleName, variableGroup.name ].join('/') : moduleName.name;
                                    const tensor = new keras.Tensor(name, variable.shape, variable.type, null, variable.littleEndian, variable.type === 'string' ? variable.value : variable.data);
                                    weights.add(moduleName, tensor);
                                }
                            }
                            return openModel(format, '', '', null, weights);
                        }
                        const walk = function(group) {
                            if (group.attributes.size === 0 && group.value === null && group.groups.size > 0) {
                                for (const subGroup of group.groups.values()) {
                                    walk(subGroup);
                                }
                                return;
                            }
                            const subKeys = new Set([ 'index', 'need_grad' ]);
                            const attribtues = Array.from(group.attributes.keys());
                            const match = attribtues.filter((key) => !subKeys.has(key)).length === 0;
                            if (match && attribtues.length !== 0) {
                                format = 'nnabla HDF5 Weights';
                            }
                            if (match && group.value !== null && group.groups.size === 0) {
                                const variable = group.value;
                                const variableName = group.path;
                                let moduleName = variableName;
                                const parts = variableName.split('/');
                                if (parts.length > 1) {
                                    parts.pop();
                                    moduleName = parts.join('/');
                                }
                                const tensor = new keras.Tensor(variableName, variable.shape, variable.type, null, variable.littleEndian, variable.type === 'string' ? variable.value : variable.data);
                                weights.add(moduleName, tensor);
                                return;
                            }
                            throw new keras.Error('Module group format is not HDF5 Weights');
                        };
                        walk(weights_group);
                        return openModel(format, '', '', null, weights);
                    }
                });
            }
            case 'keras.json': {
                const obj = context.open('json');
                const format = 'Keras' + (obj.keras_version ? ' v' + obj.keras_version : '');
                const backend = obj.backend || '';
                const config = obj.model_config ? obj.model_config : obj;
                const weights = new keras.Weights();
                return openModel(format, '', backend, config, weights);
            }
            case 'keras.json.tfjs': {
                const obj = context.open('json');
                const modelTopology = obj.modelTopology;
                const backend = modelTopology.backend || '';
                const format = 'TensorFlow.js ' + (obj.format ? obj.format : 'Keras' + (modelTopology.keras_version ? (' v' + modelTopology.keras_version) : ''));
                const producer = obj.convertedBy || obj.generatedBy || '';
                const manifests = obj.weightsManifest;
                for (const manifest of manifests) {
                    for (const weight of manifest.weights) {
                        weight.identifier = '';
                    }
                }
                const model_config = modelTopology.model_config ? modelTopology.model_config : modelTopology;
                return openManifests(manifests).then((weights) => {
                    return openModel(format, producer, backend, model_config, weights);
                });
            }
            case 'keras.json.tfjs.weights': {
                const obj = context.open('json');
                const manifests = [];
                const format = 'TensorFlow.js Weights';
                manifests.push(...obj);
                for (const manifest of manifests) {
                    for (const weight of manifest.weights) {
                        const parts = weight.name.split('/');
                        parts.pop();
                        weight.identifier = parts.join('/');
                    }
                }
                return openManifests(manifests).then((weights) => {
                    return openModel(format, '', '', null, weights);
                });
            }
            default: {
                throw new keras.Error("Unsupported Keras format '" + match + "'.");
            }
        }
    }
};

keras.Model = class {

    constructor(metadata, format, producer, backend, config, weights) {
        this._format = format;
        this._backend = backend;
        this._producer = producer;
        this._graphs = [ new keras.Graph(new keras.GraphMetadata(metadata), config, weights) ];
    }

    get name() {
        return null;
    }

    get description() {
        return null;
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get runtime() {
        return this._backend;
    }

    get graphs() {
        return this._graphs;
    }
};

keras.Graph = class {

    constructor(metadata, config, weights) {
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._groups = false;

        if (config) {
            this._name = config.name || (config.config && config.config.name ? config.config.name : '');
            switch (config.class_name) {
                case 'AllCNN':
                case 'Sequential':
                    this._loadSequential(config.config, weights, '', null, null);
                    break;
                case 'Functional':
                case 'Model':
                    this._loadModel(config.config, weights, '', null, null);
                    break;
                default:
                    throw new keras.Error('\'' + config.class_name + '\' is not supported.');
            }
        }
        else if (weights) {
            for (const layer of weights.keys()) {
                if (weights.get('', layer).length <= 6) {
                    const node = new keras.Node(metadata, 'Weights', { name: layer }, [], [], '', weights);
                    this._nodes.push(node);
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get groups() {
        return this._groups ? true : false;
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

    _loadModel(config, weights, group, inputs, outputs) {
        if (group) {
            this._groups = true;
        }
        const nodeMap = new Map();
        if (config.layers) {
            for (const layer of config.layers) {
                layer._inputs = [];
                layer._outputs = [];
                if (layer.name) {
                    if (!nodeMap.has(layer.name)) {
                        nodeMap.set(layer.name, layer);
                    }
                }
            }
            for (const layer of config.layers) {
                if (layer.inbound_nodes) {
                    for (let inbound_node of layer.inbound_nodes) {
                        const is_connection = (item) => {
                            return Array.isArray(item) && (item.length === 3 || item.length === 4) && typeof item[0] === 'string';
                        };
                        // wrap
                        if (is_connection(inbound_node)) {
                            inbound_node = [ inbound_node ];
                        }
                        // unwrap
                        if (Array.isArray(inbound_node) && inbound_node.every((array) => Array.isArray(array) && array.every((item) => is_connection(item)))) {
                            inbound_node = inbound_node.flat();
                        }
                        for (const inbound_connection of inbound_node) {
                            let inputName = inbound_connection[0];
                            const inputNode = nodeMap.get(inputName);
                            if (inputNode) {
                                const inputIndex = inbound_connection[2];
                                if (inputIndex != 0) {
                                    inputName += ':' + inputIndex.toString();
                                }
                                while (inputIndex >= inputNode._outputs.length) {
                                    inputNode._outputs.push('');
                                }
                                inputNode._outputs[inputIndex] = inputName;
                            }
                            layer._inputs.push(inputName);
                        }
                    }
                }
            }
        }
        const input_layers = config.input_layers;
        if (input_layers) {
            for (let i = 0; i < input_layers.length; i++) {
                const input_layer = input_layers[i];
                const name = input_layer[0];
                let type = null;
                const node = nodeMap.get(name);
                if (node && node.class_name == 'InputLayer') {
                    type = this._getInputType(node);
                    nodeMap.delete(name);
                }
                if (inputs && i < inputs.length) {
                    if (config.layers) {
                        for (const layer of config.layers) {
                            if (layer._inputs) {
                                layer._inputs = layer._inputs.map((input) => {
                                    return input === name ? inputs[i] : input;
                                });
                            }
                        }
                    }
                }
                else {
                    this._inputs.push(new keras.Parameter(name, true, [ new keras.Argument(name, type, null) ]));
                }
            }
        }
        const inputMap = new Map();
        const output_layers = config.output_layers;
        if (output_layers) {
            for (let j = 0; j < output_layers.length; j++) {
                const output_layer = output_layers[j];
                let outputName = output_layer[0];
                const outputNode = nodeMap.get(outputName);
                let addGraphOutput = true;
                if (outputs && j < outputs.length) {
                    inputMap.set(outputName, outputs[j]);
                    outputName = outputs[j];
                    addGraphOutput = false;
                }
                if (outputNode) {
                    const outputIndex = output_layer[2];
                    if (outputIndex != 0) {
                        outputName += ':' + outputIndex.toString();
                    }
                    while (outputIndex >= outputNode._outputs.length) {
                        outputNode._outputs.push('');
                    }
                    outputNode._outputs[outputIndex] = outputName;
                }
                if (addGraphOutput) {
                    this._outputs.push(new keras.Parameter(outputName, true, [ new keras.Argument(outputName, null, null) ]));
                }
            }
        }

        if (config.layers) {
            for (const layer of config.layers) {
                if (nodeMap.has(layer.name)) {
                    this._loadNode(layer, layer._inputs, layer._outputs, weights, group, inputMap);
                }
            }
        }
    }

    _loadSequential(config, weights, group, inputs, outputs) {
        if (group) {
            this._groups = true;
        }
        const inputName = 'input';
        let inputType = null;
        let argument = inputName;
        let index = 0;
        const layers = config.layers ? config.layers : config;
        for (const layer of layers) {
            let name = index.toString();
            let nodeInputs = [ argument ];
            if (index == 0) {
                if (inputs && inputs.length > 0) {
                    nodeInputs = [ inputs[0] ];
                }
                else {
                    inputType = this._getInputType(layer);
                }
            }
            index++;
            if (layer.config && layer.config.name) {
                name = layer.config.name;
            }
            argument = name;
            let nodeOutputs = [ argument ];
            if (index == layers.length) {
                if (outputs && outputs.length > 0) {
                    nodeOutputs = [ outputs[0] ];
                    argument = null;
                }
            }

            this._loadNode(layer, nodeInputs, nodeOutputs, weights, group);
        }
        if (!inputs) {
            this._inputs.push(new keras.Parameter(inputName, true, [ new keras.Argument(inputName, inputType, null) ]));
        }
        if (argument) {
            this._outputs.push(new keras.Parameter(argument, true, [ new keras.Argument(argument, null, null) ]));
        }
    }

    _loadNode(layer, inputs, outputs, weights, group, inputMap) {
        const class_name = layer.class_name;
        switch (class_name) {
            case 'Sequential': {
                const name = layer.name || (layer.config ? layer.config.name : '');
                this._loadSequential(layer.config, weights, (group ? group + '/' : '') + name, inputs, outputs);
                break;
            }
            case 'Functional':
            case 'Model': {
                const name = layer.name || (layer.config ? layer.config.name : '');
                this._loadModel(layer.config, weights, (group ? group + '/' : '') + name, inputs, outputs);
                break;
            }
            default: {
                inputs = inputs.map((input) => inputMap && inputMap.has(input) ? inputMap.get(input) : input);
                const node = new keras.Node(this._metadata, class_name, layer.config, inputs, outputs, group, weights);
                this._nodes.push(node);
                break;
            }
        }
    }

    _getInputType(layer) {
        if (layer && layer.config) {
            let dataType = '?';
            let shape = [];
            const config = layer.config;
            if (config.dtype) {
                dataType = config.dtype;
                delete config.dtype;
            }
            if (config.batch_input_shape) {
                shape = config.batch_input_shape.map(s => s == null ? '?' : s);
                delete config.batch_input_shape;
            }
            return new keras.TensorType(dataType, new keras.TensorShape(shape));
        }
        return null;
    }
};

keras.Parameter = class {

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

keras.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new keras.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name= name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
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

keras.Node = class {

    constructor(metadata, type, config, inputs, outputs, group, weights) {
        this._group = group || '';
        const name = config && config.name ? config.name : '';
        this._name = (this._group ? this._group + '/' : '') + name;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._chain = [];

        let names = [ name ];
        if ((type == 'Bidirectional' || type == 'TimeDistributed') && (config && config.layer)) {
            const inner = config.layer;
            delete config.layer;
            this._inner = new keras.Node(metadata, inner.class_name, inner.config, [], [], null, null);
            if (type == 'Bidirectional' && inner.config.name) {
                names = [ name + '/forward_' + inner.config.name, name + '/backward_' + inner.config.name ];
                if (!group) {
                    group = name;
                }
            }
        }

        const initializers = {};
        if (weights) {
            for (const name of names) {
                for (const initializer of weights.get(group, name)) {
                    inputs.push(initializer.name);
                    initializers[initializer.name] = initializer;
                }
            }
        }

        if (config) {
            for (const name of Object.keys(config)) {
                const value = config[name];
                if (name === 'activation' && value !== 'linear') {
                    if (typeof value === 'string') {
                        const set = new Map([ [ 'elu', 'ELU' ], [ 'exponential', 'Exponential' ], [ 'hard_sigmoid', 'HardSigmoid' ], [ 'linear', 'Linear' ], [ 'relu', 'ReLU' ], [ 'selu', 'SELU' ], [ 'softmax', 'Softmax'], [ 'sigmoid', 'Sigmoid' ], [ 'softplus', 'Softplus' ], [ 'softsign', 'Softsign' ], [ 'tanh', 'TanH' ] ]);
                        const type = set.has(value) ? set.get(value) : value;
                        this.chain.push(new keras.Node(metadata, type, {}, [], [], null, null));
                    }
                    else if (value && typeof value.class_name === 'string' && value.config) {
                        const type = value.class_name;
                        if (!metadata.type(type)) {
                            metadata.add(type, { category: 'Activation' });
                        }
                        this.chain.push(new keras.Node(metadata, type, value.config, [], [], null, null));
                    }
                }
                if (name !== 'name' && value !== null) {
                    const attribute = new keras.Attribute(metadata.attribute(type, name), name, value);
                    this._attributes.push(attribute);
                }
            }
        }

        this._type = metadata.type(type) || { name: type };
        const innerType = this.inner ? this.inner.type : null;
        const innerSchema = innerType ? metadata.type(innerType) : null;
        let inputIndex = 0;
        while (inputs.length > 0) {
            let list = false;
            let inputName = null;
            let visible = true;
            if (!innerSchema || inputIndex == 0) {
                if (this._type && this._type.inputs && inputIndex < this._type.inputs.length) {
                    const input = this._type.inputs[inputIndex];
                    inputName = input.name;
                    if (type === 'BatchNormalization' && inputName === 'gamma' && config.scale === false) {
                        inputIndex++;
                        continue;
                    }
                    visible = input.visible == false ? false : true;
                    if (this._type.inputs[inputIndex].list) {
                        list = true;
                    }
                }
            }
            else {
                switch (type) {
                    case 'Bidirectional': {
                        let innerIndex = inputIndex;
                        if (innerSchema && innerSchema.inputs) {
                            if (innerIndex < innerSchema.inputs.length) {
                                inputName = 'forward_' + innerSchema.inputs[innerIndex].name;
                            }
                            else {
                                innerIndex = innerIndex - innerSchema.inputs.length + 1;
                                if (innerIndex < innerSchema.inputs.length) {
                                    inputName = 'backward_' + innerSchema.inputs[innerIndex].name;
                                }
                            }
                        }
                        visible = false;
                        break;
                    }
                    case 'TimeDistributed':
                        if (innerSchema && innerSchema.inputs && inputIndex < innerSchema.inputs.length) {
                            inputName = innerSchema.inputs[inputIndex].name;
                        }
                        break;
                }
            }
            const input = !list ? [ inputs.shift() ] : inputs.splice(0, inputs.length);
            const inputArguments = input.map((id) => {
                if (typeof id === 'string') {
                    return new keras.Argument(id, null, initializers[id]);
                }
                if (Array.isArray(id) && id.every((item) => Array.isArray(item) && item.length === 4 && item[0] === '_CONSTANT_VALUE' && item[1] === -1)) {
                    return new keras.Argument('', null, new keras.Tensor());
                }
                throw new keras.Error("Invalid argument '" + JSON.stringify(id) + "'.");
            });
            if (!inputName && inputArguments.length == 1 && inputArguments[0].initializer && inputArguments[0].initializer.name) {
                if (names.length === 1 && names[0] === '') {
                    inputName = inputArguments[0].initializer.name;
                }
                else {
                    const parts = inputArguments[0].initializer.name.split('/').pop().split(':').shift().split('_');
                    const inputName1 = parts.pop();
                    const inputName2 = parts.length > 0 ? [ parts.pop(), inputName1 ].join('_') : '';
                    const inputNames = new Set([ 'recurrent_kernel', 'running_mean', 'running_std', 'moving_mean', 'moving_variance', 'depthwise_filter', 'pointwise_filter' ]);
                    inputName = inputNames.has(inputName2) ? inputName2 : inputName1;
                }
            }
            this._inputs.push(new keras.Parameter(inputName || inputIndex.toString(), visible, inputArguments));
            inputIndex++;
        }

        this._outputs = outputs.map((output, outputIndex) => {
            const outputName =
                (this._type && this._type.outputs && outputIndex < this._type.outputs.length && this._type.outputs[outputIndex] && this._type.outputs[outputIndex].name) ?
                    this._type.outputs[outputIndex].name :
                    outputIndex.toString();
            return new keras.Parameter(outputName, true, [ new keras.Argument(output, null, null) ]);
        });

        if (typeof this.type.name !== 'string' || !this.type.name.split) { // #416
            throw new keras.Error("Unknown node type '" + JSON.stringify(this.type.name) + "'.");
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group;
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

    get chain() {
        return this._chain;
    }

    get inner() {
        return this._inner;
    }
};

keras.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        if (typeof value == 'object' && value.class_name && value.config) {
            this._value = keras.Attribute._convert(value);
        }
        switch (name) {
            case 'trainable':
                this._type = 'boolean';
                this._visible = false;
                break;
            case 'dtype':
                this._visible = false;
                break;
            default: {
                if (metadata) {
                    if (metadata.type) {
                        this._type = metadata.type;
                    }
                    if (Object.prototype.hasOwnProperty.call(metadata, 'visible')) {
                        this._visible = metadata.visible;
                    }
                    else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                        if (keras.Attribute._isEquivalent(metadata.default, value)) {
                            this._visible = false;
                        }
                    }
                }
                break;
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

    static _convert(value) {
        if (Array.isArray(value) || value !== Object(value)) {
            return value;
        }
        const obj = {};
        if (value.class_name) {
            obj.__type__ = value.class_name;
        }
        if (value.config) {
            for (const entry of Object.entries(value.config)) {
                const key = entry[0];
                const value = entry[1];
                obj[key] = keras.Attribute._convert(value);
            }
        }
        return obj;
    }

    static _isEquivalent(a, b) {
        if (a === b) {
            return a !== 0 || 1 / a === 1 / b;
        }
        if (a == null || b == null) {
            return false;
        }
        if (a !== a) {
            return b !== b;
        }
        const type = typeof a;
        if (type !== 'function' && type !== 'object' && typeof b != 'object') {
            return false;
        }
        const className = toString.call(a);
        if (className !== toString.call(b)) {
            return false;
        }
        switch (className) {
            case '[object RegExp]':
            case '[object String]':
                return '' + a === '' + b;
            case '[object Number]':
                if (+a !== +a) {
                    return +b !== +b;
                }
                return +a === 0 ? 1 / +a === 1 / b : +a === +b;
            case '[object Date]':
            case '[object Boolean]':
                return +a === +b;
            case '[object Array]': {
                let length = a.length;
                if (length !== b.length) {
                    return false;
                }
                while (length--) {
                    if (!keras.Attribute._isEquivalent(a[length], b[length])) {
                        return false;
                    }
                }
                return true;
            }
        }

        const keys = Object.keys(a);
        let size = keys.length;
        if (Object.keys(b).length != size) {
            return false;
        }
        while (size--) {
            const key = keys[size];
            if (!(Object.prototype.hasOwnProperty.call(b, key) && keras.Attribute._isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
};

keras.Tensor = class {

    constructor(name, shape, type, quantization, littleEndian, data) {
        this._name = name;
        this._type = new keras.TensorType(type, new keras.TensorShape(shape));
        this._quantization = quantization;
        this._littleEndian = littleEndian;
        this._data = data;
    }

    get kind() {
        return 'Weights';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        if (this._quantization && (this._quantization.scale !== 0 || this._quantization.min !== 0)) {
            const scale = this._quantization.scale || 0;
            const min = this._quantization.min || 0;
            return scale.toString() + ' * ' + (min == 0 ? 'q' : ('(q - ' + min.toString() + ')'));
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
        return keras.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        switch (this._type.dataType) {
            case 'boolean':
            case 'float16':
            case 'float32':
            case 'float64':
            case 'uint8':
            case 'int32':
            case 'int64':
                context.dataType = this._type.dataType;
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                context.littleEndian = this._littleEndian;
                break;
            case 'string':
                context.dataType = this._type.dataType;
                context.data = this._data;
                break;
            default:
                context.state = 'Tensor data type is not supported.';
                break;
        }
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        const littleEndian = context.littleEndian;
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push(null);
                    return results;
                }
                switch (context.dataType) {
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, littleEndian));
                        context.index += 2;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, littleEndian));
                        context.index += 4;
                        break;
                    case 'float64':
                        results.push(context.data.getFloat64(context.index, littleEndian));
                        context.index += 8;
                        break;
                    case 'boolean':
                        results.push(context.data.getInt8(context.index) !== 0);
                        context.index += 1;
                        break;
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, littleEndian));
                        context.index += 4;
                        break;
                    case 'int64':
                        results.push(context.data.getInt64(context.index, littleEndian));
                        context.index += 8;
                        break;
                    case 'string':
                        results.push(context.data[context.index]);
                        context.index++;
                        break;
                }
                context.count++;
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push(null);
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => keras.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value === null) {
            return indentation + '...';
        }
        if (typeof value == 'string') {
            return indentation + '"' + value + '"';
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

keras.TensorType = class {

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
        return this._dataType + this._shape.toString();
    }
};

keras.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

keras.GraphMetadata = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._map = new Map();
    }

    type(name) {
        if (this._map.has(name)) {
            return this._map.get(name);
        }
        return this._metadata.type(name);
    }

    attribute(type, name) {
        return this._metadata.attribute(type, name);
    }

    add(type, metadata) {
        this._map.set(type, metadata);
    }
};

keras.Metadata = class {

    static open(context) {
        if (keras.Metadata._metadata) {
            return Promise.resolve(keras.Metadata._metadata);
        }
        return context.request('keras-metadata.json', 'utf-8', null).then((data) => {
            keras.Metadata._metadata = new keras.Metadata(data);
            return keras.Metadata._metadata;
        }).catch(() => {
            keras.Metadata._metadata = new keras.Metadata(null);
            return keras.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
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
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

keras.Weights = class {

    constructor() {
        this._map = new Map();
    }

    add(layer_name, tensor) {
        if (!this._map.has(layer_name)) {
            this._map.set(layer_name, []);
        }
        this._map.get(layer_name).push(tensor);
    }

    get(group, name) {
        if (group) {
            const list = this._map.get(group.split('/').shift());
            if (list) {
                const match1 = list.filter((tensor) => tensor.name.startsWith(name + '/'));
                if (match1.length > 0) {
                    return match1;
                }
                const match2 = list.filter((tensor) => tensor.name.startsWith(group + '/' + name + '/'));
                if (match2.length > 0) {
                    return match2;
                }
            }
        }
        else {
            const match1 = this._map.get(name);
            if (match1 && match1.length > 0) {
                return match1;
            }
            const match2 = this._map.get('');
            if (match2 && match2.length > 0) {
                const match3 = match2.filter((tensor) => tensor.name.startsWith((group ? group + '/' : '') + name + '/'));
                if (match3.length > 0) {
                    return match3;
                }
            }
        }
        return [];
    }

    keys() {
        return this._map.keys();
    }
};

keras.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Keras model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = keras.ModelFactory;
}