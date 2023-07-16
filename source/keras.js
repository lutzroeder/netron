
var keras = keras || {};
var tfjs = tfjs || {};
var json = require('./json');
var python = require('./python');

keras.ModelFactory = class {

    match(context) {
        const group = context.open('hdf5');
        if (group && group.attributes.get('CLASS') !== 'hickle') {
            return 'keras.h5';
        }
        const json = context.open('json');
        if (json) {
            if (json.mxnet_version || (json.nodes && json.arg_nodes && json.heads)) {
                return null;
            }
            if (json.model_config || (json.class_name && json.config)) {
                return 'keras.json';
            }
        }
        if (tfjs.Container.open(context)) {
            return 'tfjs.json';
        }
        const pickle = context.open('pkl');
        if (pickle &&
            pickle.__class__ &&
            pickle.__class__.__module__ === 'keras.engine.sequential' &&
            pickle.__class__.__name__ === 'Sequential') {
            return 'keras.pickle';
        }
        return null;
    }

    async open(context, target) {
        const openModel = async (format, producer, backend, config, weights) => {
            const metadata = await context.metadata('keras-metadata.json');
            return new keras.Model(metadata, format, producer, backend, config, weights);
        };
        switch (target) {
            case 'keras.h5': {
                const find_root_group = (root_group) => {
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
                const group = context.open('hdf5');
                const root_group = find_root_group(group);
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
                                            const tensor = new keras.Tensor(weight_name, variable.shape, variable.type, null, variable.littleEndian ? '<' : '>', variable.data);
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
                                        const layout = variable.littleEndian ? '<' : '>';
                                        const tensor = new keras.Tensor(weight_name, variable.shape, variable.type, null, layout, variable.data);
                                        weights.add(name, tensor);
                                    }
                                }
                            }
                        }
                    }
                    return openModel(format, '', backend, null, weights);
                }
                if (context.identifier === 'model.weights.h5' &&
                    group.attributes.size === 0 &&
                    group.groups.has('_layer_checkpoint_dependencies')) {
                    const checkpoint = group.groups.get('_layer_checkpoint_dependencies');
                    for (const layer of checkpoint.groups) {
                        for (const vars of layer[1].groups) {
                            for (const entry of vars[1].groups) {
                                const variable = entry[1].value;
                                const layout = variable.littleEndian ? '<' : '>';
                                const tensor = new keras.Tensor(entry[0], variable.shape, variable.type, null, layout, variable.data);
                                weights.add(layer[0], tensor);
                            }
                        }
                    }
                    let model_config = null;
                    try {
                        const stream = await context.request('config.json', 'utf-8');
                        const reader = json.TextReader.open(stream);
                        model_config = reader.read();
                    } catch (error) {
                        // continue regardless of error
                    }
                    let metadata = null;
                    try {
                        const stream = await context.request('metadata.json', 'utf-8');
                        const reader = json.TextReader.open(stream);
                        metadata = reader.read();
                    } catch (error) {
                        // continue regardless of error
                    }
                    const format = 'Keras' + (metadata && metadata.keras_version ? ' v' + metadata.keras_version : '');
                    return openModel(format, '', '', model_config, weights);
                }
                const rootKeys = new Set(root_group.attributes.keys());
                rootKeys.delete('nb_layers');
                if (rootKeys.size > 0 || root_group.value !== null) {
                    throw new keras.Error('File format is not HDF5 Weights.');
                }
                const format = 'HDF5 Weights';
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
                        const layout = variable.littleEndian ? '<' : '>';
                        const tensor = new keras.Tensor(group.name, variable.shape, variable.type, null, layout, variable.type === 'string' ? variable.value : variable.data);
                        weights.add('', tensor);
                    }
                    return openModel(format, '', '', null, weights);
                }
                if (groups.every((group) => group.value === null && Array.from(group.attributes.keys()).filter((key) => !tensorKeys.has(key)).length === 0 && Array.from(group.groups.values()).every((variable) => Object.keys(variable.attributes).length === 0 && variable.value !== null))) {
                    for (const group of groups) {
                        const moduleName = group.attributes.has('name') ? group.attributes.get('name') : group.name;
                        for (const variableGroup of group.groups.values()) {
                            if (variableGroup.attributes.size !== 0 || variableGroup.groups.size !== 0) {
                                throw new keras.Error('Variable format is not HDF5 Weights.');
                            }
                            const variable = variableGroup.value;
                            if (!variable) {
                                throw new keras.Error('Variable value is not HDF5 Weights.');
                            }
                            const name = moduleName ? [ moduleName, variableGroup.name ].join('/') : moduleName.name;
                            const layout = variable.littleEndian ? '<' : '>';
                            const tensor = new keras.Tensor(name, variable.shape, variable.type, null, layout, variable.type === 'string' ? variable.value : variable.data);
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
                    if (match && group.value !== null && group.groups.size === 0) {
                        const variable = group.value;
                        const variableName = group.path;
                        let moduleName = variableName;
                        const parts = variableName.split('/');
                        if (parts.length > 1) {
                            parts.pop();
                            moduleName = parts.join('/');
                        }
                        const layout = variable.littleEndian ? '<' : '>';
                        const tensor = new keras.Tensor(variableName, variable.shape, variable.type, null, layout, variable.type === 'string' ? variable.value : variable.data);
                        weights.add(moduleName, tensor);
                        return;
                    }
                    throw new keras.Error('Module group format is not HDF5 Weights.');
                };
                walk(weights_group);
                return openModel(format, '', '', null, weights);
            }
            case 'keras.json': {
                const obj = context.open('json');
                const format = 'Keras' + (obj.keras_version ? ' v' + obj.keras_version : '');
                const backend = obj.backend || '';
                const config = obj.model_config ? obj.model_config : obj;
                const weights = new keras.Weights();
                return openModel(format, '', backend, config, weights);
            }
            case 'tfjs.json': {
                const container = tfjs.Container.open(context);
                await container.open();
                return openModel(container.format, container.producer, container.backend, container.config, container.weights);
            }
            case 'keras.pickle': {
                const execution = new python.Execution();
                const obj = context.open('pkl');
                const decoder = new TextDecoder('utf-8');
                const format = 'Keras Pickle' + (obj.keras_version ? ' v' + decoder.decode(obj.keras_version) : '');
                const backend = obj.backend ? decoder.decode(obj.backend) : '';
                const reader = json.TextReader.open(obj.model_config);
                const model_config = reader.read();
                const weights = new keras.Weights();
                const model_weights_group = obj.model_weights;
                if (model_weights_group) {
                    const layer_names = model_weights_group.layer_names.map((buffer) => decoder.decode(buffer));
                    for (const layer_name of layer_names) {
                        const layer_weights = model_weights_group[layer_name];
                        if (layer_weights) {
                            const weight_names = layer_weights.weight_names.map((buffer) => decoder.decode(buffer));
                            if (Array.isArray(weight_names) && weight_names.length > 0) {
                                for (const weight_name of weight_names) {
                                    const buffer = layer_weights[weight_name];
                                    const unpickler = execution.invoke('pickle.Unpickler', [ buffer ]);
                                    const variable = unpickler.load();
                                    const tensor = new keras.Tensor(weight_name, variable.shape, variable.dtype.__name__, null, '<', variable.data);
                                    weights.add(layer_name, tensor);
                                }
                            }
                        }
                    }
                }
                return openModel(format, '', backend, model_config, weights);
            }
            default: {
                throw new keras.Error("Unsupported Keras format '" + target + "'.");
            }
        }
    }
};

keras.Model = class {

    constructor(metadata, format, producer, backend, config, weights) {
        this._format = format;
        this._backend = backend;
        this._producer = producer;
        metadata = new keras.GraphMetadata(metadata);
        this._graphs = [ new keras.Graph(metadata, config, weights) ];
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

    constructor(metadata, config, weights, group) {
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        group = group || '';
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (tensor) {
                return new keras.Value(name, type || null, tensor);
            }
            if (!args.has(name)) {
                args.set(name, new keras.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new keras.Error("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        const loadNode = (layer, inputs, outputs, weights, group) => {
            layer = Object.assign({}, layer);
            layer.inputs = inputs;
            layer.outputs = outputs;
            return new keras.Node(this._metadata, layer, group, weights, arg);
        };
        const getInputType = (layer) => {
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
        };
        if (config) {
            this._name = config.name || (config.config && config.config.name ? config.config.name : '');
            const is_connection = (item) => {
                return Array.isArray(item) && (item.length === 3 || item.length === 4) && typeof item[0] === 'string' && typeof item[1] === 'number'  && typeof item[2] === 'number';
            };
            const is_constant = (item) => {
                return Array.isArray(item) && (item.length === 3 || item.length === 4) && item[0] === '_CONSTANT_VALUE' && item[1] === -1;
            };
            switch (config.class_name) {
                case 'AllCNN':
                case 'Sequential': {
                    config = config.config;
                    const outputs = null;
                    const inputName = 'input';
                    let inputType = null;
                    let value = inputName;
                    let index = 0;
                    const layers = config.layers ? config.layers : config;
                    for (const layer of layers) {
                        let name = index.toString();
                        const nodeInputs = [ { name: value } ];
                        if (index == 0) {
                            inputType = getInputType(layer);
                            this._inputs.push(new keras.Argument(inputName, true, [ arg(inputName, inputType) ]));
                        }
                        index++;
                        if (layer.config && layer.config.name) {
                            name = layer.config.name;
                        }
                        value = name;
                        let nodeOutputs = [ value ];
                        if (index == layers.length) {
                            if (outputs && outputs.length > 0) {
                                nodeOutputs = [ outputs[0] ];
                                value = null;
                            }
                        }
                        this.nodes.push(loadNode(layer, nodeInputs, nodeOutputs, weights, group));
                    }
                    if (value) {
                        this._outputs.push(new keras.Argument(value, true, [ arg(value) ]));
                    }
                    break;
                }
                case 'Functional':
                case 'Model': {
                    config = config.config;
                    const nodes = new Map();
                    if (config.layers) {
                        for (const layer of config.layers) {
                            layer.inputs = [];
                            layer.outputs = [];
                            layer.args = {};
                            if (layer.name && !nodes.has(layer.name)) {
                                nodes.set(layer.name, layer);
                            }
                        }
                        const read_connection = (input_data) => {
                            let name = input_data[0];
                            const node = nodes.get(name);
                            if (node) {
                                // const node_index = input_data[1];
                                const tensor_index = input_data[2];
                                if (tensor_index !== 0) {
                                    name += ':' + tensor_index.toString();
                                }
                                while (tensor_index >= node.outputs.length) {
                                    node.outputs.push('');
                                }
                                node.outputs[tensor_index] = name;
                            }
                            return { name: name };
                        };
                        const read_value = (input_data) => {
                            if (!Array.isArray(input_data)) {
                                return { shape: [], value: [ input_data ] };
                            }
                            const shape = (value) => {
                                if (value.every((item) => is_constant(item))) {
                                    for (let i = 0; i < value.length; i++) {
                                        value[i] = value[i][2];
                                    }
                                } else if (value.every((item) => Array.isArray(item))) {
                                    const dims = value.map((item) => shape(item));
                                    const dim = dims[0];
                                    for (let i = 1; i < dims.length; i++) {
                                        if (dim.length === dims[i].length) {
                                            if (!dims[i].every((value, i) => value ===dim[i])) {
                                                throw new python.Error('Invalid array shape.');
                                            }
                                        }
                                    }
                                    return [ value.length ].concat(dim);
                                }
                                return [ value.length ];
                            };
                            const flatten = (input) => input.reduce((a, b) => a.concat(Array.isArray(b) ? flatten(b) : b), []);
                            return { shape: shape(input_data), value: flatten(input_data) };
                        };
                        for (const layer of config.layers) {
                            if (layer.inbound_nodes) {
                                for (const inbound_node of layer.inbound_nodes) {
                                    if (is_constant(inbound_node)) {
                                        layer.inputs.push(read_value(inbound_node[2]));
                                        const args = inbound_node[3] || {};
                                        layer.args = {};
                                        for (const entry of Object.entries(args)) {
                                            const key = entry[0];
                                            const value = entry[1];
                                            layer.args[key] = is_connection(value) ? read_connection(value) : read_value(value);
                                        }
                                    } else if (is_connection(inbound_node)) {
                                        layer.inputs.push(read_connection(inbound_node));
                                        const args = inbound_node[3] || {};
                                        layer.args = {};
                                        for (const entry of Object.entries(args)) {
                                            const key = entry[0];
                                            const value = entry[1];
                                            layer.args[key] = is_connection(value) ? read_connection(value) : read_value(value);
                                        }
                                    } else if (Array.isArray(inbound_node)) {
                                        for (const input_data of inbound_node) {
                                            if (is_connection(input_data)) {
                                                layer.inputs.push(read_connection(input_data));
                                            } else if (Array.isArray(input_data) && input_data.every((item) => is_connection(item))) {
                                                for (const input of input_data) {
                                                    layer.inputs.push(read_connection(input));
                                                }
                                            } else if (Array.isArray(input_data)) {
                                                layer.inputs.push(read_value(input_data));
                                            } else {
                                                throw new keras.Error("Invalid inbound connection '" + JSON.stringify(input_data) + "'.");
                                            }
                                        }
                                    } else {
                                        throw new keras.Error("Invalid inbound node '" + JSON.stringify(inbound_node) + "'.");
                                    }
                                }
                            }
                        }
                    }
                    const input_layers = is_connection(config.input_layers) ? [ config.input_layers ] : config.input_layers;
                    if (input_layers) {
                        for (let i = 0; i < input_layers.length; i++) {
                            const input_layer = input_layers[i];
                            const name = input_layer[0];
                            let type = null;
                            const node = nodes.get(name);
                            if (node && node.class_name == 'InputLayer') {
                                type = getInputType(node);
                                nodes.delete(name);
                            }
                            const argument = new keras.Argument(name, true, [ arg(name, type) ]);
                            this._inputs.push(argument);
                        }
                    }
                    const output_layers = is_connection(config.output_layers) ? [ config.output_layers ] : config.output_layers;
                    if (output_layers) {
                        for (let j = 0; j < output_layers.length; j++) {
                            const output_layer = output_layers[j];
                            let outputName = output_layer[0];
                            const outputNode = nodes.get(outputName);
                            if (outputNode) {
                                const outputIndex = output_layer[2];
                                if (outputIndex != 0) {
                                    outputName += ':' + outputIndex.toString();
                                }
                                while (outputIndex >= outputNode.outputs.length) {
                                    outputNode.outputs.push('');
                                }
                                outputNode.outputs[outputIndex] = outputName;
                            }
                            const argument = new keras.Argument(outputName, true, [ arg(outputName) ]);
                            this._outputs.push(argument);
                        }
                    }
                    if (config.layers) {
                        for (const layer of config.layers) {
                            if (nodes.has(layer.name)) {
                                this._nodes.push(loadNode(layer, layer.inputs, layer.outputs, weights, group));
                            }
                        }
                    }
                    break;
                }
                default:
                    throw new keras.Error('\'' + config.class_name + '\' is not supported.');
            }
        } else if (weights) {
            for (const name of weights.keys()) {
                if (weights.get('', name).length <= 6) {
                    const layer = { class_name: 'Weights', config: { name: name } };
                    const node = new keras.Node(metadata, layer, '', weights, arg);
                    this._nodes.push(node);
                }
            }
        }
    }

    get name() {
        return this._name;
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
};

keras.Argument = class {

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

keras.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new keras.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, layer, group, weights, arg) {
        const config = layer.config || {};
        const args = layer.args || {};
        let inputs = layer.inputs || [];
        let outputs = layer.outputs || [];
        const name = config && config.name ? config.name : '';
        this._group = group || '';
        this._name = (this._group ? this._group + '/' : '') + name;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._chain = [];
        let names = [ name ];
        let type = layer.class_name;
        let model = false;
        switch (type) {
            case 'Model':
            case 'Functional':
            case 'Sequential': {
                const name = layer.name || (layer.config ? layer.config.name : '');
                this._type = new keras.Graph(metadata, layer, weights, (group ? group + '/' : '') + name);
                model = true;
                if (config) {
                    delete config.layers;
                    delete config.input_layers;
                    delete config.output_layers;
                }
                this._inputs = [ new keras.Argument('inputs', true, inputs.map((input) => arg(input.name))) ];
                this._outputs = [ new keras.Argument('outputs', true, outputs.map((name) => arg(name))) ];
                inputs = [];
                outputs = [];
                break;
            }
            case 'Bidirectional':
            case 'TimeDistributed': {
                if (config && config.layer) {
                    const inner = config.layer;
                    delete config.layer;
                    this._inner = new keras.Node(metadata, inner, null, null, arg);
                    if (type == 'Bidirectional' && inner.config.name) {
                        names = [ name + '/forward_' + inner.config.name, name + '/backward_' + inner.config.name ];
                        if (!group) {
                            group = name;
                        }
                    }
                }
                this._type = metadata.type(type) || { name: type };
                break;
            }
            case 'TFOpLambda': {
                if (config && config.function) {
                    type = config.function;
                    delete config.function;
                }
                this._type = metadata.type(type) || { name: type };
                break;
            }
            default: {
                this._type = metadata.type(type) || { name: type };
                break;
            }
        }

        const initializers = {};
        if (weights && !model) {
            for (const name of names) {
                let tensors = weights.get(group, name);
                if (tensors.length > 0) {
                    for (const initializer of tensors) {
                        inputs.push({ name: initializer.name });
                        initializers[initializer.name] = initializer;
                    }
                } else {
                    tensors = weights.get('', name);
                    for (const initializer of tensors) {
                        inputs.push({ name: initializer.name });
                        initializers[initializer.name] = initializer;
                    }
                }
            }
        }

        if (config && !Array.isArray(config)) {
            for (const entry of Object.entries(config)) {
                const name = entry[0];
                const value = entry[1];
                if (name === 'activation' && value !== 'linear') {
                    if (typeof value === 'string') {
                        const set = new Map([ [ 'elu', 'ELU' ], [ 'exponential', 'Exponential' ], [ 'hard_sigmoid', 'HardSigmoid' ], [ 'linear', 'Linear' ], [ 'relu', 'ReLU' ], [ 'selu', 'SELU' ], [ 'softmax', 'Softmax'], [ 'sigmoid', 'Sigmoid' ], [ 'softplus', 'SoftPlus' ], [ 'softsign', 'SoftSign' ], [ 'tanh', 'TanH' ] ]);
                        const type = set.has(value) ? set.get(value) : value;
                        this.chain.push(new keras.Node(metadata, { class_name: type }, null, null, arg));
                    } else if (value && typeof value.class_name === 'string' && value.config) {
                        const type = value.class_name;
                        if (!metadata.type(type)) {
                            metadata.add(type, { name: type, category: 'Activation' });
                        }
                        this.chain.push(new keras.Node(metadata, value, null, null, arg));
                    }
                }
                if (name !== 'name') {
                    const attribute = new keras.Attribute(metadata.attribute(type, name), name, value);
                    this._attributes.push(attribute);
                }
            }
        }

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
            } else {
                switch (type) {
                    case 'Bidirectional': {
                        let innerIndex = inputIndex;
                        if (innerSchema && innerSchema.inputs) {
                            if (innerIndex < innerSchema.inputs.length) {
                                inputName = 'forward_' + innerSchema.inputs[innerIndex].name;
                            } else {
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
                    default:
                        break;
                }
            }
            const input = !list ? [ inputs.shift() ] : inputs.splice(0, inputs.length);
            const inputArguments = input.map((input) => {
                if (input.name) {
                    return arg(input.name, null, initializers[input.name]);
                }
                if (input.value !== undefined) {
                    const tensor = new keras.Tensor('', input.shape, config.dtype || '?', null, '|', input.value);
                    return arg('', null, tensor);
                }
                throw new keras.Error("Invalid argument '" + JSON.stringify(input.name) + "'.");
            });
            if (!inputName && inputArguments.length == 1 && inputArguments[0].initializer && inputArguments[0].initializer.name) {
                if (names.length === 1 && names[0] === '') {
                    inputName = inputArguments[0].initializer.name;
                } else {
                    const parts = inputArguments[0].initializer.name.split('/').pop().split(':').shift().split('_');
                    const inputName1 = parts.pop();
                    const inputName2 = parts.length > 0 ? [ parts.pop(), inputName1 ].join('_') : '';
                    const inputNames = new Set([ 'recurrent_kernel', 'running_mean', 'running_std', 'moving_mean', 'moving_variance', 'depthwise_filter', 'pointwise_filter' ]);
                    inputName = inputNames.has(inputName2) ? inputName2 : inputName1;
                }
            }
            this._inputs.push(new keras.Argument(inputName || inputIndex.toString(), visible, inputArguments));
            inputIndex++;
        }

        for (let i = 0; i < outputs.length; i++) {
            const output = outputs[i];
            const outputName = (this._type && this._type.outputs && i < this._type.outputs.length && this._type.outputs[i] && this._type.outputs[i].name) ? this._type.outputs[i].name : i.toString();
            const args = output.length === 0 ? [] : [ arg(output) ];
            const argument = new keras.Argument(outputName, true, args);
            this._outputs.push(argument);
        }

        const inputTypes = new Map((this._type.inputs || []).map((input) => [ input.name, input.type ]));
        for (const entry of Object.entries(args)) {
            const name = entry[0];
            const value = entry[1];
            if (name !== 'name') {
                if (value.name || (inputTypes.has(name) && inputTypes.get(name) === 'Tensor' && value)) {
                    if (value.name) {
                        const argument = new keras.Argument(name, true, [ arg(value.name) ]);
                        this._inputs.push(argument);
                    } else {
                        const tensor = new keras.Tensor('', value.shape, config.dtype || '?', null, '|', value.value);
                        const argument = new keras.Argument(name, true, [ arg('', null, tensor) ]);
                        this._inputs.push(argument);
                    }
                } else {
                    const attribute = new keras.Attribute(metadata.attribute(type, name), name, value);
                    this._attributes.push(attribute);
                }
            }
        }

        if (typeof this.type.name !== 'string' || !this.type.name.split) { // #416
            throw new keras.Error("Unsupported node type '" + JSON.stringify(this.type.name) + "'.");
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
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
        if (value && typeof value == 'object' && value.class_name && value.config) {
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
                    if (metadata.visible === false) {
                        this._visible = false;
                    } else if (metadata.default !== undefined) {
                        if (Array.isArray(value)) {
                            if (Array.isArray(metadata.default)) {
                                this._visible = value.length !== metadata.default || !this.value.every((item, index) => item == metadata.default[index]);
                            } else {
                                this._visible = !this.value.every((item) => item == metadata.default);
                            }
                        } else {
                            this._visible = this.value !== metadata.default;
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
};

keras.Tensor = class {

    constructor(name, shape, type, quantization, layout, data) {
        this._name = name;
        this._type = new keras.TensorType(type, new keras.TensorShape(shape));
        this._quantization = quantization;
        this._layout = layout;
        this._data = data;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get layout() {
        return this._layout;
    }

    get quantization() {
        if (this._quantization && (this._quantization.scale !== 0 || this._quantization.min !== 0)) {
            const scale = this._quantization.scale || 0;
            const min = this._quantization.min || 0;
            return scale.toString() + ' * ' + (min == 0 ? 'q' : ('(q - ' + min.toString() + ')'));
        }
        return null;
    }

    get values() {
        if (this._layout === '|') {
            return this._data;
        }
        if (this._data === null) {
            return null;
        }
        return this._data instanceof Uint8Array ? this._data : this._data.peek();
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
        return this._dimensions && this._dimensions.length > 0 ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

keras.GraphMetadata = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._types = new Map();
    }

    type(name) {
        if (this._types.has(name)) {
            return this._types.get(name);
        }
        return this._metadata.type(name);
    }

    attribute(type, name) {
        return this._metadata.attribute(type, name);
    }

    add(type, metadata) {
        this._types.set(type, metadata);
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
        } else {
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

tfjs.Container = class {

    static open(context) {
        const json = context.open('json');
        if (json) {
            if (json.modelTopology && (json.format === 'layers-model' || json.modelTopology.class_name || json.modelTopology.model_config)) {
                return new tfjs.Container(context, '');
            }
            if (Array.isArray(json) && json.every((item) => item.weights && item.paths)) {
                return new tfjs.Container(context, 'weights');
            }
            if (json.tfjsVersion) {
                return new tfjs.Container(context, 'metadata');
            }
        }
        return null;
    }

    constructor(context, type) {
        this._context = context;
        this._type = type;
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer || '';
    }

    get backend() {
        return this._backend || '';
    }

    get config() {
        return this._config;
    }

    get weights() {
        return this._weights;
    }

    async open() {
        switch (this._type) {
            case '': {
                const obj = this._context.open('json');
                return this._openModelJson(obj);
            }
            case 'weights': {
                this._format = 'TensorFlow.js Weights';
                this._config = null;
                const obj = this._context.open('json');
                const manifests = Array.from(obj);
                for (const manifest of manifests) {
                    for (const weight of manifest.weights) {
                        const name = weight.name;
                        const index = name.lastIndexOf('/');
                        weight.identifier = index === -1 ? name : name.substring(0, index);
                    }
                }
                return this._openManifests(manifests);
            }
            case 'metadata': {
                const stream = await this._context.request('model.json');
                const reader = json.TextReader.open(stream);
                const obj = reader.read();
                return this._openModelJson(obj);
            }
            default: {
                throw new tfjs.Error("Unsupported TensorFlow.js format '" + this._type + "'.");
            }
        }
    }

    _openShards(manifests, shards) {
        this._weights = new keras.Weights();
        const dtype_size_map = new Map([
            [ 'float16', 2 ], [ 'float32', 4 ], [ 'float64', 8 ],
            [ 'int8', 1 ], [ 'int16', 2 ], [ 'int32', 4 ], [ 'int64', 8 ],
            [ 'uint8', 1 ], [ 'uint16', 2 ], [ 'uint32', 4 ], [ 'uint64', 8 ]
        ]);
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
                    throw new keras.Error("Unsupported weight data type size '" + dtype + "'.");
                }
                const itemsize = dtype_size_map.get(dtype);
                const size = weight.shape.reduce((a, b) => a * b, 1);
                const length = itemsize * size;
                const data = buffer ? buffer.slice(offset, offset + length) : null;
                this._weights.add(weight.identifier, new keras.Tensor(weight.name, weight.shape, dtype, weight.quantization, '<', data));
                offset += length;
            }
        }
    }

    async _openManifests(manifests) {
        const shards = new Map();
        for (const manifest of manifests) {
            for (const path of manifest.paths) {
                if (!shards.has(path)) {
                    const promise = this._context.request(path, null);
                    shards.set(path, promise);
                }
            }
        }
        const promises = shards.values();
        try {
            const streams = await Promise.all(promises);
            for (const key of shards.keys()) {
                shards.set(key, streams.shift().peek());
            }
            this._openShards(manifests, shards);
            return;
        } catch (error) {
            shards.clear();
            this._openShards(manifests, shards);
            return;
        }
    }

    _openModelJson(obj) {
        const modelTopology = obj.modelTopology;
        this._format = 'TensorFlow.js ' + (obj.format ? obj.format : 'Keras' + (modelTopology.keras_version ? (' v' + modelTopology.keras_version) : ''));
        this._producer = obj.convertedBy || obj.generatedBy || '';
        this._backend = modelTopology.backend || '';
        const manifests = obj.weightsManifest;
        for (const manifest of manifests) {
            for (const weight of manifest.weights) {
                weight.identifier = '';
            }
        }
        this._config = modelTopology.model_config ? modelTopology.model_config : modelTopology;
        return this._openManifests(manifests);
    }
};

tfjs.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow.js model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = keras.ModelFactory;
}