
var keras = {};
var tfjs = {};
var json = require('./json');
var hdf5 = require('./hdf5');
var python = require('./python');
var protobuf = require('./protobuf');
var zip = require('./zip');

keras.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const group = context.open('hdf5');
        if (group && group.attributes && group.attributes.get('CLASS') !== 'hickle') {
            if (identifier === 'model.weights.h5') {
                return 'keras.model.weights.h5';
            }
            return 'keras.h5';
        }
        const json = context.open('json');
        if (json) {
            if (json.mxnet_version || (json.nodes && json.arg_nodes && json.heads)) {
                return null;
            }
            if (json.model_config || (json.class_name && json.config)) {
                return 'keras.config.json';
            }
            if (identifier === 'metadata.json' && json.keras_version) {
                return 'keras.metadata.json';
            }
        }
        if (tfjs.Container.open(context)) {
            return 'tfjs.json';
        }
        const pickle = context.open('pkl');
        if (pickle && pickle.__class__ &&
            pickle.__class__.__module__ === 'keras.engine.sequential' &&
            pickle.__class__.__name__ === 'Sequential') {
            return 'keras.pickle';
        }
        // model.weights.npz
        const entries = context.entries('zip');
        const regex = /^(__root__|layers\/.+|_layer_checkpoint_dependencies\/.+)\.npy$/;
        if (entries.size > 0 && Array.from(entries).every((entry) => regex.test(entry[0]))) {
            return 'keras.model.weights.npz';
        }
        // keras_metadata.pb
        if (extension === 'pb' && context.stream && context.stream.length > 16) {
            const tags = context.tags('pb');
            if (tags.size === 1 && tags.get(1) === 2) {
                const stream = context.stream;
                const buffer = stream.peek(Math.min(stream.length, 1024));
                const content = String.fromCharCode.apply(null, buffer);
                if (/root"/.test(content) && /\{\s*"class_name"\s*:/.test(content)) {
                    return 'keras.pb.SavedMetadata';
                }
            }
        }
        return null;
    }

    async open(context, target) {
        const request = async (context, name) => {
            try {
                return await context.request(name, null);
            } catch (error) {
                return null;
            }
        };
        const request_json = async (context, name) => {
            const stream = await request(context, name);
            if (stream) {
                const reader = json.TextReader.open(stream);
                return reader.read();
            }
            return null;
        };
        const _create_config = (weights_store) => {
            const config = {};
            config.class_name = 'Model';
            config.config = {};
            config.config.layers = [];
            const snake_to_pascal_case = (name) => {
                return name.replace(/(^|_|\d)([a-z])/g, (match, p1, p2) => p1 === '_' ? p2.toUpperCase() : p1 + p2.toUpperCase());
            };
            for (const entry of weights_store) {
                const layer = {};
                layer.name = entry[0];
                layer.class_name = entry[0].split('/').pop().replace(/_[0-9]+$/, '');
                layer.class_name = snake_to_pascal_case(layer.class_name);
                layer.config = {};
                layer.config.name = entry[0];
                layer._trainable_variables = entry[1];
                config.config.layers.push(layer);
            }
            return config;
        };
        const _load_state = (trackable, weights_store, assets_store, inner_path) => {
            inner_path = inner_path || '';
            if (trackable && trackable.config && Array.isArray(trackable.config.layers)) {
                /* eslint-disable no-use-before-define */
                _load_container_state(trackable, weights_store, assets_store, inner_path ? inner_path + '/layers' : 'layers');
                /* eslint-enable no-use-before-define */
            } else {
                const weights = weights_store.get(inner_path);
                if (weights) {
                    trackable._trainable_variables = weights;
                }
            }
        };
        const _load_container_state = (container, weights_store, assets_store, inner_path) => {
            const used_names = new Map();
            for (const trackable of container.config.layers) {
                const pascal_to_snake_case = (name) => {
                    name = name.replace(/\W+/g, "");
                    name = name.replace(/(.)([A-Z][a-z]+)/g, (match, p1, p2) => p1 + '_' + p2);
                    name = name.replace(/([a-z])([A-Z])/g, (match, p1, p2) => p1 + '_' + p2);
                    return name.toLowerCase();
                };
                let name = pascal_to_snake_case(trackable.class_name);
                if (!used_names.has(name)) {
                    used_names.set(name, 0);
                } else {
                    const next = used_names.get(name) + 1;
                    used_names.set(name, next);
                    name = name + '_' + next.toString();
                }
                _load_state(trackable, weights_store, assets_store, inner_path + '/' + name);
            }
        };
        const read_weights_hdf5 = (group) => {
            const walk = (group, path, weights_store) => {
                const checkpoint = group.groups.get('layers') || group.groups.get('_layer_checkpoint_dependencies');
                if (checkpoint) {
                    for (const layer of checkpoint.groups) {
                        const name = (path ? path + '/' : '') + 'layers/' + layer[0];
                        walk(layer[1], name, weights_store);
                        const values = [];
                        for (const vars of layer[1].groups) {
                            for (const entry of vars[1].groups) {
                                const variable = entry[1].value;
                                if (variable) {
                                    const layout = variable.littleEndian ? '<' : '>';
                                    const tensor = new keras.Tensor(entry[0], variable.shape, variable.type, null, null, layout, variable.data);
                                    values.push(tensor);
                                }
                            }
                        }
                        if (values.length > 0) {
                            weights_store.set(name, values);
                        }
                    }
                }
            };
            const weights_store = new Map();
            walk(group, '', weights_store);
            return weights_store;
        };
        const read_weights_numpy = (entries) => {
            const weights_store = new Map();
            const execution = new python.Execution();
            for (const entry of entries) {
                entry[0] = entry[0].split('/').map((name) => name === '_layer_checkpoint_dependencies' ? 'layers' : name).join('/');
                if (entry[0].endsWith('.npy') && entry[0].startsWith('layers/')) {
                    const name = entry[0].replace(/\.npy$/, '');
                    const stream = entry[1];
                    const buffer = stream.peek();
                    const bytes = execution.invoke('io.BytesIO', [ buffer ]);
                    const array = execution.invoke('numpy.load', [ bytes ]);
                    if (array.dtype.name === 'object' && array.shape.length === 0 &&
                        Array.isArray(array.data) && array.data.length === 1) {
                        const values = Object.values(array.data[0]).map((array) => {
                            const stride = array.strides.map((stride) => stride / array.itemsize);
                            const dataType = array.dtype.__name__;
                            const values = dataType == 'string' || dataType == 'object' ? array.flatten().tolist() : array.tobytes();
                            const encoding = dataType == 'string' || dataType == 'object' ? '|' : array.dtype.byteorder;
                            return new keras.Tensor('', array.shape, dataType, stride, null, encoding, values);
                        });
                        if (values.length > 0) {
                            weights_store.set(name, values);
                        }
                    }
                }
            }
            return weights_store;
        };
        const request_weights = async (context) => {
            const stream = await request(context, 'model.weights.h5');
            if (stream) {
                const file = hdf5.File.open(stream);
                const group = file.read();
                if (group) {
                    return read_weights_hdf5(group);
                }
            } else {
                const stream = await request(context, 'model.weights.npz');
                if (stream) {
                    const entries = zip.Archive.open(stream);
                    if (entries && entries.length > 0) {
                        return read_weights_numpy(entries);
                    }
                }
            }
            return new Map();
        };
        const open_model = async (format, producer, backend, config, weights) => {
            const metadata = await context.metadata('keras-metadata.json');
            return new keras.Model(metadata, format, producer, backend, config, weights);
        };
        switch (target) {
            case 'keras.config.json': {
                const obj = context.open('json');
                const config = obj.model_config ? obj.model_config : obj;
                const backend = obj.backend || '';
                let version = obj.keras_version ? obj.keras_version : null;
                if (!version) {
                    const metadata = await request_json(context, 'metadata.json');
                    if (metadata && metadata.keras_version) {
                        version = metadata.keras_version;
                    }
                }
                const format = 'Keras' + (version ? ' v' + version : '');
                const weights_store = await request_weights(context);
                _load_state(config, weights_store);
                return open_model(format, '', backend, config, null);
            }
            case 'keras.model.weights.h5': {
                const weights_store = read_weights_hdf5(context.open('hdf5'));
                const metadata = await request_json(context, 'metadata.json');
                let config = await request_json(context, 'config.json');
                const name = config ? 'Keras' : 'Keras Weights';
                const format = name + (metadata && metadata.keras_version ? ' v' + metadata.keras_version : '');
                if (config) {
                    _load_state(config, weights_store);
                } else {
                    config = _create_config(weights_store);
                }
                return open_model(format, '', '', config, null);
            }
            case 'keras.model.weights.npz': {
                const weights_store = read_weights_numpy(context.entries('zip'));
                const metadata = await request_json(context, 'metadata.json');
                let config = await request_json(context, 'config.json');
                const name = config ? 'Keras' : 'Keras Weights';
                const format = name + (metadata && metadata.keras_version ? ' v' + metadata.keras_version : '');
                if (config) {
                    _load_state(config, weights_store);
                } else {
                    config = _create_config(weights_store);
                }
                return open_model(format, '', '', config, null);
            }
            case 'keras.metadata.json': {
                const metadata = target;
                let config = await request_json(context, 'config.json');
                const name = config ? 'Keras' : 'Keras Weights';
                const format = name + (metadata.keras_version ? 'v' + metadata.keras_version : '');
                const weights_store = await request_weights(context);
                if (!config && (!weights_store || weights_store.size === 0)) {
                    throw new keras.Error("'config.json' or 'model.weights.*' not present.");
                }
                if (config) {
                    _load_state(config, weights_store);
                } else {
                    config = _create_config(weights_store);
                }
                return open_model(format, '', '', config, null);
            }
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
                                            const tensor = new keras.Tensor(weight_name, variable.shape, variable.type, null, null, variable.littleEndian ? '<' : '>', variable.data);
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
                    return open_model(format, '', backend, model_config, weights);
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
                                        const encoding = variable.littleEndian ? '<' : '>';
                                        const tensor = new keras.Tensor(weight_name, variable.shape, variable.type, null, null, encoding, variable.data);
                                        weights.add(name, tensor);
                                    }
                                }
                            }
                        }
                    }
                    return open_model(format, '', backend, null, weights);
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
                        const tensor = new keras.Tensor(group.name, variable.shape, variable.type, null, null, layout, variable.type === 'string' ? variable.value : variable.data);
                        weights.add('', tensor);
                    }
                    return open_model(format, '', '', null, weights);
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
                            const tensor = new keras.Tensor(name, variable.shape, variable.type, null, null, layout, variable.type === 'string' ? variable.value : variable.data);
                            weights.add(moduleName, tensor);
                        }
                    }
                    return open_model(format, '', '', null, weights);
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
                        const tensor = new keras.Tensor(variableName, variable.shape, variable.type, null, null, layout, variable.type === 'string' ? variable.value : variable.data);
                        weights.add(moduleName, tensor);
                        return;
                    }
                    throw new keras.Error('Module group format is not HDF5 Weights.');
                };
                walk(weights_group);
                return open_model(format, '', '', null, weights);
            }
            case 'tfjs.json': {
                const container = tfjs.Container.open(context);
                await container.open();
                return open_model(container.format, container.producer, container.backend, container.config, container.weights);
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
                                    const tensor = new keras.Tensor(weight_name, variable.shape, variable.dtype.__name__, null, null, '<', variable.data);
                                    weights.add(layer_name, tensor);
                                }
                            }
                        }
                    }
                }
                return open_model(format, '', backend, model_config, weights);
            }
            case 'keras.pb.SavedMetadata': {
                await context.require('./keras-proto');
                keras.proto = protobuf.get('tf');
                const format = 'Keras Saved Metadata';
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                const saved_metadata = keras.proto.third_party.tensorflow.python.keras.protobuf.SavedMetadata.decode(reader);
                if (!saved_metadata || !Array.isArray(saved_metadata.nodes) ||
                    !saved_metadata.nodes.every((node) => node && typeof node.metadata === 'string' && node.metadata.length > 0)) {
                    throw new keras.Error('Invalid keras.protobuf.SavedMetadata.');
                }
                const objects = new Map();
                for (const node of saved_metadata.nodes) {
                    const reader = json.TextReader.open(node.metadata);
                    node.metadata = reader.read();
                    objects.set(node.node_path, node);
                }
                const model_config = objects.get('root').metadata;
                return open_model(format, '', '', model_config, null);
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
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (tensor) {
                return new keras.Value(name, type || null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new keras.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new keras.Error("Duplicate value '" + name + "'.");
            }
            return values.get(name);
        };
        if (config) {
            const getInputType = (layer) => {
                if (layer && layer.config) {
                    let dataType = '?';
                    let shape = [];
                    const config = layer.config;
                    if (config.dtype) {
                        dataType = config.dtype;
                        delete config.dtype;
                    }
                    if (Array.isArray(config.batch_input_shape)) {
                        shape = config.batch_input_shape.map((s) => s == null ? '?' : s);
                        delete config.batch_input_shape;
                    } else if (config.batch_input_shape &&
                        config.batch_input_shape.class_name === '__tuple__' &&
                        Array.isArray(config.batch_input_shape.items)) {
                        shape = config.batch_input_shape.items.map((s) => s == null ? '?' : s);
                        delete config.batch_input_shape;
                    }
                    return new keras.TensorType(dataType, new keras.TensorShape(shape));
                }
                return null;
            };
            this._name = config.name || (config.config && config.config.name ? config.config.name : '');
            switch (config.class_name) {
                case 'AllCNN':
                case 'Sequential': {
                    config = config.config;
                    const outputs = null;
                    let name = 'input';
                    let index = -1;
                    const layers = Array.from(config.layers ? config.layers : config);
                    while (layers.length > 0) {
                        const layer = layers.shift();
                        let current = index.toString();
                        index++;
                        if (index == 0) {
                            const type = getInputType(layer);
                            let remove = false;
                            if (layer.class_name === 'InputLayer' && layer.config && layer.config.name) {
                                name = layer.config.name;
                                remove = true;
                            }
                            const value = values.map(name, type);
                            const argument = new keras.Argument(name, true, [ value ]);
                            this._inputs.push(argument);
                            if (remove) {
                                continue;
                            }
                        }
                        const nodeInputs = [ { name: name } ];
                        if (layer.config && layer.config.name) {
                            current = layer.config.name;
                        }
                        name = current;
                        let nodeOutputs = [ name ];
                        if (index == layers.length) {
                            if (outputs && outputs.length > 0) {
                                nodeOutputs = [ outputs[0] ];
                                name = null;
                            }
                        }
                        layer.inputs = nodeInputs;
                        layer.outputs = nodeOutputs;
                        const node = new keras.Node(this._metadata, layer, group, weights, values);
                        this.nodes.push(node);
                    }
                    if (name) {
                        const value = values.map(name);
                        const argument = new keras.Argument(name, true, [ value ]);
                        this._outputs.push(argument);
                    }
                    break;
                }
                case '__Function__':
                case 'Functional':
                case 'Model': {
                    config = config.config;
                    const nodes = new Map();
                    if (config.layers) {
                        const is_constant = (item) => {
                            return Array.isArray(item) && (item.length === 3 || item.length === 4) && item[0] === '_CONSTANT_VALUE' && item[1] === -1;
                        };
                        const is_connection = (item) => {
                            return Array.isArray(item) && (item.length === 3 || item.length === 4) && typeof item[0] === 'string' && typeof item[1] === 'number'  && typeof item[2] === 'number';
                        };
                        const read_value = (input_data) => {
                            if (!Array.isArray(input_data)) {
                                return input_data;
                            }
                            const transform = (value) => {
                                if (value.every((item) => is_constant(item))) {
                                    for (let i = 0; i < value.length; i++) {
                                        value[i] = value[i][2];
                                    }
                                } else if (value.every((item) => Array.isArray(item))) {
                                    const dims = value.map((item) => transform(item));
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
                            const shape = transform(input_data);
                            const flatten = (input) => input.reduce((a, b) => a.concat(Array.isArray(b) ? flatten(b) : b), []);
                            const value = flatten(input_data);
                            return { shape: shape, value: value };
                        };
                        const functional = config.layers.every((layer) => Array.isArray(layer.inbound_nodes));
                        const layers = new Map();
                        if (functional) {
                            const read_connection = (input_data) => {
                                const node_name = input_data[0];
                                const node_index = input_data[1];
                                const tensor_index = input_data[2];
                                const inbound_node_key = node_name + '[' + node_index.toString() + ']';
                                const inbound_node = nodes.get(inbound_node_key);
                                const tensor_key = node_name + '[' + node_index + '][' + tensor_index + ']';
                                if (inbound_node) {
                                    while (tensor_index >= inbound_node.outputs.length) {
                                        inbound_node.outputs.push(undefined);
                                    }
                                    inbound_node.outputs[tensor_index] = tensor_key;
                                }
                                return tensor_key;
                            };
                            const process_node = (node, inbound_node) => {
                                if (Array.isArray(inbound_node) && inbound_node.length === 4 && typeof inbound_node[0] === 'string') {
                                    const key = read_connection(inbound_node);
                                    node.inputs.push({ name: key });
                                    for (const entry of Object.entries(inbound_node[3])) {
                                        const name = entry[0];
                                        const value = entry[1];
                                        if (is_connection(value)) {
                                            const key = read_connection(value);
                                            node.inputs.push({ name: key });
                                        } else if (Array.isArray(value)) {
                                            const array = read_value(value);
                                            node.args[name] = array;
                                        } else {
                                            node.args[name] = value;
                                        }
                                    }
                                } else if (Array.isArray(inbound_node)) {
                                    for (const input_data of inbound_node) {
                                        // [ 'conv2d', 0, 0 ] or [ 'conv2d', 0, 0, {} ]
                                        if (Array.isArray(input_data) && is_connection(input_data)) {
                                            const key = read_connection(input_data);
                                            node.inputs.push({ name: key });
                                        } else if (Array.isArray(input_data) && input_data.every((item) => is_connection(item))) {
                                            for (const input of input_data) {
                                                const key = read_connection(input);
                                                node.inputs.push({ name: key });
                                            }
                                        } else if (Array.isArray(input_data)) {
                                            const value = read_value(input_data);
                                            node.inputs.push(value);
                                        } else {
                                            throw new keras.Error("Invalid inbound connection '" + JSON.stringify(input_data) + "'.");
                                        }
                                    }
                                } else if (inbound_node && inbound_node.args) {
                                    for (const arg of inbound_node.args) {
                                        if (arg && arg.class_name === '__keras_tensor__' && arg.config && is_connection(arg.config.keras_history)) {
                                            const key = read_connection(arg.config.keras_history);
                                            node.inputs.push({ name: key });
                                        } else if (Array.isArray(arg) && arg.every((arg) => arg && arg.class_name === '__keras_tensor__' && arg.config && is_connection(arg.config.keras_history))) {
                                            for (const input of arg) {
                                                const key = read_connection(input.config.keras_history);
                                                node.inputs.push({ name: key });
                                            }
                                        }
                                    }
                                }
                            };
                            let legacy_format = true;
                            for (const layer of config.layers) {
                                if (Array.isArray(layer.inbound_nodes)) {
                                    for (const inbound_node of layer.inbound_nodes) {
                                        if (Array.isArray(inbound_node.args)) {
                                            legacy_format = false;
                                        }
                                    }
                                }
                            }
                            for (const layer of config.layers) {
                                const class_name = layer.class_name;
                                let first_index = 0;
                                if (legacy_format) {
                                    const keys = new Set(Object.keys(layer.config));
                                    const is_functional_config = keys.has('name') && keys.has('layers') && keys.has('input_layers') && keys.has('output_layers');
                                    if (class_name == 'Sequential' ||
                                        (is_functional_config && Array.isArray(layer.config.layers) && layer.config.layers.length > 0 && layer.config.layers[0].class_name === 'InputLayer')) {
                                        first_index++;
                                    }
                                }
                                layers.set(layer.name, layers);
                                if (Array.isArray(layer.inbound_nodes) && layer.inbound_nodes.length === 0) {
                                    layer.inputs = [];
                                    layer.outputs = [];
                                    layer.args = {};
                                    nodes.set(layer.name + '[' + first_index + ']', layer);
                                } else if (Array.isArray(layer.inbound_nodes) && layer.inbound_nodes.length === 1) {
                                    layer.inputs = [];
                                    layer.outputs = [];
                                    layer.args = {};
                                    layer.inbound_node = layer.inbound_nodes[0];
                                    nodes.set(layer.name + '[' + first_index + ']', layer);
                                } else {
                                    let config = {};
                                    switch (class_name) {
                                        case 'Functional':
                                        case 'Sequential':
                                        case 'Model': {
                                            config = layer;
                                            break;
                                        }
                                        default: {
                                            config.class_name = '__Function__';
                                            config.name = layer.name;
                                            config.config = {};
                                            config.config.layers = [ Object.assign({}, layer) ];
                                            delete config.config.layers[0].inbound_nodes;
                                            delete config.config.layers[0].input_layers;
                                            delete config.config.layers[0].output_layers;
                                            break;
                                        }
                                    }
                                    const type = new keras.Graph(this._metadata, config, weights, '');
                                    for (let i = 0; i < layer.inbound_nodes.length; i++) {
                                        const index = i + first_index;
                                        const key = layer.name + '[' + index.toString() + ']';
                                        const node = {};
                                        node.name = key;
                                        node.class_name = '__Function__';
                                        node.config = {};
                                        node.config.name = key;
                                        node.inputs = [];
                                        node.outputs = [];
                                        node.args = {};
                                        node.__type__ = type;
                                        node.inbound_node = layer.inbound_nodes[i];
                                        nodes.set(key, node);
                                    }
                                }
                            }
                            for (const entry of nodes) {
                                if (entry[1].inbound_node) {
                                    process_node(entry[1], entry[1].inbound_node);
                                }
                            }
                            if (Array.isArray(config.input_layers)) {
                                for (let i = 0; i < config.input_layers.length; i++) {
                                    const input_data = config.input_layers[i];
                                    const name = read_connection(input_data);
                                    const node_name = input_data[0];
                                    const node_index = input_data[1];
                                    const inbound_node_key = node_name + '[' + node_index.toString() + ']';
                                    const node = nodes.get(inbound_node_key);
                                    let type = null;
                                    if (node && node.class_name === 'InputLayer') {
                                        type = getInputType(node);
                                        nodes.delete(name);
                                        nodes.delete(inbound_node_key);
                                    }
                                    const value = values.map(name, type);
                                    const argument = new keras.Argument(node_name, true, [ value ]);
                                    this._inputs.push(argument);
                                }
                            }
                            if (Array.isArray(config.output_layers)) {
                                for (let i = 0; i < config.output_layers.length; i++) {
                                    const output_data = config.output_layers[i];
                                    const name = output_data[0];
                                    const key = read_connection(output_data);
                                    const value = values.map(key);
                                    const argument = new keras.Argument(name, true, [ value ]);
                                    this._outputs.push(argument);
                                }
                            }
                        } else {
                            for (const layer of config.layers) {
                                layer.inputs = [];
                                layer.outputs = [];
                                layer.args = {};
                                nodes.set(layer.name + '[0]', layer);
                            }
                        }
                    }
                    for (const entry of nodes) {
                        const node = new keras.Node(this._metadata, entry[1], group, weights, values);
                        this._nodes.push(node);
                    }
                    break;
                }
                default: {
                    throw new keras.Error('\'' + config.class_name + '\' is not supported.');
                }
            }
        } else if (weights) {
            for (const name of weights.keys()) {
                if (weights.get('', name).length <= 6) {
                    const layer = { class_name: 'Weights', config: { name: name } };
                    const node = new keras.Node(metadata, layer, '', weights, values);
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

    constructor(metadata, layer, group, weights, values) {
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
        let class_name = layer.class_name;
        let model = false;
        switch (class_name) {
            case '__Function__': {
                this._type = layer.__type__;
                model = true;
                break;
            }
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
                this._inputs = [ new keras.Argument('inputs', true, inputs.map((input) => values.map(input.name))) ];
                this._outputs = [ new keras.Argument('outputs', true, outputs.map((name) => values.map(name))) ];
                inputs = [];
                outputs = [];
                break;
            }
            case 'Wrapper':
            case 'Bidirectional':
            case 'TimeDistributed': {
                if (config && config.layer) {
                    const inner = config.layer;
                    delete config.layer;
                    this._inner = new keras.Node(metadata, inner, null, null, values);
                    if (class_name == 'Bidirectional' && inner.config.name) {
                        names = [ name + '/forward_' + inner.config.name, name + '/backward_' + inner.config.name ];
                        if (!group) {
                            group = name;
                        }
                    }
                }
                this._type = metadata.type(class_name) || { name: class_name };
                break;
            }
            case 'TFOpLambda': {
                if (config && config.function) {
                    class_name = config.function;
                    delete config.function;
                }
                this._type = metadata.type(class_name) || { name: class_name };
                break;
            }
            default: {
                this._type = metadata.type(class_name) || { name: class_name };
                break;
            }
        }

        if (layer._trainable_variables) {
            if (inputs.length === 0 && Array.isArray(this._type.inputs) && this._type.inputs.length > 0) {
                // weights-only, remove 'input' from type metadata
                this._type = Object.assign({}, this._type);
                this._type.inputs = this._type.inputs.slice(1);
            }
            for (const variable of layer._trainable_variables) {
                inputs.push({ name: '', initializer: variable });
            }
        } else if (weights && !model) {
            for (const name of names) {
                let tensors = weights.get(group, name);
                if (tensors.length > 0) {
                    for (const initializer of tensors) {
                        inputs.push({ name: initializer.name, initializer: initializer });
                    }
                } else {
                    tensors = weights.get('', name);
                    for (const initializer of tensors) {
                        inputs.push({ name: initializer.name, initializer: initializer });
                    }
                }
            }
        }

        if (config && !Array.isArray(config)) {
            for (const entry of Object.entries(config)) {
                const name = entry[0];
                const value = entry[1];
                if (class_name !== 'Activation' && name === 'activation' && value !== 'linear') {
                    if (typeof value === 'string') {
                        const config = { activation: value };
                        const node = new keras.Node(metadata, { class_name: 'Activation', config: config }, null, null, value);
                        this.chain.push(node);
                    } else if (value && typeof value.class_name === 'string' && value.config) {
                        const type = value.class_name;
                        if (!metadata.type(type)) {
                            metadata.add(type, { name: type, category: 'Activation' });
                        }
                        const node = new keras.Node(metadata, value, null, null, value);
                        this.chain.push(node);
                    }
                }
                if (name !== 'name' && name !== 'batch_input_shape') {
                    const attribute = new keras.Attribute(metadata.attribute(class_name, name), name, value);
                    this._attributes.push(attribute);
                }
            }
        }

        const innerType = this.inner ? this.inner.type : null;
        const innerMetadata = innerType ? metadata.type(innerType) : null;
        let inputIndex = 0;
        while (inputs.length > 0) {
            let list = false;
            let name = null;
            let visible = true;
            if (!innerMetadata || inputIndex == 0) {
                if (this._type && this._type.inputs && inputIndex < this._type.inputs.length) {
                    const input = this._type.inputs[inputIndex];
                    name = input.name;
                    if (class_name === 'BatchNormalization' && name === 'gamma' && config.scale === false) {
                        inputIndex++;
                        continue;
                    }
                    visible = input.visible == false ? false : true;
                    if (this._type.inputs[inputIndex].list) {
                        list = true;
                    }
                }
            } else {
                switch (class_name) {
                    case 'Bidirectional': {
                        let innerIndex = inputIndex;
                        if (innerMetadata && innerMetadata.inputs) {
                            if (innerIndex < innerMetadata.inputs.length) {
                                name = 'forward_' + innerMetadata.inputs[innerIndex].name;
                            } else {
                                innerIndex = innerIndex - innerMetadata.inputs.length + 1;
                                if (innerIndex < innerMetadata.inputs.length) {
                                    name = 'backward_' + innerMetadata.inputs[innerIndex].name;
                                }
                            }
                        }
                        visible = false;
                        break;
                    }
                    case 'TimeDistributed':
                        if (innerMetadata && innerMetadata.inputs && inputIndex < innerMetadata.inputs.length) {
                            name = innerMetadata.inputs[inputIndex].name;
                        }
                        break;
                    default:
                        break;
                }
            }
            const input = !list ? [ inputs.shift() ] : inputs.splice(0, inputs.length);
            const inputArguments = input.map((input) => {
                if (input.name) {
                    return values.map(input.name, null, input.initializer);
                }
                if (input.initializer) {
                    return values.map(input.name, null, input.initializer);
                }
                if (input.value !== undefined) {
                    const tensor = new keras.Tensor('', input.shape, config.dtype || '?', null, null, '|', input.value);
                    return values.map('', null, tensor);
                }
                throw new keras.Error("Invalid argument '" + JSON.stringify(input.name) + "'.");
            });
            if (!name && inputArguments.length == 1 && inputArguments[0].initializer && inputArguments[0].initializer.name) {
                if (names.length === 1 && names[0] === '') {
                    name = inputArguments[0].initializer.name;
                } else {
                    const parts = inputArguments[0].initializer.name.split('/').pop().split(':').shift().split('_');
                    const inputName1 = parts.pop();
                    const inputName2 = parts.length > 0 ? [ parts.pop(), inputName1 ].join('_') : '';
                    const inputNames = new Set([ 'recurrent_kernel', 'running_mean', 'running_std', 'moving_mean', 'moving_variance', 'depthwise_filter', 'pointwise_filter' ]);
                    name = inputNames.has(inputName2) ? inputName2 : inputName1;
                }
            }
            const argument = new keras.Argument(name || inputIndex.toString(), visible, inputArguments);
            this._inputs.push(argument);
            inputIndex++;
        }

        for (let i = 0; i < outputs.length; i++) {
            const output = outputs[i];
            const name = this._type && this._type.outputs && i < this._type.outputs.length && this._type.outputs[i] && this._type.outputs[i].name ? this._type.outputs[i].name : i.toString();
            const argument = new keras.Argument(name, true, output === undefined || output.length === 0 ? [] : [ values.map(output) ]);
            this._outputs.push(argument);
        }

        const inputTypes = new Map((this._type.inputs || []).map((input) => [ input.name, input.type ]));
        for (const entry of Object.entries(args)) {
            const name = entry[0];
            const arg = entry[1];
            if (name !== 'name') {
                if ((arg && arg.name) || (inputTypes.has(name) && inputTypes.get(name) === 'Tensor' && arg)) {
                    if (arg.name) {
                        const value = values.map(arg.name);
                        const argument = new keras.Argument(name, true, [ value ]);
                        this._inputs.push(argument);
                    } else {
                        const tensor = new keras.Tensor('', arg.shape, config.dtype || '?', null, null, '|', arg.value);
                        const value = values.map('', null, tensor);
                        const argument = new keras.Argument(name, true, [ value ]);
                        this._inputs.push(argument);
                    }
                } else {
                    const attribute = new keras.Attribute(metadata.attribute(class_name, name), name, arg);
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

    constructor(name, shape, type, stride, quantization, encoding, data) {
        this.name = name;
        this.type = new keras.TensorType(type, new keras.TensorShape(shape));
        this.stride = stride;
        this._quantization = quantization;
        this.encoding = encoding;
        this._data = data;
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
        if (this.encoding === '|') {
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

    get empty() {
        return this._map.size === 0;
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
                this._weights.add(weight.identifier, new keras.Tensor(weight.name, weight.shape, dtype, null, weight.quantization, '<', data));
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