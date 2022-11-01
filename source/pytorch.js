
// Experimental

var pytorch = {};
var python = require('./python');
var base = require('./base');

pytorch.ModelFactory = class {

    match(context) {
        return pytorch.Container.open(context);
    }

    open(context, match) {
        const identifier = context.identifier;
        return pytorch.Metadata.open(context).then((metadata) => {
            const container = match;
            container.metadata = metadata;
            container.exception = (error, fatal) => {
                const message = error && error.message ? error.message : error.toString();
                context.exception(new pytorch.Error(message.replace(/\.$/, '') + " in '" + identifier + "'."), fatal);
            };
            return new pytorch.Model(metadata, container);
        });
    }
};

pytorch.Model = class {

    constructor(metadata, container) {
        this._format = container.format;
        this._producer = container.producer || '';
        this._graphs = container.graphs.map((graph) => new pytorch.Graph(metadata, graph, container));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

pytorch.Graph = class {

    constructor(metadata, graph, container) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        this._name = graph.name || '';
        const type = graph.type;
        switch (type) {
            case 'script': {
                const traced = graph.trace();
                const initializers = new Map();
                if (graph.constants) {
                    for (const constant of graph.constants) {
                        if (pytorch.Utility.isTensor(constant)) {
                            constant.initializer = new pytorch.Tensor(constant.__variable__, constant);
                            initializers.set(constant.__variable__, constant);
                        }
                        else if (constant && constant.__class__ && constant.__class__.__module__ && constant.__class__.__name__) {
                            const type = constant.__class__.__module__ + '.' + constant.__class__.__name__;
                            switch (type) {
                                case '__torch__.torch.classes.xnnpack.LinearOpContext':
                                case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                                case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                                case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                                    for (const key of Object.keys(constant)) {
                                        const value = constant[key];
                                        if (pytorch.Utility.isTensor(value)) {
                                            value.initializer = new pytorch.Tensor(value.__variable__, value);
                                            initializers.set(value.__variable__, value);
                                        }
                                    }
                                    break;
                                default:
                                    throw new pytorch.Error("Unsupported constant context '" + type + "'.");
                            }
                        }
                        else {
                            throw new pytorch.Error('Unsupported constant.');
                        }
                    }
                }
                if (graph.data) {
                    const queue = [ graph.data ];
                    while (queue.length > 0) {
                        const module = queue.shift();
                        if (module.__class__ && module.__class__.__module__ === '__torch__.torch.classes._nnapi' && module.__class__.__name__ === 'Compilation') {
                            continue;
                        }
                        for (const key of Object.keys(module)) {
                            if (key !== '__module__' && key !== '__name__' && key !== '__class__' && key !== '__parent__') {
                                const obj = module[key];
                                if (!Array.isArray(obj) && obj === Object(obj)) {
                                    if (pytorch.Utility.isTensor(obj)) {
                                        const parameter = obj;
                                        parameter.__parent__ = module;
                                        if (!parameter.initializer && parameter.storage()) {
                                            parameter.initializer = new pytorch.Tensor(parameter.name, parameter);
                                        }
                                        if (parameter.__variable__ && parameter.__count__ === 1) {
                                            initializers.set(parameter.__variable__, parameter);
                                        }
                                    }
                                    else if (obj && obj.__class__) {
                                        obj.__parent__ = module;
                                        if (!obj.__id__) {
                                            obj.__id__ = key;
                                        }
                                        queue.push(obj);
                                    }
                                }
                            }
                        }
                    }
                }
                if (traced) {
                    if (graph.inputs) {
                        for (const input of graph.inputs) {
                            this._inputs.push(new pytorch.Parameter(input, true, [
                                new pytorch.Argument(input, null, null)
                            ]));
                        }
                    }
                    if (graph.outputs) {
                        for (const output of graph.outputs) {
                            this._outputs.push(new pytorch.Parameter(output, true, [
                                new pytorch.Argument(output, null, null)
                            ]));
                        }
                    }
                    if (graph.nodes) {
                        for (const node of graph.nodes) {
                            const item = {
                                type: node.type,
                                node: node
                            };
                            this._nodes.push(new pytorch.Node(metadata, '', item, initializers));
                        }
                    }
                }
                if (graph) {
                    this._loadScriptModule(metadata, container, graph.data, initializers);
                }
                break;
            }
            case 'module': {
                this._type = (graph.data.__module__ && graph.data.__name__) ? (graph.data.__module__ + '.' + graph.data.__name__) : '';
                this._loadModule(metadata, graph.data, [], []);
                break;
            }
            case 'weights': {
                for (const state_group of graph.data) {
                    const attributes = state_group.attributes || [];
                    const inputs = state_group.states.map((parameter) => {
                        return new pytorch.Parameter(parameter.name, true,
                            parameter.arguments.map((state) => {
                                const tensor = new pytorch.Tensor(state.id, pytorch.Utility.toTensor(state.value));
                                return new pytorch.Argument(state.id, null, tensor);
                            }));
                    });
                    const obj = {
                        name: state_group.name,
                        type: state_group.type || 'torch.nn.Module',
                        attributes: attributes,
                        inputs: inputs,
                        outputs: []
                    };
                    this._nodes.push(new pytorch.Node(metadata, '', obj, null));
                }
                break;
            }
            default: {
                throw new pytorch.Error("Unsupported container type '" + type + "'.");
            }
        }
    }

    _loadModule(metadata, current, groups, inputs) {

        if (current.__class__ && current.__class__.__module__ !== 'torch.nn.modules.container' && (!current._modules || current._modules.size == 0)) {
            this._createNode(metadata, groups, '', current, inputs, false);
            return [];
        }

        if (!current._modules) {
            throw new pytorch.Error('Module does not contain modules.');
        }

        const sequential = current.__class__ && current.__class__.__module__ === 'torch.nn.modules.container' && current.__class__.__name__ === 'Sequential';

        for (const pair of current._modules) {
            const key = pair[0];
            const value = pair[1];
            if (value) {
                const type = value.__class__.__module__ + '.' + value.__class__.__name__;
                switch (type) {
                    case 'torch.nn.modules.container.Sequential':
                        groups.push(key);
                        inputs = this._loadModule(metadata, value, groups, sequential ? inputs : []);
                        groups.pop(key);
                        break;
                    default: {
                        inputs = this._createNode(metadata, groups, key, value, sequential ? inputs : [], sequential);
                        break;
                    }
                }
            }
        }
        return inputs;
    }

    _createNode(metadata, groups, key, obj, args, output) {

        const type = obj.__class__.__module__ + '.' + obj.__class__.__name__;
        const schema = metadata.type(type);

        let inputSchema = [ { name: 'input'} ];
        if (schema && schema.inputs && schema.inputs.length > 0) {
            inputSchema = schema.inputs.slice();
        }

        const inputName = inputSchema.shift().name;
        const inputs = [];
        if (args.length > 0) {
            inputs.push(new pytorch.Parameter(inputName, true, args.map((argument) => {
                return new pytorch.Argument(argument, null, null);
            })));
        }

        const parameters = obj._parameters || obj._buffers || [];
        for (const parameter of parameters) {
            const key = parameter[0];
            const value = pytorch.Utility.toTensor(parameter[1]);
            let visible = true;
            let inputName = '';
            if (inputSchema.length > 0) {
                const input = inputSchema.shift();
                inputName = input.name;
                visible = input.visible === false ? false : true;
            }
            if (value) {
                const initializer = new pytorch.Tensor('', value);
                inputs.push(new pytorch.Parameter(inputName || key, visible, [ new pytorch.Argument('', null, initializer) ]));
            }
        }

        const group = groups.join('/');
        const name = group ? (group + '/' + key) : key;

        const outputs = output ? [ new pytorch.Parameter('output', true, [ new pytorch.Argument(name, null, null) ]) ] : [];

        const attributes = [];
        for (const name of Object.keys(obj)) {
            if (name.startsWith('_')) {
                continue;
            }
            attributes.push({ name: name, value: obj[name] });
        }
        const item = {
            name: name,
            type: type,
            attributes: attributes,
            children: obj._modules && obj._modules.size > 0 ? true : false,
            inputs: inputs,
            outputs: outputs
        };
        const node = new pytorch.Node(metadata, group, item, {});
        this._nodes.push(node);
        return [ node.name ];
    }

    _loadScriptModule(metadata, container, module, initializers) {
        if (module) {
            if (pytorch.Graph._getParameters(module).length > 0 && !module.__hide__) {
                const item = { module: module };
                this._nodes.push(new pytorch.Node(metadata, '', item, initializers));
            }
            const submodules = pytorch.Graph._getSubmodules(module);
            for (const submodule of submodules) {
                this._loadScriptModule(metadata, container, submodule, initializers);
            }
        }
    }

    static _getParameters(module) {
        const parameters = [];
        if (module && module.__class__.__module__ && module.__class__.__name__) {
            for (const key of Object.keys(module)) {
                if (pytorch.Utility.isTensor(module[key])) {
                    const parameter = module[key];
                    parameter.__id__ = key;
                    parameters.push(parameter);
                }
            }
        }
        return parameters;
    }

    static _getSubmodules(module) {
        const submodules = [];
        if (module && module.__class__ && module.__class__.__module__ && module.__class__.__name__) {
            for (const key of Object.keys(module)) {
                if (!key.startsWith('__')) {
                    const value = module[key];
                    if (value && value.__class__ && value.__class__.__module__ && value.__class__.__name__ && !pytorch.Utility.isTensor(value)) {
                        submodules.push(value);
                    }
                }
            }
        }
        return submodules;
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get groups() {
        return this._groups;
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

pytorch.Parameter = class {

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

pytorch.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new pytorch.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
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

    get initializer() {
        return this._initializer;
    }
};

pytorch.Node = class {

    constructor(metadata, group, item, initializers) {
        this._group = group || '';
        this._name = item.name || '';
        const type = (metadata, name) => {
            if (name instanceof pytorch.nnapi.Graph) {
                return name;
            }
            const type = Object.assign({}, metadata.type(name) || { name: name });
            type.identifier = type.name;
            type.name = type.name.indexOf('::') !== -1 ? type.name.split('::').pop().split('.')[0] : type.name;
            return type;
        };
        if (!item.module && !item.node) {
            this._type = type(metadata, item.type);
            this._nodes = item.children;
            this._inputs = item.inputs;
            this._outputs = item.outputs;
            this._attributes = item.attributes.map((attribute) => {
                const schema = metadata.attribute(this._type.identifier, attribute.name);
                return new pytorch.Attribute(schema, attribute.name, attribute.value);
            });
        }
        else {
            this._attributes = [];
            this._inputs = [];
            this._outputs = [];

            let module = item.module;
            if (module) {
                this._type = { name: 'torch.nn.modules.module.Module' };
                for (const parameter of pytorch.Graph._getParameters(module)) {
                    this._inputs.push(new pytorch.Parameter(parameter.__id__, true, [
                        new pytorch.Argument('', null, parameter.initializer || null)
                    ]));
                    if (parameter.__variable__) {
                        this._outputs.push(new pytorch.Parameter(parameter.__id__, true, [
                            new pytorch.Argument(parameter.__variable__, null, null)
                        ]));
                    }
                }
            }

            if (item.node) {
                this._type = type(metadata, item.type);
                module = null;
                let match = true;
                let count = 0;
                for (const input of item.node.inputs) {
                    for (const argument of input.arguments) {
                        const parameter = initializers.get(argument.id);
                        if (parameter) {
                            if (parameter.__parent__ && (module == null || module == parameter.__parent__)) {
                                module = parameter.__parent__;
                                count++;
                            }
                            else if (parameter.__variable__.startsWith('CONSTANTS.c')) {
                                argument.initializer = parameter.initializer;
                                count++;
                            }
                            else {
                                match = false;
                                break;
                            }
                        }
                    }
                    if (!match) {
                        break;
                    }
                }
                if (module) {
                    const params = pytorch.Graph._getParameters(module).filter((p) => p.__id__ !== 'num_batches_tracked');
                    if (params.length == count && match) {
                        module.__hide__ = true;
                        for (const input of item.node.inputs) {
                            for (const argument of input.arguments) {
                                const parameter = initializers.get(argument.id);
                                if (parameter && parameter.initializer) {
                                    argument.initializer = parameter.initializer;
                                }
                            }
                        }
                    }
                    else {
                        module = null;
                    }
                }

                const node = item.node;
                for (let inputIndex = 0; inputIndex < node.inputs.length; inputIndex++) {
                    const input = node.inputs[inputIndex];
                    if (!input.name) {
                        input.name = inputIndex.toString();
                        if (this._type && this._type.inputs && this._type.inputs.length > inputIndex) {
                            input.name = this._type.inputs[inputIndex].name;
                        }
                    }
                    const args = input.arguments.map((argument) => new pytorch.Argument(argument.id, null, argument.initializer || null));
                    const parameter = new pytorch.Parameter(input.name, true, args);
                    this._inputs.push(parameter);
                }

                for (let i = 0; i < node.outputs.length; i++) {
                    const output = node.outputs[i];
                    if (!output.name) {
                        output.name = i === 0 ? 'output' : 'output' + i.toString();
                        if (this._type && this._type.outputs && i > this._type.outputs.length && this._type.outputs[i] && this._type.outputs[i].name) {
                            output.name = this._type.outputs[i].name;
                        }
                    }
                    const args = output.arguments.map((argument) => new pytorch.Argument(argument.id, null, argument.initializer || null));
                    const parameter = new pytorch.Parameter(output.name, true, args);
                    this._outputs.push(parameter);
                }

                for (const attribute of node.attributes) {
                    const name = attribute.name;
                    const value = attribute.value;
                    const schema = metadata.attribute(this._type.identifier, name);
                    this._attributes.push(new pytorch.Attribute(schema, name, value));
                }
            }
            if (module) {
                if (module.__id__) {
                    let current = module;
                    this._name = current.__id__;
                    while (current.__parent__ != null) {
                        current = current.__parent__;
                        if (!current.__parent__ && !current.__id__) {
                            break;
                        }
                        this._name = [ current.__id__, this._name ].join('.');
                    }
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group;
    }

    get type() {
        return this._type;
    }

    get attributes() {
        return this._attributes;
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

pytorch.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;

        if (this._name === 'training') {
            this._visible = false;
            this._type = 'boolean';
        }
        else if (metadata) {
            if (metadata.type) {
                this._type = metadata.type;
            }
            if (metadata.visible === false) {
                this._visible = false;
            }
            else if (metadata.default !== undefined) {
                if (Array.isArray(value)) {
                    if (Array.isArray(metadata.default)) {
                        this._visible = value.length !== metadata.default || !this.value.every((item, index) => item == metadata.default[index]);
                    }
                    else {
                        this._visible = !this.value.every((item) => item == metadata.default);
                    }
                }
                else {
                    this._visible = this.value !== metadata.default;
                }
            }
        }
        if (Array.isArray(value) && value.length > 0 && value.every((obj) => obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('torch.nn'))) {
            this._value = '?';
        }
    }

    get type() {
        return this._type;
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
};

pytorch.Tensor = class {

    constructor(name, tensor) {
        this._name = name || '';
        const storage = tensor.storage();
        const size = tensor.size();
        this._type = new pytorch.TensorType(storage.dtype.__reduce__(), new pytorch.TensorShape(size));
        const layout = tensor.layout ? tensor.layout.__str__() : null;
        this._stride = tensor.stride();
        if (layout && layout.startsWith('torch.sparse_')) {
            this._layout = layout.split('.').pop().replace('_', '.');
            this._indices = new pytorch.Tensor('', tensor.indices);
            this._values = new pytorch.Tensor('', tensor.values);
        }
        else if (!layout || layout === 'torch.strided') {
            this._data = storage.data;
            this._layout = '<';
            this._indices = null;
        }
        else {
            throw new pytorch.Error("Unsupported tensor layout '" + layout + "'.");
        }
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

    get stride() {
        return this._stride;
    }

    get indices() {
        return this._indices;
    }

    get values() {
        if (this._layout && this._layout.startsWith('sparse.')) {
            return this._values;
        }
        return this._data instanceof Uint8Array ? this._data : this._data.peek();
    }

    decode() {
        if (this._layout !== '<') {
            throw new pytorch.Error("Tensor layout '" + this._layout + "' not implemented.");
        }
        const type = this._type;
        const data = this.values;
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        switch (type.dataType) {
            case 'int16': {
                const array = new Uint16Array(data.length >> 1);
                for (let i = 0; i < array.length; i++) {
                    array[i] = view.getInt16(i << 1, true);
                }
                return array;
            }
            case 'int64': {
                const array = new Uint32Array(data.length >> 3);
                for (let i = 0; i < array.length; i++) {
                    array[i] = view.getUint32(i << 3, true);
                    if (view.getUint32((i << 3) + 4, true) !== 0) {
                        throw new pytorch.Error('Signed 64-bit value exceeds 32-bit range.');
                    }
                }
                return array;
            }
            default: {
                throw new pytorch.Error("Tensor data type '" + type.dataType + "' not implemented.");
            }
        }
    }
};

pytorch.TensorType = class {

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

pytorch.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions || [];
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions && this._dimensions.length > 0) {
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

pytorch.Execution = class extends python.Execution {

    constructor(sources, exceptionCallback) {
        super(sources, exceptionCallback);
        const execution = this;
        this.registerType('__torch__.torch.classes._nnapi.Compilation', class {
            constructor() {
                this.__hide__ = true;
            }
            __init__() {
            }
            init(serialized_model_tensor, parameter_buffers) {
                this.serialized_model_tensor = serialized_model_tensor;
                this.parameter_buffers = parameter_buffers;
                const buffers = parameter_buffers.map((buffer) => buffer.__source__.storage().data);
                const serialized_model = serialized_model_tensor.storage().data;
                this.serialized_model = new pytorch.nnapi.SerializedModel(serialized_model, buffers);
            }
            run(inputs, outputs) {
                this.serialized_model_tensor.__variable__ = this.serialized_model_tensor.__variable__ || execution.variable();
                this.serialized_model_tensor.__count__ = (this.serialized_model_tensor.__count__ || 0) + 1;
                const type = new pytorch.nnapi.Graph(this.serialized_model);
                const input = {
                    arguments: inputs.map((input) => {
                        return { id: input.__variable__ };
                    })
                    // [ { id: this.serialized_model_tensor.__variable__ } ] //,
                    // this.parameter_buffers.map((buffer) => { return { id: buffer.__variable__ }; })
                };
                const output = {
                    arguments: outputs.map((output) => {
                        return { id: output.__variable__ };
                    })
                };
                execution.push({
                    type: type,
                    attributes: [],
                    inputs: [ input ],
                    outputs: [ output ],
                });
            }
        });
        this.registerType('__torch__.torch.classes.quantized.Conv2dPackedParamsBase', class {
            __setstate__(state) {
                const pack_version = state[0];
                if (pack_version !== '2') {
                    throw new pytorch.Error("Unsupported pack version '" + pack_version.toString() + "'.");
                }
                const tensors = state[1];
                const opt_tensors = state[2];
                const packed_config_tensor = new pytorch.Tensor('', tensors[0], true);
                const packed_config = packed_config_tensor.decode();
                this.weight = tensors[1];
                this.bias = opt_tensors[0];
                this.stride = [ packed_config[1], packed_config[2] ];
                this.padding = [ packed_config[3], packed_config[4] ];
                this.dilation = [ packed_config[5], packed_config[6] ];
                this.output_padding = [ packed_config[7], packed_config[8] ];
                this.groups = packed_config[9];
            }
        });
        this.registerType('__torch__.torch.classes.quantized.Conv3dPackedParamsBase', class {
            __setstate__(state) {
                const pack_version = state[0];
                if (pack_version !== '2') {
                    throw new pytorch.Error("Unsupported pack version '" + pack_version.toString() + "'.");
                }
                const tensors = state[1];
                const opt_tensors = state[2];
                const packed_config_tensor = new pytorch.Tensor('', tensors[0], true);
                const packed_config = packed_config_tensor.decode();
                this.weight = tensors[1];
                this.bias = opt_tensors[0];
                this.stride = [ packed_config[1], packed_config[2] ];
                this.padding = [ packed_config[3], packed_config[4] ];
                this.dilation = [ packed_config[5], packed_config[6] ];
                this.output_padding = [ packed_config[7], packed_config[8] ];
                this.groups = packed_config[9];
            }
        });
        this.registerType('__torch__.torch.classes.quantized.LinearPackedParamsBase', class {
            __setstate__(state) {
                this.weight = state[0];
                this.bias = state[1];
            }
        });
        this.registerType('__torch__.torch.classes.xnnpack.Conv2dOpContext', class {
            __setstate__(state) {
                this.weight = state[0];
                this.bias = state[1];
                this.stride = state[2];
                this.padding = state[3];
                this.dilation = state[4];
                this.groups = state[5];
                this.output_min = state[6];
                this.output_max = state[7];
            }
        });
        this.registerType('__torch__.torch.classes.xnnpack.LinearOpContext', class {
            __setstate__(state) {
                this.weight = state[0];
                this.bias = state[1];
                this.output_min = state[2];
                this.output_max = state[3];
            }
        });
    }

    debug(file) {
        const buffer = this.source(file + '.debug_pkl');
        if (buffer) {
            return null;
            // const unpickler = this.invoke('pickle.Unpickler', [ buffer ]);
            // return unpickler.load();
        }
        return null;
    }
};

pytorch.Container = class {

    static open(context) {
        const zip = pytorch.Container.Zip.open(context.entries('zip'));
        if (zip) {
            return zip;
        }
        const pickle = pytorch.Container.Pickle.open(context.stream);
        if (pickle) {
            return pickle;
        }
        const tar = pytorch.Container.Tar.open(context.entries('tar'));
        if (tar) {
            return tar;
        }
        const torch_utils = pytorch.Container.torch_utils.open(context);
        if (torch_utils) {
            return torch_utils;
        }
        return null;
    }
};

pytorch.Container.Tar = class {

    static open(entries) {
        if (entries.has('pickle')) {
            return new pytorch.Container.Tar(entries);
        }
        return null;
    }

    constructor(entries) {
        this._entries = entries;
        this._graphs = [ this ];
    }

    set metadata(value) {
        this._metadata = value;
    }

    set exception(value) {
        this._exceptionCallack = value;
    }

    get format() {
        return 'PyTorch v0.1.1';
    }

    get graphs() {
        if (this._entries) {
            this._type = '';
            this._data = null;
            const execution = new pytorch.Execution(null, this._exceptionCallback);
            const obj = execution.invoke('torch.load', [ this._entries ]);
            const weights = pytorch.Utility.findWeights(obj);
            if (!weights) {
                throw new pytorch.Error('File does not contain root module or state dictionary.');
            }
            for (const graph of weights) {
                graph.type = 'weights';
            }
            this._exceptionCallback = null;
            this._entries = null;
            this._graphs = weights;
        }
        return this._graphs;
    }
};

pytorch.Container.Pickle = class {

    static open(stream) {
        const signature = [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            return new pytorch.Container.Pickle(stream);
        }
        return null;
    }

    constructor(stream) {
        this._stream = stream;
        this._graphs = [];
    }

    set metadata(value) {
        this._metadata = value;
    }

    set exception(value) {
        this._exceptionCallback = value;
    }

    get format() {
        return 'PyTorch v0.1.10';
    }

    get graphs() {
        if (this._stream) {
            const data = this._stream.length < 0x7ffff000 ? this._stream.peek() : this._stream;
            const execution = new pytorch.Execution(null, this._exceptionCallback);
            this._stream = null;
            this._exceptionCallback = null;
            const obj = execution.invoke('torch.load', [ data ]);
            this._graphs = pytorch.Utility.find(obj);
        }
        return this._graphs;
    }
};

pytorch.Container.torch_utils = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length > 1) {
            const buffer = stream.peek(Math.min(1024, stream.length));
            if (buffer[0] === 0x80) {
                const content = String.fromCharCode.apply(null, buffer);
                if (content.indexOf('torch_utils') !== -1) {
                    const obj = context.open('pkl');
                    if (obj && Object.entries(obj).some((entry) => pytorch.Utility.isInstance(entry[1], 'torch.nn.modules.module.Module'))) {
                        return new pytorch.Container.torch_utils(obj);
                    }
                }
            }
        }
        return null;
    }

    constructor(obj) {
        this._obj = obj;
        this._graphs = [];
    }

    get format() {
        return 'PyTorch torch_utils';
    }

    get graphs() {
        if (this._obj) {
            this._graphs = pytorch.Utility.find(this._obj);
            this._obj = null;
        }
        return this._graphs;
    }
};

pytorch.Container.Zip = class {

    static open(entries) {
        if (entries.size > 0) {
            let prefix = [];
            const paths = Array.from(entries.keys()).map((path) => path.split('/').reverse());
            for (;;) {
                const set = new Set(paths.map((path) => path.length > 1 ? path.pop() : null));
                if (set.size !== 1 || set.keys().next().value === null) {
                    break;
                }
                prefix.push(set.keys().next().value);
            }
            prefix = prefix.join('/');
            prefix = prefix.length > 0 ? prefix + '/' : prefix;
            entries = new Map(Array.from(entries).map((entry) => [ entry[0].substring(prefix.length), entry[1] ]));
            if (entries.has('model.json')) {
                try {
                    const stream = entries.get('model.json');
                    const buffer = stream.peek();
                    const decoder = new TextDecoder('utf-8');
                    const content = decoder.decode(buffer);
                    const model = JSON.parse(content);
                    if (model.mainModule) {
                        return new pytorch.Container.Zip.Json(entries, model);
                    }
                }
                catch (error) {
                    // continue regardless of error
                }
            }
            if (entries.has('data.pkl')) {
                return new pytorch.Container.Zip.Pickle(entries);
            }
            if (Array.from(entries.keys()).find((name) => name.startsWith('.data/'))) {
                return new pytorch.Container.Zip.Package(entries);
            }
        }
        return null;
    }

    constructor(entries) {
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
        this._entries = entries;
        this._producer = '';
    }

    set metadata(value) {
        this._metadata = value;
    }

    set exception(value) {
        this._exceptionCallback = value;
    }

    get producer() {
        return this._producer;
    }

    version(name) {
        const stream = this._entries.get(name);
        if (stream) {
            const decoder = new TextDecoder('utf-8');
            const buffer = stream.peek();
            const text = decoder.decode(buffer);
            const value = text.split('\n').shift();
            // https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/inline_container.h
            // kProducedFileFormatVersion
            const versions = new Map([
                [ '1',  'v1.3'  ],
                [ '2',  'v1.5'  ], // 7a2889b014ce36fcc333b2c6de6f29f976652f84 (#28122)
                [ '3',  'v1.6'  ], // 2ec6a30722b0ef85632a2f3e7ce6f80da403008a (#36085)
                [ '4',  'v1.6'  ], // 95489b590f00801bdee7f41783f30874883cf6bb (#38620)
                [ '5',  'v1.7'  ], // cb26661fe4faf26386703180a9045e6ac6d157df (#40364)
                [ '6',  'v1.9'  ], // 3ee7637ffa50df0d9b231c7b40778ac1c390bf4a (#59714)
                [ '7',  'v1.10' ], // 880098a7e34a20628f960daa8eab0eb1ad566c39 (#63651)
                [ '8',  'v1.11' ], // b28e696516a7f0c7a6ead6da967590ce6c1d6698 (#71486)
                [ '9',  'v1.11' ], // 8757e21c6a4fc00e83539aa7f9c28eb11eff53c1 (#72051)
                [ '10', 'v1.12' ]  // 4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8 (#61439)
            ]);
            if (!versions.has(value)) {
                this._exceptionCallback(new pytorch.Error("Unsupported PyTorch Zip version '" + value + "'."));
            }
            return versions.get(value) || 'v-' + value.toString();
        }
        return '';
    }
};

pytorch.Container.Zip.Script = class {

    constructor(entries, execution, location, name) {
        this._entries = entries;
        this._execution = execution;
        this._location = location || {};
        this._name = name || '';
    }

    get name() {
        return this._name;
    }

    get type() {
        return 'script';
    }

    trace() {
        this._inputs = [];
        this._outputs = [];
        this.execution.reset();
        if (this.data.forward) {
            const args = [ this.data ]; // self
            if (this.data.forward.__code__ && this.data.forward.__code__.parameters) {
                for (const parameter of this.data.forward.__code__.parameters) {
                    const defaultValue = (type, name) => {
                        if (type.type === 'type' && type.name.type) {
                            switch (type.name.value) {
                                case 'Tensor': {
                                    const tensor = this.execution.invoke('torch.Tensor', []);
                                    tensor.__variable__ = name;
                                    tensor.__origin__ = 'graph-input';
                                    return tensor;
                                }
                                case 'Tuple': {
                                    return type.arguments.map((type, index) => defaultValue(type, name + '[' + index.toString() + ']'));
                                }
                                case 'List': {
                                    return type.arguments.map((type, index) => defaultValue(type, name + '[' + index.toString() + ']' ));
                                }
                                case 'Dict': {
                                    if (type.arguments[1].name.value === 'Tensor') {
                                        const Dict = class extends Map {
                                            get(key) {
                                                if (!super.has(key)) {
                                                    super.set(key, defaultValue(type.arguments[1], name + ':' + key));
                                                }
                                                return super.get(key);
                                            }
                                        };
                                        return new Dict();
                                    }
                                    return new Map();
                                }
                                case 'int': {
                                    return 0;
                                }
                                case 'float': {
                                    return 0.0;
                                }
                                case 'bool': {
                                    return false;
                                }
                                case 'Optional': {
                                    return undefined;
                                }
                                case 'str':
                                    return '';
                                default: {
                                    break;
                                }
                            }
                        }
                        throw new pytorch.Error("Unsupported parameter type '" + JSON.stringify(type) + "'.");
                    };
                    if (parameter.name !== 'self') {
                        const type = parameter.parameterType;
                        const value = defaultValue(type, parameter.name);
                        if (pytorch.Utility.isTensor(value)) {
                            value.__variable__ = parameter.name;
                            value.__origin__ = 'graph-input';
                            this._inputs.push(parameter.name);
                        }
                        args.push(value);
                    }
                }
            }
            const result = this.data.forward.__call__(args);
            if (Array.isArray(result)) {
                for (const output of result) {
                    if (pytorch.Utility.isTensor(output)) {
                        this._outputs.push(output.__variable__);
                    }
                }
            }
            else if (pytorch.Utility.isTensor(result)) {
                this._outputs.push(result.__variable__);
            }
            else if (Object(result) === result) {
                for (const key of Object.keys(result)) {
                    const value = result[key];
                    if (Array.isArray(value)) {
                        for (const output of value) {
                            if (pytorch.Utility.isTensor(output)) {
                                this._outputs.push(output.__variable__);
                            }
                        }
                    }
                    else if (pytorch.Utility.isTensor(value)) {
                        this._outputs.push(value.__variable__);
                    }
                }
            }
            this._nodes = this.execution.nodes;
            return true;
        }
        throw new pytorch.Error("Module 'forward' not implemented.");
    }

    get execution() {
        const directory = this._location.code || 'code/';
        const sources = new Map();
        for (const entry of this._entries) {
            const name = entry[0];
            if (name.startsWith(directory) && name.endsWith('.py')) {
                const file = name.substring(directory.length);
                if (sources.has(file)) {
                    throw new pytorch.Error("Duplicate source file '" + file + "'.");
                }
                const stream = entry[1];
                const buffer = stream.peek();
                this._execution.add(file, buffer);
                sources.set(file, buffer);
            }
        }
        const torch = this._execution.import('torch');
        this._execution.builtins.torch = torch;
        this._execution.builtins.Tensor = torch.Tensor;
        this._execution.builtins.ops = torch.ops;
        this._execution.builtins.inf = torch.inf;
        const constants = {};
        for (let i = 0; i < this.constants.length; i++) {
            constants['c' + i.toString()] = this.constants[i];
        }
        this._execution.builtins.CONSTANTS = constants;
        return this._execution;
    }

    _unpickle(data, storage_map) {
        const loaded_storages = new Map();
        const execution = this.execution;
        const unpickler = execution.invoke('pickle.Unpickler', [ data ]);
        unpickler.persistent_load = (saved_id) => {
            const typename = saved_id[0];
            switch (typename) {
                case 'storage': {
                    const storage_type = saved_id[1];
                    const root_key = saved_id[2];
                    // const location = saved_id[3];
                    const size = saved_id[4];
                    if (!loaded_storages.has(root_key)) {
                        const storage = new storage_type(size);
                        storage._set_cdata(storage_map.get(root_key));
                        loaded_storages.set(root_key, storage);
                    }
                    const storage = loaded_storages.get(root_key);
                    const view_metadata = saved_id[5];
                    if (view_metadata) {
                        const view_key = view_metadata.shift();
                        view_metadata.shift(); // view_offset
                        view_metadata.shift(); // view_size
                        let view = null;
                        if (loaded_storages.has(view_key)) {
                            view = loaded_storages.get(root_key);
                        }
                        else {
                            view = null; // storage.slice(view_offset, view_offset + view_size);
                            loaded_storages.set(view_key, view);
                        }
                        return view;
                    }
                    return storage;
                }
                default: {
                    throw new pytorch.Error("Unsupported persistent load type '" + typename + "'.");
                }
            }
        };
        return unpickler.load();
    }

    get constants() {
        if (this._constants === undefined) {
            this._constants = [];
            const stream = this._entries.get('constants.pkl');
            if (stream) {
                const buffer = stream.peek();
                this._constants = this._unpickle(buffer, this._storage('constants/'));
                for (let i = 0; i < this._constants.length; i++) {
                    const constant = this._constants[i];
                    const variable = 'CONSTANTS.c' + i.toString();
                    if (pytorch.Utility.isTensor(constant)) {
                        constant.__variable__ = variable;
                    }
                    else if (constant && constant.__class__ && constant.__class__.__module__ && constant.__class__.__name__) {
                        const type = constant.__class__.__module__ + '.' + constant.__class__.__name__;
                        switch (type) {
                            case '__torch__.torch.classes.xnnpack.LinearOpContext':
                            case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                            case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                            case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                                if (pytorch.Utility.isTensor(constant.weight)) {
                                    constant.weight.__variable__ = variable + '.weight';
                                }
                                if (pytorch.Utility.isTensor(constant.bias)) {
                                    constant.bias.__variable__ = variable + '.bias';
                                }
                                break;
                            default:
                                throw new pytorch.Error("Unsupported constant context '" + type + "'.");
                        }
                    }
                    else {
                        throw new pytorch.Error('Unsupported constant.');
                    }
                }
            }
        }
        return this._constants;
    }

    _storage(dirname) {
        const map = new Map();
        const prefix = dirname;
        for (const entry of this._entries) {
            if (entry[0].startsWith(prefix)) {
                const key = entry[0].substring(prefix.length);
                const buffer = entry[1].peek();
                map.set(key, buffer);
            }
        }
        return map;
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

pytorch.Container.Zip.Json = class extends pytorch.Container.Zip {

    constructor(entries, model) {
        super(entries);
        this._producer = model && model.producerName ? model.producerName + (model.producerVersion ? ' v' + model.producerVersion : '') : '';
        this._model = model;
    }

    get format() {
        return this._entries.get('attributes.pkl') ? 'TorchScript v1.1' : 'TorchScript v1.0';
    }

    get graphs() {
        if (!this._graphs) {
            const execution = new pytorch.Container.Zip.Execution(null, this._exceptionCallback, this._metadata);
            const graph = new pytorch.Container.Zip.Json.Script(this._entries, execution, this._model);
            this._graphs = graph.data.forward ? [ graph ] : pytorch.Utility.find(graph.data);
        }
        return this._graphs;
    }
};

pytorch.Container.Zip.Json.Script = class extends pytorch.Container.Zip.Script {

    constructor(entries, execution, model) {
        super(entries);
        this._execution = execution;
        this._model = model;
        this._name = model.mainModule.name || '';
    }

    get name() {
        return this._name;
    }

    get data() {
        if (!this._data) {
            this._data = this._model.mainModule || {};
            const queue = [ this._data ];
            const entries = new Map();
            for (const entry of this._entries) {
                const name = entry[0];
                const stream = entry[1];
                const buffer = stream.peek();
                entries.set(name, buffer);
            }
            const tensorTypeMap = new Map([
                [ 'FLOAT', 'Float' ],
                [ 'FLOAT16', 'Half' ],
                [ 'DOUBLE', 'Double' ],
                [ 'INT8', 'Char' ],
                [ 'INT32', 'Int' ],
                [ 'INT64', 'Long' ]
            ]);
            const constants = this._model.tensors || [];
            this._constants = constants.map((constant) => {
                const key = constant.data.key;
                if (!tensorTypeMap.has(constant.dataType)) {
                    throw new pytorch.Error("Unsupported tensor data type '" + constant.dataType + "'.");
                }
                const type = tensorTypeMap.get(constant.dataType);
                const shape = constant.dims ? constant.dims.map((dim) => parseInt(dim, 10)) : null;
                const storage_type = this.execution.resolve('torch.' + type + 'Storage');
                const size = (shape || []).reduce((a, b) => a * b, 1);
                const offset = parseInt(constant.offset, 10) || 0;
                const storage = new storage_type([ size ]);
                const itemsize = storage.dtype.itemsize();
                const buffer = entries.get(key);
                const length = size * itemsize;
                const data = buffer.slice(offset, offset + length);
                storage._set_cdata(data);
                const tensor = this.execution.invoke('torch._utils._rebuild_tensor', [ storage, 0, shape, 0 ]);
                tensor.name = constant.data.key;
                return tensor;
            });
            this._attributes = [];
            const stream = this._entries.get('attributes.pkl');
            if (stream) {
                const buffer = stream.peek();
                const unpickler = this.execution.invoke('pickle.Unpickler', [ buffer ]);
                const obj = unpickler.load();
                this._attributes.push(...obj);
            }
            while (queue.length > 0) {
                const module = queue.shift();
                if (!module.__class__) {
                    module.__class__ = {
                        __module__: 'torch.nn.modules.module',
                        __name__: 'Module'
                    };
                }
                if (module.name) {
                    module.__id__ = module.name;
                }
                if (module.submodules) {
                    for (const submodule of module.submodules) {
                        module[submodule.name] = submodule;
                        submodule.__parent__ = module;
                        queue.push(submodule);
                    }
                    delete module.submodules;
                }
                const attributes = [];
                if (module.attributes) {
                    attributes.push(...module.attributes);
                    delete module.attributes;
                }
                const parameters = [];
                if (module.parameters) {
                    parameters.push(...module.parameters);
                    delete module.parameters;
                }
                if (module.arguments) {
                    parameters.push(...module.arguments);
                    delete module.arguments;
                }
                for (const parameter of parameters) {
                    const tensor = this._constants[parameter.tensorId];
                    module[parameter.name] = tensor;
                    if (!parameter.__class__) {
                        parameter.__class__ = {
                            __module__: 'torch',
                            __name__: 'Tensor'
                        };
                    }
                }
                for (const attribute of attributes) {
                    module[attribute.name] = this._attributes[attribute.id];
                }
            }
            const code = this._data.torchscriptArena;
            if (code && code.key && code.key.startsWith('code/')) {
                const file = code.key.substring('code/'.length);
                const name = file.replace(/\.py$/, '').split('/').join('.');
                const module = this.execution.import(name);
                if (module.forward.__class__ === this.execution.builtins.function) {
                    this._data.forward = module.forward;
                }
            }
            delete this._model;
        }
        return this._data;
    }
};

pytorch.Container.Zip.Pickle = class extends pytorch.Container.Zip {

    constructor(entries) {
        super(entries);
    }

    get format() {
        const version = this.version('version') || this.version('.data/version');
        return (this._entries.get('constants.pkl') ? 'TorchScript' : 'PyTorch') + (version ? ' ' + version : '');
    }

    get graphs() {
        if (!this._graphs) {
            const execution = new pytorch.Container.Zip.Execution(null, this._exceptionCallback, this._metadata);
            const graph = new pytorch.Container.Zip.Pickle.Script(this._entries, execution);
            if (graph.data && graph.data.forward) {
                this._graphs = [ graph ];
            }
            else if (graph.data && graph.data.__class__ && graph.data.__class__.__module__ == 'fastai.learner' && graph.data.__class__.__name__ == 'Learner') {
                this._graphs = pytorch.Utility.find(graph.data.model);
            }
            else {
                this._graphs = pytorch.Utility.find(graph.data);
            }
        }
        return this._graphs;
    }
};

pytorch.Container.Zip.Pickle.Script = class extends pytorch.Container.Zip.Script {

    constructor(entries, execution, location, name) {
        super(entries, execution, location, name);
    }

    get data() {
        if (!this._data) {
            const stream = this._entries.get(this._location.model || 'data.pkl');
            const buffer = stream.peek();
            this._data = this._unpickle(buffer, this._storage(this._location.data || 'data/'));
        }
        return this._data;
    }
};

pytorch.Container.Zip.Package = class extends pytorch.Container.Zip {

    constructor(entries) {
        super(entries);
    }

    get format() {
        const version = this.version('.data/version');
        return 'PyTorch Package' + (version ? ' ' + version : '');
    }

    get graphs() {
        if (!this._graphs) {
            this._graphs = [];
            const entries = Array.from(this._entries).filter((entry) => !entry[0].startsWith('.data/') && !entry[0].endsWith('py'));
            if (entries.length > 0) {
                const execution = new pytorch.Container.Zip.Execution(null, this._exceptionCallback, this._metadata);
                const torch_jit_script = execution.register('torch.jit._script');
                execution.registerType('torch.package.PackageImporter', class {
                    constructor(entries) {
                        this._entries = entries;
                    }
                    load_pickle(name) {
                        const stream = this._entries.get(name);
                        const loaded_reduces = new Map();
                        const loaded_storages = new Map();
                        const unpickler = execution.invoke('pickle.Unpickler', [ stream ]);
                        unpickler.persistent_load = (saved_id) => {
                            const typename = saved_id.shift();
                            switch (typename) {
                                case 'storage': {
                                    const storage_type = saved_id[0];
                                    const key = saved_id[1];
                                    /* const location = saved_id[2]; */
                                    const size = saved_id[3];
                                    if (!loaded_storages.has(key)) {
                                        const storage = new storage_type(size);
                                        const stream = this._entries.get('.data/' + key + '.storage');
                                        const buffer = stream.peek();
                                        storage._set_cdata(buffer);
                                        loaded_storages.set(key, storage);
                                    }
                                    return loaded_storages.get(key);
                                }
                                case 'reduce_package': {
                                    if (saved_id.left === 2) {
                                        const func = saved_id[0];
                                        const args = saved_id[1];
                                        return execution.invoke(func, args);
                                    }
                                    const reduce_id = saved_id[0];
                                    const func = saved_id[1];
                                    const args = saved_id[2];
                                    if (!loaded_reduces.has(reduce_id)) {
                                        const value = execution.invoke(func, [ this ].concat(args));
                                        loaded_reduces.set(reduce_id, value);
                                    }
                                    return loaded_reduces.get(reduce_id);
                                }
                                default: {
                                    throw new pytorch.Error("Unknown package typename '" + typename + "'.");
                                }
                            }
                        };
                        return unpickler.load();
                    }
                });
                execution.registerFunction('torch.jit._script.unpackage_script_module', function(importer, script_module_id) {
                    return execution.invoke('torch.jit._script.RecursiveScriptModule', [ script_module_id ]);
                });
                execution.registerType('torch.jit._script.ScriptModule', class {});
                execution.registerType('torch.jit._script.RecursiveScriptModule', class extends torch_jit_script.ScriptModule {
                    constructor(script_module_id) {
                        super();
                        this.script_module_id = script_module_id;
                    }
                });
                for (const entry of this._entries) {
                    if (!entry[0].startsWith('.data/') && entry[0].endsWith('.py')) {
                        const name = entry[0];
                        const stream = entry[1];
                        const buffer = stream.peek();
                        execution.add(name, buffer);
                    }
                }
                const importer = execution.invoke('torch.package.PackageImporter', [ new Map(this._entries) ]);
                for (const entry of entries) {
                    const name = entry[0];
                    const root = importer.load_pickle(name);
                    this._graphs.push({
                        name: name,
                        type: 'module',
                        data: root
                    });
                }
            }
        }
        return this._graphs;
    }
};

pytorch.Container.Zip.Execution = class extends pytorch.Execution {

    constructor(sources, exceptionCallback, metadata) {
        super(sources, exceptionCallback);
        this._metadata = metadata;
        this._types = new Map();
        for (const entry of this._metadata._types) {
            const name = entry[1].name;
            if (name.indexOf('::') !== -1) {
                const index = name.lastIndexOf('.');
                const key = index === -1 ? name : name.substring(0, index);
                if (!this._types.has(key)) {
                    this._types.set(key, []);
                }
                this._types.get(key).push(entry[1]);
            }
        }
        this.reset();
    }

    reset() {
        this._nodes = [];
        this._variableIndex = 0;
    }

    get nodes() {
        return this._nodes;
    }

    resolve(name) {
        const index = name.lastIndexOf('.');
        const memberName = index === -1 ? name : name.substring(index + 1, name.length);
        const moduleName = index === -1 ? '' : name.substring(0, index);
        const module = this.import(moduleName);
        let type = module ? module[memberName] : null;
        if (!type) {
            if (name.startsWith('__torch__.')) {
                throw new pytorch.Error("Unknown type name '" + name + "'.");
            }
            type = super.resolve(name);
        }
        return type;
    }

    target(expression, context) {
        if (expression.type === 'id') {
            switch (expression.value) {
                case 'torch':
                case 'ops':
                case 'CONSTANTS':
                case 'uninitialized':
                    return this.builtins[expression.value];
                default:
                    break;
            }
        }
        let current = expression;
        let path = [];
        for (;;) {
            if (current.type === '.' && current.member && current.member.type === 'id') {
                path.push(current.member.value);
                current = current.target;
            }
            else if (current.type === 'id' && current.value !== 'self' && current.value !== 'CONSTANTS') {
                path.push(current.value);
                break;
            }
            else {
                path = null;
                break;
            }
        }
        if (path) {
            let target = null;
            for (let i = path.length - 1; i >= 0; i--) {
                target = target ? target[path[i]] : context.get(path[i]);
                if (!target) {
                    break;
                }
            }
            if (!target) {
                path.reverse();
                const name = path.join('.');
                const file = path.join('/') + '.py';
                if (this.source(file)) {
                    return this.import(name);
                }
                return this.resolve(name);
            }
        }
        return super.target(expression, context);
    }

    call(target, name, args, context) {
        let moduleName = pytorch.Utility.target(target);
        if (moduleName && name) {
            let outputTypes = null;
            let type = moduleName + '.' + name;
            if (type === 'ops.prim.NumToTensor' && args.length === 1 && args[0].type === 'call' && args[0].target.member.type == 'id') {
                const arg = args[0];
                moduleName = pytorch.Utility.target(arg.target.target);
                name = arg.target.member.value;
                args = arg.arguments;
                outputTypes = [ 'int64' ];
                type = moduleName + '.' + name;
            }
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
            let schemas = null;
            if (type.startsWith('torch.')) {
                schemas = this._types.get('aten::' + type.substring(6));
            }
            else if (type.startsWith('ops.')) {
                const path = type.split('.');
                if (path.length === 3) {
                    schemas = this._types.get(path[1] + '::' + path[2]);
                }
                if (!schemas) {
                    const module = this.import(moduleName);
                    if (!module || !module[name]) {
                        const metadata = {};
                        metadata.name = type;
                        metadata.inputs = [];
                        metadata.outputs = [];
                        for (let i = 0; i< args.length; i++) {
                            const input = {};
                            let argument = args[i];
                            input.name = i.toString();
                            if (argument.type === '=' && argument.target && argument.target.type === 'id') {
                                input.name = this.expression(argument.target, context);
                                argument = argument.expression;
                            }
                            const obj = this.expression(argument, context);
                            input.type = pytorch.Utility.getType(obj);
                            metadata.inputs.push(input);
                        }
                        const count = context.target.length > 0 ? context.target[context.target.length - 1].length : 0;
                        for (let i = 0; i < count; i++) {
                            metadata.outputs.push({ name: '', type: '' });
                        }
                        this._metadata.add(type, metadata);
                        schemas = [ metadata ];
                    }
                }
            }
            if (schemas) {
                schemas = !Array.isArray(schemas) ? [ schemas ] : schemas;
                // schemas = schemas.sort((a, b) => b.inputs.length - a.inputs.length);
                const evalArgs = args.map((argument) => {
                    if (argument.type === '=' && argument.target && argument.target.type === 'id') {
                        argument = argument.expression;
                    }
                    return this.expression(argument, context);
                });
                for (const schema of schemas) {
                    const copyArgs = Array.prototype.slice.call(args);
                    const copyEvalArgs = Array.prototype.slice.call(evalArgs);
                    const node = {
                        type: schema.name,
                        inputs: [],
                        attributes: [],
                        outputs: []
                    };
                    const referencedParameters = [];
                    let next = false;
                    const parameters = Array.prototype.slice.call(schema.inputs || []).concat(Array.prototype.slice.call(schema.attributes || []));
                    while (copyEvalArgs.length > 0) {
                        if (parameters.length <= 0) {
                            next = !schema.name.startsWith('_caffe2::');
                            break;
                        }
                        if (copyArgs.every((arg) => arg.type === '=' && arg.target && arg.target.type === 'id') &&
                            parameters.every((parameter) => parameter.type !== 'Tensor' && parameter.type !== 'Tensor[]')) {
                            const map = new Map(parameters.map((parameter) => [ parameter.name, parameter ]));
                            while (copyArgs.length > 0) {
                                const argument = copyArgs.shift();
                                const value = copyEvalArgs.shift();
                                const parameter = map.get(argument.target.value);
                                if (!parameter) {
                                    next = true;
                                    break;
                                }
                                if (!pytorch.Utility.isType(value, parameter.type)) {
                                    if (parameter.optional) {
                                        continue;
                                    }
                                    next = true;
                                    break;
                                }
                                node.attributes.push({ name: parameter.name, value: value });
                            }
                            continue;
                        }
                        if (next) {
                            break;
                        }
                        const parameter = parameters.shift();
                        const argument = copyEvalArgs[0];
                        if (parameter.type === 'Tensor' || (parameter.type === 'Scalar' && pytorch.Utility.isTensor(argument))) {
                            if (Array.isArray(argument) || (!pytorch.Utility.isTensor(argument) && argument !== null && argument !== undefined)) {
                                if (parameter.optional) {
                                    if (argument === undefined) {
                                        copyArgs.shift();
                                        copyEvalArgs.shift();
                                    }
                                    continue;
                                }
                                next = true;
                            }
                            else {
                                copyArgs.shift();
                                copyEvalArgs.shift();
                                const tensor = (argument === null || argument === undefined) ? {} : argument;
                                tensor.__variable__ = tensor.__variable__ || this.variable();
                                referencedParameters.push(tensor);
                                const parameter = {};
                                parameter.arguments = [ { id: tensor.__variable__ } ];
                                node.inputs.push(parameter);
                            }
                        }
                        else if (parameter.type === 'Tensor[]') {
                            const argument = copyEvalArgs[0];
                            if (!Array.isArray(argument) || !argument.every((item) => pytorch.Utility.isTensor(item) || item === null)) {
                                if (parameter.optional) {
                                    continue;
                                }
                                next = true;
                            }
                            else {
                                copyArgs.shift();
                                copyEvalArgs.shift();
                                const parameter = {};
                                parameter.arguments = argument.map((tensor) => {
                                    if (tensor === null) {
                                        tensor = {};
                                    }
                                    tensor.__variable__ = tensor.__variable__ || this.variable();
                                    referencedParameters.push(tensor);
                                    return { id: tensor.__variable__ };
                                });
                                node.inputs.push(parameter);
                            }
                        }
                        else {
                            const arg = copyArgs[0];
                            if (!pytorch.Utility.isType(argument, parameter.type) && argument !== null) {
                                if (parameter.optional) {
                                    continue;
                                }
                                next = true;
                            }
                            else if (arg.type !== '=') {
                                copyArgs.shift();
                                copyEvalArgs.shift();
                                switch (parameter.type) {
                                    case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                                    case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                                    case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                                    case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                                    case '__torch__.torch.classes.xnnpack.LinearOpContext':
                                        for (const entry of Object.entries(argument)) {
                                            const key = entry[0];
                                            if (pytorch.Utility.isTensor(entry[1])) {
                                                const tensor = entry[1];
                                                tensor.__variable__ = tensor.__variable__ || this.variable();
                                                const parameter = {};
                                                parameter.name = /* parameter.name + '.' + */ key;
                                                parameter.arguments = [
                                                    { id: tensor.__variable__ }
                                                ];
                                                referencedParameters.push(tensor);
                                                node.inputs.push(parameter);
                                            }
                                            else {
                                                const attribute = {};
                                                attribute.name = /* parameter.name + '.' + */ key;
                                                attribute.value = entry[1];
                                                node.attributes.push(attribute);
                                            }
                                        }
                                        break;
                                    default:
                                        node.attributes.push({ name: parameter.name, value: argument });
                                        break;
                                }
                            }
                            else {
                                throw new pytorch.Error('Expected named argument.');
                            }
                        }
                        if (next) {
                            break;
                        }
                    }
                    if (next) {
                        continue;
                    }
                    const result = [];
                    for (let i = 0; i < schema.outputs.length; i++) {
                        const parameter = schema.outputs[i];
                        switch (parameter.type) {
                            case 'Tensor': {
                                const output = this.invoke('torch.Tensor', []);
                                output.__origin__ = type;
                                if (i === 0) {
                                    switch (type) {
                                        case 'torch.conv1d':
                                        case 'torch.embedding': {
                                            output.resize_([ NaN, NaN, NaN ]);
                                            break;
                                        }
                                        case 'torch.cat':
                                        case 'torch.conv2d':
                                        case 'torch.dropout':
                                        case 'torch.flatten':
                                        case 'torch.max_pool2d':
                                        case 'torch.adaptive_avg_pool2d':
                                        case 'torch.avg_pool2d':
                                        case 'torch.quantize_per_tensor':
                                        case 'torch.relu_':
                                        case 'torch.hardtanh_':
                                        case 'torch.upsample_bilinear2d':
                                        case 'ops.prepacked.conv2d_clamp_run': {
                                            const input = evalArgs[0];
                                            if (pytorch.Utility.isTensor(input) && input.size() === undefined) {
                                                input.resize_([ NaN, NaN, NaN, NaN ]);
                                            }
                                            output.resize_([ NaN, NaN, NaN, NaN ]);
                                            break;
                                        }
                                        case 'torch.slice': {
                                            const input = evalArgs[0];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                const size = input.size();
                                                output.resize_(size);
                                            }
                                            break;
                                        }
                                        case 'torch.to': {
                                            const input = evalArgs[0];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                const size = input.size();
                                                output.resize_(size);
                                            }
                                            break;
                                        }
                                        case 'torch.conv3d': {
                                            output.resize_([ NaN, NaN, NaN, NaN, NaN ]);
                                            break;
                                        }
                                        case 'torch.detach':
                                        case 'torch.mean':
                                        case 'torch.mul':
                                        case 'torch.div':
                                        case 'torch.batch_norm':
                                        case 'torch.gelu':
                                        case 'torch.relu':
                                        case 'torch.clamp_':
                                        case 'torch._add_relu_':
                                        case 'torch.hardswish_': {
                                            const input = evalArgs[0];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                output.resize_(input.size());
                                            }
                                            break;
                                        }
                                        case 'torch.add':
                                        case 'torch.sub': {
                                            const input = evalArgs[0];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                output.resize_(input.size());
                                            }
                                            else {
                                                const other = evalArgs[1];
                                                if (pytorch.Utility.isTensor(other) && Array.isArray(other.size())) {
                                                    output.resize_(other.size());
                                                }
                                            }
                                            break;
                                        }
                                        case 'torch.select': {
                                            const input = evalArgs[0];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                output.resize_(Array(input.size().length - 1).fill(NaN));
                                            }
                                            break;
                                        }
                                        case 'torch.layer_norm': {
                                            const input = evalArgs[0];
                                            const normalized_shape = evalArgs[1];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                const shape = input.size();
                                                if (Array.isArray(normalized_shape) && normalized_shape.length === 1) {
                                                    shape[shape.length - 1] = normalized_shape[0];
                                                }
                                                output.resize_(shape);
                                            }
                                            break;
                                        }
                                        case 'torch.empty':
                                        case 'torch.ones':
                                        case 'torch.zeros':
                                        case 'torch.zeros_like': {
                                            output.resize_(evalArgs[0]);
                                            break;
                                        }
                                        case 'torch.view':
                                        case 'torch.reshape':
                                        case 'torch.new_full': {
                                            output.resize_(evalArgs[1]);
                                            break;
                                        }
                                        case 'torch.squeeze': {
                                            const input = evalArgs[0];
                                            const size = input.size();
                                            if (Array.isArray(size)) {
                                                switch (evalArgs.length) {
                                                    case 1: {
                                                        output.resize_(size.filter((value) => value !== 1));
                                                        break;
                                                    }
                                                    case 2: {
                                                        const dim = evalArgs[1];
                                                        output.resize_(size.filter((value, index) => (value !== 1 && !isNaN(value)) || index !== dim));
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                            }
                                            break;
                                        }
                                        case 'torch.unsqueeze': {
                                            const input = evalArgs[0];
                                            const size = input.size();
                                            const dim = evalArgs[1];
                                            if (Array.isArray(size) && dim !== undefined) {
                                                const shape = size.slice();
                                                shape.splice(dim, 0, 1);
                                                output.resize_(shape);
                                            }
                                            else {
                                                output.resize_([ NaN, NaN, NaN, NaN ]);
                                            }
                                            break;
                                        }
                                        case 'torch.transpose': {
                                            const input = evalArgs[0];
                                            let dim0 = evalArgs[1];
                                            let dim1 = evalArgs[2];
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                const size = input.size().slice();
                                                dim0 = dim0 >= 0 ? dim0 : size.length + dim0;
                                                dim1 = dim1 >= 0 ? dim1 : size.length + dim1;
                                                const value = size[dim0];
                                                size[dim0] = size[1];
                                                size[dim1] = value;
                                                output.resize_(size);
                                            }
                                            break;
                                        }
                                        case 'ops.quantized.cat':
                                        case 'ops.quantized.cat_relu':
                                        case 'ops.quantized.linear':
                                        case 'ops.quantized.conv2d':
                                        case 'ops.quantized.conv2d_relu':
                                        case 'ops.quantized.add':
                                        case 'ops.quantized.add_relu':
                                            output.resize_([ NaN, NaN, NaN, NaN ]);
                                            output.__quantized__ = true;
                                            break;
                                        case 'torch.contiguous':
                                            output.__source__ = evalArgs[0];
                                            break;
                                        default:
                                            break;
                                    }
                                }
                                output.__variable__ = this.variable();
                                const parameter = {};
                                parameter.arguments = [ { id: output.__variable__ } ];
                                result.push(output);
                                node.outputs.push(parameter);
                                break;
                            }
                            case 'Tensor[]': {
                                let count = 1;
                                switch (type) {
                                    case 'torch.chunk':
                                        count = node.attributes.filter((attribute) => attribute.name == 'chunks')[0].value;
                                        break;
                                    case 'torch.meshgrid':
                                        count = node.inputs[0].arguments.length;
                                        break;
                                    case 'torch.unbind':
                                        count = args[0].__tuple__ || count;
                                        break;
                                    case 'torch.broadcast_tensors':
                                    case 'torch.split':
                                    case 'torch.split_with_sizes':
                                        if (context.target.length > 0) {
                                            count = context.target[context.target.length - 1].length;
                                        }
                                        break;
                                    default:
                                        break;
                                }
                                const tensors = [];
                                const parameter = { arguments: [] };
                                for (let i = 0; i < count; i ++) {
                                    const output = this.invoke('torch.Tensor', []);
                                    output.__origin__ = type;
                                    output.__variable__ = this.variable();
                                    tensors.push(output);
                                    parameter.arguments.push({ id: output.__variable__ });
                                }
                                result.push(tensors);
                                node.outputs.push(parameter);
                                break;
                            }
                            default: {
                                if (!outputTypes || schema.outputs.length !== 1 || schema.outputs[0].type !== outputTypes[0]) {
                                    next = true;
                                    break;
                                }
                                const parameter = { arguments: [] };
                                const output = this.invoke('torch.Tensor', []);
                                output.resize_([]);
                                output.__origin__ = type;
                                output.__variable__ = this.variable();
                                parameter.arguments.push({ id: output.__variable__ });
                                result.push(output);
                                node.outputs.push(parameter);
                                break;
                            }
                        }
                    }
                    if (next) {
                        continue;
                    }
                    for (const referencedParameter of referencedParameters) {
                        referencedParameter.__count__ = (referencedParameter.__count__ || 0) + 1;
                    }
                    this.push(node);
                    if (result.length > 1) {
                        return result;
                    }
                    return result[0];
                }
            }
        }
        return super.call(target, name, args, context);
    }

    block(statements, context) {
        statements = Array.prototype.slice.call(statements);
        while (statements.length > 0) {
            if (statements.length > 1) {
                const assign = statements[0];
                const condition = statements[1];
                // _x = torch.ne(torch.len(torch.size(input)), 5)
                // if _x:
                //   ops.prim.RaiseException(...)
                if (assign.type === '=' &&
                    condition.type === 'if' &&
                    pytorch.Utility.isEqual(assign.target, condition.condition) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.ne', 2) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0].arguments[0], 'torch.size', 1) &&
                    condition.then.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.then.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.arguments[0].arguments[0].arguments[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.size) {
                        const number = this.expression(assign.expression.arguments[1], context);
                        const size = tensor.size();
                        if (number >= 3 && number <= 5) {
                            if (!Array.isArray(size) || size.length !== number) {
                                tensor.resize_(Array(number).fill(NaN));
                            }
                        }
                    }
                }
                // _x = torch.ne(torch.dim(input), 5)
                // if _x:
                //   ops.prim.RaiseException(...)
                if (assign.type === '=' &&
                    condition.type === 'if' &&
                    pytorch.Utility.isEqual(assign.target, condition.condition) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.ne', 2) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0], 'torch.dim', 1) &&
                    condition.then.statements.length > 0 &&
                    pytorch.Utility.isCall(condition.then.statements[condition.then.statements.length - 1], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.arguments[0].arguments[0], context);
                    if (pytorch.Utility.isTensor(tensor)) {
                        const size = this.expression(assign.expression.arguments[1], context);
                        tensor.resize_(Array(size).fill(NaN));
                    }
                }
                // _0 = torch.eq(torch.len(torch.size(x)), 2)
                // if _0:
                //   pass
                // else:
                //   ops.prim.RaiseException("AssertionError: ")
                if (assign.type === '=' &&
                    condition.type === 'if' &&
                    pytorch.Utility.isEqual(assign.target, condition.condition) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.eq', 2) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0].arguments[0], 'torch.size', 1) &&
                    condition.else.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.else.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.arguments[0].arguments[0].arguments[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                        const number = this.expression(assign.expression.arguments[1], context);
                        tensor.resize_(Array(number).fill(NaN));
                    }
                }
                // val = torch.slice(torch.size(img), -2)
                // if torch.eq(torch.len(val), 2):
                //   pass
                // else:
                //   ops.prim.RaiseException("AssertionError: ")
                if (assign.type === '=' &&
                    condition.type === 'if' &&
                    pytorch.Utility.isCall(assign.expression, 'torch.slice', 2) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0], 'torch.size', 1) &&
                    pytorch.Utility.isCall(condition.condition, 'torch.eq', 2) &&
                    pytorch.Utility.isCall(condition.condition.arguments[0], 'torch.len', 1) &&
                    pytorch.Utility.isEqual(condition.condition.arguments[0].arguments[0], assign.target) &&
                    condition.else.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.else.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.arguments[0].arguments[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                        const start = this.expression(assign.expression.arguments[1], context);
                        const value = this.expression(condition.condition.arguments[1], context);
                        if (Number.isInteger(start) && start < 0 && Number.isInteger(value) && value > 0) {
                            tensor.resize_(Array(value - start).fill(NaN));
                        }
                    }
                }
            }
            if (statements.length > 1) {
                // getattr_1 = torch.size(x)
                // getitem = torch.slice(getattr_1, -2, 9223372036854775807, 1)
                const size = statements[0];
                const statement = statements[1];
                if (size.type === '=' && statement.type === '=' &&
                    size.target.type === 'id' &&
                    pytorch.Utility.isCall(size.expression, 'torch.size', 1) &&
                    pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                    statement.expression.arguments[0].type === 'id' && size.target.value === statement.expression.arguments[0].value) {
                    const tensor = this.expression(size.expression.arguments[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                        tensor.resize_([ 1, 3, 299, 299 ]);
                    }
                }
            }
            if (statements.length > 1) {
                // _0 = torch.split_with_sizes(...)
                // a, a_1, a_2, = _0
                const statement = statements[0];
                const tuple = statements[1];
                if (statement.type === '=' && statement.target.type === 'id' && statement.expression.type == 'call' &&
                    tuple.type === '=' && tuple.target.type === 'tuple' &&
                    tuple.target.value.every((item) => item.type === 'id') &&
                    tuple.expression.value === statement.target.value) {
                    const containsVariableReference = (queue, value) => {
                        while (queue.length > 0) {
                            const obj = queue.shift();
                            if (obj && obj.type === 'id' && obj.value === value) {
                                return true;
                            }
                            else if (Array.isArray(obj)) {
                                for (const item of obj) {
                                    if (Array.isArray(item) || (Object(item) === item && item.type)) {
                                        queue.push(item);
                                    }
                                }
                            }
                            else if (Object(obj) === obj) {
                                for (const entry of Object.entries(obj)) {
                                    const key = entry[0];
                                    const value = entry[1];
                                    if (key === 'location') {
                                        continue;
                                    }
                                    if (Array.isArray(value)) {
                                        for (const item of value) {
                                            if (Array.isArray(item) || (Object(item) === item && item.type)) {
                                                queue.push(item);
                                            }
                                        }
                                    }
                                    else if (Object(value) === value && value.type) {
                                        queue.push(value);
                                    }
                                }
                            }
                        }
                        return false;
                    };
                    if (!containsVariableReference(statements.slice(2, statements.length - 1), statement.target.value)) {
                        statements[0] = Object.assign({}, statement);
                        statements[0].target = tuple.target;
                        statements.splice(1, 1);
                    }
                }
            }
            const statement = statements.shift();
            // input_shape = torch.slice(torch.size(x), -2, 9223372036854775807, 1)
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                pytorch.Utility.isCall(statement.expression.arguments[0], 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.arguments[0].arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                    tensor.resize_([ 1, 3, 299, 299 ]);
                }
            }
            // torch.slice(ops.prim.shape(input), 0, 2, 1)
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                pytorch.Utility.isCall(statement.expression.arguments[0], 'ops.prim.shape', 1)) {
                const tensor = this.expression(statement.expression.arguments[0].arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // _3 = torch.le(xxxx, torch.dim(f0))
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.le', 2) &&
                pytorch.Utility.isCall(statement.expression.arguments[1], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.arguments[1].arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // if torch.ne(torch.dim(image), 3):
            //   xxxx
            //   ops.prim.RaiseException(_7)
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.condition, 'torch.ne', 2) &&
                pytorch.Utility.isCall(statement.condition.arguments[0], 'torch.dim', 1) &&
                statement.then.statements.length > 0 &&
                pytorch.Utility.isCall(statement.then.statements.slice(-1).pop(), 'ops.prim.RaiseException', 1)) {
                const tensor = this.expression(statement.condition.arguments[0].arguments[0], context);
                const size = this.expression(statement.condition.arguments[1], context);
                if (pytorch.Utility.isTensor(tensor) && Number.isInteger(size) && size < 10) {
                    tensor.resize_(Array.isArray(tensor.shape) && tensor.shape.length > size ? tensor.shape.slice(-size) : Array(size).fill(NaN));
                }
            }
            // if bool(...):
            //   ops.prim.RaiseException(torch.format(_1, dtype))
            // else:
            //   pass
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.condition, 'bool', 1) &&
                statement.then.statements.length > 0 &&
                pytorch.Utility.isCall(statement.then.statements.slice(-1).pop(), 'ops.prim.RaiseException', 1)) {
                statement.condition = { type: 'id', value: 'False' };
            }
            // dim = torch.sub(torch.dim(input), 2)
            if (statement.type === '=' &&
                statement.target.type === 'id' && statement.target.value === 'dim' &&
                pytorch.Utility.isCall(statement.expression, 'torch.sub', 2) &&
                pytorch.Utility.isCall(statement.expression.arguments[0], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.arguments[0].arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // a, b = torch.unbind(size, 0)
            if (statement.type === '=' &&
                statement.target.type === 'tuple' &&
                (pytorch.Utility.isCall(statement.expression, 'torch.unbind', 1) ||
                 pytorch.Utility.isCall(statement.expression, 'torch.unbind', 2))) {
                statement.expression.arguments[0].__tuple__ = statement.target.value.length;
            }
            // a, b, c = torch.size(input)
            if (statement.type === '=' &&
                statement.target.type === 'tuple' &&
                pytorch.Utility.isCall(statement.expression, 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    const dim = statement.target.value.length;
                    tensor.resize_(Array(dim).fill(NaN));
                }
            }
            // x = torch.len(input)
            if (statement.type === '=' &&
                statement.target.type === 'id' &&
                pytorch.Utility.isCall(statement.expression, 'torch.len', 1)) {
                const tensor = this.expression(statement.expression.arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            if (statement.type === '=' &&
                statement.expression.type === 'call' && statement.expression.arguments.length > 0 &&
                pytorch.Utility.isCall(statement.expression.arguments[0], 'torch.size', 2)) {
                const tensor = this.expression(statement.expression.arguments[0].arguments[0], context);
                const dim = this.expression(statement.expression.arguments[0].arguments[1], context);
                if (pytorch.Utility.isTensor(tensor) && Number.isInteger(dim)) {
                    if (tensor.shape === undefined) {
                        tensor.resize_(Array(dim + 1).fill(NaN));
                    }
                    else if (Array.isArray(tensor.shape) && tensor.shape.length <= dim) {
                        tensor.resize_(tensor.shape.concat(Array(dim + 1 - tensor.shape.length).fill(NaN)));
                    }
                }
            }
            if (statement.type === '=' && statement.target.type === 'tuple' &&
                statement.expression.type === 'call' && statement.expression.arguments.length > 0 &&
                pytorch.Utility.isCall(statement.expression, 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.arguments[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input') {
                    if (tensor.shape === undefined) {
                        tensor.resize_(Array(statement.target.value.length).fill(NaN));
                    }
                }
            }
            const value = this.statement(statement, context);
            if (value !== undefined) {
                return value;
            }
        }
        return undefined;
    }

    push(node) {
        this._nodes.push(node);
    }

    variable() {
        this._variableIndex++;
        return this._variableIndex.toString();
    }
};

pytorch.MemoryFormat = {
    Contiguous: 0,
    Preserve: 1,
    ChannelsLast: 2,
    ChannelsLast3d: 3
};

pytorch.Layout = {
    Strided: 0,
    Sparse: 1,
    Mkldnn: 2
};

pytorch.Utility = class {

    static getScalarType(scalarType) {
        if (!pytorch.Utility._scalarTypes) {
            pytorch.Utility._scalarTypes = [
            ];
        }
        if (scalarType < pytorch.Utility._scalarTypes.length) {
            return pytorch.Utility._scalarTypes[scalarType];
        }
        throw new pytorch.Error("Unsupported scalar type '" + scalarType + "'.");
    }

    static target(expression) {
        if (expression.type == 'id') {
            return expression.value;
        }
        if (expression.type == '.') {
            return pytorch.Utility.target(expression.target) + '.' + pytorch.Utility.target(expression.member);
        }
        return null;
    }

    static isTensor(obj) {
        const name = obj && obj.__class__ ? obj.__class__.__module__ : null;
        switch (name) {
            case 'torch':
            case 'torch.cuda':
                return obj.__class__.__name__.endsWith('Tensor');
            case 'torch.nn.parameter':
                return obj.__class__.__name__ === 'Parameter';
            default:
                return false;
        }
    }

    static toTensor(obj) {
        const name = obj && obj.__class__ ? obj.__class__.__module__ : null;
        switch (name) {
            case 'torch':
            case 'torch.cuda':
                return obj.__class__.__name__.endsWith('Tensor') ? obj : null;
            case 'torch.nn.parameter':
                return obj.__class__.__name__ === 'Parameter' ? obj.data : null;
            default:
                return null;
        }
    }

    static getType(value) {
        if (value === null) {
            return undefined;
        }
        else if (value === true || value === false) {
            return 'boolean';
        }
        else if (pytorch.Utility.isTensor(value)) {
            return 'Tensor';
        }
        else if (typeof value === 'string') {
            return 'string';
        }
        else if (Number(value) === value && value % 1 === 0) {
            return 'int64';
        }
        else if (Number(value) === value) {
            return 'float32';
        }
        else if (Array.isArray(value) && value.every((item) => Number(item) === item && item % 1 === 0)) {
            return 'int64[]';
        }
        else if (Array.isArray(value) && value.every((item) => Number(item) === item)) {
            return 'float32[]';
        }
        throw new pytorch.Error("Unsupported ops argument type '" + JSON.stringify(value).substring(0, 10) + "'.");
    }

    static isType(obj, type) {
        switch (type) {
            case 'Tensor':
                return !Array.isArray(obj) && (pytorch.Utility.isTensor(obj) || obj === null);
            case 'Tensor[]':
                return Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => pytorch.Utility.isTensor(tensor) || tensor === null);
            case 'Scalar':
                return (obj !== null && obj !== Object(obj)) || (pytorch.Utility.isTensor(obj) && Array.isArray(obj.size()) && obj.size().length === 0);
            case 'boolean':
                return obj === true || obj === false;
            case 'string':
                return obj === null || typeof obj === 'string';
            case 'SymInt':
            case 'int64':
                return Number.isInteger(obj) || obj instanceof base.Int64 || (typeof obj === 'number' && isNaN(obj));
            case 'SymInt[]':
            case 'SymInt[2]':
            case 'SymInt[3]':
            case 'SymInt[4]':
            case 'SymInt[5]':
            case 'SymInt[6]':
            case 'int64[]':
            case 'int64[2]':
            case 'int64[3]':
                return Array.isArray(obj) && obj.every((item) => Number.isInteger(item) || (typeof item === 'number' && isNaN(item)) || item === undefined);
            case 'int64[1]':
            case 'SymInt[1]':
                return pytorch.Utility.isType(obj, 'int64') || pytorch.Utility.isType(obj, 'int64[]');
            case 'float32':
            case 'float64':
                return obj !== null && obj !== Object(obj);
            case 'float32[]':
                return Array.isArray(obj) && obj.every((item) => typeof item === 'number' && !isNaN(item));
            case 'string[][]':
                return Array.isArray(obj) && obj.every((item) => Array.isArray(item) && item.every((item) => typeof item === 'string'));
            case 'Layout':
            case 'ScalarType':
            case 'MemoryFormat':
                return Number.isInteger(obj) || obj === null;
            case 'Dimname':
                return obj === null || typeof obj === 'string';
            case 'Dimname[]':
                return Array.isArray(obj) && obj.every((item) => item === null || typeof item === 'string');
            case 'Device':
                return obj === null || obj === Object(obj);
            default:
                if (type && type.startsWith('__torch__.') &&
                    obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                    return type === obj.__class__.__module__ + '.' + obj.__class__.__name__;
                }
                return true;
        }
    }

    static isSubclass(value, name) {
        if (value.__module__ && value.__name__) {
            if (name === value.__module__ + '.' + value.__name__) {
                return true;
            }
        }
        if (value.__bases__) {
            for (const base of value.__bases__) {
                if (pytorch.Utility.isSubclass(base, name)) {
                    return true;
                }
            }
        }
        return false;
    }

    static isInstance(value, name) {
        return value.__class__ ? pytorch.Utility.isSubclass(value.__class__, name) : false;
    }

    static isCall(expression, name, size) {
        if (expression.type === 'call' &&
            expression.arguments.length === size &&
            pytorch.Utility.target(expression.target) === name) {
            return true;
        }
        return false;
    }

    static isEqual(a, b) {
        return (a.type === 'id' && b.type === 'id' && a.value === b.value);
    }

    static find(data) {
        const root = pytorch.Utility.findModule(data);
        if (root) {
            for (const graph of root) {
                graph.type = 'module';
            }
            return root;
        }
        const weights = pytorch.Utility.findWeights(data);
        if (weights) {
            for (const graph of weights) {
                graph.type = 'weights';
            }
            return weights;
        }
        throw new pytorch.Error('File does not contain root module or state dictionary.');
    }

    static findModule(root) {
        if (root) {
            const keys = [ '', 'model', 'net' ];
            for (const key of keys) {
                const obj = key === '' ? root : root[key];
                if (obj && obj instanceof Map && obj.has('engine')) {
                    // https://github.com/NVIDIA-AI-IOT/torch2trt/blob/master/torch2trt/torch2trt.py
                    const data = obj.get('engine');
                    const signature = [ 0x70, 0x74, 0x72, 0x74 ]; // ptrt
                    if (data instanceof Uint8Array && data.length > signature.length && signature.every((value, index) => value === data[index])) {
                        const buffer = data.slice(0, 24);
                        const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                        throw new pytorch.Error("Invalid file content. File contains undocumented PyTorch TensorRT engine data (" + content.substring(8) + ").");
                    }
                }
                if (obj) {
                    if (obj._modules) {
                        return [ { name: '', data: obj } ];
                    }
                    const objKeys = Object.keys(obj).filter((key) => obj[key] && obj[key]._modules);
                    if (objKeys.length > 1) {
                        return objKeys.map((key) => {
                            return { name: key, data: obj[key] };
                        });
                    }
                }
            }
        }
        return null;
    }

    static findWeights(root) {
        if (!root) {
            return null;
        }
        if (root instanceof Map) {
            const obj = {};
            for (const pair of root) {
                const key = pair[0];
                const value = pair[1];
                obj[key] = value;
            }
            root = obj;
        }
        const keys = !Array.isArray(root) ? Object.keys(root) : [];
        if (keys.length > 1) {
            keys.splice(0, keys.length);
        }
        keys.push(...[
            'state_dict', 'state', 'model_state', 'model', 'model_state_dict', 'model_dict', 'net_dict', 'params', 'generator', 'module', 'weights',
            'discriminator', 'g_state', 'network', 'net', 'netG', 'net_states', 'state_dict_stylepredictor', 'state_dict_ghiasi', 'runner', ''
        ]);
        for (const key of keys) {
            const obj = key === '' ? root : root[key];
            let graphs = null;
            graphs = graphs || pytorch.Utility._convertTensor(obj);
            graphs = graphs || pytorch.Utility._convertObjectList(obj);
            graphs = graphs || pytorch.Utility._convertStateDict(obj);
            if (graphs) {
                return graphs;
            }
        }
        return null;
    }

    static _convertTensor(obj) {
        if (obj && pytorch.Utility.isTensor(obj)) {
            const layers = [];
            const argument = { id: '', value: obj };
            const parameter = {};
            parameter.name = 'value';
            parameter.arguments = [ argument ];
            layers.push({ states: [ parameter ] });
            return [ { data: layers } ];
        }
        return null;
    }

    static _convertObjectList(obj) {
        if (obj && Array.isArray(obj)) {
            if (obj.every((item) => typeof item === 'number' || typeof item === 'string')) {
                const layers = [];
                const type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : '?';
                const layer = { type: type, states: [], attributes: [] };
                for (let i = 0; i < obj.length; i++) {
                    const key = i.toString();
                    const value = obj[i];
                    if (pytorch.Utility.isTensor(value)) {
                        layer.states.push({ name: key, arguments: [ { id: '', value: value } ] });
                    }
                    else {
                        layer.attributes.push({ name: key, value: value });
                    }
                }
                layers.push(layer);
                return [ { data: layers } ];
            }
            if (obj.every((item) => item && Object.values(item).filter((value) => pytorch.Utility.isTensor(value)).length > 0)) {
                const layers = [];
                for (const item of obj) {
                    const type = item.__class__ ? item.__class__.__module__ + '.' + item.__class__.__name__ : '?';
                    const layer = { type: type, states: [], attributes: [] };
                    if (item instanceof Map) {
                        return null;
                    }
                    for (const entry of Object.entries(item)) {
                        const key = entry[0];
                        const value = entry[1];
                        if (pytorch.Utility.isTensor(value)) {
                            layer.states.push({ name: key, arguments: [ { id: '', value: value } ] });
                        }
                        else {
                            layer.attributes.push({ name: key, value: value });
                        }
                    }
                    layers.push(layer);
                }
                return [ { data: layers } ];
            }
        }
        return null;
    }

    static _convertStateDict(obj) {
        const clean = (obj) => {
            if (obj && Array.isArray(obj)) {
                return obj;
            }
            if (obj && obj instanceof Map) {
                return obj;
            }
            if (obj && Object(obj) === obj) {
                const target = {};
                const map_count = Object.entries(obj).filter((entry) => entry[1] instanceof Map).length;
                for (const entry of Object.entries(obj)) {
                    const key = entry[0];
                    const value = entry[1];
                    if (key.indexOf('optim') !== -1 || key.indexOf('opt') !== -1) {
                        if (value === null || (value.state && value.param_groups)) {
                            continue;
                        }
                    }
                    if (map_count > 2 && key.endsWith('_avg') && pytorch.Utility.isTensor(value)) {
                        continue;
                    }
                    if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
                        continue;
                    }
                    if (key === '__class__' && value.__module__ && value.__name__) {
                        continue;
                    }
                    if (Array.isArray(value) && (key.indexOf('loss') !== -1 || value.length === 0)) {
                        continue;
                    }
                    if (value && value.__class__ && value.__class__.__module__ === 'datetime' && value.__class__.__name__ === 'datetime') {
                        continue;
                    }
                    if ((key.startsWith('dico_') && Object(value) === value) ||
                        (key.startsWith('best_metrics') && Object(value) === value) ||
                        (key === 'args' && Object(value) === value) ||
                        (key.startsWith('params') && Object(value) === value && (value.id2lang || value.lang2id)) ||
                        (key.startsWith('spk_dict_') && Object(value) === value && Object.keys(value).length === 0) ||
                        (key === 'blk_det')) {
                        continue;
                    }
                    target[key] = value;
                }
                return target;
            }
            return obj;
        };
        const validate = (map) => {
            let tensor = false;
            if (map && map instanceof Map) {
                for (const pair of map) {
                    const key = pair[0];
                    const value = pair[1];
                    const separator = key.indexOf('.') === -1 && key.indexOf('|') !== -1 ? '|' : '.';
                    const keys = key.split(separator);
                    if (keys[keys.length - 1] === '_metadata') {
                        continue;
                    }
                    else if (keys.length >= 2 && keys[keys.length - 2] === '_packed_params') {
                        continue;
                    }
                    else if (pytorch.Utility.isTensor(value)) {
                        tensor = true;
                        continue;
                    }
                    else if (value && Array.isArray(value) && value.every((item) => pytorch.Utility.isTensor(item))) {
                        tensor = true;
                        continue;
                    }
                    else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                        continue;
                    }
                    else if (value === null) {
                        continue;
                    }
                    return false;
                }
            }
            return tensor;
        };
        const flatten = (obj) => {
            if (!obj || Array.isArray(obj) || ArrayBuffer.isView(obj)) {
                return null;
            }
            if (obj instanceof Map) {
                if (validate(obj)) {
                    return obj;
                }
                return null;
            }
            if (Object(obj) !== obj) {
                return null;
            }
            const map = new Map(Object.keys(obj).map((key) => [ key, obj[key] ]));
            if (validate(map)) {
                return map;
            }
            map.clear();
            for (const key of Object.keys(obj)) {
                const value = flatten(obj[key]);
                if (value && value instanceof Map) {
                    for (const pair of value) {
                        map.set(key + '.' + pair[0], pair[1]);
                    }
                    continue;
                }
                return null;
            }
            return map;
        };
        if (!obj) {
            return null;
        }
        obj = clean(obj);
        const map = new Map();
        if (Array.isArray(obj) && obj.every((item) => validate(item))) {
            for (let i = 0; i < obj.length; i++) {
                map.set(i.toString(), flatten(obj[i]));
            }
        }
        else if (obj instanceof Map && validate(obj)) {
            map.set('', flatten(obj));
        }
        else if (Object(obj) === obj && Object.entries(obj).every((entry) => validate(entry[1]))) {
            for (const entry of Object.entries(obj)) {
                map.set(entry[0], entry[1]);
            }
        }
        else if (Object(obj) === obj && Object.entries(obj).every((entry) => pytorch.Utility.isTensor(entry[1]))) {
            map.set('', new Map(Object.keys(obj).map((key) => [ key, obj[key] ])));
        }
        else {
            const value = flatten(obj);
            if (value) {
                map.set('', value);
            }
        }
        if (map.size > 0) {
            const graphs = [];
            for (const entry of map) {
                const graph_key = entry[0];
                const layer_map = entry[1];
                const layers = new Map();
                for (const item of layer_map) {
                    const key = item[0];
                    const value = item[1];
                    let layerName = '';
                    let parameter = '';
                    const separator = key.indexOf('.') === -1 && key.indexOf('|') !== -1 ? '|' : '.';
                    const keys = key.split(separator);
                    if (keys[keys.length - 1] === '_metadata') {
                        continue;
                    }
                    if (keys.length >= 2 && keys[keys.length - 2] === '_packed_params') {
                        parameter = keys.slice(-2).join(separator);
                        keys.pop();
                        keys.pop();
                    }
                    else {
                        parameter = keys.pop();
                        if (keys.length < 0) {
                            keys.push('');
                        }
                    }
                    layerName = keys.join(separator);
                    if (!layers.has(layerName)) {
                        layers.set(layerName, { name: layerName, states: [], attributes: [] });
                    }
                    const layer = layers.get(layerName);
                    if (pytorch.Utility.isTensor(value)) {
                        layer.states.push({ name: parameter, arguments: [ { id: key, value: value } ] });
                        if (layer.name == '' && layer.states.length > 12) {
                            return null;
                        }
                    }
                    else if (value && Array.isArray(value) && value.every((item) => pytorch.Utility.isTensor(item))) {
                        layer.states.push({ name: parameter, arguments: value.map((item) => {
                            return { id: '', value: item };
                        }) });
                    }
                    else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                        layer.attributes.push({ name: parameter, value: value });
                    }
                }
                graphs.push({
                    name: graph_key,
                    data: layers.values()
                });
            }
            return graphs;
        }
        return null;
    }
};

pytorch.nnapi = {};

pytorch.nnapi.SerializedModel = class {

    constructor(serialized_model, buffers) {
        const reader = new base.BinaryReader(serialized_model);
        this.version = reader.int32();
        if (this.version !== 1) {
            throw new pytorch.Error('Invalid NNAPI serialized model version.');
        }
        const operands = new Array(reader.int32());
        const values = new Array(reader.int32());
        this.operations = new Array(reader.int32());
        this.inputs = new Array(reader.int32());
        this.outputs = new Array(reader.int32());
        const data_types = new Map([
            [ 0, 'float32' ],
            [ 1, 'int32' ],
            [ 2, 'uint32' ],
            [ 3, 'float32[]' ],
            [ 4, 'int32[]' ],
            [ 5, 'quant8_asymm[]' ],
            [ 6, 'boolean' ],
            [ 7, 'quant16_symm[]' ],
            [ 8, 'float16[]' ],
            [ 9, 'boolean[]' ],
            [ 10, 'float16' ],
            [ 11, 'quant8_symm_per_channel[]' ],
            [ 12, 'quant16_asymm[]' ],
            [ 13, 'quant8_symm[]' ],
            [ 14, 'quant8_asymm_signed[]' ],
            [ 16, 'model' ]
        ]);
        for (let i = 0; i < operands.length; i++) {
            const data_type = reader.int32();
            operands[i] = {
                index: i,
                data_type: data_types.has(data_type) ? data_types.get(data_type) : data_type,
                dimensions: new Array(reader.uint32()),
                scale: reader.float32(),
                zero_point: reader.int32()
            };
        }
        for (let i = 0; i < values.length; i++) {
            values[i] = {
                index: reader.int32(),
                source_type: reader.int32(),
                source_length: reader.uint32()
            };
        }
        for (let i = 0; i < this.operations.length; i++) {
            this.operations[i] = {
                index: reader.int32(),
                location: i,
                inputs: new Array(reader.uint32()),
                outputs: new Array(reader.uint32())
            };
        }
        for (const operand of operands) {
            for (let i = 0; i< operand.dimensions.length; i++) {
                operand.dimensions[i] = reader.uint32();
            }
        }
        for (const value of values) {
            const index = value.index;
            const operand = operands[index];
            switch (value.source_type) {
                case 0: { // immediate
                    switch (operand.data_type) {
                        case 'boolean':
                            operand.value = reader.byte() ? true : false;
                            reader.skip(3);
                            break;
                        case 'int32':
                            operand.value = reader.int32();
                            break;
                        case 'float32':
                            operand.value = reader.float32();
                            break;
                        case 'int32[]':
                            operand.data = reader.read(value.source_length);
                            break;
                        case 'float32[]':
                            operand.data = reader.read(value.source_length);
                            break;
                        default:
                            throw new pytorch.Error("Unsupported NNAPI operand type '" + operand.data_type.toString() + "'.");
                    }
                    break;
                }
                case 2: { // numbered buffer
                    if (value.source_length !== 12) {
                        throw new pytorch.Error('Invalid NNAPI numbered buffer source length.');
                    }
                    const number = reader.uint32();
                    const offset = reader.uint32();
                    const operand_length = reader.uint32();
                    const buffer = buffers[number];
                    operand.data = buffer.slice(offset, operand_length);
                    break;
                }
                case 3: { // numbered memory
                    throw new pytorch.Error('NNAPI numbered memory buffer not implemented.');
                }
                default: {
                    throw new pytorch.Error('Unsupported NNAPI value source type.');
                }
            }
        }
        for (const operation of this.operations) {
            for (let i = 0; i< operation.inputs.length; i++) {
                const index = reader.uint32();
                operation.inputs[i] = operands[index];
            }
            for (let i = 0; i< operation.outputs.length; i++) {
                const index = reader.uint32();
                operation.outputs[i] = operands[index];
            }
        }
        for (let i = 0; i< this.inputs.length; i++) {
            const index = reader.uint32();
            this.inputs[i] = operands[index];
        }
        for (let i = 0; i< this.outputs.length; i++) {
            const index = reader.uint32();
            this.outputs[i] = operands[index];
        }

        if (reader.position !== reader.length) {
            throw new pytorch.Error('Invalid NNAPI serialized model length.');
        }
    }
};

pytorch.nnapi.Metadata = class {

    constructor() {
        this._types = new Map();
        // https://developer.android.com/ndk/reference/group/neural-networks
        // https://github.com/pytorch/pytorch/commits/master/torch/backends/_nnapi/serializer.py
        this.register( 0, 'ADD', '', [ 'A', 'B' ], [ [ 'activation', 'int32'] ], [ 'C' ]);
        this.register( 1, 'AVERAGE_POOL_2D', 'Pool', [ 'input' ], [ [ 'padding_left', 'int32' ], [ 'padding_right', 'int32' ], [ 'padding_top', 'int32' ], [ 'padding_bottom', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'filter_x', 'int32' ], [ 'filter_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ] ], [ 'output' ]);
        this.register( 1, 'AVERAGE_POOL_2D', 'Pool', [ 'input' ], [ [ 'padding_scheme', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'filter_x', 'int32' ], [ 'filter_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ] ], [ 'output' ]);
        this.register( 2, 'CONCATENATION');
        this.register( 3, 'CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_left', 'int32' ], [ 'padding_right', 'int32' ], [ 'padding_top', 'int32' ], [ 'padding_bottom', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register( 3, 'CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_scheme', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register( 4, 'DEPTHWISE_CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_left', 'int32' ], [ 'padding_right', 'int32' ], [ 'padding_top', 'int32' ], [ 'padding_bottom', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register( 4, 'DEPTHWISE_CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_scheme', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register( 5, 'DEPTH_TO_SPACE');
        this.register( 6, 'DEQUANTIZE');
        this.register( 7, 'EMBEDDING_LOOKUP');
        this.register( 8, 'FLOOR');
        this.register( 9, 'FULLY_CONNECTED', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'activation', 'int32' ] ], [ 'output' ]);
        this.register(10, 'HASHTABLE_LOOKUP');
        this.register(11, 'L2_NORMALIZATION');
        this.register(12, 'L2_POOL_2D', 'Pool');
        this.register(13, 'LOCAL_RESPONSE_NORMALIZATION');
        this.register(14, 'LOGISTIC');
        this.register(15, 'LSH_PROJECTION');
        this.register(16, 'LSTM', 'Layer');
        this.register(17, 'MAX_POOL_2D', 'Pool');
        this.register(18, 'MUL');
        this.register(19, 'RELU', 'Activation', [ 'input' ], [], [ 'output' ]);
        this.register(20, 'RELU1', 'Activation');
        this.register(21, 'RELU6', 'Activation');
        this.register(22, 'RESHAPE', 'Shape', [ 'input', 'shape' ], [], [ 'output' ]);
        this.register(23, 'RESIZE_BILINEAR');
        this.register(24, 'RNN', 'Layer');
        this.register(25, 'SOFTMAX', 'Activation');
        this.register(26, 'SPACE_TO_DEPTH');
        this.register(27, 'SVDF');
        this.register(28, 'TANH');
        this.register(29, 'BATCH_TO_SPACE_ND');
        this.register(30, 'DIV');
        this.register(31, 'MEAN');
        this.register(32, 'PAD');
        this.register(33, 'SPACE_TO_BATCH_ND');
        this.register(34, 'SQUEEZE');
        this.register(35, 'STRIDED_SLICE');
        this.register(36, 'SUB');
        this.register(37, 'TRANSPOSE');
        this.register(38, 'ABS');
        this.register(39, 'ARGMAX');
        this.register(40, 'ARGMIN');
        this.register(41, 'AXIS_ALIGNED_BBOX_TRANSFORM');
        this.register(42, 'BIDIRECTIONAL_SEQUENCE_LSTM');
        this.register(43, 'BIDIRECTIONAL_SEQUENCE_RNN');
        this.register(44, 'BOX_WITH_NMS_LIMIT');
        this.register(45, 'CAST');
        this.register(46, 'CHANNEL_SHUFFLE');
        this.register(47, 'DETECTION_POSTPROCESSING');
        this.register(48, 'EQUAL');
        this.register(49, 'EXP');
        this.register(50, 'EXPAND_DIMS');
        this.register(51, 'GATHER');
        this.register(52, 'GENERATE_PROPOSALS');
        this.register(53, 'GREATER');
        this.register(54, 'GREATER_EQUAL');
        this.register(55, 'GROUPED_CONV_2D');
        this.register(56, 'HEATMAP_MAX_KEYPOINT');
        this.register(57, 'INSTANCE_NORMALIZATION');
        this.register(58, 'LESS');
        this.register(59, 'LESS_EQUAL');
        this.register(60, 'LOG');
        this.register(61, 'LOGICAL_AND');
        this.register(62, 'LOGICAL_NOT');
        this.register(63, 'LOGICAL_OR');
        this.register(64, 'LOG_SOFTMAX');
        this.register(65, 'MAXIMUM');
        this.register(66, 'MINIMUM');
        this.register(67, 'NEG');
        this.register(68, 'NOT_EQUAL');
        this.register(69, 'PAD_V2');
        this.register(70, 'POW');
        this.register(71, 'PRELU');
        this.register(72, 'QUANTIZE');
        this.register(73, 'QUANTIZED_16BIT_LSTM');
        this.register(74, 'RANDOM_MULTINOMIAL');
        this.register(75, 'REDUCE_ALL');
        this.register(76, 'REDUCE_ANY');
        this.register(77, 'REDUCE_MAX');
        this.register(78, 'REDUCE_MIN');
        this.register(79, 'REDUCE_PROD');
        this.register(80, 'REDUCE_SUM');
        this.register(81, 'ROI_ALIGN');
        this.register(82, 'ROI_POOLING');
        this.register(83, 'RSQRT');
        this.register(84, 'SELECT');
        this.register(85, 'SIN');
        this.register(86, 'SLICE');
        this.register(87, 'SPLIT');
        this.register(88, 'SQRT');
        this.register(89, 'TILE');
        this.register(90, 'TOPK_V2');
        this.register(91, 'TRANSPOSE_CONV_2D', 'Layer');
        this.register(92, 'UNIDIRECTIONAL_SEQUENCE_LSTM', 'Layer');
        this.register(93, 'UNIDIRECTIONAL_SEQUENCE_RNN', 'Layer');
        this.register(94, 'RESIZE_NEAREST_NEIGHBOR');
        this.register(95, 'QUANTIZED_LSTM', 'Layer');
        this.register(96, 'IF');
        this.register(97, 'WHILE');
        this.register(98, 'ELU', 'Activation');
        this.register(99, 'HARD_SWISH', 'Activation');
        this.register(100, 'FILL');
        this.register(101, 'RANK');
    }

    register(index, name, category, inputs, attributes, outputs) {
        inputs = inputs || [];
        outputs = outputs || [];
        attributes = attributes || [];
        const type = {
            name: name,
            inputs: inputs.map((name) => {
                return { name: name, type: 'Tensor' };
            }),
            outputs: outputs.map((name) => {
                return { name: name, type: 'Tensor' };
            }),
            attributes: attributes.map((pair) => {
                return { name: pair[0], type: pair[1] };
            })
        };
        if (category) {
            type.category = category;
        }
        if (!this._types.has(index)) {
            this._types.set(index, []);
        }
        this._types.get(index).push(type);
    }

    type(index, signature) {
        if (!this._types.has(index)) {
            this._types.set(index, { name: index.toString(), inputs: [], outputs: [], attributes: [] });
        }
        const types = this._types.get(index);
        for (const type of types) {
            const inputs = type.inputs.concat(type.attributes);
            if (signature.length < inputs.length) {
                let match = true;
                for (let i = 0; i < inputs.length; i++) {
                    const input = inputs[i];
                    if (input.type === undefined || input.type === 'Tensor' || input.type === signature[i]) {
                        continue;
                    }
                    match = false;
                }
                if (match) {
                    return type;
                }
            }
        }
        return types[0];
    }
};

pytorch.nnapi.Graph = class {

    constructor(model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        const args = new Map();
        const arg = (operand) => {
            if (!args.has(operand.index)) {
                const argument = new pytorch.nnapi.Argument(operand);
                args.set(operand.index, argument);
            }
            return args.get(operand.index);
        };

        const metadata = new pytorch.nnapi.Metadata();
        for (const operation of model.operations) {
            const node = new pytorch.nnapi.Node(metadata, operation, arg);
            this._nodes.push(node);
        }

        for (let i = 0; i < model.inputs.length; i++) {
            const operand = model.inputs[i];
            const argument = arg(operand);
            const parameter = new pytorch.Parameter(i.toString(), true, [ argument ]);
            this._inputs.push(parameter);
        }

        for (let i = 0; i < model.outputs.length; i++) {
            const operand = model.outputs[i];
            const argument = arg(operand);
            const parameter = new pytorch.Parameter(i.toString(), true, [ argument ]);
            this._outputs.push(parameter);
        }
    }

    get name() {
        return 'torch.classes._nnapi.Compilation';
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

pytorch.nnapi.Argument = class {

    constructor(operand) {
        this._name = operand.index.toString();
        const shape = new pytorch.TensorShape(operand.dimensions);
        let dataType = operand.data_type.replace('[]', '');
        let quantizationType = null;
        switch (dataType) {
            case 'quant8_asymm':
            case 'quant8_symm_per_channel':
            case 'quant8_symm':
            case 'quant8_asymm_signed[]':
            case 'quant16_asymm':
            case 'quant16_symm':
                quantizationType = dataType;
                dataType = dataType.indexOf('16') !== -1 ? 'uint16' : 'uint8';
                break;
            default:
                break;
        }
        this._type = new pytorch.TensorType(dataType, shape);
        this._initializer = operand.data ? new pytorch.nnapi.Tensor(this._type, operand.data) : null;
        if (quantizationType || operand.scale !== undefined || operand.zero_point !== undefined) {
            this._quantization = {};
            if (quantizationType) {
                this._quantization.type = quantizationType;
            }
            if (operand.scale !== undefined) {
                this._quantization.scale = operand.scale;
            }
            if (operand.zero_point !== undefined) {
                this._quantization.zeroPoint = operand.zero_point;
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        if (this._quantization) {
            let value = '';
            if (this._quantization.scale != 0 || this._quantization.zeroPoint != 0) {
                value = this._quantization.scale.toString() + ' * ' + (this._quantization.zeroPoint == 0 ? 'q' : ('(q - ' + this._quantization.zeroPoint.toString() + ')'));
            }
            if (this._quantization.type) {
                return this._quantization.type + '(' + value + ')';
            }
            return value;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

pytorch.nnapi.Node = class {

    constructor(metadata, operation, arg) {
        const signature = (operation.inputs || []).map((input) => input.data_type);
        this._type = metadata.type(operation.index, signature);
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._chain = [];

        if (operation.location !== undefined) {
            this._location = operation.location.toString();
        }

        const inputs = this._type.inputs.concat(this._type.attributes);

        if (operation.inputs) {
            for (let i = 0; i < operation.inputs.length; i++) {
                const name = i < inputs.length ? inputs[i].name : i.toString();
                const operand = operation.inputs[i];
                if (operand.dimensions.length > 0) {
                    const argument = arg(operand);
                    const parameter = new pytorch.Parameter(name, true, [ argument ]);
                    this._inputs.push(parameter);
                }
                else if (name === 'activation') {
                    const activation = new Map([ [ 1, 19 ], [ 2, 20 ], [ 3, 21 ] ]).get(operand.value) || 0;
                    if (activation !== 0) {
                        this._chain.push(new pytorch.nnapi.Node(metadata, { index: activation }));
                    }
                }
                else {
                    const attribute = new pytorch.nnapi.Attribute(name, operand);
                    this._attributes.push(attribute);
                }
            }
        }

        if (operation.outputs) {
            for (let i = 0; i < operation.outputs.length; i++) {
                const name = i < inputs.length ? inputs[i].name : i.toString();
                const operand = operation.outputs[i];
                const argument = arg(operand);
                const parameter = new pytorch.Parameter(name, true, [ argument ]);
                this._outputs.push(parameter);
            }
        }
    }

    get type() {
        return this._type;
    }

    get location() {
        return this._location;
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
};

pytorch.nnapi.Attribute = class {

    constructor(name, operand) {
        this._name = name;
        this._type = operand.data_type;
        this._value = operand.value;
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return false;
    }
};

pytorch.nnapi.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get type() {
        return this._type;
    }

    get layout() {
        return '<';
    }

    get values() {
        return this._data;
    }
};

pytorch.Metadata = class {

    static open(context) {
        if (pytorch.Metadata._metadata) {
            return Promise.resolve(pytorch.Metadata._metadata);
        }
        return context.request('pytorch-metadata.json', 'utf-8', null).then((data) => {
            pytorch.Metadata._metadata = new pytorch.Metadata(data);
            return pytorch.Metadata._metadata;
        }).catch(() => {
            pytorch.Metadata._metadata = new pytorch.Metadata(null);
            return pytorch.Metadata._metadata;
        });
    }

    constructor(data) {
        this._types = new Map();
        this._attributes = new Map();
        this._index = new Map();
        if (data) {
            const items = JSON.parse(data);
            for (const item of items) {
                this._types.set(item.name, item);
            }
        }
    }

    add(name, value) {
        this._types.set(name, value);
    }

    type(name) {
        return this._types.get(name);
    }

    attribute(type, name) {
        const attributeName = type + ':' + name;
        if (!this._attributes.has(attributeName)) {
            this._attributes.set(attributeName, null);
            const schema = this.type(type);
            if (schema) {
                if (schema.inputs) {
                    for (const input of schema.inputs) {
                        this._attributes.set(type + ':' + input.name, input);
                    }
                }
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        this._attributes.set(type + ':' + attribute.name, attribute);
                    }
                }
            }
        }
        return this._attributes.get(attributeName);
    }
};

pytorch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading PyTorch model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = pytorch.ModelFactory;
}
