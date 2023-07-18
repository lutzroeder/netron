
// Experimental

var pytorch = {};
var python = require('./python');
var base = require('./base');
var flatbuffers = require('./flatbuffers');

pytorch.ModelFactory = class {

    match(context) {
        return pytorch.Container.open(context);
    }

    async open(context, target) {
        const metadata = await pytorch.Metadata.open(context);
        const container = target;
        container.on('resolve', (_, name) => {
            context.exception(new pytorch.Error("Unknown type name '" + name + "'."), false);
        });
        await container.read(metadata);
        return new pytorch.Model(metadata, container);
    }
};

pytorch.Model = class {

    constructor(metadata, container) {
        this._format = container.format;
        this._producer = container.producer || '';
        this._graphs = [];
        for (const entry of container.modules) {
            const graph = new pytorch.Graph(metadata, entry[0], entry[1]);
            this._graphs.push(graph);
        }
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get graphs() {
        return this._graphs;
    }
};

pytorch.Graph = class {

    constructor(metadata, name, module) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        this._name = name || '';
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (tensor) {
                return new pytorch.Value(name, type || null, tensor);
            }
            if (!args.has(name)) {
                args.set(name, new pytorch.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new pytorch.Error("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        const createNode = (groups, key, obj, args, output) => {
            let type = obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : '?';
            if (type === 'torch.jit._script.RecursiveScriptModule' && obj._c && obj._c.qualified_name) {
                type = obj._c.qualified_name;
            }
            const schema = metadata.type(type);
            const inputSchema = schema && schema.inputs && schema.inputs.length > 0 ? schema.inputs.slice() : [ { name: 'input' } ];
            const inputName = inputSchema.shift().name;
            const inputs = [];
            if (args.length > 0) {
                const argument = new pytorch.Argument(inputName, true, args.map((argument) => arg(argument)));
                inputs.push(argument);
            }
            const filterParameters = (obj) => {
                const entries = Object.entries(obj).filter((entry) => {
                    if (Array.isArray(entry[1]) && entry[1].every((tensor) => pytorch.Utility.isTensor(tensor))) {
                        return true;
                    }
                    return pytorch.Utility.isTensor(entry[1]);
                });
                return new Map(entries);
            };
            const parameters = obj._parameters || obj._buffers || filterParameters(obj);
            for (const entry of parameters) {
                const key = entry[0];
                const list = Array.isArray(entry[1]) ? entry[1].map((item) => pytorch.Utility.toTensor(item)) : [ pytorch.Utility.toTensor(entry[1]) ];
                let visible = true;
                let name = '';
                if (inputSchema.length > 0) {
                    const input = inputSchema.shift();
                    name = input.name;
                    visible = input.visible === false ? false : true;
                }
                if (list) {
                    const args = list.filter((value) => value !== null).map((value) => {
                        const identifier = value && value.name ? value.name : '';
                        const initializer = value ? new pytorch.Tensor(identifier, value) : null;
                        return new pytorch.Value(identifier, null, initializer);
                    });
                    const argument = new pytorch.Argument(name || key, visible, args);
                    inputs.push(argument);
                }
            }
            const group = groups.join('/');
            const name = group ? (group + '/' + key) : key;
            const outputs = output ? [ new pytorch.Argument('output', true, [ arg(name) ]) ] : [];
            const attributes = [];
            for (const entry of Object.entries(obj)) {
                const name = entry[0];
                const value = entry[1];
                if (!name.startsWith('_') && !parameters.has(name)) {
                    attributes.push({ name: name, value: value });
                }
            }
            const item = {
                name: name,
                type: type,
                attributes: attributes,
                children: obj._modules && obj._modules.size > 0 ? true : false,
                inputs: inputs,
                outputs: outputs
            };
            const node = new pytorch.Node(metadata, group, item, {}, arg);
            this._nodes.push(node);
            return [ node.name ];
        };
        const loadModule = (current, groups, inputs) => {
            if (!current._modules || current._modules.size == 0) {
                createNode(groups, '', current, inputs, false);
            } else {
                const sequential = current.__class__ && current.__class__.__module__ === 'torch.nn.modules.container' && current.__class__.__name__ === 'Sequential';
                for (const pair of current._modules) {
                    const key = pair[0];
                    const value = pair[1];
                    if (value) {
                        const type = value.__class__.__module__ + '.' + value.__class__.__name__;
                        switch (type) {
                            case 'torch.nn.modules.container.Sequential':
                                groups.push(key);
                                inputs = loadModule(value, groups, sequential ? inputs : []);
                                groups.pop(key);
                                break;
                            default: {
                                inputs = createNode(groups, key, value, sequential ? inputs : [], sequential);
                                break;
                            }
                        }
                    }
                }
            }
            return inputs;
        };
        const getSubmodules = (module) => {
            const submodules = [];
            if (module && module.__class__ && module.__class__.__module__ && module.__class__.__name__) {
                for (const entry of Object.entries(module)) {
                    const key = entry[0];
                    if (!key.startsWith('__')) {
                        const value = entry[1];
                        if (value && value.__class__ && value.__class__.__module__ && value.__class__.__name__ && !pytorch.Utility.isTensor(value)) {
                            submodules.push(value);
                        }
                    }
                }
            }
            return submodules;
        };
        const loadScriptModule = (module, initializers) => {
            if (module) {
                if (pytorch.Graph._getParameters(module).size > 0 && !module.__hide__) {
                    const item = { module: module };
                    this._nodes.push(new pytorch.Node(metadata, '', item, initializers, arg));
                }
                const submodules = getSubmodules(module);
                for (const submodule of submodules) {
                    loadScriptModule(submodule, initializers);
                }
            }
        };
        const type = module && module.__class__ && module.__class__.__module__ && module.__class__.__name__ ? module.__class__.__module__ + '.' + module.__class__.__name__ : null;
        if ((type === 'torch.ScriptModule' || type === 'torch.jit._script.ScriptModule' || type === 'torch.jit._script.RecursiveScriptModule') && module.graph) {
            const initializers = new Map();
            const graph = module.graph;
            const constants = module.code_with_constants[1].const_mapping;
            if (constants) {
                for (const entry of constants) {
                    const name = 'CONSTANTS.' + entry[0];
                    const value = entry[1];
                    if (pytorch.Utility.isTensor(value)) {
                        initializers.set(value, new pytorch.Tensor(name, value));
                    } else if (value && value.__class__ && value.__class__.__module__ && value.__class__.__name__) {
                        const type = value.__class__.__module__ + '.' + value.__class__.__name__;
                        switch (type) {
                            case '__torch__.torch.classes.xnnpack.LinearOpContext':
                            case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                            case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                            case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase': {
                                for (const entry of Object.entries(value)) {
                                    const key = entry[0];
                                    const value = entry[1];
                                    if (pytorch.Utility.isTensor(value)) {
                                        initializers.set(value, new pytorch.Tensor(name + '.' + key, value));
                                    }
                                }
                                break;
                            }
                            default: {
                                throw new pytorch.Error("Unsupported constant context '" + type + "'.");
                            }
                        }
                    } else {
                        throw new pytorch.Error('Unsupported constant.');
                    }
                }
            }
            const queue = [ module.data ];
            while (queue.length > 0) {
                const module = queue.shift();
                if (module.__class__ && module.__class__.__module__ === '__torch__.torch.classes._nnapi' && module.__class__.__name__ === 'Compilation') {
                    continue;
                }
                for (const entry of Object.entries(module)) {
                    const key = entry[0];
                    if (key !== '__module__' && key !== '__name__' && key !== '__class__' && key !== '__parent__') {
                        const obj = entry[1];
                        if (!Array.isArray(obj) && obj === Object(obj)) {
                            if (pytorch.Utility.isTensor(obj)) {
                                const parameter = obj;
                                parameter.__parent__ = module;
                                if (parameter.storage()) {
                                    if (parameter.__count__ === undefined || parameter.__count__ === 1) {
                                        initializers.set(parameter, new pytorch.Tensor(parameter.name, parameter));
                                    }
                                }
                            } else if (obj && obj.__class__) {
                                obj.__parent__ = module;
                                obj.__name__ = obj.__name__ || key;
                                queue.push(obj);
                            }
                        }
                    }
                }
            }
            for (const value of graph.inputs()) {
                const identifier = value.unique().toString();
                const name = value.debugName() || identifier;
                this._inputs.push(new pytorch.Argument(name, true, [ arg(identifier) ]));
            }
            for (const value of graph.outputs()) {
                const identifier = value.unique().toString();
                this._outputs.push(new pytorch.Argument(identifier, true, [ arg(identifier) ]));
            }
            for (const node of graph.nodes()) {
                if (node === graph.param_node() ||
                    node === graph.return_node()) {
                    continue;
                }
                if (node.kind() === 'prim::ListConstruct' &&
                    node.outputs().length === 1 &&
                    node.outputs().every((output) => output.uses().length === 1) &&
                    node.inputs().every((input) => pytorch.Utility.isTensor(input.value))) {
                    continue;
                }
                if (node.kind() === 'prim::ListUnpack' &&
                    node.inputs().length === 1 &&
                    node.inputs().every((input) => input.uses().length === 1) &&
                    node.outputs().every((output) => pytorch.Utility.isTensor(output.value))) {
                    continue;
                }
                const item = {
                    type: node.kind(),
                    node: node
                };
                this._nodes.push(new pytorch.Node(metadata, '', item, initializers, arg));
            }
            if (module) {
                loadScriptModule(module.data, initializers);
            }
        } else if (Array.isArray(module) && module.every((module) => module && module._modules !== undefined)) {
            for (const value of module) {
                loadModule(value, [], []);
            }
        } else {
            this._type = (module.__module__ && module.__name__) ? (module.__module__ + '.' + module.__name__) : '';
            loadModule(module, [], []);
        }
    }

    static _getParameters(module) {
        const parameters = new Map();
        if (module && module.__class__.__module__ && module.__class__.__name__) {
            for (const entry of Object.entries(module)) {
                const key = entry[0];
                const value = entry[1];
                if (pytorch.Utility.isTensor(value)) {
                    parameters.set(key, value);
                }
            }
        }
        return parameters;
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

pytorch.Argument = class {

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

pytorch.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new pytorch.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, group, item, initializers, arg) {
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
        } else {
            this._attributes = [];
            this._inputs = [];
            this._outputs = [];
            let module = item.module;
            if (module) {
                this._type = { name: 'torch.nn.modules.module.Module' };
                for (const entry of pytorch.Graph._getParameters(module)) {
                    const name = entry[0];
                    const tensor = entry[1];
                    const initializer = initializers.get(tensor) || (tensor ? new pytorch.Tensor('', tensor) : null);
                    const value = arg('', null, initializer || null);
                    this._inputs.push(new pytorch.Argument(name, true, [ value ]));
                    if (tensor.__variable__) {
                        this._outputs.push(new pytorch.Argument(name, true, [ arg(tensor.__variable__) ]));
                    }
                }
            }
            const node = item.node;
            if (node) {
                this._type = type(metadata, item.type);
                module = null;
                let match = true;
                let count = 0;
                for (const input of node.inputs()) {
                    const value = input.value;
                    const name = value && value.__class__ && value.__class__.__module__ && value.__class__.__name__ ? value.__class__.__module__ + '.' + value.__class__.__name__ : '';
                    let values = [];
                    switch (name) {
                        case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                        case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                        case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                        case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                        case '__torch__.torch.classes.xnnpack.LinearOpContext': {
                            values = Object.values(value);
                            break;
                        }
                        default: {
                            if (pytorch.Utility.isTensor(value)) {
                                values = [ value ];
                            }
                            if (input.node() &&
                                input.node().kind() === 'prim::ListConstruct' &&
                                input.uses().length === 1 &&
                                input.node().inputs().every((input) => pytorch.Utility.isTensor(input.value))) {
                                values = input.node().inputs().map((input) => input.value);
                            }
                            break;
                        }
                    }
                    for (const value of values) {
                        const parameter = initializers.get(value);
                        if (parameter) {
                            if (value.__parent__ && (module == null || module == value.__parent__)) {
                                module = value.__parent__;
                                count++;
                            } else if (value.__name__ && value.__name__.startsWith('CONSTANTS.c')) {
                                count++;
                            } else {
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
                    const parameters = pytorch.Graph._getParameters(module);
                    parameters.delete('num_batches_tracked');
                    if (parameters.size == count && match) {
                        module.__hide__ = true;
                    } else {
                        module = null;
                    }
                }
                const inputs = node.inputs();
                for (let i = 0; i < inputs.length; i++) {
                    const input = inputs[i];
                    const metadata = this._type && this._type.inputs && i < this._type.inputs.length ? this._type.inputs[i] : null;
                    const name = metadata && metadata.name ? metadata.name : i.toString();
                    const type = metadata && metadata.type ? metadata.type : null;
                    switch (type) {
                        case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                        case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                        case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                        case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                        case '__torch__.torch.classes.xnnpack.LinearOpContext': {
                            for (const entry of Object.entries(input.value)) {
                                const key = entry[0];
                                const value = entry[1];
                                if (key.startsWith('__') && key.endsWith('__')) {
                                    continue;
                                }
                                if (pytorch.Utility.isTensor(value)) {
                                    const initializer = initializers.get(value);
                                    const identifier = initializer ? initializer.name : input.unique().toString();
                                    const argument = new pytorch.Argument(key, true, [ arg(identifier, null, initializer) ]);
                                    this._inputs.push(argument);
                                } else {
                                    const attribute = new pytorch.Attribute(null, key, value);
                                    this._attributes.push(attribute);
                                }
                            }
                            break;
                        }
                        default: {
                            if (pytorch.Utility.isTensor(input.value) || input.value === undefined || input.value === null) {
                                let list = [ input ];
                                if (input.node() &&
                                    input.node().kind() === 'prim::ListConstruct' &&
                                    input.uses().length === 1 &&
                                    input.node().inputs().every((input) => pytorch.Utility.isTensor(input.value))) {
                                    list = input.node().inputs();
                                }
                                const args = list.map((input) => {
                                    let initializer = null;
                                    let identifier = input.unique().toString();
                                    if (input.value) {
                                        const value = input.value;
                                        const hide = value.__parent__ ? value.__parent__.__hide__ : true;
                                        initializer = hide ? initializers.get(value) : null;
                                        identifier = initializer ? initializer.name : identifier;
                                    }
                                    if (initializer) {
                                        return new pytorch.Value(identifier, null, initializer);
                                    }
                                    return arg(identifier);
                                });
                                const argument = new pytorch.Argument(name, true, args);
                                this._inputs.push(argument);
                            } else {
                                const attribute = new pytorch.Attribute(metadata, metadata.name, input.value);
                                this._attributes.push(attribute);
                            }
                            break;
                        }
                    }
                }

                const outputs = node.outputs();
                for (let i = 0; i < outputs.length; i++) {
                    const output = outputs[i];
                    const metadata = this._type && this._type.outputs && i < this._type.outputs.length ? this._type.outputs[i] : null;
                    const name = metadata && metadata.name ? metadata.name : i === 0 ? 'output' : 'output' + i.toString();
                    let list = [ output ];
                    if (output.uses().length === 1 &&
                        output.uses()[0].user &&
                        output.uses()[0].user.kind() == 'prim::ListUnpack' &&
                        output.uses()[0].user.outputs().every((output) => pytorch.Utility.isTensor(output.value))) {
                        list = output.uses()[0].user.outputs();
                    }
                    const args = list.map((output) => arg(output.unique().toString()));
                    const argument = new pytorch.Argument(name, true, args);
                    this._outputs.push(argument);
                }
            }
            if (module) {
                if (module.__name__) {
                    let current = module;
                    this._name = current.__name__;
                    while (current.__parent__ != null) {
                        current = current.__parent__;
                        if (!current.__parent__ && !current.__name__) {
                            break;
                        }
                        this._name = [ current.__name__, this._name ].join('.');
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
        } else if (metadata) {
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
        } else if (!layout || layout === 'torch.strided') {
            this._data = storage.data;
            this._layout = '<';
            this._indices = null;
        } else {
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

pytorch.Container = class {

    static open(context) {
        const zip = pytorch.Container.Zip.open(context);
        if (zip) {
            return zip;
        }
        const pickle = pytorch.Container.Pickle.open(context);
        if (pickle) {
            return pickle;
        }
        const tar = pytorch.Container.Tar.open(context);
        if (tar) {
            return tar;
        }
        const data = pytorch.Container.data_pkl.open(context);
        if (data) {
            return data;
        }
        const torch_utils = pytorch.Container.torch_utils.open(context);
        if (torch_utils) {
            return torch_utils;
        }
        const mobile = pytorch.Container.Mobile.open(context);
        if (mobile) {
            return mobile;
        }
        return null;
    }

    constructor() {
        this._events = [];
    }

    async read() {
    }

    on(event, callback) {
        this._events.push([ event, callback ]);
    }

    get format() {
        throw new pytorch.Error('Container format not implemented.');
    }

    get modules() {
        throw new pytorch.Error('Container modules not implemented.');
    }
};

pytorch.Container.Tar = class extends pytorch.Container {

    static open(context) {
        const entries = context.entries('tar');
        if (entries.has('pickle')) {
            return new pytorch.Container.Tar(entries);
        }
        return null;
    }

    constructor(entries) {
        super();
        this._entries = entries;
    }

    async read() {
        const entries = this._entries;
        delete this._entries;
        const execution = new pytorch.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const obj = torch.load(entries);
        this._modules = pytorch.Utility.findWeights(obj);
        if (!this._modules) {
            throw new pytorch.Error('File does not contain root module or state dictionary.');
        }
    }

    get format() {
        return 'PyTorch v0.1.1';
    }

    get modules() {
        return this._modules;
    }
};

pytorch.Container.Pickle = class extends pytorch.Container {

    static open(context) {
        const stream = context.stream;
        const signature = [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            return new pytorch.Container.Pickle(stream);
        }
        return null;
    }

    constructor(stream) {
        super();
        this._stream = stream;
    }

    async read() {
        const data = this._stream.length < 0x7ffff000 ? this._stream.peek() : this._stream;
        delete this._stream;
        const execution = new pytorch.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const obj = torch.load(data);
        this._modules = pytorch.Utility.find(obj);
    }

    get format() {
        return 'PyTorch v0.1.10';
    }

    get modules() {
        return this._modules;
    }
};

pytorch.Container.data_pkl = class extends pytorch.Container {

    static open(context) {
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
            const name = obj.__class__.__module__ + '.' + obj.__class__.__name__;
            if (name.startsWith('__torch__.')) {
                return new pytorch.Container.data_pkl(obj);
            }
        }
        return null;
    }

    constructor(data) {
        super();
        this._data = data;
    }

    get format() {
        return 'PyTorch Pickle';
    }

    get modules() {
        throw new pytorch.Error("PyTorch data.pkl format not supported.");
    }
};

pytorch.Container.torch_utils = class extends pytorch.Container {

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
        super();
        this._obj = obj;
    }

    async read() {
        this._modules = pytorch.Utility.find(this._obj);
        delete this._obj;
    }

    get format() {
        return 'PyTorch torch_utils';
    }

    get modules() {
        return this._modules;
    }
};

pytorch.Container.Mobile = class extends pytorch.Container {

    static open(context) {
        const tags = context.tags('flatbuffers');
        if (tags.get('file_identifier') === 'PTMF') {
            return new pytorch.Container.Mobile(context);
        }
        return null;
    }

    constructor(context) {
        super();
        this._context = context;
    }

    async read(metadata) {
        await this._context.require('./pytorch-schema');
        this._modules = new Map();
        const execution = new pytorch.jit.Execution(null, metadata);
        for (const event in this._events) {
            execution.on(event[0], event[1]);
        }
        const stream = this._context.stream;
        const torch = execution.__import__('torch');
        const module = torch.jit.jit_module_from_flatbuffer(stream);
        this._version = pytorch.Utility.version(module._c._bytecode_version);
        if (module && module.forward) {
            this._modules = new Map([ ['', module] ]);
        } else {
            this._modules = pytorch.Utility.find(module);
        }
        delete this._context;
    }

    get format() {
        return 'PyTorch Mobile' + (this._version ? ' ' + this._version : '');
    }

    get modules() {
        return this._modules;
    }
};

pytorch.Container.Zip = class extends pytorch.Container {

    static open(context) {
        const entries = context.entries('zip');
        if (entries.size > 0) {
            const reader = new pytorch.jit.StreamReader(entries);
            if (reader.hasRecord('model.json')) {
                try {
                    const stream = reader.getRecord('model.json');
                    const buffer = stream.peek();
                    const decoder = new TextDecoder('utf-8');
                    const content = decoder.decode(buffer);
                    const model = JSON.parse(content);
                    if (model.mainModule) {
                        return new pytorch.Container.Zip(reader, model);
                    }
                } catch (error) {
                    // continue regardless of error
                }
            }
            if (reader.hasRecord('data.pkl')) {
                return new pytorch.Container.Zip(reader);
            }
            if (reader.hasRecord('.data/version')) {
                return new pytorch.Container.Package(reader);
            }
            const tags = context.tags('flatbuffers');
            if (tags.get('file_identifier') === 'PTMF') {
                return new pytorch.Container.Mobile(context);
            }
        }
        return null;
    }

    constructor(reader, model) {
        super();
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
        this._reader = reader;
        this._torchscript = model || this._reader.hasRecord('constants.pkl');
        if (model) {
            this._producer = model && model.producerName ? model.producerName + (model.producerVersion ? ' v' + model.producerVersion : '') : '';
            this._format = this._reader.hasRecord('attributes.pkl') ? 'TorchScript v1.1' : 'TorchScript v1.0';
        } else {
            const name = this._torchscript ? 'TorchScript' : 'PyTorch';
            const version = pytorch.Utility.version(reader.version());
            this._format = name + ' ' + version;
        }
    }

    async read(metadata) {
        const execution = new pytorch.jit.Execution(null, metadata);
        for (const event in this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        if (this._torchscript) {
            const module = torch.jit.load(this._reader);
            if (module.data && module.data.forward) {
                this._modules = new Map([ [ '', module ] ]);
            } else {
                this._modules = pytorch.Utility.find(module.data);
            }
        } else {
            const entries = new Map(this._reader.getAllRecords().map((key) => [ key, this._reader.getRecord(key) ]));
            const module = torch.load(entries);
            this._modules = pytorch.Utility.find(module);
        }
        delete this._reader;
    }

    get format() {
        return this._format;
    }

    get modules() {
        return this._modules;
    }

    get producer() {
        return this._producer || '';
    }
};

pytorch.Execution = class extends python.Execution {

    constructor(sources) {
        super(sources);
        const execution = this;
        const torch = this.register('torch');
        const pickle = this.register('pickle');
        this.register('torch.jit._script');
        this.register('torch.jit._trace');
        this.registerType('torch.package.PackageImporter', class {
            constructor(reader) {
                this.zip_reader = reader;
            }
            load_pickle(name) {
                const stream = this.zip_reader.getRecord(name);
                const loaded_reduces = new Map();
                this.storage_context = new torch._C.DeserializationStorageContext();
                const unpickler = new pickle.Unpickler(stream);
                unpickler.persistent_load = (saved_id) => {
                    const typename = saved_id.shift();
                    switch (typename) {
                        case 'storage': {
                            const storage_type = saved_id[0];
                            const key = saved_id[1];
                            /* const location = saved_id[2]; */
                            const size = saved_id[3];
                            if (!this.storage_context.has_storage(key)) {
                                const storage = new storage_type(size);
                                const stream = this.zip_reader.getRecord('.data/' + key + '.storage');
                                const buffer = stream.peek();
                                storage._set_cdata(buffer);
                                this.storage_context.add_storage(key, storage);
                            }
                            return this.storage_context.get_storage(key);
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
                const obj = unpickler.load();
                this.storage_context = null;
                return obj;
            }
        });
        this.registerFunction('torch.jit.load', function(file, map_location, extra_files) {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const cpp_module = torch._C.import_ir_module(cu, file, map_location, extra_files);
            return new torch.jit._script.RecursiveScriptModule(cpp_module);
        });
        this.registerFunction('torch._C.import_ir_module', function(cu, reader) {
            switch (arguments.length) {
                case 4: {
                    const reader = arguments[1];
                    const device = arguments[2];
                    const extra_files = arguments[3];
                    const deserializer = new pytorch.jit.ScriptModuleDeserializer(cu, reader);
                    return deserializer.deserialize(device, extra_files);
                }
                case 5: {
                    const storage_context = arguments[2];
                    const device = arguments[3];
                    const ts_id = arguments[4];
                    const deserializer = new pytorch.jit.ScriptModuleDeserializer(cu, reader, '.data/ts_code/' + ts_id + '/', '.data/', storage_context);
                    return deserializer.deserialize(device, null);
                }
                default: {
                    throw new pytorch.Error("Invalid 'torch._C.import_ir_module' signature.");
                }
            }

        });
        this.registerFunction('torch._C._import_ir_module_from_package', function(cu, reader, storage_context, map_location, ts_id) {
            return torch._C.import_ir_module(cu, reader, storage_context, null, ts_id);
        });
        this.registerFunction('torch._C._jit_pass_inline', function(graph) {
            const tryToGraphFunction = (node) => {
                if (node.kind() === 'prim::CallFunction') {
                    // TODO
                }
                if (node.kind() === 'prim::CallMethod') {
                    const name = null; // node.s(attr::name);
                    const class_type = node.input(0).type();
                    if (class_type) {
                        const fn = class_type.getMethod(name);
                        return tryToGraphFunction(fn);
                    }
                }
                return null;
            };
            const inlineCallTo = (/* to_replace, callee, use_graph */) => {
                // TODO
            };
            const inlineCalls = (block) => {
                for (const cur of block.nodes()) {
                    switch (cur.kind()) {
                        case 'prim::CallFunction': {
                            throw new pytorch.Error();
                        }
                        case 'prim::CallMethod': {
                            const graphFunction = tryToGraphFunction(cur);
                            inlineCallTo(cur, graphFunction, true);
                            break;
                        }
                        default: {
                            for (const b of block.nodes()) {
                                inlineCalls(b);
                            }
                        }
                    }
                }
            };
            inlineCalls(graph.blocks());
        });
        this.registerFunction('torch.jit._script.unpackage_script_module', function(importer, script_module_id) {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const cpp_module = torch._C._import_ir_module_from_package(cu, importer.zip_reader, importer.storage_context, importer.last_map_location, script_module_id);
            return new torch.jit._script.RecursiveScriptModule(cpp_module);
        });
        this.registerFunction('torch.jit.jit_module_from_flatbuffer', function(f) {
            pytorch.mobile = flatbuffers.get('torch').torch.jit.mobile;
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const stream = f;
            const reader = flatbuffers.BinaryReader.open(stream);
            const module = pytorch.mobile.serialization.Module.create(reader);
            const loader = new pytorch.jit.FlatBuffersLoader(cu);
            const cpp_module = loader.parseModule(module);
            // parse_and_initialize_jit_module
            //   const mobilem = parse_and_initialize_mobile_module_for_jit(data, jit_files, jit_constants);
            //   const m = jitModuleFromSourceAndConstants(mobilem._ivalue(), jit_files, jit_constants, mobilem.bytecode_version());
            // throw new pytorch.Error('torch.jit.mobile.serialization.Module not supported.');
            return torch.jit._script.wrap_cpp_module(cpp_module);
        });
        this.registerFunction('torch.jit._script.wrap_cpp_module', function(cpp_module) {
            const init_fn = (script_module) => {
                for (const entry of new torch.ModuleDict(script_module._c).items()) {
                    script_module.__setattr__(entry[0], torch.jit._script.wrap_cpp_module(entry[1]));
                }
            };
            return torch.jit._script.RecursiveScriptModule._construct(cpp_module, init_fn);
        });
        this.registerType('torch._C.DeserializationStorageContext', class extends Map {
            has_storage(name) {
                return this.has(name);
            }
            get_storage(name) {
                return this.get(name);
            }
            add_storage(name, storage) {
                return this.set(name, storage);
            }
        });
        this.registerType('torch.Type', class {});
        this.registerType('torch.ClassType', class extends torch.Type {
            constructor(qualified_name, cu, is_module) {
                super();
                this._qualified_name = qualified_name;
                this._is_module = is_module;
            }
            qualified_name() {
                return this._qualified_name;
            }
            name() {
                return this._qualified_name.split('.').pop();
            }
            is_module() {
                return this._is_module;
            }
            addMethod(/* name, fn */) {
                // TODO
            }
            addAttribute(/* name */) {
                // TODO
            }
            hasAttribute(/* name */) {
                // TODO
            }
            hasConstant(/* name */) {
                // TODO
            }
            methods() {
                // TODO
            }
        });
        this.registerType('torch.TupleType', class extends torch.Type {
            constructor(/* elements, name, schema */) {
                super();
                // TODO
            }
        });
        this.registerType('torch.ScriptFunction', class {
            constructor(name, graph /*, function_creator */) {
                this._name = name;
                this._graph = graph;
            }
        });
        this.registerType('torch.ScriptMethod', class {
            constructor(owner, value) {
                this._owner = owner;
                this._function = value;
            }
            get name() {
                return this._function.name();
            }
            get owner() {
                return this._owner;
            }
            __call__(/* args, kwargs */) {
                throw new pytorch.Error();
            }
            get graph() {
                return this._function.graph();
            }
            get schema() {
                // return this.function().getSchema();
                throw new pytorch.Error();
            }
            get code() {
                throw new pytorch.Error();
            }
            get code_with_constants() {
                throw new pytorch.Error();
            }
        });
        this.registerType('torch.ScriptObject', class {
            constructor(type) {
                this._type = type;
            }
            static create(type) {
                if (type.is_module()) {
                    return new torch.ScriptModule(type);
                }
                return new torch.ScriptObject(type);
            }
            _type() {
                return this._type;
            }
            _get_method(name) {
                for (const method of this._type.methods()) {
                    if (name == method.name) {
                        return method;
                    }
                }
                return null;
            }
            _has_method(/* name */) {
                throw new pytorch.Error();
            }
            __setattr__(name, value) {
                // TODO if (this._type.hasContant(name))
                this[name] = value;
            }
            __getattr__(name) {
                return this[name];
            }
            hasattr(name) {
                return this._type.hasAttribute(name) || this._type.hasConstant(name);
            }
            _properties() {
                throw new pytorch.Error();
            }
        });
        this.registerType('torch.ScriptModule', class extends torch.ScriptObject {
            constructor(type) {
                super(type);
            }
            get qualified_name() {
                return this._type.qualified_name();
            }
            get code_with_constants() {
                const const_map = {};
                const_map.const_mapping = new Map(Object.entries(execution.builtins.CONSTANTS));
                return [ null, const_map ];
            }
            get graph() {
                if (!this._graph) {
                    if (!this.data) {
                        return null;
                    }
                    if (!this.data.forward) {
                        throw new pytorch.Error("Module 'forward' not implemented.");
                    }
                    const args = [ this.data ]; // self
                    if (this.data.forward.__code__ && this.data.forward.__code__.parameters) {
                        for (const parameter of this.data.forward.__code__.parameters) {
                            const defaultValue = (type, name) => {
                                if (type.type === 'type' && type.name.type) {
                                    switch (type.name.value) {
                                        case 'Tensor': {
                                            const tensor = execution.invoke('torch.Tensor', []);
                                            tensor.__variable__ = name;
                                            tensor.__origin__ = 'graph-input';
                                            const value = execution.variable(tensor, execution.graph.param_node());
                                            if (value && name) {
                                                value.setDebugName(name);
                                            }
                                            return tensor;
                                        }
                                        case 'Tuple': {
                                            return type.arguments.map((type, index) => defaultValue(type, name + '[' + index.toString() + ']'));
                                        }
                                        case 'List': {
                                            return type.arguments.map((type, index) => defaultValue(type, name + '[' + index.toString() + ']'));
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
                                }
                                args.push(value);
                            }
                        }
                    }
                    const result = this.data.forward.__call__(args);
                    if (Array.isArray(result)) {
                        for (const output of result) {
                            if (pytorch.Utility.isTensor(output)) {
                                const value = execution.variable(output);
                                execution.graph.return_node().addInput(value);
                            }
                        }
                    } else if (pytorch.Utility.isTensor(result)) {
                        const value = execution.variable(result);
                        execution.graph.return_node().addInput(value);
                    } else if (Object(result) === result) {
                        for (const key of Object.keys(result)) {
                            const item = result[key];
                            if (Array.isArray(item)) {
                                for (const output of item) {
                                    if (pytorch.Utility.isTensor(output)) {
                                        const value = execution.variable(output);
                                        execution.graph.return_node().addInput(value);
                                    }
                                }
                            } else if (pytorch.Utility.isTensor(item)) {
                                const value = execution.variable(item);
                                execution.graph.return_node().addInput(value);
                            }
                        }
                    }
                    this._graph = execution.graph;
                }
                return this._graph;
            }
        });
        this.registerType('torch.ModuleDict', class {
            constructor(module) {
                this._items = Object.entries(module).filter((entry) => entry[1] instanceof torch.ScriptModule);
            }
            items() {
                return this._items;
            }
        });
        this.registerType('torch.jit.CompilationUnit', class {
            constructor() {
                this._functions = new Map();
                this._classes = new Map();
            }
            register_function(fn) {
                this._functions.set(fn.name, fn);
            }
            define(prefix, properties, propResolvers, definitions /*, defResolvers, self, shouldMangle, operator_set_version */) {
                for (const def of definitions) {
                    const name = def.name;
                    const qualified_name = prefix ? prefix + '.' + name : name;
                    const graph = new torch.Graph();
                    const fn = new torch.ScriptFunction(qualified_name, graph, null);
                    this.register_function(fn);
                }
            }
            get_class(name) {
                return this._classes.get(name);
            }
            register_type(name, cls) {
                this._classes.set(name, cls);
            }
        });
        this.registerType('torch.jit._script.ScriptModule', class extends torch.nn.modules.module.Module {
            constructor(/* obj */) {
                super();
                // TODO
            }
        });
        this.registerType('torch.jit._trace.TracedModule', class extends torch.jit._script.ScriptModule {});
        this.registerType('torch.jit._trace.TopLevelTracedModule', class extends torch.jit._trace.TracedModule {});
        this.registerType('torch.jit._script.RecursiveScriptModule', class extends torch.jit._script.ScriptModule {
            constructor(cpp_module) {
                super();
                this._initializing = true;
                this._c = cpp_module;
            }
            static _construct(cpp_module, init_fn) {
                const script_module = new torch.jit._script.RecursiveScriptModule(cpp_module);
                init_fn(script_module);
                torch.jit._script.RecursiveScriptModule._finalize_scriptmodule(script_module);
                return script_module;
            }
            static _finalize_scriptmodule() {
                this._initializing = false;
            }
            get data() {
                return this._c.data;
            }
            get graph() {
                // return this._c._get_method("forward").graph;
                return this._c.graph;
            }
            get code_with_constants() {
                // return this.forward.code_with_constants;
                return this._c.code_with_constants;
            }
            __setattr__(name, value) {
                if (this._initializing) {
                    super.__setattr__(name, value);
                } else if (this._modules.has(name)) {
                    this._modules.set(name, value);
                } else if (this._c.hasattr(name)) {
                    this._c.setattr(name, value);
                } else {
                    // TODO
                }
            }
            __getattr__(name) {
                if (this._initializing) {
                    return super.__getattr__(name);
                }
                if (this._modules.has(name)) {
                    return this._modules.get(name);
                }
                if (this._c.hasattr(name)) {
                    return this._c.getattr(name);
                }
                if (this._c._has_method(name)) {
                    // TODO
                }
                return super.__getattr__(name);
            }
        });
        torch.jit.ScriptModule = torch.jit._script.ScriptModule;
        torch.jit.RecursiveScriptModule = torch.jit._script.RecursiveScriptModule;
        torch.jit.TopLevelTracedModule = torch.jit._trace.TopLevelTracedModule;
        torch.CompilationUnit = torch.jit.CompilationUnit;
        torch._C.CompilationUnit = torch.jit.CompilationUnit;
        torch._C.ScriptModule = torch.ScriptModule;
        torch._C.ClassType = torch.ClassType;
    }
};

pytorch.jit = {};

pytorch.jit.Execution = class extends pytorch.Execution {

    constructor(sources, metadata) {
        super(sources);
        this._metadata = metadata;
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
                const buffers = parameter_buffers.map((buffer) => buffer.__source__.storage());
                const serialized_model = serialized_model_tensor.storage().data;
                this.serialized_model = new pytorch.nnapi.SerializedModel(serialized_model, buffers);
            }
            run(inputs, outputs) {
                execution.variable(this.serialized_model_tensor);
                this.serialized_model_tensor.__count__ = (this.serialized_model_tensor.__count__ || 0) + 1;
                const type = new pytorch.nnapi.Graph(this.serialized_model);
                const node = execution._graph.create(type);
                for (const tensor of inputs) {
                    const value = execution.variable(tensor);
                    node.addInput(value);
                }
                for (const tensor of outputs) {
                    execution.variable(tensor, node);
                }
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
        this.registerType('torch.Graph', class {
            constructor() {
                this._unique = 1;
                this._nodes = [];
                this._block = execution.invoke('torch.Block', [ this ]);
            }
            create(kind) {
                return execution.invoke('torch.Node', [ this, kind ]);
            }
            inputs() {
                return this._block.inputs();
            }
            outputs() {
                return this._block.outputs();
            }
            nodes() {
                return this._nodes;
                // return this._block.nodes();
            }
            param_node() {
                return this._block.param_node();
            }
            return_node() {
                return this._block.return_node();
            }
        });
        this.registerType('torch.Block', class {
            constructor(graph) {
                this._unique = 1;
                this._graph = graph;
                this._input = graph.create('prim::Param');
                this._output = graph.create('prim::Return');
            }
            param_node() {
                return this._input;
            }
            return_node() {
                return this._output;
            }
            inputs() {
                return this._input.outputs();
            }
            outputs() {
                return this._output.inputs();
            }
            addInput(name) {
                const value = this._input.addOutput();
                value.setDebugName(name || '');
                return value;
            }
            registerOutput(value) {
                this._output.addInput(value);
                return this.outputs().length - 1;
            }
        });
        this.registerType('torch.Node', class {
            constructor(graph, kind) {
                this._graph = graph;
                this._graph._nodes.push(this);
                this._kind = kind;
                this._inputs = [];
                this._outputs = [];
                this._blocks = [];
            }
            kind() {
                return this._kind;
            }
            inputs() {
                return this._inputs;
            }
            outputs() {
                return this._outputs;
            }
            blocks() {
                return this._blocks;
            }
            addInput(value) {
                const use = execution.invoke('torch.Use', [ this ]);
                value.uses().push(use);
                this._inputs.push(value);
                return value;
            }
            addOutput() {
                const value = execution.invoke('torch.Value', [ this ]);
                this._outputs.push(value);
                return value;
            }
            addBlock() {
                const block = execution.invoke('torch.Block', [ this._graph, this ]);
                this._blocks.push(block);
                return block;
            }
        });
        this.registerType('torch.Value', class {
            constructor(node) {
                this._unique = node && node._unique ? node._unique++ : node._graph._unique++;
                this._node = node && node._unique ? null : node;
                this._uses = [];
            }
            unique() {
                return this._unique;
            }
            node() {
                return this._node;
            }
            uses() {
                return this._uses;
            }
            setDebugName(name) {
                this._unique_name = name;
            }
            debugName() {
                return this._unique_name;
            }
        });
        this.registerType('torch.Use', class {
            constructor(node) {
                this._node = node;
            }
            get user() {
                return this._node;
            }
        });
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
        this._graph = this.invoke('torch.Graph', []);
        this._values = new Map();
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

    get graph() {
        return this._graph;
    }

    variable(tensor, node) {
        if (this._values.has(tensor)) {
            return this._values.get(tensor);
        }
        const value = node ? node.addOutput() : this.invoke('torch.Value', [ node ? node : this._graph ]);
        value.value = tensor;
        if (typeof tensor !== 'string' && typeof tensor !== 'number') {
            this._values.set(tensor, value);
        }
        if (pytorch.Utility.isTensor(tensor)) {
            tensor.__variable__ = value.unique().toString();
        }
        return value;
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
            } else if (current.type === 'id' && current.value !== 'self' && current.value !== 'CONSTANTS') {
                path.push(current.value);
                break;
            } else {
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
        const overload = this._overload(target, name, args, context);
        if (overload) {
            const schema = overload[0];
            const args = overload[1];
            const evalArgs = overload[2];
            const copyArgs = Array.prototype.slice.call(args);
            const copyEvalArgs = Array.prototype.slice.call(evalArgs);
            const node = this._graph.create(schema.name);
            node.schema = schema;
            const referencedParameters = [];
            const parameters = Array.prototype.slice.call(schema.inputs || []).concat(Array.prototype.slice.call(schema.attributes || []));
            while (copyEvalArgs.length > 0) {
                if (parameters.length <= 0) {
                    if (schema.name.startsWith('_caffe2::')) {
                        break;
                    }
                    throw new pytorch.Error();
                }
                if (copyArgs.every((arg) => arg.type === '=' && arg.target && arg.target.type === 'id') &&
                    parameters.every((parameter) => parameter.type !== 'Tensor' && parameter.type !== 'Tensor[]')) {
                    const map = new Map(parameters.map((parameter) => [ parameter.name, parameter ]));
                    while (copyArgs.length > 0) {
                        const argument = copyArgs.shift();
                        const arg = copyEvalArgs.shift();
                        const parameter = map.get(argument.target.value);
                        if (!parameter) {
                            throw new pytorch.Error();
                        }
                        if (!pytorch.Utility.isType(arg, parameter.type)) {
                            if (parameter.optional) {
                                continue;
                            }
                            throw new pytorch.Error();
                        }
                        const value = this.variable(arg);
                        value.value = arg;
                        node.addInput(value);
                    }
                    continue;
                }
                const parameter = parameters.shift();
                const argument = copyEvalArgs[0];
                if (parameter.type === 'Tensor' || (parameter.type === 'Scalar' && pytorch.Utility.isTensor(argument))) {
                    if (Array.isArray(argument) || (!pytorch.Utility.isTensor(argument) && argument !== null && argument !== undefined)) {
                        if (parameter.optional) {
                            continue;
                        }
                        throw new pytorch.Error();
                    } else {
                        copyArgs.shift();
                        copyEvalArgs.shift();
                        const tensor = (argument === null || argument === undefined) ? {} : argument;
                        const value = this.variable(tensor);
                        referencedParameters.push(tensor);
                        node.addInput(value);
                    }
                } else if (parameter.type === 'Tensor[]') {
                    const argument = copyEvalArgs[0];
                    if (!Array.isArray(argument) || !argument.every((item) => pytorch.Utility.isTensor(item) || item === null)) {
                        if (parameter.optional) {
                            continue;
                        }
                        throw new pytorch.Error();
                    } else {
                        copyArgs.shift();
                        copyEvalArgs.shift();

                        const list = this._graph.create('prim::ListConstruct');
                        for (const arg of argument) {
                            const tensor = arg;
                            if (tensor) {
                                tensor.__count__ = (tensor.__count__ || 0) + 1;
                            }
                            const value = this.variable(tensor);
                            list.addInput(value);
                        }

                        const value = list.addOutput();
                        node.addInput(value);
                    }
                } else {
                    const arg = copyArgs[0];
                    if (!pytorch.Utility.isType(argument, parameter.type) && argument !== null) {
                        if (parameter.optional) {
                            continue;
                        }
                        throw new pytorch.Error();
                    } else if (arg.type !== '=') {
                        copyArgs.shift();
                        copyEvalArgs.shift();
                        switch (parameter.type) {
                            case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                            case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                            case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                            case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                            case '__torch__.torch.classes.xnnpack.LinearOpContext': {
                                const value = this.variable(argument);
                                value.value = argument;
                                node.addInput(value);
                                for (const entry of Object.entries(argument)) {
                                    if (pytorch.Utility.isTensor(entry[1])) {
                                        const tensor = entry[1];
                                        referencedParameters.push(tensor);
                                    }
                                }
                                break;
                            }
                            default: {
                                const value = this.variable(argument);
                                node.addInput(value);
                                value.value = argument;
                                break;
                            }
                        }
                    } else {
                        throw new pytorch.Error('Expected named argument.');
                    }
                }
            }
            const result = [];
            for (let i = 0; i < schema.outputs.length; i++) {
                const parameter = schema.outputs[i];
                switch (parameter.type) {
                    case 'Scalar':
                    case 'Tensor': {
                        const output = this.invoke('torch.Tensor', []);
                        output.__origin__ = schema.name;
                        if (i === 0) {
                            switch (schema.name) {
                                case 'aten::conv1d':
                                case 'aten::embedding': {
                                    output.resize_([ NaN, NaN, NaN ]);
                                    break;
                                }
                                case 'aten::cat':
                                case 'aten::conv2d':
                                case 'aten::dropout':
                                case 'aten::flatten':
                                case 'aten::flatten.named_out_dim':
                                case 'aten::max_pool2d':
                                case 'aten::adaptive_avg_pool2d':
                                case 'aten::avg_pool2d':
                                case 'aten::quantize_per_tensor':
                                case 'aten::relu_':
                                case 'aten::prelu':
                                case 'aten::hardtanh_':
                                case 'aten::upsample_bilinear2d':
                                case 'prepacked::conv2d_clamp_run': {
                                    const input = evalArgs[0];
                                    if (pytorch.Utility.isTensor(input) && input.size() === undefined) {
                                        input.resize_([ NaN, NaN, NaN, NaN ]);
                                    }
                                    output.resize_([ NaN, NaN, NaN, NaN ]);
                                    break;
                                }
                                case 'aten::slice':
                                case 'aten::slice.Tensor': {
                                    const input = evalArgs[0];
                                    if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                        const size = input.size();
                                        output.resize_(size);
                                    }
                                    break;
                                }
                                case 'aten::to':
                                case 'aten::to.device':
                                case 'aten::to.dtype':
                                case 'aten::to.dtype_layout': {
                                    const input = evalArgs[0];
                                    if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                        const size = input.size();
                                        output.resize_(size);
                                    }
                                    break;
                                }
                                case 'aten::conv3d': {
                                    output.resize_([ NaN, NaN, NaN, NaN, NaN ]);
                                    break;
                                }
                                case 'aten::detach':
                                case 'aten::mean':
                                case 'aten::mul':
                                case 'aten::mul.Scalar':
                                case 'aten::div':
                                case 'aten::div.Scalar':
                                case 'aten::batch_norm':
                                case 'aten::gelu':
                                case 'aten::relu':
                                case 'aten::clamp':
                                case 'aten::clamp_':
                                case 'aten::_add_relu_':
                                case 'aten::hardswish_': {
                                    const input = evalArgs[0];
                                    if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                        output.resize_(input.size());
                                    }
                                    break;
                                }
                                case 'aten::add':
                                case 'aten::add.Scalar':
                                case 'aten::sub':
                                case 'aten::sub.Scalar': {
                                    const input = evalArgs[0];
                                    if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                        output.resize_(input.size());
                                    } else {
                                        const other = evalArgs[1];
                                        if (pytorch.Utility.isTensor(other) && Array.isArray(other.size())) {
                                            output.resize_(other.size());
                                        }
                                    }
                                    break;
                                }
                                case 'aten::select':
                                case 'aten::select.int': {
                                    const input = evalArgs[0];
                                    if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                        output.resize_(Array(input.size().length - 1).fill(NaN));
                                    }
                                    break;
                                }
                                case 'aten::layer_norm': {
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
                                case 'aten::empty':
                                case 'aten::ones':
                                case 'aten::zeros':
                                case 'aten::zeros_like': {
                                    output.resize_(evalArgs[0]);
                                    break;
                                }
                                case 'aten::view':
                                case 'aten::reshape':
                                case 'aten::new_full': {
                                    output.resize_(evalArgs[1]);
                                    break;
                                }
                                case 'aten::squeeze':
                                case 'aten::squeeze.dim': {
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
                                case 'aten::unsqueeze': {
                                    const input = evalArgs[0];
                                    const size = input.size();
                                    const dim = evalArgs[1];
                                    if (Array.isArray(size) && dim !== undefined) {
                                        const shape = size.slice();
                                        shape.splice(dim, 0, 1);
                                        output.resize_(shape);
                                    } else {
                                        output.resize_([ NaN, NaN, NaN, NaN ]);
                                    }
                                    break;
                                }
                                case 'aten::transpose':
                                case 'aten::transpose.int': {
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
                                case 'aten::contiguous':
                                    output.__source__ = evalArgs[0];
                                    break;
                                case 'quantized::cat':
                                case 'quantized::cat_relu':
                                case 'quantized::linear':
                                case 'quantized::conv2d':
                                case 'quantized::conv2d.new':
                                case 'quantized::conv2d_relu':
                                case 'quantized::conv2d_relu.new':
                                case 'quantized::add':
                                case 'quantized::add_relu':
                                    output.resize_([ NaN, NaN, NaN, NaN ]);
                                    output.__quantized__ = true;
                                    break;
                                default:
                                    break;
                            }
                        }
                        this.variable(output, node);
                        result.push(output);
                        break;
                    }
                    case 'Tensor[]': {
                        let count = 1;
                        switch (schema.name) {
                            case 'aten::chunk':
                                count = node.inputs()[1].value;
                                break;
                            case 'aten::meshgrid': {
                                const list = node.inputs()[0].node();
                                if (list.kind() === 'prim::ListConstruct') {
                                    count = list.inputs().length;
                                }
                                break;
                            }
                            case 'aten::unbind':
                            case 'aten::unbind.int':
                                count = args[0].__tuple__ || count;
                                break;
                            case 'aten::broadcast_tensors':
                            case 'aten::split':
                            case 'aten::split.Tensor':
                            case 'aten::split_with_sizes':
                                if (context.target.length > 0) {
                                    count = context.target[context.target.length - 1].length;
                                }
                                break;
                            default:
                                break;
                        }

                        const value = node.addOutput();
                        const list = this._graph.create('prim::ListUnpack');
                        list.addInput(value);

                        const tensors = [];
                        for (let i = 0; i < count; i ++) {
                            const tensor = this.invoke('torch.Tensor', []);
                            tensor.__origin__ = schema.name;
                            this.variable(tensor, list);
                            tensors.push(tensor);
                        }
                        result.push(tensors);
                        break;
                    }
                    case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                    case '__torch__.torch.classes.xnnpack.LinearOpContext': {
                        const value = this.invoke(parameter.type, []);
                        this.variable(value, node);
                        result.push(value);
                        break;
                    }
                    default: {
                        const output = this.invoke('torch.Tensor', []);
                        output.resize_([]);
                        output.__origin__ = schema.name;
                        this.variable(output, node);
                        result.push(output);
                        break;
                    }
                }
            }
            for (const referencedParameter of referencedParameters) {
                referencedParameter.__count__ = (referencedParameter.__count__ || 0) + 1;
            }
            if (result.length > 1) {
                return result;
            }
            return result[0];
        }
        return super.call(target, name, args, context);
    }

    _overload(target, name, args, context) {
        let moduleName = pytorch.Utility.target(target);
        if (moduleName && name) {
            let outputTypes = null;
            let type = moduleName + '.' + name;
            if (type === 'ops.prim.NumToTensor' && args.length === 1 && args[0].type === 'call' && args[0].target.member.type == 'id') {
                const arg = args[0];
                moduleName = pytorch.Utility.target(arg.target.target);
                name = arg.target.member.value;
                args = arg.args;
                outputTypes = [ 'int64' ];
                type = moduleName + '.' + name;
            }
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
            let overloads = null;
            if (type.startsWith('torch.')) {
                overloads = this._types.get('aten::' + type.substring(6));
            } else if (type.startsWith('ops.') && !type.startsWith('ops.prim.')) {
                const path = type.split('.');
                if (path.length === 3) {
                    overloads = this._types.get(path[1] + '::' + path[2]);
                }
                if (!overloads) {
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
                        overloads = [ metadata ];
                    }
                }
            }
            if (overloads) {
                overloads = !Array.isArray(overloads) ? [ overloads ] : overloads;
                const evalArgs = args.map((argument) => {
                    if (argument.type === '=' && argument.target && argument.target.type === 'id') {
                        argument = argument.expression;
                    }
                    return this.expression(argument, context);
                });
                for (const schema of overloads) {
                    const copyArgs = Array.prototype.slice.call(args);
                    const copyEvalArgs = Array.prototype.slice.call(evalArgs);
                    const parameters = Array.prototype.slice.call(schema.inputs || []).concat(Array.prototype.slice.call(schema.attributes || []));
                    let next = false;
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
                                const arg = copyEvalArgs.shift();
                                const parameter = map.get(argument.target.value);
                                if (!parameter) {
                                    next = true;
                                    break;
                                }
                                if (!pytorch.Utility.isType(arg, parameter.type)) {
                                    if (parameter.optional) {
                                        continue;
                                    }
                                    next = true;
                                    break;
                                }
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
                                    continue;
                                }
                                next = true;
                            } else {
                                copyArgs.shift();
                                copyEvalArgs.shift();
                            }
                        } else if (parameter.type === 'Tensor[]') {
                            const argument = copyEvalArgs[0];
                            if (!Array.isArray(argument) || !argument.every((item) => pytorch.Utility.isTensor(item) || item === null)) {
                                if (parameter.optional) {
                                    continue;
                                }
                                next = true;
                            } else {
                                copyArgs.shift();
                                copyEvalArgs.shift();
                            }
                        } else {
                            const arg = copyArgs[0];
                            if (!pytorch.Utility.isType(argument, parameter.type) && argument !== null) {
                                if (parameter.optional) {
                                    continue;
                                }
                                next = true;
                            } else if (arg.type !== '=') {
                                copyArgs.shift();
                                copyEvalArgs.shift();
                            } else {
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
                    for (let i = 0; i < schema.outputs.length; i++) {
                        const parameter = schema.outputs[i];
                        switch (parameter.type) {
                            case 'Scalar':
                            case 'Tensor':
                            case 'Tensor[]':
                                break;
                            case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                            case '__torch__.torch.classes.xnnpack.LinearOpContext':
                                break;
                            default: {
                                if (!outputTypes || schema.outputs.length !== 1 || schema.outputs[0].type !== outputTypes[0]) {
                                    next = true;
                                }
                                break;
                            }
                        }
                    }
                    if (next) {
                        continue;
                    }
                    return [ schema, args, evalArgs ];
                }
            }
        }
        return null;
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
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.args[0].args[0], 'torch.size', 1) &&
                    condition.then.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.then.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.args[0].args[0].args[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.size) {
                        const number = this.expression(assign.expression.args[1], context);
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
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.dim', 1) &&
                    condition.then.statements.length > 0 &&
                    pytorch.Utility.isCall(condition.then.statements[condition.then.statements.length - 1], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.args[0].args[0], context);
                    if (pytorch.Utility.isTensor(tensor)) {
                        const size = this.expression(assign.expression.args[1], context);
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
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.args[0].args[0], 'torch.size', 1) &&
                    condition.else.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.else.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.args[0].args[0].args[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                        const number = this.expression(assign.expression.args[1], context);
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
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.size', 1) &&
                    pytorch.Utility.isCall(condition.condition, 'torch.eq', 2) &&
                    pytorch.Utility.isCall(condition.condition.args[0], 'torch.len', 1) &&
                    pytorch.Utility.isEqual(condition.condition.args[0].args[0], assign.target) &&
                    condition.else.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.else.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.args[0].args[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                        const start = this.expression(assign.expression.args[1], context);
                        const value = this.expression(condition.condition.args[1], context);
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
                            } else if (Array.isArray(obj)) {
                                for (const item of obj) {
                                    if (Array.isArray(item) || (Object(item) === item && item.type)) {
                                        queue.push(item);
                                    }
                                }
                            } else if (Object(obj) === obj) {
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
                                    } else if (Object(value) === value && value.type) {
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
                pytorch.Utility.isCall(statement.expression.args[0], 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.args[0].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                    tensor.resize_([ 1, 3, 299, 299 ]);
                }
            }
            // torch.slice(ops.prim.shape(input), 0, 2, 1)
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                pytorch.Utility.isCall(statement.expression.args[0], 'ops.prim.shape', 1)) {
                const tensor = this.expression(statement.expression.args[0].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // _3 = torch.le(xxxx, torch.dim(f0))
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.le', 2) &&
                pytorch.Utility.isCall(statement.expression.args[1], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.args[1].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // if torch.ne(torch.dim(image), 3):
            //   xxxx
            //   ops.prim.RaiseException(_7)
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.condition, 'torch.ne', 2) &&
                pytorch.Utility.isCall(statement.condition.args[0], 'torch.dim', 1) &&
                statement.then.statements.length > 0 &&
                pytorch.Utility.isCall(statement.then.statements.slice(-1).pop(), 'ops.prim.RaiseException', 1)) {
                const tensor = this.expression(statement.condition.args[0].args[0], context);
                const size = this.expression(statement.condition.args[1], context);
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
                pytorch.Utility.isCall(statement.expression.args[0], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.args[0].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // a, b = torch.unbind(size, 0)
            if (statement.type === '=' &&
                statement.target.type === 'tuple' &&
                (pytorch.Utility.isCall(statement.expression, 'torch.unbind', 1) ||
                 pytorch.Utility.isCall(statement.expression, 'torch.unbind', 2))) {
                statement.expression.args[0].__tuple__ = statement.target.value.length;
            }
            // a, b, c = torch.size(input)
            if (statement.type === '=' &&
                statement.target.type === 'tuple' &&
                pytorch.Utility.isCall(statement.expression, 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    const dim = statement.target.value.length;
                    tensor.resize_(Array(dim).fill(NaN));
                }
            }
            // x = torch.len(input)
            if (statement.type === '=' &&
                statement.target.type === 'id' &&
                pytorch.Utility.isCall(statement.expression, 'torch.len', 1)) {
                const tensor = this.expression(statement.expression.args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // x = _(torch.size(foo ,2))
            if (statement.type === '=' &&
                statement.expression.type === 'call' && statement.expression.args.length > 0 &&
                pytorch.Utility.isCall(statement.expression.args[0], 'torch.size', 2)) {
                const tensor = this.expression(statement.expression.args[0].args[0], context);
                const dim = this.expression(statement.expression.args[0].args[1], context);
                if (pytorch.Utility.isTensor(tensor) && Number.isInteger(dim) && dim >= 0) {
                    if (tensor.shape === undefined) {
                        tensor.resize_(Array(dim + 1).fill(NaN));
                    } else if (Array.isArray(tensor.shape) && tensor.shape.length <= dim) {
                        tensor.resize_(tensor.shape.concat(Array(dim + 1 - tensor.shape.length).fill(NaN)));
                    }
                }
            }
            if (statement.type === '=' && statement.target.type === 'tuple' &&
                statement.expression.type === 'call' && statement.expression.args.length > 0 &&
                pytorch.Utility.isCall(statement.expression, 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.args[0], context);
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
};

pytorch.jit.Source = class {

    constructor(text) {
        this._text = text;
    }
};

pytorch.jit.SourceLoader = class {

    constructor(reader, code_prefix) {
        this._reader = reader;
        this._code_prefix = code_prefix;
    }

    loadSource(qualifier) {
        const path = this._code_prefix + '/' + qualifier + '.py';
        if (this._reader.hasRecord(path)) {
            const data = this._reader.getRecord(path);
            return new pytorch.jit.Source(data);
        }
        return null;
    }
};

pytorch.jit.SourceImporter = class {

    constructor(cu, constant_table, source_loader, version) {
        this._cu = cu;
        this._constant_table = constant_table;
        this._source_loader = source_loader;
        this._version = version;
    }

    loadType(/* name */) {
        // TODO;
    }

    resolveType(name) {
        return this.findNamedType(new pytorch.jit.QualifiedName(name));
    }

    findNamedType(name) {
        // TODO
        this.parseSourceIfNeeded(name.prefix());
    }

    parseSourceIfNeeded(/* qualifier */) {
        // TODO
    }
};

pytorch.jit.ScriptModuleDeserializer = class {

    constructor(cu, reader, pickle_dir_prefix, tensor_dir_prefix, storage_context) {
        this._compilation_unit = cu;
        this._reader = reader;
        this._storage_context = storage_context;
        this._code_prefix = !pickle_dir_prefix && !tensor_dir_prefix ? 'code/' : '.data/ts_code/code/';
        this._pickle_dir_prefix = pickle_dir_prefix || '';
        this._tensor_dir_prefix = tensor_dir_prefix || '';
        this._source_importer = new pytorch.jit.SourceImporter(
            this._compilation_unit, this._constants_table,
            new pytorch.jit.SourceLoader(this._reader, this._code_prefix),
            reader.version());
    }

    deserialize() {
        const execution = this._compilation_unit.execution;
        const code_prefix = this._code_prefix;
        for (const name of this._reader.getAllRecords()) {
            if (name.startsWith(code_prefix) && name.endsWith('.py')) {
                const file = name.substring(code_prefix.length);
                const stream = this._reader.getRecord(name);
                const buffer = stream.peek();
                execution.add(file, buffer);
            }
        }
        const torch = execution.import('torch');
        execution.builtins.torch = torch;
        execution.builtins.Tensor = torch.Tensor;
        execution.builtins.ops = torch.ops;
        execution.builtins.inf = torch.inf;
        execution.builtins.CONSTANTS = {};
        if (this._reader.hasRecord('model.json')) {
            return this.LEGACY_deserialize();
        }
        const constants = this.readArchive('constants');
        for (let i = 0; i < constants.length; i++) {
            execution.builtins.CONSTANTS['c' + i.toString()] = constants[i];
        }
        const module = this.readArchive('data');
        const result = new torch.ScriptModule();
        result.data = module;
        return result;
    }

    LEGACY_deserialize() {
        const execution = this._compilation_unit.execution;
        const torch = execution.import('torch');
        const stream = this._reader.getRecord('model.json');
        const buffer = stream.peek();
        const decoder = new TextDecoder('utf-8');
        const content = decoder.decode(buffer);
        const model = JSON.parse(content);
        const data = model.mainModule || {};
        const queue = [ data ];
        const tensorTypeMap = new Map([
            [ 'FLOAT', 'Float' ],
            [ 'FLOAT16', 'Half' ],
            [ 'DOUBLE', 'Double' ],
            [ 'INT8', 'Char' ],
            [ 'INT32', 'Int' ],
            [ 'INT64', 'Long' ]
        ]);
        const constants = (model.tensors || []).map((constant) => {
            const key = constant.data.key;
            if (!tensorTypeMap.has(constant.dataType)) {
                throw new pytorch.Error("Unsupported tensor data type '" + constant.dataType + "'.");
            }
            const type = tensorTypeMap.get(constant.dataType);
            const shape = constant.dims ? constant.dims.map((dim) => parseInt(dim, 10)) : null;
            const storage_type = execution.resolve('torch.' + type + 'Storage');
            const size = (shape || []).reduce((a, b) => a * b, 1);
            const offset = parseInt(constant.offset, 10) || 0;
            const storage = new storage_type([ size ]);
            const itemsize = storage.dtype.itemsize();
            const stream = this._reader.getRecord(key);
            const buffer = stream.peek();
            const length = size * itemsize;
            const data = buffer.slice(offset, offset + length);
            storage._set_cdata(data);
            const tensor = execution.invoke('torch._utils._rebuild_tensor', [ storage, 0, shape, 0 ]);
            tensor.name = constant.data.key;
            return tensor;
        });
        execution.builtins.CONSTANTS = {};
        for (let i = 0; i < constants.length; i++) {
            execution.builtins.CONSTANTS['c' + i.toString()] = constants[i];
        }
        const attributes = [];
        if (this._reader.hasRecord('attributes.pkl')) {
            const stream = this._reader.getRecord('attributes.pkl');
            const buffer = stream.peek();
            const unpickler = execution.invoke('pickle.Unpickler', [ buffer ]);
            const obj = unpickler.load();
            attributes.push(...obj);
        }
        while (queue.length > 0) {
            const module = queue.shift();
            module.__class__ = module.__class__ || { __module__: 'torch.nn.modules.module', __name__: 'Module' };
            if (module.name) {
                module.__name__ = module.name;
            }
            if (module.submodules) {
                for (const submodule of module.submodules) {
                    module[submodule.name] = submodule;
                    submodule.__parent__ = module;
                    queue.push(submodule);
                }
                delete module.submodules;
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
                const tensor = constants[parameter.tensorId];
                module[parameter.name] = tensor;
                parameter.__class__ = parameter.__class__ || { __module__: 'torch', __name__: 'Tensor' };
            }
            for (const attribute of module.attributes || []) {
                module[attribute.name] = attributes[attribute.id];
            }
            delete module.attributes;
        }
        const arena = data.torchscriptArena;
        if (arena && arena.key && arena.key.startsWith('code/')) {
            const file = arena.key.substring('code/'.length);
            const name = file.replace(/\.py$/, '').split('/').join('.');
            const module = execution.import(name);
            if (module.forward.__class__ === execution.builtins.function) {
                data.forward = module.forward;
            }
        }
        const result = new torch.ScriptModule();
        result.data = data;
        return result;
    }

    readArchive(archive_name) {
        const type_resolver = null;
        const obj_loader = null;
        return this.readArchiveAndTensors(archive_name, this._pickle_dir_prefix, this._tensor_dir_prefix, type_resolver, obj_loader, this._device, this._reader, null, this._storage_context);
    }

    readArchiveAndTensors(archive_name, pickle_prefix, tensor_prefix, type_resolver, obj_loader, device, stream_reader, type_parser, storage_context) {
        const picklename = pickle_prefix + archive_name + ".pkl";
        const stream = stream_reader.getRecord(picklename);
        const buffer = stream.peek();
        const tensor_dir_path = tensor_prefix ? tensor_prefix : archive_name + '/';
        const read_record = (name) => {
            const stream = stream_reader.getRecord(tensor_dir_path + name);
            return stream.length <= 0x40000 ? stream.peek() : stream;
        };
        const execution = this._compilation_unit.execution;
        const pickle = execution.__import__('pickle');
        const Unpickler = class extends pickle.Unpickler {
            find_class(module, name) {
                return super.find_class(module, name);
            }
        };
        const unpickler = new Unpickler(buffer);
        unpickler.persistent_load = (saved_id) => {
            const typename = saved_id[0];
            if (typename !== 'storage') {
                throw new pytorch.Error("Unsupported persistent load type '" + typename + "'.");
            }
            const storage_type = saved_id[1];
            const key = saved_id[2];
            const size = saved_id[4];
            if (storage_context && storage_context.has_storage(key)) {
                return storage_context.get_storage(key);
            }
            const storage = new storage_type(size);
            const storage_ptr = read_record(key);
            storage._set_cdata(storage_ptr);
            if (storage_context) {
                storage_context.add_storage(key);
            }
            return storage;
        };
        return unpickler.load();
    }
};

pytorch.jit.FlatBuffersLoader = class {

    constructor(cu) {
        this._cu = cu;
        const torch = cu.execution.__import__('torch');
        this._torch = torch;
        const dtypes = Array.from(new Set(Object.values(torch).filter((obj) => obj instanceof torch.dtype)));
        this._dtypes = new Map(dtypes.map((dtype) => [ dtype.scalar_type(), dtype ]));
        this._ivalue_parsers = new Map();
        this._ivalue_parsers.set(pytorch.mobile.serialization.Int, (ivalue) => ivalue.val.int_val);
        this._ivalue_parsers.set(pytorch.mobile.serialization.Bool, (ivalue) => ivalue.val.bool_val);
        this._ivalue_parsers.set(pytorch.mobile.serialization.Double, (ivalue) => ivalue.val.double_val);
        this._ivalue_parsers.set(pytorch.mobile.serialization.TensorMetadata, (ivalue) => this.parseTensor(ivalue));
        this._ivalue_parsers.set(pytorch.mobile.serialization.Object, (ivalue) => this.parseObject(ivalue));
    }

    parseModule(module) {
        this._module = module;
        this._all_functions = new Map();
        this._all_ivalues = new Array(module.ivalues.length);
        this._all_types = new Array(module.object_types.length);
        const mobile_ivalue_size = module.mobile_ivalue_size ? module.mobile_ivalue_size : module.ivalues.length;
        for (let i = 0; i < mobile_ivalue_size; i++) {
            this.parseAndPopulate(i, module.ivalues[i]);
        }
        const m = this._all_ivalues[module.state_obj];

        for (const entry of this._all_functions) {
            const name = entry[0];
            const class_index = module.ivalues[name].val.class_type;
            const class_type = this._all_types[class_index];
            class_type.addMethod(entry[1]);
        }

        m._min_operator_version = module.operator_version;
        m._bytecode_version = module.bytecode_version;
        return m;
    }

    parseAndPopulate(i, ivalue) {
        if (ivalue.val instanceof pytorch.mobile.serialization.Function) {
            this._all_functions.set(i, this.parseFunction(ivalue.val));
        } else {
            this._all_ivalues[i] = this.parseIValue(ivalue);
        }
    }

    parseFunction(/* val */) {
        return null;
    }

    parseIValue(ivalue) {
        if (ivalue.val) {
            const callback = this._ivalue_parsers.get(ivalue.val.constructor);
            return callback(ivalue);
        }
        return null;
    }

    parseTensor(ivalue) {
        return this.parseTensorFromMetadata(ivalue.val);
    }

    parseTensorFromMetadata(metadata) {
        if (metadata.quantized_schema) {
            throw new pytorch.Error('Quantized schema not implemented.');
        }
        const index = metadata.storage_location_index;
        const data = this._module.storage_data[index].data;
        const dtype = this._dtypes.get(metadata.scalar_type);
        const size = data.length / dtype.itemsize();
        const storage = this._cu.execution.invoke('torch.storage._TypedStorage', [ size, dtype ]);
        storage._set_cdata(data);
        const tensor = this._cu.execution.invoke('torch.Tensor', []);
        const shape = Array.from(metadata.sizes);
        const stride = Array.from(metadata.strides);
        tensor.__setstate__([ storage, metadata.storage_offset, shape, stride ]);
        return tensor;
    }

    parseObject(ivalue) {
        const object = ivalue.val;
        const obj_type = this._module.object_types[object.type_index];
        const cls = this.getOrCreateClassTypeForObject(object);
        switch (obj_type.type) {
            case pytorch.mobile.serialization.TypeType.CLASS_WITH_FIELD: {
                const torch = this._torch;
                const obj = torch.ScriptObject.create(cls);
                for (let i = 0; i < object.attrs.length; i++) {
                    const attr_name = obj_type.attr_names[i];
                    const val = this._all_ivalues[object.attrs[i]];
                    obj.__setattr__(attr_name, val);
                }
                return obj;
            }
            case pytorch.mobile.serialization.TypeType.CUSTOM_CLASS:
            case pytorch.mobile.serialization.TypeType.CLASS_WITH_SETSTATE:
            default: {
                throw new pytorch.Error("Unknown object type type '" + obj_type.type + "'.");
            }
        }
    }

    getOrCreateClassTypeForObject(object) {
        let cls = this._all_types[object.type_index];
        const obj_type = this._module.object_types[object.type_index];
        if (!cls) {
            const name = obj_type.type_name;
            if (name.startsWith('__torch__') || name.startsWith('torch.jit')) {
                cls = this._cu.get_class(name);
                if (!cls) {
                    const torch = this._torch;
                    cls = new torch.ClassType(name, this._cu, true);
                    this._cu.register_type(cls);
                }
            } else {
                // TODO cls = c10::parseType(qn_str)->cast<ClassType>();
            }
            this._all_types[object.type_index] = cls;
            if (obj_type.type === pytorch.mobile.serialization.TypeType.CLASS_WITH_FIELD) {
                for (let i = 0; i < object.attrs.length; i++) {
                    // const val = this._all_ivalues[object.attrs[i]];
                    cls.addAttribute(obj_type.attr_names[i] /*, null val.type(c10::DynamicType) */);
                }
            }
        }
        return cls;
    }
};

pytorch.jit.StreamReader = class {

    constructor(entries) {
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
        entries = Array.from(entries).map((entry) => [ entry[0].substring(prefix.length), entry[1] ]);
        this._entries = new Map(entries);
        this._version = 0;
        const stream = this.getRecord('.data/version') || this.getRecord('version') || null;
        if (stream) {
            const decoder = new TextDecoder('utf-8');
            const buffer = stream.peek();
            const text = decoder.decode(buffer);
            const value = text.split('\n').shift();
            this._version = parseInt(value, 10);
        }
    }

    hasRecord(name) {
        return this._entries.has(name);
    }

    getRecord(name) {
        return this._entries.get(name);
    }

    getAllRecords() {
        return Array.from(this._entries.keys());
    }

    version() {
        return this._version;
    }
};

pytorch.Container.Package = class extends pytorch.Container {

    constructor(reader) {
        super();
        this._reader = reader;
        this._format = 'PyTorch Package ' + pytorch.Utility.version(reader.version());
    }

    async read() {
        this._modules = new Map();
        const pickles = this._reader.getAllRecords().filter((name) => !name.startsWith('.data/') && !name.endsWith('py'));
        if (pickles.length > 0) {
            const execution = new pytorch.Execution();
            for (const event of this._events) {
                execution.on(event[0], event[1]);
            }
            for (const name of this._reader.getAllRecords()) {
                if (!name.startsWith('.data/') && name.endsWith('.py')) {
                    const stream = this._reader.getRecord(name);
                    const buffer = stream.peek();
                    execution.add(name, buffer);
                }
            }
            const torch = execution.__import__('torch');
            const importer = new torch.package.PackageImporter(this._reader);
            for (const name of pickles) {
                const module = importer.load_pickle(name);
                this._modules.set(name, module);
            }
        }
    }

    get format() {
        return this._format;
    }

    get modules() {
        return this._modules;
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
        if (value === null || value === undefined) {
            return undefined;
        } else if (value === true || value === false) {
            return 'boolean';
        } else if (pytorch.Utility.isTensor(value)) {
            return 'Tensor';
        } else if (typeof value === 'string') {
            return 'string';
        } else if (Number(value) === value && value % 1 === 0) {
            return 'int64';
        } else if (Number(value) === value) {
            return 'float32';
        } else if (Array.isArray(value) && value.every((item) => Number(item) === item && item % 1 === 0)) {
            return 'int64[]';
        } else if (Array.isArray(value) && value.every((item) => Number(item) === item)) {
            return 'float32[]';
        }
        const text = (JSON.stringify(value) || '(undefined)').substring(0, 10);
        throw new pytorch.Error("Unsupported ops argument type '" + text + "'.");
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
            expression.args.length === size &&
            pytorch.Utility.target(expression.target) === name) {
            return true;
        }
        return false;
    }

    static isEqual(a, b) {
        return (a.type === 'id' && b.type === 'id' && a.value === b.value);
    }

    static module() {
        const module = {};
        module.__class__ = module.__class__ || { __module__: 'torch.nn.modules.module', __name__: 'Module' };
        return module;
    }

    static version(value) {
        // https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/inline_container.h
        // kProducedFileFormatVersion
        const versions = new Map([
            [  1,  'v1.3'  ],
            [  2,  'v1.5'  ], // 7a2889b014ce36fcc333b2c6de6f29f976652f84 (#28122)
            [  3,  'v1.6'  ], // 2ec6a30722b0ef85632a2f3e7ce6f80da403008a (#36085)
            [  4,  'v1.6'  ], // 95489b590f00801bdee7f41783f30874883cf6bb (#38620)
            [  5,  'v1.7'  ], // cb26661fe4faf26386703180a9045e6ac6d157df (#40364)
            [  6,  'v1.9'  ], // 3ee7637ffa50df0d9b231c7b40778ac1c390bf4a (#59714)
            [  7,  'v1.10' ], // 880098a7e34a20628f960daa8eab0eb1ad566c39 (#63651)
            [  8,  'v1.11' ], // b28e696516a7f0c7a6ead6da967590ce6c1d6698 (#71486)
            [  9,  'v1.11' ], // 8757e21c6a4fc00e83539aa7f9c28eb11eff53c1 (#72051)
            [ 10, 'v1.12' ]  // 4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8 (#61439)
        ]);
        if (!versions.has(value)) {
            throw new pytorch.Error("Unsupported PyTorch Zip version '" + value + "'.");
        }
        return versions.get(value) || 'v-' + value.toString();
    }

    static find(data) {
        const root = pytorch.Utility.findModule(data);
        if (root) {
            return root;
        }
        const weights = pytorch.Utility.findWeights(data);
        if (weights) {
            return weights;
        }
        if (data && Array.isArray(data) && data === Object(data) && Object.entries(data).length === 0) {
            return [];
        }
        throw new pytorch.Error('File does not contain root module or state dictionary.');
    }

    static findModule(root) {
        if (root) {
            const keys = [ '', 'model', 'net' ];
            for (const key of keys) {
                const obj = key === '' ? root : root[key];
                if (obj) {
                    if (obj instanceof Map && obj.has('engine')) {
                        // https://github.com/NVIDIA-AI-IOT/torch2trt/blob/master/torch2trt/torch2trt.py
                        const data = obj.get('engine');
                        const signatures = [
                            [ 0x70, 0x74, 0x72, 0x74 ], // ptrt
                            [ 0x66, 0x74, 0x72, 0x74 ]  // ftrt
                        ];
                        for (const signature of signatures) {
                            if (data instanceof Uint8Array && data.length > signature.length && signature.every((value, index) => value === data[index])) {
                                // const buffer = data.slice(0, 24);
                                // const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                                throw new pytorch.Error('Invalid file content. File contains undocumented PyTorch TensorRT engine data.');
                            }
                        }
                    }
                    if (obj._modules) {
                        return new Map([ ['', obj] ]);
                    }
                    const entries = Object.entries(obj).filter((entry) => entry[0] && entry[1] && entry[1]._modules);
                    if (entries.length > 1) {
                        return new Map(entries);
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
            'state_dict', 'state_dict_stylepredictor', 'state_dict_ghiasi',
            'state', 'model_state', 'model', 'model_state_dict', 'model_dict', 'net_dict',
            'generator', 'discriminator',  'g_state', 'module', 'params',
            'weights', 'network_weights', 'network', 'net', 'netG', 'net_states',
            'runner', ''
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
            const module = pytorch.Utility.module();
            module._parameters = new Map();
            module._parameters.set('value', obj);
            return new Map([ [ '', { _modules: new Map([ [ '', module ] ]) } ] ]);
        }
        return null;
    }

    static _convertObjectList(obj) {
        if (obj && Array.isArray(obj)) {
            if (obj.every((item) => typeof item === 'number' || typeof item === 'string')) {
                return new Map([ ['', obj] ]);
            }
            if (obj.every((item) => item && Object.values(item).filter((value) => pytorch.Utility.isTensor(value)).length > 0)) {
                return new Map([ ['', obj] ]);
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
                    if (value && Number.isInteger(value.epoch) && value.state_dict) {
                        target[key] = value.state_dict;
                        continue;
                    }
                    if ((key.startsWith('dico_') && Object(value) === value) ||
                        (key.startsWith('best_metrics') && Object(value) === value) ||
                        (key === 'args' && Object(value) === value) ||
                        (key.startsWith('params') && Object(value) === value && (value.id2lang || value.lang2id)) ||
                        (key.startsWith('spk_dict_') && Object(value) === value && Object.keys(value).length === 0) ||
                        (key === 'blk_det') ||
                        (key === 'random_state') ||
                        (key === 'train_cfg' || key === 'test_cfg' || key === '_is_full_backward_hook')) {
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
                    } else if (keys.length >= 2 && keys[keys.length - 2] === '_packed_params') {
                        continue;
                    } else if (pytorch.Utility.isTensor(value)) {
                        tensor = true;
                        continue;
                    } else if (value && Array.isArray(value) && value.every((item) => pytorch.Utility.isTensor(item))) {
                        tensor = true;
                        continue;
                    } else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                        continue;
                    } else if (value === null) {
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
            const map = new Map(Object.entries(obj).map((entry) => [ entry[0], entry[1] ]));
            if (validate(map)) {
                return map;
            }
            const target = new Map();
            for (const entry of map) {
                const value = flatten(entry[1]);
                if (value && value instanceof Map) {
                    for (const pair of value) {
                        target.set(entry[0] + '.' + pair[0], pair[1]);
                    }
                    continue;
                }
                return null;
            }
            return target;
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
        } else if (obj instanceof Map && validate(obj)) {
            map.set('', flatten(obj));
        } else if (Object(obj) === obj && Object.entries(obj).every((entry) => validate(entry[1]))) {
            for (const entry of Object.entries(obj)) {
                map.set(entry[0], entry[1]);
            }
        } else if (Object(obj) === obj && Object.entries(obj).every((entry) => pytorch.Utility.isTensor(entry[1]))) {
            map.set('', new Map(Object.keys(obj).map((key) => [ key, obj[key] ])));
        } else {
            const value = flatten(obj);
            if (value) {
                map.set('', value);
            }
        }
        if (map.size > 0) {
            const modules = new Map();
            for (const entry of map) {
                const graph_name = entry[0];
                const layer_map = entry[1];
                const layers = new Map();
                for (const item of layer_map) {
                    const key = item[0];
                    const value = item[1];
                    let layer_name = '';
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
                    } else {
                        parameter = keys.pop();
                        if (keys.length < 0) {
                            keys.push('');
                        }
                    }
                    layer_name = keys.join(separator);
                    if (!layers.has(layer_name)) {
                        const module = pytorch.Utility.module();
                        layers.set(layer_name, module);
                    }
                    const layer = layers.get(layer_name);
                    if (pytorch.Utility.isTensor(value)) {
                        layer._parameters = layer._parameters || new Map();
                        value.name = key;
                        layer._parameters.set(parameter, value);
                        if (layer_name == '' && layer._parameters.length > 12) {
                            return null;
                        }
                    } else if (value && Array.isArray(value) && value.every((item) => pytorch.Utility.isTensor(item))) {
                        layer._parameters.set(parameter, value);
                    } else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                        layer[parameter] = value;
                    }
                }
                modules.set(graph_name, { _modules: layers });
            }
            return modules;
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
                    const storage = buffers[number];
                    const data = storage.data && storage.data.peek ? storage.data.peek() : storage.data;
                    operand.data = data.slice(offset, operand_length);
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
        this.register(0, 'ADD', '', [ 'A', 'B' ], [ [ 'activation', 'int32'] ], [ 'C' ]);
        this.register(1, 'AVERAGE_POOL_2D', 'Pool', [ 'input' ], [ [ 'padding_left', 'int32' ], [ 'padding_right', 'int32' ], [ 'padding_top', 'int32' ], [ 'padding_bottom', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'filter_x', 'int32' ], [ 'filter_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ] ], [ 'output' ]);
        this.register(1, 'AVERAGE_POOL_2D', 'Pool', [ 'input' ], [ [ 'padding_scheme', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'filter_x', 'int32' ], [ 'filter_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ] ], [ 'output' ]);
        this.register(2, 'CONCATENATION');
        this.register(3, 'CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_left', 'int32' ], [ 'padding_right', 'int32' ], [ 'padding_top', 'int32' ], [ 'padding_bottom', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register(3, 'CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_scheme', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register(4, 'DEPTHWISE_CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_left', 'int32' ], [ 'padding_right', 'int32' ], [ 'padding_top', 'int32' ], [ 'padding_bottom', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register(4, 'DEPTHWISE_CONV_2D', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'padding_scheme', 'int32' ], [ 'stride_x', 'int32' ], [ 'stride_y', 'int32' ], [ 'activation', 'int32' ], [ 'nchw', 'boolean' ], [ 'dilation_width', 'int32' ], [ 'dilation_height', 'int32' ] ], [ 'output' ]);
        this.register(5, 'DEPTH_TO_SPACE');
        this.register(6, 'DEQUANTIZE');
        this.register(7, 'EMBEDDING_LOOKUP');
        this.register(8, 'FLOOR');
        this.register(9, 'FULLY_CONNECTED', 'Layer', [ 'input', 'weights', 'bias' ], [ [ 'activation', 'int32' ] ], [ 'output' ]);
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
        const type = {};
        type.name = name;
        type.inputs = (inputs || []).map((name) => ({ name: name, type: 'Tensor' }));
        type.outputs = (outputs || []).map((name) => ({ name: name, type: 'Tensor' }));
        type.attributes = (attributes || []).map((pair) => ({ name: pair[0], type: pair[1] }));
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




                const value = new pytorch.nnapi.Argument(operand);
                args.set(operand.index, value);
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
            const value = arg(operand);
            const argument = new pytorch.Argument(i.toString(), true, [ value ]);
            this._inputs.push(argument);
        }

        for (let i = 0; i < model.outputs.length; i++) {
            const operand = model.outputs[i];
            const value = arg(operand);
            const argument = new pytorch.Argument(i.toString(), true, [ value ]);
            this._outputs.push(argument);
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
                    const value = arg(operand);
                    const argument = new pytorch.Argument(name, true, [ value ]);
                    this._inputs.push(argument);
                } else if (name === 'activation') {
                    const activation = new Map([ [ 1, 19 ], [ 2, 20 ], [ 3, 21 ] ]).get(operand.value) || 0;
                    if (activation !== 0) {
                        this._chain.push(new pytorch.nnapi.Node(metadata, { index: activation }));
                    }
                } else {
                    const attribute = new pytorch.nnapi.Attribute(name, operand);
                    this._attributes.push(attribute);
                }
            }
        }

        if (operation.outputs) {
            for (let i = 0; i < operation.outputs.length; i++) {
                const name = i < inputs.length ? inputs[i].name : i.toString();
                const operand = operation.outputs[i];
                const value = arg(operand);
                const argument = new pytorch.Argument(name, true, [ value ]);
                this._outputs.push(argument);
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

    static async open(context) {
        if (pytorch.Metadata._metadata) {
            return pytorch.Metadata._metadata;
        }
        try {
            const data = await context.request('pytorch-metadata.json', 'utf-8', null);
            pytorch.Metadata._metadata = new pytorch.Metadata(data);
            return pytorch.Metadata._metadata;
        } catch (error) {
            pytorch.Metadata._metadata = new pytorch.Metadata(null);
            return pytorch.Metadata._metadata;
        }
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
