
// Experimental

import * as base from './base.js';
import * as flatbuffers from './flatbuffers.js';
import * as python from './python.js';

const pytorch = {};
const numpy = {};

pytorch.ModelFactory = class {

    match(context) {
        const container = pytorch.Container.open(context);
        if (container) {
            context.type = container.type;
            context.target = container;
        }
    }

    filter(context, type) {
        if (context.type === 'pytorch.export' && type === 'pytorch.zip') {
            return false;
        }
        if (context.type === 'pytorch.index' && type === 'pytorch.zip') {
            return false;
        }
        if (context.type === 'pytorch.model.json' && type === 'pytorch.data.pkl') {
            return false;
        }
        if (context.type === 'pytorch.model.json' && type === 'pickle') {
            return false;
        }
        return true;
    }

    async open(context) {
        const metadata = await pytorch.Metadata.open(context);
        const target = context.target;
        target.on('resolve', (_, name) => {
            context.error(new pytorch.Error(`Unknown type name '${name}'.`), false);
        });
        await target.read(metadata);
        if (!target.format || (!target.modules && !target.module)) {
            throw new pytorch.Error("Container not implemented.");
        }
        return new pytorch.Model(metadata, target);
    }
};

pytorch.Model = class {

    constructor(metadata, target) {
        this.format = target.format;
        this.producer = target.producer || '';
        this.graphs = [];
        if (target.module) {
            const graph = new pytorch.Graph(metadata, null, '', target.module);
            this.graphs.push(graph);
        } else if (target.modules) {
            for (const [name, value] of target.modules) {
                const graph = new pytorch.Graph(metadata, null, name, value);
                this.graphs.push(graph);
            }
        }
    }
};

pytorch.Graph = class {

    constructor(metadata, type, name, module) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        this.name = name || '';
        this.type = type;
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (tensor) {
                return new pytorch.Value(name, type, null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new pytorch.Value(name, type, null, tensor));
            } else if (type || tensor) {
                throw new pytorch.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        type = module && module.__class__ && module.__class__.__module__ && module.__class__.__name__ ? `${module.__class__.__module__}.${module.__class__.__name__}` : null;
        if ((type === 'torch.ScriptModule' || type === 'torch.jit._script.ScriptModule' || type === 'torch.jit._script.RecursiveScriptModule') && module.graph) {
            const initializers = new Map();
            const graph = module.graph;
            const constants = module.code_with_constants[1].const_mapping;
            if (constants) {
                for (const [key, value] of constants) {
                    const name = `CONSTANTS.${key}`;
                    if (pytorch.Utility.isTensor(value)) {
                        initializers.set(value, new pytorch.Tensor(name, value));
                    } else if (pytorch.Utility.isObject(value)) {
                        initializers.set(value, value);
                    } else {
                        // throw new pytorch.Error('Unsupported constant.');
                    }
                }
            }
            const queue = [module.data];
            while (queue.length > 0) {
                const module = queue.shift();
                for (const [key, obj] of Object.entries(module)) {
                    if (key !== '__module__' && key !== '__name__' && key !== '__class__' && key !== '__parent__') {
                        if (!Array.isArray(obj) && obj === Object(obj)) {
                            if (pytorch.Utility.isTensor(obj)) {
                                const parameter = obj;
                                parameter.__parent__ = module;
                                if (parameter.storage() && !parameter.__origin__) {
                                    if (parameter.__count__ === undefined || parameter.__count__ === 1) {
                                        initializers.set(parameter, new pytorch.Tensor(parameter.name, parameter));
                                    }
                                }
                            } else if (pytorch.Utility.isObject(obj)) {
                                if (obj.__count__ === undefined || obj.__count__ === 1) {
                                    initializers.set(obj, obj);
                                }
                                queue.push(obj);
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
                this.inputs.push(new pytorch.Argument(name, [values.map(identifier)]));
            }
            for (const value of graph.outputs()) {
                const identifier = value.unique().toString();
                this.outputs.push(new pytorch.Argument(identifier, [values.map(identifier)]));
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
                this.nodes.push(new pytorch.Node(metadata, null, null, node, initializers, values));
            }
            if (module) {
                const queue = [module.data];
                while (queue.length > 0) {
                    const module = queue.pop();
                    if (module && !pytorch.Utility.isObject(module)) {
                        if (!module.__hide__ && pytorch.Graph._getParameters(module).size > 0) {
                            for (const [name, obj] of Object.entries(module)) {
                                if ((obj && obj.__hide__) || (obj !== null && !pytorch.Utility.isTensor(obj)) && typeof obj !== 'boolean' && typeof obj !== 'number' && typeof obj !== 'string') {
                                    delete module[name];
                                }
                            }
                            const node = new pytorch.Node(metadata, null, null, module, initializers, values);
                            this.nodes.push(node);
                        }
                        const modules = [];
                        if (module.__class__ && module.__class__.__module__ && module.__class__.__name__) {
                            for (const [key, value] of Object.entries(module)) {
                                if (!key.startsWith('__') && value && value.__class__ && value.__class__.__module__ && value.__class__.__name__ && !pytorch.Utility.isTensor(value)) {
                                    modules.push(value);
                                }
                            }
                        }
                        queue.push(...modules.reverse());
                    }
                }
            }
        } else if (pytorch.Utility.isTensor(module)) {
            const node = new pytorch.Node(metadata, null, type, { value: module });
            this.nodes.push(node);
        } else {
            const weights = this.type === 'weights' ? module : pytorch.Utility.weights(module);
            if (weights) {
                this.name = !this.name && typeof module.__name__ === 'string' ? module.__name__ : this.name;
                for (const [name, module] of weights) {
                    const node = new pytorch.Node(metadata, name, 'Weights', module);
                    this.nodes.push(node);
                }
            } else {
                const modules = Array.isArray(module) && module.every((module) => module && !pytorch.Utility.isTensor(module) && (module._modules !== undefined || module.__class__)) ? module : [module];
                for (const module of modules) {
                    const type = this.type === 'weights' ? 'Weights' : null;
                    const node = new pytorch.Node(metadata, null, type, module, null, values);
                    this.nodes.push(node);
                }
            }
        }
    }

    static _getParameters(module) {
        const parameters = new Map();
        if (module && module.__class__.__module__ && module.__class__.__name__) {
            for (const [key, value] of Object.entries(module)) {
                if (pytorch.Utility.isTensor(value)) {
                    parameters.set(key, value);
                }
            }
        }
        return parameters;
    }
};

pytorch.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

pytorch.Value = class {

    constructor(name, type, quantization, initializer) {
        if (typeof name !== 'string') {
            throw new pytorch.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer && initializer.type ? initializer.type : type || null;
        this.quantization = quantization;
        this.initializer = initializer || null;
    }
};

pytorch.Node = class {

    constructor(metadata, name, type, obj, initializers, values, stack) {
        this.name = name || '';
        this.nodes = [];
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        const createType = (metadata, name) => {
            if (name instanceof pytorch.nnapi.Graph) {
                return name;
            }
            const key = name.startsWith('__torch__.') ? name.substring(10) : name;
            const value = metadata.type(key);
            const type = value ? { ...value } : { name };
            type.identifier = name;
            type.name = type.name.indexOf('::') === -1 ? type.name : type.name.split('::').pop().split('.')[0];
            return type;
        };
        const createAttribute = (metadata, name, value) => {
            let visible = true;
            let type = 'attribute';
            metadata = name === 'training' ? { type: 'boolean', visible: false } : metadata;
            if (metadata) {
                if (metadata.type) {
                    type = metadata.type;
                }
                if (metadata.visible === false) {
                    visible = false;
                } else if (metadata.default !== undefined) {
                    if (Array.isArray(value)) {
                        if (Array.isArray(metadata.default)) {
                            visible = value.length !== metadata.default || !value.every((item, index) => item === metadata.default[index]);
                        } else {
                            visible = !value.every((item) => item === metadata.default);
                        }
                    } else {
                        visible = value !== metadata.default;
                    }
                }
            }
            if (Array.isArray(value) && value.length > 0 && value.every((obj) => obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('torch.nn'))) {
                value = '?';
            }
            return new pytorch.Argument(name, value, type, visible);
        };
        let module = null;
        if (pytorch.Utility.isInstance(obj, 'torch.Node')) {
            const node = obj;
            this.type = createType(metadata, node.kind());
            let match = true;
            let count = 0;
            for (const input of node.inputs()) {
                const value = input.value;
                let values = [];
                if (pytorch.Utility.isObject(value)) {
                    values = Object.values(value);
                } else if (pytorch.Utility.isTensor(value)) {
                    values = [value];
                    if (input.node() &&
                        input.node().kind() === 'prim::ListConstruct' &&
                        input.uses().length === 1 &&
                        input.node().inputs().every((input) => pytorch.Utility.isTensor(input.value))) {
                        values = input.node().inputs().map((input) => input.value);
                    }
                }
                for (const value of values) {
                    const parameter = initializers.get(value);
                    if (parameter) {
                        if (value.__parent__ && (module === null || module === value.__parent__)) {
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
                if (parameters.size === count && match) {
                    module.__hide__ = true;
                } else {
                    module = null;
                }
            }
            const inputs = node.inputs();
            for (let i = 0; i < inputs.length; i++) {
                const input = inputs[i];
                const schema = this.type && this.type.inputs && i < this.type.inputs.length ? this.type.inputs[i] : null;
                const name = schema && schema.name ? schema.name : i.toString();
                let type = schema && schema.type ? schema.type : null;
                let array = false;
                if (type && type.endsWith('[]')) {
                    array = true;
                    type = type.slice(0, -2);
                }
                let argument = null;
                if (pytorch.Utility.isObjectType(type)) {
                    const obj = input.value;
                    if (!array && initializers.has(obj)) {
                        const node = new pytorch.Node(metadata, name, type, obj, initializers, values);
                        argument = new pytorch.Argument(name, node, 'object');
                    } else if (array && Array.isArray(obj) && obj.every((obj) => initializers.has(obj))) {
                        const node = obj.map((obj) => new pytorch.Node(metadata, name, type, obj, initializers, values));
                        argument = new pytorch.Argument(name, node, 'object[]');
                    } else {
                        const identifier = input.unique().toString();
                        const value = values.map(identifier);
                        argument = new pytorch.Argument(name, [value]);
                    }
                } else if (pytorch.Utility.isTensor(input.value) || input.value === undefined || input.value === null) {
                    let list = [input];
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
                            return new pytorch.Value(identifier, null, null, initializer);
                        }
                        return values.map(identifier);
                    });
                    argument = new pytorch.Argument(name, args);
                } else {
                    argument = createAttribute(schema, schema.name, input.value);
                }
                this.inputs.push(argument);
            }
            const outputs = node.outputs();
            for (let i = 0; i < outputs.length; i++) {
                const output = outputs[i];
                const metadata = this.type && this.type.outputs && i < this.type.outputs.length ? this.type.outputs[i] : null;
                let name = '';
                if (metadata && metadata.name) {
                    name = metadata.name;
                } else {
                    name = i === 0 ? 'output' : `output${i}`;
                }
                let list = [output];
                if (output.uses().length === 1 &&
                    output.uses()[0].user &&
                    output.uses()[0].user.kind() === 'prim::ListUnpack' &&
                    output.uses()[0].user.outputs().every((output) => pytorch.Utility.isTensor(output.value))) {
                    list = output.uses()[0].user.outputs();
                }
                const args = list.map((output) => values.map(output.unique().toString()));
                const argument = new pytorch.Argument(name, args);
                this.outputs.push(argument);
            }
        } else {
            if (!type) {
                if (pytorch.Utility.isInstance(obj, 'torch.jit._script.RecursiveScriptModule') && obj._c && obj._c.qualified_name) {
                    type = obj._c.qualified_name;
                } else if (pytorch.Utility.isInstance(obj, 'builtins.function')) {
                    type = `${obj.__module__}.${obj.__name__}`;
                    obj = {};
                } else if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                    type = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                } else {
                    type = 'builtins.object';
                }
            }
            this.type = createType(metadata, type);
            stack = stack || new Set();
            const weights = pytorch.Utility.weights(obj);
            if (weights) {
                const type = this.type.name;
                this.type = new pytorch.Graph(metadata, 'weights', '', weights);
                this.type.name = type;
            } else if (obj && pytorch.Utility.isInstance(obj, 'fastai.data.core.DataLoaders')) {
                // continue
            } else if (obj && pytorch.Utility.isInstance(obj, '__torch__.torch.classes._nnapi.Compilation')) {
                // continue
            } else if (obj && type === 'builtins.bytearray') {
                const argument = new pytorch.Argument('value', Array.from(obj), 'byte[]');
                this.inputs.push(argument);
            } else if (obj) {
                const inputs = new Map(Array.isArray(this.type.inputs) ? this.type.inputs.map((input) => [input.name, input]) : []);
                const list = obj instanceof Map ? Array.from(obj) : Object.entries(obj);
                for (const [name, value] of list) {
                    if (name === '__class__' || name === '__parent__' || name === '__name__') {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'collections.OrderedDict') && value instanceof Map && value.size === 0) {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'builtins.set') && value instanceof Set && value.size === 0) {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'builtins.list') && Array.isArray(value) && value.length === 0) {
                        continue;
                    }
                    const parameters = new Map();
                    if ((name === '_parameters' || name === '_buffers') && value instanceof Map && value.size > 0) {
                        for (const [name, obj] of Array.from(value)) {
                            parameters.set(name, obj);
                        }
                    } else if (Array.isArray(value) && value.every((tensor) => pytorch.Utility.isTensor(tensor))) {
                        parameters.set(name, value);
                    } else if (pytorch.Utility.isTensor(value)) {
                        parameters.set(name, value);
                    }
                    if (parameters.size > 0) {
                        for (const [name, value] of parameters) {
                            const list = Array.isArray(value) ? value.map((item) => pytorch.Utility.toTensor(item)) : [pytorch.Utility.toTensor(value)];
                            const visible = inputs.has(name) ? inputs.get(name).visible || true : true;
                            const args = list.filter((value) => value !== null && !value.__origin__).map((value) => {
                                const name = value && value.name ? value.name : '';
                                const identifier = list.length === 1 && value && value.__name__ ? value.__name__ : name;
                                let tensor = null;
                                if (initializers && initializers.has(value)) {
                                    tensor = initializers.get(value);
                                } else {
                                    value = value.__source__ ? value.__source__ : value;
                                    tensor = value ? new pytorch.Tensor(identifier, value) : null;
                                }
                                return new pytorch.Value(identifier, null, null, tensor);
                            });
                            const argument = new pytorch.Argument(name, args, null, visible);
                            this.inputs.push(argument);
                            if (value && value.__variable__) {
                                const argument = new pytorch.Argument(name, [values.map(value.__variable__)]);
                                this.outputs.push(argument);
                            }
                        }
                        continue;
                    }
                    const type = this.type.identifier;
                    if (pytorch.Utility.isTensor(value)) {
                        const tensor = new pytorch.Tensor('', value);
                        const argument = new pytorch.Argument(name, tensor, 'tensor');
                        this.inputs.push(argument);
                    } else if (value && pytorch.Utility.isInstance(value, 'torch.dtype')) {
                        const node = new pytorch.Node(metadata, null, value.toString(), {});
                        const argument = new pytorch.Argument(name, node, 'object');
                        this.inputs.push(argument);
                    } else if (Array.isArray(value) && value.some((value) => pytorch.Utility.isTensor(value)) && value.every((value) => pytorch.Utility.isTensor(value) || value === null)) {
                        const tensors = value.map((value) => value === null ? value : new pytorch.Tensor('', value));
                        const argument = new pytorch.Argument(name, tensors, 'tensor[]');
                        this.inputs.push(argument);
                    } else if (pytorch.Utility.isInstance(value, 'numpy.ndarray') || pytorch.Utility.isInstance(value, 'numpy.matrix')) {
                        const tensor = new numpy.Tensor(value);
                        const argument = new pytorch.Argument(name, tensor, 'tensor');
                        this.inputs.push(argument);
                    } else if (Array.isArray(value) && value.every((value) => typeof value === 'string')) {
                        const argument = new pytorch.Argument(name, value, 'string[]');
                        this.inputs.push(argument);
                    } else if (Array.isArray(value) && value.every((value) => typeof value === 'number')) {
                        const argument = new pytorch.Argument(name, value, 'attribute');
                        this.inputs.push(argument);
                    } else if (name === '_modules' && pytorch.Utility.isInstance(value, 'collections.OrderedDict') &&
                        value instanceof Map && Array.from(value).every(([, value]) => value === null || value.__class__)) {
                        const values = Array.from(value).filter(([, value]) => !stack.has(value)).map(([name, obj]) => {
                            stack.add(value);
                            const type = obj === null ? 'builtins.NoneType' : `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                            const node = new pytorch.Node(metadata, this.name ? `${this.name}.${name}` : name, type, obj);
                            stack.delete(value);
                            return node;
                        });
                        const argument = new pytorch.Argument(name, values, 'object[]');
                        this.inputs.push(argument);
                    } else if (value && Array.isArray(value) && value.length > 0 && value.every((obj) => Array.isArray(obj) && obj.every((item) => typeof item === 'string' || typeof item === 'number'))) {
                        const argument = new pytorch.Argument(name, value, 'attribute');
                        this.inputs.push(argument);
                    } else if (value && Array.isArray(value) && value.length > 0 && value.every((obj) => obj && (obj.__class__ || obj === Object(obj)))) {
                        const list = value.filter((value) => !stack.has(value));
                        const nodes = list.map((value) => {
                            stack.add(value);
                            const node = new pytorch.Node(metadata, null, null, value, initializers, values, stack);
                            stack.delete(value);
                            return node;
                        });
                        const argument = new pytorch.Argument(name, nodes, 'object[]');
                        this.inputs.push(argument);
                    } else if (value && (value.__class__ || typeof value === 'object') && !stack.has(value)) {
                        stack.add(value);
                        const node = new pytorch.Node(metadata, null, null, value, initializers, values, stack);
                        stack.delete(value);
                        const visible = name !== '_metadata' || !pytorch.Utility.isMetadataObject(value);
                        const argument = new pytorch.Argument(name, node, 'object', visible);
                        this.inputs.push(argument);
                    } else {
                        const argument = createAttribute(metadata.attribute(type, name), name, value);
                        this.inputs.push(argument);
                    }
                }
            }
        }
        if (module && module.__name__) {
            this.name = module.__name__;
            while (module.__parent__) {
                module = module.__parent__;
                if (module.__name__) {
                    this.name = `${module.__name__}.${this.name}`;
                }
            }
        }
    }
};

pytorch.Tensor = class {

    constructor(name, tensor) {
        this.name = name || '';
        const layout = tensor.layout ? tensor.layout.__str__() : null;
        const storage = tensor.storage();
        const size = tensor.size() || [];
        if (layout && layout.startsWith('torch.sparse_')) {
            this.type = new pytorch.TensorType(storage.dtype.__reduce__(), new pytorch.TensorShape(size), layout.split('.').pop().replace('_', '.'));
            this.indices = new pytorch.Tensor('', tensor.indices);
            this._values = new pytorch.Tensor('', tensor.values);
        } else if (!layout || layout === 'torch.strided') {
            this.type = new pytorch.TensorType(storage.dtype.__reduce__(), new pytorch.TensorShape(size));
            this._data = storage.data;
            this.encoding = '<';
            this.indices = null;
            this.stride = tensor.stride();
            const stride = this.stride;
            const offset = tensor.storage_offset();
            let length = 0;
            if (!Array.isArray(stride)) {
                length = storage.size();
            } else if (size.every((v) => v !== 0)) {
                length = size.reduce((a, v, i) => a + stride[i] * (v - 1), 1);
            }
            if (offset !== 0 || length !== storage.size()) {
                const itemsize = storage.dtype.itemsize();
                this._offset = itemsize * offset;
                this._length = itemsize * length;
            }
        } else {
            throw new pytorch.Error(`Unsupported tensor layout '${layout}'.`);
        }
    }

    get values() {
        const type = this.type.layout;
        if (type && type.startsWith('sparse.')) {
            return this._values;
        }
        if (this._data instanceof Uint8Array) {
            return this._data;
        }
        if (this._offset !== undefined) {
            const stream = this._data;
            const position = stream.position;
            stream.seek(this._offset);
            const values = stream.peek(this._length);
            stream.seek(position);
            return values;
        }
        if (this._data) {
            return this._data.peek();
        }
        return null;
    }

    decode() {
        if (this.encoding !== '<') {
            throw new pytorch.Error(`Tensor encoding '${this.encoding}' not implemented.`);
        }
        const type = this.type;
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
                throw new pytorch.Error(`Tensor data type '${type.dataType}' not implemented.`);
            }
        }
    }
};

pytorch.TensorType = class {

    constructor(dataType, shape, layout) {
        this.dataType = dataType;
        this.shape = shape;
        this.layout = layout;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

pytorch.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions || [];
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

pytorch.Container = class {

    static open(context) {
        const types = [
            pytorch.Container.Zip,
            pytorch.Container.Pickle,
            pytorch.Container.Tar,
            pytorch.Container.data_pkl,
            pytorch.Container.torch_utils,
            pytorch.Container.Mobile,
            pytorch.Container.ModelJson,
            pytorch.Container.Index,
            pytorch.Container.ExportedProgram,
            pytorch.Container.ExecuTorch,
        ];
        for (const type of types) {
            const container = type.open(context);
            if (container) {
                return container;
            }
        }
        return null;
    }

    constructor() {
        this._events = [];
    }

    async read() {
    }

    on(event, callback) {
        this._events.push([event, callback]);
    }
};

pytorch.Container.Tar = class extends pytorch.Container {

    static open(context) {
        const entries = context.peek('tar');
        if (entries instanceof Map && entries.has('pickle')) {
            return new pytorch.Container.Tar(entries);
        }
        return null;
    }

    constructor(entries) {
        super();
        this.type = 'pytorch.tar';
        this.entries = entries;
    }

    async read() {
        this.format = 'PyTorch v0.1.1';
        const execution = new pytorch.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        this.module = torch.load(this.entries);
        delete this.entries;
    }
};

pytorch.Container.Pickle = class extends pytorch.Container {

    static open(context) {
        const stream = context.stream;
        const signature = [0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            return new pytorch.Container.Pickle(stream);
        }
        return null;
    }

    constructor(stream) {
        super();
        this.type = 'pytorch.pickle';
        this.stream = stream;
    }

    async read() {
        this.format = 'PyTorch v0.1.10';
        const data = this.stream.length < 0x7ffff000 ? this.stream.peek() : this.stream;
        delete this.stream;
        const execution = new pytorch.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        this.module = torch.load(data);
    }
};

pytorch.Container.data_pkl = class extends pytorch.Container {

    static open(context) {
        const obj = context.peek('pkl');
        if (obj) {
            if (obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                const name = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                if (name.startsWith('__torch__.')) {
                    return new pytorch.Container.data_pkl('', obj);
                }
            }
            if (pytorch.Utility.isTensor(obj)) {
                return new pytorch.Container.data_pkl('tensor', obj);
            }
            if (Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => pytorch.Utility.isTensor(tensor))) {
                return new pytorch.Container.data_pkl('tensor', obj);
            }
            if (obj instanceof Map) {
                const entries = Array.from(obj).filter(([, value]) => pytorch.Utility.isTensor(value));
                if (entries.length > 0) {
                    return new pytorch.Container.data_pkl('tensor', obj);
                }
            } else if (!Array.isArray(obj)) {
                const entries = Object.entries(obj).filter(([, value]) => pytorch.Utility.isTensor(value));
                if (entries.length > 0) {
                    return new pytorch.Container.data_pkl('tensor', obj);
                }
            }
            for (const key of ['', 'model', 'net']) {
                const module = key === '' ? obj : obj[key];
                if (module && module._modules && pytorch.Utility.isInstance(module._modules, 'collections.OrderedDict')) {
                    return new pytorch.Container.data_pkl('module', module);
                }
            }
        }
        return null;
    }

    constructor(type, data) {
        super();
        this.type = 'pytorch.data.pkl';
        this._type = type;
        this._data = data;
    }

    async read() {
        this.format = 'PyTorch Pickle';
        switch (this._type) {
            case 'module': {
                if (this._data) {
                    this.module = this._data;
                    delete this._data;
                }
                return this.module;
            }
            case 'tensor':
            case 'tensor[]':
            case 'tensor<>': {
                if (this._data) {
                    this.module = this._data;
                    delete this._data;
                }
                return this.module;
            }
            default: {
                throw new pytorch.Error("PyTorch standalone 'data.pkl' not supported.");
            }
        }
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
                    const obj = context.peek('pkl');
                    if (obj && Object.entries(obj).some(([, value]) => pytorch.Utility.isInstance(value, 'torch.nn.modules.module.Module'))) {
                        return new pytorch.Container.torch_utils(obj);
                    }
                }
            }
        }
        return null;
    }

    constructor(obj) {
        super();
        this.type = 'pytorch.torch_utils';
        this.obj = obj;
    }

    async read() {
        this.format = 'PyTorch torch_utils';
        this.module = this.obj;
        delete this.obj;
    }
};

pytorch.Container.Mobile = class extends pytorch.Container {

    static open(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'PTMF') {
            return new pytorch.Container.Mobile(context);
        }
        return null;
    }

    constructor(context) {
        super();
        this.type = 'pytorch.mobile';
        this.context = context;
    }

    async read(metadata) {
        pytorch.mobile = await this.context.require('./pytorch-schema');
        pytorch.mobile = pytorch.mobile.torch.jit.mobile;
        const execution = new pytorch.jit.Execution(null, metadata);
        for (const event in this._events) {
            execution.on(event[0], event[1]);
        }
        const stream = this.context.stream;
        const torch = execution.__import__('torch');
        this.module = torch.jit.jit_module_from_flatbuffer(stream);
        const version = this.module._c._bytecode_version.toString();
        this.format = pytorch.Utility.format('PyTorch Mobile', version);
        delete this.context;
    }
};

pytorch.Container.ExecuTorch = class extends pytorch.Container {

    static open(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'ET12') {
            return new pytorch.Container.ExecuTorch(context);
        }
        return null;
    }

    constructor(context) {
        super();
        this.type = 'pytorch.executorch';
        this.context = context;
    }

    async read() {
        pytorch.executorch = await this.context.require('./pytorch-schema');
        pytorch.executorch = pytorch.executorch.executorch_flatbuffer;
        const reader = this.context.read('flatbuffers.binary');
        /* const program = */ pytorch.executorch.Program.create(reader);
        throw new pytorch.Error('Invalid file content. File contains executorch.Program data.');
    }
};

pytorch.Container.Zip = class extends pytorch.Container {

    static open(context) {
        const entries = context.peek('zip');
        if (entries instanceof Map && entries.size > 0) {
            let prefix = 0;
            const paths = Array.from(entries.keys()).map((path) => path.replace(/\\/g, '/').split('/').reverse());
            for (let set = new Set(); set && paths.length > 0;) {
                set = new Set(paths.map((path) => path.length > 1 ? path.pop() : null));
                set = set.size > 1 || set.keys().next().value === null ? null : set;
                prefix += set ? set.keys().next().value.length + 1 : 0;
            }
            const records = new Map(Array.from(entries).map(([name, value]) => [name.substring(prefix), value]));
            if (records.has('model.json')) {
                return null;
            }
            if (records.has('data.pkl')) {
                return new pytorch.Container.Zip(entries);
            }
            if (records.has('.data/version')) {
                return new pytorch.Container.Package(entries);
            }
        }
        return null;
    }

    constructor(entries) {
        super();
        this.type = 'pytorch.zip';
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
        this._entries = entries;
    }

    async read(metadata) {
        const execution = new pytorch.jit.Execution(null, metadata);
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this._entries);
        let torchscript = reader.has_record('constants.pkl');
        const version = reader.version();
        if (torchscript) {
            const module = torch.jit.load(reader);
            execution.trace = true;
            if (module.data && module.data.forward) {
                this.module = module;
            } else {
                torchscript = false;
                this.module = module.data;
            }
        } else {
            const records = reader.get_all_records().map((key) => [key, reader.get_record(key)]);
            const entries = new Map(records);
            this.module = torch.load(entries);
        }
        const name = torchscript ? 'TorchScript' : 'PyTorch';
        this.format = pytorch.Utility.format(name, version);
        delete this._model;
        delete this._entries;
    }
};

pytorch.Container.ModelJson = class extends pytorch.Container {

    static open(context) {
        const identifier = context.identifier;
        if (identifier === 'model.json') {
            const model = context.peek('json');
            if (model && model.mainModule) {
                const entries = new Map();
                entries.set('model.json', context.stream);
                return new pytorch.Container.ModelJson(context, entries, model);
            }
        }
        return null;
    }

    constructor(context, entries, model) {
        super();
        this.type = 'pytorch.model.json';
        this._context = context;
        this._entries = entries;
        this._model = model;
    }

    async read(metadata) {
        const keys = [
            'attributes.pkl',
            'version',
            ...this._model.tensors.filter((tensor) => tensor && tensor.data && tensor.data.key).map((tensor) => tensor.data.key)
        ];
        if (this._model.mainModule.torchscriptArena && this._model.mainModule.torchscriptArena.key) {
            keys.push(this._model.mainModule.torchscriptArena.key);
        }
        const values = await Promise.all(keys.map((name) => this._context.fetch(name).then((context) => context.stream).catch(() => null)));
        for (let i = 0; i < keys.length; i++) {
            if (values[i]) {
                this._entries.set(keys[i], values[i]);
            }
        }
        const execution = new pytorch.jit.Execution(null, metadata);
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this._entries);
        if (this._model && this._model.producerName) {
            this.producer = this._model.producerName + (this._model.producerVersion ? ` v${this._model.producerVersion}` : '');
        }
        this.format = reader.has_record('attributes.pkl') ? 'TorchScript v1.1' : 'TorchScript v1.0';
        const module = torch.jit.load(reader);
        execution.trace = true;
        if (module.data && module.data.forward) {
            this.module = module;
        } else {
            this.module = module.data;
        }
        delete this._context;
        delete this._model;
        delete this._entries;
    }
};

pytorch.Container.Index = class extends pytorch.Container {

    static open(context) {
        const obj = context.peek('json');
        if (obj && obj.weight_map) {
            const entries = Object.entries(obj.weight_map);
            if (entries.length > 0 && entries.every(([, value]) => typeof value === 'string' && value.endsWith('.bin'))) {
                return new pytorch.Container.Index(context, entries);
            }
        }
        return null;
    }

    constructor(context, entries) {
        super();
        this.type = 'pytorch.index';
        this.context = context;
        this._entries = entries;
    }

    async read(metadata) {
        this.format = 'PyTorch';
        const weight_map = new Map(this._entries);
        const keys = new Set(weight_map.keys());
        const files = Array.from(new Set(weight_map.values()));
        const contexts = await Promise.all(files.map((name) => this.context.fetch(name)));
        const execution = new pytorch.jit.Execution(null, metadata);
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const archives = contexts.map((context) => {
            return context.peek('zip');
        });
        const formats = new Set(archives.map((entries) => {
            const reader = new torch.PyTorchFileReader(entries);
            const version = reader.version();
            return pytorch.Utility.format('PyTorch', version);
        }));
        if (formats.size === 1) {
            this.format = formats.values().next().value;
        }
        const shards = archives.map((entries) => {
            return torch.load(entries);
        });
        const entries = new Map();
        for (const shard of shards) {
            for (const [key, value] of Array.from(shard)) {
                if (keys.has(key)) {
                    entries.set(key, value);
                }
            }
        }
        this.module = entries;
        delete this.context;
        delete this._entries;
    }
};

pytorch.Container.ExportedProgram = class extends pytorch.Container {

    static open(context) {
        const program = context.peek('json');
        if (program && program.schema_version && program.graph_module) {
            return new pytorch.Container.ExportedProgram(context, program);
        }
        return null;
    }

    constructor(context, serialized_exported_program) {
        super();
        this.type = 'pytorch.export';
        this.context = context;
        this.serialized_exported_program = serialized_exported_program;
    }

    async read() {
        this.format = 'PyTorch Export';
        const serialized_state_dict = await this._fetch('serialized_state_dict.pt') || await this._fetch('serialized_state_dict.json');
        const serialized_constants = await this._fetch('serialized_constants.pt') || await this._fetch('serialized_constants.json');
        const f = new Map();
        f.set('serialized_exported_program.json', this.serialized_exported_program);
        f.set('serialized_state_dict.pt', serialized_state_dict);
        f.set('serialized_constants.pt', serialized_constants);
        const execution = new pytorch.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        if (this.serialized_exported_program.graph_module.graph.constants) {
            const zip = await import('./zip.js');
            const constants = this.serialized_exported_program.graph_module.graph.constants;
            for (const key of Object.keys(constants)) {
                const value = constants[key];
                const str = atob(value);
                const buffer = new Uint8Array(str.length);
                for (let i = 0; i < str.length; i++) {
                    buffer[i] = str.charCodeAt(i);
                }
                const archive = zip.Archive.open(buffer);
                constants[key] = archive.entries;
            }
        }
        delete this.serialized_exported_program;
        delete this.context;
        /* const exported_program = */ torch._export.load(f);
        throw new pytorch.Error(`'torch.export' not supported.`);
    }

    async _fetch(name) {
        try {
            const context = await this._context.fetch(name);
            if (context) {
                return context.peek('zip');
            }
        } catch {
            // continue regardless of error
        }
        return null;
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
            load_pickle(module, resource) {
                const name = `${module.replace(/\./, '/')}/${resource}`;
                const stream = this.zip_reader.get_record(name);
                const loaded_reduces = new Map();
                this.storage_context = new torch._C.DeserializationStorageContext();
                const unpickler = new pickle.Unpickler(stream);
                unpickler.persistent_load = (saved_id) => {
                    switch (saved_id[0]) {
                        case 'storage': {
                            const [, storage_type, key, , size] = saved_id;
                            if (!this.storage_context.has_storage(key)) {
                                const storage = new storage_type(size);
                                const stream = this.zip_reader.get_record(`.data/${key}.storage`);
                                const buffer = stream.peek();
                                storage._set_cdata(buffer);
                                this.storage_context.add_storage(key, storage);
                            }
                            return this.storage_context.get_storage(key);
                        }
                        case 'reduce_package': {
                            if (saved_id.length === 2) {
                                const [, func, args] = saved_id;
                                return execution.invoke(func, args);
                            }
                            const [, reduce_id, func, args] = saved_id;
                            if (!loaded_reduces.has(reduce_id)) {
                                const value = execution.invoke(func, [this].concat(args));
                                loaded_reduces.set(reduce_id, value);
                            }
                            return loaded_reduces.get(reduce_id);
                        }
                        default: {
                            throw new pytorch.Error(`Unknown package typename '${saved_id[0]}'.`);
                        }
                    }
                };
                const obj = unpickler.load();
                this.storage_context = null;
                return obj;
            }
            import_module(name) {
                return execution.import(name);
            }
        });
        this.registerFunction('torch.jit.load', (file, map_location, extra_files) => {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const cpp_module = torch._C.import_ir_module(cu, file, map_location, extra_files);
            return new torch.jit._script.RecursiveScriptModule(cpp_module);
        });
        this.registerFunction('torch._C.import_ir_module', function(cu, reader, ...args) {
            switch (arguments.length) {
                case 4: {
                    const [device, extra_files] = args;
                    const deserializer = new pytorch.jit.ScriptModuleDeserializer(cu, reader);
                    return deserializer.deserialize(device, extra_files);
                }
                case 5: {
                    const [storage_context, device, ts_id] = args;
                    const deserializer = new pytorch.jit.ScriptModuleDeserializer(cu, reader, `.data/ts_code/${ts_id}/`, '.data/', storage_context);
                    return deserializer.deserialize(device, null);
                }
                default: {
                    throw new pytorch.Error("Invalid 'torch._C.import_ir_module' signature.");
                }
            }

        });
        this.registerFunction('torch._C._import_ir_module_from_package', (cu, reader, storage_context, map_location, ts_id) => {
            return torch._C.import_ir_module(cu, reader, storage_context, null, ts_id);
        });
        this.registerFunction('torch._C._jit_pass_inline', (graph) => {
            const tryToGraphFunction = (node) => {
                if (node.kind() === 'prim::CallFunction') {
                    //
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
        this.registerFunction('torch.jit._script.unpackage_script_module', (importer, script_module_id) => {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const cpp_module = torch._C._import_ir_module_from_package(cu, importer.zip_reader, importer.storage_context, importer.last_map_location, script_module_id);
            return new torch.jit._script.RecursiveScriptModule(cpp_module);
        });
        this.registerFunction('torch.jit.jit_module_from_flatbuffer', (f) => {
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
        this.registerFunction('torch.jit._script.wrap_cpp_module', (cpp_module) => {
            const init_fn = (script_module) => {
                for (const [name, module] of new torch.ModuleDict(script_module._c).items()) {
                    script_module.__setattr__(name, torch.jit._script.wrap_cpp_module(module));
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
            }
            addAttribute(/* name */) {
            }
            hasAttribute(/* name */) {
            }
            hasConstant(/* name */) {
            }
            methods() {
            }
        });
        this.registerType('torch.TupleType', class extends torch.Type {});
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
                    if (name === method.name) {
                        return method;
                    }
                }
                return null;
            }
            _has_method(/* name */) {
                throw new pytorch.Error();
            }
            __setattr__(name, value) {
                // if (this._type.hasContant(name))
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
            get qualified_name() {
                return this._type.qualified_name();
            }
            get code_with_constants() {
                const const_map = {};
                const_map.const_mapping = new Map(Object.entries(execution.builtins.CONSTANTS));
                return [null, const_map];
            }
            get graph() {
                if (!this._graph) {
                    if (!this.data) {
                        return null;
                    }
                    if (!this.data.forward) {
                        throw new pytorch.Error("Module 'forward' not implemented.");
                    }
                    const args = [this.data]; // self
                    if (this.data.forward.__code__ && this.data.forward.__code__.args) {
                        for (const arg of this.data.forward.__code__.args) {
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
                                            return type.arguments.map((type, index) => defaultValue(type, `${name}[${index}]`));
                                        }
                                        case 'List': {
                                            return type.arguments.map((type, index) => defaultValue(type, `${name}[${index}]`));
                                        }
                                        case 'Dict': {
                                            if (type.arguments[1].name.value === 'Tensor') {
                                                const Dict = class extends Map {
                                                    get(key) {
                                                        if (!super.has(key)) {
                                                            super.set(key, defaultValue(type.arguments[1], `${name}:${key}`));
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
                                throw new pytorch.Error(`Unsupported parameter type '${JSON.stringify(type)}'.`);
                            };
                            if (arg.name !== 'self') {
                                const type = arg.parameterType;
                                const value = defaultValue(type, arg.name);
                                if (pytorch.Utility.isTensor(value)) {
                                    value.__variable__ = arg.name;
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
                this._items = Object.entries(module).filter(([, value]) => value instanceof torch.ScriptModule);
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
                    const qualified_name = prefix ? `${prefix}.${name}` : name;
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
        this.registerType('torch.jit._script.ScriptModule', class extends torch.nn.modules.module.Module {});
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
                } else if (this.modules.has(name)) {
                    this.modules.set(name, value);
                } else if (this._c.hasattr(name)) {
                    this._c.setattr(name, value);
                } else {
                    //
                }
            }
            __getattr__(name) {
                if (this._initializing) {
                    return super.__getattr__(name);
                }
                if (this.modules.has(name)) {
                    return this.modules.get(name);
                }
                if (this._c.hasattr(name)) {
                    return this._c.getattr(name);
                }
                if (this._c._has_method(name)) {
                    //
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
                if (state[0] !== '2') {
                    throw new pytorch.Error(`Unsupported pack version '${state[0]}'.`);
                }
                const [/* pack_version */, tensors, opt_tensors] = state;
                const packed_config_tensor = new pytorch.Tensor('', tensors[0], true);
                const packed_config = packed_config_tensor.decode();
                /* eslint-disable prefer-destructuring */
                this.weight = tensors[1];
                this.bias = opt_tensors[0];
                this.stride = [packed_config[1], packed_config[2]];
                this.padding = [packed_config[3], packed_config[4]];
                this.dilation = [packed_config[5], packed_config[6]];
                this.output_padding = [packed_config[7], packed_config[8]];
                this.groups = packed_config[9];
                /* eslint-enable prefer-destructuring */
            }
        });
        this.registerType('__torch__.torch.classes.quantized.Conv3dPackedParamsBase', class {
            __setstate__(state) {
                if (state[0] !== '2') {
                    throw new pytorch.Error(`Unsupported pack version '${state[0]}'.`);
                }
                const [/* pack_version */, tensors, opt_tensors] = state;
                const packed_config_tensor = new pytorch.Tensor('', tensors[0], true);
                const packed_config = packed_config_tensor.decode();
                /* eslint-disable prefer-destructuring */
                this.weight = tensors[1];
                this.bias = opt_tensors[0];
                this.stride = [packed_config[1], packed_config[2]];
                this.padding = [packed_config[3], packed_config[4]];
                this.dilation = [packed_config[5], packed_config[6]];
                this.output_padding = [packed_config[7], packed_config[8]];
                this.groups = packed_config[9];
                /* eslint-enable prefer-destructuring */
            }
        });
        this.registerType('__torch__.torch.classes.quantized.LinearPackedParamsBase', class {
            __setstate__(state) {
                [this.weight, this.bias] = state;
            }
        });
        this.registerType('__torch__.torch.classes.rnn.CellParamsBase', class {
            __setstate__(state) {
                [this.type, this.tensors, this.doubles, this.longs, this.packed_params] = state;
            }
        });
        this.registerType('__torch__.torch.classes.xnnpack.Conv2dOpContext', class {
            __setstate__(state) {
                [this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups, this.output_min, this.output_max] = state;
            }
        });
        this.registerType('__torch__.torch.classes.xnnpack.LinearOpContext', class {
            __setstate__(state) {
                [this.weight, this.bias, this.output_min, this.output_max] = state;
            }
        });
        this.registerType('__torch__.torch.classes.xnnpack.TransposeConv2dOpContext', class {
            __setstate__(state) {
                [this.weight, this.bias, this.stride, this.padding, this.output_padding, this.dilation, this.groups, this.output_min, this.output_max] = state;
            }
        });
        this.registerType('torch.Graph', class {
            constructor() {
                this._unique = 1;
                this._nodes = [];
                this._block = execution.invoke('torch.Block', [this]);
            }
            create(kind) {
                return execution.invoke('torch.Node', [this, kind]);
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
                const use = execution.invoke('torch.Use', [this]);
                value.uses().push(use);
                this._inputs.push(value);
                return value;
            }
            addOutput() {
                const value = execution.invoke('torch.Value', [this]);
                this._outputs.push(value);
                return value;
            }
            addBlock() {
                const block = execution.invoke('torch.Block', [this._graph, this]);
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
        for (const [, value] of this._metadata._types) {
            const name = value.name;
            if (name.indexOf('::') !== -1) {
                const index = name.lastIndexOf('.');
                const key = index === -1 ? name : name.substring(0, index);
                if (!this._types.has(key)) {
                    this._types.set(key, []);
                }
                this._types.get(key).push(value);
            }
        }
        this._graph = this.invoke('torch.Graph', []);
        this._values = new Map();
    }

    debug(file) {
        const buffer = this.source(`${file}.debug_pkl`);
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
        const value = node ? node.addOutput() : this.invoke('torch.Value', [node ? node : this._graph]);
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
                throw new pytorch.Error(`Unknown type name '${name}'.`);
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
                const file = `${path.join('/')}.py`;
                if (this.source(file)) {
                    return this.import(name);
                }
                return this.resolve(name);
            }
        }
        return super.target(expression, context);
    }

    call(target, name, args, context) {
        if (this.trace) {
            const overload = this._overload(target, name, args, context);
            if (overload) {
                const [schema, args, evalArgs] = overload;
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
                        const map = new Map(parameters.map((parameter) => [parameter.name, parameter]));
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
                    const [argument] = copyEvalArgs;
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
                        const [argument] = copyEvalArgs;
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
                        const [arg] = copyArgs;
                        if (!pytorch.Utility.isType(argument, parameter.type) && argument !== null) {
                            if (parameter.optional) {
                                continue;
                            }
                            throw new pytorch.Error();
                        } else if (arg.type === '=') {
                            throw new pytorch.Error('Expected named argument.');
                        } else {
                            copyArgs.shift();
                            copyEvalArgs.shift();
                            const value = this.variable(argument);
                            node.addInput(value);
                            value.value = argument;
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
                                        output.resize_([NaN, NaN, NaN]);
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
                                        const [input] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && input.size() === undefined) {
                                            input.resize_([NaN, NaN, NaN, NaN]);
                                        }
                                        output.resize_([NaN, NaN, NaN, NaN]);
                                        break;
                                    }
                                    case 'aten::slice':
                                    case 'aten::slice.Tensor': {
                                        const [input] = evalArgs;
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
                                        const [input] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                            const size = input.size();
                                            output.resize_(size);
                                        }
                                        break;
                                    }
                                    case 'aten::conv3d': {
                                        output.resize_([NaN, NaN, NaN, NaN, NaN]);
                                        break;
                                    }
                                    case 'aten::roll':
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
                                        const [input] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                            output.resize_(input.size());
                                        }
                                        break;
                                    }
                                    case 'aten::add':
                                    case 'aten::add.Scalar':
                                    case 'aten::sub':
                                    case 'aten::sub.Scalar': {
                                        const [input] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                            output.resize_(input.size());
                                        } else {
                                            const [, other] = evalArgs;
                                            if (pytorch.Utility.isTensor(other) && Array.isArray(other.size())) {
                                                output.resize_(other.size());
                                            }
                                        }
                                        break;
                                    }
                                    case 'aten::select':
                                    case 'aten::select.int': {
                                        const [input] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                            output.resize_(Array(input.size().length - 1).fill(NaN));
                                        }
                                        break;
                                    }
                                    case 'aten::layer_norm': {
                                        const [input, normalized_shape] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                            const shape = input.size();
                                            if (Array.isArray(normalized_shape) && normalized_shape.length === 1) {
                                                const [value] = normalized_shape;
                                                shape[shape.length - 1] = value;
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
                                        const [input] = evalArgs;
                                        const size = input.size();
                                        if (Array.isArray(size)) {
                                            switch (evalArgs.length) {
                                                case 1: {
                                                    output.resize_(size.filter((value) => value !== 1));
                                                    break;
                                                }
                                                case 2: {
                                                    const [, dim] = evalArgs;
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
                                        const [input, dim] = evalArgs;
                                        const size = input.size();
                                        if (Array.isArray(size) && dim !== undefined) {
                                            const shape = size.slice();
                                            shape.splice(dim, 0, 1);
                                            output.resize_(shape);
                                        } else {
                                            output.resize_([NaN, NaN, NaN, NaN]);
                                        }
                                        break;
                                    }
                                    case 'aten::transpose':
                                    case 'aten::transpose.int': {
                                        const [input, dim0, dim1] = evalArgs;
                                        if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                            const size = input.size().slice();
                                            const d0 = dim0 >= 0 ? dim0 : size.length + dim0;
                                            const d1 = dim1 >= 0 ? dim1 : size.length + dim1;
                                            const value = size[dim0];
                                            /* eslint-disable prefer-destructuring */
                                            size[d0] = size[1];
                                            /* eslint-enable prefer-destructuring */
                                            size[d1] = value;
                                            output.resize_(size);
                                        }
                                        break;
                                    }
                                    case 'aten::contiguous': {
                                        const [source] = evalArgs;
                                        output.__source__ = source;
                                        break;
                                    }
                                    case 'quantized::cat':
                                    case 'quantized::cat_relu':
                                    case 'quantized::linear':
                                    case 'quantized::conv2d':
                                    case 'quantized::conv2d.new':
                                    case 'quantized::conv2d_relu':
                                    case 'quantized::conv2d_relu.new':
                                    case 'quantized::add':
                                    case 'quantized::add_relu':
                                        output.resize_([NaN, NaN, NaN, NaN]);
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
                        case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                        case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                        case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                        case '__torch__.torch.classes.rnn.CellParamsBase':
                        case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                        case '__torch__.torch.classes.xnnpack.LinearOpContext':
                        case '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext': {
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
        }
        return super.call(target, name, args, context);
    }

    _overload(target, name, args, context) {
        let moduleName = pytorch.Utility.target(target);
        if (moduleName) {
            let outputTypes = null;
            let type = name ? `${moduleName}.${name}` : moduleName;
            if (type === 'ops.prim.NumToTensor' && args.length === 1 && args[0].type === 'call' && args[0].target.member.type === 'id') {
                const [arg] = args;
                moduleName = pytorch.Utility.target(arg.target.target);
                name = arg.target.member.value;
                args = arg.args;
                outputTypes = ['int64'];
                type = `${moduleName}.${name}`;
            }
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
            let overloads = null;
            if (type.startsWith('torch.')) {
                overloads = this._types.get(`aten::${type.substring(6)}`);
            /* } else if (type.startsWith('ops.prim.')) {
                overloads = this._types.get(`prim::${type.substring(9)}`);
            } else if (type === 'int') {
                overloads = this._types.get(`aten::Int`);
                // "bool": "aten::Bool"
                // "int": "aten::Int"
                // "float": "aten::Float"
                // "complex": "aten::Complex"
                // "abs": "prim::abs"
                // "max": "prim::max"
                // "min": "prim::min"
                // "range": "fake::does_not_exist"
            */
            } else if (type.startsWith('ops.') && !type.startsWith('ops.prim.')) {
                const path = type.split('.');
                if (path.length === 3) {
                    overloads = this._types.get(`${path[1]}::${path[2]}`);
                }
                if (!overloads) {
                    const module = this.import(moduleName);
                    if (!module || !module[name]) {
                        const metadata = {};
                        metadata.name = type;
                        metadata.inputs = [];
                        metadata.outputs = [];
                        for (let i = 0; i < args.length; i++) {
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
                        overloads = [metadata];
                    }
                }
            }
            if (overloads) {
                overloads = Array.isArray(overloads) ? overloads : [overloads];
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
                            const map = new Map(parameters.map((parameter) => [parameter.name, parameter]));
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
                        const [argument] = copyEvalArgs;
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
                            const [argument] = copyEvalArgs;
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
                            const [arg] = copyArgs;
                            if (!pytorch.Utility.isType(argument, parameter.type) && argument !== null) {
                                if (parameter.optional) {
                                    continue;
                                }
                                next = true;
                            } else if (arg.type === '=') {
                                throw new pytorch.Error('Expected named argument.');
                            } else {
                                copyArgs.shift();
                                copyEvalArgs.shift();
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
                            // case 'int64':
                            //     break;
                            case '__torch__.torch.classes.xnnpack.LinearOpContext':
                            case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                            case '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext':
                            case '__torch__.torch.classes.rnn.CellParamsBase':
                            case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                            case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                            case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
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
                    return [schema, args, evalArgs];
                }
            }
        }
        return null;
    }

    block(statements, context) {
        statements = Array.prototype.slice.call(statements);
        while (statements.length > 0) {
            if (statements.length > 1) {
                const [assign, condition] = statements;
                // _x = torch.ne(torch.len(torch.size(input)), 5)
                // if _x:
                //   ops.prim.RaiseException(...)
                if (assign.type === '=' &&
                    condition.type === 'if' &&
                    pytorch.Utility.isEqual(assign.target, condition.test) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.ne', 2) &&
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.args[0].args[0], 'torch.size', 1) &&
                    condition.body.statements.length === 1 &&
                    pytorch.Utility.isCall(condition.body.statements[0], 'ops.prim.RaiseException', 1)) {
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
                    pytorch.Utility.isEqual(assign.target, condition.test) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.ne', 2) &&
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.dim', 1) &&
                    condition.body.statements.length > 0 &&
                    pytorch.Utility.isCall(condition.body.statements[condition.body.statements.length - 1], 'ops.prim.RaiseException', 1)) {
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
                    pytorch.Utility.isEqual(assign.target, condition.test) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.eq', 2) &&
                    pytorch.Utility.isCall(assign.expression.args[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.args[0].args[0], 'torch.size', 1) &&
                    condition.orelse.statements.length === 1 &&
                    pytorch.Utility.isCall(condition.orelse.statements[0], 'ops.prim.RaiseException', 1)) {
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
                    pytorch.Utility.isCall(condition.test, 'torch.eq', 2) &&
                    pytorch.Utility.isCall(condition.test.args[0], 'torch.len', 1) &&
                    pytorch.Utility.isEqual(condition.test.args[0].args[0], assign.target) &&
                    condition.orelse.statements.length === 1 &&
                    pytorch.Utility.isCall(condition.orelse.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.args[0].args[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.shape === undefined) {
                        const start = this.expression(assign.expression.args[1], context);
                        const value = this.expression(condition.test.args[1], context);
                        if (Number.isInteger(start) && start < 0 && Number.isInteger(value) && value > 0) {
                            tensor.resize_(Array(value - start).fill(NaN));
                        }
                    }
                }
            }
            if (statements.length > 1) {
                // getattr_1 = torch.size(x)
                // getitem = torch.slice(getattr_1, -2, 9223372036854775807, 1)
                const [size, statement] = statements;
                if (size.type === '=' && statement.type === '=' &&
                    size.target.type === 'id' &&
                    pytorch.Utility.isCall(size.expression, 'torch.size', 1) &&
                    pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                    statement.expression.arguments[0].type === 'id' && size.target.value === statement.expression.arguments[0].value) {
                    const tensor = this.expression(size.expression.arguments[0], context);
                    if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                        tensor.resize_([1, 3, 299, 299]);
                    }
                }
            }
            if (statements.length > 1) {
                // _0 = torch.split_with_sizes(...)
                // a, a_1, a_2, = _0
                const [statement, tuple] = statements;
                if (statement.type === '=' && statement.target.type === 'id' && statement.expression.type === 'call' &&
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
                                for (const [key, value] of Object.entries(obj)) {
                                    if (key !== 'identifier') {
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
                        }
                        return false;
                    };
                    if (!containsVariableReference(statements.slice(2, statements.length - 1), statement.target.value)) {
                        statements[0] = { ...statement };
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
                    tensor.resize_([1, 3, 299, 299]);
                }
            }
            // torch.slice(ops.prim.shape(input), 0, 2, 1)
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                pytorch.Utility.isCall(statement.expression.args[0], 'ops.prim.shape', 1)) {
                const tensor = this.expression(statement.expression.args[0].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([NaN, NaN, NaN, NaN]);
                }
            }
            // _3 = torch.le(xxxx, torch.dim(f0))
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.le', 2) &&
                pytorch.Utility.isCall(statement.expression.args[1], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.args[1].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([NaN, NaN, NaN, NaN]);
                }
            }
            // if torch.ne(torch.dim(image), 3):
            //   xxxx
            //   ops.prim.RaiseException(_7)
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.test, 'torch.ne', 2) &&
                pytorch.Utility.isCall(statement.test.args[0], 'torch.dim', 1) &&
                statement.body.statements.length > 0 &&
                pytorch.Utility.isCall(statement.body.statements.slice(-1).pop(), 'ops.prim.RaiseException', 1)) {
                const tensor = this.expression(statement.test.args[0].args[0], context);
                const size = this.expression(statement.test.args[1], context);
                if (pytorch.Utility.isTensor(tensor) && Number.isInteger(size) && size < 10) {
                    tensor.resize_(Array.isArray(tensor.shape) && tensor.shape.length > size ? tensor.shape.slice(-size) : Array(size).fill(NaN));
                }
            }
            // if torch.gt(torch.dim(x), 1):
            //   xxxx
            //   ops.prim.RaiseException(...)
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.test, 'torch.gt', 2) &&
                pytorch.Utility.isCall(statement.test.args[0], 'torch.dim', 1) &&
                statement.body.statements.length > 0 &&
                pytorch.Utility.isCall(statement.body.statements.slice(-1).pop(), 'ops.prim.RaiseException')) {
                const tensor = this.expression(statement.test.args[0].args[0], context);
                const size = this.expression(statement.test.args[1], context);
                if (pytorch.Utility.isTensor(tensor) && Number.isInteger(size) && size < 10) {
                    tensor.resize_(Array.isArray(tensor.shape) && tensor.shape.length > size ? tensor.shape.slice(-size) : Array(size).fill(NaN));
                }
            }
            // if bool(...):
            //   ops.prim.RaiseException(torch.format(_1, dtype))
            // else:
            //   pass
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.test, 'bool', 1) &&
                statement.body.statements.length > 0 &&
                pytorch.Utility.isCall(statement.body.statements.slice(-1).pop(), 'ops.prim.RaiseException', 1)) {
                statement.test = { type: 'id', value: 'False' };
            }
            // dim = torch.sub(torch.dim(input), 2)
            if (statement.type === '=' &&
                statement.target.type === 'id' && statement.target.value === 'dim' &&
                pytorch.Utility.isCall(statement.expression, 'torch.sub', 2) &&
                pytorch.Utility.isCall(statement.expression.args[0], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.args[0].args[0], context);
                if (pytorch.Utility.isTensor(tensor) && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([NaN, NaN, NaN, NaN]);
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
                    tensor.resize_([NaN, NaN, NaN, NaN]);
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
        const path = `${this._code_prefix}/${qualifier}.py`;
        if (this._reader.has_record(path)) {
            const data = this._reader.get_record(path);
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
        //
    }

    resolveType(name) {
        return this.findNamedType(new pytorch.jit.QualifiedName(name));
    }

    findNamedType(name) {
        this.parseSourceIfNeeded(name.prefix());
    }

    parseSourceIfNeeded(/* qualifier */) {
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
            new pytorch.jit.SourceLoader(this._reader, this._code_prefix), reader.version());
    }

    deserialize() {
        const execution = this._compilation_unit.execution;
        const code_prefix = this._code_prefix;
        for (const name of this._reader.get_all_records()) {
            if (name.startsWith(code_prefix) && name.endsWith('.py')) {
                const file = name.substring(code_prefix.length);
                const stream = this._reader.get_record(name);
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
        if (this._reader.has_record('model.json')) {
            return this.LEGACY_deserialize();
        }
        const constants = this.readArchive('constants');
        for (let i = 0; i < constants.length; i++) {
            execution.builtins.CONSTANTS[`c${i}`] = constants[i];
        }
        const module = this.readArchive('data');
        const type = new torch.ClassType(`${module.__class__.__module__}.${module.__class__.__name__}`, null, true);
        const result = new torch.ScriptModule(type);
        result.data = module;
        return result;
    }

    LEGACY_deserialize() {
        const execution = this._compilation_unit.execution;
        const torch = execution.import('torch');
        const stream = this._reader.get_record('model.json');
        const buffer = stream.peek();
        const decoder = new TextDecoder('utf-8');
        const content = decoder.decode(buffer);
        const model = JSON.parse(content);
        const data = model.mainModule || {};
        const queue = [data];
        const tensorTypeMap = new Map([
            ['FLOAT', 'Float'],
            ['FLOAT16', 'Half'],
            ['DOUBLE', 'Double'],
            ['INT8', 'Char'],
            ['INT32', 'Int'],
            ['INT64', 'Long']
        ]);
        const constants = (model.tensors || []).map((constant) => {
            const key = constant.data.key;
            if (!tensorTypeMap.has(constant.dataType)) {
                throw new pytorch.Error(`Unsupported tensor data type '${constant.dataType}'.`);
            }
            const type = tensorTypeMap.get(constant.dataType);
            const shape = constant.dims ? constant.dims.map((dim) => parseInt(dim, 10)) : null;
            const strides = constant.strides ? constant.strides.map((dim) => parseInt(dim, 10)) : null;
            const storage_type = execution.resolve(`torch.${type}Storage`);
            const size = (shape || []).reduce((a, b) => a * b, 1);
            const offset = parseInt(constant.offset, 10) || 0;
            const storage = new storage_type(size);
            const itemsize = storage.dtype.itemsize();
            const stream = this._reader.get_record(key);
            if (stream) {
                const buffer = stream.peek();
                const length = size * itemsize;
                const data = buffer.slice(offset, offset + length);
                storage._set_cdata(data);
            }
            const tensor = execution.invoke('torch._utils._rebuild_tensor', [storage, 0, shape, strides]);
            tensor.name = constant.data.key;
            return tensor;
        });
        execution.builtins.CONSTANTS = {};
        for (let i = 0; i < constants.length; i++) {
            execution.builtins.CONSTANTS[`c${i}`] = constants[i];
        }
        const attributes = [];
        if (this._reader.has_record('attributes.pkl')) {
            const stream = this._reader.get_record('attributes.pkl');
            const buffer = stream.peek();
            const unpickler = execution.invoke('pickle.Unpickler', [buffer]);
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
            if (!this._reader.has_record(arena.key)) {
                throw new pytorch.Error(`File '${arena.key}' not found.`);
            }
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
        const picklename = `${pickle_prefix + archive_name}.pkl`;
        const stream = stream_reader.get_record(picklename);
        if (!stream) {
            throw new pytorch.Error(`File '${picklename}' is not found.`);
        }
        const buffer = stream.peek();
        const tensor_dir_path = tensor_prefix ? tensor_prefix : `${archive_name}/`;
        const read_record = (name) => {
            const stream = stream_reader.get_record(tensor_dir_path + name);
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
            if (saved_id[0] !== 'storage') {
                throw new pytorch.Error(`Unsupported persistent load type '${saved_id[0]}'.`);
            }
            const [, storage_type, key, , size] = saved_id;
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
        this._dtypes = new Map(dtypes.map((dtype) => [dtype.scalar_type(), dtype]));
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
        for (const [name, value] of this._all_functions) {
            const class_index = module.ivalues[name].val.class_type;
            const class_type = this._all_types[class_index];
            class_type.addMethod(value);
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
        const storage = this._cu.execution.invoke('torch.storage.TypedStorage', [size, dtype]);
        storage._set_cdata(data);
        const tensor = this._cu.execution.invoke('torch.Tensor', []);
        const shape = Array.from(metadata.sizes);
        const stride = Array.from(metadata.strides);
        tensor.__setstate__([storage, metadata.storage_offset, shape, stride]);
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
                throw new pytorch.Error(`Unknown object type type '${obj_type.type}'.`);
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
                // cls = c10::parseType(qn_str)->cast<ClassType>();
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

pytorch.Container.Package = class extends pytorch.Container {

    constructor(entries) {
        super();
        this.type = 'pytorch.package';
        this.entries = entries;
    }

    async read() {
        const execution = new pytorch.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this.entries);
        const version = reader.version();
        this.format = pytorch.Utility.format('PyTorch Package', version);
        this.modules = new Map();
        const records = reader.get_all_records().filter((name) => {
            if (!name.startsWith('.data/') && !name.endsWith('.py')) {
                const stream = reader.get_record(name);
                if (stream && stream.length > 2) {
                    const signature = stream.peek(2);
                    if (signature[0] === 0x80 && signature[1] < 7) {
                        return true;
                    }
                }
            }
            return false;
        });
        const entries = records.map((name) => {
            const parts = name.split('/');
            const resource = parts.pop();
            const module = parts.join('.');
            return [module, resource];
        });
        if (entries.length > 0) {
            for (const name of reader.get_all_records()) {
                if (!name.startsWith('.data/') && name.endsWith('.py')) {
                    const stream = reader.get_record(name);
                    const buffer = stream.peek();
                    execution.add(name, buffer);
                }
            }
            const importer = new torch.package.PackageImporter(reader);
            for (const entry of entries) {
                const module = importer.load_pickle(entry[0], entry[1]);
                const key = `${entry[0].replace(/\./, '/')}/${entry[1]}`;
                this.modules.set(key, module);
            }
        }
        delete this.entries;
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
        if (expression.type === 'id') {
            return expression.value;
        }
        if (expression.type === '.') {
            return `${pytorch.Utility.target(expression.target)}.${pytorch.Utility.target(expression.member)}`;
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

    static isObjectType(type) {
        switch (type) {
            case '__torch__.torch.classes.xnnpack.LinearOpContext':
            case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
            case '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext':
            case '__torch__.torch.classes.rnn.CellParamsBase':
            case '__torch__.torch.classes.rnn.CellParamsBase[]':
            case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
            case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
            case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                return true;
            default:
                return false;
        }
    }

    static isObject(obj) {
        const type = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : null;
        return pytorch.Utility.isObjectType(type);
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
        throw new pytorch.Error(`Unsupported ops argument type '${text}'.`);
    }

    static isType(obj, type) {
        switch (type) {
            case 'Tensor':
                return !Array.isArray(obj) && (pytorch.Utility.isTensor(obj) || obj === null);
            case 'Tensor[]':
                return Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => pytorch.Utility.isTensor(tensor) || tensor === null);
            case 'Scalar':
                return (obj !== null && (obj !== Object(obj) || obj instanceof Number)) || (pytorch.Utility.isTensor(obj) && Array.isArray(obj.size()) && obj.size().length === 0);
            case 'boolean':
                return obj === true || obj === false;
            case 'string':
                return obj === null || typeof obj === 'string';
            case 'SymInt':
            case 'int64':
                return Number.isInteger(obj) || typeof obj === 'bigint' ||
                    (typeof obj === 'number' && isNaN(obj)) || (obj instanceof Number);
            case 'SymInt[]':
            case 'SymInt[2]':
            case 'SymInt[3]':
            case 'SymInt[4]':
            case 'SymInt[5]':
            case 'SymInt[6]':
                return Array.isArray(obj) && obj.every((item) => pytorch.Utility.isType(item, 'SymInt') || item === undefined || (item.__class__ === 'number' && isNaN(item)));
            case 'int64[]':
            case 'int64[2]':
            case 'int64[3]':
                return Array.isArray(obj) && obj.every((item) => pytorch.Utility.isType(item, 'int64') || item === undefined || (item.__class__ === 'number' && isNaN(item)));
            case 'int64[1]':
            case 'SymInt[1]':
                return pytorch.Utility.isType(obj, 'int64') || pytorch.Utility.isType(obj, 'int64[]');
            case 'float32':
            case 'float64':
                return obj !== null && (typeof obj === 'number' || obj instanceof Number);
            case 'float32[]':
                return Array.isArray(obj) && obj.every((item) => (typeof item === 'number' || item instanceof Number) && !isNaN(item));
            case 'string[][]':
                return Array.isArray(obj) && obj.every((item) => Array.isArray(item) && item.every((item) => typeof item === 'string'));
            case 'Layout':
            case 'ScalarType':
            case 'MemoryFormat':
                return Number.isInteger(obj) || obj === null;
            case 'Dimname':
                return obj === null || (typeof obj === 'string' || obj instanceof String);
            case 'Dimname[]':
                return Array.isArray(obj) && obj.every((item) => item === null || typeof item === 'string');
            case 'Device':
                return obj === null || obj === Object(obj);
            default:
                if (type && type.startsWith('__torch__.') &&
                    obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                    return type === `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                }
                return true;
        }
    }

    static isSubclass(value, name) {
        if (value && value.__module__ && value.__name__) {
            return name === `${value.__module__}.${value.__name__}`;
        } else if (value && value.__bases__) {
            return value.__bases__.some((obj) => pytorch.Utility.isSubclass(obj, name));
        }
        return false;
    }

    static isInstance(value, name) {
        return value && value.__class__ ? pytorch.Utility.isSubclass(value.__class__, name) : false;
    }

    static isCall(expression, name, size) {
        if (expression.type === 'call' &&
            (size === undefined || size === expression.args.length) &&
            pytorch.Utility.target(expression.target) === name) {
            return true;
        }
        return false;
    }

    static isEqual(a, b) {
        return (a.type === 'id' && b.type === 'id' && a.value === b.value);
    }

    static format(name, value) {
        // https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/inline_container.h
        // kProducedFileFormatVersion
        const versions = new Map([
            ['1', 'v1.3'],
            ['2', 'v1.5'], // 7a2889b014ce36fcc333b2c6de6f29f976652f84 (#28122)
            ['3', 'v1.6'], // 2ec6a30722b0ef85632a2f3e7ce6f80da403008a (#36085)
            ['4', 'v1.6'], // 95489b590f00801bdee7f41783f30874883cf6bb (#38620)
            ['5', 'v1.7'], // cb26661fe4faf26386703180a9045e6ac6d157df (#40364)
            ['6', 'v1.9'], // 3ee7637ffa50df0d9b231c7b40778ac1c390bf4a (#59714)
            ['7', 'v1.10'], // 880098a7e34a20628f960daa8eab0eb1ad566c39 (#63651)
            ['8', 'v1.11'], // b28e696516a7f0c7a6ead6da967590ce6c1d6698 (#71486)
            ['9', 'v1.11'], // 8757e21c6a4fc00e83539aa7f9c28eb11eff53c1 (#72051)
            ['10', 'v1.12']  // 4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8 (#61439)
        ]);
        if (!versions.has(value)) {
            throw new pytorch.Error(`Unsupported '${name}' version '${value}'.`);
        }
        return `${name} ${versions.get(value)}`;
    }

    static weights(obj) {
        const type = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : null;
        if (type && type !== 'builtins.dict' && type !== 'builtins.object' && type !== 'collections.OrderedDict' && type !== 'torch.nn.modules.module.Module') {
            return null;
        }
        if (pytorch.Utility.isTensor(obj)) {
            return null;
        }
        if (obj instanceof Map === false && obj && !Array.isArray(obj) && Object(obj) === obj) {
            const entries = Object.entries(obj);
            const named = entries.filter(([name, value]) => (typeof name === 'string' && (name.indexOf('.') !== -1 || name.indexOf('|') !== -1)) && pytorch.Utility.isTensor(value));
            if (named.length > 0 && (named.length / entries.length) >= 0.8) {
                obj = new Map(entries);
            }
        }
        if (obj instanceof Map) {
            const entries = Array.from(obj).filter(([name]) => name !== '_metadata');
            const names = entries.filter(([name]) => typeof name === 'string' && (name.indexOf('.') !== -1 || name.indexOf('|') !== -1));
            if (names.length > 1 &&
                (names.length / entries.length) >= 0.8 &&
                entries.every(([, value]) => !pytorch.Utility.isInstance(value, 'builtins.dict') || Array.from(value.values()).every((value) => !pytorch.Utility.isTensor(value)))) {
                const modules = new Map();
                for (const [name, value] of entries) {
                    const separator = name.indexOf('.') === -1 && name.indexOf('|') !== -1 ? '|' : '.';
                    const path = name.split(separator);
                    let property = path.pop();
                    if (path.length > 1 && path[path.length - 1] === '_packed_params') {
                        property = `${path.pop()}.${property}`;
                    }
                    const key = path.join(separator);
                    if (!modules.has(key)) {
                        modules.set(key, {});
                    }
                    const module = modules.get(key);
                    if (pytorch.Utility.isTensor(value)) {
                        value.__name__ = name;
                    }
                    module[property] = value;
                }
                return modules;
            }
        }
        if (obj && !Array.isArray(obj) && Object(obj) === obj) {
            const modules = new Map();
            const entries = obj instanceof Map ? Array.from(obj) : Object.entries(obj);
            if (entries.length > 0 && entries) {
                for (const [key, value] of entries) {
                    const name = key.toString();
                    if (!value || Object(value) !== value || pytorch.Utility.isTensor(value) || ArrayBuffer.isView(value)) {
                        return null;
                    }
                    if (!modules.has(name)) {
                        modules.set(name, {});
                    }
                    const module = modules.get(name);
                    let tensor = false;
                    const entries = value instanceof Map ? value : new Map(Object.entries(value));
                    for (const [name, value] of entries) {
                        if (typeof name !== 'string') {
                            return null;
                        }
                        if (name.indexOf('.') !== -1) {
                            return null;
                        }
                        if (name === '_metadata') {
                            continue;
                        }
                        if (typeof value === 'string' || typeof value === 'number') {
                            module[name] = value;
                            continue;
                        }
                        if (pytorch.Utility.isTensor(value)) {
                            value.__name__ = name;
                            module[name] = value;
                            tensor = true;
                        }
                    }
                    if (!tensor) {
                        return null;
                    }
                }
                return modules;
            }
        }
        return null;
    }

    static isMetadataObject(obj) {
        if (pytorch.Utility.isInstance(obj, 'collections.OrderedDict')) {
            for (const value of obj.values()) {
                if (pytorch.Utility.isInstance(value, 'builtins.dict')) {
                    const entries = Array.from(value);
                    if (entries.length !== 1 && entries[0] !== 'version' && entries[1] !== 1) {
                        return false;
                    }
                }
            }
            return true;
        }
        return false;
    }
};

pytorch.nnapi = {};

pytorch.nnapi.SerializedModel = class {

    constructor(serialized_model, buffers) {
        const reader = base.BinaryReader.open(serialized_model);
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
            [0, 'float32'],
            [1, 'int32'],
            [2, 'uint32'],
            [3, 'float32[]'],
            [4, 'int32[]'],
            [5, 'quant8_asymm[]'],
            [6, 'boolean'],
            [7, 'quant16_symm[]'],
            [8, 'float16[]'],
            [9, 'boolean[]'],
            [10, 'float16'],
            [11, 'quant8_symm_per_channel[]'],
            [12, 'quant16_asymm[]'],
            [13, 'quant8_symm[]'],
            [14, 'quant8_asymm_signed[]'],
            [16, 'model']
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
                identifier: i,
                inputs: new Array(reader.uint32()),
                outputs: new Array(reader.uint32())
            };
        }
        for (const operand of operands) {
            for (let i = 0; i < operand.dimensions.length; i++) {
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
                            throw new pytorch.Error(`Unsupported NNAPI operand type '${operand.data_type}'.`);
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
            for (let i = 0; i < operation.inputs.length; i++) {
                const index = reader.uint32();
                operation.inputs[i] = operands[index];
            }
            for (let i = 0; i < operation.outputs.length; i++) {
                const index = reader.uint32();
                operation.outputs[i] = operands[index];
            }
        }
        for (let i = 0; i < this.inputs.length; i++) {
            const index = reader.uint32();
            this.inputs[i] = operands[index];
        }
        for (let i = 0; i < this.outputs.length; i++) {
            const index = reader.uint32();
            this.outputs[i] = operands[index];
        }
        if (reader.position !== reader.length) {
            throw new pytorch.Error('Invalid NNAPI serialized model length.');
        }
    }
};

pytorch.nnapi.Graph = class {

    constructor(model) {
        this.name = 'torch.classes._nnapi.Compilation';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (operand) => {
            if (!values.has(operand.index)) {
                const name = operand.index.toString();
                const dimensions = operand.dimensions;
                const shape = new pytorch.TensorShape(dimensions);
                let dataType = operand.data_type.replace('[]', '');
                let quantization = null;
                switch (dataType) {
                    case 'quant8_asymm':
                    case 'quant8_symm_per_channel':
                    case 'quant8_symm':
                    case 'quant8_asymm_signed[]':
                    case 'quant16_asymm':
                    case 'quant16_symm':
                        quantization = dataType;
                        dataType = dataType.indexOf('16') === -1 ? 'uint8' : 'uint16';
                        break;
                    default:
                        break;
                }
                const type = new pytorch.TensorType(dataType, shape);
                let initializer = null;
                if (operand.data) {
                    const size = dimensions.reduce((a, b) => a * b, 1);
                    const tensor = {
                        size: () => dimensions,
                        stride: () => null,
                        storage_offset: () => 0,
                        storage: () => ({
                            dtype: { __reduce__: () => type.dataType },
                            data: operand.data, size: () => size
                        })
                    };
                    initializer = new pytorch.Tensor(null, tensor);
                }
                if (quantization || (operand.scale !== undefined && operand.scale !== 0) || (operand.zero_point !== undefined && operand.zero_point !== 0)) {
                    quantization = {
                        type: quantization || 'linear',
                        scale: [operand.scale],
                        offset: [operand.zero_point]
                    };
                }
                const value = new pytorch.Value(name, type, quantization, initializer);
                values.set(operand.index, value);
            }
            return values.get(operand.index);
        };
        const metadata = new pytorch.nnapi.Metadata();
        for (const operation of model.operations) {
            const node = new pytorch.nnapi.Node(metadata, operation, values);
            this.nodes.push(node);
        }
        for (let i = 0; i < model.inputs.length; i++) {
            const name = i.toString();
            const operand = model.inputs[i];
            const argument = new pytorch.Argument(name, [values.map(operand)]);
            this.inputs.push(argument);
        }
        for (let i = 0; i < model.outputs.length; i++) {
            const name = i.toString();
            const operand = model.outputs[i];
            const argument = new pytorch.Argument(name, [values.map(operand)]);
            this.outputs.push(argument);
        }
    }
};

pytorch.nnapi.Node = class {

    constructor(metadata, operation, values) {
        const signature = (operation.inputs || []).map((input) => input.data_type);
        this.name = '';
        this.type = metadata.type(operation.index, signature);
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        if (operation.identifier !== undefined) {
            this.identifier = operation.identifier.toString();
        }
        if (Array.isArray(operation.inputs)) {
            const inputs = this.type.inputs;
            for (let i = 0; i < operation.inputs.length; i++) {
                const name = i < inputs.length ? inputs[i].name : i.toString();
                const operand = operation.inputs[i];
                if (operand.dimensions.length > 0) {
                    const value = values.map(operand);
                    const argument = new pytorch.Argument(name, [value]);
                    this.inputs.push(argument);
                } else if (name === 'activation') {
                    const activation = new Map([[1, 19], [2, 20], [3, 21]]).get(operand.value) || 0;
                    if (activation !== 0) {
                        this.chain.push(new pytorch.nnapi.Node(metadata, { index: activation }));
                    }
                } else {
                    const attribute = new pytorch.Argument(name, operand.value, operand.data_type, false);
                    this.inputs.push(attribute);
                }
            }
        }
        if (Array.isArray(operation.outputs)) {
            const outputs = this.type.outputs;
            for (let i = 0; i < operation.outputs.length; i++) {
                const name = i < outputs.length ? outputs[i].name : i.toString();
                const operand = operation.outputs[i];
                const value = values.map(operand);
                const argument = new pytorch.Argument(name, [value]);
                this.outputs.push(argument);
            }
        }
    }
};

pytorch.nnapi.Metadata = class {

    constructor() {
        this._types = new Map();
        // https://developer.android.com/ndk/reference/group/neural-networks
        // https://github.com/pytorch/pytorch/commits/master/torch/backends/_nnapi/serializer.py
        this.register(0, 'ADD', '', ['A', 'B'], [['activation', 'int32']], ['C']);
        this.register(1, 'AVERAGE_POOL_2D', 'Pool', ['input'], [['padding_left', 'int32'], ['padding_right', 'int32'], ['padding_top', 'int32'], ['padding_bottom', 'int32'], ['stride_x', 'int32'], ['stride_y', 'int32'], ['filter_x', 'int32'], ['filter_y', 'int32'], ['activation', 'int32'], ['nchw', 'boolean']], ['output']);
        this.register(1, 'AVERAGE_POOL_2D', 'Pool', ['input'], [['padding_scheme', 'int32'], ['stride_x', 'int32'], ['stride_y', 'int32'], ['filter_x', 'int32'], ['filter_y', 'int32'], ['activation', 'int32'], ['nchw', 'boolean']], ['output']);
        this.register(2, 'CONCATENATION');
        this.register(3, 'CONV_2D', 'Layer', ['input', 'weights', 'bias'], [['padding_left', 'int32'], ['padding_right', 'int32'], ['padding_top', 'int32'], ['padding_bottom', 'int32'], ['stride_x', 'int32'], ['stride_y', 'int32'], ['activation', 'int32'], ['nchw', 'boolean'], ['dilation_width', 'int32'], ['dilation_height', 'int32']], ['output']);
        this.register(3, 'CONV_2D', 'Layer', ['input', 'weights', 'bias'], [['padding_scheme', 'int32'], ['stride_x', 'int32'], ['stride_y', 'int32'], ['activation', 'int32'], ['nchw', 'boolean'], ['dilation_width', 'int32'], ['dilation_height', 'int32']], ['output']);
        this.register(4, 'DEPTHWISE_CONV_2D', 'Layer', ['input', 'weights', 'bias'], [['padding_left', 'int32'], ['padding_right', 'int32'], ['padding_top', 'int32'], ['padding_bottom', 'int32'], ['stride_x', 'int32'], ['stride_y', 'int32'], ['activation', 'int32'], ['nchw', 'boolean'], ['dilation_width', 'int32'], ['dilation_height', 'int32']], ['output']);
        this.register(4, 'DEPTHWISE_CONV_2D', 'Layer', ['input', 'weights', 'bias'], [['padding_scheme', 'int32'], ['stride_x', 'int32'], ['stride_y', 'int32'], ['activation', 'int32'], ['nchw', 'boolean'], ['dilation_width', 'int32'], ['dilation_height', 'int32']], ['output']);
        this.register(5, 'DEPTH_TO_SPACE');
        this.register(6, 'DEQUANTIZE');
        this.register(7, 'EMBEDDING_LOOKUP');
        this.register(8, 'FLOOR');
        this.register(9, 'FULLY_CONNECTED', 'Layer', ['input', 'weights', 'bias'], [['activation', 'int32']], ['output']);
        this.register(10, 'HASHTABLE_LOOKUP');
        this.register(11, 'L2_NORMALIZATION');
        this.register(12, 'L2_POOL_2D', 'Pool');
        this.register(13, 'LOCAL_RESPONSE_NORMALIZATION');
        this.register(14, 'LOGISTIC');
        this.register(15, 'LSH_PROJECTION');
        this.register(16, 'LSTM', 'Layer');
        this.register(17, 'MAX_POOL_2D', 'Pool');
        this.register(18, 'MUL');
        this.register(19, 'RELU', 'Activation', ['input'], [], ['output']);
        this.register(20, 'RELU1', 'Activation');
        this.register(21, 'RELU6', 'Activation');
        this.register(22, 'RESHAPE', 'Shape', ['input', 'shape'], [], ['output']);
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
        const type = {};
        type.name = name;
        type.inputs = inputs.map((name) => ({ name, type: 'Tensor' }));
        type.inputs = type.inputs.concat(attributes.map(([name, type]) => ({ name, type })));
        type.outputs = outputs.map((name) => ({ name, type: 'Tensor' }));
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
            const inputs = type.inputs;
            if (signature.length < inputs.length) {
                if (inputs.every((input, i) => input.type === undefined || input.type === 'Tensor' || input.type === signature[i])) {
                    return type;
                }
            }
        }
        return types[0];
    }
};

pytorch.Metadata = class {

    static async open(context) {
        if (!pytorch.Metadata._metadata) {
            let data = null;
            try {
                data = await context.request('pytorch-metadata.json');
            } catch {
                // continue regardless of error
            }
            pytorch.Metadata._metadata = new pytorch.Metadata(data);
        }
        return pytorch.Metadata._metadata;
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
        const key = `${type}:${name}`;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this.type(type);
            if (metadata) {
                if (metadata.inputs) {
                    for (const input of metadata.inputs) {
                        this._attributes.set(`${type}:${input.name}`, input);
                    }
                }
                if (metadata.attributes) {
                    for (const attribute of metadata.attributes) {
                        this._attributes.set(`${type}:${attribute.name}`, attribute);
                    }
                }
            }
        }
        return this._attributes.get(key);
    }
};

numpy.Tensor = class  {

    constructor(array) {
        this.type = new numpy.TensorType(array.dtype.__name__, new numpy.TensorShape(array.shape));
        this.stride = array.strides.map((stride) => stride / array.itemsize);
        this.values = this.type.dataType === 'string' || this.type.dataType === 'object' || this.type.dataType === 'void' ? array.flatten().tolist() : array.tobytes();
        this.encoding = this.type.dataType === 'string' || this.type.dataType === 'object' ? '|' : array.dtype.byteorder;
    }
};

numpy.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

numpy.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions && this.dimensions.length > 0 ? `[${this.dimensions.join(',')}]` : '';
    }
};

pytorch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading PyTorch model.';
    }
};

export const ModelFactory = pytorch.ModelFactory;
