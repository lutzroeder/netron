
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
                            } else if (pytorch.Utility.isInstance(obj, 'torch.Value') || pytorch.Utility.isInstance(obj, 'torch.Node')) {
                                continue;
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
                if (node.kind() === 'prim::TupleConstruct' &&
                    node.inputs().length === 0 &&
                    node.outputs().length === 1 &&
                    node.outputs().every((output) => output.uses().length === 0)) {
                    continue;
                }
                if (node.kind() === 'prim::ListConstruct') {
                    if (node.outputs().length === 1 &&
                        node.outputs().every((output) => output.uses().length === 1) &&
                        node.inputs().every((input) => pytorch.Utility.isTensor(input.value) || pytorch.Utility.isInstance(input, 'torch.Value'))) {
                        continue;
                    }
                    if (node.inputs().length === 0 &&
                        node.outputs().length === 1 &&
                        node.outputs().every((output) => output.uses().length === 0)) {
                        continue;
                    }
                    if (node.inputs().every((value) => value && (pytorch.Utility.isInstance(value.type(), 'torch.IntType') || pytorch.Utility.isInstance(value.type(), 'torch.FloatType') || pytorch.Utility.isInstance(value.type(), 'torch.StringType') || pytorch.Utility.isInstance(value.type(), 'torch.ComplexType'))) &&
                        node.outputs().length === 1 &&
                        node.outputs().every((output) => output.uses().length === 1)) {
                        continue;
                    }
                }
                if (node.kind() === 'prim::ListUnpack' &&
                    node.inputs().length === 1 &&
                    node.inputs().every((input) => input.uses().length === 1) &&
                    node.outputs().every((output) => pytorch.Utility.isTensor(output.value))) {
                    continue;
                }
                if (node.kind() === 'prim::Constant' && node.outputs().length === 1 && node.outputs()[0].uses().length === 1) {
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
                                    if (pytorch.Utility.isInstance(value, 'torch.Value')) {
                                        continue;
                                    }
                                    modules.push(value);
                                }
                            }
                        }
                        queue.push(...modules.reverse());
                    }
                }
            }
        } else if (type === 'torch.export.exported_program.ExportedProgram' && module.graph) {
            const exported_program = module;
            const graph = exported_program.graph;
            const inputs_to_parameters = exported_program.graph_signature.inputs_to_parameters();
            const inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers();
            const values = new Map();
            values.map = (obj) => {
                if (!values.has(obj)) {
                    let type = null;
                    const val = obj.meta.get('val');
                    if (val && val.dtype) {
                        const dataType = val.dtype.__reduce__();
                        const shape = new pytorch.TensorShape(val.shape);
                        type = new pytorch.TensorType(dataType, shape);
                    }
                    const value = new pytorch.Value(obj.name, type);
                    values.set(obj, value);
                }
                return values.get(obj);
            };
            const nodes = new Map(graph.nodes.map((node) => [node.name, node]));
            for (const node of graph.nodes) {
                if (node.op === 'placeholder') {
                    if (inputs_to_parameters.has(node.name)) {
                        const key = inputs_to_parameters.get(node.name);
                        const parameter = exported_program.state_dict.get(key);
                        if (parameter) {
                            const tensor = new pytorch.Tensor(key, parameter.data);
                            const value = new pytorch.Value(key, null, null, tensor);
                            values.set(node, value);
                        }
                    }
                    if (inputs_to_buffers.has(node.name)) {
                        const key = inputs_to_buffers.get(node.name);
                        const buffer = exported_program.state_dict.get(key);
                        if (buffer) {
                            const tensor = new pytorch.Tensor(key, buffer);
                            const value = new pytorch.Value(key, null, null, tensor);
                            values.set(node, value);
                        }
                    }
                }
            }
            for (const obj of graph.nodes) {
                if (obj.op === 'placeholder' && obj.users.size <= 1) {
                    continue;
                }
                if (obj.op === 'call_function') {
                    if (obj.target.__module__ === 'operator' && obj.target.__name__ === 'getitem') {
                        continue;
                    }
                    if (obj.users.size === 0) {
                        continue;
                    }
                }
                if (obj.op === 'output') {
                    for (const output of obj.args) {
                        if (output.op === 'call_function' && output.target.__module__ === 'operator' && output.target.__name__ === 'getitem') {
                            continue;
                        }
                        const value = values.map(output);
                        const argument = new pytorch.Argument(output.name, [value]);
                        this.outputs.push(argument);
                    }
                    continue;
                }
                const node = new pytorch.Node(metadata, obj.name, null, obj, null, values);
                this.nodes.push(node);
            }
            for (const input_spec of exported_program.graph_signature.user_inputs()) {
                const node = nodes.get(input_spec);
                const value = values.map(node);
                const argument = new pytorch.Argument(input_spec, [value]);
                this.inputs.push(argument);
            }
            /*
            for (const output_spec of exported_program.graph_signature.user_outputs()) {
                const value = values.map(output_spec);
                const argument = new pytorch.Argument(output_spec, [value]);
                this.outputs.push(argument);
            }
            */
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

pytorch.Value = class Value {

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
        this.metadata = [];
        let module = null;
        if (pytorch.Utility.isInstance(obj, 'torch.Node')) {
            const node = obj;
            const kind = node.kind();
            this.type = {
                identifier: kind,
                name: kind.indexOf('::') === -1 ? kind : kind.split('::').pop().split('.')[0]
            };
            const schema = node.schema();
            if (schema && schema.category) {
                this.type.category = schema.category;
            }
            const inputs = node.inputs();
            const outputs = node.outputs();
            const getAttribute = (node, name) => {
                const kind = node.kindOf(name);
                let value = null;
                let type = null;
                switch (kind) {
                    case 's': value = node.s(name); type = 'string'; break;
                    case 'i': value = node.i(name); type = 'int64'; break;
                    case 'f': value = node.f(name); type = 'float32'; break;
                    case 'ss': value = node.ss(name); type = 'string[]'; break;
                    case 'ival': value = node.ival(name); break;
                    default: throw new pytorch.Error(`Unsupported attribute kind '${kind}'.`);
                }
                return [type, value];
            };
            for (const name of node.attributeNames()) {
                const [type, value] = getAttribute(node, name);
                const attribute = new pytorch.Argument(name, value, type);
                this.attributes.push(attribute);
            }
            let match = true;
            let count = 0;
            for (const input of inputs) {
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
            const mapTensor = (input) => {
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
            };
            for (let i = 0; i < inputs.length; i++) {
                const input = inputs[i];
                const arg = schema && schema.arguments && i < schema.arguments.length ? schema.arguments[i] : null;
                const name = arg && arg.name ? arg.name : i.toString();
                let type = arg ? arg.real_type : null;
                let array = false;
                if (pytorch.Utility.isInstance(type, 'torch.ListType')) {
                    array = true;
                    type = type.getElementType();
                }
                let argument = null;
                if (type && pytorch.Utility.isInstance(type, 'torch.ClassType')) {
                    const obj = input.value;
                    if (!array && initializers.has(obj)) {
                        const node = new pytorch.Node(metadata, name, type.qualified_name(), obj, initializers, values);
                        argument = new pytorch.Argument(name, node, 'object');
                    } else if (array && Array.isArray(obj) && obj.every((obj) => initializers.has(obj))) {
                        const node = obj.map((obj) => new pytorch.Node(metadata, name, type.qualified_name(), obj, initializers, values));
                        argument = new pytorch.Argument(name, node, 'object[]');
                    } else {
                        const identifier = input.unique().toString();
                        const value = values.map(identifier);
                        argument = new pytorch.Argument(name, [value]);
                    }
                } else if (pytorch.Utility.isInstance(input, 'torch.Value') && !pytorch.Utility.isTensor(input.value)) {
                    if (input.node() === null && input.value !== undefined) {
                        if (Array.isArray(input.value) && input.value.every((value) => pytorch.Utility.isTensor(value))) {
                            continue;
                        }
                        const type = input.type() ? pytorch.Utility.toType(input.type()) : null;
                        argument = new pytorch.Argument(name, input.value, type || 'attribute');
                    } else if (pytorch.Utility.isInstance(input.type(), 'torch.ListType')) {
                        if (input.node() && input.node().kind() === 'prim::ListConstruct' && input.uses().length === 1 &&
                        input.node().inputs().every((value) => pytorch.Utility.isInstance(value, 'torch.Value') || pytorch.Utility.isInstance(value.type(), 'torch.IntType') || pytorch.Utility.isInstance(value.type(), 'torch.FloatType') || pytorch.Utility.isInstance(value.type(), 'torch.StringType') || pytorch.Utility.isInstance(value.type(), 'torch.ComplexType') || pytorch.Utility.isInstance(value.type(), 'torch.TensorType'))) {
                            const list = input.node().inputs();
                            const args = list.map((value) => {
                                if (pytorch.Utility.isTensor(value.value)) {
                                    return mapTensor(value);
                                }
                                if (value.uses().length === 1 && value.node().kind() === 'prim::Constant') {
                                    return getAttribute(value.node(), 'value')[1];
                                }
                                if (value.uses().length === 1 && value.node() === input.node() && value.value !== undefined) {
                                    return value.value;
                                }
                                const identifier = value.unique().toString();
                                return values.map(identifier);
                            });
                            const type = list.every((value) => (pytorch.Utility.isTensor(value.value)) || value.value === null) ? null : pytorch.Utility.toType(input.type());
                            argument = new pytorch.Argument(name, args, type);
                        } else {
                            const identifier = input.unique().toString();
                            argument = new pytorch.Argument(name, [values.map(identifier)]);
                        }
                    } else if (pytorch.Utility.isInstance(input.type(), 'torch.StringType') && typeof input.value === 'string') {
                        argument = new pytorch.Argument(name, input.value, 'string');
                    } else if (input.node() && input.uses().length === 1 && input.node().kind() === 'prim::Constant') {
                        let [type, value] = getAttribute(input.node(), 'value');
                        const valueType = input.node().outputs()[0].type();
                        if (valueType) {
                            type = pytorch.Utility.toType(valueType);
                            if (type === 'boolean') {
                                value = Boolean(value);
                            }
                        }
                        argument = new pytorch.Argument(name, value, type || 'attribute');
                    } else {
                        const identifier = input.unique().toString();
                        const value = values.map(identifier);
                        argument = new pytorch.Argument(name, [value]);
                    }
                } else if (pytorch.Utility.isTensor(input.value) || input.value === undefined || input.value === null) {
                    let list = [input];
                    if (input.node() && node !== input.node() &&
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
                } else if (Array.isArray(input.value) && input.value.some((value) => pytorch.Utility.isInstance(value, 'torch.Value'))) {
                    const args = input.value.map((value) => {
                        if (pytorch.Utility.isInstance(value, 'torch.Value')) {
                            const identifier = value.unique().toString();
                            return values.map(identifier);
                        }
                        return value;
                    });
                    argument = new pytorch.Argument(name, args, pytorch.Utility.toType(type));
                } else {
                    throw new pytorch.Error('Unsupported input value');
                }
                this.inputs.push(argument);
            }
            for (let i = 0; i < outputs.length; i++) {
                const output = outputs[i];
                const ret = schema && schema.returns && i < schema.returns.length ? schema.returns[i] : null;
                if (ret && ret.name) {
                    name = ret.name;
                } else {
                    name = i === 0 && outputs.length === 1 ? 'output' : `${i}`;
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
        } else if (pytorch.Utility.isInstance(obj, 'torch.fx.node.Node')) {
            if (obj.op === 'call_function') {
                const name = obj.target.name;
                this.type = {
                    identifier: name,
                    name: name.indexOf('::') === -1 ? name : name.split('::').pop().split('.')[0]
                };
                const schema = obj.target._schema;
                if (schema && schema.category) {
                    this.type.category = schema.category;
                }
                let args = obj.args.map((arg, index) => {
                    const name = schema && Array.isArray(schema.arguments) ? schema.arguments[index].name : '';
                    return [name, arg];
                });
                const inputs = new Map((schema ? schema.arguments : []).map((arg) => [arg.name, arg]));
                args = args.concat(Array.from(obj.kwargs));
                for (const [name, arg] of args) {
                    const type = inputs.has(name) ? pytorch.Utility.toType(inputs.get(name).real_type) : null;
                    if (pytorch.Utility.isInstance(arg, 'torch.fx.node.Node')) {
                        const value = values.map(arg);
                        const argument = new pytorch.Argument(name, [value]);
                        this.inputs.push(argument);
                    } else if (Array.isArray(arg) && arg.every((arg) => pytorch.Utility.isInstance(arg, 'torch.fx.node.Node') || arg === null)) {
                        const list = arg.map((arg) => arg === null ? null : values.map(arg));
                        const argument = new pytorch.Argument(name, list);
                        this.inputs.push(argument);
                    } else if (Array.isArray(arg)) {
                        const list = arg.map((arg) => pytorch.Utility.isInstance(arg, 'torch.fx.node.Node') ? values.map(arg) : arg);
                        const argument = new pytorch.Argument(name, list, type || 'attribute');
                        this.inputs.push(argument);
                    } else if (pytorch.Utility.isInstance(arg, 'torch.dtype') ||
                        pytorch.Utility.isInstance(arg, 'torch.device') ||
                        pytorch.Utility.isInstance(arg, 'torch.layout') ||
                        pytorch.Utility.isInstance(arg, 'torch.memory_format')) {
                        const argument = new pytorch.Argument(name, arg.toString(), type || 'attribute');
                        this.inputs.push(argument);
                    } else {
                        const argument = new pytorch.Argument(name, arg, type || 'attribute');
                        this.inputs.push(argument);
                    }
                }
                let outputs = [obj];
                if (obj.users.size > 1) {
                    const users = Array.from(obj.users.keys());
                    if (users.every((user) => user.op === 'call_function' && user.target.__module__ === 'operator' && user.target.__name__ === 'getitem')) {
                        outputs = new Array(obj.users.size);
                        for (const user of users) {
                            const [, index] = user.args;
                            outputs[index] = user;
                        }
                    }
                }
                for (let i = 0; i < outputs.length; i++) {
                    const node = outputs[i];
                    const value = values.map(node);
                    const name = schema && schema.returns && schema.returns[i] ? schema.returns[i].name || 'output' : 'output';
                    const argument = new pytorch.Argument(name, [value]);
                    this.outputs.push(argument);
                }
                for (const [name, value] of obj.meta) {
                    if (name === 'val' || name === 'stack_trace' || name === 'torch_fn' ||
                        (Array.isArray(value) && value.length === 0) ||
                        (value instanceof Map && value.size === 0)) {
                        continue;
                    }
                    if (typeof value === 'string') {
                        const argument = new pytorch.Argument(name, value, 'string');
                        this.metadata.push(argument);
                    } else if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
                        const argument = new pytorch.Argument(name, value, 'string[]');
                        this.metadata.push(argument);
                    } else if (value instanceof Map && value.size > 0) {
                        // const argument = new pytorch.Argument(name, Object.fromEntries(Array.from(value)));
                        // this.metadata.push(argument);
                    } else {
                        // const argument = new pytorch.Argument(name, value);
                        // this.metadata.push(argument);
                    }
                }
            } else if (obj.op === 'placeholder') {
                this.type = { name: obj.op };
                const value = values.map(obj);
                const argument = new pytorch.Argument('value', [value]);
                this.inputs.push(argument);
            } else {
                throw new pytorch.Error(`Unsupported node operation '${obj.op}'.`);
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
            if (type instanceof pytorch.nnapi.Graph) {
                this.type = type;
            } else {
                const key = type.startsWith('__torch__.') ? type.substring(10) : type;
                const value = metadata.type(key);
                this.type = value ? { ...value } : { name: type };
                this.type.identifier = type;
                if (this.type.name.indexOf('(') !== -1) {
                    throw new Error();
                }
                if (this.type.name.indexOf('::') !== -1) {
                    throw new Error();
                }
                // [name] = this.type.name.split('(');
                // this.type.name = name.indexOf('::') === -1 ? name : name.split('::').pop().split('.')[0];
            }
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
        const execution = new python.Execution();
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
        const execution = new python.Execution();
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
        const execution = new pytorch.Execution(null, metadata);
        for (const event in this._events) {
            execution.on(event[0], event[1]);
        }
        const stream = this.context.stream;
        const torch = execution.__import__('torch');
        torch.mobile = await this.context.require('./pytorch-schema');
        torch.mobile = torch.mobile.torch.jit.mobile;
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
        const execution = new pytorch.Execution(null, metadata);
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this._entries);
        let torchscript = reader.has_record('constants.pkl');
        const version = reader.version();
        if (torchscript) {
            execution.trace = false;
            const module = torch.jit.load(reader);
            execution.trace = true;
            metadata.register(execution);
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
        const execution = new pytorch.Execution(null, metadata);
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        const torch = execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this._entries);
        if (this._model && this._model.producerName) {
            this.producer = this._model.producerName + (this._model.producerVersion ? ` v${this._model.producerVersion}` : '');
        }
        this.format = reader.has_record('attributes.pkl') ? 'TorchScript v1.1' : 'TorchScript v1.0';
        execution.trace = false;
        const module = torch.jit.load(reader);
        execution.trace = true;
        metadata.register(execution);
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
        const execution = new pytorch.Execution(null, metadata);
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

    async read(metadata) {
        this.format = 'PyTorch Export';
        try {
            const content = await this.context.fetch('version');
            if (content) {
                const reader = content.read('text');
                if (reader) {
                    this.version = reader.read();
                    this.version = this.version.split('\n').shift().trim();
                }
            }
        } catch {
            // continue regardless of error
        }
        const serialized_state_dict = await this._fetch('serialized_state_dict.pt') || await this._fetch('serialized_state_dict.json');
        const serialized_constants = await this._fetch('serialized_constants.pt') || await this._fetch('serialized_constants.json');
        const serialized_example_inputs = await this._fetch('serialized_example_inputs.pt');
        const f = new Map();
        f.set('serialized_exported_program.json', this.serialized_exported_program);
        f.set('serialized_state_dict.pt', serialized_state_dict);
        f.set('serialized_constants.pt', serialized_constants);
        f.set('serialized_example_inputs.pt', serialized_example_inputs);
        if (!this.version && this.serialized_exported_program) {
            const version = this.serialized_exported_program.schema_version;
            if (version && version.major && version.minor) {
                this.version = `${version.major}.${version.minor}`;
            }
        }
        this.format = this.version ? `${this.format} v${this.version}` : this.format;
        const execution = new python.Execution();
        for (const event of this._events) {
            execution.on(event[0], event[1]);
        }
        metadata.register(execution);
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
        this.module = torch._export.load(f);
    }

    async _fetch(name) {
        try {
            const context = await this.context.fetch(name);
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

    constructor(sources, metadata) {
        super(sources);
        this._metadata = metadata;
        const execution = this;
        this.torch = this.register('torch');
        const torch = this.torch;
        this.registerFunction('torch.jit.jit_module_from_flatbuffer', (f) => {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const stream = f;
            const reader = flatbuffers.BinaryReader.open(stream);
            const module = torch.mobile.serialization.Module.create(reader);
            const loader = new torch.jit.FlatBuffersLoader(cu);
            const cpp_module = loader.parseModule(module);
            // parse_and_initialize_jit_module
            //   const mobilem = parse_and_initialize_mobile_module_for_jit(data, jit_files, jit_constants);
            //   const m = jitModuleFromSourceAndConstants(mobilem._ivalue(), jit_files, jit_constants, mobilem.bytecode_version());
            // throw new pytorch.Error('torch.jit.mobile.serialization.Module not supported.');
            return torch.jit._script.wrap_cpp_module(cpp_module);
        });
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
                /*
                let buffers = [];
                if (!pytorch.Utility.isInstance(parameter_buffers, 'torch.Value')) {
                    buffers = parameter_buffers.map((buffer) => buffer.__source__.storage());
                }
                */
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
        this.register('__torch__').torch.classes._nnapi.Compilation.__type__ = new torch.ClassType('__torch__.torch.classes._nnapi.Compilation');
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
        this._metadata = metadata;
        this._types = new Map();
        for (const [name, value] of this._metadata._types) {
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

    constant(value) {
        const torch = this.torch;
        const node = this.graph.create('prim::Constant');
        let type = null;
        if (value === null) {
            node.ival_('value', value);
            type = torch.NoneType.get();
        } else if (typeof value === 'string') {
            node.s_('value', value);
            type = torch.StringType.get();
        } else if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
            node.ss_('value', value);
            type = torch.ListType.get(torch.StringType.get());
        } else if (typeof value === 'boolean') {
            // return value;
            node.i_('value', value === true ? 1 : 0);
            type = torch.BoolType.get();
        } else if (Number.isInteger(value)) {
            node.i_('value', value);
            type = torch.IntType.get();
        } else if (typeof value === 'number') {
            // return value;
            node.f_('value', value);
            type = torch.FloatType.get();
        } else {
            throw new pytorch.Error(`Unsupported value type '${typeof value}'.`);
        }
        if (type) {
            value = node.addOutput();
            value.setType(type);
        }
        return value;
    }

    variable(obj, node) {
        const torch = this.torch;
        if (this._values.has(obj)) {
            return this._values.get(obj);
        }
        let value = null;
        if (node) {
            value = node.addOutput();
        } else if (obj instanceof torch.Value) {
            value = obj;
        } else {
            value = new torch.Value(node ? node : this._graph);
        }
        if (pytorch.Utility.isTensor(obj)) {
            value.value = obj;
            value.setType(torch.TensorType.get());
            if (typeof obj !== 'string' && typeof obj !== 'number') {
                this._values.set(obj, value);
            }
            if (pytorch.Utility.isTensor(obj)) {
                obj.__variable__ = value.unique().toString();
            }
        }
        if (typeof obj === 'string') {
            value.value = obj;
            value.setType(torch.StringType.get());
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
                case 'uninitialized':
                    return this.builtins[expression.value];
                case 'CONSTANTS': {
                    if (!this._constants) {
                        const value = this.builtins[expression.value];
                        const entries = Object.entries(value).map(([name, value]) => {
                            if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'string')) {
                                value = this.constant(value);
                                return [name, value];
                            }
                            return [name, value];
                        });
                        this._constants = Object.fromEntries(entries);
                    }
                    return this._constants;
                }
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

    expression(expression, context) {
        if (!this.trace) {
            return super.expression(expression, context);
        }
        const torch = this.torch;
        switch (expression.type) {
            case 'id': {
                switch (expression.value) {
                    case 'True': return this.constant(true);
                    case 'False': return this.constant(false);
                    default: break;
                }
                return super.expression(expression, context);
            }
            case '=': {
                const target = expression.target;
                if (target.type === 'id') {
                    let value = this.expression(expression.expression, context);
                    if (typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number') {
                        value = this.constant(value);
                    } else if (typeof value !== 'object' && value !== undefined) {
                        throw new pytorch.Error(`Unsupported assignment value type '${typeof value}'.`);
                    }
                    context.set(target.value, value);
                    return undefined;
                } else if (target.type === 'tuple') {
                    context.target.push(target.value);
                    const value = this.expression(expression.expression, context);
                    context.target.pop();
                    if (target.value.every((item) => item.type === 'id')) {
                        if (value instanceof torch.Value) {
                            const node = this._graph.create('prim::TupleUnpack');
                            node.addInput(value);
                            const outputs = [];
                            for (let i = 0; i < target.value.length; i++) {
                                const item = target.value[i];
                                const output = node.addOutput();
                                const type = value.type();
                                if (type instanceof torch.ListType) {
                                    output.setType(value.type().getElementType());
                                } else if (type instanceof torch.TupleType) {
                                    output.setType(type.elements()[i]);
                                } else {
                                    throw new pytorch.Error(`Unsupported tuple unpack type '${type.kind()}'.`);
                                }
                                output.setDebugName(item.value);
                                context.set(item.value, output);
                                outputs.push(output);
                            }
                            return outputs;
                        }
                        if (target.value.length < value.length) {
                            throw new python.Error(`ValueError: too many values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        if (target.value.length > value.length) {
                            throw new python.Error(`ValueError: not enough values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        for (let i = 0; i < value.length; i++) {
                            context.set(target.value[i].value, value[i]);
                        }
                        return undefined;
                    }
                }
                break;
            }
            case 'call': {
                if (expression.target.type === 'id' && expression.target.value === 'annotate') {
                    const type = this.type(expression.args[0]);
                    let value = this.expression(expression.args[1], context);
                    if (value instanceof torch.Tensor) {
                        let name = null;
                        if (type instanceof torch.IntType) {
                            name = 'aten::IntImplicit';
                        } else if (type instanceof torch.FloatType) {
                            name = 'aten::FloatImplicit';
                        } else if (type instanceof torch.StringType) {
                            name = 'aten::StringImplicit';
                        } else if (type instanceof torch.ComplexType) {
                            name = 'aten::ComplexImplicit';
                        } else if (type instanceof torch.NumberType) {
                            name = 'aten::ScalarImplicit';
                        } else {
                            throw new pytorch.Error(`Unsupported annotation type '${type.kind()}'.`);
                        }
                        const target = { 'type': 'id', value: 'torch' };
                        name = name.replace('aten::', '');
                        return this.call(target, name, expression.args.slice(1), context);
                    }
                    if (value instanceof torch.Value) {
                        value.setType(type);
                    }
                    if (value === null) {
                        value = this.constant(value);
                        value.setType(type);
                    }
                    return value;
                }
                if (expression.target.type === 'id' && expression.target.value === 'uninitialized') {
                    const type = this.type(expression.args[0], context);
                    const node = this._graph.create('prim::Uninitialized');
                    const value = node.addOutput();
                    value.setType(type);
                    return value;
                }
                if (expression.target.type === 'id' && expression.target.value === 'unchecked_cast') {
                    let value = this.expression(expression.args[1], context);
                    const type = this.type(expression.args[0], context);
                    const node = this._graph.create('prim::unchecked_cast');
                    node.addInput(this.variable(value));
                    value = node.addOutput();
                    value.setType(type);
                    return value;
                }
                if (expression.target.type === 'id' && expression.target.value === 'isinstance') {
                    let value = this.expression(expression.args[1], context);
                    // const type = this.type(expression.args[0]);
                    const node = this._graph.create('prim::isinstance');
                    node.addInput(this.variable(value));
                    value = node.addOutput();
                    value.setType(torch.BoolType.get());
                    return value;
                }
                /*
                if (expression.target.type === '.') {
                    const target = this.target(expression.target.target, context); // this.expression(expression.target.target, context);
                    if (target instanceof torch.Value && target.type() instanceof torch.ClassType) {
                        const node = this._graph.create('prim::CallMethod');
                        const name = this.variable(expression.target.member.value, node);
                        node.addInput(name);
                        const args = expression.args.map((expression) => this.expression(expression, context));
                        for (const arg of args) {
                            const value = this.variable(arg, node);
                            node.addInput(value);
                        }
                        return node.addOutput();
                    }
                }
                */
                return super.expression(expression, context);
            }
            case '[]': {
                if (expression.arguments.type === 'list' && expression.arguments.value.length === 1) {
                    const target = this.expression(expression.target, context);
                    if (target instanceof torch.Value && target.type() instanceof torch.ListType) {
                        let index = this.expression(expression.arguments.value[0], context);
                        const node = this._graph.create('aten::__getitem__.t');
                        node.addInput(target);
                        if (Number.isInteger(index)) {
                            index = this.constant(index);
                        }
                        node.addInput(index);
                        const value = node.addOutput();
                        value.setType(target.type().getElementType());
                        return value;
                    }
                    if (target instanceof torch.Value && target.type() instanceof torch.DictType) {
                        let key = this.expression(expression.arguments.value[0], context);
                        const node = this._graph.create('aten::__getitem__.t');
                        node.addInput(target);
                        if (target.type().getKeyType() instanceof torch.StringType && typeof key === 'string') {
                            const value = new torch.Value(node);
                            value.value = key;
                            key = value;
                        } else if (target.type().getKeyType() instanceof torch.StringType && key.type() instanceof torch.StringType) {
                            // continue
                        } else {
                            throw new pytorch.Error(`Unsupported dictionary key type.`);
                        }
                        node.addInput(key);
                        const value = node.addOutput();
                        value.setType(target.type().getValueType());
                        return value;
                    }
                    if (target instanceof torch.Value && target.type() instanceof torch.TupleType) {
                        let index = this.expression(expression.arguments.value[0], context);
                        const node = this._graph.create('prim::TupleIndex');
                        const value = node.addOutput();
                        value.setType(target.type().elements()[index]);
                        node.addInput(target);
                        if (Number.isInteger(index)) {
                            const value = this.invoke('torch.Value', [node]);
                            value.value = index;
                            index = value;
                        }
                        node.addInput(index);
                        return value;
                    }
                }
                break;
            }
            case '.': {
                if (expression.member.type === 'id') {
                    const target = this.target(expression.target, context);
                    if (typeof expression.member.value === 'string' && target instanceof torch.Value && target.type() instanceof torch.ClassType) {
                        const type = target.type().findAttribute(expression.member.value);
                        const node = this.graph.create('prim::GetAttr');
                        node.s_(expression.member.value);
                        node.addInput(target);
                        const value = node.addOutput();
                        value.setType(type);
                        return value;
                    }
                    return target[expression.member.value];
                }
                throw new python.Error("Unsupported field expression.");
            }
            case 'list': {
                const list = expression.value.map((item) => this.expression(item, context));
                if (/* list.length > 0 && */ list.every((item) => pytorch.Utility.isInstance(item, 'torch.Value') || pytorch.Utility.isTensor(item) || Number.isInteger(item) || typeof item === 'string' || item === null)) {
                    const node = this._graph.create('prim::ListConstruct');
                    const output = node.addOutput();
                    for (const item of list) {
                        if (item instanceof torch.Value) {
                            node.addInput(item);
                            output.setType(torch.ListType.get(item.type()));
                        } else if (Number.isInteger(item)) {
                            const value = this.constant(item);
                            node.addInput(value);
                            output.setType(torch.ListType.get(torch.IntType.get()));
                        } else if (typeof item === 'string') {
                            const value = this.constant(item);
                            node.addInput(value);
                            output.setType(torch.ListType.get(torch.StringType.get()));
                        } else if (pytorch.Utility.isTensor(item)) {
                            const value = this.variable(item, null);
                            node.addInput(value);
                            output.setType(torch.ListType.get(torch.TensorType.get()));
                        } else {
                            const value = new torch.Value(node);
                            value.value = item;
                            node.addInput(value);
                        }
                    }
                    return output;
                }
                break;
            }
            case 'tuple': {
                const args = expression.value.map((expression) => this.expression(expression, context));
                const node = this._graph.create('prim::TupleConstruct');
                const types = [];
                const elements = [];
                for (const item of args) {
                    if (item instanceof torch.Value) {
                        node.addInput(item);
                        types.push(item.type());
                        elements.push(item);
                    } else if (pytorch.Utility.isTensor(item)) {
                        const value = this.variable(item, node);
                        node.addInput(value);
                        // value.value = item;
                        // value.setType(torch.TensorType.get());
                        types.push(value.type());
                        elements.push(item);
                    } else if (Number.isInteger(item)) {
                        const value = new torch.Value(node);
                        value.value = item;
                        types.push(torch.IntType.get());
                        elements.push(item);
                    } else if (typeof item === 'boolean') {
                        const value = new torch.Value(node);
                        value.value = item;
                        node.addInput(value);
                        types.push(torch.BoolType.get());
                        elements.push(item);
                    } else if (item === null) {
                        const value = new torch.Value(node);
                        value.value = item;
                        node.addInput(value);
                        types.push(torch.NoneType.get());
                        elements.push(item);
                    } else {
                        const value = new torch.Value(node);
                        value.value = item;
                        node.addInput(value);
                        types.push(torch.Type.get());
                        elements.push(item);
                    }
                }
                const value = node.addOutput();
                value.value = elements;
                value.setType(torch.TupleType.get(types));
                return value;
            }
            case 'dict': {
                const node = this._graph.create('prim::DictConstruct');
                let keyType = null;
                let valueType = null;
                for (const pair of expression.value) {
                    if (pair.type !== 'pair') {
                        throw new python.Error(`Unsupported dict item type '${pair.type}'.`);
                    }
                    const key = this.expression(pair.key, context);
                    const keyValue = this.variable(key, null);
                    keyType = keyValue.type();
                    const value = this.expression(pair.value, context);
                    const valueValue = this.variable(value, null);
                    valueType = valueValue.type();
                    node.addInput(keyValue);
                    node.addInput(valueValue);
                }
                const output = node.addOutput();
                if (keyType && valueType) {
                    output.setType(torch.DictType.get(keyType, valueType));
                }
                return output;
            }
            default: {
                break;
            }
        }
        return super.expression(expression, context);
    }

    statement(statement, context) {
        const torch = this.torch;
        if (!this.trace) {
            return super.statement(statement, context);
        }
        switch (statement.type) {
            case 'class': {
                super.statement(statement, context);
                const value = context.get(statement.name);
                const type = new torch.ClassType(`${value.__module__}.${value.__name__}`);
                for (const entry of statement.body.statements) {
                    if (entry.type === 'var') {
                        const variableType = this.type(entry.variableType, context);
                        type.addAttribute(entry.name, variableType);
                    }
                }
                value.__type__ = type;
                return undefined;
            }
            case 'if': {
                const test = this.expression(statement.test, context);
                if (test instanceof torch.Value) {
                    const node = this._graph.create('prim::If');
                    node.addInput(test);
                }
                if (test === true || test) {
                    const value = this.block(statement.body.statements, context);
                    if (value !== undefined) {
                        return value;
                    }
                    return undefined;
                } else if (test === false) {
                    if (statement.orelse) {
                        const value = this.block(statement.orelse.statements, context);
                        if (value !== undefined) {
                            return value;
                        }
                    }
                    return undefined;
                }
                throw new python.Error("Unsupported condition.");
            }
            default: {
                break;
            }
        }
        return super.statement(statement, context);
    }

    type(expression, context) {
        const torch = this.torch;
        if (expression.type === '[]' && expression.target.type === 'id') {
            switch (expression.target.value) {
                case 'List': {
                    const elementType = this.type(expression.arguments.value[0]);
                    return torch.ListType.get(elementType);
                }
                case 'Optional': {
                    const elementType = this.type(expression.arguments.value[0]);
                    return torch.OptionalType.get(elementType);
                }
                case 'Tuple': {
                    const elements = expression.arguments.value.map((expression) => this.type(expression));
                    return torch.TupleType.get(elements);
                }
                case 'Dict': {
                    const key = this.type(expression.arguments.value[0]);
                    const value = this.type(expression.arguments.value[1]);
                    return torch.DictType.get(key, value);
                }
                case 'Final': {
                    return this.type(expression.arguments.value[0]);
                }
                default: {
                    throw new pytorch.Error(`Unsupported type element expression '${expression.target.value}'.`);
                }
            }
        }
        if (expression.type === 'id') {
            switch (expression.value) {
                case 'Tensor': return torch.TensorType.get();
                case 'int': return torch.IntType.get();
                case 'str': return torch.StringType.get();
                case 'float': return torch.FloatType.get();
                case 'number': return torch.NumberType.get();
                case 'bool': return torch.BoolType.get();
                case 'None': return torch.NoneType.get();
                case 'NoneType': return torch.NoneType.get();
                default: throw new pytorch.Error(`Unsupported type expression '${expression.value}'.`);
            }
        }
        if (expression.type === '.') {
            const target = this.expression(expression, context);
            if (target && target.__type__ instanceof torch.ClassType) {
                return target.__type__;
            }
        }
        throw new pytorch.Error(`Unsupported type expression '${expression.type}'.`);
    }

    call(target, name, args, context) {
        if (!this.trace) {
            return super.call(target, name, args, context);
        }
        const torch = this.torch;
        if (name === '__new__') {
            const identifier = pytorch.Utility.target(target);
            if (identifier) {
                const type = this.resolve(identifier);
                if (type && type.__type__) {
                    const node = this.graph.create('prim::CreateObject');
                    const value = node.addOutput();
                    value.setType(type.__type__);
                    return value;
                }
            }
        }
        if (name === '__init__') {
            const obj = this.expression(target, context);
            if (args.length === 0) {
                return obj;
            }
            const node = this.graph.create('prim::CallMethod');
            node.s_('name', name);
            node.addInput(obj);
            const evalArgs = args.map((arg) => this.expression(arg, context));
            for (const arg of evalArgs) {
                this.variable(arg, node);
            }
            const value = node.addOutput();
            value.setType(obj.type());
            return value;
        }
        const overload = this._overload(target, name, args, context);
        if (!overload) {
            const moduleTarget = this.target(target, context); // this.expression(expression.target.target, context);
            if (moduleTarget instanceof torch.Value && moduleTarget.type() instanceof torch.ClassType) {
                const node = this.graph.create('prim::CallMethod');
                node.s_('name', name);
                const evalArgs = args.map((expression) => this.expression(expression, context));
                for (const arg of evalArgs) {
                    const value = this.variable(arg);
                    node.addInput(value);
                }
                return node.addOutput();
            }
            return super.call(target, name, args, context);
        }
        const [schema, evalArgs] = overload;
        const op = schema.overload_name ? `${schema.name}.${schema.overload_name}` : schema.name;
        const node = this._graph.create(op);
        const referencedParameters = [];
        const parameters = schema.arguments;
        const varTypes = new Map();
        varTypes.map = function(type) {
            if (type.kind() === 'VarType') {
                const key = type.annotation_str;
                if (!varTypes.has(key)) {
                    throw new pytorch.Error(`Unknown var type '${key}'.`);
                }
                return varTypes.get(key);
            }
            return type;
        };
        let position = 0;
        let index = 0;
        while (position < evalArgs.length) {
            if (index >= parameters.length) {
                if (schema.name.startsWith('_caffe2::') || schema.is_vararg) {
                    break;
                }
                throw new pytorch.Error('Invalid parameter length.');
            }
            const arg = parameters[index];
            if (arg.kwarg_only) {
                break;
            }
            index++;
            const v = evalArgs[position];
            let match = false;
            let input = null;
            let optional = false;
            let type = arg.real_type;
            if (type instanceof torch.OptionalType) {
                type = type.getElementType();
                optional = true;
            }
            if (optional === true &&
                (type instanceof torch.FloatType || type instanceof torch.BoolType || type instanceof torch.IntType || type instanceof torch.ComplexType || type.kind() === 'ScalarTypeType' || type instanceof torch.DeviceObjType || type.kind() === 'LayoutKind') &&
                v instanceof torch.Value && v.type() instanceof torch.NoneType) {
                position++;
                input = v;
                match = true;
            } else if (type instanceof torch.ListType && type.getElementType() instanceof torch.TensorType) {
                const v = evalArgs[position];
                if ((v instanceof torch.Value && v.type() instanceof torch.ListType && v.type().getElementType() instanceof torch.TensorType) ||
                    (Array.isArray(v) && v.every((item) => pytorch.Utility.isTensor(item) || item === null || (item instanceof torch.Value && item.type() instanceof torch.TensorType)))) {
                    position++;
                    if (v instanceof torch.Value) {
                        input = v;
                        match = true;
                    } else {
                        const list = this._graph.create('prim::ListConstruct');
                        for (const arg of v) {
                            const tensor = arg;
                            if (tensor) {
                                tensor.__count__ = (tensor.__count__ || 0) + 1;
                            }
                            const value = this.variable(tensor);
                            value.setType(torch.TensorType.get());
                            list.addInput(value);
                        }
                        const output = list.addOutput();
                        output.setType(torch.ListType.get(torch.TensorType.get()));
                        input = output;
                        match = true;
                    }
                } else {
                    if (optional) {
                        continue;
                    }
                    throw new pytorch.Error();
                }
            } else if (!this.isType(v, type, arg.N) && v !== null) {
                if (optional) {
                    continue;
                }
                throw new pytorch.Error('Invalid argument type.');
            } else if (args[position].type === '=' && args[position].target.value !== arg.name) {
                throw new pytorch.Error('Expected named argument.');
            } else {
                position++;
                if (v instanceof torch.Value) {
                    input = v;
                    match = true;
                } else {
                    const value = this.variable(v);
                    value.value = v;
                    input = value;
                    match = true;
                }
            }
            if (match) {
                node.addInput(input);
                if (type.kind() === 'VarType') {
                    const key = type.annotation_str;
                    if (input instanceof torch.Value && input.type()) {
                        varTypes.set(key, input.type());
                    } else if (input instanceof torch.Value && Number.isInteger(input.value)) {
                        varTypes.set(key, torch.IntType.get());
                    }
                    // throw new pytorch.Error("Unknown value type 't'.");
                }
                if (type instanceof torch.ListType && type.getElementType().kind() === 'VarType') {
                    const key = type.getElementType().annotation_str;
                    if (input instanceof torch.Value && input.type() instanceof torch.OptionalType && input.type().getElementType() instanceof torch.ListType) {
                        varTypes.set(key, input.type().getElementType().getElementType());
                    } else if (input instanceof torch.Value && input.type() instanceof torch.ListType) {
                        varTypes.set(key, input.type().getElementType());
                    } else if (Array.isArray(input) && input.length > 0 && input.every((item) => Number.isInteger(item))) {
                        varTypes.set(key, torch.IntType.get());
                    } else if (input.value && Array.isArray(input.value) && input.value.length > 0 && input.value.every((item) => Number.isInteger(item) || isNaN(item))) {
                        varTypes.set(key, torch.IntType.get());
                    } else if (input.value && Array.isArray(input.value) && input.value.length > 0 && input.value.every((item) => pytorch.Utility.isTensor(item))) {
                        varTypes.set(key, torch.TensorType.get());
                    } else {
                        // throw new pytorch.Error("Unknown value type 't'.");
                        continue;
                    }
                }
                if (type instanceof torch.DictType && type.getValueType().kind() === 'VarType') {
                    const key = type.getValueType().annotation_str;
                    if (input instanceof torch.Value && input.type() instanceof torch.DictType) {
                        varTypes.set(key, input.type().getValueType());
                    } else if (input.value && Object.values(input.value).every((item) => pytorch.Utility.isTensor(item))) {
                        varTypes.set(key, input.type().getValueType());
                    } else {
                        throw new pytorch.Error("Unknown dict type 't[]'.");
                    }
                }
                if (type instanceof torch.ListType && type.getElementType() instanceof torch.TupleType && type.getElementType().elements().length === 2 && type.getElementType().elements()[1].kind() === 'VarType') {
                    const key = type.getElementType().elements()[1].annotation_str;
                    if (input instanceof torch.Value && input.type() instanceof torch.ListType && input.type().getElementType() instanceof torch.TupleType) {
                        const elements = input.type().getElementType().elements();
                        if (elements.length === 2) {
                            varTypes.set(key, elements[1]);
                        }
                    }
                }
            }
        }
        if (args.every((arg, index) => index < position || (arg.type === '=' && arg.target && arg.target.type === 'id'))) {
            const params = new Map(parameters.slice(index).map((a) => [a.name, a]));
            while (position < args.length) {
                const v = evalArgs[position];
                const arg = params.get(args[position].target.value);
                position++;
                if (!arg) {
                    throw new pytorch.Error();
                }
                let type = arg.real_type;
                let optional = false;
                if (type instanceof torch.OptionalType) {
                    type = type.getElementType();
                    optional = true;
                }
                if (!this.isType(v, type)) {
                    if (optional) {
                        continue;
                    }
                    throw new pytorch.Error();
                }
                const value = this.variable(v);
                value.value = v;
                node.addInput(value);
            }
        }
        const result = [];
        for (let i = 0; i < schema.returns.length; i++) {
            const arg = schema.returns[i];
            const type = arg.real_type;
            switch (type.str()) {
                case 'Tensor': {
                    const output = this.createTensorOutput(schema.name, evalArgs, i);
                    output.__origin__ = schema.name;
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
                    value.setType(torch.ListType.get(torch.TensorType.get()));
                    result.push(value);
                    break;
                }
                case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                case '__torch__.torch.classes.rnn.CellParamsBase':
                case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                case '__torch__.torch.classes.xnnpack.LinearOpContext':
                case '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext': {
                    const value = this.invoke(type.qualified_name(), []);
                    this.variable(value, node);
                    result.push(value);
                    break;
                }
                case 'int': {
                    const value = this.variable(null, node);
                    value.__origin__ = schema.name;
                    value.setType(torch.IntType.get());
                    switch (schema.name) {
                        case 'aten::div.int': value.value = torch.div(evalArgs[0], evalArgs[1]); break;
                        case 'aten::dim': value.value = torch.dim(evalArgs[0]); break;
                        case 'aten::len.t': value.value = torch.len(evalArgs[0]); break;
                        // case 'aten::size.int': value.value = torch.size(evalArgs[0], evalArgs[1]); break;
                        default: break;
                    }
                    result.push(value);
                    break;
                }
                case 'int[]': {
                    const value = this.variable(null, node);
                    value.__origin__ = schema.name;
                    value.setType(torch.ListType.get(torch.IntType.get()));
                    switch (schema.name) {
                        // case 'aten::size': value.value = torch.size(evalArgs[0], evalArgs[1]); break;
                        default: break;
                    }
                    result.push(value);
                    break;
                }
                case 'Scalar':
                case 'Dict(str, Tensor)':
                case 'str':
                case 'str[]':
                case 'float':
                case 'float[]':
                case 'complex':
                case 'bool':
                case 'bool[]': {
                    const value = this.variable(null, node);
                    value.__origin__ = schema.name;
                    value.setType(type);
                    result.push(value);
                    break;
                }
                case 'Device': {
                    const value = this.variable(null, node);
                    value.__origin__ = schema.name;
                    value.setType(torch.DeviceObjType.get());
                    result.push(value);
                    break;
                }
                case 't': {
                    const value = this.variable(null, node);
                    value.__origin__ = schema.name;
                    const t = varTypes.map(type);
                    if (!t) {
                        throw new pytorch.Error(`Unknown var type 't'.`);
                    }
                    value.setType(t);
                    result.push(value);
                    break;
                }
                case 't[]': {
                    const value = this.variable(null, node);
                    value.__origin__ = schema.name;
                    const t = varTypes.map(type.getElementType());
                    if (!t) {
                        throw new pytorch.Error();
                    }
                    value.setType(torch.ListType.get(t));
                    result.push(value);
                    break;
                }
                default: {
                    if (type instanceof torch.DictType) {
                        const value = this.variable(null, node);
                        value.__origin__ = schema.name;
                        const keyType = varTypes.map(type.getKeyType());
                        const valueType = varTypes.map(type.getValueType());
                        value.setType(torch.DictType.get(keyType, valueType));
                        result.push(value);
                        break;
                    }
                    if (type instanceof torch.TupleType && type.elements().length === 2) {
                        const value = this.variable(null, node);
                        value.__origin__ = schema.name;
                        const keyType = varTypes.map(type.elements()[0]);
                        const valueType = varTypes.map(type.elements()[1]);
                        value.setType(torch.ListType.get(torch.TupleType.get([keyType, valueType])));
                        result.push(value);
                        break;
                    }
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

    createTensorOutput(op_name, evalArgs, i) {
        const torch = this.torch;
        const output = new torch.Tensor();
        if (i === 0) {
            switch (op_name) {
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
                    if (input instanceof torch.Value === false) {
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
                    }
                    break;
                }
                case 'aten::unsqueeze': {
                    const [input, dim] = evalArgs;
                    if (pytorch.Utility.isTensor(input)) {
                        const size = input.size();
                        if (Array.isArray(size) && dim !== undefined) {
                            const shape = size.slice();
                            shape.splice(dim, 0, 1);
                            output.resize_(shape);
                        } else {
                            output.resize_([NaN, NaN, NaN, NaN]);
                        }
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
        return output;
    }

    isType(obj, type, N) {
        const torch = this.torch;
        switch (type.str()) {
            case 'Tensor':
                return !Array.isArray(obj) && (pytorch.Utility.isTensor(obj) || obj === null ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.TensorType) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.OptionalType && obj.type().getElementType() instanceof torch.TensorType));
            case 'Tensor[]':
                return (Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => pytorch.Utility.isTensor(tensor) || tensor === null || (tensor instanceof torch.Value && tensor.type() instanceof torch.TensorType))) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.TensorType);
            case 'Tensor?[]':
                return (Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => pytorch.Utility.isTensor(tensor) || tensor === null || (tensor instanceof torch.Value && tensor.type() instanceof torch.TensorType))) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.OptionalType && obj.type().getElementType().getElementType() instanceof torch.TensorType);
            case 'Scalar':
                return (obj !== null && (obj !== Object(obj) || obj instanceof Number)) ||
                    (pytorch.Utility.isTensor(obj) && Array.isArray(obj.size()) && obj.size().length === 0) ||
                    (obj instanceof torch.Value && (obj.type() instanceof torch.IntType || obj.type() instanceof torch.FloatType || obj.type() instanceof torch.NumberType));
            case 'bool':
                return obj === true || obj === false || (pytorch.Utility.isInstance(obj, 'torch.Value') && obj.type() instanceof torch.BoolType);
            case 'bool[]':
                if (Array.isArray(obj) && obj.every((item) => item === true || item === false)) {
                    return true;
                }
                if (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.BoolType) {
                    return true;
                }
                return false;
            case 'SymInt':
            case 'int':
                return Number.isInteger(obj) || typeof obj === 'bigint' ||
                    (typeof obj === 'number' && isNaN(obj)) || (obj instanceof Number) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.IntType) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.OptionalType && obj.type().getElementType() instanceof torch.IntType);
            case 'SymInt[]':
            case 'SymInt[2]':
            case 'SymInt[3]':
            case 'SymInt[4]':
            case 'SymInt[5]':
            case 'SymInt[6]':
                if (Array.isArray(obj) && obj.every((item) => this.isType(item, torch.SymIntType.get()) || item === undefined || (item.__class__ === 'number' && isNaN(item)))) {
                    return true;
                }
                if (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.IntType) {
                    return true;
                }
                return false;
            case 'int[]':
                if (N === 1 && this.isType(obj, torch.IntType.get())) {
                    return true;
                }
                return (Array.isArray(obj) && obj.every((item) => this.isType(item, torch.IntType.get()) || item === undefined || (item.__class__ === 'number' && isNaN(item))) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.IntType)) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.OptionalType && obj.type().getElementType() instanceof torch.ListType && obj.type().getElementType().getElementType() instanceof torch.IntType);
            case 'SymInt[1]':
                return this.isType(obj, torch.IntType.get()) || this.isType(obj, torch.ListType.get(torch.IntType.get()));
            case 'float':
                return obj !== null && (typeof obj === 'number' || obj instanceof Number) ||
                    (pytorch.Utility.isInstance(obj, 'torch.Value') && pytorch.Utility.isInstance(obj.type(), 'torch.FloatType'));
            case 'float[]':
                if (Array.isArray(obj) && obj.every((item) => (typeof item === 'number' || item instanceof Number) && !isNaN(item))) {
                    return true;
                }
                if (pytorch.Utility.isInstance(obj, 'torch.Value') && pytorch.Utility.isInstance(obj.type(), 'torch.ListType') && (pytorch.Utility.isInstance(obj.type().getElementType(), 'torch.IntType') || pytorch.Utility.isInstance(obj.type().getElementType(), 'torch.FloatType'))) {
                    return true;
                }
                return false;
            case 'str':
                return obj === null || typeof obj === 'string' ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.StringType);
            case 'str[]':
                return (Array.isArray(obj) && obj.every((item) => item === null || typeof item === 'string')) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.StringType);
            case 'str[][]':
                return Array.isArray(obj) && obj.every((item) => Array.isArray(item) && item.every((item) => typeof item === 'string'));
            case 'Layout':
            case 'ScalarType':
            case 'MemoryFormat':
                return Number.isInteger(obj) || obj === null ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.IntType) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.OptionalType && obj.type().getElementType() instanceof torch.IntType);
            case 'Dimname':
                return obj === null || (typeof obj === 'string' || obj instanceof String);
            case 'Dimname[]':
                return Array.isArray(obj) && obj.every((item) => item === null || typeof item === 'string');
            case 'Device':
                return obj === null || obj === Object(obj);
            case 't[]':
                return Array.isArray(obj) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.OptionalType && obj.type().getElementType() instanceof torch.ListType);
            case 't':
                return true;
            case 'AnyEnumType':
                return false;
            case 'complex':
                return obj instanceof torch.Value && obj.type() instanceof torch.ComplexType;
            case 'Any':
                return true;
            case 'Any[]':
                if (Array.isArray(obj)) {
                    return true;
                }
                if (obj instanceof torch.Value && obj.type() instanceof torch.ListType) {
                    return true;
                }
                return false;
            case 't1':
            case 't2':
                return true;
            default: {
                if (type instanceof torch.ClassType &&
                    obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                    return type.qualified_name() === `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                }
                if (type instanceof torch.TupleType) {
                    throw new pytorch.Error('Not implemented.');
                    /*
                    if (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.TupleType) {
                        const elements = obj.type().getElementType().elements();
                        if (elements.length === 2) {
                            if (pytorch.Utility.toType(elements[0]) === match[1]) {
                                return true;
                            }
                        }
                    }
                    return false;
                    */
                }
                if (type instanceof torch.DictType) {
                    if (obj instanceof torch.Value && obj.type() instanceof torch.DictType) {
                        if ((type.getKeyType().kind() === 'VarType' || type.getKeyType().str() === obj.type().getKeyType().str()) ||
                            (type.getValueType().kind() === 'VarType' || type.getValueType().str() === obj.type().getValueType().str())) {
                            return true;
                        }
                    }
                    return false;
                }
                // throw new pytorch.Error(`Unknown type '${type}'.`);
                return true;
            }
        }
    }

    getType(value) { // rename
        const torch = this.torch;
        if (value === null || value === undefined) {
            return undefined;
        } else if (value === true || value === false) {
            return torch.BoolType.get();
        } else if (pytorch.Utility.isTensor(value)) {
            return torch.TensorType.get();
        } else if (typeof value === 'string') {
            return torch.StringType.get();
        } else if (Number(value) === value && value % 1 === 0) {
            return torch.IntType.get();
        } else if (Number(value) === value) {
            return torch.FloatType.get();
        } else if (Array.isArray(value) && value.every((item) => Number(item) === item && item % 1 === 0)) {
            return torch.ListType.get(torch.IntType.get());
        } else if (Array.isArray(value) && value.every((item) => Number(item) === item)) {
            return torch.ListType.get(torch.FloatType.get());
        } else if (value instanceof torch.Value) {
            return value.type();
        }
        const text = (JSON.stringify(value) || '(undefined)').substring(0, 10);
        throw new pytorch.Error(`Unsupported ops argument type '${text}'.`);
    }

    _overload(target, name, args, context) {
        const moduleName = pytorch.Utility.target(target);
        if (!moduleName) {
            return null;
        }
        const torch = this.torch;
        const type = name ? `${moduleName}.${name}` : moduleName;
        // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
        let op_name = null;
        if (type.startsWith('torch.')) {
            op_name = `aten::${type.substring(6)}`;
        } else if (type.startsWith('ops.')) {
            op_name = type.substring(4).replace('.', '::');
        } else if (type === 'int') {
            op_name = 'aten::Int';
        } else if (type === 'str') {
            op_name = 'aten::str';
        } else if (type === 'bool') {
            op_name = 'aten::Bool';
        } else if (type === 'float') {
            op_name = 'aten::Float';
        } else if (type === 'complex') {
            op_name = 'aten::Complex';
        }
        let overloads = null;
        let evalArgs = null;
        overloads = torch._C._jit_get_schemas_for_operator(op_name);
        if ((!overloads || overloads.length === 0) && type.startsWith('ops.') && !type.startsWith('ops.prim')) {
            const module = this.import(moduleName);
            if (!module || !module[name]) {
                const schema = new torch.FunctionSchema(op_name, null, [], [], false, false);
                for (let i = 0; i < args.length; i++) {
                    let argument = args[i];
                    let name = i.toString();
                    if (argument.type === '=' && argument.target && argument.target.type === 'id') {
                        name = this.expression(argument.target, context);
                        argument = argument.expression;
                    }
                    const obj = this.expression(argument, context);
                    const real_type = this.getType(obj);
                    schema.arguments.push(new torch.Argument(name, null, real_type, null, null, false, null));
                }
                const count = context.target.length > 0 ? context.target[context.target.length - 1].length : 0;
                for (let i = 0; i < count; i++) {
                    schema.returns.push(new torch.Argument('', null, null, null, null, false, null));
                }
                const op = new torch._C.Operator(schema);
                torch._C._get_registry().registerOperator(op);
                overloads = [schema];
            }
        }
        if (!overloads || overloads.length === 0) {
            if (type.startsWith('aten::') || type.startsWith('prim::')) {
                throw new pytorch.Error(`Unknown function '${type}'.`);
            }
            return null;
        }
        evalArgs = args.map((argument) => {
            if (argument.type === '=' && argument.target && argument.target.type === 'id') {
                argument = argument.expression;
            }
            return this.expression(argument, context);
        });
        const matches = [];
        for (const schema of overloads) {
            const parameters = schema.arguments || [];
            let next = false;
            let kwarg_only = false;
            let position = 0;
            let index = 0;
            while (position < evalArgs.length) {
                if (index >= parameters.length) {
                    next = !schema.name.startsWith('_caffe2::') && !schema.is_vararg;
                    break;
                }
                const arg = parameters[index];
                if (arg.kwarg_only) {
                    break;
                }
                index++;
                const v = evalArgs[position];
                let type = arg.real_type;
                let optional = false;
                if (type instanceof torch.OptionalType) {
                    type = type.getElementType();
                    optional = true;
                }
                if (optional === true &&
                    (type instanceof torch.FloatType || type instanceof torch.BoolType || type instanceof torch.IntType || type instanceof torch.ComplexType || type.kind() === 'ScalarTypeType' || type instanceof torch.DeviceObjType || type.kind() === 'LayoutKind') &&
                    v instanceof torch.Value && v.type() instanceof torch.NoneType) {
                    position++;
                } else if (!this.isType(v, type, arg.N) && v !== null) {
                    if (optional) {
                        continue;
                    }
                    next = true;
                    break;
                } else if (args[position].type === '=' && args[position].target.value !== arg.name) {
                    next = true;
                    break;
                } else {
                    position++;
                }
            }
            if (next) {
                continue;
            }
            if (args.every((arg, index) => index < position || (arg.type === '=' && arg.target && arg.target.type === 'id'))) {
                const params = new Map(parameters.slice(index).map((a) => [a.name, a]));
                while (position < args.length) {
                    const value = evalArgs[position];
                    const arg = params.get(args[position].target.value);
                    position++;
                    if (!arg) {
                        next = true;
                        break;
                    }
                    if (arg.kwarg_only) {
                        kwarg_only = true;
                    }
                    let type = arg.real_type;
                    let optional = false;
                    if (type instanceof torch.OptionalType) {
                        type = type.getElementType();
                        optional = true;
                    }
                    if (!this.isType(value, type, arg.N)) {
                        if (optional) {
                            continue;
                        }
                        next = true;
                        break;
                    }
                }
            }
            if (next) {
                continue;
            }
            if (position < evalArgs.length && !schema.is_vararg && !schema.name.startsWith('_caffe2::')) {
                continue;
            }
            if (!kwarg_only && parameters.slice(index).some((arg) => !arg.has_default_value())) {
                continue;
            }
            matches.push(schema);
        }
        if (matches.length > 1) {
            const keys = new Map([['IntType', 1], ['FloatType', 2], ['TensorType', 3], ['NumberType', 4]]);
            matches.sort((a, b) => {
                let keyA = keys.get(a.arguments[0].real_type.kind()) || 5;
                let keyB = keys.get(b.arguments[0].real_type.kind()) || 5;
                if (keyA === keyB && a.arguments.length > 1 && b.arguments.length > 1) {
                    keyA = keys.get(a.arguments[1].real_type.kind()) || 5;
                    keyB = keys.get(b.arguments[1].real_type.kind()) || 5;
                }
                return keyA - keyB;
            });
        }
        if (matches.length === 0) {
            throw new pytorch.Error(`Unknown function '${op_name}'.`);
        }
        return [matches[0], evalArgs];
    }

    block(statements, context) {
        if (!this.traceIf) {
            return super.block(statements, context);
        }
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

pytorch.Container.Package = class extends pytorch.Container {

    constructor(entries) {
        super();
        this.type = 'pytorch.package';
        this.entries = entries;
    }

    async read() {
        const execution = new python.Execution();
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
                if (obj.__class__.__name__ === 'Parameter') {
                    const data = obj.data;
                    if (typeof obj.__name__ === 'string') {
                        data.__name__ = obj.__name__;
                    }
                    return data;
                }
                return null;
            default:
                return null;
        }
    }

    static toType(type) {
        switch (type.kind()) {
            case 'OptionalType': return `${pytorch.Utility.toType(type.getElementType())}?`;
            case 'ListType': return `${pytorch.Utility.toType(type.getElementType())}[]`;
            case 'BoolType': return `boolean`;
            case 'IntType': return `int64`;
            case 'FloatType': return `float32`;
            case 'StringType': return `string`;
            case 'ComplexType': return `complex`;
            case 'NumberType': return `scalar`;
            case 'TensorType': return `tensor`;
            case 'TupleType': return `tuple<${type.elements().map((type) => pytorch.Utility.toType(type)).join(', ')}>`;
            case 'DictType': return `map<${pytorch.Utility.toType(type.getKeyType())}, ${pytorch.Utility.toType(type.getValueType())}>`;
            case 'DeviceObjType': return `device`;
            case 'SymIntType': return `SymInt`;
            case 'ScalarTypeType': return `ScalarType`;
            case 'MemoryFormat': return `MemoryFormat`;
            case 'Layout': return `Layout`;
            default: throw new pytorch.Error(`Unsupported type '${type.kind()}'.`);
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
        if (type && type !== 'builtins.dict' && type !== 'builtins.object' && type !== 'collections.OrderedDict' && type !== 'torch.nn.modules.module.Module' && type !== '__torch__.Module') {
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
                    if (number < buffers.length && buffers[number].data) {
                        const storage = buffers[number];
                        const data = storage.data && storage.data.peek ? storage.data.peek() : storage.data;
                        operand.data = data.slice(offset, operand_length);
                    }
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
                const index = item.name.indexOf('(');
                const key = index === -1 ? item.name : item.name.substring(0, index);
                this._types.set(key, item);
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

    register(execution) {
        const torch = execution.register('torch');
        const registry = torch._C._get_registry();
        const modules = new Set();
        for (const [name, type] of this._types) {
            if (name.indexOf('::') !== -1) {
                const schema = torch.FunctionSchema.parse(type.name);
                if (type.category) {
                    schema.category = type.category;
                }
                const op = new torch._C.Operator(schema);
                registry.registerOperator(op);
                modules.add(type.name.split('::')[0]);
            }
        }
        for (const module of modules) {
            const namespace = new torch._ops._OpNamespace(module);
            execution.register(`torch.ops.${module}`, namespace);
        }
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

export const Metadata = pytorch.Metadata;
export const ModelFactory = pytorch.ModelFactory;
