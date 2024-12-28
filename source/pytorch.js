
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
            const graph = new pytorch.Graph(target.execution, metadata, null, '', target.module);
            this.graphs.push(graph);
            delete target.execution;
        } else if (target.modules) {
            for (const [name, value] of target.modules) {
                const graph = new pytorch.Graph(target.execution, metadata, null, name, value);
                this.graphs.push(graph);
                delete target.execution;
            }
        }
    }
};

pytorch.Graph = class {

    constructor(execution, metadata, type, name, module) {
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
        const torch = execution ? execution.torch : null;
        if (torch && module instanceof torch.jit._script.RecursiveScriptModule && module.graph) {
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
            const param_node = graph.param_node();
            const self = param_node && param_node.outputs().length > 0 && param_node.outputs()[0].type() === module._c._type() ? param_node.outputs()[0] : null;
            if (self) {
                const getattr = (value) => {
                    if (value.value === undefined) {
                        const node = value.node();
                        if (node.kind() === 'prim::GetAttr') {
                            const [input] = node.inputs();
                            getattr(input);
                            if (input.value !== undefined) {
                                const name = node.s('name');
                                value.value = input.value.__getattr__(name);
                                value.identifier = input.identifier ? `${input.identifier}.${name}` : name;
                            }
                        }
                        if (node === param_node && value === param_node.outputs()[0]) {
                            value.value = module;
                            value.identifier = '';
                        }
                    }
                };
                for (const node of graph.nodes()) {
                    for (const input of node.inputs()) {
                        getattr(input, node);
                    }
                }
                const delattr = (value) => {
                    for (const use of Array.from(value.uses())) {
                        const node = use.user;
                        if (node.kind() === 'prim::GetAttr') {
                            for (const output of node.outputs()) {
                                delattr(output);
                            }
                            node.destroy();
                        }
                    }
                };
                delattr(param_node.outputs()[0], '');
            }
            for (const node of graph.nodes()) {
                if (node.kind() === 'prim::Constant') {
                    const kind = node.kindOf('value');
                    const value = node[kind]('value');
                    for (const output of node.outputs()) {
                        output.identifier = output.debugName();
                        output.value = value;
                    }
                    node.destroy();
                }
            }
            for (const node of graph.nodes()) {
                if (node.kind() === 'prim::TupleUnpack') {
                    const value = node.inputs()[0].value;
                    if (Array.isArray(value) && value.length === node.outputs().length && value.every((value) => typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean')) {
                        for (let i = 0; i < node.outputs().length; i++) {
                            const output = node.outputs()[i];
                            output.value = value[i];
                        }
                        node.destroy();
                    }
                }
            }
            for (const node of graph.nodes()) {
                if (node.kind() === 'prim::ListConstruct' && node.inputs().every((value) => typeof value.value === 'number' && typeof value.value === 'string' && typeof value.value === 'boolean')) {
                    node.outputs()[0].value = node.inputs().map((value) => value.value);
                    node.destroy();
                }
            }
            for (const v of graph.inputs()) {
                if (self.uses().length === 0 && v === self) {
                    continue;
                }
                const identifier = pytorch.Utility.unique(v);
                const name = v.debugName() || identifier;
                const value = values.map(identifier);
                this.inputs.push(new pytorch.Argument(name, [value]));
            }
            for (const value of graph.outputs()) {
                const identifier = pytorch.Utility.unique(value);
                this.outputs.push(new pytorch.Argument(identifier, [values.map(identifier)]));
            }
            for (const node of graph.nodes()) {
                if (node === graph.param_node() ||
                    node === graph.return_node()) {
                    continue;
                }
                if (node.kind() === 'prim::ListConstruct') {
                    if (node.outputs().length === 1 &&
                        node.outputs().every((output) => output.uses().length === 1) &&
                        node.inputs().every((input) => pytorch.Utility.isTensor(input.value) || input instanceof torch.Value)) {
                        continue;
                    }
                }
                this.nodes.push(new pytorch.Node(execution, metadata, null, null, node, initializers, values));
            }
        } else if (torch && module instanceof torch.export.exported_program.ExportedProgram && module.graph) {
            const exported_program = module;
            const graph = exported_program.graph;
            const inputs_to_parameters = exported_program.graph_signature.inputs_to_parameters();
            const inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers();
            const inputs_to_lifted_tensor_constants = exported_program.graph_signature.inputs_to_lifted_tensor_constants();
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
            for (const obj of graph.nodes) {
                if (obj.op === 'placeholder') {
                    if (inputs_to_parameters.has(obj.name)) {
                        const key = inputs_to_parameters.get(obj.name);
                        const parameter = exported_program.state_dict.get(key);
                        if (parameter) {
                            const tensor = new pytorch.Tensor(key, parameter.data);
                            const value = new pytorch.Value(key, null, null, tensor);
                            values.set(obj, value);
                        }
                    } else if (inputs_to_buffers.has(obj.name)) {
                        const key = inputs_to_buffers.get(obj.name);
                        const buffer = exported_program.state_dict.get(key);
                        if (buffer) {
                            const tensor = new pytorch.Tensor(key, buffer);
                            const value = new pytorch.Value(key, null, null, tensor);
                            values.set(obj, value);
                        }
                    } else if (inputs_to_lifted_tensor_constants.has(obj.name)) {
                        const key = inputs_to_lifted_tensor_constants.get(obj.name);
                        const constant = exported_program.constants.get(key);
                        if (exported_program) {
                            const tensor = new pytorch.Tensor(key, constant);
                            const value = new pytorch.Value(key, null, null, tensor);
                            values.set(obj, value);
                        }
                    }
                    if (obj.users.size > 1 && values.has(obj)) {
                        const node = new pytorch.Node(execution, metadata, obj.name, null, obj, null, values);
                        this.nodes.push(node);
                        values.set(obj, node.outputs[0].value[0]);
                    }
                }
            }
            for (const obj of graph.nodes) {
                if (obj.op === 'placeholder') {
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
                const node = new pytorch.Node(execution, metadata, obj.name, null, obj, null, values);
                this.nodes.push(node);
            }
            for (const input_spec of exported_program.graph_signature.user_inputs()) {
                if (nodes.has(input_spec)) {
                    const node = nodes.get(input_spec);
                    const value = values.map(node);
                    const argument = new pytorch.Argument(input_spec, [value]);
                    this.inputs.push(argument);
                }
            }
        } else if (pytorch.Utility.isTensor(module)) {
            const node = new pytorch.Node(execution, metadata, null, type, { value: module });
            this.nodes.push(node);
        } else {
            const weights = this.type === 'weights' ? module : pytorch.Utility.weights(module);
            if (weights) {
                this.name = !this.name && typeof module.__name__ === 'string' ? module.__name__ : this.name;
                for (const [name, module] of weights) {
                    const node = new pytorch.Node(execution, metadata, name, 'Weights', module);
                    this.nodes.push(node);
                }
            } else {
                const modules = Array.isArray(module) && module.every((module) => module && !pytorch.Utility.isTensor(module) && (module._modules !== undefined || module.__class__)) ? module : [module];
                for (const module of modules) {
                    const type = this.type === 'weights' ? 'Weights' : null;
                    const node = new pytorch.Node(execution, metadata, null, type, module, null, values);
                    this.nodes.push(node);
                }
            }
        }
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

    constructor(execution, metadata, name, type, obj, initializers, values, stack) {
        const torch = execution ? execution.torch : null;
        this.name = name || '';
        this.nodes = [];
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        this.metadata = [];
        if (torch && obj instanceof torch.Node) {
            const node = obj;
            const kind = node.kind();
            const schema = node.schema();
            const inputs = node.inputs();
            const outputs = node.outputs();
            this.type = {
                name: kind.indexOf('::') === -1 ? kind : kind.split('::').pop().split('.')[0],
                identifier: kind
            };
            if (schema && schema.category) {
                this.type.category = schema.category;
            }
            const getAttribute = (node, name) => {
                const kind = node.kindOf(name);
                let value = null;
                let type = null;
                switch (kind) {
                    case 's': value = node.s(name); type = 'string'; break;
                    case 'i': value = node.i(name); type = 'int64'; break;
                    case 'f': value = node.f(name); type = 'float32'; break;
                    case 't': value = node.t(name); type = 'tensor'; break;
                    case 'ss': value = node.ss(name); type = 'string[]'; break;
                    case 'tys': value = node.tys(name).map((ty) => pytorch.Utility.toType(ty)); type = 'type[]'; break;
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
            const mapTensor = (value) => {
                if (value.identifier && pytorch.Utility.isTensor(value.value)) {
                    const identifier = value.identifier;
                    if (!values.has(identifier)) {
                        const tensor = new pytorch.Tensor(identifier, value.value);
                        values.set(identifier, new pytorch.Value(identifier, null, null, tensor));
                    }
                    return values.map(identifier);
                }
                let initializer = null;
                let identifier = value.hasDebugName() ? `%${value.debugName().toString()}` : `%${value.unique().toString()}`;
                if (value.value) {
                    const obj = value.value;
                    const hide = obj.__parent__ ? obj.__parent__.__hide__ : true;
                    initializer = hide ? initializers.get(obj) : null;
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
                if (type instanceof torch.ListType) {
                    array = true;
                    type = type.getElementType();
                }
                let argument = null;
                if (type && type instanceof torch.ClassType) {
                    const obj = input.value;
                    if (!array && initializers.has(obj)) {
                        const node = new pytorch.Node(execution, metadata, name, type.qualified_name(), obj, initializers, values);
                        argument = new pytorch.Argument(name, node, 'object');
                    } else if (array && Array.isArray(obj) && obj.every((obj) => initializers.has(obj))) {
                        const node = obj.map((obj) => new pytorch.Node(execution, metadata, name, type.qualified_name(), obj, initializers, values));
                        argument = new pytorch.Argument(name, node, 'object[]');
                    } else if (array && input.node().kind() === 'prim::ListConstruct' && input.uses().length === 1 && input.node().inputs().every((input) => input.value)) {
                        const node = input.node().inputs().map((input) => new pytorch.Node(execution, metadata, name, null, input.value, initializers, values));
                        argument = new pytorch.Argument(name, node, 'object[]');
                    } else if (input.value === undefined) {
                        const identifier = pytorch.Utility.unique(input);
                        const value = values.map(identifier);
                        argument = new pytorch.Argument(name, [value]);
                    } else {
                        const node = new pytorch.Node(execution, metadata, null, null, input.value, initializers, values);
                        argument = new pytorch.Argument(name, node, 'object');
                    }
                } else if ((input.type() instanceof torch.TensorType || (input.type() instanceof torch.OptionalType && input.type().getElementType() instanceof torch.TensorType)) && pytorch.Utility.isTensor(input.value)) {
                    const value = mapTensor(input);
                    argument = new pytorch.Argument(name, [value]);
                } else if (input instanceof torch.Value && !pytorch.Utility.isTensor(input.value)) {
                    if (input.node() === null && input.value !== undefined) {
                        if (Array.isArray(input.value) && input.value.every((value) => pytorch.Utility.isTensor(value))) {
                            continue;
                        }
                        const type = input.type() ? pytorch.Utility.toType(input.type()) : null;
                        argument = new pytorch.Argument(name, input.value, type || 'attribute');
                    } else if (input.type() instanceof torch.ListType) {
                        if (input.node() && input.node().kind() === 'prim::ListConstruct' && input.uses().length === 1 &&
                        input.node().inputs().every((value) => value instanceof torch.Value || value.type() instanceof torch.IntType || value.type() instanceof torch.FloatType || value.type() instanceof torch.StringType || value.type() instanceof torch.ComplexType || value.type() instanceof torch.TensorType)) {
                            const list = input.node().inputs();
                            const args = list.map((value) => {
                                if (pytorch.Utility.isTensor(value.value)) {
                                    return mapTensor(value);
                                }
                                if (value.uses().length === 1 && value.value !== undefined) {
                                    return value.value;
                                }
                                const identifier = pytorch.Utility.unique(value);
                                return values.map(identifier);
                            });
                            const type = list.every((value) => (pytorch.Utility.isTensor(value.value)) || value.value === null) ? null : pytorch.Utility.toType(input.type());
                            argument = new pytorch.Argument(name, args, type);
                        } else {
                            const identifier = pytorch.Utility.unique(input);
                            argument = new pytorch.Argument(name, [values.map(identifier)]);
                        }
                    } else if (input.type() instanceof torch.StringType && typeof input.value === 'string') {
                        argument = new pytorch.Argument(name, input.value, 'string');
                    } else if (input.type() instanceof torch.BoolType && (typeof input.value === 'boolean' || input.value === 0 || input.value === 1)) {
                        argument = new pytorch.Argument(name, Boolean(input.value), 'boolean');
                    } else if (input.type() instanceof torch.IntType && typeof input.value === 'number') {
                        argument = new pytorch.Argument(name, input.value, 'int64');
                    } else if (input.type() instanceof torch.FloatType && typeof input.value === 'number') {
                        argument = new pytorch.Argument(name, input.value, 'float32');
                    } else if (input.type() instanceof torch.NoneType && input.value === null) {
                        argument = new pytorch.Argument(name, null, 'attribute');
                    } else {
                        const identifier = pytorch.Utility.unique(input);
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
                        let identifier = pytorch.Utility.unique(input);
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
                } else if (Array.isArray(input.value) && input.value.some((value) => value instanceof torch.Value)) {
                    const args = input.value.map((value) => {
                        if (value instanceof torch.Value) {
                            const identifier = pytorch.Utility.unique(value);
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
                const args = list.map((output) => values.map(pytorch.Utility.unique(output)));
                const argument = new pytorch.Argument(name, args);
                this.outputs.push(argument);
            }
            const blocks = node.blocks();
            for (let i = 0; i < blocks.length; i++) {
                const block = blocks[i];
                if (block.nodes().length > 2) {
                    const name = `block${i.toString()}`;
                    const graph = { name: '', nodes: [] }; // new pytorch.Graph(execution, metadata, null, name, blocks[i]);
                    const argument = new pytorch.Argument(name, graph, 'graph');
                    this.inputs.push(argument);
                }
            }
            const sourceRange = node.sourceRange();
            if (sourceRange) {
                this.metadata.push(new pytorch.Argument('source', sourceRange.replace(/^at\s/, '').replace(/\.$/, '')));
            }
        } else if (torch && obj instanceof torch.fx.node.Node) {
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
                    if (arg instanceof torch.fx.node.Node) {
                        const value = values.map(arg);
                        const argument = new pytorch.Argument(name, [value]);
                        this.inputs.push(argument);
                    } else if (Array.isArray(arg) && arg.every((arg) => arg instanceof torch.fx.node.Node || arg === null)) {
                        const list = arg.map((arg) => arg === null ? null : values.map(arg));
                        const argument = new pytorch.Argument(name, list);
                        this.inputs.push(argument);
                    } else if (Array.isArray(arg)) {
                        const list = arg.map((arg) => arg instanceof torch.fx.node.Node ? values.map(arg) : arg);
                        const argument = new pytorch.Argument(name, list, type || 'attribute');
                        this.inputs.push(argument);
                    } else if (arg instanceof torch.dtype || arg instanceof torch.device || arg instanceof torch.layout || arg instanceof torch.memory_format) {
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
                {
                    const value = values.map(obj);
                    const argument = new pytorch.Argument('value', [value]);
                    this.inputs.push(argument);
                }
                {
                    const value = values.map({ name: obj.name, meta: obj.meta });
                    const argument = new pytorch.Argument('value', [value]);
                    this.outputs.push(argument);
                }
            } else if (obj.op === 'root') {
                this.type = { name: obj.op };
            } else {
                throw new pytorch.Error(`Unsupported node operation '${obj.op}'.`);
            }
        } else {
            if (torch && obj instanceof torch.ScriptObject) {
                type = obj._type().qualified_name();
                obj = obj._ivalue;
            } else if (torch && obj instanceof torch.jit._script.RecursiveScriptModule && obj._c && obj._c.qualified_name) {
                type = obj._c._type();
                const target = {
                    _modules: obj._modules,
                    _parameters: obj._parameters,
                    _buffers: obj._buffers,
                };
                for (let i = 0; i < type.numAttributes(); i++) {
                    if (!type.is_parameter(i) && !type.is_buffer(i) && !type.getAttribute(i).is_module()) {
                        const k = type.getAttributeName(i);
                        target[k] = obj.__getattr__(k);
                    }
                }
                type = obj._c.qualified_name;
                obj = target;
            }
            if (!type) {
                if (torch && obj instanceof torch.jit._script.RecursiveScriptModule && obj._c && obj._c.qualified_name) {
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
            }
            stack = stack || new Set();
            const weights = pytorch.Utility.weights(obj);
            if (weights) {
                const type = this.type.name;
                this.type = new pytorch.Graph(execution, metadata, 'weights', '', weights);
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
                    if (name === '__class__' || name === '__name__') {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'collections.OrderedDict') && value instanceof Map && value.size === 0) {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'builtins.set') && value instanceof Set && value.size === 0) {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'builtins.list') && Array.isArray(value) && value.length === 0) {
                        continue;
                    } else if (pytorch.Utility.isInstance(value, 'torch.Size') && Array.isArray(value) && value.length === 0) {
                        continue;
                    }
                    let parameters = null;
                    if ((name === '_parameters' || name === '_buffers') && value instanceof Map) {
                        parameters = value;
                    } else if (pytorch.Utility.isTensor(value) || (Array.isArray(value) && value.every((tensor) => pytorch.Utility.isTensor(tensor)))) {
                        parameters = new Map([[name, value]]);
                    }
                    if (parameters) {
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
                    if (pytorch.Utility.isTensor(value)) {
                        const tensor = new pytorch.Tensor('', value);
                        const argument = new pytorch.Argument(name, tensor, 'tensor');
                        this.inputs.push(argument);
                    } else if (value && pytorch.Utility.isInstance(value, 'torch.dtype')) {
                        const node = new pytorch.Node(execution, metadata, null, value.toString(), {});
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
                        const list = Array.from(value).filter(([, value]) => !stack.has(value)).map(([name, obj]) => {
                            stack.add(value);
                            const type = obj === null ? 'builtins.NoneType' : `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                            const node = new pytorch.Node(execution, metadata, this.name ? `${this.name}.${name}` : name, type, obj, initializers, values, stack);
                            stack.delete(value);
                            return node;
                        });
                        const argument = new pytorch.Argument(name, list, 'object[]');
                        this.inputs.push(argument);
                    } else if (value && Array.isArray(value) && value.length > 0 && value.every((obj) => Array.isArray(obj) && obj.every((item) => typeof item === 'string' || typeof item === 'number'))) {
                        const argument = new pytorch.Argument(name, value, 'attribute');
                        this.inputs.push(argument);
                    } else if (value && Array.isArray(value) && value.length > 0 && value.every((obj) => obj && (obj.__class__ || obj === Object(obj)))) {
                        const list = value.filter((value) => !stack.has(value));
                        const nodes = list.map((value) => {
                            stack.add(value);
                            const node = new pytorch.Node(execution, metadata, null, null, value, initializers, values, stack);
                            stack.delete(value);
                            return node;
                        });
                        const argument = new pytorch.Argument(name, nodes, 'object[]');
                        this.inputs.push(argument);
                    } else if (value && (value.__class__ || typeof value === 'object') && !stack.has(value)) {
                        stack.add(value);
                        const node = new pytorch.Node(execution, metadata, null, null, value, initializers, values, stack);
                        stack.delete(value);
                        const visible = name !== '_metadata' || !pytorch.Utility.isMetadataObject(value);
                        const argument = new pytorch.Argument(name, node, 'object', visible);
                        this.inputs.push(argument);
                    } else {
                        let schema = metadata.attribute(this.type.identifier, name);
                        schema = name === 'training' ? { type: 'boolean', visible: false } : schema;
                        let visible = true;
                        let obj = value;
                        const type = schema && schema.type ? schema.type : 'attribute';
                        if (schema) {
                            if (schema.visible === false) {
                                visible = false;
                            } else if (schema.default !== undefined) {
                                if (Array.isArray(obj)) {
                                    if (Array.isArray(schema.default)) {
                                        visible = obj.length !== schema.default || !obj.every((item, index) => item === schema.default[index]);
                                    } else {
                                        visible = !obj.every((item) => item === schema.default);
                                    }
                                } else {
                                    visible = obj !== schema.default;
                                }
                            }
                        }
                        if (Array.isArray(obj) && obj.length > 0 && obj.every((obj) => obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('torch.nn'))) {
                            obj = '?';
                        }
                        const argument = new pytorch.Argument(name, obj, type, visible);
                        this.inputs.push(argument);
                    }
                }
            }
        }
    }
};

pytorch.Tensor = class {

    constructor(name, tensor) {
        this.name = name || '';
        tensor = tensor.data ? tensor.data : tensor;
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
            pytorch.Container.IR,
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
        // https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
        this._entries = entries;
    }

    async read(metadata) {
        this.execution = new pytorch.Execution(null, metadata);
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        const torch = this.execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this._entries);
        let torchscript = reader.has_record('constants.pkl');
        const version = reader.version();
        if (torchscript) {
            this.execution.trace = false;
            this.module = torch.jit.load(reader);
            this.execution.trace = true;
            metadata.register(this.execution);
            torchscript = this.module.forward;
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
        pytorch.proto = await this._context.require('./pytorch-proto');
        const keys = [
            'attributes.pkl',
            'version',
            ...this._model.tensors.filter((tensor) => tensor && tensor.data && tensor.data.key).map((tensor) => tensor.data.key)
        ];
        const walk = (module) => {
            if (module.torchscriptArena && module.torchscriptArena.key) {
                keys.push(module.torchscriptArena.key);
            }
            for (const submodule of module.submodules || []) {
                walk(submodule);
            }
        };
        walk(this._model.mainModule);
        const values = await Promise.all(keys.map((name) => this._context.fetch(name).then((context) => context.stream).catch(() => null)));
        for (let i = 0; i < keys.length; i++) {
            if (values[i]) {
                this._entries.set(keys[i], values[i]);
            }
        }
        this.execution = new pytorch.Execution(null, metadata);
        this.execution.proto = pytorch.proto;
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        const torch = this.execution.__import__('torch');
        const reader = new torch.PyTorchFileReader(this._entries);
        if (this._model && this._model.producerName) {
            this.producer = this._model.producerName + (this._model.producerVersion ? ` v${this._model.producerVersion}` : '');
        }
        this.format = reader.has_record('attributes.pkl') ? 'TorchScript v1.1' : 'TorchScript v1.0';
        this.execution.trace = false;
        this.module = torch.jit.load(reader);
        this.execution.trace = true;
        metadata.register(this.execution);
        delete this._context;
        delete this._model;
        delete this._entries;
    }
};

pytorch.Container.IR = class extends pytorch.Container {

    static open(context) {
        const reader = context.read('text', 0x100);
        if (reader && reader.length > 0) {
            const line = reader.read('\n');
            if (line.startsWith('graph(')) {
                return new pytorch.Container.IR(context);
            }
        }
        return null;
    }

    constructor(context) {
        super();
        this.type = 'pytorch.ir';
        this.context = context;
    }

    async read(metadata) {
        this.format = 'TorchScript IR';
        this.execution = new pytorch.Execution(null, metadata);
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        // this.execution.graph;
        // context reader = context.read('text', 0x100);
        throw new pytorch.Error('TorchScript IR parser not implemented.');
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
        this.execution = new pytorch.Execution(null, metadata);
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        const torch = this.execution.__import__('torch');
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
        this.execution = new python.Execution();
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        metadata.register(this.execution);
        const torch = this.execution.__import__('torch');
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
                const node = execution.graph.create(type, 0);
                execution.graph.insertNode(node);
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
        this._constants = new Map();
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

    target(expr, context) {
        const ast = this.ast;
        const torch = this.torch;
        if (expr instanceof ast.Name) {
            switch (expr.id) {
                case 'torch':
                case 'ops':
                case 'uninitialized':
                    return this.builtins[expr.id];
                case 'CONSTANTS': {
                    if (!this._constants) {
                        const value = this.builtins[expr.id];
                        const entries = Object.entries(value).map(([name, value]) => {
                            if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === 'string')) {
                                value = this._graph.insertConstant(value);
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
        let current = expr;
        let path = [];
        for (;;) {
            if (current instanceof ast.Attribute) {
                path.push(current.attr);
                current = current.value;
            } else if (current instanceof ast.Name && current.id !== 'self' && current.id !== 'CONSTANTS') {
                path.push(current.id);
                break;
            } else {
                path = null;
                break;
            }
        }
        if (path) {
            let target = null;
            for (let i = path.length - 1; i >= 0; i--) {
                const name = path[i];
                if (target) {
                    if (target instanceof torch.Value && target.type() instanceof torch.ClassType) {
                        const node = this._graph.createGetAttr(target, name);
                        this._graph.insertNode(node);
                        target = node.output();
                    } else {
                        target = target.__getattr__ ? target.__getattr__(name) : target[name];
                    }
                } else {
                    target = context.get(name);
                }
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
            if (target instanceof torch.Value) {
                return target;
            }
        }
        return super.target(expr, context);
    }

    create(kind, loc, n_outputs) {
        return this._graph.create(kind, n_outputs).setSourceRange(loc);
    }

    value(expr, context, typehint) {
        const torch = this.torch;
        const value = this.expression(expr, context, typehint);
        if (value instanceof torch.Value) {
            return value;
        }
        return this.constant(value);
    }

    expression(expr, context, typehint) {
        if (!this.trace) {
            return super.expression(expr, context);
        }
        const ast = this.ast;
        const torch = this.torch;
        switch (expr.__class__.__name__) {
            case 'Name': {
                if (this.traceAttr && expr.id === 'self') {
                    return context.get('self');
                }
                break;
            }
            case 'Constant': {
                if (expr.value === true || expr.value === false) {
                    // debugger;
                    return this._graph.insertConstant(expr.value);
                }
                break;
            }
            case 'Assign': {
                const target = expr.targets;
                if (target instanceof ast.Name) {
                    let value = this.expression(expr.value, context);
                    if (typeof value === 'string' || typeof value === 'boolean' || typeof value === 'number') {
                        value = this.constant(value);
                    } else if (typeof value !== 'object' && value !== undefined) {
                        throw new pytorch.Error(`Unsupported assignment value type '${typeof value}'.`);
                    }
                    if (value instanceof torch.Value && !value.hasDebugName()) {
                        value.setDebugName(target.value);
                    }
                    context.set(target.id, value);
                    return undefined;
                } else if (target instanceof ast.Tuple) {
                    context.target.push(target.elts);
                    const value = this.expression(expr.value, context);
                    context.target.pop();
                    if (target.elts.every((item) => item instanceof ast.Name)) {
                        if (value instanceof torch.Value) {
                            let outputs = null;
                            if (value.type() instanceof torch.TupleType) {
                                const node = this._graph.createTupleUnpack(value);
                                node.setSourceRange(expr.location);
                                this._graph.insertNode(node);
                                outputs = node.outputs();
                            } else if (value.type() instanceof torch.ListType) {
                                const size = target.elts.length;
                                const node = this._graph.createListUnpack(value, size);
                                node.setSourceRange(expr.location);
                                this._graph.insertNode(node);
                                outputs = node.outputs();
                            }
                            if (outputs === null) {
                                throw new pytorch.Error(`Unsupported unpack type '${value.type().kind()}'.`);
                            }
                            for (let i = 0; i < target.elts.length; i++) {
                                const item = target.elts[i];
                                const output = outputs[i];
                                output.setDebugName(item.id);
                                context.set(item.id, output);
                            }
                            return outputs;
                        }
                        const elts = value;
                        if (target.elts.length < elts.length) {
                            throw new pytorch.Error(`ValueError: too many values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        if (target.elts.length > elts.length) {
                            throw new pytorch.Error(`ValueError: not enough values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        for (let i = 0; i < elts.length; i++) {
                            const value = elts[i];
                            const name = target.elts[i].id;
                            if (value instanceof torch.Value && !value.hasDebugName()) {
                                value.setDebugName(name);
                            }
                            context.set(name, value);
                        }
                        return undefined;
                    }
                }
                break;
            }
            case 'Call': {
                const func = expr.func;
                if (func instanceof ast.Name && func.id === 'annotate') {
                    const type = this.type(expr.args[0]);
                    const [, obj] = expr.args;
                    let value = this.expression(obj, context, type);
                    if (value instanceof torch.Tensor ||
                        (value instanceof torch.Value && value.type() instanceof torch.TensorType)) {
                        let name = null;
                        if (type instanceof torch.IntType) {
                            name = 'IntImplicit';
                        } else if (type instanceof torch.FloatType) {
                            name = 'FloatImplicit';
                        } else if (type instanceof torch.StringType) {
                            name = 'StringImplicit';
                        } else if (type instanceof torch.ComplexType) {
                            name = 'ComplexImplicit';
                        } else if (type instanceof torch.NumberType) {
                            name = 'ScalarImplicit';
                        } else {
                            throw new pytorch.Error(`Unsupported annotation type '${type.kind()}'.`);
                        }
                        const ast = this.ast;
                        const target = new ast.Name('torch');
                        return this.call(target, name, expr.args.slice(1), context);
                    }
                    if (value instanceof torch.Value && !type.equals(value.type())) {
                        throw new pytorch.Error('Invalid annotation type hint.');
                    }
                    if (value === null) {
                        value = this._graph.insertConstant(value);
                        value.setType(type);
                    }
                    return value;
                }
                if (func instanceof ast.Name && func.id === 'uninitialized') {
                    const type = this.type(expr.args[0]);
                    const node = this._graph.createUninitialized(type);
                    node.setSourceRange(expr.location);
                    this._graph.insertNode(node);
                    return node.output();
                }
                if (func instanceof ast.Name && func.id === 'unchecked_cast') {
                    const value = this.value(expr.args[1], context);
                    const type = this.type(expr.args[0]);
                    return this._graph.insertUncheckedCast(value, type);
                }
                if (func instanceof ast.Name && func.id === 'isinstance') {
                    const value = this.value(expr.args[0], context);
                    let [, types] = expr.args;
                    if (types instanceof ast.Tuple) {
                        types = types.elts.map((expr) => this.type(expr));
                    } else {
                        types = [this.type(types)];
                    }
                    const node = this._graph.createIsInstance(value, types);
                    this._graph.insertNode(node);
                    return node.output();
                }
                if (func.attr === 'tolist' && expr.args.length === 0) {
                    const target = this.target(func.value, context);
                    return this._graph.insertToList(target, typehint);
                }
                if (this.traceAttr) {
                    if (func instanceof ast.Name && func.id === 'getattr') {
                        const obj = this.expression(expr.args[0], context);
                        const field = this.expression(expr.args[1], context);
                        const n = this._graph.createGetAttr(obj, field);
                        this._graph.insertNode(n);
                        return n.output();
                    }
                }
                return super.expression(expr, context);
            }
            case 'Subscript': {
                if (expr.slice instanceof ast.List && expr.slice.elts.length === 1) {
                    const value = this.expression(expr.value, context);
                    const [elt] = expr.slice.elts;
                    if (value instanceof torch.Value) {
                        let type = value.type();
                        if (type instanceof torch.OptionalType) {
                            type = type.getElementType();
                        }
                        if (type instanceof torch.ListType) {
                            let index = this.expression(elt, context);
                            if (Number.isInteger(index)) {
                                index = this._graph.insertConstant(index);
                            }
                            const node = this._graph.create('aten::__getitem__.t', [value, index]);
                            this._graph.insertNode(node);
                            node.output().setType(type.getElementType());
                            return node.output();
                        }
                        if (type instanceof torch.DictType) {
                            let key = this.expression(elt, context);
                            const node = this._graph.create('aten::__getitem__.t', [value]);
                            this._graph.insertNode(node);
                            if (type.getKeyType() instanceof torch.StringType && typeof key === 'string') {
                                const value = new torch.Value(node);
                                value.value = key;
                                key = value;
                            } else if (type.getKeyType() instanceof torch.StringType && key.type() instanceof torch.StringType) {
                                // continue
                            } else {
                                throw new pytorch.Error(`Unsupported dictionary key type.`);
                            }
                            node.addInput(key);
                            node.output().setType(type.getValueType());
                            return node.output();
                        }
                        if (type instanceof torch.TupleType) {
                            let index = this.expression(elt, context);
                            if (!Number.isInteger(index)) {
                                throw new pytorch.Error(`Unsupported tuple index type.`);
                            }
                            const output_type = type.elements()[index];
                            index = this._graph.insertConstant(index);
                            const node = this._graph.createTupleIndex(value, index, output_type);
                            this._graph.insertNode(node);
                            return node.output();
                        }
                    }
                }
                break;
            }
            case 'Attribute': {
                if (expr.value instanceof ast.Name && expr.value.id === 'CONSTANTS') {
                    const constant = this.builtins[expr.value.id][expr.attr];
                    const value = this._graph.insertConstant(constant);
                    value.setDebugName(`${expr.value.id}.${expr.attr}`);
                    return value;
                }
                const target = this.target(expr.value, context);
                const attr = expr.attr;
                if (target instanceof torch.Value && target.type() instanceof torch.ClassType) {
                    const node = this._graph.createGetAttr(target, attr);
                    this._graph.insertNode(node);
                    return node.output();
                }
                return target.__getattr__ ? target.__getattr__(attr) : target[attr];
            }
            case 'List': {
                const list = expr.elts.map((item) => this.expression(item, context));
                if (/* list.length > 0 && */ list.every((item) => item instanceof torch.Value || pytorch.Utility.isTensor(item) || typeof item === 'number' || typeof item === 'string' || item === null)) {
                    const values = [];
                    let item_type = null;
                    for (const item of list) {
                        let value = null;
                        if (item instanceof torch.Value) {
                            value = item;
                        } else if (typeof item === 'number' || typeof item === 'string' || item === null) {
                            value = this._graph.insertConstant(item);
                        } else if (pytorch.Utility.isTensor(item)) {
                            value = item;
                        } else {
                            throw new pytorch.Error('Unsupported list item type.');
                        }
                        values.push(value);
                        const type = value.type();
                        if (!item_type || item_type.isSubtypeOf(type)) {
                            item_type = type;
                        }
                    }
                    const contained_type = typehint ? typehint.getElementType() : item_type;
                    const node = this._graph.createList(contained_type, values);
                    this._graph.insertNode(node);
                    return node.output();
                }
                break;
            }
            case 'Tuple': {
                const elts = expr.elts.map((expr) => this.expression(expr, context));
                const values = [];
                for (const elt of elts) {
                    let value = null;
                    if (elt instanceof torch.Value) {
                        value = elt;
                    } else if (pytorch.Utility.isTensor(elt)) {
                        throw new pytorch.Error();
                    } else if (elt === null || Number.isInteger(elt) || typeof elt === 'number' || typeof elt === 'boolean' || typeof elt === 'string') {
                        value = this._graph.insertConstant(elt);
                    } else {
                        throw new pytorch.Error('Unsupported tuple element.');
                    }
                    values.push(value);
                }
                const node = this._graph.createTuple(values);
                node.setSourceRange(expr.location);
                this._graph.insertNode(node);
                return node.output();
            }
            case 'Dict': {
                const keys = [];
                const values = [];
                let keyType = null;
                let valueType = null;
                for (let i = 0; i < expr.keys.length; i++) {
                    const key = this.value(expr.keys[i], context);
                    if (!keyType || keyType.isSubtypeOf(key.type())) {
                        keyType = key.type();
                    }
                    keys.push(key);
                    const value = this.value(expr.values[i], context);
                    if (!valueType || valueType.isSubtypeOf(value.type())) {
                        valueType = value.type();
                    }
                    values.push(value);
                }
                const key_type = typehint ? typehint.getKeyType() : keyType;
                const value_type = typehint ? typehint.getValueType() : valueType;
                const node = this._graph.createDict(key_type, value_type, keys, values);
                this._graph.insertNode(node);
                return node.output();
            }
            default: {
                break;
            }
        }
        return super.expression(expr, context);
    }

    static(expr, context, state) {
        const ast = this.ast;
        const builtins = this.builtins;
        const torch = this.torch;
        switch (expr.__class__.__name__) {
            case 'Name': {
                const value = context.get(expr.id);
                if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') {
                    return value;
                }
                if (value instanceof torch.Tensor && value.storage() && value.storage().size() !== undefined) {
                    return value;
                }
                if (value instanceof Map) {
                    return value;
                }
                if (value instanceof torch.Value) {
                    const node = value.node();
                    if (node.kind() === 'prim::Constant') {
                        state.push(node);
                        return pytorch.Utility.constant(node, 'value');
                    } else if (node.kind() === 'prim::ListConstruct' && node.inputs().every((value) => value instanceof torch.Value && value.node().kind() === 'prim::Constant')) {
                        state.push(node);
                        for (const value of node.inputs()) {
                            state.push(value.node());
                        }
                        return node.inputs().map((value) => pytorch.Utility.constant(value.node(), 'value'));
                    } else if (node.kind() === 'prim::TupleUnpack') {
                        const index = node.outputs().indexOf(value);
                        const input = node.inputs()[0].node();
                        if (input.kind() === 'prim::TupleConstruct') {
                            const value = input.inputs()[index];
                            const constant = value.node();
                            if (constant.kind() === 'prim::Constant') {
                                state.push(node);
                                state.push(constant);
                                return pytorch.Utility.constant(constant, 'value');
                            }
                        }
                    }
                    state.splice(0, state.length);
                }
                break;
            }
            case 'List': {
                return expr.elts.map((expr) => this.static(expr, context, state));
            }
            case 'Constant': {
                return expr.value;
            }
            case 'Attribute': {
                const target = this.target(expr.value, context);
                return target.__getattr__ ? target.__getattr__(expr.attr) : target[expr.attr];
            }
            case 'Call': {
                const func = expr.func;
                if (func instanceof ast.Name && func.id === 'annotate') {
                    return this.static(expr.args[1], context, state);
                }
                const args = expr.args.map((expression) => this.static(expression, context, state));
                if (args.every((arg) => arg !== undefined)) {
                    if (func instanceof ast.Attribute && func.value instanceof ast.Name && func.value.id === 'torch') {
                        const target = this.target(func, context);
                        if (typeof target === 'function') {
                            if (target && target.__class__ === builtins.type) {
                                // debugger;
                            } else {
                                return target(...args);
                            }
                        }
                    }
                }
                state.splice(0, state.length);
                break;
            }
            default: {
                break;
            }
        }
        return undefined;
    }

    variables(value, scope) {
        if (!scope.refs) {
            scope.refs = new Set();
        }
        switch (value.__class__.__name__) {
            case 'Assign': {
                this.variables(value.targets, scope);
                this.variables(value.value, scope);
                break;
            }
            case 'AnnAssign': {
                this.variables(value.value, scope);
                break;
            }
            case 'Attribute': {
                this.variables(value.value, scope);
                break;
            }
            case 'Name': {
                scope.refs.add(value.id);
                break;
            }
            case 'Import':
            case 'Constant': {
                break;
            }
            case 'List': {
                for (const item of value.elts) {
                    this.variables(item, scope);
                }
                break;
            }
            case 'Dict': {
                for (let i = 0; i < value.keys.length; i++) {
                    this.variables(value.keys[i], scope);
                    this.variables(value.values[i], scope);
                }
                break;
            }
            case 'Tuple': {
                for (const item of value.elts) {
                    this.variables(item, scope);
                }
                break;
            }
            case 'pair': {
                this.variables(value.key, scope);
                this.variables(value.value, scope);
                break;
            }
            case 'Subscript': {
                this.variables(value.value, scope);
                this.variables(value.slice, scope);
                break;
            }
            case 'Call': {
                this.variables(value.func, scope);
                for (const arg of value.args) {
                    this.variables(arg, scope);
                }
                break;
            }
            case 'If': {
                this.variables(value.test, scope);
                for (const stmt of value.body) {
                    this.variables(stmt, scope);
                }
                for (const stmt of value.orelse) {
                    this.variables(stmt, scope);
                }
                break;
            }
            case 'For': {
                this.variables(value.target, scope);
                this.variables(value.iter, scope);
                for (const stmt of value.body) {
                    this.variables(stmt, scope);
                }
                break;
            }
            case 'Return': {
                this.variables(value.value, scope);
                break;
            }
            case 'While': {
                this.variables(value.test, scope);
                for (const stmt of value.body) {
                    this.variables(stmt, scope);
                }
                break;
            }
            case 'UnaryOp': {
                this.variables(value.operand, scope);
                break;
            }
            case 'Pass': {
                break;
            }
            default: {
                throw new pytorch.Error(`Unsupported type '${value.type}'.`);
            }
        }
    }

    emitSugaredExpr(tree, n_binders, type_hint) {
        const ast = this.ast;
        if (tree instanceof ast.Var) {
            //
        } else if (tree instanceof ast.Attribute) {
            //
        } else if (tree instanceof ast.Apply) {
            //
        } if (tree instanceof ast.Subscript) {
            //
        }
        return this.emitSimpleExpr(tree, type_hint);
    }

    block(statements, context) {
        const ast = this.ast;
        const torch = this.torch;
        statements = Array.prototype.slice.call(statements);
        for (let i = 0; i < statements.length;) {
            if (i < statements.length - 1) {
                const containsVariableReference = (statements, value) => {
                    if (statements) {
                        for (const stmt of statements) {
                            if (!stmt.refs) {
                                this.variables(stmt, stmt);
                            }
                            if (stmt.refs.has(value)) {
                                return true;
                            }
                        }
                    }
                    return false;
                };
                const assign = statements[i];
                const condition = statements[i + 1];
                // _x = <expr>
                // if _x:
                //   ...
                if (assign instanceof ast.Assign && condition instanceof ast.If &&
                    assign.targets instanceof ast.Name && condition.test instanceof ast.Name &&
                    assign.targets.id === condition.test.id &&
                    !containsVariableReference(statements.slice(i + 2), condition.test.id) &&
                    (!condition.body || !containsVariableReference(condition.body.statements), condition.test.id) &&
                    (!condition.orelse || !containsVariableReference(condition.orelse.statements, condition.test.id))) {
                    const node = new ast.If(assign.value, condition.body, condition.orelse);
                    node.location = condition.location;
                    statements.splice(i, 2, node);
                }
            }
            const condition = statements[i];
            if (condition instanceof ast.If) {
                const state = [];
                let test = this.static(condition.test, context, state);
                if (test === null) {
                    test = false;
                } else if (typeof test === 'boolean') {
                    test = test === true;
                } else if (Number.isInteger(test)) {
                    test = test !== 0;
                } else if (typeof test === 'string') {
                    test = test && test.length > 0;
                }
                if (test === true) {
                    statements.splice(i, 1, ...condition.body);
                } else if (test === false) {
                    statements.splice(i, 1, ...condition.orelse);
                }
                for (const node of state) {
                    this.purge.add(node);
                }
                if (test === true || test === false) {
                    continue;
                }
            }
            if (i < statements.length) {
                const stmt = statements[i];
                if (stmt instanceof ast.If) {
                    const condition = stmt;
                    const test = this.expression(condition.test, context);
                    if (test instanceof torch.Value && test.type() instanceof torch.BoolType) {
                        const refs = new Set();
                        for (let j = i + 1; j < statements.length; j++) {
                            const stmt = statements[j];
                            if (!stmt.refs) {
                                this.variables(stmt, stmt);
                            }
                            for (const variable of Array.from(stmt.refs)) {
                                refs.add(variable);
                            }
                        }
                        const __variables = (statements) => {
                            const set = new Set();
                            for (const stmt of statements) {
                                if (stmt instanceof ast.Assign) {
                                    const target = stmt.targets;
                                    if (target instanceof ast.Name) {
                                        set.add(target.id);
                                    } else if (target instanceof ast.Tuple) {
                                        for (const value of target.elts) {
                                            if (value instanceof ast.Name) {
                                                set.add(value.id);
                                            } else {
                                                // debugger;
                                            }
                                        }
                                    } else {
                                        // debugger;
                                    }
                                }
                            }
                            return set;
                        };
                        const __type = (value) => {
                            if (!value) {
                                return null;
                            }
                            if (pytorch.Utility.isTensor(value)) {
                                return torch.TensorType.get();
                            }
                            if (value && value.__class__ && value instanceof torch.Value === false) {
                                const identifier = `${value.__class__.__module__}.${value.__class__.__name__}`;
                                const type = this._resolver.resolveType(identifier);
                                return type;
                            }
                            return value.type();
                        };
                        this.variables(condition, condition);
                        const node = this.create('prim::If', stmt.location, 0);
                        this._graph.insertNode(node);
                        node.addInput(test);
                        const prev = this._graph.insertPoint();
                        const true_block = node.addBlock();
                        this._graph.setInsertPoint(true_block);
                        let vars = __variables(condition.body.concat(stmt.orelse));
                        vars = new Map(Array.from(vars).map((name) => [name, {}]));
                        this.block(condition.body, context);
                        for (const [name, entry] of vars) {
                            entry.body = context.get(name);
                        }
                        const false_block = node.addBlock();
                        this._graph.setInsertPoint(false_block);
                        this.block(condition.orelse, context);
                        for (const [name, entry] of vars) {
                            entry.orelse = context.get(name);
                        }
                        this._graph.setInsertPoint(prev);
                        for (const [name, entry] of vars) {
                            if (!refs.has(name)) {
                                continue;
                            }
                            const value = node.addOutput();
                            context.set(name, value);
                            let type = null;
                            if (entry.body && !entry.orelse) {
                                type = __type(entry.body);
                            } else if (entry.orelse && !entry.body) {
                                type = __type(entry.orelse);
                            } else {
                                // compare
                                const t1 = __type(entry.body);
                                const t2 = __type(entry.orelse);
                                if (t1 === null && t2 === null) {
                                    type = null;
                                } else if (t1 === t2) {
                                    type = t1;
                                } else if (t1.equals(t2)) {
                                    type = t2;
                                } else if (t1 instanceof torch.NoneType && t2 instanceof torch.NoneType === false) {
                                    type = t2 instanceof torch.OptionalType ? t2 : torch.OptionalType.get(t2);
                                } else if (t1 instanceof torch.NoneType === false && t2 instanceof torch.NoneType) {
                                    type = t1 instanceof torch.OptionalType ? t1 : torch.OptionalType.get(t1);
                                } else if (t2.isSubtypeOf(t1)) {
                                    type = t1;
                                } else if (t1.isSubtypeOf(t2)) {
                                    type = t2;
                                } else {
                                    throw new pytorch.Error(`Unsupported condition type.`);
                                }
                            }
                            value.setType(type);
                        }
                        i++;
                        continue;
                    }
                    throw new pytorch.Error("Unsupported condition.");
                }
                if (stmt instanceof ast.For) {
                    const range = stmt.location;
                    const n = this._graph.insertNode(this.create('prim::Loop', range, 0));
                    const itrs = stmt.iter instanceof ast.Tuple ? stmt.iter.elts : [stmt.iter];
                    // const targets = stmt.target instanceof ast.Tuple ? stmt.target.elts : [stmt.target];
                    if (itrs.length !==  1) {
                        throw new pytorch.Error('List of iterables is not supported currently.');
                    }
                    /*
                    // const sv = this.expression(itrs[0], context);
                    const sv = this.emitSugaredExpr(itrs[0], 1);
                    const iterable = sv.iter(range, method);
                    if (iterable.shouldEmitUnrolled()) {
                        this.emitUnrolledLoop(loc, emit_body, iterable, targets);
                    } else {
                        this.emitLoopCommon(loc, emit_body, iterable, targets, {});
                    }
                    */

                    /* const body_block = */ n.addBlock();
                    /* const condition_block = */ n.addBlock();

                    const loop = stmt;
                    if (loop.target instanceof ast.Name && loop.iter instanceof ast.Tuple === false) {
                        const range = this.expression(loop.iter, context);
                        const variable = loop.target;
                        for (const current of range) {
                            const constant = new ast.Constant(current);
                            const stmt = new ast.Assign(variable, constant);
                            this.statement(stmt, context);
                            const value = this.block(loop.body, context);
                            if (value !== undefined) {
                                return value;
                            }
                        }
                        i++;
                        continue;
                    }
                }
                if (stmt instanceof ast.While) {
                    const node = this._graph.create('prim::Loop', stmt.location, 0);
                    this._graph.insertNode(node);
                    const test = this.expression(stmt.test, context);
                    if (test) {
                        const value = this.block(stmt.body, context);
                        if (value !== undefined) {
                            return value;
                        }
                    }
                    i++;
                    continue;
                }
                const value = this.statement(stmt, context);
                if (value !== undefined) {
                    return value;
                }
                i++;
            }
        }
        return undefined;
    }

    statement(stmt, context) {
        if (stmt.__class__.__name__ === 'ClassDef') {
            const name = `${context.get('__name__')}.${stmt.name}`;
            if (this._resolver) {
                this._resolver.resolveType(name);
            }
        }

        if (!this.trace) {
            return super.statement(stmt, context);
        }

        switch (stmt.__class__.__name__) {
            case 'ClassDef': {
                super.statement(stmt, context);
                /*
                const value = context.get(stmt.name);
                const type = torch.ClassType.create(`${value.__module__}.${value.__name__}`);
                for (const entry of stmt.body) {
                    if (entry instanceof ast.AnnAssign) {
                        const target = this.identifier(entry.target);
                        const annotation = this.type(entry.annotation, context);
                        type.addAttribute(target, annotation);
                    }
                }
                value.__type__ = type;
                */
                return undefined;
            }
            case 'If': {
                throw new pytorch.Error('Not implemented.');
            }
            default: {
                break;
            }
        }
        return super.statement(stmt, context);
    }

    type(expr) {
        const ast = this.ast;
        const torch = this.torch;
        if (expr instanceof ast.Subscript && expr.value instanceof ast.Name) {
            const elts = expr.slice.elts;
            switch (expr.value.id) {
                case 'List': {
                    const type = this.type(elts[0]);
                    return torch.ListType.create(type);
                }
                case 'Optional': {
                    const type = this.type(elts[0]);
                    return torch.OptionalType.get(type);
                }
                case 'Tuple': {
                    const types = elts.map((expr) => this.type(expr));
                    return torch.TupleType.create(types);
                }
                case 'Dict': {
                    const key = this.type(elts[0]);
                    const value = this.type(elts[1]);
                    return torch.DictType.create(key, value);
                }
                case 'Final': {
                    return this.type(elts[0]);
                }
                default: {
                    throw new pytorch.Error(`Unsupported type element expression '${expr.value.id}'.`);
                }
            }
        }
        if (expr instanceof ast.Name) {
            switch (expr.id) {
                case 'Tensor': return torch.TensorType.get();
                case 'int': return torch.IntType.get();
                case 'str': return torch.StringType.get();
                case 'float': return torch.FloatType.get();
                case 'number': return torch.NumberType.get();
                case 'bool': return torch.BoolType.get();
                case 'list': return torch.Type.get('AnyListType');
                case 'tuple': return torch.Type.get('AnyTupleType');
                case 'Device': return torch.DeviceObjType.get();
                case 'None': return torch.NoneType.get();
                case 'NoneType': return torch.NoneType.get();
                case 'Any': return torch.AnyType.get();
                default: throw new pytorch.Error(`Unsupported type expression '${expr.id}'.`);
            }
        }
        if (expr instanceof ast.Constant) {
            if (expr.value === null) {
                return torch.NoneType.get();
            }
            throw new pytorch.Error(`Unsupported type expression '${expr.value}'.`);
        }
        if (expr instanceof ast.Attribute) {
            const identifier = this.identifier(expr);
            const type = this._resolver.resolveType(identifier);
            if (type) {
                return type;
            }
        }
        throw new pytorch.Error(`Unsupported type expression '${expr.type}'.`);
    }

    constant(constant) {
        if (!this._constants.has(constant)) {
            const value = this._graph.insertConstant(constant);
            this._constants.set(constant, value);
        }
        return this._constants.get(constant);
    }

    call(target, name, args, context, location) {
        if (!this.trace) {
            return super.call(target, name, args, context);
        }
        const ast = this.ast;
        const torch = this.torch;
        if (name === '__new__') {
            const identifier = this.identifier(target);
            if (identifier) {
                const type = this._resolver.resolveType(identifier);
                if (type) {
                    const node = this._graph.createObject(type);
                    node.setSourceRange(location);
                    this._graph.insertNode(node);
                    return node.output();
                }
            }
        }
        /*
        if (name === '__init__') {
            const obj = this.expression(target, context);
            if (args.length === 0) {
                return obj;
            }
            const node = this._graph.create('prim::CallMethod', 0);
            node.setSourceRange(location);
            this._graph.insertNode(node);
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
        */
        const overload = this._overload(target, name, args, context);
        if (!overload) {
            const moduleTarget = this.target(target, context);
            if (moduleTarget instanceof torch.Value && moduleTarget.type() instanceof torch.ClassType) {
                const class_type = moduleTarget.type().expect(torch.ClassType);
                const method = class_type.getMethod(name);
                const evalArgs = args.map((expression) => this.expression(expression, context));
                if (this.traceAttr && method.__ast__) {
                    return this.apply(method.__ast__, [moduleTarget].concat(evalArgs), context);
                }
                const schema = method.getSchema();
                const return_field_names = [schema.returns[0].name];
                const return_types = [schema.returns[0].real_type];
                const inputs = [moduleTarget];
                for (const value of evalArgs) {
                    inputs.push(value);
                }
                const matchedSchema = new torch.jit.MatchedSchema(inputs, return_types, return_field_names, name);
                const node = this._graph.insertMethodCall(name, matchedSchema);
                return node.output();
            }
            const prefix = this.identifier(target);
            if (prefix && prefix !== 'self' && !prefix.startsWith('self.') && prefix.indexOf('.') !== -1) {
                const identifier = `${prefix}.${name}`;
                const type = this._resolver.resolveType(identifier);
                if (type instanceof torch.TupleType) {
                    const values = args.map((expression) => this.value(expression, context));
                    const node = this._graph.createTuple(values, type);
                    node.setSourceRange(location);
                    this._graph.insertNode(node);
                    return node.output();
                }
                if (type instanceof torch.ClassType) {
                    const node = this._graph.create('prim::CallMethod');
                    this._graph.insertNode(node);
                    node.s_('name', name);
                    const evalArgs = args.map((expression) => this.expression(expression, context));
                    for (const value of evalArgs) {
                        node.addInput(value);
                    }
                    return node.output();
                }
            }
            return super.call(target, name, args, context);
        }
        const [schema, evalArgs] = overload;
        const op = schema.overload_name ? `${schema.name}.${schema.overload_name}` : schema.name;
        const node = this.create(op, location, 0);
        this._graph.insertNode(node);
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
                if (schema.is_vararg) {
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
                (type instanceof torch.FloatType || type instanceof torch.BoolType || type instanceof torch.IntType || type instanceof torch.ComplexType || type instanceof torch.TensorType || type.kind() === 'ScalarTypeType' || type instanceof torch.DeviceObjType || type.kind() === 'LayoutKind') &&
                v instanceof torch.Value && v.type() instanceof torch.NoneType) {
                position++;
                input = v;
                match = true;
            } else if (type instanceof torch.ListType && type.getElementType() instanceof torch.TensorType) {
                const v = evalArgs[position];
                if ((v instanceof torch.Value && v.type() instanceof torch.ListType && v.type().getElementType() instanceof torch.TensorType) ||
                    (v === null || Array.isArray(v) && v.every((item) => pytorch.Utility.isTensor(item) || item === null || (item instanceof torch.Value && item.type() instanceof torch.TensorType)))) {
                    position++;
                    if (v instanceof torch.Value) {
                        input = v;
                        match = true;
                    } else if (v === null) {
                        input = this.constant(v);
                        match = true;
                    } else {
                        throw new pytorch.Error();
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
            } else if (args[position] instanceof ast.Assign && args[position].targets.id !== arg.name) {
                throw new pytorch.Error('Expected named argument.');
            } else {
                position++;
                if (v instanceof torch.Value) {
                    input = v;
                    match = true;
                } else if (v === null || typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') {
                    input = this.constant(v);
                    match = true;
                } else {
                    throw new pytorch.Error();
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
        if (args.every((arg, index) => index < position || (arg instanceof ast.Assign && arg.targets && arg.targets instanceof ast.Name))) {
            const params = new Map(parameters.slice(index).map((a) => [a.name, a]));
            while (position < args.length) {
                const arg = params.get(args[position].targets.id);
                const v = evalArgs[position];
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
                if (v instanceof torch.Value) {
                    node.addInput(v);
                } else if (v === null || typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') {
                    const value = this.constant(v);
                    node.addInput(value);
                } else {
                    throw new pytorch.Error();
                }
            }
        }
        for (const arg of schema.returns) {
            let type = arg.real_type;
            switch (type.str()) {
                case '__torch__.torch.classes.quantized.Conv2dPackedParamsBase':
                case '__torch__.torch.classes.quantized.Conv3dPackedParamsBase':
                case '__torch__.torch.classes.quantized.LinearPackedParamsBase':
                case '__torch__.torch.classes.rnn.CellParamsBase':
                case '__torch__.torch.classes.xnnpack.Conv2dOpContext':
                case '__torch__.torch.classes.xnnpack.LinearOpContext':
                case '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext': {
                    type = this._resolver.resolveType(type.qualified_name());
                    break;
                }
                case 'Tensor':
                case 'Tensor[]':
                case 'Scalar':
                case 'Dict(str, Tensor)':
                case 'int':
                case 'int[]':
                case 'str':
                case 'str[]':
                case 'float':
                case 'float[]':
                case 'complex':
                case 'bool':
                case 'bool[]':
                case 'Device':
                case 'Layout': {
                    break;
                }
                case 't': {
                    type = varTypes.map(type);
                    if (!type) {
                        throw new pytorch.Error(`Unknown var type 't'.`);
                    }
                    break;
                }
                case 't[]': {
                    type = varTypes.map(type.getElementType());
                    if (!type) {
                        throw new pytorch.Error();
                    }
                    type = torch.ListType.create(type);
                    break;
                }
                default: {
                    if (type instanceof torch.DictType) {
                        const keyType = varTypes.map(type.getKeyType());
                        const valueType = varTypes.map(type.getValueType());
                        type = torch.DictType.create(keyType, valueType);
                    } else if (type instanceof torch.TupleType && type.elements().length === 2) {
                        const elements = type.elements().map((type) => varTypes.map(type));
                        type = torch.ListType.create(torch.TupleType.create(elements));
                    } else if (type instanceof torch.ListType && type.getElementType() instanceof torch.TupleType) {
                        const elements = type.getElementType().elements().map((type) => varTypes.map(type));
                        type = torch.ListType.create(torch.TupleType.create(elements));
                    } else {
                        throw new pytorch.Error(`Unsupported return type '${type.str()}'.`);
                    }
                    break;
                }
            }
            const output = node.addOutput();
            output.__origin__ = schema.name;
            output.setType(type);
        }
        for (const referencedParameter of referencedParameters) {
            referencedParameter.__count__ = (referencedParameter.__count__ || 0) + 1;
        }
        const outputs = node.outputs();
        return outputs.length > 1 ? outputs : outputs[0];
    }

    isType(obj, type, N) {
        const torch = this.torch;
        const builtins = this.builtins;
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
                return obj === true || obj === false || (obj instanceof torch.Value && obj.type() instanceof torch.BoolType);
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
                if (obj instanceof torch.Value && obj.type() instanceof torch.OptionalType && obj.type().getElementType() instanceof torch.ListType && obj.type().getElementType().getElementType() instanceof torch.IntType) {
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
                return this.isType(obj, torch.IntType.get()) || this.isType(obj, torch.ListType.create(torch.IntType.get()));
            case 'float': {
                return obj !== null && (typeof obj === 'number' || obj instanceof Number) || (obj instanceof torch.Value && (obj.type() instanceof torch.FloatType || obj.type() instanceof torch.IntType));
            }
            case 'float[]': {
                if (Array.isArray(obj) && obj.every((item) => (typeof item === 'number' || item instanceof Number) && !isNaN(item))) {
                    return true;
                }
                if (obj instanceof torch.Value) {
                    const t = obj.type() instanceof torch.OptionalType ? obj.type().getElementType() : obj.type();
                    if (t instanceof torch.ListType && (t.getElementType() instanceof torch.IntType || t.getElementType() instanceof torch.FloatType)) {
                        return true;
                    }
                }
                return false;
            }
            case 'str':
                return obj === null || typeof obj === 'string' ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.StringType);
            case 'str[]':
                return (Array.isArray(obj) && obj.every((item) => item === null || typeof item === 'string')) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.StringType);
            case 'str[][]':
                return Array.isArray(obj) && obj.every((item) => Array.isArray(item) && item.every((item) => typeof item === 'string')) ||
                    (obj instanceof torch.Value && obj.type() instanceof torch.ListType && obj.type().getElementType() instanceof torch.ListType && obj.type().getElementType().getElementType() instanceof torch.StringType);
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
                if (type instanceof torch.ClassType) {
                    if (obj instanceof torch.Value && obj.type() instanceof torch.ClassType) {
                        return type.qualified_name() === obj.type().qualified_name();
                    }
                    if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                        return type.qualified_name() === `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                    }
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
                    if (obj instanceof builtins.dict) {
                        return true;
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
            return torch.ListType.create(torch.IntType.get());
        } else if (Array.isArray(value) && value.every((item) => Number(item) === item)) {
            return torch.ListType.create(torch.FloatType.get());
        } else if (value instanceof torch.Value) {
            return value.type();
        }
        const text = (JSON.stringify(value) || '(undefined)').substring(0, 10);
        throw new pytorch.Error(`Unsupported ops argument type '${text}'.`);
    }

    _overload(target, name, args, context) {
        const ast = this.ast;
        const torch = this.torch;
        const prefix = this.identifier(target);
        if (!prefix) {
            return null;
        }
        const type = name ? `${prefix}.${name}` : prefix;
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
            const module = this.import(prefix);
            if (!module || !module[name]) {
                const schema = new torch.FunctionSchema(op_name, null, [], [], false, false);
                for (let i = 0; i < args.length; i++) {
                    let argument = args[i];
                    let name = i.toString();
                    if (argument instanceof ast.Assign && argument.targets instanceof ast.Name) {
                        name = this.expression(argument.targets, context);
                        argument = argument.value;
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
        evalArgs = args.map((expr) => {
            if (expr instanceof ast.Assign && expr.targets instanceof ast.Name) {
                expr = expr.value;
            }
            return this.expression(expr, context);
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
                    next = !schema.is_vararg;
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
                    (type instanceof torch.FloatType || type instanceof torch.BoolType || type instanceof torch.IntType || type instanceof torch.ComplexType || type instanceof torch.TensorType || type.kind() === 'ScalarTypeType' || type instanceof torch.DeviceObjType || type.kind() === 'LayoutKind') &&
                    v instanceof torch.Value && v.type() instanceof torch.NoneType) {
                    position++;
                } else if (!this.isType(v, type, arg.N) && v !== null) {
                    if (optional) {
                        continue;
                    }
                    next = true;
                    break;
                } else if (args[position] instanceof ast.Assign && args[position].targets.id !== arg.name) {
                    next = true;
                    break;
                } else {
                    position++;
                }
            }
            if (next) {
                continue;
            }
            if (args.every((arg, index) => index < position || (arg instanceof ast.Assign && arg.targets instanceof ast.Name))) {
                const params = new Map(parameters.slice(index).map((a) => [a.name, a]));
                while (position < args.length) {
                    const value = evalArgs[position];
                    const arg = params.get(args[position].targets.id);
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
};

pytorch.Container.Package = class extends pytorch.Container {

    constructor(entries) {
        super();
        this.type = 'pytorch.package';
        this.entries = entries;
    }

    async read(metadata) {
        this.execution = new pytorch.Execution(null, metadata);
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        const torch = this.execution.__import__('torch');
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
                    this.execution.add(name, buffer);
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
            case 'BoolType': return 'boolean';
            case 'IntType': return 'int64';
            case 'FloatType': return 'float32';
            case 'StringType': return 'string';
            case 'ComplexType': return 'complex';
            case 'NumberType': return 'scalar';
            case 'TensorType': return 'tensor';
            case 'TupleType': return `tuple<${type.elements().map((type) => pytorch.Utility.toType(type)).join(', ')}>`;
            case 'DictType': return `map<${pytorch.Utility.toType(type.getKeyType())}, ${pytorch.Utility.toType(type.getValueType())}>`;
            case 'DeviceObjType': return 'device';
            case 'SymIntType': return 'SymInt';
            case 'ScalarTypeType': return 'ScalarType';
            case 'MemoryFormat': return 'MemoryFormat';
            case 'Layout': return 'Layout';
            case 'VarType': return type.annotation_str;
            case 'NoneType': return 'None';
            case 'AnyListType': return 'list';
            case 'AnyTupleType': return 'tuple';
            default: throw new pytorch.Error(`Unsupported type '${type.kind()}'.`);
        }
    }

    static constant(node, name) {
        const kind = node.kindOf(name);
        switch (kind) {
            case 's': return node.s(name);
            case 'i': return node.i(name);
            case 'f': return node.f(name);
            case 'ss': return node.ss(name);
            case 'ival': return node.ival(name);
            default: throw new pytorch.Error(`Unsupported attribute kind '${kind}'.`);
        }
    }

    static unique(value) {
        return value.hasDebugName() ? `%${value.debugName().toString()}` : `%${value.unique().toString()}`;
    }

    static isObject(obj) {
        const type = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : null;
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
        let type = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : null;
        if (type === 'torch.jit._script.RecursiveScriptModule') {
            type = obj._c._type();
            const target = {};
            for (let i = 0; i < type.numAttributes(); i++) {
                const k = type.getAttributeName(i);
                target[k] = obj.__getattr__(k);
            }
            type = obj._c.qualified_name;
            obj = target;
        } else if (type && type !== 'builtins.dict' && type !== 'builtins.object' && type !== 'collections.OrderedDict' && type !== 'torch.nn.modules.module.Module' && type !== '__torch__.Module') {
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
            if (names.length > 1 && (names.length / entries.length) >= 0.8 &&
                (entries.every(([, value]) => !pytorch.Utility.isInstance(value, 'builtins.dict') || Array.from(value.values()).every((value) => !pytorch.Utility.isTensor(value)))) &&
                (!entries.every(([, value]) => Array.isArray(value)))) {
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
                    if (!value || Object(value) !== value || pytorch.Utility.isTensor(value) || ArrayBuffer.isView(value) || value._modules instanceof Map) {
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
