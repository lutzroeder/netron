
// Experimental

import * as base from './base.js';
import * as flatbuffers from './flatbuffers.js';
import * as python from './python.js';

const pytorch = {};
const numpy = {};

pytorch.ModelFactory = class {

    async match(context) {
        const container = await pytorch.Container.open(context);
        if (container) {
            return context.set(container.type, container);
        }
        return null;
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
        const target = context.value;
        target.on('resolve', (sender, name) => {
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
        if (torch && module instanceof torch.jit._script.RecursiveScriptModule && module._c._has_method('forward')) {
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
            const deleted = new Set();
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
                            // deleted.add(node);
                            node.destroy();
                        }
                    }
                };
                delattr(param_node.outputs()[0], '');
            }
            for (const node of graph.nodes()) {
                if (node.kind() === 'prim::Constant' && node.hasAttribute('value')) {
                    const kind = node.kindOf('value');
                    const value = node[kind]('value');
                    for (const output of node.outputs()) {
                        output.identifier = output.debugName();
                        output.value = value;
                    }
                    // deleted.add(node);
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
                        // deleted.add(node);
                        node.destroy();
                    }
                }
            }
            for (const node of graph.nodes()) {
                if (node.kind() === 'prim::ListConstruct' &&
                    node.inputs().every((value) => typeof value.value === 'number' || typeof value.value === 'string' || typeof value.value === 'boolean') &&
                    node.outputs().every((value) => value.uses().every((use) => use.user.kind() !== 'prim::CallMethod'))) {
                    node.outputs()[0].value = node.inputs().map((value) => value.value);
                    // deleted.add(node);
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
                if (deleted.has(node)) {
                    continue;
                }
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
                        const tensor = parameter && parameter.data ? parameter.data : obj.meta.get('val');
                        const initializer = new pytorch.Tensor(key, tensor);
                        const value = new pytorch.Value(key, null, null, initializer);
                        values.set(obj, value);
                    } else if (inputs_to_buffers.has(obj.name)) {
                        const key = inputs_to_buffers.get(obj.name);
                        const buffer = exported_program.state_dict.get(key);
                        const tensor = buffer || obj.meta.get('val');
                        const initializer = new pytorch.Tensor(key, tensor);
                        const value = new pytorch.Value(key, null, null, initializer);
                        values.set(obj, value);
                    } else if (inputs_to_lifted_tensor_constants.has(obj.name)) {
                        const key = inputs_to_lifted_tensor_constants.get(obj.name);
                        const constant = exported_program.constants.get(key);
                        const tensor = constant && constant.data ? constant.data : obj.meta.get('val');
                        const initializer = new pytorch.Tensor(key, tensor);
                        const value = new pytorch.Value(key, null, null, initializer);
                        values.set(obj, value);
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
                    if (input.value !== undefined) {
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
                                if (value.value !== undefined) {
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
                this.metadata.push(new pytorch.Argument('source', sourceRange.toString().replace(/^at\s/, '').replace(/\.$/, ''), 'attribute'));
                if (sourceRange.source()) {
                    const orig = sourceRange.source().findSourceRangeThatGenerated(sourceRange);
                    if (orig) {
                        this.metadata.push(new pytorch.Argument('generated', orig.toString(), 'attribute'));
                    }
                }
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

    static async open(context) {
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
            pytorch.Container.ExportedProgram
        ];
        for (const type of types) {
            /* eslint-disable no-await-in-loop */
            const container = await type.open(context);
            /* eslint-enable no-await-in-loop */
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

    static async open(context) {
        const entries = await context.peek('tar');
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

    static async open(context) {
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

    static async open(context) {
        const obj = await context.peek('pkl');
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

    constructor(type, module) {
        super();
        this.type = 'pytorch.data.pkl';
        this.format = 'PyTorch Pickle';
        this.module = module;
    }

    async read() {
    }
};

pytorch.Container.torch_utils = class extends pytorch.Container {

    static async open(context) {
        const stream = context.stream;
        if (stream && stream.length > 1) {
            const buffer = stream.peek(Math.min(1024, stream.length));
            if (buffer[0] === 0x80) {
                const content = String.fromCharCode.apply(null, buffer);
                if (content.indexOf('torch_utils') !== -1) {
                    const obj = await context.peek('pkl');
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

    static async open(context) {
        const reader = await context.peek('flatbuffers.binary');
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
        for (const event of this._events) {
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

pytorch.Container.Zip = class extends pytorch.Container {

    static async open(context) {
        const entries = await context.peek('zip');
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
            if (records.has('.data/version') && !records.has('archive_format')) {
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
            metadata.register(this.execution);
            this.module = torch.jit.load(reader);
            torchscript = this.module._c._has_method('forward');
            if (torchscript) {
                // console.log(this.module.graph.toString());
                torch._C._jit_pass_inline(this.module.graph);
                // console.log(this.module.graph.toString());
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

    static async open(context) {
        const identifier = context.identifier;
        if (identifier === 'model.json') {
            const model = await context.peek('json');
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
        metadata.register(this.execution);
        this.module = torch.jit.load(reader);
        if (this.module._c._has_method('forward')) {
            torch._C._jit_pass_inline(this.module.graph);
        }
        delete this._context;
        delete this._model;
        delete this._entries;
    }
};

pytorch.Container.IR = class extends pytorch.Container {

    static async open(context) {
        const reader = await context.read('text', 0x100);
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
        // context reader = await context.read('text', 0x100);
        throw new pytorch.Error('TorchScript IR parser not implemented.');
    }
};

pytorch.Container.Index = class extends pytorch.Container {

    static async open(context) {
        const obj = await context.peek('json');
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
        const archives = await Promise.all(contexts.map((context) => context.peek('zip')));
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

    static async open(context) {
        const program = await context.peek('json');
        if (program && program.schema_version && program.graph_module) {
            return new pytorch.Container.ExportedProgram(context, program);
        }
        if (context.identifier === 'archive_format' && context.stream && context.stream.length < 10) {
            const buffer = context.stream.peek();
            const archive_format = String.fromCharCode.apply(null, buffer);
            if (archive_format === 'pt2') {
                return new pytorch.Container.ExportedProgram(context, null, context);
            }
        }
        return null;
    }

    constructor(context, exported_program, archive_format) {
        super();
        this.type = 'pytorch.export';
        this.context = context;
        this.archive_format = archive_format;
        this.exported_program = exported_program;
    }

    async read(metadata) {
        this.format = 'PyTorch Export';
        const f = new Map();
        const exported_programs = new Map();
        if (this.archive_format) {
            for (const name of this.context.container.entries.keys()) {
                const match = name.match(/^models\/([^/]+)\.json$/);
                if (match) {
                    const [, model_name] = match;
                    /* eslint-disable no-await-in-loop */
                    const model = await this.context.fetch(`models/${model_name}.json`);
                    const constants = await this._fetch(`data/constants/${model_name}.pt`);
                    const sample_inputs = await this._fetch(`data/sample_inputs/${model_name}.pt`);
                    const weights = await this._fetch(`data/weights/${model_name}.pt`);
                    const exported_program = await model.read('json');
                    /* eslint-enable no-await-in-loop */
                    exported_programs.set(model_name, exported_program);
                    f.set(`models/${model_name}.json`, exported_program);
                    f.set(`data/weights/${model_name}.pt`, weights);
                    f.set(`data/constants/${model_name}.pt`, constants);
                    f.set(`data/sample_inputs/${model_name}.pt`, sample_inputs);
                }
            }
            const byteorder = await this._text('byteorder') || 'little';
            f.set('byteorder', byteorder);
        } else {
            this.version = await this._text('version');
            this.version = this.version.split('\n').shift().trim();
            const weights = await this._fetch('serialized_state_dict.pt') || await this._fetch('serialized_state_dict.json');
            const constants = await this._fetch('serialized_constants.pt') || await this._fetch('serialized_constants.json');
            const sample_inputs = await this._fetch('serialized_example_inputs.pt');
            f.set('models/model.json', this.exported_program);
            f.set('data/weights/model.pt', weights);
            f.set('data/constants/model.pt', constants);
            f.set('data/sample_inputs/model.pt', sample_inputs);
            exported_programs.set('', this.exported_program);
        }
        if (!this.version) {
            const versions = new Set();
            for (const exported_program of exported_programs.values()) {
                const schema_version = exported_program.schema_version;
                if (schema_version && schema_version.major && schema_version.minor) {
                    versions.add(`${schema_version.major}.${schema_version.minor}`);
                }
            }
            if (versions.size === 1) {
                this.version = versions.values().next().value;
            }
        }
        this.format = this.version ? `${this.format} v${this.version}` : this.format;
        this.execution = new python.Execution();
        for (const event of this._events) {
            this.execution.on(event[0], event[1]);
        }
        metadata.register(this.execution);
        const torch = this.execution.__import__('torch');
        for (const exported_program of exported_programs.values()) {
            if (exported_program.graph_module.graph.constants) {
                /* eslint-disable no-await-in-loop */
                const zip = await import('./zip.js');
                /* eslint-enable no-await-in-loop */
                const constants = this.exported_program.graph_module.graph.constants;
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
        }
        delete this.exported_program;
        delete this.context;
        const pt2_contents = torch.export.pt2_archive._package.load_pt2(f);
        this.modules = pt2_contents.exported_programs;
    }

    async _fetch(name) {
        try {
            const context = await this.context.fetch(name);
            if (context) {
                return await context.peek('zip');
            }
        } catch {
            // continue regardless of error
        }
        return null;
    }

    async _text(name) {
        try {
            const content = await this.context.fetch(name);
            if (content) {
                const reader = await content.read('text');
                if (reader) {
                    return reader.read();
                }
            }
        } catch {
            // continue regardless of error
        }
        return '';
    }
};

pytorch.Execution = class extends python.Execution {

    constructor(sources, metadata) {
        super(sources);
        this._metadata = metadata;
        /* eslint-disable consistent-this */
        const execution = this;
        /* eslint-enable consistent-this */
        const torch = this.torch;
        this.registerFunction('torch.jit.jit_module_from_flatbuffer', (f) => {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const stream = f;
            const reader = flatbuffers.BinaryReader.open(stream);
            const module = torch.mobile.serialization.Module.create(reader);
            const loader = new torch._C.FlatBuffersLoader(cu);
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
        this.registerType('__torch__.torch.classes.quantized.EmbeddingPackedParamsBase', class {
            __setstate__(state) {
                [this.version, this.tensors, this.doubles, this.longs] = state;
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
        this.registerType('__torch__.torch.classes.tensorrt.Engine', class {
            __setstate__(state) {
                [this.abi_target, this.name, this.device, this.engine, this.input_binding_names, this.output_binding_names, this.hw_compatible, this.serialized_metadata, this.target_platform] = state;
            }
        });
        const custom_classes = [
            { name: '__torch__.torch.classes._nnapi.Compilation', methods: [
                '__init__(__torch__.torch.classes._nnapi.Compilation self) -> NoneType',
                'init(__torch__.torch.classes._nnapi.Compilation self, Tensor serialized_model_tensor, Tensor[] parameter_buffers) -> NoneType',
                'init2(__torch__.torch.classes._nnapi.Compilation self, Tensor serialized_model_tensor, Tensor[] parameter_buffers, int compilation_preference, bool relax_f32_to_f16) -> NoneType',
                'run(__torch__.torch.classes._nnapi.Compilation self, Tensor[] inputs, Tensor[] outputs) -> NoneType'
            ] },
            { name: '__torch__.torch.classes.quantized.Conv2dPackedParamsBase', attributes: 'Tensor weight, Tensor bias, int[] stride, int[] padding, int[] dilation, int groups', methods: ['unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase self) -> ((Tensor, Tensor?))'] },
            { name: '__torch__.torch.classes.quantized.Conv3dPackedParamsBase', attributes: 'Tensor weight, Tensor bias, int[] stride, int[] padding, int[] dilation, int groups', methods: ['unpack(__torch__.torch.classes.quantized.Conv3dPackedParamsBase self) -> ((Tensor, Tensor?))'] },
            { name: '__torch__.torch.classes.quantized.LinearPackedParamsBase', attributes: 'Tensor weight, Tensor? bias' },
            { name: '__torch__.torch.classes.quantized.EmbeddingPackedParamsBase', attributes: 'int version, Tensor[] tensors, float[] doubles, int[] longs', methods: [] },
            { name: '__torch__.torch.classes.rnn.CellParamsBase', attributes: 'str type, Tensor[] tensors, float[] doubles, int[] longs, __torch__.torch.classes.quantized.LinearPackedParamsBase[] packed_params' },
            { name: '__torch__.torch.classes.xnnpack.Conv2dOpContext', attributes: 'Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, int[] output_min, int[] output_max' },
            { name: '__torch__.torch.classes.xnnpack.LinearOpContext', attributes: 'Tensor weight, Tensor bias, int[] output_min, int[] output_max' },
            { name: '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext', attributes: 'Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups, int[] output_min, int[] output_max' },
            { name: '__torch__.torch.classes.tensorrt.Engine' }
        ];
        for (const known_type of custom_classes) {
            const prefix = new torch._C.QualifiedName(known_type.name);
            const type = torch.ClassType.create(known_type.name, this._compilation_unit, false);
            for (const known_method of known_type.methods || []) {
                const schema = new torch.FunctionSchema(known_method);
                const name = new torch._C.QualifiedName(prefix, schema.name);
                const fn = new torch._C.BuiltinOpFunction(name, schema);
                type.addMethod(fn);
            }
            if (known_type.attributes) {
                const schema = new torch.FunctionSchema(`(${known_type.attributes}) -> ()`);
                for (const arg of schema.arguments) {
                    type.addAttribute(arg.name, arg.real_type);
                }
            }
            torch._C.registerCustomClass(type);
        }
    }

    call(target, name, args, keywords, context) {
        const ast = this.ast;
        const torch = this.torch;
        if (target instanceof ast.Name && target.id === 'torch') {
            const fn = torch.ops.aten[name];
            if (fn) {
                const evalArgs = args.map((arg) => this.expression(arg, context));
                return fn(...evalArgs);
            }
        }
        return super.call(target, name, args, keywords, context);
    }

    invoke(target, args) {
        if (target && Array.isArray(target.__bases__) && target.__bases__.length > 0 && target.__bases__[0] === this.enum.Enum) {
            const instance = new target();
            instance.value = args;
            return instance;
        }
        return super.invoke(target, args);
    }

    base(expr, context) {
        const ast = this.ast;
        if (expr instanceof ast.Name) {
            switch (expr.id) {
                case 'Enum': return this.enum.Enum;
                default: break;
            }
        }
        return this.expression(expr, context);
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
            metadata.register(this.execution);
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
            case 'ClassType': return type.annotation_str;
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
            case '__torch__.torch.classes.quantized.EmbeddingPackedParamsBase':
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
        value = value.toString();
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
        const registry = torch._C.getRegistry();
        const modules = new Set();
        for (const [name, type] of this._types) {
            if (name.indexOf('::') !== -1) {
                const schema = torch.FunctionSchema.parse(type.name);
                if (type.category) {
                    schema.category = type.category;
                }
                schema.setAliasAnalysis('FROM_SCHEMA');
                const op = new torch._C.Operator(schema);
                registry.registerOperator(op);
                modules.add(type.name.split('::')[0]);
            }
        }
        for (const module of modules) {
            const existing = execution.register(`ops.${module}`);
            const namespace = new torch._ops._OpNamespace(module);
            const created = execution.register(`torch.ops.${module}`, namespace);
            for (const [name, obj] of Object.entries(existing)) {
                if (name !== '__module__' && name !== '__name__' && !(name in created)) {
                    created[name] = obj;
                }
            }
        }
    }
};

numpy.Tensor = class  {

    constructor(array) {
        this.type = new numpy.TensorType(array.dtype.__name__, new numpy.TensorShape(array.shape));
        this.stride = array.strides.map((stride) => stride / array.itemsize);
        this.encoding = this.type.dataType === 'string' || this.type.dataType === 'object' ? '|' : array.dtype.byteorder;
        this.values = this.type.dataType === 'string' || this.type.dataType === 'object' || this.type.dataType === 'void' ? array.flatten().tolist() : array.tobytes();
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
