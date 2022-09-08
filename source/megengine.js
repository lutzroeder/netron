
// Experimental

var megengine = megengine || {};
var base = base || require('./base');

megengine.ModelFactory = class {

    match(context) {
        const obj = context.open('pkl');
        if (obj.__class__ && obj.__class__.__module__ === 'megengine.traced_module.traced_module' && obj.__class__.__name__ === 'TracedModule') {
            return 'megengine.pickle';
        }
        return '';
    }

    open(context) {
        return context.metadata('megengine-metadata.json').then((metadata) => {
            const obj = context.open('pkl');
            return new megengine.Model(metadata, obj);
        });
    }
};

megengine.Model = class {

    constructor(metadata, obj) {
        this._format = 'MegEngine' + (obj.dump_info && obj.dump_info.version ? ' v' + obj.dump_info.version : '');
        this._graphs = [ new megengine.Graph(metadata, obj) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

megengine.Graph = class {

    constructor(metadata, obj) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const loadgraph = (tmodule, igraph, context, name_prefix, metadata, isroot) =>{
            const expressions = igraph._exprs;
            const isTensor = (obj) => {
                return obj && obj.__class__ && obj.__class__.__module__ == 'megengine.tensor' && (obj.__class__.__name__ === 'Tensor' || obj.__class__.__name__ === 'Parameter');
            };
            const getTensorType = (dtype, shape) => {
                const dt = dtype !== null ? dtype.__name__ : null;
                return new megengine.TensorType(dt, new megengine.TensorShape(shape));
            };
            const getOpNode = (metadata, item, expr, state) => {
                const op = new megengine.Node(metadata, item);
                let inp_idx = 0;
                for (const i of expr.inputs) {
                    if (i.__class__.__name__ !== 'ModuleNode') {
                        const initializer = i.initializer !== undefined ? i.initializer : null;
                        const inp_name = 'inp' + inp_idx;
                        op._inputs.push(new megengine.Parameter(inp_name, true, [
                            new megengine.Argument(i._fullname, getTensorType(i._dtype, i._shape), initializer)
                        ]));
                        inp_idx += 1;
                    }
                }
                const out_idx = 0;
                let qparams = null;
                for (const o of expr.outputs) {
                    if (o._qparams !== null) {
                        qparams = o._qparams[1];
                    }
                    op._outputs.push(new megengine.Parameter('out' + out_idx, true, [
                        new megengine.Argument(o._fullname, getTensorType(o._dtype, o._shape), null)
                    ]));
                }
                if (qparams !== null) {
                    state = state === null? {} : state;
                    state['scale'] = qparams.scale;
                    state['zero_point'] = qparams.zero_point;
                    state['quant_dtype_meta'] = qparams.dtype_meta;
                }
                if (state !== null) {
                    for (const key in state) {
                        const isModule = (obj) => {
                            return obj && (obj.state || obj._forward_pre_hooks);
                        };
                        if (!key.startsWith('_') && !isModule(state[key])) {
                            if (!isTensor(state[key])) {
                                op._attributes.push(new megengine.Attribute(null, key, state[key] !== null ? state[key] : 'None'));
                            }
                            else {
                                const tensor = state[key];
                                op._inputs.push(new megengine.Parameter(key, true, [
                                    new megengine.Argument('', getTensorType(tensor.dtype, tensor.data.shape), new megengine.Tensor(key, tensor))
                                ]));
                            }
                        }
                    }
                }
                return op;
            };
            if (isroot) {
                for (const node of igraph._inputs) {
                    if (node.__class__.__name__ === 'ModuleNode') {
                        continue;
                    }
                    this._inputs.push(new megengine.Parameter(node._name, true, [new megengine.Argument(node._name, getTensorType(node._dtype, node._shape), null)]));
                }
                for (const node of igraph._outputs) {
                    this._outputs.push(new megengine.Parameter(node._name, true, [new megengine.Argument(node._name, getTensorType(node._dtype, node._shape), null)]));
                }
            }
            const parse_getattr = (tmodule, getattr_expr) => {
                let attr_name = getattr_expr.name.split('.');
                while (getattr_expr.inputs[0].expr.__class__.__name__ === 'GetAttr') {
                    getattr_expr = getattr_expr.inputs[0].expr;
                    attr_name = getattr_expr.name.split('.').concat(attr_name);
                }
                let attr_obj = tmodule;
                for (const n of attr_name) {
                    attr_obj = attr_obj[n];
                }
                return attr_obj;
            };
            const parseargs = (args, kwargs, meta) => {
                const state = {};
                let arg_idx = 0;
                let attr_name = '';
                const process_args = (inp, start_idx) => {
                    while (typeof inp === 'string' && inp.indexOf('Tensor') !== -1) {
                        inp = inp.replace('Tensor', 'inp' + start_idx);
                        start_idx += 1;
                    }
                    return [inp, start_idx];
                };
                const formatTreeDef = (obj) => {
                    if (obj.__class__.__name__ !== 'TreeDef' && obj.__class__.__name__ !== 'LeafDef') {
                        throw new megengine.Error('formatTreeDef gets invalid argument');
                    }
                    if (obj.__class__.__name__ === 'TreeDef') {
                        const type = typeof obj.type !== 'string' ? obj.type.__name__ : obj.type.split('.').slice(-1)[0];
                        const list = obj.children_defs.map((child) => formatTreeDef(child));
                        switch (type) {
                            case 'tuple': {
                                return '(' + list.join(',') + ')';
                            }
                            case 'slice': {
                                return list.join(':');
                            }
                            case 'list': {
                                return '[' + list.join(',') + ']';
                            }
                            case 'dict': {
                                let content = '';
                                for (let i = 0; i < this.children_defs.length; i++) {
                                    content += this.aux_data[i] + ':' + list[i];
                                }
                                return '{' + content + '}';
                            }
                            default: {
                                return type + '(' + list.join(',') + ')';
                            }
                        }
                    }
                    if (obj.const_val !== null) {
                        return obj.const_val;
                    }
                    else if (obj.type[0].__module__ !== undefined) {
                        return obj.type[0].__name__;
                    }
                    return 'None';
                };
                let inp_idx = 0;
                for (const arg of args.children_defs) {
                    if (meta.attributes === undefined || (meta.attributes.length !== args.children_defs.length && meta.varargs === null)) {
                        attr_name = 'arg' + arg_idx;
                    }
                    else if (arg_idx < meta.attributes.length) {
                        attr_name = meta.attributes[arg_idx].name;
                    }
                    else {
                        attr_name = meta.varargs + (arg_idx - meta.attributes.length);
                    }
                    const rst = process_args(formatTreeDef(arg), inp_idx);
                    state[attr_name] = rst[0];
                    inp_idx = rst[1];
                    arg_idx += 1;
                }
                for (let i = 0; i < kwargs.children_defs.length; i++) {
                    const rst = process_args(formatTreeDef(kwargs.children_defs[i]), inp_idx);
                    inp_idx = rst[1];
                    state[kwargs.aux_data[i]] = rst[0];
                }
                return state;
            };
            const getname = (context, name) => {
                let rst = name;
                while (context.get(rst) !== undefined) {
                    if (rst === context.get(rst)) {
                        return rst;
                    }
                    rst = context.get(rst);
                }
                return rst;
            };
            const getfullname = (prefix, name) => {
                return prefix === '' ? name : prefix + '_' + name;
            };
            for (const expr of expressions) {
                const type = expr.__class__.__name__;
                for (const i of expr.inputs) {
                    i._fullname = getname(context, getfullname(name_prefix, i._name));
                }
                for (const o of expr.outputs) {
                    o._fullname = getname(context, getfullname(name_prefix, o._name));
                }
                switch (type) {
                    case 'Input': {
                        break;
                    }
                    case 'GetAttr': {
                        if (expr.outputs[0].__class__.__name__ === 'TensorNode') {
                            const tensor = parse_getattr(tmodule, expr);
                            expr.outputs[0].initializer = new megengine.Tensor(expr.name, tensor);
                        }
                        break;
                    }
                    case 'Constant': {
                        if (expr.outputs[0].__class__.__name__ === 'TensorNode') {
                            expr.outputs[0].initializer = new megengine.Tensor('', expr.value);
                        }
                        break;
                    }
                    case 'CallMethod': {
                        if (expr.method === '__call__') {
                            const getattr_expr = expr.inputs[0].expr;
                            const called_module = parse_getattr(tmodule, getattr_expr);
                            const getModuleType = (obj) => {
                                if (obj.module !== undefined) {
                                    return obj.module[0] + '.' + obj.module[1];
                                }
                                return obj.__class__.__module__ + '.' + obj.__class__.__name__;
                            };
                            const module_type = called_module.__class__.__name__ !== 'TracedModule' ? getModuleType(called_module) : 'TracedModule';
                            if (module_type === 'TracedModule') {
                                const module_name = expr.outputs[0]._name.endsWith("_out")?expr.outputs[0]._name.substring(0, expr.outputs[0]._name.length-4):expr.outputs[0]._name;
                                const prefix = getfullname(name_prefix, module_name);
                                const internal_graph = called_module.argdef_graph_map[expr.arg_def.toString()];
                                for (let i = 0; i < expr.inputs.length; i++) {
                                    const actual_name = getfullname(name_prefix, expr.inputs[i]._name);
                                    const internal_name = getfullname(prefix, internal_graph._inputs[i]._name);
                                    context.set(internal_name, actual_name);
                                }
                                for (let i = 0; i < expr.outputs.length; i++) {
                                    const actual_name = getfullname(name_prefix, expr.outputs[i]._name);
                                    const internal_name = getfullname(prefix, internal_graph._outputs[i]._name);
                                    if (context.get(internal_name) !== undefined) {
                                        context.set(actual_name, context.get(internal_name));
                                    }
                                    else {
                                        context.set(internal_name, actual_name);
                                    }
                                }
                                loadgraph(called_module, internal_graph, context, prefix, metadata, false);
                                continue;
                            }
                            const item = { 'name': '', 'type': module_type };
                            let state = called_module.__class__.__name__ !== 'TracedModule' ? called_module.state : called_module;
                            if (state === undefined) {
                                state = called_module;
                            }
                            this._nodes.push(getOpNode(metadata, item, expr, state));
                        }
                        else {
                            const item = { 'name': '', 'type': expr.method };
                            const args = expr.arg_def.children_defs[0];
                            const kwargs = expr.arg_def.children_defs[1];
                            const schema = metadata.type(expr.method);
                            const state = parseargs(args, kwargs, schema);
                            this._nodes.push(getOpNode(metadata, item, expr, state));
                        }
                        break;
                    }
                    case 'CallFunction': {
                        const getFunctionType = (obj) => {
                            if (obj.func.__module__ !== undefined) {
                                return obj.func.__module__ + '.' + obj.func.__name__;
                            }
                            return obj.func[0] + '.' + obj.func[1];
                        };
                        const func = getFunctionType(expr);
                        const item = { 'name': '', 'type': func };
                        const args = expr.arg_def.children_defs[0];
                        const kwargs = expr.arg_def.children_defs[1];
                        const schema = metadata.type(func);
                        const state = parseargs(args, kwargs, schema);
                        this._nodes.push(getOpNode(metadata, item, expr, state));
                        break;
                    }
                    case 'Apply': {
                        const opdef = expr.opdef_state ? expr.opdef_state.opdef_type : expr.opdef.type;
                        const item = { 'name': '', 'type': opdef.__module__ + '.' + opdef.__name__ };
                        this._nodes.push(getOpNode(metadata, item, expr, expr.opdef_state));
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
        };
        const graph = Object.values(obj.argdef_graph_map)[0];
        loadgraph(obj, graph, new Map(), '', metadata, true);
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

megengine.Parameter = class {

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

megengine.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new megengine.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

megengine.Node = class {

    constructor(metadata, item) {
        this._name = item.name || '';
        this._type = Object.assign({}, metadata.type(item.type));
        if (this._type.name.length > 4 && this._type.name.startsWith('__') && this._type.name.endsWith('__')) {
            this._type.name = this._type.name.substring(2, this._type.name.length - 2);
        }
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        this._attributes = [];
    }

    get name() {
        return this._name;
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

megengine.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;

        if (this._name === 'training') {
            this._visible = false;
            this._type = 'boolean';
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

megengine.Tensor = class {

    constructor(name, tensor) {
        this._name = name || '';
        this._type = new megengine.TensorType(tensor.dtype.__name__, new megengine.TensorShape(tensor.data.shape));
        this._data = tensor.data.data;
    }

    get category() {
        return 'Weights';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._data;
    }
};

megengine.TensorType = class {

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

megengine.TensorShape = class {

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

megengine.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MegEngine model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = megengine.ModelFactory;
}