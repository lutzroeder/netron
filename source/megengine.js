
// Experimental

var megengine = megengine || {};
var python = python || require('./python');
var base = base || require('./base');

megengine.ModelFactory = class {

    match(context) {
        return megengine.Pickle.open(context);
    }

    open(context, match) {
        const identifier = context.identifier;
        return context.metadata('megengine-metadata.json').then((metadata) => {
            const container = match;
            try {
                container.metadata = metadata;
                container.exception = (error, fatal) => {
                    const message = error && error.message ? error.message : error.toString();
                    context.exception(new megengine.Error(message.replace(/\.$/, '') + " in '" + identifier + "'."), fatal);
                };
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new megengine.Error('File format is not megengine (' + message.replace(/\.$/, '') + ').');
            }
            return new megengine.Model(metadata, container);
        });
    }
};

megengine.Model = class {

    constructor(metadata, container) {
        this._format = container.format;
        this._producer = container.producer || '';
        this._graphs = container.graphs.map((graph) => new megengine.Graph(metadata, graph));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

megengine.Graph = class {

    constructor(metadata, graph) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        this._name = graph.name || '';
        const tmodule = graph.data;
        const igraph = Object.values(tmodule.argdef_graph_map)[0];
        const context = new Map();
        const name_prefix = "";
        this._loadgraph(tmodule, igraph, context, name_prefix, metadata, true);
    }

    _loadgraph(tmodule, igraph, context, name_prefix, metadata, isroot) {
        const exprs = igraph._exprs;
        if (isroot) {
            for (const node of igraph._inputs) {
                if (node.__class__.__name__ === "ModuleNode") {
                    continue;
                }
                this._inputs.push(new megengine.Parameter(node._name, true, [new megengine.Argument(node._name, megengine.Utility.getTensorType(node._dtype, node._shape), null)]));
            }
            for (const node of igraph._outputs) {
                this._outputs.push(new megengine.Parameter(node._name, true, [new megengine.Argument(node._name, megengine.Utility.getTensorType(node._dtype, node._shape), null)]));
            }
        }
        const parse_getattr = (tmodule, getattr_expr) => {
            var attr_name = getattr_expr.name.split(".");
            while (getattr_expr.inputs[0].expr.__class__.__name__ === "GetAttr") {
                getattr_expr = getattr_expr.inputs[0].expr;
                attr_name = getattr_expr.name.split(".").concat(attr_name);
            }
            var attr_obj = tmodule;
            for (const n of attr_name) {
                attr_obj = attr_obj[n];
            }
            return attr_obj;
        };
        for (const expr of exprs) {
            const expr_type = expr.__class__.__name__;
            for (const i of expr.inputs) {
                i._fullname = this._getname(context, this._getfullname(name_prefix, i._name));
            }
            for (const o of expr.outputs) {
                o._fullname = this._getname(context, this._getfullname(name_prefix, o._name));
            }
            if (expr_type === "Input") {continue;}
            else if (expr_type === "GetAttr") {
                if (expr.outputs[0].__class__.__name__ === "TensorNode") {
                    var tensor = parse_getattr(tmodule, expr);
                    expr.outputs[0].initializer = megengine.Utility.createTensor(expr.name, tensor, true);
                }
            }
            else if (expr_type === "Constant") {
                if (expr.outputs[0].__class__.__name__ === "TensorNode") {
                    expr.outputs[0].initializer = megengine.Utility.createTensor("", expr.value, true);
                }
            }
            else if (expr_type === "CallMethod" && expr.method === "__call__") {
                var getattr_expr = expr.inputs[0].expr;
                var called_module = parse_getattr(tmodule, getattr_expr);
                var module_type = called_module.__class__.__name__ !== "TracedModule" ? megengine.Utility.getModuleType(called_module) : "TracedModule";
                if (module_type === "TracedModule") {
                    const prefix = this._getfullname(name_prefix, expr.inputs[0]._name);
                    const internal_graph = called_module.argdef_graph_map[expr.arg_def.toString()];
                    for (let i = 0; i < expr.inputs.length; i++) {
                        const actual_name = this._getfullname(name_prefix, expr.inputs[i]._name);
                        const internal_name = this._getfullname(prefix, internal_graph._inputs[i]._name);
                        context.set(internal_name, actual_name);
                    }
                    for (let i = 0; i < expr.outputs.length; i++) {
                        const actual_name = this._getfullname(name_prefix, expr.outputs[i]._name);
                        const internal_name = this._getfullname(prefix, internal_graph._outputs[i]._name);
                        context.set(internal_name, actual_name);
                    }
                    this._loadgraph(called_module, internal_graph, context, prefix, metadata, false);
                    continue;
                }

                const item = { "name": "", "type": module_type };
                var state = called_module.__class__.__name__ !== "TracedModule" ? called_module.state : called_module;
                if (state === undefined) {state = called_module;}
                this._nodes.push(megengine.Utility.getOpNode(metadata, item, expr, state));
            }
            else if (expr_type === "CallMethod") {
                const item = { "name": "", "type": expr.method };
                const args = expr.arg_def.children_defs[0];
                const kwargs = expr.arg_def.children_defs[1];
                var schema = metadata._types.get(expr.method);
                state = this._parseargs(args, kwargs, schema);
                this._nodes.push(megengine.Utility.getOpNode(metadata, item, expr, state));
            }
            else if (expr_type === "CallFunction") {
                const func = megengine.Utility.getFunctionType(expr);
                const item = { "name": "", "type": func };
                const args = expr.arg_def.children_defs[0];
                const kwargs = expr.arg_def.children_defs[1];
                const schema = metadata._types.get(func);
                state = this._parseargs(args, kwargs, schema);
                this._nodes.push(megengine.Utility.getOpNode(metadata, item, expr, state));
            }
            else if (expr_type === "Apply") {
                const opdef = expr.opdef_state ? expr.opdef_state.opdef_type : expr.opdef.type;
                const item = { "name": "", "type": opdef.__module__ + "." + opdef.__name__ };
                this._nodes.push(megengine.Utility.getOpNode(metadata, item, expr, expr.opdef_state));
            }

        }
    }

    _getname(context, name) {
        let rst = name;
        while (context.get(rst) !== undefined) {
            if (rst === context.get(rst)) {
                return rst;
            }
            rst = context.get(rst);
        }
        return rst;
    }

    _getfullname(prefix, name) {
        if (prefix === "") {return name;}
        return prefix + "_" + name;
    }

    _parseargs(args, kwargs, meta) {
        var state = {};
        var schema = meta !== undefined ? meta.schema : undefined;
        var arg_idx = 0;
        var attr_name = "";
        const process_args = (inp, start_idx) => {

            while (typeof inp === "string" && inp.indexOf("Tensor") != -1) {
                inp = inp.replace("Tensor", "inp" + start_idx);
                start_idx += 1;
            }
            return [inp, start_idx];

        };
        var inp_idx = 0;
        for (const arg of args.children_defs) {
            if (schema === undefined || (schema.attributes.length !== args.children_defs.length && schema.varargs === null)) {
                attr_name = "arg" + arg_idx;
            }
            else if (arg_idx < schema.attributes.length) {
                attr_name = schema.attributes[arg_idx];
            }
            else {
                attr_name = schema.varargs + (arg_idx - schema.attributes.length);
            }
            const rst = process_args(megengine.Utility.formatTreeDef(arg), inp_idx);
            state[attr_name] = rst[0];
            inp_idx = rst[1];
            arg_idx += 1;
        }
        for (var i = 0; i < kwargs.children_defs.length; i++) {
            const rst = process_args(megengine.Utility.formatTreeDef(kwargs.children_defs[i]), inp_idx);
            inp_idx = rst[1];
            state[kwargs.aux_data[i]] = rst[0];
        }
        return state;
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

    constructor(metadata, group, item) {
        this._group = group || '';
        this._name = item.name || '';
        this._type = { name: item.type };
        this._inputs = [];
        this._outputs = [];
        var schema = metadata._types.get(item.type);
        if (schema !== undefined && schema.schema.category) {
            this._type.category = schema.schema.category;
        }
        if (schema !== undefined && schema.schema.opname) {
            this._type.name = schema.schema.opname;
        }
        this._chain = [];
        this._attributes = [];
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

megengine.Attribute = class {

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

megengine.Tensor = class {

    constructor(name, type, data, littleEndian) {
        this._name = name || '';
        this._type = type;
        this._data = data;
        this._littleEndian = littleEndian;
    }

    get kind() {
        return 'Tensor';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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
        return megengine.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (!this._type.dataType) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        switch (this._type.dataType) {
            case 'boolean':
            case 'uint8':
            case 'qint8':
            case 'int8':
            case 'int16':
            case 'int32':
            case 'int64':
            case 'float16':
            case 'float32':
            case 'float64':
            case 'bfloat16':
                break;
            default:
                context.state = "Tensor data type '" + this._type.dataType + "' is not supported.";
                return context;
        }
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        try {
            context.data = this._data instanceof Uint8Array ? this._data : this._data.peek();
        }
        catch (err) {
            context.state = err.message;
            return context;
        }

        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const results = [];
        const dimensions = (context.dimensions.length == 0) ? [1] : context.dimensions;
        const size = dimensions[dimension];
        if (dimension == dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'boolean':
                        results.push(context.dataView.getUint8(context.index) === 0 ? false : true);
                        context.index++;
                        context.count++;
                        break;
                    case 'uint8':
                        results.push(context.dataView.getUint8(context.index));
                        context.index++;
                        context.count++;
                        break;
                    case 'qint8':
                    case 'int8':
                        results.push(context.dataView.getInt8(context.index));
                        context.index++;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.dataView.getInt16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.dataView.getInt32(context.index, this._littleEndian));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.dataView.getInt64(context.index, this._littleEndian));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.dataView.getFloat16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.dataView.getFloat32(context.index, this._littleEndian));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.dataView.getFloat64(context.index, this._littleEndian));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'bfloat16':
                        results.push(context.dataView.getBfloat16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    default:
                        throw new megengine.Error("Unsupported tensor data type '" + context.dataType + "'.");
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => megengine.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value && (value instanceof base.Int64 || value instanceof base.Uint64)) {
            return indentation + value.toString();
        }
        if (typeof value == 'string') {
            return indentation + value;
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

megengine.Execution = class extends python.Execution {

    constructor(sources, exceptionCallback) {
        super(sources, exceptionCallback);
        this.registerType('megengine.traced_module.traced_module.TracedModule', class { });
        this.registerType('megengine.module.module.Module', class { });
        this.registerType('megengine.traced_module.node.ModuleNode', class { });
        this.registerType('megengine.traced_module.node.NodeMixin', class { });
        this.registerType('megengine.traced_module.pytree.ArgsIndex', class { });
        this.registerType('megengine.traced_module.node.TensorNode', class { });
        this.registerType('megengine.traced_module.traced_module.InternalGraph', class { });
        this.registerType('megengine.traced_module.expr.GetAttr', class { });
        this.registerType('megengine.traced_module.expr.Input', class { });
        this.registerType('megengine.traced_module.expr.CallMethod', class { });
        this.registerType('megengine.core._imperative_rt.common.CompNode', class { });
        this.registerType('megengine.traced_module.traced_module.NameSpace', class { });
        this.registerType('megengine.traced_module.expr.CallFunction', class { });
        this.registerType('megengine.traced_module.expr.Apply', class { });
        this.registerType('megengine.traced_module.serialization._ModuleState', class { });
        this.registerType('megengine.core._imperative_rt.ops.GetVarShape', class { });
        this.registerType('megengine.tensor.Parameter', class {
            constructor(data, dtype, device) {
                this.data = data;
                this.dtype = dtype;
                this.device = device;
            }
        });
        this.registerType('megengine.traced_module.pytree.TreeDef', class {
            toString() {
                let content = '';
                for (const child of this.children_defs) {
                    content += child.toString() + ",";
                }
                if (typeof this.type === "string") {return this.type.split(".").slice(-1) + '(' + content + ')';}
                return this.type.__name__ + '(' + content + ')';
            }
        });
        this.registerType('megengine.traced_module.pytree.LeafDef', class {
            toString() {
                let content = '';
                if (this.const_val !== null) {content += this.const_val;}
                else {content += '[';}
                for (var t of Object.values(this.type)) {
                    content += t.__name__;
                }
                content += ']';
                return content;
            }
        });
        this.registerType('megengine.tensor.Tensor', class {
            constructor(data, dtype, device) {
                this.data = data;
                this.dtype = dtype;
                this.device = device;
            }
        });

        this.registerType('megengine.core.tensor.dtype.QuantDtypeMeta', class {
            constructor(name, cname, np_dtype, qmin, qmax, is_signed) {
                this.name = name;
                this.cname = cname;
                this.np_dtype = np_dtype;
                this.qmin = qmin;
                this.qmax = qmax;
                this.is_signed = is_signed;
            }
        });

    }
};


megengine.Pickle = class {

    static open(context) {
        return new megengine.Pickle(context.stream);
    }

    constructor(stream) {
        this._stream = stream;
        this._graphs = [this];
    }

    set metadata(value) {
        this._metadata = value;
    }

    set exception(value) {
        this._exceptionCallback = value;
    }

    get format() {
        this._unpickle();
        var version = "1.6";
        var dump_info = this._graphs[0].data.dump_info;
        if (dump_info !== undefined) {
            version = dump_info.version;
        }
        return 'MegEngine ' + version;
    }

    get graphs() {
        this._unpickle();
        return this._graphs;
    }

    get littleEndian() {
        this._unpickle();
        return this._littleEndian;
    }

    _unpickle() {
        if (!this._stream) {
            return;
        }

        const data = this._stream.length < 0x7ffff000 ? this._stream.peek() : this._stream;
        const execution = new megengine.Execution(null, this._exceptionCallback);
        const unpickler = python.Unpickler.open(data, execution);

        this._stream = null;
        this._exceptionCallback = null;

        const obj = unpickler.load();
        if (!obj) {
            throw new megengine.Error('File format is not megengine.');
        }
        if (obj === 'None') {
            throw new megengine.Error("File contains 'None' root object.");
        }

        this._graphs = megengine.Utility.find(obj);
    }
};


megengine.Utility = class {

    static isTensor(obj) {
        const name = obj && obj.__class__ ? obj.__class__.__module__ : null;
        switch (name) {
            case 'megengine.tensor':
                return obj.__class__.__name__ === 'Tensor' || obj.__class__.__name__ === "Parameter";
            default:
                return false;
        }
    }

    static isModule(obj) {
        return obj && (obj.state || obj._forward_pre_hooks);
    }

    static isTreeDef(obj) {
        return obj.__class__.__name__ === "TreeDef" || obj.__class__.__name__ === "LeafDef";
    }

    static formatTreeDef(treedef) {
        if (!megengine.Utility.isTreeDef(treedef)) {
            throw new megengine.Error("formatTreeDef gets invalid argument");
        }
        if (treedef.__class__.__name__ === "TreeDef") {
            var type = typeof treedef.type !== "string" ? treedef.type.__name__ : treedef.type.split(".").slice(-1)[0];
            var child_defs = [];
            for (const child of treedef.children_defs) {
                child_defs.push(megengine.Utility.formatTreeDef(child));
            }
            switch (type) {
                case "tuple":
                    return "(" + child_defs.join(",") + ")";
                case "slice":
                    return child_defs.join(":");
                case "list":
                    return "[" + child_defs.join(",") + "]";
                case "dict": {
                    let content = "";
                    for (var i = 0; i < this.children_defs.length; i++) {
                        content += this.aux_data[i] + ":" + child_defs[i];
                    }
                    return "{" + content + "}";
                }
                default:
                    return type + "(" + child_defs.join(",") + ")";
            }
        }
        if (treedef.const_val !== null) { return treedef.const_val; }
        else if (treedef.type[0].__module__ !== undefined) { return treedef.type[0].__name__; }
        return "None";
    }



    static getTensorType(dtype, shape) {
        var dt = dtype !== null ? dtype.__name__ : null;
        return new megengine.TensorType(dt, new megengine.TensorShape(shape));

    }

    static getOpNode(metadata, item, expr, state) {
        const op = new megengine.Node(metadata, null, item);
        var inp_idx = 0;
        for (const i of expr.inputs) {
            if (i.__class__.__name__ !== "ModuleNode") {
                var initializer = null;
                var inp_name = "inp" + inp_idx;
                if (i.initializer !== undefined) {
                    initializer = i.initializer;
                }
                op._inputs.push(new megengine.Parameter(inp_name, true, [new megengine.Argument(i._fullname, megengine.Utility.getTensorType(i._dtype, i._shape), initializer)]));
                inp_idx += 1;
            }
        }
        var out_idx = 0;
        var qparams = null;
        for (const o of expr.outputs) {
            if (o._qparams !== null) {qparams = o._qparams[1];}
            op._outputs.push(new megengine.Parameter("out" + out_idx, true, [new megengine.Argument(o._fullname, megengine.Utility.getTensorType(o._dtype, o._shape), null)]));
        }
        if (qparams !== null) {
            state = state === null? {} : state;
            state["scale"] = qparams.scale;
            state["zero_point"] = qparams.zero_point;
            state["quant_dtype_meta"] = qparams.dtype_meta;
        }
        if (state !== null) {
            for (const key in state) {
                if (!key.startsWith("_") && !megengine.Utility.isModule(state[key])) {
                    if (!megengine.Utility.isTensor(state[key])) {op._attributes.push(new megengine.Attribute(null, key, state[key] !== null ? state[key] : "None"));}
                    else {op._inputs.push(new megengine.Parameter(key, true, [new megengine.Argument("", megengine.Utility.getTensorType(state[key].dtype, state[key].data.shape), megengine.Utility.createTensor(key, state[key], true))]));}
                }

            }
        }
        return op;
    }

    static getModuleType(obj) {
        if (obj.module !== undefined) {return obj.module[0] + "." + obj.module[1];}
        return obj.__class__.__module__ + "." + obj.__class__.__name__;
    }

    static getFunctionType(obj) {
        if (obj.func.__module__ !== undefined) {return obj.func.__module__ + "." + obj.func.__name__;}
        return obj.func[0] + "." + obj.func[1];
    }

    static createTensor(name, tensor, littleEndian) {
        const storage = tensor.data;
        const size = storage.shape;
        const type = new megengine.TensorType(tensor.dtype.__name__, new megengine.TensorShape(size));
        return new megengine.Tensor(name || '', type, storage.data, littleEndian);
    }

    static find(data) {
        const root = data && data.__class__.__name__ === "TracedModule"?[{ "name": "", "data": data }]:null;
        if (root) {
            for (const graph of root) {
                graph.type = 'module';
            }
            return root;
        }
        throw new megengine.Error('File does not contain root module or state dictionary.');

    }

};

megengine.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading megengine model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = megengine.ModelFactory;
}



