
// Experimental

var megengine = {};
var flatbuffers = require('./flatbuffers');

megengine.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream && stream.length >= 12) {
            const buffer = stream.peek(12);
            const size = buffer[0] + (buffer[1] << 8) + (buffer[2] << 16) + (buffer[3] << 24);
            if (size === (stream.length - 4)) {
                const reader = flatbuffers.BinaryReader.open(buffer.slice(4, 12));
                if (reader.identifier === 'mgv2') {
                    return 'megengine.mge';
                }
            }
        }
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ === 'megengine.traced_module.traced_module' && obj.__class__.__name__ === 'TracedModule') {
            return 'megengine.tm';
        }
        return '';
    }

    open(context, match) {
        return context.metadata('megengine-metadata.json').then((metadata) => {
            switch (match) {
                case 'megengine.tm': {
                    const obj = context.open('pkl');
                    return new megengine.Model(metadata, obj, match);
                }
                case 'megengine.mge': {
                    return context.require('./megengine-schema').then(() => {
                        megengine.schema = flatbuffers.get('megengine').mgb.serialization.fbs;
                        let model = null;

                        const stream = context.stream;
                        try {
                            stream.skip(4);
                            const reader = flatbuffers.BinaryReader.open(stream);
                            model = megengine.schema.v2.Model.create(reader);
                        }
                        catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new megengine.Error('File format is not megengine.Model (' + message.replace(/\.$/, '') + ').');
                        }
                        return new megengine.Model(metadata, model, match);
                    });
                }
                default: {
                    throw new megengine.Error("Unsupported megengine format '" + match + "'.");
                }
            }
        });
    }
};

megengine.Model = class {

    constructor(metadata, obj, modelType) {
        switch (modelType) {
            case 'megengine.tm': {
                this._format = 'MegEngine' + (obj.dump_info && obj.dump_info.version ? ' v' + obj.dump_info.version : '');
                break;
            }
            case 'megengine.mge': {
                this._format = 'MegEngine Mge' + (obj.model_version ? ' v' + obj.model_version : '');
                break;
            }
            default: {
                break;
            }
        }
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
        this._name = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const loadGraph = (tmodule, igraph, context, namePrefix, metadata, isRoot) => {
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
                let inpIdx = 0;
                for (const i of expr.inputs) {
                    if (i.__class__.__name__ !== 'ModuleNode') {
                        const initializer = i.initializer !== undefined ? i.initializer : null;
                        const inpName = 'inp' + inpIdx;
                        const type = getTensorType(i._dtype, i._shape);
                        const argument = new megengine.Argument(i._fullname, type, initializer);
                        const parameter = new megengine.Parameter(inpName, true, [ argument ]);
                        op._inputs.push(parameter);
                        inpIdx += 1;
                    }
                }
                const outIdx = 0;
                let qparams = null;
                for (const o of expr.outputs) {
                    if (o._qparams !== null) {
                        qparams = o._qparams[1];
                    }
                    const type = getTensorType(o._dtype, o._shape);
                    const argument = new megengine.Argument(o._fullname, type, null);
                    const parameter = new megengine.Parameter('out' + outIdx, true, [ argument ]);
                    op._outputs.push(parameter);
                }
                if (qparams !== null) {
                    state = state === null ? {} : state;
                    state.scale = qparams.scale;
                    state.zero_point = qparams.zero_point;
                    state.quant_dtype_meta = qparams.dtype_meta;
                }
                if (state !== null) {
                    for (const key in state) {
                        const isModule = (obj) => {
                            return obj && (obj.state || obj._forward_pre_hooks);
                        };
                        if (!key.startsWith('_') && !isModule(state[key])) {
                            if (!isTensor(state[key])) {
                                const attribute = new megengine.Attribute(null, key, state[key] !== null ? state[key] : 'None');
                                op._attributes.push(attribute);
                            }
                            else {
                                const tensor = state[key];
                                const type = getTensorType(tensor.dtype, tensor.data.shape);
                                const data = tensor.data.data;
                                const initializer = new megengine.Tensor(key, type, data);
                                const argument = new megengine.Argument('', type, initializer);
                                const parameter = new megengine.Parameter(key, true, [ argument ]);
                                op._inputs.push(parameter);
                            }
                        }
                    }
                }
                return op;
            };
            if (isRoot) {
                for (const node of igraph._inputs) {
                    if (node.__class__.__name__ !== 'ModuleNode') {
                        const type = getTensorType(node._dtype, node._shape);
                        const argument = new megengine.Argument(node._name, type, null);
                        const parameter = new megengine.Parameter(node._name, true, [ argument ]);
                        this._inputs.push(parameter);
                    }
                }
                for (const node of igraph._outputs) {
                    const type = getTensorType(node._dtype, node._shape);
                    const argument = new megengine.Argument(node._name, type, null);
                    const parameter = new megengine.Parameter(node._name, true, [ argument ]);
                    this._outputs.push(parameter);
                }
            }
            const parseGetAttr = (tmodule, getAttrExpr) => {
                let attrName = getAttrExpr.name.split('.');
                while (getAttrExpr.inputs[0].expr.__class__.__name__ === 'GetAttr') {
                    getAttrExpr = getAttrExpr.inputs[0].expr;
                    attrName = getAttrExpr.name.split('.').concat(attrName);
                }
                let attrObj = tmodule;
                for (const n of attrName) {
                    attrObj = attrObj[n];
                }
                return attrObj;
            };
            const parseArgs = (args, kwargs, meta) => {
                const state = {};
                let argIdx = 0;
                let attrName = '';
                const processArgs = (inp, startIdx) => {
                    while (typeof inp === 'string' && inp.indexOf('Tensor') !== -1) {
                        inp = inp.replace('Tensor', 'inp' + startIdx);
                        startIdx += 1;
                    }
                    return [ inp, startIdx ];
                };
                const formatTreeDef = (obj) => {
                    if (obj.__class__.__name__ !== 'TreeDef' && obj.__class__.__name__ !== 'LeafDef') {
                        throw new megengine.Error("Invalid argument '" + obj.__class__.__name__ + "'.");
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
                let inpIdx = 0;
                for (const arg of args.children_defs) {
                    if (meta.attributes === undefined || (meta.attributes.length !== args.children_defs.length && meta.varargs === null)) {
                        attrName = 'arg' + argIdx;
                    }
                    else if (argIdx < meta.attributes.length) {
                        attrName = meta.attributes[argIdx].name;
                    }
                    else {
                        attrName = meta.varargs + (argIdx - meta.attributes.length);
                    }
                    const rst = processArgs(formatTreeDef(arg), inpIdx);
                    state[attrName] = rst[0];
                    inpIdx = rst[1];
                    argIdx += 1;
                }
                for (let i = 0; i < kwargs.children_defs.length; i++) {
                    const rst = processArgs(formatTreeDef(kwargs.children_defs[i]), inpIdx);
                    inpIdx = rst[1];
                    state[kwargs.aux_data[i]] = rst[0];
                }
                return state;
            };
            const getName = (context, name) => {
                let rst = name;
                while (context.get(rst) !== undefined) {
                    if (rst === context.get(rst)) {
                        return rst;
                    }
                    rst = context.get(rst);
                }
                return rst;
            };
            const getFullName = (prefix, name) => {
                return prefix === '' ? name : prefix + '_' + name;
            };
            for (const expr of expressions) {
                const type = expr.__class__.__name__;
                for (const i of expr.inputs) {
                    i._fullname = getName(context, getFullName(namePrefix, i._name));
                }
                for (const o of expr.outputs) {
                    o._fullname = getName(context, getFullName(namePrefix, o._name));
                }
                switch (type) {
                    case 'Input': {
                        break;
                    }
                    case 'GetAttr': {
                        if (expr.outputs[0].__class__.__name__ === 'TensorNode') {
                            const tensor = parseGetAttr(tmodule, expr);
                            const type = getTensorType(tensor.dtype, tensor.data.shape);
                            const data = tensor.data.data;
                            expr.outputs[0].initializer = new megengine.Tensor(expr.name, type, data);
                        }
                        break;
                    }
                    case 'Constant': {
                        if (expr.outputs[0].__class__.__name__ === 'TensorNode') {
                            const tensor = expr.value;
                            const type = getTensorType(tensor.dtype, tensor.data.shape);
                            const data = tensor.data.data;
                            expr.outputs[0].initializer = new megengine.Tensor('', type, data);
                        }
                        break;
                    }
                    case 'CallMethod': {
                        if (expr.method === '__call__') {
                            const getAttrExpr = expr.inputs[0].expr;
                            const calledModule = parseGetAttr(tmodule, getAttrExpr);
                            const getModuleType = (obj) => {
                                if (obj.module !== undefined) {
                                    return obj.module[0] + '.' + obj.module[1];
                                }
                                return obj.__class__.__module__ + '.' + obj.__class__.__name__;
                            };
                            const moduleType = calledModule.__class__.__name__ !== 'TracedModule' ? getModuleType(calledModule) : 'TracedModule';
                            if (moduleType === 'TracedModule') {
                                const moduleName = expr.outputs[0]._name.endsWith("_out") ? expr.outputs[0]._name.substring(0, expr.outputs[0]._name.length - 4) : expr.outputs[0]._name;
                                const prefix = getFullName(namePrefix, moduleName);
                                const internalGraph = calledModule.argdef_graph_map[expr.arg_def.toString()];
                                for (let i = 0; i < expr.inputs.length; i++) {
                                    const actualName = getFullName(namePrefix, expr.inputs[i]._name);
                                    const internalName = getFullName(prefix, internalGraph._inputs[i]._name);
                                    context.set(internalName, actualName);
                                }
                                for (let i = 0; i < expr.outputs.length; i++) {
                                    const actualName = getFullName(namePrefix, expr.outputs[i]._name);
                                    const internalName = getFullName(prefix, internalGraph._outputs[i]._name);
                                    if (context.get(internalName) !== undefined) {
                                        context.set(actualName, context.get(internalName));
                                    }
                                    else {
                                        context.set(internalName, actualName);
                                    }
                                }
                                loadGraph(calledModule, internalGraph, context, prefix, metadata, false);
                                continue;
                            }
                            const item = { 'name': '', 'type': moduleType };
                            let state = calledModule.__class__.__name__ !== 'TracedModule' ? calledModule.state : calledModule;
                            if (state === undefined) {
                                state = calledModule;
                            }
                            const node = getOpNode(metadata, item, expr, state);
                            this._nodes.push(node);
                        }
                        else {
                            const item = { 'name': '', 'type': expr.method };
                            const args = expr.arg_def.children_defs[0];
                            const kwargs = expr.arg_def.children_defs[1];
                            const schema = metadata.type(expr.method);
                            const state = parseArgs(args, kwargs, schema);
                            const node = getOpNode(metadata, item, expr, state);
                            this._nodes.push(node);
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
                        const state = parseArgs(args, kwargs, schema);
                        const node = getOpNode(metadata, item, expr, state);
                        this._nodes.push(node);
                        break;
                    }
                    case 'Apply': {
                        const opdef = expr.opdef_state ? expr.opdef_state.opdef_type : expr.opdef.type;
                        const item = { 'name': '', 'type': opdef.__module__ + '.' + opdef.__name__ };
                        const node = getOpNode(metadata, item, expr, expr.opdef_state);
                        this._nodes.push(node);
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
        };
        if(obj.argdef_graph_map) {
            const graph = Object.values(obj.argdef_graph_map)[0];
            loadGraph(obj, graph, new Map(), '', metadata, true);
            return;
        }
        const extraInfoNameset = new Set();
        const getExtraInfo = (opr) => {
            let name = opr.name;
            while (extraInfoNameset.has(name)) {
                for (const id of opr.inputs) {
                    name = name + '[' + id + ']';
                }
            }
            extraInfoNameset.add(name);
            const type = opr.type.replace(/V(\d+)$/, '');
            const args = [];
            if (opr.tensors.length !== 0) {
                const tensor = opr.tensors[0];
                const type = new megengine.TensorType(tensor.dtype.type, new megengine.TensorShape(tensor.shape));
                const data = tensor.data.byteLength !== 0 ? tensor.data.slice(0) : undefined;
                const initializer = opr.type === 'Host2DeviceCopy' ? undefined : new megengine.Tensor('', type, data);
                let quantization;
                if(tensor.dtype.param) {
                    quantization = {scale: tensor.dtype.param.scale, zeroPoint: tensor.dtype.param.zero_point};
                }
                const argument = new megengine.Argument(name, type, initializer, quantization);
                args.push(argument);
            }
            else {
                const argument = new megengine.Argument(name);
                args.push(argument);
            }
            return {name: name, type: type, args: args};
        };
        const getAllOprAndTensor = (oprs) => {
            const allOprAndTensor = new Map();
            let keyId = 0;
            for (const opr of oprs) {
                if (opr.type === 'MultipleDeviceTensorWithFormatHolder') {
                    for (var id = 0; id < opr.outputs.length; id++) {
                        const name = obj.middle_tensors[opr.outputs[id]].name;
                        const tensors = [ opr.tensors[id] ];
                        const type = 'ImmutableTensor';
                        allOprAndTensor.set(keyId, { name: name, type: type, tensors: tensors });
                        const _opr = allOprAndTensor.get(keyId);
                        _opr.extraInfo = getExtraInfo(_opr);
                        keyId = keyId + 1;
                    }
                }
                else {
                    if (opr.outputs.length !== 1) {
                        throw new megengine.Error('The length of opr.outputs in the model must be one');
                    }
                    opr.name = obj.middle_tensors[opr.outputs[0]].name;
                    allOprAndTensor.set(keyId, opr);
                    const _opr = allOprAndTensor.get(keyId);
                    _opr.extraInfo = getExtraInfo(_opr);
                    keyId = keyId + 1;
                }
            }
            return allOprAndTensor;
        };
        const allOprAndTensor = getAllOprAndTensor(obj.oprs);
        for (const pair of allOprAndTensor) {
            const opr = pair[1];
            if (opr.type === 'Host2DeviceCopy') {
                const parameter = new megengine.Parameter('input', true, opr.extraInfo.args);
                this._inputs.push(parameter);
            }
            else if (opr.param && opr.tensors.length === 0) {
                const node = new megengine.Node(metadata, opr, allOprAndTensor);
                this._nodes.push(node);
            }
        }
        for (let i = 0; i < obj.output_vars_idx.length; i++) {
            const id = obj.output_vars_idx[i].compact_id;
            const out_type = 'output' + (i === 0 ? '' : i);
            const parameter = new megengine.Parameter(out_type, true, allOprAndTensor.get(id).extraInfo.args);
            this._outputs.push(parameter);
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

    constructor(name, type, initializer, quantization) {
        if (typeof name !== 'string') {
            throw new megengine.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._initializer = initializer;
        this._type = type;
        if(quantization && this._type.dataType.startsWith('q')) {
            this._scale = quantization.scale;
            this._zeroPoint = quantization.zeroPoint;
        }
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

    get quantization() {
        if (this._scale !== undefined && this._zeroPoint !== undefined) {
            return this._scale.toString() + ' * ' + (this._zeroPoint == 0 ? 'q' : '(q - ' + this._zeroPoint.toString() + ')');
        }
        return undefined;
    }
};

megengine.Node = class {

    constructor(metadata, item, allOprAndTensor) {
        this._name = '';
        this._type = Object.assign({}, metadata.type(item.type));
        this._type.name = this._type.name.replace(/V(\d+)$/, '');
        if (this._type.name.length > 4 && this._type.name.startsWith('__') && this._type.name.endsWith('__')) {
            this._type.name = this._type.name.substring(2, this._type.name.length - 2);
        }
        this._type.category = this._type.category? this._type.category: metadata.type(item.type.replace(/V(\d+)$/, '')).category;

        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        this._attributes = [];

        if(item.inputs && item.outputs && item.param) {
            const inputSchemas = this._type && this._type.inputs ? [ ...this._type.inputs ] : [];
            for (let i = 0; i < item.inputs.length; i++) {
                const inputOpr = allOprAndTensor.get(item.inputs[i]);
                const inputSchema = inputSchemas.length > 0 ? inputSchemas.shift() : { name: ('input' + i) };
                const parameter = new megengine.Parameter(inputSchema.name, true, inputOpr.extraInfo.args);
                this._inputs.push(parameter);
            }
            const outputSchemas = this._type && this._type.outputs ? [ ...this._type.outputs ] : [];
            for (let i = 0; i < item.outputs.length; i++) {
                const outputOpr = allOprAndTensor.get(item.outputs[i]);
                const outputSchema = outputSchemas.length > 0 ? outputSchemas.shift() : { name: ('output' + i) };
                const parameter = new megengine.Parameter(outputSchema.name, true, outputOpr.extraInfo.args);
                this._outputs.push(parameter);
            }
            for (const pair of Object.entries(item.param)) {
                const name = pair[0];
                const value = pair[1];
                if (value !== null) {
                    const attribute = new megengine.Attribute(metadata.attribute(item.param.constructor.name, name), name, value);
                    this._attributes.push(attribute);
                }
            }
        }
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
        this._type = metadata ? metadata.type : null;
        this._name = name;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        if (this._name === 'training') {
            this._visible = false;
            this._type = 'boolean';
        }
        if (megengine.schema) {
            if (megengine.schema.param[this._type]) {
                this._value = megengine.Utility.enum(megengine.schema.param, this._type, this._value);
            }
            else if (megengine.schema[this._type]) {
                this._value = megengine.Utility.enum(megengine.schema, this._type, this._value);
            }
            else if (megengine.schema.v2[this._type]) {
                this._value = megengine.Utility.enum(megengine.schema.v2, this._type, this._value);
            }
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

    constructor(name, type, data) {
        this._name = name || '';
        this._type = type;
        this._data = data;
    }

    get category() {
        return 'Tensor';
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
        this._dataTypeName = dataType;
        this._shape = shape;
        const dtype_class = 'DTypeEnum';
        if (megengine.schema && megengine.schema[dtype_class]) {
            this._value = ArrayBuffer.isView(dataType) ? Array.from(dataType) : dataType;
            this._dataTypeName = megengine.Utility.enum(megengine.schema, dtype_class, this._value).toLowerCase();
        }
        megengine.TensorType._dataTypeMap = megengine.TensorType._dataTypeMap || new Map([
            [ 'bool', 'boolean' ],
            [ 'byte', 'uint8' ], [ 'quantizeds4asymm', 'uint8' ], [ 'quantizeds8asymm', 'uint8' ], [ 'uintb4', 'uint8' ],
            [ 'quantizeds1', 'int8' ], [ 'quantizeds4', 'int8' ], [ 'quantizeds8', 'int8' ], [ 'intb1', 'int8' ], [ 'intb2', 'int8' ], [ 'intb4', 'int8' ], [ 'qint8', 'int8' ],
            [ 'quantizeds16', 'int16' ],
            [ 'quantizeds32', 'int32' ]
        ]);
        this._dataType = megengine.TensorType._dataTypeMap.get(this._dataTypeName) || this._dataTypeName;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataTypeName + this._shape.toString();
    }
};

megengine.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = Array.from(dimensions || []);
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

megengine.Utility = class {

    static enum(schema, name, value) {
        const type = name && schema ? schema[name] : undefined;
        if (type) {
            megengine.Utility._enums = megengine.Utility._enums || new Map();
            if (!megengine.Utility._enums.has(name)) {
                const map = new Map(Object.keys(type).map((key) => [ type[key], key ]));
                megengine.Utility._enums.set(name, map);
            }
            const map = megengine.Utility._enums.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
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
