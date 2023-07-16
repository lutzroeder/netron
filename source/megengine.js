
// Experimental

var megengine = {};
var flatbuffers = require('./flatbuffers');

megengine.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream && stream.length >= 12) {
            let buffer = stream.peek(12);
            const tag = String.fromCharCode.apply(null, buffer);
            const position = tag.startsWith('mgbtest0') ? 12 : 0;
            if (stream.length > (position + 12)) {
                buffer = stream.peek(24).slice(position, position + 12);
                const size = buffer[0] + (buffer[1] << 8) + (buffer[2] << 16) + (buffer[3] << 24);
                if (position > 0 || size === (stream.length - position - 4)) {
                    const reader = flatbuffers.BinaryReader.open(buffer.slice(4, 12));
                    if (reader.identifier === 'mgv2') {
                        return 'megengine.mge';
                    }
                }
            }
            for (const value of [ 'mgb0001', 'mgb0000a', 'MGBS', 'MGBC' ]) {
                if (tag.startsWith(value)) {
                    return 'megengine.' + value;
                }
            }
        }
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ === 'megengine.traced_module.traced_module' && obj.__class__.__name__ === 'TracedModule') {
            return 'megengine.tm';
        }
        return '';
    }

    async open(context, target) {
        const metadata = await context.metadata('megengine-metadata.json');
        switch (target) {
            case 'megengine.tm': {
                const obj = context.open('pkl');
                return new megengine.Model(metadata, obj, target);
            }
            case 'megengine.mge': {
                await context.require('./megengine-schema');
                megengine.schema = flatbuffers.get('megengine').mgb.serialization.fbs;
                let model = null;
                const stream = context.stream;
                try {
                    const buffer = stream.peek(12);
                    const tag = String.fromCharCode.apply(null, buffer);
                    stream.skip(tag.startsWith('mgbtest0') ? 12 : 0);
                    stream.skip(4);
                    const reader = flatbuffers.BinaryReader.open(stream);
                    model = megengine.schema.v2.Model.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new megengine.Error('File format is not megengine.Model (' + message.replace(/\.$/, '') + ').');
                }
                return new megengine.Model(metadata, model, target);
            }
            default: {
                throw new megengine.Error("Unsupported MegEngine format '" + target.replace(/^megengine\./, '') + "'.");
            }
        }
    }
};

megengine.Model = class {

    constructor(metadata, obj, type) {
        this.format = 'MegEngine';
        if (type === 'megengine.tm') {
            this.format += (obj.dump_info && obj.dump_info.version ? ' v' + obj.dump_info.version : '');
        } else if (type === 'megengine.mge') {
            this.format += ' Mge' + (obj.model_version ? ' v' + obj.model_version : '');
        }
        this.graphs = [ new megengine.Graph(metadata, obj) ];
    }
};

megengine.Graph = class {

    constructor(metadata, obj) {
        this.name = '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        const value = (name, type, tensor) => {
            if (tensor && name.length === 0) {
                return new megengine.Value(name, type || null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new megengine.Value(name, type || null, tensor || null));
            } else if ((type && !type.equals(values.get(name).type)) || tensor) {
                throw new megengine.Error("Duplicate value '" + name + "'.");
            }
            return values.get(name);
        };
        const loadGraph = (tmodule, igraph, context, namePrefix, metadata, isRoot) => {
            const expressions = igraph._exprs;
            const getTensorType = (dtype, shape) => {
                dtype = dtype ? dtype.__name__ : null;
                return new megengine.TensorType(dtype, new megengine.TensorShape(shape));
            };
            const getOpNode = (metadata, item, expr, state) => {
                const node = new megengine.Node(metadata, item);
                let inpIdx = 0;
                for (const i of expr.inputs) {
                    if (i.__class__.__name__ !== 'ModuleNode') {
                        const initializer = i.initializer !== undefined ? i.initializer : null;
                        const name = 'inp' + inpIdx;
                        const type = getTensorType(i._dtype, i._shape);
                        const argument = new megengine.Argument(name, [ value(i._fullname, type, initializer) ]);
                        node.inputs.push(argument);
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
                    const argument = new megengine.Argument('out' + outIdx, [ value(o._fullname, type, null) ]);
                    node.outputs.push(argument);
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
                        const isTensor = (obj) => {
                            return obj && obj.__class__ && obj.__class__.__module__ == 'megengine.tensor' && (obj.__class__.__name__ === 'Tensor' || obj.__class__.__name__ === 'Parameter');
                        };
                        if (!key.startsWith('_') && !isModule(state[key])) {
                            if (!isTensor(state[key])) {
                                const attribute = new megengine.Attribute(null, key, state[key] !== null ? state[key] : 'None');
                                node.attributes.push(attribute);
                            } else {
                                const tensor = state[key];
                                const type = getTensorType(tensor.dtype, tensor.data.shape);
                                const data = tensor.data.data;
                                const initializer = new megengine.Tensor(key, type, data);
                                const argument = new megengine.Argument(key, [ value('', type, initializer) ]);
                                node.inputs.push(argument);
                            }
                        }
                    }
                }
                return node;
            };
            if (isRoot) {
                for (const node of igraph._inputs) {
                    if (node.__class__.__name__ !== 'ModuleNode') {
                        const type = getTensorType(node._dtype, node._shape);
                        const argument = new megengine.Argument(node._name, [ value(node._name, type, null) ]);
                        this.inputs.push(argument);
                    }
                }
                for (const node of igraph._outputs) {
                    const type = getTensorType(node._dtype, node._shape);
                    const argument = new megengine.Argument(node._name, [ value(node._name, type, null) ]);
                    this.outputs.push(argument);
                }
            }
            const parseGetAttr = (module, expression) => {
                let names = expression.name.split('.');
                while (expression.inputs[0].expr.__class__.__name__ === 'GetAttr') {
                    expression = expression.inputs[0].expr;
                    names = expression.name.split('.').concat(names);
                }
                let obj = module;
                for (const name of names) {
                    obj = obj[name];
                }
                return obj;
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
                    } else if (obj.type[0].__module__ !== undefined) {
                        return obj.type[0].__name__;
                    }
                    return 'None';
                };
                let inpIdx = 0;
                for (const arg of args.children_defs) {
                    if (meta.attributes === undefined || (meta.attributes.length !== args.children_defs.length && meta.varargs === null)) {
                        attrName = 'arg' + argIdx;
                    } else if (argIdx < meta.attributes.length) {
                        attrName = meta.attributes[argIdx].name;
                    } else {
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
            for (const expression of expressions) {
                const type = expression.__class__.__name__;
                for (const input of expression.inputs) {
                    input._fullname = getName(context, getFullName(namePrefix, input._name));
                }
                for (const output of expression.outputs) {
                    output._fullname = getName(context, getFullName(namePrefix, output._name));
                }
                switch (type) {
                    case 'Input': {
                        break;
                    }
                    case 'GetAttr': {
                        if (expression.outputs[0].__class__.__name__ === 'TensorNode') {
                            const tensor = parseGetAttr(tmodule, expression);
                            const type = getTensorType(tensor.dtype, tensor.data.shape);
                            const data = tensor.data.data;
                            expression.outputs[0].initializer = new megengine.Tensor(expression.name, type, data);
                        }
                        break;
                    }
                    case 'Constant': {
                        if (expression.outputs[0].__class__.__name__ === 'TensorNode') {
                            const tensor = expression.value;
                            const type = getTensorType(tensor.dtype, tensor.data.shape);
                            const data = tensor.data.data;
                            expression.outputs[0].initializer = new megengine.Tensor('', type, data);
                        }
                        break;
                    }
                    case 'CallMethod': {
                        if (expression.method === '__call__') {
                            const module = parseGetAttr(tmodule, expression.inputs[0].expr);
                            const getModuleType = (obj) => {
                                if (obj.module !== undefined) {
                                    return obj.module[0] + '.' + obj.module[1];
                                }
                                return obj.__class__.__module__ + '.' + obj.__class__.__name__;
                            };
                            const moduleType = module.__class__.__name__ !== 'TracedModule' ? getModuleType(module) : 'TracedModule';
                            if (moduleType === 'TracedModule') {
                                const moduleName = expression.outputs[0]._name.endsWith("_out") ? expression.outputs[0]._name.substring(0, expression.outputs[0]._name.length - 4) : expression.outputs[0]._name;
                                const prefix = getFullName(namePrefix, moduleName);
                                const internalGraph = module.argdef_graph_map[expression.arg_def.toString()];
                                for (let i = 0; i < expression.inputs.length; i++) {
                                    const actualName = getFullName(namePrefix, expression.inputs[i]._name);
                                    const internalName = getFullName(prefix, internalGraph._inputs[i]._name);
                                    context.set(internalName, actualName);
                                }
                                for (let i = 0; i < expression.outputs.length; i++) {
                                    const actualName = getFullName(namePrefix, expression.outputs[i]._name);
                                    const internalName = getFullName(prefix, internalGraph._outputs[i]._name);
                                    if (context.get(internalName) !== undefined) {
                                        context.set(actualName, context.get(internalName));
                                    } else {
                                        context.set(internalName, actualName);
                                    }
                                }
                                loadGraph(module, internalGraph, context, prefix, metadata, false);
                                continue;
                            }
                            const item = { 'name': '', 'type': moduleType };
                            let state = module.__class__.__name__ !== 'TracedModule' ? module.state : module;
                            if (state === undefined) {
                                state = module;
                            }
                            const node = getOpNode(metadata, item, expression, state);
                            this.nodes.push(node);
                        } else {
                            const item = { 'name': '', 'type': expression.method };
                            const args = expression.arg_def.children_defs[0];
                            const kwargs = expression.arg_def.children_defs[1];
                            const schema = metadata.type(expression.method);
                            const state = parseArgs(args, kwargs, schema);
                            const node = getOpNode(metadata, item, expression, state);
                            this.nodes.push(node);
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
                        const func = getFunctionType(expression);
                        const item = { 'name': '', 'type': func };
                        const args = expression.arg_def.children_defs[0];
                        const kwargs = expression.arg_def.children_defs[1];
                        const schema = metadata.type(func);
                        const state = parseArgs(args, kwargs, schema);
                        const node = getOpNode(metadata, item, expression, state);
                        this.nodes.push(node);
                        break;
                    }
                    case 'Apply': {
                        const opdef = expression.opdef_state ? expression.opdef_state.opdef_type : expression.opdef.type;
                        const item = { 'name': '', 'type': opdef.__module__ + '.' + opdef.__name__ };
                        const node = getOpNode(metadata, item, expression, expression.opdef_state);
                        this.nodes.push(node);
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
        };
        if (obj.argdef_graph_map) {
            const graph = Object.values(obj.argdef_graph_map)[0];
            loadGraph(obj, graph, new Map(), '', metadata, true);
            return;
        }
        const extraInfoNameset = new Set();
        const getExtraInfo = (opr) => {
            let name = opr.name;
            let repeatIdx = 0;
            while (extraInfoNameset.has(name)) {
                for (const id of opr.inputs) {
                    name = name + '[' + id + ']';
                }
                name += repeatIdx;
                repeatIdx += 1;
            }
            extraInfoNameset.add(name);
            const type = opr.type.replace(/V(\d+)$/, '');
            const args = [];
            if (opr.tensors.length !== 0) {
                const tensor = opr.tensors[0];
                const type = new megengine.TensorType(tensor.dtype.type, new megengine.TensorShape(tensor.shape));
                const data = tensor.data.byteLength !== 0 ? tensor.data.slice(0) : undefined;
                const initializer = opr.type === 'Host2DeviceCopy' ? undefined : new megengine.Tensor('', type, data);
                const quantization = tensor.dtype.param ? { scale: tensor.dtype.param.scale, zeroPoint: tensor.dtype.param.zero_point } : null;
                args.push(value(name, type, initializer, quantization));
            } else if (opr.shape) {
                const type = new megengine.TensorType('?', new megengine.TensorShape(opr.shape));
                args.push(value(name, type));
            } else {
                args.push(value(name));
            }
            return { name: name, type: type, args: args };
        };
        const getAllOprAndTensor = (oprs) => {
            const allOprAndTensor = new Map();
            for (const opr of oprs) {
                if (opr.type === 'MultipleDeviceTensorWithFormatHolder' || opr.outputs.length > 1) {
                    if (opr.type === 'MultipleDeviceTensorWithFormatHolder' || opr.type === 'MultipleDeviceTensorHolder') {
                        opr.type = 'ImmutableTensor';
                    }
                    for (var id = 0; id < opr.outputs.length; id++) {
                        const keyId = opr.outputs[id];
                        const name = obj.middle_tensors[keyId] ? obj.middle_tensors[keyId].name : String(keyId);
                        const type = opr.type;
                        const tensors = opr.tensors.length ? [opr.tensors[id]] : [];
                        const onlyShape = obj.middle_tensors[keyId] ? obj.middle_tensors[keyId].shape : [];
                        allOprAndTensor.set(keyId, { name: name, type: type, tensors: tensors, shape: onlyShape, inputs: opr.inputs, outputs: opr.outputs });
                        const _opr = allOprAndTensor.get(keyId);
                        _opr.extraInfo = getExtraInfo(_opr);
                    }
                } else {
                    const keyId = opr.outputs[0];
                    opr.name = obj.middle_tensors[keyId] ? obj.middle_tensors[keyId].name : String(keyId);
                    if (obj.middle_tensors[keyId] && obj.middle_tensors[keyId].shape) {
                        opr.shape = obj.middle_tensors[keyId].shape;
                    }
                    allOprAndTensor.set(keyId, opr);
                    const _opr = allOprAndTensor.get(keyId);
                    _opr.extraInfo = getExtraInfo(_opr);
                }
            }
            return allOprAndTensor;
        };
        const allOprAndTensor = getAllOprAndTensor(obj.oprs);
        for (const entry of allOprAndTensor) {
            const op = entry[1];
            if (op.type === 'Host2DeviceCopy') {
                const argument = new megengine.Argument('input', op.extraInfo.args);
                this.inputs.push(argument);
            } else if (op.type !== 'ImmutableTensor') {
                this.nodes.push(new megengine.Node(metadata, op, allOprAndTensor));
            }
        }
        for (let i = 0; i < obj.output_vars_idx.length; i++) {
            const id = obj.output_vars_idx[i].compact_id;
            const out_type = 'output' + (i === 0 ? '' : i);
            const argument = new megengine.Argument(out_type, allOprAndTensor.get(id).extraInfo.args);
            this.outputs.push(argument);
        }
    }
};

megengine.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

megengine.Value = class {

    constructor(name, type, initializer, quantization) {
        if (typeof name !== 'string') {
            throw new megengine.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        this.initializer = initializer;
        this._type = type;
        if (quantization && this._type.dataType.startsWith('q')) {
            this._scale = quantization.scale;
            this._zeroPoint = quantization.zeroPoint;
        }
    }

    get type() {
        if (this.initializer) {
            return this.initializer.type;
        }
        return this._type;
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
        this.name = '';
        this.type = Object.assign({}, metadata.type(item.type));
        this.type.name = this.type.name.replace(/V(\d+)$/, '');
        if (this.type.name.length > 4 && this.type.name.startsWith('__') && this.type.name.endsWith('__')) {
            this.type.name = this.type.name.substring(2, this.type.name.length - 2);
        }
        this.type.category = this.type.category? this.type.category: metadata.type(item.type.replace(/V(\d+)$/, '')).category;
        this.inputs = [];
        this.outputs = [];
        this.chain = [];
        this.attributes = [];
        if (item.inputs && item.outputs) {
            const inputSchemas = this.type && this.type.inputs ? [ ...this.type.inputs ] : [];
            for (let i = 0; i < item.inputs.length; i++) {
                const inputOpr = allOprAndTensor.get(item.inputs[i]);
                const inputSchema = inputSchemas.length > 0 ? inputSchemas.shift() : { name: ('input' + i) };
                const argument = new megengine.Argument(inputSchema.name, inputOpr.extraInfo.args);
                this.inputs.push(argument);
            }
            const outputSchemas = this.type && this.type.outputs ? [ ...this.type.outputs ] : [];
            for (let i = 0; i < item.outputs.length; i++) {
                const outputOpr = allOprAndTensor.get(item.outputs[i]);
                const outputSchema = outputSchemas.length > 0 ? outputSchemas.shift() : { name: ('output' + i) };
                const argument = new megengine.Argument(outputSchema.name, outputOpr.extraInfo.args);
                this.outputs.push(argument);
            }
            if (item.param) {
                for (const pair of Object.entries(item.param)) {
                    const name = pair[0];
                    const value = pair[1];
                    if (value !== null) {
                        const attribute = new megengine.Attribute(metadata.attribute(item.param.constructor.name, name), name, value);
                        this.attributes.push(attribute);
                    }
                }
            }
        }
    }
};

megengine.Attribute = class {

    constructor(metadata, name, value) {
        this.type = metadata ? metadata.type : null;
        this.name = name;
        this.value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        if (this.name === 'training') {
            this.visible = false;
            this.type = 'boolean';
        }
        if (megengine.schema) {
            if (megengine.schema.param[this.type]) {
                this.value = megengine.Utility.enum(megengine.schema.param, this.type, this.value);
            } else if (megengine.schema[this.type]) {
                this.value = megengine.Utility.enum(megengine.schema, this.type, this.value);
            } else if (megengine.schema.v2[this.type]) {
                this.value = megengine.Utility.enum(megengine.schema.v2, this.type, this.value);
            }
        }
    }
};

megengine.Tensor = class {

    constructor(name, type, data) {
        this.category = 'Tensor';
        this.name = name || '';
        this.type = type;
        this.values = data;
    }
};

megengine.TensorType = class {

    constructor(dataType, shape) {
        dataType = megengine.Utility.enum(megengine.schema, 'DTypeEnum', dataType).toLowerCase();
        megengine.TensorType._dataTypes = megengine.TensorType._dataTypes || new Map([
            [ 'bool', 'boolean' ],
            [ 'byte', 'uint8' ], [ 'quantizeds4asymm', 'uint8' ], [ 'quantizeds8asymm', 'uint8' ], [ 'uintb4', 'uint8' ],
            [ 'quantizeds1', 'int8' ], [ 'quantizeds4', 'int8' ], [ 'quantizeds8', 'int8' ], [ 'intb1', 'int8' ], [ 'intb2', 'int8' ], [ 'intb4', 'int8' ], [ 'qint8', 'int8' ],
            [ 'quantizeds16', 'int16' ],
            [ 'quantizeds32', 'int32' ]
        ]);
        this.dataType = megengine.TensorType._dataTypes.get(dataType) || dataType;
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

megengine.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions || []);
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) &&
            Array.isArray(this.dimensions) && this.dimensions.length === obj.dimensions.length
            && obj.dimensions.every((value, index) => this.dimensions[index] === value);
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return '[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']';
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
