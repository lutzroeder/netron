
// Experimental

import * as flatbuffers from './flatbuffers.js';

const megengine = {};

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
                    const reader = flatbuffers.BinaryReader.open(stream, position + 4);
                    if (reader.identifier === 'mgv2') {
                        context.type = 'megengine.mge';
                        context.target = reader;
                        return;
                    }
                }
            }
            for (const value of ['mgb0001', 'mgb0000a', 'MGBS', 'MGBC']) {
                if (tag.startsWith(value)) {
                    context.type = `megengine.${value}`;
                    return;
                }
            }
        }
        const obj = context.peek('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ === 'megengine.traced_module.traced_module' && obj.__class__.__name__ === 'TracedModule') {
            context.type = 'megengine.tm';
        }
    }

    async open(context) {
        const metadata = await context.metadata('megengine-metadata.json');
        switch (context.type) {
            case 'megengine.tm': {
                const obj = context.peek('pkl');
                return new megengine.Model(metadata, obj, context.type);
            }
            case 'megengine.mge': {
                megengine.schema = await context.require('./megengine-schema');
                megengine.schema = megengine.schema.mgb.serialization.fbs;
                let model = null;
                try {
                    const reader = context.target;
                    model = megengine.schema.v2.Model.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new megengine.Error(`File format is not megengine.Model (${message.replace(/\.$/, '')}).`);
                }
                return new megengine.Model(metadata, model, context.type);
            }
            default: {
                throw new megengine.Error(`Unsupported MegEngine format '${context.type.replace(/^megengine\./, '')}'.`);
            }
        }
    }
};

megengine.Model = class {

    constructor(metadata, obj, type) {
        this.format = 'MegEngine';
        if (type === 'megengine.tm') {
            this.format += obj.dump_info && obj.dump_info.has('version') ? ` v${obj.dump_info.get('version')}` : '';
        } else if (type === 'megengine.mge') {
            this.format += ` Mge${obj.model_version ? ` v${obj.model_version}` : ''}`;
        }
        this.graphs = [new megengine.Graph(metadata, obj)];
    }
};

megengine.Graph = class {

    constructor(metadata, obj) {
        this.name = '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, tensor, quantization) => {
            type = type || null;
            tensor = tensor || null;
            if (tensor) {
                return new megengine.Value(name, type, tensor, quantization);
            }
            if (!values.has(name)) {
                values.set(name, new megengine.Value(name, type, tensor, quantization));
            } else if ((type && !type.equals(values.get(name).type)) || tensor) {
                throw new megengine.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const loadGraph = (tmodule, igraph, context, namePrefix, metadata, isRoot) => {
            const expressions = igraph._exprs;
            const getTensorType = (dtype, shape) => {
                dtype = dtype ? dtype.__name__ : null;
                return new megengine.TensorType(dtype, new megengine.TensorShape(shape));
            };
            if (isRoot) {
                for (const node of igraph._inputs) {
                    if (node.__class__.__name__ !== 'ModuleNode') {
                        const type = getTensorType(node._dtype, node._shape);
                        const value = values.map(node._name, type, null);
                        const argument = new megengine.Argument(node._name, [value]);
                        this.inputs.push(argument);
                    }
                }
                for (const node of igraph._outputs) {
                    const type = getTensorType(node._dtype, node._shape);
                    const value = values.map(node._name, type, null);
                    const argument = new megengine.Argument(node._name, [value]);
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
                const processArgs = (inp, startIdx) => {
                    while (typeof inp === 'string' && inp.indexOf('Tensor') !== -1) {
                        inp = inp.replace('Tensor', `input${startIdx === 0 ? '' : startIdx}`);
                        startIdx += 1;
                    }
                    return [inp, startIdx];
                };
                const formatTreeDef = (obj) => {
                    if (obj.__class__.__name__ !== 'TreeDef' && obj.__class__.__name__ !== 'LeafDef') {
                        throw new megengine.Error(`Invalid argument '${obj.__class__.__name__}'.`);
                    }
                    if (obj.__class__.__name__ === 'TreeDef') {
                        const type = typeof obj.type === 'string' ? obj.type.split('.').slice(-1)[0] : obj.type.__name__;
                        const list = obj.children_defs.map((child) => formatTreeDef(child));
                        switch (type) {
                            case 'tuple': {
                                return `(${list.join(',')})`;
                            }
                            case 'slice': {
                                return list.join(':');
                            }
                            case 'list': {
                                return `[${list.join(',')}]`;
                            }
                            case 'dict': {
                                let content = '';
                                for (let i = 0; i < this.children_defs.length; i++) {
                                    content += `${this.aux_data[i]}:${list[i]}`;
                                }
                                return `{${content}}`;
                            }
                            default: {
                                return `${type}(${list.join(',')})`;
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
                    let name = '';
                    if (meta.attributes === undefined || (meta.attributes.length !== args.children_defs.length && meta.varargs === null)) {
                        name = `arg${argIdx}`;
                    } else if (argIdx < meta.attributes.length) {
                        name = meta.attributes[argIdx].name;
                    } else {
                        name = meta.varargs + (argIdx - meta.attributes.length);
                    }
                    const [value, index] = processArgs(formatTreeDef(arg), inpIdx);
                    state[name] = value;
                    inpIdx = index;
                    argIdx += 1;
                }
                for (let i = 0; i < kwargs.children_defs.length; i++) {
                    const [value, index] = processArgs(formatTreeDef(kwargs.children_defs[i]), inpIdx);
                    state[kwargs.aux_data[i]] = value;
                    inpIdx = index;
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
                return prefix === '' ? name : `${prefix}_${name}`;
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
                        const obj = { 'name': '' };
                        let state = {};
                        if (expression.method === '__call__') {
                            const module = parseGetAttr(tmodule, expression.inputs[0].expr);
                            const getModuleType = (obj) => {
                                if (obj.module !== undefined) {
                                    return `${obj.module[0]}.${obj.module[1]}`;
                                }
                                return `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                            };
                            const moduleType = module.__class__.__name__ === 'TracedModule' ? 'TracedModule' : getModuleType(module);
                            if (moduleType === 'TracedModule') {
                                const moduleName = expression.outputs[0]._name.endsWith("_out") ? expression.outputs[0]._name.substring(0, expression.outputs[0]._name.length - 4) : expression.outputs[0]._name;
                                const prefix = getFullName(namePrefix, moduleName);
                                const internalGraph = module.argdef_graph_map.get(expression.arg_def);
                                for (let i = 0; i < expression.inputs.length; i++) {
                                    const actualName = getFullName(namePrefix, expression.inputs[i]._name);
                                    const internalName = getFullName(prefix, internalGraph._inputs[i]._name);
                                    context.set(internalName, actualName);
                                }
                                for (let i = 0; i < expression.outputs.length; i++) {
                                    const actualName = getFullName(namePrefix, expression.outputs[i]._name);
                                    const internalName = getFullName(prefix, internalGraph._outputs[i]._name);
                                    if (context.get(internalName) === undefined) {
                                        context.set(internalName, actualName);
                                    } else {
                                        context.set(actualName, context.get(internalName));
                                    }
                                }
                                loadGraph(module, internalGraph, context, prefix, metadata, false);
                                continue;
                            }
                            obj.type = moduleType;
                            state = module.__class__.__name__ === 'TracedModule' ? module : module.state;
                            if (state === undefined) {
                                state = module;
                            }
                        } else {
                            obj.type = expression.method;
                            const [args, kwargs] = expression.arg_def.children_defs;
                            const schema = metadata.type(expression.method);
                            state = parseArgs(args, kwargs, schema);
                        }
                        const node = new megengine.Node(metadata, obj, values, null, expression, state);
                        this.nodes.push(node);
                        break;
                    }
                    case 'CallFunction': {
                        const getFunctionType = (obj) => {
                            if (obj.func.__module__ !== undefined) {
                                return `${obj.func.__module__}.${obj.func.__name__}`;
                            }
                            return `${obj.func[0]}.${obj.func[1]}`;
                        };
                        const func = getFunctionType(expression);
                        const item = { 'name': '', 'type': func };
                        const [args, kwargs] = expression.arg_def.children_defs;
                        const schema = metadata.type(func);
                        const state = parseArgs(args, kwargs, schema);
                        const node = new megengine.Node(metadata, item, values, null, expression, state);
                        this.nodes.push(node);
                        break;
                    }
                    case 'Apply': {
                        const opdef = expression.opdef_state ? expression.opdef_state.get('opdef_type') : expression.opdef.type;
                        const item = { 'name': '', 'type': `${opdef.__module__}.${opdef.__name__}` };
                        const node = new megengine.Node(metadata, item, values, null, expression, expression.opdef_state);
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
            const [graph] = Array.from(obj.argdef_graph_map.values());
            loadGraph(obj, graph, new Map(), '', metadata, true);
            return;
        }
        const extraInfoNameset = new Set();
        const getExtraInfo = (opr) => {
            let name = opr.name;
            let repeatIdx = 0;
            while (extraInfoNameset.has(name)) {
                for (const id of opr.inputs) {
                    name = `${name}[${id}]`;
                }
                name += repeatIdx;
                repeatIdx += 1;
            }
            extraInfoNameset.add(name);
            const type = opr.type.replace(/V(\d+)$/, '');
            const args = [];
            if (opr.tensors.length > 0) {
                const [tensor] = opr.tensors;
                const type = new megengine.TensorType(tensor.dtype.type, new megengine.TensorShape(tensor.shape));
                const data = tensor.data.byteLength === 0 ? undefined : tensor.data.slice(0);
                const initializer = opr.type === 'Host2DeviceCopy' ? undefined : new megengine.Tensor('', type, data);
                const quantization = tensor.dtype.param ? { scale: tensor.dtype.param.scale, zeroPoint: tensor.dtype.param.zero_point } : null;
                const value = values.map(name, type, initializer, quantization);
                args.push(value);
            } else {
                const type = opr.shape ? new megengine.TensorType('?', new megengine.TensorShape(opr.shape)) : null;
                const value = values.map(name, type);
                args.push(value);
            }
            return { name, type, args };
        };
        const getAllOprAndTensor = (oprs) => {
            const allOprAndTensor = new Map();
            for (const opr of oprs) {
                if (opr.type === 'MultipleDeviceTensorWithFormatHolder' || opr.outputs.length > 1) {
                    if (opr.type === 'MultipleDeviceTensorWithFormatHolder' || opr.type === 'MultipleDeviceTensorHolder') {
                        opr.type = 'ImmutableTensor';
                    }
                    for (let id = 0; id < opr.outputs.length; id++) {
                        const keyId = opr.outputs[id];
                        const name = obj.middle_tensors[keyId] ? obj.middle_tensors[keyId].name : String(keyId);
                        const type = opr.type;
                        const tensors = opr.tensors.length ? [opr.tensors[id]] : [];
                        const onlyShape = obj.middle_tensors[keyId] ? obj.middle_tensors[keyId].shape : [];
                        allOprAndTensor.set(keyId, { name, type, tensors, shape: onlyShape, inputs: opr.inputs, outputs: opr.outputs });
                        const _opr = allOprAndTensor.get(keyId);
                        _opr.extraInfo = getExtraInfo(_opr);
                    }
                } else {
                    const [keyId] = opr.outputs;
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
        for (const op of Array.from(allOprAndTensor.values())) {
            if (op.type === 'Host2DeviceCopy') {
                const argument = new megengine.Argument('input', op.extraInfo.args);
                this.inputs.push(argument);
            } else if (op.type !== 'ImmutableTensor') {
                this.nodes.push(new megengine.Node(metadata, op, values, allOprAndTensor));
            }
        }
        for (let i = 0; i < obj.output_vars_idx.length; i++) {
            const id = obj.output_vars_idx[i].compact_id;
            const out_type = `output${i === 0 ? '' : i}`;
            const argument = new megengine.Argument(out_type, allOprAndTensor.get(id).extraInfo.args);
            this.outputs.push(argument);
        }
    }
};

megengine.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

megengine.Value = class {

    constructor(name, type, initializer, quantization) {
        if (typeof name !== 'string') {
            throw new megengine.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer;
        if (quantization && ((quantization.scale !== undefined && quantization.scale !== 1) || (quantization.zeroPoint !== undefined && quantization.zeroPoint !== 0))) {
            this.quantization = {
                type: 'linear',
                scale: [quantization.scale],
                offset: [quantization.zeroPoint]
            };
        }
    }
};

megengine.Node = class {

    constructor(metadata, obj, values, allOprAndTensor, expr, state) {
        this.name = '';
        const type = metadata.type(obj.type);
        this.type = type ? { ...type } : { name: obj.type };
        this.type.name = this.type.name.replace(/V(\d+)$/, '');
        if (this.type.name.length > 4 && this.type.name.startsWith('__') && this.type.name.endsWith('__')) {
            this.type.name = this.type.name.substring(2, this.type.name.length - 2);
        }
        this.inputs = [];
        this.outputs = [];
        this.chain = [];
        this.attributes = [];
        const attributes = [];
        if (obj.inputs && obj.outputs) {
            const inputSchemas = this.type && this.type.inputs ? [...this.type.inputs] : [];
            for (let i = 0; i < obj.inputs.length; i++) {
                const inputOpr = allOprAndTensor.get(obj.inputs[i]);
                const schema = inputSchemas.length > 0 ? inputSchemas.shift() : { name: `input${i === 0 ? '' : i.toString()}` };
                const argument = new megengine.Argument(schema.name, inputOpr.extraInfo.args);
                this.inputs.push(argument);
            }
            const outputSchemas = this.type && this.type.outputs ? [...this.type.outputs] : [];
            for (let i = 0; i < obj.outputs.length; i++) {
                const outputOpr = allOprAndTensor.get(obj.outputs[i]);
                const schema = outputSchemas.length > 0 ? outputSchemas.shift() : { name: `output${i === 0 ? '' : i.toString()}` };
                const argument = new megengine.Argument(schema.name, outputOpr.extraInfo.args);
                this.outputs.push(argument);
            }
            if (obj.param) {
                for (const [name, value] of Object.entries(obj.param)) {
                    if (value !== null) {
                        const schema = metadata.attribute(obj.param.constructor.name, name);
                        attributes.push([schema, name, value]);
                    }
                }
            }
        }
        if (expr) {
            let inpIdx = 0;
            for (const i of expr.inputs) {
                if (i.__class__.__name__ !== 'ModuleNode') {
                    const initializer = i.initializer === undefined ? null : i.initializer;
                    const name = `input${inpIdx === 0 ? '' : inpIdx}`;
                    const dtype = i._dtype ? i._dtype.__name__ : null;
                    const shape = new megengine.TensorShape(i._shape);
                    const type = new megengine.TensorType(dtype, shape);
                    const value = values.map(i._fullname, type, initializer);
                    const argument = new megengine.Argument(name, [value]);
                    this.inputs.push(argument);
                    inpIdx += 1;
                }
            }
            let outIdx = 0;
            let qparams = null;
            for (const o of expr.outputs) {
                if (o._qparams !== null) {
                    /* eslint-disable prefer-destructuring */
                    qparams = o._qparams[1];
                    /* eslint-enable prefer-destructuring */
                }
                const name = `output${outIdx === 0 ? '' : outIdx}`;
                const dtype = o._dtype ? o._dtype.__name__ : null;
                const shape = new megengine.TensorShape(o._shape);
                const type = new megengine.TensorType(dtype, shape);
                const value = values.map(o._fullname, type, null);
                const argument = new megengine.Argument(name, [value]);
                this.outputs.push(argument);
                outIdx += 1;
            }
            if (qparams !== null) {
                state = state === null ? {} : state;
                state.scale = qparams.scale;
                state.zero_point = qparams.zero_point;
                state.quant_dtype_meta = qparams.dtype_meta;
            }
            if (state !== null) {
                for (const [key, obj] of Array.from(state)) {
                    const isModule = (obj) => {
                        return obj && (obj.state || obj._forward_pre_hooks);
                    };
                    const isTensor = (obj) => {
                        return obj && obj.__class__ && obj.__class__.__module__ === 'megengine.tensor' && (obj.__class__.__name__ === 'Tensor' || obj.__class__.__name__ === 'Parameter');
                    };
                    if (!key.startsWith('_') && !isModule(obj)) {
                        if (isTensor(obj)) {
                            const tensor = obj;
                            const dtype = tensor.dtype ? tensor.dtype.__name__ : null;
                            const shape = new megengine.TensorShape(tensor.data.shape);
                            const type = new megengine.TensorType(dtype, shape);
                            const data = tensor.data.data;
                            const initializer = new megengine.Tensor(key, type, data);
                            const value = values.map('', type, initializer);
                            const argument = new megengine.Argument(key, [value]);
                            this.inputs.push(argument);
                        } else {
                            const value = obj === null ? 'None' : obj;
                            attributes.push([null, key, value]);
                        }
                    }
                }
            }
        }
        this.attributes = attributes.map(([metadata, name, value]) => {
            value = ArrayBuffer.isView(value) ? Array.from(value) : value;
            let type = metadata ? metadata.type : null;
            let visible = true;
            if (name === 'training') {
                visible = false;
                type = 'boolean';
            }
            if (megengine.schema) {
                if (megengine.schema.param[type]) {
                    value = megengine.Utility.enum(megengine.schema.param, type, value);
                } else if (megengine.schema[type]) {
                    value = megengine.Utility.enum(megengine.schema, type, value);
                } else if (megengine.schema.v2[type]) {
                    value = megengine.Utility.enum(megengine.schema.v2, type, value);
                }
            }
            return new megengine.Argument(name, value, type, visible);
        });
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
        dataType = megengine.Utility.enum(megengine.schema, 'DTypeEnum', dataType);
        dataType = typeof dataType === 'string' ? dataType.toLowerCase() : dataType;
        megengine.TensorType._dataTypes = megengine.TensorType._dataTypes || new Map([
            ['bool', 'boolean'],
            ['byte', 'uint8'], ['quantizeds4asymm', 'uint8'], ['quantizeds8asymm', 'uint8'], ['uintb4', 'uint8'],
            ['quantizeds1', 'int8'], ['quantizeds4', 'int8'], ['quantizeds8', 'int8'], ['intb1', 'int8'], ['intb2', 'int8'], ['intb4', 'int8'], ['qint8', 'int8'],
            ['quantizeds16', 'int16'],
            ['quantizeds32', 'int32']
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
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
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
                const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
                megengine.Utility._enums.set(name, entries);
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

export const ModelFactory = megengine.ModelFactory;

