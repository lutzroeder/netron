
import * as base from './base.js';
import * as flatbuffers from './flatbuffers.js';
import * as protobuf from './protobuf.js';
import * as python from './python.js';

const paddle = {};

paddle.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (identifier === '__model__' || extension === '__model__' || extension === 'paddle' || extension === 'pdmodel') {
            const tags = context.tags('pb');
            if (tags.get(1) === 2) {
                context.type = 'paddle.pb';
                return;
            }
        }
        if (extension === 'pbtxt' || extension === 'txt') {
            const tags = context.tags('pbtxt');
            if (tags.has('blocks')) {
                context.type = 'paddle.pbtxt';
                return;
            }
        }
        const stream = context.stream;
        if (stream && stream.length > 16 && stream.peek(16).every((value) => value === 0x00)) {
            context.type = 'paddle.params';
            return;
        }
        const pickle = paddle.Pickle.open(context);
        if (pickle) {
            context.target = pickle;
            context.type = context.target.name;
            return;
        }
        const entries = paddle.Entries.open(context);
        if (entries) {
            context.target = entries;
            context.type = context.target.name;
            return;
        }
        const naive = paddle.NaiveBuffer.open(context);
        if (naive) {
            context.target = naive;
            context.type = context.target.name;
        }
        const obj = context.peek('json');
        if (obj && obj.base_code && obj.program) {
            context.target = obj;
            context.type = 'paddle.ir';
        }
    }

    filter(context, type) {
        if (context.type === 'paddle.pb' && (type === 'paddle.params' || type === 'paddle.pickle')) {
            return false;
        }
        if (context.type === 'paddle.naive.model' && type === 'paddle.naive.param') {
            return false;
        }
        return true;
    }

    async open(context) {
        const metadata = await context.metadata('paddle-metadata.json');
        switch (context.type) {
            case 'paddle.naive':
            case 'paddle.naive.model':
            case 'paddle.naive.param': {
                paddle.schema = await context.require('./paddle-schema');
                paddle.schema = paddle.schema.paddle.lite.fbs.proto;
                const target = context.target;
                target.read();
                return new paddle.Model(metadata, target.format, target.model, target.weights);
            }
            case 'paddle.ir': {
                throw new paddle.Error('Invalid file content. File contains PaddlePaddle IR data.');
            }
            default: {
                paddle.proto = await context.require('./paddle-proto');
                paddle.proto = paddle.proto.paddle.framework.proto;
                const identifier = context.identifier;
                const parts = identifier.split('.');
                const extension = parts.pop().toLowerCase();
                const base = parts.join('.');
                const openProgram = (context, type) => {
                    const program = {};
                    switch (type) {
                        case 'paddle.pbtxt': {
                            try {
                                const reader = context.read('protobuf.text');
                                program.desc = paddle.proto.ProgramDesc.decodeText(reader);
                            } catch (error) {
                                const message = error && error.message ? error.message : error.toString();
                                throw new paddle.Error(`File text format is not paddle.ProgramDesc (${message.replace(/\.$/, '')}).`);
                            }
                            break;
                        }
                        case 'paddle.pb': {
                            try {
                                const reader = context.read('protobuf.binary');
                                program.desc = paddle.proto.ProgramDesc.decode(reader);
                            } catch (error) {
                                const message = error && error.message ? error.message : error.toString();
                                throw new paddle.Error(`File format is not paddle.ProgramDesc (${message.replace(/\.$/, '')}).`);
                            }
                            break;
                        }
                        default: {
                            throw new paddle.Error(`Unsupported Paddle format '${type}'.`);
                        }
                    }
                    const formatVersion = (version) => {
                        if (version && version.version !== undefined) {
                            const number = version.version.toNumber();
                            if (number > 0) {
                                const list = [Math.floor(number / 1000000) % 1000, Math.floor(number / 1000) % 1000, number % 1000];
                                if (list.slice(-1).pop() === 0) {
                                    list.pop();
                                    if (list.slice(-1).pop() === 0) {
                                        list.pop();
                                    }
                                }
                                return ` v${list.map((item) => item.toString()).join('.')}`;
                            }
                        }
                        return '';
                    };
                    program.format = `PaddlePaddle${formatVersion(program.desc.version)}`;
                    const variables = new Set();
                    for (const block of program.desc.blocks) {
                        const blockVars = new Set();
                        for (const variable of block.vars) {
                            if (variable.persistable && variable.type &&
                                variable.type.type !== paddle.DataType.FETCH_LIST &&
                                variable.type.type !== paddle.DataType.FEED_MINIBATCH) {
                                blockVars.add(variable.name);
                            }
                        }
                        for (const op of block.ops) {
                            for (const input of op.inputs) {
                                for (const argument of input.arguments) {
                                    if (blockVars.has(argument)) {
                                        variables.add(argument);
                                    }
                                }
                            }
                        }
                    }
                    program.vars = Array.from(variables).sort();
                    return program;
                };
                const loadParams = (stream) => {
                    const params = [];
                    while (stream.position < stream.length) {
                        const tensor = paddle.Utility.openTensorDesc(stream);
                        params.push(tensor);
                    }
                    return params;
                };
                const mapParams = (params, program) => {
                    const weights = new Map();
                    const vars = program.vars.slice();
                    for (const param of params) {
                        weights.set(vars.shift(), param);
                    }
                    return weights;
                };
                switch (context.type) {
                    case 'paddle.pickle': {
                        const target = context.target;
                        return new paddle.Model(metadata, target.format, null, target.weights);
                    }
                    case 'paddle.entries': {
                        const target = context.target;
                        target.read();
                        return new paddle.Model(metadata, target.format, null, target.weights);
                    }
                    case 'paddle.params': {
                        const file = identifier === 'params' ? 'model' : `${base}.pdmodel`;
                        const params = loadParams(context.stream);
                        try {
                            const content = await context.fetch(file);
                            const program = openProgram(content, 'paddle.pb');
                            const weights = mapParams(params, program);
                            return new paddle.Model(metadata, program.format, program.desc, weights);
                        } catch {
                            const weights = new Map(params.map((param, index) => [index.toString(), param]));
                            return new paddle.Model(metadata, 'PaddlePaddle Inference Weights', null, weights);
                        }
                    }
                    case 'paddle.pb':
                    case 'paddle.pbtxt': {
                        const loadEntries = async (context, program) => {
                            const promises = program.vars.map((name) => context.fetch(name).then((context) => context.stream).catch(() => null));
                            const streams = await Promise.all(promises);
                            const params = streams.map((stream) => stream ? paddle.Utility.openTensorDesc(stream) : null);
                            const weights = mapParams(params, program);
                            return new paddle.Model(metadata, program.format, program.desc, weights);
                        };
                        const openNumPyArrayPickle = (stream) => {
                            const execution = new python.Execution();
                            const unpickler = execution.invoke('pickle.Unpickler', [stream]);
                            const obj = unpickler.load();
                            const container = new paddle.Pickle(obj);
                            return container.weights || new Map();
                        };
                        const program = openProgram(context, context.type);
                        if (extension === 'pdmodel') {
                            try {
                                const name = `${base}.pdiparams`;
                                const content = await context.fetch(name);
                                const params = loadParams(content.stream);
                                const weights = mapParams(params, program);
                                return new paddle.Model(metadata, program.format, program.desc, weights);
                            } catch {
                                try {
                                    const name = `${base}.pdparams`;
                                    const content = await context.fetch(name);
                                    const weights = openNumPyArrayPickle(content.stream);
                                    try {
                                        const name = `${base}.pdopt`;
                                        const content = await context.fetch(name);
                                        for (const [name, value] of openNumPyArrayPickle(content.stream)) {
                                            if (!weights.has(name)) {
                                                weights.set(name, value);
                                            }
                                        }
                                        return new paddle.Model(metadata, program.format, program.desc, weights);
                                    } catch {
                                        return new paddle.Model(metadata, program.format, program.desc, weights);
                                    }
                                } catch {
                                    try {
                                        const name = `${base}.pdopt`;
                                        const content = await context.fetch(name);
                                        const weights = openNumPyArrayPickle(content.stream);
                                        return new paddle.Model(metadata, program.format, program.desc, weights);
                                    } catch {
                                        return loadEntries(context, program);
                                    }
                                }
                            }
                        }
                        if (identifier === 'model') {
                            try {
                                const content = await context.fetch('params');
                                const params = loadParams(content.stream);
                                const weights = mapParams(params, program);
                                return new paddle.Model(metadata, program.format, program.desc, weights);
                            } catch {
                                return loadEntries(context, program);
                            }
                        }
                        return loadEntries(context, program);
                    }
                    default: {
                        throw new paddle.Error(`Unsupported PaddlePaddle format '${context.type}'.`);
                    }
                }
            }
        }
    }
};

paddle.Model = class {

    constructor(metadata, format, desc, tensors) {
        desc = desc && Array.isArray(desc.blocks) ? desc : { blocks: [null] };
        this.format = format;
        this.graphs = desc.blocks.map((block) => new paddle.Graph(metadata, block, tensors));
    }
};

paddle.Graph = class {

    constructor(metadata, block, tensors) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        if (block) {
            this.name = block.idx.toString();
            const values = new Map();
            for (const variable of block.vars) {
                const type = variable.type && variable.type.type && variable.type.lod_tensor && variable.type.lod_tensor.tensor ? paddle.Utility.createTensorType(variable.type.lod_tensor.tensor.data_type, variable.type.lod_tensor.tensor.dims) : null;
                const tensor = variable.persistable && variable.type && variable.type.type !== paddle.DataType.FETCH_LIST && variable.type.type !== paddle.DataType.FEED_MINIBATCH ? (tensors.get(variable.name) || new paddle.Tensor(type)) : null;
                values.set(variable.name, new paddle.Value(variable.name, type, tensor));
            }
            const scope = {};
            for (let i = 0; i < block.ops.length; i++) {
                for (const input of block.ops[i].inputs) {
                    input.arguments = input.arguments.map((argument) => scope[argument] ? scope[argument] : argument);
                }
                for (const output of block.ops[i].outputs) {
                    output.arguments = output.arguments.map((argument) => {
                        if (scope[argument]) {
                            const next = `${argument}\n${i}`; // custom argument id
                            scope[argument] = next;
                            return next;
                        }
                        scope[argument] = argument;
                        return argument;
                    });
                }
            }
            for (const op of block.ops) {
                for (const input of op.inputs) {
                    for (const name of input.arguments) {
                        if (!values.has(name)) {
                            values.set(name, new paddle.Value(name, null, null));
                        }
                    }
                }
                for (const output of op.outputs) {
                    for (const name of output.arguments) {
                        if (!values.has(name)) {
                            values.set(name, new paddle.Value(name, null, null));
                        }
                    }
                }
            }
            let lastNode = null;
            let lastOutput = null;
            for (const op of block.ops) {
                if (op.type === 'feed') {
                    const name = op.attrs.filter((attr) => attr.name === 'col')[0].i.toString();
                    const argument = new paddle.Argument(name, op.outputs[0].arguments.map((id) => values.get(id)));
                    this.inputs.push(argument);
                } else if (op.type === 'fetch') {
                    const name = op.attrs.filter((attr) => attr.name === 'col')[0].i.toString();
                    const argument = new paddle.Argument(name, op.inputs[0].arguments.map((id) => values.get(id)));
                    this.outputs.push(argument);
                } else {
                    const node = new paddle.Node(metadata, op, values);
                    if (op.inputs.length === 1 && op.inputs[0].arguments.length === 1 &&
                        op.outputs.length >= 1 && op.outputs[0].arguments.length === 1 &&
                        op.inputs[0].arguments[0].split('\n').shift() === op.outputs[0].arguments[0].split('\n').shift() &&
                        lastNode &&
                        lastOutput === op.inputs[0].arguments[0].split('\n').shift()) {
                        lastNode.chain.push(node);
                    } else {
                        this.nodes.push(node);
                        lastNode = null;
                        lastOutput = null;
                        if (op.outputs.length === 1 && op.outputs[0].arguments.length === 1) {
                            lastNode = node;
                            lastOutput = op.outputs[0].arguments[0].split('\n').shift();
                        }
                    }
                }
            }
        } else {
            const values = new Map();
            const ops = new Map();
            for (const [name, tensor] of tensors) {
                values.set(name, new paddle.Value(name, tensor.type, tensor));
                const separator = name.indexOf('.') === -1 ? '_' : '.';
                const regex = /(.*)_((w_attr|scale|weights|offset|b|w|b_attr)_(moment|beta|velocity|mean_square|mean_grad).*)/;
                let parts = [];
                if (separator === '.') {
                    parts = name.split(separator);
                } else if (regex.test(name)) {
                    parts = regex.exec(name).slice(1, 3);
                } else {
                    parts = ['', name];
                }
                const parameter_name = parts.pop();
                const op_name = parts.join(separator);
                if (!ops.has(op_name)) {
                    ops.set(op_name, { name: op_name, type: 'Weights', inputs: [] });
                }
                const op = ops.get(op_name);
                op.inputs.push({ parameter: parameter_name, arguments: [name] });
            }
            for (const op of Array.from(ops.values())) {
                this.nodes.push(new paddle.Node(metadata, op, values));
            }
        }
    }
};

paddle.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        if (type) {
            this.type = type;
        }
        if (visible === false) {
            this.visible = visible;
        }
    }
};

paddle.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new paddle.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

paddle.Node = class {

    constructor(metadata, op, values) {
        const type = op.type;
        this.type = metadata.type(type) || { name: type };
        this.name = op.name || '';
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        this.chain = [];
        if (op.attrs) {
            this.attributes = op.attrs.map((attr) => {
                const name = attr.name;
                const meta = metadata.attribute(this.type.name, name);
                let value = '?';
                let visible = true;
                let type = null;
                switch (attr.type) {
                    case paddle.AttributeType.STRING:
                        type = 'string';
                        value = attr.s;
                        break;
                    case paddle.AttributeType.STRINGS:
                        type = 'string[]';
                        value = Array.from(attr.strings);
                        break;
                    case paddle.AttributeType.BOOLEAN:
                        type = 'boolean';
                        value = attr.b;
                        break;
                    case paddle.AttributeType.BOOLEANS:
                        type = 'boolean[]';
                        value = attr.bools ? Array.from(attr.bools) : attr.bools;
                        break;
                    case paddle.AttributeType.FLOAT:
                        type = 'float32';
                        value = attr.f;
                        break;
                    case paddle.AttributeType.FLOATS:
                        type = 'float32[]';
                        value = attr.floats ? Array.from(attr.floats) : attr.floats;
                        break;
                    case paddle.AttributeType.FLOAT64:
                        type = 'float64';
                        value = attr.float64;
                        break;
                    case paddle.AttributeType.FLOAT64S:
                        type = 'float64[]';
                        value = attr.float64s ? Array.from(attr.float64s) : attr.float64s;
                        break;
                    case paddle.AttributeType.INT:
                        type = 'int32';
                        value = attr.i;
                        break;
                    case paddle.AttributeType.INTS:
                        type = 'int32[]';
                        value = attr.ints ? Array.from(attr.ints) : attr.ints;
                        break;
                    case paddle.AttributeType.LONG:
                        type = 'int64';
                        break;
                    case paddle.AttributeType.LONGS:
                        type = 'int64[]';
                        break;
                    default:
                        break;
                }
                switch (name) {
                    case 'use_mkldnn':
                    case 'use_cudnn':
                    case 'op_callstack':
                    case 'op_role':
                    case 'op_role_var':
                    case 'op_namescope':
                    case 'is_test':
                        visible = false;
                        break;
                    default:
                        break;
                }
                if (meta) {
                    if (meta.default !== undefined) {
                        const defaultValue = meta.default;
                        if (defaultValue === value) {
                            visible = false;
                        } else if (Array.isArray(value) && Array.isArray(defaultValue) && value.length === defaultValue.length) {
                            if (value.every((item, index) => item === defaultValue[index])) {
                                visible = false;
                            }
                        }
                    }
                }
                return new paddle.Argument(name, value, type, visible);
            });
        }
        if (op.inputs) {
            for (const input of op.inputs) {
                if (input.arguments.length > 0) {
                    this.inputs.push(new paddle.Argument(input.parameter, input.arguments.map((name) => values.get(name))));
                }
            }
        }
        if (op.outputs) {
            for (const output of op.outputs) {
                if (output.arguments.length > 0) {
                    this.outputs.push(new paddle.Argument(output.parameter, output.arguments.map((name) => values.get(name))));
                }
            }
        }
        const updates = [
            [this.inputs, 'X'],
            [this.inputs, 'Input'],
            [this.outputs, 'Y'],
            [this.outputs, 'Out']
        ];
        for (const [list, name] of updates) {
            let item = null;
            for (let i = 0; i < list.length; i++) {
                if (list[i].name === name) {
                    item = list[i];
                    list.splice(i, 1);
                    break;
                }
            }
            if (item) {
                list.splice(0, 0, item);
            }
        }
    }
};

paddle.Tensor = class {

    constructor(type, data, category) {
        this.type = type;
        this.values = data;
        this.category = category || '';
    }
};

paddle.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

paddle.TensorShape = class {

    constructor(dimensions) {
        dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
        this.dimensions = dimensions.map((dimension) => {
            return dimension === -1 ? '?' : dimension;
        });
    }

    toString() {
        return (this.dimensions && this.dimensions.length) ? (`[${this.dimensions.join(',')}]`) : '';
    }
};

paddle.Entries = class {

    static open(context) {
        let entries = context.peek('zip');
        entries = entries instanceof Map ? entries : context.peek('tar');
        if (entries instanceof Map) {
            entries = Array.from(entries);
            entries = new Map(entries.filter(([name]) => !name.endsWith('/') && !name.split('/').pop().startsWith('.')).slice());
            if (entries.size > 2 && Array.from(entries).every(([name, value]) => name.split('_').length > 0 && value.peek(16).every((value) => value === 0x00))) {
                return new paddle.Entries(entries);
            }
        }
        return null;
    }

    constructor(data) {
        this.name = 'paddle.entries';
        this.format = 'PaddlePaddle Weights';
        this.data = data;
    }

    read() {
        if (this.data) {
            let rootFolder = null;
            for (const [name] of this.data) {
                if (!name.startsWith('.') || name.startsWith('./')) {
                    const parts = name.split('/');
                    let folder = '';
                    if (parts.length > 2 && parts[0] === '.') {
                        folder = `./${parts[1]}/`;
                    } else if (parts.length > 1) {
                        folder = `${parts[0]}/`;
                    }
                    if (rootFolder !== null && rootFolder !== '' && folder !== rootFolder) {
                        rootFolder = '';
                    } else {
                        rootFolder = folder;
                    }
                }
            }
            this.weights = new Map();
            for (const [name, stream] of this.data) {
                if (name.startsWith(rootFolder)) {
                    const key = name.substring(rootFolder.length);
                    const tensor = paddle.Utility.openTensorDesc(stream);
                    this.weights.set(key, tensor);
                }
            }
            delete this.data;
        }
    }
};

paddle.Pickle = class {

    static open(context) {
        const obj = context.peek('pkl');
        const container = new paddle.Pickle(obj);
        if (container.weights !== null) {
            return container;
        }
        return null;
    }

    constructor(obj) {
        this.name = 'paddle.pickle';
        this.format = 'PaddlePaddle Pickle';
        this._weights = null;
        if (obj && !Array.isArray(obj) && (obj instanceof Map || Object(obj) === obj)) {
            const entries = (obj) => {
                if (obj instanceof Map) {
                    return Array.from(obj);
                } else if (Object(obj) === obj) {
                    return Object.entries(obj);
                }
                return [];
            };
            const filter = (obj) => {
                const list = [];
                if (obj && !Array.isArray(obj)) {
                    for (const [name, value] of entries(obj)) {
                        if (name !== 'StructuredToParameterName@@') {
                            const obj = value && Array.isArray(value) && value.length === 2 && value[0] === name ? value[1] : value;
                            if (obj && !Array.isArray(obj) && obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray') {
                                list.push([name, obj]);
                            }
                        }
                    }
                }
                return list;
            };
            const weights = filter(obj);
            if (weights.length > 0) {
                this._weights = weights;
            } else {
                const list = entries(obj);
                if (list.filter(([name]) => name !== 'StructuredToParameterName@@').length === 1) {
                    const weights = filter(list[0][1]);
                    if (weights.length > 0) {
                        this._weights = weights;
                    }
                }
                if (this._weights === null && list.filter(([name]) => name === 'StructuredToParameterName@@').length > 0) {
                    this._weights = [];
                }
            }
        }
    }

    get weights() {
        if (this._weights && Array.isArray(this._weights)) {
            const weights = new Map();
            for (const [name, value] of this._weights) {
                const type = new paddle.TensorType(value.dtype.__name__, new paddle.TensorShape(value.shape));
                const data = value.data;
                const tensor = new paddle.Tensor(type, data, 'NumPy Array');
                weights.set(name, tensor);
            }
            this._weights = weights;
        }
        return this._weights;
    }
};

paddle.NaiveBuffer = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            if (buffer[0] > 2 || buffer[1] !== 0x00 || buffer[2] !== 0x76 || buffer[2] !== 0x32) {
                if (context.identifier === '__model__.nb') {
                    return new paddle.NaiveBuffer('paddle.naive.model', stream, -1);
                }
                if (context.identifier === 'param.nb') {
                    return new paddle.NaiveBuffer('paddle.naive.param', stream, -1);
                }
            }
            if (buffer[1] === 0x00 && buffer[0] <= 2) {
                return new paddle.NaiveBuffer('paddle.naive', stream, buffer[0]);
            }
        }
        return null;
    }

    constructor(name, stream, meta_version) {
        this.name = name;
        this.stream = stream;
        this.meta_version = meta_version;
    }

    read() {
        const reader = base.BinaryReader.open(this.stream);
        if (this.meta_version >= 2) {
            reader.skip(2);
        }
        const decoder = new TextDecoder();
        const opt_version = reader.read(16);
        const version = decoder.decode(opt_version.slice(0, opt_version.indexOf(0x00)));
        this.format = `Paddle Lite${version && version.match(/^v\d+\.\d+.\d+$/) ? ` ${version}` : ''}`;
        const topo_size = reader.uint64().toNumber();
        const openProgramDesc = (buffer) => {
            const reader = flatbuffers.BinaryReader.open(buffer);
            return paddle.schema.ProgramDesc.create(reader);
        };
        const openParamDesc = (buffer) => {
            const reader = flatbuffers.BinaryReader.open(buffer);
            return paddle.schema.ParamDesc.create(reader);
        };
        switch (this.meta_version) {
            case -1: {
                throw new paddle.Error('Paddle Lite naive buffer format is deprecated.');
            }
            case 0:
            case 1: {
                throw new paddle.Error(`Paddle Lite meta format '${this.meta_version}' is deprecated.`);
            }
            case 2: {
                const topo_data = new Uint8Array(topo_size);
                topo_data.set(reader.read(topo_size), 0);
                this.model = openProgramDesc(topo_data);
                reader.uint16(); // version
                reader.uint16(); // meta_size
                const header_size = reader.uint16();
                const params_size = reader.uint16();
                reader.uint32(); // max_tensor_size
                reader.skip(header_size - 6);
                this.weights = new Map();
                for (let i = 0; i < params_size; i++) {
                    const total_size = reader.uint32();
                    const offset = reader.uint32();
                    const param_bytes = total_size - offset;
                    const param_data = reader.read(param_bytes);
                    const desc = openParamDesc(param_data);
                    const data = desc.variable.data;
                    const data_type = desc.variable.data_type;
                    const dim = desc.variable.dim;
                    const type = paddle.Utility.createTensorType(data_type, dim);
                    const tensor = new paddle.Tensor(type, data);
                    this.weights.set(desc.name, tensor);
                }
                break;
            }
            default: {
                throw new paddle.Error(`Unsupported Paddle Lite naive buffer meta format '${this.meta_version}'.`);
            }
        }
        delete this.stream;
    }
};

paddle.Utility = class {

    static createTensorType(data_type, shape) {
        if (!paddle.Utility._dataTypes) {
            const length = Math.max.apply(null, Object.entries(paddle.DataType).map(([, value]) => value));
            paddle.Utility._dataTypes = new Array(length);
            const types = new Map([
                ['bool', 'boolean'],
                ['bf16', 'bfloat16'],
                ['fp16', 'float16'],
                ['fp32', 'float32'],
                ['fp64', 'float64'],
                ['fp8_e4m3fn', 'float8e4m3fn'],
                ['fp8_e5m2', 'float8e5m2']
            ]);
            for (const [name, index] of Object.entries(paddle.DataType)) {
                const key = name.toLowerCase();
                paddle.Utility._dataTypes[index] = types.has(key) ? types.get(key) : key;
            }
        }
        const dataType = data_type < paddle.Utility._dataTypes.length ? paddle.Utility._dataTypes[data_type] : '?';
        return new paddle.TensorType(dataType, new paddle.TensorShape(shape));
    }

    static openTensorDesc(stream) {
        const signature = stream.read(16);
        if (!signature.every((value) => value === 0x00)) {
            throw new paddle.Error('Invalid paddle.TensorDesc signature.');
        }
        const length = base.BinaryReader.open(stream.read(4)).uint32();
        const buffer = stream.read(length);
        const reader = protobuf.BinaryReader.open(buffer);
        const tensorDesc = paddle.proto.VarType.TensorDesc.decode(reader);
        const dims = tensorDesc.dims.map((dim) => dim.toNumber());
        const size = dims.reduce((a, b) => a * b, 1);
        let itemsize = 0;
        switch (tensorDesc.data_type) {
            case paddle.DataType.FP16: itemsize = 2; break;
            case paddle.DataType.FP32: itemsize = 4; break;
            case paddle.DataType.FP64: itemsize = 8; break;
            case paddle.DataType.INT8: itemsize = 1; break;
            case paddle.DataType.INT16: itemsize = 2; break;
            case paddle.DataType.INT32: itemsize = 4; break;
            case paddle.DataType.INT64: itemsize = 8; break;
            case paddle.DataType.UINT8: itemsize = 1; break;
            default: throw new paddle.Error(`Invalid inference params data type '${tensorDesc.data_type}'.`);
        }
        const type = paddle.Utility.createTensorType(tensorDesc.data_type, tensorDesc.dims);
        const data = stream.read(itemsize * size);
        return new paddle.Tensor(type, data);
    }
};

paddle.DataType = {
    BOOL: 0,
    INT16: 1,
    INT32: 2,
    INT64: 3,
    FP16: 4,
    FP32: 5,
    FP64: 6,
    LOD_TENSOR: 7,
    SELECTED_ROWS: 8,
    FEED_MINIBATCH: 9,
    FETCH_LIST: 10,
    STEP_SCOPES: 11,
    LOD_RANK_TABLE: 12,
    LOD_TENSOR_ARRAY: 13,
    PLACE_LIST: 14,
    READER: 15,
    RAW: 17,
    TUPLE: 18,
    SIZE_T: 19,
    UINT8: 20,
    INT8: 21,
    BF16: 22,
    COMPLEX64: 23,
    COMPLEX128: 24,
    STRING: 25,
    STRINGS: 26,
    FP8_E4M3FN: 32,
    FP8_E5M2: 33,
};

paddle.AttributeType = {
    INT: 0,
    FLOAT: 1,
    STRING: 2,
    INTS: 3,
    FLOATS: 4,
    STRINGS: 5,
    BOOLEAN: 6,
    BOOLEANS: 7,
    BLOCK: 8,
    LONG: 9,
    BLOCKS: 10,
    LONGS: 11,
    FLOAT64S: 12,
    VAR: 13,
    VARS: 14,
    FLOAT64: 15
};

paddle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading PaddlePaddle model.';
    }
};

export const ModelFactory = paddle.ModelFactory;
