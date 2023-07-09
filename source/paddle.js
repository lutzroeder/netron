
var paddle = {};
var flatbuffers = require('./flatbuffers');
var protobuf = require('./protobuf');
var python = require('./python');
var base = require('./base');

paddle.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (identifier === '__model__' || extension === '__model__' || extension === 'paddle' || extension === 'pdmodel') {
            const tags = context.tags('pb');
            if (tags.get(1) === 2) {
                return 'paddle.pb';
            }
        }
        if (extension === 'pbtxt' || extension === 'txt') {
            const tags = context.tags('pbtxt');
            if (tags.has('blocks')) {
                return 'paddle.pbtxt';
            }
        }
        const stream = context.stream;
        if (stream && stream.length > 16 && stream.peek(16).every((value) => value === 0x00)) {
            return 'paddle.params';
        }
        if (paddle.Pickle.open(context)) {
            return 'paddle.pickle';
        }
        if (paddle.Entries.open(context)) {
            return 'paddle.entries';
        }
        if (paddle.NaiveBuffer.open(context)) {
            return 'paddle.naive';
        }
        return undefined;
    }

    async open(context, target) {
        const metadata = await context.metadata('paddle-metadata.json');
        switch (target) {
            case 'paddle.naive': {
                await context.require('./paddle-schema');
                paddle.schema = flatbuffers.get('paddlelite').paddle.lite.fbs.proto;
                const file = paddle.NaiveBuffer.open(context);
                return new paddle.Model(metadata, file.format, file.model, file.weights);
            }
            default: {
                await context.require('./paddle-proto');
                paddle.proto = protobuf.get('paddle').paddle.framework.proto;
                const identifier = context.identifier;
                const parts = identifier.split('.');
                const extension = parts.pop().toLowerCase();
                const base = parts.join('.');
                const openProgram = (stream, target) => {
                    const program = {};
                    switch (target) {
                        case 'paddle.pbtxt': {
                            try {
                                const reader = protobuf.TextReader.open(stream);
                                program.desc = paddle.proto.ProgramDesc.decodeText(reader);
                            } catch (error) {
                                const message = error && error.message ? error.message : error.toString();
                                throw new paddle.Error('File text format is not paddle.ProgramDesc (' + message.replace(/\.$/, '') + ').');
                            }
                            break;
                        }
                        case 'paddle.pb': {
                            try {
                                const reader = protobuf.BinaryReader.open(stream);
                                program.desc = paddle.proto.ProgramDesc.decode(reader);
                            } catch (error) {
                                const message = error && error.message ? error.message : error.toString();
                                throw new paddle.Error('File format is not paddle.ProgramDesc (' + message.replace(/\.$/, '') + ').');
                            }
                            break;
                        }
                        default: {
                            throw new paddle.Error("Unsupported Paddle format '" + target + "'.");
                        }
                    }
                    const formatVersion = (version) => {
                        if (version && version.version && version.version.toNumber) {
                            const number = version.version.toNumber();
                            if (number > 0) {
                                const list = [ Math.floor(number / 1000000) % 1000, Math.floor(number / 1000) % 1000, number % 1000 ];
                                if (list.slice(-1).pop() === 0) {
                                    list.pop();
                                    if (list.slice(-1).pop() === 0) {
                                        list.pop();
                                    }
                                }
                                return ' v' + list.map((item) => item.toString()).join('.');
                            }
                        }
                        return '';
                    };
                    program.format = 'PaddlePaddle' + formatVersion(program.desc.version);
                    const variables = new Set();
                    for (const block of program.desc.blocks) {
                        const blockVars = new Set();
                        for (const variable of block.vars) {
                            if (variable.persistable && variable.type &&
                                variable.type.type != paddle.DataType.FETCH_LIST &&
                                variable.type.type != paddle.DataType.FEED_MINIBATCH) {
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
                const createModel = (metadata, format, desc, tensors) => {
                    return new paddle.Model(metadata, format, desc, tensors);
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
                switch (target) {
                    case 'paddle.pickle': {
                        const container = paddle.Pickle.open(context);
                        return createModel(metadata, container.format, null, container.weights);
                    }
                    case 'paddle.entries': {
                        const container = paddle.Entries.open(context);
                        return createModel(metadata, container.format, null, container.weights);
                    }
                    case 'paddle.params': {
                        const file = identifier !== 'params' ? base + '.pdmodel' : 'model';
                        const params = loadParams(context.stream);
                        try {
                            const stream = await context.request(file, null);
                            const program = openProgram(stream, 'paddle.pb');
                            const weights = mapParams(params, program);
                            return createModel(metadata, program.format, program.desc, weights);
                        } catch (error) {
                            const weights = new Map(params.map((param, index) => [ index.toString(), param ]));
                            return createModel(metadata, 'PaddlePaddle Inference Weights', null, weights);
                        }
                    }
                    case 'paddle.pb':
                    case 'paddle.pbtxt': {
                        const loadEntries = async (context, program) => {
                            const promises = program.vars.map((name) => context.request(name, null).then((stream) => stream).catch(() => null));
                            const streams = await Promise.all(promises);
                            const params = streams.map((stream) => stream ? paddle.Utility.openTensorDesc(stream) : null);
                            const weights = mapParams(params, program);
                            return createModel(metadata, program.format, program.desc, weights);
                        };
                        const openNumPyArrayPickle = (stream) => {
                            const execution = new python.Execution();
                            const unpickler = execution.invoke('pickle.Unpickler', [ stream ]);
                            const obj = unpickler.load();
                            const container = new paddle.Pickle(obj);
                            return container.weights || new Map();
                        };
                        const program = openProgram(context.stream, target);
                        if (extension === 'pdmodel') {
                            try {
                                const stream = await context.request(base + '.pdiparams', null);
                                const params = loadParams(stream);
                                const weights = mapParams(params, program);
                                return createModel(metadata, program.format, program.desc, weights);
                            } catch (error) {
                                try {
                                    const stream = await context.request(base + '.pdparams', null);
                                    const weights = openNumPyArrayPickle(stream);
                                    try {
                                        const stream = await context.request(base + '.pdopt', null);
                                        for (const entry of openNumPyArrayPickle(stream)) {
                                            if (!weights.has(entry[0])) {
                                                weights.set(entry[0], entry[1]);
                                            }
                                        }
                                        return createModel(metadata, program.format, program.desc, weights);
                                    } catch (error) {
                                        return createModel(metadata, program.format, program.desc, weights);
                                    }
                                } catch (error) {
                                    try {
                                        const stream = await context.request(base + '.pdopt', null);
                                        const weights = openNumPyArrayPickle(stream);
                                        return createModel(metadata, program.format, program.desc, weights);
                                    } catch (error) {
                                        return loadEntries(context, program);
                                    }
                                }
                            }
                        }
                        if (identifier === 'model') {
                            try {
                                const stream = await context.request('params', null);
                                const params = loadParams(stream);
                                const weights = mapParams(params, program);
                                return createModel(metadata, program.format, program.desc, weights);
                            } catch (error) {
                                return loadEntries(context, program);
                            }
                        }
                        return loadEntries(context, program);
                    }
                    default: {
                        throw new paddle.Error("Unsupported PaddlePaddle format '" + target + "'.");
                    }
                }
            }
        }
    }
};

paddle.Model = class {

    constructor(metadata, format, programDesc, tensors) {
        this._format = format;
        this._graphs = programDesc ?
            programDesc.blocks.map((block) => new paddle.Graph(metadata, block, tensors)) :
            [ new paddle.Graph(metadata, null, tensors) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

paddle.Graph = class {

    constructor(metadata, block, tensors) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        if (block) {
            this._name = block.idx.toString();

            const args = new Map();
            for (const variable of block.vars) {
                const type = variable.type && variable.type.type && variable.type.lod_tensor && variable.type.lod_tensor.tensor ? paddle.Utility.createTensorType(variable.type.lod_tensor.tensor.data_type, variable.type.lod_tensor.tensor.dims) : null;
                const tensor = variable.persistable && variable.type && variable.type.type != paddle.DataType.FETCH_LIST && variable.type.type != paddle.DataType.FEED_MINIBATCH ? (tensors.get(variable.name) || new paddle.Tensor(type)) : null;
                args.set(variable.name, new paddle.Value(variable.name, type, tensor));
            }

            const scope = {};
            for (let i = 0; i < block.ops.length; i++) {
                for (const input of block.ops[i].inputs) {
                    input.arguments = input.arguments.map((argument) => scope[argument] ? scope[argument] : argument);
                }
                for (const output of block.ops[i].outputs) {
                    output.arguments = output.arguments.map((argument) => {
                        if (scope[argument]) {
                            const next = argument + '\n' + i.toString(); // custom argument id
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
                    for (const argument of input.arguments) {
                        const name = argument;
                        if (!args.has(name)) {
                            args.set(name, new paddle.Value(name, null, null));
                        }
                    }
                }
                for (const output of op.outputs) {
                    for (const argument of output.arguments) {
                        const name = argument;
                        if (!args.has(name)) {
                            args.set(name, new paddle.Value(name, null, null));
                        }
                    }
                }
            }

            let lastNode = null;
            let lastOutput = null;
            for (const op of block.ops) {
                if (op.type == 'feed') {
                    const inputName = op.attrs.filter((attr) => attr.name == 'col')[0].i.toString();
                    this._inputs.push(new paddle.Argument(inputName, op.outputs[0].arguments.map((id) => args.get(id))));
                } else if (op.type == 'fetch') {
                    const outputName = op.attrs.filter((attr) => attr.name == 'col')[0].i.toString();
                    this._outputs.push(new paddle.Argument(outputName, op.inputs[0].arguments.map((id) => args.get(id))));
                } else {
                    const node = new paddle.Node(metadata, op, args);
                    if (op.inputs.length == 1 && op.inputs[0].arguments.length == 1 &&
                        op.outputs.length >= 1 && op.outputs[0].arguments.length == 1 &&
                        op.inputs[0].arguments[0].split('\n').shift() == op.outputs[0].arguments[0].split('\n').shift() &&
                        lastNode &&
                        lastOutput == op.inputs[0].arguments[0].split('\n').shift()) {
                        lastNode.chain.push(node);
                    } else {
                        this._nodes.push(node);
                        lastNode = null;
                        lastOutput = null;
                        if (op.outputs.length == 1 && op.outputs[0].arguments.length == 1) {
                            lastNode = node;
                            lastOutput = op.outputs[0].arguments[0].split('\n').shift();
                        }
                    }
                }
            }
        } else {
            const args = new Map();
            const ops = new Map();
            for (const pair of tensors) {
                const name = pair[0];
                const tensor = pair[1];
                args.set(name, new paddle.Value(name, tensor.type, tensor));
                const separator = name.indexOf('.') !== -1 ? '.' : '_';
                const regex = /(.*)_((w_attr|scale|weights|offset|b|w|b_attr)_(moment|beta|velocity|mean_square|mean_grad).*)/;
                const parts = separator === '.' ? name.split(separator) : (regex.test(name) ? regex.exec(name).slice(1, 3) : [ '', name ]);
                const parameter_name = parts.pop();
                const op_name = parts.join(separator);
                if (!ops.has(op_name)) {
                    ops.set(op_name, { name: op_name, type: 'Weights', inputs: [] });
                }
                const op = ops.get(op_name);
                op.inputs.push({ parameter: parameter_name, arguments: [ name ] });
            }
            for (const pair of ops) {
                const op = pair[1];
                this._nodes.push(new paddle.Node(metadata, op, args));
            }
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

paddle.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

paddle.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new paddle.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._type) {
            return this._type;
        }
        if (this._initializer) {
            return this._initializer.type;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

paddle.Node = class {

    constructor(metadata, op, args) {
        const type = op.type;
        this._type = metadata.type(type) || { name: type };
        this._name = op.name || '';
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        if (op.attrs) {
            this._attributes = op.attrs.map((attr) => new paddle.Attribute(metadata.attribute(type, this._name), attr));
        }
        if (op.inputs) {
            for (const input of op.inputs) {
                if (input.arguments.length > 0) {
                    this._inputs.push(new paddle.Argument(input.parameter, input.arguments.map((name) => args.get(name))));
                }
            }
        }
        if (op.outputs) {
            for (const output of op.outputs) {
                if (output.arguments.length > 0) {
                    this._outputs.push(new paddle.Argument(output.parameter, output.arguments.map((name) => args.get(name))));
                }
            }
        }
        this._update(this._inputs, 'X');
        this._update(this._inputs, 'Input');
        this._update(this._outputs, 'Y');
        this._update(this._outputs, 'Out');
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
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

    get chain() {
        return this._chain;
    }

    _update(list, name) {
        let item = null;
        for (let i = 0; i < list.length; i++) {
            if (list[i].name == name) {
                item = list[i];
                list.splice(i, 1);
                break;
            }
        }
        if (item) {
            list.splice(0, 0, item);
        }
    }
};

paddle.Attribute = class {

    constructor(schema, attr) {
        this._name = attr.name;
        this._value = '?';
        switch (attr.type) {
            case paddle.AttributeType.STRING:
                this._type = 'string';
                this._value = attr.s;
                break;
            case paddle.AttributeType.STRINGS:
                this._type = 'string[]';
                this._value = Array.from(attr.strings);
                break;
            case paddle.AttributeType.BOOLEAN:
                this._type = 'boolean';
                this._value = attr.b;
                break;
            case paddle.AttributeType.BOOLEANS:
                this._type = 'boolean[]';
                this._value = attr.bools ? Array.from(attr.bools) : attr.bools;
                break;
            case paddle.AttributeType.FLOAT:
                this._type = 'float32';
                this._value = attr.f;
                break;
            case paddle.AttributeType.FLOATS:
                this._type = 'float32[]';
                this._value = attr.floats ? Array.from(attr.floats) : attr.floats;
                break;
            case paddle.AttributeType.FLOAT64:
                this._type = 'float64';
                this._value = attr.float64;
                break;
            case paddle.AttributeType.FLOAT64S:
                this._type = 'float64[]';
                this._value = attr.float64s ? Array.from(attr.float64s) : attr.float64s;
                break;
            case paddle.AttributeType.INT:
                this._type = 'int32';
                this._value = attr.i;
                break;
            case paddle.AttributeType.INTS:
                this._type = 'int32[]';
                this._value = attr.ints ? Array.from(attr.ints) : attr.ints;
                break;
            case paddle.AttributeType.LONG:
                this._type = 'int64';
                break;
            case paddle.AttributeType.LONGS:
                this._type = 'int64[]';
                break;
            default:
                break;
        }
        switch (this._name) {
            case 'use_mkldnn':
            case 'use_cudnn':
            case 'op_callstack':
            case 'op_role':
            case 'op_role_var':
            case 'op_namescope':
            case 'is_test':
                this._visible = false;
                break;
            default:
                break;
        }
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                const defaultValue = schema.default;
                const value = this._value;
                if (defaultValue == value) {
                    this._visible = false;
                } else if (Array.isArray(value) && Array.isArray(defaultValue) && value.length == defaultValue.length) {
                    if (value.every((item, index) => item == defaultValue[index])) {
                        this._visible = false;
                    }
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

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

paddle.Tensor = class {

    constructor(type, data, category) {
        this._type = type;
        this._data = data;
        this._category = category || '';
    }

    get category() {
        return this._category;
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._data;
    }
};

paddle.TensorType = class {

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

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

paddle.TensorShape = class {

    constructor(dimensions) {
        dimensions = dimensions.map((dimension) => Number.isInteger(dimension) ? dimension : dimension.toNumber());
        this._dimensions = dimensions.map((dimension) => {
            return dimension != -1 ? dimension : '?';
        });
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return (this._dimensions && this._dimensions.length) ? ('[' + this._dimensions.join(',') + ']') : '';
    }
};

paddle.Entries = class {

    static open(context) {
        const extension = [ 'zip', 'tar' ].find((extension) => context.entries(extension).size > 0);
        if (extension) {
            const entries = new Map(Array.from(context.entries(extension)).filter((entry) => !entry[0].endsWith('/') && !entry[0].split('/').pop().startsWith('.')).slice());
            if (entries.size > 2 && Array.from(entries).every((entry) => entry[0].split('_').length > 0 && entry[1].peek(16).every((value) => value === 0x00))) {
                return new paddle.Entries(entries);
            }
        }
        return null;
    }

    constructor(data) {
        this._data = data;
    }

    get format() {
        return 'PaddlePaddle Weights';
    }

    get weights() {
        this._read();
        return this._weights;
    }

    _read() {
        if (!this._weights) {
            let rootFolder = null;
            for (const entry of this._data) {
                const name = entry[0];
                if (!name.startsWith('.') || name.startsWith('./')) {
                    const parts = name.split('/');
                    const folder = ((parts.length > 2 && parts[0] === '.') ? ('./' + parts[1] + '/') : (parts.length > 1 ? parts[0] + '/' : ''));
                    rootFolder = (rootFolder === null) ? folder : (rootFolder !== '' && folder !== rootFolder) ? '' : folder;
                }
            }
            this._weights = new Map();
            for (const entry of this._data) {
                if (entry[0].startsWith(rootFolder)) {
                    const name = entry[0].substring(rootFolder.length);
                    const stream = entry[1];
                    const tensor = paddle.Utility.openTensorDesc(stream);
                    this._weights.set(name, tensor);
                }
            }
        }
    }
};

paddle.Pickle = class {

    static open(context) {
        const obj = context.open('pkl');
        const container = new paddle.Pickle(obj);
        return container.weights !== null ? container : null;
    }

    constructor(obj) {
        this._weights = null;
        if (obj && !Array.isArray(obj) && (obj instanceof Map || Object(obj) === obj)) {
            const entries = (obj) => {
                return obj instanceof Map ? Array.from(obj) : Object(obj) === obj ? Object.entries(obj) : [];
            };
            const filter = (obj) => {
                const list = [];
                if (obj && !Array.isArray(obj)) {
                    for (const entry of entries(obj)) {
                        const name = entry[0];
                        if (name !== 'StructuredToParameterName@@') {
                            let value = entry[1];
                            value = value && Array.isArray(value) && value.length === 2 && value[0] === name ? value[1] : value;
                            if (value && !Array.isArray(value) && value.__class__ && value.__class__.__module__ === 'numpy' && value.__class__.__name__ === 'ndarray') {
                                list.push([ name, value ]);
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
                if (list.filter((entry) => entry[0] !== 'StructuredToParameterName@@').length === 1) {
                    const weights = filter(list[0][1]);
                    if (weights.length > 0) {
                        this._weights = weights;
                    }
                }
                if (this._weights === null && list.filter((entry) => entry[0] === 'StructuredToParameterName@@').length > 0) {
                    this._weights = [];
                }
            }
        }
    }

    get format() {
        return 'PaddlePaddle Pickle';
    }

    get weights() {
        if (this._weights && Array.isArray(this._weights)) {
            const weights = new Map();
            for (const entry of this._weights) {
                const name = entry[0];
                const value = entry[1];
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
            if (context.identifier === '__model__.nb' || context.identifier === 'param.nb') {
                if (buffer[0] > 2 || buffer[1] !== 0x00 || buffer[2] !== 0x76 || buffer[2] !== 0x32) {
                    return new paddle.NaiveBuffer(stream, -1);
                }
            }
            if (buffer[1] === 0x00 && buffer[0] <= 2) {
                return new paddle.NaiveBuffer(stream, buffer[0]);
            }
        }
        return null;
    }

    constructor(stream, meta_version) {
        this.stream = stream;
        this.meta_version = meta_version;
    }

    get format() {
        this._read();
        return this._format;
    }

    get model() {
        this._read();
        return this._model;
    }

    get weights() {
        this._read();
        return this._weights;
    }

    _read() {
        if (this.stream) {
            const reader = new base.BinaryReader(this.stream);
            if (this.meta_version >= 2) {
                reader.skip(2);
            }
            delete this.stream;
            const decoder = new TextDecoder();
            const opt_version = reader.read(16);
            const version = decoder.decode(opt_version.slice(0, opt_version.indexOf(0x00)));
            this._format = 'Paddle Lite' + (version && version.match(/^v\d+\.\d+.\d+$/) ? ' ' + version : '');
            const topo_size = reader.uint64();
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
                    throw new paddle.Error("Paddle Lite meta format '" + this.meta_version.toString() + "' is deprecated.");
                }
                case 2: {
                    const topo_data = new Uint8Array(topo_size);
                    topo_data.set(reader.read(topo_size), 0);
                    this._model = openProgramDesc(topo_data);
                    reader.uint16(); // version
                    reader.uint16(); // meta_size
                    const header_size = reader.uint16();
                    const params_size = reader.uint16();
                    reader.uint32(); // max_tensor_size
                    reader.skip(header_size - 6);
                    this._weights = new Map();
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
                        this._weights.set(desc.name, tensor);
                    }
                    break;
                }
                default: {
                    throw new paddle.Error("Unsupported Paddle Lite naive buffer meta format '" + this.meta_version.toString() + "'.");
                }
            }
        }
    }
};


paddle.Utility = class {

    static createTensorType(data_type, shape) {
        if (!paddle.Utility._dataTypes) {
            const length = Math.max.apply(null, Object.entries(paddle.DataType).map((entry) => entry[1]));
            paddle.Utility._dataTypes = new Array(length);
            const map = new Map([ [ 'bool', 'boolean' ], [ 'bf16', 'bfloat16' ], [ 'fp16', 'float16' ], [ 'fp32', 'float32' ], [ 'fp64', 'float64' ] ]);
            for (const entry of Object.entries(paddle.DataType)) {
                const index = entry[1];
                const key = entry[0].toLowerCase();
                paddle.Utility._dataTypes[index] = map.has(key) ? map.get(key) : key;
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
        const length = new base.BinaryReader(stream.read(4)).uint32();
        const buffer = stream.read(length);
        const reader = protobuf.BinaryReader.open(buffer);
        const tensorDesc = paddle.proto.VarType.TensorDesc.decode(reader);
        const size = tensorDesc.dims.reduce((a, b) => a * b.toNumber(), 1);
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
            default: throw new paddle.Error("Invalid inference params data type '" + tensorDesc.data_type + "'.");
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = paddle.ModelFactory;
}
