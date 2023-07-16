
var caffe2 = {};
var protobuf = require('./protobuf');

caffe2.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'pb') {
            const tags = context.tags('pb');
            if (tags.size > 0 &&
                Array.from(tags.keys()).every((tag) => tag <= 9) &&
                Array.from(tags.values()).every((type) => type <= 4)) {
                if (tags.size === 1 && tags.get(2) === 2 && identifier.endsWith('saved_model.pb')) {
                    return undefined;
                }
                const schema = [[1,2],[2,2],[3,2],[4,0],[5,2],[6,2],[7,2],[8,2],[9,2]];
                if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                    const stream = context.stream;
                    if (stream.length > 3) {
                        const buffer = stream.peek(Math.min(stream.length, 67));
                        if (buffer[0] == 0x0A) {
                            const size = buffer[1];
                            if (size < 64 &&
                                buffer.length > 2 + size + 1 &&
                                buffer.slice(2, 2 + size).every((c) => c >= 32 && c <= 127) &&
                                buffer[2 + size] == 0x12) {
                                return 'caffe2.pb';
                            }
                        }
                        if (buffer[0] == 0x12) {
                            return 'caffe2.pb';
                        }
                    }
                }
            }
        }
        if (extension === 'pbtxt' || extension === 'prototxt') {
            const tags = context.tags('pbtxt');
            if (tags.has('op') && !tags.has('op.attr') && !tags.has('op.graph_op_name') && !tags.has('op.endpoint')) {
                return 'caffe2.pbtxt';
            }
        }
        return undefined;
    }

    async open(context, target) {
        await context.require('./caffe2-proto');
        const metadata = await context.metadata('caffe2-metadata.json');
        const identifier = context.identifier;
        const parts = identifier.split('.');
        const extension = parts.pop().toLowerCase();
        const base = parts.join('.');
        switch (target) {
            case 'caffe2.pbtxt': {
                const openText = (predictBuffer, initBuffer, initTextFormat) => {
                    let predict_net = null;
                    let init_net = null;
                    try {
                        caffe2.proto = protobuf.get('caffe2').caffe2;
                        const reader = protobuf.TextReader.open(predictBuffer);
                        reader.field = function(tag, message) {
                            if (message instanceof caffe2.proto.DeviceOption) {
                                message[tag] = this.read();
                                return;
                            }
                            throw new Error("Unknown field '" + tag + "'" + this.location());
                        };
                        predict_net = caffe2.proto.NetDef.decodeText(reader);
                    } catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new caffe2.Error('File text format is not caffe2.NetDef (' + message.replace(/\.$/, '') + ').');
                    }
                    try {
                        caffe2.proto = protobuf.get('caffe2').caffe2;
                        if (initBuffer) {
                            if (initTextFormat) {
                                const reader = protobuf.TextReader.open(initBuffer);
                                init_net = caffe2.proto.NetDef.decodeText(reader);
                            } else {
                                const reader = protobuf.BinaryReader.open(initBuffer);
                                init_net = caffe2.proto.NetDef.decode(reader);
                            }
                        }
                    } catch (error) {
                        // continue regardless of error
                    }
                    return new caffe2.Model(metadata, predict_net, init_net);
                };
                if (base.toLowerCase().endsWith('init_net') || base.toLowerCase().startsWith('init_net')) {
                    try {
                        const stream =  await context.request(identifier.replace('init_net', 'predict_net'), null);
                        const buffer = stream.read();
                        return openText(buffer, context.stream.peek(), true);
                    } catch (error) {
                        return openText(context.stream.peek(), null, true);
                    }
                }
                if (base.toLowerCase().endsWith('predict_net') || base.toLowerCase().startsWith('predict_net')) {
                    try {
                        const stream = await context.request(identifier.replace('predict_net', 'init_net').replace(/\.pbtxt/, '.pb'), null);
                        const buffer = stream.read();
                        return openText(context.stream.peek(), buffer, false);
                    } catch (error) {
                        try {
                            const stream = await context.request(identifier.replace('predict_net', 'init_net'), null);
                            const buffer = stream.read();
                            return openText(context.stream.peek(), buffer, true);
                        } catch (error) {
                            return openText(context.stream.peek(), null, true);
                        }
                    }
                }
                try {
                    const stream = await context.request(base + '_init.pb', null);
                    const buffer = stream.read();
                    return openText(context.stream.peek(), buffer, false);
                } catch (error) {
                    return openText(context.stream.peek(), null, false);
                }
            }
            case 'caffe2.pb': {
                const openBinary = (predictBuffer, initBuffer) => {
                    let predict_net = null;
                    let init_net = null;
                    try {
                        caffe2.proto = protobuf.get('caffe2').caffe2;
                        const reader = protobuf.BinaryReader.open(predictBuffer);
                        predict_net = caffe2.proto.NetDef.decode(reader);
                    } catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new caffe2.Error('File format is not caffe2.NetDef (' + message.replace(/\.$/, '') + ').');
                    }
                    try {
                        if (initBuffer) {
                            caffe2.proto = protobuf.get('caffe2').caffe2;
                            const reader = protobuf.BinaryReader.open(initBuffer);
                            init_net = caffe2.proto.NetDef.decode(reader);
                        }
                    } catch (error) {
                        // continue regardless of error
                    }
                    return new caffe2.Model(metadata, predict_net, init_net);
                };
                if (base.toLowerCase().endsWith('init_net')) {
                    try {
                        const stream = await context.request(base.replace(/init_net$/, '') + 'predict_net.' + extension, null);
                        const buffer = stream.read();
                        return openBinary(buffer, context.stream.peek());
                    } catch (error) {
                        return openBinary(context.stream.peek(), null);
                    }
                }
                if (base.toLowerCase().endsWith('_init')) {
                    try {
                        const stream = await context.request(base.replace(/_init$/, '') + '.' + extension, null);
                        const buffer = stream.read();
                        return openBinary(buffer, context.stream.peek());
                    } catch (error) {
                        return openBinary(context.stream.peek(), null);
                    }
                }
                if (base.toLowerCase().endsWith('predict_net') || base.toLowerCase().startsWith('predict_net')) {
                    try {
                        const stream = await context.request(identifier.replace('predict_net', 'init_net'), null);
                        const buffer = stream.read();
                        return openBinary(context.stream.peek(), buffer);
                    } catch (error) {
                        return openBinary(context.stream.peek(), null);
                    }
                }
                try {
                    const stream = await context.request(base + '_init.' + extension, null);
                    const buffer = stream.read();
                    return openBinary(context.stream.peek(), buffer);
                } catch (error) {
                    return openBinary(context.stream.peek(), null);
                }
            }
            default: {
                throw new caffe2.Error("Unsupported Caffe2 format '" + target + "'.");
            }
        }
    }
};

caffe2.Model = class {

    constructor(metadata, predict_net, init_net) {
        this._domain = predict_net.domain || null;
        const graph = new caffe2.Graph(metadata, predict_net, init_net);
        this._graphs = [ graph ];
    }

    get format() {
        return 'Caffe2';
    }

    get domain() {
        return this._domain;
    }

    get graphs() {
        return this._graphs;
    }
};

caffe2.Graph = class {

    constructor(metadata, netDef, init) {
        this._name = netDef.name || '';
        this._type = netDef.type || '';
        this._nodes = [];
        const initializers = new Set();
        const tensors = new Map();
        for (const name of netDef.external_input) {
            tensors.set(name, new caffe2.Tensor(name, {}));
        }
        if (init) {
            const dataTypes = new Map([
                [ 'GivenTensorFill', 'float32' ],
                [ 'GivenTensorDoubleFill', 'float64' ],
                [ 'GivenTensorBoolFill', 'boolean' ],
                [ 'GivenTensorByteStringToUInt8Fill', 'uint8' ],
                [ 'GivenTensorInt16Fill', 'int16' ],
                [ 'GivenTensorSInt16Fill', 'int16' ],
                [ 'GivenTensorIntFill', 'int32' ],
                [ 'GivenTensorInt64Fill', 'int64' ],
                [ 'GivenTensorStringFill', 'string' ],
                [ 'Int8GivenIntTensorFill', 'int32' ],
                [ 'Int8GivenTensorFill', 'int8' ],
                [ 'XavierFill', null ],
                [ 'ConstantFill', null ]
            ]);
            for (const op of init.op) {
                if (op.output && op.output.length == 1) {
                    const name = op.output[0];
                    const tensor = {};
                    for (const arg of op.arg) {
                        tensor[arg.name] = arg;
                    }
                    if (!dataTypes.has(op.type)) {
                        throw new caffe2.Error("Unsupported init op '" + op.type + "'.");
                    }
                    tensor.dataType = dataTypes.get(op.type);
                    if (tensor.values && tensor.values.floats && (tensor.values.floats.length !== 1 || tensor.values.floats[0] !== 0)) {
                        initializers.add(name);
                    }
                    tensors.set(name, new caffe2.Tensor(name, tensor));
                }
            }
        }
        const scope = {};
        let index = 0;
        for (const op of netDef.op) {
            op.input = op.input.map((input) => scope[input] ? scope[input] : input);
            op.output = op.output.map((output) => {
                if (scope[output]) {
                    const next = output + '\n' + index.toString(); // custom argument id
                    scope[output] = next;
                    return next;
                }
                scope[output] = output;
                return output;
            });
            index++;
        }
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (!args.has(name)) {
                args.set(name, new caffe2.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new caffe2.Value("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        for (const op of netDef.op) {
            let index = 0;
            for (const name of op.input) {
                if (index > 0 && tensors.has(name)) {
                    if (!args.has(name)) {
                        args.set(name, new caffe2.Value(name, null, tensors.get(name)));
                    }
                    initializers.add(name);
                }
                index++;
            }
        }
        for (const op of netDef.op) {
            for (const name of op.output) {
                if (tensors.has(name)) {
                    initializers.add(name);
                }
            }
        }
        let lastNode = null;
        let lastOutput = null;
        for (const op of netDef.op) {
            const node = new caffe2.Node(metadata, op, arg);
            if (op.input.length == 1 &&
                op.output.length >= 1 &&
                op.input[0].split('\n').shift() == op.output[0].split('\n').shift() &&
                lastNode &&
                lastOutput == op.input[0].split('\n').shift()) {
                lastNode.chain.push(node);
            } else {
                this._nodes.push(node);
                lastNode = null;
                lastOutput = null;
                if (op.output.length == 1) {
                    lastNode = node;
                    lastOutput = op.output[0].split('\n').shift();
                }
            }
        }
        this._inputs = [];
        for (const input of netDef.external_input) {
            if (netDef.external_input.length > 1 && initializers.has(input)) {
                continue;
            }
            this._inputs.push(new caffe2.Argument(input, [ arg(input) ]));
        }
        this._outputs = [];
        for (const output of netDef.external_output) {
            this._outputs.push(new caffe2.Argument(output, [ arg(output) ]));
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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

caffe2.Argument = class {

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

caffe2.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new caffe2.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
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

    get quantization() {
        if (this._initializer) {
            return this._initializer.quantization;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

caffe2.Node = class {

    constructor(metadata, op, arg) {
        this._name = op.name || '';
        this._device = op.engine || '';
        this._metadata = metadata;
        this._chain = [];
        this._type = metadata.type(op.type);
        this._attributes = op.arg.map((arg) => new caffe2.Attribute(metadata, this._type.name, arg));
        const inputs = op.input;
        const outputs = op.output;
        this._inputs = [];
        let inputIndex = 0;
        if (this._type && this._type.inputs) {
            for (const inputDef of this._type.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => arg(id));
                    this._inputs.push(new caffe2.Argument(inputDef.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        } else {
            this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new caffe2.Argument(inputName, [ arg(input) ]);
            }));
        }
        this._outputs = [];
        let outputIndex = 0;
        if (this._type && this._type.outputs) {
            for (const outputDef of this._type.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => arg(id));
                    this._outputs.push(new caffe2.Argument(outputDef.name, outputArguments));
                    outputIndex += outputCount;
                }
            }
        } else {
            this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new caffe2.Argument(outputName, [ arg(output) ]);
            }));
        }
    }

    get name() {
        return this._name || '';
    }

    get device() {
        return this._device || '';
    }

    get type() {
        return this._type;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get chain() {
        return this._chain;
    }
};

caffe2.Attribute = class {

    constructor(metadata, type, arg) {
        this._name = arg.name;
        if (arg.floats && arg.floats.length > 0) {
            this._value = arg.floats;
        } else if (arg.ints && arg.ints.length > 0) {
            this._value = arg.ints;
        } else if (arg.nets && arg.nets.length > 0) {
            this._value = arg.nets.map((net) => new caffe2.Graph(metadata, net, null));
            this._type = 'graph[]';
        } else if (arg.n) {
            this._value = new caffe2.Graph(metadata, arg.n, null);
            this._type = 'graph';
        } else if (arg.i != 0) {
            this._value = arg.i;
        } else {
            this._value = arg.i;
        }
        metadata = metadata.attribute(type, arg.name);
        if (metadata) {
            if (Object.prototype.hasOwnProperty.call(metadata, 'type')) {
                this._type = metadata.type;
                if (this._type == 'boolean') {
                    this._value = this._value !== 0 && this._value.toString() !== '0' ? true : false;
                }
            }
        }

        if (metadata) {
            if (metadata.visible === false) {
                this._visible = false;
            } else if (metadata.default !== undefined) {
                if (this._value == metadata.default || (this._value && this._value.toString() == metadata.default.toString())) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type || null;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

caffe2.Tensor = class {

    constructor(name, tensor) {
        this._name = name;
        const shape = tensor.shape && tensor.shape.ints ? tensor.shape.ints : null;
        this._type = new caffe2.TensorType(tensor.dataType, new caffe2.TensorShape(shape));
        this._values = tensor.values || null;
        this._scale = tensor.Y_scale ? tensor.Y_scale.f : 0;
        this._zeroPoint = tensor.Y_zero_point ? tensor.Y_zero_point.i : 0;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get category() {
        return 'Initializer';
    }

    get quantization() {
        if (this._scale != 0 || this._zeroPoint != 0) {
            return this._scale.toString() + ' * ' + (this._zeroPoint == 0 ? 'q' : ('(q - ' + this._zeroPoint.toString() + ')'));
        }
        return null;
    }

    get layout() {
        return '|';
    }

    get values() {
        if (!this._values) {
            return null;
        }
        switch (this._type.dataType) {
            case 'float32': return this._values.floats;
            case 'boolean': return this._values.ints;
            case 'int8': return new Int8Array(this._values.s);
            case 'int32': return this._values.ints;
            default: return null;
        }
    }
};

caffe2.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType || '?';
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

caffe2.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

caffe2.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe2 model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = caffe2.ModelFactory;
}