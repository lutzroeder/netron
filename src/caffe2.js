/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var caffe2 = caffe2 || {};
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');

caffe2.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'pb') {
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb') ||
                identifier.startsWith('predict_net') || identifier.startsWith('init_net')) {
                return true;
            }
            const tags = context.tags('pb');
            // ignore input_0.pb, output_0.pb
            if (tags.size > 0 &&
                tags.has(1) && tags.get(1) == 0 &&
                tags.has(2) && tags.get(2) == 0 &&
                tags.has(9) && tags.get(9) == 2) {
                return false;
            }
            if (tags.size > 0 &&
                Array.from(tags.values()).some((v) => v === 5)) {
                return false;
            }
            if (tags.size > 0 &&
                (!tags.has(1) || tags.get(1) === 2) &&
                (!tags.has(2) || tags.get(2) === 2) &&
                (!tags.has(7) || tags.get(7) === 2) &&
                (!tags.has(8) || tags.get(8) === 2)) {
                let buffer = context.buffer;
                if (buffer.length > 3 && buffer[0] == 0x0A) {
                    let size = buffer[1];
                    if (size < 64 && buffer.length > 2 + size + 1 && buffer.slice(2, 2 + size).every((c) => c >= 32 && c <= 127) && buffer[2 + size] == 0x12) {
                        return true;
                    }
                }
                if (buffer.length > 3 && buffer[0] == 0x12) {
                    return true;
                }
            }
        }
        if (extension == 'pbtxt' || extension == 'prototxt') {
            if (identifier.endsWith('predict_net')) {
                return true;
            }
            const tags = context.tags('pbtxt');
            if (tags.has('op')) {
                if (context.identifier === 'ops.pbtxt' && context.text.indexOf('  attr {') !== -1) {
                    return false;
                }
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./caffe2-proto').then(() => {
            return caffe2.Metadata.open(host).then((metadata) => {
                const identifier = context.identifier;
                const parts = identifier.split('.');
                const extension = parts.pop().toLowerCase();
                const base = parts.join('.');
                if (extension == 'pbtxt' || extension == 'prototxt') {
                    const open_text = (predict, init) => {
                        let predict_net = null;
                        let init_net = null;
                        try {
                            caffe2.proto = protobuf.roots.caffe2.caffe2;
                            const reader = prototxt.TextReader.create(predict);
                            reader.field = function(tag, message) {
                                if (message instanceof caffe2.proto.DeviceOption) {
                                    message[tag] = this.skip();
                                    return;
                                }
                                throw new Error("Unknown field '" + tag + "'" + this.location());
                            };
                            predict_net = caffe2.proto.NetDef.decodeText(reader);
                        }
                        catch (error) {
                            throw new caffe2.Error("File text format is not caffe2.NetDef (" + error.message + ") in '" + identifier + "'.");
                        }
                        try {
                            caffe2.proto = protobuf.roots.caffe2.caffe2;
                            init_net = (typeof init === 'string') ?
                                caffe2.proto.NetDef.decodeText(prototxt.TextReader.create(init)) :
                                caffe2.proto.NetDef.decode(init);
                        }
                        catch (error) {
                            // continue regardless of error
                        }
                        try {
                            return new caffe2.Model(metadata, predict_net, init_net);
                        }
                        catch (error) {
                            host.exception(error, false);
                            const message = error && error.message ? error.message : error.toString();
                            throw new caffe2.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                        }
                    };
                    if (base.toLowerCase().endsWith('init_net') || base.toLowerCase().startsWith('init_net')) {
                        return context.request(identifier.replace('init_net', 'predict_net'), 'utf-8').then((text) => {
                            return open_text(text, context.text);
                        }).catch(() => {
                            return open_text(context.text, null);
                        });
                    }
                    else if (base.toLowerCase().endsWith('predict_net') || base.toLowerCase().startsWith('predict_net')) {
                        return context.request(identifier.replace('predict_net', 'init_net').replace(/\.pbtxt/, '.pb'), null).then((buffer) => {
                            return open_text(context.text, buffer);
                        }).catch(() => {
                            return context.request(identifier.replace('predict_net', 'init_net'), 'utf-8').then((text) => {
                                return open_text(context.text, text);
                            }).catch(() => {
                                return open_text(context.text, null);
                            });
                        });
                    }
                    else {
                        return context.request(base + '_init.pb', null).then((buffer) => {
                            return open_text(context.text, buffer);
                        }).catch(() => {
                            return open_text(context.text, null);
                        });
                    }
                }
                else {
                    const open_binary = (predict, init) => {
                        let predict_net = null;
                        let init_net = null;
                        try {
                            caffe2.proto = protobuf.roots.caffe2.caffe2;
                            predict_net = caffe2.proto.NetDef.decode(predict);
                        }
                        catch (error) {
                            throw new caffe2.Error("File format is not caffe2.NetDef (" + error.message + ") in '" + identifier + "'.");
                        }
                        try {
                            if (init) {
                                caffe2.proto = protobuf.roots.caffe2.caffe2;
                                init_net = caffe2.proto.NetDef.decode(init);
                            }
                        }
                        catch (error) {
                            // continue regardless of error
                        }
                        try {
                            return new caffe2.Model(metadata, predict_net, init_net);
                        }
                        catch (error) {
                            host.exception(error, false);
                            const message = error && error.message ? error.message : error.toString();
                            throw new caffe2.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                        }
                    };
                    if (base.toLowerCase().endsWith('init_net')) {
                        return context.request(base.replace(/init_net$/, '') + 'predict_net.' + extension, null).then((buffer) => {
                            return open_binary(buffer, context.buffer);
                        }).catch(() => {
                            return open_binary(context.buffer, null);
                        });
                    }
                    else if (base.toLowerCase().endsWith('_init')) {
                        return context.request(base.replace(/_init$/, '') + '.' + extension, null).then((buffer) => {
                            return open_binary(buffer, context.buffer);
                        }).catch(() => {
                            return open_binary(context.buffer, null);
                        });
                    }
                    else if (base.toLowerCase().endsWith('predict_net') || base.toLowerCase().startsWith('predict_net')) {
                        return context.request(identifier.replace('predict_net', 'init_net'), null).then((buffer) => {
                            return open_binary(context.buffer, buffer);
                        }).catch(() => {
                            return open_binary(context.buffer, null);
                        });
                    }
                    else {
                        return context.request(base + '_init.' + extension, null).then((buffer) => {
                            return open_binary(context.buffer, buffer);
                        }).catch(() => {
                            return open_binary(context.buffer, null);
                        });
                    }
                }
            });
        });
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

        let inputs = new Map();
        for (const input of netDef.external_input) {
            inputs.set(input, {});
        }
        if (init) {
            for (const op of init.op) {
                if (op.output && op.output.length == 1) {
                    const name = op.output[0];
                    if (!inputs.has(name)) {
                        inputs.set(name, {});
                    }
                    let initializer = inputs.get(name);
                    for (const arg of op.arg) {
                        initializer[arg.name] = arg;
                    }
                    switch (op.type) {
                        case 'GivenTensorFill':
                            initializer.dataType = 'float32';
                            break;
                        case 'GivenTensorDoubleFill':
                            initializer.dataType = 'float64';
                            break;
                        case 'GivenTensorBoolFill':
                            initializer.dataType = 'boolean';
                            break;
                        case 'GivenTensorByteStringToUInt8Fill':
                            initializer.dataType = 'uint8';
                            break;
                        case 'GivenTensorInt16Fill':
                        case 'GivenTensorSInt16Fill':
                            initializer.dataType = 'int16';
                            break;
                        case 'GivenTensorIntFill':
                            initializer.dataType = 'int32';
                            break;
                        case 'GivenTensorInt64Fill':
                            initializer.dataType = 'int64';
                            break;
                        case 'GivenTensorStringFill':
                            initializer.dataType = 'string';
                            break;
                        case 'Int8GivenIntTensorFill':
                            initializer.dataType = 'int32';
                            break;
                        case 'Int8GivenTensorFill':
                            initializer.dataType = 'int8';
                            break;
                        case 'XavierFill':
                            break;
                        case 'ConstantFill':
                            break;
                        default:
                            throw new caffe2.Error("Unknown init op '" + op.type + "'.");
                    }
                    if (initializer.values && (initializer.values.floats.length !== 1 || initializer.values.floats[0] !== 0)) {
                        initializer.input = false;
                    }
                }
            }
        }

        let scope = {};
        let index = 0;
        for (let op of netDef.op) {
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

        let lastNode = null;
        let lastOutput = null;
        for (let op of netDef.op) {
            let node = new caffe2.Node(metadata, op, inputs);
            if (op.input.length == 1 &&
                op.output.length >= 1 &&
                op.input[0].split('\n').shift() == op.output[0].split('\n').shift() &&
                lastNode &&
                lastOutput == op.input[0].split('\n').shift()) {
                lastNode.chain.push(node);
            }
            else {
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
        for (let input of netDef.external_input) {
            if (netDef.external_input.length > 1) {
                const initializer = inputs.get(input);
                if (initializer && initializer.input === false) {
                    continue;
                }
            }
            this._inputs.push(new caffe2.Parameter(input, [ new caffe2.Argument(input, null, null) ]));
        }

        this._outputs = [];
        for (let output of netDef.external_output) {
            this._outputs.push(new caffe2.Parameter(output, [ new caffe2.Argument(output, null, null) ]));
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

    toString() {
        return 'graph(' + this.name + ')';
    }
};

caffe2.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

caffe2.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new caffe2.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, op, initializers) {
        this._name = op.name || '';
        this._device = op.engine || '';
        this._metadata = metadata;
        this._operator = op.type;
        this._chain = [];

        this._attributes = [];
        for (let arg of op.arg) {
            this._attributes.push(new caffe2.Attribute(metadata, this, arg));
        }

        const schema = metadata.type(this._operator);

        const inputs = op.input;
        const outputs = op.output;

        let tensors = {};
        let index = 0;
        for (const input of inputs) {
            if (index > 0 && initializers.has(input)) {
                const initializer = initializers.get(input);
                tensors[input] = new caffe2.Tensor(input, initializer);
                initializer.input = false;
            }
            index++;
        }
        for (const output of outputs) {
            if (initializers.has(output)) {
                const initializer = initializers.get(output);
                initializer.input = false;
            }
        }
        this._inputs = [];
        let inputIndex = 0;
        if (schema && schema.inputs) {
            for (let inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    let inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    let inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => {
                        return new caffe2.Argument(id, null, tensors[id]);
                    });
                    this._inputs.push(new caffe2.Parameter(inputDef.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        else {
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => {
                let inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new caffe2.Parameter(inputName, [
                    new caffe2.Argument(input, null, tensors[input])
                ]);
            }));
        }

        this._outputs = [];
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (let outputDef of schema.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    let outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    let outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => new caffe2.Argument(id));
                    this._outputs.push(new caffe2.Parameter(outputDef.name, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        else {
            this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
                let outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new caffe2.Parameter(outputName, [
                    new caffe2.Argument(output, null, null)
                ]);
            }));
        }
    }

    get name() {
        return this._name || '';
    }

    get device() {
        return this._device || '';
    }

    get operator() {
        return this._operator;
    }

    get metadata() {
        return this._metadata.type(this._operator);
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

    constructor(metadata, node, arg) {
        this._node = node;
        this._name = arg.name;
        if (arg.floats && arg.floats.length > 0) {
            this._value = arg.floats;
        }
        else if (arg.ints && arg.ints.length > 0) {
            this._value = arg.ints;
        }
        else if (arg.nets && arg.nets.length > 0) {
            this._value = arg.nets.map((net) => new caffe2.Graph(metadata, net, null));
            this._type = 'graph[]';
        }
        else if (arg.n) {
            this._value = new caffe2.Graph(metadata, arg.n, null);
            this._type = 'graph';
        }
        else if (arg.i != 0) {
            this._value = arg.i;
        }
        else {
            this._value = arg.i;
        }

        const schema = metadata.attribute(this._node.operator, this._name);
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'type')) {
                this._type = schema.type;
                if (this._type == 'boolean') {
                    switch (this._value) {
                        case 1: this._value = true; break;
                        case 0: this._value = false; break;
                    }
                }
            }
        }

        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default || (this._value && this._value.toString() == schema.default.toString())) {
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

    get kind() {
        return 'Initializer';
    }

    get quantization() {
        if (this._scale != 0 || this._zeroPoint != 0) {
            return this._scale.toString() + ' * ' + (this._zeroPoint == 0 ? 'q' : ('(q - ' + this._zeroPoint.toString() + ')'));
        }
        return null;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return caffe2.Tensor._stringify(value, '', '    ');
    }

    _context() {
        let context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        if (!this._values) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        if (this._values.floats == -1) {
            context.state = 'Tensor data is too large to load in Chrome.';
            return context;
        }
        switch (this._type.dataType) {
            case 'float32':
                context.data = this._values.floats;
                break;
            case 'boolean':
                context.data = this._values.ints;
                break;
            case 'int8':
                context.data = new Int8Array(this._values.s);
                break;
            case 'int32':
                context.data = this._values.ints;
                break;
            default:
                context.state = 'Unknown data type.';
                return context;
        }
        context.shape = this._type.shape.dimensions;
        context.dataType = this._type.dataType;
        return context;
    }

    _decode(context, dimension) {
        let results = [];
        let size = context.shape[dimension];
        if (dimension == context.shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float32':
                        results.push(context.data[context.index]);
                        break;
                    case 'boolean':
                        results.push(context.data[context.index] == 0 ? false : true);
                        break;
                    case 'int8':
                        results.push(context.data[context.index]);
                        break;
                    case 'int32':
                        results.push(context.data[context.index]);
                        break;
                    default:
                        context.state = 'Unknown data type.';
                        break;
                }
                context.index++;
                context.count++;
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
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            let result = [];
            result.push(indentation + '[');
            const items = value.map((item) => caffe2.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
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

caffe2.Metadata = class {

    static open(host) {
        if (caffe2.Metadata._metadata) {
            return Promise.resolve(caffe2.Metadata._metadata);
        }
        return host.request(null, 'caffe2-metadata.json', 'utf-8').then((data) => {
            caffe2.Metadata._metadata = new caffe2.Metadata(data);
            return caffe2.Metadata._metadata;
        }).catch(() => {
            caffe2.Metadata._metadata = new caffe2.Metadata(null);
            return caffe2.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (let attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
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