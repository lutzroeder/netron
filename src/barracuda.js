/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var barracuda = barracuda || {};
var base = base || require('./base');
var long = long || { Long: require('long') };

barracuda.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'nn') {
            const buffer = context.buffer;
            if (buffer.length > 12 && buffer[0] <= 0x10 && buffer.subarray(1, 8).every((v) => v == 0x00)) {
                return true;
            }
        }
        return false;
    }

    open(context /*, host */) {
        return barracuda.Metadata.open().then((metadata) => {
            try {
                const nn = new barracuda.NNModel(context.buffer);
                return new barracuda.Model(metadata, nn);
            }
            catch (error) {
                const identifier = context.identifier.toLowerCase();
                const message = error && error.message ? error.message : error.toString();
                throw new barracuda.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
            }
        });
    }
};

barracuda.Model = class {

    constructor(metadata, nn) {
        this._version = nn.version.toString();
        this._graphs = [ new barracuda.Graph(metadata, nn) ];
    }

    get format() {
        return "Barracuda v" + this._version;
    }

    get graphs() {
        return this._graphs;
    }
};

barracuda.Graph = class {

    constructor(metadata, nn) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        for (const input of nn.inputs) {
            this._inputs.push(new barracuda.Parameter(input.name, [
                new barracuda.Argument(input.name, new barracuda.TensorType(4, new barracuda.TensorShape(input.shape)))
            ]));
        }
        for (const output of nn.outputs) {
            this._outputs.push(new barracuda.Parameter(output, [
                new barracuda.Argument(output)
            ]));
        }
        const layers = [];
        const initializers = new Map();
        for (const layer of nn.layers) {
            if (layer.type !== 255 || layer.inputs.length > 0) {
                layers.push(layer);
            }
            else {
                for (const tensor of layer.tensors) {
                    initializers.set(tensor.name, new barracuda.Tensor(tensor));
                }
            }
        }

        for (const layer of layers) {
            this._nodes.push(new barracuda.Node(metadata, layer, initializers));
        }
    }

    get name() {
        return '';
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

barracuda.Parameter = class {

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

barracuda.Argument = class {

    constructor(name, type, initializer) {
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};


barracuda.Node = class {

    constructor(metadata, layer, initializers) {

        this._name = layer.name;
        this._metadata = metadata.type(layer.type) || { name: layer.type.toString() };
        this._operator = this._metadata.name;

        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        const inputs = Array.prototype.slice.call(this._metadata.inputs || [ 'input' ]);
        if (this._metadata.inputs && this._metadata.inputs.length === 1 && this._metadata.inputs[0] === 'inputs') {
            this._inputs.push(new barracuda.Parameter('inputs', layer.inputs.map((input) => {
                const initializer = initializers.has(input) ? initializers.get(input) : null;
                return new barracuda.Argument(input, initializer ? initializer.type : null, initializer);
            })));
        }
        else {
            for (let i = 0; i < layer.inputs.length; i++) {
                const input = layer.inputs[i];
                const initializer = initializers.has(input) ? initializers.get(input) : null;
                this._inputs.push(new barracuda.Parameter(inputs.length > 0 ? inputs.shift() : i.toString(), [
                    new barracuda.Argument(input, initializer ? initializer.type : null, initializer)
                ]));
            }
        }
        for (let i = 0; i < layer.tensors.length; i++) {
            const tensor = layer.tensors[i];
            const initializer = new barracuda.Tensor(tensor);
            this._inputs.push(new barracuda.Parameter(inputs.length > 0 ? inputs.shift() : i.toString(), [
                new barracuda.Argument(tensor.name, initializer.type, initializer)
            ]));
        }
        this._outputs.push(new barracuda.Parameter('output', [
            new barracuda.Argument(this._name)
        ]));
        if (this._operator === 'Activation') {
            if (!barracuda.Activation[layer.activation]) {
                throw new barracuda.Error("Unknown activation '" + layer.activation + "'.");
            }
            this._operator = barracuda.Activation[layer.activation];
        }
        else if (layer.activation !== 0) {
            throw new barracuda.Error("Unsupported activation '" + layer.activation + "' for '" + this._operator + "'.");
        }
        const attribute = (name, type, value, defaultValue) => {
            if (Array.isArray(defaultValue) && Array.isArray(value) && value.length == defaultValue.length && value.every((v, i) => v === defaultValue[i])) {
                return;
            }
            if (typeof defaultValue == 'function' && defaultValue(value)) {
                return;
            }
            if (defaultValue === value) {
                return;
            }
            this._attributes.push(new barracuda.Attribute(name, type, value));
        };
        attribute('strides', 'int32[]', layer.strides, []);
        attribute('pads', 'int32[]', layer.pads, (value) => Array.isArray(value) && (value.every((v) => v === 0) || value.every((v) => v === -1)));
        attribute('size', 'int32[]', layer.pool_size, []);
        attribute('alpha', 'float32', layer.alpha, 1);
        attribute('beta', 'float32', layer.beta, 0);
        attribute('axis', 'int32', layer.axis, -1);
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata;
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
};

barracuda.Attribute = class {

    constructor(name, type, value) {
        this._name = name;
        this._type = type;
        this._value = value;
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
        return true;
    }
};

barracuda.Tensor = class {

    constructor(tensor) {
        this._type = new barracuda.TensorType(tensor.itemsize, new barracuda.TensorShape(tensor.shape));
        this._data = tensor.data;
    }

    get kind() {
        return '';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
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
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.shape || (this._type.shape.dimensions && this._type.shape.dimensions.length == 0)) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'float32':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length == 0 ? [ 1 ] : context.shape;
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
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
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};

barracuda.TensorType = class {

    constructor(itemsize, shape) {
        switch (itemsize) {
            case 4: this._dataType = 'float32'; break;
            default: throw new barracuda.Error("Unsupported data type size '" + itemsize.toString() + "'.");
        }
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

barracuda.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

barracuda.NNModel = class {

    constructor(buffer) {

        // https://github.com/Unity-Technologies/ml-agents/blob/master/ml-agents/mlagents/trainers/barracuda.py
        // https://github.com/Unity-Technologies/ml-agents/blob/master/ml-agents/mlagents/trainers/tensorflow_to_barracuda.py

        const reader = new barracuda.BinaryReader(buffer);
        this._version = reader.int32();
        reader.int32();

        this._inputs = [];
        const modelInputsLength = reader.int32();
        for (let i = 0; i < modelInputsLength; i++) {
            this._inputs.push({
                name: reader.string(),
                shape: reader.shape()
            });
        }
        this._outputs = reader.strings();

        this._memories = [];
        const memoriesLength = reader.int32();
        for (let i = 0; i < memoriesLength; i++) {
            // debugger;
            this._memories.push({
                shape: reader.shape(),
                in: reader.string(),
                out: reader.string()
            });
        }

        this._layers = [];
        const layersLength = reader.int32();
        for (let i = 0; i < layersLength; i++) {
            const layer = {};
            layer.name = reader.string();
            layer.type = reader.int32();
            layer.activation = reader.int32();
            reader.int32();
            reader.int32();
            layer.pads = reader.int32s();
            layer.strides = reader.int32s();
            layer.pool_size = reader.int32s();
            layer.axis = reader.int32();
            layer.alpha = reader.float32();
            layer.beta = reader.float32();
            reader.int32();
            layer.inputs = reader.strings();
            layer.tensors = [];
            const tensorsLength = reader.int32();
            for (let j = 0; j < tensorsLength; j++) {
                layer.tensors.push({
                    name: reader.string(),
                    shape: reader.shape(),
                    offset: reader.int64(),
                    itemsize: reader.int32(),
                    length: reader.int32()
                });
            }
            this._layers.push(layer);
        }
        for (const layer of this._layers) {
            for (const tensor of layer.tensors) {
                tensor.data = reader.bytes(tensor.offset * tensor.itemsize, tensor.length * tensor.itemsize);
            }
        }
    }

    get version() {
        return this._version;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get memories() {
        return this._memories;
    }

    get layers() {
        return this._layers;
    }
};

barracuda.Activation = {
    0: "Linear", 1: "Relu", 2: "Softmax", 3: "Tanh", 4: "Sigmoid", 5: "Elu", 6: "Relu6", 7: "LeakyRelu",
    8: "Selu", 9: "Swish", 10: "LogSoftmax", 11: "Softplus", 12: "Softsign",
    100: "Abs", 101: "Neg", 102: "Ceil", 104: "Floor", 111: "Sqrt", 113: "Exp", 114: "Log",
    200: "Acos", 201: "Acosh", 202: "Asin", 203: "Asinh", 204: "Atan", 205: "Atanh", 206: "Cos", 207: "Cosh", 208: "Sin", 209: "Sinh", 210: "Tan"
};

barracuda.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new barracuda.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    bytes(offset, length) {
        const start = this._position + offset;
        const end = start + length;
        if (end > this._buffer.length) {
            throw new barracuda.Error('Expected ' + (end - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        return this._buffer.slice(start, end);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    int32s() {
        const values = [];
        const count = this.int32();
        for (let i = 0; i < count; i++) {
            values.push(this.int32());
        }
        return values;
    }

    int64() {
        const value = this.int32();
        if (this.int32() !== 0) {
            throw new barracuda.Error('Invalid int64 value.');
        }
        return value;
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    string() {
        let text = '';
        const size = this.int32();
        let position = this._position;
        this.skip(size);
        for (let i = 0; i < size; i++) {
            text += String.fromCharCode(this._buffer[position++]);
        }
        return text;
    }

    strings() {
        const values = [];
        const length = this.int32();
        for (let i = 0; i < length; i++) {
            values.push(this.string());
        }
        return values;
    }

    shape() {
        return this.int32s();
    }
};

barracuda.Metadata = class {

    static open() {
        barracuda.Metadata._metadata = barracuda.Metadata._metadata || new barracuda.Metadata();
        return Promise.resolve(barracuda.Metadata._metadata);
    }

    constructor() {
        this._map = new Map();
        this._register(0, 'Nop', '');
        this._register(1, 'Dense', 'Layer', [ 'input', 'kernel', 'bias' ]);
        this._register(2, 'MatMul', '', [ 'input', 'kernel', 'bias' ]);
        this._register(20, 'Conv2D', 'Layer', [ 'input', 'kernel', 'bias' ]);
        this._register(21, 'DepthwiseConv2dNative', 'Layer', [ 'input', 'kernel', 'bias' ]);
        this._register(22, 'Conv2DBackpropInput', '');
        this._register(23, 'Upsample2D', '');
        this._register(25, 'MaxPool', 'Pool');
        this._register(26, 'AvgPool', 'Pool');
        this._register(28, 'GlobalAvgPool', 'Pool');
        this._register(29, 'Pad', '');
        this._register(50, 'Activation', 'Activation');
        this._register(51, 'ScaleBias', 'Normalization', [ 'input', 'scale', 'bias' ]);
        this._register(52, 'InstanceNormalization', 'Normalization');
        this._register(53, 'LRN', 'Normalization');
        this._register(64, 'RandomStandardNormal', '');
        this._register(65, 'RandomUniform', '');
        this._register(67, 'OneHot', '');
        this._register(100, 'Add', '', [ 'inputs' ]);
        this._register(101, 'Sub', '', [ 'inputs' ]);
        this._register(102, 'Mul', '', [ 'inputs' ]);
        this._register(103, 'RealDiv', '', [ 'inputs' ]);
        this._register(104, 'Pow', '', [ 'inputs' ]);
        this._register(110, 'Minimum', '', [ 'inputs' ]);
        this._register(111, 'Maximum', '', [ 'inputs' ]);
        this._register(124, 'Max', '', [ 'inputs' ]);
        this._register(125, 'Mean', '', [ 'inputs' ]);
        this._register(126, 'Min', '', [ 'inputs' ]);
        this._register(127, 'Prod', '', [ 'inputs' ]);
        this._register(128, 'Sum', '', [ 'inputs' ]);
        this._register(200, 'Flatten', 'Shape');
        this._register(201, 'Reshape', 'Shape');
        this._register(210, 'Concat', 'Tensor', [ 'inputs' ]);
        this._register(211, 'StridedSlice', 'Shape');
    }

    _register(id, name, category, inputs) {
        this._map.set(id, { name: name, category: category, inputs: inputs });
    }

    type(operator) {
        if (this._map.has(operator)) {
            return this._map.get(operator);
        }
        return null;
    }
};

barracuda.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Barracuda model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = barracuda.ModelFactory;
}
