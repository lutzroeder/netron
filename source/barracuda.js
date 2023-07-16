
// Experimental

var barracuda = {};
var base = require('./base');

barracuda.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream && stream.length > 12) {
            const buffer = stream.peek(12);
            if (buffer[0] <= 0x20 && buffer.subarray(1, 8).every((value) => value == 0x00)) {
                return 'barracuda';
            }
        }
        return null;
    }

    async open(context) {
        const metadata = barracuda.Metadata.open();
        const model = new barracuda.NNModel(context.stream.peek());
        return new barracuda.Model(metadata, model);
    }
};

barracuda.Model = class {

    constructor(metadata, model) {
        this._version = model.version.toString();
        this._graphs = [ new barracuda.Graph(metadata, model) ];
    }

    get format() {
        return "Barracuda v" + this._version;
    }

    get graphs() {
        return this._graphs;
    }
};

barracuda.Graph = class {

    constructor(metadata, model) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (!args.has(name)) {
                type = tensor ? tensor.type : type;
                args.set(name, new barracuda.Value(name, type, tensor));
            } else if (type || tensor) {
                throw new barracuda.Error("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        const layers = [];
        for (const layer of model.layers) {
            if (layer.type !== 255 || layer.inputs.length > 0) {
                layers.push(layer);
            } else {
                for (const tensor of layer.tensors) {
                    arg(tensor.name, null, new barracuda.Tensor(tensor));
                }
            }
        }
        for (const input of model.inputs) {
            this._inputs.push(new barracuda.Argument(input.name, [
                arg(input.name, new barracuda.TensorType(4, new barracuda.TensorShape(input.shape)))
            ]));
        }
        for (const output of model.outputs) {
            this._outputs.push(new barracuda.Argument(output, [
                arg(output)
            ]));
        }
        for (const layer of layers) {
            this._nodes.push(new barracuda.Node(metadata, layer, null, arg));
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

barracuda.Argument = class {

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

barracuda.Value = class {

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

    constructor(metadata, layer, type, arg) {
        this._name = layer.name || '';
        this._type = type ? type : metadata.type(layer.type);
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        const inputs = Array.prototype.slice.call(this._type.inputs || [ 'input' ]);
        if (this._type.inputs && this._type.inputs.length === 1 && this._type.inputs[0].name === 'inputs') {
            this._inputs.push(new barracuda.Argument('inputs', layer.inputs.map((input) => arg(input))));
        } else if (layer.inputs) {
            for (let i = 0; i < layer.inputs.length; i++) {
                const input = layer.inputs[i];
                const name = inputs.length > 0 ? inputs.shift().name : i.toString();
                const argument = new barracuda.Argument(name, [ arg(input) ]);
                this._inputs.push(argument);
            }
        }
        if (layer.tensors) {
            for (let i = 0; i < layer.tensors.length; i++) {
                const tensor = layer.tensors[i];
                const initializer = new barracuda.Tensor(tensor);
                this._inputs.push(new barracuda.Argument(inputs.length > 0 ? inputs.shift().name : i.toString(), [
                    arg(tensor.name, initializer.type, initializer)
                ]));
            }
        }
        if (layer.inputs !== undefined) {
            this._outputs.push(new barracuda.Argument('output', [ arg(this._name) ]));
        }
        if (layer.activation !== undefined && (layer.type === 50 || layer.activation !== 0)) {
            const type = barracuda.Activation[layer.activation];
            if (!type) {
                throw new barracuda.Error("Unsupported activation '" + layer.activation + "'.");
            }
            this._chain = [ new barracuda.Node(metadata, {}, { name: type, category: 'Activation' }, arg) ];
        }
        const attribute = (name, type, value, defaultValue) => {
            if (value === undefined) {
                return;
            }
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
};

barracuda.Tensor = class {

    constructor(tensor) {
        this._type = new barracuda.TensorType(tensor.itemsize, new barracuda.TensorShape(tensor.shape));
        this._values = tensor.data;
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._values;
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
        // https://github.com/Unity-Technologies/barracuda-release/blob/release/1.3.2/Barracuda/Runtime/Core/Model.cs
        const reader = new barracuda.BinaryReader(buffer);
        this._version = reader.int32();
        reader.int32();
        this._inputs = new Array(reader.int32());
        for (let i = 0; i < this._inputs.length; i++) {
            this._inputs[i] = {
                name: reader.string(),
                shape: reader.shape()
            };
        }
        this._outputs = reader.strings();
        this._memories = new Array(reader.int32());
        for (let i = 0; i < this._memories.length; i++) {
            this._memories[i] = {
                shape: reader.shape(),
                in: reader.string(),
                out: reader.string()
            };
        }
        this._layers = new Array(reader.int32());
        for (let i = 0; i < this._layers.length; i++) {
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
            this._layers[i] = layer;
        }
        const position = reader.position;
        for (const layer of this._layers) {
            for (const tensor of layer.tensors) {
                reader.seek(position + (tensor.offset * tensor.itemsize));
                tensor.data = reader.read(tensor.length * tensor.itemsize);
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
    0: "Linear", 1: "Relu", 2: "Softmax", 3: "Tanh", 4: "Sigmoid", 5: "Elu", 6: "Relu6", 7: "LeakyRelu", 8: "Selu", 9: "Swish",
    10: "LogSoftmax", 11: "Softplus", 12: "Softsign", 13: "PRelu",
    20: "Hardmax", 21: "HardSigmoid",
    100: "Abs", 101: "Neg", 102: "Ceil", 103: "Clip", 104: "Floor", 105: "Round",
    110: "Reciprocal", 111: "Sqrt", 113: "Exp", 114: "Log",
    200: "Acos", 201: "Acosh", 202: "Asin", 203: "Asinh", 204: "Atan", 205: "Atanh", 206: "Cos", 207: "Cosh", 208: "Sin", 209: "Sinh", 210: "Tan"
};

barracuda.BinaryReader = class extends base.BinaryReader {

    int32s() {
        const values = new Array(this.int32());
        for (let i = 0; i < values.length; i++) {
            values[i] = this.int32();
        }
        return values;
    }

    string() {
        let content = '';
        const size = this.int32();
        let position = this._position;
        this.skip(size);
        for (let i = 0; i < size; i++) {
            content += String.fromCharCode(this._buffer[position++]);
        }
        return content;
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
        return barracuda.Metadata._metadata;
    }

    constructor() {
        this._types = new Map();
        const register = (id, name, category, inputs) => {
            this._types.set(id, { name: name, category: category, inputs: (inputs || []).map((input) => {
                return { name: input };
            }) });
        };
        register(0, 'Nop', '');
        register(1, 'Dense', 'Layer', [ 'input', 'kernel', 'bias' ]);
        register(2, 'MatMul', '', [ 'input', 'kernel', 'bias' ]);
        register(20, 'Conv2D', 'Layer', [ 'input', 'kernel', 'bias' ]);
        register(21, 'DepthwiseConv2D', 'Layer', [ 'input', 'kernel', 'bias' ]);
        register(22, 'Conv2DTrans', 'Layer', [ 'input', 'kernel', 'bias' ]);
        register(23, 'Upsample2D', 'Data');
        register(25, 'MaxPool2D', 'Pool');
        register(26, 'AvgPool2D', 'Pool');
        register(27, 'GlobalMaxPool2D', 'Pool');
        register(28, 'GlobalAvgPool2D', 'Pool');
        register(29, 'Border2D', '');
        register(30, 'Conv3D', 'Layer');
        register(32, 'Conv3DTrans', 'Layer');
        register(33, 'Upsample3D', 'Data');
        register(35, 'MaxPool3D', 'Pool');
        register(36, 'AvgPool3D', 'Pool');
        register(37, 'GlobalMaxPool3D', 'Pool');
        register(38, 'GlobalAvgPool3D', 'Pool');
        register(39, 'Border3D', '');
        register(50, 'Activation', '', [ 'input' ]);
        register(51, 'ScaleBias', 'Normalization', [ 'input', 'scale', 'bias' ]);
        register(52, 'Normalization', 'Normalization');
        register(53, 'LRN', 'Normalization');
        register(60, 'Dropout', 'Dropout');
        register(64, 'RandomNormal', '');
        register(65, 'RandomUniform', '');
        register(66, 'Multinomial', '');
        register(67, 'OneHot', '');
        register(68, 'TopKIndices', '');
        register(69, 'TopKValues', '');
        register(100, 'Add', '', [ 'inputs' ]);
        register(101, 'Sub', '', [ 'inputs' ]);
        register(102, 'Mul', '', [ 'inputs' ]);
        register(103, 'RealDiv', '', [ 'inputs' ]);
        register(104, 'Pow', '', [ 'inputs' ]);
        register(110, 'Minimum', '', [ 'inputs' ]);
        register(111, 'Maximum', '', [ 'inputs' ]);
        register(112, 'Mean', '', [ 'inputs' ]);
        register(120, 'ReduceL1', '', [ 'inputs' ]);
        register(121, 'ReduceL2', '', [ 'inputs' ]);
        register(122, 'ReduceLogSum', '', [ 'inputs' ]);
        register(123, 'ReduceLogSumExp', '', [ 'inputs' ]);
        register(124, 'ReduceMax', '', [ 'inputs' ]);
        register(125, 'ReduceMean', '', [ 'inputs' ]);
        register(126, 'ReduceMin', '', [ 'inputs' ]);
        register(127, 'ReduceProd', '', [ 'inputs' ]);
        register(128, 'ReduceSum', '', [ 'inputs' ]);
        register(129, 'ReduceSumSquare', '', [ 'inputs' ]);
        register(140, 'Greater', '');
        register(141, 'GreaterEqual', '');
        register(142, 'Less', '');
        register(143, 'LessEqual', '');
        register(144, 'Equal', '');
        register(145, 'LogicalOr', '');
        register(146, 'LogicalAnd', '');
        register(147, 'LogicalNot', '');
        register(148, 'LogicalXor', '');
        register(160, 'Pad2DReflect', '');
        register(161, 'Pad2DSymmetric', '');
        register(162, 'Pad2DEdge', '');
        register(200, 'Flatten', 'Shape');
        register(201, 'Reshape', 'Shape');
        register(202, 'Transpose', '');
        register(203, 'Squeeze', '');
        register(204, 'Unsqueeze', '');
        register(205, 'Gather', '');
        register(206, 'DepthToSpace', '');
        register(207, 'SpaceToDepth', '');
        register(208, 'Expand', '');
        register(209, 'Resample2D', '');
        register(210, 'Concat', 'Tensor', [ 'inputs' ]);
        register(211, 'StridedSlice', 'Shape');
        register(212, 'Tile', '');
        register(213, 'Shape', '');
        register(214, 'NonMaxSuppression', '');
        register(215, 'LSTM', '');
        register(255, 'Load', '');
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name: name.toString() });
        }
        return this._types.get(name);
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
