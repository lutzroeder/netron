
// Experimental

const barracuda = {};

barracuda.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream && stream.length > 12) {
            const buffer = stream.peek(12);
            if (buffer[0] <= 0x20 && buffer.subarray(1, 8).every((value) => value === 0x00)) {
                context.type = 'barracuda';
            }
        }
    }

    async open(context) {
        const metadata = barracuda.Metadata.open();
        const reader = context.read('binary');
        const model = new barracuda.NNModel(reader);
        return new barracuda.Model(metadata, model);
    }
};

barracuda.Model = class {

    constructor(metadata, model) {
        const version = model.version.toString();
        this.format = `Barracuda v${version}`;
        this.graphs = [new barracuda.Graph(metadata, model)];
    }
};

barracuda.Graph = class {

    constructor(metadata, model) {
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (!values.has(name)) {
                type = tensor ? tensor.type : type;
                values.set(name, new barracuda.Value(name, type, tensor));
            } else if (type || tensor) {
                throw new barracuda.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const layers = [];
        for (const layer of model.layers) {
            if (layer.type !== 255 || layer.inputs.length > 0) {
                layers.push(layer);
            } else {
                for (const tensor of layer.tensors) {
                    values.map(tensor.name, null, new barracuda.Tensor(tensor));
                }
            }
        }
        for (const input of model.inputs) {
            const shape = new barracuda.TensorShape(input.shape);
            const type = new barracuda.TensorType(4, shape);
            const argument = new barracuda.Argument(input.name, [values.map(input.name, type)]);
            this.inputs.push(argument);
        }
        for (const output of model.outputs) {
            const argument = new barracuda.Argument(output, [values.map(output)]);
            this.outputs.push(argument);
        }
        for (const layer of layers) {
            const node = new barracuda.Node(metadata, layer, null, values);
            this.nodes.push(node);
        }
    }
};

barracuda.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

barracuda.Value = class {

    constructor(name, type, initializer) {
        this.name = name;
        this.type = type || null;
        this.initializer = initializer || null;
    }
};

barracuda.Node = class {

    constructor(metadata, layer, type, values) {
        this.name = layer.name || '';
        this.type = type ? type : metadata.type(layer.type);
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const inputs = Array.prototype.slice.call(this.type.inputs || ['input']);
        if (this.type.inputs && this.type.inputs.length === 1 && this.type.inputs[0].name === 'inputs') {
            const argument = new barracuda.Argument('inputs', layer.inputs.map((input) => values.map(input)));
            this.inputs.push(argument);
        } else if (layer.inputs) {
            for (let i = 0; i < layer.inputs.length; i++) {
                const input = layer.inputs[i];
                const name = inputs.length > 0 ? inputs.shift().name : i.toString();
                const argument = new barracuda.Argument(name, [values.map(input)]);
                this.inputs.push(argument);
            }
        }
        if (layer.tensors) {
            for (let i = 0; i < layer.tensors.length; i++) {
                const tensor = layer.tensors[i];
                const initializer = new barracuda.Tensor(tensor);
                const name = inputs.length > 0 ? inputs.shift().name : i.toString();
                const argument = new barracuda.Argument(name, [values.map(tensor.name, initializer.type, initializer)]);
                this.inputs.push(argument);
            }
        }
        if (layer.inputs !== undefined) {
            const argument = new barracuda.Argument('output', [values.map(this.name)]);
            this.outputs.push(argument);
        }
        if (layer.activation !== undefined && (layer.type === 50 || layer.activation !== 0)) {
            const type = barracuda.Activation[layer.activation];
            if (!type) {
                throw new barracuda.Error(`Unsupported activation '${layer.activation}'.`);
            }
            const node = new barracuda.Node(metadata, {}, { name: type, category: 'Activation' }, values);
            this.chain = [node];
        }
        const attributes = [
            ['strides', 'int32[]', []],
            ['pads', 'int32[]', (value) => Array.isArray(value) && (value.every((v) => v === 0) || value.every((v) => v === -1))],
            ['pool_size', 'int32[]', []],
            ['alpha', 'float32', 1],
            ['beta', 'float32', 0],
            ['axis', 'int32', -1]
        ];
        for (const [name, type, defaultValue] of attributes) {
            const value = layer[name];
            if ((value === undefined) ||
                (Array.isArray(defaultValue) && Array.isArray(value) && value.length === defaultValue.length && value.every((v, i) => v === defaultValue[i])) ||
                (typeof defaultValue === 'function' && defaultValue(value)) ||
                (defaultValue === value)) {
                continue;
            }
            const attribute = new barracuda.Argument(name, value, type);
            this.attributes.push(attribute);
        }
    }
};

barracuda.Tensor = class {

    constructor(tensor) {
        this.type = new barracuda.TensorType(tensor.itemsize, new barracuda.TensorShape(tensor.shape));
        this.values = tensor.data;
    }
};

barracuda.TensorType = class {

    constructor(itemsize, shape) {
        switch (itemsize) {
            case 4: this.dataType = 'float32'; break;
            default: throw new barracuda.Error(`Unsupported data type size '${itemsize}'.`);
        }
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

barracuda.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`) : '';
    }
};

barracuda.NNModel = class {

    constructor(reader) {
        // https://github.com/Unity-Technologies/barracuda-release/blob/release/1.3.2/Barracuda/Runtime/Core/Model.cs
        reader = new barracuda.BinaryReader(reader);
        this.version = reader.int32();
        reader.int32();
        this.inputs = new Array(reader.int32());
        for (let i = 0; i < this.inputs.length; i++) {
            this.inputs[i] = {
                name: reader.string(),
                shape: reader.shape()
            };
        }
        this.outputs = reader.strings();
        this.memories = new Array(reader.int32());
        for (let i = 0; i < this.memories.length; i++) {
            this.memories[i] = {
                shape: reader.shape(),
                in: reader.string(),
                out: reader.string()
            };
        }
        this.layers = new Array(reader.int32());
        for (let i = 0; i < this.layers.length; i++) {
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
                    offset: reader.int64().toNumber(),
                    itemsize: reader.int32(),
                    length: reader.int32()
                });
            }
            this.layers[i] = layer;
        }
        const position = reader.position;
        for (const layer of this.layers) {
            for (const tensor of layer.tensors) {
                const offset = tensor.offset;
                reader.seek(position + (offset * tensor.itemsize));
                tensor.data = reader.read(tensor.length * tensor.itemsize);
            }
        }
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

barracuda.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get position() {
        return this._reader.position;
    }

    seek(position) {
        this._reader.seek(position);
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    read(length) {
        return this._reader.read(length);
    }

    byte() {
        return this._reader.byte();
    }

    int32() {
        return this._reader.int32();
    }

    int32s() {
        const values = new Array(this.int32());
        for (let i = 0; i < values.length; i++) {
            values[i] = this.int32();
        }
        return values;
    }

    int64() {
        return this._reader.int64();
    }

    float32() {
        return this._reader.float32();
    }

    string() {
        let content = '';
        const size = this.int32();
        for (let i = 0; i < size; i++) {
            const c = this.byte();
            content += String.fromCharCode(c);
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
            this._types.set(id, { name, category, inputs: (inputs || []).map((input) => {
                return { name: input };
            }) });
        };
        register(0, 'Nop', '');
        register(1, 'Dense', 'Layer', ['input', 'kernel', 'bias']);
        register(2, 'MatMul', '', ['input', 'kernel', 'bias']);
        register(20, 'Conv2D', 'Layer', ['input', 'kernel', 'bias']);
        register(21, 'DepthwiseConv2D', 'Layer', ['input', 'kernel', 'bias']);
        register(22, 'Conv2DTrans', 'Layer', ['input', 'kernel', 'bias']);
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
        register(50, 'Activation', '', ['input']);
        register(51, 'ScaleBias', 'Normalization', ['input', 'scale', 'bias']);
        register(52, 'Normalization', 'Normalization');
        register(53, 'LRN', 'Normalization');
        register(60, 'Dropout', 'Dropout');
        register(64, 'RandomNormal', '');
        register(65, 'RandomUniform', '');
        register(66, 'Multinomial', '');
        register(67, 'OneHot', '');
        register(68, 'TopKIndices', '');
        register(69, 'TopKValues', '');
        register(100, 'Add', '', ['inputs']);
        register(101, 'Sub', '', ['inputs']);
        register(102, 'Mul', '', ['inputs']);
        register(103, 'RealDiv', '', ['inputs']);
        register(104, 'Pow', '', ['inputs']);
        register(110, 'Minimum', '', ['inputs']);
        register(111, 'Maximum', '', ['inputs']);
        register(112, 'Mean', '', ['inputs']);
        register(120, 'ReduceL1', '', ['inputs']);
        register(121, 'ReduceL2', '', ['inputs']);
        register(122, 'ReduceLogSum', '', ['inputs']);
        register(123, 'ReduceLogSumExp', '', ['inputs']);
        register(124, 'ReduceMax', '', ['inputs']);
        register(125, 'ReduceMean', '', ['inputs']);
        register(126, 'ReduceMin', '', ['inputs']);
        register(127, 'ReduceProd', '', ['inputs']);
        register(128, 'ReduceSum', '', ['inputs']);
        register(129, 'ReduceSumSquare', '', ['inputs']);
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
        register(210, 'Concat', 'Tensor', ['inputs']);
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

export const ModelFactory = barracuda.ModelFactory;

