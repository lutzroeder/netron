
// Experimental
//
// https://github.com/katolikov/vknn

const vknn = {};

vknn.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 8) {
            const buffer = stream.peek(4);
            if (buffer[0] === 0x56 && buffer[1] === 0x58 && buffer[2] === 0x4d && buffer[3] === 0x31) {
                return context.set('vknn');
            }
        }
        return null;
    }

    async open(context) {
        const metadata = await context.metadata('vknn-metadata.json');
        const reader = await context.read('binary');
        const graph = new vknn.Reader(reader);
        return new vknn.Model(metadata, graph);
    }
};

vknn.Model = class {

    constructor(metadata, graph) {
        this.format = 'VKNN';
        this.modules = [new vknn.Graph(metadata, graph)];
    }
};

vknn.Graph = class {

    constructor(metadata, graph) {
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const tensors = graph.tensors;
        const values = new Array(tensors.length);
        const value = (id) => {
            id = Number(id);
            if (!Number.isInteger(id) || id < 0 || id >= tensors.length) {
                return null;
            }
            if (!values[id]) {
                const tensor = tensors[id];
                const name = tensor.name && tensor.name.length > 0 ? tensor.name : id.toString();
                const type = new vknn.TensorType(tensor.dtype, new vknn.TensorShape(tensor.shape), tensor.format);
                const data = graph.initializers.get(id);
                const initializer = data ? new vknn.Tensor(name, type, data) : null;
                values[id] = new vknn.Value(name, type, initializer);
            }
            return values[id];
        };
        for (const id of graph.inputs) {
            const argument = value(id);
            this.inputs.push(new vknn.Argument(argument ? argument.name : id.toString(), argument ? [argument] : []));
        }
        for (const id of graph.outputs) {
            const argument = value(id);
            this.outputs.push(new vknn.Argument(argument ? argument.name : id.toString(), argument ? [argument] : []));
        }
        for (const node of graph.nodes) {
            this.nodes.push(new vknn.Node(metadata, node, value));
        }
    }
};

vknn.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

vknn.Value = class {

    constructor(name, type = null, initializer = null) {
        if (typeof name !== 'string') {
            throw new vknn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type;
        this.initializer = initializer;
    }
};

vknn.Node = class {

    constructor(metadata, node, value) {
        let type = null;
        switch (node.type) {
            case 0: type = 'Unknown'; break;
            case 1: type = 'Conv'; break;
            case 2: type = 'Clip'; break;
            case 3: type = 'Relu'; break;
            case 4: type = 'Add'; break;
            case 5: type = 'GlobalAvgPool'; break;
            case 6: type = 'AvgPool'; break;
            case 7: type = 'MaxPool'; break;
            case 8: type = 'Gemm'; break;
            case 9: type = 'MatMul'; break;
            case 10: type = 'Einsum'; break;
            case 11: type = 'Reshape'; break;
            case 12: type = 'Expand'; break;
            case 13: type = 'Tile'; break;
            case 14: type = 'Squeeze'; break;
            case 15: type = 'Flatten'; break;
            case 16: type = 'Softmax'; break;
            case 17: type = 'LayerNorm'; break;
            case 18: type = 'BatchNorm'; break;
            case 19: type = 'Concat'; break;
            case 20: type = 'Pad'; break;
            case 21: type = 'Identity'; break;
            case 22: type = 'Constant'; break;
            case 23: type = 'Shape'; break;
            case 24: type = 'Gather'; break;
            case 25: type = 'Unsqueeze'; break;
            case 26: type = 'Unary'; break;
            case 27: type = 'Binary'; break;
            case 28: type = 'PRelu'; break;
            case 29: type = 'Resize'; break;
            case 30: type = 'GridSample'; break;
            case 31: type = 'Transpose'; break;
            case 32: type = 'Slice'; break;
            case 33: type = 'Reduce'; break;
            case 34: type = 'DepthToSpace'; break;
            case 35: type = 'Cast'; break;
            case 36: type = 'Split'; break;
            case 37: type = 'Where'; break;
            case 38: type = 'Equal'; break;
            case 39: type = 'ConstantOfShape'; break;
            case 40: type = 'EyeLike'; break;
            case 41: type = 'ScatterND'; break;
            case 42: type = 'FusedSE'; break;
            case 43: type = 'FusedDwPw'; break;
            case 44: type = 'ConvertLayout'; break;
            default: type = node.type.toString(); break;
        }
        this.type = metadata.type(type);
        this.name = node.name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        // A fused residual tensor is kept in node.inputs by the graph passes (for liveness/scheduling),
        // but it is surfaced separately below as a named 'residual' input, so skip it here.
        const residual = Number(node.fusedResidual);
        const hasResidual = Number.isInteger(residual) && residual >= 0;
        const inputNames = Array.isArray(this.type.inputs) ? this.type.inputs.map((input) => input.name) : [];
        for (let i = 0; i < node.inputs.length; i++) {
            if (hasResidual && node.inputs[i] === residual) {
                continue;
            }
            const argument = value(node.inputs[i]);
            const name = i < inputNames.length ? inputNames[i] : i.toString();
            this.inputs.push(new vknn.Argument(name, argument ? [argument] : []));
        }
        const outputNames = Array.isArray(this.type.outputs) ? this.type.outputs.map((output) => output.name) : [];
        for (let i = 0; i < node.outputs.length; i++) {
            const argument = value(node.outputs[i]);
            const name = i < outputNames.length ? outputNames[i] : i.toString();
            this.outputs.push(new vknn.Argument(name, argument ? [argument] : []));
        }
        for (const attr of node.attributes) {
            let content = null;
            let kind = null;
            switch (attr.kind) {
                case 1: content = attr.i; kind = 'int64'; break;
                case 2: content = attr.f; kind = 'float32'; break;
                case 3: content = attr.ints; kind = 'int64[]'; break;
                case 4: content = attr.floats; kind = 'float32[]'; break;
                case 5: content = attr.str; kind = 'string'; break;
                default: continue;
            }
            this.attributes.push(new vknn.Argument(attr.name, content, kind));
        }
        if (node.fusedAct !== 0) {
            const activations = ['None', 'Relu', 'Relu6', 'Clip', 'HardSwish', 'SiLU'];
            const activation = node.fusedAct < activations.length ? activations[node.fusedAct] : node.fusedAct.toString();
            this.attributes.push(new vknn.Argument('fused_activation', activation, 'string'));
            if (node.actLo !== 0 || node.actHi !== 0) {
                this.attributes.push(new vknn.Argument('activation_min', node.actLo, 'float32'));
                this.attributes.push(new vknn.Argument('activation_max', node.actHi, 'float32'));
            }
        }
        let operation = null;
        switch (type) {
            case 'Unary': operation = ['Sigmoid', 'Tanh', 'HardSwish', 'HardSigmoid', 'LeakyRelu', 'Elu', 'Abs', 'Neg', 'Exp', 'Log', 'Sqrt', 'Floor', 'Ceil', 'Relu', 'SiLU', 'Erf', 'Cos', 'Sin', 'Reciprocal', 'Softplus'][node.subOp]; break;
            case 'Binary': operation = ['Mul', 'Sub', 'Div', 'Max', 'Min', 'Pow', 'Add'][node.subOp]; break;
            case 'Reduce': operation = ['Mean', 'Sum', 'Max', 'Min', 'Prod', 'L2'][node.subOp]; break;
            default: break;
        }
        if (operation) {
            this.attributes.push(new vknn.Argument('operation', operation, 'string'));
        }
        if (hasResidual) {
            const argument = value(residual);
            if (argument) {
                this.inputs.push(new vknn.Argument('residual', [argument]));
            }
        }
    }
};

vknn.TensorType = class {

    constructor(dtype, shape, format) {
        switch (dtype) {
            case 0: this.dataType = 'float32'; break;
            case 1: this.dataType = 'float16'; break;
            case 2: this.dataType = 'int32'; break;
            case 3: this.dataType = 'int8'; break;
            case 4: this.dataType = 'uint8'; break;
            case 5: this.dataType = 'int64'; break;
            default: this.dataType = '?'; break;
        }
        this.shape = shape;
        // The IR is canonically NCHW, so only annotate a boundary tensor with a non-default layout.
        switch (format) {
            case 1: this.denotation = 'NHWC'; break;
            case 2: this.denotation = 'NC4HW4'; break;
            case 3: this.denotation = 'Auto'; break;
            case 255: this.denotation = 'Unknown'; break;
            default: break;
        }
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

vknn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension === null || dimension === undefined ? '?' : dimension.toString()).join(',')}]`;
    }
};

vknn.Tensor = class {

    constructor(name, type, data) {
        this.name = name;
        this.type = type;
        this.values = data;
        this.encoding = '<';
    }
};

vknn.Reader = class {

    constructor(reader) {
        const magic = reader.uint32();
        if (magic !== 0x314d5856) {
            throw new vknn.Error(`Invalid magic number '0x${magic.toString(16)}'.`);
        }
        const int32s = () => {
            const length = reader.uint32();
            const values = new Array(length);
            for (let i = 0; i < length; i++) {
                values[i] = reader.int32();
            }
            return values;
        };
        const int64s = () => {
            const length = reader.uint32();
            const values = new Array(length);
            for (let i = 0; i < length; i++) {
                values[i] = reader.int64().toNumber();
            }
            return values;
        };
        const float32s = () => {
            const length = reader.uint32();
            const values = new Array(length);
            for (let i = 0; i < length; i++) {
                values[i] = reader.float32();
            }
            return values;
        };
        const bytes = () => {
            const length = reader.uint32();
            return reader.read(length);
        };
        const tensorCount = reader.uint32();
        this.tensors = new Array(tensorCount);
        for (let i = 0; i < tensorCount; i++) {
            const name = reader.string();
            const shape = int64s();
            const dtype = reader.uint32();
            const format = reader.uint32();
            const flags = reader.uint32();
            this.tensors[i] = {
                name, shape, dtype, format,
                isInput: (flags & 1) !== 0,
                isOutput: (flags & 2) !== 0,
                isInitializer: (flags & 4) !== 0
            };
        }
        const nodeCount = reader.uint32();
        this.nodes = new Array(nodeCount);
        for (let i = 0; i < nodeCount; i++) {
            const type = reader.uint32();
            const name = reader.string();
            const inputs = int32s();
            const outputs = int32s();
            const fusedAct = reader.uint32();
            const actLo = reader.float32();
            const actHi = reader.float32();
            const subOp = reader.int64().toNumber();
            const fusedResidual = reader.int64().toNumber();
            const attributeCount = reader.uint32();
            const attributes = new Array(attributeCount);
            for (let j = 0; j < attributeCount; j++) {
                const name = reader.string();
                const kind = reader.uint32();
                const i = reader.int64().toNumber();
                const f = reader.float32();
                const ints = int64s();
                const floats = float32s();
                const str = reader.string();
                attributes[j] = { name, kind, i, f, ints, floats, str };
            }
            this.nodes[i] = { type, name, inputs, outputs, fusedAct, actLo, actHi, subOp, fusedResidual, attributes };
        }
        this.inputs = int32s();
        this.outputs = int32s();
        const initializerCount = reader.uint32();
        this.initializers = new Map();
        for (let i = 0; i < initializerCount; i++) {
            const id = reader.int64().toNumber();
            this.initializers.set(id, bytes());
        }
    }
};

vknn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading VKNN model.';
    }
};

export const ModelFactory = vknn.ModelFactory;
