
// Experimental
//
// VKNN (Vulkan Neural Network) compiled model — ".vxm".
// VKNN is a Vulkan-compute inference engine: https://github.com/katolikov/vknn
// The ".vxm" container is defined by vknn::saveGraphBin:
// https://github.com/katolikov/vknn/blob/main/src/core/model_io.cpp
// A compact, self-contained binary: an optimized NCHW graph (post-import, post-passes) with embedded
// weights, so a reload skips both ONNX parsing and all graph passes. Little-endian throughout. Layout:
//   u32  magic 'VXM1' (0x314d5856)
//   u32  tensorCount;  per tensor: str name, vec<i64> shape, u32 dtype, u32 format, u32 flags
//   u32  nodeCount;    per node:   u32 type, str name, vec<i32> inputs, vec<i32> outputs,
//                                  u32 fusedAct, f32 actLo, f32 actHi, i64 subOp, i64 fusedResidual,
//                                  u32 attrCount; per attr: str key, u32 kind, i64 i, f32 f,
//                                                           vec<i64> ints, vec<f32> floats, str str
//   vec<i32> graph inputs; vec<i32> graph outputs
//   u32  initializerCount; per initializer: i64 tensorId, vec<u8> bytes
// where str = u32 length + bytes, vec<T> = u32 count + count*sizeof(T) elements.

const vxm = {};

vxm.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 8) {
            const buffer = stream.peek(4);
            if (buffer[0] === 0x56 && buffer[1] === 0x58 && buffer[2] === 0x4d && buffer[3] === 0x31) {
                return context.set('vxm');
            }
        }
        return null;
    }

    async open(context) {
        const metadata = await context.metadata('vxm-metadata.json');
        const reader = await context.read('binary');
        const graph = new vxm.Reader(reader);
        return new vxm.Model(metadata, graph);
    }
};

vxm.Model = class {

    constructor(metadata, graph) {
        this.format = 'VKNN';
        this.modules = [new vxm.Graph(metadata, graph)];
    }
};

vxm.Graph = class {

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
                const type = new vxm.TensorType(tensor.dtype, new vxm.TensorShape(tensor.shape), tensor.format);
                const data = graph.initializers.get(id);
                const initializer = data ? new vxm.Tensor(name, type, data) : null;
                values[id] = new vxm.Value(name, type, initializer);
            }
            return values[id];
        };
        for (const id of graph.inputs) {
            const argument = value(id);
            this.inputs.push(new vxm.Argument(argument ? argument.name : id.toString(), argument ? [argument] : []));
        }
        for (const id of graph.outputs) {
            const argument = value(id);
            this.outputs.push(new vxm.Argument(argument ? argument.name : id.toString(), argument ? [argument] : []));
        }
        for (const node of graph.nodes) {
            this.nodes.push(new vxm.Node(metadata, node, value));
        }
    }
};

vxm.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

vxm.Value = class {

    constructor(name, type = null, initializer = null) {
        if (typeof name !== 'string') {
            throw new vxm.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type;
        this.initializer = initializer;
    }
};

vxm.Node = class {

    constructor(metadata, node, value) {
        const type = node.type < vxm.OpType.length ? vxm.OpType[node.type] : node.type.toString();
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
            this.inputs.push(new vxm.Argument(name, argument ? [argument] : []));
        }
        const outputNames = Array.isArray(this.type.outputs) ? this.type.outputs.map((output) => output.name) : [];
        for (let i = 0; i < node.outputs.length; i++) {
            const argument = value(node.outputs[i]);
            const name = i < outputNames.length ? outputNames[i] : i.toString();
            this.outputs.push(new vxm.Argument(name, argument ? [argument] : []));
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
            this.attributes.push(new vxm.Argument(attr.name, content, kind));
        }
        if (node.fusedAct !== 0) {
            const activations = ['None', 'Relu', 'Relu6', 'Clip', 'HardSwish', 'SiLU'];
            const activation = node.fusedAct < activations.length ? activations[node.fusedAct] : node.fusedAct.toString();
            this.attributes.push(new vxm.Argument('fused_activation', activation, 'string'));
            if (node.actLo !== 0 || node.actHi !== 0) {
                this.attributes.push(new vxm.Argument('activation_min', node.actLo, 'float32'));
                this.attributes.push(new vxm.Argument('activation_max', node.actHi, 'float32'));
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
            this.attributes.push(new vxm.Argument('operation', operation, 'string'));
        }
        if (hasResidual) {
            const argument = value(residual);
            if (argument) {
                this.inputs.push(new vxm.Argument('residual', [argument]));
            }
        }
    }
};

vxm.TensorType = class {

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

vxm.TensorShape = class {

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

vxm.Tensor = class {

    constructor(name, type, data) {
        this.name = name;
        this.type = type;
        this.values = data;
        this.encoding = '<';
    }
};

vxm.Reader = class {

    constructor(reader) {
        const magic = reader.uint32();
        if (magic !== 0x314d5856) {
            throw new vxm.Error(`Invalid magic number '0x${magic.toString(16)}'.`);
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

// Op-type names by enum value (vknn::OpType in include/vknn/op.h).
vxm.OpType = [
    'Unknown', 'Conv', 'Clip', 'Relu', 'Add', 'GlobalAvgPool', 'AvgPool', 'MaxPool', 'Gemm', 'MatMul',
    'Einsum', 'Reshape', 'Expand', 'Tile', 'Squeeze', 'Flatten', 'Softmax', 'LayerNorm', 'BatchNorm', 'Concat',
    'Pad', 'Identity', 'Constant', 'Shape', 'Gather', 'Unsqueeze', 'Unary', 'Binary', 'PRelu', 'Resize',
    'GridSample', 'Transpose', 'Slice', 'Reduce', 'DepthToSpace', 'Cast', 'Split', 'Where', 'Equal',
    'ConstantOfShape', 'EyeLike', 'ScatterND', 'FusedSE', 'FusedDwPw', 'ConvertLayout'
];

vxm.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading VKNN model.';
    }
};

export const ModelFactory = vxm.ModelFactory;
