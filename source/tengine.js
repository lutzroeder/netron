
// Experimental

const tengine = {};

tengine.ModelFactory = class {

    match(context) {
        const reader = tengine.Reader.open(context);
        if (reader) {
            context.type = 'tengine';
            context.target = reader;
        }
    }

    async open(context) {
        const metadata = await tengine.Metadata.open(context);
        const reader = context.target;
        reader.read();
        return new tengine.Model(metadata, reader);
    }
};

tengine.Model = class {

    constructor(metadata, reader) {
        this.format = `Tengine v${reader.version}`;
        this.source = reader.source;
        this.graphs = reader.graphs.map((graph) => new tengine.Graph(metadata, graph));
    }
};

tengine.Graph = class {

    constructor(metadata, graph) {
        this.name = graph.id.toString();
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const tensors = graph.tensors.map((tensor) => new tengine.Value(tensor));
        for (const input of graph.inputs) {
            const node = graph.nodes[input];
            this.inputs.push(new tengine.Argument(node.name, node.outputs.map((output) => tensors[output])));
        }
        for (const output of graph.outputs) {
            const node = graph.nodes[output];
            this.outputs.push(new tengine.Argument(node.name, node.outputs.map((output) => tensors[output])));
        }
        for (const node of graph.nodes) {
            switch (node.type) {
                case 'INPUT':
                case 'Const':
                    break;
                default:
                    this.nodes.push(new tengine.Node(metadata, node, tensors));
                    break;
            }
        }
    }
};

tengine.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

tengine.Value = class {

    constructor(tensor) {
        this.name = tensor.name;
        this.type = new tengine.TensorType(tensor.dataType, new tengine.TensorShape(tensor.dims));
        this.initializer = (tensor.type === 2) ? new tengine.Tensor(this.type, tensor.buffer) : null;
    }
};

tengine.Node = class {

    constructor(metadata, node, tensors) {
        this.name = node.name;
        const type = node.type;
        const version = node.version;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.type = metadata.type(type, version) || { name: type };
        for (let i = 0; i < node.params.length; i++) {
            const metadata = (this.type && this.type.attributes && i < this.type.attributes.length) ? this.type.attributes[i] : null;
            const value = node.params[i];
            let name = metadata ? metadata.name : i.toString();
            let type = null;
            let visible = true;
            if (metadata) {
                name = !name && metadata.name ? metadata.name : name;
                type = !type && metadata.type ? metadata.type : type;
                if (metadata.visible === false) {
                    visible = false;
                } else if (metadata.default !== undefined) {
                    if (value === metadata.default || (value && value.toString() === metadata.default.toString())) {
                        visible = false;
                    }
                }
            }
            const attribute = new tengine.Argument(name, value, type, visible);
            this.attributes.push(attribute);
        }
        const inputs = node.inputs;
        let inputIndex = 0;
        if (this.type && this.type.inputs) {
            for (const inputDef of this.type.inputs) {
                if (inputIndex < inputs.length || inputDef.option !== 'optional') {
                    const inputCount = (inputDef.option === 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id !== '' || inputDef.option !== 'optional').map((id) => tensors[id]);
                    this.inputs.push(new tengine.Argument(inputDef.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        } else {
            this.inputs.push(...inputs.slice(inputIndex).map((id, index) => {
                const inputName = ((inputIndex + index) === 0) ? 'input' : (inputIndex + index).toString();
                return new tengine.Argument(inputName, [tensors[id]]);
            }));
        }
        const outputs = node.outputs;
        let outputIndex = 0;
        if (this.type && this.type.outputs) {
            for (const outputDef of this.type.outputs) {
                if (outputIndex < outputs.length || outputDef.option !== 'optional') {
                    const outputCount = (outputDef.option === 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => tensors[id]);
                    this.outputs.push(new tengine.Argument(outputDef.name, outputArguments));
                    outputIndex += outputCount;
                }
            }
        } else {
            this.outputs.push(...outputs.slice(outputIndex).map((id, index) => {
                const outputName = ((outputIndex + index) === 0) ? 'output' : (outputIndex + index).toString();
                return new tengine.Argument(outputName, [tensors[id]]);
            }));
        }
    }
};

tengine.Tensor = class {

    constructor(type, values) {
        this.type = type;
        this.values = values;
    }
};

tengine.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case 0: this.dataType = 'float32'; break;
            case 1: this.dataType = 'float16'; break;
            case 2: this.dataType = 'int8'; break;
            case 3: this.dataType = 'uint8'; break;
            case 4: this.dataType = 'int32'; break;
            case 5: this.dataType = 'int16'; break;
            default: throw new tengine.Error(`Unsupported data type '${dataType}'.`);
        }
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

tengine.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`) : '';
    }
};

tengine.Metadata = class {

    static async open(context) {
        if (!tengine.Metadata._metadata) {
            let data = null;
            try {
                data = await context.request('tengine-metadata.json');
            } catch {
                // continue regardless of error
            }
            tengine.Metadata._metadata = new tengine.Metadata(data);
        }
        return tengine.Metadata._metadata;
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            for (const item of metadata) {
                if (item.name) {
                    const version = item.version || 0;
                    const name = `${item.name}:${version}`;
                    this._map.set(name, item);
                }
            }
        }
    }

    type(name, version) {
        let current = version;
        while (current > 0) {
            if (this._map.has(`${name}:${current}`)) {
                break;
            }
            current--;
        }
        if (current >= 0) {
            const schema = this._map.get(`${name}:${current}`);
            if (current !== version) {
                this._map.set(`${name}:${version}`, schema);
            }
            return schema;
        }
        return null;
    }
};

tengine.Reader = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length > 12) {
            const buffer = stream.peek(4);
            if (buffer[0] < 4 && buffer[1] === 0 && buffer[3] === 0) {
                return new tengine.Reader(context);
            }
        }
        return null;
    }

    constructor(context) {
        this.context = context;
        // https://github.com/OAID/Tengine/wiki/The-format-of-tmfile
        // https://github.com/OAID/Tengine/blob/tengine-lite/source/serializer/tmfile/tm2_format.h
    }

    read() {
        const types = new Map();
        const register = (index, version, name, params) => {
            types.set(`${index}:${version}`, { name, params });
        };
        const operator = (index, version) => {
            let current = version;
            while (current >= 0) {
                if (types.has(`${index}:${current}`)) {
                    break;
                }
                current--;
            }
            if (current >= 0) {
                const schema = types.get(`${index}:${current}`);
                if (current !== version) {
                    types.set(`${index}:${version}`, schema);
                }
                return schema;
            }
            return null;
        };
        register(0, 0, 'Accuracy', []);
        register(1, 0, 'BatchNormalization', ['f', 'f', 'i']);
        register(2, 0, 'BilinearResize', ['f', 'f', 'i']);
        register(3, 0, 'Concat', ['i']);
        register(4, 0, 'Const', []);
        register(5, 0, 'Convolution', ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(6, 0, 'Deconvolution', ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(7, 0, 'DetectionOutput', ['i', 'i', 'i', 'f', 'f']);
        register(8, 0, 'DropOut', []);
        register(9, 0, 'Eltwise', ['i', 'i']);
        register(10, 0, 'Flatten', ['i']);
        register(11, 0, 'FullyConnected', ['i']);
        register(12, 0, 'INPUT', []);
        register(13, 0, 'LRN', ['i', 'f', 'f', 'i', 'f']);
        register(14, 0, 'Normalize', ['i', 'i']);
        register(15, 0, 'Permute', ['i', 'i', 'i', 'i', 'i']);
        register(16, 0, 'Pooling', ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(17, 0, 'Prelu', []);
        register(18, 0, 'PriorBox', ['f[]', 'f[]', 'f[]', 'f[]', 'i', 'i', 'i', 'i', 'i', 'f', 'f', 'f', 'i', 'i']);
        register(19, 0, 'Region', ['i', 'i', 'i', 'i', 'f', 'f', 'f[]']);
        register(20, 0, 'ReLU', ['f']);
        register(21, 0, 'ReLU6', []);
        register(22, 0, 'Reorg', ['i']);
        register(23, 0, 'Reshape', ['i', 'i', 'i', 'i', 'i', 'i']);
        // register(23, 0, 'Reshape', [ 'i', 'i', 'i[]' ]);
        register(24, 0, 'RoiPooling', ['i', 'i', 'f']);
        register(25, 0, 'RPN', ['f[]', 'f[]', 'i', 'i', 'i', 'i', 'i', 'f', 'anchors']);
        register(26, 0, 'Scale', ['i', 'i', 'i']);
        register(27, 0, 'Slice', ['i', 'i[]', 'i[]', 'i[]', 'i', 'i', 'i', 'i', 'i']);
        register(28, 0, 'SoftMax', ['i']);
        register(29, 0, 'Split', ['i', 'i', 'boolean', 'boolean', 'i[]']);
        register(30, 0, 'DetectionPostProcess', ['i', 'i', 'f', 'f', 'i', 'f[]']);
        register(31, 0, 'Gemm', ['f', 'f', 'i', 'i']);
        register(32, 0, 'Generic', ['i', 'i', 'string']);
        register(33, 0, 'Logistic', []);
        register(34, 0, 'LSTM', ['f', 'f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(35, 0, 'RNN', ['f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(36, 0, 'TanH', []);
        register(37, 0, 'Sigmoid', []);
        register(38, 0, 'Squeeze', ['i', 'i', 'i', 'i']);
        register(39, 0, 'FusedbnScaleRelu', []);
        register(40, 0, 'Pad', ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'f']);
        register(41, 0, 'StridedSlice', ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(42, 0, 'ArgMax', ['i']);
        register(43, 0, 'ArgMin', ['i']);
        register(44, 0, 'TopKV2', ['i', 'i']);
        register(45, 0, 'Reduction', ['i', 'i', 'i', 'i', 'i', 'i']);
        register(46, 0, 'Max', []);
        register(47, 0, 'Min', []);
        register(48, 0, 'GRU', ['f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']);
        register(49, 0, 'Addn', 'i');
        register(50, 0, 'SwapAxis', ['i', 'i']);
        register(51, 0, 'Upsample', ['f']);
        register(52, 0, 'SpaceToBatchND', ['i', 'i', 'i', 'i', 'i', 'i']);
        register(53, 0, 'BatchToSpaceND', ['i', 'i', 'i', 'i', 'i', 'i']);
        register(54, 0, 'Resize', ['f', 'f', 'i']);
        register(55, 0, 'ShuffleChannel', ['i']);
        register(56, 0, 'Crop', ['i', 'i', 'i', 'i', 'i', 'i', 'boolean', 'i', 'i']);
        register(57, 0, 'ROIAlign', ['i', 'i', 'f']);
        register(58, 0, 'Psroipooling', ['i', 'i', 'f', 'i']);
        register(59, 0, 'Unary', ['i']);
        register(60, 0, 'Expanddims', ['i']);
        register(61, 0, 'Bias', ['i']);
        register(62, 0, 'Noop', []);
        register(63, 0, 'Threshold', ['f']);
        register(64, 0, 'Hardsigmoid', ['f', 'f']);
        register(65, 0, 'Embed', ['f', 'f', 'f', 'f']);
        register(66, 0, 'InstanceNorm', ['f']);
        register(67, 0, 'MVN', ['i', 'i', 'f']);
        register(68, 0, 'Absval', []);
        register(69, 0, 'Cast', ['i', 'i']);
        register(70, 0, 'HardSwish', ['f', 'f']);
        register(71, 0, 'Interp', ['i', 'f', 'f', 'i', 'i']);
        register(72, 0, 'SELU', ['f', 'f']);
        register(73, 0, 'ELU', ['f']);
        register(74, 0, 'BroadMul', []);
        register(75, 0, 'Logical', ['i']);
        register(76, 0, 'Gather', ['i', 'i']);
        register(77, 0, 'Transpose', ['i[]']);
        register(78, 0, 'Comparison', ['i']);
        register(79, 0, 'SpaceToDepth', ['i']);
        register(80, 0, 'DepthToSpace', ['i']);
        register(81, 0, 'Reverse', []);
        register(82, 0, 'SparseToDense', ['i','i','i']);
        register(83, 0, 'Ceil', []);
        register(84, 0, 'SquaredDifference', []);
        register(85, 0, 'Round', []);
        register(86, 0, 'ZerosLike', []);
        register(87, 0, 'Clip', ['f','f']);
        register(88, 0, 'Unsqueeze', ['i[]']);
        register(89, 0, 'ReduceL2', ['i','i']);
        register(90, 0, 'Mean', []);
        register(91, 0, 'MatMul', []);
        register(92, 0, 'Expand', ['i[]']);
        register(93, 0, 'Scatter', ['i','boolean']);
        register(94, 0, 'Shape', []);
        register(95, 0, 'Where', []);
        register(96, 0, 'Tile', ['i','i']);
        register(97, 0, 'Mish', []);
        register(98, 0, 'L2Pool', []);
        register(99, 0, 'LogSoftmax', []);
        register(100, 0, 'ReLU1', []);
        register(101, 0, 'L2Normalization', []);
        register(102, 0, 'PackModel', ['i','i']);
        register(103, 0, 'Num', []);
        const reader = new tengine.BinaryReader(this.context.read('binary'));
        const major = reader.uint16();
        const minor = reader.uint16();
        if (major !== 2) {
            throw new tengine.Error(`Unsupported format version 'v${this.version}'.`);
        }
        this.version = `${major}.${minor}`;
        reader.uint16(); // compileVersion
        reader.skip(2); // struct align
        reader.seek(reader.uint32()); // root table
        const originalFormat = reader.int32();
        const subFormat = reader.int32();
        const sources = [
            '', 'Tengine', 'Caffe', 'ONNX',
            'MXNet', 'TensorFlow', 'TensorFlow Lite', 'Darknet',
            `DLA v${subFormat}`, 'ncnn', 'MegEngine', 'OneFlow',
            'Horizon', 'Bitman'
        ];
        if (originalFormat >= sources.length) {
            throw new tengine.Error(`Unsupported source '${originalFormat}'.`);
        }
        this.source = sources[originalFormat];
        this.graphs = [];
        const subgraphOffsets = reader.uint32s();
        for (const subgraphOffset of subgraphOffsets) {
            reader.seek(subgraphOffset);
            const subgraph = {};
            subgraph.id = reader.int32();
            subgraph.graphLayout = reader.int32();
            /*
            if (graphLayout === 0) {
                return "NCHW";
            }
            if (graphLayout === 1) {
                return "NHWC";
            }
            */
            subgraph.originalLayout = reader.int32();
            subgraph.inputs = reader.uint32s();
            subgraph.outputs = reader.uint32s();
            const nodeOffsets = reader.uint32s();
            const tensorOffsets = reader.uint32s();
            const bufferOffsets = reader.uint32s();
            subgraph.name = reader.string();
            subgraph.nodes = [];
            subgraph.tensors = [];
            this.graphs.push(subgraph);
            // nodes
            for (const nodeOffset of nodeOffsets) {
                reader.seek(nodeOffset);
                const node = {};
                node.id = reader.int32();
                node.inputs = reader.uint32s();
                node.outputs = reader.uint32s();
                const typeOffset = reader.int32();
                node.name = reader.string();
                const attributeOffsets = reader.uint32s();
                node.dynamicShape = reader.boolean();
                reader.seek(typeOffset);
                node.version = reader.int32();
                const index = reader.int32();
                const paramsOffset = reader.uint32();
                const schema = operator(index, node.version);
                node.type = schema ? schema.name : index.toString();
                const paramTypes = schema ? schema.params : [];
                node.params = [];
                if (paramsOffset) {
                    reader.seek(paramsOffset);
                    for (const paramType of paramTypes) {
                        if (paramType !== 'boolean') {
                            reader.align(4);
                        }
                        switch (paramType) {
                            case 'i':
                                node.params.push(reader.int32());
                                break;
                            case 'f':
                                node.params.push(reader.float32());
                                break;
                            case 'i[]':
                                node.params.push(reader.int32s());
                                break;
                            case 'f[]':
                                node.params.push(reader.float32s());
                                break;
                            case 'boolean':
                                node.params.push(reader.boolean());
                                break;
                            case 'string':
                                node.params.push(reader.string());
                                break;
                            case 'anchors':
                                node.params.push(reader.anchors(4));
                                break;
                            default:
                                throw new tengine.Error(`Unsupported param type '${paramType}' in '${node.type}'.`);
                        }
                    }
                }
                if (node.type === 'Slice') {
                    node.params[6] = (originalFormat === 5) ? node.params[6] : 0;
                }
                node.attributes = attributeOffsets.map((attributeOffset) => {
                    reader.seek(attributeOffset);
                    const name = reader.string();
                    const value = reader.string();
                    const type = reader.int32();
                    return { name, value, type };
                });
                subgraph.nodes.push(node);
            }
            // buffers
            const buffers = bufferOffsets.map((bufferOffset) => {
                reader.seek(bufferOffset);
                const size = reader.uint32();
                const offset = reader.int32();
                if (offset !== 0) {
                    reader.seek(offset);
                    return reader.read(size);
                }
                return null;
            });
            // tensors
            subgraph.tensors = tensorOffsets.map((tensorOffset) => {
                reader.seek(tensorOffset);
                const tensor = {};
                tensor.id = reader.int32();
                tensor.buffer = buffers[reader.int32()];
                tensor.dims = reader.int32s();
                tensor.name = reader.string();
                const quantparamsOffset = reader.int32();
                tensor.layout = reader.int32();
                tensor.type = reader.int32(); // ar = 1, const = 2, input = 3, vdep, unknown
                tensor.dataType = reader.int32();
                if (quantparamsOffset) {
                    reader.seek(quantparamsOffset);
                    tensor.quantparams = {
                        zeroPoint: reader.int32(),
                        scale: reader.float32(),
                        width: reader.int32()
                    };
                }
                return tensor;
            });
            for (const node of subgraph.nodes) {
                if (node.type === 'Convolution') {
                    switch (subgraph.graphLayout) {
                        case 0: // NCHW
                            /* eslint-disable prefer-destructuring */
                            node.params[6] = subgraph.tensors[node.inputs[1]].dims[1];
                            /* eslint-enable prefer-destructuring */
                            break;
                        case 1: // NHWC
                            /* eslint-disable prefer-destructuring */
                            node.params[6] = subgraph.tensors[node.inputs[1]].dims[3];
                            /* eslint-enable prefer-destructuring */
                            break;
                        default:
                            throw new tengine.Error(`Unsupported 'Convolution' layout '${subgraph.graphLayout}'.`);
                    }
                }
            }
        }
        delete this.stream;
    }
};

tengine.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get position() {
        return this._reader.position;
    }

    seek(offset) {
        this._reader.seek(offset);
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    align(mod) {
        return this._reader.align(mod);
    }

    read(length) {
        return this._reader.read(length);
    }

    boolean() {
        return this._reader.boolean();
    }

    byte() {
        return this._reader.byte();
    }

    int32() {
        return this._reader.int32();
    }

    int32s() {
        const values = [];
        const offset = this.uint32();
        if (offset) {
            const next = this.position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                values.push(this.int32());
            }
            this.seek(next);
        }
        return values;
    }

    uint16() {
        return this._reader.uint16();
    }

    uint32() {
        return this._reader.uint32();
    }

    uint32s() {
        const values = [];
        const offset = this.uint32();
        if (offset) {
            const next = this.position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                values.push(this.uint32());
            }
            this.seek(next);
        }
        return values;
    }

    float32() {
        return this._reader.float32();
    }

    float32s() {
        const values = [];
        const offset = this.uint32();
        if (offset) {
            const next = this.position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                values.push(this.float32());
            }
            this.seek(next);
        }
        return values;
    }

    string() {
        const position = this.uint32();
        let content = '';
        if (position) {
            const next = this.position;
            this.seek(position);
            const size = this.uint32();
            this.seek(this.uint32());
            for (let i = 0; i < size - 1; i++) {
                content += String.fromCharCode(this.byte());
            }
            this.seek(next);
        }
        return content;
    }

    anchors(length) {
        const arrays = [];
        const offset = this.uint32();
        if (offset) {
            const next = this._position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                const array = [];
                for (let j = 0; j < length; j++) {
                    array.push(this.float32());
                }
                arrays.push(array);
            }
            this.seek(next);
        }
        return arrays;
    }
};

tengine.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Tengine model.';
    }
};

export const ModelFactory = tengine.ModelFactory;
