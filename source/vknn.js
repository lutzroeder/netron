
// Experimental
//
// https://github.com/katolikov/vknn
//
// Reads the vknn ".vxm" container: the engine's post-optimization graph serialization, whose byte
// layout is defined by src/core/model_io.cpp in github.com/katolikov/vknn. The body is a custom
// fixed-width little-endian binary (fwrite of pod/vec/str) -- NOT msgpack -- read here field for
// field in the engine's exact write order.
//
// The parser is deliberately topology-first so that most future format bumps cost nothing here:
//   * Dispatch is by container KIND, not a hardcoded magic. A base body is either one graph (VXM3)
//     or several shape buckets (VXM4); a quantized wrapper (magic >= VXM5) is a [subtag: 3|4][base
//     body], so a future wrapper magic that keeps that shape unwraps and reads with no new code.
//     Each bucket becomes its own Netron module/graph, labelled by bucket name.
//   * Weights are OPAQUE. An initializer is read as { name, shape, dtype, byteLength }; its payload
//     bytes are skipped, never decoded. New weight-storage/quant formats (VXM5 int4, VXM6
//     int8/lut4, ...) change only those skipped bytes, so they need no parser change here.
//   * An unrecognized op type renders as a node by its numeric name, and an unrecognized dtype as
//     'dtype#N', rather than throwing -- a newly added engine op then shows up automatically (just
//     less decorated). vknn-metadata.json only ENRICHES the ops it already knows.

const vknn = {};

vknn.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 8) {
            const buffer = stream.peek(4);
            // Every container magic is ASCII "VXM<version>" little-endian, so the low three bytes
            // spell "VXM" and the top byte is the version digit. Match any such file so an
            // out-of-range version still reaches open() and reports a clean "incompatible version"
            // message instead of falling through to a generic "unsupported format".
            if (buffer[0] === 0x56 && buffer[1] === 0x58 && buffer[2] === 0x4d && buffer[3] >= 0x30 && buffer[3] <= 0x39) {
                return context.set('vknn');
            }
        }
        return null;
    }

    async open(context) {
        const metadata = await context.metadata('vknn-metadata.json');
        const reader = await context.read('binary');
        const container = new vknn.Reader(reader);
        return new vknn.Model(metadata, container);
    }
};

vknn.Model = class {

    constructor(metadata, container) {
        this.format = container.format;
        // One Netron module per shape bucket (a single-graph VXM3/VXM6-sub3 file yields exactly one).
        this.modules = container.buckets.map((bucket) => new vknn.Graph(metadata, container, bucket));
    }
};

vknn.Graph = class {

    constructor(metadata, container, bucket) {
        this.name = bucket.name || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const tensors = bucket.tensors;
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
                const weight = bucket.initializers.get(id);
                const initializer = weight ? new vknn.Tensor(name, type, weight) : null;
                values[id] = new vknn.Value(name, type, initializer);
            }
            return values[id];
        };
        for (const id of bucket.inputs) {
            const argument = value(id);
            this.inputs.push(new vknn.Argument(argument ? argument.name : id.toString(), argument ? [argument] : []));
        }
        for (const id of bucket.outputs) {
            const argument = value(id);
            this.outputs.push(new vknn.Argument(argument ? argument.name : id.toString(), argument ? [argument] : []));
        }
        for (const node of bucket.nodes) {
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
        const type = vknn.Utility.operator(node.type);
        this.type = metadata.type(type) || { name: type };
        this.name = node.name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        // A fused residual/bias tensor is kept in node.inputs by the graph passes (for liveness and
        // scheduling), but each is surfaced separately below as a named 'residual'/'bias' input, so
        // skip it in the positional input pass.
        const residual = Number(node.fusedResidual);
        const hasResidual = Number.isInteger(residual) && residual >= 0;
        const bias = Number(node.fusedBias);
        const hasBias = Number.isInteger(bias) && bias >= 0;
        const inputNames = Array.isArray(this.type.inputs) ? this.type.inputs.map((input) => input.name) : [];
        for (let i = 0; i < node.inputs.length; i++) {
            if (hasResidual && node.inputs[i] === residual) {
                continue;
            }
            if (hasBias && node.inputs[i] === bias) {
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
        // Decode the packed-weight quantization tag (the kWq="wq" attr; engine
        // core/quant_weights.h) into a readable format alongside the raw value, so a quantized
        // node's weight storage (int4 / int8 / lut4) is visible.
        const wq = node.attributes.find((attr) => attr.name === 'wq' && attr.kind === 1);
        if (wq) {
            const formats = { 1: 'int4', 2: 'int8', 3: 'lut4' };
            this.attributes.push(new vknn.Argument('weight_quant', formats[wq.i] || `format#${wq.i}`, 'string'));
        }
        if (node.fusedAct !== 0) {
            this.attributes.push(new vknn.Argument('fused_activation', vknn.Utility.activation(node.fusedAct), 'string'));
            if (node.actLo !== 0 || node.actHi !== 0) {
                this.attributes.push(new vknn.Argument('activation_min', node.actLo, 'float32'));
                this.attributes.push(new vknn.Argument('activation_max', node.actHi, 'float32'));
            }
        }
        // Recover the specific op for the elementwise families that collapse onto one OpType, from
        // the sub-op code the engine stores in Node::subOp (unary_type.h / binary_type.h /
        // reduce_type.h).
        const operation = vknn.Utility.subOperation(type, node.subOp);
        if (operation) {
            this.attributes.push(new vknn.Argument('operation', operation, 'string'));
        }
        if (hasResidual) {
            const argument = value(residual);
            if (argument) {
                this.inputs.push(new vknn.Argument('residual', [argument]));
            }
        }
        if (hasBias) {
            const argument = value(bias);
            if (argument) {
                this.inputs.push(new vknn.Argument('bias', [argument]));
            }
        }
    }
};

vknn.TensorType = class {

    constructor(dtype, shape, format) {
        // dtype.h DType codes. An unrecognized code shows as 'dtype#N' rather than throwing, so a
        // future element type still renders (just unnamed).
        switch (dtype) {
            case 0: this.dataType = 'float32'; break;
            case 1: this.dataType = 'float16'; break;
            case 2: this.dataType = 'int32'; break;
            case 3: this.dataType = 'int8'; break;
            case 4: this.dataType = 'uint8'; break;
            case 5: this.dataType = 'int64'; break;
            default: this.dataType = `dtype#${dtype}`; break;
        }
        this.shape = shape;
        // The IR is canonically NCHW (tensor_format_enum.h), so only annotate a boundary tensor
        // whose stored layout is not the default.
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

    constructor(name, type, weight) {
        this.name = name;
        this.type = type;
        // Weights are opaque: the payload bytes are never read from the file, so there is no data to
        // decode. Netron renders the type/shape and shows "Tensor data is empty" for the value.
        this.encoding = '<';
        this.values = null;
        if (weight && Number.isFinite(weight.byteLength)) {
            this.location = vknn.Utility.byteSize(weight.byteLength);
        }
        if (weight && weight.category) {
            this.category = weight.category;
        }
    }
};

// Reads a whole .vxm container into { buckets, format }. Each bucket is { name, tensors, nodes,
// inputs, outputs, initializers }, where initializers maps a tensor id to { byteLength } (weights
// stay opaque). The byte layout is model_io.cpp's Writer order, read little-endian.
vknn.Reader = class {

    constructor(reader) {
        const magic = reader.uint32();
        if ((magic & 0x00ffffff) !== 0x004d5856) { // low three bytes must be "VXM"
            throw new vknn.Error(`Invalid magic number '0x${(magic >>> 0).toString(16).padStart(8, '0')}' (not a .vxm file).`);
        }
        const version = (magic >>> 24) - 0x30; // 1..9 for "VXM1".."VXM9"
        // Quantized wrapper (VXM5 = int4-only, VXM6 = int8/lut4/other; core/quant_weights.h): one
        // subcontainer tag then the exact VXM3/VXM4 body. Remap to the base container and read that;
        // a future wrapper magic with the same [subtag][body] shape reuses this path unchanged.
        let base = magic;
        let quant = '';
        if (magic === 0x354d5856 || magic === 0x364d5856) {
            quant = magic === 0x354d5856 ? 'int4' : 'int8/lut4';
            const subtag = reader.uint32();
            base = 0;
            if (subtag === 3) {
                base = 0x334d5856; // VXM3 single-graph body
            } else if (subtag === 4) {
                base = 0x344d5856; // VXM4 multi-bucket body
            } else {
                throw new vknn.Error(`Unsupported VKNN quantized subcontainer tag '${subtag}'.`);
            }
        }
        if (base !== 0x334d5856 && base !== 0x344d5856) {
            // A recognizable "VXM<n>" outside the readable set (VXM1/VXM2 legacy, or a newer bump)
            // was written by an incompatible engine version; mirror model_io.cpp's diagnostic.
            throw new vknn.Error(`VXM${version} container is from an incompatible vknn version (this build reads VXM3-VXM6) -- reconvert the model from its .onnx with the current vknn_compile.`);
        }

        // ---- primitive vector readers (little-endian; mirror Writer::vec/str in model_io.cpp) ----
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
                values[i] = Number(reader.int64());
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

        // Read the tensor/node/I-O tables of one graph body (writeGraphStructure in model_io.cpp).
        // Initializers are read by the caller, because VXM3 inlines their bytes while VXM4 references
        // a shared pool.
        const readStructure = () => {
            const tensorCount = reader.uint32();
            const tensors = new Array(tensorCount);
            for (let i = 0; i < tensorCount; i++) {
                const name = reader.string();
                const shape = int64s();
                const dtype = reader.uint32();
                const format = reader.uint32();
                const flags = reader.uint32();
                tensors[i] = {
                    name, shape, dtype, format,
                    isInput: (flags & 1) !== 0,
                    isOutput: (flags & 2) !== 0,
                    isInitializer: (flags & 4) !== 0
                };
            }
            const nodeCount = reader.uint32();
            const nodes = new Array(nodeCount);
            for (let i = 0; i < nodeCount; i++) {
                const type = reader.uint32();
                const name = reader.string();
                const inputs = int32s(); // TensorId == int32
                const outputs = int32s();
                const fusedAct = reader.uint32();
                const actLo = reader.float32();
                const actHi = reader.float32();
                const subOp = Number(reader.int64());
                const fusedResidual = Number(reader.int64());
                const fusedBias = Number(reader.int64());
                const attributeCount = reader.uint32();
                const attributes = new Array(attributeCount);
                for (let j = 0; j < attributeCount; j++) {
                    const attrName = reader.string();
                    const kind = reader.uint32();
                    const i = Number(reader.int64());
                    const f = reader.float32();
                    const ints = int64s();
                    const floats = float32s();
                    const str = reader.string();
                    attributes[j] = { name: attrName, kind, i, f, ints, floats, str };
                }
                nodes[i] = { type, name, inputs, outputs, fusedAct, actLo, actHi, subOp, fusedResidual, fusedBias, attributes };
            }
            const inputs = int32s();
            const outputs = int32s();
            return { tensors, nodes, inputs, outputs };
        };

        this.buckets = [];
        if (base === 0x334d5856) {
            // VXM3 single-graph body: the structure, then inline initializers (tensor id + opaque
            // byte blob). The payload bytes are skipped, never materialized.
            const bucket = readStructure();
            bucket.name = '';
            bucket.initializers = new Map();
            const initializerCount = reader.uint32();
            for (let i = 0; i < initializerCount; i++) {
                const id = Number(reader.int64());
                const byteLength = reader.uint32();
                reader.skip(byteLength);
                bucket.initializers.set(id, { byteLength });
            }
            this.buckets.push(bucket);
        } else {
            // VXM4 multi-bucket body: bucket count, a shared content-deduped initializer pool, then
            // each bucket's structure + its (tensor id -> pool index) table. The pool bytes are
            // skipped; only each blob's byte size is kept, so every bucket's weights resolve to
            // { byteLength } through the pool index.
            const bucketCount = reader.uint32();
            const poolCount = reader.uint32();
            const pool = new Array(poolCount);
            for (let i = 0; i < poolCount; i++) {
                const byteLength = reader.uint32();
                reader.skip(byteLength);
                pool[i] = { byteLength };
            }
            for (let b = 0; b < bucketCount; b++) {
                const name = reader.string();
                const bucket = readStructure();
                bucket.name = name;
                bucket.initializers = new Map();
                const refCount = reader.uint32();
                for (let i = 0; i < refCount; i++) {
                    const id = Number(reader.int64());
                    const poolIndex = reader.uint32();
                    const blob = poolIndex < pool.length ? pool[poolIndex] : { byteLength: 0 };
                    bucket.initializers.set(id, { byteLength: blob.byteLength });
                }
                this.buckets.push(bucket);
            }
        }

        // Container label: version + quantization + bucket count (e.g. "VKNN v5 (int4, 2 buckets)").
        const notes = [];
        if (quant) {
            notes.push(quant);
        }
        if (this.buckets.length > 1) {
            notes.push(`${this.buckets.length} buckets`);
        }
        this.format = notes.length > 0 ? `VKNN v${version} (${notes.join(', ')})` : `VKNN v${version}`;
    }
};

vknn.Utility = class {

    // OpType (include/vknn/op_type.h) is APPEND-ONLY, so the enum's integer value is a stable index
    // into this table; the spellings are the engine's own opTypeName() (src/core/op.cpp), which are
    // ONNX-style, so vknn-metadata.json enriches them by the same name. An index past the table
    // renders as 'Op#N' -- a newly added engine op shows up automatically, just undecorated.
    static operator(type) {
        vknn.Utility._operators = vknn.Utility._operators || [
            'Unknown',                          // 0
            'Conv',                             // 1
            'ConvTranspose',                    // 2
            'Clip',                             // 3
            'Relu',                             // 4
            'Add',                              // 5
            'GlobalAveragePool',                // 6
            'AveragePool',                      // 7
            'MaxPool',                          // 8
            'Gemm',                             // 9
            'MatMul',                           // 10
            'Einsum',                           // 11
            'Reshape',                          // 12
            'Expand',                           // 13
            'Tile',                             // 14
            'Squeeze',                          // 15
            'Flatten',                          // 16
            'Softmax',                          // 17
            'LayerNormalization',               // 18
            'BatchNormalization',               // 19
            'Concat',                           // 20
            'Pad',                              // 21
            'Identity',                         // 22
            'Constant',                         // 23
            'Shape',                            // 24
            'Gather',                           // 25
            'Unsqueeze',                        // 26
            'Unary',                            // 27
            'Binary',                           // 28
            'PRelu',                            // 29
            'Resize',                           // 30
            'GridSample',                       // 31
            'Transpose',                        // 32
            'Slice',                            // 33
            'Reduce',                           // 34
            'DepthToSpace',                     // 35
            'Cast',                             // 36
            'Split',                            // 37
            'Where',                            // 38
            'Equal',                            // 39
            'Greater',                          // 40
            'GreaterEqual',                     // 41
            'ConstantOfShape',                  // 42
            'EyeLike',                          // 43
            'ScatterND',                        // 44
            'FusedSE',                          // 45
            'FusedDwPw',                        // 46
            'FusedPointwise',                   // 47
            'ConvertLayout',                    // 48
            'ConvertDtype',                     // 49
            'Range',                            // 50
            'ConvGemm',                         // 51
            'Less',                             // 52
            'LessEqual',                        // 53
            'Dropout',                          // 54
            'TopK',                             // 55
            'InstanceNormalization',            // 56
            'QuantizeLinear',                   // 57
            'DequantizeLinear',                 // 58
            'DynamicQuantizeLinear',            // 59
            'QLinearConv',                      // 60
            'QLinearMatMul',                    // 61
            'QLinearAdd',                       // 62
            'QLinearGlobalAveragePool',         // 63
            'MatMulInteger',                    // 64
            'ConvInteger',                      // 65
            'QGemm',                            // 66
            'IsNaN',                            // 67
            'And',                              // 68
            'RMSNorm',                          // 69
            'SimplifiedLayerNormalization',     // 70
            'SkipSimplifiedLayerNormalization', // 71
            'SkipLayerNormalization',           // 72
            'RotaryEmbedding',                  // 73
            'MultiHeadAttention',               // 74
            'GroupQueryAttention',              // 75
            'MatMulNBits',                      // 76
            'Rope',                             // 77
            'FusedAttention'                    // 78
        ];
        const operators = vknn.Utility._operators;
        return type >= 0 && type < operators.length ? operators[type] : `Op#${type}`;
    }

    // Fused-activation name for an ActType code (act_type.h). An unrecognized code shows as its
    // number.
    static activation(code) {
        vknn.Utility._activations = vknn.Utility._activations || [
            'None',      // 0
            'Relu',      // 1
            'Relu6',     // 2
            'Clip',      // 3
            'HardSwish', // 4
            'SiLU'       // 5
        ];
        const activations = vknn.Utility._activations;
        return code >= 0 && code < activations.length ? activations[code] : code.toString();
    }

    // The specific op an elementwise family (Unary/Binary/Reduce) resolves to, from the sub-op code
    // the engine stores in Node::subOp. The tables are the engine's own enum orders (unary_type.h /
    // binary_type.h / reduce_type.h); an out-of-range code returns null (no attribute emitted).
    static subOperation(family, code) {
        vknn.Utility._subOperations = vknn.Utility._subOperations || {
            Unary: [
                'Sigmoid',    // 0
                'Tanh',       // 1
                'HardSwish',  // 2
                'HardSigmoid',// 3
                'LeakyRelu',  // 4
                'Elu',        // 5
                'Abs',        // 6
                'Neg',        // 7
                'Exp',        // 8
                'Log',        // 9
                'Sqrt',       // 10
                'Floor',      // 11
                'Ceil',       // 12
                'Relu',       // 13
                'SiLU',       // 14
                'Erf',        // 15
                'Cos',        // 16
                'Sin',        // 17
                'Reciprocal', // 18
                'Softplus',   // 19
                'Round',      // 20
                'Trunc'       // 21
            ],
            Binary: [
                'Mul', // 0
                'Sub', // 1
                'Div', // 2
                'Max', // 3
                'Min', // 4
                'Pow', // 5
                'Add'  // 6
            ],
            Reduce: [
                'Mean', // 0
                'Sum',  // 1
                'Max',  // 2
                'Min',  // 3
                'Prod', // 4
                'L2'    // 5
            ]
        };
        const table = vknn.Utility._subOperations[family];
        return table && code >= 0 && code < table.length ? table[code] : null;
    }

    // Compact human-readable byte count for a weight's on-disk size.
    static byteSize(bytes) {
        if (bytes < 1024) {
            return `${bytes} B`;
        }
        const units = ['KB', 'MB', 'GB', 'TB'];
        let value = bytes / 1024;
        let unit = 0;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit++;
        }
        return `${value.toFixed(2)} ${units[unit]}`;
    }
};

vknn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading VKNN model.';
    }
};

export const ModelFactory = vknn.ModelFactory;
