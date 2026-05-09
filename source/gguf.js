
const gguf = {};

gguf.ModelFactory = class {

    async match(context) {
        const reader = gguf.Reader.open(context);
        if (reader) {
            return context.set('gguf', reader);
        }
        return null;
    }

    async open(context) {
        const metadata = await context.metadata('gguf-metadata.json');
        const target = context.value;
        await target.read();
        return new gguf.Model(metadata, target);
    }
};

gguf.Model = class {

    constructor(metadata, target) {
        this.format = target.format;
        this.metadata = [];
        const extra = new Map();
        let architecture = '?';
        for (const [name, value] of target.metadata) {
            switch (name) {
                case 'general.name': this.name = value; break;
                case 'general.architecture': architecture = value; break;
                case 'general.description': this.description = value; break;
                case 'general.author': this.metadata.push(new gguf.Argument('author', value)); break;
                case 'general.license': this.metadata.push(new gguf.Argument('license', value)); break;
                case 'general.file_type':
                case 'general.quantization_version':
                    break;
                default:
                    extra.set(name, value);
                    break;
            }
        }
        const tokenizer = { type: 'tokenizer', metadata: new Map(), layers: [] };
        const graph = {};
        graph.type = architecture;
        graph.attributes = [];
        for (const [name, value] of extra) {
            if (name.startsWith('tokenizer.')) {
                const match = name.match(/^(.*)\.(.*?)$/);
                if (match) {
                    const [, param] = match.slice(1);
                    tokenizer.metadata.set(param, value);
                }
            } else if (architecture !== '?' && name.startsWith(`${architecture}.`)) {
                graph.attributes.push(new gguf.Argument(name, value));
            } else {
                this.metadata.push(new gguf.Argument(name, value));
            }
        }
        const context = new gguf.Context(metadata, target, extra, architecture);
        graph.layers = context.build();
        if (tokenizer.metadata.size > 0) {
            graph.layers = graph.layers || [];
            graph.layers.unshift(tokenizer);
            if (context.structured) {
                graph.layers.push({ type: 'tokenizer', metadata: new Map(), layers: [] });
            }
        }
        this.modules = [new gguf.Graph(graph)];
    }

};

gguf.Graph = class {

    constructor(graph) {
        this.name = graph.type;
        this.type = '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        this.attributes = graph.attributes || [];
        let valueIndex = 0;
        let prevValue = null;
        let ropeFreqsValue = null;
        const newValue = () => new gguf.Value(`v${valueIndex++}`);
        const addNode = (entry, inputValues, outputValue) => {
            const node = new gguf.Node(entry);
            for (const v of inputValues) {
                node.inputs.unshift(new gguf.Argument('input', [v]));
            }
            if (outputValue) {
                node.outputs.push(new gguf.Argument('output', [outputValue]));
            }
            this.nodes.push(node);
        };
        const addOp = (type, inputValues, outputValue) => {
            addNode({ name: '', type, weights: new Map(), metadata: new Map(), layers: [] }, inputValues, outputValue);
        };
        for (const layer of graph.layers) {
            if (Array.isArray(layer.layers) && layer.layers.length > 0) {
                const map = new Map();
                for (const item of layer.layers) {
                    map.set(item.name, item);
                }
                const has = (name) => map.has(name);
                const get = (name) => map.get(name);
                const used = new Set();
                const use = (name) => {
                    used.add(name);
                    return get(name);
                };
                const hasMoe = has('ffn_gate_inp');
                const hasFusedExps = has('ffn_gate_up_exps') && has('ffn_down_exps');
                const hasFfn = has('ffn_up') || hasMoe;
                const buildLinearFfn = (input, gateKey, upKey, downKey) => {
                    if (!has(downKey)) {
                        return input;
                    }
                    const inputs = [input];
                    if (has(gateKey)) {
                        const g = newValue();
                        addNode(use(gateKey), [input], g);
                        inputs.push(g);
                    }
                    if (has(upKey)) {
                        const u = newValue();
                        addNode(use(upKey), [input], u);
                        inputs.push(u);
                    }
                    const d = newValue();
                    addNode(use(downKey), inputs, d);
                    return d;
                };
                const buildFusedExpsFfn = (input) => {
                    const gu = newValue();
                    addNode(use('ffn_gate_up_exps'), [input], gu);
                    const d = newValue();
                    addNode(use('ffn_down_exps'), [gu], d);
                    return d;
                };
                const applyNorm = (groupName, value) => {
                    if (!has(groupName)) {
                        return value;
                    }
                    const out = newValue();
                    addNode(use(groupName), [value], out);
                    return out;
                };
                const buildFfn = (input) => {
                    if (hasMoe) {
                        const moeInput = applyNorm('ffn_pre_norm_2', input);
                        const g1 = newValue();
                        addNode(use('ffn_gate_inp'), [moeInput], g1);
                        let moeOut = hasFusedExps ?
                            buildFusedExpsFfn(g1) :
                            buildLinearFfn(g1, 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps');
                        moeOut = applyNorm('ffn_post_norm_2', moeOut);
                        if (has('ffn_up_shexp')) {
                            const sharedOut = buildLinearFfn(input, 'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp');
                            const sum = newValue();
                            addOp('ADD', [moeOut, sharedOut], sum);
                            return sum;
                        }
                        if (hasFusedExps && has('ffn_up')) {
                            let sharedOut = buildLinearFfn(input, 'ffn_gate', 'ffn_up', 'ffn_down');
                            sharedOut = applyNorm('ffn_post_norm_1', sharedOut);
                            const sum = newValue();
                            addOp('ADD', [moeOut, sharedOut], sum);
                            return sum;
                        }
                        return moeOut;
                    }
                    return buildLinearFfn(input, 'ffn_gate', 'ffn_up', 'ffn_down');
                };
                const applyLayerOutScale = (value) => {
                    if (!has('layer_out_scale')) {
                        return value;
                    }
                    const out = newValue();
                    addNode(use('layer_out_scale'), [value], out);
                    return out;
                };
                const ropeFreqs = ropeFreqsValue;
                const buildAttention = (input, output) => {
                    const inputs = ropeFreqs ? [input, ropeFreqs] : [input];
                    addNode(use('attention'), inputs, output);
                };
                if (has('attn_norm') && has('attention') && !has('ffn_norm') && !has('attn_post_norm') && hasFfn) {
                    // Parallel attention + FFN (phi-2, falcon)
                    const inp = prevValue || newValue();
                    const normOut = newValue();
                    const attnOut = newValue();
                    const out = newValue();
                    addNode(use('attn_norm'), [inp], normOut);
                    buildAttention(normOut, attnOut);
                    const ffnOut = buildFfn(normOut);
                    addOp('ADD', [attnOut, ffnOut, inp], out);
                    prevValue = out;
                } else if (has('attn_norm') && has('attention') && has('cross_attn_norm') && has('cross_attention') && has('ffn_norm') && hasFfn) {
                    // Pre-norm with cross-attention (T5 decoder)
                    const inp = prevValue || newValue();
                    let cur = inp;
                    const n1 = newValue();
                    addNode(use('attn_norm'), [cur], n1);
                    const a1 = newValue();
                    buildAttention(n1, a1);
                    const r1 = newValue();
                    addOp('ADD', [a1, cur], r1);
                    cur = r1;
                    const cn = newValue();
                    addNode(use('cross_attn_norm'), [cur], cn);
                    const ca = newValue();
                    addNode(use('cross_attention'), [cn], ca);
                    const r2 = newValue();
                    addOp('ADD', [ca, cur], r2);
                    cur = r2;
                    const n2 = newValue();
                    addNode(use('ffn_norm'), [cur], n2);
                    const f1 = buildFfn(n2);
                    const r3 = newValue();
                    addOp('ADD', [f1, cur], r3);
                    prevValue = r3;
                } else if (has('attn_norm') && has('attention') && has('ffn_norm') && hasFfn) {
                    // Pre-norm transformer (llama, qwen, gemma, etc.)
                    const inp = prevValue || newValue();
                    let cur = inp;
                    const n1 = newValue();
                    addNode(use('attn_norm'), [cur], n1);
                    const a1 = newValue();
                    buildAttention(n1, a1);
                    let preAdd1 = a1;
                    if (has('ssm')) {
                        const s1 = newValue();
                        addNode(use('ssm'), [n1], s1);
                        const sum = newValue();
                        addOp('ADD', [a1, s1], sum);
                        preAdd1 = sum;
                    }
                    if (has('attn_post_norm')) {
                        const pn = newValue();
                        addNode(use('attn_post_norm'), [preAdd1], pn);
                        preAdd1 = pn;
                    }
                    const r1 = newValue();
                    addOp('ADD', [preAdd1, cur], r1);
                    cur = r1;
                    const n2 = newValue();
                    addNode(use('ffn_norm'), [cur], n2);
                    const f1 = buildFfn(n2);
                    let preAdd2 = f1;
                    if (has('ffn_post_norm')) {
                        const pn = newValue();
                        addNode(use('ffn_post_norm'), [f1], pn);
                        preAdd2 = pn;
                    }
                    const r2 = newValue();
                    addOp('ADD', [preAdd2, cur], r2);
                    let final = r2;
                    // Elides the gating mul by `inp_per_layer_slice` between
                    // GELU and proj: that slice is a precomputation over the
                    // per_layer_* globals we don't model.
                    if (has('inp_gate') && has('proj') && has('post_norm')) {
                        const peIn = final;
                        const g = newValue();
                        addNode(use('inp_gate'), [peIn], g);
                        const gAct = newValue();
                        addOp('GELU', [g], gAct);
                        const p = newValue();
                        addNode(use('proj'), [gAct], p);
                        const pn = newValue();
                        addNode(use('post_norm'), [p], pn);
                        const r3 = newValue();
                        addOp('ADD', [pn, peIn], r3);
                        final = r3;
                    }
                    prevValue = applyLayerOutScale(final);
                } else if (has('attention') && (has('attn_output_norm') || has('layer_output_norm'))) {
                    // Post-norm (BERT)
                    const inp = prevValue || newValue();
                    const a1 = newValue();
                    buildAttention(inp, a1);
                    const r1 = newValue();
                    addOp('ADD', [a1, inp], r1);
                    let cur = r1;
                    if (has('attn_output_norm')) {
                        const n1 = newValue();
                        addNode(use('attn_output_norm'), [cur], n1);
                        cur = n1;
                    }
                    if (hasFfn) {
                        const residual = cur;
                        const f1 = buildFfn(cur);
                        const r2 = newValue();
                        addOp('ADD', [f1, residual], r2);
                        cur = r2;
                    }
                    if (has('layer_output_norm')) {
                        const n2 = newValue();
                        addNode(use('layer_output_norm'), [cur], n2);
                        cur = n2;
                    }
                    prevValue = cur;
                } else if (has('attn_norm') && has('ssm') && hasFfn) {
                    // SSM + FFN (hybrid SSM-variant block: jamba, granitehybrid, etc.)
                    const inp = prevValue || newValue();
                    const n1 = newValue();
                    addNode(use('attn_norm'), [inp], n1);
                    const s1 = newValue();
                    addNode(use('ssm'), [n1], s1);
                    let preAdd1 = s1;
                    if (has('attn_post_norm')) {
                        const pn = newValue();
                        addNode(use('attn_post_norm'), [s1], pn);
                        preAdd1 = pn;
                    }
                    const r1 = newValue();
                    addOp('ADD', [preAdd1, inp], r1);
                    let cur = r1;
                    if (has('ffn_norm')) {
                        const n2 = newValue();
                        addNode(use('ffn_norm'), [cur], n2);
                        cur = n2;
                    }
                    const f1 = buildFfn(cur);
                    let preAdd2 = f1;
                    if (has('ffn_post_norm')) {
                        const pn = newValue();
                        addNode(use('ffn_post_norm'), [f1], pn);
                        preAdd2 = pn;
                    }
                    const r2 = newValue();
                    addOp('ADD', [preAdd2, r1], r2);
                    prevValue = r2;
                } else if (has('attn_norm') && has('ssm')) {
                    // SSM (Mamba)
                    const inp = prevValue || newValue();
                    const n1 = newValue();
                    addNode(use('attn_norm'), [inp], n1);
                    const s1 = newValue();
                    addNode(use('ssm'), [n1], s1);
                    const r1 = newValue();
                    addOp('ADD', [s1, inp], r1);
                    prevValue = r1;
                } else if (has('attn_norm') && has('time_mix') && has('channel_mix')) {
                    // RWKV
                    const inp = prevValue || newValue();
                    const n1 = newValue();
                    addNode(use('attn_norm'), [inp], n1);
                    const t1 = newValue();
                    addNode(use('time_mix'), [n1], t1);
                    const r1 = newValue();
                    addOp('ADD', [t1, inp], r1);
                    let cur = r1;
                    let cmInput = cur;
                    if (has('ffn_norm')) {
                        const n2 = newValue();
                        addNode(use('ffn_norm'), [cur], n2);
                        cmInput = n2;
                    }
                    const c1 = newValue();
                    addNode(use('channel_mix'), [cmInput], c1);
                    const r2 = newValue();
                    addOp('ADD', [c1, cur], r2);
                    cur = r2;
                    prevValue = cur;
                } else if (has('attn_norm') && has('attention') && hasFfn) {
                    // Pre-norm without ffn_norm (some variants)
                    const inp = prevValue || newValue();
                    const n1 = newValue();
                    addNode(use('attn_norm'), [inp], n1);
                    const a1 = newValue();
                    buildAttention(n1, a1);
                    let preAdd1 = a1;
                    if (has('attn_post_norm')) {
                        const pn = newValue();
                        addNode(use('attn_post_norm'), [a1], pn);
                        preAdd1 = pn;
                    }
                    const r1 = newValue();
                    addOp('ADD', [preAdd1, inp], r1);
                    const f1 = buildFfn(r1);
                    const r2 = newValue();
                    addOp('ADD', [f1, r1], r2);
                    prevValue = r2;
                } else {
                    // Fallback: linear chain
                    for (const item of layer.layers) {
                        const node = new gguf.Node(item);
                        if (prevValue) {
                            node.inputs.unshift(new gguf.Argument('input', [prevValue]));
                        }
                        const out = newValue();
                        node.outputs.push(new gguf.Argument('output', [out]));
                        prevValue = out;
                        this.nodes.push(node);
                    }
                    continue;
                }
                for (const item of layer.layers) {
                    if (!used.has(item.name)) {
                        this.nodes.push(new gguf.Node(item));
                    }
                }
            } else if (layer.type === 'ROPE_FREQS') {
                const node = new gguf.Node(layer);
                ropeFreqsValue = newValue();
                node.outputs.push(new gguf.Argument('output', [ropeFreqsValue]));
                this.nodes.push(node);
            } else {
                const node = new gguf.Node(layer);
                if (prevValue && layer.type !== 'weights') {
                    node.inputs.unshift(new gguf.Argument('input', [prevValue]));
                }
                if (layer.type !== 'weights') {
                    const outputValue = newValue();
                    node.outputs.push(new gguf.Argument('output', [outputValue]));
                    prevValue = outputValue;
                }
                this.nodes.push(node);
            }
        }
    }
};

gguf.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

gguf.Value = class {

    constructor(name, type, initializer) {
        this.name = name;
        this.type = type || null;
        this.quantization = initializer && initializer.quantization ? initializer.quantization : null;
        this.initializer = initializer || null;
    }
};

gguf.Node = class {

    constructor(layer) {
        this.type = layer.category ? { name: layer.type, category: layer.category } : { name: layer.type };
        this.name = layer.name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (layer.weights) {
            for (const [name, weight] of layer.weights) {
                const tensor = new gguf.Tensor(weight);
                const value = new gguf.Value(weight.name, tensor.type, tensor);
                const argument = new gguf.Argument(name, [value]);
                this.inputs.push(argument);
            }
        }
        if (layer.metadata) {
            for (const [name, value] of layer.metadata) {
                const attribute = new gguf.Argument(name, value);
                this.attributes.push(attribute);
            }
        }
    }
};

gguf.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return (this.dataType || '?') + this.shape.toString();
    }
};

gguf.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

gguf.Tensor = class {

    constructor(tensor) {
        const shape = new gguf.TensorShape(tensor.ne);
        this.type = new gguf.TensorType(tensor.dtype, shape);
        const type = gguf.QuantizationType[tensor.type];
        if (type.block_size > 1) {
            this.quantization = { type: type.name.toLowerCase() };
        } else {
            this.encoding = '<';
            this._data = tensor.data;
        }
    }

    get values() {
        if (this._data) {
            return this._data.peek();
        }
        return null;
    }
};

gguf.Reader = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature === 'GGUF') {
                return new gguf.Reader(context);
            }
        }
        return null;
    }

    constructor(context) {
        this.context = context;
    }

    async read() {
        const context = this.context;
        const stream = context.stream;
        let reader = await context.read('binary');
        reader = new gguf.BinaryReader(reader);
        this.tensors = new Map();
        this.metadata = new Map();
        this.header = {};
        this.header.magic = String.fromCharCode.apply(null, reader.read(4));
        this.header.version = reader.uint32();
        this.format = `GGUF v${this.header.version}`;
        if (this.header.version >= 2) {
            this.header.n_tensors = reader.uint64().toNumber();
            this.header.n_kv = reader.uint64().toNumber();
            for (let i = 0; i < this.header.n_kv; i++) {
                const entry = reader.entry();
                this.metadata.set(entry.name, entry.value);
            }
            const tensors = this.header.n_tensors;
            if (tensors > 0) {
                for (let i = 0; i < tensors; i++) {
                    const tensor = reader.tensor();
                    this.tensors.set(tensor.name, tensor);
                }
                this.alignment = this.metadata.get('general.alignment') || 32;
                if (reader.position % this.alignment !== 0) {
                    reader.skip(this.alignment - (reader.position % this.alignment));
                }
                const offset = reader.position;
                for (const tensor of this.tensors.values()) {
                    const type = gguf.QuantizationType[tensor.type];
                    if (!type) {
                        throw new gguf.Error(`Unsupported tensor quantization type '${tensor.type}'.`);
                    }
                    tensor.dtype = type.name;
                    if (offset < reader.length) {
                        const n_elems = tensor.ne.reduce((a, b) => a * b, 1);
                        const n_bytes = Math.floor((n_elems * type.type_size) / type.block_size);
                        reader.seek(offset + tensor.offset);
                        tensor.data = reader.stream(n_bytes);
                    }
                }
            }
        }
        stream.seek(0);
        delete this.context;
    }
};

gguf.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get length() {
        return this._reader.length;
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

    stream(length) {
        return this._reader.stream(length);
    }

    read(length) {
        return this._reader.read(length);
    }

    byte() {
        return this._reader.byte();
    }

    int8() {
        return this._reader.int8();
    }

    uint16() {
        return this._reader.uint16();
    }

    int16() {
        return this._reader.int16();
    }

    uint32() {
        return this._reader.uint32();
    }

    int32() {
        return this._reader.int32();
    }

    uint64() {
        return this._reader.uint64();
    }

    int64() {
        return this._reader.int64();
    }

    float32() {
        return this._reader.float32();
    }

    float64() {
        return this._reader.float64();
    }

    string() {
        const size = this.uint64().toNumber();
        const buffer = this.read(size);
        return String.fromCharCode.apply(null, buffer);
    }

    value(type) {
        switch (type) {
            case gguf.Type.UINT8: return this.byte();
            case gguf.Type.INT8: return this.int8();
            case gguf.Type.UINT16: return this.uint16();
            case gguf.Type.INT16: return this.int16();
            case gguf.Type.UINT32: return this.uint32();
            case gguf.Type.INT32: return this.int32();
            case gguf.Type.UINT64: return this.uint64();
            case gguf.Type.INT64: return this.int64();
            case gguf.Type.FLOAT32: return this.float32();
            case gguf.Type.FLOAT64: return this.float64();
            case gguf.Type.BOOL: return this.byte() !== 0;
            case gguf.Type.STRING: return this.string();
            case gguf.Type.ARRAY: {
                const type = this.uint32();
                const size = this.uint64().toNumber();
                const value = new Array(size);
                for (let i = 0; i < size; i++) {
                    value[i] = this.value(type);
                }
                return value;
            }
            default: {
                throw new gguf.Error(`Unsupported GGUF type '${type}'.`);
            }
        }
    }

    entry() {
        const name = this.string();
        const type = this.uint32();
        const value = this.value(type);
        return { name, value, type };
    }

    tensor() {
        const tensor = {};
        tensor.name = this.string();
        const n_dims = this.uint32();
        tensor.ne = new Array(n_dims);
        for (let i = 0; i < n_dims; i++) {
            tensor.ne[i] = this.uint64().toNumber();
        }
        tensor.type = this.uint32();
        tensor.offset = this.uint64().toNumber();
        return tensor;
    }
};

gguf.Type = {
    UINT8: 0,
    INT8: 1,
    UINT16: 2,
    INT16: 3,
    UINT32: 4,
    INT32: 5,
    FLOAT32: 6,
    BOOL: 7,
    STRING: 8,
    ARRAY: 9,
    UINT64: 10,
    INT64: 11,
    FLOAT64: 12,
};

// https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h - ggml_type
// https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml.c
// https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py - GGML_QUANT_SIZES
gguf.QuantizationType = [
    /*  0 */ { name: 'float32',    block_size: 1,   type_size: 4 },
    /*  1 */ { name: 'float16',    block_size: 1,   type_size: 2 },
    /*  2 */ { name: 'q4_0',       block_size: 32,  type_size: 2 + 16 },
    /*  3 */ { name: 'q4_1',       block_size: 32,  type_size: 2 + 2 + 16 },
    /*  4 */ { name: 'q4_2',       block_size: 16,  type_size: 2 + 8 }, // deprecated
    /*  5 */ { name: 'q4_3',       block_size: 16,  type_size: 2 + 2 + 8 }, // deprecated
    /*  6 */ { name: 'q5_0',       block_size: 32,  type_size: 2 + 4 + 16 },
    /*  7 */ { name: 'q5_1',       block_size: 32,  type_size: 2 + 2 + 4 + 16 },
    /*  8 */ { name: 'q8_0',       block_size: 32,  type_size: 2 + 32 },
    /*  9 */ { name: 'q8_1',       block_size: 32,  type_size: 4 + 4 + 32 },
    /* 10 */ { name: 'q2_K',       block_size: 256, type_size: 2 + 2 + 16 + 64 },
    /* 11 */ { name: 'q3_K',       block_size: 256, type_size: 2 + 64 + 32 + 12 },
    /* 12 */ { name: 'q4_K',       block_size: 256, type_size: 2 + 2 + 128 + 12 },
    /* 13 */ { name: 'q5_K',       block_size: 256, type_size: 2 + 2 + 128 + 32 + 12 },
    /* 14 */ { name: 'q6_K',       block_size: 256, type_size: 2 + 128 + 64 + 16 },
    /* 15 */ { name: 'q8_K',       block_size: 256, type_size: 4 + 256 + 32 },
    /* 16 */ { name: 'iq2_xxs',    block_size: 256, type_size: 2 + 64 },
    /* 17 */ { name: 'iq2_xs',     block_size: 256, type_size: 2 + 64 + 8 },
    /* 18 */ { name: 'iq3_xxs',    block_size: 256, type_size: 2 + 64 + 32 },
    /* 19 */ { name: 'iq1_s',      block_size: 256, type_size: 2 + 32 + 16 },
    /* 20 */ { name: 'iq4_nl',     block_size: 32,  type_size: 2 + 16 },
    /* 21 */ { name: 'iq3_s',      block_size: 256, type_size: 2 + 64 + 32 + 8 + 4 },
    /* 22 */ { name: 'iq2_s',      block_size: 256, type_size: 2 + 64 + 16 },
    /* 23 */ { name: 'iq4_xs',     block_size: 256, type_size: 2 + 2 + 128 + 4 },
    /* 24 */ { name: 'int8',       block_size: 1,   type_size: 1 },
    /* 25 */ { name: 'int16',      block_size: 1,   type_size: 2 },
    /* 26 */ { name: 'int32',      block_size: 1,   type_size: 4 },
    /* 27 */ { name: 'int64',      block_size: 1,   type_size: 8 },
    /* 28 */ { name: 'float64',    block_size: 1,   type_size: 8 },
    /* 29 */ { name: 'iq1_m',      block_size: 256, type_size: 32 + 16 + 8 },
    /* 30 */ { name: 'bfloat16',   block_size: 1,   type_size: 2 },
    /* 31 */ { name: 'q4_0_4_4',   block_size: 32,  type_size: 2 + 16 }, // deprecated
    /* 32 */ { name: 'q4_0_4_8',   block_size: 32,  type_size: 2 + 16 }, // deprecated
    /* 33 */ { name: 'q4_0_8_8',   block_size: 32,  type_size: 2 + 16 }, // deprecated
    /* 34 */ { name: 'tq1_0',      block_size: 256, type_size: 2 + 4 * 13 },
    /* 35 */ { name: 'tq2_0',      block_size: 256, type_size: 2 + 64 },
    /* 36 */ { name: 'iq4_nl_4_4', block_size: 32,  type_size: 2 + 16 }, // deprecated
    /* 37 */ { name: 'iq4_nl_4_8', block_size: 32,  type_size: 2 + 16 }, // deprecated
    /* 38 */ { name: 'iq4_nl_8_8', block_size: 32,  type_size: 2 + 16 }, // deprecated
    /* 39 */ { name: 'mxfp4',      block_size: 32,  type_size: 1 + 16 }
];

gguf.Context = class {

    constructor(metadata, target, kvMetadata, architecture) {
        const archType = metadata ? metadata.type(architecture) : null;
        const archDef = archType && archType.graph ? archType : null;
        this._archDef = archDef;
        this._tensors = target.tensors;
        this._architecture = architecture;
        this._blockCount = kvMetadata.get(`${architecture}.block_count`) || 0;
        this._blockTypes = new Map();
        if (archDef && archDef.graph) {
            const registerSection = (section) => {
                if (section) {
                    for (const block of section) {
                        this._blockTypes.set(block.name, block);
                        if (block.tensors) {
                            for (const tensor of block.tensors) {
                                this._blockTypes.set(tensor, block);
                            }
                        }
                    }
                }
            };
            for (const section of [archDef.graph.input, archDef.graph.blocks, archDef.graph.output]) {
                registerSection(section);
            }
            for (const subgraph of [archDef.graph.encoder, archDef.graph.decoder]) {
                if (subgraph) {
                    for (const section of [subgraph.input, subgraph.blocks, subgraph.output]) {
                        registerSection(section);
                    }
                }
            }
        }
    }

    get structured() {
        return this._archDef !== null && this._tensors.size > 0;
    }

    build() {
        const tensors = this._tensors;
        const layers = [];
        const claimed = new Set();
        const archDef = this._archDef;
        const collectWeights = (prefix) => {
            const weights = new Map();
            for (const [name, tensor] of tensors) {
                if (name.startsWith(`${prefix}.`) || name === prefix) {
                    const suffix = name.slice(prefix.length + 1) || 'data';
                    weights.set(suffix, tensor);
                    claimed.add(name);
                }
            }
            return weights;
        };
        // Resolve display type/category for a component group from metadata
        // (when an arch definition is loaded), otherwise default to 'weights'.
        const resolveBlock = (group) => {
            const block = this._blockTypes.get(group);
            return block ? { type: block.type, category: block.category } : { type: 'weights' };
        };
        const pushFlat = (prefix, weights) => {
            const resolved = resolveBlock(prefix);
            layers.push({ name: prefix, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
        };
        // Build a structured block at `blockPrefix`, returning sub-layers in
        // discovery order. Tensors in the block are grouped by component
        // (attn, ffn, ...) via _classifyTensor.
        const buildBlockLayers = (blockPrefix) => {
            const groups = new Map();
            const order = [];
            for (const [name] of tensors) {
                if (name.startsWith(`${blockPrefix}.`)) {
                    const rest = name.slice(blockPrefix.length + 1);
                    const group = this._classifyTensor(rest);
                    if (!groups.has(group)) {
                        groups.set(group, new Map());
                        order.push(group);
                    }
                    groups.get(group).set(rest, name);
                }
            }
            const blockLayers = [];
            for (const group of order) {
                const weights = new Map();
                for (const [suffix, fullName] of groups.get(group)) {
                    weights.set(suffix, tensors.get(fullName));
                    claimed.add(fullName);
                }
                if (weights.size > 0) {
                    const resolved = resolveBlock(group);
                    blockLayers.push({ name: group, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
                }
            }
            return blockLayers;
        };
        // Discover block indices from tensor names.
        const blockIndices = new Set();
        const encDecIndices = new Map([['enc', new Set()], ['dec', new Set()]]);
        const blockRe = /^(?:(enc|dec)\.)?blk\.(\d+)\./;
        for (const [name] of tensors) {
            const m = name.match(blockRe);
            if (m) {
                const idx = parseInt(m[2], 10);
                if (m[1]) {
                    encDecIndices.get(m[1]).add(idx);
                } else {
                    blockIndices.add(idx);
                }
            }
        }
        // When an arch definition is loaded, honor its declared block_count
        // even if some indices have no tensors (those produce empty blocks
        // and get skipped). Otherwise iterate only discovered indices.
        const expandIndices = (discovered) => {
            if (archDef && this._blockCount > 0) {
                const out = [];
                for (let i = 0; i < this._blockCount; i++) {
                    out.push(i);
                }
                return out;
            }
            return Array.from(discovered).sort((a, b) => a - b);
        };
        // Common globals across architectures (also used as the fallback when
        // no arch metadata is loaded). Arch-specific entries from
        // gguf-metadata.json's `graph.input` / `graph.output` are unioned in.
        const globalPrefixes = new Set(['token_embd', 'token_types', 'token_embd_norm', 'position_embd', 'rope_freqs']);
        const outputPrefixes = new Set(['output_norm', 'output']);
        const collectNames = (set, section) => {
            if (section) {
                for (const entry of section) {
                    set.add(entry.name);
                }
            }
        };
        if (archDef && archDef.graph) {
            collectNames(globalPrefixes, archDef.graph.input);
            collectNames(outputPrefixes, archDef.graph.output);
            for (const sub of [archDef.graph.encoder, archDef.graph.decoder]) {
                if (sub) {
                    collectNames(globalPrefixes, sub.input);
                    collectNames(outputPrefixes, sub.output);
                }
            }
        }
        // Section builder: global inputs, structured blocks, global outputs,
        // optionally prefixed (for T5 enc/dec subgraphs).
        const buildSection = (prefix, blockType, indices) => {
            const fullPrefix = (name) => prefix ? `${prefix}.${name}` : name;
            for (const name of globalPrefixes) {
                const weights = collectWeights(fullPrefix(name));
                if (weights.size > 0) {
                    pushFlat(fullPrefix(name), weights);
                }
            }
            for (const i of indices) {
                const blockPrefix = fullPrefix(`blk.${i}`);
                const blockLayers = buildBlockLayers(blockPrefix);
                if (blockLayers.length > 0) {
                    layers.push({ name: blockPrefix, type: blockType, layers: blockLayers, metadata: new Map(), weights: new Map() });
                }
            }
            for (const name of outputPrefixes) {
                const weights = collectWeights(fullPrefix(name));
                if (weights.size > 0) {
                    pushFlat(fullPrefix(name), weights);
                }
            }
        };
        const archName = this._architecture;
        buildSection('', archName, expandIndices(blockIndices));
        for (const [encPrefix, label] of [['enc', 'Encoder'], ['dec', 'Decoder']]) {
            const subgraph = archDef && archDef.graph ? archDef.graph[encPrefix === 'enc' ? 'encoder' : 'decoder'] : null;
            const indices = expandIndices(encDecIndices.get(encPrefix));
            if (subgraph || indices.length > 0) {
                buildSection(encPrefix, `${archName} ${label}`, indices);
            }
        }
        // Flush unclaimed tensors as flat 'weights' nodes, grouping by tensor key.
        for (const [name, tensor] of tensors) {
            if (!claimed.has(name)) {
                const parts = name.split('.');
                const param = parts.pop();
                const key = parts.join('.');
                const existing = layers.find((l) => l.name === key && l.type === 'weights');
                if (existing) {
                    existing.weights.set(param, tensor);
                } else {
                    layers.push({ name: key || name, type: 'weights', metadata: new Map(), weights: new Map([[param, tensor]]), layers: [] });
                }
            }
        }
        return layers;
    }

    _classifyTensor(name) {
        if (!gguf.Context._componentGroups) {
            gguf.Context._componentGroups = [
                { match: /^attn_norm/, group: 'attn_norm' },
                { match: /^attn_q_norm/, group: 'attn_norm' },
                { match: /^attn_k_norm/, group: 'attn_norm' },
                { match: /^attn_sub_norm/, group: 'attn_norm' },
                { match: /^attn_q_a/, group: 'attention' },
                { match: /^attn_q_b/, group: 'attention' },
                { match: /^attn_kv_a/, group: 'attention' },
                { match: /^attn_kv_b/, group: 'attention' },
                { match: /^attn_k_b/, group: 'attention' },
                { match: /^attn_qkv/, group: 'attention' },
                { match: /^attn_q/, group: 'attention' },
                { match: /^attn_k/, group: 'attention' },
                { match: /^attn_v/, group: 'attention' },
                { match: /^attn_output/, group: 'attention' },
                { match: /^attn_out/, group: 'attention' },
                { match: /^attn_o\./, group: 'attention' },
                { match: /^attn_rel_b/, group: 'attention' },
                { match: /^attn_sinks/, group: 'attention' },
                { match: /^attn_rot_embd/, group: 'attention' },
                { match: /^attn_post_norm/, group: 'attn_post_norm' },
                { match: /^cross_attn_norm/, group: 'cross_attn_norm' },
                { match: /^cross_attn_/, group: 'cross_attention' },
                { match: /^ffn_norm/, group: 'ffn_norm' },
                { match: /^ffn_gate_inp/, group: 'ffn_gate_inp' },
                { match: /^ffn_exp_probs/, group: 'ffn_gate_inp' },
                { match: /^ffn_gate_exps/, group: 'ffn_gate_exps' },
                { match: /^ffn_gate\.\d+/, group: 'ffn_gate_exps' },
                { match: /^ffn_gate_up_exps/, group: 'ffn_gate_up_exps' },
                { match: /^ffn_up_exps/, group: 'ffn_up_exps' },
                { match: /^ffn_up\.\d+/, group: 'ffn_up_exps' },
                { match: /^ffn_down_exps/, group: 'ffn_down_exps' },
                { match: /^ffn_down\.\d+/, group: 'ffn_down_exps' },
                { match: /^ffn_gate_shexp/, group: 'ffn_gate_shexp' },
                { match: /^ffn_up_shexp/, group: 'ffn_up_shexp' },
                { match: /^ffn_down_shexp/, group: 'ffn_down_shexp' },
                { match: /^ffn_gate/, group: 'ffn_gate' },
                { match: /^ffn_up/, group: 'ffn_up' },
                { match: /^ffn_down/, group: 'ffn_down' },
                { match: /^ffn_act/, group: 'ffn_act' },
                { match: /^ffn_sub_norm/, group: 'ffn_sub_norm' },
                { match: /^ffn_post_norm/, group: 'ffn_post_norm' },
                { match: /^pre_ffw_norm_2/, group: 'ffn_pre_norm_2' },
                { match: /^post_ffw_norm_1/, group: 'ffn_post_norm_1' },
                { match: /^post_ffw_norm_2/, group: 'ffn_post_norm_2' },
                { match: /^post_ffw_norm/, group: 'ffn_post_norm' },
                { match: /^ssm_/, group: 'ssm' },
                { match: /^time_mix_/, group: 'time_mix' },
                { match: /^channel_mix_/, group: 'channel_mix' },
                { match: /^layer_output_norm/, group: 'layer_output_norm' },
                { match: /^layer_output_scale/, group: 'layer_out_scale' },
                { match: /^attn_output_norm/, group: 'attn_output_norm' },
                { match: /^post_attention_norm/, group: 'attn_post_norm' },
                { match: /^inp_gate/, group: 'inp_gate' },
                { match: /^proj\./, group: 'proj' },
                { match: /^post_norm/, group: 'post_norm' }
            ];
        }
        for (const rule of gguf.Context._componentGroups) {
            if (rule.match.test(name)) {
                return rule.group;
            }
        }
        return 'other';
    }
};

gguf.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'GGML Error';
    }
};

export const ModelFactory = gguf.ModelFactory;
