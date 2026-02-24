
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
                const hasFfn = has('ffn_up') || hasMoe;
                const buildLinearFfn = (input, gateKey, upKey, downKey) => {
                    const inputs = [input];
                    if (has(gateKey)) {
                        const g = newValue();
                        addNode(use(gateKey), [input], g);
                        inputs.push(g);
                    }
                    const u = newValue();
                    addNode(use(upKey), [input], u);
                    inputs.push(u);
                    const d = newValue();
                    addNode(use(downKey), inputs, d);
                    return d;
                };
                const buildFfn = (input) => {
                    if (hasMoe) {
                        const g1 = newValue();
                        addNode(use('ffn_gate_inp'), [input], g1);
                        const moeOut = buildLinearFfn(g1, 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps');
                        if (has('ffn_up_shexp')) {
                            const sharedOut = buildLinearFfn(input, 'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp');
                            const sum = newValue();
                            addOp('ADD', [moeOut, sharedOut], sum);
                            return sum;
                        }
                        return moeOut;
                    }
                    return buildLinearFfn(input, 'ffn_gate', 'ffn_up', 'ffn_down');
                };
                if (has('attn_norm') && has('attention') && !has('ffn_norm') && !has('attn_post_norm') && hasFfn) {
                    // Parallel attention + FFN (phi-2, falcon)
                    const inp = prevValue || newValue();
                    const normOut = newValue();
                    const attnOut = newValue();
                    const out = newValue();
                    addNode(use('attn_norm'), [inp], normOut);
                    addNode(use('attention'), [normOut], attnOut);
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
                    addNode(use('attention'), [n1], a1);
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
                    addNode(use('attention'), [n1], a1);
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
                    prevValue = r2;
                } else if (has('attention') && (has('attn_output_norm') || has('layer_output_norm'))) {
                    // Post-norm (BERT)
                    const inp = prevValue || newValue();
                    const a1 = newValue();
                    addNode(use('attention'), [inp], a1);
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
                    addNode(use('attention'), [n1], a1);
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
        if (tensor.type !== gguf.QuantizationType.F32 && tensor.type !== gguf.QuantizationType.F16) {
            this.quantization = {
                type: gguf.Utility.enum(gguf.QuantizationType, tensor.type).toLowerCase()
            };
        }
        if (tensor.dtype === 'float32' || tensor.dtype === 'float16' ||
            tensor.dtype === 'int8' || tensor.dtype === 'int16' || tensor.dtype === 'int32') {
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
        const QK_K = 256;
        // https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py
        gguf.Reader.GGML_QUANT_SIZES = gguf.Reader.GGML_QUANT_SIZES || new Map([
            [gguf.QuantizationType.F32,        [1, 4, 'float32']],
            [gguf.QuantizationType.F16,        [1, 2, 'float16']],
            [gguf.QuantizationType.Q4_0,       [32, 2 + 16, 'q4_0']],
            [gguf.QuantizationType.Q4_1,       [32, 2 + 2 + 16, 'q4_1']],
            [gguf.QuantizationType.Q4_2,       [16, 2 + 8, 'q4_2']],
            [gguf.QuantizationType.Q4_3,       [16, 2 + 2 + 8, 'q4_3']],
            [gguf.QuantizationType.Q5_0,       [32, 2 + 4 + 16, 'q5_0']],
            [gguf.QuantizationType.Q5_1,       [32, 2 + 2 + 4 + 16, 'q5_1']],
            [gguf.QuantizationType.Q8_0,       [32, 2 + 32, 'q8_0']],
            [gguf.QuantizationType.Q8_1,       [32, 4 + 4 + 32, 'q8_1']],
            [gguf.QuantizationType.Q2_K,       [256, 2 + 2 + Math.floor(QK_K / 16) + Math.floor(QK_K / 4), 'q2_K']],
            [gguf.QuantizationType.Q3_K,       [256, 2 + Math.floor(QK_K / 4) + Math.floor(QK_K / 8) + 12, 'q3_K']],
            [gguf.QuantizationType.Q4_K,       [256, 2 + 2 + Math.floor(QK_K / 2) + 12, 'q4_K']],
            [gguf.QuantizationType.Q5_K,       [256, 2 + 2 + Math.floor(QK_K / 2) + Math.floor(QK_K / 8) + 12, 'q5_K']],
            [gguf.QuantizationType.Q6_K,       [256, 2 + Math.floor(QK_K / 2) + Math.floor(QK_K / 4) + Math.floor(QK_K / 16), 'q6_K']],
            [gguf.QuantizationType.Q8_K,       [256, 4 + QK_K + Math.floor(QK_K / 8), 'q8_K']],
            [gguf.QuantizationType.IQ2_XXS,    [256, 2 + Math.floor(QK_K / 4), 'iq2_xxs']],
            [gguf.QuantizationType.IQ2_XS,     [256, 2 + Math.floor(QK_K / 4) + Math.floor(QK_K / 32), 'iq2_xs']],
            [gguf.QuantizationType.IQ3_XXS,    [256, 2 + Math.floor(QK_K / 4) + Math.floor(QK_K / 8), 'iq3_xxs']],
            [gguf.QuantizationType.IQ1_S,      [256, 2 + Math.floor(QK_K / 8) + Math.floor(QK_K / 16), 'iq1_s']],
            [gguf.QuantizationType.IQ4_NL,     [32, 2 + 16, 'iq4_nl']],
            [gguf.QuantizationType.IQ3_S,      [256, 2 + Math.floor(QK_K / 4) + Math.floor(QK_K / 8) + Math.floor(QK_K / 32) + 4, 'iq3_s']],
            [gguf.QuantizationType.IQ2_S,      [256, 2 + Math.floor(QK_K / 4) + Math.floor(QK_K / 16), 'iq2_s']],
            [gguf.QuantizationType.IQ4_XS,     [256, 2 + 2 + Math.floor(QK_K / 2) + Math.floor(QK_K / 64), 'iq4_xs']],
            [gguf.QuantizationType.I8,         [1, 1, 'int8']],
            [gguf.QuantizationType.I16,        [1, 2, 'int16']],
            [gguf.QuantizationType.I32,        [1, 4, 'int32']],
            [gguf.QuantizationType.I64,        [1, 8, 'int64']],
            [gguf.QuantizationType.F64,        [1, 8, 'float64']],
            [gguf.QuantizationType.IQ1_M,      [256, Math.floor(QK_K / 8) + Math.floor(QK_K / 16)  + Math.floor(QK_K / 32), 'iq1_m']],
            [gguf.QuantizationType.BF16,       [1, 2, 'bfloat16']],
            [gguf.QuantizationType.Q4_0_4_4,   [32, 2 + 16, 'q4_0_4_4']],
            [gguf.QuantizationType.Q4_0_4_8,   [32, 2 + 16, 'q4_0_4_8']],
            [gguf.QuantizationType.Q4_0_8_8,   [32, 2 + 16, 'q4_0_8_8']],
            [gguf.QuantizationType.TQ1_0,      [256, 2 + 4 * 13, 'tq1_0']],
            [gguf.QuantizationType.TQ2_0,      [256, 2 + 64, 'tq2_0']],
            [gguf.QuantizationType.IQ4_NL_4_4, [32, 2 + 16, 'iq4_nl_4_4']],
            [gguf.QuantizationType.IQ4_NL_4_8, [32, 2 + 16, 'iq4_nl_4_8']],
            [gguf.QuantizationType.IQ4_NL_8_8, [32, 2 + 16, 'iq4_nl_8_8']],
            [gguf.QuantizationType.MXFP4,      [32, 1 + 16, 'mxfp4']]
        ]);
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
                    if (!gguf.Reader.GGML_QUANT_SIZES.has(tensor.type)) {
                        throw new gguf.Error(`Unsupported tensor quantization type '${tensor.type}'.`);
                    }
                    const [block_size, type_size, dtype] = gguf.Reader.GGML_QUANT_SIZES.get(tensor.type);
                    tensor.block_size = block_size;
                    tensor.type_size = type_size;
                    tensor.dtype = dtype || '?';
                    if (offset < reader.length) {
                        const n_elems = tensor.ne.reduce((a, b) => a * b, 1);
                        const n_bytes = Math.floor((n_elems * type_size) / block_size);
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

// https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
gguf.QuantizationType = {
    F32: 0,
    F16: 1,
    Q4_0: 2,
    Q4_1: 3,
    Q4_2: 4, // deprecated
    Q4_3: 5, // deprecated
    Q5_0: 6,
    Q5_1: 7,
    Q8_0: 8,
    Q8_1: 9,
    Q2_K: 10,
    Q3_K: 11,
    Q4_K: 12,
    Q5_K: 13,
    Q6_K: 14,
    Q8_K: 15,
    IQ2_XXS: 16,
    IQ2_XS: 17,
    IQ3_XXS: 18,
    IQ1_S: 19,
    IQ4_NL: 20,
    IQ3_S: 21,
    IQ2_S: 22,
    IQ4_XS: 23,
    I8: 24,
    I16: 25,
    I32: 26,
    I64: 27,
    F64: 28,
    IQ1_M: 29,
    BF16: 30,
    Q4_0_4_4: 31, // deprecated
    Q4_0_4_8: 32, // deprecated
    Q4_0_8_8: 33, // deprecated
    TQ1_0: 34,
    TQ2_0: 35,
    IQ4_NL_4_4: 36, // deprecated
    IQ4_NL_4_8: 37, // deprecated
    IQ4_NL_8_8: 38, // deprecated
    MXFP4: 39
};

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
        if (!this._archDef || this._tensors.size === 0) {
            return this._buildFlat();
        }
        const tensors = this._tensors;
        const layers = [];
        const claimed = new Set();
        // Collect tensors matching a prefix into a weights map
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
        // Classify tensor prefix into a semantic component group
        const classifyTensor = (name) => this._classifyTensor(name);
        // Resolve display type and category for a component group using metadata
        const resolveBlock = (group) => {
            const block = this._blockTypes.get(group);
            if (block) {
                return { type: block.type, category: block.category };
            }
            return { type: 'weights' };
        };
        // Build global (non-block) input tensors
        const globalPrefixes = ['token_embd', 'token_types', 'token_embd_norm', 'position_embd', 'rope_freqs'];
        for (const prefix of globalPrefixes) {
            const weights = collectWeights(prefix);
            if (weights.size > 0) {
                const resolved = resolveBlock(prefix);
                layers.push({ name: prefix, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
            }
        }
        // Build block sub-graphs
        for (let i = 0; i < this._blockCount; i++) {
            const blockPrefix = `blk.${i}`;
            // Collect all tensors in this block and group by component
            const groups = new Map();
            const order = [];
            for (const [name] of tensors) {
                if (name.startsWith(`${blockPrefix}.`)) {
                    const rest = name.slice(blockPrefix.length + 1);
                    const group = classifyTensor(rest);
                    if (!groups.has(group)) {
                        groups.set(group, new Map());
                        order.push(group);
                    }
                    groups.get(group).set(rest, name);
                }
            }
            // Build sub-nodes for each component group in discovery order
            const blockLayers = [];
            for (const group of order) {
                const tensorMap = groups.get(group);
                const weights = new Map();
                for (const [suffix, fullName] of tensorMap) {
                    const tensor = tensors.get(fullName);
                    if (tensor) {
                        weights.set(suffix, tensor);
                        claimed.add(fullName);
                    }
                }
                if (weights.size > 0) {
                    const resolved = resolveBlock(group);
                    blockLayers.push({ name: group, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
                }
            }
            if (blockLayers.length > 0) {
                layers.push({ name: `blk.${i}`, type: this._architecture, layers: blockLayers, metadata: new Map(), weights: new Map() });
            }
        }
        // Build global output tensors
        const outputPrefixes = ['output_norm', 'output'];
        for (const prefix of outputPrefixes) {
            const weights = collectWeights(prefix);
            if (weights.size > 0) {
                const resolved = resolveBlock(prefix);
                layers.push({ name: prefix, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
            }
        }
        // Build encoder/decoder sub-graphs for T5-style models
        for (const [encPrefix, label] of [['enc', 'Encoder'], ['dec', 'Decoder']]) {
            const graph = this._archDef.graph;
            const subgraph = graph ? graph[encPrefix === 'enc' ? 'encoder' : 'decoder'] : null;
            if (subgraph) {
                for (const prefix of globalPrefixes) {
                    const fullPrefix = `${encPrefix}.${prefix}`;
                    const weights = collectWeights(fullPrefix);
                    if (weights.size > 0) {
                        const resolved = resolveBlock(prefix);
                        layers.push({ name: fullPrefix, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
                    }
                }
            }
            for (let i = 0; i < this._blockCount; i++) {
                const blockPrefix = `${encPrefix}.blk.${i}`;
                const groups = new Map();
                const order = [];
                for (const [name] of tensors) {
                    if (name.startsWith(`${blockPrefix}.`)) {
                        const rest = name.slice(blockPrefix.length + 1);
                        const group = classifyTensor(rest);
                        if (!groups.has(group)) {
                            groups.set(group, new Map());
                            order.push(group);
                        }
                        groups.get(group).set(rest, name);
                    }
                }
                const blockLayers = [];
                for (const group of order) {
                    const tensorMap = groups.get(group);
                    const weights = new Map();
                    for (const [suffix, fullName] of tensorMap) {
                        const tensor = tensors.get(fullName);
                        if (tensor) {
                            weights.set(suffix, tensor);
                            claimed.add(fullName);
                        }
                    }
                    if (weights.size > 0) {
                        const resolved = resolveBlock(group);
                        blockLayers.push({ name: group, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
                    }
                }
                if (blockLayers.length > 0) {
                    layers.push({ name: blockPrefix, type: `${this._architecture} ${label}`, layers: blockLayers, metadata: new Map(), weights: new Map() });
                }
            }
            if (subgraph) {
                for (const prefix of outputPrefixes) {
                    const fullPrefix = `${encPrefix}.${prefix}`;
                    const weights = collectWeights(fullPrefix);
                    if (weights.size > 0) {
                        const resolved = resolveBlock(prefix);
                        layers.push({ name: fullPrefix, type: resolved.type, category: resolved.category, weights, metadata: new Map(), layers: [] });
                    }
                }
            }
        }
        // Collect any unclaimed tensors
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

    _buildFlat() {
        const tensors = this._tensors;
        const layers = [];
        const claimed = new Set();
        const blockPattern = /^blk\.(\d+)\./;
        const encDecPattern = /^(enc|dec)\.blk\.(\d+)\./;
        const blockIndices = new Set();
        const encDecIndices = new Map();
        for (const [name] of tensors) {
            const m = name.match(blockPattern);
            if (m) {
                blockIndices.add(parseInt(m[1], 10));
            }
            const m2 = name.match(encDecPattern);
            if (m2) {
                if (!encDecIndices.has(m2[1])) {
                    encDecIndices.set(m2[1], new Set());
                }
                encDecIndices.get(m2[1]).add(parseInt(m2[2], 10));
            }
        }
        if (blockIndices.size > 0 || encDecIndices.size > 0) {
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
            const globalPrefixes = ['token_embd', 'token_types', 'token_embd_norm', 'position_embd', 'rope_freqs'];
            for (const prefix of globalPrefixes) {
                const weights = collectWeights(prefix);
                if (weights.size > 0) {
                    layers.push({ name: prefix, type: 'weights', metadata: new Map(), weights, layers: [] });
                }
            }
            const buildStructuredBlocks = (blockPrefix) => {
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
                    const tensorMap = groups.get(group);
                    const weights = new Map();
                    for (const [suffix, fullName] of tensorMap) {
                        const tensor = tensors.get(fullName);
                        if (tensor) {
                            weights.set(suffix, tensor);
                            claimed.add(fullName);
                        }
                    }
                    if (weights.size > 0) {
                        blockLayers.push({ name: group, type: 'weights', metadata: new Map(), weights, layers: [] });
                    }
                }
                return blockLayers;
            };
            for (const i of Array.from(blockIndices).sort((a, b) => a - b)) {
                const blockLayers = buildStructuredBlocks(`blk.${i}`);
                if (blockLayers.length > 0) {
                    layers.push({ name: `blk.${i}`, type: this._architecture, layers: blockLayers, metadata: new Map(), weights: new Map() });
                }
            }
            for (const [prefix, indices] of encDecIndices) {
                for (const i of Array.from(indices).sort((a, b) => a - b)) {
                    const blockLayers = buildStructuredBlocks(`${prefix}.blk.${i}`);
                    if (blockLayers.length > 0) {
                        const label = prefix === 'enc' ? 'Encoder' : 'Decoder';
                        layers.push({ name: `${prefix}.blk.${i}`, type: `${this._architecture} ${label}`, layers: blockLayers, metadata: new Map(), weights: new Map() });
                    }
                }
            }
            const outputPrefixes = ['output_norm', 'output'];
            for (const prefix of outputPrefixes) {
                const weights = collectWeights(prefix);
                if (weights.size > 0) {
                    layers.push({ name: prefix, type: 'weights', metadata: new Map(), weights, layers: [] });
                }
            }
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
        const flatLayers = new Map();
        for (const [name, tensor] of tensors) {
            const parts = name.split('.');
            const param = parts.pop();
            const key = parts.join('.');
            if (!flatLayers.has(key)) {
                flatLayers.set(key, { name: key, type: 'weights', metadata: new Map(), weights: new Map() });
            }
            flatLayers.get(key).weights.set(param, tensor);
        }
        return Array.from(flatLayers.values());
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
                { match: /^ssm_/, group: 'ssm' },
                { match: /^time_mix_/, group: 'time_mix' },
                { match: /^channel_mix_/, group: 'channel_mix' },
                { match: /^layer_output_norm/, group: 'layer_output_norm' },
                { match: /^attn_output_norm/, group: 'attn_output_norm' },
                { match: /^post_attention_norm/, group: 'attn_post_norm' }
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

gguf.Utility = class {

    static enum(type, value) {
        gguf.Utility._enums = gguf.Utility._enums || new Map();
        if (!gguf.Utility._enums.has(type)) {
            const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
            gguf.Utility._enums.set(type, entries);
        }
        const entries = gguf.Utility._enums.get(type);
        if (entries.has(value)) {
            return entries.get(value);
        }
        return value;
    }
};

gguf.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'GGML Error';
    }
};

export const ModelFactory = gguf.ModelFactory;
