
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
        const metadata = await context.asset('gguf-metadata.json');
        const entries = JSON.parse(metadata);
        const schemas = new Map(entries.map((entry) => [entry.name, entry]));
        const target = context.value;
        await target.read();
        return new gguf.Model(schemas, target);
    }
};

gguf.Model = class {

    constructor(schemas, target) {
        this.format = target.format;
        this.metadata = [];
        const metadata = new Map();
        let architecture = '?';
        for (const [name, entry] of target.metadata) {
            switch (name) {
                case 'general.name': {
                    this.name = entry.value;
                    break;
                }
                case 'general.architecture': {
                    architecture = entry.value;
                    break;
                }
                case 'general.description': {
                    this.description = entry.value;
                    break;
                }
                default: {
                    const path = name.split('.');
                    if (path[0] === 'general') {
                        const argument = new gguf.Argument(path.pop(), entry.value, entry.type);
                        this.metadata.push(argument);
                    } else {
                        metadata.set(entry.name, entry);
                    }
                    break;
                }
            }
        }
        const tokenizer = { type: 'tokenizer', metadata: new Map(), layers: [] };
        const graph = {};
        graph.type = architecture;
        graph.metadata = [];
        for (const [name, entry] of metadata) {
            const path = name.split('.');
            if (path[0] === 'tokenizer') {
                const match = entry.name.match(/^(.*)\.(.*?)$/);
                if (match) {
                    const [, param] = match.slice(1);
                    tokenizer.metadata.set(param, entry);
                }
            } else if (architecture !== '?' && path[0] === architecture) {
                const name = path.slice(1).join('.');
                const argument = new gguf.Argument(name, entry.value, entry.type);
                graph.metadata.push(argument);
            } else {
                const argument = new gguf.Argument(entry.name, entry.value, entry.type);
                this.metadata.push(argument);
            }
        }
        const schema = schemas.get(architecture);
        const context = new gguf.Context(schema, target, metadata, architecture);
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
        this.metadata = graph.metadata || [];
        let valueIndex = 0;
        let prevValue = null;
        let ropeFreqsValue = null;
        let perLayerInputValue = null;
        const perLayerOutputs = {};
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
                        let g1 = newValue();
                        addNode(use('ffn_gate_inp'), [moeInput], g1);
                        // Expert routing bias (deepseek/step/bailing MoE): added to the
                        // router logits before top-k selection.
                        if (has('exp_probs_b')) {
                            const biased = newValue();
                            addNode(use('exp_probs_b'), [g1], biased);
                            g1 = biased;
                        }
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
                    if (has('inp_gate') && has('proj') && has('post_norm')) {
                        const peIn = final;
                        const g = newValue();
                        addNode(use('inp_gate'), [peIn], g);
                        const gAct = newValue();
                        addOp('GELU', [g], gAct);
                        let gated = gAct;
                        if (perLayerInputValue) {
                            gated = newValue();
                            addOp('MUL', [gAct, perLayerInputValue], gated);
                        }
                        const p = newValue();
                        addNode(use('proj'), [gated], p);
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
                    let preAdd2 = f1;
                    if (has('ffn_post_norm')) {
                        const pn = newValue();
                        addNode(use('ffn_post_norm'), [f1], pn);
                        preAdd2 = pn;
                    }
                    const r2 = newValue();
                    addOp('ADD', [preAdd2, r1], r2);
                    prevValue = r2;
                } else {
                    // Fallback: unrecognized block shape, render components without
                    // fabricating sequential data-flow edges.
                    for (const item of layer.layers) {
                        this.nodes.push(new gguf.Node(item));
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
            } else if (layer.name === 'per_layer_token_embd' || layer.name === 'per_layer_model_proj' || layer.name === 'per_layer_proj_norm') {
                // Gemma 3n / 4 per-layer-embedding precomputation (gemma4.cpp:431-451):
                //   per_layer_proj  = per_layer_model_proj * token_embd_output
                //   per_layer_proj  = norm(per_layer_proj, per_layer_proj_norm)
                //   inp_per_layer   = per_layer_proj + per_layer_token_embd
                // The result fans out into each block's per-layer-embedding gating mul.
                const node = new gguf.Node(layer);
                const out = newValue();
                if (layer.name === 'per_layer_model_proj' && prevValue) {
                    node.inputs.unshift(new gguf.Argument('input', [prevValue]));
                } else if (layer.name === 'per_layer_proj_norm' && perLayerOutputs.per_layer_model_proj) {
                    node.inputs.unshift(new gguf.Argument('input', [perLayerOutputs.per_layer_model_proj]));
                }
                node.outputs.push(new gguf.Argument('output', [out]));
                this.nodes.push(node);
                perLayerOutputs[layer.name] = out;
                if (perLayerOutputs.per_layer_token_embd && perLayerOutputs.per_layer_proj_norm && !perLayerInputValue) {
                    perLayerInputValue = newValue();
                    addOp('ADD', [perLayerOutputs.per_layer_proj_norm, perLayerOutputs.per_layer_token_embd], perLayerInputValue);
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

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
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
            for (const [name, entry] of layer.metadata) {
                const attribute = new gguf.Argument(name, entry.value, entry.type);
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
                const entry = reader.value();
                this.metadata.set(entry.name, entry);
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

    scalar(type) {
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
            default: throw new gguf.Error(`Unsupported GGUF type '${type}'.`);
        }
    }

    type(type) {
        switch (type) {
            case gguf.Type.UINT8: return 'uint8';
            case gguf.Type.INT8: return 'int8';
            case gguf.Type.UINT16: return 'uint16';
            case gguf.Type.INT16: return 'int16';
            case gguf.Type.UINT32: return 'uint32';
            case gguf.Type.INT32: return 'int32';
            case gguf.Type.UINT64: return 'uint64';
            case gguf.Type.INT64: return 'int64';
            case gguf.Type.FLOAT32: return 'float32';
            case gguf.Type.FLOAT64: return 'float64';
            case gguf.Type.BOOL: return 'boolean';
            case gguf.Type.STRING: return 'string';
            default: throw new gguf.Error(`Unsupported GGUF type '${type}'.`);
        }
    }

    value() {
        const name = this.string();
        const type = this.uint32();
        if (type === gguf.Type.ARRAY) {
            const elementType = this.uint32();
            const size = this.uint64().toNumber();
            const value = new Array(size);
            for (let i = 0; i < size; i++) {
                value[i] = this.scalar(elementType);
            }
            return { name, value, type: `${this.type(elementType)}[]` };
        }
        return { name, value: this.scalar(type), type: this.type(type) };
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
    /* 39 */ { name: 'mxfp4',      block_size: 32,  type_size: 1 + 16 },
    /* 40 */ { name: 'nvfp4',      block_size: 64,  type_size: 4 + 32 },
    /* 41 */ { name: 'q1_0',       block_size: 128, type_size: 2 + 16 }
];

gguf.Context = class {

    constructor(schema, target, metadata, architecture) {
        this._schema = schema;
        this._tensors = target.tensors;
        this._architecture = architecture;
        this._metadata = metadata;
        const blockCountEntry = metadata.get(`${architecture}.block_count`);
        this._blockCount = (blockCountEntry && blockCountEntry.value) || 0;
        this._blockTypes = new Map();
        // Classifier rules are derived from this arch's `blocks` entries:
        // each entry's `name` and its `tensors` aliases are prefixes that route
        // to the entry's `name` as the group label. Matching is strict
        // prefix-with-dot-boundary so e.g. `attn_o` matches `attn_o.weight` but
        // not `attn_output.weight`.
        this._classifierRules = [];
        if (schema && schema.graph) {
            const registerSection = (section, classify, sectionPrefix) => {
                if (section) {
                    for (const block of section) {
                        this._blockTypes.set(block.name, block);
                        if (block.tensors) {
                            for (const tensor of block.tensors) {
                                this._blockTypes.set(tensor, block);
                            }
                        }
                        if (classify) {
                            // T5-style encoder/decoder blocks register the implicit
                            // entry.name pattern under the section prefix (e.g.
                            // `enc.attn_norm`); curator-supplied `tensors` aliases
                            // already carry the prefix and pass through verbatim.
                            this._classifierRules.push({ pattern: `${sectionPrefix}${block.name}`, group: block.name });
                            for (const tensor of block.tensors || []) {
                                this._classifierRules.push({ pattern: tensor, group: block.name });
                            }
                        }
                    }
                }
            };
            registerSection(schema.graph.input, false, '');
            registerSection(schema.graph.blocks, true, '');
            registerSection(schema.graph.output, false, '');
            if (schema.graph.encoder) {
                registerSection(schema.graph.encoder.input, false, 'enc.');
                registerSection(schema.graph.encoder.blocks, true, 'enc.');
                registerSection(schema.graph.encoder.output, false, 'enc.');
            }
            if (schema.graph.decoder) {
                registerSection(schema.graph.decoder.input, false, 'dec.');
                registerSection(schema.graph.decoder.blocks, true, 'dec.');
                registerSection(schema.graph.decoder.output, false, 'dec.');
            }
            // Longest-pattern-first ensures specific aliases (e.g. ffn_gate.{N},
            // attn_q_norm) win over their shorter prefixes (ffn_gate, attn_q).
            this._classifierRules.sort((a, b) => b.pattern.length - a.pattern.length);
        }
    }

    get structured() {
        return this._schema !== null && this._tensors.size > 0;
    }

    build() {
        const tensors = this._tensors;
        const layers = [];
        const claimed = new Set();
        const schema = this._schema;
        const collectWeights = (prefix) => {
            const weights = new Map();
            for (const [name, tensor] of tensors) {
                if (name.startsWith(`${prefix}.`) || name === prefix) {
                    const suffix = name.slice(prefix.length + 1) || name;
                    weights.set(suffix, tensor);
                    claimed.add(name);
                }
            }
            return weights;
        };
        // Resolve display type/category for a component group from metadata
        // (when an arch definition is loaded), otherwise default to 'weights'.
        // Per-node attributes are resolved from the entry's `attributes` list
        // by looking up `<arch>.<key>` in the model KV (e.g. an `attention`
        // entry listing `attention.head_count` pulls that KV onto every node).
        const resolveBlock = (group) => {
            let block = this._blockTypes.get(group);
            if (!block && (group.startsWith('enc.') || group.startsWith('dec.'))) {
                // T5 enc/dec output sections register block names bare
                // (e.g. `output_norm`), but pushFlat passes the section-prefixed
                // tensor key (`enc.output_norm`). Strip the prefix on lookup.
                block = this._blockTypes.get(group.slice(4));
            }
            if (!block) {
                return { type: 'weights', metadata: new Map() };
            }
            // Explicit `attributes` on the entry wins; otherwise fall back to
            // the type-default list. `attributes: []` opts a component out of
            // per-node KV synthesis. Each entry is `{ key, name }` (display
            // label) or a bare string (label derived from the key's last segment).
            const entries = Array.isArray(block.attributes) ? block.attributes : (gguf.Context._typeAttributes.get(block.type) || []);
            const metadata = new Map();
            for (const entry of entries) {
                const label = entry.name;
                const key = `${this._architecture}.${entry.key}`;
                if (this._metadata.has(key)) {
                    metadata.set(label, this._metadata.get(key));
                }
            }
            return { type: block.type || 'weights', category: block.category, metadata };
        };
        const pushFlat = (prefix, weights) => {
            const resolved = resolveBlock(prefix);
            layers.push({ name: prefix, type: resolved.type, category: resolved.category, weights, metadata: resolved.metadata, layers: [] });
        };
        // Build a structured block at `blockPrefix`, returning sub-layers in
        // discovery order. Tensors in the block are grouped by component
        // (attn, ffn, ...) via _classifyTensor.
        const buildBlockLayers = (blockPrefix) => {
            // For T5-style encoder/decoder blocks (`enc.blk.N` / `dec.blk.N`),
            // metadata aliases preserve the `enc.`/`dec.` segment (e.g.
            // `enc.attn_q`) to disambiguate the two subgraphs. Prepend it back
            // onto the bare tensor name before classifying.
            let sectionPrefix = '';
            if (blockPrefix.startsWith('enc.')) {
                sectionPrefix = 'enc.';
            } else if (blockPrefix.startsWith('dec.')) {
                sectionPrefix = 'dec.';
            }
            const groups = new Map();
            const order = [];
            for (const [name] of tensors) {
                if (name.startsWith(`${blockPrefix}.`)) {
                    const rest = name.slice(blockPrefix.length + 1);
                    const group = this._classifyTensor(`${sectionPrefix}${rest}`);
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
                    blockLayers.push({ name: group, type: resolved.type, category: resolved.category, weights, metadata: resolved.metadata, layers: [] });
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
            if (schema && this._blockCount > 0) {
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
        // Output prefixes follow metadata declaration order; the hardcoded
        // defaults are appended last as the no-metadata fallback.
        const outputPrefixes = new Set();
        const collectNames = (set, section) => {
            if (section) {
                for (const entry of section) {
                    set.add(entry.name);
                }
            }
        };
        if (schema && schema.graph) {
            collectNames(globalPrefixes, schema.graph.input);
            collectNames(outputPrefixes, schema.graph.output);
            for (const sub of [schema.graph.encoder, schema.graph.decoder]) {
                if (sub) {
                    collectNames(globalPrefixes, sub.input);
                    collectNames(outputPrefixes, sub.output);
                }
            }
        }
        outputPrefixes.add('output_norm');
        outputPrefixes.add('output');
        if (schema && schema.graph) {
            // An explicit `output` placement wins over the hardcoded global
            // defaults (e.g. LFM2 stores its final norm as `token_embd_norm`).
            for (const name of outputPrefixes) {
                globalPrefixes.delete(name);
            }
        }
        // Section builder phases — inputs/blocks/outputs are split so encoder-decoder
        // archs can defer global outputs until after enc/dec sections.
        const fullPrefix = (prefix, name) => prefix ? `${prefix}.${name}` : name;
        const sectionFlat = (prefix, names) => {
            for (const name of names) {
                const key = fullPrefix(prefix, name);
                const weights = collectWeights(key);
                if (weights.size > 0) {
                    pushFlat(key, weights);
                }
            }
        };
        const sectionBlocks = (prefix, blockType, indices) => {
            for (const i of indices) {
                const blockPrefix = fullPrefix(prefix, `blk.${i}`);
                const blockLayers = buildBlockLayers(blockPrefix);
                if (blockLayers.length > 0) {
                    layers.push({ name: blockPrefix, type: blockType, layers: blockLayers, metadata: new Map(), weights: new Map() });
                }
            }
        };
        const archName = this._architecture;
        const subgraphs = [];
        for (const [encPrefix, label] of [['enc', 'Encoder'], ['dec', 'Decoder']]) {
            const subgraph = schema && schema.graph ? schema.graph[encPrefix === 'enc' ? 'encoder' : 'decoder'] : null;
            const indices = expandIndices(encDecIndices.get(encPrefix));
            if (subgraph || indices.length > 0) {
                subgraphs.push({ prefix: encPrefix, label, indices });
            }
        }
        sectionFlat('', globalPrefixes);
        sectionBlocks('', archName, expandIndices(blockIndices));
        for (const sg of subgraphs) {
            sectionFlat(sg.prefix, globalPrefixes);
            sectionBlocks(sg.prefix, `${archName} ${sg.label}`, sg.indices);
            sectionFlat(sg.prefix, outputPrefixes);
        }
        sectionFlat('', outputPrefixes);
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
        for (const rule of this._classifierRules) {
            const pattern = rule.pattern;
            if (pattern.includes('{N}')) {
                let regex = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                regex = regex.replace(/\\\{N\\\}/g, '\\d+');
                if (new RegExp(`^${regex}(\\.|$)`).test(name)) {
                    return rule.group;
                }
            } else if (name === pattern || name.startsWith(`${pattern}.`)) {
                return rule.group;
            }
        }
        return 'other';
    }
};

// Per-node attribute defaults keyed by node type. A component's `attributes`
// field in `gguf-metadata.json` overrides this; absence falls through to
// these defaults. Only KV keys actually present in the file resolve to values.
gguf.Context._typeAttributes = new Map([
    ['RMS_NORM',                [{ key: 'attention.layer_norm_rms_epsilon', name: 'epsilon' }]],
    ['LAYER_NORM',              [{ key: 'attention.layer_norm_epsilon',     name: 'epsilon' }]],
    ['MULTI_HEAD_ATTENTION',    [
        { key: 'attention.head_count',     name: 'head_count' },
        { key: 'attention.head_count_kv',  name: 'head_count_kv' },
        { key: 'attention.key_length',     name: 'key_length' },
        { key: 'attention.value_length',   name: 'value_length' },
        { key: 'attention.sliding_window', name: 'sliding_window' }
    ]],
    ['MULTI_LATENT_ATTENTION',  [
        { key: 'attention.head_count',       name: 'head_count' },
        { key: 'attention.head_count_kv',    name: 'head_count_kv' },
        { key: 'attention.q_lora_rank',      name: 'q_lora_rank' },
        { key: 'attention.kv_lora_rank',     name: 'kv_lora_rank' },
        { key: 'attention.key_length_mla',   name: 'key_length_mla' },
        { key: 'attention.value_length_mla', name: 'value_length_mla' }
    ]],
    ['CROSS_ATTENTION',         [
        { key: 'attention.head_count',    name: 'head_count' },
        { key: 'attention.head_count_kv', name: 'head_count_kv' },
        { key: 'attention.key_length',    name: 'key_length' },
        { key: 'attention.value_length',  name: 'value_length' }
    ]],
    ['ROPE_FREQS',              [
        { key: 'rope.dimension_count',                name: 'dimension_count' },
        { key: 'rope.freq_base',                      name: 'freq_base' },
        { key: 'rope.scaling.type',                   name: 'scaling_type' },
        { key: 'rope.scaling.factor',                 name: 'scaling_factor' },
        { key: 'rope.scaling.original_context_length', name: 'original_context_length' }
    ]],
    ['MAMBA',                   [
        { key: 'ssm.state_size',     name: 'state_size' },
        { key: 'ssm.conv_kernel',    name: 'conv_kernel' },
        { key: 'ssm.inner_size',     name: 'inner_size' },
        { key: 'ssm.time_step_rank', name: 'time_step_rank' }
    ]],
    ['MAMBA2',                  [
        { key: 'ssm.state_size',  name: 'state_size' },
        { key: 'ssm.conv_kernel', name: 'conv_kernel' },
        { key: 'ssm.inner_size',  name: 'inner_size' },
        { key: 'ssm.group_count', name: 'group_count' }
    ]],
    ['CONV_1D',                 [{ key: 'shortconv.l_cache', name: 'l_cache' }]]
]);

gguf.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'GGML Error';
    }
};

export const ModelFactory = gguf.ModelFactory;
