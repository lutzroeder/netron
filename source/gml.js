
const gml = {};

// https://en.wikipedia.org/wiki/Graph_Modelling_Language

gml.ModelFactory = class {

    async match(context) {
        const reader = await context.read('text', 0x10000);
        if (reader) {
            try {
                for (let i = 0; i < 64; i++) {
                    const line = reader.read('\n');
                    if (line === undefined) {
                        break;
                    }
                    const trimmed = line.trim();
                    if (trimmed.length === 0 || trimmed.startsWith('#')) {
                        continue;
                    }
                    if (/^graph\s*\[$/.test(trimmed)) {
                        return context.set('gml');
                    }
                    if (/^[A-Za-z][A-Za-z0-9_]*\s+("([^"\\]|\\.)*"|[+-]?[\d.]+)$/.test(trimmed)) {
                        continue;
                    }
                    break;
                }
            } catch {
                // continue regardless of error
            }
        }
        return null;
    }

    async open(context) {
        const decoder = await context.read('text.decoder');
        const parser = new gml.Parser(decoder);
        const entries = parser.parse();
        return new gml.Model(entries);
    }
};

gml.Model = class {

    constructor(entries) {
        this.format = 'GML';
        const graphEntry = entries.find((entry) => entry.key === 'graph');
        if (!graphEntry || !Array.isArray(graphEntry.value)) {
            throw new gml.Error("File does not contain a 'graph' section.");
        }
        const creator = entries.find((entry) => entry.key === 'Creator');
        if (creator && typeof creator.value === 'string') {
            this.producer = creator.value;
        }
        const version = entries.find((entry) => entry.key === 'Version');
        if (version && version.value !== undefined && version.value !== null) {
            this.version = version.value.toString();
        }
        this.metadata = [];
        for (const entry of graphEntry.value) {
            if (entry.key !== 'node' && entry.key !== 'edge') {
                const value = Array.isArray(entry.value) ? gml.Utility.map(entry.value) : entry.value;
                this.metadata.push(new gml.Argument(entry.key, value));
            }
        }
        this.modules = [new gml.Graph(graphEntry.value)];
    }
};

gml.Graph = class {

    constructor(entries) {
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        this.groups = true;
        const nodes = new Map();
        const groupLabels = new Map();
        for (const entry of entries) {
            if (entry.key === 'node') {
                const map = gml.Utility.map(entry.value);
                const id = map.get('id');
                if (id === undefined || id === null) {
                    continue;
                }
                if (map.get('isGroup')) {
                    const label = map.get('label');
                    if (typeof label === 'string' && label.length > 0) {
                        groupLabels.set(id, label);
                    }
                    continue;
                }
                if (!nodes.has(id)) {
                    nodes.set(id, { id, map, inputs: [], outputs: [] });
                }
            }
        }
        const values = new Map();
        const value = (id) => {
            if (!values.has(id)) {
                values.set(id, new gml.Value(id.toString()));
            }
            return values.get(id);
        };
        for (const entry of entries) {
            if (entry.key === 'edge') {
                const map = gml.Utility.map(entry.value);
                const source = map.get('source');
                const target = map.get('target');
                const from = nodes.get(source);
                const to = nodes.get(target);
                if (!from || !to) {
                    continue;
                }
                const argument = value(source);
                if (argument.metadata.length === 0) {
                    for (const [key, val] of map) {
                        if (key !== 'source' && key !== 'target') {
                            argument.metadata.push(new gml.Argument(key, val));
                        }
                    }
                }
                if (!from.outputs.includes(argument)) {
                    from.outputs.push(argument);
                }
                to.inputs.push(argument);
            }
        }
        for (const [id, node] of nodes) {
            this.nodes.push(new gml.Node(id, node, groupLabels));
        }
    }
};

gml.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

gml.Value = class {

    constructor(name) {
        this.name = name;
        this.type = null;
        this.initializer = null;
        this.metadata = [];
    }
};

gml.Node = class {

    constructor(id, node, groupLabels) {
        const map = node.map;
        const label = map.get('label');
        const name = map.get('name');
        const opType = map.get('op_type');
        this.name = gml.Node._title(label) || gml.Node._title(name) || `node${id}`;
        if (typeof opType === 'string' && opType.length > 0) {
            this.type = { name: opType, category: gml.Node._category(opType) };
        } else if (map.get('is_buffer')) {
            this.type = { name: 'Buffer', category: 'Constant' };
        } else {
            this.type = { name: gml.Node._title(label) || gml.Node._title(name) || 'Node' };
        }
        const gid = map.get('gid');
        if (gid !== undefined && gid !== null && groupLabels && groupLabels.has(gid)) {
            this.group = groupLabels.get(gid);
        }
        this.attributes = [];
        for (const [key, value] of map) {
            if (key === 'id' || key === 'op_type') {
                continue;
            }
            this.attributes.push(new gml.Argument(key, value, 'attribute'));
        }
        this.inputs = node.inputs.map((value, index) => new gml.Argument(index.toString(), [value]));
        this.outputs = node.outputs.map((value, index) => new gml.Argument(index.toString(), [value]));
    }

    static _title(text) {
        if (typeof text !== 'string' || text.length === 0) {
            return null;
        }
        return text.split('&#10;')[0].split('\n')[0].trim();
    }

    static _category(opType) {
        const name = opType.toLowerCase();
        for (const [regex, category] of gml.Node._categories) {
            if (regex.test(name)) {
                return category;
            }
        }
        return 'Layer';
    }
};

gml.Node._categories = [
    [/(quantize|dequantize|quant)/, 'Quantization'],
    [/attention/, 'Attention'],
    [/(batchnorm|layernorm|rmsnorm|instancenorm|groupnorm|normalization|^norm)/, 'Normalization'],
    [/(relu|sigmoid|tanh|gelu|silu|swish|softmax|prelu|leakyrelu|elu|clip|hardswish|hardsigmoid|activation)/, 'Activation'],
    [/(maxpool|avgpool|averagepool|globalpool|^pool)/, 'Pool'],
    [/dropout/, 'Dropout'],
    [/(reshape|transpose|flatten|squeeze|unsqueeze|permute|^view|split|concat|slice|^pad|gather|scatter|tile|expand|upsample|resize|interp)/, 'Shape'],
];

gml.Utility = class {

    static map(entries) {
        const map = new Map();
        for (const entry of entries) {
            const value = Array.isArray(entry.value) ? gml.Utility.map(entry.value) : entry.value;
            if (map.has(entry.key)) {
                const existing = map.get(entry.key);
                if (Array.isArray(existing)) {
                    existing.push(value);
                } else {
                    map.set(entry.key, [existing, value]);
                }
            } else {
                map.set(entry.key, value);
            }
        }
        return map;
    }
};

gml.Parser = class {

    constructor(decoder) {
        this._tokenizer = new gml.Tokenizer(decoder);
    }

    parse() {
        return this._parseList(true);
    }

    _parseList(root) {
        const list = [];
        for (;;) {
            const token = this._tokenizer.peek();
            if (token.kind === 'eof') {
                if (root) {
                    break;
                }
                throw new gml.Error(`Unexpected end of input ${this._tokenizer.location()}`);
            }
            if (token.kind === ']') {
                if (root) {
                    throw new gml.Error(`Unexpected ']' ${this._tokenizer.location()}`);
                }
                break;
            }
            if (token.kind !== 'key') {
                throw new gml.Error(`Expected key ${this._tokenizer.location()}`);
            }
            const key = this._tokenizer.read().value;
            const next = this._tokenizer.peek();
            let value = null;
            if (next.kind === '[') {
                this._tokenizer.read();
                value = this._parseList(false);
                const close = this._tokenizer.read();
                if (close.kind !== ']') {
                    throw new gml.Error(`Expected ']' ${this._tokenizer.location()}`);
                }
            } else if (next.kind === 'number' || next.kind === 'string') {
                value = this._tokenizer.read().value;
            } else {
                throw new gml.Error(`Expected value ${this._tokenizer.location()}`);
            }
            list.push({ key, value });
        }
        return list;
    }
};

gml.Tokenizer = class {

    constructor(decoder) {
        this._decoder = decoder;
        this._position = 0;
        this._char = this._decoder.decode();
        this._token = null;
    }

    peek() {
        if (!this._token) {
            this._token = this._next();
        }
        return this._token;
    }

    read() {
        const token = this.peek();
        this._token = null;
        return token;
    }

    location() {
        return `at ${this._position}.`;
    }

    _next() {
        for (;;) {
            while (this._char !== undefined && /\s/.test(this._char)) {
                this._advance();
            }
            if (this._char === '#') {
                while (this._char !== undefined && this._char !== '\n') {
                    this._advance();
                }
                continue;
            }
            break;
        }
        this._position = this._decoder.position;
        if (this._char === undefined) {
            return { kind: 'eof' };
        }
        if (this._char === '[' || this._char === ']') {
            const value = this._char;
            this._advance();
            return { kind: value, value };
        }
        if (this._char === '"') {
            this._advance();
            let value = '';
            while (this._char !== undefined && this._char !== '"') {
                if (this._char === '\\') {
                    this._advance();
                    if (this._char === undefined) {
                        break;
                    }
                    if (this._char === 'n') {
                        value += '\n';
                    } else if (this._char === 't') {
                        value += '\t';
                    } else {
                        value += this._char;
                    }
                } else {
                    value += this._char;
                }
                this._advance();
            }
            if (this._char !== '"') {
                throw new gml.Error(`Unterminated string ${this.location()}`);
            }
            this._advance();
            return { kind: 'string', value };
        }
        let value = '';
        while (this._char !== undefined && !/[\s[\]"#]/.test(this._char)) {
            value += this._char;
            this._advance();
        }
        if (value.length === 0) {
            throw new gml.Error(`Unexpected character '${this._char}' ${this.location()}`);
        }
        if (/^[+-]?\d+$/.test(value)) {
            return { kind: 'number', value: parseInt(value, 10) };
        }
        if (/^[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$/.test(value)) {
            return { kind: 'number', value: parseFloat(value) };
        }
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
            return { kind: 'key', value };
        }
        return { kind: 'string', value };
    }

    _advance() {
        this._char = this._decoder.decode();
    }
};

gml.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading GML graph';
    }
};

export const ModelFactory = gml.ModelFactory;
