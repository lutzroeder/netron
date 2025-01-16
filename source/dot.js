
const dot = {};

dot.ModelFactory = class {

    match(context) {
        const reader = context.read('text', 0x10000);
        if (reader) {
            try {
                for (let i = 0; i < 64; i++) {
                    const line = reader.read('\n');
                    if (line === undefined) {
                        break;
                    }
                    if (line.trim().startsWith('//') || line.trim().startsWith('#')) {
                        continue;
                    }
                    if (line.trim().match(/^(strict)?\s*digraph/)) {
                        context.type = 'dot';
                        break;
                    }
                }
            } catch {
                // continue regardless of error
            }
        }
    }

    async open(context) {
        const decoder = context.read('text.decoder');
        const parser = new dot.Parser(decoder);
        const graph = parser.parse();
        if (graph.kind !== 'digraph') {
            throw new dot.Error(`Graph type '${graph.type}' is not supported.`);
        }
        return new dot.Model(graph);
    }
};

dot.Model = class {

    constructor(graph) {
        this.format = 'DOT';
        this.graphs = [new dot.Graph(graph)];
    }
};

dot.Graph = class {

    constructor(graph) {
        this.name = graph.name || '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, tensor, metadata) => {
            if (typeof name !== 'string') {
                throw new dot.Error('Invalid value name.');
            }
            if (!values.has(name) || tensor) {
                values.set(name, new dot.Value(name, type, tensor, metadata));
            }
            return values.get(name);
        };
        const nodes = new Map();
        nodes.map = (name) => {
            if (typeof name !== 'string') {
                throw new dot.Error('Invalid node name.');
            }
            if (!nodes.has(name)) {
                const node = {
                    kind: 'node',
                    name: { id: name, key: name },
                    type: { name },
                    inputs: [],
                    outputs: [],
                    attributes: new Map(),
                    metadata: new Map()
                };
                nodes.set(name, node);
            }
            return nodes.get(name);
        };
        for (const node of graph.statements) {
            if (node.kind === 'node') {
                node.inputs = [];
                node.outputs = [];
                node.metadata = new Map([...node.defaults, ...node.attributes]);
                node.attributes = new Map();
                delete node.defaults;
                const metadata = node.metadata;
                if (metadata.has('label')) {
                    const label = metadata.get('label');
                    if (label.startsWith('{') && label.endsWith('}')) {
                        const lines = label.substring(1, label.length - 1).split('|');
                        if (lines.length > 1 && node.name.id === lines[0] && lines[1].startsWith('op_code=')) {
                            const def = lines[1].split('\\l');
                            const op_code = def[0].split('=').pop();
                            node.type = { name: op_code };
                            if (op_code === 'call_module') {
                                node.type = { name: def[1], type: 'function' };
                            } else if (op_code === 'call_function') {
                                const vals = lines[2].split('\\l');
                                node.type = { name: vals[0] };
                            } else if (op_code.startsWith('get_parameter')) {
                                node.attributes.set('type', op_code.substring(13, op_code.length).trim());
                                node.type = { name: 'get_parameter' };
                            }
                            if (lines.length > 2) {
                                const attributes = lines[2].split('\\l');
                                for (const attribute of attributes) {
                                    const parts = attribute.split(':');
                                    if (parts.length === 2) {
                                        const key = parts[0].trim();
                                        let value = parts[1].trim();
                                        if (value.startsWith('(') && value.endsWith(')')) {
                                            value = JSON.parse(`[${value.substring(1, value.length - 1)}]`);
                                        }
                                        node.attributes.set(key, value);
                                    }
                                }
                            }
                            metadata.delete('label');
                        } else if (lines.length === 1 && lines[0].startsWith('buffer\\l')) {
                            const def = lines[0].split('\\l');
                            node.type = { name: def[0] };
                            if (def.length > 1) {
                                node.attributes.set('type', def[1]);
                            }
                            metadata.delete('label');
                        }
                    } else {
                        const match = label.match(/^name:\s*([A-Za-z][A-Za-z0-9_]*)\stype:\s*([A-Za-z][A-Za-z0-9_]*)$/);
                        if (match && node.name.id === match[1]) {
                            node.type = { name: match[2] };
                            metadata.delete('label');
                        }
                    }
                }
                if (!node.type) {
                    const lines = node.name.id.split('\\n');
                    const match = lines[0].match(/^([A-Z][A-Za-z0-9_]*)\/([A-Z][A-Za-z0-9_]*)\s\(op#(\d+)\)$/);
                    if (match) {
                        node.type = { name: match[2] };
                    } else {
                        const match = lines[0].match(/^([A-Z][A-Za-z0-9_]*)\s\(op#(\d+)\)$/);
                        if (match) {
                            node.type = { name: match[1] };
                        } else {
                            // debugger;
                        }
                    }

                }
                if (!node.type) {
                    node.type = { name: node.name.id };
                }
                nodes.set(node.name.id, node);
            }
        }
        for (const edge of graph.statements) {
            if (edge.kind === 'edge') {
                edge.uses = edge.uses || [];
                const to = nodes.map(edge.to.id);
                to.inputs.push(edge);
                edge.uses.push(to);
                edge.from = nodes.map(edge.name.id);
                edge.from.outputs.push(edge);
            }
        }
        for (const [key, node] of nodes) {
            const keys = new Set(['pos', 'height', 'width', 'shape', 'label']);
            if (node.metadata.get('shape') === 'octagon' && node.metadata.keys().every((key) => keys.has(key)) &&
                node.inputs.length === 1 && node.inputs[0].uses.length === 1 && node.inputs[0].from.outputs.length === 1 && node.inputs[0].from.outputs[0].uses.length === 1 &&
                new Set(node.outputs.map((output) => output.name.id)).size === 1 && node.outputs.every((output) => output.uses.length === 1)) {
                const [from] = node.inputs[0].from.outputs;
                for (const e of node.outputs) {
                    const [n] = e.uses;
                    n.inputs = n.inputs.map((edge) => edge === e ? from : edge);
                }
                nodes.delete(key);
            }
        }
        for (const [key, node] of nodes) {
            if ((node.type.name === 'get_parameter' || node.type.name === 'buffer' || node.type.name === 'Constant') &&
                node.inputs.length === 0 &&
                node.outputs.length === 1 && node.outputs[0].uses.length === 1) {
                node.outputs[0].initializer = node;
                nodes.delete(key);
            }
        }
        for (const [, obj] of nodes) {
            const node = new dot.Node(obj, values);
            this.nodes.push(node);
        }
        for (const edge of graph.statements) {
            if (edge.kind === 'edge') {
                const value = values.map(edge.name.id);
                const metadata = new Map([...edge.defaults, ...edge.attributes]);
                value.metadata = Array.from(metadata).map(([key, value]) => new dot.Argument(key, value));
            }
        }
    }
};

dot.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

dot.Value = class {

    constructor(name, type, initializer, metadata) {
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer || null;
        this.metadata = metadata;
    }
};

dot.Node = class {

    constructor(node, values) {
        this.name = node.name.key;
        this.type = node.type;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.metadata = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const edge = node.inputs[i];
            const initializer = edge.initializer ? new dot.Tensor(edge.initializer) : null;
            const value = values.map(edge.name.key, null, initializer);
            const argument = new dot.Argument(i.toString(), [value]);
            this.inputs.push(argument);
        }
        for (let i = 0; i < node.outputs.length; i++) {
            const edge = node.outputs[i];
            const value = values.map(edge.name.key);
            const argument = new dot.Argument(i.toString(), [value]);
            this.outputs.push(argument);
        }
        for (const [name, value] of node.attributes) {
            const argument = new dot.Argument(name, value, 'attribute');
            this.attributes.push(argument);
        }
        for (const [name, value] of node.metadata) {
            const argument = new dot.Argument(name, value);
            this.metadata.push(argument);
        }
    }
};

dot.TensorType = class {

    constructor(type) {
        const index = type.indexOf('[');
        const dtype = type.substring(0, index);
        this.dataType = dtype.split('.').pop();
        if (index > 0) {
            const dimensions = JSON.parse(type.substring(index, type.length));
            this.shape = new dot.TensorShape(dimensions);
        } else {
            this.shape = new dot.TensorShape([]);
        }
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

dot.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

dot.Tensor = class {

    constructor(stmt) {
        if (stmt.attributes.has('type')) {
            const type = stmt.attributes.get('type');
            this.type = new dot.TensorType(type);
        } else {
            this.type = new dot.TensorType('?');
        }
    }
};

dot.Parser = class {

    constructor(decoder) {
        // https://graphviz.org/doc/info/lang.html
        this._tokenizer = new dot.Tokenizer(decoder);
        this._token = this._tokenizer.read();
    }

    parse() {
        const graph = {};
        if (this._eat('id', 'strict')) {
            graph.strict = true;
        }
        let edgeop = '';
        if (this._match('id', 'graph')) {
            graph.kind = this._read();
            edgeop = '--';
        } else if (this._match('id', 'digraph')) {
            graph.kind = this._read();
            edgeop = '->';
        } else {
            throw new dot.Error('Invalid graph type.');
        }
        if (this._match('id')) {
            graph.name = this._read();
        }
        const defaults = {};
        defaults.graph = new Map();
        defaults.node = new Map();
        defaults.edge = new Map();
        graph.statements = this._parseBlock(defaults, edgeop, 0);
        graph.defaults = new Map(defaults.graph);
        return graph;
    }

    _parseBlock(defaults, edgeop) {
        defaults = {
            graph: new Map(defaults.graph),
            node: new Map(defaults.node),
            edge: new Map(defaults.edge)
        };
        const list = [];
        this._read('{');
        while (!this._match('}')) {
            if (this._eat('id', 'subgraph')) {
                const stmt = {};
                stmt.kind = 'subgraph';
                if (this._match('id')) {
                    stmt.name = this._read();
                }
                stmt.statements = this._parseBlock(defaults, edgeop);
            } else if (this._match('{')) {
                const stmt = {};
                const statements = this._parseBlock(defaults, edgeop);
                if (this._eat(edgeop)) {
                    if (!statements.every((stmt) => stmt.kind === 'node' && stmt.attributes.size === 0)) {
                        throw new dot.Error('Invalid edge group statement.');
                    }
                    const sources = statements.map((stmt) => stmt.name);
                    list.push(...this._parseEdges(sources, edgeop, defaults.edge));
                } else {
                    stmt.kind = 'subgraph';
                    stmt.statements = statements;
                }
            } else if (this._match('id')) {
                const name = this._parseNodeId();
                if (this._eat('=')) { // attr
                    if (this._match('id')) {
                        const value = this._read();
                        defaults.graph.set(name, value);
                    } else {
                        throw new dot.Error('Invalid attribute value.');
                    }
                } else if (this._eat(edgeop)) {
                    list.push(...this._parseEdges([name], edgeop, defaults.edge));
                } else {
                    const attributes = this._parseAttributes();
                    if (name.key === 'node' || name.key === 'edge' || name.key === 'graph') {
                        for (const [key, value] of attributes) {
                            defaults[name.key].set(key, value);
                        }
                    } else {
                        list.push({ kind: 'node', name, attributes, defaults: new Map(defaults.node) });
                    }
                }
            }
            if (this._match(';') || this._match(',')) {
                this._read();
            }
        }
        this._read('}');
        return list;
    }

    _parseNodeIds() {
        const list = [];
        const open = this._eat('{');
        while (!this._match('}')) {
            const value = this._parseNodeId();
            list.push(value);
            if (this._match(',')) {
                this._read();
                continue;
            } else if (this._match(';')) {
                this._read();
                if (!open) {
                    break;
                }
            } else if (!open) {
                break;
            }
        }
        if (open) {
            this._read('}');
        }
        return list;
    }

    _parseNodeId() {
        const name = {};
        const list = [];
        name.id = this._read('id');
        list.push(name.id);
        if (this._eat(':')) {
            name.port = this._read('id');
            list.push(name.port);
            if (this._eat(':')) {
                name.compass_pt = this._read('id');
                list.push(name.compass_pt);
            }
        }
        name.key = list.join(':');
        return name;
    }

    _parseAttributes() {
        const table = new Map();
        if (this._eat('[')) {
            while (this._match('id')) {
                const name = this._read('id');
                this._read('=');
                const value = this._read('id');
                table.set(name, value);
                if (this._match(';') || this._match(',')) {
                    this._read();
                }
            }
            this._read(']');
        }
        return table;
    }

    _parseEdges(sources, edgeop, defaults) {
        const list = [];
        do {
            const targets = this._parseNodeIds();
            for (const name of sources) {
                for (const to of targets) {
                    list.push({ kind: 'edge', name, to });
                }
            }
            sources = targets;
        } while (this._eat(edgeop));
        const attributes = this._parseAttributes();
        for (const edge of list) {
            edge.attributes = attributes;
            edge.defaults = new Map(defaults.edge);
        }
        return list;
    }

    _match(kind, value) {
        return (this._token.kind === kind && (!value || this._token.value === value));
    }

    _read(kind, value) {
        if (kind && this._token.kind !== kind) {
            throw new dot.Error(`Expected token of type '${kind}', but got '${this._token.kind}' ${this._tokenizer.location()}`);
        }
        if (value && this._token.value !== value) {
            throw new dot.Error(`Expected token with value '${value}', but got '${this._token.value}' ${this._tokenizer.location()}`);
        }
        const token = this._token;
        this._token = this._tokenizer.read();
        return token.value;
    }

    _eat(kind, value) {
        if (this._match(kind, value)) {
            return this._read();
        }
        return null;
    }
};

dot.Tokenizer = class {

    constructor(decoder) {
        this._decoder = decoder;
        this._position = 0;
        this._char = this._decoder.decode();
    }

    _read() {
        if (this._char === undefined) {
            this._unexpected();
        }
        const char = this._char;
        this._position = this._decoder.position;
        this._char = this._decoder.decode();
        return char;
    }

    _peek() {
        const position = this._decoder.position;
        const char = this._decoder.decode();
        this._decoder.position = position;
        return char;
    }

    read() {
        while (this._char) {
            if (/\s/.test(this._char)) {
                this._skipWhitespace();
                continue;
            }
            if (this._char === '/' || this._char === '#') {
                this._skipComment();
                continue;
            }
            if (/[{}[\]=:;,]/.test(this._char)) {
                const value = this._read();
                return { kind: value, value };
            } else if (this._char === '-') {
                let value = this._read();
                if (this._char === '>' || this._char === '-') {
                    value += this._read();
                    return { kind: value, value };
                }
                throw new dot.Error(`Unexpected character '${value}' ${this.location()}`);
            } else if (/[a-zA-Z0-9_$"<]/.test(this._char)) {
                const value = this._identifier();
                return { kind: 'id', value };
            } else {
                throw new dot.Error(`Unexpected character '${this._char}' ${this.location()}`);
            }
        }
        return { type: 'eof' };
    }

    _skipWhitespace() {
        while (this._char !== undefined && /\s/.test(this._char)) {
            this._read();
        }
    }

    _skipComment() {
        if (this._char === '#' || (this._char === '/' && this._peek() === '/')) {
            while (this._char && this._char !== '\n') {
                this._read();
            }
            return;
        }
        if (this._char === '/' && this._peek() === '*') {
            while (this._char && (this._char !== '*' || this._peek() !== '/')) {
                this._read();
            }
            this._read();
            this._read();
            return;
        }
        throw new dot.Error('Invalid comment.');
    }

    _identifier() {
        let value = '';
        if (this._char === '"') { // double quoted string
            this._read();
            while (this._char && this._char !== '"') {
                value += this._read();
            }
            this._read('"');
        } if (this._char === '<') { // HTML String
            value += this._read();
            let depth = 0;
            while (depth > 0 || this._char !== '>') {
                const c = this._read();
                value += c;
                if (c === '<') {
                    depth += 1;
                } else if (c === '>') {
                    depth -= 1;
                }
            }
            value += this._read();
        } else {
            while (/[a-zA-Z0-9_$.*]/.test(this._char)) {
                value += this._read();
            }
        }
        return value;
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new dot.Error('Unexpected end of input.');
        } else if (c === '"') {
            c = 'string';
        } else if ((c >= '0' && c <= '9') || c === '-') {
            c = 'number';
        } else {
            if (c < ' ' || c > '\x7F') {
                const name = Object.keys(this._escape).filter((key) => this._escape[key] === c);
                c = (name.length === 1) ? `\\${name}` : `\\u${(`000${c.charCodeAt(0).toString(16)}`).slice(-4)}`;
            }
            c = `token '${c}'`;
        }
        this._throw(`Unexpected ${c}`);
    }

    _throw(message) {
        message = message.replace(/\.$/, '');
        throw new dot.Error(`${message} ${this._location()}`);
    }

    location() {
        let line = 1;
        let column = 1;
        const position = this._decoder.position;
        this._decoder.position = 0;
        let c = '';
        do {
            if (this._decoder.position === this._position) {
                this._decoder.position = position;
                return `at ${line}:${column}.`;
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        while (c !== undefined);
        this._decoder.position = position;
        return `at ${line}:${column}.`;
    }
};

dot.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loadig DOT graph';
    }
};

export const ModelFactory = dot.ModelFactory;
