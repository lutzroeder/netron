
// import * as python from './python.js';
// import * as safetensors from './safetensors.js';

const transformers = {};

transformers.ModelFactory = class {

    async match(context) {
        const obj = await context.peek('json');
        if (obj) {
            if (obj.architectures && (obj.model_type || obj.transformers_version)) {
                return context.set('transformers.config', obj);
            }
            if (obj.version && obj.added_tokens && obj.model) {
                return context.set('transformers.tokenizer', obj);
            }
            if (obj.tokenizer_class ||
                (obj.bos_token && obj.eos_token && obj.unk_token) ||
                (obj.pad_token && obj.additional_special_tokens) ||
                obj.special_tokens_map_file || obj.full_tokenizer_file) {
                return context.set('transformers.tokenizer.config', obj);
            }
            if (obj.transformers_version && obj.do_sample !== undefined && obj.temperature !== undefined) {
                return context.set('transformers.generation_config', obj);
            }
            if (obj.transformers_version && obj._from_model_config !== undefined) {
                return context.set('transformers.generation_config', obj);
            }
            if (obj.crop_size !== undefined && obj.do_center_crop !== undefined && obj.image_mean !== undefined && obj.image_std !== undefined && obj.do_resize !== undefined) {
                return context.set('transformers.preprocessor_config.json', obj);
            }
            if (!Array.isArray(obj) && typeof obj === 'object') {
                const entries = Object.entries(obj);
                if (entries.every(([key, value]) => typeof key === 'string' && key.length < 256 && Number.isInteger(value) && value < 0x80000)) {
                    if (obj["<|im_start|>"] || obj["<|endoftext|>"]) {
                        return context.set('transformers.vocab', obj);
                    }
                }
                const dtypes = new Set(['BF16', 'FP4', 'UE8']);
                if (entries.every(([key, value]) => typeof key === 'string' && dtypes.has(value))) {
                    return context.set('transformers.dtypes', obj);
                }
            }
        }
        return null;
    }

    async open(context) {
        const fetch = async (name) => {
            try {
                const content = await context.fetch(name);
                await this.match(content);
                if (content.value) {
                    return content;
                }
            } catch {
                // continue regardless of error
            }
            return null;
        };
        const type = context.type;
        const config = type === 'transformers.config' ? context : await fetch('config.json');
        const tokenizer = type === 'transformers.tokenizer' ? context : await fetch('tokenizer.json');
        const tokenizer_config = type === 'transformers.tokenizer.config' ? context : await fetch('tokenizer_config.json');
        const vocab = type === 'transformers.vocab' ? context : await fetch('vocab.json');
        const generation_config = type === 'transformers.generation_config' ? context : await fetch('generation_config.json');
        const preprocessor_config = type === 'transformers.preprocessor_config.json' ? context : await fetch('preprocessor_config.json');
        return new transformers.Model(config, tokenizer, tokenizer_config, vocab, generation_config, preprocessor_config);
    }

    filter(context, match) {
        const priority = new Map([
            ['transformers.config', 7],
            ['transformers.tokenizer', 6],
            ['transformers.tokenizer.config', 5],
            ['transformers.vocab', 4],
            ['transformers.generation_config', 3],
            ['transformers.preprocessor_config.json', 2],
            ['transformers.dtypes', 1],
            ['safetensors.json', 0],
            ['safetensors', 0]
        ]);
        const a = priority.has(context.type) ? priority.get(context.type) : -1; // current
        const b = priority.has(match.type) ? priority.get(match.type) : -1;
        if (a !== -1 && b !== -1) {
            return a < b;
        }
        return true;
    }
};

transformers.Model = class {

    constructor(config, tokenizer, tokenizer_config, vocab) {
        this.format = 'Transformers';
        this.metadata = [];
        this.modules = [new transformers.Graph(config, tokenizer, tokenizer_config, vocab)];
    }
};

transformers.Graph = class {

    constructor(config, tokenizer, tokenizer_config, vocab) {
        this.type = 'graph';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        this.metadata = [];
        if (config) {
            for (const [key, value] of Object.entries(config.value)) {
                const argument = new transformers.Argument(key, value);
                this.metadata.push(argument);
            }
        }
        if (tokenizer || tokenizer_config) {
            const node = new transformers.Tokenizer(tokenizer, tokenizer_config, vocab);
            this.nodes.push(node);
        }
    }
};

transformers.Tokenizer = class {

    constructor(tokenizer, tokenizer_config) {
        this.type = { name: 'Tokenizer' };
        this.name = (tokenizer || tokenizer_config).identifier;
        this.attributes = [];
        if (tokenizer) {
            const obj = tokenizer.value;
            const keys = new Set(['decoder', 'post_processor', 'pre_tokenizer']);
            for (const [key, value] of Object.entries(tokenizer.value)) {
                if (!keys.has(key)) {
                    const argument = new transformers.Argument(key, value);
                    this.attributes.push(argument);
                }
            }
            for (const key of keys) {
                const value = obj[key];
                if (value) {
                    const module = new transformers.Object(value);
                    const argument = new transformers.Argument(key, module, 'object');
                    this.attributes.push(argument);
                }
            }
        }
    }
};

transformers.Object = class {

    constructor(obj, type) {
        this.type = { name: type || obj.type };
        this.attributes = [];
        for (const [key, value] of Object.entries(obj)) {
            if (key !== 'type') {
                let argument = null;
                if (Array.isArray(value) && value.every((item) => typeof item === 'object' && Object.keys(item).length === 1 && typeof Object.entries(item)[0][1] === 'object')) {
                    const values = value.map((item) => new transformers.Object(Object.entries(item)[0][1], Object.entries(item)[0][0]));
                    argument = new transformers.Argument(key, values, 'object[]');
                } else if (Array.isArray(value) && value.every((item) => typeof item === 'object')) {
                    const values = value.map((item) => new transformers.Object(item));
                    argument = new transformers.Argument(key, values, 'object[]');
                } else {
                    argument = new transformers.Argument(key, value);
                }
                this.attributes.push(argument);
            }
        }
    }
};

transformers.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

transformers.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Transformers model.';
    }
};

export const ModelFactory = transformers.ModelFactory;
