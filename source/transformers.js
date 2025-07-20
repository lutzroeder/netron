
// import * as python from './python.js';
// import * as safetensors from './safetensors.js';

const transformers = {};

transformers.ModelFactory = class {

    async match(context) {
        const obj = await context.peek('json');
        if (obj) {
            if (obj.model_type && obj.architectures) {
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
            if (context.identifier === 'vocab.json' && Object.keys(obj).length > 256) {
                return context.set('transformers.vocab', obj);
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
        switch (context.type) {
            case 'transformers.config': {
                const tokenizer = await fetch('tokenizer.json');
                const tokenizer_config = await fetch('tokenizer_config.json');
                const vocab = await fetch('vocab.json');
                return new transformers.Model(context, tokenizer, tokenizer_config, vocab);
            }
            case 'transformers.tokenizer': {
                const config = await fetch('config.json');
                const tokenizer_config = await fetch('tokenizer_config.json');
                const vocab = await fetch('vocab.json');
                return new transformers.Model(config, context, tokenizer_config, vocab);
            }
            case 'transformers.tokenizer.config': {
                const config = await fetch('config.json');
                const tokenizer = await fetch('tokenizer.json');
                const vocab = await fetch('vocab.json');
                return new transformers.Model(config, tokenizer, context, vocab);
            }
            case 'transformers.vocab': {
                const config = await fetch('config.json');
                const tokenizer = await fetch('tokenizer.json');
                const tokenizer_config = await fetch('tokenizer_config.json');
                return new transformers.Model(config, tokenizer, tokenizer_config, context);
            }
            default: {
                throw new transformers.Error(`Unsupported Transformers format '${context.type}'.`);
            }
        }
    }

    filter(context, type) {
        return context.type !== 'transformers.config' || (type !== 'transformers.tokenizer' && type !== 'transformers.tokenizer.config' && type !== 'transformers.vocab' && type !== 'safetensors.json');
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
            const keys = new Set(['decoder', 'model', 'post_processor', 'pre_tokenizer']);
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

    constructor(obj) {
        this.type = { name: obj.type };
        this.attributes = [];
        for (const [key, value] of Object.entries(obj)) {
            if (key !== 'type') {
                let argument = null;
                if (Array.isArray(value) && value.every((item) => typeof item === 'object')) {
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

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

transformers.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Transformers model.';
    }
};

export const ModelFactory = transformers.ModelFactory;
