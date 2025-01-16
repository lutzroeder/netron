
const sentencepiece = {};

sentencepiece.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if ((tags.size >= 3 && tags.size <= 5 &&
            tags.get(1) === 2 && tags.get(2) === 2 && tags.get(3) === 2) &&
            Array.from(tags).every(([key, value]) => (key <= 5 && value === 2))) {
            const model = context.tags('pb+');
            if (model &&
                model['1'] && model['1']['1'] === 2 && model['1']['2'] === 5 && model['1']['3'] === 0 &&
                model['2'] && model['2']['3'] === 0 && model['2']['4'] === 0 &&
                model['3'] && model['3']['1'] === 2) {
                context.type = 'sentencepiece';
            }
        }
    }

    async open(context) {
        sentencepiece.proto = await context.require('./sentencepiece-proto');
        sentencepiece.proto = sentencepiece.proto.sentencepiece;
        let model = null;
        try {
            const reader = context.read('protobuf.binary');
            model = sentencepiece.proto.ModelProto.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new sentencepiece.Error(`File format is not sentencepiece.ModelProto (${message.replace(/\.$/, '')}).`);
        }
        return new sentencepiece.Model(model);
    }
};

sentencepiece.Model = class {

    constructor(model) {
        this.format = 'SentencePiece';
        this.graphs = [new sentencepiece.Graph(model)];
    }
};

sentencepiece.Graph = class {

    constructor(model) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        for (const [name, value] of Object.entries(model)) {
            const node = new sentencepiece.Node(name, value);
            this.nodes.push(node);
        }
    }
};

sentencepiece.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

sentencepiece.Node = class {

    constructor(name, obj) {
        this.name = name;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (Array.isArray(obj)) {
            const type = new Set(obj.map((value) => value.constructor.name));
            this.type = { name: `${Array.from(type)[0]}[]` };
            const attribute = new sentencepiece.Argument(name, obj);
            this.attributes.push(attribute);
        } else {
            this.type = { name: obj.constructor.name };
            for (const [name, value] of Object.entries(obj)) {
                const data = ArrayBuffer.isView(value) ? Array.from(value) : value;
                const attribute = new sentencepiece.Argument(name, data);
                this.attributes.push(attribute);
            }
        }
    }
};

sentencepiece.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading SentencePiece model.';
    }
};

export const ModelFactory = sentencepiece.ModelFactory;
