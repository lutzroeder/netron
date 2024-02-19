
// Experimental

const flux = {};

flux.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        if (stream && extension === 'bson') {
            context.type = 'flux.bson';
        }
    }

    async open(context) {
        let root = null;
        try {
            root = context.read('bson');
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new flux.Error(`File format is not Flux BSON (${message.replace(/\.$/, '')}).`);
        }
        /* const metadata = */ context.metadata('flux-metadata.json');
        const backref = (obj, root) => {
            if (Array.isArray(obj)) {
                for (let i = 0; i < obj.length; i++) {
                    obj[i] = backref(obj[i], root);
                }
            } else if (obj === Object(obj)) {
                if (obj.tag === 'backref' && obj.ref) {
                    if (!root._backrefs[obj.ref - 1]) {
                        throw new flux.Error(`Invalid backref '${obj.ref}'.`);
                    }
                    obj = root._backrefs[obj.ref - 1];
                }
                for (const key of Object.keys(obj)) {
                    if (obj !== root || key !== '_backrefs') {
                        obj[key] = backref(obj[key], root);
                    }
                }
            }
            return obj;
        };
        const obj = backref(root, root);
        const model = obj.model;
        if (!model) {
            throw new flux.Error('File does not contain Flux model.');
        }
        throw new flux.Error("File contains unsupported Flux data.");
    }
};

flux.Model = class {

    constructor(/* root */) {
        this._format = 'Flux';
        this._graphs = [];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

flux.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Flux Error';
    }
};

export const ModelFactory = flux.ModelFactory;
