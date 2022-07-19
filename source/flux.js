
// Experimental

var flux = flux || {};
var json = json || require('./json');

flux.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        if (stream && extension === 'bson') {
            return 'flux.bson';
        }
        return undefined;
    }

    open(context) {
        return Promise.resolve().then(() => {
            let root = null;
            try {
                const stream = context.stream;
                const reader = json.BinaryReader.open(stream);
                root = reader.read();
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new flux.Error('File format is not Flux BSON (' + message.replace(/\.$/, '') + ').');
            }
            return context.metadata('flux-metadata.json').then((metadata) => {
                const obj = flux.ModelFactory._backref(root, root);
                const model = obj.model;
                if (!model) {
                    throw new flux.Error('File does not contain Flux model.');
                }
                return new flux.Model(metadata, model);
            });
        });
    }

    static _backref(obj, root) {
        if (Array.isArray(obj)) {
            for (let i = 0; i < obj.length; i++) {
                obj[i] = flux.ModelFactory._backref(obj[i], root);
            }
        }
        else if (obj === Object(obj)) {
            if (obj.tag == 'backref' && obj.ref) {
                if (!root._backrefs[obj.ref - 1]) {
                    throw new flux.Error("Invalid backref '" + obj.ref + "'.");
                }
                obj = root._backrefs[obj.ref - 1];
            }
            for (const key of Object.keys(obj)) {
                if (obj !== root || key !== '_backrefs') {
                    obj[key] = flux.ModelFactory._backref(obj[key], root);
                }
            }
        }
        return obj;
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

if (module && module.exports) {
    module.exports.ModelFactory = flux.ModelFactory;
}