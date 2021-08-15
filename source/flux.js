/* jshint esversion: 6 */

// Experimental

var flux = flux || {};
var json = json || require('./json');

flux.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'bson') {
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
            return flux.Metadata.open(context).then((metadata) => {
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

flux.Metadata = class {

    static open(context) {
        if (flux.Metadata._metadata) {
            return Promise.resolve(flux.Metadata._metadata);
        }
        return context.request('flux-metadata.json', 'utf-8', null).then((data) => {
            flux.Metadata._metadata = new flux.Metadata(data);
            return flux.Metadata._metadata;
        }).catch(() => {
            flux.Metadata._metadata = new flux.Metadata(null);
            return flux.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    type(name) {
        return this._map[name] || null;
    }

    attribute(type, name) {
        let map = this._attributeCache[type];
        if (!map) {
            map = {};
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[type] = map;
        }
        return map[name] || null;
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