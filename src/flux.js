/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var flux = flux || {};

flux.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'bson') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return host.require('./bson').then((bson) => {
            let model = null;
            const identifier = context.identifier;
            try {
                const reader = new bson.Reader(context.buffer);
                const root = reader.read();
                const obj = flux.ModelFactory._backref(root, root);
                model = obj.model;
                if (!model) {
                    throw new flux.Error('File does not contain Flux model.');
                }
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new flux.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
            }
            return flux.Metadata.open(host).then((metadata) => {
                try {
                    return new flux.Model(metadata, model);
                }
                catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new flux.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
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

    static open(host) {
        if (flux.Metadata._metadata) {
            return Promise.resolve(flux.Metadata._metadata);
        }
        return host.request(null, 'flux-metadata.json', 'utf-8').then((data) => {
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

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
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