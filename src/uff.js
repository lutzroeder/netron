/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var uff = uff || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');

uff.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'uff' || extension === 'pb') {
            const tags = context.tags('pb');
            if (tags.size > 0 &&
                tags.has(1) && tags.get(1) === 0 &&
                tags.has(2) && tags.get(2) === 0 &&
                tags.has(3) && tags.get(3) === 2 &&
                tags.has(4) && tags.get(4) === 2 &&
                tags.has(5) && tags.get(5) === 2) {
                return true;
            }
        }
        if (extension === 'pbtxt' || identifier.toLowerCase().endsWith('.uff.txt')) {
            const tags = context.tags('pbtxt');
            if (tags.has('version') && tags.has('descriptors') && tags.has('graphs')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./uff-proto').then(() => {
            let meta_graph = null;
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            if (extension === 'pbtxt' || identifier.toLowerCase().endsWith('.uff.txt')) {
                try {
                    uff.proto = protobuf.roots.uff.uff;
                    const reader = prototxt.TextReader.create(context.text);
                    const field = reader.field;
                    reader.field = (token, module) => {
                        if (token === 'descriptors' || token === 'graphs' || token == 'referenced_data') {
                            return reader.skip();
                        }
                        return field(token, module);
                    };
                    meta_graph = uff.proto.MetaGraph.decodeText(reader);
                }
                catch (error) {
                    throw new uff.Error("File text format is not uff.MetaGraph (" + error.message + ") in '" + identifier + "'.");
                }
            }
            else {
                try {
                    uff.proto = protobuf.roots.uff.uff;
                    meta_graph = uff.proto.MetaGraph.decode(context.buffer);
                }
                catch (error) {
                    throw  new uff.Error("File format is not uff.MetaGraph (" + error.message + ") in '" + identifier + "'.");
                }
            }
            return uff.Metadata.open(host).then((metadata) => {
                try {
                    return new uff.Model(metadata, meta_graph);
                }
                catch (error) {
                    host.exception(error, false);
                    const message = error && error.message ? error.message : error.toString();
                    throw new uff.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
            });
        });
    }
};

uff.Model = class {

    constructor(metadata, meta_graph) {
        meta_graph.graphs = meta_graph.graphs || [];
        this._version = meta_graph.version;
        this._graphs = meta_graph.graphs.map((graph) => new uff.Graph(metadata, graph));
    }

    get format() {
        return 'UFF' + (this._version ? ' v' + this._version.toString() : '');
    }

    get imports() {
        return this._imports;
    }

    get graphs() {
        return this._graphs;
    }
};

uff.Graph = class {

    constructor(metadata, graph) {
        this._name = graph.id;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

uff.Metadata = class {

    static open(host) {
        if (uff.Metadata._metadata) {
            return Promise.resolve(uff.Metadata._metadata);
        }
        return host.request(null, 'uff-metadata.json', 'utf-8').then((data) => {
            uff.Metadata._metadata = new uff.Metadata(data);
            return uff.Metadata._metadata;
        }).catch(() => {
            uff.Metadata._metadata = new uff.Metadata(null);
            return uff.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

uff.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading UFF model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = uff.ModelFactory;
}
