/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var darknet = darknet || {};
var base = base || require('./base');

darknet.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'cfg') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return darknet.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier;
            const parts = identifier.split('.');
            parts.pop();
            const basename = parts.join('.');
            return context.request(basename + '.weights', null).then((weights) => {
                return this._openModel(metadata, identifier, context.text, weights);
            }).catch(() => {
                return this._openModel(metadata, identifier, context.text, null);
            });
        });
    }
    _openModel( metadata, identifier, cfg, weights) {
        try {
            return new darknet.Model(metadata, cfg, weights);
        }
        catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            throw new darknet.Error(message + " in '" + identifier + "'.");
        }
    }
};

darknet.Model = class {

    constructor(metadata, cfg, weights) {
        this._graphs = [];
        this._graphs.push(new darknet.Graph(metadata, cfg, weights));
    }

    get format() {
        return 'Darknet';
    }

    get graphs() {
        return this._graphs;
    }
};

darknet.Graph = class {
    
    constructor(metadata, cfg, weights) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        // read_cfg
        let sections = [];
        let section = null;
        let lines = cfg.split('\n');
        let nu = 0;
        while (lines.length > 0) {
            nu++;
            let line = lines.shift();
            line = line.replace(/\s/g, '');
            if (line.length > 0) {
                switch (line[0]) {
                    case '#':
                    case ';':
                        break;
                    case '[': {
                        section = {};
                        section.type = line[line.length - 1] === ']' ? line.substring(1, line.length - 1) : line.substring(1);
                        section.options = {};
                        sections.push(section);
                        break;
                    }
                    default: {
                        if (section) {
                            let property = line.split('=');
                            if (property.length != 2) {
                                throw new darknet.Error("Invalid cfg '" + line + "' at line " + nu.toString() + ".");
                            }
                            let key = property[0].trim();
                            let value = property[1].trim();
                            section.options[key] = value;
                        }
                        break;
                    }
                }
            }
        }

        for (let section of sections) {
            section.values = {};
            const schema = metadata.getSchema(section.type);
            if (schema && schema.attributes) {
                for (let attribute of schema.attributes) {
                    if (attribute.name) {
                        if (section.options[attribute.name] !== undefined) {
                            switch (attribute.type) {
                                case 'int32':
                                    section.values[attribute.name] = parseInt(section.options[attribute.name], 10);
                                    break;
                                case 'float32':
                                    section.values[attribute.name] = parseFloat(section.options[attribute.name]);
                                    break;
                                case 'string':
                                    section.values[attribute.name] = section.options[attribute.name];
                                    break;
                            }
                        }
                        else if (attribute.default !== undefined) {
                            section.values[attribute.name] = attribute.default
                        }
                    }
                }
            }
        }

        if (sections.length === 0) {
            throw new darknet.Error('Config file has no sections.');
        }

        let net = sections.shift();
        if (net.type !== 'net' && net.type !== 'network') {
            throw new darknet.Error('First section must be [net] or [network].');
        }

        const inputType = new darknet.TensorType('float32', new darknet.TensorShape([ net.values.width, net.values.height, net.values.channels ]));

        const inputName = 'input';
        this._inputs.push(new darknet.Parameter(inputName, true, [
            new darknet.Argument(inputName, inputType, null)
        ]));

        for (let i = 0; i < sections.length; i++) {
            sections[i]._outputs = [ i.toString() ];
        }

        let inputs = [ inputName ];
        for (let i = 0; i < sections.length; i++) {
            const layer = sections[i];
            layer._inputs = inputs;
            inputs = [ i.toString() ];
            switch (layer.type) {
                case 'shortcut': {
                    let from = Number.parseInt(layer.options.from, 10);
                    from = (from >= 0) ? from : (i + from);
                    const shortcut = sections[from];
                    if (shortcut) {
                        layer._inputs.push(shortcut._outputs[0]);
                    }
                    break;
                }
                case 'route': {
                    layer._inputs = [];
                    const routes = layer.options.layers.split(',').map((route) => Number.parseInt(route.trim(), 10));
                    for (let j = 0; j < routes.length; j++) {
                        const index = (routes[j] < 0) ? i + routes[j] : routes[j];
                        const route = sections[index];
                        if (route) {
                            layer._inputs.push(route._outputs[0]);
                        }
                    }
                    break;
                }
            }
        }
        for (let i = 0; i < sections.length; i++) {
            this._nodes.push(new darknet.Node(metadata, net, sections[i], i.toString()));
        }

        if (sections.length > 0) {
            const lastLayer = sections[sections.length - 1];
            for (let i = 0; i < lastLayer._outputs.length; i++) {
                this._outputs.push(new darknet.Parameter('output' + (i > 1 ? i.toString() : ''), true, [
                    new darknet.Argument(lastLayer._outputs[i], null, null)
                ]));
            }
        }
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

darknet.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

darknet.Argument = class {

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type;
        this._initializer = initializer;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

darknet.Node = class {

    constructor(metadata, net, layer, name) {
        this._name = name;
        this._metadata = metadata;
        this._operator = layer.type;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        if (layer._inputs && layer._inputs.length > 0) {
            this._inputs.push(new darknet.Parameter(layer._inputs.length <= 1 ? 'input' : 'inputs', true, layer._inputs.map((input) => {
                return new darknet.Argument(input, null, null);
            })));
        }
        if (layer._outputs && layer._outputs.length > 0) {
            this._outputs.push(new darknet.Parameter(layer._outputs.length <= 1 ? 'output' : 'outputs', true, layer._outputs.map((output) => {
                return new darknet.Argument(output, null, null);
            })));
        }
        switch (layer.type) {
            case 'convolutional':
            case 'deconvolutional':
                this._initializer('biases', [ layer.values.filters ]);
                this._initializer('weights', [ net.values.channels, layer.values.size, layer.values.size, layer.values.filters ]);
                this._batch_normalize(metadata, net, layer, layer.values.filters);
                this._activation(metadata, net, layer, 'logistic');
                break;
            case 'connected':
                this._initializer('biases', [ layer.values.output ]);
                this._initializer('weights');
                this._batch_normalize(metadata, net, layer, layer.values.output);
                this._activation(metadata, net, layer);
                break;
            case 'crnn':
                this._batch_normalize(metadata, net, layer);
                this._activation(metadata, net, layer);
                break;
            case 'rnn':
                this._batch_normalize(metadata, net, layer, layer.values.output);
                this._activation(metadata, net, layer);
                break;
            case 'gru':
                this._batch_normalize(metadata, net, layer);
                break;
            case 'lstm':
                this._batch_normalize(metadata, net, layer);
                break;
            case 'shortcut':
                this._activation(metadata, net, layer);
                break;
            case 'batch_normalize':
                this._initializer('scale', [ layer.values.size ]);
                this._initializer('mean', [ layer.values.size ]);
                this._initializer('variance', [ layer.values.size ]);
                break;
        }

        switch (layer.type) {
            case 'shortcut':
                delete layer.options.from;
                break;
            case 'route':
                delete layer.options.layers;
                break;
        }
        for (let key of Object.keys(layer.options)) {
            this._attributes.push(new darknet.Attribute(metadata, this._operator, key, layer.options[key]));
        }
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get documentation() {
        return '';
    }

    get category() {
        const schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }

    _initializer(name, shape) {
        const id = this._name.toString() + '_' + name;
        this._inputs.push(new darknet.Parameter(name, true, [
            new darknet.Argument(id, null, new darknet.Tensor(id, shape))
        ]));
    }

    _batch_normalize(metadata, net, layer, size) {
        if (layer.values.batch_normalize === 1) {
            const batch_normalize_layer = { type: 'batch_normalize', options: {}, values: { size: size || 0 }, _inputs: [], _outputs: [] };
            this._chain.push(new darknet.Node(metadata, net, batch_normalize_layer, ''));
        }
        delete layer.options.batch_normalize;
    }

    _activation(metadata, net, layer) {
        const attributeSchema = metadata.getAttributeSchema(layer.type, 'activation');
        if (attributeSchema) {
            if (layer.options.activation !== attributeSchema.default) {
                this._chain.push(new darknet.Node(metadata, net, { type: layer.options.activation, options: {}, values: {}, _inputs: [], _outputs: [] }, ''));
            }
            delete layer.options.activation;
        }
    }
};

darknet.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;
        const schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            this._type = schema.type || '';
            switch (this._type) {
                case 'int32': {
                    this._value = parseInt(this._value, 10);
                    break;
                }
                case 'float32': {
                    this._value = parseFloat(this._value);
                    break;
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

darknet.Tensor = class {

    constructor(id, shape) {
        shape = shape || null;
        this._id = id;
        this._type = new darknet.TensorType('?', new darknet.TensorShape(shape));
    }

    get name() {
        return this._id;
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Tensor data not implemented.';
    }

    get value() {
        return null;
    }

    toString() {
        return '';
    }
};

darknet.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this._dataType || '?') + this._shape.toString();
    }
};

darknet.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length == 0) {
                return '';
            }
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

darknet.Metadata = class {

    static open(host) {
        if (darknet.Metadata._metadata) {
            return Promise.resolve(darknet.Metadata._metadata);
        }
        return host.request(null, 'darknet-metadata.json', 'utf-8').then((data) => {
            darknet.Metadata._metadata = new darknet.Metadata(data);
            return darknet.Metadata._metadata;
        }).catch(() => {
            darknet.Metadata._metadata = new darknet.Metadata(null);
            return darknet.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map.get(operator) || null;
    }

    getAttributeSchema(operator, name) {
        let map = this._attributeCache.get(operator);
        if (!map) {
            map = new Map();
            let schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (let attribute of schema.attributes) {
                    map.set(attribute.name, attribute);
                }
            }
            this._attributeCache.set(operator, map);
        }
        return map.get(name) || null;
    }
};

darknet.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Darknet model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = darknet.ModelFactory;
}
