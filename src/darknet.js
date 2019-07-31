/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var darknet = darknet || {};
var base = base || require('./base');

darknet.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'cfg') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return darknet.Metadata.open(host).then((metadata) => {
            var identifier = context.identifier;
            try {
                var reader = new darknet.CfgReader(context.text);
                var cfg = reader.read();
                return new darknet.Model(metadata, cfg);
            }
            catch (error) {
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new darknet.Error(message + " in '" + identifier + "'.");
            }
        });
    }
};

darknet.Model = class {

    constructor(metadata, cfg) {
        this._graphs = [];
        this._graphs.push(new darknet.Graph(metadata, cfg));
    }

    get format() {
        return 'Darknet';
    }

    get graphs() {
        return this._graphs;
    }
};

darknet.Graph = class {
    
    constructor(metadata, cfg) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        if (cfg.length === 0) {
            throw new darknet.Error('Config file has no sections.');
        }

        var net = cfg.shift();
        if (net.__type__ !== 'net' && net.__type__ !== 'network') {
            throw new darknet.Error('First section must be [net] or [network].');
        }

        net.width = net.width ? Number.parseInt(net.width, 10) : 0;
        net.height = net.height ? Number.parseInt(net.height, 10) : 0;
        net.channels = net.channels ? Number.parseInt(net.channels, 10) : 0;

        var inputType = new darknet.TensorType('float32', new darknet.TensorShape([ net.width, net.height, net.channels ]));

        var input = 'input';
        this._inputs.push(new darknet.Parameter(input, true, [
            new darknet.Argument(input, inputType, null)
        ]));

        var i;
        for (i = 0; i < cfg.length; i++) {
            cfg[i]._outputs = [ i.toString() ];
        }

        var inputs = [ 'input' ];
        for (i = 0; i < cfg.length; i++) {
            var layer = cfg[i];
            layer._inputs = inputs;
            inputs = [ i.toString() ];
            switch (layer.__type__) {
                case 'shortcut':
                    var shortcut = cfg[i + Number.parseInt(layer.from, 10)];
                    if (shortcut) {
                        layer._inputs.push(shortcut._outputs[0]);
                    }
                    break;
                case 'route':
                    layer._inputs = [];
                    var routes = layer.layers.split(',').map((route) => Number.parseInt(route.trim(), 10));
                    for (var j = 0; j < routes.length; j++) {
                        var index = (routes[j] < 0) ? i + routes[j] : routes[j];
                        var route = cfg[index];
                        if (route) {
                            layer._inputs.push(route._outputs[0]);
                        }
                    }
                    break;
            }
        }
        for (i = 0; i < cfg.length; i++) {
            this._nodes.push(new darknet.Node(metadata, net, cfg[i], i.toString()));
        }

        if (cfg.length > 0) {
            var lastLayer = cfg[cfg.length - 1];
            for (i = 0; i < lastLayer._outputs.length; i++) {
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
        this._operator = layer.__type__;
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
        switch (layer.__type__) {
            case 'convolutional':
            case 'deconvolutional':
                layer.filters = layer.filters ? Number.parseInt(layer.filters, 10) : 1;
                layer.size = layer.size ? Number.parseInt(layer.size, 10) : 1;
                this._initializer('biases', [ layer.filters ]);
                this._initializer('weights', [ net.channels, layer.size, layer.size, layer.filters ]);
                this._batch_normalize(metadata, net, layer, layer.filters);
                this._activation(metadata, layer, 'logistic');
                break;
            case 'connected':
                layer.output = layer.output ? Number.parseInt(layer.output, 10) : 1; 
                this._initializer('biases', [ layer.output ]);
                this._initializer('weights');
                this._batch_normalize(metadata, net, layer, layer.output);
                this._activation(metadata, layer, 'logistic');
                break;
            case 'crnn':
                this._batch_normalize(metadata, net, layer);
                this._activation(metadata, layer, "logistic");
                break;
            case 'rnn':
                layer.output = layer.output ? Number.parseInt(layer.output, 10) : 1; 
                this._batch_normalize(metadata, net, layer, layer.output);
                this._activation(metadata, layer, "logistic");
                break;
            case 'gru':
                this._batch_normalize(metadata, net, layer);
                break;
            case 'lstm':
                this._batch_normalize(metadata, net, layer);
                break;
            case 'shortcut':
                this._activation(metadata, net, layer, "linear");
                break;
            case 'batch_normalize':
                this._initializer('scale', [ layer.size ]);
                this._initializer('mean', [ layer.size ]);
                this._initializer('variance', [ layer.size ]);
                break;
        }

        switch (layer.__type__) {
            case 'shortcut':
                delete layer.from;
                break;
            case 'route':
                delete layer.layers;
                break;
        }
        for (var key of Object.keys(layer)) {
            if (key != '__type__' && key != '_inputs' && key != '_outputs') {
                this._attributes.push(new darknet.Attribute(metadata, this._operator, key, layer[key]));
            }
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
        var schema = this._metadata.getSchema(this._operator);
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
        var id = this._name.toString() + '_' + name;
        this._inputs.push(new darknet.Parameter(name, true, [
            new darknet.Argument(id, null, new darknet.Tensor(id, shape))
        ]));
    }

    _batch_normalize(metadata, net, layer, size) {
        if (layer.batch_normalize == "1") {
            var batch_normalize_layer = { __type__: 'batch_normalize', _inputs: [], _outputs: [], size: size || 0 };
            this._chain.push(new darknet.Node(metadata, net, batch_normalize_layer, this._name + ':batch_normalize'));
            delete layer.batch_normalize;
        }
    }

    _activation(metadata, net, layer, defaultValue) {
        if (layer.activation && layer.activation != defaultValue) {
            this._chain.push(new darknet.Node(metadata, net, { __type__: layer.activation, _inputs: [], _outputs: [] }, this._name + ':activation'));
            delete layer.activation;
        }
    }
};

darknet.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;

        var intValue = Number.parseInt(this._value, 10);
        if (!Number.isNaN(this._value - intValue)) {
            this._value = intValue;
        }
        else {
            var floatValue = Number.parseFloat(this._value);
            if (!Number.isNaN(this._value - floatValue)) {
                this._value = floatValue;
            }
        }

        var schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            if (schema.type == 'boolean') {
                switch (this._value) {
                    case 0: this._value = false; break;
                    case 1: this._value = true; break;
                }
            }

            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default'))
            {
                if (this._value == schema.default) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
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
        return (this.dataType || '?') + this._shape.toString();
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
        this._map = {};
        this._attributeCache = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

darknet.CfgReader = class {

    constructor(text) {
        this._lines = text.split('\n');
        this._line = 0;
    }

    read() {
        var options = [];
        var section = null;
        while (this._line < this._lines.length) {
            var line = this._lines[this._line];
            line = line.replace(/\s/g, '');
            if (line.length > 0) {
                switch (line[0]) {
                    case '#':
                    case ';':
                        break;
                    case '[':
                        section = {};
                        section.__type__ = line[line.length - 1] === ']' ? line.substring(1, line.length - 1) : line.substring(1);
                        options.push(section);
                        break;
                    default:
                        if (section) {
                            var property = line.split('=');
                            if (property.length != 2) {
                                throw new darknet.Error("Invalid cfg '" + line + "' at line " + (this._line + 1).toString() + ".");
                            }
                            var key = property[0].trim();
                            var value = property[1].trim();
                            section[key] = value;
                        }
                        break;
                }
            }
            this._line++;
        }
        return options;
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
