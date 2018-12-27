
/*jshint esversion: 6 */

var darknet = darknet || {};
var base = base || require('./base');

darknet.ModelFactory = class {

    match(context, host) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'cfg') {
            return true;
        }
        return false;
    }

    open(context, host, callback) {
        darknet.Metadata.open(host, (err, metadata) => {
            var identifier = context.identifier;
            try {
                var reader = new darknet.CfgReader(context.text);
                var cfg = reader.read();
                var model = new darknet.Model(metadata, cfg);
                callback(null, model);
                return;
            }
            catch (error) {
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                callback(new darknet.Error(message + " in '" + identifier + "'."), null);
                return;
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

        var net = cfg.shift();

        var inputType = null;
        if (net && net.hasOwnProperty('width') && net.hasOwnProperty('height') && net.hasOwnProperty('channels')) {
            var width = Number.parseInt(net.width);
            var height = Number.parseInt(net.height);
            var channels = Number.parseInt(net.channels);
            inputType = new darknet.TensorType('float32', new darknet.TensorShape([ width, height, channels ]));
        }

        var input = 'input';
        this._inputs.push(new darknet.Argument(input, true, [
            new darknet.Connection(input, inputType, null)
        ]));

        cfg.forEach((layer, index) => {
            layer._outputs = [ index.toString() ];
        });

        var inputs = [ 'input' ];
        cfg.forEach((layer, index) => {
            layer._inputs = inputs;
            inputs = [ index.toString() ];
            switch (layer.__type__) {
                case 'shortcut':
                    var shortcut = cfg[index + Number.parseInt(layer.from, 10)]._outputs[0];
                    layer._inputs.push(shortcut);
                    break;
                case 'route':
                    var routes = layer.layers.split(',').map((route) => Number.parseInt(route.trim(), 10));
                    layer._inputs = routes.map((route) => {
                        var layer = (route < 0) ? index + route : route;
                        return cfg[layer]._outputs[0];
                    });
                    break;
            }
        });
        cfg.forEach((layer, index) => {
            this._nodes.push(new darknet.Node(metadata, layer, index.toString()));
        });

        if (cfg.length > 0) {
            var lastLayer = cfg[cfg.length - 1];
            lastLayer._outputs.forEach((output, index) => {
                this._outputs.push(new darknet.Argument('output' + (index > 1 ? index.toString() : ''), true, [
                    new darknet.Connection(output, null, null)
                ]));
            });
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

darknet.Argument = class {

    constructor(name, visible, connections) {
        this._name = name;
        this._visible = visible;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get connections() {
        return this._connections;
    }
};

darknet.Connection = class {

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

    constructor(metadata, layer, name) {
        this._name = name;
        this._metadata = metadata;
        this._operator = layer.__type__;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        if (layer._inputs && layer._inputs.length > 0) {
            this._inputs.push(new darknet.Argument(layer._inputs.length <= 1 ? 'input' : 'inputs', true, layer._inputs.map((input) => {
                return new darknet.Connection(input, null, null);
            })));
        }
        if (layer._outputs && layer._outputs.length > 0) {
            this._outputs.push(new darknet.Argument(layer._outputs.length <= 1 ? 'output' : 'outputs', true, layer._outputs.map((output) => {
                return new darknet.Connection(output, null, null);
            })));
        }
        switch (layer.__type__) {
            case 'convolutional':
            case 'deconvolutional':
                this._initializer('biases');
                this._initializer('weights');
                this._batch_normalize(metadata, layer);
                this._activation(metadata, layer, 'logistic');
                break;
            case 'connected':
                this._initializer('biases');
                this._initializer('weights');
                this._batch_normalize(metadata, layer);
                this._activation(metadata, layer, 'logistic');
                break;
            case 'crnn':
                this._batch_normalize(metadata, layer);
                this._activation(metadata, layer, "logistic");
                break;
            case 'rnn':
                this._batch_normalize(metadata, layer);
                this._activation(metadata, layer, "logistic");
                break;
            case 'gru':
                this._batch_normalize(metadata, layer);
                break;
            case 'lstm':
                this._batch_normalize(metadata, layer);
                break;
            case 'shortcut':
                this._activation(metadata, layer, "linear");
                break;
            case 'batch_normalize':
                this._initializer('scale');
                this._initializer('mean');
                this._initializer('variance');
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
        Object.keys(layer).forEach((key) => {
            if (key != '__type__' && key != '_inputs' && key != '_outputs') {
                this._attributes.push(new darknet.Attribute(metadata, this._operator, key, layer[key]));
            }
        });
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : null;
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

    _initializer(name) {
        var id = this._name.toString() + '_' + name;
        this._inputs.push(new darknet.Argument(name, true, [
            new darknet.Connection(id, null, new darknet.Tensor(id))
        ]));
    }

    _batch_normalize(metadata, layer) {
        if (layer.batch_normalize == "1") {
            var batch_normalize_layer = { __type__: 'batch_normalize', _inputs: [], _outputs: [] };
            this._chain.push(new darknet.Node(metadata, batch_normalize_layer, this._name + ':batch_normalize'));
            delete layer.batch_normalize;
        }
    }

    _activation(metadata, layer, defaultValue) {
        if (layer.activation && layer.activation != defaultValue) {
            this._chain.push(new darknet.Node(metadata, { __type__: layer.activation, _inputs: [], _outputs: [] }, this._name + ':activation'));
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

            if (schema.hasOwnProperty('visible') && !schema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default'))
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

    constructor(id) {
        this._id = id;
    }

    get name() {
        return this._id;
    }

    get type() {
        return null;
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

    static open(host, callback) {
        if (darknet.Metadata._metadata) {
            callback(null, darknet.Metadata._metadata);
            return;
        }
        host.request(null, 'darknet-metadata.json', 'utf-8', (err, data) => {
            darknet.Metadata._metadata = new darknet.Metadata(data);
            callback(null, darknet.Metadata._metadata);
            return;
        });
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.__attributesMap) {
                schema.__attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.__attributesMap[attribute.name] = attribute;
                });
            }
            return schema.__attributesMap[name];
        }
        return null;
    }
};

darknet.CfgReader = class {

    constructor(text) {
        this._lines = text.split('\n');
        this._line = 0;
    }

    read() {
        var array = [];
        var item = {};
        while (this._line < this._lines.length) {
            var line = this._lines[this._line];
            line = line.split('#')[0].trim();
            if (line.length > 0) {
                if (line.length > 3 && line[0] == '[' && line[line.length - 1] == ']') {
                    if (item.__type__) {
                        array.push(item);
                        item = {};
                    }
                    item.__type__ = line.substring(1, line.length - 1);
                }
                else {
                    var property = line.split('=');
                    if (property.length == 2) {
                        var key = property[0].trim();
                        var value = property[1].trim();
                        item[key] = value;
                    }
                    else {
                        throw new darknet.Error("Invalid cfg \'" + line + "\' at line " + (this._line + 1).toString() + ".");
                    }
                }
            }
            this._line++;
        }
        if (item.__type__) {
            array.push(item);
        }
        return array;
    }
}

darknet.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Darknet model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = darknet.ModelFactory;
}
