
// Experimental

var hailo = hailo || {};
var json = json || require('./json');

hailo.ModelFactory = class {

    match(context) {
        return hailo.Container.open(context);
    }

    open(context, match) {
        return context.metadata('hailo-metadata.json').then((metadata) => {
            return new hailo.Model(metadata, match);
        });
    }
};

hailo.Model = class {

    constructor(metadata, container) {
        const configuration = container.configuration;
        this._graphs = [ new hailo.Graph(metadata, configuration) ];
        this._name = configuration && configuration.name || "";
        this._format = container.format + (container.metadata && container.metadata.sdk_version ? ' v' + container.metadata.sdk_version : '');
        this._metadata = [];
        if (container.metadata && container.metadata.state) {
            this._metadata.push({ name: 'state', value: container.metadata.state });
        }
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata;
    }
};

hailo.Graph = class {

    constructor(metadata, configuration) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const mapLayersObjectToArray = (layers_object) => {
            const entries = Object.entries(layers_object);
            return entries.map(([layer_name, layer_object]) => {
                layer_object.name = layer_name;
                return layer_object;
            });
        };
        const layers = mapLayersObjectToArray(configuration.layers || {}) || [];
        for (const layer of layers) {
            switch (layer.type) {
                case 'input_layer': {
                    for (let i = 0; i < layer.output.length; i++) {
                        const shape = layer.output_shapes ? layer.output_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape()) : null;
                        const argument = new hailo.Argument(layer.name, type);
                        const parameter = new hailo.Parameter('input', true, [ argument ]);
                        this._inputs.push(parameter);
                    }
                    break;
                }
                case 'output_layer': {
                    for (let i = 0; i < layer.input.length; i++) {
                        const shape = layer.input_shapes ? layer.input_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape()) : null;
                        const argument = new hailo.Argument(layer.input[i], type);
                        const parameter = new hailo.Parameter('output', true, [ argument ]);
                        this._outputs.push(parameter);
                    }
                    break;
                }
                default: {
                    const node = new hailo.Node(metadata, layer);
                    this._nodes.push(node);
                    break;
                }
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

hailo.Parameter = class {

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

hailo.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hailo.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
    }

    get name() {
        return this._name;
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

hailo.Node = class {

    constructor(metadata, layer) {
        const getNodeInputs = (layer) => {
            const inputs = layer.input.map((name, index) => {
                const shape = layer.input_shapes ? layer.input_shapes[index] : null;
                const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                const argument = new hailo.Argument(name, type);
                return new hailo.Parameter("input", true, [ argument ]);
            });
            const getParams = (params_array) => {
                return params_array.reduce((acc, obj) => {
                    const name = obj[0];
                    const value = obj[1];
                    const schema = metadata.attribute(layer.type, name) || {};
                    if (schema.visible) {
                        const label = schema.label ? schema.label : name;
                        const shape = new hailo.TensorShape(value, schema.type);
                        const type = new hailo.TensorType('?', shape);
                        const tensor = new hailo.Tensor(type, value);
                        acc.push(new hailo.Parameter(label, true, [
                            new hailo.Argument(label, type, tensor)
                        ]));
                    }
                    return acc;
                }, []);
            };
            const params_array = layer.params ? Object.entries(layer.params) : [];
            const params_list = getParams(params_array || []);
            return inputs.concat(params_list);
        };
        this._name = layer.name || '';
        this._type = metadata.type(layer.type);
        if (layer.type === 'activation') {
            this._type = Object.assign({}, this._type, { name: layer.params.activation || layer.name || '' });
        }
        this._inputs = getNodeInputs(layer);
        this._outputs = (layer.output || []).map((_, index) => {
            const shape = layer.output_shapes ? layer.output_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            const argument = new hailo.Argument(layer.name, type);
            return new hailo.Parameter("output", true, [ argument ]);
        });
        const attrs = Object.assign(layer.params || {}, { original_names: layer.original_names || [] });
        this._attributes = Object.entries(attrs).map((entry) => new hailo.Attribute(metadata.attribute(layer.type, entry[0]), entry[0], entry[1]));
        this._chain = [];
        if (layer && layer.params && layer.params.activation && layer.params.activation !== 'linear' && layer.type !== 'activation') {
            const node = new hailo.Node(metadata, {
                type: layer.params.activation,
                name: layer.params.activation,
                input: [],
                output: []
            });
            this._chain.push(node);
        }
    }

    get type() {
        return this._type;
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

    get attributes() {
        return this._attributes;
    }

    get description() {
        return this._descripton;
    }

    get chain() {
        return this._chain;
    }
};

hailo.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        this._type = metadata && metadata.type ? metadata.type : '';
        this._visible = metadata && metadata.visible !== false ? true : false;
        if (name === 'original_names') {
            this._visible = false;
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
        return this._visible !== false;
    }
};

hailo.Tensor = class {

    constructor(type) {
        this._type = type;
    }

    get type() {
        return this._type;
    }
};

hailo.TensorType = class {

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

hailo.TensorShape = class {

    constructor(dimensions, type) {
        this._dimensions = dimensions;
        this._type = type || '?';
    }

    get dimensions() {
        return this._dimensions;
    }

    get type() {
        return this._type;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length === 0) {
                return '';
            }
            return '[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

hailo.Container = class {

    static open(context) {
        const parts = context.identifier.split('.');
        const extension = parts.pop().toLowerCase();
        const basename = parts.join('.');
        let format = '';
        let configuration = null;
        let metadata = null;
        switch (extension) {
            case 'hn': {
                format = 'Hailo NN';
                configuration = context.open('json');
                break;
            }
            case 'har': {
                const read = (name) => {
                    const entries = context.entries('tar');
                    const stream = entries.get(name);
                    if (stream) {
                        try {
                            const buffer = stream.peek();
                            const decoder = new TextDecoder('utf-8');
                            const content = decoder.decode(buffer);
                            return JSON.parse(content);
                        } catch (err) {
                            // continue regardless of error
                        }
                    }
                    return null;
                };
                format = 'Hailo Archive';
                configuration = read(basename + '.hn');
                metadata = read(basename + '.metadata.json');
                break;
            }
            default: {
                break;
            }
        }
        if (configuration && configuration.name && configuration.net_params && configuration.layers) {
            return new hailo.Container(format, configuration, metadata);
        }
        return null;
    }

    constructor(format, configuration, metadata) {
        this.format = format;
        this.configuration = configuration;
        this.metadata = metadata;
    }
};

hailo.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Hailo model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = hailo.ModelFactory;
}
