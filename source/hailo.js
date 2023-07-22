
// Experimental

var hailo = hailo || {};
var json = json || require('./json');

hailo.ModelFactory = class {

    match(context) {
        return hailo.Container.open(context);
    }

    async open(context, target) {
        const metadata = await context.metadata('hailo-metadata.json');
        return new hailo.Model(metadata, target);
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
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new hailo.Value(name, type || null, tensor);
            }
            if (!args.has(name)) {
                args.set(name, new hailo.Value(name, type || null, tensor || null));
            } else if (tensor) {
                throw new hailo.Error("Duplicate value '" + name + "'.");
            } else if (type && !type.equals(args.get(name).type)) {
                return new hailo.Value(name, type, null);
            }
            return args.get(name);
        };
        const layers = Object.entries(configuration.layers || {}).map((entry) => {
            entry[1].name = entry[0];
            return entry[1];
        });
        for (const layer of layers) {
            switch (layer.type) {
                case 'input_layer': {
                    for (let i = 0; i < layer.output.length; i++) {
                        const shape = layer.output_shapes ? layer.output_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                        const argument = new hailo.Argument('input', [ arg(layer.name, type) ]);
                        this._inputs.push(argument);
                    }
                    break;
                }
                case 'output_layer': {
                    for (let i = 0; i < layer.input.length; i++) {
                        const shape = layer.input_shapes ? layer.input_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                        const argument = new hailo.Argument('output', [ arg(layer.input[i], type) ]);
                        this._outputs.push(argument);
                    }
                    break;
                }
                default: {
                    const node = new hailo.Node(metadata, layer, arg);
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

hailo.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

hailo.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hailo.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, layer, arg) {
        this._name = layer.name || '';
        this._type = metadata.type(layer.type);
        if (layer.type === 'activation') {
            this._type = Object.assign({}, this._type, { name: layer.params.activation || layer.name || '' });
        }
        this._inputs = layer.input.map((name, index) => {
            const shape = layer.input_shapes ? layer.input_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            return new hailo.Argument("input", [ arg(name, type) ]);
        });
        const getParams = (params_array) => {
            return params_array.reduce((acc, obj) => {
                const name = obj[0];
                const value = obj[1];
                const schema = metadata.attribute(layer.type, name) || {};
                if (schema.visible) {
                    const label = schema.label ? schema.label : name;
                    const shape = new hailo.TensorShape(value);
                    const type = new hailo.TensorType('?', shape);
                    const tensor = new hailo.Tensor(type, value);
                    acc.push(new hailo.Argument(label, [ arg('', type, tensor) ]));
                }
                return acc;
            }, []);
        };
        const params_list = getParams(layer.params ? Object.entries(layer.params) : []);
        this._inputs = this._inputs.concat(params_list);
        this._outputs = (layer.output || []).map((_, index) => {
            const shape = layer.output_shapes ? layer.output_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            return new hailo.Argument("output", [ arg(layer.name, type) ]);
        });
        const attrs = Object.assign(layer.params || {}, { original_names: layer.original_names || [] });
        this._attributes = Object.entries(attrs).map((entry) => new hailo.Attribute(metadata.attribute(layer.type, entry[0]), entry[0], entry[1]));
        this._chain = [];
        if (layer && layer.params && layer.params.activation && layer.params.activation !== 'linear' && layer.type !== 'activation') {
            const activation = {
                type: layer.params.activation,
                name: layer.params.activation,
                input: [],
                output: []
            };
            const node = new hailo.Node(metadata, activation, arg);
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
        if (metadata && metadata.visible === false) {
            this._visible = false;
        }
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
        this.dataType = dataType;
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return (this.dataType || '?') + this.shape.toString();
    }
};

hailo.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length) {
                return obj.dimensions.every((value, index) => this.dimensions[index] === value);
            }
            const a = this.dimensions.filter((value, index) => index === 0 || index === this.dimensions.length - 1 || value !== 1);
            const b = obj.dimensions.filter((value, index) => index === 0 || index === obj.dimensions.length - 1 || value !== 1);
            if (a.length === b.length) {
                return a.every((value, index) => b[index] === value);
            }
        }
        return false;
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
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
