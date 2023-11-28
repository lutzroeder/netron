// Experimental

var hailo = {};

hailo.ModelFactory = class {

    match(context) {
        return hailo.Container.open(context);
    }

    async open(context, target) {
        const metadata = await context.metadata('hailo-metadata.json');
        await target.read();
        return new hailo.Model(metadata, target);
    }
};

hailo.Model = class {

    constructor(metadata, container) {
        const configuration = container.configuration;
        this.graphs = [ new hailo.Graph(metadata, configuration, container.weights) ];
        this.name = configuration && configuration.name || "";
        this.format = container.format + (container.metadata && container.metadata.sdk_version ? ' v' + container.metadata.sdk_version : '');
        this.metadata = [];
        if (container.metadata && container.metadata.state) {
            this.metadata.push({ name: 'state', value: container.metadata.state });
        }
    }
};

hailo.Graph = class {

    constructor(metadata, configuration, weights) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
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
                throw new hailo.Error("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        const layers = Object.entries(configuration.layers || {}).map((entry) => {
            entry[1].name = entry[0];
            return entry[1];
        });
        for (const layer of layers) {
            for (let i = 0; i < layer.input.length; i++) {
                const name = layer.input[i];
                const shape = layer.input_shapes ? layer.input_shapes[i] : null;
                const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                arg(name, type);
            }
        }
        for (const layer of layers) {
            switch (layer.type) {
                case 'input_layer':
                case 'const_input': {
                    const shape = Array.isArray(layer.output_shapes) && layer.output_shapes.length > 0 ? layer.output_shapes[0] : null;
                    const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                    const argument = new hailo.Argument('input', [ arg(layer.name, type) ]);
                    this.inputs.push(argument);
                    break;
                }
                case 'output_layer': {
                    for (let i = 0; i < layer.input.length; i++) {
                        const shape = Array.isArray(layer.input_shapes) && layer.input_shapes.length > 0 ? layer.input_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                        const argument = new hailo.Argument('output', [ arg(layer.input[i], type) ]);
                        this.outputs.push(argument);
                    }
                    break;
                }
                default: {
                    const node = new hailo.Node(metadata, layer, arg, weights.get(layer.name));
                    this.nodes.push(node);
                    break;
                }
            }
        }
    }
};

hailo.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

hailo.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hailo.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

hailo.Node = class {

    constructor(metadata, layer, arg, weights) {
        weights = weights || new Map();
        this.name = layer.name || '';
        this.type = metadata.type(layer.type);
        if (layer.type === 'activation') {
            this.type = Object.assign({}, this.type, { name: layer.params.activation || layer.name || '' });
        }
        this.inputs = layer.input.map((name, index) => {
            const shape = layer.input_shapes ? layer.input_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            return new hailo.Argument("input", [ arg(name, type) ]);
        });
        const layer_params = layer.params ? Object.entries(layer.params) : [];
        const params_list = layer_params.reduce((acc, entry) => {
            const name = entry[0];
            const value = entry[1];
            const schema = metadata.attribute(layer.type, name) || {};
            if (schema.visible) {
                const label = schema.label ? schema.label : name;
                if (!weights.has(label)) {
                    const array = weights.get(label);
                    const tensor = new hailo.Tensor(array, value);
                    acc.push(new hailo.Argument(label, [ arg('', tensor.type, tensor) ]));
                }
            }
            return acc;
        }, []);
        const params_from_npz = Array.from(weights).filter((entry) => entry[1]).map((entry) => {
            const name = entry[0];
            const tensor = new hailo.Tensor(entry[1]);
            return new hailo.Argument(name, [ arg('', tensor.type, tensor) ]);
        });
        this.inputs = this.inputs.concat(params_list).concat(params_from_npz);
        this.outputs = (layer.output || []).map((_, index) => {
            const shape = layer.output_shapes ? layer.output_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            return new hailo.Argument("output", [ arg(layer.name, type) ]);
        });
        const attrs = Object.assign(layer.params || {}, { original_names: layer.original_names || [] });
        this.attributes = Object.entries(attrs).map((entry) => new hailo.Attribute(metadata.attribute(layer.type, entry[0]), entry[0], entry[1]));
        this.chain = [];
        if (layer && layer.params && layer.params.activation && layer.params.activation !== 'linear' && layer.type !== 'activation') {
            const activation = {
                type: layer.params.activation,
                name: layer.params.activation,
                input: [],
                output: []
            };
            const node = new hailo.Node(metadata, activation, arg);
            this.chain.push(node);
        }
    }
};

hailo.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.value = value;
        this.type = metadata && metadata.type ? metadata.type : '';
        if (metadata && metadata.visible === false) {
            this.visible = false;
        }
        if (name === 'original_names') {
            this.visible = false;
        }
    }
};

hailo.Tensor = class {

    constructor(array, shape) {
        const dataType = array && array.dtype ? array.dtype.__name__ : '?';
        shape = array && array.shape ? array.shape : shape;
        this.type = new hailo.TensorType(dataType, new hailo.TensorShape(shape));
        if (array) {
            this.stride = array.strides.map((stride) => stride / array.itemsize);
            this.layout = this.type.dataType == 'string' || this.type.dataType == 'object' ? '|' : array.dtype.byteorder;
            this.values = this.type.dataType == 'string' || this.type.dataType == 'object' ? array.tolist() : array.tobytes();
        }
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
        const identifier = context.identifier;
        const parts = identifier.split('.');
        const extension = parts.pop().toLowerCase();
        const basename = parts.join('.');
        if (extension === 'hn') {
            const configuration = context.peek('json');
            if (configuration && configuration.name && configuration.net_params && configuration.layers) {
                return new hailo.Container(context, basename, configuration);
            }
        }
        return null;
    }

    constructor(context, basename, configuration) {
        this._context = context;
        this._basename = basename;
        this.configuration = configuration;
    }

    async _request(extension, type) {
        try {
            const content = await this._context.fetch(this._basename + extension);
            if (content) {
                return content.read(type);
            }
        } catch (error) {
            // continue regardless of error
        }
        return null;
    }

    async read() {
        this.format = 'Hailo NN';
        this.weights = new Map();
        this.metadata = await this._request('.metadata.json', 'json');
        if (this.metadata) {
            this.format = 'Hailo Archive';
            let extension = undefined;
            switch (this.metadata.state) {
                case 'fp_optimized_model': extension = '.fpo.npz'; break;
                case 'quantized_model': extension = '.q.npz'; break;
                case 'compiled_model': extension = '.q.npz'; break;
                default: extension = '.npz'; break;
            }
            const entries = await this._request(extension, 'npz');
            if (entries && entries.size > 0) {
                const inputs = new Set([
                    'kernel', 'bias',
                    'input_activation_bits', 'output_activation_bits', 'weight_bits', 'bias_decomposition'
                ]);
                for (const entry of entries) {
                    const key = entry[0].split('.').slice(0, -1).join('.');
                    const match = key.match(/.*?(?=:[0-9])/);
                    if (match) {
                        const path = match[0].split('/');
                        if (inputs.has(path[2])) {
                            const layer = path[0] + '/' + path[1];
                            if (!this.weights.has(layer)) {
                                this.weights.set(layer, new Map());
                            }
                            const weights = this.weights.get(layer);
                            weights.set(path[2], entry[1]);
                        }
                    }
                }
            }
        }
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
