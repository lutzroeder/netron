// Experimental

const hailo = {};

hailo.ModelFactory = class {

    match(context) {
        const container = hailo.Container.open(context);
        if (container) {
            context.type = container.type;
            context.target = container;
        }
    }

    filter(context, type) {
        if (context.type === 'hailo.metadata' && (type === 'hailo.configuration' || type === 'npz' || type === 'onnx.proto')) {
            return false;
        }
        if (context.type === 'hailo.configuration' && type === 'npz') {
            return false;
        }
        return true;
    }

    async open(context) {
        const metadata = await context.metadata('hailo-metadata.json');
        const target = context.target;
        await target.read();
        return new hailo.Model(metadata, target);
    }
};

hailo.Model = class {

    constructor(metadata, container) {
        const configuration = container.configuration;
        this.graphs = [new hailo.Graph(metadata, configuration, container.weights)];
        this.name = configuration && configuration.name || "";
        this.format = container.format + (container.metadata && container.metadata.sdk_version ? ` v${container.metadata.sdk_version}` : '');
        this.metadata = [];
        if (container.metadata && container.metadata.state) {
            this.metadata.push(new hailo.Argument('state', container.metadata.state));
        }
    }
};

hailo.Graph = class {

    constructor(metadata, configuration, weights) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new hailo.Value(name, type || null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new hailo.Value(name, type || null, tensor || null));
            } else if (tensor) {
                throw new hailo.Error(`Duplicate value '${name}'.`);
            } else if (type && !type.equals(values.get(name).type)) {
                throw new hailo.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const layers = Object.entries(configuration.layers || {}).map(([name, value]) => {
            value.name = name;
            return value;
        });
        const inputs = new Set();
        for (const layer of layers) {
            switch (layer.type) {
                case 'input_layer': {
                    for (let i = 0; i < layer.output.length; i++) {
                        const shape = Array.isArray(layer.output_shapes) && layer.output_shapes.length > 0 ? layer.output_shapes[0] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                        const output = layer.output[i];
                        if (!inputs.has(output)) {
                            const name = `${layer.name}\n${output}`;
                            const argument = new hailo.Argument('input', [values.map(name, type)]);
                            this.inputs.push(argument);
                            inputs.add(output);
                        }
                    }
                    break;
                }
                case 'output_layer': {
                    for (let i = 0; i < layer.input.length; i++) {
                        const shape = Array.isArray(layer.input_shapes) && layer.input_shapes.length > 0 ? layer.input_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                        const input = layer.input[i];
                        const name = `${input}\n${layer.name}`;
                        const argument = new hailo.Argument('output', [values.map(name, type)]);
                        this.outputs.push(argument);
                    }
                    break;
                }
                default: {
                    const node = new hailo.Node(metadata, layer, values, weights.get(layer.name));
                    this.nodes.push(node);
                    break;
                }
            }
        }
    }
};

hailo.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

hailo.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hailo.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

hailo.Node = class {

    constructor(metadata, layer, values, weights) {
        weights = weights || new Map();
        this.name = layer.name || '';
        this.type = metadata.type(layer.type);
        if (layer.type === 'activation') {
            const name = layer.params.activation || layer.name || '';
            this.type = { ...this.type, name };
        }
        this.inputs = layer.input.map((name, index) => {
            const shape = layer.input_shapes ? layer.input_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            name = `${name}\n${layer.name}`;
            return new hailo.Argument("input", [values.map(name, type, null)]);
        });
        const layer_params = layer.params ? Object.entries(layer.params) : [];
        const params_list = layer_params.reduce((acc, [name, value]) => {
            const schema = metadata.attribute(layer.type, name) || {};
            if (schema.visible) {
                const label = schema.label ? schema.label : name;
                if (!weights.has(label)) {
                    const array = weights.get(label);
                    const tensor = new hailo.Tensor(array, value);
                    acc.push(new hailo.Argument(label, [values.map('', tensor.type, tensor)]));
                }
            }
            return acc;
        }, []);
        const params_from_npz = Array.from(weights).filter(([, value]) => value).map(([name, value]) => {
            const tensor = new hailo.Tensor(value);
            return new hailo.Argument(name, [values.map('', tensor.type, tensor)]);
        });
        this.inputs = this.inputs.concat(params_list).concat(params_from_npz);
        this.outputs = (layer.output || []).map((name, index) => {
            const shape = layer.output_shapes ? layer.output_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            name = `${layer.name}\n${name}`;
            return new hailo.Argument("output", [values.map(name, type, null)]);
        });
        this.attributes = [];
        const attrs = Object.assign(layer.params || {}, { original_names: layer.original_names || [] });
        for (const [name, value] of Object.entries(attrs)) {
            const schema = metadata.attribute(layer.type, name);
            const type = schema && schema.type ? schema.type : '';
            const visible = name === 'original_names' || (schema && schema.visible === false) ? false : true;
            const attribute = new hailo.Argument(name, value, type, visible);
            this.attributes.push(attribute);
        }
        this.chain = [];
        if (layer && layer.params && layer.params.activation && layer.params.activation !== 'linear' && layer.type !== 'activation') {
            const activation = {
                type: layer.params.activation,
                name: layer.params.activation,
                input: [],
                output: []
            };
            const node = new hailo.Node(metadata, activation, values.map);
            this.chain.push(node);
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
            this.layout = this.type.dataType === 'string' || this.type.dataType === 'object' ? '|' : array.dtype.byteorder;
            this.values = this.type.dataType === 'string' || this.type.dataType === 'object' || this.type.dataType === 'void' ? array.tolist() : array.tobytes();
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
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

hailo.Container = class {

    static open(context) {
        const identifier = context.identifier;
        const basename = identifier.split('.');
        basename.pop();
        if (identifier.toLowerCase().endsWith('.hn')) {
            if (basename.length > 1 && (basename[basename.length - 1] === 'native' || basename[basename.length - 1] === 'fp')) {
                basename.pop();
            }
            const configuration = context.peek('json');
            if (configuration && configuration.name && configuration.net_params && configuration.layers) {
                return new hailo.Container(context, 'hailo.configuration', basename.join('.'), configuration, null);
            }
        } else if (identifier.toLowerCase().endsWith('.metadata.json')) {
            basename.pop();
            const metadata = context.peek('json');
            if (metadata && metadata.state && metadata.hn) {
                return new hailo.Container(context, 'hailo.metadata', basename.join('.'), null, metadata);
            }
        }
        return null;
    }

    constructor(context, type, basename, configuration, metadata) {
        this.type = type;
        this.context = context;
        this.basename = basename;
        this.configuration = configuration;
        this.metadata = metadata;
    }

    async _request(name, type) {
        try {
            const content = await this.context.fetch(name);
            if (content) {
                return content.read(type);
            }
        } catch {
            // continue regardless of error
        }
        return null;
    }

    async read() {
        this.format = 'Hailo NN';
        this.weights = new Map();
        if (!this.metadata) {
            this.metadata = await this._request(`${this.basename}.metadata.json`, 'json');
        }
        if (this.metadata) {
            this.format = 'Hailo Archive';
            this.configuration = await this._request(this.metadata.hn, 'json');
            if (!this.configuration) {
                throw new hailo.Error("Archive does not contain '.nn' configuration.");
            }
            let extension = '';
            switch (this.metadata.state) {
                case 'fp_optimized_model': extension = '.fpo.npz'; break;
                case 'quantized_model': extension = '.q.npz'; break;
                case 'compiled_model': extension = '.q.npz'; break;
                default: extension = '.npz'; break;
            }
            const entries = await this._request(this.basename + extension, 'npz');
            if (entries && entries.size > 0) {
                const inputs = new Set([
                    'kernel', 'bias',
                    'input_activation_bits', 'output_activation_bits',
                    'weight_bits', 'bias_decomposition'
                ]);
                for (const [name, value] of entries) {
                    const key = name.split('.').slice(0, -1).join('.');
                    const match = key.match(/.*?(?=:[0-9])/);
                    if (match) {
                        const path = match[0].split('/');
                        if (inputs.has(path[2])) {
                            const layer = `${path[0]}/${path[1]}`;
                            if (!this.weights.has(layer)) {
                                this.weights.set(layer, new Map());
                            }
                            const weights = this.weights.get(layer);
                            weights.set(path[2], value);
                        }
                    }
                }
            }
        }
        delete this.context;
        delete this.basename;
    }
};

hailo.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Hailo model.';
    }
};

export const ModelFactory = hailo.ModelFactory;

