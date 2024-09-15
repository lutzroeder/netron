
const espresso = {};

espresso.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.espresso.net')) {
            const obj = context.peek('json');
            if (obj && Array.isArray(obj.layers) && obj.format_version) {
                context.type = 'espresso.net';
                context.target = obj;
                return;
            }
        }
        if (identifier.endsWith('.espresso.shape')) {
            const obj = context.peek('json');
            if (obj && obj.layer_shapes) {
                context.type = 'espresso.shape';
                context.target = obj;
                return;
            }
        }
        if (identifier.endsWith('.espresso.weights')) {
            context.type = 'espresso.weights';
            context.target = context.read('binary');
        }
    }

    filter(context, type) {
        if (context.type === 'espresso.net' && (type === 'espresso.weights' || type === 'espresso.shape' || type === 'coreml.metadata.mlmodelc')) {
            return false;
        }
        if (context.type === 'espresso.shape' && (type === 'espresso.weights' || type === 'coreml.metadata.mlmodelc')) {
            return false;
        }
        return true;
    }

    async open(context) {
        const metadata = await context.metadata('espresso-metadata.json');
        switch (context.type) {
            case 'espresso.net': {
                const reader = new espresso.Reader(context.target, null, null);
                await reader.read(context);
                return new espresso.Model(metadata, reader);
            }
            case 'espresso.weights': {
                const reader = new espresso.Reader(null, context.target, null);
                await reader.read(context);
                return new espresso.Model(metadata, reader);
            }
            case 'espresso.shape': {
                const reader = new espresso.Reader(null, null, context.target);
                await reader.read(context);
                return new espresso.Model(metadata, reader);
            }
            default: {
                throw new espresso.Error(`Unsupported Core ML format '${context.type}'.`);
            }
        }
    }
};

espresso.Model = class {

    constructor(metadata, reader) {
        this.format = reader.format;
        this.metadata = [];
        this.graphs = [new espresso.Graph(metadata, reader)];
        if (reader.version) {
            this.version = reader.version;
        }
        if (reader.description) {
            this.description = reader.description;
        }
        for (const argument of reader.properties) {
            this.metadata.push(argument);
        }
    }
};

espresso.Graph = class {

    constructor(metadata, reader) {
        this.name = '';
        this.type = reader.type;
        for (const value of reader.values.values()) {
            const name = value.name;
            const type = value.type;
            const description = value.description;
            const initializer = value.initializer;
            if (!value.value) {
                value.value = new espresso.Value(name, type, description, initializer);
            }
        }
        this.inputs = reader.inputs.map((argument) => {
            const values = argument.value.map((value) => value.value);
            return new espresso.Argument(argument.name, values, null, argument.visible);
        });
        this.outputs = reader.outputs.map((argument) => {
            const values = argument.value.map((value) => value.value);
            return new espresso.Argument(argument.name, values, null, argument.visible);
        });
        for (const obj of reader.nodes) {
            const attributes = obj.attributes;
            switch (obj.type) {
                case 'loop':
                    attributes.conditionNetwork = new espresso.Graph(attributes.conditionNetwork);
                    attributes.bodyNetwork = new espresso.Graph(attributes.bodyNetwork);
                    break;
                case 'branch':
                    attributes.ifBranch = new espresso.Graph(attributes.ifBranch);
                    attributes.elseBranch = new espresso.Graph(attributes.elseBranch);
                    break;
                default:
                    break;
            }
        }
        this.nodes = reader.nodes.map((obj) => new espresso.Node(metadata, obj));
    }
};

espresso.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

espresso.Value = class {

    constructor(name, type, description, initializer) {
        if (typeof name !== 'string') {
            throw new espresso.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.description = description || null;
        this.initializer = initializer || null;
        this.quantization = initializer ? initializer.quantization : null;
    }
};

espresso.Node = class {

    constructor(metadata, obj) {
        if (!obj.type) {
            throw new Error('Undefined node type.');
        }
        const type = metadata.type(obj.type);
        this.type = type ? { ...type } : { name: obj.type };
        this.type.name = obj.type.split(':').pop();
        this.name = obj.name || '';
        this.description = obj.description || '';
        this.inputs = (obj.inputs || []).map((argument) => {
            const values = argument.value.map((value) => value.value);
            return new espresso.Argument(argument.name, values, null, argument.visible);
        });
        this.outputs = (obj.outputs || []).map((argument) => {
            const values = argument.value.map((value) => value.value);
            return new espresso.Argument(argument.name, values, null, argument.visible);
        });
        this.attributes = Object.entries(obj.attributes || []).map(([name, value]) => {
            const schema = metadata.attribute(obj.type, name);
            let type = null;
            let visible = true;
            if (schema) {
                type = schema.type ? schema.type : type;
                if (schema.visible === false) {
                    visible = false;
                } else if (schema.default !== undefined) {
                    if (Array.isArray(value)) {
                        value = value.map((item) => Number(item));
                    }
                    if (typeof value === 'bigint') {
                        value = value.toNumber();
                    }
                    if (JSON.stringify(schema.default) === JSON.stringify(value)) {
                        visible = false;
                    }
                }
            }
            return new espresso.Argument(name, value, type, visible);
        });
        if (Array.isArray(obj.chain)) {
            this.chain = obj.chain.map((obj) => new espresso.Node(metadata, obj));
        }
    }
};

espresso.Tensor = class {

    constructor(type, data, quantization, category) {
        this.type = type;
        this.values = data;
        this.quantization = quantization;
        this.category = category;
        this.encoding = '<';
    }
};

espresso.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape || new espresso.TensorShape([]);
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

espresso.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions) &&
            this.dimensions.length === obj.dimensions.length &&
            obj.dimensions.every((value, index) => this.dimensions[index] === value);
    }

    toString() {
        return Array.isArray(this.dimensions) && this.dimensions.length > 0 ?
            `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]` : '';
    }
};

espresso.Reader = class {

    constructor(net, weights, shape) {
        this.targets = [net, shape, weights];
    }

    async read(context) {
        this.format = 'Espresso';
        this.properties = [];
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        let [net, shape, weights] = this.targets;
        delete this.targets;
        if (!net) {
            const name = context.identifier.replace(/\.espresso\.(net|weights|shape)$/i, '.espresso.net');
            const content = await context.fetch(name);
            net = content.read('json');
        }
        this.shapes = new Map();
        if (!shape) {
            const name = context.identifier.replace(/\.espresso\.(net|weights|shape)$/i, '.espresso.shape');
            try {
                const content = await context.fetch(name);
                shape = content.read('json');
            } catch {
                // continue regardless of error
            }
        }
        if (shape && shape.layer_shapes) {
            for (const [name, value] of Object.entries(shape.layer_shapes)) {
                const dimensions = [value.n, value.k, value.w, value.h];
                const shape = new espresso.TensorShape(dimensions);
                this.shapes.set(name, shape);
            }
        }
        this.blobs = new Map();
        if (!weights) {
            const name = net && net.storage ? net.storage : context.identifier.replace(/\.espresso\.(net|weights|shape)$/i, '.espresso.weights');
            try {
                const content = await context.fetch(name);
                weights = content.read('binary');
            } catch {
                // continue regardless of error
            }
        }
        if (weights) {
            const reader = weights;
            const length = reader.uint64().toNumber();
            for (let i = 0; i < length; i++) {
                const key = reader.uint64().toNumber();
                const size = reader.uint64().toNumber();
                this.blobs.set(key, size);
            }
            for (const [key, size] of this.blobs) {
                const buffer = reader.read(size);
                this.blobs.set(key, buffer);
            }
        }
        this.values = new Map();
        if (net.format_version) {
            const major = Math.floor(net.format_version / 100);
            const minor = net.format_version % 100;
            this.format += ` v${major}.${minor}`;
        }
        if (net && Array.isArray(net.layers)) {
            for (const layer of net.layers) {
                const type = layer.type;
                const data = { ...layer };
                const top = layer.top.split(',').map((name) => this._value(name));
                const bottom = layer.bottom.split(',').map((name) => this._value(name));
                const obj = {};
                obj.type = type;
                obj.name = layer.name;
                obj.attributes = data;
                obj.inputs = [{ name: 'inputs', value: bottom }];
                obj.outputs = [{ name: 'outputs', value: top }];
                obj.chain = [];
                switch (type) {
                    case 'convolution':
                    case 'deconvolution': {
                        this._weights(obj, data, [data.C, data.K, data.Nx, data.Ny]);
                        if (data.has_biases) {
                            obj.inputs.push(this._initializer('biases', data.blob_biases, 'float32', [data.C]));
                        }
                        delete data.has_biases;
                        delete data.blob_biases;
                        break;
                    }
                    case 'batchnorm': {
                        obj.inputs.push(this._initializer('params', data.blob_batchnorm_params, 'float32', [4, data.C]));
                        delete data.blob_batchnorm_params;
                        break;
                    }
                    case 'inner_product': {
                        this._weights(obj, data, [data.nC, data.nB]);
                        if (data.has_biases) {
                            obj.inputs.push(this._initializer('biases', data.blob_biases, 'float32', [data.nC]));
                        }
                        delete data.has_biases;
                        delete data.blob_biases;
                        break;
                    }
                    case 'instancenorm_1d':
                    case 'dynamic_dequantize': {
                        this._weights(obj, data, null);
                        break;
                    }
                    default: {
                        break;
                    }
                }
                const blobs = Object.keys(data).filter((key) => key.startsWith('blob_'));
                if (blobs.length > 0) {
                    throw new espresso.Error(`Unknown blob '${blobs.join(',')}' for type '${type}'.`);
                }
                if (data.has_prelu) {
                    obj.chain.push({ type: 'prelu' });
                }
                if (data.fused_relu || data.has_relu) {
                    obj.chain.push({ type: 'relu' });
                }
                if (data.fused_tanh || data.has_tanh) {
                    obj.chain.push({ type: 'tanh' });
                }
                if (data.has_batch_norm) {
                    obj.chain.push({ type: 'batch_norm' });
                }
                if (data.weights) {
                    for (const [name, identifier] of Object.entries(data.weights)) {
                        obj.inputs.push(this._initializer(name, identifier, 'float32', null));
                    }
                    delete data.weights;
                }
                delete data.name;
                delete data.type;
                delete data.top;
                delete data.bottom;
                delete data.fused_tanh;
                delete data.fused_relu;
                delete data.has_prelu;
                delete data.has_relu;
                delete data.has_tanh;
                delete data.has_batch_norm;
                this.nodes.push(obj);
            }
        }
        delete this.shapes;
        delete this.blobs;
    }

    _value(name) {
        if (!this.values.has(name)) {
            const shape = this.shapes.get(name);
            const type = shape ? new espresso.TensorType('float32', shape) : null;
            this.values.set(name, { name, type });
        }
        return this.values.get(name);
    }

    _weights(obj, data, dimensions) {
        if (data.blob_weights !== undefined) {
            obj.inputs.push(this._initializer('weights', data.blob_weights, 'float32', dimensions));
            delete data.blob_weights;
            return;
        }
        if (data.blob_weights_f16 !== undefined) {
            obj.inputs.push(this._initializer('weights', data.blob_weights_f16, 'float16', dimensions));
            delete data.blob_weights_f16;
            return;
        }
        const keys = ['wBeta', 'wGamma', 'W_S8', 'W_int8', 'W_t_int8'];
        for (const key of keys) {
            if (data.weights && data.weights[key] !== undefined) {
                let dataType = 'float32';
                let name = key;
                if (key.endsWith('_S8')) {
                    dataType = 'int8';
                    name = key.replace(/_S8$/, '');
                } else if (key.endsWith('_int8')) {
                    dataType = 'int8';
                    name = key.replace(/_int8$/, '');
                }
                obj.inputs.push(this._initializer(name, data.weights[key], dataType, dimensions));
                delete data.weights[key];
            }
        }
    }

    _initializer(name, identifier, dataType, dimensions) {
        if (!Number.isInteger(identifier)) {
            throw new espresso.Error(`Invalid '${identifier}' blob identifier.`);
        }
        dataType = dataType || 'float32';
        const blob = this.blobs.get(identifier);
        if (!dimensions) {
            const itemsize = dataType === 'float32' ? 4 : 1;
            dimensions = blob ? [blob.length / itemsize] : ['?'];
        }
        const shape = new espresso.TensorShape(dimensions);
        const type = new espresso.TensorType(dataType, shape);
        const value = {};
        const initializer = new espresso.Tensor(type, blob, null, 'Blob');
        value.value = new espresso.Value(`${identifier}\nblob`, type, null, initializer);
        return { name, value: [value] };
    }
};

espresso.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Espresso model.';
    }
};

export const ModelFactory = espresso.ModelFactory;
