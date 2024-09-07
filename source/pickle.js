
// Experimental

const pickle = {};

pickle.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            // Reject PyTorch models with .pkl file extension.
            return;
        }
        const obj = context.peek('pkl');
        if (obj !== undefined) {
            const name = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : '';
            if (!name.startsWith('__torch__.')) {
                context.type = 'pickle';
                context.target = obj;
            }
        }
    }

    async open(context) {
        let format = 'Pickle';
        const obj = context.target;
        if (obj === null || obj === undefined) {
            context.error(new pickle.Error("Unsupported Pickle null object."));
        } else if (obj instanceof Error) {
            throw obj;
        } else if (!Array.isArray(obj) && obj && obj.__class__) {
            const formats = new Map([
                ['cuml.ensemble.randomforestclassifier.RandomForestClassifier', 'cuML'],
                ['shap.explainers._linear.LinearExplainer', 'SHAP'],
                ['gensim.models.word2vec.Word2Vec', 'Gensim'],
                ['builtins.bytearray', 'Pickle'],
                ['builtins.dict', 'Pickle'],
                ['collections.OrderedDict', 'Pickle'],
                ['numpy.ndarray', 'NumPy NDArray'],
            ]);
            const type = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
            if (formats.has(type)) {
                format = formats.get(type);
            } else {
                context.error(new pickle.Error(`Unsupported Pickle type '${type}'.`));
            }
        }
        return new pickle.Model(obj, format);
    }
};

pickle.Model = class {

    constructor(value, format) {
        this.format = format;
        this.graphs = [new pickle.Graph(null, value)];
    }
};

pickle.Graph = class {

    constructor(type, obj) {
        this.type = type || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const weights = this.type === 'weights' ? obj : pickle.Utility.weights(obj);
        if (weights) {
            for (const [name, module] of weights) {
                const node = new pickle.Node(module, name, 'Weights');
                this.nodes.push(node);
            }
        } else if (pickle.Utility.isTensor(obj)) {
            const type = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
            const node = new pickle.Node({ value: obj }, null, type);
            this.nodes.push(node);
        } else if (Array.isArray(obj) && (obj.every((item) => item.__class__) || (obj.every((item) => Array.isArray(item))))) {
            for (const item of obj) {
                this.nodes.push(new pickle.Node(item));
            }
        } else if (obj && obj.__class__) {
            this.nodes.push(new pickle.Node(obj));
        } else if (obj && Object(obj) === obj) {
            this.nodes.push(new pickle.Node(obj));
        }
    }
};

pickle.Node = class {

    constructor(obj, name, type, stack) {
        if (typeof type !== 'string') {
            type = obj.__class__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : 'builtins.object';
        }
        this.type = { name: type };
        this.name = name || '';
        this.inputs = [];
        if (type === 'builtins.bytearray') {
            const argument = new pickle.Argument('value', Array.from(obj), 'byte[]');
            this.inputs.push(argument);
            return;
        }
        const weights = pickle.Utility.weights(obj);
        if (weights) {
            const type = this.type.name;
            this.type = new pickle.Graph('weights', weights);
            this.type.name = type;
            return;
        }
        const entries = obj instanceof Map ? Array.from(obj) : Object.entries(obj);
        for (const [name, value] of entries) {
            if (name === '__class__') {
                continue;
            } else if (value && pickle.Utility.isTensor(value)) {
                const identifier = value.__name__ || '';
                const tensor = new pickle.Tensor(value);
                const values = [new pickle.Value(identifier, null, tensor)];
                const argument = new pickle.Argument(name, values);
                this.inputs.push(argument);
            } else if (Array.isArray(value) && value.length > 0 && value.every((obj) => pickle.Utility.isTensor(obj))) {
                const values = value.map((obj) => new pickle.Value(obj.__name__ || '', null, new pickle.Tensor(obj)));
                const argument = new pickle.Argument(name, values);
                this.inputs.push(argument);
            } else if (value && value.__class__ && value.__class__.__module__ === 'builtins' && (value.__class__.__name__ === 'function' || value.__class__.__name__ === 'type')) {
                const obj = {};
                obj.__class__ = value;
                const node = new pickle.Node(obj, null, null, stack);
                const argument = new pickle.Argument(name, node, 'object');
                this.inputs.push(argument);
            } else if (pickle.Utility.isByteArray(value)) {
                const argument = new pickle.Argument(name, Array.from(value), 'byte[]');
                this.inputs.push(argument);
            } else {
                stack = stack || new Set();
                if (value && Array.isArray(value) && value.every((obj) => typeof obj === 'string')) {
                    const argument = new pickle.Argument(name, value, 'string[]');
                    this.inputs.push(argument);
                } else if (value && Array.isArray(value) && value.every((obj) => typeof obj === 'number')) {
                    const argument = new pickle.Argument(name, value, 'attribute');
                    this.inputs.push(argument);
                } else if (value && Array.isArray(value) && value.length > 0 && value.every((obj) => obj && (obj.__class__ || obj === Object(obj)))) {
                    const chain = stack;
                    const values = value.filter((value) => !chain.has(value));
                    const nodes = values.map((value) => {
                        chain.add(value);
                        const node = new pickle.Node(value, null, null, chain);
                        chain.delete(value);
                        return node;
                    });
                    const argument = new pickle.Argument(name, nodes, 'object[]');
                    this.inputs.push(argument);
                } else if (value && (value.__class__ || pickle.Utility.isObject(value)) && !stack.has(value)) {
                    stack.add(value);
                    const node = new pickle.Node(value, null, null, stack);
                    const visible = name !== '_metadata' || !pickle.Utility.isMetadataObject(value);
                    const argument = new pickle.Argument(name, node, 'object', visible);
                    this.inputs.push(argument);
                    stack.delete(value);
                } else {
                    const argument = new pickle.Argument(name, value, 'attribute');
                    this.inputs.push(argument);
                }
            }
        }
    }
};

pickle.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name.toString();
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

pickle.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new pickle.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer && initializer.type ? initializer.type : type || null;
        this.initializer = initializer || null;
    }
};

pickle.Tensor = class {

    constructor(obj) {
        if (obj.__class__ && (obj.__class__.__module__ === 'torch' || obj.__class__.__module__ === 'torch.nn.parameter')) {
            // PyTorch tensor
            const tensor = obj.__class__.__module__ === 'torch.nn.parameter' && obj.__class__.__name__ === 'Parameter' ? obj.data : obj;
            const layout = tensor.layout ? tensor.layout.__str__() : null;
            const storage = tensor.storage();
            const size = tensor.size() || [];
            if (!layout || layout === 'torch.strided') {
                this.type = new pickle.TensorType(storage.dtype.__reduce__(), new pickle.TensorShape(size));
                this.values = storage.data;
                this.encoding = '<';
                this.indices = null;
                this.stride = tensor.stride();
                const stride = this.stride;
                const offset = tensor.storage_offset();
                let length = 0;
                if (!Array.isArray(stride)) {
                    length = storage.size();
                } else if (size.every((v) => v !== 0)) {
                    length = size.reduce((a, v, i) => a + stride[i] * (v - 1), 1);
                }
                if (offset !== 0 || length !== storage.size()) {
                    const itemsize = storage.dtype.itemsize();
                    const stream = this.values;
                    const position = stream.position;
                    stream.seek(itemsize * offset);
                    this.values = stream.peek(itemsize * length);
                    stream.seek(position);
                } else if (this.values) {
                    this.values = this.values.peek();
                }
            } else {
                throw new pickle.Error(`Unsupported tensor layout '${layout}'.`);
            }
        } else {
            // NumPy array
            const array = obj;
            this.type = new pickle.TensorType(array.dtype.__name__, new pickle.TensorShape(array.shape));
            this.stride = Array.isArray(array.strides) ? array.strides.map((stride) => stride / array.itemsize) : null;
            this.encoding = this.type.dataType === 'string' || this.type.dataType === 'object' ? '|' : array.dtype.byteorder;
            this.values = this.type.dataType === 'string' || this.type.dataType === 'object' || this.type.dataType === 'void' ? array.flatten().tolist() : array.tobytes();
        }
    }
};

pickle.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

pickle.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`) : '';
    }
};

pickle.Utility = class {

    static isSubclass(value, name) {
        if (value && value.__module__ && value.__name__) {
            return name === `${value.__module__}.${value.__name__}`;
        } else if (value && value.__bases__) {
            return value.__bases__.some((obj) => pickle.Utility.isSubclass(obj, name));
        }
        return false;
    }

    static isInstance(value, name) {
        return value && value.__class__ ? pickle.Utility.isSubclass(value.__class__, name) : false;
    }

    static isMetadataObject(obj) {
        if (pickle.Utility.isInstance(obj, 'collections.OrderedDict')) {
            for (const value of obj.values()) {
                if (pickle.Utility.isInstance(value, 'builtins.dict')) {
                    const entries = Array.from(value);
                    if (entries.length !== 1 && entries[0] !== 'version' && entries[1] !== 1) {
                        return false;
                    }
                }
            }
            return true;
        }
        return false;
    }

    static isByteArray(obj) {
        return obj && obj.__class__ && obj.__class__.__module__ === 'builtins' && obj.__class__.__name__ === 'bytearray';
    }

    static isObject(obj) {
        if (obj && typeof obj === 'object') {
            const proto = Object.getPrototypeOf(obj);
            return proto === Object.prototype || proto === null;
        }
        return false;
    }

    static isTensor(obj) {
        return obj && obj.__class__ && obj.__class__.__name__ &&
            ((obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray') ||
             (obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'matrix') ||
             (obj.__class__.__module__ === 'jax' && obj.__class__.__name__ === 'Array') ||
             (obj.__class__.__module__ === 'torch.nn.parameter' && obj.__class__.__name__ === 'Parameter') ||
             (obj.__class__.__module__ === 'torch' && obj.__class__.__name__.endsWith('Tensor')));
    }

    static weights(obj) {
        const type = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : null;
        if (type && type !== 'builtins.dict' && type !== 'builtins.object' && type !== 'collections.OrderedDict' && type !== 'torch.nn.modules.module.Module') {
            return null;
        }
        if (pickle.Utility.isTensor(obj)) {
            return null;
        }
        if (obj instanceof Map === false && obj && !Array.isArray(obj) && Object(obj) === obj) {
            const entries = Object.entries(obj);
            const named = entries.filter(([name, value]) => (typeof name === 'string' && (name.indexOf('.') !== -1 || name.indexOf('|') !== -1)) && pickle.Utility.isTensor(value));
            if (named.length > 0 && (named.length / entries.length) >= 0.8) {
                obj = new Map(entries);
            }
        }
        if (obj instanceof Map) {
            const entries = Array.from(obj).filter(([name]) => name !== '_metadata');
            let dot = 0;
            let pipe = 0;
            let underscore = 0;
            let count = 0;
            let valid = true;
            for (const [name, value] of entries) {
                if (typeof name === 'string') {
                    count++;
                    dot += name.indexOf('.') !== -1;
                    pipe += name.indexOf('|') !== -1;
                    underscore += name.endsWith('_w') || name.endsWith('_b') || name.endsWith('_bn_s');
                }
                if (pickle.Utility.isInstance(value, 'builtins.dict') && !Array.from(value.values()).every((value) => !pickle.Utility.isTensor(value))) {
                    valid = false;
                }
            }
            if (valid && count > 1 && (dot >= count || pipe >= count || underscore >= count) && (count / entries.length) >= 0.8) {
                let separator = null;
                if (dot >= pipe && dot >= underscore) {
                    separator = '.';
                } else if (pipe >= underscore) {
                    separator = '|';
                } else {
                    separator = '_';
                }
                const modules = new Map();
                for (const [name, value] of entries) {
                    let c = separator;
                    if (!c) {
                        c = name.indexOf('.') === -1 && name.indexOf('|') !== -1 ? '|' : '.';
                    }
                    const path = name.split(c);
                    let property = path.pop();
                    if (path.length > 1 && path[path.length - 1] === '_packed_params') {
                        property = `${path.pop()}.${property}`;
                    }
                    const key = path.join(separator);
                    if (!modules.has(key)) {
                        modules.set(key, {});
                    }
                    const module = modules.get(key);
                    if (pickle.Utility.isTensor(value)) {
                        value.__name__ = name;
                    }
                    module[property] = value;
                }
                return modules;
            }
        }
        if (obj && !Array.isArray(obj) && Object(obj) === obj) {
            const modules = new Map();
            const entries = obj instanceof Map ? Array.from(obj) : Object.entries(obj);
            if (entries.length > 0 && entries) {
                for (const [key, value] of entries) {
                    const name = key.toString();
                    if (!value || Object(value) !== value || pickle.Utility.isTensor(value) || ArrayBuffer.isView(value)) {
                        return null;
                    }
                    if (!modules.has(name)) {
                        modules.set(name, {});
                    }
                    const module = modules.get(name);
                    let tensor = false;
                    const entries = value instanceof Map ? value : new Map(Object.entries(value));
                    for (const [name, value] of entries) {
                        if (typeof name !== 'string') {
                            return null;
                        }
                        if (name.indexOf('.') !== -1) {
                            return null;
                        }
                        if (name === '_metadata') {
                            continue;
                        }
                        if (typeof value === 'string' || typeof value === 'number') {
                            module[name] = value;
                            continue;
                        }
                        if (pickle.Utility.isTensor(value)) {
                            value.__name__ = name;
                            module[name] = value;
                            tensor = true;
                        }
                    }
                    if (!tensor) {
                        return null;
                    }
                }
                return modules;
            }
        }
        return null;
    }
};

pickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Pickle model.';
    }
};

export const ModelFactory = pickle.ModelFactory;
