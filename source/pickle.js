
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
        } else if (Array.isArray(obj)) {
            if (obj.length > 0 && obj[0] && obj.every((item) => item && item.__class__ && obj[0].__class__ && item.__class__.__module__ === obj[0].__class__.__module__ && item.__class__.__name__ === obj[0].__class__.__name__)) {
                const type = `${obj[0].__class__.__module__}.${obj[0].__class__.__name__}`;
                context.error(new pickle.Error(`Unsupported Pickle '${type}' array object.`));
            } else if (obj.length > 0) {
                context.error(new pickle.Error("Unsupported Pickle array object."));
            }
        } else if (obj && obj.__class__) {
            const formats = new Map([
                ['cuml.ensemble.randomforestclassifier.RandomForestClassifier', 'cuML']
            ]);
            const type = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
            if (formats.has(type)) {
                format = formats.get(type);
            } else {
                context.error(new pickle.Error(`Unsupported Pickle type '${type}'.`));
            }
        } else {
            context.error(new pickle.Error('Unsupported Pickle object.'));
        }
        return new pickle.Model(obj, format);
    }
};

pickle.Model = class {

    constructor(value, format) {
        this.format = format;
        this.graphs = [new pickle.Graph(value)];
    }
};

pickle.Graph = class {

    constructor(obj) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        if (Array.isArray(obj) && (obj.every((item) => item.__class__) || (obj.every((item) => Array.isArray(item))))) {
            for (const item of obj) {
                this.nodes.push(new pickle.Node(item));
            }
        } else if (obj && obj instanceof Map && !Array.from(obj.values()).some((value) => typeof value === 'string' || typeof value === 'number')) {
            for (const [name, value] of obj) {
                const node = new pickle.Node(value, name);
                this.nodes.push(node);
            }
        } else if (obj && obj.__class__) {
            this.nodes.push(new pickle.Node(obj));
        } else if (obj && Object(obj) === obj) {
            this.nodes.push(new pickle.Node(obj));
        }
    }
};

pickle.Node = class {

    constructor(obj, name, stack) {
        const type = obj.__class__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : 'builtins.object';
        this.type = { name: type };
        this.name = name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const isArray = (obj) => {
            return obj && obj.__class__ &&
                ((obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray') ||
                 (obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'matrix'));
        };
        const isObject = (obj) => {
            if (obj && typeof obj === 'object') {
                const proto = Object.getPrototypeOf(obj);
                return proto === Object.prototype || proto === null;
            }
            return false;
        };
        const entries = obj instanceof Map ? Array.from(obj) : Object.entries(obj);
        for (const [name, value] of entries) {
            if (name === '__class__') {
                continue;
            } else if (value && isArray(value)) {
                const tensor = new pickle.Tensor(value);
                const attribute = new pickle.Argument(name, tensor, 'tensor');
                this.attributes.push(attribute);
            } else if (Array.isArray(value) && value.length > 0 && value.every((obj) => isArray(obj))) {
                const tensors = value.map((obj) => new pickle.Tensor(obj));
                const attribute = new pickle.Argument(name, tensors, 'tensor[]');
                this.attributes.push(attribute);
            } else if (value && value.__class__ && value.__class__.__module__ === 'builtins' && (value.__class__.__name__ === 'function' || value.__class__.__name__ === 'type')) {
                const obj = {};
                obj.__class__ = value;
                const node = new pickle.Node(obj, '', stack);
                const attribute = new pickle.Argument(name, node, 'object');
                this.attributes.push(attribute);
            } else {
                stack = stack || new Set();
                if (value && Array.isArray(value) && value.every((obj) => typeof obj === 'string')) {
                    const attribute = new pickle.Argument(name, value, 'string[]');
                    this.attributes.push(attribute);
                } else if (value && Array.isArray(value) && value.every((obj) => typeof obj === 'number')) {
                    const attribute = new pickle.Argument(name, value);
                    this.attributes.push(attribute);
                } else if (value && Array.isArray(value) && value.length > 0 && value.every((obj) => obj && (obj.__class__ || obj === Object(obj)))) {
                    const values = value.filter((value) => !stack.has(value));
                    const nodes = values.map((value) => {
                        stack.add(value);
                        const node = new pickle.Node(value, '', stack);
                        stack.delete(value);
                        return node;
                    });
                    const attribute = new pickle.Argument(name, nodes, 'object[]');
                    this.attributes.push(attribute);
                } else if (value && (value.__class__ || isObject(value))) {
                    if (!stack.has(value)) {
                        stack.add(value);
                        const node = new pickle.Node(value, '', stack);
                        const attribute = new pickle.Argument(name, node, 'object');
                        this.attributes.push(attribute);
                        stack.delete(value);
                    }
                } else {
                    const attribute = new pickle.Argument(name, value);
                    this.attributes.push(attribute);
                }
            }
        }
    }
};

pickle.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name.toString();
        this.value = value;
        if (type) {
            this.type = type;
        }
        if (visible === false) {
            this.visible = visible;
        }
    }
};

pickle.Tensor = class {

    constructor(array) {
        this.type = new pickle.TensorType(array.dtype.__name__, new pickle.TensorShape(array.shape));
        this.stride = array.strides.map((stride) => stride / array.itemsize);
        this.encoding = this.type.dataType === 'string' || this.type.dataType === 'object' ? '|' : array.dtype.byteorder;
        this.values = this.type.dataType === 'string' || this.type.dataType === 'object' ? array.flatten().tolist() : array.tobytes();
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

pickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Pickle model.';
    }
};

export const ModelFactory = pickle.ModelFactory;
