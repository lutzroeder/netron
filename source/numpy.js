
// Experimental

import * as python from './python.js';

const numpy = {};

numpy.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            context.type = 'npy';
        } else {
            const entries = context.peek('npz');
            if (entries && entries.size > 0) {
                context.type = 'npz';
                context.target = entries;
            }
        }
    }

    async open(context) {
        let format = '';
        const graphs = [];
        switch (context.type) {
            case 'npy': {
                format = 'NumPy Array';
                const unresolved = new Set();
                const execution = new python.Execution();
                execution.on('resolve', (_, name) => unresolved.add(name));
                const stream = context.stream;
                const buffer = stream.peek();
                const bytes = execution.invoke('io.BytesIO', [buffer]);
                const array = execution.invoke('numpy.load', [bytes]);
                if (unresolved.size > 0) {
                    const name = unresolved.values().next().value;
                    throw new numpy.Error(`Unknown type name '${name}'.`);
                }
                const layer = { type: 'numpy.ndarray', parameters: [{ name: 'value', tensor: { name: '', array } }] };
                graphs.push({ layers: [layer] });
                break;
            }
            case 'npz': {
                format = 'NumPy Zip';
                const layers = new Map();
                for (const [key, array] of context.target) {
                    const name = key.replace(/\.npy$/, '');
                    const parts = name.split('/');
                    const parameterName = parts.pop();
                    const groupName = parts.join('/');
                    if (!layers.has(groupName)) {
                        layers.set(groupName, { name: groupName, parameters: [] });
                    }
                    const layer = layers.get(groupName);
                    layer.parameters.push({
                        name: parameterName,
                        tensor: { name, array }
                    });
                }
                graphs.push({ layers: Array.from(layers.values()) });
                break;
            }
            default: {
                throw new numpy.Error(`Unsupported NumPy format '${context.type}'.`);
            }
        }
        return new numpy.Model(format, graphs);
    }
};

numpy.Model = class {

    constructor(format, graphs) {
        this.format = format;
        this.graphs = graphs.map((graph) => new numpy.Graph(graph));
    }
};

numpy.Graph = class {

    constructor(graph) {
        this.name = graph.name || '';
        this.nodes = graph.layers.map((layer) => new numpy.Node(layer));
        this.inputs = [];
        this.outputs = [];
    }
};

numpy.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

numpy.Value = class {

    constructor(name, initializer) {
        if (typeof name !== 'string') {
            throw new numpy.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer.type;
        this.initializer = initializer || null;
    }
};

numpy.Node = class {

    constructor(layer) {
        this.name = layer.name || '';
        this.type = { name: layer.type || 'Object' };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        for (const parameter of layer.parameters) {
            const initializer = new numpy.Tensor(parameter.tensor.array);
            const value = new numpy.Value(parameter.tensor.name || '', initializer);
            const argument = new numpy.Argument(parameter.name, [value]);
            this.inputs.push(argument);
        }
    }
};

numpy.Tensor = class  {

    constructor(array) {
        this.type = new numpy.TensorType(array.dtype.__name__, new numpy.TensorShape(array.shape));
        this.stride = array.strides.map((stride) => stride / array.itemsize);
        this.values = this.type.dataType === 'string' || this.type.dataType === 'object' || this.type.dataType === 'void' ? array.flatten().tolist() : array.tobytes();
        this.encoding = this.type.dataType === 'string' || this.type.dataType === 'object' ? '|' : array.dtype.byteorder;
    }
};

numpy.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

numpy.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions && this.dimensions.length > 0 ? `[${this.dimensions.join(',')}]` : '';
    }
};

numpy.Utility = class {

    static isTensor(obj) {
        return obj && obj.__class__ &&
            ((obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray') ||
             (obj.__class__.__module__ === 'numpy.core.memmap' && obj.__class__.__name__ === 'memmap'));
    }

    static weights(obj) {
        const dict = (obj, key) => {
            const dict = key === '' ? obj : obj[key];
            if (dict) {
                const weights = new Map();
                if (dict instanceof Map) {
                    for (const [key, obj] of dict) {
                        if (numpy.Utility.isTensor(obj)) {
                            weights.set(key, obj);
                            continue;
                        } else if (obj instanceof Map && Array.from(obj).every(([, value]) => numpy.Utility.isTensor(value))) {
                            for (const [name, value] of obj) {
                                weights.set(`${key}.${name}`, value);
                            }
                            continue;
                        } else if (key === '_metadata') {
                            continue;
                        }
                        return null;
                    }
                    return weights;
                } else if (!Array.isArray(dict)) {
                    const set = new Set(['weight_order', 'lr', 'model_iter', '__class__']);
                    for (const [name, value] of Object.entries(dict)) {
                        if (numpy.Utility.isTensor(value)) {
                            weights.set(name, value);
                            continue;
                        }
                        if (set.has(name)) {
                            continue;
                        }
                        if (value && !Array.isArray(value) && Object.entries(value).every(([, value]) => numpy.Utility.isTensor(value))) {
                            if (value && value.__class__ && value.__class__.__module__ && value.__class__.__name__) {
                                weights.set(`${name}.__class__`, `${value.__class__.__module__}.${value.__class__.__name__}`);
                            }
                            for (const [name, obj] of Object.entries(value)) {
                                weights.set(`${name}.${name}`, obj);
                            }
                            continue;
                        }
                        return null;
                    }
                    return weights;
                }
            }
            return null;
        };
        const list = (obj, key) => {
            let list = key === '' ? obj : obj[key];
            if (list && Array.isArray(list) && list.every((obj) => Object.values(obj).every((value) => numpy.Utility.isTensor(value)))) {
                list = list.map((obj) => obj instanceof Map ? obj : new Map(Object.entries(obj)));
            }
            if (list && Array.isArray(list)) {
                const weights = new Map();
                for (let i = 0; i < list.length; i++) {
                    const obj = list[i];
                    if (numpy.Utility.isTensor(obj)) {
                        weights.set(i.toString(), obj);
                        continue;
                    } else if (obj instanceof Map && Array.from(obj).every(([, value]) => numpy.Utility.isTensor(value))) {
                        for (const [name, value] of obj) {
                            weights.set(`${i}.${name}`, value);
                        }
                        continue;
                    }
                    return null;
                }
                return weights;
            }
            return null;
        };
        const keys = ['', 'blobs', 'model', 'experiment_state'];
        for (const key of keys) {
            const weights = dict(obj, key);
            if (weights && weights.size > 0) {
                return weights;
            }
        }
        for (const key of keys) {
            const weights = list(obj, key);
            if (weights) {
                return weights;
            }
        }
        return null;
    }
};

numpy.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Chainer model.';
    }
};

export const ModelFactory = numpy.ModelFactory;
