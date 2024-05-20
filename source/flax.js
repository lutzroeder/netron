
// Experimental

import * as python from './python.js';

const flax = {};

flax.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream.length > 4) {
            const buffer = stream.peek(1);
            if (buffer[0] === 0xDE || buffer[0] === 0xDF || ((buffer[0] & 0x80) === 0x80)) {
                context.type = 'flax.msgpack.map';
            }
        }
    }

    async open(context) {
        const stream = context.stream;
        const packed = stream.peek();
        const execution = new python.Execution();
        // https://github.com/google/flax/blob/main/flax/serialization.py
        const ext_hook = (code, data) => {
            switch (code) {
                case 1: { // _MsgpackExtType.ndarray
                    const tuple = execution.invoke('msgpack.unpackb', [data]);
                    const dtype = execution.invoke('numpy.dtype', [tuple[1]]);
                    dtype.byteorder = '<';
                    return execution.invoke('numpy.ndarray', [tuple[0], dtype, tuple[2]]);
                }
                default: {
                    throw new flax.Error(`Unsupported MessagePack extension '${code}'.`);
                }
            }
        };
        const obj = execution.invoke('msgpack.unpackb', [packed, ext_hook]);
        return new flax.Model(obj);
    }
};

flax.Model = class {

    constructor(obj) {
        this.format = 'Flax';
        this.graphs = [new flax.Graph(obj)];
    }
};

flax.Graph = class {

    constructor(obj) {
        this.inputs = [];
        this.outputs = [];
        const layers = new Map();
        const layer = (path) => {
            const name = path.join('.');
            if (!layers.has(name)) {
                layers.set(name, {});
            }
            return layers.get(name);
        };
        const flatten = (path, obj) => {
            for (const [name, value] of Object.entries(obj)) {
                if (flax.Utility.isTensor(value)) {
                    const obj = layer(path);
                    obj[name] = value;
                } else if (Array.isArray(value)) {
                    const obj = layer(path);
                    obj[name] = value;
                } else if (Object(value) === value) {
                    flatten(path.concat(name), value);
                } else {
                    const obj = layer(path);
                    obj[name] = value;
                }
            }
        };
        if (Array.isArray(obj)) {
            layer([]).value = obj;
        } else {
            flatten([], obj);
        }
        this.nodes = Array.from(layers).map(([name, value]) => new flax.Node(name, value));
    }
};

flax.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

flax.Value = class {

    constructor(name, initializer) {
        if (typeof name !== 'string') {
            throw new flax.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : null;
        this.initializer = initializer || null;
    }
};

flax.Node = class {

    constructor(name, layer) {
        this.name = name;
        this.type = { name: 'Module' };
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        for (const [name, value] of Object.entries(layer)) {
            if (flax.Utility.isTensor(value)) {
                const tensor = new flax.Tensor(value);
                const argument = new flax.Argument(name, [new flax.Value('', tensor)]);
                this.inputs.push(argument);
            } else if (Array.isArray(value)) {
                const attribute = new flax.Argument(name, value);
                this.attributes.push(attribute);
            } else {
                const attribute = new flax.Argument(name, value);
                this.attributes.push(attribute);
            }
        }
    }
};

flax.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

flax.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return (Array.isArray(this.dimensions) && this.dimensions.length > 0) ?
            `[${this.dimensions.join(',')}]` : '';
    }
};

flax.Tensor = class {

    constructor(array) {
        this.type = new flax.TensorType(array.dtype.__name__, new flax.TensorShape(array.shape));
        const dataType = this.type.dataType;
        this.encoding = dataType === 'string' || dataType === 'object' ? '|' : array.dtype.byteorder;
        this._data = array.tobytes();
        this._itemsize = array.dtype.itemsize;
    }

    get values() {
        switch (this.type.dataType) {
            case 'string': {
                if (this._data instanceof Uint8Array) {
                    const data = this._data;
                    const decoder = new TextDecoder('utf-8');
                    const size = this.type.shape.dimensions.reduce((a, b) => a * b, 1);
                    this._data = new Array(size);
                    let offset = 0;
                    for (let i = 0; i < size; i++) {
                        const buffer = data.subarray(offset, offset + this._itemsize);
                        const index = buffer.indexOf(0);
                        this._data[i] = decoder.decode(index >= 0 ? buffer.subarray(0, index) : buffer);
                        offset += this._itemsize;
                    }
                }
                return this._data;
            }
            default:
                return this._data;
        }
    }
};

flax.Utility = class {

    static isTensor(obj) {
        return obj && obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray';
    }
};

flax.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Flax model.';
    }
};

export const ModelFactory = flax.ModelFactory;
