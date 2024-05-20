
import * as json from './json.js';

const safetensors = {};

safetensors.ModelFactory = class {

    match(context) {
        const container = safetensors.Container.open(context);
        if (container) {
            context.type = 'safetensors';
            context.target = container;
        } else {
            const obj = context.peek('json');
            if (obj && obj.weight_map) {
                const entries = Object.entries(obj.weight_map);
                if (entries.length > 0 && entries.every(([, value]) => typeof value === 'string' && value.endsWith('.safetensors'))) {
                    context.type = 'safetensors.json';
                    context.target = entries;
                }
            }
        }
    }

    async open(context) {
        switch (context.type) {
            case 'safetensors': {
                const container = context.target;
                await container.read();
                return new safetensors.Model(container.entries);
            }
            case 'safetensors.json': {
                const weight_map = new Map(context.target);
                const keys = new Set(weight_map.keys());
                const files = Array.from(new Set(weight_map.values()));
                const contexts = await Promise.all(files.map((name) => context.fetch(name)));
                const containers = contexts.map((context) => safetensors.Container.open(context));
                await Promise.all(containers.map((container) => container.read()));
                const entries = new Map();
                for (const container of containers) {
                    for (const [key, value] of Array.from(container.entries)) {
                        if (keys.has(key)) {
                            entries.set(key, value);
                        }
                    }
                }
                return new safetensors.Model(entries);
            }
            default: {
                throw new safetensors.Error(`Unsupported Safetensors format '${context.type}'.`);
            }
        }
    }
};

safetensors.Model = class {

    constructor(entries) {
        this.format = 'Safetensors';
        this.graphs = [new safetensors.Graph(entries)];
    }
};

safetensors.Graph = class {

    constructor(entries) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const layers = new Map();
        for (const [key, value] of Array.from(entries)) {
            if (key === '__metadata__') {
                continue;
            }
            const parts = key.split('.');
            const name = parts.pop();
            const layer = parts.join('.');
            if (!layers.has(layer)) {
                layers.set(layer, []);
            }
            layers.get(layer).push([name, key, value]);
        }
        for (const [name, values] of layers) {
            const node = new safetensors.Node(name, values);
            this.nodes.push(node);
        }
    }
};

safetensors.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

safetensors.Value = class {

    constructor(name, value) {
        this.name = name;
        this.type = value.type;
        this.initializer = value;
    }
};

safetensors.Node = class {

    constructor(name, values) {
        this.name = name;
        this.type = { name: 'Module' };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        for (const [name, identifier, obj] of values) {
            const tensor = new safetensors.Tensor(obj);
            const value = new safetensors.Value(identifier, tensor);
            const argument = new safetensors.Argument(name, [value]);
            this.inputs.push(argument);
        }
    }
};

safetensors.TensorType = class {

    constructor(dtype, shape) {
        switch (dtype) {
            case 'I8':      this.dataType = 'int8'; break;
            case 'I16':     this.dataType = 'int16'; break;
            case 'I32':     this.dataType = 'int32'; break;
            case 'I64':     this.dataType = 'int64'; break;
            case 'U8':      this.dataType = 'uint8'; break;
            case 'U16':     this.dataType = 'uint16'; break;
            case 'U32':     this.dataType = 'uint32'; break;
            case 'U64':     this.dataType = 'uint64'; break;
            case 'BF16':    this.dataType = 'bfloat16'; break;
            case 'F16':     this.dataType = 'float16'; break;
            case 'F32':     this.dataType = 'float32'; break;
            case 'F64':     this.dataType = 'float64'; break;
            case 'BOOL':    this.dataType = 'boolean'; break;
            case 'F8_E4M3': this.dataType = 'float8e4m3fn'; break;
            case 'F8_E5M2': this.dataType = 'float8e5m2'; break;
            default: throw new safetensors.Error(`Unsupported data type '${dtype}'.`);
        }
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

safetensors.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

safetensors.Tensor = class {

    constructor(obj) {
        const shape = new safetensors.TensorShape(obj.shape);
        this.type = new safetensors.TensorType(obj.dtype, shape);
        this.encoding = '<';
        this.data = obj.__data__;
    }

    get values() {
        if (this.data instanceof Uint8Array) {
            return this.data;
        }
        if (this.data && this.data.peek) {
            return this.data.peek();
        }
        return null;
    }
};

safetensors.Container = class {

    static open(context) {
        const identifier = context.identifier;
        const stream = context.stream;
        if (stream.length > 9) {
            const buffer = stream.peek(9);
            if (buffer[6] === 0 && buffer[7] === 0 && buffer[8] === 0x7b) {
                const size = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24 | buffer [3] << 32 | buffer [3] << 40;
                if (size < stream.length) {
                    return new safetensors.Container(identifier, stream, size);
                }
            }
        }
        return null;
    }

    constructor(identifier, stream, size) {
        this.identifier = identifier;
        this.size = size;
        this.stream = stream;
        this.entries = new Map();
    }

    async read() {
        const stream = this.stream;
        const position = stream.position;
        stream.seek(8);
        const buffer = stream.read(this.size);
        const reader = json.TextReader.open(buffer);
        const obj = reader.read();
        const offset = stream.position;
        for (const [key, value] of Object.entries(obj)) {
            if (key === '__metadata__') {
                continue;
            }
            const [start, end] = value.data_offsets;
            stream.seek(offset + start);
            value.__data__ = stream.stream(end - start);
            this.entries.set(key, value);
        }
        stream.seek(position);
        delete this.size;
        delete this.stream;
    }
};

safetensors.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Safetensors model.';
    }
};

export const ModelFactory = safetensors.ModelFactory;
