
var safetensors = {};
var json = require('./json');

safetensors.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream.length > 9) {
            const buffer = stream.peek(9);
            if (buffer[6] === 0 && buffer[7] === 0 && buffer[8] === 0x7b) {
                const size = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24 | buffer [3] << 32 | buffer [3] << 40;
                if (size < stream.length) {
                    return { size: size };
                }
            }
        }
        return '';
    }

    async open(context, target) {
        const stream = context.stream;
        stream.seek(8);
        const buffer = stream.read(target.size);
        const reader = json.TextReader.open(buffer);
        const obj = reader.read();
        const model = new safetensors.Model(obj, stream.position, stream);
        stream.seek(0);
        return model;
    }
};

safetensors.Model = class {

    constructor(obj, position, stream) {
        this.format = 'Safetensors';
        this.graphs = [ new safetensors.Graph(obj, position, stream) ];
    }
};

safetensors.Graph = class {

    constructor(obj, position, stream) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const layers = new Map();
        for (const entry of Object.entries(obj)) {
            if (entry[0] === '__metadata__') {
                continue;
            }
            const parts = entry[0].split('.');
            const name = parts.pop();
            const layer = parts.join('.');
            if (!layers.has(layer)) {
                layers.set(layer, []);
            }
            layers.get(layer).push([ name, entry[0], entry[1]]);
        }
        for (const entry of layers) {
            this.nodes.push(new safetensors.Node(entry[0], entry[1], position, stream));
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
        this.initializer = value;
    }

    get type() {
        return this.initializer.type;
    }
};

safetensors.Node = class {

    constructor(name, values, position, stream) {
        this.name = name;
        this.type = { name: 'Module' };
        this.inputs = values.map((value) => new safetensors.Argument(value[0], [
            new safetensors.Value(value[1], new safetensors.Tensor(value[2], position, stream))
        ]));
        this.outputs = [];
        this.attributes = [];
    }
};

safetensors.TensorType = class {

    constructor(dtype, shape) {
        switch (dtype) {
            case 'I8':   this.dataType = 'int8'; break;
            case 'I16':  this.dataType = 'int16'; break;
            case 'I32':  this.dataType = 'int32'; break;
            case 'I64':  this.dataType = 'int64'; break;
            case 'U8':   this.dataType = 'uint8'; break;
            case 'U16':  this.dataType = 'uint16'; break;
            case 'U32':  this.dataType = 'uint32'; break;
            case 'U64':  this.dataType = 'uint64'; break;
            case 'BF16': this.dataType = 'bfloat16'; break;
            case 'F16':  this.dataType = 'float16'; break;
            case 'F32':  this.dataType = 'float32'; break;
            case 'F64':  this.dataType = 'float64'; break;
            default: throw new safetensors.Error("Unsupported data type '" + dtype + "'.");
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
        return '[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

safetensors.Tensor = class {

    constructor(obj, position, stream) {
        const shape = new safetensors.TensorShape(obj.shape);
        this.type = new safetensors.TensorType(obj.dtype, shape);
        this.layout = '<';
        const size = obj.data_offsets[1] - obj.data_offsets[0];
        position += obj.data_offsets[0];
        stream.seek(position);
        this.values = stream.read(size);
    }
};


safetensors.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Safetensors model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = safetensors.ModelFactory;
}
