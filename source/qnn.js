
// Experimental

const qnn = {};

qnn.ModelFactory = class {

    match(context) {
        const obj = context.peek('json');
        if (obj && obj['model.cpp'] && obj.graph) {
            context.type = 'qnn.json';
            context.target = obj;
        }
        const entries = context.peek('tar');
        if (entries && entries.size > 0 && Array.from(entries).every(([name]) => name.endsWith('.raw'))) {
            context.type = 'qnn.weights';
            context.target = entries;
        }
    }

    async open(context) {
        const metadata = await context.metadata('qnn-metadata.json');
        switch (context.type) {
            case 'qnn.json': {
                const obj = context.target;
                let weights = new Map();
                try {
                    if (obj['model.bin']) {
                        const name = obj['model.bin'].split('/').pop();
                        const content = await context.fetch(name);
                        weights = content.read('tar');
                    }
                } catch {
                    // continue regardless of error
                }
                return new qnn.Model(metadata, obj, weights);
            }
            case 'qnn.weights': {
                const weights = context.target;
                const identifier = context.identifier;
                const parts = identifier.split('.');
                parts.pop();
                const base = parts.join('.');
                const content = await context.fetch(`${base}_net.json`);
                const obj = content.read('json');
                return new qnn.Model(metadata, obj, weights);
            }
            default: {
                throw new qnn.Error(`Unsupported QNN format '${context.type}'.`);
            }
        }
    }
};

qnn.Model = class {

    constructor(metadata, obj, weights) {
        this.format = 'QNN';
        this.metadata = [];
        this.graphs = [new qnn.Graph(metadata, obj.graph, weights)];
    }
};

qnn.Graph = class {

    constructor(metadata, obj, weights) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (!values.has(name)) {
                const value = new qnn.Value(name, type, tensor);
                values.set(name, value);
            }
            return values.get(name);
        };
        const dataType = (value) => {
            switch (value) {
                case 0x0008: return 'int8';
                case 0x0016: return 'int16';
                case 0x0032: return 'int32';
                case 0x0108: return 'int8';
                case 0x0132: return 'int32';
                case 0x0216: return 'float16';
                case 0x0232: return 'float32';
                case 0x0308: return 'qint8';
                case 0x0316: return 'qint16';
                case 0x0332: return 'qint32';
                case 0x0408: return 'uint8';
                case 0x0416: return 'uint16';
                case 0x0432: return 'uint32';
                case 0x0508: return 'boolean';
                default: throw new qnn.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
            }
        };
        const tensors = Object.entries(obj.tensors);
        for (const [name, obj] of tensors) {
            const shape = new qnn.TensorShape(obj.dims);
            const type = new qnn.TensorType(dataType(obj.data_type), shape);
            switch (obj.type) {
                case 0: {
                    const value = values.map(name, type);
                    const argument = new qnn.Argument(name, [value]);
                    this.inputs.push(argument);
                    break;
                }
                case 1: {
                    const value = values.map(name, type);
                    const argument = new qnn.Argument(name, [value]);
                    this.outputs.push(argument);
                    break;
                }
                case 3: {
                    values.map(name, type);
                    break;
                }
                case 4: {
                    const reader = weights.get(`${name}.raw`);
                    const initializer = new qnn.Tensor(obj, type, reader);
                    values.map(name, type, initializer);
                    break;
                }
                default: {
                    throw new qnn.Error(`Unsupported tensor type 'type'.`);
                }
            }
        }
        const nodes = Object.entries(obj.nodes);
        for (const [name, obj] of nodes) {
            const node = new qnn.Node(metadata, name, obj, values);
            this.nodes.push(node);
        }
    }
};

qnn.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type;
        this.visible = visible !== false;
    }
};

qnn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new qnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type;
        this.initializer = initializer;
    }
};

qnn.Node = class {

    constructor(metadata, name, obj, values) {
        this.name = name;
        this.type = { name: obj.type, ...metadata.type(obj.type) };
        this.type.module = obj.package;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const inputs = Array.isArray(obj.input_names) ? Array.from(obj.input_names).map((name) => values.map(name)) : [];
        if (Array.isArray(this.type.inputs) && inputs.length === this.type.inputs.length) {
            for (let i = 0; i < inputs.length; i++) {
                const argument = new qnn.Argument(this.type.inputs[i].name, [inputs[i]]);
                this.inputs.push(argument);
            }
        } else if (inputs.length > 0) {
            const argument = new qnn.Argument(inputs.length === 1 ? 'input' : 'inputs', inputs);
            this.inputs.push(argument);
        }
        const outputs = Array.isArray(obj.output_names) ? Array.from(obj.output_names).map((name) => values.map(name)) : [];
        if (Array.isArray(this.type.outputs) && outputs.length === this.type.outputs.length) {
            for (let i = 0; i < outputs.length; i++) {
                const argument = new qnn.Argument(this.type.outputs[i].name, [outputs[i]]);
                this.outputs.push(argument);
            }
        } else if (outputs.length > 0) {
            const argument = new qnn.Argument(outputs.length === 1 ? 'output' : 'outputs', outputs);
            this.outputs.push(argument);
        }
    }
};

qnn.Tensor = class {

    constructor(obj, type, reader) {
        this.type = type;
        this.encoding = '<';
        this._reader = reader;
    }

    get values() {
        if (this._reader) {
            return this._reader.peek();
        }
        return null;
    }
};

qnn.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

qnn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

qnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading QNN model.';
    }
};

export const ModelFactory = qnn.ModelFactory;
