
// Experimental

const qnn = {};

qnn.ModelFactory = class {

    match(context) {
        const obj = context.peek('json');
        if (obj && obj['model.cpp'] && obj.graph) {
            context.type = 'qnn.json';
            context.target = obj;
            return;
        }
        const entries = context.peek('tar');
        if (entries && entries.size > 0 && Array.from(entries).every(([name]) => name.endsWith('.raw'))) {
            context.type = 'qnn.weights';
            context.target = entries;
            return;
        }
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.bin') || identifier.endsWith('.serialized')) {
            const stream = context.stream;
            const signatures = [
                [0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01],
            ];
            if (stream.length >= 16 && signatures.some((signature) => stream.peek(signature.length).every((value, index) => value === signature[index]))) {
                context.type = 'qnn.serialized';
            }
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
            case 'qnn.serialized': {
                throw new qnn.Error("File contains undocumented QNN serialized context.");
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
        if (obj.converter_command) {
            this.producer = obj.converter_command.split(' ').shift();
        }
        this.metadata = [];
        if (obj.copyright_str) {
            this.metadata.push(new qnn.Argument('License', obj.copyright_str));
        }
        this.graphs = [new qnn.Graph(metadata, obj.graph, weights)];
    }
};

qnn.Graph = class {

    constructor(metadata, obj, weights) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, type, tensor, quantization) => {
            type = type || null;
            tensor = tensor || null;
            if (!values.has(name)) {
                const value = new qnn.Value(name, type, tensor, quantization);
                values.set(name, value);
            } else if ((type && !type.equals(values.get(name).type)) || tensor) {
                throw new qnn.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const tensors = Object.entries(obj.tensors);
        for (const [name, obj] of tensors) {
            const type = new qnn.TensorType(obj);
            switch (obj.type) {
                case 0: {
                    const value = values.map(name, type, null, obj.quant_params);
                    const argument = new qnn.Argument(name, [value]);
                    this.inputs.push(argument);
                    break;
                }
                case 1: {
                    const value = values.map(name, type, null, obj.quant_params);
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
                    const tensor = new qnn.Tensor(name, type, obj, reader);
                    values.map(name, type, tensor, obj.quant_params);
                    break;
                }
                default: {
                    throw new qnn.Error(`Unsupported tensor type '${obj.type}'.`);
                }
            }
        }
        const nodes = Object.entries(obj.nodes);
        for (const [name, obj] of nodes) {
            const node = new qnn.Node(metadata, name, obj, values, weights);
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

    constructor(name, type, initializer, quantization) {
        if (typeof name !== 'string') {
            throw new qnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type;
        this.initializer = initializer;
        if (quantization && quantization.definition === 1 && quantization.scale_offset) {
            this.quantization = {
                type: 'linear',
                scale: [quantization.scale_offset.scale],
                offset: [quantization.scale_offset.offset]
            };
        }
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
        for (const [name, value] of Object.entries(obj.scalar_params)) {
            const entries = Object.entries(value);
            if (entries.length === 1 && name !== 'packageName') {
                const dataType = qnn.Utility.dataType(parseInt(entries[0][0], 10));
                const argument = new qnn.Argument(name, entries[0][1], dataType);
                this.attributes.push(argument);
            }
        }
        for (const [name, value] of Object.entries(obj.tensor_params)) {
            const entries = Object.entries(value);
            if (entries.length === 1 && name !== 'packageName') {
                const tensor = new qnn.Tensor(name, null, entries[0][1]);
                const argument = new qnn.Argument(name, tensor, 'tensor');
                this.attributes.push(argument);
            }
        }
    }
};

qnn.Tensor = class {

    constructor(name, type, obj, data) {
        this.type = type || new qnn.TensorType(obj);
        this.data = obj.data ? obj.data.flat() : data;
        this.encoding = Array.isArray(this.data) ? '|' : '<';
    }

    get values() {
        if (this.data && this.data.peek) {
            return this.data.peek();
        }
        return this.data;
    }
};

qnn.TensorType = class {

    constructor(obj) {
        this.dataType = qnn.Utility.dataType(obj.data_type);
        this.shape = new qnn.TensorShape(obj.dims);
        this.denotation = obj.axis_format && obj.axis_format !== 'ANY' ? obj.axis_format : '';
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

qnn.Utility = class {

    static dataType(value) {
        switch (value) {
            case 0x0008: return 'int8';
            case 0x0016: return 'int16';
            case 0x0032: return 'int32';
            case 0x0064: return 'int64';
            case 0x0108: return 'uint8';
            case 0x0116: return 'uint16';
            case 0x0132: return 'uint32';
            case 0x0164: return 'uint64';
            case 0x0216: return 'float16';
            case 0x0232: return 'float32';
            case 0x0304: return 'qint4';
            case 0x0308: return 'qint8';
            case 0x0316: return 'qint16';
            case 0x0332: return 'qint32';
            case 0x0404: return 'quint4';
            case 0x0408: return 'quint8';
            case 0x0416: return 'quint16';
            case 0x0432: return 'quint32';
            case 0x0508: return 'boolean';
            case 0x0608: return 'string';
            case 0x7fffffff: return 'undefined';
            default: throw new qnn.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
        }
    }
};

qnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading QNN model.';
    }
};

export const ModelFactory = qnn.ModelFactory;
