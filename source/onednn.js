
const onednn = {};

onednn.ModelFactory = class {

    match(context) {
        const obj = context.peek('json');
        if (obj && obj.version && obj.engine_kind && obj.fpmath_mode && obj.graph) {
            context.type = 'onednn';
            context.target = obj;
        }
    }

    async open(context) {
        const metadata = await context.metadata('onednn-metadata.json');
        return new onednn.Model(metadata, context.target);
    }
};

onednn.Model = class {

    constructor(metadata, symbol) {
        const version = symbol.version;
        this._format = `oneDNN Graph${version ? ` v${version}` : ''}`;
        this._runtime = `${symbol.engine_kind} ${symbol.fpmath_mode}`;
        this._graphs = [new onednn.Graph(metadata, symbol)];
    }

    get format() {
        return this._format;
    }

    get version() {
        return this._version;
    }

    get runtime() {
        return this._runtime;
    }

    get graphs() {
        return this._graphs;
    }
};

onednn.Graph = class {

    constructor(metadata, symbol) {
        this._metadata = metadata;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const nodes = [];
        const tensors = new Set();
        for (const node of symbol.graph) {
            if (node.kind === 'Wildcard' && node.inputs.length === 0) {
                for (const output of node.outputs) {
                    tensors.add(output.id);
                }
            } else {
                nodes.push(node);
            }
        }
        const values = new Map();
        const value = (obj) => {
            const id = obj.id;
            const shape = !obj.shape || (obj.shape.length === 1 && obj.shape[0] === -1) ? null : new onednn.TensorShape(obj.shape);
            const type = new onednn.TensorType(obj.dtype, shape);
            const tensor = tensors.has(id) ? new onednn.Tensor(type, obj.property_type) : null;
            if (!values.has(id)) {
                values.set(id, new onednn.Value(id.toString(), type, tensor));
            } else if ((type && !type.equals(values.get(id).type)) || (tensor && !tensor.equals(values.get(id).initializer))) {
                throw new onednn.Error(`Duplicate value '${id}'.`);
            }
            return values.get(id);
        };
        for (const node of nodes) {
            for (const input of node.inputs) {
                value(input);
            }
            for (const output of node.outputs) {
                value(output);
            }
        }
        const engine = symbol.engine_kind;
        for (const node of nodes) {
            this._nodes.push(new onednn.Node(this._metadata, node, engine, value, tensors));
        }
        const inputs = symbol.input_ports || [];
        for (let i = 0; i < inputs.length; i++) {
            const id = inputs[i];
            const value = values.get(id);
            if (value) {
                this._inputs.push(new onednn.Argument(id.toString(), [value]));
            }
        }
        const outputs = symbol.output_ports || [];
        for (let i = 0; i < outputs.length; i++) {
            const id = outputs[i];
            const value = values.get(id);
            if (value) {
                this._outputs.push(new onednn.Argument(id.toString(), [value]));
            }
        }
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

onednn.Node = class {

    constructor(metadata, node, device, value) {
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._type = metadata.type(node.kind) || { name: node.kind };
        this._device = device;
        this._location = node.id;
        const attrs = node.attrs;
        if (attrs) {
            for (const [name, value] of Object.entries(attrs)) {
                this._attributes.push(new onednn.Attribute(name, value.type, value.value));
            }
        }
        const inputs = node.inputs || [];
        for (let i = 0; i < inputs.length; i++) {
            let name = inputs.length === 1 ? 'input' : i.toString();
            if (this._type && this._type.inputs && this._type.inputs.length > 0) {
                name = this._type.inputs[i].name;
            }
            this._inputs.push(new onednn.Argument(name, [value(inputs[i])]));
        }
        const outputs = node.outputs || [];
        for (let i = 0; i < outputs.length; i++) {
            let name = outputs.length === 1 ? 'output' : i.toString();
            if (this._type && this._type.outputs && this._type.outputs.length > 0) {
                name = this._type.outputs[i].name;
            }
            this._outputs.push(new onednn.Argument(name, [value(outputs[i])]));
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get location() {
        return this._location;
    }

    get device() {
        return this._device;
    }
};

onednn.Attribute = class {

    constructor(name, type, value) {
        this._name = name;
        this._value = value;
        switch (type) {
            case 'bool':
                this._type = 'boolean';
                switch (value) {
                    case 1: this._value = true; break;
                    case 0: this._value = false; break;
                    default: throw new onednn.Error(`Unsupported attribute boolean value '${value}'.`);
                }
                break;
            case 's64': {
                this._type = 'int64';
                const number = Number.parseInt(this._value, 10);
                this._value = Number.isNaN(this._value - number) ? value : number;
                break;
            }
            case 's64[]':
                this._type = 'int64[]';
                if (this._value.length > 2 && this._value.toString().startsWith('[') && this._value.toString().endsWith(']')) {
                    let array = [];
                    const items = this._value.substring(1, this._value.length - 1).split(',')
                        .map((item) => item.trim())
                        .map((item) => item.endsWith('L') ? item.substring(0, item.length - 1) : item);
                    for (const item of items) {
                        const value = Number.parseInt(item, 10);
                        if (Number.isNaN(item - value)) {
                            array = null;
                        } else if (array !== null) {
                            array.push(value);
                        }
                    }
                    if (array !== null) {
                        this._value = array;
                    }
                }
                break;
            case 'f32': {
                this._type = 'float32';
                const number = Number.parseFloat(this._value);
                this._value = Number.isNaN(this._value - number) ? value : number;
                break;
            }
            case 'f32[]':
                this._type = 'float32[]';
                if (this._value.length > 2 && this._value.toString().startsWith('[') && this._value.toString().endsWith(']')) {
                    let array = [];
                    const items = this._value.substring(1, this._value.length - 1).split(',')
                        .map((item) => item.trim())
                        .map((item) => item.endsWith('L') ? item.substring(0, item.length - 1) : item);
                    for (const item of items) {
                        const value = Number.parseFloat(item);
                        if (Number.isNaN(item - value)) {
                            array = null;
                        } else if (array !== null) {
                            array.push(value);
                        }
                    }
                    if (array !== null) {
                        this._value = array;
                    }
                }
                break;
            case 'string':
                this._type = 'string';
                break;
            default: {
                throw new onednn.Error(`Unsupported attribute array data type '${type}'.`);
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible !== false;
    }
};

onednn.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

onednn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new onednn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

onednn.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case 'f8_e4m3': this._dataType = 'float8e4m3'; break;
            case 'f8_e5m2': this._dataType = 'float8e5m2'; break;
            case 'f16': this._dataType = 'float16'; break;
            case 'f32': this._dataType = 'float32'; break;
            case 's8': this._dataType = 'int8'; break;
            case 's32': this._dataType = 'int32'; break;
            case 'u8': this._dataType = 'uint8'; break;
            case 'bf16': this._dataType = 'bfloat16'; break;
            case 'boolean': this._dataType = 'boolean'; break;
            case 'undef': this._dataType = '?'; break;
            default: throw new onednn.Error(`Unsupported tensor data type '${dataType}'.`);
        }
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    equals(obj) {
        return obj && this._dataType === obj.dataType &&
            ((this._shape && this._shape.equals(obj.shape)) || (this._shape === null && obj.shape === null));
    }

    toString() {
        return this._dataType + (this._shape ? this._shape.toString() : '[?]');
    }
};

onednn.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) &&
            Array.isArray(this._dimensions) && this._dimensions.length === obj.dimensions.length
            && obj.dimensions.every((value, index) => this._dimensions[index] === value);
    }

    toString() {
        return this._dimensions ? (`[${this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`) : '';
    }
};

onednn.Tensor = class {

    constructor(type, property_type) {
        this._type = type;
        this._category = property_type;
    }

    get type() {
        return this._type;
    }

    get category() {
        return this._category;
    }

    equals(obj) {
        return obj && this._type.equals(obj.type) && this.category === obj.category;
    }
};

onednn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading oneDNN Graph model.';
    }
};

export const ModelFactory = onednn.ModelFactory;

