
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
        this.format = `oneDNN${version ? ` v${version}` : ''}`;
        this.runtime = `${symbol.engine_kind} ${symbol.fpmath_mode}`;
        this.graphs = [new onednn.Graph(metadata, symbol)];
    }
};

onednn.Graph = class {

    constructor(metadata, symbol) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const layers = [];
        const tensors = new Set();
        for (const layer of symbol.graph) {
            if (layer.kind === 'Wildcard' && layer.inputs.length === 0) {
                for (const output of layer.outputs) {
                    tensors.add(output.id);
                }
            } else {
                layers.push(layer);
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
        for (const layer of layers) {
            for (const input of layer.inputs) {
                value(input);
            }
            for (const output of layer.outputs) {
                value(output);
            }
        }
        const engine = symbol.engine_kind;
        for (const layer of layers) {
            const node = new onednn.Node(metadata, layer, engine, value, tensors);
            this.nodes.push(node);
        }
        const inputs = symbol.input_ports || [];
        for (const input of inputs) {
            const value = values.get(input);
            if (value) {
                const argument = new onednn.Argument(input.toString(), [value]);
                this.inputs.push(argument);
            }
        }
        const outputs = symbol.output_ports || [];
        for (const output of outputs) {
            const value = values.get(output);
            if (value) {
                const argument = new onednn.Argument(output.toString(), [value]);
                this.outputs.push(argument);
            }
        }
    }
};

onednn.Node = class {

    constructor(metadata, node, device, value) {
        this.name = node.name;
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        this.type = metadata.type(node.kind) || { name: node.kind };
        this.device = device;
        this.identifier = node.id;
        const attrs = node.attrs;
        if (attrs) {
            for (const [name, obj] of Object.entries(attrs)) {
                let type = obj.type;
                let value = obj.value;
                switch (type) {
                    case 'bool':
                        type = 'boolean';
                        switch (value) {
                            case 1: value = true; break;
                            case 0: value = false; break;
                            default: throw new onednn.Error(`Unsupported attribute boolean value '${value}'.`);
                        }
                        break;
                    case 's64': {
                        type = 'int64';
                        const number = Number.parseInt(value, 10);
                        value = Number.isNaN(value - number) ? value : number;
                        break;
                    }
                    case 's64[]':
                        type = 'int64[]';
                        if (value.length > 2 && value.toString().startsWith('[') && value.toString().endsWith(']')) {
                            let array = [];
                            const items = value.substring(1, value.length - 1).split(',')
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
                                value = array;
                            }
                        }
                        break;
                    case 'f32': {
                        type = 'float32';
                        const number = Number.parseFloat(value);
                        value = Number.isNaN(value - number) ? value : number;
                        break;
                    }
                    case 'f32[]':
                        type = 'float32[]';
                        if (value.length > 2 && value.toString().startsWith('[') && value.toString().endsWith(']')) {
                            let array = [];
                            const items = value.substring(1, value.length - 1).split(',')
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
                                value = array;
                            }
                        }
                        break;
                    case 'string':
                        type = 'string';
                        break;
                    default: {
                        throw new onednn.Error(`Unsupported attribute array data type '${type}'.`);
                    }
                }
                const attribute = new onednn.Argument(name, value, type);
                this.attributes.push(attribute);
            }
        }
        const inputs = node.inputs || [];
        for (let i = 0; i < inputs.length; i++) {
            let name = inputs.length === 1 ? 'input' : i.toString();
            if (this.type && this.type.inputs && this.type.inputs.length > 0) {
                name = this.type.inputs[i].name;
            }
            const argument = new onednn.Argument(name, [value(inputs[i])]);
            this.inputs.push(argument);
        }
        const outputs = node.outputs || [];
        for (let i = 0; i < outputs.length; i++) {
            let name = outputs.length === 1 ? 'output' : i.toString();
            if (this.type && this.type.outputs && this.type.outputs.length > 0) {
                name = this.type.outputs[i].name;
            }
            const argument = new onednn.Argument(name, [value(outputs[i])]);
            this.outputs.push(argument);
        }
    }
};

onednn.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

onednn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new onednn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type || null;
        this.initializer = initializer || null;
    }
};

onednn.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case 'f8_e4m3': this.dataType = 'float8e4m3'; break;
            case 'f8_e5m2': this.dataType = 'float8e5m2'; break;
            case 'f16': this.dataType = 'float16'; break;
            case 'f32': this.dataType = 'float32'; break;
            case 's4': this.dataType = 'int4'; break;
            case 's8': this.dataType = 'int8'; break;
            case 's32': this.dataType = 'int32'; break;
            case 'u4': this.dataType = 'uint4'; break;
            case 'u8': this.dataType = 'uint8'; break;
            case 'bf16': this.dataType = 'bfloat16'; break;
            case 'boolean': this.dataType = 'boolean'; break;
            case 'undef': this.dataType = '?'; break;
            default: throw new onednn.Error(`Unsupported tensor data type '${dataType}'.`);
        }
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType &&
            ((this.shape && this.shape.equals(obj.shape)) || (this.shape === null && obj.shape === null));
    }

    toString() {
        return this.dataType + (this.shape ? this.shape.toString() : '[?]');
    }
};

onednn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) &&
            Array.isArray(this.dimensions) && this.dimensions.length === obj.dimensions.length
            && obj.dimensions.every((value, index) => this.dimensions[index] === value);
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`) : '';
    }
};

onednn.Tensor = class {

    constructor(type, property_type) {
        this.type = type;
        this.category = property_type;
    }

    equals(obj) {
        return obj && this.type.equals(obj.type) && this.category === obj.category;
    }
};

onednn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading oneDNN Graph model.';
    }
};

export const ModelFactory = onednn.ModelFactory;
