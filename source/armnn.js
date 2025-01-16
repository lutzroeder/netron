
const armnn = {};

armnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'armnn') {
            const reader = context.peek('flatbuffers.binary');
            if (reader) {
                context.type = 'armnn.flatbuffers';
                context.target = reader;
                return;
            }
        }
        if (extension === 'json') {
            const obj = context.peek('json');
            if (obj && obj.layers && obj.inputIds && obj.outputIds) {
                context.type = 'armnn.flatbuffers.json';
                context.target = obj;
            }
        }
    }

    async open(context) {
        armnn.schema = await context.require('./armnn-schema');
        armnn.schema = armnn.schema.armnnSerializer;
        let model = null;
        switch (context.type) {
            case 'armnn.flatbuffers': {
                try {
                    const reader = context.read('flatbuffers.binary');
                    model = armnn.schema.SerializedGraph.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new armnn.Error(`File format is not armnn.SerializedGraph (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            case 'armnn.flatbuffers.json': {
                try {
                    const reader = context.read('flatbuffers.text');
                    model = armnn.schema.SerializedGraph.createText(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new armnn.Error(`File text format is not armnn.SerializedGraph (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            default: {
                throw new armnn.Error(`Unsupported Arm NN format '${context.type}'.`);
            }
        }
        const metadata = await context.metadata('armnn-metadata.json');
        return new armnn.Model(metadata, model);
    }
};

armnn.Model = class {

    constructor(metadata, model) {
        this.format = 'Arm NN';
        this.graphs = [new armnn.Graph(metadata, model)];
    }
};

armnn.Graph = class {

    constructor(metadata, graph) {
        this.name = '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const counts = new Map();
        for (const layer of graph.layers) {
            const base = armnn.Node.getBase(layer);
            for (const slot of base.inputSlots) {
                const name = `${slot.connection.sourceLayerIndex}:${slot.connection.outputSlotIndex}`;
                counts.set(name, counts.has(name) ? counts.get(name) + 1 : 1);
            }
        }
        const values = new Map();
        const value = (layerIndex, slotIndex, tensor) => {
            const name = `${layerIndex}:${slotIndex}`;
            if (!values.has(name)) {
                const layer = graph.layers[layerIndex];
                const base = layerIndex < graph.layers.length ? armnn.Node.getBase(layer) : null;
                const tensorInfo = base && slotIndex < base.outputSlots.length ? base.outputSlots[slotIndex].tensorInfo : null;
                values.set(name, new armnn.Value(name, tensorInfo, tensor));
            }
            return values.get(name);
        };
        const layers = graph.layers.filter((layer) => {
            const base = armnn.Node.getBase(layer);
            if (base.layerType === armnn.schema.LayerType.Constant && base.outputSlots.length === 1 && layer.layer.input) {
                /* eslint-disable prefer-destructuring */
                const slot = base.outputSlots[0];
                /* eslint-enable prefer-destructuring */
                const name = `${base.index}:${slot.index}`;
                if (counts.get(name) === 1) {
                    const tensor = new armnn.Tensor(layer.layer.input, 'Constant');
                    value(base.index, slot.index, tensor);
                    return false;
                }
            }
            return true;
        });
        for (const layer of layers) {
            const base = armnn.Node.getBase(layer);
            for (const slot of base.inputSlots) {
                value(slot.connection.sourceLayerIndex, slot.connection.outputSlotIndex);
            }
        }
        for (const layer of layers) {
            const base = armnn.Node.getBase(layer);
            switch (base.layerType) {
                case armnn.schema.LayerType.Input: {
                    const name = base ? base.layerName : '';
                    for (const slot of base.outputSlots) {
                        const argument = new armnn.Argument(name, [value(base.index, slot.index)]);
                        this.inputs.push(argument);
                    }
                    break;
                }
                case armnn.schema.LayerType.Output: {
                    const base = armnn.Node.getBase(layer);
                    const name = base ? base.layerName : '';
                    for (const slot of base.inputSlots) {
                        const argument = new armnn.Argument(name, [value(slot.connection.sourceLayerIndex, slot.connection.outputSlotIndex)]);
                        this.outputs.push(argument);
                    }
                    break;
                }
                default:
                    this.nodes.push(new armnn.Node(metadata, layer, value));
                    break;
            }
        }
    }
};

armnn.Node = class {

    constructor(metadata, layer, value) {
        const name = layer.layer.constructor.name;
        const type = metadata.type(name);
        this.type = type ? { ...type } : { name };
        this.type.name = this.type.name.replace(/Layer$/, '');
        this.name = '';
        this.outputs = [];
        this.inputs = [];
        this.attributes = [];
        const inputSchemas = (this.type && this.type.inputs) ? [...this.type.inputs] : [{ name: 'input' }];
        const outputSchemas = (this.type && this.type.outputs) ? [...this.type.outputs] : [{ name: 'output' }];
        const base = armnn.Node.getBase(layer);
        if (base) {
            this.name = base.layerName;
            const inputs = [...base.inputSlots];
            while (inputs.length > 0) {
                const schema = inputSchemas.length > 0 ? inputSchemas.shift() : { name: '?' };
                const count = schema.list ? inputs.length : 1;
                const argument = new armnn.Argument(schema.name, inputs.splice(0, count).map((inputSlot) => {
                    return value(inputSlot.connection.sourceLayerIndex, inputSlot.connection.outputSlotIndex);
                }));
                this.inputs.push(argument);
            }
            const outputs = [...base.outputSlots];
            while (outputs.length > 0) {
                const schema = outputSchemas.length > 0 ? outputSchemas.shift() : { name: '?' };
                const count = schema.list ? outputs.length : 1;
                this.outputs.push(new armnn.Argument(schema.name, outputs.splice(0, count).map((outputSlot) => {
                    return value(base.index, outputSlot.index);
                })));
            }
        }
        if (layer.layer) {
            if (layer.layer.descriptor && this.type.attributes) {
                for (const [key, obj] of Object.entries(layer.layer.descriptor)) {
                    const schema = metadata.attribute(name, key);
                    const type = schema ? schema.type : null;
                    let value = ArrayBuffer.isView(obj) ? Array.from(obj) : obj;
                    if (armnn.schema[type]) {
                        value = armnn.Utility.enum(type, value);
                    }
                    const attribute = new armnn.Argument(key, value, type);
                    this.attributes.push(attribute);
                }
            }
            for (const [name, tensor] of Object.entries(layer.layer).filter(([, value]) => value instanceof armnn.schema.ConstTensor)) {
                const value = new armnn.Value('', tensor.info, new armnn.Tensor(tensor));
                const argument = new armnn.Argument(name, [value]);
                this.inputs.push(argument);
            }
        }
    }

    static getBase(layer) {
        return layer.layer.base.base ? layer.layer.base.base : layer.layer.base;
    }

    static makeKey(layer_id, index) {
        return `${layer_id}_${index}`;
    }
};

armnn.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

armnn.Value = class {

    constructor(name, tensorInfo, initializer) {
        if (typeof name !== 'string') {
            throw new armnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = new armnn.TensorType(tensorInfo);
        this.initializer = initializer;
        if (tensorInfo.quantizationScale !== 0 ||
            tensorInfo.quantizationOffset !== 0 ||
            tensorInfo.quantizationScales.length > 0 ||
            tensorInfo.quantizationDim !== 0) {
            this.quantization = {
                type: 'linear',
                dimension: tensorInfo.quantizationDim,
                scale: [tensorInfo.quantizationScale],
                offset: [tensorInfo.quantizationOffset]
            };
        }
    }
};

armnn.Tensor = class {

    constructor(tensor, category) {
        this.type = new armnn.TensorType(tensor.info);
        this.category = category || '';
        const data = tensor.data.data.slice(0);
        this.values = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    }
};

armnn.TensorType = class {

    constructor(tensorInfo) {
        const dataType = tensorInfo.dataType;
        switch (dataType) {
            case 0: this.dataType = 'float16'; break;
            case 1: this.dataType = 'float32'; break;
            case 2: this.dataType = 'quint8'; break; // QuantisedAsymm8
            case 3: this.dataType = 'int32'; break;
            case 4: this.dataType = 'boolean'; break;
            case 5: this.dataType = 'qint16'; break; // QuantisedSymm16
            case 6: this.dataType = 'quint8'; break; // QAsymmU8
            case 7: this.dataType = 'qint16'; break; // QSymmS16
            case 8: this.dataType = 'qint8'; break; // QAsymmS8
            case 9: this.dataType = 'qint8'; break; // QSymmS8
            default:
                throw new armnn.Error(`Unsupported data type '${JSON.stringify(dataType)}'.`);
        }
        this.shape = new armnn.TensorShape(tensorInfo.dimensions);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

armnn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

armnn.Utility = class {

    static enum(name, value) {
        const type = name && armnn.schema ? armnn.schema[name] : undefined;
        if (type) {
            armnn.Utility._enums = armnn.Utility._enums || new Map();
            if (!armnn.Utility._enums.has(name)) {
                const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
                armnn.Utility._enums.set(name, entries);
            }
            const entries = armnn.Utility._enums.get(name);
            if (entries.has(value)) {
                return entries.get(value);
            }
        }
        return value;
    }
};

armnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Arm NN model.';
    }
};

export const ModelFactory = armnn.ModelFactory;
