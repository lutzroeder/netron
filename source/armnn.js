
var armnn = {};
var flatbuffers = require('./flatbuffers');

armnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        if (stream && extension === 'armnn') {
            return 'armnn.flatbuffers';
        }
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj && obj.layers && obj.inputIds && obj.outputIds) {
                return 'armnn.flatbuffers.json';
            }
        }
        return undefined;
    }

    async open(context, target) {
        await context.require('./armnn-schema');
        armnn.schema = flatbuffers.get('armnn').armnnSerializer;
        let model = null;
        switch (target) {
            case 'armnn.flatbuffers': {
                try {
                    const stream = context.stream;
                    const reader = flatbuffers.BinaryReader.open(stream);
                    model = armnn.schema.SerializedGraph.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new armnn.Error('File format is not armnn.SerializedGraph (' + message.replace(/\.$/, '') + ').');
                }
                break;
            }
            case 'armnn.flatbuffers.json': {
                try {
                    const obj = context.open('json');
                    const reader = flatbuffers.TextReader.open(obj);
                    model = armnn.schema.SerializedGraph.createText(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new armnn.Error('File text format is not armnn.SerializedGraph (' + message.replace(/\.$/, '') + ').');
                }
                break;
            }
            default: {
                throw new armnn.Error("Unsupported Arm NN '" + target + "'.");
            }
        }
        const metadata = await context.metadata('armnn-metadata.json');
        return new armnn.Model(metadata, model);
    }
};

armnn.Model = class {

    constructor(metadata, model) {
        this.format = 'Arm NN';
        this.graphs = [ new armnn.Graph(metadata, model) ];
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
                const name = slot.connection.sourceLayerIndex.toString() + ':' + slot.connection.outputSlotIndex.toString();
                counts.set(name, counts.has(name) ? counts.get(name) + 1 : 1);
            }
        }
        const values = new Map();
        const value = (layerIndex, slotIndex, tensor) => {
            const name = layerIndex.toString() + ':' + slotIndex.toString();
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
            if (base.layerType == armnn.schema.LayerType.Constant && base.outputSlots.length === 1 && layer.layer.input) {
                const slot = base.outputSlots[0];
                const name = base.index.toString() + ':' + slot.index.toString();
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
                        const argument = new armnn.Argument(name, [ value(base.index, slot.index) ]);
                        this.inputs.push(argument);
                    }
                    break;
                }
                case armnn.schema.LayerType.Output: {
                    const base = armnn.Node.getBase(layer);
                    const name = base ? base.layerName : '';
                    for (const slot of base.inputSlots) {
                        const argument = new armnn.Argument(name, [ value(slot.connection.sourceLayerIndex, slot.connection.outputSlotIndex) ]);
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
        const type = layer.layer.constructor.name;
        this.type = Object.assign({}, metadata.type(type) || { name: type });
        this.type.name = this.type.name.replace(/Layer$/, '');
        this.name = '';
        this.outputs = [];
        this.inputs = [];
        this.attributes = [];
        const inputSchemas = (this.type && this.type.inputs) ? [...this.type.inputs] : [ { name: 'input' } ];
        const outputSchemas = (this.type && this.type.outputs) ? [...this.type.outputs] : [ { name: 'output' } ];
        const base = armnn.Node.getBase(layer);
        if (base) {
            this.name = base.layerName;
            const inputs = [...base.inputSlots];
            while (inputs.length > 0) {
                const inputSchema = inputSchemas.length > 0 ? inputSchemas.shift() : { name: '?' };
                const count = inputSchema.list ? inputs.length : 1;
                const argument = new armnn.Argument(inputSchema.name, inputs.splice(0, count).map((inputSlot) => {
                    return value(inputSlot.connection.sourceLayerIndex, inputSlot.connection.outputSlotIndex);
                }));
                this.inputs.push(argument);
            }
            const outputs = [...base.outputSlots];
            while (outputs.length > 0) {
                const outputSchema = outputSchemas.length > 0 ? outputSchemas.shift() : { name: '?' };
                const count = outputSchema.list ? outputs.length : 1;
                this.outputs.push(new armnn.Argument(outputSchema.name, outputs.splice(0, count).map((outputSlot) => {
                    return value(base.index, outputSlot.index);
                })));
            }
        }
        if (layer.layer) {
            if (layer.layer.descriptor && this.type.attributes) {
                for (const entry of Object.entries(layer.layer.descriptor)) {
                    const name = entry[0];
                    const value = entry[1];
                    const attribute = new armnn.Attribute(metadata.attribute(type, name), name, value);
                    this.attributes.push(attribute);
                }
            }
            for (const entry of Object.entries(layer.layer).filter((entry) => entry[1] instanceof armnn.schema.ConstTensor)) {
                const name = entry[0];
                const tensor = entry[1];
                const value = new armnn.Value('', tensor.info, new armnn.Tensor(tensor));
                this.inputs.push(new armnn.Argument(name, [ value ]));
            }
        }
    }

    static getBase(layer) {
        return layer.layer.base.base ? layer.layer.base.base : layer.layer.base;
    }

    static makeKey(layer_id, index) {
        return layer_id.toString() + "_" + index.toString();
    }
};

armnn.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.type = metadata ? metadata.type : null;
        this.value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        if (armnn.schema[this.type]) {
            this.value = armnn.Utility.enum(this.type, this.value);
        }
    }
};

armnn.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

armnn.Value = class {

    constructor(name, tensorInfo, initializer) {
        if (typeof name !== 'string') {
            throw new armnn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        this.type = new armnn.TensorType(tensorInfo);
        this.initializer = initializer;

        if (this.type.dataType.startsWith('q') && tensorInfo) {
            this._scale = tensorInfo.quantizationScale;
            this._zeroPoint = tensorInfo.quantizationOffset;
        }
    }

    get quantization() {
        if (this._scale !== undefined && this._zeroPoint !== undefined) {
            return this._scale.toString() + ' * ' + (this._zeroPoint == 0 ? 'q' : ('(q - ' + this._zeroPoint.toString() + ')'));
        }
        return undefined;
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
                throw new armnn.Error("Unsupported data type '" + JSON.stringify(dataType) + "'.");
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
        if (!this.dimensions || this.dimensions.length == 0) {
            return '';
        }
        return '[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

armnn.Utility = class {

    static enum(name, value) {
        const type = name && armnn.schema ? armnn.schema[name] : undefined;
        if (type) {
            armnn.Utility._enums = armnn.Utility._enums || new Map();
            if (!armnn.Utility._enums.has(name)) {
                const map = new Map(Object.keys(type).map((key) => [ type[key], key ]));
                armnn.Utility._enums.set(name, map);
            }
            const map = armnn.Utility._enums.get(name);
            if (map.has(value)) {
                return map.get(value);
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = armnn.ModelFactory;
}
