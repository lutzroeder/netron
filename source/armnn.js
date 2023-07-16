
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
        this._graphs = [];
        this._graphs.push(new armnn.Graph(metadata, model));
    }

    get format() {
        return 'Arm NN';
    }

    get graphs() {
        return this._graphs;
    }
};

armnn.Graph = class {

    constructor(metadata, graph) {
        this._name = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const counts = new Map();
        for (const layer of graph.layers) {
            const base = armnn.Node.getBase(layer);
            for (const slot of base.inputSlots) {
                const name = slot.connection.sourceLayerIndex.toString() + ':' + slot.connection.outputSlotIndex.toString();
                counts.set(name, counts.has(name) ? counts.get(name) + 1 : 1);
            }
        }
        const args = new Map();
        const arg = (layerIndex, slotIndex, tensor) => {
            const name = layerIndex.toString() + ':' + slotIndex.toString();
            if (!args.has(name)) {
                const layer = graph.layers[layerIndex];
                const base = layerIndex < graph.layers.length ? armnn.Node.getBase(layer) : null;
                const tensorInfo = base && slotIndex < base.outputSlots.length ? base.outputSlots[slotIndex].tensorInfo : null;
                args.set(name, new armnn.Value(name, tensorInfo, tensor));
            }
            return args.get(name);
        };
        const layers = graph.layers.filter((layer) => {
            const base = armnn.Node.getBase(layer);
            if (base.layerType == armnn.schema.LayerType.Constant && base.outputSlots.length === 1 && layer.layer.input) {
                const slot = base.outputSlots[0];
                const name = base.index.toString() + ':' + slot.index.toString();
                if (counts.get(name) === 1) {
                    const tensor = new armnn.Tensor(layer.layer.input, 'Constant');
                    arg(base.index, slot.index, tensor);
                    return false;
                }
            }
            return true;
        });
        for (const layer of layers) {
            const base = armnn.Node.getBase(layer);
            for (const slot of base.inputSlots) {
                arg(slot.connection.sourceLayerIndex, slot.connection.outputSlotIndex);
            }
        }
        for (const layer of layers) {
            const base = armnn.Node.getBase(layer);
            switch (base.layerType) {
                case armnn.schema.LayerType.Input: {
                    const name = base ? base.layerName : '';
                    for (const slot of base.outputSlots) {
                        const value = arg(base.index, slot.index);
                        this._inputs.push(new armnn.Argument(name, [ value ]));
                    }
                    break;
                }
                case armnn.schema.LayerType.Output: {
                    const base = armnn.Node.getBase(layer);
                    const name = base ? base.layerName : '';
                    for (const slot of base.inputSlots) {
                        const value = arg(slot.connection.sourceLayerIndex, slot.connection.outputSlotIndex);
                        this._outputs.push(new armnn.Argument(name, [ value ]));
                    }
                    break;
                }
                default:
                    this._nodes.push(new armnn.Node(metadata, layer, arg));
                    break;
            }
        }
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

    get nodes() {
        return this._nodes;
    }
};

armnn.Node = class {

    constructor(metadata, layer, arg) {
        const type = layer.layer.constructor.name;
        this._type = Object.assign({}, metadata.type(type) || { name: type });
        this._type.name = this._type.name.replace(/Layer$/, '');
        this._name = '';
        this._outputs = [];
        this._inputs = [];
        this._attributes = [];
        const inputSchemas = (this._type && this._type.inputs) ? [...this._type.inputs] : [ { name: 'input' } ];
        const outputSchemas = (this._type && this._type.outputs) ? [...this._type.outputs] : [ { name: 'output' } ];
        const base = armnn.Node.getBase(layer);
        if (base) {
            this._name = base.layerName;
            const inputSlots = [...base.inputSlots];
            while (inputSlots.length > 0) {
                const inputSchema = inputSchemas.length > 0 ? inputSchemas.shift() : { name: '?' };
                const inputCount = inputSchema.list ? inputSlots.length : 1;
                this._inputs.push(new armnn.Argument(inputSchema.name, inputSlots.splice(0, inputCount).map((inputSlot) => {
                    return arg(inputSlot.connection.sourceLayerIndex, inputSlot.connection.outputSlotIndex);
                })));
            }
            const outputSlots = [...base.outputSlots];
            while (outputSlots.length > 0) {
                const outputSchema = outputSchemas.length > 0 ? outputSchemas.shift() : { name: '?' };
                const outputCount = outputSchema.list ? outputSlots.length : 1;
                this._outputs.push(new armnn.Argument(outputSchema.name, outputSlots.splice(0, outputCount).map((outputSlot) => {
                    return arg(base.index, outputSlot.index);
                })));
            }
        }
        if (layer.layer && layer.layer.descriptor && this._type.attributes) {
            for (const pair of Object.entries(layer.layer.descriptor)) {
                const name = pair[0];
                const value = pair[1];
                const attribute = new armnn.Attribute(metadata.attribute(type, name), name, value);
                this._attributes.push(attribute);
            }
        }
        if (layer.layer) {
            for (const entry of Object.entries(layer.layer).filter((entry) => entry[1] instanceof armnn.schema.ConstTensor)) {
                const name = entry[0];
                const tensor = entry[1];
                const value = new armnn.Value('', tensor.info, new armnn.Tensor(tensor));
                this._inputs.push(new armnn.Argument(name, [ value ]));
            }
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

    static getBase(layer) {
        return layer.layer.base.base ? layer.layer.base.base : layer.layer.base;
    }

    static makeKey(layer_id, index) {
        return layer_id.toString() + "_" + index.toString();
    }
};

armnn.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._type = metadata ? metadata.type : null;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        if (armnn.schema[this._type]) {
            this._value = armnn.Utility.enum(this._type, this._value);
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
        return this._visible == false ? false : true;
    }
};

armnn.Argument = class {

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

armnn.Value = class {

    constructor(name, tensorInfo, initializer) {
        if (typeof name !== 'string') {
            throw new armnn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = new armnn.TensorType(tensorInfo);
        this._initializer = initializer;

        if (this._type.dataType.startsWith('q') && tensorInfo) {
            this._scale = tensorInfo.quantizationScale;
            this._zeroPoint = tensorInfo.quantizationOffset;
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        if (this._scale !== undefined && this._zeroPoint !== undefined) {
            return this._scale.toString() + ' * ' + (this._zeroPoint == 0 ? 'q' : ('(q - ' + this._zeroPoint.toString() + ')'));
        }
        return undefined;
    }

    get initializer() {
        return this._initializer;
    }
};

armnn.Tensor = class {

    constructor(tensor, category) {
        this._type = new armnn.TensorType(tensor.info);
        this._category = category || '';
        const data = tensor.data.data.slice(0);
        this._values = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    }

    get category() {
        return this._category;
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._values;
    }
};

armnn.TensorType = class {

    constructor(tensorInfo) {
        const dataType = tensorInfo.dataType;
        switch (dataType) {
            case 0: this._dataType = 'float16'; break;
            case 1: this._dataType = 'float32'; break;
            case 2: this._dataType = 'quint8'; break; // QuantisedAsymm8
            case 3: this._dataType = 'int32'; break;
            case 4: this._dataType = 'boolean'; break;
            case 5: this._dataType = 'qint16'; break; // QuantisedSymm16
            case 6: this._dataType = 'quint8'; break; // QAsymmU8
            case 7: this._dataType = 'qint16'; break; // QSymmS16
            case 8: this._dataType = 'qint8'; break; // QAsymmS8
            case 9: this._dataType = 'qint8'; break; // QSymmS8
            default:
                throw new armnn.Error("Unsupported data type '" + JSON.stringify(dataType) + "'.");
        }
        this._shape = new armnn.TensorShape(tensorInfo.dimensions);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

armnn.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = Array.from(dimensions);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
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
