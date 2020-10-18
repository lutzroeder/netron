/* jshint esversion: 6 */

var armnn = armnn || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

armnn.ModelFactory = class {

    match(context) {
        switch (context.identifier.split('.').pop().toLowerCase()) {
            case 'armnn': {
                return true;
            }
            case 'json': {
                const tags = context.tags('json');
                if (tags.has('layers') && tags.has('inputIds') && tags.has('outputIds')) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./armnn-schema').then((schema) => {
            armnn.schema = flatbuffers.get('armnn').armnnSerializer;
            let model = null;
            try {
                const identifier = context.identifier;
                const extension = identifier.split('.').pop().toLowerCase();
                switch (extension) {
                    case 'armnn': {
                        const reader = new flatbuffers.Reader(context.buffer);
                        model = armnn.schema.SerializedGraph.create(reader);
                        break;
                    }
                    case 'json': {
                        const reader = new flatbuffers.TextReader(context.buffer);
                        model = armnn.schema.SerializedGraph.createText(reader);
                        break;
                    }
                }
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new armnn.Error('File format is not armnn.SerializedGraph (' + message.replace(/\.$/, '') + ').');
            }

            return armnn.Metadata.open(host).then((metadata) => {
                return new armnn.Model(metadata, model);
            });
        });
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

    get description() {
        return '';
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

        // generate parameters
        const args = {};
        for (let i = 0; i < graph.layers.length; i++) {
            const base = armnn.Node.getBase(graph.layers[i]);
            for (let j = 0 ; j < base.outputSlots.length ; j++) {
                const key = base.index.toString() + ':' + j.toString();
                args[key] = new armnn.Argument(key, base.outputSlots[j].tensorInfo, null);
            }
        }
        for (let i = 0; i < graph.layers.length; i++) {
            const layer = graph.layers[i];
            const type = layer.layer.constructor.name;
            switch (type) {
                case 'InputLayer': {
                    const base = armnn.Node.getBase(layer);
                    const name = base ? base.layerName : '';
                    for (let j = 0; j < base.outputSlots.length; j++) {
                        const argument = args[base.index.toString() + ':' + j.toString()];
                        this._inputs.push(new armnn.Parameter(name, [ argument ]));
                    }
                    break;
                }
                case 'OutputLayer': {
                    const base = armnn.Node.getBase(layer);
                    const name = base ? base.layerName : '';
                    for (let i = 0; i < base.inputSlots.length; i++) {
                        const connection = base.inputSlots[i].connection;
                        const sourceLayerIndex = connection.sourceLayerIndex;
                        const sourceOutputIndex = connection.outputSlotIndex;
                        const argument = args[sourceLayerIndex.toString() + ':' + sourceOutputIndex.toString()];
                        this._outputs.push(new armnn.Parameter(name, [ argument ]));
                    }
                    break;
                }
                default:
                    this._nodes.push(new armnn.Node(metadata, layer, args));
                    break;
            }

        }
    }

    get name() {
        return this._name;
    }

    get groups() {
        return false;
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

    constructor(metadata, layer, args) {
        this._metadata = metadata;
        this._type = layer.layer.constructor.name;
        this._name = '';
        this._outputs = [];
        this._inputs = [];
        this._attributes = [];

        const base = armnn.Node.getBase(layer);
        if (base) {
            this._name = base.layerName;

            for (let i = 0; i < base.inputSlots.length; i++) {
                const connection = base.inputSlots[i].connection;
                const sourceLayerIndex = connection.sourceLayerIndex;
                const sourceOutputIndex = connection.outputSlotIndex;
                const argument = args[sourceLayerIndex.toString() + ':' + sourceOutputIndex.toString()];
                this._inputs.push(new armnn.Parameter('input', [ argument ]));
            }

            for (let i = 0; i < base.outputSlots.length; i++) {
                const argument = args[base.index.toString() + ':' + i.toString()];
                this._outputs.push(new armnn.Parameter('output', [ argument ]));
            }
        }

        const schema = this._metadata.type(this._type);
        if (schema) {
            const _layer = armnn.Node.castLayer(layer);

            if (schema.bindings) {
                for (let i = 0 ; i < schema.bindings.length ; i++) {
                    const binding = schema.bindings[i];
                    const value = _layer.base()[binding.src]();
                    this._attributes.push(new armnn.Attribute(binding.name, binding.type, value));
                }
            }
            if (schema.attributes) {
                for (const attribute of schema.attributes) {
                    const value = this.packAttr(_layer, attribute);
                    this._attributes.push(new armnn.Attribute(attribute.name, attribute.type, value));
                }
            }
            if (schema.inputs) {
                for (let i = 0 ; i < schema.inputs.length ; i++) {
                    const input = schema.inputs[i];
                    const initializer = _layer[input.src];
                    if (initializer) {
                        const args = [ new armnn.Argument('', null, initializer) ];
                        this._inputs.push(new armnn.Parameter(input.name, args));
                    }
                }
            }
        }
    }

    get type() {
        return this._type.replace(/Layer$/, '');
    }

    get name() {
        return this._name;
    }

    get domain() {
        return null;
    }

    get metadata() {
        return this._metadata.type(this._type);
    }

    get group() {
        return null;
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

    static castLayer(layer) {
        return layer.layer;
    }

    static getBase(layer) {
        layer = armnn.Node.castLayer(layer);
        return layer.base.base ? layer.base.base : layer.base;
    }

    getAttr(descriptor, key) {
        if (typeof descriptor[key] == "undefined")
            return "undefined";

        const values = descriptor[key];
        if (Array.isArray(values)) {
            return values.join(", ");
        }
        else {
            return values;
        }
    }

    packAttr(layer, attr) {
        const descriptor = layer === null ? null : layer.descriptor;
        const key = attr.src;
        const type = attr.src_type;

        if (typeof type != "undefined") {
            const value = this.getAttr(descriptor, key);
            if (typeof armnn.schema[type + "Name"] != "undefined") {
                return armnn.schema[type + "Name"][value];
            }
            else {
                return value;
            }
        }
        else if (Array.isArray(key)) {
            const values = [];
            for (let i = 0 ; i < key.length ; i++) {
                values.push(this.getAttr(descriptor, key[i]));
            }
            return values.join(", ");
        }
        else {
            return this.getAttr(descriptor, key);
        }
    }

    static makeKey(layer_id, index) {
        return layer_id.toString() + "_" + index.toString();
    }
};

armnn.Attribute = class {

    constructor(name, type, value) {
        this._name = name;
        this._value = value;
        this._visible = true;
        switch (type) {
            case 'int': this._type = 'int32'; break;
            case 'uint': this._type = 'uint32'; break;
            case 'float': this._type = 'float32'; break;
            case 'string': this._type = 'string'; break;
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

armnn.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

armnn.Argument = class {

    constructor(name, tensorInfo, initializer) {
        if (typeof name !== 'string') {
            throw new armnn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        const info = initializer ? initializer.info : tensorInfo;
        this._name = name;
        this._type = new armnn.TensorType(info);
        this._initializer = initializer ? new armnn.Tensor(info, initializer) : null;

        if (this._type.dataType.startsWith('q') && info) {
            this._scale = info.quantizationScale;
            this._zeroPoint = info.quantizationOffset;
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

    constructor(tensorInfo, tensor) {
        this._name = '';
        this._type = new armnn.TensorType(tensorInfo);
        this._kind = 'Initializer';
        this._data = tensor.data.data.slice(0);
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        const size = shape[dimension];
        const results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'quint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'qint16':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'boolean':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
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
                throw new armnn.Error("Unknown data type '" + JSON.stringify(dataType) + "'.");
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

armnn.Metadata = class {

    static open(host) {
        if (armnn.Metadata._metadata) {
            return Promise.resolve(armnn.Metadata._metadata);
        }
        return host.request(null, 'armnn-metadata.json', 'utf-8').then((data) => {
            armnn.Metadata._metadata = new armnn.Metadata(data);
            return armnn.Metadata._metadata;
        }).catch(() => {
            armnn.Metadata._metadata = new armnn.Metadata(null);
            return armnn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    type(name) {
        return this._map[name];
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
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
