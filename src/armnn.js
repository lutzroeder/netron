/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var armnn = armnn || {};
var base = base || require('./base');
var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;
var long = long || { Long: require('long') };

armnn.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'armnn') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return host.require('./armnn-schema').then((armnn_schema) => {
            const identifier = context.identifier;
            let model = null;
            try {
                const buffer = context.buffer;
                const byteBuffer = new flatbuffers.ByteBuffer(buffer);
                armnn.schema = armnn_schema;
                model = armnn.schema.SerializedGraph.getRootAsSerializedGraph(byteBuffer);
            }
            catch (error) {
                host.exception(error, false);
                let message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new armnn.Error(message + " in '" + identifier + "'.");
            }

            return armnn.Metadata.open(host).then((metadata) => {
                try {
                    return new armnn.Model(model, metadata);
                }
                catch (error) {
                    let message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new new armnn.Error(message + " in '" + identifier + "'.");
                }
            });
        });
    }
};

armnn.Model = class {

    constructor(model, metadata) {
        this._graphs = [];
        this._graphs.push(new armnn.Graph(model, metadata))
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

    constructor(graph, metadata) {
        this._name = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        let params = {};

        // generate parameters
        let paramIdx = 0;
        for (let j = 0; j < graph.layersLength(); j++) {
            let base = armnn.Node.getBase(graph.layers(j));
            for (let i = 0 ; i < base.outputSlotsLength() ; i++) {
                let key = armnn.Parameter.makeKey(base.index(), i);
                let name = paramIdx.toString();
                let args = [ new armnn.Argument(name, base.outputSlots(i).tensorInfo(), null) ];
                params[key] = new armnn.Parameter(name, name, args);
                paramIdx++;
            }
        }

        // generate nodes
        for (let j = 0; j < graph.layersLength(); j++) {
            this._nodes.push(new armnn.Node(graph.layers(j), params, metadata));
        }

        // link inputs
        for (let k = 0; k < graph.inputIdsLength(); k++) {
            // need to do something?
        }

        // link outputs
        for (let l = 0; l < graph.outputIdsLength(); l++) {
            // need to do something?
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

    constructor(layer, params, metadata) {
        this._metadata = metadata;
        this._operator = armnn.schema.LayerName[layer.layerType()].replace(/Layer$/, '');

        this._name = '';
        this._outputs = [];
        this._inputs = [];
        this._category = '';
        this._attributes = [];

        let base = armnn.Node.getBase(layer)

        if (base) {
            this._name = base.layerName();

            for (let i = 0; i < base.inputSlotsLength(); i++) {
                let srcConnection = base.inputSlots(i).connection();
                let srcLayerIdx = srcConnection.sourceLayerIndex()
                let srcOutputIdx = srcConnection.outputSlotIndex()

                this._inputs.push(params[armnn.Parameter.makeKey(srcLayerIdx, srcOutputIdx)]);
            }

            for (let j = 0; j < base.outputSlotsLength(); j++) {
                this._outputs.push(params[armnn.Parameter.makeKey(base.index(), j)]);
            }
        }
        this.setAttribute(layer);
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get domain() {
        return null;
    }

    get documentation() {
        return '';
    }

    get group() {
        return null;
    }

    get category() {
        return this._category;
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
        let layerType = layer.layerType();

        for (let k of Object.keys(armnn.schema.Layer)) {
            if (layerType == armnn.schema.Layer[k]) 
                return layer.layer(new armnn.schema[k]);
        }
        return null;
    }

    static getBase(layer) {
        layer = armnn.Node.castLayer(layer);
        return (layer.base().base)? layer.base().base() : layer.base();
    }

    getDescriptor(layer) {
        if (layer == null)
            return null;

        return layer.descriptor();
    }

    getAttr(descriptor, key) {
        if (typeof descriptor[key] == "undefined")
            return "undefined";

        let values = descriptor[key]();
        if (Array.isArray(values)) {
            return values.join(", ");
        }
        else {
            return values;
        }
    }

    packAttr(layer, attr) {
        let descriptor = this.getDescriptor(layer);

        let key  = attr.src;
        let type = attr.src_type;

        if (typeof type != "undefined") {
            let value = this.getAttr(descriptor, key);
            if (typeof armnn.schema[type + "Name"] != "undefined")
                return armnn.schema[type + "Name"][value];
            else
                return value;
        }
        else if (Array.isArray(key)) {
            let values = [];
            for (let i = 0 ; i < key.length ; i++) {
                values.push(this.getAttr(descriptor, key[i]));
            }
            return values.join(", ");
        }
        else {
            return this.getAttr(descriptor, key);
        }
    }

    setAttribute(layer) {
        let layerType = layer.layerType();
        let layerName = armnn.schema.LayerName[layerType];

        let schema = this._metadata.getSchema(layerName);

        // ignore unknown layer
        if (!schema)
            return;

        let _layer = armnn.Node.castLayer(layer);

        if (typeof schema["bindings"] != "undefined") {
            for (let i = 0 ; i < schema.bindings.length ; i++) {
                let binding = schema.bindings[i];

                let value = _layer.base()[binding.src]();
                this._attributes.push(new armnn.Attribute(binding.name, binding.type, value));
            }
        }

        if (typeof schema["attributes"] != "undefined") {
            for (let i = 0 ; i < schema.attributes.length ; i++) {
                let attr = schema.attributes[i];

                let value = this.packAttr(_layer, attr);
                this._attributes.push(new armnn.Attribute(attr.name, attr.type, value));
            }
        }

        if (typeof schema["inputs"] != "undefined") {
            for (let i = 0 ; i < schema.inputs.length ; i++) {
                let input = schema.inputs[i];
                let value = _layer[input["src"]]();

                if (value) {
                    let args = [ new armnn.Argument('', null, value) ];
                    this._inputs.push(new armnn.Parameter(input["name"], '', args));
                }
            }
        }

        this._category = schema["category"];
    }

};

armnn.Attribute = class {

    constructor(name, type, value) {
        this._name = name;
        this._value = value;
        this._visible = true;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

armnn.Parameter = class {

    constructor(name, id, args) {
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

    static makeKey(layer_id, index) {
        return layer_id.toString() + "_" + index.toString();
    }
};

armnn.Argument = class {

    constructor(id, tensorInfo, initializer) {
        let info = initializer ? initializer.info() : tensorInfo;

        this._id = id;
        this._type = new armnn.TensorType(info);
        this._initializer = initializer ? new armnn.Tensor(info, initializer) : null;

        if (this._type.dataType.startsWith('q') && info) {
            this._scale = info.quantizationScale();
            this._zeroPoint = info.quantizationOffset();
        }
    }

    get id() {
        return this._id;
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
        this._kind = 'ConstTensor';

        let data = null;
        if (tensor.dataType() == armnn.schema.ConstTensorData.ByteData)
            data = tensor.data(new armnn.schema.ByteData);
        else if (tensor.dataType() == armnn.schema.ConstTensorData.ShortData)
            data = tensor.data(new armnn.schema.ShortData);
        else if (tensor.dataType() == armnn.schema.ConstTensorData.IntData)
            data = tensor.data(new armnn.schema.IntData);
        else if (tensor.dataType() == armnn.schema.ConstTensorData.LongData)
            data = tensor.data(new armnn.schema.LongData);

        this._data = data.dataLength() > 0 ? data.dataArray() : null;
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
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
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
        let size = shape[dimension];
        let results = [];
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

        const dataType = tensorInfo.dataType();
        switch (dataType) {
            case 0: this._dataType = 'float16'; break;
            case 1: this._dataType = 'float32'; break;
            case 2: this._dataType = 'quint8'; break;
            case 3: this._dataType = 'int32'; break;
            case 4: this._dataType = 'boolean'; break;
            case 5: this._dataType = 'qint16'; break;
            default: throw new armnn.Error("Unknown data type '" + dataType + "'.");
        }

        let dimensions = [];
        let dimensionsLength = tensorInfo.dimensionsLength();
        if (dimensionsLength > 0) {
            for (let i = 0; i < dimensionsLength; i++) {
                dimensions.push(tensorInfo.dimensions(i));
            }
        }
        this._shape = new armnn.TensorShape(dimensions);
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
        this._dimensions = dimensions;
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
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        const schema = this.getSchema(operator);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (let attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            let attributeSchema = attributeMap[name];
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
