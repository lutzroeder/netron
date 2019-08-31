/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var ncnn = ncnn || {};
var base = base || require('./base');

// https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
// https://github.com/Tencent/ncnn/wiki/operation-param-weight-table

ncnn.ModelFactory = class {

    match(context) {
        let identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.param') || identifier.endsWith('.cfg.ncnn')) {
            let text = context.text;
            text = text.substring(0, Math.min(text.length, 32));
            let signature = text.split('\n').shift().trim();
            if (signature === '7767517') {
                return true;
            }
        }
        if (identifier.endsWith('.param.bin')) {
            let buffer = context.buffer;
            if (buffer.length > 4) {
                let signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
                if (signature == 0x007685DD) {
                    return true;
                }
            }
        }
        if (identifier.endsWith('.bin') || identifier.endsWith('.weights.ncnn')) {
            let buffer = context.buffer;
            if (buffer.length > 4) {
                let signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
                if (signature === 0x00000000 || signature === 0x00000001 || 
                    signature === 0x01306B47 || signature === 0x000D4B38 || signature === 0x0002C056) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        return ncnn.Metadata.open(host).then((metadata) => {
            let identifier = context.identifier.toLowerCase();
            let param = (param, bin) => {
                try {
                    return new ncnn.Model(metadata, param, bin);
                }
                catch (error) {
                    let message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new ncnn.Error(message + " in '" + identifier + "'.");
                }
            };
            let bin = null;
            if (identifier.endsWith('.param') || identifier.endsWith('.cfg.ncnn')) {
                if (identifier.endsWith('.param')) {
                    bin = context.identifier.substring(0, context.identifier.length - 6) + '.bin';
                }
                else if (identifier.endsWith('.cfg.ncnn')) {
                    bin = context.identifier.substring(0, context.identifier.length - 9) + '.weights.ncnn';
                }
                return context.request(bin, null).then((bin) => {
                    return param(context.text, bin);
                }).catch(() => {
                    return param(context.text, null);
                });
            }
            else if (identifier.endsWith('.param.bin')) {
                bin = context.identifier.substring(0, context.identifier.length - 10) + '.bin';
                return context.request(bin, null).then((bin) => {
                    return param(context.buffer, bin);
                }).catch(() => {
                    return param(context.buffer, null);
                });
            }
            else if (identifier.endsWith('.bin') || identifier.endsWith('.weights.ncnn')) {
                let text = null;
                if (identifier.endsWith('bin')) {
                    text = context.identifier.substring(0, context.identifier.length - 4) + '.param';
                }
                else if (identifier.endsWith('.weights.ncnn')) {
                    text = context.identifier.substring(0, context.identifier.length - 13) + '.cfg.ncnn';
                }
                return context.request(text, 'utf-8').then((text) => {
                    return param(text, context.buffer);
                }).catch((error) => {
                    let message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new ncnn.Error(message + " in '" + identifier + "'.");
                });
            }
        });
    }
}

ncnn.Model = class {

    constructor(metadata, param, bin) {
        this._format = 'NCNN'
        this._graphs = [];
        this._graphs.push(new ncnn.Graph(metadata, param, bin));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
}

ncnn.Graph = class {

    constructor(metadata, param, bin) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        let blobReader = new ncnn.BlobReader(bin);

        let layers = (typeof param == 'string') ?
            this._param(metadata, param, bin) :
            this._param_bin(metadata, param, bin);
 
        for (let layer of layers) {
            if (layer.type == 'Input') {
                let dimensions = layer.attributes.map((a) => parseInt(a.value, 10));
                let shape = new ncnn.TensorShape(dimensions);
                let type = new ncnn.TensorType('float32', shape);
                this._inputs.push(new ncnn.Parameter(layer.name, true, layer.outputs.map((output) => new ncnn.Argument(output, type, null))));
            }
            else {
                this._nodes.push(new ncnn.Node(metadata, blobReader, layer));
            }
        }
    }

    _param(metadata, param) {
        let lines = param.split('\n');
        let signature = lines.shift();
        if (signature !== '7767517') {
            throw new ncnn.Error('Invalid signature.')
        }
        let header = lines.shift().split(' ');
        if (header.length !== 2) {
            throw new ncnn.Error('Invalid header count.');
        }

        let layers = [];
        let layer;
        while (lines.length > 0) {
            let line = lines.shift().trim();
            if (line.length > 0) {
                let columns = line.split(' ').filter((s) => s.length != 0);
                layer = {};
                layer.type = columns.shift();
                layer.name = columns.shift();
                let inputCount = parseInt(columns.shift(), 10);
                let outputCount = parseInt(columns.shift(), 10);
                layer.inputs = columns.splice(0, inputCount);
                layer.outputs = columns.splice(0, outputCount);
                layer.attr = {};
                layer.attributes = columns.map((attribute) => {
                    let list = attribute.split('=');
                    let key = list[0].trim();
                    let value = list[1].trim();
                    let keyInt = parseInt(key, 10);
                    if (key < 0) {
                        value = value.split(',').map((v) => v.trim());
                        value.shift();
                        key = (-(keyInt + 23300)).toString();
                    }
                    layer.attr[key] = value;
                    return { key: key, value: value };
                });
                layers.push(layer);
            }
        }
        return layers;
    }

    _param_bin(metadata, param) {
        let reader = new ncnn.BinaryParamReader(param);
        if (!reader.signature()) {
            throw new ncnn.Error('Invalid signature.')
        }
        let layerCount = reader.int32();
        /* var blobCount = */ reader.int32();
        let layers = [];
        for (let i = 0; i < layerCount; i++) {
            let layer = {};
            let typeIndex = reader.int32();
            let operator = metadata.getOperatorName(typeIndex);
            layer.type = operator || typeIndex.toString();
            layer.name = i.toString();
            layer.inputs = [];
            layer.outputs = [];
            layer.attr = {};
            layer.attributes = [];
            let inputCount = reader.int32();
            let outputCount = reader.int32();
            for (let j = 0; j < inputCount; j++) {
                layer.inputs.push(reader.int32().toString());
            }
            for (let k = 0; k < outputCount; k++) {
                layer.outputs.push(reader.int32().toString());
            }
            let id = reader.int32();
            while (id != -233) {
                let isArray = id <= -23300;
                if (isArray) {
                    id = -id - 23300;
                }
                if (isArray) {
                    let len = reader.int32();
                    let values = [];
                    for (let i = 0; i < len; i++) {
                        values.push(reader.int32());
                    }
                    layer.attributes.push({ key: id.toString(), value: values.toString() });
                    layer.attr[id.toString()] = values;
                }
                else {
                    let value = reader.int32();
                    layer.attributes.push({ key: id.toString(), value: value.toString() });
                    layer.attr[id.toString()] = value.toString();
                }
                id = reader.int32();
            }
            layers.push(layer);
        }
        return layers;
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
}

ncnn.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

ncnn.Argument = class {
    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

ncnn.Node = class {

    constructor(metadata, blobReader, layer) {
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._operator = layer.type;
        this._name = layer.name;

        let operator = metadata.getOperatorName(this._operator);
        if (operator) {
            this._operator = operator;
        }

        let schema = metadata.getSchema(this._operator);

        let attributeMetadata = {};
        if (schema && schema.attributes) {
            for (let i = 0; i < schema.attributes.length; i++) {
                let id = schema.attributes[i].id || i.toString();
                attributeMetadata[id] = schema.attributes[i];
            }
        }
        for (let attribute of layer.attributes) {
            let attributeSchema = attributeMetadata[attribute.key];
            this._attributes.push(new ncnn.Attribute(attributeSchema, attribute.key, attribute.value));
        }

        let inputs = layer.inputs;
        let inputIndex = 0;
        if (schema && schema.inputs) {
            for (let inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    let inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    let inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => {
                        return new ncnn.Argument(id, null, null);
                    });
                    this._inputs.push(new ncnn.Parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        else {
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => {
                let inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new ncnn.Parameter(inputName, true, [
                    new ncnn.Argument(input, null, null)
                ]);
            }));
        }

        let outputs = layer.outputs;
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (let outputDef of schema.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    let outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    let outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => {
                        return new ncnn.Argument(id, null, null)
                    });
                    this._outputs.push(new ncnn.Parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        else {
            this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
                let outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new ncnn.Parameter(outputName, true, [
                    new ncnn.Argument(output, null, null)
                ]);
            }));
        }

        let num_output;
        let weight_data_size;
        let channels;
        let scale_data_size;
        let bias_data_size;
        switch (this._operator) {
            case 'BatchNorm':
                channels = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'slope', [ channels ], 'float32');
                this._weight(blobReader, 'mean', [ channels ], 'float32');
                this._weight(blobReader, 'variance', [ channels ], 'float32');
                this._weight(blobReader, 'bias', [ channels ], 'float32');
                blobReader.next();
                break;
            case 'InnerProduct':
                num_output = parseInt(layer.attr['0'] || 0, 10);
                weight_data_size = parseInt(layer.attr['2'] || 0, 10);
                this._weight(blobReader, 'weight', [ num_output, weight_data_size / num_output ]);
                if (layer.attr['1'] == '1') {
                    this._weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                blobReader.next();
                break;
            case 'Bias':
                bias_data_size = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                blobReader.next();
                break;
            case 'Embed':
                num_output = parseInt(layer.attr['0'] || 0, 10);
                weight_data_size = parseInt(layer.attr['3'] || 0, 10);
                this._weight(blobReader, 'weight', [ weight_data_size ]);
                if (layer.attr['2'] == '1') {
                    this._weight(blobReader, 'bias', [ num_output], 'float32');
                }
                blobReader.next();
                break;
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise':
                num_output = parseInt(layer.attr['0'] || 0, 10);
                var kernel_w = parseInt(layer.attr['1'] || 0, 10);
                var kernel_h = parseInt(layer.attr['1'] || kernel_w, 10);
                weight_data_size = parseInt(layer.attr['6'] || 0, 10);
                this._weight(blobReader, 'weight', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h), kernel_w, kernel_h ]);
                if (layer.attr['5'] == '1') {
                    this._weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                blobReader.next();
                break;
            case 'Dequantize':
                if (layer.attr['1'] == '1') {
                    bias_data_size = parseInt(layer.attr['2'] || 0, 10);
                    this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                }
                blobReader.next();
                break;
            case 'Requantize':
                if (layer.attr['2'] == '1') {
                    bias_data_size = parseInt(layer.attr['3'] || 0, 10);
                    this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                }
                blobReader.next();
                break;
            case 'InstanceNorm':
                channels = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'gamma', [ channels ], 'float32');
                this._weight(blobReader, 'beta', [ channels ], 'float32');
                blobReader.next();
                break;
            case 'Scale':
                scale_data_size = parseInt(layer.attr['0'] || 0, 10);
                if (scale_data_size != -233) {
                    this._weight(blobReader, 'scale', [ scale_data_size], 'float32');
                    if (layer.attr['1'] == '1') {
                        this._weight(blobReader, 'bias', [ scale_data_size ], 'float32');
                    }
                    blobReader.next();
                }
                break;
            case 'Normalize':
                scale_data_size = parseInt(layer.attr['3'] || 0, 10);
                this._weight(blobReader, 'scale', [ scale_data_size ], 'float32');
                blobReader.next();
                break;
            case 'PReLU':
                var num_slope = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'slope', [ num_slope ], 'float32');
                blobReader.next();
                break;
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get category() {
        let schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        return '';
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    _weight(blobReader, name, dimensions, dataType) {
        dimensions = dimensions || null;
        let data = null;
        if (dimensions) {
            let size = 1;
            for (let dimension of dimensions) {
                size *= dimension;
            }
            if (!dataType) {
                dataType = blobReader.dataType;
            }
            data = blobReader.read(size, dataType);
        }
        else {
            dataType = dataType || '?';
            blobReader.dispose();
        }
        this._inputs.push(new ncnn.Parameter(name, true, [
            new ncnn.Argument('', null, new ncnn.Tensor(new ncnn.TensorType(dataType, new ncnn.TensorShape(dimensions)), data))
        ]));
    }
}

ncnn.Attribute = class {

    constructor(schema, key, value) {
        this._type = '';
        this._name = key;
        this._value = value;
        if (schema) {
            this._name = schema.name;
            if (schema.type) {
                this._type = schema.type;
            }
            switch (this._type) {
                case 'int32':
                    this._value = parseInt(this._value, 10);
                    break;
                case 'float32':
                    this._value = parseFloat(this._value);
                    break;
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default || (this._value && this._value.toString() == schema.default.toString())) {
                    this._visible = false;
                }
            }
        }
    }

    get type() {
        return this._type;
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
}

ncnn.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Weight';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
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
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'float16':
            case 'float32':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (context.shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        let size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType)
                {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
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

}

ncnn.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape()
    {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
}

ncnn.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

ncnn.Metadata = class {

    static open(host) {
        if (ncnn.Metadata._metadata) {
            return Promise.resolve(ncnn.Metadata._metadata);
        }
        return host.request(null, 'ncnn-metadata.json', 'utf-8').then((data) => {
            ncnn.Metadata._metadata = new ncnn.Metadata(data);
            return ncnn.Metadata._metadata;
        }).catch(() => {
            ncnn.Metadata._metadata = new ncnn.Metadata(null);
            return ncnn.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._operatorMap = {}; 
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                        if (Object.prototype.hasOwnProperty.call(item.schema, 'operator')) {
                            this._operatorMap[item.schema.operator.toString()] = item.name;
                        }
                    }
                }
            }
        }
    }

    getOperatorName(code) {
        return this._operatorMap[code] || null;
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            let schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (let attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

ncnn.BinaryParamReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._f32 = new Float32Array([ 0 ]);
        this._f8b = new Uint8Array(this._f32.buffer);
    }

    signature() {
        return this.int32() === 0x007685DD;
    }

    int32() {
        let i0 = this._buffer[this._position++];
        let i1 = this._buffer[this._position++];
        let i2 = this._buffer[this._position++];
        let i3 = this._buffer[this._position++];
        return i0 | i1 << 8 | i2 << 16 | i3 << 24;
    }

    float32() {
        this._f8b[0] = this._buffer[this._position++];
        this._f8b[1] = this._buffer[this._position++];
        this._f8b[2] = this._buffer[this._position++];
        this._f8b[3] = this._buffer[this._position++];
        return this._f32[0];
    }
}

ncnn.BlobReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
    }

    get dataType() {
        if (!this._dataType && this._buffer && this._position + 4 < this._buffer.length) {
            let f0 = this._buffer[this._position++];
            let f1 = this._buffer[this._position++];
            let f2 = this._buffer[this._position++];
            let f3 = this._buffer[this._position++];
            let type = f0 | f1 << 8 | f2 << 16 | f3 << 24;
            switch (type) {
                case 0x00000000:
                    this._dataType = 'float32';
                    break;
                case 0x01306B47:
                    this._dataType = 'float16';
                    break;
                case 0x000D4B38:
                    this._dataType = 'int8';
                    break;
                case 0x00000001:
                    this._dataType = 'qint8';
                    break;
                case 0x0002C056: // size * sizeof(float) - raw data with extra scaling
                default:
                    throw new ncnn.Error("Unknown weight type '" + type + "'.");
            }
        }
        return this._dataType || '?';
    }

    read(size, dataType) {
        if (this._buffer) {
            dataType = dataType || this.dataType;
            let position = this._position
            switch (dataType) {
                case 'float32': 
                    size *= 4;
                    this._position += size;
                    return this._buffer.subarray(position, this._position);
                case 'float16': 
                    size *= 2;
                    this._position += size;
                    return this._buffer.subarray(position, this._position);
                case 'int8': 
                    this._position += size;
                    return this._buffer.subarray(position, this._position);
                case 'qint8':
                    this._position += size + 1024;
                    return null;
                default:
                    this.dispose();
                    break;
            }
        }
        return null;
    }

    next() {
        this._dataType = null;
    }

    dispose() {
        this._dataType = null;
        this._buffer = null;
    }
}

ncnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ncnn model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = ncnn.ModelFactory;
}