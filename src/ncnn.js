/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var ncnn = ncnn || {};
var base = base || require('./base');

// https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
// https://github.com/Tencent/ncnn/wiki/operation-param-weight-table

ncnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.param') || identifier.endsWith('.cfg.ncnn')) {
            let text = context.text;
            text = text.substring(0, Math.min(text.length, 32));
            const signature = text.split('\n').shift().trim();
            if (signature === '7767517') {
                return true;
            }
        }
        if (identifier.endsWith('.param.bin')) {
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
                if (signature == 0x007685DD) {
                    return true;
                }
            }
        }
        if (identifier.endsWith('.bin') || identifier.endsWith('.weights.ncnn')) {
            if (identifier == 'snapshot_blob.bin' || identifier === 'v8_context_snapshot.bin') {
                return false;
            }
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
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
            const identifier = context.identifier.toLowerCase();
            const param = (param, bin) => {
                try {
                    return new ncnn.Model(metadata, param, bin);
                }
                catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new ncnn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
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
                    const message = error && error.message ? error.message : error.toString();
                    throw new ncnn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                });
            }
        });
    }
};

ncnn.Model = class {

    constructor(metadata, param, bin) {
        this._graphs = [];
        this._graphs.push(new ncnn.Graph(metadata, param, bin));
    }

    get format() {
        return 'ncnn';
    }

    get graphs() {
        return this._graphs;
    }
};

ncnn.Graph = class {

    constructor(metadata, param, bin) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const blobReader = new ncnn.BlobReader(bin);

        const layers = (typeof param == 'string') ?
            this._param(metadata, param, bin) :
            this._param_bin(metadata, param, bin);

        for (const layer of layers) {
            if (layer.type == 'Input') {
                const dimensions = layer.attributes.map((a) => !isNaN(parseInt(a.value, 10)) ? parseInt(a.value, 10) : a.value);
                const shape = new ncnn.TensorShape(dimensions);
                const type = new ncnn.TensorType('float32', shape);
                this._inputs.push(new ncnn.Parameter(layer.name, true, layer.outputs.map((output) => new ncnn.Argument(output, type, null))));
            }
            else {
                this._nodes.push(new ncnn.Node(metadata, blobReader, layer));
            }
        }
    }

    _param(metadata, param) {
        const lines = param.split(/\r?\n/);
        const signature = lines.shift();
        if (signature !== '7767517') {
            throw new ncnn.Error('Invalid signature.');
        }
        const header = lines.shift().split(' ');
        if (header.length !== 2) {
            throw new ncnn.Error('Invalid header count.');
        }

        const layers = [];
        let layer;
        while (lines.length > 0) {
            const line = lines.shift().trim();
            if (line.length > 0) {
                const columns = line.split(' ').filter((s) => s.length != 0);
                layer = {};
                layer.type = columns.shift();
                layer.name = columns.shift();
                const inputCount = parseInt(columns.shift(), 10);
                const outputCount = parseInt(columns.shift(), 10);
                layer.inputs = columns.splice(0, inputCount);
                layer.outputs = columns.splice(0, outputCount);
                layer.attr = {};
                layer.attributes = [];
                for (const column of columns) {
                    const parts = column.split('=');
                    if (parts.length === 2) {
                        let key = parts[0].trim();
                        let value = parts[1].trim();
                        const keyInt = parseInt(key, 10);
                        if (keyInt < 0) {
                            value = value.split(',').map((v) => v.trim());
                            value.shift();
                            key = (-(keyInt + 23300)).toString();
                        }
                        layer.attr[key] = value;
                        layer.attributes.push({ key: key, value: value });
                    }
                }
                layers.push(layer);
            }
        }
        return layers;
    }

    _param_bin(metadata, param) {
        const reader = new ncnn.BinaryReader(param);
        if (reader.int32() !== 0x007685DD) {
            throw new ncnn.Error('Invalid signature.');
        }
        const layerCount = reader.int32();
        /* const blobCount = */ reader.int32();
        const layers = [];
        for (let i = 0; i < layerCount; i++) {
            const layer = {};
            const typeIndex = reader.int32();
            const operator = metadata.operator(typeIndex);
            layer.type = operator || typeIndex.toString();
            layer.name = i.toString();
            layer.inputs = [];
            layer.outputs = [];
            layer.attr = {};
            layer.attributes = [];
            const inputCount = reader.int32();
            const outputCount = reader.int32();
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
                    const len = reader.int32();
                    const values = [];
                    for (let i = 0; i < len; i++) {
                        values.push(reader.int32());
                    }
                    layer.attributes.push({ key: id.toString(), value: values.toString() });
                    layer.attr[id.toString()] = values;
                }
                else {
                    const value = reader.int32();
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
};

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

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new ncnn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
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

        const operator = metadata.operator(this._operator);
        if (operator) {
            this._operator = operator;
        }

        const schema = metadata.type(this._operator);

        const attributeMetadata = {};
        if (schema && schema.attributes) {
            for (let i = 0; i < schema.attributes.length; i++) {
                const id = schema.attributes[i].id || i.toString();
                attributeMetadata[id] = schema.attributes[i];
            }
        }
        for (const attribute of layer.attributes) {
            const attributeSchema = attributeMetadata[attribute.key];
            this._attributes.push(new ncnn.Attribute(attributeSchema, attribute.key, attribute.value));
        }

        const inputs = layer.inputs;
        let inputIndex = 0;
        if (schema && schema.inputs) {
            for (const inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => {
                        return new ncnn.Argument(id, null, null);
                    });
                    this._inputs.push(new ncnn.Parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        else {
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new ncnn.Parameter(inputName, true, [
                    new ncnn.Argument(input, null, null)
                ]);
            }));
        }

        const outputs = layer.outputs;
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (const outputDef of schema.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => {
                        return new ncnn.Argument(id, null, null);
                    });
                    this._outputs.push(new ncnn.Parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        else {
            this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
                const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
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
            case 'BatchNorm': {
                channels = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'slope', [ channels ], 'float32');
                this._weight(blobReader, 'mean', [ channels ], 'float32');
                this._weight(blobReader, 'variance', [ channels ], 'float32');
                this._weight(blobReader, 'bias', [ channels ], 'float32');
                break;
            }
            case 'InnerProduct': {
                num_output = parseInt(layer.attr['0'] || 0, 10);
                weight_data_size = parseInt(layer.attr['2'] || 0, 10);
                this._weight(blobReader, 'weight', [ num_output, weight_data_size / num_output ]);
                if (layer.attr['1'] == '1') {
                    this._weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                break;
            }
            case 'Bias': {
                bias_data_size = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                break;
            }
            case 'Embed': {
                num_output = parseInt(layer.attr['0'] || 0, 10);
                weight_data_size = parseInt(layer.attr['3'] || 0, 10);
                this._weight(blobReader, 'weight', [ weight_data_size ]);
                if (layer.attr['2'] == '1') {
                    this._weight(blobReader, 'bias', [ num_output], 'float32');
                }
                break;
            }
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                num_output = parseInt(layer.attr['0'] || 0, 10);
                const kernel_w = parseInt(layer.attr['1'] || 0, 10);
                const kernel_h = parseInt(layer.attr['11'] || kernel_w, 10);
                weight_data_size = parseInt(layer.attr['6'] || 0, 10);
                this._weight(blobReader, 'weight', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h), kernel_w, kernel_h ]);
                if (layer.attr['5'] == '1') {
                    this._weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                break;
            }
            case 'Dequantize': {
                if (layer.attr['1'] == '1') {
                    bias_data_size = parseInt(layer.attr['2'] || 0, 10);
                    this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                }
                break;
            }
            case 'Requantize': {
                if (layer.attr['2'] == '1') {
                    bias_data_size = parseInt(layer.attr['3'] || 0, 10);
                    this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                }
                break;
            }
            case 'InstanceNorm': {
                channels = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'gamma', [ channels ], 'float32');
                this._weight(blobReader, 'beta', [ channels ], 'float32');
                break;
            }
            case 'Scale': {
                scale_data_size = parseInt(layer.attr['0'] || 0, 10);
                if (scale_data_size != -233) {
                    this._weight(blobReader, 'scale', [ scale_data_size], 'float32');
                    if (layer.attr['1'] == '1') {
                        this._weight(blobReader, 'bias', [ scale_data_size ], 'float32');
                    }
                }
                break;
            }
            case 'Normalize': {
                scale_data_size = parseInt(layer.attr['3'] || 0, 10);
                this._weight(blobReader, 'scale', [ scale_data_size ], 'float32');
                break;
            }
            case 'PReLU': {
                const num_slope = parseInt(layer.attr['0'] || 0, 10);
                this._weight(blobReader, 'slope', [ num_slope ], 'float32');
                break;
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata.type(this._operator);
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
        const blob = blobReader.read(dimensions, dataType);
        dataType = blob ? (blob.dataType || '?') : (dataType || '?');
        const data = blob ? blob.data : null;
        this._inputs.push(new ncnn.Parameter(name, true, [
            new ncnn.Argument('', null, new ncnn.Tensor(new ncnn.TensorType(dataType, new ncnn.TensorShape(dimensions)), data))
        ]));
    }
};

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
                case 'float32[]':
                    this._value = this._value.map((v) => parseFloat(v));
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
};

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
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
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
};

ncnn.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

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
        this._operatorMap = new Map();
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                        if (Object.prototype.hasOwnProperty.call(item.schema, 'operator')) {
                            this._operatorMap.set(item.schema.operator, item.name);
                        }
                    }
                }
            }
        }
    }

    operator(code) {
        return this._operatorMap.get(code);
    }

    type(operator) {
        return this._map.get(operator);
    }

    attribute(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(operator + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

ncnn.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    int32() {
        const position = this._position;
        this._position += 4;
        if (this._position > this._buffer.length) {
            throw new ncnn.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        return this._dataView.getInt32(position, true);
    }
};

ncnn.BlobReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
    }

    read(shape, dataType) {
        if (this._buffer) {
            if (!dataType) {
                if (this._buffer && this._position + 4 < this._buffer.length) {
                    const f0 = this._buffer[this._position++];
                    const f1 = this._buffer[this._position++];
                    const f2 = this._buffer[this._position++];
                    const f3 = this._buffer[this._position++];
                    const type = f0 | f1 << 8 | f2 << 16 | f3 << 24;
                    switch (type) {
                        case 0x00000000:
                            dataType = 'float32';
                            break;
                        case 0x01306B47:
                            dataType = 'float16';
                            break;
                        case 0x000D4B38:
                            dataType = 'int8';
                            break;
                        case 0x00000001:
                            dataType = 'qint8';
                            break;
                        case 0x0002C056: // size * sizeof(float) - raw data with extra scaling
                        default:
                            throw new ncnn.Error("Unknown weight type '" + type + "'.");
                    }
                }
                else {
                    this._buffer = null;
                }
            }
            let data = null;
            let size = 1;
            if (shape) {
                for (const dimension of shape) {
                    size *= dimension;
                }
            }
            else {
                this._buffer = null;
            }
            if (this._buffer) {
                if (dataType) {
                    const position = this._position;
                    switch (dataType) {
                        case 'float32':
                            size *= 4;
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'float16':
                            size *= 2;
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'int8':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'qint8':
                            this._position += size + 1024;
                            data = null;
                            break;
                        default:
                            throw new ncnn.Error("Unknown weight type '" + dataType + "'.");
                    }
                }
            }
            return { dataType: dataType, data: data };
        }
        return null;
    }
};

ncnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ncnn model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = ncnn.ModelFactory;
}