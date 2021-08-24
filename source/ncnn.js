/* jshint esversion: 6 */

var ncnn = ncnn || {};
var base = base || require('./base');

// https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
// https://github.com/Tencent/ncnn/wiki/operation-param-weight-table

ncnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.param') || identifier.endsWith('.cfg.ncnn')) {
            const reader = base.TextReader.open(context.stream.peek(), 2048);
            const signature = reader.read();
            if (signature !== undefined) {
                if (signature.trim() === '7767517') {
                    return 'ncnn.model';
                }
                const header = signature.trim().split(' ');
                if (header.length === 2 && header.every((value) => value >>> 0 === parseFloat(value))) {
                    return 'ncnn.model';
                }
            }
        }
        if (identifier.endsWith('.param.bin')) {
            const stream = context.stream;
            if (stream.length > 4) {
                const buffer = stream.peek(4);
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature == 0x007685DD) {
                    return 'ncnn.model.bin';
                }
            }
        }
        if (identifier.endsWith('.bin') || identifier.endsWith('.weights.ncnn')) {
            if (identifier == 'snapshot_blob.bin' || identifier === 'v8_context_snapshot.bin') {
                return undefined;
            }
            const stream = context.stream;
            if (stream.length > 4) {
                const buffer = stream.peek(4);
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature === 0x00000000 || signature === 0x00000001 ||
                    signature === 0x01306B47 || signature === 0x000D4B38 || signature === 0x0002C056) {
                    return 'ncnn.weights';
                }
            }
        }
        return undefined;
    }

    open(context, match) {
        return ncnn.Metadata.open(context).then((metadata) => {
            const identifier = context.identifier.toLowerCase();
            const openBinary = (param, bin) => {
                const reader = new ncnn.BinaryParamReader(metadata, param);
                return new ncnn.Model(metadata, reader, bin);
            };
            const openText = (param, bin) => {
                const reader = new ncnn.TextParamReader(param);
                return new ncnn.Model(metadata, reader, bin);
            };
            let bin = null;
            switch (match) {
                case 'ncnn.model': {
                    if (identifier.endsWith('.param')) {
                        bin = context.identifier.substring(0, context.identifier.length - 6) + '.bin';
                    }
                    else if (identifier.endsWith('.cfg.ncnn')) {
                        bin = context.identifier.substring(0, context.identifier.length - 9) + '.weights.ncnn';
                    }
                    return context.request(bin, null).then((stream) => {
                        const buffer = stream.read();
                        return openText(context.stream.peek(), buffer);
                    }).catch(() => {
                        return openText(context.stream.peek(), null);
                    });
                }
                case 'ncnn.model.bin': {
                    bin = context.identifier.substring(0, context.identifier.length - 10) + '.bin';
                    return context.request(bin, null).then((stream) => {
                        const buffer = stream.read();
                        return openBinary(context.stream.peek(), buffer);
                    }).catch(() => {
                        return openBinary(context.stream.peek(), null);
                    });
                }
                case 'ncnn.weights': {
                    let text = null;
                    if (identifier.endsWith('bin')) {
                        text = context.identifier.substring(0, context.identifier.length - 4) + '.param';
                    }
                    else if (identifier.endsWith('.weights.ncnn')) {
                        text = context.identifier.substring(0, context.identifier.length - 13) + '.cfg.ncnn';
                    }
                    return context.request(text, null).then((stream) => {
                        const buffer = stream.peek();
                        return openText(buffer, context.stream.peek());
                    }).catch(() => {
                        return context.request(text + '.bin', null).then((stream) => {
                            const buffer = stream.peek();
                            return openBinary(buffer, context.stream.peek());
                        });
                    });
                }
            }
        });
    }
};

ncnn.Model = class {

    constructor(metadata, param, bin) {
        this._graphs = [
            new ncnn.Graph(metadata, param, bin)
        ];
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
        const layers = param.layers;
        const args = new Map();
        const arg = (name, type) => {
            if (!args.has(name)) {
                args.set(name, new ncnn.Argument(name, type, null));
            }
            return args.get(name);
        };
        for (const layer of layers) {
            const attributes = layer.attributes;
            for (const pair of attributes) {
                const key = pair[0];
                const list = pair[1];
                if (key === '30' && Array.isArray(list)) {
                    const value = list.map((item) => parseInt(item, 10));
                    for (const output of layer.outputs || []) {
                        if (value.length > 0 && value[0] <= value.length - 1) {
                            const shape = new Array(value.shift());
                            for (let i = 0; i < shape.length; i++) {
                                shape[i] = value.shift();
                            }
                            const type = new ncnn.TensorType('?', new ncnn.TensorShape(shape));
                            arg(output, type, null);
                        }
                        attributes.delete(key);
                    }
                }
            }
        }
        for (const layer of layers) {
            if (layer.type == 'Input') {
                const values = Array.from(layer.attributes.values());
                const dimensions = values.map((value) => !isNaN(parseInt(value, 10)) ? parseInt(value, 10) : value);
                const shape = new ncnn.TensorShape(dimensions);
                const type = new ncnn.TensorType('float32', shape);
                const input = new ncnn.Parameter(layer.name, true, layer.outputs.map((output) => new ncnn.Argument(output, type, null)));
                this._inputs.push(input);
            }
            else {
                const node = new ncnn.Node(metadata, blobReader, layer, arg);
                this._nodes.push(node);
            }
        }
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

    constructor(metadata, blobReader, layer, arg) {
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        this._name = layer.name || '';
        const type = layer.type;
        this._type = metadata.type(type) || metadata.operator(type) || { name: type };
        const attributeMetadata = this._type && this._type.attributes ? this._type.attributes : [];
        const attributes = layer.attributes;
        const inputs = layer.inputs || [];
        let inputIndex = 0;
        if (this._type && this._type.inputs) {
            for (const inputDef of this._type.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => arg(id));
                    this._inputs.push(new ncnn.Parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
            const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
            return new ncnn.Parameter(inputName, true, [ arg(input) ]);
        }));

        const outputs = layer.outputs || [];
        let outputIndex = 0;
        if (this._type && this._type.outputs) {
            for (const outputDef of this._type.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => arg(id));
                    this._outputs.push(new ncnn.Parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
            const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
            return new ncnn.Parameter(outputName, true, [ arg(output) ]);
        }));
        switch (this._type.name) {
            case 'BatchNorm': {
                const channels = parseInt(attributes.get('0') || 0, 10);
                this._weight(blobReader, 'slope', [ channels ], 'float32');
                this._weight(blobReader, 'mean', [ channels ], 'float32');
                this._weight(blobReader, 'variance', [ channels ], 'float32');
                this._weight(blobReader, 'bias', [ channels ], 'float32');
                break;
            }
            case 'InnerProduct': {
                const activation_names = [ '', 'ReLU', 'Leaky ReLU', 'Clip', 'Sigmoid', 'Mish' ];
                const activation_type = parseInt(attributes.get('9') || 0, 10);
                if (activation_type > 0 && activation_type < activation_names.length) {
                    const layer = {
                        type: activation_names[activation_type],
                        attributes: new Map()
                    };
                    this._chain.push(new ncnn.Node(metadata, blobReader, layer, arg));
                }
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('2') || 0, 10);
                this._weight(blobReader, 'weight', [ num_output, weight_data_size / num_output ]);
                if (parseInt(attributes.get('1') || 0, 10) === 1) {
                    this._weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.delete('2');
                break;
            }
            case 'Bias': {
                const bias_data_size = parseInt(attributes.get('0') || 0, 10);
                this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                break;
            }
            case 'Embed': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('3') || 0, 10);
                this._weight(blobReader, 'weight', [ weight_data_size ]);
                if (parseInt(attributes.get('2') || 0, 10) === 1) {
                    this._weight(blobReader, 'bias', [ num_output], 'float32');
                }
                attributes.get('3');
                break;
            }
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                const activation_names = [ '', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish' ];
                const activation_type = parseInt(attributes.get('9') || 0, 10);
                if (activation_type > 0 && activation_type < activation_names.length) {
                    const layer = {
                        type: activation_names[activation_type],
                        attributes: new Map()
                    };
                    this._chain.push(new ncnn.Node(metadata, blobReader, layer, arg));
                }
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const kernel_w = parseInt(attributes.get('1') || 0, 10);
                const kernel_h = parseInt(attributes.get('11') || kernel_w, 10);
                const weight_data_size = parseInt(attributes.get('6') || 0, 10);
                this._weight(blobReader, 'weight', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h), kernel_w, kernel_h ]);
                if (parseInt(attributes.get('5') || 0, 10) === 1) {
                    this._weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.delete('6');
                break;
            }
            case 'Dequantize': {
                if (parseInt(attributes.get('1') || 0, 10) === 1) {
                    const bias_data_size = parseInt(attributes.get('2') || 0, 10);
                    this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                }
                break;
            }
            case 'Requantize': {
                if (parseInt(attributes.get('2') || 0, 10) === 1) {
                    const bias_data_size = parseInt(attributes.get('3') || 0, 10);
                    this._weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                }
                break;
            }
            case 'InstanceNorm': {
                const affine = parseInt(attributes.get('2') || 1, 10);
                if (affine === 1) {
                    const channels = parseInt(attributes.get('0') || 0, 10);
                    this._weight(blobReader, 'gamma', [ channels ], 'float32');
                    this._weight(blobReader, 'beta', [ channels ], 'float32');
                }
                break;
            }
            case 'Scale': {
                const scale_data_size = parseInt(attributes.get('0') || 0, 10);
                if (scale_data_size != -233) {
                    this._weight(blobReader, 'scale', [ scale_data_size], 'float32');
                    if (attributes.get('1') == '1') {
                        this._weight(blobReader, 'bias', [ scale_data_size ], 'float32');
                    }
                }
                break;
            }
            case 'Normalize': {
                const scale_data_size = parseInt(attributes.get('3') || 0, 10);
                this._weight(blobReader, 'scale', [ scale_data_size ], 'float32');
                break;
            }
            case 'PReLU': {
                const num_slope = parseInt(attributes.get('0') || 0, 10);
                this._weight(blobReader, 'slope', [ num_slope ], 'float32');
                break;
            }
            case 'Padding': {
                const per_channel_pad_data_size = parseInt(attributes.get('6') || 0, 10);
                this._weight(blobReader, 'per_channel_pad_data', [ per_channel_pad_data_size ], 'float32');
                break;
            }
            case 'MemoryData': {
                const w = parseInt(attributes.get('0') || 0, 10);
                const h = parseInt(attributes.get('1') || 0, 10);
                const c = parseInt(attributes.get('2') || 0, 10);
                if (c != 0) {
                    this._weight(blobReader, 'data', [ c, h, w ], 'float32');
                }
                else if (h != 0) {
                    this._weight(blobReader, 'data', [ h, w ], 'float32');
                }
                else if (w != 0) {
                    this._weight(blobReader, 'data', [ w ], 'float32');
                }
                else {
                    this._weight(blobReader, 'data', [ 1 ], 'float32');
                }
                break;
            }
            case 'GroupNorm': {
                const affine = parseInt(attributes.get('3') || 1, 10);
                if (affine === 1) {
                    const channels = parseInt(attributes.get('1') || 0, 10);
                    this._weight(blobReader, 'gamma', [ channels ], 'float32');
                    this._weight(blobReader, 'beta', [ channels ], 'float32');
                }
                break;
            }
            case 'LayerNorm': {
                const channels = parseInt(attributes.get('0') || 0, 10);
                this._weight(blobReader, 'gamma', [ channels ], 'float32');
                this._weight(blobReader, 'beta', [ channels ], 'float32');
                break;
            }
            case 'RNN': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('1') || 0, 10);
                const direction = parseInt(attributes.get('2') || 0, 10);
                const num_directions = direction == 2 ? 2 : 1;
                this._weight(blobReader, 'weight_xc', [ num_directions, num_output, weight_data_size / num_directions / num_output ]);
                this._weight(blobReader, 'bias_c', [ num_directions, num_output ]);
                this._weight(blobReader, 'weight_hc', [ num_directions, num_output, num_output ]);
                attributes.delete('1');
                break;
            }
            case 'LSTM': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('1') || 0, 10);
                const direction = parseInt(attributes.get('2') || 0, 10);
                const num_directions = direction == 2 ? 2 : 1;
                this._weight(blobReader, 'weight_xc', [ num_directions, 4, num_output, weight_data_size / num_directions / num_output / 4 ]);
                this._weight(blobReader, 'bias_c', [ num_directions, 4, num_output ]);
                this._weight(blobReader, 'weight_hc', [ num_directions, 4, num_output, num_output ]);
                attributes.delete('1');
                break;
            }
            case 'GRU': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('1') || 0, 10);
                const direction = parseInt(attributes.get('2') || 0, 10);
                const num_directions = direction == 2 ? 2 : 1;
                this._weight(blobReader, 'weight_xc', [ num_directions, 3, num_output, weight_data_size / num_directions / num_output / 3 ]);
                this._weight(blobReader, 'bias_c', [ num_directions, 4, num_output ]);
                this._weight(blobReader, 'weight_hc', [ num_directions, 3, num_output, num_output ]);
                attributes.delete('1');
                break;
            }
        }

        this._attributes = Array.from(attributes).map((attribute) => {
            const key = attribute[0];
            const value = attribute[1];
            const metadata = attributeMetadata[key];
            return new ncnn.Attribute(metadata, key, value);
        });
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
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

    get chain() {
        return this._chain;
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

    constructor(metadata, key, value) {
        this._type = '';
        this._name = key;
        this._value = value;
        if (metadata) {
            this._name = metadata.name;
            if (metadata.type) {
                this._type = metadata.type;
            }
            switch (this._type) {
                case 'int32': {
                    this._value = parseInt(this._value, 10);
                    break;
                }
                case 'float32': {
                    this._value = parseFloat(this._value);
                    break;
                }
                case 'float32[]': {
                    this._value = this._value.map((v) => parseFloat(v));
                    break;
                }
                default: {
                    if (this._type) {
                        this._value = ncnn.Utility.value(this._value, this._type);
                    }
                    break;
                }
            }
            if (Object.prototype.hasOwnProperty.call(metadata, 'visible') && !metadata.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (this._value == metadata.default || (this._value && this._value.toString() == metadata.default.toString())) {
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

    static open(context) {
        if (ncnn.Metadata._metadata) {
            return Promise.resolve(ncnn.Metadata._metadata);
        }
        return context.request('ncnn-metadata.json', 'utf-8', null).then((data) => {
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
        this._attributes = new Map();
        if (data) {
            const items = JSON.parse(data);
            for (const item of items) {
                if (item.name) {
                    this._map.set(item.name, item);
                    if (Object.prototype.hasOwnProperty.call(item, 'operator')) {
                        this._operatorMap.set(item.operator, item.name);
                    }
                }
            }
        }
    }

    operator(code) {
        return this._operatorMap.get(code);
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributes.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributes.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }
};

ncnn.Utility = class {

    static value(value, type) {
        ncnn.Utility._enum = ncnn.Utility._enum || new Map([
            [ 'BinaryOpType', [ 'Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'Pow', 'RSub', 'RDiv' ] ],
            [ 'EltwiseType', [ 'Prod', 'Sum', 'Max' ] ],
            [ 'PoolingType', [ 'Max', 'Average' ] ],
            [ 'InterpResizeType', [ '', 'Nearest', 'Bilinear', 'Bicubic' ] ],
            [ 'PermuteOrderType', [ 'WHC', 'HWC', 'WCH', 'CWH', 'HCW', 'CHW'] ]
        ]);
        if (this._enum.has(type) && typeof value === 'string') {
            const index = parseInt(value, 10);
            const list = this._enum.get(type);
            if (Number.isInteger(index) && index < list.length) {
                return list[index];
            }
        }
        return value;
    }
};

ncnn.TextParamReader = class {

    constructor(buffer) {
        const reader = base.TextReader.open(buffer);
        const lines = [];
        for (;;) {
            const line = reader.read();
            if (line === undefined) {
                break;
            }
            lines.push(line.trim());
        }
        const signature = lines.shift();
        const header = (signature !== '7767517' ? signature : lines.shift()).split(' ');
        if (header.length !== 2 || !header.every((value) => value >>> 0 === parseFloat(value))) {
            throw new ncnn.Error('Invalid header.');
        }
        const layers = [];
        while (lines.length > 0) {
            const line = lines.shift();
            if (line.length > 0) {
                const columns = line.split(' ').filter((s) => s.length != 0);
                const layer = {};
                layer.type = columns.shift();
                layer.name = columns.shift();
                const inputCount = parseInt(columns.shift(), 10);
                const outputCount = parseInt(columns.shift(), 10);
                layer.inputs = columns.splice(0, inputCount);
                layer.outputs = columns.splice(0, outputCount);
                layer.attributes = new Map();
                const attributes = layer.attributes;
                let index = 0;
                for (const column of columns) {
                    const parts = column.split('=');
                    if (parts.length > 2) {
                        throw new ncnn.Attribute("Invalid attribute '" + column + "'.");
                    }
                    let key = (parts.length === 2) ? parts[0].trim() : index.toString();
                    let value = (parts.length === 2) ? parts[1].trim() : parts[0].trim();
                    const keyInt = parseInt(key, 10);
                    if (keyInt < 0) {
                        value = value.split(',').map((v) => v.trim());
                        value.shift();
                        key = (-(keyInt + 23300)).toString();
                    }
                    attributes.set(key, value);
                    index++;
                }
                layers.push(layer);
            }
        }
        this._layers = layers;
    }

    get layers() {
        return this._layers;
    }
};

ncnn.BinaryParamReader = class {

    constructor(metadata, buffer) {
        const reader = new ncnn.BinaryReader(buffer);
        if (reader.int32() !== 0x007685DD) {
            throw new ncnn.Error('Invalid signature.');
        }
        const layerCount = reader.int32();
        /* const blobCount = */ reader.int32();
        this._layers = [];
        for (let i = 0; i < layerCount; i++) {
            const typeIndex = reader.int32();
            const operator = metadata.operator(typeIndex);
            const layer = {
                type: operator || typeIndex.toString(),
                name: i.toString(),
                attributes: new Map(),
                inputs: [],
                outputs: []
            };
            const inputCount = reader.int32();
            const outputCount = reader.int32();
            for (let j = 0; j < inputCount; j++) {
                layer.inputs.push(reader.int32().toString());
            }
            for (let j = 0; j < outputCount; j++) {
                layer.outputs.push(reader.int32().toString());
            }
            const attributes = layer.attributes;
            let id = reader.int32();
            while (id != -233) {
                const isArray = id <= -23300;
                if (isArray) {
                    id = -id - 23300;
                }
                const key = id.toString();
                if (isArray) {
                    const length = reader.int32();
                    const values = [];
                    for (let i = 0; i < length; i++) {
                        values.push(reader.int32());
                    }
                    attributes.set(key, values);
                }
                else {
                    const value = reader.int32();
                    attributes.set(key, value);
                }
                id = reader.int32();
            }
            this._layers.push(layer);
        }
    }

    get layers() {
        return this._layers;
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

ncnn.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    skip(size) {
        this._position += size;
        if (this._position > this._buffer.length) {
            throw new ncnn.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
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