
var ncnn = ncnn || {};
var text = require('./text');
var base = require('./base');

// https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
// https://github.com/Tencent/ncnn/wiki/operation-param-weight-table
// https://github.com/Tencent/ncnn/wiki/operators

ncnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.param.bin') || identifier.endsWith('.ncnnmodel')) {
            const stream = context.stream;
            if (stream.length > 4) {
                const buffer = stream.peek(4);
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature == 0x007685DD) {
                    return 'ncnn.model.bin';
                }
            }
        }
        if (identifier.endsWith('.param') || identifier.endsWith('.cfg.ncnn')) {
            try {
                const reader = text.Reader.open(context.stream, 2048);
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
            } catch (err) {
                // continue regardless of error
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

    async open(context, target) {
        const openBinary = (param, bin) => {
            const reader = new ncnn.BinaryParamReader(param);
            return new ncnn.Model(metadata, reader, bin);
        };
        const openText = (param, bin) => {
            const reader = new ncnn.TextParamReader(param);
            return new ncnn.Model(metadata, reader, bin);
        };
        const metadata = await context.metadata('ncnn-metadata.json');
        const identifier = context.identifier.toLowerCase();
        let bin = null;
        switch (target) {
            case 'ncnn.model': {
                if (identifier.endsWith('.param')) {
                    bin = context.identifier.substring(0, context.identifier.length - 6) + '.bin';
                } else if (identifier.endsWith('.cfg.ncnn')) {
                    bin = context.identifier.substring(0, context.identifier.length - 9) + '.weights.ncnn';
                }
                try {
                    const stream = await context.request(bin, null);
                    const buffer = stream.read();
                    return openText(context.stream.peek(), buffer);
                } catch (error) {
                    return openText(context.stream.peek(), null);
                }
            }
            case 'ncnn.model.bin': {
                bin = context.identifier.substring(0, context.identifier.length - 10) + '.bin';
                try {
                    const stream = await context.request(bin, null);
                    const buffer = stream.read();
                    return openBinary(context.stream.peek(), buffer);
                } catch (error) {
                    return openBinary(context.stream.peek(), null);
                }
            }
            case 'ncnn.weights': {
                let content = null;
                if (identifier.endsWith('bin')) {
                    content = context.identifier.substring(0, context.identifier.length - 4) + '.param';
                } else if (identifier.endsWith('.weights.ncnn')) {
                    content = context.identifier.substring(0, context.identifier.length - 13) + '.cfg.ncnn';
                }
                try {
                    const stream = await context.request(content, null);
                    const buffer = stream.peek();
                    return openText(buffer, context.stream.peek());
                } catch (error) {
                    const stream = await context.request(content + '.bin', null);
                    const buffer = stream.peek();
                    return openBinary(buffer, context.stream.peek());
                }
            }
            default: {
                throw new ncnn.Error("Unsupported ncnn format '" + target + "'.");
            }
        }
    }
};

ncnn.Model = class {

    constructor(metadata, param, bin) {
        this._format = 'ncnn';
        this._graphs = [ new ncnn.Graph(metadata, param, bin) ];
    }

    get format() {
        return this._format;
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
        const arg = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new ncnn.Value(name, type, tensor);
            }
            if (!args.has(name)) {
                args.set(name, new ncnn.Value(name, type || null, tensor || null));
            } else if (tensor || (type && !type.equals(args.get(name).type))) {
                throw new ncnn.Error("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        for (const layer of layers) {
            const attributes = layer.attributes;
            for (const entry of attributes) {
                const key = entry[0];
                const list = entry[1];
                if (key === '30' && Array.isArray(list)) {
                    const value = list.map((item) => parseInt(item, 10));
                    for (const output of layer.outputs || []) {
                        if (value.length > 0 && value[0] <= value.length - 1) {
                            const shape = new Array(value.shift());
                            for (let i = 0; i < shape.length; i++) {
                                shape[i] = value.shift();
                            }
                            const type = new ncnn.TensorType('float32', new ncnn.TensorShape(shape));
                            arg(output, type);
                        }
                        attributes.delete(key);
                    }
                }
            }
        }
        for (const layer of layers) {
            if (layer.type === 'Input' || layer.type === 16) {
                const values = Array.from(layer.attributes.values());
                const dimensions = values.map((value) => !isNaN(parseInt(value, 10)) ? parseInt(value, 10) : value);
                const shape = new ncnn.TensorShape(dimensions);
                const type = new ncnn.TensorType('float32', shape);
                const input = new ncnn.Argument(layer.name, layer.outputs.map((output) => arg(output, type)));
                this._inputs.push(input);
            } else {
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

ncnn.Argument = class {

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

ncnn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new ncnn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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
        this._type = metadata.type(type);
        const attributeMetadata = this._type && this._type.attributes ? this._type.attributes : [];
        const attributes = layer.attributes;
        const inputs = layer.inputs || [];
        let inputIndex = 0;
        if (this._type && this._type.inputs) {
            for (const inputDef of this._type.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => arg(id));
                    this._inputs.push(new ncnn.Argument(inputDef.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
            const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
            return new ncnn.Argument(inputName, [ arg(input) ]);
        }));

        const outputs = layer.outputs || [];
        let outputIndex = 0;
        if (this._type && this._type.outputs) {
            for (const outputDef of this._type.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => arg(id));
                    this._outputs.push(new ncnn.Argument(outputDef.name, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
            const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
            return new ncnn.Argument(outputName, [ arg(output) ]);
        }));
        const weight = (blobReader, name, dimensions, dataType) => {
            const blob = blobReader.read(dimensions, dataType);
            dataType = blob ? (blob.dataType || '?') : (dataType || '?');
            const data = blob ? blob.data : null;
            const type = new ncnn.TensorType(dataType, new ncnn.TensorShape(dimensions));
            const tensor = new ncnn.Tensor(type, data);
            this._inputs.push(new ncnn.Argument(name, [ arg('', null, tensor) ]));
        };
        switch (this._type.name) {
            case 'BatchNorm': {
                const channels = parseInt(attributes.get('0') || 0, 10);
                weight(blobReader, 'slope', [ channels ], 'float32');
                weight(blobReader, 'mean', [ channels ], 'float32');
                weight(blobReader, 'variance', [ channels ], 'float32');
                weight(blobReader, 'bias', [ channels ], 'float32');
                break;
            }
            case 'InnerProduct': {
                const activation_names = [ '', 'ReLU', 'Leaky ReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish' ];
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
                weight(blobReader, 'weight', [ num_output, weight_data_size / num_output ]);
                if (parseInt(attributes.get('1') || 0, 10) === 1) {
                    weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.delete('2');
                break;
            }
            case 'Bias': {
                const bias_data_size = parseInt(attributes.get('0') || 0, 10);
                weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                break;
            }
            case 'Embed': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('3') || 0, 10);
                weight(blobReader, 'weight', [ weight_data_size / num_output, num_output ]);
                if (parseInt(attributes.get('2') || 0, 10) === 1) {
                    weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.get('3');
                break;
            }
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                const activation_names = [ '', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish' ];
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
                weight(blobReader, 'weight', [ num_output, weight_data_size / (num_output * kernel_w * kernel_h), kernel_h, kernel_w ]);
                if (parseInt(attributes.get('5') || 0, 10) === 1) {
                    weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.delete('6');
                break;
            }
            case 'Convolution1D':
            case 'ConvolutionDepthWise1D': {
                const activation_names = [ '', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish' ];
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
                const weight_data_size = parseInt(attributes.get('6') || 0, 10);
                weight(blobReader, 'weight', [ num_output, weight_data_size / (num_output * kernel_w), kernel_w ]);
                if (parseInt(attributes.get('5') || 0, 10) === 1) {
                    weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.delete('6');
                break;
            }
            case 'Convolution3D':
            case 'ConvolutionDepthWise3D': {
                const activation_names = [ '', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish' ];
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
                const kernel_d = parseInt(attributes.get('21') || kernel_w, 10);
                const weight_data_size = parseInt(attributes.get('6') || 0, 10);
                weight(blobReader, 'weight', [ num_output, weight_data_size / (num_output * kernel_w * kernel_h * kernel_d), kernel_d, kernel_h, kernel_w ]);
                if (parseInt(attributes.get('5') || 0, 10) === 1) {
                    weight(blobReader, 'bias', [ num_output ], 'float32');
                }
                attributes.delete('6');
                break;
            }
            case 'Quantize': {
                const scale_data_size = parseInt(attributes.get('0') || 1, 10);
                weight(blobReader, 'scale', [ scale_data_size ], 'float32');
                break;
            }
            case 'Dequantize': {
                const scale_data_size = parseInt(attributes.get('0') || 1, 10);
                const bias_data_size = parseInt(attributes.get('1') || 0, 10);
                weight(blobReader, 'scale', [ scale_data_size ], 'float32');
                weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                break;
            }
            case 'Requantize': {
                const scale_in_data_size = parseInt(attributes.get('0') || 1, 10);
                const scale_out_data_size = parseInt(attributes.get('1') || 1, 10);
                const bias_data_size = parseInt(attributes.get('2') || 0, 10);
                weight(blobReader, 'scale_in', [ scale_in_data_size ], 'float32');
                weight(blobReader, 'scale_out', [ scale_out_data_size ], 'float32');
                weight(blobReader, 'bias', [ bias_data_size ], 'float32');
                break;
            }
            case 'InstanceNorm': {
                const affine = parseInt(attributes.get('2') || 1, 10);
                if (affine === 1) {
                    const channels = parseInt(attributes.get('0') || 0, 10);
                    weight(blobReader, 'gamma', [ channels ], 'float32');
                    weight(blobReader, 'beta', [ channels ], 'float32');
                }
                break;
            }
            case 'Scale': {
                const scale_data_size = parseInt(attributes.get('0') || 0, 10);
                if (scale_data_size != -233) {
                    weight(blobReader, 'scale', [ scale_data_size], 'float32');
                    if (attributes.get('1') == '1') {
                        weight(blobReader, 'bias', [ scale_data_size ], 'float32');
                    }
                }
                break;
            }
            case 'Normalize': {
                const scale_data_size = parseInt(attributes.get('3') || 0, 10);
                weight(blobReader, 'scale', [ scale_data_size ], 'float32');
                break;
            }
            case 'PReLU': {
                const num_slope = parseInt(attributes.get('0') || 0, 10);
                weight(blobReader, 'slope', [ num_slope ], 'float32');
                break;
            }
            case 'Padding': {
                const per_channel_pad_data_size = parseInt(attributes.get('6') || 0, 10);
                weight(blobReader, 'per_channel_pad_data', [ per_channel_pad_data_size ], 'float32');
                break;
            }
            case 'MemoryData': {
                const w = parseInt(attributes.get('0') || 0, 10);
                const h = parseInt(attributes.get('1') || 0, 10);
                const d = parseInt(attributes.get('11') || 0, 10);
                const c = parseInt(attributes.get('2') || 0, 10);
                if (d != 0) {
                    weight(blobReader, 'data', [ c, d, h, w ], 'float32');
                } else if (c != 0) {
                    weight(blobReader, 'data', [ c, h, w ], 'float32');
                } else if (h != 0) {
                    weight(blobReader, 'data', [ h, w ], 'float32');
                } else if (w != 0) {
                    weight(blobReader, 'data', [ w ], 'float32');
                } else {
                    weight(blobReader, 'data', [ 1 ], 'float32');
                }
                break;
            }
            case 'GroupNorm': {
                const affine = parseInt(attributes.get('3') || 1, 10);
                if (affine === 1) {
                    const channels = parseInt(attributes.get('1') || 0, 10);
                    weight(blobReader, 'gamma', [ channels ], 'float32');
                    weight(blobReader, 'beta', [ channels ], 'float32');
                }
                break;
            }
            case 'LayerNorm': {
                const channels = parseInt(attributes.get('0') || 0, 10);
                weight(blobReader, 'gamma', [ channels ], 'float32');
                weight(blobReader, 'beta', [ channels ], 'float32');
                break;
            }
            case 'RNN': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('1') || 0, 10);
                const direction = parseInt(attributes.get('2') || 0, 10);
                const num_directions = direction == 2 ? 2 : 1;
                weight(blobReader, 'weight_xc', [ num_directions, num_output, weight_data_size / num_directions / num_output ]);
                weight(blobReader, 'bias_c', [ num_directions, num_output ]);
                weight(blobReader, 'weight_hc', [ num_directions, num_output, num_output ]);
                attributes.delete('1');
                break;
            }
            case 'LSTM': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('1') || 0, 10);
                const direction = parseInt(attributes.get('2') || 0, 10);
                const num_directions = direction == 2 ? 2 : 1;
                weight(blobReader, 'weight_xc', [ num_directions, 4, num_output, weight_data_size / num_directions / num_output / 4 ]);
                weight(blobReader, 'bias_c', [ num_directions, 4, num_output ]);
                weight(blobReader, 'weight_hc', [ num_directions, 4, num_output, num_output ]);
                attributes.delete('1');
                break;
            }
            case 'GRU': {
                const num_output = parseInt(attributes.get('0') || 0, 10);
                const weight_data_size = parseInt(attributes.get('1') || 0, 10);
                const direction = parseInt(attributes.get('2') || 0, 10);
                const num_directions = direction == 2 ? 2 : 1;
                weight(blobReader, 'weight_xc', [ num_directions, 3, num_output, weight_data_size / num_directions / num_output / 3 ]);
                weight(blobReader, 'bias_c', [ num_directions, 4, num_output ]);
                weight(blobReader, 'weight_hc', [ num_directions, 3, num_output, num_output ]);
                attributes.delete('1');
                break;
            }
            case 'MultiHeadAttention': {
                const embed_dim = parseInt(attributes.get('0') || 0, 10);
                // const num_head = parseInt(attributes.get('1') || 0, 10);
                // const weight_data_size = parseInt(attributes.get('2') || 0, 10);
                weight(blobReader, 'weight_q', [ embed_dim, embed_dim ]);
                weight(blobReader, 'bias_q', [ embed_dim ], 'float32');
                weight(blobReader, 'weight_k', [ embed_dim, embed_dim ]);
                weight(blobReader, 'bias_k', [ embed_dim ], 'float32');
                weight(blobReader, 'weight_v', [ embed_dim, embed_dim ]);
                weight(blobReader, 'bias_v', [ embed_dim ], 'float32');
                weight(blobReader, 'weight_out', [ embed_dim, embed_dim ]);
                weight(blobReader, 'bias_out', [ embed_dim ], 'float32');
                attributes.delete('2');
                break;
            }
            default: {
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
            if (metadata && metadata.visible === false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
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

    get category() {
        return 'Weights';
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._data;
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

    equals(obj) {
        return obj && this._dataType === obj.dataType && this._shape && this._shape.equals(obj.shape);
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

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) &&
            Array.isArray(this._dimensions) && this._dimensions.length === obj.dimensions.length
            && obj.dimensions.every((value, index) => this._dimensions[index] === value);
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

ncnn.Utility = class {

    static value(value, type) {
        ncnn.Utility._enum = ncnn.Utility._enum || new Map([
            [ 'BinaryOpType', [ 'Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'Pow', 'RSub', 'RDiv' ] ],
            [ 'CastOpType', [ 'Auto', 'Float32', 'Float16', 'Int8', 'BFloat16' ] ],
            [ 'EltwiseType', [ 'Prod', 'Sum', 'Max' ] ],
            [ 'PaddingType', [ 'Constant', 'Replicate', 'Reflect' ] ],
            [ 'PoolingType', [ 'Max', 'Average' ] ],
            [ 'InterpResizeType', [ '', 'Nearest', 'Bilinear', 'Bicubic' ] ],
            [ 'PermuteOrderType', [ 'WH WHC WHDC', 'HW HWC HWDC', 'WCH WDHC', 'CWH DWHC', 'HCW HDWC', 'CHW DHWC', 'WHCD', 'HWCD', 'WCHD', 'CWHD', 'HCWD', 'CHWD', 'WDCH', 'DWCH', 'WCDH', 'CWDH', 'DCWH', 'CDWH', 'HDCW', 'DHCW', 'HCDW', 'CHDW', 'DCHW', 'CDHW' ] ],
            [ 'ReductionOpType', [ 'Sum', 'ASum', 'SumSq', 'Mean', 'Max', 'Min', 'Prod', 'L1', 'L2', 'LogSum', 'LogSumExp' ] ],
            [ 'UnaryOpType', [ 'Abs', 'Neg', 'Floor', 'Ceil', 'Square', 'Sqrt', 'Rsq', 'Exp', 'Log', 'Sin', 'Cos', 'Tan', 'ASin', 'ACos', 'ATan', 'Reciprocal', 'Tanh' ] ]
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
        const reader = text.Reader.open(buffer);
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

    constructor(buffer) {
        const reader = new base.BinaryReader(buffer);
        if (reader.int32() !== 0x007685DD) {
            throw new ncnn.Error('Invalid signature.');
        }
        const layerCount = reader.int32();
        /* const blobCount = */ reader.int32();
        this._layers = [];
        for (let i = 0; i < layerCount; i++) {
            const layer = {
                type: reader.int32(),
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
                } else {
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
                            throw new ncnn.Error("Unsupported weight type '" + type + "'.");
                    }
                } else {
                    this._buffer = null;
                }
            }
            let data = null;
            let size = 1;
            if (shape) {
                for (const dimension of shape) {
                    size *= dimension;
                }
            } else {
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
                            throw new ncnn.Error("Unsupported weight type '" + dataType + "'.");
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
