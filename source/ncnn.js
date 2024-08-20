
import * as base from './base.js';

const ncnn = {};

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
                if (signature === 0x007685dd) {
                    context.type = 'ncnn.model.bin';
                }
            }
        } else if (identifier.endsWith('.param') || identifier.endsWith('.cfg.ncnn')) {
            const reader = context.read('text', 0x10000);
            const type = identifier.endsWith('.pnnx.param') ? 'pnnx.model' : 'ncnn.model';
            if (reader) {
                try {
                    const signature = reader.read('\n');
                    if (signature !== undefined) {
                        if (signature.trim() === '7767517') {
                            context.type = type;
                            return;
                        }
                        const header = signature.trim().split(' ');
                        if (header.length === 2 && header.every((value) => value >>> 0 === parseFloat(value))) {
                            context.type = type;
                        }
                    }
                } catch {
                    // continue regardless of error
                }
            }
        } else if (identifier.endsWith('.ncnn.bin')) {
            context.type = 'ncnn.weights';
        } else if (identifier.endsWith('.pnnx.bin')) {
            const entries = context.peek('zip');
            if (entries && entries.size > 0) {
                context.type = 'pnnx.weights';
                context.target = entries;
            }
        } else if (identifier.endsWith('.bin') || identifier.endsWith('.weights.ncnn')) {
            const stream = context.stream;
            const length = stream.length;
            if (length > 4) {
                const buffer = stream.peek(4);
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature === 0x00000000) {
                    const size = Math.min(stream.length, 1024) & 0xFFFC;
                    const buffer = stream.peek(size);
                    const length = size >> 2;
                    const array = new Float32Array(buffer.buffer, buffer.byteOffset, length);
                    const values = Array.from(array).slice(1);
                    if (values.every((value) => !Number.isNaN(value) && Number.isFinite(value) && value > -20.0 && value < 20.0)) {
                        context.type = 'ncnn.weights';
                    }
                } else {
                    const buffer = stream.peek(Math.min(0x20000, length));
                    for (let i = 0; i < buffer.length - 4; i++) {
                        const signature = (buffer[i] | buffer[i + 1] << 8 | buffer[i + 2] << 16 | buffer [i + 3] << 24) >>> 0;
                        if (signature === 0xdeadbeef) { // Core ML
                            return;
                        }
                        if (signature === 0x01306b47 || signature === 0x000d4b38 || signature === 0x0002c056) {
                            context.type = 'ncnn.weights';
                            return;
                        }
                    }
                }
            }
        }
    }

    filter(context, type) {
        return (context.type !== 'ncnn.model' && context.type !== 'ncnn.model.bin') || type !== 'ncnn.weights';
    }

    async open(context) {
        const metadata = await context.metadata('ncnn-metadata.json');
        const identifier = context.identifier.toLowerCase();
        const format = context.type.split('.').shift();
        switch (context.type) {
            case 'pnnx.model':
            case 'ncnn.model': {
                let file = null;
                if (identifier.endsWith('.param')) {
                    file = context.identifier.replace(/\.param$/, '.bin');
                } else if (identifier.endsWith('.cfg.ncnn')) {
                    file = context.identifier.replace(/\.cfg\.ncnn$/, '.weights.ncnn');
                }
                let buffer = null;
                try {
                    const content = await context.fetch(file);
                    buffer = content.stream.peek();
                } catch {
                    // continue regardless of error
                }
                const param = context.read('text');
                const reader = new ncnn.TextParamReader(param);
                return new ncnn.Model(metadata, format, reader, buffer);
            }
            case 'ncnn.model.bin': {
                const bin = `${context.identifier.substring(0, context.identifier.length - 10)}.bin`;
                let buffer = null;
                try {
                    const content = await context.fetch(bin);
                    buffer = content.stream.peek();
                } catch {
                    // continue regardless of error
                }
                const param = context.stream.peek();
                const reader = new ncnn.BinaryParamReader(param);
                return new ncnn.Model(metadata, format, reader, buffer);
            }
            case 'pnnx.weights':
            case 'ncnn.weights': {
                let file = null;
                if (identifier.endsWith('.bin')) {
                    file = context.identifier.replace(/\.bin$/, '.param');
                } else if (identifier.endsWith('.weights.ncnn')) {
                    file = context.identifier.replace(/\.weights\.ncnn$/, '.cfg.ncnn');
                }
                let reader = null;
                try {
                    const content = await context.fetch(file);
                    const param = content.read('text');
                    reader = new ncnn.TextParamReader(param);
                } catch {
                    const content = await context.fetch(`${file}.bin`);
                    const param = content.stream.peek();
                    reader = new ncnn.BinaryParamReader(param);
                }
                const buffer = context.stream.peek();
                return new ncnn.Model(metadata, format, reader, buffer);
            }
            default: {
                throw new ncnn.Error(`Unsupported ncnn format '${context.type}'.`);
            }
        }
    }
};

ncnn.Model = class {

    constructor(metadata, format, param, bin) {
        this.format = format === 'pnnx' ? 'PNNX' : 'ncnn';
        this.graphs = [new ncnn.Graph(metadata, format, param, bin)];
    }
};

ncnn.Graph = class {

    constructor(metadata, format, param, bin) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const blobs = new ncnn.BlobReader(bin);
        const layers = param.layers;
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new ncnn.Value(name, type, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new ncnn.Value(name, type || null, tensor || null));
            } else if (tensor || (type && !type.equals(values.get(name).type))) {
                throw new ncnn.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        for (const layer of layers) {
            const params = layer.params;
            if (params && params.size > 0) {
                for (const [key, list] of params) {
                    if (key === '30' && Array.isArray(list)) {
                        const value = list.map((item) => parseInt(item, 10));
                        for (const output of layer.outputs || []) {
                            if (value.length > 0 && value[0] <= value.length - 1) {
                                const shape = new Array(value.shift());
                                for (let i = 0; i < shape.length; i++) {
                                    shape[i] = value.shift();
                                }
                                const type = new ncnn.TensorType('float32', new ncnn.TensorShape(shape));
                                values.map(output, type);
                            }
                            params.delete(key);
                        }
                    }
                }
            }
        }
        for (const layer of layers) {
            if (layer.type === 'Input' || layer.type === 16) {
                const dimensions = Array.from(layer.params.values()).map((value) => isNaN(parseInt(value, 10)) ? value : parseInt(value, 10));
                const shape = new ncnn.TensorShape(dimensions);
                const type = new ncnn.TensorType('float32', shape);
                const input = new ncnn.Argument(layer.name, layer.outputs.map((output) => values.map(output, type)));
                this.inputs.push(input);
            } else {
                const node = new ncnn.Node(metadata, format, blobs, layer, values);
                this.nodes.push(node);
            }
        }
    }
};

ncnn.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type;
        this.visible = visible !== false;
    }
};

ncnn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new ncnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

ncnn.Node = class {

    constructor(metadata, format, blobs, layer, values) {
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        this.name = layer.name || '';
        this.type = { ...metadata.type(layer.type) };
        delete this.type.identifier;
        const params = layer.params;
        const inputs = layer.inputs || [];
        let inputIndex = 0;
        if (this.type && Array.isArray(this.type.inputs)) {
            for (const input of this.type.inputs) {
                if (inputIndex < inputs.length || input.optional === false) {
                    const count = (input.type === 'Tensor[]') ? (inputs.length - inputIndex) : 1;
                    const list = inputs.slice(inputIndex, inputIndex + count).filter((id) => id !== '' || input.option !== 'optional').map((id) => values.map(id));
                    const argument = new ncnn.Argument(input.name, list);
                    this.inputs.push(argument);
                    inputIndex += count;
                }
            }
        }
        this.inputs.push(...inputs.slice(inputIndex).map((input, index) => {
            const name = ((inputIndex + index) === 0) ? 'input' : (inputIndex + index).toString();
            return new ncnn.Argument(name, [values.map(input)]);
        }));
        const outputs = layer.outputs || [];
        let outputIndex = 0;
        if (this.type && Array.isArray(this.type.outputs)) {
            for (const output of this.type.outputs) {
                if (outputIndex < outputs.length || output.option !== 'optional') {
                    const count = (output.type === 'Tensor[]') ? (outputs.length - outputIndex) : 1;
                    const list = outputs.slice(outputIndex, outputIndex + count).map((id) => values.map(id));
                    const argument = new ncnn.Argument(output.name, list);
                    this.outputs.push(argument);
                    outputIndex += count;
                }
            }
        }
        this.outputs.push(...outputs.slice(outputIndex).map((output, index) => {
            const name = ((outputIndex + index) === 0) ? 'output' : (outputIndex + index).toString();
            return new ncnn.Argument(name, [values.map(output)]);
        }));
        blobs.weight = (name, dimensions, dataType) => {
            const blob = blobs.read(dimensions, dataType);
            dataType = blob ? (blob.dataType || '?') : (dataType || '?');
            const data = blob ? blob.data : null;
            const type = new ncnn.TensorType(dataType, new ncnn.TensorShape(dimensions));
            const tensor = new ncnn.Tensor(type, data);
            const argument = new ncnn.Argument(name, [values.map('', null, tensor)]);
            this.inputs.push(argument);
        };
        switch (this.type.name) {
            case 'BatchNorm': {
                const channels = parseInt(params.get('0') || 0, 10);
                blobs.weight('slope', [channels], 'float32');
                blobs.weight('mean', [channels], 'float32');
                blobs.weight('variance', [channels], 'float32');
                blobs.weight('bias', [channels], 'float32');
                break;
            }
            case 'InnerProduct': {
                const num_output = parseInt(params.get('0') || 0, 10);
                const bias_term = parseInt(params.get('1') || 0, 10);
                const weight_data_size = parseInt(params.get('2') || 0, 10);
                const int8_scale_term = parseInt(params.get('8') || 0, 10);
                const activation_type = parseInt(params.get('9') || 0, 10);
                blobs.weight('weight', [num_output, weight_data_size / num_output]);
                if (bias_term) {
                    blobs.weight('bias', [num_output], 'float32');
                }
                if (int8_scale_term) {
                    blobs.weight('weight_scales', [num_output], 'float32');
                    blobs.weight('bottom_scales', [1], 'float32');
                }
                const activation_names = ['', 'ReLU', 'Leaky ReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish'];
                if (activation_type > 0 && activation_type < activation_names.length) {
                    const layer = { type: activation_names[activation_type] };
                    this.chain.push(new ncnn.Node(metadata, format, blobs, layer, values));
                }
                params.delete('2');
                break;
            }
            case 'Bias': {
                const bias_data_size = parseInt(params.get('0') || 0, 10);
                blobs.weight('bias', [bias_data_size], 'float32');
                break;
            }
            case 'Embed': {
                const num_output = parseInt(params.get('0') || 0, 10);
                const weight_data_size = parseInt(params.get('3') || 0, 10);
                blobs.weight('weight', [weight_data_size / num_output, num_output]);
                if (parseInt(params.get('2') || 0, 10) === 1) {
                    blobs.weight('bias', [num_output], 'float32');
                }
                params.get('3');
                break;
            }
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                const num_output = parseInt(params.get('0') || 0, 10);
                const kernel_w = parseInt(params.get('1') || 0, 10);
                const kernel_h = parseInt(params.get('11') || kernel_w, 10);
                const weight_data_size = parseInt(params.get('6') || 0, 10);
                blobs.weight('weight', [num_output, weight_data_size / (num_output * kernel_w * kernel_h), kernel_h, kernel_w]);
                if (parseInt(params.get('5') || 0, 10) === 1) {
                    blobs.weight('bias', [num_output], 'float32');
                }
                const int8_scale_term = parseInt(params.get('8') || 0, 10);
                if (this.type.name === 'Convolution') {
                    if (int8_scale_term) {
                        blobs.weight('weight_scales', [num_output], 'float32');
                        blobs.weight('bottom_scales', [1], 'float32');
                    }
                    if (int8_scale_term > 100) {
                        blobs.weight('top_scales', [1], 'float32');
                    }
                } else if (this.type.name === 'ConvolutionDepthWise') {
                    const group =  parseInt(params.get('7') || 1, 10);
                    if (int8_scale_term === 1 || int8_scale_term === 101) {
                        blobs.weight('weight_scales', [group], 'float32');
                        blobs.weight('bottom_scales', [1], 'float32');
                    } else if (int8_scale_term === 2 || int8_scale_term === 102) {
                        blobs.weight('weight_scales', [1], 'float32');
                        blobs.weight('bottom_scales', [1], 'float32');
                    }
                    if (int8_scale_term > 100) {
                        blobs.weight('top_scales', [1], 'float32');
                    }
                }
                params.delete('6');
                const activation_names = ['', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish'];
                const activation_type = parseInt(params.get('9') || 0, 10);
                if (activation_type > 0 && activation_type < activation_names.length) {
                    const layer = { type: activation_names[activation_type] };
                    this.chain.push(new ncnn.Node(metadata, format, blobs, layer, values));
                }
                break;
            }
            case 'Convolution1D':
            case 'ConvolutionDepthWise1D': {
                const activation_names = ['', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish'];
                const activation_type = parseInt(params.get('9') || 0, 10);
                if (activation_type > 0 && activation_type < activation_names.length) {
                    const layer = { type: activation_names[activation_type] };
                    const node = new ncnn.Node(metadata, format, blobs, layer, values);
                    this.chain.push(node);
                }
                const num_output = parseInt(params.get('0') || 0, 10);
                const kernel_w = parseInt(params.get('1') || 0, 10);
                const weight_data_size = parseInt(params.get('6') || 0, 10);
                blobs.weight('weight', [num_output, weight_data_size / (num_output * kernel_w), kernel_w]);
                if (parseInt(params.get('5') || 0, 10) === 1) {
                    blobs.weight('bias', [num_output], 'float32');
                }
                params.delete('6');
                break;
            }
            case 'Convolution3D':
            case 'ConvolutionDepthWise3D': {
                const activation_names = ['', 'ReLU', 'LeakyReLU', 'Clip', 'Sigmoid', 'Mish', 'HardSwish'];
                const activation_type = parseInt(params.get('9') || 0, 10);
                if (activation_type > 0 && activation_type < activation_names.length) {
                    const layer = { type: activation_names[activation_type] };
                    this.chain.push(new ncnn.Node(metadata, format, blobs, layer, values));
                }
                const num_output = parseInt(params.get('0') || 0, 10);
                const kernel_w = parseInt(params.get('1') || 0, 10);
                const kernel_h = parseInt(params.get('11') || kernel_w, 10);
                const kernel_d = parseInt(params.get('21') || kernel_w, 10);
                const weight_data_size = parseInt(params.get('6') || 0, 10);
                blobs.weight('weight', [num_output, weight_data_size / (num_output * kernel_w * kernel_h * kernel_d), kernel_d, kernel_h, kernel_w]);
                if (parseInt(params.get('5') || 0, 10) === 1) {
                    blobs.weight('bias', [num_output], 'float32');
                }
                params.delete('6');
                break;
            }
            case 'Quantize': {
                const scale_data_size = parseInt(params.get('0') || 1, 10);
                blobs.weight('scale', [scale_data_size], 'float32');
                break;
            }
            case 'Dequantize': {
                const scale_data_size = parseInt(params.get('0') || 1, 10);
                const bias_data_size = parseInt(params.get('1') || 0, 10);
                blobs.weight('scale', [scale_data_size], 'float32');
                blobs.weight('bias', [bias_data_size], 'float32');
                break;
            }
            case 'Requantize': {
                const scale_in_data_size = parseInt(params.get('0') || 1, 10);
                const scale_out_data_size = parseInt(params.get('1') || 1, 10);
                const bias_data_size = parseInt(params.get('2') || 0, 10);
                blobs.weight('scale_in', [scale_in_data_size], 'float32');
                blobs.weight('scale_out', [scale_out_data_size], 'float32');
                blobs.weight('bias', [bias_data_size], 'float32');
                break;
            }
            case 'InstanceNorm': {
                const affine = parseInt(params.get('2') || 1, 10);
                if (affine === 1) {
                    const channels = parseInt(params.get('0') || 0, 10);
                    blobs.weight('gamma', [channels], 'float32');
                    blobs.weight('beta', [channels], 'float32');
                }
                break;
            }
            case 'Scale': {
                const scale_data_size = parseInt(params.get('0') || 0, 10);
                if (scale_data_size !== -233) {
                    blobs.weight('scale', [scale_data_size], 'float32');
                    if (params.get('1') === '1') {
                        blobs.weight('bias', [scale_data_size], 'float32');
                    }
                }
                break;
            }
            case 'Normalize': {
                const scale_data_size = parseInt(params.get('3') || 0, 10);
                blobs.weight('scale', [scale_data_size], 'float32');
                break;
            }
            case 'PReLU': {
                const num_slope = parseInt(params.get('0') || 0, 10);
                blobs.weight('slope', [num_slope], 'float32');
                break;
            }
            case 'Padding': {
                const per_channel_pad_data_size = parseInt(params.get('6') || 0, 10);
                blobs.weight('per_channel_pad_data', [per_channel_pad_data_size], 'float32');
                break;
            }
            case 'MemoryData': {
                const w = parseInt(params.get('0') || 0, 10);
                const h = parseInt(params.get('1') || 0, 10);
                const d = parseInt(params.get('11') || 0, 10);
                const c = parseInt(params.get('2') || 0, 10);
                /* eslint-disable no-negated-condition */
                if (d !== 0) {
                    blobs.weight('data', [c, d, h, w], 'float32');
                } else if (c !== 0) {
                    blobs.weight('data', [c, h, w], 'float32');
                } else if (h !== 0) {
                    blobs.weight('data', [h, w], 'float32');
                } else if (w !== 0) {
                    blobs.weight('data', [w], 'float32');
                } else {
                    blobs.weight('data', [1], 'float32');
                }
                /* eslint-enable no-negated-condition */
                break;
            }
            case 'GroupNorm': {
                const affine = parseInt(params.get('3') || 1, 10);
                if (affine === 1) {
                    const channels = parseInt(params.get('1') || 0, 10);
                    blobs.weight('gamma', [channels], 'float32');
                    blobs.weight('beta', [channels], 'float32');
                }
                break;
            }
            case 'LayerNorm': {
                const channels = parseInt(params.get('0') || 0, 10);
                blobs.weight('gamma', [channels], 'float32');
                blobs.weight('beta', [channels], 'float32');
                break;
            }
            case 'RNN': {
                const num_output = parseInt(params.get('0') || 0, 10);
                const weight_data_size = parseInt(params.get('1') || 0, 10);
                const direction = parseInt(params.get('2') || 0, 10);
                const num_directions = direction === 2 ? 2 : 1;
                blobs.weight('weight_xc', [num_directions, num_output, weight_data_size / num_directions / num_output]);
                blobs.weight('bias_c', [num_directions, num_output]);
                blobs.weight('weight_hc', [num_directions, num_output, num_output]);
                params.delete('1');
                break;
            }
            case 'LSTM': {
                const num_output = parseInt(params.get('0') || 0, 10);
                const weight_data_size = parseInt(params.get('1') || 0, 10);
                const direction = parseInt(params.get('2') || 0, 10);
                const num_directions = direction === 2 ? 2 : 1;
                blobs.weight('weight_xc', [num_directions, 4, num_output, weight_data_size / num_directions / num_output / 4]);
                blobs.weight('bias_c', [num_directions, 4, num_output]);
                blobs.weight('weight_hc', [num_directions, 4, num_output, num_output]);
                params.delete('1');
                break;
            }
            case 'GRU': {
                const num_output = parseInt(params.get('0') || 0, 10);
                const weight_data_size = parseInt(params.get('1') || 0, 10);
                const direction = parseInt(params.get('2') || 0, 10);
                const num_directions = direction === 2 ? 2 : 1;
                blobs.weight('weight_xc', [num_directions, 3, num_output, weight_data_size / num_directions / num_output / 3]);
                blobs.weight('bias_c', [num_directions, 4, num_output]);
                blobs.weight('weight_hc', [num_directions, 3, num_output, num_output]);
                params.delete('1');
                break;
            }
            case 'MultiHeadAttention': {
                const embed_dim = parseInt(params.get('0') || 0, 10);
                // const num_head = parseInt(params.get('1') || 0, 10);
                // const weight_data_size = parseInt(params.get('2') || 0, 10);
                blobs.weight('weight_q', [embed_dim, embed_dim]);
                blobs.weight('bias_q', [embed_dim], 'float32');
                blobs.weight('weight_k', [embed_dim, embed_dim]);
                blobs.weight('bias_k', [embed_dim], 'float32');
                blobs.weight('weight_v', [embed_dim, embed_dim]);
                blobs.weight('bias_v', [embed_dim], 'float32');
                blobs.weight('weight_out', [embed_dim, embed_dim]);
                blobs.weight('bias_out', [embed_dim], 'float32');
                params.delete('2');
                break;
            }
            case 'Gemm': {
                const transA = parseInt(params.get('2') || 0, 10);
                const transB = parseInt(params.get('3') || 0, 10);
                const constantA = parseInt(params.get('4') || 0, 10);
                const constantB = parseInt(params.get('5') || 0, 10);
                const constantC = parseInt(params.get('6') || 0, 10);
                const M = parseInt(params.get('7') || 0, 10);
                const N = parseInt(params.get('8') || 0, 10);
                const K = parseInt(params.get('9') || 0, 10);
                const constant_broadcast_type_C = parseInt(params.get('10') || 0, 10);
                if (constantA === 1) {
                    blobs.weight('A', transA === 0 ? [K, M] : [M, K]);
                }
                if (constantB === 1) {
                    blobs.weight('B', transB === 1 ? [N, K] : [K, N]);
                }
                if (constantC === 1 && constant_broadcast_type_C !== -1) {
                    let shape = null;
                    switch (constant_broadcast_type_C) {
                        case 0: shape = [1]; break;
                        case 1: shape = [M]; break;
                        case 2: shape = [1, M]; break;
                        case 3: shape = [N, M]; break;
                        case 4: shape = [N, 1]; break;
                        default: break;
                    }
                    if (shape) {
                        blobs.weight('C', shape);
                    }
                }
                break;
            }
            default: {
                break;
            }
        }
        if (params && params.size > 0) {
            const attributes = this.type && Array.isArray(this.type.attributes) ? this.type.attributes : [];
            for (const [index, obj] of params) {
                const metadata = attributes[index];
                let name = index;
                let value = obj;
                let type = '';
                let visible = true;
                if (metadata) {
                    name = metadata.name;
                    type = metadata.type ? metadata.type : type;
                    switch (type) {
                        case 'int32': {
                            value = parseInt(obj, 10);
                            break;
                        }
                        case 'float32': {
                            value = parseFloat(obj);
                            break;
                        }
                        case 'float32[]': {
                            value = obj.map((v) => parseFloat(v));
                            break;
                        }
                        default: {
                            value = type ? ncnn.Utility.value(obj, type) : obj;
                            break;
                        }
                    }
                    if (metadata && metadata.visible === false) {
                        visible = false;
                    } else if (metadata.default !== undefined) {
                        if (value === metadata.default || (value && value.toString() === metadata.default.toString())) {
                            visible = false;
                        }
                    }
                }
                const argument = new ncnn.Argument(name, value, type, visible);
                this.attributes.push(argument);
            }
        }
    }
};

ncnn.Tensor = class {

    constructor(type, values) {
        this.type = type;
        this.values = values;
    }
};

ncnn.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

ncnn.TensorShape = class {

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

ncnn.Utility = class {

    static value(value, type) {
        ncnn.Utility._enum = ncnn.Utility._enum || new Map([
            ['BinaryOpType', ['Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'Pow', 'RSub', 'RDiv']],
            ['CastOpType', ['Auto', 'Float32', 'Float16', 'Int8', 'BFloat16']],
            ['EltwiseType', ['Prod', 'Sum', 'Max']],
            ['PaddingType', ['Constant', 'Replicate', 'Reflect']],
            ['PoolingType', ['Max', 'Average']],
            ['InterpResizeType', ['', 'Nearest', 'Bilinear', 'Bicubic']],
            ['PermuteOrderType', ['WH WHC WHDC', 'HW HWC HWDC', 'WCH WDHC', 'CWH DWHC', 'HCW HDWC', 'CHW DHWC', 'WHCD', 'HWCD', 'WCHD', 'CWHD', 'HCWD', 'CHWD', 'WDCH', 'DWCH', 'WCDH', 'CWDH', 'DCWH', 'CDWH', 'HDCW', 'DHCW', 'HCDW', 'CHDW', 'DCHW', 'CDHW']],
            ['ReductionOpType', ['Sum', 'ASum', 'SumSq', 'Mean', 'Max', 'Min', 'Prod', 'L1', 'L2', 'LogSum', 'LogSumExp']],
            ['UnaryOpType', ['Abs', 'Neg', 'Floor', 'Ceil', 'Square', 'Sqrt', 'Rsq', 'Exp', 'Log', 'Sin', 'Cos', 'Tan', 'ASin', 'ACos', 'ATan', 'Reciprocal', 'Tanh']]
        ]);
        if (ncnn.Utility._enum.has(type) && typeof value === 'string') {
            const index = parseInt(value, 10);
            const list = ncnn.Utility._enum.get(type);
            if (Number.isInteger(index) && index < list.length) {
                return list[index];
            }
        }
        return value;
    }
};

ncnn.TextParamReader = class {

    constructor(reader) {
        const lines = [];
        for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
            line = line.trim();
            lines.push(line);
        }
        const signature = lines.shift();
        const header = (signature === '7767517' ? lines.shift() : signature).split(' ');
        if (header.length !== 2 || !header.every((value) => value >>> 0 === parseFloat(value))) {
            throw new ncnn.Error('Invalid header.');
        }
        this.layers = [];
        while (lines.length > 0) {
            const line = lines.shift();
            if (line.length > 0) {
                const columns = line.split(' ').filter((s) => s.length !== 0);
                const type = columns.shift();
                const name = columns.shift();
                const inputCount = parseInt(columns.shift(), 10);
                const outputCount = parseInt(columns.shift(), 10);
                const inputs = columns.splice(0, inputCount);
                const outputs = columns.splice(0, outputCount);
                const params = new Map();
                let index = 0;
                for (const column of columns) {
                    const parts = column.split('=');
                    if (parts.length > 2) {
                        throw new ncnn.Error(`Invalid attribute '${column}'.`);
                    }
                    let key = (parts.length === 2) ? parts[0].trim() : index.toString();
                    let value = (parts.length === 2) ? parts[1].trim() : parts[0].trim();
                    const keyInt = parseInt(key, 10);
                    if (keyInt < 0) {
                        value = value.split(',').map((v) => v.trim());
                        value.shift();
                        key = (-(keyInt + 23300)).toString();
                    }
                    params.set(key, value);
                    index++;
                }
                this.layers.push({ type, name, inputs, outputs, params });
            }
        }
    }
};

ncnn.BinaryParamReader = class {

    constructor(buffer) {
        const reader = base.BinaryReader.open(buffer);
        if (reader.int32() !== 0x007685dd) {
            throw new ncnn.Error('Invalid signature.');
        }
        const layerCount = reader.int32();
        /* const blobCount = */ reader.int32();
        this.layers = [];
        for (let i = 0; i < layerCount; i++) {
            const type = reader.int32();
            const name = i.toString();
            const inputs = new Array(reader.int32());
            const outputs = new Array(reader.int32());
            for (let j = 0; j < inputs.length; j++) {
                inputs[j] = reader.int32().toString();
            }
            for (let j = 0; j < outputs.length; j++) {
                outputs[j] = reader.int32().toString();
            }
            const params = new Map();
            let id = reader.int32();
            while (id !== -233) {
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
                    params.set(key, values);
                } else {
                    const value = reader.int32();
                    params.set(key, value);
                }
                id = reader.int32();
            }
            this.layers.push({ type, name, inputs, outputs, params });
        }
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
                    // https://github.com/Tencent/ncnn/blob/c59885aeac6cec0dbfa010efc0b5c25bed5208b7/src/modelbin.cpp#L197
                    switch (type) {
                        case 0x00000000: dataType = 'float32'; break;
                        case 0x01306b47: dataType = 'float16'; break;
                        case 0x000d4b38: dataType = 'int8'; break;
                        case 0x00000001: dataType = 'qint8'; break;
                        // case 0x0002C056: size * sizeof(float) - raw data with extra scaling
                        default: {
                            const hex = (type >>> 0).toString(16).padStart(8, '0');
                            throw new ncnn.Error(`Unsupported weight type '${hex}'.`);
                        }
                    }
                } else {
                    this._buffer = null;
                }
            }
            if (!shape) {
                this._buffer = null;
            }
            let data = null;
            if (this._buffer) {
                if (dataType) {
                    const dataTypes = new Map([['float32', 4], ['float16', 2], ['int8', 1], ['qint8', 1]]);
                    if (!dataTypes.has(dataType)) {
                        throw new ncnn.Error(`Unsupported weight type '${dataType}'.`);
                    }
                    const itemsize = dataTypes.get(dataType);
                    const size = shape.reduce((a, b) => a * b, 1) * itemsize;
                    const position = this._position;
                    if (dataType === 'qint8') {
                        this._position += size + 1024;
                        data = null;
                    } else {
                        this._position += size;
                        data = this._buffer.subarray(position, this._position);
                    }
                    const remainder = this._position % 4;
                    if (remainder !== 0) {
                        this._position += 4 - remainder;
                    }
                }
            }
            return { dataType, data };
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

export const ModelFactory = ncnn.ModelFactory;
