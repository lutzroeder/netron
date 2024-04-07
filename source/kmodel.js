
import * as base from './base.js';

const kmodel = {};

kmodel.ModelFactory = class {

    match(context) {
        const reader = kmodel.Reader.open(context.stream);
        if (reader) {
            context.type = 'kmodel';
            context.target = reader;
        }
    }

    async open(context) {
        const target = context.target;
        target.read();
        return new kmodel.Model(target);
    }
};

kmodel.Model = class {

    constructor(model) {
        this.format = `kmodel v${model.version}`;
        this.graphs = model.modules.map((module) => new kmodel.Graph(module));
    }
};

kmodel.Graph = class {

    constructor(module) {
        this.name = module.name || '';
        this.type = module.type || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const scopes = new Map();
        let index = 0;
        const values = new Map();
        const value = (arg) => {
            const name = arg.name;
            const type = arg.shape ? new kmodel.TensorType(arg.datatype || '?', arg.shape) : null;
            if (arg.data) {
                const tensor = arg.data ? new kmodel.Tensor(type, arg.data) : null;
                return new kmodel.Value(name, type || null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new kmodel.Value(name, type || null, null));
            } if ((type && !type.equals(values.get(name).type))) {
                return new kmodel.Value(name, type);
            }
            return values.get(name);
        };
        for (const layer of module.layers) {
            for (const input of layer.inputs || []) {
                for (const arg of input.value) {
                    arg.name = scopes.has(arg.name) ? scopes.get(arg.name) : arg.name;
                }
            }
            for (const output of layer.outputs || []) {
                for (const arg of output.value) {
                    const name = scopes.has(arg.name) ? `${arg.name}#${index}` : arg.name;
                    scopes.set(arg.name, name); // custom argument id
                    arg.name = name;
                    if (arg.name && arg.shape && !arg.data) {
                        value(arg);
                    }
                }
            }
            index++;
        }
        for (const layer of module.layers) {
            for (const output of layer.outputs || []) {
                for (const arg of output.value) {
                    if (arg.name && arg.shape && !arg.data) {
                        value(arg);
                    }
                }
            }
        }
        for (const layer of module.layers) {
            for (const input of layer.inputs || []) {
                for (const arg of input.value) {
                    if (arg.name && arg.shape && !arg.data) {
                        value(arg);
                    }
                }
            }
        }
        for (const layer of module.layers) {
            switch (layer.type.name) {
                case 'INPUT':
                case 'input': {
                    for (const input of layer.outputs) {
                        const values = input.value.map((arg) => value(arg));
                        const argument = new kmodel.Argument('input', values);
                        this.inputs.push(argument);
                    }
                    break;
                }
                case 'OUTPUT':
                case 'output': {
                    for (const output of layer.inputs) {
                        const values = output.value.map((arg) => value(arg));
                        const argument = new kmodel.Argument(output.name, values);
                        this.outputs.push(argument);
                    }
                    break;
                }
                default: {
                    const node = new kmodel.Node(layer, value);
                    this.nodes.push(node);
                    break;
                }
            }
        }
    }
};

kmodel.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

kmodel.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new kmodel.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

kmodel.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = new kmodel.TensorShape(shape);
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

kmodel.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length) {
                return obj.dimensions.every((value, index) => this.dimensions[index] === value);
            }
            if (obj.dimensions.every((dim) => Number.isInteger(dim)) && this.dimensions.every((dim) => Number.isInteger(dim))) {
                const a = obj.dimensions.reduce((a, b) => a * b, 1);
                const b = this.dimensions.reduce((a, b) => a * b, 1);
                return a === b;
            }
        }
        return false;
    }

    toString() {
        if (this.dimensions && Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dim) => dim ? dim.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

kmodel.Tensor = class {

    constructor(type, data) {
        this.type = type;
        this.values = data;
    }
};

kmodel.Node = class {

    constructor(layer, value) {
        this.location = layer.location === undefined ? layer.location : layer.location.toString();
        this.name = '';
        this.type = layer.type;
        this.inputs = [];
        this.outputs = [];
        this.chain = [];
        this.attributes = [];
        this.chain = [];
        for (const [name, value] of Object.entries(layer)) {
            if (name === 'type' || name === 'location' || name === 'inputs' || name === 'outputs' || name === 'chain') {
                continue;
            }
            const attribute = new kmodel.Attribute(name, value);
            this.attributes.push(attribute);
        }
        for (const input of layer.inputs || []) {
            const values = input.value.map((arg) => value(arg));
            const argument = new kmodel.Argument(input.name, values);
            this.inputs.push(argument);
        }
        for (const output of layer.outputs || []) {
            const values = output.value.map((arg) => value(arg));
            const argument = new kmodel.Argument(output.name, values);
            this.outputs.push(argument);
        }
        for (const chain of layer.chain || []) {
            const node = new kmodel.Node(chain, value);
            this.chain.push(node);
        }
    }
};

kmodel.Attribute = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

kmodel.Reader = class {

    static open(stream) {
        if (stream && stream.length >= 4) {
            const length = Math.min(8, stream.length);
            const buffer = stream.peek(length);
            if ([0x03, 0x00, 0x00, 0x00].every((value, index) => value === buffer[index])) {
                return new kmodel.Reader(stream, 3);
            }
            if ([0x4C, 0x44, 0x4D, 0x4B].every((value, index) => value === buffer[index]) && buffer.length >= 8) {
                const reader = base.BinaryReader.open(buffer);
                reader.skip(4);
                const version = reader.uint32();
                return new kmodel.Reader(stream, version);
            }
        }
        return null;
    }

    constructor(stream, version) {
        this.stream = stream;
        this.version = version;
        this.modules = [];
    }

    read() {
        if (this.version < 3 || this.version > 5) {
            throw new kmodel.Error(`Unsupported model version '${this.version}'.`);
        }
        const types = new Map();
        const register = (type, name, category, callback) => {
            types.set(type, { type: { name, category: category || '' }, callback });
        };
        switch (this.version) {
            case 3: {
                const reader = new kmodel.BinaryReader.v3(this.stream);
                const model_header = reader.kpu_model_header_t();
                const layers = new Array(model_header.layers_length);
                const outputs = new Array(model_header.output_count);
                for (let i = 0; i < model_header.output_count; i++) {
                    outputs[i] = reader.kpu_model_output_t(`output${i > 0 ? i.toString() : ''}`);
                }
                for (let i = 0; i < layers.length; i++) {
                    layers[i] = reader.kpu_model_layer_header_t();
                    layers[i].location = i;
                }
                let offset = reader.position;
                for (const layer of layers) {
                    layer.offset = offset;
                    offset += layer.body_size;
                }
                /* eslint-disable space-in-parens */
                register(   -1, 'DUMMY');
                register(    0, 'INVALID');
                register(    1, 'ADD');
                register(    2, 'QUANTIZED_ADD');
                register(    3, 'GLOBAL_MAX_POOL2D', 'Pool');
                register(    4, 'QUANTIZED_GLOBAL_MAX_POOL2D', 'Pool');
                register(    5, 'GLOBAL_AVERAGE_POOL2D', 'Pool', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.kernel_size = reader.uint32();
                    layer.channels = reader.uint32();
                });
                register(    6, 'QUANTIZED_GLOBAL_AVERAGE_POOL2D', 'Pool');
                register(    7, 'MAX_POOL2D', 'Pool');
                register(    8, 'QUANTIZED_MAX_POOL2D', 'Pool', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.outputs[0].value[0].shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.kernel = [reader.uint32(), reader.uint32()];
                    layer.stride = [reader.uint32(), reader.uint32()];
                    layer.padding = [reader.uint32(), reader.uint32()];
                });
                register(    9, 'AVERAGE_POOL2D', 'Pool');
                register(   10, 'QUANTIZED_AVERAGE_POOL2D', 'Pool');
                register(   11, 'QUANTIZE', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.count = reader.uint32();
                    layer.scale = reader.float32();
                    layer.bias = reader.float32();
                });
                register(   12, 'DEQUANTIZE', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.count = reader.uint32();
                    layer.scale = reader.float32();
                    layer.bias = reader.float32();
                });
                register(   13, 'REQUANTIZE', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.count = reader.uint32();
                    layer.table = reader.read(256);
                });
                register(   14, 'L2_NORMALIZATION', 'Normalization');
                register(   15, 'SOFTMAX', 'Activation', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.channels = reader.uint32();
                });
                register(   16, 'CONCAT', 'Tensor', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs_mem = new Array(reader.uint32());
                    for (let i = 0; i < layer.inputs_mem.length; i++) {
                        layer.inputs_mem[i] = {
                            start: reader.uint32(),
                            end: reader.uint32()
                        };
                    }
                });
                register(   17, 'QUANTIZED_CONCAT', 'Tensor', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs_mem = new Array(reader.uint32());
                    for (let i = 0; i < layer.inputs_mem.length; i++) {
                        layer.inputs_mem[i] = {
                            start: reader.uint32(),
                            end: reader.uint32()
                        };
                    }
                });
                register(   18, 'FULLY_CONNECTED', 'Layer', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.in_channels = reader.uint32();
                    layer.out_channels = reader.uint32();
                    const act = reader.uint32();
                    const activations = [
                        { name: 'LINEAR', category: 'Activation' },
                        { name: 'RELU', category: 'Activation' },
                        { name: 'RELU6', category: 'Activation' },
                    ];
                    if (act !== 0) {
                        if (act > activations.length) {
                            throw new kmodel.Error(`Unsupported FULLY_CONNECTED activation '${act}'.`);
                        }
                        layer.chain = [{ type: activations[act] }];
                    }
                    layer.inputs.push({ name: 'weights', value: [{ name: '', datatype: 'float32', shape: [layer.in_channels, layer.out_channels], data: reader.read(4 * layer.in_channels * layer.out_channels) }] });
                    layer.inputs.push({ name: 'bias', value: [{ name: '', datatype: 'float32', shape: [layer.out_channels], data: reader.read(4 * layer.out_channels) }] });
                });
                register(   19, 'QUANTIZED_FULLY_CONNECTED', 'Layer');
                register(   20, 'TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    const shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.inputs[0].value[0].shape = shape;
                    layer.outputs[0].value[0].shape = shape;
                });
                register(   21, 'QUANTIZED_TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    const shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.inputs[0].value[0].shape = shape;
                    layer.outputs[0].value[0].shape = shape;
                });
                register(   22, 'RESIZE_NEAREST_NEIGHBOR', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.out_width = reader.uint32();
                    layer.out_height = reader.uint32();
                    layer.align_corners = reader.uint32();
                });
                register(   23, 'QUANTIZED_RESIZE_NEAREST_NEIGHBOR', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.out_width = reader.uint32();
                    layer.out_height = reader.uint32();
                    layer.align_corners = reader.uint32();
                });
                register( 1000, 'CONV', 'Layer');
                register( 1001, 'DWCONV', 'Layer');
                register( 1002, 'QUANTIZED_RESHAPE', 'Shape');
                register( 1003, 'RESHAPE', 'Shape');
                register(10240, 'K210_CONV', 'Layer', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.outputs = [reader.parameter('output')];
                    const layer_offset = reader.uint32();
                    const weights_offset = reader.uint32();
                    const bn_offset = reader.uint32();
                    const act_offset = reader.uint32();
                    reader.seek(layer_offset);
                    layer.interrupt_enabe = reader.uint64_bits({ int_en: 0, ram_flag: 1, full_add: 2, depth_wise_layer: 3 });
                    layer.inputs = [reader.parameter('input', 'kpu')];
                    const outputs = [reader.parameter('output', 'kpu')];
                    layer.outputs[0].value.push(outputs[0].value[0]);
                    // layer.outputs = layer.flags & 1 ? layer.outputs : outputs;
                    layer.image_channel_num = reader.uint64_bits({ i_ch_num: 0, o_ch_num: 32, o_ch_num_coef: 48 });
                    layer.image_size =  reader.uint64_bits({ i_row_wid: 0, i_col_high: 10, o_row_wid: 32, o_col_high : 42 });
                    layer.kernel_pool_type_cfg = reader.uint64_bits({ kernel_type: 0, pad_type: 3, pool_type: 4, first_stride: 8, bypass_conv: 9, load_para: 10, dma_burst_size: 16, pad_value: 24, bwsx_base_addr: 32 });
                    layer.kernel_load_cfg = reader.uint64_bits({ load_coor: 0, load_time: 1, para_size: 15, para_start_addr: 32 });
                    layer.kernel_offset = reader.uint64_bits({ coef_column_offset: 0, coef_row_offset: 4 });
                    layer.kernel_calc_type_cfg = reader.uint64_bits({ channel_switch_addr: 0, row_switch_addr: 16, coef_size: 20, coef_group: 28, load_act: 31, active_addr: 32 });
                    layer.write_back_cfg = reader.uint64_bits({ wb_channel_switch_addr: 0, wb_row_switch_addr: 16, wb_group: 20 });
                    layer.conv_value = reader.uint64_bits({ shr_w: 0, shr_x: 4, arg_w: 8, arg_x: 32 });
                    layer.conv_value2 = reader.uint64_bits({ arg_add: 0 });
                    layer.dma_parameter = reader.uint64_bits({ send_data_out: 0, channel_byte_num: 16, dma_total_byte: 32 });
                    layer.chain = [];
                    const ic = layer.image_channel_num.i_ch_num + 1;
                    const oc = layer.image_channel_num.o_ch_num + 1;
                    layer.outputs[0].value[0].shape = [layer.image_size.o_row_wid + 1, layer.image_size.o_col_high + 1, oc];
                    const filter = [1, 3][layer.kernel_pool_type_cfg.kernel_type];
                    const weights_shape = layer.interrupt_enabe.depth_wise_layer ? [oc, filter, filter] : [ic, oc, filter, filter];
                    const weights_size = weights_shape.reduce((a, b) => a * b);
                    reader.seek(bn_offset);
                    const batch_norm = {
                        type: { name: 'BATCH_NORM', category: 'Normalization' },
                        weights: []
                    };
                    batch_norm.weights = new Array(oc);
                    for (let i = 0; i < oc; i++) {
                        batch_norm.weights[i] = reader.uint64_bits({ norm_mul: 0, norm_add: 24, norm_shift: 56, reserved: 60 });
                        delete batch_norm.weights[i].reserved;
                    }
                    layer.chain.push(batch_norm);
                    reader.seek(act_offset);
                    const activation = {};
                    activation.type = { name: 'ACTIVATION', category: 'Activation' };
                    activation.activate_para = new Array(16);
                    for (let i = 0; i < 16; i++) {
                        activation.activate_para[i] = reader.uint64_bits({ shift_number: 0, y_mul: 8, x_start: 24, reserved: 60 });
                        delete activation.activate_para[i].reserved;
                    }
                    for (let i = 0; i < 16; i++) {
                        activation.activate_para[i].bias = reader.int8();
                    }
                    layer.chain.push(activation);
                    reader.seek(weights_offset);
                    layer.inputs.push({
                        name: 'weights',
                        value: [{
                            name: '',
                            datatype: 'uint8',
                            shape: weights_shape,
                            data: reader.read(weights_size)
                        }]
                    });
                    delete layer.kernel_pool_type_cfg.bwsx_base_addr;
                    delete layer.kernel_calc_type_cfg.active_addr;
                    delete layer.kernel_load_cfg.para_start_addr;
                });
                register(10241, 'K210_ADD_PADDING', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output', 'kpu')];
                    layer.channels = reader.uint32();
                });
                register(10242, 'K210_REMOVE_PADDING', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.channels = reader.uint32();
                });
                register(10243, 'K210_UPLOAD', '', (layer, reader) => {
                    layer.flags = reader.uint32();
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output', 'kpu')];
                    const shape = [reader.uint32(), reader.uint32(), reader.uint32()];
                    layer.inputs[0].value[0].shape = shape;
                    layer.outputs[0].value[0].shape = shape;
                });
                /* eslint-enable space-in-parens */
                for (const layer of layers) {
                    const type = types.get(layer.type);
                    if (!type) {
                        throw new kmodel.Error(`Unsupported version '${this.version}' layer type '${layer.type}'.`);
                    }
                    if (!type.callback) {
                        throw new kmodel.Error(`Unsupported version '${this.version}' layer '${type.type.name}'.`);
                    }
                    layer.type = type.type;
                    reader.seek(layer.offset);
                    type.callback(layer, reader);
                    delete layer.offset;
                    delete layer.body_size;
                }
                if (layers.length > 0) {
                    layers.unshift({
                        type: { name: 'input' },
                        outputs: [layers[0].inputs[0]]
                    });
                }
                for (const output of outputs) {
                    layers.push({
                        type: { name: 'output' },
                        inputs: output.address
                    });
                }
                this.modules.push({
                    name: '',
                    layers
                });
                break;
            }
            case 4: {
                const reader = new kmodel.BinaryReader.v4(this.stream);
                const model_header = {
                    flags: reader.uint32(),
                    target: reader.uint32(), // 0=CPU, 1=K210
                    constants: reader.uint32(),
                    main_mem: reader.uint32(),
                    nodes: reader.uint32(),
                    inputs: reader.uint32(),
                    outputs: reader.uint32(),
                    reserved0: reader.uint32(),
                };
                const inputs = new Array(model_header.inputs);
                for (let i = 0; i < inputs.length; i++) {
                    inputs[i] = reader.parameter(`input${i === 0 ? '' : (i + 1)}`);
                }
                for (let i = 0; i < inputs.length; i++) {
                    inputs[i].value[0].shape = reader.runtime_shape_t();
                }
                const outputs = new Array(model_header.outputs);
                for (let i = 0; i < outputs.length; i++) {
                    outputs[i] = reader.parameter(`output${i === 0 ? '' : (i + 1)}`);
                }
                reader.constants(model_header.constants);
                const layers = new Array(model_header.nodes);
                for (let i = 0; i < layers.length; i++) {
                    layers[i] = {
                        location: i,
                        opcode: reader.uint32(),
                        body_size: reader.uint32()
                    };
                }
                let offset = reader.position;
                for (const layer of layers) {
                    layer.offset = offset;
                    offset += layer.body_size;
                }
                /* eslint-disable space-in-parens */
                register(  0x00, 'binary', '', (layer, reader) => {
                    layer.inputs = [
                        reader.parameter('a'),
                        reader.parameter('b')
                    ];
                    layer.outputs = [reader.parameter('outputs')];
                    layer.binary_op = reader.binary_op_t();
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.inputs[1].value[0].shape = reader.runtime_shape_t();
                    layer.outputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.fused_activation = [reader.float32(), reader.float32()];
                });
                register(  0x01, 'concat', 'Tensor', (layer, reader) => {
                    layer.outputs = [reader.parameter('output')];
                    layer.inner_size = reader.uint32();
                    layer.outer_size = reader.uint32();
                    const inputs_count = reader.uint32();
                    layer.inputs = [{ name: 'inputs', value: [] }];
                    for (let i = 0; i < inputs_count; i++) {
                        layer.inputs[0].value[i] = reader.argument();
                    }
                    layer.dims = new Array(inputs_count);
                    for (let i = 0; i < inputs_count; i++) {
                        layer.dims[i] = reader.int32();
                    }
                });
                register(  0x02, 'conv2d', 'Layer', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.groups = reader.int32();
                    layer.out_channels = reader.int32();
                    layer.padding_h = reader.padding();
                    layer.padding_w = reader.padding();
                    layer.filter_h = reader.int32();
                    layer.filter_w = reader.int32();
                    layer.stride_h = reader.int32();
                    layer.stride_w = reader.int32();
                    layer.dilation_h = reader.int32();
                    layer.dilation_w = reader.int32();
                    layer.fused_activation = [reader.float32(), reader.float32()];
                    const weights_shape = [layer.out_channels, layer.inputs[0].value[0].shape[1] / layer.groups, layer.filter_h, layer.filter_w];
                    const weights_size = 4 * weights_shape.reduce((a, b) => a * b);
                    layer.inputs.push({
                        name: 'weights',
                        value: [{
                            name: '',
                            datatype: 'float32',
                            shape: weights_shape,
                            data: reader.read(weights_size)
                        }]
                    });
                    const bias_shape = [layer.out_channels];
                    const bias_size = 4 * layer.out_channels;
                    layer.inputs.push({
                        name: 'bias',
                        value: [{
                            name: '',
                            datatype: 'float32',
                            shape: bias_shape,
                            data: reader.read(bias_size)
                        }]
                    });
                });
                register(  0x03, 'dequantize', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.zero_point = reader.int32();
                    layer.scale = reader.float32();
                });
                register(  0x04, 'matmul', '', (layer, reader) => {
                    layer.inputs = [
                        reader.parameter('a'),
                        reader.parameter('b'),
                    ];
                    layer.outputs = [reader.parameter('output')];
                    layer.a_rows = reader.int32();
                    layer.a_cols = reader.int32();
                    layer.b_cols = reader.int32();
                    layer.inputs[1].value[0].shape = [layer.a_cols, layer.b_cols];
                    layer.fused_activation = [reader.float32(), reader.float32()];
                    const bias = reader.read(4 * layer.b_cols);
                    if (!bias.every((value) => value === 0)) {
                        layer.inputs.push({
                            name: 'bias',
                            value: [{ name: '', datatype: 'float32', shape: [layer.b_cols], data: bias }]
                        });
                    }
                });
                register(  0x05, 'pad', 'Shape', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.paddings = reader.runtime_paddings_t();
                    layer.pad_value = reader.scalar();
                });
                register(  0x06, 'quantize', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.zero_point = reader.int32();
                    layer.scale = reader.float32();
                });
                register(  0x07, 'reduce', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.reduce_op = reader.reduce_op_t();
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.outputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.init_value = reader.float32();
                });
                register(  0x08, 'reduce_window2d', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.reduce_op = reader.reduce_op_t();
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.padding_h = reader.padding();
                    layer.padding_w = reader.padding();
                    layer.filter_h = reader.int32();
                    layer.filter_w = reader.int32();
                    layer.stride_h = reader.int32();
                    layer.stride_w = reader.int32();
                    layer.dilation_h = reader.int32();
                    layer.dilation_w = reader.int32();
                    layer.init_value = reader.float32();
                    layer.fused_activation = [reader.float32(), reader.float32()];
                });
                register(  0x09, 'memory_copy', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                });
                register(  0x0A, 'resize_image', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.reduce_op = reader.reduce_op_t();
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.out_h = reader.int32();
                    layer.out_w = reader.int32();
                    layer.mode = reader.image_resize_mode_t();
                    layer.align_corners = reader.boolean();
                });
                register(  0x0B, 'softmax', 'Activation');
                register(  0x0C, 'transpose', 'Transform', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.perm = reader.runtime_shape_t();
                });
                register(  0x0D, 'strided_slice', 'Tensor');
                register(  0x0E, 'unary', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.unary_op = reader.unary_op_t();
                });
                register(  0x0F, 'quantized_conv2d', 'Layer', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.groups = reader.int32();
                    layer.out_channels = reader.int32();
                    layer.padding_h = reader.padding();
                    layer.padding_w = reader.padding();
                    layer.filter_h = reader.int32();
                    layer.filter_w = reader.int32();
                    layer.stride_h = reader.int32();
                    layer.stride_w = reader.int32();
                    layer.dilation_h = reader.int32();
                    layer.dilation_w = reader.int32();
                    layer.input_offset = reader.int32();
                    layer.filter_offset = reader.int32();
                    layer.output_mul = reader.int32();
                    layer.output_shift = reader.int32();
                    layer.output_offset = reader.int32();
                    const bias = reader.span('int32', [layer.out_channels]);
                    if (bias) {
                        layer.inputs.push({ name: 'bias', value: [bias] });
                    }
                    const weights = reader.span('uint8', [layer.out_channels, layer.inputs[0].value[0].shape[1] / layer.groups, layer.filter_h, layer.filter_w]);
                    if (weights) {
                        layer.inputs.push({ name: 'weights', value: [weights] });
                    }
                });
                register(  0x10, 'quantized_matmul', '', (layer, reader) => {
                    layer.inputs = [
                        reader.parameter('a'),
                        reader.parameter('b'),
                    ];
                    layer.outputs = [reader.parameter('output')];
                    layer.a_rows = reader.int32();
                    layer.a_cols = reader.int32();
                    layer.b_cols = reader.int32();
                    layer.inputs[1].value[0].shape = [layer.a_cols, layer.b_cols];
                    layer.input_a_offset = reader.int32();
                    layer.input_b_offset = reader.int32();
                    layer.output_mul = reader.int32();
                    layer.output_shift = reader.int32();
                    layer.output_offset = reader.int32();
                    const bias = reader.span('int32', [layer.b_cols]);
                    if (bias) {
                        layer.inputs.push({ name: 'bias', value: [bias] });
                    }
                });
                register(  0x11, 'quantized_binary', '', (layer, reader) => {
                    layer.inputs = [
                        reader.parameter('a'),
                        reader.parameter('b')
                    ];
                    layer.outputs = [reader.parameter('outputs')];
                    layer.binary_op = reader.binary_op_t();
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.inputs[1].value[0].shape = reader.runtime_shape_t();
                    layer.outputs[0].value[0].shape = reader.runtime_shape_t();
                    layer.input_a_offset = reader.int32();
                    layer.input_a_mul = reader.int32();
                    layer.input_a_shift = reader.int32();
                    layer.input_b_offset = reader.int32();
                    layer.input_b_mul = reader.int32();
                    layer.input_b_shift = reader.int32();
                    layer.output_offset = reader.int32();
                    layer.output_mul = reader.int32();
                    layer.output_shift = reader.int32();
                });
                register(  0x12, 'table_lookup1d', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input'), reader.parameter('table')];
                    layer.outputs = [reader.parameter('output')];
                });
                register(  0x13, 'conv2d_transpose', 'Layer');
                register(  0x14, 'nnil_unary_method', '', (layer, reader, size) => {
                    const position = reader.position;
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.body = reader.read(size - (reader.position - position));
                });
                register(0x1001, 'cpu_conv2d', 'Layer');
                register(0x1002, 'cpu_depthwise_conv2d', 'Layer');
                register(0x1003, 'cpu_reduce_window2d');
                register(0x1004, 'cpu_quantized_conv2d', 'Layer');
                register(0x1005, 'cpu_quantized_depthwise_conv2d', 'Layer');
                register(0x2001, 'kpu_upload', '', (layer, reader) => {
                    layer.inputs = [reader.parameter('input')];
                    layer.outputs = [reader.parameter('output')];
                    layer.inputs[0].value[0].shape = reader.runtime_shape_t();
                });
                register(0x2002, 'kpu_conv2d', 'Layer', (layer, reader) => {
                    layer.outputs = [reader.parameter('output')];
                    layer.batches = reader.int32();
                    layer.reserved0 = reader.int32();
                    layer.interrupt_enabe = reader.uint64_bits({ int_en: 0, ram_flag: 1, full_add: 2, depth_wise_layer: 3 });
                    const image_src_addr = reader.uint32();
                    const image_dst_addr = reader.uint32();
                    layer.inputs = [{ name: 'input', value: [{ name: `kpu:${image_src_addr}` }] }];
                    const outputs = [{ name: 'output', value: [{ name: `kpu:${image_dst_addr}` }] }];
                    layer.outputs[0].value.push(outputs[0].value[0]);
                    // layer.outputs = layer.flags & 1 ? layer.outputs : outputs;
                    layer.image_channel_num = reader.uint64_bits({ i_ch_num: 0, o_ch_num: 32, o_ch_num_coef: 48 });
                    layer.image_size =  reader.uint64_bits({ i_row_wid: 0, i_col_high: 10, o_row_wid: 32, o_col_high : 42 });
                    layer.kernel_pool_type_cfg = reader.uint64_bits({ kernel_type: 0, pad_type: 3, pool_type: 4, first_stride: 8, bypass_conv: 9, load_para: 10, dma_burst_size: 16, pad_value: 24, bwsx_base_addr: 32 });
                    layer.kernel_load_cfg = reader.uint64_bits({ load_coor: 0, load_time: 1, para_size: 15, para_start_addr: 32 });
                    layer.kernel_offset = reader.uint64_bits({ coef_column_offset: 0, coef_row_offset: 4 });
                    layer.kernel_calc_type_cfg = reader.uint64_bits({ channel_switch_addr: 0, row_switch_addr: 16, coef_size: 20, coef_group: 28, load_act: 31, active_addr: 32 });
                    layer.write_back_cfg = reader.uint64_bits({ wb_channel_switch_addr: 0, wb_row_switch_addr: 16, wb_group: 20 });
                    layer.conv_value = reader.uint64_bits({ shr_w: 0, shr_x: 4, arg_w: 8, arg_x: 32 });
                    layer.conv_value2 = reader.uint64_bits({ arg_add: 0 });
                    layer.dma_parameter = reader.uint64_bits({ send_data_out: 0, reserved: 1, channel_byte_num: 16, dma_total_byte: 32 });
                    layer.chain = [];
                    const ic = layer.image_channel_num.i_ch_num + 1;
                    const oc = layer.image_channel_num.o_ch_num + 1;
                    layer.outputs[0].value[0].shape = [layer.image_size.o_row_wid + 1, layer.image_size.o_col_high + 1, oc];
                    const filter = [1, 3][layer.kernel_pool_type_cfg.kernel_type];
                    const weights_shape = layer.interrupt_enabe.depth_wise_layer ? [oc, filter, filter] : [ic, oc, filter, filter];
                    reader.skip(layer.kernel_pool_type_cfg.bwsx_base_addr);
                    delete layer.kernel_pool_type_cfg.bwsx_base_addr;
                    const batch_norm = {
                        type: { name: 'batch_norm', category: 'Normalization' },
                        weights: []
                    };
                    batch_norm.weights = new Array(oc);
                    for (let i = 0; i < oc; i++) {
                        batch_norm.weights[i] = reader.uint64_bits({ norm_mul: 0, norm_add: 24, norm_shift: 56, reserved: 60 });
                        delete batch_norm.weights[i].reserved;
                    }
                    layer.chain.push(batch_norm);
                    reader.skip(layer.kernel_calc_type_cfg.active_addr);
                    delete layer.kernel_calc_type_cfg.active_addr;
                    const activation = reader.kpu_activate_table_t();
                    activation.type = { name: 'activation', category: 'Activation' };
                    layer.chain.push(activation);
                    reader.skip(layer.kernel_load_cfg.para_start_addr);
                    delete layer.kernel_load_cfg.para_start_addr;
                    const weights = reader.span('uint8', weights_shape);
                    if (weights) {
                        layer.inputs.push({ name: 'weights', value: [weights] });
                    }
                });
                /* eslint-enable space-in-parens */
                for (const layer of layers) {
                    const type = types.get(layer.opcode);
                    if (!type) {
                        throw new kmodel.Error(`Unsupported version '${this.version}' layer type '${layer.type}'.`);
                    }
                    if (!type.callback) {
                        throw new kmodel.Error(`Unsupported version '${this.version}' layer '${type.type.name}'.`);
                    }
                    layer.type = type.type;
                    reader.seek(layer.offset);
                    if (type.callback) {
                        type.callback(layer, reader, layer.body_size);
                    }
                    delete layer.offset;
                    delete layer.body_size;
                    delete layer.opcode;
                }
                for (const input of inputs) {
                    layers.unshift({
                        type: { name: 'INPUT' },
                        outputs: [input]
                    });
                }
                for (const output of outputs) {
                    layers.push({
                        type: { name: 'OUTPUT' },
                        inputs: [output]
                    });
                }
                this.modules.push({
                    name: '',
                    layers
                });
                break;
            }
            case 5: {
                const reader = new kmodel.BinaryReader.v5(this.stream);
                const model_header = reader.model_header();
                if (model_header.header_size < 32) {
                    throw new kmodel.Error(`Invalid header size '${model_header.header_size}'.`);
                }
                if (model_header.header_size > reader.position) {
                    reader.skip(model_header.header_size - reader.position);
                }
                delete model_header.header_size;
                this.modules = new Array(model_header.modules);
                for (let i = 0; i < this.modules.length; i++) {
                    const start = reader.position;
                    const module_header = reader.module_header();
                    if (module_header.header_size > (reader.position - start)) {
                        reader.skip(module_header.header_size - (reader.position - start));
                    }
                    const mempools = new Array(module_header.mempools);
                    for (let i = 0; i < mempools.length; i++) {
                        mempools[i] = reader.mempool_desc();
                    }
                    const shared_mempools = new Array(module_header.shared_mempools);
                    for (let i = 0; i < shared_mempools.length; i++) {
                        shared_mempools[i] = reader.mempool_desc();
                    }
                    const function_headers = new Array(module_header.functions);
                    const functions = new Array(module_header.functions);
                    for (let i = 0; i < functions.length; i++) {
                        const position = reader.position;
                        const function_header = reader.function_header();
                        const header_size = reader.position - position;
                        if (function_header.header_size > header_size) {
                            reader.skip(function_header.header_size - header_size);
                        }
                        const inputs = new Array(function_header.inputs);
                        for (let i = 0; i < inputs.length; i++) {
                            inputs[i] = reader.parameter(`input${i === 0 ? '' : (i + 1)}`);
                        }
                        for (let i = 0; i < inputs.length; i++) {
                            inputs[i].value[0].shape = reader.shape();
                        }
                        const outputs = new Array(function_header.outputs);
                        for (let i = 0; i < outputs.length; i++) {
                            outputs[i] = reader.parameter(`output${i === 0 ? '' : (i + 1)}`);
                        }
                        for (let i = 0; i < outputs.length; i++) {
                            outputs[i].value[0].shape = reader.shape();
                        }
                        reader.align(8);
                        const size = reader.size - position;
                        if (function_header.size > size) {
                            reader.skip(function_header.size - size);
                        }
                        function_headers[i] = function_header;
                        functions[i] = {
                            type: { name: 'Unknown' },
                            inputs,
                            outputs
                        };
                    }
                    const sections = new Map();
                    for (let i = 0; i < module_header.sections; i++) {
                        const section_header = reader.section_header();
                        reader.skip(section_header.body_start);
                        const body = reader.read(section_header.body_size);
                        const section = {
                            reader: base.BinaryReader.open(body),
                            flags: section_header.flags
                        };
                        reader.align(8);
                        sections.set(section_header.name, section);
                    }
                    for (let i = 0; i < function_headers.length; i++) {
                        const function_header = function_headers[i];
                        const reader = sections.get('.text').reader;
                        reader.seek(function_header.entrypoint);
                        function_header.text = reader.read(function_header.text_size);
                        const layer = functions[i];
                        switch (module_header.type) {
                            case 'stackvm':
                                layer.type = { name: 'stackvm' };
                                break;
                            case 'k210':
                                break;
                            case 'k510':
                                break;
                            default:
                                throw new kmodel.Error(`Unsupported module type '${module_header.type}'.`);
                        }
                    }
                    const name = this.modules.length > 1 ? i.toString() : '';
                    this.modules[i] = {
                        name,
                        type: module_header.type,
                        layers: functions
                    };
                }
                break;
            }
            default: {
                throw new kmodel.Error(`Unsupported model version '${this.version}'.`);
            }
        }
        delete this.stream;
    }
};

kmodel.BinaryReader = class {

    constructor(data) {
        this._reader = base.BinaryReader.open(data);
    }

    get position() {
        return this._reader.position;
    }

    seek(position) {
        this._reader.seek(position);
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    align(size) {
        this._reader.align(size);
    }

    read(length) {
        return this._reader.read(length);
    }

    boolean() {
        return this._reader.boolean();
    }

    byte() {
        return this._reader.byte();
    }

    int8() {
        return this._reader.int8();
    }

    int16() {
        return this._reader.int16();
    }

    int32() {
        return this._reader.int32();
    }

    uint16() {
        return this._reader.uint16();
    }

    uint32() {
        return this._reader.uint32();
    }

    float32() {
        return this._reader.float32();
    }

    uint64_bits(fields) {
        const buffer = this.read(8);
        fields = Object.entries(fields);
        fields.push([null, Math.min(64, fields[fields.length - 1][1] + 56)]);
        const obj = {};
        for (let i = 0; i < fields.length - 1; i++) {
            const current = fields[i];
            const next = fields[i + 1];
            const [key, start] = current;
            const [, end] = next;
            let value = 0;
            let position = start;
            while (position < end) {
                const offset = (position / 8) >> 0;
                const start = (position & 7);
                const count = Math.min((offset + 1) * 8, end) - position;
                value |= ((buffer[offset] >>> start) & ((1 << count) - 1)) << (position - fields[i][1]);
                position += count;
            }
            obj[key] = value;
        }
        return obj;
    }
};

kmodel.BinaryReader.v3 = class extends kmodel.BinaryReader {

    constructor(buffer) {
        super(buffer);
        this.skip(4);
    }

    kpu_model_header_t() {
        return {
            flags: this.uint32(),
            arch: this.uint32(),
            layers_length: this.uint32(),
            max_start_address: this.uint32(),
            main_mem_usage: this.uint32(),
            output_count: this.uint32()
        };
    }

    kpu_model_output_t(name) {
        return {
            address: [this.parameter(name)],
            size: this.uint32()
        };
    }

    kpu_model_layer_header_t() {
        return {
            type: this.uint32(),
            body_size: this.uint32()
        };
    }

    argument(memory_type) {
        memory_type = memory_type || 'main';
        const address = this.uint32();
        return { name: `${memory_type}:${address}` };
    }

    parameter(name, memory_type) {
        return { name, value: [this.argument(memory_type)] };
    }
};

kmodel.BinaryReader.v4 = class extends kmodel.BinaryReader {

    constructor(buffer) {
        super(buffer);
        this.skip(8);
        this._memory_types = ['const', 'main', 'kpu'];
        this._datatypes = ['float32', 'uint8'];
    }

    memory_type_t() {
        const value = this.uint32();
        return this._memory_types[value];
    }

    datatype_t() {
        const value = this.uint32();
        return this._datatypes[value];
    }

    memory_range() {
        return {
            memory_type: this.memory_type_t(),
            datatype: this.datatype_t(),
            start: this.uint32(),
            size: this.uint32()
        };
    }

    argument() {
        const memory = this.memory_range();
        const value = {
            name: `${memory.memory_type}:${memory.start}`,
            datatype: memory.datatype
        };
        if (memory.memory_type === 'const') {
            value.data = this._constants.slice(memory.start, memory.start + memory.size);
            switch (value.datatype) {
                case 'uint8': value.shape = [value.data.length]; break;
                case 'float32': value.shape = [value.data.length >> 2]; break;
                default: break;
            }
        }
        return value;
    }

    parameter(name) {
        return { name, value: [this.argument()] };
    }

    runtime_shape_t() {
        return [this.uint32(), this.uint32(), this.uint32(), this.uint32()];
    }

    padding() {
        return { before: this.int32(), after: this.int32() };
    }

    runtime_paddings_t() {
        return [this.padding(), this.padding(), this.padding(), this.padding()];
    }

    scalar() {
        return {
            datatype_t: this.uint32(),
            storage: this.read(4)
        };
    }

    kpu_activate_table_t() {
        const value = {};
        value.activate_para = new Array(16);
        for (let i = 0; i < 16; i++) {
            value.activate_para[i] = this.uint64_bits({ shift_number: 0, y_mul: 8, x_start: 24, reserved: 60 });
            delete value.activate_para[i].reserved;
        }
        for (let i = 0; i < 16; i++) {
            value.activate_para[i].bias = this.int8();
        }
        return value;
    }

    unary_op_t() {
        const value = this.uint32();
        return ['abs', 'ceil', 'cos', 'exp', 'floor', 'log', 'neg', 'rsqrt', 'sin', 'square'][value];
    }

    binary_op_t() {
        const value = this.uint32();
        return ['add', 'sub', 'mul', 'div', 'min', 'max'][value];
    }

    reduce_op_t() {
        const value = this.uint32();
        return ['mean', 'min', 'max', 'sum'][value];
    }

    image_resize_mode_t() {
        const value = this.uint32();
        return ['bilinear', 'nearest_neighbor'][value];
    }

    constants(size) {
        this._constants = this.read(size);
    }

    span(datatype, shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const itemsize = { 'int32': 4, 'uint8': 1 };
        const buffer = this.read(itemsize[datatype] * size);
        if (!buffer.every((value) => value === 0)) {
            const array = {};
            array.name = '';
            array.datatype = datatype;
            array.shape = shape;
            array.data = buffer;
            return array;
        }
        return null;
    }
};

kmodel.BinaryReader.v5 = class extends kmodel.BinaryReader {

    constructor(buffer) {
        super(buffer);
        this.skip(8);
        this._datatypes = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'bfloat16'];
        this._memory_locations = new Map([[0, 'input'], [1, 'output'], [2, 'rdata'], [3, 'data'], [4, 'shared_data'], [64, 'kpu']]);
    }

    model_header() {
        return {
            header_size: this.uint32(),
            flags: this.uint32(),
            alignment: this.uint32(),
            modules: this.uint32(),
            entry_module: this.uint32(),
            entry_function: this.uint32()
        };
    }

    module_type_t() {
        const buffer = this.read(16);
        const decoder = new TextDecoder('ascii');
        const text = decoder.decode(buffer);
        return text.replace(/\0.*$/, '');
    }

    module_header() {
        return {
            type: this.module_type_t(),
            version: this.uint32(),
            header_size: this.uint32(),
            size: this.uint32(),
            mempools: this.uint32(),
            shared_mempools: this.uint32(),
            sections: this.uint32(),
            functions: this.uint32(),
            reserved0: this.uint32()
        };
    }

    mempool_desc() {
        return {
            location: this.byte(),
            reserved0: this.read(3),
            size: this.uint32()
        };
    }

    section_header() {
        const buffer = this.read(16);
        const decoder = new TextDecoder('ascii');
        const name = decoder.decode(buffer);
        return {
            name: name.replace(/\0.*$/, ''),
            flags: this.uint32(),
            body_start: this.uint32(),
            body_size: this.uint32(),
            reserved0: this.uint32()
        };
    }

    function_header() {
        return {
            header_size: this.uint32(),
            size: this.uint32(),
            input_pool_size: this.uint32(),
            output_pool_size: this.uint32(),
            inputs: this.uint32(),
            outputs: this.uint32(),
            entrypoint: this.uint32(),
            text_size: this.uint32()
        };
    }

    memory_location_t() {
        const value = this.byte();
        if (!this._memory_locations.has(value)) {
            throw new kmodel.Error(`Unsupported memory location '${value}'.`);
        }
        return this._memory_locations.get(value);
    }

    datatype_t() {
        const value = this.byte();
        return this._datatypes[value];
    }

    memory_range() {
        return {
            memory_location: this.memory_location_t(),
            datatype: this.datatype_t(),
            shared_module: this.uint16(),
            start: this.uint32(),
            size: this.uint32()
        };
    }

    argument() {
        const memory = this.memory_range();
        const value = {
            name: `${memory.memory_location}:${memory.start}`,
            datatype: memory.datatype
        };
        /*
        if (memory.memory_type === 'const') {
            value.data = constants.slice(memory.start, memory.start + memory.size);
            switch (value.datatype) {
                case 'uint8': value.shape = [ value.data.length ]; break;
                case 'float32': value.shape = [ value.data.length >> 2 ]; break;
                default: break;
            }
        }
        */
        return value;
    }

    parameter(name) {
        return { name, value: [this.argument()] };
    }

    shape() {
        const array = new Array(this.uint32());
        for (let i = 0; i < array.length; i++) {
            array[i] = this.uint32();
        }
        return array;
    }
};

kmodel.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading kmodel.';
    }
};

export const ModelFactory = kmodel.ModelFactory;
