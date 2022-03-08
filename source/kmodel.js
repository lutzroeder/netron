
var kmodel = kmodel || {};
var base = base || require('./base');

kmodel.ModelFactory = class {

    match(context) {
        return kmodel.Reader.open(context.stream);
    }

    open(context, match) {
        return Promise.resolve().then(() => {
            const reader = match;
            return new kmodel.Model(reader);
        });
    }
};

kmodel.Model = class {

    constructor(model) {
        this._format = 'kmodel v' + model.version.toString();
        this._graphs = [ new kmodel.Graph(model) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

kmodel.Graph = class {

    constructor(model) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const scopes = new Map();
        let index = 0;
        for (const layer of model.layers) {
            for (const input of layer.inputs || []) {
                for (const argument of input.arguments) {
                    argument.name = scopes.has(argument.name) ? scopes.get(argument.name) : argument.name;
                }
            }
            for (const output of layer.outputs || []) {
                for (const argument of output.arguments) {
                    const value = scopes.has(argument.name) ? argument.name + '#' + index.toString() : argument.name;
                    scopes.set(argument.name, value); // custom argument id
                    argument.name = value;
                }
            }
            index++;
        }
        for (const layer of model.layers) {
            switch (layer.type.name) {
                case 'INPUT':
                case 'input': {
                    for (const input of layer.outputs) {
                        this._inputs.push(new kmodel.Parameter('input', input.arguments.map((argument) => {
                            return new kmodel.Argument(argument.name);
                        })));
                    }
                    break;
                }
                case 'OUTPUT':
                case 'output': {
                    for (const output of layer.inputs) {
                        this._outputs.push(new kmodel.Parameter(output.name, output.arguments.map((argument) => {
                            return new kmodel.Argument(argument.name);
                        })));
                    }
                    break;
                }
                default:
                    this._nodes.push(new kmodel.Node(layer));
                    break;
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

kmodel.Parameter = class {

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

kmodel.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new kmodel.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
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

kmodel.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = new kmodel.TensorShape(shape);
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

kmodel.TensorShape = class {

    constructor(shape) {
        this._shape = shape;
    }

    get dimensions() {
        return this._shape;
    }

    toString() {
        if (this._shape && Array.isArray(this._shape) && this._shape.length > 0) {
            return '[' + this._shape.map((dim) => dim ? dim.toString() : '?').join(',') + ']';
        }
        return '';
    }
};

kmodel.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
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
        if (this._data == null || this._data.length === 0) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        const dataType = this.type.dataType;
        const shape = this.type.shape.dimensions;
        if (dataType !== 'uint8' && dataType !== 'float32') {
            context.state = "Tensor data type '" + dataType + "' is not implemented.";
            return context;
        }
        context.dataType = dataType;
        context.shape = shape;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length == 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        const results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
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

kmodel.Node = class {

    constructor(layer) {
        this._location = layer.location !== undefined ? layer.location.toString() : layer.location;
        this._type = layer.type;
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        this._attributes = [];
        this._chain = [];
        for (const entry of Object.entries(layer)) {
            const name = entry[0];
            if (name === 'type' || name === 'location' || name === 'inputs' || name === 'outputs' || name === 'chain') {
                continue;
            }
            const value = entry[1];
            const attribute = new kmodel.Attribute(name, value);
            this._attributes.push(attribute);
        }
        for (const input of layer.inputs || []) {
            this._inputs.push(new kmodel.Parameter(input.name, input.arguments.map((argument) => {
                const type = argument.shape ? new kmodel.TensorType(argument.datatype || '?', argument.shape) : null;
                const tensor = argument.data ? new kmodel.Tensor(type, argument.data) : null;
                return new kmodel.Argument(argument.name, type, tensor);
            })));
        }
        for (const output of layer.outputs || []) {
            this._outputs.push(new kmodel.Parameter(output.name, output.arguments.map((argument) => {
                return new kmodel.Argument(argument.name);
            })));
        }
        for (const chain of layer.chain || []) {
            this._chain.push(new kmodel.Node(chain));
        }
    }

    get location() {
        return this._location;
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
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

    get chain() {
        return this._chain;
    }
};

kmodel.Attribute = class {

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

kmodel.Reader = class {

    static open(stream) {
        const reader = new base.BinaryReader(stream);
        if (reader.length > 4) {
            const signature = reader.uint32();
            if (signature === 3) {
                return new kmodel.Reader(reader, 3);
            }
            if (signature === 0x4B4D444C) {
                const version = reader.uint32();
                return new kmodel.Reader(reader, version);
            }
        }
        return null;
    }

    constructor(reader, version) {
        this._reader = reader;
        this._version = version;
    }

    get version() {
        return this._version;
    }

    get layers() {
        this._read();
        return this._layers;
    }

    _read() {
        if (this._reader) {
            const reader = this._reader;
            if (this._version < 3 || this._version > 5) {
                throw new kmodel.Error("Unsupported model version '" + this.version.toString() + "'.");
            }
            const types = new Map();
            const register = (type, name, category, callback) => {
                types.set(type, { type: { name: name, category: category || '' }, callback: callback });
            };
            reader.uint64_bits = function(fields) {
                const buffer = reader.read(8);
                fields = Object.entries(fields);
                fields.push([ null, Math.min(64, fields[fields.length - 1][1] + 56)]);
                const obj = {};
                for (let i = 0; i < fields.length - 1; i++) {
                    const key = fields[i][0];
                    let value = 0;
                    let position = fields[i][1];
                    const end = fields[i + 1][1];
                    while (position < end) {
                        const offset = (position / 8) >> 0;
                        const start = (position & 7);
                        const count = Math.min((offset + 1) * 8, end) - position;
                        value = value | ((buffer[offset] >>> start) & ((1 << count) - 1)) << (position - fields[i][1]);
                        position += count;
                    }
                    obj[key] = value;
                }
                return obj;
            };
            switch (this._version) {
                case 3: {
                    reader.kpu_model_header_t = function() {
                        return {
                            flags: reader.uint32(),
                            arch: reader.uint32(),
                            layers_length: reader.uint32(),
                            max_start_address: reader.uint32(),
                            main_mem_usage: reader.uint32(),
                            output_count: reader.uint32()
                        };
                    };
                    reader.kpu_model_output_t = function(name) {
                        return {
                            address: [ this.parameter(name) ],
                            size: reader.uint32()
                        };
                    };
                    reader.kpu_model_layer_header_t = function() {
                        return {
                            type: reader.uint32(),
                            body_size: reader.uint32()
                        };
                    };
                    reader.argument = function(memory_type) {
                        memory_type = memory_type || 'main';
                        const address = this.uint32();
                        return { name: memory_type + ':' + address.toString() };
                    };
                    reader.parameter = function(name, memory_type) {
                        const argument = this.argument(memory_type);
                        return { name: name, arguments: [ argument ] };
                    };
                    const model_header = reader.kpu_model_header_t();
                    this._layers = new Array(model_header.layers_length);
                    const outputs = new Array(model_header.output_count);
                    for (let i = 0; i < model_header.output_count; i++) {
                        outputs[i] = reader.kpu_model_output_t('output' + (i > 0 ? i.toString() : ''));
                    }
                    for (let i = 0; i < this._layers.length; i++) {
                        this._layers[i] = reader.kpu_model_layer_header_t();
                        this._layers[i].location = i;
                    }
                    let offset = reader.position;
                    for (const layer of this._layers) {
                        layer.offset = offset;
                        offset += layer.body_size;
                    }
                    register(   -1, 'DUMMY');
                    register(    0, 'INVALID');
                    register(    1, 'ADD');
                    register(    2, 'QUANTIZED_ADD');
                    register(    3, 'GLOBAL_MAX_POOL2D', 'Pool');
                    register(    4, 'QUANTIZED_GLOBAL_MAX_POOL2D', 'Pool');
                    register(    5, 'GLOBAL_AVERAGE_POOL2D', 'Pool', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.kernel_size = reader.uint32();
                        layer.channels = reader.uint32();
                    });
                    register(    6, 'QUANTIZED_GLOBAL_AVERAGE_POOL2D', 'Pool');
                    register(    7, 'MAX_POOL2D', 'Pool');
                    register(    8, 'QUANTIZED_MAX_POOL2D', 'Pool', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.inputs[0].arguments[0].shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.outputs[0].arguments[0].shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.kernel = [ reader.uint32(), reader.uint32() ];
                        layer.stride = [ reader.uint32(), reader.uint32() ];
                        layer.padding = [ reader.uint32(), reader.uint32() ];
                    });
                    register(    9, 'AVERAGE_POOL2D', 'Pool');
                    register(   10, 'QUANTIZED_AVERAGE_POOL2D', 'Pool');
                    register(   11, 'QUANTIZE', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.count = reader.uint32();
                        layer.scale = reader.float32();
                        layer.bias = reader.float32();
                    });
                    register(   12, 'DEQUANTIZE', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.count = reader.uint32();
                        layer.scale = reader.float32();
                        layer.bias = reader.float32();
                    });
                    register(   13, 'REQUANTIZE');
                    register(   14, 'L2_NORMALIZATION', 'Normalization');
                    register(   15, 'SOFTMAX', 'Activation', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.channels = reader.uint32();
                    });
                    register(   16, 'CONCAT', 'Tensor', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.outputs = [ reader.parameter('output') ];
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
                        layer.outputs = [ reader.parameter('output') ];
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
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
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
                                throw new kmodel.Error("Unsupported FULLY_CONNECTED activation '" + act.toString() + "'.");
                            }
                            layer.chain = [ { type: activations[act] } ];
                        }
                        layer.inputs.push({ name: 'weights', arguments: [ { name: '', datatype: 'float32', shape: [ layer.in_channels, layer.out_channels ], data: reader.read(4 * layer.in_channels * layer.out_channels) } ] });
                        layer.inputs.push({ name: 'bias', arguments: [ { name: '', datatype: 'float32', shape: [ layer.out_channels ], data: reader.read(4 * layer.out_channels) } ] });
                    });
                    register(   19, 'QUANTIZED_FULLY_CONNECTED', 'Layer');
                    register(   20, 'TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        const shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.inputs[0].arguments[0].shape = shape;
                        layer.outputs[0].arguments[0].shape = shape;
                    });
                    register(   21, 'QUANTIZED_TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        const shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.inputs[0].arguments[0].shape = shape;
                        layer.outputs[0].arguments[0].shape = shape;
                    });
                    register( 1000, 'CONV', 'Layer');
                    register( 1001, 'DWCONV', 'Layer');
                    register( 1002, 'QUANTIZED_RESHAPE', 'Shape');
                    register( 1003, 'RESHAPE', 'Shape');
                    register(10240, 'K210_CONV', 'Layer', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.outputs = [ reader.parameter('output') ];
                        const layer_offset = reader.uint32();
                        const weights_offset = reader.uint32();
                        const bn_offset = reader.uint32();
                        const act_offset = reader.uint32();
                        reader.seek(layer_offset);
                        layer.interrupt_enabe = reader.uint64_bits({ int_en: 0, ram_flag: 1, full_add: 2, depth_wise_layer: 3 });
                        layer.inputs = [ reader.parameter('input', 'kpu') ];
                        const outputs = [ reader.parameter('output', 'kpu') ];
                        layer.outputs[0].arguments.push(outputs[0].arguments[0]);
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
                        const filter = [ 1, 3 ][layer.kernel_pool_type_cfg.kernel_type];
                        const weights_shape = layer.interrupt_enabe.depth_wise_layer ? [ oc, filter, filter ] : [ ic, oc, filter, filter ];
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
                            arguments: [ {
                                name: 'const',
                                datatype: 'uint8',
                                shape: weights_shape,
                                data: reader.read(weights_size)
                            } ]
                        });
                        delete layer.kernel_pool_type_cfg.bwsx_base_addr;
                        delete layer.kernel_calc_type_cfg.active_addr;
                        delete layer.kernel_load_cfg.para_start_addr;
                    });
                    register(10241, 'K210_ADD_PADDING', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output', 'kpu') ];
                        layer.channels = reader.uint32();
                    });
                    register(10242, 'K210_REMOVE_PADDING', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.channels = reader.uint32();
                    });
                    register(10243, 'K210_UPLOAD', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output', 'kpu') ];
                        const shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.inputs[0].arguments[0].shape = shape;
                        layer.outputs[0].arguments[0].shape = shape;
                    });
                    for (const layer of this._layers) {
                        const type = types.get(layer.type);
                        if (!type) {
                            throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer type '" + layer.type.toString() + "'.");
                        }
                        if (!type.callback) {
                            throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer '" + type.type.name + "'.");
                        }
                        layer.type = type.type;
                        reader.seek(layer.offset);
                        type.callback(layer, reader);
                        delete layer.offset;
                        delete layer.body_size;
                        // console.log(JSON.stringify(Object.fromEntries(Object.entries(layer).filter((entry) => !(entry[1] instanceof Uint8Array))), null, 2));
                    }
                    if (this._layers.length > 0) {
                        this._layers.unshift({
                            type: { name: 'input' },
                            outputs: [ this._layers[0].inputs[0] ]
                        });
                    }
                    for (const output of outputs) {
                        this._layers.push({
                            type: { name: 'output' },
                            inputs: output.address
                        });
                    }
                    break;
                }
                case 4: {
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
                    reader.memory_type_t = function() {
                        const value = this.uint32();
                        return [ 'const', 'main', 'kpu' ][value];
                    };
                    reader.datatype_t = function() {
                        const value = this.uint32();
                        return [ 'float32', 'uint8' ][value];
                    };
                    reader.memory_range = function() {
                        return {
                            memory_type: this.memory_type_t(),
                            datatype: this.datatype_t(),
                            start: this.uint32(),
                            size: this.uint32()
                        };
                    };
                    reader.argument = function() {
                        const memory = this.memory_range();
                        const value = {
                            name: memory.memory_type + ':' + memory.start.toString(),
                            datatype: memory.datatype
                        };
                        if (memory.memory_type === 'const') {
                            value.data = constants.slice(memory.start, memory.start + memory.size);
                        }
                        return value;
                    };
                    reader.parameter = function(name) {
                        const argument = this.argument();
                        return { name: name, arguments: [ argument ] };
                    };
                    reader.runtime_shape_t = function() {
                        return [ reader.uint32(), reader.uint32(), reader.uint32(), reader.uint32() ];
                    };
                    reader.padding = function() {
                        return { before: reader.int32(), after: reader.int32() };
                    };
                    reader.runtime_paddings_t = function() {
                        return [ this.padding(), this.padding(), this.padding(), this.padding() ];
                    };
                    reader.scalar = function() {
                        return {
                            datatype_t: reader.uint32(),
                            storage: reader.read(4)
                        };
                    };
                    reader.kpu_activate_table_t = function() {
                        const value = {};
                        value.activate_para = new Array(16);
                        for (let i = 0; i < 16; i++) {
                            value.activate_para[i] = this.uint64_bits({ shift_number: 0, y_mul: 8, x_start: 24, reserved: 60 });
                            delete value.activate_para[i].reserved;
                        }
                        for (let i = 0; i < 16; i++) {
                            value.activate_para[i].bias = reader.int8();
                        }
                        return value;
                    };
                    reader.unary_op_t = function() {
                        const value = reader.uint32();
                        return [ 'abs', 'ceil', 'cos', 'exp', 'floor', 'log', 'neg', 'rsqrt', 'sin', 'square' ][value];
                    };
                    reader.binary_op_t = function() {
                        const value = reader.uint32();
                        return [ 'add', 'sub', 'mul', 'div', 'min', 'max' ][value];
                    };
                    reader.reduce_op_t = function() {
                        const value = reader.uint32();
                        return [ 'mean', 'min', 'max', 'sum' ][value];
                    };
                    const inputs = new Array(model_header.inputs);
                    for (let i = 0; i < inputs.length; i++) {
                        inputs[i] = reader.parameter('input' + (i == 0 ? '' : (i + 1).toString()));
                    }
                    for (let i = 0; i < inputs.length; i++) {
                        inputs[i].arguments[0].shape = reader.runtime_shape_t();
                    }
                    const outputs = new Array(model_header.outputs);
                    for (let i = 0; i < outputs.length; i++) {
                        outputs[i] = reader.parameter('output' + (i == 0 ? '' : (i + 1).toString()));
                    }
                    const constants = reader.read(model_header.constants);
                    this._layers = new Array(model_header.nodes);
                    for (let i = 0; i < this._layers.length; i++) {
                        this._layers[i] = {
                            location: i,
                            opcode: reader.uint32(),
                            body_size: reader.uint32()
                        };
                    }
                    let offset = reader.position;
                    for (const layer of this._layers) {
                        layer.offset = offset;
                        offset += layer.body_size;
                    }
                    register(  0x00, 'binary', '');
                    register(  0x01, 'concat', 'Tensor', (layer, reader) => {
                        layer.outputs = [ reader.parameter('output') ];
                        layer.inner_size = reader.uint32();
                        layer.outer_size = reader.uint32();
                        const inputs_count = reader.uint32();
                        layer.inputs = [ { name: 'inputs', arguments: [] } ];
                        for (let i = 0; i < inputs_count; i++) {
                            layer.inputs[0].arguments[i] = reader.argument();
                        }
                        layer.dims = new Array(inputs_count);
                        for (let i = 0; i < inputs_count; i++) {
                            layer.dims[i] = reader.int32();
                        }
                    });
                    register(  0x02, 'conv2d', 'Layer');
                    register(  0x03, 'dequantize', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.zero_point = reader.int32();
                        layer.scale = reader.float32();
                    });
                    register(  0x04, 'matmul', '', (layer, reader) => {
                        layer.inputs = [
                            reader.parameter('a'),
                            reader.parameter('b'),
                        ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.a_rows = reader.int32();
                        layer.a_cols = reader.int32();
                        layer.b_cols = reader.int32();
                        layer.fused_activation = [ reader.float32(), reader.float32() ];
                        const bias = reader.read(4 * layer.b_cols);
                        if (!bias.every((value) => value === 0)) {
                            layer.inputs.push({
                                name: 'bias',
                                arguments: [ { name: 'const', datatype: 'float32', shape: [ layer.b_cols ], data: bias } ]
                            });
                        }
                    });
                    register(  0x05, 'pad', 'Shape', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.inputs[0].arguments[0].shape = reader.runtime_shape_t();
                        layer.paddings = reader.runtime_paddings_t();
                        layer.pad_value = reader.scalar();
                    });
                    register(  0x06, 'quantize', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.zero_point = reader.int32();
                        layer.scale = reader.float32();
                    });
                    register(  0x07, 'reduce', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.reduce_op = reader.reduce_op_t();
                        layer.inputs[0].arguments[0].shape = reader.runtime_shape_t();
                        layer.outputs[0].arguments[0].shape = reader.runtime_shape_t();
                        layer.init_value = reader.float32();
                    });
                    register(  0x08, 'reduce_window2d', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.reduce_op = reader.reduce_op_t();
                        layer.inputs[0].arguments[0].shape = reader.runtime_shape_t();
                        layer.padding_h = reader.padding();
                        layer.padding_w = reader.padding();
                        layer.filter_h = reader.int32();
                        layer.filter_w = reader.int32();
                        layer.stride_h = reader.int32();
                        layer.stride_w = reader.int32();
                        layer.dilation_h = reader.int32();
                        layer.dilation_w = reader.int32();
                        layer.init_value = reader.float32();
                        layer.fused_activation = [ reader.float32(), reader.float32() ];
                    });
                    register(  0x09, 'memory_copy', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                    });
                    register(  0x0A, 'resize_image', '');
                    register(  0x0B, 'softmax', 'Activation');
                    register(  0x0C, 'transpose', 'Transform', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.inputs[0].arguments[0].shape = reader.runtime_shape_t();
                        layer.perm = reader.runtime_shape_t();
                    });
                    register(  0x0D, 'strided_slice', 'Tensor');
                    register(  0x0E, 'unary', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.unary_op = reader.unary_op_t();
                    });
                    register(  0x0F, 'quantized_conv2d', 'Layer');
                    register(  0x10, 'quantized_matmul', '');
                    register(  0x11, 'quantized_binary', '', (layer, reader) => {
                        layer.inputs = [
                            reader.parameter('a'),
                            reader.parameter('b')
                        ];
                        layer.outputs = [ reader.parameter('outputs') ];
                        layer.binary_op = reader.binary_op_t();
                        layer.inputs[0].arguments[0].shape = reader.runtime_shape_t();
                        layer.inputs[1].arguments[0].shape = reader.runtime_shape_t();
                        layer.outputs[0].arguments[0].shape = reader.runtime_shape_t();
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
                    register(  0x12, 'table_lookup1d', '');
                    register(  0x13, 'conv2d_transpose', 'Layer');
                    register(  0x14, 'nnil_unary_method', '');
                    register(0x1001, 'cpu_conv2d', 'Layer');
                    register(0x1002, 'cpu_depthwise_conv2d', 'Layer');
                    register(0x1003, 'cpu_reduce_window2d');
                    register(0x1004, 'cpu_quantized_conv2d', 'Layer');
                    register(0x1005, 'cpu_quantized_depthwise_conv2d', 'Layer');
                    register(0x2001, 'kpu_upload', '', (layer, reader) => {
                        layer.inputs = [ reader.parameter('input') ];
                        layer.outputs = [ reader.parameter('output') ];
                        layer.inputs[0].arguments[0].shape = reader.runtime_shape_t();
                    });
                    register(0x2002, 'kpu_conv2d', 'Layer', (layer, reader) => {
                        layer.outputs = [ reader.parameter('output') ];
                        layer.batches = reader.int32();
                        layer.reserved0 = reader.int32();
                        layer.interrupt_enabe = reader.uint64_bits({ int_en: 0, ram_flag: 1, full_add: 2, depth_wise_layer: 3 });
                        const image_src_addr = reader.uint32();
                        const image_dst_addr = reader.uint32();
                        layer.inputs = [ { name: 'input', arguments: [ { name: 'kpu:' + image_src_addr.toString() } ] } ];
                        const outputs = [ { name: 'output', arguments: [ { name: 'kpu:' + image_dst_addr.toString() } ] } ];
                        layer.outputs[0].arguments.push(outputs[0].arguments[0]);
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
                        const filter = [ 1, 3 ][layer.kernel_pool_type_cfg.kernel_type];
                        const weights_shape = layer.interrupt_enabe.depth_wise_layer ? [ oc, filter, filter ] : [ ic, oc, filter, filter ];
                        const weights_size = weights_shape.reduce((a, b) => a * b);
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
                        layer.inputs.push({
                            name: 'weights',
                            arguments: [ {
                                name: 'const',
                                datatype: 'uint8',
                                shape: weights_shape,
                                data: reader.read(weights_size)
                            } ]
                        });
                    });
                    for (const layer of this._layers) {
                        const type = types.get(layer.opcode);
                        if (!type) {
                            throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer type '" + layer.type.toString() + "'.");
                        }
                        if (!type.callback) {
                            throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer '" + type.type.name + "'.");
                        }
                        layer.type = type.type;
                        reader.seek(layer.offset);
                        if (type.callback) {
                            type.callback(layer, reader);
                        }
                        delete layer.offset;
                        delete layer.body_size;
                        if (reader.position != (layer.offset + layer.body_size)) {
                            // debugger;
                        }
                        // console.log(JSON.stringify(Object.fromEntries(Object.entries(layer).filter((entry) => !(entry[1] instanceof Uint8Array))), null, 2));
                        delete layer.opcode;
                    }
                    for (const input of inputs) {
                        this._layers.unshift({
                            type: { name: 'INPUT' },
                            outputs: [ input ]
                        });
                    }
                    for (const output of outputs) {
                        this._layers.push({
                            type: { name: 'OUTPUT' },
                            inputs: [ output ]
                        });
                    }
                    break;
                }
                case 5: {
                    reader.model_header = function() {
                        return {
                            header_size: reader.uint32(),
                            flags: reader.uint32(),
                            alignment: reader.uint32(),
                            modules: reader.uint32(),
                            entry_module: reader.uint32(),
                            entry_function: reader.uint32()
                        };
                    };
                    reader.module_type_t = function() {
                        const buffer = reader.read(16);
                        return new TextDecoder('ascii').decode(buffer);
                    };
                    reader.module_header = function() {
                        return {
                            type: reader.module_type_t(),
                            version: reader.uint32(),
                            header_size: reader.uint32(),
                            size: reader.uint32(),
                            mempools: reader.uint32(),
                            shared_mempools: reader.uint32(),
                            sections: reader.uint32(),
                            functions: reader.uint32(),
                            reserved0: reader.uint32()
                        };
                    };
                    reader.mempool_desc = function() {
                        return {
                            location: reader.byte(),
                            reserved0: reader.read(3),
                            size: reader.uint32()
                        };
                    };
                    reader.section_header = function() {
                        return {
                            name: new TextDecoder('ascii').decode(reader.read(16)),
                            flags: reader.uint32(),
                            body_start: reader.uint32(),
                            body_size: reader.uint32(),
                            reserved0: reader.uint32()
                        };
                    };
                    reader.function_header = function() {
                        return {
                            header_size: reader.uint32(),
                            size: reader.uint32(),
                            input_pool_size: reader.uint32(),
                            output_pool_size: reader.uint32(),
                            inputs: reader.uint32(),
                            outputs: reader.uint32(),
                            entrypoint: reader.uint32(),
                            text_size: reader.uint32()
                        };
                    };
                    reader.memory_range = function() {
                        return {
                            memory_type: this.byte(), // 0=const, 1=main, 2=k210_kpu
                            datatype: this.byte(),
                            shared_module: this.uint16(),
                            start: this.uint32(),
                            size: this.uint32()
                        };
                    };
                    reader.shape = function() {
                        const array = new Array(reader.uint32());
                        for (let i = 0; i < array.length; i++) {
                            array[i] = reader.uint32();
                        }
                        return array;
                    };
                    reader.align_position = function(alignment) {
                        const remainder = this._position % alignment;
                        if (remainder !== 0) {
                            this.skip(alignment - remainder);
                        }
                    };
                    const model_header = reader.model_header();
                    if (model_header.header_size > reader.position) {
                        reader.skip(model_header.header_size - reader.position);
                    }
                    this._modules = new Array(model_header.modules);
                    for (let i = 0; i < this._modules.length; i++) {
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
                        const functions = new Array(module_header.functions);
                        for (let i = 0; i < functions.length; i++) {
                            const function_header = reader.function_header();
                            const inputs = new Array(function_header.inputs);
                            for (let i = 0; i < inputs.length; i++) {
                                inputs[i] = reader.memory_range();
                            }
                            for (let i = 0; i < inputs.length; i++) {
                                inputs[i].shape = reader.shape();
                            }
                            const outputs = new Array(function_header.outputs);
                            for (let i = 0; i < outputs.length; i++) {
                                outputs[i] = reader.memory_range();
                            }
                            for (let i = 0; i < outputs.length; i++) {
                                outputs[i].shape = reader.shape();
                            }
                            reader.align_position(8);
                        }
                        const sections = new Array(module_header.sections);
                        for (let i = 0; i < sections.length; i++) {
                            sections[i] = reader.section_header();
                        }
                    }
                    throw new kmodel.Error("Unsupported model version '" + this.version.toString() + "'.");
                }
                default: {
                    throw new kmodel.Error("Unsupported model version '" + this.version.toString() + "'.");
                }
            }
            delete this._reader;
        }
    }
};

kmodel.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading kmodel.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = kmodel.ModelFactory;
}