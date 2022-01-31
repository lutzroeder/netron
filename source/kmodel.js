
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
        this._nodes = model.layers.map((layer) => new kmodel.Node(layer));
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
        return 'Tensor data not implemented.';
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
            if (name === 'type' || name === 'location' || name === 'inputs' || name === 'outputs') {
                continue;
            }
            const value = entry[1];
            const attribute = new kmodel.Attribute(name, value);
            this._attributes.push(attribute);
        }
        for (const input of layer.inputs || []) {
            this._inputs.push(new kmodel.Parameter(input.name, input.arguments.map((argument) => {
                const type = argument.shape ? new kmodel.TensorType(argument.data_type || '?', argument.shape) : null;
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
                    reader.kpu_model_output_t = function() {
                        return {
                            address: reader.uint32(),
                            size: reader.uint32()
                        };
                    };
                    reader.kpu_model_layer_header_t = function() {
                        return {
                            type: reader.uint32(),
                            body_size: reader.uint32()
                        };
                    };
                    reader.mem_address = function(memory, name) {
                        const mem_address = this.uint32();
                        const argument = { name: memory + ':' + mem_address.toString() };
                        const parameter = { name: name, arguments: [ argument ] };
                        return [ parameter ];
                    };
                    reader.kpu_layer_config_field = function(fields) {
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
                    const model_header = reader.kpu_model_header_t();
                    this._layers = new Array(model_header.layers_length);
                    this._outputs = new Array(model_header.output_count);
                    for (let i = 0; i < this._outputs.length; i++) {
                        this._outputs[i] = reader.kpu_model_output_t();
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
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
                        layer.kernel_size = reader.uint32();
                        layer.channels = reader.uint32();
                    });
                    register(    6, 'QUANTIZED_GLOBAL_AVERAGE_POOL2D', 'Pool');
                    register(    7, 'MAX_POOL2D', 'Pool');
                    register(    8, 'QUANTIZED_MAX_POOL2D', 'Pool', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
                        layer.in_shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.out_shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.kernel = [ reader.uint32(), reader.uint32() ];
                        layer.stride = [ reader.uint32(), reader.uint32() ];
                        layer.padding = [ reader.uint32(), reader.uint32() ];
                    });
                    register(    9, 'AVERAGE_POOL2D', 'Pool');
                    register(   10, 'QUANTIZED_AVERAGE_POOL2D', 'Pool');
                    register(   11, 'QUANTIZE', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
                        layer.count = reader.uint32();
                        layer.scale = reader.float32();
                        layer.bias = reader.float32();
                    });
                    register(   12, 'DEQUANTIZE', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
                        layer.count = reader.uint32();
                        layer.scale = reader.float32();
                        layer.bias = reader.float32();
                    });
                    register(   13, 'REQUANTIZE');
                    register(   14, 'L2_NORMALIZATION', 'Normalization');
                    register(   15, 'SOFTMAX', 'Activation', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
                        layer.channels = reader.uint32();
                    });
                    register(   16, 'CONCAT', 'Tensor', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.outputs = reader.mem_address('main', 'outputs');
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
                        layer.outputs = reader.mem_address('main', 'outputs');
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
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
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
                        layer.inputs.push({ name: 'weights', arguments: [ { name: '', data_type: 'float32', shape: [ layer.in_channels, layer.out_channels ], data: reader.read(4 * layer.in_channels * layer.out_channels) } ] });
                        layer.inputs.push({ name: 'bias', arguments: [ { name: '', data_type: 'float32', shape: [ layer.out_channels ], data: reader.read(4 * layer.out_channels) } ] });
                    });
                    register(   19, 'QUANTIZED_FULLY_CONNECTED', 'Layer');
                    register(   20, 'TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
                        const shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                        layer.inputs[0].arguments[0].shape = shape;
                        layer.outputs[0].arguments[0].shape = shape;
                    });
                    register(   21, 'QUANTIZED_TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('main', 'outputs');
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
                        layer.outputs = reader.mem_address('main', 'outputs');
                        const layer_offset = reader.uint32();
                        const weights_offset = reader.uint32();
                        const bn_offset = reader.uint32();
                        const act_offset = reader.uint32();
                        reader.seek(layer_offset);
                        layer.interrupt_enabe = reader.kpu_layer_config_field({ int_en: 0, ram_flag: 1, full_add: 2, depth_wise_layer: 3 });
                        layer.inputs = reader.mem_address('kpu', 'inputs');
                        const outputs = reader.mem_address('kpu', 'outputs');
                        layer.outputs = layer.flags & 1 ? layer.outputs : outputs;
                        layer.image_channel_num = reader.kpu_layer_config_field({ i_ch_num: 0, o_ch_num: 32, o_ch_num_coef: 48 });
                        layer.image_size =  reader.kpu_layer_config_field({ i_row_wid: 0, i_col_high: 10, o_row_wid: 32, o_col_high : 42 });
                        layer.kernel_pool_type_cfg = reader.kpu_layer_config_field({ kernel_type: 0, pad_type: 3, pool_type: 4, first_stride: 8, bypass_conv: 9, load_para: 10, dma_burst_size: 16, pad_value: 24, bwsx_base_addr: 32 });
                        layer.kernel_load_cfg = reader.kpu_layer_config_field({ load_coor: 0, load_time: 1, para_size: 15, para_start_addr: 32 });
                        layer.kernel_offset = reader.kpu_layer_config_field({ coef_column_offset: 0, coef_row_offset: 4 });
                        layer.kernel_calc_type_cfg = reader.kpu_layer_config_field({ channel_switch_addr: 0, row_switch_addr: 16, coef_size: 20, coef_group: 28, load_act: 31, active_addr: 32 });
                        layer.write_back_cfg = reader.kpu_layer_config_field({ wb_channel_switch_addr: 0, wb_row_switch_addr: 16, wb_group: 20 });
                        layer.conv_value = reader.kpu_layer_config_field({ shr_w: 0, shr_x: 4, arg_w: 8, arg_x: 32 });
                        layer.conv_value2 = reader.kpu_layer_config_field({ arg_add: 0 });
                        layer.dma_parameter = reader.kpu_layer_config_field({ send_data_out: 0, channel_byte_num: 16, dma_total_byte: 32 });
                        reader.seek(weights_offset);
                        reader.seek(bn_offset);
                        reader.seek(act_offset);
                    });
                    register(10241, 'K210_ADD_PADDING', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('kpu', 'outputs');
                        layer.channels = reader.uint32();
                    });
                    register(10242, 'K210_REMOVE_PADDING', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address(layer.flags & 1 ? 'main' : 'kpu', 'outputs');
                        layer.channels = reader.uint32();
                    });
                    register(10243, 'K210_UPLOAD', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.inputs = reader.mem_address('main', 'inputs');
                        layer.outputs = reader.mem_address('kpu', 'outputs');
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
                            throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer '" + type.name + "'.");
                        }
                        layer.type = type.type;
                        reader.seek(layer.offset);
                        type.callback(layer, reader);
                        delete layer.offset;
                        delete layer.body_size;
                        // console.log(JSON.stringify(Object.fromEntries(Object.entries(layer).filter((entry) => !(entry[1] instanceof Uint8Array))), null, 2));
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
                    reader.memory_range = function() {
                        return {
                            memory_type: this.uint32(), // 0=const, 1=main, 2=k210_kpu
                            datatype: this.uint32(), // 0=float32, 1=uint8
                            start: this.uint32(),
                            size: this.uint32()
                        };
                    };
                    reader.runtime_shape_t = function() {
                        return [ reader.uint32(), reader.uint32(), reader.uint32(), reader.uint32() ];
                    };
                    this._inputs = new Array(model_header.inputs);
                    for (let i = 0; i < this._inputs.length; i++) {
                        this._inputs[i] = reader.memory_range();
                    }
                    for (let i = 0; i < this._inputs.length; i++) {
                        this._inputs[i].shape = reader.runtime_shape_t();
                    }
                    this._outputs = new Array(model_header.outputs);
                    for (let i = 0; i < this._outputs.length; i++) {
                        this._outputs[i] = reader.memory_range();
                    }
                    this._constants = reader.read(model_header.constants);
                    this._layers = new Array(model_header.nodes);
                    for (let i = 0; i < this._layers.length; i++) {
                        this._layers[i] = {
                            location: i,
                            op_code: reader.uint32(),
                            body_size: reader.uint32()
                        };
                    }
                    let offset = reader.position;
                    for (const layer of this._layers) {
                        layer.offset = offset;
                        offset += layer.body_size;
                    }

                    register(  0x00, 'binary', '');
                    register(  0x01, 'concat', 'Tensor');
                    register(  0x02, 'conv2d', 'Layer');
                    register(  0x03, 'dequantize', '');
                    register(  0x04, 'matmul', '');
                    register(  0x05, 'pad', 'Shape');
                    register(  0x06, 'quantize', '');
                    register(  0x07, 'reduce', '');
                    register(  0x08, 'reduce_window2d');
                    register(  0x09, 'memory_copy', '');
                    register(  0x0A, 'resize_image', '');
                    register(  0x0B, 'softmax', 'Activation');
                    register(  0x0C, 'transpose', 'Transform');
                    register(  0x0D, 'strided_slice', 'Tensor');
                    register(  0x0E, 'unary', '');
                    register(  0x0F, 'quantized_conv2d', 'Layer');
                    register(  0x10, 'quantized_matmul', '');
                    register(  0x11, 'quantized_binary', '');
                    register(  0x12, 'table_lookup1d', '');
                    register(  0x13, 'conv2d_transpose', 'Layer');
                    register(  0x14, 'nnil_unary_method', '');
                    register(0x1001, 'cpu_conv2d', 'Layer');
                    register(0x1002, 'cpu_depthwise_conv2d', 'Layer');
                    register(0x1003, 'cpu_reduce_window2d');
                    register(0x1004, 'cpu_quantized_conv2d', 'Layer');
                    register(0x1005, 'cpu_quantized_depthwise_conv2d', 'Layer');
                    register(0x2001, 'kpu_upload', '');
                    register(0x2002, 'kpu_conv2d', 'Layer');
                    for (const layer of this._layers) {
                        const type = types.get(layer.op_code);
                        if (!type) {
                            throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer type '" + layer.type.toString() + "'.");
                        }
                        if (!type.callback) {
                            // throw new kmodel.Error("Unsupported version '" + this._version.toString() + "' layer '" + type.name + "'.");
                        }
                        layer.type = type.type;
                        reader.seek(layer.offset);
                        // type.callback(layer, reader);
                        delete layer.offset;
                        delete layer.body_size;
                        if (reader.position != (layer.offset + layer.body_size)) {
                            // debugger;
                        }
                        // console.log(JSON.stringify(Object.fromEntries(Object.entries(layer).filter((entry) => !(entry[1] instanceof Uint8Array))), null, 2));
                        delete layer.op_code;
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