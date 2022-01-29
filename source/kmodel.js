
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

kmodel.Node = class {

    constructor(layer) {
        this._location = layer.location;
        this._type = layer.type;
        this._attributes = [];
        for (const entry of Object.entries(layer)) {
            const name = entry[0];
            if (name === 'type' || name === 'location' || name === 'params') {
                continue;
            }
            const value = entry[1];
            const attribute = new kmodel.Attribute(name, value);
            this._attributes.push(attribute);
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
        return [];
    }

    get outputs() {
        return [];
    }

    get attributes() {
        return this._attributes;
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
                    const model_header = {
                        flags: reader.uint32(),
                        arch: reader.uint32(),
                        layers_length: reader.uint32(),
                        max_start_address: reader.uint32(),
                        main_mem_usage: reader.uint32(),
                        output_count: reader.uint32()
                    };
                    this._layers = new Array(model_header.layers_length);
                    this._outputs = new Array(model_header.output_count);
                    for (let i = 0; i < this._outputs.length; i++) {
                        this._outputs[i] = {
                            address: reader.uint32(),
                            size: reader.uint32()
                        };
                    }
                    for (let i = 0; i < this._layers.length; i++) {
                        this._layers[i] = {
                            location: i,
                            type: reader.uint32(),
                            body_size: reader.uint32()
                        };
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
                        layer.main_mem_in_address = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
                        layer.kernel_size = reader.uint32();
                        layer.channels = reader.uint32();
                    });
                    register(    6, 'QUANTIZED_GLOBAL_AVERAGE_POOL2D', 'Pool');
                    register(    7, 'MAX_POOL2D', 'Pool');
                    register(    8, 'QUANTIZED_MAX_POOL2D', 'Pool', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
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
                        layer.main_mem_in_address = reader.uint32();
                        layer.mem_out_address = reader.uint32();
                        layer.count = reader.uint32();
                        layer.scale = reader.float32();
                        layer.bias = reader.float32();
                    });
                    register(   12, 'DEQUANTIZE', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.mem_out_address = reader.uint32();
                        layer.count = reader.uint32();
                        layer.scale = reader.float32();
                        layer.bias = reader.float32();
                    });
                    register(   13, 'REQUANTIZE');
                    register(   14, 'L2_NORMALIZATION', 'Normalization');
                    register(   15, 'SOFTMAX', 'Activation', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
                        layer.channels = reader.uint32();
                    });
                    register(   16, 'CONCAT', 'Tensor', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
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
                        layer.main_mem_out_address = reader.uint32();
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
                        layer.main_mem_in_address = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
                        layer.in_channels = reader.uint32();
                        layer.out_channels = reader.uint32();
                        layer.act = reader.uint32(); // {'linear':0, 'relu':1, 'relu6':2}
                        layer.params = {
                            weights: reader.read(4 * layer.in_channels * layer.out_channels),
                            bias: reader.read(4 * layer.out_channels)
                        };
                    });
                    register(   19, 'QUANTIZED_FULLY_CONNECTED', 'Layer');
                    register(   20, 'TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
                        layer.shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                    });
                    register(   21, 'QUANTIZED_TENSORFLOW_FLATTEN', 'Shape', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
                        layer.shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
                    });
                    register( 1000, 'CONV', 'Layer');
                    register( 1001, 'DWCONV', 'Layer');
                    register( 1002, 'QUANTIZED_RESHAPE', 'Shape');
                    register( 1003, 'RESHAPE', 'Shape');
                    register(10240, 'K210_CONV', 'Layer', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_out_address = reader.uint32();
                        /* const layer_offset = */ reader.uint32();
                        /* const weights_offset = */ reader.uint32();
                        /* const bn_offset = */ reader.uint32();
                        /* const act_offset = */ reader.uint32();
                    });
                    register(10241, 'K210_ADD_PADDING', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.kpu_mem_out_address = reader.uint32();
                        layer.channels = reader.uint32();
                    });
                    register(10242, 'K210_REMOVE_PADDING', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.kpu_mem_out_address = reader.uint32();
                        layer.channels = reader.uint32();
                    });
                    register(10243, 'K210_UPLOAD', '', (layer, reader) => {
                        layer.flags = reader.uint32();
                        layer.main_mem_in_address = reader.uint32();
                        layer.kpu_mem_out_address = reader.uint32();
                        layer.shape = [ reader.uint32(), reader.uint32(), reader.uint32() ];
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
                        if (reader.position != (layer.offset + layer.body_size)) {
                            // debugger;
                        }
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