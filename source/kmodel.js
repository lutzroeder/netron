
var kmodel = kmodel || {};
var base = base || require('./base');

kmodel.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signatures = [
            { identifer: [ 0x03, 0x00, 0x00, 0x00 ], match: 'kmodel.KPU' },
            { identifer: [ 0x4C, 0x44, 0x4D, 0x4B ], match: 'kmodel.LDMK'}
        ];
        const signature = signatures.find((signature) => signature.identifer.length <= stream.length && stream.peek(signature.identifer.length).every((value, index) => signature.identifer[index] === undefined || signature.identifer[index] === value));
        if (signature) {
            return signature.match;
        }
        return undefined;
    }

    open(context, match) {
        return Promise.resolve().then(() => {
            const stream = context.stream;
            switch (match) {
                case 'kmodel.KPU': {
                    return new kmodel.Model(new kmodel.KPU(stream));
                }
                case 'kmodel.LDMK': {
                    return new kmodel.Model(new kmodel.LDMK(stream));
                }
            }
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
        this._type = { name: layer.typename, category: layer.category };
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
        return [];
    }
};


kmodel.KPU = class {

    constructor(stream) {
        const reader = new base.BinaryReader(stream);
        this.version = reader.uint32();
        /* const flags = */ reader.uint32();
        /* const arch = */ reader.uint32();
        this.layers = new Array(reader.uint32());
        /* const max_start_address = */ reader.uint32();
        /* const main_mem_usage = */ reader.uint32();
        const outputs = new Array(reader.uint32());
        for (let i = 0; i < outputs.length; i++) {
            outputs[i] = {
                address: reader.uint32(),
                size: reader.uint32()
            };
        }
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i] = {
                location: i,
                type: reader.uint32(),
                body_size: reader.uint32()
            };
        }
        let offset = reader.position;
        for (const layer of this.layers) {
            layer.offset = offset;
            offset += layer.body_size;
            // layer.body = reader.read(layer.body_size);
            // delete layer.body_size;
        }
        const types = new Map();
        const register = (type, name, category, callback) => {
            types.set(type, { name: name, category: category || '', callback: callback || function() {} });
        };
        register(  -1, 'DUMMY');
        register(   0, 'INVALID');
        register(   1, 'ADD');
        register(   2, 'QUANTIZED_ADD');
        register(   3, 'GLOBAL_MAX_POOL2D', 'Pool');
        register(   4, 'QUANTIZED_GLOBAL_MAX_POOL2D', 'Pool');
        register(   5, 'GLOBAL_AVERAGE_POOL2D', 'Pool');
        register(   6, 'QUANTIZED_GLOBAL_AVERAGE_POOL2D', 'Pool');
        register(   7, 'MAX_POOL2D', 'Pool');
        register(   8, 'QUANTIZED_MAX_POOL2D', 'Pool');
        register(   9, 'AVERAGE_POOL2D', 'Pool');
        register(   10, 'QUANTIZED_AVERAGE_POOL2D', 'Pool');
        register(   11, 'QUANTIZE');
        register(   12, 'DEQUANTIZE');
        register(   13, 'REQUANTIZE');
        register(   14, 'L2_NORMALIZATION', 'Normalization');
        register(   15, 'SOFTMAX', 'Activation');
        register(   16, 'CONCAT', 'Tensor');
        register(   17, 'QUANTIZED_CONCAT', 'Tensor');
        register(   18, 'FULLY_CONNECTED', 'Layer');
        register(   19, 'QUANTIZED_FULLY_CONNECTED', 'Layer');
        register(   20, 'TENSORFLOW_FLATTEN');
        register(   21, 'QUANTIZED_TENSORFLOW_FLATTEN');
        register( 1000, 'CONV', 'Layer');
        register( 1001, 'DWCONV', 'Layer');
        register( 1002, 'QUANTIZED_RESHAPE', 'Shape');
        register( 1003, 'RESHAPE', 'Shape');
        register(10240, 'K210_CONV', 'Layer', (/* layer, reader */) => {
            /*
            const flags = reader.uint32();
            const main_mem_out_address = reader.uint32();
            const layer_offset = reader.uint32();
            const weights_offset = reader.uint32();
            const bn_offset = reader.uint32();
            const act_offset = reader.uint32();
            */
        });
        register(10241, 'K210_ADD_PADDING');
        register(10242, 'K210_REMOVE_PADDING');
        register(10243, 'K210_UPLOAD');
        for (const layer of this.layers) {
            const type = types.get(layer.type);
            if (!type || !type.callback) {
                throw new kmodel.Error("Unsupported layer type '" + layer.type.toString() + "'.");
            }
            layer.typename = type.name;
            layer.category = type.category;
            reader.seek(layer.offset);
            type.callback(layer, reader);
            // delete layer.offset;
            // delete layer.body_size;
        }
    }
};

kmodel.LDMK = class {

    constructor(stream) {
        const reader = new base.BinaryReader(stream);
        /* const identifier = */ reader.uint32();
        this.version = reader.uint32();
        if (this.version > 5) {
            throw new kmodel.Error("Unsupported LDMK model version '" + this.version.toString() + "'.");
        }
        const header_size = reader.uint32();
        /* const flags = */ reader.uint32();
        /* const alignment = */ reader.uint32();
        this.modules = new Array(reader.uint32());
        /* const entry_module = */ reader.uint32();
        /* const entry_function = */ reader.uint32();
        if (header_size > reader.position) {
            reader.skip(header_size - reader.position);
        }
        for (let i = 0; i < this.modules.length; i++) {
            /*
            char[16] type;
            uint32_t version;
            uint32_t header_size;
            uint32_t size;
            uint32_t mempools;
            uint32_t shared_mempools;
            uint32_t sections;
            uint32_t functions;
            uint32_t reserved0;
            */
        }
        throw new kmodel.Error('kmodel.LDMK not supported.');
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
