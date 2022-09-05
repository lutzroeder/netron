
// Experimental

var om = om || {};
var protobuf = protobuf || require('./protobuf');
var base = base || require('./base');

om.ModelFactory = class {

    match(context) {
        return om.File.open(context);
    }

    open(context, match) {
        const file = match;
        if (!file.model) {
            throw new om.Error('File does not contain a model definition.');
        }
        return context.require('./om-proto').then(() => {
            let model = null;
            try {
                om.proto = protobuf.get('om').ge.proto;
                const reader = protobuf.BinaryReader.open(file.model);
                model = om.proto.ModelDef.decode(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new om.Error('File format is not ge.proto.ModelDef (' + message.replace(/\.$/, '') + ').');
            }
            return context.metadata('om-metadata.json').then((metadata) => {
                return new om.Model(metadata, model, file.weights);
            });
        });
    }
};

om.Model = class {

    constructor(metadata, model, weights) {
        this._graphs = [];
        const context = { metadata: metadata, weights: weights };
        for (const graph of model.graph) {
            this._graphs.push(new om.Graph(context, graph));
        }
    }

    get format() {
        return 'DaVinci OM';
    }

    get graphs() {
        return this._graphs;
    }
};

om.Graph = class {

    constructor(context, graph) {
        this._name = graph.name;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        for (const op of graph.op) {
            if (op.type === 'Const') {
                continue;
            }
            const node = new om.Node(context, op, graph);
            this._nodes.push(node);
        }
    }

    get name() {
        return this._name;
    }

    get nodes() {
        return this._nodes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

};

om.Node = class {

    constructor(context, op, graph) {
        this._name = op.name;
        this._type = context.metadata.type(op.type) || { name: op.type };
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._chain = [];
        this._controlDependencies = [];
        this._device = null;
        if (op.input) {
            for (let i = 0; i < op.input.length; i++) {
                if (op.input[i] === '') {
                    continue;
                }
                const pos = op.input[i].lastIndexOf(':');
                const name = pos === 0 ? 'internal_unnamed' : op.input[i].slice(0, pos);
                const src_index = op.input[i].slice(pos + 1);
                if (src_index === '-1') {
                    this._controlDependencies.push(new om.Argument(name));
                    continue;
                }
                const parameterName = this._type.inputs && i < this._type.inputs.length ? this._type.inputs[i].name : 'input' + (i === 0 ? '' : i.toString());
                const inputNode = graph.op.find(node => node.name === name);
                const desc = op.input_desc[i];
                const format = desc.layout;
                if (inputNode && inputNode.type === 'Const' && inputNode.attr && inputNode.attr.value && inputNode.attr) {
                    let shape = null;
                    const value = inputNode.attr.value.t;
                    if (value.desc.shape != null) {
                        shape = value.desc.shape.dim;
                    }
                    else if (value.desc.attr.origin_shape) {
                        shape = value.desc.attr.origin_shape.list.i;
                    }
                    let data = null;
                    if (value.data.length === 0) {
                        if (context.weights == null) {
                            data = null;
                        }
                        else if (value.desc.attr.merged_offset) {
                            const offset = value.desc.attr.merged_offset.i;
                            data = context.weights.slice(offset, offset + value.desc.weight_size);
                        }
                        else {
                            const offset = value.desc.data_offset;
                            data = context.weights.slice(offset, offset + value.desc.weight_size);
                        }
                    }
                    else {
                        data = value.data;
                    }
                    const dataType = om.Utility.dtype(value.desc.dtype);
                    const tensorType = new om.TensorType(dataType, shape, format, value.desc.layout);
                    const tensor = new om.Tensor('Constant', tensorType, data);
                    const argument = new om.Argument(name, null, tensor);
                    this._inputs.push(new om.Parameter(parameterName, true, [ argument ]));
                }
                else {
                    const dataType = desc ? om.Utility.dtype(desc.dtype) : '?';
                    const shape = desc.shape ? desc.shape.dim : undefined;
                    const tensorType = new om.TensorType(dataType, shape, format, null);
                    const identifier = src_index === '0' ? name : name + ':' + src_index;
                    const argument = new om.Argument(identifier, tensorType, null);
                    this._inputs.push(new om.Parameter(parameterName, true, [ argument ]));
                }
            }
        }
        if (op.output_desc) {
            for (let i = 0; i < op.output_desc.length; i++) {
                const desc = op.output_desc[i];
                let shape = desc.shape ? desc.shape.dim : undefined;
                if (op.type === 'Data' || op.type === 'ImageData' || op.type === 'DynamicImageData') {
                    shape = desc.shape ? desc.shape.dim : op.input_desc[0].shape.dim;
                }
                const dataType = om.Utility.dtype(desc.dtype);
                const format = desc.layout;
                const tensorType = new om.TensorType(dataType, shape, format);
                const identifier = i === 0 ? this._name : this._name + ':' + i;
                const argument = new om.Argument(identifier, tensorType, null);
                const outputName = this._type.outputs && i < this._type.outputs.length ? this._type.outputs[i].name : 'output' + (i === 0 ? '' : i.toString());
                this._outputs.push(new om.Parameter(outputName, true, [ argument ]));
            }
        }
        if (op.attr) {
            for (const attr of Object.entries(op.attr)) {
                const name = attr[0];
                const value = attr[1];
                if (name === 'device') {
                    this._device = value;
                    continue;
                }
                if (name === 'original_op_names') {
                    continue;
                }
                if (name === 'relu_flag' && value.b) {
                    this._chain.push(new om.Node(context, { type: 'ReLU' }, graph));
                    continue;
                }
                const attribute = new om.Attribute(context, name, value);
                this._attributes.push(attribute);
            }
        }
    }

    get device() {
        return this._device;
    }

    get name() {
        return this._name || '';
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

    get controlDependencies() {
        return this._controlDependencies;
    }
};

om.Attribute = class {

    constructor(context, name, value) {
        this._name = name;
        this._value = value;
        switch (value.value) {
            case 'i': {
                this._value = value.i;
                this._type = 'int64';
                break;
            }
            case 'f': {
                this._value = value.f;
                this._type = 'float32';
                break;
            }
            case 'b': {
                this._value = value.b;
                this._type = 'boolean';
                break;
            }
            case 'bt': {
                this._value = null;
                if (value.bt.length !== 0) {
                    this._type = 'tensor';
                    this._value = new om.Tensor('Constant', new om.TensorType('float32', [ value.bt.length / 4 ], null), value.bt);
                }
                break;
            }
            case 'dt': {
                this._type = 'DataType';
                this._value = om.Utility.dtype(value.dt.toNumber());
                break;
            }
            case 's': {
                if (typeof value.s === 'string') {
                    this._value = value.s;
                }
                else if (value.s.filter(c => c <= 32 && c >= 128).length === 0) {
                    this._value = om.Utility.decodeText(value.s);
                }
                else {
                    this._value = value.s;
                }
                this._type = 'string';
                break;
            }
            case 'g': {
                this._type = 'graph';
                this._value = new om.Graph(context, value.g);
                break;
            }
            case 'func': {
                break;
            }
            case 'list': {
                const list = value.list;
                this._value = [];
                if (list.s && list.s.length > 0) {
                    this._value = list.s.map(v => String.fromCharCode.apply(null, new Uint16Array(v))).join(', ');
                    this._type = 'string[]';
                }
                else if (list.b && list.b.length > 0) {
                    this._value = list.b;
                    this._type = 'boolean[]';
                }
                else if (list.i && list.i.length > 0) {
                    this._value = list.i;
                    this._type = 'int64[]';
                }
                else if (list.f && list.f.length > 0) {
                    this._value = list.f;
                    this._type = 'float32[]';
                }
                else if (list.type && list.type.length > 0) {
                    this._type = 'type[]';
                    this._value = list.type.map((type) => om.Node.enum2Dtype(type) || '?');
                }
                else if (list.shape && list.shape.length > 0) {
                    this._type = 'shape[]';
                    this._value = list.shape.map((shape) => new om.TensorShape(shape));
                }
                break;
            }
            case undefined: {
                this._value = null;
                break;
            }
            default: {
                throw new om.Error("Unsupported attribute type '" + JSON.stringify(value).substring(0, 32) + "'.");
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return true;
    }
};

om.Parameter = class {

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

om.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new om.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

om.Tensor = class {

    constructor(category, type, value) {
        this._type = type;
        this._category = category;
        this._data = value;
    }

    get category() {
        return this._category;
    }

    get type() {
        return this._type;
    }
};

om.TensorType = class {

    constructor(dataType, shape, format, denotation) {
        this._dataType = dataType;
        this._shape = new om.TensorShape(shape);
        const list = [];
        if (format) {
            list.push(format);
        }
        if (denotation && denotation !== format) {
            list.push(denotation);
        }
        this._denotation = list.join(' ');
    }

    get dataType() {
        return this._dataType;
    }

    set shape(dims) {
        this._shape = dims;
    }

    get shape() {
        return this._shape;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

om.TensorShape = class {

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

om.File = class {

    static open(context) {
        const stream = context.stream;
        const signature = [ 0x49, 0x4D, 0x4F, 0x44 ]; // IMOD
        if (stream && stream.length >= 256 && stream.peek(4).every((value, index) => value === signature[index])) {
            const reader = new base.BinaryReader(stream);
            return new om.File(reader);
        }
        return null;
    }

    constructor(reader) {
        this._reader = reader;
    }

    get model() {
        this._read();
        return this._model;
    }

    get weights() {
        this._read();
        return this._weights;
    }

    _read() {
        if (this._reader) {
            const reader = this._reader;
            delete this._reader;
            const decoder = new TextDecoder('utf-8');
            this.header = reader.uint32();
            const size = reader.uint32();
            this.version = reader.uint32();
            this.checksum = reader.read(64);
            reader.skip(4);
            this.is_encrypt = reader.byte();
            this.is_checksum = reader.byte();
            this.type = reader.byte(); // 0=IR model, 1=standard model, 2=OM Tiny model
            this.mode = reader.byte(); // 0=offline, 1=online
            this.name = decoder.decode(reader.read(32));
            this.ops = reader.uint32();
            this.userdefineinfo = reader.read(32);
            this.ir_version = reader.uint32();
            this.model_num = reader.uint32();
            this.platform_version = reader.read(20);
            this.platform_type = reader.byte();
            reader.seek(0);
            reader.skip(size);
            const partitions = new Array(reader.uint32());
            for (let i = 0; i < partitions.length; i++) {
                partitions[i] = {
                    type: reader.uint32(),
                    offset: reader.uint32(),
                    size: reader.uint32()
                };
            }
            const offset = 256 + 4 + 12 * partitions.length;
            for (const partition of partitions) {
                reader.seek(offset + partition.offset);
                const buffer = reader.read(partition.size);
                switch (partition.type) {
                    case 0: { // MODEL_DEF
                        this._model = buffer;
                        break;
                    }
                    case 1: { // WEIGHTS_DATA
                        this._weights = buffer;
                        break;
                    }
                    case 2: // TASK_INFO
                    case 3: // TBE_KERNELS
                    case 4: { // CUST_AICPU_KERNELS
                        break;
                    }
                    case 5: { // DEVICE_CONFIG
                        this.devices = new Map();
                        const decoder = new TextDecoder('ascii');
                        const reader = new base.BinaryReader(buffer);
                        reader.uint32();
                        for (let position = 4; position < partition.size; ) {
                            const length = reader.uint32();
                            const buffer = reader.read(length);
                            const name = decoder.decode(buffer);
                            const device = reader.uint32();
                            this.devices.set(name, device);
                            position += 4 + length + 4;
                        }
                        break;
                    }
                    default: {
                        throw new om.Error("Unsupported partition type '" + partition.type + "'.");
                    }
                }
            }
        }
    }
};

om.Utility = class {

    static dtype(value) {
        om.Utility._types = om.Utility._types || [
            'undefined', 'float32', 'float16', 'int8', 'uint8', 'int16', 'uint16', 'int32',
            'int64', 'uint32', 'uint64', 'boolean', 'float64', 'string', 'dual_sub_int8', 'dual_sub_uint8',
            'complex64', 'complex128', 'qint8', 'qint16', 'qint32', 'quint8', 'quint16', 'resource',
            'stringref', 'dual', 'variant', 'bfloat16', 'int4', 'uint1', 'int2', 'uint2'
        ];
        if (value >= om.Utility._types.length) {
            throw new om.Error("Unsupported dtype '" + value + "'.");
        }
        return om.Utility._types[value];
    }

    static decodeText(value) {
        om.Utility._textDecoder = om.Utility._textDecoder || new TextDecoder('utf-8');
        return om.Utility._textDecoder.decode(value);
    }
};

om.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = om.ModelFactory;
}