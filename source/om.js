
// Experimental

var om = {};
var protobuf = require('./protobuf');
var base = require('./base');

om.ModelFactory = class {

    match(context) {
        return om.Container.open(context);
    }

    open(context, match) {
        const container = match;
        return container.open().then(() => {
            return context.metadata('om-metadata.json').then((metadata) => {
                return new om.Model(metadata, container);
            });
        });
    }
};

om.Model = class {

    constructor(metadata, container) {
        this._graphs = [];
        this._format = container.format;
        const context = { metadata: metadata, weights: container.weights };
        for (const graph of container.model.graph) {
            this._graphs.push(new om.Graph(context, graph));
        }
    }

    get format() {
        return this._format;
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
                    const dataType = desc && desc.dtype ? om.Utility.dtype(value.desc.dtype) : '?';
                    const tensorType = new om.TensorType(dataType, shape, format, value.desc.layout);
                    const tensor = new om.Tensor('Constant', tensorType, data);
                    const argument = new om.Argument(name, null, tensor);
                    this._inputs.push(new om.Parameter(parameterName, true, [ argument ]));
                }
                else {
                    const dataType = desc && desc.dtype ? om.Utility.dtype(desc.dtype) : '?';
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
                const dataType = desc && desc.dtype ? om.Utility.dtype(desc.dtype) : '?';
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
            case 'list_list_int': {
                this._value = value.list_list_int.list_list_i.map((list) => list.list_i);
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

om.Container = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length >= 256) {
            const buffer = stream.peek(4);
            const signature = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
            if (signature === 'IMOD' || signature === 'PICO') {
                return new om.Container(context, signature);
            }
        }
        return null;
    }

    constructor(context, signature) {
        this._context = context;
        this._signature = signature;
    }

    open() {
        const stream = this._context.stream;
        const reader = new base.BinaryReader(stream);
        const buffer = reader.read(4);
        this.signature = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
        switch (this.signature) {
            case 'IMOD': {
                this.format = 'DaVinci OM';
                const decoder = new TextDecoder('utf-8');
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
                            this.model = buffer;
                            break;
                        }
                        case 1: { // WEIGHTS_DATA
                            this.weights = buffer;
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
                if (!this.model) {
                    throw new om.Error('File does not contain a model definition.');
                }
                return this._context.require('./om-proto').then(() => {
                    try {
                        om.proto = protobuf.get('om').ge.proto;
                        const reader = protobuf.BinaryReader.open(this.model);
                        this.model = om.proto.ModelDef.decode(reader);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new om.Error('File format is not ge.proto.ModelDef (' + message.replace(/\.$/, '') + ').');
                    }
                });
            }
            case 'PICO': {
                this.format = 'DaVinci OM SVM';
                reader.uint32(); // reserved
                this.size = reader.uint32();
                const param_size = reader.uint32();
                const param_offset = reader.uint32();
                reader.uint32(); // tmp_bufsize
                const tfm_offset = reader.uint32();
                reader.uint32(); // tfm_size
                this.type = 2;
                reader.seek(param_offset);
                this.param = reader.read(param_size);
                const buffer = reader.read(tfm_offset - reader.position);
                this.model = new svp.ModelDef(buffer);
                // return Promise.resolve();
                return Promise.reject(new om.Error('Unsupported DaVinci OM ' + this.signature + ' signature.'));
            }
            default: {
                return Promise.reject(new om.Error('Unsupported DaVinci OM ' + this.signature + ' signature.'));
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

var svp = svp || {};

svp.ModelDef = class ModelDef {

    constructor(buffer) {
        const reader = new svp.BinaryReader(buffer);
        this.attr = {};
        this.graph = [];
        this.name = reader.find(0x800D, 'string');
        this.batch_num = reader.find(0x600A);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            const value = reader.value(tag);
            switch (tag & 0x1fff) {
                case 0x0040: {
                    this.graph.push(new svp.GraphDef(value));
                    break;
                }
                case 0x0111: {
                    const op = new svp.OpDef(value);
                    for (const item of this.graph) {
                        if (op.seg_id == item.id) {
                            let out_num;
                            if (op.output_index) {
                                out_num = op.output_index + 1;
                            }
                            else {
                                const input_num = op.input.map((element) => element.split(":")[1]);
                                out_num = input_num.length > 0 ? Math.max(...input_num) + 1 : 1;
                            }
                            const out_types = [];
                            if (op.data_flow != null && op.data_flow != '') {
                                const data = op.data_flow;
                                if (data.indexOf('o[{t') != -1) {
                                    const outs = data.substring(data.indexOf('o[{t')).split(',');
                                    for (const out of outs) {
                                        const startIndex = out.indexOf("\"");
                                        const endIndex = out.indexOf("\"", startIndex + 1);
                                        out_types.push(out.substring(startIndex + 1, endIndex));
                                    }
                                }
                            }
                            const out_list = [];
                            while (out_num > 0) {
                                const output_desc = {};
                                output_desc.shape = { dim: op.output_shape_vector };
                                output_desc.layout = 'NCHW';
                                if (op.data_flow && out_types.length >= out_num) {
                                    output_desc.dtype = out_types[op.output_index + 1 - out_num];
                                }
                                out_list.push(output_desc);
                                out_num--;
                            }

                            let curr_op = null;
                            for (const op_item of item.op) {
                                if (op_item.id == op.id) {
                                    curr_op = op_item;
                                    break;
                                }
                            }
                            if (curr_op != null) {
                                curr_op.output_desc = curr_op.output_desc.concat(out_list);
                            }
                            else {
                                op.output_desc = op.output_desc.concat(out_list);
                                item.op.push(op);
                            }
                            break;
                        }
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
        if (this.graph.length > 1) {
            for (let i = 1; i < this.graph.length; i++) {
                this.graph[0].op = this.graph[0].op.concat(this.graph[i].op);
            }
        }
    }
};

svp.GraphDef = class {

    constructor(buffer) {
        this.input = [];
        this.output = [];
        this.op = [];
        this.attr = {};
        const reader = new svp.BinaryReader(buffer);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            const value = reader.value(tag);
            switch (tag & 0x1fff) {
                case 0x0041: this.id = value; break;
                case 0x0050: this.input.push(this._input(value)); break;
                case 0x0060: this.output.push(this._output(value)); break;
                default: break;
            }
        }
    }

    _input(buffer) {
        const input = {};
        const reader = new svp.BinaryReader(buffer);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            switch (tag & 0x1fff) {
                case 0x0051: input.id = reader.value(tag); break;
                case 0x0058: input.name = reader.value(tag, 'string'); break;
                case 0x005a: input.shape_vector = reader.value(tag, 'uint32[]'); break;
                default: reader.value(tag); break;
            }
        }
        return input;
    }

    _output(buffer) {
        const output = {};
        const reader = new svp.BinaryReader(buffer);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            switch (tag & 0x1fff) {
                case 0x0061: output.id = reader.value(tag); break;
                case 0x0066: output.name = reader.value(tag, 'string'); break;
                case 0x0069: output.shape_vector = reader.value(tag, 'uint32[]'); break;
                case 0x0110: output.layer_num = reader.value(tag); break;
                default: reader.value(tag); break;
            }
        }
        return output;
    }
};

svp.OpDef = class {

    constructor(buffer) {
        this.input = [];
        this.attr = {};
        this.input_i = [];
        this.output_i = [];
        this.input_desc = [];
        this.output_desc = [];
        const reader = new svp.BinaryReader(buffer);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            switch (tag & 0x1fff) {
                case 0x0114: this.name = reader.value(tag, 'string').trim(); break;
                case 0x0112: this.id = reader.value(tag); break;
                case 0x0119: this.attr.output_m2m_flag = reader.attribute(tag, 'i'); break;
                case 0x0121: this.attr.batch_flag = reader.attribute(tag, 'i'); break;
                case 0x0124: this.attr.dequant_scale = reader.attribute(tag, 'i'); break;
                case 0x0126: this.attr.output_address = reader.attribute(tag, 'i'); break;
                case 0x0125: this.attr.dequant_offset = reader.attribute(tag, 'i'); break;
                case 0x0127: this.attr.first_inst_addr = reader.attribute(tag, 'i'); break;
                case 0x0128: this.attr.last_inst_addr = reader.attribute(tag, 'i'); break;
                case 0x013B: this.attr.is_fusion_layer = reader.attribute(tag, 'i'); break;
                case 0x013C: this.input = reader.value(tag, 'string').split(','); break;
                case 0x014B: this.seg_id = reader.value(tag); break;
                case 0x0150: this.attr.is_not_last_merge_layer = reader.attribute(tag, 'i'); break;
                case 0x0151: this.attr.is_dump_avavilable = reader.attribute(tag, 'i'); break;
                case 0x0153: this.attr.debug_dump_offset = reader.attribute(tag, 'i'); break;
                case 0x0152: this.type = reader.value(tag, 'string'); break;
                case 0x0154: this.output_shape_vector = reader.value(tag, 'uint32[]'); break;
                case 0x0155: this.input_index = reader.value(tag, 'string'); break;
                case 0x015B: this.output_index = reader.value(tag, 'string'); break;
                case 0x0156: this.attr.trap_inst_pc = reader.attribute(tag, 'i'); break;
                case 0x0157: this.attr.profile_layer_id = reader.attribute(tag, 'i'); break;
                case 0xA15A:
                    this.data_flow = reader.value(tag, 'string');
                    this.attr.data_flow = new svp.AttrDef(this.data_flow.replace('i[{t', 'input[{type').replace(',f[{t', '\tforward[{type').replace(',o[{t', '\toutput[{type').replace(',{[t', ',{type'), 's');
                    break;
                default: reader.value(tag); break;
            }
        }
        for (let i = 0; i < this.input.length; i++) {
            this.input_desc.push({ layout: 'NCHW', shape: {} });
        }
    }
};

svp.AttrDef = class {

    constructor(item, type) {
        switch (type) {
            case 's': this.s = item; break;
            case 'i': this.i = item; break;
            default: throw new svp.Error("Unsupported attribute type '" + type + "'.");
        }
    }

    get value() {
        if (this.s !== undefined) {
            return 's';
        }
        if (this.i !== undefined) {
            return 'i';
        }
        return undefined;
    }
};

svp.BinaryReader = class extends base.BinaryReader {

    constructor(buffer) {
        super(buffer);
    }

    value(tag, type) {
        let value;
        switch (tag >> 13) {
            case 1: value = this.int8(); break;
            case 2: value = this.uint16(); break;
            case 3: value = this.uint32(); break;
            case 4: value = this.read(this.int8()); break;
            case 5: value = this.read(this.uint16()); break;
            case 6: value = this.read(this.uint32()); break;
            default: throw new svp.Error("Unsupported value identifier '" + tag + "'.");
        }
        return type ? this._cast(value, type) : value;
    }

    find(tag, type) {
        let value = null;
        let match = false;
        while (!match && this.position < this.length) {
            const current = this.uint16();
            value = this.value(current);
            match = current == tag;
        }
        this.seek(0);
        return match && type ? this._cast(value, type) : value;
    }

    attribute(tag, type) {
        const value = this.value(tag);
        return new svp.AttrDef(value, type);
    }

    _cast(value, type) {
        switch (type) {
            case 'string': {
                svp.BinaryReader._decoder = svp.BinaryReader._decoder || new TextDecoder('utf-8');
                return svp.BinaryReader._decoder.decode(value).replace(/\0.*$/g, '');
            }
            case 'uint32[]': {
                const reader = new base.BinaryReader(value);
                value = [];
                while (reader.position < reader.length) {
                    value.push(reader.uint32());
                }
                return value;
            }
            default:
                return value;
        }
    }
};

svp.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci SVP model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = om.ModelFactory;
}