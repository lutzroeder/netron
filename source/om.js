
// Experimental

import * as base from './base.js';
import * as protobuf from './protobuf.js';

const om = {};
const svp = {};

om.ModelFactory = class {

    match(context) {
        const container = om.Container.open(context);
        if (container) {
            context.type = 'om';
            context.target = container;
        }
    }

    async open(context) {
        const target = context.target;
        await target.read();
        const metadata = await context.metadata('om-metadata.json');
        return new om.Model(metadata, target);
    }
};

om.Model = class {

    constructor(metadata, target) {
        this.format = target.format;
        const context = {
            metadata,
            signature: target.signature,
            weights: target.weights
        };
        this.graphs = target.model.graph.map((graph) => new om.Graph(context, graph));
    }
};

om.Graph = class {

    constructor(context, graph) {
        switch (context.signature) {
            case 'IMOD': this.name = graph.name; break;
            case 'PICO': this.name = graph.id.toString(); break;
            default: throw new om.Error(`Unsupported DaVinci OM ${context.signature} signature.`);
        }
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (!values.has(name)) {
                values.set(name, new om.Value(name, type || null, tensor || null));
            } else if ((type && !type.equals(values.get(name).type)) ||
                       (tensor && tensor !== values.get(name).initializer)) {
                throw new om.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const tensors = new Map();
        const ops = [];
        for (const op of graph.op) {
            if (op.type === 'Const' && op.attr && op.attr.value) {
                const desc = op.attr.value.t.desc;
                let data = null;
                if (op.attr.value.t.data.length !== 0) {
                    data = op.attr.value.t.data;
                } else if (context.weights === null) {
                    data = null;
                } else if (desc.attr.merged_offset) {
                    const offset = Number(desc.attr.merged_offset.i);
                    data = context.weights.slice(offset, offset + Number(desc.weight_size));
                } else {
                    const offset = Number(desc.data_offset);
                    data = context.weights.slice(offset, offset + Number(desc.weight_size));
                }
                const type = om.Utility.tensorType(desc);
                const tensor = new om.Tensor('Constant', type, data);
                tensors.set(op.name, tensor);
                continue;
            }
            ops.push(op);
        }
        for (const op of ops) {
            const node = new om.Node(context, op, graph, values, tensors);
            this.nodes.push(node);
        }
    }
};

om.Node = class {

    constructor(context, op, graph, values, tensors) {
        this.name = op.name || '';
        this.type = context.metadata.type(op.type) || { name: op.type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        this.controlDependencies = [];
        this.device = null;
        if (op.input) {
            let index = 0;
            for (let i = 0; i < op.input.length; i++) {
                const input = op.input[i];
                if (input === '') {
                    continue;
                }
                const name = this.type.inputs && i < this.type.inputs.length ? this.type.inputs[i].name : `input${index === 0 ? '' : index}`;
                index++;
                const end = this.type.inputs && i < this.type.inputs.length && this.type.inputs[i].type && this.type.inputs[i].type === 'Tensor[]' ? op.input.length : i + 1;
                const list = [];
                for (let j = i; j < end; j++) {
                    const input = op.input[j];
                    if (input === '') {
                        continue;
                    }
                    const index = input.lastIndexOf(':');
                    const identifier = input.substring(0, index);
                    const src_index = input.substring(index + 1);
                    if (src_index === '-1') {
                        this.controlDependencies.push(values.map(name));
                        continue;
                    }
                    const type = om.Utility.tensorType(op.input_desc[j]);
                    const tensor = tensors.get(identifier);
                    const value = values.map(input, type, tensor);
                    list.push(value);
                }
                const argument = new om.Argument(name, list);
                this.inputs.push(argument);
                i = end - 1;
            }
        }
        if (op.output_desc) {
            for (let i = 0; i < op.output_desc.length; i++) {
                const identifier = `${this.name}:${i}`;
                const type = om.Utility.tensorType(op.output_desc[i]);
                const name = this.type.outputs && i < this.type.outputs.length ? this.type.outputs[i].name : `output${i === 0 ? '' : i}`;
                const value = values.map(identifier, type);
                const argument = new om.Argument(name, [value]);
                this.outputs.push(argument);
            }
        }
        for (const [name, value] of Object.entries(op.attr || {})) {
            if (name === 'device') {
                this.device = value;
                continue;
            }
            if (name === 'original_op_names') {
                continue;
            }
            if (name === 'relu_flag' && value.b) {
                this.chain.push(new om.Node(context, { type: 'ReLU' }, graph, value));
                continue;
            }
            const attribute = new om.Attribute(context, name, value);
            this.attributes.push(attribute);
        }
    }
};

om.Attribute = class {

    constructor(context, name, value) {
        this.name = name;
        this.value = value;
        switch (value.value) {
            case 'i': {
                this.value = value.i;
                this.type = 'int64';
                break;
            }
            case 'f': {
                this.value = value.f;
                this.type = 'float32';
                break;
            }
            case 'b': {
                this.value = value.b;
                this.type = 'boolean';
                break;
            }
            case 'bt': {
                this.value = null;
                if (value.bt.length !== 0) {
                    this.type = 'tensor';
                    const shape = new om.TensorShape([value.bt.length / 4]);
                    const type = new om.TensorType('float32', shape);
                    this.value = new om.Tensor('Constant', type, value.bt);
                }
                break;
            }
            case 'dt': {
                this.type = 'DataType';
                this.value = om.Utility.dtype(Number(value.dt));
                break;
            }
            case 's': {
                if (typeof value.s === 'string') {
                    this.value = value.s;
                } else if (value.s.filter((c) => c <= 32 && c >= 128).length === 0) {
                    this.value = om.Utility.decodeText(value.s);
                } else {
                    this.value = value.s;
                }
                this.type = 'string';
                break;
            }
            case 'g': {
                this.type = 'graph';
                this.value = new om.Graph(context, value.g);
                break;
            }
            case 'func': {
                break;
            }
            case 'list': {
                const list = value.list;
                this.value = [];
                if (list.s && list.s.length > 0) {
                    this.value = list.s.map((v) => String.fromCharCode.apply(null, new Uint16Array(v))).join(', ');
                    this.type = 'string[]';
                } else if (list.b && list.b.length > 0) {
                    this.value = list.b;
                    this.type = 'boolean[]';
                } else if (list.i && list.i.length > 0) {
                    this.value = list.i;
                    this.type = 'int64[]';
                } else if (list.f && list.f.length > 0) {
                    this.value = list.f;
                    this.type = 'float32[]';
                } else if (list.type && list.type.length > 0) {
                    this.type = 'type[]';
                    this.value = list.type.map((type) => om.Node.enum2Dtype(type) || '?');
                } else if (list.shape && list.shape.length > 0) {
                    this.type = 'shape[]';
                    this.value = list.shape.map((shape) => new om.TensorShape(shape));
                }
                break;
            }
            case 'list_list_int': {
                this.value = value.list_list_int.list_list_i.map((list) => list.list_i);
                break;
            }
            case 't': {
                const type = om.Utility.tensorType(value.t.desc);
                this.value = new om.Tensor('Constant', type, value.t.bytes);
                this.type = 'tensor';
                break;
            }
            case undefined: {
                this.value = null;
                break;
            }
            default: {
                throw new om.Error(`Unsupported attribute type '${JSON.stringify(value).substring(0, 32)}'.`);
            }
        }
    }
};

om.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

om.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new om.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

om.Tensor = class {

    constructor(category, type, value) {
        this.category = category;
        this.type = type;
        this.data = value;
    }
};

om.TensorType = class {

    constructor(dataType, shape, denotation) {
        this.dataType = dataType;
        this.shape = shape;
        this.denotation = denotation;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

om.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length &&
                obj.dimensions.every((value, index) => this.dimensions[index] === value)) {
                return true;
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
        this.context = context;
        this.signature = signature;
        this.weights = null;
    }

    async read() {
        if (this.context) {
            const stream = this.context.stream;
            const reader = base.BinaryReader.open(stream);
            reader.skip(4);
            switch (this.signature) {
                case 'IMOD': {
                    const decoder = new TextDecoder('utf-8');
                    this.format = 'DaVinci OM';
                    const header = {};
                    header.headsize = reader.uint32();
                    header.version = reader.uint32();
                    header.checksum = reader.read(64);
                    header.length = reader.uint32();
                    header.is_encrypt = reader.byte();
                    header.is_checksum = reader.byte();
                    header.modeltype = reader.byte(); // 0=IR model, 1=standard model, 2=OM Tiny model
                    header.genmode = reader.byte(); // 0=offline, 1=online
                    header.name = decoder.decode(reader.read(32));
                    header.ops = reader.uint32();
                    header.userdefineinfo = reader.read(32);
                    header.om_ir_version = reader.uint32();
                    header.model_num = header.version >= 0x20000000 ? reader.uint32() : 1;
                    header.platform_version = decoder.decode(reader.read(20));
                    header.platform_type = reader.byte();
                    header.padd = [reader.byte(), reader.byte(), reader.byte()];
                    header.model_length = reader.uint64().toNumber();
                    header.need_check_os_cpu_info = reader.byte();
                    header.is_unknow_model = reader.byte(); // 0:static model 1:dynamic model
                    header.reserved = reader.read(62);
                    const partitions = new Map();
                    let size = -1;
                    for (let align = 4; align <= 8; align += 4) {
                        reader.seek(header.headsize);
                        const count = reader.uint32();
                        reader.skip(align - 4);
                        size = 4 + (align - 4) + (count * 3 * align);
                        for (let i = 0; i < count; i++) {
                            const type = align === 4 ? reader.uint32() : reader.uint64().toNumber();
                            const offset = align === 4 ? reader.uint32() : reader.uint64().toNumber();
                            const size = align === 4 ? reader.uint32() : reader.uint64().toNumber();
                            if (type >= 32 || partitions.has(type) || (offset + size) >= stream.length) {
                                partitions.clear();
                                break;
                            }
                            partitions.set(type, { offset, size });
                        }
                        if (partitions.size > 0) {
                            break;
                        }
                    }
                    if (!partitions.has(0)) {
                        throw new om.Error('File does not contain a model definition.');
                    }
                    const offset = header.headsize + size;
                    for (const [type, partition] of partitions) {
                        reader.seek(offset + partition.offset);
                        const buffer = reader.read(partition.size);
                        switch (type) {
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
                            case 5: { // DEVICE_CONFIG, SO_BINS
                                this.devices = new Map();
                                const decoder = new TextDecoder('ascii');
                                const reader = base.BinaryReader.open(buffer);
                                reader.uint32();
                                for (let position = 4; position < partition.size;) {
                                    const length = reader.uint32();
                                    const buffer = reader.read(length);
                                    const name = decoder.decode(buffer);
                                    const device = reader.uint32();
                                    this.devices.set(name, device);
                                    position += 4 + length + 4;
                                }
                                break;
                            }
                            case 6: // FLOW_MODEL
                            case 7: // FLOW_SUBMODEL
                            case 8: // MODEL_INOUT_INFO
                            case 9: // STATIC_TASK_DESC
                            case 10: // DYNAMIC_TASK_DESC
                            case 11: // TASK_PARAM
                            case 20: // PRE_MODEL_DESC
                            case 21: // PRE_MODEL_SQE
                            case 22: { // PRE_KERNEL_ARGS
                                break;
                            }
                            default: {
                                throw new om.Error('Unsupported DaVinci OM partition type.');
                            }
                        }
                    }
                    om.proto = await this.context.require('./om-proto');
                    om.proto = om.proto.ge.proto;
                    try {
                        const reader = protobuf.BinaryReader.open(this.model);
                        this.model = om.proto.ModelDef.decode(reader);
                    } catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new om.Error(`File format is not ge.proto.ModelDef (${message.replace(/\.$/, '')}).`);
                    }
                    break;
                }
                case 'PICO': {
                    this.format = 'DaVinci OM SVP'; // SVP = Smart Vision PICO
                    reader.uint32(); // reserved
                    this.size = reader.uint32();
                    const param_size = reader.uint32();
                    const param_offset = reader.uint32();
                    reader.uint32(); // tmp_bufsize
                    const tfm_offset = reader.uint32();
                    reader.uint32(); // tfm_size
                    reader.seek(param_offset);
                    this.param = reader.read(param_size);
                    const buffer = reader.read(tfm_offset - reader.position);
                    this.model = new svp.ModelDef(buffer);
                    break;
                }
                default: {
                    throw new om.Error(`Unsupported DaVinci OM ${this.signature} signature.`);
                }
            }
            delete this.context;
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
            throw new om.Error(`Unsupported dtype '${value}'.`);
        }
        return om.Utility._types[value];
    }

    static tensorType(desc) {
        if (desc.shape && Array.isArray(desc.shape.dim)) {
            const dataType = desc && desc.dtype ? om.Utility.dtype(desc.dtype) : '?';
            const shape = new om.TensorShape(desc.shape.dim);
            return new om.TensorType(dataType, shape, desc.layout);
        }
        return null;
    }

    static decodeText(value) {
        om.Utility._textDecoder = om.Utility._textDecoder || new TextDecoder('utf-8');
        return om.Utility._textDecoder.decode(value);
    }
};

om.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci OM model.';
    }
};

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
                        if (op.attr && op.attr.seg_id && op.attr.seg_id.i === item.id) {
                            let out_num = 0;
                            if (typeof op.output_index === 'number') {
                                out_num = op.output_index + 1;
                            } else {
                                const input_num = op.input.map((element) => element.split(":")[1]);
                                out_num = input_num.length > 0 ? Math.max(...input_num) + 1 : 1;
                            }
                            const out_types = [];
                            if (op.data_flow && op.data_flow !== '') {
                                const data = op.data_flow;
                                if (data.indexOf('o[{t') !== -1) {
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
                                if (op_item.id === op.id) {
                                    curr_op = op_item;
                                    break;
                                }
                            }
                            if (curr_op === null) {
                                op.output_desc = op.output_desc.concat(out_list);
                                item.op.push(op);
                            } else {
                                curr_op.output_desc = curr_op.output_desc.concat(out_list);
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
        const input = (buffer) => {
            const input = {};
            const reader = new svp.BinaryReader(buffer);
            while (reader.position < reader.length) {
                const tag = reader.uint16();
                switch (tag & 0x1fff) {
                    case 0x0051: input.id = reader.value(tag); break;
                    case 0x0058: input.name = reader.value(tag, 'string').trim(); break;
                    case 0x005a: input.shape_vector = reader.value(tag, 'uint32[]'); break;
                    default: reader.value(tag); break;
                }
            }
            return input;
        };
        const output = (buffer) => {
            const output = {};
            const reader = new svp.BinaryReader(buffer);
            while (reader.position < reader.length) {
                const tag = reader.uint16();
                switch (tag & 0x1fff) {
                    case 0x0061: output.id = reader.value(tag); break;
                    case 0x0066: output.name = reader.value(tag, 'string').trim(); break;
                    case 0x0069: output.shape_vector = reader.value(tag, 'uint32[]'); break;
                    case 0x0110: output.layer_num = reader.value(tag); break;
                    default: reader.value(tag); break;
                }
            }
            return output;
        };
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            const value = reader.value(tag);
            switch (tag & 0x1fff) {
                case 0x0041: this.id = value; break;
                case 0x0050: this.input.push(input(value)); break;
                case 0x0060: this.output.push(output(value)); break;
                default: break;
            }
        }
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
                case 0x014B: this.attr.seg_id = reader.attribute(tag, 'i'); break;
                case 0x0150: this.attr.is_not_last_merge_layer = reader.attribute(tag, 'i'); break;
                case 0x0151: this.attr.is_dump_avavilable = reader.attribute(tag, 'i'); break;
                case 0x0153: this.attr.debug_dump_offset = reader.attribute(tag, 'i'); break;
                case 0x0152: this.type = reader.value(tag, 'string'); break;
                case 0x0154: this.output_shape_vector = reader.value(tag, 'uint32[]'); break;
                case 0x0155: this.input_index = reader.value(tag); break;
                case 0x015B: this.output_index = reader.value(tag); break;
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
            default: throw new svp.Error(`Unsupported attribute type '${type}'.`);
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

svp.BinaryReader = class {

    constructor(buffer) {
        this._reader = base.BinaryReader.open(buffer);
    }

    get length() {
        return this._reader.length;
    }

    get position() {
        return this._reader.position;
    }

    seek(position) {
        this._reader.seek(position);
    }

    read(length) {
        return this._reader.read(length);
    }

    int8() {
        return this._reader.int8();
    }

    uint16() {
        return this._reader.uint16();
    }

    uint32() {
        return this._reader.uint32();
    }

    value(tag, type) {
        let value = 0;
        switch (tag >> 13) {
            case 1: value = this.int8(); break;
            case 2: value = this.uint16(); break;
            case 3: value = this.uint32(); break;
            case 4: value = this.read(this.int8()); break;
            case 5: value = this.read(this.uint16()); break;
            case 6: value = this.read(this.uint32()); break;
            default: throw new svp.Error(`Unsupported value identifier '${tag}'.`);
        }
        return type ? this._cast(value, type, tag) : value;
    }

    find(tag, type) {
        let value = null;
        let match = false;
        while (!match && this.position < this.length) {
            const current = this.uint16();
            value = this.value(current);
            match = current === tag;
        }
        this.seek(0);
        return match && type ? this._cast(value, type, tag) : value;
    }

    attribute(tag, type) {
        const value = this.value(tag);
        return new svp.AttrDef(value, type);
    }

    _cast(value, type, tag) {
        switch (type) {
            case 'string': {
                if (value instanceof Uint8Array) {
                    svp.BinaryReader._decoder = svp.BinaryReader._decoder || new TextDecoder('utf-8');
                    return svp.BinaryReader._decoder.decode(value).replace(/\0.*$/g, '');
                }
                throw new om.Error(`Invalid 'string' tag '${tag.toString(16)}'.`);
            }
            case 'uint32[]': {
                const reader = base.BinaryReader.open(value);
                value = [];
                while (reader.position < reader.length) {
                    value.push(reader.uint32());
                }
                return value;
            }
            default: {
                return value;
            }
        }
    }
};

svp.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci SVP model.';
    }
};

export const ModelFactory = om.ModelFactory;
