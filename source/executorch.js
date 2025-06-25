
// Experimental

const executorch = {};
const coreml = {};
const vulkan = {};
const xnnpack = {};

import * as base from './base.js';
import * as python from './python.js';
import * as pytorch from './pytorch.js';

executorch.ModelFactory = class {

    async match(context) {
        const reader = await executorch.Reader.open(context);
        if (reader) {
            return context.set('executorch', reader);
        }
        return null;
    }

    async open(context) {
        executorch.schema = await context.require('./executorch-schema');
        const target = context.value;
        await target.read();
        return new executorch.Model(target);
    }
};

executorch.Model = class {

    constructor(target) {
        this.format = `ExecuTorch v${target.program.version}`;
        this.graphs = [];
        for (const plan of target.program.execution_plan) {
            for (const chain of plan.chains) {
                const graph = new executorch.Graph(target, plan, chain);
                this.graphs.push(graph);
            }
        }
    }
};

executorch.Graph = class {

    constructor(target, plan, chain) {
        this.name = plan.name || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.tensors = (index, items) => {
            const list = [];
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                const type = item ? new executorch.TensorType(item) : null;
                let initializer = null;
                if (item && item.data_buffer_idx > 0) {
                    initializer = new executorch.Tensor(item, target);
                }
                const identifier = items.length > 1 ? `${index}.${i}` : index.toString();
                const value = new executorch.Value(identifier, type, initializer);
                list.push(value);
            }
            return list;
        };
        values.map = (index, output) => {
            if (!values.has(index)) {
                const executorch_flatbuffer = executorch.schema.executorch_flatbuffer;
                const val = plan.values[index].val;
                const tensor = val instanceof executorch_flatbuffer.Tensor || val instanceof executorch_flatbuffer.TensorList || val instanceof executorch_flatbuffer.OptionalTensorList;
                if (output && !tensor) {
                    const value = [new executorch.Value(index.toString(), null, null)];
                    values.set(index, { type: null, value });
                } else if (val instanceof executorch_flatbuffer.Null) {
                    values.set(index, { type: 'attribute', value: null });
                } else if (val instanceof executorch_flatbuffer.Int) {
                    values.set(index, { type: 'int64', value: val.int_val });
                } else if (val instanceof executorch_flatbuffer.Bool) {
                    values.set(index, { type: 'int64', value: val.bool_val });
                } else if (val instanceof executorch_flatbuffer.Double) {
                    values.set(index, { type: 'float64', value: val.double_val });
                } else if (val instanceof executorch_flatbuffer.Tensor) {
                    const items = [val];
                    values.set(index, { type: null, value: values.tensors(index, items) });
                } else if (val instanceof executorch_flatbuffer.String) {
                    values.set(index, { type: 'string', value: val.string_val });
                } else if (val instanceof executorch_flatbuffer.IntList) {
                    const list = val.items.map((index) => plan.values[index].val.int_val);
                    values.set(index, { type: 'int64[]', value: list });
                } else if (val instanceof executorch_flatbuffer.DoubleList) {
                    throw new executorch.Error('executorch_flatbuffer.DoubleList not implemented.');
                } else if (val instanceof executorch_flatbuffer.BoolList) {
                    throw new executorch.Error('executorch_flatbuffer.BoolList not implemented.');
                } else if (val instanceof executorch_flatbuffer.TensorList) {
                    const items = Array.from(val.items).map((arg) => arg === -1 ? null : plan.values[arg].val);
                    values.set(index, { type: null, value: values.tensors(index, items) });
                } else if (val instanceof executorch_flatbuffer.OptionalTensorList) {
                    const items = Array.from(val.items).map((arg) => arg === -1 ? null : plan.values[arg].val);
                    values.set(index, { type: null, value: values.tensors(index, items) });
                } else {
                    throw new Error(`Value type '${val.constructor.name}' not implemented.`);
                }
            }
            return values.get(index);
        };
        for (let i = 0; i < plan.inputs.length; i++) {
            const input = plan.inputs[i];
            const value = values.map(input);
            const name = plan.inputs.length === 1 ? 'input' : `input.${i}`;
            const argument = new executorch.Argument(name, value.value, value.type);
            this.inputs.push(argument);
        }
        for (let i = 0; i < plan.outputs.length; i++) {
            const output = plan.outputs[i];
            const value = values.map(output);
            const name = plan.outputs.length === 1 ? 'output' : `output.${i}`;
            const argument = new executorch.Argument(name, value.value, value.type);
            this.outputs.push(argument);
        }
        for (const instruction of chain.instructions) {
            const node = new executorch.Node(target, plan, chain, instruction, values);
            this.nodes.push(node);
        }
    }
};

executorch.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

executorch.Value = class Value {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new executorch.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer && initializer.type ? initializer.type : type || null;
        this.initializer = initializer || null;
    }
};

executorch.Node = class {

    constructor(target, plan, chain, instruction, values) {
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const instr_args = instruction.instr_args;
        const executorch_flatbuffer = executorch.schema.executorch_flatbuffer;
        if (instr_args instanceof executorch_flatbuffer.KernelCall) {
            const op = plan.operators[instr_args.op_index];
            const name = op.name.split('::').pop();
            const identifier = op.overload ? `${op.name}.${op.overload}` : op.name;
            const schemas = target.execution.invoke('torch._C._jit_get_schemas_for_operator', [op.name]);
            const schema = schemas.find((schema) => schema.name === op.name && schema.overload_name === op.overload);
            if (!schema) {
                throw new executorch.Error(`Operator schema for '${identifier}' not found.`);
            }
            const category = schema && schema.category ? schema.category : '';
            const alias = (arg) => arg && arg.alias_info && arg.alias_info.before_set.length === 1 ? arg.alias_info.before_set[0] : null;
            const outputs = new Set(schema && Array.isArray(schema.returns) ? schema.returns.map((arg) => alias(arg)).filter((alias) => alias !== null) : []);
            const inputs = new Map();
            this.type = { name, identifier, category };
            let i = 0;
            const args = instr_args.args;
            for (; i < schema.arguments.length; i++) {
                const index = args[i];
                const arg = schema && i < schema.arguments.length ? schema.arguments[i] : null;
                const output = arg ? alias(schema.arguments[i]) : null;
                if (output && outputs.has(output)) {
                    inputs.set(output, index);
                    continue;
                }
                const name = arg ? arg.name : i.toString();
                const value = values.map(index);
                const argument = new executorch.Argument(name, value.value, value.type);
                this.inputs.push(argument);
            }
            for (let j = 0; j < schema.returns.length; j++) {
                const ret = schema.returns[j];
                const output = alias(ret);
                let index = args[i++];
                index = output && inputs.has(output) ? inputs.get(output) : index;
                const name = ret.name;
                const value = values.map(index, true);
                const argument = new executorch.Argument(name || '', value.value, value.type);
                this.outputs.push(argument);
            }
        } else if (instr_args instanceof executorch_flatbuffer.DelegateCall) {
            const delegate = plan.delegates[instr_args.delegate_index];
            const args = instr_args.args;
            if (!delegate.backend || !delegate.backend.type) {
                throw new executorch.Error(`ExecuTorch delegate '${delegate.id}' not implemented.`);
            }
            this.type = delegate.backend.type;
            const inputs = args.slice(0, this.type.inputs.length);
            for (let i = 0; i < inputs.length; i++) {
                const input = inputs[i];
                const value = values.map(input);
                const name = inputs.length === 1 ? 'input' : `input.${i}`;
                const argument = new executorch.Argument(name, value.value, value.type);
                this.inputs.push(argument);
            }
            const outputs = args.slice(this.type.inputs.length, this.type.inputs.length + this.type.outputs.length);
            for (let i = 0; i < outputs.length; i++) {
                const output = outputs[i];
                const value = values.map(output);
                const name = inputs.length === 1 ? 'output' : `output.${i}`;
                const argument = new executorch.Argument(name, value.value, value.type);
                this.outputs.push(argument);
            }
            for (const spec of delegate.compile_specs) {
                const value = spec.value instanceof Uint8Array ? new TextDecoder('utf-8').decode(spec.value) : spec.value;
                const attribute = new executorch.Argument(spec.key, value, 'attribute');
                this.attributes.push(attribute);
            }
        } else {
            throw new Error(`Instruction type '${instr_args.constructor.name}' not implemented.`);
        }
    }
};

executorch.TensorType = class {

    constructor(tensor) {
        executorch.TensorType._types = executorch.TensorType._types || [
            'uint8',
            'int8', 'int16', 'int32', 'int64',
            'float16', 'float32', 'float64',
            'complex16', 'complex32', 'complex64',
            'boolean',
            'qint8', 'quint8', 'qint32',
            'bfloat16',
            'quint4x2', 'quint2x4', 'bits1x8', 'bits2x4', 'bits4x2', 'bits8', 'bits16',
            'float8e5m2', 'float8e4m3fn', 'float8e5m2fnuz', 'float8e4m3fnuz',
            'uint16', 'uint32', 'uint64'
        ];
        if (tensor.scalar_type >= executorch.TensorType._types.length) {
            throw new executorch.Error(`Unknown tensor data type '${tensor.scalar_type}'.`);
        }
        this.dataType = executorch.TensorType._types[tensor.scalar_type];
        this.shape = new executorch.TensorShape(Array.from(tensor.sizes));
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

executorch.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions || [];
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

executorch.Tensor = class {

    constructor(tensor, target) {
        this.type = new executorch.TensorType(tensor);
        const data_buffer_idx = tensor.data_buffer_idx;
        const program = target.program;
        if (tensor.extra_tensor_info) {
            throw new executorch.Error('Extra tensor info not implemented.');
        } else if (program.constant_buffers) {
            throw new executorch.Error('Constant buffers not implemented.');
        } else if (tensor.allocation_info === null) {
            const constant_segment = program.constant_segment;
            const data_segment = program.segments[constant_segment.segment_index];
            const offset = constant_segment.offsets[data_buffer_idx].toNumber();
            const next = data_buffer_idx + 1 < constant_segment.offsets.length ? constant_segment.offsets[data_buffer_idx + 1].toNumber() : data_segment.size.toNumber();
            const size = next - offset;
            this.values = target.blob(data_segment.offset.toNumber() + offset, size);
            this.encoding = '<';
        } else {
            throw new executorch.Error('Tensor allocation info not implemented.');
        }
    }
};

executorch.Reader = class {

    static async open(context) {
        const reader = await context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'ET12') {
            return new executorch.Reader(context, reader);
        }
        return null;
    }

    constructor(context, reader) {
        this.context = context;
        this.reader = reader;
    }

    async read() {
        const context = this.context;
        this.metadata = await pytorch.Metadata.open(context);
        this.execution = new python.Execution();
        this.metadata.register(this.execution);
        const executorch_flatbuffer = executorch.schema.executorch_flatbuffer;
        this.program = executorch_flatbuffer.Program.create(this.reader);
        this.reader = await context.read('binary');
        if (this.reader.length >= 32) {
            this.reader.seek(8);
            const magic = String.fromCharCode(...this.reader.read(4));
            if (magic === 'eh00') {
                this.extended_file_header = {
                    length: this.reader.uint32(),
                    program_size: this.reader.uint64().toNumber(),
                    segment_base_offset: this.reader.uint64().toNumber(),
                };
            }
            this.reader.seek(0);
        }
        for (const plan of this.program.execution_plan) {
            for (const chain of plan.chains) {
                for (const instruction of chain.instructions) {
                    const instr_args = instruction.instr_args;
                    if (instr_args instanceof executorch_flatbuffer.DelegateCall) {
                        const delegate = plan.delegates[instr_args.delegate_index];
                        if (delegate.backend) {
                            continue;
                        }
                        let data = null;
                        switch (delegate.processed.location) {
                            case executorch_flatbuffer.DataLocation.INLINE: {
                                data = this.program.backend_delegate_data[delegate.processed.index].data;
                                break;
                            }
                            case executorch_flatbuffer.DataLocation.SEGMENT: {
                                const segment = this.program.segments[delegate.processed.index];
                                data = this.blob(segment.offset.toNumber(), segment.size.toNumber());
                                break;
                            }
                            default: {
                                throw new executorch.Error(`Delegate data location '${delegate.processed.location}' not implemented.`);
                            }
                        }
                        switch (delegate.id) {
                            case 'XnnpackBackend': {
                                delegate.backend = xnnpack.Reader.open(data, this);
                                break;
                            }
                            case 'CoreMLBackend': {
                                delegate.backend = coreml.Reader.open(data, this);
                                break;
                            }
                            case 'VulkanBackend': {
                                delegate.backend = vulkan.Reader.open(data, this);
                                break;
                            }
                            default: {
                                throw new executorch.Error(`ExecuTorch delegate '${delegate.id}' not implemented.`);
                            }
                        }
                        /* eslint-disable no-await-in-loop */
                        await delegate.backend.read();
                        /* eslint-enable no-await-in-loop */
                    }
                }
            }
        }
    }

    blob(offset, size) {
        if (this.extended_file_header) {
            this.reader.seek(this.extended_file_header.segment_base_offset + offset);
            const data = this.reader.read(size);
            this.reader.seek(0);
            return data;
        }
        return null;
    }
};

executorch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ExecuTorch model.';
    }
};

xnnpack.Reader = class {

    static open(data, target) {
        if (data.length >= 30) {
            const reader = base.BinaryReader.open(data);
            reader.skip(4);
            const magic = String.fromCharCode(...reader.read(4));
            if (magic === 'XH00') {
                return new xnnpack.Reader(reader, target);
            }
        }
        return null;
    }

    constructor(reader, target) {
        this.reader = reader;
        this.target = target;
        reader.skip(2);
        this.flatbuffer = {
            offset: reader.uint32(),
            size: reader.uint32(),
        };
        this.constants = {
            offset: reader.uint32(),
            size: reader.uint32(),
        };
    }

    async read() {
        this.reader.seek(this.flatbuffer.offset);
        const flatbuffers = await import('./flatbuffers.js');
        const data = this.reader.read(this.flatbuffer.size);
        const reader = flatbuffers.BinaryReader.open(data);
        if (!executorch.schema.fb_xnnpack.XNNGraph.identifier(reader)) {
            throw new xnnpack.Error('Invalid XNNPACK data.');
        }
        this.graph = executorch.schema.fb_xnnpack.XNNGraph.create(reader);
        this.reader.seek(0);
        const metadata = new xnnpack.Metadata();
        this.type = new xnnpack.Graph(metadata, this.graph, this);
    }

    constant(idx) {
        const constant_data = this.graph.constant_data[idx];
        this.reader.seek(this.constants.offset + constant_data.offset.toNumber());
        const data = this.reader.read(constant_data.size.toNumber());
        this.reader.seek(0);
        return data;
    }
};

xnnpack.Graph = class {

    constructor(metadata, graph, reader) {
        this.name = 'XnnpackBackend';
        this.type = 'graph';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (id) => {
            if (!values.has(id)) {
                const fb_xnnpack = executorch.schema.fb_xnnpack;
                const name = id.toString();
                const xvalue = graph.xvalues[id].xvalue_union;
                if (xvalue instanceof fb_xnnpack.XNNTensorValue) {
                    const type = new xnnpack.TensorType(xvalue);
                    const initializer = xvalue.constant_buffer_idx === 0 ? null : new xnnpack.Tensor(xvalue, reader);
                    const value = new xnnpack.Value(name, type, initializer);
                    values.set(id, value);
                } else if (xvalue instanceof fb_xnnpack.XNNQuantizedTensorValue) {
                    const value = new xnnpack.Value(name, null, null);
                    values.set(id, value);
                } else {
                    throw new xnnpack.Error(`Value type '${xvalue.constructor.name}' not implemented.`);
                }
            }
            return values.get(id);
        };
        for (let i = 0; i < graph.input_ids.length; i++) {
            const id = graph.input_ids[i];
            const value = values.map(id);
            const name = graph.input_ids.length === 1 ? 'input' : `input.${i}`;
            const argument = new xnnpack.Argument(name, [value]);
            this.inputs.push(argument);
        }
        for (let i = 0; i < graph.output_ids.length; i++) {
            const id = graph.output_ids[i];
            const value = values.map(id);
            const name = graph.output_ids.length === 1 ? 'output' : `output.${i}`;
            const argument = new xnnpack.Argument(name, [value]);
            this.outputs.push(argument);
        }
        for (const xnode of graph.xnodes) {
            const node = new xnnpack.Node(metadata, xnode, values);
            this.nodes.push(node);
        }
    }
};

xnnpack.Node = class {

    constructor(metadata, xnode, values) {
        const node = xnode.xnode_union;
        this.type = metadata.type(node.constructor.name) || { name: node.constructor.name };
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        for (const [name, obj] of Object.entries(node)) {
            let value = ArrayBuffer.isView(obj) ? Array.from(obj) : obj;
            let type = 'attribute';
            if (name.endsWith('_id')) {
                value = obj === -1 ? [] : [values.map(obj)];
                type = null;
            }
            const argument = new xnnpack.Argument(name, value, type);
            if (name === 'output_id') {
                this.outputs.push(argument);
            } else {
                this.inputs.push(argument);
            }
        }
    }
};

xnnpack.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

xnnpack.Value = class Value {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new executorch.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer && initializer.type ? initializer.type : type || null;
        this.initializer = initializer || null;
    }
};

xnnpack.Metadata = class {

    constructor() {
        this._types = new Map();
        this.register('_XNNCat', 'Tensor');
        this.register('_XNNNodeConv', 'Layer');
        this.register('XNNArgMaxPooling2d', 'Pool');
        this.register('XNNAvgPooling2d', 'Pool');
        this.register('XNNCeiling', 'Activation');
        this.register('XNNConcatenate2', 'Tensor');
        this.register('XNNConcatenate3', 'Tensor');
        this.register('XNNConcatenate4', 'Tensor');
        this.register('XNNConcatenate5', 'Tensor');
        this.register('XNNConv2d', 'Layer');
        this.register('XNNConvTranspose2d', 'Layer');
        this.register('XNNDepthwiseConv2d', 'Layer');
        this.register('XNNELU', 'Activation');
        this.register('XNNFullyConnected', 'Layer');
        this.register('XNNGelu', 'Activation');
        this.register('XNNGlobalAvgPooling2d', 'Pool');
        this.register('XNNGlobalAvgPooling2d', 'Pool');
        this.register('XNNHardswish', 'Activation');
        this.register('XNNLeakyReLU', 'Activation');
        this.register('XNNMaxPooling2d', 'Pool');
        this.register('XNNPReLU', 'Activation');
        this.register('XNNSigmoid', 'Activation');
        this.register('XNNSoftmax', 'Activation');
        this.register('XNNTanh', 'Activation');
        this.register('XNNStaticTranspose', 'Transform');
    }

    register(name, category) {
        this._types.set(name, { name, category });
    }

    type(name) {
        return this._types.get(name);
    }
};

xnnpack.TensorType = class {

    constructor(tensor) {
        xnnpack.TensorType._types = executorch.TensorType._types || [
            'invalid', 'float32', 'float16',
            'qint8', 'quint8', 'qint32',
            'qcint8', 'qcint32', 'qcint4',
            'qdint8', 'qbint4', 'qpint8',
            'int32', 'pfp32', 'bfloat16'
        ];
        if (tensor.datatype >= xnnpack.TensorType._types.length) {
            throw new xnnpack.Error(`Unknown tensor data type '${tensor.datatype}'.`);
        }
        this.dataType = xnnpack.TensorType._types[tensor.datatype];
        this.shape = new xnnpack.TensorShape(Array.from(tensor.dims));
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

xnnpack.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions || [];
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

xnnpack.Tensor = class {

    constructor(tensor, reader) {
        this.type = new xnnpack.TensorType(tensor);
        this.values = reader.constant(tensor.constant_buffer_idx);
        this.encoding = '<';
    }
};

xnnpack.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading XNNPACK model.';
    }
};

vulkan.Reader = class {

    static open(data, target) {
        if (data.length >= 30) {
            const reader = base.BinaryReader.open(data);
            reader.skip(4);
            const magic = String.fromCharCode(...reader.read(4));
            if (magic === 'VH00') {
                return new vulkan.Reader(reader, target);
            }
        }
        return null;
    }

    constructor(reader, target) {
        this.reader = reader;
        this.target = target;
        reader.skip(2);
        this.flatbuffer = {
            offset: reader.uint32(),
            size: reader.uint32(),
        };
        this.constants = {
            offset: reader.uint32(),
            size: reader.uint32(),
        };
    }

    async read() {
        this.reader.seek(this.flatbuffer.offset);
        const metadata = new vulkan.Metadata(this.target.execution);
        metadata.register('conv_with_clamp(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max) -> Tensor)');
        const flatbuffers = await import('./flatbuffers.js');
        const data = this.reader.read(this.flatbuffer.size);
        const reader = flatbuffers.BinaryReader.open(data);
        if (!executorch.schema.vkgraph.VkGraph.identifier(reader)) {
            throw new xnnpack.Error('Invalid Vuklan data.');
        }
        this.graph = executorch.schema.vkgraph.VkGraph.create(reader);
        this.reader.seek(0);
        this.type = new vulkan.Graph(metadata, this.graph, this);
    }

    constant(id) {
        const constant = this.graph.constants[id];
        this.reader.seek(this.constants.offset + constant.offset.toNumber());
        const data = this.reader.read(constant.length.toNumber());
        this.reader.seek(0);
        return data;
    }
};

vulkan.Graph = class {

    constructor(metadata, graph, reader) {
        this.name = 'VulkanBackend';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (id) => {
            if (!values.has(id)) {
                const vkgraph = executorch.schema.vkgraph;
                const arg = graph.values[id].value;
                if (arg instanceof vkgraph.VkTensor) {
                    const type = new vulkan.TensorType(arg);
                    const initializer = arg.constant_id === -1 ? null : new vulkan.Tensor(arg, reader);
                    const value = new vulkan.Value(id.toString(), type, initializer);
                    values.set(id, { type: null, value: [value] });
                } else if (arg instanceof vkgraph.Int) {
                    values.set(id, { type: 'int64', value: arg.int_val });
                } else if (arg instanceof vkgraph.IntList) {
                    values.set(id, { type: 'int64[]', value: Array.from(arg.items) });
                } else if (arg instanceof vkgraph.Double) {
                    values.set(id, { type: 'float64', value: arg.double_val });
                } else if (arg instanceof vkgraph.Bool) {
                    values.set(id, { type: 'boolean', value: arg.bool_val });
                } else if (arg instanceof vkgraph.Null) {
                    values.set(id, { type: 'attribute', value: null });
                } else {
                    throw new Error(`Value type '${arg.constructor.name}' not implemented.`);
                }
            }
            return values.get(id);
        };
        for (let i = 0; i < graph.input_ids.length; i++) {
            const id = graph.input_ids[i];
            const value = values.map(id);
            const name = graph.input_ids.length === 1 ? 'input' : `input.${i}`;
            const argument = new vulkan.Argument(name, value.value, value.type);
            this.inputs.push(argument);
        }
        for (let i = 0; i < graph.output_ids.length; i++) {
            const id = graph.output_ids[i];
            const value = values.map(id);
            const name = graph.output_ids.length === 1 ? 'output' : `output.${i}`;
            const argument = new vulkan.Argument(name, value.value, value.type);
            this.outputs.push(argument);
        }
        for (const op of graph.chain) {
            const node = new vulkan.Node(metadata, op, values);
            this.nodes.push(node);
        }
    }
};

vulkan.Node = class {

    constructor(metadata, op, values) {
        const schema = metadata.type(op.name);
        if (!schema) {
            throw new vulkan.Error(`Operator schema for '${op.name}' not found.`);
        }
        this.type = {
            name: op.name.split(/\.([^.]*)$/)[0],
            identifier: op.name,
            category: schema.category || ''
        };
        this.name = op.node_id.toString();
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        for (let i = 0; i < op.args.length; i++) {
            const arg = op.args[i];
            const input = schema && i < schema.arguments.length;
            const def = input ? schema.arguments[i] : schema.returns[i - schema.arguments.length];
            const value = values.map(arg);
            const argument = new vulkan.Argument(def.name || '', value.value, value.type);
            if (input) {
                this.inputs.push(argument);
            } else {
                this.outputs.push(argument);
            }
        }

    }
};

vulkan.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

vulkan.Value = class Value {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new executorch.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer && initializer.type ? initializer.type : type || null;
        this.initializer = initializer || null;
    }
};

vulkan.TensorType = class {

    constructor(tensor) {
        const types = ['bool', 'uint8', 'int8', 'int32', 'float16', 'float32'];
        if (tensor.datatype >= types.length) {
            throw new vulkan.Error(`Unknown tensor data type '${tensor.datatype}'.`);
        }
        this.dataType = types[tensor.datatype];
        this.shape = new vulkan.TensorShape(Array.from(tensor.dims));
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

vulkan.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions || [];
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

vulkan.Tensor = class {

    constructor(tensor, reader) {
        this.type = new vulkan.TensorType(tensor);
        this.values = reader.constant(tensor.constant_id);
        this.encoding = '<';
    }
};

vulkan.Metadata = class {

    constructor(execution) {
        this.execution = execution;
    }

    register(signature) {
        const torch = this.execution.register('torch');
        const registry = torch._C.getRegistry();
        const schema = torch.FunctionSchema.parse(signature);
        const op = new torch._C.Operator(schema);
        registry.registerOperator(op);
    }

    type(identifier) {
        identifier = identifier.split(/\.([^.]*)$/);
        const name = identifier[0].replace('.', '::');
        const overload = identifier[1] === 'default' ? '' : identifier[1];
        const schemas = this.execution.invoke('torch._C._jit_get_schemas_for_operator', [name]);
        const schema = schemas.find((schema) => schema.name === name && schema.overload_name === overload);
        return schema;
    }
};

vulkan.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Vulkan model.';
    }
};

coreml.Reader = class {

    static open(data, target) {
        const reader = base.BinaryReader.open(data);
        return new coreml.Reader(reader, target);
    }

    constructor(reader, target) {
        this.reader = reader;
        this.target = target;
    }

    async factory() {
        const coreml = await import('./coreml.js');
        return new coreml.ModelFactory();
    }

    async read() {
        const entries = this.entries(this.reader);
        const factory = await this.factory();
        const protobuf = await import('./protobuf.js');
        for (const [key, value] of entries) {
            const path = key.split('/');
            const identifier = path.pop();
            const folder = path.length === 0 ? '' : `${path.join('/')}/`;
            const locals = new Map(Array.from(entries).filter(([key]) => key.startsWith(folder)).map(([key, value]) => [key.substring(folder.length), value]));
            const context = new coreml.Context(this, identifier, value, locals, protobuf);
            /* eslint-disable no-await-in-loop */
            const type = await factory.match(context);
            /* eslint-enable no-await-in-loop */
            if (type === 'coreml.manifest') {
                /* eslint-disable no-await-in-loop */
                const model = await factory.open(context);
                /* eslint-enable no-await-in-loop */
                [this.type] = model.graphs;
                this.type.name = 'CoreMLBackend';
                return;
            }
        }
    }

    stream(offset, size) {
        this.reader.seek(offset);
        const stream = this.reader.stream(size);
        this.reader.seek(0);
        return stream;
    }

    entries(reader) {
        const files = new Map();
        reader.seek(reader.length - 1);
        const str = [];
        let depth = 0;
        do {
            const c = String.fromCharCode(reader.byte());
            reader.skip(-2);
            if (c === '{') {
                depth++;
            } else if (c === '}') {
                depth--;
            }
            str.push(c);
        } while (depth > 0);
        const metadata = JSON.parse(str.join(''));
        const nodes = metadata.nodes;
        const roots = Array.from(nodes);
        for (const root of roots) {
            if (root !== null) {
                for (const index of Object.values(root.children)) {
                    roots[index] = null;
                }
            }
        }
        const process = (path, node) => {
            path = path ? `${path}/${node.name}` : node.name;
            if (node.kind === 0) {
                files.set(path, node.dataRegion);
            } else if (node.kind === 1) {
                for (const index of Object.values(node.children)) {
                    process(path, nodes[index]);
                }
            } else {
                throw new Error(`Node kind '${node.kind}' not implemented.`);
            }
        };
        for (const root of roots.filter((node) => node !== null)) {
            process('', root);
        }
        return files;
    }
};

coreml.Context = class {

    constructor(reader, identifier, location, entries, protobuf) {
        this._reader = reader;
        this._location = location;
        this._identifier = identifier;
        this._entries = entries;
        this._protobuf = protobuf;
    }

    get identifier() {
        return this._identifier;
    }

    get stream() {
        if (!this._stream) {
            this._stream = this._reader.stream(this._location.offset, this._location.size);
        }
        return this._stream;
    }

    async tags(type) {
        if (type === 'pb' && this.identifier.endsWith('.mlmodel')) {
            return new Map([[1,0],[2,2]]);
        }
        return new Map();
    }

    async peek(type) {
        if (type === 'json') {
            const data = this.stream.peek();
            const decoder = new TextDecoder('utf-8');
            const text = decoder.decode(data);
            return JSON.parse(text);
        }
        return null;
    }

    async read(type) {
        if (type === 'protobuf.binary') {
            return this._protobuf.BinaryReader.open(this.stream);
        }
        return null;
    }

    async fetch(file) {
        if (this._entries.has(file)) {
            const location = this._entries.get(file);
            const identifier = file.split('/').pop();
            return new coreml.Context(this._reader, identifier, location, this._entries, this._protobuf);
        }
        return null;
    }

    async require(id) {
        return this._reader.target.context.require(id);
    }

    async metadata(name) {
        return this._reader.target.context.metadata(name);
    }

    set(type, value) {
        this.type = type;
        this.value = value;
        return type;
    }
};

export const ModelFactory = executorch.ModelFactory;