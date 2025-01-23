
// Experimental

const executorch = {};
const coreml = {};
const vulkan = {};
const xnnpack = {};

import * as base from './base.js';
import * as python from './python.js';
import * as pytorch from './pytorch.js';

executorch.ModelFactory = class {

    match(context) {
        const reader = executorch.Reader.open(context);
        if (reader) {
            context.type = 'executorch';
            context.target = reader;
        }
    }

    async open(context) {
        executorch.schema = await context.require('./executorch-schema');
        const target = context.target;
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
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (index, output) => {
            if (!values.has(index)) {
                const executorch_flatbuffer = executorch.schema.executorch_flatbuffer;
                const v = plan.values[index].val;
                const tensor = v instanceof executorch_flatbuffer.Tensor || v instanceof executorch_flatbuffer.TensorList || v instanceof executorch_flatbuffer.OptionalTensorList;
                if (output && !tensor) {
                    const value = [new executorch.Value(index.toString(), null, null)];
                    values.set(index, { type: null, value });
                } else if (tensor) {
                    const tensors = v instanceof executorch_flatbuffer.Tensor ? [v] : Array.from(v.items).map((arg) => plan.values[arg].val);
                    const list = [];
                    for (let i = 0; i < tensors.length; i++) {
                        const tensor = tensors[i];
                        const type = new executorch.TensorType(tensor);
                        let initializer = null;
                        if (v.data_buffer_idx > 0) {
                            initializer = new executorch.Tensor(tensor, target);
                        }
                        const identifier = tensors.length > 1 ? `${index}.${i}` : index.toString();
                        const value = new executorch.Value(identifier, type, initializer);
                        list.push(value);
                    }
                    values.set(index, { type: null, value: list });
                } else if (v instanceof executorch_flatbuffer.Bool) {
                    values.set(index, { type: 'int64', value: v.bool_val });
                } else if (v instanceof executorch_flatbuffer.Int) {
                    values.set(index, { type: 'int64', value: v.int_val });
                } else if (v instanceof executorch_flatbuffer.IntList) {
                    const list = v.items.map((index) => plan.values[index].val.int_val);
                    values.set(index, { type: 'int64[]', value: list });
                } else if (v instanceof executorch_flatbuffer.Double) {
                    values.set(index, { type: 'float64', value: v.double_val });
                } else if (v instanceof executorch_flatbuffer.Null) {
                    values.set(index, { type: 'attribute', value: null });
                } else {
                    throw new Error('Value type not implemented.');
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
            switch (delegate.id) {
                case 'XnnpackBackend': {
                    this.type = delegate.backend.type || { name: delegate.id };
                    for (const arg of args.slice(0, this.type.inputs.length)) {
                        const value = values.map(arg);
                        const argument = new executorch.Argument('', value.value, value.type);
                        this.inputs.push(argument);
                    }
                    for (const arg of args.slice(this.type.inputs.length, this.type.inputs.length + this.type.outputs.length)) {
                        const value = values.map(arg);
                        const argument = new executorch.Argument('', value.value, value.type);
                        this.outputs.push(argument);
                    }
                    break;
                }
                case 'CoreMLBackend': {
                    this.type = delegate.backend.type || { name: delegate.id };
                    const input = values.map(args[0]);
                    const output = values.map(args[1], true);
                    this.inputs.push(new executorch.Argument('input', input.value, input.type));
                    this.outputs.push(new executorch.Argument('output', output.value, output.type));
                    break;
                }
                case 'VulkanBackend': {
                    this.type = delegate.backend.type || { name: delegate.id };
                    const input = values.map(args[0]);
                    const output = values.map(args[1], true);
                    this.inputs.push(new executorch.Argument('input', input.value, input.type));
                    this.outputs.push(new executorch.Argument('output', output.value, output.type));
                    break;
                }
                default: {
                    throw new executorch.Error(`ExecuTorch delegate '${delegate.id}' not implemented.`);
                }
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

    static open(context) {
        const reader = context.peek('flatbuffers.binary');
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
        this.metadata = await pytorch.Metadata.open(this.context);
        this.execution = new python.Execution();
        this.metadata.register(this.execution);
        const executorch_flatbuffer = executorch.schema.executorch_flatbuffer;
        this.program = executorch_flatbuffer.Program.create(this.reader);
        this.reader = this.context.read('binary');
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
        const values = new Map();
        values.map = (id) => {
            if (!values.has(id)) {
                const fb_xnnpack = executorch.schema.fb_xnnpack;
                const name = id.toString();
                const value = graph.xvalues[id].xvalue_union;
                if (value instanceof fb_xnnpack.XNNTensorValue) {
                    const type = new xnnpack.TensorType(value);
                    const initializer = value.constant_buffer_idx === 0 ? null : new xnnpack.Tensor(value, reader);
                    values.set(id, new xnnpack.Value(name, type, initializer));
                } else if (value instanceof fb_xnnpack.XNNQuantizedTensorValue) {
                    values.set(id, new xnnpack.Value(name, null, null));
                } else {
                    throw new xnnpack.Error('XNNPACK value type not implemented.');
                }
            }
            return values.get(id);
        };
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
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
        this.register('XNNStaticTranspose', 'Transform');
        this.register('_XNNNodeConv', 'Layer');
        this.register('XNNFullyConnected', 'Layer');
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
            'qdint8', 'qbint4'
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
};

vulkan.Graph = class {

    constructor(metadata, graph /*, reader */) {
        this.name = 'VulkanBackend';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        for (const op of graph.chain) {
            const node = new vulkan.Node(metadata, op);
            this.nodes.push(node);
        }
    }
};

vulkan.Node = class {

    constructor(metadata, op) {
        const schema = metadata.type(op.name);
        this.type = {
            name: schema.name,
            identifier: op.name
        };
        if (schema.category) {
            this.type.category = schema.category;
        }
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
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
            const identifier = key.split('/').pop();
            this.reader.seek(value.offset);
            const stream = this.reader.stream(value.size);
            this.reader.seek(0);
            const context = new coreml.Context(this.target.context, identifier, stream, entries, protobuf);
            factory.match(context);
            if (context.type === 'coreml.pb') {
                /* eslint-disable no-await-in-loop */
                const model = await factory.open(context);
                /* eslint-enable no-await-in-loop */
                [this.type] = model.graphs;
                this.type.name = 'CoreMLBackend';
                return;
            }
        }
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
        if (!Array.from(files.keys()).every((key) => key.startsWith('lowered_module/'))) {
            throw new executorch.Error('');
        }
        const entries = new Map(Array.from(files).map(([key, value]) => {
            return [key.replace(/^lowered_module\//, ''), value];
        }));
        return entries;
    }
};

coreml.Context = class {

    constructor(context, identifier, stream, entries, protobuf) {
        this.context = context;
        this.identifier = identifier;
        this.stream = stream;
        this.entries = entries;
        this.protobuf = protobuf;
    }

    tags(type) {
        if (type === 'pb' && this.identifier.endsWith('.mlmodel')) {
            return new Map([[1,0],[2,2]]);
        }
        return new Map();
    }

    peek(type) {
        if (type === 'json') {
            const data = this.stream.peek();
            const decoder = new TextDecoder('utf-8');
            const text = decoder.decode(data);
            return JSON.parse(text);
        }
        return null;
    }

    read(type) {
        if (type === 'protobuf.binary') {
            return this.protobuf.BinaryReader.open(this.stream);
        }
        return null;
    }

    async fetch(/* file */) {
        return Promise.resolve(null);
    }

    async require(id) {
        return this.context.require(id);
    }

    async metadata(name) {
        return this.context.metadata(name);
    }
};

export const ModelFactory = executorch.ModelFactory;