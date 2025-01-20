
// Experimental

const executorch = {};

import * as flatbuffers from './flatbuffers.js';
import * as python from './python.js';
import * as pytorch from './pytorch.js';

executorch.ModelFactory = class {

    match(context) {
        const container = executorch.Container.open(context);
        if (container) {
            context.type = 'executorch';
            context.target = container;
        }
    }

    async open(context) {
        executorch.schema = await context.require('./executorch-schema');
        executorch.schema = executorch.schema.executorch_flatbuffer;
        const metadata = await pytorch.Metadata.open(context);
        const execution = new python.Execution();
        metadata.register(execution);
        const target = context.target;
        await target.read();
        return new executorch.Model(execution, target);
    }
};

executorch.Model = class {

    constructor(execution, context) {
        this.format = `ExecuTorch v${context.program.version}`;
        this.graphs = [];
        for (const plan of context.program.execution_plan) {
            for (const chain of plan.chains) {
                const graph = new executorch.Graph(execution, context, plan, chain);
                this.graphs.push(graph);
            }
        }
    }
};

executorch.Graph = class {

    constructor(execution, context, plan, chain) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (index, output) => {
            if (!values.has(index)) {
                const v = plan.values[index].val;
                const tensor = v instanceof executorch.schema.Tensor || v instanceof executorch.schema.TensorList || v instanceof executorch.schema.OptionalTensorList;
                if (output && !tensor) {
                    const value = [new executorch.Value(index.toString(), null, null)];
                    values.set(index, { type: null, value });
                } else if (tensor) {
                    const tensors = v instanceof executorch.schema.Tensor ? [v] : Array.from(v.items).map((arg) => plan.values[arg].val);
                    const list = [];
                    for (let i = 0; i < tensors.length; i++) {
                        const tensor = tensors[i];
                        const type = new executorch.TensorType(tensor);
                        let initializer = null;
                        if (v.data_buffer_idx > 0) {
                            initializer = new executorch.Tensor(tensor, context);
                        }
                        const identifier = tensors.length > 1 ? `${index}.${i}` : index.toString();
                        const value = new executorch.Value(identifier, type, initializer);
                        list.push(value);
                    }
                    values.set(index, { type: null, value: list });
                } else if (v instanceof executorch.schema.Bool) {
                    values.set(index, { type: 'int64', value: v.bool_val });
                } else if (v instanceof executorch.schema.Int) {
                    values.set(index, { type: 'int64', value: v.int_val });
                } else if (v instanceof executorch.schema.IntList) {
                    const list = v.items.map((index) => plan.values[index].val.int_val);
                    values.set(index, { type: 'int64[]', value: list });
                } else if (v instanceof executorch.schema.Double) {
                    values.set(index, { type: 'float64', value: v.double_val });
                } else if (v instanceof executorch.schema.Null) {
                    values.set(index, { type: 'attribute', value: null });
                } else {
                    throw new Error('Value type not implemented.');
                }
            }
            return values.get(index);
        };
        for (const input of plan.inputs) {
            const value = values.map(input);
            const argument = new executorch.Argument(input.toString(), value.value, value.type);
            this.inputs.push(argument);
        }
        for (const output of plan.outputs) {
            const value = values.map(output);
            const argument = new executorch.Argument(output.toString(), value.value, value.type);
            this.outputs.push(argument);
        }
        for (const instruction of chain.instructions) {
            const node = new executorch.Node(execution, context, plan, chain, instruction, values);
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

    constructor(execution, context, plan, chain, instruction, values) {
        this.name = '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const instr_args = instruction.instr_args;
        if (instr_args instanceof executorch.schema.KernelCall) {
            const op = plan.operators[instr_args.op_index];
            const name = op.name.split('::').pop();
            const identifier = op.overload ? `${op.name}.${op.overload}` : op.name;
            const schemas = execution.invoke('torch._C._jit_get_schemas_for_operator', [op.name]);
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
        } else if (instr_args instanceof executorch.schema.DelegateCall) {
            const delegate = plan.delegates[instr_args.delegate_index];
            const args = instr_args.args;
            const name = delegate.id;
            this.type = { name };
            let data = null;
            const DataLocation = executorch.schema.DataLocation;
            switch (delegate.processed.location) {
                case DataLocation.INLINE: {
                    data = context.program.backend_delegate_data[delegate.processed.index].data;
                    break;
                }
                case DataLocation.SEGMENT: {
                    // const segment = program.segments[delegate.processed.index];
                    break;
                }
                default: {
                    throw new executorch.Error(`Delegate data location '${delegate.processed.location}' not implemented.`);
                }
            }
            switch (name) {
                case 'XnnpackBackend': {
                    const input = values.map(args[0]);
                    const output = values.map(args[1], true);
                    this.inputs.push(new executorch.Argument('input', input.value, input.type));
                    this.outputs.push(new executorch.Argument('output', output.value, output.type));
                    flatbuffers.BinaryReader.open(data);
                    // executorch/backends/xnnpack/serialization/schema.fbs
                    break;
                }
                case 'CoreMLBackend': {
                    const input = values.map(args[0]);
                    const output = values.map(args[1], true);
                    this.inputs.push(new executorch.Argument('input', input.value, input.type));
                    this.outputs.push(new executorch.Argument('output', output.value, output.type));
                    break;
                }
                case 'VulkanBackend': {
                    const input = values.map(args[0]);
                    const output = values.map(args[1], true);
                    this.inputs.push(new executorch.Argument('input', input.value, input.type));
                    this.outputs.push(new executorch.Argument('output', output.value, output.type));
                    break;
                }
                default: {
                    throw new executorch.Error(`ExecuTorch delegate '${name}' not implemented.`);
                }
            }
            for (const spec of delegate.compile_specs) {
                const value = ArrayBuffer.isView(spec.value) ? Array.from(spec.value) : spec.value;
                const attribute = new executorch.Argument(spec.key, value);
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

    constructor(tensor, context) {
        this.type = new executorch.TensorType(tensor);
        const data_buffer_idx = tensor.data_buffer_idx;
        if (tensor.extra_tensor_info) {
            throw new executorch.Error('Extra tensor info not implemented.');
        } else if (context.program.constant_buffers) {
            throw new executorch.Error('Constant buffers not implemented.');
        } else if (tensor.allocation_info === null) {
            const constant_segment = context.program.constant_segment;
            const data_segment = context.program.segments[constant_segment.segment_index];
            const offset = constant_segment.offsets[data_buffer_idx].toNumber();
            const next = data_buffer_idx + 1 < constant_segment.offsets.length ? constant_segment.offsets[data_buffer_idx + 1].toNumber() : data_segment.size.toNumber();
            const size = next - offset;
            const reader = context.reader;
            reader.seek(context.extended_file_header.segment_base_offset + data_segment.offset.toNumber() + offset);
            this.encoding = '<';
            this.values = reader.read(size);
            reader.seek(0);
        } else {
            throw new executorch.Error('Tensor allocation info not implemented.');
        }
    }
};

executorch.Container = class {

    static open(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'ET12') {
            return new executorch.Container(context, reader);
        }
        return null;
    }

    constructor(context, reader) {
        this.context = context;
        this.reader = reader;
    }

    async read() {
        this.program = executorch.schema.Program.create(this.reader);
        this.reader = this.context.read('binary');
        if (this.reader.length >= 32) {
            this.reader.seek(8);
            const magic = String.fromCharCode(...this.reader.read(4));
            if (magic === 'eh00') {
                this.extended_file_header = {
                    magic,
                    length: this.reader.uint32(),
                    program_size: this.reader.uint64().toNumber(),
                    segment_base_offset: this.reader.uint64().toNumber(),
                };
            }
            this.reader.seek(0);
        }
    }
};

executorch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ExecuTorch model.';
    }
};

export const ModelFactory = executorch.ModelFactory;
