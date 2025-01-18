
// Experimental

const executorch = {};

import * as python from './python.js';
import * as pytorch from './pytorch.js';

executorch.ModelFactory = class {

    match(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'ET12') {
            context.type = 'executorch';
            context.target = reader;
        }
    }

    async open(context) {
        executorch.schema = await context.require('./executorch-schema');
        executorch.schema = executorch.schema.executorch_flatbuffer;
        const metadata = await pytorch.Metadata.open(context);
        const execution = new python.Execution();
        metadata.register(execution);
        const reader = context.target;
        const program = executorch.schema.Program.create(reader);
        return new executorch.Model(execution, program);
    }
};

executorch.Model = class {

    constructor(execution, program) {
        this.format = `ExecuTorch v${program.version}`;
        this.graphs = [];
        for (const plan of program.execution_plan) {
            for (const chain of plan.chains) {
                const graph = new executorch.Graph(execution, chain, plan);
                this.graphs.push(graph);
            }
        }
    }
};

executorch.Graph = class {

    constructor(execution, chain, plan) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (index, output) => {
            if (!values.has(index)) {
                const v = plan.values[index].val;
                const tensor = v instanceof executorch.schema.Tensor || v instanceof executorch.schema.TensorList;
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
                            initializer = new executorch.Tensor(tensor);
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
            const node = new executorch.Node(execution, instruction, plan, values);
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

    constructor(execution, instruction, plan, values) {
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
            switch (name) {
                case 'XnnpackBackend': {
                    const input = values.map(args[0]);
                    const output = values.map(args[1], true);
                    this.inputs.push(new executorch.Argument('input', input.value, input.type));
                    this.outputs.push(new executorch.Argument('output', output.value, output.type));
                    break;
                }
                case 'CoreMLBackend': {
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
        this.dataType = executorch.TensorType._types.length[tensor.scalar_type];
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

    constructor(tensor) {
        this.type = new executorch.TensorType(tensor);
    }
};

executorch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ExecuTorch model.';
    }
};

export const ModelFactory = executorch.ModelFactory;
