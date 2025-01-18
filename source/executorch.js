
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
        values.map = (arg) => {
            if (!values.has(arg)) {
                const v = plan.values[arg].val;
                if (v instanceof executorch.schema.Tensor || v instanceof executorch.schema.TensorList) {
                    const tensors = v instanceof executorch.schema.Tensor ? [v] : Array.from(v.items).map((arg) => plan.values[arg].val);
                    const list = [];
                    for (let i = 0; i < tensors.length; i++) {
                        const tensor = tensors[i];
                        const type = new executorch.TensorType(tensor);
                        let initializer = null;
                        if (v.data_buffer_idx > 0) {
                            initializer = new executorch.Tensor(tensor);
                        }
                        const identifier = tensors.length > 1 ? `${arg}.${i}` : arg.toString();
                        list.push(new executorch.Value(identifier, type, initializer));
                    }
                    values.set(arg, { type: null, value: list });
                } else if (v instanceof executorch.schema.Bool) {
                    values.set(arg, { type: 'int64', value: v.bool_val });
                } else if (v instanceof executorch.schema.Int) {
                    values.set(arg, { type: 'int64', value: v.int_val });
                } else if (v instanceof executorch.schema.IntList) {
                    const list = v.items.map((index) => plan.values[index].val.int_val);
                    values.set(arg, { type: 'int64[]', value: list });
                } else if (v instanceof executorch.schema.Double) {
                    values.set(arg, { type: 'float64', value: v.double_val });
                } else if (v instanceof executorch.schema.Null) {
                    values.set(arg, { type: 'attribute', value: null });
                } else {
                    throw new Error('Value type not implemented.');
                }
            }
            return values.get(arg);
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
        const instr_args = instruction.instr_args;
        if (instr_args instanceof executorch.schema.KernelCall) {
            const op = plan.operators[instr_args.op_index];
            const name = op.name.split('::').pop();
            const identifier = op.overload ? `${op.name}.${op.overload}` : op.name;
            const schemas = execution.invoke('torch._C._jit_get_schemas_for_operator', [op.name]);
            const schema = schemas.find((schema) => schema.name === op.name && schema.overload_name === op.overload);
            const category = schema && schema.category ? schema.category : '';
            const alias = (arg) => arg && arg.alias_info && arg.alias_info.before_set.length === 1 ? arg.alias_info.before_set[0] : null;
            const outputs = new Set(schema && Array.isArray(schema.returns) ? schema.returns.map((arg) => alias(arg)).filter((alias) => alias !== null) : []);
            const inputs = new Map();
            this.type = { name, identifier, category };
            let i = 0;
            const args = instr_args.args;
            for (; i < schema.arguments.length; i++) {
                const v = args[i];
                const arg = schema && i < schema.arguments.length ? schema.arguments[i] : null;
                const output = arg ? alias(schema.arguments[i]) : null;
                if (output && outputs.has(output)) {
                    inputs.set(output, v);
                    continue;
                }
                const name = arg ? arg.name : i.toString();
                const value = values.map(v);
                const argument = new executorch.Argument(name, value.value, value.type);
                this.inputs.push(argument);
            }
            for (let j = 0; j < schema.returns.length; j++) {
                const ret = schema.returns[j];
                const output = alias(ret);
                const v = output && inputs.has(output) ? inputs.get(output) : args[i++];
                const name = ret.name;
                const value = values.map(v);
                const argument = new executorch.Argument(name || '', value.value, value.type);
                this.outputs.push(argument);
            }
        } else if (instr_args instanceof executorch.schema.DelegateCall) {
            const delegate = plan.delegates[instr_args.delegate_index];
            const name = delegate.id;
            this.type = { name };
        } else {
            throw new Error('Instruction argument not implemented.');
        }
    }
};

executorch.TensorType = class {

    constructor(tensor) {
        const ScalarType = executorch.schema.ScalarType;
        switch (tensor.scalar_type) {

            case ScalarType.BOOL: this.dataType = 'boolean'; break;
            case ScalarType.BYTE: this.dataType = 'uint8'; break;
            case ScalarType.CHAR: this.dataType = 'int8'; break;
            case ScalarType.SHORT: this.dataType = 'int16'; break;
            case ScalarType.INT: this.dataType = 'int32'; break;
            case ScalarType.LONG: this.dataType = 'int64'; break;
            case ScalarType.HALF: this.dataType = 'float16'; break;
            case ScalarType.FLOAT: this.dataType = 'float32'; break;
            case ScalarType.DOUBLE: this.dataType = 'float64'; break;
            case ScalarType.UINT16: this.dataType = 'uint16'; break;
            case ScalarType.UINT32: this.dataType = 'uint32'; break;
            case ScalarType.UINT64: this.dataType = 'uint64'; break;
            default: throw new executorch.Error(`Unknown tensor data type '${tensor.scalar_type}'.`);
        }
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
