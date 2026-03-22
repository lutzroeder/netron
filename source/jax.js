
import * as base from './base.js';
import * as flatbuffers from './flatbuffers.js';

const jax = {};

jax.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const reader = flatbuffers.BinaryReader.open(stream);
            if (reader) {
                return context.set('jax.flatbuffers', reader);
            }
        }
        return null;
    }

    async open(context) {
        const reader = context.value;
        jax.schema = await context.require('./jax-schema');
        const exported = jax.schema.jax_export.serialization.Exported.create(reader);
        const mlir_module = exported.mlir_module_serialized;
        let model = null;
        if (mlir_module && mlir_module.length > 0) {
            const buffer = new Uint8Array(mlir_module.buffer, mlir_module.byteOffset, mlir_module.byteLength);
            const stream = new base.BinaryStream(buffer);
            const content = context.context('module.mlirbc', stream);
            const mlir = await import('./mlir.js');
            const factory = new mlir.ModelFactory();
            const type = await factory.match(content);
            if (type) {
                model = await factory.open(content);
            }
        }
        if (!model) {
            model = new jax.Model(exported);
        }
        model.format = `JAX Export v${exported.serialization_version}`;
        if (exported.function_name) {
            model.metadata.push({ name: 'function_name', value: exported.function_name });
        }
        if (exported.calling_convention_version) {
            model.metadata.push({ name: 'calling_convention_version', value: exported.calling_convention_version.toString() });
        }
        return model;
    }
};

jax.Model = class {

    constructor(exported) {
        this.metadata = [];
        this.modules = [new jax.Graph(exported)];
    }
};

jax.Graph = class {

    constructor(exported) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        if (exported.in_avals) {
            for (let i = 0; i < exported.in_avals.length; i++) {
                const aval = exported.in_avals[i];
                const name = `input_${i}`;
                const type = aval ? new jax.TensorType(aval) : null;
                const value = new jax.Value(name, type);
                this.inputs.push(new jax.Argument(name, [value]));
            }
        }
        if (exported.out_avals) {
            for (let i = 0; i < exported.out_avals.length; i++) {
                const aval = exported.out_avals[i];
                const name = `output_${i}`;
                const type = aval ? new jax.TensorType(aval) : null;
                const value = new jax.Value(name, type);
                this.outputs.push(new jax.Argument(name, [value]));
            }
        }
    }
};

jax.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

jax.Value = class {

    constructor(name, type) {
        this.name = name;
        this.type = type || null;
    }
};

jax.TensorType = class {

    constructor(aval) {
        switch (aval.dtype) {
            case 1: this.dataType = 'bool'; break;
            case 2: this.dataType = 'int8'; break;
            case 3: this.dataType = 'int16'; break;
            case 4: this.dataType = 'int32'; break;
            case 5: this.dataType = 'int64'; break;
            case 6: this.dataType = 'uint8'; break;
            case 7: this.dataType = 'uint16'; break;
            case 8: this.dataType = 'uint32'; break;
            case 9: this.dataType = 'uint64'; break;
            case 10: this.dataType = 'float16'; break;
            case 11: this.dataType = 'float32'; break;
            case 12: this.dataType = 'float64'; break;
            case 13: this.dataType = 'complex64'; break;
            case 14: this.dataType = 'complex128'; break;
            case 15: this.dataType = 'bfloat16'; break;
            case 16: this.dataType = 'int4'; break;
            case 17: this.dataType = 'uint4'; break;
            case 18: this.dataType = 'float8_e4m3b11fnuz'; break;
            case 19: this.dataType = 'float8_e4m3fn'; break;
            case 20: this.dataType = 'float8_e4m3fnuz'; break;
            case 21: this.dataType = 'float8_e5m2'; break;
            case 22: this.dataType = 'float8_e5m2fnuz'; break;
            default: throw new jax.Error(`Unsupported dtype '${aval.dtype}'.`);
        }
        this.shape = aval.shape && aval.shape.length > 0 ? { dimensions: Array.from(aval.shape) } : null;
    }

    toString() {
        return this.dataType + (this.shape ? `[${this.shape.dimensions.join(',')}]` : '');
    }
};

jax.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading JAX model.';
    }
};

export const ModelFactory = jax.ModelFactory;
