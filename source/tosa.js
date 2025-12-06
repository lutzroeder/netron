// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

/* jshint esversion: 6 */
import * as version from './tosa-version.js';

const tosa = {};

tosa.ModelFactory = class {

    async match(context) {
        const reader = await context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'TOSA') {
            return context.set('tosa.flatbuffers', reader);
        }
        const obj = await context.peek('json');
        if (obj && obj.regions && obj.version) {
            return context.set('tosa.flatbuffers.json', obj);
        }
        return null;
    }

    async open(context) {
        const identifier = context.identifier;
        let model = null;
        let file_version = null;
        let schema_version = null;
        const loadSchema = async (version) => {
            file_version = [version._major, version._minor, version._patch].join('.');
            schema_version = [version._major, version._minor].join('.');  // Ignore patch version
            const schema = await context.require(`./tosa-schema-v${schema_version}`);
            return schema.tosa;
        };
        try {
            switch (context.type) {
                case 'tosa.flatbuffers': {
                    const reader = context.value;
                    model = version.tosa.TosaGraph.create(reader);
                    tosa.schema = await loadSchema(model.version);
                    model = tosa.schema.TosaGraph.create(reader);
                    break;
                }
                case 'tosa.flatbuffers.json': {
                    const reader = await context.read('flatbuffers.text');
                    model = version.tosa.TosaGraph.createText(reader);
                    tosa.schema = await loadSchema(model.version);
                    model = tosa.schema.TosaGraph.createText(reader);
                    break;
                }
                default: {
                    throw new tosa.Error(`Unsupported TOSA format '${context.type}'.`);
                }
            }
        } catch (error) {
            if (error && error.code && error.code === 'ERR_MODULE_NOT_FOUND') {
                throw new tosa.Error(`'${identifier}' file version (${file_version}) does not match any available schema version.`);
            } else {
                const message = error && error.message ? error.message : error.toString();
                throw new tosa.Error(`${message.replace(/\.$/, '')} in '${identifier}'.`);
            }
        }
        tosa.schema.InverseOp = new Map(Object.entries(tosa.schema.Op).map(([key, value]) => [value, key]));
        tosa.version = schema_version;
        const metadata = await context.metadata(`tosa-metadata-v${schema_version}.json`);
        try {
            return new tosa.Model(metadata, model, file_version);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new tosa.Error(`${message.replace(/\.$/, '')} in '${identifier}'.`);
        }
    }
};

tosa.Model = class {

    constructor(metadata, model, version) {
        this.format = `TOSA v${version}`;
        this.modules = [];
        for (const region of model.regions) {
            for (const block of region.blocks) {
                this.modules.push(new tosa.Graph(metadata, block));
            }
        }
    }
};

tosa.Graph = class {

    constructor(metadata, graph) {
        this.name = graph.name;
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];

        // populate tensors
        const tensors = Object.fromEntries(graph.tensors.map((tensor, index) => [tensor.name, new tosa.Value(index, tensor)]));
        // populate shapes
        const shapes = graph.shapes ? Object.fromEntries(graph.shapes.map((shape, index) => [shape.name, new tosa.Value(index, shape)])) : undefined;
        // populate operator nodes
        this.nodes = graph.operators.map((operator, index) => {
            return new tosa.Node(metadata, operator, index.toString(), tensors, shapes);
        }).filter((node) => !node.isConst);

        // populate inputs and outputs
        this.inputs = graph.inputs.map((input) => new tosa.Parameter(input, [tensors[input]]));
        this.outputs = graph.outputs.map((output) => new tosa.Parameter(output, [tensors[output]]));
    }
};

tosa.Node = class {

    constructor(metadata, operator, identifier, tensors, shapes) {
        const opType = tosa.schema.InverseOp.get(operator.op) || 'UNKNOWN';
        this.type = metadata.type(opType);
        this.identifier = identifier;
        this.outputs = [];
        this.inputs = [];
        this.attributes = [];

        if (operator) {
            const outputs = Array.from(operator.outputs || new Int32Array(0));
            for (let i = 0; i < outputs.length; i++) {
                let name = `output-${i}`;
                let value = tensors[outputs[i]];
                if (this.type && this.type.outputs && i <  this.type.outputs.length) {
                    const id = outputs[i];
                    const output = this.type.outputs[i];
                    name = output.name;
                    if (shapes && output.type === "shape_t" && shapes[id]) {
                        value = shapes[id];
                    }
                }
                this.outputs.push(new tosa.Parameter(name, [value]));
            }

            const inputs = Array.from(operator.inputs || new Int32Array(0));
            for (let i = 0; i < inputs.length; i++) {
                let name = `input-${i}`;
                let value = tensors[inputs[i]];
                if (this.type && this.type.inputs && i <  this.type.inputs.length) {
                    const id = inputs[i];
                    const input = this.type.inputs[i];
                    name = input.name;
                    if (shapes && input.type === "shape_t" && shapes[id]) {
                        value = shapes[id];
                    }
                }
                this.inputs.push(new tosa.Parameter(name, [value]));
            }

            const attributes = operator.attribute;
            if (attributes) {
                for (const [name, value] of Object.entries(attributes)) {
                    const schema = metadata.attribute(opType, name);
                    const type = schema ? schema.type : '?';
                    this.attributes.push(new tosa.Attribute(name, type, this.getTypedValue(value, type)));
                }
            }
        }
    }

    get isConst() {
        return this.type.name.startsWith('CONST');
    }

    getTypedValue(value, type) {
        const values = typeof value === 'object' ? Object.values(value) : value;
        if (Array.isArray(values) && values.length > 1) {
            return `[${values.join(", ")}]`;
        }
        // Translate type to enum
        const tr = new Map([
            ['var_t', 'DType'],
            ['acc_type_t', 'DType'],
            ['resize_mode_t', 'ResizeMode'],
            ['nan_propagation_mode_t', 'NanPropagationMode'],
            ['rounding_mode_t', 'RoundingMode']
        ]);
        return tosa.Utility.enum(tr.has(type) ? tr.get(type) : type, values);
    }
};

tosa.Attribute = class {

    constructor(name, type, value) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

tosa.Parameter = class {

    constructor(name, args) {
        this.name = name;
        this.value = args;
    }
};

tosa.Value = class {

    constructor(index, arg) {
        this.name = arg.name;
        this.identifier = index.toString();
        if (Object.getPrototypeOf(arg).constructor.name === 'TosaShape') {
            this.initializer = new tosa.ConstShape(arg);
            this.type = this.initializer.type;
        } else {
            if (arg.data && arg.data.length || arg.variable) {
                this.initializer = new tosa.ConstTensor(arg);
            }
            this.type = this.initializer ? this.initializer.type : new tosa.TensorType(arg);
            this.visible = arg.variable;
        }
    }
};

tosa.ConstShape = class {

    constructor(shape) {
        this.name = shape.name;
        this.type = new tosa.TensorType({ 'type': tosa.schema.DType.SHAPE, 'shape': [shape.rank] });
        this.values = shape.data;
    }
};

tosa.ConstTensor = class {

    constructor(tensor) {
        this.name = tensor.name;
        this.type = new tosa.TensorType(tensor);
        this.values = tensor.data;
        this.category = tensor.variable ? 'Variable' : null;
    }
};

tosa.TensorType = class {

    constructor(tensor) {
        this.dataType = tosa.Utility.dataType(tensor.type);
        this.shape = new tosa.TensorShape(tensor.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

tosa.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

tosa.Utility = class {

    static dataType(type) {
        tosa.Utility._tensorTypes = tosa.Utility._tensorTypes || new Map();
        const tensorTypes = tosa.Utility._tensorTypes;
        if (!tensorTypes.has(tosa.version)) {
            const tr = new Map([
                ['fp', 'float'],
                ['bf', 'bfloat'],
                ['shape', 'int64'],
                ['8e4m3', '8e4m3fn'],
                ['bool', 'boolean']
            ]);
            const re = new RegExp(Array.from(tr.keys()).join('|'), 'g');
            const map = new Map(Object.entries(tosa.schema.DType).map(([key, value]) => [value, key.toLowerCase().replace(re, (k) => tr.get(k))]));
            tensorTypes.set(tosa.version, map);
        }
        const tt = tensorTypes.get(tosa.version);
        return tt.has(type) ? tt.get(type) : '?';
    }

    static enum(name, value) {
        const type = name && tosa.schema ? tosa.schema[name] : undefined;
        if (type) {
            tosa.Utility._enums = tosa.Utility._enums || new Map();
            let enums = tosa.Utility._enums;
            if (!enums.has(tosa.version)) {
                enums.set(tosa.version, new Map());
            }
            enums = enums.get(tosa.version);
            if (!enums.has(name)) {
                const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
                enums.set(name, entries);
            }
            const map = enums.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }
};

tosa.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TOSA model.';
    }
};

export const ModelFactory = tosa.ModelFactory;
