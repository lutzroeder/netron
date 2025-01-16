
const tvm = {};

tvm.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.peek('json');
            if (obj && Array.isArray(obj.nodes) && Array.isArray(obj.arg_nodes) && Array.isArray(obj.heads) &&
                obj.nodes.every((node) => node && (node.op === 'null' || node.op === 'tvm_op'))) {
                context.type = 'tvm.json';
                context.target = obj;
                return;
            }
        }
        const stream = context.stream;
        const signature = [0xB7, 0x9C, 0x04, 0x05, 0x4F, 0x8D, 0xE5, 0xF7];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            context.type = 'tvm.params';
        }
    }

    filter(context, type) {
        return context.type !== 'tvm.json' || type !== 'tvm.params';
    }

    async open(context) {
        const metadata = await context.metadata('tvm-metadata.json');
        let obj = null;
        let params = null;
        switch (context.type) {
            case 'tvm.json': {
                obj = context.target;
                const identifier = context.identifier.replace(/\.json$/, '.params');
                try {
                    const content = await context.fetch(identifier);
                    const reader = content.read('binary');
                    params = tvm.NDArray.loadParams(reader);
                } catch {
                    // continue regardless of error
                }
                break;
            }
            case 'tvm.params': {
                const identifier = context.identifier.replace(/\.params$/, '.json');
                try {
                    const content = await context.fetch(identifier);
                    obj = content.read('json');
                } catch {
                    // continue regardless of error
                }
                const reader = context.read('binary');
                params = tvm.NDArray.loadParams(reader);
                break;
            }
            default:
                throw new tvm.Error(`Unsupported TVN format '${context.type}'.`);
        }
        return new tvm.Model(metadata, obj, params);
    }
};

tvm.Model = class {

    constructor(metadata, obj, params) {
        this.format = 'TVM';
        this.graphs = [new tvm.Graph(metadata, obj, params)];
    }
};

tvm.Graph = class {

    constructor(metadata, obj, params) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const tensors = new Map();
        if (params) {
            for (const [name, value] of params) {
                const shape = new tvm.TensorShape(value.shape);
                const type = new tvm.TensorType(value.dtype, shape);
                const tensor = new tvm.Tensor(name, type, value.data);
                tensors.set(name, tensor);
            }
        }
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (!values.has(name)) {
                values.set(name, new tvm.Value(name, type || null, tensor || null));
            } else if (type || (tensor && tensor !== values.get(name).initializer)) {
                throw new tvm.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const updateOutput = (nodes, input) => {
            const [nodeIndex, outputIndex] = input;
            const node = nodes[nodeIndex];
            if (node) {
                while (outputIndex >= node.outputs.length) {
                    node.outputs.push([nodeIndex, node.outputs.length]);
                }
            }
            return [nodeIndex, outputIndex];
        };
        if (obj) {
            const nodes = obj.nodes;
            const inputs = {};
            const outputs = {};
            for (const node of nodes) {
                node.outputs = [];
            }
            for (const node of nodes) {
                node.inputs = node.inputs || [];
                node.inputs = node.inputs.map((input) => updateOutput(nodes, input));
            }
            const arg_nodes = new Map(obj.arg_nodes.map((index) => [index, index < nodes.length ? nodes[index] : null]));
            for (let i = 0; i < obj.heads.length; i++) {
                const head = obj.heads[i];
                const identifier = updateOutput(nodes, head);
                const name = `output${(i === 0) ? '' : (i + 1)}`;
                const signature = outputs[name];
                const type = signature && signature.data_shape ? new tvm.TensorType(-1, new tvm.TensorShape(signature.data_shape)) : null;
                const value = values.map(`[${identifier.join(',')}]`, type);
                const argument = new tvm.Argument(name, [value]);
                this.outputs.push(argument);
            }
            const filtered = nodes.filter((node, index) => !arg_nodes.has(index));
            const initializers = new Map();
            for (const node of filtered) {
                for (const input of node.inputs) {
                    const identifier = `[${input.join(',')}]`;
                    if (!initializers.has(identifier)) {
                        const [index] = input;
                        const arg_node = arg_nodes.get(index);
                        if (arg_node && arg_node.name && (!arg_node.inputs || arg_node.inputs.length === 0) && (arg_node.outputs && arg_node.outputs.length === 1)) {
                            if (tensors.has(arg_node.name)) {
                                initializers.set(identifier, tensors.get(arg_node.name));
                                arg_nodes.delete(index);
                            }
                        }
                    }
                }
                if (node.params) {
                    for (const param of node.params) {
                        values.map(param.id, null, tensors.get(param.id));
                    }
                }
            }
            for (const [, arg_node] of arg_nodes) {
                if (arg_node && (!arg_node.inputs || arg_node.inputs.length === 0) && (arg_node.outputs && arg_node.outputs.length === 1)) {
                    const identifier = `[${arg_node.outputs[0].join(',')}]`;
                    const name = arg_node.name;
                    const signature = inputs[name];
                    const type = signature && signature.data_shape ? new tvm.TensorType(-1, new tvm.TensorShape(signature.data_shape)) : null;
                    const value = values.map(identifier, type, tensors.get(identifier));
                    const argument = new tvm.Argument(name, [value]);
                    this.inputs.push(argument);
                }
            }
            for (const node of filtered) {
                this.nodes.push(new tvm.Node(metadata, node, initializers, values));
            }
        } else if (params) {
            const blocks = new Map();
            const separator = Array.from(params.keys()).every((key) => key.indexOf('_') !== -1) ? '_' : '';
            for (const [key] of params) {
                const parts = separator ? key.split(separator) : [key];
                let argumentName = parts.pop();
                if (key.endsWith('moving_mean') || key.endsWith('moving_var')) {
                    argumentName = [parts.pop(), argumentName].join(separator);
                }
                const nodeName = parts.join(separator);
                if (!blocks.has(nodeName)) {
                    blocks.set(nodeName, { name: nodeName, op: 'Weights', params: [] });
                }
                blocks.get(nodeName).params.push({ name: argumentName, id: key });
                values.map(key, null, tensors.get(key));
            }
            for (const block of blocks.values()) {
                const node = new tvm.Node(metadata, block, new Map(), values);
                this.nodes.push(node);
            }
        }
    }
};

tvm.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

tvm.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new tvm.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = !name && initializer && initializer.name ? initializer.name : name;
        this.type = !type && initializer && initializer.type ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

tvm.Node = class {

    constructor(metadata, node, initializers, values) {
        this.type = { name: node.op };
        this.name = node.name;
        this.attributes = Object.entries(node.attrs || {}).map(([name, value]) => new tvm.Argument(name, value));
        this.inputs = (node.inputs || []).map((input, index) => {
            const name = index.toString();
            const identifier = `[${input.join(',')}]`;
            const value = values.map(identifier, null, initializers.get(identifier));
            return new tvm.Argument(name, [value]);
        });
        this.outputs = (node.outputs || []).map((output, index) => {
            const name = index.toString();
            const value = values.map(`[${output.join(',')}]`);
            return new tvm.Argument(name, [value]);
        });
        for (const param of node.params || []) {
            const value = values.map(param.id);
            const argument = new tvm.Argument(param.name, [value]);
            this.inputs.push(argument);
        }
    }
};

tvm.Tensor = class {

    constructor(name, type, data) {
        this.name = name;
        this.type = type;
        this.values = data;
        this.encoding = '<';
    }
};

tvm.TensorType = class {

    constructor(dtype, shape) {
        let type = '';
        switch (dtype.code) { // TVMArgTypeCode
            case 0: type = 'int'; break;
            case 1: type = 'uint'; break;
            case 2: type = 'float'; break;
            default: throw new tvm.Error(`Unsupported data type code '${dtype.code}'.`);
        }
        if (dtype.lanes !== 1) {
            throw new tvm.Error(`Unsupported data type lanes '${dtype.lanes}'.`);
        }
        this.dataType = `${type}${dtype.bits}`;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

tvm.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.isArray(dimensions) ? dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim) : dimensions;
    }

    toString() {
        if (this.dimensions) {
            if (this.dimensions.length === 0) {
                return '';
            }
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

tvm.NDArray = class {

    static loadParams(reader) {
        // https://github.com/apache/tvm/blob/main/src/runtime/file_utils.cc
        reader = new tvm.BinaryReader(reader);
        const header = reader.read(8);
        const signature = [0xB7, 0x9C, 0x04, 0x05, 0x4F, 0x8D, 0xE5, 0xF7];
        if (!header.every((value, index) => value === signature[index])) {
            throw new tvm.Error('Invalid signature.');
        }
        reader.skip(8); // reserved
        const names = reader.strings();
        const values = new Array(reader.uint64().toNumber());
        if (names.length !== values.length) {
            throw new tvm.Error('Invalid parameters.');
        }
        const params = new Map();
        for (let i = 0; i < values.length; i++) {
            const value = new tvm.NDArray(reader);
            params.set(names[i], value);
        }
        return params;
    }

    constructor(reader) {
        // https://github.com/apache/tvm/blob/main/include/tvm/runtime/ndarray.h
        const header = reader.read(8);
        const signature = [0x3F, 0xA1, 0xB4, 0x96, 0xF0, 0x40, 0x5E, 0xDD];
        if (!header.every((value, index) => value === signature[index])) {
            throw new tvm.Error('Invalid signature.');
        }
        reader.skip(8); // reserved
        this.device = {
            deviceType: reader.uint32(),
            deviceId: reader.uint32()
        };
        this.shape = new Array(reader.uint32());
        this.dtype = {
            code: reader.uint8(),
            bits: reader.uint8(),
            lanes: reader.uint16(),
        };
        for (let i = 0; i < this.shape.length; i++) {
            this.shape[i] = reader.uint64();
        }
        const size = reader.uint64().toNumber();
        this.data = reader.read(size);
    }
};

tvm.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    read(length) {
        return this._reader.read(length);
    }

    uint8() {
        return this._reader.byte();
    }

    uint16() {
        return this._reader.uint16();
    }

    uint32() {
        return this._reader.uint32();
    }

    uint64() {
        return this._reader.uint64();
    }

    string() {
        const length = this.uint64().toNumber();
        const buffer = this._reader.read(length);
        return String.fromCharCode.apply(null, new Uint8Array(buffer));
    }

    strings() {
        const list = new Array(this.uint64().toNumber());
        for (let i = 0; i < list.length; i++) {
            list[i] = this.string();
        }
        return list;
    }
};

tvm.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TVM model.';
    }
};

export const ModelFactory = tvm.ModelFactory;
