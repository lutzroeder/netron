
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
        tosa.schema = await context.require('./tosa-schema');
        tosa.schema = tosa.schema.tosa;
        let major = -1;
        let minor = -1;
        let patch = -1;
        switch (context.type) {
            case 'tosa.flatbuffers': {
                const reader = context.value;
                try {
                    const root = reader.root;
                    const vtable = root - reader.int32(root);
                    const vtableSize = reader.int16(vtable);
                    if (vtableSize > 4) {
                        const versionFieldOffset = reader.int16(vtable + 4);
                        if (versionFieldOffset) {
                            const ref = root + versionFieldOffset;
                            const position = ref + reader.int32(ref);
                            const vvtable = position - reader.int32(position);
                            const vvtableSize = reader.int16(vvtable);
                            const field = (index) => {
                                const offset = 4 + index * 2;
                                if (offset < vvtableSize) {
                                    const value = reader.int16(vvtable + offset);
                                    if (value) {
                                        return reader.int32(position + value);
                                    }
                                }
                                return -1;
                            };
                            major = field(0);
                            minor = field(1);
                            patch = field(2);
                        }
                    }
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tosa.Error(`File format is not tosa.TosaGraph (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            case 'tosa.flatbuffers.json': {
                const obj = context.value;
                if (obj.version) {
                    const v = obj.version;
                    major = v._major === undefined ? -1 : v._major;
                    minor = v._minor === undefined ? -1 : v._minor;
                    patch = v._patch === undefined ? -1 : v._patch;
                }
                break;
            }
            default: {
                throw new tosa.Error(`Unsupported TOSA format '${context.type}'.`);
            }
        }
        const file_version = `${major}.${minor}.${patch}`;
        let schema_version = '';
        if (major === 0 && minor >= 80) {
            schema_version = '0.80';
        } else if (major === 1) {
            schema_version = '1.0';
        }
        const schema = { '0.80': tosa.schema.v0, '1.0': tosa.schema.v1 }[schema_version];
        if (!schema) {
            throw new tosa.Error(`Unsupported TOSA version '${file_version}'.`);
        }
        let model = null;
        try {
            switch (context.type) {
                case 'tosa.flatbuffers': {
                    model = schema.TosaGraph.create(context.value);
                    break;
                }
                case 'tosa.flatbuffers.json': {
                    const reader = await context.read('flatbuffers.text');
                    model = schema.TosaGraph.createText(reader);
                    break;
                }
                default: {
                    break;
                }
            }
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new tosa.Error(`File format is not tosa.TosaGraph (${message.replace(/\.$/, '')}).`);
        }
        const data = await context.request('tosa-metadata.json', 'utf-8', null);
        const metadata = new tosa.Metadata(data, schema_version);
        return new tosa.Model(new tosa.Context(schema, metadata), model, file_version);
    }
};

tosa.Model = class {

    constructor(context, model, version) {
        this.format = `TOSA v${version}`;
        this.modules = [];
        for (const region of model.regions) {
            for (const block of region.blocks) {
                this.modules.push(new tosa.Graph(context, block, region.name));
            }
        }
    }
};

tosa.Graph = class {

    constructor(context, block, region) {
        this.name = region ? `${region}/${block.name}` : block.name || '';
        const tensors = new Map();
        for (const tensor of block.tensors) {
            const type = new tosa.TensorType(context, tensor.type, tensor.shape);
            const data = tensor.data && tensor.data.length > 0 ? tensor.data : null;
            const initializer = data ? new tosa.Tensor(tensor.name, type, data) : null;
            if (initializer && tensor.variable) {
                initializer.category = 'Variable';
            }
            const value = new tosa.Value(tensor.name, type, initializer);
            if (tensor.variable) {
                value.visible = tensor.variable;
            }
            tensors.set(tensor.name, value);
        }
        const value = (name) => {
            if (!tensors.has(name)) {
                tensors.set(name, new tosa.Value(name, null, null));
            }
            return tensors.get(name);
        };
        this.inputs = (block.inputs || []).map((name) => new tosa.Argument(name, [value(name)]));
        this.outputs = (block.outputs || []).map((name) => new tosa.Argument(name, [value(name)]));
        this.nodes = (block.operators || []).map((operator, index) => new tosa.Node(context, operator, index, value)).filter((node) => !node.type.name.startsWith('CONST'));
    }
};

tosa.Node = class {

    constructor(context, operator, index, value) {
        const enumTypes = {
            acc_type: 'DType',
            accum_dtype: 'DType',
            type: 'DType',
            mode: 'ResizeMode',
            nan_mode: 'NanPropagationMode',
            rounding_mode: 'RoundingMode'
        };
        const op = operator.op;
        const opName = context.enum('Op', op);
        this.type = context.type(opName);
        this.name = '';
        this.identifier = index.toString();
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const inputs = operator.inputs || [];
        for (let i = 0; i < inputs.length; i++) {
            const name = this.type && this.type.inputs && i < this.type.inputs.length ? this.type.inputs[i].name : `input${i}`;
            const values = [value(inputs[i])];
            this.inputs.push(new tosa.Argument(name, values));
        }
        const outputs = operator.outputs || [];
        for (let i = 0; i < outputs.length; i++) {
            const name = this.type && this.type.outputs && i < this.type.outputs.length ? this.type.outputs[i].name : `output${i}`;
            const values = [value(outputs[i])];
            this.outputs.push(new tosa.Argument(name, values));
        }
        const options = operator.attribute;
        if (options) {
            for (const [name, obj] of Object.entries(options)) {
                if (name === 'type') {
                    continue;
                }
                const schema = context.attribute(opName, name);
                let attrValue = obj;
                attrValue = ArrayBuffer.isView(attrValue) ? Array.from(attrValue) : attrValue;
                let visible = true;
                let type = null;
                if (schema) {
                    if (schema.visible === false) {
                        visible = false;
                    } else if (schema.default !== undefined) {
                        if (attrValue === schema.default) {
                            visible = false;
                        }
                    }
                }
                if (typeof attrValue === 'bigint') {
                    attrValue = Number(attrValue);
                }
                if (typeof attrValue === 'number' && Number.isInteger(attrValue)) {
                    const enumType = enumTypes[name] || null;
                    if (enumType) {
                        const enumValue = context.enum(enumType, attrValue);
                        if (enumValue !== attrValue.toString()) {
                            type = enumType;
                            attrValue = enumValue;
                        }
                    }
                }
                this.attributes.push(new tosa.Argument(name, attrValue, type, visible));
            }
        }
    }
};

tosa.Argument = class {

    constructor(name, value, type = null, visible = true) {
        this.name = name;
        this.value = value;
        this.type = type;
        this.visible = visible;
    }
};

tosa.Value = class {

    constructor(name, type, initializer) {
        this.name = name;
        this.type = type;
        this.initializer = initializer;
    }
};

tosa.Tensor = class {

    constructor(name, type, data) {
        this.name = name;
        this.type = type;
        this.encoding = '<';
        this._data = data;
    }

    get values() {
        if (this._data instanceof Uint8Array) {
            return this._data;
        }
        if (this._data && this._data.peek) {
            return this._data.peek();
        }
        return null;
    }
};

tosa.TensorType = class {

    constructor(context, dataType, shape) {
        this.dataType = context.dataType(dataType);
        this.shape = new tosa.TensorShape(shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

tosa.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions ? Array.from(dimensions) : [];
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((d) => d.toString()).join(',')}]`;
        }
        return '';
    }
};

tosa.Context = class {

    constructor(schema, metadata) {
        this._schema = schema;
        this._metadata = metadata;
        this._enums = new Map();
        const mapping = {
            BOOL: 'boolean',
            UINT8: 'uint8', UINT16: 'uint16',
            INT4: 'int4', INT8: 'int8', INT16: 'int16', INT32: 'int32', INT48: 'int48',
            FP16: 'float16', BF16: 'bfloat16', FP32: 'float32',
            FP8E4M3: 'float8e4m3', FP8E5M2: 'float8e5m2',
            SHAPE: 'int64'
        };
        this._dataTypes = new Map();
        for (const [key, value] of Object.entries(schema.DType)) {
            this._dataTypes.set(value, mapping[key] || key.toLowerCase());
        }
    }

    type(name) {
        return this._metadata.type(name);
    }

    attribute(type, name) {
        return this._metadata.attribute(type, name);
    }

    dataType(type) {
        return this._dataTypes.get(type) || '?';
    }

    enum(name, value) {
        const type = name && this._schema ? this._schema[name] : undefined;
        if (type) {
            if (!this._enums.has(name)) {
                this._enums.set(name, new Map(Object.entries(type).map(([key, val]) => [val, key])));
            }
            const map = this._enums.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value.toString();
    }
};

tosa.Metadata = class {

    constructor(data, version) {
        this._types = new Map();
        this._attributes = new Map();
        if (data) {
            const types = JSON.parse(data);
            for (const type of types) {
                if (type.version === version) {
                    this._types.set(type.name, type);
                }
            }
        }
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name });
        }
        return this._types.get(name);
    }

    attribute(type, name) {
        const key = `${type}:${name}`;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this._types.get(type);
            if (metadata && metadata.attributes) {
                for (const attribute of metadata.attributes) {
                    this._attributes.set(`${type}:${attribute.name}`, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }
};

tosa.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TOSA model.';
    }
};

export const ModelFactory = tosa.ModelFactory;
