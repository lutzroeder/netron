import * as flatbuffers from './flatbuffers.js';

const espdl = {};

espdl.ModelFactory = class {

    async match(context) {
        const identifier = context.identifier;
        const extension = identifier.lastIndexOf('.') > 0 ? identifier.split('.').pop().toLowerCase() : '';
        if (extension === 'espdl') {
            const stream = context.stream;
            if (stream && stream.length >= 16) {
                const buffer = stream.peek(16);
                const header = String.fromCharCode(...buffer.slice(0, 4));
                if (header === 'EDL2') {
                    return context.set('espdl.binary', null);
                }
            }
        }
        return null;
    }

    async open(context) {
        espdl.schema = await context.require('./espdl-schema');
        espdl.schema = espdl.schema.espdl;
        const stream = context.stream;
        const reader = flatbuffers.BinaryReader.open(stream, 16);
        let model = null;
        try {
            model = espdl.schema.Model.create(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new espdl.Error(`File format is not espdl.Model (${message.replace(/\.$/, '')}).`);
        }
        const metadata = await espdl.Metadata.open(context);
        return new espdl.Model(metadata, model, stream);
    }
};

espdl.Model = class {

    constructor(metadata, model, stream) {
        this.format = `ESP-DL v${model.ir_version}`;
        this.description = model.doc_string || '';
        this.modules = [];
        this.metadata = [];
        if (model.metadata_props) {
            for (const prop of model.metadata_props) {
                this.metadata.push(new espdl.Argument(prop.key, prop.value));
            }
        }
        if (model.graph) {
            const graph = new espdl.Graph(metadata, model.graph, model, stream);
            this.modules.push(graph);
        }
    }
};

espdl.Graph = class {

    constructor(metadata, graph) {
        this.name = graph.name || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        this.signatures = [];
        const context = new espdl.Context(graph);
        if (graph.node) {
            for (let i = 0; i < graph.node.length; i++) {
                const node = graph.node[i];
                const nodeObj = new espdl.Node(metadata, context, node, i.toString());
                this.nodes.push(nodeObj);
            }
        }
        if (graph.input) {
            for (let i = 0; i < graph.input.length; i++) {
                const valueInfo = graph.input[i];
                const tensor = context.initializer(valueInfo.name);
                if (!tensor) {
                    const value = context.value(valueInfo.name);
                    const values = value ? [value] : [];
                    const argument = new espdl.Argument(valueInfo.name, values);
                    this.inputs.push(argument);
                }
            }
        }
        if (graph.output) {
            for (let i = 0; i < graph.output.length; i++) {
                const valueInfo = graph.output[i];
                const value = context.value(valueInfo.name);
                const values = value ? [value] : [];
                const argument = new espdl.Argument(valueInfo.name, values);
                this.outputs.push(argument);
            }
        }
    }
};

espdl.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

espdl.Value = class {

    constructor(name, type, initializer) {
        this.name = name;
        this.type = type || null;
        this.initializer = initializer || null;
    }
};

espdl.Node = class {

    constructor(metadata, context, node, identifier) {
        this.name = node.name || '';
        this.identifier = identifier;
        this.type = null;
        if (metadata) {
            this.type = metadata.type('espdl', node.op_type);
        }
        if (!this.type) {
            this.type = { name: node.op_type };
        }
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (node.input) {
            for (let i = 0; i < node.input.length;) {
                const inputMeta = this.type && Array.isArray(this.type.inputs) && i < this.type.inputs.length ? this.type.inputs[i] : { name: i.toString() };
                const count = inputMeta.list ? node.input.length - i : 1;
                const list = node.input.slice(i, i + count);
                const values = list.map((inputName) => {
                    if (!inputName) {
                        return null;
                    }
                    return context.value(inputName);
                }).filter((v) => v);
                const argument = new espdl.Argument(inputMeta.name, values);
                this.inputs.push(argument);
                i += count;
            }
        }
        if (node.output) {
            for (let i = 0; i < node.output.length;) {
                const outputMeta = this.type && Array.isArray(this.type.outputs) && i < this.type.outputs.length ? this.type.outputs[i] : { name: i.toString() };
                const count = outputMeta.list ? node.output.length - i : 1;
                const list = node.output.slice(i, i + count);
                const values = list.map((outputName) => {
                    if (!outputName) {
                        return null;
                    }
                    return context.value(outputName);
                }).filter((v) => v);
                const argument = new espdl.Argument(outputMeta.name, values);
                this.outputs.push(argument);
                i += count;
            }
        }
        if (node.attribute) {
            for (const attr of node.attribute) {
                const name = attr.name || '';
                let value = null;
                let type = null;
                switch (attr.attr_type) {
                    case espdl.schema.AttributeType.FLOAT:
                        value = attr.f ? attr.f.f : 0;
                        type = 'float32';
                        break;
                    case espdl.schema.AttributeType.INT:
                        value = attr.i ? Number(attr.i.i) : 0;
                        type = 'int64';
                        break;
                    case espdl.schema.AttributeType.STRING:
                        value = attr.s ? new TextDecoder('utf-8').decode(attr.s) : '';
                        type = 'string';
                        break;
                    case espdl.schema.AttributeType.TENSOR:
                        value = attr.t ? new espdl.Tensor(0, attr.t) : null;
                        type = 'tensor';
                        break;
                    case espdl.schema.AttributeType.FLOATS:
                        value = attr.floats ? Array.from(attr.floats) : [];
                        type = 'float32[]';
                        break;
                    case espdl.schema.AttributeType.INTS:
                        value = attr.ints ? Array.from(attr.ints).map((i) => Number(i)) : [];
                        type = 'int64[]';
                        break;
                    case espdl.schema.AttributeType.STRINGS:
                        value = attr.strings ? attr.strings.map((s) => new TextDecoder('utf-8').decode(s)) : [];
                        type = 'string[]';
                        break;
                    default:
                        break;
                }
                const attribute = new espdl.Argument(name, value, type);
                this.attributes.push(attribute);
            }
        }
    }
};

espdl.Tensor = class {

    constructor(index, tensor) {
        this.identifier = index.toString();
        this.name = tensor.name || '';
        this.type = new espdl.TensorType(tensor);
        this.category = '';
        this.encoding = this.type.dataType === 'string' ? '|' : '<';
        this.values = null;
        if (tensor.float_data && tensor.float_data.length > 0) {
            this.values = new Float32Array(tensor.float_data);
        } else if (tensor.int32_data && tensor.int32_data.length > 0) {
            this.values = new Int32Array(tensor.int32_data);
        } else if (tensor.int64_data && tensor.int64_data.length > 0) {
            this.values = new BigInt64Array(tensor.int64_data);
        } else if (tensor.string_data && tensor.string_data.length > 0) {
            this.values = tensor.string_data;
        } else if (tensor.raw_data && tensor.raw_data.length > 0) {
            const length = tensor.raw_data.length * 16;
            const data = new Uint8Array(length);
            for (let i = 0; i < tensor.raw_data.length; i++) {
                data.set(tensor.raw_data[i].bytes, i * 16);
            }
            this.values = data;
        }
    }
};

espdl.TensorType = class {

    constructor(tensor) {
        let dataType = '';
        if (tensor.value_info_type === undefined) {
            dataType = tensor.data_type;
            this.shape = new espdl.TensorShape(tensor.dims ? Array.from(tensor.dims).map((d) => Number(d)) : []);
        } else {
            const value = tensor.value_info_type.value;
            dataType = value ? value.elem_type : undefined;
            let shape = [];
            const dim = value && value.shape && value.shape.dim;
            if (dim && dim.length > 0) {
                shape = dim.map((d) => {
                    if (d && d.value) {
                        if (d.value.dim_type === 1) {
                            return Number(d.value.dim_value);
                        } else if (d.value.dim_type === 2) {
                            return d.value.dim_param || '?';
                        }
                    }
                    return '?';
                });
            }
            this.shape = new espdl.TensorShape(shape);
        }
        switch (dataType) {
            case espdl.schema.TensorDataType.FLOAT: this.dataType = 'float32'; break;
            case espdl.schema.TensorDataType.DOUBLE: this.dataType = 'float64'; break;
            case espdl.schema.TensorDataType.BOOL: this.dataType = 'boolean'; break;
            default: this.dataType = espdl.schema.TensorDataType[dataType] ? espdl.schema.TensorDataType[dataType].toLowerCase() : '?'; break;
        }
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

espdl.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

espdl.Metadata = class {

    static async open(context) {
        if (!espdl.Metadata._metadata) {
            let data = null;
            try {
                data = await context.request('espdl-metadata.json');
            } catch {
                // continue regardless of error
            }
            espdl.Metadata._metadata = new espdl.Metadata(data);
        }
        return espdl.Metadata._metadata;
    }

    constructor(data) {
        this._types = new Map();
        if (data) {
            const types = JSON.parse(data);
            for (const type of types) {
                if (!this._types.has(type.module)) {
                    this._types.set(type.module, new Map());
                }
                const types = this._types.get(type.module);
                if (!types.has(type.name)) {
                    types.set(type.name, []);
                }
                types.get(type.name).push(type);
            }
        }
    }

    type(domain, name) {
        domain = domain || 'espdl';
        let current = null;
        if (this._types.has(domain)) {
            const types = this._types.get(domain);
            if (types.has(name)) {
                for (const type of types.get(name)) {
                    if (!current || type.version > current.version) {
                        current = type;
                    }
                }
            }
        }
        return current;
    }
};

espdl.Context = class {

    constructor(graph) {
        this._initializers = new Map();
        this._values = new Map();
        if (graph.initializer) {
            for (let i = 0; i < graph.initializer.length; i++) {
                const tensor = graph.initializer[i];
                const name = tensor.name || '';
                if (name) {
                    const initializer = new espdl.Tensor(i, tensor);
                    this._initializers.set(name, initializer);
                    this._values.set(name, new espdl.Value(name, initializer.type, initializer));
                }
            }
        }
        if (graph.input) {
            for (const valueInfo of graph.input) {
                const name = valueInfo.name || '';
                if (name && !this._values.has(name)) {
                    const type = valueInfo.value_info_type ? new espdl.TensorType(valueInfo) : null;
                    this._values.set(name, new espdl.Value(name, type, null));
                }
            }
        }
        if (graph.output) {
            for (const valueInfo of graph.output) {
                const name = valueInfo.name || '';
                if (name && !this._values.has(name)) {
                    const type = valueInfo.value_info_type ? new espdl.TensorType(valueInfo) : null;
                    this._values.set(name, new espdl.Value(name, type, null));
                }
            }
        }
        if (graph.value_info) {
            for (const valueInfo of graph.value_info) {
                const name = valueInfo.name || '';
                if (name && !this._values.has(name)) {
                    const type = valueInfo.value_info_type ? new espdl.TensorType(valueInfo) : null;
                    this._values.set(name, new espdl.Value(name, type, null));
                }
            }
        }
    }

    value(name) {
        if (!this._values.has(name)) {
            this._values.set(name, new espdl.Value(name, null, null));
        }
        return this._values.get(name);
    }

    initializer(name) {
        return this._initializers.get(name) || null;
    }
};

espdl.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ESP-DL model.';
    }
};

export const ModelFactory = espdl.ModelFactory;