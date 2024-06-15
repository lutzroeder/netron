
import * as flatbuffers from './flatbuffers.js';
import * as flexbuffers from './flexbuffers.js';
import * as zip from './zip.js';

const tflite = {};

tflite.ModelFactory = class {

    match(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'TFL3') {
            context.type = 'tflite.flatbuffers';
            context.target = reader;
            return;
        }
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'tflite' && reader && reader.identifier === '') {
            const version = reader.uint32_(reader.root, 4, 0);
            if (version === 3) {
                context.type = 'tflite.flatbuffers';
                context.target = reader;
                return;
            }
        }
        const obj = context.peek('json');
        if (obj && obj.subgraphs && obj.operator_codes) {
            context.type = 'tflite.flatbuffers.json';
            context.target = obj;
        }
    }

    async open(context) {
        tflite.schema = await context.require('./tflite-schema');
        tflite.schema = tflite.schema.tflite;
        let model = null;
        const attachments = new Map();
        switch (context.type) {
            case 'tflite.flatbuffers.json': {
                try {
                    const reader = context.read('flatbuffers.text');
                    model = tflite.schema.Model.createText(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tflite.Error(`File text format is not tflite.Model (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            case 'tflite.flatbuffers': {
                try {
                    const reader = context.target;
                    model = tflite.schema.Model.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tflite.Error(`File format is not tflite.Model (${message.replace(/\.$/, '')}).`);
                }
                try {
                    const stream = context.stream;
                    const archive = zip.Archive.open(stream);
                    if (archive) {
                        for (const [name, value] of archive.entries) {
                            attachments.set(name, value);
                        }
                    }
                } catch {
                    // continue regardless of error
                }
                break;
            }
            default: {
                throw new tflite.Error(`Unsupported TensorFlow Lite format '${context.type}'.`);
            }
        }
        const stream = context.stream;
        const metadata = await context.metadata('tflite-metadata.json');
        return new tflite.Model(metadata, model, stream);
    }
};

tflite.Model = class {

    constructor(metadata, model, stream) {
        this.graphs = [];
        this.format = 'TensorFlow Lite';
        this.format = `${this.format} v${model.version}`;
        this.description = model.description || '';
        this.metadata = [];
        const builtinOperators = new Map();
        const upperCase = new Set(['2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM']);
        for (const key of Object.keys(tflite.schema.BuiltinOperator)) {
            const value = key === 'BATCH_MATMUL' ? 'BATCH_MAT_MUL' : key;
            const name = value.split('_').map((s) => (s.length < 1 || upperCase.has(s)) ? s : s[0] + s.substring(1).toLowerCase()).join('');
            const index = tflite.schema.BuiltinOperator[key];
            builtinOperators.set(index, name);
        }
        const operators = model.operator_codes.map((operator) => {
            const code = Math.max(operator.deprecated_builtin_code, operator.builtin_code || 0);
            const value = {};
            if (code === tflite.schema.BuiltinOperator.CUSTOM) {
                value.name = operator.custom_code ? operator.custom_code : 'Custom';
                value.version = operator.version;
                value.custom = true;
            } else {
                value.name = builtinOperators.has(code) ? builtinOperators.get(code) : code.toString();
                value.version = operator.version;
                value.custom = false;
            }
            return value;
        });
        let modelMetadata = null;
        for (const metadata of model.metadata) {
            const buffer = model.buffers[metadata.buffer];
            let data = null;
            const position = stream.position;
            if (buffer && buffer.data && buffer.data.length > 0) {
                data = buffer.data;
            } else if (buffer && buffer.offset !== 0n && buffer.size !== 0n) {
                const offset = buffer.offset.toNumber();
                const size = buffer.size.toNumber();
                stream.seek(offset);
                data = stream.read(size);
            }
            stream.seek(position);
            if (data) {
                switch (metadata.name) {
                    case 'min_runtime_version': {
                        const decoder = new TextDecoder();
                        this.runtime = decoder.decode(data);
                        break;
                    }
                    case 'TFLITE_METADATA': {
                        const reader = flatbuffers.BinaryReader.open(data);
                        if (!reader || !tflite.schema.ModelMetadata.identifier(reader)) {
                            throw new tflite.Error('Invalid TensorFlow Lite metadata.');
                        }
                        modelMetadata = tflite.schema.ModelMetadata.create(reader);
                        if (modelMetadata.name) {
                            this.name = modelMetadata.name;
                        }
                        if (modelMetadata.version) {
                            this.version = modelMetadata.version;
                        }
                        if (modelMetadata.description) {
                            this.description = this.description ? [this.description, modelMetadata.description].join(' ') : modelMetadata.description;
                        }
                        if (modelMetadata.author) {
                            this.metadata.push(new tflite.Argument('author', modelMetadata.author));
                        }
                        if (modelMetadata.license) {
                            this.metadata.push(new tflite.Argument('license', modelMetadata.license));
                        }
                        break;
                    }
                    default: {
                        const value = data.length < 256 && data.every((c) => c >= 32 && c < 128) ? String.fromCharCode.apply(null, data) : '?';
                        const argument = new tflite.Argument(metadata.name, value);
                        this.metadata.push(argument);
                        break;
                    }
                }
            }
        }
        const subgraphs = model.subgraphs;
        const subgraphsMetadata = modelMetadata ? modelMetadata.subgraph_metadata : null;
        for (let i = 0; i < subgraphs.length; i++) {
            const subgraph = subgraphs[i];
            const name = subgraphs.length > 1 ? i.toString() : '';
            const subgraphMetadata = subgraphsMetadata && i < subgraphsMetadata.length ? subgraphsMetadata[i] : null;
            const signatures = model.signature_defs.filter((signature) => signature.subgraph_index === i);
            const graph = new tflite.Graph(metadata, subgraph, signatures, subgraphMetadata, name, operators, model, stream);
            this.graphs.push(graph);
        }
    }
};

tflite.Graph = class {

    constructor(metadata, subgraph, signatures, subgraphMetadata, name, operators, model, stream) {
        this.name = subgraph.name || name;
        if (subgraph.operators.length === 0 && subgraph.tensors.length > 0 && operators.length === 0) {
            operators.push({ name: 'Weights', custom: true });
            const layers = new Map();
            for (let i = 0; i < subgraph.tensors.length; i++) {
                const tensor = subgraph.tensors[i];
                const parts = tensor.name.split('.');
                parts.pop();
                const key = parts.join('.');
                if (!layers.has(key)) {
                    const operator = { opcode_index: 0, inputs: [], outputs: [] };
                    layers.set(key, operator);
                    subgraph.operators.push(operator);
                }
                const operator = layers.get(key);
                operator.inputs.push(i);
            }
        }
        const tensors = new Map();
        tensors.map = (index, metadata) => {
            if (index === -1) {
                return null;
            }
            if (!tensors.has(index)) {
                let tensor = { name: '' };
                let initializer = null;
                let description = '';
                let denotation = '';
                if (index < subgraph.tensors.length) {
                    tensor = subgraph.tensors[index];
                    const buffer = model.buffers[tensor.buffer];
                    const is_variable = tensor.is_variable;
                    const variable = is_variable || (buffer && buffer.data && buffer.data.length > 0) || (buffer && buffer.offset !== 0n && buffer.size !== 0n);
                    initializer = variable ? new tflite.Tensor(index, tensor, buffer, stream, is_variable) : null;
                }
                if (metadata) {
                    description = metadata.description;
                    const content = metadata.content;
                    if (content) {
                        const contentProperties = content.content_properties;
                        if (contentProperties instanceof tflite.schema.FeatureProperties) {
                            denotation = 'Feature';
                        } else if (contentProperties instanceof tflite.schema.ImageProperties) {
                            denotation = 'Image';
                            switch (contentProperties.color_space) {
                                case 0: denotation += '(Unknown)'; break;
                                case 1: denotation += '(RGB)'; break;
                                case 2: denotation += '(Grayscale)'; break;
                                default: throw tflite.Error(`Unsupported image color space '${contentProperties.color_space}'.`);
                            }
                        } else if (contentProperties instanceof tflite.schema.BoundingBoxProperties) {
                            denotation = 'BoundingBox';
                        } else if (contentProperties instanceof tflite.schema.AudioProperties) {
                            denotation = `Audio(${contentProperties.sample_rate},${contentProperties.channels})`;
                        }
                    }
                }
                const value = new tflite.Value(index, tensor, initializer, description, denotation);
                tensors.set(index, value);
            }
            return tensors.get(index);
        };
        this.inputs = Array.from(subgraph.inputs).map((tensor_index, index) => {
            const metadata = subgraphMetadata && index < subgraphMetadata.input_tensor_metadata.length ? subgraphMetadata.input_tensor_metadata[index] : null;
            const value = tensors.map(tensor_index, metadata);
            const values = value ? [value] : [];
            const name = value ? value.name.split('\n')[0] : '?';
            return new tflite.Argument(name, values);
        });
        this.outputs = Array.from(subgraph.outputs).map((tensor_index, index) => {
            const metadata = subgraphMetadata && index < subgraphMetadata.output_tensor_metadata.length ? subgraphMetadata.output_tensor_metadata[index] : null;
            const value = tensors.map(tensor_index, metadata);
            const values = value ? [value] : [];
            const name = value ? value.name.split('\n')[0] : '?';
            return new tflite.Argument(name, values);
        });
        this.signatures = signatures.map((signature) => {
            return new tflite.Signature(signature, tensors);
        });
        this.nodes = Array.from(subgraph.operators).map((operator, index) => {
            const opcode_index = operator.opcode_index;
            const opcode = opcode_index < operators.length ? operators[opcode_index] : { name: `(${opcode_index})` };
            return new tflite.Node(metadata, operator, opcode, index.toString(), tensors);
        });
    }
};

tflite.Signature = class {

    constructor(signature, tensors) {
        this.name = signature.signature_key;
        this.inputs = signature.inputs.map((input) => {
            const value = tensors.map(input.tensor_index);
            const values = value ? [value] : [];
            return new tflite.Argument(input.name, values);
        });
        this.outputs = signature.outputs.map((output) => {
            const value = tensors.map(output.tensor_index);
            const values = value ? [value] : [];
            return new tflite.Argument(output.name, values);
        });
    }
};

tflite.Node = class {

    constructor(metadata, node, type, identifier, tensors) {
        this.name = '';
        this.identifier = identifier;
        this.type = type.custom ? { name: type.name } : metadata.type(type.name);
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (node) {
            const attributes = [];
            const inputs = Array.from(node.inputs || new Int32Array(0));
            for (let i = 0; i < inputs.length;) {
                let count = 1;
                let name = null;
                let visible = true;
                const values = [];
                if (this.type && this.type.inputs && i < this.type.inputs.length) {
                    const input = this.type.inputs[i];
                    name = input.name;
                    if (input.list) {
                        count = inputs.length - i;
                    }
                    if (input.visible === false) {
                        visible = false;
                    }
                }
                for (const index of inputs.slice(i, i + count)) {
                    const value = tensors.map(index);
                    if (value) {
                        values.push(value);
                    }
                }
                name = name ? name : (i + 1).toString();
                i += count;
                const argument = new tflite.Argument(name, values, null, visible);
                this.inputs.push(argument);
            }
            const outputs = Array.from(node.outputs || new Int32Array(0));
            for (let i = 0; i < outputs.length; i++) {
                const index = outputs[i];
                const value = tensors.map(index);
                const values = value ? [value] : [];
                let name = (i + 1).toString();
                if (this.type && this.type.outputs && i < this.type.outputs.length) {
                    const output = this.type.outputs[i];
                    if (output && output.name) {
                        name = output.name;
                    }
                }
                const argument = new tflite.Argument(name, values);
                this.outputs.push(argument);
            }
            if (type.custom && node.custom_options && node.custom_options.length > 0) {
                let decoded = false;
                if (node.custom_options_format === tflite.schema.CustomOptionsFormat.FLEXBUFFERS) {
                    try {
                        const reader = flexbuffers.BinaryReader.open(node.custom_options);
                        if (reader) {
                            const custom_options = reader.read();
                            if (Array.isArray(custom_options)) {
                                attributes.push([null, 'custom_options', custom_options]);
                                decoded = true;
                            } else if (custom_options) {
                                for (const [key, value] of Object.entries(custom_options)) {
                                    const schema = metadata.attribute(type.name, key);
                                    attributes.push([schema, key, value]);
                                }
                                decoded = true;
                            }
                        }
                    } catch {
                        // continue regardless of error
                    }
                }
                if (!decoded) {
                    const schema = metadata.attribute(type.name, 'custom');
                    attributes.push([schema, 'custom', Array.from(node.custom_options)]);
                }
            }
            const options = node.builtin_options;
            if (options) {
                for (const [name, value] of Object.entries(options)) {
                    if (name === 'fused_activation_function' && value) {
                        if (value < 1 || value > 5) {
                            throw new tflite.Error(`Unsupported activation funtion index '${value}'.`);
                        }
                        const list = ['Unknown', 'Relu', 'ReluN1To1', 'Relu6', 'Tanh', 'SignBit'];
                        const type = list[value];
                        const node = new tflite.Node(metadata, null, { name: type }, null, []);
                        this.chain = [node];
                    }
                    const schema = metadata.attribute(type.name, name);
                    attributes.push([schema, name, value]);
                }
            }
            this.attributes = attributes.map(([metadata, name, value]) => {
                const type = metadata && metadata.type ? metadata.type : null;
                value = ArrayBuffer.isView(value) ? Array.from(value) : value;
                let visible = true;
                if (name === 'fused_activation_function') {
                    visible = false;
                }
                if (type) {
                    value = tflite.Utility.enum(type, value);
                }
                if (metadata) {
                    if (metadata.visible === false) {
                        visible = false;
                    } else if (metadata.default !== undefined) {
                        if (typeof value === 'function') {
                            value = value();
                        }
                        if (value === metadata.default) {
                            visible = false;
                        }
                    }
                }
                return new tflite.Argument(name, value, type, visible);
            });
        }
    }
};

tflite.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

tflite.Value = class {

    constructor(index, tensor, initializer, description, denotation) {
        const name = tensor.name || '';
        this.name = `${name}\n${index}`;
        this.identifier = index.toString();
        this.type = tensor.type !== undefined && tensor.shape !== undefined ? new tflite.TensorType(tensor, denotation) : null;
        this.initializer = initializer;
        this.description = description;
        const quantization = tensor.quantization;
        if (quantization && (quantization.scale.length > 0 || quantization.zero_point.length > 0 || quantization.min.length > 0 || quantization.max.length)) {
            this.quantization = {
                type: 'linear',
                dimension: quantization.quantized_dimension,
                scale: quantization.scale,
                offset: quantization.zero_point,
                min: quantization.min,
                max: quantization.max
            };
        }
    }
};

tflite.Tensor = class {

    constructor(index, tensor, buffer, stream, is_variable) {
        this.identifier = index.toString();
        this.name = tensor.name;
        this.type = new tflite.TensorType(tensor);
        this.category = is_variable ? 'Variable' : '';
        this.encoding = this.type.dataType === 'string' ? '|' : '<';
        if (buffer && buffer.data && buffer.data.length > 0) {
            this._data = buffer.data.slice(0);
        } else if (buffer && buffer.offset !== 0n && buffer.size !== 0n) {
            const offset = buffer.offset.toNumber();
            const size = buffer.size.toNumber();
            stream.seek(offset);
            this._data = stream.stream(size);
        } else {
            this._data = null;
        }
    }

    get values() {
        switch (this.type.dataType) {
            case 'string': {
                let offset = 0;
                const data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                const count = data.getInt32(0, true);
                offset += 4;
                const offsetTable = [];
                for (let j = 0; j < count; j++) {
                    offsetTable.push(data.getInt32(offset, true));
                    offset += 4;
                }
                offsetTable.push(this._data.length);
                const stringTable = [];
                const utf8Decoder = new TextDecoder('utf-8');
                for (let k = 0; k < count; k++) {
                    const textArray = this._data.subarray(offsetTable[k], offsetTable[k + 1]);
                    stringTable.push(utf8Decoder.decode(textArray));
                }
                return stringTable;
            }
            default: {
                if (this._data instanceof Uint8Array) {
                    return this._data;
                }
                if (this._data && this._data.peek) {
                    return this._data.peek();
                }
                return null;
            }
        }
    }
};

tflite.TensorType = class {

    constructor(tensor, denotation) {
        const shape = tensor.shape_signature && tensor.shape_signature.length > 0 ? tensor.shape_signature : tensor.shape;
        this.dataType = tflite.Utility.dataType(tensor.type);
        this.shape = new tflite.TensorShape(Array.from(shape || []));
        this.denotation = denotation;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

tflite.TensorShape = class {

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

tflite.Utility = class {

    static dataType(type) {
        if (!tflite.Utility._tensorTypes) {
            tflite.Utility._tensorTypes = new Map(Object.entries(tflite.schema.TensorType).map(([key, value]) => [value, key.toLowerCase()]));
            tflite.Utility._tensorTypes.set(6, 'boolean');
        }
        return tflite.Utility._tensorTypes.has(type) ? tflite.Utility._tensorTypes.get(type) : '?';
    }

    static enum(name, value) {
        const type = name && tflite.schema ? tflite.schema[name] : undefined;
        if (type) {
            tflite.Utility._enums = tflite.Utility._enums || new Map();
            if (!tflite.Utility._enums.has(name)) {
                const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
                tflite.Utility._enums.set(name, entries);
            }
            const map = tflite.Utility._enums.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }
};

tflite.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow Lite model.';
    }
};

export const ModelFactory = tflite.ModelFactory;
