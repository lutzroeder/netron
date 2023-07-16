
var tflite = {};
var flatbuffers = require('./flatbuffers');
var flexbuffers = require('./flexbuffers');
var zip = require('./zip');

tflite.ModelFactory = class {

    match(context) {
        const tags = context.tags('flatbuffers');
        if (tags.get('file_identifier') === 'TFL3') {
            return 'tflite.flatbuffers';
        }
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        if (extension === 'tflite' && stream.length >= 8) {
            const buffer = stream.peek(Math.min(32, stream.length));
            const reader = flatbuffers.BinaryReader.open(buffer);
            if (reader.root === 0x00000018) {
                const version = reader.uint32_(reader.root, 4, 0);
                if (version === 3) {
                    return 'tflite.flatbuffers';
                }
            }
        }
        const obj = context.open('json');
        if (obj && obj.subgraphs && obj.operator_codes) {
            return 'tflite.flatbuffers.json';
        }
        return undefined;
    }

    async open(context, target) {
        await context.require('./tflite-schema');
        tflite.schema = flatbuffers.get('tflite').tflite;
        let model = null;
        const attachments = new Map();
        switch (target) {
            case 'tflite.flatbuffers.json': {
                try {
                    const obj = context.open('json');
                    const reader = new flatbuffers.TextReader(obj);
                    model = tflite.schema.Model.createText(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tflite.Error('File text format is not tflite.Model (' + message.replace(/\.$/, '') + ').');
                }
                break;
            }
            case 'tflite.flatbuffers': {
                const stream = context.stream;
                try {
                    const reader = flatbuffers.BinaryReader.open(stream);
                    model = tflite.schema.Model.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tflite.Error('File format is not tflite.Model (' + message.replace(/\.$/, '') + ').');
                }
                try {
                    const archive = zip.Archive.open(stream);
                    if (archive) {
                        for (const entry of archive.entries) {
                            attachments.set(entry[0], entry[1]);
                        }
                    }
                } catch (error) {
                    // continue regardless of error
                }
                break;
            }
            default: {
                throw new tflite.Error("Unsupported TensorFlow Lite format '" + target + "'.");
            }
        }
        const metadata = await context.metadata('tflite-metadata.json');
        return new tflite.Model(metadata, model);
    }
};

tflite.Model = class {

    constructor(metadata, model) {
        this._graphs = [];
        this._format = 'TensorFlow Lite';
        this._format = this._format + ' v' + model.version.toString();
        this._description = model.description || '';
        this._metadata = [];
        const builtinOperators = new Map();
        const upperCase = new Set([ '2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM' ]);
        for (const key of Object.keys(tflite.schema.BuiltinOperator)) {
            const value = key === 'BATCH_MATMUL' ? 'BATCH_MAT_MUL' : key;
            const name = value.split('_').map((s) => (s.length < 1 || upperCase.has(s)) ? s : s[0] + s.substring(1).toLowerCase()).join('');
            const index = tflite.schema.BuiltinOperator[key];
            builtinOperators.set(index, name);
        }
        const operators = model.operator_codes.map((operator) => {
            const code = Math.max(operator.deprecated_builtin_code, operator.builtin_code || 0);
            const version = operator.version;
            const custom = code === tflite.schema.BuiltinOperator.CUSTOM;
            const name = custom ? operator.custom_code ? operator.custom_code : 'Custom' : builtinOperators.has(code) ? builtinOperators.get(code) : code.toString();
            return custom ? { name: name, version: version, custom: true } : { name: name, version: version };
        });
        let modelMetadata = null;
        for (const metadata of model.metadata) {
            const buffer = model.buffers[metadata.buffer];
            if (buffer) {
                switch (metadata.name) {
                    case 'min_runtime_version': {
                        const data = buffer.data || new Uint8Array(0);
                        this._runtime = new TextDecoder().decode(data);
                        break;
                    }
                    case 'TFLITE_METADATA': {
                        const data = buffer.data || new Uint8Array(0);
                        const reader = flatbuffers.BinaryReader.open(data);
                        if (tflite.schema.ModelMetadata.identifier(reader)) {
                            modelMetadata = tflite.schema.ModelMetadata.create(reader);
                            if (modelMetadata.name) {
                                this._name = modelMetadata.name;
                            }
                            if (modelMetadata.version) {
                                this._version = modelMetadata.version;
                            }
                            if (modelMetadata.description) {
                                this._description = this._description ? [ this._description, modelMetadata.description].join(' ') : modelMetadata.description;
                            }
                            if (modelMetadata.author) {
                                this._metadata.push({ name: 'author', value: modelMetadata.author });
                            }
                            if (modelMetadata.license) {
                                this._metadata.push({ name: 'license', value: modelMetadata.license });
                            }
                        }
                        break;
                    }
                    default: {
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
            this._graphs.push(new tflite.Graph(metadata, subgraph, subgraphMetadata, name, operators, model));
        }
    }

    get format() {
        return this._format;
    }

    get runtime() {
        return this._runtime;
    }

    get name() {
        return this._name;
    }

    get version() {
        return this._version;
    }

    get description() {
        return this._description;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

tflite.Graph = class {

    constructor(metadata, subgraph, subgraphMetadata, name, operators, model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._name = subgraph.name || name;
        const tensors = new Map();
        const args = (index) => {
            if (index === -1) {
                return null;
            }
            if (!tensors.has(index)) {
                if (index < subgraph.tensors.length) {
                    const tensor = subgraph.tensors[index];
                    const buffer = model.buffers[tensor.buffer];
                    const is_variable = tensor.is_variable;
                    const data = buffer ? buffer.data : null;
                    const initializer = (data && data.length > 0) || is_variable ? new tflite.Tensor(index, tensor, buffer, is_variable) : null;
                    tensors.set(index, new tflite.Value(index, tensor, initializer));
                } else {
                    tensors.set(index, new tflite.Value(index, { name: '' }, null));
                }
            }
            return tensors.get(index);
        };
        for (let i = 0; i < subgraph.operators.length; i++) {
            const node = subgraph.operators[i];
            const index = node.opcode_index;
            const operator = index < operators.length ? operators[index] : { name: '(' + index.toString() + ')' };
            this._nodes.push(new tflite.Node(metadata, node, operator, i.toString(), args));
        }
        const applyTensorMetadata = (argument, tensorMetadata) => {
            if (tensorMetadata) {
                const description = tensorMetadata.description;
                if (description) {
                    argument.description = description;
                }
                const content = tensorMetadata.content;
                if (argument.type && content) {
                    let denotation = null;
                    const contentProperties = content.content_properties;
                    if (contentProperties instanceof tflite.schema.FeatureProperties) {
                        denotation = 'Feature';
                    } else if (contentProperties instanceof tflite.schema.ImageProperties) {
                        denotation = 'Image';
                        switch (contentProperties.color_space) {
                            case 0: denotation += '(Unknown)'; break;
                            case 1: denotation += '(RGB)'; break;
                            case 2: denotation += '(Grayscale)'; break;
                            default: throw tflite.Error("Unsupported image color space '" + contentProperties.color_space + "'.");
                        }
                    } else if (contentProperties instanceof tflite.schema.BoundingBoxProperties) {
                        denotation = 'BoundingBox';
                    } else if (contentProperties instanceof tflite.schema.AudioProperties) {
                        denotation = 'Audio(' + contentProperties.sample_rate.toString() + ',' + contentProperties.channels.toString() + ')';
                    }
                    if (denotation) {
                        argument.type.denotation = denotation;
                    }
                }
            }
        };
        const inputs = subgraph.inputs;
        for (let i = 0; i < inputs.length; i++) {
            const input = inputs[i];
            const value = args(input);
            if (subgraphMetadata && i < subgraphMetadata.input_tensor_metadata.length) {
                applyTensorMetadata(value, subgraphMetadata.input_tensor_metadata[i]);
            }
            this._inputs.push(new tflite.Argument(value ? value.name : '?', true, value ? [ value ] : []));
        }
        const outputs = subgraph.outputs;
        for (let i = 0; i < outputs.length; i++) {
            const output = outputs[i];
            const value = args(output);
            if (subgraphMetadata && i < subgraphMetadata.output_tensor_metadata.length) {
                applyTensorMetadata(value, subgraphMetadata.output_tensor_metadata[i]);
            }
            this._outputs.push(new tflite.Argument(value ? value.name : '?', true, value ? [ value ] : []));
        }
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

tflite.Node = class {

    constructor(metadata, node, type, location, args) {
        this._location = location;
        this._type = type.custom ? { name: type.name, category: 'custom' } : metadata.type(type.name);
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        if (node) {
            let inputs = [];
            let outputs = [];
            inputs = Array.from(node.inputs || new Int32Array(0));
            outputs = Array.from(node.outputs || new Int32Array(0));
            let inputIndex = 0;
            while (inputIndex < inputs.length) {
                let count = 1;
                let name = null;
                let visible = true;
                const values = [];
                if (this._type && this._type.inputs && inputIndex < this._type.inputs.length) {
                    const input = this._type.inputs[inputIndex];
                    name = input.name;
                    if (input.list) {
                        count = inputs.length - inputIndex;
                    }
                    if (input.visible === false) {
                        visible = false;
                    }
                }
                const inputArray = inputs.slice(inputIndex, inputIndex + count);
                for (const index of inputArray) {
                    const value = args(index);
                    if (value) {
                        values.push(value);
                    }
                }
                inputIndex += count;
                name = name ? name : inputIndex.toString();
                const argument = new tflite.Argument(name, visible, values);
                this._inputs.push(argument);
            }
            for (let k = 0; k < outputs.length; k++) {
                const index = outputs[k];
                const outputArguments = [];
                const value = args(index);
                if (value) {
                    outputArguments.push(value);
                }
                let outputName = k.toString();
                if (this._type && this._type.outputs && k < this._type.outputs.length) {
                    const output = this._type.outputs[k];
                    if (output && output.name) {
                        outputName = output.name;
                    }
                }
                this._outputs.push(new tflite.Argument(outputName, true, outputArguments));
            }
            if (type.custom && node.custom_options.length > 0) {
                let decoded = false;
                if (node.custom_options_format === tflite.schema.CustomOptionsFormat.FLEXBUFFERS) {
                    try {
                        const reader = flexbuffers.BinaryReader.open(node.custom_options);
                        if (reader) {
                            const custom_options = reader.read();
                            if (Array.isArray(custom_options)) {
                                const attribute = new tflite.Attribute(null, 'custom_options', custom_options);
                                this._attributes.push(attribute);
                                decoded = true;
                            } else if (custom_options) {
                                for (const pair of Object.entries(custom_options)) {
                                    const key = pair[0];
                                    const value = pair[1];
                                    const schema = metadata.attribute(type.name, key);
                                    const attribute = new tflite.Attribute(schema, key, value);
                                    this._attributes.push(attribute);
                                }
                                decoded = true;
                            }
                        }
                    } catch (err) {
                        // continue regardless of error
                    }
                }
                if (!decoded) {
                    const schema = metadata.attribute(type.name, 'custom');
                    this._attributes.push(new tflite.Attribute(schema, 'custom', Array.from(node.custom_options)));
                }
            }
            const options = node.builtin_options;
            if (options) {
                for (const entry of Object.entries(options)) {
                    const name = entry[0];
                    const value = entry[1];
                    if (name === 'fused_activation_function' && value) {
                        const activationFunctionMap = { 1: 'Relu', 2: 'ReluN1To1', 3: 'Relu6', 4: 'Tanh', 5: 'SignBit' };
                        if (!activationFunctionMap[value]) {
                            throw new tflite.Error("Unsupported activation funtion index '" + JSON.stringify(value) + "'.");
                        }
                        const type = activationFunctionMap[value];
                        this._chain = [ new tflite.Node(metadata, null, { name: type }, null, []) ];
                    }
                    const schema = metadata.attribute(type.name, name);
                    this._attributes.push(new tflite.Attribute(schema, name, value));
                }
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return '';
    }

    get location() {
        return this._location;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }

    get attributes() {
        return this._attributes;
    }
};

tflite.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        this._type = metadata && metadata.type ? metadata.type : null;
        if (this._name == 'fused_activation_function') {
            this._visible = false;
        }
        if (this._type) {
            this._value = tflite.Utility.enum(this._type, this._value);
        }
        if (metadata) {
            if (metadata && metadata.visible == false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                value = this._value;
                if (typeof value == 'function') {
                    value = value();
                }
                if (value == metadata.default) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

tflite.Argument = class {

    constructor(name, visible, value) {
        this._name = name;
        this._visible = visible;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get value() {
        return this._value;
    }
};

tflite.Value = class {

    constructor(index, tensor, initializer) {
        const name = tensor.name || '';
        this._name = name + '\n' + index.toString();
        this._location = index.toString();
        this._type = tensor.type !== undefined && tensor.shape !== undefined ? new tflite.TensorType(tensor) : null;
        this._initializer = initializer;
        const quantization = tensor.quantization;
        if (quantization) {
            const length = Math.max(quantization.scale.length, quantization.zero_point.length, quantization.min.length, quantization.max.length);
            const list = [];
            for (let i = 0; i < length; i++) {
                let value = 'q';
                const scale = i < quantization.scale.length ? quantization.scale[i] : 0;
                const zeroPoint = (i < quantization.zero_point.length ? quantization.zero_point[i] : 0).toString();
                if (scale !== 0 || zeroPoint !== '0') {
                    value = scale.toString() + ' * ' + (zeroPoint === '0' ? 'q' : ('(q' + (!zeroPoint.startsWith('-') ? ' - ' + zeroPoint : ' + ' + zeroPoint.substring(1)) + ')'));
                }
                if (i < quantization.min.length) {
                    value = quantization.min[i].toString() + ' \u2264 ' + value;
                }
                if (i < quantization.max.length) {
                    value = value + ' \u2264 ' + quantization.max[i].toString();
                }
                list.push(value);
            }
            if (list.length > 0 && !list.every((value) => value === 'q')) {
                this._quantization = list.length === 1 ? list[0] : list;
            }
        }
    }

    get name() {
        return this._name;
    }

    get location() {
        return this._location;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        return this._quantization;
    }

    set description(value) {
        this._description = value;
    }

    get description() {
        return this._description;
    }

    get initializer() {
        return this._initializer;
    }
};

tflite.Tensor = class {

    constructor(index, tensor, buffer, is_variable) {
        this._location = index.toString();
        this._type = new tflite.TensorType(tensor);
        this._is_variable = is_variable;
        this._name = tensor.name;
        this._data = buffer.data.slice(0);
    }

    get category() {
        return this._is_variable ? 'Variable' : '';
    }

    get name() {
        return this._name;
    }

    get location() {
        return this._location;
    }

    get type() {
        return this._type;
    }

    get layout() {
        switch (this._type.dataType) {
            case 'string': return '|';
            default: return '<';
        }
    }

    get values() {
        switch (this._type.dataType) {
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
                return this._data;
            }
        }
    }
};

tflite.TensorType = class {

    constructor(tensor) {
        this._dataType = tflite.Utility.dataType(tensor.type);
        this._shape = new tflite.TensorShape(Array.from(tensor.shape || []));
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    set denotation(value) {
        this._denotation = value;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

tflite.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

tflite.Utility = class {

    static dataType(type) {
        if (!tflite.Utility._tensorTypeMap) {
            tflite.Utility._tensorTypeMap = new Map(Object.keys(tflite.schema.TensorType).map((key) => [ tflite.schema.TensorType[key], key.toLowerCase() ]));
            tflite.Utility._tensorTypeMap.set(6, 'boolean');
        }
        return tflite.Utility._tensorTypeMap.has(type) ? tflite.Utility._tensorTypeMap.get(type) : '?';
    }

    static enum(name, value) {
        const type = name && tflite.schema ? tflite.schema[name] : undefined;
        if (type) {
            tflite.Utility._enums = tflite.Utility._enums || new Map();
            if (!tflite.Utility._enums.has(name)) {
                const map = new Map(Object.keys(type).map((key) => [ type[key], key ]));
                tflite.Utility._enums.set(name, map);
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tflite.ModelFactory;
}