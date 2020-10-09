/* jshint esversion: 6 */

var tflite = tflite || {};
var flatbuffers = flatbuffers || require('./flatbuffers');
var flexbuffers = {};

tflite.ModelFactory = class {

    match(context) {
        const buffer = context.buffer;
        const signature = 'TFL3';
        if (buffer && buffer.length > 8 && buffer.subarray(4, 8).every((x, i) => x === signature.charCodeAt(i))) {
            return true;
        }
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const tags = context.tags('json');
            if (tags.has('subgraphs') && tags.has('operator_codes')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./tflite-schema').then(() => {
            tflite.schema = flatbuffers.get('tflite').tflite;
            let model = null;
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'json':
                    try {
                        const reader = new flatbuffers.TextReader(context.buffer);
                        model = tflite.schema.Model.createText(reader);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new tflite.Error('File text format is not tflite.Model (' + message.replace(/\.$/, '') + ').');
                    }
                    break;
                default:
                    try {
                        const reader = new flatbuffers.Reader(context.buffer);
                        if (!tflite.schema.Model.identifier(reader)) {
                            throw new tflite.Error('Invalid identifier.');
                        }
                        model = tflite.schema.Model.create(reader);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new tflite.Error('File format is not tflite.Model (' + message.replace(/\.$/, '') + ').');
                    }
                    break;
            }
            return tflite.Metadata.open(host).then((metadata) => {
                return new tflite.Model(metadata, model);
            });
        });
    }
};

tflite.Model = class {

    constructor(metadata, model) {
        this._graphs = [];
        this._format = 'TensorFlow Lite';
        this._format = this._format + ' v' + model.version.toString();
        this._description = model.description || '';
        const operatorList = [];
        const builtinOperatorMap = new Map();
        for (const key of Object.keys(tflite.schema.BuiltinOperator)) {
            const index = tflite.schema.BuiltinOperator[key];
            builtinOperatorMap.set(index, tflite.Utility.type(key));
        }
        for (let i = 0; i < model.operator_codes.length; i++) {
            const operatorCode = model.operator_codes[i];
            const code = operatorCode.deprecated_builtin_code < tflite.schema.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES ? operatorCode.deprecated_builtin_code : operatorCode.builtin_code;
            const version = operatorCode.version;
            const custom = code === tflite.schema.BuiltinOperator.CUSTOM;
            const name = custom ? operatorCode.custom_code : builtinOperatorMap.get(code);
            if (!name) {
                throw new tflite.Error("Invalid built-in code '" + code.toString() + "' at '" + i.toString() + "'.");
            }
            operatorList.push(custom ? { name: name, version: version, custom: true } : { name: name, version: version });
        }
        let modelMetadata = null;
        for (const metadata of model.metadata) {
            switch (metadata.name) {
                case 'min_runtime_version': {
                    const data = model.buffers[metadata.buffer].data;
                    this._runtime = data ? new TextDecoder().decode(data) : undefined;
                    break;
                }
                case 'TFLITE_METADATA': {
                    const data = model.buffers[metadata.buffer].data || new Uint8Array(0);
                    const reader = new flatbuffers.Reader(data);
                    if (tflite.schema.ModelMetadata.identifier(reader)) {
                        modelMetadata = tflite.schema.ModelMetadata.create(reader);
                        this._name = modelMetadata.name || '';
                        this._version = modelMetadata.version || '';
                        this._description = modelMetadata.description ? [ this.description, modelMetadata.description].join(' ') : this._description;
                        this._author = modelMetadata.author || '';
                        this._license = modelMetadata.license || '';
                    }
                    break;
                }
            }
        }
        const subgraphs = model.subgraphs;
        const subgraphsMetadata = modelMetadata ? modelMetadata.subgraph_metadata : null;
        for (let i = 0; i < subgraphs.length; i++) {
            const subgraph = subgraphs[i];
            const name = subgraphs.length > 1 ? i.toString() : '';
            const subgraphMetadata = subgraphsMetadata && i < subgraphsMetadata.length ? subgraphsMetadata[i] : null;
            this._graphs.push(new tflite.Graph(metadata, subgraph, subgraphMetadata, name, operatorList, model));
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

    get author() {
        return this._author;
    }

    get license() {
        return this._license;
    }

    get graphs() {
        return this._graphs;
    }
};

tflite.Graph = class {

    constructor(metadata, subgraph, subgraphMetadata, name, operatorList, model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._name = subgraph.name || name;
        const args = [];
        const tensorNames = [];
        for (let i = 0; i < subgraph.tensors.length; i++) {
            const tensor = subgraph.tensors[i];
            const buffer = model.buffers[tensor.buffer];
            const is_variable = tensor.is_variable;
            const data = buffer.data;
            const initializer = (data && data.length > 0) || is_variable ? new tflite.Tensor(i, tensor, buffer, is_variable) : null;
            args.push(new tflite.Argument(i, tensor, initializer));
            tensorNames.push(tensor.name);
        }
        const operators = subgraph.operators;
        for (let i = 0; i < subgraph.operators.length; i++) {
            const node = operators[i];
            const index = node.opcode_index;
            const operator = index < operatorList.length ? operatorList[index] : { name: '(' + index.toString() + ')' };
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
                    }
                    else if (contentProperties instanceof tflite.schema.ImageProperties) {
                        denotation = 'Image';
                        switch(contentProperties.color_space) {
                            case 1: denotation += '(RGB)'; break;
                            case 2: denotation += '(Grayscale)'; break;
                        }
                    }
                    else if (contentProperties instanceof tflite.schema.BoundingBoxProperties) {
                        denotation = 'BoundingBox';
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
            const argument = args[input];
            if (subgraphMetadata && i < subgraphMetadata.input_tensor_metadata.length) {
                applyTensorMetadata(argument, subgraphMetadata.input_tensor_metadata[i]);
            }
            this._inputs.push(new tflite.Parameter(tensorNames[input], true, [ argument ]));
        }
        const outputs = subgraph.outputs;
        for (let i = 0; i < outputs.length; i++) {
            const output = outputs[i];
            const argument = args[output];
            if (subgraphMetadata && i < subgraphMetadata.output_tensor_metadata.length) {
                applyTensorMetadata(argument, subgraphMetadata.output_tensor_metadata[i]);
            }
            this._outputs.push(new tflite.Parameter(tensorNames[output], true, [ argument ]));
        }
    }

    get name() {
        return this._name;
    }

    get groups() {
        return false;
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
        this._metadata = metadata;
        this._location = location;
        this._type = type;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        if (node) {
            let inputs = [];
            let outputs = [];
            inputs = Array.from(node.inputs || new Int32Array(0));
            outputs = Array.from(node.outputs || new Int32Array(0));
            const schema = this._metadata.type(this.type);
            let inputIndex = 0;
            while (inputIndex < inputs.length) {
                let count = 1;
                let inputName = null;
                let inputVisible = true;
                const inputArguments = [];
                if (schema && schema.inputs && inputIndex < schema.inputs.length) {
                    const input = schema.inputs[inputIndex];
                    inputName = input.name;
                    if (input.option == 'variadic') {
                        count = inputs.length - inputIndex;
                    }
                    if (Object.prototype.hasOwnProperty.call(input, 'visible') && !input.visible) {
                        inputVisible = false;
                    }
                }
                const inputArray = inputs.slice(inputIndex, inputIndex + count);
                for (let j = 0; j < inputArray.length; j++) {
                    if (inputArray[j] != -1) {
                        inputArguments.push(args[inputArray[j]]);
                    }
                }
                inputIndex += count;
                inputName = inputName ? inputName : inputIndex.toString();
                this._inputs.push(new tflite.Parameter(inputName, inputVisible, inputArguments));
            }
            for (let k = 0; k < outputs.length; k++) {
                const outputIndex = outputs[k];
                const argument = args[outputIndex];
                let outputName = k.toString();
                if (schema && schema.outputs && k < schema.outputs.length) {
                    const output = schema.outputs[k];
                    if (output && (!output.option || output.opcodeIndex != 'variadic') && output.name) {
                        outputName = output.name;
                    }
                }
                this._outputs.push(new tflite.Parameter(outputName, true, [ argument ]));
            }
            if (type.custom && node.custom_options.length > 0) {
                let decoded = false;
                if (node.custom_options_format === tflite.schema.CustomOptionsFormat.FLEXBUFFERS) {
                    try {
                        const reader = flexbuffers.Reader.create(node.custom_options);
                        const custom_options = reader.read();
                        for (const key of Object.keys(custom_options)) {
                            const schema = metadata.attribute(this.type, key);
                            const value = custom_options[key];
                            this._attributes.push(new tflite.Attribute(schema, key, value));
                        }
                        decoded = true;
                    }
                    catch (err) {
                        // continue regardless of error
                    }
                }
                if (!decoded) {
                    const schema = metadata.attribute(this.type, 'custom');
                    this._attributes.push(new tflite.Attribute(schema, 'custom', Array.from(node.custom_options)));
                }
            }
            const options = node.builtin_options;
            if (options) {
                for (const name of Object.keys(options)) {
                    const value = options[name];
                    if (name === 'fused_activation_function' && value !== 0) {
                        const activationFunctionMap = { 1: 'Relu', 2: 'ReluN1To1', 3: 'Relu6', 4: 'Tanh', 5: 'SignBit' };
                        if (!activationFunctionMap[value]) {
                            throw new tflite.Error("Unknown activation funtion index '" + JSON.stringify(value) + "'.");
                        }
                        const type = activationFunctionMap[value];
                        this._chain = [ new tflite.Node(metadata, null, { name: type }, null, []) ];
                    }
                    const schema = metadata.attribute(this.type, name);
                    this._attributes.push(new tflite.Attribute(schema, name, value));
                }
            }
        }
    }

    get type() {
        return this._type.name;
    }

    get name() {
        return '';
    }

    get location() {
        return this._location;
    }

    get domain() {
        return null;
    }

    get metadata() {
        if (this._type.custom) {
            return { name: this.type, category: 'custom' };
        }
        return this._metadata.type(this.type);
    }

    get group() {
        return null;
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

    constructor(schema, name, value) {
        this._type = null;
        this._name = name;
        this._value = value;
        if (this._name == 'fused_activation_function') {
            this._visible = false;
        }
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
            }
            if (this._type) {
                switch (this._type) {
                    case 'shape':
                        this._value = new tflite.TensorShape(value);
                        break;
                    case 'TensorType':
                        this._value = tflite.Utility.dataType(this._value);
                        break;
                    default:
                        this._value = tflite.Utility.enum(this._type, this._value);
                        break;
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                value = this._value;
                if (typeof value == 'function') {
                    value = value();
                }
                if (value == schema.default) {
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

tflite.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

tflite.Argument = class {

    constructor(index, tensor, initializer) {
        this._location = index.toString();
        this._type = new tflite.TensorType(tensor);
        this._initializer = initializer;
        this._name = tensor.name;
        const quantization = tensor.quantization;
        if (quantization) {
            let value = 'q';
            const scale = (quantization.scale.length == 1) ? quantization.scale[0] : 0;
            const zeroPoint = (quantization.zero_point.length == 1) ? quantization.zero_point[0] : 0;
            if (scale != 0 || zeroPoint != 0) {
                value = scale.toString() + ' * ' + (zeroPoint == 0 ? 'q' : ('(q - ' + zeroPoint.toString() + ')'));
            }
            if (quantization.min.length == 1) {
                value = quantization.min[0].toString() + ' \u2264 ' + value;
            }
            if (quantization.max.length == 1) {
                value = value + ' \u2264 ' + quantization.max[0].toString();
            }
            if (value != 'q') {
                this._quantization = value;
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

    get kind() {
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

    get state() {
        return this._context().state;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null || this._data.length === 0) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        if (this._type.dataType == 'string') {
            let offset = 0;
            const count = context.data.getInt32(0, true);
            offset += 4;
            const offsetTable = [];
            for (let j = 0; j < count; j++) {
                offsetTable.push(context.data.getInt32(offset, true));
                offset += 4;
            }
            offsetTable.push(this._data.length);
            const stringTable = [];
            const utf8Decoder = new TextDecoder('utf-8');
            for (let k = 0; k < count; k++) {
                const textArray = this._data.subarray(offsetTable[k], offsetTable[k + 1]);
                stringTable.push(utf8Decoder.decode(textArray));
            }
            context.data = stringTable;
        }
        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length == 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        const results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.data.getInt16(context.index));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.data.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.data.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'string':
                        results.push(context.data[context.index++]);
                        context.count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
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

tflite.Metadata = class {

    static open(host) {
        if (tflite.Metadata._metadata) {
            return Promise.resolve(tflite.Metadata._metadata);
        }
        return host.request(null, 'tflite-metadata.json', 'utf-8').then((data) => {
            tflite.Metadata._metadata = new tflite.Metadata(data);
            return tflite.Metadata._metadata;
        }).catch(() => {
            tflite.Metadata._metadata = new tflite.Metadata(null);
            return tflite.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    item.schema.name = item.name;
                    this._map.set(item.name, item.schema);
                }
            }
        }
    }

    type(name) {
        return this._map.has(name) ? this._map.get(name) : null;
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
    }
};

tflite.Utility = class {

    static dataType(type) {
        if (!tflite.Utility._tensorTypeMap) {
            tflite.Utility._tensorTypeMap = new Map();
            for (const name of Object.keys(tflite.schema.TensorType)) {
                tflite.Utility._tensorTypeMap.set(tflite.schema.TensorType[name], name.toLowerCase());
            }
            tflite.Utility._tensorTypeMap.set(6, 'boolean');
        }
        return tflite.Utility._tensorTypeMap.has(type) ? tflite.Utility._tensorTypeMap.get(type) : '?';
    }

    static enum(name, value) {
        const type = name && tflite.schema ? tflite.schema[name] : undefined;
        if (type) {
            tflite.Utility._enumKeyMap = tflite.Utility._enumKeyMap || new Map();
            if (!tflite.Utility._enumKeyMap.has(name)) {
                const map = new Map();
                for (const key of Object.keys(type)) {
                    map.set(type[key], key);
                }
                tflite.Utility._enumKeyMap.set(name, map);
            }
            const map = tflite.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }

    static type(name) {
        const upperCase = new Set([ '2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM' ]);
        name === 'BATCH_MATMUL' ? 'BATCH_MAT_MUL' : name;
        return name.split('_').map((s) => (s.length < 1 || upperCase.has(s)) ? s : s[0] + s.substring(1).toLowerCase()).join('');
    }
};

tflite.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow Lite model.';
    }
};

flexbuffers.Reader = class {

    constructor(buffer) {
        this._reader = new flexbuffers.BinaryReader(buffer);
    }

    static create(buffer) {
        return new flexbuffers.Reader(buffer);
    }

    read() {
        const length = this._reader.length;
        if (length < 3) {
            throw 'Invalid buffer size.';
        }
        const byteSize = this._reader.uint(length - 1, 0);
        if (byteSize > 8) {
            throw 'Invalid byte size.';
        }
        const bitSize = byteSize >> 2;
        const packedType = this._reader.uint(length - 2, 0);
        const offset = length - 2 - byteSize;
        return new flexbuffers.Reference(this._reader, offset, bitSize, packedType).read();
    }
};

flexbuffers.Reference = class {

    constructor(reader, offset, parentBitSize, packedType) {
        this._reader = reader;
        this._offset = offset;
        this._parentBitSize = parentBitSize;
        this._bitSize = packedType & 3;
        this._byteSize = 1 << this._bitSize;
        this._valueType = packedType >> 2;
    }

    read() {
        switch (this._valueType) {
            case 0x00:   // null
                return null;
            case 0x01:   // int
                return this._reader.int(this._offset, this._parentBitSize);
            case 0x02:   // uint
                return this._reader.uint(this._offset, this._parentBitSize);
            case 0x03:   // float
                return this._reader.float(this._offset, this._parentBitSize);
            case 0x04: {
                const offset = this._reader.indirect(this._offset, this._parentBitSize);
                let size = 0;
                while (this._reader.int(offset + size, 0) !== 0) {
                    size++;
                }
                return this._reader.string(offset, size);
            }
            case 0x05: { // string
                const offset = this._reader.indirect(this._offset, this._parentBitSize);
                let sizeByteSize = this._byteSize;
                let size = this._reader.int(offset - sizeByteSize, this._bitSize);
                while (this._reader.int(offset + size, 0) !== 0) {
                    sizeByteSize <<= 1;
                    size = this._reader.int(offset - sizeByteSize, this._bitSize);
                }
                return this._reader.string(offset, size);
            }
            case 0x06: // indirect int
                return this._reader.int(this._offset, this._reader.indirect(this._offset, this._parentBitSize), this._bitSize);
            case 0x07: // indirect uint
                return this._reader.uint(this._offset, this._reader.indirect(this._offset, this._parentBitSize), this._bitSize);
            case 0x08:   // indirect float
                return this._reader.float(this._reader.indirect(this._offset, this._parentBitSize), this._bitSize);
            case 0x09: { // map
                const length = this._reader.int(this._reader.indirect(this._offset, this._parentBitSize) - this._byteSize, this._bitSize);
                const keysOffset = this._reader.indirect(this._offset, this._parentBitSize) - (this._byteSize * 3);
                const keysVectorOffset = this._reader.indirect(keysOffset, this._bitSize);
                const keyByteSize = this._reader.int(keysOffset + this._byteSize, this._bitSize);
                let keyBitSize;
                switch (keyByteSize) {
                    case 1: keyBitSize = 0; break;
                    case 2: keyBitSize = 1; break;
                    case 4: keyBitSize = 2; break;
                    case 8: keyBitSize = 3; break;
                }
                const valuesOffset = this._reader.indirect(this._offset, this._parentBitSize);
                const obj = {};
                for (let i = 0; i < length; i++) {
                    const keyOffset = keysVectorOffset + (i * keyByteSize);
                    const keyReference = new flexbuffers.Reference(this._reader, keyOffset, keyBitSize, (0x04 << 2) | keyBitSize);
                    const key = keyReference.read();
                    const valueOffset = valuesOffset + (i * this._byteSize);
                    const packedType = this._reader.uint(valuesOffset + (length * this._byteSize) + i, 0);
                    const valueReference = new flexbuffers.Reference(this._reader, valueOffset, this._bitSize, packedType);
                    const value = valueReference.read();
                    obj[key] = value;
                }
                return obj;
            }
            case 0x0a: { // vector
                const length = this._reader.int(this._reader.indirect(this._offset, this._parentBitSize) - this._byteSize, this._bitSize);
                const arr = new Array(length);
                for (let i = 0; i < length; i++) {
                    const itemsOffset = this._reader.indirect(this._offset, this._parentBitSize);
                    const itemOffset = itemsOffset + (i * this._byteSize);
                    const packedType = this._reader.uint(itemsOffset + (length * this._byteSize) + i, 0);
                    const itemReference = new flexbuffers.Reference(this._reader, itemOffset, this._bitSize, packedType);
                    arr[i] = itemReference.read();
                }
                return arr;
            }
            case 0x0b:   // vector int
            case 0x0c:   // vector uint
            case 0x0d:   // vector float
            case 0x0e:   // vector key
            case 0x0f:   // vector string deprecated
            case 0x24: { // vector bool
                const length = this._reader.int(this._reader.indirect(this._offset, this._parentBitSize) - this._byteSize, this._bitSize);
                const valueType = this._valueType - 0x0b + 0x01;
                const packedType = valueType << 2 | 0;
                const arr = new Array(length);
                for (let i = 0; i < length; i++) {
                    const itemsOffset = this._reader.indirect(this._offset, this._parentBitSize);
                    const itemOffset = itemsOffset + (i * this._byteSize);
                    const itemReference = new flexbuffers.Reference(this._reader, itemOffset, this._bitSize, packedType);
                    arr[i] = itemReference.read();
                }
                return arr;
            }
            case 0x10:   // vector int2
            case 0x11:   // vector uint2
            case 0x12:   // vector float2
            case 0x13:   // vector int3
            case 0x14:   // vector uint3
            case 0x15:   // vector float3
            case 0x16:   // vector int4
            case 0x17:   // vector uint4
            case 0x18: { // vector float4
                const length = (((this._valueType - 0x10) / 3) >> 0) + 2;
                const valueType = ((this._valueType - 0x10) % 3) + 0x01;
                const packedType = valueType << 2 | 0;
                const arr = new Array(length);
                for (let i = 0; i < length; i++) {
                    const itemsOffset = this._reader.indirect(this._offset, this._parentBitSize);
                    const itemOffset = itemsOffset + (i * this._byteSize);
                    const itemReference = new flexbuffers.Reference(this._reader, itemOffset, this._bitSize, packedType);
                    arr[i] = itemReference.read();
                }
                return arr;
            }
            case 0x19: { // blob
                const sizeOffset = this._reader.indirect(this._offset, this._parentBitSize) - this._byteSize;
                const size = this._reader.int(sizeOffset, this._bitSize);
                const offset = this._reader.indirect(this._offset, this._parentBitSize);
                return this._reader.bytes(offset, size);
            }
            case 0x1A: { // bool
                return this._reader.int(this._offset, this._parentBitSize) > 0;
            }
        }
        return undefined;
    }
};

flexbuffers.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._utf8Decoder = new TextDecoder('utf-8');
    }

    get length() {
        return this._length;
    }

    int(offset, size) {
        switch (size) {
            case 0: return this._view.getInt8(offset);
            case 1: return this._view.getInt16(offset, true);
            case 2: return this._view.getInt32(offset, true);
            case 3: return this._view.getInt64(offset, true);
        }
        throw new flexbuffers.Error('Invalid int size.');
    }

    uint(offset, size) {
        switch (size) {
            case 0: return this._view.getUint8(offset);
            case 1: return this._view.getUint16(offset, true);
            case 2: return this._view.getUint32(offset, true);
            case 3: return this._view.getUint64(offset, true);
        }
        throw new flexbuffers.Error('Invalid uint size.');
    }

    float(offset, size) {
        switch (size) {
            case 2:
                return this._view.getFloat32(offset, true);
            case 3:
                return this._view.getFloat64(offset, true);
        }
        throw new flexbuffers.Error('Invalid float size.');
    }

    string(offset, size) {
        const bytes = this._buffer.subarray(offset, offset + size);
        return this._utf8Decoder.decode(bytes);
    }

    bytes(offset, size) {
        return this._buffer.slice(offset, offset + size);
    }

    indirect(offset, size) {
        return offset - this.uint(offset, size);
    }
};

flexbuffers.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'FlexBuffers Error';
        this.message = message;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tflite.ModelFactory;
}