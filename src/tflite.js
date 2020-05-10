/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var tflite = tflite || {};
var base = base || require('./base');
var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;
var long = long || { Long: require('long') };

tflite.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'tflite' || extension === 'lite' || extension === 'tfl' || extension === 'bin' || extension === 'pb' || extension === 'model') {
            const buffer = context.buffer;
            const signature = 'TFL3';
            if (buffer && buffer.length > 8 && buffer.subarray(4, 8).every((x, i) => x === signature.charCodeAt(i))) {
                return true;
            }
        }
        if (extension === 'json') {
            const json = context.text;
            if (json.indexOf("\"subgraphs\"", 0) !== -1 && json.indexOf("\"operator_codes\"", 0) !== -1) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./tflite-schema').then((schema) => {
            tflite.schema = schema.tflite_schema;
            tflite.metadata_schema = schema.tflite_metadata_schema;
            return tflite.Metadata.open(host).then((metadata) => {
                const identifier = context.identifier;
                try {
                    const extension = identifier.split('.').pop().toLowerCase();
                    switch (extension) {
                        default: {
                            const buffer = new flatbuffers.ByteBuffer(context.buffer);
                            if (!tflite.schema.Model.bufferHasIdentifier(buffer)) {
                                throw new tflite.Error("File format is not tflite.Model.");
                            }
                            const model = tflite.schema.Model.getRootAsModel(buffer);
                            return new tflite.Model(metadata, null, model);
                        }
                        case 'json': {
                            const model = JSON.parse(context.text);
                            return new tflite.Model(metadata, 'json', model);
                        }
                    }
                }
                catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tflite.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
            });
        });
    }
};

tflite.Model = class {

    constructor(metadata, format, model) {
        this._graphs = [];
        this._format = 'TensorFlow Lite';
        switch (format) {
            default: {
                this._format = this._format + ' v' + model.version().toString();
                this._description = model.description() || '';
                const operators = [];
                const builtinOperatorMap = {};
                for (const key of Object.keys(tflite.schema.BuiltinOperator)) {
                    const index = tflite.schema.BuiltinOperator[key];
                    builtinOperatorMap[index] = tflite.Utility.operator(key);
                }
                for (let i = 0; i < model.operatorCodesLength(); i++) {
                    const operatorCode = model.operatorCodes(i);
                    const code = operatorCode.builtinCode();
                    const custom = code === tflite.schema.BuiltinOperator.CUSTOM;
                    const name = custom ? operatorCode.customCode() : builtinOperatorMap[code];
                    if (!name) {
                        throw new tflite.Error("Invalid built-in code '" + code.toString() + "' at '" + i.toString() + "'.");
                    }
                    operators.push(custom ? { name: name, custom: true } : { name: name });
                }
                /*
                for (let i = 0; i < model.metadataBufferLength(); i++) {
                    const metadataBufferIndex = model.metadataBuffer(i);
                    const data = model.buffers(metadataBufferIndex).dataArray();
                    // file_identifier "FDMD"
                    // file_identifier "MSMD"
                    // file_identifier "SEMD"
                }
                */
                let modelMetadata = null;
                for (let i = 0; i < model.metadataLength(); i++) {
                    const metadata = model.metadata(i);
                    switch (metadata.name()) {
                        case 'min_runtime_version': {
                            const data = model.buffers(metadata.buffer()).dataArray();
                            this._runtime = data ? new TextDecoder().decode(data) : undefined;
                            break;
                        }
                        case 'TFLITE_METADATA': {
                            const buffer = new flatbuffers.ByteBuffer(model.buffers(metadata.buffer()).dataArray() || []);
                            if (tflite.metadata_schema.ModelMetadata.bufferHasIdentifier(buffer)) {
                                modelMetadata = tflite.metadata_schema.ModelMetadata.getRootAsModelMetadata(buffer);
                                this._name = modelMetadata.name() || '';
                                this._version = modelMetadata.version() || '';
                                this._description = modelMetadata.description() ? [ this.description, modelMetadata.description()].join(' ') : this._description;
                                this._author = modelMetadata.author() || '';
                                this._license = modelMetadata.license() || '';
                            }
                            break;
                        }
                    }
                }
                const subgraphsLength = model.subgraphsLength();
                for (let i = 0; i < subgraphsLength; i++) {
                    const subgraph = model.subgraphs(i);
                    const name = subgraphsLength > 1 ? i.toString() : '';
                    const subgraphMetadata = modelMetadata && i < modelMetadata.subgraphMetadataLength() ? modelMetadata.subgraphMetadata(i) : null;
                    this._graphs.push(new tflite.Graph(metadata, format, subgraph, subgraphMetadata, name, operators, model));
                }
                break;
            }
            case 'json': {
                this._format = this._format + (model.version ? ' v' + model.version.toString() : '');
                this._description = model.description || '';
                const operators = [];
                if (model.operator_codes && Array.isArray(model.operator_codes)) {
                    for (let i = 0; i < model.operator_codes.length; i++) {
                        const operatorCode = model.operator_codes[i];
                        const code = operatorCode.builtin_code;
                        const custom = code === 'CUSTOM';
                        const name = custom ? operatorCode.custom_code : tflite.Utility.operator(code);
                        if (!name) {
                            throw new tflite.Error("Invalid built-in code '" + code.toString() + "' at '" + i.toString() + "'.");
                        }
                        operators.push(custom ? { name: name, custom: true } : { name: name });
                    }
                }
                if (model.subgraphs && Array.isArray(model.subgraphs)) {
                    const subgraphsLength = model.subgraphs.length;
                    for (let i = 0; i < subgraphsLength; i++) {
                        const subgraph = model.subgraphs[i];
                        const name = subgraphsLength > 1 ? i.toString() : '';
                        this._graphs.push(new tflite.Graph(metadata, format, subgraph, null, name, operators, model));
                    }
                }
                break;
            }
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

    constructor(metadata, format, subgraph, subgraphMetadata, name, operators, model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        switch (format) {
            default: {
                this._name = subgraph.name() || name;
                const args = [];
                const tensorNames = [];
                for (let i = 0; i < subgraph.tensorsLength(); i++) {
                    const tensor = subgraph.tensors(i);
                    const buffer = model.buffers(tensor.buffer());
                    const is_variable = tensor.isVariable();
                    const initializer = buffer.dataLength() > 0 || is_variable ? new tflite.Tensor(format, i, tensor, buffer, is_variable) : null;
                    args.push(new tflite.Argument(format, i, tensor, initializer));
                    tensorNames.push(tensor.name());
                }
                for (let i = 0; i < subgraph.operatorsLength(); i++) {
                    const node = subgraph.operators(i);
                    const index = node.opcodeIndex();
                    const operator = index < operators.length ? operators[index] : { name: '(' + index.toString() + ')' };
                    this._nodes.push(new tflite.Node(metadata, format, node, operator, i.toString(), args));
                }
                const applyTensorMetadata = (argument, tensorMetadata) => {
                    if (tensorMetadata) {
                        const description = tensorMetadata.description();
                        if (description) {
                            argument.description = description;
                        }
                        const content = tensorMetadata.content();
                        if (argument.type && content) {
                            let denotation = null;
                            switch (content.contentPropertiesType()) {
                                case 1: {
                                    denotation = 'Feature';
                                    break;
                                }
                                case 2: {
                                    denotation = 'Image';
                                    const imageProperties = content.contentProperties(Reflect.construct(tflite.metadata_schema.ImageProperties, []));
                                    switch(imageProperties.colorSpace()) {
                                        case 1: denotation += '(RGB)'; break;
                                        case 2: denotation += '(Grayscale)'; break;
                                    }
                                    break;
                                }
                                case 3: {
                                    denotation = 'BoundingBox';
                                    break;
                                }
                            }
                            if (denotation) {
                                argument.type.denotation = denotation;
                            }
                        }
                    }
                };
                for (let i = 0; i < subgraph.inputsLength(); i++) {
                    const input = subgraph.inputs(i);
                    const argument = args[input];
                    if (subgraphMetadata && i < subgraphMetadata.inputTensorMetadataLength()) {
                        applyTensorMetadata(argument, subgraphMetadata.inputTensorMetadata(i));
                    }
                    this._inputs.push(new tflite.Parameter(tensorNames[input], true, [ argument ]));
                }
                for (let i = 0; i < subgraph.outputsLength(); i++) {
                    const output = subgraph.outputs(i);
                    const argument = args[output];
                    if (subgraphMetadata && i < subgraphMetadata.outputTensorMetadataLength()) {
                        applyTensorMetadata(argument, subgraphMetadata.outputTensorMetadata(i));
                    }
                    this._outputs.push(new tflite.Parameter(tensorNames[output], true, [ argument ]));
                }
                break;
            }
            case 'json': {
                this._name = subgraph.name || '';
                const args = [];
                const tensorNames = [];
                if (subgraph.tensors && Array.isArray(subgraph.tensors)) {
                    for (let i = 0; i < subgraph.tensors.length; i++) {
                        const tensor = subgraph.tensors[i];
                        const buffer = model.buffers[tensor.buffer];
                        const is_variable = tensor.isVariable;
                        const initializer = buffer.data && buffer.data.length > 0 || is_variable ? new tflite.Tensor(format, i, tensor, buffer, is_variable) : null;
                        args.push(new tflite.Argument(format, i, tensor, initializer));
                        tensorNames.push(tensor.name);
                    }
                }
                if (subgraph.operators && Array.isArray(subgraph.operators)) {
                    for (let i = 0; i < subgraph.operators.length; i++) {
                        const node = subgraph.operators[i];
                        const index = node.opcode_index;
                        const operator = index < operators.length ? operators[index] : { name: '(' + index.toString() + ')' };
                        this._nodes.push(new tflite.Node(metadata, format, node, operator, i.toString(), args));
                    }
                }
                if (subgraph.inputs && Array.isArray(subgraph.inputs)) {
                    for (const input of subgraph.inputs) {
                        this._inputs.push(new tflite.Parameter(tensorNames[input], true, [ args[input] ]));
                    }
                }
                if (subgraph.outputs && Array.isArray(subgraph.outputs)) {
                    for (const output of subgraph.outputs) {
                        this._outputs.push(new tflite.Parameter(tensorNames[output], true, [ args[output] ]));
                    }
                }
                break;
            }
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

    constructor(metadata, format, node, operator, location, args) {
        this._metadata = metadata;
        this._location = location;
        this._operator = operator;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        if (node) {
            let inputs = [];
            let outputs = [];
            switch (format) {
                default: {
                    inputs = Array.from(node.inputsArray() || []);
                    outputs = Array.from(node.outputsArray() || []);
                    break;
                }
                case 'json': {
                    inputs = node.inputs && Array.isArray(node.inputs) ? node.inputs : [];
                    outputs = node.outputs && Array.isArray(node.outputs) ? node.outputs : [];
                    break;
                }
            }
            const schema = this._metadata.type(this.operator);
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
            switch (format) {
                default: {
                    if (operator.custom && node.customOptionsLength() > 0) {
                        const custom = Array.from(node.customOptionsArray() || []);
                        this._attributes.push(new tflite.Attribute(this._metadata, format, this.operator, 'custom', custom));
                    }
                    let optionsTypeName = this.operator + 'Options';
                    switch (this.operator) {
                        case 'AveragePool2D':
                        case 'MaxPool2D':
                            optionsTypeName = 'Pool2DOptions';
                            break;
                        case 'Mean':
                        case 'ReduceMax':
                        case 'ReduceMin':
                        case 'Sum':
                            optionsTypeName = 'ReducerOptions';
                            break;
                        case 'Minimum':
                        case 'Maximum':
                            optionsTypeName = 'MaximumMinimumOptions';
                            break;
                    }
                    const optionsType = tflite.schema[optionsTypeName] || null;
                    if (typeof optionsType === 'function') {
                        const options = node.builtinOptions(Reflect.construct(optionsType, []));
                        if (options) {
                            const names = new Set(Object.keys(Object.getPrototypeOf(options)).filter((name) => name !== '__init'));
                            const arrayNames = new Set();
                            for (const name of new Set(names)) {
                                if (names.has(name + 'Array') && names.has(name + 'Length')) {
                                    names.delete(name + 'Array');
                                    names.delete(name + 'Length');
                                    arrayNames.add(name);
                                }
                            }
                            for (const name of names) {
                                if (options[name] && typeof options[name] == 'function') {
                                    const value = arrayNames.has(name) ? Array.from(options[name + 'Array']() || []) : options[name]();
                                    if (name === 'fusedActivationFunction' && value !== 0) {
                                        const activationFunctionMap = { 1: 'Relu', 2: 'ReluN1To1', 3: 'Relu6', 4: 'Tanh', 5: 'SignBit' };
                                        if (!activationFunctionMap[value]) {
                                            throw new tflite.Error("Unknown activation funtion index '" + JSON.stringify(value) + "'.");
                                        }
                                        const operator = activationFunctionMap[value];
                                        this._chain = [ new tflite.Node(metadata, format, null, { name: operator }, null, []) ];
                                    }
                                    this._attributes.push(new tflite.Attribute(this._metadata, format, this.operator, name, value));
                                }
                            }
                        }
                    }
                    break;
                }
                case 'json': {
                    if (node.builtin_options && !Array.isArray(node.builtin_options)) {
                        if (operator.custom && Array.isArray(operator.custom)) {
                            this._attributes.push(new tflite.Attribute(this._metadata, format, this.operator, 'custom', operator.custom));
                        }
                        for (const name of Object.keys(node.builtin_options)) {
                            const value = node.builtin_options[name];
                            if (name === 'fused_activation_function' && value !== 'NONE') {
                                const activationFunctionMap = { 'RELU': 'Relu', 'RELU_N1_TO_1': 'ReluN1To1', 'RELU6': 'Relu6', 'TANH': 'Tanh', 'SIGN_BIT': 'SignBit' };
                                if (!activationFunctionMap[value]) {
                                    throw new tflite.Error("Unknown activation funtion index '" + JSON.stringify(value) + "'.");
                                }
                                const operator = activationFunctionMap[value];
                                this._chain = [ new tflite.Node(metadata, format, null, { name: operator }, null, []) ];
                            }
                            this._attributes.push(new tflite.Attribute(this._metadata, format, this.operator, name, value));
                        }
                    }
                    break;
                }
            }
        }
    }

    get operator() {
        return this._operator.name;
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
        if (this._operator.custom) {
            return { name: this.operator, category: 'custom' };
        }
        return this._metadata.type(this.operator);
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

    constructor(metadata, format, operator, name, value) {
        this._type = null;
        this._name = '';
        this._value = value;
        const lower = name.toLowerCase();
        for (let i = 0; i < name.length; i++) {
            this._name += (name[i] == lower[i]) ? name[i] : ('_' + lower[i]);
        }
        if (this._name == 'fused_activation_function') {
            this._visible = false;

        }
        const schema = metadata.attribute(operator, this._name);
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
                        this._value = tflite.Utility.dataType(format, this._value);
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

    constructor(format, index, tensor, initializer) {
        this._location = index.toString();
        this._type = new tflite.TensorType(format, tensor);
        this._initializer = initializer;
        switch (format) {
            default: {
                this._name = tensor.name();
                const quantization = tensor.quantization();
                if (quantization) {
                    let value = 'q';
                    const scale = (quantization.scaleLength() == 1) ? quantization.scale(0) : 0;
                    const zeroPoint = (quantization.zeroPointLength() == 1) ? quantization.zeroPoint(0).toFloat64() : 0;
                    if (scale != 0 || zeroPoint != 0) {
                        value = scale.toString() + ' * ' + (zeroPoint == 0 ? 'q' : ('(q - ' + zeroPoint.toString() + ')'));
                    }
                    if (quantization.minLength() == 1) {
                        value = quantization.min(0).toString() + ' \u2264 ' + value;
                    }
                    if (quantization.maxLength() == 1) {
                        value = value + ' \u2264 ' + quantization.max(0).toString();
                    }
                    if (value != 'q') {
                        this._quantization = value;
                    }
                }
                break;
            }
            case 'json': {
                this._name = tensor.name || '';
                break;
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

    constructor(format, index, tensor, buffer, is_variable) {
        this._location = index.toString();
        this._type = new tflite.TensorType(format, tensor);
        this._is_variable = is_variable;
        switch (format) {
            default: {
                this._name = tensor.name();
                this._data = buffer.dataLength() > 0 ? buffer.dataArray() || [] : null;
                break;
            }
            case 'json': {
                this._name = tensor.name || '';
                this._data = buffer.data && buffer.data.length > 0 ? new Uint8Array(buffer.data) : null;
                break;
            }
        }
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

        if (this._data == null) {
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
                        results.push(new long.Long(context.data.getUint32(context.index, true), context.data.getUint32(context.index + 4, true), false));
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

    constructor(format, tensor) {
        switch (format) {
            default: {
                this._dataType = tflite.Utility.dataType(format, tensor.type());
                this._shape = new tflite.TensorShape(Array.from(tensor.shapeArray() || []));
                break;
            }
            case 'json': {
                this._dataType = tflite.Utility.dataType(format, tensor.type);
                this._shape = new tflite.TensorShape(tensor.shape || []);
                break;
            }
        }
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

    type(operator) {
        return this._map.has(operator) ? this._map.get(operator) : null;
    }

    attribute(operator, name) {
        const schema = this.type(operator);
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

    static dataType(format, type) {
        switch (format) {
            default: {
                if (!tflite.Utility._tensorTypeMap) {
                    tflite.Utility._tensorTypeMap = new Map();
                    for (const name of Object.keys(tflite.schema.TensorType)) {
                        tflite.Utility._tensorTypeMap.set(tflite.schema.TensorType[name], name.toLowerCase());
                    }
                    tflite.Utility._tensorTypeMap.set(6, 'boolean');
                }
                return tflite.Utility._tensorTypeMap.has(type) ? tflite.Utility._tensorTypeMap.get(type) : '?';
            }
            case 'json': {
                switch (type) {
                    case 'BOOL': return 'boolean';
                    default: return type.toLowerCase();
                }
            }
        }
    }

    static enum(type, value) {
        if (type && tflite.schema && tflite.schema[type]) {
            if (!tflite.Utility._enumTypeMap) {
                tflite.Utility._enumTypeMap = new Map();
            }
            let typeMap = tflite.Utility._enumTypeMap.get(type);
            if (!typeMap) {
                typeMap = new Map();
                const enumType = tflite.schema[type];
                if (enumType) {
                    for (const key of Object.keys(enumType)) {
                        typeMap.set(enumType[key], key);
                    }
                }
                tflite.Utility._enumTypeMap.set(type, typeMap);
            }
            if (typeMap.has(value)) {
                return typeMap.get(value);
            }
        }
        return value;
    }

    static operator(name) {
        const upperCase = new Set([ '2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM' ]);
        if (name === 'BATCH_MATMUL') {
            return "BatchMatMul";
        }
        return name.split('_').map((s) => (s.length < 1 || upperCase.has(s)) ? s : s[0] + s.substring(1).toLowerCase()).join('');
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