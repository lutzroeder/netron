
var cntk = {};
var base = require('./base');
var protobuf = require('./protobuf');

var cntk_v1 = {};
var cntk_v2 = null;

cntk.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        // CNTK v1
        const signature = [ 0x42, 0x00, 0x43, 0x00, 0x4e, 0x00, 0x00, 0x00 ];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            return 'cntk.v1';
        }
        // CNTK v2
        const tags = context.tags('pb');
        if (tags.get(1) === 0 && tags.get(2) === 2) {
            return 'cntk.v2';
        }
        return undefined;
    }

    async open(context, target) {
        const metadata = await context.metadata('cntk-metadata.json');
        switch (target) {
            case 'cntk.v1': {
                let obj = null;
                try {
                    const stream = context.stream;
                    const buffer = stream.peek();
                    obj = new cntk_v1.ComputationNetwork(buffer);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new cntk.Error('File format is not CNTK v1 (' + message.replace(/\.$/, '') + ').');
                }
                return new cntk.Model(metadata, 1, obj);
            }
            case 'cntk.v2': {
                await context.require('./cntk-proto');
                let obj = null;
                try {
                    cntk_v2 = protobuf.get('cntk').CNTK.proto;
                    cntk_v2.PoolingType = { 0: 'Max', 1: 'Average' };
                    const stream = context.stream;
                    const reader = protobuf.BinaryReader.open(stream);
                    const dictionary = cntk_v2.Dictionary.decode(reader);
                    obj = cntk.ModelFactory._convertDictionary(dictionary);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new cntk.Error('File format is not cntk.Dictionary (' + message.replace(/\.$/, '') + ').');
                }
                return new cntk.Model(metadata, 2, obj);
            }
            default: {
                throw new cntk.Error("Unsupported CNTK format '" + target + "'.");
            }
        }
    }

    static _convertDictionary(dictionary) {
        const target = {};
        for (const key of Object.keys(dictionary.data).filter((key) => key != 'version')) {
            target[key] = cntk.ModelFactory._convertDictionaryValue(dictionary.data[key]);
        }
        return target;
    }

    static _convertDictionaryValue(dictionaryValue) {
        switch (dictionaryValue.value_type) {
            case cntk_v2.DictionaryValue.Type.Bool:
                return dictionaryValue.bool_value;
            case cntk_v2.DictionaryValue.Type.Int:
                return dictionaryValue.int_value;
            case cntk_v2.DictionaryValue.Type.SizeT:
                return dictionaryValue.size_t_value;
            case cntk_v2.DictionaryValue.Type.Float:
                return dictionaryValue.float_value;
            case cntk_v2.DictionaryValue.Type.Double:
                return dictionaryValue.double_value;
            case cntk_v2.DictionaryValue.Type.String:
                return dictionaryValue.string_value;
            case cntk_v2.DictionaryValue.Type.Vector:
                return cntk.ModelFactory._convertVectorValue(dictionaryValue.vector_value);
            case cntk_v2.DictionaryValue.Type.NDShape:
                return dictionaryValue.nd_shape_value;
            case cntk_v2.DictionaryValue.Type.Axis:
                return dictionaryValue.axis_value;
            case cntk_v2.DictionaryValue.Type.Dictionary:
                return cntk.ModelFactory._convertDictionary(dictionaryValue.dictionary_value);
            case cntk_v2.DictionaryValue.Type.NDArrayView:
                return dictionaryValue.nd_array_view_value;
            default:
                throw new cntk.Error("Unsupported dictionary value type '" + dictionaryValue.value_type.toString() + "'.");
        }
    }

    static _convertVectorValue(vectorValue) {
        return vectorValue.value.map((item) => {
            return cntk.ModelFactory._convertDictionaryValue(item);
        });
    }
};

cntk.Model = class {

    constructor(metadata, version, obj) {
        switch (version) {
            case 1:
                this._format = 'CNTK v1' + (obj.version ? ('.' + obj.version.toString()) : '');
                break;
            case 2:
                this._format = 'CNTK v2';
                break;
            default:
                throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
        }
        this._graphs = [];
        this._graphs.push(new cntk.Graph(metadata, version, obj));
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return this._format;
    }
};

cntk.Graph = class {

    constructor(metadata, version, obj) {
        metadata = new cntk.GraphMetadata(metadata);
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const args = new Map();
        const arg = (name, version, obj) => {
            if (obj && args.has(name)) {
                throw new cntk.Error("Duplicate value '" + name + "'.");
            }
            if (!args.has(name)) {
                switch (version) {
                    case 1:
                        args.set(name, new cntk.Value(version, obj ? obj : { name: name }));
                        break;
                    case 2:
                        args.set(name, new cntk.Value(version, obj ? obj : { uid: name }));
                        break;
                    default:
                        throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
                }
            }
            return args.get(name);
        };
        switch (version) {
            case 1: {
                for (const name of Object.keys(obj.nodes)) {
                    const node = obj.nodes[name];
                    switch (node.__type__) {
                        case 'InputValue':
                            this._inputs.push(new cntk.Argument(node.name, [ arg(node.name, version, node) ]));
                            break;
                        case 'LearnableParameter':
                            arg(node.name, version, node);
                            break;
                        default:
                            break;
                    }
                }
                for (const name of Object.keys(obj.nodes)) {
                    const node = obj.nodes[name];
                    if (node.__type__ != 'InputValue' && node.__type__ != 'LearnableParameter') {
                        this._nodes.push(new cntk.Node(metadata, version, node, arg));
                    }
                }
                if (obj.output) {
                    for (const output of obj.output) {
                        this._outputs.push(new cntk.Argument(output, [ arg(output, version) ]));
                    }
                }
                break;
            }
            case 2: {
                const map = new Map(obj.primitive_functions.map((node) => [ node.uid, node ]));
                for (const input of obj.inputs) {
                    const value = arg(input.uid, version, input);
                    // VariableKind { 0: 'input', 1: 'output', 2: 'parameter', 3: 'constant', 4: 'placeholder' }
                    if (input.kind == 0) {
                        const inputName = input.name || input.uid;
                        this._inputs.push(new cntk.Argument(inputName, [ value ]));
                    }
                }
                for (const block of obj.primitive_functions) {
                    if (block.op == 57 && block.block_function_composite) {
                        const list = [ block.block_function_composite.root ];
                        const output = map.get(block.block_function_composite.root);
                        const keys = block.block_function_composite_arguments_map_keys;
                        const values = block.block_function_composite_arguments_map_values;
                        block.inputs = values;
                        if (!Array.isArray(keys) || !Array.isArray(values) || keys.length !== values.length) {
                            throw new cntk.Error('Invalid block function composite arguments.');
                        }
                        const inputs = keys.map((key) => new cntk.Argument(key, [ arg(key, version) ]));
                        const outputs = [ new cntk.Argument('output', [ arg(output.uid + '_Output_0', version) ]) ];
                        const nodes = [];
                        while (list.length > 0) {
                            const name = list.shift();
                            if (map.has(name)) {
                                const node = map.get(name);
                                nodes.push(new cntk.Node(metadata, version, node, arg));
                                map.delete(name);
                                for (let i = 0; i < node.inputs.length; i++) {
                                    const parts = node.inputs[i].split('_');
                                    if (parts.length >= 3) {
                                        parts.pop();
                                        if (parts.pop() == 'Output') {
                                            list.push(parts.join('_'));
                                        }
                                    }
                                }
                            }
                        }
                        const func = new cntk.Function(block.block_function_op_name, nodes, inputs, outputs);
                        metadata.add(block.uid, func);
                    }
                }
                for (const node of map.values()) {
                    this._nodes.push(new cntk.Node(metadata, version, node, arg));
                }
                break;
            }
            default: {
                throw new cntk.Error("Unsupported graph version '" + version + "'.");
            }
        }
    }

    get nodes() {
        return this._nodes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }
};

cntk.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

cntk.Value = class {

    constructor(version, obj) {
        switch (version) {
            case 1:
                switch (obj.__type__) {
                    case 'InputValue':
                        this._name = obj.name;
                        this._type = new cntk.TensorType(version, obj.precision, obj.sampleLayout);
                        this._initializer = null;
                        break;
                    case 'LearnableParameter':
                        this._name = obj.name;
                        this._type = null;
                        this._initializer = new cntk.Tensor(version, obj);
                        break;
                    default:
                        this._name = obj.name;
                        this._type = null;
                        this._initializer = null;
                        break;
                }
                break;
            case 2:
                if (obj.value) {
                    this._name = obj.name || obj.uid;
                    this._type = null;
                    this._initializer = new cntk.Tensor(version, obj);
                } else {
                    this._name = obj.uid;
                    if (obj.data_type && obj.shape) {
                        this._type = new cntk.TensorType(version, obj.data_type, obj.shape);
                    }
                    this._initializer = null;
                }
                break;
            default:
                throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._type) {
            return this._type;
        }
        if (this._initializer) {
            return this._initializer.type;
        }
        return null;
    }

    get description() {
        return '';
    }

    get initializer() {
        return this._initializer;
    }
};

cntk.Node = class {

    constructor(metadata, version, obj, arg) {
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        let inputs = [];
        let outputs = [];
        switch (version) {
            case 1: {
                const type = obj.__type__;
                this._type = metadata.type(type) || { name: type };
                this._name = obj.name;
                for (const entry of Object.entries(obj)) {
                    const name = entry[0];
                    const value = entry[1];
                    if (name != '__type__' && name != 'name' && name != 'inputs' && name != 'precision') {
                        this._attributes.push(new cntk.Attribute(metadata.attribute(type, name), name, value));
                    }
                }
                inputs = obj.inputs.map((input) => arg(input, version));
                outputs = [ arg(this._name, version) ];
                break;
            }
            case 2: {
                this._name = obj.name || obj.uid || null;
                const output = obj.uid;
                if (obj.op == 57) {
                    this._type = metadata.type(obj.uid) || { name: obj.uid };
                } else if (Object.prototype.hasOwnProperty.call(obj, 'op')) {
                    // cntk/Source/CNTKv2LibraryDll/API/Internals/PrimitiveOpType.h
                    this._type = metadata.type(obj.op.toNumber());
                } else {
                    const type = obj.type;
                    this._type = metadata.type(type) || { name: type };
                    if (obj.user_defined_state) {
                        for (const attributeName of Object.keys(obj.user_defined_state)) {
                            this._attributes.push(new cntk.Attribute(metadata.attribute(type, attributeName), attributeName, obj.user_defined_state[attributeName]));
                        }
                    }
                }
                if (obj.attributes) {
                    for (const entry of Object.entries(obj.attributes)) {
                        this._attributes.push(new cntk.Attribute(metadata.attribute(this._type, entry[0]), entry[0], entry[1]));
                    }
                }
                inputs = obj.inputs.map((input) => arg(input, version));
                outputs.push(arg(output + '_Output_0', version));
                break;
            }
            default: {
                throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
            }
        }
        let inputIndex = 0;
        if (this._type && this._type.inputs) {
            for (const inputSchema of this._type.inputs) {
                if (inputIndex < inputs.length || inputSchema.option != 'optional') {
                    const inputCount = inputSchema.type === 'Tensor[]' ? (inputs.length - inputIndex) : 1;
                    const inputArguments = [];
                    for (const inputArgument of inputs.slice(inputIndex, inputIndex + inputCount)) {
                        if (inputArgument.name != '' || inputSchema.option != 'optional') {
                            inputArguments.push(inputArgument);
                        }
                    }
                    this._inputs.push(new cntk.Argument(inputSchema.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        this._inputs.push(...inputs.slice(inputIndex).map((argument, index) => {
            return new cntk.Argument((inputIndex + index).toString(), [ argument ]);
        }));

        let outputIndex = 0;
        if (this._type && this._type.outputs) {
            for (const outputSchema of this._type.outputs) {
                if (outputIndex < outputs.length || !outputSchema.optional) {
                    const outputCount = outputSchema.type === 'Tensor[]' ? (outputs.length - outputIndex) : 1;
                    this._outputs.push(new cntk.Argument(outputSchema.name, outputs.slice(outputIndex, outputIndex + outputCount)));
                    outputIndex += outputCount;
                }
            }
        }
        this._outputs.push(...outputs.slice(outputIndex).map((argument) => {
            return new cntk.Argument(outputIndex.toString(), [ argument ]);
        }));
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }
};

cntk.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        this._type = null;
        if (cntk_v1 && this._value instanceof cntk_v1.TensorShape) {
            this._value = new cntk.TensorShape(1, value);
            this._type = 'shape';
        }
        if (cntk_v2 && this._value instanceof cntk_v2.NDShape) {
            this._value = new cntk.TensorShape(2, value);
            this._type = 'shape';
        }
        if (cntk_v2 && this._value instanceof cntk_v2.Axis) {
            const axis = { __type__: 'Axis' };
            for (const key of Object.keys(value).filter((key) => key !== 'name')) {
                axis[key] = value[key];
            }
            this._value = axis;
        }
        if (metadata) {
            if (metadata.type) {
                this._type = metadata.type;
                const type = cntk_v1[this._type] || cntk_v2[this._type];
                if (type && type[this._value]) {
                    this._value = type[this._value];
                }
            }
            if (metadata.visible === false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                let defaultValue = metadata.default;
                value = this._value;
                if (typeof value == 'function') {
                    value = value();
                }
                if (this._type == 'shape') {
                    value = value.dimensions;
                }
                if (value == defaultValue) {
                    this._visible = false;
                } else if (Array.isArray(value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]);
                        }
                    }
                    if (value.every((item, index) => item == defaultValue[index])) {
                        this._visible = false;
                    }
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

cntk.Tensor = class {

    constructor(version, tensor) {
        switch (version) {
            case 1:
                if (tensor.__type__ == 'LearnableParameter') {
                    this._name = tensor.name || null;
                    this._type = new cntk.TensorType(version, tensor.precision, tensor.sampleLayout);
                }
                break;
            case 2:
                this._name = tensor.name || tensor.uid || null;
                this._type = new cntk.TensorType(version, tensor.data_type, tensor.shape);
                this._value = tensor.value;
                break;
            default:
                throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get layout() {
        return '|';
    }

    get values() {
        if (this._type.dataType === 'float32' &&
            this._value &&
            this._value.float_values &&
            this._value.float_values.value &&
            this._value.float_values.value.length > 0) {
            return this._value.float_values.value;
        }
        return null;
    }
};

cntk.TensorType = class {

    constructor(version, dataType, shape) {
        this._dataType = '?';
        switch (version) {
            case 1:
                switch (dataType) {
                    case 'float': this._dataType = 'float32'; break;
                    case 'double': this._dataType = 'float64'; break;
                    case 'half': this._dataType = 'float16'; break;
                    case '': this._dataType = 'float32'; break;
                    default: throw new cntk.Error("Unsupported tensor data type '" + dataType + "'.");
                }
                this._shape = new cntk.TensorShape(version, shape);
                break;
            case 2:
                dataType = dataType.toNumber();
                switch (dataType) {
                    case 1: this._dataType = 'float32'; break;
                    default: throw new cntk.Error("Unsupported tensor data type '" + dataType + "'.");
                }
                this._shape = new cntk.TensorShape(version, shape);
                break;
            default:
                throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
        }
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

cntk.TensorShape = class {

    constructor(version, shape) {
        switch (version) {
            case 1:
                this._dimensions = shape.dims;
                break;
            case 2:
                this._dimensions = shape.shape_dim.map((dimension) => dimension.toNumber());
                break;
            default:
                throw new cntk.Error("Unsupported CNTK version '" + version + "'.");
        }
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return (this._dimensions && this._dimensions.length) ? ('[' + this._dimensions.join(',') + ']') : '';
    }
};

cntk.Function = class {

    constructor(name, nodes, inputs, outputs) {
        this._name = name;
        this._inputs = inputs;
        this._outputs = outputs;
        this._nodes = nodes;
    }

    get type() {
        return 'function';
    }

    get name() {
        return this._name;
    }

    get category() {
        switch (this._name) {
            case 'PReLU':
            case 'Softmax':
                return 'Activation';
            case 'Dropout':
                return 'Dropout';
            case 'Convolution':
            case 'ConvolutionTranspose':
            case 'Dense':
            case 'linear':
            case 'LSTM':
                return 'Layer';
            case 'BatchNormalization':
            case 'lrn':
                return 'Normalization';
            case 'AveragePooling':
            case 'MaxPooling':
                return 'Pool';
            default:
                return null;
        }
    }

    get description() {
        return '';
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

cntk.GraphMetadata = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._functions = new Map();
        this._attributes = new Map();
    }

    add(name, func) {
        if (this._functions.has(name)) {
            throw new cntk.Error("Duplicate function identifier '" + func.name + "'.");
        }
        this._functions.set(name, func);
    }

    name(code) {
        // cntk/Source/CNTKv2LibraryDll/API/Internals/PrimitiveOpType.h
        return this._metadata.name(code);
    }

    type(name) {
        if (this._functions.has(name)) {
            return this._functions.get(name);
        }
        return this._metadata.type(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributes.has(key)) {
            const metadata = this.type(type);
            if (metadata && metadata.attributes && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    this._attributes.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }
};

cntk_v1.ComputationNetwork = class {

    constructor(buffer) {
        const reader = new cntk_v1.BinaryReader(buffer);
        reader.assert('BCN');
        reader.assert('BVersion');
        this.version = reader.uint64();
        reader.assert('EVersion');
        const numNodes = reader.uint64();
        reader.assert('BNodeList');
        const op = {};
        op.Minus = function() {};
        op.Plus = function() {};
        op.GreaterEqual = function() {};
        op.Equal = function() {};
        op.NotEqual = function() {};
        op.GreaterEqual = function() {};
        op.Exp = function() {};
        op.Log = function() {};
        op.Reciprocal = function() {};
        op.ElementTimes = function() {};
        op.ClassificationError = function() {};
        op.RectifiedLinear = function() {};
        op.InputValue = function(reader, version) {
            this.rows = reader.uint64();
            this.cols = reader.uint64();
            this.sampleLayout = new cntk_v1.TensorShape(reader, true);
            this.dynamicAxisNodeName = '';
            if (version >= 8) {
                const nrAxes = reader.uint32();
                if (nrAxes == 1) {
                    this.dynamicAxisNodeName = reader.string();
                }
            }
            this.learningRateMultiplier = 0;
            if (version >= 10) {
                this.learningRateMultiplier = reader.float32();
            }
        };
        op.LearnableParameter = function(reader, version) {
            if (version >= 3) {
                this.learningRateMultiplier = reader.float32();
                this.sampleLayout = new cntk_v1.TensorShape(reader);
            } else {
                throw new cntk.Error('LeanableParameter reader implemented.');
            }
            this.value = new cntk_v1.Matrix(reader);
        };
        op.CrossEntropyWithSoftmax = function(reader) {
            this.evalMode = reader.uint32();
            if (this.evalMode > 2) {
                this.evalMode = 0;
                reader.skip(-4);
            }
        };
        op.Times = function(reader, version) {
            this.outputRank = (version >= 3) ? reader.uint64() : 1;
            this.inferInputRankToMap = (version >= 12) ? reader.int32() : -1;
        };
        op.Dropout = function(reader, version) {
            if (version >= 16) {
                this.rngSeed = (version == 16) ? reader.uint32() : reader.uint64();
                this.rngOffset = reader.uint64();
            }
        };
        op.ConvolutionBase = function(reader, version) {
            if (version >= 5) {
                this.kernelShape = new cntk_v1.TensorShape(reader);
                this.mapCount = new cntk_v1.TensorShape(reader);
                this.strides = new cntk_v1.TensorShape(reader);
                this.sharing = reader.booleans();
                this.autoPadding = reader.booleans();
                this.lowerPad = new cntk_v1.TensorShape(reader);
                this.upperPad = new cntk_v1.TensorShape(reader);
                this.poolKind = reader.int32();
                this.imageLayoutKind = reader.int32();
                this.maxTempMemSizeInSamples = reader.uint64();
            }
            if (version >= 9) {
                this.transpose = reader.boolean();
            }
            if (version >= 20) {
                this.outputShape = new cntk_v1.TensorShape(reader);
            }
            if (version >= 21) {
                this.ceilOutDim = reader.boolean();
            }
            if (version >= 23) {
                this.includePad = reader.boolean();
            }
        };
        op.Convolution = function(reader, version) {
            op.ConvolutionBase.apply(this, [ reader, version ]);
            if (version < 5) {
                this.kernelShape = new cntk_v1.TensorShape([ reader.uint64(), reader.uint64(), 1 ]);
                this.strides = new cntk_v1.TensorShape([ reader.uint64(), reader.uint64(), 1 ]);
                this.mapCount = new cntk_v1.TensorShape([ reader.uint32() ]);
                this.imageLayoutKind = reader.int32();
                this.autoPadding = [ reader.boolean() ];
                this.maxTempMemSizeInSamples = reader.uint64();
                this.poolKind = 'None';
                this.convolution2D = true;
                this.sharing = [ true ];
                this.lowerPad = new cntk_v1.TensorShape([ 0 ]);
                this.upperPad = new cntk_v1.TensorShape([ 0 ]);
            } else {
                this.convolution2D = reader.boolean();
                if (version >= 18) {
                    this.dilation = new cntk_v1.TensorShape(reader);
                } else {
                    this.dilation = new cntk_v1.TensorShape([ 1 ]);
                }
            }
        };
        op.Pooling = function(reader, version) {
            op.ConvolutionBase.apply(this, [ reader, version ]);
        };
        op.PoolingBase = function(reader) {
            this.imageLayoutKind = reader.int32();
            this.windowWidth = reader.uint32();
            this.windowHeight = reader.uint64();
            this.horizontalSubsample = reader.uint64();
            this.verticalSubsample = reader.uint64();
        };
        op.MaxPooling = function(reader, version) {
            op.PoolingBase.apply(this, [ reader, version ]);
        };
        op.ROIPooling = function(reader, version) {
            this.roiOutputShape = new cntk_v1.TensorShape(reader);
            this.poolKind = (version < 26) ? 'Max' : reader.int32();
            this.spatialScale = (version < 26) ? 0.0625 : reader.float64();
        };
        op.Reshape = function(reader) {
            this.beginDimParameter = reader.uint32();
            this.endDimParameter = reader.uint32();
            this.replacementSampleLayout = new cntk_v1.TensorShape(reader);
        };
        op.ReduceElements = function(reader, version) {
            let num_axes = 1;
            if (version >= 27) {
                num_axes = reader.uint32();
            }
            this.axes = [];
            for (let i = 0; i < num_axes; i++) {
                this.axes.push(reader.uint32());
            }
            this.operation = reader.string();
            if (version >= 24) {
                this.keepDimensions = reader.boolean();
            }
        };
        op.BatchNormalization = function(reader, version) {
            let mbCount = 0;
            if (version >= 6) {
                this.spatial = reader.boolean();
                this.normalizationTimeConstant = reader.float64();
                this.blendTimeConstant = reader.float64();
                this.imageLayoutKind = reader.int32();
                if (version >= 13) {
                    if (version != 19) {
                        this.runCountUntied = reader.uint64();
                    } else {
                        this.runCountUntied = reader.boolean() ? 0 : 'SIZE_MAX'; // TODO
                    }
                } else {
                    mbCount = reader.uint64();
                }
                this.epsilon = reader.float64();
                this.useCntkEngine = reader.boolean();
            } else {
                const verWritten = reader.int32();
                const verReadable = reader.int32();
                if (verReadable > verWritten || verWritten < 0x00010001 || verReadable > 0x00010004) {
                    throw new cntk.Error('BatchNormalization version not supported.');
                }
                this.eval = reader.boolean();
                this.spatial = reader.boolean();
                if (verWritten >= 0x00010004) {
                    this.normalizationTimeConstant = reader.float64();
                } else {
                    reader.float64(); // expAvgFactor
                }
                if (verWritten >= 0x00010002) {
                    this.imageLayoutKind = reader.int32();
                    mbCount = reader.uint64();
                }
                if (verWritten >= 0x00010003) {
                    this.epsilon = reader.float64();
                    this.useCntkEngine = reader.boolean();
                }
            }
            if (version < 13) {
                this.runCountUntied = 16 * mbCount;
                this.convertRunningVariancePending = true;
            }
        };
        op.Tanh = function() {};
        op.Sigmoid = function() {};
        op.Logistic = function() {};
        op.SquareError = function() {};
        op.ErrorPrediction = function() {};
        op.RowStack = function(reader, version) {
            this.spliceDim = (version >= 3) ? reader.int32() : 1;
        };
        op.Slice = function(reader, version) {
            let num = 1;
            if (version >= 22) {
                num = reader.int32();
            }
            this.index = [];
            this.axis = [];
            this.strideMultiplier = [];
            for (let i = 0; i < num; i++) {
                this.index.push([ [ reader.uint64(), reader.uint64() ] ]);
                if (version >= 3) {
                    this.axis.push(reader.int32());
                }
                if (version >= 27) {
                    this.strideMultiplier.push(reader.int32());
                }
            }
        };
        op.PastValue = function(reader, version) {
            this.timeStep = reader.int32();
            if (version > 3) {
                this.sampleLayout = new cntk_v1.TensorShape(reader, false);
            } else {
                const rows = reader.uint64();
                reader.uint64();
                this.sampleLayout = new cntk_v1.TensorShape([ rows ], true);
            }
            if (version >= 2) {
                this.initialStateValue = reader.int32();
            }
        };
        op.FutureValue = function(reader, version) {
            this.timeStep = reader.int32();
            if (version > 3) {
                this.sampleLayout = new cntk_v1.TensorShape(reader, false);
            } else {
                const rows = reader.uint64();
                reader.uint64();
                this.sampleLayout = new cntk_v1.TensorShape([ rows ], true);
            }
            if (version >= 2) {
                this.initialStateValue = reader.int32();
            }
        };
        op.TransposeDimensions = function(reader, version) {
            if (version >= 3) {
                this.axis1 = reader.int32();
                this.axis2 = reader.int32();
                if (version >= 25 && this.axis1 == 0 && this.axis2 == 0) {
                    const size = reader.uint64();
                    this.perm = [];
                    for (let i = 0; i < size; i++) {
                        this.perm.push(reader.uint64());
                    }
                }
            } else {
                this.axis1 = 1;
                this.axis2 = 2;
            }
        };
        op.AveragePooling = function(reader, version) {
            op.PoolingBase.apply(this, [ reader, version ]);
        };
        op.InvStdDev = function(reader) {
            this.hasComputed = reader.boolean();
            this.value = new cntk_v1.Matrix(reader);
        };
        op.Mean = function(reader) {
            this.hasComputed = reader.boolean();
            this.value = new cntk_v1.Matrix(reader);
        };
        op.PerDimMeanVarNormalization = function() {};
        op.Softmax = function() {};
        op.DynamicAxis = function() {};

        const nodes = [];
        this.nodes = {};
        for (let i = 0; i < numNodes; i++) {
            const precision = this.version >= 7 ? reader.string() : '';
            if (precision != 'float' && precision != 'double' && precision != 'half' && precision != '') {
                throw new cntk.Error("Invalid precision format '" + precision + "'.");
            }
            const obj = { __type__: reader.string() };
            obj.name = reader.string();
            obj.precision = precision;
            const constructor = op[obj.__type__];
            if (!constructor) {
                throw new cntk.Error("Unsupported node type '" + obj.__type__ + "'.");
            }
            constructor.apply(obj, [ reader, this.version ]);
            nodes.push(obj);
            this.nodes[obj.name] = obj;
        }
        reader.assert('ENodeList');
        reader.assert('BRelation');
        for (let j = 0; j < numNodes; j++) {
            const nodeName = reader.string();
            const node = this.nodes[nodeName];
            const numChildren = reader.uint64();
            const children = [];
            for (let k = 0; k < numChildren; k++) {
                children.push(reader.string());
            }
            if (this.version < 19 && node.__type__ == 'BatchNormalization') {
                const runSampleCount = {
                    __type__: 'LearnableParameter',
                    name: nodeName + '.run_sample_count',
                    precision: node.precision,
                    sampleLayout: new cntk_v1.TensorShape([ 1 ]), // TODO set value = 0
                    learningRateMultiplier: 0
                };
                nodes.push(runSampleCount);
                this.nodes[runSampleCount.name] = runSampleCount;
                children.push(runSampleCount.name);
            }
            if (node.__type__ == 'Convolution' && children.length > 1) {
                children.splice(0, 0, children.pop());
            }
            node.inputs = children;
        }
        reader.assert('ERelation');
        reader.assert('BRootNodes');
        if (reader.match('BFeatureNodes')) {
            this.feature = reader.strings();
            reader.assert('EFeatureNodes');
        }
        if (reader.match('BLabelNodes')) {
            this.label = reader.strings();
            reader.assert('ELabelNodes');
        }
        if (reader.match('BCriterionNodes')) {
            this.criterion = reader.strings();
            reader.assert('ECriterionNodes');
        }
        if (this.criterion.length == 0) {
            if (reader.match('BCriteriaNodes')) {
                this.criterion = reader.strings();
                reader.assert('ECriteriaNodes');
            }
        }
        if (reader.match('BNodesReqMultiSeqHandling')) {
            reader.strings();
            reader.assert('ENodesReqMultiSeqHandling');
        }
        if (reader.match('BEvalNodes')) {
            this.eval = reader.strings();
            reader.assert('EEvalNodes');
        }
        if (reader.match('BOutputNodes')) {
            this.output = reader.strings();
            reader.assert('EOutputNodes');
        }
        if (reader.match('BPairNodes')) {
            this.pair = reader.strings();
            reader.assert('EPairNodes');
        }
        reader.assert('ERootNodes');
        reader.assert('ECN');
    }
};

cntk_v1.BinaryReader = class extends base.BinaryReader {

    match(text) {
        const position = this.position;
        for (let i = 0; i < text.length; i++) {
            if (this.uint16() != text.charCodeAt(i)) {
                this.seek(position);
                return false;
            }
        }
        if (this.uint16() != 0) {
            this.seek(position);
            return false;
        }
        return true;
    }

    assert(text) {
        if (!this.match(text)) {
            throw new cntk_v1.Error("Invalid '" + text + "' signature.");
        }
    }

    string() {
        const content = [];
        let c = this.uint16();
        while (c != 0) {
            content.push(String.fromCharCode(c));
            c = this.uint16();
        }
        return content.join('');
    }

    strings() {
        const count = this.uint64();
        const array = new Array(count);
        for (let i = 0; i < count; i++) {
            array[i] = this.string();
        }
        return array;
    }

    booleans() {
        const count = this.uint64();
        const array = new Array(count);
        for (let i = 0; i < count; i++) {
            array[i] = this.boolean();
        }
        return array;
    }
};

cntk_v1.TensorShape = class {

    constructor(reader, acceptLegacyFormat = false) {
        if (reader && Array.isArray(reader)) {
            this.dims = reader;
            return;
        }
        this.dims = [];
        const rank = reader.uint32();
        let dim0 = 0;
        if (rank > 0) {
            dim0 = reader.uint32();
        }
        if (!acceptLegacyFormat || dim0 != 0) {
            if (rank > 0) {
                this.dims.push(dim0);
            }
            for (let i = 1; i < rank; i++) {
                this.dims.push(reader.uint32());
            }
        } else {
            const dim = reader.uint32();
            this.dims.push(reader.uint32());
            this.dims.push(rank);
            this.dims.push(dim);
        }
    }
};

cntk_v1.Matrix = class {

    constructor(reader) {
        const type = reader.byte();
        switch (type) {
            case 100: {
                // dense
                reader.assert('BMAT');
                const elsize = reader.uint64();
                this.name = reader.string();
                this.format = reader.uint32();
                this.rows = reader.uint64();
                this.columns = reader.uint64();
                reader.read(elsize * this.rows * this.columns);
                reader.assert('EMAT');
                break;
            }
            case 115: // sparse
                throw new cntk_v1.Error('Matrix sparse type not implemented.');
            default:
                throw new cntk_v1.Error("Matrix type '" + type.toString() + "' not implemented.");
        }
    }
};

cntk_v1.ImageLayoutKind = {
    0: 'CHW',
    1: 'HWC'
};

cntk_v1.PoolKind = {
    0: 'None',
    1: 'Max',
    2: 'Average'
};

cntk_v1.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading CNTK v1 model.';
    }
};

cntk.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading CNTK model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = cntk.ModelFactory;
}