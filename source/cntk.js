
const cntk = {};

cntk.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        // CNTK v1
        const signature = [0x42, 0x00, 0x43, 0x00, 0x4e, 0x00, 0x00, 0x00];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            context.type = 'cntk.v1';
            return;
        }
        // CNTK v2
        const tags = context.tags('pb');
        if (tags.get(1) === 0 && tags.get(2) === 2) {
            context.type = 'cntk.v2';
        }
    }

    async open(context) {
        const metadata = await context.metadata('cntk-metadata.json');
        switch (context.type) {
            case 'cntk.v1': {
                let obj = null;
                try {
                    const reader = context.read('binary');
                    obj = new cntk.ComputationNetwork(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new cntk.Error(`File format is not CNTK v1 (${message.replace(/\.$/, '')}).`);
                }
                return new cntk.Model(metadata, 1, obj);
            }
            case 'cntk.v2': {
                cntk.proto = await context.require('./cntk-proto');
                cntk.proto = cntk.proto.CNTK.proto;
                cntk.proto.PoolingType = { 0: 'Max', 1: 'Average' };
                let obj = null;
                try {
                    const reader = context.read('protobuf.binary');
                    const dictionary = cntk.proto.Dictionary.decode(reader);
                    obj = cntk.ModelFactory._convertDictionary(dictionary);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new cntk.Error(`File format is not cntk.Dictionary (${message.replace(/\.$/, '')}).`);
                }
                return new cntk.Model(metadata, 2, obj);
            }
            default: {
                throw new cntk.Error(`Unsupported CNTK format '${context.type}'.`);
            }
        }
    }

    static _convertDictionary(dictionary) {
        const target = {};
        for (const key of Object.keys(dictionary.data).filter((key) => key !== 'version')) {
            target[key] = cntk.ModelFactory._convertDictionaryValue(dictionary.data[key]);
        }
        return target;
    }

    static _convertDictionaryValue(dictionaryValue) {
        switch (dictionaryValue.value_type) {
            case cntk.proto.DictionaryValue.Type.Bool:
                return dictionaryValue.bool_value;
            case cntk.proto.DictionaryValue.Type.Int:
                return dictionaryValue.int_value;
            case cntk.proto.DictionaryValue.Type.SizeT:
                return dictionaryValue.size_t_value;
            case cntk.proto.DictionaryValue.Type.Float:
                return dictionaryValue.float_value;
            case cntk.proto.DictionaryValue.Type.Double:
                return dictionaryValue.double_value;
            case cntk.proto.DictionaryValue.Type.String:
                return dictionaryValue.string_value;
            case cntk.proto.DictionaryValue.Type.Vector:
                return cntk.ModelFactory._convertVectorValue(dictionaryValue.vector_value);
            case cntk.proto.DictionaryValue.Type.NDShape:
                return dictionaryValue.nd_shape_value;
            case cntk.proto.DictionaryValue.Type.Axis:
                return dictionaryValue.axis_value;
            case cntk.proto.DictionaryValue.Type.Dictionary:
                return cntk.ModelFactory._convertDictionary(dictionaryValue.dictionary_value);
            case cntk.proto.DictionaryValue.Type.NDArrayView:
                return dictionaryValue.nd_array_view_value;
            default:
                throw new cntk.Error(`Unsupported dictionary value type '${dictionaryValue.value_type}'.`);
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
                this.format = `CNTK v1${obj.version ? (`.${obj.version}`) : ''}`;
                break;
            case 2:
                this.format = 'CNTK v2';
                break;
            default:
                throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
        }
        this.graphs = [new cntk.Graph(metadata, version, obj)];
    }
};

cntk.Graph = class {

    constructor(metadata, version, obj) {
        metadata = new cntk.GraphMetadata(metadata);
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, version, obj) => {
            if (obj && values.has(name)) {
                throw new cntk.Error(`Duplicate value '${name}'.`);
            }
            if (!values.has(name)) {
                switch (version) {
                    case 1:
                        values.set(name, new cntk.Value(version, obj ? obj : { name }));
                        break;
                    case 2:
                        values.set(name, new cntk.Value(version, obj ? obj : { uid: name }));
                        break;
                    default:
                        throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
                }
            }
            return values.get(name);
        };
        switch (version) {
            case 1: {
                for (const name of Object.keys(obj.nodes)) {
                    const node = obj.nodes[name];
                    switch (node.__type__) {
                        case 'InputValue': {
                            const argument = new cntk.Argument(node.name, [values.map(node.name, version, node)]);
                            this.inputs.push(argument);
                            break;
                        }
                        case 'LearnableParameter': {
                            values.map(node.name, version, node);
                            break;
                        }
                        default:
                            break;
                    }
                }
                for (const name of Object.keys(obj.nodes)) {
                    const node = obj.nodes[name];
                    if (node.__type__ !== 'InputValue' && node.__type__ !== 'LearnableParameter') {
                        this.nodes.push(new cntk.Node(metadata, version, node, values));
                    }
                }
                if (obj.output) {
                    for (const output of obj.output) {
                        const argument = new cntk.Argument(output, [values.map(output, version)]);
                        this.outputs.push(argument);
                    }
                }
                break;
            }
            case 2: {
                const map = new Map(obj.primitive_functions.map((node) => [node.uid, node]));
                for (const input of obj.inputs) {
                    const value = values.map(input.uid, version, input);
                    // VariableKind { 0: 'input', 1: 'output', 2: 'parameter', 3: 'constant', 4: 'placeholder' }
                    if (input.kind === 0) {
                        const inputName = input.name || input.uid;
                        this.inputs.push(new cntk.Argument(inputName, [value]));
                    }
                }
                for (const block of obj.primitive_functions) {
                    if (block.op === 57 && block.block_function_composite) {
                        const list = [block.block_function_composite.root];
                        const output = map.get(block.block_function_composite.root);
                        const keys = block.block_function_composite_arguments_map_keys;
                        const args = block.block_function_composite_arguments_map_values;
                        block.inputs = args;
                        if (!Array.isArray(keys) || !Array.isArray(args) || keys.length !== args.length) {
                            throw new cntk.Error('Invalid block function composite arguments.');
                        }
                        const inputs = keys.map((key) => new cntk.Argument(key, [values.map(key, version)]));
                        const outputs = [new cntk.Argument('output', [values.map(`${output.uid}_Output_0`, version)])];
                        const nodes = [];
                        while (list.length > 0) {
                            const name = list.shift();
                            if (map.has(name)) {
                                const node = map.get(name);
                                nodes.push(new cntk.Node(metadata, version, node, values));
                                map.delete(name);
                                for (let i = 0; i < node.inputs.length; i++) {
                                    const parts = node.inputs[i].split('_');
                                    if (parts.length >= 3) {
                                        parts.pop();
                                        if (parts.pop() === 'Output') {
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
                    this.nodes.push(new cntk.Node(metadata, version, node, values));
                }
                break;
            }
            default: {
                throw new cntk.Error(`Unsupported graph version '${version}'.`);
            }
        }
    }
};

cntk.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

cntk.Value = class {

    constructor(version, obj) {
        switch (version) {
            case 1:
                switch (obj.__type__) {
                    case 'InputValue':
                        this.name = obj.name;
                        this.type = new cntk.TensorType(version, obj.precision, obj.sampleLayout);
                        this.initializer = null;
                        break;
                    case 'LearnableParameter':
                        this.name = obj.name;
                        this.initializer = new cntk.Tensor(version, obj);
                        this.type = this.initializer.type;
                        break;
                    default:
                        this.name = obj.name;
                        this.type = null;
                        this.initializer = null;
                        break;
                }
                break;
            case 2:
                if (obj.value) {
                    this.name = obj.name || obj.uid;
                    this.type = null;
                    this.initializer = new cntk.Tensor(version, obj);
                } else {
                    this.name = obj.uid;
                    if (obj.data_type && obj.shape) {
                        this.type = new cntk.TensorType(version, obj.data_type, obj.shape);
                    }
                    this.initializer = null;
                }
                break;
            default:
                throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
        }
    }
};

cntk.Node = class {

    constructor(metadata, version, obj, values) {
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        let inputs = [];
        let outputs = [];
        switch (version) {
            case 1: {
                const type = obj.__type__;
                this.type = metadata.type(type) || { name: type };
                this.name = obj.name;
                for (const [name, value] of Object.entries(obj)) {
                    if (name !== '__type__' && name !== 'name' && name !== 'inputs' && name !== 'precision') {
                        const attribute = new cntk.Attribute(metadata.attribute(type, name), name, value);
                        this.attributes.push(attribute);
                    }
                }
                inputs = obj.inputs.map((input) => values.map(input, version));
                outputs = [values.map(this.name, version)];
                break;
            }
            case 2: {
                this.name = obj.name || obj.uid || null;
                const output = obj.uid;
                if (obj.op === 57) {
                    this.type = metadata.type(obj.uid) || { name: obj.uid };
                } else if (Object.prototype.hasOwnProperty.call(obj, 'op')) {
                    // cntk/Source/CNTKv2LibraryDll/API/Internals/PrimitiveOpType.h
                    const op = Number(obj.op);
                    this.type = metadata.type(op);
                } else {
                    const type = obj.type;
                    this.type = metadata.type(type) || { name: type };
                    if (obj.user_defined_state) {
                        for (const [name, value] of Object.entries(obj.user_defined_state)) {
                            const attribute = new cntk.Attribute(metadata.attribute(type, name), name, value);
                            this.attributes.push(attribute);
                        }
                    }
                }
                if (obj.attributes) {
                    for (const [name, value] of Object.entries(obj.attributes)) {
                        const attribute = new cntk.Attribute(metadata.attribute(this.type, name), name, value);
                        this.attributes.push(attribute);
                    }
                }
                inputs = obj.inputs.map((input) => values.map(input, version));
                outputs.push(values.map(`${output}_Output_0`, version));
                break;
            }
            default: {
                throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
            }
        }
        let inputIndex = 0;
        if (this.type && this.type.inputs) {
            for (const inputSchema of this.type.inputs) {
                if (inputIndex < inputs.length || inputSchema.option !== 'optional') {
                    const inputCount = inputSchema.type === 'Tensor[]' ? (inputs.length - inputIndex) : 1;
                    const inputArguments = [];
                    for (const inputArgument of inputs.slice(inputIndex, inputIndex + inputCount)) {
                        if (inputArgument.name !== '' || inputSchema.option !== 'optional') {
                            inputArguments.push(inputArgument);
                        }
                    }
                    this.inputs.push(new cntk.Argument(inputSchema.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        this.inputs.push(...inputs.slice(inputIndex).map((argument, index) => {
            return new cntk.Argument((inputIndex + index).toString(), [argument]);
        }));

        let outputIndex = 0;
        if (this.type && this.type.outputs) {
            for (const outputSchema of this.type.outputs) {
                if (outputIndex < outputs.length || !outputSchema.optional) {
                    const outputCount = outputSchema.type === 'Tensor[]' ? (outputs.length - outputIndex) : 1;
                    this.outputs.push(new cntk.Argument(outputSchema.name, outputs.slice(outputIndex, outputIndex + outputCount)));
                    outputIndex += outputCount;
                }
            }
        }
        this.outputs.push(...outputs.slice(outputIndex).map((argument) => {
            return new cntk.Argument(outputIndex.toString(), [argument]);
        }));
    }
};

cntk.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.value = value;
        this.type = null;
        if (this.value && this.value.__type__ === 'shape') {
            this.value = new cntk.TensorShape(1, value);
            this.type = 'shape';
        }
        if (cntk.proto && this.value instanceof cntk.proto.NDShape) {
            this.value = new cntk.TensorShape(2, value);
            this.type = 'shape';
        }
        if (cntk.proto && this.value instanceof cntk.proto.Axis) {
            const axis = { __type__: 'Axis' };
            for (const key of Object.keys(value).filter((key) => key !== 'name')) {
                axis[key] = value[key];
            }
            this.value = axis;
        }
        if (metadata) {
            if (metadata.type) {
                this.type = metadata.type;
                const type = cntk[this.type] || cntk.proto[this.type];
                if (type && type[this.value]) {
                    this.value = type[this.value];
                }
            }
            if (metadata.visible === false) {
                this.visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                let defaultValue = metadata.default;
                value = this.value;
                if (typeof value === 'function') {
                    value = value();
                }
                if (this.type === 'shape') {
                    value = value.dimensions;
                }
                if (value === defaultValue) {
                    this.visible = false;
                } else if (Array.isArray(value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] === null) {
                        defaultValue.pop();
                        while (defaultValue.length < value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]);
                        }
                    }
                    if (value.every((item, index) => item === defaultValue[index])) {
                        this.visible = false;
                    }
                }
            }
        }
    }
};

cntk.Tensor = class {

    constructor(version, tensor) {
        this.encoding = '|';
        this.values = null;
        switch (version) {
            case 1: {
                if (tensor.__type__ === 'LearnableParameter') {
                    this.name = tensor.name || null;
                    this.type = new cntk.TensorType(version, tensor.precision, tensor.sampleLayout);
                }
                break;
            }
            case 2: {
                this.name = tensor.name || tensor.uid || null;
                this.type = new cntk.TensorType(version, tensor.data_type, tensor.shape);
                const value = tensor.value;
                if (this.type.dataType === 'float32' && value && value.float_values && value.float_values.value && value.float_values.value.length > 0) {
                    this.values = value.float_values.value;
                }
                break;
            }
            default:
                throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
        }
    }
};

cntk.TensorType = class {

    constructor(version, dataType, shape) {
        this.dataType = '?';
        switch (version) {
            case 1:
                switch (dataType) {
                    case 'float': this.dataType = 'float32'; break;
                    case 'double': this.dataType = 'float64'; break;
                    case 'half': this.dataType = 'float16'; break;
                    case '': this.dataType = 'float32'; break;
                    default: throw new cntk.Error(`Unsupported tensor data type '${dataType}'.`);
                }
                this.shape = new cntk.TensorShape(version, shape);
                break;
            case 2:
                switch (dataType) {
                    case 1n: this.dataType = 'float32'; break;
                    default: throw new cntk.Error(`Unsupported tensor data type '${dataType}'.`);
                }
                this.shape = new cntk.TensorShape(version, shape);
                break;
            default:
                throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
        }
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

cntk.TensorShape = class {

    constructor(version, shape) {
        switch (version) {
            case 1:
                this.dimensions = shape.dims;
                break;
            case 2:
                this.dimensions = shape.shape_dim.map((dimension) => Number(dimension));
                break;
            default:
                throw new cntk.Error(`Unsupported CNTK version '${version}'.`);
        }
    }

    toString() {
        return (this.dimensions && this.dimensions.length) ? (`[${this.dimensions.join(',')}]`) : '';
    }
};

cntk.Function = class {

    constructor(name, nodes, inputs, outputs) {
        this.type = 'function';
        this.name = name;
        this.inputs = inputs;
        this.outputs = outputs;
        this.nodes = nodes;
        switch (this.name) {
            case 'PReLU':
            case 'Softmax':
                this.category = 'Activation';
                break;
            case 'Dropout':
                this.category = 'Dropout';
                break;
            case 'Convolution':
            case 'ConvolutionTranspose':
            case 'Dense':
            case 'linear':
            case 'LSTM':
                this.category = 'Layer';
                break;
            case 'BatchNormalization':
            case 'lrn':
                this.category = 'Normalization';
                break;
            case 'AveragePooling':
            case 'MaxPooling':
                this.category = 'Pool';
                break;
            default:
                this.category = null;
                break;
        }
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
            throw new cntk.Error(`Duplicate function identifier '${func.name}'.`);
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
        const key = `${type}:${name}`;
        if (!this._attributes.has(key)) {
            const metadata = this.type(type);
            if (metadata && metadata.attributes && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    this._attributes.set(`${type}:${attribute.name}`, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }
};

cntk.ComputationNetwork = class {

    constructor(reader) {
        reader = new cntk.BinaryReader(reader);
        const shape = (dims) => {
            return { __type__: 'shape', dims };
        };
        reader.assert('BCN');
        reader.assert('BVersion');
        this.version = reader.uint64().toNumber();
        reader.assert('EVersion');
        const numNodes = reader.uint64().toNumber();
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
            this.rows = reader.uint64().toNumber();
            this.cols = reader.uint64().toNumber();
            this.sampleLayout = reader.shape(true);
            this.dynamicAxisNodeName = '';
            if (version >= 8) {
                const nrAxes = reader.uint32();
                if (nrAxes === 1) {
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
                this.sampleLayout = reader.shape(false);
            } else {
                throw new cntk.Error('LeanableParameter reader implemented.');
            }
            this.value = reader.matrix();
        };
        op.CrossEntropyWithSoftmax = function(reader) {
            this.evalMode = reader.uint32();
            if (this.evalMode > 2) {
                this.evalMode = 0;
                reader.skip(-4);
            }
        };
        op.Times = function(reader, version) {
            this.outputRank = (version >= 3) ? reader.uint64().toNumber() : 1;
            this.inferInputRankToMap = (version >= 12) ? reader.int32() : -1;
        };
        op.Dropout = function(reader, version) {
            if (version >= 16) {
                this.rngSeed = (version === 16) ? reader.uint32() : reader.uint64().toNumber();
                this.rngOffset = reader.uint64().toNumber();
            }
        };
        op.ConvolutionBase = function(reader, version) {
            if (version >= 5) {
                this.kernelShape = reader.shape(false);
                this.mapCount = reader.shape(false);
                this.strides = reader.shape(false);
                this.sharing = reader.booleans();
                this.autoPadding = reader.booleans();
                this.lowerPad = reader.shape(false);
                this.upperPad = reader.shape(false);
                this.poolKind = reader.int32();
                this.imageLayoutKind = reader.int32();
                this.maxTempMemSizeInSamples = reader.uint64().toNumber();
            }
            if (version >= 9) {
                this.transpose = reader.boolean();
            }
            if (version >= 20) {
                this.outputShape = reader.shape(false);
            }
            if (version >= 21) {
                this.ceilOutDim = reader.boolean();
            }
            if (version >= 23) {
                this.includePad = reader.boolean();
            }
        };
        op.Convolution = function(reader, version) {
            op.ConvolutionBase.apply(this, [reader, version]);
            if (version < 5) {
                this.kernelShape = shape([reader.uint64().toNumber(), reader.uint64().toNumber(), 1]);
                this.strides = shape([reader.uint64().toNumber(), reader.uint64().toNumber(), 1]);
                this.mapCount = shape([reader.uint32()]);
                this.imageLayoutKind = reader.int32();
                this.autoPadding = [reader.boolean()];
                this.maxTempMemSizeInSamples = reader.uint64().toNumber();
                this.poolKind = 'None';
                this.convolution2D = true;
                this.sharing = [true];
                this.lowerPad = shape([0]);
                this.upperPad = shape([0]);
            } else {
                this.convolution2D = reader.boolean();
                if (version >= 18) {
                    this.dilation = reader.shape();
                } else {
                    this.dilation = shape([1]);
                }
            }
        };
        op.Pooling = function(reader, version) {
            op.ConvolutionBase.apply(this, [reader, version]);
        };
        op.PoolingBase = function(reader) {
            this.imageLayoutKind = reader.int32();
            this.windowWidth = reader.uint32();
            this.windowHeight = reader.uint64().toNumber();
            this.horizontalSubsample = reader.uint64().toNumber();
            this.verticalSubsample = reader.uint64().toNumber();
        };
        op.MaxPooling = function(reader, version) {
            op.PoolingBase.apply(this, [reader, version]);
        };
        op.ROIPooling = function(reader, version) {
            this.roiOutputShape = reader.shape(false);
            this.poolKind = (version < 26) ? 'Max' : reader.int32();
            this.spatialScale = (version < 26) ? 0.0625 : reader.float64();
        };
        op.Reshape = function(reader) {
            this.beginDimParameter = reader.uint32();
            this.endDimParameter = reader.uint32();
            this.replacementSampleLayout = reader.shape(false);
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
                    if (version === 19) {
                        this.runCountUntied = reader.boolean() ? 0 : 'SIZE_MAX';
                    } else {
                        this.runCountUntied = reader.uint64().toNumber();
                    }
                } else {
                    mbCount = reader.uint64().toNumber();
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
                    mbCount = reader.uint64().toNumber();
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
                this.index.push([[reader.uint64().toNumber(), reader.uint64().toNumber()]]);
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
                this.sampleLayout = reader.shape(false);
            } else {
                const rows = reader.uint64().toNumber();
                reader.uint64();
                this.sampleLayout = shape([rows], true);
            }
            if (version >= 2) {
                this.initialStateValue = reader.int32();
            }
        };
        op.FutureValue = function(reader, version) {
            this.timeStep = reader.int32();
            if (version > 3) {
                this.sampleLayout = reader.shape(false);
            } else {
                const rows = reader.uint64().toNumber();
                reader.uint64();
                this.sampleLayout = shape([rows], true);
            }
            if (version >= 2) {
                this.initialStateValue = reader.int32();
            }
        };
        op.TransposeDimensions = function(reader, version) {
            if (version >= 3) {
                this.axis1 = reader.int32();
                this.axis2 = reader.int32();
                if (version >= 25 && this.axis1 === 0 && this.axis2 === 0) {
                    const size = reader.uint64().toNumber();
                    this.perm = [];
                    for (let i = 0; i < size; i++) {
                        this.perm.push(reader.uint64().toNumber());
                    }
                }
            } else {
                this.axis1 = 1;
                this.axis2 = 2;
            }
        };
        op.AveragePooling = function(reader, version) {
            op.PoolingBase.apply(this, [reader, version]);
        };
        op.InvStdDev = function(reader) {
            this.hasComputed = reader.boolean();
            this.value = reader.matrix();
        };
        op.Mean = function(reader) {
            this.hasComputed = reader.boolean();
            this.value = reader.matrix();
        };
        op.PerDimMeanVarNormalization = function() {};
        op.Softmax = function() {};
        op.DynamicAxis = function() {};

        const nodes = [];
        this.nodes = {};
        for (let i = 0; i < numNodes; i++) {
            const precision = this.version >= 7 ? reader.string() : '';
            if (precision !== 'float' && precision !== 'double' && precision !== 'half' && precision !== '') {
                throw new cntk.Error(`Invalid precision format '${precision}'.`);
            }
            const obj = { __type__: reader.string() };
            obj.name = reader.string();
            obj.precision = precision;
            const constructor = op[obj.__type__];
            if (!constructor) {
                throw new cntk.Error(`Unsupported node type '${obj.__type__}'.`);
            }
            constructor.apply(obj, [reader, this.version]);
            nodes.push(obj);
            this.nodes[obj.name] = obj;
        }
        reader.assert('ENodeList');
        reader.assert('BRelation');
        for (let j = 0; j < numNodes; j++) {
            const nodeName = reader.string();
            const node = this.nodes[nodeName];
            const numChildren = reader.uint64().toNumber();
            const children = [];
            for (let k = 0; k < numChildren; k++) {
                children.push(reader.string());
            }
            if (this.version < 19 && node.__type__ === 'BatchNormalization') {
                const runSampleCount = {
                    __type__: 'LearnableParameter',
                    name: `${nodeName}.run_sample_count`,
                    precision: node.precision,
                    sampleLayout: shape([1]),
                    learningRateMultiplier: 0
                };
                nodes.push(runSampleCount);
                this.nodes[runSampleCount.name] = runSampleCount;
                children.push(runSampleCount.name);
            }
            if (node.__type__ === 'Convolution' && children.length > 1) {
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
        if (this.criterion.length === 0) {
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

cntk.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get position() {
        return this._reader.position;
    }

    seek(offset) {
        this._reader.seek(offset);
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    read(length) {
        return this._reader.read(length);
    }

    boolean() {
        return this._reader.boolean();
    }

    byte() {
        return this._reader.byte();
    }

    int32() {
        return this._reader.int32();
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

    float32() {
        return this._reader.float32();
    }

    float64() {
        return this._reader.float64();
    }

    match(text) {
        const position = this.position;
        for (let i = 0; i < text.length; i++) {
            if (this.uint16() !== text.charCodeAt(i)) {
                this.seek(position);
                return false;
            }
        }
        if (this.uint16() !== 0) {
            this.seek(position);
            return false;
        }
        return true;
    }

    assert(text) {
        if (!this.match(text)) {
            throw new cntk.Error(`Invalid '${text}' signature.`);
        }
    }

    string() {
        const content = [];
        let c = this.uint16();
        while (c !== 0) {
            content.push(String.fromCharCode(c));
            c = this.uint16();
        }
        return content.join('');
    }

    strings() {
        const size = this.uint64().toNumber();
        const array = new Array(size);
        for (let i = 0; i < size; i++) {
            array[i] = this.string();
        }
        return array;
    }

    booleans() {
        const size = this.uint64().toNumber();
        const array = new Array(size);
        for (let i = 0; i < size; i++) {
            array[i] = this.boolean();
        }
        return array;
    }

    matrix() {
        const type = this.byte();
        switch (type) {
            case 100: {
                // dense
                this.assert('BMAT');
                const elsize = this.uint64().toNumber();
                const value = {};
                value.name = this.string();
                value.format = this.uint32();
                value.rows = this.uint64().toNumber();
                value.columns = this.uint64().toNumber();
                this.read(elsize * value.rows * value.columns);
                this.assert('EMAT');
                return value;
            }
            case 115: // sparse
                throw new cntk.Error('Matrix sparse type not implemented.');
            default:
                throw new cntk.Error(`Matrix type '${type}' not implemented.`);
        }
    }

    shape(acceptLegacyFormat) {
        const dims = [];
        const rank = this.uint32();
        let dim0 = 0;
        if (rank > 0) {
            dim0 = this.uint32();
        }
        if (!acceptLegacyFormat || dim0 !== 0) {
            if (rank > 0) {
                dims.push(dim0);
            }
            for (let i = 1; i < rank; i++) {
                dims.push(this.uint32());
            }
        } else {
            const dim = this.uint32();
            dims.push(this.uint32());
            dims.push(rank);
            dims.push(dim);
        }
        return { __type__: 'shape', dims };
    }
};

cntk.ImageLayoutKind = {
    0: 'CHW',
    1: 'HWC'
};

cntk.PoolKind = {
    0: 'None',
    1: 'Max',
    2: 'Average'
};

cntk.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading CNTK model.';
    }
};

export const ModelFactory = cntk.ModelFactory;
