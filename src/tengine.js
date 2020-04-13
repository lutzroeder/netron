/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var tengine = tengine || {};
var base = base || require('./base');

tengine.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'tmfile') {
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const majorVersion = buffer[0] | buffer[1] << 8 ;
                if (majorVersion < 4) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        return tengine.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier.toLowerCase();
            try {
                const buffer = context.buffer;
                const majorVersion = buffer[0] | buffer[1] << 8;
                const minorVersion = buffer[2] | buffer[3] << 8;
                if (majorVersion !== 2) {
                    throw new tengine.Error("Unsupported format version 'v" + majorVersion.toString() + "." + minorVersion.toString() + "'.");
                }
                return new tengine.Model(metadata, buffer);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new tengine.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
            }
        });
    }
}

tengine.Model = class {

    constructor(metadata, buffer) {
        const reader = new tengine.ModelFileReader(buffer);
        this._version = reader.version;
        this._producer = reader.producer;
        this._graphs = reader.graphs.map((graph) => new tengine.Graph(metadata, graph));
    }

    get format() {
        return "Tengine v" + this._version;
    }

    get producer() {
        return this._producer;
    }

    get graphs() {
        return this._graphs; 
    }
}

tengine.Graph = class {

    constructor(metadata, graph) {
        this._name = graph.id.toString();
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const tensors = graph.tensors.map((tensor) => new tengine.Argument(tensor));

        for (const input of graph.inputs) {
            const argument = tensors[input];
            this._inputs.push(new tengine.Parameter(argument.name, true, [ argument ]));
        }

        for (const output of graph.outputs) {
            const argument = tensors[output];
            if (argument.type && argument.type.shape && argument.type.shape.dimensions && argument.type.shape.dimensions.length == 0 && argument.initializer !== null) {
                continue;
            }
            this._outputs.push(new tengine.Parameter(argument.name, true, [ argument ]));
        }

        for (const node of graph.nodes) {
            if (node.operator !== 'INPUT') {
                this._nodes.push(new tengine.Node(metadata, node, tensors));
            }
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
}

tengine.Parameter = class {

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

tengine.Argument = class { 

    constructor(tensor) {
        this._name = tensor.name;
        this._type = new tengine.TensorType(tensor.dataType, new tengine.TensorShape(tensor.dims));
        this._initializer = (tensor.type === 2) ? new tengine.Tensor(this._type, tensor.buffer) : null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get quantization() {
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};


tengine.Node = class {

    constructor(metadata, node, tensors) {
        this._metadata = metadata;
        this._name = node.name;
        this._operator = node.operator;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        const schema = metadata.type(this._operator);

        for (let i = 0; i < node.params.length; i++) {
            const attributeSchema = (schema && schema.attributes && i < schema.attributes.length) ? schema.attributes[i] : null;
            const attributeName = attributeSchema ? attributeSchema.name : i.toString();
            this._attributes.push(new tengine.Attribute(attributeSchema, attributeName, node.params[i])); 
        }

        let inputs = node.inputs; 
        let inputIndex = 0;
        if (schema && schema.inputs) { 
            for (const inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    let inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    let inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => tensors[id]);
                    this._inputs.push(new tengine.Parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        else {
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((id, index) => { 
                let inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString(); 
                return new tengine.Parameter(inputName, true, [ tensors[id] ]);
            }));
        }

        let outputs = node.outputs;
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (const outputDef of schema.outputs) { 
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    let outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    let outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => tensors[id]);
                    this._outputs.push(new tengine.Parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        else {
            this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((id, index) => {
                let outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new tengine.Parameter(outputName, true, [ tensors[id] ]);
            }));
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata.type(this._operator);
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
}

tengine.Attribute = class {

    constructor(schema, key, value) {
        this._type = '';
        this._name = key;
        this._value = value;
        if (schema) {
            this._name = schema.name;
            if (schema.type) {
                this._type = schema.type;
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default || (this._value && this._value.toString() == schema.default.toString())) {
                    this._visible = false;
                }
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
}

tengine.Tensor = class {

    constructor(type, data, kind) {
        this._type = type;
        this._data = data;
        this._kind = kind;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.shape || (this._type.shape.dimensions && this._type.shape.dimensions.length == 0)) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'int8':
            case 'uint8':
            case 'float16':
            case 'float32':
            case 'int32':
            case 'int16':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (context.shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        let size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'uint8':
                        results.push(context.data.getUint8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
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

}

tengine.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case 0: this._dataType = 'float32'; break;
            case 1: this._dataType = 'float16'; break;
            case 2: this._dataType = 'int8'; break;
            case 3: this._dataType = 'uint8'; break;
            case 4: this._dataType = 'int32'; break;
            case 5: this._dataType = 'int16'; break;
            default: throw new tengine.Error("Unknown data type'" + dataType + "'.");
        }
        this._shape = shape;
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
}

tengine.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

tengine.Metadata = class {

    static open(host) {
        if (tengine.Metadata._metadata) {
            return Promise.resolve(tengine.Metadata._metadata);
        }
        return host.request(null, 'tengine-metadata.json', 'utf-8').then((data) => {
            tengine.Metadata._metadata = new tengine.Metadata(data);
            return tengine.Metadata._metadata;
        }).catch(() => {
            tengine.Metadata._metadata = new tengine.Metadata(null);
            return tengine.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

tengine.ModelFileReader = class {

    constructor(buffer) {

        // ./third_party/src/tengine/serializer/include/tengine/v2/tm2_format.h
        // https://github.com/OAID/Tengine/wiki/The-format-of-tmfile

        let operators = new Map();
        operators.set( 0, { name: 'Accuracy', params: [] });
        operators.set( 1, { name: 'BatchNormalization', params: [ 'f', 'f', 'i' ] });
        operators.set( 2, { name: 'BilinearResize', params: [ 'f', 'f', 'i' ] });
        operators.set( 3, { name: 'Concat', params: [ 'i' ] });
        operators.set( 4, { name: 'Const', params: [] });
        operators.set( 5, { name: 'Convolution', params: [ 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set( 6, { name: 'DeConvolution', params: [ 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set( 7, { name: 'DetectionOutput', params: [ 'i', 'i', 'i', 'f', 'f' ] });
        operators.set( 8, { name: 'DropOut', params: [] });
        operators.set( 9, { name: 'Eltwise', params: [ 'i', 'i' ] });
        operators.set(10, { name: 'Flatten', params: [ 'i' ] });
        operators.set(11, { name: 'FullyConnected', params: [ 'i' ] });
        operators.set(12, { name: 'INPUT', params: [] });
        operators.set(13, { name: 'LRN', params: [ 'i', 'f', 'f', 'i', 'f' ] });
        operators.set(14, { name: 'Normalize', params: [ 'i', 'i' ] });
        operators.set(15, { name: 'Permute', params: [ 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(16, { name: 'Pooling', params: [ 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(17, { name: 'Prelu', params: [] });
        operators.set(18, { name: 'PriorBox', params: [ 'f[]', 'f[]', 'f[]', 'f[]', 'i', 'i', 'i', 'i', 'i', 'f', 'f', 'f', 'i', 'i' ] });
        operators.set(19, { name: 'Region', params: [ 'i', 'i', 'i', 'i', 'f', 'f', 'f[]' ] });
        operators.set(20, { name: 'ReLU', params: [ 'f' ] });
        operators.set(21, { name: 'ReLU6', params: [] });
        operators.set(22, { name: 'Reorg', params: [ 'i' ] });
        operators.set(23, { name: 'Reshape', params: [ 'i', 'i', 'i[]' ] });
        operators.set(24, { name: 'RoiPooling', params: [ 'i', 'i', 'f' ] });
        operators.set(25, { name: 'RPN', params: [ 'f[]', 'f[]', 'i', 'i', 'i', 'i', 'i', 'f', 'anchors' ] });
        operators.set(26, { name: 'Scale', params: [ 'i', 'i', 'i' ]});
        operators.set(27, { name: 'Slice', params: [ 'i', 'i[]', 'i[]', 'i[]', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(28, { name: 'SoftMax', params: [ 'i' ] });
        operators.set(29, { name: 'Split', params: [ 'i', 'i', 'boolean', 'boolean', 'i[]' ] });
        operators.set(30, { name: 'DetectionPostProcess', params: [ 'i', 'i', 'f', 'f', 'i', 'f[]' ] });
        operators.set(31, { name: 'Gemm', params: [ 'f', 'f', 'i', 'i' ] });
        operators.set(32, { name: 'Generic', params: [ 'i', 'i', 'string' ] });
        operators.set(33, { name: 'Logistic', params: [] });
        operators.set(34, { name: 'LSTM', params: [ 'f', 'f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(35, { name: 'RNN', params: [ 'f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(36, { name: 'TanH', params: [] });
        operators.set(37, { name: 'Sigmoid', params: [] });
        operators.set(38, { name: 'Squeeze', params: [ 'i', 'i', 'i', 'i' ] });
        operators.set(39, { name: 'FusedbnScaleRelu', params: [] });
        operators.set(40, { name: 'Pad', params: [ 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'f' ] });
        operators.set(41, { name: 'StridedSlice', params: [ 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(42, { name: 'ArgMax', params: [ 'i' ] });
        operators.set(43, { name: 'ArgMin', params: [ 'i' ] });
        operators.set(44, { name: 'TopKV2', params: [ 'i', 'i' ] });
        operators.set(45, { name: 'Reduction', params: [ 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(46, { name: 'Max', params: [] });
        operators.set(47, { name: 'Min', params: [] });
        operators.set(48, { name: 'GRU', params: [ 'f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(49, { name: 'Addn', params: 'i' });
        operators.set(50, { name: 'SwapAxis', params: [ 'i', 'i' ] });
        operators.set(51, { name: 'Upsample', params: [ 'f' ] });
        operators.set(52, { name: 'SpaceToBatchND', params: [ 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(53, { name: 'BatchToSpaceND', params: [ 'i', 'i', 'i', 'i', 'i', 'i' ] });
        operators.set(54, { name: 'Resize', params: [ 'f', 'f', 'i' ] });
        operators.set(55, { name: 'ShuffleChannel', params: [ 'i' ] });
        operators.set(56, { name: 'Crop', params: [ 'i', 'i', 'i', 'i', 'i', 'i', 'boolean', 'i', 'i' ] });
        operators.set(57, { name: 'ROIAlign', params: [ 'i', 'i', 'f' ] });
        operators.set(58, { name: 'Psroipooling', params: [ 'i', 'i', 'f', 'i' ] });
        operators.set(59, { name: 'Unary', params: [ 'i' ] });
        operators.set(60, { name: 'Expanddims', params: [ 'i' ] });
        operators.set(61, { name: 'Bias', params: [ 'i' ] });
        operators.set(62, { name: 'Noop', params: [] });
        operators.set(63, { name: 'Threshold', params: [ 'f' ] });
        operators.set(64, { name: 'Hardsigmoid', params: [ 'f', 'f' ] });
        operators.set(65, { name: 'Embed', params: [ 'f', 'f', 'f', 'f' ] });
        operators.set(66, { name: 'InstanceNorm', params: [ 'f' ] });
        operators.set(67, { name: 'MVN', params: [ 'i', 'i', 'f' ] });
        operators.set(68, { name: 'Absval', params: [] });
        operators.set(69, { name: 'Cast', params: [ 'i', 'i' ] });
        operators.set(70, { name: 'HardSwish', params: [ 'f', 'f' ] });
        operators.set(71, { name: 'Interp', params: [ 'i', 'i', 'f', 'f', 'i' ] });
        operators.set(72, { name: 'SELU', params: [ 'f', 'f' ] });
        operators.set(73, { name: 'ELU', params: [ 'f' ] });
        operators.set(74, { name: 'BroadMul', params: [] });
        operators.set(75, { name: 'Logical', params: [ 'i' ] });
        operators.set(76, { name: 'Gather', params: [ 'i', 'i' ] });
        operators.set(77, { name: 'Transpose', params: [ 'i[]' ] });
        operators.set(78, { name: 'Num', params: [] });

        const reader = new tengine.BinaryReader(buffer);
        this._majorVersion = reader.uint16();
        this._minorVersion = reader.uint16();
        this._compileVersion = reader.uint16();
        reader.skip(2); // struct align
        reader.seek(reader.uint32()); // root table
        this._originalFormat = reader.int32();
        this._subFormat = reader.int32();
        this._graphs = [];
        const subgraphOffsets = reader.uint32s();
        for (const subgraphOffset of subgraphOffsets) {
            reader.seek(subgraphOffset);

            let subgraph = {};
            subgraph.id = reader.int32();
            subgraph.graphLayout = reader.int32();
            /* 
            if (graphLayout == 0) {
                return "NCHW";
            }
            if (graphLayout == 1) {
                return "NHWC";
            }
            */
            subgraph.originalLayout = reader.int32();
            subgraph.inputs = reader.uint32s();
            subgraph.outputs = reader.uint32s();
            const nodeOffsets = reader.uint32s();
            const tensorOffsets = reader.uint32s();
            const bufferOffsets = reader.uint32s();
            subgraph.name = reader.string();
            subgraph.nodes = [];
            subgraph.tensors = [];
            this._graphs.push(subgraph);

            // nodes
            for (const nodeOffset of nodeOffsets) {
                reader.seek(nodeOffset);
                let node = {};
                node.id = reader.int32();
                node.inputs = reader.uint32s();
                node.outputs = reader.uint32s();
                const operatorOffset = reader.int32();
                node.name = reader.string();
                const attributeOffsets = reader.uint32s();
                node.dynamicShape = reader.boolean() ? true : false;

                reader.seek(operatorOffset);
                node.operatorVersion = reader.int32(); 
                const operatorIndex = reader.int32();
                const paramsOffset = reader.uint32();

                const schema = operators.has(operatorIndex) ? operators.get(operatorIndex) : null;
                node.operator = schema ? schema.name : operatorIndex.toString();
                const paramTypes = schema ? schema.params : []

                node.params = [];
                if (paramsOffset) {
                    reader.seek(paramsOffset);
                    for (const paramType of paramTypes) {
                        if (paramType !== 'boolean') {
                            reader.align(4);
                        }
                        switch (paramType) {
                            case 'i':
                                node.params.push(reader.int32());
                                break;
                            case 'f':
                                node.params.push(reader.float32());
                                break;
                            case 'i[]':
                                node.params.push(reader.int32s());
                                break;
                            case 'f[]':
                                node.params.push(reader.float32s());
                                break;
                            case 'boolean':
                                node.params.push(reader.boolean());
                                break;
                            case 'string':
                                node.params.push(reader.string());
                                break;
                            case 'anchors':
                                node.params.push(reader.anchors(4));
                                break;
                            default:
                                throw new tengine.Error("Unsupported param type '" + paramType + "' in '" + node.operator + "'.");
                        }
                    }
                }

                if (node.operator === 'Slice') {
                    node.params[6] = (this._originalFormat == 5) ? node.params[6] : 0;
                }

                node.attributes = [];
                for (const attributeOffset of attributeOffsets) {
                    reader.seek(attributeOffset);
                    const name = reader.string();
                    const value = reader.string();
                    const type = reader.int32();
                    node.attributes.push({ name: name, value: value, type: type });
                }

                if (node.operator !== 'Const') {
                    subgraph.nodes.push(node);
                }
            }

            // buffers
            let buffers = [];
            for (const buffersOffset of bufferOffsets) {
                reader.seek(buffersOffset);
                const size = reader.uint32();
                reader.seek(reader.int32())
                buffers.push(reader.bytes(size));
            }

            // tensors
            for (const tensorOffset of tensorOffsets) {
                reader.seek(tensorOffset);
                let tensor = {};
                tensor.id = reader.int32();
                tensor.buffer = buffers[reader.int32()];
                tensor.dims = reader.int32s();
                tensor.name = reader.string(); 
                const quantparamsOffset = reader.int32(); 
                tensor.layout = reader.int32();
                tensor.type = reader.int32(); // const = 2, input = 3, var = 1, dep, unknown
                tensor.dataType = reader.int32();
                if (quantparamsOffset) {
                    reader.seek(quantparamsOffset);
                    tensor.quantparams = {
                        zeroPoint: reader.int32(),
                        scale: reader.float32(),
                        width: reader.int32()
                    };
                }
                subgraph.tensors.push(tensor);
            }

            for (const node of subgraph.nodes) {
                if (node.operator === 'Convolution') {
                    switch (subgraph.graphLayout) {
                        case 0: // NCHW
                            node.params[6] = subgraph.tensors[node.inputs[1]].dims[1];
                            break;
                        case 1: // NHWC
                            node.params[6] = subgraph.tensors[node.inputs[1]].dims[3];
                            break;
                    }
                }
            }

        }
    }

    get version() {
        return this._majorVersion + '.' + this._minorVersion;
    }

    get producer() {
        switch (this._originalFormat) {
            case 0: return '';
            case 1: return 'Tengine';
            case 2: return 'Caffe';
            case 3: return 'ONNX';
            case 4: return 'MXNet';
            case 5: return 'TensorFlow';
            case 6: return 'TensorFlow Lite';
            case 7: return 'Darknet';
            case 8: return 'DLA v' + this._subFormat;
            default: throw new tengine.Error("Unknown producer '" + this._originalFormat.toString() + "'.");
        }
    }

    get graphs() {
        return this._graphs;
    }
}

tengine.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    seek(position) {
        this._position = position;
        if (this._position > this._buffer.length) {
            throw new tengine.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new tengine.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    align(mod) {
        if (this._position % mod != 0) {
            this.skip(mod - (this._position % mod));
        }
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.slice(position, this._position);
    }

    byte() {
        this.skip(1);
        return this._dataView.getUint8(this._position);
    }


    boolean() {
        return this.byte() == 0x00 ? true : false;
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getUint16(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    uint32s() {
        let values = [];
        const offset = this.uint32();
        if (offset) {
            const next = this._position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                values.push(this.uint32());
            }
            this.seek(next);
        }
        return values;
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    int32s() {
        let values = [];
        const offset = this.uint32();
        if (offset) {
            const next = this._position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                values.push(this.int32());
            }
            this.seek(next);
        }
        return values;
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    float32s() {
        let values = [];
        const offset = this.uint32();
        if (offset) {
            const next = this._position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                values.push(this.float32());
            }
            this.seek(next);
        }
        return values;
    }

    anchors(length) {
        let arrays = [];
        const offset = this.uint32();
        if (offset) {
            const next = this._position;
            this.seek(offset);
            const count = this.uint32();
            for (let i = 0; i < count; i++) {
                let array = [];
                for (let j = 0; j < length; j++) {
                    array.push(this.float32());
                }
                arrays.push(array);
            }
            this.seek(next);
        }
        return arrays;
    }

    string() {
        const position = this.uint32();
        let text = '';
        if (position) {
            const next = this._position;
            this.seek(position);
            const size = this.uint32();
            this.seek(this.uint32());
            for(let i = 0; i < size - 1; i++) {
                text += String.fromCharCode(this._buffer[this._position++]);
            }
            this.seek(next);
        }
        return text;
    }
}

tengine.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Tengine model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tengine.ModelFactory;
}
