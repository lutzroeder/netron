/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var cntk = cntk || {};
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');

var cntk_v1 = {};
var cntk_v2 = null;

cntk.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'model' || extension == 'cmf' || extension == 'dnn' || extension == 'cntk') {
            const buffer = context.buffer;
            // Reject PyTorch models with .model file extension.
            const torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                return false;
            }
            // CNTK v1
            if (buffer && buffer.length >= 8 && 
                buffer[0] == 0x42 && buffer[1] == 0x00 && buffer[2] == 0x43 && buffer[3] == 0x00 &&
                buffer[4] == 0x4E && buffer[5] == 0x00 && buffer[6] == 0x00 && buffer[7] == 0x00) {
                return true;
            }
            // CNTK v2
            const tags = context.tags('pb');
            if (tags.get(1) === 0 && tags.get(2) === 2) {
                return true;
            }
            return false;
        }
    }

    open(context, host) { 
        return host.require('./cntk-proto').then(() => {
            let version = 0;
            let obj = null;
            try {
                const buffer = context.buffer;
                if (buffer && buffer.length >= 8 && 
                    buffer[0] == 0x42 && buffer[1] == 0x00 && buffer[2] == 0x43 && buffer[3] == 0x00 &&
                    buffer[4] == 0x4E && buffer[5] == 0x00 && buffer[6] == 0x00 && buffer[7] == 0x00) {
                    obj = new cntk_v1.ComputationNetwork(buffer);
                    version = 1;
                }
            }
            catch (error) {
                throw new cntk.Error("File format is not CNTK v1 (" + error.message + ") in '" + context.identifier + "'.");
            }
            try {
                if (!obj) {
                    cntk_v2 = protobuf.roots.cntk.CNTK.proto;
                    cntk_v2.PoolingType = { 0: 'Max', 1: 'Average' };
                    const dictionary = cntk_v2.Dictionary.decode(context.buffer);
                    obj = cntk.ModelFactory._convertDictionary(dictionary);
                    version = 2;
                }
            }
            catch (error) {
                throw new cntk.Error("File format is not cntk.Dictionary (" + error.message + ") in '" + context.identifier + "'.");
            }
            return cntk.Metadata.open(host).then((metadata) => {
                try {
                    return new cntk.Model(metadata, version, obj);
                }
                catch (error) {
                    throw new cntk.Error(error.message);
                }
            });
        });
    }

    static _convertDictionary(dictionary) {
        let target = {};
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
        }
        throw new cntk.Error("Unknown dictionary value type '" + dictionaryValue.value_type.toString() + "'.");
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
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._functions = [];

        let args = {};
        switch (version) {
            case 1: {
                for (const name of Object.keys(obj.nodes)) {
                    const node = obj.nodes[name];
                    switch (node.__type__) {
                        case 'InputValue':
                            this._inputs.push(new cntk.Parameter(node.name, [ 
                                new cntk.Argument(version, node)
                            ]));
                            break;
                        case 'LearnableParameter':
                            args[node.name] = new cntk.Argument(version, node);
                            break;
                    }
                }
                for (const name of Object.keys(obj.nodes)) {
                    const node = obj.nodes[name];
                    if (node.__type__ != 'InputValue' && node.__type__ != 'LearnableParameter') {
                        this._nodes.push(new cntk.Node(metadata, version, node, args));
                    }
                }
                if (obj.output) {
                    for (const output of obj.output) {
                        this._outputs.push(new cntk.Parameter(output, [ 
                            new cntk.Argument(version, output)
                        ]));
                    }
                }
                break;
            }
            case 2: {
                let nodeMap = new Map();
                for (const node of obj.primitive_functions) {
                    nodeMap.set(node.uid, node);
                }
                for (const input of obj.inputs) {
                    let argument = new cntk.Argument(version, input);
                    args[input.uid] = argument;
                    // VariableKind { 0: 'input', 1: 'output', 2: 'parameter', 3: 'constant', 4: 'placeholder' }
                    if (input.kind == 0) {
                        let inputName = input.name || input.uid;
                        this._inputs.push(new cntk.Parameter(inputName, [ argument ]));
                    }
                }
                for (const block of obj.primitive_functions) {
                    if (block.op == 57 && block.block_function_composite) {
                        let list = [ block.block_function_composite.root ];
                        let nodes = [];
                        while (list.length > 0) {
                            const name = list.shift();
                            if (nodeMap.has(name)) {
                                const node = nodeMap.get(name);
                                nodes.push(new cntk.Node(metadata, version, node, args));
                                nodeMap.delete(name);
                                for (let i = 0; i < node.inputs.length; i++) {
                                    let parts = node.inputs[i].split('_');
                                    if (parts.length >= 3) {
                                        parts.pop();
                                        if (parts.pop() == 'Output') {
                                            list.push(parts.join('_'));
                                        }
                                    }
                                }
                            }
                        }
                        let inputs = [];
                        let outputs = [ block.block_function_composite.root ];
                        this._functions.push(new cntk.Function(block.block_function_op_name, nodes, inputs, outputs));
                    }
                }
                for (const node of obj.primitive_functions) {
                    if (nodeMap.has(node.uid)) {
                        this._nodes.push(new cntk.Node(metadata, version, node, args));
                    }
                }
                break;
            }
            default:
                throw new cntk.Error("Unsupported graph version '" + version + "'.");
        }
    }

    get nodes() {
        return this._nodes;
    }

    get functions() {
        return this._functions;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }
};

cntk.Function = class {

    constructor(name, nodes, inputs, outputs) {
        this._name = name;
        this._inputs = inputs;
        this._outputs = outputs;
        this._nodes = nodes;
    }

    get name() {
        return this._name;
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

cntk.Parameter = class {
    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

cntk.Argument = class {

    constructor(version, obj) {
        if (typeof obj === 'string') {
            this._name = obj;
        }
        else {
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
                    }
                    break;
                case 2:
                    if (obj.value) {
                        this._name = obj.name || obj.uid;
                        this._type = null;
                        this._initializer = new cntk.Tensor(version, obj);
                    }
                    else {
                        this._name = obj.uid;
                        this._type = new cntk.TensorType(version, obj.data_type, obj.shape);
                        this._initializer = null;
                    }    
                    break;
            }
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

    constructor(metadata, version, obj, args) {

        this._metadata = metadata;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        let inputs = [];
        let outputs = [];
        let initializers = [];

        switch (version) {
            case 1: {
                this._operator = obj.__type__;
                this._name = obj.name;
                for (const attributeName of Object.keys(obj)) {
                    if (attributeName != '__type__' && attributeName != 'name' && attributeName != 'inputs' && attributeName != 'precision') {
                        this._attributes.push(new cntk.Attribute(this._metadata, this._operator, attributeName, obj[attributeName]));
                    }
                }
                inputs = obj.inputs.map((input) => { 
                    if (args[input]) {
                        return args[input];
                    }
                    return new cntk.Argument(version, input);
                });
                outputs = [ new cntk.Argument(version, this._name) ];
                break;
            }
            case 2: {
                this._name = obj.name || obj.uid || null;
                const output = obj.uid;
                if (obj.op == 57) {
                    this._operator = 'Block';
                    if (obj.block_function_op_name) {
                        this._operator = obj.block_function_op_name;
                        this._function = true;
                    }
                }
                else {
                    if (!Object.prototype.hasOwnProperty.call(obj, 'op')) {
                        this._operator = obj.type;
                        if (obj.user_defined_state) {
                            for (const attributeName of Object.keys(obj.user_defined_state)) {
                                this._attributes.push(new cntk.Attribute(this._metadata, this._operator, attributeName, obj.user_defined_state[attributeName]));
                            }
                        }
                    }
                    else {
                        this._operator = this._metadata.name(obj.op);
                        if (this._operator == null) {
                            this._operator = obj.op ? obj.op.toString() : '?';
                        }
                    }
                }
                if (obj.attributes) {
                    for (const attributeName of Object.keys(obj.attributes)) {
                        this._attributes.push(new cntk.Attribute(this._metadata, this._operator, attributeName, obj.attributes[attributeName]));
                    }
                }
                for (const input of obj.inputs) {
                    let argument = args[input];
                    if (argument) {
                        if (argument.initializer) {
                            initializers.push(argument);
                        }
                        else {
                            inputs.push(argument);
                        }
                    }
                    else {
                        inputs.push(new cntk.Argument(version, input));
                    }
                }
                outputs.push(new cntk.Argument(version, output + '_Output_0'));
                inputs = inputs.concat(initializers);
                break;
            }
        }

        let inputIndex = 0;
        const schema = this._metadata.type(this._function ? ('Function:' + this._operator) : this._operator);
        if (schema && schema.inputs) {
            for (const inputSchema of schema.inputs) {
                if (inputIndex < inputs.length || inputSchema.option != 'optional') {
                    let inputCount = (inputSchema.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    let inputArguments = [];
                    for (const inputArgument of inputs.slice(inputIndex, inputIndex + inputCount)) {
                        if (inputArgument.name != '' || inputSchema.option != 'optional') {
                            inputArguments.push(inputArgument);
                        }
                    }
                    this._inputs.push(new cntk.Parameter(inputSchema.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((argument, index) => {
            return new cntk.Parameter((inputIndex + index).toString(), [ argument ]);
        }));

        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (const outputSchema of schema.outputs) {
                if (outputIndex < outputs.length || outputSchema.option != 'optional') {
                    let outputCount = (outputSchema.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    this._outputs.push(new cntk.Parameter(outputSchema.name, outputs.slice(outputIndex, outputIndex + outputCount)));
                    outputIndex += outputCount;
                }
            }
        }
        this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((argument) => {
            return new cntk.Parameter(outputIndex.toString(), [ argument ]);
        }));
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get function() {
        return this._function || false;
    }

    get metadata() {
        return this._metadata.type(this._function ? ('Function:' + this._operator) : this._operator);
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

    constructor(metadata, operator, name, value) {
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

        const schema = metadata.attribute(operator, name);
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
                let type = cntk_v1[this._type] || cntk_v2[this._type];
                if (type && type[this._value]) {
                    this._value = type[this._value];
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                let defaultValue = schema.default;
                value = this._value;
                if (typeof value == 'function') {
                    value = value();
                }
                if (this._type == 'shape') {
                    value = value.dimensions;
                }
                if (value == defaultValue) {
                    this._visible = false;
                }
                else if (Array.isArray(value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]); 
                        }
                    }
                    if (value.every((item, index) => { return item == defaultValue[index]; })) {
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
        }
    }

    get name() {
        return this._name;
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
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        let value = this._value;
        if (!value) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'float32':
                if (value.float_values && value.float_values.value && value.float_values.value.length > 0) {
                    context.data = value.float_values.value;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
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
                results.push(context.data[context.index++]);
                context.count++;
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
                }
                this._shape = new cntk.TensorShape(version, shape);
                break;
            case 2:
                if (long.Long.isLong(dataType)) {
                    dataType = dataType.toNumber(); 
                }
                switch (dataType) {
                    case 1: this._dataType = 'float32'; break;
                }
                this._shape = new cntk.TensorShape(version, shape);
                break;
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
                this._dimensions = shape.shape_dim.map((dimension) => {
                    if (dimension.low == -1 && dimension.high == -1 && dimension.unsigned == true) {
                        return -1;
                    }
                    if (dimension && long.Long.isLong(dimension)) {
                        return dimension.toNumber();
                    }
                    return dimension;
                });
                break;
        }
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return (this._dimensions && this._dimensions.length) ? ('[' + this._dimensions.join(',') + ']') : '';
    }
};

cntk.Metadata = class {

    static open(host) {
        if (cntk.Metadata._metadata) {
            return Promise.resolve(cntk.Metadata._metadata);
        }
        return host.request(null, 'cntk-metadata.json', 'utf-8').then((data) => {
            cntk.Metadata._metadata = new cntk.Metadata(data);
            return cntk.Metadata._metadata;
        }).catch(() => {
            cntk.Metadata._metadata = new cntk.Metadata(null);
            return cntk.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        this._operatorMap = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        const name = item.name;
                        const schema = item.schema;
                        schema.name = name;
                        this._map[name] = schema;
                        if (Object.prototype.hasOwnProperty.call(schema, 'operator')) {
                            this._operatorMap[schema.operator.toString()] = name;
                        }
                    }
                }
            }
        }
    }

    name(code) {
        // cntk/Source/CNTKv2LibraryDll/API/Internals/PrimitiveOpType.h
        return this._operatorMap[code] || null;
    }

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            let schema = this.type(operator);
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

cntk_v1.ComputationNetwork = class {

    constructor(buffer) {
        let reader = new cntk_v1.Reader(buffer);
        reader.assert('BCN');
        reader.assert('BVersion');
        this.version = reader.uint64();
        reader.assert('EVersion');
        let numNodes = reader.uint64();
        reader.assert('BNodeList');
        let op = {};
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
                let nrAxes = reader.uint32();
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
            }
            else {
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
                this.sharing = reader.booleans(reader.uint64());
                this.autoPadding = reader.booleans(reader.uint64());
                this.lowerPad = new cntk_v1.TensorShape(reader);
                this.upperPad = new cntk_v1.TensorShape(reader);
                this.poolKind = reader.enum();
                this.imageLayoutKind = reader.enum();
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
                this.imageLayoutKind = reader.enum();
                this.autoPadding = [ reader.boolean() ];
                this.maxTempMemSizeInSamples = reader.uint64();
                this.poolKind = 'None';
                this.convolution2D = true;
                this.sharing = [ true ];
                this.lowerPad = new cntk_v1.TensorShape([ 0 ]);
                this.upperPad = new cntk_v1.TensorShape([ 0 ]);
            }
            else {
                this.convolution2D = reader.boolean();
                if (version >= 18) {
                    this.dilation = new cntk_v1.TensorShape(reader);
                }
                else {
                    this.dilation = new cntk_v1.TensorShape([ 1 ]);
                }
            }
        };
        op.Pooling = function(reader, version) {
            op.ConvolutionBase.apply(this, [ reader, version ]);
        };
        op.PoolingBase = function(reader) {
            this.imageLayoutKind = reader.enum();
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
            this.poolKind = (version < 26) ? 'Max' : reader.enum();
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
                this.imageLayoutKind = reader.enum();
                if (version >= 13) {
                    if (version != 19) {
                        this.runCountUntied = reader.uint64();
                    }
                    else {
                        this.runCountUntied = reader.boolean() ? 0 : 'SIZE_MAX'; // TODO
                    }
                }
                else {
                    mbCount = reader.uint64();
                }
                this.epsilon = reader.float64();
                this.useCntkEngine = reader.boolean();
            }
            else {
                let verWritten = reader.int32();
                let verReadable = reader.int32();
                if (verReadable > verWritten || verWritten < 0x00010001 || verReadable > 0x00010004) {
                    throw new cntk.Error('BatchNormalization version not supported.');
                }
                this.eval = reader.boolean();
                this.spatial = reader.boolean();
                if (verWritten >= 0x00010004) {
                    this.normalizationTimeConstant = reader.float64();
                }
                else {
                    reader.float64(); // expAvgFactor
                }
                if (verWritten >= 0x00010002) {
                    this.imageLayoutKind = reader.enum();
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
            }
            else {
                let rows = reader.uint64();
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
            }
            else {
                let rows = reader.uint64();
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
                    let size = reader.uint64();
                    this.perm = [];
                    for (let i = 0; i < size; i++) {
                        this.perm.push(reader.uint64());
                    }
                }
            }
            else {
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

        let nodes = [];
        this.nodes = {};
        for (let i = 0; i < numNodes; i++) {
            const precision = this.version >= 7 ? reader.string() : '';
            if (precision != 'float' && precision != 'double' && precision != 'half' && precision != '') {
                throw new cntk.Error("Invalid precision format '" + precision + "'.");
            }
            let obj = { __type__: reader.string() };
            obj.name = reader.string();
            obj.precision = precision;
            const constructor = op[obj.__type__];
            if (!constructor) {
                throw new cntk.Error("Unknown operator '" + obj.__type__ + "'.");
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
            let children = [];
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
            this.feature = reader.strings(reader.uint64());
            reader.assert('EFeatureNodes');
        }
        if (reader.match('BLabelNodes')) {
            this.label = reader.strings(reader.uint64());
            reader.assert('ELabelNodes');
        }
        if (reader.match('BCriterionNodes')) {
            this.criterion = reader.strings(reader.uint64());
            reader.assert('ECriterionNodes');
        }
        if (this.criterion.length == 0) {
            if (reader.match('BCriteriaNodes')) {
                this.criterion = reader.strings(reader.uint64());
                reader.assert('ECriteriaNodes');
            }
        }
        if (reader.match('BNodesReqMultiSeqHandling')) {
            reader.strings(reader.uint64());
            reader.assert('ENodesReqMultiSeqHandling');
        }
        if (reader.match('BEvalNodes')) {
            this.eval = reader.strings(reader.uint64());
            reader.assert('EEvalNodes');
        }
        if (reader.match('BOutputNodes')) {
            this.output = reader.strings(reader.uint64());
            reader.assert('EOutputNodes');
        }
        if (reader.match('BPairNodes')) {
            this.pair = reader.strings(reader.uint64());
            reader.assert('EPairNodes');
        }
        reader.assert('ERootNodes');
        reader.assert('ECN');
    }
};

cntk_v1.Reader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    match(text) {
        let position = this._position;
        for (let i = 0; i < text.length; i++) {
            if (this.uint16() != text.charCodeAt(i)) {
                this._position = position;
                return false;
            }
        }
        if (this.uint16() != 0) {
            this._position = position;
            return false;
        }
        return true;
    }

    assert(text) {
        if (!this.match(text)) {
            throw new cntk_v1.Error("Invalid '" + text + "' signature.");
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new cntk.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    boolean() {
        return this.byte() != 0 ? true : false;
    }

    booleans(count) {
        let array = [];
        for (let i = 0; i < count; i++) {
            array.push(this.boolean());
        }
        return array;
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position);
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getUint16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    uint64() {
        const low = this.uint32();
        const hi = this.uint32();
        if (hi > 65536) {
            throw new cntk_v1.Error('Value not in 48-bit range.');
        }
        return (hi << 32) | low;
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._dataView.getFloat64(position, true);
    }

    string() {
        let text = '';
        let c = this.uint16();
        while (c != 0) {
            text += String.fromCharCode(c);
            c = this.uint16();
        }
        return text;
    }

    strings(count) {
        let array = [];
        for (let i = 0; i < count; i++) {
            array.push(this.string());
        }
        return array;
    }

    enum() {
        return this.int32();
    }
};

cntk_v1.TensorShape = class {

    constructor(reader, acceptLegacyFormat = false) {
        if (reader && Array.isArray(reader)) {
            this.dims = reader;
            return;
        }
        this.dims = [];
        let rank = reader.uint32();
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
        }
        else {
            let dim = reader.uint32();
            this.dims.push(reader.uint32());
            this.dims.push(rank);
            this.dims.push(dim);
        }
    }
};

cntk_v1.Matrix = class {

    constructor(reader) {
        let type = reader.byte();
        switch (type) {
            case 100: { 
                // dense
                reader.assert('BMAT');
                const elsize = reader.uint64();
                this.name = reader.string();
                this.format = reader.uint32();
                this.rows = reader.uint64();
                this.columns = reader.uint64();
                reader.bytes(elsize * this.rows * this.columns);
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