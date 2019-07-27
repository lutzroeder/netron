/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var cntk = cntk || {};
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');

var cntk_v1 = {};
var cntk_v2 = null;

cntk.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        var buffer = null;
        if (extension == 'model' || extension == 'cmf' || extension == 'dnn' || extension == 'cntk') {
            buffer = context.buffer;
            // Reject PyTorch models with .model file extension.
            var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
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
            var tags = context.tags('pb');
            if (tags.get(1) === 0 && tags.get(2) === 2) {
                return true;
            }
            return false;
        }
    }

    open(context, host) { 
        return host.require('./cntk-proto').then(() => {
            var version = 0;
            var obj = null;
            try {
                var buffer = context.buffer;
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
                    var dictionary = cntk_v2.Dictionary.decode(context.buffer);
                    obj = cntk.ModelFactory._convertDictionary(dictionary);
                    version = 2;
                }
            }
            catch (error) {
                throw new new cntk.Error("File format is not cntk.Dictionary (" + error.message + ") in '" + context.identifier + "'.");
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
        var target = {};
        for (var key of Object.keys(dictionary.data).filter((key) => key != 'version')) {
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

        var name;
        var node;
        var args = {};
        switch (version) {
            case 1:
                for (name of Object.keys(obj.nodes)) {
                    node = obj.nodes[name];
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
                for (name of Object.keys(obj.nodes)) {
                    node = obj.nodes[name];
                    if (node.__type__ != 'InputValue' && node.__type__ != 'LearnableParameter') {
                        this._nodes.push(new cntk.Node(metadata, version, node, args));
                    }
                }
                if (obj.output) {
                    for (var output of obj.output) {
                        this._outputs.push(new cntk.Parameter(output, [ 
                            new cntk.Argument(version, output)
                        ]));
                    }
                }
                break;
            case 2:
                var nodeMap = {};
                for (node of obj.primitive_functions) {
                    nodeMap[node.uid] = node;
                }
                var argumentNames = {};
                for (var input of obj.inputs) {
                    var argument = new cntk.Argument(version, input);
                    args[input.uid] = argument;
                    // VariableKind { 0: 'input', 1: 'output', 2: 'parameter', 3: 'constant', 4: 'placeholder' }
                    if (input.kind == 0) {
                        var inputName = input.name || input.uid;
                        this._inputs.push(new cntk.Parameter(inputName, [ argument ]));
                    }
                    argumentNames[input.uid] = input;
                }
                for (var block of obj.primitive_functions) {
                    if (block.op == 57 && block.block_function_composite) {
                        var list = [ block.block_function_composite.root ];
                        var nodes = [];
                        while (list.length > 0) {
                            name = list.shift();
                            node = nodeMap[name];
                            if (node) {
                                nodes.push(new cntk.Node(metadata, version, node, args));
                                nodeMap[name] = null;
                                for (var i = 0; i < node.inputs.length; i++) {
                                    var parts = node.inputs[i].split('_');
                                    if (parts.length >= 3) {
                                        parts.pop();
                                        if (parts.pop() == 'Output') {
                                            list.push(parts.join('_'));
                                        }
                                    }
                                }
                            }
                        }
                        var inputs = [];
                        var outputs = [ block.block_function_composite.root ];
                        this._functions.push(new cntk.Function(block.block_function_op_name, nodes, inputs, outputs));
                    }
                }
                for (node of obj.primitive_functions) {
                    if (nodeMap[node.uid]) {
                        this._nodes.push(new cntk.Node(metadata, version, node, args));
                    }
                }
                break;
            default:
                throw new new cntk.Error("Unsupported graph version '" + version + "'.");
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
            this._id = obj;
        }
        else {
            switch (version) {
                case 1:
                    switch (obj.__type__) {
                        case 'InputValue':
                            this._id = obj.name;
                            this._type = new cntk.TensorType(version, obj.precision, obj.sampleLayout);
                            this._initializer = null;
                            break;
                        case 'LearnableParameter':
                            this._id = obj.name;
                            this._type = null;
                            this._initializer = new cntk.Tensor(version, obj);
                            break;
                    }
                    break;
                case 2:
                    if (obj.value) {
                        this._id = obj.name || obj.uid;
                        this._type = null;
                        this._initializer = new cntk.Tensor(version, obj);
                    }
                    else {
                        this._id = obj.uid;
                        this._type = new cntk.TensorType(version, obj.data_type, obj.shape);
                        this._initializer = null;
                    }    
                    break;
            }
        }
    }

    get id() {
        return this._id;
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

        var inputs = [];
        var outputs = [];
        var initializers = [];

        var attributeName;

        switch (version) {
            case 1:
                this._operator = obj.__type__;
                this._name = obj.name;
                for (attributeName of Object.keys(obj)) {
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
            case 2:
                this._name = obj.name || obj.uid || null;
                var output = obj.uid;
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
                            for (attributeName of Object.keys(obj.user_defined_state)) {
                                this._attributes.push(new cntk.Attribute(this._metadata, this._operator, attributeName, obj.user_defined_state[attributeName]));
                            }
                        }
                    }
                    else {
                        this._operator = this._metadata.getOperatorName(obj.op);
                        if (this._operator == null) {
                            this._operator = obj.op ? obj.op.toString() : '?';
                        }
                    }
                }
                if (obj.attributes) {
                    for (attributeName of Object.keys(obj.attributes)) {
                        this._attributes.push(new cntk.Attribute(this._metadata, this._operator, attributeName, obj.attributes[attributeName]));
                    }
                }
                for (var input of obj.inputs) {
                    var argument = args[input];
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
        }

        var inputIndex = 0;
        var schema = this._metadata.getSchema(this._function ? ('Function:' + this._operator) : this._operator);
        if (schema && schema.inputs) {
            for (var inputSchema of schema.inputs) {
                if (inputIndex < inputs.length || inputSchema.option != 'optional') {
                    var inputCount = (inputSchema.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    var inputArguments = [];
                    for (var inputArgument of inputs.slice(inputIndex, inputIndex + inputCount)) {
                        if (inputArgument.id != '' || inputSchema.option != 'optional') {
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

        var outputIndex = 0;
        if (schema && schema.outputs) {
            for (var outputSchema of schema.outputs) {
                if (outputIndex < outputs.length || outputSchema.option != 'optional') {
                    var outputCount = (outputSchema.option == 'variadic') ? (outputs.length - outputIndex) : 1;
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

    get category() {
        var schema = this._metadata.getSchema(this._function ? ('Function:' + this._operator) : this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() { 
        return '';
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
            var axis = { __type__: 'Axis' };
            for (var key of Object.keys(value).filter((key) => key !== 'name')) {
                axis[key] = value[key];
            }
            this._value = axis;
        }

        var schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
                var type = cntk_v1[this._type] || cntk_v2[this._type];
                if (type && type[this._value]) {
                    this._value = type[this._value];
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                var defaultValue = schema.default;
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
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
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

        var value = this._value;
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
        var shape = context.shape;
        if (context.shape.length == 0) {
            shape = [ 1 ];
        }
        var results = [];
        var size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.data[context.index++]);
                context.count++;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
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
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema)
                    {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                        if (Object.prototype.hasOwnProperty.call(schema, 'operator')) {
                            this._operatorMap[schema.operator.toString()] = name;
                        }
                    }
                }
            }
        }
    }

    getOperatorName(code) {
        // cntk/Source/CNTKv2LibraryDll/API/Internals/PrimitiveOpType.h
        return this._operatorMap[code] || null;
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
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
        var reader = new cntk_v1.Reader(buffer);
        reader.assert('BCN');
        reader.assert('BVersion');
        this.version = reader.uint64();
        reader.version = this.version;
        reader.assert('EVersion');
        var numNodes = reader.uint64();
        reader.assert('BNodeList');
        var op = {};
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
        op.InputValue = function(reader) {
            this.rows = reader.uint64();
            this.cols = reader.uint64();
            this.sampleLayout = new cntk_v1.TensorShape(reader, true);
            this.dynamicAxisNodeName = '';
            if (reader.version >= 8) {
                var nrAxes = reader.uint32();
                if (nrAxes == 1) {
                    this.dynamicAxisNodeName = reader.string();
                }
            }
            this.learningRateMultiplier = 0;
            if (reader.version >= 10) {
                this.learningRateMultiplier = reader.float32();
            }
        };
        op.LearnableParameter = function(reader) {
            if (reader.version >= 3) {
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
                reader.seek(-4);
            }
        };
        op.Times = function(reader) {
            this.outputRank = (reader.version >= 3) ? reader.uint64() : 1;
            this.inferInputRankToMap = (reader.version >= 12) ? reader.int32() : -1;
        };
        op.Dropout = function(reader) {
            if (reader.version >= 16) {
                this.rngSeed = (reader.version == 16) ? reader.uint32() : reader.uint64();
                this.rngOffset = reader.uint64();
            }
        };
        op.ConvolutionBase = function(reader) {
            if (reader.version >= 5)
            {
                this.kernelShape = new cntk_v1.TensorShape(reader);
                this.mapCount = new cntk_v1.TensorShape(reader);
                this.strides = new cntk_v1.TensorShape(reader);
                this.sharing = reader.bools(reader.uint64());
                this.autoPadding = reader.bools(reader.uint64());
                this.lowerPad = new cntk_v1.TensorShape(reader);
                this.upperPad = new cntk_v1.TensorShape(reader);
                this.poolKind = reader.enum();
                this.imageLayoutKind = reader.enum();
                this.maxTempMemSizeInSamples = reader.uint64();
            }
            if (reader.version >= 9) {
                this.transpose = reader.bool();
            }
            if (reader.version >= 20) {
                this.outputShape = new cntk_v1.TensorShape(reader);
            }
            if (reader.version >= 21) {
                this.ceilOutDim = reader.bool();
            }
            if (reader.version >= 23) {
                this.includePad = reader.bool();
            }
        };
        op.Convolution = function(reader) {
            op.ConvolutionBase.apply(this, [ reader ]);
            if (reader.version < 5) {
                this.kernelShape = new cntk_v1.TensorShape([ reader.uint64(), reader.uint64(), 1 ]);
                this.strides = new cntk_v1.TensorShape([ reader.uint64(), reader.uint64(), 1 ]);
                this.mapCount = new cntk_v1.TensorShape([ reader.uint32() ]);
                this.imageLayoutKind = reader.enum();
                this.autoPadding = [ reader.bool() ];
                this.maxTempMemSizeInSamples = reader.uint64();
                this.poolKind = 'None';
                this.convolution2D = true;
                this.sharing = [ true ];
                this.lowerPad = new cntk_v1.TensorShape([ 0 ]);
                this.upperPad = new cntk_v1.TensorShape([ 0 ]);
            }
            else {
                this.convolution2D = reader.bool();
                if (reader.version >= 18) {
                    this.dilation = new cntk_v1.TensorShape(reader);
                }
                else {
                    this.dilation = new cntk_v1.TensorShape([ 1 ]);
                }
            }
        };
        op.Pooling = function(reader) {
            op.ConvolutionBase.apply(this, [ reader ]);
        };
        op.PoolingBase = function(reader) {
            this.imageLayoutKind = reader.enum();
            this.windowWidth = reader.uint32();
            this.windowHeight = reader.uint64();
            this.horizontalSubsample = reader.uint64();
            this.verticalSubsample = reader.uint64();
        };
        op.MaxPooling = function(reader) {
            op.PoolingBase.apply(this, [ reader ]);
        };
        op.ROIPooling = function(reader) {
            this.roiOutputShape = new cntk_v1.TensorShape(reader);
            this.poolKind = (reader.version < 26) ? 'Max' : reader.enum();
            this.spatialScale = (reader.version < 26) ? 0.0625 : reader.float64();
        };
        op.Reshape = function(reader) {
            this.beginDimParameter = reader.uint32();
            this.endDimParameter = reader.uint32();
            this.replacementSampleLayout = new cntk_v1.TensorShape(reader);
        };
        op.ReduceElements = function(reader) {
            var num_axes = 1;
            if (reader.version >= 27) {
                num_axes = reader.uint32();
            }
            this.axes = [];
            for (var i = 0; i < num_axes; i++)
            {
                this.axes.push(reader.uint32());
            }
            this.operation = reader.string();
            if (reader.version >= 24) {
                this.keepDimensions = reader.bool();
            }
        };
        op.BatchNormalization = function(reader) {
            var mbCount = 0;
            if (reader.version >= 6)
            {
                this.spatial = reader.bool();
                this.normalizationTimeConstant = reader.float64();
                this.blendTimeConstant = reader.float64();
                this.imageLayoutKind = reader.enum();
                if (reader.version >= 13)
                {
                    if (reader.version != 19) {
                        this.runCountUntied = reader.uint64();
                    }
                    else
                    {
                        this.runCountUntied = reader.bool() ? 0 : 'SIZE_MAX'; // TODO
                    }
                }
                else {
                    mbCount = reader.uint64();
                }
                this.epsilon = reader.float64();
                this.useCntkEngine = reader.bool();
            }
            else
            {
                var verWritten = reader.int32();
                var verReadable = reader.int32();
                if (verReadable > verWritten || verWritten < 0x00010001 || verReadable > 0x00010004) {
                    throw new cntk.Error('BatchNormalization version not supported.');
                }
                this.eval = reader.bool();
                this.spatial = reader.bool();
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
                    this.useCntkEngine = reader.bool();
                }
            }
            if (reader.version < 13)
            {
                this.runCountUntied = 16 * mbCount;
                this.convertRunningVariancePending = true;
            }
        };
        op.Tanh = function() {};
        op.Sigmoid = function() {};
        op.Logistic = function() {};
        op.SquareError = function() {};
        op.ErrorPrediction = function() {};
        op.RowStack = function(reader) {
            this.spliceDim = (reader.version >= 3) ? reader.int32() : 1;
        };
        op.Slice = function(reader) {
            var num = 1;
            if (reader.version >= 22) {
                num = reader.int32();
            }
            this.index = [];
            this.axis = [];
            this.strideMultiplier = [];
            for (var i = 0; i < num; i++) {
                this.index.push([ [ reader.uint64(), reader.uint64() ] ]);
                if (reader.version >= 3) {
                    this.axis.push(reader.int32());
                }
                if (reader.version >= 27) {
                    this.strideMultiplier.push(reader.int32());
                }
            }
        };
        op.PastValue = function(reader) {
            this.timeStep = reader.int32();
            if (reader.version > 3)
            {
                this.sampleLayout = new cntk_v1.TensorShape(reader, false);
            }
            else
            {
                var rows = reader.uint64();
                reader.uint64();
                this.sampleLayout = new cntk_v1.TensorShape([ rows ], true);
            }
            if (reader.version >= 2)
            {
                this.initialStateValue = reader.int32();
            }
        };
        op.FutureValue = function(reader) {
            this.timeStep = reader.int32();
            if (reader.version > 3)
            {
                this.sampleLayout = new cntk_v1.TensorShape(reader, false);
            }
            else
            {
                var rows = reader.uint64();
                reader.uint64();
                this.sampleLayout = new cntk_v1.TensorShape([ rows ], true);
            }
            if (reader.version >= 2)
            {
                this.initialStateValue = reader.int32();
            }
        };
        op.TransposeDimensions = function(reader) {
            if (reader.version >= 3) 
            {
                this.axis1 = reader.int32();
                this.axis2 = reader.int32();
                if (reader.version >= 25 && this.axis1 == 0 && this.axis2 == 0)
                {
                    var size = reader.uint64();
                    this.perm = [];
                    for (var i = 0; i < size; i++) {
                        this.perm.push(reader.uint64());
                    }
                }
            }
            else {
                this.axis1 = 1;
                this.axis2 = 2;
            }
        };
        op.AveragePooling = function(reader) {
            op.PoolingBase.apply(this, [ reader ]);
        };
        op.InvStdDev = function(reader) {
            this.hasComputed = reader.bool();
            this.value = new cntk_v1.Matrix(reader);
        };
        op.Mean = function(reader) {
            this.hasComputed = reader.bool();
            this.value = new cntk_v1.Matrix(reader);
        };
        op.PerDimMeanVarNormalization = function() {};
        op.Softmax = function() {};

        var nodes = [];
        this.nodes = {};
        for (var i = 0; i < numNodes; i++) {
            var precision = reader.version >= 7 ? reader.string() : '';
            if (precision != 'float' && precision != 'double' && precision != 'half' && precision != '') {
                throw new cntk.Error("Invalid precision format '" + precision + "'.");
            }
            var obj = { __type__: reader.string() };
            obj.name = reader.string();
            obj.precision = precision;
            var constructor = op[obj.__type__];
            if (!constructor) {
                throw new cntk.Error("Unknown operator '" + obj.__type__ + "'.");
            } 
            constructor.apply(obj, [ reader ]);
            nodes.push(obj);
            this.nodes[obj.name] = obj;
        }
        reader.assert('ENodeList');
        reader.assert('BRelation');
        for (var j = 0; j < numNodes; j++) {
            var nodeName = reader.string();
            var node = this.nodes[nodeName];
            var numChildren = reader.uint64();
            var children = [];
            for (var k = 0; k < numChildren; k++) {
                children.push(reader.string());
            }
            if (this.version < 19 && node.__type__ == 'BatchNormalization') {
                var runSampleCount = {
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
        this._offset = 0;
    }

    set version(value) {
        this._version = value;
    }

    get version() {
        return this._version;
    }

    match(text) {
        var offset = this._offset;
        for (var i = 0; i < text.length; i++) {
            if (this.uint16() != text.charCodeAt(i)) {
                this._offset = offset;
                return false;
            }
        }
        if (this.uint16() != 0) {
            this._offset = offset;
            return false;
        }
        return true;
    }

    assert(text) {
        if (!this.match(text)) {
            throw new cntk_v1.Error("Invalid '" + text + "' signature.");
        }
    }

    seek(offset) {
        this._offset += offset;
    }

    bool() {
        return this.byte() != 0 ? true : false;
    }

    bools(count) {
        var array = [];
        for (var i = 0; i < count; i++) {
            array.push(this.bool());
        }
        return array;
    }

    byte() {
        var value = this._dataView.getUint8(this._offset);
        this._offset++;
        return value;
    }

    bytes(count) {
        var data = this._buffer.subarray(this._offset, this._offset + count);
        this._offset += count;
        return data;
    }

    uint16() {
        var value = this._dataView.getUint16(this._offset, true);
        this._offset += 2;
        return value;
    }

    int32() {
        var value = this._dataView.getInt32(this._offset, true);
        this._offset += 4;
        return value;
    }

    uint32() {
        var value = this._dataView.getUint32(this._offset, true);
        this._offset += 4;
        return value;
    }

    uint64() {
        var low = this.uint32();
        var hi = this.uint32();
        if (hi > 65536) {
            throw new cntk_v1.Error('Value not in 48-bit range.');
        }
        return (hi << 32) | low;
    }

    float32() {
        var value = this._dataView.getFloat32(this._offset, true);
        this._offset += 4;
        return value;
    }

    float64() {
        var value = this._dataView.getFloat64(this._offset, true);
        this._offset += 8;
        return value;
    }

    string() {
        var text = '';
        var c = this.uint16();
        while (c != 0) {
            text += String.fromCharCode(c);
            c = this.uint16();
        }
        return text;
    }

    strings(count) {
        var array = [];
        for (var i = 0; i < count; i++) {
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
        var rank = reader.uint32();
        var dim0 = 0;
        if (rank > 0) {
            dim0 = reader.uint32();
        }
        if (!acceptLegacyFormat || dim0 != 0) {
            this.dims.push(dim0);
            for (var i = 1; i < rank; i++) {
                this.dims.push(reader.uint32());
            }
        }
        else {
            var dim = reader.uint32();
            this.dims.push(reader.uint32());
            this.dims.push(rank);
            this.dims.push(dim);
        }
    }
};

cntk_v1.Matrix = class {

    constructor(reader) {
        var type = reader.byte();
        switch (type) {
            case 100: // dense
                reader.assert('BMAT');
                var elsize = reader.uint64();
                this.name = reader.string();
                this.format = reader.uint32();
                this.rows = reader.uint64();
                this.columns = reader.uint64();
                reader.bytes(elsize * this.rows * this.columns);
                reader.assert('EMAT');
                break;
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