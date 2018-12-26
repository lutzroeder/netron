/*jshint esversion: 6 */

var caffe2 = caffe2 || {};
var protobuf = protobuf || require('protobufjs');
var marked = marked || require('marked');

caffe2.ModelFactory = class {

    match(context, host) {
        var identifier = context.identifier.toLowerCase();
        var extension = identifier.split('.').pop().toLowerCase();
        var tags = null;
        if (extension == 'pb') {
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
                return true;
            }
            tags = context.tags('pb');
            // ignore input_0.pb, output_0.pb
            if (Object.keys(tags).length > 0 &&
                tags.hasOwnProperty(1) && tags[1] == 0 && 
                tags.hasOwnProperty(2) && tags[2] == 0 && 
                tags.hasOwnProperty(9) && tags[9] == 2) {
                return false;
            }
            if (Object.keys(tags).length > 0 &&
                (!tags.hasOwnProperty(1) || tags[1] == 2) &&
                (!tags.hasOwnProperty(2) || tags[2] == 2) &&
                (!tags.hasOwnProperty(7) || tags[7] == 2) &&
                (!tags.hasOwnProperty(8) || tags[8] == 2)) {
                var buffer = context.buffer;
                if (buffer.length > 3 && buffer[0] == 0x0A) {
                    var size = buffer[1];
                    if (size < 64 && buffer.length > 2 + size + 1 && buffer.slice(2, 2 + size).every((c) => c >= 32 && c <= 127) && buffer[2 + size] == 0x12) {
                        return true;
                    }
                }
                if (buffer.length > 3 && buffer[0] == 0x12) {
                    return true;
                }
            }
        }
        if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
            identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
            tags = context.tags('pbtxt');
            if (tags.op) {
                return true;
            }
        }
        return false;
    }    

    open(context, host, callback) {
        host.require('./caffe2-proto', (err, module) => {
            if (err) {
                callback(err, null);
                return;
            }
            var netDef = null;
            var identifier = context.identifier; 
            var extension = identifier.split('.').pop().toLowerCase();
            if (extension == 'pbtxt' || extension == 'prototxt') {
                try {
                    caffe2.proto = protobuf.roots.caffe2.caffe2;
                    var reader = new protobuf.TextReader(context.text);
                    reader.handle = function(tag, message) {
                        if (message instanceof caffe2.proto.DeviceOption) {
                            message[tag] = this.skip();
                            return;
                        }
                        throw new Error("Unknown field '" + tag + "'" + this.location());
                    };
                    netDef = caffe2.proto.NetDef.decodeText(reader);
                }
                catch (error) {
                    callback(new caffe2.Error("File text format is not caffe2.NetDef (" + error.message + ") in '" + identifier + "'."), null);
                    return;
                }    
            }
            else {
                try {
                    caffe2.proto = protobuf.roots.caffe2.caffe2;
                    netDef = caffe2.proto.NetDef.decode(context.buffer);
                }
                catch (error) {
                    callback(new caffe2.Error("File format is not caffe2.NetDef (" + error.message + ") in '" + identifier + "'."), null);
                    return;
                }    
            }
            caffe2.Metadata.open(host, (err, metadata) => {
                context.request('init_net.pb', null, (err, data) => {
                    var init = null;
                    if (!err && data) {
                        try {
                            init = caffe2.proto.NetDef.decode(data);
                        }
                        catch (error) {
                        }
                    }

                    var model = null;
                    try {
                        model = new caffe2.Model(metadata, netDef, init);
                    }
                    catch (error) {
                        host.exception(error, false);
                        callback(new caffe2.Error(error.message), null);
                        return;
                    }
                    callback(null, model);
                }); 
            });
        });
    }

};

caffe2.Model = class {

    constructor(metadata, netDef, init) {
        this._domain = netDef.domain || null;
        var graph = new caffe2.Graph(metadata, netDef, init);
        this._graphs = [ graph ];
    }

    get format() {
        return 'Caffe2';
    }

    get domain() {
        return this._domain;
    }

    get graphs() {
        return this._graphs;
    }
};

caffe2.Graph = class {

    constructor(metadata, netDef, init) {
        this._name = netDef.name || '';
        this._type = netDef.type || '';
        this._nodes = [];
        this._operators = {};

        var initializers = {};
        netDef.external_input.forEach((input) => {
            initializers[input] = {};
        });
        if (init) {
            init.op.forEach((op) => {
                if (op.output && op.output.length == 1) {
                    var name = op.output[0];
                    var dataType = null;
                    switch (op.type) {
                        case 'GivenTensorFill':
                            dataType = 'float32';
                            break;
                        case 'GivenTensorBoolFill':
                            dataType = 'boolean';
                            break;
                        case 'GivenTensorByteStringToUInt8Fill':
                            dataType = 'uint8';
                            break;
                        case 'GivenTensorIntFill':
                            dataType = 'int32';
                            break;
                        case 'GivenTensorInt64Fill':
                            dataType = 'int64';
                            break;
                        case 'GivenTensorStringFill':
                            dataType = 'string';
                            break;
                        case 'Int8GivenIntTensorFill':
                            dataType = 'int32';
                            break;
                        case 'Int8GivenTensorFill':
                            dataType = 'int8';
                            break;
                        default:
                            debugger;
                            break;
                    }
                    if (dataType) {
                        op.dataType = dataType;
                        initializers[name] = op;
                    }    
                }
            });
        }

        var scope = {};
        netDef.op.forEach((op, index) => {
            op.input = op.input.map((input) => scope[input] ? scope[input] : input);
            op.output = op.output.map((output) => {
                if (scope[output]) {
                    var next = output + '\n' + index.toString(); // custom connection id
                    scope[output] = next;
                    return next;
                }
                scope[output] = output;   
                return output;
            });
        });

        var lastNode = null;
        var lastOutput = null;
        netDef.op.forEach((op) => {
            this._operators[op.type] = (this._operators[op.type] || 0) + 1;
            var node = new caffe2.Node(metadata, op, initializers);
            if (op.input.length == 1 &&
                op.output.length >= 1 && 
                op.input[0].split('\n').shift() == op.output[0].split('\n').shift() && 
                lastNode &&
                lastOutput == op.input[0].split('\n').shift()) {
                lastNode.chain.push(node);
            }
            else {
                this._nodes.push(node);
                lastNode = null;
                lastOutput = null;
                if (op.output.length == 1) {
                    lastNode = node;
                    lastOutput = op.output[0].split('\n').shift();
                }
            }
        });

        this._inputs = [];
        var inputs = Object.keys(initializers);
        inputs.forEach((input) => {
            if (inputs.length == 1 || !input.startsWith('caffe.')) {
                this._inputs.push(new caffe2.Argument(input, [ new caffe2.Connection(input, null, null) ]));
            }
        });

        this._outputs = [];
        netDef.external_output.forEach((output) => {
            this._outputs.push(new caffe2.Argument(output, [ new caffe2.Connection(output, null, null) ]));
        });
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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

    get operators() {
        return this._operators;
    }

    toString() {
        return 'graph(' + this.name + ')';
    }
};

caffe2.Argument = class {
    constructor(name, connections) {
        this._name = name;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get connections() {
        return this._connections;
    }
};

caffe2.Connection = class {
    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get quantization() {
        if (this._initializer) {
            return this._initializer.quantization;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

caffe2.Node = class {

    constructor(metadata, op, initializers) {
        if (op.name) {
            this._name = op.name;
        }
        if (op.engine) {
            this._device = op.engine;
        }
        this._metadata = metadata;
        this._operator = op.type;
        this._inputs = op.input;
        this._outputs = op.output;
        this._chain = [];
        this._attributes = [];
        op.arg.forEach((arg) => {
            this._attributes.push(new caffe2.Attribute(this._metadata, this, arg));
        });

        this._initializers = {};
        this._inputs.forEach((input, index) => {
            if (index > 0) {
                var tensor = initializers[input];
                if (tensor) {
                    this._initializers[input] = new caffe2.Tensor(input, tensor, 'Initializer');
                    delete initializers[input];
                }
            }
        });
    }

    get name() {
        return this._name || '';
    }

    get device() {
        return this._device || '';
    }

    get operator() {
        return this._operator;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : null;
    }

    get documentation() {
        var schema = this._metadata.getSchema(this._operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.references) {
                schema.references.forEach((reference) => {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                });
            }
            return schema;
        }
        return '';
    }

    get inputs() {
        var inputs = this._metadata.getInputs(this._operator, this._inputs);
        return inputs.map((input) => {
            return new caffe2.Argument(input.name, input.connections.map((connection) => {
                return new caffe2.Connection(connection.id, null, this._initializers[connection.id]);
            }));
        });
    }

    get outputs() {
        var outputs = this._metadata.getOutputs(this._operator, this._outputs);
        return outputs.map((output) => {
            return new caffe2.Argument(output.name, output.connections.map((connection) => {
                return new caffe2.Connection(connection.id, null, null);
            }));
        });
    }

    get attributes() {
        return this._attributes;
    }

    get chain() {
        return this._chain;
    }
};

caffe2.Attribute = class {

    constructor(metadata, node, arg) {
        this._node = node;
        this._name = arg.name;
        if (arg.floats && arg.floats.length > 0) {
            this._value = arg.floats;
        }
        else if (arg.ints && arg.ints.length > 0) {
            this._value = arg.ints;
        }
        else if (arg.nets && arg.nets.length > 0) {
            this._value = arg.nets.map((net) => new caffe2.Graph(metadata, net, null));
            this._type = 'graph[]';
        }
        else if (arg.n) {
            this._value = new caffe2.Graph(metadata, arg.n, null);
            this._type = 'graph';
        }
        else if (arg.i != 0) {
            this._value = arg.i;
        }
        else {
            this._value = arg.i;
        }

        var schema = metadata.getAttributeSchema(this._node.operator, this._name);
        if (schema) {
            if (schema.hasOwnProperty('type')) {
                this._type = schema.type;
            }
            if (schema.hasOwnProperty('visible') && !schema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default')) {
                if (this._value == schema.default.toString()) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type || null;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

caffe2.Tensor = class {

    constructor(name, tensor, kind) {
        this._name = name;
        this._kind = kind;

        var args = {};
        if (tensor && tensor.arg) {
            tensor.arg.forEach((arg) => {
                args[arg.name] = arg;
            });
        }
        var shape = null;
        if (args.shape && args.shape.ints) {
            shape = args.shape.ints;
        }
        if (args.values) {
            this._values = args.values;
        }
        this._scale = args.hasOwnProperty('Y_scale') ? args.Y_scale.f : 0;
        this._zeroPoint = args.hasOwnProperty('Y_zero_point') ? args.Y_zero_point.i : 0;
        this._type = new caffe2.TensorType(tensor.dataType, new caffe2.TensorShape(shape));
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get kind() {
        return this._kind;
    }

    get quantization() {
        if (this._scale != 0 || this._zeroPoint != 0) {
            return this._scale.toString() + ' * ' + (this._zeroPoint == 0 ? 'q' : ('(q - ' + this._zeroPoint.toString() + ')'));
        }
        return null;
    }

    get state() {
        return this._context().state;
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
        return caffe2.Tensor._stringify(value, '', '    ');
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        if (!this._values) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        if (this._values.floats == -1) {
            context.state = 'Tensor data is too large to load in Chrome.';
            return context;
        }
        switch (this._type.dataType) {
            case 'float32':
                context.data = this._values.floats;
                break;
            case 'boolean':
                context.data = this._values.ints;
                break;
            case 'int8':
                context.data = this._values.s;
                break;
            case 'int32':
                context.data = this._values.ints;
                break;
            default:
                context.state = 'Unknown data type.';
                debugger;
                return context;
        }
        context.shape = this._type.shape.dimensions;
        context.dataType = this._type.dataType;
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.shape[dimension];
        if (dimension == context.shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float32':
                        results.push(context.data[context.index]);
                        break;
                    case 'boolean':
                        results.push(context.data[context.index] == 0 ? false : true);
                        break;
                    case 'int8':
                        results.push(context.data[context.index]);
                        break;
                    case 'int32':
                        results.push(context.data[context.index]);
                        break;
                    default:
                        context.state = 'Unknown data type.';
                        debugger;
                        break;
                }
                context.index++;
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
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push(indentation + '[');
            var items = value.map((item) => caffe2.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

caffe2.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType || '?';
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

caffe2.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

caffe2.Metadata = class {

    static open(host, callback) {
        if (caffe2.Metadata._metadata) {
            callback(null, caffe2.Metadata._metadata);
        }
        else {
            host.request(null, 'caffe2-metadata.json', 'utf-8', (err, data) => {
                caffe2.Metadata._metadata = new caffe2.Metadata(data);
                callback(null, caffe2.Metadata._metadata);
            });
        }    
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getInputs(type, inputs) {
        var results = [];
        var index = 0;
        var schema = this.getSchema(type);
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputDef) => {
                if (index < inputs.length || inputDef.option != 'optional') {
                    var input = {};
                    input.name = inputDef.name;
                    input.type = inputDef.type;
                    var count = (inputDef.option == 'variadic') ? (inputs.length - index) : 1;
                    input.connections = [];
                    inputs.slice(index, index + count).forEach((id) => {
                        if (id != '' || inputDef.option != 'optional') {
                            input.connections.push({ id: id});
                        }
                    });
                    index += count;
                    results.push(input);
                }
            });
        }
        else {
            inputs.slice(index).forEach((input) => {
                var name = (index == 0) ? 'input' : ('(' + index.toString() + ')');
                results.push({
                    name: name,
                    connections: [ { id: input } ]
                });
                index++;
            });

        }
        return results;
    }

    getOutputs(type, outputs) {
        var results = [];
        var index = 0;
        var schema = this.getSchema(type);
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputDef) => {
                if (index < outputs.length || outputDef.option != 'optional') {
                    var output = {};
                    output.name = outputDef.name;
                    var count = (outputDef.option == 'variadic') ? (outputs.length - index) : 1;
                    output.connections = outputs.slice(index, index + count).map((id) => {
                        return { id: id };
                    });
                    index += count;
                    results.push(output);
                }
            });
        }
        else {
            outputs.slice(index).forEach((output) => {
                var name = (index == 0) ? 'output' : ('(' + index.toString() + ')');
                results.push({
                    name: name,
                    connections: [ { id: output } ]
                });
                index++;
            });

        }
        return results;
    }

    getAttributeSchema(operator, name, value) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributesMap) {
                schema.attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributesMap[attribute.name] = attribute;
                });
            }
            return schema.attributesMap[name] || null;
        }
        return null;
    }
};

caffe2.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe2 model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = caffe2.ModelFactory;
}