/*jshint esversion: 6 */

var caffe2 = null;

class Caffe2ModelFactory {

    match(context, host) {
        var identifier = context.identifier;
        if (identifier.endsWith('predict_net.pb')) {
            return true;
        }
        if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt')) {
            if (context.text) {
                var lines = context.text.split('\n');
                if (lines.some((line) =>
                        (line.startsWith('net') && line.replace(/\s+/g, '').startsWith('net:')) ||
                        (line.startsWith('train_net') && line.replace(/\s+/g, '').startsWith('train_net:')) || 
                        (line.startsWith('net_param') && line.replace(/\s+/g, '').startsWith('net_param{')) ||
                        (line.startsWith('layer') && line.replace(/\s+/g, '').startsWith('layer{')) ||
                        (line.startsWith('layers') && line.replace(/\s+/g, '').startsWith('layers{')) || 
                        (line.startsWith('ir_version') && line.replace(/\s+/g, '').startsWith('ir_version:')) ||
                        (line.startsWith('node') && line.replace(/\s+/g, '').startsWith('node{')))) {
                    return false;
                }
            }
            return host.environment('PROTOTXT') ? true : false;
        }
        return false;
    }    

    open(context, host, callback) {
        host.require('caffe2', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var netDef = null;
            var extension = context.identifier.split('.').pop();
            if (extension == 'pbtxt' || extension == 'prototxt') {
                try {
                    caffe2 = protobuf.roots.caffe2.caffe2;
                    netDef = caffe2.NetDef.decodeText(context.text);
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new Caffe2Error('File text format is not caffe2.NetDef (' + error.message + ').'), null);
                    return;
                }    
            }
            else {
                try {
                    caffe2 = protobuf.roots.caffe2.caffe2;
                    netDef = caffe2.NetDef.decode(context.buffer);
                }
                catch (error) {
                    callback(new Caffe2Error('File format is not caffe2.NetDef (' + error.message + ').'), null);
                    return;
                }    
            }
            Caffe2OperatorMetadata.open(host, (err, metadata) => {
                context.request('init_net.pb', null, (err, data) => {
                    var init = null;
                    if (!err && data) {
                        try {
                            init = caffe2.NetDef.decode(data);
                        }
                        catch (error) {
                        }
                    }

                    var model = null;
                    try {
                        model = new Caffe2Model(netDef, init);
                    }
                    catch (error) {
                        host.exception(error, false);
                        callback(new Caffe2Error(error.message), null);
                        return;
                    }
                    callback(null, model);
                }); 
            });
        });
    }

}

class Caffe2Model {

    constructor(netDef, init) {
        var graph = new Caffe2Graph(netDef, init);
        this._graphs = [ graph ];
    }

    get format() {
        return 'Caffe2';
    }

    get graphs() {
        return this._graphs;
    }
}

class Caffe2Graph {

    constructor(netDef, init) {
        this._name = netDef.name ? netDef.name : '';
        this._type = netDef.type ? netDef.type : '';
        this._nodes = [];
        this._operators = {};

        var initializers = {};
        netDef.external_input.forEach((input) => {
            initializers[input] = {};
        });
        if (init) {
            init.op.forEach((op) => {
                if (op.type == 'GivenTensorFill' && op.output && op.output.length == 1) {
                    var name = op.output[0];
                    initializers[name] = op;
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

        netDef.op.forEach((op) => {
            this._operators[op.type] = (this._operators[op.type] || 0) + 1;
            this._nodes.push(new Caffe2Node(op, initializers));
        });

        this._inputs = [];
        var inputs = Object.keys(initializers);
        inputs.forEach((input) => {
            if (inputs.length == 1 || !input.startsWith('caffe.')) {
                this._inputs.push(new Caffe2Argument(input, [ new Caffe2Connection(input, null, null) ]));
            }
        });

        this._outputs = [];
        netDef.external_output.forEach((output) => {
            this._outputs.push(new Caffe2Argument(output, [ new Caffe2Connection(output, null, null) ]));
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
}

class Caffe2Argument {
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
}

class Caffe2Connection {
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

    get initializer() {
        return this._initializer;
    }
}

class Caffe2Node {

    constructor(op, initializers) {
        if (op.name) {
            this._name = op.name;
        }
        if (op.engine) {
            this._device = op.engine;
        }
        this._operator = op.type;
        this._inputs = op.input;
        this._outputs = op.output;

        this._attributes = [];
        op.arg.forEach((arg) => {
            this._attributes.push(new Caffe2Attribute(this, arg));
        });

        this._initializers = {};
        this._inputs.forEach((input, index) => {
            if (index > 0) {
                var tensor = initializers[input];
                if (tensor) {
                    this._initializers[input] = new Caffe2Tensor(input, tensor, 'Initializer');
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
        var schema = Caffe2OperatorMetadata.operatorMetadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : null;
    }

    get documentation() {
        return Caffe2OperatorMetadata.operatorMetadata.getOperatorDocumentation(this._operator);
    }

    get inputs() {
        var inputs = Caffe2OperatorMetadata.operatorMetadata.getInputs(this._operator, this._inputs);
        return inputs.map((input) => {
            return new Caffe2Argument(input.name, input.connections.map((connection) => {
                return new Caffe2Connection(connection.id, null, this._initializers[connection.id]);
            }));
        });
    }

    get outputs() {
        var outputs = Caffe2OperatorMetadata.operatorMetadata.getOutputs(this._operator, this._outputs);
        return outputs.map((output) => {
            return new Caffe2Argument(output.name, output.connections.map((connection) => {
                return new Caffe2Connection(connection.id, null, null);
            }));
        });
    }

    get attributes() {
        return this._attributes;
    }
}

class Caffe2Attribute {

    constructor(node, arg) {
        this._node = node;
        this._name = arg.name;
        if (arg.floats && arg.floats.length > 0) {
            this._value = arg.floats;
        }
        else if (arg.ints && arg.ints.length > 0) {
            this._value = arg.ints;
        }
        else if (arg.nets && arg.nets.length > 0) {
            this._value = () => '...';
        }
        else if (arg.i != 0) {
            this._value = arg.i;
        }
        else {
            this._value = arg.i;
        }

        var schema = Caffe2OperatorMetadata.operatorMetadata.getAttributeSchema(this._node.operator, this._name);
        if (schema) {
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

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
}

class Caffe2Tensor {

    constructor(name, tensor, kind) {
        this._name = name;
        this._kind = kind;

        var args = {};
        if (tensor && tensor.arg) {
            tensor.arg.forEach((arg) => {
                args[arg.name] = arg;
            });
        }
        if (args.shape && args.shape.ints) {
            this._shape = args.shape.ints;
        }
        if (args.values) {
            this._values = args.values;
            if (this._values.floats || this._values.floats == -1) {
                this._dataType = 'float32';
            }
            else {
                debugger;
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return new Caffe2TensorType(this._dataType, this._shape);
    }

    get kind() {
        return this._kind;
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
        return JSON.stringify(value, null, 4);
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
        if (!this._dataType) {
            context.state = 'Unknown data type.';
            return context;
        }
        if (this._values.floats == -1) {
            context.state = 'Tensor data is too large to load in Chrome.';
            return context;
        }
        if (this._values.floats) {
            context.data = this._values.floats;
        }
        else {
            context.state = 'Unknown data format.';
        }
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.data[context.index]);
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
}

class Caffe2TensorType {

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
        return this.dataType + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }

}

class Caffe2OperatorMetadata 
{

    static open(host, callback) {
        if (Caffe2OperatorMetadata.operatorMetadata) {
            callback(null, Caffe2OperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'caffe2-metadata.json', 'utf-8', (err, data) => {
                Caffe2OperatorMetadata.operatorMetadata = new Caffe2OperatorMetadata(data);
                callback(null, Caffe2OperatorMetadata.operatorMetadata);
            });
        }    
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema)
                    {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getOperatorDocumentation(operator) {
        var schema = this.getSchema(operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = operator;
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
}

class Caffe2Error extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe2 model.';
    }
}
