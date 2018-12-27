/*jshint esversion: 6 */

var paddle = paddle || {};
var protobuf = protobuf || require('protobufjs');
var base = base || require('./base');

paddle.ModelFactory = class {

    match(context, host) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop().toLowerCase();
        if (identifier == '__model__' || extension == 'paddle') {
            return true;
        }
        return false;
    }

    open(context, host, callback) {
        host.require('./paddle-proto', (err, module) => {
            if (err) {
                callback(err, null);
                return;
            }
            var desc = null;
            var identifier = context.identifier; 
            try {
                paddle.proto = protobuf.roots.paddle.paddle.framework.proto;
                desc = paddle.proto.ProgramDesc.decode(context.buffer);
            }
            catch (error) {
                callback(new paddle.Error("File format is not paddle.ProgramDesc (" + error.message + ") in '" + identifier + "'."), null);
                return;
            }
            paddle.Metadata.open(host, (err, metadata) => {
                try {
                    var model = new paddle.Model(metadata, desc);
                    callback(null, model);
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new paddle.Error(error.message), null);
                    return;
                }
            });
        });
    }
};

paddle.Model = class {

    constructor(metadata, desc) {
        this._graphs = [];
        desc.blocks.forEach((block) => {
            this._graphs.push(new paddle.Graph(metadata, block));
        });
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return 'PaddlePaddle';
    }
};

paddle.Graph = class {

    constructor(metadata, block) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        var initializers = {};
        var types = {};
        block.vars.forEach((variable) => {
            if (variable.persistable && variable.type && 
                variable.type.type != paddle.proto.VarType.Type.FETCH_LIST && 
                variable.type.type != paddle.proto.VarType.Type.FEED_MINIBATCH) {
                initializers[variable.name] = new paddle.Tensor(variable);
            }
            else {
                types[variable.name] = paddle.Graph._type(variable);
            }
        });

        var scope = {};
        block.ops.forEach((op, index) => {
            op.inputs.forEach((input) => {
                input.arguments = input.arguments.map((argument) => scope[argument] ? scope[argument] : argument);
            });
            op.outputs.forEach((output) => {
                output.arguments = output.arguments.map((argument) => {
                    if (scope[argument]) {
                        var next = argument + '\n' + index.toString(); // custom connection id
                        scope[argument] = next;
                        return next;
                    }
                    scope[argument] = argument;   
                    return argument;
                });
            });
        });

        var lastNode = null;
        var lastOutput = null;
        block.ops.forEach((op) => {
            if (op.type == 'feed') {
                var inputName = op.attrs.filter((attr) => attr.name == 'col')[0].i.toString();
                this._inputs.push(new paddle.Argument(inputName, op.outputs[0].arguments.map((id) => {
                    return new paddle.Connection(id, types[id], null, null);
                })));
            }
            else if (op.type == 'fetch') {
                var outputName = op.attrs.filter((attr) => attr.name == 'col')[0].i.toString();
                this._outputs.push(new paddle.Argument(outputName, op.inputs[0].arguments.map((id) => {
                    return new paddle.Connection(id, types[id], null, null);
                })));
            }
            else {
                var node = new paddle.Node(metadata, op, initializers, types);
                if (op.inputs.length == 1 && op.inputs[0].arguments.length == 1 &&
                    op.outputs.length >= 1 && op.outputs[0].arguments.length == 1 &&
                    op.inputs[0].arguments[0].split('\n').shift() == op.outputs[0].arguments[0].split('\n').shift() && 
                    lastNode &&
                    lastOutput == op.inputs[0].arguments[0].split('\n').shift()) {
                    lastNode.chain.push(node);
                }
                else {
                    this._nodes.push(node);
                    lastNode = null;
                    lastOutput = null;
                    if (op.outputs.length == 1 && op.outputs[0].arguments.length == 1) {
                        lastNode = node;
                        lastOutput = op.outputs[0].arguments[0].split('\n').shift();
                    }
                }
            }
        });
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

    static _type(variable) {
        switch (variable.type.type) {
            case paddle.proto.VarType.Type.LOD_TENSOR:
                return new paddle.TensorType(variable.type.lod_tensor.tensor);
            default:
                debugger;
                break;
        }
        return null;
    }
};


paddle.Argument = class {
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

paddle.Connection = class {

    constructor(id, type, description, initializer) {
        this._id = id;
        this._type = type || null;
        this._description = description || null;
        this._initializer = initializer || null;
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
        return this._description;
    }

    get initializer() {
        return this._initializer;
    }
};

paddle.Node = class {

    constructor(metadata, op, initializers, types) {
        this._metadata = metadata;
        this._operator = op.type;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        op.attrs.forEach((attr) => {
            this._attributes.push(new paddle.Attribute(metadata, this._operator, attr));
        });
        op.inputs.forEach((input) => {
            if (input.arguments.length > 0) {
                var connections = input.arguments.map((argument) => new paddle.Connection(argument, types[argument.split('\n').shift()], null, initializers[argument]));
                this._inputs.push(new paddle.Argument(input.parameter, connections));              
            }
        });
        op.outputs.forEach((output) => {
            if (output.arguments.length > 0) {
                var connections = output.arguments.map((argument) => new paddle.Connection(argument, types[argument.split('\n').shift()], null, null));
                this._outputs.push(new paddle.Argument(output.parameter, connections));              
            }
        });
        this._update(this._inputs, 'X');
        this._update(this._inputs, 'Input');
        this._update(this._outputs, 'Y');
        this._update(this._outputs, 'Out');
    }

    get operator() {
        return this._operator;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : null;
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

    get chain() {
        return this._chain;
    }

    _update(list, name) {
        var item = null;
        for (var i = 0; i < list.length; i++) {
            if (list[i].name == name) {
                item = list[i];
                list.splice(i, 1);
                break;
            }
        }
        if (item) {
            list.splice(0, 0, item);
        }
    }
};

paddle.Attribute = class {

    constructor(metadata, operator, attr) {
        this._name = attr.name;
        this._value = '?';
        switch (attr.type) {
            case paddle.proto.AttrType.STRING:
                this._type = 'string';
                this._value = attr.s;
                break;
            case paddle.proto.AttrType.STRINGS:
                this._type = 'string[]';
                this._value = attr.strings;
                break;
            case paddle.proto.AttrType.BOOLEAN:
                this._type = 'boolean';
                this._value = attr.b;
                break;
            case paddle.proto.AttrType.BOOLEANS:
                this._type = 'boolean[]';
                this._value = attr.bools;
                break;
            case paddle.proto.AttrType.FLOAT:
                this._type = 'float32';
                this._value = attr.f;
                break;
            case paddle.proto.AttrType.FLOATS:
                this._type = 'float[]';
                this._value = attr.floats;
                break;
            case paddle.proto.AttrType.INT:
                this._type = 'int32';
                this._value = attr.i;
                break;
            case paddle.proto.AttrType.INTS:
                this._type = 'int32[]';
                this._value = attr.ints;
                break;
            case paddle.proto.AttrType.LONG:
                this._type = 'int64';
                debugger;
                break;
            case paddle.proto.AttrType.LONGS:
                this._type = 'int64[]';
                debugger;
                break;
            default:
                debugger;
                break;
        }
        switch (this._name) {
            case 'use_mkldnn':
            case 'use_cudnn':
            case 'op_callstack':
            case 'op_role':
            case 'op_role_var':
            case 'op_namescope':
            case 'is_test':
                this._visible = false;
                break;
        }

        var schema = metadata.getAttributeSchema(operator, this._name);
        if (schema) {
            if (schema.hasOwnProperty('default')) {
                if (schema.default == this._value) {
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

paddle.Tensor = class {

    constructor(variable) {
        this._type = paddle.Graph._type(variable);
    }

    get type() {
        return this._type;
    }
};

paddle.TensorType = class {

    constructor(desc) {
        switch (desc.data_type) {
            case paddle.proto.VarType.Type.INT32:
                this._dataType = 'int32';
                break;
            case paddle.proto.VarType.Type.INT64:
                this._dataType = 'int64';
                break;
            case paddle.proto.VarType.Type.FP32:
                this._dataType = 'float32';
                break;
            case paddle.proto.VarType.Type.FP64:
                this._dataType = 'float64';
                break;
            default:
                this._dataType = '?';
                debugger;
                break;
        }
        this._shape = new paddle.TensorShape(desc.dims);    
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get denotation() { 
        return this._denotation;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

paddle.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions.map((dimension) => {
            return dimension != -1 ? dimension : '?';
        });
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return (this._dimensions && this._dimensions.length) ? ('[' + this._dimensions.join(',') + ']') : '';
    }
};

paddle.Metadata = class {

    static open(host, callback) {
        if (paddle.Metadata._metadata) {
            callback(null, paddle.Metadata._metadata);
        }
        else {
            host.request(null, 'paddle-metadata.json', 'utf-8', (err, data) => {
                paddle.Metadata._metadata = new paddle.Metadata(data);
                callback(null, paddle.Metadata._metadata);
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

    getAttributeSchema(operator, name) {
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

paddle.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading PaddlePaddle model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = paddle.ModelFactory;   
}
