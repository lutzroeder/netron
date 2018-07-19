/*jshint esversion: 6 */

// Experimental

var caffe2 = null;

class Caffe2ModelFactory {

    match(buffer, identifier) {
        return identifier.endsWith('predict_net.pb');
    }    

    open(buffer, identifier, host, callback) {
        host.import('/caffe2.js', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var netDef = null;
            try {
                caffe2 = protobuf.roots.caffe2.caffe2;
                netDef = caffe2.NetDef.decode(buffer);
            }
            catch (error) {
                callback(new Caffe2Error('Protocol Buffer loader failed to decode caffe2.NetDef input stream (' + error.message + ').'), null);
                return;
            }
            var model = null;
            try {
                model = new Caffe2Model(netDef);
            }
            catch (error) {
                callback(new Caffe2Error(error.message), null);
                return;
            }
            Caffe2OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            }); 
        });
    }

}

class Caffe2Model {

    constructor(netDef) {
        var graph = new Caffe2Graph(netDef);
        this._graphs = [ graph ];
    }

    get properties() {
        var results = [];
        results.push({ name: 'format', value: 'Caffe2' });
        return results;
    }

    get graphs() {
        return this._graphs;
    }
}

class Caffe2Graph {

    constructor(netDef) {
        this._name = netDef.name ? netDef.name : '';
        this._type = netDef.type ? netDef.type : '';
        this._nodes = [];
        this._operators = {};

        var inplaceIndices = [];
        var inplaceMap = {};

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

        var initializerMap = {};
        netDef.externalInput.forEach((input) => {
            initializerMap[input] = true;
        });

        netDef.op.forEach((op) => {
            this._operators[op.type] = (this._operators[op.type] || 0) + 1;
            this._nodes.push(new Caffe2Node(op, initializerMap));
        });

        this._inputs = [];
        var inputs = Object.keys(initializerMap);
        inputs.forEach((input) => {
            if (inputs.length == 1 || !input.startsWith('caffe.')) {
                this._inputs.push({
                    id: input,
                    name: input,
                    type: 'T'
                });
            }
        });

        this._outputs = [];
        netDef.externalOutput.forEach((output) => {
            this._outputs.push({ 
                id: output,
                name: output,
                type: 'T'
            });
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

class Caffe2Node {

    constructor(op, initializerMap) {
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
                var initializer = initializerMap[input];
                if (initializer) {
                    delete initializerMap[input];
                    this._initializers[input] = new Caffe2Tensor('Initializer');
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
        return Caffe2OperatorMetadata.operatorMetadata.getOperatorCategory(this._operator);
    }

    get documentation() {
        return Caffe2OperatorMetadata.operatorMetadata.getOperatorDocumentation(this._operator);
    }

    get inputs() {
        var inputs = Caffe2OperatorMetadata.operatorMetadata.getInputs(this._operator, this._inputs);
        inputs.forEach((input) => {
            input.connections.forEach((connection) => {
                var initializer = this._initializers[connection.id];
                if (initializer) {
                    connection.initializer = initializer;
                }
            });
        });
        return inputs;
    }

    get outputs() {
        var outputs = Caffe2OperatorMetadata.operatorMetadata.getOutputs(this._operator, this._outputs);
        return outputs;
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
            this._value = JSON.stringify(arg.floats);
        }
        else if (arg.ints && arg.ints.length > 0) {
            debugger;
            this._value = JSON.stringify(arg.ints);
        }
        else if (arg.nets && arg.nets.length > 0) {
            debugger;
            this._value = '...';
        }
        else if (arg.i != 0) {
            this._value = arg.i.toString();
        }
        else {
            this._value = arg.i.toString();
        }

    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return Caffe2OperatorMetadata.operatorMetadata.getAttributeVisible(this._node.operator, this._name, this._value)
    }
}

class Caffe2Tensor {

    constructor(kind) {
        this._kind = kind;
    }

    get kind() {
        return this._kind;
    }

    get value() {
        return null;
    }

    toString() {
        return null;
    }
}

class Caffe2OperatorMetadata 
{

    static open(host, callback) {
        if (Caffe2OperatorMetadata.operatorMetadata) {
            callback(null, Caffe2OperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/caffe2-metadata.json', (err, data) => {
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

    getOperatorCategory(operator) {
        var schema = this._map[operator];
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    getOperatorDocumentation(operator) {
        var schema = this._map[operator];
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
        var schema = this._map[type];
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
        var schema = this._map[type];
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

    getAttributeVisible(operator, name, value) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributesMap) {
                schema.attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributesMap[attribute.name] = attribute;
                });
            }
            var attribute = schema.attributesMap[name];
            if (attribute) {
                if (attribute.hasOwnProperty('visible')) {
                    return attribute.visible;
                }
                if (attribute.hasOwnProperty('default')) {
                    return value != attribute.default.toString();
                }
            }
        }
        return true;
    }
    
    static isEquivalent(a, b) {
        if (a === b) {
            return a !== 0 || 1 / a === 1 / b;
        }
        if (a == null || b == null) {
            return false;
        }
        if (a !== a) {
            return b !== b;
        }
        var type = typeof a;
        if (type !== 'function' && type !== 'object' && typeof b != 'object') {
            return false;
        }
        var className = toString.call(a);
        if (className !== toString.call(b)) {
            return false;
        }
        switch (className) {
            case '[object RegExp]':
            case '[object String]':
                return '' + a === '' + b;
            case '[object Number]':
                if (+a !== +a) {
                    return +b !== +b;
                }
                return +a === 0 ? 1 / +a === 1 / b : +a === +b;
            case '[object Date]':
            case '[object Boolean]':
                return +a === +b;
            case '[object Array]':
                var length = a.length;
                if (length !== b.length) {
                    return false;
                }
                while (length--) {
                    if (!Caffe2OperatorMetadata.isEquivalent(a[length], b[length])) {
                        return false;
                    }
                }
                return true;
        }

        var keys = Object.keys(a);
        var size = keys.length;
        if (Object.keys(b).length != size) {
            return false;
        } 
        while (size--) {
            var key = keys[size];
            if (!(b.hasOwnProperty(key) && Caffe2OperatorMetadata.isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
}

class Caffe2Error extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe2 model.';
    }
}
