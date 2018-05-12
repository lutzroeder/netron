/*jshint esversion: 6 */

// Experimental

class MXNetModelFactory {

    match(buffer, identifier) {
        if (identifier.endsWith('-symbol.json')) {
            return true;
        }
        var extension = identifier.split('.').pop();
        if (extension == 'json') {
            var decoder = new TextDecoder('utf-8');
            var json = decoder.decode(buffer);
            if (json.includes('\"mxnet_version\":')) {
                return true;
            }
        }
        return false;
    }

    open(buffer, identifier, host, callback) {
        try {
            var decoder = new TextDecoder('utf-8');
            var json = decoder.decode(buffer);
            var model = new MXNetModel(json);
            MXNetOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(new MXNetError(err.message), null);
        }
    }

}

class MXNetModel {

    constructor(json) {
        var model = JSON.parse(json);
        if (!model) {
            throw new MXNetError('JSON file does not contain MXNet data.');
        }
        if (!model.hasOwnProperty('nodes')) {
            throw new MXNetError('JSON file does not contain an MXNet \'nodes\' property.');
        }
        if (!model.hasOwnProperty('arg_nodes')) {
            throw new MXNetError('JSON file does not contain an MXNet \'arg_nodes\' property.');
        }
        if (!model.hasOwnProperty('heads')) {
            throw new MXNetError('JSON file does not contain an MXNet \'heads\' property.');
        }

        if (model.attrs && model.attrs.mxnet_version && model.attrs.mxnet_version.length == 2 && model.attrs.mxnet_version[0] == 'int') {
            var version = model.attrs.mxnet_version[1];
            var revision = version % 100;
            var minor = Math.floor(version / 100) % 100;
            var major = Math.floor(version / 10000) % 100;
            this._version = major.toString() + '.' + minor.toString() + '.' + revision.toString(); 
        }

        this._graphs = [ new MXNetGraph(model) ];
    }

    get properties() {
        var results = [];
        results.push({ name: 'Format', value: 'MXNet' + (this._version ? (' v' + this._version) : '') });
        return results;
    }

    get graphs() {
        return this._graphs;
    }

}

class MXNetGraph {

    constructor(json)
    {
        var nodes = json.nodes;

        this._nodes = [];
        json.nodes.forEach((node) => {
            node.outputs = [];
        });

        nodes.forEach((node) => {
            node.inputs = node.inputs.map((input) => {
                return MXNetGraph.updateOutput(nodes, input);
            });
        });

        var argumentMap = {};
        json.arg_nodes.forEach((index) => {
            argumentMap[index] = (index < nodes.length) ? nodes[index] : null;
        });

        this._outputs = [];
        var headMap = {};
        json.heads.forEach((head, index) => {
            var id = MXNetGraph.updateOutput(nodes, head);
            var name = 'output' + ((index == 0) ? '' : (index + 1).toString());
            this._outputs.push({ id: id, name: name });
        });

        nodes.forEach((node, index) => {
            if (!argumentMap[index]) {
                this._nodes.push(new MXNetNode(node, argumentMap));
            }
        });

        this._inputs = [];
        Object.keys(argumentMap).forEach((key) => {
            var argument = argumentMap[key];
            if ((!argument.inputs || argument.inputs.length == 0) &&
                (argument.outputs && argument.outputs.length == 1)) {
                this._inputs.push( { id: argument.outputs[0], name: argument.name });
            }
        });
    }

    get name() {
        return '';
    }

    get inputs() {
        return this._inputs.map((input) => {
            return { 
                name: input.name,
                type: 'T',
                id: '[' + input.id.join(',') + ']' 
            };
        });
    }

    get outputs() {
        return this._outputs.map((output) => {
            return { 
                name: output.name,
                type: 'T',
                id: '[' + output.id.join(',') + ']' 
            };
        });
    }

    get nodes() {
        return this._nodes;
    }

    static updateOutput(nodes, input) {
        var sourceNodeIndex = input[0];
        var sourceNode = nodes[sourceNodeIndex];
        var sourceOutputIndex = input[1];
        while (sourceOutputIndex >= sourceNode.outputs.length) {
            sourceNode.outputs.push([ sourceNodeIndex, sourceNode.outputs.length ]);
        }
        return [ sourceNodeIndex, sourceOutputIndex ];
    }
}

class MXNetNode {

    constructor(json, argumentMap) {
        this._operator = json.op;
        this._name = json.name;
        this._inputs = json.inputs;
        this._outputs = json.outputs;
        this._attributes = [];
        var attrs = json.attrs;
        if (!attrs) {
            attrs = json.attr;
        }
        if (!attrs) {
            attrs = json.param;
        }
        if (attrs) {
            Object.keys(attrs).forEach((key) => {
                var value = attrs[key];
                this._attributes.push(new MXNetAttribute(this, key, value));
            });
        }
        this._initializers = {};
        this._inputs.forEach((input) => {
            var argumentNodeIndex = input[0];
            var argument = argumentMap[argumentNodeIndex];
            if (argument) {
                if ((!argument.inputs || argument.inputs.length == 0) &&
                    (argument.outputs && argument.outputs.length == 1)) {
                    var prefix = this._name + '_';
                    if (prefix.endsWith('_fwd_')) {
                        prefix = prefix.slice(0, -4);
                    }
                    if (argument.name && argument.name.startsWith(prefix)) {
                        var id = '[' + input.join(',') + ']';
                        this._initializers[id] = new MXNetTensor(argument);
                        delete argumentMap[argumentNodeIndex];
                    }
                }
            }
        });
    }

    get operator() {
        return this._operator;
    }

    get category() {
        return MXNetOperatorMetadata.operatorMetadata.getOperatorCategory(this._operator);
    }

    get documentation() {
        return MXNetOperatorMetadata.operatorMetadata.getOperatorDocumentation(this.operator);
    }

    get name() {
        return this._name;
    }

    get inputs() {
        var inputs = this._inputs.map((inputs) => {
            return '[' + inputs.join(',') + ']'; 
        });        
        var results = MXNetOperatorMetadata.operatorMetadata.getInputs(this._operator, inputs);
        results.forEach((input) => {
            input.connections.forEach((connection) => {
                var initializer = this._initializers[connection.id];
                if (initializer) {
                    connection.type = initializer.type;
                    connection.initializer = initializer;
                }
            });
        });
        return results;
    }

    get outputs() {
        var outputs = this._outputs.map((output) => {
            return '[' + output.join(',') + ']'; 
        });
        return MXNetOperatorMetadata.operatorMetadata.getOutputs(this._type, outputs);
    }

    get attributes() {
        return this._attributes;
    }
}

class MXNetAttribute {

    constructor(owner, name, value) {
        this._owner = owner;
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return MXNetOperatorMetadata.operatorMetadata.getAttributeVisible(this._owner.operator, this._name, this._value);
    }
}

class MXNetTensor {
    
    constructor(json) {
        this._json = json;
        this._type = '';
        var attrs = this._json.attrs; 
        if (attrs) {
            var dtype = attrs.__dtype__;
            var shape = attrs.__shape__;
            if (dtype && shape) {
                dtype = dtype.replace('0', 'float');
                shape = shape.split(' ').join('').replace('(', '[').replace(')', ']');
                this._type = dtype + shape;
            }
        }
    }

    get name() {
        return this._json.name;
    }

    get kind() {
        return 'Initializer';
    }

    get type() {
        return this._type;
    }
}

class MXNetOperatorMetadata {

    static open(host, callback) {
        if (MXNetOperatorMetadata.operatorMetadata) {
            callback(null, MXNetOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/mxnet-metadata.json', (err, data) => {
                MXNetOperatorMetadata.operatorMetadata = new MXNetOperatorMetadata(data);
                callback(null, MXNetOperatorMetadata.operatorMetadata);
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
                    value = MXNetOperatorMetadata.formatTuple(value); 
                    return !MXNetOperatorMetadata.isEquivalent(attribute.default, value);
                }
            }
        }
        return true;
    }

    static formatTuple(value) {
        if (value.startsWith('(') && value.endsWith(')')) {
            var list = value.substring(1, value.length - 1).split(',');
            list = list.map(item => item.trim());
            if (list.length > 1) {
                if (list.every(item => item == list[0])) {
                    list = [ list[0], '' ];
                }
            }
            return '(' + list.join(',') + ')';
        }
        return value;
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
                    if (!KerasOperatorMetadata.isEquivalent(a[length], b[length])) {
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
            if (!(b.hasOwnProperty(key) && KerasOperatorMetadata.isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
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
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return '';
    }
}

class MXNetError extends Error {
    constructor(message) {
        super(message);
        this.name = 'MXNet Error';
    }
}