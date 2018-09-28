/*jshint esversion: 6 */

var cntk = null;

class CntkModelFactory {

    match(context, host) {
        if (!host.environment('CNTK')) {
            return false;
        }
        var extension = context.identifier.split('.').pop();
        switch (extension) {
            case 'model':
            case 'cntk':
                return true;
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('cntk', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var root = null;
            try {
                var buffer = context.buffer;
                if (buffer && buffer.length > 6 && 
                    buffer[0] == 0x42 && buffer[1] == 0x00 &&
                    buffer[2] == 0x43 && buffer[3] == 0x00 &&
                    buffer[4] == 0x4E && buffer[5] == 0x00)
                {
                    callback(new CntkError('CNTK v1 format not supported.'), null);
                    return;
                }    
                cntk = protobuf.roots.cntk.CNTK.proto;
                var dictionary = cntk.Dictionary.decode(buffer);
                root = CntkModelFactory._convertDictionary(dictionary);
            }
            catch (error) {
                callback(new CntkError('File format is not cntk.Dictionary (' + error.message + ').'), null);
                return;
            }
            CntkOperatorMetadata.open(host, (err, metadata) => {
                try {
                    var model = new CntkModel(root);
                    callback(null, model);
                }
                catch (error) {
                    callback(new CntkError(error.message), null);
                }
            });
        });
    }

    static _convertDictionary(dictionary) {
        var target = {};
        Object.keys(dictionary.data).filter((key) => key != 'version').forEach((key) => {
            target[key] = CntkModelFactory._convertDictionaryValue(dictionary.data[key]);
        });
        return target;
    }

    static _convertDictionaryValue(dictionaryValue) {
        switch (dictionaryValue.value_type) {
            case cntk.DictionaryValue.Type.Bool:
                return dictionaryValue.bool_value;
            case cntk.DictionaryValue.Type.Int:
                return dictionaryValue.int_value;
            case cntk.DictionaryValue.Type.SizeT:
                return dictionaryValue.size_t_value;
            case cntk.DictionaryValue.Type.Float:
                return dictionaryValue.float_value;
            case cntk.DictionaryValue.Type.Double:
                return dictionaryValue.double_value;
            case cntk.DictionaryValue.Type.String:
                return dictionaryValue.string_value;
            case cntk.DictionaryValue.Type.Vector:
                return CntkModelFactory._convertVectorValue(dictionaryValue.vector_value);
            case cntk.DictionaryValue.Type.NDShape:
                return dictionaryValue.nd_shape_value;
            case cntk.DictionaryValue.Type.Axis:
                return dictionaryValue.axis_value;
            case cntk.DictionaryValue.Type.Dictionary:
                return CntkModelFactory._convertDictionary(dictionaryValue.dictionary_value);
            case cntk.DictionaryValue.Type.NDArrayView:
                return dictionaryValue.nd_array_view_value;
        }
        debugger;
    }

    static _convertVectorValue(vectorValue) {
        var target = [];
        vectorValue.value.forEach((item) => {
            target.push(CntkModelFactory._convertDictionaryValue(item));
        });
        return target;
    }
}

class CntkModel {

    constructor(root) {
        this._format = 'CNTK v2';
        this._graphs = [];
        this._graphs.push(new CntkGraph(root));
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return this._format;
    }
}

class CntkGraph {

    constructor(root) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        var connections = {};
        var names = {};
        root.inputs.forEach((input) => {
            var connection = new CntkConnection(input);
            connections[input.uid] = connection;

            if (input.kind == 0) {
                this._inputs.push(new CntkArgument(input.name, [ connection ]));
            }

            if (input.name) {
                names[input.uid] = input.name;
            }
        });
        root.primitive_functions.forEach((node) => {
            this._nodes.push(new CntkNode(node, connections, names));
        });
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
}

class CntkArgument {
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

class CntkConnection {

    constructor(tensor) {
        if (typeof tensor === 'string') {
            this._id = tensor.toString();
        }
        else {
            this._id = tensor.uid;
            if (tensor.value) {
                this._type = null;
                this._initializer = new CntkTensor(tensor);
            }
            else {
                this._type = new CntkTensorType(tensor.data_type, tensor.shape);
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
        return null;
    }

    get initializer() {
        return this._initializer;
    }
}

class CntkNode { 

    constructor(node, connections, names) {

        var output = node.uid;

        this._name = node.name || null;

        if (node.op == 57) {
            this._operator = 'Block';
            debugger;
        }
        else {
            this._operator = CntkOperatorMetadata.operatorMetadata.getOperatorName(node.op);
            if (this._operator == null) {
                this._operator = node.op.toString();
                debugger;
            }                
        }
        if (node.block_function_op_name) {
            this._operator = '[' + node.block_function_op_name + ']';
        }

        this._attributes = [];
        this._inputs = [];
        this._initializers = [];
        this._outputs = [];

        Object.keys(node.attributes).forEach((key) => {
            this._attributes.push(new CntkAttribute(key, node.attributes[key]));
        });

        node.inputs.forEach((input) => {
            var x = node;
            var connection = connections[input];
            var name = names[input];
            if (connection) {
                if (connection.initializer) {
                    this._initializers.push(new CntkArgument(name, [ connection ]));
                }
                else {
                    this._inputs.push(new CntkArgument('?', [ connection ]));
                }
            }
            else {
                this._inputs.push(new CntkArgument('input', [ 
                    new CntkConnection(input)
                ]));
            }
        });

        this._outputs.push(new CntkArgument('output', [ 
            new CntkConnection(output + '_Output_0')
        ]));
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get category() {
        return CntkOperatorMetadata.operatorMetadata.getOperatorCategory(this._operator);
    }

    get attributes() { 
        return this._attributes;
    }

    get inputs() {
        return this._inputs.concat(this._initializers);
    }

    get outputs() {
        return this._outputs;
    }
}

class CntkAttribute {

    constructor(name, value) {
        this._name = name;
        this._value = value;
        if (this._value.constructor.name == 'NDShape') {
            this._value = () => value.shape_dim.map((dimension) => {
                if (dimension.low == -1 && dimension.high == -1 && dimension.unsigned == true) {
                    return -1;
                }
                if (dimension && dimension.__isLong__) {
                    return dimension.toNumber();
                }
                return dimension;
            }).join(', ');
        }
        if (this._value.constructor.name == 'Axis') {
            this._value = () => '\'' + value.name + '\', ' + value.static_axis_idx + ', ' + value.is_ordered_dynamic_axis.toString();
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return true;
    }
}

class CntkTensor {

    constructor(tensor) {
        this._tensor = tensor;
        this._name = tensor.uid;
        this._type = new CntkTensorType(tensor.data_type, tensor.shape);
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Not implemented.';
    }
}

class CntkTensorType {

    constructor(dataType, shape) {
        switch (dataType.toNumber()) {
            case 1: this._dataType = 'float32'; break;
            default: this._dataType = '?'; debugger; break;
        }
        this._shape = shape.shape_dim.map((dimension) => {
            if (dimension && dimension.__isLong__) {
                return dimension.toNumber();
            }
            return dimension;            
        });
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + ((this._shape && this._shape.length) ? ('[' + this._shape.join(',') + ']') : '');
    }
}

class CntkOperatorMetadata 
{

    static open(host, callback) {
        if (CntkOperatorMetadata.operatorMetadata) {
            callback(null, CntkOperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'cntk-metadata.json', 'utf-8', (err, data) => {
                CntkOperatorMetadata.operatorMetadata = new CntkOperatorMetadata(data);
                callback(null, CntkOperatorMetadata.operatorMetadata);
            });
        }    
    }

    constructor(data) {
        this._map = {};
        this._operatorMap = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema)
                    {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                        if (schema.operator) {
                            this._operatorMap[schema.operator] = name;
                        }
                    }
                });
            }
        }
    }

    getOperatorName(code) {
        return this._operatorMap[code] || null;
    }

    getOperatorCategory(operator) {
        var schema = this._map[operator];
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
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
}

class CntkError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading CNTK model.';
    }
}