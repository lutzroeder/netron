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
            var obj = null;
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
                obj = CntkModelFactory._convertDictionary(dictionary);
            }
            catch (error) {
                callback(new CntkError('File format is not cntk.Dictionary (' + error.message + ').'), null);
                return;
            }
            CntkOperatorMetadata.open(host, (err, metadata) => {
                try {
                    var model = new CntkModel(obj);
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

    constructor(obj) {
        this._format = 'CNTK v2';
        this._graphs = [];
        this._graphs.push(new CntkGraph(obj));
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return this._format;
    }
}

class CntkGraph {

    constructor(obj) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        var connections = {};
        var names = {};
        obj.inputs.forEach((input) => {
            var connection = new CntkConnection(input);
            connections[input.uid] = connection;

            if (input.kind == 0) {
                this._inputs.push(new CntkArgument(input.name, [ connection ]));
            }

            if (input.name) {
                names[input.uid] = input.name;
            }
        });
        obj.primitive_functions.forEach((node) => {
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
            this._attributes.push(new CntkAttribute(this._operator, key, node.attributes[key]));
        });

        var inputs = [];
        var outputs = [];
        var initializers = [];

        node.inputs.forEach((input) => {
            var x = node;
            var connection = connections[input];
            var name = names[input];
            if (connection) {
                if (connection.initializer) {
                    initializers.push(connection);
                }
                else {
                    inputs.push(connection);
                }
            }
            else {
                inputs.push(new CntkConnection(input));
            }
        });

        outputs.push(new CntkConnection(output + '_Output_0'));

        inputs = inputs.concat(initializers);

        var inputIndex = 0;
        var schema = CntkOperatorMetadata.operatorMetadata.getSchema(this.operator);
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputSchema) => {
                if (inputIndex < inputs.length || inputSchema.option != 'optional') {
                    var connections = [];
                    var input = {};
                    input.name = inputSchema.name;
                    var count = (inputSchema.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    inputs.slice(inputIndex, inputIndex + count).forEach((connection) => {
                        if (connection.id != '' || inputSchema.option != 'optional') {
                            connections.push(connection);
                        }
                    });
                    inputIndex += count;
                    this._inputs.push(new CntkArgument(inputSchema.name, connections));
                }
            });
        }
        else {
            inputs.slice(inputIndex).forEach((connection) => {
                this._inputs.push(new CntkArgument('(' + inputIndex.toString() + ')', [ connection ]));
                inputIndex++;
            });
        }

        var outputIndex = 0;
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputSchema) => {
                if (outputIndex < outputs.length || outputSchema.option != 'optional') {
                    var count = (outputSchema.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    var connections = outputs.slice(outputIndex, outputIndex + count);
                    outputIndex += count;
                    this._outputs.push(new CntkArgument(outputSchema.name, connections));
                }
            });
        }
        else {
            outputs.slice(outputIndex).forEach((connection) => {
                this._outputs.push(new CntkArgument('(' + outputIndex.toString() + ')', [ connection ]));
                outputIndex++;
            });
        }
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get category() {
        var schema = CntkOperatorMetadata.operatorMetadata.getSchema(this._operator);
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    get documentation() { 
        return null;
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

    constructor(operator, name, value) {
        this._name = name;
        this._value = value;
        if (this._value.constructor.name == 'NDShape') {
            this._value = value.shape_dim.map((dimension) => {
                if (dimension.low == -1 && dimension.high == -1 && dimension.unsigned == true) {
                    return -1;
                }
                if (dimension && dimension.__isLong__) {
                    return dimension.toNumber();
                }
                return dimension;
            });
        }
        if (this._value.constructor.name == 'Axis') {
            this._value = () => '\'' + value.name + '\', ' + value.static_axis_idx + ', ' + value.is_ordered_dynamic_axis.toString();
        }

        var attributeSchema = CntkOperatorMetadata.operatorMetadata.getAttributeSchema(operator, name);
        if (attributeSchema) {
            if (attributeSchema.hasOwnProperty('visible') && !attributeSchema.visible) {
                this._visible = false;
            }
            else if (attributeSchema.hasOwnProperty('default')) {
                var defaultValue = attributeSchema.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < this._value.length) {
                           defaultValue.push(defaultValue[defaultValue.length - 1]); 
                        }
                    }
                    if (this._value.every((item, index) => { return item == defaultValue[index]; })) {
                        this._visible = false;
                    }
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
        if (!this._tensor.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        var value = this._tensor.value;
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
                // debugger;
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape;
        
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
        return results;
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
        // cntk/Source/CNTKv2LibraryDll/API/Internals/PrimitiveOpType.h
        return this._operatorMap[code] || null;
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema._attributesMap) {
                schema._attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema._attributesMap[attribute.name] = attribute;
                });
            }
            return schema._attributesMap[name] || null;
        }
        return null;
    }
}

class CntkError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading CNTK model.';
    }
}