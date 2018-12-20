/*jshint esversion: 6 */

var mxnet = mxnet || {};
var marked = marked || require('marked');
var base = base || require('./base');
var zip = zip || require('./zip');

mxnet.ModelFactory = class {

    match(context, host) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'model') {
            var buffer = context.buffer;
            if (buffer && buffer.length > 2 && buffer[0] == 0x50 && buffer[1] == 0x4B) {
                return true;
            }
        }
        if (extension == 'json') {
            var json = context.text;
            if (json.indexOf('\"nodes\":', 0) != -1) {
                try {
                    var symbol = JSON.parse(json);
                    if (symbol && symbol.nodes && symbol.arg_nodes && symbol.heads) {
                        return true;
                    }
                }
                catch (err) {
                }
            }
        }
        return false;
    }

    open(context, host, callback) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        switch (extension) {
            case 'json':
                this._openSymbol(context, host, callback);
                break;
            case 'model':
                this._openModelServer(context, host, callback);
                break;
            default:
                callback(new mxnet.Error('Unsupported file extension.'));
                break;
        }
    }

    _openSymbol(context, host, callback) {
        try {
            var identifier = context.identifier;
            var symbol = JSON.parse(context.text);
            var format = null;
            if (symbol && symbol.nodes && symbol.nodes.some((node) => node && node.op == 'tvm_op')) {
                format  = 'TVM';
            }
            var mxnet_extension = '-symbol.json';
            if (identifier.toLowerCase().endsWith(mxnet_extension)) {
                var paramsIdentifier = identifier.substring(0, identifier.length - mxnet_extension.length) + '-0000.params';
                context.request(paramsIdentifier, null, (err, params) => {
                    this._openModel(format, null, symbol, null, params, host, callback);
                    return;
                });
                return;
            }
            this._openModel(format, null, symbol, null, null, host, callback);
            return;
        }
        catch (error) {
            host.exception(error, false);
            callback(new mxnet.Error(error.message), null);
            return;
        }
    }

    _openModelServer(context, host, callback) {
        var entries = {};
        try {
            var archive = new zip.Archive(context.buffer, host.inflateRaw);
            archive.entries.forEach((entry) => {
                entries[entry.name] = entry;
            });
        }
        catch (err) {
            callback(new mxnet.Error('Failed to decompress ZIP archive. ' + err.message), null);
            return;
        }

        var manifestEntry = entries['MANIFEST.json'];
        var rootFolder = '';
        if (!manifestEntry) {
            var folders = Object.keys(entries).filter((name) => name.endsWith('/')).filter((name) => entries[name + 'MANIFEST.json']);
            if (folders.length != 1) {
                callback(new mxnet.Error("Manifest not found in '" + context.identifier + "'."), null);
                return;
            }
            rootFolder = folders[0];
            manifestEntry = entries[rootFolder + 'MANIFEST.json'];
        }

        var decoder = new TextDecoder('utf-8');
        var manifest = null;
        try {
            manifest = JSON.parse(decoder.decode(manifestEntry.data));
        }
        catch (err) {
            callback(new mxnet.Error('Failed to read manifest. ' + err.message), null);
            return;
        }

        if (!manifest.Model) {
            callback(new mxnet.Error('Manifest does not contain model.'), null);
            return;
        }

        var modelFormat = manifest.Model['Model-Format'];
        if (modelFormat && modelFormat != 'MXNet-Symbolic') {
            callback(new mxnet.Error('Model format \'' + modelFormat + '\' not supported.'), null);
            return;
        }

        if (!manifest.Model.Symbol) {
            callback(new mxnet.Error('Manifest does not contain symbol entry.'), null);
            return;
        }

        var symbol = null;
        try {
            var symbolEntry = entries[rootFolder + manifest.Model.Symbol];
            symbol = JSON.parse(decoder.decode(symbolEntry.data));
        }
        catch (err) {
            callback(new mxnet.Error('Failed to load symbol entry. ' + err.message), null);
            return;
        }

        var signature = null;
        try {
            if (manifest.Model.Signature) {
                var signatureEntry = entries[rootFolder + manifest.Model.Signature];
                if (signatureEntry) {
                    signature = JSON.parse(decoder.decode(signatureEntry.data));
                }
            }
        }
        catch (err) {
        }

        var params = null;
        try {
            if (manifest.Model.Parameters) {
                var parametersEntry = entries[rootFolder + manifest.Model.Parameters];
                if (parametersEntry) {
                    params = parametersEntry.data;
                }
            }
        }
        catch (err) {
        }

        try {
            var format = null;
            if (manifest) {
                format = 'MXNet Model Server';
                if (manifest['Model-Archive-Version']) {
                    format += ' v' + manifest['Model-Archive-Version'].toString();
                }
            }
            this._openModel(format, manifest, symbol, signature, params, host, callback);
            return;
        } 
        catch (error) {
            callback(new mxnet.Error(error.message), null);
            return;
        }
    }

    _openModel(format, manifest, symbol, signature, params, host, callback) {
        mxnet.Metadata.open(host, 'mxnet-metadata.json', (err, metadata) => {
            var parameters = {};
            if (params) {
                try {
                    var stream = new ndarray.Stream(params);
                    Object.keys(stream.arrays).forEach((key) => {
                        var name = key;
                        if (name.startsWith('arg:') || name.startsWith('aux:')) {
                            name = key.substring(4);
                        }
                        parameters[name] = stream.arrays[key];
                    });
                }
                catch (error) {
                }
            }
            try {
                var model = new mxnet.Model(metadata, format, manifest, symbol, signature, parameters);
                callback(null, model);
                return;
            }
            catch (error) {
                host.exception(error, false);
                callback(new mxnet.Error(error.message), null);
                return;
            }
        });
    }
};

mxnet.Model = class {

    constructor(metadata, format, manifest, symbol, signature, parameters) {
        if (!symbol) {
            throw new mxnet.Error('JSON file does not contain MXNet data.');
        }
        if (!symbol.hasOwnProperty('nodes')) {
            throw new mxnet.Error('JSON file does not contain an MXNet \'nodes\' property.');
        }
        if (!symbol.hasOwnProperty('arg_nodes')) {
            throw new mxnet.Error('JSON file does not contain an MXNet \'arg_nodes\' property.');
        }
        if (!symbol.hasOwnProperty('heads')) {
            throw new mxnet.Error('JSON file does not contain an MXNet \'heads\' property.');
        }

        this._format = format;

        if (manifest) {
            if (manifest.Model && manifest.Model['Model-Name']) {
                this._name = manifest.Model['Model-Name'];
            }
            if (manifest.Model && manifest.Model.Description && this._name != manifest.Model.Description) {
                this._description = manifest.Model.Description;
            }
            if (manifest.Engine && manifest.Engine.MXNet) {
                var engineVersion = mxnet.Model._convert_version(manifest.Engine.MXNet);
                this._engine = 'MXNet v' + (engineVersion ? engineVersion : manifest.Engine.MXNet.toString());
            }
        }

        if (!this._format) {
            if (symbol.attrs && symbol.attrs.mxnet_version) {
                var version = mxnet.Model._convert_version(symbol.attrs.mxnet_version);
                if (version) {
                    this._format = 'MXNet v' + version;
                }
            }
        }

        if (!this._format) {
            this._format = 'MXNet';
        }

        this._graphs = [];
        this._graphs.push(new mxnet.Graph(metadata, manifest, symbol, signature, parameters));
    }

    get name() {
        return this._name;
    }

    get format() {
        return this._format;
    }

    get description() {
        return this._description;
    }

    get runtime() {
        return this._engine;
    }

    get graphs() {
        return this._graphs;
    }

    static _convert_version(value) {
        if (Array.isArray(value)) {
            if (value.length == 2 && value[0] == 'int') {
                var major = Math.floor(value[1] / 10000) % 100;
                var minor = Math.floor(value[1] / 100) % 100;
                var patch = Math.floor(value[1]) % 100;
                return [ major.toString(), minor.toString(), patch.toString() ].join('.');
            }
        }
        return null;
    }
};

mxnet.Graph = class {

    constructor(metadata, manifest, symbol, signature, parameters)
    {
        this._metadata = metadata;
        this._nodes = [];
        this._operators = [];

        var nodes = symbol.nodes;
        nodes.forEach((node) => {
            if (node.op && node.op != 'null') { 
                var operator = node.op;
                var attrs = node.attrs || node.attr || node.param;
                if (operator == 'tvm_op' && attrs && attrs.func_name) {
                    operator = attrs.func_name;
                }
                this._operators[operator] = (this._operators[operator] || 0) + 1;
            }
        });

        var inputs = {};
        if (signature && signature.inputs) {
            signature.inputs.forEach((input) => {
                inputs[input.data_name] = input;
            });
        }
        var outputs = {};
        if (signature && signature.outputs) {
            signature.outputs.forEach((output) => {
                outputs[output.data_name] = output;
            });
        }

        nodes.forEach((node) => {
            node.outputs = [];
        });
        nodes.forEach((node) => {
            node.inputs = node.inputs.map((input) => {
                return mxnet.Graph._updateOutput(nodes, input);
            });
        });

        var outputCountMap = {};
        nodes.forEach((node) => {
            node.outputs.forEach((output) => {
                outputCountMap[output] = (outputCountMap[output] || 0) + 1;
            });
        });

        var argumentMap = {};
        symbol.arg_nodes.forEach((index) => {
            argumentMap[index] = (index < nodes.length) ? nodes[index] : null;
        });

        this._outputs = [];
        symbol.heads.forEach((head, index) => {
            var outputId = mxnet.Graph._updateOutput(nodes, head);
            var outputName = nodes[outputId[0]] ? nodes[outputId[0]].name : ('output' + ((index == 0) ? '' : (index + 1).toString()));
            var outputType = null;
            var outputSignature = outputs[outputName];
            if (outputSignature && outputSignature.data_shape) {
                outputType = new mxnet.TensorType(null, new mxnet.TensorShape(outputSignature.data_shape));
            }
            this._outputs.push(new mxnet.Argument(outputName, [ new mxnet.Connection('[' + outputId.join(',') + ']', outputType, null) ]));
        });

        nodes.forEach((node, index) => {
            if (!argumentMap[index]) {
                this._nodes.push(new mxnet.Node(this._metadata, node, argumentMap, parameters));
            }
        });

        this._inputs = [];
        Object.keys(argumentMap).forEach((key) => {
            var argument = argumentMap[key];
            if ((!argument.inputs || argument.inputs.length == 0) &&
                (argument.outputs && argument.outputs.length == 1)) {
                var inputId = argument.outputs[0];
                var inputName = argument.name;
                var inputType = null;
                var inputSignature = inputs[inputName];
                if (inputSignature && inputSignature.data_shape) {
                    inputType = new mxnet.TensorType(null, new mxnet.TensorShape(inputSignature.data_shape));
                }
                this._inputs.push(new mxnet.Argument(inputName, [ new mxnet.Connection('[' + inputId.join(',') + ']', inputType) ]));
            }
        });
    }

    get operators() { 
        return this._operators;
    }

    get name() {
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

    static _updateOutput(nodes, input) {
        var nodeIndex = input[0];
        var node = nodes[nodeIndex];
        var outputIndex = input[1];
        if (node) {
            while (outputIndex >= node.outputs.length) {
                node.outputs.push([ nodeIndex, node.outputs.length ]);
            }
        }
        return [ nodeIndex, outputIndex ];
    }
};

mxnet.Argument = class {
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

mxnet.Connection = class {
    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        if (this._initializer) {
            return this._initializer.name;
        }
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
};

mxnet.Node = class {

    constructor(metadata, node, argumentMap, parameters) {
        this._metadata = metadata;
        this._operator = node.op;
        this._name = node.name;
        this._inputs = node.inputs;
        this._outputs = node.outputs;
        this._attributes = [];
        var attrs = node.attrs || node.attr || node.param;
        if (attrs) {
            if (this._operator == 'tvm_op' && attrs.func_name) {
                this._operator = attrs.func_name;
            }
            Object.keys(attrs).forEach((key) => {
                if (this._operator != 'tvm_op' && key != 'func_name') {
                    var value = attrs[key];
                    this._attributes.push(new mxnet.Attribute(this._metadata, this.operator, key, value));
                }
            });
        }
        if (this._operator == 'RNN') {
            this._inputs = this._inputs.map((input) => {
                var argumentNodeIndex = input[0];
                var argument = argumentMap[argumentNodeIndex];
                if (argument && argument.op == 'null' && argument.name &&
                    argument.name.endsWith('_parameters') && argument.attr && argument.attr.__init__) {
                    this._attributes.push(new mxnet.Attribute(this._metadata, this.operator, argument.name, argument.attr.__init__));
                    delete argumentMap[argumentNodeIndex];
                    return null;
                }
                return input;
            }); 
            this._inputs = this._inputs.filter((item) => item != null);
        }

        this._initializers = {};
        this._inputs.forEach((input) => {
            var argumentNodeIndex = input[0];
            var argument = argumentMap[argumentNodeIndex];
            if (argument && argument.name &&
                (!argument.inputs || argument.inputs.length == 0) &&
                (argument.outputs && argument.outputs.length == 1)) {
                var id = '[' + input.join(',') + ']';
                var parameter = parameters[argument.name];
                if (parameter) {
                    this._initializers[id] = new mxnet.Tensor('Initializer', argument.name, parameter.dataType, parameter.shape.dimensions, parameter.data);
                    delete argumentMap[argumentNodeIndex];
                }
                else {
                    var prefix = this._name;
                    if (prefix.endsWith('_fwd')) {
                        prefix = prefix.slice(0, -3);
                    }
                    if (argument.name && (argument.name.startsWith(prefix + '_') || argument.name.startsWith(prefix + '.'))) {
                        var dataType = '?';
                        var shape = [];
                        if (argument.attrs && argument.attrs.__dtype__ && argument.attrs.__shape__) {
                            try {
                                dataType = parseInt(argument.attrs.__dtype__);
                                shape = JSON.parse('[' + argument.attrs.__shape__.replace('(', '').replace(')', '').split(' ').join('').split(',').map((dimension => dimension || '\"?\"' )).join(',') + ']');
                            }
                            catch (err) {
                            }
                        }
                        this._initializers[id] = new mxnet.Tensor('Initializer', argument.name, dataType, shape, null);
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
        var schema = this._metadata.getSchema(this._operator); 
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
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
            return schema;
        }
        return '';
    }

    get name() {
        return this._name;
    }

    get inputs() {
        var args = [];
        var index = 0;
        var inputs = this._inputs;
        var schema = this._metadata.getSchema(this.operator);
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputDef) => {
                if (index < inputs.length || inputDef.option != 'optional') {
                    var count = (inputDef.option == 'variadic') ? (inputs.length - index) : 1;
                    var connections = [];
                    inputs.slice(index, index + count).forEach((input) => {
                        var id = '[' + input.join(',') + ']';
                        if (id != '' || inputDef.option != 'optional') {
                            connections.push(new mxnet.Connection(id, inputDef.type, this._initializers[id]));
                        }
                    });
                    index += count;
                    args.push(new mxnet.Argument(inputDef.name, connections));
                }
            });
        }
        if (index < inputs.length) {
            inputs.slice(index).forEach((input) => {
                var name = index.toString();
                var id = '[' + input.join(',') + ']';
                var connection = new mxnet.Connection(id, null, this._initializers[id]);
                args.push(new mxnet.Argument(name, [ connection ]));
                index++;
            });
        }
        return args;
    }

    get outputs() {
        var args = [];
        var index = 0;
        var outputs = this._outputs;
        var schema = this._metadata.getSchema(this.operator);
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputDef) => {
                if (index < outputs.length || outputDef.option != 'optional') {
                    var output = {};
                    var connections = [];
                    var count = (outputDef.option == 'variadic') ? (outputs.length - index) : 1;
                    outputs.slice(index, index + count).forEach((input) => {
                        connections.push(new mxnet.Connection('[' + input.join(',') + ']', null, null));
                    });
                    index += count;
                    args.push(new mxnet.Argument(outputDef.name, connections));
                }
            });
        }
        if (index < outputs.length) {
                outputs.slice(index).forEach((output) => {
                var name = index.toString();
                var connection = new mxnet.Connection('[' + output.join(',') + ']', null, null);
                args.push(new mxnet.Argument(name, [ connection ]));
                index++;
            });
        }
        return args;
    }

    get attributes() {
        return this._attributes;
    }
};

mxnet.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;

        var schema = metadata.getAttributeSchema(operator, name);
        if (schema && schema.type) {
            switch (schema.type) {
                case 'bool':
                    if (this._value == 'True') {
                        this._value = true;
                    }
                    else if (this._value == 'False') {
                        this._value = false;
                    }
                    break;
                case 'int32':
                    var intValue = Number.parseInt(this._value, 10);
                    this._value = Number.isNaN(this._value - intValue) ? value : intValue;
                    break;
                case 'float32':
                case 'float64':
                    var floatValue = Number.parseFloat(this._value);
                    this._value = Number.isNaN(this._value - floatValue) ? value : floatValue;
                    break;
                case 'int32[]':
                    if (this._value.length > 2 && this._value.startsWith('(') && this._value.endsWith(')')) {
                        var array = [];
                        var items = this._value.substring(1, this._value.length - 1).split(',');
                        items = items.map((item) => item.trim());
                        items = items.map((item) => item.endsWith('L') ? item.substring(0, item.length - 1) : item);
                        items = items.map((item) => {
                            var intValue = Number.parseInt(item, 10);
                            if (Number.isNaN(item - intValue)) {
                                array = null;
                            }
                            else if (array != null) {
                                array.push(intValue);
                            }        
                        });
                        if (array != null) {
                            this._value = array;
                        }
                    }
                    break;
            }    
        }

        if (schema) {
            if (schema.hasOwnProperty('visible') && !schema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default')) {
                var defaultValue = schema.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
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

mxnet.Tensor = class {
    
    constructor(kind, name, dataType, shape, data) {
        this._kind = kind;
        this._name = name;
        this._dataType = dataType;
        this._data = data;
        this._type = new mxnet.TensorType(dataType, new mxnet.TensorShape(shape));
    }

    get kind() {
        return 'Initializer';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        if (this._type.dataType == '?') {
            context.state = 'Tensor has no data type.';
            return context;
        }

        if (this._type.dataType.length <= 1) {
            context.state = 'Tensor has unknown data type.';
            return context;
        }

        if (this._type.shape.length < 1) {
            context.state = 'Tensor has unknown shape.';
            return context;
        }

        context.dimensions = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._dataType)
                {
                    case 0: // float32
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 1: // float64
                        results.push(context.data.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 2: // float16:
                        results.push(mxnet.Tensor._decodeNumberFromFloat16(context.data.getUint16(context.index, true)));
                        context.index += 2;
                        context.count++;
                        break;
                    case 3: // uint8
                        results.push(context.data.getUint8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 4: // int32
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 5: // int8
                        results.push(context.data.getInt8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 6: // int64
                        results.push(new base.Int64(context.data.subarray(context.index, context.index + 8)));
                        context.index += 8;
                        context.count++;
                        break;
                }
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

    static _decodeNumberFromFloat16(value) {
        var s = (value & 0x8000) >> 15;
        var e = (value & 0x7C00) >> 10;
        var f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
    }
};

mxnet.TensorType = class {

    constructor(dataType, shape) {
        mxnet.TensorType._dataTypeNameTable = mxnet.Tensor._dataTypeTable || [ 'float32', 'float64', 'float16', 'uint8', 'int32', 'int8', 'int64' ];
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        var dataType = '?';
        if (this._dataType || this._dataType === 0) {
            if (this._dataType < mxnet.TensorType._dataTypeNameTable.length) {
                dataType = mxnet.TensorType._dataTypeNameTable[this._dataType];
            }
            else {
                dataType = this._dataType.toString();
            }
        }
        return dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

mxnet.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length == 0) {
                return '';
            }
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

mxnet.Metadata = class {

    static open(host, file, callback) {
        mxnet.Metadata._metadata = {};
        if (mxnet.Metadata._metadata[file]) {
            callback(null, mxnet.Metadata._metadata[file]);
            return;
        }
        host.request(null, file, 'utf-8', (err, data) => {
            mxnet.Metadata._metadata[file] = new mxnet.Metadata(data);
            callback(null, mxnet.Metadata._metadata[file]);
            return;
        });
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
            if (!schema.__attributesMap) {
                schema.__attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.__attributesMap[attribute.name] = attribute;
                });
            }
            return schema.__attributesMap[name];
        }
        return null;
    }
};

mxnet.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MXNet model.';
    }
};

var ndarray = ndarray || {};

ndarray.Stream = class {

    constructor(buffer) {

        this._arrays = {};

        var reader = new ndarray.Reader(buffer);
        if (!reader.checkSignature([ 0x12, 1, 0, 0, 0, 0, 0, 0 ])) {
            throw new ndarray.Error('Invalid signature.');
        }
        if (!reader.checkSignature([ 0, 0, 0, 0, 0, 0, 0, 0 ])) {
            throw new ndarray.Error('Invalid reserved block.');
        }

        var data = [];
        for (var dataSize = reader.readUint64(); dataSize > 0; dataSize--) {
            data.push(new ndarray.Array(reader));
        }

        var decoder = new TextDecoder('ascii');
        var names = [];
        for (var namesSize = reader.readUint64(); namesSize > 0; namesSize--) {
            var length = reader.readUint64();
            var name = decoder.decode(reader.read(length));
            names.push(name);
        }

        if (names.length != data.length) {
            throw new ndarray.Error('Label count mismatch.');
        }

        for (var i = 0; i < names.length; i++) {
            this._arrays[names[i]] = data[i];
        }
    }

    get arrays() {
        return this._arrays;
    }

};

ndarray.Array = class { 

    constructor(reader) {

        ndarray.Array._dataTypeSizeTable = [ 4, 8, 2, 1, 4, 1, 8 ];

        if (reader.checkSignature([ 0xc9, 0xfa, 0x93, 0xF9 ])) {
            this._loadV2(reader);
        }
        else if (reader.checkSignature([ 0xc8, 0xfa, 0x93, 0xF9 ])) {
            this._loadV1(reader);
        }
        else {
            this._loadV0(reader);
        }
    }

    _loadV2(reader) {
        var stype = reader.readUint32();
        var num_aux_data = 0;
        switch (stype) {
            case 0: num_aux_data = 0; break; // kDefaultStorage
            case 1: num_aux_data = 1; break; // kRowSparseStorage
            case 2: num_aux_data = 2; break; // kCSRStorage
        }
        var sshape = null;
        if (num_aux_data > 0) {
            sshape = new ndarray.Shape(reader, true);
        }
        this._shape = new ndarray.Shape(reader, true);
        if (this._shape.dimensions.length == 0) {
            return;
        }
        var context = new ndarray.Context(reader);
        this._dataType = reader.readUint32();
        if (num_aux_data > 0) {
            throw new ndarray.Error('Not implemented.');
        }
        var dataTypeSize = (this._dataType < ndarray.Array._dataTypeSizeTable.length) ? ndarray.Array._dataTypeSizeTable[this._dataType] : 0;
        var size = dataTypeSize * this._shape.size();
        this._data = reader.read(size);
    }

    _loadV1(reader) {
        this._shape = new ndarray.Shape(reader, true);
        if (this._shape.dimensions.length == 0) {
            return;
        }
        var context = new ndarray.Context(reader);
        this._dataType = reader.readUint32();
        var dataTypeSize = (this._dataType < ndarray.Array._dataTypeSizeTable.length) ? ndarray.Array._dataTypeSizeTable[this._dataType] : 0;
        var size = dataTypeSize * this._shape.size();
        this._data = reader.read(size);
    }

    _loadV0(reader) {
        this._shape = new ndarray.Shape(reader, false);
        var context = new ndarray.Context(reader);
        this._dataType = reader.readUint32();
        var dataTypeSize = (this._dataType < ndarray.Array._dataTypeSizeTable.length) ? ndarray.Array._dataTypeSizeTable[this._dataType] : 0;
        var size = dataTypeSize * this._shape.size();
        this._data = reader.read(size);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() { 
        return this._shape;
    }

    get data() {
        return this._data;
    }
};

ndarray.Shape = class {

    constructor(reader, uint64) {
        var ndim = reader.readUint32();
        this._dimensions = [];
        for (var i = 0; i < ndim; i++) {
            this._dimensions.push(uint64 ? reader.readUint64() : reader.readUint32());
        }
    }

    get dimensions() {
        return this._dimensions;
    }

    size() {
        var result = 1;
        this._dimensions.forEach((dimension) => {
            result *= dimension;
        });
        return result;
    }
};

ndarray.Context = class {

    constructor(reader) {
        this._deviceType = reader.readUint32();
        this._deviceId = reader.readUint32();
    }
};

ndarray.Reader = class { 

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._end = buffer.length;
    }

    checkSignature(signature) {
        if (this._position + signature.length <= this._end) {
            for (var i = 0; i < signature.length; i++) {
                if (this._buffer[this._position + i] != signature[i]) {
                    return false;
                }
            }
        }
        this._position += signature.length;
        return true;
    }

    read(size) {
        if (this._position + size > this._end) {
            throw new ndarray.Error('Data not available.');
        }
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    readUint16() {
        if (this._position + 2 > this._end) {
            throw new ndarray.Error('Data not available.');
        }
        var value = this._buffer[this._position] | (this._buffer[this._position + 1] << 8);
        this._position += 2;
        return value;
    }

    readUint32() {
        return this.readUint16() | (this.readUint16() << 16);
    }

    readUint64() {
        var value = this.readUint32();
        if (this.readUint32() != 0) {
            throw new ndarray.Error('Large int64 value.');
        }
        return value;
    }
};

ndarray.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'NDArray Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mxnet.ModelFactory;
}