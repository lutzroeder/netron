/*jshint esversion: 6 */

// Experimental

class MXNetModelFactory {

    match(buffer, identifier) {
        if (identifier.endsWith('-symbol.json')) {
            return true;
        }
        var extension = identifier.split('.').pop();
        if (extension == 'model') {
            return true;
        }
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
        var extension = identifier.split('.').pop();
        switch (extension) {
            case 'json':
                this._openSymbol(buffer, callback);
                break;
            case 'model':
                host.import('/unzip.js', (err) => {
                    if (err) {
                        callback(err, null);
                        return;
                    }
                    this._openModel(buffer, host, callback);
                });
                break;
            default:
                callback(new MXNetError('Unsupported file extension.'));
                break;
        }
    }

    _openSymbol(buffer, callback) {
        try {
            var decoder = new TextDecoder('utf-8');
            var symbol = JSON.parse(decoder.decode(buffer));
            var model = new MXNetModel(null, symbol, null, {});
            MXNetOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(new MXNetError(err.message), null);
        }
    }

    _openModel(buffer, host, callback) {
        var entries = {};
        try {
            var archive = new zip.Archive(buffer, host.inflate);
            archive.entries.forEach((entry) => {
                entries[entry.name] = entry;
            });
        }
        catch (err) {
            callback(new MXNetError('Failed to decompress ZIP archive. ' + err.message), null);
            return;
        }

        var manifestEntry = entries['MANIFEST.json'];
        var rootFolder = '';
        if (!manifestEntry) {
            var folders = Object.keys(entries).filter((name) => name.endsWith('/')).filter((name) => entries[name + 'MANIFEST.json']);
            if (folders.length != 1) {
                callback(new MXNetError('Manifest not found.'), null);
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
            callback(new MXNetError('Failed to read manifest. ' + err.message), null);
            return;
        }

        if (!manifest.Model) {
            callback(new MXNetError('Manifest does not contain model.'), null);
            return;
        }

        var modelFormat = manifest.Model['Model-Format'];
        if (modelFormat && modelFormat != 'MXNet-Symbolic') {
            callback(new MXNetError('Model format \'' + modelFormat + '\' not supported.'), null);
            return;
        }

        if (!manifest.Model.Symbol) {
            callback(new MXNetError('Manifest does not contain symbol entry.'), null);
            return;
        }

        var symbol = null;
        try {
            var symbolEntry = entries[rootFolder + manifest.Model.Symbol];
            symbol = JSON.parse(decoder.decode(symbolEntry.data));
        }
        catch (err) {
            callback(new MXNetError('Failed to load symbol entry. ' + err.message), null);
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

        var parameters = {};
        try {
            if (manifest.Model.Parameters) {
                var parametersEntry = entries[rootFolder + manifest.Model.Parameters];
                if (parametersEntry) {
                    var parametersData = parametersEntry.data;
                    var stream = new ndarray.Stream(parametersData);
                    Object.keys(stream.arrays).forEach((key) => {
                        var name = key;
                        if (name.startsWith('arg:') || name.startsWith('aux:')) {
                            name = key.substring(4);
                        }
                        parameters[name] = stream.arrays[key];
                    });
                }
            }
        }
        catch (err) {
        }

        try {
            var model = new MXNetModel(manifest, symbol, signature, parameters);
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

    constructor(manifest, symbol, signature, parameters) {
        if (!symbol) {
            throw new MXNetError('JSON file does not contain MXNet data.');
        }
        if (!symbol.hasOwnProperty('nodes')) {
            throw new MXNetError('JSON file does not contain an MXNet \'nodes\' property.');
        }
        if (!symbol.hasOwnProperty('arg_nodes')) {
            throw new MXNetError('JSON file does not contain an MXNet \'arg_nodes\' property.');
        }
        if (!symbol.hasOwnProperty('heads')) {
            throw new MXNetError('JSON file does not contain an MXNet \'heads\' property.');
        }

        if (symbol.attrs && symbol.attrs.mxnet_version && symbol.attrs.mxnet_version.length == 2 && symbol.attrs.mxnet_version[0] == 'int') {
            var version = symbol.attrs.mxnet_version[1];
            var revision = version % 100;
            var minor = Math.floor(version / 100) % 100;
            var major = Math.floor(version / 10000) % 100;
            this._version = major.toString() + '.' + minor.toString() + '.' + revision.toString(); 
        }

        if (!this._version && manifest && manifest.Engine && manifest.Engine.MXNet) {
            this._version = manifest.Engine.MXNet.toString();
        }

        this._graphs = [ new MXNetGraph(manifest, symbol, signature, parameters) ];
    }

    get properties() {
        var results = [];
        results.push({ name: 'format', value: 'MXNet' + (this._version ? (' v' + this._version) : '') });
        return results;
    }

    get graphs() {
        return this._graphs;
    }

}

class MXNetGraph {

    constructor(manifest, symbol, signature, parameters)
    {
        var nodes = symbol.nodes;

        this._nodes = [];

        this._operators = [];
        nodes.forEach((node) => {
            if (node.op && node.op != 'null') { 
                this._operators[node.op] = (this._operators[node.op] || 0) + 1;
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
                return MXNetGraph._updateOutput(nodes, input);
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
            var output = {};
            output.id = MXNetGraph._updateOutput(nodes, head);
            output.name = nodes[output.id[0]] ? nodes[output.id[0]].name : ('output' + ((index == 0) ? '' : (index + 1).toString()));
            output.type = 'T';
            var outputSignature = outputs[output.name];
            if (outputSignature && outputSignature.data_shape) {
                output.type = '?' + '[' + outputSignature.data_shape.toString() + ']';
            }
            this._outputs.push(output);
        });

        nodes.forEach((node, index) => {
            if (!argumentMap[index]) {
                this._nodes.push(new MXNetNode(node, argumentMap, parameters));
            }
        });

        this._inputs = [];
        Object.keys(argumentMap).forEach((key) => {
            var argument = argumentMap[key];
            if ((!argument.inputs || argument.inputs.length == 0) &&
                (argument.outputs && argument.outputs.length == 1)) {
                var input = {};
                input.id = argument.outputs[0];
                input.name = argument.name;
                input.type = 'T';
                var inputSignature = inputs[input.name];
                if (inputSignature && inputSignature.data_shape) {
                    input.type = '?' + '[' + inputSignature.data_shape.toString() + ']';
                }
                this._inputs.push(input);
            }
        });

        this._inputs = this._inputs.map((input) => {
            input.id = '[' + input.id.join(',') + ']';
            return input;
        });
        this._outputs = this._outputs.map((output) => {
            output.id = '[' + output.id.join(',') + ']';
            return output;
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
        while (outputIndex >= node.outputs.length) {
            node.outputs.push([ nodeIndex, node.outputs.length ]);
        }
        return [ nodeIndex, outputIndex ];
    }
}

class MXNetNode {

    constructor(json, argumentMap, parameters) {
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
        if (this._operator == 'RNN') {
            this._inputs = this._inputs.map((input) => {
                var argumentNodeIndex = input[0];
                var argument = argumentMap[argumentNodeIndex];
                if (argument && argument.op == 'null' && argument.name &&
                    argument.name.endsWith('_parameters') && argument.attr && argument.attr.__init__) {
                    this._attributes.push(new MXNetAttribute(this, argument.name, argument.attr.__init__));
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
                    this._initializers[id] = new MXNetTensor('Initializer', argument.name, parameter.dataType, parameter.shape.dimensions, parameter.data);
                    delete argumentMap[argumentNodeIndex];
                }
                else {
                    var prefix = this._name + '_';
                    if (prefix.endsWith('_fwd_')) {
                        prefix = prefix.slice(0, -4);
                    }
                    if (argument.name && argument.name.startsWith(prefix)) {
                        var dataType = '?';
                        var shape = '';
                        if (argument.attrs && argument.attrs.__dtype__ && argument.attrs.__shape__) {
                            try {
                                dataType = parseInt(argument.attrs.__dtype__);
                                shape = argument.attrs.__shape__.replace('(', '').replace(')', '').split(' ').join('');
                            }
                            catch (err) {
                            }
                        }
                        this._initializers[id] = new MXNetTensor('Initializer', argument.name, dataType, shape, null);
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
                    connection.id = initializer.name || connection.id;
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
    
    constructor(kind, name, dataType, shape, data) {
        this._kind = kind;
        this._name = name;
        this._dataType = dataType;
        this._shape = shape;
        this._data = data;
        MXNetTensor._dataTypeNameTable = MXNetTensor._dataTypeTable || [ 'float32', 'float64', 'float16', 'uint8', 'int32', 'int8', 'int64' ];
    }

    get kind() {
        return 'Initializer';
    }

    get name() {
        return this._name;
    }

    get type() {
        var dataType = '?';
        if (this._dataType || this._dataType === 0) {
            if (this._dataType < MXNetTensor._dataTypeNameTable.length) {
                dataType = MXNetTensor._dataTypeNameTable[this._dataType];
            }
            else {
                dataType = this._dataType.toString();
            }
        }
        var shape = Array.isArray(this._shape) ? this._shape.join(',') : this._shape.toString();
        return dataType + '[' + shape + ']';
    }

    get value() {
        var result = this._decode(Number.MAX_SAFE_INTEGER);
        if (result.error) {
            return null;
        }
        return result.value;
    }

    toString() {
        var result = this._decode(10000);
        if (result.error) {
            return result.error;
        }
        return JSON.stringify(result.value, null, 4);
    }

    _decode(limit) {

        var result = {};

        if (!this._data) {
            return { error: 'Tensor data is empty.' };
        }

        if ((!this._dataType && this._dataType !== 0) || this._dataType == '?') {
            return { error: 'Tensor has no data type.' };
        }

        if (this._dataType >= MXNetTensor._dataTypeNameTable.length) {
            return { error: 'Tensor has unknown data type.' };
        }

        if (!Array.isArray(this._shape) && this._shape.length < 1) {
            return { error: 'Tensor has unknown shape.' };
        }

        var context = {};
        context.index = 0;
        context.count = 0;
        context.limit = limit;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        return { value: this._decodeDimension(context, 0) };
    }

    _decodeDimension(context, dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
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
                        results.push(MXNetTensor._decodeNumberFromFloat16(context.data.getUint16(context.index, true)));
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
                        results.push(new Int64(context.data.subarray(context.index, context.index + 8)));
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
                results.push(this._decodeDimension(context, dimension + 1));
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
                    value = MXNetOperatorMetadata._formatTuple(value); 
                    return !MXNetOperatorMetadata._isEquivalent(attribute.default, value);
                }
            }
        }
        return true;
    }

    static _formatTuple(value) {
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

    static _isEquivalent(a, b) {
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
                    if (!KerasOperatorMetadata._isEquivalent(a[length], b[length])) {
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
            if (!(b.hasOwnProperty(key) && KerasOperatorMetadata._isEquivalent(a[key], b[key]))) {
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
            return schema;
        }
        return '';
    }
}

class MXNetError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading MXNet model.';
    }
}

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