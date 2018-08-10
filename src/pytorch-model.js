/*jshint esversion: 6 */

class PyTorchModelFactory {

    match(buffer, identifier) {
        var extension = identifier.split('.').pop();
        return extension == 'pt' || extension == 'pth';
    }

    open(buffer, identifier, host, callback) { 

        try {
            var unpickler = new pickle.Unpickler(buffer);

            var signature = [ 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            var magic_number = unpickler.load();
            if (!Array.isArray(magic_number) ||
                signature.length != magic_number.length ||
                !signature.every((value, index) => value == magic_number[index])) 
            {
                throw new pickle.Error('Invalid signature.');
            }
            var protocol_version = unpickler.load();
            if (protocol_version != 1001) {
                throw new pickle.Error("Unsupported protocol version '" + protocol_version + "'.");
            }
            var sysInfo = unpickler.load();
            if (!sysInfo.little_endian) {
                throw new pickle.Error("Unsupported system.");
            }

            var functionTable = {};
            functionTable['collections.OrderedDict'] = function (args) { this.items = args; };
            functionTable['torchvision.models.alexnet.AlexNet'] = function () {};
            functionTable['torch.nn.modules.container.Sequential'] = function () {};
            functionTable['torchvision.models.alexnet.AlexNet'] = function () {};
            functionTable['torchvision.models.resnet.ResNet'] = function () {};
            functionTable['torchvision.models.resnet.Bottleneck'] = function () {};
            functionTable['torchvision.models.inception.Inception3'] = function () {};
            functionTable['torchvision.models.inception.InceptionA'] = function () {};
            functionTable['torchvision.models.inception.InceptionB'] = function () {};
            functionTable['torchvision.models.inception.InceptionC'] = function () {};
            functionTable['torchvision.models.inception.InceptionD'] = function () {};
            functionTable['torchvision.models.inception.InceptionE'] = function () {};
            functionTable['torchvision.models.inception.InceptionAux'] = function () {};
            functionTable['torchvision.models.inception.BasicConv2d'] = function () {};
            functionTable['torchvision.models.densenet.DenseNet'] = function () {};
            functionTable['torchvision.models.densenet._DenseBlock'] = function () {};
            functionTable['torchvision.models.densenet._DenseLayer'] = function () {};
            functionTable['torchvision.models.densenet._Transition'] = function () {};
            functionTable['torchvision.models.vgg.VGG'] = function () {};
            functionTable['torch.nn.backends.thnn._get_thnn_function_backend'] = function () {};
            functionTable['torch.nn.modules.conv.Conv2d'] = function () {};
            functionTable['torch.nn.modules.dropout.Dropout'] = function () {};
            functionTable['torch.nn.modules.batchnorm.BatchNorm2d'] = function () {};
            functionTable['torch.nn.modules.activation.ReLU'] = function () {};
            functionTable['torch.nn.modules.linear.Linear'] = function () {};
            functionTable['torch.nn.modules.pooling.MaxPool2d'] = function () {};
            functionTable['torch.nn.modules.pooling.AvgPool2d'] = function () {};
            functionTable['torch.nn.modules.container.Sequential'] = function () {};

            functionTable['argparse.Namespace'] = function (args) { this.args = args; };
            functionTable['fairseq.meters.AverageMeter'] = function () {};
            functionTable['fairseq.meters.TimeMeter'] = function () {};

            functionTable['torch.nn.parameter.Parameter'] = function() {};
            functionTable['torch.FloatStorage'] = function (size) { this.size = size; this.dataType = 'float32'; };
            functionTable['torch.LongStorage'] = function (size) { this.size = size; this.dataType = 'int64'; };

            functionTable['torch._utils._rebuild_tensor'] = function (storage, storage_offset, size, stride) {
                this.__type__ = storage.__type__.replace('Storage', 'Tensor');
                this.storage = storage;
                this.storage_offset = storage_offset;
                this.size = size;
                this.stride = stride;
            };

            functionTable['torch._utils._rebuild_tensor_v2'] = function (storage, storage_offset, size, stride, requires_grad, backward_hooks) {
                this.__type__ = storage.__type__.replace('Storage', 'Tensor');
                this.storage = storage;
                this.storage_offset = storage_offset;
                this.size = size;
                this.stride = stride;
                this.requires_grad = requires_grad;
                this.backward_hooks =  backward_hooks;
            };

            var function_load = (type, args) => {
                if (functionTable[type]) {
                    var func = functionTable[type];
                    return func;
                }
                throw new pickle.Error("Unknown function '" + type + "'.");
            };

            var deserialized_objects = {};

            var persistent_load = (saved_id) => {
                var typename = saved_id.shift();
                var data = saved_id;
                switch (typename) {
                    case 'module':
                        return saved_id[0];
                    case 'storage':
                        var data_type = data.shift();
                        var root_key = data.shift();
                        var location = data.shift();
                        var size = data.shift();
                        var view_metadata = data.shift();
                        var storage = deserialized_objects[root_key];
                        if (!storage) {
                            var constructor = function_load(data_type);
                            storage = { __type__: data_type };
                            constructor.apply(storage, [ size ]);
                            deserialized_objects[root_key] = storage;
                        }
                        if (view_metadata) {
                            var view_key = view_metadata.shift();
                            var view_offset = view_metadata.shift();
                            var view_size = view_metadata.shift();
                            var view = deserialized_objects[view_key];
                            if (!view) {
                                view = null; // storage.slice(view_offset, view_offset + view_size);
                                deserialized_objects[view_key] = view;
                            }
                            return view;
                        }
                        return storage;
                }
                throw new pickle.Error("Unknown persistent load type '" + typename + "'.");
            };

            var result = unpickler.load(function_load, persistent_load);
            var deserialized_storage_keys = unpickler.load();
            deserialized_storage_keys.forEach((key) => {
                if (deserialized_objects[key]) {
                    var value = deserialized_objects[key];
                    // var dataView = unpickler.loadDataView(value.length);
                    // value.load(dataView);
                }
            });

            var model = new PyTorchModel(sysInfo, result, deserialized_storage_keys); 

            PyTorchOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(new PyTorchError(err.message), null);
        }
    }
}

class PyTorchModel { 
    constructor(sysInfo, result, deserialized_storage_keys) {

        this._graphs = [ new PyTorchGraph(sysInfo, result, deserialized_storage_keys) ];
    }

    get properties() {
        var results = [];
        results.push({ name: 'format', value: 'PyTorch' });
        return results;
    }

    get graphs() {
        return this._graphs;
    }

}

class PyTorchGraph {

    constructor(sysInfo, result, deserialized_storage_keys) {
        this._type = result.__type__;

        this._nodes = [];

        this._groups = true;

        var input = 'data';

        result._modules.items.forEach((item) => {
            var group = item[0];
            item[1]._modules.items.forEach((data) => {
                var node = new PyTorchNode(group + data[0], data[1], group, input);
                this._nodes.push(node);
                input = node.name;
            });
        });

    }

    get type() {
        return this._type;
    }

    get groups() {
        return this._groups;
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [];
    }

    get nodes() {
        return this._nodes;
    }

}

class PyTorchNode {

    constructor(name, data, group, input) {
        this._name = name;
        this._operator = data.__type__.split('.').pop();
        this._group = group;

        this._inputs = [];

        this._inputs.push({ name: 'input', connections: [ { id: input } ] });
        data._parameters.items.forEach((parameter) => {
            var input = {};
            input.name = parameter[0];
            input.connections = [];
            this._inputs.push(input);
            if (parameter[1] && parameter[1].storage) {
                var connection = {};
                connection.initializer = new PyTorchTensor(parameter[1]);
                connection.type = connection.initializer.type.toString();
                input.connections.push(connection);
            }
        });

        this._outputs = [];
        this._outputs.push({ name: 'output', connections: [ { id: this._name } ] });

        this._attributes = [];

        Object.keys(data).forEach((key) => {
            if (!key.startsWith('_')) {
                this._attributes.push(new PyTorchAttribute(this, key, data[key]));
            }
        });
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group;
    }

    get operator() {
        return this._operator;
    }

    get category() {
        return PyTorchOperatorMetadata.operatorMetadata.getOperatorCategory(this._operator);
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
}

class PyTorchAttribute {

    constructor(node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return JSON.stringify(this._value);
    }

    get visible() {
        return PyTorchOperatorMetadata.operatorMetadata.getAttributeVisible(this._node.operator, this._name);
    }
}

class PyTorchTensor {
    constructor(tensor) {
        this._type = new PyTorchTensorType(tensor.storage.dataType, tensor.size);
    }

    get kind() {
        return 'Tensor';
    }

    get type() {
        return this._type;
    }
}

class PyTorchTensorType {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }
}

class PyTorchOperatorMetadata {

    static open(host, callback) {
        if (PyTorchOperatorMetadata.operatorMetadata) {
            callback(null, PyTorchOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/pytorch-metadata.json', (err, data) => {
                PyTorchOperatorMetadata.operatorMetadata = new PyTorchOperatorMetadata(data);
                callback(null, PyTorchOperatorMetadata.operatorMetadata);
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
                    return JSON.stringify(attribute.default) == JSON.stringify(value);
                 }
            }
        }
        return true;
    }
}

class PyTorchError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading PyTorch model.';
    }
}

var pickle = pickle || {};

pickle.Unpickler = class {

    constructor(buffer) {
        this._reader = new pickle.Reader(buffer, 0);
    }

    load(function_load, persistent_load) {
        var reader = this._reader;
        var stack = [];
        var marker = [];
        var table = {};    
        while (reader.offset < reader.length) {
            var opcode = reader.readByte();
            // console.log(reader.offset.toString() + ': ' + opcode + ' ' + String.fromCharCode(opcode));
            switch (opcode) {
                case pickle.OpCode.PROTO:
                    var version = reader.readByte();
                    if (version != 2) {
                        throw new pickle.Error('Unhandled pickle protocol version: ' + version);
                    }
                    break;
                case pickle.OpCode.GLOBAL:
                    var module = reader.readLine();
                    var name = reader.readLine();
                    var type = [ module, name ].join('.');
                    stack.push(type);
                    break;
                case pickle.OpCode.PUT:
                    var key = reader.readLine();
                    table[key] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.OBJ:
                    var obj = new (stack.pop())();
                    var index = marker.pop();
                    for (var position = index; position < stack.length; pos += 2) {
                        obj[stack[position]] = stack[position + 1];
                    }
                    stack = stack.slice(0, index);
                    stack.push(obj);
                    break;
                case pickle.OpCode.GET:
                    var key = reader.readLine();
                    stack.push(table[key]);
                    break;
                case pickle.OpCode.POP:
                    stack.pop();
                    break;
                case pickle.OpCode.POP_MARK:
                    var index = marker.pop();
                    stack = stack.slice(0, index);
                    break;
                case pickle.OpCode.DUP:
                    var value = stack[stack.length-1];
                    stack.push(value);
                    break;
                case pickle.OpCode.PERSID:
                    throw new pickle.Error("Unknown opcode 'PERSID'.");
                case pickle.OpCode.BINPERSID:
                    var saved_id = stack.pop();
                    var saved_obj = persistent_load(saved_id);
                    stack.push(saved_obj);
                    break;
                case pickle.OpCode.REDUCE:
                    var args = stack.pop();
                    var type = stack.pop();
                    var func = function_load(type);
                    var obj = { __type__: type };
                    func.apply(obj, args);
                    stack.push(obj);
                    break;
                case pickle.OpCode.NEWOBJ:
                    var args = stack.pop();
                    var type = stack.pop();
                    var constructor = function_load(type);
                    var obj = { __type__: type };
                    constructor.apply(obj, args);
                    stack.push(obj);
                    break;
                case pickle.OpCode.BINGET:
                    var index = reader.readByte();
                    stack.push(table[index]);
                    break;
                case pickle.OpCode.LONG_BINGET:
                    var index = reader.readUInt32();
                    stack.push(table[index]);
                    break;
                case pickle.OpCode.BINPUT:
                    var index = reader.readByte();
                    table[index] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.LONG_BINPUT:
                    var index = reader.readUInt32();
                    table[index] = stack[stack.length - 1];
                    break;
                case pickle.OpCode.BININT:
                    stack.push(reader.readInt32());
                    break;
                case pickle.OpCode.BININT1:
                    stack.push(reader.readByte());
                    break;
                case pickle.OpCode.LONG:
                    stack.push(parseInt(reader.readLine()));
                    break;
                case pickle.OpCode.BININT2:
                    stack.push(reader.readUInt16());
                    break;
                case pickle.OpCode.FLOAT:
                    stack.push(parseFloat(reader.readLine()));
                    break;
                case pickle.OpCode.BINFLOAT:
                    stack.push(reader.readFloat64());
                    break;
                case pickle.OpCode.INT:
                    var value = reader.readLine();
                    if (value == '01') {
                        stack.push(true);
                    }
                    else if (value == '00') {
                        stack.push(false);
                    }
                    else {
                        stack.push(parseInt(value));
                    }
                    break;
                case pickle.OpCode.EMPTY_LIST:
                    stack.push([]);
                    break;
                case pickle.OpCode.EMPTY_TUPLE:
                    stack.push([]);
                    break;
                case pickle.OpCode.DICT:
                    var index = marker.pop();
                    var obj = {};
                    for (var position = index; position < stack.length; position += 2) {
                        obj[stack[position]] = stack[position + 1];
                    }
                    stack = stack.slice(0, index);
                    stack.push(obj);
                    break;
                case pickle.OpCode.LIST:
                    var index = marker.pop();
                    var list = stack.slice(index);
                    stack = stack.slice(0, index);
                    stack.push(list);
                    break;
                case pickle.OpCode.TUPLE:
                    var index = marker.pop();
                    var list = stack.slice(index);
                    stack = stack.slice(0, index);
                    stack.push(list);
                    break;
                case pickle.OpCode.SETITEM:
                    var value = stack.pop();
                    var key = stack.pop();
                    var obj = stack[stack.length - 1];
                    obj[key] = value;
                    break;
                case pickle.OpCode.SETITEMS:
                    var index = marker.pop();
                    var obj = stack[index - 1];
                    for (var position = index; position < stack.length; position += 2) {
                        obj[stack[position]] = stack[position + 1];
                    }
                    stack = stack.slice(0, index);
                    break;
                case pickle.OpCode.EMPTY_DICT:
                    stack.push({});
                    break;
                case pickle.OpCode.APPEND:
                    var append = stack.pop();
                    stack[stack.length-1].push(append);
                    break;
                case pickle.OpCode.APPENDS:
                    var index = marker.pop();
                    var list = stack[index - 1];
                    list.push.apply(list, stack.slice(index));
                    stack = stack.slice(0, index);
                    break;
                case pickle.OpCode.STRING:
                    var value = reader.readLine();
                    stack.push(value.substr(1, value.length - 2));
                    break;
                case pickle.OpCode.BINSTRING:
                    stack.push(reader.readString(reader.readUInt32()));
                    break;
                case pickle.OpCode.SHORT_BINSTRING:
                    stack.push(reader.readString(reader.readByte()));
                    break;
                case pickle.OpCode.UNICODE:
                    stack.push(reader.readLine());
                    break;
                case pickle.OpCode.BINUNICODE:
                    stack.push(reader.readString(reader.readUInt32(), 'utf-8'));
                    break;
                case pickle.OpCode.BUILD:
                    var dict = stack.pop();
                    var obj = stack.pop();
                    for (var p in dict) {
                        obj[p] = dict[p];
                    }
                    stack.push(obj);
                    if (obj.unpickle) {
                        obj.unpickle(reader);
                    }
                    break;
                case pickle.OpCode.MARK:
                    marker.push(stack.length);
                    break;
                case pickle.OpCode.NEWTRUE:
                    stack.push(true);
                    break;
                case pickle.OpCode.NEWFALSE:
                    stack.push(false);
                    break;
                case pickle.OpCode.LONG1:
                    var length = reader.readByte();
                    var data = reader.readBytes(length);
                    var value = 0;
                    switch (length) {
                        case 0: value = 0; break;
                        case 1: value = data[0]; break;
                        case 2: value = data[1] << 8 | data[0]; break;
                        case 3: value = data[2] << 16 | data[1] << 8 | data[0]; break;
                        case 4: value = data[3] << 24 | data[2] << 16 | data[0]; break;
                        default: value = Array.prototype.slice.call(data, 0); break; 
                    }
                    stack.push(value);
                    break;
                case pickle.OpCode.LONG4:
                    var length = reader.readUInt32(i);
                    // TODO decode LONG4
                    var data = reader.readBytes(length);
                    stack.push(data);
                    break;
                case pickle.OpCode.TUPLE1:
                    var a = stack.pop();
                    stack.push([a]);
                    break;
                case pickle.OpCode.TUPLE2:
                    var b = stack.pop();
                    var a = stack.pop();
                    stack.push([a, b]);
                    break;
                case pickle.OpCode.TUPLE3:
                    var c = stack.pop();
                    var b = stack.pop();
                    var a = stack.pop();
                    stack.push([a, b, c]);
                    break;
                case pickle.OpCode.NONE:
                    stack.push(null);
                    break;
                case pickle.OpCode.STOP:
                    return stack.pop();
                default:
                    throw new pickle.Error("Unknown opcode '" + opcode + "'.");
            }
        }

        throw new pickle.Error('Unexpected end of file.');
    }

    loadDataView(length) {
        return this._reader.readDataView(length);
    }
};

// https://svn.python.org/projects/python/trunk/Lib/pickletools.py
pickle.OpCode = {
    MARK: 40,            // '('
    EMPTY_TUPLE: 41,     // ')'
    STOP: 46,            // '.'
    POP: 48,             // '0'
    POP_MARK: 49,        // '1'
    DUP: 50,             // '2'
    FLOAT: 70,           // 'F'
    BINFLOAT: 71,        // 'G'
    INT: 73,             // 'I'
    BININT: 74,          // 'J'
    BININT1: 75,         // 'K'
    LONG: 76,            // 'L'
    BININT2: 77,         // 'M'
    NONE: 78,            // 'N'
    PERSID: 80,          // 'P'
    BINPERSID: 81,       // 'Q'
    REDUCE: 82,          // 'R'
    STRING: 83,          // 'S'
    BINSTRING: 84,       // 'T'
    SHORT_BINSTRING: 85, // 'U'
    UNICODE: 86,         // 'V'
    BINUNICODE: 88,      // 'X'
    EMPTY_LIST: 93,      // ']'
    APPEND: 97,          // 'a'
    BUILD: 98,           // 'b'
    GLOBAL: 99,          // 'c'
    DICT: 100,           // 'd'
    APPENDS: 101,        // 'e'
    GET: 103,            // 'g'
    BINGET: 104,         // 'h'
    LONG_BINGET: 106,    // 'j'
    LIST: 108,           // 'l'
    OBJ: 111,            // 'o'
    PUT: 112,            // 'p'
    BINPUT: 113,         // 'q'
    LONG_BINPUT: 114,    // 'r'
    SETITEM: 115,        // 's'
    TUPLE: 116,          // 't'
    SETITEMS: 117,       // 'u'
    EMPTY_DICT: 125,     // '}'
    PROTO: 128,
    NEWOBJ: 129,
    TUPLE1: 133,         // '\x85'
    TUPLE2: 134,         // '\x86'
    TUPLE3: 135,         // '\x87'
    NEWTRUE: 136,        // '\x88'
    NEWFALSE: 137,       // '\x89'
    LONG1: 138,          // '\x8a'
    LONG4: 139           // '\x8b'
};

pickle.Reader = class { 

    constructor(buffer) {
        if (buffer) {
            this._buffer = buffer;
            this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            this._offset = 0;
        }
        pickle.Reader._utf8Decoder = pickle.Reader._utf8Decoder || new TextDecoder('utf-8');
        pickle.Reader._asciiDecoder = pickle.Reader._asciiDecoder || new TextDecoder('ascii');
    }

    get length() {
        return this._buffer.byteLength;
    }

    get offset() {
        return this._offset;
    }

    readByte() {
        var value = this._dataView.getUint8(this._offset);
        this._offset++;
        return value;
    }

    readBytes(length) {
        var data = this._buffer.subarray(this._offset, this._offset + length);
        this._offset += length;
        return data;
    }

    readUInt16() {
        var value = this._dataView.getUint16(this._offset, true);
        this._offset += 2;
        return value;
    }

    readInt32() {
        var value = this._dataView.getInt32(this._offset, true);
        this._offset += 4;
        return value;
    }

    readUInt32() {
        var value = this._dataView.getUint32(this._offset, true);
        this._offset += 4;
        return value;
    }

    readFloat32() {
        var value = this._dataView.getFloat32(this._offset, true);
        this._offset += 4;
        return value;
    }

    readFloat64() {
        var value = this._dataView.getFloat64(this._offset, false);
        this._offset += 8;
        return value;
    }

    skipBytes(length) {
        this._offset += length;
    }

    readString(size, encoding) {
        return pickle.Reader.decodeString(this.readBytes(size), encoding);
    }

    readLine() {
        var index = this._buffer.indexOf(0x0A, this._offset);
        if (index == -1) {
            throw new pickle.Error("Could not find end of line.");
        }
        var size = index - this._offset;
        var text = this.readString(size, 'ascii');
        this.skipBytes(1);
        return text;
    }

    readDataView(length) {
        var dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset + this._offset, length);
        this._offset += length;
        return dataView;
    }

    static decodeString(data, encoding) {
        var text = '';
        if (encoding == 'utf-8') {
            text = pickle.Reader._utf8Decoder.decode(data);    
        }
        else {
            text = pickle.Reader._asciiDecoder.decode(data);
        }
        return text.replace(/\0/g, '');
    }
};


pickle.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Unpickle Error';
    }
}
