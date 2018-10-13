/*jshint esversion: 6 */

// Experimental

class PyTorchModelFactory {

    match(context, host) {
        var extension = context.identifier.split('.').pop();
        if (extension == 'pt' || extension == 'pth') {
            return true;
        }
        if (extension == 'pkl') {
            var buffer = context.buffer;
            var torch = [ 0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > torch.length) {
                if (torch.every((value, index) => value == buffer[index])) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('pickle', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            PyTorchOperatorMetadata.open(host, (err, metadata) => {
                this._openModel(context, host, callback);
            });
        });
    }

    _openModel(context, host, callback) {
        try {
            var unpickler = new pickle.Unpickler(context.buffer);

            var signature = [ 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            var magic_number = unpickler.load();
            if (!Array.isArray(magic_number) ||
                signature.length != magic_number.length ||
                !signature.every((value, index) => value == magic_number[index])) 
            {
                callback(new PyTorchError('Invalid signature.', null));
                return;
            }
            var protocol_version = unpickler.load();
            if (protocol_version != 1001) {
                callback(new PyTorchError("Unsupported protocol version '" + protocol_version + "'.", null));
                return;
            }
            var sysInfo = unpickler.load();
            if (!sysInfo.little_endian) {
                callback(new PyTorchError('Unsupported endian format.'));
                return;
            }
            if (sysInfo.protocol_version != 1001) {
                callback(new PyTorchError("Unsupported protocol version '" + sysInfo.protocol_version + "'.", null));
                return;
            }
            if (sysInfo.type_sizes) {
                if ((sysInfo.type_sizes.int && sysInfo.type_sizes.int != 4) ||
                    (sysInfo.type_sizes.long && sysInfo.type_sizes.long != 4) ||
                    (sysInfo.type_sizes.short && sysInfo.type_sizes.short != 2))
                {
                    callback(new PyTorchError('Unsupported type sizes.'));
                    return;
                }
            }

            var constructorTable = {};
            var functionTable = {};

            constructorTable['argparse.Namespace'] = function (args) { this.args = args; };
            constructorTable['torch.nn.modules.activation.ReLU'] = function () {};
            constructorTable['torch.nn.modules.activation.Tanh'] = function () {};
            constructorTable['torch.nn.modules.activation.Sigmoid'] = function () {};
            constructorTable['torch.nn.modules.batchnorm.BatchNorm2d'] = function () {};
            constructorTable['torch.nn.modules.container.Sequential'] = function () {};
            constructorTable['torch.nn.modules.conv.Conv1d'] = function () {};
            constructorTable['torch.nn.modules.conv.Conv2d'] = function () {};
            constructorTable['torch.nn.modules.conv.Conv3d'] = function () {};
            constructorTable['torch.nn.modules.conv.ConvTranspose1d'] = function () {};
            constructorTable['torch.nn.modules.conv.ConvTranspose2d'] = function () {};
            constructorTable['torch.nn.modules.conv.ConvTranspose3d'] = function () {};
            constructorTable['torch.nn.modules.dropout.Dropout'] = function () {};
            constructorTable['torch.nn.modules.dropout.Dropout2d'] = function () {};
            constructorTable['torch.nn.modules.linear.Linear'] = function () {};
            constructorTable['torch.nn.modules.pooling.AvgPool2d'] = function () {};
            constructorTable['torch.nn.modules.pooling.MaxPool2d'] = function () {};
            constructorTable['torch.nn.modules.rnn.LSTM'] = function () {};
            constructorTable['torch.nn.modules.sparse.Embedding'] = function () {};
            constructorTable['torchvision.models.alexnet.AlexNet'] = function () {};
            constructorTable['torchvision.models.densenet.DenseNet'] = function () {};
            constructorTable['torchvision.models.densenet._DenseBlock'] = function () {};
            constructorTable['torchvision.models.densenet._DenseLayer'] = function () {};
            constructorTable['torchvision.models.densenet._Transition'] = function () {};
            constructorTable['torchvision.models.inception.BasicConv2d'] = function () {};
            constructorTable['torchvision.models.inception.Inception3'] = function () {};
            constructorTable['torchvision.models.inception.InceptionAux'] = function () {};
            constructorTable['torchvision.models.inception.InceptionA'] = function () {};
            constructorTable['torchvision.models.inception.InceptionB'] = function () {};
            constructorTable['torchvision.models.inception.InceptionC'] = function () {};
            constructorTable['torchvision.models.inception.InceptionD'] = function () {};
            constructorTable['torchvision.models.inception.InceptionE'] = function () {};
            constructorTable['torchvision.models.resnet.Bottleneck'] = function () {};
            constructorTable['torchvision.models.resnet.ResNet'] = function () {};
            constructorTable['torchvision.models.vgg.VGG'] = function () {};
            constructorTable['torch.nn.backends.thnn._get_thnn_function_backend'] = function () {};
            constructorTable['torch.nn.parameter.Parameter'] = function(data, requires_grad) { this.data = data; this.requires_grad = requires_grad; };
            constructorTable['torch.FloatStorage'] = function (size) { this.size = size; this.dataTypeSize = 4; this.dataType = 'float32'; };
            constructorTable['torch.DoubleStorage'] = function (size) { this.size = size; this.dataTypeSize = 8; this.dataType = 'float64'; };
            constructorTable['torch.LongStorage'] = function (size) { this.size = size; this.dataTypeSize = 4; this.dataType = 'int64'; };

            functionTable['torch._utils._rebuild_tensor'] = function (storage, storage_offset, size, stride) {
                var obj = {};
                obj.__type__ = storage.__type__.replace('Storage', 'Tensor');
                obj.storage = storage;
                obj.storage_offset = storage_offset;
                obj.size = size;
                obj.stride = stride;
                return obj;
            };
            functionTable['torch._utils._rebuild_tensor_v2'] = function (storage, storage_offset, size, stride, requires_grad, backward_hooks) {
                var obj = {};
                obj.__type__ = storage.__type__.replace('Storage', 'Tensor');
                obj.storage = storage;
                obj.storage_offset = storage_offset;
                obj.size = size;
                obj.stride = stride;
                obj.requires_grad = requires_grad;
                obj.backward_hooks =  backward_hooks;
                return obj;
            };

            var function_call = (name, args) => {
                if (name == 'collections.OrderedDict') {
                    if (args.length == 0) {
                        return [];
                    }
                    return args[0].map((arg) => {
                        var item = arg[1] || {};
                        item.__id__ = arg[0];
                        return item;
                    });
                }
                var func = functionTable[name];
                if (func) {
                    return func.apply(null, args);
                }
                var obj = { __type__: name };
                var constructor = constructorTable[name];
                if (constructor) {
                    constructor.apply(obj, args);
                }
                else {
                    host.exception(new SklearnError("Unknown function '" + name + "'."), false);
                }
                return obj;
            };

            var module_source_map = {};
            var deserialized_objects = {};

            var persistent_load = (saved_id) => {
                var typename = saved_id.shift();
                var data = saved_id;
                switch (typename) {
                    case 'module':
                        module_source_map[data[0]] = data[2];
                        return data[0];
                    case 'storage':
                        var data_type = data.shift();
                        var root_key = data.shift();
                        var location = data.shift();
                        var size = data.shift();
                        var view_metadata = data.shift();
                        var storage = deserialized_objects[root_key];
                        if (!storage) {
                            storage = function_call(data_type, [ size ]);
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

            var root = unpickler.load(function_call, persistent_load);
            var deserialized_storage_keys = unpickler.load();
            deserialized_storage_keys.forEach((key) => {
                if (deserialized_objects[key]) {
                    var storage = deserialized_objects[key];
                    storage.data = unpickler.read(storage.dataTypeSize * storage.size);
                }
            });

            if (!root._modules) {
                throw new PyTorchError('Root object does not contain modules.');
            }

            var model = new PyTorchModel(sysInfo, root); 
            callback(null, model);
        }
        catch (error) {
            host.exception(error, false);
            callback(new PyTorchError(error.message), null);
        }
    }
}

class PyTorchModel { 

    constructor(sysInfo, root) {
        this._graphs = [ new PyTorchGraph(sysInfo, root) ];
    }

    get format() {
        return 'PyTorch';
    }

    get graphs() {
        return this._graphs;
    }

}

class PyTorchGraph {

    constructor(sysInfo, root) {
        this._type = root.__type__;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;

        var input = 'data';
        this._inputs.push(new PyTorchArgument(input, true, [ new PyTorchConnection(input, null, null) ]));

        var outputs = this._loadModule(root, [], [ input ]);
        outputs.forEach((output) => {
            this._outputs.push(new PyTorchArgument(output, true, [ new PyTorchConnection(output, null, null) ]));
        });
    }

    _loadModule(parent, groups, inputs) {

        if (parent.__type__ &&
            !parent.__type__.startsWith('torch.nn.modules.container.')) {
            var node = new PyTorchNode(parent, groups, inputs);
            this._nodes.push(node);
            return [];
        }

        if (!parent._modules) {
            throw new PyTorchError('Module does not contain modules.');
        }

        parent._modules.forEach((module) => {
            switch (module.__type__) {
                case 'torch.nn.modules.container.Sequential':
                    groups.push(module.__id__);
                    inputs = this._loadModule(module, groups, inputs);
                    groups.pop(module.__id__);
                    break;
                case 'torchvision.models.densenet._Transition':
                case 'torchvision.models.resnet.Bottleneck':
                case 'torchvision.models.densenet._DenseBlock':
                case 'torchvision.models.densenet._DenseLayer':
                case 'torchvision.models.inception.BasicConv2d':
                case 'torchvision.models.inception.InceptionAux':
                case 'torchvision.models.inception.InceptionA':
                case 'torchvision.models.inception.InceptionB':
                case 'torchvision.models.inception.InceptionC':
                case 'torchvision.models.inception.InceptionD':
                case 'torchvision.models.inception.InceptionE':
                    groups.push(module.__id__);
                    inputs = this._loadSource(module, groups, inputs);
                    groups.pop(module.__id__);
                    break; 
                default:
                    var node = new PyTorchNode(module, groups, inputs);
                    this._nodes.push(node);
                    inputs = [ node.name ];
                    break;
            }
        });

        return inputs;
    }

    _loadSource(parent, groups, inputs) {
        var map = {};
        parent._modules.forEach((module) => {
            map[module.__id__] = module;
        });

        var node = new PyTorchNode(parent, groups, inputs);
        this._nodes.push(node);
        inputs = [ node.name ];

        return inputs;
    }

    get type() {
        return this._type;
    }

    get groups() {
        return this._groups;
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
}

class PyTorchArgument {
    constructor(name, visible, connections) {
        this._name = name;
        this._visible = visible;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get connections() {
        return this._connections;
    }
}

class PyTorchConnection {
    constructor(id, type, initializer) {
        this._id = id;
        this._type = type;
        this._initializer = initializer;
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

class PyTorchNode {

    constructor(module, groups, connections) {
        this._group = groups.join('/');
        groups.push(module.__id__);
        this._name = groups.join('/');
        groups.pop();
        this._operator = module.__type__.split('.').pop();

        this._inputs = [];
        this._inputs.push(new PyTorchArgument('input', true, connections.map((connection) => {
            return new PyTorchConnection(connection, null, null);
        })));

        var initializers = [];
        if (module._parameters) {
            module._parameters.forEach((parameter) => {
                initializers.push(parameter);
            });
        }
        if (module._buffers) {
            module._buffers.forEach((buffer) => {
                initializers.push(buffer);
            });
        }

        initializers.forEach((parameter) => {
            if (parameter && (parameter.data || parameter.storage)) {
                var initializer = null;
                if (parameter.data) {
                    initializer = new PyTorchTensor(parameter.data);
                }
                else if (parameter.storage) {
                    initializer = new PyTorchTensor(parameter);
                }
                var visible = (this._operator != 'LSTM' || initializer == null);
                this._inputs.push(new PyTorchArgument(parameter.__id__, visible, [ new PyTorchConnection(null, null, initializer) ]));
            }
        });

        this._outputs = [];
        this._outputs.push(new PyTorchArgument('output', true, [ new PyTorchConnection(this._name, null, null) ]));

        this._attributes = [];
        Object.keys(module).forEach((key) => {
            if (!key.startsWith('_')) {
                this._attributes.push(new PyTorchAttribute(this, key, module[key]));
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
        var schema = PyTorchOperatorMetadata.operatorMetadata.getSchema(this._operator);
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
}

class PyTorchAttribute {

    constructor(node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;

        var schema = PyTorchOperatorMetadata.operatorMetadata.getAttributeSchema(this._node.operator, this._name);
        if (schema) {
            if (schema.hasOwnProperty('visible') && !schema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default')) {
                if (JSON.stringify(schema.default) == JSON.stringify(value)) {
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

class PyTorchTensor {
    constructor(tensor) {
        this._tensor = tensor;
        this._dataType = tensor.storage.dataType;
        this._shape = tensor.size;
    }

    get kind() {
        return 'Tensor';
    }

    get type() {
        return new PyTorchTensorType(this._dataType, this._shape);
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
        switch (this.dataType) {
            case 'int64':
                return OnnxTensor._stringify(value, '', '    ');
        }
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (!this._dataType) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        if (!this._shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }
        if (!this._tensor.storage || !this._tensor.storage.data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.data = this._tensor.storage.data;
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
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
                switch (this._dataType)
                {
                    case 'float32':
                        results.push(context.dataView.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.dataView.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'int64':
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
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push('[');
            var items = value.map((item) => OnnxTensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(']');
            return result.join('\n');
        }
        return indentation + value.toString();
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
            host.request(null, 'pytorch-metadata.json', 'utf-8', (err, data) => {
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
}

class PyTorchError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading PyTorch model.';
    }
}
