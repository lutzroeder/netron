/*jshint esversion: 6 */

class PyTorchModelFactory {

    match(context) {
        var extension = context.identifier.split('.').pop();
        return extension == 'pt' || extension == 'pth';
    }

    open(context, host, callback) { 
        host.import('pickle.js', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            this._openModel(context, host, callback);
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
                callback(new PyTorchError('Unsupported system information.'));
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

            var functionTable = {};
            functionTable['argparse.Namespace'] = function (args) { this.args = args; };
            functionTable['torch.nn.modules.activation.ReLU'] = function () {};
            functionTable['torch.nn.modules.batchnorm.BatchNorm2d'] = function () {};
            functionTable['torch.nn.modules.container.Sequential'] = function () {};
            functionTable['torch.nn.modules.conv.Conv2d'] = function () {};
            functionTable['torch.nn.modules.dropout.Dropout'] = function () {};
            functionTable['torch.nn.modules.linear.Linear'] = function () {};
            functionTable['torch.nn.modules.pooling.AvgPool2d'] = function () {};
            functionTable['torch.nn.modules.pooling.MaxPool2d'] = function () {};
            functionTable['torchvision.models.alexnet.AlexNet'] = function () {};
            functionTable['torchvision.models.densenet.DenseNet'] = function () {};
            functionTable['torchvision.models.densenet._DenseBlock'] = function () {};
            functionTable['torchvision.models.densenet._DenseLayer'] = function () {};
            functionTable['torchvision.models.densenet._Transition'] = function () {};
            functionTable['torchvision.models.inception.BasicConv2d'] = function () {};
            functionTable['torchvision.models.inception.Inception3'] = function () {};
            functionTable['torchvision.models.inception.InceptionAux'] = function () {};
            functionTable['torchvision.models.inception.InceptionA'] = function () {};
            functionTable['torchvision.models.inception.InceptionB'] = function () {};
            functionTable['torchvision.models.inception.InceptionC'] = function () {};
            functionTable['torchvision.models.inception.InceptionD'] = function () {};
            functionTable['torchvision.models.inception.InceptionE'] = function () {};
            functionTable['torchvision.models.resnet.Bottleneck'] = function () {};
            functionTable['torchvision.models.resnet.ResNet'] = function () {};
            functionTable['torchvision.models.vgg.VGG'] = function () {};
            functionTable['torch.nn.backends.thnn._get_thnn_function_backend'] = function () {};
            functionTable['torch.nn.parameter.Parameter'] = function(data, requires_grad) { this.data = data; this.requires_grad = requires_grad; };
            functionTable['torch.FloatStorage'] = function (size) { this.size = size; this.dataTypeSize = 4; this.dataType = 'float32'; };
            functionTable['torch.LongStorage'] = function (size) { this.size = size; this.dataTypeSize = 4; this.dataType = 'int64'; };

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

            var function_call = (name, args) => {
                if (name == 'collections.OrderedDict') {
                    if (args.length == 0) {
                        return [];
                    }
                    return args[0].map((arg) => { 
                        arg[1] = arg[1] || {};
                        arg[1].__id__ = arg[0];
                        return arg[1];
                    });
                }
                if (functionTable[name]) {
                    var obj = { __type__: name };
                    functionTable[name].apply(obj, args);

                    if (module_source_map[name]) {
                        obj.__source__ = module_source_map[name];
                    }

                    return obj;
                }
                throw new pickle.Error("Unknown function '" + type + "'.");
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

            var model = new PyTorchModel(sysInfo, root, deserialized_storage_keys); 

            PyTorchOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (error) {
            host.exception(error, false);
            callback(new PyTorchError(error.message), null);
        }
    }
}

class PyTorchModel { 

    constructor(sysInfo, root, deserialized_storage_keys) {
        this._graphs = [ new PyTorchGraph(sysInfo, root, deserialized_storage_keys) ];
    }

    get format() {
        return 'PyTorch';
    }

    get graphs() {
        return this._graphs;
    }

}

class PyTorchGraph {

    constructor(sysInfo, root, deserialized_storage_keys) {
        this._type = root.__type__;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;

        var input = 'data';
        this._inputs.push({ id: input, name: input });

        var outputs = this._loadModule(root, [], [ input ]);

        outputs.forEach((output) => {
            this._outputs.push({ id: output, name: output });
        });

    }

    _loadModule(parent, groups, inputs) {

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

class PyTorchNode {

    constructor(module, groups, connections) {
        this._group = groups.join('/');
        groups.push(module.__id__);
        this._name = groups.join('/');
        groups.pop();
        this._operator = module.__type__.split('.').pop();

        var input = { name: 'input', connections: [] };
        connections.forEach((connection) => {
            input.connections.push({ id: connection });
        });
        this._inputs = [ input ];

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
                var input = {};
                input.name = parameter.__id__;
                input.connections = [];
                this._inputs.push(input);
                var connection = {};
                if (parameter.data) {
                    connection.initializer = new PyTorchTensor(parameter.data);
                    connection.type = connection.initializer.type.toString();
                }
                else if (parameter.storage) {
                    connection.initializer = new PyTorchTensor(parameter);
                    connection.type = connection.initializer.type.toString();
                }
                input.connections.push(connection);
            }
        });

        this._outputs = [];
        this._outputs.push({ name: 'output', connections: [ { id: this._name } ] });

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
        switch (this.dataType) {
            case 'int64':
                return OnnxTensor._stringify(result.value, '', '    ');
        }
        return JSON.stringify(result.value, null, 4);
    }

    _decode(limit) {

        var result = {};

        if (!this._dataType) {
            return { error: 'Tensor has no data type.' };
        }
        if (!this._shape) {
            return { error: 'Tensor has no dimensions.' };
        }
        if (!this._tensor.storage || !this._tensor.storage.data) {
            return { error: 'Tensor data is empty.' };
        }

        var context = {};
        context.index = 0;
        context.count = 0;
        context.limit = limit;
        context.data = this._tensor.storage.data;
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);

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
                    case 'float32':
                        results.push(context.dataView.getFloat32(context.index, true));
                        context.index += 4;
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
                results.push(this._decodeDimension(context, dimension + 1));
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
