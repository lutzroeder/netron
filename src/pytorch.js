/*jshint esversion: 6 */

// Experimental

var pytorch = pytorch || {};
var base = base || require('./base');
var tar = tar || require('./tar');

pytorch.ModelFactory = class {

    match(context, host) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'pt' || extension == 'pth' || extension == 'pkl' || extension == 'h5' || extension == 'model') {
            var buffer = context.buffer;
            var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > torch.length + 2 && 
                buffer[0] == 0x80 && buffer[1] > 0x00 && buffer[1] < 0x05) {
                if (torch.every((value, index) => value == buffer[index + 2])) {
                    return true;
                }
            }
            if (this._isLegacyFormat(buffer)) {
                return true;
            }
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('./pickle', (err, pickle) => {
            if (err) {
                callback(err, null);
                return;
            }
            pytorch.OperatorMetadata.open(host, (err, metadata) => {
                this._openModel(context, host, pickle, callback);
            });        
        });
    }

    _openModel(context, host, pickle, callback) {
        try {
            var identifier = context.identifier;
            var buffer = context.buffer;
            var unpickler = new pickle.Unpickler(buffer);

            var signature = [ 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            var magic_number = unpickler.load();
            if (!Array.isArray(magic_number) ||
                signature.length != magic_number.length ||
                !signature.every((value, index) => value == magic_number[index])) 
            {
                if (this._isLegacyFormat(buffer)) {
                    callback(new pytorch.Error('PyTorch legacy tar format not supported.', null));
                    return;
                }
                callback(new pytorch.Error('Invalid signature.', null));
                return;
            }
            var protocol_version = unpickler.load();
            if (protocol_version != 1001) {
                callback(new pytorch.Error("Unsupported protocol version '" + protocol_version + "'.", null));
                return;
            }
            var sysInfo = unpickler.load();
            if (sysInfo.protocol_version != 1001) {
                callback(new pytorch.Error("Unsupported protocol version '" + sysInfo.protocol_version + "'.", null));
                return;
            }
            if (sysInfo.type_sizes) {
                if ((sysInfo.type_sizes.int && sysInfo.type_sizes.int != 4) ||
                    (sysInfo.type_sizes.long && sysInfo.type_sizes.long != 4) ||
                    (sysInfo.type_sizes.short && sysInfo.type_sizes.short != 2))
                {
                    callback(new pytorch.Error('Unsupported type sizes.'));
                    return;
                }
            }

            var constructorTable = {};
            var functionTable = {};

            constructorTable['argparse.Namespace'] = function (args) { this.args = args; };
            constructorTable['torch.nn.modules.activation.LeakyReLU'] = function () {};
            constructorTable['torch.nn.modules.activation.ReLU'] = function () {};
            constructorTable['torch.nn.modules.activation.ReLU6'] = function () {};
            constructorTable['torch.nn.modules.activation.PReLU'] = function () {};
            constructorTable['torch.nn.modules.activation.Sigmoid'] = function () {};
            constructorTable['torch.nn.modules.activation.Softmax'] = function () {};
            constructorTable['torch.nn.modules.activation.Tanh'] = function () {};
            constructorTable['torch.nn.modules.batchnorm.BatchNorm1d'] = function () {};
            constructorTable['torch.nn.modules.batchnorm.BatchNorm2d'] = function () {};
            constructorTable['torch.nn.modules.batchnorm.BatchNorm3d'] = function () {};
            constructorTable['torch.nn.modules.container.ModuleList'] = function () {};
            constructorTable['torch.nn.modules.container.Sequential'] = function () {};
            constructorTable['torch.nn.modules.conv.Conv1d'] = function () {};
            constructorTable['torch.nn.modules.conv.Conv2d'] = function () {};
            constructorTable['torch.nn.modules.conv.Conv3d'] = function () {};
            constructorTable['torch.nn.modules.conv.ConvTranspose1d'] = function () {};
            constructorTable['torch.nn.modules.conv.ConvTranspose2d'] = function () {};
            constructorTable['torch.nn.modules.conv.ConvTranspose3d'] = function () {};
            constructorTable['torch.nn.modules.dropout.Dropout'] = function () {};
            constructorTable['torch.nn.modules.dropout.Dropout2d'] = function () {};
            constructorTable['torch.nn.modules.dropout.Dropout3d'] = function () {};
            constructorTable['torch.nn.modules.instancenorm.InstanceNorm1d'] = function() {};
            constructorTable['torch.nn.modules.instancenorm.InstanceNorm2d'] = function() {};
            constructorTable['torch.nn.modules.instancenorm.InstanceNorm3d'] = function() {};
            constructorTable['torch.nn.modules.linear.Linear'] = function () {};
            constructorTable['torch.nn.modules.normalization.GroupNorm'] = function () {};
            constructorTable['torch.nn.modules.pooling.AvgPool1d'] = function () {};
            constructorTable['torch.nn.modules.pooling.AvgPool2d'] = function () {};
            constructorTable['torch.nn.modules.pooling.AvgPool3d'] = function () {};
            constructorTable['torch.nn.modules.pooling.MaxPool1d'] = function() {};
            constructorTable['torch.nn.modules.pooling.MaxPool2d'] = function () {};
            constructorTable['torch.nn.modules.pooling.MaxPool3d'] = function() {};
            constructorTable['torch.nn.modules.pooling.AdaptiveAvgPool1d'] = function() {};
            constructorTable['torch.nn.modules.pooling.AdaptiveAvgPool2d'] = function() {};
            constructorTable['torch.nn.modules.pooling.AdaptiveAvgPool3d'] = function() {};
            constructorTable['torch.nn.modules.rnn.LSTM'] = function () {};
            constructorTable['torch.nn.modules.sparse.Embedding'] = function () {};
            constructorTable['torch.nn.parallel.data_parallel.DataParallel'] = function() {}; 
            constructorTable['torchvision.models.squeezenet.Fire'] = function () {};
            constructorTable['torchvision.models.squeezenet.SqueezeNet'] = function () {};
            constructorTable['torch.nn.modules.upsampling.Upsample'] = function() {};
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
            constructorTable['torch.nn.modules.padding.ReflectionPad1d'] = function () {};
            constructorTable['torch.nn.modules.padding.ReflectionPad2d'] = function () {};
            constructorTable['torch.nn.modules.padding.ReplicationPad1d'] = function () {};
            constructorTable['torch.nn.modules.padding.ReplicationPad2d'] = function () {};
            constructorTable['torch.nn.modules.padding.ReplicationPad3d'] = function () {};
            constructorTable['torch.nn.modules.padding.ZeroPad2d'] = function () {};
            constructorTable['torch.nn.modules.padding.ConstantPad1d'] = function () {};
            constructorTable['torch.nn.modules.padding.ConstantPad2d'] = function () {};
            constructorTable['torch.nn.modules.padding.ConstantPad3d'] = function () {};
            constructorTable['torchvision.models.resnet.Bottleneck'] = function () {};
            constructorTable['torchvision.models.resnet.BasicBlock'] = function() {};
            constructorTable['torchvision.models.resnet.ResNet'] = function () {};
            constructorTable['torchvision.models.vgg.VGG'] = function () {};
            constructorTable['torch.nn.backends.thnn._get_thnn_function_backend'] = function () {};
            constructorTable['torch.nn.parameter.Parameter'] = function(data, requires_grad) { this.data = data; this.requires_grad = requires_grad; };
            constructorTable['torch.ByteStorage'] = function (size) { this.size = size; this.dataTypeSize = 1; this.dataType = 'uint8'; };
            constructorTable['torch.LongStorage'] = function (size) { this.size = size; this.dataTypeSize = 4; this.dataType = 'int64'; };
            constructorTable['torch.HalfStorage'] = function (size) { this.size = size; this.dataTypeSize = 2; this.dataType = 'float16'; };
            constructorTable['torch.FloatStorage'] = function (size) { this.size = size; this.dataTypeSize = 4; this.dataType = 'float32'; };
            constructorTable['torch.DoubleStorage'] = function (size) { this.size = size; this.dataTypeSize = 8; this.dataType = 'float64'; };
            constructorTable['torch.FloatTensor'] = function () {
                this.__setstate__ = function(state) {
                    this.storage = state[0];
                    this.storage_offset = state[1];
                    this.size = state[2];
                    this.stride = state[3];
                };
            };

            functionTable['collections.OrderedDict'] = function(args) {
                var obj = [];
                obj.__setitem__ = function(key, value) {
                    obj.push({ key: key, value: value });
                };
                if (args) {
                    args.forEach((arg) => {
                        obj.__setitem__(arg[0], arg[1]);
                    });
                }
                return obj;
            };
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
            functionTable['torch._utils._rebuild_parameter'] = function(data, requires_grad, backward_hooks) {
                var obj = {};
                obj.__type__ = 'torch.nn.parameter.Parameter';
                constructorTable[obj.__type__].apply(obj, [ data, requires_grad ]);
                obj.backward_hooks = backward_hooks;
                return obj;
            };

            var function_call = (name, args) => {
                var func = functionTable[name];
                if (func) {
                    return func.apply(null, args);
                }
                var obj = { __type__: name };
                var constructor = constructorTable[name];
                if (constructor) {
                    constructor.apply(obj, args);
                }
                else if (!name.startsWith('__main__.') && !name.startsWith('networks.') && !name.startsWith('nets.') && 
                         !name.startsWith('model.') && !name.startsWith('models.') &&
                         !name.startsWith('modeling.') && !name.startsWith('src.model.') && !name.startsWith('resnet.') &&
                         !name.startsWith('Layers.') && !name.startsWith('Sublayers.') && !name.startsWith('parts.') &&
                         !name.startsWith('Embed.') && !name.startsWith('fpn.') && !name.startsWith('retinanet.') &&
                         !name.startsWith('darknet.') && !name.startsWith('self_attn.') && !name.startsWith('base.')) {
                    debugger;
                    host.exception(new pytorch.Error("Unknown function '" + name + "' in '" + identifier + "'."), false);
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
                throw new pytorch.Error("Unknown persistent load type '" + typename + "'.");
            };

            var root = unpickler.load(function_call, persistent_load);
            var deserialized_storage_keys = unpickler.load();
            deserialized_storage_keys.forEach((key) => {
                if (deserialized_objects[key]) {
                    var storage = deserialized_objects[key];
                    storage.data = unpickler.read(storage.dataTypeSize * storage.size);
                }
            });

            if ((Array.isArray(root) && root.__setitem__ && root.every((item) => item.value.__type__.startsWith('torch.') && item.value.__type__.endsWith('Tensor'))) ||
                (root != null && root.state_dict && Array.isArray(root.state_dict))) {
                callback(new pytorch.Error("File does not contain a model graph. Use 'torch.save()' to save both the graph and tensor data."), null);
                return;
            }

            if (!root._modules) {
                callback(new pytorch.Error("Root object does not contain modules in '" + identifier + "'."), null);
                return;
            }

            var model = new pytorch.Model(sysInfo, root); 
            callback(null, model);
        }
        catch (error) {
            host.exception(error, false);
            callback(new pytorch.Error(error.message), null);
            return;
        }
    }

    _isLegacyFormat(buffer) {
        try {
            var archive = new tar.Archive(buffer);
            if (archive.entries.some((entry) => entry.name == 'pickle') &&
                archive.entries.some((entry) => entry.name == 'storages') &&
                archive.entries.some((entry) => entry.name == 'tensors')) {
                return true;
            }
        }
        catch (err) {
        }
        return false;
    }
};

pytorch.Model = class { 

    constructor(sysInfo, root) {
        this._graphs = [ new pytorch.Graph(sysInfo, root) ];
    }

    get format() {
        return 'PyTorch';
    }

    get graphs() {
        return this._graphs;
    }
};

pytorch.Graph = class {

    constructor(sysInfo, root) {
        this._type = root.__type__;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        this._littleEndian = sysInfo.little_endian;

        var input = 'data';
        this._inputs.push(new pytorch.Argument(input, true, [ new pytorch.Connection(input, null, null) ]));

        var outputs = this._loadModule(root, [], [ input ]);
        outputs.forEach((output) => {
            this._outputs.push(new pytorch.Argument(output, true, [ new pytorch.Connection(output, null, null) ]));
        });
    }

    _loadModule(parent, groups, inputs) {

        if (parent.__type__ &&
            !parent.__type__.startsWith('torch.nn.modules.container.') &&
            (!parent._modules || parent._modules.length == 0)) {
            var node = new pytorch.Node('', parent, groups, inputs, this._littleEndian);
            this._nodes.push(node);
            return [];
        }

        if (!parent._modules) {
            throw new pytorch.Error('Module does not contain modules.');
        }

        parent._modules.forEach((module) => {
            switch (module.value.__type__) {
                case 'torch.nn.modules.container.Sequential':
                    groups.push(module.key);
                    inputs = this._loadModule(module.value, groups, inputs);
                    groups.pop(module.key);
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
                    groups.push(module.key);
                    inputs = this._loadSource(module, groups, inputs);
                    groups.pop(module.key);
                    break; 
                default:
                    var node = new pytorch.Node(module.key, module.value, groups, inputs, this._littleEndian);
                    this._nodes.push(node);
                    inputs = [ node.name ];
                    break;
            }
        });

        return inputs;
    }

    _loadSource(parent, groups, inputs) {

        var node = new pytorch.Node(parent.key, parent.value, groups, inputs);
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
};

pytorch.Argument = class {
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
};

pytorch.Connection = class {
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
};

pytorch.Node = class {

    constructor(key, obj, groups, connections, littleEndian) {
        this._group = groups.join('/');
        this._name = this._group + '/' + key;
        this._operator = obj.__type__.split('.').pop();

        this._inputs = [];
        this._inputs.push(new pytorch.Argument('input', true, connections.map((connection) => {
            return new pytorch.Connection(connection, null, null);
        })));

        var initializers = [];
        if (obj._parameters) {
            obj._parameters.forEach((parameter) => {
                initializers.push(parameter);
            });
        }
        if (obj._buffers) {
            obj._buffers.forEach((buffer) => {
                initializers.push(buffer);
            });
        }

        initializers.forEach((parameter) => {
            if (parameter && parameter.value && (parameter.value.data || parameter.value.storage)) {
                var initializer = null;
                if (parameter.value.data) {
                    initializer = new pytorch.Tensor(parameter.value.data, littleEndian);
                }
                else if (parameter.value.storage) {
                    initializer = new pytorch.Tensor(parameter.value, littleEndian);
                }
                var visible = (this._operator != 'LSTM' || initializer == null);
                this._inputs.push(new pytorch.Argument(parameter.key, visible, [ new pytorch.Connection(null, null, initializer) ]));
            }
        });

        this._outputs = [];
        this._outputs.push(new pytorch.Argument('output', true, [ new pytorch.Connection(this._name, null, null) ]));

        this._attributes = [];
        Object.keys(obj).forEach((key) => {
            if (!key.startsWith('_')) {
                this._attributes.push(new pytorch.Attribute(this, key, obj[key]));
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
        var schema = pytorch.OperatorMetadata.operatorMetadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : null;
    }

    get documentation() {
        var schema = pytorch.OperatorMetadata.operatorMetadata.getSchema(this._operator);
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
        return null;
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
};

pytorch.Attribute = class {

    constructor(node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;

        var schema = pytorch.OperatorMetadata.operatorMetadata.getAttributeSchema(this._node.operator, this._name);
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
        return (this._visible == false || this.name == 'training') ? false : true;
    }
};

pytorch.Tensor = class {
    constructor(tensor, littleEndian) {
        this._tensor = tensor;
        this._type = new pytorch.TensorType(tensor.storage.dataType, new pytorch.TensorShape(tensor.size));
        this._littleEndian = littleEndian;
    }

    get kind() {
        return 'Tensor';
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
        switch (this.dataType) {
            case 'int64':
                return pytorch.Tensor._stringify(value, '', '    ');
        }
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (!this._type.dataType) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }
        if (!this._tensor.storage || !this._tensor.storage.data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.data = this._tensor.storage.data;
        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
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
                switch (context.dataType)
                {
                    case 'uint8':
                        results.push(context.dataView.getUint8(context.index, this._littleEndian));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.dataView.getFloat16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.dataView.getFloat32(context.index, this._littleEndian));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.dataView.getFloat64(context.index, this._littleEndian));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'int64':
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

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push('[');
            var items = value.map((item) => pytorch.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(']');
            return result.join('\n');
        }
        return indentation + value.toString();
    }
};

pytorch.TensorType = class {

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
        return this._dataType + this._shape.toString();
    }
};

pytorch.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

pytorch.OperatorMetadata = class {

    static open(host, callback) {
        if (pytorch.OperatorMetadata.operatorMetadata) {
            callback(null, pytorch.OperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'pytorch-metadata.json', 'utf-8', (err, data) => {
                pytorch.OperatorMetadata.operatorMetadata = new pytorch.OperatorMetadata(data);
                callback(null, pytorch.OperatorMetadata.operatorMetadata);
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
};

pytorch.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading PyTorch model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = pytorch.ModelFactory;
}
