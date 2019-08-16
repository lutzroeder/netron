/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var pytorch = pytorch || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var tar = tar || require('./tar');
var marked = marked || require('marked');
var zip = zip || require('./zip');

pytorch.ModelFactory = class {

    match(context) {
        var identifier = context.identifier; 
        var extension = identifier.split('.').pop().toLowerCase();
        var buffer = null;
        if (extension === 'pth' || extension === 'pkl' || extension === 'pt' || extension === 'bin' ||
            extension === 'h5' || extension === 't7' || extension === 'dms' || extension === 'model' ||
            extension === 'ckpt' || identifier.endsWith('.pth.tar')) {
            buffer = context.buffer;
            var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                return true;
            }
            if (pytorch.ModelFactory._loadLegacyFormat(buffer)) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./pickle').then((pickle) => {
            var identifier = context.identifier;
            var buffer = context.buffer;
            try {
                var unpickler = new pickle.Unpickler(buffer);

                var signature = [ 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
                var magic_number = unpickler.load();

                var module_source_map = {};
                var deserialized_objects = {};
                var sys_info = null;
                var root_module = null;
                var state_dict = null;
                var storage = null;

                var constructorTable = {};
                var functionTable = {};

                constructorTable['argparse.Namespace'] = function (args) {
                    this.args = args;
                };
                constructorTable['torch.autograd.variable.Variable'] = function() {};
                constructorTable['torch.backends.cudnn.rnn.Unserializable'] = function() {};
                constructorTable['torch.nn.backends.thnn._get_thnn_function_backend'] = function() {};
                constructorTable['torch.nn.modules.activation.ELU'] = function() {};
                constructorTable['torch.nn.modules.activation.GLU'] = function() {};
                constructorTable['torch.nn.modules.activation.Hardtanh'] = function() {};
                constructorTable['torch.nn.modules.activation.LeakyReLU'] = function() {};
                constructorTable['torch.nn.modules.activation.LogSigmoid'] = function() {};
                constructorTable['torch.nn.modules.activation.LogSoftmax'] = function() {};
                constructorTable['torch.nn.modules.activation.ReLU'] = function() {};
                constructorTable['torch.nn.modules.activation.ReLU6'] = function() {};
                constructorTable['torch.nn.modules.activation.PReLU'] = function() {};
                constructorTable['torch.nn.modules.activation.SELU'] = function() {};
                constructorTable['torch.nn.modules.activation.Sigmoid'] = function() {};
                constructorTable['torch.nn.modules.activation.Softmax'] = function() {};
                constructorTable['torch.nn.modules.activation.Softmax2d'] = function() {};
                constructorTable['torch.nn.modules.activation.Softplus'] = function() {};
                constructorTable['torch.nn.modules.activation.Tanh'] = function() {};
                constructorTable['torch.nn.modules.activation.Threshold'] = function() {};
                constructorTable['torch.nn.modules.batchnorm.BatchNorm1d'] = function() {};
                constructorTable['torch.nn.modules.batchnorm.BatchNorm2d'] = function() {};
                constructorTable['torch.nn.modules.batchnorm.BatchNorm3d'] = function() {};
                constructorTable['torch.nn.modules.batchnorm.SyncBatchNorm'] = function() {};
                constructorTable['torch.nn.modules.container.ModuleDict'] = function() {};
                constructorTable['torch.nn.modules.container.ModuleList'] = function() {};
                constructorTable['torch.nn.modules.container.ParameterList'] = function() {};
                constructorTable['torch.nn.modules.container.Sequential'] = function() {};
                constructorTable['torch.nn.modules.conv.Conv1d'] = function() {};
                constructorTable['torch.nn.modules.conv.Conv2d'] = function() {};
                constructorTable['torch.nn.modules.conv.Conv3d'] = function() {};
                constructorTable['torch.nn.modules.conv.ConvTranspose1d'] = function() {};
                constructorTable['torch.nn.modules.conv.ConvTranspose2d'] = function() {};
                constructorTable['torch.nn.modules.conv.ConvTranspose3d'] = function() {};
                constructorTable['torch.nn.modules.distance.CosineSimilarity'] = function() {};
                constructorTable['torch.nn.modules.dropout.Dropout'] = function() {};
                constructorTable['torch.nn.modules.dropout.Dropout2d'] = function() {};
                constructorTable['torch.nn.modules.dropout.Dropout3d'] = function() {};
                constructorTable['torch.nn.modules.instancenorm.InstanceNorm1d'] = function() {};
                constructorTable['torch.nn.modules.instancenorm.InstanceNorm2d'] = function() {};
                constructorTable['torch.nn.modules.instancenorm.InstanceNorm3d'] = function() {};
                constructorTable['torch.nn.modules.linear.Linear'] = function() {};
                constructorTable['torch.nn.modules.loss.BCELoss'] = function() {};
                constructorTable['torch.nn.modules.loss.BCEWithLogitsLoss'] = function() {}; 
                constructorTable['torch.nn.modules.loss.CrossEntropyLoss'] = function() {};
                constructorTable['torch.nn.modules.loss.L1Loss'] = function() {};
                constructorTable['torch.nn.modules.loss.MSELoss'] = function() {};
                constructorTable['torch.nn.modules.loss.NLLLoss'] = function() {};
                constructorTable['torch.nn.modules.normalization.CrossMapLRN2d'] = function() {};
                constructorTable['torch.nn.modules.normalization.GroupNorm'] = function() {};
                constructorTable['torch.nn.modules.normalization.LayerNorm'] = function() {};
                constructorTable['torch.nn.modules.normalization.LocalResponseNorm'] = function() {};
                constructorTable['torch.nn.modules.padding.ReflectionPad1d'] = function() {};
                constructorTable['torch.nn.modules.padding.ReflectionPad2d'] = function() {};
                constructorTable['torch.nn.modules.padding.ReplicationPad1d'] = function() {};
                constructorTable['torch.nn.modules.padding.ReplicationPad2d'] = function() {};
                constructorTable['torch.nn.modules.padding.ReplicationPad3d'] = function() {};
                constructorTable['torch.nn.modules.padding.ZeroPad2d'] = function() {};
                constructorTable['torch.nn.modules.padding.ConstantPad1d'] = function() {};
                constructorTable['torch.nn.modules.padding.ConstantPad2d'] = function() {};
                constructorTable['torch.nn.modules.padding.ConstantPad3d'] = function() {};
                constructorTable['torch.nn.modules.pixelshuffle.PixelShuffle'] = function() {};
                constructorTable['torch.nn.modules.pooling.AdaptiveAvgPool1d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AdaptiveAvgPool2d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AdaptiveAvgPool3d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AdaptiveMaxPool1d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AdaptiveMaxPool2d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AdaptiveMaxPool3d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AvgPool1d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AvgPool2d'] = function() {};
                constructorTable['torch.nn.modules.pooling.AvgPool3d'] = function() {};
                constructorTable['torch.nn.modules.pooling.FractionalMaxPool2d'] = function() {};
                constructorTable['torch.nn.modules.pooling.MaxPool1d'] = function() {};
                constructorTable['torch.nn.modules.pooling.MaxPool2d'] = function() {};
                constructorTable['torch.nn.modules.pooling.MaxPool3d'] = function() {};
                constructorTable['torch.nn.modules.pooling.MaxUnpool1d'] = function() {};
                constructorTable['torch.nn.modules.pooling.MaxUnpool2d'] = function() {};
                constructorTable['torch.nn.modules.pooling.MaxUnpool3d'] = function() {};
                constructorTable['torch.nn.modules.rnn.GRU'] = function() {};
                constructorTable['torch.nn.modules.rnn.GRUCell'] = function() {};
                constructorTable['torch.nn.modules.rnn.LSTM'] = function() {};
                constructorTable['torch.nn.modules.rnn.LSTMCell'] = function() {};
                constructorTable['torch.nn.modules.rnn.RNN'] = function() {};
                constructorTable['torch.nn.modules.sparse.Embedding'] = function() {};
                constructorTable['torch.nn.modules.upsampling.Upsample'] = function() {};
                constructorTable['torch.nn.modules.upsampling.UpsamplingBilinear2d'] = function() {};
                constructorTable['torch.nn.modules.upsampling.UpsamplingNearest2d'] = function() {};
                constructorTable['torch.nn.parallel.data_parallel.DataParallel'] = function() {}; 
                constructorTable['torch.nn.parallel.distributed.DistributedDataParallel'] = function() {};
                constructorTable['torch.nn.parameter.Parameter'] = function(data, requires_grad) {
                    this.data = data;
                    this.requires_grad = requires_grad;
                };
                constructorTable['torch.nn.utils.spectral_norm.SpectralNorm'] = function() {};
                constructorTable['torch.nn.utils.spectral_norm.SpectralNormStateDictHook'] = function() {};
                constructorTable['torch.nn.utils.spectral_norm.SpectralNormLoadStateDictPreHook'] = function() {};
                constructorTable['torch.nn.utils.weight_norm.WeightNorm'] = function() {};
                constructorTable['torch.optim.adam.Adam'] = function() {};
                constructorTable['torch.optim.adagrad.Adagrad'] = function() {};
                constructorTable['torch.optim.lr_scheduler.MultiStepLR'] = function() {};
                constructorTable['torch.optim.lr_scheduler.StepLR'] = function() {};
                constructorTable['torch.optim.rmsprop.RMSprop'] = function() {};
                constructorTable['torch.optim.sgd.SGD'] = function() {};
                constructorTable['torchvision.datasets.folder.ImageFolder'] = function() {};
                constructorTable['torchvision.models.alexnet.AlexNet'] = function() {};
                constructorTable['torchvision.models.densenet.DenseNet'] = function() {};
                constructorTable['torchvision.models.densenet._DenseBlock'] = function() {};
                constructorTable['torchvision.models.densenet._DenseLayer'] = function() {};
                constructorTable['torchvision.models.densenet._Transition'] = function() {};
                constructorTable['torchvision.models.inception.BasicConv2d'] = function() {};
                constructorTable['torchvision.models.inception.Inception3'] = function() {};
                constructorTable['torchvision.models.inception.InceptionAux'] = function() {};
                constructorTable['torchvision.models.inception.InceptionA'] = function() {};
                constructorTable['torchvision.models.inception.InceptionB'] = function() {};
                constructorTable['torchvision.models.inception.InceptionC'] = function() {};
                constructorTable['torchvision.models.inception.InceptionD'] = function() {};
                constructorTable['torchvision.models.inception.InceptionE'] = function() {};
                constructorTable['torchvision.models.mobilenet.ConvBNReLU'] = function() {};
                constructorTable['torchvision.models.mobilenet.MobileNetV2'] = function() {};
                constructorTable['torchvision.models.mobilenet.InvertedResidual'] = function() {};
                constructorTable['torchvision.models.resnet.Bottleneck'] = function() {};
                constructorTable['torchvision.models.resnet.BasicBlock'] = function() {};
                constructorTable['torchvision.models.segmentation.deeplabv3.ASPP'] = function() {};
                constructorTable['torchvision.models.segmentation.deeplabv3.ASPPConv'] = function() {}; 
                constructorTable['torchvision.models.segmentation.deeplabv3.ASPPPooling'] = function() {};
                constructorTable['torchvision.models.segmentation.deeplabv3.DeepLabHead'] = function() {};
                constructorTable['torchvision.models.segmentation.deeplabv3.DeepLabV3'] = function() {};
                constructorTable['torchvision.models.squeezenet.Fire'] = function() {};
                constructorTable['torchvision.models.squeezenet.SqueezeNet'] = function() {};
                constructorTable['torchvision.models.resnet.ResNet'] = function() {};
                constructorTable['torchvision.models.vgg.VGG'] = function() {};
                constructorTable['torchvision.models._utils.IntermediateLayerGetter'] = function() {};
                constructorTable['torchvision.transforms.transforms.Compose'] = function() {};
                constructorTable['torchvision.transforms.transforms.Normalize'] = function() {};
                constructorTable['torchvision.transforms.transforms.Resize'] = function() {};
                constructorTable['torchvision.transforms.transforms.ToTensor'] = function() {};

                constructorTable['torch.ByteStorage'] = function (size) { 
                    this.size = size; this.dataTypeSize = 1; this.dataType = 'uint8'; 
                };
                constructorTable['torch.CharStorage'] = function (size) { 
                    this.size = size; this.dataTypeSize = 1; this.dataType = 'int8'; 
                };
                constructorTable['torch.ShortStorage'] = function (size) { 
                    this.size = size; this.dataTypeSize = 2; this.dataType = 'int16';
                };
                constructorTable['torch.IntStorage'] = function (size) { 
                    this.size = size; this.dataTypeSize = 4; this.dataType = 'int32';
                };
                constructorTable['torch.LongStorage'] = function (size) { 
                    this.size = size; this.dataTypeSize = 8; this.dataType = 'int64';
                };
                constructorTable['torch.HalfStorage'] = function (size) {
                    this.size = size; this.dataTypeSize = 2; this.dataType = 'float16';
                };
                constructorTable['torch.FloatStorage'] = function (size) {
                    this.size = size; this.dataTypeSize = 4; this.dataType = 'float32';
                };
                constructorTable['torch.DoubleStorage'] = function (size) { 
                    this.size = size; this.dataTypeSize = 8; this.dataType = 'float64';
                };
                constructorTable['torch.FloatTensor'] = function () {
                    this.__setstate__ = function(state) {
                        this.storage = state[0];
                        this.storage_offset = state[1];
                        this.size = state[2];
                        this.stride = state[3];
                    };
                };
                constructorTable['torch.DoubleTensor'] = function () {
                    this.__setstate__ = function(state) {
                        this.storage = state[0];
                        this.storage_offset = state[1];
                        this.size = state[2];
                        this.stride = state[3];
                    };
                };
                constructorTable['numpy.dtype'] = function(obj, align, copy) { 
                    switch (obj) {
                        case 'i1': this.name = 'int8'; this.itemsize = 1; break;
                        case 'i2': this.name = 'int16'; this.itemsize = 2; break;
                        case 'i4': this.name = 'int32'; this.itemsize = 4; break;
                        case 'i8': this.name = 'int64'; this.itemsize = 8; break;
                        case 'u1': this.name = 'uint8'; this.itemsize = 1; break;
                        case 'u2': this.name = 'uint16'; this.itemsize = 2; break;
                        case 'u4': this.name = 'uint32'; this.itemsize = 4; break;
                        case 'u8': this.name = 'uint64'; this.itemsize = 8; break;
                        case 'f4': this.name = 'float32'; this.itemsize = 4; break;
                        case 'f8': this.name = 'float64'; this.itemsize = 8; break;
                        default:
                            if (obj.startsWith('V')) {
                                this.itemsize = Number(obj.substring(1));
                                this.name = 'void' + (this.itemsize * 8).toString();
                            }
                            else if (obj.startsWith('O')) {
                                this.itemsize = Number(obj.substring(1));
                                this.name = 'object';
                            }
                            else if (obj.startsWith('S')) {
                                this.itemsize = Number(obj.substring(1));
                                this.name = 'string';
                            }
                            else if (obj.startsWith('U')) {
                                this.itemsize = Number(obj.substring(1));
                                this.name = 'string';
                            }
                            else if (obj.startsWith('M')) {
                                this.itemsize = Number(obj.substring(1));
                                this.name = 'datetime';
                            }
                            else {
                                throw new pytorch.Error("Unknown dtype '" + obj.toString() + "'.");
                            }
                            break;
                    }
                    this.align = align;
                    this.copy = copy;
                    this.__setstate__ = function(state) {
                        switch (state.length) {
                            case 8:
                                this.version = state[0];
                                this.byteorder = state[1];
                                this.subarray = state[2];
                                this.names = state[3];
                                this.fields = state[4];
                                this.elsize = state[5];
                                this.alignment = state[6];
                                this.int_dtypeflags = state[7];
                                break;
                            default:
                                throw new pytorch.Error("Unknown numpy.dtype setstate length '" + state.length.toString() + "'.");
                        }
                    };
                };
                constructorTable['numpy.core.multiarray._reconstruct'] = function(subtype, shape, dtype) {
                    this.subtype = subtype;
                    this.shape = shape;
                    this.dtype = dtype;
                    this.__setstate__ = function(state) {
                        this.version = state[0];
                        this.shape = state[1];
                        this.typecode = state[2];
                        this.is_f_order = state[3];
                        this.rawdata = state[4];
                    };
                    this.__read__ = function(unpickler) {
                        var array = {};
                        array.__type__ = this.subtype;
                        array.dtype = this.typecode;
                        array.shape = this.shape;
                        var size = array.dtype.itemsize;
                        for (var i = 0; i < array.shape.length; i++) {
                            size = size * array.shape[i];
                        }
                        if (typeof this.rawdata == 'string') {
                            array.data = unpickler.unescape(this.rawdata, size);
                            if (array.data.length != size) {
                                throw new pytorch.Error('Invalid string array data size.');
                            }
                        }
                        else {
                            array.data = this.rawdata;
                            if (array.data.length != size) {
                                throw new pytorch.Error('Invalid array data size.');
                            }
                        }
                        return array;
                    };
                };

                functionTable['collections.Counter'] = function(/* iterable */) {
                    return {};
                };
                functionTable['collections.OrderedDict'] = function(args) {
                    var obj = [];
                    obj.__setitem__ = function(key, value) {
                        obj.push({ key: key, value: value });
                    };
                    if (args) {
                        for (var arg of args) {
                            obj.__setitem__(arg[0], arg[1]);
                        }
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
                functionTable['numpy.core.multiarray.scalar'] = function(dtype, rawData) {
                    var data = rawData;
                    if (rawData.constructor !== Uint8Array) {
                        data = new Uint8Array(rawData.length);
                        for (var i = 0; i < rawData.length; i++) {
                            data[i] = rawData.charCodeAt(i);
                        }
                    }
                    var dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    switch (dtype.name) {
                        case 'float32':
                            return dataView.getFloat32(0, true);
                        case 'float64':
                            return dataView.getFloat64(0, true);
                        case 'int8':
                            return dataView.getInt8(0, true);
                        case 'int16':
                            return dataView.getInt16(0, true);
                        case 'int32':
                            return dataView.getInt32(0, true);
                        case 'int64':
                            return new long.Long(dataView.getInt32(0, true), dataView.getInt32(4, true), false);
                    }
                    throw new pytorch.Error("Unknown scalar type '" + dtype.name + "'.");
                };
                functionTable['_codecs.encode'] = function(obj /*, econding */) {
                    return obj;
                };
                functionTable['collections.defaultdict'] = function(/* default_factory */) {
                    return {};
                };

                var unknownNameMap = new Set();
                var knownPackageMap = new Set([ 'torch', 'torchvision', 'collections', '__builtin__', '_codecs', 'argparse', 'numpy' ]);

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
                    else if (name && unknownNameMap.has(name)) {
                        unknownNameMap.add(name);
                        if (knownPackageMap.has(name.split('.').shift())) {
                            host.exception(new pytorch.Error("Unknown function '" + name + "' in '" + identifier + "'."), false);
                        }
                    }
                    return obj;
                };

                var persistent_load = (saved_id) => {
                    var typename = saved_id.shift();
                    var data = saved_id;
                    switch (typename) {
                        case 'module':
                            var module = data[0];
                            var source = data[2];
                            module_source_map[module] = source;
                            return data[0];
                        case 'storage':
                            var data_type = data.shift();
                            var root_key = data.shift();
                            data.shift(); // location
                            var size = data.shift();
                            var view_metadata = data.shift();
                            var storage = deserialized_objects[root_key];
                            if (!storage) {
                                storage = function_call(data_type, [ size ]);
                                deserialized_objects[root_key] = storage;
                            }
                            if (view_metadata) {
                                var view_key = view_metadata.shift();
                                view_metadata.shift(); // view_offset
                                view_metadata.shift(); // view_size
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

                if (Array.isArray(magic_number) && signature.length == magic_number.length && signature.every((value, index) => value == magic_number[index])) 
                {
                    var protocol_version = unpickler.load();
                    if (protocol_version != 1001) {
                        throw new pytorch.Error("Unsupported protocol version '" + protocol_version + "'.");
                    }
                    sys_info = unpickler.load();
                    if (sys_info.protocol_version != 1001) {
                        throw new pytorch.Error("Unsupported protocol version '" + sys_info.protocol_version + "'.");
                    }

                    var root = unpickler.load(function_call, persistent_load);
                    if (!root) {
                        throw new pytorch.Error("File format is not PyTorch.");
                    }

                    var deserialized_storage_keys = unpickler.load();
                    for (var deserialized_storage_key of deserialized_storage_keys) {
                        storage = deserialized_objects[deserialized_storage_key];
                        let size = long.Long.fromBytesLE(unpickler.read(8), false).toNumber();
                        if (size != storage.size) {
                            throw new pytorch.Error("Storage size mismatch.")
                        }
                        storage.data = unpickler.read(storage.dataTypeSize * storage.size);
                    }
        
                    root_module = pytorch.ModelFactory._findRootModule(root);
                    if (!root_module) {
                        state_dict = pytorch.ModelFactory._findStateDict(root);
                        if (!state_dict) {
                            throw new pytorch.Error('File does not contain root module or state dictionary.');
                        }
                    }
                }
                else {
                    var legacyRoot = pytorch.ModelFactory._loadLegacyFormat(buffer);
                    if (!legacyRoot) {
                        throw new pytorch.Error('Invalid signature.');
                    } 

                    unpickler = new pickle.Unpickler(legacyRoot.sys_info);
                    sys_info = unpickler.load();
                    if (sys_info.protocol_version != 1000) {
                        throw new pytorch.Error("Unsupported protocol version '" + sys_info.protocol_version + "'.");
                    }

                    unpickler = new pickle.Unpickler(legacyRoot.storages);
                    var num_storages = unpickler.load();
                    for (var i = 0; i < num_storages; i++) {
                        var storage_args = unpickler.load();
                        var storage_key = storage_args[0];
                        var storage_type = storage_args[2];
                        var size = long.Long.fromBytesLE(unpickler.read(8), false).toNumber();
                        storage = function_call(storage_type, [ size ]);
                        storage.data = unpickler.read(storage.dataTypeSize * storage.size);
                        deserialized_objects[storage_key] = storage;
                    }
                    /*
                    var storage_views = unpickler.load();
                    for target_cdata, root_cdata, offset, size in storage_views:
                        root = deserialized_objects[root_cdata]
                        deserialized_objects[target_cdata] = root[offset:offset + size]
                    */

                    unpickler = new pickle.Unpickler(legacyRoot.tensors);
                    var num_tensors = unpickler.load();
                    for (var j = 0; j < num_tensors; j++) {
                        var tensor_args = unpickler.load();
                        var tensor_key = tensor_args[0];
                        var storage_id = tensor_args[1];
                        storage = deserialized_objects[storage_id];
                        var ndim = long.Long.fromBytesLE(unpickler.read(4), false).toNumber();
                        unpickler.read(4);
                        var shape = [];
                        for (var k = 0; k < ndim; k++) {
                            shape.push(long.Long.fromBytesLE(unpickler.read(8), false).toNumber())
                        }
                        var stride = [];
                        for (var l = 0; l < ndim; l++) {
                            stride.push(long.Long.fromBytesLE(unpickler.read(8), false).toNumber())
                        }
                        var storage_offset = long.Long.fromBytesLE(unpickler.read(8), false).toNumber();
                        var tensor_type = storage.__type__.replace('Storage', 'Tensor');
                        var tensor = function_call(tensor_type, []);
                        tensor.__setstate__([ storage, storage_offset, shape, stride ]);
                        deserialized_objects[tensor_key] = tensor;
                    }

                    unpickler = new pickle.Unpickler(legacyRoot.pickle);
                    var obj = unpickler.load(function_call, (saved_id) => {
                        return deserialized_objects[saved_id];
                    });

                    root_module = null;
                    state_dict = pytorch.ModelFactory._convertStateDictLegacyFormat(obj);
                }

                if (sys_info.type_sizes &&
                    ((sys_info.type_sizes.int && sys_info.type_sizes.int != 4) ||
                    (sys_info.type_sizes.long && sys_info.type_sizes.long != 4) ||
                    (sys_info.type_sizes.short && sys_info.type_sizes.short != 2)))
                {
                    throw new pytorch.Error('Unsupported type sizes.');
                }
            }
            catch (error) {
                host.exception(error, false);
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new pytorch.Error(message + " in '" + identifier + "'.");
            }

            return pytorch.Metadata.open(host).then((metadata) => {
                try {
                    return new pytorch.Model(metadata, sys_info, root_module, state_dict, module_source_map);
                }
                catch (error) {
                    host.exception(error, false);
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new pytorch.Error(message + " in '" + identifier + "'.");
                }
            });
        });
    }

    static _loadLegacyFormat(buffer) {
        try {
            if (buffer.length >= 512) {
                var sum = 0;
                for (var i = 0; i < 512; i++) {
                    sum += (i >= 148 && i < 156) ? 32 : buffer[i];
                }
                var checksum = '';
                for (var j = 148; j < 156 && buffer[j] != 0; j++) {
                    checksum += String.fromCharCode(buffer[j]);
                }
                checksum = parseInt(checksum, 8);
                if (!isNaN(checksum) && sum == checksum) {
                    var archive = new tar.Archive(buffer);
                    var entries = {};
                    for (var entry of archive.entries) {
                        entries[entry.name] = entry.data;
                    }
                    if (entries.sys_info && entries.pickle && entries.storages && entries.tensors) {
                        return entries;
                    }
                }
            }
        }
        catch (error) {
            // continue regardless of error
        }
        return null;
    }

    static _findRootModule(root) {
        var candidates = [ root, root.model, root.net ];
        for (var obj of candidates) {
            if (obj && obj._modules) {
                return obj;
            }
        }
        return null;
    }

    static _findStateDict(root) {
        if (root) {
            if (root.encoder && Array.isArray(root.encoder) && 
                root.decoder && Array.isArray(root.decoder) && !root.state_dict) {
                root = root.encoder.concat(root.decoder);
            }
            if (Array.isArray(root) && root.every((item) => item.key && item.value)) {
                var obj = {};
                for (var item of root) {
                    obj[item.key] = item.value;
                }
                root = obj;
            }
        }
        var candidates = [ 
            root, root.state_dict, root.state, 
            root.model_state, root.model, root.model_state_dict,
            root.params, root.generator, root.discriminator, root.g_state,
            root.network, root.net, root.netG,
            root.state_dict_stylepredictor, root.state_dict_ghiasi
        ];
        for (var dict of candidates) {
            let state_dict =
                pytorch.ModelFactory._convertStateDictList(dict) ||
                pytorch.ModelFactory._convertStateDictMap(dict) || 
                pytorch.ModelFactory._convertStateDictGroupMap(dict);
            if (state_dict) {
                return state_dict;
            }
        }
        return null;
    }

    static _convertStateDictList(list) {
        if (!list || !Array.isArray(list) || 
            !list.every((item) => item && item.key && pytorch.ModelFactory._isTensor(item.value))) {
            return null;
        }
        let state_dict = [];
        let state_map = {};
        for (let item of list) {
            let split = item.key.split('.');
            if (split.length < 2) {
                return null;
            }
            let state = {};
            state.id = item.key;
            state.name = split.pop();
            state.value = item.value;
            let state_group_name = split.join('.');
            let state_group = state_map[state_group_name];
            if (!state_group) {
                state_group = {};
                state_group.name = state_group_name;
                state_group.states = [];
                state_map[state_group_name] = state_group;
                state_dict.push(state_group);
            }
            state_group.states.push(state);
        }
        return state_dict;
    }

    static _convertStateDictMap(obj) {
        if (!obj || Array.isArray(obj)) {
            return null
        }
        let state_dict = [];
        let state_map = {};
        for (var key in obj) {
            let split = key.split('.');
            if (split.length < 1) {
                return null;
            }
            let state = {};
            state.id = key;
            state.name = split.pop();
            state.value = obj[key];
            if (!pytorch.ModelFactory._isTensor(state.value)) {
                return null;
            }
            let state_group_name = split.join('.');
            let state_group = state_map[state_group_name];
            if (!state_group) {
                state_group = {};
                state_group.name = state_group_name;
                state_group.states = [];
                state_map[state_group_name] = state_group;
                state_dict.push(state_group);
            }
            state_group.states.push(state);
        }
        return state_dict;
    }

    static _convertStateDictGroupMap(obj) {
        if (!obj || Array.isArray(obj)) {
            return null;
        }
        let state_dict = [];
        let state_map = {};
        for (let state_group_name in obj) {

            let state_group = state_map[state_group_name];
            if (!state_group) {
                state_group = {};
                state_group.name = state_group_name;
                state_group.states = [];
                state_group.attributes = [];
                state_map[state_group_name] = state_group;
                state_dict.push(state_group);
            }
            var item = obj[state_group_name];
            if (!item) {
                return null;
            } 
            if (Array.isArray(item)) {
                for (let entry of item) {
                    if (!entry || !entry.key || !entry.value || !pytorch.ModelFactory._isTensor(entry.value)) {
                        return null;
                    }
                    let state = {};
                    state.id = state_group_name + '.' + entry.key;
                    state.name = entry.key;
                    state.value = entry.value;
                    state_group.states.push(state);
                }
            }
            else {
                for (let key in item) {
                    let value = item[key];
                    if (pytorch.ModelFactory._isTensor(value)) {
                        state_group.states.push({ name: key, value: value, id: state_group_name + '.' + key });
                    }
                    else if (value !== Object(value)) {
                        state_group.attributes.push({ name: key, value: value });
                    }
                    else if (value && value.__type__ == 'torch.nn.parameter.Parameter' && value.data) {
                        state_group.states.push({ name: key, value: value.data, id: state_group_name + '.' + key });
                    }
                    else {
                        return null;
                    }
                }
            }
        }
        return state_dict;
    }


    static _convertStateDictLegacyFormat(obj) {
        if (!obj) {
            return null;
        }
        if (obj && !Array.isArray(obj)) {
            var array = [];
            for (var key of Object.keys(obj)) {
                array.push({ key: key, value: obj[key] });
            }
            obj = array;
        }
        var state_dict = [];
        var state_map = {};
        if (obj && Array.isArray(obj)) {
            for (var item of obj) {
                if (!item || !item.key || !item.value) {
                    return null;
                }
                let state = {};
                state.id = item.key;
                state.value = null;
                if (item.value.__type__ == 'torch.nn.parameter.Parameter') {
                    state.value = item.value[0];
                }
                else if (pytorch.ModelFactory._isTensor(item.value)) {
                    state.value = item.value;
                }
                if (!state.value) {
                    return null;
                }
                let split = state.id.split('.');
                if (split.length < 2) {
                    return null;
                }
                state.name = split.pop();
                let state_group_name = split.join('.');
                let state_group = state_map[state_group_name];
                if (!state_group) {
                    state_group = {};
                    state_group.name = state_group_name;
                    state_group.states = [];
                    state_map[state_group_name] = state_group;
                    state_dict.push(state_group);
                }
                state_group.states.push(state);
            }
        }
        return state_dict;
    }

    static _isTensor(obj) {
        return obj && obj.__type__ && obj.__type__.startsWith('torch.') && obj.__type__.endsWith('Tensor');
    }
};

pytorch.Model = class { 

    constructor(metadata, sysInfo, root, state_dict, module_source_map) {
        this._graphs = [];
        this._graphs.push(new pytorch.Graph(metadata, sysInfo, root, state_dict, module_source_map));
    }

    get format() {
        return 'PyTorch';
    }

    get graphs() {
        return this._graphs;
    }
};

pytorch.Graph = class {

    constructor(metadata, sysInfo, root, state_dict, module_source_map) {
        this._metadata = metadata;
        this._type = root ? root.__type__ : '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        this._littleEndian = sysInfo.little_endian;

        if (!state_dict) {
            var input = 'data';
            this._inputs.push(new pytorch.Parameter(input, true, [ new pytorch.Argument(input, null, null) ]));
            var outputs = this._loadModule(root, module_source_map, [], [ input ]);
            for (var output of outputs) {
                this._outputs.push(new pytorch.Parameter(output, true, [ new pytorch.Argument(output, null, null) ]));
            }
        }
        else {
            for (let state_group of state_dict) {
                let type = 'torch.nn.modules._.Module';
                let attributes = state_group.attributes || [];
                let inputs = state_group.states.map((state) => {
                    var tensor = new pytorch.Tensor(state.id, state.value, sysInfo.little_endian);
                    var visible = state_group.states.length == 0 || tensor.type.toString() != 'int64' || tensor.value < 1000;
                    return new pytorch.Parameter(state.name, visible, [
                        new pytorch.Argument(state.id, null, tensor)
                    ]);
                });
                this._nodes.push(new pytorch.Node(this._metadata, '', state_group.name, type, attributes, inputs, []));
            }
        }
    }

    _loadModule(parent, module_source_map, groups, inputs) {

        if (parent.__type__ &&
            !parent.__type__.startsWith('torch.nn.modules.container.') &&
            (!parent._modules || parent._modules.length == 0)) {
            this._createNode(groups, '', parent, inputs);
            return [];
        }

        if (!parent._modules) {
            throw new pytorch.Error('Module does not contain modules.');
        }

        for (var module of parent._modules) {
            var node;
            switch (module.value.__type__) {
                case 'torch.nn.modules.container.Sequential':
                    groups.push(module.key);
                    inputs = this._loadModule(module.value, module_source_map, groups, inputs);
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
                    node = this._createNode(groups, module.key, module.value, inputs, this._littleEndian);
                    inputs = [ node.name ];
                    groups.pop(module.key);
                    break; 
                default:
                    node = this._createNode(groups, module.key, module.value, inputs);
                    inputs = [ node.name ];
                    break;
            }
        }

        return inputs;
    }

    _createNode(groups, key, obj, args) {

        var operator = obj.__type__.split('.').pop();
        var schema = this._metadata.getSchema(operator);

        var inputSchema = [ { name: 'input'} ];
        if (schema && schema.inputs && schema.inputs.length > 0) {
            inputSchema = schema.inputs.slice();
        }

        var inputs = [];
        inputs.push(new pytorch.Parameter(inputSchema.shift().name, true, args.map((argument) => {
            return new pytorch.Argument(argument, null, null);
        })));

        var parameters = [];
        if (obj._parameters) {
            parameters = parameters.concat(obj._parameters);
        }
        if (obj._buffers) {
            parameters = parameters.concat(obj._buffers);
        }

        for (var parameter of parameters) {
            var visible = true;
            var inputName = ''; 
            if (inputSchema.length > 0) {
                var input = inputSchema.shift();
                inputName = input.name;
                visible = input.visible === false ? false : true;
            }
            if (parameter && parameter.value && (parameter.value.data || parameter.value.storage)) {
                var initializer = null;
                if (parameter.value.data) {
                    initializer = new pytorch.Tensor('', parameter.value.data, this._littleEndian);
                }
                else if (parameter.value.storage) {
                    initializer = new pytorch.Tensor('', parameter.value, this._littleEndian);
                }
                inputs.push(new pytorch.Parameter(inputName || parameter.key, visible, [ new pytorch.Argument('', null, initializer) ]));
            }
        }

        var group = groups.join('/');
        var name = group ? (group + '/' + key) : key;
        var type = obj.__type__ || '';

        var outputs = [ new pytorch.Parameter('output', true, [ new pytorch.Argument(name, null, null) ]) ];

        var attributes = [];
        for (let name of Object.keys(obj)) {
            if (!name.startsWith('_')) {
                attributes.push({ name: name, value: obj[name] });
            }
        }

        var node = new pytorch.Node(this._metadata, group, name, type, attributes, inputs, outputs);
        this._nodes.push(node);
        return node;
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

pytorch.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

pytorch.Argument = class {

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

    constructor(metadata, group, name, type, attributes, inputs, outputs) {
        this._metadata = metadata;
        this._group = group || '';
        this._name = name || '';
        let split = type.split('.');
        this._operator = split.pop();
        this._package = split.join('.');
        this._attributes = attributes.map((attribute) => new pytorch.Attribute(this._metadata, this, attribute.name, attribute.value));
        this._inputs = inputs;
        this._outputs = outputs;
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
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
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
                for (var attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (var input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get function() {
        return !this._package.startsWith('torch.nn.modules.');
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

    constructor(metadata, node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;

        var schema = metadata.getAttributeSchema(this._node.operator, this._name);
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (JSON.stringify(schema.default) == JSON.stringify(value)) {
                    this._visible = false;
                }
            }
        }

        if (Array.isArray(value) && value.every((obj) => obj.__type__ && obj.__type__.startsWith('torch.nn'))) {
            this._value = '?';
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

    constructor(name, tensor, littleEndian) {
        this._name = name || '';
        this._tensor = tensor;
        this._type = new pytorch.TensorType(tensor.storage.dataType, new pytorch.TensorShape(tensor.size));
        this._littleEndian = littleEndian;
    }

    get kind() {
        return 'Tensor';
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
        return pytorch.Tensor._stringify(value, '', '    ');
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
        var dimensions = context.dimensions;
        if (dimensions.length == 0) {
            dimensions = [ 1 ];
        }
        var size = dimensions[dimension];
        if (dimension == dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType)
                {
                    case 'uint8':
                        results.push(context.dataView.getUint8(context.index, this._littleEndian));
                        context.index++;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.dataView.getInt8(context.index, this._littleEndian));
                        context.index++;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.dataView.getInt16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.dataView.getInt32(context.index, this._littleEndian));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(new long.Long(context.dataView.getUint32(context.index, true), context.dataView.getUint32(context.index + 4, true), false));
                        context.index += 8;
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
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push(indentation + '[');
            var items = value.map((item) => pytorch.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value && long.Long.isLong(value)) {
            return indentation + value.toString();
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
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
        if (this._dimensions && this._dimensions.length > 0) {
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

pytorch.Metadata = class {

    static open(host) {
        if (pytorch.Metadata._metadata) {
            return Promise.resolve(pytorch.Metadata._metadata);
        }
        else {
            return host.request(null, 'pytorch-metadata.json', 'utf-8').then((data) => {
                pytorch.Metadata._metadata = new pytorch.Metadata(data);
                return pytorch.Metadata._metadata;
            }).catch(() => {
                pytorch.Metadata._metadata = new pytorch.Metadata(null);
                return pytorch.Metadata._metadata;
            });
        }
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
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
