
/*jshint esversion: 6 */

var torch = torch || {};
var base = base || require('./base');

torch.ModelFactory = class {

    match(context, host) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 't7') {
            var buffer = context.buffer;
            if (buffer.length >= 1 && buffer[0] > 58) {
                return false;
            }
            return true;
        }
        return false;
    }

    open(context, host, callback) {
        torch.Metadata.open(host, (err, metadata) => {
            var identifier = context.identifier;
            try {
                var reader = new torch.T7Reader(context.buffer, (name) => {
                    debugger;
                    host.exception(new torch.Error("Unknown type '" + name + "' in '" + identifier + "'."), false);
                    return null;
                });
                var root = reader.read();
                var model = new torch.Model(metadata, root);
                callback(null, model);
                return;
            }
            catch (error) {
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                callback(new torch.Error(message + " in '" + identifier + "'."), null);
                return;
            }
        });
    }
};

torch.Model = class {
    
    constructor(metadata, root) {
        this._graphs = [];
        this._graphs.push(new torch.Graph(metadata, root));
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return 'Torch v7';
    }
};

torch.Graph = class {

    constructor(metadata, root) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._groups = 'false';

        if (root.hasOwnProperty('model')) {
            root = root.model;
        }

        var inputs = [];
        var outputs = [];

        this._loadModule(metadata, root, [], '', inputs, outputs);

        inputs.forEach((input, index) => {
            this._inputs.push(new torch.Argument('input' + (index != 0 ? (index + 1).toString() : ''), true, [ input ]));
        });
        outputs.forEach((output, index) => {
            this._outputs.push(new torch.Argument('output' + (index != 0 ? (index + 1).toString() : ''), true, [ output ]));
        });
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

    get groups() {
        return this._groups;
    }

    _loadModule(metadata, module, groups, key, inputs, outputs) {
        if (groups.length > 0) {
            this._groups = true;
        }
        switch (module.__type__) {
            case 'nn.Sequential':
                groups.push(key);
                var subInputs = inputs;
                var subOutputs = [];
                var length = module.modules.length;
                module.modules.forEach((module, index) => {
                    if (index == length - 1) {
                        subOutputs = outputs;
                    }                    
                    this._loadModule(metadata, module, groups, index.toString(), subInputs, subOutputs);
                    subInputs = subOutputs;
                    subOutputs = [];
                });
                groups.pop();
                break;
            case 'nn.Parallel':
                groups.push(key);
                var newInputs = [];
                var newOutputs = [];
                module.modules.forEach((module, index) => {
                    var subInputs = inputs.map((input) => input);
                    var subOutputs = outputs.map((output) => output);
                    this._loadModule(metadata, module, groups, index.toString(), subInputs, subOutputs);
                    if (inputs.length == 0) {
                        subInputs.forEach((input) => {
                            newInputs.push(input);
                        });
                    }
                    if (outputs.length == 0) {
                        subOutputs.forEach((output) => {
                            newOutputs.push(output);
                        });
                    }
                });
                newInputs.forEach((input) => {
                    inputs.push(input);
                });
                newOutputs.forEach((output) => {
                    outputs.push(output);
                });
                groups.pop();
                break;
            case 'nn.Concat':
            case 'nn.ConcatTable':
                var prefix = key;
                if (inputs.length == 0) {
                    inputs.push(new torch.Connection(groups.join('/') + ':' + key + ':in', null, null));
                }
                var concatInputs = [];
                module.modules.forEach((module, index) => {
                    var streamInputs = inputs.map((input) => input);
                    var streamOutputs = [];
                    this._loadModule(metadata, module, groups, prefix + '.' + index.toString(), streamInputs, streamOutputs);
                    streamOutputs.forEach((output) => {
                        concatInputs.push(output);
                    });
                });
                delete module.modules;
                delete module.dimension;
                this._createNode(metadata, module, groups, key, concatInputs, outputs);
                break;
            case 'nn.Inception':
                delete module.modules; // TODO
                delete module.module; // TODO
                delete module.transfer; // TODO
                delete module.pool; // TODO
                this._createNode(metadata, module, groups, key, inputs, outputs);
                break;
            default:
                this._createNode(metadata, module, groups, key, inputs, outputs);
                break;
        }
    }

    _createNode(metadata, module, group, subIndex, inputs, outputs) {
        this._nodes.push(new torch.Node(metadata, module, group, subIndex, inputs, outputs));
    }
};

torch.Argument = class {

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

torch.Connection = class {

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

torch.Node = class {

    constructor(metadata, module, groups, name, inputs, outputs) {
        this._metadata = metadata;
        this._group = groups.join('/');
        if (module.name) {
            this._name = module.name;
            delete module.name;
        }
        else {
            this._name = this._group ? (this._group + ':' + name) : name;
        }
        var type = module.__type__;
        this._operator = type ? type.split('.').pop() : 'Unknown';
        var initializers = [];
        Object.keys(module).forEach((key) => {
            var obj = module[key];
            if (obj.__type__ && obj.__type__ == 'torch.LongStorage') {
                var array = [];
                obj.reset();
                for (var i = 0; i < obj.size; i++) {
                    array.push(obj.read());    
                }
                module[key] = array;
            }
        });
        delete module.iSize;
        delete module.finput;
        delete module.fgradInput;
        delete module.output;
        delete module.gradInput;
        delete module.gradWeight;
        delete module.gradBias;
        delete module.grad_tmp;
        delete module.scaleT;
        delete module._input;
        delete module._output;
        delete module._gradInput;
        delete module._gradOutput;
        delete module.buffer;
        delete module.buffer2;
        switch (type) {
            case 'nn.Linear':
                delete module.addBuffer;
                break;
            case 'nn.Normalize':
                delete module.addBuffer;
                delete module.normp;
                delete module.norm;
                break;
            case 'cudnn.SpatialConvolution':
            case 'cudnn.SpatialFullConvolution':
            case 'nn.SpatialConvolution':
            case 'nn.SpatialConvolutionMM':
            case 'nn.SpatialDilatedConvolution':
            case 'nn.SpatialFullConvolution':
                delete module.ones;
                delete module.input_slice;
                delete module.output_slice;
                delete module.convDescData;
                this._updateSize(module, 'adj');
                this._updateSize(module, 'd');
                this._updateSize(module, 'dilation');
                this._updateSize(module, 'k');
                this._updateSize(module, 'pad');
                break;
            case 'cudnn.BatchNormalization':
            case 'cudnn.SpatialBatchNormalization':
            case 'nn.BatchNormalization':
            case 'nn.SpatialBatchNormalization':
            case 'nn.InstanceNormalization':
                delete module.save_mean;
                delete module.save_std;
                delete module.gradWeight;
                delete module.normalized;
                delete module.centered;
                delete module.bn; // TODO InstanceNormalization
                break;
            case 'nn.SpatialCrossMapLRN':
                delete module.scale;
                break;
            case 'cudnn.SpatialMaxPooling':
            case 'cudnn.SpatialAveragePooling':
            case 'inn.SpatialMaxPooling':
            case 'nn.SpatialMaxPooling':
            case 'nn.SpatialAveragePooling':
                delete module.indices;
                this._updateSize(module, 'pad');
                this._updateSize(module, 'd');
                this._updateSize(module, 'k');
                break;
            case 'nn.SpatialZeroPadding':
            case 'nn.SpatialReflectionPadding':
                this._updateBox(module, 'pad');
                break;
            case 'nn.SpatialFullConvolution':
                delete module.ones;
                break;
            case 'nn.Dropout':
                delete module.noise;
                break;
            case 'nn.gModule':
                delete module.forwardnodes;
                delete module.backwardnodes;
                break;
        }
        this._attributes = [];
        if (module.__type__) {
            Object.keys(module).forEach((key) => {
                if (key == '__type__' || key == '_type') {
                    return;
                }
                var obj = module[key];
                if (obj.__type__ && obj.__type__.startsWith('torch.') && obj.__type__.endsWith('Tensor')) {
                    if (obj.size.length == 0) {
                        debugger;
                        // console.log("  " + type + "::" + key);
                    }
                    initializers.push(new torch.Argument(key, true, [ 
                        new torch.Connection(key, null, new torch.Tensor(obj))
                    ]));
                    return;
                }
                if (key == 'modules' || obj.__type__) {
                    debugger;                
                    // console.log("  " + type + "::" + key);
                    return;
                }
                this._attributes.push(new torch.Attribute(this._metadata, this._operator, key, obj));
            });
        }
        this._inputs = [];
        if (inputs.length == 0) {
            inputs.push(new torch.Connection(this._name + ':in', null, null));
        }
        this._inputs.push(new torch.Argument('input', true, inputs));
        this._outputs = [];
        if (outputs.length == 0) {
            outputs.push(new torch.Connection(this._name, null, null));
        }
        this._outputs.push(new torch.Argument('output', true, outputs));
        initializers = initializers.filter((argument) => {
            if (argument.name == 'weight') {
                this._inputs.push(argument);
                return false;
            }
            return true;
        });
        initializers = initializers.filter((argument) => {
            if (argument.name == 'bias') {
                this._inputs.push(argument);
                return false;
            }
            return true;
        });
        initializers.forEach((initialier) => {
            this._inputs.push(initialier);
        });
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get group() {
        return this._group;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
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

    _updateSize(module, name) {
        if (module.hasOwnProperty(name + 'W') && module.hasOwnProperty(name + 'H')) {
            module[name] = [ module[name + 'W'], module[name + 'H'] ];
            delete module[name + 'W'];
            delete module[name + 'H'];
        }
    }

    _updateBox(module, name) {
        if (module.hasOwnProperty(name + '_t') && module.hasOwnProperty(name + '_r') &&
            module.hasOwnProperty(name + '_b') && module.hasOwnProperty(name + '_l')) {
            module[name] = [ module[name + '_t'], module[name + '_r'], module[name + '_b'], module[name + '_l'] ];
            delete module[name + '_t'];
            delete module[name + '_r'];
            delete module[name + '_b'];
            delete module[name + '_l'];
        }
    }
};

torch.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;
        if (name == 'train') {
            this._visible = false;
        }
        var schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            if (schema.hasOwnProperty('visible')) {
                this._visible = schema.visible;
            }
            else if (schema.hasOwnProperty('default')) {
                if (JSON.stringify(schema.default) == JSON.stringify(this._value)) {
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
};

torch.Tensor = class {

    constructor(tensor) {
        this._type = new torch.TensorType(tensor);
        this._storage = tensor.storage;
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
        context.limit = 1000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        if (!this._storage || !this._storage.reader) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        switch (this._type.dataType) {
            case 'uint8':
            case 'int8':
            case 'int16':
            case 'int32':
            case 'int64':
            case 'float32':
            case 'float64':
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }
        context.dimensions = this._type.shape.dimensions;
        if (!context.dimensions && context.dimensions.length == 0) {
            context.state =  'Tensor has no dimensions.';
            return context;
        }
        context.storage = this._storage;
        context.storage.reset();
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
                results.push(context.storage.read());
                context.index++;
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
};

torch.TensorType = class {

    constructor(tensor) {
        this._dataType = tensor.storage ? tensor.storage.dataType : '?';
        this._shape = new torch.TensorShape(tensor.size);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

torch.TensorShape = class {

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


torch.Metadata = class {

    static open(host, callback) {
        if (torch.Metadata._metadata) {
            callback(null, torch.Metadata._metadata);
            return;
        }
        host.request(null, 'torch-metadata.json', 'utf-8', (err, data) => {
            torch.Metadata._metadata = new torch.Metadata(data);
            callback(null, torch.Metadata._metadata);
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

torch.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Torch model.';
    }
};

torch.T7Reader = class {

    constructor(buffer, callback) {
        this._callback = callback; 
        this._memo = {};

        this._registry = {};
        this._registry['cudnn.BatchNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.ReLU'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SoftMax'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialAveragePooling'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialBatchNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialFullConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialMaxPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.Tanh'] = function(reader, version) { reader.nn(this); };
        this._registry['inn.SpatialMaxPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.BinActiveZ'] = function(reader, version) { reader.nn(this); }; // allenai/XNOR-Net
        this._registry['nn.CAddTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Concat'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Copy'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.ConcatTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.DepthConcat'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Dropout'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Identity'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Inception'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.InstanceNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.JoinTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.LeakyReLU'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Linear'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.LogSoftMax'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Mean'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.MulConstant'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.MM'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Normalize'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Parallel'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.ParallelTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.ReLU'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Reshape'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.ShaveImage'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SelectTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Sequential'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Sigmoid'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SoftMax'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialAveragePooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialBatchNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialConvolutionMM'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialCrossMapLRN'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialDilatedConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialDropout'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialFullConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialLPPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialMaxPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialReflectionPadding'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialUpSamplingNearest'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialZeroPadding'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Square'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Sqrt'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Tanh'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.TotalVariation'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.View'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.gModule'] = function(reader, version) { reader.nn(this); };
        this._registry['nngraph.Node'] = function(reader, version) { reader.nn(this); };
        this._registry['graph.Edge'] = function(reader, version) { reader.nn(this); };
        this._registry['graph.Graph'] = function(reader, version) { reader.nn(this); };
        this._registry['torch.ByteTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.CharTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.ShortTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.IntTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.LongTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.FloatTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.DoubleTensor'] = function(reader, version) { reader.tensor(this); };
        this._registry['torch.CudaByteTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.CudaCharTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.CudaShortTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.CudaIntTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.CudaLongTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.CudaTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.CudaDoubleTensor'] = function(reader, version) {reader.tensor(this); };
        this._registry['torch.ByteStorage'] = function(reader, version) { reader.storage(this, 'uint8', 1); };
        this._registry['torch.CharStorage'] = function(reader, version) { reader.storage(this, 'int8', 1); };
        this._registry['torch.ShortStorage'] = function(reader, version) { reader.storage(this, 'int16', 2); };
        this._registry['torch.IntStorage'] = function(reader, version) { reader.storage(this, 'int32', 4); };
        this._registry['torch.LongStorage'] = function(reader, version) { reader.storage(this, 'int64', 8); };
        this._registry['torch.FloatStorage'] = function(reader, version) { reader.storage(this, 'float32', 4); };
        this._registry['torch.DoubleStorage'] = function(reader, version) { reader.storage(this, 'float64', 8); };
        this._registry['torch.CudaByteStorage'] = function(reader, version) { reader.storage(this, 'uint8', 1); };
        this._registry['torch.CudaCharStorage'] = function(reader, version) { reader.storage(this, 'int8', 1); };
        this._registry['torch.CudaShortStorage'] = function(reader, version) { reader.storage(this, 'int16', 2); };
        this._registry['torch.CudaIntStorage'] = function(reader, version) { reader.storage(this, 'int32', 4); };
        this._registry['torch.CudaLongStorage'] = function(reader, version) { reader.storage(this, 'int64', 8); };
        this._registry['torch.CudaIntStorage'] = function(reader, version) { reader.storage(this, 'int32', 4); };
        this._registry['torch.CudaStorage'] = function(reader, version) { reader.storage(this, 'float32', 4); };
        this._registry['torch.CudaFloatStorage'] = function(reader, version) { reader.storage(this, 'float64', 8); };
        this._registry['w2nn.AuxiliaryLossTable'] = function(reader, version) { reader.nn(this); };
        this._registry['w2nn.InplaceClip01'] = function(reader, version) { reader.nn(this); };
        this._registry['w2nn.ScaleTable'] = function(reader, version) { reader.nn(this); };

        if (buffer.length == 0) {
            throw new torch.Error('File is empty.');            
        }
        if (buffer[0] <= 8) {
            this._reader = new torch.BinaryReader(buffer);
        }
        else {
            this._reader = new torch.TextReader(buffer);
            this._reader.int32();
            this._reader.reset();
        }
    }

    read() {
        var type = this.int32();
        switch (type) {
            case 0: return null;
            case 1: return this.float64();
            case 2: return this.string();
            case 3: return this.table();
            case 4: return this.object();
            case 5: return this.boolean();
            case 6: return this.function();
            case 7: return this.function();
            case 8: return this.function();
            default: throw new torch.Error("File format has invalid type '" + type + "'.");
        }
    }

    boolean() {
        return this._reader.boolean();
    }

    bytes(size) {
        return this._reader.bytes(size);
    }

    int32() {
        return this._reader.int32();
    }

    int64() {
        return this._reader.int64();
    }

    int64s(size) {
        return this._reader.int64s(size);
    }

    float64() {
        return this._reader.float64();
    }

    string() {
        return this._reader.string();
    }

    object() {
        var index = this.int32();
        if (this._memo[index]) {
            return this._memo[index];
        }

        var version = this.string();
        var name = null;
        if (version.startsWith('V ')) {
            name = this.string();
            version = Number(version.split(' ')[1]);
        }
        else {
            name = version;
            version = 0;
        }

        var obj = { __type__: name };
        this._memo[index] = obj;

        var constructor = this._registry[name];
        if (constructor) {
            constructor.apply(obj, [ this, version ]);
        }
        else {
            constructor = this._callback(name);
            if (constructor) {
                constructor.apply(obj, [ this, version ]);
            }
            this.nn(obj);
        }
        return obj;
    }

    table() {
        var index = this.int32();
        if (this._memo[index]) {
            return this._memo[index];
        }
        var table = {};
        this._memo[index] = table;
        var size = this.int32();
        var convert = true;
        var sum = 0;
        for (var i = 0; i < size; i++) {
            var key = this.read();
            var value = this.read();
            table[key] = value;
            if (Number.isInteger(key) && key >= 0) {
                sum += key;
            }
            else {
                convert = false;
            }
        }
        var n = Object.keys(table).length;
        if (convert && (n * (n + 1)) == (2 * sum)) {
            var list = [];
            for (var j = 0; j < n; j++) {
                var item = table[j + 1];
                if (item == table) {
                    item = list;
                }
                list.push(item);
            }
            this._memo[index] = list;
            if (index == 1016) {
                debugger;
            }
            return list;
        }
        return table;
    }

    function() {
        var size = this.int32();
        var dumped = this.bytes(size);
        var upvalues = this.read();
        return { size: size, dumped: dumped, upvalues: upvalues };
    }

    nn(obj) {
        var attributes = this.read();
        if (attributes != null) {
            Object.keys(attributes).forEach((key) => {
                obj[key] = attributes[key];
            });
        }
    }

    tensor(obj) {
        var dim = this.int32();
        obj.size = this.int64s(dim);
        obj.stride = this.int64s(dim);
        obj.storage_offset = this.int64() - 1;
        obj.storage = this.read();
    }

    storage(obj, dataType, itemSize) {
        obj.dataType = dataType;
        obj.itemSize = itemSize;
        obj.size = this.int64();
        obj.reader = this._reader.storage(obj.size, obj.itemSize);
        obj.reset = function() {
            this.reader.reset();
        };
        obj.read = function() {
            switch (dataType) {
                case 'uint8':
                    return this.reader.byte();
                case 'int8':
                    return this.reader.int8();
                case 'int16':
                    return this.reader.int16();
                case 'int32':
                    return this.reader.int32();
                case 'int64':
                    return this.reader.int64();
                case 'float32':
                    return this.reader.float32();
                case 'float64':
                    return this.reader.float64();
            }
            return null;
        };
    }
};

torch.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        this._textDecoder = new TextDecoder('ascii');
    }

    reset() {
        this._position = 0;
    }

    boolean() {
        return this.int32() == 1;
    }

    bytes(size) {
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    int8() {
        var value = this._dataView.getInt8(this._position, true);
        this._position += 1;
        return value;
    }

    int16() {
        var value = this._dataView.getInt16(this._position, true);
        this._position += 2;
        return value;
    }

    int32() {
        var value = this._dataView.getInt32(this._position, true);
        this._position += 4;
        return value;
    }

    int64() {
        var lo = this._dataView.getInt32(this._position, true);
        var hi = this._dataView.getInt32(this._position, true);
        this._position += 8;
        if (hi == 0 || hi == -1) {
            return lo;
        }
        lo = this._dataView.getUint32(this._position - 8, true);
        hi = this._dataView.getInt32(this._position - 4, true);
        if (hi < 32677) {
            return hi << 32 || lo;
        }
        throw new torch.Error('Invalid int64 value.');
    }

    int64s(size) {
        var array = [];
        for (var i = 0; i < size; i++) {
            array.push(this.int64());
        }
        return array;
    }
    
    float32() {
        var value = this._dataView.getFloat32(this._position, true);
        this._position += 4;
        return value;
    }

    float64() {
        var value = this._dataView.getFloat64(this._position, true);
        this._position += 8;
        return value;
    }

    string() {
        var size = this.int32();
        var buffer = this.bytes(size);
        return this._textDecoder.decode(buffer);
    }

    storage(size, itemSize) {
        return new torch.BinaryReader(this.bytes(size * itemSize));
    }
};

torch.TextReader = class {

    constructor(buffer, separator) {
        this._buffer = buffer;
        this._position = 0;
        this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        this._textDecoder = new TextDecoder('ascii');
        this._separator = separator || 0x0a;
    }

    reset() {
        this._position = 0;
    }

    line(size) {
        var start = this._position;
        while (this._position < this._buffer.length && size > -1) {
            var c = this._buffer[this._position++];
            if (c == this._separator) {
                return this._buffer.slice(start, this._position - 1);
            }
            else if (this._position == this._buffer.length) {
                return this._buffer.slice(start, this._position);
            }
            size--;
        }
        throw torch.Error('Line exceeded maximum length.');
    }

    boolean() {
        return this.int32() == 1;
    }

    bytes(size) {
        throw new torch.Error('Byte array not supported in ASCII mode.');
    }

    int8() {
        return this.int64();
    }

    int16() {
        return this.int64();
    }

    int32() {
        return this.int64();
    }

    int64() {
        var token = this._textDecoder.decode(this.line(20));
        var number = Number.parseInt(token, 10);
        if (Number.isNaN(token - number)) {
            throw new torch.Error("Couldn't parse int64 '" + token + "'.");
        }
        return number;
    }

    int64s(size) {
        var array = [];
        if (size > 0) {
            var text = this._textDecoder.decode(this.line(Number.MAX_SAFE_INTEGER));
            text.split(' ').forEach((token) => {
                var number = Number.parseInt(token, 10);
                if (Number.isNaN(token - number)) {
                    throw new torch.Error("Couldn't parse int64 '" + token + "'.");
                }
                array.push(number);
            });
        }
        return array;
    }
    
    float32() {
        return this.float64();
    }

    float64() {
        var token = this._textDecoder.decode(this.line(20));
        if (token.startsWith('-nan')) {
            return -NaN;
        }
        if (token.startsWith('nan')) {
            return NaN;
        }
        if (token.startsWith('inf')) {
            return Infinity;
        }
        if (token.startsWith('-inf')) {
            return -Infinity;
        }
        var number = Number.parseFloat(token);
        if (Number.isNaN(token - number)) {
            throw new Error("Couldn't parse float '" + token + "'.");
        }
        return number;
    }

    string() {
        var size = this.int32();
        var text = this._textDecoder.decode(this.line(size));
        if (size != text.length) {
            throw torch.Error('Invalid text length.');
        }
        return text;
    }

    storage(size, itemSize) {
        var data = size > 0 ? this.line(Number.MAX_SAFE_INTEGER) : new Uint8Array(0);
        return new torch.TextReader(data, 0x20);
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = torch.ModelFactory;
}
