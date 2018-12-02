
/*jshint esversion: 6 */

var torch = torch || {};
var base = base || require('./base');

torch.ModelFactory = class {

    match(context, host) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 't7') {
            return true;
        }
        return false;
    }

    open(context, host, callback) {
        torch.OperatorMetadata.open(host, (err, metadata) => {
            var identifier = context.identifier;
            try {
                var buffer = context.buffer;
                var reader = new torch.T7Reader(buffer, (name) => {
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
                var keys = Object.keys(module.modules);
                keys.sort(); 
                var last = keys[keys.length - 1];
                keys.forEach((key, index) => {
                    if (key == last.toString()) {
                        subOutputs = outputs;
                    }
                    this._loadModule(metadata, module.modules[key], groups, key, subInputs, subOutputs);
                    subInputs = subOutputs;
                    subOutputs = [];
                });
                groups.pop();
                break;
            case 'nn.Parallel':
                groups.push(key);
                var keys = Object.keys(module.modules);
                keys.sort();
                var newInputs = [];
                var newOutputs = [];
                keys.forEach((key, index) => {
                    var subInputs = inputs.map((input) => input);
                    var subOutputs = outputs.map((output) => output);
                    this._loadModule(metadata, module.modules[key], groups, key, subInputs, subOutputs);
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
                groups.push(key);
                var keys = Object.keys(module.modules);
                keys.sort();
                if (inputs.length == 0) {
                    inputs.push(new torch.Connection(groups.join('/') + '/' + key, null, null));
                }
                var concatInputs = [];
                keys.forEach((key, index) => {
                    var streamInputs = inputs.map((input) => input);
                    var streamOutputs = [];
                    this._loadModule(metadata, module.modules[key], groups, key, streamInputs, streamOutputs);
                    streamOutputs.forEach((output) => {
                        concatInputs.push(output);
                    });
                });
                groups.pop();
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

    constructor(metadata, module, groups, key, inputs, outputs) {
        this._metadata = metadata;
        this._group = groups.join('/');
        this._name = this._group + '/' + key;
        var type = module.__type__;
        this._operator = type ? type.split('.').pop() : 'Object';
        var initializers = [];
        Object.keys(module).forEach((key) => {
            var obj = module[key];
            if (obj.__type__ && obj.__type__ == 'torch.LongStorage') {
                var array = [];
                var reader = new torch.T7Reader(obj.data);
                for (var i = 0; i < obj.size; i++) {
                    array.push(reader.int64());
                }
                module[key] = array;
            }
        });
        delete module.iSize;
        delete module.gradInput;
        delete module.finput;
        delete module.fgradInput;
        delete module.output;
        delete module.gradWeight;
        delete module.gradBias;
        delete module.scaleT;
        switch (type) {
            case 'nn.Linear':
                delete module.addBuffer;
                break;
            case 'nn.Reshape':
                delete module._input;
                delete module._gradOutput;
                break;
            case 'cudnn.SpatialConvolution':
            case 'nn.SpatialConvolution':
            case 'nn.SpatialDilatedConvolution':
            case 'nn.SpatialFullConvolution':
                delete module.ones;
                this._updateWidthHeight(module, 'adj');
                this._updateWidthHeight(module, 'd');
                this._updateWidthHeight(module, 'dilation');
                this._updateWidthHeight(module, 'k');
                this._updateWidthHeight(module, 'pad');
                break;
            case 'cudnn.BatchNormalization':
            case 'cudnn.SpatialBatchNormalization':
            case 'nn.BatchNormalization':
            case 'nn.SpatialBatchNormalization':
                delete module.save_mean;
                delete module.save_std;
                delete module.gradWeight;
                module.mean = module.running_mean;
                module.var = module.running_var;
                delete module.running_mean;
                delete module.running_var;
                break;
            case 'cudnn.SpatialMaxPooling':
            case 'inn.SpatialMaxPooling':
            case 'nn.SpatialMaxPooling':
                delete module.indices;
                this._updateWidthHeight(module, 'pad');
                this._updateWidthHeight(module, 'd');
                this._updateWidthHeight(module, 'k');
                break;
            case 'cudnn.SpatialAveragePooling':
            case 'nn.SpatialAveragePooling':
                this._updateWidthHeight(module, 'd');
                this._updateWidthHeight(module, 'k');
                break;    
            case 'nn.SpatialFullConvolution':
                delete module.ones;
                break;
            case 'nn.Dropout':
                delete module.noise;
                break;
        }
        this._attributes = [];
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
        this._inputs = [];
        if (inputs.length == 0) {
            inputs.push(new torch.Connection(this._name + '/in', null, null));
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

    _updateWidthHeight(module, name) {
        if (module.hasOwnProperty(name + 'W') && module.hasOwnProperty(name + 'H')) {
            module[name] = [ module[name + 'W'], module[name + 'H'] ];
            delete module[name + 'W'];
            delete module[name + 'H'];
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
        return 'Not implemented.';
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

torch.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Torch model.';
    }
};

torch.T7Reader = class {

    constructor(buffer, callback) {
        this._buffer = buffer;
        this._position = 0;
        this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        this._textDecoder = new TextDecoder('ascii');
        this._callback = callback; 
        this._memo = {};
        this._registry = {};
        this._registry['cudnn.BatchNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.ReLU'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialAveragePooling'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialBatchNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['cudnn.SpatialMaxPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['inn.SpatialMaxPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.CAddTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Concat'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.ConcatTable'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.DepthConcat'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Dropout'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Identity'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Inception'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Linear'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Parallel'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.ReLU'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Reshape'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Sequential'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.Sigmoid'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialAveragePooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialBatchNormalization'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialDilatedConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialFullConvolution'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialMaxPooling'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.SpatialZeroPadding'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.View'] = function(reader, version) { reader.nn(this); };
        this._registry['nn.gModule'] = function(reader, version) { reader.nn(this); };
        this._registry['nngraph.Node'] = function(reader, version) { reader.nn(this); };
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
    }

    read() {
        var type = this.int32();
        switch (type) {
            case 0:
                return null;
            case 1:
                return  this.float64();
            case 2:
                return this.string();
            case 3:
                return this.table();
            case 4:
                return this.object();
            case 5:
                return this.boolean();
            case 6:
            case 7:
            case 8:
                return this.function();
            default:
                throw new torch.Error("File format has invalid type '" + type + "'.");
        }
    }

    boolean() {
        return this.int32() == 1;
    }

    bytes(size) {
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    int32() {
        var value = this._dataView.getInt32(this._position, true);
        this._position += 4;
        return value;
    }

    int64() {
        var lo = this.int32();
        var hi = this.int32();
        if (lo == -1 && hi == -1) {
            return -1;
        }
        if (hi != 0) {
            throw new torch.Error('Invalid int64 value.');
        }
        return lo;
    }

    int64s(size) {
        var array = [];
        for (var i = 0; i < size; i++) {
            array.push(this.int64());
        }
        return array;
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
        this._memo[index] = obj;
        return obj;
    }

    table() {
        var index = this.int32();
        if (this._memo[index]) {
            return this._memo[index];
        }
        var size = this.int32();
        var table = {};
        for (var i = 0; i < size; i++) {
            var key = this.read();
            var value = this.read();
            table[key] = value;
        }
        var keys = Object.keys(table);
        keys.sort();
        var list = true;
        for (var j = 0; j < keys.length; j++) {
            if (keys[j] != j.toString()) {
                list = false;
            }
        }
        if (list && keys.length > 0) {
            debugger;
        }

        this._memo[index] = table;
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
        obj.size = [];
        for (var i = 0; i < dim; i++) {
            obj.size.push(this.int64());
        }
        obj.stride = [];
        for (var j = 0; j < dim; j++) {
            obj.stride.push(this.int64());
        }
        obj.storage_offset = this.int64() - 1;
        obj.storage = this.read();
    }

    storage(obj, dataType, itemSize) {
        obj.dataType = dataType;
        obj.itemSize = itemSize;
        obj.size = this.int64();
        obj.data = this.bytes(obj.size * obj.itemSize);
    }
};

torch.OperatorMetadata = class {

    static open(host, callback) {
        if (torch.OperatorMetadata._metadata) {
            callback(null, torch.OperatorMetadata._metadata);
            return;
        }
        host.request(null, 'torch-metadata.json', 'utf-8', (err, data) => {
            torch.OperatorMetadata._metadata = new torch.OperatorMetadata(data);
            callback(null, torch.OperatorMetadata._metadata);
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = torch.ModelFactory;
}

