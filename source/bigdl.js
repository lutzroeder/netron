/* jshint esversion: 6 */

// Experimental

var bigdl = bigdl || {};
var protobuf = protobuf || require('./protobuf');

bigdl.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.has(2) && tags.has(7) && tags.has(8) && tags.has(9) && tags.has(10) && tags.has(11) && tags.has(12)) {
            return true;
        }
        return false;
    }

    open(context) {
        return context.require('./bigdl-proto').then(() => {
            let module = null;
            try {
                // https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/resources/serialization/bigdl.proto
                bigdl.proto = protobuf.get('bigdl').com.intel.analytics.bigdl.serialization;
                const reader = protobuf.Reader.create(context.stream.peek());
                module = bigdl.proto.BigDLModule.decode(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new bigdl.Error('File format is not bigdl.BigDLModule (' + message.replace(/\.$/, '') + ').');
            }
            return bigdl.Metadata.open(context).then((metadata) => {
                return new bigdl.Model(metadata, module);
            });
        });
    }
};

bigdl.Model = class {

    constructor(metadata, module) {
        this._version = module && module.version ? module.version : '';
        this._graphs = [ new bigdl.Graph(metadata, module) ];
    }

    get format() {
        return 'BigDL' + (this._version ? ' v' + this._version : '');
    }

    get graphs() {
        return this._graphs;
    }
};

bigdl.Graph = class {

    constructor(metadata, module) {
        this._type = module.moduleType;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._loadModule(metadata, '', module);
    }

    _loadModule(metadata, group, module) {
        switch (module.moduleType) {
            case 'com.intel.analytics.bigdl.nn.StaticGraph': {
                this._loadStaticGraph(metadata, group, module);
                break;
            }
            case 'com.intel.analytics.bigdl.nn.Sequential': {
                this._loadSequential(metadata, group, module);
                break;
            }
            case 'com.intel.analytics.bigdl.nn.Input': {
                this._inputs.push(new bigdl.Parameter(module.name, [
                    new bigdl.Argument(module.name)
                ]));
                break;
            }
            default: {
                this._nodes.push(new bigdl.Node(metadata, group, module));
                break;
            }
        }
    }

    _loadSequential(metadata, group, module) {
        group = group.length > 0 ?  group + '.' + module.namePostfix : module.namePostfix;
        for (const submodule of module.subModules) {
            this._loadModule(metadata, group, submodule);
        }
    }

    _loadStaticGraph(metadata, group, module) {
        group = group.length > 0 ?  group + '.' + module.namePostfix : module.namePostfix;
        for (const submodule of module.subModules) {
            this._loadModule(metadata, group, submodule);
        }
    }

    get groups() {
        return this._groups || false;
    }

    get type() {
        return this._type;
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

bigdl.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

bigdl.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new bigdl.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
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

bigdl.Node = class {

    constructor(metadata, group, module) {
        this._metadata = metadata;
        this._group = group;
        this._type = module.moduleType.split('.').pop();
        this._name = module.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._inputs.push(new bigdl.Parameter('input', module.preModules.map((id) => new bigdl.Argument(id, null, null))));
        const schema =  metadata.type(this.type);
        const inputs = (schema && schema.inputs) ? schema.inputs.slice() : [];
        inputs.shift();
        if (module.weight) {
            inputs.shift();
            this._inputs.push(new bigdl.Parameter('weight', [
                new bigdl.Argument('', null, new bigdl.Tensor(module.weight))
            ]));
        }
        if (module.bias) {
            inputs.shift();
            this._inputs.push(new bigdl.Parameter('bias', [
                new bigdl.Argument('', null, new bigdl.Tensor(module.bias))
            ]));
        }
        if (module.parameters && module.parameters.length > 0) {
            for (const parameter of module.parameters) {
                const input = inputs.shift();
                const inputName = input ? input.name : this._inputs.length.toString();
                this._inputs.push(new bigdl.Parameter(inputName, [
                    new bigdl.Argument('', null, new bigdl.Tensor(parameter))
                ]));
            }
        }
        for (const key of Object.keys(module.attr)) {
            const value = module.attr[key];
            if (key === 'module_numerics' || key === 'module_tags') {
                continue;
            }
            if (value.dataType === bigdl.proto.DataType.TENSOR) {
                if (value.value) {
                    this._inputs.push(new bigdl.Parameter(key, [ new bigdl.Argument('', null, new bigdl.Tensor(value.tensorValue)) ]));
                }
                continue;
            }
            if (value.dataType === bigdl.proto.DataType.REGULARIZER && value.value === undefined) {
                continue;
            }
            if (value.dataType === bigdl.proto.DataType.ARRAY_VALUE && value.arrayValue.datatype === bigdl.proto.DataType.TENSOR) {
                this._inputs.push(new bigdl.Parameter(key, value.arrayValue.tensor.map((tensor) => new bigdl.Argument('', null, new bigdl.Tensor(tensor)))));
                continue;
            }
            this._attributes.push(new bigdl.Attribute(key, value));
        }
        const output = this._name || this._type + module.namePostfix;
        this._outputs.push(new bigdl.Parameter('output', [
            new bigdl.Argument(output, null, null)
        ]));
    }

    get group() {
        return this._group;
    }

    get type() {
        return this._type;
    }

    get metadata() {
        return this._metadata.type(this._type);
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }
};

bigdl.Attribute = class {

    constructor(name, value) {
        this._name = name;
        switch (value.dataType) {
            case bigdl.proto.DataType.INT32: {
                this._type = 'int32';
                this._value = value.int32Value;
                break;
            }
            case bigdl.proto.DataType.FLOAT: {
                this._type = 'float32';
                this._value = value.floatValue;
                break;
            }
            case bigdl.proto.DataType.DOUBLE: {
                this._type = 'float64';
                this._value = value.doubleValue;
                break;
            }
            case bigdl.proto.DataType.BOOL: {
                this._type = 'boolean';
                this._value = value.boolValue;
                break;
            }
            case bigdl.proto.DataType.REGULARIZER: {
                this._value = value.value;
                break;
            }
            case bigdl.proto.DataType.MODULE: {
                this._value = value.bigDLModule;
                break;
            }
            case bigdl.proto.DataType.NAME_ATTR_LIST: {
                this._value = value.nameAttrListValue;
                break;
            }
            case bigdl.proto.DataType.ARRAY_VALUE: {
                switch (value.arrayValue.datatype) {
                    case bigdl.proto.DataType.INT32: {
                        this._type = 'int32[]';
                        this._value = value.arrayValue.i32;
                        break;
                    }
                    case bigdl.proto.DataType.FLOAT: {
                        this._type = 'float32[]';
                        this._value = value.arrayValue.flt;
                        break;
                    }
                    case bigdl.proto.DataType.STRING: {
                        this._type = 'string[]';
                        this._value = value.arrayValue.str;
                        break;
                    }
                    case bigdl.proto.DataType.TENSOR: {
                        this._type = 'tensor[]';
                        this._value = value.arrayValue.tensor;
                        break;
                    }
                    default: {
                        throw new bigdl.Error("Unsupported attribute array data type '" + value.arrayValue.datatype + "'.");
                    }
                }
                break;
            }
            case bigdl.proto.DataType.DATA_FORMAT: {
                this._dataType = 'InputDataFormat';
                switch (value.dataFormatValue) {
                    case 0: this._value = 'NCHW'; break;
                    case 1: this._value = 'NHWC'; break;
                }
                break;
            }
            default: {
                throw new bigdl.Error("Unsupported attribute data type '" + value.dataType + "'.");
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return true;
    }
};

bigdl.Tensor = class {

    constructor(tensor) {
        this._type = new bigdl.TensorType(tensor.datatype, new bigdl.TensorShape(tensor.size));
    }

    get kind() {
        return 'Parameter';
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Not supported.';
    }

    get value() {
        return null;
    }

    toString() {
        return '';
    }
};

bigdl.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case bigdl.proto.DataType.FLOAT: this._dataType = 'float32'; break;
            case bigdl.proto.DataType.DOUBLE: this._dataType = 'float64'; break;
            default: throw new bigdl.Error("Unsupported tensor type '" + dataType + "'.");
        }
        this._shape = shape;
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

bigdl.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
        if (!dimensions.every((dimension) => Number.isInteger(dimension))) {
            throw new bigdl.Error("Invalid tensor shape '" + JSON.stringify(dimensions) + "'.");
        }
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

bigdl.Metadata = class {

    static open(context) {
        if (bigdl.Metadata._metadata) {
            return Promise.resolve(bigdl.Metadata._metadata);
        }
        return context.request('bigdl-metadata.json', 'utf-8', null).then((data) => {
            bigdl.Metadata._metadata = new bigdl.Metadata(data);
            return bigdl.Metadata._metadata;
        }).catch(() => {
            bigdl.Metadata._metadata = new bigdl.Metadata(null);
            return bigdl.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = {};
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }
};

bigdl.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading BigDL model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = bigdl.ModelFactory;
}
