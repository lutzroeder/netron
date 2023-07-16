
// Experimental

var bigdl = {};
var protobuf = require('./protobuf');

bigdl.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.has(2) && tags.has(7) && tags.has(8) && tags.has(9) && tags.has(10) && tags.has(11) && tags.has(12)) {
            return 'bigdl';
        }
        return '';
    }

    async open(context) {
        await context.require('./bigdl-proto');
        let module = null;
        try {
            // https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/resources/serialization/bigdl.proto
            bigdl.proto = protobuf.get('bigdl').com.intel.analytics.bigdl.serialization;
            const stream = context.stream;
            const reader = protobuf.BinaryReader.open(stream);
            module = bigdl.proto.BigDLModule.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new bigdl.Error('File format is not bigdl.BigDLModule (' + message.replace(/\.$/, '') + ').');
        }
        const metadata = await context.metadata('bigdl-metadata.json');
        return new bigdl.Model(metadata, module);
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
        const tensors = module.attr && module.attr.global_storage && module.attr.global_storage.nameAttrListValue && module.attr.global_storage.nameAttrListValue.attr ? module.attr.global_storage.nameAttrListValue.attr : {};
        const args = new Map();
        const arg = (name) => {
            if (!args.has(name)) {
                args.set(name, new bigdl.Value(name));
            }
            return args.get(name);
        };
        const loadModule = (metadata, module, tensors) => {
            switch (module.moduleType) {
                case 'com.intel.analytics.bigdl.nn.StaticGraph':
                case 'com.intel.analytics.bigdl.nn.Sequential': {
                    for (const submodule of module.subModules) {
                        loadModule(metadata, submodule, tensors);
                    }
                    break;
                }
                case 'com.intel.analytics.bigdl.nn.Input': {
                    this._inputs.push(new bigdl.Argument(module.name, [ arg(module.name) ]));
                    break;
                }
                default: {
                    this._nodes.push(new bigdl.Node(metadata, module, tensors, arg));
                    break;
                }
            }
        };
        loadModule(metadata, module, tensors);
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

bigdl.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

bigdl.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new bigdl.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, module, tensors, arg) {
        const type = module.moduleType;
        this._name = module.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._inputs.push(new bigdl.Argument('input', module.preModules.map((id) => arg(id))));
        this._type =  metadata.type(type) || { name: type };
        const inputs = (this._type && this._type.inputs) ? this._type.inputs.slice() : [];
        inputs.shift();
        if (module.weight) {
            inputs.shift();
            this._inputs.push(new bigdl.Argument('weight', [
                new bigdl.Value('', null, new bigdl.Tensor(module.weight, tensors))
            ]));
        }
        if (module.bias) {
            inputs.shift();
            this._inputs.push(new bigdl.Argument('bias', [
                new bigdl.Value('', null, new bigdl.Tensor(module.bias, tensors))
            ]));
        }
        if (module.parameters && module.parameters.length > 0) {
            for (const parameter of module.parameters) {
                const input = inputs.shift();
                const inputName = input ? input.name : this._inputs.length.toString();
                this._inputs.push(new bigdl.Argument(inputName, [
                    new bigdl.Value('', null, new bigdl.Tensor(parameter, tensors))
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
                    this._inputs.push(new bigdl.Argument(key, [ new bigdl.Value('', null, new bigdl.Tensor(value.tensorValue, tensors)) ]));
                }
                continue;
            }
            if (value.dataType === bigdl.proto.DataType.REGULARIZER && value.value === undefined) {
                continue;
            }
            if (value.dataType === bigdl.proto.DataType.ARRAY_VALUE && value.arrayValue.datatype === bigdl.proto.DataType.TENSOR) {
                this._inputs.push(new bigdl.Argument(key, value.arrayValue.tensor.map((tensor) => new bigdl.Value('', null, new bigdl.Tensor(tensor, tensors)))));
                continue;
            }
            this._attributes.push(new bigdl.Attribute(key, value));
        }
        const output = this._name || this._type + module.namePostfix;
        this._outputs.push(new bigdl.Argument('output', [ arg(output) ]));
    }

    get type() {
        return this._type;
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
                    default: throw new bigdl.Error("Unsupported data format '" + value.dataFormatValue + "'.");
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
};

bigdl.Tensor = class {

    constructor(tensor /*, tensors */) {
        this._type = new bigdl.TensorType(tensor.datatype, new bigdl.TensorShape(tensor.size));
        /*
        if (tensor && tensor.id && tensors && tensors[tensor.id] && tensors[tensor.id].tensorValue && tensors[tensor.id].tensorValue.storage) {
            const storage = tensors[tensor.id].tensorValue.storage;
            switch (this._type.dataType) {
                case 'float32':
                    if (storage.bytes_data && storage.bytes_data.length > 0) {
                        this._values = storage.bytes_data[0];
                        this._layout = '<';
                    }
                    else if (storage.float_data && storage.float_data.length > 0) {
                        this._values = storage.float_data;
                        this._layout = '|';
                    }
                    break;
                default:
                    break;
            }
        }
        */
    }

    get category() {
        return 'Weights';
    }

    get type() {
        return this._type;
    }

    get layout() {
        return this._layout;
    }

    get values() {
        return this._values;
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

bigdl.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading BigDL model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = bigdl.ModelFactory;
}
