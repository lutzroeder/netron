
// Experimental

const bigdl = {};

bigdl.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.has(2) && tags.has(7) && tags.has(8) &&
            tags.has(9) && tags.has(10) && tags.has(11) && tags.has(12)) {
            context.type = 'bigdl';
        }
    }

    async open(context) {
        bigdl.proto = await context.require('./bigdl-proto');
        bigdl.proto = bigdl.proto.com.intel.analytics.bigdl.serialization;
        let module = null;
        try {
            // https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/resources/serialization/bigdl.proto
            const reader = context.read('protobuf.binary');
            module = bigdl.proto.BigDLModule.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new bigdl.Error(`File format is not bigdl.BigDLModule (${message.replace(/\.$/, '')}).`);
        }
        const metadata = await context.metadata('bigdl-metadata.json');
        return new bigdl.Model(metadata, module);
    }
};

bigdl.Model = class {

    constructor(metadata, module) {
        const version = module && module.version ? module.version : '';
        this.format = `BigDL${version ? ` v${version}` : ''}`;
        this.graphs = [new bigdl.Graph(metadata, module)];
    }
};

bigdl.Graph = class {

    constructor(metadata, module) {
        this.type = module.moduleType;
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const tensors = module.attr && module.attr.global_storage && module.attr.global_storage.nameAttrListValue && module.attr.global_storage.nameAttrListValue.attr ? module.attr.global_storage.nameAttrListValue.attr : {};
        const values = new Map();
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, new bigdl.Value(name));
            }
            return values.get(name);
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
                    const argument = new bigdl.Argument(module.name, [values.map(module.name)]);
                    this.inputs.push(argument);
                    break;
                }
                default: {
                    const node = new bigdl.Node(metadata, module, tensors, values);
                    this.nodes.push(node);
                    break;
                }
            }
        };
        loadModule(metadata, module, tensors);
    }
};

bigdl.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

bigdl.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new bigdl.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

bigdl.Node = class {

    constructor(metadata, module, tensors, values) {
        const type = module.moduleType;
        this.name = module.name;
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        this.inputs.push(new bigdl.Argument('input', module.preModules.map((id) => values.map(id))));
        this.type =  metadata.type(type) || { name: type };
        const inputs = this.type && this.type.inputs ? this.type.inputs.slice() : [];
        inputs.shift();
        if (module.weight) {
            inputs.shift();
            this.inputs.push(new bigdl.Argument('weight', [
                new bigdl.Value('', null, new bigdl.Tensor(module.weight, tensors))
            ]));
        }
        if (module.bias) {
            inputs.shift();
            this.inputs.push(new bigdl.Argument('bias', [
                new bigdl.Value('', null, new bigdl.Tensor(module.bias, tensors))
            ]));
        }
        if (module.parameters && module.parameters.length > 0) {
            for (const parameter of module.parameters) {
                const input = inputs.shift();
                const inputName = input ? input.name : this.inputs.length.toString();
                this.inputs.push(new bigdl.Argument(inputName, [
                    new bigdl.Value('', null, new bigdl.Tensor(parameter, tensors))
                ]));
            }
        }
        for (const [key, obj] of Object.entries(module.attr)) {
            if (key === 'module_numerics' || key === 'module_tags') {
                continue;
            }
            if (obj.dataType === bigdl.proto.DataType.TENSOR) {
                if (obj.value) {
                    this.inputs.push(new bigdl.Argument(key, [new bigdl.Value('', null, new bigdl.Tensor(obj.tensorValue, tensors))]));
                }
                continue;
            }
            if (obj.dataType === bigdl.proto.DataType.REGULARIZER && obj.value === undefined) {
                continue;
            }
            if (obj.dataType === bigdl.proto.DataType.ARRAY_VALUE && obj.arrayValue.datatype === bigdl.proto.DataType.TENSOR) {
                this.inputs.push(new bigdl.Argument(key, obj.arrayValue.tensor.map((tensor) => new bigdl.Value('', null, new bigdl.Tensor(tensor, tensors)))));
                continue;
            }
            let type = null;
            let value = null;
            switch (obj.dataType) {
                case bigdl.proto.DataType.INT32: {
                    type = 'int32';
                    value = obj.int32Value;
                    break;
                }
                case bigdl.proto.DataType.FLOAT: {
                    type = 'float32';
                    value = obj.floatValue;
                    break;
                }
                case bigdl.proto.DataType.DOUBLE: {
                    type = 'float64';
                    value = obj.doubleValue;
                    break;
                }
                case bigdl.proto.DataType.BOOL: {
                    type = 'boolean';
                    value = obj.boolValue;
                    break;
                }
                case bigdl.proto.DataType.REGULARIZER: {
                    value = obj.value;
                    break;
                }
                case bigdl.proto.DataType.MODULE: {
                    value = obj.bigDLModule;
                    break;
                }
                case bigdl.proto.DataType.NAME_ATTR_LIST: {
                    value = value.nameAttrListValue;
                    break;
                }
                case bigdl.proto.DataType.ARRAY_VALUE: {
                    switch (obj.arrayValue.datatype) {
                        case bigdl.proto.DataType.INT32: {
                            type = 'int32[]';
                            value = obj.arrayValue.i32;
                            break;
                        }
                        case bigdl.proto.DataType.FLOAT: {
                            type = 'float32[]';
                            value = obj.arrayValue.flt;
                            break;
                        }
                        case bigdl.proto.DataType.STRING: {
                            type = 'string[]';
                            value = obj.arrayValue.str;
                            break;
                        }
                        case bigdl.proto.DataType.TENSOR: {
                            type = 'tensor[]';
                            value = obj.arrayValue.tensor;
                            break;
                        }
                        default: {
                            throw new bigdl.Error(`Unsupported attribute array data type '${obj.arrayValue.datatype}'.`);
                        }
                    }
                    break;
                }
                case bigdl.proto.DataType.DATA_FORMAT: {
                    switch (obj.dataFormatValue) {
                        case 0: value = 'NCHW'; break;
                        case 1: value = 'NHWC'; break;
                        default: throw new bigdl.Error(`Unsupported data format '${obj.dataFormatValue}'.`);
                    }
                    break;
                }
                default: {
                    throw new bigdl.Error(`Unsupported attribute data type '${obj.dataType}'.`);
                }
            }
            const argument = new bigdl.Argument(key, value, type);
            this.attributes.push(argument);
        }
        const output = this.name || this.type + module.namePostfix;
        this.outputs.push(new bigdl.Argument('output', [values.map(output)]));
    }
};

bigdl.Tensor = class {

    constructor(tensor /*, tensors */) {
        this.type = new bigdl.TensorType(tensor.datatype, new bigdl.TensorShape(tensor.size));
        /*
        if (tensor && tensor.id && tensors && tensors[tensor.id] && tensors[tensor.id].tensorValue && tensors[tensor.id].tensorValue.storage) {
            const storage = tensors[tensor.id].tensorValue.storage;
            switch (this.type.dataType) {
                case 'float32':
                    if (storage.bytes_data && storage.bytes_data.length > 0) {
                        this.values = storage.bytes_data[0];
                        this.encoding = '<';
                    }
                    else if (storage.float_data && storage.float_data.length > 0) {
                        this.values = storage.float_data;
                        this.encoding = '|';
                    }
                    break;
                default:
                    break;
            }
        }
        */
    }
};

bigdl.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case bigdl.proto.DataType.FLOAT: this.dataType = 'float32'; break;
            case bigdl.proto.DataType.DOUBLE: this.dataType = 'float64'; break;
            default: throw new bigdl.Error(`Unsupported tensor type '${dataType}'.`);
        }
        this.shape = shape;
    }

    toString() {
        return (this.dataType || '?') + this.shape.toString();
    }
};

bigdl.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
        if (!dimensions.every((dimension) => Number.isInteger(dimension))) {
            throw new bigdl.Error(`Invalid tensor shape '${JSON.stringify(dimensions)}'.`);
        }
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`) : '';
    }
};

bigdl.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading BigDL model.';
    }
};

export const ModelFactory = bigdl.ModelFactory;

