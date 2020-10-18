/* jshint esversion: 6 */

// Experimental

var dnn = dnn || {};

dnn.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(4) == 0 && tags.get(10) == 2) {
            return true;
        }
        return false;
    }

    open(context, host) {
        return host.require('./dnn-proto').then(() => {
            let model = null;
            const identifier = context.identifier;
            try {
                dnn.proto = protobuf.get('dnn').dnn;
                const reader = protobuf.Reader.create(context.buffer);
                model = dnn.proto.Model.decode(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new dnn.Error('File format is not dnn.Graph (' + message.replace(/\.$/, '') + ').');
            }
            return dnn.Metadata.open(host).then((metadata) => {
                return new dnn.Model(metadata, model);
            });
        });
    }
};

dnn.Model = class {

    constructor(metadata, model) {
        this._name = model.name || '';
        this._format = 'SnapML' + (model.version ? ' v' + model.version.toString() : '');
        this._graphs = [ new dnn.Graph(metadata, model) ];
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get graphs() {
        return this._graphs;
    }
};

dnn.Graph = class {

    constructor(metadata, model) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const scope = {};
        let index = 0;
        for (const node of model.node) {
            node.input = node.input.map((input) => scope[input] ? scope[input] : input);
            node.output = node.output.map((output) => {
                scope[output] = scope[output] ? output + '\n' + index.toString() : output; // custom argument id
                return scope[output];
            });
            index++;
        }

        const args = new Map();
        const arg = (name, type) => {
            if (!args.has(name)) {
                args.set(name, new dnn.Argument(name, type));
            }
            return args.get(name);
        };

        for (const input of model.input) {
            const shape = input.shape;
            const type = new dnn.TensorType('float32', new dnn.TensorShape([ shape.dim0, shape.dim1, shape.dim2, shape.dim3 ]));
            this._inputs.push(new dnn.Parameter(input.name, [ arg(input.name, type) ]));
        }
        for (const output of model.output) {
            const shape = output.shape;
            const type = new dnn.TensorType('float32', new dnn.TensorShape([ shape.dim0, shape.dim1, shape.dim2, shape.dim3 ]));
            this._outputs.push(new dnn.Parameter(output.name, [ arg(output.name, type) ]));
        }
        if (this._inputs.length === 0 && model.input_name && model.input_shape && model.input_shape.length === model.input_name.length * 4) {
            for (let i = 0; i < model.input_name.length; i++) {
                const name = model.input_name[i];
                const shape = model.input_shape.slice(i * 4, (i * 4 + 4));
                const type = new dnn.TensorType('float32', new dnn.TensorShape([ shape[1], shape[3], shape[2], shape[0] ]));
                this._inputs.push(new dnn.Parameter(name, [ arg(name, type) ]));
            }
        }
        if (this._inputs.length === 0 &&  model.input_shape && model.input_shape.length === 4 &&
            model.node.length > 0 && model.node[0].input.length > 0) {
            const name = model.node[0].input[0];
            const shape = model.input_shape;
            const type = new dnn.TensorType('float32', new dnn.TensorShape([ shape[1], shape[3], shape[2], shape[0] ]));
            this._inputs.push(new dnn.Parameter(name, [ arg(name, type) ]));
        }

        for (const node of model.node) {
            this._nodes.push(new dnn.Node(metadata, node, arg));
        }
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

dnn.Parameter = class {

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

dnn.Argument = class {

    constructor(name, type, initializer, quantization) {
        if (typeof name !== 'string') {
            throw new dnn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._quantization = quantization || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        if (this._quantization) {
            return this._quantization.map((value, index) => index.toString() + ' = ' + value.toString()).join('; ');
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

dnn.Node = class {

    constructor(metadata, node, arg) {
        const layer = node.layer;
        this._name = layer.name;
        this._type = layer.type;
        this._metadata = metadata.type(this._type);
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        const inputs = node.input.map((input) => { return arg(input); });
        for (const weight of layer.weight) {
            let quantization = null;
            if (layer.is_quantized && weight === layer.weight[0] && layer.quantization && layer.quantization.data) {
                const data = layer.quantization.data;
                quantization = new Array(data.length >> 2);
                const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                for (let i = 0; i < quantization.length; i++) {
                    quantization[i] = view.getFloat32(i << 2, true);
                }
            }
            const initializer = new dnn.Tensor(weight, quantization);
            inputs.push(new dnn.Argument('', initializer.type, initializer, quantization));
        }
        const outputs = node.output.map((output) => { return arg(output); });

        const schema = this._metadata;
        if (inputs && inputs.length > 0) {
            let inputIndex = 0;
            if (schema && schema.inputs) {
                for (const inputSchema of schema.inputs) {
                    if (inputIndex < inputs.length || inputSchema.option != 'optional') {
                        const inputCount = (inputSchema.option == 'variadic') ? (node.input.length - inputIndex) : 1;
                        const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount);
                        this._inputs.push(new dnn.Parameter(inputSchema.name, inputArguments));
                        inputIndex += inputCount;
                    }
                }
            }
            this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new dnn.Parameter(inputName, [ input ]);
            }));
        }
        if (outputs.length > 0) {
            this._outputs = outputs.map((output, index) => {
                const inputName = (index == 0) ? 'output' : index.toString();
                return new dnn.Parameter(inputName, [ output ]);
            });
        }

        for (const key of Object.keys(layer)) {
            switch (key) {
                case 'name':
                case 'type':
                case 'weight':
                case 'is_quantized':
                case 'quantization':
                    break;
                default:
                    this._attributes.push(new dnn.Attribute(metadata.attribute(this._type, key), key, layer[key]));
                    break;
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get metadata() {
        return this._metadata;
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

dnn.Attribute = class {

    constructor(metadata, name, value) {
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

dnn.Tensor = class {

    constructor(weight, quantization) {
        const shape = new dnn.TensorShape([ weight.dim0, weight.dim1, weight.dim2, weight.dim3 ]);
        this._data = quantization ? weight.quantized_data : weight.data;

        const size = shape.dimensions.reduce((a, b) => a * b, 1);
        const itemSize = Math.floor(this._data.length / size);
        const remainder = this._data.length - (itemSize * size);
        if (remainder < 0 || remainder > itemSize) {
            throw new dnn.Error('Invalid tensor data size.');
        }
        switch (itemSize) {
            case 1:
                this._type = new dnn.TensorType('int8', shape);
                break;
            case 2:
                this._type = new dnn.TensorType('float16', shape);
                break;
            case 4:
                this._type = new dnn.TensorType('float16', shape);
                break;
            default:
                this._type = new dnn.TensorType('?', shape);
                break;
        }
    }

    get kind() {
        return 'Weight';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        switch (this._type.dataType) {
            case 'int8':
            case 'float16':
            case 'float32':
                break;
            default:
                context.state = "Tensor data type '" + this._type.dataType + "' is not supported.";
                return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length == 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        const results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index++;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};

dnn.TensorType = class {

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
        return this.dataType + this._shape.toString();
    }
};

dnn.TensorShape = class {

    constructor(shape) {
        this._dimensions = shape;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.join(',') + ']';
    }
};

dnn.Metadata = class {

    static open(host) {
        if (dnn.Metadata._metadata) {
            return Promise.resolve(dnn.Metadata._metadata);
        }
        return host.request(null, 'dnn-metadata.json', 'utf-8').then((data) => {
            dnn.Metadata._metadata = new dnn.Metadata(data);
            return dnn.Metadata._metadata;
        }).catch(() => {
            dnn.Metadata._metadata = new dnn.Metadata(null);
            return dnn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

dnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading SnapML model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = dnn.ModelFactory;
}
