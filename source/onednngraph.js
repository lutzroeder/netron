
var onednngraph = onednngraph || {};
var json = json || require('./json');
var base = base || require('./base');

onednngraph.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj && obj.version && obj.graph) {
                return 'onednngraph.json';
            }
        }
        return undefined;
    }

    open(context, match) {
        return onednngraph.Metadata.open(context).then((metadata) => {
            const createModel = (metadata, symbol, version) => {
                return new onednngraph.Model(metadata, symbol, version);
            };
            switch (match) {
                case 'onednngraph.json': {
                    let symbol = null;
                    try {
                        symbol = context.open('json');
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onednngraph.Error("Failed to load symbol entry (" + message.replace(/\.$/, '') + ').');
                    }
                    const version = symbol && symbol.version ? symbol.version : null;
                    return createModel(metadata, symbol, version);
                }
                default: {
                    throw new onednngraph.Error("Unsupported oneDNN Graph format '" + match + "'.");
                }
            }
        });
    }
};

onednngraph.Model = class {

    constructor(metadata, symbol, version) {
        if (!symbol) {
            throw new onednngraph.Error('JSON symbol data not available.');
        }
        if (symbol) {
            if (!Object.prototype.hasOwnProperty.call(symbol, 'graph')) {
                throw new onednngraph.Error('JSON file does not contain an oneDNN Graph \'graph\' property.');
            }
            if (!Object.prototype.hasOwnProperty.call(symbol, 'version')) {
                throw new onednngraph.Error('JSON file does not contain an oneDNN Graph \'version\' property.');
            }
        }
        this._format = 'oneDNN Graph';
        this._version = 'oneDNN' + (version ? ' v' + version : '');
        this._license = 'Apache License 2.0';
        this._graphs = [new onednngraph.Graph(metadata, symbol)];
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get name() {
        return this._name;
    }

    get version() {
        return this._version;
    }

    get description() {
        return this._description;
    }

    get author() {
        return this._author;
    }

    get license() {
        return this._license;
    }

    get runtime() {
        return this._runtime;
    }

    get graphs() {
        return this._graphs;
    }
};

onednngraph.Graph = class {

    constructor(metadata, symbol) {
        this._metadata = metadata;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        if (symbol) {
            this._type = 'engine: ' + symbol.engine_kind + '; fpmath: ' + symbol.fpmath_mode + ';';
            const nodes = symbol.graph;

            const args = new Map();
            const arg = (name, type, lt) => {
                if (!args.has(name)) {
                    args.set(name, new onednngraph.Argument(name, type, null, lt));
                }
                return args.get(name);
            };
            for (const node of nodes) {
                this._nodes.push(new onednngraph.Node(this._metadata, node, arg, symbol.engine_kind));
            }
        }
    }

    get name() {
        return '';
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

onednngraph.Node = class {

    constructor(metadata, node, arg, device) {
        const type = node.kind;
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._type = { name: type };
        this._device = device;
        this._description = 'oneDNN Graph OP with unique id: ' + node.id;

        const attrs = node.attrs;
        if (attrs) {
            for (const attributeName of Object.keys(attrs)) {
                this._attributes.push(new onednngraph.Attribute(metadata, type, attributeName, attrs[attributeName].type, attrs[attributeName].value));
                if (attributeName == 'backend') {
                    if (attrs[attributeName].value == 'dnnl_backend') {
                        this._type.category = "onednngraph-backend0";
                    }
                    else {
                        // leave color of fake_backend node by default
                    }
                }
            }
        }

        const inputs = node.inputs || [];
        let inputIndex = 0;
        for (const input of inputs) {
            if (inputIndex < inputs.length) {
                const inputArguments = [];
                const lt = new onednngraph.LogicalTensor(input.id, input.dtype, input.shape, input.stride, input.layout_type, input.property_type);
                const shape = new onednngraph.TensorShape(lt.shape);
                const type = new onednngraph.TensorType(lt.dataType, shape);
                inputArguments.push(arg(input.id.toString(), type, lt));
                const inputName = (inputs.length == 1) ? 'input' : ('input' + (inputIndex)).toString();
                this._inputs.push(new onednngraph.Parameter(inputName, inputArguments));
                inputIndex += 1;
            }
        }
        this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
            const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
            return new onednngraph.Parameter(inputName, true, [arg(input)]);
        }));

        const outputs = node.outputs || [];
        let outputIndex = 0;
        for (const output of outputs) {
            if (outputIndex < outputs.length) {
                const outputArguments = [];
                const lt = new onednngraph.LogicalTensor(output.id, output.dtype, output.shape, output.stride, output.layout_type, output.property_type);
                const shape = new onednngraph.TensorShape(lt.shape);
                const type = new onednngraph.TensorType(lt.dataType, shape);
                outputArguments.push(arg(output.id.toString(), type, lt));
                const outputName = (outputs.length == 1) ? 'output' : ('output' + (outputIndex)).toString();
                this._outputs.push(new onednngraph.Parameter(outputName, outputArguments));
                outputIndex += 1;
            }
        }
        this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
            const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
            return new onednngraph.Parameter(outputName, true, [arg(output)]);
        }));
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

    get description() {
        return this._description;
    }

    get device() {
        return this._device;
    }
};

onednngraph.Attribute = class {

    constructor(metadata, type, name, attr_type, attr_value) {
        this._name = name;
        this._type = attr_type;
        this._value = attr_value;

        let number;
        const schema = metadata.attribute(type, name);
        if (schema && schema.type) {
            switch (schema.type) {
                case 'bool':
                    switch (attr_value) {
                        case 1: this._value = true; break;
                        case 0: this._value = false; break;
                        default: throw new onednngraph.Error("Unsupported attribute boolean value '" + attr_value + "'.");
                    }
                    break;
                case 'string':
                    break;
                case 's64':
                    number = Number.parseInt(this._value, 10);
                    this._value = Number.isNaN(this._value - number) ? attr_value : number;
                    break;
                case 'f32':
                    number = Number.parseFloat(this._value);
                    this._value = Number.isNaN(this._value - number) ? attr_value : number;
                    break;
                case 's64[]':
                    if (this._value.length > 2 && this._value.startsWith('[') && this._value.endsWith(']')) {
                        let array = [];
                        const items = this._value.substring(1, this._value.length - 1).split(',')
                            .map((item) => item.trim())
                            .map((item) => item.endsWith('L') ? item.substring(0, item.length - 1) : item);
                        for (const item of items) {
                            number = Number.parseInt(item, 10);
                            if (Number.isNaN(item - number)) {
                                array = null;
                            }
                            else if (array != null) {
                                array.push(number);
                            }
                        }
                        if (array != null) {
                            this._value = array;
                        }
                    }
                    break;
                case 'f32[]':
                    if (this._value.length > 2 && this._value.startsWith('[') && this._value.endsWith(']')) {
                        let array = [];
                        const items = this._value.substring(1, this._value.length - 1).split(',')
                            .map((item) => item.trim())
                            .map((item) => item.endsWith('L') ? item.substring(0, item.length - 1) : item);
                        for (const item of items) {
                            number = Number.parseFloat(item);
                            if (Number.isNaN(item - number)) {
                                array = null;
                            }
                            else if (array != null) {
                                array.push(number);
                            }
                        }
                        if (array != null) {
                            this._value = array;
                        }
                    }
                    break;
                default:
                    throw new onednngraph.Error("Unsupported attribute type '" + schema.type + "'.");
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

onednngraph.Parameter = class {

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

onednngraph.Argument = class {

    constructor(name, type, initializer, lt) {
        if (typeof name !== 'string') {
            throw new onednngraph.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._lt = lt;
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

    get description() {
        const stride = this._lt.stride ? ('[' + this._lt.stride.map((stride) => stride ? stride.toString() : '?').join(',') + ']') : '';
        const layout_type = this._lt.layout_type;
        const property_type = this._lt.property_type;
        return 'stride = ' + stride + ', layout_type = ' + layout_type + ', property_type = ' + property_type;
    }
};

onednngraph.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
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

onednngraph.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

onednngraph.LogicalTensor = class {

    constructor(id, dataType, shape, stride, layout_type, property_type) {
        this._id = id;
        this._shape = shape;
        this._dataType = dataType;
        this._stride = stride;
        this._layout_type = layout_type;
        this._property_type = property_type;
    }

    get id() {
        return this._id;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get stride() {
        return this._stride;
    }

    get layout_type() {
        return this._layout_type;
    }

    get property_type() {
        return this._property_type;
    }
};

onednngraph.Metadata = class {

    static open(context) {
        if (onednngraph.Metadata._metadata) {
            return Promise.resolve(onednngraph.Metadata._metadata);
        }
        return context.request('onednngraph-metadata.json', 'utf-8', null).then((data) => {
            onednngraph.Metadata._metadata = new onednngraph.Metadata(data);
            return onednngraph.Metadata._metadata;
        }).catch(() => {
            onednngraph.Metadata._metadata = new onednngraph.Metadata(null);
            return onednngraph.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = {};
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [item.name, item]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        let map = this._attributeCache[type];
        if (!map) {
            map = {};
            const schema = this.type(type);
            if (schema && schema.attributes) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[type] = map;
        }
        return map[name] || null;
    }
};

onednngraph.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading oneDNN Graph model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = onednngraph.ModelFactory;
}
