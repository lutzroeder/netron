
var onednn = onednn || {};
var json = json || require('./json');
var base = base || require('./base');

onednn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.open('json');
            // note: onednn graph should contains version, engine_kind, fpmath_mode along with graph body
            if (obj && obj.version && obj.engine_kind && obj.fpmath_mode && obj.graph) {
                this._runtime = obj.engine_kind + ' ' + obj.fpmath_mode;
                return obj;
            }
        }
        return undefined;
    }

    open(context, match) {
        return context.metadata('onednn-metadata.json').then((metadata) => {
            if (match) {
                const version = match && match.version ? match.version : null;
                return new onednn.Model(metadata, match, version);
            }
            throw new onednn.Error("Unsupported oneDNN Graph format '" + match + "'.");
        });
    }
};

onednn.Model = class {

    constructor(metadata, symbol, version) {
        if (!symbol) {
            throw new onednn.Error('JSON symbol data not available.');
        }
        if (symbol.graph) {
            if (!Object.prototype.hasOwnProperty.call(symbol, 'version')) {
                throw new onednn.Error('JSON file does not contain an oneDNN Graph \'version\' property.');
            }
        }
        this._format = 'oneDNN Graph' + (version ? ' v' + version : '');
        this._runtime = 'engine: ' + symbol.engine_kind + '; fpmath: ' + symbol.fpmath_mode + ';';
        this._graphs = [new onednn.Graph(metadata, symbol)];
    }

    get format() {
        return this._format;
    }

    get version() {
        return this._version;
    }

    get runtime() {
        return this._runtime;
    }

    get graphs() {
        return this._graphs;
    }
};

onednn.Graph = class {

    constructor(metadata, symbol) {
        this._metadata = metadata;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        if (symbol) {
            const nodes = symbol.graph;

            const args = new Map();
            const arg = (name, type, lt) => {
                if (!args.has(name)) {
                    args.set(name, new onednn.Argument(name, type, null, lt));
                }
                return args.get(name);
            };
            for (const node of nodes) {
                this._nodes.push(new onednn.Node(this._metadata, node, arg, symbol.engine_kind));
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

onednn.Node = class {

    constructor(metadata, node, arg, device) {
        const type = node.kind;
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._type = { name: type };
        this._device = device;
        this._location = node.id;

        const attrs = node.attrs;
        if (attrs) {
            for (const pair of Object.entries(attrs)) {
                const name = pair[0];
                const value = pair[1];
                this._attributes.push(new onednn.Attribute(metadata, type, name, value.type, value.value));
            }
        }

        const inputs = node.inputs || [];
        let inputIndex = 0;
        for (const input of inputs) {
            if (inputIndex < inputs.length) {
                const inputArguments = [];
                const shape = new onednn.TensorShape(input.shape);
                const type = new onednn.TensorType(input.dtype, shape);
                inputArguments.push(arg(input.id.toString(), type, input.layout_type, input.property_type));
                const inputName = (inputs.length == 1) ? 'input' : ('input' + (inputIndex)).toString();
                this._inputs.push(new onednn.Parameter(inputName, inputArguments));
                inputIndex += 1;
            }
        }
        this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
            const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
            return new onednn.Parameter(inputName, true, [arg(input)]);
        }));

        const outputs = node.outputs || [];
        let outputIndex = 0;
        for (const output of outputs) {
            if (outputIndex < outputs.length) {
                const outputArguments = [];
                const shape = new onednn.TensorShape(output.shape);
                const type = new onednn.TensorType(output.dtype, shape, output.stride);
                outputArguments.push(arg(output.id.toString(), type, output.layout_type, output.property_type));
                const outputName = (outputs.length == 1) ? 'output' : ('output' + (outputIndex)).toString();
                this._outputs.push(new onednn.Parameter(outputName, outputArguments));
                outputIndex += 1;
            }
        }
        this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
            const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
            return new onednn.Parameter(outputName, true, [arg(output)]);
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

    get location() {
        return this._location;
    }

    get device() {
        return this._device;
    }
};

onednn.Attribute = class {

    constructor(metadata, type, name, attr_type, attr_value) {
        this._name = name;
        this._value = attr_value;
        switch (attr_type) {
            case 'bool':
                this._type = 'boolean';
                break;
            case 's64':
                this._type = 'int64';
                break;
            case 's64[]':
                this._type = 'int64[]';
                break;
            case 'f32':
                this._type = 'float32';
                break;
            case 'f32[]':
                this._type = 'float32[]';
                break;
            case 'string':
                this._type = 'string';
                break;
            default: {
                throw new onednn.Error("Unsupported attribute array data type '" + attr_type + "'.");
            }
        }

        let number;
        const schema = metadata.attribute(type, name);
        if (schema && schema.type) {
            switch (schema.type) {
                case 'bool':
                    switch (attr_value) {
                        case 1: this._value = true; break;
                        case 0: this._value = false; break;
                        default: throw new onednn.Error("Unsupported attribute boolean value '" + attr_value + "'.");
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
                    throw new onednn.Error("Unsupported attribute type '" + schema.type + "'.");
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

onednn.Parameter = class {

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

onednn.Argument = class {

    constructor(name, type, initializer, layout_type, property_type) {
        if (typeof name !== 'string') {
            throw new onednn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._layout_type = layout_type;
        this._property_type = property_type;
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
        return 'layout_type = ' + this._layout_type + ', property_type = ' + this._property_type;
    }
};

onednn.TensorType = class {

    constructor(dataType, shape, stride) {
        this._stride = stride ? ('[' + stride.map((stride) => stride ? stride.toString() : '?').join(',') + ']') : '';
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

onednn.TensorShape = class {

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

onednn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading oneDNN Graph model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = onednn.ModelFactory;
}
