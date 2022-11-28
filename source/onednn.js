
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
        this._format = 'oneDNN Graph' + (version ? ' v' + version : '');
        this._runtime = symbol.engine_kind + ' ' + symbol.fpmath_mode;
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
            const fold_list = [];
            for (const node of nodes) {
                if (node.kind == 'Wildcard' && node.inputs.length == 0) {
                    for (const output of node.outputs) {
                        fold_list.push(output.id);
                    }
                }
            }
            for (const node of nodes) {
                if (!(node.kind == 'Wildcard' && node.inputs.length == 0)) {
                    this._nodes.push(new onednn.Node(this._metadata, node, symbol.engine_kind, fold_list));
                }
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

    constructor(metadata, node, device, fold_list) {
        const type = node.kind;
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._type = metadata.type(type) || { name: type };
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

        const schema = metadata.type(node.kind);
        const inputs = node.inputs || [];
        let inputIndex = 0;
        for (const input of inputs) {
            const shape = new onednn.TensorShape(input.shape);
            const type = new onednn.TensorType(input.dtype, shape);
            let inputName = (inputs.length == 1) ? 'input' : ('input' + (inputIndex)).toString();
            if (schema && schema.inputs && schema.inputs.length > 0) {
                const inputSchema = schema.inputs.slice();
                inputName = inputSchema[inputIndex].name;
            }
            if (fold_list.includes(input.id)) {
                this._inputs.push(new onednn.Parameter(inputName, [new onednn.Argument(input.id.toString(), type, new onednn.Tensor(type, input.property_type))]));
            }
            else {
                this._inputs.push(new onednn.Parameter(inputName, [new onednn.Argument(input.id.toString(), type)]));
            }
            inputIndex += 1;
        }

        const outputs = node.outputs || [];
        let outputIndex = 0;
        for (const output of outputs) {
            const shape = new onednn.TensorShape(output.shape);
            const type = new onednn.TensorType(output.dtype, shape, output.stride);
            let outputName = (outputs.length == 1) ? 'output' : ('output' + (outputIndex)).toString();
            if (schema && schema.outputs && schema.outputs.length > 0) {
                const outputSchema = schema.outputs.slice();
                outputName = outputSchema[outputIndex].name;
            }
            this._outputs.push(new onednn.Parameter(outputName, [new onednn.Argument(output.id.toString(), type)]));
            outputIndex += 1;
        }
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
        let number;
        switch (attr_type) {
            case 'bool':
                this._type = 'boolean';
                switch (attr_value) {
                    case 1: this._value = true; break;
                    case 0: this._value = false; break;
                    default: throw new onednn.Error("Unsupported attribute boolean value '" + attr_value + "'.");
                }
                break;
            case 's64':
                this._type = 'int64';
                number = Number.parseInt(this._value, 10);
                this._value = Number.isNaN(this._value - number) ? attr_value : number;
                break;
            case 's64[]':
                this._type = 'int64[]';
                if (this._value.length > 2 && this._value.toString().startsWith('[') && this._value.toString().endsWith(']')) {
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
            case 'f32':
                this._type = 'float32';
                number = Number.parseFloat(this._value);
                this._value = Number.isNaN(this._value - number) ? attr_value : number;
                break;
            case 'f32[]':
                this._type = 'float32[]';
                if (this._value.length > 2 && this._value.toString().startsWith('[') && this._value.toString().endsWith(']')) {
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
            case 'string':
                this._type = 'string';
                break;
            default: {
                throw new onednn.Error("Unsupported attribute array data type '" + attr_type + "'.");
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

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new onednn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

onednn.TensorType = class {

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

onednn.Tensor = class {

    constructor(type, property_type) {
        this._type = type;
        this._category = property_type;
    }

    get type() {
        return this._type;
    }

    get category() {
        return this._category;
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
