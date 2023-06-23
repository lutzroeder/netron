
var onednn = onednn || {};
var json = json || require('./json');
var base = base || require('./base');

onednn.ModelFactory = class {

    match(context) {
        const obj = context.open('json');
        if (obj && obj.version && obj.engine_kind && obj.fpmath_mode && obj.graph) {
            return obj;
        }
        return null;
    }

    async open(context, match) {
        const metadata = await context.metadata('onednn-metadata.json');
        return new onednn.Model(metadata, match);
    }
};

onednn.Model = class {

    constructor(metadata, symbol) {
        const version = symbol.version;
        this._format = 'oneDNN Graph' + (version ? ' v' + version : '');
        this._runtime = symbol.engine_kind + ' ' + symbol.fpmath_mode;
        this._graphs = [ new onednn.Graph(metadata, symbol) ];
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
        const initializers = [];
        for (const node of symbol.graph) {
            if (node.kind == 'Wildcard' && node.inputs.length == 0) {
                for (const output of node.outputs) {
                    initializers.push(output.id);
                }
            }
        }
        for (const node of symbol.graph) {
            if (!(node.kind == 'Wildcard' && node.inputs.length == 0)) {
                this._nodes.push(new onednn.Node(this._metadata, node, symbol.engine_kind, initializers));
            }
        }

        const values = [];
        for (const node of symbol.graph) {
            for (const input of node.inputs) {
                values.push(input);
            }
            for (const output of node.outputs) {
                values.push(output);
            }
        }
        let inputIndex = 0;
        const inputs = symbol.input_ports || [];
        for (const input_id of inputs) {
            const input = values.find((value) => value.id == input_id);
            const shape = !input.shape || (input.shape.length === 1 && input.shape[0] === -1) ? null : new onednn.TensorShape(input.shape);
            const type = new onednn.TensorType(input.dtype, shape);
            const inputName = (inputs.length == 1) ? 'input' : ('input' + (inputIndex)).toString();
            this._inputs.push(new onednn.Argument(inputName, [
                new onednn.Value(input.id.toString(), type, initializers.includes(input.id) ? new onednn.Tensor(type, input.property_type) : null)
            ]));
            inputIndex += 1;
        }
        let outputIndex = 0;
        const outputs = symbol.output_ports || [];
        for (const output_id of outputs) {
            const output = values.find((value) => value.id == output_id);
            const shape = !output.shape || (output.shape.length === 1 && output.shape[0] === -1) ? null : new onednn.TensorShape(output.shape);
            const type = new onednn.TensorType(output.dtype, shape);
            const outputName = (outputs.length == 1) ? 'output' : ('output' + (outputIndex)).toString();
            this._outputs.push(new onednn.Argument(outputName, [new onednn.Value(output.id.toString(), type)]));
            outputIndex += 1;
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

    constructor(metadata, node, device, initializers) {
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._type = metadata.type(node.kind) || { name: node.kind };
        this._device = device;
        this._location = node.id;
        const attrs = node.attrs;
        if (attrs) {
            for (const entry of Object.entries(attrs)) {
                const name = entry[0];
                const value = entry[1];
                this._attributes.push(new onednn.Attribute(name, value.type, value.value));
            }
        }
        const inputs = node.inputs || [];
        let inputIndex = 0;
        for (const input of inputs) {
            const shape = !input.shape || (input.shape.length === 1 && input.shape[0] === -1) ? null : new onednn.TensorShape(input.shape);
            const type = new onednn.TensorType(input.dtype, shape);
            let inputName = (inputs.length == 1) ? 'input' : ('input' + (inputIndex)).toString();
            if (this._type && this._type.inputs && this._type.inputs.length > 0) {
                inputName = this._type.inputs[inputIndex].name;
            }
            this._inputs.push(new onednn.Argument(inputName, [
                new onednn.Value(input.id.toString(), type, initializers.includes(input.id) ? new onednn.Tensor(type, input.property_type) : null)
            ]));
            inputIndex += 1;
        }
        const outputs = node.outputs || [];
        let outputIndex = 0;
        for (const output of outputs) {
            const shape = !output.shape || (output.shape.length === 1 && output.shape[0] === -1) ? null : new onednn.TensorShape(output.shape);
            const type = new onednn.TensorType(output.dtype, shape);
            let outputName = (outputs.length == 1) ? 'output' : ('output' + (outputIndex)).toString();
            if (this._type && this._type.outputs && this._type.outputs.length > 0) {
                outputName = this._type.outputs[outputIndex].name;
            }
            this._outputs.push(new onednn.Argument(outputName, [new onednn.Value(output.id.toString(), type)]));
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

    constructor(name, type, value) {
        this._name = name;
        this._value = value;
        let number;
        switch (type) {
            case 'bool':
                this._type = 'boolean';
                switch (value) {
                    case 1: this._value = true; break;
                    case 0: this._value = false; break;
                    default: throw new onednn.Error("Unsupported attribute boolean value '" + value + "'.");
                }
                break;
            case 's64':
                this._type = 'int64';
                number = Number.parseInt(this._value, 10);
                this._value = Number.isNaN(this._value - number) ? value : number;
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
                        } else if (array != null) {
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
                this._value = Number.isNaN(this._value - number) ? value : number;
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
                        } else if (array != null) {
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
                throw new onednn.Error("Unsupported attribute array data type '" + type + "'.");
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

onednn.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get value() {
        return this._value;
    }
};

onednn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new onednn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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
        switch (dataType) {
            case 'f16': this._dataType = 'float16'; break;
            case 'f32': this._dataType = 'float32'; break;
            case 's8': this._dataType = 'int8'; break;
            case 's32': this._dataType = 'int32'; break;
            case 'u8': this._dataType = 'uint8'; break;
            case 'bf16': this._dataType = 'bfloat16'; break;
            case 'boolean': this._dataType = 'boolean'; break;
            case 'undef': this._dataType = '?'; break;
            default: throw new onednn.Error("Unsupported tensor data type '" + dataType.toString() + "'.");
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
        return this._dataType + (this._shape ? this._shape.toString() : '[?]');
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
