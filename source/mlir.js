
var mlir = {};
var text = require('./text');

mlir.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream) {
            const reader = text.Reader.open(stream, 2048);
            for (;;) {
                const line = reader.read();
                if (line === undefined) {
                    break;
                }
                if (line.indexOf('module ') !== -1) {
                    return 'mlir';
                }
            }
        }
        return null;
    }

    open(context) {
        const stream = context.stream;
        const decoder = text.Decoder.open(stream);
        const parser = new mlir.Parser(decoder);
        const obj = parser.read();
        const model = new mlir.Model(obj);
        return Promise.resolve(model);
    }
};

mlir.Model = class {

    constructor(/* obj */) {
        this._format = 'MLIR';
        this._graphs = [];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

mlir.Tokenizer = class {

    constructor(decoder) {
        this._decoder = decoder;
    }
};

mlir.Parser = class {

    constructor(decoder) {
        this._tokenizer = new mlir.Tokenizer(decoder);
    }

    read() {
        throw new mlir.Error('MLIR support is not implemented.');
    }
};

mlir.Graph = class {

    constructor() {
        this._inputs = [];  // [mlir.Parameter]
        this._outputs = []; // [mlir.Parameter]
        this._nodes = [];   // [mlir.Node]

        // TODO
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

mlir.Parameter = class {

    constructor(name, args) {
        this._name = name;      // string
        this._arguments = args; // [mlir.Argument]
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

mlir.Argument = class {
    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new coreml.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;          // string
        this._type = type || null;  // mlir.TensorType
        this._description = description || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    set name(value) {
        this._name = value;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    set type(value) {
        this._type = value;
    }

    get description() {
        return this._description;
    }

    set description(value) {
        this._description = value;
    }

    get quantization() {
        if (this._initializer) {
            return this._initializer.quantization;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
}

mlir.Node = class {
    constructor(node) {
        this._name = 'Node_name';   // string
        this._type = 'Node_type';   // string
        this._inputs = [];          // [mlir.Parameter]
        this._outputs = [];         // [mlir.Parameter]
        this._attributes = [];      // [mlir.Attributes]

        // TODO
    }

    get name() {
        return this._name;
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

    get attributes() {
        return this._attributes;
    }
}

mlir.Attributes = class {
    constructor() {
        this._name = 'Attributes_name';
        this._type = 'Attributes_type';
        this._value = 'Attributes_value';
        this._visible = true;

        // TODO
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
}

mlir.Tensor = class {
    
    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        // TODO
        return null;
    }
    
    get layout() {
        switch (this._type.dataType) {
            case 'float32': return '|';
            default: return '<';
        }
    }

    get values() {
        return this._data;
    }
}

mlir.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape || new mlir.TensorShape([]);
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
}

mlir.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};


mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mlir.ModelFactory;
}
