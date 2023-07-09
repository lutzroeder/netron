
var mxnet = {};
var json = require('./json');
var base = require('./base');

mxnet.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj && obj.nodes && obj.arg_nodes && obj.heads) {
                return 'mxnet.json';
            }
        }
        if (extension === 'params') {
            const stream = context.stream;
            const signature = [ 0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ];
            if (stream && stream.length > signature.length && stream.peek(signature.length).every((value, index) => value == signature[index])) {
                return 'mxnet.params';
            }
        }
        return undefined;
    }

    async open(context, target) {
        const metadata = await context.metadata('mxnet-metadata.json');
        const basename = (base, identifier, extension, suffix, append) => {
            if (!base) {
                if (identifier.toLowerCase().endsWith(extension)) {
                    const items = identifier.substring(0, identifier.length - extension.length).split('-');
                    if (items.length >= 2) {
                        const token = items.pop();
                        if ((suffix && token === suffix) || /[a-zA-Z0-9]*/.exec(token)) {
                            return items.join('-') + append;
                        }
                    }
                }
            }
            return base;
        };
        const convertVersion = (value) => {
            if (Array.isArray(value)) {
                if (value.length === 2 && value[0] === 'int') {
                    const major = Math.floor(value[1] / 10000) % 100;
                    const minor = Math.floor(value[1] / 100) % 100;
                    const patch = Math.floor(value[1]) % 100;
                    return [ major.toString(), minor.toString(), patch.toString() ].join('.');
                }
            }
            return null;
        };
        const requestManifest = async () => {
            const parse = async (stream) => {
                try {
                    const manifest = {};
                    if (stream) {
                        const reader = json.TextReader.open(stream);
                        const obj = reader.read();
                        if (obj.Model) {
                            const modelFormat = obj.Model['Model-Format'];
                            if (modelFormat && modelFormat !== 'MXNet-Symbolic') {
                                throw new mxnet.Error('Model format \'' + modelFormat + '\' not supported.');
                            }
                            manifest.format = 'MXNet Model Server';
                            if (obj['Model-Archive-Version']) {
                                manifest.format += ' v' + obj['Model-Archive-Version'].toString();
                            }
                            if (!obj.Model.Symbol) {
                                throw new mxnet.Error('Manifest does not contain symbol entry.');
                            }
                            manifest.symbol = obj.Model.Symbol;
                            if (obj.Model.Signature) {
                                manifest.signature = obj.Model.Signature;
                            }
                            if (obj.Model.Parameters) {
                                manifest.params = obj.Model.Parameters;
                            }
                            if (obj.Model['Model-Name']) {
                                manifest.name = obj.Model['Model-Name'];
                            }
                            if (obj.Model.Description && manifest.name !== obj.Model.Description) {
                                manifest.description = obj.Model.Description;
                            }
                        } else if (obj.model) {
                            manifest.format = 'MXNet Model Archive';
                            if (obj.specificationVersion) {
                                manifest.format += ' v' + obj.specificationVersion.toString();
                            }
                            if (obj.model.modelName) {
                                manifest.symbol = obj.model.modelName + '-symbol.json';
                            }
                            if (obj.model.modelName) {
                                manifest.name = obj.model.modelName;
                            }
                            if (manifest.model && obj.model.modelVersion) {
                                manifest.version = obj.model.modelVersion;
                            }
                            if (manifest.model && manifest.model.modelName && manifest.name != obj.model.description) {
                                manifest.description = obj.model.description;
                            }
                        } else {
                            throw new mxnet.Error('Manifest does not contain model.');
                        }
                        if (obj.Engine && obj.Engine.MXNet) {
                            const version = convertVersion(obj.Engine.MXNet);
                            manifest.runtime = 'MXNet v' + (version ? version : obj.Engine.MXNet.toString());
                        }
                        if (obj.License) {
                            manifest.license = obj.License;
                        }
                        if (obj.runtime) {
                            manifest.runtime = obj.runtime;
                        }
                        if (obj.engine && obj.engine.engineName) {
                            const engine = obj.engine.engineVersion ? obj.engine.engineName + ' ' + obj.engine.engineVersion : obj.engine.engineName;
                            manifest.runtime = manifest.runtime ? (manifest.runtime + ' (' + engine + ')') : engine;
                        }
                        if (obj.publisher && obj.publisher.author) {
                            manifest.author = obj.publisher.author;
                            if (obj.publisher.email) {
                                manifest.author = manifest.author + ' <' + obj.publisher.email + '>';
                            }
                        }
                        if (obj.license) {
                            manifest.license = obj.license;
                        }
                        if (obj.Model && obj.Model.Signature) {
                            try {
                                const stream = await context.request(obj.Model.Signature);
                                const reader = json.TextReader.open(stream);
                                manifest.signature = reader.read();
                                return manifest;
                            } catch (error) {
                                return manifest;
                            }
                        }
                    }
                    return manifest;
                } catch (err) {
                    throw new mxnet.Error('Failed to read manifest. ' + err.message);
                }
            };
            try {
                const stream = await context.request('MANIFEST.json');
                return parse(stream);
            } catch (error) {
                try {
                    const stream = await context.request('MAR-INF/MANIFEST.json');
                    return parse(stream);
                } catch (error) {
                    return parse(null);
                }
            }
        };
        const createModel = (metadata, manifest, symbol, params) => {
            const parameters = new Map();
            if (params) {
                try {
                    for (const entry of mxnet.ndarray.load(params)) {
                        const key = entry[0];
                        const array = entry[1];
                        const name = (key.startsWith('arg:') || key.startsWith('aux:')) ? key.substring(4) : key;
                        parameters.set(name, array);
                    }
                } catch (error) {
                    // continue regardless of error
                }
            }
            if (symbol) {
                if (!manifest.format) {
                    const version = convertVersion(symbol.attrs && symbol.attrs.mxnet_version ? symbol.attrs.mxnet_version : null);
                    manifest.format = 'MXNet' + (version ? ' v' + version : '');
                }
                if (symbol.nodes && symbol.nodes.some((node) => node && node.op == 'tvm_op')) {
                    manifest.producer  = 'TVM';
                }
            }
            return new mxnet.Model(metadata, manifest, symbol, parameters);
        };
        const identifier = context.identifier;
        switch (target) {
            case 'mxnet.json': {
                let symbol = null;
                try {
                    symbol = context.open('json');
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new mxnet.Error("Failed to load symbol entry (" + message.replace(/\.$/, '') + ').');
                }
                const requestParams = async (manifest) => {
                    const file = basename(manifest.params, identifier, '.json', 'symbol', '-0000.params');
                    if (file) {
                        try {
                            const stream = await context.request(file, null);
                            const buffer = stream.peek();
                            return createModel(metadata, manifest, symbol, buffer);
                        } catch (error) {
                            return createModel(metadata, manifest, symbol, null);
                        }
                    }
                    return createModel(metadata, manifest, symbol, null);
                };
                const manifest = await requestManifest();
                return requestParams(manifest);
            }
            case 'mxnet.params': {
                const params = context.stream.peek();
                const requestSymbol = async (manifest) => {
                    const file = basename(manifest.symbol, identifier, '.params', null, '-symbol.json');
                    if (file) {
                        try {
                            const text = await context.request(file, 'utf-8');
                            const symbol = JSON.parse(text);
                            return createModel(metadata, manifest, symbol, params);
                        } catch (error) {
                            return createModel(metadata, manifest, null, params);
                        }
                    }
                    return createModel(metadata, manifest, null, params);
                };
                const manifest = await requestManifest();
                return requestSymbol(manifest);
            }
            default: {
                throw new mxnet.Error("Unsupported MXNet format '" + target + "'.");
            }
        }
    }
};

mxnet.Model = class {

    constructor(metadata, manifest, symbol, params) {
        if (!symbol && !params) {
            throw new mxnet.Error('JSON symbol data not available.');
        }
        if (symbol) {
            if (!Object.prototype.hasOwnProperty.call(symbol, 'nodes')) {
                throw new mxnet.Error('JSON file does not contain an MXNet \'nodes\' property.');
            }
            if (!Object.prototype.hasOwnProperty.call(symbol, 'arg_nodes')) {
                throw new mxnet.Error('JSON file does not contain an MXNet \'arg_nodes\' property.');
            }
            if (!Object.prototype.hasOwnProperty.call(symbol, 'heads')) {
                throw new mxnet.Error('JSON file does not contain an MXNet \'heads\' property.');
            }
        }
        this._format = manifest.format || 'MXNet';
        this._producer = manifest.producer || '';
        this._name = manifest.name || '';
        this._version = manifest.version;
        this._description = manifest.description || '';
        this._runtime = manifest.runtime || '';
        this._metadata = [];
        if (manifest.author) {
            this._metadata.push({ name: 'author', value: manifest.author });
        }
        if (manifest.license) {
            this._metadata.push({ name: 'license', value: manifest.license });
        }
        this._graphs = [ new mxnet.Graph(metadata, manifest, symbol, params) ];
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get runtime() {
        return this._runtime;
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

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

mxnet.Graph = class {

    constructor(metadata, manifest, symbol, params) {
        this._metadata = metadata;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const tensors = new Map();
        if (params) {
            for (const pair of params) {
                const key = pair[0];
                const value = pair[1];
                tensors.set(key, new mxnet.Tensor(key, new mxnet.TensorType(value.dtype, new mxnet.TensorShape(value.shape)), value.data));
            }
        }
        const values = new Map();
        const value = (name, type, tensor) => {
            if (!values.has(name)) {
                values.set(name, new mxnet.Value(name, type || null, tensor || null));
            } else if (type || (tensor && tensor !== values.get(name).initializer)) {
                throw new mxnet.Error("Duplicate value '" + name + "'.");
            }
            return values.get(name);
        };
        const updateOutput = (nodes, input) => {
            const nodeIndex = input[0];
            const node = nodes[nodeIndex];
            const outputIndex = input[1];
            if (node) {
                while (outputIndex >= node.outputs.length) {
                    node.outputs.push([ nodeIndex, node.outputs.length ]);
                }
            }
            return [ nodeIndex, outputIndex ];
        };
        if (symbol) {
            let nodes = symbol.nodes;
            const inputs = {};
            if (manifest && manifest.signature && manifest.signature.inputs) {
                for (const input of manifest.signature.inputs) {
                    inputs[input.data_name] = input;
                }
            }
            const outputs = {};
            if (manifest && manifest.signature && manifest.signature.outputs) {
                for (const output of manifest.signature.outputs) {
                    outputs[output.data_name] = output;
                }
            }
            for (const node of nodes) {
                node.outputs = [];
            }
            for (const node of nodes) {
                node.inputs = node.inputs || [];
                node.inputs = node.inputs.map((input) => updateOutput(nodes, input));
            }
            const outputCountMap = {};
            for (const node of nodes) {
                for (const output of node.outputs) {
                    outputCountMap[output] = (outputCountMap[output] || 0) + 1;
                }
            }
            const arg_nodes = new Map(symbol.arg_nodes.map((index) => [ index, index < nodes.length ? nodes[index] : null ]));
            for (let i = 0; i < symbol.heads.length; i++) {
                const head = symbol.heads[i];
                const identifier = updateOutput(nodes, head);
                const name = nodes[identifier[0]] ? nodes[identifier[0]].name : ('output' + ((i == 0) ? '' : (i + 1).toString()));
                const signature = outputs[name];
                const type = signature && signature.data_shape ? new mxnet.TensorType(-1, new mxnet.TensorShape(signature.data_shape)) : null;
                this._outputs.push(new mxnet.Argument(name, [ value('[' + identifier.join(',') + ']', type) ]));
            }
            nodes = nodes.filter((node, index) => !arg_nodes.has(index));
            const initializers = new Map();
            for (const node of nodes) {
                if (node.op == 'RNN') {
                    node.inputs = node.inputs.filter((input) => {
                        const index = input[0];
                        const arg_node = arg_nodes.get(index);
                        if (arg_node && arg_node.op == 'null' && arg_node.name && arg_node.name.endsWith('_parameters') && arg_node.attr && arg_node.attr.__init__) {
                            let attr = node.attrs || node.attr || node.param;
                            if (!attr) {
                                node.attr = {};
                                attr = node.attr;
                            }
                            attr[arg_node.name] = arg_node.attr.__init__;
                            arg_nodes.delete(index);
                            return false;
                        }
                        return true;
                    });
                }
                for (const input of node.inputs) {
                    const identifier = '[' + input.join(',') + ']';
                    if (!initializers.has(identifier)) {
                        const index = input[0];
                        const arg_node = arg_nodes.get(index);
                        if (arg_node && arg_node.name && (!arg_node.inputs || arg_node.inputs.length == 0) && (arg_node.outputs && arg_node.outputs.length == 1)) {
                            if (tensors.has(arg_node.name)) {
                                initializers.set(identifier, tensors.get(arg_node.name));
                                arg_nodes.delete(index);
                            } else {
                                const prefix = node.name.endsWith('_fwd') ? node.name.slice(0, -3) : node.name;
                                if (arg_node.name && (arg_node.name.startsWith(prefix + '_') || arg_node.name.startsWith(prefix + '.'))) {
                                    let dataType = -1;
                                    let shape = [];
                                    if (arg_node.attrs && arg_node.attrs.__dtype__ && arg_node.attrs.__shape__) {
                                        try {
                                            dataType = parseInt(arg_node.attrs.__dtype__);
                                            shape = JSON.parse('[' + arg_node.attrs.__shape__.replace('(', '').replace(')', '').split(' ').join('').split(',').map((dimension => dimension || '"?"')).join(',') + ']');
                                        } catch (err) {
                                            // continue regardless of error
                                        }
                                    }
                                    const type = (dataType !== -1 || shape.length > 0) ?
                                        new mxnet.TensorType(dataType, new mxnet.TensorShape(shape)) :
                                        new mxnet.TensorType(-1, new mxnet.TensorShape(null));
                                    initializers.set(identifier, new mxnet.Tensor(arg_node.name, type, null));
                                    arg_nodes.delete(index);
                                }
                            }
                        }
                    }
                }
                if (node.params) {
                    for (const param of node.params) {
                        value(param.id, null, tensors.get(param.id));
                    }
                }
            }
            for (const entry of arg_nodes) {
                const arg_node = entry[1];
                if (arg_node && (!arg_node.inputs || arg_node.inputs.length == 0) && (arg_node.outputs && arg_node.outputs.length == 1)) {
                    const identifier = '[' + arg_node.outputs[0].join(',') + ']';
                    const name = arg_node.name;
                    const signature = inputs[name];
                    const type = signature && signature.data_shape ? new mxnet.TensorType(-1, new mxnet.TensorShape(signature.data_shape)) : null;
                    const argument = new mxnet.Argument(name, [ value(identifier, type, tensors.get(identifier)) ]);
                    this._inputs.push(argument);
                }
            }
            for (const node of nodes) {
                this._nodes.push(new mxnet.Node(this._metadata, node, initializers, value));
            }
        } else if (params) {
            const blocks = new Map();
            let separator = Array.from(params.keys()).every((key) => key.indexOf('_') != -1) ? '_' : '';
            if (separator.length == 0) {
                separator = Array.from(params.keys()).every((key) => key.indexOf('.') != -1) ? '.' : '';
            }
            if (separator.length > 0) {
                for (const param of params) {
                    const key = param[0];
                    const parts = key.split(separator);
                    let argumentName = parts.pop();
                    if (key.endsWith('moving_mean') || key.endsWith('moving_var')) {
                        argumentName = [ parts.pop(), argumentName ].join(separator);
                    }
                    const nodeName = parts.join(separator);
                    if (!blocks.has(nodeName)) {
                        blocks.set(nodeName, { name: nodeName, op: 'Weights', params: [] });
                    }
                    blocks.get(nodeName).params.push({ name: argumentName, id: key });
                    value(key, null, tensors.get(key));
                }
            } else {
                throw new mxnet.Error("Unsupported key format in params.");
            }

            for (const block of blocks.values()) {
                this._nodes.push(new mxnet.Node(metadata, block, new Map(), value));
            }
        }
    }

    get name() {
        return '';
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

mxnet.Argument = class {

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

mxnet.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new mxnet.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        if (this._initializer) {
            return this._initializer.name;
        }
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

mxnet.Node = class {

    constructor(metadata, node, initializers, value) {
        let type = node.op;
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        const attrs = node.attrs || node.attr || node.param;
        if (attrs) {
            if (type == 'tvm_op' && attrs.func_name) {
                type = attrs.func_name;
            }
            for (const entry of Object.entries(attrs)) {
                if (type != 'tvm_op' && entry[0] != 'func_name') {
                    const attribute = new mxnet.Attribute(metadata, type, entry[0], entry[1]);
                    this._attributes.push(attribute);
                }
            }
        }
        this._type = metadata.type(type) || { name: type };
        if (node.inputs) {
            const inputs = node.inputs;
            let inputIndex = 0;
            if (this._type && this._type.inputs) {
                for (const inputDef of this._type.inputs) {
                    if (inputIndex < inputs.length || inputDef.option != 'optional') {
                        const count = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                        const values = [];
                        for (const input of inputs.slice(inputIndex, inputIndex + count)) {
                            const identifier = '[' + input.join(',') + ']';
                            if (identifier !== '' || inputDef.option != 'optional') {
                                values.push(value(identifier, inputDef.type, initializers.get(identifier)));
                            }
                        }
                        const argument = new mxnet.Argument(inputDef.name, values);
                        this._inputs.push(argument);
                        inputIndex += count;
                    }
                }
            }
            if (inputIndex < inputs.length) {
                this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                    const identifier = '[' + input.join(',') + ']';
                    return new mxnet.Argument((inputIndex + index).toString(), [
                        value(identifier, null, initializers.get(identifier))
                    ]);
                }));
            }
        }
        if (node.outputs) {
            const outputs = node.outputs;
            let outputIndex = 0;
            if (this._type && this._type.outputs) {
                for (const outputDef of this._type.outputs) {
                    if (outputIndex < outputs.length || outputDef.option != 'optional') {
                        const values = [];
                        const count = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                        for (const output of outputs.slice(outputIndex, outputIndex + count)) {
                            values.push(value('[' + output.join(',') + ']'));
                        }
                        const argument = new mxnet.Argument(outputDef.name, values);
                        this._outputs.push(argument);
                        outputIndex += count;
                    }
                }
            }
            if (outputIndex < outputs.length) {
                this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                    const name = (outputIndex + index).toString();
                    return new mxnet.Argument(name, [ value('[' + output.join(',') + ']') ]);
                }));
            }
        }
        if (node.params) {
            for (const param of node.params) {
                const argument = new mxnet.Argument(param.name, [ value(param.id) ]);
                this._inputs.push(argument);
            }
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
};

mxnet.Attribute = class {

    constructor(metadata, type, name, value) {
        this._name = name;
        this._value = value;

        let number;
        metadata = metadata.attribute(type, name);
        if (metadata && metadata.type) {
            switch (metadata.type) {
                case 'boolean':
                    switch (value) {
                        case 0:
                        case '0':
                        case 'False':
                            this._value = false;
                            break;
                        case 1:
                        case '1':
                        case 'True':
                            this._value = true;
                            break;
                        default:
                            throw new mxnet.Error("Unsupported attribute boolean value '" + value + "'.");
                    }
                    break;
                case 'int32':
                    number = Number.parseInt(this._value, 10);
                    this._value = Number.isNaN(this._value - number) ? value : number;
                    break;
                case 'float32':
                case 'float64':
                    number = Number.parseFloat(this._value);
                    this._value = Number.isNaN(this._value - number) ? value : number;
                    break;
                case 'int32[]':
                    if (this._value.length > 2 && this._value.startsWith('(') && this._value.endsWith(')')) {
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
                default:
                    throw new mxnet.Error("Unsupported attribute type '" + metadata.type + "'.");
            }
        }
        if (metadata) {
            if (metadata.visible === false) {
                this._visible = false;
            } else if (metadata.default !== undefined) {
                let defaultValue = metadata.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                } else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < this._value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]);
                        }
                    }
                    if (this._value.every((item, index) => item == defaultValue[index])) {
                        this._visible = false;
                    }
                }
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

mxnet.Tensor = class {

    constructor(name, type, data) {
        this._name = name;
        this._type = type;
        this._data = data;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get layout() {
        return '<';
    }

    get values() {
        return this._data;
    }
};

mxnet.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case 0: this._dataType = 'float32'; break;
            case 1: this._dataType = 'float64'; break;
            case 2: this._dataType = 'float16'; break;
            case 3: this._dataType = 'uint8'; break;
            case 4: this._dataType = 'int32'; break;
            case 5: this._dataType = 'int8'; break;
            case 6: this._dataType = 'int64'; break;
            case -1: this._dataType = '?'; break;
            default: throw new mxnet.Error("Unsupported type '" + dataType + "'.");
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
        return this._dataType + this._shape.toString();
    }
};

mxnet.TensorShape = class {

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

mxnet.ndarray = class {

    static load(buffer) {
        // NDArray::Load(dmlc::Stream* fi, std::vector<NDArray>* data, std::vector<std::string>* keys)
        const map = new Map();
        const reader = new mxnet.BinaryReader(buffer);
        if (reader.uint64() !== 0x112) { // kMXAPINDArrayListMagic
            throw new mxnet.Error('Invalid signature.');
        }
        if (reader.uint64() !== 0) {
            throw new mxnet.Error('Invalid reserved block.');
        }
        const data = new Array(reader.uint64());
        for (let i = 0; i < data.length; i++) {
            data[i] = new mxnet.ndarray.NDArray(reader);
        }
        const decoder = new TextDecoder('ascii');
        const names = new Array(reader.uint64());
        for (let i = 0; i < names.length; i++) {
            names[i] = decoder.decode(reader.read(reader.uint64()));
        }
        if (names.length != data.length) {
            throw new mxnet.Error('Label count mismatch.');
        }
        for (let i = 0; i < names.length; i++) {
            map.set(names[i], data[i]);
        }
        return map;
    }
};

mxnet.ndarray.NDArray = class {

    constructor(reader) {
        mxnet.ndarray.NDArray._dataTypeSizeTable = [ 4, 8, 2, 1, 4, 1, 8 ];
        switch (reader.uint32()) {
            case 0xf993faca: { // NDARRAY_V3_MAGIC
                throw new mxnet.Array('mxnet.ndarray.NDArray v3 not supported.');
            }
            case 0xf993fac9: { // NDARRAY_V2_MAGIC
                const stype = reader.uint32();
                let num_aux_data = 0;
                switch (stype) {
                    case 0: num_aux_data = 0; break; // kDefaultStorage
                    case 1: num_aux_data = 1; break; // kRowSparseStorage
                    case 2: num_aux_data = 2; break; // kCSRStorage
                    default: throw mxnet.Error("Unsupported NDArray type '" + stype + "'.");
                }
                this.sshape = null;
                if (num_aux_data > 0) {
                    this.sshape = reader.uint64s();
                }
                this.shape = reader.uint64s();
                if (this.shape.length !== 0) {
                    this.context = new mxnet.context.Context(reader);
                    this.dtype = reader.uint32();
                    if (num_aux_data > 0) {
                        throw new mxnet.Error('Not implemented.');
                    }
                    const dataTypeSize = (this.dtype < mxnet.ndarray.NDArray._dataTypeSizeTable.length) ? mxnet.ndarray.NDArray._dataTypeSizeTable[this.dtype] : 0;
                    const size = dataTypeSize * this.size;
                    this.data = reader.read(size);
                }
                break;
            }
            case 0xf993fac8: { // NDARRAY_V1_MAGIC
                this.shape = reader.uint64s();
                if (this.shape.length !== 0) {
                    this.context = new mxnet.context.Context(reader);
                    this.dtype = reader.uint32();
                    const itemsize = (this.dtype < mxnet.ndarray.NDArray._dataTypeSizeTable.length) ? mxnet.ndarray.NDArray._dataTypeSizeTable[this.dtype] : 0;
                    const size = itemsize * this.size;
                    this.data = reader.read(size);
                }
                break;
            }
            default: {
                reader.skip(-4);
                this.shape = reader.uint32s();
                this.context = new mxnet.context.Context(reader);
                this.dtype = reader.uint32();
                const itemsize = (this.dtype < mxnet.ndarray.NDArray._dataTypeSizeTable.length) ? mxnet.ndarray.NDArray._dataTypeSizeTable[this.dtype] : 0;
                const size = itemsize * this.size;
                this.data = reader.read(size);
                break;
            }
        }
    }

    get size() {
        return this.shape.reduce((a, b) => a * b, 1);
    }
};

mxnet.BinaryReader = class extends base.BinaryReader {

    uint32s() {
        const count = this.uint32();
        const array = new Array(count);
        for (let i = 0; i < array.length; i++) {
            array[i] = this.uint32();
        }
        return array;
    }

    uint64s() {
        const count = this.uint32();
        const array = new Array(count);
        for (let i = 0; i < array.length; i++) {
            array[i] = this.uint64();
        }
        return array;
    }
};

mxnet.context = {};

mxnet.context.Context = class {

    constructor(reader) {
        this._deviceType = reader.uint32();
        this._deviceId = reader.uint32();
    }
};

mxnet.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MXNet model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mxnet.ModelFactory;
}