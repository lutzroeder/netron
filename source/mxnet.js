
import * as json from './json.js';

const mxnet = {};

mxnet.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.peek('json');
            if (obj && Array.isArray(obj.nodes) && Array.isArray(obj.arg_nodes) && Array.isArray(obj.heads) &&
                !obj.nodes.some((node) => node && node.op === 'tvm_op')) {
                context.type = 'mxnet.json';
                context.target = obj;
                return;
            }
        }
        const stream = context.stream;
        const signature = [0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        if (stream && stream.length > signature.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            context.type = 'mxnet.params';
        }
    }

    filter(context, type) {
        return context.type !== 'mxnet.json' || type !== 'mxnet.params';
    }

    async open(context) {
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
                    return [major.toString(), minor.toString(), patch.toString()].join('.');
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
                                throw new mxnet.Error(`Model format '${modelFormat}' not supported.`);
                            }
                            manifest.format = 'MXNet Model Server';
                            if (obj['Model-Archive-Version']) {
                                manifest.format += ` v${obj['Model-Archive-Version']}`;
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
                                manifest.format += ` v${obj.specificationVersion}`;
                            }
                            if (obj.model.modelName) {
                                manifest.symbol = `${obj.model.modelName}-symbol.json`;
                            }
                            if (obj.model.modelName) {
                                manifest.name = obj.model.modelName;
                            }
                            if (manifest.model && obj.model.modelVersion) {
                                manifest.version = obj.model.modelVersion;
                            }
                            if (manifest.model && manifest.model.modelName && manifest.name !== obj.model.description) {
                                manifest.description = obj.model.description;
                            }
                        } else {
                            throw new mxnet.Error('Manifest does not contain model.');
                        }
                        if (obj.Engine && obj.Engine.MXNet) {
                            const version = convertVersion(obj.Engine.MXNet);
                            manifest.runtime = `MXNet v${version ? version : obj.Engine.MXNet}`;
                        }
                        if (obj.License) {
                            manifest.license = obj.License;
                        }
                        if (obj.runtime) {
                            manifest.runtime = obj.runtime;
                        }
                        if (obj.engine && obj.engine.engineName) {
                            const engine = obj.engine.engineVersion ? `${obj.engine.engineName} ${obj.engine.engineVersion}` : obj.engine.engineName;
                            manifest.runtime = manifest.runtime ? (`${manifest.runtime} (${engine})`) : engine;
                        }
                        if (obj.publisher && obj.publisher.author) {
                            manifest.author = obj.publisher.author;
                            if (obj.publisher.email) {
                                manifest.author = `${manifest.author} <${obj.publisher.email}>`;
                            }
                        }
                        if (obj.license) {
                            manifest.license = obj.license;
                        }
                        if (obj.Model && obj.Model.Signature) {
                            try {
                                const content = await context.fetch(obj.Model.Signature);
                                manifest.signature = content.read('json');
                                return manifest;
                            } catch {
                                return manifest;
                            }
                        }
                    }
                    return manifest;
                } catch (error) {
                    throw new mxnet.Error(`Failed to read manifest. ${error.message}`);
                }
            };
            try {
                const content = await context.fetch('MANIFEST.json');
                return parse(content.stream);
            } catch {
                try {
                    const content = await context.fetch('MAR-INF/MANIFEST.json');
                    return parse(content.stream);
                } catch {
                    return parse(null);
                }
            }
        };
        const createModel = (metadata, manifest, symbol, params) => {
            const parameters = new Map();
            if (params) {
                try {
                    for (const [key, array] of mxnet.ndarray.load(params)) {
                        const name = (key.startsWith('arg:') || key.startsWith('aux:')) ? key.substring(4) : key;
                        parameters.set(name, array);
                    }
                } catch {
                    // continue regardless of error
                }
            }
            if (symbol) {
                if (!manifest.format) {
                    const version = convertVersion(symbol.attrs && symbol.attrs.mxnet_version ? symbol.attrs.mxnet_version : null);
                    manifest.format = `MXNet${version ? ` v${version}` : ''}`;
                }
                if (symbol.nodes && symbol.nodes.some((node) => node && node.op === 'tvm_op')) {
                    manifest.format  = 'TVM';
                }
            }
            return new mxnet.Model(metadata, manifest, symbol, parameters);
        };
        const identifier = context.identifier;
        switch (context.type) {
            case 'mxnet.json': {
                let symbol = null;
                try {
                    symbol = context.target;
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new mxnet.Error(`Failed to load symbol entry (${message.replace(/\.$/, '')}).`);
                }
                const requestParams = async (manifest) => {
                    const file = basename(manifest.params, identifier, '.json', 'symbol', '-0000.params');
                    if (file) {
                        try {
                            const content = await context.fetch(file);
                            const reader = content.read('binary');
                            return createModel(metadata, manifest, symbol, reader);
                        } catch {
                            return createModel(metadata, manifest, symbol, null);
                        }
                    }
                    return createModel(metadata, manifest, symbol, null);
                };
                const manifest = await requestManifest();
                return requestParams(manifest);
            }
            case 'mxnet.params': {
                const params = context.read('binary');
                const requestSymbol = async (manifest) => {
                    const name = basename(manifest.symbol, identifier, '.params', null, '-symbol.json');
                    if (name) {
                        try {
                            const content = await context.fetch(name);
                            const symbol = content.read('json');
                            return createModel(metadata, manifest, symbol, params);
                        } catch {
                            return createModel(metadata, manifest, null, params);
                        }
                    }
                    return createModel(metadata, manifest, null, params);
                };
                const manifest = await requestManifest();
                return requestSymbol(manifest);
            }
            default: {
                throw new mxnet.Error(`Unsupported MXNet format '${context.type}'.`);
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
        this.format = manifest.format || 'MXNet';
        this.producer = manifest.producer || '';
        this.name = manifest.name || '';
        this.version = manifest.version;
        this.description = manifest.description || '';
        this.runtime = manifest.runtime || '';
        this.metadata = [];
        if (manifest.author) {
            this.metadata.push(new mxnet.Argument('author', manifest.author));
        }
        if (manifest.license) {
            this.metadata.push(new mxnet.Argument('license', manifest.license));
        }
        this.graphs = [new mxnet.Graph(metadata, manifest, symbol, params)];
    }
};

mxnet.Graph = class {

    constructor(metadata, manifest, symbol, params) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const tensors = new Map();
        if (params) {
            for (const [name, value] of params) {
                const shape = new mxnet.TensorShape(value.shape);
                const type = new mxnet.TensorType(value.dtype, shape);
                const tensor = new mxnet.Tensor(name, type, value.data);
                tensors.set(name, tensor);
            }
        }
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (!values.has(name)) {
                values.set(name, new mxnet.Value(name, type || null, tensor || null));
            } else if (type || (tensor && tensor !== values.get(name).initializer)) {
                throw new mxnet.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const updateOutput = (nodes, input) => {
            const [nodeIndex, outputIndex] = input;
            const node = nodes[nodeIndex];
            if (node) {
                while (outputIndex >= node.outputs.length) {
                    node.outputs.push([nodeIndex, node.outputs.length]);
                }
            }
            return [nodeIndex, outputIndex];
        };
        if (symbol) {
            const nodes = symbol.nodes;
            const inputs = {};
            const outputs = {};
            if (manifest && manifest.signature && manifest.signature.inputs) {
                for (const input of manifest.signature.inputs) {
                    inputs[input.data_name] = input;
                }
            }
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
            const arg_nodes = new Map(symbol.arg_nodes.map((index) => [index, index < nodes.length ? nodes[index] : null]));
            for (let i = 0; i < symbol.heads.length; i++) {
                const head = symbol.heads[i];
                const identifier = updateOutput(nodes, head);
                const name = `output${(i === 0) ? '' : (i + 1)}`;
                const signature = outputs[name];
                const type = signature && signature.data_shape ? new mxnet.TensorType(-1, new mxnet.TensorShape(signature.data_shape)) : null;
                const value = values.map(`[${identifier.join(',')}]`, type);
                const argument = new mxnet.Argument(name, [value]);
                this.outputs.push(argument);
            }
            const filtered = nodes.filter((node, index) => !arg_nodes.has(index));
            const initializers = new Map();
            for (const node of filtered) {
                if (node.op === 'RNN') {
                    node.inputs = node.inputs.filter((input) => {
                        const [index] = input;
                        const arg_node = arg_nodes.get(index);
                        if (arg_node && arg_node.op === 'null' && arg_node.name && arg_node.name.endsWith('_parameters') && arg_node.attr && arg_node.attr.__init__) {
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
                    const identifier = `[${input.join(',')}]`;
                    if (!initializers.has(identifier)) {
                        const [index] = input;
                        const arg_node = arg_nodes.get(index);
                        if (arg_node && arg_node.name && (!arg_node.inputs || arg_node.inputs.length === 0) && (arg_node.outputs && arg_node.outputs.length === 1)) {
                            if (tensors.has(arg_node.name)) {
                                initializers.set(identifier, tensors.get(arg_node.name));
                                arg_nodes.delete(index);
                            } else {
                                const prefix = node.name.endsWith('_fwd') ? node.name.slice(0, -3) : node.name;
                                if (arg_node.name && (arg_node.name.startsWith(`${prefix}_`) || arg_node.name.startsWith(`${prefix}.`))) {
                                    let dataType = -1;
                                    let shape = [];
                                    if (arg_node.attrs && arg_node.attrs.__dtype__ && arg_node.attrs.__shape__) {
                                        try {
                                            dataType = parseInt(arg_node.attrs.__dtype__, 10);
                                            shape = JSON.parse(`[${arg_node.attrs.__shape__.replace('(', '').replace(')', '').split(' ').join('').split(',').map(((dimension) => dimension || '"?"')).join(',')}]`);
                                        } catch {
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
                        values.map(param.id, null, tensors.get(param.id));
                    }
                }
            }
            for (const [, arg_node] of arg_nodes) {
                if (arg_node && (!arg_node.inputs || arg_node.inputs.length === 0) && (arg_node.outputs && arg_node.outputs.length === 1)) {
                    const identifier = `[${arg_node.outputs[0].join(',')}]`;
                    const name = arg_node.name;
                    const signature = inputs[name];
                    const type = signature && signature.data_shape ? new mxnet.TensorType(-1, new mxnet.TensorShape(signature.data_shape)) : null;
                    const value = values.map(identifier, type, tensors.get(identifier));
                    const argument = new mxnet.Argument(name, [value]);
                    this.inputs.push(argument);
                }
            }
            for (const node of filtered) {
                this.nodes.push(new mxnet.Node(metadata, node, initializers, values));
            }
        } else if (params) {
            const blocks = new Map();
            let separator = Array.from(params.keys()).every((key) => key.indexOf('_') !== -1) ? '_' : '';
            if (separator.length === 0) {
                separator = Array.from(params.keys()).every((key) => key.indexOf('.') !== -1) ? '.' : '';
            }
            if (separator.length > 0) {
                for (const [key] of params) {
                    const parts = key.split(separator);
                    let argumentName = parts.pop();
                    if (key.endsWith('moving_mean') || key.endsWith('moving_var')) {
                        argumentName = [parts.pop(), argumentName].join(separator);
                    }
                    const nodeName = parts.join(separator);
                    if (!blocks.has(nodeName)) {
                        blocks.set(nodeName, { name: nodeName, op: 'Weights', params: [] });
                    }
                    blocks.get(nodeName).params.push({ name: argumentName, id: key });
                    values.map(key, null, tensors.get(key));
                }
            } else {
                throw new mxnet.Error("Unsupported key format in params.");
            }

            for (const block of blocks.values()) {
                this.nodes.push(new mxnet.Node(metadata, block, new Map(), values));
            }
        }
    }
};

mxnet.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

mxnet.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new mxnet.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = !name && initializer && initializer.name ? initializer.name : name;
        this.type = !type && initializer && initializer.type ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

mxnet.Node = class {

    constructor(metadata, node, initializers, values) {
        let type = node.op;
        this.name = node.name;
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        const attrs = node.attrs || node.attr || node.param;
        if (attrs) {
            if (type === 'tvm_op' && attrs.func_name) {
                type = attrs.func_name;
            }
            for (const [name, obj] of Object.entries(attrs)) {
                if (type !== 'tvm_op' && name !== 'func_name') {
                    let value = obj;
                    let visible = true;
                    const schema = metadata.attribute(type, name);
                    if (schema && schema.type) {
                        switch (schema.type) {
                            case 'boolean':
                                switch (value) {
                                    case 0:
                                    case '0':
                                    case 'False':
                                        value = false;
                                        break;
                                    case 1:
                                    case '1':
                                    case 'True':
                                        value = true;
                                        break;
                                    default:
                                        throw new mxnet.Error(`Unsupported attribute boolean value '${value}'.`);
                                }
                                break;
                            case 'int32': {
                                const number = Number.parseInt(value, 10);
                                value = Number.isNaN(value - number) ? value : number;
                                break;
                            }
                            case 'float32':
                            case 'float64': {
                                const number = Number.parseFloat(value);
                                value = Number.isNaN(value - number) ? value : number;
                                break;
                            }
                            case 'int32[]':
                                if (value.length > 2 && value.startsWith('(') && value.endsWith(')')) {
                                    let array = [];
                                    const items = value.substring(1, value.length - 1).split(',')
                                        .map((item) => item.trim())
                                        .map((item) => item.endsWith('L') ? item.substring(0, item.length - 1) : item);
                                    for (const item of items) {
                                        const value = Number.parseInt(item, 10);
                                        if (Number.isNaN(item - value)) {
                                            array = null;
                                        } else if (array !== null) {
                                            array.push(value);
                                        }
                                    }
                                    if (array !== null) {
                                        value = array;
                                    }
                                }
                                break;
                            default:
                                throw new mxnet.Error(`Unsupported attribute type '${metadata.type}'.`);
                        }
                    }
                    if (metadata) {
                        if (metadata.visible === false) {
                            visible = false;
                        } else if (metadata.default !== undefined) {
                            let defaultValue = metadata.default;
                            if (value === defaultValue) {
                                visible = false;
                            } else if (Array.isArray(value) && Array.isArray(defaultValue)) {
                                defaultValue = defaultValue.slice(0, defaultValue.length);
                                if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] === null) {
                                    defaultValue.pop();
                                    while (defaultValue.length < value.length) {
                                        defaultValue.push(defaultValue[defaultValue.length - 1]);
                                    }
                                }
                                if (value.every((item, index) => item === defaultValue[index])) {
                                    visible = false;
                                }
                            }
                        }
                    }
                    const attribute = new mxnet.Argument(name, value, type, visible);
                    this.attributes.push(attribute);
                }
            }
        }
        this.type = metadata.type(type) || { name: type };
        if (node.inputs) {
            const inputs = node.inputs;
            let inputIndex = 0;
            if (this.type && this.type.inputs) {
                for (const inputDef of this.type.inputs) {
                    if (inputIndex < inputs.length || inputDef.optional !== true) {
                        const count = (inputDef.type === 'Tensor[]') ? (inputs.length - inputIndex) : 1;
                        const list = [];
                        for (const input of inputs.slice(inputIndex, inputIndex + count)) {
                            const identifier = `[${input.join(',')}]`;
                            if (identifier !== '' || (inputDef.optional !== true || inputDef.type === 'Tensor[]')) {
                                const value = values.map(identifier, null, initializers.get(identifier));
                                list.push(value);
                            }
                        }
                        const argument = new mxnet.Argument(inputDef.name, list);
                        this.inputs.push(argument);
                        inputIndex += count;
                    }
                }
            }
            if (inputIndex < inputs.length) {
                this.inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                    const name = (inputIndex + index).toString();
                    const identifier = `[${input.join(',')}]`;
                    const value = values.map(identifier, null, initializers.get(identifier));
                    return new mxnet.Argument(name, [value]);
                }));
            }
        }
        if (node.outputs) {
            const outputs = node.outputs;
            let outputIndex = 0;
            if (this.type && this.type.outputs) {
                for (const outputDef of this.type.outputs) {
                    if (outputIndex < outputs.length || outputDef.optional !== true) {
                        const list = [];
                        const count = (outputDef.type === 'Tensor[]') ? (outputs.length - outputIndex) : 1;
                        for (const output of outputs.slice(outputIndex, outputIndex + count)) {
                            const value = values.map(`[${output.join(',')}]`);
                            list.push(value);
                        }
                        const argument = new mxnet.Argument(outputDef.name, list);
                        this.outputs.push(argument);
                        outputIndex += count;
                    }
                }
            }
            if (outputIndex < outputs.length) {
                this.outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                    const name = (outputIndex + index).toString();
                    const value = values.map(`[${output.join(',')}]`);
                    return new mxnet.Argument(name, [value]);
                }));
            }
        }
        if (node.params) {
            for (const param of node.params) {
                const value = values.map(param.id);
                const argument = new mxnet.Argument(param.name, [value]);
                this.inputs.push(argument);
            }
        }
    }
};

mxnet.Tensor = class {

    constructor(name, type, data) {
        this.name = name;
        this.type = type;
        this.values = data;
        this.encoding = '<';
    }
};

mxnet.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case 0: this.dataType = 'float32'; break;
            case 1: this.dataType = 'float64'; break;
            case 2: this.dataType = 'float16'; break;
            case 3: this.dataType = 'uint8'; break;
            case 4: this.dataType = 'int32'; break;
            case 5: this.dataType = 'int8'; break;
            case 6: this.dataType = 'int64'; break;
            case -1: this.dataType = '?'; break;
            default: throw new mxnet.Error(`Unsupported type '${dataType}'.`);
        }
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

mxnet.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (this.dimensions) {
            if (this.dimensions.length === 0) {
                return '';
            }
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

mxnet.ndarray = class {

    static load(reader) {
        // NDArray::Load(dmlc::Stream* fi, std::vector<NDArray>* data, std::vector<std::string>* keys)
        const params = new Map();
        reader = new mxnet.BinaryReader(reader);
        if (reader.uint64().toNumber() !== 0x112) { // kMXAPINDArrayListMagic
            throw new mxnet.Error('Invalid signature.');
        }
        if (reader.uint64().toNumber() !== 0) {
            throw new mxnet.Error('Invalid reserved block.');
        }
        const values = new Array(reader.uint64().toNumber());
        for (let i = 0; i < values.length; i++) {
            values[i] = new mxnet.ndarray.NDArray(reader);
        }
        const decoder = new TextDecoder('ascii');
        const names = new Array(reader.uint64().toNumber());
        for (let i = 0; i < names.length; i++) {
            const size = reader.uint64().toNumber();
            const buffer = reader.read(size);
            names[i] = decoder.decode(buffer);
        }
        if (names.length !== values.length) {
            throw new mxnet.Error('Invalid parameters.');
        }
        for (let i = 0; i < names.length; i++) {
            params.set(names[i], values[i]);
        }
        return params;
    }
};

mxnet.ndarray.NDArray = class {

    constructor(reader) {
        mxnet.ndarray.NDArray._dataTypeSizeTable = [4, 8, 2, 1, 4, 1, 8];
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
                    default: throw mxnet.Error(`Unsupported NDArray type '${stype}'.`);
                }
                this.sshape = null;
                if (num_aux_data > 0) {
                    this.sshape = reader.uint64s();
                }
                this.shape = reader.uint64s();
                if (this.shape.length !== 0) {
                    this.context = {
                        deviceType: reader.uint32(),
                        deviceId: reader.uint32()
                    };
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
                    this.context = {
                        deviceType: reader.uint32(),
                        deviceId: reader.uint32()
                    };
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
                this.context = {
                    deviceType: reader.uint32(),
                    deviceId: reader.uint32()
                };
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

mxnet.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    read(length) {
        return this._reader.read(length);
    }

    uint32() {
        return this._reader.uint32();
    }

    uint32s() {
        const size = this.uint32();
        const array = new Array(size);
        for (let i = 0; i < size; i++) {
            array[i] = this.uint32();
        }
        return array;
    }

    uint64() {
        return this._reader.uint64();
    }

    uint64s() {
        const size = this.uint32();
        const array = new Array(size);
        for (let i = 0; i < size; i++) {
            array[i] = this.uint64().toNumber();
        }
        return array;
    }
};

mxnet.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MXNet model.';
    }
};

export const ModelFactory = mxnet.ModelFactory;
