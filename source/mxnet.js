/* jshint esversion: 6 */

var mxnet = mxnet || {};
var json = json || require('./json');
var zip = zip || require('./zip');
var ndarray = ndarray || {};

mxnet.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        switch (extension) {
            case 'json': {
                const obj = context.open('json');
                if (obj && obj.nodes && obj.arg_nodes && obj.heads) {
                    return true;
                }
                break;
            }
            case 'params': {
                const stream = context.stream;
                const signature = [ 0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ];
                if (stream.length > signature.length && stream.peek(signature.length).every((value, index) => value == signature[index])) {
                    return true;
                }
                break;
            }
        }
        return false;
    }

    open(context) {
        return mxnet.Metadata.open(context).then((metadata) => {
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
            const requestManifest = () => {
                const parse = (stream) => {
                    try {
                        const manifest = {};
                        const decoder = new TextDecoder('utf-8');
                        if (stream) {
                            const buffer = stream.peek();
                            const text = decoder.decode(buffer);
                            const json = JSON.parse(text);
                            if (json.Model) {
                                const modelFormat = json.Model['Model-Format'];
                                if (modelFormat && modelFormat != 'MXNet-Symbolic') {
                                    throw new mxnet.Error('Model format \'' + modelFormat + '\' not supported.');
                                }
                                manifest.format = 'MXNet Model Server';
                                if (json['Model-Archive-Version']) {
                                    manifest.format += ' v' + json['Model-Archive-Version'].toString();
                                }
                                if (!json.Model.Symbol) {
                                    throw new mxnet.Error('Manifest does not contain symbol entry.');
                                }
                                manifest.symbol = json.Model.Symbol;
                                if (json.Model.Signature) {
                                    manifest.signature = json.Model.Signature;
                                }
                                if (json.Model.Parameters) {
                                    manifest.params = json.Model.Parameters;
                                }
                                if (json.Model['Model-Name']) {
                                    manifest.name = json.Model['Model-Name'];
                                }
                                if (json.Model.Description && manifest.name !== json.Model.Description) {
                                    manifest.description = json.Model.Description;
                                }
                            }
                            else if (json.model) {
                                manifest.format = 'MXNet Model Archive';
                                if (json.specificationVersion) {
                                    manifest.format += ' v' + json.specificationVersion.toString();
                                }
                                if (json.model.modelName) {
                                    manifest.symbol = json.model.modelName + '-symbol.json';
                                }
                                if (json.model.modelName) {
                                    manifest.name = json.model.modelName;
                                }
                                if (manifest.model && json.model.modelVersion) {
                                    manifest.version = json.model.modelVersion;
                                }
                                if (manifest.model && manifest.model.modelName && manifest.name != json.model.description) {
                                    manifest.description = json.model.description;
                                }
                            }
                            else {
                                throw new mxnet.Error('Manifest does not contain model.');
                            }
                            if (json.Engine && json.Engine.MXNet) {
                                const version = convertVersion(json.Engine.MXNet);
                                manifest.runtime = 'MXNet v' + (version ? version : json.Engine.MXNet.toString());
                            }
                            if (json.License) {
                                manifest.license = json.License;
                            }
                            if (json.runtime) {
                                manifest.runtime = json.runtime;
                            }
                            if (json.engine && json.engine.engineName) {
                                const engine = json.engine.engineVersion ? json.engine.engineName + ' ' + json.engine.engineVersion : json.engine.engineName;
                                manifest.runtime = manifest.runtime ? (manifest.runtime + ' (' + engine + ')') : engine;
                            }
                            if (json.publisher && json.publisher.author) {
                                manifest.author = json.publisher.author;
                                if (json.publisher.email) {
                                    manifest.author = manifest.author + ' <' + json.publisher.email + '>';
                                }
                            }
                            if (json.license) {
                                manifest.license = json.license;
                            }
                            if (json.Model && json.Model.Signature) {
                                return context.request(json.Model.Signature).then((stream) => {
                                    const buffer = stream.peek();
                                    const text = decoder.decode(buffer);
                                    manifest.signature = JSON.parse(text);
                                    return manifest;
                                }).catch (() => {
                                    return manifest;
                                });
                            }
                        }
                        return manifest;
                    }
                    catch (err) {
                        throw new mxnet.Error('Failed to read manifest. ' + err.message);
                    }
                };
                return context.request('MANIFEST.json').then((stream) => {
                    return parse(stream);
                }).catch (() => {
                    return context.request('MAR-INF/MANIFEST.json').then((stream) => {
                        return parse(stream);
                    }).catch(() => {
                        return parse(null);
                    });
                });
            };
            const createModel = (metadata, manifest, symbol, params) => {
                const parameters = new Map();
                if (params) {
                    try {
                        const stream = new ndarray.Stream(params);
                        for (const key of Object.keys(stream.arrays)) {
                            const name = (key.startsWith('arg:') || key.startsWith('aux:')) ? key.substring(4) : key;
                            parameters.set(name, stream.arrays[key]);
                        }
                    }
                    catch (error) {
                        // continue regardless of error
                    }
                }
                if (symbol) {
                    if (!manifest.format) {
                        const version = convertVersion(symbol && symbol.attrs && symbol.attrs.mxnet_version ? symbol.attrs.mxnet_version : null);
                        manifest.format = 'MXNet' + (version ? ' v' + version : '');
                    }
                    if (symbol.nodes && symbol.nodes.some((node) => node && node.op == 'tvm_op')) {
                        manifest.producer  = 'TVM';
                    }
                }
                return new mxnet.Model(metadata, manifest, symbol, parameters);
            };
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'json': {
                    let symbol = null;
                    try {
                        symbol = context.open('json');
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new mxnet.Error("Failed to load symbol entry (" + message.replace(/\.$/, '') + ').');
                    }
                    const requestParams = (manifest) => {
                        const file = basename(manifest.params, identifier, '.json', 'symbol', '-0000.params');
                        if (file) {
                            return context.request(file, null).then((stream) => {
                                const buffer = stream.peek();
                                return createModel(metadata, manifest, symbol, buffer);
                            }).catch(() => {
                                return createModel(metadata, manifest, symbol, null);
                            });
                        }
                        return createModel(metadata, manifest, symbol, null);
                    };
                    return requestManifest().then((manifest) => {
                        return requestParams(manifest);
                    });
                }
                case 'params': {
                    const params = context.stream.peek();
                    const requestSymbol = (manifest) => {
                        const file = basename(manifest.symbol, identifier, '.params', null, '-symbol.json');
                        if (file) {
                            return context.request(file, 'utf-8').then((text) => {
                                const symbol = JSON.parse(text);
                                return createModel(metadata, manifest, symbol, params);
                            }).catch(() => {
                                return createModel(metadata, manifest, null, params);
                            });
                        }
                        return createModel(metadata, manifest, null, params);
                    };
                    return requestManifest().then((manifest) => {
                        return requestSymbol(manifest);
                    });
                }
                default: {
                    throw new mxnet.Error('Unsupported file extension.');
                }
            }
        });
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
        this._author = manifest.author || '';
        this._license = manifest.license || '';
        this._graphs = [ new mxnet.Graph(metadata, manifest, symbol, params) ];
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
                tensors.set(key, new mxnet.Tensor('Initializer', key, new mxnet.TensorType(value.dataType, new mxnet.TensorShape(value.shape.dimensions)), value.data));
            }
        }

        if (symbol) {
            const nodes = symbol.nodes;
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
                node.inputs = node.inputs.map((input) => {
                    return mxnet.Graph._updateOutput(nodes, input);
                });
            }

            const outputCountMap = {};
            for (const node of nodes) {
                for (const output of node.outputs) {
                    outputCountMap[output] = (outputCountMap[output] || 0) + 1;
                }
            }

            const argumentMap = {};
            for (const index of symbol.arg_nodes) {
                argumentMap[index] = (index < nodes.length) ? nodes[index] : null;
            }

            for (let i = 0; i < symbol.heads.length; i++) {
                const head = symbol.heads[i];
                const outputId = mxnet.Graph._updateOutput(nodes, head);
                const outputName = nodes[outputId[0]] ? nodes[outputId[0]].name : ('output' + ((i == 0) ? '' : (i + 1).toString()));
                let outputType = null;
                const outputSignature = outputs[outputName];
                if (outputSignature && outputSignature.data_shape) {
                    outputType = new mxnet.TensorType(-1, new mxnet.TensorShape(outputSignature.data_shape));
                }
                this._outputs.push(new mxnet.Parameter(outputName, [ new mxnet.Argument('[' + outputId.join(',') + ']', outputType, null) ]));
            }

            const initializerMap = {};
            for (const node of nodes.filter((node, index) => !argumentMap[index])) {
                this._nodes.push(new mxnet.Node(this._metadata, node, argumentMap, initializerMap, tensors));
            }

            for (const argumentKey of Object.keys(argumentMap)) {
                const argument = argumentMap[argumentKey];
                if (argument && (!argument.inputs || argument.inputs.length == 0) && (argument.outputs && argument.outputs.length == 1)) {
                    const inputId = argument.outputs[0];
                    const inputName = argument.name;
                    let inputType = null;
                    const inputSignature = inputs[inputName];
                    if (inputSignature && inputSignature.data_shape) {
                        inputType = new mxnet.TensorType(-1, new mxnet.TensorShape(inputSignature.data_shape));
                    }
                    this._inputs.push(new mxnet.Parameter(inputName, [ new mxnet.Argument('[' + inputId.join(',') + ']', inputType) ]));
                }
            }
        }
        else if (params) {
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
                }
            }
            else {
                throw new mxnet.Error("Unsupported key format in params.");
            }

            for (const block of blocks.values()) {
                this._nodes.push(new mxnet.Node(metadata, block, {}, {}, tensors));
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

    static _updateOutput(nodes, input) {
        const nodeIndex = input[0];
        const node = nodes[nodeIndex];
        const outputIndex = input[1];
        if (node) {
            while (outputIndex >= node.outputs.length) {
                node.outputs.push([ nodeIndex, node.outputs.length ]);
            }
        }
        return [ nodeIndex, outputIndex ];
    }
};

mxnet.Parameter = class {

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

mxnet.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new mxnet.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, node, argumentMap, initializerMap, tensors) {
        this._metadata = metadata;
        this._type = node.op;
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        const attrs = node.attrs || node.attr || node.param;
        if (attrs) {
            if (this._type == 'tvm_op' && attrs.func_name) {
                this._type = attrs.func_name;
            }
            for (const attributeName of Object.keys(attrs)) {
                if (this._type != 'tvm_op' && attributeName != 'func_name') {
                    this._attributes.push(new mxnet.Attribute(this._metadata, this.type, attributeName, attrs[attributeName]));
                }
            }
        }

        let initializer = null;
        const schema = metadata.type(this.type);
        if (node.inputs) {
            let inputs = node.inputs;
            if (this._type == 'RNN') {
                inputs = inputs.map((input) => {
                    const argumentNodeIndex = input[0];
                    const argument = argumentMap[argumentNodeIndex];
                    if (argument && argument.op == 'null' && argument.name &&
                        argument.name.endsWith('_parameters') && argument.attr && argument.attr.__init__) {
                        this._attributes.push(new mxnet.Attribute(this._metadata, this.type, argument.name, argument.attr.__init__));
                        delete argumentMap[argumentNodeIndex];
                        return null;
                    }
                    return input;
                });
                inputs = inputs.filter((item) => item != null);
            }
            const initializers = {};
            for (const input of inputs) {
                const id = '[' + input.join(',') + ']';
                initializer = initializerMap[id];
                if (!initializer) {
                    const argumentNodeIndex = input[0];
                    const argument = argumentMap[argumentNodeIndex];
                    if (argument && argument.name &&
                        (!argument.inputs || argument.inputs.length == 0) &&
                        (argument.outputs && argument.outputs.length == 1)) {
                        initializer = tensors.get(argument.name) || null;
                        if (initializer) {
                            delete argumentMap[argumentNodeIndex];
                        }
                        else {
                            let prefix = this._name;
                            if (prefix.endsWith('_fwd')) {
                                prefix = prefix.slice(0, -3);
                            }
                            if (argument.name && (argument.name.startsWith(prefix + '_') || argument.name.startsWith(prefix + '.'))) {
                                let dataType = -1;
                                let shape = [];
                                if (argument.attrs && argument.attrs.__dtype__ && argument.attrs.__shape__) {
                                    try {
                                        dataType = parseInt(argument.attrs.__dtype__);
                                        shape = JSON.parse('[' + argument.attrs.__shape__.replace('(', '').replace(')', '').split(' ').join('').split(',').map((dimension => dimension || '"?"' )).join(',') + ']');
                                    }
                                    catch (err) {
                                        // continue regardless of error
                                    }
                                }
                                let argumentType = null;
                                if (dataType !== -1 || shape.length > 0) {
                                    argumentType = new mxnet.TensorType(dataType, new mxnet.TensorShape(shape));
                                }
                                else {
                                    argumentType = new mxnet.TensorType(-1, new mxnet.TensorShape(null));
                                }
                                initializer = new mxnet.Tensor('Initializer', argument.name, argumentType, null);
                                delete argumentMap[argumentNodeIndex];
                            }
                        }
                    }
                }
                if (initializer) {
                    initializers[id] = initializer;
                    initializerMap[id] = initializer;
                }
            }

            let inputIndex = 0;
            if (schema && schema.inputs) {
                for (const inputDef of schema.inputs) {
                    if (inputIndex < inputs.length || inputDef.option != 'optional') {
                        const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                        const inputArguments = [];
                        for (const input of inputs.slice(inputIndex, inputIndex + inputCount)) {
                            const inputId = '[' + input.join(',') + ']';
                            if (inputId != '' || inputDef.option != 'optional') {
                                inputArguments.push(new mxnet.Argument(inputId, inputDef.type, initializers[inputId]));
                            }
                        }
                        this._inputs.push(new mxnet.Parameter(inputDef.name, inputArguments));
                        inputIndex += inputCount;
                    }
                }
            }
            if (inputIndex < inputs.length) {
                this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                    const inputId = '[' + input.join(',') + ']';
                    return new mxnet.Parameter((inputIndex + index).toString(), [
                        new mxnet.Argument(inputId, null, initializers[inputId])
                    ]);
                }));
            }
        }

        if (node.outputs) {
            const outputs = node.outputs;
            let outputIndex = 0;
            if (schema && schema.outputs) {
                for (const outputDef of schema.outputs) {
                    if (outputIndex < outputs.length || outputDef.option != 'optional') {
                        const outputArguments = [];
                        const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                        for (const output of outputs.slice(outputIndex, outputIndex + outputCount)) {
                            outputArguments.push(new mxnet.Argument('[' + output.join(',') + ']', null, null));
                        }
                        this._outputs.push(new mxnet.Parameter(outputDef.name, outputArguments));
                        outputIndex += outputCount;
                    }
                }
            }
            if (outputIndex < outputs.length) {
                this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                    return new mxnet.Parameter((outputIndex + index).toString(), [
                        new mxnet.Argument('[' + output.join(',') + ']', null, null)
                    ]);
                }));
            }
        }

        if (node.params) {
            for (const param of node.params) {
                this._inputs.push(new mxnet.Parameter(param.name, [
                    new mxnet.Argument(param.id, null, tensors.get(param.id) || null)
                ]));
            }
        }
    }

    get type() {
        return this._type;
    }

    get metadata() {
        return this._metadata.type(this._type);
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
        const schema = metadata.attribute(type, name);
        if (schema && schema.type) {
            switch (schema.type) {
                case 'boolean':
                    switch (value) {
                        case 'True':
                            this._value = true;
                            break;
                        case 'False':
                            this._value = false;
                            break;
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
            }
        }

        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                let defaultValue = schema.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < this._value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]);
                        }
                    }
                    if (this._value.every((item, index) => { return item == defaultValue[index]; })) {
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

    constructor(kind, name, type, data) {
        this._kind = kind;
        this._name = name;
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Initializer';
    }

    get name() {
        return this._name;
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

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        if (!this._type && this._type.dataType === '?') {
            context.state = 'Tensor has no data type.';
            return context;
        }

        if (this._type.shape.length < 1) {
            context.state = 'Tensor has unknown shape.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const results = [];
        const size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.data.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(mxnet.Tensor._decodeNumberFromFloat16(context.data.getUint16(context.index, true)));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'uint8':
                        results.push(context.data.getUint8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.data.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
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
        return results;
    }

    static _decodeNumberFromFloat16(value) {
        const s = (value & 0x8000) >> 15;
        const e = (value & 0x7C00) >> 10;
        const f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
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
            default: throw new mxnet.Error("Unknown type '" + dataType + "'.");
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

mxnet.Metadata = class {

    static open(context) {
        if (mxnet.Metadata._metadata) {
            return Promise.resolve(mxnet.Metadata._metadata);
        }
        return context.request('mxnet-metadata.json', 'utf-8', null).then((data) => {
            mxnet.Metadata._metadata = new mxnet.Metadata(data);
            return mxnet.Metadata._metadata;
        }).catch(() => {
            mxnet.Metadata._metadata = new mxnet.Metadata(null);
            return mxnet.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = {};
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
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

mxnet.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MXNet model.';
    }
};

ndarray.Stream = class {

    constructor(buffer) {

        this._arrays = {};

        const reader = new ndarray.Reader(buffer);
        if (!reader.checkSignature([ 0x12, 1, 0, 0, 0, 0, 0, 0 ])) {
            throw new ndarray.Error('Invalid signature.');
        }
        if (!reader.checkSignature([ 0, 0, 0, 0, 0, 0, 0, 0 ])) {
            throw new ndarray.Error('Invalid reserved block.');
        }

        const data = [];
        for (let dataSize = reader.uint64(); dataSize > 0; dataSize--) {
            data.push(new ndarray.Array(reader));
        }

        const decoder = new TextDecoder('ascii');
        const names = [];
        for (let namesSize = reader.uint64(); namesSize > 0; namesSize--) {
            const name = decoder.decode(reader.read(reader.uint64()));
            names.push(name);
        }

        if (names.length != data.length) {
            throw new ndarray.Error('Label count mismatch.');
        }

        for (let i = 0; i < names.length; i++) {
            this._arrays[names[i]] = data[i];
        }
    }

    get arrays() {
        return this._arrays;
    }

};

ndarray.Array = class {

    constructor(reader) {

        ndarray.Array._dataTypeSizeTable = [ 4, 8, 2, 1, 4, 1, 8 ];

        if (reader.checkSignature([ 0xc9, 0xfa, 0x93, 0xF9 ])) {
            this._loadV2(reader);
        }
        else if (reader.checkSignature([ 0xc8, 0xfa, 0x93, 0xF9 ])) {
            this._loadV1(reader);
        }
        else {
            this._loadV0(reader);
        }
    }

    _loadV2(reader) {
        const stype = reader.uint32();
        let num_aux_data = 0;
        switch (stype) {
            case 0: num_aux_data = 0; break; // kDefaultStorage
            case 1: num_aux_data = 1; break; // kRowSparseStorage
            case 2: num_aux_data = 2; break; // kCSRStorage
        }
        this.sshape = null;
        if (num_aux_data > 0) {
            this.sshape = new ndarray.Shape(reader, true);
        }
        this._shape = new ndarray.Shape(reader, true);
        if (this._shape.dimensions.length == 0) {
            return;
        }
        this._context = new ndarray.Context(reader);
        this._dataType = reader.uint32();
        if (num_aux_data > 0) {
            throw new ndarray.Error('Not implemented.');
        }
        const dataTypeSize = (this._dataType < ndarray.Array._dataTypeSizeTable.length) ? ndarray.Array._dataTypeSizeTable[this._dataType] : 0;
        const size = dataTypeSize * this._shape.size();
        this._data = reader.read(size);
    }

    _loadV1(reader) {
        this._shape = new ndarray.Shape(reader, true);
        if (this._shape.dimensions.length == 0) {
            return;
        }
        this._context = new ndarray.Context(reader);
        this._dataType = reader.uint32();
        const dataTypeSize = (this._dataType < ndarray.Array._dataTypeSizeTable.length) ? ndarray.Array._dataTypeSizeTable[this._dataType] : 0;
        const size = dataTypeSize * this._shape.size();
        this._data = reader.read(size);
    }

    _loadV0(reader) {
        this._shape = new ndarray.Shape(reader, false);
        this._context = new ndarray.Context(reader);
        this._dataType = reader.uint32();
        const dataTypeSize = (this._dataType < ndarray.Array._dataTypeSizeTable.length) ? ndarray.Array._dataTypeSizeTable[this._dataType] : 0;
        const size = dataTypeSize * this._shape.size();
        this._data = reader.read(size);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get data() {
        return this._data;
    }
};

ndarray.Shape = class {

    constructor(reader, uint64) {
        const ndim = reader.uint32();
        this._dimensions = [];
        for (let i = 0; i < ndim; i++) {
            this._dimensions.push(uint64 ? reader.uint64() : reader.uint32());
        }
    }

    get dimensions() {
        return this._dimensions;
    }

    size() {
        return this._dimensions.reduce((a, b) => a * b);
    }
};

ndarray.Context = class {

    constructor(reader) {
        this._deviceType = reader.uint32();
        this._deviceId = reader.uint32();
    }
};

ndarray.Reader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._end = buffer.length;
    }

    checkSignature(signature) {
        if (this._position + signature.length <= this._end) {
            for (let i = 0; i < signature.length; i++) {
                if (this._buffer[this._position + i] != signature[i]) {
                    return false;
                }
            }
        }
        this._position += signature.length;
        return true;
    }

    read(size) {
        if (this._position + size > this._end) {
            throw new ndarray.Error('Data not available.');
        }
        const data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    uint16() {
        if (this._position + 2 > this._end) {
            throw new ndarray.Error('Data not available.');
        }
        const value = this._buffer[this._position] | (this._buffer[this._position + 1] << 8);
        this._position += 2;
        return value;
    }

    uint32() {
        return this.uint16() | (this.uint16() << 16);
    }

    uint64() {
        const value = this.uint32();
        if (this.uint32() != 0) {
            throw new ndarray.Error('Large int64 value.');
        }
        return value;
    }
};

ndarray.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'NDArray Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mxnet.ModelFactory;
}