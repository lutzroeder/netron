/* jshint esversion: 6 */

// Experimental

var pytorch = pytorch || {};
var python = python || require('./python');
var base = base || require('./base');

pytorch.ModelFactory = class {

    match(context) {
        if (pytorch.Container.open(context)) {
            return true;
        }
        return false;
    }

    open(context) {
        const identifier = context.identifier;
        return pytorch.Metadata.open(context).then((metadata) => {
            let container = null;
            try {
                container = pytorch.Container.open(context, metadata, (error, fatal) => {
                    const message = error && error.message ? error.message : error.toString();
                    context.exception(new pytorch.Error(message.replace(/\.$/, '') + " in '" + identifier + "'."), fatal);
                });
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new pytorch.Error('File format is not PyTorch (' + message.replace(/\.$/, '') + ').');
            }
            return new pytorch.Model(metadata, container);
        });
    }
};

pytorch.Model = class {

    constructor(metadata, container) {
        this._format = container.format;
        this._producer = container.producer || '';
        this._graphs = [];
        const type = container.type;
        switch (type) {
            case 'script':
            case 'module':
                this._graphs.push(new pytorch.Graph(metadata, type, container.data, container));
                break;
            case 'weights':
                for (const data of container.data) {
                    this._graphs.push(new pytorch.Graph(metadata, type, data, container));
                }
                break;
        }
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

pytorch.Graph = class {

    constructor(metadata, type, data, container) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        this._littleEndian = container.littleEndian;

        switch (type) {
            case 'script': {
                this._name = container.name;
                const traced = container.trace();
                const initializers = new Map();
                if (container.constants) {
                    for (const constant of container.constants) {
                        constant.initializer = new pytorch.Tensor(constant.__variable__, constant, this._littleEndian);
                        initializers.set(constant.__variable__, constant);
                    }
                }
                if (data) {
                    const queue = [ data ];
                    while (queue.length > 0) {
                        const module = queue.shift();
                        if (module.__class__ && module.__class__.__module__ === '__torch__.torch.classes._nnapi' && module.__class__.__name__ === 'Compilation') {
                            continue;
                        }
                        for (const key of Object.keys(module)) {
                            if (key !== '__module__' && key !== '__name__' && key !== '__class__' && key !== '__parent__') {
                                const obj = module[key];
                                if (!Array.isArray(obj) && obj === Object(obj)) {
                                    if (pytorch.Utility.isTensor(obj)) {
                                        const parameter = obj;
                                        parameter.__parent__ = module;
                                        if (!parameter.initializer && parameter.storage()) {
                                            parameter.initializer = new pytorch.Tensor(parameter.name, parameter, this._littleEndian);
                                        }
                                        if (parameter.__variable__ && parameter.__count__ === 1) {
                                            initializers.set(parameter.__variable__, parameter);
                                        }
                                    }
                                    else if (obj && obj.__class__) {
                                        obj.__parent__ = module;
                                        if (!obj.__id__) {
                                            obj.__id__ = key;
                                        }
                                        queue.push(obj);
                                    }
                                }
                            }
                        }
                    }
                }
                if (traced) {
                    if (container.inputs) {
                        for (const input of container.inputs) {
                            this._inputs.push(new pytorch.Parameter(input, true, [
                                new pytorch.Argument(input, null, null)
                            ]));
                        }
                    }
                    if (container.outputs) {
                        for (const output of container.outputs) {
                            this._outputs.push(new pytorch.Parameter(output, true, [
                                new pytorch.Argument(output, null, null)
                            ]));
                        }
                    }
                    if (container.nodes) {
                        for (const node of container.nodes) {
                            const item = {
                                type: node.type,
                                node: node
                            };
                            this._nodes.push(new pytorch.Node(metadata, '', item, initializers));
                        }
                    }
                }
                if (data) {
                    this._loadScriptModule(metadata, container, data, initializers);
                }
                break;
            }
            case 'module': {
                this._type = (data.__module__ && data.__name__) ? (data.__module__ + '.' + data.__name__) : '';
                this._loadModule(metadata, container.data, [], []);
                break;
            }
            case 'weights': {
                this._name = data.name || '';
                for (const state_group of data.layers) {
                    const attributes = state_group.attributes || [];
                    const inputs = state_group.states.map((parameter) => {
                        return new pytorch.Parameter(parameter.name, true,
                            parameter.arguments.map((state) => {
                                const tensor = new pytorch.Tensor(state.id, pytorch.Utility.toTensor(state.value), this._littleEndian);
                                return new pytorch.Argument(state.id, null, tensor);
                            }));
                    });
                    const obj = {
                        name: state_group.name,
                        type: state_group.type || 'torch.nn.Module',
                        attributes: attributes,
                        inputs: inputs,
                        outputs: []
                    };
                    this._nodes.push(new pytorch.Node(metadata, '', obj, null));
                }
            }
        }
    }

    _loadModule(metadata, current, groups, inputs) {

        if (current.__class__ && current.__class__.__module__ !== 'torch.nn.modules.container' && (!current._modules || current._modules.size == 0)) {
            this._createNode(metadata, groups, '', current, inputs, false);
            return [];
        }

        if (!current._modules) {
            throw new pytorch.Error('Module does not contain modules.');
        }

        const sequential = current.__class__ && current.__class__.__module__ === 'torch.nn.modules.container' && current.__class__.__name__ === 'Sequential';

        for (const pair of current._modules) {
            const key = pair[0];
            const value = pair[1];
            if (value) {
                const type = value.__class__.__module__ + '.' + value.__class__.__name__;
                switch (type) {
                    case 'torch.nn.modules.container.Sequential':
                        groups.push(key);
                        inputs = this._loadModule(metadata, value, groups, sequential ? inputs : []);
                        groups.pop(key);
                        break;
                    default: {
                        inputs = this._createNode(metadata, groups, key, value, sequential ? inputs : [], sequential);
                        break;
                    }
                }
            }
        }
        return inputs;
    }

    _createNode(metadata, groups, key, obj, args, output) {

        const type = obj.__class__.__module__ + '.' + obj.__class__.__name__;
        const schema = metadata.type(type);

        let inputSchema = [ { name: 'input'} ];
        if (schema && schema.inputs && schema.inputs.length > 0) {
            inputSchema = schema.inputs.slice();
        }

        const inputName = inputSchema.shift().name;
        const inputs = [];
        if (args.length > 0) {
            inputs.push(new pytorch.Parameter(inputName, true, args.map((argument) => {
                return new pytorch.Argument(argument, null, null);
            })));
        }

        const parameters = obj._parameters || obj._buffers || [];
        for (const parameter of parameters) {
            const key = parameter[0];
            const value = pytorch.Utility.toTensor(parameter[1]);
            let visible = true;
            let inputName = '';
            if (inputSchema.length > 0) {
                const input = inputSchema.shift();
                inputName = input.name;
                visible = input.visible === false ? false : true;
            }
            if (value) {
                const initializer = new pytorch.Tensor('', value, this._littleEndian);
                inputs.push(new pytorch.Parameter(inputName || key, visible, [ new pytorch.Argument('', null, initializer) ]));
            }
        }

        const group = groups.join('/');
        const name = group ? (group + '/' + key) : key;

        const outputs = output ? [ new pytorch.Parameter('output', true, [ new pytorch.Argument(name, null, null) ]) ] : [];

        const attributes = [];
        for (const name of Object.keys(obj)) {
            if (name.startsWith('_')) {
                continue;
            }
            attributes.push({ name: name, value: obj[name] });
        }
        const item = {
            name: name,
            type: type,
            attributes: attributes,
            inputs: inputs,
            outputs: outputs
        };
        const node = new pytorch.Node(metadata, group, item, {});
        this._nodes.push(node);
        return [ node.name ];
    }

    _loadScriptModule(metadata, container, module, initializers) {
        if (module) {
            if (pytorch.Graph._getParameters(module).length > 0 && !module.__hide__) {
                const item = { module: module };
                this._nodes.push(new pytorch.Node(metadata, '', item, initializers));
            }
            const submodules = pytorch.Graph._getSubmodules(module);
            for (const submodule of submodules) {
                this._loadScriptModule(metadata, container, submodule, initializers);
            }
        }
    }

    static _getParameters(module) {
        const parameters = [];
        if (module && module.__class__.__module__ && module.__class__.__name__) {
            for (const key of Object.keys(module)) {
                if (pytorch.Utility.isTensor(module[key])) {
                    const parameter = module[key];
                    parameter.__id__ = key;
                    parameters.push(parameter);
                }
            }
        }
        return parameters;
    }

    static _getSubmodules(module) {
        const submodules = [];
        if (module && module.__class__ && module.__class__.__module__ && module.__class__.__name__) {
            for (const key of Object.keys(module)) {
                if (!key.startsWith('__')) {
                    const value = module[key];
                    if (value && value.__class__ && value.__module__ && value.__name__ && !pytorch.Utility.isTensor(value)) {
                        submodules.push(value);
                    }
                }
            }
        }
        return submodules;
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get groups() {
        return this._groups;
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

pytorch.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

pytorch.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new pytorch.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
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
};

pytorch.Node = class {

    constructor(metadata, group, item, initializers) {
        this._metadata = metadata;
        this._group = group || '';
        this._name = item.name || '';

        if (!item.module && !item.node) {
            this._type = item.type;
            this._inputs = item.inputs;
            this._outputs = item.outputs;
            this._attributes = item.attributes.map((attribute) => {
                const schema = metadata.attribute(this._type, attribute.name);
                return new pytorch.Attribute(schema, attribute.name, attribute.value);
            });
        }
        else {
            this._attributes = [];
            this._inputs = [];
            this._outputs = [];

            let module = item.module;
            if (module) {
                this._type = 'torch.nn.modules.module.Module';
                for (const parameter of pytorch.Graph._getParameters(module)) {
                    this._inputs.push(new pytorch.Parameter(parameter.__id__, true, [
                        new pytorch.Argument('', null, parameter.initializer || null)
                    ]));
                    if (parameter.__variable__) {
                        this._outputs.push(new pytorch.Parameter(parameter.__id__, true, [
                            new pytorch.Argument(parameter.__variable__, null, null)
                        ]));
                    }
                }
            }

            if (item.node) {
                this._type = item.type;
                const schema = metadata.type(this._type);
                module = null;
                let match = true;
                let count = 0;
                for (const input of item.node.inputs) {
                    for (const argument of input) {
                        const parameter = initializers.get(argument.id);
                        if (parameter) {
                            if (parameter.__parent__ && (module == null || module == parameter.__parent__)) {
                                module = parameter.__parent__;
                                count++;
                            }
                            else if (parameter.__variable__.startsWith('CONSTANTS.c')) {
                                argument.initializer = parameter.initializer;
                                count++;
                            }
                            else {
                                match = false;
                                break;
                            }
                        }
                    }
                    if (!match) {
                        break;
                    }
                }
                if (module) {
                    const params = pytorch.Graph._getParameters(module).filter((p) => p.__id__ !== 'num_batches_tracked');
                    if (params.length == count && match) {
                        module.__hide__ = true;
                        for (const input of item.node.inputs) {
                            for (const argument of input) {
                                const parameter = initializers.get(argument.id);
                                if (parameter && parameter.initializer) {
                                    argument.initializer = parameter.initializer;
                                }
                            }
                        }
                    }
                    else {
                        module = null;
                    }
                }

                for (let inputIndex = 0; inputIndex < item.node.inputs.length; inputIndex++) {
                    let inputName = inputIndex.toString();
                    if (schema && schema.inputs && schema.inputs.length > inputIndex) {
                        inputName = schema.inputs[inputIndex].name;
                    }
                    this._inputs.push(new pytorch.Parameter(inputName, true,
                        item.node.inputs[inputIndex].map((input) => new pytorch.Argument(input.id, null, input.initializer || null))
                    ));
                }

                for (let outputIndex = 0; outputIndex < item.node.outputs.length; outputIndex++) {
                    let outputName = outputIndex.toString();
                    if (schema && schema.outputs && schema.outputs.length > outputIndex) {
                        outputName = schema.outputs[outputIndex].name;
                    }
                    this._outputs.push(new pytorch.Parameter(outputName, true,
                        item.node.outputs[outputIndex].map((output) => new pytorch.Argument(output.id, null, null))
                    ));
                }

                for (const attribute of item.node.attributes) {
                    const name = attribute.name;
                    const value = attribute.value;
                    const schema = metadata.attribute(this._type, name);
                    this._attributes.push(new pytorch.Attribute(schema, name, value));
                }
            }
            if (module) {
                if (module.__id__) {
                    let current = module;
                    this._name = current.__id__;
                    while (current.__parent__ != null) {
                        current = current.__parent__;
                        if (!current.__parent__ && !current.__id__) {
                            break;
                        }
                        this._name = [ current.__id__, this._name ].join('.');
                    }
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group;
    }

    get type() {
        const index = this._type.indexOf(':');
        return index === -1 ? this._type : this._type.substring(0, index);
    }

    get metadata() {
        return this._metadata.type(this._type);
    }

    get function() {
        return this._type.startsWith('torch.nn.modules.') && this._type !== 'torch.nn.modules.module.Module';
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }
};

pytorch.Attribute = class {

    constructor(schema, name, value) {
        this._name = name;
        this._value = value;

        if (this._name === 'training') {
            this._visible = false;
            this._type = 'boolean';
            return;
        }
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'type')) {
                this._type = schema.type;
            }
            if (schema.visible === false) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (JSON.stringify(schema.default) == JSON.stringify(this._value)) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && !Array.isArray(schema.default) && this.value.every((item) => item == schema.default)) {
                    this._visible = false;
                }
            }
        }
        if (Array.isArray(value) && value.length > 0 && value.every((obj) => obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('torch.nn'))) {
            this._value = '?';
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

pytorch.Tensor = class {

    constructor(name, tensor, littleEndian) {
        const storage = tensor.storage();
        const size = tensor.size();
        this._name = name || '';
        this._type = new pytorch.TensorType(storage.dtype, new pytorch.TensorShape(size));
        this._data = storage.data;
        this._littleEndian = littleEndian;
    }

    get kind() {
        return 'Tensor';
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
        return pytorch.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (!this._type.dataType) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        switch (this._type.dataType) {
            case 'boolean':
            case 'uint8':
            case 'qint8':
            case 'int8':
            case 'int16':
            case 'int32':
            case 'int64':
            case 'float16':
            case 'float32':
            case 'float64':
                break;
            default:
                context.state = "Tensor data type '" + this._type.dataType + "' is not supported.";
                return context;
        }
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        try {
            context.data = this._data instanceof Uint8Array ? this._data : this._data.peek();
        }
        catch (err) {
            context.state = err.message;
            return context;
        }

        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const results = [];
        const dimensions = (context.dimensions.length == 0) ? [ 1 ] : context.dimensions;
        const size = dimensions[dimension];
        if (dimension == dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'boolean':
                        results.push(context.dataView.getUint8(context.index) === 0 ?  false : true);
                        context.index++;
                        context.count++;
                        break;
                    case 'uint8':
                        results.push(context.dataView.getUint8(context.index));
                        context.index++;
                        context.count++;
                        break;
                    case 'qint8':
                    case 'int8':
                        results.push(context.dataView.getInt8(context.index));
                        context.index++;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.dataView.getInt16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.dataView.getInt32(context.index, this._littleEndian));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.dataView.getInt64(context.index, this._littleEndian));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.dataView.getFloat16(context.index, this._littleEndian));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.dataView.getFloat32(context.index, this._littleEndian));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.dataView.getFloat64(context.index, this._littleEndian));
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
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => pytorch.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value && (value instanceof base.Int64 || value instanceof base.Uint64)) {
            return indentation + value.toString();
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

pytorch.TensorType = class {

    constructor(dtype, shape) {
        this._dataType = dtype.__reduce__();
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

pytorch.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions || [];
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions && this._dimensions.length > 0) {
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

pytorch.Execution = class extends python.Execution {

    constructor(sources, exceptionCallback) {
        super(sources, exceptionCallback);
        this.registerModule('ops');
        this.registerModule('torch');
        this.registerModule('torchvision');
        this.context.scope.ops._caffe2 = { __name__: 'torch', __class__: this.context.scope.builtins.module };
        const self = this;
        const torch = this.context.scope.torch;
        this.registerType('__torch__.torch.classes._nnapi.Compilation', class {
            constructor() {
                this.__hide__ = true;
            }
            __init__() {
            }
            init(serialized_model_tensor, parameter_buffers) {
                this.serialized_model_tensor = serialized_model_tensor;
                this.parameter_buffers = parameter_buffers;
                const storage = serialized_model_tensor.storage();
                new pytorch.nnapi.SerializedModel(storage.data, parameter_buffers);
            }
            run(inputs, outputs) {
                this.serialized_model_tensor.__variable__ = this.serialized_model_tensor.__variable__ || self.variable();
                this.serialized_model_tensor.__count__ = (this.serialized_model_tensor.__count__ || 0) + 1;
                self.push({
                    type: 'torch.classes._nnapi.Compilation',
                    attributes: [],
                    inputs: [
                        [ { id: this.serialized_model_tensor.__variable__ } ],
                        inputs.map((input) => { return { id: input.__variable__ }; }),
                        this.parameter_buffers.map((buffer) => { return { id: buffer.__variable__ }; })
                    ],
                    outputs: [
                        outputs.map((output) => { return { id: output.__variable__ }; })
                    ],
                });
            }
        });
        this.registerType('torch.autograd.variable.Variable', class {});
        this.registerType('torch.backends.cudnn.rnn.Unserializable', class {});
        this.registerType('torch.distributions.constraints._LowerCholesky', class {});
        this.registerType('torch.distributions.constraints._Real', class {});
        this.registerType('torch.distributions.multivariate_normal.MultivariateNormal', class {});
        this.registerType('torch.distributions.transforms.LowerCholeskyTransform', class {});
        this.registerType('torch.nn.backends.thnn._get_thnn_function_backend', class {});
        this.registerType('torch.nn.intrinsic.modules.fused.ConvReLU2d', class {});
        this.registerType('torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d', class {});
        this.registerType('torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d', class {});
        this.registerType('torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU', class {});
        this.registerType('torch.nn.modules.activation.CELU', class {});
        this.registerType('torch.nn.modules.activation.ELU', class {});
        this.registerType('torch.nn.modules.activation.GELU', class {});
        this.registerType('torch.nn.modules.activation.GLU', class {});
        this.registerType('torch.nn.modules.activation.Hardtanh', class {});
        this.registerType('torch.nn.modules.activation.Hardswish', class {});
        this.registerType('torch.nn.modules.activation.Hardsigmoid', class {});
        this.registerType('torch.nn.modules.activation.LeakyReLU', class {});
        this.registerType('torch.nn.modules.activation.LogSigmoid', class {});
        this.registerType('torch.nn.modules.activation.LogSoftmax', class {});
        this.registerType('torch.nn.modules.activation.MultiheadAttention', class {});
        this.registerType('torch.nn.modules.activation.ReLU', class {});
        this.registerType('torch.nn.modules.activation.ReLU6', class {});
        this.registerType('torch.nn.modules.activation.PReLU', class {});
        this.registerType('torch.nn.modules.activation.RReLU', class {});
        this.registerType('torch.nn.modules.activation.SELU', class {});
        this.registerType('torch.nn.modules.activation.Sigmoid', class {});
        this.registerType('torch.nn.modules.activation.SiLU', class {});
        this.registerType('torch.nn.modules.activation.Softmax', class {});
        this.registerType('torch.nn.modules.activation.Softmax2d', class {});
        this.registerType('torch.nn.modules.activation.Softplus', class {});
        this.registerType('torch.nn.modules.activation.Tanh', class {});
        this.registerType('torch.nn.modules.activation.Threshold', class {});
        this.registerType('torch.nn.modules.batchnorm.BatchNorm1d', class {});
        this.registerType('torch.nn.modules.batchnorm.BatchNorm2d', class {});
        this.registerType('torch.nn.modules.batchnorm.BatchNorm3d', class {});
        this.registerType('torch.nn.modules.batchnorm.SyncBatchNorm', class {});
        this.registerType('torch.nn.modules.container.ModuleDict', class {});
        this.registerType('torch.nn.modules.container.ModuleList', class {});
        this.registerType('torch.nn.modules.container.ParameterList', class {});
        this.registerType('torch.nn.modules.container.Sequential', class {});
        this.registerType('torch.nn.modules.conv.Conv1d', class {});
        this.registerType('torch.nn.modules.conv.Conv2d', class {});
        this.registerType('torch.nn.modules.conv.Conv3d', class {});
        this.registerType('torch.nn.modules.conv.ConvTranspose1d', class {});
        this.registerType('torch.nn.modules.conv.ConvTranspose2d', class {});
        this.registerType('torch.nn.modules.conv.ConvTranspose3d', class {});
        this.registerType('torch.nn.modules.distance.CosineSimilarity', class {});
        this.registerType('torch.nn.modules.dropout.AlphaDropout', class {});
        this.registerType('torch.nn.modules.dropout.Dropout', class {});
        this.registerType('torch.nn.modules.dropout.Dropout2d', class {});
        this.registerType('torch.nn.modules.dropout.Dropout3d', class {});
        this.registerType('torch.nn.modules.fold.Unfold', class {});
        this.registerType('torch.nn.modules.flatten.Flatten', class {});
        this.registerType('torch.nn.modules.flatten.Unflatten', class {});
        this.registerType('torch.nn.modules.instancenorm.InstanceNorm1d', class {});
        this.registerType('torch.nn.modules.instancenorm.InstanceNorm2d', class {});
        this.registerType('torch.nn.modules.instancenorm.InstanceNorm3d', class {});
        this.registerType('torch.nn.modules.linear._LinearWithBias', class {});
        this.registerType('torch.nn.modules.linear.Bilinear', class {});
        this.registerType('torch.nn.modules.linear.Linear', class {});
        this.registerType('torch.nn.modules.linear.Identity', class {});
        this.registerType('torch.nn.modules.loss.BCELoss', class {});
        this.registerType('torch.nn.modules.loss.BCEWithLogitsLoss', class {});
        this.registerType('torch.nn.modules.loss.CrossEntropyLoss', class {});
        this.registerType('torch.nn.modules.loss.CTCLoss', class {});
        this.registerType('torch.nn.modules.loss.KLDivLoss', class {});
        this.registerType('torch.nn.modules.loss.L1Loss', class {});
        this.registerType('torch.nn.modules.loss.MarginRankingLoss', class {});
        this.registerType('torch.nn.modules.loss.MSELoss', class {});
        this.registerType('torch.nn.modules.loss.NLLLoss', class {});
        this.registerType('torch.nn.modules.loss.NLLLoss2d', class {});
        this.registerType('torch.nn.modules.loss.SmoothL1Loss', class {});
        this.registerType('torch.nn.modules.module._IncompatibleKeys', class {});
        this.registerType('torch.nn.modules.module.Module', class {});
        this.registerType('torch.nn.modules.normalization.CrossMapLRN2d', class {});
        this.registerType('torch.nn.modules.normalization.GroupNorm', class {});
        this.registerType('torch.nn.modules.normalization.LayerNorm', class {});
        this.registerType('torch.nn.modules.normalization.LocalResponseNorm', class {});
        this.registerType('torch.nn.modules.padding.ReflectionPad1d', class {});
        this.registerType('torch.nn.modules.padding.ReflectionPad2d', class {});
        this.registerType('torch.nn.modules.padding.ReplicationPad1d', class {});
        this.registerType('torch.nn.modules.padding.ReplicationPad2d', class {});
        this.registerType('torch.nn.modules.padding.ReplicationPad3d', class {});
        this.registerType('torch.nn.modules.padding.ZeroPad2d', class {});
        this.registerType('torch.nn.modules.padding.ConstantPad1d', class {});
        this.registerType('torch.nn.modules.padding.ConstantPad2d', class {});
        this.registerType('torch.nn.modules.padding.ConstantPad3d', class {});
        this.registerType('torch.nn.modules.pixelshuffle.PixelShuffle', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveAvgPool1d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveAvgPool2d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveAvgPool3d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveMaxPool1d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveMaxPool2d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveMaxPool3d', class {});
        this.registerType('torch.nn.modules.pooling.AvgPool1d', class {});
        this.registerType('torch.nn.modules.pooling.AvgPool2d', class {});
        this.registerType('torch.nn.modules.pooling.AvgPool3d', class {});
        this.registerType('torch.nn.modules.pooling.FractionalMaxPool2d', class {});
        this.registerType('torch.nn.modules.pooling.LPPool2d', class {});
        this.registerType('torch.nn.modules.pooling.MaxPool1d', class {});
        this.registerType('torch.nn.modules.pooling.MaxPool2d', class {});
        this.registerType('torch.nn.modules.pooling.MaxPool3d', class {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool1d', class {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool2d', class {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool3d', class {});
        this.registerType('torch.nn.modules.rnn.GRU', class {});
        this.registerType('torch.nn.modules.rnn.GRUCell', class {});
        this.registerType('torch.nn.modules.rnn.LSTM', class {});
        this.registerType('torch.nn.modules.rnn.LSTMCell', class {});
        this.registerType('torch.nn.modules.rnn.RNN', class {});
        this.registerType('torch.nn.modules.sparse.Embedding', class {});
        this.registerType('torch.nn.modules.sparse.EmbeddingBag', class {});
        this.registerType('torch.nn.modules.transformer.Transformer', class {});
        this.registerType('torch.nn.modules.transformer.TransformerDecoder', class {});
        this.registerType('torch.nn.modules.transformer.TransformerDecoderLayer', class {});
        this.registerType('torch.nn.modules.transformer.TransformerEncoder', class {});
        this.registerType('torch.nn.modules.transformer.TransformerEncoderLayer', class {});
        this.registerType('torch.nn.modules.upsampling.Upsample', class {});
        this.registerType('torch.nn.modules.upsampling.UpsamplingBilinear2d', class {});
        this.registerType('torch.nn.modules.upsampling.UpsamplingNearest2d', class {});
        this.registerType('torch.nn.parallel.data_parallel.DataParallel', class {});
        this.registerType('torch.nn.parallel.distributed.DistributedDataParallel', class {});
        this.registerType('torch.nn.qat.modules.conv.Conv2d', class {});
        this.registerType('torch.nn.quantized.modules.activation.ReLU', class {});
        this.registerType('torch.nn.quantized.modules.batchnorm.BatchNorm2d', class {});
        this.registerType('torch.nn.quantized.modules.conv.Conv2d', class {});
        this.registerType('torch.nn.quantized.modules.conv.ConvTranspose2d', class {});
        this.registerType('torch.nn.quantized.modules.DeQuantize', class {});
        this.registerType('torch.nn.quantized.modules.functional_modules.FloatFunctional', class {});
        this.registerType('torch.nn.quantized.modules.functional_modules.QFunctional', class {});
        this.registerType('torch.nn.quantized.modules.linear.Linear', class {});
        this.registerType('torch.nn.quantized.modules.linear.LinearPackedParams', class {});
        this.registerType('torch.nn.quantized.modules.Quantize', class {});
        this.registerType('torch.nn.utils.prune.L1Unstructured', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNorm', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNormStateDictHook', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNormLoadStateDictPreHook', class {});
        this.registerType('torch.nn.utils.weight_norm.WeightNorm', class {});
        this.registerType('torch.optim.adam.Adam', class {});
        this.registerType('torch.optim.adagrad.Adagrad', class {});
        this.registerType('torch.optim.adadelta.Adadelta', class {});
        this.registerType('torch.optim.lr_scheduler.CosineAnnealingLR', class {});
        this.registerType('torch.optim.lr_scheduler.CyclicLR', class {});
        this.registerType('torch.optim.lr_scheduler.ExponentialLR', class {});
        this.registerType('torch.optim.lr_scheduler.LambdaLR', class {});
        this.registerType('torch.optim.lr_scheduler.MultiStepLR', class {});
        this.registerType('torch.optim.lr_scheduler.OneCycleLR', class {});
        this.registerType('torch.optim.lr_scheduler.ReduceLROnPlateau', class {});
        this.registerType('torch.optim.lr_scheduler.StepLR', class {});
        this.registerType('torch.optim.optimizer._RequiredParameter', class {});
        this.registerType('torch.optim.rmsprop.RMSprop', class {});
        this.registerType('torch.optim.sgd.SGD', class {});
        this.registerType('torch.quantization.fake_quantize.FakeQuantize', class {});
        this.registerType('torch.quantization.observer._PartialWrapper', class {});
        this.registerType('torch.quantization.observer.MinMaxObserver', class {});
        this.registerType('torch.quantization.QConfig.QConfig', class {});
        this.registerType('torch.quantization.stubs.DeQuantStub', class {});
        this.registerType('torch.quantization.stubs.QuantStub', class {});
        this.registerType('torch.utils.data.dataloader.DataLoader', class {});
        this.registerType('torch.utils.data.dataset.ConcatDataset', class {});
        this.registerType('torch.utils.data.sampler.BatchSampler', class {});
        this.registerType('torch.utils.data.sampler.SequentialSampler', class {});
        this.registerType('torchvision.datasets.folder.ImageFolder', class {});
        this.registerType('torchvision.datasets.mnist.MNIST', class {});
        this.registerType('torchvision.datasets.vision.StandardTransform', class {});
        this.registerType('torchvision.models.alexnet.AlexNet', class {});
        this.registerType('torchvision.models.densenet.DenseNet', class {});
        this.registerType('torchvision.models.densenet._DenseBlock', class {});
        this.registerType('torchvision.models.densenet._DenseLayer', class {});
        this.registerType('torchvision.models.densenet._Transition', class {});
        this.registerType('torchvision.models.detection._utils.BalancedPositiveNegativeSampler', class {});
        this.registerType('torchvision.models.detection._utils.BoxCoder', class {});
        this.registerType('torchvision.models.detection._utils.Matcher', class {});
        this.registerType('torchvision.models.detection.anchor_utils.AnchorGenerator', class {});
        this.registerType('torchvision.models.detection.backbone_utils.BackboneWithFPN', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.FasterRCNN', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.FastRCNNPredictor', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.TwoMLPHead', class {});
        this.registerType('torchvision.models.detection.keypoint_rcnn.KeypointRCNN', class {});
        this.registerType('torchvision.models.detection.keypoint_rcnn.KeypointRCNNHeads', class {});
        this.registerType('torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor', class {});
        this.registerType('torchvision.models.detection.mask_rcnn.MaskRCNN', class {});
        this.registerType('torchvision.models.detection.mask_rcnn.MaskRCNNHeads', class {});
        this.registerType('torchvision.models.detection.mask_rcnn.MaskRCNNPredictor', class {});
        this.registerType('torchvision.models.detection.roi_heads.RoIHeads', class {});
        this.registerType('torchvision.models.detection.rpn.AnchorGenerator', class {});
        this.registerType('torchvision.models.detection.rpn.RegionProposalNetwork', class {});
        this.registerType('torchvision.models.detection.rpn.RPNHead', class {});
        this.registerType('torchvision.models.detection.transform.GeneralizedRCNNTransform', class {});
        this.registerType('torchvision.models.googlenet.BasicConv2d', class {});
        this.registerType('torchvision.models.googlenet.GoogLeNet', class {});
        this.registerType('torchvision.models.googlenet.Inception', class {});
        this.registerType('torchvision.models.googlenet.InceptionAux', class {});
        this.registerType('torchvision.models.inception.BasicConv2d', class {});
        this.registerType('torchvision.models.inception.Inception3', class {});
        this.registerType('torchvision.models.inception.InceptionAux', class {});
        this.registerType('torchvision.models.inception.InceptionA', class {});
        this.registerType('torchvision.models.inception.InceptionB', class {});
        this.registerType('torchvision.models.inception.InceptionC', class {});
        this.registerType('torchvision.models.inception.InceptionD', class {});
        this.registerType('torchvision.models.inception.InceptionE', class {});
        this.registerType('torchvision.models.mnasnet._InvertedResidual', class {});
        this.registerType('torchvision.models.mnasnet.MNASNet', class {});
        this.registerType('torchvision.models.mobilenet.ConvBNReLU', class {});
        this.registerType('torchvision.models.mobilenet.MobileNetV2', class {});
        this.registerType('torchvision.models.mobilenet.InvertedResidual', class {});
        this.registerType('torchvision.models.mobilenetv2.ConvBNActivation', class {});
        this.registerType('torchvision.models.mobilenetv2.InvertedResidual', class {});
        this.registerType('torchvision.models.mobilenetv3.InvertedResidual', class {});
        this.registerType('torchvision.models.mobilenetv3.MobileNetV3', class {});
        this.registerType('torchvision.models.mobilenetv3.SqueezeExcitation', class {});
        this.registerType('torchvision.models.resnet.Bottleneck', class {});
        this.registerType('torchvision.models.resnet.BasicBlock', class {});
        this.registerType('torchvision.models.quantization.mobilenet.QuantizableInvertedResidual', class {});
        this.registerType('torchvision.models.quantization.mobilenet.QuantizableMobileNetV2', class {});
        this.registerType('torchvision.models.quantization.resnet.QuantizableBasicBlock', class {});
        this.registerType('torchvision.models.quantization.resnet.QuantizableBottleneck', class {});
        this.registerType('torchvision.models.quantization.resnet.QuantizableResNet', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.ASPP', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.ASPPConv', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.ASPPPooling', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.DeepLabHead', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.DeepLabV3', class {});
        this.registerType('torchvision.models.segmentation.fcn.FCN', class {});
        this.registerType('torchvision.models.segmentation.fcn.FCNHead', class {});
        this.registerType('torchvision.models.shufflenetv2.ShuffleNetV2', class {});
        this.registerType('torchvision.models.shufflenetv2.InvertedResidual', class {});
        this.registerType('torchvision.models.squeezenet.Fire', class {});
        this.registerType('torchvision.models.squeezenet.SqueezeNet', class {});
        this.registerType('torchvision.models.resnet.ResNet', class {});
        this.registerType('torchvision.models.vgg.VGG', class {});
        this.registerType('torchvision.models.video.resnet.BasicBlock', class {});
        this.registerType('torchvision.models.video.resnet.BasicStem', class {});
        this.registerType('torchvision.models.video.resnet.Conv2Plus1D', class {});
        this.registerType('torchvision.models.video.resnet.Conv3DNoTemporal', class {});
        this.registerType('torchvision.models.video.resnet.Conv3DSimple', class {});
        this.registerType('torchvision.models.video.resnet.R2Plus1dStem', class {});
        this.registerType('torchvision.models.video.resnet.VideoResNet', class {});
        this.registerType('torchvision.models._utils.IntermediateLayerGetter', class {});
        this.registerType('torchvision.ops.deform_conv.DeformConv2d', class {});
        this.registerType('torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork', class {});
        this.registerType('torchvision.ops.feature_pyramid_network.LastLevelMaxPool', class {});
        this.registerType('torchvision.ops.feature_pyramid_network.LastLevelP6P7', class {});
        this.registerType('torchvision.ops.misc.ConvTranspose2d', class {});
        this.registerType('torchvision.ops.misc.FrozenBatchNorm2d', class {});
        this.registerType('torchvision.ops.poolers.LevelMapper', class {});
        this.registerType('torchvision.ops.poolers.MultiScaleRoIAlign', class {});
        this.registerType('torchvision.transforms.transforms.Compose', class {});
        this.registerType('torchvision.transforms.transforms.Normalize', class {});
        this.registerType('torchvision.transforms.transforms.Resize', class {});
        this.registerType('torchvision.transforms.transforms.ToPILImage', class {});
        this.registerType('torchvision.transforms.transforms.ToTensor', class {});
        this.registerFunction('annotate', function(type, value) {
            return value;
        });
        this.registerFunction('int', function(value) {
            if (pytorch.Utility.isTensor(value)) {
                const storage = value.storage();
                if (storage && storage.dtype.__reduce__() === 'int64' && storage.data.length === 8) {
                    const buffer = storage.data;
                    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                    return view.getInt64(0, true);
                }
            }
            if (Number.isInteger(value)) {
                return value;
            }
            return NaN;
        });
        this.registerFunction('float', function(value) {
            if (pytorch.Utility.isTensor(value)) {
                const storage = value.storage();
                if (storage && storage.dtype.__reduce__() === 'float32') {
                    if (storage.size() !== undefined && storage.data.length === 4) {
                        const buffer = storage.data;
                        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                        return view.getFloat32(0, true);
                    }
                    return NaN;
                }
            }
            if (Number(value) === value) {
                return value;
            }
            return NaN;
        });
        this.registerFunction('str', function(value) {
            return JSON.stringify(value);
        });
        this.registerFunction('unchecked_cast', function(type, value) {
            return value;
        });
        this.registerFunction('ops.prim.data', function(tensor) {
            return tensor;
        });
        this.registerFunction('ops.prim.device', function(tensor) {
            return tensor.device;
        });
        this.registerFunction('ops.prim.dtype', function(tensor) {
            return tensor.dtype.scalar_type();
        });
        this.registerFunction('ops.prim.unchecked_unwrap_optional', function(value) {
            return value;
        });
        this.registerFunction('ops.prim.NumToTensor', function(value) {
            const tensor = self.invoke('torch.Tensor', []);
            tensor.value = value; // TODO
            return tensor;
        });
        this.registerFunction('ops.prim.min', function(value) {
            return Math.min.apply(null, value);
        });
        this.registerFunction('ops.prim.shape', function(tensor) {
            return tensor && tensor.size ? tensor.size() : undefined;
        });
        this.registerFunction('ops.quantized.conv_prepack', function(weight, bias, stride, padding, dilation, groups) {
            return {
                __class__: {
                    __module__: '__torch__.torch.classes.quantized',
                    __name__: 'Conv2dPackedParamsBase'
                },
                weight: weight,
                bias: bias,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups
            };
        });
        this.registerFunction('ops.quantized.conv1d_prepack', function(weight, bias, stride, padding, dilation, groups) {
            return {
                __class__: {
                    __module__: '__torch__.torch.classes.quantized',
                    __name__: 'Conv2dPackedParamsBase'
                },
                weight: weight,
                bias: bias,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups
            };
        });
        this.registerFunction('ops.quantized.conv2d_prepack', function(weight, bias, stride, padding, dilation, groups) {
            return {
                __class__: {
                    __module__: '__torch__.torch.classes.quantized',
                    __name__: 'Conv2dPackedParamsBase'
                },
                weight: weight,
                bias: bias,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups
            };
        });
        this.registerFunction('ops.quantized.conv3d_prepack', function(weight, bias, stride, padding, dilation, groups) {
            return {
                __class__: {
                    __module__: '__torch__.torch.classes.quantized',
                    __name__: 'Conv3dPackedParamsBase'
                },
                weight: weight,
                bias: bias,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups
            };
        });
        this.registerFunction('ops.quantized.conv_transpose2d_prepack', function(weight, bias, stride, padding, dilation, groups) {
            return {
                __class__: {
                    __module__: '__torch__.torch.classes.quantized',
                    __name__: 'Conv2dPackedParamsBase'
                },
                weight: weight,
                bias: bias,
                stride: stride,
                padding: padding,
                dilation: dilation,
                groups: groups
            };
        });
        this.registerFunction('ops.quantized.linear_prepack', function(weight, bias) {
            return {
                __class__: {
                    __module__: '__torch__.torch.classes.quantized',
                    __name__: 'LinearPackedParamsBase'
                },
                weight: weight,
                bias: bias
            };
        });
        this.registerFunction('ops.prim.RaiseException', function(message) {
            throw new pytorch.Error(message);
        });
        this.registerFunction('range', function(start, stop, step) {
            if (start !== undefined && Number.isInteger(start) && stop === undefined && step === undefined) {
                return Array(start).keys();
            }
            throw new pytorch.Error('Unsupported function range(' + JSON.stringify(start) + ', ' + JSON.stringify(stop) + ', ' + JSON.stringify(step) + ')');
        });
        this.registerFunction('torch._utils._rebuild_tensor', function (storage, storage_offset, size, stride) {
            const name = storage.__class__.__module__ + '.' + storage.__class__.__name__.replace('Storage', 'Tensor');
            const tensor = self.invoke(name, []);
            tensor.__setstate__([ storage, storage_offset, size, stride ]);
            return tensor;
        });
        this.registerFunction('torch._utils._rebuild_tensor_v2', function (storage, storage_offset, size, stride, requires_grad, backward_hooks) {
            const name = storage.__class__.__module__ + '.' + storage.__class__.__name__.replace('Storage', 'Tensor');
            const tensor = self.invoke(name, []);
            tensor.__setstate__([ storage, storage_offset, size, stride ]);
            tensor.requires_grad = requires_grad;
            tensor.backward_hooks = backward_hooks;
            return tensor;
        });
        this.registerFunction('torch._utils._rebuild_parameter', function(data, requires_grad, backward_hooks) {
            const obj = self.invoke('torch.nn.parameter.Parameter', [ data, requires_grad ]);
            obj.backward_hooks = backward_hooks;
            return obj;
        });
        this.registerFunction('torch._utils._rebuild_qtensor', function(storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks) {
            const name = storage.__class__.__module__ + '.' + storage.__class__.__name__.replace('Storage', 'Tensor');
            const tensor = self.invoke(name, []);
            tensor.__setstate__([ storage, storage_offset, size, stride ]);
            tensor.quantizer_params = quantizer_params;
            tensor.requires_grad = requires_grad;
            tensor.backward_hooks = backward_hooks;
            return tensor;
        });
        this.registerFunction('torch._set_item', function(dict, key, value) {
            dict[key] = value;
        });
        this.registerFunction('torch.__and__', function(left, right) {
            return left && right;
        });
        this.registerFunction('torch.__contains__', function(dict, key) {
            return dict[key] !== undefined;
        });
        this.registerFunction('torch.__derive_index', function(index, start, step) {
            return start + index * step;
        });
        this.registerFunction('torch.__is__', function(left, right) {
            if (left === null && right === null) {
                return true;
            }
            if ((left !== null && right === null) || (left === null && right !== null)) {
                return false;
            }
            throw new pytorch.Error("Unknown 'torch.__is__' expression type.");
        });
        this.registerFunction('torch.__isnot__', function(left, right) {
            if (left === null && right === null) {
                return false;
            }
            if ((left !== null && right === null) || (left === null && right !== null)) {
                return true;
            }
            throw new pytorch.Error("Unknown 'torch.__isnot__' expression type.");
        });
        this.registerFunction('torch.__not__', function(value) {
            if (typeof value === 'boolean') {
                return !value;
            }
            throw new pytorch.Error("Unknown 'torch.__not__' expression type.");
        });
        this.registerFunction('torch.__range_length', function(lo, hi, step) {
            if (step === 0) {
                throw new pytorch.Error('range() arg 3 must not be zero');
            }
            if (step > 0 && lo < hi) {
                return 1 + (hi - 1 - lo) / step;
            }
            else if (step < 0 && lo > hi) {
                return 1 + (lo - 1 - hi) / (0 - step);
            }
            return 0;
        });
        this.registerFunction('torch._unwrap_optional', function(value) {
            return value; // TODO
        });
        this.registerFunction('torch.add', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                return left * right;
            }
            if (Array.isArray(left) && Array.isArray(right)) {
                return left.concat(right);
            }
            if (typeof left === 'string' && typeof right === 'string') {
                return left + right;
            }
            throw new pytorch.Error('Unknown torch.add expression type.');
        });
        this.registerFunction('torch.append', function(list, value) {
            list.push(value);
            return value;
        });
        this.registerFunction('torch.insert', function(list, index, value) {
            list.splice(index, 0, value);
            return value;
        });
        this.registerFunction('torch.clear', function(value) {
            if (Object(value) === value) {
                for (const key of Object.keys(value)) {
                    delete value[key];
                }
            }
        });
        this.registerFunction('torch.dict', function(args) {
            const obj = {};
            if (args) {
                if (Array.isArray(args)) {
                    for (const pair of args) {
                        const key = pair[0];
                        const value = pair[1];
                        obj[key] = value;
                    }
                }
                else {
                    throw new pytorch.Error("'torch.dict' arguments not supported.");
                }
            }
            return obj;
        });
        this.registerFunction('torch.dim', function(tensor) {
            if (tensor && tensor.size) {
                const size = tensor.size();
                if (size) {
                    return size.length;
                }
            }
            return 0; // TODO
        });
        this.registerFunction('torch.eq', function(left, right) {
            if (typeof left === 'string' && typeof right === 'string') {
                return left === right;
            }
            if (typeof left === 'number' && typeof right === 'number') {
                return left === right;
            }
            throw new pytorch.Error("Unknown 'torch.eq' expression type.");
        });
        this.registerFunction('torch.floor', function(value) {
            return Math.floor(value);
        });
        this.registerFunction('torch.ceil', function(value) {
            return Math.ceil(value);
        });
        this.registerFunction('torch.floordiv', function(left, right) {
            return Math.floor(left / right);
        });
        this.registerFunction('torch.format', function() {
            const args = Array.from(arguments);
            const list = args.shift().split(/({}D?)/);
            return list.map((text) => {
                if (text === '{}' || text === '{}D') {
                    const arg = args.shift();
                    return Array.isArray(arg) ? '[' + arg.map((item) => item.toString()).join(', ') + ']' : arg.toString();
                }
                return text;
            }).join('');
        });
        this.registerFunction('torch.gt', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                if (!isNaN(left) && !isNaN(right)) {
                    return left > right;
                }
            }
            if (isNaN(left) && !isNaN(right)) {
                return true;
            }
            throw new pytorch.Error("Unknown 'torch.gt' expression type.");
        });
        this.registerFunction('torch.ge', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                if (!isNaN(left) && !isNaN(right)) {
                    return left > right;
                }
            }
            if (isNaN(left) && !isNaN(right)) {
                return true;
            }
            throw new pytorch.Error("Unknown 'torch.ge' expression type.");
        });
        this.registerFunction('torch.jit._pickle.build_boollist', function(data) {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_doublelist', function(data) {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_intlist', function(data) {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_tensorlist', function(data) {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_tensor_from_id', function(data) {
            const constants = self.context.getx('CONSTANTS');
            return constants['c' + data.toString()];
        });
        this.registerFunction('torch.jit._pickle.restore_type_tag', function(value /*, type_str */) {
            return value;
        });
        this.registerFunction('torch.keys', function(dict) {
            return Object.keys(dict);
        });
        this.registerFunction('torch.len', function(value) {
            if (Array.isArray(value)) {
                return value.length;
            }
            if (value && value.__len__) {
                return value.__len__();
            }
            return NaN;
        });
        this.registerFunction('torch.le', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                if (isNaN(left) || isNaN(right)) {
                    return false;
                }
                return left <= right;
            }
            throw new pytorch.Error("Unknown 'torch.le' expression type.");
        });
        this.registerFunction('torch.list', function(args) {
            return args;
        });
        this.registerFunction('torch.list_with_default', function(size /*, defaults */) {
            return size;
        });
        this.registerFunction('torch.lt', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                return left < right;
            }
            throw new pytorch.Error("Unknown 'torch.lt' expression type.");
        });
        this.registerFunction('torch.mul', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                return left * right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            throw new pytorch.Error("Unknown 'torch.mul' expression type.");
        });
        this.registerFunction('torch.div', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                return left / right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            throw new pytorch.Error("Unknown 'torch.div' expression type.");
        });
        this.registerFunction('torch.remainder', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                return left % right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            throw new pytorch.Error("Unknown 'torch.remainder' expression type.");
        });
        this.registerFunction('torch.ne', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                if (isNaN(left) || isNaN(right)) {
                    return false;
                }
                return left !== right;
            }
            if (Array.isArray(left) && Array.isArray(right) && left.length === right.length) {
                return false;
            }
            if (typeof left === 'string' && typeof right === 'string') {
                return left !== right;
            }
            throw new pytorch.Error("Unknown 'torch.ne' expression type.");
        });
        this.registerFunction('torch.neg', function(value) {
            if (typeof value === 'number') {
                return -value;
            }
            throw new pytorch.Error("Unknown 'torch.neg' expression type.");
        });
        this.registerFunction('torch.q_scale', function(/* tensor */) {
            return -1; // TODO
        });
        this.registerFunction('torch.t', function(tensor) {
            return tensor;
        });
        this.registerFunction('torch.size', function(tensor, dim) {
            if (tensor && tensor.size) {
                const size = tensor.size();
                if (Array.isArray(size)) {
                    if (dim === undefined) {
                        return size;
                    }
                    if (Number.isInteger(dim)) {
                        if (dim >= 0 && dim < size.length) {
                            return size[dim];
                        }
                        if (dim < 0 && -dim < size.length) {
                            return size[size.length + dim];
                        }
                    }
                    throw new pytorch.Error('Dimension out of range (expected to be in range of ' + JSON.stringify(size) + ', but got ' + JSON.stringify(dim) + ').');
                }
            }
            return NaN;
        });
        this.registerFunction('torch.slice', function(l, start, end, step) {
            if (step !== 1) {
                throw new pytorch.Error('Slicing only supports step=1');
            }
            start = Math.max(0, start >= 0 ? start : l.length + start);
            end = Math.min(l.length, end);
            return l.slice(start, end);
        });
        this.registerFunction('torch.sub', function(left, right) {
            if (typeof left === 'number' && typeof right === 'number') {
                return left - right;
            }
            throw new pytorch.Error("Unknown 'torch.sub' expression type.");
        });
        this.registerFunction('torch.values', function(dict) {
            return Object.keys(dict).map((key) => dict[key]);
        });
        this.registerFunction('torch.warn', function() {
        });
        this.registerFunction('uninitialized', function(/* type */) {
            return undefined;
        });
        this.registerType('torch.device', class {
            constructor(type, index) {
                this.type = type;
                if (index) {
                    this.index = index;
                }
            }
        });
        this.registerType('torch.dtype', class {
            constructor(type) {
                this._type = type;
                this._data = pytorch.Utility.getScalarType(type);
            }
            scalar_type() {
                return this._type;
            }
            itemsize() {
                return this._data.itemsize;
            }
            __reduce__() {
                return this._data.name;
            }
            __str__() {
                return 'torch.' + this._data.name;
            }
        });
        this.registerType('torch.utils.hooks.RemovableHandle', class {
            __setstate__(state) {
                this.hooks_dict_ref = state[0] || new Map();
                this.id = state[1];
            }
        });
        this.registerType('torch.storage._StorageBase', class {
            constructor(size, dtype) {
                this._size = size;
                this._dtype = dtype;
            }
            get dtype() {
                return this._dtype;
            }
            get data() {
                return this._cdata;
            }
            element_size() {
                return this._dtype.element_size;
            }
            size() {
                return this._size;
            }
            _set_cdata(data) {
                const length = this.size() * this.dtype.itemsize();
                if (length !== data.length) {
                    throw new pytorch.Error('Storage data size mismatch.');
                }
                this._cdata = data;
            }
            _set_from_file(unpickler) {
                const size = unpickler.int64();
                if (size !== this.size()) {
                    throw new pytorch.Error('Storage size mismatch.');
                }
                const itemsize = this.dtype.itemsize();
                const data = unpickler.stream(itemsize * size);
                this._set_cdata(data);
            }
            static _new_with_file(unpickler) {
                const size = unpickler.int64();
                const storage = new this(size);
                const itemsize = storage.dtype.itemsize();
                const data = unpickler.stream(itemsize * size);
                storage._set_cdata(data);
                return storage;
            }
        });
        this.registerType('torch.BoolStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.bool);
            }
        });
        this.registerType('torch.ByteStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.uint8);
            }
        });
        this.registerType('torch.CharStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int8);
            }
        });
        this.registerType('torch.ShortStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int16);
            }
        });
        this.registerType('torch.IntStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int32);
            }
        });
        this.registerType('torch.LongStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int64);
            }
        });
        this.registerType('torch.HalfStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.float16);
            }
        });
        this.registerType('torch.FloatStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.float32);
            }
        });
        this.registerType('torch.DoubleStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.float64);
            }
        });
        this.registerType('torch.QInt8Storage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.qint8);
            }
        });
        this.registerType('torch.QUInt8Storage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.quint8);
            }
        });
        this.registerType('torch.Tensor', class {
            constructor() {
            }
            get dtype() {
                return this.storage().dtype;
            }
            get shape() {
                return this._shape;
            }
            size() {
                return this._shape;
            }
            storage() {
                if (!this._storage) {
                    const name = this.__class__.__name__ == 'Tensor' ? 'FloatStorage' : this.__storage__.__name__.replace('Tensor', 'Storage');
                    this._storage = self.invoke(this.__class__.__module__ + '.' + name, []);
                }
                return this._storage;
            }
            storage_offset() {
                return this._storage_offset;
            }
            stride() {
                return this._stride;
            }
            resize_(shape) {
                this._shape = shape;
            }
            __len__() {
                return this._shape[0];
            }
            __setstate__(state) {
                this._storage = state[0];
                this._storage_offset = state[1];
                this._shape = state[2];
                this._stride = state[3];
            }
        });
        this.registerType('torch.nn.parameter.Parameter', class extends torch.Tensor {
            constructor(data, requires_grad) {
                super();
                this.data = data;
                this.requires_grad = requires_grad !== undefined ? requires_grad : true;
            }
            __setstate__(state) {
                switch (state.length) {
                    case 4:
                        this.data = state[0];
                        break;
                    case 5:
                        this.data = state[0];
                        break;
                }
            }
        });
        this.registerType('torch.BoolTensor', class extends torch.Tensor {});
        this.registerType('torch.ByteTensor', class extends torch.Tensor {});
        this.registerType('torch.CharTensor', class extends torch.Tensor {});
        this.registerType('torch.ShortTensor', class extends torch.Tensor {});
        this.registerType('torch.IntTensor', class extends torch.Tensor {});
        this.registerType('torch.LongTensor', class extends torch.Tensor {});
        this.registerType('torch.HalfTensor', class extends torch.Tensor {});
        this.registerType('torch.FloatTensor', class extends torch.Tensor {});
        this.registerType('torch.DoubleTensor', class extends torch.Tensor {});
        this.registerType('torch.QInt8Tensor', class extends torch.Tensor {});
        this.registerType('torch.QUInt8Tensor', class extends torch.Tensor {});
        this.registerType('torch.cuda.FloatTensor', class extends torch.Tensor {});
        this.registerType('torch.cuda.DoubleTensor', class extends torch.Tensor {});
        torch.uint8 = new torch.dtype(pytorch.ScalarType.uint8);
        torch.int8 = new torch.dtype(pytorch.ScalarType.int8);
        torch.int16 = new torch.dtype(pytorch.ScalarType.int16);
        torch.int32 = new torch.dtype(pytorch.ScalarType.int32);
        torch.int64 = new torch.dtype(pytorch.ScalarType.int64);
        torch.float16 = new torch.dtype(pytorch.ScalarType.float16);
        torch.float32 = new torch.dtype(pytorch.ScalarType.float32);
        torch.float64 = new torch.dtype(pytorch.ScalarType.float64);
        torch.complex32 = new torch.dtype(pytorch.ScalarType.complex32);
        torch.complex64 = new torch.dtype(pytorch.ScalarType.complex64);
        torch.complex128 = new torch.dtype(pytorch.ScalarType.complex128);
        torch.bool = new torch.dtype(pytorch.ScalarType.boolean);
        torch.qint8 = new torch.dtype(pytorch.ScalarType.qint8);
        torch.quint8 = new torch.dtype(pytorch.ScalarType.quint8);
        torch.qint32 = new torch.dtype(pytorch.ScalarType.qint32);
        torch.bfloat16 = new torch.dtype(pytorch.ScalarType.bfloat16);
    }

    debug(file) {
        const buffer = this.source(file + '.debug_pkl');
        if (buffer) {
            return null;
            // const unpickler = new python.Unpickler(buffer);
            // return unpickler.load((name, args) => this.invoke(name, args), null);
        }
        return null;
    }
};

pytorch.Container = class {

    static open(context, metadata, exception) {
        if (context.entries('zip').some((entry) => entry.name === 'model.json' || entry.name === 'data.pkl' || entry.name.endsWith('/model.json') || entry.name.endsWith('/data.pkl'))) {
            return new pytorch.Container.Zip(context.entries('zip'), metadata, exception);
        }
        const stream = context.stream;
        const signature = [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
        if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            return new pytorch.Container.Pickle(stream, exception);
        }
        if (context.entries('tar').some((entry) => entry.name == 'pickle')) {
            return new pytorch.Container.Tar(context.entries('tar'), exception);
        }
        return null;
    }
};

pytorch.Container.Tar = class {

    constructor(entries, exceptionCallback) {
        this._entries = entries;
        this._exceptionCallack = exceptionCallback;
    }

    get format() {
        return 'PyTorch v0.1.1';
    }

    get type() {
        this._unpickle();
        return this._type;
    }

    get data() {
        this._unpickle();
        return this._data;
    }

    get littleEndian() {
        this._unpickle();
        return this._littleEndian;
    }

    _unpickle() {
        if (!this._entries) {
            return;
        }
        this._type = '';
        this._data = null;
        this._littleEndian = true;

        const execution = new pytorch.Execution(null, this._exceptionCallback);

        const entries = {};
        for (const entry of this._entries) {
            switch (entry.name) {
                case 'sys_info': entries.sys_info = entry.data; break;
                case 'pickle': entries.pickle = entry.data; break;
                case 'storages': entries.storages = entry.data; break;
                case 'tensors': entries.tensors = entry.data; break;
            }
        }

        this._exceptionCallback = null;
        this._entries = null;

        if (entries.sys_info) {
            const unpickler = new python.Unpickler(entries.sys_info);
            const sys_info = unpickler.load((name, args) => execution.invoke(name, args));
            if (sys_info.protocol_version != 1000) {
                throw new pytorch.Error("Unsupported protocol version '" + sys_info.protocol_version + "'.");
            }
            if (sys_info.type_sizes &&
                ((sys_info.type_sizes.int && sys_info.type_sizes.int != 4) ||
                (sys_info.type_sizes.long && sys_info.type_sizes.long != 4) ||
                (sys_info.type_sizes.short && sys_info.type_sizes.short != 2))) {
                throw new pytorch.Error('Unsupported type sizes.');
            }
            this._littleEndian = sys_info.little_endian;
        }

        const deserialized_objects = {};
        if (entries.storages) {
            const unpickler = new python.Unpickler(entries.storages);
            const num_storages = unpickler.load((name, args) => execution.invoke(name, args));
            for (let i = 0; i < num_storages; i++) {
                const args = unpickler.load();
                const key = args[0];
                const storage_type = execution.type(args[2]);
                const obj = storage_type._new_with_file(unpickler);
                deserialized_objects[key] = obj;
            }
            /*
            let storage_views = unpickler.load();
            for target_cdata, root_cdata, offset, size in storage_views:
                root = deserialized_objects[root_cdata]
                deserialized_objects[target_cdata] = root[offset:offset + size]
            */
        }

        if (entries.tensors) {
            const unpickler = new python.Unpickler(entries.tensors);
            const num_tensors = unpickler.load((name, args) => execution.invoke(name, args));
            for (let i = 0; i < num_tensors; i++) {
                const args = unpickler.load();
                const key = args[0];
                const storage_id = args[1];
                const storage = deserialized_objects[storage_id];
                const ndim = unpickler.int32();
                unpickler.read(4);
                const shape = new Array(ndim);
                for (let j = 0; j < ndim; j++) {
                    shape[j] = unpickler.int64();
                }
                const stride = new Array(ndim);
                for (let j = 0; j < ndim; j++) {
                    stride[j] = unpickler.int64();
                }
                const storage_offset = unpickler.int64();
                const tensor = execution.invoke('torch._utils._rebuild_tensor', [ storage, storage_offset, shape, stride ]);
                deserialized_objects[key] = tensor;
            }
        }

        if (entries.pickle) {
            const unpickler = new python.Unpickler(entries.pickle);
            const persistent_load = (saved_id) => {
                return deserialized_objects[saved_id];
            };
            const obj = unpickler.load((name, args) => execution.invoke(name, args), persistent_load);
            const weights = pytorch.Utility.findWeights(obj);
            if (weights) {
                this._type = 'weights';
                this._data = weights;
            }
            else {
                throw new pytorch.Error('File does not contain root module or state dictionary.');
            }
        }
    }
};

pytorch.Container.Pickle = class {

    constructor(stream, exception) {
        this._stream = stream;
        this._exceptionCallback = exception;
    }

    get format() {
        return 'PyTorch v0.1.10';
    }

    get type() {
        this._unpickle();
        return this._type;
    }

    get data() {
        this._unpickle();
        return this._data;
    }

    get littleEndian() {
        this._unpickle();
        return this._littleEndian;
    }

    _unpickle() {
        if (!this._stream) {
            return;
        }

        const execution = new pytorch.Execution(null, this._exceptionCallback);
        const unpickler = new python.Unpickler(this._stream.length < 0x7ffff000 ? this._stream.peek() : this._stream);

        this._stream = null;
        this._exceptionCallback = null;

        unpickler.load(); // magic_number
        const protocol_version = unpickler.load();
        if (protocol_version != 1001) {
            throw new pytorch.Error("Unsupported protocol version '" + protocol_version + "'.");
        }
        const sys_info = unpickler.load();
        if (sys_info.protocol_version != 1001) {
            throw new pytorch.Error("Unsupported protocol version '" + sys_info.protocol_version + "'.");
        }
        this._littleEndian = sys_info.little_endian;

        const module_source_map = new Map();
        const deserialized_objects = new Map();
        const persistent_load = (saved_id) => {
            const typename = saved_id.shift();
            const data = saved_id;
            switch (typename) {
                case 'module': {
                    const module = data[0];
                    const source = data[2];
                    module_source_map.set(module, source);
                    return data[0];
                }
                case 'storage': {
                    const data_type = execution.type(data.shift());
                    const root_key = data.shift();
                    data.shift(); // location
                    const size = data.shift();
                    const view_metadata = data.shift();
                    if (!deserialized_objects.has(root_key)) {
                        const obj = new data_type(size);
                        deserialized_objects.set(root_key, obj);
                    }
                    if (view_metadata) {
                        const view_key = view_metadata.shift();
                        view_metadata.shift(); // view_offset
                        view_metadata.shift(); // view_size
                        if (!deserialized_objects.has(view_key)) {
                            const view = null; // storage.slice(view_offset, view_offset + view_size);
                            deserialized_objects.set(view_key, view);
                        }
                        return deserialized_objects.get(view_key);
                    }
                    return deserialized_objects.get(root_key);
                }
            }
            throw new pytorch.Error("Unknown persistent load type '" + typename + "'.");
        };

        const obj = unpickler.load((name, args) => execution.invoke(name, args), persistent_load);
        if (!obj) {
            throw new pytorch.Error('File format is not PyTorch.');
        }
        if (obj === 'None') {
            throw new pytorch.Error("File contains 'None' root object.");
        }

        const deserialized_storage_keys = unpickler.load();
        for (const deserialized_storage_key of deserialized_storage_keys) {
            const storage = deserialized_objects.get(deserialized_storage_key);
            storage._set_from_file(unpickler);
        }

        const root = pytorch.Utility.findModule(obj);
        if (root) {
            this._type = 'module';
            this._data = root;
        }
        else {
            const weights = pytorch.Utility.findWeights(obj);
            if (weights) {
                this._type = 'weights';
                this._data = weights;
            }
            else {
                throw new pytorch.Error('File does not contain root module or state dictionary.');
            }
        }
    }
};

pytorch.Container.Zip = class {

    constructor(entries, metadata, exceptionCallback) {
        this._entries = entries;
        this._metadata = metadata;
        this._exceptionCallback = exceptionCallback;
        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
        const entry = this._entries.find((entry) => entry.name == 'model.json' || entry.name == 'data.pkl' || entry.name.endsWith('/model.json') || entry.name.endsWith('/data.pkl'));
        if (!entry) {
            throw new pytorch.Error("PyTorch Zip container does not contain 'data.pkl' or 'model.json'.");
        }
        const lastIndex = entry.name.lastIndexOf('/');
        this._prefix = lastIndex === -1 ? '' : entry.name.substring(0, lastIndex + 1);
        this._utf8Decoder = new TextDecoder('utf-8');
    }

    get format() {
        if (this._format === undefined) {
            if (this._entry('model.json')) {
                this._format = this._entry('attributes.pkl') ? 'TorchScript v1.1' : 'TorchScript v1.0';
            }
            else if (this._entry('data.pkl')) {
                const versionEntry = this._entry('version');
                const versionNumber = versionEntry ? this._utf8Decoder.decode(versionEntry.data).split('\n').shift() : '';
                // https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/inline_container.h
                // kProducedFileFormatVersion
                const versionTable = {
                    '1': 'v1.3',
                    '2': 'v1.5', // 7a2889b014ce36fcc333b2c6de6f29f976652f84 (#28122)
                    '3': 'v1.6', // 2ec6a30722b0ef85632a2f3e7ce6f80da403008a (#36085)
                    '4': 'v1.6', // 95489b590f00801bdee7f41783f30874883cf6bb (#38620)
                    '5': 'v1.7'  // cb26661fe4faf26386703180a9045e6ac6d157df (#40364)
                };
                const version = versionTable[versionNumber];
                if (!version) {
                    this._exceptionCallback(new pytorch.Error("Unsupported PyTorch Zip version '" + versionNumber + "'."));
                }
                const constants = this._entry('constants.pkl');
                this._format = (constants ? 'TorchScript' : 'PyTorch') + ' ' + (version || 'v-' + versionNumber.toString() );
            }
        }
        return this._format;
    }

    get producer() {
        return this.data ? this._producer : '';
    }

    get name() {
        return this._name;
    }

    get littleEndian() {
        return true;
    }

    get type() {
        this._load();
        return this._type;
    }

    get data() {
        this._load();
        return this._data;
    }

    get constants() {
        if (this._constants === undefined) {
            this._constants = [];
            const entry = this._entry('constants.pkl');
            if (entry && entry.data) {
                this._constants = this._unpickle(entry.data, this._storage('constants'));
                for (let i = 0; i < this._constants.length; i++) {
                    this._constants[i].__variable__ = 'CONSTANTS.c' + i.toString();
                }
            }
        }
        return this._constants;
    }

    get execution() {
        if (this._execution === undefined) {
            const sources = new Map();
            for (const entry of this._entries) {
                if (entry.name.startsWith(this._prefix + 'code')) {
                    const file = entry.name.substring(this._prefix.length);
                    if (sources.has(file)) {
                        throw new pytorch.Error("Duplicate source file '" + file + "'.");
                    }
                    sources.set(file, entry.data);
                }
            }
            this._execution = new pytorch.Container.Zip.Execution(sources, this._exceptionCallback, this._metadata);
            const constants = {};
            for (let i = 0; i < this.constants.length; i++) {
                constants['c' + i.toString()] = this.constants[i];
            }
            this._execution.context.set('CONSTANTS', constants);
        }
        return this._execution;
    }

    _entry(name) {
        return this._entries.find((entry) => entry.name == this._prefix + name);
    }

    _load() {
        if (this._data === undefined) {
            this._data = null;
            const dataEntry = this._entry('data.pkl');
            if (dataEntry && dataEntry.data) {
                this._data = this._unpickle(dataEntry.data, this._storage('data'));
            }
            else {
                const modelEntry = this._entry('model.json');
                if (modelEntry) {
                    const model = JSON.parse(this._utf8Decoder.decode(modelEntry.data));
                    this._producer = model.producerName + (model.producerVersion ? ' v' + model.producerVersion : '');
                    this._data = model.mainModule || {};
                    this._name = this._data.name || '';
                    if (this._data.torchscriptArena) {
                        this._torchscriptArena = this._data.torchscriptArena.key;
                    }
                    const queue = [ this._data ];
                    const entries = new Map();
                    for (const entry of this._entries) {
                        entries.set(entry.name, entry.data);
                    }
                    const tensorTypeMap = new Map([
                        [ 'FLOAT', 'Float' ],
                        [ 'FLOAT16', 'Half' ],
                        [ 'DOUBLE', 'Double' ],
                        [ 'INT8', 'Char' ],
                        [ 'INT32', 'Int' ],
                        [ 'INT64', 'Long' ]
                    ]);
                    const constants = model.tensors || [];
                    this._constants = constants.map((constant) => {
                        const key = this._prefix + constant.data.key;
                        if (!tensorTypeMap.has(constant.dataType)) {
                            throw new pytorch.Error("Unknown tensor data type '" + constant.dataType + "'.");
                        }
                        const type = tensorTypeMap.get(constant.dataType);
                        const shape = constant.dims ? constant.dims.map((dim) => parseInt(dim, 10)) : null;
                        const storage_type = this.execution.type('torch.' + type + 'Storage');
                        const size = shape && shape.length > 0 ? shape.reduce((a, b) => a * b) : 1;
                        const offset = parseInt(constant.offset, 10) || 0;
                        const storage = new storage_type([ size ]);
                        const itemsize = storage.dtype.itemsize();
                        const buffer = entries.get(key);
                        const length = size * itemsize;
                        const data = buffer.slice(offset, offset + length);
                        storage._set_cdata(data);
                        const tensor = this.execution.invoke('torch._utils._rebuild_tensor', [ storage, 0, shape, 0 ]);
                        tensor.name = constant.data.key;
                        return tensor;
                    });
                    this._attributes = [];
                    const attributesEntry = this._entry('attributes.pkl');
                    if (attributesEntry && attributesEntry.data) {
                        this._attributes.push(...new python.Unpickler(attributesEntry.data).load((name, args) => this.execution.invoke(name, args)));
                    }
                    while (queue.length > 0) {
                        const module = queue.shift();
                        if (!module.__class__) {
                            module.__class__ = {
                                __module__: 'torch.nn.modules.module',
                                __name__: 'Module'
                            };
                        }
                        if (module.name) {
                            module.__id__ = module.name;
                        }
                        if (module.submodules) {
                            for (const submodule of module.submodules) {
                                module[submodule.name] = submodule;
                                submodule.__parent__ = module;
                                queue.push(submodule);
                            }
                            delete module.submodules;
                        }
                        const attributes = [];
                        if (module.attributes) {
                            attributes.push(...module.attributes);
                            delete module.attributes;
                        }
                        const parameters = [];
                        if (module.parameters) {
                            parameters.push(...module.parameters);
                            delete module.parameters;
                        }
                        if (module.arguments) {
                            parameters.push(...module.arguments);
                            delete module.arguments;
                        }
                        for (const parameter of parameters) {
                            const tensor = this._constants[parameter.tensorId];
                            module[parameter.name] = tensor;
                            if (!parameter.__class__) {
                                parameter.__class__ = {
                                    __module__: 'torch',
                                    __name__: 'Tensor'
                                };
                            }
                        }
                        for (const attribute of attributes) {
                            module[attribute.name] = this._attributes[attribute.id];
                        }
                    }
                }
            }
            if (this.format.startsWith('TorchScript ')) {
                this._type = 'script';
            }
            else {
                const obj = this._data;
                const root = pytorch.Utility.findModule(obj);
                if (root) {
                    this._type = 'module';
                    this._data = root;
                }
                else {
                    const weights = pytorch.Utility.findWeights(obj);
                    if (weights) {
                        this._type = 'weights';
                        this._data = weights;
                    }
                    else {
                        throw new pytorch.Error('File does not contain root module or state dictionary.');
                    }
                }
            }
        }
    }

    _unpickle(data, storage_map) {
        const loaded_storages = new Map();
        const persistent_load = (saved_id) => {
            const typename = saved_id.shift();
            if (typename !== 'storage') {
                throw new pytorch.Error("Unknown persistent load type '" + typename + "'.");
            }
            const data_type = this.execution.type(saved_id.shift());
            const root_key = saved_id.shift();
            /* const location = */ saved_id.shift();
            const size = saved_id.shift();
            if (!loaded_storages.has(root_key)) {
                const storage = new data_type(size);
                storage._set_cdata(storage_map.get(root_key));
                loaded_storages.set(root_key, storage);
            }
            const storage = loaded_storages.get(root_key);
            const view_metadata = saved_id.shift();
            if (view_metadata) {
                const view_key = view_metadata.shift();
                view_metadata.shift(); // view_offset
                view_metadata.shift(); // view_size
                let view = null;
                if (loaded_storages.has(view_key)) {
                    view = loaded_storages.get(root_key);
                }
                else {
                    view = null; // storage.slice(view_offset, view_offset + view_size);
                    loaded_storages.set(view_key, view);
                }
                return view;
            }
            return storage;
        };
        return new python.Unpickler(data).load((name, args) => this.execution.invoke(name, args), persistent_load);
    }

    _storage(dirname) {
        const map = new Map();
        const prefix = this._prefix + dirname + '/';
        for (const entry of this._entries) {
            if (entry.name.startsWith(prefix)) {
                const key = entry.name.substring(prefix.length);
                map.set(key, entry.data);
            }
        }
        return map;
    }

    trace() {
        this._inputs = [];
        this._outputs = [];
        this.execution.reset();
        if (this._torchscriptArena) {
            const program = this.execution.parse(this._torchscriptArena);
            for (const statement of program.body) {
                if (statement.type == 'def') {
                    const self = this;
                    const globals = this.execution.context;
                    const func = {
                        __class__: this.execution.context.scope.builtins.function,
                        __name__: statement.name,
                        __code__: statement,
                        __call__: function(args) {
                            return self.execution.apply(this.__code__, args, globals);
                        }
                    };
                    this.data[statement.name] = func;
                }
            }
        }
        if (this.data.forward) {
            const args = [ this.data ]; // self
            if (this.data.forward.__code__ && this.data.forward.__code__.parameters) {
                for (const parameter of this.data.forward.__code__.parameters) {
                    const defaultValue = (type, name) => {
                        if (type.type === 'type' && type.name.type) {
                            switch (type.name.value) {
                                case 'Tensor': {
                                    const tensor = this.execution.invoke('torch.Tensor', []);
                                    tensor.__variable__ = name;
                                    tensor.__origin__ = 'graph-input';
                                    return tensor;
                                }
                                case 'Tuple':
                                    return type.arguments.map((type, index) => defaultValue(type, name + '[' + index.toString() + ']'));
                                case 'List':
                                    return type.arguments.map((type, index) => defaultValue(type, name + '[' + index.toString() + ']' ));
                                case 'Dict':
                                    return {};
                                case 'int':
                                    return 0;
                                case 'float':
                                    return 0.0;
                                case 'bool':
                                    return false;
                                case 'Optional':
                                    return undefined;
                            }
                        }
                        throw new pytorch.Error("Unknown function parameter type '" + JSON.stringify(type) + "'.");
                    };
                    if (parameter.name !== 'self') {
                        const type = parameter.parameterType;
                        const value = defaultValue(type, parameter.name);
                        if (pytorch.Utility.isTensor(value)) {
                            value.__variable__ = parameter.name;
                            value.__origin__ = 'graph-input';
                            this._inputs.push(parameter.name);
                        }
                        args.push(value);
                    }
                }
            }
            const result = this.data.forward.__call__(args);
            if (Array.isArray(result)) {
                for (const output of result) {
                    if (pytorch.Utility.isTensor(output)) {
                        this._outputs.push(output.__variable__);
                    }
                }
            }
            else if (pytorch.Utility.isTensor(result)) {
                this._outputs.push(result.__variable__);
            }
            else if (Object(result) === result) {
                for (const key of Object.keys(result)) {
                    const value = result[key];
                    if (Array.isArray(value)) {
                        for (const output of value) {
                            if (pytorch.Utility.isTensor(output)) {
                                this._outputs.push(output.__variable__);
                            }
                        }
                    }
                    else if (pytorch.Utility.isTensor(value)) {
                        this._outputs.push(value.__variable__);
                    }
                }
            }
            this._nodes = this.execution.nodes;
            return true;
        }
        throw new pytorch.Error("Module 'forward' not implemented.");
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

pytorch.Container.Zip.Execution = class extends pytorch.Execution {

    constructor(sources, exceptionCallback, metadata) {
        super(sources, exceptionCallback);
        this._metadata = metadata;
        this.reset();
    }

    reset() {
        this._nodes = [];
        this._variableIndex = 0;
    }

    get nodes() {
        return this._nodes;
    }

    call(target, name, args, context) {
        let resolvedTarget = pytorch.Utility.target(target);
        let outputTypes = null;
        if (resolvedTarget && resolvedTarget + '.' + name === 'ops.prim.NumToTensor' &&
            args.length === 1 && args[0].type === 'call' && args[0].target.member.type == 'id') {
            const innerCall = args[0];
            resolvedTarget = pytorch.Utility.target(innerCall.target.target);
            name = innerCall.target.member.value;
            args = innerCall.arguments;
            outputTypes = [ 'int64' ];
        }
        if (resolvedTarget) {
            const type = resolvedTarget + '.' + name;
            // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
            let schemas = this._metadata.type(type);
            if (schemas) {
                schemas = !Array.isArray(schemas) ? [ schemas ] : schemas;
                const evalArgs = args.map((argument) => argument.type === '=' && argument.target && argument.target.type === 'id' ? this.expression(argument.expression, context) : this.expression(argument, context));
                for (const schema of schemas) {
                    const copyArgs = Array.prototype.slice.call(args);
                    const copyEvalArgs = Array.prototype.slice.call(evalArgs);
                    const node = {
                        type: schema.name,
                        inputs: [],
                        attributes: [],
                        outputs: []
                    };
                    const referencedParameters = [];
                    let next = false;
                    const parameters = Array.prototype.slice.call(schema.inputs || []).concat(Array.prototype.slice.call(schema.attributes || []));
                    while (copyEvalArgs.length > 0) {

                        if (parameters.length <= 0) {
                            next = true;
                            break;
                        }

                        const paramsBase = copyEvalArgs[0];
                        if (paramsBase && paramsBase.__class__ && paramsBase.__class__.__module__ === '__torch__.torch.classes.quantized') {
                            switch (paramsBase.__class__.__name__) {
                                case 'Conv2dPackedParamsBase':
                                case 'Conv3dPackedParamsBase': {
                                    copyArgs.shift();
                                    copyEvalArgs.shift();
                                    copyArgs.unshift({ type: null });
                                    copyEvalArgs.unshift(paramsBase.bias);
                                    copyArgs.unshift({ type: null });
                                    copyEvalArgs.unshift(paramsBase.weight);
                                    break;
                                }
                                case 'LinearPackedParamsBase': {
                                    copyArgs.shift();
                                    copyEvalArgs.shift();
                                    copyArgs.unshift({ type: null });
                                    copyEvalArgs.unshift(paramsBase.bias);
                                    copyArgs.unshift({ type: null });
                                    copyEvalArgs.unshift(paramsBase.weight);
                                    break;
                                }
                                default:
                                    throw new pytorch.Error("Unsupported type '" + paramsBase.__name__ + "'.");
                            }
                        }
                        const op_context = copyEvalArgs[0];
                        if (op_context && op_context.__class__ && op_context.__class__.__module__ === '__torch__.torch.classes.xnnpack') {
                            switch (op_context.__class__.__name__) {
                                case 'LinearOpContext':
                                case 'Conv2dOpContext':
                                    copyArgs.shift();
                                    copyEvalArgs.shift();
                                    for (const key of Object.keys(op_context).filter((key) => Number.isInteger(parseInt(key, 10)))) {
                                        copyArgs.push({ type: null });
                                        copyEvalArgs.push(op_context[key]);
                                    }
                                    break;
                                default:
                                    throw new pytorch.Error("Unsupported type '" + paramsBase.__name__ + "'.");
                            }
                        }

                        if (copyArgs.every((arg) => arg.type === '=' && arg.target && arg.target.type === 'id') &&
                            parameters.every((parameter) => parameter.type !== 'Tensor' && parameter.type !== 'Tensor[]')) {
                            const map = new Map();
                            for (const parameter of parameters) {
                                map.set(parameter.name, parameter);
                            }
                            while (copyArgs.length > 0) {
                                const argument = copyArgs.shift();
                                const value = copyEvalArgs.shift();
                                const parameter = map.get(argument.target.value);
                                if (!parameter) {
                                    next = true;
                                    break;
                                }
                                if (!pytorch.Utility.isType(value, parameter.type)) {
                                    if (parameter.optional) {
                                        continue;
                                    }
                                    next = true;
                                    break;
                                }
                                node.attributes.push({ name: parameter.name, value: value });
                            }
                            continue;
                        }
                        if (next) {
                            break;
                        }

                        const parameter = parameters.shift();
                        const argument = copyEvalArgs[0];

                        switch (parameter.type) {
                            case 'Tensor': {
                                if (Array.isArray(argument) || (!pytorch.Utility.isTensor(argument) && argument !== null && argument !== undefined)) {
                                    if (parameter.optional) {
                                        if (argument === undefined) {
                                            copyArgs.shift();
                                            copyEvalArgs.shift();
                                        }
                                        continue;
                                    }
                                    next = true;
                                    break;
                                }
                                copyArgs.shift();
                                copyEvalArgs.shift();
                                const item = (argument === null || argument === undefined) ? {} : argument;
                                item.__variable__ = item.__variable__ || this.variable();
                                const inputs = [];
                                inputs.push({ id: item.__variable__ });
                                referencedParameters.push(item);
                                node.inputs.push(inputs);
                                break;
                            }
                            case 'Tensor[]': {
                                const argument = copyEvalArgs[0];
                                if (!Array.isArray(argument) || !argument.every((item) => pytorch.Utility.isTensor(item) || item === null)) {
                                    if (parameter.optional) {
                                        continue;
                                    }
                                    next = true;
                                    break;
                                }
                                copyArgs.shift();
                                copyEvalArgs.shift();
                                const inputs = [];
                                for (let item of argument) {
                                    if (item === null) {
                                        item = {};
                                    }
                                    item.__variable__ = item.__variable__ || this.variable();
                                    inputs.push({ id: item.__variable__ });
                                    referencedParameters.push(item);
                                }
                                node.inputs.push(inputs);
                                break;
                            }
                            default: {
                                const arg = copyArgs[0];
                                if (!pytorch.Utility.isType(argument, parameter.type) && argument !== null) {
                                    if (parameter.optional) {
                                        continue;
                                    }
                                    next = true;
                                    break;
                                }
                                if (arg.type !== '=') {
                                    copyArgs.shift();
                                    copyEvalArgs.shift();
                                    node.attributes.push({ name: parameter.name, value: argument });
                                }
                                else {
                                    throw new pytorch.Error('Expected named argument.');
                                }
                                break;
                            }
                        }
                        if (next) {
                            break;
                        }
                    }
                    if (next) {
                        continue;
                    }
                    const result = [];
                    for (let i = 0; i < schema.outputs.length; i++) {
                        const parameter = schema.outputs[i];
                        switch (parameter.type) {
                            case 'Tensor': {
                                const parameter = this.invoke('torch.Tensor', []);
                                parameter.__origin__ = type;
                                if (i === 0) {
                                    switch (type) {
                                        case 'torch.cat':
                                        case 'torch.conv2d':
                                        case 'torch.dropout':
                                        case 'torch.flatten':
                                        case 'torch.max_pool2d':
                                        case 'torch.adaptive_avg_pool2d':
                                        case 'torch.avg_pool2d':
                                        case 'torch.quantize_per_tensor':
                                        case 'torch.relu_':
                                        case 'torch.hardtanh_':
                                        case 'torch.unsqueeze':
                                        case 'torch.slice': {
                                            parameter.resize_([ NaN, NaN, NaN, NaN ]);
                                            break;
                                        }
                                        case 'torch.conv3d': {
                                            parameter.resize_([ NaN, NaN, NaN, NaN, NaN ]);
                                            break;
                                        }
                                        case 'torch.embedding': {
                                            parameter.resize_([ NaN, NaN, NaN ]);
                                            break;
                                        }
                                        case 'torch.add':
                                        case 'torch.batch_norm':
                                        case 'torch.relu': {
                                            const input = this.expression(args[0], context);
                                            if (pytorch.Utility.isTensor(input) && Array.isArray(input.size())) {
                                                parameter.resize_(input.size());
                                            }
                                            break;
                                        }
                                        case 'torch.ones':
                                        case 'torch.zeros':
                                        case 'torch.zeros_like': {
                                            parameter.resize_(this.expression(args[0], context));
                                            break;
                                        }
                                        case 'torch.new_full': {
                                            parameter.resize_(this.expression(args[1], context));
                                            break;
                                        }
                                        case 'ops.quantized.cat':
                                        case 'ops.quantized.cat_relu':
                                        case 'ops.quantized.linear':
                                        case 'ops.quantized.conv2d':
                                        case 'ops.quantized.conv2d_relu':
                                        case 'ops.quantized.add_relu':
                                            parameter.resize_([ NaN, NaN, NaN, NaN ]);
                                            break;
                                    }
                                }
                                parameter.__variable__ = this.variable();
                                result.push(parameter);
                                node.outputs.push([ { id: parameter.__variable__ } ]);
                                break;
                            }
                            case 'Tensor[]': {
                                let count = 1;
                                switch (type) {
                                    case 'torch.chunk':
                                        count = node.attributes.filter((attribute) => attribute.name == 'chunks')[0].value;
                                        break;
                                    case 'torch.meshgrid':
                                        count = node.inputs[0].length;
                                        break;
                                }
                                const tensors = [];
                                const outputs = [];
                                for (let i = 0; i < count; i ++) {
                                    const tensor = this.invoke('torch.Tensor', []);
                                    tensor.__origin__ = type;
                                    tensor.__variable__ = this.variable();
                                    tensors.push(tensor);
                                    outputs.push({ id: tensor.__variable__ });
                                }
                                result.push(tensors);
                                node.outputs.push(outputs);
                                break;
                            }
                            default: {
                                if (!outputTypes || schema.outputs.length !== 1 || schema.outputs[0].type !== outputTypes[0]) {
                                    next = true;
                                    break;
                                }
                                const tensor = this.invoke('torch.Tensor', []);
                                tensor.resize_([]);
                                tensor.__origin__ = type;
                                tensor.__variable__ = this.variable();
                                result.push(tensor);
                                node.outputs.push([ { id: tensor.__variable__ } ]);
                                break;
                            }
                        }
                    }
                    if (next) {
                        continue;
                    }
                    for (const parameter of referencedParameters) {
                        parameter.__count__ = (parameter.__count__ || 0) + 1;
                    }
                    this.push(node);
                    if (result.length > 1) {
                        return result;
                    }
                    return result[0];
                }
            }
        }
        return super.call(target, name, args, context);
    }

    block(statements, context) {
        statements = Array.prototype.slice.call(statements);
        while (statements.length > 0) {
            if (statements.length > 1) {
                const assign = statements[0];
                const condition = statements[1];
                // _x = torch.ne(torch.len(torch.size(input)), 5)
                // if _x:
                //   ops.prim.RaiseException(...)
                if (assign.type === '=' &&
                    condition.type === 'if' &&
                    pytorch.Utility.isEqual(assign.target, condition.condition) &&
                    pytorch.Utility.isCall(assign.expression, 'torch.ne', 2) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0], 'torch.len', 1) &&
                    pytorch.Utility.isCall(assign.expression.arguments[0].arguments[0], 'torch.size', 1) &&
                    condition.then.statements.length == 1 &&
                    pytorch.Utility.isCall(condition.then.statements[0], 'ops.prim.RaiseException', 1)) {
                    const tensor = this.expression(assign.expression.arguments[0].arguments[0].arguments[0], context);
                    if (tensor && tensor.size) {
                        const number = this.expression(assign.expression.arguments[1], context);
                        const size = tensor.size();
                        if (size && size.length && size.length !== number &&
                            size.every((item) => isNaN(item)) && number >= 3 && number <= 5) {
                            if (tensor.__origin__ === 'torch.quantize_per_tensor') {
                                tensor.resize_(Array(number).fill(NaN));
                            }
                        }
                    }
                }
            }
            const statement = statements.shift();
            // input_shape = torch.slice(torch.size(x), -2, 9223372036854775807, 1)
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                pytorch.Utility.isCall(statement.expression.arguments[0], 'torch.size', 1)) {
                const tensor = this.expression(statement.expression.arguments[0].arguments[0], context);
                if (tensor && tensor.shape === undefined) {
                    tensor.resize_([ 1, 3, 299, 299 ]);
                }
            }
            // torch.slice(ops.prim.shape(input), 0, 2, 1)
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.slice', 4) &&
                pytorch.Utility.isCall(statement.expression.arguments[0], 'ops.prim.shape', 1)) {
                const tensor = this.expression(statement.expression.arguments[0].arguments[0], context);
                if (tensor && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // _3 = torch.le(xxxx, torch.dim(f0))
            if (statement.type === '=' &&
                pytorch.Utility.isCall(statement.expression, 'torch.le', 2) &&
                pytorch.Utility.isCall(statement.expression.arguments[1], 'torch.dim', 1)) {
                const tensor = this.expression(statement.expression.arguments[1].arguments[0], context);
                if (tensor && tensor.__origin__ === 'graph-input' && tensor.shape === undefined) {
                    tensor.resize_([ NaN, NaN, NaN, NaN ]);
                }
            }
            // if torch.ne(torch.dim(image), 3):
            //   xxxx
            //   ops.prim.RaiseException(_7)
            if (statement.type === 'if' &&
                pytorch.Utility.isCall(statement.condition, 'torch.ne', 2) &&
                pytorch.Utility.isCall(statement.condition.arguments[0], 'torch.dim', 1) &&
                statement.then.statements.length > 0 &&
                pytorch.Utility.isCall(statement.then.statements.slice(-1).pop(), 'ops.prim.RaiseException', 1)) {
                const tensor = this.expression(statement.condition.arguments[0].arguments[0], context);
                const size = this.expression(statement.condition.arguments[1], context);
                if (tensor && Number.isInteger(size) && size < 10) {
                    tensor.resize_(Array.isArray(tensor.shape) && tensor.shape.length > size ? tensor.shape.slice(-size) : Array(size).fill(NaN));
                }
            }
            const value = this.statement(statement, context);
            if (value !== undefined) {
                return value;
            }
        }
    }

    push(node) {
        this._nodes.push(node);
    }

    variable() {
        this._variableIndex++;
        return this._variableIndex.toString();
    }
};

pytorch.ScalarType = {
    uint8: 0,
    int8: 1,
    int16: 2,
    int32: 3,
    int64: 4,
    float16: 5,
    float32: 6,
    float64: 7,
    complex32: 8,
    complex64: 9,
    complex128: 10,
    boolean: 11,
    qint8: 12,
    quint8: 13,
    qint32: 14,
    bfloat16: 15,
    quint4x2: 16
};

pytorch.MemoryFormat = {
    Contiguous: 0,
    Preserve: 1,
    ChannelsLast: 2,
    ChannelsLast3d: 3
};

pytorch.Layout = {
    Strided: 0,
    Sparse: 1,
    Mkldnn: 2
};

pytorch.Utility = class {

    static getScalarType(scalarType) {
        if (!pytorch.Utility._scalarTypes) {
            pytorch.Utility._scalarTypes = [
                { name: 'uint8', itemsize: 1 },
                { name: 'int8', itemsize: 1 },
                { name: 'int16', itemsize: 2 },
                { name: 'int32', itemsize: 4 },
                { name: 'int64', itemsize: 8 },
                { name: 'float16', itemsize: 2 },
                { name: 'float32', itemsize: 4 },
                { name: 'float64', itemsize: 8 },
                { name: 'complex32', itemsize: 4 },
                { name: 'complex64', itemsize: 8 },
                { name: 'complex128', itemsize: 16 },
                { name: 'boolean', itemsize: 1 },
                { name: 'qint8', itemsize: 1 },
                { name: 'quint8', itemsize: 1 },
                { name: 'qint32', itemsize: 4 },
                { name: 'bfloat16', itemsize: 2 },
                { name: 'quint4x2' }
            ];
        }
        if (scalarType < pytorch.Utility._scalarTypes.length) {
            return pytorch.Utility._scalarTypes[scalarType];
        }
        throw new pytorch.Error("Unknown scalar type '" + scalarType + "'.");
    }

    static target(expression) {
        if (expression.type == 'id') {
            return expression.value;
        }
        if (expression.type == '.') {
            return pytorch.Utility.target(expression.target) + '.' + pytorch.Utility.target(expression.member);
        }
        return null;
    }

    static isTensor(obj) {
        if (obj && obj.__class__) {
            switch (obj.__class__.__module__) {
                case 'torch':
                case 'torch.cuda':
                    return obj.__class__.__name__.endsWith('Tensor');
                case 'torch.nn.parameter':
                    return obj.__class__.__name__ === 'Parameter';
            }
        }
        return false;
    }

    static toTensor(obj) {
        if (obj && obj.__class__) {
            switch (obj.__class__.__module__) {
                case 'torch':
                case 'torch.cuda':
                    return obj.__class__.__name__.endsWith('Tensor') ? obj : null;
                case 'torch.nn.parameter':
                    return obj.__class__.__name__ === 'Parameter' ? obj.data : null;
            }
        }
        return null;
    }

    static isType(obj, type) {
        switch (type) {
            case 'Tensor':
                return !Array.isArray(obj) && (pytorch.Utility.isTensor(obj) || obj === null);
            case 'Tensor[]':
                return Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => pytorch.Utility.isTensor(tensor) || tensor === null);
            case 'Scalar':
                return (obj !== null && obj !== Object(obj)) || (pytorch.Utility.isTensor(obj) && Array.isArray(obj.size()) && obj.size().length === 0);
            case 'boolean':
                return obj === true || obj === false;
            case 'int64':
                return Number.isInteger(obj) || isNaN(obj) || obj instanceof base.Int64;
            case 'int64[]':
                return Array.isArray(obj) && obj.every((item) => Number.isInteger(item) || Number.isNaN(item) || item === undefined);
            case 'int64[1]':
                return pytorch.Utility.isType(obj, 'int64') || pytorch.Utility.isType(obj, 'int64[]');
            case 'float32':
            case 'float64':
                return obj !== null && obj !== Object(obj);
            case 'Layout':
            case 'ScalarType':
            case 'MemoryFormat':
                return Number.isInteger(obj) || obj === null;
            case 'Device':
                return obj === null || obj === Object(obj);
        }
        return true;
    }

    static isCall(expression, name, size) {
        if (expression.type === 'call' &&
            expression.arguments.length === size &&
            pytorch.Utility.target(expression.target) === name) {
            return true;
        }
        return false;
    }

    static isEqual(a, b) {
        return (a.type === 'id' && b.type === 'id' && a.value === b.value);
    }

    static findModule(root) {
        if (root) {
            const keys = [ '', 'model', 'net' ];
            for (const key of keys) {
                const obj = key === '' ? root : root[key];
                if (obj && obj._modules) {
                    return obj;
                }
            }
        }
        return null;
    }

    static findWeights(root) {
        if (!root) {
            return null;
        }
        if (root instanceof Map) {
            const obj = {};
            for (const pair of root) {
                const key = pair[0];
                const value = pair[1];
                obj[key] = value;
            }
            root = obj;
        }
        const keys = root && !Array.isArray(root) ? Object.keys(root) : [];
        if (keys.length > 1) {
            keys.splice(0, keys.length);
        }
        keys.push(...[
            'state_dict', 'state', 'model_state', 'model', 'model_state_dict', 'model_dict', 'net_dict', 'params', 'generator',
            'discriminator', 'g_state', 'network', 'net', 'netG', 'net_states', 'state_dict_stylepredictor', 'state_dict_ghiasi', ''
        ]);
        for (const key of keys) {
            const obj = key === '' ? root : root[key];
            let graphs = null;
            graphs = graphs || pytorch.Utility._convertTensor(obj);
            graphs = graphs || pytorch.Utility._convertObjectList(obj);
            graphs = graphs || pytorch.Utility._convertStateDict(obj);
            if (graphs) {
                return graphs;
            }
        }
        return null;
    }

    static _convertTensor(obj) {
        if (obj && pytorch.Utility.isTensor(obj)) {
            const layers = [];
            const argument = { id: '', value: obj };
            const parameter = { name: 'value', arguments: [ argument ] };
            layers.push({ states: [ parameter ] });
            return [ { layers: layers } ];
        }
        return null;
    }

    static _convertObjectList(list) {
        if (list && Array.isArray(list) && list.every((obj) => obj.__class__ && Object.keys(obj).filter((key) => pytorch.Utility.isTensor(obj[key]).length > 0))) {
            const layers = [];
            for (const obj of list) {
                const layer = { type: obj.__class__.__module__ + '.' + obj.__class__.__name__, states: [], attributes: [] };
                for (const key of Object.keys(obj)) {
                    const value = obj[key];
                    if (pytorch.Utility.isTensor(value)) {
                        layer.states.push({ name: key, arguments: [ { id: '', value: value } ] });
                    }
                    else {
                        layer.attributes.push({ name: key, value: value });
                    }
                }
                layers.push(layer);
            }
            return [ { layers: layers } ];
        }
        return null;
    }

    static _convertStateDict(obj) {
        const clean = (obj) => {
            if (obj && Array.isArray(obj)) {
                return obj;
            }
            if (obj && obj instanceof Map) {
                return obj;
            }
            if (obj && Object(obj) === obj) {
                const value = {};
                for (const key of Object.keys(obj)) {
                    if (key.indexOf('optim') !== -1) {
                        const optimizer = obj[key];
                        if (optimizer.state && optimizer.param_groups) {
                            continue;
                        }
                    }
                    value[key] = obj[key];
                }
                return value;
            }
            return obj;
        };
        const validate = (map) => {
            let tensor = false;
            if (map && map instanceof Map) {
                for (const pair of map) {
                    const key = pair[0];
                    const value = pair[1];
                    if (key.split('.').pop() === '_metadata') {
                        continue;
                    }
                    if (pytorch.Utility.isTensor(value)) {
                        tensor = true;
                        continue;
                    }
                    else if (value && Array.isArray(value) && value.every((item) => pytorch.Utility.isTensor(item))) {
                        tensor = true;
                        continue;
                    }
                    else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                        continue;
                    }
                    return false;
                }
            }
            return tensor;
        };
        const flatten = (obj) => {
            if (!obj || Array.isArray(obj)) {
                return null;
            }
            if (obj instanceof Map) {
                if (validate(obj)) {
                    return obj;
                }
                return null;
            }
            if (Object(obj) !== obj) {
                return null;
            }
            const map = new Map(Object.keys(obj).map((key) => [ key, obj[key] ]));
            if (validate(map)) {
                return map;
            }
            map.clear();
            for (const key of Object.keys(obj)) {
                const value = flatten(obj[key]);
                if (value && value instanceof Map) {
                    for (const pair of value) {
                        map.set(key + '.' + pair[0], pair[1]);
                    }
                    continue;
                }
                return null;
            }
            return map;
        };
        if (!obj) {
            return null;
        }
        obj = clean(obj);
        const map = new Map();
        if (Array.isArray(obj) && obj.every((item) => validate(item))) {
            for (let i = 0; i < obj.length; i++) {
                map.set(i.toString(), flatten(obj[i]));
            }
        }
        else if (obj instanceof Map && validate(obj)) {
            map.set('', flatten(obj));
        }
        else if (Object(obj) === obj && Object.keys(obj).every((key) => validate(obj[key]))) {
            for (const key of Object.keys(obj)) {
                map.set(key, obj[key]);
            }
        }
        else if (Object(obj) === obj && Object.keys(obj).every((key) => pytorch.Utility.isTensor(obj[key]))) {
            map.set('', new Map(Object.keys(obj).map((key) => [ key, obj[key] ])));
        }
        else {
            const value = flatten(obj);
            if (value) {
                map.set('', value);
            }
        }
        if (map.size > 0) {
            const graphs = [];
            for (const pair of map) {
                const graph_key = pair[0];
                const layer_map = pair[1];
                const layers = new Map();
                for (const item of layer_map) {
                    const key = item[0];
                    const value = item[1];
                    let layerName = '';
                    let parameter = '';
                    const keys = key.split('.');
                    if (keys[keys.length - 1] === '_metadata') {
                        continue;
                    }
                    if (keys.length >= 2 && keys[keys.length - 2] === '_packed_params') {
                        parameter = keys.slice(-2).join('.');
                        keys.pop();
                        keys.pop();
                    }
                    else {
                        parameter = keys.pop();
                        if (keys.length < 0) {
                            keys.push('');
                        }
                    }
                    layerName = keys.join('.');
                    if (!layers.has(layerName)) {
                        layers.set(layerName, { name: layerName, states: [], attributes: [] });
                    }
                    const layer = layers.get(layerName);
                    if (pytorch.Utility.isTensor(value)) {
                        layer.states.push({ name: parameter, arguments: [ { id: key, value: value } ] });
                        if (layer.name == '' && layer.states.length > 4) {
                            return null;
                        }
                    }
                    else if (value && Array.isArray(value) && value.every((item) => pytorch.Utility.isTensor(item))) {
                        layer.states.push({ name: parameter, arguments: value.map((item) => { return { id: '', value: item }; }) });
                    }
                    else if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
                        layer.attributes.push({ name: parameter, value: value });
                    }
                }
                graphs.push({
                    name: graph_key,
                    layers: layers.values()
                });
            }
            return graphs;
        }
        return null;
    }
};

pytorch.nnapi = {};

pytorch.nnapi.SerializedModel = class {

    constructor(serialized_model /*, buffer_ptrs */) {
        const reader = new pytorch.nnapi.SerializedModel.BinaryReader(serialized_model);
        this.version = reader.int32();
        if (this.version !== 1) {
            throw new pytorch.Error('Invalid NNAPI serialized model version.');
        }
        const operands = new Array(reader.int32());
        const values = new Array(reader.int32());
        this.operations = new Array(reader.int32());
        this.inputs = new Array(reader.int32());
        this.outputs = new Array(reader.int32());
        const types = new Map([
            [ 0, 'float32' ],
            [ 1, 'int32' ],
            [ 2, 'uint32' ],
            [ 3, 'float32' ],
            [ 4, 'int32*' ],
            [ 5, 'quant8_asymm*' ],
            [ 6, 'boolean' ],
            [ 7, 'quant16_symm*' ],
            [ 8, 'float16*' ],
            [ 9, 'boolean*' ],
            [ 10, 'float16' ],
            [ 11, 'quant8_symm_per_channel*' ],
            [ 12, 'quant16_asymm*' ],
            [ 13, 'quant8_symm*' ],
            [ 14, 'quant8_asymm_signed*' ],
            [ 16, 'model' ]
        ]);
        const operations = new Map([
            [ 0, 'ADD' ],
            [ 1, 'AVERAGE_POOL_2D' ],
            [ 2, 'CONCATENATION' ],
            [ 3, 'CONV_2D' ],
            [ 4, 'DEPTHWISE_CONV_2D' ],
            [ 5, 'DEPTH_TO_SPACE' ],
            [ 6, 'DEQUANTIZE' ],
            [ 7, 'EMBEDDING_LOOKUP' ],
            [ 8, 'FLOOR' ],
            [ 9, 'FULLY_CONNECTED' ],
            [ 10, 'HASHTABLE_LOOKUP' ],
            [ 11, 'L2_NORMALIZATION' ],
            [ 12, 'L2_POOL_2D' ],
            [ 13, 'LOCAL_RESPONSE_NORMALIZATION' ],
            [ 14, 'LOGISTIC' ],
            [ 15, 'LSH_PROJECTION' ],
            [ 16, 'LSTM' ],
            [ 17, 'MAX_POOL_2D' ],
            [ 18, 'MUL' ],
            [ 19, 'RELU' ],
            [ 20, 'RELU1' ],
            [ 21, 'RELU6' ],
            [ 22, 'RESHAPE' ],
            [ 23, 'RESIZE_BILINEAR' ],
            [ 24, 'RNN' ],
            [ 25, 'SOFTMAX' ],
            [ 26, 'SPACE_TO_DEPTH' ],
            [ 27, 'SVDF' ],
            [ 28, 'TANH' ],
            [ 29, 'BATCH_TO_SPACE_ND' ],
            [ 30, 'DIV' ],
            [ 31, 'MEAN' ],
            [ 32, 'PAD' ],
            [ 33, 'SPACE_TO_BATCH_ND' ],
            [ 34, 'SQUEEZE' ],
            [ 35, 'STRIDED_SLICE' ],
            [ 36, 'SUB' ],
            [ 37, 'TRANSPOSE' ],
            [ 38, 'ABS' ],
            [ 39, 'ARGMAX' ],
            [ 40, 'ARGMIN' ],
            [ 41, 'AXIS_ALIGNED_BBOX_TRANSFORM' ],
            [ 42, 'BIDIRECTIONAL_SEQUENCE_LSTM' ],
            [ 43, 'BIDIRECTIONAL_SEQUENCE_RNN' ],
            [ 44, 'BOX_WITH_NMS_LIMIT' ],
            [ 45, 'CAST' ],
            [ 46, 'CHANNEL_SHUFFLE' ],
            [ 47, 'DETECTION_POSTPROCESSING' ],
            [ 48, 'EQUAL' ],
            [ 49, 'EXP' ],
            [ 50, 'EXPAND_DIMS' ],
            [ 51, 'GATHER' ],
            [ 52, 'GENERATE_PROPOSALS' ],
            [ 53, 'GREATER' ],
            [ 54, 'GREATER_EQUAL' ],
            [ 55, 'GROUPED_CONV_2D' ],
            [ 56, 'HEATMAP_MAX_KEYPOINT' ],
            [ 57, 'INSTANCE_NORMALIZATION' ],
            [ 58, 'LESS' ],
            [ 59, 'LESS_EQUAL' ],
            [ 60, 'LOG' ],
            [ 61, 'LOGICAL_AND' ],
            [ 62, 'LOGICAL_NOT' ],
            [ 63, 'LOGICAL_OR' ],
            [ 64, 'LOG_SOFTMAX' ],
            [ 65, 'MAXIMUM' ],
            [ 66, 'MINIMUM' ],
            [ 67, 'NEG' ],
            [ 68, 'NOT_EQUAL' ],
            [ 69, 'PAD_V2' ],
            [ 70, 'POW' ],
            [ 71, 'PRELU' ],
            [ 72, 'QUANTIZE' ],
            [ 73, 'QUANTIZED_16BIT_LSTM' ],
            [ 74, 'RANDOM_MULTINOMIAL' ],
            [ 75, 'REDUCE_ALL' ],
            [ 76, 'REDUCE_ANY' ],
            [ 77, 'REDUCE_MAX' ],
            [ 78, 'REDUCE_MIN' ],
            [ 79, 'REDUCE_PROD' ],
            [ 80, 'REDUCE_SUM' ],
            [ 81, 'ROI_ALIGN' ],
            [ 82, 'ROI_POOLING' ],
            [ 83, 'RSQRT' ],
            [ 84, 'SELECT' ],
            [ 85, 'SIN' ],
            [ 86, 'SLICE' ],
            [ 87, 'SPLIT' ],
            [ 88, 'SQRT' ],
            [ 89, 'TILE' ],
            [ 90, 'TOPK_V2' ],
            [ 91, 'TRANSPOSE_CONV_2D' ],
            [ 92, 'UNIDIRECTIONAL_SEQUENCE_LSTM' ],
            [ 93, 'UNIDIRECTIONAL_SEQUENCE_RNN' ],
            [ 94, 'RESIZE_NEAREST_NEIGHBOR' ],
            [ 95, 'QUANTIZED_LSTM' ],
            [ 96, 'IF' ],
            [ 97, 'WHILE' ],
            [ 98, 'ELU' ],
            [ 99, 'HARD_SWISH' ],
            [ 100, 'FILL' ],
            [ 101, 'RANK' ],
        ]);
        for (let i = 0; i < operands.length; i++) {
            const type = reader.int32();
            operands[i] = {
                type: types.has(type) ? types.get(type) : type,
                dimensions: new Array(reader.uint32()),
                scale: reader.float32(),
                zero_point: reader.int32()
            };
        }
        for (let i = 0; i < values.length; i++) {
            values[i] = {
                index: reader.int32(),
                source_type: reader.int32(),
                source_length: reader.uint32()
            };
        }
        for (let i = 0; i < this.operations.length; i++) {
            const operation_type = reader.int32();
            this.operations[i] = {
                operation_type: operations.has(operation_type) ? operations.get(operation_type) : operation_type,
                inputs: new Array(reader.uint32()),
                outputs: new Array(reader.uint32())
            };
        }
        for (const operand of operands) {
            for (let i = 0; i< operand.dimensions.length; i++) {
                operand.dimensions[i] = reader.uint32();
            }
        }
        for (const value of values) {
            const operand = operands[value.index];
            switch (value.source_type) {
                case 0: { // immediate
                    switch (operand.type) {
                        case 'boolean':
                            operand.value = reader.byte() ? true : false;
                            reader.skip(3);
                            break;
                        case 'int32':
                            operand.value = reader.int32();
                            break;
                        case 'float32':
                            operand.value = reader.float32();
                            break;
                        case 'int32*':
                            operand.value = reader.read(value.source_length);
                            break;
                        default:
                            throw new pytorch.Error("Unsupported NNAPI operand type '" + operand.type.toString() + "'.");
                    }
                    break;
                }
                case 2: { // numbered buffer
                    if (value.source_length !== 12) {
                        throw new pytorch.Error('Invalid NNAPI numbered buffer source length.');
                    }
                    const number = reader.uint32();
                    const offset = reader.uint32();
                    const operand_length = reader.uint32();
                    operand.value = [ number, offset, operand_length ];
                    break;
                }
                case 3: { // numbered memory
                    throw new pytorch.Error('NNAPI numbered memory buffer not implemented.');
                }
                default: {
                    throw new pytorch.Error('Unsupported NNAPI value source type.');
                }
            }
        }
        for (const operation of this.operations) {
            for (let i = 0; i< operation.inputs.length; i++) {
                operation.inputs[i] = operands[reader.uint32()];
            }
            for (let i = 0; i< operation.outputs.length; i++) {
                operation.outputs[i] = operands[reader.uint32()];
            }
        }
        for (let i = 0; i< this.inputs.length; i++) {
            this.inputs[i] = operands[reader.uint32()];
        }
        for (let i = 0; i< this.outputs.length; i++) {
            this.outputs[i] = operands[reader.uint32()];
        }
        if (!reader.end()) {
            throw new pytorch.Error('Invalid NNAPI serialized model length.');
        }
    }
};

pytorch.nnapi.SerializedModel.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    end() {
        return this._position >= this._buffer.length;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new pytorch.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    read(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    int64() {
        const value = this.int32();
        if (this.int32() !== 0) {
            throw new pytorch.Error('Invalid int64 value.');
        }
        return value;
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }
};

pytorch.Metadata = class {

    static open(context) {
        if (pytorch.Metadata._metadata) {
            return Promise.resolve(pytorch.Metadata._metadata);
        }
        else {
            return context.request('pytorch-metadata.json', 'utf-8', null).then((data) => {
                pytorch.Metadata._metadata = new pytorch.Metadata(data);
                return pytorch.Metadata._metadata;
            }).catch(() => {
                pytorch.Metadata._metadata = new pytorch.Metadata(null);
                return pytorch.Metadata._metadata;
            });
        }
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            for (const item of items) {
                this._map.set(item.name, item);
                const index = item.name.indexOf(':');
                if (index !== -1) {
                    const name = item.name.substring(0, index);
                    if (!this._map.has(name)) {
                        this._map.set(name, []);
                    }
                    this._map.get(name).push(item.name);
                }
            }
        }
    }

    type(name) {
        const schema = this._map.get(name);
        if (schema) {
            return Array.isArray(schema) ? schema.map((name) => this._map.get(name)) : schema;
        }
        return null;
    }

    attribute(type, name) {
        const attributeName = type + ':' + name;
        if (!this._attributeCache.has(attributeName)) {
            this._attributeCache.set(attributeName, null);
            const schema = this.type(type);
            if (schema) {
                if (schema.inputs) {
                    for (const input of schema.inputs) {
                        this._attributeCache.set(type + ':' + input.name, input);
                    }
                }
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        this._attributeCache.set(type + ':' + attribute.name, attribute);
                    }
                }
            }
        }
        return this._attributeCache.get(attributeName);
    }
};

pytorch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading PyTorch model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = pytorch.ModelFactory;
}
