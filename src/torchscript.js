/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var torchscript = torchscript || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var marked = marked || require('marked');
var zip = zip || require('./zip');

torchscript.ModelFactory = class {

    match(context) {
        let identifier = context.identifier; 
        let extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'pt' || extension == 'pth' || extension == 'pkl' || extension == 'h5' || extension == 't7' ||
            extension == 'dms' || extension == 'model' || extension == 'ckpt' || identifier.endsWith('.pth.tar')) {
            if (torchscript.ModelFactory._openContainer(context.entries)) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./python').then((python) => {
            return host.require('./pickle').then((pickle) => {
                let identifier = context.identifier;
                try {
                    let container = torchscript.ModelFactory._openContainer(context.entries);
                    container.identifier = identifier;
                    container.constants = torchscript.ModelFactory._unpickle(host, identifier, pickle, container.constants, torchscript.ModelFactory._storage(container, 'constants'));
                    container.constants = (container.constants || []).map((tensor) => new torchscript.Tensor('pickle', tensor));
                    container.data = torchscript.ModelFactory._unpickle(host, identifier, pickle, container.data, torchscript.ModelFactory._storage(container, 'data'));
                    container.attributes = torchscript.ModelFactory._unpickle(host, identifier, pickle, container.attributes, null);
                    let textDecoder = new TextDecoder('utf-8');
                    if (container.version) {
                        container.version = JSON.parse(textDecoder.decode(container.version));
                    }
                    if (container.model) {
                        container.model = JSON.parse(textDecoder.decode(container.model));
                    }            
                    return torchscript.Metadata.open(host).then((metadata) => {
                        try {
                            return new torchscript.Model(metadata, host, python, container);
                        }
                        catch (error) {
                            host.exception(error, false);
                            let message = error && error.message ? error.message : error.toString();
                            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                            throw new torchscript.Error(message + " in '" + identifier + "'.");
                        }
                    });
                }
                catch (error) {
                    host.exception(error, false);
                    let message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    return Promise.reject(new torchscript.Error(message + " in '" + identifier + "'."));
                }
            });
        });
    }

    static _openContainer(entries) {
        if (entries && entries.length > 0) {
            let container = { };
            let version = entries.find((entry) => entry.name == 'version' || entry.name.endsWith('/version'));
            if (version) {
                container.entries = entries;
                container.prefix = version.name.substring(0, version.name.length - 7);
                let find = (name) => {
                    let entry = container.entries.find((entry) => entry.name == container.prefix + name);
                    return entry ? entry.data : null;
                }
                // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
                container.version = version.data;
                container.attributes = find('attribtues.pkl');
                container.constants = find('constants.pkl');
                container.data = find('data.pkl');
                container.model = find('model.json');
                if (container.version && (container.model || container.data)) {
                    return container;
                }
            }
        }
        return null;
    }

    static _storage(container, dirname) {
        let map = new Map();
        let prefix = container.prefix + dirname + '/';
        for (let entry of container.entries) {
            if (entry.name.startsWith(prefix)) {
                let key = entry.name.substring(prefix.length);
                map.set(key, entry.data);
            }
        }
        return map;
    }

    static _unpickle(host, identifier, pickle, data, storage_map) {
        if (!data) {
            return null;
        }
        let functionTable = {};
        functionTable['collections.OrderedDict'] = function(args) {
            let obj = [];
            obj.__setitem__ = function(key, value) {
                obj.push({ key: key, value: value });
            };
            if (args) {
                for (let arg of args) {
                    obj.__setitem__(arg[0], arg[1]);
                }
            }
            return obj;
        };
        functionTable['torch._utils._rebuild_tensor_v2'] = function (storage, storage_offset, size, stride, requires_grad, backward_hooks) {
            return {
                __type__: storage.__type__.replace('Storage', 'Tensor'),
                storage: storage,
                storage_offset: storage_offset,
                size: size,
                stride: stride,
                requires_grad:requires_grad,
                backward_hooks: backward_hooks
            };
        };
        let constructorTable = {};
        constructorTable['torch.ByteStorage'] = function (size) { 
            this.size = size; this.dataTypeSize = 1; this.dataType = 'uint8'; 
        };
        constructorTable['torch.CharStorage'] = function (size) { 
            this.size = size; this.dataTypeSize = 1; this.dataType = 'int8'; 
        };
        constructorTable['torch.ShortStorage'] = function (size) { 
            this.size = size; this.dataTypeSize = 2; this.dataType = 'int16';
        };
        constructorTable['torch.IntStorage'] = function (size) { 
            this.size = size; this.dataTypeSize = 4; this.dataType = 'int32';
        };
        constructorTable['torch.LongStorage'] = function (size) { 
            this.size = size; this.dataTypeSize = 8; this.dataType = 'int64';
        };
        constructorTable['torch.HalfStorage'] = function (size) {
            this.size = size; this.dataTypeSize = 2; this.dataType = 'float16';
        };
        constructorTable['torch.FloatStorage'] = function (size) {
            this.size = size; this.dataTypeSize = 4; this.dataType = 'float32';
        };
        constructorTable['torch.DoubleStorage'] = function (size) { 
            this.size = size; this.dataTypeSize = 8; this.dataType = 'float64';
        };
        let function_call = (name, args) => {
            let func = functionTable[name];
            if (func) {
                return func.apply(null, args);
            }
            let obj = { __type__: name };
            let constructor = constructorTable[name];
            if (constructor) {
                constructor.apply(obj, args);
            }
            else if (!name.startsWith('__torch__.')) {
                host.exception(new torchscript.Error("Unknown function '" + name + "' in '" + identifier + "'."), false);
            }
            return obj;
        };
        let deserialized_objects = new Map();
        let persistent_load = (saved_id) => {
            let typename = saved_id.shift();
            if (typename !== 'storage') {
                throw new torchscript.Error("Unknown persistent load type '" + typename + "'.");
            }
            let data_type = saved_id.shift();
            let root_key = saved_id.shift();
            saved_id.shift(); // location
            let size = saved_id.shift();
            let storage = null;
            if (deserialized_objects.has(root_key)) {
                storage = deserialized_objects.get(root_key);
            }
            else {
                storage = function_call(data_type, [ size ]);
                storage.data = storage_map.get(root_key);
                deserialized_objects[root_key] = storage;
            }
            let view_metadata = saved_id.shift();
            if (view_metadata) {
                let view_key = view_metadata.shift();
                view_metadata.shift(); // view_offset
                view_metadata.shift(); // view_size
                let view = deserialized_objects[view_key];
                if (!view) {
                    view = null; // storage.slice(view_offset, view_offset + view_size);
                    deserialized_objects[view_key] = view;
                }
                return view;
            }
            return storage;
        };
        return new pickle.Unpickler(data).load(function_call, persistent_load);
    }
};

torchscript.Model = class { 

    constructor(metadata, host, python, container) {
        this._format = 'TorchScript v' + container.version.toString();
        if (container.model) {
            if (container.producerName) {
                this._producer = container.producerName;
                if (container.producerVersion) {
                    this._producer = this._producer + ' v' + container.producerVersion;
                }
            }
            container.tensors = container.model.tensors.map((tensor) => {
                let key = container.prefix + tensor.data.key;
                let entry = container.entries.find((entry) => entry.name == key);
                return new torchscript.Tensor('json', { tensor: tensor, data: entry.data });
            });
            container.constants = container.tensors;
        }
        this._graphs = [];
        this._graphs.push(new torchscript.Graph(metadata, host, python, container));
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get graphs() {
        return this._graphs;
    }
};

torchscript.Graph = class {

    constructor(metadata, host, python, container) {
        if (container.model && container.model.mainModule) {
            this._name = container.model.mainModule.name;
        }
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        let mainModule = null;
        let context = null;
        try {
            let script = '';
            let className = null;
            if (container.model && container.model.mainModule) {
                mainModule = container.model.mainModule;
                script = mainModule.torchscriptArena.key;
            }
            else if (container.data) {
                mainModule = container.data;
                let typeName = mainModule.__type__.split('.');
                className = typeName.pop();
                script = 'code/' + typeName.join('/') + '.py';
            }
            context = new torchscript.GraphContext(container, python, mainModule, script, className);
        }
        catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            host.exception(new torchscript.Error(message + " in '" + container.identifier + "'."), false);
        }

        container.parameters = {};
        if (container.model && container.model.mainModule) {
            let queue = [ container.model.mainModule ];
            while (queue.length > 0) {
                let module = queue.shift();
                if (module.name && !module.__name__) {
                    module.__name__ = module.name;
                }
                if (module.parameters) {
                    for (let parameter of module.parameters) {
                        if (parameter.tensorId) {
                            let tensorId = parseInt(parameter.tensorId, 10);
                            parameter.initializer = container.tensors[tensorId];
                            if (parameter.__outputs__ && parameter.__outputs__.length == 1) {
                                container.parameters[parameter.__outputs__[0]] = parameter;
                            }
                        }
                    }
                }
                if (module.submodules) {
                    for (let submodule of module.submodules) {
                        submodule.__parent__ = module;
                        queue.push(submodule);
                    }
                }
            }
        }
        if (container.data) {
            let queue = [ container.data ];
            while (queue.length > 0) {
                let module = queue.shift();
                for (let key of Object.keys(module)) {
                    if (key !== '__type__' && key !== '__parent__') {
                        let obj = module[key];
                        if (!Array.isArray(obj) && obj === Object(obj)) {
                            if (torchscript.Graph._isTensor(obj)) {
                                let parameter = obj;
                                if (!parameter.initializer) {
                                    parameter.initializer = new torchscript.Tensor('pickle', parameter);
                                }
                                if (parameter.__outputs__ && parameter.__outputs__.length == 1) {
                                    container.parameters[parameter.__outputs__[0]] = parameter;
                                }
                            }
                            else if (obj && obj.__type__) {
                                obj.__parent__ = module;
                                if (!obj.__name__) {
                                    obj.__name__ = key;
                                }
                                queue.push(obj);
                            }
                        }
                    }
                }
            }
        }

        if (context) {
            for (let input of context.inputs) {
                this._inputs.push(new torchscript.Parameter(input, true, [
                    new torchscript.Argument(input, null, null)
                ]));
            }
            for (let output of context.outputs) {
                this._outputs.push(new torchscript.Parameter(output, true, [
                    new torchscript.Argument(output, null, null)
                ]));
            }
            for (let node of context.nodes) {
                this._nodes.push(new torchscript.Node(metadata, container, null, node));
            }
        }

        if (container.model || container.data) {
            this._loadModule(metadata, container, mainModule);
        }
    }

    _loadModule(metadata, container, module) {
        if (module) {
            if (torchscript.Graph._parameters(module).length > 0 && !module.__hide__) {
                let node = new torchscript.Node(metadata, container, module, null);
                this._nodes.push(node);
            }
            let submodules = torchscript.Graph._submodules(module);
            for (let submodule of submodules) {
                this._loadModule(metadata, container, submodule);
            }
        }
    }

    static _isTensor(obj) {
        return obj && obj.__type__ && obj.__type__.endsWith('Tensor');
    }

    static _parameters(module) {
        if (module) {
            if (module.__type__) {
                let parameters = [];
                for (let key of Object.keys(module)) {
                    if (torchscript.Graph._isTensor(module[key])) {
                        var parameter = module[key];
                        parameter.__name__ = key;
                        parameters.push(parameter);
                    }
                }
                return parameters;
            }
            else {
                if (module.parameters) {
                    for (let parameter of module.parameters) {
                        parameter.__name__ = parameter.name;
                    }
                    return module.parameters;
                }
            }
        }
        return [];
    }

    static _submodules(module) {
        if (module) {
            if (module.__type__) {
                let submodules = [];
                for (let key of Object.keys(module)) {
                    if (!key.startsWith('__')) {
                        let value = module[key];
                        if (value.__type__ && !torchscript.Graph._isTensor(value)) {
                            submodules.push(value);
                        }
                    }
                }
                return submodules;
            }
            else {
                return module.submodules || [];
            }
        }
        return [];
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

torchscript.Parameter = class {

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

torchscript.Argument = class {

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type;
        this._initializer = initializer;
    }

    get id() {
        return this._id;
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

torchscript.Node = class {

    constructor(metadata, container, module, node) {

        this._metadata = metadata;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        if (module) {
            this._operator = 'Module';
            let parameters = torchscript.Graph._parameters(module);
            for (let parameter of parameters) {
                this._inputs.push(new torchscript.Parameter(parameter.__name__, true, [
                    new torchscript.Argument('', null, parameter.initializer || null)
                ]));
                if (parameter.__outputs__) {
                    this._outputs.push(new torchscript.Parameter(parameter.__name__, true,
                        parameter.__outputs__.map((id) => new torchscript.Argument(id, null, null))
                    ));
                }
            }
        }

        if (node) {
            this._operator = node.name;
            this._name = '';

            let schema = metadata.getSchema(this._operator);

            module = null; 
            let match = true;
            let count = 0;
            for (let input of node.inputs) {
                for (let argument of input) {
                    let parameter = container.parameters[argument.id];
                    if (parameter) {
                        if (parameter.__module__ && (module == null || module == parameter.__module__)) {
                            module = parameter.__module__;
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
                let parameters = torchscript.Graph._parameters(module).filter((p) => p.__name__ !== 'num_batches_tracked');
                if (parameters.length == count && match) {
                    module.__hide__ = true;
                    for (let input of node.inputs) {
                        for (let argument of input) {
                            let parameter = container.parameters[argument.id];
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

            for (let inputIndex = 0; inputIndex < node.inputs.length; inputIndex++) {
                let inputName = inputIndex.toString(); 
                if (schema && schema.inputs && schema.inputs.length > inputIndex) {
                    inputName = schema.inputs[inputIndex].name;
                }
                this._inputs.push(new torchscript.Parameter(inputName, true,
                    node.inputs[inputIndex].map((input) => new torchscript.Argument(input.id, null, input.initializer || null))
                ));
            }

            for (let outputIndex = 0; outputIndex < node.outputs.length; outputIndex++) {
                let outputName = outputIndex.toString(); 
                if (schema && schema.outputs && schema.outputs.length > outputIndex) {
                    outputName = schema.outputs[outputIndex].name;
                }
                this._outputs.push(new torchscript.Parameter(outputName, true, [
                    new torchscript.Argument(node.outputs[outputIndex], null, null)
                ]));
            }

            for (let attributeIndex = 0; attributeIndex < node.attributes.length; attributeIndex++) {
                let attributeSchema = null;
                let attributeName = attributeIndex.toString();
                let attributeValue = node.attributes[attributeIndex];
                if (attributeValue && attributeValue.type === '=' && attributeValue.target.type == 'identifier') {
                    attributeName = attributeValue.target.value;
                    attributeValue = attributeValue.expression;
                    if (schema && schema.attributes) {
                        attributeSchema = schema.attributes.find((s) => s.name == attributeName);
                    }
                }
                else {
                    if (schema && schema.attributes && schema.attributes.length > attributeIndex) {
                        attributeSchema = schema.attributes[attributeIndex];
                        attributeName = attributeSchema.name;
                    }
                }
                this._attributes.push(new torchscript.Attribute(this, attributeSchema, attributeName, attributeValue));
            }
        }
        
        if (module) {
            if (module.__name__) {
                let current = module;
                this._name = current.__name__;
                while (current.__parent__ != null) {
                    current = current.__parent__;
                    if (!current.__parent__ && !current.__name__) {
                        break;
                    }
                    this._name = [ current.__name__, this._name ].join('.')
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

    get operator() {
        return this._operator;
    }

    get category() {
        let schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        let schema = this._metadata.getSchema(this._operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (let attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (let input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (let output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get function() {
        return false;
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

torchscript.Attribute = class {

    constructor(node, schema, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;

        if (value && value.type) {
            switch (value.type) {
                case 'number':
                    this._value = value.value;
                    break;
                case 'string':
                    this._value = value.value;
                    break;
                case 'boolean':
                    this._value = value.value;
                    break;
                case 'identifier':
                    this._value = value.value;
                    break;
            }
        }

        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'type')) {
                this._type = schema.type;
            }

            switch (this._type) {
                case 'boolean':
                    if (this._value == 'False') {
                        this._value = false;
                    }
                    else if (this._value == 'True') {
                        this._value = true;
                    }
                    break;
                case 'int32':
                case 'int64':
                    this._value = parseInt(this._value, 10);
                    break;
                case 'float32':
                case 'float64':
                    this._value = parseFloat(this._value);
                    break;
                case 'int32[]':
                case 'int64[]':
                    if (this._value.type == 'list' && this._value.value.every((item) => item.type === 'number')) {
                        this._value = this._value.value.map((item) => {
                            let number = parseInt(item.value, 10);
                            if (!Number.isNaN(item.value - number)) {
                                return number;
                            }
                            return item.value;
                        });
                    }
                    break;
            }

            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (JSON.stringify(schema.default) == JSON.stringify(this._value)) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && 
                    !Array.isArray(schema.default) &&
                    this.value.every((item) => item == schema.default)) {
                    this._visible = false;
                }
            }
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
        return (this._visible == false || this.name == 'training') ? false : true;
    }
};

torchscript.Tensor = class {

    constructor(format, data) {
        switch (format) {
            case 'json':
                torchscript.Tensor._dataTypeMap = torchscript.Tensor._dataTypeMap || new Map([
                    [ 'FLOAT', 'float32' ],
                    [ 'DOUBLE', 'float64' ],
                    [ 'INT32', 'int32' ],
                    [ 'INT64', 'int64' ]
                ]);
                if (!torchscript.Tensor._dataTypeMap.has(data.tensor.dataType)) {
                    throw new torchscript.Error("Unknown tensor data type '" + data.tensor.dataType + "'.");
                }
                this._type = new torchscript.TensorType(torchscript.Tensor._dataTypeMap.get(data.tensor.dataType), new torchscript.TensorShape(data.tensor.dims));
                this._name = data.tensor.data.key;
                this._data = data.data;
                break;
            case 'pickle':
                this._type = new torchscript.TensorType(data.storage.dataType, new torchscript.TensorShape(data.size));
                this._data = data.storage.data;
                break;
        }
        this._littleEndian = true;
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
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return torchscript.Tensor._stringify(value, '', '    ');
    }

    _context() {
        let context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (!this._type.dataType) {
            context.state = 'Tensor has no data type.';
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

        context.data = this._data;
        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.dataView = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        let results = [];
        let dimensions = context.dimensions;
        if (dimensions.length == 0) {
            dimensions = [ 1 ];
        }
        let size = dimensions[dimension];
        if (dimension == dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType)
                {
                    case 'uint8':
                        results.push(context.dataView.getUint8(context.index, this._littleEndian));
                        context.index++;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.dataView.getInt8(context.index, this._littleEndian));
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
                        results.push(new long.Long(context.dataView.getUint32(context.index, true), context.dataView.getUint32(context.index + 4, true), false));
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
            let result = [];
            result.push(indentation + '[');
            let items = value.map((item) => torchscript.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value && long.Long.isLong(value)) {
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

torchscript.TensorType = class {

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
        return this._dataType + this._shape.toString();
    }
};

torchscript.TensorShape = class {

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

torchscript.Metadata = class {

    static open(host) {
        if (torchscript.Metadata._metadata) {
            return Promise.resolve(torchscript.Metadata._metadata);
        }
        else {
            return host.request(null, 'torchscript-metadata.json', 'utf-8').then((data) => {
                torchscript.Metadata._metadata = new torchscript.Metadata(data);
                return torchscript.Metadata._metadata;
            }).catch(() => {
                torchscript.Metadata._metadata = new torchscript.Metadata(null);
                return torchscript.Metadata._metadata;
            });
        }
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            let schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (let attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

torchscript.GraphContext = class {

    constructor(container, python, mainModule, script, className) {

        this._container = container;
        this._mainModule = mainModule;

        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        this._moduleMap = {};
        this._argumentMap = {};
        this._numToTensorMap = {};

        if (script) {
            let codeKey = container.prefix + script;
            let codeEntries = container.entries.filter((e) => e.name === codeKey);
            if (codeEntries.length == 1) {
                let codeEntry = codeEntries[0];
                let textDecoder = new TextDecoder('utf-8');
                let code = textDecoder.decode(codeEntry.data);
                let reader = new python.Parser(code);
                let program = reader.parse();
                let statements = program.body;
                if (className) {
                    let block = statements.find((statment) => statment.type == 'class' && statment.name == className);
                    statements = block.body.statements;
                }
                let method = statements.find((statement) => statement.type == 'def' && statement.name == 'forward');
                if (method) {
                    this._body = method.body.statements;
                    let methodParameters = method.parameters;
                    if (methodParameters.length > 0 && methodParameters[0].name == 'self') {
                        methodParameters.shift();
                    }
                    for (let parameter of methodParameters) {
                        this._parameter(parameter);
                    }

                    if (this._body.length >= 2) {
                        let returnStatement = this._body[this._body.length - 1];
                        let assignStatement = this._body[this._body.length - 2];
                        if (returnStatement.type == 'return' && 
                            returnStatement.expression.type == 'identifier' &&
                            assignStatement.target.type == 'identifier' &&
                            assignStatement.target.value == returnStatement.expression.value) {
                            returnStatement.expression = assignStatement.expression;
                            this._body.pop();
                            this._body.pop();
                            this._body.push(returnStatement);
                        }
                    }

                    while (this._body.length > 0) {
                        let statement = this._body.shift();
                        if (this._attributeStatement(statement)) {
                            continue;
                        }
                        if (this._moduleStatement(statement)) {
                            continue;
                        }
                        if (this._argumentStatement(statement)) {
                            continue;
                        }
                        if (this._nodeStatement(statement)) {
                            continue;
                        }
                        if (this._returnStatement(statement)) {
                            continue;
                        }
                        throw new torchscript.Error("Unknown statement '" + JSON.stringify(statement) + "'.");
                    }
                }
            }
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

    _parameter(parameter) {
        let type = parameter.parameterType; 
        if (type.type == 'type' && type.value == 'Tuple' && type.arguments && type.arguments.length > 0) {
            if (this._body.length > 0) {
                let statement = this._body[0];
                if (statement.expression.type == 'identifier' && statement.expression.value == parameter.name) {
                    if (statement.type === '=' && statement.target.type === 'tuple') {
                        for (let input of statement.target.value) {
                            if (input) {
                                this._inputs.push(input.value);
                            }
                        }
                        this._body.shift();
                    }
                }
            }
        }
        else {
            this._inputs.push(parameter.name);
        }
    }

    _returnStatement(statement) {
        if (statement.type == 'return') {
            let variable = this._variable();
            if (this._nodeExpression(statement.expression, variable)) {
                this._outputs.push(variable.value);
                return true;
            }
            if (statement.expression.type == 'identifier') {
                this._outputs.push(statement.expression.value);
                return true;
            }
            if (statement.expression.type == 'tuple') {
                let outputs = [];
                for (let expression of statement.expression.value) {
                    variable = this._variable();
                    if (this._nodeExpression(expression, variable)) {
                        outputs.push(variable.value);
                        continue
                    }
                    if (expression.type == 'identifier') {
                        outputs.push(expression.value);
                        continue;
                    }
                    return false;
                }
                this._outputs = this._outputs.concat(outputs);
                return true;
            }
        }
        return false;
    }

    _nodeExpression(expression, target) {
        if (expression.type == 'call' && (target.type == 'identifier' || target.type == 'tuple')) {
            let name = this._name(expression.target);
            let namespace = 'torch.';
            if (name.startsWith(namespace)) {
                let node = {};
                node.name = name.substring(namespace.length);
                node.inputs = [];
                node.outputs = [];
                node.attributes = [];
                let args = expression.arguments;
                while (args.length > 0) {
                    let argument = args[0];
                    argument = this._moduleTensor(argument);
                    if (argument.type == 'identifier' && this._argumentMap[argument.value]) {
                        argument = this._argumentMap[argument.value];
                        delete this._argumentMap[argument.value];
                    }
                    if (argument.type == 'identifier') {
                        if (argument.value === 'False' || argument.value === 'True') {
                            break;
                        }
                        node.inputs.push([ { id: argument.value } ]);
                        args.shift();
                        continue;
                    }
                    if (argument.type == 'list') {
                        let list = [];
                        for (let input of argument.value) {
                            let variable = this._variable();
                            if (this._nodeExpression(input, variable)) {
                                list.push({ id: variable.value });
                            }
                            else if (this._argumentExpression(input, variable)) {
                                list.push({ id: variable.value });
                            }
                            else if (input.type == 'identifier') {
                                list.push({ id: input.value });
                            }
                            else {
                                list = null;
                                break;
                            }
                        }
                        if (list) {
                            node.inputs.push(list);
                            args.shift();
                            continue;
                        }
                    }
                    if (argument.type == 'list') {
                        break;
                    }
                    if (argument.type == 'number' || argument.type == 'string' || argument.type == 'boolean') {
                        break;
                    }
                    if (argument.type == '=') {
                        break;
                    }
                    let variable = this._variable();
                    if (this._nodeExpression(argument, variable)) {
                        node.inputs.push([ { id: variable.value } ]);
                        args.shift();
                        continue;
                    }
                    if (this._argumentExpression(argument, variable)) {
                        node.inputs.push([ { id: variable.value } ]);
                        args.shift();
                        continue;
                    }
                    if (argument.type == '.' &&
                        argument.target.type == 'identifier' &&
                        argument.target.value == 'CONSTANTS' &&
                        argument.member.type == 'identifier' &&
                        argument.member.value.startsWith('c')) {
                        let constantId = [ argument.target.value, argument.member.value ].join('.');
                        let constantIndex = parseInt(argument.member.value.substring(1), 10);
                        let constantTensor = this._container.constants[constantIndex];
                        node.inputs.push([ { id: constantId, initializer: constantTensor } ]);
                        args.shift();
                        continue;
                    }
                    throw new torchscript.Error('Unknown function argument.');
                }
                while (args.length > 0) {
                    if (args[0].type == 'list') {
                        for (let i = 0; i < args[0].value.length; i++) {
                            args[0].value[i] = this._attributeExpression(args[0].value[i]);
                        }
                    }
                    let intExpression = this._attributeExpression(args[0]);
                    if (intExpression) {
                        args[0] = intExpression;
                    }
                    node.attributes.push(args[0]);
                    args.shift();
                }
                if (target.type == 'identifier') {
                    node.outputs.push(target.value);
                }
                if (target.type == 'tuple') {
                    for (let identifier of target.value) {
                        node.outputs.push(identifier.value);
                    }
                }
                this._nodes.push(node);
                return true;
            }
        }
        return false;
    }

    _nodeStatement(statement) {
        if (statement.type == '=') {
            if (this._nodeExpression(statement.expression, statement.target)) {
                return true;
            }
        }
        return false;
    }

    _attributeExpression(expression) {
        if (expression.type == 'identifier') {
            if (this._numToTensorMap[expression.value]) {
                return { type: 'number', value: this._numToTensorMap[expression.value] };
            }
        }
        if (expression.type == 'call' && 
            expression.target.type == 'identifier' &&
            expression.target.value == 'int' &&
            expression.arguments.length == 1) 
        {
            let replace = this._attributeExpression(expression.arguments[0]);
            if (replace) {
                return replace;
            }
        }
        return expression;
    }

    _attributeStatement(statement) {
        if (statement.type == '=' &&
            statement.target.type == 'identifier') { 
            if (statement.expression.type == 'call' &&
                this._name(statement.expression.target) == 'ops.prim.NumToTensor' && 
                statement.expression.arguments.length == 1) {
                let size = statement.expression.arguments[0];
                if (size.type == 'call' &&
                    size.arguments.length == 2 &&
                    this._name(size.target) == 'torch.size' &&
                    size.arguments[0].type == 'identifier' &&
                    size.arguments[1].type == 'number') {
                    this._numToTensorMap[statement.target.value] = this._name(size.target) + '(' + size.arguments.map((a) => a.value.toString()).join(',') + ')';
                    return true;
                }
                if (size.type == 'identifier') {
                    let duplicate1 = this._numToTensorMap[size.value];
                    if (duplicate1) {
                        this._numToTensorMap[statement.target.value] = duplicate1;
                        return true;
                    }
                }
            }
            if (statement.expression.type == 'call' &&
                statement.expression.arguments.length == 2 &&
                this._name(statement.expression.target) == 'torch.size' &&
                statement.expression.arguments[0].type == 'identifier' &&
                statement.expression.arguments[1].type == 'number') {
                this._numToTensorMap[statement.target.value] = this._name(statement.expression.target) + '(' + statement.expression.arguments.map((a) => a.value.toString()).join(',') + ')';
                return true;
            }
            if (statement.expression.type == 'call' &&
                statement.expression.target.type == 'identifier' &&
                statement.expression.target.value == 'int' &&
                statement.expression.arguments.length == 1 &&
                statement.expression.arguments[0].type == 'identifier') {
                let duplicate2 = this._numToTensorMap[statement.expression.arguments[0].value];
                if (duplicate2) {
                    this._numToTensorMap[statement.target.value] = duplicate2;
                    return true;
                }
            }
        }
        return false;
    }

    _submodule(module, name) {
        var obj = module[name];
        if (obj && (!obj.__type__ || !obj.__type__.endsWith('Tensor'))) {
            return obj;
        }
        if (module.submodules) {
            for (let submodule of module.submodules) {
                if (submodule.name === name) {
                    return submodule;
                }
            }
        }
        return null;
    }

    _module(expression) {
        if (expression.type === '.') {
            let module = this._module(expression.target);
            if (module) {
                let submodule = this._submodule(module, expression.member.value);
                if (submodule) {
                    return submodule;
                }
            }
        }
        if (expression.type == 'call' && 
            expression.target.type == 'identifier' && expression.target.value == 'getattr' && expression.arguments.length == 2) {
            let module = this._module(expression.arguments[0]);
            if (!module) {
                return null;
            }
            let name = null;
            if (expression.arguments[1].type == 'string') {
                name = expression.arguments[1].value.substring(1, expression.arguments[1].value.length - 1);
            }
            if (module) {
                let submodule = this._submodule(module, name);
                if (submodule) {
                    return submodule;
                }
            }
        }
        if (expression.type == 'identifier') {
            if (expression.value == 'self') {
                return this._mainModule;
            }
            let module = this._moduleMap[expression.value];
            if (module) {
                return module;
            }
        }
        return null;
    }

    _moduleStatement(statement) {
        if (statement.type == '=' && 
            statement.target.type === 'identifier') {
            let moduleName = statement.target.value;
            let module = this._module(statement.expression);
            if (module) {
                this._moduleMap[moduleName] = module;
                return true;
            }
        }
        return false;
    }

    _argumentExpression(expression, target) {
        expression = this._moduleTensor(expression);
        if (expression.type === '.' && expression.member.type == 'identifier') {
            let targetModule = this._module(expression.target);
            if (targetModule) {
                if (targetModule.parameters) {
                    for (let parameter of targetModule.parameters) {
                        parameter.__module__ = targetModule;
                        if (parameter.name === expression.member.value) {
                            parameter.__outputs__ = parameter.__outputs__ || [];
                            parameter.__outputs__.push(target.value);
                            return true;
                        }
                    }
                }
                let obj = targetModule[expression.member.value];
                if (torchscript.Graph._isTensor(obj)) {
                    obj.__module__ = targetModule;
                    obj.__outputs__ = obj.__outputs__ || [];
                    obj.__outputs__.push(target.value);
                    return true;
                }
            }
        }
        return false;
    }

    _argumentStatement(statement) {
        if (statement.type === '=' && statement.target.type === 'identifier') {
            if (this._argumentExpression(statement.expression, statement.target)) {
                return true;
            }
            if (statement.target.type == 'identifier' &&
                statement.expression.type == 'list') {
                this._argumentMap[statement.target.value] = statement.expression;
                return true;
            }
        }
        return false;
    }

    _variable() {
        return { type: 'identifier', value: '_gen' + Math.random().toString(36).substring(7) };
    }

    _name(expression) {
        if (expression.type == 'identifier') {
            return expression.value;
        }
        if (expression.type == '.') {
            return [ this._name(expression.target), this._name(expression.member) ].join('.');
        }
        throw new torchscript.Error("Failed to resolve name '" + JSON.stringify(expression) + "'.");
    }

    _moduleTensor(expression) {
        if (expression.type == 'call' && 
            expression.arguments.length == 1 &&
            this._name(expression.target) == 'torch.t') {
            return expression.arguments[0];
        }
        return expression;
    }
}

torchscript.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading TorchScript model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = torchscript.ModelFactory;
}
