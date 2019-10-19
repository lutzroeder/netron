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
        const identifier = context.identifier; 
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'pt' || extension == 'pt1' || extension == 'pth' || extension == 'pkl' || extension == 'h5' || extension == 't7' ||
            extension == 'dms' || extension == 'model' || extension == 'ckpt' || identifier.endsWith('.pth.tar')) {
            const entries = context.entries;
            if (entries && entries.length > 0) {
                const versionEntry = entries.find((entry) => entry.name == 'version' || entry.name.endsWith('/version'));
                if (versionEntry) {
                    const prefix = versionEntry.name.substring(0, versionEntry.name.length - 7);
                    if (entries.some((entry) => entry.name == prefix + 'model.json') ||
                        entries.some((entry) => entry.name == prefix + 'data.pkl')) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./python').then((python) => {
            return host.require('./pickle').then((pickle) => {
                const identifier = context.identifier;
                try {
                    let container = new torchscript.Container(context.identifier, context.entries, python, pickle);
                    return torchscript.Metadata.open(host).then((metadata) => {
                        try {
                            return new torchscript.Model(metadata, host, container);
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
};

torchscript.Model = class { 

    constructor(metadata, host, container) {
        this._format = 'TorchScript v' + container.version.toString();
        if (container.model && container.model.producerName) {
            this._producer = container.model.producerName;
            if (container.model.producerVersion) {
                this._producer = this._producer + ' v' + container.model.producerVersion;
            }
        }
        this._graphs = [];
        this._graphs.push(new torchscript.Graph(metadata, host, container));
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

    constructor(metadata, host, container) {
        if (container.model && container.model.mainModule) {
            this._name = container.model.mainModule.name;
        }
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        let tensors = [];
        let constants = [];

        if (container.model && container.model.tensors) {
            tensors = container.model.tensors.map((tensor) => {
                return new torchscript.Tensor('json', tensor);
            });
        }

        if (container.constants) {
            constants = container.constants.map((tensor) => new torchscript.Tensor('pickle', tensor));
        }
        else {
            constants = tensors;
        }

        let context = null;
        try {
            context = new torchscript.GraphContext(container, constants);
        }
        catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            host.exception(new torchscript.Error(message + " in '" + container.identifier + "'."), false);
        }

        container.parameters = {};
        if (container.model && container.model.mainModule) {
            let queue = [ container.model.mainModule ];
            let tensorMax = 0;
            while (queue.length > 0) {
                let module = queue.shift();
                if (module.parameters) {
                    for (let parameter of module.parameters) {
                        if (parameter.tensorId) {
                            let tensorId = parseInt(parameter.tensorId, 10);
                            tensorMax = Math.max(tensorId, tensorMax);
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
            tensorMax++;
            queue = [ container.model.mainModule ];
            while (queue.length > 0) {
                let module = queue.shift();
                if (module.name && !module.__name__) {
                    module.__name__ = module.name;
                }
                if (module.parameters) {
                    for (let parameter of module.parameters) {
                        if (parameter.tensorId) {
                            let tensorId = parseInt(parameter.tensorId, 10);
                            parameter.initializer = tensors[tensorId];
                            if (parameter.__outputs__ && parameter.__outputs__.length == 1) {
                                container.parameters[parameter.__outputs__[0]] = parameter;
                            }
                        }
                    }
                }
                if (module.attributes) {
                    for (let attribute of module.attributes) {
                        if (attribute.id) {
                            let attributeId = parseInt(attribute.id, 10);
                            attribute.initializer = tensors[tensorMax + attributeId];
                            if (attribute.__outputs__ && attribute.__outputs__.length == 1) {
                                container.parameters[attribute.__outputs__[0]] = attribute;
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
                            if (torchscript.Utility.isTensor(obj)) {
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
            this._loadModule(metadata, container, container.model || container.data);
        }
    }

    _loadModule(metadata, container, module) {
        if (module) {
            if (torchscript.Graph._getParameters(module).length > 0 && !module.__hide__) {
                let node = new torchscript.Node(metadata, container, module, null);
                this._nodes.push(node);
            }
            let submodules = torchscript.Graph._getSubmodules(module);
            for (let submodule of submodules) {
                this._loadModule(metadata, container, submodule);
            }
        }
    }

    static _getParameters(module) {
        let parameters = [];
        if (module) {
            if (module.__type__) {
                for (let key of Object.keys(module)) {
                    if (torchscript.Utility.isTensor(module[key])) {
                        const parameter = module[key];
                        parameter.__name__ = key;
                        parameters.push(parameter);
                    }
                }
            }
            else {
                if (module.parameters) {
                    for (let parameter of module.parameters) {
                        parameter.__name__ = parameter.name;
                        parameters.push(parameter);
                    }
                }
                if (module.attributes) {
                    for (let attribute of module.attributes) {
                        attribute.__name__ = attribute.name;
                        parameters.push(attribute);
                    }
                }
            }
        }
        return parameters;
    }

    static _getSubmodules(module) {
        if (module) {
            if (module.__type__) {
                let submodules = [];
                for (let key of Object.keys(module)) {
                    if (!key.startsWith('__')) {
                        let value = module[key];
                        if (value.__type__ && !torchscript.Utility.isTensor(value)) {
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
            let parameters = torchscript.Graph._getParameters(module);
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

            const schema = metadata.getSchema(this._operator);

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
                let parameters = torchscript.Graph._getParameters(module).filter((p) => p.__name__ !== 'num_batches_tracked');
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

            for (let i = 0; i < node.attributes.length; i++) {
                let attributeSchema = null;
                let name = i.toString();
                let value = node.attributes[i];
                if (value && value.type === '=' && value.target.type == 'id') {
                    name = value.target.value;
                    value = value.expression;
                    if (schema && schema.attributes) {
                        attributeSchema = schema.attributes.find((s) => s.name == name);
                    }
                }
                else {
                    if (schema && schema.attributes && schema.attributes.length > i) {
                        attributeSchema = schema.attributes[i];
                        name = attributeSchema.name;
                    }
                }
                this._attributes.push(new torchscript.Attribute(attributeSchema, name, value));
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
        const schema = this._metadata.getSchema(this._operator);
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

    constructor(schema, name, value) {
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
                case 'id':
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
                    if (typeof this._value !== 'number') {
                        if (typeof this._value === 'string') {
                            this._value = parseInt(this._value, 10);
                        }
                        else {
                            this._value = torchscript.Utility.format(this._value);
                        }
                    }
                    break;
                case 'float32':
                case 'float64':
                    if (typeof this._value !== 'number') {
                        if (typeof this._value === 'string') {
                            this._value = parseFloat(this._value);
                        }
                        else {
                            this._value = torchscript.Utility.format(this._value);
                        }
                    }
                    break;
                case 'int32[]':
                case 'int64[]': {
                    switch (this._value.type) {
                        case 'list':
                            this._value = this._value.value.map((item) => {
                                if (item.type === 'number') {
                                    let number = parseInt(item.value, 10);
                                    if (!Number.isNaN(item.value - number)) {
                                        return number;
                                    }
                                }
                                if (item.type === 'call') {
                                    return torchscript.Utility.format(item);
                                }
                                return item;
                            });
                            break;
                        case 'call':
                            this._value = torchscript.Utility.format(this._value);
                            break;
                    }
                    break;
                }
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

    constructor(format, tensor) {
        switch (format) {
            case 'json':
                torchscript.Tensor._dataTypeMap = torchscript.Tensor._dataTypeMap || new Map([
                    [ 'FLOAT', 'float32' ],
                    [ 'FLOAT16', 'float16' ],
                    [ 'DOUBLE', 'float64' ],
                    [ 'INT32', 'int32' ],
                    [ 'INT64', 'int64' ]
                ]);
                if (!torchscript.Tensor._dataTypeMap.has(tensor.dataType)) {
                    throw new torchscript.Error("Unknown tensor data type '" + tensor.dataType + "'.");
                }
                this._type = new torchscript.TensorType(torchscript.Tensor._dataTypeMap.get(tensor.dataType), new torchscript.TensorShape(tensor.dims));
                this._name = tensor.data.key;
                this._data = tensor.storage;
                break;
            case 'pickle':
                this._type = new torchscript.TensorType(tensor.storage.dataType, new torchscript.TensorShape(tensor.size));
                this._data = tensor.storage.data;
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
        switch (this._type.dataType) {
            case 'uint8':
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
                switch (context.dataType) {
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
            const items = value.map((item) => torchscript.Tensor._stringify(item, indentation + indent, indent));
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
            const schema = this.getSchema(operator);
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

torchscript.Utility = class {

    static target(expression) {
        if (expression.type == 'id') {
            return expression.value;
        }
        if (expression.type == '.') {
            return torchscript.Utility.target(expression.target) + '.' + torchscript.Utility.target(expression.member)
        }
        throw new torchscript.Error("Failed to resolve name '" + JSON.stringify(expression) + "'.");
    }

    static format(expression) {
        switch (expression.type) {
            case 'call': {
                let builder = [];
                for (let argument of expression.arguments) {
                    builder.push(torchscript.Utility.format(argument));
                }
                return this.target(expression.target) + '(' + builder.join(',') + ')';
            }
            case 'number':
            case 'id': {
                return expression.value;
            }
            case 'list': {
                let builder = [];
                for (let item of expression.value) {
                    builder.push(torchscript.Utility.format(item));
                }
                return '[' + builder.join(',') + ']';
            }
            case '.': {
                return torchscript.Utility.target(expression);
            }
            default:
                throw new torchscript.Error("Unknown expression type '" + expression.type + "'.");
        }
    }

    static isTensor(obj) {
        return obj && obj.__type__ && (obj.__type__.endsWith('Tensor') || obj.__type__ == '__tensor__');
    }
}

torchscript.Container = class {

    constructor(identifier, entries, python, pickle) {
        this._identifier = identifier;
        this._entries = entries;
        this._python = python;
        this._pickle = pickle;
        this._utf8Decoder = new TextDecoder('utf-8');
        this._types = new Map();
        this._packages = new Map();

        // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md
        const versionEntry = this._entries.find((entry) => entry.name == 'version' || entry.name.endsWith('/version'));
        if (!versionEntry) {
            throw new torchscript.Error('TorchScript container does not contain version signature.');
        }
        this._prefix = versionEntry.name.substring(0, versionEntry.name.length - 7);
        this._version = JSON.parse(this._utf8Decoder.decode(versionEntry.data));
    }

    get identifier() {
        return this._identifier;
    }

    get version() {
        return this._version;
    }

    get data() {
        if (this._data ===  undefined) {
            this._data = null;
            let entry = this._entries.find((entry) => entry.name == this._prefix + 'data.pkl');
            if (entry) {
                this._data = this._unpickle(entry.data, this._storage('data'));
            }
        }
        return this._data;
    }

    get model() {
        if (this._model === undefined) {
            this._model = null;
            let entry = this._entries.find((entry) => entry.name == this._prefix + 'model.json');
            if (entry) {
                this._model = JSON.parse(this._utf8Decoder.decode(entry.data))
                let entries = new Map();
                for (let entry of this._entries) {
                    entries.set(entry.name, entry.data);
                }
                for (let tensor of this._model.tensors) {
                    const key = this._prefix + tensor.data.key;
                    tensor.storage = entries.get(key);
                }
            }
        }
        return this._model;
    }

    get attributes() {
        if (this._attributes ===  undefined) {
            this._attributes = null;
            let entry = this._entries.find((entry) => entry.name == this._prefix + 'attributes.pkl');
            if (entry) {
                this._attributes = this._unpickle(entry.data, null);

            }
        }
        return this._attributes;

    }

    get constants() {
        if (this._constants ===  undefined) {
            this._constants = null;
            let entry = this._entries.find((entry) => entry.name == this._prefix + 'constants.pkl');
            if (entry) {
                this._constants = this._unpickle(entry.data, this._storage('constants'));
            }
        }
        return this._constants;
    }

    parse(file) {
        if (!this._packages.has(file)) {
            const key = this._prefix + file;
            const entries = this._entries.filter((e) => e.name === key);
            if (entries.length === 1) {
                const code = this._utf8Decoder.decode(entries[0].data);
                const reader = new this._python.Parser(code);
                const program = reader.parse();
                this._packages.set(file, program);
            }
        }
        return this._packages.get(file);
    }

    _storage(dirname) {
        let map = new Map();
        const prefix = this._prefix + dirname + '/';
        for (let entry of this._entries) {
            if (entry.name.startsWith(prefix)) {
                const key = entry.name.substring(prefix.length);
                map.set(key, entry.data);
            }
        }
        return map;
    }

    _unpickle(data, storage_map) {
        if (!data) {
            return null;
        }
        let functionTable = new Map();
        functionTable.set('collections.OrderedDict', function(args) {
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
        });
        functionTable.set('torch._utils._rebuild_tensor_v2', function(storage, storage_offset, size, stride, requires_grad, backward_hooks) {
            return {
                __type__: storage.__type__.replace('Storage', 'Tensor'),
                storage: storage,
                storage_offset: storage_offset,
                size: size,
                stride: stride,
                requires_grad:requires_grad,
                backward_hooks: backward_hooks
            };
        });
        functionTable.set('torch._utils._rebuild_qtensor', function(storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks) {
            return {
                __type__: storage.__type__.replace('Storage', 'Tensor'),
                storage: storage,
                storage_offset: storage_offset,
                size: size,
                stride: stride,
                quantizer_params: quantizer_params,
                requires_grad:requires_grad,
                backward_hooks: backward_hooks
            };
        });
        functionTable.set('torch.jit._pickle.build_intlist', function(data) {
            return data;
        });
        functionTable.set('torch.jit._pickle.build_tensorlist', function(data) {
            return data;
        });
        let constructorTable = new Map();
        constructorTable.set('torch.ByteStorage', function (size) { 
            this.size = size; this.dataTypeSize = 1; this.dataType = 'uint8'; 
        });
        constructorTable.set('torch.CharStorage', function (size) { 
            this.size = size; this.dataTypeSize = 1; this.dataType = 'int8'; 
        });
        constructorTable.set('torch.ShortStorage', function (size) { 
            this.size = size; this.dataTypeSize = 2; this.dataType = 'int16';
        });
        constructorTable.set('torch.IntStorage', function (size) { 
            this.size = size; this.dataTypeSize = 4; this.dataType = 'int32';
        });
        constructorTable.set('torch.LongStorage', function (size) { 
            this.size = size; this.dataTypeSize = 8; this.dataType = 'int64';
        });
        constructorTable.set('torch.HalfStorage', function (size) {
            this.size = size; this.dataTypeSize = 2; this.dataType = 'float16';
        });
        constructorTable.set('torch.FloatStorage', function (size) {
            this.size = size; this.dataTypeSize = 4; this.dataType = 'float32';
        });
        constructorTable.set('torch.DoubleStorage', function (size) { 
            this.size = size; this.dataTypeSize = 8; this.dataType = 'float64';
        });
        constructorTable.set('torch.QInt8Storage', function (size) { 
            this.size = size; this.dataTypeSize = 1; this.dataType = 'qint8';
        });
        let function_call = (name, args) => {
            if (functionTable.has(name)) {
                const func = functionTable.get(name);
                return func.apply(null, args);
            }
            let obj = { __type__: name };
            if (constructorTable.has(name)) {
                const constructor = constructorTable.get(name);
                constructor.apply(obj, args);
                return obj;
            }
            const type = this.type(name);
            if (type) {
                // this._invoke(type.body.statements, obj, args);
                return obj;
            }
            throw new torchscript.Error("Unknown function '" + name + "'.");
        };
        let deserialized_objects = new Map();
        const persistent_load = (saved_id) => {
            const typename = saved_id.shift();
            if (typename !== 'storage') {
                throw new torchscript.Error("Unknown persistent load type '" + typename + "'.");
            }
            const data_type = saved_id.shift();
            const root_key = saved_id.shift();
            saved_id.shift(); // location
            const size = saved_id.shift();
            let storage = null;
            if (deserialized_objects.has(root_key)) {
                storage = deserialized_objects.get(root_key);
            }
            else {
                storage = function_call(data_type, [ size ]);
                storage.data = storage_map.get(root_key);
                deserialized_objects[root_key] = storage;
            }
            const view_metadata = saved_id.shift();
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
        return new this._pickle.Unpickler(data).load(function_call, persistent_load);
    }

    type(name) {
        if (!this._types.has(name)) {
            let parts = name.split('.');
            const className = parts.pop();
            const file = 'code/' + parts.join('/') + '.py';
            const program = this.parse(file);
            if (program) {
                for (let statement of program.body) {
                    if (statement.type === 'class' && statement.name == className) {
                        this._types.set(name, statement);
                        break;
                    }
                }
            }
        }
        return this._types.get(name);
    }

    _invoke(statements, obj, args) {
        for (let statement of statements) {
            switch (statement.type) {
                case 'def': {
                    const method = statement;
                    const self = this;
                    obj[statement.name] = function() {
                        var locals = {};
                        var args = Array.prototype.slice.call(arguments);
                        for (let parameter of method.parameters) {
                            if (parameter.name == 'self') {
                                locals['self'] = obj;
                            }
                            else {
                                locals[parameter.name] = args.shift();
                            }
                        }
                        self._invoke(method.body.statements, obj, locals);
                    }
                    break;
                }
                case 'var':
                    break;
                default:
                    this._statement(statement, obj, args);
                    break;
            }
        }
    }

    _statement(statement, obj, locals) {
        switch (statement.type) {
            case '=':
                this._expression(statement, obj, locals);
                return;
        }
        throw new torchscript.Error("Unknown statement '" + statement + "'.");
    }

    _expression(expression, obj, locals) {
        switch (expression.type) {
            case '=': {
                const target = expression.target;
                if (target.type === 'id') {
                    locals[target.value] = this._expression(expression.expression, obj, locals);
                    return;
                }
                else if (target.type === '[]') {
                    if (target.target.type === 'id' &&
                        target.arguments.type === 'list' &&
                        target.arguments.value.length === 1) {
                        const index = this._expression(target.arguments.value[0], obj, locals);
                        locals[target.target.value][index] = this._expression(expression.expression, obj, locals);
                        return;
                    }
                }
                else if (target.type === '.' && 
                    target.member.type === 'id') {
                    this._expression(target.target, obj, locals)[target.member.value] = this._expression(expression.expression, obj, locals);
                    return;
                }
                else if (target.type === 'tuple') {
                    const value = this._expression(expression.expression, obj, locals);
                    if  (target.value.length == value.length && target.value.every((item) => item.type === 'id')) {
                        for (let i = 0; i < value.length; i++) {
                            locals[target.value[i].value] = value[i];
                        }
                        return;
                    }
                }
                break;
            }
            case 'list': {
                return expression.value.map((item) => this._expression(item, obj, locals));
            }
            case 'string': {
                return expression.value.substring(1, expression.value.length - 1);
            }
            case 'number': {
                return expression.value;
            }
            case '[]': {
                if (expression.target.type === 'id' &&
                    expression.arguments.type === 'list' &&
                    expression.arguments.value.length === 1)  {
                    const index = this._expression(expression.arguments.value[0], obj, locals);
                    return locals[expression.target.value][index];
                }
                break;
            }
            case '.': {
                const targetName = torchscript.Utility.target(expression);
                if (targetName) {
                    const type = this.type(targetName);
                    if (type) {
                        return { __type__: type };
                    }
                }
                if (expression.member.type == 'id') {
                    return this._expression(expression.target, obj, locals)[expression.member.value];
                }
                break;
            }
            case 'id': {
                if (expression.value === 'self') {
                    return obj;
                }
                return locals[expression.value];
            }

        }
        throw new torchscript.Error("Unknown expression '" + expression + "'.");
    }
}

torchscript.GraphContext = class {

    constructor(container, constants) {

        this._container = container;
        this._constants = constants;

        this._mainModule = null;
        let script = '';
        let namespaceName = null;
        let className = null;
        if (container.model && container.model.mainModule) {
            this._mainModule = container.model.mainModule;
            script = this._mainModule.torchscriptArena.key;
        }
        else if (container.data) {
            this._mainModule = container.data;
            const typeName = this._mainModule.__type__.split('.');
            className = typeName.pop();
            namespaceName = typeName.join('.');
            script = 'code/' + typeName.join('/') + '.py';
        }

        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        this._moduleMap = new Map();
        this._classMap = new Map();
        this._state = {};

        if (script) {
            const program = container.parse(script);
            if (program) {
                let statements = program.body;
                if (className) {
                    let main = null;
                    for (let statement of statements) {
                        if (statement.type == 'class') {
                            if (namespaceName) {
                                this._classMap.set(namespaceName + '.' + statement.name, statement);
                            }
                            if (statement.name == className) {
                                main = statement;
                            }
                        }
                    }
                    statements = main.body.statements;
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
                        // x = ...
                        // return x
                        let returnStatement = this._body[this._body.length - 1];
                        let assignStatement = this._body[this._body.length - 2];
                        if (returnStatement.type === 'return' && 
                            returnStatement.expression.type === 'id' &&
                            assignStatement.type === '=' &&
                            assignStatement.target.type === 'id' &&
                            assignStatement.target.value === returnStatement.expression.value) {
                            returnStatement.expression = assignStatement.expression;
                            this._body.pop();
                            this._body.pop();
                            this._body.push(returnStatement);
                        }
                    }

                    while (this._body.length > 0) {
                        let statement = this._body.shift();
                        if (this._conditionStatement(statement)) {
                            continue;
                        }
                        if (this._assignStatement(statement)) {
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
                        if (statement.type === 'pass') {
                            continue;
                        }
                        if (this._isCall(statement, 'torch.warn', [ {}, {} ])) {
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
                if (statement.expression.type == 'id' && statement.expression.value == parameter.name) {
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

    _conditionStatement(statement) {
        if (statement.type === 'if') {
            let expression = statement.condition;
            if (!this._isBooleanLiteral(expression)) {
                if (expression.type == 'id' && this._state[expression.value]) {
                    expression = this._state[expression.value]
                }
                else {
                    expression = this._evaluateBooleanExpression(statement.condition);
                }
            }
            if (this._isBooleanLiteral(expression)) {
                switch (expression.value) {
                    case 'True':
                        this._body = statement.then.statements.concat(this._body);
                        return true;
                    case 'False':
                        this._body = statement.else.statements.concat(this._body);
                        return true;
                }
            }
        }
        return false;
    }

    _returnStatement(statement) {
        if (statement.type == 'return') {
            let variable = this._variable();
            let expression = statement.expression;
            if (this._nodeExpression(expression, variable)) {
                this._outputs.push(variable.value);
                return true;
            }
            if (expression.type == 'id' && this._state[expression.value] && this._state[expression.value].type === 'tuple' ) {
                expression = this._state[expression.value];
            }
            if (expression.type == 'id') {
                this._outputs.push(expression.value);
                return true;
            }
            if (expression.type == 'tuple') {
                let outputs = [];
                for (let item of expression.value) {
                    variable = this._variable();
                    if (this._nodeExpression(item, variable)) {
                        outputs.push(variable.value);
                        continue;
                    }
                    if (item.type == 'id') {
                        outputs.push(item.value);
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
        if (expression.type == 'call' && (target.type == 'id' || target.type == 'tuple')) {
            let name = torchscript.Utility.target(expression.target);
            let namespace = 'torch.';
            if (name.startsWith(namespace)) {
                let inputs = [];
                let outputs = [];
                let args = expression.arguments;
                while (args.length > 0) {
                    let argumentExpression = args[0];
                    argumentExpression = this._moduleTensor(argumentExpression);
                    if (this._isCall(argumentExpression, 'ops.prim.data', [ {} ])) {
                        argumentExpression = argumentExpression.arguments[0];
                    }
                    if (argumentExpression.type == 'id' &&
                        this._state[argumentExpression.value]) {
                        const valueExpression = this._state[argumentExpression.value];
                        if (!torchscript.Utility.isTensor(valueExpression)) {
                            argumentExpression = this._state[argumentExpression.value];
                        }
                    }
                    if (argumentExpression.type === 'id') {
                        if (this._isBooleanLiteral(argumentExpression)) {
                            break;
                        }
                        let argument = argumentExpression.value;
                        inputs.push([ { id: argument } ]);
                        args.shift();
                        continue;
                    }
                    if (argumentExpression.type == 'list') {
                        let list = [];
                        for (let input of argumentExpression.value) {
                            let variable = this._variable();
                            if (this._nodeExpression(input, variable)) {
                                list.push({ id: variable.value });
                            }
                            else if (this._argumentExpression(input, variable)) {
                                list.push({ id: variable.value });
                            }
                            else if (input.type == 'id') {
                                list.push({ id: input.value });
                            }
                            else {
                                list = null;
                                break;
                            }
                        }
                        if (list) {
                            inputs.push(list);
                            args.shift();
                            continue;
                        }
                    }
                    if (argumentExpression.type == 'list') {
                        break;
                    }
                    if (argumentExpression.type === 'number' || argumentExpression.type == 'string' || argumentExpression.type == 'boolean') {
                        break;
                    }
                    if (argumentExpression.type === '=') {
                        break;
                    }
                    if (this._isCall(argumentExpression, 'torch.list_with_default', [ {}, {} ])) {
                        break;
                    }
                    if (this._isCall(argumentExpression, 'torch.device', [ { type: 'string' } ])) {
                        break;
                    }
                    if (this._isCall(argumentExpression, 'int', [ {} ]) ||
                        this._isCall(argumentExpression, 'float', [ {} ])) {
                        break;
                    }
                    const variable = this._variable();
                    if (this._nodeExpression(argumentExpression, variable)) {
                        inputs.push([ { id: variable.value } ]);
                        args.shift();
                        continue;
                    }
                    if (this._argumentExpression(argumentExpression, variable)) {
                        inputs.push([ { id: variable.value } ]);
                        args.shift();
                        continue;
                    }
                    if (argumentExpression.type == '.' &&
                        argumentExpression.target.type == 'id' &&
                        argumentExpression.target.value == 'CONSTANTS' &&
                        argumentExpression.member.type == 'id' &&
                        argumentExpression.member.value.startsWith('c')) {
                        const constantId = [ argumentExpression.target.value, argumentExpression.member.value ].join('.');
                        const constantIndex = parseInt(argumentExpression.member.value.substring(1), 10);
                        const constantTensor = this._constants[constantIndex];
                        inputs.push([ { id: constantId, initializer: constantTensor } ]);
                        args.shift();
                        continue;
                    }
                    if (argumentExpression.type == '.') {
                        const value = this._evaluateExpression(argumentExpression);
                        if (!torchscript.Utility.isTensor(value)) {
                            break;
                        }
                    }
                    throw new torchscript.Error('Unknown function argument.');
                }
                let attributes = [];
                while (args.length > 0) {
                    let attributeExpression = args[0]; 
                    if (attributeExpression.type == 'list') {
                        for (let i = 0; i < attributeExpression.value.length; i++) {
                            attributeExpression.value[i] = this._attributeExpression(attributeExpression.value[i]);
                        }
                    }
                    let intExpression = this._attributeExpression(attributeExpression);
                    if (intExpression) {
                        attributeExpression = intExpression;
                    }
                    attributes.push(attributeExpression);
                    args.shift();
                }
                if (target.type == 'id') {
                    outputs.push(target.value);
                }
                if (target.type == 'tuple') {
                    for (let identifier of target.value) {
                        outputs.push(identifier.value);
                    }
                }
                this._nodes.push({
                    name: name.substring(namespace.length),
                    attributes: attributes,
                    inputs: inputs,
                    outputs: outputs
                });
                return true;
            }
        }
        return false;
    }

    _nodeStatement(statement) {
        if (statement.type == '=') {
            const target = statement.target;
            const expression = statement.expression;
            if (target.type == 'id') {
                if (this._nodeExpression(expression, target)) {
                    this._state[target.value] = { __type__: '__tensor__' };
                    return true;
                }
            }
            if (target.type == 'tuple' && target.value.every((e) => e.type == 'id')) {
                if (this._nodeExpression(expression, target)) {
                    for (let item of target.value) {
                        this._state[item.value] = { __type__: '__tensor__' };
                    }
                    return true;
                }
            }
            if (target.type == 'id' &&
                expression.type == 'id' &&
                this._state[expression.value]) {
                this._state[target.value] = expression;
                return true;
            }
        }
        return false;
    }

    _attributeExpression(expression) {
        if (expression.type == 'id') {
            if (this._state[expression.value]) {
                return this._state[expression.value];
            }
        }
        return this._evaluateExpression(expression);
    }

    _assignStatement(statement) {
        if (statement.type == '=' &&
            statement.target.type == 'id') {
            const target = statement.target;
            const expression = statement.expression;
            // _0 = ops.prim.NumToTensor(...)
            if (this._isCall(expression, 'ops.prim.NumToTensor', [ {} ])) { 
                let sizeExpression = expression.arguments[0];
                if (this._isCall(sizeExpression, 'torch.size', [ { type: 'id' }, {} ])) { 
                    this._state[target.value] = sizeExpression;
                    return true;
                }
                if (sizeExpression.type == 'id') {
                    let duplicate1 = this._state[sizeExpression.value];
                    if (duplicate1) {
                        this._state[target.value] = duplicate1;
                        return true;
                    }
                }
            }
            // _stride_3 = torch._unwrap_optional(_3)
            // _stride_3 = ops.prim.unchecked_unwrap_optional(_127)
            if (this._isCall(expression, 'torch._unwrap_optional', [ {} ]) ||
                this._isCall(expression, 'ops.prim.unchecked_unwrap_optional', [ {} ])) {
                let argument = expression.arguments[0];
                if (argument && 
                    argument.type == 'id' && 
                    this._state[argument.value] &&
                    !torchscript.Utility.isTensor(this._state[argument.value])) {
                    argument = this._state[argument.value];
                }
                this._state[target.value] = argument;
                return true;
            }
            // stride = unchecked_cast(List[int], _134)
            if (this._isCall(expression, 'unchecked_cast', [ {}, { type: 'id' } ])) {
                this._state[target.value] = expression.arguments[1];
                return true;
            }
            // _0 = torch.size(... , ...)
            if (this._isCall(expression, 'torch.size', [ { type: 'id' }, { type: 'number' } ])) {
                this._state[target.value] = expression;
                return true;
            }
            // _output_size = torch.list_with_default([7, 7], torch.size(x0))
            if (this._isCall(expression, 'torch.list_with_default', [ {}, {} ])) {
                this._state[target.value] = expression;
                return true;
            }
            // _0 = int(...)
            if (this._isCall(expression, 'int', [ { type: 'id' }] )) {
                let duplicate2 = this._state[statement.expression.arguments[0].value];
                if (duplicate2) {
                    this._state[target.value] = duplicate2;
                    return true;
                }
            }
            // _14 = _15
            if (expression.type === 'id' && this._isBooleanLiteral(this._state[expression.value])) {
                this._state[target.value] = this._state[expression.value];
                return true;
            }
            // exponential_average_factor = 0.10000000000000001
            if (expression.type === 'number') {
                this._state[target.value] = expression;
                return true;
            }
            // _8 = (empty, empty)
            if (expression.type === 'tuple') {
                this._state[target.value] = expression;
                return true;
            }
            // _6 = [torch.mul(self.lstm_depth, 2), torch.size(token_emb, 0), self.lstm_width]
            if (expression.type === '[]') {
                this._state[target.value] = expression;
                return true;
            }
            const valueExpression = this._evaluateExpression(expression);
            if (valueExpression.type === 'number' || this._isBooleanLiteral(valueExpression)) {
                this._state[target.value] = expression;
                return true;
            }
            // _aux = None
            if (expression.type === 'id' && expression.value === 'None') {
                this._state[target.value] = expression;
                return true;
            }
            // _0 = <boolean expression>
            const booleanExpression = this._evaluateBooleanExpression(expression);
            if (booleanExpression) {
                this._state[target.value] = booleanExpression;
                return true;
            }
            // _0 = self.features
            const moduleName = target.value;
            const module = this._getModule(expression);
            if (module) {
                this._moduleMap.set(moduleName, module);
                return true;
            }
            // _14190 = __torch__.torchvision.models.inception.InceptionOutputs(x219, aux)
            if (expression.type == 'call') {
                const className = torchscript.Utility.target(expression.target);
                if (this._classMap.has(className)) {
                    const tuple = this._classMap.get(className);
                    if (tuple && tuple.base && tuple.base.length > 0 &&
                        tuple.base[0].type === 'id' && tuple.base[0].value === 'NamedTuple') {
                        this._state[target.value] = { type: 'tuple', value: expression.arguments };
                        return true;
                    }
                }
            }
        }
        return false;
    }

    _getParameter(expression) {
        expression = this._moduleTensor(expression);
        if (expression.type === '.' && expression.member.type == 'id') {
            let targetModule = this._getModule(expression.target);
            if (targetModule) {
                if (targetModule.parameters) {
                    for (let parameter of targetModule.parameters) {
                        parameter.__type__ = '__tensor__';
                        parameter.__module__ = targetModule;
                        if (parameter.name === expression.member.value) {
                            return parameter;
                        }
                    }
                }
                if (targetModule.attributes) {
                    for (let attribute of targetModule.attributes) {
                        attribute.__type__ = '__tensor__';
                        attribute.__module__ = targetModule;
                        if (attribute.name === expression.member.value) {
                            return attribute;
                        }
                    }
                }
                let obj = targetModule[expression.member.value];
                if (torchscript.Utility.isTensor(obj)) {
                    obj.__module__ = targetModule;
                    return obj;
                }
            }
        }
        return null;
    }

    _getSubmodule(module, name) {
        const obj = module[name];
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

    _getModule(expression) {
        if (expression.type === '.') {
            let module = this._getModule(expression.target);
            if (module) {
                let submodule = this._getSubmodule(module, expression.member.value);
                if (submodule) {
                    return submodule;
                }
            }
        }
        if (expression.type == 'call' && 
            expression.target.type == 'id' && expression.target.value == 'getattr' && expression.arguments.length == 2) {
            let module = this._getModule(expression.arguments[0]);
            if (!module) {
                return null;
            }
            let name = null;
            if (expression.arguments[1].type == 'string') {
                name = expression.arguments[1].value.substring(1, expression.arguments[1].value.length - 1);
            }
            if (module) {
                let submodule = this._getSubmodule(module, name);
                if (submodule) {
                    return submodule;
                }
            }
        }
        if (expression.type == 'id') {
            if (expression.value == 'self') {
                return this._mainModule;
            }
            const moduleName = expression.value;
            if (this._moduleMap.has(moduleName)) {
                return this._moduleMap.get(moduleName);
            }
        }
        return null;
    }

    _argumentExpression(expression, target) {
        const parameter = this._getParameter(expression);
        if (parameter) {
            parameter.__outputs__ = parameter.__outputs__ || [];
            parameter.__outputs__.push(target.value);
            this._state[target.value] = parameter;
            return true;
        }
        return false;
    }

    _argumentStatement(statement) {
        if (statement.type === '=' && 
            statement.target.type === 'id') {
            const target = statement.target;
            const expression = statement.expression;
            // _1 = self.conv1
            if (this._argumentExpression(expression, target)) {
                return true;
            }
            if (expression.type == 'list') {
                this._state[target.value] = expression;
                return true;
            }
            // _0 = "Implicit dimension choice for {} has been deprecated. Change the call to include dim=X as an argument."
            if (expression.type == 'string') {
                this._state[target.value] = expression;
                return true;
            }
            // _5 = False
            if (this._isBooleanLiteral(expression)) {
                this._state[target.value] = expression;
                return true;
            }
            // _3 = uninitialized(Tensor)
            if (this._isCall(expression, 'uninitialized', [ {} ])) {
                this._state[target.value] = expression;
                return true;
            }
        }
        // _4, _5 = False, _3
        if (statement.type === '=' &&
            statement.target.type === 'tuple' &&
            statement.expression.type === 'tuple' &&
            statement.target.value.length == statement.expression.value.length) {
            for (let i = 0; i < statement.target.value.length; i++) {
                const target = statement.target.value[i];
                const expression = statement.expression.value[i];
                if (target.type == 'id' && expression.type == 'id') {
                    this._state[target.value] = expression;
                    continue;
                }
                if (this._argumentExpression(expression, target)) {
                    continue;
                }
            }
            return true;
        }
        return false;
    }

    _variable() {
        return { type: 'id', value: '_gen' + Math.random().toString(36).substring(7) };
    }

    _moduleTensor(expression) {
        if (this._isCall(expression, 'torch.t', [ {} ])) {
            return expression.arguments[0];
        }
        return expression;
    }

    _isCall(expression, name, args) {
        if (expression.type !== 'call') {
            return false;
        }
        if (torchscript.Utility.target(expression.target) !== name) {
            return false;
        }
        if (expression.arguments.length !== args.length) {
            return false;
        }
        for (let i = 0; i < args.length; i++) {
            const argument = args[i];
            if (argument.type && argument.type !== expression.arguments[i].type) {
                return false;
            }
            if (argument.value && argument.value !== expression.arguments[i].value) {
                return false;
            }
        }
        return true;
    }

    _toBooleanLiteral(value) {
        return { 'type': 'id', 'value': value ? 'True' : 'False' }; 
    }

    _isBooleanLiteral(expression) {
        return expression && expression.type === 'id' && (expression.value === 'True' || expression.value === 'False');
    }

    _evaluateExpression(expression) {
        // _150.drop_rate
        if (expression.type === '.') {
            const module = this._getModule(expression.target);
            if (module &&
                expression.member.type === 'id' &&
                Object.prototype.hasOwnProperty.call(module, expression.member.value)) {
                const value = module[expression.member.value];
                if (typeof value === 'number') {
                    return { type: 'number', value: value };
                }
                if (torchscript.Utility.isTensor(value)) {
                    return value;
                }
            }
        }
        // int(x)
        if (this._isCall(expression, 'int', [ {} ])) {
            return this._evaluateExpression(expression.arguments[0]);
        }
        // float(x)
        if (this._isCall(expression, 'float', [ {} ])) {
            return this._evaluateExpression(expression.arguments[0]);
        }
        // _foo
        if (expression.type == 'id' && this._state[expression.value]) {
            return this._state[expression.value];
        }
        return expression;
    }

    _evaluateBooleanExpression(expression) {
        // torch.eq("zeros", "circular"):
        if (this._isCall(expression, 'torch.eq', [ {}, {} ])) {
            const left = this._evaluateExpression(expression.arguments[0]);
            const right = this._evaluateExpression(expression.arguments[1]);
            if (left.type === 'number' && right.type === 'number') {
                return this._toBooleanLiteral(Number(left.value) === Number(right.value));
            }
            if (left.type === 'string' && right.type === 'string') {
                return this._toBooleanLiteral(left.value === right.value);
            }
        }
        // torch.eq(torch.dim(x4), 2):
        if (this._isCall(expression, 'torch.eq', [ {}, { type: 'number' } ]) &&
            this._isCall(expression.arguments[0], 'torch.dim', [ { type: 'id' } ])) {
            return this._toBooleanLiteral(true); // TODO
        }
        // torch.ne(torch.dim(x4), 4):
        if (this._isCall(expression, 'torch.ne', [ {}, {} ])) {
            const right = this._evaluateExpression(expression.arguments[1]);
            const left = this._evaluateExpression(expression.arguments[0]);
            if (right.type === 'number') {
                if (this._isCall(expression.arguments[0], 'torch.dim', [ { type: 'id' } ]) ||
                    this._isCall(expression.arguments[0], 'torch.len', [ {} ])) {
                    return this._toBooleanLiteral(false); // TODO
                }
            }
            if (left.type === 'number') {
                if (this._isCall(expression.arguments[1], 'torch.size', [ {}, {} ])) {
                    return this._toBooleanLiteral(false); // TODO
                }
            }
            return this._toBooleanLiteral(false); // TODO
        }
        // torch.__is__(None, None)
        if (this._isCall(expression, 'torch.__is__', [ { type: 'id', value: 'None' }, { type: 'id', value: 'None' } ])) {
            return this._toBooleanLiteral(true);
        }
        // torch.__is__(<id>, None)
        if (this._isCall(expression, 'torch.__is__', [ { type: 'id' }, { type: 'id', value: 'None' } ])) {
            const argument = this._state[expression.arguments[0].value];
            return this._toBooleanLiteral(!argument && argument.value == 'None');
        }
        // torch.__is__(annotate(Optional[int], None), None)
        if (this._isCall(expression, 'torch.__is__', [ { type: 'call' }, { type: 'id', value: 'None' } ]) &&
            this._isCall(expression.arguments[0], 'annotate', [ {}, { type: 'id', value: 'None' } ])) {
            return this._toBooleanLiteral(true);
        }
        // _torch.__is__(1, None)
        if (this._isCall(expression, 'torch.__is__', [ { type: 'number' }, { type: 'id', value: 'None' } ])) {
            return this._toBooleanLiteral(false);
        }
        // torch.__isnot__(<id>, None)
        if (this._isCall(expression, 'torch.__isnot__', [ { type: 'id' }, { type: 'id', value: 'None' } ])) {
            let argumentExpression = expression.arguments[0];
            if (this._state[argumentExpression.value]) {
                argumentExpression = this._state[argumentExpression.value];
            }
            if (argumentExpression) {
                return this._toBooleanLiteral(argumentExpression.value !== 'None');
            }
        }
        // torch.__isnot__(self.fc1.bias, None)
        if (this._isCall(expression, 'torch.__isnot__', [ { type: '.' }, { type: 'id', value: 'None' } ])) {
            const parameter = this._getParameter(expression.arguments[0]);
            if (parameter) {
                return this._toBooleanLiteral(true);
            }
        }
        // torch.lt(0.5, 0.)
        if (this._isCall(expression, 'torch.lt', [ { type: 'number' }, { type: 'number' } ])) {
            return this._toBooleanLiteral(Number(expression.arguments[0].value) < Number(expression.arguments[1].value));
        }
        // torch.gt(0.5, 0.)
        if (this._isCall(expression, 'torch.gt', [ {}, {} ])) {
            const left = this._evaluateExpression(expression.arguments[0]);
            const right = this._evaluateExpression(expression.arguments[1]);
            if (left.type === 'number' && right.type === 'number') {
                return this._toBooleanLiteral(Number(left.value) > Number(right.value));
            }
        }
        // torch.__not__(...)
        if (this._isCall(expression, 'torch.__not__', [ { type: 'id' } ])) {
            let argumentExpression = expression.arguments[0];
            if (!this._isBooleanLiteral(argumentExpression)) {
                argumentExpression = this._state[argumentExpression.value];
            }
            if (this._isBooleanLiteral(argumentExpression)) {
                switch (argumentExpression.value) {
                    case 'True': return this._toBooleanLiteral(false);
                    case 'False': return this._toBooleanLiteral(true);
                }
            }
        }
        // torch.is_scripting()
        if (this._isCall(expression, 'torch.is_scripting', [])) {
            return this._toBooleanLiteral(true);
        }
        // _2.training
        if (expression.type === '.') {
            const module = this._getModule(expression.target);
            if (module &&
                expression.member.type === 'id' &&
                Object.prototype.hasOwnProperty.call(module, expression.member.value)) {
                const value = module[expression.member.value];
                if (Object(value) !== value) {
                    if (value === true || value === false) {
                        return this._toBooleanLiteral(value);
                    } 
                }
            }
        }
        return null;
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
