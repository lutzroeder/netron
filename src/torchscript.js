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
        var identifier = context.identifier; 
        var extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'pt' || extension == 'pth' || extension == 'pkl' || extension == 'h5' || extension == 't7' ||
            extension == 'dms' || extension == 'model' || extension == 'ckpt' || identifier.endsWith('.pth.tar')) {
            if (torchscript.ModelFactory._openContainer(context.buffer)) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        var identifier = context.identifier;
        try {
            var container = torchscript.ModelFactory._openContainer(context.buffer);
            return torchscript.Metadata.open(host).then((metadata) => {
                try {
                    return new torchscript.Model(metadata, container);
                }
                catch (error) {
                    host.exception(error, false);
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new torchscript.Error(message + " in '" + identifier + "'.");
                }
            });
        }
        catch (error) {
            host.exception(error, false);
            var message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            return Promise.reject(new torchscript.Error(message + " in '" + identifier + "'."));
        }
    }

    static _openContainer(buffer) {
        if (buffer && buffer.length > 2 && buffer[0] == 0x50 && buffer[1] == 0x4B) {
            var archive = new zip.Archive(buffer);
            var container = { };
            container.version = archive.entries.find((entry) => entry.name == 'version' || entry.name.endsWith('/version'));
            if (container.version) {
                container.prefix = container.version.name.substring(0, container.version.name.length - 7);
                container.model = archive.entries.find((entry) => entry.name == container.prefix + 'model.json');
                container.entries = archive.entries;
                if (container.version && container.model) {
                    return container;
                }
            }
        }
        return null;
    }
};

torchscript.Model = class { 

    constructor(metadata, container) {
        var textDecoder = new TextDecoder('utf-8');
        var model = JSON.parse(textDecoder.decode(container.model.data));
        var version = JSON.parse(textDecoder.decode(container.version.data));
        this._format = 'TorchScript v' + version.toString();
        if (model.producerName) {
            this._producer = model.producerName;
            if (model.producerVersion) {
                this._producer = this._producer + ' v' + model.producerVersion;
            }
        }
        this._graphs = [];
        this._graphs.push(new torchscript.Graph(metadata, container, model.mainModule, model.tensors));
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

    constructor(metadata, container, mainModule, tensors) {
        this._name = mainModule.name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        container.tensors = tensors.map((tensor) => new torchscript.Tensor(tensor, container));

        var context = new torchscript.GraphContext(container, mainModule);

        container.parameters = {};
        var queue = [ mainModule ];
        while (queue.length > 0) {
            var module = queue.shift();
            if (module.parameters) {
                for (var parameter of module.parameters) {
                    if (parameter.tensorId) {
                        var tensorId = parseInt(parameter.tensorId, 10);
                        parameter.initializer = container.tensors[tensorId];
                        if (parameter.outputs && parameter.outputs.length == 1) {
                            container.parameters[parameter.outputs[0]] = parameter;
                        }
                    }
                }
            }
            if (module.submodules) {
                for (var submodule of module.submodules) {
                    submodule.parent = module;
                    queue.push(submodule);
                }
            }
        }

        for (var input of context.inputs) {
            this._inputs.push(new torchscript.Parameter(input, true, [
                new torchscript.Argument(input, null, null)
            ]));
        }
        for (var output of context.outputs) {
            this._outputs.push(new torchscript.Parameter(output, true, [
                new torchscript.Argument(output, null, null)
            ]));
        }

        for (var node of context.nodes) {
            this._nodes.push(new torchscript.Node(metadata, container, null, node));
        }

        this._loadModule(metadata, container, mainModule);
    }

    _loadModule(metadata, container, module) {
        if (module.parameters && module.parameters.length > 0 && !module.hide) {
            var node = new torchscript.Node(metadata, container, module, null);
            this._nodes.push(node);
        }
        if (module.submodules) {
            for (var submodule of module.submodules) {
                this._loadModule(metadata, container, submodule);
            }
        }
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

        var input = null;
        var argument = null;
        var parameter = null;

        if (module) {
            this._operator = 'Module';
            if (module.parameters) {
                for (parameter of module.parameters) {
                    this._inputs.push(new torchscript.Parameter(parameter.name, true, [
                        new torchscript.Argument('', null, parameter.initializer || null)
                    ]));
                    if (parameter.outputs) {
                        this._outputs.push(new torchscript.Parameter(parameter.name, true,
                            parameter.outputs.map((id) => new torchscript.Argument(id, null, null))
                        ));
                    }
                }
            }
        }

        if (node) {
            this._operator = node.name;
            this._name = '';

            var schema = metadata.getSchema(this._operator);

            module = null; 
            var match = true;
            var count = 0;
            for (input of node.inputs) {
                for (argument of input) {
                    parameter = container.parameters[argument.id];
                    if (parameter) {
                        if (parameter.module && (module == null || module == parameter.module)) {
                            module = parameter.module;
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
            if (module && module.parameters.length == count && match) {
                module.hide = true;
                for (input of node.inputs) {
                    for (argument of input) {
                        parameter = container.parameters[argument.id];
                        if (parameter && parameter.initializer) {
                            argument.initializer = parameter.initializer;
                        }
                    }
                }
            }
            else {
                module = null;
            }

            for (var inputIndex = 0; inputIndex < node.inputs.length; inputIndex++) {
                var inputName = inputIndex.toString(); 
                if (schema && schema.inputs && schema.inputs.length > inputIndex) {
                    inputName = schema.inputs[inputIndex].name;
                }
                this._inputs.push(new torchscript.Parameter(inputName, true,
                    node.inputs[inputIndex].map((input) => new torchscript.Argument(input.id, null, input.initializer || null))
                ));
            }

            for (var outputIndex = 0; outputIndex < node.outputs.length; outputIndex++) {
                var outputName = outputIndex.toString(); 
                if (schema && schema.outputs && schema.outputs.length > outputIndex) {
                    outputName = schema.outputs[outputIndex].name;
                }
                this._outputs.push(new torchscript.Parameter(outputName, true, [
                    new torchscript.Argument(node.outputs[outputIndex], null, null)
                ]));
            }

            for (var attributeIndex = 0; attributeIndex < node.attributes.length; attributeIndex++) {
                var attributeSchema = null;
                var attributeName = attributeIndex.toString();
                var attributeValue = node.attributes[attributeIndex];
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
            if (module.name) {
                var current = module;
                this._name = current.name;
                while (current.parent != null) {
                    current = current.parent;
                    this._name = [ current.name, this._name ].join('.')
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
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        var schema = this._metadata.getSchema(this._operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (var attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (var input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
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
                            var number = parseInt(item.value, 10);
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

    constructor(tensor, container) {
        this._type = new torchscript.TensorType(tensor.dataType, new torchscript.TensorShape(tensor.dims));
        var key = container.prefix + tensor.data.key;
        var entry = container.entries.find((entry) => entry.name == key);
        this._name = tensor.data.key;
        this._data = entry.data;
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
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return torchscript.Tensor._stringify(value, '', '    ');
    }

    _context() {
        var context = {};
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
        var results = [];
        var dimensions = context.dimensions;
        if (dimensions.length == 0) {
            dimensions = [ 1 ];
        }
        var size = dimensions[dimension];
        if (dimension == dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
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
                        results.push(new long.Long(context.dataView.getUint32(context.index, true), context.dataView.getUint32(context.index + 4, true), true));
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
            for (var j = 0; j < size; j++) {
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
            var result = [];
            result.push(indentation + '[');
            var items = value.map((item) => torchscript.Tensor._stringify(item, indentation + indent, indent));
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
        switch(dataType) {
            case 'FLOAT': this._dataType = 'float32'; break;
            case 'DOUBLE': this._dataType = 'float64'; break;
            case 'INT32': this._dataType = 'int32'; break;
            case 'INT64': this._dataType = 'int64'; break;
            default: throw new torchscript.Error("Unknown tensor data type '" + dataType + "'.");
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
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
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
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

torchscript.GraphContext = class {

    constructor(container, mainModule) {

        this._container = container;
        this._mainModule = mainModule;

        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        this._moduleMap = {};
        this._argumentMap = {};
        this._numToTensorMap = {};

        if (mainModule.torchscriptArena && mainModule.torchscriptArena.key) {
            var codeKey = container.prefix + mainModule.torchscriptArena.key;
            var codeEntries = container.entries.filter((e) => e.name === codeKey);
            if (codeEntries.length == 1) {
                var codeEntry = codeEntries[0];
                var textDecoder = new TextDecoder('utf-8');
                var code = textDecoder.decode(codeEntry.data);
                var reader = new torchscript.PythonReader(code);
                var statements = reader.statements();
                var method = statements.find((statement) => statement.type == 'def' && statement.name == 'forward');
                if (method) {
                    this._body = method.body;
                    var methodParameters = method.parameters;
                    if (methodParameters.length > 0 && methodParameters[0].name == 'self') {
                        methodParameters.shift();
                    }
                    for (var parameter of methodParameters) {
                        this._parameter(parameter);
                    }

                    if (this._body.length >= 2) {
                        var returnStatement = this._body[this._body.length - 1];
                        var assignStatement = this._body[this._body.length - 2];
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
                        var statement = this._body.shift();
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
        var type = parameter.parameterType; 
        if (type.type == 'type' && type.value == 'Tuple' && type.arguments && type.arguments.length > 0) {
            if (this._body.length > 0) {
                var statement = this._body[0];
                if (statement.expression.type == 'identifier' && statement.expression.value == parameter.name) {
                    if (statement.type === '=' && statement.target.type === 'identifier_list') {
                        for (var input of statement.target.value) {
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
            var variable = this._variable();
            if (this._nodeExpression(statement.expression, variable)) {
                this._outputs.push(variable.value);
                return true;
            }
            if (statement.expression.type == 'identifier') {
                this._outputs.push(statement.expression.value);
                return true;
            }
            if (statement.expression.type == 'tuple') {
                var outputs = [];
                for (var expression of statement.expression.value) {
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
        if (expression.type == 'call' && (target.type == 'identifier' || target.type == 'identifier_list')) {
            var name = this._name(expression.target);
            var namespace = 'torch.';
            if (name.startsWith(namespace)) {
                var node = {};
                node.name = name.substring(namespace.length);
                node.inputs = [];
                node.outputs = [];
                node.attributes = [];
                var args = expression.arguments;
                while (args.length > 0) {
                    var argument = args[0];
                    argument = this._moduleTensor(argument);
                    if (argument.type == 'identifier' && this._argumentMap[argument.value]) {
                        argument = this._argumentMap[argument.value];
                        delete this._argumentMap[argument.value];
                    }
                    if (argument.type == 'identifier') {
                        node.inputs.push([ { id: argument.value } ]);
                        args.shift();
                        continue;
                    }
                    if (argument.type == 'list') {
                        var list = [];
                        for (var input of argument.value) {
                            var variable = this._variable();
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
                    variable = this._variable();
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
                        var constantId = [ argument.target.value, argument.member.value ].join('.');
                        var constantIndex = parseInt(argument.member.value.substring(1), 10);
                        var constantTensor = this._container.tensors[constantIndex];
                        node.inputs.push([ { id: constantId, initializer: constantTensor } ]);
                        args.shift();
                        continue;
                    }
                    throw new torchscript.Error('Unknown function argument.');
                }
                while (args.length > 0) {
                    if (args[0].type == 'list') {
                        for (var i = 0; i < args[0].value.length; i++) {
                            args[0].value[i] = this._attributeExpression(args[0].value[i]);
                        }
                    }
                    var intExpression = this._attributeExpression(args[0]);
                    if (intExpression) {
                        args[0] = intExpression;
                    }
                    node.attributes.push(args[0]);
                    args.shift();
                }
                if (target.type == 'identifier') {
                    node.outputs.push(target.value);
                }
                if (target.type == 'identifier_list') {
                    for (var identifier of target.value) {
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
            var replace = this._attributeExpression(expression.arguments[0]);
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
                var size = statement.expression.arguments[0];
                if (size.type == 'call' &&
                    size.arguments.length == 2 &&
                    this._name(size.target) == 'torch.size' &&
                    size.arguments[0].type == 'identifier' &&
                    size.arguments[1].type == 'number') {
                    this._numToTensorMap[statement.target.value] = this._name(size.target) + '(' + size.arguments.map((a) => a.value.toString()).join(',') + ')';
                    return true;
                }
                if (size.type == 'identifier') {
                    var duplicate1 = this._numToTensorMap[size.value];
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
                var duplicate2 = this._numToTensorMap[statement.expression.arguments[0].value];
                if (duplicate2) {
                    this._numToTensorMap[statement.target.value] = duplicate2;
                    return true;
                }
            }
        }
        return false;
    }

    _module(expression) {
        var module;
        var submodule;
        if (expression.type === '.') {
            module = this._module(expression.target);
            if (module && module.submodules) {
                for (submodule of module.submodules) {
                    if (submodule.name === expression.member.value) {
                        return submodule;
                    }
                }
            }
        }
        if (expression.type == 'call' && 
            expression.target.type == 'identifier' && expression.target.value == 'getattr' && expression.arguments.length == 2) {
            module = this._module(expression.arguments[0]);
            if (!module) {
                return null;
            }
            var name = null;
            if (expression.arguments[1].type == 'string') {
                name = expression.arguments[1].value.substring(1, expression.arguments[1].value.length - 1);
            }
            if (module) {
                for (submodule of module.submodules) {
                    if (submodule.name === name) {
                        return submodule;
                    }
                }
            }
        }
        if (expression.type == 'identifier') {
            if (expression.value == 'self') {
                return this._mainModule;
            }
            module = this._moduleMap[expression.value];
            if (module) {
                return module;
            }
        }
        return null;
    }

    _moduleStatement(statement) {
        if (statement.type == '=' && 
            statement.target.type === 'identifier') {
            var moduleName = statement.target.value;
            var module = this._module(statement.expression);
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
            var targetModule = this._module(expression.target);
            if (targetModule && targetModule.parameters) {
                for (var parameter of targetModule.parameters) {
                    parameter.module = targetModule;
                    if (parameter.name === expression.member.value) {
                        parameter.outputs = parameter.outputs || [];
                        parameter.outputs.push(target.value);
                        return true;
                    }
                }
                targetModule.unresolvedParameters = targetModule.unresolvedParameters || [];
                for (var unresolvedParameter of targetModule.unresolvedParameters) {
                    unresolvedParameter.module = targetModule;
                    if (unresolvedParameter.name === expression.member.value) {
                        unresolvedParameter.outputs = unresolvedParameter.outputs || [];
                        unresolvedParameter.outputs.push(target.value);
                        return true;
                    }
                }
                targetModule.unresolvedParameters.push({
                    module: targetModule,
                    name: expression.member.value,
                    outputs: [ target.value ]
                });
                return true;
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

torchscript.PythonReader = class {

    constructor(text) {
        this._text = text;
        this._position = 0;
        this._lineEnd = -1;
        this._lineStart = 0;
        this._line = -1;
        this._indentation = [];
    }

    whitespace() {
        for (;;) {
            while (this._position > this._lineEnd) {
                this._lineStart = this._lineEnd + 1;
                this._position = this._lineStart;
                if (this._position >= this._text.length) {
                    return false;
                }
                this._lineEnd = this._text.indexOf("\n", this._position);
                if (this._lineEnd === -1) {
                    this._lineEnd = this._text.length;
                }
                this._line++;
            }
            var c = this._text[this._position];
            switch (c) {
                case " ":
                case "\r":
                case "\t":
                    this._position++;
                    break;
                case "#":
                    this._position = this._lineEnd;
                    break;
                default:
                    return true;
            }
        }
    }
    
    tokenize() {
        if (!this.whitespace()) {
            this._token = { type: 'eof', value: "" };
            return this._token;
        }
        var c = this._text[this._position];
        if (c == '\n') {
            this._token = { type: 'newline', value: c };
            return this._token;
        }
        if (c === '=' || c === '(' || c === ')' || c === ":" || c === "," || c === '[' || c === ']') {
            this._token = { type: 'separator', value: c };
            return this._token;
        }
        var position = this._position + 1;
        if (c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c === "_") {
            while (position < this._lineEnd) {
                c = this._text[position];
                if (c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c >= "0" && c <= "9" || c === "_" || c === "+" || c === "-") {
                    position++;
                    continue;
                }
                break;
            }
            var identifier = this._text.substring(this._position, position);
            if (identifier == 'True' || identifier == 'False') {
                this._token = { type: 'boolean', value: identifier };
            }
            else {
                this._token = { type: 'identifier', value: identifier };
            }
            return this._token;
        }
        if (c === "-") {
            if (position < this._lineEnd) {
                if (this._text[position] === '>') {
                    position++;
                    this._token = { type: 'arrow', value: '->' };
                    return this._token;
                }
            }
        }
        if (c >= "0" && c <= "9" || c === "-" || c === "+") {
            while (position < this._lineEnd) {
                c = this._text[position];
                if (c >= "a" && c <= "z" || c >= "A" && c <= "Z" || c >= "0" && c <= "9" || c === "_" || c === "+" || c === "-" || c === ".") {
                    position++;
                    continue;
                }
                break;
            }
            this._token = { type: 'number', value: this._text.substring(this._position, position) };
            return this._token;
        }
        if (c === "\"" || c === "'") {
            var quote = c;
            while (position < this._lineEnd) {
                c = this._text[position];
                if (c === "\\" && position < this._lineEnd) {
                    position += 2;
                    continue;
                }
                position++;
                if (c === quote) {
                    break;
                }
            }
            this._token = { type: 'string', value: this._text.substring(this._position, position) };
            return this._token;
        }
        if (c === '.') {
            this._token = { type: 'dot', value: c };
            return this._token;
        }
        throw new torchscript.Error("Unexpected token '" + c + "'" + this.location());
    }

    peek() {
        if (!this._cache) {
            this._token = this.tokenize();
            this._cache = true;
        }
        return this._token;
    }
    
    read() {
        if (!this._cache) {
            this._token = this.tokenize();
        }
        this._position += this._token.value.length;
        this._cache = false;
        return this._token;
    }

    match(value) {
        if (this.peek().value === value) {
            this.read();
            return true;
        }
        return false;
    }

    expect(value) {
        var token = this.read();
        if (token.value !== value) {
            throw new torchscript.Error("Unexpected '" + token + "' instead of '" + value + "'" + this.location());
        }
    }

    location() {
        return " at " + (this._line + 1).toString() + ":" + (this._position - this._lineStart + 1).toString();
    }

    letter(c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }

    number(c) {
        return c >= '0' && c <= '9';
    }

    identifier() {
        var token = this.peek();
        if (token.type == 'identifier') {
            this.read();
            return token;
        }
        return null;
    }

    literal() {
        var token = this.peek();
        if (token.type == 'string' || token.type == 'number' || token.type == 'boolean') {
            this.read();
            return token;
        }
        return null;
    }

    typeArguments() {
        var list = [];
        this.expect('[');
        while (!this.match(']')) {
            var type = this.type();
            if (type == null) {
                throw new torchscript.Error('Expected type ' + this.location());
            }
            list.push(type);
            if (!this.match(',')) {
                this.expect(']');
                break;
            }
        }
        return list;
    }

    type() {
        var identifier = this.identifier();
        if (identifier) {
            var type = { type: 'type', value: identifier.value };
            if (this.peek().value === '[') {
                type.arguments = this.typeArguments();
            }
            return type;
        }
        return null;
    }

    parameter() {
        var identifier = this.identifier();
        if (identifier != null) {
            var parameterType = null
            if (this.match(':')) {
                parameterType = this.type();
            }
            return { type: 'parameter', name: identifier.value, parameterType: parameterType };
        }
        return null;
    }

    parameters() {
        var list = [];
        this.expect('(');
        while (!this.match(')')) {
            this.match('\n');
            list.push(this.parameter());
            this.match('\n');
            if (!this.match(',')) {
                this.expect(')');
                break;
            }
        }
        return list;
    }

    arguments() {
        var list = [];
        this.expect('(');
        while (!this.match(')')) {
            var expression = this.expression();
            if (expression == null) {
                throw new torchscript.Error('Expected expression ' + this.location());
            }
            list.push(expression);
            if (!this.match(',')) {
                this.expect(')');
                break;
            }
        }
        return list;
    }

    expression() {
        var stack = [];
        for (;;) {
            var identifier = this.identifier();
            if (identifier) {
                stack.push(identifier);
                continue;
            }
            var literal = this.literal();
            if (literal) {
                stack.push(literal);
                continue;
            }
            if (this.match('.')) {
                stack.push({
                    type: '.',
                    target: stack.pop(),
                    member: this.identifier(),
                });
                continue;
            }
            if (this.peek().value === '(') {
                if (stack.length == 0) {
                    stack.push({ type: 'tuple', value: this.arguments() });
                }
                else {
                    stack.push({ type: 'call', target: stack.pop(), arguments: this.arguments() });
                }
                continue;
            }
            if (this.peek().value === '[') {
                stack.push({ type: 'list', value: this.expressions() });
                continue;
            }
            if (this.match('=')) {
                stack.push({ type: '=', target: stack.pop(), expression: this.expression() });
                continue;
            }
            break;
        }

        if (stack.length == 1) {
            return stack.pop();
        }
        if (stack.length != 0) {
            throw new torchscript.Error('Unexpected expression ' + this.location());
        }
        return null;
    }

    expressions() {
        var list = [];
        this.expect('[');
        while (!this.match(']')) {
            var expression = this.expression();
            if (expression == null) {
                throw new torchscript.Error('Expected expression ' + this.location());
            }
            list.push(expression);
            if (!this.match(',')) {
                this.expect(']');
                break;
            }
        }
        return list;
    }

    statement() {
        var stack = [];
        while (this.peek().type !== 'eof') {

            if (this.match('def')) {
                var node = { type: 'def' };
                node.name = this.identifier().value;
                node.parameters = this.parameters();
                if (this.match('->')) {
                    node.returnType = this.type();
                }
                this.expect(':');
                this.expect('\n');
                var position = this._position;
                while (this.match('\n')) {
                    position = this._position;
                }
                this.peek();
                this._indentation.push(this._text.substring(position, this._position));
                this._position = position;
                node.body = this.statements();
                this._indentation.pop();
                stack.push(node);
                break;
            }

            if (this.match('return')) {
                stack.push({ type: 'return', expression: this.expression() });
                break; 
            }

            var expression = this.expression();
            if (expression) {
                if (expression.type == 'identifier') {
                    if (this.peek().value === ',') {
                        var list = [ expression ];
                        while (this.match(',')) {
                            var identifier = this.identifier();
                            if (!identifier) {
                                if (this.peek().value != '=') {
                                    throw new torchscript.Error('Expected identifier' + this.location());
                                }
                            }
                            list.push(identifier);
                        }
                        expression = { type: 'identifier_list', value: list };
                        if (this.match('=')) {
                            expression = { type: '=', target: expression, expression: this.expression() };
                        }
                    }
                }
                if (expression.type == '=') {
                    stack.push(expression);
                    this.match('\n');
                    break;
                }
                throw new torchscript.Error('Unhandled expression ' + this.location);
            }

            if (this.match('\n')) {
                break;
            }
        }

        if (stack.length == 1) {
            return stack.pop();
        }
        if (stack.length != 0) {
            throw new torchscript.Error('Unexpected statement ' + this.location());
        }
        return null;
    }

    statements() {
        var indentation = this._indentation.join('');
        var stack = [];
        while (this._position < this._text.length) {
            if (this._text.substring(this._position, this._position + indentation.length) !== indentation) {
                return stack;
            }
            this._position = this._position + indentation.length;

            var statement = this.statement();
            if (statement) { 
                stack.push(statement);
                continue;
            }
        }
        return stack;
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
