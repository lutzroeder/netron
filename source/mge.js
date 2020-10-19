/* jshint esversion: 6 */

// Experimental

var mge = mge || {};
var base = base || require('./base');

mge.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'mge') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return host.require('./mge-pickle').then((pickle) => {
            const identifier = context.identifier;
            return mge.Metadata.open(host).then((metadata) => {
                try {
                    const container = new mge.Container(context.buffer, pickle, (error, fatal) => {
                        const message = error && error.message ? error.message : error.toString();
                        host.exception(new mge.Error(message.replace(/\.$/, '') + " in '" + identifier + "'."), fatal);
                    }); 
                    return new mge.Model(metadata, container);
                }
                catch (error) {
                    host.exception(error, false);
                    const message = error && error.message ? error.message : error.toString();
                    throw new mge.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
            });
        });
    }
};

mge.Model = class {

    constructor(metadata, container) {
        this._format = container.format;
        this._graphs = [ new mge.Graph(metadata, container) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

mge.Graph = class {

    constructor(metadata, container) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._groups = true;
        //this._littleEndian = container.littleEndian;
        if (container.data) {
            const data = container.data;
            this._type = (data.__module__ && data.__name__) ? (data.__module__ + '.' + data.__name__) : '';
            const input = 'data';
            this._inputs.push(new mge.Parameter(input, true, [ new mge.Argument(input, null, null) ]));
            const outputs = this._loadModule(metadata, container.data, [], [ input ]);
            for (const output of outputs) {
                this._outputs.push(new mge.Parameter(output, true, [ new mge.Argument(output, null, null) ]));
            }
        }
        
        else if (container.state) {
            for (const state_group of container.state) {
                const attributes = state_group.attributes || [];
                const inputs = state_group.states.map((parameter) => {
                    return new mge.Parameter(parameter.name, true,
                        parameter.arguments.map((state) => {
                            const tensor = new mge.Tensor(state.id, state.value, this._littleEndian);
                            return new mge.Argument(state.id, null, tensor);
                        }));
                });
                const obj = {
                    name: state_group.name,
                    type: state_group.type || "megengine.module",
                    attributes: attributes,
                    inputs: inputs,
                    outputs: []
                };
                this._nodes.push(new mge.Node(metadata, '', obj, null));
            }
        }
        
    }

    _loadModule(metadata, parent, groups, inputs) {
        for(const [key, value] of Object.entries(parent)){
            if(value.__module__===undefined){
                continue;
            }
            if (key && value) {
                const type = value.__module__ + '.' + value.__name__;
                switch (type) {
                    case 'megengine.module.sequential.Sequential': 
                        groups.push(key);
                        inputs = this._loadModule(metadata, value, groups, inputs);
                        groups.pop(key);
                        break;
                    case 'megengine.module.inception.InceptionE': {
                        groups.push(key);
                        const node = this._createNode(metadata, groups, key, value, inputs, this._littleEndian);
                        inputs = [ node.name ];
                        groups.pop(key);
                        break;
                    }
                    default: {
                        const node = this._createNode(metadata, groups, key, value, inputs);
                        inputs = [ node.name ];
                        break;
                    }
                }
            }
        }
        return inputs;
    }

    _createNode(metadata, groups, key, obj, args) {

        const type = obj.__module__ + '.' + obj.__name__;
        const schema = metadata.type(type);

        let inputSchema = [ { name: 'input'} ];
        if (schema && schema.inputs && schema.inputs.length > 0) {
            inputSchema = schema.inputs.slice();
        }

        const inputs = [
            new mge.Parameter(inputSchema.shift().name, true, args.map((argument) => {
                return new mge.Argument(argument, null, null);
            }))
        ];

        const attributes = [];
        for (const [key, value] of Object.entries(obj)){           
            if(key.startsWith('_')){
                continue;
            }
            if(value && value.dtype){
                let visible = true;
                let inputName = '';
                if (inputSchema.length > 0) {
                    const input = inputSchema.shift();
                    inputName = input.name;
                    visible = input.visible === false ? false : true;
                }
                if (value) {
                    let initializer = null;
                    if (value) {
                        initializer = new mge.Tensor('', value.data, this._littleEndian);
                    }
                    inputs.push(new mge.Parameter(inputName || key, visible, [ new mge.Argument('', null, initializer) ]));
                 }
             } 
             else{
                attributes.push({ name: key, value: obj});   
             }
        }         

        const group = groups.join('/');
        const name = group ? (group + '/' + key) : key;

        const outputs = [ new mge.Parameter('output', true, [ new mge.Argument(name, null, null) ]) ];

        const item = {
            name: name,
            type: type,
            attributes: attributes,
            inputs: inputs,
            outputs: outputs
        };
        const node = new mge.Node(metadata, group, item, {});
        this._nodes.push(node);
        return node;
    }

    static _getParameters(module) {
        const parameters = [];
        if (module && module.__module__ && module.__name__) {
            for (const key of Object.keys(module)) {
                if (mge.Utility.isTensor(module[key])) {
                    const parameter = module[key];
                    parameter.__id__ = key;
                    parameters.push(parameter);
                }
            }
        }
        return parameters;
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

mge.Parameter = class {
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

mge.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new mge.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
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

mge.Node = class {

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
                return new mge.Attribute(schema, attribute.name, attribute.value);
            });
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
        return this._type.startsWith('megengine.module.') && this._type !== 'megengine.module.Module';
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

mge.Attribute = class {

    constructor(schema, name, value) {
        this._name = name;
        this._value = Object.getOwnPropertyDescriptor(value, name).value;
        
        if(this._name === 'training'){
            this._visible = false;
            this._type = 'boolean';
            return;
        }

        if(name === 'compute_mode' || name === 'conv_mode'){
            this._value = 'module: ' + this._value.__module__ ;
        }

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

mge.Tensor = class {

    constructor(name, tensor, littleEndian) {
        this._name = name || '';
        this._type = new mge.TensorType(tensor.dtype.name, new mge.TensorShape(tensor.shape));
        this._data = tensor.data;
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
        return mge.Tensor._stringify(value, '', '    ');
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

        context.data = this._data;
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
                    case 'uint8':
                        results.push(context.dataView.getUint8(context.index, this._littleEndian));
                        context.index++;
                        context.count++;
                        break;
                    case 'qint8':
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
            const items = value.map((item) => mge.Tensor._stringify(item, indentation + indent, indent));
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

mge.TensorType = class {

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
        return this.dataType + this._shape.toString();
    }
};

mge.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

mge.Metadata = class {

    static open(host) {
        if (mge.Metadata._metadata) {
            return Promise.resolve(mge.Metadata._metadata);
        }
        return host.request(null, 'mge-metadata.json', 'utf-8').then((data) => {
            mge.Metadata._metadata = new mge.Metadata(data);
            return mge.Metadata._metadata;
        }).catch(() => {
            mge.Metadata._metadata = new mge.Metadata(null);
            return mge.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

mge.Execution = class {

    constructor(python, sources, exceptionCallback) {
        const self = this;
        this._python = python;
        this._sources = sources;
        this._exceptionCallback = exceptionCallback;
        this._utf8Decoder = new TextDecoder('utf-8');
        this._unknownNameMap = new Set();
        this._knownPackageMap = new Set([ 'mge', 'collections', '__builtin__', '_codecs', 'argparse', 'numpy' ]);
        this._packages = new Map();
        this._context = new mge.Execution.Context();
        this._context.scope.builtins = {};
        this._context.scope.builtins.type = { __module__: 'builtins', __name__: 'type' };
        this._context.scope.builtins.module = { __module__: 'builtins', __name__: 'module', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.function = { __module__: 'builtins', __name__: 'function', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.method = { __module__: 'builtins', __name__: 'method', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.dict = { __module__: 'builtins', __name__: 'dict', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.list = { __module__: 'builtins', __name__: 'list', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.str = { __module__: 'builtins', __name__: 'str', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.tuple = { __module__: 'builtins', __name__: 'tuple', __class__: this._context.scope.builtins.type };
        this._context.scope.typing = { __name__: 'typing', __class__: this._context.scope.builtins.module };
        this._context.scope.typing._GenericAlias = { __module__: 'typing', __name__: '_GenericAlias', __class__: this._context.scope.builtins.type };
        this._context.scope.typing._SpecialForm = { __module__: 'typing', __name__: '_SpecialForm', __class__: this._context.scope.builtins.type };
        this._context.scope.typing._VariadicGenericAlias = { __module__: 'typing', __name__: '_VariadicGenericAlias', __class__: this._context.scope.builtins.type };
        this._context.scope.typing.Dict = { __module__: 'typing', __name__: 'Dict', __class__: this._context.scope.typing._VariadicGenericAlias, __origin__: this._context.scope.builtins.dict };
        this._context.scope.typing.List = { __module__: 'typing', __name__: 'List', __class__: this._context.scope.typing._GenericAlias, __origin__: this._context.scope.builtins.list };
        this._context.scope.typing.Optional = { __module__: 'typing', __class__: this._context.scope.typing._SpecialForm };
        this._context.scope.typing.Tuple = { __module__: 'typing', __name__: 'Tuple', __class__: this._context.scope.typing._GenericAlias, __origin__: this._context.scope.builtins.tuple };
        this._context.scope.mge = { __name__: 'mge', __class__: this._context.scope.builtins.module };
        this._context.scope.mge.Tensor = { __module__: 'mge', __name__: 'Tensor', __class__: this._context.scope.builtins.type };
        this._registerConstructor('argparse.Namespace', function (args) {
            this.args = args;
        });
        this._registerConstructor('numpy.dtype', function(obj, align, copy) {   //hsa it
            switch (obj) {
                case 'i1': this.name = 'int8'; this.itemsize = 1; break;
                case 'i2': this.name = 'int16'; this.itemsize = 2; break;
                case 'i4': this.name = 'int32'; this.itemsize = 4; break;
                case 'i8': this.name = 'int64'; this.itemsize = 8; break;
                case 'b1': this.name = 'uint8'; this.itemsize = 1; break;
                case 'u1': this.name = 'uint8'; this.itemsize = 1; break;
                case 'u2': this.name = 'uint16'; this.itemsize = 2; break;
                case 'u4': this.name = 'uint32'; this.itemsize = 4; break;
                case 'u8': this.name = 'uint64'; this.itemsize = 8; break;
                case 'f4': this.name = 'float32'; this.itemsize = 4; break;
                case 'f8': this.name = 'float64'; this.itemsize = 8; break;
                default:
                    if (obj.startsWith('V')) {
                        this.itemsize = Number(obj.substring(1));
                        this.name = 'void' + (this.itemsize * 8).toString();
                    }
                    else if (obj.startsWith('O')) {
                        this.itemsize = Number(obj.substring(1));
                        this.name = 'object';
                    }
                    else if (obj.startsWith('S')) {
                        this.itemsize = Number(obj.substring(1));
                        this.name = 'string';
                    }
                    else if (obj.startsWith('U')) {
                        this.itemsize = Number(obj.substring(1));
                        this.name = 'string';
                    }
                    else if (obj.startsWith('M')) {
                        this.itemsize = Number(obj.substring(1));
                        this.name = 'datetime';
                    }
                    else {
                        throw new mge.Error("Unknown dtype '" + obj.toString() + "'.");
                    }
                    break;
            }
            this.align = align;
            this.copy = copy;
            this.__setstate__ = function(state) {
                switch (state.length) {
                    case 8:
                        this.version = state[0];
                        this.byteorder = state[1];
                        this.subarray = state[2];
                        this.names = state[3];
                        this.fields = state[4];
                        this.elsize = state[5];
                        this.alignment = state[6];
                        this.int_dtypeflags = state[7];
                        break;
                    default:
                        throw new mge.Error("Unknown numpy.dtype setstate length '" + state.length.toString() + "'.");
                }
            };
        });
        this._registerConstructor('numpy.core.multiarray._reconstruct', function(subtype, shape, dtype) {   //has it
            this.subtype = subtype;
            this.shape = shape;
            this.dtype = dtype;
            this.__setstate__ = function(state) {
                this.version = state[0];
                this.shape = state[1];
                this.typecode = state[2];
                this.is_f_order = state[3];
                this.rawdata = state[4];
            };
            this.__read__ = function(unpickler) {
                const array = {};
                const subtype = this.subtype.split('.');
                array.__name__ = subtype.pop();
                array.__module__ = subtype.join('.');
                array.dtype = this.typecode;
                array.shape = this.shape;
                let size = array.dtype.itemsize;
                for (let i = 0; i < array.shape.length; i++) {
                    size = size * array.shape[i];
                }
                if (typeof this.rawdata == 'string') {
                    array.data = unpickler.unescape(this.rawdata, size);
                    if (array.data.length != size) {
                        throw new mge.Error('Invalid string array data size.');
                    }
                }
                else {
                    array.data = this.rawdata;
                    if (array.data.length != size) {
                        // throw new mge.Error('Invalid array data size.');
                    }
                }
                return array;
            };
        });
        this._registerFunction('collections.Counter', function(/* iterable */) {
            return {};
        });
        this._registerFunction('collections.OrderedDict', function(args) {
            const obj = new Map();
            obj.__setitem__ = function(key, value) {
                obj.set(key, value);
            };
            if (args) {
                for (const arg of args) {
                    obj.__setitem__(arg[0], arg[1]);
                }
            }
            return obj;
        });
       this._registerFunction('numpy.core.multiarray.scalar', function(dtype, rawData) {
            let data = rawData;
            if (rawData.constructor !== Uint8Array) {
                data = new Uint8Array(rawData.length);
                for (let i = 0; i < rawData.length; i++) {
                    data[i] = rawData.charCodeAt(i);
                }
            }
            const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
            switch (dtype.name) {
                case 'float32':
                    return dataView.getFloat32(0, true);
                case 'float64':
                    return dataView.getFloat64(0, true);
                case 'uint8':
                    return dataView.getUint8(0, true);
                case 'int8':
                    return dataView.getInt8(0, true);
                case 'int16':
                    return dataView.getInt16(0, true);
                case 'int32':
                    return dataView.getInt32(0, true);
                case 'int64':
                    return new long.Long(dataView.getInt32(0, true), dataView.getInt32(4, true), false);
            }
            throw new mge.Error("Unknown scalar type '" + dtype.name + "'.");
        });        
    }

    get context() {
        return this._context;
    }

    parse(file) {
        const data = this._sources[file];
        if (data) {
            const code = this._utf8Decoder.decode(data);
            const reader = new this._python.Parser(code, file);
            const program = reader.parse();
            if (!program) {
                throw new mge.Error("Module '" + file + "' parse error.");
            }
            return program;
        }
        return null;
    }

    package(name, file, raw) {
        if (this._python && !this._packages.has(name)) {
            file = file || 'code/' + name.split('.').join('/') + '.py';
            const program = this.parse(file);
            if (program) {
                let globals = this._context.getx(name);
                if (globals === undefined) {
                    globals = {};
                    this._context.setx(name, globals);
                }
                globals.__class__ = this._context.scope.builtins.module;
                globals.__name__ = name;
                globals.__file__ = file;
                this._packages.set(name, globals);
                const context = this._context.push(globals);
                this._block(program.body, context);
                if (raw) {
                    return program;
                }
            }
        }
        return this._packages.get(name);
    }

    type(name) {
        const type = this._context.getx(name);
        if (type !== undefined) {
            return type;
        }
        const parts = name.split('.');
        const className = parts.pop();
        const moduleName = parts.join('.');
        const module = this.package(moduleName);
        if (module) {
            return module[className];
        }
        return null;
    }

    invoke(name, args) {
        const target = this.type(name);
        if (target) {
            if (target.__class__ === this._context.scope.builtins.type) {
                const obj = {};
                obj.__proto__ = target;
                if (obj.__init__ && typeof obj.__init__ === 'function') {
                    obj.__init__.apply(obj, args);
                }
                return obj;
            }
            else if (target.__class__ === this._context.scope.builtins.function) {
                if (target.__call__) {
                    return target.__call__(args);
                    // throw new mge.Error('Unexpected function __call__.');
                }
                else {
                    return target.apply(null, args);
                }
            }
        }
        this._raiseUnkownName(name);
        const typeParts = name.split('.');
        const typeName = typeParts.pop();
        const typeModule = typeParts.join('.');
        return {
            __module__: typeModule,
            __name__: typeName
        };
    }

    call(target, name, args, context) {
        const callTarget = this._target(target, context);
        const callArguments = args.map((argument) => this.expression(argument, context));
        if (!callTarget || (name !== null && !callTarget[name])) {
            const targetName = mge.Utility.target(target) + '.' + name;
            if (this.type(targetName)) {
                return this.invoke(targetName, callArguments);
            }
            throw new mge.Error("Unsupported function '" +  targetName + "'.");
        }
        const func = name ? callTarget[name] : callTarget;
        if (func.__class__ === this._context.scope.builtins.type) {
            const obj = {};
            obj.__proto__ = func;
            if (obj.__init__ && typeof obj.__init__ === 'function') {
                obj.__init__.apply(obj, args);
            }
            return obj;
        }
        if (func.__class__ === this._context.scope.builtins.function) {
            if (func.__call__) {
                return func.__call__(callArguments);
            }
        }
        if (func.__class__ === this._context.scope.builtins.method) {
            if (func.__call__) {
                return func.__call__([ callTarget ].concat(callArguments));
            }
        }
        if (typeof func === 'function') {
            return func.apply(callTarget, callArguments);
        }
        throw new mge.Error("Unsupported call expression.");
    }

    apply(method, args, context) {
        const locals = Array.prototype.slice.call(args);
        context = context.push();
        for (const parameter of method.parameters) {
            context.set(parameter.name, locals.shift());
        }
        return this._block(method.body.statements, context);
    }

    _block(statements, context) {
        statements = Array.prototype.slice.call(statements);
        while (statements.length > 0) {
            const statement = statements.shift();
            switch (statement.type) {
                case 'pass': {
                    break;
                }
                case 'return': {
                    return this.expression(statement.expression, context);
                }
                case 'def': {
                    const module = context.get('__name__');
                    const self = this;
                    const parent = context.get('__class__');
                    let type = null;
                    if (parent === this._context.scope.builtins.type) {
                        type = this._context.scope.builtins.method;
                    }
                    else if (parent === this._context.scope.builtins.module) {
                        type = this._context.scope.builtins.function;
                    }
                    else {
                        throw new mge.Error('Invalid function scope.');
                    }
                    const func = {
                        __class__: type,
                        __globals__: context,
                        __module__: module,
                        __name__: statement.name,
                        __code__: statement,
                        __call__: function(args) {
                            return self.apply(this.__code__, args, this.__globals__);
                        }
                    };
                    context.set(statement.name, func);
                    break;
                }
                case 'class': {
                    const scope = {
                        __class__:this._context.scope.builtins.type,
                        __module__: context.get('__name__'),
                        __name__: statement.name,
                    };
                    context.set(statement.name, scope);
                    context = context.push(scope);
                    this._block(statement.body.statements, context);
                    context = context.pop();
                    break;
                }
                case 'var': {
                    context.set(statement.name, undefined);
                    break;
                }
                case '=': {
                    this.expression(statement, context);
                    break;
                }
                case 'if': {
                    const condition = this.expression(statement.condition, context);
                    if (condition === true || condition) {
                        statements = statement.then.statements.concat(statements);
                        break;
                    }
                    else if (condition === false) {
                        statements = statement.else.statements.concat(statements);
                        break;
                    }
                    throw new mge.Error("Unknown condition.");
                }
                case 'for': {
                    if (statement.target.length == 1 &&
                        statement.variable.length === 1 && statement.variable[0].type === 'id') {
                        const range = this.expression(statement.target[0], context);
                        const variable = statement.variable[0];
                        let loop = [];
                        for (const value of range) {
                            loop.push({ type: '=', target: variable, expression: { type: 'number', value: value }});
                            loop = loop.concat(statement.body.statements);
                        }
                        statements = loop.concat(statements);
                        break;
                    }
                    throw new mge.Error("Unsupported 'for' statement.");
                }
                case 'call': {
                    this.expression(statement, context);
                    break;
                }
                case 'import': {
                    for (const module of statement.modules) {
                        const moduleName = mge.Utility.target(module.name);
                        const globals = this.package(moduleName);
                        if (module.as) {
                            context.set(module.as, globals);
                        }
                    }
                    break;
                }
                default: {
                    throw new mge.Error("Unknown statement '" + statement.type + "'.");
                }
            }
        }
    }

    expression(expression, context) {
        const self = context.getx('self');
        switch (expression.type) {
            case '=': {
                const target = expression.target;
                if (target.type === 'id') {
                    context.set(target.value, this.expression(expression.expression, context));
                    return;
                }
                else if (target.type === '[]') {
                    if (target.target.type === 'id' &&
                        target.arguments.type === 'list' &&
                        target.arguments.value.length === 1) {
                        const index = this.expression(target.arguments.value[0], context);
                        if (target.target.value === '__annotations__') {
                            context.set(target.target.value, context.get(target.target.value) || {});
                        }
                        context.get(target.target.value)[index] = this.expression(expression.expression, context);
                        return;
                    }
                }
                else if (target.type === '.' &&
                    target.member.type === 'id') {
                    this.expression(target.target, context)[target.member.value] = this.expression(expression.expression, context);
                    return;
                }
                else if (target.type === 'tuple') {
                    const value = this.expression(expression.expression, context);
                    if  (target.value.length == value.length && target.value.every((item) => item.type === 'id')) {
                        for (let i = 0; i < value.length; i++) {
                            context.set(target.value[i].value, value[i]);
                        }
                        return;
                    }
                }
                break;
            }
            case 'list': {
                return expression.value.map((item) => this.expression(item, context));
            }
            case 'string': {
                return expression.value.substring(1, expression.value.length - 1);
            }
            case 'number': {
                return Number(expression.value);
            }
            case '[]': {
                if (expression.target.type === 'id' &&
                    expression.arguments.type === 'list' &&
                    expression.arguments.value.length === 1) {
                    if (context.get(expression.target.value)) {
                        const index = this.expression(expression.arguments.value[0], context);
                        return context.get(expression.target.value)[index];
                    }
                }
                const target = this.expression(expression.target, context);
                if (target && expression.arguments.type === 'list' &&
                    (target.__class__ === this.context.scope.typing._VariadicGenericAlias ||
                     target.__class__ === this.context.scope.typing._GenericAlias ||
                     target.__class__ === this.context.scope.typing._SpecialForm)) {
                    const type = Object.assign({}, target);
                    type.__args__ = expression.arguments.value.map((arg) => this.expression(arg, context));
                    return type;
                }
                if (expression.arguments.type === 'list' && expression.arguments.value.length === 1) {
                    const index = this.expression(expression.arguments.value[0], context);
                    return target[index];
                }
                break;
            }
            case '.': {
                if (expression.member.type == 'id') {
                    const target = this._target(expression.target, context);
                    return target[expression.member.value];
                }
                throw new mge.Error("Unsupported field expression.");
            }
            case 'call': {
                if (expression.target.type === 'id' && expression.target.value === 'annotate' && expression.arguments.length === 2) {
                    return this.expression(expression.arguments[1], context);
                }
                if (expression.target.type === 'id' && expression.target.value === 'unchecked_cast' && expression.arguments.length === 2) {
                    return this.expression(expression.arguments[1], context);
                }
                if (expression.target.type === '.') {
                    return this.call(expression.target.target, expression.target.member.value, expression.arguments, context);
                }
                return this.call(expression.target, null, expression.arguments, context);
            }
            case 'id': {
                switch (expression.value) {
                    case 'self': return self;
                    case 'None': return null;
                    case 'True': return true;
                    case 'False': return false;
                }
                const type =
                    this._context.scope.builtins[expression.value] ||
                    this._context.scope.typing[expression.value] ||
                    this._context.scope.mge[expression.value];
                if (type &&
                    (type.__class__ === this._context.scope.builtins.type ||
                     type.__class__ === this._context.scope.typing._VariadicGenericAlias ||
                     type.__class__ === this._context.scope.typing._GenericAlias ||
                     type.__class__ === this._context.scope.typing._SpecialForm)) {
                    return type;
                }
                return context.get(expression.value);
            }
            case 'tuple': {
                return expression.value.map((expression) => this.expression(expression, context));
            }
        }
        throw new mge.Error("Unknown expression '" + expression.type + "'.");
    }

    _target(expression, context) {
        let current = expression;
        let packageName = '';
        for (;;) {
            if (current.type === '.' && current.member && current.member.type === 'id') {
                packageName = '.' + current.member.value + packageName;
                current = current.target;
            }
            else if (current.type === 'id' && current.value !== 'self' && current.value !== 'CONSTANTS') {
                packageName = current.value + packageName;
                break;
            }
            else {
                packageName = null;
                break;
            }
        }
        if (packageName) {
            let target = context.getx(packageName);
            if (!target) {
                target = this.package(packageName);
                if (!target) {
                    throw new mge.Error("Failed to resolve module '" + packageName + "'.");
                }
            }
            return target;
        }
        return this.expression(expression, context);
    }

    _registerFunction(name, callback) {
        if (this._context.getx(name)) {
            throw new mge.Error("Function '" + name + "' is already registered.");
        }
        const parts = name.split('.');
        callback.__class__ = this._context.scope.builtins.function;
        callback.__name__ = parts.pop();
        callback.__module__ = parts.join('.');
        this._context.setx(name, callback);
    }

    _registerConstructor(name, callback) {
        if (this._context.getx(name)) {
            throw new mge.Error("Constructor '" + name + "' is already registered.");
        }
        const parts = name.split('.');
        const typeName = parts.pop();
        const typeModule = parts.join('.');
        const type = {
            __class__: this._context.scope.builtins.type,
            __name__: typeName,
            __module__: typeModule,
            __init__: function() {
                callback.apply(this, arguments);
            }
        };
        this._context.setx(name, type);
    }

    _raiseUnkownName(name) {
        if (name && !this._unknownNameMap.has(name)) {
            this._unknownNameMap.add(name);
            if (this._knownPackageMap.has(name.split('.').shift())) {
                this._exceptionCallback(new mge.Error("Unknown function '" + name + "'."), false);
            }
        }
    }
};

mge.Execution.Context = class {

    constructor(parent, scope) {
        this._parent = parent || null;
        this._scope = scope || {};
    }

    push(scope) {
        return new mge.Execution.Context(this, scope);
    }

    pop() {
        return this._parent;
    }

    get scope() {
        return this._scope;
    }

    set(name, value) {
        this._scope[name] = value;
    }

    get(name) {
        if (name in this._scope) {
            return this._scope[name];
        }
        if (this._parent) {
            return this._parent.get(name);
        }
        return undefined;
    }

    setx(name, value) {
        const parts = name.split('.');
        if (parts.length == 1) {
            this.set(parts[0], value);
        }
        else {
            let parent = this.get(parts[0]);
            if (!parent) {
                parent = {};
                this.set(parts[0], parent);
            }
            parts.shift();
            while (parts.length > 1) {
                const part = parts.shift();
                parent[part] = parent[part] || {};
                parent = parent[part];
            }
            parent[parts[0]] = value;
        }
    }

    getx(name) {
        const parts = name.split('.');
        let value = this.get(parts[0]);
        if (value) {
            parts.shift();
            while (parts.length > 0 && value[parts[0]]) {
                value = value[parts[0]];
                parts.shift();
            }
            if (parts.length === 0) {
                return value;
            }
        }
        return undefined;
    }
};

mge.Container = class {

    constructor(buffer, pickle, exception) {
        this._buffer = buffer;
        this._pickle = pickle;
        this._exceptionCallback = exception;
    }

    get format() {
        return 'mge v1.0rc';
    }

    get data() {
        this._unpickle();
        return this._data;
    }

    get state() {
        this._unpickle();
        return this._state;
    }

    _unpickle() {
        if (!this._buffer) {
            return;
        }

        const execution = new mge.Execution(null, [], this._exceptionCallback);
        const unpickler = new this._pickle.Unpickler(this._buffer);

        this._buffer = null;
        this._pickle = null;
        this._exceptionCallback = null;

        const module_source_map = new Map();
        const deserialized_objects = new Map();
        
        const data = unpickler.load((name, args) => execution.invoke(name, args), null);
        if (!data) {
            throw new mge.Error('File format is not mge.');
        }
        this._data = mge.Utility.findRootModule(data);
        if (!this._data) {
            this._state = mge.Utility._findStateDict(data);
        }
        if (!this._data && !this._state && data !== 'None') {
            throw new mge.Error('File does not contain root module or state dictionary.');
        }
    }

};

mge.Utility = class {

    static target(expression) {
        if (expression.type == 'id') {
            return expression.value;
        }
        if (expression.type == '.') {
            return mge.Utility.target(expression.target) + '.' + mge.Utility.target(expression.member);
        }
        return null;
    }

    static isTensor(obj) {
        return obj &&    (obj.__module__ === 'numpy' ) && obj.__name__ && obj.__name__.endsWith('ndarray');
    }

    static isType(obj, type) {
        switch (type) {
            case 'tensor':
                return !Array.isArray(obj) && (mge.Utility.isTensor(obj) || obj === null);
            case 'tensor[]':
                return Array.isArray(obj) && obj.length > 0 && obj.every((tensor) => mge.Utility.isTensor(tensor) || tensor === null);
            case 'boolean':
                return obj === true || obj === false;
            case 'int64':
                return Number.isInteger(obj) || isNaN(obj);
            case 'int64[]':
                return Array.isArray(obj) && obj.every((item) => Number.isInteger(item) || Number.isNaN(item) || item === undefined);
            case 'float32':
            case 'float64':
                return obj !== null && obj !== Object(obj);
            case 'Layout':
            case 'ScalarType':
            case 'MemoryFormat':
                return Number.isInteger(obj);
            case 'Device':
                return obj === null || obj === Object(obj);
            case 'scalar':
                return obj !== null || obj !== Object(obj);
        }
        return true;
    }

    static findRootModule(root) {
        const candidates = [ root, root.model, root.net ];
        for (const obj of candidates) {
            if (obj && obj.__module__) {
                return obj;
            }
        }
        return null;
    }

    static _findStateDict(root) {
        if (!root) {
            return null;
        }
        if (root.encoder && Array.isArray(root.encoder) &&
            root.decoder && Array.isArray(root.decoder) && !root.state_dict) {
            root = root.encoder.concat(root.decoder);
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
        const candidates = [
            root.state_dict, root.state,
            root.model_state, root.model, root.model_state_dict, root.net_dict,
            root.params, root.generator, root.discriminator, root.g_state,
            root.network, root.net, root.netG, root.net_states,
            root.state_dict_stylepredictor, root.state_dict_ghiasi,
            root
        ];
        for (const dict of candidates) {
            let state_dict = null;
            state_dict = state_dict || mge.Utility._convertStateDictList(dict);
            state_dict = state_dict || mge.Utility._convertStateDictMap(dict);
            state_dict = state_dict || mge.Utility._convertStateDictGroupMap(dict);
            if (state_dict) {
                return state_dict;
            }
        }
        return null;
    }

    static _convertStateDictList(list) {
        if (list && Array.isArray(list) && list.every((obj) => obj.__module__ && obj.__name__ && Object.keys(obj).filter((key) => mge.Utility.isTensor(obj[key]).length > 0))) {
            const layers = [];
            for (const obj of list) {
                const layer = { type: obj.__module__ + '.' + obj.__name__, states: [], attributes: [] };
                for (const key of Object.keys(obj)) {
                    const value = obj[key];
                    if (mge.Utility.isTensor(value)) {
                        layer.states.push({ name: key, arguments: [ { id: '', value: value } ] });
                    }
                    else {
                        layer.attributes.push({ name: key, value: value });
                    }
                }
                layers.push(layer);
            }
            return layers;
        }
        if (list && !Array.isArray(list) && !(list instanceof Map)) {
            list = new Map(Object.keys(list).map((key) => [ key, list[key] ]));
        }
        if (list && list instanceof Map) {
            for (const item of list) {
                const key = item[0];
                const value = item[1];
                if (!key || !value) {
                    return null;
                }
                if (mge.Utility.isTensor(value)) {
                    continue;
                }
                if (key.endsWith('._packed_params.dtype')) {
                    continue;
                }
                if (key.endsWith('._packed_params._packed_params') && Array.isArray(value) && value.every((item) => mge.Utility.isTensor(item))) {
                    continue;
                }
                return null;
            }
            const layers = new Map();
            for (const item of list) {
                const key = item[0];
                const value = item[1];
                if (value !== null) {
                    let layerName = '';
                    let parameter = '';
                    if (key.endsWith('_packed_params.dtype')) {
                        parameter = '_packed_params.dtype';
                        layerName = key.substring(0, key.length - parameter.length - 1);
                    }
                    else if (key.endsWith('_packed_params._packed_params') && Array.isArray(value)) {
                        parameter = '_packed_params._packed_params';
                        layerName = key.substring(0, key.length - parameter.length - 1);
                    }
                    else {
                        let split = key.split('.');
                        if (split.length < 2) {
                            split = [ '', split[0] ];
                        }
                        parameter = split.pop();
                        layerName = split.join('.');
                    }
                    if (!layers.has(layerName)) {
                        layers.set(layerName, { name: layerName, states: [], attributes: [] });
                    }
                    const layer = layers.get(layerName);
                    switch (parameter) {
                        case '_packed_params.dtype':
                            layer.attributes.push({ name: parameter, value: value });
                            break;
                        case '_packed_params._packed_params':
                            layer.states.push({ name: parameter, arguments: value.map((item) => { return { id: '', value: item }; }) });
                            break;
                        default:
                            layer.states.push({ name: parameter, arguments: [ { id: key, value: value } ] });
                            if (layer.name == '' && layer.states.length > 4) {
                                return null;
                            }
                            break;
                    }
                }
            }
            return layers.values();
        }
        return null;
    }

    static _convertStateDictMap(obj) {
        if (!obj || Array.isArray(obj)) {
            return null;
        }
        const state_dict = [];
        const state_map = {};
        for (const key in obj) {
            const split = key.split('.');
            if (split.length < 1) {
                return null;
            }
            const state = {};
            state.id = key;
            state.name = split.pop();
            state.value = obj[key];
            if (state.value && state.value.__module__ === 'megengine.parameter' && state.value.__name__ === 'Parameter') {
                if (mge.Utility.isTensor(state.value.data)) {
                    state.value = state.value.data;
                }
            }
            if (!mge.Utility.isTensor(state.value)) {
                return null;
            }
            const state_group_name = split.join('.');
            let state_group = state_map[state_group_name];
            if (!state_group) {
                state_group = {};
                state_group.name = state_group_name;
                state_group.states = [];
                state_map[state_group_name] = state_group;
                state_dict.push(state_group);
            }
            state_group.states.push({ name: state.name, arguments: [ state ] });
        }
        return state_dict;
    }

    static _convertStateDictGroupMap(obj) {
        if (!obj || Array.isArray(obj)) {
            return null;
        }
        const state_dict = [];
        const state_map = {};
        for (const state_group_name in obj) {
            let state_group = state_map[state_group_name];
            if (!state_group) {
                state_group = {};
                state_group.name = state_group_name;
                state_group.states = [];
                state_group.attributes = [];
                state_map[state_group_name] = state_group;
                state_dict.push(state_group);
            }
            const item = obj[state_group_name];
            if (!item) {
                return null;
            }
            if (item instanceof Map) {
                for (const pair of item) {
                    const key = pair[0];
                    const value = pair[1];
                    if (!key) {
                        return null;
                    }
                    if (value && !mge.Utility.isTensor(value)) {
                        return null;
                    }
                    const argument = { id: state_group_name + '.' + key, value: value };
                    state_group.states.push({ name: key, arguments: [ argument ] });
                }
            }
            else if (item instanceof Uint8Array) {
                return null;
            }
            else if (Object(item) === item) {
                let hasTensors = false;
                for (const key in item) {
                    const value = item[key];
                    if (mge.Utility.isTensor(value)) {
                        const argument = { id: state_group_name + '.' + key, value: value };
                        state_group.states.push({ name: key, arguments: [ argument ] });
                        hasTensors = true;
                    }
                    else if (value !== Object(value)) {
                        state_group.attributes.push({ name: key, value: value });
                    }
                    else if (value && value.data && value.__module__ === 'megengine.parameter' && value.__name__ === 'Parameter') {
                        const argument = { id: state_group_name + '.' + key, value: value.data };
                        state_group.states.push({ name: key, arguments: [ argument ] });
                        hasTensors = true;
                    }
                    else {
                        return null;
                    }
                }
                if (!hasTensors) {
                    return null;
                }
            }
            else {
                return null;
            }
        }
        return state_dict;
    }

    static readInt32(buffer) {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        return view.getInt32(0, true);
    }

    static readInt64(buffer) {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        return view.getInt64(0, true).toNumber();
    }
};

mge.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading mge model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mge.ModelFactory;
}