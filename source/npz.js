/* jshint esversion: 6 */

// Experimental

var npz = npz || {};
var pickle = pickle || require('./pickle');

npz.ModelFactory = class {

    match(context) {
        const entries = context.entries('zip');
        return entries.length > 0 && entries.every((entry) => entry.name.endsWith('.npy'));
    }

    open(context) {
        return context.require('./numpy').then((numpy) => {
            const modules = [];
            const modulesMap = new Map();
            const functionTable = new Map();
            const constructorTable = new Map();
            functionTable.set('_codecs.encode', function(obj /*, econding */) {
                return obj;
            });
            constructorTable.set('numpy.core.multiarray._reconstruct', function(subtype, shape, dtype) {
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
                    array.__type__ = this.subtype;
                    array.dtype = this.typecode;
                    array.shape = this.shape;
                    let size = array.dtype.itemsize;
                    for (let i = 0; i < array.shape.length; i++) {
                        size = size * array.shape[i];
                    }
                    if (typeof this.rawdata == 'string') {
                        array.data = unpickler.unescape(this.rawdata, size);
                        if (array.data.length != size) {
                            throw new npz.Error('Invalid string array data size.');
                        }
                    }
                    else {
                        array.data = this.rawdata;
                        if (array.data.length != size) {
                            // TODO
                            // throw new npz.Error('Invalid array data size.');
                        }
                    }
                    return array;
                };
            });
            constructorTable.set('numpy.dtype', function(obj, align, copy) {
                switch (obj) {
                    case 'i1': this.name = 'int8'; this.itemsize = 1; break;
                    case 'i2': this.name = 'int16'; this.itemsize = 2; break;
                    case 'i4': this.name = 'int32'; this.itemsize = 4; break;
                    case 'i8': this.name = 'int64'; this.itemsize = 8; break;
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
                            throw new npz.Error("Unknown dtype '" + obj.toString() + "'.");
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
                            throw new npz.Error("Unknown numpy.dtype setstate length '" + state.length.toString() + "'.");
                    }
                };
            });
            const function_call = (name, args) => {
                if (functionTable.has(name)) {
                    const func = functionTable.get(name);
                    return func.apply(null, args);
                }
                const obj = { __type__: name };
                if (constructorTable.has(name)) {
                    const constructor = constructorTable.get(name);
                    constructor.apply(obj, args);
                }
                else {
                    throw new npz.Error("Unknown function '" + name + "'.");
                }
                return obj;
            };

            const dataTypeMap = new Map([
                [ 'i1', 'int8'], [ 'i2', 'int16' ], [ 'i4', 'int32'], [ 'i8', 'int64' ],
                [ 'u1', 'uint8'], [ 'u2', 'uint16' ], [ 'u4', 'uint32'], [ 'u8', 'uint64' ],
                [ 'f2', 'float16'], [ 'f4', 'float32' ], [ 'f8', 'float64']
            ]);

            for (const entry of context.entries('zip')) {
                if (!entry.name.endsWith('.npy')) {
                    throw new npz.Error("Invalid file name '" + entry.name + "'.");
                }
                const name = entry.name.replace(/\.npy$/, '');
                const parts = name.split('/');
                const parameterName = parts.pop();
                const moduleName = parts.join('/');
                if (!modulesMap.has(moduleName)) {
                    const newModule = { name: moduleName, parameters: [] };
                    modules.push(newModule);
                    modulesMap.set(moduleName, newModule);
                }
                const module = modulesMap.get(moduleName);
                const data = entry.data;
                let array = new numpy.Array(data);
                if (array.byteOrder === '|') {
                    if (array.dataType !== 'O') {
                        throw new npz.Error("Invalid data type '" + array.dataType + "'.");
                    }
                    const unpickler = new pickle.Unpickler(array.data);
                    const root = unpickler.load(function_call);
                    array = { dataType: root.dtype.name, shape: null, data: null, byteOrder: '|' };
                }

                module.parameters.push({
                    name: parameterName,
                    dataType: dataTypeMap.has(array.dataType) ? dataTypeMap.get(array.dataType) : array.dataType,
                    shape: array.shape,
                    data: array.data,
                    byteOrder: array.byteOrder
                });
            }
            return new npz.Model(modules, 'NumPy Zip');
        });
    }
};

npz.Model = class {

    constructor(modules, format) {
        this._format = format;
        this._graphs = [];
        this._graphs.push(new npz.Graph(modules));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

npz.Graph = class {

    constructor(modules) {
        this._nodes = [];
        for (const module of modules) {
            this._nodes.push(new npz.Node(module));
        }
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [];
    }

    get nodes() {
        return this._nodes;
    }
};

npz.Parameter = class {

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

npz.Argument = class {

    constructor(name, initializer) {
        if (typeof name !== 'string') {
            throw new npz.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._initializer.type;
    }

    get initializer() {
        return this._initializer;
    }
};

npz.Node = class {

    constructor(module) {
        this._name = module.name;
        this._inputs = [];
        for (const parameter of module.parameters) {
            const name = this._name ? [ this._name, parameter.name ].join('/') : parameter.name;
            const initializer = new npz.Tensor(name, parameter.dataType, parameter.shape, parameter.data, parameter.byteOrder);
            this._inputs.push(new npz.Parameter(parameter.name, [
                new npz.Argument(name, initializer)
            ]));
        }
    }

    get type() {
        return 'Module';
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return null;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return [];
    }

    get attributes() {
        return [];
    }
};

npz.Tensor = class  {

    constructor(name, dataType, shape, data, byteOrder) {
        this._name = name;
        this._type = new npz.TensorType(dataType, new npz.TensorShape(shape));
        this._shape = shape;
        this._data = data;
        this._byteOrder = byteOrder;
    }

    get kind() {
        return '';
    }

    get name() {
        return this._name;
    }

    get type(){
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
        return npz.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;
        if (this._byteOrder !== '<' && this._byteOrder !== '>') {
            context.state = 'Tensor byte order is not supported.';
            return context;
        }
        if (this._reference) {
            context.state = 'Tensor reference not implemented.';
            return context;
        }
        if (!this._data || this._data.length == 0) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        switch (this._type.dataType) {
            case 'float16':
                context.itemSize = 2;
                break;
            case 'float32':
                context.itemSize = 4;
                break;
            case 'float64':
                context.itemSize = 8;
                break;
            case 'int8':
                context.itemSize = 1;
                break;
            case 'int16':
                context.itemSize = 2;
                break;
            case 'int32':
                context.itemSize = 4;
                break;
            case 'int64':
                context.itemSize = 8;
                break;
            case 'uint8':
                context.itemSize = 1;
                break;
            case 'uint16':
                context.itemSize = 2;
                break;
            case 'uint32':
                context.itemSize = 4;
                break;
            default:
                context.state = 'Tensor data type is not supported.';
                return context;
        }
        context.dimensions = this._type.shape.dimensions;
        context.dataType = this._type.dataType;
        context.littleEndian = this._byteOrder == '<';
        context.data = this._data;
        context.rawData = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const littleEndian = context.littleEndian;
        const shape = context.dimensions.length == 0 ? [ 1 ] : context.dimensions;
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.rawData) {
                    switch (context.dataType) {
                        case 'float16':
                            results.push(context.rawData.getFloat16(context.index, littleEndian));
                            break;
                        case 'float32':
                            results.push(context.rawData.getFloat32(context.index, littleEndian));
                            break;
                        case 'float64':
                            results.push(context.rawData.getFloat64(context.index, littleEndian));
                            break;
                        case 'int8':
                            results.push(context.rawData.getInt8(context.index, littleEndian));
                            break;
                        case 'int16':
                            results.push(context.rawData.getInt16(context.index, littleEndian));
                            break;
                        case 'int32':
                            results.push(context.rawData.getInt32(context.index, littleEndian));
                            break;
                        case 'int64':
                            results.push(context.rawData.getInt64(context.index, littleEndian));
                            break;
                        case 'uint8':
                            results.push(context.rawData.getUint8(context.index, littleEndian));
                            break;
                        case 'uint16':
                            results.push(context.rawData.getUint16(context.index, littleEndian));
                            break;
                        case 'uint32':
                            results.push(context.rawData.getUint32(context.index, littleEndian));
                            break;
                    }
                    context.index += context.itemSize;
                    context.count++;
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
            const items = value.map((item) => npz.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
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

npz.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType || '?';
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

npz.TensorShape = class {

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
        return '[' + this._dimensions.join(',') + ']';
    }
};

npz.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Chainer model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = npz.ModelFactory;
}
