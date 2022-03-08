
// Experimental

var sklearn = sklearn || {};

sklearn.ModelFactory = class {

    match(context) {
        const obj = context.open('pkl');
        const validate = (obj, name) => {
            if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                const key = obj.__class__.__module__ + '.' + obj.__class__.__name__;
                return key.startsWith(name);
            }
            return false;
        };
        const formats = [
            { name: 'sklearn.', format: 'sklearn' },
            { name: 'xgboost.sklearn.', format: 'sklearn' },
            { name: 'lightgbm.sklearn.', format: 'sklearn' },
            { name: 'scipy.', format: 'scipy' }
        ];
        for (const format of formats) {
            if (validate(obj, format.name)) {
                return format.format;
            }
            if (Array.isArray(obj) && obj.every((item) => validate(item, format.name))) {
                return format.format + '.list';
            }
            if ((Object(obj) === obj) && Object.entries(obj).every((entry) => validate(entry[1], format.name))) {
                return format.format + '.map';
            }
        }
        return undefined;
    }

    open(context, match) {
        return sklearn.Metadata.open(context).then((metadata) => {
            const obj = context.open('pkl');
            return new sklearn.Model(metadata, match, obj);
        });
    }
};

sklearn.Model = class {

    constructor(metadata, match, obj) {
        const formats = new Map([ [ 'sklearn', 'scikit-learn' ], [ 'scipy', 'SciPy' ] ]);
        this._format = formats.get(match.split('.').shift());
        this._graphs = [];
        const version = [];
        switch (match) {
            case 'sklearn':
            case 'scipy': {
                version.push(obj._sklearn_version ? ' v' + obj._sklearn_version.toString() : '');
                this._graphs.push(new sklearn.Graph(metadata, '', obj));
                break;
            }
            case 'sklearn.list':
            case 'scipy.list': {
                const list = obj;
                for (let i = 0; i < list.length; i++) {
                    const obj = list[i];
                    this._graphs.push(new sklearn.Graph(metadata, i.toString(), obj));
                    version.push(obj._sklearn_version ? ' v' + obj._sklearn_version.toString() : '');
                }
                break;
            }
            case 'sklearn.map':
            case 'scipy.map': {
                for (const entry of Object.entries(obj)) {
                    const obj = entry[1];
                    this._graphs.push(new sklearn.Graph(metadata, entry[0], obj));
                    version.push(obj._sklearn_version ? ' v' + obj._sklearn_version.toString() : '');
                }
                break;
            }
            default: {
                throw new sklearn.Error("Unsupported scikit-learn format '" + match + "'.");
            }
        }
        if (version.every((value) => value === version[0])) {
            this._format += version[0];
        }
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

sklearn.Graph = class {

    constructor(metadata, name, obj) {
        this._name = name || '';
        this._metadata = metadata;
        this._nodes = [];
        this._groups = false;
        this._process('', '', obj, ['data']);
    }

    _process(group, name, obj, inputs) {
        const type = obj.__class__.__module__ + '.' + obj.__class__.__name__;
        switch (type) {
            case 'sklearn.pipeline.Pipeline': {
                this._groups = true;
                name = name || 'pipeline';
                const childGroup = this._concat(group, name);
                for (const step of obj.steps) {
                    inputs = this._process(childGroup, step[0], step[1], inputs);
                }
                return inputs;
            }
            case 'sklearn.pipeline.FeatureUnion': {
                this._groups = true;
                const outputs = [];
                name = name || 'union';
                const output = this._concat(group, name);
                const subgroup = this._concat(group, name);
                this._nodes.push(new sklearn.Node(this._metadata, subgroup, output, obj, inputs, [ output ]));
                for (const transformer of obj.transformer_list){
                    outputs.push(...this._process(subgroup, transformer[0], transformer[1], [ output ]));
                }
                return outputs;
            }
            case 'sklearn.compose._column_transformer.ColumnTransformer': {
                this._groups = true;
                name = name || 'transformer';
                const output = this._concat(group, name);
                const subgroup = this._concat(group, name);
                const outputs = [];
                this._nodes.push(new sklearn.Node(this._metadata, subgroup, output, obj, inputs, [ output ]));
                for (const transformer of obj.transformers){
                    if (transformer[1] !== 'passthrough') {
                        outputs.push(...this._process(subgroup, transformer[0], transformer[1], [ output ]));
                    }
                }
                return outputs;
            }
            default: {
                const output = this._concat(group, name);
                this._nodes.push(new sklearn.Node(this._metadata, group, output, obj, inputs, output === '' ? [] : [ output ]));
                return [ output ];
            }
        }
    }

    _concat(parent, name){
        return (parent === '' ?  name : `${parent}/${name}`);
    }

    get name() {
        return this._name;
    }

    get groups() {
        return this._groups;
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

sklearn.Parameter = class {
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

sklearn.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new sklearn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

sklearn.Node = class {

    constructor(metadata, group, name, obj, inputs, outputs) {
        this._group = group || '';
        this._name = name || '';
        const type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : 'Object';
        this._type = metadata.type(type) || { name: type };
        this._inputs = inputs.map((input) => new sklearn.Parameter(input, [ new sklearn.Argument(input, null, null) ]));
        this._outputs = outputs.map((output) => new sklearn.Parameter(output, [ new sklearn.Argument(output, null, null) ]));
        this._attributes = [];

        for (const entry of Object.entries(obj)) {
            const name = entry[0];
            const value = entry[1];
            if (value && sklearn.Utility.isTensor(value)) {
                const argument = new sklearn.Argument('', null, new sklearn.Tensor(value));
                const paramter = new sklearn.Parameter(name, [ argument ]);
                this._inputs.push(paramter);
            }
            else if (Array.isArray(value) && value.every((obj) => sklearn.Utility.isTensor(obj))) {
                const args = value.map((obj) => new sklearn.Argument('', null, new sklearn.Tensor(obj)));
                const paramter = new sklearn.Parameter(name, args);
                this._inputs.push(paramter);
            }
            else if (!name.startsWith('_')) {
                this._attributes.push(new sklearn.Attribute(metadata.attribute(this._type, name), name, value));
            }
        }
    }

    get type() {
        return this._type; // .split('.').pop();
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group ? this._group : null;
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

sklearn.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        if (metadata) {
            if (metadata.option === 'optional' && this._value == null) {
                this._visible = false;
            }
            else if (metadata.visible === false) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (sklearn.Attribute._isEquivalent(metadata.default, this._value)) {
                    this._visible = false;
                }
            }
        }
        if (value) {
            if (Array.isArray(value) && value.length > 0 && value.every((obj) => obj.__class__ && obj.__class__.__module__ === value[0].__class__.__module__ && obj.__class__.__name__ === value[0].__class__.__name__)) {
                this._type = value[0].__class__.__module__ + '.' + value[0].__class__.__name__ + '[]';
            }
            else if (value.__class__) {
                this._type = value.__class__.__module__ + '.' + value.__class__.__name__;
            }
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get type() {
        return this._type;
    }

    get visible() {
        return this._visible == false ? false : true;
    }

    static _isEquivalent(a, b) {
        if (a === b) {
            return a !== 0 || 1 / a === 1 / b;
        }
        if (a == null || b == null) {
            return false;
        }
        if (a !== a) {
            return b !== b;
        }
        const type = typeof a;
        if (type !== 'function' && type !== 'object' && typeof b != 'object') {
            return false;
        }
        const className = toString.call(a);
        if (className !== toString.call(b)) {
            return false;
        }
        switch (className) {
            case '[object RegExp]':
            case '[object String]':
                return '' + a === '' + b;
            case '[object Number]': {
                if (+a !== +a) {
                    return +b !== +b;
                }
                return +a === 0 ? 1 / +a === 1 / b : +a === +b;
            }
            case '[object Date]':
            case '[object Boolean]': {
                return +a === +b;
            }
            case '[object Array]': {
                let length = a.length;
                if (length !== b.length) {
                    return false;
                }
                while (length--) {
                    if (!sklearn.Attribute._isEquivalent(a[length], b[length])) {
                        return false;
                    }
                }
                return true;
            }
        }

        const keys = Object.keys(a);
        let size = keys.length;
        if (Object.keys(b).length != size) {
            return false;
        }
        while (size--) {
            const key = keys[size];
            if (!(Object.prototype.hasOwnProperty.call(b, key) && sklearn.Attribute._isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
};

sklearn.Tensor = class {

    constructor(value) {
        if (!sklearn.Utility.isTensor(value)) {
            const type = value.__class__.__module__ + '.' + value.__class__.__name__;
            throw new sklearn.Error("Unknown tensor type '" + type + "'.");
        }
        this._kind = 'NumPy Array';
        this._type = new sklearn.TensorType(value.dtype.name, new sklearn.TensorShape(value.shape));
        this._data = value.data;
        if (value.dtype.name === 'string') {
            this._itemsize = value.dtype.itemsize;
        }
    }

    get type() {
        return this._type;
    }

    get kind() {
        return this._kind;
    }

    get state() {
        return this._context().state || null;
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
        switch (this._type.dataType) {
            case 'int64':
            case 'uint64':
                return sklearn.Tensor._stringify(value, '', '    ');
        }
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (!this._type) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        if (!this._data) {
            context.state = 'Tensor is data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;

        switch (context.dataType) {
            case 'float32':
            case 'float64':
            case 'int32':
            case 'uint32':
            case 'int64':
            case 'uint64':
                context.view = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            case 'string':
                context.data = this._data;
                context.itemsize = this._itemsize;
                context.decoder = new TextDecoder('utf-8');
                break;
            default:
                context.state = "Tensor data type '" + context.dataType + "' is not implemented.";
                return context;
        }

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
                    case 'float32': {
                        results.push(context.view.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    }
                    case 'float64': {
                        results.push(context.view.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    }
                    case 'int32': {
                        results.push(context.view.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    }
                    case 'uint32': {
                        results.push(context.view.getUint32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    }
                    case 'int64': {
                        results.push(context.view.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    }
                    case 'uint64': {
                        results.push(context.view.getUint64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    }
                    case 'string': {
                        const buffer = context.data.subarray(context.index, context.index + context.itemsize);
                        const index = buffer.indexOf(0);
                        const content = context.decoder.decode(index >= 0 ? buffer.subarray(0, index) : buffer);
                        results.push(content);
                        context.index += context.itemsize;
                        context.count++;
                        break;
                    }
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

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push('[');
            const items = value.map((item) => sklearn.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(']');
            return result.join('\n');
        }
        return indentation + value.toString();
    }
};

sklearn.TensorType = class {

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

sklearn.TensorShape = class {

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

sklearn.Metadata = class {

    static open(context) {
        if (sklearn.Metadata._metadata) {
            return Promise.resolve(sklearn.Metadata._metadata);
        }
        return context.request('sklearn-metadata.json', 'utf-8', null).then((data) => {
            sklearn.Metadata._metadata = new sklearn.Metadata(data);
            return sklearn.Metadata._metadata;
        }).catch(() => {
            sklearn.Metadata._metadata = new sklearn.Metadata(null);
            return sklearn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._types = new Map();
        this._attributes = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._types = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._types.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributes.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributes.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }
};

sklearn.Utility = class {

    static isTensor(obj) {
        return obj && obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray';
    }

    static findWeights(obj) {
        const keys = [ '', 'blobs' ];
        for (const key of keys) {
            const dict = key === '' ? obj : obj[key];
            if (dict) {
                const weights = new Map();
                if (dict instanceof Map) {
                    for (const pair of dict) {
                        if (!sklearn.Utility.isTensor(pair[1])) {
                            return null;
                        }
                        weights.set(pair[0], pair[1]);
                    }
                    return weights;
                }
                else if (!Array.isArray(dict)) {
                    for (const key in dict) {
                        const value = dict[key];
                        if (key != 'weight_order' && key != 'lr') {
                            if (!key || !sklearn.Utility.isTensor(value)) {
                                return null;
                            }
                            weights.set(key, value);
                        }
                    }
                    return weights;
                }
            }
        }
        for (const key of keys) {
            const list = key === '' ? obj : obj[key];
            if (list && Array.isArray(list)) {
                const weights = new Map();
                for (let i = 0; i < list.length; i++) {
                    const value = list[i];
                    if (!sklearn.Utility.isTensor(value, 'numpy.ndarray')) {
                        return null;
                    }
                    weights.set(i.toString(), value);
                }
                return weights;
            }
        }
        return null;
    }
};

sklearn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading scikit-learn model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = sklearn.ModelFactory;
}