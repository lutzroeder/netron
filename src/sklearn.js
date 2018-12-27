/*jshint esversion: 6 */

// Experimental

var sklearn = sklearn || {};
var marked = marked || require('marked');
var base = base || require('./base');

sklearn.ModelFactory = class {

    match(context, host) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'pkl' || extension == 'joblib') {
            var buffer = context.buffer;
            var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                return false;
            }
            return true;
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('./pickle', (err, pickle) => {
            if (err) {
                callback(err, null);
                return;
            }
            
            var obj = null;
            try {
                var identifier = context.identifier;
                var unpickler = new pickle.Unpickler(context.buffer);

                var constructorTable = {};
                var functionTable = {};

                constructorTable['numpy.dtype'] = function(obj, align, copy) { 
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
                            else {
                                throw new sklearn.Error("Unknown dtype '" + obj.toString() + "'.");
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
                                throw new sklearn.Error("Unknown numpy.dtype setstate length '" + state.length.toString() + "'.");
                        }
                    };
                };
                constructorTable['numpy.core.multiarray._reconstruct'] = function(subtype, shape, dtype) {
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
                        var array = {};
                        array.__type__ = this.subtype;
                        array.dtype = this.typecode;
                        array.shape = this.shape;
                        var size = array.dtype.itemsize;
                        for (var i = 0; i < array.shape.length; i++) {
                            size = size * array.shape[i];                                
                        }
                        if (typeof this.rawdata == 'string') {
                            array.data = unpickler.unescape(this.rawdata, size);
                            if (array.data.length != size) {
                                throw new sklearn.Error('Invalid string array data size.');
                            }
                        }
                        else {
                            array.data = this.rawdata;
                            if (array.data.length != size) {
                                throw new sklearn.Error('Invalid array data size.');
                            }
                        }
                        return array;
                    };
                };
                constructorTable['joblib.numpy_pickle.NumpyArrayWrapper'] = function(subtype, shape, dtype) {
                    this.__setstate__ = function(state) {
                        this.subclass = state.subclass;
                        this.dtype = state.dtype;
                        this.shape = state.shape;
                        this.order = state.order;
                        this.allow_mmap = state.allow_mmap;
                    };
                    this.__read__ = function(unpickler) {
                        var size = 1;
                        this.shape.forEach((dimension) => {
                            size *= dimension;
                        });
                        if (this.dtype.name == 'object') {
                            debugger;
                            return unpickler.load(function_call, null);
                        }
                        else {
                            this.data = unpickler.read(size * this.dtype.itemsize);
                        }

                        var array = {};
                        array.__type__ = this.subclass;
                        array.dtype = this.dtype;
                        array.shape = this.shape;
                        array.data = this.data;
                        return array;
                    };
                };

                constructorTable['lightgbm.sklearn.LGBMRegressor'] = function() {};
                constructorTable['lightgbm.sklearn.LGBMClassifier'] = function() {};
                constructorTable['lightgbm.basic.Booster'] = function() {};
                constructorTable['sklearn.calibration.CalibratedClassifierCV​'] = function() {};
                constructorTable['sklearn.compose._column_transformer.ColumnTransformer'] = function() {};
                constructorTable['sklearn.decomposition.PCA'] = function() {};
                constructorTable['sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper'] = constructorTable['joblib.numpy_pickle.NumpyArrayWrapper'];
                constructorTable['sklearn.ensemble.forest.RandomForestClassifier'] = function() {};
                constructorTable['sklearn.ensemble.forest.RandomForestRegressor'] = function() {};
                constructorTable['sklearn.ensemble.forest.ExtraTreesClassifier'] = function() {};
                constructorTable['sklearn.ensemble.weight_boosting.AdaBoostClassifier'] = function() {};
                constructorTable['sklearn.feature_extraction.text.CountVectorizer​'] = function() {};
                constructorTable['sklearn.feature_extraction.text.TfidfVectorizer​'] = function() {};
                constructorTable['sklearn.impute.SimpleImputer'] = function() {};
                constructorTable['sklearn.linear_model.base.LinearRegression'] = function() {};
                constructorTable['sklearn.linear_model.LogisticRegression'] = function() {};
                constructorTable['sklearn.linear_model.logistic.LogisticRegression'] = function() {};
                constructorTable['sklearn.linear_model.LassoLars​'] = function() {};
                constructorTable['sklearn.model_selection._search.GridSearchCV'] = function() {};
                constructorTable['sklearn.naive_bayes.BernoulliNB'] = function() {};
                constructorTable['sklearn.naive_bayes.ComplementNB'] = function() {};
                constructorTable['sklearn.naive_bayes.GaussianNB'] = function() {};
                constructorTable['sklearn.naive_bayes.MultinomialNB'] = function() {};
                constructorTable['sklearn.neighbors.KNeighborsClassifier​'] = function() {};
                constructorTable['sklearn.neighbors.KNeighborsRegressor'] = function() {};
                constructorTable['sklearn.neural_network.rbm.BernoulliRBM'] = function() {};
                constructorTable['sklearn.neural_network.multilayer_perceptron.MLPRegressor'] = function() {};
                constructorTable['sklearn.neural_network._stochastic_optimizers.AdamOptimizer'] = function() {};
                constructorTable['sklearn.neural_network._stochastic_optimizers.SGDOptimizer'] = function() {};
                constructorTable['sklearn.pipeline.Pipeline'] = function() {};
                constructorTable['sklearn.preprocessing._encoders.OneHotEncoder'] = function() {};
                constructorTable['sklearn.preprocessing.data.Binarizer'] = function() {};
                constructorTable['sklearn.preprocessing.data.StandardScaler'] = function() {};
                constructorTable['sklearn.preprocessing.label.LabelEncoder'] = function() {};
                constructorTable['sklearn.svm.classes.SVC'] = function() {};
                constructorTable['sklearn.svm.classes.SVR'] = function() {};
                constructorTable['sklearn.tree._tree.Tree'] = function(n_features, n_classes, n_outputs) {
                    this.n_features = n_features;
                    this.n_classes = n_classes;
                    this.n_outputs = n_outputs;
                    this.__setstate__ = function(state) {
                        this.max_depth = state.max_depth;
                        this.node_count = state.node_count;
                        this.nodes = state.nodes;
                        this.values = state.values;
                    };
                };
                constructorTable['sklearn.tree.tree.DecisionTreeClassifier'] = function() {};
                constructorTable['sklearn.tree.tree.DecisionTreeRegressor'] = function() {};
                constructorTable['sklearn.tree.tree.ExtraTreeClassifier'] = function() {};
                constructorTable['sklearn.utils.deprecation.DeprecationDict'] = function() {};
                constructorTable['xgboost.core.Booster'] = function() {};
                constructorTable['xgboost.sklearn.XGBClassifier'] = function() {};

                functionTable['copy_reg._reconstructor'] = function(cls, base, state) {
                    if (base == '__builtin__.object') {
                        var obj = {};
                        obj.__type__ = cls;
                        return obj;
                    }
                    if (base == '__builtin__.tuple') {
                        return state;
                    }
                    throw new sklearn.Error("Unknown base type '" + base + "'.");
                };
                functionTable['numpy.core.multiarray.scalar'] = function(dtype, rawData) {
                    var data = rawData;
                    if (rawData.constructor !== Uint8Array) {
                        data = new Uint8Array(rawData.length);
                        for (var i = 0; i < rawData.length; i++) {
                            data[i] = rawData.charCodeAt(i);
                        }
                    }
                    var dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    switch (dtype.name) {
                        case 'float64':
                            return dataView.getFloat64(0, true);
                        case 'int64':
                            return new base.Int64(data.subarray(0, dtype.itemsize));
                    }
                    throw new sklearn.Error("Unknown scalar type '" + dtype.name + "'.");
                };
                functionTable['numpy.ma.core._mareconstruct'] = function(subtype, baseclass, baseshape, basetype) {
                    // _data = ndarray.__new__(baseclass, baseshape, basetype)
                    // _mask = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
                    // return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)
                    var obj = {};
                    obj.__type__ = subtype;
                    return obj;
                };
                functionTable['_codecs.encode'] = function(obj, econding) {
                    return obj;
                };
                functionTable['collections.defaultdict'] = function(default_factory) {
                    return {};
                };
                functionTable['__builtin__.bytearray'] = function(data, encoding) {
                    return { data: data, encoding: encoding };
                };
                functionTable['numpy.random.__RandomState_ctor'] = function() {
                    return {};
                };

                var function_call = (name, args) => {
                    var func = functionTable[name];
                    if (func) {
                        return func.apply(null, args);
                    }
                    var obj = { __type__: name };
                    var constructor = constructorTable[name];
                    if (constructor) {
                        constructor.apply(obj, args);
                    }
                    else {
                        debugger;
                        host.exception(new sklearn.Error("Unknown function '" + name + "' in '" + identifier + "'."), false);
                    }
                    return obj;
                };

                obj = unpickler.load(function_call, null);
                if (obj && Array.isArray(obj)) {
                    throw new sklearn.Error('Array is not a valid root object.');
                }
                if (!obj || !obj.__type__) {
                    throw new sklearn.Error('Root object has no type.');
                }
            }
            catch (error) {
                host.exception(error, false);
                callback(error);
                return;
            }

            sklearn.Metadata.open(host, (err, metadata) => {
                try {
                    var model = new sklearn.Model(metadata, obj);
                    callback(null, model);
                    return;
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new sklearn.Error(error.message), null);
                    return;
                }
            });
        });
    }
};

sklearn.Model = class {

    constructor(metadata, obj) {
        this._format = 'scikit-learn';
        if (obj._sklearn_version) {
            this._format += ' ' + obj._sklearn_version.toString();
        }

        this._graphs = [];
        this._graphs.push(new sklearn.Graph(metadata, obj));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

sklearn.Graph = class {

    constructor(metadata, obj) {
        this._nodes = [];
        this._groups = false;

        var input = 'data';
        switch (obj.__type__) {
            case 'sklearn.pipeline.Pipeline':
                this._groups = true;
                for (var step of obj.steps) {
                    this._nodes.push(new sklearn.Node(metadata, 'pipeline', step[0], step[1], [ input ], [ step[0] ]));
                    input = step[0];
                }
                break;
            default:
                this._nodes.push(new sklearn.Node(metadata, null, null, obj, [], []));
                break;
        }

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

sklearn.Argument = class {
    constructor(name, connections) {
        this._name = name;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get connections() {
        return this._connections;
    }
};

sklearn.Connection = class {
    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
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

sklearn.Node = class {

    constructor(metadata, group, name, obj, inputs, outputs) {
        this._metadata = metadata;
        if (group) {
            this._group = group;
        }
        var operator = obj.__type__.split('.');
        this._type = operator.pop();
        this._package = operator.join('.');
        this._name = name;
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        this._initializers = [];

        Object.keys(obj).forEach((key) => {
            if (!key.startsWith('_')) {
                var value = obj[key];

                if (Array.isArray(value) || Number.isInteger(value) || value == null) {
                    this._attributes.push(new sklearn.Attribute(this._metadata, this, key, value));
                }
                else {
                    switch (value.__type__) {
                        case 'numpy.ndarray':
                            this._initializers.push(new sklearn.Tensor(key, value));
                            break;
                        default: 
                            this._attributes.push(new sklearn.Attribute(this._metadata, this, key, value));
                    }
                }
            }
        });
    }

    get operator() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group ? this._group : null;
    }

    get documentation() {
        var schema = this._metadata.getSchema(this.operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this.operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.references) {
                schema.references.forEach((reference) => {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                });
            }
            return schema;
        }
        return '';
    }

    get category() {
        var schema = this._metadata.getSchema(this.operator);
        return (schema && schema.category) ? schema.category : null;
    }

    get inputs() {
        var inputs = this._inputs.map((input) => {
            return new sklearn.Argument(input, [ new sklearn.Connection(input, null, null) ]);
        });
        this._initializers.forEach((initializer) => {
            inputs.push(new sklearn.Argument(initializer.name, [ new sklearn.Connection(null, null, initializer) ]));
        });
        return inputs;
    }

    get outputs() {
        return this._outputs.map((output) => {
            return new sklearn.Argument(output, [ new sklearn.Connection(output, null, null) ]);
        });
    }

    get attributes() {
        return this._attributes;
    }
};

sklearn.Attribute = class {

    constructor(metadata, node, name, value) {
        this._name = name;
        this._value = value;

        var schema = metadata.getAttributeSchema(node.operator, this._name);
        if (schema) {
            if (schema.hasOwnProperty('option') && schema.option == 'optional' && this._value == null) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('visible') && !schema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default')) {
                if (sklearn.Attribute._isEquivalent(schema.default, this._value)) {
                    this._visible = false;
                }
            }
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
        var type = typeof a;
        if (type !== 'function' && type !== 'object' && typeof b != 'object') {
            return false;
        }
        var className = toString.call(a);
        if (className !== toString.call(b)) {
            return false;
        }
        switch (className) {
            case '[object RegExp]':
            case '[object String]':
                return '' + a === '' + b;
            case '[object Number]':
                if (+a !== +a) {
                    return +b !== +b;
                }
                return +a === 0 ? 1 / +a === 1 / b : +a === +b;
            case '[object Date]':
            case '[object Boolean]':
                return +a === +b;
            case '[object Array]':
                var length = a.length;
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

        var keys = Object.keys(a);
        var size = keys.length;
        if (Object.keys(b).length != size) {
            return false;
        } 
        while (size--) {
            var key = keys[size];
            if (!(b.hasOwnProperty(key) && sklearn.Attribute._isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
};

sklearn.Tensor = class {

    constructor(name, value) {
        this._name = name;
        switch (value.__type__) {
            case 'numpy.ndarray':
                this._kind = 'Array';
                this._type = new sklearn.TensorType(value.dtype.name, new sklearn.TensorShape(value.shape));
                this._data = value.data;
                break;
            default:
                throw new sklearn.Error("Unknown tensor type '" + value.__type__ + "'.");
        }
    }

    get name() {
        return this._name;
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
        switch (this._type.dataType) {
            case 'int64':
            case 'uint64':
                return sklearn.Tensor._stringify(value, '', '    ');
        }
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
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
                context.rawData = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            case 'int64':
            case 'uint64':
                context.rawData = this._data;
                break;
            default:
                context.state = "Tensor data type '" + context.dataType + "' is not implemented.";
                return context;
        }

        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType)
                {
                    case 'float32':
                        results.push(context.rawData.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.rawData.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.rawData.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'uint32':
                        results.push(context.rawData.getUint32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(new base.Int64(context.rawData.subarray(context.index, context.index + 8)));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'uint64':
                        results.push(new base.Uint64(context.rawData.subarray(context.index, context.index + 8)));
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
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push('[');
            var items = value.map((item) => sklearn.Tensor._stringify(item, indentation + indent, indent));
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

    static open(host, callback) {
        if (sklearn.Metadata._metadata) {
            callback(null, sklearn.Metadata._metadata);
        }
        else {
            host.request(null, 'sklearn-metadata.json', 'utf-8', (err, data) => {
                sklearn.Metadata._metadata = new sklearn.Metadata(data);
                callback(null, sklearn.Metadata._metadata);
            });    
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var schema = this.getSchema(operator);
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributeMap) {
                schema.attributeMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributeMap[attribute.name] = attribute;
                });
            }
            return schema.attributeMap[name] || null;
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