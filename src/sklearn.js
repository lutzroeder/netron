/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var sklearn = sklearn || {};
var long = long || { Long: require('long') };

sklearn.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (['pkl', 'joblib', 'model', 'meta', 'pb'].indexOf(extension) !== -1) {
            const buffer = context.buffer;
            if (buffer) {
                // Reject PyTorch models with .pkl file extension.
                const torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
                if (buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                    return false;
                }
                if (buffer.length > 1 && buffer[buffer.length - 1] === 0x2E) {
                    return true;
                }
                if (buffer.length > 2 && buffer[0] === 0x80 && buffer[1] < 5) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./pickle').then((pickle) => {
            const identifier = context.identifier;
            return sklearn.Metadata.open(host).then((metadata) => {
                try {
                    const container = new sklearn.Container(context.buffer, pickle, (error, fatal) => {
                        const message = error && error.message ? error.message : error.toString();
                        host.exception(new sklearn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'."), fatal);
                    });
                    if (!container.weights) {
                        if (!container.data) {
                            throw new sklearn.Error('No root object.');
                        }
                        if (Array.isArray(container.data)) {
                            throw new sklearn.Error('Array is not a valid root object.');
                        }
                    }
                    return new sklearn.Model(metadata, container.data, container.weights);
                }
                catch (error) {
                    host.exception(error, false);
                    const message = error && error.message ? error.message : error.toString();
                    throw new sklearn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
            });
        });
    }
};

sklearn.Model = class {

    constructor(metadata, obj, weights) {
        if (obj && obj.__module__ && obj.__module__.startsWith('sklearn.')) {
            this._format = 'scikit-learn' + (obj._sklearn_version ? ' v' + obj._sklearn_version.toString() : '');
        }
        else if (obj && obj.__module__ && obj.__module__.startsWith('xgboost.')) {
            this._format = 'XGBoost' + (obj._sklearn_version ? ' v' + obj._sklearn_version.toString() : '');
        }
        else if (obj && obj.__module__ && obj.__module__.startsWith('nolearn.lasagne.')) {
            this._format = 'Lasagne';
        }
        else if (obj && obj.__module__ && obj.__module__.startsWith('gensim.')) {
            this._format = 'gensim';
        }
        else {
            this._format = 'Pickle';
        }
        this._graphs = [];
        this._graphs.push(new sklearn.Graph(metadata, obj, weights));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

sklearn.Graph = class {

    constructor(metadata, obj, weights) {
        this._metadata = metadata;
        this._nodes = [];
        this._groups = false;

        if (obj) {
            let input = 'data';
            switch ([ obj.__module__, obj.__name__].join('.')) {
                case 'sklearn.pipeline.Pipeline':
                    this._groups = true;
                    for (const step of obj.steps) {
                        this._add('pipeline', step[0], step[1], [ input ], [ step[0] ]);
                        input = step[0];
                    }
                    break;
                default:
                    this._add(null, null, obj, [], []);
                    break;
            }
        }
        else if (weights instanceof Map) {
            const group_map = {};
            const groups = [];
            for (const pair of weights) {
                const key = pair[0];
                const parts = key.split('_');
                const value = pair[1];
                const name = parts.length > 1 ? parts.pop() : '?';
                const id = parts.join('_');
                let group = group_map[id];
                if (!group) {
                    group = { id: id, arrays: [] };
                    groups.push(group);
                    group_map[id] = group;
                }
                group.arrays.push({
                    key: key,
                    name: name,
                    value: value
                });
            }
            this._nodes = this._nodes.concat(groups.map((group) => {
                const inputs = group.arrays.map((array) => {
                    return new sklearn.Parameter(array.name, [
                        new sklearn.Argument(array.key, null, new sklearn.Tensor(array.key, array.value))
                    ]);
                });
                return new sklearn.Node(this._metadata, '', group.id, { __module__: 'sklearn._', __name__: 'Weights' }, inputs, []);
            }));
        }
    }
    _add(group, name, obj, inputs, outputs) {
        const initializers = [];
        for (const key of Object.keys(obj)) {
            if (!key.startsWith('_')) {
                const value = obj[key];
                if (sklearn.Utility.isTensor(value)) {
                    initializers.push(new sklearn.Tensor(key, value));
                }
            }
        }
        inputs = inputs.map((input) => {
            return new sklearn.Parameter(input, [ new sklearn.Argument(input, null, null) ]);
        });
        inputs = inputs.concat(initializers.map((initializer) => {
            return new sklearn.Parameter(initializer.name, [ new sklearn.Argument('', null, initializer) ]);
        }));
        outputs = outputs.map((output) => {
            return new sklearn.Parameter(output, [ new sklearn.Argument(output, null, null) ]);
        });
        this._nodes.push(new sklearn.Node(this._metadata, group, name, obj, inputs, outputs));
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
        this._metadata = metadata;
        this._group = group || '';
        this._type = (obj.__module__ && obj.__name__) ? (obj.__module__ + '.' + obj.__name__) : (obj.__name__ ? obj.__name__ : 'Object');
        this._name = name || '';
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        this._initializers = [];
        for (const name of Object.keys(obj)) {
            if (!name.startsWith('_')) {
                const value = obj[name];
                if (value && !Array.isArray(value) && value === Object(value) && sklearn.Utility.isTensor(value)) {
                    this._initializers.push(new sklearn.Tensor(name, value));
                }
                else {
                    const schema = metadata.attribute(this.operator, name);
                    this._attributes.push(new sklearn.Attribute(schema, name, value));
                }
            }
        }
    }

    get operator() {
        return this._type.split('.').pop();
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group ? this._group : null;
    }

    get metadata() {
        return this._metadata.type(this.operator);
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

    constructor(schema, name, value) {
        this._name = name;
        this._value = value;
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'option') && schema.option == 'optional' && this._value == null) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
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
            let key = keys[size];
            if (!(Object.prototype.hasOwnProperty.call(b, key) && sklearn.Attribute._isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
};

sklearn.Tensor = class {

    constructor(name, value) {
        this._name = name;
        if (sklearn.Utility.isTensor(value)) {
            this._kind = 'Array';
            this._type = new sklearn.TensorType(value.dtype.name, new sklearn.TensorShape(value.shape));
            this._data = value.data;
        }
        else {
            const type = [ value.__module__, value.__name__ ].join('.');
            throw new sklearn.Error("Unknown tensor type '" + type + "'.");
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
                context.rawData = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
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
                        results.push(new long.Long(context.rawData.getUint32(context.index, true), context.rawData.getUint32(context.index + 4, true), false));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'uint64':
                        results.push(new long.Long(context.rawData.getUint32(context.index, true), context.rawData.getUint32(context.index + 4, true), true));
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

    static open(host) {
        if (sklearn.Metadata._metadata) {
            return Promise.resolve(sklearn.Metadata._metadata);
        }
        return host.request(null, 'sklearn-metadata.json', 'utf-8').then((data) => {
            sklearn.Metadata._metadata = new sklearn.Metadata(data);
            return sklearn.Metadata._metadata;
        }).catch(() => {
            sklearn.Metadata._metadata = new sklearn.Metadata(null);
            return sklearn.Metadata._metadata;
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

    type(operator) {
        return this._map.get(operator);
    }

    attribute(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(operator + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

sklearn.Container = class {

    constructor(buffer, pickle, exception) {
        const unpickler = new pickle.Unpickler(buffer);

        const constructorTable = {};
        const functionTable = {};

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
                case 'f2': this.name = 'float16'; this.itemsize = 2; break;
                case 'f4': this.name = 'float32'; this.itemsize = 4; break;
                case 'f8': this.name = 'float64'; this.itemsize = 8; break;
                case 'b1': this.name = 'int8'; this.itemsize = 1; break;
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
                const array = {};
                sklearn.Utility.applyType(array, this.subtype);
                array.dtype = this.typecode;
                array.shape = this.shape;
                const dims = array.shape && array.shape.length > 0 ? array.shape.reduce((a, b) => a * b) : 1;
                const size = array.dtype.itemsize * dims;
                if (typeof this.rawdata == 'string') {
                    array.data = unpickler.unescape(this.rawdata, size);
                    if (array.data.length != size) {
                        throw new sklearn.Error('Invalid string array data size.');
                    }
                }
                else {
                    array.data = this.rawdata;
                }
                return array;
            };
        };
        constructorTable['joblib.numpy_pickle.NumpyArrayWrapper'] = function(/* subtype, shape, dtype */) {
            this.__setstate__ = function(state) {
                this.subclass = state.subclass;
                this.dtype = state.dtype;
                this.shape = state.shape;
                this.order = state.order;
                this.allow_mmap = state.allow_mmap;
            };
            this.__read__ = function(unpickler) {
                if (this.dtype.name == 'object') {
                    return unpickler.load(function_call, null);
                }
                else {
                    const size = this.dtype.itemsize * this.shape.reduce((a, b) => a * b);
                    this.data = unpickler.read(size);
                }
                const obj = {
                    dtype: this.dtype,
                    shape: this.shape,
                    data: this.data,
                };
                sklearn.Utility.applyType(obj, this.subclass);
                return obj;
            };
        };

        constructorTable['gensim.models.doc2vec.Doctag'] = function() {};
        constructorTable['gensim.models.doc2vec.Doc2Vec'] = function() {};
        constructorTable['gensim.models.doc2vec.Doc2VecTrainables'] = function() {};
        constructorTable['gensim.models.doc2vec.Doc2VecVocab'] = function() {};
        constructorTable['gensim.models.keyedvectors.Doc2VecKeyedVectors'] = function() {};
        constructorTable['gensim.models.keyedvectors.Vocab'] = function() {};
        constructorTable['gensim.models.keyedvectors.Word2VecKeyedVectors'] = function() {};
        constructorTable['gensim.models.word2vec.Vocab'] = function() {};
        constructorTable['gensim.models.word2vec.Word2Vec'] = function() {};
        constructorTable['lightgbm.sklearn.LGBMRegressor'] = function() {};
        constructorTable['lightgbm.sklearn.LGBMClassifier'] = function() {};
        constructorTable['lightgbm.basic.Booster'] = function() {};
        constructorTable['nolearn.lasagne.base.BatchIterator'] = function() {};
        constructorTable['nolearn.lasagne.base.Layers'] = function() {};
        constructorTable['nolearn.lasagne.base.NeuralNet'] = function() {};
        constructorTable['nolearn.lasagne.base.TrainSplit'] = function() {};
        constructorTable['nolearn.lasagne.handlers.PrintLayerInfo'] = function() {};
        constructorTable['nolearn.lasagne.handlers.PrintLog'] = function() {};
        constructorTable['sklearn.calibration._CalibratedClassifier'] = function() {};
        constructorTable['sklearn.calibration._SigmoidCalibration'] = function() {};
        constructorTable['sklearn.calibration.CalibratedClassifierCV​'] = function() {};
        constructorTable['sklearn.compose._column_transformer.ColumnTransformer'] = function() {};
        constructorTable['sklearn.compose._target.TransformedTargetRegressor'] = function() {};
        constructorTable['sklearn.cluster._dbscan.DBSCAN'] = function() {};
        constructorTable['sklearn.decomposition.PCA'] = function() {};
        constructorTable['sklearn.decomposition.pca.PCA'] = function() {};
        constructorTable['sklearn.decomposition.truncated_svd.TruncatedSVD'] = function() {};
        constructorTable['sklearn.discriminant_analysis.LinearDiscriminantAnalysis'] = function() {};
        constructorTable['sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper'] = constructorTable['joblib.numpy_pickle.NumpyArrayWrapper'];
        constructorTable['sklearn.externals.joblib.numpy_pickle.NDArrayWrapper'] = function() {};
        constructorTable['sklearn.ensemble._bagging.BaggingClassifier'] = function() {};
        constructorTable['sklearn.ensemble._forest.RandomForestRegressor'] = function() {};
        constructorTable['sklearn.ensemble._forest.RandomForestClassifier'] = function() {};
        constructorTable['sklearn.ensemble._forest.ExtraTreesClassifier'] = function() {};
        constructorTable['sklearn.ensemble._gb_losses.BinomialDeviance'] = function() {};
        constructorTable['sklearn.ensemble._gb.GradientBoostingClassifier'] = function() {};
        constructorTable['sklearn.ensemble._voting.VotingClassifier'] = function() {};
        constructorTable['sklearn.ensemble.forest.RandomForestClassifier'] = function() {};
        constructorTable['sklearn.ensemble.forest.RandomForestRegressor'] = function() {};
        constructorTable['sklearn.ensemble.forest.ExtraTreesClassifier'] = function() {};
        constructorTable['sklearn.ensemble.gradient_boosting.BinomialDeviance'] = function() {};
        constructorTable['sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'] = function() {};
        constructorTable['sklearn.ensemble.gradient_boosting.LogOddsEstimator'] = function() {};
        constructorTable['sklearn.ensemble.gradient_boosting.MultinomialDeviance'] = function() {};
        constructorTable['sklearn.ensemble.gradient_boosting.PriorProbabilityEstimator'] = function() {};
        constructorTable['sklearn.ensemble.weight_boosting.AdaBoostClassifier'] = function() {};
        constructorTable['sklearn.feature_extraction._hashing.FeatureHasher'] = function() {};
        constructorTable['sklearn.feature_extraction.text.CountVectorizer​'] = function() {};
        constructorTable['sklearn.feature_extraction.text.HashingVectorizer'] = function() {};
        constructorTable['sklearn.feature_extraction.text.TfidfVectorizer​'] = function() {};
        constructorTable['sklearn.feature_extraction.text.TfidfTransformer​'] = function() {};
        constructorTable['sklearn.feature_selection.variance_threshold.VarianceThreshold'] = function() {};
        constructorTable['sklearn.impute._base.SimpleImputer'] = function() {};
        constructorTable['sklearn.impute.SimpleImputer'] = function() {};
        constructorTable['sklearn.linear_model._base.LinearRegression'] = function() {};
        constructorTable['sklearn.linear_model._coordinate_descent.ElasticNet'] = function() {};
        constructorTable['sklearn.linear_model.base.LinearRegression'] = function() {};
        constructorTable['sklearn.linear_model.sgd_fast.Hinge'] = function() {};
        constructorTable['sklearn.linear_model.LogisticRegression'] = function() {};
        constructorTable['sklearn.linear_model.logistic.LogisticRegression'] = function() {};
        constructorTable['sklearn.linear_model._logistic.LogisticRegression'] = function() {};
        constructorTable['sklearn.linear_model.LassoLars​'] = function() {};
        constructorTable['sklearn.linear_model.ridge.Ridge'] = function() {};
        constructorTable['sklearn.linear_model.sgd_fast.Log'] = function() {};
        constructorTable['sklearn.linear_model.stochastic_gradient.SGDClassifier'] = function() {};
        constructorTable['sklearn.metrics.scorer._PredictScorer'] = function() {};
        constructorTable['sklearn.model_selection._search.GridSearchCV'] = function() {};
        constructorTable['sklearn.naive_bayes.BernoulliNB'] = function() {};
        constructorTable['sklearn.naive_bayes.ComplementNB'] = function() {};
        constructorTable['sklearn.naive_bayes.GaussianNB'] = function() {};
        constructorTable['sklearn.naive_bayes.MultinomialNB'] = function() {};
        constructorTable['sklearn.neighbors.classification.KNeighborsClassifier'] = function() {};
        constructorTable['sklearn.neighbors.dist_metrics.newObj'] = function() {};
        constructorTable['sklearn.neighbors.kd_tree.newObj'] = function() {};
        constructorTable['sklearn.neighbors.KNeighborsClassifier​'] = function() {};
        constructorTable['sklearn.neighbors.KNeighborsRegressor'] = function() {};
        constructorTable['sklearn.neural_network._multilayer_perceptron.MLPClassifier'] = function() {};
        constructorTable['sklearn.neural_network._stochastic_optimizers.AdamOptimizer'] = function() {};
        constructorTable['sklearn.neural_network._stochastic_optimizers.SGDOptimizer'] = function() {};
        constructorTable['sklearn.neural_network.rbm.BernoulliRBM'] = function() {};
        constructorTable['sklearn.neural_network.multilayer_perceptron.MLPClassifier'] = function() {};
        constructorTable['sklearn.neural_network.multilayer_perceptron.MLPRegressor'] = function() {};
        constructorTable['sklearn.neural_network.stochastic_gradient.SGDClassifier'] = function() {};
        constructorTable['sklearn.pipeline.Pipeline'] = function() {};
        constructorTable['sklearn.pipeline.FeatureUnion'] = function() {};
        constructorTable['sklearn.preprocessing._data.RobustScaler'] = function() {};
        constructorTable['sklearn.preprocessing._data.StandardScaler'] = function() {};
        constructorTable['sklearn.preprocessing._discretization.KBinsDiscretizer'] = function() {};
        constructorTable['sklearn.preprocessing._encoders.OneHotEncoder'] = function() {};
        constructorTable['sklearn.preprocessing._function_transformer.FunctionTransformer'] = function() {};
        constructorTable['sklearn.preprocessing._label.LabelBinarizer'] = function() {};
        constructorTable['sklearn.preprocessing.data.Binarizer'] = function() {};
        constructorTable['sklearn.preprocessing.data.MaxAbsScaler'] = function() {};
        constructorTable['sklearn.preprocessing.data.MinMaxScaler'] = function() {};
        constructorTable['sklearn.preprocessing.data.Normalizer'] = function() {};
        constructorTable['sklearn.preprocessing.data.OneHotEncoder'] = function() {};
        constructorTable['sklearn.preprocessing.data.PolynomialFeatures'] = function() {};
        constructorTable['sklearn.preprocessing.data.PowerTransformer'] = function() {};
        constructorTable['sklearn.preprocessing.data.RobustScaler'] = function() {};
        constructorTable['sklearn.preprocessing.data.QuantileTransformer'] = function() {};
        constructorTable['sklearn.preprocessing.data.StandardScaler'] = function() {};
        constructorTable['sklearn.preprocessing.imputation.Imputer'] = function() {};
        constructorTable['sklearn.preprocessing.label.LabelBinarizer'] = function() {};
        constructorTable['sklearn.preprocessing.label.LabelEncoder'] = function() {};
        constructorTable['sklearn.preprocessing.label.MultiLabelBinarizer'] = function() {};
        constructorTable['sklearn.svm._classes.SVC'] = function() {};
        constructorTable['sklearn.svm.classes.LinearSVC'] = function() {};
        constructorTable['sklearn.svm.classes.SVC'] = function() {};
        constructorTable['sklearn.svm.classes.SVR'] = function() {};
        constructorTable['sklearn.tree._classes.DecisionTreeClassifier'] = function() {};
        constructorTable['sklearn.tree._classes.DecisionTreeRegressor'] = function() {};
        constructorTable['sklearn.tree._classes.ExtraTreeClassifier'] = function() {};
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
        constructorTable['xgboost.sklearn.XGBRegressor'] = function() {};

        functionTable['copy_reg._reconstructor'] = function(cls, base, state) {
            if (base == '__builtin__.object') {
                const obj = {};
                sklearn.Utility.applyType(obj, cls);
                return obj;
            }
            if (base == '__builtin__.tuple') {
                return state;
            }
            throw new sklearn.Error("Unknown base type '" + base + "'.");
        };
        functionTable['numpy.core.multiarray.scalar'] = function(dtype, rawData) {
            let data = rawData;
            if (typeof rawData === 'string' || rawData instanceof String) {
                data = new Uint8Array(rawData.length);
                for (let i = 0; i < rawData.length; i++) {
                    data[i] = rawData.charCodeAt(i);
                }
            }
            const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
            switch (dtype.name) {
                case 'uint8':
                    return dataView.getUint8(0);
                case 'float32':
                    return dataView.getFloat32(0, true);
                case 'float64':
                    return dataView.getFloat64(0, true);
                case 'int8':
                    return dataView.getInt8(0, true);
                case 'int16':
                    return dataView.getInt16(0, true);
                case 'int32':
                    return dataView.getInt32(0, true);
                case 'int64':
                    return new long.Long(dataView.getInt32(0, true), dataView.getInt32(4, true), false);
            }
            throw new sklearn.Error("Unknown scalar type '" + dtype.name + "'.");
        };
        functionTable['numpy.ma.core._mareconstruct'] = function(subtype /* , baseclass, baseshape, basetype */) {
            // _data = ndarray.__new__(baseclass, baseshape, basetype)
            // _mask = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
            // return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)
            const obj = {};
            sklearn.Utility.applyType(obj, subtype);
            return obj;
        };
        functionTable['numpy.random.__RandomState_ctor'] = function() {
            return {};
        };
        functionTable['_codecs.encode'] = function(obj /*, econding */) {
            return obj;
        };
        functionTable['collections.defaultdict'] = function(/* default_factory */) {
            return {};
        };
        functionTable['collections.OrderedDict'] = function(args) {
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
        };
        functionTable['__builtin__.bytearray'] = function(source, encoding /*, errors */) {
            if (source) {
                if (encoding === 'latin-1') {
                    const array = new Uint8Array(source.length);
                    for (let i = 0; i < source.length; i++) {
                        array[i] = source.charCodeAt(i);
                    }
                    return array;
                }
                throw new sklearn.Error("Unsupported bytearray encoding '" + JSON.stringify(encoding) + "'.");
            }
            return [];
        };
        functionTable['__builtin__.bytes'] = function(source, encoding /*, errors */) {
            if (source) {
                if (encoding === 'latin-1') {
                    const array = new Uint8Array(source.length);
                    for (let i = 0; i < source.length; i++) {
                        array[i] = source.charCodeAt(i);
                    }
                    return array;
                }
                throw new sklearn.Error("Unsupported bytearray encoding '" + JSON.stringify(encoding) + "'.");
            }
            return [];
        };
        functionTable['builtins.bytearray'] = function(data) {
            return { data: data };
        };
        functionTable['builtins.slice'] = function(start, stop, step) {
            return { start: start, stop: stop, step: step };
        };
        functionTable['cloudpickle.cloudpickle._builtin_type'] = function(name) {
            return name;
        };

        const unknownNameMap = new Set();
        const knownPackageMap = new Set([
            'sklearn', 'collections', '__builtin__', 'builtins',
            'copy_reg', 'gensim', 'joblib','xgboost', 'lightgbm', 'nolearn', 'numpy'
        ]);

        const function_call = (name, args) => {
            const func = functionTable[name];
            if (func) {
                return func.apply(null, args);
            }
            const obj = {};
            sklearn.Utility.applyType(obj, name);
            const constructor = constructorTable[name];
            if (constructor) {
                constructor.apply(obj, args);
            }
            else if (name && !unknownNameMap.has(name)) {
                unknownNameMap.add(name);
                if (knownPackageMap.has(name.split('.').shift())) {
                    exception(new sklearn.Error("Unknown function '" + name + "'."), false);
                }
            }
            return obj;
        };

        this._data = unpickler.load(function_call, null);

        const find_weights = function(objs) {

            for (const dict of objs) {
                if (dict && !Array.isArray(dict)) {
                    const weights = new Map();
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

            for (const list of objs) {
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
        };

        if (this._data) {
            this._weights = find_weights([ this._data, this._data.blobs ]);
            if (this._weights) {
                this._data = null;
            }
        }
    }

    get data() {
        return this._data;
    }

    get weights() {
        return this._weights;
    }
};

sklearn.Utility = class {

    static isTensor(obj) {
        return obj && obj.__module__ === 'numpy' && obj.__name__ === 'ndarray';
    }

    static applyType(obj, name){
        const parts = name.split('.');
        obj.__name__ = parts.pop();
        obj.__module__ = parts.join('.');
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