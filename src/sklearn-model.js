/*jshint esversion: 6 */

class SklearnModelFactory {

    match(context) {
        var extension = context.identifier.split('.').pop();
        return extension == 'pkl' || extension == 'joblib';
    }

    open(context, host, callback) { 
        host.require('pickle', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            var obj = null;

            try {
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
                            else {
                                debugger;
                                throw new SklearnError("Unknown dtype '" + obj.toString() + "'.");
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
                                throw new SklearnError("Unknown numpy.dtype setstate length '" + state.length.toString() + "'.");
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
                };
                constructorTable['joblib.numpy_pickle.NumpyArrayWrapper'] = function(subtype, shape, dtype) {
                    this.__setstate__ = function(state, reader) {
                        this.subclass = state.subclass;
                        this.dtype = state.dtype;
                        this.shape = state.shape;
                        this.order = state.order;
                        this.allow_mmap = state.allow_mmap;
                        var size = 1;
                        this.shape.forEach((dimension) => {
                            size *= dimension;
                        });
                        if (this.dtype.name == 'object') {
                            this._object = unpickler.load(function_call, null);
                        }
                        else {
                            this.data = reader.readBytes(size * this.dtype.itemsize);
                        }
                    };
                };
                constructorTable['lightgbm.sklearn.LGBMRegressor'] = function() {};
                constructorTable['lightgbm.basic.Booster'] = function() {};
                constructorTable['sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper'] = constructorTable['joblib.numpy_pickle.NumpyArrayWrapper'];
                constructorTable['sklearn.compose._column_transformer.ColumnTransformer'] = function() {};
                constructorTable['sklearn.utils.deprecation.DeprecationDict'] = function() {};
                constructorTable['sklearn.ensemble.forest.RandomForestClassifier'] = function() {};
                constructorTable['sklearn.ensemble.forest.ExtraTreesClassifier'] = function() {};
                constructorTable['sklearn.ensemble.weight_boosting.AdaBoostClassifier'] = function() {};
                constructorTable['sklearn.impute.SimpleImputer'] = function() {};
                constructorTable['sklearn.linear_model.LogisticRegression'] = function() {}; 
                constructorTable['sklearn.linear_model.logistic.LogisticRegression'] = function() {};
                constructorTable['sklearn.model_selection._search.GridSearchCV'] = function() {};
                constructorTable['sklearn.naive_bayes.BernoulliNB'] = function() {};
                constructorTable['sklearn.naive_bayes.ComplementNB'] = function() {};
                constructorTable['sklearn.naive_bayes.GaussianNB'] = function() {};
                constructorTable['sklearn.naive_bayes.MultinomialNB'] = function() {};
                constructorTable['sklearn.neural_network.rbm.BernoulliRBM'] = function() {};
                constructorTable['sklearn.pipeline.Pipeline'] = function() {};
                constructorTable['sklearn.preprocessing._encoders.OneHotEncoder'] = function() {};
                constructorTable['sklearn.preprocessing.data.Binarizer'] = function() {};
                constructorTable['sklearn.preprocessing.data.StandardScaler'] = function() {};
                constructorTable['sklearn.svm.classes.SVC'] = function() {};
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
                constructorTable['sklearn.tree.tree.ExtraTreeClassifier'] = function() { };
                constructorTable['collections.defaultdict'] = function(default_factory) {
                };

                functionTable['copy_reg._reconstructor'] = function(cls, base, state) {
                    if (base == '__builtin__.object') {
                        var obj = {};
                        obj.__type__ = cls;
                        return obj;
                    }
                    throw new SklearnError("Unknown base type '" + base + "'.");
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
                            // var offset = (position * dtype.itemsize) + dataView.byteOffset;
                            // return new Int64(new Uint8Array(dataView.buffer.slice(offset, offset + dtype.itemsize)));
                            return new Int64(data.subarray(0, dtype.itemsize));
                    }
                    throw new SklearnError("Unknown scalar type '" + dtype.name + "'.");
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
                        host.exception(new SklearnError("Unknown function '" + name + "'."), false);
                    }
                    return obj;
                };

                obj = unpickler.load(function_call, null);
            }
            catch (error) {
                callback(error);
                return;
            }

            try {
                var model = new SklearnModel(obj);
                SklearnOperatorMetadata.open(host, (err, metadata) => {
                    callback(null, model);
                });
            }
            catch (error) {
                callback(new SklearnError(error.message), null);
                return;
            }
        });
    }
}

class SklearnModel {

    constructor(obj) {
        this._format = 'scikit-learn';
        if (obj._sklearn_version) {
            this._format += ' ' + obj._sklearn_version.toString();
        }

        this._graphs = [];
        this._graphs.push(new SklearnGraph(obj));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }

}

class SklearnGraph {

    constructor(obj) {
        this._nodes = [];
        this._groups = false;

        var input = 'data';
        switch (obj.__type__) {
            case 'sklearn.pipeline.Pipeline':
                this._groups = true;
                for (var step of obj.steps) {
                    this._nodes.push(new SklearnNode('pipeline', step[0], step[1], [ input ], [ step[0] ]));
                    input = step[0];
                }
                break;
            default:
                this._nodes.push(new SklearnNode(null, null, obj, [], []));
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

}

class SklearnNode {

    constructor(group, name, obj, inputs, outputs) {
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
                    this._attributes.push(new SklearnAttribute(this, key, value));
                }
                else {
                    switch (value.__type__) {
                        case 'joblib.numpy_pickle.NumpyArrayWrapper':
                        case 'sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper':
                        case 'numpy.core.multiarray._reconstruct':
                            this._initializers.push(new SklearnTensor(key, value));
                            break;
                        default: 
                            this._attributes.push(new SklearnAttribute(this, key, value));
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
        return SklearnOperatorMetadata.operatorMetadata.getOperatorDocumentation(this.operator);
    }

    get inputs() {

        var inputs = this._inputs.map((input) => {
            return {
                name: input,
                connections: [ { id: input } ]
            };
        });

        this._initializers.forEach((initializer) => {
            var input = { connections: [] };
            input.name = initializer.name;
            input.connections.push({
                initializer: initializer,
                type: initializer.type
            });
            inputs.push(input);
        });

        return inputs;
    }

    get outputs() {
        return this._outputs.map((output) => {
            return {
                name: output,
                connections: [ { id: output } ]
            };
        });
    }

    get attributes() {
        return this._attributes;
    }
}

class SklearnAttribute {

    constructor(node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        if (this._value && this._value.constructor.name == 'Int64') {
            return this._value.toString();
        }
        return JSON.stringify(this._value);
    }

    get visible() {
        return SklearnOperatorMetadata.operatorMetadata.getAttributeVisible(this._node.operator, this._name, this._value);
    }
}

class SklearnTensor {

    constructor(name, value) {
        SklearnTensor._escapeRegex = /\\(u\{([0-9A-Fa-f]+)\}|u([0-9A-Fa-f]{4})|x([0-9A-Fa-f]{2})|([1-7][0-7]{0,2}|[0-7]{2,3})|(['"tbrnfv0\\]))|\\U([0-9A-Fa-f]{8})/g;
        SklearnTensor._escapeMap = { '0': '\0', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v', '\'': '\'', '"': '"', '\\': '\\' };

        this._name = name;

        switch (value.__type__) {
            case 'joblib.numpy_pickle.NumpyArrayWrapper':
            case 'sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper':
                this._kind = 'Array Wrapper';
                this._type = new SklearnTensorType(value.dtype.name, value.shape);
                this._data = value.data;
                break;
            case 'numpy.core.multiarray._reconstruct':
                this._kind = 'Array';
                this._type = new SklearnTensorType(value.typecode.name, value.shape);
                var rawdata = SklearnTensor._unescape(value.rawdata);
                this._data = new Uint8Array(rawdata.length);
                for (var i = 0; i < this._data.length; i++) {
                    this._data[i] = rawdata.charCodeAt(i);
                }
                break;
            default:
                debugger;
        }
    }

    get id() {
        return this._name;
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
                return OnnxTensor._stringify(value, '', '    ');
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
        context.shape = this._type.shape;

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
        var size = context.shape[dimension];
        if (dimension == context.shape.length - 1) {
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
                        results.push(new Int64(context.rawData.subarray(context.index, context.index + 8)));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'uint64':
                        results.push(new Uint64(context.rawData.subarray(context.index, context.index + 8)));
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
            var items = value.map((item) => OnnxTensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(']');
            return result.join('\n');
        }
        return indentation + value.toString();
    }

    static _unescape(text) {
        return text.replace(SklearnTensor._escapeRegex, (_, __, varHex, longHex, shortHex, octal, specialCharacter, python) => {
            if (varHex !== undefined) {
                return String.fromCodePoint(parseInt(varHex, 16));
            } else if (longHex !== undefined) {
                return String.fromCodePoint(parseInt(longHex, 16));
            } else if (shortHex !== undefined) {
                return String.fromCodePoint(parseInt(shortHex, 16));
            } else if (octal !== undefined) {
                return String.fromCodePoint(parseInt(octal, 8));
            } else if (python !== undefined) {
                return String.fromCodePoint(parseInt(python, 16));
            } else {
                return SklearnTensor._escapeMap[specialCharacter];
            }
        });
    };

}

class SklearnTensorType {

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
        return this.dataType + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }
}

class SklearnOperatorMetadata {

    static open(host, callback) {
        if (SklearnOperatorMetadata.operatorMetadata) {
            callback(null, SklearnOperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'sklearn-metadata.json', 'utf-8', (err, data) => {
                SklearnOperatorMetadata.operatorMetadata = new SklearnOperatorMetadata(data);
                callback(null, SklearnOperatorMetadata.operatorMetadata);
            });    
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema)
                    {
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getOperatorDocumentation(operator) {
        var schema = this._map[operator];
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = operator;
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

    getAttributeVisible(operator, attributeName, attributeValue) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributeMap) {
                schema.attributeMap = {};
                schema.attributes.forEach(attribute => {
                    schema.attributeMap[attribute.name] = attribute;
                });
            }
            var attribute = schema.attributeMap[attributeName];
            if (attribute) {
                if (attribute.hasOwnProperty('option')) {
                    if (attribute.option == 'optional' && attributeValue == null) {
                        return false;
                    }
                }
                if (attribute.hasOwnProperty('visible')) {
                    return attribute.visible;
                }
                if (attribute.hasOwnProperty('default')) {
                    return !KerasOperatorMetadata.isEquivalent(attribute.default, attributeValue);
                }
            }
        }
        return true;
    }

    getOperatorCategory(operator) {
        var schema = this._map[operator];
        if (schema) {
            var category = schema.category;
            if (category) {
                return category;
            }
        }
        return null;
    }

}

class SklearnError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading scikit-learn model.';
    }
}
