/*jshint esversion: 6 */

class SklearnModelFactor {

    match(context) {
        var extension = context.identifier.split('.').pop();
        return extension == 'pkl';
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

                var functionTable = {};

                functionTable['sklearn.linear_model.LogisticRegression'] = function() {}; 
                functionTable['sklearn.naive_bayes.GaussianNB'] = function() {};
                functionTable['sklearn.preprocessing.data.Binarizer'] = function() {};
                functionTable['sklearn.svm.classes.SVC'] = function() {};
                functionTable['numpy.dtype'] = function(obj, align, copy) { 
                    switch (obj) {
                        case 'i4': this.name = 'int32'; this.itemsize = 4; break;
                        case 'i8': this.name = 'int64'; this.itemsize = 8; break;
                        case 'f4': this.name = 'float32'; this.itemsize = 4; break;
                        case 'f8': this.name = 'float64'; this.itemsize = 8; break;
                        default: throw new SklearnError("Unknown dtype '" + obj.toString() + "'.");
                    }

                    this.obj = obj;
                    this.align = align;
                    this.copy = copy;
                    this.__setstate__ = function(state) {
                        switch (state.length) {
                            case 8:
                                this.version = state[0];
                                this.byteorder = state[1];
                                // &subarray, &names, &fields, &elsize, &alignment, &int_dtypeflags
                                break;
                            default:
                                throw new pickle.Error('Unknown dtype length');
                        }
                    };
                };
                functionTable['numpy.core.multiarray._reconstruct'] = function(subtype, shape, dtype) {
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
                functionTable['joblib.numpy_pickle.NumpyArrayWrapper'] = function(subtype, shape, dtype) {
                    this.__setstate__ = function(state, reader) {
                        this.subclass = state.subclass;
                        this.dtype = state.dtype;
                        this.shape = state.shape;
                        this.order = state.order;
                        this.allow_mmap = state.allow_mmap;
                        var size = this.dtype.itemsize;
                        this.shape.forEach((dimension) => {
                            size *= dimension;
                        });
                        this.data = reader.readBytes(size);
                    };
                };

                var function_call = (name, args) => {
                    if (name == 'copy_reg._reconstructor' && args[1] == '__builtin__.object') {
                        name = args[0];
                        args = [];
                    }
                    if (functionTable[name]) {
                        var obj = { __type__: name };
                        functionTable[name].apply(obj, args);
                        return obj;
                    }
                    throw new pickle.Error("Unknown function '" + name + "'.");
                };
    
                obj = unpickler.load(function_call, null);
            }
            catch (error) {
                callback(error);
            }

            try {
                var model = new SklearnModel(obj);
                SklearnOperatorMetadata.open(host, (err, metadata) => {
                    callback(null, model);
                });
            }
            catch (error) {
                callback(new SklearnError(error.message), null);
            }
        });
    }
}

class SklearnModel {

    constructor(obj) {
        this._format = 'scikit-learn';
        if (root._sklearn_version) {
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
        this._nodes.push(new SklearnNode(obj));
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

    constructor(obj) {
        this._operator = obj.__type__.split('.').pop(); 
        this._attributes = [];

        this._initializers = [];

        Object.keys(obj).forEach((key) => {
            if (!key.startsWith('_')) {
                if (key.endsWith('_')) {
                    var name = key.substring(0, key.length - 1);
                    var value = obj[key];
                    if (Array.isArray(value) || Number.isInteger(value)) {
                        this._attributes.push(new SklearnAttribute(name, value));
                    }
                    else {
                        this._initializers.push(new SklearnTensor(name, value));
                    }
                }
                else {
                    this._attributes.push(new SklearnAttribute(key, obj[key]));
                }
            }
        });
    }

    get operator() {
        return this._operator;
    }

    get inputs() {
        var inputs = [];
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
        return [];
    }

    get attributes() {
        return this._attributes;
    }
}

class SklearnAttribute {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return JSON.stringify(this._value);
    }

    get visible() {
        return true;
    }
}

class SklearnTensor {

    constructor(name, value) {
        this._name = name;
        this._value = value;

        switch (value.__type__) {
            case 'joblib.numpy_pickle.NumpyArrayWrapper':
                this._kind = 'NumpyArrayWrapper';
                this._type = new SklearnTensorType(value.dtype.name, value.shape);
                this._data = value.data;
                break;
            case 'numpy.core.multiarray._reconstruct':
                this._kind = 'NumPy Array';
                this._type = new SklearnTensorType(value.typecode.name, value.shape);
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
        return 'Not Implemented';
    }

    get value() {
        return null;
    }

    get toString() {
        return '';
    }
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
