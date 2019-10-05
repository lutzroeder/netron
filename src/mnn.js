/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var mnn = mnn || {};
var mnn_private = {};
var base = base || require('./base');
var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;

mnn.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'mnn') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return host.require('./mnn-schema').then((mnn_schema) => {
            var identifier = context.identifier;
            var model = null;
            try {
                var buffer = context.buffer;
                var byteBuffer = new flatbuffers.ByteBuffer(buffer);
                mnn.schema = mnn_schema;
                model = mnn.schema.Net.getRootAsNet(byteBuffer);
            }
            catch (error) {
                host.exception(error, false);
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new mnn.Error(message + " in '" + identifier + "'.");
            }

            return mnn.Metadata.open(host).then((metadata) => {
                try {
                    return new mnn.Model(metadata, model, identifier);
                }
                catch (error) {
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new new mnn.Error(message + " in '" + identifier + "'.");
                }
            });
        });
    }
};

mnn.Model = class {
    constructor(metadata, net, fileName) {
        this._graphs = [];
        this._format = "MNN V2";
        this._description = "MNN Visualization Tool";
        this._graphs.push(new mnn.Graph(metadata, net, fileName));
    }

    get format() {
        return this._format;
    }

    get description() {
        return this._description;
    }

    get graphs() {
        return this._graphs;
    }
};

mnn.Graph = class {
    constructor(metadata, net, fileName) {
        this._name = fileName || "TestMNN";
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        var inputSet = new Set();
        for (var i = 0; i < net.oplistsLength(); i++) {
            var op = net.oplists(i);

            if (mnn.schema.OpTypeName[op.type()] === 'Input') {
                var argumentss = [];
                for (var o = 0; o < op.outputIndexesLength(); o++) {
                    var oindex = op.outputIndexes(o);
                    var tensorName = net.tensorName(oindex);
                    var tensorDataType = mnn_private.fetchDataType(metadata, net, oindex);
                    argumentss.push(new mnn.Argument(tensorName, tensorDataType, null))
                }
                this._inputs.push(new mnn.Parameter(op.name(), true, argumentss))
            } else {
                var opnode = new mnn.Node(metadata, op, i, net, []);
                this._nodes.push(opnode);
            }

            for (var k = 0; k < op.inputIndexesLength(); k++) {
                var iindex = op.inputIndexes(k);
                inputSet.add(iindex);
            }
        }

        for (var o = 0; o < net.tensorNameLength(); o++) {
            if (!inputSet.has(o)) {
                var tensorName = net.tensorName(o);
                var tensorDataType = mnn_private.fetchDataType(metadata, net, o);
                this._outputs.push(new mnn.Parameter(tensorName, true, [new mnn.Argument(tensorName, tensorDataType, null)]));
            }
        }
    }

    get name() {
        return this._name;
    }

    get groups() {
        return false;
    }

    get nodes() {
        return this._nodes;
    }

    get outputs() {
        return this._outputs;
    }

    get inputs() {
        return this._inputs;
    }
};

mnn.Node = class {
    constructor(metadata, op, index, net, args) {
        this._metadata = metadata;
        this._op = op;
        this._inputs = [];
        this._index = index;
        this._outputs = [];
        this._chains = [];
        this._attributes = [];

        this._buildInput(metadata, net);
        this._buildOutput(metadata, net);
        this._buildAttributes(metadata, op, net, args);
    }

    _buildInput(metadata, net) {
        for (var i = 0; i < this._op.inputIndexesLength(); i++) {
            var iindex = this._op.inputIndexes(i);
            var tensorName = net.tensorName(iindex);
            var tensorDataType = mnn_private.fetchDataType(metadata, net, iindex);   
            this._inputs.push(new mnn.Parameter(tensorName, true, [new mnn.Argument(tensorName, tensorDataType, null)]));
        }
    }

    _buildOutput(metadata, net) {
        for (var j = 0; j < this._op.outputIndexesLength(); j++) {
            var oindex = this._op.outputIndexes(j);
            var tensorName = net.tensorName(oindex);
            var tensorDataType = mnn_private.fetchDataType(metadata, net, oindex);
            this._outputs.push(new mnn.Parameter(tensorName, true, [new mnn.Argument(tensorName, tensorDataType, null)]));
        }
    }

    _buildExtraInfo(metadata, opParameterObject, opParameterName,  net) {
        if (!opParameterObject) return;

        // weights & bias
        switch (opParameterName) {
            case 'Convolution2D': {
                var common = opParameterObject.common();
                var outputCount = common.outputCount();
                var inputCount = common.inputCount();
                var kernelX = common.kernelX();
                var kernelY = common.kernelY();

                var weightArray = [];
                var weightSize = opParameterObject.weightLength();
                for (var i = 0; i < weightSize; i++) {
                    weightArray.push(opParameterObject.weight(i));
                }
                this._buildTensor("Float32", "weight", [outputCount, inputCount, kernelX, kernelY], weightArray);

                var biasArray = [];
                var biasSize = opParameterObject.biasLength();
                for (var i = 0; i < biasSize; i++) {
                    biasArray.push(opParameterObject.bias(i));
                }
                this._buildTensor("Float32", "bias", [outputCount], biasArray);

                break;
            }

            case 'InnerProduct': {
                var outputCount = opParameterObject.outputCount();
                var inputCount = opParameterObject.weightSize() / outputCount;

                var weightLength = opParameterObject.weightLength();
                var weightArray = [];
                for (var i = 0; i < weightLength; i++) {
                    weightArray.push(opParameterObject.weight(i));
                }
                this._buildTensor("Float32", "weight", [outputCount, inputCount], weightArray);

                var biasLength = opParameterObject.biasLength();
                var biasArray = [];
                for (var i = 0; i < biasLength; i++) {
                    biasArray.push(opParameterObject.bias(i));
                }
                this._buildTensor("Float32", "bias", [outputCount], biasArray);
                break;
            }

            case 'Scale': {
                var scaleDataCount = opParameterObject.channels();

                var scaleDataLength = opParameterObject.scaleDataLength();
                var scaleDataArray = [];
                for (var i = 0; i < scaleDataLength; i++) {
                    scaleDataArray.push(opParameterObject.scaleData(i));
                }
                this._buildTensor("Float32", "scale", [scaleDataCount], scaleDataArray);

                var biasDataArray = [];
                var biasDataLength = opParameterObject.biasDataLength();
                for (var i = 0; i < biasDataLength; i++) {
                    biasDataArray.push(opParameterObject.biasData(i));
                }
                this._buildTensor("Float32", "bias", [scaleDataCount], biasDataArray);
        
                break;
            }

            case 'BatchNorm': {
                var channels = opParameterObject.channels();

                var slopeDataArray = [];
                var slopeDataLength = opParameterObject.slopeDataLength();
                for (var i = 0; i < slopeDataLength; i++) {
                    slopeDataArray.push(opParameterObject.slopeData(i));
                }
                this._buildTensor("Float32", "slope", [channels], slopeDataArray);

                var meanDataArray = [];
                var meanDataLength = opParameterObject.meanDataLength();
                for (var i = 0; i < meanDataLength; i++) {
                    meanDataArray.push(opParameterObject.meanData(i));
                }
                this._buildTensor("Float32", "mean", [channels], meanDataArray);

                var varDataArray = [];
                var varDataLength = opParameterObject.varDataLength();
                for (var i = 0; i < varDataLength; i++) {
                    varDataArray.push(opParameterObject.varData(i));
                }
                this._buildTensor("Float32", "variance", [channels], varDataArray);
                
                var biasDataArray = [];
                var biasDataLength = opParameterObject.biasDataLength();
                for (var i = 0; i < biasDataLength; i++) {
                    biasDataArray.push(opParameterObject.biasData(i));
                }
                this._buildTensor("Float32", "bias", [channels], biasDataArray);

                break;
            }

            case 'PRelu':{
                var slopeCount = opParameterObject.slopeCount();

                var slopeLength = opParameterObject.slopeLength();
                var slopeArray = [];
                for (var i = 0; i < slopeLength; i++) {
                    slopeArray.push(opParameterObject.slope(i));
                }
                this._buildTensor("Float32", "slope", [slopeCount], slopeArray);
                break;
            }

            case 'Normalize':{
                var scaleLength = opParameterObject.scaleLength();
                var scaleArray = [];
                for (var i = 0; i < scaleLength; i++) {
                    scaleArray.push(opParameterObject.scale(i));
                } 
                this._buildTensor("Float32", "scale", [scaleLength], scaleArray);
                break;
            }
        }
    }

    _buildTensor(dataType, name, dimensions, value) {
        this._inputs.push(new mnn.Parameter(name, true, [
            new mnn.Argument (
                '', 
                null, 
                new mnn.Tensor (
                    new mnn.TensorType (
                        dataType, 
                        new mnn.TensorShape(dimensions)
                    ),
                    value
                ),    
            )
        ]));
    }

    _buildAttributes(metadata, op, net, args) {
        var opParameter = op.mainType();    
        var opParameterName = mnn.schema.OpParameterName[opParameter];
        
        // get corresponding main type
        var mainConstructor = mnn.schema[opParameterName];
        var opParameterObject = null;
        if (typeof mainConstructor === 'function') {
            var mainTemplate = Reflect.construct(mainConstructor, []);
            opParameterObject = op.main(mainTemplate);
        }

        this._recursivelyBuildAttributes(metadata, net, opParameterObject, opParameterName, this._attributes);
        this._buildExtraInfo(metadata, opParameterObject, opParameterName, net);
    }

    _recursivelyBuildAttributes(metadata, net, opParameterObject, opParameterName, attributeHolders) {
        if (!opParameterObject) return;

        var attributeName;
        var attributeNames = [];
        var attributeNamesMap = {};
        for (attributeName of Object.keys(Object.getPrototypeOf(opParameterObject))) {
            if (attributeName != '__init') {
                attributeNames.push(attributeName);
            }
            attributeNamesMap[attributeName] = true;
        }

        var attributeArrayNamesMap = {}; 
        for (attributeName of Object.keys(attributeNamesMap)) {
            if (attributeNamesMap[attributeName + 'Length']) { // some bugs without array
                attributeArrayNamesMap[attributeName] = true;
                attributeNames = attributeNames.filter((item) => item != (attributeName + 'Array') && item != (attributeName + 'Length'));
            }
        }

        for (attributeName of attributeNames) {
            if (mnn_private.isAttributeInvisible(opParameterName, attributeName)) {
                continue;
            }

            if (opParameterObject[attributeName] && typeof opParameterObject[attributeName] == 'function') {
                var value = null;
                if (attributeArrayNamesMap[attributeName]) {
                    var array = [];
                    var length = opParameterObject[attributeName + 'Length']();
                    for (var l = 0; l < length; l++) {
                        array.push(opParameterObject[attributeName](l));
                    }
                    value = array;
                }
                else {
                    value = opParameterObject[attributeName]();
                    if (typeof value === 'object') {
                        var name = mnn_private.findOpParameterObjectClassName(value);
                        console.log(name);
                        this._recursivelyBuildAttributes(metadata, net, value, name, attributeHolders);   
                        value = null;
                    } else {
                        if (opParameterName) { 
                            // If we found candidate alias name, replace value with xxxName!
                            var aliasName = metadata.getAlias(opParameterName, attributeName);
                            if (aliasName) {
                                value = mnn.schema[aliasName + "Name"][value];
                            }
                        }
                    }
                }

                if (value != null) {
                    var attribute = new mnn.Attribute(metadata, attributeName, value);
                    attributeHolders.push(attribute);
                }                
            }
        }
    }

    get operator() {
        return mnn.schema.OpTypeName[this._op.type()];
    }

    get name() {
        return this._op.name();
    }

    get domain() {
        return null;
    }

    get documentation() {
        return 'MNN V2 Operator';
    }

    get group() {
        return null;
    }

    get category() {
        let schema = this._metadata.getSchema(this.operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chains;
    }

    get attributes() {
        return this._attributes;
    }
};

mnn.Attribute = class {

    constructor(metadata, name, value, visible) {
        this._type = null;
        this._value = value;
        this._name = name;
        this._visible = visible;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

mnn.Parameter = class {
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

mnn.Argument = class {
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

mnn.Tensor = class {
    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Weight';
    }

    toString() {
        var context = this._context();
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    get type() {
        return this._type;
    }

    get state() {
        return null;
    }

    get value() {
        var context = this._context();
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.data = this._dataType;
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
                results.push(this._data[context.index]);
                context.index++;
                context.count++;
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
};

mnn.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape()
    {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
}

mnn.TensorShape = class {
    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

// 这里换成需要的数据
mnn.Metadata = class {

    static open(host) {
        if (mnn.Metadata._metadata) {
            return Promise.resolve(tflite.Metadata._metadata);
        }
        return host.request(null, 'mnn-metadata.json', 'utf-8').then((data) => {
            mnn.Metadata._metadata = new mnn.Metadata(data);
            return mnn.Metadata._metadata;
        }).catch(() => {
            mnn.Metadata._metadata = new mnn.Metadata(null);
            return mnn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._opTypeStyle = {};
        this._opParameterAlias = {};
        if (data) {
            var jsonData = JSON.parse(data);
            var items = jsonData.OpTypeStyle;
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        this._opTypeStyle[item.name] = item.schema;
                    }
                }
            }

            var aliasItems = jsonData.OpParameterAlias;
            if (aliasItems) {
                for (var item of aliasItems) {
                    if (item.name && item.alias) {
                        this._opParameterAlias[item.name] = item.alias;
                    }
                }
            }
        }
    }

    getSchema(operatorName) {
        return this._opTypeStyle[operatorName];
    }

    getAlias(opParameterName, propertyName) {
        var opParameterAlias = this._opParameterAlias[opParameterName];
        return opParameterAlias ? opParameterAlias[propertyName]: null;
    }
};

mnn.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading MNN model.';
    }
};

/************************* MNN Private Util ************************/
mnn_private.isAttributeInvisible = function(opParameterName, attributeName) {
    let invisibleAttributes = {
        "Convolution2D":{"weight":true, "bias":true},
        "InnerProduct":{"weight":true,"bias":true},
        "Scale":{"scaleData":true, "biasData":true},
        "BatchNorm":{"slopeData":true, "meanData":true, "varData":true, "biasData":true},
        "Normalize":{"scale":true},
        "PRelu":{"slope":true}
    };

    let invisibleSet = invisibleAttributes[opParameterName];
    if (!invisibleSet) return false;

    if (invisibleSet[attributeName]) return true;
    return false;
}

mnn_private.findOpParameterObjectClassName = function(opParameterObject) {
    var keys = Object.getOwnPropertyNames(mnn.schema);
    for (var key of keys) {
        var cls = mnn.schema[key];
        if (typeof cls === "function" && opParameterObject instanceof cls) {
            return key;
        }
    }
    return null;
}

mnn_private.fetchDataType = function(metadata, net, tensorIndex) {
    var tensorDescribe = net.extraTensorDescribe(tensorIndex);
    var tensorBlob = tensorDescribe ? tensorDescribe.blob() : null;
    var tensorDataType = tensorBlob ? tensorBlob.dataType() : null;
    return tensorDataType ? mnn.schema.DataTypeName[tensorDataType] : "Unknown";
}

/*************************  MNN Private Util ************************/
if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mnn.ModelFactory;
}

