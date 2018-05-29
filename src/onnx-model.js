/*jshint esversion: 6 */

var onnx = null;

class OnnxModelFactory {

    match(buffer, identifier) {
        var extension = identifier.split('.').pop();
        return (identifier != 'saved_model.pb') && (identifier != 'predict_net.pb') && (extension == 'onnx' || extension == 'pb');
    }

    open(buffer, identifier, host, callback) { 
        host.import('/onnx.js', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var model = null;
            try {
                onnx = protobuf.roots.onnx.onnx;
                model = onnx.ModelProto.decode(buffer);
            }
            catch (error) {
                callback(new OnnxError('Protocol Buffer loader failed to decode onnx.ModelProto input stream (' + error.message + ').'), null);
                return;
            }
            var result = null;
            try {
                result = new OnnxModel(model);
            }
            catch (error) {
                callback(new OnnxError(error.message), null);
                return;
            }
            OnnxOperatorMetadata.open(host, (err, metadata) => {
                callback(null, result);
            });
        });
    }
}

class OnnxModel {

    constructor(model) {
        this._model = model;
        this._graphs = [];
        this._activeGraph = null;
        if (this._model.graph) {
            var metadata = new OnnxGraphOperatorMetadata(this._model);
            var graph = new OnnxGraph(this, metadata, this._model.graph, 0);
            this._graphs.push(graph);
            this._activeGraph = graph;
        }
    }

    get properties() {
        var results = [];
        var format = 'ONNX';
        if (this._model.irVersion) {
            format = format + ' v' + this._model.irVersion.toString();
            // var major = (this._model.irVersion >> 16) & 0x0f;
            // var minor = (this._model.irVersion >> 8) & 0x0f;
            // var revision = (this._model.irVersion) & 0x0f;
            // format = format + ' v' + major.toString() + '.' + minor.toString() + '.' + revision.toString();
        }
        results.push({ name: 'Format', value: format });
        if (this._model.opsetImport && this._model.opsetImport.length > 0) {
            var opsetImports = [];
            this._model.opsetImport.forEach((opsetImport) => {
                var domain = opsetImport.domain ? opsetImport.domain : 'ai.onnx';
                var result = domain + ' v' + opsetImport.version;
                if (!opsetImports.includes(result)) {
                    opsetImports.push(result);
                }
            });
            results.push({ name: 'Imports', value: opsetImports.join(', ') });
        }
        var producer = [];
        if (this._model.producerName) {
            producer.push(this._model.producerName);
        }
        if (this._model.producerVersion && this._model.producerVersion.length > 0) {
            producer.push(this._model.producerVersion);
        }
        if (producer.length > 0) {
            results.push({ 'name': 'Producer', 'value': producer.join(' ') });
        }
        if (this._model.domain) {
            results.push({ name: 'Domain', value: this._model.domain });
        }
        if (this._model.modelVersion) {
            results.push({ name: 'Version', value: this._model.modelVersion });
        }
        if (this._model.docString) {
            results.push({ name: 'Description', value: this._model.docString });
        }
        var metadata = {};
        if (this._model.metadataProps)
        {
            this._model.metadataProps.forEach((metadataProp) => {
                metadata[metadataProp.key] = metadataProp.value;
            });
        }
        if (metadata.author) {
            results.push({ name: 'Author', value: metadata.author });
        }
        if (metadata.company) {
            results.push({ name: 'Company', value: metadata.company });
        }
        if (metadata.converted_from) {
            results.push({ name: 'Source', value: metadata.converted_from });
        }
        var license = [];
        if (metadata.license && metadata.license.length > 0) {
            license.push(metadata.license);
        }
        if (metadata.license_url && metadata.license_url.length > 0) {
            license.push('<a href=\'' + metadata.license_url + '\'>' + metadata.license_url + '</a>');
        }
        if (license.length > 0) {
            results.push({ name: 'License', value: license.join(' ') });
        }

        return results;
    }

    get graphs() {
        return this._graphs;
    }
}

class OnnxGraph {

    constructor(model, metadata, graph, index) {
        this._model = model;
        this._metadata = metadata;
        this._graph = graph;
        this._nodes = [];

        if (this._graph) {
            this._name = this._graph.name ? this._graph.name : ('(' + index.toString() + ')');
            
            this._outputMap = {};
            this._graph.node.forEach((node) => {
                node.output.forEach((output) => {
                    this._outputMap[output] = (this._outputMap[output] || 0) + 1;
                });
            });
    
            this._initializerMap = [];
            this._graph.initializer.forEach((tensor) => {
                this._initializerMap[tensor.name] = new OnnxTensor(tensor, tensor.name, 'Initializer');
            });
            this._graph.node.forEach((node) => {
                var add = true;
                if (node.opType == 'Constant' && node.output && node.output.length == 1 && this._outputMap[node.output[0]] == 1) {
                    node.attribute.forEach((attribute) => {
                        if (attribute.name == 'value' && attribute.t) {
                            var name = node.output[0];
                            this._initializerMap[name] = new OnnxTensor(attribute.t, name, 'Constant');
                            add = false;
                        }
                    });
                }
                if (add) {
                    this._nodes.push(new OnnxNode(this, node));
                }
            });
        }
    }

    get model() {
        return this._model;
    }

    get name() {
        return this._name;
    }

    get description() {
        if (this._graph && this._graph.docString) {
            return this._graph.docString;
        }
        return '';
    }

    get groups() {
        return false;
    }

    get inputs() {
        if (!this._inputs) {
            this._inputs = [];
            if (this._graph) {
                var initializerMap = {};
                this._graph.initializer.forEach((tensor) => {
                    initializerMap[tensor.name] = true;
                });
                this._graph.input.forEach((valueInfo, index) => {
                    if (!initializerMap[valueInfo.name]) {
                        this._inputs.push({
                            id: valueInfo.name,
                            name: valueInfo.name,
                            description: valueInfo.docString,
                            type: OnnxTensor.formatType(valueInfo.type)
                        });
                    }
                });
            }
        }
        return this._inputs;
    }

    get outputs() {
        if (!this._outputs) {
            this._outputs = [];
            if (this._graph) {
                this._outputs = this._graph.output.map((valueInfo) => {
                    return {
                        id: valueInfo.name,
                        name: valueInfo.name,
                        description: valueInfo.docString,
                        type: OnnxTensor.formatType(valueInfo.type)
                    };
                });
            }
        }
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    getInitializer(input) {
        var initializer = this._initializerMap[input];
        return initializer ? initializer : null;
    }

    get metadata() {
        return this._metadata;
    }
}

class OnnxNode {

    constructor(graph, node) {
        this._graph = graph;
        this._node = node;
    }

    get operator() {
        return this._node.opType;
    }

    get name() {
        return this._node.name ? this._node.name : null;
    }

    get description() {
        return this._node.docString ? this._node.docString : null;
    }

    get primitive() {
        return null;
    }

    get documentation() {
        return this._graph.metadata.getOperatorDocumentation(this);
    }

    get domain() {
        return this._node.domain ? this._node.domain : null;
    }

    get category() {
        return this._graph.metadata.getOperatorCategory(this);
    }

    get group() {
        return null;
    }

    get inputs() {
        if (this._node.input) {
            var inputs = this._graph.metadata.getInputs(this);
            inputs.forEach((input) => {
                input.connections.forEach((connection) => {
                    var initializer = this._graph.getInitializer(connection.id);
                    if (initializer) {
                        connection.initializer = initializer;
                        connection.type = initializer.type;
                    }
                });
            });          
            return inputs;
        }
        return [];
    }

    get outputs() {
        return this._graph.metadata.getOutputs(this);
    }

    get dependencies() {
        return [];
    }

    get attributes() {
        var result = null;
        var node = this._node;
        if (node.attribute && node.attribute.length > 0) {
            result = [];
            node.attribute.forEach((attribute) => { 
                result.push(new OnnxAttribute(this, attribute));
            });
        }
        return result;
    }

    get graph() {
        return this._graph;
    }

    get data() {
        return this._node;
    }
}

class OnnxAttribute {
    constructor(node, attribute) {
        this._node = node;
        this._attribute = attribute;
    }

    get name() {
        return this._attribute.name;
    }

    get type() {
        if (this._attribute.hasOwnProperty('type')) { 
            var type = OnnxTensor.formatElementType(this._attribute.type);
            if ((this._attribute.ints && this._attribute.ints.length > 0) ||
                (this._attribute.floats && this._attribute.floats.length > 0) ||
                (this._attribute.strings && this._attribute.strings.length > 0)) {
                return type + '[]';
            }
        }
        else if (this._attribute.hasOwnProperty('t')) {
            return OnnxTensor.formatTensorType(this._attribute.t);
        }
        return this._node.graph.metadata.getAttributeType(this._node, this._attribute.name);
    }

    get value() {
        if (this._attribute.ints && this._attribute.ints.length > 0) {
            if (this._attribute.ints.length > 65536) {
                return "...";
            }
            return this._attribute.ints.map((v) => { return v.toString(); }).join(', '); 
        }
        else if (this._attribute.floats && this._attribute.floats.length > 0) {
            if (this._attribute.floats.length > 65536) {
                return "...";
            }
            return this._attribute.floats.map(v => v.toString()).join(', ');
        }
        else if (this._attribute.strings && this._attribute.strings.length > 0) {
            if (this._attribute.strings.length > 65536) {
                return "...";
            }
            return this._attribute.strings.map((s) => {
                if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                    return '"' + String.fromCharCode.apply(null, s) + '"';
                }
                return s.map(v => v.toString()).join(', ');
            }).join(', ');
        }
        else if (this._attribute.s && this._attribute.s.length > 0) {
            if (this._attribute.s.filter(c => c <= 32 && c >= 128).length == 0) {
                return '"' + String.fromCharCode.apply(null, this._attribute.s) + '"';
            }
            return this._attribute.s.map(v => v.toString()).join(', ');
        }
        else if (this._attribute.hasOwnProperty('f')) {
            return this._attribute.f.toString();
        }
        else if (this._attribute.hasOwnProperty('i')) {
            return this._attribute.i.toString();
        }
        else if (this._attribute.hasOwnProperty('t')) {
            return new OnnxTensor(this._attribute.t).value;
        }
        // debugger;
        return '?';
    }

    get description() {
        return this._attribute.docString ? this._attribute.docString : null;
    }

    get visible() {
        return this._node.graph.metadata.getAttributeVisible(this._node, this);
    }

    get tensor() {
        return this._attribute.hasOwnProperty('t');
    }
}

class OnnxTensor {

    constructor(tensor, id, kind) {
        this._tensor = tensor;
        this._id = id;
        if (kind) {
            this._kind = kind;
        }
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._tensor.name ? this._tensor.name : this._id; 
    }

    get kind() {
        return this._kind ? this._kind : null;
    }

    get type() {
        return OnnxTensor.formatTensorType(this._tensor);
    }

    get value() { 
        if (!this._tensor.dataType) {
            return 'Tensor has no data type.';
        }
        if (!this._tensor.dims) {
            return 'Tensor has no dimensions.';
        }

        switch (this._tensor.dataType) {
            case onnx.TensorProto.DataType.FLOAT:
                if (this._tensor.floatData && this._tensor.floatData.length > 0) {
                    this._data = this._tensor.floatData;
                }
                else if (this._tensor.rawData && this._tensor.rawData.length > 0) {
                    this._rawData = new DataView(this._tensor.rawData.buffer, this._tensor.rawData.byteOffset, this._tensor.rawData.byteLength);
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.DOUBLE:
                if (this._tensor.doubleData && this._tensor.doubleData.length > 0) {
                    this._data = tensor.doubleData;
                }
                else if (this._tensor.rawData && this._tensor.rawData.length > 0) {
                    this._rawData = new DataView(this._tensor.rawData.buffer, this._tensor.rawData.byteOffset, this._tensor.rawData.byteLength);
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.INT32:
                if (this._tensor.int32Data && this._tensor.int32Data.length > 0) {
                    this._data = this._tensor.int32Data;
                }
                else if (this._tensor.rawData && this._tensor.rawData.length > 0) {
                    this._rawData = new DataView(this._tensor.rawData.buffer, this._tensor.rawData.byteOffset, this._tensor.rawData.byteLength);
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.UINT32:
                if (this._tensor.uint64Data && this._tensor.uint64Data.length > 0) {
                    this._data = this._tensor.uint64Data;
                }
                else if (this._tensor.rawData && this._tensor.rawData.length > 0) {
                    this._rawData = this._tensor.rawData;
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.INT64:
                if (this._tensor.int64Data && this._tensor.int64Data.length > 0) {
                    this._data = this._tensor.int64Data;
                }
                else if (this._tensor.rawData && this._tensor.rawData.length > 0) {
                    this._rawData = this._tensor.rawData;
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.UINT64:
                if (this._tensor.uint64Data && this._tensor.uint64Data.length > 0) {
                    this._data = this._tensor.uint64Data;
                }
                else if (this._tensor.rawData && this._tensor.rawData.length > 0) {
                    this._rawData = this._tensor.rawData;
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            default:
                // debugger;
                return 'Tensor data type is not implemented.';
        }

        this._index = 0;
        this._count = 0;
        var result = this.read(0);
        delete this._index;
        delete this._count;
        delete this._data;
        delete this._rawData;

        return JSON.stringify(result, null, 4);
    }

    read(dimension) {
        var results = [];
        var size = this._tensor.dims[dimension];
        if (dimension == this._tensor.dims.length - 1) {
            for (var i = 0; i < size; i++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                if (this._data) {
                    results.push(this._data[this._index++]);
                    this._count++;
                }
                else if (this._rawData) {
                    switch (this._tensor.dataType)
                    {
                        case onnx.TensorProto.DataType.FLOAT:
                            results.push(this._rawData.getFloat32(this._index, true));
                            this._index += 4;
                            this._count++;
                            break;
                        case onnx.TensorProto.DataType.DOUBLE:
                            results.push(this._rawData.getFloat64(this._index, true));
                            this._index += 8;
                            this._count++;
                            break;
                        case onnx.TensorProto.DataType.INT32:
                            results.push(this._rawData.getInt32(this._index, true));
                            this._index += 4;
                            this._count++;
                            break;
                        case onnx.TensorProto.DataType.UINT32:
                            results.push(this._rawData.getUint32(this._index, true));
                            this._index += 4;
                            this._count++;
                            break;
                        case onnx.TensorProto.DataType.INT64:
                            results.push(new Int64(this._rawData.subarray(this._index, 8)));
                            this._index += 8;
                            this._count++;
                            break;
                        case onnx.TensorProto.DataType.UINT64:
                            results.push(new Uint64(this._rawData.subarray(this._index, 8)));
                            this._index += 8;
                            this._count++;
                            break;
                    }
                }
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                results.push(this.read(dimension + 1));
            }
        }
        return results;
    }

    static formatTensorType(tensor) {
        var result = '';
        if (tensor.hasOwnProperty('dataType')) {
            result = OnnxTensor.formatElementType(tensor.dataType);
            if (tensor.dims) { 
                result += '[' + tensor.dims.map(dimension => dimension.toString()).join(',') + ']';
            }
        }
        return result;
    }

    static formatElementType(elementType) {
        if (!OnnxTensor._elementTypeMap) {
            var map = {};
            OnnxTensor._elementTypeMap = map;
            map[onnx.TensorProto.DataType.UNDEFINED] = 'UNDEFINED';
            map[onnx.TensorProto.DataType.FLOAT] = 'float';
            map[onnx.TensorProto.DataType.UINT8] = 'uint8';
            map[onnx.TensorProto.DataType.INT8] = 'int8';
            map[onnx.TensorProto.DataType.UINT16] = 'uint16';
            map[onnx.TensorProto.DataType.INT16] = 'int16';
            map[onnx.TensorProto.DataType.INT32] = 'int32';
            map[onnx.TensorProto.DataType.INT64] = 'int64';
            map[onnx.TensorProto.DataType.STRING] = 'string';
            map[onnx.TensorProto.DataType.BOOL] = 'bool';
            map[onnx.TensorProto.DataType.FLOAT16] = 'float16';
            map[onnx.TensorProto.DataType.DOUBLE] = 'double';
            map[onnx.TensorProto.DataType.UINT32] = 'uint32';
            map[onnx.TensorProto.DataType.UINT64] = 'uint64';
            map[onnx.TensorProto.DataType.COMPLEX64] = 'complex64';
            map[onnx.TensorProto.DataType.COMPLEX128] = 'complex128';    
        }
        var name = OnnxTensor._elementTypeMap[elementType];
        if (name) {
            return name;
        }
        return OnnxTensor._elementTypeMap[onnx.TensorProto.DataType.UNDEFINED];
    }

    static formatType(type) {
        if (type) {
            switch (type.value) {
                case 'tensorType':
                    var tensorType = type.tensorType;
                    var text = OnnxTensor.formatElementType(tensorType.elemType); 
                    if (tensorType.shape && tensorType.shape.dim) {
                        text += '[' + tensorType.shape.dim.map((dimension) => {
                            if (dimension.dimParam) {
                                return dimension.dimParam;
                            }
                            return dimension.dimValue.toString();
                        }).join(',') + ']';
                    }
                    return text;
                case 'mapType':
                    var mapType = type.mapType;
                    return '<' + OnnxTensor.formatElementType(mapType.keyType) + ',' + OnnxTensor.formatType(mapType.valueType) + '>';                    
                default:
                    // debugger;
                    return '?';
            }
        }
        // debugger;
        return '?';
    }
}

class OnnxGraphOperatorMetadata {

    constructor(model) {
        this._cache = {};
        this._imports = {};
        if (model.opsetImport) {
            model.opsetImport.forEach((opsetImport) => {
                var domain = opsetImport.domain || '';
                if (domain == 'ai.onnx') {
                    domain = '';
                }
                if (!this._imports[domain] || this._imports[domain] > opsetImport.version) {
                    this._imports[domain] = opsetImport.version;
                }
            });
        }
        if (Object.keys(this._imports).length == 0) {
            this._imports[''] = 1;
            this._imports['ai.onnx.ml'] = 1;
        }
    }

    getSchema(node) {
        var operator = node.operator;
        var schema = this._cache[operator];
        if (!schema) {
            schema = OnnxOperatorMetadata.operatorMetadata.getSchema(operator, this._imports);
            if (schema) {
                this._cache[operator] = schema;
            }
        }
        return schema;
    }

    getAttributeSchema(node, name) {
        var schema = this.getSchema(node);
        if (schema) {
            var attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    schema.attributes.forEach((attribute) => {
                        attributeMap[attribute.name] = attribute;
                    });
                }
                schema.attributeMap = attributeMap;
            }
            var attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema; 
            }
        }
        return null;
    }

    getInputs(node) {
        var inputs = [];
        var index = 0;
        var schema = this.getSchema(node);
        var data = node.data;
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputDef) => {
                if (index < data.input.length || inputDef.option != 'optional') {
                    var input = {};
                    input.name = inputDef.name;
                    input.type = inputDef.type;
                    var count = (inputDef.option == 'variadic') ? (data.input.length - index) : 1;
                    input.connections = [];
                    data.input.slice(index, index + count).forEach((id) => {
                        if (id != '' || inputDef.option != 'optional') {
                            input.connections.push({ id: id});
                        }
                    });
                    index += count;
                    inputs.push(input);
                }
            });
        }
        else {
            data.input.slice(index).forEach((input) => {
                inputs.push({
                    name: '(' + index.toString() + ')',
                    connections: [ { id: input } ]
                });
                index++;
            });

        }
        return inputs;
    }

    getOutputs(node) {
        var outputs = [];
        var index = 0;
        var schema = this.getSchema(node);
        var data = node.data;
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputDef) => {
                if (index < data.output.length || outputDef.option != 'optional') {
                    var output = {};
                    output.name = outputDef.name;
                    var count = (outputDef.option == 'variadic') ? (data.output.length - index) : 1;
                    output.connections = data.output.slice(index, index + count).map((id) => {
                        return { id: id };
                    });
                    index += count;
                    outputs.push(output);
                }
            });
        }
        else {
            data.output.slice(index).forEach((output) => {
                outputs.push({
                    name: '(' + index.toString() + ')',
                    connections: [ { id: output } ]
                });
                index++;
            });

        }
        return outputs;
    }

    getAttributeType(node, name) {
        var schema = this.getAttributeSchema(node, name);
        if (schema && schema.type) {
            return schema.type;
        }
        return '';
    }

    getAttributeVisible(node, attribute) {
        var schema = this.getAttributeSchema(node, attribute.name);
        if (schema && schema.hasOwnProperty('default') && schema.default) {
            if (attribute.value == schema.default.toString()) {
                return false;
            }
        }
        return true;     
    }

    getOperatorCategory(node) {
        var schema = this.getSchema(node);
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    getOperatorDocumentation(node) {
        var schema = this.getSchema(node);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = node.operator;
            if (schema.description) {
                var input = schema.description.split('\n');
                var output = [];
                var lines = [];
                var code = true;
                while (input.length > 0) {
                    var line = input.shift();
                    if (line.length > 0)
                    {
                        code = code && line.startsWith('  ');
                        lines.push(line + "\n");
                    }
                    if (line.length == 0 || input.length == 0) {
                        if (lines.length > 0) {
                            if (code) {
                                lines = lines.map((text) => { return text.substring(2); });
                                output.push('<pre>' + lines.join('') + '</pre>');
                            }
                            else {
                                var text = lines.join('');
                                text = this.markdown(text);
                                output.push('<p>' + text + '</p>');
                            }
                        }
                        lines = [];
                        code = true;
                    }
                }
                schema.description = output.join('');
            }
            var formatRange = (value) => {
                return (value == 2147483647) ? '&#8734;' : value.toString();
            };
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = this.markdown(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = this.markdown(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = this.markdown(output.description);
                    }
                });
            }
            if (schema.min_input != schema.max_input) {
                schema.inputs_range = formatRange(schema.min_input) + ' - ' + formatRange(schema.max_input);
            }
            if (schema.min_output != schema.max_output) {
                schema.outputs_range = formatRange(schema.min_output) + ' - ' + formatRange(schema.max_output);
            }
            if (schema.type_constraints) {
                schema.type_constraints.forEach((item) => {
                    if (item.allowed_type_strs) {
                        item.allowed_type_strs_display = item.allowed_type_strs.map((type) => { return type; }).join(', ');
                    }
                });
            }
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return "";
    }

    markdown(text) {
        text = text.replace(/\`\`(.*?)\`\`/gm, (match, content) => '<code>' + content + '</code>');
        text = text.replace(/\`(.*?)\`/gm, (match, content) => '<code>' + content + '</code>');
        return text;
    }
}

class OnnxOperatorMetadata {

    static open(host, callback) {
        if (OnnxOperatorMetadata.operatorMetadata) {
            callback(null, OnnxOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/onnx-metadata.json', (err, data) => {
                OnnxOperatorMetadata.operatorMetadata = new OnnxOperatorMetadata(data);
                callback(null, OnnxOperatorMetadata.operatorMetadata);
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
                        var name = item.name;
                        this._map[name] = this._map[name] || [];
                        this._map[name].push(item.schema);
                    }
                });
            }
        }
    }

    getSchema(operator, imports) {
        var result = null;
        var schemas = this._map[operator];
        if (schemas) {
            var version = -1;
            schemas.forEach((schema) => {
                var domain = schema.domain;
                if (domain == 'ai.onnx') {
                    domain = '';
                }
                var importVersion = imports[domain];
                var sinceVersion = schema.since_version;
                if (importVersion >= sinceVersion && version < sinceVersion) {
                    version = sinceVersion;
                    result = schema;
                }
            });
        }
        return result;
    }
}

class OnnxError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ONNX Error';
    }
}
