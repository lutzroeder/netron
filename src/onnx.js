/*jshint esversion: 6 */

var onnx = onnx || {};
var protobuf = protobuf || require('protobufjs');
var marked = marked || require('marked');
var base = base || require('./base');

onnx.ModelFactory = class {

    match(context, host) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'onnx') {
            return true;
        }
        var tags = null;
        if (extension == 'pb') {
            if (identifier.endsWith('saved_model.pb')) {
                return false;
            }
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
                return false;
            }
            tags = context.tags('pb');
            if (Object.keys(tags).length == 0) {
                return false;
            }
            // ignore input_0.pb, output_0.pb
            if (tags.hasOwnProperty(1) && tags[1] == 0 && 
                tags.hasOwnProperty(2) && tags[2] == 0 && 
                tags.hasOwnProperty(9) && tags[9] == 2) {
                return false;
            }
            // check ir_version and graph present
            if (!tags.hasOwnProperty(1) || tags[1] != 0 ||
                !tags.hasOwnProperty(7) || tags[7] != 2) {
                return false;
            }
            return true;
        }
        if (extension == 'pbtxt' || extension == 'prototxt') {
            if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return false;
            }
            tags = context.tags('pbtxt');
            if (tags.ir_version || tags.graph) {
                return true;
            }
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('./onnx-proto', (err, module) => {
            if (err) {
                callback(err, null);
                return;
            }
            var model = null;
            var identifier = context.identifier; 
            var extension = identifier.split('.').pop().toLowerCase();
            if (extension == 'pbtxt' || extension == 'prototxt') {
                try {
                    onnx.proto = protobuf.roots.onnx.onnx;
                    model = onnx.proto.ModelProto.decodeText(context.text);
                }
                catch (error) {
                    callback(new onnx.Error("File text format is not onnx.ModelProto (" + error.message + ") in '" + identifier + "'."), null);
                    return;
                }
            }
            else {
                try {
                    onnx.proto = protobuf.roots.onnx.onnx;
                    model = onnx.proto.ModelProto.decode(context.buffer);
                }
                catch (error) {
                    callback(new onnx.Error("File format is not onnx.ModelProto (" + error.message + ") in '" + identifier + "'."), null);
                    return;
                }
            }
            onnx.Metadata.open(host, (err, metadata) => {
                try {
                    var result = new onnx.Model(metadata, model);
                    callback(null, result);
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new onnx.Error(error.message), null);
                    return;
                }
            });
        });
    }
};

onnx.Model = class {

    constructor(metadata, model) {
        this._graphs = [];
        this._irVersion = model.ir_version;
        this._opsetImport = model.opset_import;
        this._producerName = model.producer_name;
        this._producerVersion = model.producer_version;
        this._domain = model.domain;
        this._modelVersion = model.model_version;
        this._description = model.doc_string;
        this._metadata = [];
        this._imageFormat = '';
        if (model.metadata_props)
        {
            var imageMetadata = {};
            model.metadata_props.forEach((metadata_prop) => {
                switch (metadata_prop.key) {
                    case 'author':
                        this._author = metadata_prop.value;
                        break;
                    case 'company':
                        this._company = metadata_prop.value;
                        break;
                    case 'converted_from':
                        this._converted_from = metadata_prop.value;
                        break;
                    case 'license':
                        this._license = metadata_prop.value;
                        break;
                    case 'license_url':
                        this._licenseUrl = metadata_prop.value;
                        break;
                    case 'Image.BitmapPixelFormat':
                    case 'Image.ColorSpaceGamma':
                    case 'Image.NominalPixelRange':
                        imageMetadata[metadata_prop.key] = metadata_prop.value;
                        break;
                    default:
                        this._metadata.push({ name: metadata_prop.key, value: metadata_prop.value});
                        break;
                }
            });
            this._imageFormat = [ imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange'] ].filter((item) => item);
        }
        this._graphs = [];
        if (model && model.graph) {
            var graphMetadata = new onnx.GraphMetadata(metadata, this._opsetImport);
            var graph = new onnx.Graph(graphMetadata, this._imageFormat, model.graph);
            this._graphs.push(graph);
        }
    }

    get format() {
        var format = 'ONNX';
        if (this._irVersion) {
            format = format + ' v' + this._irVersion.toString();
        }
        return format;
    }

    get imports() {
        if (this._opsetImport && this._opsetImport.length > 0) {
            var opsetImports = [];
            this._opsetImport.forEach((opsetImport) => {
                var domain = opsetImport.domain ? opsetImport.domain : 'ai.onnx';
                var result = domain + ' v' + opsetImport.version;
                if (!opsetImports.includes(result)) {
                    opsetImports.push(result);
                }
            });
            return opsetImports.join(', ');
        }
        return null;
    }

    get producer() {
        var producer = [];
        if (this._producerName) {
            producer.push(this._producerName);
        }
        if (this._producerVersion && this._producerVersion.length > 0) {
            producer.push(this._producerVersion);
        }
        if (producer.length > 0) {
            return producer.join(' ');
        }
        return null;
    }

    get domain() {
        return this._domain || null;        
    }

    get description() {
        return this._description || null;        
    }

    get author() {
        return this._author || null;   
    }

    get company() {
        return this._company || null;   
    }

    get source() {
        return this._converted_from || null;
    }

    get license() {
        var license = [];
        if (this._license && this._license.length > 0) {
            license.push(this._license);
        }
        if (this._licenseUrl && this._licenseUrl.length > 0) {
            license.push('<a href=\'' + this._licenseUrl + '\'>' + this._licenseUrl + '</a>');
        }
        if (license.length > 0) {
            return license;
        }
        return null;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

onnx.Graph = class {

    constructor(metadata, imageFormat, graph) {
        this._node = '';
        this._description = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._operators = {};

        if (graph) {
            this._name = graph.name || null;
            this._description = graph.doc_string || '';

            var initializers = {};
            graph.initializer.forEach((tensor) => {
                initializers[tensor.name] = new onnx.Tensor(tensor, 'Initializer');
            });
            var nodes = [];
            var outputCountMap = {};
            graph.node.forEach((node) => {
                this._operators[node.op_type] = (this._operators[node.op_type] || 0) + 1; 
                node.output.forEach((output) => {
                    outputCountMap[output] = (outputCountMap[output] || 0) + 1;
                });
            });
            graph.node.forEach((node) => {
                var initializerNode = false;
                if (node.op_type == 'Constant' && node.output && node.output.length == 1) {
                    var name = node.output[0];
                    if (outputCountMap[name] == 1) {
                        var attribute = node.attribute.find((attribute) => { return attribute.name == 'value' && attribute.t; }); 
                        if (attribute) {
                            initializers[name] = new onnx.Tensor(attribute.t, 'Constant');
                            initializerNode = true;
                        }
                    }
                }
                if (!initializerNode) {
                    nodes.push(node);
                }
            });

            var connections = {};
            graph.value_info.forEach((valueInfo) => {
                this._connection(connections, valueInfo.name, valueInfo.type, valueInfo.doc_string, initializers[valueInfo.name]);
            });
            graph.input.forEach((valueInfo) => {
                var connection = this._connection(connections, valueInfo.name, valueInfo.type, valueInfo.doc_string, initializers[valueInfo.name]);
                if (!initializers[valueInfo.name]) {
                    this._inputs.push(new onnx.Argument(valueInfo.name, [ connection ]));
                }
            });
            graph.output.map((valueInfo) => {
                var connection = this._connection(connections, valueInfo.name, valueInfo.type, valueInfo.doc_string, initializers[valueInfo.name]);
                this._outputs.push(new onnx.Argument(valueInfo.name, [ connection ]));
            });
    
            nodes.forEach((node) => {
                var inputs = [];
                if (node.input) {
                    inputs = metadata.getInputs(node.op_type, node.input);
                    inputs = inputs.map((input) => {
                        return new onnx.Argument(input.name, input.connections.map((connection) => {
                            return this._connection(connections, connection.id, null, null, initializers[connection.id]);
                        }));
                    });
                }
                var outputs = [];
                if (node.output) {
                    outputs = metadata.getOutputs(node.op_type, node.output);
                    outputs = outputs.map((output) => {
                        return new onnx.Argument(output.name, output.connections.map((connection) => {
                            return this._connection(connections, connection.id, null, null, initializers[connection.id]);
                        }));
                    });
                }
                this._nodes.push(new onnx.Node(metadata, imageFormat, node.op_type, node.domain, node.name, node.doc_string, node.attribute, inputs, outputs));
            });
        }
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get operators() {
        return this._operators;
    }

    get groups() {
        return false;
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

    get metadata() {
        return this._metadata;
    }

    toString() {
        return 'graph(' + this.name + ')';
    }

    _connection(connections, id, type, doc_string, initializer) {
        var connection = connections[id];
        if (!connection) {
            connection = new onnx.Connection(id, type ? onnx.Tensor._formatType(type, this._imageFormat) : null, doc_string, initializer);
            connections[id] = connection;
        }
        return connection;
    }
};

onnx.Argument = class {
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

onnx.Connection = class {

    constructor(id, type, description, initializer) {
        this._id = id;
        this._type = type || null;
        this._description = description || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._type) {
            return this._type;
        }
        if (this._initializer) {
            return this._initializer.type;
        }
        return null;
    }

    get description() {
        return this._description;
    }

    get initializer() {
        return this._initializer;
    }
};

onnx.Node = class {

    constructor(metadata, imageFormat, operator, domain, name, description, attributes, inputs, outputs) {
        this._metadata = metadata;
        this._operator = operator;
        if (domain) {
            this._domain = domain;
        }
        if (name) {
            this._name = name;
        }
        if (description) {
            this._description = description;
        }
        this._attributes = [];
        if (attributes && attributes.length > 0) {
            attributes.forEach((attribute) => { 
                this._attributes.push(new onnx.Attribute(this._metadata, imageFormat, this.operator, attribute));
            });
        }            
        this._inputs = inputs;
        this._outputs = outputs;
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name || null;
    }

    get description() {
        return this._description || null;
    }

    get documentation() {
        var schema = this._metadata.getSchema(this._operator);
        if (schema) {
            var options = { baseUrl: 'https://github.com/onnx/onnx/blob/master/docs/' };
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description, options);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description, options);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description, options);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description, options);
                    }
                });
            }
            var formatRange = (value) => {
                return (value == 2147483647) ? '&#8734;' : value.toString();
            };
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
            return schema;
        }
        return null;
    }

    get domain() {
        return this._domain || null;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : null;
    }

    get group() {
        return null;
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

    get dependencies() {
        return [];
    }
};

onnx.Attribute = class {

    constructor(metadata, imageFormat, operator, attribute) {
        this._name = attribute.name;
        this._type = null;
        this._value = null;

        if (attribute.doc_string) {
            this._description = this._attribute.doc_string;
        }

        if (attribute.ints && attribute.ints.length > 0) {
            if (attribute.ints.length > 65536) {
                this._value = () => '[...]';
            }
            else {
                this._value = attribute.ints; 
            }
        }
        else if (attribute.floats && attribute.floats.length > 0) {
            if (attribute.floats.length > 65536) {
                this._value = () => '[...]';
            }
            else {
                this._value = attribute.floats;
            }
        }
        else if (attribute.strings && attribute.strings.length > 0) {
            if (attribute.strings.length > 65536) {
                this._value = () => '[...]';
            }
            else {
                this._value = attribute.strings.map((s) => {
                    if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                        return String.fromCharCode.apply(null, s);
                    }
                    else {
                        return s.map(v => v.toString()).join(', ');
                    }
                });
            }
        }
        else if (attribute.graphs && attribute.graphs.length > 0) {
            this._value = arg.graphs.map((graph) => new onnx.Graph(metadata, imageFormat, graph));
            this._type = 'graph[]';
        }
        else if (attribute.s && attribute.s.length > 0) {
            if (attribute.s.filter(c => c <= 32 && c >= 128).length == 0) {
                this._value = String.fromCharCode.apply(null, attribute.s);
            }
            else {
                this._value = attribute.s;
            }
        }
        else if (attribute.hasOwnProperty('f')) {
            this._value = attribute.f;
        }
        else if (attribute.hasOwnProperty('i')) {
            this._value = attribute.i;
        }
        else if (attribute.hasOwnProperty('t')) {
            this._type = 'tensor';
            this._value = new onnx.Tensor(attribute.t).value;
        }
        else if (attribute.hasOwnProperty('g')) {
            this._type = 'graph';
            this._value = new onnx.Graph(metadata, imageFormat, attribute.g);
        }

        var attributeSchema = metadata.getAttributeSchema(operator, attribute.name);

        if (!this._type) {
            if (attribute.hasOwnProperty('type')) {
                if (!onnx.Attribute._attributeTypeMap) {
                    var map = {};
                    map[onnx.proto.AttributeProto.AttributeType.UNDEFINED] = 'undefined';
                    map[onnx.proto.AttributeProto.AttributeType.FLOAT] = 'float';
                    map[onnx.proto.AttributeProto.AttributeType.INT] = 'int';
                    map[onnx.proto.AttributeProto.AttributeType.STRING] = 'string';
                    map[onnx.proto.AttributeProto.AttributeType.TENSOR] = 'tensor';
                    map[onnx.proto.AttributeProto.AttributeType.GRAPH] = 'graph';
                    map[onnx.proto.AttributeProto.AttributeType.FLOATS] = 'float';
                    map[onnx.proto.AttributeProto.AttributeType.INTS] = 'int[]';
                    map[onnx.proto.AttributeProto.AttributeType.STRINGS] = 'string[]';
                    map[onnx.proto.AttributeProto.AttributeType.TENSORS] = 'tensor[]';
                    map[onnx.proto.AttributeProto.AttributeType.GRAPHS] = 'graph[]';
                    onnx.Attribute._attributeTypeMap = map;
                }
                var attributeType = onnx.Attribute._attributeTypeMap[attribute.type];
                this._type = attributeType || onnx.Attribute._attributeTypeMap[onnx.proto.AttributeProto.AttributeType.UNDEFINED];
            }
            else if (attributeSchema && attributeSchema.type) {
                this._type = attributeSchema.type;
            }
        }

        if (attributeSchema && attributeSchema.hasOwnProperty('default') && attributeSchema.default) {
            if (this._value == attributeSchema.default) {
                this._visible = false;
            }
        }
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

    get description() {
        return this._description || null;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

onnx.Tensor = class {

    constructor(tensor, kind) {
        this._tensor = tensor;
        this._name = tensor.name || '';
        this._kind = kind || null;
        this._type = new onnx.TensorType(this._tensor.data_type, new onnx.TensorShape(this._tensor.dims.map((dim) => dim)), null);

        if (this._tensor.data_type == onnx.proto.TensorProto.DataType.FLOAT16 && this._tensor.int32_data && this._tensor.int32_data.length > 0) {
            var array = new Uint8Array(this._tensor.int32_data.length << 1);
            var dataView = new DataView(array.buffer, array.byteOffset, array.byteLength);
            this._tensor.int32_data.forEach((value, index) => {
                dataView.setUint16(index << 1, value, true);
            });
            this._tensor.raw_data = array;
            this._tensor.int32_data = undefined;
        }
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
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
        return onnx.Tensor._stringify(value, '', '    ');
    }

    _context() {
        var context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (!this._tensor.data_type) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        if (!this._tensor.dims) {
            context.state =  'Tensor has no dimensions.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;

        switch (this._tensor.data_type) {
            case onnx.proto.TensorProto.DataType.FLOAT:
                if (this._tensor.float_data && this._tensor.float_data.length > 0) {
                    context.data = this._tensor.float_data;
                }
                else if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case onnx.proto.TensorProto.DataType.DOUBLE:
                if (this._tensor.double_data && this._tensor.double_data.length > 0) {
                    context.data = this._tensor.double_data;
                }
                else if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case onnx.proto.TensorProto.DataType.FLOAT16:
                if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case onnx.proto.TensorProto.DataType.INT8:
            case onnx.proto.TensorProto.DataType.UINT8:
            case onnx.proto.TensorProto.DataType.INT16:
            case onnx.proto.TensorProto.DataType.UINT16:
            case onnx.proto.TensorProto.DataType.INT32:
                if (this._tensor.int32_data && this._tensor.int32_data.length > 0) {
                    context.data = this._tensor.int32_data;
                }
                else if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case onnx.proto.TensorProto.DataType.UINT32:
                if (this._tensor.uint64_data && this._tensor.uint64_data.length > 0) {
                    context.data = this._tensor.uint64_data;
                }
                else if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = this._tensor.raw_data;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case onnx.proto.TensorProto.DataType.INT64:
                if (this._tensor.int64_data && this._tensor.int64_data.length > 0) {
                    context.data = this._tensor.int64_data;
                }
                else if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = this._tensor.raw_data;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case onnx.proto.TensorProto.DataType.UINT64:
                if (this._tensor.uint64_data && this._tensor.uint64_data.length > 0) {
                    context.data = this._tensor.uint64_data;
                }
                else if (this._tensor.raw_data && this._tensor.raw_data.length > 0) {
                    context.rawData = this._tensor.raw_data;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            default:
                // debugger;
                context.state = 'Tensor data type is not implemented.';
                break;
        }
        return context;
    }

    _decode(context, dimension) {
        var shape = context.shape;
        if (context.shape.length == 0) {
            shape = [ 1 ];
        }
        var results = [];
        var size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.data) {
                    results.push(context.data[context.index++]);
                    context.count++;
                }
                else if (context.rawData) {
                    switch (this._tensor.data_type)
                    {
                        case onnx.proto.TensorProto.DataType.FLOAT16:
                            results.push(context.rawData.getFloat16(context.index, true));
                            context.index += 2;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.FLOAT:
                            results.push(context.rawData.getFloat32(context.index, true));
                            context.index += 4;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.DOUBLE:
                            results.push(context.rawData.getFloat64(context.index, true));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT8:
                            results.push(context.rawData.getInt8(context.index, true));
                            context.index++;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT8:
                            results.push(context.rawData.getUint8(context.index, true));
                            context.index++;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT16:
                            results.push(context.rawData.getInt16(context.index, true));
                            context.index += 2;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT16:
                            results.push(context.rawData.getUint16(context.index, true));
                            context.index += 2;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT32:
                            results.push(context.rawData.getInt32(context.index, true));
                            context.index += 4;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT32:
                            results.push(context.rawData.getUint32(context.index, true));
                            context.index += 4;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT64:
                            results.push(new base.Int64(context.rawData.subarray(context.index, context.index + 8)));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT64:
                            results.push(new base.Uint64(context.rawData.subarray(context.index, context.index + 8)));
                            context.index += 8;
                            context.count++;
                            break;
                    }
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
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push(indentation + '[');
            var items = value.map((item) => onnx.Tensor._stringify(item, indentation + indent, indent));
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

    static _formatElementType(elementType) {
        if (!onnx.Tensor._elementTypeMap) {
            var map = {};
            map[onnx.proto.TensorProto.DataType.UNDEFINED] = 'UNDEFINED';
            map[onnx.proto.TensorProto.DataType.FLOAT] = 'float32';
            map[onnx.proto.TensorProto.DataType.UINT8] = 'uint8';
            map[onnx.proto.TensorProto.DataType.INT8] = 'int8';
            map[onnx.proto.TensorProto.DataType.UINT16] = 'uint16';
            map[onnx.proto.TensorProto.DataType.INT16] = 'int16';
            map[onnx.proto.TensorProto.DataType.INT32] = 'int32';
            map[onnx.proto.TensorProto.DataType.INT64] = 'int64';
            map[onnx.proto.TensorProto.DataType.STRING] = 'string';
            map[onnx.proto.TensorProto.DataType.BOOL] = 'bool';
            map[onnx.proto.TensorProto.DataType.FLOAT16] = 'float16';
            map[onnx.proto.TensorProto.DataType.DOUBLE] = 'float64';
            map[onnx.proto.TensorProto.DataType.UINT32] = 'uint32';
            map[onnx.proto.TensorProto.DataType.UINT64] = 'uint64';
            map[onnx.proto.TensorProto.DataType.COMPLEX64] = 'complex64';
            map[onnx.proto.TensorProto.DataType.COMPLEX128] = 'complex128';    
            map[onnx.proto.TensorProto.DataType.BFLOAT16] = 'bfloat16';
            onnx.Tensor._elementTypeMap = map;
        }
        var name = onnx.Tensor._elementTypeMap[elementType];
        if (name) {
            return name;
        }
        return onnx.Tensor._elementTypeMap[onnx.proto.TensorProto.DataType.UNDEFINED];
    }

    static _formatType(type, imageFormat) {
        if (!type) {
            return null;
        }
        var denotation = '';
        switch (type.denotation) {
            case 'TENSOR':
                denotation = 'Tensor';
                break;
            case 'IMAGE':
                denotation = 'Image' + (imageFormat ? '(' + imageFormat.join(',') + ')' : '');
                break;
            case 'AUDIO':
                denotation = 'Audio';
                break;
            case 'TEXT':
                denotation = 'Text';
                break;
        }
        switch (type.value) {
            case 'tensor_type':
            case 'sparse_tensor_type':
                var shape = [];
                if (type.tensor_type.shape && type.tensor_type.shape.dim) {
                    shape = type.tensor_type.shape.dim.map((dim) => {
                        return dim.dim_param ? dim.dim_param : dim.dim_value;
                    });
                }
                return new onnx.TensorType(type.tensor_type.elem_type, new onnx.TensorShape(shape), denotation);
            case 'map_type':
                return new onnx.MapType(type.map_type.key_type, onnx.Tensor._formatType(type.map_type.value_type, imageFormat), denotation);
            case 'sequence_type':
                return new onnx.SequenceType(onnx.Tensor._formatType(type.sequence_type.elem_type, imageFormat), denotation);
            case 'opaque_type':
                return new onnx.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
        }
        return null;
    }
};

onnx.TensorType = class {

    constructor(dataType, shape, denotation) {
        this._dataType = onnx.Tensor._formatElementType(dataType);
        this._shape = shape;
        this._denotation = denotation || null;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get denotation() { 
        return this._denotation;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

onnx.TensorShape = class {

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

onnx.SequenceType = class {

    constructor(elementType, denotation) {
        this._elementType = elementType;
        this._denotation = denotation;
    }

    get elementType() {
        return this._elementType;
    }

    get dennotation() {
        return this._dennotation;
    }

    toString() {
        return 'sequence<' + this._elementType.toString() + '>';
    }
};

onnx.MapType = class {

    constructor(keyType, valueType, denotation) {
        this._keyType = onnx.Tensor._formatElementType(keyType);
        this._valueType = valueType;
        this._denotation = denotation;
    }

    get keyType() {
        return this._keyType;
    }

    get valueType() {
        return this._valueType;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return 'map<' + this._keyType + ',' + this._valueType.toString() + '>';
    }
};

onnx.OpaqueType = class {

    constructor(domain, name, parameters) {
        this._domain = domain;
        this._name = name;
    }

    toString() {
        var name = (this._domain ? (this._domain + '.') : '') + this._name;
        return 'opaque<' + name + '>';
    }
};

onnx.GraphMetadata = class {

    constructor(metadata, opsetImport) {
        this._metadata = metadata;
        this._cache = {};
        this._imports = {};
        if (opsetImport) {
            opsetImport.forEach((opsetImport) => {
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

    getSchema(operator) {
        var schema = this._cache[operator];
        if (!schema) {
            schema = this._metadata.getSchema(operator, this._imports);
            if (schema) {
                this._cache[operator] = schema;
            }
        }
        return schema;
    }

    getAttributeSchema(operator, name) {
        var schema = this.getSchema(operator);
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

    getInputs(operator, inputs) {
        var results = [];
        var index = 0;
        var schema = this.getSchema(operator);
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputDef) => {
                if (index < inputs.length || inputDef.option != 'optional') {
                    var input = {};
                    input.name = inputDef.name;
                    input.type = inputDef.type;
                    var count = (inputDef.option == 'variadic') ? (inputs.length - index) : 1;
                    input.connections = [];
                    inputs.slice(index, index + count).forEach((id) => {
                        if (id != '' || inputDef.option != 'optional') {
                            input.connections.push({ id: id});
                        }
                    });
                    index += count;
                    results.push(input);
                }
            });
        }
        else {
            inputs.slice(index).forEach((input) => {
                results.push({
                    name: index.toString(),
                    connections: [ { id: input } ]
                });
                index++;
            });

        }
        return results;
    }

    getOutputs(operator, outputs) {
        var results = [];
        var index = 0;
        var schema = this.getSchema(operator);
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputDef) => {
                if (index < outputs.length || outputDef.option != 'optional') {
                    var output = {};
                    output.name = outputDef.name;
                    var count = (outputDef.option == 'variadic') ? (outputs.length - index) : 1;
                    output.connections = outputs.slice(index, index + count).map((id) => {
                        return { id: id };
                    });
                    index += count;
                    results.push(output);
                }
            });
        }
        else {
            outputs.slice(index).forEach((output) => {
                results.push({
                    name: index.toString(),
                    connections: [ { id: output } ]
                });
                index++;
            });

        }
        return results;
    }
};

onnx.Metadata = class {

    static open(host, callback) {
        if (onnx.Metadata._metadata) {
            callback(null, onnx.Metadata._metadata);
        }
        else {
            host.request(null, 'onnx-metadata.json', 'utf-8', (err, data) => {
                onnx.Metadata._metadata = new onnx.Metadata(data);
                callback(null, onnx.Metadata._metadata);
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
};

onnx.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading ONNX model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = onnx.ModelFactory;   
}
