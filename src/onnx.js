/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var onnx = onnx || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');
var marked = marked || require('marked');

onnx.ModelFactory = class {

    match(context) {
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
            if (tags.size === 0) {
                return false;
            }
            // ignore input_0.pb, output_0.pb
            if (tags.size > 0 &&
                tags.has(1) && tags.get(1) === 0 && 
                tags.has(2) && tags.get(2) === 0 && 
                tags.has(9) && tags.get(9) === 2) {
                return false;
            }
            if (tags.size > 0 &&
                Array.from(tags.values()).some((v) => v === 5)) {
                return false;
            }
            // check ir_version and graph present
            if (tags.has(1) && tags.get(1) != 0 ||
                tags.has(2) && tags.get(2) != 2 ||
                tags.has(3) && tags.get(3) != 2 ||
                tags.has(4) && tags.get(4) != 2 ||
                tags.has(5) && tags.get(5) != 0 ||
                tags.has(6) && tags.get(6) != 2 ||
                tags.has(8) && tags.get(8) != 2 ||
                tags.has(14) && tags.get(14) != 2 ||
                (!tags.has(7) || tags.get(7) != 2)) {
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
            if (tags.has('ir_version') || tags.has('graph')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) { 
        return host.require('./onnx-proto').then(() => {
            var model = null;
            var identifier = context.identifier; 
            var extension = identifier.split('.').pop().toLowerCase();
            if (extension == 'pbtxt' || extension == 'prototxt') {
                try {
                    onnx.proto = protobuf.roots.onnx.onnx;
                    var reader = prototxt.TextReader.create(context.text);
                    model = onnx.proto.ModelProto.decodeText(reader);
                }
                catch (error) {
                    throw new onnx.Error("File text format is not onnx.ModelProto (" + error.message + ") in '" + identifier + "'.");
                }
            }
            else {
                try {
                    onnx.proto = protobuf.roots.onnx.onnx;
                    model = onnx.proto.ModelProto.decode(context.buffer);
                }
                catch (error) {
                    throw  new onnx.Error("File format is not onnx.ModelProto (" + error.message + ") in '" + identifier + "'.");
                }
            }
            return onnx.Metadata.open(host).then((metadata) => {
                try {
                    return new onnx.Model(metadata, model);
                }
                catch (error) {
                    host.exception(error, false);
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new onnx.Error(message + " in '" + identifier + "'.");
                }
            });
        });
    }
};

onnx.Model = class {

    constructor(metadata, model) {
        this._graphs = [];
        this._irVersion = model.ir_version;
        this._producerName = model.producer_name;
        this._producerVersion = model.producer_version;
        this._domain = model.domain;
        this._modelVersion = model.model_version;
        this._description = model.doc_string;
        this._metadata = [];
        this._imageFormat = '';
        this._imports = null;

        var imports = {};
        if (model.opset_import && model.opset_import.length > 0) {
            var results = [];
            for (var opset_import of model.opset_import) {
                var domain = opset_import.domain || 'ai.onnx';
                var result = domain + ' v' + opset_import.version;
                if (!results.includes(result)) {
                    results.push(result);
                }
                domain = domain == 'ai.onnx' ? '' : domain;
                if (!imports[domain] || imports[domain] > opset_import.version) {
                    imports[domain] = opset_import.version;
                }
            }
            this._imports = results.join(', ');
        }
        if (Object.keys(imports).length == 0) {
            imports[''] = 1;
            imports['ai.onnx.ml'] = 1;
        }

        if (model.metadata_props)
        {
            var imageMetadata = {};
            for (var metadata_prop of model.metadata_props) {
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
            }
            this._imageFormat = [ imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange'] ].filter((item) => item);
        }
        this._graphs = [];
        if (model && model.graph) {
            var graphMetadata = new onnx.GraphMetadata(metadata, imports);
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
        return this._imports;
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
        this._imageFormat = imageFormat;

        if (graph) {
            this._name = graph.name || null;
            this._description = graph.doc_string || '';

            var initializers = {};
            for (var tensor of graph.initializer) {
                initializers[tensor.name] = new onnx.Tensor(tensor, 'Initializer');
            }
            var nodes = [];
            var outputCountMap = {};
            var inputCountMap = {};
            var node;
            var input;
            var output;
            for (node of graph.node) {
                for (input of node.input) {
                    inputCountMap[input] = (inputCountMap[input] || 0) + 1;
                }
                for (output of node.output) {
                    outputCountMap[output] = (outputCountMap[output] || 0) + 1;
                }
            }
            for (input of graph.input) {
                delete inputCountMap[input];
            }
            for (output of graph.output) {
                delete outputCountMap[output];
            }
            for (node of graph.node) {
                var initializerNode = false;
                if (node.op_type == 'Constant' && node.input.length == 0 && node.output.length == 1) {
                    var name = node.output[0];
                    if (inputCountMap[name] == 1 && outputCountMap[name] == 1 && node.attribute.length == 1) {
                        var attribute = node.attribute[0];
                        if (attribute && attribute.name == 'value' && attribute.t) {
                            initializers[name] = new onnx.Tensor(attribute.t, 'Constant');
                            initializerNode = true;
                        }
                    }
                }
                if (!initializerNode) {
                    nodes.push(node);
                }
            }

            this._arguments = {};
            var valueInfo;
            for (valueInfo of graph.value_info) {
                this._argument(valueInfo.name, valueInfo.type, valueInfo.doc_string, initializers[valueInfo.name]);
            }
            var argument;
            for (valueInfo of graph.input) {
                argument = this._argument(valueInfo.name, valueInfo.type, valueInfo.doc_string, initializers[valueInfo.name]);
                if (!initializers[valueInfo.name]) {
                    this._inputs.push(new onnx.Parameter(valueInfo.name, [ argument ]));
                }
            }
            for (valueInfo of graph.output) {
                argument = this._argument(valueInfo.name, valueInfo.type, valueInfo.doc_string, initializers[valueInfo.name]);
                this._outputs.push(new onnx.Parameter(valueInfo.name, [ argument ]));
            }
            for (node of nodes) {
                var inputs = [];
                var schema = metadata.getSchema(node.op_type);
                if (node.input && node.input.length > 0) {
                    var inputIndex = 0;
                    if (schema && schema.inputs) {
                        for (var inputSchema of schema.inputs) {
                            if (inputIndex < node.input.length || inputSchema.option != 'optional') {
                                var inputCount = (inputSchema.option == 'variadic') ? (node.input.length - inputIndex) : 1;
                                var inputArguments = node.input.slice(inputIndex, inputIndex + inputCount).map((id) => {
                                    return this._argument(id, null, null, initializers[id]);
                                });
                                inputIndex += inputCount;
                                inputs.push(new onnx.Parameter(inputSchema.name, inputArguments));
                            }
                        }
                    }
                    else {
                        inputs = inputs.concat(node.input.slice(inputIndex).map((id, index) => {
                            return new onnx.Parameter((inputIndex + index).toString(), [
                                this._argument(id, null, null)
                            ])
                        }));
                    }
                }
                var outputs = [];
                if (node.output && node.output.length > 0) {
                    var outputIndex = 0;
                    if (schema && schema.outputs) {
                        for (var outputSchema of schema.outputs) {
                            if (outputIndex < node.output.length || outputSchema.option != 'optional') {
                                var outputCount = (outputSchema.option == 'variadic') ? (node.output.length - outputIndex) : 1;
                                var outputArguments = node.output.slice(outputIndex, outputIndex + outputCount).map((id) => {
                                    return this._argument(id, null, null, null);
                                });
                                outputIndex += outputCount;
                                outputs.push(new onnx.Parameter(outputSchema.name, outputArguments));
                            }
                        }
                    }
                    else {
                        outputs = outputs.concat(node.output.slice(outputIndex).map((id, index) => {
                            return new onnx.Parameter((outputIndex + index).toString(), [
                                this._argument(id, null, null)
                            ]);
                        }));
                    }
                }
                this._nodes.push(new onnx.Node(metadata, imageFormat, node.op_type, node.domain, node.name, node.doc_string, node.attribute, inputs, outputs));
            }
        }

        delete this._arguments;
        delete this._imageFormat;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
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

    toString() {
        return 'graph(' + this.name + ')';
    }

    _argument(id, type, doc_string, initializer) {
        var argument = this._arguments[id];
        if (!argument) {
            argument = new onnx.Argument(id, type ? onnx.Tensor._formatType(type, this._imageFormat) : null, doc_string, initializer);
            this._arguments[id] = argument;
        }
        return argument;
    }
};

onnx.Parameter = class {

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

onnx.Argument = class {

    constructor(id, type, description, initializer) {
        this._id = id;
        this._type = type || null;
        this._description = description || '';
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
        this._domain = domain || '';
        this._name = name || '';
        this._description = description || '';
        this._attributes = [];
        if (attributes && attributes.length > 0) {
            for (var attribute of attributes) {
                this._attributes.push(new onnx.Attribute(this._metadata, imageFormat, this.operator, attribute));
            }
        }            
        this._inputs = inputs;
        this._outputs = outputs;
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
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
                for (var attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description, options);
                    }
                }
            }
            if (schema.inputs) {
                for (var input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description, options);
                    }
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description, options);
                    }
                }
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
                for (var type_constraint of schema.type_constraints) {
                    if (type_constraint.allowed_type_strs) {
                        type_constraint.allowed_type_strs_display = type_constraint.allowed_type_strs.map((type) => { return type; }).join(', ');
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get domain() {
        return this._domain;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
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
};

onnx.Attribute = class {

    constructor(metadata, imageFormat, operator, attribute) {
        this._name = attribute.name;
        this._description = attribute.doc_string || '';
        this._type = null;
        this._value = null;

        if (attribute.ints && attribute.ints.length > 0) {
            this._value = attribute.ints; 
        }
        else if (attribute.floats && attribute.floats.length > 0) {
            this._value = attribute.floats;
        }
        else if (attribute.strings && attribute.strings.length > 0) {
            this._value = attribute.strings.map((s) => {
                if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                    return String.fromCharCode.apply(null, s);
                }
                else {
                    return s.map(v => v.toString()).join(', ');
                }
            });
        }
        else if (attribute.graphs && attribute.graphs.length > 0) {
            this._value = attribute.graphs.map((graph) => new onnx.Graph(metadata, imageFormat, graph));
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
        else if (Object.prototype.hasOwnProperty.call(attribute, 'f')) {
            this._value = attribute.f;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'i')) {
            this._value = attribute.i;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 't')) {
            this._type = 'tensor';
            this._value = new onnx.Tensor(attribute.t).value;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'g')) {
            this._type = 'graph';
            this._value = new onnx.Graph(metadata, imageFormat, attribute.g);
        }

        var attributeSchema = metadata.getAttributeSchema(operator, attribute.name);

        if (!this._type) {
            if (Object.prototype.hasOwnProperty.call(attribute, 'type')) {
                if (!onnx.Attribute._attributeTypeMap) {
                    var map = {};
                    map[onnx.proto.AttributeProto.AttributeType.UNDEFINED] = 'undefined';
                    map[onnx.proto.AttributeProto.AttributeType.FLOAT] = 'float32';
                    map[onnx.proto.AttributeProto.AttributeType.INT] = 'int64';
                    map[onnx.proto.AttributeProto.AttributeType.STRING] = 'string';
                    map[onnx.proto.AttributeProto.AttributeType.TENSOR] = 'tensor';
                    map[onnx.proto.AttributeProto.AttributeType.GRAPH] = 'graph';
                    map[onnx.proto.AttributeProto.AttributeType.FLOATS] = 'float32';
                    map[onnx.proto.AttributeProto.AttributeType.INTS] = 'int64[]';
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

        if (attributeSchema && Object.prototype.hasOwnProperty.call(attributeSchema, 'default') && attributeSchema.default) {
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
        return this._description;
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
            var data = this._tensor.int32_data;
            for (var i = 0; i < data.length; i++) {
                dataView.setUint16(i << 1, data[i], true);
            }
            this._tensor.raw_data = array;
            delete this._tensor.int32_data;
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

        if (this._tensor.data_location === onnx.proto.TensorProto.DataLocation.EXTERNAL) {
            context.state =  'External data not implemented.';
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
            case onnx.proto.TensorProto.DataType.BOOL:
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
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
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
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
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
                    context.rawData = new DataView(this._tensor.raw_data.buffer, this._tensor.raw_data.byteOffset, this._tensor.raw_data.byteLength);
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            default:
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
                    var value = context.data[context.index++];
                    switch (this._tensor.data_type)
                    {   
                        case onnx.proto.TensorProto.DataType.BOOL:
                            value = value === 0 ? false : true;
                            break;
                    }
                    results.push(value);
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
                            results.push(new long.Long(context.rawData.getUint32(context.index, true), context.rawData.getUint32(context.index + 4, true), false));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT64:
                            results.push(new long.Long(context.rawData.getUint32(context.index, true), context.rawData.getUint32(context.index + 4, true), true));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.BOOL:
                            results.push(context.rawData.getInt8(context.index, true) === 0 ? false : true);
                            context.index += 1;
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

    constructor(domain, name) {
        this._domain = domain;
        this._name = name;
    }

    toString() {
        var name = (this._domain ? (this._domain + '.') : '') + this._name;
        return 'opaque<' + name + '>';
    }
};

onnx.GraphMetadata = class {

    constructor(metadata, imports) {
        this._metadata = metadata;
        this._cache = {};
        this._attributeCache = {};
        this._imports = imports;
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
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

onnx.Metadata = class {

    static open(host) {
        if (onnx.Metadata._metadata) {
            return Promise.resolve(onnx.Metadata._metadata);
        }
        return host.request(null, 'onnx-metadata.json', 'utf-8').then((data) => {
            onnx.Metadata._metadata = new onnx.Metadata(data);
            return onnx.Metadata._metadata;
        }).catch(() => {
            onnx.Metadata._metadata = new onnx.Metadata(null);
            return onnx.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        var name = item.name;
                        this._map[name] = this._map[name] || [];
                        this._map[name].push(item.schema);
                    }
                }
            }
        }
    }

    getSchema(operator, imports) {
        var result = null;
        var schemas = this._map[operator];
        if (schemas) {
            var version = -1;
            for (var schema of schemas) {
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
            }
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
