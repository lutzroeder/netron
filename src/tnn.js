/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var tnn = tnn || {};
var base = base || require('./base');

tnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.tnnproto') || identifier.endsWith('.rapidproto')) {
            let text = context.text;
            text = text.substring(0, Math.min(text.length, 128));
            const line = text.split('\n').shift().trim();
            if (line.startsWith('"') && line.endsWith('"')) {
                const header = line.replace(/(^")|("$)/g, '').split(',').shift().trim().split(' ');
                if (header.length >= 3 || (header.length >= 4 && header[3] === '4206624770')) {
                    return true;
                }
            }
        }
        /*
        if (identifier.endsWith('.tnnproto.tnnmodel')|| identifier.endsWith('.rapidproto.rapidmodel')) {
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature == 0x007685DD) {
                    return true;
                }
            }
        }
        */
        if (identifier.endsWith('.tnnmodel') || identifier.endsWith('.rapidmodel')) {
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature === 0xFABC0002) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        return tnn.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier.toLowerCase();
            const tnnproto = (tnnproto, tnnmodel) => {
                try {
                    return new tnn.Model(metadata, tnnproto, tnnmodel);
                }
                catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tnn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
            };
            let tnnmodel = null;
            if (identifier.endsWith('.tnnproto') || identifier.endsWith('.rapidproto')) {
                if (identifier.endsWith('.tnnproto')) {
                    tnnmodel = context.identifier.substring(0, context.identifier.length - 9) + '.tnnmodel';
                }
                else if (identifier.endsWith('.rapidproto')) {
                    tnnmodel = context.identifier.substring(0, context.identifier.length - 11) + '.rapidmodel';
                }
                return context.request(tnnmodel, null).then((tnnmodel) => {
                    return tnnproto(context.text, tnnmodel);
                }).catch(() => {
                    return tnnproto(context.text, null);
                });
            }
            else if (identifier.endsWith('.tnnproto.tnnmodel')) {
                tnnmodel = context.identifier.substring(0, context.identifier.length - 18) + '.tnnmodel';
                return context.request(tnnmodel, null).then((tnnmodel) => {
                    return tnnproto(context.buffer, tnnmodel);
                }).catch(() => {
                    return tnnproto(context.buffer, null);
                });
            }
            else if (identifier.endsWith('.tnnmodel')|| identifier.endsWith('.rapidproto')) {
                let text = null;
                if  (identifier.endsWith('.tnnmodel')){
                    text = context.identifier.substring(0, context.identifier.length - 9) + '.tnnproto';
                }
                else if(identifier.endsWith('.rapidmodel')){
                    text = context.identifier.substring(0, context.identifier.length - 11) + '.rapidproto';
                }
                return context.request(text, 'utf-8').then((text) => {
                    return tnnproto(text, context.buffer);
                }).catch((error) => {
                    const message = error && error.message ? error.message : error.toString();
                    throw new tnn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                });
            }
        });
    }
};

tnn.Model = class {

    constructor(metadata, tnnproto, tnnmodel) {
        this._graphs = [];
        this._graphs.push(new tnn.Graph(metadata, tnnproto, tnnmodel));
    }

    get format() {
        return 'TNN';
    }

    get graphs() {
        return this._graphs;
    }
};

tnn.Graph = class {

    constructor(metadata, tnnproto, tnnmodel) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const resources = new tnn.LayerResourceReader(tnnmodel);
        const reader = (typeof tnnproto == 'string') ? new tnn.TextProtoReader(tnnproto) : new tnn.BinaryProtoReader(metadata, tnnproto);
        for (const input of reader.inputs) {
            const shape = new tnn.TensorShape(input.shape);
            const type = new tnn.TensorType('float32', shape);
            this._inputs.push(new tnn.parameter(input.name, true, [ new tnn.Argument(input.name, type, null) ]));
        }
        for (const output of reader.outputs) {
            this._outputs.push(new tnn.parameter(output.name, true, [ new tnn.Argument(output.name, null, null) ]));
        }
        for (const layer of reader.layers) {
            this._nodes.push(new tnn.Node(metadata, resources, layer));
        }
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
};

tnn.parameter = class {

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

tnn.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new tnn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

tnn.Node = class {

    constructor(metadata, resources, layer) {
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._type = layer.type;
        this._name = layer.name;

        const operator = metadata.operator(this._type);
        if (operator) {
            this._type = operator;
        }

        const schema = metadata.type(this._type);

        const attributeSchemas = schema && schema.attributes ? schema && schema.attributes.slice() : [];
        const attributes = layer.attributes.slice();
        while (attributes.length > 0) {
            const attributeSchema = attributeSchemas.shift();
            let value = null;
            let name = '';
            if (attributeSchema && attributeSchema.type === 'int32[]' && attributeSchema.size) {
                name = attributeSchema.name;
                value = attributes.splice(0, layer.attr[attributeSchema.size]).map((attribute) => parseInt(attribute.value, 10));
            }
            else {
                const attribute = attributes.shift();
                name = attribute.key;
                value = attribute.value;
            }
            this._attributes.push(new tnn.Attribute(attributeSchema, name, value));
        }

        const inputs = layer.inputs;
        let inputIndex = 0;
        if (schema && schema.inputs) {
            for (const inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => {
                        return new tnn.Argument(id, null, null);
                    });
                    this._inputs.push(new tnn.parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        else {
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new tnn.parameter(inputName, true, [
                    new tnn.Argument(input, null, null)
                ]);
            }));
        }

        const outputs = layer.outputs;
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (const outputDef of schema.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => {
                        return new tnn.Argument(id, null, null);
                    });
                    this._outputs.push(new tnn.parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        else {
            this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
                const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new tnn.parameter(outputName, true, [
                    new tnn.Argument(output, null, null)
                ]);
            }));
        }
        switch (this._type) {
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                const resource = resources.read(this._name);
                if (resource) {
                    const num_output = parseInt(layer.attr['2'] || 0, 10);
                    const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                    const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                    const weight_data_size = resource.filter.length;
                    this._weight(resource, 'filter', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h), kernel_w, kernel_h ]);
                    if (resource.bias) {
                        this._weight(resource, 'bias', [ num_output ]);
                    }
                    if (resource.quantized) {
                        this._weight(resource, 'quantized', [ num_output ]);
                    }
                }
                break;
            }
            case 'Conv3D':{
                const resource = resources.read(this._name);
                if (resource) {
                    const num_output = parseInt(layer.attr['2'] || 0, 10);
                    const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                    const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                    const kernel_d = parseInt(layer.attr['5'] || kernel_w, 10);
                    const weight_data_size = resource.filter.length;
                    this._weight(resource, 'weight', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h  * kernel_d), kernel_w, kernel_h, kernel_d ]);
                    if (resource.bias) {
                        this._weight(resources, 'bias', [ num_output ]);
                    }
                }
                break;
            }
            case 'InnerProduct': {
                const resource = resources.read(this._name);
                if (resource) {
                    const num_output = parseInt(layer.attr['0'] || 0, 10);
                    const weight_data_size = resource.weight.length;
                    this._weight(resource, 'weight', [ num_output, weight_data_size / num_output ]);
                    this._weight(resource, 'bias', [ num_output ]);
                    if (resource.weight.dataType === 'int8') {
                        this._weight(resource, 'scale', [ num_output ]);
                    }
                }
                break;
            }
            case 'PReLU': {
                const resource = resources.read(this._name);
                if (resource) {
                    this._weight(resource, 'slope', [ resource.slope.length ]);
                }
                break;
            }
            case 'BatchNormCxx': {
                const resource = resources.read(this._name);
                if (resource) {
                    this._weight(resource, 'scale', [ resource.scale.length ]);
                    this._weight(resource, 'bias', [ resource.bias.length ]);
                }
                break;
            }
            case 'Div':
            case 'Sub':
            case 'Add':
            case 'Mul': {
                if (this._inputs.length === 1) {
                    const resource = resources.read(this._name);
                    if (resource) {
                        const num_output = resource.slope.length;
                        this._weight(resource, 'slope', [ num_output ]);
                    }
                }
                break;
            }
            case 'HdrGuide': {
                const resource = resources.read(this._name);
                if (resource) {
                    const weight_size = resource.ccm_weight.length;
                    this._weight(resource, 'ccm_weight', [ weight_size ]);
                    this._weight(resource, 'ccm_bias', [ weight_size ]);
                    this._weight(resource, 'shifts', [ weight_size ]);
                    this._weight(resource, 'slopes', [ weight_size ]);
                    this._weight(resource, 'projection_weight', [ weight_size ]);
                    this._weight(resource, 'projection_bias', [ weight_size ]);
                }
                break;
            }
            case 'BlobScale': {
                const resource = resources.read(this._name);
                if (resource) {
                    const scale_data_size = resource.scale.length;
                    this._weight(resource, 'scale', [ scale_data_size]);
                    this._weight(resource, 'bias', [ scale_data_size ]);
                }
                break;
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata.type(this._type);
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

    _weight(resource, name, shape) {
        const initializer = resource[name];
        if (!initializer) {
            throw new tnn.Error("Layer initializer'" + resource.type + "." + name + "' not found '");
        }
        this._inputs.push(new tnn.parameter(name, true, [
            new tnn.Argument('', null, new tnn.Tensor(new tnn.TensorType(initializer.dataType, new tnn.TensorShape(shape)), initializer.value))
        ]));
    }
};

tnn.Attribute = class {

    constructor(schema, key, value) {
        this._type = '';
        this._name = key.toString();
        this._value = value;
        if (schema) {
            this._name = schema.name;
            if (schema.type) {
                this._type = schema.type;
            }
            switch (this._type) {
                case 'int32':
                    this._value = parseInt(this._value, 10);
                    break;
                case 'float32':
                    this._value = parseFloat(this._value);
                    break;
                case 'float32[]':
                    this._value = this._value.map((v) => parseFloat(v));
                    break;
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default || (this._value && this._value.toString() == schema.default.toString())) {
                    this._visible = false;
                }
            }
        }
    }

    get type() {
        return this._type;
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
};

tnn.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Weight';
    }

    get type() {
        return this._type;
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
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'float16':
            case 'float32':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
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
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};

tnn.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

tnn.TensorShape = class {

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

tnn.Metadata = class {

    static open(host) {
        if (tnn.Metadata._metadata) {
            return Promise.resolve(tnn.Metadata._metadata);
        }
        return host.request(null, 'tnn-metadata.json', 'utf-8').then((data) => {
            tnn.Metadata._metadata = new tnn.Metadata(data);
            return tnn.Metadata._metadata;
        }).catch(() => {
            tnn.Metadata._metadata = new tnn.Metadata(null);
            return tnn.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._operatorMap = new Map();
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                        if (Object.prototype.hasOwnProperty.call(item.schema, 'operator')) {
                            this._operatorMap.set(item.schema.operator, item.name);
                        }
                    }
                }
            }
        }
    }

    operator(code) {
        return this._operatorMap.get(code);
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

tnn.TextProtoReader = class {

    constructor(text) {
        const split = (line, delimiter, trim, ignore_blank) => {
            return line.split(delimiter).map((v) => trim ? v.trim() : v).filter((v) => !ignore_blank || v);
        };
        const lines = split(text.replace(/\r?\n|"/g, ''), ',', true, false);
        if (lines.length <= 5) {
            throw new tnn.Error('Invalid line count.');
        }
        const header = split(lines.shift(), ' ', true, false);
        if (header.length < 3) {
            throw new tnn.Error('Invalid header size.');
        }
        else if (header.length > 3 && header[3] !== '4206624770') {
            throw new tnn.Error("Invalid signature '" + header[3] + "'.");
        }
        this._inputs = split(lines.shift(), ':', true, false).map((input) => {
            const array = split(input, ' ', true, false);
            const name = array.shift();
            const shape = array.map((dim) => parseInt(dim, 19));
            return { name: name, shape: shape };
        });
        lines.shift();
        this._outputs = split(lines.shift(), ' ', true, false).map((output) => { return { name: output }; });
        lines.shift();
        this._layers = [];
        while (lines.length > 0) {
            const line = lines.shift().trim();
            if (line.length > 0) {
                const array = split(line, ' ', true, true);
                const layer = {};
                layer.type = array.shift();
                layer.name = array.shift();
                const inputCount = parseInt(array.shift(), 10);
                const outputCount = parseInt(array.shift(), 10);
                layer.inputs = array.splice(0, inputCount);
                layer.outputs = array.splice(0, outputCount);
                layer.attr = {};
                layer.attributes = [];
                let count = 0;
                for (const column of array) {
                    const parts = column.split(' ');
                    if (parts.length === 1) {
                        let key = count;
                        let value = parts.toString();
                        const keyInt = parseInt(key, 10);
                        if (keyInt < 0) {
                            value = value.split(',').map((v) => v.trim());
                            value.shift();
                            key = (-(keyInt + 23300)).toString();
                        }
                        layer.attr[key] = value;
                        layer.attributes.push({ key: key, value: value });
                        count++;
                    }
                }
                this._layers.push(layer);
            }
        }
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get layers() {
        return this._layers;
    }
};

/*
tnn.BinaryProtoReader = class {

    constructor(metadata, buffer) {
        const reader = new tnn.BinaryProtoReader(buffer);
        if (reader.int32() !== 0x007685DD) {
            throw new tnn.Error('Invalid signature.');
        }
        const layerCount = reader.int32();
        this._inputs = [];
        this._outputs = [];
        this._layers = layers;
        reader.int32(); // blobCount
        const layers = [];
        for (let i = 0; i < layerCount; i++) {
            const typeIndex = reader.int32();
            const operator = metadata.operator(typeIndex);
            const layer = {
                type: operator || typeIndex.toString(),
                name: i.toString(),
                inputs: [],
                outputs: [],
                attr: {},
                attributes: []
            };
            const inputCount = reader.int32();
            const outputCount = reader.int32();
            for (let j = 0; j < inputCount; j++) {
                layer.inputs.push(reader.int32().toString());
            }
            for (let k = 0; k < outputCount; k++) {
                layer.outputs.push(reader.int32().toString());
            }
            let id = reader.int32();
            while (id != -233) {
                let isArray = id <= -23300;
                if (isArray) {
                    id = -id - 23300;
                }
                if (isArray) {
                    const len = reader.int32();
                    const values = [];
                    for (let i = 0; i < len; i++) {
                        values.push(reader.int32());
                    }
                    layer.attributes.push({ key: id.toString(), value: values.toString() });
                    layer.attr[id.toString()] = values;
                }
                else {
                    const value = reader.int32();
                    layer.attributes.push({ key: id.toString(), value: value.toString() });
                    layer.attr[id.toString()] = value.toString();
                }
                id = reader.int32();
            }
            this._layers.push(layer);
        }
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get layers() {
        return this._layers;
    }
};
*/

tnn.LayerResourceReader = class {

    constructor(buffer) {
        this._layerResources = [];
        if (buffer) {
            const reader = new tnn.BinaryReader(buffer);
            const magic_number = reader.uint32();
            if (magic_number !== 0xFABC0002) {
                throw new tnn.Error("Invalid blob header signature '" + magic_number.toString() + "'.");
            }
            const layerCount = reader.int32() & 0x1FFFFFFF;
            const raw = (reader) => {
                const magic_number = reader.uint32();
                if (magic_number !== 0xFABC0002) {
                    throw new tnn.Error("Invalid raw signature '" + magic_number.toString() + "'.");
                }
                const data_type = reader.int32();
                if (data_type > 4) {
                    throw new tnn.Error("Unknown data type '" + data_type + "'.");
                }
                const length = reader.int32();
                if (length <= 0) {
                    return null;
                }
                return {
                    dataType: [ 'float32', 'float16', 'int8', 'int32', 'bfloat16' ][data_type],
                    length: length / [ 4, 2, 1, 4, 2 ][data_type],
                    value: reader.bytes(length)
                };
            };
            for (let i = 0; i < layerCount; i++) {
                const resource = {};
                resource.operator = reader.int32();
                resource.type = reader.string();
                resource.name = reader.string();
                switch (resource.type) {
                    case 'Convolution':
                    case 'ConvolutionDepthWise':
                    case 'Deconvolution':
                    case 'DeconvolutionDepthWise': {
                        reader.expect(resource.name);
                        const bias = reader.int32();
                        resource.filter = raw(reader);
                        if (bias) {
                            resource.bias = raw(reader);
                        }
                        if (resource.filter.dataType === 'int8') {
                            resource.quantized = raw();
                        }
                        break;
                    }
                    case 'Conv3D': {
                        reader.expect(resource.name);
                        const bias = reader.int32();
                        resource.filter = raw(reader);
                        if (bias) {
                            resource.bias = raw(reader);
                        }
                        break;
                    }
                    case 'InnerProduct': {
                        reader.expect(resource.name);
                        resource.weight = raw(reader);
                        resource.bias = raw(reader);
                        if (resource.weight.dataType === 'int8') {
                            resource.scale = raw();
                        }
                        break;
                    }
                    case 'PReLU': {
                        reader.expect(resource.name);
                        resource.slope = raw(reader);
                        break;
                    }
                    case 'Add':
                    case 'Mul': {
                        resource.slope = raw(reader);
                        break;
                    }
                    case 'BatchNormCxx':
                        resource.scale = raw(reader);
                        resource.bias = raw(reader);
                        break;
                    case 'HdrGuide':
                        resource.ccm_weight = raw(reader);
                        resource.ccm_bias = raw(reader);
                        resource.shifts = raw(reader);
                        resource.slopes = raw(reader);
                        resource.projection_weight = raw(reader);
                        resource.projection_bias = raw(reader);
                        break;
                    case 'BlobScale':
                        resource.scale = raw(reader);
                        resource.bias = raw(reader);
                        break;
                    default:
                        throw new tnn.Error("Unknown layer resource type '" + resource.type + "'.");
                }
                this._layerResources.push(resource);
            }
            if (!reader.end()) {
                throw new tnn.Error("Invalid blob size.");
            }
        }
    }

    read(name) {
        const resource = this._layerResources.shift();
        if (resource && resource.name !== name) {
            throw new tnn.Error("Invalid blob layer name '" + name + "'.");
        }
        return resource;
    }
};

tnn.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    end() {
        return this._position === this._buffer.length;
    }

    skip(size) {
        this._position += size;
        if (this._position > this._buffer.length) {
            throw new tnn.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    bytes(size) {
        const position = this._position;
        this.skip(size);
        return this._buffer.subarray(position, this._position);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    string() {
        const length = this.int32();
        const position = this._position;
        this.skip(length);
        const data = this._buffer.subarray(position, this._position);
        return new TextDecoder('utf-8').decode(data);
    }

    expect(name) {
        const text = this.string();
        if (name !== text) {
            throw new tnn.Error("Invalid string '" + text + "' instead of '" + name + "'.");
        }
    }
};

tnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tnn.ModelFactory;
}