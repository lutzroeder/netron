/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var tnn = tnn || {};
var base = base || require('./base');

// https://github.com/Tencent/tnn/wiki/tnnproto-and-model-file-structure
// https://github.com/Tencent/tnn/wiki/operation-tnnproto-weight-table

tnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.tnnproto') || identifier.endsWith('.rapidproto')) {
            let text = context.text;
            const lines = text.split(/\r?\n/);
            const header = lines.shift().split(' ').slice(3,-1);
            const signature = header.shift();
            if (signature === '4206624770') {
                return true;
            }
        }
        if (identifier.endsWith('.tnnproto.tnnmodel')|| identifier.endsWith('.rapidproto.rapidmodel')) {
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
                if (signature == 0x007685DD) {
                    return true;
                }
            }
        }
        if (identifier.endsWith('.tnnmodel') || identifier.endsWith('.rapidmodel')) {
            if (identifier == 'snapshot_blob.tnnmodel' || identifier === 'v8_context_snapshot.tnnmodel') {
                return false;
            }
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
                if ( signature === -88342526) {
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
}

tnn.Model = class {

    constructor(metadata, tnnproto, tnnmodel) {
        this._graphs = [];
        this._graphs.push(new tnn.Graph(metadata, tnnproto, tnnmodel));
    }

    get format() {
        return 'tnn';
    }

    get graphs() {
        return this._graphs;
    }
}

tnn.Graph = class {

    constructor(metadata, tnnproto, tnnmodel) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const blobReader = new tnn.BlobReader(tnnmodel);

        const layers = (typeof tnnproto == 'string') ?
            this._tnnproto(metadata, tnnproto, tnnmodel) :
            this._tnnproto_tnnmodel(metadata, tnnproto, tnnmodel);
        for (const layer of layers) {
            if (layer.type == 'Input') {
                const dimensions = layer.attributes.map((a) => !isNaN(parseInt(a.value, 10)) ? parseInt(a.value, 10) : a.value);
                const shape = new tnn.TensorShape(dimensions);
                const type = new tnn.TensorType('float32', shape);
                this._inputs.push(new tnn.parameter(layer.name, true, layer.outputs.map((output) => new tnn.Argument(output, type, null))));
            }
            else {
                this._nodes.push(new tnn.Node(metadata, blobReader, layer));
            }
        }
    }

    _tnnproto(metadata, tnnproto) {
        const lines = tnnproto.split(/\r?\n/);
        const header = lines.shift().split(' ').slice(3,-1);
        const signature = header.shift();
        if (signature !== '4206624770') {
            throw new tnn.Error('Invalid signature.');
        }
        if (header.length !== 0) {
            throw new tnn.Error('Invalid header count.');
        }

        const layers = [];
        let layer;
        const inputline = lines.shift().trim().slice(1,-2);
        const inlumns = inputline.split(' ').filter((s) => s.length != 0);
        layer = {};
        layer.type = 'Input';
        layer.name= 'input';
        layer.inputs = {};
        layer.outputs = inlumns.splice(0, 1);
        layer.attr = {};
        layer.attributes = [];
        let incount=0;
        for (const inlumn of inlumns) {
            const inparts = inlumn.split(' ');
            if (inparts.length === 1) {
                let key = incount;
                let value = inparts.toString();
                layer.attr[key] = value;
                layer.attributes.push({ key: key, value: value });
                incount++;
            }
        }
        layers.push(layer);

        var uknownuse=lines.shift().split(' ');
        var uknownuse=lines.shift().split(' ');
        var uknownuse=lines.shift().split(' ');

        while (lines.length > 0) {
            const line = lines.shift().trim().slice(1,-2);
            if (line.length > 0) {
                const columns = line.split(' ').filter((s) => s.length != 0);
                layer = {};
                layer.type = columns.shift();
                layer.name = columns.shift();
                const inputCount = parseInt(columns.shift(), 10);
                const outputCount = parseInt(columns.shift(), 10);
                layer.inputs = columns.splice(0, inputCount);
                layer.outputs = columns.splice(0, outputCount);
                layer.attr = {};
                layer.attributes = [];
                let count=0;
                for (const column of columns) {
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
                layers.push(layer);
            }
        }
        return layers;
    }

    _tnnproto_tnnmodel(metadata, tnnproto) {
        const reader = new tnn.BinryReader(tnnproto);
        if (reader.int32() !== 0x007685DD) {
            throw new tnn.Error('Invalid signature.')
        }
        const layerCount = reader.int32();
        /* const blobCount = */ reader.int32();
        const layers = [];
        for (let i = 0; i < layerCount; i++) {
            const layer = {};
            const typeIndex = reader.int32();
            const operator = metadata.operator(typeIndex);
            layer.type = operator || typeIndex.toString();
            layer.name = i.toString();
            layer.inputs = [];
            layer.outputs = [];
            layer.attr = {};
            layer.attributes = [];
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
            layers.push(layer);
        }
        return layers;
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
}

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

    constructor(metadata, blobReader, layer) {
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._operator = layer.type;
        this._name = layer.name;

        const operator = metadata.operator(this._operator);
        if (operator) {
            this._operator = operator;
        }

        const schema = metadata.type(this._operator);

        const attributeMetadata = {};
        if (schema && schema.attributes) {
            for (let i = 0; i < schema.attributes.length; i++) {
                const id = schema.attributes[i].id || i.toString();
                attributeMetadata[id] = schema.attributes[i];
            }
        }
        for (const attribute of layer.attributes) {
            const attributeSchema = attributeMetadata[attribute.key];
            this._attributes.push(new tnn.Attribute(attributeSchema, attribute.key, attribute.value));
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
                        return new tnn.Argument(id, null, null)
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
        let weight_input_index;
        let num_output;
        let weight_data_size;
        let channels;
        let scale_data_size;
        switch (this._operator) {
            case 'InstBatchNormCxx':
            case 'InstanceNorm': {
                channels = blobReader.getlongth(this._operator);
                this._weight(blobReader, this._operator,'scale', [ channels ], 'float32');
                this._biasweight(blobReader, this._operator,'bias', [ channels ], 'float32');
                break;
            }
            case 'BatchNormCxx':
            case 'BatchNorm': {
                channels = blobReader.getlongth(this._operator);
                this._weight(blobReader, this._operator,'scale', [ channels ], 'float32');
                this._biasweight(blobReader, this._operator,'bias', [ channels ], 'float32');
                break;
            }
            case 'InnerProduct': {
                num_output = parseInt(layer.attr['0'] || 0, 10);
                weight_data_size = blobReader.getlongth(this._operator);
                this._weight(blobReader,this._operator, 'weight', [ num_output, weight_data_size / num_output ]);
                if (layer.attr['1'] == '1') {
                    this._biasweight(blobReader, this._operator,'bias', [ num_output ], 'float32');
                }
                if(blobReader.getquantized(this._operator)){
                    this._biasweight(blobReader, this._operator,'scale', [ num_output ], 'float32');
                }
                break;
            }
            case 'Conv3D':{
                num_output = parseInt(layer.attr['2'] || 0, 10);
                const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                const kernel_d = parseInt(layer.attr['5'] || kernel_w, 10);
                weight_data_size = blobReader.getlongth(this._operator);
                this._weight(blobReader, this._operator,'weight', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h  * kernel_d), kernel_w, kernel_h, kernel_d ]);
                if (layer.attr['11'] == '1') {
                    this._biasweight(blobReader, this._operator,'bias', [ num_output ], 'float32');
                }
                if(blobReader.getquantized()){
                    this._biasweight(blobReader, this._operator,'quantized', [ num_output ], 'float32');
                }
                break;
                break;
            }
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                num_output = parseInt(layer.attr['2'] || 0, 10);
                const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                weight_data_size = blobReader.getlongth(this._operator);
                this._weight(blobReader, this._operator,'weight', [ num_output, weight_data_size / ( num_output * kernel_w * kernel_h), kernel_w, kernel_h ]);
                if (layer.attr['9'] == '1') {
                    this._biasweight(blobReader, this._operator,'bias', [ num_output ], 'float32');
                }
                if(blobReader.getquantized()){
                    this._biasweight(blobReader, this._operator,'quantized', [ num_output ], 'float32');
                }
                break;
            }
            case 'BlobScale': {
                scale_data_size = blobReader.getlongth(this._operator);
                if (scale_data_size != -233) {
                    this._weight(blobReader, this._operator,'scale', [ scale_data_size], 'float32');
                    this._biasweight(blobReader, this._operator,'bias', [ scale_data_size ], 'float32');

                }
                break;
            }
            case 'PReLU': {
                const num_slope = blobReader.getlongth(this._operator);
                this._weight(blobReader, this._operator,'slope', [ num_slope ], 'float32');
                break;
            }

            case 'HdrGuide': {
                const weight_size = blobReader.getlongth(this._operator);
                this._weight(blobReader, this._operator,'ccm_weight', [ weight_size ], 'float32');
                this._biasweight(blobReader, this._operator,'ccm_bias', [ weight_size ], 'float32');
                this._biasweight(blobReader, this._operator,'shifts', [ weight_size ], 'float32');
                this._biasweight(blobReader, this._operator,'slopes', [ weight_size ], 'float32');
                this._biasweight(blobReader, this._operator,'projection_weight', [ weight_size ], 'float32');
                this._biasweight(blobReader, this._operator,'projection_bias', [ weight_size ], 'float32');
                break;
            }
            case 'Div':
            case 'Sub':
            case 'Add':
            case 'Mul': {
                weight_input_index=blobReader.getsymbol();
                if(weight_input_index){
                    const num_slope = blobReader.getlongth(this._operator);
                    this._weight(blobReader, this._operator,'slope', [ num_slope ], 'float32');
                    break;
                }
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata.type(this._operator);
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

    _weight(blobReader,operator, name, dimensions, dataType) {
        const blob = blobReader.read(dimensions, dataType, operator);
        dataType = blob ? (blob.dataType || '?') : (dataType || '?');
        const data = blob ? blob.data : null;
        this._inputs.push(new tnn.parameter(name, true, [
            new tnn.Argument('', null, new tnn.Tensor(new tnn.TensorType(dataType, new tnn.TensorShape(dimensions)), data))
        ]));
    }
    _biasweight(blobReader,operator, name, dimensions, dataType) {
        const blob = blobReader.GetRaw(dimensions,dataType,operator);
        dataType = blob ? (blob.dataType || '?') : (dataType || '?');
        const data = blob ? blob.data : null;
        this._inputs.push(new tnn.parameter(name, true, [
            new tnn.Argument('', null, new tnn.Tensor(new tnn.TensorType(dataType, new tnn.TensorShape(dimensions)), data))
        ]));
    }
    _weight2(blobReader,operator, name, dimensions, dataType) {
        const blob = blobReader.read2(dimensions,dataType,operator);
        dataType = blob ? (blob.dataType || '?') : (dataType || '?');
        const data = blob ? blob.data : null;
        this._inputs.push(new tnn.parameter(name, true, [
            new tnn.Argument('', null, new tnn.Tensor(new tnn.TensorType(dataType, new tnn.TensorShape(dimensions)), data))
        ]));
    }
}

tnn.Attribute = class {

    constructor(schema, key, value) {
        this._type = '';
        this._name = key;
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
}

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

}

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
}

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

tnn.BinryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    int32() {
        const position = this._position;
        this._position += 4;
        if (this._position > this._buffer.length) {
            throw new tnn.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        return this._dataView.getInt32(position, true);
    }
}



tnn.BlobReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._bais=0;
        this._position = 8;
    }
    Getint() {
        const f0 = this._buffer[this._position++];
        const f1 = this._buffer[this._position++];
        const f2 = this._buffer[this._position++];
        const f3 = this._buffer[this._position++];
        const int = f0 | f1 << 8 | f2 << 16 ;
        const signature= f3*256*256*256+int;
        return signature;

    }


    GetRaw(shape, dataType,operator){
        if(this._buffer){
            const pos=this._position;
            const magic_number=this.Getint();
            const type =this.Getint();
            const longth=this.Getint();
            if ( magic_number != 4206624770) {
                throw new tnn.Error("wrong position begin"+pos+" and happened in "+this._position +"with the wrong magic_number is "+ magic_number+". And datatype is "+ type +". Longth is "+longth+ ". Operater is"+operator );
            }
            switch (type) {
                case 0x00000000:
                    dataType = 'float32';
                    break;
                case 0x00000001:
                    dataType = 'float16';
                    break;
                case 0x00000002:
                    dataType = 'int8';
                    break;
                case 0x00000003:
                    dataType = 'int32';
                    break;
                case 0x00000004:
                    dataType = 'bfp16';
                    break;
                case 0x0002C056: // size * sizeof(float) - raw data with extra scaling
                default:
                    throw new tnn.Error("Unknown Data type '" + type + "'.");
            }
            let data = null;
            let size = longth;
            if (this._buffer) {
                if (dataType) {
                    const position = this._position;
                    switch (dataType) {
                        case 'float32':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'float16':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'int8':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'int32':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'bfp16':
                            this._position += size + 1024;
                            data = null;
                            break;
                        default:
                            throw new tnn.Error("Unknown weight type '" + dataType + "'.");
                    }
                }
            }
            return {dataType: dataType, data: data};
        }
        return null;
    }

    read(shape, dataType,operator) {
        if (this._buffer) {

            this._position+=4;
            let data = null;
            let size = 1;
            const typelength = this.Getint() ;
            this._position += typelength;
            const namelength = this.Getint();
            this._position += namelength;
            switch(operator) {
                case 'InnerProduct': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    break;
                }
                case 'Convolution':
                case 'ConvolutionDepthWise':
                case 'Deconvolution':
                case 'Conv3D':
                case 'DeconvolutionDepthWise': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    this._bais=this.Getint();
                    break;
                }
                case 'PReLU': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    break;
                }
            }

            if (this._buffer) {
                const magic_number=this.Getint();

                const type =this.Getint();
                if ( magic_number != 4206624770) {
                    throw new tnn.Error(" Read unknown magic_number "+ magic_number+". And type is"+ type+ ". Operator is "+operator+" ." );
                }
                switch (type) {
                    case 0x00000000:
                        dataType = 'float32';
                        break;
                    case 0x00000001:
                        dataType = 'float16';
                        break;
                    case 0x00000002:
                        dataType = 'int8';
                        break;
                    case 0x00000003:
                        dataType = 'int32';
                        break;
                    case 0x00000004:
                        dataType = 'bfp16';
                        break;
                    case 0x0002C056: // size * sizeof(float) - raw data with extra scaling
                    default:
                        throw new tnn.Error("Unknown data Type '" + type + "'.");
                }
                if (dataType) {
                    size=this.Getint();
                    const position = this._position;
                    switch (dataType) {
                        case 'float32':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'float16':

                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'int8':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'qint8':
                            this._position += size + 1024;
                            data = null;
                            break;
                        default:
                            throw new tnn.Error("Unknown Weight type '" + dataType + "'.");
                    }
                }
            }
            return { dataType: dataType, data: data }
        }
        return null;
    }


    getlongth(operator){
        if(this._buffer){

            const position=this._position;
            this._position+=4;
            const typelength = this.Getint() ;
            this._position += typelength;
            const namelength = this.Getint();
            this._position += namelength;
            switch(operator) {
                case 'InnerProduct': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    break;
                }
                case 'Convolution':
                case 'ConvolutionDepthWise':
                case 'Conv3D':
                case 'Deconvolution':
                case 'DeconvolutionDepthWise': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    this._bais=this.Getint();
                    break;
                }
                case 'PReLU': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    break;
                }
            }
            const magic_number=this.Getint();
            const dataType=this.Getint();
            let size=this.Getint();
            if ( magic_number != 4206624770) {
                throw new tnn.Error(" Unknown magic_number "+ magic_number +" happened in position "+ position+ ". Wrong operator is "+operator );
            }
            this._position=position;
            switch (dataType) {
                case 0x00000000:
                    size = size/4;
                    break;
                case 0x00000001:
                    size = size/2;
                    break;
                case 0x00000002:
                    size = size/1;
                    break;
                case 0x00000003:
                    size = size/4;
                    break;
                case 0x00000004:
                    size = size/4;
                    break;
                case 0x0002C056: // size * sizeof(float) - raw data with extra scaling
                default:
                    throw new tnn.Error("i am a pig '" + dataType + "'.");
            }
            return size;
        }
        return 0;
    }

    getquantized(operator){
        if(this._buffer){
            const position=this._position;
            this._position+=4;
            const typelength = this.Getint() ;
            this._position += typelength;
            const namelength = this.Getint();
            this._position += namelength;
            switch(operator) {
                case 'InnerProduct': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    break;
                }
                case 'Convolution':
                case 'ConvolutionDepthWise':
                case 'Conv3D':
                case 'Deconvolution':
                case 'DeconvolutionDepthWise': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    this._bais=this.Getint();
                    break;
                }
                case 'PReLU': {
                    const namelength = this.Getint() ;
                    this._position += namelength;
                    break;
                }
            }
            this._position+=4;
            let dataType=this.Getint();
            this._position=position;
            if(dataType == 0x00000002){
                return 1;
            }
        }
        return 0;
    }
    getsymbol()
    {
        const position = this._position;
        this._position+=4;
        const symbol=this.Getint();
        this._position=position;
        if(symbol == 3){
            return 1;
        }
        return 0;
    }
}

tnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading tnn model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tnn.ModelFactory;
}