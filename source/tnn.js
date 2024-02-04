
import * as base from './base.js';
import * as text from './text.js';

const tnn = {};

tnn.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        const stream = context.stream;
        if (stream && identifier.endsWith('.tnnproto')) {
            try {
                const buffer = stream.peek();
                const reader = text.Reader.open(buffer, 2048);
                const content = reader.read();
                if (content !== undefined) {
                    const line = content.trim();
                    if (line.startsWith('"') && line.endsWith('"')) {
                        const header = line.replace(/(^")|("$)/g, '').split(',').shift().trim().split(' ');
                        if (header.length === 3 || (header.length >= 4 && (header[3] === '4206624770' || header[3] == '4206624772'))) {
                            context.type = 'tnn.model';
                            return;
                        }
                    }
                }
            } catch (err) {
                // continue regardless of error
            }
        }
        if (stream && identifier.endsWith('.tnnmodel')) {
            for (const signature of [ [ 0x02, 0x00, 0xbc, 0xfa ], [ 0x04, 0x00, 0xbc, 0xfa ] ]) {
                if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                    context.type = 'tnn.params';
                    return;
                }
            }
        }
    }

    async open(context) {
        const metadata = await context.metadata('tnn-metadata.json');
        switch (context.type) {
            case 'tnn.model': {
                const name = `${context.identifier.substring(0, context.identifier.length - 9)}.tnnmodel`;
                try {
                    const content = await context.fetch(name);
                    const buffer = content.stream.peek();
                    return new tnn.Model(metadata, context.stream.peek(), buffer);
                } catch (error) {
                    return new tnn.Model(metadata, context.stream.peek(), null);
                }
            }
            case 'tnn.params': {
                const name = `${context.identifier.substring(0, context.identifier.length - 9)}.tnnproto`;
                const content = await context.fetch(name, null);
                const buffer = content.stream.peek();
                return new tnn.Model(metadata, buffer, context.stream.peek());
            }
            default: {
                throw new tnn.Error(`Unsupported TNN format '${context.type}'.`);
            }
        }
    }
};

tnn.Model = class {

    constructor(metadata, tnnproto, tnnmodel) {
        this.format = 'TNN';
        this.graphs = [
            new tnn.Graph(metadata, tnnproto, tnnmodel)
        ];
    }
};

tnn.Graph = class {

    constructor(metadata, tnnproto, tnnmodel) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const resources = new tnn.LayerResourceReader(tnnmodel);
        const reader = new tnn.TextProtoReader(tnnproto);
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (name.length === 0) {
                return new tnn.Value(name, type || null, tensor || null);
            }
            if (!values.has(name)) {
                values.set(name, new tnn.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new tnn.Value(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        for (const input of reader.inputs) {
            const shape = new tnn.TensorShape(input.shape);
            const type = new tnn.TensorType(input.data_type, shape);
            this.inputs.push(new tnn.Argument(input.name, [ values.map(input.name, type) ]));
        }
        for (const output of reader.outputs) {
            this.outputs.push(new tnn.Argument(output.name, [ values.map(output.name) ]));
        }
        for (const layer of reader.layers) {
            this.nodes.push(new tnn.Node(metadata, resources, layer, values));
        }
    }
};

tnn.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

tnn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new tnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
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

    constructor(metadata, resources, layer, values) {
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.name = layer.name;
        this.type = metadata.type(layer.type);
        const attributeSchemas = this.type && this.type.attributes ? this.type && this.type.attributes.slice() : [];
        const attributes = layer.attributes.slice();
        while (attributes.length > 0) {
            const attributeSchema = attributeSchemas.shift();
            let value = null;
            let name = '';
            if (attributeSchema && attributeSchema.type === 'int32[]' && attributeSchema.size) {
                name = attributeSchema.name;
                value = attributes.splice(0, layer.attr[attributeSchema.size]).map((attribute) => parseInt(attribute.value, 10));
            } else {
                const attribute = attributes.shift();
                name = attribute.key;
                value = attribute.value;
            }
            this.attributes.push(new tnn.Attribute(attributeSchema, name, value));
        }

        const inputs = layer.inputs;
        let inputIndex = 0;
        if (this.type && this.type.inputs) {
            for (const inputDef of this.type.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => values.map(id));
                    this.inputs.push(new tnn.Argument(inputDef.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        } else {
            this.inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new tnn.Argument(inputName, [ values.map(input) ]);
            }));
        }

        const outputs = layer.outputs;
        let outputIndex = 0;
        if (this.type && this.type.outputs) {
            for (const outputDef of this.type.outputs) {
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    const outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => values.map(id));
                    this.outputs.push(new tnn.Argument(outputDef.name, outputArguments));
                    outputIndex += outputCount;
                }
            }
        } else {
            this.outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                const outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new tnn.Argument(outputName, [ values.map(output) ]);
            }));
        }
        const weight = (resource, name, shape) => {
            const initializer = resource[name];
            if (!initializer) {
                throw new tnn.Error(`Layer initializer'${resource.type}.${name}' not found '`);
            }
            const tensor = new tnn.Tensor(new tnn.TensorType(initializer.dataType, new tnn.TensorShape(shape)), initializer.value);
            this.inputs.push(new tnn.Argument(name, [ values.map('', null, tensor) ]));
        };
        switch (this.type.name) {
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                const resource = resources.read(this.name);
                if (resource) {
                    const num_output = parseInt(layer.attr['2'] || 0, 10);
                    const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                    const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                    const weight_data_size = resource.filter.length;
                    weight(resource, 'filter', [ num_output, weight_data_size / (num_output * kernel_w * kernel_h), kernel_w, kernel_h ]);
                    if (resource.bias) {
                        weight(resource, 'bias', [ num_output ]);
                    }
                    if (resource.quantized) {
                        weight(resource, 'quantized', [ num_output ]);
                    }
                }
                break;
            }
            case 'Conv3D':{
                const resource = resources.read(this.name);
                if (resource) {
                    const num_output = parseInt(layer.attr['2'] || 0, 10);
                    const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                    const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                    const kernel_d = parseInt(layer.attr['5'] || kernel_w, 10);
                    const weight_data_size = resource.filter.length;
                    weight(resource, 'weight', [ num_output, weight_data_size / (num_output * kernel_w * kernel_h  * kernel_d), kernel_w, kernel_h, kernel_d ]);
                    if (resource.bias) {
                        weight(resources, 'bias', [ num_output ]);
                    }
                }
                break;
            }
            case 'InnerProduct': {
                const resource = resources.read(this.name);
                if (resource) {
                    const num_output = parseInt(layer.attr['0'] || 0, 10);
                    const weight_data_size = resource.weight.length;
                    weight(resource, 'weight', [ num_output, weight_data_size / num_output ]);
                    weight(resource, 'bias', [ num_output ]);
                    if (resource.weight.dataType === 'int8') {
                        weight(resource, 'scale', [ num_output ]);
                    }
                }
                break;
            }
            case 'PReLU': {
                const resource = resources.read(this.name);
                if (resource) {
                    weight(resource, 'slope', [ resource.slope.length ]);
                }
                break;
            }
            case 'BatchNormCxx':
            case 'InstBatchNormCxx': {
                const resource = resources.read(this.name);
                if (resource) {
                    weight(resource, 'scale', [ resource.scale.length ]);
                    weight(resource, 'bias', [ resource.bias.length ]);
                }
                break;
            }
            case 'Div':
            case 'Sub':
            case 'Add':
            case 'Mul':
            case 'MatMul': {
                if (this.inputs.length === 1) {
                    const resource = resources.read(this.name);
                    if (resource) {
                        const num_output = resource.slope.length;
                        weight(resource, 'slope', [ num_output ]);
                    }
                }
                break;
            }
            case 'HdrGuide': {
                const resource = resources.read(this.name);
                if (resource) {
                    const weight_size = resource.ccm_weight.length;
                    weight(resource, 'ccm_weight', [ weight_size ]);
                    weight(resource, 'ccm_bias', [ weight_size ]);
                    weight(resource, 'shifts', [ weight_size ]);
                    weight(resource, 'slopes', [ weight_size ]);
                    weight(resource, 'projection_weight', [ weight_size ]);
                    weight(resource, 'projection_bias', [ weight_size ]);
                }
                break;
            }
            case 'BlobScale': {
                const resource = resources.read(this.name);
                if (resource) {
                    const scale_data_size = resource.scale.length;
                    weight(resource, 'scale', [ scale_data_size]);
                    weight(resource, 'bias', [ scale_data_size ]);
                }
                break;
            }
            case 'Gather': {
                const resource = resources.read(this.name);
                if (resource) {
                    if (resource.data) {
                        weight(resource, 'data', [ resource.data.length ]);
                    }
                    if (resource.indices) {
                        weight(resource, 'indices', [ resource.indices.length ]);
                    }
                }
                break;
            }
            default: {
                break;
            }
        }
    }
};

tnn.Attribute = class {

    constructor(metadata, key, value) {
        this.type = '';
        this.name = key.toString();
        this.value = value;
        if (metadata) {
            this.name = metadata.name;
            if (metadata.type) {
                this.type = metadata.type;
            }
            switch (this.type) {
                case '':
                    break;
                case 'int32':
                    this.value = parseInt(this.value, 10);
                    break;
                case 'float32':
                    this.value = parseFloat(this.value);
                    break;
                case 'int32[]':
                    this.value = this.value.map((v) => parseInt(v, 10));
                    break;
                case 'float32[]':
                    this.value = this.value.map((v) => parseFloat(v));
                    break;
                default:
                    throw new tnn.Error(`Unsupported attribute type '${this.type}'.`);
            }
            if (metadata && metadata.visible === false) {
                this.visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (this.value == metadata.default || (this.value && this.value.toString() == metadata.default.toString())) {
                    this.visible = false;
                }
            }
        }
    }
};

tnn.Tensor = class {

    constructor(type, values) {
        this.type = type;
        this.values = values;
    }
};

tnn.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

tnn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`) : '';
    }
};

tnn.TextProtoReader = class {

    constructor(buffer) {
        const reader = text.Reader.open(buffer);
        let lines = [];
        for (;;) {
            const line = reader.read();
            if (line === undefined) {
                break;
            }
            lines.push(line.replace(/\r|"/g, ''));
        }
        const split = (line, delimiter, trim, ignore_blank) => {
            return line.split(delimiter).map((v) => trim ? v.trim() : v).filter((v) => !ignore_blank || v);
        };
        lines = split(lines.join(''), ',', true, false);
        if (lines.length <= 5) {
            throw new tnn.Error('Invalid line count.');
        }
        const header = split(lines.shift(), ' ', true, false);
        if (header.length < 3) {
            throw new tnn.Error('Invalid header size.');
        } else if (header.length > 3 && (header[3] !== '4206624770' && header[3] !== '4206624772')) {
            throw new tnn.Error(`Invalid signature '${header[3]}'.`);
        }
        this.inputs = split(lines.shift(), ':', true, false).map((input) => {
            const array = split(input, ' ', true, false);
            const name = array.shift();
            if (header[3] === '4206624772') {
                const shape_size = parseInt(array.shift(), 10);
                const data_type_index = parseInt(array[shape_size], 10);
                return {
                    name: name,
                    data_type: [ 'float32', 'float16', 'int8', 'int32', 'bfloat16' ][data_type_index],
                    shape: array.slice(0, -1).map((dim) => parseInt(dim, 10)),
                };

            }
            return {
                name: name,
                data_type: 'float32',
                shape: array.map((dim) => parseInt(dim, 10))
            };
        });
        lines.shift();
        this.outputs = split(lines.shift(), ' ', true, false).map((output) => {
            return { name: output };
        });
        lines.shift();
        this.layers = [];
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
                this.layers.push(layer);
            }
        }
    }
};

tnn.LayerResourceReader = class {

    constructor(buffer) {
        this.layerResources = [];
        if (buffer) {
            const reader = new base.BinaryReader(buffer);
            const magic_number = reader.uint32();
            if (magic_number !== 0xFABC0002 && magic_number !== 0xFABC0004) {
                throw new tnn.Error(`Invalid blob header signature '${magic_number}'.`);
            }
            this.layerResources = new Array(reader.int32() & 0x1FFFFFFF);
            const raw = (reader) => {
                const magic_number = reader.uint32();
                if (magic_number !== 0xFABC0002 && magic_number !== 0xFABC0004) {
                    throw new tnn.Error(`Invalid raw signature '${magic_number}'.`);
                }
                const data_type = reader.int32();
                if (data_type > 4) {
                    throw new tnn.Error(`Unsupported data type '${data_type}'.`);
                }
                const length = reader.int32();
                if (length <= 0) {
                    return null;
                }
                let dims = null;
                if (magic_number === 0xFABC0004) {
                    const dim_size = reader.int32();
                    dims = reader.read(dim_size * 4);
                }
                return {
                    dataType: [ 'float32', 'float16', 'int8', 'int32', 'bfloat16' ][data_type],
                    length: length / [ 4, 2, 1, 4, 2 ][data_type],
                    value: reader.read(length),
                    shape: dims
                };
            };
            const expect = (reader, name) => {
                const content = reader.string();
                if (name !== content) {
                    throw new tnn.Error(`Invalid string '${content}' instead of '${name}'.`);
                }
            };
            for (let i = 0; i < this.layerResources.length; i++) {
                const resource = {};
                resource.operator = reader.int32();
                resource.type = reader.string();
                resource.name = reader.string();
                switch (resource.type) {
                    case 'Convolution':
                    case 'ConvolutionDepthWise':
                    case 'Deconvolution':
                    case 'DeconvolutionDepthWise': {
                        expect(reader, resource.name);
                        const bias = reader.int32();
                        resource.filter = raw(reader);
                        if (bias) {
                            resource.bias = raw(reader);
                        }
                        if (resource.filter.dataType === 'int8') {
                            resource.quantized = raw(reader);
                        }
                        break;
                    }
                    case 'Conv3D': {
                        expect(reader, resource.name);
                        const bias = reader.int32();
                        resource.filter = raw(reader);
                        if (bias) {
                            resource.bias = raw(reader);
                        }
                        break;
                    }
                    case 'InnerProduct': {
                        expect(reader, resource.name);
                        resource.weight = raw(reader);
                        resource.bias = raw(reader);
                        if (resource.weight.dataType === 'int8') {
                            resource.scale = raw(reader);
                        }
                        break;
                    }
                    case 'PReLU': {
                        expect(reader, resource.name);
                        resource.slope = raw(reader);
                        break;
                    }
                    case 'Add':
                    case 'Div':
                    case 'Mul':
                    case 'Sub':
                    case 'MatMul': {
                        resource.slope = raw(reader);
                        break;
                    }
                    case 'BatchNormCxx':
                    case 'InstBatchNormCxx':
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
                    case 'Gather': {
                        // reader.expect(resource.name);
                        const has_data = reader.int32();
                        if (has_data) {
                            resource.data = raw(reader);
                        }
                        const has_indices = reader.int32();
                        if (has_indices) {
                            resource.indices = raw(reader);
                        }
                        break;
                    }
                    default: {
                        throw new tnn.Error(`Unsupported layer resource type '${resource.type}'.`);
                    }
                }
                this.layerResources[i] = resource;
            }
            if (reader.position !== reader.length) {
                throw new tnn.Error("Invalid blob size.");
            }
        }
    }

    read(name) {
        const resource = this.layerResources.shift();
        if (resource && resource.name !== name) {
            throw new tnn.Error(`Invalid blob layer name '${name}'.`);
        }
        return resource;
    }
};

tnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TNN model.';
    }
};

export const ModelFactory = tnn.ModelFactory;
