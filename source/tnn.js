
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
                        if (header.length === 3 || (header.length >= 4 && (header[3] === '4206624770' || header[3] === '4206624772'))) {
                            context.type = 'tnn.model';
                            return;
                        }
                    }
                }
            } catch {
                // continue regardless of error
            }
        }
        if (stream && identifier.endsWith('.tnnmodel')) {
            for (const signature of [[0x02, 0x00, 0xbc, 0xfa], [0x04, 0x00, 0xbc, 0xfa]]) {
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
                    return new tnn.Model(metadata, context, content);
                } catch {
                    return new tnn.Model(metadata, context, null);
                }
            }
            case 'tnn.params': {
                const name = `${context.identifier.substring(0, context.identifier.length - 9)}.tnnproto`;
                const content = await context.fetch(name, null);
                return new tnn.Model(metadata, content, context);
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
        const resources = new tnn.LayerResourceReader();
        if (tnnmodel) {
            resources.read(tnnmodel);
        }
        const reader = new tnn.TextProtoReader(tnnproto.stream);
        reader.read();
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
            const argument = new tnn.Argument(input.name, [values.map(input.name, type)]);
            this.inputs.push(argument);
        }
        for (const output of reader.outputs) {
            const argument = new tnn.Argument(output.name, [values.map(output.name)]);
            this.outputs.push(argument);
        }
        for (const layer of reader.layers) {
            const node = new tnn.Node(metadata, resources, layer, values);
            this.nodes.push(node);
        }
    }
};

tnn.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type;
        this.visible = visible !== false;
    }
};

tnn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new tnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer || null;
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
            const metadata = attributeSchemas.shift();
            let name = '';
            let value = null;
            let type = '';
            let visible = true;
            if (metadata && metadata.type === 'int32[]' && metadata.size) {
                const size = layer.attr[metadata.size];
                value = attributes.splice(0, size).map((attribute) => parseInt(attribute.value, 10));
            } else {
                const attribute = attributes.shift();
                name = attribute.key;
                value = attribute.value;
            }
            if (metadata) {
                name = metadata.name;
                if (metadata.type) {
                    type = metadata.type;
                }
                switch (type) {
                    case '':
                        break;
                    case 'int32':
                        value = parseInt(value, 10);
                        break;
                    case 'float32':
                        value = parseFloat(value);
                        break;
                    case 'int32[]':
                        value = value.map((v) => parseInt(v, 10));
                        break;
                    default:
                        throw new tnn.Error(`Unsupported attribute type '${type}'.`);
                }
                if (metadata && metadata.visible === false) {
                    visible = false;
                } else if (metadata.default !== undefined) {
                    if (value === metadata.default || (value && value.toString() === metadata.default.toString())) {
                        visible = false;
                    }
                }
            }
            const argument = new tnn.Argument(name, value, type, visible);
            this.attributes.push(argument);
        }

        const inputs = layer.inputs;
        let inputIndex = 0;
        if (this.type && this.type.inputs) {
            for (const inputDef of this.type.inputs) {
                if (inputIndex < inputs.length || inputDef.option !== 'optional') {
                    const inputCount = (inputDef.option === 'variadic') ? (inputs.length - inputIndex) : 1;
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id !== '' || inputDef.option !== 'optional').map((id) => values.map(id));
                    const argument = new tnn.Argument(inputDef.name, inputArguments);
                    this.inputs.push(argument);
                    inputIndex += inputCount;
                }
            }
        } else {
            this.inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) === 0) ? 'input' : (inputIndex + index).toString();
                return new tnn.Argument(inputName, [values.map(input)]);
            }));
        }

        const outputs = layer.outputs;
        let outputIndex = 0;
        if (this.type && this.type.outputs) {
            for (const outputDef of this.type.outputs) {
                if (outputIndex < outputs.length || outputDef.option !== 'optional') {
                    const outputCount = (outputDef.option === 'variadic') ? (outputs.length - outputIndex) : 1;
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => values.map(id));
                    const argument = new tnn.Argument(outputDef.name, outputArguments);
                    this.outputs.push(argument);
                    outputIndex += outputCount;
                }
            }
        } else {
            this.outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                const outputName = ((outputIndex + index) === 0) ? 'output' : (outputIndex + index).toString();
                return new tnn.Argument(outputName, [values.map(output)]);
            }));
        }
        const weight = (resource, name, shape) => {
            const initializer = resource[name];
            if (!initializer) {
                throw new tnn.Error(`Layer initializer'${resource.type}.${name}' not found '`);
            }
            const tensor = new tnn.Tensor(new tnn.TensorType(initializer.dataType, new tnn.TensorShape(shape)), initializer.value);
            const argument = new tnn.Argument(name, [values.map('', null, tensor)]);
            this.inputs.push(argument);
        };
        switch (this.type.name) {
            case 'Convolution':
            case 'ConvolutionDepthWise':
            case 'Deconvolution':
            case 'DeconvolutionDepthWise': {
                const resource = resources.get(this.name);
                if (resource) {
                    const num_output = parseInt(layer.attr['2'] || 0, 10);
                    const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                    const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                    const weight_data_size = resource.filter.length;
                    weight(resource, 'filter', [num_output, weight_data_size / (num_output * kernel_w * kernel_h), kernel_w, kernel_h]);
                    if (resource.bias) {
                        weight(resource, 'bias', [num_output]);
                    }
                    if (resource.quantized) {
                        weight(resource, 'quantized', [num_output]);
                    }
                }
                break;
            }
            case 'Conv3D':{
                const resource = resources.get(this.name);
                if (resource) {
                    const num_output = parseInt(layer.attr['2'] || 0, 10);
                    const kernel_w = parseInt(layer.attr['3'] || 0, 10);
                    const kernel_h = parseInt(layer.attr['4'] || kernel_w, 10);
                    const kernel_d = parseInt(layer.attr['5'] || kernel_w, 10);
                    const weight_data_size = resource.filter.length;
                    weight(resource, 'weight', [num_output, weight_data_size / (num_output * kernel_w * kernel_h  * kernel_d), kernel_w, kernel_h, kernel_d]);
                    if (resource.bias) {
                        weight(resources, 'bias', [num_output]);
                    }
                }
                break;
            }
            case 'InnerProduct': {
                const resource = resources.get(this.name);
                if (resource) {
                    const num_output = parseInt(layer.attr['0'] || 0, 10);
                    const weight_data_size = resource.weight.length;
                    weight(resource, 'weight', [num_output, weight_data_size / num_output]);
                    weight(resource, 'bias', [num_output]);
                    if (resource.weight.dataType === 'int8') {
                        weight(resource, 'scale', [num_output]);
                    }
                }
                break;
            }
            case 'PReLU': {
                const resource = resources.get(this.name);
                if (resource) {
                    weight(resource, 'slope', [resource.slope.length]);
                }
                break;
            }
            case 'BatchNormCxx':
            case 'InstBatchNormCxx': {
                const resource = resources.get(this.name);
                if (resource) {
                    weight(resource, 'scale', [resource.scale.length]);
                    weight(resource, 'bias', [resource.bias.length]);
                }
                break;
            }
            case 'Div':
            case 'Sub':
            case 'Add':
            case 'Mul':
            case 'MatMul': {
                if (this.inputs.length === 1) {
                    const resource = resources.get(this.name);
                    if (resource) {
                        const num_output = resource.slope.length;
                        weight(resource, 'slope', [num_output]);
                    }
                }
                break;
            }
            case 'HdrGuide': {
                const resource = resources.get(this.name);
                if (resource) {
                    const weight_size = resource.ccm_weight.length;
                    weight(resource, 'ccm_weight', [weight_size]);
                    weight(resource, 'ccm_bias', [weight_size]);
                    weight(resource, 'shifts', [weight_size]);
                    weight(resource, 'slopes', [weight_size]);
                    weight(resource, 'projection_weight', [weight_size]);
                    weight(resource, 'projection_bias', [weight_size]);
                }
                break;
            }
            case 'BlobScale': {
                const resource = resources.get(this.name);
                if (resource) {
                    const scale_data_size = resource.scale.length;
                    weight(resource, 'scale', [scale_data_size]);
                    weight(resource, 'bias', [scale_data_size]);
                }
                break;
            }
            case 'Gather': {
                const resource = resources.get(this.name);
                if (resource) {
                    if (resource.data) {
                        weight(resource, 'data', [resource.data.length]);
                    }
                    if (resource.indices) {
                        weight(resource, 'indices', [resource.indices.length]);
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

    constructor(stream) {
        this.stream = stream;
        this.inputs = [];
        this.outputs = [];
        this.layers = [];
    }

    read() {
        if (this.stream) {
            const reader = text.Reader.open(this.stream);
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
                        name,
                        data_type: ['float32', 'float16', 'int8', 'int32', 'bfloat16'][data_type_index],
                        shape: array.slice(0, -1).map((dim) => parseInt(dim, 10)),
                    };

                }
                return {
                    name,
                    data_type: 'float32',
                    shape: array.map((dim) => parseInt(dim, 10))
                };
            });
            lines.shift();
            this.outputs = split(lines.shift(), ' ', true, false).map((output) => {
                return { name: output };
            });
            lines.shift();
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
                            layer.attributes.push({ key, value });
                            count++;
                        }
                    }
                    this.layers.push(layer);
                }
            }
            delete this.stream;
        }
    }
};

tnn.LayerResourceReader = class {

    constructor() {
        this.resources = new Map();
    }

    read(context) {
        this.reader = context.read('binary');
        const magic_number = this.reader.uint32();
        if (magic_number !== 0xFABC0002 && magic_number !== 0xFABC0004) {
            throw new tnn.Error(`Invalid blob header signature '${magic_number}'.`);
        }
        const size = this.reader.int32() & 0x1FFFFFFF;
        for (let i = 0; i < size; i++) {
            const resource = {};
            resource.operator = this.reader.int32();
            resource.type = this.reader.string();
            resource.name = this.reader.string();
            switch (resource.type) {
                case 'Convolution':
                case 'ConvolutionDepthWise':
                case 'Deconvolution':
                case 'DeconvolutionDepthWise': {
                    this._expect(resource.name);
                    const bias = this.reader.int32();
                    resource.filter = this._read();
                    if (bias) {
                        resource.bias = this._read();
                    }
                    if (resource.filter.dataType === 'int8') {
                        resource.quantized = this._read();
                    }
                    break;
                }
                case 'Conv3D': {
                    this._expect(resource.name);
                    const bias = this.reader.int32();
                    resource.filter = this._read();
                    if (bias) {
                        resource.bias = this._read();
                    }
                    break;
                }
                case 'InnerProduct': {
                    this._expect(resource.name);
                    resource.weight = this._read();
                    resource.bias = this._read();
                    if (resource.weight.dataType === 'int8') {
                        resource.scale = this._read();
                    }
                    break;
                }
                case 'PReLU': {
                    this._expect(resource.name);
                    resource.slope = this._read();
                    break;
                }
                case 'Add':
                case 'Div':
                case 'Mul':
                case 'Sub':
                case 'MatMul': {
                    resource.slope = this._read();
                    break;
                }
                case 'BatchNormCxx':
                case 'InstBatchNormCxx':
                    resource.scale = this._read();
                    resource.bias = this._read();
                    break;
                case 'HdrGuide':
                    resource.ccm_weight = this._read();
                    resource.ccm_bias = this._read();
                    resource.shifts = this._read();
                    resource.slopes = this._read();
                    resource.projection_weight = this._read();
                    resource.projection_bias = this._read();
                    break;
                case 'BlobScale':
                    resource.scale = this._read();
                    resource.bias = this._read();
                    break;
                case 'Gather': {
                    // reader.expect(resource.name);
                    const has_data = this.reader.int32();
                    if (has_data) {
                        resource.data = this._read();
                    }
                    const has_indices = this.reader.int32();
                    if (has_indices) {
                        resource.indices = this._read();
                    }
                    break;
                }
                default: {
                    throw new tnn.Error(`Unsupported layer resource type '${resource.type}'.`);
                }
            }
            this.resources.set(resource.name, resource);
        }
        if (this.reader.position !== this.reader.length) {
            throw new tnn.Error("Invalid blob size.");
        }
        delete this.reader;
    }

    _read() {
        const magic_number = this.reader.uint32();
        if (magic_number !== 0xFABC0002 && magic_number !== 0xFABC0004) {
            throw new tnn.Error(`Invalid raw signature '${magic_number}'.`);
        }
        const data_type = this.reader.int32();
        if (data_type > 4) {
            throw new tnn.Error(`Unsupported data type '${data_type}'.`);
        }
        const length = this.reader.int32();
        if (length <= 0) {
            return null;
        }
        let dims = null;
        if (magic_number === 0xFABC0004) {
            const dim_size = this.reader.int32();
            dims = this.reader.read(dim_size * 4);
        }
        return {
            dataType: ['float32', 'float16', 'int8', 'int32', 'bfloat16'][data_type],
            length: length / [4, 2, 1, 4, 2][data_type],
            value: this.reader.read(length),
            shape: dims
        };
    }

    _expect(name) {
        const content = this.reader.string();
        if (name !== content) {
            throw new tnn.Error(`Invalid string '${content}' instead of '${name}'.`);
        }
    }

    get(name) {
        if (this.resources.size === 0) {
            return null;
        }
        if (!this.resources.has(name)) {
            throw new tnn.Error(`Invalid blob layer name '${name}'.`);
        }
        return this.resources.get(name);
    }
};

tnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TNN model.';
    }
};

export const ModelFactory = tnn.ModelFactory;
