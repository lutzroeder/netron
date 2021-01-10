/* jshint esversion: 6 */

// Experimental

var uff = uff || {};
var protobuf = protobuf || require('./protobuf');

uff.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'uff' || extension === 'pb') {
            const tags = context.tags('pb');
            if (tags.size > 0 &&
                tags.has(1) && tags.get(1) === 0 &&
                tags.has(2) && tags.get(2) === 0 &&
                tags.has(3) && tags.get(3) === 2 &&
                tags.has(4) && tags.get(4) === 2 &&
                tags.has(5) && tags.get(5) === 2) {
                return true;
            }
        }
        if (extension === 'pbtxt' || identifier.toLowerCase().endsWith('.uff.txt')) {
            const tags = context.tags('pbtxt');
            if (tags.has('version') && tags.has('descriptors') && tags.has('graphs')) {
                return true;
            }
        }
        return false;
    }

    open(context) {
        return context.require('./uff-proto').then(() => {
            let meta_graph = null;
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            if (extension === 'pbtxt' || identifier.toLowerCase().endsWith('.uff.txt')) {
                try {
                    uff.proto = protobuf.get('uff').uff;
                    const buffer = context.stream.peek();
                    const reader = protobuf.TextReader.create(buffer);
                    meta_graph = uff.proto.MetaGraph.decodeText(reader);
                }
                catch (error) {
                    throw new uff.Error('File text format is not uff.MetaGraph (' + error.message + ').');
                }
            }
            else {
                try {
                    uff.proto = protobuf.get('uff').uff;
                    const buffer = context.stream.peek();
                    const reader = protobuf.Reader.create(buffer);
                    meta_graph = uff.proto.MetaGraph.decode(reader);
                }
                catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw  new uff.Error('File format is not uff.MetaGraph (' + message.replace(/\.$/, '') + ').');
                }
            }
            return uff.Metadata.open(context).then((metadata) => {
                return new uff.Model(metadata, meta_graph);
            });
        });
    }
};

uff.Model = class {

    constructor(metadata, meta_graph) {
        this._version = meta_graph.version;
        this._imports = meta_graph.descriptors.map((descriptor) => descriptor.id + ' v' + descriptor.version.toString()).join(', ');
        const references = new Map(meta_graph.referenced_data.map((item) => [ item.key, item.value ]));
        for (const graph of meta_graph.graphs) {
            for (const node of graph.nodes) {
                for (const field of node.fields) {
                    if (field.value.type === 'ref' && references.has(field.value.ref)) {
                        field.value = references.get(field.value.ref);
                    }
                }
            }
        }
        this._graphs = meta_graph.graphs.map((graph) => new uff.Graph(metadata, graph));
    }

    get format() {
        return 'UFF' + (this._version ? ' v' + this._version.toString() : '');
    }

    get imports() {
        return this._imports;
    }

    get graphs() {
        return this._graphs;
    }
};

uff.Graph = class {

    constructor(metadata, graph) {
        this._name = graph.id;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const args = new Map();
        const inputCountMap = new Map();
        for (const node of graph.nodes) {
            for (const input of node.inputs) {
                inputCountMap.set(input, inputCountMap.has(input) ? inputCountMap.get(input) + 1 : 1);
                args.set(input, new uff.Argument(input));
            }
            if (!args.has(node.id)) {
                args.set(node.id, new uff.Argument(node.id));
            }
        }
        for (let i = graph.nodes.length - 1; i >= 0; i--) {
            const node = graph.nodes[i];
            if (node.operation === 'Const' && node.inputs.length === 0 && inputCountMap.get(node.id) === 1) {
                const fields = {};
                for (const field of node.fields) {
                    fields[field.key] = field.value;
                }
                if (fields.dtype && fields.shape && fields.values) {
                    const tensor = new uff.Tensor(fields.dtype.dtype, fields.shape, fields.values);
                    args.set(node.id, new uff.Argument(node.id, tensor.type, tensor));
                    graph.nodes.splice(i, 1);
                }
            }
            if (node.operation === 'Input' && node.inputs.length === 0) {
                const fields = {};
                for (const field of node.fields) {
                    fields[field.key] = field.value;
                }
                const type = fields.dtype && fields.shape ? new uff.TensorType(fields.dtype.dtype, fields.shape) : null;
                args.set(node.id, new uff.Argument(node.id, type, null));
            }
        }

        for (const node of graph.nodes) {
            if (node.operation === 'Input') {
                this._inputs.push(new uff.Parameter(node.id, [ args.get(node.id) ]));
                continue;
            }
            if (node.operation === 'MarkOutput' && node.inputs.length === 1) {
                this._outputs.push(new uff.Parameter(node.id, [ args.get(node.inputs[0]) ]));
                continue;
            }
            this._nodes.push(new uff.Node(metadata, node, args));
        }
    }

    get name() {
        return this._name;
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

uff.Parameter = class {

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

uff.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new uff.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

uff.Node = class {

    constructor(metadata, node, args) {
        this._name = node.id;
        this._operation = node.operation;
        this._metadata = metadata.type(node.operation);
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        const schema = metadata.type(node.operation);
        if (node.inputs && node.inputs.length > 0) {
            let inputIndex = 0;
            if (schema && schema.inputs) {
                for (const inputSchema of schema.inputs) {
                    if (inputIndex < node.inputs.length || inputSchema.optional !== true) {
                        const inputCount = inputSchema.list ? (node.inputs.length - inputIndex) : 1;
                        const inputArguments = node.inputs.slice(inputIndex, inputIndex + inputCount).map((id) => {
                            return args.get(id);
                        });
                        inputIndex += inputCount;
                        this._inputs.push(new uff.Parameter(inputSchema.name, inputArguments));
                    }
                }
            }
            this._inputs.push(...node.inputs.slice(inputIndex).map((id, index) => {
                const inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString();
                return new uff.Parameter(inputName, [ args.get(id) ]);
            }));
        }

        this._outputs.push(new uff.Parameter('output', [
            args.get(node.id)
        ]));

        for (const field of node.fields) {
            this._attributes.push(new uff.Attribute(metadata.attribute(this._operation, field.key), field.key, field.value));
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._operation;
    }

    get metadata() {
        return this._metadata;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }
};

uff.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        switch(value.type) {
            case 's': this._value = value.s; this._type = 'string'; break;
            case 's_list': this._value = value.s_list; this._type = 'string[]'; break;
            case 'd': this._value = value.d; this._type = 'float64'; break;
            case 'd_list': this._value = value.d_list.val; this._type = 'float64[]'; break;
            case 'b': this._value = value.b; this._type = 'boolean'; break;
            case 'b_list': this._value = value.b_list; this._type = 'boolean[]'; break;
            case 'i': this._value = value.i; this._type = 'int64'; break;
            case 'i_list': this._value = value.i_list.val; this._type = 'int64[]'; break;
            case 'blob': this._value = value.blob; break;
            case 'ref': this._value = value.ref; this._type = 'ref'; break;
            case 'dtype': this._value = new uff.TensorType(value.dtype, null).dataType; this._type = 'uff.DataType'; break;
            case 'dtype_list': this._value = value.dtype_list.map((type) => new uff.TensorType(type, null).dataType); this._type = 'uff.DataType[]'; break;
            case 'dim_orders': this._value = value.dim_orders; break;
            case 'dim_orders_list': this._value = value.dim_orders_list.val; break;
            default: throw new uff.Error("Unknown attribute '" + name + "' value '" + JSON.stringify(value) + "'.");
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
        return true;
    }
};

uff.Tensor = class {

    constructor(dataType, shape, values) {
        this._type = new uff.TensorType(dataType, shape);
        switch (values.type) {
            case 'blob': this._data = values.blob; break;
            default: throw new uff.Error("Unknown values format '" + JSON.stringify(values.type) + "'.");
        }
    }

    get kind() {
        return 'Const';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
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
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        if (this._data.length > 8 &&
            this._data[0] === 0x28 && this._data[1] === 0x2e && this._data[2] === 0x2e && this._data[3] === 0x2e &&
            this._data[this._data.length - 1] === 0x29 && this._data[this._data.length - 2] === 0x2e && this._data[this._data.length - 3] === 0x2e && this._data[this._data.length - 4] === 0x2e) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        if (this._type.dataType === '?') {
            context.state = 'Tensor data type is unknown.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length == 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        const results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.data.getInt16(context.index));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.data.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    default:
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

uff.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case uff.proto.DataType.DT_INT8: this._dataType = 'int8'; break;
            case uff.proto.DataType.DT_INT16: this._dataType = 'int16'; break;
            case uff.proto.DataType.DT_INT32: this._dataType = 'int32'; break;
            case uff.proto.DataType.DT_INT64: this._dataType = 'int64'; break;
            case uff.proto.DataType.DT_FLOAT16: this._dataType = 'float16'; break;
            case uff.proto.DataType.DT_FLOAT32: this._dataType = 'float32'; break;
            case 7: this._dataType = '?'; break;
            default:
                throw new uff.Error("Unknown data type '" + JSON.stringify(dataType) + "'.");
        }
        this._shape = shape ? new uff.TensorShape(shape) : null;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

uff.TensorShape = class {

    constructor(shape) {
        if (shape.type !== 'i_list') {
            throw new uff.Error("Unknown shape format '" + JSON.stringify(shape.type) + "'.");
        }
        this._dimensions = shape.i_list.val;
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

uff.Metadata = class {

    static open(context) {
        if (uff.Metadata._metadata) {
            return Promise.resolve(uff.Metadata._metadata);
        }
        return context.request('uff-metadata.json', 'utf-8', null).then((data) => {
            uff.Metadata._metadata = new uff.Metadata(data);
            return uff.Metadata._metadata;
        }).catch(() => {
            uff.Metadata._metadata = new uff.Metadata(null);
            return uff.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeCache = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(type + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
    }
};

uff.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading UFF model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = uff.ModelFactory;
}
