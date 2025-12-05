import * as flatbuffers from './flatbuffers.js';
import * as zip from './zip.js';

const espdl = {};

espdl.ModelFactory = class {

    async match(context) {
        // Check by file extension
        const identifier = context.identifier;
        const extension = identifier.lastIndexOf('.') > 0 ? identifier.split('.').pop().toLowerCase() : '';
        if (extension === 'espdl') {
            // Check for EDL2 header using stream directly
            const stream = context.stream;
            if (stream && stream.length >= 16) {
                const buffer = stream.peek(16); // Peek first 16 bytes
                const header = String.fromCharCode(...buffer.slice(0, 4));
                if (header === 'EDL2') {
                    // We'll set a custom context type, but we need to skip header later
                    // For now, just indicate match
                    return context.set('espdl.binary', null); // We'll read the buffer in open method
                }
            }
        }

        return null;
    }

    async open(context) {
        try {
            const schemaModule = await context.require('./espdl-schema');
            espdl.schema = schemaModule.espdl;
            if (!espdl.schema || !espdl.schema.Model) {
                throw new espdl.Error(`Failed to load ESPDL schema: Model is ${espdl.schema?.Model ? 'defined' : 'undefined'}`);
            }
        } catch (error) {
            throw new espdl.Error(`Failed to load ESPDL schema: ${error.message}`);
        }
        let model = null;
        const attachments = new Map();
        if (context.type == 'espdl.binary') {
            // Read from stream directly
            const stream = context.stream;
            if (!stream) {
                throw new espdl.Error('No stream available for ESP-DL binary file.');
            }

            // Read 16-byte header
            const headerBuffer = stream.read(16);
            if (headerBuffer.length < 16) {
                throw new espdl.Error('Invalid ESP-DL file: header too short.');
            }

            // Verify magic
            const magic = String.fromCharCode(...headerBuffer.slice(0, 4));
            if (magic !== 'EDL2') {
                throw new espdl.Error(`Invalid ESP-DL magic: ${magic}, expected EDL2`);
            }

            // Read the rest of the file as FlatBuffers data
            const data = stream.read(stream.length - stream.position);

            // Create flatbuffers reader
            const reader = flatbuffers.BinaryReader.open(data);
            if (!reader) {
                throw new espdl.Error('Invalid FlatBuffers data after header.');
            }
            try {
                model = espdl.schema.Model.create(reader);
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new espdl.Error(`File format is not espdl.Model (${message.replace(/\.$/, '')}).`);
            }
        } else {
            throw new espdl.Error(`Unsupported ESP-DL format '${context.type}'.`);
        }
        const stream = context.stream;
        const metadata = await espdl.Metadata.open(context);
        return new espdl.Model(metadata, model, stream);
    }
};

espdl.Model = class {

    constructor(metadata, model, stream) {
        this.format = 'espdl';
        this.description = model.doc_string || '';
        this.modules = [];
        this.metadata = [];
        // Parse model metadata props
        if (model.metadata_props) {
            for (const prop of model.metadata_props) {
                this.metadata.push(new espdl.Argument(prop.key, prop.value));
            }
        }
        // Process graph
        const graph = model.graph;
        if (graph) {
            const graphObj = new espdl.Graph(metadata, graph, model, stream);
            this.modules.push(graphObj);
        }
    }
};

espdl.Graph = class {

    constructor(metadata, graph, model, stream) {
        this.name = graph.name || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        this.signatures = [];

        // Create context for tensor management
        const context = new espdl.Context(graph);

        // Process nodes with context and metadata
        if (graph.node) {
            for (let i = 0; i < graph.node.length; i++) {
                const node = graph.node[i];
                const nodeObj = new espdl.Node(metadata, context, node, i.toString());
                this.nodes.push(nodeObj);
            }
        }

        // Process inputs (only non-initializer inputs)
        if (graph.input) {
            for (let i = 0; i < graph.input.length; i++) {
                const valueInfo = graph.input[i];
                const tensor = context.initializer(valueInfo.name);
                if (!tensor) {
                    const value = context.value(valueInfo.name);
                    const values = value ? [value] : [];
                    const argument = new espdl.Argument(valueInfo.name, values);
                    this.inputs.push(argument);
                }
            }
        }

        // Process outputs
        if (graph.output) {
            for (let i = 0; i < graph.output.length; i++) {
                const valueInfo = graph.output[i];
                const value = context.value(valueInfo.name);
                const values = value ? [value] : [];
                const argument = new espdl.Argument(valueInfo.name, values);
                this.outputs.push(argument);
            }
        }
    }
};

espdl.Node = class {

    constructor(metadata, context, node, identifier) {
        this.name = node.name || '';
        this.identifier = identifier;

        // Get operator type from metadata
        const opType = node.op_type;
        this.type = metadata ? metadata.type('espdl', opType) : null;
        if (!this.type) {
            this.type = { name: opType };
        }

        this.inputs = [];
        this.outputs = [];
        this.attributes = [];

        // Map inputs using context and metadata
        if (node.input) {
            for (let i = 0; i < node.input.length;) {
                const inputMeta = this.type && Array.isArray(this.type.inputs) && i < this.type.inputs.length ? this.type.inputs[i] : { name: i.toString() };
                const count = inputMeta.list ? node.input.length - i : 1;
                const list = node.input.slice(i, i + count);
                const values = list.map((inputName) => {
                    if (!inputName) return null; // Skip empty names
                    return context.value(inputName);
                }).filter(v => v);
                const argument = new espdl.Argument(inputMeta.name, values);
                this.inputs.push(argument);
                i += count;
            }
        }

        // Map outputs using context and metadata
        if (node.output) {
            for (let i = 0; i < node.output.length;) {
                const outputMeta = this.type && Array.isArray(this.type.outputs) && i < this.type.outputs.length ? this.type.outputs[i] : { name: i.toString() };
                const count = outputMeta.list ? node.output.length - i : 1;
                const list = node.output.slice(i, i + count);
                const values = list.map((outputName) => {
                    // Get or create tensor for output
                    if (!outputName) return null; // Skip empty names
                    let tensor = context.value(outputName);
                    if (!tensor) {
                        // Check if we have value_info for this tensor
                        // If not, create a basic tensor with just the name
                        const tensorObj = { name: outputName };
                        tensor = new espdl.Tensor(context._values.size, tensorObj, null);
                        context._values.set(outputName, tensor);
                    }
                    return tensor;
                }).filter(v => v);
                const argument = new espdl.Argument(outputMeta.name, values);
                this.outputs.push(argument);
                i += count;
            }
        }

        // Process attributes
        if (node.attribute) {
            for (const attr of node.attribute) {
                const attrObj = new espdl.Attribute(attr);
                this.attributes.push(attrObj);
            }
        }
    }
};

espdl.Tensor = class {

    constructor(index, tensor, stream) {
        this.identifier = index.toString();
        this.name = tensor.name || '';
        if (this.name === undefined) {
            this.name = '';
        }
        this.type = tensor.data_type !== undefined ? new espdl.TensorType(tensor) : null;
        this.category = '';
        this.encoding = '<'; // little-endian assumption
        // TODO: load tensor data from stream if external
        this._data = null;
        // Reference to initializer tensor (for weight display)
        this.initializer = null; // Will be set to self for initializers

        // For initializers or tensors with data
        if (tensor.raw_data && tensor.raw_data.length > 0) {
            // raw_data is array of AlignedBytes
            // Note: AlignedBytes.bytes is undefined in current implementation
            // So we cannot extract data from raw_data
            this._data = null;
        } else if (tensor.float_data && tensor.float_data.length > 0) {
            this._data = new Float32Array(tensor.float_data);
        } else if (tensor.int32_data && tensor.int32_data.length > 0) {
            this._data = new Int32Array(tensor.int32_data);
        } else if (tensor.int64_data && tensor.int64_data.length > 0) {
            this._data = new BigInt64Array(tensor.int64_data);
        } else if (tensor.string_data && tensor.string_data.length > 0) {
            this._data = tensor.string_data;
        }

        // For intermediate tensors (outputs), we might not have type info
        if (!this.type) {
            let shapeDims = [];
            // Check if we have dims directly on tensor
            if (tensor.dims) {
                shapeDims = Array.from(tensor.dims).map(d => Number(d));
            }
            // Check if this is a ValueInfo object
            else if (tensor.value_info_type !== undefined) {
                const dims = espdl.Utility.getShapeFromValueInfo(tensor);
                if (dims) {
                    shapeDims = dims;
                }
            }

            if (shapeDims.length > 0) {
                // Create a basic type with shape only
                this.type = {
                    _dataType: '?',
                    _shape: new espdl.TensorShape(shapeDims),
                    get dataType() {
                        return this._dataType;
                    },
                    get shape() {
                        return this._shape;
                    },
                    toString: function() {
                        return this._dataType + this._shape.toString();
                    }
                };
            }
        }
    }

    get values() {
        return this._data;
    }
};

espdl.TensorType = class {

    constructor(tensor) {
        // Check if this is a ValueInfo object (has value_info_type)
        if (tensor.value_info_type !== undefined) {
            // Extract data type from ValueInfo
            const dataType = espdl.Utility.getDataTypeFromValueInfo(tensor);
            this._dataType = dataType !== undefined ? espdl.Utility.dataType(dataType) : '?';

            // Extract shape from ValueInfo
            const shapeDims = espdl.Utility.getShapeFromValueInfo(tensor);
            this._shape = new espdl.TensorShape(shapeDims || []);
        } else {
            // Regular tensor object
            this._dataType = tensor.data_type !== undefined ? espdl.Utility.dataType(tensor.data_type) : '?';
            this._shape = new espdl.TensorShape(tensor.dims ? Array.from(tensor.dims).map(d => Number(d)) : []);
        }
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

espdl.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length === 0) {
            return '';
        }
        return `[${this._dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

espdl.Attribute = class {

    constructor(attr) {
        this.name = attr.name || '';
        this.value = null;
        this.type = null;
        this.visible = true;
        // Convert attribute value based on attr_type
        switch (attr.attr_type) {
            case espdl.schema.AttributeType.FLOAT:
                this.value = attr.f ? attr.f.f : 0;
                break;
            case espdl.schema.AttributeType.INT:
                this.value = attr.i ? Number(attr.i.i) : 0;
                break;
            case espdl.schema.AttributeType.STRING:
                this.value = attr.s ? new TextDecoder('utf-8').decode(attr.s) : '';
                break;
            case espdl.schema.AttributeType.TENSOR:
                this.value = '<Tensor>';
                break;
            case espdl.schema.AttributeType.FLOATS:
                this.value = attr.floats ? Array.from(attr.floats) : [];
                break;
            case espdl.schema.AttributeType.INTS:
                this.value = attr.ints ? Array.from(attr.ints).map(i => Number(i)) : [];
                break;
            case espdl.schema.AttributeType.STRINGS:
                this.value = attr.strings ? attr.strings.map(s => new TextDecoder('utf-8').decode(s)) : [];
                break;
            default:
                this.value = '<Unknown>';
        }
    }
};

espdl.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

espdl.Utility = class {

    static dataType(type) {
        if (!espdl.Utility._tensorTypes) {
            espdl.Utility._tensorTypes = new Map(Object.entries(espdl.schema.TensorDataType).map(([key, value]) => [value, key.toLowerCase()]));
        }
        return espdl.Utility._tensorTypes.has(type) ? espdl.Utility._tensorTypes.get(type) : '?';
    }

    // Extract shape from ValueInfo object
    static getShapeFromValueInfo(valueInfo) {
        if (!valueInfo || !valueInfo.value_info_type) {
            return null;
        }
        const typeInfo = valueInfo.value_info_type;
        if (!typeInfo.value) {
            return null;
        }
        // Check if value is a TensorTypeAndShape object (union type 1)
        // In FlatBuffers, union objects don't have a 'type' property
        // They are directly the object if the union type matches
        const tensorType = typeInfo.value;
        if (!tensorType.shape) {
            return null;
        }
        const shape = tensorType.shape;
        if (!shape.dim || shape.dim.length === 0) {
            return [];
        }
        // Convert Dimension objects to dimension values
        const dimensions = [];
        for (const dim of shape.dim) {
            if (dim && dim.value) {
                const dimValue = dim.value;
                // DimensionValueType enum values: UNKNOWN=0, VALUE=1, PARAM=2
                if (dimValue.dim_type === 1) { // VALUE
                    dimensions.push(Number(dimValue.dim_value));
                } else if (dimValue.dim_type === 2) { // PARAM
                    dimensions.push(dimValue.dim_param || '?');
                } else {
                    dimensions.push('?');
                }
            } else {
                dimensions.push('?');
            }
        }
        return dimensions;
    }

    // Extract data type from ValueInfo object
    static getDataTypeFromValueInfo(valueInfo) {
        if (!valueInfo || !valueInfo.value_info_type) {
            return undefined;
        }
        const typeInfo = valueInfo.value_info_type;
        if (!typeInfo.value) {
            return undefined;
        }
        const tensorType = typeInfo.value;
        return tensorType.elem_type !== undefined ? tensorType.elem_type : undefined;
    }
};

espdl.Metadata = class {

    static async open(context) {
        if (!espdl.Metadata._metadata) {
            let data = null;
            try {
                data = await context.request('espdl-metadata.json');
            } catch {
                // continue regardless of error
            }
            espdl.Metadata._metadata = new espdl.Metadata(data);
        }
        return espdl.Metadata._metadata;
    }

    constructor(data) {
        this._types = new Map();
        if (data) {
            const types = JSON.parse(data);
            for (const type of types) {
                if (!this._types.has(type.module)) {
                    this._types.set(type.module, new Map());
                }
                const typesByModule = this._types.get(type.module);
                if (!typesByModule.has(type.name)) {
                    typesByModule.set(type.name, []);
                }
                typesByModule.get(type.name).push(type);
            }
        }
    }

    type(domain, name) {
        domain = domain || 'espdl';
        if (this._types.has(domain)) {
            const types = this._types.get(domain);
            if (types.has(name)) {
                // Return the highest version
                const typeList = types.get(name);
                return typeList.reduce((max, current) =>
                    (current.version > max.version) ? current : max, typeList[0]);
            }
        }
        return null;
    }
};

espdl.Context = class {

    constructor(graph) {
        this._initializers = new Map();
        this._tensors = new Map();
        this._values = new Map();

        // initializers
        if (graph.initializer) {
            for (let i = 0; i < graph.initializer.length; i++) {
                const tensor = graph.initializer[i];
                const tensorName = tensor.name || '';
                if (tensorName) {
                    const tensorObj = new espdl.Tensor(i, tensor, null);
                    tensorObj.initializer = tensorObj; // Self-reference for initializer
                    this._initializers.set(tensorName, tensorObj);
                    this._values.set(tensorName, tensorObj);
                }
            }
        }

        // inputs
        if (graph.input) {
            for (const valueInfo of graph.input) {
                const valueName = valueInfo.name || '';
                if (valueName && !this._values.has(valueName)) {
                    const tensor = new espdl.Tensor(this._values.size, valueInfo, null);
                    this._values.set(valueName, tensor);
                }
            }
        }

        // outputs
        if (graph.output) {
            for (const valueInfo of graph.output) {
                const valueName = valueInfo.name || '';
                if (valueName && !this._values.has(valueName)) {
                    const tensor = new espdl.Tensor(this._values.size, valueInfo, null);
                    this._values.set(valueName, tensor);
                }
            }
        }

        // value_info (intermediate tensors with type information)
        if (graph.value_info) {
            for (const valueInfo of graph.value_info) {
                const valueName = valueInfo.name || '';
                if (valueName && !this._values.has(valueName)) {
                    const tensor = new espdl.Tensor(this._values.size, valueInfo, null);
                    this._values.set(valueName, tensor);
                }
            }
        }
    }

    value(name) {
        return this._values.get(name) || null;
    }

    tensor(name) {
        return this._values.get(name) || null;
    }

    initializer(name) {
        return this._initializers.get(name) || null;
    }
};

espdl.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ESP-DL model.';
    }
};

export const ModelFactory = espdl.ModelFactory;