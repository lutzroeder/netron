// Experimental

var hn = hn || {};
var json = json || require('./json');

hn.DataTypes = {
    STRING: 'string',
    BOOLEAN: 'boolean',
    NUMBER: 'number',
    ARRAY: 'array'
};

hn.Stages = {
    native: 'Native Hailo Model',
    fp_optimized_model: 'Full Precision Hailo Model',
    quantized: 'Quantized Hailo Model'
};


hn.MetadataFile = 'hailo-metadata.json';

hn.FileExtensions = {
    HN: 'hn',
    TAR: 'tar',
    HAR: 'har',
    JSON: 'json',
    METADATA: 'metadata.json'
};

hn.Formats = {
    HN: 'HailoNN',
    HAR: 'Hailo Archive',
};

hn.ModelFactory = class {
    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        let json;
        if (extension === hn.FileExtensions.HN) {
            json = context.open(hn.FileExtensions.JSON);
        }
        if (extension === hn.FileExtensions.HAR) {
            json = this._getHNFromTAR(context);
        }

        const { name, net_params, layers } = json || {};
        if (name && net_params && layers) {
            return extension;
        }

        return undefined;
    }

    open(context, match) {
        return context.metadata(hn.MetadataFile).then((metadata) => {
            switch (match) {
                case hn.FileExtensions.HN: {
                    const configuration = context.open(hn.FileExtensions.JSON);
                    const graph_metadata = new hn.GraphMetadata(metadata);
                    return new hn.Model(graph_metadata, configuration, hn.Formats.HN);
                }

                case hn.FileExtensions.HAR: {
                    const configuration = this._getHNFromTAR(context);
                    const har_metadata = this._getMetadataFromTAR(context);
                    const graph_metadata = new hn.GraphMetadata(metadata);
                    return new hn.Model(graph_metadata, configuration, hn.Formats.HAR, har_metadata);
                }

                default: {
                    throw new hn.Error("Unsupported Hailo Network format '" + match + "'.");
                }
            }
        });
    }

    _getHNFromTAR(context) {
        const entries = [...context.entries(hn.FileExtensions.TAR)];
        const regExp = new RegExp(`.${hn.FileExtensions.HN}$`);
        const [, stream] = entries.find(([name]) => regExp.test(name));
        const buffer = stream.peek();
        const decoder = new TextDecoder('utf-8');
        const content = decoder.decode(buffer);
        return JSON.parse(content);
    }

    _getMetadataFromTAR(context) {
        const entries = [...context.entries(hn.FileExtensions.TAR)];
        const regExp = new RegExp(`.${hn.FileExtensions.METADATA}$`);
        const [, stream] = entries.find(([name]) => regExp.test(name));
        const buffer = stream.peek();
        const decoder = new TextDecoder('utf-8');
        const content = decoder.decode(buffer);
        return JSON.parse(content);
    }
};

hn.Model = class {
    constructor(metadata, configuration, format, har_metadata) {
        const getStageFromMetadata = (
            metadata
        ) => {
            const rawStage = metadata && metadata.state;
            return hn.Stages[rawStage];
        };

        this._graphs = [];
        this._graphs.push(new hn.Graph(metadata, configuration));
        this._name = configuration && configuration.name || "";
        const { net_params: { version, stage, dtype = [] } } = configuration;
        this._version = har_metadata && har_metadata.sdk_version || version || 0.0;
        this._format = format;
        this._description = `${getStageFromMetadata(har_metadata) || stage} of ${this._name}`;
        this._dtype = dtype;
    }

    get dtype() {
        return this._dtype;
    }

    get description() {
        return this._description;
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get version() {
        return this._version;
    }
};

hn.GraphMetadata = class {
    constructor(metadata) {
        this._metadata = metadata;
        this._types = new Map();
    }

    type(name) {
        if (this._types.has(name)) {
            return this._types.get(name);
        }
        return this._metadata.type(name);
    }

    attribute(type, name) {
        return this._metadata.attribute(type, name);
    }

    add(type, metadata) {
        this._types.set(type, metadata);
    }
};

hn.Graph = class {
    constructor(metadata, configuration) {
        this._inputs = [];
        this._outputs = [];

        const mapLayersObjectToArray = (layers_object = {}) => {
            const entries = Object.entries(layers_object);
            return entries.map(([layer_name, layer_object]) => {
                layer_object.name = layer_name;
                return layer_object;
            });
        };

        const mapLayerToNode = (layer) => {
            const { type } = layer;
            const layer_metadata = metadata.type(type);
            return new hn.Node(layer_metadata, layer);
        };

        const getNodes = (layers = []) => {
            const filtered_layers = layers.filter((layer) => {
                return !['input_layer', 'output_layer'].includes(layer.type);
            });
            return filtered_layers.map(mapLayerToNode);
        };

        const getInputs = (layers = []) => {
            const filtered_layers = layers.filter((layer) => {
                return ['input_layer'].includes(layer.type);
            });
            const result = [];
            filtered_layers.forEach(({ name, output, output_shapes: [output_shape] = [] }) => {
                return output.forEach(() => {
                    const param = new hn.Parameter(name, true, [
                        new hn.Argument(name, new hn.TensorType(hn.DataTypes.ARRAY, new hn.TensorShape(output_shape)))
                    ]);
                    result.push(param);
                });
            });
            return result;
        };

        const getOutputs = (layers) => {
            const filtered_layers = layers.filter((layer) => {
                return ['output_layer'].includes(layer.type);
            });
            const result = [];
            filtered_layers.forEach(({ name, input, input_shapes: [input_shape] = [] }) => {
                return input.forEach((item) => {
                    const param = new hn.Parameter(name, true, [
                        new hn.Argument(item, new hn.TensorType(hn.DataTypes.ARRAY, new hn.TensorShape(input_shape)))
                    ]);
                    result.push(param);
                });
            });
            return result;
        };

        const layers = mapLayersObjectToArray(configuration.layers);

        this._inputs = configuration && configuration.layers && getInputs(layers);
        this._outputs = configuration && configuration.layers && getOutputs(layers);
        this._nodes = configuration && configuration.layers && getNodes(layers);
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

hn.Parameter = class {
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

hn.Argument = class {
    constructor(name, type, initializer) {
        if (typeof name !== hn.DataTypes.STRING) {
            throw new hn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
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

hn.Node = class {
    constructor(layer_metadata, layer) {
        const getTypeByName = ((layer_metadata) => (attribute_name) => {
            return layer_metadata && layer_metadata.attributes && layer_metadata.attributes.find(({ name }) => {
                return name === attribute_name;
            }) || {
                name: attribute_name,
                type: "Layer",
                visible: false
            };
        })(layer_metadata);

        const getNodeAttributes = (layer) => {
            const {
                original_names = [],
                params = {}
            } = layer;

            const params_object = Object.assign(params, { original_names });

            return Object.entries(params_object).reduce((acc, [name, value]) => {
                const schema = getTypeByName(name);
                const attribute = new hn.Attribute(schema, name, value);
                acc.push(attribute);
                return acc;
            }, []);
        };

        const getNodeInputs = ({ input, input_shapes: [input_shape] = [], params }) => {
            const input_shapes = input.map((name) => {
                return new hn.Parameter("input", true, [
                    new hn.Argument(name, new hn.TensorType(hn.DataTypes.ARRAY, new hn.TensorShape(input_shape)))
                ]);
            });

            const getParams = (params_array = []) => {
                return params_array.reduce((acc, [name, value]) => {
                    const schema = getTypeByName(name);
                    if (schema.visible) {
                        const label = schema.label ? schema.label : name;
                        acc.push(new hn.Parameter(label, true, [
                            new hn.Argument(label, new hn.TensorType(
                                hn.DataTypes.ARRAY,
                                new hn.TensorShape(value, schema.type, schema.show_array_length)),
                            new hn.Tensor(schema.type, value, schema.show_array_length
                            ), value)
                        ]));
                    }
                    return acc;
                }, []);
            };

            const params_array = params ? Object.entries(params) : [];
            const params_list = getParams(params_array, layer_metadata);

            return input_shapes.concat(params_list);
        };

        const getNodeOutputs = ({ name, output = [], output_shapes: [output_shape = []] = [] }) => {
            return output.map(() => {
                return new hn.Parameter("output", true, [
                    new hn.Argument(name, new hn.TensorType(hn.DataTypes.ARRAY, new hn.TensorShape(output_shape)))
                ]);
            });
        };

        const getChain = (layer) => {
            return layer && layer.params && layer.params.activation && layer.params.activation !== 'linear' && layer.type !== 'activation' ? [new hn.Node({
                name: "activation",
                type: "activation",
                visible: true
            }, {
                type: layer.params.activation,
                name: layer.params.activation,
                input: [],
                output: []
            })] : [];
        };

        const getNodeType = (layer) => {
            if (layer.type === 'activation') {
                return layer.params.activation || layer.name || '';
            }
            return layer.type;
        };


        this._name = layer.name || '';
        this._type = {
            category: layer_metadata.type,
            name: getNodeType(layer),
            attributes: layer_metadata.attributes,
            description: layer_metadata.description
        };
        this._inputs = getNodeInputs(layer);
        this._outputs = getNodeOutputs(layer);
        this._attributes = getNodeAttributes(layer);
        this._chain = getChain(layer);
    }

    get type() {
        return this._type;
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

    get attributes() {
        return this._attributes;
    }

    get description() {
        return this._descripton;
    }

    get chain() {
        return this._chain;
    }
};

hn.Attribute = class {
    constructor(schema, name, value) {
        this._name = name;
        this._value = value;
        if (schema) {
            this._type = schema.type || '';
            this._visible = schema.visible || true;
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

    get visible() {
        return this._visible !== false;
    }
};

hn.Tensor = class {
    constructor(dataType, shape, show_array_length) {
        this._type = new hn.TensorType(dataType, new hn.TensorShape(shape, dataType, show_array_length), show_array_length);
    }

    get type() {
        return this._type;
    }

    get state() {
        if (this._type.show_array_length) {
            return `[${this._type.shape._dimensions.join(',')}]`;
        }
        return 'Tensor data not implemented.';
    }
};

hn.TensorType = class {
    constructor(dataType, shape, show_array_length) {
        this._dataType = dataType;
        this._shape = shape;
        this._show_array_length = show_array_length;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get show_array_length() {
        return this._show_array_length;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

hn.TensorShape = class {
    constructor(dimensions, type = hn.DataTypes.ARRAY, show_array_length) {
        this._dimensions = dimensions;
        this._type = type;
        this._show_array_length = show_array_length;
    }

    get dimensions() {
        if (this._show_array_length) {
            return [this._dimensions.length];
        }
        return this._dimensions;
    }

    get show_array_length() {
        return this._show_array_length;
    }

    get type() {
        return this._type;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length === 0) {
                return '';
            }

            return ` [${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};


hn.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading HN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = hn.ModelFactory;
}
