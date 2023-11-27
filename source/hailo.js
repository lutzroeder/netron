// Experimental
const zip = require("./zip");

var hailo = {};

hailo.dataTypes = {
    "<u1": {
        name: "uint8",
        size: 8,
        constructor: Uint8Array,
    },
    "|u1": {
        name: "uint8",
        size: 8,
        constructor: Uint8Array,
    },
    "<u2": {
        name: "uint16",
        size: 16,
        constructor: Uint16Array,
    },
    "|i1": {
        name: "int8",
        size: 8,
        constructor: Int8Array,
    },
    "<i2": {
        name: "int16",
        size: 16,
        constructor: Int16Array,
    },
    "<u4": {
        name: "uint32",
        size: 32,
        constructor: Int32Array,
    },
    "<i4": {
        name: "int32",
        size: 32,
        constructor: Int32Array,
    },
    "<u8": {
        name: "uint64",
        size: 64,
        constructor: BigUint64Array,
    },
    "<i8": {
        name: "int64",
        size: 64,
        constructor: BigInt64Array,
    },
    "<f4": {
        name: "float32",
        size: 32,
        constructor: Float32Array
    },
    "<f8": {
        name: "float64",
        size: 64,
        constructor: Float64Array
    },
};

hailo.npyParser = {
    parse: (array_buffer) => {
        const header_length = new DataView(array_buffer.slice(8, 10)).getUint8(0);
        const offset_bytes = 10 + header_length;

        const hcontents = new TextDecoder("utf-8").decode(
            new Uint8Array(array_buffer.slice(10, 10 + offset_bytes))
        );

        const normalized_hcontents = hcontents
            .toLowerCase() // True -> true
            .replace(/'/g, '"')
            .replace("(", "[")
            .replace(/,*\),*/g, "]");

        const matched_string = normalized_hcontents.match(/\{"descr":\s?"(<u1|\|u1|<u2|\|i1|<i2|<u4|<i4|<u8|<i8|<f4|<f8)",\s?"fortran_order":\s?(false|true),\s?"shape":\s?(\[(\d+(?:,\s*\d+)*|)\]|\d+)\s?\}/);
        const [json] = matched_string;

        const header = JSON.parse(json);
        const shape = header.shape;
        const dtype = hailo.dataTypes[header.descr];
        const nums = new dtype["constructor"](
            array_buffer,
            offset_bytes
        );
        return {
            dtype: dtype.name,
            data: nums,
            shape,
            fortranOrder: header.fortran_order
        };
    }
};

hailo.ModelFactory = class {
    match(context) {
        return hailo.Container.open(context);
    }

    async open(context, target) {
        const metadata = await context.metadata('hailo-metadata.json');
        return new hailo.Model(metadata, target);
    }
};

hailo.Model = class {

    constructor(metadata, container) {
        const configuration = container.configuration;
        this.graphs = [ new hailo.Graph(metadata, configuration, container.npz) ];
        this.name = configuration && configuration.name || "";
        this.format = container.format + (container.metadata && container.metadata.sdk_version ? ' v' + container.metadata.sdk_version : '');
        this.metadata = [];
        if (container.metadata && container.metadata.state) {
            this.metadata.push({ name: 'state', value: container.metadata.state });
        }
    }
};

hailo.Graph = class {

    constructor(metadata, configuration, npz_configuration) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new hailo.Value(name, type || null, tensor);
            }
            if (!args.has(name)) {
                args.set(name, new hailo.Value(name, type || null, tensor || null));
            } else if (tensor) {
                throw new hailo.Error("Duplicate value '" + name + "'.");
            } else if (type && !type.equals(args.get(name).type)) {
                return new hailo.Value(name, type, null);
            }
            return args.get(name);
        };

        const layers = Object.entries(configuration.layers || {}).map((entry) => {
            entry[1].name = entry[0];
            return entry[1];
        });

        for (const layer of layers) {
            switch (layer.type) {
                case 'input_layer': {
                    const { name, output_shapes: [output_shape] = [] } = layer;

                    const type = output_shape ? new hailo.TensorType('?', new hailo.TensorShape(output_shape)) : null;
                    const argument = new hailo.Argument('input', [ arg(name, type) ]);
                    this.inputs.push(argument);
                    break;
                }

                case 'const_input': {
                    const { name, output_shapes: [output_shape] = [] } = layer;

                    const type = output_shape ? new hailo.TensorType('?', new hailo.TensorShape(output_shape)) : null;
                    const argument = new hailo.Argument('input', [ arg(name, type) ]);
                    this.inputs.push(argument);

                    break;
                }

                case 'output_layer': {
                    for (let i = 0; i < layer.input.length; i++) {
                        const shape = layer.input_shapes ? layer.input_shapes[i] : null;
                        const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
                        const argument = new hailo.Argument('output', [ arg(layer.input[i], type) ]);
                        this.outputs.push(argument);
                    }
                    break;
                }
                default: {
                    const layer_npz_data = npz_configuration[layer.name];
                    const node = new hailo.Node(metadata, layer, arg, layer_npz_data);
                    this.nodes.push(node);
                    break;
                }
            }
        }
    }
};

hailo.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

hailo.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hailo.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

hailo.Node = class {

    constructor(metadata, layer, arg, npz_data = {}) {
        const getParams = (params_array) => {
            return params_array.reduce((acc, [name, value]) => {
                const schema = metadata.attribute(layer.type, name) || {};
                if (schema.visible) {
                    const label = schema.label ? schema.label : name;
                    if (!npz_data[label]) {
                        const shape = new hailo.TensorShape(value);
                        const type = new hailo.TensorType(npz_data && npz_data[label] && npz_data[label].dataType, shape);
                        const tensor = new hailo.Tensor(type, npz_data && npz_data[label] && npz_data[label].buffer);
                        acc.push(new hailo.Argument(label, [ arg('', type, tensor) ]));
                    }
                }
                return acc;
            }, []);
        };

        const getNPZParams = (npz_params) => {
            const entries = npz_params ? Object.entries(npz_params) : [];
            return entries.map(([key, params]) => {
                const label = key;
                const shape = new hailo.TensorShape(params.shape);
                const type = new hailo.TensorType(params.dataType, shape);
                const tensor = new hailo.Tensor(type, params.buffer);
                return new hailo.Argument(label, [ arg('', type, tensor) ]);
            });
        };

        this.name = layer.name || '';
        this.type = metadata.type(layer.type);
        if (layer.type === 'activation') {
            this.type = Object.assign({}, this.type, { name: layer.params.activation || layer.name || '' });
        }
        this.inputs = layer.input.map((name, index) => {
            const shape = layer.input_shapes ? layer.input_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            return new hailo.Argument("input", [ arg(name, type) ]);
        });
        const layer_params = layer.params ? Object.entries(layer.params) : [];
        const params_list = getParams(layer_params);
        const params_from_npz = getNPZParams(npz_data);
        this.inputs = this.inputs.concat(params_list).concat(params_from_npz);
        this.outputs = (layer.output || []).map((_, index) => {
            const shape = layer.output_shapes ? layer.output_shapes[index] : null;
            const type = shape ? new hailo.TensorType('?', new hailo.TensorShape(shape)) : null;
            return new hailo.Argument("output", [ arg(layer.name, type) ]);
        });
        const attrs = Object.assign(layer.params || {}, { original_names: layer.original_names || [] });
        this.attributes = Object.entries(attrs).map((entry) => new hailo.Attribute(metadata.attribute(layer.type, entry[0]), entry[0], entry[1]));
        this.chain = [];
        if (layer && layer.params && layer.params.activation && layer.params.activation !== 'linear' && layer.type !== 'activation') {
            const activation = {
                type: layer.params.activation,
                name: layer.params.activation,
                input: [],
                output: []
            };
            const node = new hailo.Node(metadata, activation, arg);
            this.chain.push(node);
        }
    }
};

hailo.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.value = value;
        this.type = metadata && metadata.type ? metadata.type : '';
        if (metadata && metadata.visible === false) {
            this.visible = false;
        }
        if (name === 'original_names') {
            this.visible = false;
        }
    }
};

hailo.Tensor = class {

    constructor(type, values) {
        this.type = type;
        this.values = values;
    }
};

hailo.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return (this.dataType || '?') + this.shape.toString();
    }
};

hailo.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length) {
                return obj.dimensions.every((value, index) => this.dimensions[index] === value);
            }
            const a = this.dimensions.filter((value, index) => index === 0 || index === this.dimensions.length - 1 || value !== 1);
            const b = obj.dimensions.filter((value, index) => index === 0 || index === obj.dimensions.length - 1 || value !== 1);
            if (a.length === b.length) {
                return a.every((value, index) => b[index] === value);
            }
        }
        return false;
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return '[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

hailo.Container = class {

    static open(context) {
        const parts = context.identifier.split('.');
        const extension = parts.pop().toLowerCase();
        let format = '';
        const npz = {};
        let configuration = null;
        let metadata = null;
        switch (extension) {
            case 'hn': {
                format = 'Hailo NN';
                configuration = context.peek('json');
                break;
            }
            case 'har': {
                const read = (extension) => {
                    const entries = context.peek('tar');
                    const reg_exp = new RegExp(extension);
                    const search_result = [...entries].find(([name]) => {
                        return reg_exp.test(name);
                    });
                    const [, stream] = search_result;
                    if (stream) {
                        try {
                            return stream.peek();
                        } catch (err) {
                            // continue regardless of error
                        }
                    }
                    return null;
                };
                format = 'Hailo Archive';
                const decoder = new TextDecoder('utf-8');

                const metadata_buffer = read('.metadata.json');
                metadata = JSON.parse(decoder.decode(metadata_buffer));

                const hn_extension = '.hn';
                let npz_extension = '.npz';

                switch (metadata.state) {
                    case 'fp_optimized_model': {
                        npz_extension = '.fpo.npz';
                        break;
                    }
                    case 'quantized_model': {
                        npz_extension = '.q.npz';
                        break;
                    }
                    case 'compiled_model': {
                        npz_extension = '.q.npz';
                        break;
                    }
                    default:
                }

                const toArrayBuffer = (buffer) => {
                    const array_buffer = new ArrayBuffer(buffer.length);
                    const view = new Uint8Array(array_buffer);
                    for (let i = 0; i < buffer.length; ++i) {
                        view[i] = buffer[i];
                    }
                    return array_buffer;
                };

                const configuration_buffer = read(hn_extension);
                configuration = JSON.parse(decoder.decode(configuration_buffer));
                const npz_buffer = read(npz_extension);
                const archive = zip.Archive.open(npz_buffer);
                const allowed_inputs = ['kernel', 'bias', 'input_activation_bits', 'weight_bits', 'output_activation_bits', 'bias_decomposition'];
                if (archive) {
                    for (const [raw_key, raw_value] of archive.entries) {
                        const key = raw_key.split('.').slice(0, -1).join('.');
                        const match_result = key.match(/.*?(?=:[0-9])/);
                        const header_offset = 128;
                        if (match_result) {
                            const [keyPath] = match_result;
                            const key_path_array = keyPath.split('/');
                            const [network_name, layer_name, parameter_name] = key_path_array;
                            const array_buffer = toArrayBuffer(raw_value._buffer);
                            const header_params = hailo.npyParser.parse(array_buffer);
                            if (header_params) {
                                const key = `${network_name}/${layer_name}`;
                                if (!npz[key]) {
                                    npz[key] = {};
                                }
                                if (allowed_inputs.includes(parameter_name)) {
                                    npz[key][parameter_name] = {
                                        buffer: raw_value._buffer.slice(header_offset),
                                        dataType: header_params.dtype,
                                        shape: header_params.shape
                                    };
                                }
                            }
                        }
                    }
                }

                break;
            }
            default: {
                break;
            }
        }
        if (configuration && configuration.name && configuration.net_params && configuration.layers) {
            return new hailo.Container(format, configuration, metadata, npz);
        }
        return null;
    }

    constructor(format, configuration, metadata, npz) {
        this.format = format;
        this.configuration = configuration;
        this.metadata = metadata;
        this.npz = npz;
    }
};

hailo.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Hailo model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = hailo.ModelFactory;
}
