/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var acuity = acuity || {};
var json = json || require('./json');

acuity.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const tags = context.tags('json');
            if (tags.has('MetaData') && tags.has('Layers')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return acuity.Metadata.open(host).then((metadata) => {
            const extension = context.identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'json': {
                    const buffer = context.stream.peek();
                    const reader = json.TextReader.create(buffer);
                    const model = reader.read();
                    if (model && model.MetaData && model.Layers) {
                        return new acuity.Model(metadata, model);
                    }
                }
            }
        });
    }
};

acuity.Model = class {

    constructor(metadata, model, data, quantization) {
        this._graphs = [];
        this._name = model.MetaData.Name;
        this._format = 'Acuity ' + 'v' + model.MetaData.AcuityVersion;
        this._version = model.MetaData.AcuityVersion;
        this._graphs.push(new acuity.Graph(metadata, model, data, quantization));
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

    get graphs() {
        return this._graphs;
    }
};

acuity.Graph = class {

    constructor(metadata, model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const args = new Map();

        for (const layerName of Object.keys(model.Layers)) {
            const layer = model.Layers[layerName];
            for (const port of layer.outputs) {
                let shape = null;
                if (layer.op.toLowerCase() == 'input' ||
                    layer.op.toLowerCase() == 'variable') {
                    if (Object.prototype.hasOwnProperty.call(layer.parameters, 'shape')) {
                        shape = layer.parameters.shape;
                    }
                    else if (Object.prototype.hasOwnProperty.call(layer.parameters, 'size') &&
                            Object.prototype.hasOwnProperty.call(layer.parameters, 'channels')) {
                        const sizes = layer.parameters.size.split(' ');
                        shape = [0, parseInt(sizes[0]), parseInt(sizes[1]), layer.parameters.channels];
                    }
                }
                const portUrl = acuity.Utility.getTensorUrl(layerName, port);
                const tensorType = new acuity.TensorType(null, new acuity.TensorShape(shape));
                const arg = new acuity.Argument(portUrl, tensorType, null, null);
                args.set(portUrl, arg);
            }
        }

        for (const layerName of Object.keys(model.Layers)) {
            const layer = model.Layers[layerName];
            if (layer.op.toLowerCase() == 'input') {
                this._inputs.push(new acuity.Parameter(layerName, true,
                    [args.get(acuity.Utility.getTensorUrl(layerName, layer.outputs[0]))]));
            }
            else if (layer.op.toLowerCase() == 'output') {
                this._outputs.push(new acuity.Parameter(layerName, true,
                    [args.get(layer.inputs[0])]));
            }
            else {
                const schema = metadata.type(layer.op);
                this._nodes.push(new acuity.Node(schema, layerName, layer, args));
            }
        }

        new acuity.ShapeInference(this).process();
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

acuity.Node = class {

    constructor(schema, name, layer, args) {
        this._schema = schema;
        this._name = name;
        this._type = layer.op;
        this._inputs = [];
        this._constants = [];
        this._outputs = [];
        this._attributes = [];
        this._layer = layer;

        if (this._schema) {
            for (const attr of this._schema.attributes) {
                if (Object.prototype.hasOwnProperty.call(layer.parameters, attr.name)) {
                    this._attributes.push(new acuity.Attribute(attr.name, layer.parameters[attr.name]));
                }
                else {
                    this._attributes.push(new acuity.Attribute(attr.name, attr.default_value + ' (default)'));
                }
            }

            if (Object.prototype.hasOwnProperty.call(this._schema, 'constants')) {
                for (const constant of this._schema.constants) {
                    const tensorUrl = acuity.Utility.getTensorUrl(name, constant.name);
                    const tensorType = new acuity.TensorType(null, new acuity.TensorShape(), true);
                    args.set(tensorUrl, new acuity.Argument(tensorUrl, tensorType, null, null));
                    this._constants.push(constant);
                }
            }
        }

        for (let i = 0; i < layer.inputs.length; i++) {
            const portName = "in" + i;
            const port = layer.inputs[i];
            const arg = args.get(port);
            this._inputs.push(new acuity.Parameter(portName, true, [arg]));
        }

        for (const constant of this._constants) {
            const tensorUrl = acuity.Utility.getTensorUrl(name, constant.name);
            const arg = args.get(tensorUrl);
            this._inputs.push(new acuity.Parameter(constant.name, true, [arg]));
        }

        for (const port of layer.outputs) {
            const portUrl = acuity.Utility.getTensorUrl(name, port);
            const arg = args.get(portUrl);
            this._outputs.push(new acuity.Parameter(port, true, [arg]));
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._schema;
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

    computeOutputShape(func) {
        let inputShapeReady = true;
        for (const input of this.inputs) {
            if (!input.arguments[0].type.isConst && !input.arguments[0].type.shape.isShapeReady()) {
                inputShapeReady = false;
            }
        }

        if (inputShapeReady) {
            const parameters = this._layer.parameters;
            const inputsShapes = [];
            for (const input of this.inputs) {
                inputsShapes.push(input.arguments[0].type.shape.dimensions);
            }

            const outputsShape = func(inputsShapes, parameters);
            const outputLength = Math.min(outputsShape.length, this.outputs.length);
            for (let i = 0; i < outputLength; i++) {
                this.outputs[i].arguments[0].type.shape.dimensions = outputsShape[i];
            }
        }
    }
};

acuity.Attribute = class {

    constructor(name, value) {
        this._type = null;
        this._name = name;
        this._value = value;
        this._visible = true;
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
        return this._visible == false ? false : true;
    }
};

acuity.Parameter = class {

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

acuity.Argument = class {

    constructor(name, type, quantization, initializer) {
        if (typeof name !== 'string') {
            throw new acuity.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._quantization = quantization || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        return this._quantization;
    }

    set quantization(quantization) {
        this._quantization = quantization;
    }

    get initializer() {
        return this._initializer;
    }

    set initializer(initializer) {
        this._initializer = initializer;
    }
};

acuity.TensorType = class {

    constructor(dataType, shape, is_const) {
        this._dataType = dataType || 'float32';
        this._shape = shape;
        this._is_const = is_const || false;
    }

    get isConst() {
        return this._is_const;
    }

    get dataType() {
        return this._dataType;
    }

    set dataType(dataType) {
        this._dataType = dataType;
    }

    get shape() {
        return this._shape;
    }

    set shape(shape) {
        this._shape = shape;
    }

    set denotation(value) {
        this._denotation = value;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

acuity.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
        this._shapeReady = this._dimensions && this._dimensions.length > 0;
    }

    get dimensions() {
        return this._dimensions;
    }

    set dimensions(dimensions) {
        this._dimensions = dimensions;
        this._shapeReady = this._dimensions && this._dimensions.length > 0;
    }

    isShapeReady() {
        return this._shapeReady;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

acuity.Metadata = class {

    static open(host) {
        if (acuity.Metadata._metadata) {
            return Promise.resolve(acuity.Metadata._metadata);
        }
        return host.request(null, 'acuity-metadata.json', 'utf-8').then((data) => {
            acuity.Metadata._metadata = new acuity.Metadata(data);
            return acuity.Metadata._metadata;
        }).catch(() => {
            acuity.Metadata._metadata = new acuity.Metadata(null);
            return acuity.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    item.schema.name = item.name;
                    this._map.set(item.name, item.schema);
                }
            }
        }
    }

    type(name) {
        return this._map.has(name) ? this._map.get(name) : null;
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
    }
};

acuity.ShapeInference =  class {

    constructor(graph) {
        this._graph = graph;
        this._tensorProducerMap = new Map();
        for (const node of this._graph.nodes) {
            for (const output of node.outputs) {
                const tensorUrl = acuity.Utility.getTensorUrl(node.name, output.name);
                this._tensorProducerMap.set(tensorUrl, node);
            }
        }
        this._computeOutputShapeFuncs = new acuity.ComputeOutputShapeFuncs();
    }

    process() {
        for (const ouput of this._graph.outputs) {
            this._shapeInference(ouput);
        }
    }

    _shapeInference(output) {
        const tensorName = output.arguments[0].name;
        if (this._tensorProducerMap.has(tensorName)) {
            let inputShapeReady = true;
            const producer = this._tensorProducerMap.get(tensorName);
            for (const input of producer.inputs) {
                if (!input.arguments[0].type.isConst && !input.arguments[0].type.shape.isShapeReady()) {
                    this._shapeInference(input);
                    if (!input.arguments[0].type.shape.isShapeReady()) {
                        inputShapeReady = false;
                        break;
                    }
                }
            }

            if (inputShapeReady) {
                producer.computeOutputShape(this._computeOutputShapeFuncs.get(producer.type));
            }
        }
        else {
            //throw new acuity.Error("Unknown tensor '" + tensorName + "'.");
        }
    }
};

acuity.ComputeOutputShapeFuncs = class {
    constructor() {
        this._no_shape_changed = function(inputsShapes, parameters) {
            return [inputsShapes[0].slice()];
        };

        this._registry = new Map();
        this._registry['a_times_b_plus_c'] = this._no_shape_changed;
        this._registry['abs'] = this._no_shape_changed;
        this._registry['add'] = function(inputsShapes, parameters) { return []; };
        this._registry['addn'] = function(inputsShapes, parameters) { return []; };
        this._registry['argmin'] = function(inputsShapes, parameters) { return []; };
        //this._registry['base_input_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['batch2space'] = function(inputsShapes, parameters) { return []; };
        this._registry['batchnorm_single'] = function(inputsShapes, parameters) { return []; };
        this._registry['batchnormalize'] = function(inputsShapes, parameters) { return []; };
        this._registry['capsule_norm'] = function(inputsShapes, parameters) { return []; };
        this._registry['cast'] = this._no_shape_changed;
        this._registry['clipbyvalue'] = this._no_shape_changed;
        this._registry['concat'] = function(inputsShapes, parameters) {
            const outputShape = inputsShapes[0].slice();
            outputShape[parameters.dim] = 0;
            for (const shape of inputsShapes) {
                outputShape[parameters.dim] += shape[parameters.dim];
            }
            return [outputShape];
        };
        this._registry['concatshift'] = function(inputsShapes, parameters) { return []; };
        this._registry['continuationindicator'] = function(inputsShapes, parameters) { return []; };
        this._registry['conv1d'] = function(inputsShapes, parameters) { return []; };
        this._registry['conv2d_op'] = function(inputsShapes, parameters) { return []; };
        this._registry['conv3d'] = function(inputsShapes, parameters) { return []; };
        this._registry['convolution'] = function(inputsShapes, parameters) {
            if (parameters.padding == 'VALID') {
                const out_h = ~~((inputsShapes[0][1] + parameters.stride_h - parameters.ksize_h) / parameters.stride_h);
                const out_w = ~~((inputsShapes[0][2] + parameters.stride_w - parameters.ksize_w) / parameters.stride_w);
                return [[inputsShapes[0][0], out_h, out_w, parameters.weights]];
            }
            else if (parameters.padding == 'SAME') {
                const out_h = ~~((inputsShapes[0][1] + parameters.stride_h - 1) / parameters.stride_h);
                const out_w = ~~((inputsShapes[0][2] + parameters.stride_w - 1) / parameters.stride_w);
                return [[inputsShapes[0][0], out_h, out_w, parameters.weights]];
            }
        };
        this._registry['crop_image'] = function(inputsShapes, parameters) { return []; };
        this._registry['cropandresize'] = function(inputsShapes, parameters) { return []; };
        this._registry['ctc_loss_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['customlayer'] = function(inputsShapes, parameters) { return []; };
        this._registry['deconvolution'] = function(inputsShapes, parameters) { return []; };
        this._registry['depth2space'] = function(inputsShapes, parameters) { return []; };
        this._registry['depthwise_conv1d'] = function(inputsShapes, parameters) { return []; };
        this._registry['depthwise_conv2d_op'] = function(inputsShapes, parameters) { return []; };
        this._registry['depthwise_convolution'] = function(inputsShapes, parameters) { return []; };
        this._registry['dequantize'] = this._no_shape_changed;
        this._registry['detectionevaluate'] = function(inputsShapes, parameters) { return []; };
        this._registry['detectionoutput'] = function(inputsShapes, parameters) { return []; };
        this._registry['digit_capsule'] = function(inputsShapes, parameters) { return []; };
        this._registry['divide'] = function(inputsShapes, parameters) { return []; };
        this._registry['dropout'] = function(inputsShapes, parameters) { return []; };
        this._registry['dtype_converter'] = this._no_shape_changed;
        this._registry['eltwise'] = function(inputsShapes, parameters) { return []; };
        this._registry['elu'] = this._no_shape_changed;
        this._registry['embedding_lookup'] = function(inputsShapes, parameters) { return []; };
        this._registry['equal'] = function(inputsShapes, parameters) { return []; };
        this._registry['exp'] = this._no_shape_changed;
        this._registry['expand_broadcast'] = function(inputsShapes, parameters) { return []; };
        this._registry['expanddims'] = function(inputsShapes, parameters) { return []; };
        this._registry['flatten'] = function(inputsShapes, parameters) { return []; };
        this._registry['floor'] = this._no_shape_changed;
        this._registry['floor_div'] = this._no_shape_changed;
        this._registry['fullconnect'] = function(inputsShapes, parameters) {
            return [inputsShapes[0].slice(0, parameters.axis).concat([parameters.weights])];
        };
        this._registry['fullconnect_op'] = function(inputsShapes, parameters) { return []; };
        this._registry['gather'] = function(inputsShapes, parameters) { return []; };
        this._registry['gathernd'] = function(inputsShapes, parameters) { return []; };
        this._registry['generator_input_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['greater'] = function(inputsShapes, parameters) { return []; };
        this._registry['greater_equal'] = function(inputsShapes, parameters) { return []; };
        this._registry['group_conv1d'] = function(inputsShapes, parameters) { return []; };
        this._registry['gru'] = function(inputsShapes, parameters) { return []; };
        this._registry['gru_cell'] = function(inputsShapes, parameters) { return []; };
        this._registry['gru_keras'] = function(inputsShapes, parameters) { return []; };
        //this._registry['h5_input_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['hard_swish'] = this._no_shape_changed;
        this._registry['image_resize'] = function(inputsShapes, parameters) { return []; };
        this._registry['image_transform'] = function(inputsShapes, parameters) { return []; };
        //this._registry['input'] = function(inputsShapes, parameters) { return []; };
        this._registry['instancenormalize'] = function(inputsShapes, parameters) { return []; };
        this._registry['keras_rnn_lstm'] = function(inputsShapes, parameters) { return []; };
        this._registry['l2normalize'] = function(inputsShapes, parameters) { return []; };
        this._registry['l2normalizescale'] = function(inputsShapes, parameters) { return []; };
        this._registry['l2pooling'] = function(inputsShapes, parameters) { return []; };
        this._registry['layernormalize'] = function(inputsShapes, parameters) { return []; };
        this._registry['leakyrelu'] = this._no_shape_changed;
        this._registry['less'] = function(inputsShapes, parameters) { return []; };
        this._registry['less_equal'] = function(inputsShapes, parameters) { return []; };
        this._registry['lmdb_input_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['localresponsenormalization'] = function(inputsShapes, parameters) { return []; };
        this._registry['localresponsenormalization_tf'] = function(inputsShapes, parameters) { return []; };
        this._registry['log'] = this._no_shape_changed;
        this._registry['log_softmax'] = this._no_shape_changed;
        this._registry['logical_and'] = function(inputsShapes, parameters) { return []; };
        this._registry['logical_or'] = function(inputsShapes, parameters) { return []; };
        this._registry['lstm'] = function(inputsShapes, parameters) { return []; };
        this._registry['lstm_keras'] = function(inputsShapes, parameters) { return []; };
        this._registry['lstmunit'] = function(inputsShapes, parameters) { return []; };
        this._registry['margin_loss_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['mat_inverse'] = function(inputsShapes, parameters) { return []; };
        this._registry['matmul'] = function(inputsShapes, parameters) { return []; };
        this._registry['minimum'] = function(inputsShapes, parameters) { return []; };
        this._registry['minimum_with_clip'] = function(inputsShapes, parameters) { return []; };
        this._registry['mish'] = function(inputsShapes, parameters) { return []; };
        this._registry['moments'] = function(inputsShapes, parameters) { return []; };
        this._registry['multiply'] = function(inputsShapes, parameters) { return []; };
        this._registry['nce_loss'] = function(inputsShapes, parameters) { return []; };
        this._registry['neg'] = this._no_shape_changed;
        this._registry['noop'] = function(inputsShapes, parameters) { return []; };
        this._registry['noop_multi_out'] = function(inputsShapes, parameters) { return []; };
        this._registry['norm_with_channel_mean'] = function(inputsShapes, parameters) { return []; };
        this._registry['norm_with_min_max'] = function(inputsShapes, parameters) { return []; };
        this._registry['norm_with_scale'] = function(inputsShapes, parameters) { return []; };
        this._registry['not_equal'] = function(inputsShapes, parameters) { return []; };
        this._registry['npy_input_layer'] = function(inputsShapes, parameters) { return []; };
        //this._registry['output'] = function(inputsShapes, parameters) { return []; };
        this._registry['pad'] = function(inputsShapes, parameters) { return []; };
        this._registry['permute'] = function(inputsShapes, parameters) { return []; };
        this._registry['pool3d'] = function(inputsShapes, parameters) { return []; };
        this._registry['pooling'] = function(inputsShapes, parameters) {
            if (parameters.padding == 'VALID') {
                const out_h = ~~((inputsShapes[0][1] + parameters.stride_h - parameters.ksize_h) / parameters.stride_h);
                const out_w = ~~((inputsShapes[0][2] + parameters.stride_w - parameters.ksize_w) / parameters.stride_w);
                return [[inputsShapes[0][0], out_h, out_w, inputsShapes[0][3]]];
            }
            else if (parameters.padding == 'SAME') {
                const out_h = ~~((inputsShapes[0][1] + parameters.stride_h - 1) / parameters.stride_h);
                const out_w = ~~((inputsShapes[0][2] + parameters.stride_w - 1) / parameters.stride_w);
                return [[inputsShapes[0][0], out_h, out_w, inputsShapes[0][3]]];
            }
        };
        this._registry['poolwithargmax'] = function(inputsShapes, parameters) { return []; };
        this._registry['postprocess'] = function(inputsShapes, parameters) { return []; };
        this._registry['pow'] = this._no_shape_changed;
        this._registry['prelu'] = this._no_shape_changed;
        this._registry['preprocess'] = function(inputsShapes, parameters) { return []; };
        this._registry['primary_capsule'] = function(inputsShapes, parameters) { return []; };
        this._registry['priorbox'] = function(inputsShapes, parameters) { return []; };
        this._registry['proposal'] = function(inputsShapes, parameters) { return []; };
        this._registry['quantize'] = this._no_shape_changed;
        this._registry['real_div'] = function(inputsShapes, parameters) { return []; };
        this._registry['reconstruction_loss'] = function(inputsShapes, parameters) { return []; };
        this._registry['recurrent'] = function(inputsShapes, parameters) { return []; };
        this._registry['reducemax'] = function(inputsShapes, parameters) { return []; };
        this._registry['reducemean'] = function(inputsShapes, parameters) { return []; };
        this._registry['reducemin'] = function(inputsShapes, parameters) { return []; };
        this._registry['reducesum'] = function(inputsShapes, parameters) { return []; };
        this._registry['region'] = function(inputsShapes, parameters) { return []; };
        this._registry['relu'] = this._no_shape_changed;
        this._registry['relu_keras'] = this._no_shape_changed;
        this._registry['relun'] = this._no_shape_changed;
        this._registry['reorg'] = function(inputsShapes, parameters) { return []; };
        this._registry['reshape'] = function(inputsShapes, parameters) { return []; };
        this._registry['resizebilinear_image'] = function(inputsShapes, parameters) { return []; };
        this._registry['resizenearest_image'] = function(inputsShapes, parameters) { return []; };
        this._registry['reverse'] = function(inputsShapes, parameters) { return []; };
        this._registry['reverse_sequence'] = function(inputsShapes, parameters) { return []; };
        this._registry['roipooling'] = function(inputsShapes, parameters) { return []; };
        this._registry['route_train'] = function(inputsShapes, parameters) { return []; };
        this._registry['rsqrt'] = this._no_shape_changed;
        this._registry['scatternd'] = function(inputsShapes, parameters) { return []; };
        this._registry['shuffle'] = function(inputsShapes, parameters) { return []; };
        this._registry['sigmoid'] = this._no_shape_changed;
        this._registry['signalframe'] = function(inputsShapes, parameters) { return []; };
        this._registry['simplernn_keras'] = function(inputsShapes, parameters) { return []; };
        this._registry['sin'] = this._no_shape_changed;
        this._registry['slice'] = function(inputsShapes, parameters) { return []; };
        this._registry['softmax'] = this._no_shape_changed;
        this._registry['softmax_with_logits_loss_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['softrelu'] = this._no_shape_changed;
        this._registry['space2batch'] = function(inputsShapes, parameters) { return []; };
        this._registry['space2depth'] = function(inputsShapes, parameters) { return []; };
        this._registry['split'] = function(inputsShapes, parameters) { return []; };
        this._registry['sqlite_input_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['sqrt'] = this._no_shape_changed;
        this._registry['square'] = this._no_shape_changed;
        this._registry['squashing'] = function(inputsShapes, parameters) { return []; };
        this._registry['squeeze'] = function(inputsShapes, parameters) { return []; };
        this._registry['stack'] = function(inputsShapes, parameters) { return []; };
        this._registry['stack_concat'] = function(inputsShapes, parameters) { return []; };
        this._registry['stridedslice'] = function(inputsShapes, parameters) { return []; };
        this._registry['subgraph'] = function(inputsShapes, parameters) { return []; };
        this._registry['subtract'] = function(inputsShapes, parameters) { return []; };
        this._registry['svdf'] = function(inputsShapes, parameters) { return []; };
        this._registry['swish'] = function(inputsShapes, parameters) { return []; };
        this._registry['tanh'] = this._no_shape_changed;
        this._registry['text_input_layer'] = function(inputsShapes, parameters) { return []; };
        this._registry['tile'] = function(inputsShapes, parameters) { return []; };
        this._registry['topk'] = function(inputsShapes, parameters) { return []; };
        this._registry['topk_score'] = function(inputsShapes, parameters) { return []; };
        this._registry['unstack'] = function(inputsShapes, parameters) { return []; };
        this._registry['upsampling'] = function(inputsShapes, parameters) { return []; };
        //this._registry['variable'] = function(inputsShapes, parameters) { return []; };
        this._registry['where'] = function(inputsShapes, parameters) { return []; };
        this._registry['word2vec_input'] = function(inputsShapes, parameters) { return []; };
        this._registry['yolo'] = function(inputsShapes, parameters) { return []; };
        this._registry['yoloprocess'] = function(inputsShapes, parameters) { return []; };
    }

    get(op) {
        if (Object.prototype.hasOwnProperty.call(this._registry, op)) {
            return this._registry[op];
        }
        else {
            return function(inputsShapes, parameters) {
                return [];
            };
        }
    }
};

acuity.Utility = class {

    static getTensorUrl(layerName, port) {
        return "@" + layerName + ":" + port;
    }
};

acuity.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Acuity model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = acuity.ModelFactory;
}