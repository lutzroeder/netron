
var acuity = acuity || {};

acuity.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj && obj.MetaData && obj.Layers) {
                return obj;
            }
        }
        return null;
    }

    async open(context, target) {
        const metadata = await context.metadata('acuity-metadata.json');
        return new acuity.Model(metadata, target);
    }
};

acuity.Model = class {

    constructor(metadata, model, data, quantization) {
        this._name = model.MetaData.Name;
        this._format = 'Acuity ' + 'v' + model.MetaData.AcuityVersion;
        this._runtime = model.MetaData.Platform;
        this._graphs = [ new acuity.Graph(metadata, model, data, quantization) ];
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get runtime() {
        return this._runtime;
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
        const arg = (name) => {
            if (!args.has(name)) {
                args.set(name, { name: name, shape: null });
            }
            return args.get(name);
        };

        for (const layerName of Object.keys(model.Layers)) {
            const layer = model.Layers[layerName];
            layer.inputs = layer.inputs.map((input) => {
                return arg(input);
            });
            layer.outputs = layer.outputs.map((port) => {
                const value = arg("@" + layerName + ":" + port);
                let shape = null;
                if (layer.op.toLowerCase() == 'input' ||
                    layer.op.toLowerCase() == 'variable') {
                    if (Object.prototype.hasOwnProperty.call(layer.parameters, 'shape') && layer.parameters.shape.length > 0) {
                        shape = layer.parameters.shape;
                    } else if (Object.prototype.hasOwnProperty.call(layer.parameters, 'size') && Object.prototype.hasOwnProperty.call(layer.parameters, 'channels')) {
                        const sizes = layer.parameters.size.split(' ');
                        shape = [0, parseInt(sizes[0]), parseInt(sizes[1]), layer.parameters.channels];
                    }
                    if (shape && shape.length === 4 && shape[0] === 0) {
                        shape[0] = 1;
                    }
                }
                value.shape = shape;
                return value;
            });
        }

        acuity.Inference.infer(model.Layers);

        for (const pair of args) {
            const type = new acuity.TensorType(null, new acuity.TensorShape(pair[1].shape));
            const arg = new acuity.Value(pair[0], type, null, null);
            args.set(pair[0], arg);
        }

        for (const layerName of Object.keys(model.Layers)) {
            const layer = model.Layers[layerName];
            switch (layer.op.toLowerCase()) {
                case 'input': {
                    this._inputs.push(new acuity.Argument(layerName, [
                        args.get(layer.outputs[0].name)
                    ]));
                    break;
                }
                case 'output': {
                    this._outputs.push(new acuity.Argument(layerName, [
                        args.get(layer.inputs[0].name)
                    ]));
                    break;
                }
                default: {
                    this._nodes.push(new acuity.Node(metadata, layerName, layer, args));
                    break;
                }
            }
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

acuity.Node = class {

    constructor(metadata, name, layer, args) {
        this._name = name;
        this._type = metadata.type(layer.op) || { name: layer.op };
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._layer = layer;
        if (this._type) {
            if (layer.parameters) {
                for (const key of Object.keys(layer.parameters)) {
                    const attribute = new acuity.Attribute(metadata.attribute(this._type.name, key), key, layer.parameters[key]);
                    this._attributes.push(attribute);
                }
            }
        }
        for (let i = 0; i < layer.inputs.length; i++) {
            const input = layer.inputs[i];
            const arg = args.get(input.name);
            const name = this._type && this._type.inputs && i < this._type.inputs.length ? this._type.inputs[i].name : 'input' + i.toString();
            this._inputs.push(new acuity.Argument(name, [ arg ]));
        }

        if (this._type && this._type.constants) {
            for (const constant of this._type.constants) {
                // const name = "@" + this._name + ":" + constant.name;
                const type = new acuity.TensorType(null, new acuity.TensorShape(null));
                const value = new acuity.Value('', type, null, new acuity.Tensor(type));
                this._inputs.push(new acuity.Argument(constant.name, [ value ]));
            }
        }

        for (let i = 0; i < layer.outputs.length; i++) {
            const output = layer.outputs[i];
            const arg = args.get(output.name);
            const name = this._type && this._type.outputs && i < this._type.outputs.length ? this._type.outputs[i].name : 'output' + i.toString();
            this._outputs.push(new acuity.Argument(name, [arg]));
        }
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
};

acuity.Attribute = class {

    constructor(metadata, name, value) {
        this._type = null;
        this._name = name;
        this._value = value;
        if (metadata) {
            this._type = metadata.type || null;
            if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (metadata.default === value) {
                    this._visible = false;
                }
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

    get visible() {
        return this._visible == false ? false : true;
    }
};

acuity.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

acuity.Value = class {

    constructor(name, type, quantization, initializer) {
        if (typeof name !== 'string') {
            throw new acuity.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    get initializer() {
        return this._initializer;
    }
};

acuity.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
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

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

acuity.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions || null;
    }

    get dimensions() {
        if (Array.isArray(this._dimensions) && this._dimensions.length == 1 && this._dimensions[0] == 0) {
            return [];
        }
        return this._dimensions;
    }

    toString() {
        if (!Array.isArray(this._dimensions) || this._dimensions.length == 0 || (this._dimensions.length == 1 && this._dimensions[0] == 0)) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']';
    }
};

acuity.Tensor = class {

    constructor(type) {
        this._type = type;
    }

    get category() {
        return 'Constant';
    }

    get type() {
        return this._type;
    }
};

acuity.Inference = class {

    static infer(layers) {
        const outputs = new Map();
        const outputLayers = [];
        for (const layerName of Object.keys(layers)) {
            const layer = layers[layerName];
            if (layer.op.toLowerCase() == 'output') {
                outputLayers.push(layer);
            }
            for (const output of layer.outputs) {
                outputs.set(output.name, layer);
            }
        }
        const broadcasts = new Set([
            'add', 'equal', 'fllor_mod', 'floor_div', 'greater', 'greater_equal', 'less', 'less_equal',
            'logical_and', 'logical_or', 'minimum', 'multiply', 'not_equal', 'pow', 'real_div',
            'squared_difference', 'subtract'
        ]);
        const passthroughs = new Set([
            'LocalResponseNormalization', 'a_times_b_plus_c', 'abs', 'batchnorm_single', 'batchnormalize',
            'cast', 'cast', 'clipbyvalue', 'dequantize', 'dtype_converter', 'elu', 'exp', 'floor',
            'groupnormalize', 'hard_sigmoid', 'hard_swish', 'instancenormalize', 'l2normalize', 'l2normalizescale',
            'layernormalize', 'leakyrelu', 'log', 'log_softmax', 'mish', 'neg', 'norm_with_channel_mean',
            'norm_with_min_max', 'norm_with_scale', 'pow', 'prelu', 'quantize', 'relu', 'relu_keras',
            'relun', 'reverse', 'round', 'rsqrt', 'sigmoid', 'sin', 'softmax', 'softrelu', 'sqrt', 'square', 'tanh'
        ]);
        const reduces = new Set([
            'reduceany', 'reducemax', 'reducemean', 'reducemin', 'reduceprod', 'reducesum'
        ]);
        const operators = new Map();
        operators.set('broadcast', (inputs) => {
            const a = inputs[0];
            const b = inputs[1];
            const longer = a.length >= b.length ? a.slice() : b.slice();
            const shorter = a.length < b.length ? a.slice() : b.slice();
            const remain = longer.length - shorter.length;
            for (let i = 0; i < remain; i++) {
                shorter.splice(0, 0, 1);
            }
            for (let i = 0; i < longer.length; i++) {
                longer[i] = longer[i] > shorter[i] ? longer[i] : shorter[i];
            }
            return [ longer ];
        });
        operators.set('concat', (inputs, params) => {
            const outputShape = inputs[0].slice();
            outputShape[params.dim] = 0;
            for (const shape of inputs) {
                outputShape[params.dim] += shape[params.dim];
            }
            return [ outputShape ];
        });
        operators.set('conv1d', (inputs, params) => {
            if (params.padding == 'VALID') {
                const out_h = ~~((inputs[0][1] + params.stride - params.ksize) / params.stride);
                return [ [ inputs[0][0], out_h, params.weights ] ];
            } else if (params.padding == 'SAME') {
                const out_h = ~~((inputs[0][1] + params.stride - 1) / params.stride);
                return [ [ inputs[0][0], out_h, params.weights ] ];
            }
            return null;
        });
        operators.set('convolution', (inputs, params) => {
            if (params.padding == 'VALID') {
                const out_h = ~~((inputs[0][1] + params.stride_h + params.pad[0] + params.pad[1] - params.ksize_h) / params.stride_h);
                const out_w = ~~((inputs[0][2] + params.stride_w + params.pad[2] + params.pad[3]- params.ksize_w) / params.stride_w);
                return [ [ inputs[0][0], out_h, out_w, params.weights ] ];
            } else if (params.padding == 'SAME') {
                const out_h = ~~((inputs[0][1] + params.stride_h - 1) / params.stride_h);
                const out_w = ~~((inputs[0][2] + params.stride_w - 1) / params.stride_w);
                return [ [ inputs[0][0], out_h, out_w, params.weights ] ];
            }
            return null;
        });
        operators.set('deconvolution', (inputs, params) => {
            return [ params.output_shape.map((item, index) => item == 0 ? inputs[0][index] : item) ];
        });
        operators.set('fullconnect', (inputs, params) => {
            return [ inputs[0].slice(0, params.axis).concat([params.weights]) ];
        });
        operators.set('gather', (inputs, params) => {
            const prefix = inputs[1].slice();
            const suffix = inputs[0].slice(params.axis + 1);
            return [ prefix.concat(suffix) ];
        });
        operators.set('lstm', (inputs, params) => {
            let batch = inputs[0][0];
            const output = params.num_proj != null ? params.num_proj : params.weights;
            if (params.time_major) {
                batch = inputs[0][1];
            }
            const newShape = params.return_sequences ? [ inputs[0][0], inputs[0][1], output ] : [ batch, output ];
            return [ newShape, [batch, output], [batch, params.weights] ];
        });
        operators.set('matmul', (inputs, params) => {
            const a = inputs[0];
            const b = inputs[1];
            let newShape = a.slice(0, -2);
            if (params.transpose_a) {
                newShape = newShape.concat(a.slice(-1));
            } else {
                newShape = newShape.concat(a.slice(-2, -1));
            }
            if (params.transpose_b) {
                newShape = newShape.concat(b.slice(-2, -1));
            } else {
                newShape = newShape.concat(b.slice(-1));
            }
            return [ newShape ];
        });
        operators.set('pad', (inputs, params) => {
            return [ inputs[0].map((item, index) => item + params.padding_value[index][0] + params.padding_value[index][1]) ];
        });
        operators.set('permute', (inputs, params) => {
            return [ inputs[0].map((item, index) => inputs[0][params.perm[index]]) ];
        });
        operators.set('pooling', (inputs, params) => {
            if (params.padding == 'VALID') {
                const out_h = ~~((inputs[0][1] + params.stride_h - params.ksize_h) / params.stride_h);
                const out_w = ~~((inputs[0][2] + params.stride_w - params.ksize_w) / params.stride_w);
                return [ [inputs[0][0], out_h, out_w, inputs[0][3]] ];
            } else if (params.padding == 'SAME') {
                const out_h = ~~((inputs[0][1] + params.stride_h - 1) / params.stride_h);
                const out_w = ~~((inputs[0][2] + params.stride_w - 1) / params.stride_w);
                return [ [inputs[0][0], out_h, out_w, inputs[0][3]] ];
            }
            return null;
        });
        operators.set('reduce', (inputs, params) => {
            const newShape = inputs[0].slice();
            if (params.keep_dims) {
                for (const i in params.axis_list) {
                    newShape[i] = 1;
                }
            } else {
                const axis_list = params.axis_list.map((item) => {
                    return item < 0 ? newShape.length + item : item;
                });
                axis_list.sort((a, b) => {
                    return b - a;
                });
                for (const item of axis_list) {
                    newShape.splice(item, 1);
                }
                if (!newShape.length) {
                    newShape.splice(0, 0, 0);
                }
            }
            return [ newShape ];
        });
        operators.set('repeat', (inputs, params) => {
            const newShape = inputs[0].slice();
            newShape[params.axis] = params.maxlen;
            return [ newShape ];
        });
        operators.set('reshape', (inputs, params) => {
            const negativeIndexs = [];
            let shape = params.shape;
            if (typeof params.shape === 'string') {
                shape = params.shape.split(/\s+/).map((item) => {
                    return parseInt(item);
                });
            }
            const newShape = shape.map((item, index) => {
                if (item == 0) {
                    return inputs[0][index];
                }
                if (item == -1) {
                    negativeIndexs.push(index);
                    return 1;
                }
                return item;
            });
            if (negativeIndexs.length > 0) {
                newShape[negativeIndexs[0]] = inputs[0].reduce((a, c) => a * c) / newShape.reduce((a, c) => a * c);
            }
            return [ newShape ];
        });
        operators.set('sequence_mask', (inputs, params) => {
            return [ inputs[0].slice().concat([params.maxlen]) ];
        });
        operators.set('slice', (inputs, params) => {
            return [ params.size.map((item, index) => item == -1 ? inputs[0][index] : item) ];
        });
        operators.set('squeeze', (inputs, params) => {
            const newShape = inputs[0].slice();
            const axis_list = [...new Set(params.axis_list)].sort((a, b) => b - a);
            for (const item of axis_list) {
                newShape.splice(item, 1);
            }
            return [ newShape ];
        });
        operators.set('space2depth', (inputs, params) => {
            const h = inputs[0][1] / params.block_size[0];
            const w = inputs[0][2] / params.block_size[1];
            const c = inputs[0][3] * params.block_size[1] * params.block_size[1];
            return [ [inputs[0][0], h, w, c] ];
        });
        operators.set('split', (inputs, params) => {
            const sizes = [];
            const slices = params.slices.slice();
            slices.splice(0, 0, 0);
            slices.push(inputs[0][params.dim]);
            slices.reduce((a, b) => {
                sizes.push(b - a);
                return b;
            });
            return sizes.map((item) => {
                const shape = inputs[0].slice();
                shape[params.dim] = item;
                return shape;
            });
        });
        operators.set('stack', (inputs, params) => {
            const newShape = inputs[0].slice();
            if (newShape.length == 1 && newShape[0] == 0) {
                newShape[0] = 1;
            } else {
                newShape.splice(params.axis, 0, inputs.length);
            }
            return [ newShape ];
        });
        operators.set('stridedslice', (inputs, params) => {
            const input_shape = inputs[0].slice();
            const begin = params.slice_begin.slice();
            const end = params.slice_end.slice();
            if (params.slice_begin_mask > 0) {
                for (let i = 0; i < begin.length; i++) {
                    if ((params.slice_begin_mask >>> i) & 0x1) {
                        begin[i] = -1;
                    }
                }
            }
            if (params.slice_end_mask > 0) {
                for (let i = 0; i < end.length; i++) {
                    if ((params.slice_end_mask >>> i) & 0x1) {
                        end[i] = -1;
                    }
                }
            }
            for (let i = 0; i < begin.length; i++) {
                if (begin[i] == -1) {
                    begin[i] = 0;
                }
            }
            if (inputs[0].length == end.length) {
                for (let i = 0; i < end.length; i++) {
                    if (end[i] == -1 || end[i] > input_shape[i]) {
                        end[i] = input_shape[i];
                    }
                }
            } else if (inputs[0].length < end.length) {
                if (params.slice_new_axis_mask) {
                    const len = (params.slice_new_axis_mask >>> 0).toString(2).length;
                    for (let i = 0; i < len; i++) {
                        if ((params.slice_new_axis_mask >>> i) & 0x1) {
                            input_shape.splice(i, 0, 1);
                        }
                    }
                    for (let i = 0; i < end.length; i++) {
                        if (end[i] == -1) {
                            end[i] = input_shape[i];
                        }
                    }
                }
            }
            let newShape = [];
            for (let i = 0; i < begin.length; i++) {
                newShape = newShape.concat([(end[i] - begin[i])/params.slice_strides[i]]);
            }
            if (params.slice_shrink_axis_mask) {
                const len = (params.slice_shrink_axis_mask >>> 0).toString(2).length;
                for (let i = 0; i < len; i++) {
                    if ((params.slice_shrink_axis_mask >>> i) & 0x1) {
                        newShape.splice(i, 1);
                    }
                }
            }
            if (params.slice_new_axis_mask) {
                const len = (params.slice_new_axis_mask >>> 0).toString(2).length;
                for (let i = 0; i < len; i++) {
                    if ((params.slice_new_axis_mask >>> i) & 0x1) {
                        if (inputs[0].length == begin.length) {
                            newShape.splice(i, 0, 1);
                        } else if (inputs[0].length < begin.length) {
                            newShape[i] = 1;
                        }
                    }
                }
            }
            return [ newShape ];
        });
        const infer = (output) => {
            if (outputs.has(output.name)) {
                let ready = true;
                const layer = outputs.get(output.name);
                for (const input of layer.inputs) {
                    if (input.shape === null) {
                        infer(input);
                        if (input.shape === null) {
                            ready = false;
                            break;
                        }
                    }
                }
                if (ready) {
                    let callback = null;
                    if (operators.has(layer.op)) {
                        callback = operators.get(layer.op);
                    } else if (passthroughs.has(layer.op)) {
                        callback = (inputs) => [ inputs[0].slice() ];
                    } else if (broadcasts.has(layer.op)) {
                        callback = operators.get('broadcast');
                    } else if (reduces.has(layer.op)) {
                        callback = operators.get('reduce');
                    } else {
                        callback = () => [];
                    }
                    const parameters = layer.parameters;
                    const inputs = layer.inputs.map((input) => input.shape);
                    const outputs = callback(inputs, parameters);
                    for (let i = 0; i < outputs.length; i++) {
                        if (i < layer.outputs.length) {
                            layer.outputs[i].shape = outputs[i];
                        }
                    }
                }
            }
        };
        for (const layer of outputLayers) {
            for (const output of layer.outputs) {
                infer(output);
            }
        }
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