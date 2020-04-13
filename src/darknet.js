/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var darknet = darknet || {};
var base = base || require('./base');
var long = long || { Long: require('long') };

darknet.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'cfg') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return darknet.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier;
            const parts = identifier.split('.');
            parts.pop();
            const basename = parts.join('.');
            return context.request(basename + '.weights', null).then((weights) => {
                return this._openModel(metadata, identifier, context.text, weights);
            }).catch(() => {
                return this._openModel(metadata, identifier, context.text, null);
            });
        });
    }
    _openModel( metadata, identifier, cfg, weights) {
        try {
            return new darknet.Model(metadata, cfg, weights ? new darknet.Weights(weights) : null);
        }
        catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new darknet.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
        }
    }
};

darknet.Model = class {

    constructor(metadata, cfg, weights) {
        this._graphs = [];
        this._graphs.push(new darknet.Graph(metadata, cfg, weights));
    }

    get format() {
        return 'Darknet';
    }

    get graphs() {
        return this._graphs;
    }
};

darknet.Graph = class {
    
    constructor(metadata, cfg, weights) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        // read_cfg
        let sections = [];
        let section = null;
        let lines = cfg.split('\n');
        let lineNumber = 0;
        while (lines.length > 0) {
            lineNumber++;
            const text = lines.shift();
            const line = text.replace(/\s/g, '');
            if (line.length > 0) {
                switch (line[0]) {
                    case '#':
                    case ';':
                        break;
                    case '[': {
                        section = {};
                        section.line = lineNumber;
                        section.type = line[line.length - 1] === ']' ? line.substring(1, line.length - 1) : line.substring(1);
                        section.options = {};
                        sections.push(section);
                        break;
                    }
                    default: {
                        if (!section || line[0] < 0x20 || line[0] > 0x7E) {
                            throw new darknet.Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trim() + "' at line " + lineNumber.toString() + ".");
                        }
                        if (section) {
                            const index = line.indexOf('=');
                            if (index < 0) {
                                throw new darknet.Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trim() + "' at line " + lineNumber.toString() + ".");
                            }
                            const key = line.substring(0, index);
                            const value = line.substring(index + 1);
                            section.options[key] = value;
                        }
                        break;
                    }
                }
            }
        }

        const option_find_int = (options, key, defaultValue) => {
            let value = options[key];
            if (typeof value === 'string' && value.startsWith('$')) {
                const key = value.substring(1);
                value = globals.has(key) ? globals.get(key) : value;
            }
            if (value !== undefined) {
                const number = parseInt(value, 10);
                if (!Number.isInteger(number)) {
                    throw new darknet.Error("Invalid int option '" + JSON.stringify(options[key]) + "'.");
                }
                return number;
            }
            return defaultValue;
        };

        const option_find_str = (options, key, defaultValue) => {
            const value = options[key];
            return value !== undefined ? value : defaultValue;
        };

        const make_shape = (dimensions, source) => {
            if (dimensions.some((dimension) => dimension === 0 || dimension === undefined || isNaN(dimension))) {
                throw new darknet.Error("Invalid tensor shape '" + JSON.stringify(dimensions) + "' in '" + source + "'.");
            }
            return new darknet.TensorShape(dimensions);
        }

        const load_weights = (name, shape, visible) => {
            let data = weights ? weights.bytes(4 * shape.reduce((a, b) => a * b)) : null;
            const type = new darknet.TensorType('float32', make_shape(shape, 'load_weights'));
            const initializer = new darknet.Tensor(type, data);
            const argument = new darknet.Argument('', null, initializer);
            return new darknet.Parameter(name, visible === false ? false : true, [ argument ]);
        }

        const load_batch_normalize_weights = (layer, prefix, size) => {
            layer.weights.push(load_weights(prefix + 'scale', [ size ], prefix === ''));
            layer.weights.push(load_weights(prefix + 'mean', [ size ], prefix === ''));
            layer.weights.push(load_weights(prefix + 'variance', [ size ], prefix === ''));
        }

        const make_convolutional_layer = (layer, prefix, w, h, c, n, groups, size, stride_x, stride_y, padding, batch_normalize) => {
            layer.out_w = Math.floor((w + 2 * padding - size) / stride_x) + 1;
            layer.out_h = Math.floor((h + 2 * padding - size) / stride_y) + 1;
            layer.out_c = n;
            layer.out = layer.out_w * layer.out_h * layer.out_c;
            layer.weights.push(load_weights(prefix + 'biases', [ n ], prefix === ''));
            if (batch_normalize) {
                load_batch_normalize_weights(layer, prefix, n);
            }
            layer.weights.push(load_weights(prefix + 'weights', [ Math.floor(c / groups), n, size, size ], prefix === ''));
            layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'make_convolutional_layer'));
        }

        const make_connected_layer = (layer, prefix, inputs, outputs, batch_normalize) => {
            layer.out_h = 1;
            layer.out_w = 1;
            layer.out_c = outputs;
            layer.out = outputs;
            layer.weights.push(load_weights(prefix + 'biases', [ outputs ], prefix === ''));
            if (batch_normalize) {
                load_batch_normalize_weights(layer, prefix, outputs);
            }
            layer.weights.push(load_weights(prefix + 'weights', [ inputs, outputs ], prefix === ''));
            layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ outputs ], 'make_connected_layer'));
        }

        let params = {};
        let globals = new Map();
        const net = sections.shift();
        switch (net.type) {
            case 'net':
            case 'network': {
                params.h = option_find_int(net.options, 'height', 0);
                params.w = option_find_int(net.options, 'width', 0);
                params.c = option_find_int(net.options, 'channels', 0);
                params.inputs = option_find_int(net.options, 'inputs', params.h * params.w * params.c);
                for (const key of Object.keys(net.options)) {
                    globals.set(key, net.options[key]);
                }
                break;
            }
        }

        const inputType = params.w && params.h && params.c ?
            new darknet.TensorType('float32', make_shape([ params.w, params.h, params.c ], 'params-if')) :
            new darknet.TensorType('float32', make_shape([ params.inputs ], 'params-else'));
        const inputName = 'input';
        params.arguments = [ new darknet.Argument(inputName, inputType, null) ];
        this._inputs.push(new darknet.Parameter(inputName, true, params.arguments));

        if (sections.length === 0) {
            throw new darknet.Error('Config file has no sections.');
        }

        let infer = true;
        for (let i = 0; i < sections.length; i++) {
            let section = sections[i];
            section.name = i.toString();
            section.chain = [];
            section.layer = {};
            let options = section.options;
            let layer = section.layer; 
            layer.inputs = [].concat(params.arguments);
            layer.outputs = [ new darknet.Argument(i.toString(), null, null) ];
            layer.weights = [];
            switch (section.type) {
                case 'shortcut': {
                    const from = options.from ? options.from.split(',').map((item) => Number.parseInt(item.trim(), 10)) : [];
                    for (let index of from) {
                        index = (index < 0) ? i + index : index;
                        const item = sections[index].layer;
                        if (item) {
                            layer.inputs.push(item.outputs[0]);
                        }
                    }
                    delete options.from;
                    break;
                }
                case 'sam':
                case 'scale_channels': {
                    let index = option_find_int(options, 'from', 0);
                    index = (index < 0) ? i + index : index;
                    const item = sections[index].layer;
                    if (item) {
                        layer.inputs.push(item.outputs[0]);
                    }
                    delete options.from;
                    break;
                }
                case 'route': {
                    layer.inputs = [];
                    layer.layers = [];
                    const routes = options.layers ? options.layers.split(',').map((route) => Number.parseInt(route.trim(), 10)) : [];
                    for (let j = 0; j < routes.length; j++) {
                        const index = (routes[j] < 0) ? i + routes[j] : routes[j];
                        const route = sections[index].layer;
                        if (route) {
                            layer.inputs.push(route.outputs[0]);
                            layer.layers.push(route);
                        }
                    }
                    delete options.layers;
                    break;
                }
            }
            if (infer) {
                switch (section.type) {
                    case 'conv':
                    case 'convolutional':
                    case 'deconvolutional': {
                        const shape = layer.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw new darknet.Error('Layer before convolutional layer must output image.');
                        }
                        const size = option_find_int(options, 'size', 1);
                        const n = option_find_int(options, 'filters', 1);
                        const pad = option_find_int(options, 'pad', 0);
                        const padding = pad ? (size >> 1) : option_find_int(options, 'padding', 0);
                        let stride_x = option_find_int(options, 'stride_x', -1);
                        let stride_y = option_find_int(options, 'stride_y', -1);
                        if (stride_x < 1 || stride_y < 1) {
                            const stride = option_find_int(options, 'stride', 1);
                            stride_x = stride_x < 1 ? stride : stride_x; 
                            stride_y = stride_y < 1 ? stride : stride_y;
                        }
                        const groups = option_find_int(options, 'groups', 1);
                        const batch_normalize = option_find_int(options, 'batch_normalize', 0);
                        const activation = option_find_str(options, 'activation', 'logistic');
                        make_convolutional_layer(layer, '', params.w, params.h, params.c, n, groups, size, stride_x, stride_y, padding, batch_normalize);
                        if (activation !== 'logistic') {
                            section.chain.push({ type: activation });
                        }
                        break;
                    }
                    case 'connected': {
                        const outputs = option_find_int(options, 'output', 1);
                        const batch_normalize = option_find_int(options, 'batch_normalize', 0);
                        const activation = option_find_str(options, 'activation', 'logistic');
                        make_connected_layer(layer, '', params.inputs, outputs, batch_normalize);
                        if (activation !== 'logistic') {
                            section.chain.push({ type: activation });
                        }
                        break;
                    }
                    case 'local': {
                        const shape = layer.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw new darknet.Error('Layer before avgpool layer must output image.');
                        }
                        const n = option_find_int(options, 'filters' , 1);
                        const size = option_find_int(options, 'size', 1);
                        const stride = option_find_int(options, 'stride', 1);
                        const pad = option_find_int(options, 'pad', 0);
                        const activation = option_find_str(options, 'activation', 'logistic');
                        layer.out_h = Math.floor((params.h - (pad ? 1 : size)) / stride) + 1;
                        layer.out_w = Math.floor((params.w - (pad ? 1 : size)) / stride) + 1;
                        layer.out_c = n;
                        layer.out = layer.out_w * layer.out_h * layer.out_c;
                        layer.weights.push(load_weights('weights', [ params.c, n, size, size, layer.out_h * layer.out_w ]));
                        layer.weights.push(load_weights('biases',[ layer.out_w * layer.out_h * layer.out_c ]));
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'local'));
                        if (activation !== 'logistic') {
                            section.chain.push({ type: activation });
                        }
                        break;
                    }
                    case 'batchnorm': {
                        layer.out_h = params.h;
                        layer.out_w = params.w;
                        layer.out_c = params.c;
                        layer.out = layer.in;
                        load_batch_normalize_weights(weights, section, '', layer.out);
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.ouputs ], 'batchnorm'));
                        break;
                    }
                    case 'activation': {
                        layer.out_h = params.h;
                        layer.out_w = params.w;
                        layer.out_c = params.c;
                        layer.out = layer.in;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.ouputs ], 'activation'));
                        break;
                    }
                    case 'max':
                    case 'maxpool': {
                        const shape = layer.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw new darknet.Error('Layer before maxpool layer must output image.');
                        }
                        const antialiasing = option_find_int(options, 'antialiasing', 0);
                        const stride = option_find_int(options, 'stride', 1);
                        const blur_stride_x = option_find_int(options, 'stride_x', stride);
                        const blur_stride_y = option_find_int(options, 'stride_y', stride);
                        const stride_x = antialiasing ? 1 : blur_stride_x;
                        const stride_y = antialiasing ? 1 : blur_stride_y;
                        const size = option_find_int(options, 'size', stride);
                        const padding = option_find_int(options, 'padding', size - 1);
                        const out_channels = option_find_int(options, 'out_channels', 1);
                        const maxpool_depth = option_find_int(options, 'maxpool_depth', 0);
                        if (maxpool_depth) {
                            layer.out_c = out_channels;
                            layer.out_w = params.w;
                            layer.out_h = params.h;
                        }
                        else {
                            layer.out_w = Math.floor((params.w + padding - size) / stride_x) + 1;
                            layer.out_h = Math.floor((params.h + padding - size) / stride_y) + 1;
                            layer.out_c = params.c;
                        }
                        if (antialiasing) {
                            const blur_size = antialiasing === 2 ? 2 : 3;
                            const blur_pad = antialiasing === 2 ? 0 : Math.floor(blur_size / 3);
                            layer.input_layer = { weights: [], outputs: layer.outputs };
                            make_convolutional_layer(layer.input_layer, '', layer.out_h, layer.out_w, layer.out_c, layer.out_c, layer.out_c, blur_size, blur_stride_x, blur_stride_y, blur_pad, 0);
                            layer.out_w = layer.input_layer.out_w;
                            layer.out_h = layer.input_layer.out_h;
                            layer.out_c = layer.input_layer.out_c;
                        }
                        else {
                            layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'maxpool'));
                        }
                        layer.out = layer.out_w * layer.out_h * layer.out_c;
                        break;
                    }
                    case 'avgpool': {
                        const shape = layer.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw new darknet.Error('Layer before avgpool layer must output image.');
                        }
                        layer.out_w = 1;
                        layer.out_h = 1;
                        layer.out_c = params.c;
                        layer.out = layer.out_c;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'avgpool'));
                        break;
                    }
                    case 'crnn': {
                        const size = option_find_int(options, 'size', 3);
                        const stride = option_find_int(options, 'stride', 1);
                        const output_filters = option_find_int(options, 'output', 1);
                        const hidden_filters = option_find_int(options, 'hidden', 1);
                        const groups = option_find_int(options, 'groups', 1);
                        const pad = option_find_int(options, 'pad', 0);
                        const padding = pad ? (size >> 1) : option_find_int(options, 'padding', 0);
                        const batch_normalize = option_find_int(options, 'batch_normalize', 0);
                        layer.input_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_convolutional_layer(layer.input_layer, 'input_', params.h, params.w, params.c, hidden_filters, groups, size, stride, stride, padding, batch_normalize);
                        layer.self_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_convolutional_layer(layer.self_layer, 'self_', params.h, params.w, hidden_filters, hidden_filters, groups, size, stride, stride, padding, batch_normalize);
                        layer.output_layer = { weights: [], outputs: layer.outputs };
                        make_convolutional_layer(layer.output_layer, 'output_', params.h, params.w, hidden_filters, output_filters, groups, size, stride, stride, padding, batch_normalize);
                        layer.weights = layer.weights.concat(layer.input_layer.weights);
                        layer.weights = layer.weights.concat(layer.self_layer.weights);
                        layer.weights = layer.weights.concat(layer.output_layer.weights);
                        layer.out_h = layer.output_layer.out_h;
                        layer.out_w = layer.output_layer.out_w;
                        layer.out_c = output_filters;
                        layer.out = layer.output_layer.out;
                        break;
                    }
                    case 'rnn': {
                        const outputs = option_find_int(options, 'output', 1);
                        const hidden = option_find_int(options, 'hidden', 1);
                        const batch_normalize = option_find_int(options, 'batch_normalize', 0);
                        const inputs = params.inputs;
                        layer.input_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.input_layer, 'input_', inputs, hidden, batch_normalize);
                        layer.self_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.self_layer, 'self_', hidden, hidden, batch_normalize);
                        layer.output_layer = { weights: [], outputs: layer.outputs };
                        make_connected_layer(layer.output_layer, 'output_', hidden, outputs, batch_normalize);
                        layer.weights = layer.weights.concat(layer.input_layer.weights);
                        layer.weights = layer.weights.concat(layer.self_layer.weights);
                        layer.weights = layer.weights.concat(layer.output_layer.weights);
                        layer.out_w = 1;
                        layer.out_h = 1;
                        layer.out_c = outputs;
                        layer.out = outputs;
                        break;
                    }
                    case 'gru': {
                        const inputs = params.inputs;
                        const outputs = option_find_int(options, 'output', 1);
                        const batch_normalize = option_find_int(options, 'batch_normalize', 0);
                        layer.input_z_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.input_z_layer, 'input_z', inputs, outputs, batch_normalize);
                        layer.state_z_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.state_z_layer, 'state_z', outputs, outputs, batch_normalize);
                        layer.input_r_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.input_r_layer, 'input_r', inputs, outputs, batch_normalize);
                        layer.state_r_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.state_r_layer, 'state_r', outputs, outputs, batch_normalize);
                        layer.input_h_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.input_h_layer, 'input_h', inputs, outputs, batch_normalize);
                        layer.state_h_layer = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.state_h_layer, 'state_h', outputs, outputs, batch_normalize);
                        layer.weights = layer.weights.concat(layer.input_z_layer.weights);
                        layer.weights = layer.weights.concat(layer.state_z_layer.weights);
                        layer.weights = layer.weights.concat(layer.input_r_layer.weights);
                        layer.weights = layer.weights.concat(layer.state_r_layer.weights);
                        layer.weights = layer.weights.concat(layer.input_h_layer.weights);
                        layer.weights = layer.weights.concat(layer.state_h_layer.weights);
                        layer.out = outputs;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ outputs ], 'gru'));
                        break;
                    }
                    case 'lstm': {
                        const inputs = params.inputs;
                        const outputs = option_find_int(options, 'output', 1);
                        const batch_normalize = option_find_int(options, 'batch_normalize', 0);
                        layer.uf = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.uf, 'uf_', inputs, outputs, batch_normalize);
                        layer.ui = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.ui, 'ui_', inputs, outputs, batch_normalize);
                        layer.ug = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.ug, 'ug_', inputs, outputs, batch_normalize);
                        layer.uo = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.uo, 'uo_', inputs, outputs, batch_normalize);
                        layer.wf = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.wf, 'wf_', outputs, outputs, batch_normalize);
                        layer.wi = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.wi, 'wi_', outputs, outputs, batch_normalize);
                        layer.wg = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.wg, 'wg_', outputs, outputs, batch_normalize);
                        layer.wo = { weights: [], outputs: [ new darknet.Argument('', null, null) ] };
                        make_connected_layer(layer.wo, 'wo_', outputs, outputs, batch_normalize);
                        layer.weights = layer.weights.concat(layer.uf.weights);
                        layer.weights = layer.weights.concat(layer.ui.weights);
                        layer.weights = layer.weights.concat(layer.ug.weights);
                        layer.weights = layer.weights.concat(layer.uo.weights);
                        layer.weights = layer.weights.concat(layer.wf.weights);
                        layer.weights = layer.weights.concat(layer.wi.weights);
                        layer.weights = layer.weights.concat(layer.wg.weights);
                        layer.weights = layer.weights.concat(layer.wo.weights);
                        layer.out_w = 1;
                        layer.out_h = 1;
                        layer.out_c = outputs;
                        layer.out = outputs;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ outputs ], 'lstm'));
                        weights = null;
                        break;
                    }
                    case 'softmax': {
                        layer.out_w = params.w;
                        layer.out_h = params.h;
                        layer.out_c = params.c;
                        layer.out = params.inputs;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out ], 'softmax'));
                        break;
                    }
                    case 'dropout': {
                        layer.out_w = params.w;
                        layer.out_h = params.h;
                        layer.out_c = params.c;
                        layer.out = params.inputs;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'dropout'));
                        break;
                    }
                    case 'upsample': {
                        const stride = option_find_int(options, 'stride', 2);
                        layer.out_w = params.w * stride;
                        layer.out_h = params.h * stride;
                        layer.out_c = params.c;
                        layer.out = layer.out_w * layer.out_h * layer.out_c;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'upsample'));
                        break;
                    }
                    case 'crop': {
                        const shape = layer.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw new darknet.Error('Layer before crop layer must output image.');
                        }
                        const crop_height = option_find_int(options, 'crop_height', 1);
                        const crop_width = option_find_int(options, 'crop_width', 1);
                        layer.out_w = crop_width;
                        layer.out_h = crop_height;
                        layer.out_c = params.c;
                        layer.out = layer.out_w * layer.out_h * layer.out_c;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'crop'));
                        break;
                    }
                    case 'yolo': {
                        const classes = option_find_int(options, 'classes', 20);
                        const n = option_find_int(options, 'num', 1);
                        layer.out_h = params.h;
                        layer.out_w = params.w;
                        layer.out_c = n * (classes + 4 + 1);
                        layer.out = layer.out_h * layer.out_w * layer.out_c;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'yolo'));
                        break;
                    }
                    case 'Gaussian_yolo': {
                        const classes = option_find_int(options, 'classes', 20);
                        const n = option_find_int(options, 'num', 1);
                        layer.out_h = params.h;
                        layer.out_w = params.w;
                        layer.out_c = n * (classes + 8 + 1);
                        layer.out = layer.out_h * layer.out_w * layer.out_c;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'Gaussian_yolo'));
                        break;
                    }
                    case 'region': {
                        const coords = option_find_int(options, 'coords', 4);
                        const classes = option_find_int(options, 'classes', 20);
                        const num = option_find_int(options, 'num', 1);
                        layer.out = params.h * params.w * num * (classes + coords + 1);
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ params.h, params.w, num, (classes + coords + 1) ], 'region'));
                        break;
                    }
                    case 'cost': {
                        layer.out = params.inputs;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out ], 'cost'));
                        break;
                    }
                    case 'reorg': {
                        const stride = option_find_int(options, 'stride', 1);
                        const reverse = option_find_int(options, 'reverse', 0);
                        const extra = option_find_int(options, 'extra', 0);
                        if (reverse) {
                            layer.out_w = params.w * stride;
                            layer.out_h = params.h * stride;
                            layer.out_c = Math.floor(params.c / (stride * stride));
                        } 
                        else {
                            layer.out_w = Math.floor(params.w / stride);
                            layer.out_h = Math.floor(params.h / stride);
                            layer.out_c = params.c * (stride * stride);
                        }
                        layer.out = layer.out_h * layer.out_w * layer.out_c;
                        if (extra) {
                            layer.out_w = 0;
                            layer.out_h = 0; 
                            layer.out_c = 0;
                            layer.out = (params.h * params.w * params.c) + extra;
                        }
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out ], 'reorg'));
                        break;
                    }
                    case 'route': {
                        let layers = [].concat(layer.layers);
                        layer.out = 0;
                        for (const next of layers) {
                            layer.out += next.out;
                        }
                        const first = layers.shift();
                        layer.out_w = first.out_w;
                        layer.out_h = first.out_h;
                        layer.out_c = first.out_c;
                        while (layers.length > 0) {
                            const next = layers.shift();
                            if (next.out_w === first.out_w && next.out_h === first.out_h) {
                                layer.out_c += next.out_c;
                            }
                            else {
                                layer.out_h = 0;
                                layer.out_w = 0;
                                layer.out_c = 0;
                            }
                        }
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out_w, layer.out_h, layer.out_c ], 'route'));
                        break;
                    }
                    case 'shortcut':
                    case 'scale_channels':
                    case 'sam': {
                        const activation = option_find_str(options, 'activation', 'linear');
                        layer.out_w = params.w;
                        layer.out_h = params.h;
                        layer.out_c = params.c;
                        layer.out = params.w * params.h * params.c;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ params.w, params.h, params.c ], 'shortcut|scale_channels|sam'));
                        if (activation !== 'linear') {
                            section.chain.push({ type: activation });
                        }
                        break;
                    }
                    case 'detection': {
                        layer.out_w = params.w;
                        layer.out_h = params.h;
                        layer.out_c = params.c;
                        layer.out = params.inputs;
                        layer.outputs[0].type = new darknet.TensorType('float32', make_shape([ layer.out ], 'detection'));
                        break;
                    }
                    default: {
                        infer = false;
                        break;
                    }
                }
                params.h = layer.out_h;
                params.w = layer.out_w;
                params.c = layer.out_c;
                params.inputs = layer.out;
                params.last = section;
            }
            params.arguments = layer.outputs;
        }

        for (let i = 0; i < sections.length; i++) {
            this._nodes.push(new darknet.Node(metadata, net, sections[i]));
        }

        /* if (sections.length > 0) {
            const last = sections[sections.length - 1].layer;
            for (let i = 0; i < last.outputs.length; i++) {
                const outputName = 'output' + (i > 1 ? i.toString() : '');
                this._outputs.push(new darknet.Parameter(outputName, true, [ last.outputs[i] ]));
            }
        } */

        if (weights) {
            weights.validate();
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

darknet.Parameter = class {

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

darknet.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new darknet.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

    set type(value) {
        if (this._type) {
            throw new darknet.Error('Invalid argument type set operation.');
        }
        this._type = value;
    }

    get initializer() {
        return this._initializer;
    }
};

darknet.Node = class {

    constructor(metadata, net, section) {
        this._name = section.name || '';
        this._location = section.line !== undefined ? section.line.toString() : undefined;
        this._metadata = metadata;
        this._operator = section.type;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        const layer = section.layer;
        if (layer && layer.inputs && layer.inputs.length > 0) {
            this._inputs.push(new darknet.Parameter(layer.inputs.length <= 1 ? 'input' : 'inputs', true, layer.inputs));
        }
        if (layer && layer.weights && layer.weights.length > 0) {
            this._inputs = this._inputs.concat(layer.weights);
        }
        if (layer && layer.outputs && layer.outputs.length > 0) {
            this._outputs.push(new darknet.Parameter(layer.outputs.length <= 1 ? 'output' : 'outputs', true, layer.outputs));
        }
        if (section.chain) {
            for (const chain of section.chain) {
                this._chain.push(new darknet.Node(metadata, net, chain, ''));
            }
        }
        const options = section.options;
        if (options) {
            for (const key of Object.keys(options)) {
                this._attributes.push(new darknet.Attribute(metadata, this._operator, key, options[key]));
            }
        }
    }

    get name() {
        return this._name;
    }

    get location() {
        return this._location;
    }

    get operator() {
        return this._operator;
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

    get chain() {
        return this._chain;
    }
};

darknet.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;
        const schema = metadata.attribute(operator, name);
        if (schema) {
            this._type = schema.type || '';
            switch (this._type) {
                case 'int32': {
                    const number = parseInt(this._value, 10);
                    if (Number.isInteger(number)) {
                        this._value = number;
                    }
                    break;
                }
                case 'float32': {
                    const number = parseFloat(this._value);
                    if (!isNaN(number)) {
                        this._value = number;
                    }
                    break;
                }
                case 'int32[]': {
                    const numbers = this._value.split(',').map((item) => parseInt(item.trim(), 10));
                    if (numbers.every((number) => Number.isInteger(number))) {
                        this._value = numbers;
                    }
                    break;
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default) {
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

darknet.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Tensor';
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.state = null;
        context.position = 0;
        context.count = 0;
        context.dataView = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        context.dimensions = this.type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        let results = [];
        const size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.dataView.getFloat32(context.position, true));
                context.position += 4;
                context.count++;
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
        return results;
    }
};

darknet.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this._dataType || '?') + this._shape.toString();
    }
};

darknet.TensorShape = class {

    constructor(dimensions) {
        if (dimensions.some((dimension) => dimension === 0 || dimension === undefined || isNaN(dimension))) {
            throw new darknet.Error("Invalid tensor shape '" + JSON.stringify(dimensions) + "'.");
        }
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length == 0) {
                return '';
            }
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

darknet.Weights = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
        const major = this.int32();
        const minor = this.int32();
        const revision = this.int32();
        this._seen = ((major * 10 + minor) >= 2) ? this.int64() : this.int32();
        const transpose = (major > 1000) || (minor > 1000);
        if (transpose) {
            throw new darknet.Error("Unsupported transpose weights file version '" + [ major, minor, revision ].join('.') + "'.");
        }
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    int64() {
        let hi = this.int32();
        let lo = this.int32();
        return new long.Long(hi, lo, true).toNumber();
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new darknet.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    validate() {
        if (this._position !== this._buffer.length) {
            throw new darknet.Error('Invalid weights size.')
        }
    }
}

darknet.Metadata = class {

    static open(host) {
        if (darknet.Metadata._metadata) {
            return Promise.resolve(darknet.Metadata._metadata);
        }
        return host.request(null, 'darknet-metadata.json', 'utf-8').then((data) => {
            darknet.Metadata._metadata = new darknet.Metadata(data);
            return darknet.Metadata._metadata;
        }).catch(() => {
            darknet.Metadata._metadata = new darknet.Metadata(null);
            return darknet.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        this._attributeMap = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item && item.name && item.schema) {
                        if (this._map.has(item.name)) {
                            throw new darknet.Error("Duplicate metadata key '" + item.name + "'.");
                        }
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(operator) {
        return this._map.get(operator) || null;
    }

    attribute(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeMap.has(key)) {
            this._attributeMap.set(key, null);
            const schema = this.type(operator);
            if (schema && schema.attributes) {
                for (const attribute of schema.attributes) {
                    this._attributeMap.set(operator + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributeMap.get(key);
    }
};

darknet.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Darknet model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = darknet.ModelFactory;
}
