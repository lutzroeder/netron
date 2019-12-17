/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var darknet = darknet || {};
var base = base || require('./base');
var marked = marked || require('marked');

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
            return new darknet.Model(metadata, cfg, weights);
        }
        catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            throw new darknet.Error(message + " in '" + identifier + "'.");
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
    
    constructor(metadata, cfg /* weights */) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        // read_cfg
        let sections = [];
        let section = null;
        let lines = cfg.split('\n');
        let nu = 0;
        while (lines.length > 0) {
            nu++;
            const text = lines.shift();
            const line = text.replace(/\s/g, '');
            if (line.length > 0) {
                switch (line[0]) {
                    case '#':
                    case ';':
                        break;
                    case '[': {
                        section = {};
                        section.type = line[line.length - 1] === ']' ? line.substring(1, line.length - 1) : line.substring(1);
                        section.options = {};
                        sections.push(section);
                        break;
                    }
                    default: {
                        if (!section || line[0] < 0x20 || line[0] > 0x7E) {
                            throw new darknet.Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trimStart().trimEnd() + "' at line " + nu.toString() + ".");
                        }
                        if (section) {
                            let property = line.split('=');
                            if (property.length != 2) {
                                throw new darknet.Error("Invalid cfg '" + text.replace(/[^\x20-\x7E]+/g, '').trimStart().trimEnd() + "' at line " + nu.toString() + ".");
                            }
                            let key = property[0].trim();
                            let value = property[1].trim();
                            section.options[key] = value;
                        }
                        break;
                    }
                }
            }
        }

        const option_find_int = (options, key, defaultValue) => {
            const value = options[key];
            return value !== undefined ? parseInt(value, 10) : defaultValue;
        };

        const option_find_str = (options, key, defaultValue) => {
            const value = options[key];
            return value !== undefined ? value : defaultValue;
        };

        let params = {};

        const net = sections.shift();
        switch (net.type) {
            case 'net':
            case 'network': {
                params.h = option_find_int(net.options, 'height', 0);
                params.w = option_find_int(net.options, 'width', 0);
                params.c = option_find_int(net.options, 'channels', 0);
                params.inputs = option_find_int(net.options, 'inputs', params.h * params.w * params.c);
                break;
            }
        }

        const inputType = params.w && params.h && params.c ?
            new darknet.TensorType('float32', new darknet.TensorShape([ params.w, params.h, params.c ])) :
            new darknet.TensorType('float32', new darknet.TensorShape([ params.inputs ]));
        const inputName = 'input';
        params.arguments = [ new darknet.Argument(inputName, inputType, null) ];
        this._inputs.push(new darknet.Parameter(inputName, true, params.arguments));

        if (sections.length === 0) {
            throw new darknet.Error('Config file has no sections.');
        }

        let infer = true;
        for (let i = 0; i < sections.length; i++) {
            let section = sections[i];
            section.layer = {};
            section.tensors = [];
            section.inputs = [];
            section.outputs = [];
            const options = section.options;
            let layer = section.layer;
            section.inputs = section.inputs.concat(params.arguments);
            section.outputs.push(new darknet.Argument(i.toString(), null, null));
            switch (section.type) {
                case 'shortcut':
                case 'sam':
                case 'scale_channels': {
                    let index = option_find_int(options, 'from', 0);
                    if (index < 0) {
                        index = i + index;
                    }
                    const from = sections[index];
                    if (from) {
                        section.inputs.push(from.outputs[0]);
                        section.from = from;
                    }
                    delete options.from;
                    break;
                }
                case 'route': {
                    section.inputs = [];
                    section.input_sections = [];
                    const routes = options.layers.split(',').map((route) => Number.parseInt(route.trim(), 10));
                    for (let j = 0; j < routes.length; j++) {
                        const index = (routes[j] < 0) ? i + routes[j] : routes[j];
                        const route = sections[index];
                        if (route) {
                            section.inputs.push(route.outputs[0]);
                            section.input_sections.push(route);
                        }
                    }
                    delete options.layers;
                    break;
                }
            }
            if (infer) {
                switch (section.type) {
                    case 'convolutional':
                    case 'deconvolutional': {
                        const w = params.w;
                        const h = params.h;
                        const c = params.c;
                        const size = option_find_int(options, 'size', 1);
                        const n = option_find_int(options, 'filters', 1);
                        const pad = option_find_int(options, 'pad', 0);
                        const padding = pad ? (size >> 1) : option_find_int(options, 'padding', 0);
                        const stride = option_find_int(options, 'stride', 1);
                        const groups = option_find_int(options, 'groups', 1);
                        layer.out_w = Math.floor((w + 2 * padding - size) / stride) + 1;
                        layer.out_h = Math.floor((h + 2 * padding - size) / stride) + 1;
                        layer.out_c = n;
                        layer.outputs = layer.out_h * layer.out_w * layer.out_c;
                        section.tensors.push({ name: 'weights', shape: [ Math.floor(c / groups), n, size, size ]});
                        section.tensors.push({ name: 'biases', shape: [ n ]});
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_h, layer.out_w, layer.out_c ]));
                        break;
                    }
                    case 'connected': {
                        const outputs = option_find_int(options, 'output', 1);
                        section.tensors.push({ name: 'weights', shape: [ params.inputs, outputs ] });
                        section.tensors.push({ name: 'biases', shape: [ outputs ] });
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ outputs ]));
                        layer.out_h = 1;
                        layer.out_w = 1;
                        layer.out_c = outputs;
                        layer.outputs = outputs;
                        break;
                    }
                    case 'local': {
                        const shape = section.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw darknet.Error('Layer before avgpool layer must output image.');
                        }
                        const n = option_find_int(options, 'filters' , 1);
                        const size = option_find_int(options, 'size', 1);
                        const stride = option_find_int(options, 'stride', 1);
                        const pad = option_find_int(options, 'pad', 0);
                        layer.out_h = Math.floor((params.h - (pad ? 1 : size)) / stride) + 1;
                        layer.out_w = Math.floor((params.w - (pad ? 1 : size)) / stride) + 1;
                        layer.out_c = n;
                        layer.outputs = layer.out_w * layer.out_h * layer.out_c;
                        section.tensors.push({ name: 'weights', shape: [ params.c, n, size, size, layer.out_h * layer.out_w ]});
                        section.tensors.push({ name: 'biases', shape: [ layer.out_w * layer.out_h * layer.out_c ]});
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'maxpool': {
                        const shape = section.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw darknet.Error('Layer before maxpool layer must output image.');
                        }
                        const stride = option_find_int(options, 'stride', 1);
                        const size = option_find_int(options, 'size', stride);
                        const padding = option_find_int(options, 'padding', size - 1);
                        layer.out_w = Math.floor((params.w + padding - size) / stride) + 1;
                        layer.out_h = Math.floor((params.h + padding - size) / stride) + 1;
                        layer.out_c = params.c;
                        layer.outputs = layer.out_w * layer.out_h * layer.out_c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'avgpool': {
                        const shape = section.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw darknet.Error('Layer before avgpool layer must output image.');
                        }
                        layer.out_w = 1;
                        layer.out_h = 1;
                        layer.out_c = params.c;
                        layer.outputs = layer.out_c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'gru':
                    case 'rnn': 
                    case 'lstm':{
                        const output = option_find_int(options, "output", 1);
                        layer.outputs = output;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ output ]));
                        break;
                    }
                    case 'softmax':
                    case 'dropout': {
                        layer.out_w = params.w;
                        layer.out_h = params.h;
                        layer.out_c = params.c;
                        layer.outputs = params.inputs;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.outputs ]));
                        break;
                    }
                    case 'upsample': {
                        const stride = option_find_int(options, 'stride', 2);
                        layer.out_w = params.w * stride;
                        layer.out_h = params.h * stride;
                        layer.out_c = params.c;
                        layer.outputs = layer.out_w * layer.out_h * layer.out_c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'crop': {
                        const shape = section.inputs[0].type.shape.dimensions;
                        if (shape[0] !== params.w || shape[1] !== params.h || shape[2] !== params.c) {
                            throw darknet.Error('Layer before crop layer must output image.');
                        }
                        const crop_height = option_find_int(options, 'crop_height', 1);
                        const crop_width = option_find_int(options, 'crop_width', 1);
                        layer.out_w = crop_width;
                        layer.out_h = crop_height;
                        layer.out_c = params.c;
                        layer.outputs = layer.out_w * layer.out_h * layer.out_c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'yolo': {
                        const classes = option_find_int(options, 'classes', 20);
                        const n = option_find_int(options, 'num', 1);
                        layer.out_h = params.h;
                        layer.out_w = params.w;
                        layer.out_c = n * (classes + 4 + 1);
                        layer.outputs = layer.out_h * layer.out_w * layer.out_c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'Gaussian_yolo': {
                        const classes = option_find_int(options, 'classes', 20);
                        const n = option_find_int(options, 'num', 1);
                        layer.out_h = params.h;
                        layer.out_w = params.w;
                        layer.out_c = n * (classes + 8 + 1);
                        layer.outputs = layer.out_h * layer.out_w * layer.out_c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_w, layer.out_h, layer.out_c ]));
                        break;
                    }
                    case 'region': {
                        const coords = option_find_int(options, 'coords', 4);
                        const classes = option_find_int(options, 'classes', 20);
                        const num = option_find_int(options, 'num', 1);
                        layer.outputs = params.h * params.w * num * (classes + coords + 1);
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ params.h, params.w, num, (classes + coords + 1) ]));
                        break;
                    }
                    case 'cost': {
                        layer.outputs = params.inputs;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.outputs ]));
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
                        layer.outputs = layer.out_h * layer.out_w * layer.out_c;
                        if (extra) {
                            layer.out_w = 0;
                            layer.out_h = 0; 
                            layer.out_c = 0;
                            layer.outputs = (params.h * params.w * params.c) + extra;
                        }
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.outputs ]));
                        break;
                    }
                    case 'scale_channels': {
                        infer = false;
                        break;
                    }
                    case 'route': {
                        let layers = section.input_sections.map((section) => section.layer);
                        layer.outputs = 0;
                        for (let input_layer of layers) {
                            layer.outputs += input_layer.outputs;
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
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ layer.out_h, layer.out_w, layer.out_c ]));
                        break;
                    }
                    case 'shortcut': {
                        const from = section.from;
                        layer.w = from.layer.out_w;
                        layer.h = from.layer.out_h;
                        layer.c = from.layer.out_c;
                        layer.out_w = params.w;
                        layer.out_h = params.h;
                        layer.out_c = params.c;
                        layer.outputs = params.w * params.h * params.c;
                        section.outputs[0].type = new darknet.TensorType('float32', new darknet.TensorShape([ params.w, params.h, params.c ]));
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
                params.inputs = layer.outputs;
            }
            params.arguments = section.outputs;

            const batch_normalize = option_find_int(section.options, 'batch_normalize', 0);
            if (batch_normalize) {
                let size = -1;
                switch (section.type) {
                    case 'convolutional': {
                        size = option_find_int(options, 'filters', 1);
                        break;
                    }
                    case 'crnn':
                    case 'gru':
                    case 'rnn':
                    case 'lstm':
                    case 'connected': {
                        size = option_find_int(options, 'output', 1);
                        break;
                    }
                }
                if (size < 0) {
                    throw new darknet.Error("Invalid batch_normalize size for '" + section.type + "'.");
                }
                let chain = {};
                chain.type = 'batch_normalize';
                chain.tensors = [
                    { name: 'scale', shape: [ size ] },
                    { name: 'mean', shape: [ size ] },
                    { name: 'variance', shape: [ size ] }
                ];
                section.chain = section.chain || [];
                section.chain.push(chain);
            }
    
            const defaultActivation = section.type === 'shortcut' ? 'linear' : 'logistic';
            const activation = option_find_str(section.options, 'activation', defaultActivation);
            if (activation !== defaultActivation) {
                let chain = {};
                chain.type = activation;
                section.chain = section.chain || [];
                section.chain.push(chain);
            }
        }

        for (let i = 0; i < sections.length; i++) {
            this._nodes.push(new darknet.Node(metadata, net, sections[i], i.toString()));
        }

        if (sections.length > 0) {
            const last = sections[sections.length - 1];
            for (let i = 0; i < last.outputs.length; i++) {
                const outputName = 'output' + (i > 1 ? i.toString() : '');
                this._outputs.push(new darknet.Parameter(outputName, true, [ last.outputs[i] ]));
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

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type;
        this._initializer = initializer;
    }

    get id() {
        return this._id;
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

    constructor(metadata, net, section, name) {
        this._name = name;
        this._metadata = metadata;
        this._operator = section.type;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        if (section.inputs && section.inputs.length > 0) {
            this._inputs.push(new darknet.Parameter(section.inputs.length <= 1 ? 'input' : 'inputs', true, section.inputs));
        }
        if (section.tensors && section.tensors.length > 0) {
            for (let tensor of section.tensors) {
                const type = new darknet.TensorType('float', new darknet.TensorShape(tensor.shape));
                this._inputs.push(new darknet.Parameter(tensor.name, true, [
                    new darknet.Argument('', null, new darknet.Tensor('', type) )
                ]))
            }
        }
        if (section.outputs && section.outputs.length > 0) {
            this._outputs.push(new darknet.Parameter(section.outputs.length <= 1 ? 'output' : 'outputs', true, section.outputs));
        }
        if (section.chain) {
            for (let chain of section.chain) {
                this._chain.push(new darknet.Node(metadata, net, chain, ''));
            }
        }
        if (section.options) {
            for (let key of Object.keys(section.options)) {
                this._attributes.push(new darknet.Attribute(metadata, this._operator, key, section.options[key]));
            }
        }
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._operator;
    }

    get documentation() {
        let schema = this._metadata.getSchema(this._operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (let attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (let input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (let output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            if (schema.references) {
                for (let reference of schema.references) {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get category() {
        const schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
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
        const schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            this._type = schema.type || '';
            switch (this._type) {
                case 'int32': {
                    this._value = parseInt(this._value, 10);
                    break;
                }
                case 'float32': {
                    this._value = parseFloat(this._value);
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

    constructor(id, type) {
        this._id = id;
        this._type = type;
    }

    get kind() {
        return 'Tensor';
    }

    get name() {
        return this._id;
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Tensor data not implemented.';
    }

    get value() {
        return null;
    }

    toString() {
        return '';
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
        if (dimensions.some((dimension) => !dimension)) {
            throw new darknet.Error('Invalid tensor shape.');
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
                for (let item of items) {
                    if (item && item.name && item.schema) {
                        if (this._map.has(item.name)) {
                            throw new darknet.Error("Duplicate metadata key '" + item.name + "'.");
                        }
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map.get(operator) || null;
    }

    getAttributeSchema(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeMap.has(key)) {
            this._attributeMap.set(key, null);
            const schema = this.getSchema(operator);
            if (schema && schema.attributes) {
                for (let attribute of schema.attributes) {
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
