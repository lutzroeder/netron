
// Experimental

var numpy = numpy || {};
var python = python || require('./python');

numpy.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59 ];
        if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            return { name: 'npy' };
        }
        const entries = context.entries('zip');
        if (entries.size > 0 && Array.from(entries.keys()).every((name) => name.endsWith('.npy'))) {
            return { name: 'npz', value: entries };
        }
        const obj = context.open('pkl');
        if (obj) {
            if (numpy.Utility.isTensor(obj)) {
                return { name: 'numpy.ndarray', value: obj };
            }
            if (Array.isArray(obj) && obj.every((obj) => obj && obj.__class__ && obj.__class__.__name__ === 'Network' && (obj.__class__.__module__ === 'dnnlib.tflib.network' || obj.__class__.__module__ === 'tfutil'))) {
                return { name: 'dnnlib.tflib.network', value: obj };
            }
            const weights = numpy.Utility.weights(obj);
            if (weights) {
                return { name: 'pickle', value: weights };
            }
        }
        return undefined;
    }

    open(context, match) {
        let format = '';
        const graphs = [];
        switch (match.name) {
            case 'npy': {
                format = 'NumPy Array';
                const execution = new python.Execution(null);
                const stream = context.stream;
                const buffer = stream.peek();
                const bytes = execution.invoke('io.BytesIO', [ buffer ]);
                const array = execution.invoke('numpy.load', [ bytes ]);
                const layer = { type: 'numpy.ndarray', parameters: [ { name: 'value', tensor: { name: '', array: array } } ] };
                graphs.push({ layers: [ layer ] });
                break;
            }
            case 'npz': {
                format = 'NumPy Zip';
                const layers = new Map();
                const execution = new python.Execution(null);
                for (const entry of match.value) {
                    if (!entry[0].endsWith('.npy')) {
                        throw new numpy.Error("Invalid file name '" + entry.name + "'.");
                    }
                    const name = entry[0].replace(/\.npy$/, '');
                    const parts = name.split('/');
                    const parameterName = parts.pop();
                    const groupName = parts.join('/');
                    if (!layers.has(groupName)) {
                        layers.set(groupName, { name: groupName, parameters: [] });
                    }
                    const layer = layers.get(groupName);
                    const stream = entry[1];
                    const buffer = stream.peek();
                    const bytes = execution.invoke('io.BytesIO', [ buffer ]);
                    let array = execution.invoke('numpy.load', [ bytes ]);
                    if (array.dtype.byteorder === '|' && array.dtype.itemsize !== 1) {
                        if (array.dtype.kind !== 'O') {
                            throw new numpy.Error("Invalid data type '" + array.dataType + "'.");
                        }
                        const unpickler = python.Unpickler.open(array.data);
                        array = unpickler.load((name, args) => execution.invoke(name, args));
                    }
                    layer.parameters.push({
                        name: parameterName,
                        tensor: { name: name, array: array }
                    });
                }
                graphs.push({ layers: Array.from(layers.values()) });
                break;
            }
            case 'pickle': {
                format = 'NumPy Weights';
                const layers = new Map();
                const weights = match.value;
                let separator = '_';
                if (Array.from(weights.keys()).every((key) => key.indexOf('.') !== -1) &&
                    !Array.from(weights.keys()).every((key) => key.indexOf('_') !== -1)) {
                    separator = '.';
                }
                for (const pair of weights) {
                    const name = pair[0];
                    const array = pair[1];
                    const parts = name.split(separator);
                    const parameterName = parts.length > 1 ? parts.pop() : '?';
                    const layerName = parts.join(separator);
                    if (!layers.has(layerName)) {
                        layers.set(layerName, { name: layerName, parameters: [] });
                    }
                    const layer = layers.get(layerName);
                    layer.parameters.push({
                        name: parameterName,
                        tensor: { name: name, array: array }
                    });
                }
                graphs.push({ layers: Array.from(layers.values()) });
                break;
            }
            case 'numpy.ndarray': {
                format = 'NumPy NDArray';
                const layer = {
                    type: 'numpy.ndarray',
                    parameters: [ { name: 'value', tensor: { name: '', array: match.value } } ]
                };
                graphs.push({ layers: [ layer ] });
                break;
            }
            case 'dnnlib.tflib.network': {
                format = 'dnnlib';
                for (const obj of match.value) {
                    const layers = new Map();
                    for (const entry of obj.variables) {
                        const name = entry[0];
                        const value = entry[1];
                        if (numpy.Utility.isTensor(value)) {
                            const parts = name.split('/');
                            const parameterName = parts.length > 1 ? parts.pop() : '?';
                            const layerName = parts.join('/');
                            if (!layers.has(layerName)) {
                                layers.set(layerName, { name: layerName, parameters: [] });
                            }
                            const layer = layers.get(layerName);
                            layer.parameters.push({
                                name: parameterName,
                                tensor: { name: name, array: value }
                            });
                        }
                    }
                    graphs.push({ name: obj.name, layers: Array.from(layers.values()) });
                }
                break;
            }
        }
        const model = new numpy.Model(format, graphs);
        return Promise.resolve(model);
    }
};

numpy.Model = class {

    constructor(format, graphs) {
        this._format = format;
        this._graphs = graphs.map((graph) => new numpy.Graph(graph));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

numpy.Graph = class {

    constructor(graph) {
        this._name = graph.name || '';
        this._nodes = graph.layers.map((layer) => new numpy.Node(layer));
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [];
    }

    get nodes() {
        return this._nodes;
    }
};

numpy.Parameter = class {

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

numpy.Argument = class {

    constructor(name, initializer) {
        if (typeof name !== 'string') {
            throw new numpy.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._initializer.type;
    }

    get initializer() {
        return this._initializer;
    }
};

numpy.Node = class {

    constructor(layer) {
        this._name = layer.name || '';
        this._type = { name: layer.type || 'Module' };
        this._inputs = [];
        for (const parameter of layer.parameters) {
            const initializer = new numpy.Tensor(parameter.tensor.array);
            this._inputs.push(new numpy.Parameter(parameter.name, [
                new numpy.Argument(parameter.tensor.name || '', initializer)
            ]));
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
        return [];
    }

    get attributes() {
        return [];
    }
};

numpy.Tensor = class  {

    constructor(array) {
        this._type = new numpy.TensorType(array.dtype.name, new numpy.TensorShape(array.shape));
        this._data = array.tobytes();
        this._byteorder = array.dtype.byteorder;
        this._itemsize = array.dtype.itemsize;
    }

    get type(){
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
        return numpy.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;
        if (this._byteorder !== '<' && this._byteorder !== '>' && this._type.dataType !== 'uint8' && this._type.dataType !== 'int8') {
            context.state = 'Tensor byte order is not supported.';
            return context;
        }
        if (!this._data || this._data.length == 0) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.itemSize = this._itemsize;
        context.dimensions = this._type.shape.dimensions;
        context.dataType = this._type.dataType;
        context.littleEndian = this._byteorder == '<';
        context.data = this._data;
        context.rawData = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        const littleEndian = context.littleEndian;
        const shape = context.dimensions.length == 0 ? [ 1 ] : context.dimensions;
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.rawData) {
                    switch (context.dataType) {
                        case 'float16':
                            results.push(context.rawData.getFloat16(context.index, littleEndian));
                            break;
                        case 'float32':
                            results.push(context.rawData.getFloat32(context.index, littleEndian));
                            break;
                        case 'float64':
                            results.push(context.rawData.getFloat64(context.index, littleEndian));
                            break;
                        case 'int8':
                            results.push(context.rawData.getInt8(context.index, littleEndian));
                            break;
                        case 'int16':
                            results.push(context.rawData.getInt16(context.index, littleEndian));
                            break;
                        case 'int32':
                            results.push(context.rawData.getInt32(context.index, littleEndian));
                            break;
                        case 'int64':
                            results.push(context.rawData.getInt64(context.index, littleEndian));
                            break;
                        case 'uint8':
                            results.push(context.rawData.getUint8(context.index, littleEndian));
                            break;
                        case 'uint16':
                            results.push(context.rawData.getUint16(context.index, littleEndian));
                            break;
                        case 'uint32':
                            results.push(context.rawData.getUint32(context.index, littleEndian));
                            break;
                    }
                    context.index += context.itemSize;
                    context.count++;
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
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => numpy.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

numpy.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType || '?';
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

numpy.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
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

numpy.Utility = class {

    static isTensor(obj) {
        return obj && obj.__class__ &&
            ((obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray') ||
             (obj.__class__.__module__ === 'numpy.core.memmap' && obj.__class__.__name__ === 'memmap'));
    }

    static weights(obj) {
        const dict = (obj, key) => {
            const dict = key === '' ? obj : obj[key];
            if (dict) {
                const weights = new Map();
                if (dict instanceof Map) {
                    for (const pair of dict) {
                        const key = pair[0];
                        const obj = pair[1];
                        if (numpy.Utility.isTensor(obj)) {
                            weights.set(key, obj);
                            continue;
                        }
                        else if (obj instanceof Map && Array.from(obj).every((pair) => numpy.Utility.isTensor(pair[1]))) {
                            for (const pair of obj) {
                                weights.set(key + '.' + pair[0], pair[1]);
                            }
                            continue;
                        }
                        return null;
                    }
                    return weights;
                }
                else if (!Array.isArray(dict)) {
                    const set = new Set([ 'weight_order', 'lr', 'model_iter', '__class__' ]);
                    for (const entry of Object.entries(dict)) {
                        const key = entry[0];
                        const value = entry[1];
                        if (key) {
                            if (numpy.Utility.isTensor(value)) {
                                weights.set(key, value);
                                continue;
                            }
                            if (set.has(key)) {
                                continue;
                            }
                            if (value && !Array.isArray(value) && Object.entries(value).every((entry) => numpy.Utility.isTensor(entry[1]))) {
                                const name = key;
                                for (const entry of Object.entries(value)) {
                                    weights.set(name + '.' + entry[0], entry[1]);
                                }
                                continue;
                            }
                        }
                        return null;
                    }
                    return weights;
                }
            }
            return null;
        };
        const list = (obj, key) => {
            const list = key === '' ? obj : obj[key];
            if (list && Array.isArray(list)) {
                const weights = new Map();
                for (let i = 0; i < list.length; i++) {
                    const obj = list[i];
                    if (numpy.Utility.isTensor(obj)) {
                        weights.set(i.toString(), obj);
                        continue;
                    }
                    else if (obj instanceof Map && Array.from(obj).every((pair) => numpy.Utility.isTensor(pair[1]))) {
                        for (const pair of obj) {
                            weights.set(i.toString() + '.' + pair[0], pair[1]);
                        }
                        continue;
                    }
                    return null;
                }
                return weights;
            }
        };
        const keys = [ '', 'blobs' ];
        for (const key of keys) {
            const weights = dict(obj, key);
            if (weights && weights.size > 0) {
                return weights;
            }
        }
        for (const key of keys) {
            const weights = list(obj, key);
            if (weights) {
                return weights;
            }
        }
        return null;
    }
};

numpy.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Chainer model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = numpy.ModelFactory;
}
