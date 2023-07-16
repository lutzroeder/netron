
var torch = torch || {};

torch.ModelFactory = class {

    match(context) {
        return torch.T7Reader.open(context);
    }

    async open(context, target) {
        const metadata = await context.metadata('torch-metadata.json');
        const reader = target;
        reader.callback = (name) => {
            if (name && name != 'nn.JointTrainModule' && !name.startsWith('nn.MSDNet_') && !name.startsWith('onmt.')) {
                context.exception(new torch.Error("Unsupported type '" + name + "'."));
            }
            return null;
        };
        const obj = reader.read();
        let graphs = [];
        if (obj && Array.isArray(obj) && obj.length >= 2 &&
            obj.slice(0, obj.length - 1).every((item) => item.__class__) &&
            !obj[obj.length - 1].__class__) {
            graphs = obj.slice(0, obj.length - 1);
        } else {
            graphs = [ obj ];
        }
        return new torch.Model(metadata, graphs);
    }
};

torch.Model = class {

    constructor(metadata, graphs) {
        this._graphs = graphs.map((graph, index) => new torch.Graph(metadata, index.toString(), graph));
    }

    get graphs() {
        return this._graphs;
    }

    get format() {
        return 'Torch v7';
    }
};

torch.Graph = class {

    constructor(metadata, name, root) {
        this._name = name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._groups = 'false';
        const args = new Map();
        const arg = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new torch.Value(name, type || null, tensor || null);
            }
            if (!args.has(name)) {
                args.set(name, new torch.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new torch.Error("Duplicate value '" + name + "'.");
            }
            return args.get(name);
        };
        if (Object.prototype.hasOwnProperty.call(root, 'model')) {
            root = root.model;
        }
        const loadModule = (metadata, module, groups, key, inputs, outputs) => {
            if (groups.length > 0) {
                this._groups = true;
            }
            const type = module.__class__ ? module.__class__.__module__ + '.' + module.__class__.__name__ : '';
            switch (type) {
                case 'nn.Sequential': {
                    groups.push(key);
                    let subInputs = inputs;
                    let subOutputs = [];
                    const length = module.modules.length;
                    let index = 0;
                    for (const subModule of module.modules) {
                        if (index == length - 1) {
                            subOutputs = outputs;
                        }
                        loadModule(metadata, subModule, groups, index.toString(), subInputs, subOutputs);
                        subInputs = subOutputs;
                        subOutputs = [];
                        index++;
                    }
                    groups.pop();
                    break;
                }
                case 'nn.Parallel':
                case 'nn.ParallelTable':
                case 'nn.JointTrain': {
                    groups.push(key);
                    let newInputs = [];
                    let newOutputs = [];
                    let index = 0;
                    for (const subModule of module.modules) {
                        const subInputs = [].concat(inputs);
                        const subOutputs = [].concat(outputs);
                        loadModule(metadata, subModule, groups, index.toString(), subInputs, subOutputs);
                        if (inputs.length == 0) {
                            newInputs = newInputs.concat(subInputs);
                        }
                        if (outputs.length == 0) {
                            newOutputs = newOutputs.concat(subOutputs);
                        }
                        index++;
                    }
                    // inputs = inputs.concat(newInputs);
                    for (const newOutput of newOutputs) {
                        outputs.push(newOutput);
                    }
                    groups.pop();
                    break;
                }
                case 'nn.Concat':
                case 'nn.ConcatTable': {
                    const prefix = key;
                    if (inputs.length == 0) {
                        inputs.push(arg(groups.join('/') + ':' + key + ':in', null, null));
                    }
                    let concatInputs = [];
                    let index = 0;
                    for (const subModule of module.modules) {
                        const streamInputs = inputs.map((input) => input);
                        const streamOutputs = [];
                        loadModule(metadata, subModule, groups, prefix + '.' + index.toString(), streamInputs, streamOutputs);
                        concatInputs = concatInputs.concat(streamOutputs);
                        index++;
                    }
                    delete module.modules;
                    delete module.dimension;
                    this._nodes.push(new torch.Node(metadata, module, groups, key, inputs, outputs, arg));
                    break;
                }
                case 'nn.Inception': {
                    delete module.modules; // TODO
                    delete module.module; // TODO
                    delete module.transfer; // TODO
                    delete module.pool; // TODO
                    this._nodes.push(new torch.Node(metadata, module, groups, key, inputs, outputs, arg));
                    break;
                }
                case 'nn.gModule': {
                    /*
                    let index = 0;
                    for (const subModule of module.modules) {
                        subModule.modules = [];
                        this._loadModule(metadata, subModule, groups, index.toString(), [], []);
                        index++;
                    }
                    */
                    this._nodes.push(new torch.Node(metadata, module, groups, key, inputs, outputs, arg));
                    break;
                }
                default: {
                    this._nodes.push(new torch.Node(metadata, module, groups, key, inputs, outputs, arg));
                    break;
                }
            }
        };
        const inputs = [];
        const outputs = [];
        loadModule(metadata, root, [], '', inputs, outputs);
        this._inputs = this._inputs.concat(inputs.map((input, index) => {
            return new torch.Argument('input' + (index != 0 ? (index + 1).toString() : ''), [ input ]);
        }));
        this._outputs = this._outputs.concat(outputs.map((output, index) => {
            return new torch.Argument('output' + (index != 0 ? (index + 1).toString() : ''), [ output ]);
        }));
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

    get groups() {
        return this._groups;
    }
};

torch.Argument = class {

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

torch.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new torch.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

torch.Node = class {

    constructor(metadata, module, groups, name, inputs, outputs, arg) {
        this._group = groups.join('/');
        if (module.name && typeof module.name === 'string') {
            this._name = module.name;
            delete module.name;
        } else {
            this._name = this._group ? (this._group + ':' + name) : name;
        }
        const type = module.__class__ ? module.__class__.__module__ + '.' + module.__class__.__name__ : 'nn.Module';
        this._type = metadata.type(type);
        let initializers = [];
        for (const entry of Object.entries(module)) {
            const key = entry[0];
            const obj = entry[1];
            if (obj && obj.__class__ && obj.__class__.__module__ === 'torch' && obj.__class__.__name__.endsWith('Storage')) {
                module[key] = obj.data();
            }
        }
        delete module.iSize;
        delete module.finput;
        delete module.fgradInput;
        delete module.output;
        delete module.gradInput;
        delete module.gradWeight;
        delete module.gradBias;
        delete module.grad_tmp;
        delete module.scaleT;
        delete module._input;
        delete module._output;
        delete module._gradInput;
        delete module._gradOutput;
        delete module.buffer;
        delete module.buffer2;
        delete module.tmp_in;
        delete module.tmp_out;
        delete module.accUpdateGradParameters;
        switch (this._type.name) {
            case 'nn.Linear':
                delete module.addBuffer;
                break;
            case 'nn.Normalize':
            case 'nn.Normalize2':
                delete module.addBuffer;
                delete module.normp;
                delete module.norm;
                break;
            case 'cudnn.SpatialConvolution':
            case 'cudnn.SpatialFullConvolution':
            case 'nn.SpatialConvolution':
            case 'nn.SpatialConvolutionMM':
            case 'nn.SpatialConvolution1_fw':
            case 'nn.SpatialDilatedConvolution':
            case 'nn.SpatialFullConvolution':
                delete module.ones;
                delete module.input_slice;
                delete module.output_slice;
                delete module.convDescData;
                this._updateSize(module, 'adj');
                this._updateSize(module, 'd');
                this._updateSize(module, 'dilation');
                this._updateSize(module, 'k');
                this._updateSize(module, 'pad');
                break;
            case 'cudnn.BatchNormalization':
            case 'cudnn.SpatialBatchNormalization':
            case 'nn.BatchNormalization':
            case 'nn.SpatialBatchNormalization':
            case 'nn.InstanceNormalization':
                delete module.save_mean;
                delete module.save_std;
                delete module.gradWeight;
                delete module.normalized;
                delete module.centered;
                delete module.bn; // TODO InstanceNormalization
                break;
            case 'nn.SpatialCrossMapLRN':
                delete module.scale;
                break;
            case 'cudnn.SpatialMaxPooling':
            case 'cudnn.SpatialAveragePooling':
            case 'inn.SpatialMaxPooling':
            case 'nn.SpatialMaxPooling':
            case 'nn.SpatialAveragePooling':
                delete module.indices;
                this._updateSize(module, 'pad');
                this._updateSize(module, 'd');
                this._updateSize(module, 'k');
                break;
            case 'nn.SpatialZeroPadding':
            case 'nn.SpatialReflectionPadding':
            case 'nn.SpatialReplicationPadding':
                this._updateBox(module, 'pad');
                break;
            case 'nn.Dropout':
                delete module.noise;
                break;
            case 'nn.gModule':
                delete module.forwardnodes;
                delete module.backwardnodes;
                break;
            case 'nn.StereoJoin':
                delete module.output_L;
                break;
            default:
                break;
        }
        this._attributes = [];
        if (module.__class__) {
            for (const entry of Object.entries(module)) {
                const key = entry[0];
                const obj = entry[1];
                if (key == '_type') {
                    continue;
                }
                if (Array.isArray(obj) && obj.every(((item) => item && item.__class__ && item.__class__.__module__ === 'nn'))) {
                    continue;
                }
                if (obj.__class__ && obj.__class__.__module__ === 'torch' && obj.__class__.__name__.endsWith('Tensor')) {
                    initializers.push(new torch.Argument(key, [ arg('', null, new torch.Tensor(obj)) ]));
                    continue;
                }
                if (key == 'modules') {
                    continue;
                }
                if (obj.__class__ && obj.__class__.__module__ !== '' && obj.__class__.__name__ != 'LuaFunction') {
                    continue;
                }
                const attribute = new torch.Attribute(metadata, type, key, obj);
                this._attributes.push(attribute);
            }
        }
        this._inputs = [];
        if (inputs.length == 0 && this._name) {
            inputs.push(arg(this._name + ':in'));
        }
        this._inputs.push(new torch.Argument('input', inputs));
        if (outputs.length == 0 && this._name) {
            outputs.push(arg(this._name));
        }
        this._outputs = [];
        this._outputs.push(new torch.Argument('output', outputs));
        initializers = initializers.filter((argument) => {
            if (argument.name == 'weight') {
                this._inputs.push(argument);
                return false;
            }
            return true;
        });
        initializers = initializers.filter((argument) => {
            if (argument.name == 'bias') {
                this._inputs.push(argument);
                return false;
            }
            return true;
        });
        this._inputs = this._inputs.concat(initializers);
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get group() {
        return this._group;
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

    _updateSize(module, name) {
        if (Object.prototype.hasOwnProperty.call(module, name + 'W') &&
            Object.prototype.hasOwnProperty.call(module, name + 'H')) {
            module[name] = [ module[name + 'W'], module[name + 'H'] ];
            delete module[name + 'W'];
            delete module[name + 'H'];
        }
    }

    _updateBox(module, name) {
        if (Object.prototype.hasOwnProperty.call(module, name + '_t') &&
            Object.prototype.hasOwnProperty.call(module, name + '_r') &&
            Object.prototype.hasOwnProperty.call(module, name + '_b') &&
            Object.prototype.hasOwnProperty.call(module, name + '_l')) {
            module[name] = [ module[name + '_t'], module[name + '_r'], module[name + '_b'], module[name + '_l'] ];
            delete module[name + '_t'];
            delete module[name + '_r'];
            delete module[name + '_b'];
            delete module[name + '_l'];
        }
    }
};

torch.Attribute = class {

    constructor(metadata, type, name, value) {
        this._name = name;
        this._value = value;
        if (name == 'train') {
            this._visible = false;
        }
        metadata = metadata.attribute(type, name);
        if (metadata) {
            if (metadata.visible === false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (JSON.stringify(metadata.default) == JSON.stringify(this._value)) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

torch.Tensor = class {

    constructor(tensor) {
        this._type = new torch.TensorType(tensor);
        this._storage = tensor.storage;
        this._offset = tensor.storage_offset;
    }

    get type() {
        return this._type;
    }

    get layout() {
        return '|';
    }

    get values() {
        if (this._type.shape.dimensions.length === 0) {
            return [];
        }
        if (this._storage) {
            const data = this._storage.data();
            if (data) {
                const size = this._type.shape.dimensions.reduce((a, b) => a * b, 1);
                return data.slice(this._offset, this._offset + size);
            }
        }
        return null;
    }
};

torch.TensorType = class {

    constructor(tensor) {
        this._dataType = tensor.dataType;
        this._shape = new torch.TensorShape(tensor.size);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

torch.TensorShape = class {

    constructor(dimensions) {
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

torch.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Torch model.';
    }
};

torch.T7Reader = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length >= 4 && stream.peek(4).every((value, index) => value === 0x00 || (index == 0 && value <= 0x08))) {
            const reader = new torch.BinaryReader(stream);
            return new torch.T7Reader(reader);
        }
        if (stream && stream.length >= 2) {
            const buffer = stream.peek(2);
            const value = String.fromCharCode(stream.peek(1)[0]);
            if (buffer[1] === 0x0a && (value >= '0' && value <= '8')) {
                const reader = new torch.TextReader(stream);
                return new torch.T7Reader(reader);
            }
        }
        return null;
    }

    constructor(reader) {
        this._reader = reader;
        this._memo = new Map();
        this._types = new Map();
        const Storage = class {
            constructor(dataType, itemSize) {
                this.dataType = dataType;
                this.itemSize = itemSize;
            }
            data() {
                if (this.reader) {
                    const reader = this.reader;
                    reader.reset();
                    const dataType = this.dataType;
                    const size = this.size;
                    const array = new Array(size);
                    for (let i = 0; i < size; i++) {
                        switch (dataType) {
                            case 'uint8':
                                array[i] = reader.byte();
                                break;
                            case 'int8':
                                array[i] = reader.int8();
                                break;
                            case 'int16':
                                array[i] = reader.int16();
                                break;
                            case 'int32':
                                array[i] = reader.int32();
                                break;
                            case 'int64':
                                array[i] = reader.int64();
                                break;
                            case 'float32':
                                array[i] = reader.float32();
                                break;
                            case 'float64':
                                array[i] = reader.float64();
                                break;
                            default:
                                throw new torch.Error("Unsupported data type '" + dataType + "'.");
                        }
                    }
                    this._data = array;
                    delete this.reader;
                }
                return this._data;
            }
            read(reader) {
                this.size = reader.int64();
                this.reader = reader.storage(this.size, this.itemSize, this.dataType);
            }
        };
        const Tensor = class {
            constructor(dataType) {
                this.dataType = dataType;
            }
            read(reader) {
                const dim = reader.int32();
                this.size = reader.int64s(dim);
                this.stride = reader.int64s(dim);
                this.storage_offset = reader.int64() - 1;
                this.storage = reader.read();
            }
        };
        this.register('bnn.Binary');
        this.register('bnn.SpatialConvolution');
        this.register('cudnn.BatchNormalization');
        this.register('cudnn.BatchBRNNReLU');
        this.register('cudnn.BLSTM');
        this.register('cudnn.ReLU');
        this.register('cudnn.RNN');
        this.register('cudnn.Sigmoid');
        this.register('cudnn.SoftMax');
        this.register('cudnn.LogSoftMax');
        this.register('cudnn.normal3DConv');
        this.register('cudnn.normal3DdeConv');
        this.register('cudnn.SpatialAveragePooling');
        this.register('cudnn.SpatialBatchNormalization');
        this.register('cudnn.SpatialConvolution');
        this.register('cudnn.SpatialFullConvolution');
        this.register('cudnn.SpatialMaxPooling');
        this.register('cudnn.SpatialSoftMax');
        this.register('cudnn.Tanh');
        this.register('cudnn.VolumetricAveragePooling');
        this.register('cudnn.VolumetricBatchNormalization');
        this.register('cudnn.VolumetricConvolution');
        this.register('cudnn.VolumetricMaxPooling');
        this.register('Dict');
        this.register('inn.ConstAffine');
        this.register('inn.SpatialMaxPooling');
        this.register('nn.Abs');
        this.register('nn.AddConstant');
        this.register('nn.BatchNormalization');
        this.register('nn.BilinearSamplerBHWD');
        this.register('nn.BinActiveZ'); // allenai/XNOR-Net
        this.register('nn.BCECriterion');
        this.register('nn.Bottle');
        this.register('nn.Clamp');
        this.register('nn.CMul');
        this.register('nn.CAddTable');
        this.register('nn.CDivTable');
        this.register('nn.CMulTable');
        this.register('nn.CSubTable');
        this.register('nn.Concat');
        this.register('nn.Copy');
        this.register('nn.ConcatTable');
        this.register('nn.Contiguous');
        this.register('nn.Constant');
        this.register('nn.CostVolMulti');
        this.register('nn.DataParallelTable');
        this.register('nn.DepthConcat');
        this.register('nn.Dropout');
        this.register('nn.Exp');
        this.register('nn.ExpOut');
        this.register('nn.FlattenTable');
        this.register('nn.GenNoise');
        this.register('nn.Identity');
        this.register('nn.Index');
        this.register('nn.Inception');
        this.register('nn.InstanceNormalization');
        this.register('nn.JoinTable');
        this.register('nn.JointTrain');
        this.register('nn.KeypointCoordinate');
        this.register('nn.LeakyReLU');
        this.register('nn.Linear');
        this.register('nn.LinearNoBias');
        this.register('nn.LogSoftMax');
        this.register('nn.LookupTable');
        this.register('nn.LSTM');
        this.register('nn.MaskZero');
        this.register('nn.MapTable');
        this.register('nn.Max');
        this.register('nn.Mean');
        this.register('nn.Min');
        this.register('nn.MulConstant');
        this.register('nn.MM');
        this.register('nn.MSECriterion');
        this.register('nn.Narrow');
        this.register('nn.NarrowTable');
        this.register('nn.Normalize');
        this.register('nn.Normalize2');
        this.register('nn.NoiseFill');
        this.register('nn.Padding');
        this.register('nn.Parallel');
        this.register('nn.ParallelCriterion');
        this.register('nn.ParallelTable');
        this.register('nn.PixelShuffle');
        this.register('nn.Power');
        this.register('nn.PReLU');
        this.register('nn.Recursor');
        this.register('nn.ReLU');
        this.register('nn.Replicate');
        this.register('nn.Reshape');
        this.register('nn.ShaveImage');
        this.register('nn.Select');
        this.register('nn.SelectTable');
        this.register('nn.Sequencer');
        this.register('nn.Sequential');
        this.register('nn.Sigmoid');
        this.register('nn.Sum');
        this.register('nn.SoftMax');
        this.register('nn.SpatialAveragePooling');
        this.register('nn.SpatialBatchNormalization');
        this.register('nn.SpatialConvolution');
        this.register('nn.SpatialConvolution1_fw');
        this.register('nn.SpatialConvolutionMM');
        this.register('nn.SpatialCrossMapLRN');
        this.register('nn.SpatialDilatedConvolution');
        this.register('nn.SpatialDropout');
        this.register('nn.SpatialFractionalMaxPooling');
        this.register('nn.SpatialFullConvolution');
        this.register('nn.SpatialLPPooling');
        this.register('nn.SpatialMaxPooling');
        this.register('nn.SpatialMaxUnpooling');
        this.register('nn.SpatialReflectionPadding');
        this.register('nn.SpatialReplicationPadding');
        this.register('nn.SpatialSoftMax');
        this.register('nn.SpatialSubtractiveNormalization');
        this.register('nn.SpatialUpSamplingBilinear');
        this.register('nn.SpatialUpSamplingNearest');
        this.register('nn.SpatialZeroPadding');
        this.register('nn.SplitTable');
        this.register('nn.Squeeze');
        this.register('nn.Square');
        this.register('nn.Sqrt');
        this.register('nn.StereoJoin');
        this.register('nn.Tanh');
        this.register('nn.Transpose');
        this.register('nn.TotalVariation');
        this.register('nn.Unpool');
        this.register('nn.View');
        this.register('nn.gModule');
        this.register('nngraph.Node');
        this.register('graph.Edge');
        this.register('graph.Graph');
        this.register('torch.ByteTensor', class extends Tensor {
            constructor() {
                super('uint8');
            }
        });
        this.register('torch.CharTensor', class extends Tensor {
            constructor() {
                super('int8');
            }
        });
        this.register('torch.ShortTensor', class extends Tensor {
            constructor() {
                super('int16');
            }
        });
        this.register('torch.IntTensor', class extends Tensor {
            constructor() {
                super('int32');
            }
        });
        this.register('torch.LongTensor', class extends Tensor {
            constructor() {
                super('int64');
            }
        });
        this.register('torch.FloatTensor', class extends Tensor {
            constructor() {
                super('float32');
            }
        });
        this.register('torch.DoubleTensor', class extends Tensor {
            constructor() {
                super('float64');
            }
        });
        this.register('torch.CudaByteTensor', class extends Tensor {
            constructor() {
                super('uint8');
            }
        });
        this.register('torch.CudaCharTensor', class extends Tensor {
            constructor() {
                super('int8');
            }
        });
        this.register('torch.CudaShortTensor', class extends Tensor {
            constructor() {
                super('int16');
            }
        });
        this.register('torch.CudaIntTensor', class extends Tensor {
            constructor() {
                super('int32');
            }
        });
        this.register('torch.CudaLongTensor', class extends Tensor {
            constructor() {
                super('int64');
            }
        });
        this.register('torch.CudaTensor', class extends Tensor {
            constructor() {
                super('float32');
            }
        });
        this.register('torch.CudaDoubleTensor', class extends Tensor {
            constructor() {
                super('float64');
            }
        });
        this.register('torch.ByteStorage', class extends Storage {
            constructor() {
                super('uint8', 1);
            }
        });
        this.register('torch.CharStorage', class extends Storage {
            constructor() {
                super('int8', 1);
            }
        });
        this.register('torch.ShortStorage', class extends Storage {
            constructor() {
                super('int16', 2);
            }
        });
        this.register('torch.IntStorage', class extends Storage {
            constructor() {
                super('int32', 4);
            }
        });
        this.register('torch.LongStorage', class extends Storage {
            constructor() {
                super('int64', 8);
            }
        });
        this.register('torch.FloatStorage', class extends Storage {
            constructor() {
                super('float32', 4);
            }
        });
        this.register('torch.DoubleStorage', class extends Storage {
            constructor() {
                super('float64', 8);
            }
        });
        this.register('torch.CudaByteStorage', class extends Storage {
            constructor() {
                super('uint8', 1);
            }
        });
        this.register('torch.CudaCharStorage', class extends Storage {
            constructor() {
                super('int8', 1);
            }
        });
        this.register('torch.CudaShortStorage', class extends Storage {
            constructor() {
                super('int16', 2);
            }
        });
        this.register('torch.CudaIntStorage', class extends Storage {
            constructor() {
                super('int32', 4);
            }
        });
        this.register('torch.CudaLongStorage', class extends Storage {
            constructor() {
                super('int64', 8);
            }
        });
        this.register('torch.CudaIntStorage', class extends Storage {
            constructor() {
                super('int32', 4);
            }
        });
        this.register('torch.CudaStorage', class extends Storage {
            constructor() {
                super('float32', 4);
            }
        });
        this.register('torch.CudaFloatStorage', class extends Storage {
            constructor() {
                super('float64', 8);
            }
        });
        this.register('w2nn.AuxiliaryLossTable');
        this.register('w2nn.InplaceClip01');
        this.register('w2nn.ScaleTable');
        this.register('LuaFunction', class {
            constructor(size, dumped, upvalues) {
                this.size = size;
                this.dumped = dumped;
                this.upvalues = upvalues;
            }
        });
    }

    register(name, type) {
        type = type || class {};
        const parts = name.split('.');
        type.__name__ = parts.pop();
        type.__module__ = parts.join('.');
        type.prototype.__class__ = type;
        this._types.set(name, type);
    }

    read() {
        const type = this.int32();
        switch (type) {
            case 0: return null;
            case 1: return this.float64();
            case 2: return this.string();
            case 3: return this.table();
            case 4: return this.object();
            case 5: return this.boolean();
            case 6: return this.function();
            case 7: return this.function();
            case 8: return this.function();
            default: throw new torch.Error("File format has invalid type '" + type + "'.");
        }
    }

    boolean() {
        return this._reader.boolean();
    }

    bytes(size) {
        return this._reader.bytes(size);
    }

    int32() {
        return this._reader.int32();
    }

    int64() {
        return this._reader.int64();
    }

    int64s(size) {
        return this._reader.int64s(size);
    }

    float64() {
        return this._reader.float64();
    }

    string() {
        return this._reader.string();
    }

    object() {
        const index = this.int32();
        if (this._memo.has(index)) {
            return this._memo.get(index);
        }

        let version = this.string();
        let name = null;
        if (version.startsWith('V ')) {
            name = this.string();
            version = Number(version.split(' ')[1]);
        } else {
            name = version;
            version = 0;
        }

        if (!this._types.has(name)) {
            this.callback(name);
            this.register(name);
        }
        const type = this._types.get(name);
        const obj = Reflect.construct(type, []);
        this._memo.set(index, obj);
        if (obj.read) {
            obj.read(this, version);
        } else {
            const attributes = this.read();
            if (attributes != null) {
                for (const entry of Object.entries(attributes)) {
                    const key = entry[0];
                    obj[key] = entry[1];
                }
            }
        }
        return obj;
    }

    table() {
        const index = this.int32();
        if (this._memo.has(index)) {
            return this._memo.get(index);
        }
        const table = {};
        this._memo.set(index, table);
        const size = this.int32();
        let convert = true;
        let sum = 0;
        for (let i = 0; i < size; i++) {
            const key = this.read();
            const value = this.read();
            table[key] = value;
            if (Number.isInteger(key) && key >= 0) {
                sum += key;
            } else {
                convert = false;
            }
        }
        const n = Object.keys(table).length;
        if (convert && (n * (n + 1)) == (2 * sum)) {
            const list = [];
            for (let j = 0; j < n; j++) {
                let item = table[j + 1];
                if (item == table) {
                    item = list;
                }
                list.push(item);
            }
            this._memo.set(index, list);
            return list;
        }
        return table;
    }

    function() {
        const index = this.int32();
        if (this._memo.has(index)) {
            return this._memo.get(index);
        }
        const size = this.int32();
        const dumped = this.bytes(size);
        const upvalues = this.read();
        const type = this._types.get('LuaFunction');
        const obj = Reflect.construct(type, [ size, dumped, upvalues ]);
        this._memo.set(index, obj);
        return obj;
    }

    storage(size, itemSize, dataType) {
        return this._reader.storage(size, itemSize, dataType);
    }
};

torch.BinaryReader = class {

    constructor(data) {
        this._buffer = data instanceof Uint8Array ? data : data.peek();
        this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        this._position = 0;
        this._textDecoder = new TextDecoder('ascii');
    }

    reset() {
        this._position = 0;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new torch.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    boolean() {
        return this.int32() == 1;
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    int8() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getInt8(position, true);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getInt16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._dataView.getInt64(position, true).toNumber();
    }

    int64s(size) {
        const array = [];
        for (let i = 0; i < size; i++) {
            array.push(this.int64());
        }
        return array;
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._dataView.getFloat64(position, true);
    }

    string() {
        return this._textDecoder.decode(this.bytes(this.int32()));
    }

    storage(size, itemSize) {
        return new torch.BinaryReader(this.bytes(size * itemSize));
    }
};

torch.TextReader = class {

    constructor(data, separator) {
        this._buffer = data instanceof Uint8Array ? data : data.peek();
        this._position = 0;
        this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        this._textDecoder = new TextDecoder('ascii');
        this._separator = separator || 0x0a;
    }

    reset() {
        this._position = 0;
    }

    line(size) {
        const start = this._position;
        while (this._position < this._buffer.length && size > -1) {
            const c = this._buffer[this._position++];
            if (c == this._separator) {
                return this._buffer.slice(start, this._position - 1);
            } else if (this._position == this._buffer.length) {
                return this._buffer.slice(start, this._position);
            }
            size--;
        }
        throw new torch.Error('Line exceeded maximum length.');
    }

    boolean() {
        return this.int32() == 1;
    }

    bytes(size) {
        return this.line(size);
    }

    int8() {
        return this.int64();
    }

    int16() {
        return this.int64();
    }

    int32() {
        return this.int64();
    }

    int64() {
        const token = this._textDecoder.decode(this.line(20));
        const number = Number.parseInt(token, 10);
        if (Number.isNaN(token - number)) {
            throw new torch.Error("Couldn't parse int64 '" + token + "'.");
        }
        return number;
    }

    int64s(size) {
        const array = [];
        if (size > 0) {
            const content = this._textDecoder.decode(this.line(Number.MAX_SAFE_INTEGER));
            for (const token of content.split(' ')) {
                const number = Number.parseInt(token, 10);
                if (Number.isNaN(token - number)) {
                    throw new torch.Error("Couldn't parse int64 '" + token + "'.");
                }
                array.push(number);
            }
        }
        return array;
    }

    float32() {
        return this.float64();
    }

    float64() {
        const token = this._textDecoder.decode(this.line(24));
        if (token.startsWith('-nan')) {
            return -NaN;
        }
        if (token.startsWith('nan')) {
            return NaN;
        }
        if (token.startsWith('inf')) {
            return Infinity;
        }
        if (token.startsWith('-inf')) {
            return -Infinity;
        }
        const number = Number.parseFloat(token);
        if (Number.isNaN(token - number)) {
            throw new torch.Error("Couldn't parse float '" + token + "'.");
        }
        return number;
    }

    string() {
        const size = this.int32();
        if (size == 0) {
            return '';
        }
        const data = this.line(size);
        const content = this._textDecoder.decode(data);
        if (size != content.length) {
            throw new torch.Error('Invalid string length.');
        }
        return content;
    }

    storage(size, itemSize, dataType) {
        if (size <= 0) {
            throw new torch.Error("Unsupported storage size '" + size + "'.");
        }
        if (dataType === 'uint8') {
            const start = this._position;
            this._position += size;
            const bytes = this._buffer.slice(start, this._position);
            this.line(0);
            return new torch.BinaryReader(bytes);
        }
        const data = this.line(Number.MAX_SAFE_INTEGER);
        return new torch.TextReader(data, 0x20);
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = torch.ModelFactory;
}
