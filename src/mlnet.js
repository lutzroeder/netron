/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var mlnet = mlnet || {};
var zip = zip || require('./zip');

mlnet.ModelFactory = class {

    match(context) {
        let identifier = context.identifier; 
        let extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'zip' && context.entries.length > 0) {
            if (context.entries.some((e) => e.name.split('\\').shift().split('/').shift() == 'TransformerChain')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        let identifier = context.identifier;
        return mlnet.Metadata.open(host).then((metadata) => {
            try {
                let reader = new mlnet.ModelReader(context.entries);
                return new mlnet.Model(metadata, reader);
            }
            catch (error) {
                let message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new mlnet.Error(message + " in '" + identifier + "'.");
            }
        });
    }
}

mlnet.Model = class {

    constructor(metadata, reader) {
        this._format = "ML.NET";
        if (reader.version && reader.version.length > 0) {
            this._format += ' v' + reader.version;
        }
        this._graphs = [];
        this._graphs.push(new mlnet.Graph(metadata, reader));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
}

mlnet.Graph = class {

    constructor(metadata, reader) {

        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._groups = false;

        if (reader.schema && reader.schema.inputs) {
            for (let input of reader.schema.inputs) {
                this._inputs.push(new mlnet.Parameter(input.name, [
                    new mlnet.Argument(input.name, new mlnet.TensorType(input.type))
                ]));
            }
        }

        let scope = new Map();
        if (reader.transformerChain) {
            this._loadTransformer(metadata, scope, '', reader.transformerChain);
        }
    }

    _loadTransformer(metadata, scope, group, transformer) {
        switch (transformer.__type__) {
            case 'TransformerChain':
                this._loadChain(metadata, scope, transformer.__name__, transformer.chain);
                break;
            case 'Text':
                this._loadChain(metadata, scope, transformer.__name__, transformer.chain);
                break;
            default:
                this._createNode(metadata, scope, group, transformer);
                break;
        }
    }

    _loadChain(metadata, scope, name, chain) {
        this._groups = true;
        let group = name.split('/').splice(1).join('/');
        for (let childTransformer of chain) {
            this._loadTransformer(metadata, scope, group, childTransformer);
        }
    }

    _createNode(metadata, scope, group, transformer) {

        if (transformer.inputs && transformer.outputs) {
            for (let input of transformer.inputs) {
                input.name = scope[input.name] ? scope[input.name].argument : input.name;
            }
            for (let output of transformer.outputs) {
                if (scope[output.name]) {
                    scope[output.name].counter++;
                    var next = output.name + '|' + scope[output.name].counter.toString(); // custom argument id
                    scope[output.name].argument = next;
                    output.name = next;
                }
                else {
                    scope[output.name] = {
                        argument: output.name,
                        counter: 0
                    };
                }
            }
        }

        let node = new mlnet.Node(metadata, group, transformer);
        this._nodes.push(node);
    }

    get groups() {
        return this._groups;
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
}

mlnet.Parameter = class {

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
}

mlnet.Argument = class {

    constructor(id, type) {
        this._id = id;
        this._type = type;
    }

    get id() {
        return this._id;
    }

    get type() {
        return this._type;
    }
}

mlnet.Node = class {

    constructor(metadata, group, transformer) {
        this._group = group;
        this._name = transformer.__name__;
        this._operator = transformer.__type__;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        if (transformer.inputs) {
            let i = 0;
            for (let input of transformer.inputs) {
                this._inputs.push(new mlnet.Parameter(i.toString(), [
                    new mlnet.Argument(input.name)
                ]));
                i++;
            }
        }

        if (transformer.outputs) {
            let i = 0;
            for (let output of transformer.outputs) {
                this._outputs.push(new mlnet.Parameter(i.toString(), [
                    new mlnet.Argument(output.name)
                ]));
                i++;
            }
        }

        for (let key of Object.keys(transformer).filter((key) => !key.startsWith('_') && key !== 'inputs' && key !== 'outputs')) {
            this._attributes.push(new mlnet.Attribute(metadata, key, transformer[key]));
        }
    }

    get group() {
        return this._group;
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get documentation() {
        return '';
    }

    get category() {
        return '';
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
}

mlnet.Attribute = class {

    constructor(metadata, name, value)
    {
        this._name = name;
        this._value = value;
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
}

mlnet.TensorType = class {

    constructor(codec) {

        mlnet.TensorType._map = mlnet.TensorType._map || new Map([ 
            [ 'Boolean', 'boolean' ],
            [ 'Single', 'float32' ],
            [ 'Double', 'float64' ],
            [ 'UInt32', 'uint32' ],
            [ 'TextSpan', 'string' ]
        ]);

        switch (codec.name) {
            case 'VBuffer':
                if (mlnet.TensorType._map.has(codec.itemType.name)) {
                    this._dataType = mlnet.TensorType._map.get(codec.itemType.name);
                }
                this._shape = new mlnet.TensorShape(codec.dims);
                break;
            case 'Key2':
                this._dataType = 'key2';
                break;
            default:
                if (mlnet.TensorType._map.has(codec.name)) {
                    this._dataType = mlnet.TensorType._map.get(codec.name);
                }
                break;
        }

        if (!this._dataType) {
            this._dataType = '?';
        }
        if (!this._shape) {
            this._shape = new mlnet.TensorShape(null);
        }
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
}

mlnet.TensorShape = class {

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


mlnet.Metadata = class {

    static open(host) {
        if (mlnet.Metadata._metadata) {
            return Promise.resolve(mlnet.Metadata._metadata);
        }
        return host.request(null, 'mlnet-metadata.json', 'utf-8').then((data) => {
            mlnet.Metadata._metadata = new mlnet.Metadata(data);
            return mlnet.Metadata._metadata;
        }).catch(() => {
            mlnet.Metadata._metadata = new mlnet.Metadata(null);
            return mlnet.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            let schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (let attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

mlnet.ModelReader = class {

    constructor(entries) {

        let catalog = new mlnet.ComponentCatalog();
        catalog.register('AnomalyPredXfer', mlnet.AnomalyPredictionTransformer);
        catalog.register('BinaryPredXfer', mlnet.BinaryPredictionTransformer);
        catalog.register('CaliPredExec', mlnet.CalibratedPredictor);
        catalog.register('CharToken', mlnet.TokenizingByCharactersTransformer);
        catalog.register('ClusteringPredXfer', mlnet.ClusteringPredictionTransformer);
        catalog.register('ConcatTransform', mlnet.ColumnConcatenatingTransformer);
        catalog.register('CopyTransform', mlnet.ColumnCopyingTransformer);
        catalog.register('ConvertTransform', mlnet.TypeConvertingTransformer);
        catalog.register('FAFMPredXfer', mlnet.FieldAwareFactorizationMachinePredictionTransformer);
        catalog.register('FastForestBinaryExec', mlnet.FastForestClassificationPredictor);
        catalog.register('FastTreeTweedieExec', mlnet.FastTreeTweedieModelParameters);
        catalog.register('FastTreeRegressionExec', mlnet.FastTreeRegressionModelParameters);
        catalog.register('FeatWCaliPredExec', mlnet.FeatureWeightsCalibratedModelParameters);
        catalog.register('FieldAwareFactMacPredict', mlnet.FieldAwareFactorizationMachineModelParameters);
        catalog.register('GcnTransform', mlnet.LpNormNormalizingTransformer);
        catalog.register('IidChangePointDetector', mlnet.IidChangePointDetector);
        catalog.register('IidSpikeDetector', mlnet.IidSpikeDetector);
        catalog.register('ImageLoaderTransform', mlnet.ImageLoadingTransformer);
        catalog.register('ImageScalerTransform', mlnet.ImageResizingTransformer);
        catalog.register('ImagePixelExtractor', mlnet.ImagePixelExtractingTransformer);
        catalog.register('KeyToValueTransform', mlnet.KeyToValueMappingTransformer);
        catalog.register('KeyToVectorTransform', mlnet.KeyToVectorMappingTransformer);
        catalog.register('KMeansPredictor', mlnet.KMeansModelParameters);
        catalog.register('LinearRegressionExec', mlnet.LinearRegressionModelParameters);
        catalog.register('LightGBMRegressionExec', mlnet.LightGbmRegressionModelParameters);
        catalog.register('PMixCaliPredExec', mlnet.ParameterMixingCalibratedModelParameters);
        catalog.register('MulticlassLinear', mlnet.LinearMulticlassModelParameters);
        catalog.register('MultiClassLRExec', mlnet.MulticlassLogisticRegressionPredictor);
        catalog.register('MulticlassPredXfer', mlnet.MulticlassPredictionTransformer);
        catalog.register('Normalizer', mlnet.NormalizingTransformer);
        catalog.register('NgramTransform', mlnet.NgramExtractingTransformer);
        catalog.register('OnnxTransform', mlnet.OnnxTransformer);
        catalog.register('OVAExec', mlnet.OneVersusAllModelParameters);
        catalog.register('pcaAnomExec', mlnet.PcaModelParameters);
        catalog.register('PcaTransform', mlnet.PrincipalComponentAnalysisTransformer);
        catalog.register('PipeDataLoader', mlnet.CompositeDataLoader);
        catalog.register('PlattCaliExec', mlnet.PlattCalibrator);
        catalog.register('PoissonRegressionExec', mlnet.PoissonRegressionModelParameters);
        catalog.register('RegressionPredXfer', mlnet.RegressionPredictionTransformer);
        catalog.register('RowToRowMapper', mlnet.RowToRowMapperTransform);
        catalog.register('SelectColumnsTransform', mlnet.ColumnSelectingTransformer);
        catalog.register('TensorFlowTransform', mlnet.TensorFlowTransformer);
        catalog.register('TermTransform', mlnet.ValueToKeyMappingTransformer);
        catalog.register('Text', mlnet.TextFeaturizingEstimator);
        catalog.register('TextLoader', mlnet.TextLoader);
        catalog.register('TextNormalizerTransform', mlnet.TextNormalizingTransformer);
        catalog.register('TokenizeTextTransform', mlnet.WordTokenizingTransformer);
        catalog.register('TransformerChain', mlnet.TransformerChain);
        catalog.register('ValueMappingTransformer', mlnet.ValueMappingTransformer)

        let root = new mlnet.ModelHeader(entries, catalog, '');

        let version = root.openText('TrainingInfo/Version.txt');
        if (version) {
            this.version = version.split(' ').shift().split('\r').shift();
        }

        let schema = root.openBinary('Schema');
        if (schema) {
            this.schema = schema.schema;
        }

        let transformerChain = root.open('TransformerChain');
        if (transformerChain) {
            this.transformerChain = transformerChain.create();
        }
    }
};

mlnet.ComponentCatalog = class {

    constructor() {
        this._map = new Map();
    }

    register(signature, type) {
        this._map.set(signature, type);
    }

    create(signature, context) {
        if (!this._map.has(signature)) {
            throw new mlnet.Error("Unknown loader signature '" + signature + "'.");
        }
        let type = this._map.get(signature);
        return Reflect.construct(type, [ context ]);
    }
};

mlnet.ModelHeader = class {

    constructor(entries, catalog, directory) {

        this._entries = entries;
        this._catalog = catalog;
        this._directory = directory;

        let dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        let name = dir + 'Model.key';
        let entry = entries.find((entry) => entry.name == name || entry.name == name.replace(/\//g, '\\'));
        if (entry) {
            let reader = new mlnet.Reader(entry.data);

            let textDecoder = new TextDecoder('ascii');
            reader.assert('ML\0MODEL');
            this.versionWritten = reader.uint32();
            this.versionReadable = reader.uint32();
    
            let modelBlockOffset = reader.uint64();
            /* let modelBlockSize = */ reader.uint64();
            let stringTableOffset = reader.uint64();
            let stringTableSize = reader.uint64();
            let stringCharsOffset = reader.uint64();
            /* v stringCharsSize = */ reader.uint64();
            this.modelSignature = textDecoder.decode(reader.bytes(8));
            this.modelVersionWritten = reader.uint32();
            this.modelVersionReadable = reader.uint32();
            this._loaderSignature = textDecoder.decode(reader.bytes(24).filter((c) => c != 0));
            this._loaderSignatureAlt = textDecoder.decode(reader.bytes(24).filter((c) => c != 0));
            let tailOffset = reader.uint64();
            /* let tailLimit = */ reader.uint64();
            let assemblyNameOffset = reader.uint64();
            let assemblyNameSize = reader.uint32();
            if (stringTableOffset != 0 && stringCharsOffset != 0) {
                reader.position = stringTableOffset;
                let stringCount = stringTableSize >> 3;
                let stringSizes = [];
                let previousStringSize = 0;
                for (let i = 0; i < stringCount; i++) {
                    let stringSize = reader.uint64();
                    stringSizes.push(stringSize - previousStringSize);
                    previousStringSize = stringSize;
                }
                reader.position = stringCharsOffset;
                this.strings = [];
                for (let i = 0; i < stringCount; i++) {
                    let cch = stringSizes[i] >> 1;
                    let sb = '';
                    for (let ich = 0; ich < cch; ich++) {
                        sb += String.fromCharCode(reader.uint16());
                    }
                    this.strings.push(sb);
                }
            }
            if (assemblyNameOffset != 0) {
                reader.position = assemblyNameOffset;
                this.assemblyName = textDecoder.decode(reader.bytes(assemblyNameSize));
            }
            reader.position = tailOffset;
            reader.assert('LEDOM\0LM');
    
            this._reader = reader;
            this._reader.position = modelBlockOffset;    
        }
    }

    get reader() {
        return this._reader;
    }

    string(empty) {
        let id = this.reader.int32();
        if (empty === null && id < 0) {
            return null;
        }
        return this.strings[id];
    }

    open(name) {
        let dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        return new mlnet.ModelHeader(this._entries, this._catalog, dir + name);
    }

    openBinary(name) {
        var dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        name = dir + name;
        let entry = this._entries.find((entry) => entry.name == name || entry.name == name.replace(/\//g, '\\'));
        if (entry) {
            let reader = new mlnet.Reader(entry.data);
            return new mlnet.BinaryLoader(reader);
        }
        return null;
    }

    openText(name) {
        var dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        name = dir + name;
        let entry = this._entries.find((entry) => entry.name.split('\\').join('/') == name);
        if (entry) {
            return new TextDecoder().decode(entry.data);
        }
        return null;
    }

    create() {
        let value = this._catalog.create(this._loaderSignature, this);
        value.__type__ = this._loaderSignature;
        value.__name__ = this._directory;
        return value;
    }
};

mlnet.Reader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    set position(value) {
        this._position = value;
    }

    get position() {
        return this._position;
    }

    seek(offset) {
        this._position += offset;
    }

    match(text) {
        let position = this._position;
        for (let i = 0; i < text.length; i++) {
            if (this.byte() != text.charCodeAt(i)) {
                this._position = position;
                return false;
            }
        }
        return true;
    }

    assert(text) {
        if (!this.match(text)) {
            throw new mlnet.Error("Invalid '" + text + "' signature.");
        }
    }

    boolean() {
        return this.byte() != 0 ? true : false;
    }

    booleans(count) {
        let values = [];
        for (let i = 0; i < count; i++) {
            values.push(this.boolean());
        }
        return values;
    }

    byte() {
        let value = this._dataView.getUint8(this._position);
        this._position++;
        return value;
    }

    bytes(count) {
        let data = this._buffer.subarray(this._position, this._position + count);
        this._position += count;
        return data;
    }

    int16() {
        let value = this._dataView.getInt16(this._position, true);
        this._position += 2;
        return value;
    }

    uint16() {
        let value = this._dataView.getUint16(this._position, true);
        this._position += 2;
        return value;
    }

    int32() {
        let value = this._dataView.getInt32(this._position, true);
        this._position += 4;
        return value;
    }

    int32s() {
        let values = [];
        let count = this.int32();
        for (let i = 0; i < count; i++) {
            values.push(this.int32());
        }
        return values;
    }

    uint32() {
        let value = this._dataView.getUint32(this._position, true);
        this._position += 4;
        return value;
    }

    int64() {
        let low = this.uint32();
        let hi = this.uint32();
        if (low == 0xffffffff && hi == 0x7fffffff) {
            return Number.MAX_SAFE_INTEGER;
        }
        if (hi == -1) {
            return -low;
        }
        if (hi != 0) {
            throw new mlnet.Error('Value not in 48-bit range.');
        }
        return (hi << 32) | low;
    }

    uint64() {
        let low = this.uint32();
        let hi = this.uint32();
        if (hi == 0) {
            return low;
        }
        if (hi > 1048576) {
            throw new mlnet.Error('Value not in 48-bit range.');
        }
        return (hi * 4294967296) + low;
    }

    float32() {
        let value = this._dataView.getFloat32(this._position, true);
        this._position += 4;
        return value;
    }

    float32s() {
        let values = [];
        let count = this.int32();
        for (let i = 0; i < count; i++) {
            values.push(this.float32());
        }
        return values;
    }

    float64() {
        let value = this._dataView.getFloat64(this._position, true);
        this._position += 8;
        return value;
    }

    string() {
        let size = this.leb128();
        let buffer = this.bytes(size);
        return new TextDecoder('utf-8').decode(buffer);
    }

    leb128() {
        let result = 0;
        let shift = 0;
        let value;
        do {
            value = this.byte();
            result |= (value & 0x7F) << shift;
            shift += 7;
        } while ((value & 0x80) != 0);
        return result;
    }
};

mlnet.BinaryLoader = class { // 'BINLOADR'

    constructor(reader) {
        // https://github.com/dotnet/machinelearning/blob/master/docs/code/IdvFileFormat.md
        reader.assert('CML\0DVB\0');
        let version = new mlnet.Version(reader);
        let compatibleVersion =  new mlnet.Version(reader);
        if (compatibleVersion.value > version.value) {
            throw new mlnet.Error("Compatibility version '" + compatibleVersion + "' cannot be greater than file version '" + version + "'.");
        }
        let tableOfContentsOffset = reader.uint64();
        let tailOffset = reader.int64();
        /* let rowCount = */ reader.int64();
        let columnCount = reader.int32();
        reader.position = tailOffset;
        reader.assert('\0BVD\0LMC');
        reader.position = tableOfContentsOffset;
        this.schema = {};
        this.schema.inputs = [];
        for (let c = 0; c < columnCount; c  ++) {
            let input = {};
            input.name = reader.string();
            input.type = new mlnet.Codec(reader);
            input.compression = reader.byte(); // None = 0, Deflate = 1
            input.rowsPerBlock = reader.leb128();
            input.lookupOffset = reader.int64();
            input.metadataTocOffset = reader.int64();
            this.schema.inputs.push(input);
        }
    }
};

mlnet.Version = class {

    constructor(reader) {
        this.major = reader.int16();
        this.minor = reader.int16();
        this.build = reader.int16();
        this.revision = reader.int16();
    }

    get value() {
        return (this.major << 24) | (this.minor << 16) | (this.build << 8) | this.revision;
    }

    toString() {
        return [ this.major, this.minor, this.build, this.revision ].join('.');
    }
}

mlnet.TransformerChain = class {

    constructor(context) {
        let reader = context.reader;
        let length = reader.int32();
        this.scopes = [];
        this.chain = [];
        for (let i = 0; i < length; i++) {
            this.scopes.push(reader.int32()); // 0x01 = Training, 0x02 = Testing, 0x04 = Scoring 
            let dirName = 'Transform_' + ('00' + i).slice(-3);
            let transformer = context.open(dirName).create();
            this.chain.push(transformer);
        }
    }
};

mlnet.ColumnCopyingTransformer = class {

    constructor(context) {
        let reader = context.reader;
        let length = reader.uint32();
        this.inputs = [];
        this.outputs = [];
        for (let i = 0; i < length; i++) {
            this.outputs.push({ name: context.string() });
            this.inputs.push({ name: context.string() });
        }
    }
}

mlnet.ColumnConcatenatingTransformer = class {

    constructor(context) {
        let reader = context.reader;
        if (context.modelVersionReadable >= 0x00010003) {
            let count = reader.int32();
            for (let i = 0; i < count; i++) {
                this.outputs = [];
                this.outputs.push({ name: context.string() });
                let n = reader.int32();
                this.inputs = [];
                for (let j = 0; j < n; j++) {
                    let input = { 
                        name: context.string()
                    };
                    let alias = context.string(null);
                    if (alias) { 
                        input.alias = alias;
                    }
                    this.inputs.push(input);
                }
            }
        }
        else {
            this.precision = reader.int32();
            let n = reader.int32();
            let names = [];
            let inputs = [];
            for (let i = 0; i < n; i++) {
                names.push(context.string());
                let numSources = reader.int32();
                let input = [];
                for (let j = 0; j < numSources; j++) {
                    input.push(context.string());
                }
                inputs.push(input);
            }
            let aliases = [];
            if (context.modelVersionReadable >= 0x00010002) {
                for (let i = 0; i < n; i++) {
                    /* let length = */ inputs[i].length;
                    let alias = {};
                    aliases.push(alias);
                    if (context.modelVersionReadable >= 0x00010002) {
                        for (;;) {
                            let j = reader.int32();
                            if (j == -1) {
                                break;
                            }
                            alias[j] = context.string();
                        }
                    }
                }
            }

            if (n > 1) {
                throw new mlnet.Error('');
            }

            this.outputs = [];
            for (let i = 0; i < n; i++) {
                this.outputs.push({ 
                    name: names[i]
                });
                this.inputs = inputs[i];
            }
        }
    }
};

mlnet.PredictionTransformerBase = class {

    constructor(context) {
        let modelContext = context.open('Model');
        this.model = modelContext.create();
        this.trainSchema = context.openBinary('TrainSchema').schema;
    }
};

mlnet.FieldAwareFactorizationMachinePredictionTransformer = class extends mlnet.PredictionTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        this.inputs = [];
        for (let i = 0; i < this.model.fieldCount; i++) {
            this.inputs.push({ name: context.string() });
        }
        this.threshold = reader.float32();
        this.thresholdColumn = context.string();
    }
}

mlnet.SingleFeaturePredictionTransformerBase = class extends mlnet.PredictionTransformerBase {

    constructor(context) {
        super(context);
        let featureColumn = context.string(null);
        this.inputs = [];
        this.inputs.push({ name: featureColumn });
        this.outputs = [];
        this.outputs.push({ name: featureColumn });
    }
};

mlnet.ClusteringPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.AnomalyPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.RegressionPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.BinaryPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.MulticlassPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        this.trainLabelColumn = context.string(null);
    }
};

mlnet.PredictorBase = class {

    constructor(context) {
        let reader = context.reader;
        if (reader.int32() != 4) {
            throw new mlnet.Error('Invalid float type size.');
        }
    }
};

mlnet.LinearMulticlassModelParametersBase = class {

    constructor(/* context */) {
    }
}

mlnet.LinearMulticlassModelParameters = class extends mlnet.LinearMulticlassModelParametersBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.RegressionModelParameters = class {

    constructor(/* context */) {
        // debugger;
    }
}

mlnet.PoissonRegressionModelParameters = class extends mlnet.RegressionModelParameters {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.LinearRegressionModelParameters = class extends mlnet.RegressionModelParameters {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.MulticlassLogisticRegressionPredictor = class extends mlnet.PredictorBase {

    constructor(header) {
        super(header);
        let reader = header.reader;
        /* let numFeatures = */ reader.int32();
        let numClasses = reader.int32();
        this.biases = [];
        for (let i = 0; i < numClasses; i++) {
            this.biases.push(reader.float32());
        }
        /* let numStarts = */ reader.int32();
        // ...
    }
};

mlnet.OneToOneTransformerBase = class {

    constructor(context) {
        let reader = context.reader;
        let n = reader.int32();
        this.inputs = [];
        this.outputs = [];
        for (let i = 0; i < n; i++) {
            let output = context.string();
            let input = context.string();
            this.outputs.push({ name: output });
            this.inputs.push({ name: input });
        }
    }
};

mlnet.TokenizingByCharactersTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.NgramExtractingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.WordTokenizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.TextNormalizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.PrincipalComponentAnalysisTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        if (context.modelVersionReadable === 0x00010001) {
            if (reader.int32() !== 4) {
                throw new mlnet.Error('This file was saved by an incompatible version.');
            }
        }
    }
}

mlnet.LpNormNormalizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.KeyToVectorMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.TypeConvertingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.ImageLoadingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        this.imageFolder = context.string(null);
    }
}

mlnet.ImageResizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        this.options = [];
        for (let i = 0; i < this.inputs.length; i++) {
            var option = {};
            option.width = reader.int32();
            option.height = reader.int32();
            option.scale = reader.byte();
            option.anchor = reader.byte();
            this.options.push(option);
        }
    }
}

mlnet.ImagePixelExtractingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
    }
}


mlnet.NormalizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.KeyToValueMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.ValueToKeyMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        if (context.modelVersionWritten >= 0x00010003) {
            this.textMetadata = reader.booleans(this.outputs.length + this.inputs.length);
        }
        else {
            this.textMetadata = [];
            for (let i = 0; i < this.columnPairs.length; i++) {
                this.textMetadata.push(false);
            }
        }
        this._vocabulary(context.open('Vocabulary'));
    }

    _vocabulary(context) {
        let reader = context.reader;
        let cmap = reader.int32();
        this.termMap = [];
        if (context.modelVersionWritten >= 0x00010002)
        {
            for (let i = 0; i < cmap; ++i) {
                this.termMap.push(new mlnet.TermMap(context));
                // debugger;
                // termMap[i] = TermMap.Load(c, host, CodecFactory);
            }
        }
        else
        {
            throw new mlnet.Error('');
            // for (let i = 0; i < cmap; ++i) {
            //    debugger;
            //    // termMap[i] = TermMap.TextImpl.Create(c, host)
            // }
        }


    }
};

mlnet.TermMap = class {

    constructor(context) {
        let reader = context.reader;
        let mtype = reader.byte();
        switch (mtype) {
            case 0: // Text
                this.values = [];
                var cstr = reader.int32();
                for (let i = 0; i < cstr; i++) {
                    this.values.push(context.string());
                }
                break;
            case 1: // Codec
                var codec = new mlnet.Codec(reader);
                var count = reader.int32();
                this.values = codec.read(reader, count);
                break;
            default:
                throw new mlnet.Error("Unknown term map type '" + mtype.toString() + "'.");
        }
    }
}


mlnet.ValueMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.KeyToVectorTransform = class {

    constructor(/* context */) {
    }
};

mlnet.GenericScoreTransform = class {

    constructor(/* context */) {
    }
};

mlnet.CompositeDataLoader = class { // PIPELODR

    constructor(context) {
        /* let loader = */ context.open('Loader').create();
        let reader = context.reader;
        // LoadTransforms
        this.floatSize = reader.int32();
        let cxf = reader.int32();
        let tagData = [];
        for (let i = 0; i < cxf; i++) {
            let tag = '';
            let args = null;
            if (context.modelVersionReadable >= 0x00010002) {
                tag = context.string();
                args = context.string(null);
            }
            tagData.push([ tag, args ]);
        }
        this.transforms = [];
        for (let j = 0; j < cxf; j++) {
            let transform = context.open('Transform_' + ('00' + j).slice(-3)).create();
            this.transforms.push(transform);
        }
    }
};

mlnet.TransformBase = class {

    constructor(/* context */) {

    }
}

mlnet.RowToRowTransformBase = class extends mlnet.TransformBase {

    constructor(context) {
        super(context);
    }
}

mlnet.RowToRowMapperTransform = class extends mlnet.RowToRowTransformBase {

    constructor(context) {
        super(context);
    }
}

mlnet.RowToRowTransformerBase = class {

    constructor(/* context */) {
    }
}

mlnet.OnnxTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        let numInputs = context.modelVersionWritten > 0x00010001 ? reader.int32() : 1;
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        let numOutputs = context.modelVersionWritten > 0x00010001 ? reader.int32() : 1;
        this.outputs = [];
        for (let i = 0; i < numOutputs; i++) {
            this.outputs.push({ name: context.string() });
        }
        if (context.modelVersionWritten > 0x0001000C) {
            let customShapeInfosLength = reader.int32();
            this.loadedCustomShapeInfos = [];
            for (let i = 0; i < customShapeInfosLength; i++) {
                this.loadedCustomShapeInfos.push({
                    name: context.string(),
                    shape: reader.int32s()
                });
            }
        }
    }
}

mlnet.TensorFlowTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        this.frozen = context.modelVersionReadable >= 0x00010002 ? reader.boolean() : true;
        this.addBatchDimensionInput = context.modelVersionReadable >= 0x00010003 ? reader.boolean() : true;
        let numInputs = reader.int32();
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        let numOutputs = context.modelVersionReadable >= 0x00010002 ? reader.int32() : 1;
        this.outputs = [];
        for (let i = 0; i < numOutputs; i++) {
            this.outputs.push({ name: context.string() });
        }
    }
}

mlnet.OneVersusAllModelParameters = class {

    constructor(/* context */) {
        // debugger;
    }
}

mlnet.TextFeaturizingEstimator = class {

    constructor(context) {

        if (context.modelVersionReadable === 0x00010001) {
            let reader = context.reader;
            let n = reader.int32();
            this.chain = [];
            // let loader = context.open('Loader').create();
            // ctx.LoadModel<ILegacyDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));
            for (let i = 0; i < n; i++) {
                let dirName = 'Step_' + ('00' + i).slice(-3);
                let transformer = context.open(dirName).create();
                this.chain.push(transformer);
                // debugger;
            }

            // throw new mlnet.Error('Unsupported TextFeaturizingEstimator format.');
        }
        else {
            let chain = context.open('Chain').create();
            this.chain = chain.chain;
        }
    }
}

mlnet.TextLoader = class {

    constructor(context) {
        let reader = context.reader;
        this.floatSize = reader.int32();
        this.maxRows = reader.int64();
        this.flags = reader.uint32();
        this.inputSize = reader.int32();
        let separatorCount = reader.int32();
        this.separators = [];
        for (let i = 0; i < separatorCount; i++) {
            this.separators.push(String.fromCharCode(reader.uint16()));            
        }
        // this.bindinds = new mlnet.TextLoader.Bindinds(context);
    }
};

mlnet.TextLoader.Bindinds = class {

    constructor(context) {
        let reader = context.reader;
        let cinfo = reader.int32();
        for (let i = 0; i < cinfo; i++) {
            // debugger;
        }
    }
};

mlnet.CalibratedPredictorBase = class {

    constructor(predictor, calibrator) {
        this.SubPredictor = predictor;
        this.Calibrator = calibrator;
    }
};

mlnet.ValueMapperCalibratedPredictorBase = class extends mlnet.CalibratedPredictorBase {

    constructor(predictor, calibrator) {
        super(predictor, calibrator);
    }
};

mlnet.ValueMapperCalibratedModelParametersBase = class {

    constructor(/* context */) {
        // debugger;
    }
};

mlnet.CalibratedPredictor = class extends mlnet.ValueMapperCalibratedPredictorBase { // 

    constructor(context) {
        let predictor = context.open('Predictor').create();
        let calibrator = context.open('Calibrator').create();
        super(predictor, calibrator);
    }
};

mlnet.ParameterMixingCalibratedModelParameters = class extends mlnet.ValueMapperCalibratedModelParametersBase {

};

mlnet.ModelParametersBase = class {

    constructor(context) {
        let reader = context.reader;
        let cbFloat = reader.int32();
        if (cbFloat !== 4) {
            throw new mlnet.Error('This file was saved by an incompatible version.');
        }
    }
};

mlnet.FieldAwareFactorizationMachineModelParameters = class {

    constructor(context) {
        let reader = context.reader;
        this.norm = reader.boolean();
        this.fieldCount = reader.int32();
        this.featureCount = reader.int32();
        this.latentDim = reader.int32();
        this.linearWeights = reader.float32s();
        this.latentWeights = reader.float32s();
    }
};

mlnet.KMeansModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.PcaModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.TreeEnsembleModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.TreeEnsembleModelParametersBasedOnRegressionTree = class extends mlnet.TreeEnsembleModelParameters {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.FastTreeTweedieModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.FastTreeRegressionModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.LightGbmRegressionModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.FeatureWeightsCalibratedModelParameters = class extends mlnet.ValueMapperCalibratedModelParametersBase {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.FastTreePredictionWrapper = class {

    constructor(/* context */) {
    }
};

mlnet.FastForestClassificationPredictor = class extends mlnet.FastTreePredictionWrapper {
    constructor(context) {
        super(context);
    }
};

mlnet.PlattCalibrator = class {

    constructor(context) {
        let reader = context.reader;
        this.ParamA = reader.float64();
        this.ParamB = reader.float64();
    }
};

mlnet.RowToRowMapper = class { 

    constructor(/* context */) {
        // debugger;
    }
};

mlnet.Codec = class {

    constructor(reader) {
        this.name = reader.string();
        let size = reader.leb128();
        let data = reader.bytes(size);
        reader = new mlnet.Reader(data);

        switch (this.name) {
            case 'Boolean':
                break;
            case 'Single':
                break;
            case 'Double':
                break;
            case 'UInt32':
                break;
            case 'TextSpan':
                break;
            case 'VBuffer':
                this.itemType = new mlnet.Codec(reader);
                this.dims = reader.int32s();
                break;
            case 'Key2':
                this.itemType = new mlnet.Codec(reader);
                this.count = reader.uint64();
                break;
            default:
                throw new mlnet.Error("Unknown codec '" + this.name + "'.");
        }
    }

    read(reader, count) {
        var values = [];
        switch (this.name) {
            case 'Single':
                for (let i = 0; i < count; i++) {
                    values.push(reader.float32());
                }
                break;
            default:
                throw new mlnet.Error("Unknown codec read operation '" + this.name + "'.");
        }
        return values;
    }
}

mlnet.SequentialTransformerBase = class {

    constructor(context) {
        let reader = context.reader;
        this.windowSize = reader.int32();
        this.initialWindowSize = reader.int32();
        this.inputs = [];
        this.inputs.push({ name: context.string() });
        this.outputs = [];
        this.outputs.push({ name: context.string() });
        this.confidenceLowerBoundColumn = reader.string();
        this.confidenceUpperBoundColumn = reader.string();
        this.type = new mlnet.Codec(reader);
    }
}

mlnet.AnomalyDetectionStateBase = class {

    constructor(context) {
        let reader = context.reader;
        this.logMartingaleUpdateBuffer = mlnet.IidAnomalyDetectionBaseWrapper._deserializeFixedSizeQueueDouble(reader);
        this.rawScoreBuffer = mlnet.IidAnomalyDetectionBaseWrapper._deserializeFixedSizeQueueDouble(reader);
        this.logMartingaleValue = reader.float64();
        this.sumSquaredDist = reader.float64();
        this.martingaleAlertCounter = reader.int32();
    }
}

mlnet.SequentialAnomalyDetectionTransformBase = class extends mlnet.SequentialTransformerBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        this.martingale = reader.byte();
        this.thresholdScore = reader.byte();
        this.side = reader.byte();
        this.powerMartingaleEpsilon = reader.float64();
        this.alertThreshold = reader.float64();

        this.state = new mlnet.AnomalyDetectionStateBase(context);
    }
}

mlnet.IidAnomalyDetectionBase = class extends mlnet.SequentialAnomalyDetectionTransformBase {

    constructor(context) {
        super(context);
        let reader = context.reader;
        this.windowedBuffer = mlnet.IidAnomalyDetectionBaseWrapper._deserializeFixedSizeQueueSingle(reader);
        this.initialWindowedBuffer = mlnet.IidAnomalyDetectionBaseWrapper._deserializeFixedSizeQueueSingle(reader);
    }
}

mlnet.IidAnomalyDetectionBaseWrapper = class { 

    constructor(context) {
        this.internalTransform = new mlnet.IidAnomalyDetectionBase(context);
        this.inputs = this.internalTransform.inputs;
        this.outputs = this.internalTransform.outputs;
    }

    static _deserializeFixedSizeQueueSingle(reader) {
        /* let capacity = */ reader.int32();
        let count = reader.int32();
        let queue = [];
        for (let i = 0; i < count; i++) {
            queue.push(reader.float32());
        }
        return queue;
    }

    static _deserializeFixedSizeQueueDouble(reader) {
        /* let capacity = */ reader.int32();
        let count = reader.int32();
        let queue = [];
        for (let i = 0; i < count; i++) {
            queue.push(reader.float64());
        }
        return queue;
    }
};

mlnet.IidChangePointDetector = class extends mlnet.IidAnomalyDetectionBaseWrapper {

    constructor(context) {
        super(context);
    }
}

mlnet.IidSpikeDetector = class extends mlnet.IidAnomalyDetectionBaseWrapper {

    constructor(context) {
        super(context);
    }
}

mlnet.ColumnSelectingTransformer = class {

    constructor(context) {
        let reader = context.reader;
        let keepColumns = reader.boolean();
        this.keepHidden = reader.boolean();
        this.ignoreMissing = reader.boolean();
        var length = reader.int32();
        this.inputs = [];
        for (let i = 0; i < length; i++) {
            this.inputs.push({ name: context.string() });
        }
        if (keepColumns) {
            this.columnsToKeep = this.inputs;
        }
        else {
            this.columnsToDrop = this.inputs;
        }
    }
}

mlnet.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'ML.NET Error';
    }
};

if (module && module.exports) {
    module.exports.ModelFactory = mlnet.ModelFactory;
    module.exports.ModelReader = mlnet.ModelReader;
}