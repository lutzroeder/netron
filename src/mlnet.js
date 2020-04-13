/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var mlnet = mlnet || {};
var zip = zip || require('./zip');

mlnet.ModelFactory = class {

    match(context) {
        const identifier = context.identifier; 
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'zip') {
            const entries = context.entries('zip');
            if (entries.length > 0) {
                const root = new Set([ 'TransformerChain', 'Predictor']);
                if (entries.some((e) => root.has(e.name.split('\\').shift().split('/').shift()))) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        const identifier = context.identifier;
        return mlnet.Metadata.open(host).then((metadata) => {
            try {
                const reader = new mlnet.ModelReader(context.entries('zip'));
                return new mlnet.Model(metadata, reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new mlnet.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
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
            for (const input of reader.schema.inputs) {
                this._inputs.push(new mlnet.Parameter(input.name, [
                    new mlnet.Argument(input.name, new mlnet.TensorType(input.type))
                ]));
            }
        }

        let scope = new Map();
        if (reader.dataLoaderModel) {
            this._loadTransformer(metadata, scope, '', reader.dataLoaderModel);
        }
        if (reader.predictor) {
            this._loadTransformer(metadata, scope, '', reader.predictor);
        }
        if (reader.transformerChain) {
            this._loadTransformer(metadata, scope, '', reader.transformerChain);
        }
    }

    _loadTransformer(metadata, scope, group, transformer) {
        switch (transformer.__type__) {
            case 'TransformerChain':
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
        const group = name.split('/').splice(1).join('/');
        for (const childTransformer of chain) {
            this._loadTransformer(metadata, scope, group, childTransformer);
        }
    }

    _createNode(metadata, scope, group, transformer) {

        if (transformer.inputs && transformer.outputs) {
            for (const input of transformer.inputs) {
                input.name = scope[input.name] ? scope[input.name].argument : input.name;
            }
            for (const output of transformer.outputs) {
                if (scope[output.name]) {
                    scope[output.name].counter++;
                    const next = output.name + '\n' + scope[output.name].counter.toString(); // custom argument id
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

        this._nodes.push(new mlnet.Node(metadata, group, transformer));
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

    constructor(name, type) {
        if (typeof name !== 'string') {
            throw new mlnet.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }
}

mlnet.Node = class {

    constructor(metadata, group, transformer) {
        this._metadata = metadata;
        this._group = group;
        this._name = transformer.__name__;
        this._operator = transformer.__type__;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        if (transformer.inputs) {
            let i = 0;
            for (const input of transformer.inputs) {
                this._inputs.push(new mlnet.Parameter(i.toString(), [
                    new mlnet.Argument(input.name)
                ]));
                i++;
            }
        }

        if (transformer.outputs) {
            let i = 0;
            for (const output of transformer.outputs) {
                this._outputs.push(new mlnet.Parameter(i.toString(), [
                    new mlnet.Argument(output.name)
                ]));
                i++;
            }
        }

        for (const key of Object.keys(transformer).filter((key) => !key.startsWith('_') && key !== 'inputs' && key !== 'outputs')) {
            this._attributes.push(new mlnet.Attribute(metadata, this._operator, key, transformer[key]));
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

    get metadata() {
        return this._metadata.type(this._operator); 
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

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;

        const schema = metadata.attribute(operator, this._name);
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
            }
            if (this._type) {
                let type = mlnet;
                let id = this._type.split('.');
                while (type && id.length > 0) {
                    type = type[id.shift()];
                }
                if (type) {
                    mlnet.Attribute._reverseMap = mlnet.Attribute._reverseMap || {};
                    let reverse = mlnet.Attribute._reverseMap[this._type];
                    if (!reverse) {
                        reverse = {};
                        for (const key of Object.keys(type)) {
                            reverse[type[key.toString()]] = key;
                        }
                        mlnet.Attribute._reverseMap[this._type] = reverse;
                    }
                    if (Object.prototype.hasOwnProperty.call(reverse, this._value)) {
                        this._value = reverse[this._value];
                    }
                }
            }
        }
    }

    get type() {
        return this._type;
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
            [ 'Byte', 'uint8' ],
            [ 'Boolean', 'boolean' ],
            [ 'Single', 'float32' ],
            [ 'Double', 'float64' ],
            [ 'UInt32', 'uint32' ],
            [ 'TextSpan', 'string' ]
        ]);

        this._dataType = '?';
        this._shape = new mlnet.TensorShape(null);

        if (mlnet.TensorType._map.has(codec.name)) {
            this._dataType = mlnet.TensorType._map.get(codec.name);
        }
        else if (codec.name == 'VBuffer') {
            if (mlnet.TensorType._map.has(codec.itemType.name)) {
                this._dataType = mlnet.TensorType._map.get(codec.itemType.name);
            }
            else {
                throw new mlnet.Error("Unknown data type '" + codec.itemType.name + "'.");
            }
            this._shape = new mlnet.TensorShape(codec.dims);
        }
        else if (codec.name == 'Key2') {
            this._dataType = 'key2';
        }
        else { 
            throw new mlnet.Error("Unknown data type '" + codec.name + "'.");
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
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
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
        catalog.register('AffineNormExec', mlnet.AffineNormSerializationUtils);
        catalog.register('AnomalyPredXfer', mlnet.AnomalyPredictionTransformer);
        catalog.register('BinaryPredXfer', mlnet.BinaryPredictionTransformer);
        catalog.register('BinaryLoader', mlnet.BinaryLoader);
        catalog.register('CaliPredExec', mlnet.CalibratedPredictor);
        catalog.register('CdfNormalizeFunction', mlnet.CdfColumnFunction);
        catalog.register('CharToken', mlnet.TokenizingByCharactersTransformer);
        catalog.register('ChooseColumnsTransform', mlnet.ColumnSelectingTransformer);       
        catalog.register('ClusteringPredXfer', mlnet.ClusteringPredictionTransformer);
        catalog.register('ConcatTransform', mlnet.ColumnConcatenatingTransformer);
        catalog.register('CopyTransform', mlnet.ColumnCopyingTransformer);
        catalog.register('ConvertTransform', mlnet.TypeConvertingTransformer);
        catalog.register('CSharpTransform', mlnet.CSharpTransform);
        catalog.register('DropColumnsTransform', mlnet.DropColumnsTransform);
        catalog.register('FAFMPredXfer', mlnet.FieldAwareFactorizationMachinePredictionTransformer);
        catalog.register('FastForestBinaryExec', mlnet.FastForestClassificationPredictor);
        catalog.register('FastTreeBinaryExec', mlnet.FastTreeBinaryModelParameters);
        catalog.register('FastTreeTweedieExec', mlnet.FastTreeTweedieModelParameters);
        catalog.register('FastTreeRankerExec', mlnet.FastTreeRankingModelParameters);
        catalog.register('FastTreeRegressionExec', mlnet.FastTreeRegressionModelParameters);
        catalog.register('FeatWCaliPredExec', mlnet.FeatureWeightsCalibratedModelParameters);
        catalog.register('FieldAwareFactMacPredict', mlnet.FieldAwareFactorizationMachineModelParameters);
        catalog.register('GcnTransform', mlnet.LpNormNormalizingTransformer);
        catalog.register('GenericScoreTransform', mlnet.GenericScoreTransform);
        catalog.register('IidChangePointDetector', mlnet.IidChangePointDetector);
        catalog.register('IidSpikeDetector', mlnet.IidSpikeDetector);
        catalog.register('ImageClassificationTrans', mlnet.ImageClassificationTransformer);
        catalog.register('ImageClassificationPred', mlnet.ImageClassificationModelParameters);
        catalog.register('ImageLoaderTransform', mlnet.ImageLoadingTransformer);
        catalog.register('ImageScalerTransform', mlnet.ImageResizingTransformer);
        catalog.register('ImagePixelExtractor', mlnet.ImagePixelExtractingTransformer);
        catalog.register('KeyToValueTransform', mlnet.KeyToValueMappingTransformer);
        catalog.register('KeyToVectorTransform', mlnet.KeyToVectorMappingTransformer);
        catalog.register('KMeansPredictor', mlnet.KMeansModelParameters);
        catalog.register('LinearRegressionExec', mlnet.LinearRegressionModelParameters);
        catalog.register('LightGBMRegressionExec', mlnet.LightGbmRegressionModelParameters);
        catalog.register('LightGBMBinaryExec', mlnet.LightGbmBinaryModelParameters);
        catalog.register('Linear2CExec', mlnet.LinearBinaryModelParameters);
        catalog.register('LinearModelStats', mlnet.LinearModelParameterStatistics);
        catalog.register('MaFactPredXf', mlnet.MatrixFactorizationPredictionTransformer);
        catalog.register('MFPredictor', mlnet.MatrixFactorizationModelParameters);
        catalog.register('MulticlassLinear', mlnet.LinearMulticlassModelParameters);
        catalog.register('MultiClassLRExec', mlnet.MaximumEntropyModelParameters);
        catalog.register('MultiClassNaiveBayesPred', mlnet.NaiveBayesMulticlassModelParameters);
        catalog.register('MultiClassNetPredictor', mlnet.MultiClassNetPredictor);
        catalog.register('MulticlassPredXfer', mlnet.MulticlassPredictionTransformer);
        catalog.register('NgramTransform', mlnet.NgramExtractingTransformer);
        catalog.register('NgramHashTransform', mlnet.NgramHashingTransformer);
        catalog.register('NltTokenizeTransform', mlnet.NltTokenizeTransform);
        catalog.register('Normalizer', mlnet.NormalizingTransformer);
        catalog.register('NormalizeTransform', mlnet.NormalizeTransform);
        catalog.register('OnnxTransform', mlnet.OnnxTransformer);
        catalog.register('OptColTransform', mlnet.OptionalColumnTransform);
        catalog.register('OVAExec', mlnet.OneVersusAllModelParameters);
        catalog.register('pcaAnomExec', mlnet.PcaModelParameters);
        catalog.register('PcaTransform', mlnet.PrincipalComponentAnalysisTransformer);
        catalog.register('PipeDataLoader', mlnet.CompositeDataLoader);
        catalog.register('PlattCaliExec', mlnet.PlattCalibrator);
        catalog.register('PMixCaliPredExec', mlnet.ParameterMixingCalibratedModelParameters);
        catalog.register('PoissonRegressionExec', mlnet.PoissonRegressionModelParameters);
        catalog.register('ProtonNNMCPred', mlnet.ProtonNNMCPred);
        catalog.register('RegressionPredXfer', mlnet.RegressionPredictionTransformer);
        catalog.register('RowToRowMapper', mlnet.RowToRowMapperTransform);
        catalog.register('SsaForecasting', mlnet.SsaForecastingTransformer);
        catalog.register('SSAModel', mlnet.AdaptiveSingularSpectrumSequenceModelerInternal);
        catalog.register('SelectColumnsTransform', mlnet.ColumnSelectingTransformer);
        catalog.register('StopWordsTransform', mlnet.StopWordsTransform);
        catalog.register('TensorFlowTransform', mlnet.TensorFlowTransformer);
        catalog.register('TermLookupTransform', mlnet.ValueMappingTransformer);
        catalog.register('TermTransform', mlnet.ValueToKeyMappingTransformer);
        catalog.register('TermManager', mlnet.TermManager);
        catalog.register('Text', mlnet.TextFeaturizingEstimator);
        catalog.register('TextLoader', mlnet.TextLoader);
        catalog.register('TextNormalizerTransform', mlnet.TextNormalizingTransformer);
        catalog.register('TokenizeTextTransform', mlnet.WordTokenizingTransformer);
        catalog.register('TransformerChain', mlnet.TransformerChain);
        catalog.register('ValueMappingTransformer', mlnet.ValueMappingTransformer);
        catalog.register('XGBoostMulticlass', mlnet.XGBoostMulticlass);

        const root = new mlnet.ModelHeader(catalog, entries, '', null);

        const version = root.openText('TrainingInfo/Version.txt');
        if (version) {
            this.version = version.split(' ').shift().split('\r').shift();
        }

        const schemaReader = root.openBinary('Schema');
        if (schemaReader) {
            this.schema = new mlnet.BinaryLoader(null, schemaReader).schema;
        }

        const transformerChain = root.open('TransformerChain');
        if (transformerChain) {
            this.transformerChain = transformerChain;
        }

        const dataLoaderModel = root.open('DataLoaderModel');
        if (dataLoaderModel) {
            this.dataLoaderModel = dataLoaderModel;
        }

        const predictor = root.open('Predictor');
        if (predictor) {
            this.predictor = predictor;
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
        const type = this._map.get(signature);
        return Reflect.construct(type, [ context ]);
    }
};

mlnet.ModelHeader = class {

    constructor(catalog, entries, directory, data) {

        this._entries = entries;
        this._catalog = catalog;
        this._directory = directory;

        if (data) {
            const reader = new mlnet.Reader(data);

            const textDecoder = new TextDecoder('ascii');
            reader.assert('ML\0MODEL');
            this.versionWritten = reader.uint32();
            this.versionReadable = reader.uint32();
    
            const modelBlockOffset = reader.uint64();
            /* let modelBlockSize = */ reader.uint64();
            const stringTableOffset = reader.uint64();
            const stringTableSize = reader.uint64();
            const stringCharsOffset = reader.uint64();
            /* v stringCharsSize = */ reader.uint64();
            this.modelSignature = textDecoder.decode(reader.bytes(8));
            this.modelVersionWritten = reader.uint32();
            this.modelVersionReadable = reader.uint32();
            this.loaderSignature = textDecoder.decode(reader.bytes(24).filter((c) => c != 0));
            this.loaderSignatureAlt = textDecoder.decode(reader.bytes(24).filter((c) => c != 0));
            const tailOffset = reader.uint64();
            /* let tailLimit = */ reader.uint64();
            const assemblyNameOffset = reader.uint64();
            const assemblyNameSize = reader.uint32();
            if (stringTableOffset != 0 && stringCharsOffset != 0) {
                reader.position = stringTableOffset;
                const stringCount = stringTableSize >> 3;
                const stringSizes = [];
                let previousStringSize = 0;
                for (let i = 0; i < stringCount; i++) {
                    const stringSize = reader.uint64();
                    stringSizes.push(stringSize - previousStringSize);
                    previousStringSize = stringSize;
                }
                reader.position = stringCharsOffset;
                this.strings = [];
                for (let i = 0; i < stringCount; i++) {
                    const cch = stringSizes[i] >> 1;
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
        const id = this.reader.int32();
        if (empty === null && id < 0) {
            return null;
        }
        return this.strings[id];
    }

    open(name) {
        const dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        name = dir + name;
        const entryName = name + '/Model.key';
        const entry = this._entries.find((entry) => entry.name == entryName || entry.name == entryName.replace(/\//g, '\\'));
        if (entry) {
            const context = new mlnet.ModelHeader(this._catalog, this._entries, name, entry.data);
            let value = this._catalog.create(context.loaderSignature, context);
            value.__type__ = value.__type__ || context.loaderSignature;
            value.__name__ = name;
            return value;
        }
        return null;
    }

    openBinary(name) {
        const dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        name = dir + name;
        const entry = this._entries.find((entry) => entry.name == name || entry.name == name.replace(/\//g, '\\'));
        return entry ? new mlnet.Reader(entry.data) : null;
    }

    openText(name) {
        const dir = this._directory.length > 0 ? this._directory + '/' : this._directory;
        name = dir + name;
        const entry = this._entries.find((entry) => entry.name.split('\\').join('/') == name);
        if (entry) {
            return new TextDecoder().decode(entry.data);
        }
        return null;
    }
    
    check(signature, verWrittenCur, verWeCanReadBack) {
        return signature === this.modelSignature && verWrittenCur >= this.modelVersionReadable && verWeCanReadBack <= this.modelVersionWritten; 
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

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new mlnet.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    match(text) {
        const position = this._position;
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
            throw new mlnet.Error("Invalid '" + text.split('\0').join('') + "' signature.");
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
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position);
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getInt16(position, true);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getUint16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    int32s(count) {
        let values = [];
        for (let i = 0; i < count; i++) {
            values.push(this.int32());
        }
        return values;
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    uint32s(count) {
        let values = [];
        for (let i = 0; i < count; i++) {
            values.push(this.uint32());
        }
        return values;
    }

    int64() {
        const low = this.uint32();
        const hi = this.uint32();
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
        const low = this.uint32();
        const hi = this.uint32();
        if (hi == 0) {
            return low;
        }
        if (hi > 1048576) {
            throw new mlnet.Error('Value not in 48-bit range.');
        }
        return (hi * 4294967296) + low;
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    float32s(count) {
        let values = [];
        for (let i = 0; i < count; i++) {
            values.push(this.float32());
        }
        return values;
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._dataView.getFloat64(position, true);
    }

    float64s(count) {
        let values = [];
        for (let i = 0; i < count; i++) {
            values.push(this.float64());
        }
        return values;
    }

    string() {
        const size = this.leb128();
        const buffer = this.bytes(size);
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

    constructor(context, reader) {
        if (context) {
            if (context.modelVersionWritten >= 0x00010002) {
                this.Threads = context.reader.int32();
                this.GeneratedRowIndexName = context.string(null);
            }
            this.ShuffleBlocks = context.modelVersionWritten >= 0x00010003 ? context.reader.float64() : 4;
            reader = context.openBinary('Schema.idv');
        }
        // https://github.com/dotnet/machinelearning/blob/master/docs/code/IdvFileFormat.md
        reader.assert('CML\0DVB\0');
        reader.bytes(8); // version
        reader.bytes(8); // compatibleVersion
        const tableOfContentsOffset = reader.uint64();
        const tailOffset = reader.int64();
        reader.int64(); // rowCount
        const columnCount = reader.int32();
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

mlnet.TransformerChain = class {

    constructor(context) {
        const reader = context.reader;
        const length = reader.int32();
        this.scopes = [];
        this.chain = [];
        for (let i = 0; i < length; i++) {
            this.scopes.push(reader.int32()); // 0x01 = Training, 0x02 = Testing, 0x04 = Scoring 
            const dirName = 'Transform_' + ('00' + i).slice(-3);
            const transformer = context.open(dirName);
            this.chain.push(transformer);
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

mlnet.RowToRowTransformerBase = class {

    constructor(/* context */) {
    }
}

mlnet.RowToRowMapperTransformBase = class extends mlnet.RowToRowTransformBase {

    constructor(context) {
        super(context);
    }
}

mlnet.OneToOneTransformerBase = class {

    constructor(context) {
        const reader = context.reader;
        const n = reader.int32();
        this.inputs = [];
        this.outputs = [];
        for (let i = 0; i < n; i++) {
            const output = context.string();
            const input = context.string();
            this.outputs.push({ name: output });
            this.inputs.push({ name: input });
        }
    }
};

mlnet.ColumnCopyingTransformer = class {

    constructor(context) {
        const reader = context.reader;
        const length = reader.uint32();
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
        const reader = context.reader;
        if (context.modelVersionReadable >= 0x00010003) {
            const count = reader.int32();
            for (let i = 0; i < count; i++) {
                this.outputs = [];
                this.outputs.push({ name: context.string() });
                const n = reader.int32();
                this.inputs = [];
                for (let j = 0; j < n; j++) {
                    let input = { 
                        name: context.string()
                    };
                    const alias = context.string(null);
                    if (alias) { 
                        input.alias = alias;
                    }
                    this.inputs.push(input);
                }
            }
        }
        else {
            this.precision = reader.int32();
            const n = reader.int32();
            let names = [];
            let inputs = [];
            for (let i = 0; i < n; i++) {
                names.push(context.string());
                const numSources = reader.int32();
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
                            const j = reader.int32();
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
        this.Model = context.open('Model');
        const trainSchemaReader = context.openBinary('TrainSchema');
        if (trainSchemaReader) {
            new mlnet.BinaryLoader(null, trainSchemaReader).schema;
        }
    }
};

mlnet.MatrixFactorizationModelParameters = class {

    constructor(context) {
        const reader = context.reader;
        this.NumberOfRows = reader.int32();
        if (context.modelVersionWritten < 0x00010002) {
            reader.uint64(); // mMin
        }
        this.NumberOfColumns = reader.int32();
        if (context.modelVersionWritten < 0x00010002) {
            reader.uint64(); // nMin
        }
        this.ApproximationRank = reader.int32();

        this._leftFactorMatrix = reader.float32s(this.NumberOfRows * this.ApproximationRank);
        this._rightFactorMatrix = reader.float32s(this.NumberOfColumns * this.ApproximationRank);
    }
}

mlnet.MatrixFactorizationPredictionTransformer = class extends mlnet.PredictionTransformerBase {

    constructor(context) {
        super(context);
        this.MatrixColumnIndexColumnName = context.string();
        this.MatrixRowIndexColumnName = context.string();
        // TODO
    }
}

mlnet.FieldAwareFactorizationMachinePredictionTransformer = class extends mlnet.PredictionTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.inputs = [];
        for (let i = 0; i < this.FieldCount; i++) {
            this.inputs.push({ name: context.string() });
        }
        this.Threshold = reader.float32();
        this.ThresholdColumn = context.string();
        this.inputs.push({ name: this.ThresholdColumn });
    }
}

mlnet.SingleFeaturePredictionTransformerBase = class extends mlnet.PredictionTransformerBase {

    constructor(context) {
        super(context);
        const featureColumn = context.string(null);
        this.inputs = [];
        this.inputs.push({ name: featureColumn });
        this.outputs = [];
        this.outputs.push({ name: featureColumn });
    }
};

mlnet.ClusteringPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
    }
};

mlnet.AnomalyPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.Threshold = reader.float32();
        this.ThresholdColumn = context.string();
    }
}

mlnet.AffineNormSerializationUtils = class {

    constructor(context) {
        const reader = context.reader;
        /* cbFloat = */ reader.int32();
        this.NumFeatures = reader.int32();
        const morphCount = reader.int32();
        if (morphCount == -1) {
            this.ScalesSparse = reader.float32s(reader.int32());
            this.OffsetsSparse = reader.float32s(reader.int32());
        }
        else {
            // debugger;
        }
    }
}

mlnet.RegressionPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
    }
}

mlnet.BinaryPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.Threshold = reader.float32();
        this.ThresholdColumn = context.string();
    }
}

mlnet.MulticlassPredictionTransformer = class extends mlnet.SingleFeaturePredictionTransformerBase {

    constructor(context) {
        super(context);
        this.TrainLabelColumn = context.string(null);
        this.inputs.push({ name: this.TrainLabelColumn });
    }
};

mlnet.PredictorBase = class {

    constructor(context) {
        const reader = context.reader;
        if (reader.int32() != 4) {
            throw new mlnet.Error('Invalid float type size.');
        }
    }
};

mlnet.ModelParametersBase = class {

    constructor(context) {
        const reader = context.reader;
        const cbFloat = reader.int32();
        if (cbFloat !== 4) {
            throw new mlnet.Error('This file was saved by an incompatible version.');
        }
    }
};

mlnet.ImageClassificationModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.classCount = reader.int32();
        this.imagePreprocessorTensorInput = reader.string();
        this.imagePreprocessorTensorOutput = reader.string();
        this.graphInputTensor = reader.string();
        this.graphOutputTensor = reader.string();
        this.modelFile = 'TFModel';
        // const modelBytes = context.openBinary('TFModel');
        // first uint32 is size of TensorFlow model
        // inputType = new VectorDataViewType(uint8);
        // outputType = new VectorDataViewType(float32, classCount);
    }
};

mlnet.NaiveBayesMulticlassModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this._labelHistogram = reader.int32s(reader.int32());
        this._featureCount = reader.int32();
        this._featureHistogram = [];
        for (let i = 0; i < this._labelHistogram.length; i++) {
            if (this._labelHistogram[i] > 0) {
                this._featureHistogram.push(reader.int32s(this._featureCount));
            }
        }
        this._absentFeaturesLogProb = reader.float64s(this._labelHistogram.length);
    }
}

mlnet.LinearModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.Bias = reader.float32();
        /* let len = */ reader.int32();
        this.Indices = reader.int32s(reader.int32());
        this.Weights = reader.float32s(reader.int32());
    }
}

mlnet.LinearBinaryModelParameters = class extends mlnet.LinearModelParameters {

    constructor(context) {
        super(context);
        if (context.modelVersionWritten > 0x00020001) {
            this.Statistics = context.open('ModelStats');
        }
    }
}

mlnet.ModelStatisticsBase = class {

    constructor(context) {
        const reader = context.reader;
        this.ParametersCount = reader.int32();
        this.TrainingExampleCount = reader.int64();
        this.Deviance = reader.float32();
        this.NullDeviance = reader.float32();

    }    
}

mlnet.LinearModelParameterStatistics = class extends mlnet.ModelStatisticsBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (context.modelVersionWritten < 0x00010002) {
            if (!reader.boolean()) {
                return;
            }
        }
        const stdErrorValues = reader.float32s(this.ParametersCount);
        const length = reader.int32();
        if (length == this.ParametersCount) {
            this._coeffStdError = stdErrorValues;
        }
        else {
            this.stdErrorIndices = reader.int32s(this.ParametersCount);
            this._coeffStdError = stdErrorValues;
        }
        this._bias = reader.float32();
        const isWeightsDense = reader.byte();
        const weightsLength = reader.int32();
        const weightsValues = reader.float32s(weightsLength);

        if (isWeightsDense) {
            this._weights = weightsValues;
        }
        else {
            this.weightsIndices = reader.int32s(weightsLength);
        }
    }
}

mlnet.LinearMulticlassModelParametersBase = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        const numberOfFeatures = reader.int32();
        const numberOfClasses = reader.int32();
        this.Biases = reader.float32s(numberOfClasses);
        const numStarts = reader.int32();
        if (numStarts == 0) {
            /* let numIndices = */ reader.int32();
            /* let numWeights = */ reader.int32();
            this.Weights = [];
            for (let i = 0; i < numberOfClasses; i++) {
                const w = reader.float32s(numberOfFeatures);
                this.Weights.push(w);
            }
        }
        else {

            const starts = reader.int32s(reader.int32());
            /* let numIndices = */ reader.int32();
            let indices = [];
            for (let i = 0; i < numberOfClasses; i++) {
                indices.push(reader.int32s(starts[i + 1] - starts[i]));
            }
            /* let numValues = */ reader.int32();
            this.Weights = [];
            for (let i = 0; i < numberOfClasses; i++) {
                const values = reader.float32s(starts[i + 1] - starts[i]);
                this.Weights.push(values); 
            }
        }

        const labelNamesReader = context.openBinary('LabelNames');
        if (labelNamesReader) {
            this.LabelNames = [];
            for (let i = 0; i < numberOfClasses; i++) {
                const id = labelNamesReader.int32();
                this.LabelNames.push(context.strings[id]);
            }
        }
        
        const statistics = context.open('ModelStats');
        if (statistics) {
            this.Statistics = statistics;
        }
    }
}

mlnet.LinearMulticlassModelParameters = class extends mlnet.LinearMulticlassModelParametersBase {

    constructor(context) {
        super(context);
    }
}

mlnet.RegressionModelParameters = class extends mlnet.LinearModelParameters {

    constructor(context) {
        super(context);
    }
}

mlnet.PoissonRegressionModelParameters = class extends mlnet.RegressionModelParameters {

    constructor(context) {
        super(context);
    }
}

mlnet.LinearRegressionModelParameters = class extends mlnet.RegressionModelParameters {

    constructor(context) {
        super(context);
    }
}

mlnet.MaximumEntropyModelParameters = class extends mlnet.LinearMulticlassModelParametersBase {

    constructor(context) {
        super(context);
    }
};

mlnet.TokenizingByCharactersTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.UseMarkerChars = reader.boolean();
        this.IsSeparatorStartEnd = context.modelVersionReadable < 0x00010002 ? true : reader.boolean();
    }
};

mlnet.SequencePool = class {

    constructor(reader) {
        this.idLim = reader.int32();
        this.start = reader.int32s(this.idLim + 1);
        this.bytes = reader.bytes(this.start[this.idLim])
    }
}

mlnet.NgramExtractingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (this.inputs.length == 1) {
            this._option(context, reader, this)
        }
        else {
            // debugger;
        }
    }

    _option(context, reader, option) {
        const readWeighting = context.modelVersionReadable >= 0x00010002;
        option.NgramLength = reader.int32();
        option.SkipLength = reader.int32();
        if (readWeighting) {
            option.Weighting = reader.int32();
        }
        option.NonEmptyLevels = reader.booleans(option.NgramLength);
        option.NgramMap = new mlnet.SequencePool(reader);
        if (readWeighting) {
            option.InvDocFreqs = reader.float64s(reader.int32());
        }
    }
};

// mlnet.NgramExtractingTransformer.WeightingCriteria

mlnet.NgramHashingTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        const loadLegacy = context.modelVersionWritten < 0x00010003
        const reader = context.reader;
        if (loadLegacy) {
            reader.int32(); // cbFloat
        }
        this.inputs = [];
        this.outputs = [];
        const columnsLength = reader.int32();
        if (loadLegacy) {
            /* TODO
            for (let i = 0; i < columnsLength; i++) {
                this.Columns.push(new NgramHashingEstimator.ColumnOptions(context));
            } */
        }
        else {
            for (let i = 0; i < columnsLength; i++) {
                this.outputs.push(context.string());
                let csrc = reader.int32();
                for (let j = 0; j < csrc; j++) {
                    let src = context.string();
                    this.inputs.push(src);
                    // TODO inputs[i][j] = src;
                }
            }
        }
    }
}

mlnet.WordTokenizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (this.inputs.length == 1) {
            this.Separators = [];
            const count = reader.int32();
            for (let i = 0; i < count; i++) {
                this.Separators.push(String.fromCharCode(reader.int16()));
            }
        }
        else {
            // debugger;
        }
    }
};

mlnet.TextNormalizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.CaseMode = reader.byte();
        this.KeepDiacritics = reader.boolean();
        this.KeepPunctuations = reader.boolean();
        this.KeepNumbers = reader.boolean();
    }
}

mlnet.TextNormalizingTransformer.CaseMode = {
    Lower: 0,
    Upper: 1,
    None: 2
}

mlnet.PrincipalComponentAnalysisTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (context.modelVersionReadable === 0x00010001) {
            if (reader.int32() !== 4) {
                throw new mlnet.Error('This file was saved by an incompatible version.');
            }
        }
        this.TransformInfos = [];
        for (let i = 0; i < this.inputs.length; i++) {
            let option = {};
            option.Dimension = reader.int32();
            option.Rank = reader.int32();
            option.Eigenvectors = [];
            for (let j = 0; j < option.Rank; j++) {
                option.Eigenvectors.push(reader.float32s(option.Dimension));
            }
            option.MeanProjected = reader.float32s(reader.int32());
            this.TransformInfos.push(option);
        }
    }
}

mlnet.LpNormNormalizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        
        if (context.modelVersionWritten <= 0x00010002) {
            /* cbFloat */ reader.int32();
        }
        // let normKindSerialized = context.modelVersionWritten >= 0x00010002;
        if (this.inputs.length == 1) {
            this.EnsureZeroMean = reader.boolean();
            this.Norm = reader.byte();
            this.Scale = reader.float32();
        }
        else {
            // debugger;
        }
    }
}

mlnet.KeyToVectorMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (context.modelVersionWritten == 0x00010001) {
            /* cbFloat = */ reader.int32();
        }
        const columnsLength = this.inputs.length; 
        this.Bags = reader.booleans(columnsLength);
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
        this.ImageFolder = context.string(null);
    }
}

mlnet.ImageResizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (this.inputs.length == 1) {
            this._option(reader, this);
        }
        else {
            this.Options = [];
            for (let i = 0; i < this.inputs.length; i++) {
                let option = {};
                this._option(reader, option);
                this.Options.push(option);
            }
        }
    }

    _option(reader, option) {
        option.Width = reader.int32();
        option.Height = reader.int32();
        option.Resizing = reader.byte();
        option.Anchor = reader.byte();
    }
}

mlnet.ImageResizingTransformer.ResizingKind = {
    IsoPad: 0,
    IsoCrop: 1,
    Fill: 2
}

mlnet.ImageResizingTransformer.Anchor = {
    Right: 0,
    Left: 1,
    Top: 2,
    Bottom: 3,
    Center: 4
}

mlnet.ImagePixelExtractingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (this.inputs.length == 1) {
            this._option(context, reader, this);
        }
        else {
            this.Options = [];
            for (let i = 0; i < this.inputs.length; i++) {
                let option = {};
                this._option(context, reader, option);
                this.Options.push(option);
            }
        }
    }

    _option(context, reader, option) {
        option.ColorsToExtract = reader.byte();
        option.OrderOfExtraction = context.modelVersionWritten <= 0x00010002 ? mlnet.ImagePixelExtractingTransformer.ColorsOrder.ARGB : reader.byte();
        let planes = option.ColorsToExtract;
        planes = (planes & 0x05) + ((planes >> 1) & 0x05);
        planes = (planes & 0x03) + ((planes >> 2) & 0x03);
        option.Planes = planes & 0xFF;
        option.OutputAsFloatArray = reader.boolean();
        option.OffsetImage = reader.float32();
        option.ScaleImage = reader.float32();
        option.InterleavePixelColors = reader.boolean();
    }
}

mlnet.ImagePixelExtractingTransformer.ColorBits = {
    Alpha: 0x01,
    Red: 0x02,
    Green: 0x04,
    Blue: 0x08,
    Rgb: 0x0E,
    All: 0x0F
}

mlnet.ImagePixelExtractingTransformer.ColorsOrder = {
    ARGB: 1,
    ARBG: 2,
    ABRG: 3,
    ABGR: 4,
    AGRB: 5,
    AGBR: 6
}

mlnet.NormalizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.Options = [];
        for (let i = 0; i < this.inputs.length; i++) {
            let isVector = false;
            let shape = 0;
            let itemKind = '';
            if (context.modelVersionWritten < 0x00010002) {
                isVector = reader.boolean();
                shape = [ reader.int32() ];
                itemKind = reader.byte();
            }
            else {
                isVector = reader.boolean();
                itemKind = reader.byte();
                shape = reader.int32s(reader.int32());
            }
            let itemType = '';
            switch (itemKind) {
                case 9: itemType = 'float32'; break;
                case 10: itemType = 'float64'; break;
                default: throw new mlnet.Error("Unknown NormalizingTransformer item kind '" + itemKind + "'.");
            }
            const type = itemType + (!isVector ? '' : '[' + shape.map((dim) => dim.toString()).join(',') + ']');
            const name = 'Normalizer_' + ('00' + i).slice(-3);
            const func = context.open(name);
            this.Options.push({ type: type, func: func });
        }
    }
}

mlnet.KeyToValueMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
    }
};

mlnet.ValueToKeyMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        if (context.modelVersionWritten >= 0x00010003) {
            this.textMetadata = reader.booleans(this.outputs.length + this.inputs.length);
        }
        else {
            this.textMetadata = [];
            for (let i = 0; i < this.columnPairs.length; i++) {
                this.textMetadata.push(false);
            }
        }
        const vocabulary = context.open('Vocabulary');
        if (vocabulary) {
            this.termMap = vocabulary.termMap;
        }
    }
};

mlnet.TermMap = class {

    constructor(context) {
        const reader = context.reader;
        const mtype = reader.byte();
        switch (mtype) {
            case 0: { // Text
                this.values = [];
                const cstr = reader.int32();
                for (let i = 0; i < cstr; i++) {
                    this.values.push(context.string());
                }
                break;
            }
            case 1: { // Codec
                const codec = new mlnet.Codec(reader);
                const count = reader.int32();
                this.values = codec.read(reader, count);
                break;
            }
            default:
                throw new mlnet.Error("Unknown term map type '" + mtype.toString() + "'.");
        }
    }
}

mlnet.TermManager = class {

    constructor(context) {
        const reader = context.reader;
        const cmap = reader.int32();
        this.termMap = [];
        if (context.modelVersionWritten >= 0x00010002) {
            for (let i = 0; i < cmap; ++i) {
                this.termMap.push(new mlnet.TermMap(context));
                // debugger;
                // termMap[i] = TermMap.Load(c, host, CodecFactory);
            }
        }
        else {
            throw new mlnet.Error('Unsupported TermManager version.');
            // for (let i = 0; i < cmap; ++i) {
            //    debugger;
            //    // termMap[i] = TermMap.TextImpl.Create(c, host)
            // }
        }
    }
}


mlnet.ValueMappingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        this.keyColumnName = 'Key';
        if (context.check('TXTLOOKT', 0x00010002, 0x00010002)) {
            this.keyColumnName = 'Term';
        }
        // TODO
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

mlnet.CompositeDataLoader = class {

    constructor(context) {
        /* let loader = */ context.open('Loader');
        const reader = context.reader;
        // LoadTransforms
        reader.int32(); // floatSize
        const cxf = reader.int32();
        const tagData = [];
        for (let i = 0; i < cxf; i++) {
            let tag = '';
            let args = null;
            if (context.modelVersionReadable >= 0x00010002) {
                tag = context.string();
                args = context.string(null);
            }
            tagData.push([ tag, args ]);
        }
        this.chain = [];
        for (let j = 0; j < cxf; j++) {
            const name = 'Transform_' + ('00' + j).slice(-3);
            const transform = context.open(name);
            this.chain.push(transform);
        }
    }
};

mlnet.RowToRowMapperTransform = class extends mlnet.RowToRowTransformBase {

    constructor(context) {
        super(context);
        const mapper = context.open('Mapper');
        this.__type__ = mapper.__type__;
        for (const key of Object.keys(mapper)) {
            this[key] = mapper[key];
        }
    }
}

mlnet.ImageClassificationTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.addBatchDimensionInput = reader.boolean();
        const numInputs = reader.int32();
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        this.outputs = [];
        const numOutputs = reader.int32();
        for (let i = 0; i < numOutputs; i++) {
            this.outputs.push({ name: context.string() });
        }
        this.labelColumn = reader.string();
        this.checkpointName = reader.string();
        this.arch = reader.int32(); // Architecture
        this.scoreColumnName = reader.string();
        this.predictedColumnName = reader.string();
        this.learningRate = reader.float32();
        this.classCount = reader.int32();
        this.keyValueAnnotations = [];
        for (let i = 0; i < this.classCount; i++) {
            this.keyValueAnnotations.push(context.string());
        }
        this.predictionTensorName = reader.string();
        this.softMaxTensorName = reader.string();
        this.jpegDataTensorName = reader.string();
        this.resizeTensorName = reader.string();
    }
}

mlnet.OnnxTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.modelFile = 'OnnxModel';
        // const modelBytes = context.openBinary('OnnxModel');
        // first uint32 is size of .onnx model
        const numInputs = context.modelVersionWritten > 0x00010001 ? reader.int32() : 1;
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        const numOutputs = context.modelVersionWritten > 0x00010001 ? reader.int32() : 1;
        this.outputs = [];
        for (let i = 0; i < numOutputs; i++) {
            this.outputs.push({ name: context.string() });
        }
        if (context.modelVersionWritten > 0x0001000C) {
            const customShapeInfosLength = reader.int32();
            this.LoadedCustomShapeInfos = [];
            for (let i = 0; i < customShapeInfosLength; i++) {
                this.LoadedCustomShapeInfos.push({
                    name: context.string(),
                    shape: reader.int32s(reader.int32())
                });
            }
        }
    }
}

mlnet.OptionalColumnTransform = class extends mlnet.RowToRowMapperTransformBase {

    constructor(context) {
        super(context);
    }
}

mlnet.TensorFlowTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.IsFrozen = context.modelVersionReadable >= 0x00010002 ? reader.boolean() : true;
        this.AddBatchDimensionInput = context.modelVersionReadable >= 0x00010003 ? reader.boolean() : true;
        const numInputs = reader.int32();
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        const numOutputs = context.modelVersionReadable >= 0x00010002 ? reader.int32() : 1;
        this.outputs = [];
        for (let i = 0; i < numOutputs; i++) {
            this.outputs.push({ name: context.string() });
        }
    }
}

mlnet.OneVersusAllModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.UseDist = reader.boolean();
        const len = reader.int32();
        this.chain = [];
        for (let i = 0; i < len; i++) {
            const name = 'SubPredictor_' + ('00' + i).slice(-3);
            const predictor = context.open(name);
            this.chain.push(predictor);
        }
    }
}

mlnet.TextFeaturizingEstimator = class {

    constructor(context) {

        if (context.modelVersionReadable === 0x00010001) {
            const reader = context.reader;
            const n = reader.int32();
            this.chain = [];
            /* let loader = */ context.open('Loader');
            for (let i = 0; i < n; i++) {
                const name = 'Step_' + ('00' + i).slice(-3);
                const transformer = context.open(name);
                this.chain.push(transformer);
                // debugger;
            }

            // throw new mlnet.Error('Unsupported TextFeaturizingEstimator format.');
        }
        else {
            let chain = context.open('Chain');
            this.chain = chain.chain;
        }
    }
}

mlnet.TextLoader = class {

    constructor(context) {
        const reader = context.reader;
        reader.int32(); // floatSize
        this.MaxRows = reader.int64();
        this.Flags = reader.uint32();
        this.InputSize = reader.int32();
        const separatorCount = reader.int32();
        this.Separators = [];
        for (let i = 0; i < separatorCount; i++) {
            this.Separators.push(String.fromCharCode(reader.uint16()));
        }
        this.Bindinds = new mlnet.TextLoader.Bindinds(context);
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

mlnet.CalibratedModelParametersBase = class {

    constructor(context) {
        this.Predictor = context.open('Predictor');
        this.Calibrator = context.open('Calibrator');
    }
}

mlnet.ValueMapperCalibratedModelParametersBase = class extends mlnet.CalibratedModelParametersBase {

    constructor(context) {
        super(context);
        // debugger;
    }
};

mlnet.CalibratedPredictor = class extends mlnet.ValueMapperCalibratedPredictorBase { // 

    constructor(context) {
        let predictor = context.open('Predictor');
        let calibrator = context.open('Calibrator');
        super(predictor, calibrator);
    }
};

mlnet.ParameterMixingCalibratedModelParameters = class extends mlnet.ValueMapperCalibratedModelParametersBase {

    constructor(context) {
        super(context);
    }
};

mlnet.FieldAwareFactorizationMachineModelParameters = class {

    constructor(context) {
        let reader = context.reader;
        this.Norm = reader.boolean();
        this.FieldCount = reader.int32();
        this.FeatureCount = reader.int32();
        this.LatentDim = reader.int32();
        this.LinearWeights = reader.float32s(reader.int32());
        this.LatentWeights = reader.float32s(reader.int32());
    }
};

mlnet.KMeansModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.k = reader.int32();
        this.Dimensionality = reader.int32();
        this.Centroids = [];
        for (let i = 0; i < this.k; i++) {
            const count = context.modelVersionWritten >= 0x00010002 ? reader.int32() : this.Dimensionality;
            const indices = count < this.Dimensionality ? reader.int32s(count) : null;
            const values = reader.float32s(count);
            this.Centroids.push({ indices: indices, values: values });
        }
        // input type = float32[dimensionality]
        // output type = float32[k]
    }
};

mlnet.PcaModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.Dimension = reader.int32();
        this.Rank = reader.int32();
        const center = reader.boolean();
        if (center) {
            this.Mean = reader.float32s(this.Dimension);
        }
        else {
            this.Mean = [];
        }
        this.EigenVectors = []
        for (let i = 0; i < this.Rank; ++i) {
            this.EigenVectors.push(reader.float32s(this.Dimension));
        }
        // input type -> float32[Dimension]
    }
};

mlnet.TreeEnsembleModelParameters = class extends mlnet.ModelParametersBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        const usingDefaultValues = context.modelVersionWritten >= this.VerDefaultValueSerialized;
        const categoricalSplits = context.modelVersionWritten >= this.VerCategoricalSplitSerialized;
        this.TrainedEnsemble = new mlnet.InternalTreeEnsemble(context, usingDefaultValues, categoricalSplits);
        this.InnerOptions = context.string(null);
        if (context.modelVersionWritten >= this.verNumFeaturesSerialized) {
            this.NumFeatures = reader.int32();
        }

        // input type -> float32[NumFeatures]
        // output type -> float32
    }
}

mlnet.InternalTreeEnsemble = class {

    constructor(context, usingDefaultValues, categoricalSplits) {
        const reader = context.reader;
        this.Trees = [];
        const numTrees = reader.int32();
        for (let i = 0; i < numTrees; i++) {
            switch (reader.byte()) {
                case mlnet.InternalTreeEnsemble.TreeType.Regression:
                    this.Trees.push(new mlnet.InternalRegressionTree(context, usingDefaultValues, categoricalSplits));
                    break;
                case mlnet.InternalTreeEnsemble.TreeType.FastForest:
                    this.Trees.push(new mlnet.InternalQuantileRegressionTree(context, usingDefaultValues, categoricalSplits));
                    break;
                case mlnet.InternalTreeEnsemble.TreeType.Affine:
                    // Affine regression trees do not actually work, nor is it clear how they ever
                    // could have worked within TLC, so the chance of this happening seems remote.
                    throw new mlnet.Error('Affine regression trees unsupported');
                default:
                    throw new mlnet.Error('Unknown ensemble tree type.');
            }
        }
        this.Bias = reader.float64();
        this.FirstInputInitializationContent = context.string(null);
    }
}

mlnet.InternalRegressionTree = class {

    constructor(context, usingDefaultValue, categoricalSplits) {
        const reader = context.reader;
        this.NumLeaves = reader.int32();
        this.MaxOuptut = reader.float64();
        this.Weight = reader.float64();
        this.LteChild = reader.int32s(reader.int32());
        this.GtChild = reader.int32s(reader.int32());
        this.SplitFeatures = reader.int32s(reader.int32());
        if (categoricalSplits) {
            const categoricalNodeIndices = reader.int32s(reader.int32());
            if (categoricalNodeIndices.length > 0) {
                this.CategoricalSplitFeatures = [];
                this.CategoricalSplitFeatureRanges = [];
                for (const index of categoricalNodeIndices) {
                    this.CategoricalSplitFeatures[index] = reader.int32s(reader.int32());
                    this.CategoricalSplitFeatureRanges[index] = reader.int32s(2);
                }
            }
        }
        this.Thresholds = reader.uint32s(reader.int32());
        this.RawThresholds = reader.float32s(reader.int32());
        this.DefaultValueForMissing = usingDefaultValue ? reader.float32s(reader.int32()) : null;
        this.LeafValues = reader.float64s(reader.int32());

        this.SplitGain = reader.float64s(reader.int32());
        this.GainPValue = reader.float64s(reader.int32());
        this.PreviousLeafValue = reader.float64s(reader.int32());
    }
}

mlnet.InternalTreeEnsemble.TreeType = {
    Regression: 0,
    Affine: 1,
    FastForest: 2
}

mlnet.TreeEnsembleModelParametersBasedOnRegressionTree = class extends mlnet.TreeEnsembleModelParameters {

    constructor(context) {
        super(context);
    }
}

mlnet.FastTreeTweedieModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
    }

    get VerNumFeaturesSerialized() { return 0x00010001; }
    get VerDefaultValueSerialized() { return 0x00010002; }
    get VerCategoricalSplitSerialized() { return 0x00010003; }
}

mlnet.FastTreeRankingModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
    }

    get VerNumFeaturesSerialized() { return 0x00010002; }
    get VerDefaultValueSerialized() { return 0x00010004; }
    get VerCategoricalSplitSerialized() { return 0x00010005; }
}

mlnet.FastTreeBinaryModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
    }

    get VerNumFeaturesSerialized() { return 0x00010002; }
    get VerDefaultValueSerialized() { return 0x00010004; }
    get VerCategoricalSplitSerialized() { return 0x00010005; }
}

mlnet.FastTreeRegressionModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
    }

    get VerNumFeaturesSerialized() { return 0x00010002; }
    get VerDefaultValueSerialized() { return 0x00010004; }
    get VerCategoricalSplitSerialized() { return 0x00010005; }
}

mlnet.LightGbmRegressionModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
    }

    get VerNumFeaturesSerialized() { return 0x00010002; }
    get VerDefaultValueSerialized() { return 0x00010004; }
    get VerCategoricalSplitSerialized() { return 0x00010005; }
}

mlnet.LightGbmBinaryModelParameters = class extends mlnet.TreeEnsembleModelParametersBasedOnRegressionTree {

    constructor(context) {
        super(context);
    }

    get VerNumFeaturesSerialized() { return 0x00010002; }
    get VerDefaultValueSerialized() { return 0x00010004; }
    get VerCategoricalSplitSerialized() { return 0x00010005; }
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
        const reader = context.reader;
        this.ParamA = reader.float64();
        this.ParamB = reader.float64();
    }
};

mlnet.Codec = class {

    constructor(reader) {
        this.name = reader.string();
        const size = reader.leb128();
        const data = reader.bytes(size);
        reader = new mlnet.Reader(data);

        switch (this.name) {
            case 'Boolean': break;
            case 'Single': break;
            case 'Double': break;
            case 'Byte': break;
            case 'Int32': break;
            case 'UInt32': break;
            case 'Int64': break;
            case 'TextSpan': break;
            case 'VBuffer':
                this.itemType = new mlnet.Codec(reader);
                this.dims = reader.int32s(reader.int32());
                break;
            case 'Key':
            case 'Key2':
                this.itemType = new mlnet.Codec(reader);
                this.count = reader.uint64();
                break;
            default:
                throw new mlnet.Error("Unknown codec '" + this.name + "'.");
        }
    }

    read(reader, count) {
        let values = [];
        switch (this.name) {
            case 'Single':
                for (let i = 0; i < count; i++) {
                    values.push(reader.float32());
                }
                break;
            case 'Int32':
                for (let i = 0; i < count; i++) {
                    values.push(reader.int32());
                }
                break;
            case 'Int64':
                for (let i = 0; i < count; i++) {
                    values.push(reader.int64());
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
        const reader = context.reader;
        this.WindowSize = reader.int32();
        this.InitialWindowSize = reader.int32();
        this.inputs = [];
        this.inputs.push({ name: context.string() });
        this.outputs = [];
        this.outputs.push({ name: context.string() });
        this.ConfidenceLowerBoundColumn = reader.string();
        this.ConfidenceUpperBoundColumn = reader.string();
        this.Type = new mlnet.Codec(reader);
    }
}

mlnet.AnomalyDetectionStateBase = class {

    constructor(context) {
        const reader = context.reader;
        this.LogMartingaleUpdateBuffer = mlnet.AnomalyDetectionStateBase._deserializeFixedSizeQueueDouble(reader);
        this.RawScoreBuffer = mlnet.AnomalyDetectionStateBase._deserializeFixedSizeQueueDouble(reader);
        this.LogMartingaleValue = reader.float64();
        this.SumSquaredDist = reader.float64();
        this.MartingaleAlertCounter = reader.int32();
    }

    static _deserializeFixedSizeQueueDouble(reader) {
        /* let capacity = */ reader.int32();
        const count = reader.int32();
        let queue = [];
        for (let i = 0; i < count; i++) {
            queue.push(reader.float64());
        }
        return queue;
    }
}

mlnet.SequentialAnomalyDetectionTransformBase = class extends mlnet.SequentialTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.Martingale = reader.byte();
        this.ThresholdScore = reader.byte();
        this.Side = reader.byte();
        this.PowerMartingaleEpsilon = reader.float64();
        this.AlertThreshold = reader.float64();
        this.State = new mlnet.AnomalyDetectionStateBase(context);
    }
}

mlnet.TimeSeriesUtils = class {

    static deserializeFixedSizeQueueSingle(reader) {
        /* const capacity = */ reader.int32();
        const count = reader.int32();
        let queue = [];
        for (let i = 0; i < count; i++) {
            queue.push(reader.float32());
        }
        return queue;
    }
}

mlnet.IidAnomalyDetectionBase = class extends mlnet.SequentialAnomalyDetectionTransformBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.WindowedBuffer = mlnet.TimeSeriesUtils.deserializeFixedSizeQueueSingle(reader);
        this.InitialWindowedBuffer = mlnet.TimeSeriesUtils.deserializeFixedSizeQueueSingle(reader);
    }
}

mlnet.IidAnomalyDetectionBaseWrapper = class { 

    constructor(context) {
        const internalTransform = new mlnet.IidAnomalyDetectionBase(context);
        for (const key of Object.keys(internalTransform)) {
            this[key] = internalTransform[key];
        }
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

mlnet.SequenceModelerBase = class {

    constructor(/* context */) {
    }
}

mlnet.RankSelectionMethod = {
    Fixed: 0,
    Exact: 1,
    Fact: 2
}

mlnet.AdaptiveSingularSpectrumSequenceModelerInternal = class extends mlnet.SequenceModelerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this._seriesLength = reader.int32();
        this._windowSize = reader.int32();
        this._trainSize = reader.int32();
        this._rank = reader.int32();
        this._discountFactor = reader.float32();
        this._rankSelectionMethod = reader.byte(); // RankSelectionMethod
        const isWeightSet = reader.byte();
        this._alpha = reader.float32s(reader.int32());
        if (context.modelVersionReadable >= 0x00010002) {
            this._state = reader.float32s(reader.int32());
        }
        this.ShouldComputeForecastIntervals = reader.byte();
        this._observationNoiseVariance = reader.float32();
        this._autoregressionNoiseVariance = reader.float32();
        this._observationNoiseMean = reader.float32();
        this._autoregressionNoiseMean = reader.float32();
        if (context.modelVersionReadable >= 0x00010002) {
            this._nextPrediction = reader.float32();
        }
        this._maxRank = reader.int32();
        this._shouldStablize = reader.byte();
        this._shouldMaintainInfo = reader.byte();
        this._maxTrendRatio = reader.float64();
        if (isWeightSet) {
            this._wTrans = reader.float32s(reader.int32());
            this._y = reader.float32s(reader.int32());
        }
        this._buffer = mlnet.TimeSeriesUtils.deserializeFixedSizeQueueSingle(reader);
    }
}

mlnet.SequentialForecastingTransformBase = class extends mlnet.SequentialTransformerBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this._outputLength = reader.int32();
    }
}

mlnet.SsaForecastingBaseWrapper = class extends mlnet.SequentialForecastingTransformBase {

    constructor(context) {
        super(context);
        const reader = context.reader;
        this.IsAdaptive = reader.boolean();
        this.Horizon = reader.int32();
        this.ConfidenceLevel = reader.float32();
        this.WindowedBuffer = mlnet.TimeSeriesUtils.deserializeFixedSizeQueueSingle(reader);
        this.InitialWindowedBuffer = mlnet.TimeSeriesUtils.deserializeFixedSizeQueueSingle(reader);
        this.Model = context.open('SSA');
    }
}

mlnet.SsaForecastingTransformer = class extends mlnet.SsaForecastingBaseWrapper {

    constructor(context) {
        super(context);
    }
}

mlnet.ColumnSelectingTransformer = class {

    constructor(context) {
        const reader = context.reader;
        if (context.check('DRPCOLST', 0x00010002, 0x00010002)) {
            throw new mlnet.Error("'LoadDropColumnsTransform' not supported.");
        }
        else if (context.check('CHSCOLSF', 0x00010001, 0x00010001)) {
            reader.int32(); // cbFloat
            this.KeepHidden = this._getHiddenOption(reader.byte());
            const count = reader.int32();
            this.inputs = [];
            for (let colIdx = 0; colIdx < count; colIdx++) {
                const dst = context.string();
                this.inputs.push(dst);
                context.string(); // src 
                this._getHiddenOption(reader.byte()); // colKeepHidden
            }
        }
        else {
            const keepColumns = reader.boolean();
            this.KeepHidden = reader.boolean();
            this.IgnoreMissing = reader.boolean();
            const length = reader.int32();
            this.inputs = [];
            for (let i = 0; i < length; i++) {
                this.inputs.push({ name: context.string() });
            }
            if (keepColumns) {
                this.ColumnsToKeep = this.inputs;
            }
            else {
                this.ColumnsToDrop = this.inputs;
            }
        }
    }

    _getHiddenOption(value) {
        switch (value) {
            case 1: return true;
            case 2: return false;
            default: throw new mlnet.Error('Unsupported hide option specified');
        }
    }
}

mlnet.XGBoostMulticlass = class {}

mlnet.NltTokenizeTransform = class {}

mlnet.DropColumnsTransform = class {}

mlnet.StopWordsTransform = class {}

mlnet.CSharpTransform = class {}

mlnet.GenericScoreTransform = class {}

mlnet.NormalizeTransform = class {}

mlnet.CdfColumnFunction = class {

    constructor(/* context, typeSrc */) {
        // TODO
    }
}

mlnet.MultiClassNetPredictor = class {}

mlnet.ProtonNNMCPred = class {}

mlnet.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'ML.NET Error';
    }
};

if (module && module.exports) {
    module.exports.ModelFactory = mlnet.ModelFactory;
}