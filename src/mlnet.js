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

        if (reader.schema && reader.schema.columns) {
            for (let column of reader.schema.columns) {
                this._inputs.push(new mlnet.Parameter(column.name, [
                    new mlnet.Argument(column.name)
                ]));
            }
        }

        if (reader.TransformerChain) {
            this._loadTransformer(metadata, '', reader.TransformerChain);
        }
    }

    _loadTransformer(metadata, group, transformer) {
        switch (transformer.__type__) {
            case 'TransformerChain':
                this._loadTransformerChain(metadata, transformer);
                break;
            default:
                this._nodes.push(new mlnet.Node(metadata, group, transformer));
                break;
        }


    }

    _loadTransformerChain(metadata, transformer) {
        this._groups = true;
        let group = transformer.__name__.split('/').splice(1).join('/');
        for (let childTransformer of transformer.transformers) {
            this._loadTransformer(metadata, group, childTransformer);
        }
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

    constructor(id) {
        this._id = id;
    }

    get id() {
        return this._id;
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

        for (let key of Object.keys(transformer).filter((key) => !key.startsWith('_'))) {
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

        let roots = new Set();
        for (let entry of entries) {
            let name = entry.name.split('\\').join('/');
            switch (name) {
                case 'Schema':
                    this.schema = new mlnet.BinaryLoader(new mlnet.Reader(entry.data)).schema;
                    break;
                case 'TrainingInfo/Version.txt':
                    this.version = new TextDecoder().decode(entry.data).split(' ').shift().split('\r').shift();
                    break;
                default:
                    roots.add(name.split('/').shift());
                    break;
            }
        }

        let catalog = new mlnet.ComponentCatalog();
        catalog.register('AnomalyPredXfer', mlnet.AnomalyPredictionTransformer);
        catalog.register('BinaryPredXfer', mlnet.BinaryPredictionTransformer);
        catalog.register('CaliPredExec', mlnet.CalibratedPredictor);
        catalog.register('CharToken', mlnet.TokenizingByCharactersTransformer);
        catalog.register('ClusteringPredXfer', mlnet.ClusteringPredictionTransformer);
        catalog.register('ConcatTransform', mlnet.ColumnConcatenatingTransformer);
        catalog.register('CopyTransform', mlnet.ColumnCopyingTransformer);
        catalog.register('ConvertTransform', mlnet.TypeConvertingTransformer);
        catalog.register('FastForestBinaryExec', mlnet.FastForestClassificationPredictor);
        catalog.register('FastTreeTweedieExec', mlnet.FastTreeTweedieModelParameters);
        catalog.register('FeatWCaliPredExec', mlnet.FeatureWeightsCalibratedModelParameters);
        catalog.register('GcnTransform', mlnet.LpNormNormalizingTransformer);
        catalog.register('IidChangePointDetector', mlnet.IidChangePointDetector);
        catalog.register('IidSpikeDetector', mlnet.IidSpikeDetector);
        catalog.register('ImageLoaderTransform', mlnet.ImageLoadingTransformer);
        catalog.register('ImageScalerTransform', mlnet.ImageResizingTransformer);
        catalog.register('ImagePixelExtractor', mlnet.ImageResizingTransformer);
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
        catalog.register('SelectColumnsTransform', mlnet.ColumnSelectingTransformer);
        catalog.register('TensorFlowTransform', mlnet.TensorFlowTransformer);
        catalog.register('TermTransform', mlnet.ValueToKeyMappingTransformer);
        catalog.register('Text', mlnet.TextFeaturizingEstimator);
        catalog.register('TextLoader', mlnet.TextLoader);
        catalog.register('TextNormalizerTransform', mlnet.TextNormalizingTransformer);
        catalog.register('TokenizeTextTransform', mlnet.WordTokenizingTransformer);
        catalog.register('TransformerChain', mlnet.TransformerChain);
        catalog.register('ValueMappingTransformer', mlnet.ValueMappingTransformer)

        for (let root of roots) {
            switch (root) {
                case 'TransformerChain':
                    var header = new mlnet.ModelHeader(entries, catalog, root);
                    this.TransformerChain = header.create();
                    break;
                default:
                    throw new mlnet.Error("Unknown '" + root + "'.");
            }
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
        this.directory = directory;
        let name = directory + '/Model.key';
        let entry = entries.find((entry) => entry.name == name || entry.name == name.replace(/\//g, '\\'));
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
        this.loaderSignature = textDecoder.decode(reader.bytes(24).filter((c) => c != 0));
        this.loaderSignatureAlt = textDecoder.decode(reader.bytes(24).filter((c) => c != 0));
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
            for (let j = 0; j < stringCount; j++) {
                this.strings.push(reader.string(stringSizes[j] >> 1));
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
        return new mlnet.ModelHeader(this._entries, this._catalog, this.directory + '/' + name);
    }

    openBinary(name) {
        name = this.directory + '/' + name;
        let entry = this._entries.find((entry) => entry.name == name || entry.name == name.replace(/\//g, '\\'));
        let reader = new mlnet.Reader(entry.data);
        return new mlnet.BinaryLoader(reader);
    }

    create() {
        let value = this._catalog.create(this.loaderSignature, this);
        value.__type__ = this.loaderSignature;
        value.__name__ = this.directory;
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

    bool() {
        return this.byte() != 0 ? true : false;
    }

    bools(count) {
        let values = [];
        for (let i = 0; i < count; i++) {
            values.push(this.bool());
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

    float64() {
        let value = this._dataView.getFloat64(this._position, true);
        this._position += 8;
        return value;
    }

    string(size) {
        let end = this._position + (size << 1);
        let text = '';
        while (size > 0) {
            let c = this.uint16();
            if (c == 0) {
                break;
            }
            size--;
            text += String.fromCharCode(c);
        }
        this._position = end;
        return text;
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
        this.schema.columns = [];
        for (let c = 0; c < columnCount; c  ++) {
            let name = this.string(reader);
            let codecName = this.string(reader);
            let codecData = reader.bytes(reader.leb128());
            // let codecReader = new mlnet.Reader(codecData);
            let codec = { name: codecName, data: codecData };
            switch (codecName) {
                case 'Single':
                    break;
                case 'TextSpan':
                    break;
                case 'VBuffer':
                    // codec.signature = codecReader.string();
                    break;
            }
            let compression = reader.byte(); // None = 0, Deflate = 1
            let rowsPerBlock = reader.leb128();
            let lookupOffset = reader.int64();
            let metadataTocOffset = reader.int64();
            this.schema.columns.push({
                name: name,
                codec: codec,
                compression: compression,
                rowsPerBlock: rowsPerBlock,
                lookupOffset: lookupOffset,
                metadataTocOffset: metadataTocOffset
            });
        }
    }

    string(reader) {
        let size = reader.leb128();
        let buffer = reader.bytes(size);
        return new TextDecoder('utf-8').decode(buffer);
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
        let r = context.reader;
        let length = r.int32();
        this.scopes = [];
        this.transformers = [];
        for (let i = 0; i < length; i++) {
            this.scopes.push(r.int32()); // 0x01 = Training, 0x02 = Testing, 0x04 = Scoring 
            let dirName = 'Transform_' + ('00' + i).slice(-3);
            let transformer = context.open(dirName).create();
            this.transformers.push(transformer);
        }
    }
};

mlnet.ColumnCopyingTransformer = class {

    constructor(context) {
        let r = context.reader;
        let length = r.uint32();
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
        let r = context.reader;
        if (context.modelVersionReadable >= 0x00010003) {
            let count = r.int32();
            for (let i = 0; i < count; i++) {
                this.outputs = [];
                this.outputs.push({ name: context.string() });
                let n = r.int32();
                this.inputs = [];
                for (let j = 0; j < n; j++) {
                    this.inputs.push({
                        name: context.string(),
                        alias: context.string(null)
                    });
                }
            }
        }
        else {
            // debugger;
            this.precision = r.int32();
            let n = r.int32();
            let names = [];
            let inputs = [];
            for (let i = 0; i < n; i++) {
                names.push(context.string());
                let numSources = r.int32();
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
                            let j = r.int32();
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
                this.outputs.push({ name: names[i] });
                this.inputs = inputs[i];
            }
        }
    }
};

mlnet.PredictionTransformerBase = class {

    constructor(context) {
        this.model = context.open('Model').create();
        this.trainSchema = context.openBinary('TrainSchema').schema;
    }
};

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
        let r = context.reader;
        if (r.int32() != 4) {
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
        let r = context.reader;
        let n = r.int32();
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
        let r = context.reader;
        if (context.modelVersionReadable === 0x00010001) {
            if (r.int32() !== 4) {
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
        // debugger;
    }
}

mlnet.ImageResizingTransformer = class extends mlnet.OneToOneTransformerBase {

    constructor(context) {
        super(context);
        // debugger;
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
        let r = context.reader;
        if (context.modelVersionWritten >= 0x00010003) {
            this.textMetadata = r.bools(this.outputs.length + this.inputs.length);
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
        let r = context.reader;
        let cmap = r.int32();
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
            for (let i = 0; i < cmap; ++i) {
                debugger;
                // termMap[i] = TermMap.TextImpl.Create(c, host)
            }
        }


    }
};

mlnet.TermMap = class {

    constructor(context) {
        let r = context.reader;
        let mtype = r.byte();
        this.pool = [];
        switch (mtype) {
            case 0: // Text
                var cstr = r.int32();
                for (let i = 0; i < cstr; i++) {
                    this.pool.push(context.string());
                }
                break;
            case 1: // Codec
                throw new mlnet.Error('');
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
        let r = context.reader;
        // LoadTransforms
        this.floatSize = r.int32();
        let cxf = r.int32();
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

mlnet.RowToRowTransformerBase = class {

    constructor(/* context */) {
    }
}

mlnet.OnnxTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        let r = context.reader;
        let numInputs = context.modelVersionWritten > 0x00010001 ? r.int32() : 1;
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        let numOutputs = context.modelVersionWritten > 0x00010001 ? r.int32() : 1;
        this.outputs = [];
        for (let i = 0; i < numOutputs; i++) {
            this.outputs.push({ name: context.string() });
        }
        if (context.modelVersionWritten > 0x0001000C) {
            let customShapeInfosLength = r.int32();
            this.loadedCustomShapeInfos = [];
            for (let i = 0; i < customShapeInfosLength; i++) {
                this.loadedCustomShapeInfos.push({
                    name: context.string(),
                    shape: r.int32s()
                });
            }
        }
    }
}

mlnet.TensorFlowTransformer = class extends mlnet.RowToRowTransformerBase {

    constructor(context) {
        super(context);
        let r = context.reader;
        this.frozen = context.modelVersionReadable >= 0x00010002 ? r.bool() : true;
        this.addBatchDimensionInput = context.modelVersionReadable >= 0x00010003 ? r.bool() : true;
        let numInputs = r.int32();
        this.inputs = [];
        for (let i = 0; i < numInputs; i++) {
            this.inputs.push({ name: context.string() });
        }
        let numOutputs = context.modelVersionReadable >= 0x00010002 ? r.int32() : 1;
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
        // let r = context.reader;
        if (context.modelVersionReadable === 0x00010001) {
            debugger;
        }
        else {
            this.chain = context.open('Chain').create();
        }
    }
}

mlnet.TextLoader = class {

    constructor(context) {
        let r = context.reader;
        this.floatSize = r.int32();
        this.maxRows = r.int64();
        this.flags = r.uint32();
        this.inputSize = r.int32();
        let separatorCount = r.int32();
        this.separators = [];
        for (let i = 0; i < separatorCount; i++) {
            this.separators.push(String.fromCharCode(r.uint16()));            
        }
        // this.bindinds = new mlnet.TextLoader.Bindinds(context);
    }
};

mlnet.TextLoader.Bindinds = class {

    constructor(context) {
        let r = context.reader;
        let cinfo = r.int32();
        for (let i = 0; i < cinfo; i++) {
            /*
            let name = r.string();
            let kind = r.byte();
            let itemType = null;
            if (r.bool()) {
                let isConfig = r.bool();
                let min = r.uint64();
                let count = r.int32();
                if (count == 0) {
                    itemType = new KeyType(kind, min, 0, isContig);
                }
                else
                {
                    itemType = new KeyType(kind, min, count);
                }
            }
            else {
                itemType = PrimitiveType.FromKind(kind);
            }

            let cseg = r.int32();
            let segs = [];
            for (let iseg = 0; iseg < cseg; iseg++) {
                segs.push({ 
                    min: r.int32(),
                    lim: r.int32(),
                    forceVector: (header.modelVersionWritten >= 0x0001000A) ? r.bool() : falses
                });
                Infos[iinfo] = ColInfo.Create(name, itemType, segs, false);
                NameToInfoIndex[name] = iinfo;
            }
            let textReader = header.openText('Header.txt');
            if (textReader.line().length > 0) {
                Parser.ParseSlotNames(parent, _header = result.AsMemory(), Infos, _slotNames);
            }
            AsSchema = Schema.Create(this);
            */
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
        let r = context.reader;
        let cbFloat = r.int32();
        if (cbFloat !== 4) {
            throw new mlnet.Error('This file was saved by an incompatible version.');
        }
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
        let r = context.reader;
        this.ParamA = r.float64();
        this.ParamB = r.float64();
    }
};

mlnet.RowToRowMapper = class { 

    constructor(/* context */) {
        // debugger;
    }
};

mlnet.IidAnomalyDetectionBaseWrapper = class { 

    constructor(/* context */) {
        // debugger;
    }
};

mlnet.IidChangePointDetector = class extends mlnet.IidAnomalyDetectionBaseWrapper {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.IidSpikeDetector = class extends mlnet.IidAnomalyDetectionBaseWrapper {

    constructor(context) {
        super(context);
        // debugger;
    }
}

mlnet.ColumnSelectingTransformer = class {

    constructor(context) {
        let r = context.reader;
        let keepColumns = r.bool();
        this.keepHidden = r.bool();
        this.ignoreMissing = r.bool();
        var length = r.int32();
        this.columns = [];
        for (let i = 0; i < length; i++) {
            this.columns.push({ name: context.string() });
        }
        if (keepColumns) {
            this.columnsToKeep = this.columns;
        }
        else {
            this.columnsToDrop = this.columns;
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