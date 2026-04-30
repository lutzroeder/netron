
import * as flatbuffers from './flatbuffers.js';
import * as protobuf from './protobuf.js';

const litertlm = {};

litertlm.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length >= 32) {
            const buffer = stream.peek(8);
            const magic = String.fromCharCode(...buffer);
            if (magic === 'LITERTLM') {
                return context.set('litertlm', stream);
            }
        }
        return null;
    }

    async open(context) {
        litertlm.schema = await context.require('./litertlm-schema');
        litertlm.schema = litertlm.schema.litert.lm.schema;
        const stream = context.value;
        const header = stream.peek(32);
        const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
        const major = view.getUint32(8, true);
        const minor = view.getUint32(12, true);
        const patch = view.getUint32(16, true);
        const version = `${major}.${minor}.${patch}`;
        const reader = flatbuffers.BinaryReader.open(stream, 32);
        if (!reader) {
            throw new litertlm.Error('Invalid LiteRT-LM header.');
        }
        const data = litertlm.schema.LiteRTLMMetaData.create(reader);
        const sections = data.section_metadata ? data.section_metadata.objects : [];
        const AnySectionDataType = litertlm.schema.AnySectionDataType;
        const modules = [];
        let metadata = null;
        for (const section of sections) {
            const begin = section.begin_offset.toNumber();
            const end = section.end_offset.toNumber();
            switch (section.data_type) {
                case AnySectionDataType.TFLiteModel: {
                    stream.seek(begin);
                    const buffer = stream.read(end - begin);
                    const content = context.context(`model_${modules.length}.tflite`, buffer);
                    const tflite = await context.require('./tflite'); // eslint-disable-line no-await-in-loop
                    const modelFactory = new tflite.ModelFactory();
                    await modelFactory.match(content); // eslint-disable-line no-await-in-loop
                    const model = await modelFactory.open(content); // eslint-disable-line no-await-in-loop
                    if (model && Array.isArray(model.modules)) {
                        for (const module of model.modules) {
                            modules.push(module);
                        }
                    }
                    break;
                }
                case AnySectionDataType.SP_Tokenizer: {
                    stream.seek(begin);
                    const buffer = stream.read(end - begin);
                    const content = context.context('tokenizer.model', buffer);
                    content.set('sentencepiece');
                    const sentencepiece = await context.require('./sentencepiece'); // eslint-disable-line no-await-in-loop
                    const modelFactory = new sentencepiece.ModelFactory();
                    const model = await modelFactory.open(content); // eslint-disable-line no-await-in-loop
                    if (model && Array.isArray(model.modules)) {
                        for (const module of model.modules) {
                            modules.push(module);
                        }
                    }
                    break;
                }
                case AnySectionDataType.LlmMetadataProto: {
                    litertlm.proto = await context.require('./litertlm-proto'); // eslint-disable-line no-await-in-loop
                    litertlm.proto = litertlm.proto.litert.lm.proto;
                    stream.seek(begin);
                    const buffer = stream.read(end - begin);
                    const pbReader = protobuf.BinaryReader.open(buffer);
                    metadata = litertlm.proto.LlmMetadata.decode(pbReader);
                    break;
                }
                default: {
                    break;
                }
            }
        }
        return new litertlm.Model(version, data, sections, modules, metadata);
    }
};

litertlm.Model = class {

    constructor(version, data, sections, modules, metadata) {
        this.format = `LiteRT-LM v${version}`;
        this.modules = modules;
        this.metadata = [];
        if (data.system_metadata) {
            for (const entry of data.system_metadata.entries) {
                if (entry.value) {
                    this.metadata.push(new litertlm.Argument(entry.key, entry.value.value));
                }
            }
        }
        if (metadata) {
            const format = (value) => {
                if (value === null || value === undefined || value === 0 || value === '') {
                    return null;
                }
                if (Array.isArray(value) || typeof value === 'object') {
                    const values = Array.isArray(value) ? value : Object.values(value);
                    const items = values.map((v) => format(v)).filter((v) => v !== null);
                    return items.length > 0 ? items.join(', ') : null;
                }
                return value;
            };
            for (const [name, value] of Object.entries(metadata)) {
                const formatted = format(value);
                if (formatted !== null) {
                    this.metadata.push(new litertlm.Argument(name, formatted));
                }
            }
        }
    }
};

litertlm.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

litertlm.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading LiteRT-LM model.';
    }
};

export const ModelFactory = litertlm.ModelFactory;
