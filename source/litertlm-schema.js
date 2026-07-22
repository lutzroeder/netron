
export const litert = {};

litert.lm = litert.lm || {};

litert.lm.schema = litert.lm.schema || {};

litert.lm.schema.UInt8 = class UInt8 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.UInt8();
        $.value = reader.uint8_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.Int8 = class Int8 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Int8();
        $.value = reader.int8_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.UInt16 = class UInt16 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.UInt16();
        $.value = reader.uint16_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.Int16 = class Int16 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Int16();
        $.value = reader.int16_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.UInt32 = class UInt32 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.UInt32();
        $.value = reader.uint32_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.Int32 = class Int32 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Int32();
        $.value = reader.int32_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.Float32 = class Float32 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Float32();
        $.value = reader.float32_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.Bool = class Bool {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Bool();
        $.value = reader.bool_(position, 4, false);
        return $;
    }
};

litert.lm.schema.UInt64 = class UInt64 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.UInt64();
        $.value = reader.uint64_(position, 4, 0n);
        return $;
    }
};

litert.lm.schema.Int64 = class Int64 {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Int64();
        $.value = reader.int64_(position, 4, 0n);
        return $;
    }
};

litert.lm.schema.Double = class Double {

    static decode(reader, position) {
        const $ = new litert.lm.schema.Double();
        $.value = reader.float64_(position, 4, 0);
        return $;
    }
};

litert.lm.schema.StringValue = class StringValue {

    static decode(reader, position) {
        const $ = new litert.lm.schema.StringValue();
        $.value = reader.string_(position, 4, null);
        return $;
    }
};

litert.lm.schema.VData = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return litert.lm.schema.UInt8.decode(reader, position);
            case 2: return litert.lm.schema.Int8.decode(reader, position);
            case 3: return litert.lm.schema.UInt16.decode(reader, position);
            case 4: return litert.lm.schema.Int16.decode(reader, position);
            case 5: return litert.lm.schema.UInt32.decode(reader, position);
            case 6: return litert.lm.schema.Int32.decode(reader, position);
            case 7: return litert.lm.schema.Float32.decode(reader, position);
            case 8: return litert.lm.schema.Bool.decode(reader, position);
            case 9: return litert.lm.schema.StringValue.decode(reader, position);
            case 10: return litert.lm.schema.UInt64.decode(reader, position);
            case 11: return litert.lm.schema.Int64.decode(reader, position);
            case 12: return litert.lm.schema.Double.decode(reader, position);
            default: return undefined;
        }
    }
};

litert.lm.schema.KeyValuePair = class KeyValuePair {

    static decode(reader, position) {
        const $ = new litert.lm.schema.KeyValuePair();
        $.key = reader.string_(position, 4, null);
        $.value = reader.union(position, 6, litert.lm.schema.VData);
        return $;
    }
};

litert.lm.schema.SystemMetadata = class SystemMetadata {

    static decode(reader, position) {
        const $ = new litert.lm.schema.SystemMetadata();
        $.entries = reader.tables(position, 4, litert.lm.schema.KeyValuePair);
        return $;
    }
};

litert.lm.schema.AnySectionDataType = {
    NONE: 0, '0': 'NONE',
    GenericBinaryData: 1, '1': 'GenericBinaryData',
    Deprecated: 2, '2': 'Deprecated',
    TFLiteModel: 3, '3': 'TFLiteModel',
    SP_Tokenizer: 4, '4': 'SP_Tokenizer',
    LlmMetadataProto: 5, '5': 'LlmMetadataProto',
    HF_Tokenizer_Zlib: 6, '6': 'HF_Tokenizer_Zlib',
    TFLiteWeights: 7, '7': 'TFLiteWeights',
    EmbeddingMetadataProto: 8, '8': 'EmbeddingMetadataProto',
    ExecutorMetadataProto: 9, '9': 'ExecutorMetadataProto'
};

litert.lm.schema.SectionObject = class SectionObject {

    static decode(reader, position) {
        const $ = new litert.lm.schema.SectionObject();
        $.items = reader.tables(position, 4, litert.lm.schema.KeyValuePair);
        $.begin_offset = reader.uint64_(position, 6, 0n);
        $.end_offset = reader.uint64_(position, 8, 0n);
        $.data_type = reader.uint8_(position, 10, 0);
        return $;
    }
};

litert.lm.schema.SectionMetadata = class SectionMetadata {

    static decode(reader, position) {
        const $ = new litert.lm.schema.SectionMetadata();
        $.objects = reader.tables(position, 4, litert.lm.schema.SectionObject);
        return $;
    }
};

litert.lm.schema.LiteRTLMMetaData = class LiteRTLMMetaData {

    static create(reader) {
        return litert.lm.schema.LiteRTLMMetaData.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new litert.lm.schema.LiteRTLMMetaData();
        $.system_metadata = reader.table(position, 4, litert.lm.schema.SystemMetadata);
        $.section_metadata = reader.table(position, 6, litert.lm.schema.SectionMetadata);
        return $;
    }
};
