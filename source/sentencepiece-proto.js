var $root = protobuf.get('sentencepiece');

$root.sentencepiece = {};

$root.sentencepiece.TrainerSpec = class TrainerSpec {

    constructor() {
        this.input = [];
        this.accept_language = [];
        this.control_symbols = [];
        this.user_defined_symbols = [];
    }

    static decode(reader, length) {
        const message = new $root.sentencepiece.TrainerSpec();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input.push(reader.string());
                    break;
                case 7:
                    message.input_format = reader.string();
                    break;
                case 2:
                    message.model_prefix = reader.string();
                    break;
                case 3:
                    message.model_type = reader.int32();
                    break;
                case 4:
                    message.vocab_size = reader.int32();
                    break;
                case 5:
                    message.accept_language.push(reader.string());
                    break;
                case 6:
                    message.self_test_sample_size = reader.int32();
                    break;
                case 50:
                    message.enable_differential_privacy = reader.bool();
                    break;
                case 51:
                    message.differential_privacy_noise_level = reader.float();
                    break;
                case 52:
                    message.differential_privacy_clipping_threshold = reader.uint64();
                    break;
                case 10:
                    message.character_coverage = reader.float();
                    break;
                case 11:
                    message.input_sentence_size = reader.uint64();
                    break;
                case 19:
                    message.shuffle_input_sentence = reader.bool();
                    break;
                case 12:
                    message.mining_sentence_size = reader.int32();
                    break;
                case 13:
                    message.training_sentence_size = reader.int32();
                    break;
                case 14:
                    message.seed_sentencepiece_size = reader.int32();
                    break;
                case 15:
                    message.shrinking_factor = reader.float();
                    break;
                case 18:
                    message.max_sentence_length = reader.int32();
                    break;
                case 16:
                    message.num_threads = reader.int32();
                    break;
                case 17:
                    message.num_sub_iterations = reader.int32();
                    break;
                case 20:
                    message.max_sentencepiece_length = reader.int32();
                    break;
                case 21:
                    message.split_by_unicode_script = reader.bool();
                    break;
                case 23:
                    message.split_by_number = reader.bool();
                    break;
                case 22:
                    message.split_by_whitespace = reader.bool();
                    break;
                case 24:
                    message.treat_whitespace_as_suffix = reader.bool();
                    break;
                case 26:
                    message.allow_whitespace_only_pieces = reader.bool();
                    break;
                case 25:
                    message.split_digits = reader.bool();
                    break;
                case 53:
                    message.pretokenization_delimiter = reader.string();
                    break;
                case 30:
                    message.control_symbols.push(reader.string());
                    break;
                case 31:
                    message.user_defined_symbols.push(reader.string());
                    break;
                case 36:
                    message.required_chars = reader.string();
                    break;
                case 35:
                    message.byte_fallback = reader.bool();
                    break;
                case 32:
                    message.vocabulary_output_piece_score = reader.bool();
                    break;
                case 33:
                    message.hard_vocab_limit = reader.bool();
                    break;
                case 34:
                    message.use_all_vocab = reader.bool();
                    break;
                case 40:
                    message.unk_id = reader.int32();
                    break;
                case 41:
                    message.bos_id = reader.int32();
                    break;
                case 42:
                    message.eos_id = reader.int32();
                    break;
                case 43:
                    message.pad_id = reader.int32();
                    break;
                case 45:
                    message.unk_piece = reader.string();
                    break;
                case 46:
                    message.bos_piece = reader.string();
                    break;
                case 47:
                    message.eos_piece = reader.string();
                    break;
                case 48:
                    message.pad_piece = reader.string();
                    break;
                case 44:
                    message.unk_surface = reader.string();
                    break;
                case 49:
                    message.train_extremely_large_corpus = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.sentencepiece.TrainerSpec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "input_format":
                    message.input_format = reader.string();
                    break;
                case "model_prefix":
                    message.model_prefix = reader.string();
                    break;
                case "model_type":
                    message.model_type = reader.enum($root.sentencepiece.TrainerSpec.ModelType);
                    break;
                case "vocab_size":
                    message.vocab_size = reader.int32();
                    break;
                case "accept_language":
                    reader.array(message.accept_language, () => reader.string());
                    break;
                case "self_test_sample_size":
                    message.self_test_sample_size = reader.int32();
                    break;
                case "enable_differential_privacy":
                    message.enable_differential_privacy = reader.bool();
                    break;
                case "differential_privacy_noise_level":
                    message.differential_privacy_noise_level = reader.float();
                    break;
                case "differential_privacy_clipping_threshold":
                    message.differential_privacy_clipping_threshold = reader.uint64();
                    break;
                case "character_coverage":
                    message.character_coverage = reader.float();
                    break;
                case "input_sentence_size":
                    message.input_sentence_size = reader.uint64();
                    break;
                case "shuffle_input_sentence":
                    message.shuffle_input_sentence = reader.bool();
                    break;
                case "mining_sentence_size":
                    message.mining_sentence_size = reader.int32();
                    break;
                case "training_sentence_size":
                    message.training_sentence_size = reader.int32();
                    break;
                case "seed_sentencepiece_size":
                    message.seed_sentencepiece_size = reader.int32();
                    break;
                case "shrinking_factor":
                    message.shrinking_factor = reader.float();
                    break;
                case "max_sentence_length":
                    message.max_sentence_length = reader.int32();
                    break;
                case "num_threads":
                    message.num_threads = reader.int32();
                    break;
                case "num_sub_iterations":
                    message.num_sub_iterations = reader.int32();
                    break;
                case "max_sentencepiece_length":
                    message.max_sentencepiece_length = reader.int32();
                    break;
                case "split_by_unicode_script":
                    message.split_by_unicode_script = reader.bool();
                    break;
                case "split_by_number":
                    message.split_by_number = reader.bool();
                    break;
                case "split_by_whitespace":
                    message.split_by_whitespace = reader.bool();
                    break;
                case "treat_whitespace_as_suffix":
                    message.treat_whitespace_as_suffix = reader.bool();
                    break;
                case "allow_whitespace_only_pieces":
                    message.allow_whitespace_only_pieces = reader.bool();
                    break;
                case "split_digits":
                    message.split_digits = reader.bool();
                    break;
                case "pretokenization_delimiter":
                    message.pretokenization_delimiter = reader.string();
                    break;
                case "control_symbols":
                    reader.array(message.control_symbols, () => reader.string());
                    break;
                case "user_defined_symbols":
                    reader.array(message.user_defined_symbols, () => reader.string());
                    break;
                case "required_chars":
                    message.required_chars = reader.string();
                    break;
                case "byte_fallback":
                    message.byte_fallback = reader.bool();
                    break;
                case "vocabulary_output_piece_score":
                    message.vocabulary_output_piece_score = reader.bool();
                    break;
                case "hard_vocab_limit":
                    message.hard_vocab_limit = reader.bool();
                    break;
                case "use_all_vocab":
                    message.use_all_vocab = reader.bool();
                    break;
                case "unk_id":
                    message.unk_id = reader.int32();
                    break;
                case "bos_id":
                    message.bos_id = reader.int32();
                    break;
                case "eos_id":
                    message.eos_id = reader.int32();
                    break;
                case "pad_id":
                    message.pad_id = reader.int32();
                    break;
                case "unk_piece":
                    message.unk_piece = reader.string();
                    break;
                case "bos_piece":
                    message.bos_piece = reader.string();
                    break;
                case "eos_piece":
                    message.eos_piece = reader.string();
                    break;
                case "pad_piece":
                    message.pad_piece = reader.string();
                    break;
                case "unk_surface":
                    message.unk_surface = reader.string();
                    break;
                case "train_extremely_large_corpus":
                    message.train_extremely_large_corpus = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.sentencepiece.TrainerSpec.prototype.input_format = "";
$root.sentencepiece.TrainerSpec.prototype.model_prefix = "";
$root.sentencepiece.TrainerSpec.prototype.model_type = 1;
$root.sentencepiece.TrainerSpec.prototype.vocab_size = 8000;
$root.sentencepiece.TrainerSpec.prototype.self_test_sample_size = 0;
$root.sentencepiece.TrainerSpec.prototype.enable_differential_privacy = false;
$root.sentencepiece.TrainerSpec.prototype.differential_privacy_noise_level = 0;
$root.sentencepiece.TrainerSpec.prototype.differential_privacy_clipping_threshold = protobuf.Uint64.create(0);
$root.sentencepiece.TrainerSpec.prototype.character_coverage = 0.9995;
$root.sentencepiece.TrainerSpec.prototype.input_sentence_size = protobuf.Uint64.create(0);
$root.sentencepiece.TrainerSpec.prototype.shuffle_input_sentence = true;
$root.sentencepiece.TrainerSpec.prototype.mining_sentence_size = 0;
$root.sentencepiece.TrainerSpec.prototype.training_sentence_size = 0;
$root.sentencepiece.TrainerSpec.prototype.seed_sentencepiece_size = 1000000;
$root.sentencepiece.TrainerSpec.prototype.shrinking_factor = 0.75;
$root.sentencepiece.TrainerSpec.prototype.max_sentence_length = 4192;
$root.sentencepiece.TrainerSpec.prototype.num_threads = 16;
$root.sentencepiece.TrainerSpec.prototype.num_sub_iterations = 2;
$root.sentencepiece.TrainerSpec.prototype.max_sentencepiece_length = 16;
$root.sentencepiece.TrainerSpec.prototype.split_by_unicode_script = true;
$root.sentencepiece.TrainerSpec.prototype.split_by_number = true;
$root.sentencepiece.TrainerSpec.prototype.split_by_whitespace = true;
$root.sentencepiece.TrainerSpec.prototype.treat_whitespace_as_suffix = false;
$root.sentencepiece.TrainerSpec.prototype.allow_whitespace_only_pieces = false;
$root.sentencepiece.TrainerSpec.prototype.split_digits = false;
$root.sentencepiece.TrainerSpec.prototype.pretokenization_delimiter = "";
$root.sentencepiece.TrainerSpec.prototype.required_chars = "";
$root.sentencepiece.TrainerSpec.prototype.byte_fallback = false;
$root.sentencepiece.TrainerSpec.prototype.vocabulary_output_piece_score = true;
$root.sentencepiece.TrainerSpec.prototype.hard_vocab_limit = true;
$root.sentencepiece.TrainerSpec.prototype.use_all_vocab = false;
$root.sentencepiece.TrainerSpec.prototype.unk_id = 0;
$root.sentencepiece.TrainerSpec.prototype.bos_id = 1;
$root.sentencepiece.TrainerSpec.prototype.eos_id = 2;
$root.sentencepiece.TrainerSpec.prototype.pad_id = -1;
$root.sentencepiece.TrainerSpec.prototype.unk_piece = "<unk>";
$root.sentencepiece.TrainerSpec.prototype.bos_piece = "<s>";
$root.sentencepiece.TrainerSpec.prototype.eos_piece = "</s>";
$root.sentencepiece.TrainerSpec.prototype.pad_piece = "<pad>";
$root.sentencepiece.TrainerSpec.prototype.unk_surface = " E28187 ";
$root.sentencepiece.TrainerSpec.prototype.train_extremely_large_corpus = false;

$root.sentencepiece.TrainerSpec.ModelType = {
    "UNIGRAM": 1,
    "BPE": 2,
    "WORD": 3,
    "CHAR": 4
};

$root.sentencepiece.NormalizerSpec = class NormalizerSpec {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.sentencepiece.NormalizerSpec();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.precompiled_charsmap = reader.bytes();
                    break;
                case 3:
                    message.add_dummy_prefix = reader.bool();
                    break;
                case 4:
                    message.remove_extra_whitespaces = reader.bool();
                    break;
                case 5:
                    message.escape_whitespaces = reader.bool();
                    break;
                case 6:
                    message.normalization_rule_tsv = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.sentencepiece.NormalizerSpec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "precompiled_charsmap":
                    message.precompiled_charsmap = reader.bytes();
                    break;
                case "add_dummy_prefix":
                    message.add_dummy_prefix = reader.bool();
                    break;
                case "remove_extra_whitespaces":
                    message.remove_extra_whitespaces = reader.bool();
                    break;
                case "escape_whitespaces":
                    message.escape_whitespaces = reader.bool();
                    break;
                case "normalization_rule_tsv":
                    message.normalization_rule_tsv = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.sentencepiece.NormalizerSpec.prototype.name = "";
$root.sentencepiece.NormalizerSpec.prototype.precompiled_charsmap = new Uint8Array([]);
$root.sentencepiece.NormalizerSpec.prototype.add_dummy_prefix = true;
$root.sentencepiece.NormalizerSpec.prototype.remove_extra_whitespaces = true;
$root.sentencepiece.NormalizerSpec.prototype.escape_whitespaces = true;
$root.sentencepiece.NormalizerSpec.prototype.normalization_rule_tsv = "";

$root.sentencepiece.SelfTestData = class SelfTestData {

    constructor() {
        this.samples = [];
    }

    static decode(reader, length) {
        const message = new $root.sentencepiece.SelfTestData();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.samples.push($root.sentencepiece.SelfTestData.Sample.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.sentencepiece.SelfTestData();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "samples":
                    message.samples.push($root.sentencepiece.SelfTestData.Sample.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.sentencepiece.SelfTestData.Sample = class Sample {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.sentencepiece.SelfTestData.Sample();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input = reader.string();
                    break;
                case 2:
                    message.expected = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.sentencepiece.SelfTestData.Sample();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    message.input = reader.string();
                    break;
                case "expected":
                    message.expected = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.sentencepiece.SelfTestData.Sample.prototype.input = "";
$root.sentencepiece.SelfTestData.Sample.prototype.expected = "";

$root.sentencepiece.ModelProto = class ModelProto {

    constructor() {
        this.pieces = [];
    }

    static decode(reader, length) {
        const message = new $root.sentencepiece.ModelProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pieces.push($root.sentencepiece.ModelProto.SentencePiece.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.trainer_spec = $root.sentencepiece.TrainerSpec.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.normalizer_spec = $root.sentencepiece.NormalizerSpec.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.self_test_data = $root.sentencepiece.SelfTestData.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.denormalizer_spec = $root.sentencepiece.NormalizerSpec.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.sentencepiece.ModelProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pieces":
                    message.pieces.push($root.sentencepiece.ModelProto.SentencePiece.decodeText(reader));
                    break;
                case "trainer_spec":
                    message.trainer_spec = $root.sentencepiece.TrainerSpec.decodeText(reader);
                    break;
                case "normalizer_spec":
                    message.normalizer_spec = $root.sentencepiece.NormalizerSpec.decodeText(reader);
                    break;
                case "self_test_data":
                    message.self_test_data = $root.sentencepiece.SelfTestData.decodeText(reader);
                    break;
                case "denormalizer_spec":
                    message.denormalizer_spec = $root.sentencepiece.NormalizerSpec.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.sentencepiece.ModelProto.prototype.trainer_spec = null;
$root.sentencepiece.ModelProto.prototype.normalizer_spec = null;
$root.sentencepiece.ModelProto.prototype.self_test_data = null;
$root.sentencepiece.ModelProto.prototype.denormalizer_spec = null;

$root.sentencepiece.ModelProto.SentencePiece = class SentencePiece {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.sentencepiece.ModelProto.SentencePiece();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.piece = reader.string();
                    break;
                case 2:
                    message.score = reader.float();
                    break;
                case 3:
                    message.type = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.sentencepiece.ModelProto.SentencePiece();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "piece":
                    message.piece = reader.string();
                    break;
                case "score":
                    message.score = reader.float();
                    break;
                case "type":
                    message.type = reader.enum($root.sentencepiece.ModelProto.SentencePiece.Type);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.sentencepiece.ModelProto.SentencePiece.prototype.piece = "";
$root.sentencepiece.ModelProto.SentencePiece.prototype.score = 0;
$root.sentencepiece.ModelProto.SentencePiece.prototype.type = 1;

$root.sentencepiece.ModelProto.SentencePiece.Type = {
    "NORMAL": 1,
    "UNKNOWN": 2,
    "CONTROL": 3,
    "USER_DEFINED": 4,
    "BYTE": 6,
    "UNUSED": 5
};
