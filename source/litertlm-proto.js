
export const litert = {};

litert.lm = {};

litert.lm.proto = {};

litert.lm.proto.PromptAffixes = class PromptAffixes {

    static decode(reader, length) {
        const message = new litert.lm.proto.PromptAffixes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.prefix = reader.string();
                    break;
                case 2:
                    message.suffix = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.PromptAffixes.prototype.prefix = "";
litert.lm.proto.PromptAffixes.prototype.suffix = "";

litert.lm.proto.Channel = class Channel {

    static decode(reader, length) {
        const message = new litert.lm.proto.Channel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.channel_name = reader.string();
                    break;
                case 2:
                    message.start = reader.string();
                    break;
                case 3:
                    message.end = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.Channel.prototype.channel_name = "";
litert.lm.proto.Channel.prototype.start = "";
litert.lm.proto.Channel.prototype.end = "";

litert.lm.proto.PromptTemplates = class PromptTemplates {

    static decode(reader, length) {
        const message = new litert.lm.proto.PromptTemplates();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.user = litert.lm.proto.PromptAffixes.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.model = litert.lm.proto.PromptAffixes.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.system = litert.lm.proto.PromptAffixes.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.PromptTemplates.prototype.user = null;
litert.lm.proto.PromptTemplates.prototype.model = null;
litert.lm.proto.PromptTemplates.prototype.system = null;

litert.lm.proto.LlmMetadata = class LlmMetadata {

    constructor() {
        this.stop_tokens = [];
        this.channels = [];
    }

    static decode(reader, length) {
        const message = new litert.lm.proto.LlmMetadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stop_tokens.push(litert.lm.proto.TokenUnion.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.prompt_templates = litert.lm.proto.PromptTemplates.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.sampler_params = litert.lm.proto.SamplerParameters.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.max_num_tokens = reader.int32();
                    break;
                case 6:
                    message.llm_model_type = litert.lm.proto.LlmModelType.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.jinja_prompt_template = reader.string();
                    break;
                case 8:
                    message.channels.push(litert.lm.proto.Channel.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.LlmMetadata.prototype.start_token = null;
litert.lm.proto.LlmMetadata.prototype.prompt_templates = null;
litert.lm.proto.LlmMetadata.prototype.sampler_params = null;
litert.lm.proto.LlmMetadata.prototype.max_num_tokens = 0;
litert.lm.proto.LlmMetadata.prototype.llm_model_type = null;
litert.lm.proto.LlmMetadata.prototype.jinja_prompt_template = "";

litert.lm.proto.LlmModelType = class LlmModelType {

    get model_type() {
        litert.lm.proto.LlmModelType.model_typeSet = litert.lm.proto.LlmModelType.model_typeSet || new Set(["generic_model", "gemma3n", "function_gemma", "gemma3", "qwen3", "qwen2p5", "gemma4", "fast_vlm"]);
        return Object.keys(this).find((key) => litert.lm.proto.LlmModelType.model_typeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new litert.lm.proto.LlmModelType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.generic_model = litert.lm.proto.GenericModel.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.gemma3n = litert.lm.proto.Gemma3N.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.function_gemma = litert.lm.proto.FunctionGemma.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.gemma3 = litert.lm.proto.Gemma3.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.qwen3 = litert.lm.proto.Qwen3.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.qwen2p5 = litert.lm.proto.Qwen2p5.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.gemma4 = litert.lm.proto.Gemma4.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.fast_vlm = litert.lm.proto.FastVlm.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.GenericModel = class GenericModel {

    static decode(reader, length) {
        const message = new litert.lm.proto.GenericModel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.model_role = reader.string();
                    break;
                case 2:
                    message.force_string_content = reader.bool();
                    break;
                case 3:
                    message.image_enabled = reader.bool();
                    break;
                case 4:
                    message.audio_enabled = reader.bool();
                    break;
                case 5:
                    message.delimiter_regex = reader.string();
                    break;
                case 6:
                    message.image_token_regex = reader.string();
                    break;
                case 7:
                    message.audio_token_regex = reader.string();
                    break;
                case 8:
                    message.image_tensor_height = reader.int32();
                    break;
                case 9:
                    message.image_tensor_width = reader.int32();
                    break;
                case 10:
                    message.patch_width = reader.int32();
                    break;
                case 11:
                    message.patch_height = reader.int32();
                    break;
                case 12:
                    message.max_num_patches = reader.int32();
                    break;
                case 13:
                    message.pooling_kernel_size = reader.int32();
                    break;
                case 14:
                    message.start_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.end_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.image_prefix = reader.string();
                    break;
                case 17:
                    message.image_suffix = reader.string();
                    break;
                case 18:
                    message.add_image_end = reader.bool();
                    break;
                case 19:
                    message.skip_mel_spectrogram_extraction = reader.bool();
                    break;
                case 20:
                    message.audio_sample_rate_hz = reader.int32();
                    break;
                case 21:
                    message.audio_num_channels = reader.int32();
                    break;
                case 22:
                    message.audio_frame_length = reader.int32();
                    break;
                case 23:
                    message.audio_hop_length = reader.int32();
                    break;
                case 24:
                    message.audio_fft_length = reader.int32();
                    break;
                case 25:
                    message.audio_input_scale = reader.float();
                    break;
                case 26:
                    message.audio_pre_emphasis_factor = reader.float();
                    break;
                case 27:
                    message.audio_num_mel_bins = reader.int32();
                    break;
                case 28:
                    message.audio_mel_low_hz = reader.float();
                    break;
                case 29:
                    message.audio_mel_high_hz = reader.float();
                    break;
                case 30:
                    message.audio_mel_floor = reader.float();
                    break;
                case 31:
                    message.audio_normalize_mel = reader.bool();
                    break;
                case 32:
                    message.audio_add_floor_to_mel_before_log = reader.bool();
                    break;
                case 33:
                    message.audio_semicausal_padding = reader.bool();
                    break;
                case 34:
                    message.audio_non_zero_hanning = reader.bool();
                    break;
                case 35:
                    message.audio_periodic_hanning = reader.bool();
                    break;
                case 36:
                    message.audio_fft_padding_type = reader.int32();
                    break;
                case 37:
                    message.start_of_audio_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 38:
                    message.end_of_audio_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 39:
                    message.audio_prefix = reader.string();
                    break;
                case 40:
                    message.audio_suffix = reader.string();
                    break;
                case 41:
                    message.add_audio_end = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.GenericModel.prototype.model_role = "";
litert.lm.proto.GenericModel.prototype.force_string_content = false;
litert.lm.proto.GenericModel.prototype.image_enabled = false;
litert.lm.proto.GenericModel.prototype.audio_enabled = false;
litert.lm.proto.GenericModel.prototype.delimiter_regex = "";
litert.lm.proto.GenericModel.prototype.image_token_regex = "";
litert.lm.proto.GenericModel.prototype.audio_token_regex = "";
litert.lm.proto.GenericModel.prototype.image_tensor_height = 0;
litert.lm.proto.GenericModel.prototype.image_tensor_width = 0;
litert.lm.proto.GenericModel.prototype.patch_width = 0;
litert.lm.proto.GenericModel.prototype.patch_height = 0;
litert.lm.proto.GenericModel.prototype.max_num_patches = 0;
litert.lm.proto.GenericModel.prototype.pooling_kernel_size = 0;
litert.lm.proto.GenericModel.prototype.start_of_image_token = null;
litert.lm.proto.GenericModel.prototype.end_of_image_token = null;
litert.lm.proto.GenericModel.prototype.image_prefix = "";
litert.lm.proto.GenericModel.prototype.image_suffix = "";
litert.lm.proto.GenericModel.prototype.add_image_end = false;
litert.lm.proto.GenericModel.prototype.skip_mel_spectrogram_extraction = false;
litert.lm.proto.GenericModel.prototype.audio_sample_rate_hz = 0;
litert.lm.proto.GenericModel.prototype.audio_num_channels = 0;
litert.lm.proto.GenericModel.prototype.audio_frame_length = 0;
litert.lm.proto.GenericModel.prototype.audio_hop_length = 0;
litert.lm.proto.GenericModel.prototype.audio_fft_length = 0;
litert.lm.proto.GenericModel.prototype.audio_input_scale = 0;
litert.lm.proto.GenericModel.prototype.audio_pre_emphasis_factor = 0;
litert.lm.proto.GenericModel.prototype.audio_num_mel_bins = 0;
litert.lm.proto.GenericModel.prototype.audio_mel_low_hz = 0;
litert.lm.proto.GenericModel.prototype.audio_mel_high_hz = 0;
litert.lm.proto.GenericModel.prototype.audio_mel_floor = 0;
litert.lm.proto.GenericModel.prototype.audio_normalize_mel = false;
litert.lm.proto.GenericModel.prototype.audio_add_floor_to_mel_before_log = false;
litert.lm.proto.GenericModel.prototype.audio_semicausal_padding = false;
litert.lm.proto.GenericModel.prototype.audio_non_zero_hanning = false;
litert.lm.proto.GenericModel.prototype.audio_periodic_hanning = false;
litert.lm.proto.GenericModel.prototype.audio_fft_padding_type = 0;
litert.lm.proto.GenericModel.prototype.start_of_audio_token = null;
litert.lm.proto.GenericModel.prototype.end_of_audio_token = null;
litert.lm.proto.GenericModel.prototype.audio_prefix = "";
litert.lm.proto.GenericModel.prototype.audio_suffix = "";
litert.lm.proto.GenericModel.prototype.add_audio_end = false;

litert.lm.proto.Gemma3N = class Gemma3N {

    static decode(reader, length) {
        const message = new litert.lm.proto.Gemma3N();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.end_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.image_tensor_height = reader.int32();
                    break;
                case 4:
                    message.image_tensor_width = reader.int32();
                    break;
                case 5:
                    message.start_of_audio_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.end_of_audio_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.Gemma3N.prototype.start_of_image_token = null;
litert.lm.proto.Gemma3N.prototype.end_of_image_token = null;
litert.lm.proto.Gemma3N.prototype.image_tensor_height = 0;
litert.lm.proto.Gemma3N.prototype.image_tensor_width = 0;
litert.lm.proto.Gemma3N.prototype.start_of_audio_token = null;
litert.lm.proto.Gemma3N.prototype.end_of_audio_token = null;

litert.lm.proto.ConstraintMode = {
    "CONSTRAINT_MODE_UNSPECIFIED": 0,
    "CONSTRAINT_MODE_TEXT_AND_OR": 1,
    "CONSTRAINT_MODE_FUNCTION_CALL_ONLY": 2
};

litert.lm.proto.FftPaddingType = {
    "FFT_PADDING_TYPE_UNSPECIFIED": 0,
    "FFT_PADDING_TYPE_RIGHT": 1,
    "FFT_PADDING_TYPE_CENTER": 2
};

litert.lm.proto.FunctionGemma = class FunctionGemma {

    static decode(reader, length) {
        const message = new litert.lm.proto.FunctionGemma();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 5:
                    message.code_fence_start = reader.string();
                    break;
                case 6:
                    message.code_fence_end = reader.string();
                    break;
                case 7:
                    message.syntax_type = reader.string();
                    break;
                case 8:
                    message.escape_fence_strings = reader.bool();
                    break;
                case 9:
                    message.tool_code_regex = reader.string();
                    break;
                case 10:
                    message.open_quote = reader.string();
                    break;
                case 11:
                    message.close_quote = reader.string();
                    break;
                case 12:
                    message.function_response_start = reader.string();
                    break;
                case 13:
                    message.use_template_for_fc_format = reader.bool();
                    break;
                case 14:
                    message.constraint_mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.FunctionGemma.prototype.code_fence_start = "";
litert.lm.proto.FunctionGemma.prototype.code_fence_end = "";
litert.lm.proto.FunctionGemma.prototype.syntax_type = "";
litert.lm.proto.FunctionGemma.prototype.escape_fence_strings = false;
litert.lm.proto.FunctionGemma.prototype.tool_code_regex = "";
litert.lm.proto.FunctionGemma.prototype.open_quote = "";
litert.lm.proto.FunctionGemma.prototype.close_quote = "";
litert.lm.proto.FunctionGemma.prototype.function_response_start = "";
litert.lm.proto.FunctionGemma.prototype.use_template_for_fc_format = false;
litert.lm.proto.FunctionGemma.prototype.constraint_mode = 0;

litert.lm.proto.Gemma3 = class Gemma3 {

    static decode(reader, length) {
        const message = new litert.lm.proto.Gemma3();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.end_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.image_tensor_height = reader.int32();
                    break;
                case 4:
                    message.image_tensor_width = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.Gemma3.prototype.start_of_image_token = null;
litert.lm.proto.Gemma3.prototype.end_of_image_token = null;
litert.lm.proto.Gemma3.prototype.image_tensor_height = 0;
litert.lm.proto.Gemma3.prototype.image_tensor_width = 0;

litert.lm.proto.Qwen3 = class Qwen3 {

    static decode(reader, length) {
        const message = new litert.lm.proto.Qwen3();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.code_fence_start = reader.string();
                    break;
                case 2:
                    message.code_fence_end = reader.string();
                    break;
                case 3:
                    message.escape_fence_strings = reader.bool();
                    break;
                case 4:
                    message.tool_code_regex = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.Qwen3.prototype.code_fence_start = "";
litert.lm.proto.Qwen3.prototype.code_fence_end = "";
litert.lm.proto.Qwen3.prototype.escape_fence_strings = false;
litert.lm.proto.Qwen3.prototype.tool_code_regex = "";

litert.lm.proto.Qwen2p5 = class Qwen2p5 {

    static decode(reader, length) {
        const message = new litert.lm.proto.Qwen2p5();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.code_fence_start = reader.string();
                    break;
                case 2:
                    message.code_fence_end = reader.string();
                    break;
                case 3:
                    message.escape_fence_strings = reader.bool();
                    break;
                case 4:
                    message.tool_code_regex = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.Qwen2p5.prototype.code_fence_start = "";
litert.lm.proto.Qwen2p5.prototype.code_fence_end = "";
litert.lm.proto.Qwen2p5.prototype.escape_fence_strings = false;
litert.lm.proto.Qwen2p5.prototype.tool_code_regex = "";

litert.lm.proto.Gemma4 = class Gemma4 {

    static decode(reader, length) {
        const message = new litert.lm.proto.Gemma4();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.end_of_image_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.patch_width = reader.int32();
                    break;
                case 16:
                    message.patch_height = reader.int32();
                    break;
                case 17:
                    message.max_num_patches = reader.int32();
                    break;
                case 18:
                    message.pooling_kernel_size = reader.int32();
                    break;
                case 3:
                    message.start_of_audio_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.end_of_audio_token = litert.lm.proto.TokenUnion.decode(reader, reader.uint32());
                    break;
                case 19:
                    message.skip_mel_spectrogram_extraction = reader.bool();
                    break;
                case 5:
                    message.code_fence_start = reader.string();
                    break;
                case 6:
                    message.code_fence_end = reader.string();
                    break;
                case 7:
                    message.syntax_type = reader.string();
                    break;
                case 8:
                    message.escape_fence_strings = reader.bool();
                    break;
                case 9:
                    message.tool_code_regex = reader.string();
                    break;
                case 10:
                    message.open_quote = reader.string();
                    break;
                case 11:
                    message.close_quote = reader.string();
                    break;
                case 12:
                    message.function_response_start = reader.string();
                    break;
                case 13:
                    message.use_template_for_fc_format = reader.bool();
                    break;
                case 14:
                    message.constraint_mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.Gemma4.prototype.start_of_image_token = null;
litert.lm.proto.Gemma4.prototype.end_of_image_token = null;
litert.lm.proto.Gemma4.prototype.patch_width = 0;
litert.lm.proto.Gemma4.prototype.patch_height = 0;
litert.lm.proto.Gemma4.prototype.max_num_patches = 0;
litert.lm.proto.Gemma4.prototype.pooling_kernel_size = 0;
litert.lm.proto.Gemma4.prototype.start_of_audio_token = null;
litert.lm.proto.Gemma4.prototype.end_of_audio_token = null;
litert.lm.proto.Gemma4.prototype.skip_mel_spectrogram_extraction = false;
litert.lm.proto.Gemma4.prototype.code_fence_start = "";
litert.lm.proto.Gemma4.prototype.code_fence_end = "";
litert.lm.proto.Gemma4.prototype.syntax_type = "";
litert.lm.proto.Gemma4.prototype.escape_fence_strings = false;
litert.lm.proto.Gemma4.prototype.tool_code_regex = "";
litert.lm.proto.Gemma4.prototype.open_quote = "";
litert.lm.proto.Gemma4.prototype.close_quote = "";
litert.lm.proto.Gemma4.prototype.function_response_start = "";
litert.lm.proto.Gemma4.prototype.use_template_for_fc_format = false;
litert.lm.proto.Gemma4.prototype.constraint_mode = 0;

litert.lm.proto.FastVlm = class FastVlm {

    static decode(reader, length) {
        const message = new litert.lm.proto.FastVlm();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 3:
                    message.image_tensor_height = reader.int32();
                    break;
                case 4:
                    message.image_tensor_width = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.FastVlm.prototype.image_tensor_height = 0;
litert.lm.proto.FastVlm.prototype.image_tensor_width = 0;

litert.lm.proto.TokenUnion = class TokenUnion {

    get token_union() {
        litert.lm.proto.TokenUnion.token_unionSet = litert.lm.proto.TokenUnion.token_unionSet || new Set(["token_ids", "token_str"]);
        return Object.keys(this).find((key) => litert.lm.proto.TokenUnion.token_unionSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new litert.lm.proto.TokenUnion();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.token_ids = litert.lm.proto.TokenIds.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.token_str = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.TokenIds = class TokenIds {

    constructor() {
        this.ids = [];
    }

    static decode(reader, length) {
        const message = new litert.lm.proto.TokenIds();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ids = reader.array(message.ids, () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.SamplerParameters = class SamplerParameters {

    static decode(reader, length) {
        const message = new litert.lm.proto.SamplerParameters();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.k = reader.int32();
                    break;
                case 3:
                    message.p = reader.float();
                    break;
                case 4:
                    message.temperature = reader.float();
                    break;
                case 5:
                    message.seed = reader.int32();
                    break;
                case 6:
                    message.backend = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

litert.lm.proto.SamplerParameters.prototype.type = 0;
litert.lm.proto.SamplerParameters.prototype.k = 0;
litert.lm.proto.SamplerParameters.prototype.p = 0;
litert.lm.proto.SamplerParameters.prototype.temperature = 0;
litert.lm.proto.SamplerParameters.prototype.seed = 0;
litert.lm.proto.SamplerParameters.prototype.backend = 0;

litert.lm.proto.SamplerParameters.Type = {
    "TYPE_UNSPECIFIED": 0,
    "TOP_K": 1,
    "TOP_P": 2,
    "GREEDY": 3
};

litert.lm.proto.SamplerParameters.Backend = {
    "UNSPECIFIED": 0,
    "CPU": 1,
    "GPU": 2
};
