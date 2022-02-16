var $root = protobuf.get('nnabla');

$root.nnabla = {};

$root.nnabla.Communicator = class Communicator {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Communicator();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Context = class Context {

    constructor() {
        this.backends = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Context();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "backends":
                    message.backends.push(reader.string());
                    break;
                case "array_class":
                    message.array_class = reader.string();
                    break;
                case "device_id":
                    message.device_id = reader.string();
                    break;
                case "backend":
                    message.backend = reader.string();
                    break;
                case "compute_backend":
                    message.compute_backend = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Context.prototype.array_class = "";
$root.nnabla.Context.prototype.device_id = "";
$root.nnabla.Context.prototype.backend = ""; // Obsolete
$root.nnabla.Context.prototype.compute_backend = ""; // Obsolete

$root.nnabla.CosineSchedulerParameter = class CosineSchedulerParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CosineSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "max_iter":
                    message.max_iter = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CosineSchedulerParameter.prototype.max_iter = 0.0;

$root.nnabla.CustomSchedulerParameter = class CustomSchedulerParameter {

    constructor() {
        this.data_variable = [];
        this.output_variable = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.CustomSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "max_iter":
                    message.max_iter = reader.float();
                    break;
                case "network_name":
                    message.network_name = reader.string();
                    break;
                case "data_variable":
                    message.data_variable.push($root.nnabla.DataVariable.decodeText(reader));
                    break;
                case "output_variable":
                    message.output_variable.push($root.nnabla.OutputVariable.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CustomSchedulerParameter.prototype.max_iter = 0.0;
$root.nnabla.CustomSchedulerParameter.prototype.network_name = "";

$root.nnabla.Dataset = class Dataset {

    constructor() {
        this.variable = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Dataset();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "uri":
                    message.uri = reader.string();
                    break;
                case "batch_size":
                    message.batch_size = reader.int64();
                    break;
                case "cache_dir":
                    message.cache_dir = reader.string();
                    break;
                case "overwrite_cache":
                    message.overwrite_cache = reader.bool();
                    break;
                case "create_cache_explicitly":
                    message.create_cache_explicitly = reader.bool();
                    break;
                case "shuffle":
                    message.shuffle = reader.bool();
                    break;
                case "no_image_normalization":
                    message.no_image_normalization = reader.bool();
                    break;
                case "variable":
                    message.variable.push(reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Dataset.prototype.name = "";
$root.nnabla.Dataset.prototype.type = "";
$root.nnabla.Dataset.prototype.uri = "";
$root.nnabla.Dataset.prototype.batch_size = 0;
$root.nnabla.Dataset.prototype.cache_dir = "";
$root.nnabla.Dataset.prototype.overwrite_cache = false;
$root.nnabla.Dataset.prototype.create_cache_explicitly = false;
$root.nnabla.Dataset.prototype.shuffle = false;
$root.nnabla.Dataset.prototype.no_image_normalization = false;

$root.nnabla.DataVariable = class DataVariable {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DataVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "data_name":
                    message.data_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.DataVariable.prototype.variable_name = "";
$root.nnabla.DataVariable.prototype.data_name = "";

$root.nnabla.Executor = class Executor {

    constructor() {
        this.data_variable = [];
        this.generator_variable = [];
        this.loss_variable = [];
        this.output_variable = [];
        this.parameter_variable = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Executor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "network_name":
                    message.network_name = reader.string();
                    break;
                case "num_evaluations":
                    message.num_evaluations = reader.int64();
                    break;
                case "repeat_evaluation_type":
                    message.repeat_evaluation_type = reader.string();
                    break;
                case "need_back_propagation":
                    message.need_back_propagation = reader.bool();
                    break;
                case "data_variable":
                    message.data_variable.push($root.nnabla.DataVariable.decodeText(reader));
                    break;
                case "generator_variable":
                    message.generator_variable.push($root.nnabla.GeneratorVariable.decodeText(reader));
                    break;
                case "loss_variable":
                    message.loss_variable.push($root.nnabla.LossVariable.decodeText(reader));
                    break;
                case "output_variable":
                    message.output_variable.push($root.nnabla.OutputVariable.decodeText(reader));
                    break;
                case "parameter_variable":
                    message.parameter_variable.push($root.nnabla.ParameterVariable.decodeText(reader));
                    break;
                case "no_image_normalization":
                    message.no_image_normalization = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Executor.prototype.name = "";
$root.nnabla.Executor.prototype.network_name = "";
$root.nnabla.Executor.prototype.num_evaluations = 0;
$root.nnabla.Executor.prototype.repeat_evaluation_type = "";
$root.nnabla.Executor.prototype.need_back_propagation = false;
$root.nnabla.Executor.prototype.no_image_normalization = false;

$root.nnabla.ExponentialSchedulerParameter = class ExponentialSchedulerParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ExponentialSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "gamma":
                    message.gamma = reader.float();
                    break;
                case "iter_interval":
                    message.iter_interval = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ExponentialSchedulerParameter.prototype.gamma = 0.0;
$root.nnabla.ExponentialSchedulerParameter.prototype.iter_interval = 0;

$root.nnabla.Function = class Function {

    constructor() {
        this.repeat_id = [];
        this.input = [];
        this.output = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Function();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "repeat_id":
                    message.repeat_id.push(reader.string());
                    break;
                case "context":
                    message.context = $root.nnabla.Context.decodeText(reader);
                    break;
                case "input":
                    message.input.push(reader.string());
                    break;
                case "output":
                    message.output.push(reader.string());
                    break;
                // Function parameter messages - Start
                case "affine_param":
                    message.affine_param = $root.nnabla.AffineParameter.decodeText(reader);
                    break;
                case "rnn_param":
                    message.rnn_param = $root.nnabla.RNNParameter.decodeText(reader);
                    break;
                case "lstm_param":
                    message.lstm_param = $root.nnabla.LSTMParameter.decodeText(reader);
                    break;
                case "gru_param":
                    message.gru_param = $root.nnabla.GRUParameter.decodeText(reader);
                    break;
                case "convolution_param":
                    message.convolution_param = $root.nnabla.ConvolutionParameter.decodeText(reader);
                    break;
                case "fused_convolution_param":
                    message.fused_convolution_param = $root.nnabla.FusedConvolutionParameter.decodeText(reader);
                    break;
                case "depthwise_convolution_param":
                    message.depthwise_convolution_param = $root.nnabla.DepthwiseConvolutionParameter.decodeText(reader);
                    break;
                case "deconvolution_param":
                    message.deconvolution_param = $root.nnabla.DeconvolutionParameter.decodeText(reader);
                    break;
                case "depthwise_deconvolution_param":
                    message.depthwise_deconvolution_param = $root.nnabla.DepthwiseDeconvolutionParameter.decodeText(reader);
                    break;
                case "deformable_convolution_param":
                    message.deformable_convolution_param = $root.nnabla.DeformableConvolutionParameter.decodeText(reader);
                    break;
                case "adaptive_separable_convolution_param":
                    message.adaptive_separable_convolution_param = $root.nnabla.AdaptiveSeparableConvolutionParameter.decodeText(reader);
                    break;
                case "max_pooling_param":
                    message.max_pooling_param = $root.nnabla.MaxPoolingParameter.decodeText(reader);
                    break;
                case "average_pooling_param":
                    message.average_pooling_param = $root.nnabla.AveragePoolingParameter.decodeText(reader);
                    break;
                case "global_average_pooling_param":
                    message.global_average_pooling_param = $root.nnabla.GlobalAveragePoolingParameter.decodeText(reader);
                    break;
                case "sum_pooling_param":
                    message.sum_pooling_param = $root.nnabla.SumPoolingParameter.decodeText(reader);
                    break;
                case "unpooling_param":
                    message.unpooling_param = $root.nnabla.UnpoolingParameter.decodeText(reader);
                    break;
                case "embed_param":
                    message.embed_param = $root.nnabla.EmbedParameter.decodeText(reader);
                    break;
                case "roi_align_param":
                    message.roi_align_param = $root.nnabla.RoiAlignParameter.decodeText(reader);
                    break;
                case "sigmoid_param":
                    message.sigmoid_param = $root.nnabla.SigmoidParameter.decodeText(reader);
                    break;
                case "swish_param":
                    message.swish_param = $root.nnabla.SwishParameter.decodeText(reader);
                    break;
                case "tanh_param":
                    message.tanh_param = $root.nnabla.TanhParameter.decodeText(reader);
                    break;
                case "relu_param":
                    message.relu_param = $root.nnabla.ReLUParameter.decodeText(reader);
                    break;
                case "leaky_relu_param":
                    message.leaky_relu_param = $root.nnabla.LeakyReLUParameter.decodeText(reader);
                    break;
                case "softmax_param":
                    message.softmax_param = $root.nnabla.SoftmaxParameter.decodeText(reader);
                    break;
                case "log_softmax_param":
                    message.log_softmax_param = $root.nnabla.LogSoftmaxParameter.decodeText(reader);
                    break;
                case "elu_param":
                    message.elu_param = $root.nnabla.ELUParameter.decodeText(reader);
                    break;
                case "selu_param":
                    message.selu_param = $root.nnabla.SELUParameter.decodeText(reader);
                    break;
                case "crelu_param":
                    message.crelu_param = $root.nnabla.CReLUParameter.decodeText(reader);
                    break;
                case "celu_param":
                    message.celu_param = $root.nnabla.CELUParameter.decodeText(reader);
                    break;
                case "prelu_param":
                    message.prelu_param = $root.nnabla.PReLUParameter.decodeText(reader);
                    break;
                case "gelu_param":
                    message.gelu_param = $root.nnabla.GELUParameter.decodeText(reader);
                    break;
                case "mish_param":
                    message.mish_param = $root.nnabla.MishParameter.decodeText(reader);
                    break;
                case "relu6_param":
                    message.relu6_param = $root.nnabla.ReLU6Parameter.decodeText(reader);
                    break;
                case "hard_sigmoid_param":
                    message.hard_sigmoid_param = $root.nnabla.HardSigmoidParameter.decodeText(reader);
                    break;
                case "hard_tanh_param":
                    message.hard_tanh_param = $root.nnabla.HardTanhParameter.decodeText(reader);
                    break;
                case "log_sigmoid_param":
                    message.log_sigmoid_param = $root.nnabla.LogSigmoidParameter.decodeText(reader);
                    break;
                case "softplus_param":
                    message.softplus_param = $root.nnabla.SoftPlusParameter.decodeText(reader);
                    break;
                case "softsign_param":
                    message.softsign_param = $root.nnabla.SoftSignParameter.decodeText(reader);
                    break;
                case "tanh_shrink_param":
                    message.tanh_shrink_param = $root.nnabla.TanhShrinkParameter.decodeText(reader);
                    break;
                case "sinc_param":
                    message.sinc_param = $root.nnabla.SincParameter.decodeText(reader);
                    break;
                case "fused_batch_normalization_param":
                    message.fused_batch_normalization_param = $root.nnabla.FusedBatchNormalizationParameter.decodeText(reader);
                    break;
                case "batch_normalization_param":
                    message.batch_normalization_param = $root.nnabla.BatchNormalizationParameter.decodeText(reader);
                    break;
                case "group_normalization_param":
                    message.group_normalization_param = $root.nnabla.GroupNormalizationParameter.decodeText(reader);
                    break;
                case "instance_normalization_param":
                    message.instance_normalization_param = $root.nnabla.InstanceNormalizationParameter.decodeText(reader);
                    break;
                case "layer_normalization_param":
                    message.layer_normalization_param = $root.nnabla.LayerNormalizationParameter.decodeText(reader);
                    break;
                case "norm_normalization_param":
                    message.norm_normalization_param = $root.nnabla.NormNormalizationParameter.decodeText(reader);
                    break;
                case "sync_batch_normalization_param":
                    message.sync_batch_normalization_param = $root.nnabla.SyncBatchNormalizationParameter.decodeText(reader);
                    break;
                case "tensor_normalization_param":
                    message.tensor_normalization_param = $root.nnabla.TensorNormalizationParameter.decodeText(reader);
                    break;
                case "weight_normalization_param":
                    message.weight_normalization_param = $root.nnabla.WeightNormalizationParameter.decodeText(reader);
                    break;
                case "weight_standardization_param":
                    message.weight_standardization_param = $root.nnabla.WeightStandardizationParameter.decodeText(reader);
                    break;
                case "spectral_norm_param":
                    message.spectral_norm_param = $root.nnabla.SpectralNormParameter.decodeText(reader);
                    break;
                case "mean_subtraction_param":
                    message.mean_subtraction_param = $root.nnabla.MeanSubtractionParameter.decodeText(reader);
                    break;
                case "clip_grad_by_value_param":
                    message.clip_grad_by_value_param = $root.nnabla.ClipGradByValueParameter.decodeText(reader);
                    break;
                case "clip_grad_by_norm_param":
                    message.clip_grad_by_norm_param = $root.nnabla.ClipGradByNormParameter.decodeText(reader);
                    break;
                case "sum_param":
                    message.sum_param = $root.nnabla.SumParameter.decodeText(reader);
                    break;
                case "cumsum_param":
                    message.cumsum_param = $root.nnabla.CumSumParameter.decodeText(reader);
                    break;
                case "mean_param":
                    message.mean_param = $root.nnabla.MeanParameter.decodeText(reader);
                    break;
                case "max_param":
                    message.max_param = $root.nnabla.MaxParameter.decodeText(reader);
                    break;
                case "min_param":
                    message.min_param = $root.nnabla.MinParameter.decodeText(reader);
                    break;
                case "norm_param":
                    message.norm_param = $root.nnabla.NormParameter.decodeText(reader);
                    break;
                case "prod_param":
                    message.prod_param = $root.nnabla.ProdParameter.decodeText(reader);
                    break;
                case "cumprod_param":
                    message.cumprod_param = $root.nnabla.CumProdParameter.decodeText(reader);
                    break;
                case "reduce_sum_param":
                    message.reduce_sum_param = $root.nnabla.ReduceSumParameter.decodeText(reader);
                    break;
                case "reduce_mean_param":
                    message.reduce_mean_param = $root.nnabla.ReduceMeanParameter.decodeText(reader);
                    break;
                case "add2_param":
                    message.add2_param = $root.nnabla.Add2Parameter.decodeText(reader);
                    break;
                case "add_n_param":
                    message.add_n_param = $root.nnabla.AddNParameter.decodeText(reader);
                    break;
                case "bc_add2_param":
                    message.bc_add2_param = $root.nnabla.BcAdd2Parameter.decodeText(reader);
                    break;
                case "sub2_param":
                    message.sub2_param = $root.nnabla.Sub2Parameter.decodeText(reader);
                    break;
                case "mul2_param":
                    message.mul2_param = $root.nnabla.Mul2Parameter.decodeText(reader);
                    break;
                case "mul_n_param":
                    message.mul_n_param = $root.nnabla.MulNParameter.decodeText(reader);
                    break;
                case "div2_param":
                    message.div2_param = $root.nnabla.Div2Parameter.decodeText(reader);
                    break;
                case "pow2_param":
                    message.pow2_param = $root.nnabla.Pow2Parameter.decodeText(reader);
                    break;
                case "add_scalar_param":
                    message.add_scalar_param = $root.nnabla.AddScalarParameter.decodeText(reader);
                    break;
                case "mul_scalar_param":
                    message.mul_scalar_param = $root.nnabla.MulScalarParameter.decodeText(reader);
                    break;
                case "pow_scalar_param":
                    message.pow_scalar_param = $root.nnabla.PowScalarParameter.decodeText(reader);
                    break;
                case "r_sub_scalar_param":
                    message.r_sub_scalar_param = $root.nnabla.RSubScalarParameter.decodeText(reader);
                    break;
                case "r_div_scalar_param":
                    message.r_div_scalar_param = $root.nnabla.RDivScalarParameter.decodeText(reader);
                    break;
                case "r_pow_scalar_param":
                    message.r_pow_scalar_param = $root.nnabla.RPowScalarParameter.decodeText(reader);
                    break;
                case "sign_param":
                    message.sign_param = $root.nnabla.SignParameter.decodeText(reader);
                    break;
                case "minimum2_param":
                    message.minimum2_param = $root.nnabla.Minimum2Parameter.decodeText(reader);
                    break;
                case "maximum2_param":
                    message.maximum2_param = $root.nnabla.Maximum2Parameter.decodeText(reader);
                    break;
                case "minimum_scalar_param":
                    message.minimum_scalar_param = $root.nnabla.MinimumScalarParameter.decodeText(reader);
                    break;
                case "maximum_scalar_param":
                    message.maximum_scalar_param = $root.nnabla.MaximumScalarParameter.decodeText(reader);
                    break;
                case "logical_and_param":
                    message.logical_and_param = $root.nnabla.LogicalAndParameter.decodeText(reader);
                    break;
                case "logical_or_param":
                    message.logical_or_param = $root.nnabla.LogicalOrParameter.decodeText(reader);
                    break;
                case "logical_xor_param":
                    message.logical_xor_param = $root.nnabla.LogicalXorParameter.decodeText(reader);
                    break;
                case "equal_param":
                    message.equal_param = $root.nnabla.EqualParameter.decodeText(reader);
                    break;
                case "not_equal_param":
                    message.not_equal_param = $root.nnabla.NotEqualParameter.decodeText(reader);
                    break;
                case "greater_equal_param":
                    message.greater_equal_param = $root.nnabla.GreaterEqualParameter.decodeText(reader);
                    break;
                case "greater_param":
                    message.greater_param = $root.nnabla.GreaterParameter.decodeText(reader);
                    break;
                case "less_equal_param":
                    message.less_equal_param = $root.nnabla.LessEqualParameter.decodeText(reader);
                    break;
                case "less_param":
                    message.less_param = $root.nnabla.LessParameter.decodeText(reader);
                    break;
                case "searchsorted_param":
                    message.searchsorted_param = $root.nnabla.SearchSortedParameter.decodeText(reader);
                    break;
                case "logical_and_scalar_param":
                    message.logical_and_scalar_param = $root.nnabla.LogicalAndScalarParameter.decodeText(reader);
                    break;
                case "logical_or_scalar_param":
                    message.logical_or_scalar_param = $root.nnabla.LogicalOrScalarParameter.decodeText(reader);
                    break;
                case "logical_xor_scalar_param":
                    message.logical_xor_scalar_param = $root.nnabla.LogicalXorScalarParameter.decodeText(reader);
                    break;
                case "equal_scalar_param":
                    message.equal_scalar_param = $root.nnabla.EqualScalarParameter.decodeText(reader);
                    break;
                case "not_equal_scalar_param":
                    message.not_equal_scalar_param = $root.nnabla.NotEqualScalarParameter.decodeText(reader);
                    break;
                case "greater_equal_scalar_param":
                    message.greater_equal_scalar_param = $root.nnabla.GreaterEqualScalarParameter.decodeText(reader);
                    break;
                case "greater_scalar_param":
                    message.greater_scalar_param = $root.nnabla.GreaterScalarParameter.decodeText(reader);
                    break;
                case "less_equal_scalar_param":
                    message.less_equal_scalar_param = $root.nnabla.LessEqualScalarParameter.decodeText(reader);
                    break;
                case "less_scalar_param":
                    message.less_scalar_param = $root.nnabla.LessScalarParameter.decodeText(reader);
                    break;
                case "logical_not_param":
                    message.logical_not_param = $root.nnabla.LogicalNotParameter.decodeText(reader);
                    break;
                case "isnan_param":
                    message.isnan_param = $root.nnabla.IsNaNParameter.decodeText(reader);
                    break;
                case "isinf_param":
                    message.isinf_param = $root.nnabla.IsInfParameter.decodeText(reader);
                    break;
                case "reset_nan_param":
                    message.reset_nan_param = $root.nnabla.ResetNaNParameter.decodeText(reader);
                    break;
                case "reset_inf_param":
                    message.reset_inf_param = $root.nnabla.ResetInfParameter.decodeText(reader);
                    break;
                case "where_param":
                    message.where_param = $root.nnabla.WhereParameter.decodeText(reader);
                    break;
                case "constant_param":
                    message.constant_param = $root.nnabla.ConstantParameter.decodeText(reader);
                    break;
                case "arange_param":
                    message.arange_param = $root.nnabla.ArangeParameter.decodeText(reader);
                    break;
                case "linspace_param":
                    message.linspace_param = $root.nnabla.LinspaceParameter.decodeText(reader);
                    break;
                case "abs_param":
                    message.abs_param = $root.nnabla.AbsParameter.decodeText(reader);
                    break;
                case "exp_param":
                    message.exp_param = $root.nnabla.ExpParameter.decodeText(reader);
                    break;
                case "log_param":
                    message.log_param = $root.nnabla.LogParameter.decodeText(reader);
                    break;
                case "identity_param":
                    message.identity_param = $root.nnabla.IdentityParameter.decodeText(reader);
                    break;
                case "batch_matmul_param":
                    message.batch_matmul_param = $root.nnabla.BatchMatmulParameter.decodeText(reader);
                    break;
                case "round_param":
                    message.round_param = $root.nnabla.RoundParameter.decodeText(reader);
                    break;
                case "ceil_param":
                    message.ceil_param = $root.nnabla.CeilParameter.decodeText(reader);
                    break;
                case "floor_param":
                    message.floor_param = $root.nnabla.FloorParameter.decodeText(reader);
                    break;
                case "sin_param":
                    message.sin_param = $root.nnabla.SinParameter.decodeText(reader);
                    break;
                case "cos_param":
                    message.cos_param = $root.nnabla.CosParameter.decodeText(reader);
                    break;
                case "tan_param":
                    message.tan_param = $root.nnabla.TanParameter.decodeText(reader);
                    break;
                case "sinh_param":
                    message.sinh_param = $root.nnabla.SinhParameter.decodeText(reader);
                    break;
                case "cosh_param":
                    message.cosh_param = $root.nnabla.CoshParameter.decodeText(reader);
                    break;
                case "asin_param":
                    message.asin_param = $root.nnabla.ASinParameter.decodeText(reader);
                    break;
                case "acos_param":
                    message.acos_param = $root.nnabla.ACosParameter.decodeText(reader);
                    break;
                case "atan_param":
                    message.atan_param = $root.nnabla.ATanParameter.decodeText(reader);
                    break;
                case "atan2_param":
                    message.atan2_param = $root.nnabla.ATan2Parameter.decodeText(reader);
                    break;
                case "asinh_param":
                    message.asinh_param = $root.nnabla.ASinhParameter.decodeText(reader);
                    break;
                case "acosh_param":
                    message.acosh_param = $root.nnabla.ACoshParameter.decodeText(reader);
                    break;
                case "atanh_param":
                    message.atanh_param = $root.nnabla.ATanhParameter.decodeText(reader);
                    break;
                case "concatenate_param":
                    message.concatenate_param = $root.nnabla.ConcatenateParameter.decodeText(reader);
                    break;
                case "split_param":
                    message.split_param = $root.nnabla.SplitParameter.decodeText(reader);
                    break;
                case "stack_param":
                    message.stack_param = $root.nnabla.StackParameter.decodeText(reader);
                    break;
                case "slice_param":
                    message.slice_param = $root.nnabla.SliceParameter.decodeText(reader);
                    break;
                case "pad_param":
                    message.pad_param = $root.nnabla.PadParameter.decodeText(reader);
                    break;
                case "transpose_param":
                    message.transpose_param = $root.nnabla.TransposeParameter.decodeText(reader);
                    break;
                case "broadcast_param":
                    message.broadcast_param = $root.nnabla.BroadcastParameter.decodeText(reader);
                    break;
                case "broadcast_to_param":
                    message.broadcast_to_param = $root.nnabla.BroadcastToParameter.decodeText(reader);
                    break;
                case "tile_param":
                    message.tile_param = $root.nnabla.TileParameter.decodeText(reader);
                    break;
                case "one_hot_param":
                    message.one_hot_param = $root.nnabla.OneHotParameter.decodeText(reader);
                    break;
                case "flip_param":
                    message.flip_param = $root.nnabla.FlipParameter.decodeText(reader);
                    break;
                case "shift_param":
                    message.shift_param = $root.nnabla.ShiftParameter.decodeText(reader);
                    break;
                case "sort_param":
                    message.sort_param = $root.nnabla.SortParameter.decodeText(reader);
                    break;
                case "reshape_param":
                    message.reshape_param = $root.nnabla.ReshapeParameter.decodeText(reader);
                    break;
                case "matrix_diag_param":
                    message.matrix_diag_param = $root.nnabla.MatrixDiagParameter.decodeText(reader);
                    break;
                case "matrix_diag_part_param":
                    message.matrix_diag_part_param = $root.nnabla.MatrixDiagPartParameter.decodeText(reader);
                    break;
                case "meshgrid_param":
                    message.meshgrid_param = $root.nnabla.MeshgridParameter.decodeText(reader);
                    break;
                case "batch_det_param":
                    message.batch_det_param = $root.nnabla.BatchDetParameter.decodeText(reader);
                    break;
                case "batch_inv_param":
                    message.batch_inv_param = $root.nnabla.BatchInvParameter.decodeText(reader);
                    break;
                case "batch_logdet_param":
                    message.batch_logdet_param = $root.nnabla.BatchLogdetParameter.decodeText(reader);
                    break;
                case "assign_param":
                    message.assign_param = $root.nnabla.AssignParameter.decodeText(reader);
                    break;
                case "gather_param":
                    message.gather_param = $root.nnabla.GatherParameter.decodeText(reader);
                    break;
                case "gather_nd_param":
                    message.gather_nd_param = $root.nnabla.GatherNdParameter.decodeText(reader);
                    break;
                case "bool_gather_param":
                    message.bool_gather_param = $root.nnabla.BoolGatherParameter.decodeText(reader);
                    break;
                case "scatter_nd_param":
                    message.scatter_nd_param = $root.nnabla.ScatterNdParameter.decodeText(reader);
                    break;
                case "scatter_add_param":
                    message.scatter_add_param = $root.nnabla.ScatterAddParameter.decodeText(reader);
                    break;
                case "bool_scatter_param":
                    message.bool_scatter_param = $root.nnabla.BoolScatterParameter.decodeText(reader);
                    break;
                case "bool_fill_param":
                    message.bool_fill_param = $root.nnabla.BoolFillParameter.decodeText(reader);
                    break;
                case "pack_padded_sequence_param":
                    message.pack_padded_sequence_param = $root.nnabla.PackPaddedSequenceParameter.decodeText(reader);
                    break;
                case "pad_packed_sequence_param":
                    message.pad_packed_sequence_param = $root.nnabla.PadPackedSequenceParameter.decodeText(reader);
                    break;
                case "interpolate_param":
                    message.interpolate_param = $root.nnabla.InterpolateParameter.decodeText(reader);
                    break;
                case "fft_param":
                    message.fft_param = $root.nnabla.FFTParameter.decodeText(reader);
                    break;
                case "ifft_param":
                    message.ifft_param = $root.nnabla.IFFTParameter.decodeText(reader);
                    break;
                case "stft_param":
                    message.stft_param = $root.nnabla.STFTParameter.decodeText(reader);
                    break;
                case "istft_param":
                    message.istft_param = $root.nnabla.ISTFTParameter.decodeText(reader);
                    break;
                case "dropout_param":
                    message.dropout_param = $root.nnabla.DropoutParameter.decodeText(reader);
                    break;
                case "top_k_data_param":
                    message.top_k_data_param = $root.nnabla.TopKDataParameter.decodeText(reader);
                    break;
                case "top_k_grad_param":
                    message.top_k_grad_param = $root.nnabla.TopKGradParameter.decodeText(reader);
                    break;
                case "rand_param":
                    message.rand_param = $root.nnabla.RandParameter.decodeText(reader);
                    break;
                case "randint_param":
                    message.randint_param = $root.nnabla.RandintParameter.decodeText(reader);
                    break;
                case "randn_param":
                    message.randn_param = $root.nnabla.RandnParameter.decodeText(reader);
                    break;
                case "rand_binomial_param":
                    message.rand_binomial_param = $root.nnabla.RandBinomialParameter.decodeText(reader);
                    break;
                case "rand_beta_param":
                    message.rand_beta_param = $root.nnabla.RandBetaParameter.decodeText(reader);
                    break;
                case "rand_gamma_param":
                    message.rand_gamma_param = $root.nnabla.RandGammaParameter.decodeText(reader);
                    break;
                case "random_choice_param":
                    message.random_choice_param = $root.nnabla.RandomChoiceParameter.decodeText(reader);
                    break;
                case "random_crop_param":
                    message.random_crop_param = $root.nnabla.RandomCropParameter.decodeText(reader);
                    break;
                case "random_flip_param":
                    message.random_flip_param = $root.nnabla.RandomFlipParameter.decodeText(reader);
                    break;
                case "random_shift_param":
                    message.random_shift_param = $root.nnabla.RandomShiftParameter.decodeText(reader);
                    break;
                case "random_erase_param":
                    message.random_erase_param = $root.nnabla.RandomEraseParameter.decodeText(reader);
                    break;
                case "image_augmentation_param":
                    message.image_augmentation_param = $root.nnabla.ImageAugmentationParameter.decodeText(reader);
                    break;
                case "sigmoid_cross_entropy_param":
                    message.sigmoid_cross_entropy_param = $root.nnabla.SigmoidCrossEntropyParameter.decodeText(reader);
                    break;
                case "binary_cross_entropy_param":
                    message.binary_cross_entropy_param = $root.nnabla.BinaryCrossEntropyParameter.decodeText(reader);
                    break;
                case "softmax_cross_entropy_param":
                    message.softmax_cross_entropy_param = $root.nnabla.SoftmaxCrossEntropyParameter.decodeText(reader);
                    break;
                case "categorical_cross_entropy_param":
                    message.categorical_cross_entropy_param = $root.nnabla.CategoricalCrossEntropyParameter.decodeText(reader);
                    break;
                case "squared_error_param":
                    message.squared_error_param = $root.nnabla.SquaredErrorParameter.decodeText(reader);
                    break;
                case "absolute_error_param":
                    message.absolute_error_param = $root.nnabla.AbsoluteErrorParameter.decodeText(reader);
                    break;
                case "huber_loss_param":
                    message.huber_loss_param = $root.nnabla.HuberLossParameter.decodeText(reader);
                    break;
                case "epsilon_insensitive_loss_param":
                    message.epsilon_insensitive_loss_param = $root.nnabla.EpsilonInsensitiveLossParameter.decodeText(reader);
                    break;
                case "kl_multinomial_param":
                    message.kl_multinomial_param = $root.nnabla.KLMultinomialParameter.decodeText(reader);
                    break;
                case "affine_grid_param":
                    message.affine_grid_param = $root.nnabla.AffineGridParameter.decodeText(reader);
                    break;
                case "warp_by_grid_param":
                    message.warp_by_grid_param = $root.nnabla.WarpByGridParameter.decodeText(reader);
                    break;
                case "warp_by_flow_param":
                    message.warp_by_flow_param = $root.nnabla.WarpByFlowParameter.decodeText(reader);
                    break;
                case "binary_sigmoid_param":
                    message.binary_sigmoid_param = $root.nnabla.BinarySigmoidParameter.decodeText(reader);
                    break;
                case "binary_tanh_param":
                    message.binary_tanh_param = $root.nnabla.BinaryTanhParameter.decodeText(reader);
                    break;
                case "binary_connect_affine_param":
                    message.binary_connect_affine_param = $root.nnabla.BinaryConnectAffineParameter.decodeText(reader);
                    break;
                case "binary_connect_convolution_param":
                    message.binary_connect_convolution_param = $root.nnabla.BinaryConnectConvolutionParameter.decodeText(reader);
                    break;
                case "binary_weight_affine_param":
                    message.binary_weight_affine_param = $root.nnabla.BinaryWeightAffineParameter.decodeText(reader);
                    break;
                case "binary_weight_convolution_param":
                    message.binary_weight_convolution_param = $root.nnabla.BinaryWeightConvolutionParameter.decodeText(reader);
                    break;
                case "inq_affine_param":
                    message.inq_affine_param = $root.nnabla.INQAffineParameter.decodeText(reader);
                    break;
                case "inq_convolution_param":
                    message.inq_convolution_param = $root.nnabla.INQConvolutionParameter.decodeText(reader);
                    break;
                case "fixed_point_quantize_param":
                    message.fixed_point_quantize_param = $root.nnabla.FixedPointQuantizeParameter.decodeText(reader);
                    break;
                case "min_max_quantize_param":
                    message.min_max_quantize_param = $root.nnabla.MinMaxQuantizeParameter.decodeText(reader);
                    break;
                case "pow2_quantize_param":
                    message.pow2_quantize_param = $root.nnabla.Pow2QuantizeParameter.decodeText(reader);
                    break;
                case "prune_param":
                    message.prune_param = $root.nnabla.PruneParameter.decodeText(reader);
                    break;
                case "quantize_linear_param":
                    message.quantize_linear_param = $root.nnabla.QuantizeLinearParameter.decodeText(reader);
                    break;
                case "dequantize_linear_param":
                    message.dequantize_linear_param = $root.nnabla.DequantizeLinearParameter.decodeText(reader);
                    break;
                case "top_n_error_param":
                    message.top_n_error_param = $root.nnabla.TopNErrorParameter.decodeText(reader);
                    break;
                case "binary_error_param":
                    message.binary_error_param = $root.nnabla.BinaryErrorParameter.decodeText(reader);
                    break;
                case "confusion_matrix_param":
                    message.confusion_matrix_param = $root.nnabla.ConfusionMatrixParameter.decodeText(reader);
                    break;
                case "vat_noise_param":
                    message.vat_noise_param = $root.nnabla.VATNoiseParameter.decodeText(reader);
                    break;
                case "unlink_param":
                    message.unlink_param = $root.nnabla.UnlinkParameter.decodeText(reader);
                    break;
                case "sink_param":
                    message.sink_param = $root.nnabla.SinkParameter.decodeText(reader);
                    break;
                case "nms_detection2d_param":
                    message.nms_detection2d_param = $root.nnabla.NmsDetection2dParameter.decodeText(reader);
                    break;
                case "max_pooling_backward_param":
                    message.max_pooling_backward_param = $root.nnabla.MaxPoolingBackwardParameter.decodeText(reader);
                    break;
                case "patch_correlation_param":
                    message.patch_correlation_param = $root.nnabla.PatchCorrelationParameter.decodeText(reader);
                    break;
                // Function parameter messages - End
                case "repeat_param":
                    message.repeat_param = $root.nnabla.RepeatParameter.decodeText(reader);
                    break;
                case "recurrent_param":
                    message.recurrent_param = $root.nnabla.RecurrentParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Function.prototype.name = "";
$root.nnabla.Function.prototype.type = "";
$root.nnabla.Function.prototype.repeat_param = null;
$root.nnabla.Function.prototype.recurrent_param = null;

$root.nnabla.GeneratorVariable = class GeneratorVariable {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GeneratorVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "multiplier":
                    message.multiplier = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GeneratorVariable.prototype.variable_name = "";
$root.nnabla.GeneratorVariable.prototype.type = "";
$root.nnabla.GeneratorVariable.prototype.multiplier = 0.0;

$root.nnabla.GlobalConfig = class GlobalConfig {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GlobalConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "default_context":
                    message.default_context = $root.nnabla.Context.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GlobalConfig.prototype.default_context = null;

$root.nnabla.Initializer = class Initializer {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Initializer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "multiplier":
                    message.multiplier = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Initializer.prototype.type = "";
$root.nnabla.Initializer.prototype.multiplier = 0.0;

$root.nnabla.LinearWarmupSchedulerParameter = class LinearWarmupSchedulerParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LinearWarmupSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "warmup_iter":
                    message.warmup_iter = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LinearWarmupSchedulerParameter.prototype.warmup_iter = 0;

$root.nnabla.LossVariable = class LossVariable {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LossVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LossVariable.prototype.variable_name = "";

$root.nnabla.ModelProto = class ModelProto {

    constructor() {
        this.network = [];
        this.parameter = [];
        this.dataset = [];
        this.optimizer = [];
        this.monitor = [];
        this.executor = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ModelProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 200:
                    message.parameter.push($root.nnabla.Parameter.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ModelProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.string();
                    break;
                case "global_config":
                    message.global_config = $root.nnabla.GlobalConfig.decodeText(reader);
                    break;
                case "training_config":
                    message.training_config = $root.nnabla.TrainingConfig.decodeText(reader);
                    break;
                case "network":
                    message.network.push($root.nnabla.Network.decodeText(reader));
                    break;
                case "dataset":
                    message.dataset.push($root.nnabla.Dataset.decodeText(reader));
                    break;
                case "optimizer":
                    message.optimizer.push($root.nnabla.Optimizer.decodeText(reader));
                    break;
                case "monitor":
                    message.monitor.push($root.nnabla.Monitor.decodeText(reader));
                    break;
                case "executor":
                    message.executor.push($root.nnabla.Executor.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ModelProto.prototype.version = "";
$root.nnabla.ModelProto.prototype.global_config = null;
$root.nnabla.ModelProto.prototype.training_config = null;

$root.nnabla.Monitor = class Monitor {

    constructor() {
        this.dataset_name = [];
        this.data_variable = [];
        this.generator_variable = [];
        this.monitor_variable = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Monitor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "network_name":
                    message.network_name = reader.string();
                    break;
                case "dataset_name":
                    message.dataset_name.push(reader.string());
                    break;
                case "data_variable":
                    message.data_variable.push($root.nnabla.DataVariable.decodeText(reader));
                    break;
                case "generator_variable":
                    message.generator_variable.push($root.nnabla.GeneratorVariable.decodeText(reader));
                    break;
                case "monitor_variable":
                    message.monitor_variable.push($root.nnabla.MonitorVariable.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MonitorVariable = class MonitorVariable {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MonitorVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "data_name":
                    message.data_name = reader.string();
                    break;
                case "multiplier":
                    message.multiplier = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MonitorVariable.prototype.variable_name = "";
$root.nnabla.MonitorVariable.prototype.type = "";
$root.nnabla.MonitorVariable.prototype.data_name = "";
$root.nnabla.MonitorVariable.prototype.multiplier = 0.0;

$root.nnabla.Network = class Network {

    constructor() {
        this.repeat_info = [];
        this.variable = [];
        this.function = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Network();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "batch_size":
                    message.batch_size = reader.int64();
                    break;
                case "repeat_info":
                    message.repeat_info.push($root.nnabla.RepeatInfo.decodeText(reader));
                    break;
                case "variable":
                    message.variable.push($root.nnabla.Variable.decodeText(reader));
                    break;
                case "function":
                    message.function.push($root.nnabla.Function.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Network.prototype.batch_size = 1;
$root.nnabla.Network.prototype.name = "";

$root.nnabla.Optimizer = class Optimizer {

    constructor() {
        this.dataset_name = [];
        this.data_variable = [];
        this.generator_variable = [];
        this.loss_variable = [];
        this.parameter_variable = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Optimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "order":
                    message.order = reader.int64();
                    break;
                case "network_name":
                    message.network_name = reader.string();
                    break;
                case "dataset_name":
                    message.dataset_name.push(reader.string());
                    break;
                case "solver":
                    message.solver = $root.nnabla.Solver.decodeText(reader);
                    break;
                case "update_interval":
                    message.update_interval = reader.int64();
                    break;
                case "data_variable":
                    message.data_variable.push($root.nnabla.DataVariable.decodeText(reader));
                    break;
                case "generator_variable":
                    message.generator_variable.push($root.nnabla.GeneratorVariable.decodeText(reader));
                    break;
                case "loss_variable":
                    message.loss_variable.push($root.nnabla.LossVariable.decodeText(reader));
                    break;
                case "parameter_variable":
                    message.parameter_variable.push($root.nnabla.ParameterVariable.decodeText(reader));
                    break;
                case "start_iter":
                    message.start_iter = reader.int64();
                    break;
                case "end_iter":
                    message.end_iter = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Optimizer.prototype.name = "";
$root.nnabla.Optimizer.prototype.order = 0;
$root.nnabla.Optimizer.prototype.network_name = "";
$root.nnabla.Optimizer.prototype.solver = null;
$root.nnabla.Optimizer.prototype.update_interval = 0;
$root.nnabla.Optimizer.prototype.start_iter = 0;
$root.nnabla.Optimizer.prototype.end_iter = 0;

$root.nnabla.OutputVariable = class OutputVariable {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.OutputVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "data_name":
                    message.data_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.OutputVariable.prototype.variable_name = "";
$root.nnabla.OutputVariable.prototype.type = "";
$root.nnabla.OutputVariable.prototype.data_name = "";

$root.nnabla.Parameter = class Parameter {

    constructor() {
        this.data = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 20:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 100:
                    reader.array(message.data, () => reader.float(), tag);
                    break;
                case 101:
                    message.need_grad = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Parameter.prototype.variable_name = "";
$root.nnabla.Parameter.prototype.shape = null;
$root.nnabla.Parameter.prototype.need_grad = false;

$root.nnabla.ParameterVariable = class ParameterVariable {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ParameterVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "learning_rate_multiplier":
                    message.learning_rate_multiplier = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ParameterVariable.prototype.variable_name = "";
$root.nnabla.ParameterVariable.prototype.learning_rate_multiplier = "";

$root.nnabla.PolynomialSchedulerParameter = class PolynomialSchedulerParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PolynomialSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "max_iter":
                    message.max_iter = reader.float();
                    break;
                case "power":
                    message.power = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PolynomialSchedulerParameter.prototype.max_iter = 0.0;
$root.nnabla.PolynomialSchedulerParameter.prototype.power = 0.0;

$root.nnabla.RecurrentParameter = class RecurrentParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RecurrentParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "repeat_id":
                    message.repeat_id = reader.string();
                    break;
                case "length":
                    message.length = reader.int64();
                    break;
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RecurrentParameter.prototype.repeat_id = "";
$root.nnabla.RecurrentParameter.prototype.length = 0;
$root.nnabla.RecurrentParameter.prototype.axis = 0;

$root.nnabla.RepeatInfo = class RepeatInfo {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RepeatInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "id":
                    message.id = reader.string();
                    break;
                case "times":
                    message.times = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RepeatInfo.prototype.id = "";
$root.nnabla.RepeatInfo.prototype.times = 0;

$root.nnabla.RepeatParameter = class RepeatParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RepeatParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "repeat_id":
                    message.repeat_id = reader.string();
                    break;
                case "times":
                    message.times = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RepeatParameter.prototype.repeat_id = "";
$root.nnabla.RepeatParameter.prototype.times = 0;

$root.nnabla.Shape = class Shape {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Shape();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.array(message.dim, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Shape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Solver = class Solver {

    constructor() {
        this.states = {};
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Solver();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "context":
                    message.context = $root.nnabla.Context.decodeText(reader);
                    break;
                case "weight_decay":
                    message.weight_decay = reader.float();
                    break;
                case "states":
                    reader.entry(message.states, () => reader.string(), () => $root.nnabla.SolverState.decodeText(reader));
                    break;
                case "lr_scheduler_type":
                    message.lr_scheduler_type = reader.string();
                    break;
                case "polynomial_scheduler_param":
                    message.polynomial_scheduler_param = $root.nnabla.PolynomialSchedulerParameter.decodeText(reader);
                    break;
                case "cosine_scheduler_param":
                    message.cosine_scheduler_param = $root.nnabla.CosineSchedulerParameter.decodeText(reader);
                    break;
                case "exponential_scheduler_param":
                    message.exponential_scheduler_param = $root.nnabla.ExponentialSchedulerParameter.decodeText(reader);
                    break;
                case "step_scheduler_param":
                    message.step_scheduler_param = $root.nnabla.StepSchedulerParameter.decodeText(reader);
                    break;
                case "custom_scheduler_param":
                    message.custom_scheduler_param = $root.nnabla.CustomSchedulerParameter.decodeText(reader);
                    break;
                case "lr_warmup_scheduler_type":
                    message.lr_warmup_scheduler_type = reader.string();
                    break;
                case "linear_warmup_scheduler_param":
                    message.linear_warmup_scheduler_param = $root.nnabla.LinearWarmupSchedulerParameter.decodeText(reader);
                    break;
                case "lr_decay":
                    message.lr_decay = reader.float();
                    break;
                case "lr_decay_interval":
                    message.lr_decay_interval = reader.int64();
                    break;
                // Solver parameter messages - Start
                case "sgd_param":
                    message.sgd_param = $root.nnabla.SgdParameter.decodeText(reader);
                    break;
                case "sgdw_param":
                    message.sgdw_param = $root.nnabla.SgdWParameter.decodeText(reader);
                    break;
                case "momentum_param":
                    message.momentum_param = $root.nnabla.MomentumParameter.decodeText(reader);
                    break;
                case "lars_param":
                    message.lars_param = $root.nnabla.LarsParameter.decodeText(reader);
                    break;
                case "nesterov_param":
                    message.nesterov_param = $root.nnabla.NesterovParameter.decodeText(reader);
                    break;
                case "adadelta_param":
                    message.adadelta_param = $root.nnabla.AdadeltaParameter.decodeText(reader);
                    break;
                case "adagrad_param":
                    message.adagrad_param = $root.nnabla.AdagradParameter.decodeText(reader);
                    break;
                case "adabelief_param":
                    message.adabelief_param = $root.nnabla.AdaBeliefParameter.decodeText(reader);
                    break;
                case "rmsprop_param":
                    message.rmsprop_param = $root.nnabla.RMSpropParameter.decodeText(reader);
                    break;
                case "rmsprop_graves_param":
                    message.rmsprop_graves_param = $root.nnabla.RMSpropGravesParameter.decodeText(reader);
                    break;
                case "adam_param":
                    message.adam_param = $root.nnabla.AdamParameter.decodeText(reader);
                    break;
                case "adamw_param":
                    message.adamw_param = $root.nnabla.AdamWParameter.decodeText(reader);
                    break;
                case "adabound_param":
                    message.adabound_param = $root.nnabla.AdaBoundParameter.decodeText(reader);
                    break;
                case "adamax_param":
                    message.adamax_param = $root.nnabla.AdamaxParameter.decodeText(reader);
                    break;
                case "amsgrad_param":
                    message.amsgrad_param = $root.nnabla.AMSGRADParameter.decodeText(reader);
                    break;
                case "amsbound_param":
                    message.amsbound_param = $root.nnabla.AMSBoundParameter.decodeText(reader);
                    break;
                // Solver parameter messages - End
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Solver.prototype.type = "";
$root.nnabla.Solver.prototype.context = null;
$root.nnabla.Solver.prototype.weight_decay = 0.0;
$root.nnabla.Solver.prototype.states = null;
$root.nnabla.Solver.prototype.lr_scheduler_type = "";
$root.nnabla.Solver.prototype.polynomial_scheduler_param = null;
$root.nnabla.Solver.prototype.cosine_scheduler_param = null;
$root.nnabla.Solver.prototype.exponential_scheduler_param = null;
$root.nnabla.Solver.prototype.step_scheduler_param = null;
$root.nnabla.Solver.prototype.custom_scheduler_param = null;
$root.nnabla.Solver.prototype.lr_warmup_scheduler_type = "";
$root.nnabla.Solver.prototype.linear_warmup_scheduler_param = null;
$root.nnabla.Solver.prototype.lr_decay = 0.0; // Obsolete
$root.nnabla.Solver.prototype.lr_decay_interval = 0; // Obsolete

$root.nnabla.SolverState = class SolverState {

    constructor() {
        this.state_parameter = {};
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SolverStateParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "t":
                    message.t = reader.uint32();
                    break;
                case "state_parameter":
                    reader.entry(message.state_parameter, () => reader.string(), () => $root.nnabla.SolverStateParameter.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SolverState.prototype.t = 0;

$root.nnabla.SolverStateParameter = class SolverStateParameter {

    constructor() {
        this.data = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SolverStateParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "data":
                    message.data.push(reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SolverStateParameter.prototype.shape = null;

$root.nnabla.StepSchedulerParameter = class StepSchedulerParameter {

    constructor() {
        this.iter_steps = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.StepSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch(tag) {
                case "gamma":
                    message.gamma = reader.float();
                    break;
                case "iter_steps":
                    message.iter_steps.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.StepSchedulerParameter.prototype.gamma = 0.0;

$root.nnabla.TrainingConfig = class TrainingConfig {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TrainingConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "max_epoch":
                    message.max_epoch = reader.int64();
                    break;
                case "iter_per_epoch":
                    message.iter_per_epoch = reader.int64();
                    break;
                case "save_best":
                    message.save_best = reader.bool();
                    break;
                case "monitor_interval":
                    message.monitor_interval = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TrainingConfig.prototype.max_epoch = 1;
$root.nnabla.TrainingConfig.prototype.iter_per_epoch = 1;
$root.nnabla.TrainingConfig.prototype.save_best = false;
$root.nnabla.TrainingConfig.prototype.monitor_interval = 1;

$root.nnabla.Variable = class Variable {

    constructor() {
        this.repeat_id = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Variable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "repeat_id":
                    message.repeat_id.push(reader.string());
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "initializer":
                    message.initializer = $root.nnabla.Initializer.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Variable.prototype.name = "";
$root.nnabla.Variable.prototype.type = "";
$root.nnabla.Variable.prototype.shape = null;
$root.nnabla.Variable.prototype.intializer = null;

$root.nnabla.AffineParameter = class AffineParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AffineParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AffineParameter.prototype.base_axis = 1;

$root.nnabla.RNNParameter = class RNNParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RNNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_layers":
                    message.num_layers = reader.int64();
                    break;
                case "nonlinearity":
                    message.nonlinearity = reader.string();
                    break;
                case "dropout":
                    message.dropout = reader.float();
                    break;
                case "bidirectional":
                    message.bidirectional = reader.bool();
                    break;
                case "training":
                    message.training = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RNNParameter.prototype.num_layers = 1;
$root.nnabla.RNNParameter.prototype.nonlinearity = "tanh";
$root.nnabla.RNNParameter.prototype.dropout = 0.0;
$root.nnabla.RNNParameter.prototype.bidirectional = false;
$root.nnabla.RNNParameter.prototype.training = true;

$root.nnabla.LSTMParameter = class LSTMParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LSTMParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_layers":
                    message.num_layers = reader.int64();
                    break;
                case "dropout":
                    message.dropout = reader.float();
                    break;
                case "bidirectional":
                    message.bidirectional = reader.bool();
                    break;
                case "training":
                    message.training = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LSTMParameter.prototype.num_layers = 1;
$root.nnabla.LSTMParameter.prototype.dropout = 0.0;
$root.nnabla.LSTMParameter.prototype.bidirectional = false;
$root.nnabla.LSTMParameter.prototype.training = true;

$root.nnabla.GRUParameter = class GRUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GRUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_layers":
                    message.num_layers = reader.int64();
                    break;
                case "dropout":
                    message.dropout = reader.float();
                    break;
                case "bidirectional":
                    message.bidirectional = reader.bool();
                    break;
                case "training":
                    message.training = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GRUParameter.prototype.num_layers = 1;
$root.nnabla.GRUParameter.prototype.dropout = 0.0;
$root.nnabla.GRUParameter.prototype.bidirectional = false;
$root.nnabla.GRUParameter.prototype.training = true;

$root.nnabla.ConvolutionParameter = class ConvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.ConvolutionParameter.prototype.pad = null;
$root.nnabla.ConvolutionParameter.prototype.stride = null;
$root.nnabla.ConvolutionParameter.prototype.dilation = null;
$root.nnabla.ConvolutionParameter.prototype.group = 1;
$root.nnabla.ConvolutionParameter.prototype.channel_last = false;

$root.nnabla.FusedConvolutionParameter = class FusedConvolutionParameter {

    constructor() {
        this.nonlinearity_args = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.FusedConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                case "decay_rate":
                    message.decay_rate = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "batch_stat":
                    message.batch_stat = reader.bool();
                    break;
                case "nonlinearity":
                    message.nonlinearity = reader.string();
                    break;
                case "nonlinearity_args":
                    message.nonlinearity_args.push(reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.FusedConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.FusedConvolutionParameter.prototype.pad = null;
$root.nnabla.FusedConvolutionParameter.prototype.stride = null;
$root.nnabla.FusedConvolutionParameter.prototype.dilation = null;
$root.nnabla.FusedConvolutionParameter.prototype.group = 1;
$root.nnabla.FusedConvolutionParameter.prototype.channel_last = false;
$root.nnabla.FusedConvolutionParameter.prototype.decay_rate = 0.9;
$root.nnabla.FusedConvolutionParameter.prototype.eps = 1e-05;
$root.nnabla.FusedConvolutionParameter.prototype.batch_stat = true;
$root.nnabla.FusedConvolutionParameter.prototype.nonlinearity = "relu";

$root.nnabla.DepthwiseConvolutionParameter = class DepthwiseConvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DepthwiseConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "multiplier":
                    message.multiplier = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.DepthwiseConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.DepthwiseConvolutionParameter.prototype.pad = null;
$root.nnabla.DepthwiseConvolutionParameter.prototype.stride = null;
$root.nnabla.DepthwiseConvolutionParameter.prototype.dilation = null;
$root.nnabla.DepthwiseConvolutionParameter.prototype.multiplier = 1;

$root.nnabla.DeconvolutionParameter = class DeconvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DeconvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                case "output_padding":
                    message.output_padding = $root.nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.DeconvolutionParameter.prototype.base_axis = 1;
$root.nnabla.DeconvolutionParameter.prototype.pad = null;
$root.nnabla.DeconvolutionParameter.prototype.stride = null;
$root.nnabla.DeconvolutionParameter.prototype.dilation = null;
$root.nnabla.DeconvolutionParameter.prototype.group = 1;
$root.nnabla.DeconvolutionParameter.prototype.channel_last = false;
$root.nnabla.DeconvolutionParameter.prototype.output_padding = null;

$root.nnabla.DepthwiseDeconvolutionParameter = class DepthwiseDeconvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DepthwiseDeconvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "divisor":
                    message.divisor = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.DepthwiseDeconvolutionParameter.prototype.base_axis = 1;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.pad = null;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.stride = null;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.dilation = null;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.divisor = 1;

$root.nnabla.DeformableConvolutionParameter = class DeformableConvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DeformableConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "deformable_group":
                    message.deformable_group = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.DeformableConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.DeformableConvolutionParameter.prototype.pad = null;
$root.nnabla.DeformableConvolutionParameter.prototype.stride = null;
$root.nnabla.DeformableConvolutionParameter.prototype.dilation = null;
$root.nnabla.DeformableConvolutionParameter.prototype.group = 1;
$root.nnabla.DeformableConvolutionParameter.prototype.deformable_group = 1;
$root.nnabla.DeformableConvolutionParameter.prototype.channel_last = false;

$root.nnabla.AdaptiveSeparableConvolutionParameter = class AdaptiveSeparableConvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdaptiveSeparableConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MaxPoolingParameter = class MaxPoolingParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MaxPoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MaxPoolingParameter.prototype.kernel = null;
$root.nnabla.MaxPoolingParameter.prototype.stride = null;
$root.nnabla.MaxPoolingParameter.prototype.ignore_border = true;
$root.nnabla.MaxPoolingParameter.prototype.pad = null;
$root.nnabla.MaxPoolingParameter.prototype.channel_last = false;

$root.nnabla.AveragePoolingParameter = class AveragePoolingParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AveragePoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                case "including_pad":
                    message.including_pad = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AveragePoolingParameter.prototype.kernel = null;
$root.nnabla.AveragePoolingParameter.prototype.stride = null;
$root.nnabla.AveragePoolingParameter.prototype.ignore_border = true;
$root.nnabla.AveragePoolingParameter.prototype.pad = null;
$root.nnabla.AveragePoolingParameter.prototype.channel_last = false;
$root.nnabla.AveragePoolingParameter.prototype.including_pad = true;

$root.nnabla.GlobalAveragePoolingParameter = class GlobalAveragePoolingParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GlobalAveragePoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SumPoolingParameter = class SumPoolingParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SumPoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SumPoolingParameter.prototype.kernel = null;
$root.nnabla.SumPoolingParameter.prototype.stride = null;
$root.nnabla.SumPoolingParameter.prototype.ignore_border = true;
$root.nnabla.SumPoolingParameter.prototype.pad = null;
$root.nnabla.SumPoolingParameter.prototype.channel_last = false;

$root.nnabla.UnpoolingParameter = class UnpoolingParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.UnpoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.UnpoolingParameter.prototype.kernel = null;
$root.nnabla.UnpoolingParameter.prototype.channel_last = false;

$root.nnabla.EmbedParameter = class EmbedParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.EmbedParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RoiAlignParameter = class RoiAlignParameter {

    constructor() {
        this.spatial_scale = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.RoiAlignParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "output_size":
                    message.output_size = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "spatial_scale":
                    message.spatial_scale.push(reader.float());
                    break;
                case "sampling_ratio":
                    message.sampling_ratio = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RoiAlignParameter.prototype.output_size = null;
$root.nnabla.RoiAlignParameter.prototype.sampling_ratio = -1;
$root.nnabla.RoiAlignParameter.prototype.channel_last = false;

$root.nnabla.SigmoidParameter = class SigmoidParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SigmoidParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SwishParameter = class SwishParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SwishParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TanhParameter = class TanhParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TanhParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ReLUParameter = class ReLUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ReLUParameter.prototype.inplace = false;

$root.nnabla.LeakyReLUParameter = class LeakyReLUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LeakyReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LeakyReLUParameter.prototype.alpha = 0.1;
$root.nnabla.LeakyReLUParameter.prototype.inplace = false;

$root.nnabla.SoftmaxParameter = class SoftmaxParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SoftmaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SoftmaxParameter.prototype.axis = 0;

$root.nnabla.LogSoftmaxParameter = class LogSoftmaxParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogSoftmaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogSoftmaxParameter.prototype.axis = 0;

$root.nnabla.ELUParameter = class ELUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ELUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ELUParameter.prototype.alpha = 1.0;

$root.nnabla.SELUParameter = class SELUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SELUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "scale":
                    message.scale = reader.double();
                    break;
                case "alpha":
                    message.alpha = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SELUParameter.prototype.scale = 1.05070098735548;
$root.nnabla.SELUParameter.prototype.alpha = 1.673263242354377;

$root.nnabla.CReLUParameter = class CReLUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CReLUParameter.prototype.axis = 1;

$root.nnabla.CELUParameter = class CELUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CELUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.double();
                    break;
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CELUParameter.prototype.alpha = 1.0;
$root.nnabla.CELUParameter.prototype.axis = 1;

$root.nnabla.PReLUParameter = class PReLUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PReLUParameter.prototype.base_axis = 1;

$root.nnabla.GELUParameter = class GELUParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GELUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MishParameter = class MishParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MishParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ReLU6Parameter = class ReLU6Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ReLU6Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.HardSigmoidParameter = class HardSigmoidParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.HardSigmoidParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.HardTanhParameter = class HardTanhParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.HardTanhParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogSigmoidParameter = class LogSigmoidParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogSigmoidParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SoftPlusParameter = class SoftPlusParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SoftPlusParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "beta":
                    message.beta = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SoftPlusParameter.prototype.beta = 1.0;

$root.nnabla.SoftSignParameter = class SoftSignParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SoftSignParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TanhShrinkParameter = class TanhShrinkParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TanhShrinkParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SincParameter = class SincParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SincParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.FusedBatchNormalizationParameter = class FusedBatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.FusedBatchNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "decay_rate":
                    message.decay_rate = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "batch_stat":
                    message.batch_stat = reader.bool();
                    break;
                case "nonlinearity":
                    message.nonlinearity = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.FusedBatchNormalizationParameter.prototype.decay_rate = 0.9;
$root.nnabla.FusedBatchNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.FusedBatchNormalizationParameter.prototype.batch_stat = true;
$root.nnabla.FusedBatchNormalizationParameter.prototype.nonlinearity = "relu";

$root.nnabla.BatchNormalizationParameter = class BatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "decay_rate":
                    message.decay_rate = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "batch_stat":
                    message.batch_stat = reader.bool();
                    break;
                case "no_scale":
                    message.no_scale = reader.bool();
                    break;
                case "no_bias":
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BatchNormalizationParameter.prototype.decay_rate = 0.9;
$root.nnabla.BatchNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.BatchNormalizationParameter.prototype.batch_stat = true;
$root.nnabla.BatchNormalizationParameter.prototype.no_scale = false;
$root.nnabla.BatchNormalizationParameter.prototype.no_bias = false;

$root.nnabla.GroupNormalizationParameter = class GroupNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.GroupNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_groups":
                    message.num_groups = reader.int64();
                    break;
                case "channel_axis":
                    message.channel_axis = reader.int64();
                    break;
                case "batch_axis":
                    message.batch_axis.push(reader.int64());
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "no_scale":
                    message.no_scale = reader.bool();
                    break;
                case "no_bias":
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GroupNormalizationParameter.prototype.num_groups = 1;
$root.nnabla.GroupNormalizationParameter.prototype.channel_axis = 1;
$root.nnabla.GroupNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.GroupNormalizationParameter.prototype.no_scale = false;
$root.nnabla.GroupNormalizationParameter.prototype.no_bias = false;

$root.nnabla.InstanceNormalizationParameter = class InstanceNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.InstanceNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "channel_axis":
                    message.channel_axis = reader.int64();
                    break;
                case "batch_axis":
                    message.batch_axis.push(reader.int64());
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "no_scale":
                    message.no_scale = reader.bool();
                    break;
                case "no_bias":
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.InstanceNormalizationParameter.prototype.channel_axis = 1;
$root.nnabla.InstanceNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.InstanceNormalizationParameter.prototype.no_scale = false;
$root.nnabla.InstanceNormalizationParameter.prototype.no_bias = false;

$root.nnabla.LayerNormalizationParameter = class LayerNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.LayerNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "batch_axis":
                    message.batch_axis.push(reader.int64());
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "no_scale":
                    message.no_scale = reader.bool();
                    break;
                case "no_bias":
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LayerNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.LayerNormalizationParameter.prototype.no_scale = false;
$root.nnabla.LayerNormalizationParameter.prototype.no_bias = false;

$root.nnabla.NormNormalizationParameter = class NormNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.NormNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "p":
                    message.p = reader.float();
                    break;
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.NormNormalizationParameter.prototype.p = 0.0;
$root.nnabla.NormNormalizationParameter.prototype.eps = 1e-12;

$root.nnabla.SyncBatchNormalizationParameter = class SyncBatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SyncBatchNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "comm":
                    message.comm = $root.nnabla.Communicator.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.string();
                    break;
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "decay_rate":
                    message.decay_rate = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "batch_stat":
                    message.batch_stat = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SyncBatchNormalizationParameter.prototype.group = "world";
$root.nnabla.SyncBatchNormalizationParameter.prototype.decay_rate = 0.9;
$root.nnabla.SyncBatchNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.SyncBatchNormalizationParameter.prototype.batch_stat = true;

$root.nnabla.TensorNormalizationParameter = class TensorNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.TensorNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "no_scale":
                    message.no_scale = reader.bool();
                    break;
                case "no_bias":
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TensorNormalizationParameter.prototype.eps = 1e-05;
$root.nnabla.TensorNormalizationParameter.prototype.no_scale = false;
$root.nnabla.TensorNormalizationParameter.prototype.no_bias = false;

$root.nnabla.WeightNormalizationParameter = class WeightNormalizationParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.WeightNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim = reader.int64();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.WeightNormalizationParameter.prototype.dim = 0;
$root.nnabla.WeightNormalizationParameter.prototype.eps = 1e-12;

$root.nnabla.WeightStandardizationParameter = class WeightStandardizationParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.WeightStandardizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "channel_axis":
                    message.channel_axis = reader.int64();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.WeightStandardizationParameter.prototype.channel_axis = 0;
$root.nnabla.WeightStandardizationParameter.prototype.eps = 1e-05;

$root.nnabla.SpectralNormParameter = class SpectralNormParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SpectralNormParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim = reader.int64();
                    break;
                case "itr":
                    message.itr = reader.int64();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "test":
                    message.test = reader.bool();
                    break;
                case "output_u":
                    message.output_u = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SpectralNormParameter.prototype.dim = 0;
$root.nnabla.SpectralNormParameter.prototype.itr = 1;
$root.nnabla.SpectralNormParameter.prototype.eps = 1e-12;
$root.nnabla.SpectralNormParameter.prototype.test = false;
$root.nnabla.SpectralNormParameter.prototype.output_u = false;

$root.nnabla.MeanSubtractionParameter = class MeanSubtractionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MeanSubtractionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "update_running_mean":
                    message.update_running_mean = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MeanSubtractionParameter.prototype.base_axis = 1;
$root.nnabla.MeanSubtractionParameter.prototype.update_running_mean = true;

$root.nnabla.ClipGradByValueParameter = class ClipGradByValueParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ClipGradByValueParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ClipGradByNormParameter = class ClipGradByNormParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ClipGradByNormParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "clip_norm":
                    message.clip_norm = reader.float();
                    break;
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ClipGradByNormParameter.prototype.clip_norm = 1.0;

$root.nnabla.SumParameter = class SumParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SumParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "keep_dims":
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SumParameter.prototype.keep_dims = false;

$root.nnabla.CumSumParameter = class CumSumParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CumSumParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "exclusive":
                    message.exclusive = reader.bool();
                    break;
                case "reverse":
                    message.reverse = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CumSumParameter.prototype.axis = 0;
$root.nnabla.CumSumParameter.prototype.exclusive = false;
$root.nnabla.CumSumParameter.prototype.reverse = false;

$root.nnabla.MeanParameter = class MeanParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.MeanParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "keep_dims":
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MeanParameter.prototype.keep_dims = false;

$root.nnabla.MaxParameter = class MaxParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.MaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "keep_dims":
                    message.keep_dims = reader.bool();
                    break;
                case "with_index":
                    message.with_index = reader.bool();
                    break;
                case "only_index":
                    message.only_index = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MaxParameter.prototype.keep_dims = false;
$root.nnabla.MaxParameter.prototype.with_index = false;
$root.nnabla.MaxParameter.prototype.only_index = false;

$root.nnabla.MinParameter = class MinParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.MinParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "keep_dims":
                    message.keep_dims = reader.bool();
                    break;
                case "with_index":
                    message.with_index = reader.bool();
                    break;
                case "only_index":
                    message.only_index = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MinParameter.prototype.keep_dims = false;
$root.nnabla.MinParameter.prototype.with_index = false;
$root.nnabla.MinParameter.prototype.only_index = false;

$root.nnabla.NormParameter = class NormParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.NormParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "p":
                    message.p = reader.float();
                    break;
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "keep_dims":
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.NormParameter.prototype.p = 0.0;
$root.nnabla.NormParameter.prototype.keep_dims = false;

$root.nnabla.ProdParameter = class ProdParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ProdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "keep_dims":
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ProdParameter.prototype.keep_dims = false;

$root.nnabla.CumProdParameter = class CumProdParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CumProdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "exclusive":
                    message.exclusive = reader.bool();
                    break;
                case "reverse":
                    message.reverse = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CumProdParameter.prototype.axis = 0;
$root.nnabla.CumProdParameter.prototype.exclusive = false;
$root.nnabla.CumProdParameter.prototype.reverse = false;

$root.nnabla.ReduceSumParameter = class ReduceSumParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ReduceSumParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ReduceMeanParameter = class ReduceMeanParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ReduceMeanParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Add2Parameter = class Add2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Add2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Add2Parameter.prototype.inplace = false;

$root.nnabla.AddNParameter = class AddNParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AddNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BcAdd2Parameter = class BcAdd2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BcAdd2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BcAdd2Parameter.prototype.inplace = false;

$root.nnabla.Sub2Parameter = class Sub2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Sub2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Sub2Parameter.prototype.inplace = false;

$root.nnabla.Mul2Parameter = class Mul2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Mul2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Mul2Parameter.prototype.inplace = false;

$root.nnabla.MulNParameter = class MulNParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MulNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Div2Parameter = class Div2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Div2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Div2Parameter.prototype.inplace = false;

$root.nnabla.Pow2Parameter = class Pow2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Pow2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Pow2Parameter.prototype.inplace = false;

$root.nnabla.AddScalarParameter = class AddScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AddScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AddScalarParameter.prototype.val = 0.0;
$root.nnabla.AddScalarParameter.prototype.inplace = false;

$root.nnabla.MulScalarParameter = class MulScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MulScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MulScalarParameter.prototype.val = 0.0;
$root.nnabla.MulScalarParameter.prototype.inplace = false;

$root.nnabla.PowScalarParameter = class PowScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PowScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PowScalarParameter.prototype.val = 0.0;
$root.nnabla.PowScalarParameter.prototype.inplace = false;

$root.nnabla.RSubScalarParameter = class RSubScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RSubScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RSubScalarParameter.prototype.val = 0.0;

$root.nnabla.RDivScalarParameter = class RDivScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RDivScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RDivScalarParameter.prototype.val = 0.0;

$root.nnabla.RPowScalarParameter = class RPowScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RPowScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RPowScalarParameter.prototype.val = 0.0;

$root.nnabla.SignParameter = class SignParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SignParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SignParameter.prototype.alpha = 1.0;

$root.nnabla.Minimum2Parameter = class Minimum2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Minimum2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Maximum2Parameter = class Maximum2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Maximum2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MinimumScalarParameter = class MinimumScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MinimumScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MinimumScalarParameter.prototype.val = 1.0;

$root.nnabla.MaximumScalarParameter = class MaximumScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MaximumScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MaximumScalarParameter.prototype.val = 1.0;

$root.nnabla.LogicalAndParameter = class LogicalAndParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalAndParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogicalOrParameter = class LogicalOrParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalOrParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogicalXorParameter = class LogicalXorParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalXorParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.EqualParameter = class EqualParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.EqualParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.NotEqualParameter = class NotEqualParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.NotEqualParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GreaterEqualParameter = class GreaterEqualParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GreaterEqualParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GreaterParameter = class GreaterParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GreaterParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LessEqualParameter = class LessEqualParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LessEqualParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LessParameter = class LessParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LessParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SearchSortedParameter = class SearchSortedParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SearchSortedParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "right":
                    message.right = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SearchSortedParameter.prototype.right = false;

$root.nnabla.LogicalAndScalarParameter = class LogicalAndScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalAndScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogicalAndScalarParameter.prototype.val = false;

$root.nnabla.LogicalOrScalarParameter = class LogicalOrScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalOrScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogicalOrScalarParameter.prototype.val = false;

$root.nnabla.LogicalXorScalarParameter = class LogicalXorScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalXorScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogicalXorScalarParameter.prototype.val = false;

$root.nnabla.EqualScalarParameter = class EqualScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.EqualScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.EqualScalarParameter.prototype.val = 0.0;

$root.nnabla.NotEqualScalarParameter = class NotEqualScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.NotEqualScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.NotEqualScalarParameter.prototype.val = 0.0;

$root.nnabla.GreaterEqualScalarParameter = class GreaterEqualScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GreaterEqualScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GreaterEqualScalarParameter.prototype.val = 0.0;

$root.nnabla.GreaterScalarParameter = class GreaterScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GreaterScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GreaterScalarParameter.prototype.val = 0.0;

$root.nnabla.LessEqualScalarParameter = class LessEqualScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LessEqualScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LessEqualScalarParameter.prototype.val = 0.0;

$root.nnabla.LessScalarParameter = class LessScalarParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LessScalarParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LessScalarParameter.prototype.val = 0.0;

$root.nnabla.LogicalNotParameter = class LogicalNotParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogicalNotParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.IsNaNParameter = class IsNaNParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.IsNaNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.IsInfParameter = class IsInfParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.IsInfParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ResetNaNParameter = class ResetNaNParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ResetNaNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ResetNaNParameter.prototype.val = 0.0;

$root.nnabla.ResetInfParameter = class ResetInfParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ResetInfParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ResetInfParameter.prototype.val = 0.0;

$root.nnabla.WhereParameter = class WhereParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.WhereParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConstantParameter = class ConstantParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ConstantParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.float();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConstantParameter.prototype.val = 0.0;
$root.nnabla.ConstantParameter.prototype.shape = null;

$root.nnabla.ArangeParameter = class ArangeParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ArangeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    message.start = reader.float();
                    break;
                case "stop":
                    message.stop = reader.float();
                    break;
                case "step":
                    message.step = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ArangeParameter.prototype.start = 0.0;
$root.nnabla.ArangeParameter.prototype.stop = 0.0;
$root.nnabla.ArangeParameter.prototype.step = 0.0;

$root.nnabla.LinspaceParameter = class LinspaceParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LinspaceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    message.start = reader.float();
                    break;
                case "stop":
                    message.stop = reader.float();
                    break;
                case "num":
                    message.num = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LinspaceParameter.prototype.start = 0.0;
$root.nnabla.LinspaceParameter.prototype.stop = 0.0;
$root.nnabla.LinspaceParameter.prototype.num = 0;

$root.nnabla.AbsParameter = class AbsParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AbsParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ExpParameter = class ExpParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ExpParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LogParameter = class LogParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LogParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.IdentityParameter = class IdentityParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.IdentityParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BatchMatmulParameter = class BatchMatmulParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchMatmulParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "transpose_a":
                    message.transpose_a = reader.bool();
                    break;
                case "transpose_b":
                    message.transpose_b = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BatchMatmulParameter.prototype.transpose_a = false;
$root.nnabla.BatchMatmulParameter.prototype.transpose_b = false;

$root.nnabla.RoundParameter = class RoundParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RoundParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CeilParameter = class CeilParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CeilParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.FloorParameter = class FloorParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.FloorParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SinParameter = class SinParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SinParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CosParameter = class CosParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CosParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TanParameter = class TanParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TanParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SinhParameter = class SinhParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SinhParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CoshParameter = class CoshParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CoshParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ASinParameter = class ASinParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ASinParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ACosParameter = class ACosParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ACosParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ATanParameter = class ATanParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ATanParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ATan2Parameter = class ATan2Parameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ATan2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ASinhParameter = class ASinhParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ASinhParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ACoshParameter = class ACoshParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ACoshParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ATanhParameter = class ATanhParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ATanhParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConcatenateParameter = class ConcatenateParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ConcatenateParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConcatenateParameter.prototype.axis = 0;

$root.nnabla.SplitParameter = class SplitParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SplitParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SplitParameter.prototype.axis = 0;

$root.nnabla.StackParameter = class StackParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.StackParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.StackParameter.prototype.axis = 0;

$root.nnabla.SliceParameter = class SliceParameter {

    constructor() {
        this.start = [];
        this.stop = [];
        this.step = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SliceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    message.start.push(reader.int64());
                    break;
                case "stop":
                    message.stop.push(reader.int64());
                    break;
                case "step":
                    message.step.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PadParameter = class PadParameter {

    constructor() {
        this.pad_width = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.PadParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pad_width":
                    message.pad_width.push(reader.int64());
                    break;
                case "mode":
                    message.mode = reader.string();
                    break;
                case "constant_value":
                    message.constant_value = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PadParameter.prototype.mode = "constant";
$root.nnabla.PadParameter.prototype.constant_value = 0.0;

$root.nnabla.TransposeParameter = class TransposeParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.TransposeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BroadcastParameter = class BroadcastParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BroadcastParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BroadcastParameter.prototype.shape = null;

$root.nnabla.BroadcastToParameter = class BroadcastToParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BroadcastToParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BroadcastToParameter.prototype.axis = -1;

$root.nnabla.TileParameter = class TileParameter {

    constructor() {
        this.reps = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.TileParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "reps":
                    message.reps.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.OneHotParameter = class OneHotParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.OneHotParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.OneHotParameter.prototype.shape = null;

$root.nnabla.FlipParameter = class FlipParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.FlipParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ShiftParameter = class ShiftParameter {

    constructor() {
        this.shifts = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ShiftParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shifts":
                    message.shifts.push(reader.int64());
                    break;
                case "border_mode":
                    message.border_mode = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ShiftParameter.prototype.border_mode = "nearest";

$root.nnabla.SortParameter = class SortParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SortParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "reverse":
                    message.reverse = reader.bool();
                    break;
                case "with_index":
                    message.with_index = reader.bool();
                    break;
                case "only_index":
                    message.only_index = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SortParameter.prototype.axis = -1;
$root.nnabla.SortParameter.prototype.reverse = false;
$root.nnabla.SortParameter.prototype.with_index = false;
$root.nnabla.SortParameter.prototype.only_index = false;

$root.nnabla.ReshapeParameter = class ReshapeParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ReshapeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ReshapeParameter.prototype.shape = null;
$root.nnabla.ReshapeParameter.prototype.inplace = true;

$root.nnabla.MatrixDiagParameter = class MatrixDiagParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MatrixDiagParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MatrixDiagPartParameter = class MatrixDiagPartParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MatrixDiagPartParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MeshgridParameter = class MeshgridParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MeshgridParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ij_indexing":
                    message.ij_indexing = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MeshgridParameter.prototype.ij_indexing = false;

$root.nnabla.BatchDetParameter = class BatchDetParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchDetParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BatchInvParameter = class BatchInvParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchInvParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BatchLogdetParameter = class BatchLogdetParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchLogdetParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AssignParameter = class AssignParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AssignParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GatherParameter = class GatherParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GatherParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "batch_dims":
                    message.batch_dims = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.GatherParameter.prototype.axis = 0;
$root.nnabla.GatherParameter.prototype.batch_dims = 0;

$root.nnabla.GatherNdParameter = class GatherNdParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.GatherNdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BoolGatherParameter = class BoolGatherParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BoolGatherParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ScatterNdParameter = class ScatterNdParameter {

    constructor() {
        this.shape = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ScatterNdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape.push(reader.int64());
                    break;
                case "add":
                    message.add = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ScatterNdParameter.prototype.add = false;

$root.nnabla.ScatterAddParameter = class ScatterAddParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ScatterAddParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ScatterAddParameter.prototype.axis = 0;

$root.nnabla.BoolScatterParameter = class BoolScatterParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BoolScatterParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BoolFillParameter = class BoolFillParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BoolFillParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BoolFillParameter.prototype.value = 0.0;

$root.nnabla.PackPaddedSequenceParameter = class PackPaddedSequenceParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PackPaddedSequenceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "batch_first":
                    message.batch_first = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PackPaddedSequenceParameter.prototype.batch_first = false;

$root.nnabla.PadPackedSequenceParameter = class PadPackedSequenceParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PadPackedSequenceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "batch_first":
                    message.batch_first = reader.bool();
                    break;
                case "padding_value":
                    message.padding_value = reader.float();
                    break;
                case "total_length":
                    message.total_length = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PadPackedSequenceParameter.prototype.batch_first = false;
$root.nnabla.PadPackedSequenceParameter.prototype.padding_value = 0.0;
$root.nnabla.PadPackedSequenceParameter.prototype.total_length = -1;

$root.nnabla.InterpolateParameter = class InterpolateParameter {

    constructor() {
        this.output_size = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.InterpolateParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "output_size":
                    message.output_size.push(reader.int64());
                    break;
                case "mode":
                    message.mode = reader.string();
                    break;
                case "align_corners":
                    message.align_corners = reader.bool();
                    break;
                case "half_pixel":
                    message.half_pixel = reader.bool();
                    break;
                case "half_pixel_for_nn":
                    message.half_pixel_for_nn = reader.bool();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.InterpolateParameter.prototype.mode = "";
$root.nnabla.InterpolateParameter.prototype.align_corners = true;
$root.nnabla.InterpolateParameter.prototype.half_pixel = false;
$root.nnabla.InterpolateParameter.prototype.half_pixel_for_nn = false;
$root.nnabla.InterpolateParameter.prototype.channel_last = false;

$root.nnabla.FFTParameter = class FFTParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.FFTParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "signal_ndim":
                    message.signal_ndim = reader.int64();
                    break;
                case "normalized":
                    message.normalized = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.FFTParameter.prototype.signal_ndim = 0;
$root.nnabla.FFTParameter.prototype.normalized = false;

$root.nnabla.IFFTParameter = class IFFTParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.IFFTParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "signal_ndim":
                    message.signal_ndim = reader.int64();
                    break;
                case "normalized":
                    message.normalized = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.IFFTParameter.prototype.signal_ndim = 0;
$root.nnabla.IFFTParameter.prototype.normalized = false;

$root.nnabla.STFTParameter = class STFTParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.STFTParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "window_size":
                    message.window_size = reader.int64();
                    break;
                case "stride":
                    message.stride = reader.int64();
                    break;
                case "fft_size":
                    message.fft_size = reader.int64();
                    break;
                case "window_type":
                    message.window_type = reader.string();
                    break;
                case "center":
                    message.center = reader.bool();
                    break;
                case "pad_mode":
                    message.pad_mode = reader.string();
                    break;
                case "as_istft_backward":
                    message.as_istft_backward = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.STFTParameter.prototype.window_size = 0;
$root.nnabla.STFTParameter.prototype.stride = 0;
$root.nnabla.STFTParameter.prototype.fft_size = 0;
$root.nnabla.STFTParameter.prototype.window_type = "hanning";
$root.nnabla.STFTParameter.prototype.center = true;
$root.nnabla.STFTParameter.prototype.pad_mode = "reflect";
$root.nnabla.STFTParameter.prototype.as_istft_backward = false;

$root.nnabla.ISTFTParameter = class ISTFTParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ISTFTParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "window_size":
                    message.window_size = reader.int64();
                    break;
                case "stride":
                    message.stride = reader.int64();
                    break;
                case "fft_size":
                    message.fft_size = reader.int64();
                    break;
                case "window_type":
                    message.window_type = reader.string();
                    break;
                case "center":
                    message.center = reader.bool();
                    break;
                case "pad_mode":
                    message.pad_mode = reader.string();
                    break;
                case "as_stft_backward":
                    message.as_stft_backward = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ISTFTParameter.prototype.window_size = 0;
$root.nnabla.ISTFTParameter.prototype.stride = 0;
$root.nnabla.ISTFTParameter.prototype.fft_size = 0;
$root.nnabla.ISTFTParameter.prototype.window_type = "hanning";
$root.nnabla.ISTFTParameter.prototype.center = true;
$root.nnabla.ISTFTParameter.prototype.pad_mode = "reflect";
$root.nnabla.ISTFTParameter.prototype.as_stft_backward = false;

$root.nnabla.DropoutParameter = class DropoutParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DropoutParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "p":
                    message.p = reader.double();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.DropoutParameter.prototype.p = 0.5;
$root.nnabla.DropoutParameter.prototype.seed = -1;

$root.nnabla.TopKDataParameter = class TopKDataParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TopKDataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.int64();
                    break;
                case "abs":
                    message.abs = reader.bool();
                    break;
                case "reduce":
                    message.reduce = reader.bool();
                    break;
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TopKDataParameter.prototype.k = 0;
$root.nnabla.TopKDataParameter.prototype.abs = false;
$root.nnabla.TopKDataParameter.prototype.reduce = true;
$root.nnabla.TopKDataParameter.prototype.base_axis = 1;

$root.nnabla.TopKGradParameter = class TopKGradParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TopKGradParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.int64();
                    break;
                case "abs":
                    message.abs = reader.bool();
                    break;
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TopKGradParameter.prototype.k = 0;
$root.nnabla.TopKGradParameter.prototype.abs = false;
$root.nnabla.TopKGradParameter.prototype.base_axis = 1;

$root.nnabla.RandParameter = class RandParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "low":
                    message.low = reader.float();
                    break;
                case "high":
                    message.high = reader.float();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandParameter.prototype.low = 0.0;
$root.nnabla.RandParameter.prototype.high = 0.0;
$root.nnabla.RandParameter.prototype.shape = null;
$root.nnabla.RandParameter.prototype.seed = -1;

$root.nnabla.RandintParameter = class RandintParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandintParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "low":
                    message.low = reader.int64();
                    break;
                case "high":
                    message.high = reader.int64();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandintParameter.prototype.low = 0;
$root.nnabla.RandintParameter.prototype.high = 1;
$root.nnabla.RandintParameter.prototype.shape = null;
$root.nnabla.RandintParameter.prototype.seed = -1;

$root.nnabla.RandnParameter = class RandnParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandnParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mu":
                    message.mu = reader.float();
                    break;
                case "sigma":
                    message.sigma = reader.float();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandnParameter.prototype.mu = 0.0;
$root.nnabla.RandnParameter.prototype.sigma = 0.0;
$root.nnabla.RandnParameter.prototype.shape = null;
$root.nnabla.RandnParameter.prototype.seed = -1;

$root.nnabla.RandBinomialParameter = class RandBinomialParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandBinomialParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "n":
                    message.n = reader.int64();
                    break;
                case "p":
                    message.p = reader.float();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandBinomialParameter.prototype.n = 1;
$root.nnabla.RandBinomialParameter.prototype.p = 0.5;
$root.nnabla.RandBinomialParameter.prototype.shape = null;
$root.nnabla.RandBinomialParameter.prototype.seed = -1;

$root.nnabla.RandBetaParameter = class RandBetaParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandBetaParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandBetaParameter.prototype.alpha = 0.5;
$root.nnabla.RandBetaParameter.prototype.beta = 0.5;
$root.nnabla.RandBetaParameter.prototype.shape = null;
$root.nnabla.RandBetaParameter.prototype.seed = -1;

$root.nnabla.RandGammaParameter = class RandGammaParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandGammaParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.float();
                    break;
                case "theta":
                    message.theta = reader.float();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandGammaParameter.prototype.k = 0.5;
$root.nnabla.RandGammaParameter.prototype.theta = 0.0;
$root.nnabla.RandGammaParameter.prototype.shape = null;
$root.nnabla.RandGammaParameter.prototype.seed = -1;

$root.nnabla.RandomChoiceParameter = class RandomChoiceParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomChoiceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "replace":
                    message.replace = reader.bool();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandomChoiceParameter.prototype.shape = null;
$root.nnabla.RandomChoiceParameter.prototype.replace = true;
$root.nnabla.RandomChoiceParameter.prototype.seed = -1;

$root.nnabla.RandomCropParameter = class RandomCropParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomCropParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandomCropParameter.prototype.shape = null;
$root.nnabla.RandomCropParameter.prototype.base_axis = 1;
$root.nnabla.RandomCropParameter.prototype.seed = -1;

$root.nnabla.RandomFlipParameter = class RandomFlipParameter {

    constructor() {
        this.axes = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomFlipParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    message.axes.push(reader.int64());
                    break;
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandomFlipParameter.prototype.base_axis = 1;
$root.nnabla.RandomFlipParameter.prototype.seed = -1;

$root.nnabla.RandomShiftParameter = class RandomShiftParameter {

    constructor() {
        this.shifts = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomShiftParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shifts":
                    message.shifts.push(reader.int64());
                    break;
                case "border_mode":
                    message.border_mode = reader.string();
                    break;
                case "constant_value":
                    message.constant_value = reader.float();
                    break;
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandomShiftParameter.prototype.border_mode = "nearest";
$root.nnabla.RandomShiftParameter.prototype.constant_value = 0.0;
$root.nnabla.RandomShiftParameter.prototype.base_axis = 1;
$root.nnabla.RandomShiftParameter.prototype.seed = -1;

$root.nnabla.RandomEraseParameter = class RandomEraseParameter {

    constructor() {
        this.area_ratios = [];
        this.aspect_ratios = [];
        this.replacements = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomEraseParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "prob":
                    message.prob = reader.float();
                    break;
                case "area_ratios":
                    message.area_ratios.push(reader.float());
                    break;
                case "aspect_ratios":
                    message.aspect_ratios.push(reader.float());
                    break;
                case "replacements":
                    message.replacements.push(reader.float());
                    break;
                case "n":
                    message.n = reader.int64();
                    break;
                case "share":
                    message.share = reader.bool();
                    break;
                case "inplace":
                    message.inplace = reader.bool();
                    break;
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                case "ste_fine_grained":
                    message.ste_fine_grained = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RandomEraseParameter.prototype.prob = 0.5;
$root.nnabla.RandomEraseParameter.prototype.n = 1;
$root.nnabla.RandomEraseParameter.prototype.share = true;
$root.nnabla.RandomEraseParameter.prototype.inplace = false;
$root.nnabla.RandomEraseParameter.prototype.base_axis = 1;
$root.nnabla.RandomEraseParameter.prototype.seed = -1;
$root.nnabla.RandomEraseParameter.prototype.channel_last = false;
$root.nnabla.RandomEraseParameter.prototype.ste_fine_grained = true;

$root.nnabla.ImageAugmentationParameter = class ImageAugmentationParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ImageAugmentationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "min_scale":
                    message.min_scale = reader.float();
                    break;
                case "max_scale":
                    message.max_scale = reader.float();
                    break;
                case "angle":
                    message.angle = reader.float();
                    break;
                case "aspect_ratio":
                    message.aspect_ratio = reader.float();
                    break;
                case "distortion":
                    message.distortion = reader.float();
                    break;
                case "flip_lr":
                    message.flip_lr = reader.bool();
                    break;
                case "flip_ud":
                    message.flip_ud = reader.bool();
                    break;
                case "brightness":
                    message.brightness = reader.float();
                    break;
                case "brightness_each":
                    message.brightness_each = reader.bool();
                    break;
                case "contrast":
                    message.contrast = reader.float();
                    break;
                case "contrast_center":
                    message.contrast_center = reader.float();
                    break;
                case "contrast_each":
                    message.contrast_each = reader.bool();
                    break;
                case "noise":
                    message.noise = reader.float();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ImageAugmentationParameter.prototype.shape = null;
$root.nnabla.ImageAugmentationParameter.prototype.pad = null;
$root.nnabla.ImageAugmentationParameter.prototype.min_scale = 1.0;
$root.nnabla.ImageAugmentationParameter.prototype.max_scale = 1.0;
$root.nnabla.ImageAugmentationParameter.prototype.angle = 0.0;
$root.nnabla.ImageAugmentationParameter.prototype.aspect_ratio = 1.0;
$root.nnabla.ImageAugmentationParameter.prototype.distortion = 0.0;
$root.nnabla.ImageAugmentationParameter.prototype.flip_lr = false;
$root.nnabla.ImageAugmentationParameter.prototype.flip_ud = false;
$root.nnabla.ImageAugmentationParameter.prototype.brightness = 0.0;
$root.nnabla.ImageAugmentationParameter.prototype.brightness_each = false;
$root.nnabla.ImageAugmentationParameter.prototype.contrast = 1.0;
$root.nnabla.ImageAugmentationParameter.prototype.contrast_center = 0.0;
$root.nnabla.ImageAugmentationParameter.prototype.contrast_each = false;
$root.nnabla.ImageAugmentationParameter.prototype.noise = 0.0;
$root.nnabla.ImageAugmentationParameter.prototype.seed = -1;

$root.nnabla.SigmoidCrossEntropyParameter = class SigmoidCrossEntropyParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SigmoidCrossEntropyParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryCrossEntropyParameter = class BinaryCrossEntropyParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryCrossEntropyParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SoftmaxCrossEntropyParameter = class SoftmaxCrossEntropyParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SoftmaxCrossEntropyParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SoftmaxCrossEntropyParameter.prototype.axis = 0;

$root.nnabla.CategoricalCrossEntropyParameter = class CategoricalCrossEntropyParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.CategoricalCrossEntropyParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.CategoricalCrossEntropyParameter.prototype.axis = 0;

$root.nnabla.SquaredErrorParameter = class SquaredErrorParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SquaredErrorParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AbsoluteErrorParameter = class AbsoluteErrorParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AbsoluteErrorParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.HuberLossParameter = class HuberLossParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.HuberLossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "delta":
                    message.delta = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.HuberLossParameter.prototype.delta = 1.0;

$root.nnabla.EpsilonInsensitiveLossParameter = class EpsilonInsensitiveLossParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.EpsilonInsensitiveLossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "epsilon":
                    message.epsilon = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.EpsilonInsensitiveLossParameter.prototype.epsilon = 0.0;

$root.nnabla.KLMultinomialParameter = class KLMultinomialParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.KLMultinomialParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.KLMultinomialParameter.prototype.base_axis = 1;

$root.nnabla.AffineGridParameter = class AffineGridParameter {

    constructor() {
        this.size = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.AffineGridParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    message.size.push(reader.int64());
                    break;
                case "align_corners":
                    message.align_corners = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AffineGridParameter.prototype.align_corners = false;

$root.nnabla.WarpByGridParameter = class WarpByGridParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.WarpByGridParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "mode":
                    message.mode = reader.string();
                    break;
                case "padding_mode":
                    message.padding_mode = reader.string();
                    break;
                case "align_corners":
                    message.align_corners = reader.bool();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.WarpByGridParameter.prototype.mode = "linear";
$root.nnabla.WarpByGridParameter.prototype.padding_mode = "zero";
$root.nnabla.WarpByGridParameter.prototype.align_corners = false;
$root.nnabla.WarpByGridParameter.prototype.channel_last = false;

$root.nnabla.WarpByFlowParameter = class WarpByFlowParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.WarpByFlowParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinarySigmoidParameter = class BinarySigmoidParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinarySigmoidParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryTanhParameter = class BinaryTanhParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryTanhParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryConnectAffineParameter = class BinaryConnectAffineParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryConnectAffineParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "quantize_zero_to":
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryConnectAffineParameter.prototype.base_axis = 1;
$root.nnabla.BinaryConnectAffineParameter.prototype.quantize_zero_to = 1.0;

$root.nnabla.BinaryConnectConvolutionParameter = class BinaryConnectConvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryConnectConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "quantize_zero_to":
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryConnectConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.pad = null;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.stride = null;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.dilation = null;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.group = 1;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.quantize_zero_to = 1.0;

$root.nnabla.BinaryWeightAffineParameter = class BinaryWeightAffineParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryWeightAffineParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "quantize_zero_to":
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryWeightAffineParameter.prototype.base_axis = 1;
$root.nnabla.BinaryWeightAffineParameter.prototype.quantize_zero_to = 1.0;

$root.nnabla.BinaryWeightConvolutionParameter = class BinaryWeightConvolutionParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryWeightConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "quantize_zero_to":
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BinaryWeightConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.pad = null;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.stride = null;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.dilation = null;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.group = 1;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.quantize_zero_to = 1.0;

$root.nnabla.INQAffineParameter = class INQAffineParameter {

    constructor() {
        this.inq_iterations = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.INQAffineParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "num_bits":
                    message.num_bits = reader.int64();
                    break;
                case "inq_iterations":
                    message.inq_iterations.push(reader.int64());
                    break;
                case "selection_algorithm":
                    message.selection_algorithm = reader.string();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.INQAffineParameter.prototype.base_axis = 1;
$root.nnabla.INQAffineParameter.prototype.num_bits = 4;
$root.nnabla.INQAffineParameter.prototype.selection_algorithm = "largest_abs";
$root.nnabla.INQAffineParameter.prototype.seed = -1;

$root.nnabla.INQConvolutionParameter = class INQConvolutionParameter {

    constructor() {
        this.inq_iterations = [];
    }

    static decodeText(reader) {
        const message = new $root.nnabla.INQConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "num_bits":
                    message.num_bits = reader.int64();
                    break;
                case "inq_iterations":
                    message.inq_iterations.push(reader.int64());
                    break;
                case "selection_algorithm":
                    message.selection_algorithm = reader.string();
                    break;
                case "seed":
                    message.seed = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.INQConvolutionParameter.prototype.base_axis = 1;
$root.nnabla.INQConvolutionParameter.prototype.pad = null;
$root.nnabla.INQConvolutionParameter.prototype.stride = null;
$root.nnabla.INQConvolutionParameter.prototype.dilation = null;
$root.nnabla.INQConvolutionParameter.prototype.group = 1;
$root.nnabla.INQConvolutionParameter.prototype.num_bits = 4;
$root.nnabla.INQConvolutionParameter.prototype.selection_algorithm = "largest_abs";
$root.nnabla.INQConvolutionParameter.prototype.seed = -1;

$root.nnabla.FixedPointQuantizeParameter = class FixedPointQuantizeParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.FixedPointQuantizeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sign":
                    message.sign = reader.bool();
                    break;
                case "n":
                    message.n = reader.int64();
                    break;
                case "delta":
                    message.delta = reader.float();
                    break;
                case "ste_fine_grained":
                    message.ste_fine_grained = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.FixedPointQuantizeParameter.prototype.sign = true;
$root.nnabla.FixedPointQuantizeParameter.prototype.n = 8;
$root.nnabla.FixedPointQuantizeParameter.prototype.delta = 0.0625;
$root.nnabla.FixedPointQuantizeParameter.prototype.ste_fine_grained = true;

$root.nnabla.MinMaxQuantizeParameter = class MinMaxQuantizeParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MinMaxQuantizeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "decay":
                    message.decay = reader.float();
                    break;
                case "x_min_max":
                    message.x_min_max = reader.bool();
                    break;
                case "ema":
                    message.ema = reader.bool();
                    break;
                case "ste_fine_grained":
                    message.ste_fine_grained = reader.bool();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MinMaxQuantizeParameter.prototype.decay = 0.999;
$root.nnabla.MinMaxQuantizeParameter.prototype.x_min_max = false;
$root.nnabla.MinMaxQuantizeParameter.prototype.ema = false;
$root.nnabla.MinMaxQuantizeParameter.prototype.ste_fine_grained = true;
$root.nnabla.MinMaxQuantizeParameter.prototype.eps = 0.01;

$root.nnabla.Pow2QuantizeParameter = class Pow2QuantizeParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.Pow2QuantizeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sign":
                    message.sign = reader.bool();
                    break;
                case "with_zero":
                    message.with_zero = reader.bool();
                    break;
                case "n":
                    message.n = reader.int64();
                    break;
                case "m":
                    message.m = reader.int64();
                    break;
                case "ste_fine_grained":
                    message.ste_fine_grained = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Pow2QuantizeParameter.prototype.sign = true;
$root.nnabla.Pow2QuantizeParameter.prototype.with_zero = true;
$root.nnabla.Pow2QuantizeParameter.prototype.n = 8;
$root.nnabla.Pow2QuantizeParameter.prototype.m = 1;
$root.nnabla.Pow2QuantizeParameter.prototype.ste_fine_grained = true;

$root.nnabla.PruneParameter = class PruneParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PruneParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "rate":
                    message.rate = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PruneParameter.prototype.rate = 0.9;

$root.nnabla.QuantizeLinearParameter = class QuantizeLinearParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.QuantizeLinearParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "round_mode":
                    message.round_mode = reader.string();
                    break;
                case "narrow_range":
                    message.narrow_range = reader.bool();
                    break;
                case "dtype":
                    message.dtype = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.QuantizeLinearParameter.prototype.round_mode = "HALF_AWAY_FROM_ZERO";
$root.nnabla.QuantizeLinearParameter.prototype.narrow_range = false;
$root.nnabla.QuantizeLinearParameter.prototype.dtype = 1;

$root.nnabla.DequantizeLinearParameter = class DequantizeLinearParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.DequantizeLinearParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TopNErrorParameter = class TopNErrorParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.TopNErrorParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "n":
                    message.n = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TopNErrorParameter.prototype.axis = 0;
$root.nnabla.TopNErrorParameter.prototype.n = 1;

$root.nnabla.BinaryErrorParameter = class BinaryErrorParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.BinaryErrorParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConfusionMatrixParameter = class ConfusionMatrixParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.ConfusionMatrixParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ConfusionMatrixParameter.prototype.axis = 0;

$root.nnabla.VATNoiseParameter = class VATNoiseParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.VATNoiseParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.VATNoiseParameter.prototype.base_axis = 1;
$root.nnabla.VATNoiseParameter.prototype.eps = 1.0;

$root.nnabla.UnlinkParameter = class UnlinkParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.UnlinkParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SinkParameter = class SinkParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SinkParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "one_input_grad":
                    message.one_input_grad = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SinkParameter.prototype.one_input_grad = true;

$root.nnabla.NmsDetection2dParameter = class NmsDetection2dParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.NmsDetection2dParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "thresh":
                    message.thresh = reader.float();
                    break;
                case "nms":
                    message.nms = reader.float();
                    break;
                case "nms_per_class":
                    message.nms_per_class = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.NmsDetection2dParameter.prototype.thresh = 0.5;
$root.nnabla.NmsDetection2dParameter.prototype.nms = 0.45;
$root.nnabla.NmsDetection2dParameter.prototype.nms_per_class = true;

$root.nnabla.MaxPoolingBackwardParameter = class MaxPoolingBackwardParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MaxPoolingBackwardParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MaxPoolingBackwardParameter.prototype.kernel = null;
$root.nnabla.MaxPoolingBackwardParameter.prototype.stride = null;
$root.nnabla.MaxPoolingBackwardParameter.prototype.ignore_border = true;
$root.nnabla.MaxPoolingBackwardParameter.prototype.pad = null;
$root.nnabla.MaxPoolingBackwardParameter.prototype.channel_last = false;

$root.nnabla.PatchCorrelationParameter = class PatchCorrelationParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.PatchCorrelationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "patch":
                    message.patch = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "shift":
                    message.shift = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "patch_step":
                    message.patch_step = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "shift_step":
                    message.shift_step = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "padding":
                    message.padding = $root.nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.PatchCorrelationParameter.prototype.patch = null;
$root.nnabla.PatchCorrelationParameter.prototype.shift = null;
$root.nnabla.PatchCorrelationParameter.prototype.patch_step = null;
$root.nnabla.PatchCorrelationParameter.prototype.shift_step = null;
$root.nnabla.PatchCorrelationParameter.prototype.padding = null;

$root.nnabla.SgdParameter = class SgdParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SgdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SgdParameter.prototype.lr = 0.001;

$root.nnabla.SgdWParameter = class SgdWParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.SgdWParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "momentum":
                    message.momentum = reader.float();
                    break;
                case "wd":
                    message.wd = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.SgdWParameter.prototype.lr = 0.001;
$root.nnabla.SgdWParameter.prototype.momentum = 0.9;
$root.nnabla.SgdWParameter.prototype.wd = 0.0001;

$root.nnabla.MomentumParameter = class MomentumParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.MomentumParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "momentum":
                    message.momentum = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.MomentumParameter.prototype.lr = 0.001;
$root.nnabla.MomentumParameter.prototype.momentum = 0.9;

$root.nnabla.LarsParameter = class LarsParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.LarsParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "momentum":
                    message.momentum = reader.float();
                    break;
                case "coefficient":
                    message.coefficient = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.LarsParameter.prototype.lr = 0.001;
$root.nnabla.LarsParameter.prototype.momentum = 0.9;
$root.nnabla.LarsParameter.prototype.coefficient = 0.001;
$root.nnabla.LarsParameter.prototype.eps = 1e-06;

$root.nnabla.NesterovParameter = class NesterovParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.NesterovParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "momentum":
                    message.momentum = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.NesterovParameter.prototype.lr = 0.001;
$root.nnabla.NesterovParameter.prototype.momentum = 0.9;

$root.nnabla.AdadeltaParameter = class AdadeltaParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdadeltaParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "decay":
                    message.decay = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdadeltaParameter.prototype.lr = 1.0;
$root.nnabla.AdadeltaParameter.prototype.decay = 0.95;
$root.nnabla.AdadeltaParameter.prototype.eps = 1e-06;

$root.nnabla.AdagradParameter = class AdagradParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdagradParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdagradParameter.prototype.lr = 0.01;
$root.nnabla.AdagradParameter.prototype.eps = 1e-08;

$root.nnabla.AdaBeliefParameter = class AdaBeliefParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdaBeliefParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "wd":
                    message.wd = reader.float();
                    break;
                case "amsgrad":
                    message.amsgrad = reader.bool();
                    break;
                case "weight_decouple":
                    message.weight_decouple = reader.bool();
                    break;
                case "fixed_decay":
                    message.fixed_decay = reader.bool();
                    break;
                case "rectify":
                    message.rectify = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdaBeliefParameter.prototype.alpha = 0.001;
$root.nnabla.AdaBeliefParameter.prototype.beta1 = 0.9;
$root.nnabla.AdaBeliefParameter.prototype.beta2 = 0.999;
$root.nnabla.AdaBeliefParameter.prototype.eps = 1e-08;
$root.nnabla.AdaBeliefParameter.prototype.wd = 0.0;
$root.nnabla.AdaBeliefParameter.prototype.amsgrad = false;
$root.nnabla.AdaBeliefParameter.prototype.weight_decouple = false;
$root.nnabla.AdaBeliefParameter.prototype.fixed_decay = false;
$root.nnabla.AdaBeliefParameter.prototype.rectify = false;

$root.nnabla.RMSpropParameter = class RMSpropParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RMSpropParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "decay":
                    message.decay = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RMSpropParameter.prototype.lr = 0.001;
$root.nnabla.RMSpropParameter.prototype.decay = 0.9;
$root.nnabla.RMSpropParameter.prototype.eps = 1e-08;

$root.nnabla.RMSpropGravesParameter = class RMSpropGravesParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.RMSpropGravesParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "decay":
                    message.decay = reader.float();
                    break;
                case "momentum":
                    message.momentum = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.RMSpropGravesParameter.prototype.lr = 0.0001;
$root.nnabla.RMSpropGravesParameter.prototype.decay = 0.95;
$root.nnabla.RMSpropGravesParameter.prototype.momentum = 0.9;
$root.nnabla.RMSpropGravesParameter.prototype.eps = 0.0001;

$root.nnabla.AdamParameter = class AdamParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdamParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdamParameter.prototype.alpha = 0.001;
$root.nnabla.AdamParameter.prototype.beta1 = 0.9;
$root.nnabla.AdamParameter.prototype.beta2 = 0.999;
$root.nnabla.AdamParameter.prototype.eps = 1e-08;

$root.nnabla.AdamWParameter = class AdamWParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdamWParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "wd":
                    message.wd = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdamWParameter.prototype.alpha = 0.001;
$root.nnabla.AdamWParameter.prototype.beta1 = 0.9;
$root.nnabla.AdamWParameter.prototype.beta2 = 0.999;
$root.nnabla.AdamWParameter.prototype.eps = 1e-08;
$root.nnabla.AdamWParameter.prototype.wd = 0.0001;

$root.nnabla.AdaBoundParameter = class AdaBoundParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdaBoundParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "final_lr":
                    message.final_lr = reader.float();
                    break;
                case "gamma":
                    message.gamma = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdaBoundParameter.prototype.alpha = 0.001;
$root.nnabla.AdaBoundParameter.prototype.beta1 = 0.9;
$root.nnabla.AdaBoundParameter.prototype.beta2 = 0.999;
$root.nnabla.AdaBoundParameter.prototype.eps = 1e-08;
$root.nnabla.AdaBoundParameter.prototype.final_lr = 0.1;
$root.nnabla.AdaBoundParameter.prototype.gamma = 0.001;

$root.nnabla.AdamaxParameter = class AdamaxParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AdamaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AdamaxParameter.prototype.alpha = 0.002;
$root.nnabla.AdamaxParameter.prototype.beta1 = 0.9;
$root.nnabla.AdamaxParameter.prototype.beta2 = 0.999;
$root.nnabla.AdamaxParameter.prototype.eps = 1e-08;

$root.nnabla.AMSGRADParameter = class AMSGRADParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AMSGRADParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "bias_correction":
                    message.bias_correction = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AMSGRADParameter.prototype.alpha = 0.001;
$root.nnabla.AMSGRADParameter.prototype.beta1 = 0.9;
$root.nnabla.AMSGRADParameter.prototype.beta2 = 0.999;
$root.nnabla.AMSGRADParameter.prototype.eps = 1e-08;
$root.nnabla.AMSGRADParameter.prototype.bias_correction = false;

$root.nnabla.AMSBoundParameter = class AMSBoundParameter {

    constructor() {

    }

    static decodeText(reader) {
        const message = new $root.nnabla.AMSBoundParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "eps":
                    message.eps = reader.float();
                    break;
                case "final_lr":
                    message.final_lr = reader.float();
                    break;
                case "gamma":
                    message.gamma = reader.float();
                    break;
                case "bias_correction":
                    message.bias_correction = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.AMSBoundParameter.prototype.alpha = 0.001;
$root.nnabla.AMSBoundParameter.prototype.beta1 = 0.9;
$root.nnabla.AMSBoundParameter.prototype.beta2 = 0.999;
$root.nnabla.AMSBoundParameter.prototype.eps = 1e-08;
$root.nnabla.AMSBoundParameter.prototype.final_lr = 0.1;
$root.nnabla.AMSBoundParameter.prototype.gamma = 0.001;
$root.nnabla.AMSBoundParameter.prototype.bias_correction = false;

