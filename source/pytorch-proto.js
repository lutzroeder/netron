
export const torch = {};
export const caffe2 = {};

torch.RecordRef = class RecordRef {

    static decodeJson(/* reader */) {
    }
};

torch.RecordRef.prototype.key = "";

torch.TensorDef = class TensorDef {

    constructor() {
        this.dims = [];
        this.strides = [];
    }

    static decodeJson(/* reader */) {
    }
};

torch.TensorDef.prototype.offset = 0n;
torch.TensorDef.prototype.requires_grad = false;
torch.TensorDef.prototype.data_type = 0;
torch.TensorDef.prototype.data = null;
torch.TensorDef.prototype.device = "";
torch.TensorDef.prototype.is_quantized = false;
torch.TensorDef.prototype.scale = 0;
torch.TensorDef.prototype.zero_point = 0n;

torch.AttributeDef = class AttributeDef {

    static decodeJson(/* reader */) {
    }
};

torch.AttributeDef.prototype.type = "";
torch.AttributeDef.prototype.name = "";
torch.AttributeDef.prototype.id = 0n;

torch.ParameterDef = class ParameterDef {

    static decodeJson(/* reader */) {
    }
};

torch.ParameterDef.prototype.is_buffer = false;
torch.ParameterDef.prototype.tensor_id = 0n;
torch.ParameterDef.prototype.name = "";

torch.ModuleDef = class ModuleDef {

    constructor() {
        this.submodules = [];
        this.caffe2_nets = [];
        this.parameters = [];
        this.attributes = [];
    }

    static decodeJson(/* reader */) {
    }
};

torch.ModuleDef.prototype.torchscript_arena = null;
torch.ModuleDef.prototype.pickle_arena = null;
torch.ModuleDef.prototype.cpp_arena = null;
torch.ModuleDef.prototype.name = "";
torch.ModuleDef.prototype.optimize = false;
torch.ModuleDef.prototype.get_state_attribute_id = 0n;
torch.ModuleDef.prototype.torchscript_debug_arena = null;

torch.LibDef = class LibDef {

    static decodeJson(/* reader */) {
    }
};

torch.LibDef.prototype.torchscript_arena = null;

torch.ProtoVersion = {
    "PROTO_VERSION_NEWEST": 6
};

torch.ModelDef = class ModelDef {

    constructor() {
        this.tensors = [];
    }

    static decodeJson(/* reader */) {
    }
};

torch.ModelDef.prototype.proto_version = 0n;
torch.ModelDef.prototype.main_module = null;
torch.ModelDef.prototype.producer_name = "";
torch.ModelDef.prototype.producer_version = "";

caffe2.TensorProto = class TensorProto {

    constructor() {
        this.dims = [];
        this.float_data = [];
        this.int32_data = [];
        this.string_data = [];
        this.double_data = [];
        this.int64_data = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorProto.prototype.data_type = 1;
caffe2.TensorProto.prototype.data_format = 0;
caffe2.TensorProto.prototype.byte_data = new Uint8Array([]);
caffe2.TensorProto.prototype.raw_data = new Uint8Array([]);
caffe2.TensorProto.prototype.name = "";
caffe2.TensorProto.prototype.device_detail = null;
caffe2.TensorProto.prototype.segment = null;

caffe2.TensorProto.DataType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "INT32": 2,
    "BYTE": 3,
    "STRING": 4,
    "BOOL": 5,
    "UINT8": 6,
    "INT8": 7,
    "UINT16": 8,
    "INT16": 9,
    "INT64": 10,
    "FLOAT16": 12,
    "DOUBLE": 13,
    "ZERO_COLLISION_HASH": 14,
    "REBATCHING_BUFFER": 15
};

caffe2.TensorProto.SerializationFormat = {
    "FMT_PROTOBUF": 0,
    "FMT_BFLOAT16": 1
};

caffe2.TensorProto.Segment = class Segment {

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorProto.Segment.prototype.begin = 0n;
caffe2.TensorProto.Segment.prototype.end = 0n;

caffe2.QTensorProto = class QTensorProto {

    constructor() {
        this.dims = [];
        this.data = [];
        this.scales = [];
        this.biases = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.QTensorProto.prototype.precision = 0;
caffe2.QTensorProto.prototype.scale = 0;
caffe2.QTensorProto.prototype.bias = 0;
caffe2.QTensorProto.prototype.is_signed = false;
caffe2.QTensorProto.prototype.name = "";
caffe2.QTensorProto.prototype.data_type = 2;
caffe2.QTensorProto.prototype.axis = 0;
caffe2.QTensorProto.prototype.is_multiparam = false;

caffe2.TensorProtos = class TensorProtos {

    constructor() {
        this.protos = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorShape = class TensorShape {

    constructor() {
        this.dims = [];
        this.unknown_dims = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorShape.prototype.data_type = 1;
caffe2.TensorShape.prototype.unknown_shape = false;
caffe2.TensorShape.prototype.name = "";

caffe2.TensorShapes = class TensorShapes {

    constructor() {
        this.shapes = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorBoundShape = class TensorBoundShape {

    constructor() {
        this.dim_type = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorBoundShape.prototype.shape = null;
caffe2.TensorBoundShape.prototype.name = "";
caffe2.TensorBoundShape.prototype.shape_is_final = false;

caffe2.TensorBoundShape.DimType = {
    "UNKNOWN": 0,
    "CONSTANT": 1,
    "BATCH": 2,
    "BATCH_OF_FEATURE_MAX": 3,
    "BATCH_OF_FEATURE_MAX_DEFAULT": 4,
    "FEATURE_MAX": 5,
    "FEATURE_MAX_DEFAULT": 6
};

caffe2.TensorBoundShapes = class TensorBoundShapes {

    constructor() {
        this.shapes = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.TensorBoundShapes.prototype.max_batch_size = 0n;
caffe2.TensorBoundShapes.prototype.max_feature_len = 0n;

caffe2.AOTConfig = class AOTConfig {

    static decodeJson(/* reader */) {
    }
};

caffe2.AOTConfig.prototype.max_batch_size = 0n;
caffe2.AOTConfig.prototype.max_seq_size = 0n;
caffe2.AOTConfig.prototype.in_batch_broadcast = false;
caffe2.AOTConfig.prototype.onnxifi_blacklist_ops = "";
caffe2.AOTConfig.prototype.onnxifi_min_ops = 0;

caffe2.Argument = class Argument {

    constructor() {
        this.floats = [];
        this.ints = [];
        this.strings = [];
        this.tensors = [];
        this.nets = [];
        this.qtensors = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.Argument.prototype.name = "";
caffe2.Argument.prototype.f = 0;
caffe2.Argument.prototype.i = 0n;
caffe2.Argument.prototype.s = new Uint8Array([]);
caffe2.Argument.prototype.t = null;
caffe2.Argument.prototype.n = null;

caffe2.DeviceTypeProto = {
    "PROTO_CPU": 0,
    "PROTO_CUDA": 1,
    "PROTO_MKLDNN": 2,
    "PROTO_OPENGL": 3,
    "PROTO_OPENCL": 4,
    "PROTO_IDEEP": 5,
    "PROTO_HIP": 6,
    "PROTO_FPGA": 7,
    "PROTO_MAIA": 8,
    "PROTO_XLA": 9,
    "PROTO_MPS": 10,
    "PROTO_COMPILE_TIME_MAX_DEVICE_TYPES": 11
};

caffe2.DeviceOption = class DeviceOption {

    constructor() {
        this.extra_info = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.DeviceOption.prototype.device_type = 0;
caffe2.DeviceOption.prototype.device_id = 0;
caffe2.DeviceOption.prototype.random_seed = 0;
caffe2.DeviceOption.prototype.node_name = "";
caffe2.DeviceOption.prototype.numa_node_id = 0;

caffe2.OperatorDef = class OperatorDef {

    constructor() {
        this.input = [];
        this.output = [];
        this.arg = [];
        this.control_input = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.OperatorDef.prototype.name = "";
caffe2.OperatorDef.prototype.type = "";
caffe2.OperatorDef.prototype.device_option = null;
caffe2.OperatorDef.prototype.engine = "";
caffe2.OperatorDef.prototype.is_gradient_op = false;
caffe2.OperatorDef.prototype.debug_info = "";
caffe2.OperatorDef.prototype.domain = "";
caffe2.OperatorDef.prototype.op_version = 0n;

caffe2.MapFieldEntry = class MapFieldEntry {

    static decodeJson(/* reader */) {
    }
};

caffe2.MapFieldEntry.prototype.key = "";
caffe2.MapFieldEntry.prototype.val = "";

caffe2.BackendOptions = class BackendOptions {

    constructor() {
        this.option = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.BackendOptions.prototype.backend_name = "";

caffe2.PartitionInfo = class PartitionInfo {

    constructor() {
        this.device_id = [];
        this.backend_options = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.PartitionInfo.prototype.name = "";
caffe2.PartitionInfo.prototype.extra_info = "";

caffe2.NetDef = class NetDef {

    constructor() {
        this.op = [];
        this.arg = [];
        this.external_input = [];
        this.external_output = [];
        this.partition_info = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.NetDef.prototype.name = "";
caffe2.NetDef.prototype.type = "";
caffe2.NetDef.prototype.num_workers = 0;
caffe2.NetDef.prototype.device_option = null;

caffe2.ExecutionStep = class ExecutionStep {

    constructor() {
        this.substep = [];
        this.network = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.ExecutionStep.prototype.name = "";
caffe2.ExecutionStep.prototype.num_iter = 0n;
caffe2.ExecutionStep.prototype.criteria_network = "";
caffe2.ExecutionStep.prototype.report_net = "";
caffe2.ExecutionStep.prototype.report_interval = 0;
caffe2.ExecutionStep.prototype.run_every_ms = 0n;
caffe2.ExecutionStep.prototype.concurrent_substeps = false;
caffe2.ExecutionStep.prototype.should_stop_blob = "";
caffe2.ExecutionStep.prototype.only_once = false;
caffe2.ExecutionStep.prototype.create_workspace = false;
caffe2.ExecutionStep.prototype.num_concurrent_instances = 0;

caffe2.PlanDef = class PlanDef {

    constructor() {
        this.network = [];
        this.execution_step = [];
    }

    static decodeJson(/* reader */) {
    }
};

caffe2.PlanDef.prototype.name = "";

caffe2.BlobProto = class BlobProto {

    static decodeJson(/* reader */) {
    }
};

caffe2.BlobProto.prototype.name = "";
caffe2.BlobProto.prototype.type = "";
caffe2.BlobProto.prototype.tensor = null;
caffe2.BlobProto.prototype.content = new Uint8Array([]);
caffe2.BlobProto.prototype.qtensor = null;
caffe2.BlobProto.prototype.content_num_chunks = 0;
caffe2.BlobProto.prototype.content_chunk_id = 0;

caffe2.DBReaderProto = class DBReaderProto {

    static decodeJson(/* reader */) {
    }
};

caffe2.DBReaderProto.prototype.name = "";
caffe2.DBReaderProto.prototype.source = "";
caffe2.DBReaderProto.prototype.db_type = "";
caffe2.DBReaderProto.prototype.key = "";

caffe2.BlobSerializationOptions = class BlobSerializationOptions {

    static decodeJson(/* reader */) {
    }
};

caffe2.BlobSerializationOptions.prototype.blob_name_regex = "";
caffe2.BlobSerializationOptions.prototype.chunk_size = 0n;
caffe2.BlobSerializationOptions.prototype.float_format = 0;

caffe2.BlobSerializationOptions.FloatFormat = {
    "FLOAT_DEFAULT": 0,
    "FLOAT_PROTOBUF": 1,
    "FLOAT_BFLOAT16": 2
};

caffe2.SerializationOptions = class SerializationOptions {

    constructor() {
        this.options = [];
    }

    static decodeJson(/* reader */) {
    }
};
