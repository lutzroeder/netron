
export const torch = {};
export const caffe2 = {};

torch.RecordRef = class RecordRef {

    static decodeJson(obj) {
        const message = new torch.RecordRef();
        if (obj.key !== undefined) {
            message.key = obj.key;
        }
        return message;
    }
};

torch.RecordRef.prototype.key = "";

torch.TensorDef = class TensorDef {

    constructor() {
        this.dims = [];
        this.strides = [];
    }

    static decodeJson(obj) {
        const message = new torch.TensorDef();
        if (obj.dims !== undefined) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if (obj.offset !== undefined) {
            message.offset = BigInt(obj.offset);
        }
        if (obj.strides !== undefined) {
            message.strides = obj.strides.map((obj) => BigInt(obj));
        }
        if (obj.requiresGrad !== undefined) {
            message.requires_grad = obj.requiresGrad;
        }
        if (obj.dataType !== undefined) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if (obj.data !== undefined) {
            message.data = torch.RecordRef.decodeJson(obj.data);
        }
        if (obj.device !== undefined) {
            message.device = obj.device;
        }
        if (obj.isQuantized !== undefined) {
            message.is_quantized = obj.isQuantized;
        }
        if (obj.scale !== undefined) {
            message.scale = Number(obj.scale);
        }
        if (obj.zeroPoint !== undefined) {
            message.zero_point = BigInt(obj.zeroPoint);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new torch.AttributeDef();
        message.type = obj.type;
        message.name = obj.name;
        message.id = BigInt(obj.id);
        return message;
    }
};

torch.AttributeDef.prototype.type = "";
torch.AttributeDef.prototype.name = "";
torch.AttributeDef.prototype.id = 0n;

torch.ParameterDef = class ParameterDef {

    static decodeJson(obj) {
        const message = new torch.ParameterDef();
        if (obj.isBuffer !== undefined) {
            message.is_buffer = obj.isBuffer;
        }
        if (obj.tensorId !== undefined) {
            message.tensor_id = BigInt(obj.tensorId);
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        return message;
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

    static decodeJson(obj) {
        const message = new torch.ModuleDef();
        if (obj.submodules !== undefined) {
            message.submodules = obj.submodules.map((obj) => torch.ModuleDef.decodeJson(obj));
        }
        if (obj.torchscriptArena !== undefined) {
            message.torchscript_arena = torch.RecordRef.decodeJson(obj.torchscriptArena);
        }
        if (obj.caffe2Nets !== undefined) {
            message.caffe2_nets = obj.caffe2Nets.map((obj) => caffe2.NetDef.decodeJson(obj));
        }
        if (obj.pickleArena !== undefined) {
            message.pickle_arena = torch.RecordRef.decodeJson(obj.pickleArena);
        }
        if (obj.cppArena !== undefined) {
            message.cpp_arena = torch.RecordRef.decodeJson(obj.cppArena);
        }
        if (obj.parameters !== undefined) {
            message.parameters = obj.parameters.map((obj) => torch.ParameterDef.decodeJson(obj));
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.optimize !== undefined) {
            message.optimize = obj.optimize;
        }
        if (obj.attributes !== undefined) {
            message.attributes = obj.attributes.map((obj) => torch.AttributeDef.decodeJson(obj));
        }
        if (obj.getStateAttributeId !== undefined) {
            message.get_state_attribute_id = BigInt(obj.getStateAttributeId);
        }
        if (obj.torchscriptDebugArena !== undefined) {
            message.torchscript_debug_arena = torch.RecordRef.decodeJson(obj.torchscriptDebugArena);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new torch.LibDef();
        if (obj.torchscriptArena !== undefined) {
            message.torchscript_arena = torch.RecordRef.decodeJson(obj.torchscriptArena);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new torch.ModelDef();
        if (obj.protoVersion !== undefined) {
            message.proto_version = BigInt(obj.protoVersion);
        }
        if (obj.mainModule !== undefined) {
            message.main_module = torch.ModuleDef.decodeJson(obj.mainModule);
        }
        if (obj.producerName !== undefined) {
            message.producer_name = obj.producerName;
        }
        if (obj.producerVersion !== undefined) {
            message.producer_version = obj.producerVersion;
        }
        if (obj.tensors !== undefined) {
            message.tensors = obj.tensors.map((obj) => torch.TensorDef.decodeJson(obj));
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.TensorProto();
        if (obj.dims !== undefined) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if (obj.dataType !== undefined) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if (obj.dataFormat !== undefined) {
            message.data_format = Number(obj.dataFormat);
        }
        if (obj.floatData !== undefined) {
            message.float_data = obj.floatData.map((obj) => Number(obj));
        }
        if (obj.int32Data !== undefined) {
            message.int32_data = obj.int32Data.map((obj) => Number(obj));
        }
        if (obj.byteData !== undefined) {
            message.byte_data = new Uint8Array(atob(obj.byteData));
        }
        if (obj.stringData !== undefined) {
            message.string_data = obj.stringData.map((obj) => new Uint8Array(atob(obj)));
        }
        if (obj.doubleData !== undefined) {
            message.double_data = obj.doubleData.map((obj) => Number(obj));
        }
        if (obj.int64Data !== undefined) {
            message.int64_data = obj.int64Data.map((obj) => BigInt(obj));
        }
        if (obj.rawData !== undefined) {
            message.raw_data = new Uint8Array(atob(obj.rawData));
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.deviceDetail !== undefined) {
            message.device_detail = caffe2.DeviceOption.decodeJson(obj.deviceDetail);
        }
        if (obj.segment !== undefined) {
            message.segment = caffe2.TensorProto.Segment.decodeJson(obj.segment);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.TensorProto.Segment();
        message.begin = BigInt(obj.begin);
        message.end = BigInt(obj.end);
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.QTensorProto();
        if (obj.dims !== undefined) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        message.precision = Number(obj.precision);
        message.scale = Number(obj.scale);
        message.bias = Number(obj.bias);
        message.is_signed = obj.isSigned;
        if (obj.data !== undefined) {
            message.data = obj.data.map((obj) => Number(obj));
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.dataType !== undefined) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if (obj.scales !== undefined) {
            message.scales = obj.scales.map((obj) => Number(obj));
        }
        if (obj.biases !== undefined) {
            message.biases = obj.biases.map((obj) => Number(obj));
        }
        if (obj.axis !== undefined) {
            message.axis = Number(obj.axis);
        }
        if (obj.isMultiparam !== undefined) {
            message.is_multiparam = obj.isMultiparam;
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.TensorProtos();
        if (obj.protos !== undefined) {
            message.protos = obj.protos.map((obj) => caffe2.TensorProto.decodeJson(obj));
        }
        return message;
    }
};

caffe2.TensorShape = class TensorShape {

    constructor() {
        this.dims = [];
        this.unknown_dims = [];
    }

    static decodeJson(obj) {
        const message = new caffe2.TensorShape();
        if (obj.dims !== undefined) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if (obj.dataType !== undefined) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if (obj.unknownDims !== undefined) {
            message.unknown_dims = obj.unknownDims.map((obj) => Number(obj));
        }
        if (obj.unknownShape !== undefined) {
            message.unknown_shape = obj.unknownShape;
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        return message;
    }
};

caffe2.TensorShape.prototype.data_type = 1;
caffe2.TensorShape.prototype.unknown_shape = false;
caffe2.TensorShape.prototype.name = "";

caffe2.TensorShapes = class TensorShapes {

    constructor() {
        this.shapes = [];
    }

    static decodeJson(obj) {
        const message = new caffe2.TensorShapes();
        if (obj.shapes !== undefined) {
            message.shapes = obj.shapes.map((obj) => caffe2.TensorShape.decodeJson(obj));
        }
        return message;
    }
};

caffe2.TensorBoundShape = class TensorBoundShape {

    constructor() {
        this.dim_type = [];
    }

    static decodeJson(obj) {
        const message = new caffe2.TensorBoundShape();
        if (obj.shape !== undefined) {
            message.shape = caffe2.TensorShape.decodeJson(obj.shape);
        }
        if (obj.dimType !== undefined) {
            message.dim_type = obj.dimType.map((key) => caffe2.TensorBoundShape.DimType[key]);
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.shapeIsFinal !== undefined) {
            message.shape_is_final = obj.shapeIsFinal;
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.TensorBoundShapes();
        if (obj.shapes !== undefined) {
            message.shapes = obj.shapes.map((obj) => caffe2.TensorBoundShape.decodeJson(obj));
        }
        if (obj.maxBatchSize !== undefined) {
            message.max_batch_size = BigInt(obj.maxBatchSize);
        }
        if (obj.maxFeatureLen !== undefined) {
            message.max_feature_len = BigInt(obj.maxFeatureLen);
        }
        return message;
    }
};

caffe2.TensorBoundShapes.prototype.max_batch_size = 0n;
caffe2.TensorBoundShapes.prototype.max_feature_len = 0n;

caffe2.AOTConfig = class AOTConfig {

    static decodeJson(obj) {
        const message = new caffe2.AOTConfig();
        message.max_batch_size = BigInt(obj.maxBatchSize);
        message.max_seq_size = BigInt(obj.maxSeqSize);
        message.in_batch_broadcast = obj.inBatchBroadcast;
        if (obj.onnxifiBlacklistOps !== undefined) {
            message.onnxifi_blacklist_ops = obj.onnxifiBlacklistOps;
        }
        if (obj.onnxifiMinOps !== undefined) {
            message.onnxifi_min_ops = Number(obj.onnxifiMinOps);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.Argument();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.f !== undefined) {
            message.f = Number(obj.f);
        }
        if (obj.i !== undefined) {
            message.i = BigInt(obj.i);
        }
        if (obj.s !== undefined) {
            message.s = new Uint8Array(atob(obj.s));
        }
        if (obj.t !== undefined) {
            message.t = caffe2.TensorProto.decodeJson(obj.t);
        }
        if (obj.n !== undefined) {
            message.n = caffe2.NetDef.decodeJson(obj.n);
        }
        if (obj.floats !== undefined) {
            message.floats = obj.floats.map((obj) => Number(obj));
        }
        if (obj.ints !== undefined) {
            message.ints = obj.ints.map((obj) => BigInt(obj));
        }
        if (obj.strings !== undefined) {
            message.strings = obj.strings.map((obj) => new Uint8Array(atob(obj)));
        }
        if (obj.tensors !== undefined) {
            message.tensors = obj.tensors.map((obj) => caffe2.TensorProto.decodeJson(obj));
        }
        if (obj.nets !== undefined) {
            message.nets = obj.nets.map((obj) => caffe2.NetDef.decodeJson(obj));
        }
        if (obj.qtensors !== undefined) {
            message.qtensors = obj.qtensors.map((obj) => caffe2.QTensorProto.decodeJson(obj));
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.DeviceOption();
        if (obj.deviceType !== undefined) {
            message.device_type = Number(obj.deviceType);
        }
        if (obj.deviceId !== undefined) {
            message.device_id = Number(obj.deviceId);
        }
        if (obj.randomSeed !== undefined) {
            message.random_seed = Number(obj.randomSeed);
        }
        if (obj.nodeName !== undefined) {
            message.node_name = obj.nodeName;
        }
        if (obj.numaNodeId !== undefined) {
            message.numa_node_id = Number(obj.numaNodeId);
        }
        if (obj.extraInfo !== undefined) {
            message.extra_info = obj.extraInfo;
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.OperatorDef();
        if (obj.input !== undefined) {
            message.input = obj.input;
        }
        if (obj.output !== undefined) {
            message.output = obj.output;
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.type !== undefined) {
            message.type = obj.type;
        }
        if (obj.arg !== undefined) {
            message.arg = obj.arg.map((obj) => caffe2.Argument.decodeJson(obj));
        }
        if (obj.deviceOption !== undefined) {
            message.device_option = caffe2.DeviceOption.decodeJson(obj.deviceOption);
        }
        if (obj.engine !== undefined) {
            message.engine = obj.engine;
        }
        if (obj.controlInput !== undefined) {
            message.control_input = obj.controlInput;
        }
        if (obj.isGradientOp !== undefined) {
            message.is_gradient_op = obj.isGradientOp;
        }
        if (obj.debugInfo !== undefined) {
            message.debug_info = obj.debugInfo;
        }
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.opVersion !== undefined) {
            message.op_version = BigInt(obj.opVersion);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.MapFieldEntry();
        message.key = obj.key;
        message.val = obj.val;
        return message;
    }
};

caffe2.MapFieldEntry.prototype.key = "";
caffe2.MapFieldEntry.prototype.val = "";

caffe2.BackendOptions = class BackendOptions {

    constructor() {
        this.option = [];
    }

    static decodeJson(obj) {
        const message = new caffe2.BackendOptions();
        message.backend_name = obj.backendName;
        if (obj.option !== undefined) {
            message.option = obj.option.map((obj) => caffe2.MapFieldEntry.decodeJson(obj));
        }
        return message;
    }
};

caffe2.BackendOptions.prototype.backend_name = "";

caffe2.PartitionInfo = class PartitionInfo {

    constructor() {
        this.device_id = [];
        this.backend_options = [];
    }

    static decodeJson(obj) {
        const message = new caffe2.PartitionInfo();
        message.name = obj.name;
        if (obj.deviceId !== undefined) {
            message.device_id = obj.deviceId.map((obj) => Number(obj));
        }
        if (obj.extraInfo !== undefined) {
            message.extra_info = obj.extraInfo;
        }
        if (obj.backendOptions !== undefined) {
            message.backend_options = obj.backendOptions.map((obj) => caffe2.BackendOptions.decodeJson(obj));
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.NetDef();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.op !== undefined) {
            message.op = obj.op.map((obj) => caffe2.OperatorDef.decodeJson(obj));
        }
        if (obj.type !== undefined) {
            message.type = obj.type;
        }
        if (obj.numWorkers !== undefined) {
            message.num_workers = Number(obj.numWorkers);
        }
        if (obj.deviceOption !== undefined) {
            message.device_option = caffe2.DeviceOption.decodeJson(obj.deviceOption);
        }
        if (obj.arg !== undefined) {
            message.arg = obj.arg.map((obj) => caffe2.Argument.decodeJson(obj));
        }
        if (obj.externalInput !== undefined) {
            message.external_input = obj.externalInput;
        }
        if (obj.externalOutput !== undefined) {
            message.external_output = obj.externalOutput;
        }
        if (obj.partitionInfo !== undefined) {
            message.partition_info = obj.partitionInfo.map((obj) => caffe2.PartitionInfo.decodeJson(obj));
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.ExecutionStep();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.substep !== undefined) {
            message.substep = obj.substep.map((obj) => caffe2.ExecutionStep.decodeJson(obj));
        }
        if (obj.network !== undefined) {
            message.network = obj.network;
        }
        if (obj.numIter !== undefined) {
            message.num_iter = BigInt(obj.numIter);
        }
        if (obj.criteriaNetwork !== undefined) {
            message.criteria_network = obj.criteriaNetwork;
        }
        if (obj.reportNet !== undefined) {
            message.report_net = obj.reportNet;
        }
        if (obj.reportInterval !== undefined) {
            message.report_interval = Number(obj.reportInterval);
        }
        if (obj.runEveryMs !== undefined) {
            message.run_every_ms = BigInt(obj.runEveryMs);
        }
        if (obj.concurrentSubsteps !== undefined) {
            message.concurrent_substeps = obj.concurrentSubsteps;
        }
        if (obj.shouldStopBlob !== undefined) {
            message.should_stop_blob = obj.shouldStopBlob;
        }
        if (obj.onlyOnce !== undefined) {
            message.only_once = obj.onlyOnce;
        }
        if (obj.createWorkspace !== undefined) {
            message.create_workspace = obj.createWorkspace;
        }
        if (obj.numConcurrentInstances !== undefined) {
            message.num_concurrent_instances = Number(obj.numConcurrentInstances);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.PlanDef();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.network !== undefined) {
            message.network = obj.network.map((obj) => caffe2.NetDef.decodeJson(obj));
        }
        if (obj.executionStep !== undefined) {
            message.execution_step = obj.executionStep.map((obj) => caffe2.ExecutionStep.decodeJson(obj));
        }
        return message;
    }
};

caffe2.PlanDef.prototype.name = "";

caffe2.BlobProto = class BlobProto {

    static decodeJson(obj) {
        const message = new caffe2.BlobProto();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.type !== undefined) {
            message.type = obj.type;
        }
        if (obj.tensor !== undefined) {
            message.tensor = caffe2.TensorProto.decodeJson(obj.tensor);
        }
        if (obj.content !== undefined) {
            message.content = new Uint8Array(atob(obj.content));
        }
        if (obj.qtensor !== undefined) {
            message.qtensor = caffe2.QTensorProto.decodeJson(obj.qtensor);
        }
        if (obj.contentNumChunks !== undefined) {
            message.content_num_chunks = Number(obj.contentNumChunks);
        }
        if (obj.contentChunkId !== undefined) {
            message.content_chunk_id = Number(obj.contentChunkId);
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.DBReaderProto();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.source !== undefined) {
            message.source = obj.source;
        }
        if (obj.dbType !== undefined) {
            message.db_type = obj.dbType;
        }
        if (obj.key !== undefined) {
            message.key = obj.key;
        }
        return message;
    }
};

caffe2.DBReaderProto.prototype.name = "";
caffe2.DBReaderProto.prototype.source = "";
caffe2.DBReaderProto.prototype.db_type = "";
caffe2.DBReaderProto.prototype.key = "";

caffe2.BlobSerializationOptions = class BlobSerializationOptions {

    static decodeJson(obj) {
        const message = new caffe2.BlobSerializationOptions();
        if (obj.blobNameRegex !== undefined) {
            message.blob_name_regex = obj.blobNameRegex;
        }
        if (obj.chunkSize !== undefined) {
            message.chunk_size = BigInt(obj.chunkSize);
        }
        if (obj.floatFormat !== undefined) {
            message.float_format = caffe2.BlobSerializationOptions.FloatFormat[obj.floatFormat];
        }
        return message;
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

    static decodeJson(obj) {
        const message = new caffe2.SerializationOptions();
        if (obj.options !== undefined) {
            message.options = obj.options.map((obj) => caffe2.BlobSerializationOptions.decodeJson(obj));
        }
        return message;
    }
};
