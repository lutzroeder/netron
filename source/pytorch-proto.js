
export const torch = {};
export const caffe2 = {};

torch.RecordRef = class RecordRef {

    static decodeJson(obj) {
        const message = new torch.RecordRef();
        if ('key' in obj) {
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
        if ('dims' in obj) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if ('offset' in obj) {
            message.offset = BigInt(obj.offset);
        }
        if ('strides' in obj) {
            message.strides = obj.strides.map((obj) => BigInt(obj));
        }
        if ('requiresGrad' in obj) {
            message.requires_grad = obj.requiresGrad;
        }
        if ('dataType' in obj) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if ('data' in obj) {
            message.data = torch.RecordRef.decodeJson(obj.data);
        }
        if ('device' in obj) {
            message.device = obj.device;
        }
        if ('isQuantized' in obj) {
            message.is_quantized = obj.isQuantized;
        }
        if ('scale' in obj) {
            message.scale = Number(obj.scale);
        }
        if ('zeroPoint' in obj) {
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
        if ('isBuffer' in obj) {
            message.is_buffer = obj.isBuffer;
        }
        if ('tensorId' in obj) {
            message.tensor_id = BigInt(obj.tensorId);
        }
        if ('name' in obj) {
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
        if ('submodules' in obj) {
            message.submodules = obj.submodules.map((obj) => torch.ModuleDef.decodeJson(obj));
        }
        if ('torchscriptArena' in obj) {
            message.torchscript_arena = torch.RecordRef.decodeJson(obj.torchscriptArena);
        }
        if ('caffe2Nets' in obj) {
            message.caffe2_nets = obj.caffe2Nets.map((obj) => caffe2.NetDef.decodeJson(obj));
        }
        if ('pickleArena' in obj) {
            message.pickle_arena = torch.RecordRef.decodeJson(obj.pickleArena);
        }
        if ('cppArena' in obj) {
            message.cpp_arena = torch.RecordRef.decodeJson(obj.cppArena);
        }
        if ('parameters' in obj) {
            message.parameters = obj.parameters.map((obj) => torch.ParameterDef.decodeJson(obj));
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('optimize' in obj) {
            message.optimize = obj.optimize;
        }
        if ('attributes' in obj) {
            message.attributes = obj.attributes.map((obj) => torch.AttributeDef.decodeJson(obj));
        }
        if ('getStateAttributeId' in obj) {
            message.get_state_attribute_id = BigInt(obj.getStateAttributeId);
        }
        if ('torchscriptDebugArena' in obj) {
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
        if ('torchscriptArena' in obj) {
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
        if ('protoVersion' in obj) {
            message.proto_version = BigInt(obj.protoVersion);
        }
        if ('mainModule' in obj) {
            message.main_module = torch.ModuleDef.decodeJson(obj.mainModule);
        }
        if ('producerName' in obj) {
            message.producer_name = obj.producerName;
        }
        if ('producerVersion' in obj) {
            message.producer_version = obj.producerVersion;
        }
        if ('tensors' in obj) {
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
        if ('dims' in obj) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if ('dataType' in obj) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if ('dataFormat' in obj) {
            message.data_format = Number(obj.dataFormat);
        }
        if ('floatData' in obj) {
            message.float_data = obj.floatData.map((obj) => Number(obj));
        }
        if ('int32Data' in obj) {
            message.int32_data = obj.int32Data.map((obj) => Number(obj));
        }
        if ('byteData' in obj) {
            message.byte_data = new Uint8Array(atob(obj.byteData));
        }
        if ('stringData' in obj) {
            message.string_data = obj.stringData.map((obj) => new Uint8Array(atob(obj)));
        }
        if ('doubleData' in obj) {
            message.double_data = obj.doubleData.map((obj) => Number(obj));
        }
        if ('int64Data' in obj) {
            message.int64_data = obj.int64Data.map((obj) => BigInt(obj));
        }
        if ('rawData' in obj) {
            message.raw_data = new Uint8Array(atob(obj.rawData));
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('deviceDetail' in obj) {
            message.device_detail = caffe2.DeviceOption.decodeJson(obj.deviceDetail);
        }
        if ('segment' in obj) {
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
        if ('dims' in obj) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        message.precision = Number(obj.precision);
        message.scale = Number(obj.scale);
        message.bias = Number(obj.bias);
        message.is_signed = obj.isSigned;
        if ('data' in obj) {
            message.data = obj.data.map((obj) => Number(obj));
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('dataType' in obj) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if ('scales' in obj) {
            message.scales = obj.scales.map((obj) => Number(obj));
        }
        if ('biases' in obj) {
            message.biases = obj.biases.map((obj) => Number(obj));
        }
        if ('axis' in obj) {
            message.axis = Number(obj.axis);
        }
        if ('isMultiparam' in obj) {
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
        if ('protos' in obj) {
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
        if ('dims' in obj) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if ('dataType' in obj) {
            message.data_type = caffe2.TensorProto.DataType[obj.dataType];
        }
        if ('unknownDims' in obj) {
            message.unknown_dims = obj.unknownDims.map((obj) => Number(obj));
        }
        if ('unknownShape' in obj) {
            message.unknown_shape = obj.unknownShape;
        }
        if ('name' in obj) {
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
        if ('shapes' in obj) {
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
        if ('shape' in obj) {
            message.shape = caffe2.TensorShape.decodeJson(obj.shape);
        }
        if ('dimType' in obj) {
            message.dim_type = obj.dimType.map((key) => caffe2.TensorBoundShape.DimType[key]);
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('shapeIsFinal' in obj) {
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
        if ('shapes' in obj) {
            message.shapes = obj.shapes.map((obj) => caffe2.TensorBoundShape.decodeJson(obj));
        }
        if ('maxBatchSize' in obj) {
            message.max_batch_size = BigInt(obj.maxBatchSize);
        }
        if ('maxFeatureLen' in obj) {
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
        if ('onnxifiBlacklistOps' in obj) {
            message.onnxifi_blacklist_ops = obj.onnxifiBlacklistOps;
        }
        if ('onnxifiMinOps' in obj) {
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
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('f' in obj) {
            message.f = Number(obj.f);
        }
        if ('i' in obj) {
            message.i = BigInt(obj.i);
        }
        if ('s' in obj) {
            message.s = new Uint8Array(atob(obj.s));
        }
        if ('t' in obj) {
            message.t = caffe2.TensorProto.decodeJson(obj.t);
        }
        if ('n' in obj) {
            message.n = caffe2.NetDef.decodeJson(obj.n);
        }
        if ('floats' in obj) {
            message.floats = obj.floats.map((obj) => Number(obj));
        }
        if ('ints' in obj) {
            message.ints = obj.ints.map((obj) => BigInt(obj));
        }
        if ('strings' in obj) {
            message.strings = obj.strings.map((obj) => new Uint8Array(atob(obj)));
        }
        if ('tensors' in obj) {
            message.tensors = obj.tensors.map((obj) => caffe2.TensorProto.decodeJson(obj));
        }
        if ('nets' in obj) {
            message.nets = obj.nets.map((obj) => caffe2.NetDef.decodeJson(obj));
        }
        if ('qtensors' in obj) {
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
        if ('deviceType' in obj) {
            message.device_type = Number(obj.deviceType);
        }
        if ('deviceId' in obj) {
            message.device_id = Number(obj.deviceId);
        }
        if ('randomSeed' in obj) {
            message.random_seed = Number(obj.randomSeed);
        }
        if ('nodeName' in obj) {
            message.node_name = obj.nodeName;
        }
        if ('numaNodeId' in obj) {
            message.numa_node_id = Number(obj.numaNodeId);
        }
        if ('extraInfo' in obj) {
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
        if ('input' in obj) {
            message.input = obj.input;
        }
        if ('output' in obj) {
            message.output = obj.output;
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('type' in obj) {
            message.type = obj.type;
        }
        if ('arg' in obj) {
            message.arg = obj.arg.map((obj) => caffe2.Argument.decodeJson(obj));
        }
        if ('deviceOption' in obj) {
            message.device_option = caffe2.DeviceOption.decodeJson(obj.deviceOption);
        }
        if ('engine' in obj) {
            message.engine = obj.engine;
        }
        if ('controlInput' in obj) {
            message.control_input = obj.controlInput;
        }
        if ('isGradientOp' in obj) {
            message.is_gradient_op = obj.isGradientOp;
        }
        if ('debugInfo' in obj) {
            message.debug_info = obj.debugInfo;
        }
        if ('domain' in obj) {
            message.domain = obj.domain;
        }
        if ('opVersion' in obj) {
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
        if ('option' in obj) {
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
        if ('deviceId' in obj) {
            message.device_id = obj.deviceId.map((obj) => Number(obj));
        }
        if ('extraInfo' in obj) {
            message.extra_info = obj.extraInfo;
        }
        if ('backendOptions' in obj) {
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
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('op' in obj) {
            message.op = obj.op.map((obj) => caffe2.OperatorDef.decodeJson(obj));
        }
        if ('type' in obj) {
            message.type = obj.type;
        }
        if ('numWorkers' in obj) {
            message.num_workers = Number(obj.numWorkers);
        }
        if ('deviceOption' in obj) {
            message.device_option = caffe2.DeviceOption.decodeJson(obj.deviceOption);
        }
        if ('arg' in obj) {
            message.arg = obj.arg.map((obj) => caffe2.Argument.decodeJson(obj));
        }
        if ('externalInput' in obj) {
            message.external_input = obj.externalInput;
        }
        if ('externalOutput' in obj) {
            message.external_output = obj.externalOutput;
        }
        if ('partitionInfo' in obj) {
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
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('substep' in obj) {
            message.substep = obj.substep.map((obj) => caffe2.ExecutionStep.decodeJson(obj));
        }
        if ('network' in obj) {
            message.network = obj.network;
        }
        if ('numIter' in obj) {
            message.num_iter = BigInt(obj.numIter);
        }
        if ('criteriaNetwork' in obj) {
            message.criteria_network = obj.criteriaNetwork;
        }
        if ('reportNet' in obj) {
            message.report_net = obj.reportNet;
        }
        if ('reportInterval' in obj) {
            message.report_interval = Number(obj.reportInterval);
        }
        if ('runEveryMs' in obj) {
            message.run_every_ms = BigInt(obj.runEveryMs);
        }
        if ('concurrentSubsteps' in obj) {
            message.concurrent_substeps = obj.concurrentSubsteps;
        }
        if ('shouldStopBlob' in obj) {
            message.should_stop_blob = obj.shouldStopBlob;
        }
        if ('onlyOnce' in obj) {
            message.only_once = obj.onlyOnce;
        }
        if ('createWorkspace' in obj) {
            message.create_workspace = obj.createWorkspace;
        }
        if ('numConcurrentInstances' in obj) {
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
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('network' in obj) {
            message.network = obj.network.map((obj) => caffe2.NetDef.decodeJson(obj));
        }
        if ('executionStep' in obj) {
            message.execution_step = obj.executionStep.map((obj) => caffe2.ExecutionStep.decodeJson(obj));
        }
        return message;
    }
};

caffe2.PlanDef.prototype.name = "";

caffe2.BlobProto = class BlobProto {

    static decodeJson(obj) {
        const message = new caffe2.BlobProto();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('type' in obj) {
            message.type = obj.type;
        }
        if ('tensor' in obj) {
            message.tensor = caffe2.TensorProto.decodeJson(obj.tensor);
        }
        if ('content' in obj) {
            message.content = new Uint8Array(atob(obj.content));
        }
        if ('qtensor' in obj) {
            message.qtensor = caffe2.QTensorProto.decodeJson(obj.qtensor);
        }
        if ('contentNumChunks' in obj) {
            message.content_num_chunks = Number(obj.contentNumChunks);
        }
        if ('contentChunkId' in obj) {
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
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('source' in obj) {
            message.source = obj.source;
        }
        if ('dbType' in obj) {
            message.db_type = obj.dbType;
        }
        if ('key' in obj) {
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
        if ('blobNameRegex' in obj) {
            message.blob_name_regex = obj.blobNameRegex;
        }
        if ('chunkSize' in obj) {
            message.chunk_size = BigInt(obj.chunkSize);
        }
        if ('floatFormat' in obj) {
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
        if ('options' in obj) {
            message.options = obj.options.map((obj) => caffe2.BlobSerializationOptions.decodeJson(obj));
        }
        return message;
    }
};
