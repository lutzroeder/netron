var $root = protobuf.get('caffe');

$root.caffe = {};

$root.caffe.BlobShape = class BlobShape {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.BlobShape();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim = reader.array(message.dim, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.BlobShape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    reader.array(message.dim, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.BlobProto = class BlobProto {

    constructor() {
        this.data = [];
        this.diff = [];
        this.double_data = [];
        this.double_diff = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.BlobProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 7:
                    message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.data = reader.floats(message.data, tag);
                    break;
                case 6:
                    message.diff = reader.floats(message.diff, tag);
                    break;
                case 8:
                    message.double_data = reader.doubles(message.double_data, tag);
                    break;
                case 9:
                    message.double_diff = reader.doubles(message.double_diff, tag);
                    break;
                case 1:
                    message.num = reader.int32();
                    break;
                case 2:
                    message.channels = reader.int32();
                    break;
                case 3:
                    message.height = reader.int32();
                    break;
                case 4:
                    message.width = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.BlobProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.caffe.BlobShape.decodeText(reader);
                    break;
                case "data":
                    reader.array(message.data, () => reader.float());
                    break;
                case "diff":
                    reader.array(message.diff, () => reader.float());
                    break;
                case "double_data":
                    reader.array(message.double_data, () => reader.double());
                    break;
                case "double_diff":
                    reader.array(message.double_diff, () => reader.double());
                    break;
                case "num":
                    message.num = reader.int32();
                    break;
                case "channels":
                    message.channels = reader.int32();
                    break;
                case "height":
                    message.height = reader.int32();
                    break;
                case "width":
                    message.width = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.BlobProto.prototype.shape = null;
$root.caffe.BlobProto.prototype.num = 0;
$root.caffe.BlobProto.prototype.channels = 0;
$root.caffe.BlobProto.prototype.height = 0;
$root.caffe.BlobProto.prototype.width = 0;

$root.caffe.BlobProtoVector = class BlobProtoVector {

    constructor() {
        this.blobs = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.BlobProtoVector();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.BlobProtoVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "blobs":
                    message.blobs.push($root.caffe.BlobProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.Datum = class Datum {

    constructor() {
        this.float_data = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.Datum();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.channels = reader.int32();
                    break;
                case 2:
                    message.height = reader.int32();
                    break;
                case 3:
                    message.width = reader.int32();
                    break;
                case 4:
                    message.data = reader.bytes();
                    break;
                case 5:
                    message.label = reader.int32();
                    break;
                case 6:
                    message.float_data = reader.floats(message.float_data, tag);
                    break;
                case 7:
                    message.encoded = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.Datum();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "channels":
                    message.channels = reader.int32();
                    break;
                case "height":
                    message.height = reader.int32();
                    break;
                case "width":
                    message.width = reader.int32();
                    break;
                case "data":
                    message.data = reader.bytes();
                    break;
                case "label":
                    message.label = reader.int32();
                    break;
                case "float_data":
                    reader.array(message.float_data, () => reader.float());
                    break;
                case "encoded":
                    message.encoded = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.Datum.prototype.channels = 0;
$root.caffe.Datum.prototype.height = 0;
$root.caffe.Datum.prototype.width = 0;
$root.caffe.Datum.prototype.data = new Uint8Array([]);
$root.caffe.Datum.prototype.label = 0;
$root.caffe.Datum.prototype.encoded = false;

$root.caffe.FillerParameter = class FillerParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.FillerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.string();
                    break;
                case 2:
                    message.value = reader.float();
                    break;
                case 3:
                    message.min = reader.float();
                    break;
                case 4:
                    message.max = reader.float();
                    break;
                case 5:
                    message.mean = reader.float();
                    break;
                case 6:
                    message.std = reader.float();
                    break;
                case 7:
                    message.sparse = reader.int32();
                    break;
                case 8:
                    message.variance_norm = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.FillerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "value":
                    message.value = reader.float();
                    break;
                case "min":
                    message.min = reader.float();
                    break;
                case "max":
                    message.max = reader.float();
                    break;
                case "mean":
                    message.mean = reader.float();
                    break;
                case "std":
                    message.std = reader.float();
                    break;
                case "sparse":
                    message.sparse = reader.int32();
                    break;
                case "variance_norm":
                    message.variance_norm = reader.enum($root.caffe.FillerParameter.VarianceNorm);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.FillerParameter.prototype.type = "constant";
$root.caffe.FillerParameter.prototype.value = 0;
$root.caffe.FillerParameter.prototype.min = 0;
$root.caffe.FillerParameter.prototype.max = 1;
$root.caffe.FillerParameter.prototype.mean = 0;
$root.caffe.FillerParameter.prototype.std = 1;
$root.caffe.FillerParameter.prototype.sparse = -1;
$root.caffe.FillerParameter.prototype.variance_norm = 0;

$root.caffe.FillerParameter.VarianceNorm = {
    "FAN_IN": 0,
    "FAN_OUT": 1,
    "AVERAGE": 2
};

$root.caffe.NetParameter = class NetParameter {

    constructor() {
        this.input = [];
        this.input_shape = [];
        this.input_dim = [];
        this.layer = [];
        this.layers = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.NetParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 3:
                    message.input.push(reader.string());
                    break;
                case 8:
                    message.input_shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.input_dim = reader.array(message.input_dim, () => reader.int32(), tag);
                    break;
                case 5:
                    message.force_backward = reader.bool();
                    break;
                case 6:
                    message.state = $root.caffe.NetState.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.debug_info = reader.bool();
                    break;
                case 100:
                    message.layer.push($root.caffe.LayerParameter.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.layers.push($root.caffe.V1LayerParameter.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.NetParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "input_shape":
                    message.input_shape.push($root.caffe.BlobShape.decodeText(reader));
                    break;
                case "input_dim":
                    reader.array(message.input_dim, () => reader.int32());
                    break;
                case "force_backward":
                    message.force_backward = reader.bool();
                    break;
                case "state":
                    message.state = $root.caffe.NetState.decodeText(reader);
                    break;
                case "debug_info":
                    message.debug_info = reader.bool();
                    break;
                case "layer":
                    message.layer.push($root.caffe.LayerParameter.decodeText(reader));
                    break;
                case "layers":
                    message.layers.push($root.caffe.V1LayerParameter.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.NetParameter.prototype.name = "";
$root.caffe.NetParameter.prototype.force_backward = false;
$root.caffe.NetParameter.prototype.state = null;
$root.caffe.NetParameter.prototype.debug_info = false;

$root.caffe.SolverParameter = class SolverParameter {

    constructor() {
        this.test_net = [];
        this.test_net_param = [];
        this.test_state = [];
        this.test_iter = [];
        this.stepvalue = [];
        this.weights = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.SolverParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 24:
                    message.net = reader.string();
                    break;
                case 25:
                    message.net_param = $root.caffe.NetParameter.decode(reader, reader.uint32());
                    break;
                case 1:
                    message.train_net = reader.string();
                    break;
                case 2:
                    message.test_net.push(reader.string());
                    break;
                case 21:
                    message.train_net_param = $root.caffe.NetParameter.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.test_net_param.push($root.caffe.NetParameter.decode(reader, reader.uint32()));
                    break;
                case 26:
                    message.train_state = $root.caffe.NetState.decode(reader, reader.uint32());
                    break;
                case 27:
                    message.test_state.push($root.caffe.NetState.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.test_iter = reader.array(message.test_iter, () => reader.int32(), tag);
                    break;
                case 4:
                    message.test_interval = reader.int32();
                    break;
                case 19:
                    message.test_compute_loss = reader.bool();
                    break;
                case 32:
                    message.test_initialization = reader.bool();
                    break;
                case 5:
                    message.base_lr = reader.float();
                    break;
                case 6:
                    message.display = reader.int32();
                    break;
                case 33:
                    message.average_loss = reader.int32();
                    break;
                case 7:
                    message.max_iter = reader.int32();
                    break;
                case 36:
                    message.iter_size = reader.int32();
                    break;
                case 8:
                    message.lr_policy = reader.string();
                    break;
                case 9:
                    message.gamma = reader.float();
                    break;
                case 10:
                    message.power = reader.float();
                    break;
                case 11:
                    message.momentum = reader.float();
                    break;
                case 12:
                    message.weight_decay = reader.float();
                    break;
                case 29:
                    message.regularization_type = reader.string();
                    break;
                case 13:
                    message.stepsize = reader.int32();
                    break;
                case 34:
                    message.stepvalue = reader.array(message.stepvalue, () => reader.int32(), tag);
                    break;
                case 35:
                    message.clip_gradients = reader.float();
                    break;
                case 14:
                    message.snapshot = reader.int32();
                    break;
                case 15:
                    message.snapshot_prefix = reader.string();
                    break;
                case 16:
                    message.snapshot_diff = reader.bool();
                    break;
                case 37:
                    message.snapshot_format = reader.int32();
                    break;
                case 17:
                    message.solver_mode = reader.int32();
                    break;
                case 18:
                    message.device_id = reader.int32();
                    break;
                case 20:
                    message.random_seed = reader.int64();
                    break;
                case 40:
                    message.type = reader.string();
                    break;
                case 31:
                    message.delta = reader.float();
                    break;
                case 39:
                    message.momentum2 = reader.float();
                    break;
                case 38:
                    message.rms_decay = reader.float();
                    break;
                case 23:
                    message.debug_info = reader.bool();
                    break;
                case 28:
                    message.snapshot_after_train = reader.bool();
                    break;
                case 30:
                    message.solver_type = reader.int32();
                    break;
                case 41:
                    message.layer_wise_reduce = reader.bool();
                    break;
                case 42:
                    message.weights.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SolverParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "net":
                    message.net = reader.string();
                    break;
                case "net_param":
                    message.net_param = $root.caffe.NetParameter.decodeText(reader);
                    break;
                case "train_net":
                    message.train_net = reader.string();
                    break;
                case "test_net":
                    reader.array(message.test_net, () => reader.string());
                    break;
                case "train_net_param":
                    message.train_net_param = $root.caffe.NetParameter.decodeText(reader);
                    break;
                case "test_net_param":
                    message.test_net_param.push($root.caffe.NetParameter.decodeText(reader));
                    break;
                case "train_state":
                    message.train_state = $root.caffe.NetState.decodeText(reader);
                    break;
                case "test_state":
                    message.test_state.push($root.caffe.NetState.decodeText(reader));
                    break;
                case "test_iter":
                    reader.array(message.test_iter, () => reader.int32());
                    break;
                case "test_interval":
                    message.test_interval = reader.int32();
                    break;
                case "test_compute_loss":
                    message.test_compute_loss = reader.bool();
                    break;
                case "test_initialization":
                    message.test_initialization = reader.bool();
                    break;
                case "base_lr":
                    message.base_lr = reader.float();
                    break;
                case "display":
                    message.display = reader.int32();
                    break;
                case "average_loss":
                    message.average_loss = reader.int32();
                    break;
                case "max_iter":
                    message.max_iter = reader.int32();
                    break;
                case "iter_size":
                    message.iter_size = reader.int32();
                    break;
                case "lr_policy":
                    message.lr_policy = reader.string();
                    break;
                case "gamma":
                    message.gamma = reader.float();
                    break;
                case "power":
                    message.power = reader.float();
                    break;
                case "momentum":
                    message.momentum = reader.float();
                    break;
                case "weight_decay":
                    message.weight_decay = reader.float();
                    break;
                case "regularization_type":
                    message.regularization_type = reader.string();
                    break;
                case "stepsize":
                    message.stepsize = reader.int32();
                    break;
                case "stepvalue":
                    reader.array(message.stepvalue, () => reader.int32());
                    break;
                case "clip_gradients":
                    message.clip_gradients = reader.float();
                    break;
                case "snapshot":
                    message.snapshot = reader.int32();
                    break;
                case "snapshot_prefix":
                    message.snapshot_prefix = reader.string();
                    break;
                case "snapshot_diff":
                    message.snapshot_diff = reader.bool();
                    break;
                case "snapshot_format":
                    message.snapshot_format = reader.enum($root.caffe.SolverParameter.SnapshotFormat);
                    break;
                case "solver_mode":
                    message.solver_mode = reader.enum($root.caffe.SolverParameter.SolverMode);
                    break;
                case "device_id":
                    message.device_id = reader.int32();
                    break;
                case "random_seed":
                    message.random_seed = reader.int64();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "delta":
                    message.delta = reader.float();
                    break;
                case "momentum2":
                    message.momentum2 = reader.float();
                    break;
                case "rms_decay":
                    message.rms_decay = reader.float();
                    break;
                case "debug_info":
                    message.debug_info = reader.bool();
                    break;
                case "snapshot_after_train":
                    message.snapshot_after_train = reader.bool();
                    break;
                case "solver_type":
                    message.solver_type = reader.enum($root.caffe.SolverParameter.SolverType);
                    break;
                case "layer_wise_reduce":
                    message.layer_wise_reduce = reader.bool();
                    break;
                case "weights":
                    reader.array(message.weights, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SolverParameter.prototype.net = "";
$root.caffe.SolverParameter.prototype.net_param = null;
$root.caffe.SolverParameter.prototype.train_net = "";
$root.caffe.SolverParameter.prototype.train_net_param = null;
$root.caffe.SolverParameter.prototype.train_state = null;
$root.caffe.SolverParameter.prototype.test_interval = 0;
$root.caffe.SolverParameter.prototype.test_compute_loss = false;
$root.caffe.SolverParameter.prototype.test_initialization = true;
$root.caffe.SolverParameter.prototype.base_lr = 0;
$root.caffe.SolverParameter.prototype.display = 0;
$root.caffe.SolverParameter.prototype.average_loss = 1;
$root.caffe.SolverParameter.prototype.max_iter = 0;
$root.caffe.SolverParameter.prototype.iter_size = 1;
$root.caffe.SolverParameter.prototype.lr_policy = "";
$root.caffe.SolverParameter.prototype.gamma = 0;
$root.caffe.SolverParameter.prototype.power = 0;
$root.caffe.SolverParameter.prototype.momentum = 0;
$root.caffe.SolverParameter.prototype.weight_decay = 0;
$root.caffe.SolverParameter.prototype.regularization_type = "L2";
$root.caffe.SolverParameter.prototype.stepsize = 0;
$root.caffe.SolverParameter.prototype.clip_gradients = -1;
$root.caffe.SolverParameter.prototype.snapshot = 0;
$root.caffe.SolverParameter.prototype.snapshot_prefix = "";
$root.caffe.SolverParameter.prototype.snapshot_diff = false;
$root.caffe.SolverParameter.prototype.snapshot_format = 1;
$root.caffe.SolverParameter.prototype.solver_mode = 1;
$root.caffe.SolverParameter.prototype.device_id = 0;
$root.caffe.SolverParameter.prototype.random_seed = protobuf.Int64.create(-1);
$root.caffe.SolverParameter.prototype.type = "SGD";
$root.caffe.SolverParameter.prototype.delta = 1e-8;
$root.caffe.SolverParameter.prototype.momentum2 = 0.999;
$root.caffe.SolverParameter.prototype.rms_decay = 0.99;
$root.caffe.SolverParameter.prototype.debug_info = false;
$root.caffe.SolverParameter.prototype.snapshot_after_train = true;
$root.caffe.SolverParameter.prototype.solver_type = 0;
$root.caffe.SolverParameter.prototype.layer_wise_reduce = true;

$root.caffe.SolverParameter.SnapshotFormat = {
    "HDF5": 0,
    "BINARYPROTO": 1
};

$root.caffe.SolverParameter.SolverMode = {
    "CPU": 0,
    "GPU": 1
};

$root.caffe.SolverParameter.SolverType = {
    "SGD": 0,
    "NESTEROV": 1,
    "ADAGRAD": 2,
    "RMSPROP": 3,
    "ADADELTA": 4,
    "ADAM": 5
};

$root.caffe.SolverState = class SolverState {

    constructor() {
        this.history = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.SolverState();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.iter = reader.int32();
                    break;
                case 2:
                    message.learned_net = reader.string();
                    break;
                case 3:
                    message.history.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.current_step = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SolverState();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "iter":
                    message.iter = reader.int32();
                    break;
                case "learned_net":
                    message.learned_net = reader.string();
                    break;
                case "history":
                    message.history.push($root.caffe.BlobProto.decodeText(reader));
                    break;
                case "current_step":
                    message.current_step = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SolverState.prototype.iter = 0;
$root.caffe.SolverState.prototype.learned_net = "";
$root.caffe.SolverState.prototype.current_step = 0;

$root.caffe.Phase = {
    "TRAIN": 0,
    "TEST": 1
};

$root.caffe.NetState = class NetState {

    constructor() {
        this.stage = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.NetState();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.phase = reader.int32();
                    break;
                case 2:
                    message.level = reader.int32();
                    break;
                case 3:
                    message.stage.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.NetState();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "phase":
                    message.phase = reader.enum($root.caffe.Phase);
                    break;
                case "level":
                    message.level = reader.int32();
                    break;
                case "stage":
                    reader.array(message.stage, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.NetState.prototype.phase = 1;
$root.caffe.NetState.prototype.level = 0;

$root.caffe.NetStateRule = class NetStateRule {

    constructor() {
        this.stage = [];
        this.not_stage = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.NetStateRule();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.phase = reader.int32();
                    break;
                case 2:
                    message.min_level = reader.int32();
                    break;
                case 3:
                    message.max_level = reader.int32();
                    break;
                case 4:
                    message.stage.push(reader.string());
                    break;
                case 5:
                    message.not_stage.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.NetStateRule();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "phase":
                    message.phase = reader.enum($root.caffe.Phase);
                    break;
                case "min_level":
                    message.min_level = reader.int32();
                    break;
                case "max_level":
                    message.max_level = reader.int32();
                    break;
                case "stage":
                    reader.array(message.stage, () => reader.string());
                    break;
                case "not_stage":
                    reader.array(message.not_stage, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.NetStateRule.prototype.phase = 0;
$root.caffe.NetStateRule.prototype.min_level = 0;
$root.caffe.NetStateRule.prototype.max_level = 0;

$root.caffe.ParamSpec = class ParamSpec {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ParamSpec();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.share_mode = reader.int32();
                    break;
                case 3:
                    message.lr_mult = reader.float();
                    break;
                case 4:
                    message.decay_mult = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ParamSpec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "share_mode":
                    message.share_mode = reader.enum($root.caffe.ParamSpec.DimCheckMode);
                    break;
                case "lr_mult":
                    message.lr_mult = reader.float();
                    break;
                case "decay_mult":
                    message.decay_mult = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ParamSpec.prototype.name = "";
$root.caffe.ParamSpec.prototype.share_mode = 0;
$root.caffe.ParamSpec.prototype.lr_mult = 1;
$root.caffe.ParamSpec.prototype.decay_mult = 1;

$root.caffe.ParamSpec.DimCheckMode = {
    "STRICT": 0,
    "PERMISSIVE": 1
};

$root.caffe.LayerParameter = class LayerParameter {

    constructor() {
        this.bottom = [];
        this.top = [];
        this.loss_weight = [];
        this.param = [];
        this.blobs = [];
        this.propagate_down = [];
        this.include = [];
        this.exclude = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.LayerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.bottom.push(reader.string());
                    break;
                case 4:
                    message.top.push(reader.string());
                    break;
                case 10:
                    message.phase = reader.int32();
                    break;
                case 5:
                    message.loss_weight = reader.floats(message.loss_weight, tag);
                    break;
                case 6:
                    message.param.push($root.caffe.ParamSpec.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.propagate_down = reader.array(message.propagate_down, () => reader.bool(), tag);
                    break;
                case 8:
                    message.include.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.exclude.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.transform_param = $root.caffe.TransformationParameter.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.loss_param = $root.caffe.LossParameter.decode(reader, reader.uint32());
                    break;
                case 102:
                    message.accuracy_param = $root.caffe.AccuracyParameter.decode(reader, reader.uint32());
                    break;
                case 103:
                    message.argmax_param = $root.caffe.ArgMaxParameter.decode(reader, reader.uint32());
                    break;
                case 139:
                    message.batch_norm_param = $root.caffe.BatchNormParameter.decode(reader, reader.uint32());
                    break;
                case 141:
                    message.bias_param = $root.caffe.BiasParameter.decode(reader, reader.uint32());
                    break;
                case 148:
                    message.clip_param = $root.caffe.ClipParameter.decode(reader, reader.uint32());
                    break;
                case 104:
                    message.concat_param = $root.caffe.ConcatParameter.decode(reader, reader.uint32());
                    break;
                case 105:
                    message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                    break;
                case 106:
                    message.convolution_param = $root.caffe.ConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 144:
                    message.crop_param = $root.caffe.CropParameter.decode(reader, reader.uint32());
                    break;
                case 107:
                    message.data_param = $root.caffe.DataParameter.decode(reader, reader.uint32());
                    break;
                case 108:
                    message.dropout_param = $root.caffe.DropoutParameter.decode(reader, reader.uint32());
                    break;
                case 109:
                    message.dummy_data_param = $root.caffe.DummyDataParameter.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.eltwise_param = $root.caffe.EltwiseParameter.decode(reader, reader.uint32());
                    break;
                case 140:
                    message.elu_param = $root.caffe.ELUParameter.decode(reader, reader.uint32());
                    break;
                case 137:
                    message.embed_param = $root.caffe.EmbedParameter.decode(reader, reader.uint32());
                    break;
                case 111:
                    message.exp_param = $root.caffe.ExpParameter.decode(reader, reader.uint32());
                    break;
                case 135:
                    message.flatten_param = $root.caffe.FlattenParameter.decode(reader, reader.uint32());
                    break;
                case 112:
                    message.hdf5_data_param = $root.caffe.HDF5DataParameter.decode(reader, reader.uint32());
                    break;
                case 113:
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                    break;
                case 114:
                    message.hinge_loss_param = $root.caffe.HingeLossParameter.decode(reader, reader.uint32());
                    break;
                case 115:
                    message.image_data_param = $root.caffe.ImageDataParameter.decode(reader, reader.uint32());
                    break;
                case 116:
                    message.infogain_loss_param = $root.caffe.InfogainLossParameter.decode(reader, reader.uint32());
                    break;
                case 117:
                    message.inner_product_param = $root.caffe.InnerProductParameter.decode(reader, reader.uint32());
                    break;
                case 143:
                    message.input_param = $root.caffe.InputParameter.decode(reader, reader.uint32());
                    break;
                case 134:
                    message.log_param = $root.caffe.LogParameter.decode(reader, reader.uint32());
                    break;
                case 118:
                    message.lrn_param = $root.caffe.LRNParameter.decode(reader, reader.uint32());
                    break;
                case 119:
                    message.memory_data_param = $root.caffe.MemoryDataParameter.decode(reader, reader.uint32());
                    break;
                case 120:
                    message.mvn_param = $root.caffe.MVNParameter.decode(reader, reader.uint32());
                    break;
                case 145:
                    message.parameter_param = $root.caffe.ParameterParameter.decode(reader, reader.uint32());
                    break;
                case 121:
                    message.pooling_param = $root.caffe.PoolingParameter.decode(reader, reader.uint32());
                    break;
                case 122:
                    message.power_param = $root.caffe.PowerParameter.decode(reader, reader.uint32());
                    break;
                case 131:
                    message.prelu_param = $root.caffe.PReLUParameter.decode(reader, reader.uint32());
                    break;
                case 130:
                    message.python_param = $root.caffe.PythonParameter.decode(reader, reader.uint32());
                    break;
                case 146:
                    message.recurrent_param = $root.caffe.RecurrentParameter.decode(reader, reader.uint32());
                    break;
                case 136:
                    message.reduction_param = $root.caffe.ReductionParameter.decode(reader, reader.uint32());
                    break;
                case 123:
                    message.relu_param = $root.caffe.ReLUParameter.decode(reader, reader.uint32());
                    break;
                case 133:
                    message.reshape_param = $root.caffe.ReshapeParameter.decode(reader, reader.uint32());
                    break;
                case 142:
                    message.scale_param = $root.caffe.ScaleParameter.decode(reader, reader.uint32());
                    break;
                case 124:
                    message.sigmoid_param = $root.caffe.SigmoidParameter.decode(reader, reader.uint32());
                    break;
                case 125:
                    message.softmax_param = $root.caffe.SoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 132:
                    message.spp_param = $root.caffe.SPPParameter.decode(reader, reader.uint32());
                    break;
                case 126:
                    message.slice_param = $root.caffe.SliceParameter.decode(reader, reader.uint32());
                    break;
                case 147:
                    message.swish_param = $root.caffe.SwishParameter.decode(reader, reader.uint32());
                    break;
                case 127:
                    message.tanh_param = $root.caffe.TanHParameter.decode(reader, reader.uint32());
                    break;
                case 128:
                    message.threshold_param = $root.caffe.ThresholdParameter.decode(reader, reader.uint32());
                    break;
                case 138:
                    message.tile_param = $root.caffe.TileParameter.decode(reader, reader.uint32());
                    break;
                case 129:
                    message.window_data_param = $root.caffe.WindowDataParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.LayerParameter();
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
                case "bottom":
                    reader.array(message.bottom, () => reader.string());
                    break;
                case "top":
                    reader.array(message.top, () => reader.string());
                    break;
                case "phase":
                    message.phase = reader.enum($root.caffe.Phase);
                    break;
                case "loss_weight":
                    reader.array(message.loss_weight, () => reader.float());
                    break;
                case "param":
                    message.param.push($root.caffe.ParamSpec.decodeText(reader));
                    break;
                case "blobs":
                    message.blobs.push($root.caffe.BlobProto.decodeText(reader));
                    break;
                case "propagate_down":
                    reader.array(message.propagate_down, () => reader.bool());
                    break;
                case "include":
                    message.include.push($root.caffe.NetStateRule.decodeText(reader));
                    break;
                case "exclude":
                    message.exclude.push($root.caffe.NetStateRule.decodeText(reader));
                    break;
                case "transform_param":
                    message.transform_param = $root.caffe.TransformationParameter.decodeText(reader);
                    break;
                case "loss_param":
                    message.loss_param = $root.caffe.LossParameter.decodeText(reader);
                    break;
                case "accuracy_param":
                    message.accuracy_param = $root.caffe.AccuracyParameter.decodeText(reader);
                    break;
                case "argmax_param":
                    message.argmax_param = $root.caffe.ArgMaxParameter.decodeText(reader);
                    break;
                case "batch_norm_param":
                    message.batch_norm_param = $root.caffe.BatchNormParameter.decodeText(reader);
                    break;
                case "bias_param":
                    message.bias_param = $root.caffe.BiasParameter.decodeText(reader);
                    break;
                case "clip_param":
                    message.clip_param = $root.caffe.ClipParameter.decodeText(reader);
                    break;
                case "concat_param":
                    message.concat_param = $root.caffe.ConcatParameter.decodeText(reader);
                    break;
                case "contrastive_loss_param":
                    message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decodeText(reader);
                    break;
                case "convolution_param":
                    message.convolution_param = $root.caffe.ConvolutionParameter.decodeText(reader);
                    break;
                case "crop_param":
                    message.crop_param = $root.caffe.CropParameter.decodeText(reader);
                    break;
                case "data_param":
                    message.data_param = $root.caffe.DataParameter.decodeText(reader);
                    break;
                case "dropout_param":
                    message.dropout_param = $root.caffe.DropoutParameter.decodeText(reader);
                    break;
                case "dummy_data_param":
                    message.dummy_data_param = $root.caffe.DummyDataParameter.decodeText(reader);
                    break;
                case "eltwise_param":
                    message.eltwise_param = $root.caffe.EltwiseParameter.decodeText(reader);
                    break;
                case "elu_param":
                    message.elu_param = $root.caffe.ELUParameter.decodeText(reader);
                    break;
                case "embed_param":
                    message.embed_param = $root.caffe.EmbedParameter.decodeText(reader);
                    break;
                case "exp_param":
                    message.exp_param = $root.caffe.ExpParameter.decodeText(reader);
                    break;
                case "flatten_param":
                    message.flatten_param = $root.caffe.FlattenParameter.decodeText(reader);
                    break;
                case "hdf5_data_param":
                    message.hdf5_data_param = $root.caffe.HDF5DataParameter.decodeText(reader);
                    break;
                case "hdf5_output_param":
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader);
                    break;
                case "hinge_loss_param":
                    message.hinge_loss_param = $root.caffe.HingeLossParameter.decodeText(reader);
                    break;
                case "image_data_param":
                    message.image_data_param = $root.caffe.ImageDataParameter.decodeText(reader);
                    break;
                case "infogain_loss_param":
                    message.infogain_loss_param = $root.caffe.InfogainLossParameter.decodeText(reader);
                    break;
                case "inner_product_param":
                    message.inner_product_param = $root.caffe.InnerProductParameter.decodeText(reader);
                    break;
                case "input_param":
                    message.input_param = $root.caffe.InputParameter.decodeText(reader);
                    break;
                case "log_param":
                    message.log_param = $root.caffe.LogParameter.decodeText(reader);
                    break;
                case "lrn_param":
                    message.lrn_param = $root.caffe.LRNParameter.decodeText(reader);
                    break;
                case "memory_data_param":
                    message.memory_data_param = $root.caffe.MemoryDataParameter.decodeText(reader);
                    break;
                case "mvn_param":
                    message.mvn_param = $root.caffe.MVNParameter.decodeText(reader);
                    break;
                case "parameter_param":
                    message.parameter_param = $root.caffe.ParameterParameter.decodeText(reader);
                    break;
                case "pooling_param":
                    message.pooling_param = $root.caffe.PoolingParameter.decodeText(reader);
                    break;
                case "power_param":
                    message.power_param = $root.caffe.PowerParameter.decodeText(reader);
                    break;
                case "prelu_param":
                    message.prelu_param = $root.caffe.PReLUParameter.decodeText(reader);
                    break;
                case "python_param":
                    message.python_param = $root.caffe.PythonParameter.decodeText(reader);
                    break;
                case "recurrent_param":
                    message.recurrent_param = $root.caffe.RecurrentParameter.decodeText(reader);
                    break;
                case "reduction_param":
                    message.reduction_param = $root.caffe.ReductionParameter.decodeText(reader);
                    break;
                case "relu_param":
                    message.relu_param = $root.caffe.ReLUParameter.decodeText(reader);
                    break;
                case "reshape_param":
                    message.reshape_param = $root.caffe.ReshapeParameter.decodeText(reader);
                    break;
                case "scale_param":
                    message.scale_param = $root.caffe.ScaleParameter.decodeText(reader);
                    break;
                case "sigmoid_param":
                    message.sigmoid_param = $root.caffe.SigmoidParameter.decodeText(reader);
                    break;
                case "softmax_param":
                    message.softmax_param = $root.caffe.SoftmaxParameter.decodeText(reader);
                    break;
                case "spp_param":
                    message.spp_param = $root.caffe.SPPParameter.decodeText(reader);
                    break;
                case "slice_param":
                    message.slice_param = $root.caffe.SliceParameter.decodeText(reader);
                    break;
                case "swish_param":
                    message.swish_param = $root.caffe.SwishParameter.decodeText(reader);
                    break;
                case "tanh_param":
                    message.tanh_param = $root.caffe.TanHParameter.decodeText(reader);
                    break;
                case "threshold_param":
                    message.threshold_param = $root.caffe.ThresholdParameter.decodeText(reader);
                    break;
                case "tile_param":
                    message.tile_param = $root.caffe.TileParameter.decodeText(reader);
                    break;
                case "window_data_param":
                    message.window_data_param = $root.caffe.WindowDataParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.LayerParameter.prototype.name = "";
$root.caffe.LayerParameter.prototype.type = "";
$root.caffe.LayerParameter.prototype.phase = 0;
$root.caffe.LayerParameter.prototype.transform_param = null;
$root.caffe.LayerParameter.prototype.loss_param = null;
$root.caffe.LayerParameter.prototype.accuracy_param = null;
$root.caffe.LayerParameter.prototype.argmax_param = null;
$root.caffe.LayerParameter.prototype.batch_norm_param = null;
$root.caffe.LayerParameter.prototype.bias_param = null;
$root.caffe.LayerParameter.prototype.clip_param = null;
$root.caffe.LayerParameter.prototype.concat_param = null;
$root.caffe.LayerParameter.prototype.contrastive_loss_param = null;
$root.caffe.LayerParameter.prototype.convolution_param = null;
$root.caffe.LayerParameter.prototype.crop_param = null;
$root.caffe.LayerParameter.prototype.data_param = null;
$root.caffe.LayerParameter.prototype.dropout_param = null;
$root.caffe.LayerParameter.prototype.dummy_data_param = null;
$root.caffe.LayerParameter.prototype.eltwise_param = null;
$root.caffe.LayerParameter.prototype.elu_param = null;
$root.caffe.LayerParameter.prototype.embed_param = null;
$root.caffe.LayerParameter.prototype.exp_param = null;
$root.caffe.LayerParameter.prototype.flatten_param = null;
$root.caffe.LayerParameter.prototype.hdf5_data_param = null;
$root.caffe.LayerParameter.prototype.hdf5_output_param = null;
$root.caffe.LayerParameter.prototype.hinge_loss_param = null;
$root.caffe.LayerParameter.prototype.image_data_param = null;
$root.caffe.LayerParameter.prototype.infogain_loss_param = null;
$root.caffe.LayerParameter.prototype.inner_product_param = null;
$root.caffe.LayerParameter.prototype.input_param = null;
$root.caffe.LayerParameter.prototype.log_param = null;
$root.caffe.LayerParameter.prototype.lrn_param = null;
$root.caffe.LayerParameter.prototype.memory_data_param = null;
$root.caffe.LayerParameter.prototype.mvn_param = null;
$root.caffe.LayerParameter.prototype.parameter_param = null;
$root.caffe.LayerParameter.prototype.pooling_param = null;
$root.caffe.LayerParameter.prototype.power_param = null;
$root.caffe.LayerParameter.prototype.prelu_param = null;
$root.caffe.LayerParameter.prototype.python_param = null;
$root.caffe.LayerParameter.prototype.recurrent_param = null;
$root.caffe.LayerParameter.prototype.reduction_param = null;
$root.caffe.LayerParameter.prototype.relu_param = null;
$root.caffe.LayerParameter.prototype.reshape_param = null;
$root.caffe.LayerParameter.prototype.scale_param = null;
$root.caffe.LayerParameter.prototype.sigmoid_param = null;
$root.caffe.LayerParameter.prototype.softmax_param = null;
$root.caffe.LayerParameter.prototype.spp_param = null;
$root.caffe.LayerParameter.prototype.slice_param = null;
$root.caffe.LayerParameter.prototype.swish_param = null;
$root.caffe.LayerParameter.prototype.tanh_param = null;
$root.caffe.LayerParameter.prototype.threshold_param = null;
$root.caffe.LayerParameter.prototype.tile_param = null;
$root.caffe.LayerParameter.prototype.window_data_param = null;

$root.caffe.TransformationParameter = class TransformationParameter {

    constructor() {
        this.mean_value = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.TransformationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.scale = reader.float();
                    break;
                case 2:
                    message.mirror = reader.bool();
                    break;
                case 3:
                    message.crop_size = reader.uint32();
                    break;
                case 4:
                    message.mean_file = reader.string();
                    break;
                case 5:
                    message.mean_value = reader.floats(message.mean_value, tag);
                    break;
                case 6:
                    message.force_color = reader.bool();
                    break;
                case 7:
                    message.force_gray = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.TransformationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "scale":
                    message.scale = reader.float();
                    break;
                case "mirror":
                    message.mirror = reader.bool();
                    break;
                case "crop_size":
                    message.crop_size = reader.uint32();
                    break;
                case "mean_file":
                    message.mean_file = reader.string();
                    break;
                case "mean_value":
                    reader.array(message.mean_value, () => reader.float());
                    break;
                case "force_color":
                    message.force_color = reader.bool();
                    break;
                case "force_gray":
                    message.force_gray = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.TransformationParameter.prototype.scale = 1;
$root.caffe.TransformationParameter.prototype.mirror = false;
$root.caffe.TransformationParameter.prototype.crop_size = 0;
$root.caffe.TransformationParameter.prototype.mean_file = "";
$root.caffe.TransformationParameter.prototype.force_color = false;
$root.caffe.TransformationParameter.prototype.force_gray = false;

$root.caffe.LossParameter = class LossParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.LossParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ignore_label = reader.int32();
                    break;
                case 3:
                    message.normalization = reader.int32();
                    break;
                case 2:
                    message.normalize = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.LossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ignore_label":
                    message.ignore_label = reader.int32();
                    break;
                case "normalization":
                    message.normalization = reader.enum($root.caffe.LossParameter.NormalizationMode);
                    break;
                case "normalize":
                    message.normalize = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.LossParameter.prototype.ignore_label = 0;
$root.caffe.LossParameter.prototype.normalization = 1;
$root.caffe.LossParameter.prototype.normalize = false;

$root.caffe.LossParameter.NormalizationMode = {
    "FULL": 0,
    "VALID": 1,
    "BATCH_SIZE": 2,
    "NONE": 3
};

$root.caffe.AccuracyParameter = class AccuracyParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.AccuracyParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.top_k = reader.uint32();
                    break;
                case 2:
                    message.axis = reader.int32();
                    break;
                case 3:
                    message.ignore_label = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.AccuracyParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "top_k":
                    message.top_k = reader.uint32();
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "ignore_label":
                    message.ignore_label = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.AccuracyParameter.prototype.top_k = 1;
$root.caffe.AccuracyParameter.prototype.axis = 1;
$root.caffe.AccuracyParameter.prototype.ignore_label = 0;

$root.caffe.ArgMaxParameter = class ArgMaxParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ArgMaxParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.out_max_val = reader.bool();
                    break;
                case 2:
                    message.top_k = reader.uint32();
                    break;
                case 3:
                    message.axis = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ArgMaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "out_max_val":
                    message.out_max_val = reader.bool();
                    break;
                case "top_k":
                    message.top_k = reader.uint32();
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ArgMaxParameter.prototype.out_max_val = false;
$root.caffe.ArgMaxParameter.prototype.top_k = 1;
$root.caffe.ArgMaxParameter.prototype.axis = 0;

$root.caffe.ClipParameter = class ClipParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ClipParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.min = reader.float();
                    break;
                case 2:
                    message.max = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ClipParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "min":
                    message.min = reader.float();
                    break;
                case "max":
                    message.max = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ClipParameter.prototype.min = 0;
$root.caffe.ClipParameter.prototype.max = 0;

$root.caffe.ConcatParameter = class ConcatParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ConcatParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.axis = reader.int32();
                    break;
                case 1:
                    message.concat_dim = reader.uint32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ConcatParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "concat_dim":
                    message.concat_dim = reader.uint32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ConcatParameter.prototype.axis = 1;
$root.caffe.ConcatParameter.prototype.concat_dim = 1;

$root.caffe.BatchNormParameter = class BatchNormParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.BatchNormParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.use_global_stats = reader.bool();
                    break;
                case 2:
                    message.moving_average_fraction = reader.float();
                    break;
                case 3:
                    message.eps = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.BatchNormParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "use_global_stats":
                    message.use_global_stats = reader.bool();
                    break;
                case "moving_average_fraction":
                    message.moving_average_fraction = reader.float();
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

$root.caffe.BatchNormParameter.prototype.use_global_stats = false;
$root.caffe.BatchNormParameter.prototype.moving_average_fraction = 0.999;
$root.caffe.BatchNormParameter.prototype.eps = 0.00001;

$root.caffe.BiasParameter = class BiasParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.BiasParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int32();
                    break;
                case 2:
                    message.num_axes = reader.int32();
                    break;
                case 3:
                    message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.BiasParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "num_axes":
                    message.num_axes = reader.int32();
                    break;
                case "filler":
                    message.filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.BiasParameter.prototype.axis = 1;
$root.caffe.BiasParameter.prototype.num_axes = 1;
$root.caffe.BiasParameter.prototype.filler = null;

$root.caffe.ContrastiveLossParameter = class ContrastiveLossParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ContrastiveLossParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.margin = reader.float();
                    break;
                case 2:
                    message.legacy_version = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ContrastiveLossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "margin":
                    message.margin = reader.float();
                    break;
                case "legacy_version":
                    message.legacy_version = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ContrastiveLossParameter.prototype.margin = 1;
$root.caffe.ContrastiveLossParameter.prototype.legacy_version = false;

$root.caffe.ConvolutionParameter = class ConvolutionParameter {

    constructor() {
        this.pad = [];
        this.kernel_size = [];
        this.stride = [];
        this.dilation = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.ConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_output = reader.uint32();
                    break;
                case 2:
                    message.bias_term = reader.bool();
                    break;
                case 3:
                    message.pad = reader.array(message.pad, () => reader.uint32(), tag);
                    break;
                case 4:
                    message.kernel_size = reader.array(message.kernel_size, () => reader.uint32(), tag);
                    break;
                case 6:
                    message.stride = reader.array(message.stride, () => reader.uint32(), tag);
                    break;
                case 18:
                    message.dilation = reader.array(message.dilation, () => reader.uint32(), tag);
                    break;
                case 9:
                    message.pad_h = reader.uint32();
                    break;
                case 10:
                    message.pad_w = reader.uint32();
                    break;
                case 11:
                    message.kernel_h = reader.uint32();
                    break;
                case 12:
                    message.kernel_w = reader.uint32();
                    break;
                case 13:
                    message.stride_h = reader.uint32();
                    break;
                case 14:
                    message.stride_w = reader.uint32();
                    break;
                case 5:
                    message.group = reader.uint32();
                    break;
                case 7:
                    message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.engine = reader.int32();
                    break;
                case 16:
                    message.axis = reader.int32();
                    break;
                case 17:
                    message.force_nd_im2col = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_output":
                    message.num_output = reader.uint32();
                    break;
                case "bias_term":
                    message.bias_term = reader.bool();
                    break;
                case "pad":
                    reader.array(message.pad, () => reader.uint32());
                    break;
                case "kernel_size":
                    reader.array(message.kernel_size, () => reader.uint32());
                    break;
                case "stride":
                    reader.array(message.stride, () => reader.uint32());
                    break;
                case "dilation":
                    reader.array(message.dilation, () => reader.uint32());
                    break;
                case "pad_h":
                    message.pad_h = reader.uint32();
                    break;
                case "pad_w":
                    message.pad_w = reader.uint32();
                    break;
                case "kernel_h":
                    message.kernel_h = reader.uint32();
                    break;
                case "kernel_w":
                    message.kernel_w = reader.uint32();
                    break;
                case "stride_h":
                    message.stride_h = reader.uint32();
                    break;
                case "stride_w":
                    message.stride_w = reader.uint32();
                    break;
                case "group":
                    message.group = reader.uint32();
                    break;
                case "weight_filler":
                    message.weight_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "engine":
                    message.engine = reader.enum($root.caffe.ConvolutionParameter.Engine);
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "force_nd_im2col":
                    message.force_nd_im2col = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ConvolutionParameter.prototype.num_output = 0;
$root.caffe.ConvolutionParameter.prototype.bias_term = true;
$root.caffe.ConvolutionParameter.prototype.pad_h = 0;
$root.caffe.ConvolutionParameter.prototype.pad_w = 0;
$root.caffe.ConvolutionParameter.prototype.kernel_h = 0;
$root.caffe.ConvolutionParameter.prototype.kernel_w = 0;
$root.caffe.ConvolutionParameter.prototype.stride_h = 0;
$root.caffe.ConvolutionParameter.prototype.stride_w = 0;
$root.caffe.ConvolutionParameter.prototype.group = 1;
$root.caffe.ConvolutionParameter.prototype.weight_filler = null;
$root.caffe.ConvolutionParameter.prototype.bias_filler = null;
$root.caffe.ConvolutionParameter.prototype.engine = 0;
$root.caffe.ConvolutionParameter.prototype.axis = 1;
$root.caffe.ConvolutionParameter.prototype.force_nd_im2col = false;

$root.caffe.ConvolutionParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.CropParameter = class CropParameter {

    constructor() {
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.CropParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int32();
                    break;
                case 2:
                    message.offset = reader.array(message.offset, () => reader.uint32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.CropParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "offset":
                    reader.array(message.offset, () => reader.uint32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.CropParameter.prototype.axis = 2;

$root.caffe.DataParameter = class DataParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.DataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.source = reader.string();
                    break;
                case 4:
                    message.batch_size = reader.uint32();
                    break;
                case 7:
                    message.rand_skip = reader.uint32();
                    break;
                case 8:
                    message.backend = reader.int32();
                    break;
                case 2:
                    message.scale = reader.float();
                    break;
                case 3:
                    message.mean_file = reader.string();
                    break;
                case 5:
                    message.crop_size = reader.uint32();
                    break;
                case 6:
                    message.mirror = reader.bool();
                    break;
                case 9:
                    message.force_encoded_color = reader.bool();
                    break;
                case 10:
                    message.prefetch = reader.uint32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.DataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "source":
                    message.source = reader.string();
                    break;
                case "batch_size":
                    message.batch_size = reader.uint32();
                    break;
                case "rand_skip":
                    message.rand_skip = reader.uint32();
                    break;
                case "backend":
                    message.backend = reader.enum($root.caffe.DataParameter.DB);
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "mean_file":
                    message.mean_file = reader.string();
                    break;
                case "crop_size":
                    message.crop_size = reader.uint32();
                    break;
                case "mirror":
                    message.mirror = reader.bool();
                    break;
                case "force_encoded_color":
                    message.force_encoded_color = reader.bool();
                    break;
                case "prefetch":
                    message.prefetch = reader.uint32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.DataParameter.prototype.source = "";
$root.caffe.DataParameter.prototype.batch_size = 0;
$root.caffe.DataParameter.prototype.rand_skip = 0;
$root.caffe.DataParameter.prototype.backend = 0;
$root.caffe.DataParameter.prototype.scale = 1;
$root.caffe.DataParameter.prototype.mean_file = "";
$root.caffe.DataParameter.prototype.crop_size = 0;
$root.caffe.DataParameter.prototype.mirror = false;
$root.caffe.DataParameter.prototype.force_encoded_color = false;
$root.caffe.DataParameter.prototype.prefetch = 4;

$root.caffe.DataParameter.DB = {
    "LEVELDB": 0,
    "LMDB": 1
};

$root.caffe.DropoutParameter = class DropoutParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.DropoutParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dropout_ratio = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.DropoutParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dropout_ratio":
                    message.dropout_ratio = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.DropoutParameter.prototype.dropout_ratio = 0.5;

$root.caffe.DummyDataParameter = class DummyDataParameter {

    constructor() {
        this.data_filler = [];
        this.shape = [];
        this.num = [];
        this.channels = [];
        this.height = [];
        this.width = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.DummyDataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.data_filler.push($root.caffe.FillerParameter.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.num = reader.array(message.num, () => reader.uint32(), tag);
                    break;
                case 3:
                    message.channels = reader.array(message.channels, () => reader.uint32(), tag);
                    break;
                case 4:
                    message.height = reader.array(message.height, () => reader.uint32(), tag);
                    break;
                case 5:
                    message.width = reader.array(message.width, () => reader.uint32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.DummyDataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "data_filler":
                    message.data_filler.push($root.caffe.FillerParameter.decodeText(reader));
                    break;
                case "shape":
                    message.shape.push($root.caffe.BlobShape.decodeText(reader));
                    break;
                case "num":
                    reader.array(message.num, () => reader.uint32());
                    break;
                case "channels":
                    reader.array(message.channels, () => reader.uint32());
                    break;
                case "height":
                    reader.array(message.height, () => reader.uint32());
                    break;
                case "width":
                    reader.array(message.width, () => reader.uint32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.EltwiseParameter = class EltwiseParameter {

    constructor() {
        this.coeff = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.EltwiseParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.operation = reader.int32();
                    break;
                case 2:
                    message.coeff = reader.floats(message.coeff, tag);
                    break;
                case 3:
                    message.stable_prod_grad = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.EltwiseParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "operation":
                    message.operation = reader.enum($root.caffe.EltwiseParameter.EltwiseOp);
                    break;
                case "coeff":
                    reader.array(message.coeff, () => reader.float());
                    break;
                case "stable_prod_grad":
                    message.stable_prod_grad = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.EltwiseParameter.prototype.operation = 1;
$root.caffe.EltwiseParameter.prototype.stable_prod_grad = true;

$root.caffe.EltwiseParameter.EltwiseOp = {
    "PROD": 0,
    "SUM": 1,
    "MAX": 2
};

$root.caffe.ELUParameter = class ELUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ELUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ELUParameter();
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

$root.caffe.ELUParameter.prototype.alpha = 1;

$root.caffe.EmbedParameter = class EmbedParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.EmbedParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_output = reader.uint32();
                    break;
                case 2:
                    message.input_dim = reader.uint32();
                    break;
                case 3:
                    message.bias_term = reader.bool();
                    break;
                case 4:
                    message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.EmbedParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_output":
                    message.num_output = reader.uint32();
                    break;
                case "input_dim":
                    message.input_dim = reader.uint32();
                    break;
                case "bias_term":
                    message.bias_term = reader.bool();
                    break;
                case "weight_filler":
                    message.weight_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.EmbedParameter.prototype.num_output = 0;
$root.caffe.EmbedParameter.prototype.input_dim = 0;
$root.caffe.EmbedParameter.prototype.bias_term = true;
$root.caffe.EmbedParameter.prototype.weight_filler = null;
$root.caffe.EmbedParameter.prototype.bias_filler = null;

$root.caffe.ExpParameter = class ExpParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ExpParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base = reader.float();
                    break;
                case 2:
                    message.scale = reader.float();
                    break;
                case 3:
                    message.shift = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ExpParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base":
                    message.base = reader.float();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "shift":
                    message.shift = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ExpParameter.prototype.base = -1;
$root.caffe.ExpParameter.prototype.scale = 1;
$root.caffe.ExpParameter.prototype.shift = 0;

$root.caffe.FlattenParameter = class FlattenParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.FlattenParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int32();
                    break;
                case 2:
                    message.end_axis = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.FlattenParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "end_axis":
                    message.end_axis = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.FlattenParameter.prototype.axis = 1;
$root.caffe.FlattenParameter.prototype.end_axis = -1;

$root.caffe.HDF5DataParameter = class HDF5DataParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.HDF5DataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.source = reader.string();
                    break;
                case 2:
                    message.batch_size = reader.uint32();
                    break;
                case 3:
                    message.shuffle = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.HDF5DataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "source":
                    message.source = reader.string();
                    break;
                case "batch_size":
                    message.batch_size = reader.uint32();
                    break;
                case "shuffle":
                    message.shuffle = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.HDF5DataParameter.prototype.source = "";
$root.caffe.HDF5DataParameter.prototype.batch_size = 0;
$root.caffe.HDF5DataParameter.prototype.shuffle = false;

$root.caffe.HDF5OutputParameter = class HDF5OutputParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.HDF5OutputParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.file_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.HDF5OutputParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "file_name":
                    message.file_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.HDF5OutputParameter.prototype.file_name = "";

$root.caffe.HingeLossParameter = class HingeLossParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.HingeLossParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.norm = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.HingeLossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "norm":
                    message.norm = reader.enum($root.caffe.HingeLossParameter.Norm);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.HingeLossParameter.prototype.norm = 1;

$root.caffe.HingeLossParameter.Norm = {
    "L1": 1,
    "L2": 2
};

$root.caffe.ImageDataParameter = class ImageDataParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ImageDataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.source = reader.string();
                    break;
                case 4:
                    message.batch_size = reader.uint32();
                    break;
                case 7:
                    message.rand_skip = reader.uint32();
                    break;
                case 8:
                    message.shuffle = reader.bool();
                    break;
                case 9:
                    message.new_height = reader.uint32();
                    break;
                case 10:
                    message.new_width = reader.uint32();
                    break;
                case 11:
                    message.is_color = reader.bool();
                    break;
                case 2:
                    message.scale = reader.float();
                    break;
                case 3:
                    message.mean_file = reader.string();
                    break;
                case 5:
                    message.crop_size = reader.uint32();
                    break;
                case 6:
                    message.mirror = reader.bool();
                    break;
                case 12:
                    message.root_folder = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ImageDataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "source":
                    message.source = reader.string();
                    break;
                case "batch_size":
                    message.batch_size = reader.uint32();
                    break;
                case "rand_skip":
                    message.rand_skip = reader.uint32();
                    break;
                case "shuffle":
                    message.shuffle = reader.bool();
                    break;
                case "new_height":
                    message.new_height = reader.uint32();
                    break;
                case "new_width":
                    message.new_width = reader.uint32();
                    break;
                case "is_color":
                    message.is_color = reader.bool();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "mean_file":
                    message.mean_file = reader.string();
                    break;
                case "crop_size":
                    message.crop_size = reader.uint32();
                    break;
                case "mirror":
                    message.mirror = reader.bool();
                    break;
                case "root_folder":
                    message.root_folder = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ImageDataParameter.prototype.source = "";
$root.caffe.ImageDataParameter.prototype.batch_size = 1;
$root.caffe.ImageDataParameter.prototype.rand_skip = 0;
$root.caffe.ImageDataParameter.prototype.shuffle = false;
$root.caffe.ImageDataParameter.prototype.new_height = 0;
$root.caffe.ImageDataParameter.prototype.new_width = 0;
$root.caffe.ImageDataParameter.prototype.is_color = true;
$root.caffe.ImageDataParameter.prototype.scale = 1;
$root.caffe.ImageDataParameter.prototype.mean_file = "";
$root.caffe.ImageDataParameter.prototype.crop_size = 0;
$root.caffe.ImageDataParameter.prototype.mirror = false;
$root.caffe.ImageDataParameter.prototype.root_folder = "";

$root.caffe.InfogainLossParameter = class InfogainLossParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.InfogainLossParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.source = reader.string();
                    break;
                case 2:
                    message.axis = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.InfogainLossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "source":
                    message.source = reader.string();
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.InfogainLossParameter.prototype.source = "";
$root.caffe.InfogainLossParameter.prototype.axis = 1;

$root.caffe.InnerProductParameter = class InnerProductParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.InnerProductParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_output = reader.uint32();
                    break;
                case 2:
                    message.bias_term = reader.bool();
                    break;
                case 3:
                    message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.axis = reader.int32();
                    break;
                case 6:
                    message.transpose = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.InnerProductParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_output":
                    message.num_output = reader.uint32();
                    break;
                case "bias_term":
                    message.bias_term = reader.bool();
                    break;
                case "weight_filler":
                    message.weight_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "transpose":
                    message.transpose = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.InnerProductParameter.prototype.num_output = 0;
$root.caffe.InnerProductParameter.prototype.bias_term = true;
$root.caffe.InnerProductParameter.prototype.weight_filler = null;
$root.caffe.InnerProductParameter.prototype.bias_filler = null;
$root.caffe.InnerProductParameter.prototype.axis = 1;
$root.caffe.InnerProductParameter.prototype.transpose = false;

$root.caffe.InputParameter = class InputParameter {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.InputParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.InputParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape.push($root.caffe.BlobShape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.LogParameter = class LogParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.LogParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base = reader.float();
                    break;
                case 2:
                    message.scale = reader.float();
                    break;
                case 3:
                    message.shift = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.LogParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base":
                    message.base = reader.float();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "shift":
                    message.shift = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.LogParameter.prototype.base = -1;
$root.caffe.LogParameter.prototype.scale = 1;
$root.caffe.LogParameter.prototype.shift = 0;

$root.caffe.LRNParameter = class LRNParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.LRNParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.local_size = reader.uint32();
                    break;
                case 2:
                    message.alpha = reader.float();
                    break;
                case 3:
                    message.beta = reader.float();
                    break;
                case 4:
                    message.norm_region = reader.int32();
                    break;
                case 5:
                    message.k = reader.float();
                    break;
                case 6:
                    message.engine = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.LRNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "local_size":
                    message.local_size = reader.uint32();
                    break;
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                case "norm_region":
                    message.norm_region = reader.enum($root.caffe.LRNParameter.NormRegion);
                    break;
                case "k":
                    message.k = reader.float();
                    break;
                case "engine":
                    message.engine = reader.enum($root.caffe.LRNParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.LRNParameter.prototype.local_size = 5;
$root.caffe.LRNParameter.prototype.alpha = 1;
$root.caffe.LRNParameter.prototype.beta = 0.75;
$root.caffe.LRNParameter.prototype.norm_region = 0;
$root.caffe.LRNParameter.prototype.k = 1;
$root.caffe.LRNParameter.prototype.engine = 0;

$root.caffe.LRNParameter.NormRegion = {
    "ACROSS_CHANNELS": 0,
    "WITHIN_CHANNEL": 1
};

$root.caffe.LRNParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.MemoryDataParameter = class MemoryDataParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.MemoryDataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.batch_size = reader.uint32();
                    break;
                case 2:
                    message.channels = reader.uint32();
                    break;
                case 3:
                    message.height = reader.uint32();
                    break;
                case 4:
                    message.width = reader.uint32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.MemoryDataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "batch_size":
                    message.batch_size = reader.uint32();
                    break;
                case "channels":
                    message.channels = reader.uint32();
                    break;
                case "height":
                    message.height = reader.uint32();
                    break;
                case "width":
                    message.width = reader.uint32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.MemoryDataParameter.prototype.batch_size = 0;
$root.caffe.MemoryDataParameter.prototype.channels = 0;
$root.caffe.MemoryDataParameter.prototype.height = 0;
$root.caffe.MemoryDataParameter.prototype.width = 0;

$root.caffe.MVNParameter = class MVNParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.MVNParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.normalize_variance = reader.bool();
                    break;
                case 2:
                    message.across_channels = reader.bool();
                    break;
                case 3:
                    message.eps = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.MVNParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "normalize_variance":
                    message.normalize_variance = reader.bool();
                    break;
                case "across_channels":
                    message.across_channels = reader.bool();
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

$root.caffe.MVNParameter.prototype.normalize_variance = true;
$root.caffe.MVNParameter.prototype.across_channels = false;
$root.caffe.MVNParameter.prototype.eps = 1e-9;

$root.caffe.ParameterParameter = class ParameterParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ParameterParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ParameterParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.caffe.BlobShape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ParameterParameter.prototype.shape = null;

$root.caffe.PoolingParameter = class PoolingParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.PoolingParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pool = reader.int32();
                    break;
                case 4:
                    message.pad = reader.uint32();
                    break;
                case 9:
                    message.pad_h = reader.uint32();
                    break;
                case 10:
                    message.pad_w = reader.uint32();
                    break;
                case 2:
                    message.kernel_size = reader.uint32();
                    break;
                case 5:
                    message.kernel_h = reader.uint32();
                    break;
                case 6:
                    message.kernel_w = reader.uint32();
                    break;
                case 3:
                    message.stride = reader.uint32();
                    break;
                case 7:
                    message.stride_h = reader.uint32();
                    break;
                case 8:
                    message.stride_w = reader.uint32();
                    break;
                case 11:
                    message.engine = reader.int32();
                    break;
                case 12:
                    message.global_pooling = reader.bool();
                    break;
                case 13:
                    message.round_mode = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.PoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pool":
                    message.pool = reader.enum($root.caffe.PoolingParameter.PoolMethod);
                    break;
                case "pad":
                    message.pad = reader.uint32();
                    break;
                case "pad_h":
                    message.pad_h = reader.uint32();
                    break;
                case "pad_w":
                    message.pad_w = reader.uint32();
                    break;
                case "kernel_size":
                    message.kernel_size = reader.uint32();
                    break;
                case "kernel_h":
                    message.kernel_h = reader.uint32();
                    break;
                case "kernel_w":
                    message.kernel_w = reader.uint32();
                    break;
                case "stride":
                    message.stride = reader.uint32();
                    break;
                case "stride_h":
                    message.stride_h = reader.uint32();
                    break;
                case "stride_w":
                    message.stride_w = reader.uint32();
                    break;
                case "engine":
                    message.engine = reader.enum($root.caffe.PoolingParameter.Engine);
                    break;
                case "global_pooling":
                    message.global_pooling = reader.bool();
                    break;
                case "round_mode":
                    message.round_mode = reader.enum($root.caffe.PoolingParameter.RoundMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.PoolingParameter.prototype.pool = 0;
$root.caffe.PoolingParameter.prototype.pad = 0;
$root.caffe.PoolingParameter.prototype.pad_h = 0;
$root.caffe.PoolingParameter.prototype.pad_w = 0;
$root.caffe.PoolingParameter.prototype.kernel_size = 0;
$root.caffe.PoolingParameter.prototype.kernel_h = 0;
$root.caffe.PoolingParameter.prototype.kernel_w = 0;
$root.caffe.PoolingParameter.prototype.stride = 1;
$root.caffe.PoolingParameter.prototype.stride_h = 0;
$root.caffe.PoolingParameter.prototype.stride_w = 0;
$root.caffe.PoolingParameter.prototype.engine = 0;
$root.caffe.PoolingParameter.prototype.global_pooling = false;
$root.caffe.PoolingParameter.prototype.round_mode = 0;

$root.caffe.PoolingParameter.PoolMethod = {
    "MAX": 0,
    "AVE": 1,
    "STOCHASTIC": 2
};

$root.caffe.PoolingParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.PoolingParameter.RoundMode = {
    "CEIL": 0,
    "FLOOR": 1
};

$root.caffe.PowerParameter = class PowerParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.PowerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.power = reader.float();
                    break;
                case 2:
                    message.scale = reader.float();
                    break;
                case 3:
                    message.shift = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.PowerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "power":
                    message.power = reader.float();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "shift":
                    message.shift = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.PowerParameter.prototype.power = 1;
$root.caffe.PowerParameter.prototype.scale = 1;
$root.caffe.PowerParameter.prototype.shift = 0;

$root.caffe.PythonParameter = class PythonParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.PythonParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.module = reader.string();
                    break;
                case 2:
                    message.layer = reader.string();
                    break;
                case 3:
                    message.param_str = reader.string();
                    break;
                case 4:
                    message.share_in_parallel = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.PythonParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "module":
                    message.module = reader.string();
                    break;
                case "layer":
                    message.layer = reader.string();
                    break;
                case "param_str":
                    message.param_str = reader.string();
                    break;
                case "share_in_parallel":
                    message.share_in_parallel = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.PythonParameter.prototype.module = "";
$root.caffe.PythonParameter.prototype.layer = "";
$root.caffe.PythonParameter.prototype.param_str = "";
$root.caffe.PythonParameter.prototype.share_in_parallel = false;

$root.caffe.RecurrentParameter = class RecurrentParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.RecurrentParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_output = reader.uint32();
                    break;
                case 2:
                    message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.debug_info = reader.bool();
                    break;
                case 5:
                    message.expose_hidden = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.RecurrentParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_output":
                    message.num_output = reader.uint32();
                    break;
                case "weight_filler":
                    message.weight_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "debug_info":
                    message.debug_info = reader.bool();
                    break;
                case "expose_hidden":
                    message.expose_hidden = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.RecurrentParameter.prototype.num_output = 0;
$root.caffe.RecurrentParameter.prototype.weight_filler = null;
$root.caffe.RecurrentParameter.prototype.bias_filler = null;
$root.caffe.RecurrentParameter.prototype.debug_info = false;
$root.caffe.RecurrentParameter.prototype.expose_hidden = false;

$root.caffe.ReductionParameter = class ReductionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ReductionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.operation = reader.int32();
                    break;
                case 2:
                    message.axis = reader.int32();
                    break;
                case 3:
                    message.coeff = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ReductionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "operation":
                    message.operation = reader.enum($root.caffe.ReductionParameter.ReductionOp);
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "coeff":
                    message.coeff = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ReductionParameter.prototype.operation = 1;
$root.caffe.ReductionParameter.prototype.axis = 0;
$root.caffe.ReductionParameter.prototype.coeff = 1;

$root.caffe.ReductionParameter.ReductionOp = {
    "SUM": 1,
    "ASUM": 2,
    "SUMSQ": 3,
    "MEAN": 4
};

$root.caffe.ReLUParameter = class ReLUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ReLUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.negative_slope = reader.float();
                    break;
                case 2:
                    message.engine = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "negative_slope":
                    message.negative_slope = reader.float();
                    break;
                case "engine":
                    message.engine = reader.enum($root.caffe.ReLUParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ReLUParameter.prototype.negative_slope = 0;
$root.caffe.ReLUParameter.prototype.engine = 0;

$root.caffe.ReLUParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.ReshapeParameter = class ReshapeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ReshapeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.axis = reader.int32();
                    break;
                case 3:
                    message.num_axes = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ReshapeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = $root.caffe.BlobShape.decodeText(reader);
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "num_axes":
                    message.num_axes = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ReshapeParameter.prototype.shape = null;
$root.caffe.ReshapeParameter.prototype.axis = 0;
$root.caffe.ReshapeParameter.prototype.num_axes = -1;

$root.caffe.ScaleParameter = class ScaleParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ScaleParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int32();
                    break;
                case 2:
                    message.num_axes = reader.int32();
                    break;
                case 3:
                    message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bias_term = reader.bool();
                    break;
                case 5:
                    message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ScaleParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "num_axes":
                    message.num_axes = reader.int32();
                    break;
                case "filler":
                    message.filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_term":
                    message.bias_term = reader.bool();
                    break;
                case "bias_filler":
                    message.bias_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ScaleParameter.prototype.axis = 1;
$root.caffe.ScaleParameter.prototype.num_axes = 1;
$root.caffe.ScaleParameter.prototype.filler = null;
$root.caffe.ScaleParameter.prototype.bias_term = false;
$root.caffe.ScaleParameter.prototype.bias_filler = null;

$root.caffe.SigmoidParameter = class SigmoidParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.SigmoidParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.engine = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SigmoidParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "engine":
                    message.engine = reader.enum($root.caffe.SigmoidParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SigmoidParameter.prototype.engine = 0;

$root.caffe.SigmoidParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.SliceParameter = class SliceParameter {

    constructor() {
        this.slice_point = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.SliceParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 3:
                    message.axis = reader.int32();
                    break;
                case 2:
                    message.slice_point = reader.array(message.slice_point, () => reader.uint32(), tag);
                    break;
                case 1:
                    message.slice_dim = reader.uint32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SliceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "slice_point":
                    reader.array(message.slice_point, () => reader.uint32());
                    break;
                case "slice_dim":
                    message.slice_dim = reader.uint32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SliceParameter.prototype.axis = 1;
$root.caffe.SliceParameter.prototype.slice_dim = 1;

$root.caffe.SoftmaxParameter = class SoftmaxParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.SoftmaxParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.engine = reader.int32();
                    break;
                case 2:
                    message.axis = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SoftmaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "engine":
                    message.engine = reader.enum($root.caffe.SoftmaxParameter.Engine);
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SoftmaxParameter.prototype.engine = 0;
$root.caffe.SoftmaxParameter.prototype.axis = 1;

$root.caffe.SoftmaxParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.SwishParameter = class SwishParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.SwishParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.beta = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SwishParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "beta":
                    message.beta = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SwishParameter.prototype.beta = 1;

$root.caffe.TanHParameter = class TanHParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.TanHParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.engine = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.TanHParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "engine":
                    message.engine = reader.enum($root.caffe.TanHParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.TanHParameter.prototype.engine = 0;

$root.caffe.TanHParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.TileParameter = class TileParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.TileParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int32();
                    break;
                case 2:
                    message.tiles = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.TileParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "tiles":
                    message.tiles = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.TileParameter.prototype.axis = 1;
$root.caffe.TileParameter.prototype.tiles = 0;

$root.caffe.ThresholdParameter = class ThresholdParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.ThresholdParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.threshold = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.ThresholdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "threshold":
                    message.threshold = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.ThresholdParameter.prototype.threshold = 0;

$root.caffe.WindowDataParameter = class WindowDataParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.WindowDataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.source = reader.string();
                    break;
                case 2:
                    message.scale = reader.float();
                    break;
                case 3:
                    message.mean_file = reader.string();
                    break;
                case 4:
                    message.batch_size = reader.uint32();
                    break;
                case 5:
                    message.crop_size = reader.uint32();
                    break;
                case 6:
                    message.mirror = reader.bool();
                    break;
                case 7:
                    message.fg_threshold = reader.float();
                    break;
                case 8:
                    message.bg_threshold = reader.float();
                    break;
                case 9:
                    message.fg_fraction = reader.float();
                    break;
                case 10:
                    message.context_pad = reader.uint32();
                    break;
                case 11:
                    message.crop_mode = reader.string();
                    break;
                case 12:
                    message.cache_images = reader.bool();
                    break;
                case 13:
                    message.root_folder = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.WindowDataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "source":
                    message.source = reader.string();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "mean_file":
                    message.mean_file = reader.string();
                    break;
                case "batch_size":
                    message.batch_size = reader.uint32();
                    break;
                case "crop_size":
                    message.crop_size = reader.uint32();
                    break;
                case "mirror":
                    message.mirror = reader.bool();
                    break;
                case "fg_threshold":
                    message.fg_threshold = reader.float();
                    break;
                case "bg_threshold":
                    message.bg_threshold = reader.float();
                    break;
                case "fg_fraction":
                    message.fg_fraction = reader.float();
                    break;
                case "context_pad":
                    message.context_pad = reader.uint32();
                    break;
                case "crop_mode":
                    message.crop_mode = reader.string();
                    break;
                case "cache_images":
                    message.cache_images = reader.bool();
                    break;
                case "root_folder":
                    message.root_folder = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.WindowDataParameter.prototype.source = "";
$root.caffe.WindowDataParameter.prototype.scale = 1;
$root.caffe.WindowDataParameter.prototype.mean_file = "";
$root.caffe.WindowDataParameter.prototype.batch_size = 0;
$root.caffe.WindowDataParameter.prototype.crop_size = 0;
$root.caffe.WindowDataParameter.prototype.mirror = false;
$root.caffe.WindowDataParameter.prototype.fg_threshold = 0.5;
$root.caffe.WindowDataParameter.prototype.bg_threshold = 0.5;
$root.caffe.WindowDataParameter.prototype.fg_fraction = 0.25;
$root.caffe.WindowDataParameter.prototype.context_pad = 0;
$root.caffe.WindowDataParameter.prototype.crop_mode = "warp";
$root.caffe.WindowDataParameter.prototype.cache_images = false;
$root.caffe.WindowDataParameter.prototype.root_folder = "";

$root.caffe.SPPParameter = class SPPParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.SPPParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pyramid_height = reader.uint32();
                    break;
                case 2:
                    message.pool = reader.int32();
                    break;
                case 6:
                    message.engine = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.SPPParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pyramid_height":
                    message.pyramid_height = reader.uint32();
                    break;
                case "pool":
                    message.pool = reader.enum($root.caffe.SPPParameter.PoolMethod);
                    break;
                case "engine":
                    message.engine = reader.enum($root.caffe.SPPParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.SPPParameter.prototype.pyramid_height = 0;
$root.caffe.SPPParameter.prototype.pool = 0;
$root.caffe.SPPParameter.prototype.engine = 0;

$root.caffe.SPPParameter.PoolMethod = {
    "MAX": 0,
    "AVE": 1,
    "STOCHASTIC": 2
};

$root.caffe.SPPParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

$root.caffe.V1LayerParameter = class V1LayerParameter {

    constructor() {
        this.bottom = [];
        this.top = [];
        this.include = [];
        this.exclude = [];
        this.blobs = [];
        this.param = [];
        this.blob_share_mode = [];
        this.blobs_lr = [];
        this.weight_decay = [];
        this.loss_weight = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.V1LayerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.bottom.push(reader.string());
                    break;
                case 3:
                    message.top.push(reader.string());
                    break;
                case 4:
                    message.name = reader.string();
                    break;
                case 32:
                    message.include.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 33:
                    message.exclude.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.type = reader.int32();
                    break;
                case 6:
                    message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                case 1001:
                    message.param.push(reader.string());
                    break;
                case 1002:
                    message.blob_share_mode = reader.array(message.blob_share_mode, () => reader.int32(), tag);
                    break;
                case 7:
                    message.blobs_lr = reader.floats(message.blobs_lr, tag);
                    break;
                case 8:
                    message.weight_decay = reader.floats(message.weight_decay, tag);
                    break;
                case 35:
                    message.loss_weight = reader.floats(message.loss_weight, tag);
                    break;
                case 27:
                    message.accuracy_param = $root.caffe.AccuracyParameter.decode(reader, reader.uint32());
                    break;
                case 23:
                    message.argmax_param = $root.caffe.ArgMaxParameter.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.concat_param = $root.caffe.ConcatParameter.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.convolution_param = $root.caffe.ConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.data_param = $root.caffe.DataParameter.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.dropout_param = $root.caffe.DropoutParameter.decode(reader, reader.uint32());
                    break;
                case 26:
                    message.dummy_data_param = $root.caffe.DummyDataParameter.decode(reader, reader.uint32());
                    break;
                case 24:
                    message.eltwise_param = $root.caffe.EltwiseParameter.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.exp_param = $root.caffe.ExpParameter.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.hdf5_data_param = $root.caffe.HDF5DataParameter.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                    break;
                case 29:
                    message.hinge_loss_param = $root.caffe.HingeLossParameter.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.image_data_param = $root.caffe.ImageDataParameter.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.infogain_loss_param = $root.caffe.InfogainLossParameter.decode(reader, reader.uint32());
                    break;
                case 17:
                    message.inner_product_param = $root.caffe.InnerProductParameter.decode(reader, reader.uint32());
                    break;
                case 18:
                    message.lrn_param = $root.caffe.LRNParameter.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.memory_data_param = $root.caffe.MemoryDataParameter.decode(reader, reader.uint32());
                    break;
                case 34:
                    message.mvn_param = $root.caffe.MVNParameter.decode(reader, reader.uint32());
                    break;
                case 19:
                    message.pooling_param = $root.caffe.PoolingParameter.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.power_param = $root.caffe.PowerParameter.decode(reader, reader.uint32());
                    break;
                case 30:
                    message.relu_param = $root.caffe.ReLUParameter.decode(reader, reader.uint32());
                    break;
                case 38:
                    message.sigmoid_param = $root.caffe.SigmoidParameter.decode(reader, reader.uint32());
                    break;
                case 39:
                    message.softmax_param = $root.caffe.SoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.slice_param = $root.caffe.SliceParameter.decode(reader, reader.uint32());
                    break;
                case 37:
                    message.tanh_param = $root.caffe.TanHParameter.decode(reader, reader.uint32());
                    break;
                case 25:
                    message.threshold_param = $root.caffe.ThresholdParameter.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.window_data_param = $root.caffe.WindowDataParameter.decode(reader, reader.uint32());
                    break;
                case 36:
                    message.transform_param = $root.caffe.TransformationParameter.decode(reader, reader.uint32());
                    break;
                case 42:
                    message.loss_param = $root.caffe.LossParameter.decode(reader, reader.uint32());
                    break;
                case 1:
                    message.layer = $root.caffe.V0LayerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.V1LayerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "bottom":
                    reader.array(message.bottom, () => reader.string());
                    break;
                case "top":
                    reader.array(message.top, () => reader.string());
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "include":
                    message.include.push($root.caffe.NetStateRule.decodeText(reader));
                    break;
                case "exclude":
                    message.exclude.push($root.caffe.NetStateRule.decodeText(reader));
                    break;
                case "type":
                    message.type = reader.enum($root.caffe.V1LayerParameter.LayerType);
                    break;
                case "blobs":
                    message.blobs.push($root.caffe.BlobProto.decodeText(reader));
                    break;
                case "param":
                    reader.array(message.param, () => reader.string());
                    break;
                case "blob_share_mode":
                    reader.array(message.blob_share_mode, () => reader.enum($root.caffe.V1LayerParameter.DimCheckMode));
                    break;
                case "blobs_lr":
                    reader.array(message.blobs_lr, () => reader.float());
                    break;
                case "weight_decay":
                    reader.array(message.weight_decay, () => reader.float());
                    break;
                case "loss_weight":
                    reader.array(message.loss_weight, () => reader.float());
                    break;
                case "accuracy_param":
                    message.accuracy_param = $root.caffe.AccuracyParameter.decodeText(reader);
                    break;
                case "argmax_param":
                    message.argmax_param = $root.caffe.ArgMaxParameter.decodeText(reader);
                    break;
                case "concat_param":
                    message.concat_param = $root.caffe.ConcatParameter.decodeText(reader);
                    break;
                case "contrastive_loss_param":
                    message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decodeText(reader);
                    break;
                case "convolution_param":
                    message.convolution_param = $root.caffe.ConvolutionParameter.decodeText(reader);
                    break;
                case "data_param":
                    message.data_param = $root.caffe.DataParameter.decodeText(reader);
                    break;
                case "dropout_param":
                    message.dropout_param = $root.caffe.DropoutParameter.decodeText(reader);
                    break;
                case "dummy_data_param":
                    message.dummy_data_param = $root.caffe.DummyDataParameter.decodeText(reader);
                    break;
                case "eltwise_param":
                    message.eltwise_param = $root.caffe.EltwiseParameter.decodeText(reader);
                    break;
                case "exp_param":
                    message.exp_param = $root.caffe.ExpParameter.decodeText(reader);
                    break;
                case "hdf5_data_param":
                    message.hdf5_data_param = $root.caffe.HDF5DataParameter.decodeText(reader);
                    break;
                case "hdf5_output_param":
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader);
                    break;
                case "hinge_loss_param":
                    message.hinge_loss_param = $root.caffe.HingeLossParameter.decodeText(reader);
                    break;
                case "image_data_param":
                    message.image_data_param = $root.caffe.ImageDataParameter.decodeText(reader);
                    break;
                case "infogain_loss_param":
                    message.infogain_loss_param = $root.caffe.InfogainLossParameter.decodeText(reader);
                    break;
                case "inner_product_param":
                    message.inner_product_param = $root.caffe.InnerProductParameter.decodeText(reader);
                    break;
                case "lrn_param":
                    message.lrn_param = $root.caffe.LRNParameter.decodeText(reader);
                    break;
                case "memory_data_param":
                    message.memory_data_param = $root.caffe.MemoryDataParameter.decodeText(reader);
                    break;
                case "mvn_param":
                    message.mvn_param = $root.caffe.MVNParameter.decodeText(reader);
                    break;
                case "pooling_param":
                    message.pooling_param = $root.caffe.PoolingParameter.decodeText(reader);
                    break;
                case "power_param":
                    message.power_param = $root.caffe.PowerParameter.decodeText(reader);
                    break;
                case "relu_param":
                    message.relu_param = $root.caffe.ReLUParameter.decodeText(reader);
                    break;
                case "sigmoid_param":
                    message.sigmoid_param = $root.caffe.SigmoidParameter.decodeText(reader);
                    break;
                case "softmax_param":
                    message.softmax_param = $root.caffe.SoftmaxParameter.decodeText(reader);
                    break;
                case "slice_param":
                    message.slice_param = $root.caffe.SliceParameter.decodeText(reader);
                    break;
                case "tanh_param":
                    message.tanh_param = $root.caffe.TanHParameter.decodeText(reader);
                    break;
                case "threshold_param":
                    message.threshold_param = $root.caffe.ThresholdParameter.decodeText(reader);
                    break;
                case "window_data_param":
                    message.window_data_param = $root.caffe.WindowDataParameter.decodeText(reader);
                    break;
                case "transform_param":
                    message.transform_param = $root.caffe.TransformationParameter.decodeText(reader);
                    break;
                case "loss_param":
                    message.loss_param = $root.caffe.LossParameter.decodeText(reader);
                    break;
                case "layer":
                    message.layer = $root.caffe.V0LayerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.V1LayerParameter.prototype.name = "";
$root.caffe.V1LayerParameter.prototype.type = 0;
$root.caffe.V1LayerParameter.prototype.accuracy_param = null;
$root.caffe.V1LayerParameter.prototype.argmax_param = null;
$root.caffe.V1LayerParameter.prototype.concat_param = null;
$root.caffe.V1LayerParameter.prototype.contrastive_loss_param = null;
$root.caffe.V1LayerParameter.prototype.convolution_param = null;
$root.caffe.V1LayerParameter.prototype.data_param = null;
$root.caffe.V1LayerParameter.prototype.dropout_param = null;
$root.caffe.V1LayerParameter.prototype.dummy_data_param = null;
$root.caffe.V1LayerParameter.prototype.eltwise_param = null;
$root.caffe.V1LayerParameter.prototype.exp_param = null;
$root.caffe.V1LayerParameter.prototype.hdf5_data_param = null;
$root.caffe.V1LayerParameter.prototype.hdf5_output_param = null;
$root.caffe.V1LayerParameter.prototype.hinge_loss_param = null;
$root.caffe.V1LayerParameter.prototype.image_data_param = null;
$root.caffe.V1LayerParameter.prototype.infogain_loss_param = null;
$root.caffe.V1LayerParameter.prototype.inner_product_param = null;
$root.caffe.V1LayerParameter.prototype.lrn_param = null;
$root.caffe.V1LayerParameter.prototype.memory_data_param = null;
$root.caffe.V1LayerParameter.prototype.mvn_param = null;
$root.caffe.V1LayerParameter.prototype.pooling_param = null;
$root.caffe.V1LayerParameter.prototype.power_param = null;
$root.caffe.V1LayerParameter.prototype.relu_param = null;
$root.caffe.V1LayerParameter.prototype.sigmoid_param = null;
$root.caffe.V1LayerParameter.prototype.softmax_param = null;
$root.caffe.V1LayerParameter.prototype.slice_param = null;
$root.caffe.V1LayerParameter.prototype.tanh_param = null;
$root.caffe.V1LayerParameter.prototype.threshold_param = null;
$root.caffe.V1LayerParameter.prototype.window_data_param = null;
$root.caffe.V1LayerParameter.prototype.transform_param = null;
$root.caffe.V1LayerParameter.prototype.loss_param = null;
$root.caffe.V1LayerParameter.prototype.layer = null;

$root.caffe.V1LayerParameter.LayerType = {
    "NONE": 0,
    "ABSVAL": 35,
    "ACCURACY": 1,
    "ARGMAX": 30,
    "BNLL": 2,
    "CONCAT": 3,
    "CONTRASTIVE_LOSS": 37,
    "CONVOLUTION": 4,
    "DATA": 5,
    "DECONVOLUTION": 39,
    "DROPOUT": 6,
    "DUMMY_DATA": 32,
    "EUCLIDEAN_LOSS": 7,
    "ELTWISE": 25,
    "EXP": 38,
    "FLATTEN": 8,
    "HDF5_DATA": 9,
    "HDF5_OUTPUT": 10,
    "HINGE_LOSS": 28,
    "IM2COL": 11,
    "IMAGE_DATA": 12,
    "INFOGAIN_LOSS": 13,
    "INNER_PRODUCT": 14,
    "LRN": 15,
    "MEMORY_DATA": 29,
    "MULTINOMIAL_LOGISTIC_LOSS": 16,
    "MVN": 34,
    "POOLING": 17,
    "POWER": 26,
    "RELU": 18,
    "SIGMOID": 19,
    "SIGMOID_CROSS_ENTROPY_LOSS": 27,
    "SILENCE": 36,
    "SOFTMAX": 20,
    "SOFTMAX_LOSS": 21,
    "SPLIT": 22,
    "SLICE": 33,
    "TANH": 23,
    "WINDOW_DATA": 24,
    "THRESHOLD": 31
};

$root.caffe.V1LayerParameter.DimCheckMode = {
    "STRICT": 0,
    "PERMISSIVE": 1
};

$root.caffe.V0LayerParameter = class V0LayerParameter {

    constructor() {
        this.blobs = [];
        this.blobs_lr = [];
        this.weight_decay = [];
    }

    static decode(reader, length) {
        const message = new $root.caffe.V0LayerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.num_output = reader.uint32();
                    break;
                case 4:
                    message.biasterm = reader.bool();
                    break;
                case 5:
                    message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.pad = reader.uint32();
                    break;
                case 8:
                    message.kernelsize = reader.uint32();
                    break;
                case 9:
                    message.group = reader.uint32();
                    break;
                case 10:
                    message.stride = reader.uint32();
                    break;
                case 11:
                    message.pool = reader.int32();
                    break;
                case 12:
                    message.dropout_ratio = reader.float();
                    break;
                case 13:
                    message.local_size = reader.uint32();
                    break;
                case 14:
                    message.alpha = reader.float();
                    break;
                case 15:
                    message.beta = reader.float();
                    break;
                case 22:
                    message.k = reader.float();
                    break;
                case 16:
                    message.source = reader.string();
                    break;
                case 17:
                    message.scale = reader.float();
                    break;
                case 18:
                    message.meanfile = reader.string();
                    break;
                case 19:
                    message.batchsize = reader.uint32();
                    break;
                case 20:
                    message.cropsize = reader.uint32();
                    break;
                case 21:
                    message.mirror = reader.bool();
                    break;
                case 50:
                    message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                case 51:
                    message.blobs_lr = reader.floats(message.blobs_lr, tag);
                    break;
                case 52:
                    message.weight_decay = reader.floats(message.weight_decay, tag);
                    break;
                case 53:
                    message.rand_skip = reader.uint32();
                    break;
                case 54:
                    message.det_fg_threshold = reader.float();
                    break;
                case 55:
                    message.det_bg_threshold = reader.float();
                    break;
                case 56:
                    message.det_fg_fraction = reader.float();
                    break;
                case 58:
                    message.det_context_pad = reader.uint32();
                    break;
                case 59:
                    message.det_crop_mode = reader.string();
                    break;
                case 60:
                    message.new_num = reader.int32();
                    break;
                case 61:
                    message.new_channels = reader.int32();
                    break;
                case 62:
                    message.new_height = reader.int32();
                    break;
                case 63:
                    message.new_width = reader.int32();
                    break;
                case 64:
                    message.shuffle_images = reader.bool();
                    break;
                case 65:
                    message.concat_dim = reader.uint32();
                    break;
                case 1001:
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.V0LayerParameter();
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
                case "num_output":
                    message.num_output = reader.uint32();
                    break;
                case "biasterm":
                    message.biasterm = reader.bool();
                    break;
                case "weight_filler":
                    message.weight_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "pad":
                    message.pad = reader.uint32();
                    break;
                case "kernelsize":
                    message.kernelsize = reader.uint32();
                    break;
                case "group":
                    message.group = reader.uint32();
                    break;
                case "stride":
                    message.stride = reader.uint32();
                    break;
                case "pool":
                    message.pool = reader.enum($root.caffe.V0LayerParameter.PoolMethod);
                    break;
                case "dropout_ratio":
                    message.dropout_ratio = reader.float();
                    break;
                case "local_size":
                    message.local_size = reader.uint32();
                    break;
                case "alpha":
                    message.alpha = reader.float();
                    break;
                case "beta":
                    message.beta = reader.float();
                    break;
                case "k":
                    message.k = reader.float();
                    break;
                case "source":
                    message.source = reader.string();
                    break;
                case "scale":
                    message.scale = reader.float();
                    break;
                case "meanfile":
                    message.meanfile = reader.string();
                    break;
                case "batchsize":
                    message.batchsize = reader.uint32();
                    break;
                case "cropsize":
                    message.cropsize = reader.uint32();
                    break;
                case "mirror":
                    message.mirror = reader.bool();
                    break;
                case "blobs":
                    message.blobs.push($root.caffe.BlobProto.decodeText(reader));
                    break;
                case "blobs_lr":
                    reader.array(message.blobs_lr, () => reader.float());
                    break;
                case "weight_decay":
                    reader.array(message.weight_decay, () => reader.float());
                    break;
                case "rand_skip":
                    message.rand_skip = reader.uint32();
                    break;
                case "det_fg_threshold":
                    message.det_fg_threshold = reader.float();
                    break;
                case "det_bg_threshold":
                    message.det_bg_threshold = reader.float();
                    break;
                case "det_fg_fraction":
                    message.det_fg_fraction = reader.float();
                    break;
                case "det_context_pad":
                    message.det_context_pad = reader.uint32();
                    break;
                case "det_crop_mode":
                    message.det_crop_mode = reader.string();
                    break;
                case "new_num":
                    message.new_num = reader.int32();
                    break;
                case "new_channels":
                    message.new_channels = reader.int32();
                    break;
                case "new_height":
                    message.new_height = reader.int32();
                    break;
                case "new_width":
                    message.new_width = reader.int32();
                    break;
                case "shuffle_images":
                    message.shuffle_images = reader.bool();
                    break;
                case "concat_dim":
                    message.concat_dim = reader.uint32();
                    break;
                case "hdf5_output_param":
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.V0LayerParameter.prototype.name = "";
$root.caffe.V0LayerParameter.prototype.type = "";
$root.caffe.V0LayerParameter.prototype.num_output = 0;
$root.caffe.V0LayerParameter.prototype.biasterm = true;
$root.caffe.V0LayerParameter.prototype.weight_filler = null;
$root.caffe.V0LayerParameter.prototype.bias_filler = null;
$root.caffe.V0LayerParameter.prototype.pad = 0;
$root.caffe.V0LayerParameter.prototype.kernelsize = 0;
$root.caffe.V0LayerParameter.prototype.group = 1;
$root.caffe.V0LayerParameter.prototype.stride = 1;
$root.caffe.V0LayerParameter.prototype.pool = 0;
$root.caffe.V0LayerParameter.prototype.dropout_ratio = 0.5;
$root.caffe.V0LayerParameter.prototype.local_size = 5;
$root.caffe.V0LayerParameter.prototype.alpha = 1;
$root.caffe.V0LayerParameter.prototype.beta = 0.75;
$root.caffe.V0LayerParameter.prototype.k = 1;
$root.caffe.V0LayerParameter.prototype.source = "";
$root.caffe.V0LayerParameter.prototype.scale = 1;
$root.caffe.V0LayerParameter.prototype.meanfile = "";
$root.caffe.V0LayerParameter.prototype.batchsize = 0;
$root.caffe.V0LayerParameter.prototype.cropsize = 0;
$root.caffe.V0LayerParameter.prototype.mirror = false;
$root.caffe.V0LayerParameter.prototype.rand_skip = 0;
$root.caffe.V0LayerParameter.prototype.det_fg_threshold = 0.5;
$root.caffe.V0LayerParameter.prototype.det_bg_threshold = 0.5;
$root.caffe.V0LayerParameter.prototype.det_fg_fraction = 0.25;
$root.caffe.V0LayerParameter.prototype.det_context_pad = 0;
$root.caffe.V0LayerParameter.prototype.det_crop_mode = "warp";
$root.caffe.V0LayerParameter.prototype.new_num = 0;
$root.caffe.V0LayerParameter.prototype.new_channels = 0;
$root.caffe.V0LayerParameter.prototype.new_height = 0;
$root.caffe.V0LayerParameter.prototype.new_width = 0;
$root.caffe.V0LayerParameter.prototype.shuffle_images = false;
$root.caffe.V0LayerParameter.prototype.concat_dim = 1;
$root.caffe.V0LayerParameter.prototype.hdf5_output_param = null;

$root.caffe.V0LayerParameter.PoolMethod = {
    "MAX": 0,
    "AVE": 1,
    "STOCHASTIC": 2
};

$root.caffe.PReLUParameter = class PReLUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.caffe.PReLUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.channel_shared = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.caffe.PReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "filler":
                    message.filler = $root.caffe.FillerParameter.decodeText(reader);
                    break;
                case "channel_shared":
                    message.channel_shared = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.caffe.PReLUParameter.prototype.filler = null;
$root.caffe.PReLUParameter.prototype.channel_shared = false;
