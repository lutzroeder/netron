
export const caffe = {};

caffe.BlobShape = class BlobShape {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new caffe.BlobShape();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.BlobShape();
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

caffe.BlobProto = class BlobProto {

    constructor() {
        this.data = [];
        this.diff = [];
        this.double_data = [];
        this.double_diff = [];
    }

    static decode(reader, length) {
        const message = new caffe.BlobProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 7:
                    message.shape = caffe.BlobShape.decode(reader, reader.uint32());
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
        const message = new caffe.BlobProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = caffe.BlobShape.decodeText(reader);
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

caffe.BlobProto.prototype.shape = null;
caffe.BlobProto.prototype.num = 0;
caffe.BlobProto.prototype.channels = 0;
caffe.BlobProto.prototype.height = 0;
caffe.BlobProto.prototype.width = 0;

caffe.BlobProtoVector = class BlobProtoVector {

    constructor() {
        this.blobs = [];
    }

    static decode(reader, length) {
        const message = new caffe.BlobProtoVector();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.blobs.push(caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.BlobProtoVector();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "blobs":
                    message.blobs.push(caffe.BlobProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.Datum = class Datum {

    constructor() {
        this.float_data = [];
    }

    static decode(reader, length) {
        const message = new caffe.Datum();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.Datum();
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

caffe.Datum.prototype.channels = 0;
caffe.Datum.prototype.height = 0;
caffe.Datum.prototype.width = 0;
caffe.Datum.prototype.data = new Uint8Array([]);
caffe.Datum.prototype.label = 0;
caffe.Datum.prototype.encoded = false;

caffe.FillerParameter = class FillerParameter {

    static decode(reader, length) {
        const message = new caffe.FillerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.FillerParameter();
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
                    message.variance_norm = reader.enum(caffe.FillerParameter.VarianceNorm);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.FillerParameter.prototype.type = "constant";
caffe.FillerParameter.prototype.value = 0;
caffe.FillerParameter.prototype.min = 0;
caffe.FillerParameter.prototype.max = 1;
caffe.FillerParameter.prototype.mean = 0;
caffe.FillerParameter.prototype.std = 1;
caffe.FillerParameter.prototype.sparse = -1;
caffe.FillerParameter.prototype.variance_norm = 0;

caffe.FillerParameter.VarianceNorm = {
    "FAN_IN": 0,
    "FAN_OUT": 1,
    "AVERAGE": 2
};

caffe.NetParameter = class NetParameter {

    constructor() {
        this.input = [];
        this.input_shape = [];
        this.input_dim = [];
        this.layer = [];
        this.layers = [];
    }

    static decode(reader, length) {
        const message = new caffe.NetParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.input_shape.push(caffe.BlobShape.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.input_dim = reader.array(message.input_dim, () => reader.int32(), tag);
                    break;
                case 5:
                    message.force_backward = reader.bool();
                    break;
                case 6:
                    message.state = caffe.NetState.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.debug_info = reader.bool();
                    break;
                case 100:
                    message.layer.push(caffe.LayerParameter.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.layers.push(caffe.V1LayerParameter.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.NetParameter();
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
                    message.input_shape.push(caffe.BlobShape.decodeText(reader));
                    break;
                case "input_dim":
                    reader.array(message.input_dim, () => reader.int32());
                    break;
                case "force_backward":
                    message.force_backward = reader.bool();
                    break;
                case "state":
                    message.state = caffe.NetState.decodeText(reader);
                    break;
                case "debug_info":
                    message.debug_info = reader.bool();
                    break;
                case "layer":
                    message.layer.push(caffe.LayerParameter.decodeText(reader));
                    break;
                case "layers":
                    message.layers.push(caffe.V1LayerParameter.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.NetParameter.prototype.name = "";
caffe.NetParameter.prototype.force_backward = false;
caffe.NetParameter.prototype.state = null;
caffe.NetParameter.prototype.debug_info = false;

caffe.SolverParameter = class SolverParameter {

    constructor() {
        this.test_net = [];
        this.test_net_param = [];
        this.test_state = [];
        this.test_iter = [];
        this.stepvalue = [];
        this.weights = [];
    }

    static decode(reader, length) {
        const message = new caffe.SolverParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 24:
                    message.net = reader.string();
                    break;
                case 25:
                    message.net_param = caffe.NetParameter.decode(reader, reader.uint32());
                    break;
                case 1:
                    message.train_net = reader.string();
                    break;
                case 2:
                    message.test_net.push(reader.string());
                    break;
                case 21:
                    message.train_net_param = caffe.NetParameter.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.test_net_param.push(caffe.NetParameter.decode(reader, reader.uint32()));
                    break;
                case 26:
                    message.train_state = caffe.NetState.decode(reader, reader.uint32());
                    break;
                case 27:
                    message.test_state.push(caffe.NetState.decode(reader, reader.uint32()));
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
        const message = new caffe.SolverParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "net":
                    message.net = reader.string();
                    break;
                case "net_param":
                    message.net_param = caffe.NetParameter.decodeText(reader);
                    break;
                case "train_net":
                    message.train_net = reader.string();
                    break;
                case "test_net":
                    reader.array(message.test_net, () => reader.string());
                    break;
                case "train_net_param":
                    message.train_net_param = caffe.NetParameter.decodeText(reader);
                    break;
                case "test_net_param":
                    message.test_net_param.push(caffe.NetParameter.decodeText(reader));
                    break;
                case "train_state":
                    message.train_state = caffe.NetState.decodeText(reader);
                    break;
                case "test_state":
                    message.test_state.push(caffe.NetState.decodeText(reader));
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
                    message.snapshot_format = reader.enum(caffe.SolverParameter.SnapshotFormat);
                    break;
                case "solver_mode":
                    message.solver_mode = reader.enum(caffe.SolverParameter.SolverMode);
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
                    message.solver_type = reader.enum(caffe.SolverParameter.SolverType);
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

caffe.SolverParameter.prototype.net = "";
caffe.SolverParameter.prototype.net_param = null;
caffe.SolverParameter.prototype.train_net = "";
caffe.SolverParameter.prototype.train_net_param = null;
caffe.SolverParameter.prototype.train_state = null;
caffe.SolverParameter.prototype.test_interval = 0;
caffe.SolverParameter.prototype.test_compute_loss = false;
caffe.SolverParameter.prototype.test_initialization = true;
caffe.SolverParameter.prototype.base_lr = 0;
caffe.SolverParameter.prototype.display = 0;
caffe.SolverParameter.prototype.average_loss = 1;
caffe.SolverParameter.prototype.max_iter = 0;
caffe.SolverParameter.prototype.iter_size = 1;
caffe.SolverParameter.prototype.lr_policy = "";
caffe.SolverParameter.prototype.gamma = 0;
caffe.SolverParameter.prototype.power = 0;
caffe.SolverParameter.prototype.momentum = 0;
caffe.SolverParameter.prototype.weight_decay = 0;
caffe.SolverParameter.prototype.regularization_type = "L2";
caffe.SolverParameter.prototype.stepsize = 0;
caffe.SolverParameter.prototype.clip_gradients = -1;
caffe.SolverParameter.prototype.snapshot = 0;
caffe.SolverParameter.prototype.snapshot_prefix = "";
caffe.SolverParameter.prototype.snapshot_diff = false;
caffe.SolverParameter.prototype.snapshot_format = 1;
caffe.SolverParameter.prototype.solver_mode = 1;
caffe.SolverParameter.prototype.device_id = 0;
caffe.SolverParameter.prototype.random_seed = -1n;
caffe.SolverParameter.prototype.type = "SGD";
caffe.SolverParameter.prototype.delta = 1e-8;
caffe.SolverParameter.prototype.momentum2 = 0.999;
caffe.SolverParameter.prototype.rms_decay = 0.99;
caffe.SolverParameter.prototype.debug_info = false;
caffe.SolverParameter.prototype.snapshot_after_train = true;
caffe.SolverParameter.prototype.solver_type = 0;
caffe.SolverParameter.prototype.layer_wise_reduce = true;

caffe.SolverParameter.SnapshotFormat = {
    "HDF5": 0,
    "BINARYPROTO": 1
};

caffe.SolverParameter.SolverMode = {
    "CPU": 0,
    "GPU": 1
};

caffe.SolverParameter.SolverType = {
    "SGD": 0,
    "NESTEROV": 1,
    "ADAGRAD": 2,
    "RMSPROP": 3,
    "ADADELTA": 4,
    "ADAM": 5
};

caffe.SolverState = class SolverState {

    constructor() {
        this.history = [];
    }

    static decode(reader, length) {
        const message = new caffe.SolverState();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.history.push(caffe.BlobProto.decode(reader, reader.uint32()));
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
        const message = new caffe.SolverState();
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
                    message.history.push(caffe.BlobProto.decodeText(reader));
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

caffe.SolverState.prototype.iter = 0;
caffe.SolverState.prototype.learned_net = "";
caffe.SolverState.prototype.current_step = 0;

caffe.Phase = {
    "TRAIN": 0,
    "TEST": 1
};

caffe.NetState = class NetState {

    constructor() {
        this.stage = [];
    }

    static decode(reader, length) {
        const message = new caffe.NetState();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.NetState();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "phase":
                    message.phase = reader.enum(caffe.Phase);
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

caffe.NetState.prototype.phase = 1;
caffe.NetState.prototype.level = 0;

caffe.NetStateRule = class NetStateRule {

    constructor() {
        this.stage = [];
        this.not_stage = [];
    }

    static decode(reader, length) {
        const message = new caffe.NetStateRule();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.NetStateRule();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "phase":
                    message.phase = reader.enum(caffe.Phase);
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

caffe.NetStateRule.prototype.phase = 0;
caffe.NetStateRule.prototype.min_level = 0;
caffe.NetStateRule.prototype.max_level = 0;

caffe.ParamSpec = class ParamSpec {

    static decode(reader, length) {
        const message = new caffe.ParamSpec();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ParamSpec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "share_mode":
                    message.share_mode = reader.enum(caffe.ParamSpec.DimCheckMode);
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

caffe.ParamSpec.prototype.name = "";
caffe.ParamSpec.prototype.share_mode = 0;
caffe.ParamSpec.prototype.lr_mult = 1;
caffe.ParamSpec.prototype.decay_mult = 1;

caffe.ParamSpec.DimCheckMode = {
    "STRICT": 0,
    "PERMISSIVE": 1
};

caffe.LayerParameter = class LayerParameter {

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
        const message = new caffe.LayerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.param.push(caffe.ParamSpec.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.blobs.push(caffe.BlobProto.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.propagate_down = reader.array(message.propagate_down, () => reader.bool(), tag);
                    break;
                case 8:
                    message.include.push(caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.exclude.push(caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.transform_param = caffe.TransformationParameter.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.loss_param = caffe.LossParameter.decode(reader, reader.uint32());
                    break;
                case 102:
                    message.accuracy_param = caffe.AccuracyParameter.decode(reader, reader.uint32());
                    break;
                case 103:
                    message.argmax_param = caffe.ArgMaxParameter.decode(reader, reader.uint32());
                    break;
                case 139:
                    message.batch_norm_param = caffe.BatchNormParameter.decode(reader, reader.uint32());
                    break;
                case 141:
                    message.bias_param = caffe.BiasParameter.decode(reader, reader.uint32());
                    break;
                case 148:
                    message.clip_param = caffe.ClipParameter.decode(reader, reader.uint32());
                    break;
                case 104:
                    message.concat_param = caffe.ConcatParameter.decode(reader, reader.uint32());
                    break;
                case 105:
                    message.contrastive_loss_param = caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                    break;
                case 106:
                    message.convolution_param = caffe.ConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 144:
                    message.crop_param = caffe.CropParameter.decode(reader, reader.uint32());
                    break;
                case 107:
                    message.data_param = caffe.DataParameter.decode(reader, reader.uint32());
                    break;
                case 108:
                    message.dropout_param = caffe.DropoutParameter.decode(reader, reader.uint32());
                    break;
                case 109:
                    message.dummy_data_param = caffe.DummyDataParameter.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.eltwise_param = caffe.EltwiseParameter.decode(reader, reader.uint32());
                    break;
                case 140:
                    message.elu_param = caffe.ELUParameter.decode(reader, reader.uint32());
                    break;
                case 137:
                    message.embed_param = caffe.EmbedParameter.decode(reader, reader.uint32());
                    break;
                case 111:
                    message.exp_param = caffe.ExpParameter.decode(reader, reader.uint32());
                    break;
                case 135:
                    message.flatten_param = caffe.FlattenParameter.decode(reader, reader.uint32());
                    break;
                case 112:
                    message.hdf5_data_param = caffe.HDF5DataParameter.decode(reader, reader.uint32());
                    break;
                case 113:
                    message.hdf5_output_param = caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                    break;
                case 114:
                    message.hinge_loss_param = caffe.HingeLossParameter.decode(reader, reader.uint32());
                    break;
                case 115:
                    message.image_data_param = caffe.ImageDataParameter.decode(reader, reader.uint32());
                    break;
                case 116:
                    message.infogain_loss_param = caffe.InfogainLossParameter.decode(reader, reader.uint32());
                    break;
                case 117:
                    message.inner_product_param = caffe.InnerProductParameter.decode(reader, reader.uint32());
                    break;
                case 143:
                    message.input_param = caffe.InputParameter.decode(reader, reader.uint32());
                    break;
                case 134:
                    message.log_param = caffe.LogParameter.decode(reader, reader.uint32());
                    break;
                case 118:
                    message.lrn_param = caffe.LRNParameter.decode(reader, reader.uint32());
                    break;
                case 119:
                    message.memory_data_param = caffe.MemoryDataParameter.decode(reader, reader.uint32());
                    break;
                case 120:
                    message.mvn_param = caffe.MVNParameter.decode(reader, reader.uint32());
                    break;
                case 145:
                    message.parameter_param = caffe.ParameterParameter.decode(reader, reader.uint32());
                    break;
                case 121:
                    message.pooling_param = caffe.PoolingParameter.decode(reader, reader.uint32());
                    break;
                case 122:
                    message.power_param = caffe.PowerParameter.decode(reader, reader.uint32());
                    break;
                case 131:
                    message.prelu_param = caffe.PReLUParameter.decode(reader, reader.uint32());
                    break;
                case 130:
                    message.python_param = caffe.PythonParameter.decode(reader, reader.uint32());
                    break;
                case 146:
                    message.recurrent_param = caffe.RecurrentParameter.decode(reader, reader.uint32());
                    break;
                case 136:
                    message.reduction_param = caffe.ReductionParameter.decode(reader, reader.uint32());
                    break;
                case 123:
                    message.relu_param = caffe.ReLUParameter.decode(reader, reader.uint32());
                    break;
                case 133:
                    message.reshape_param = caffe.ReshapeParameter.decode(reader, reader.uint32());
                    break;
                case 142:
                    message.scale_param = caffe.ScaleParameter.decode(reader, reader.uint32());
                    break;
                case 124:
                    message.sigmoid_param = caffe.SigmoidParameter.decode(reader, reader.uint32());
                    break;
                case 125:
                    message.softmax_param = caffe.SoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 132:
                    message.spp_param = caffe.SPPParameter.decode(reader, reader.uint32());
                    break;
                case 126:
                    message.slice_param = caffe.SliceParameter.decode(reader, reader.uint32());
                    break;
                case 147:
                    message.swish_param = caffe.SwishParameter.decode(reader, reader.uint32());
                    break;
                case 127:
                    message.tanh_param = caffe.TanHParameter.decode(reader, reader.uint32());
                    break;
                case 128:
                    message.threshold_param = caffe.ThresholdParameter.decode(reader, reader.uint32());
                    break;
                case 138:
                    message.tile_param = caffe.TileParameter.decode(reader, reader.uint32());
                    break;
                case 129:
                    message.window_data_param = caffe.WindowDataParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.LayerParameter();
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
                    message.phase = reader.enum(caffe.Phase);
                    break;
                case "loss_weight":
                    reader.array(message.loss_weight, () => reader.float());
                    break;
                case "param":
                    message.param.push(caffe.ParamSpec.decodeText(reader));
                    break;
                case "blobs":
                    message.blobs.push(caffe.BlobProto.decodeText(reader));
                    break;
                case "propagate_down":
                    reader.array(message.propagate_down, () => reader.bool());
                    break;
                case "include":
                    message.include.push(caffe.NetStateRule.decodeText(reader));
                    break;
                case "exclude":
                    message.exclude.push(caffe.NetStateRule.decodeText(reader));
                    break;
                case "transform_param":
                    message.transform_param = caffe.TransformationParameter.decodeText(reader);
                    break;
                case "loss_param":
                    message.loss_param = caffe.LossParameter.decodeText(reader);
                    break;
                case "accuracy_param":
                    message.accuracy_param = caffe.AccuracyParameter.decodeText(reader);
                    break;
                case "argmax_param":
                    message.argmax_param = caffe.ArgMaxParameter.decodeText(reader);
                    break;
                case "batch_norm_param":
                    message.batch_norm_param = caffe.BatchNormParameter.decodeText(reader);
                    break;
                case "bias_param":
                    message.bias_param = caffe.BiasParameter.decodeText(reader);
                    break;
                case "clip_param":
                    message.clip_param = caffe.ClipParameter.decodeText(reader);
                    break;
                case "concat_param":
                    message.concat_param = caffe.ConcatParameter.decodeText(reader);
                    break;
                case "contrastive_loss_param":
                    message.contrastive_loss_param = caffe.ContrastiveLossParameter.decodeText(reader);
                    break;
                case "convolution_param":
                    message.convolution_param = caffe.ConvolutionParameter.decodeText(reader);
                    break;
                case "crop_param":
                    message.crop_param = caffe.CropParameter.decodeText(reader);
                    break;
                case "data_param":
                    message.data_param = caffe.DataParameter.decodeText(reader);
                    break;
                case "dropout_param":
                    message.dropout_param = caffe.DropoutParameter.decodeText(reader);
                    break;
                case "dummy_data_param":
                    message.dummy_data_param = caffe.DummyDataParameter.decodeText(reader);
                    break;
                case "eltwise_param":
                    message.eltwise_param = caffe.EltwiseParameter.decodeText(reader);
                    break;
                case "elu_param":
                    message.elu_param = caffe.ELUParameter.decodeText(reader);
                    break;
                case "embed_param":
                    message.embed_param = caffe.EmbedParameter.decodeText(reader);
                    break;
                case "exp_param":
                    message.exp_param = caffe.ExpParameter.decodeText(reader);
                    break;
                case "flatten_param":
                    message.flatten_param = caffe.FlattenParameter.decodeText(reader);
                    break;
                case "hdf5_data_param":
                    message.hdf5_data_param = caffe.HDF5DataParameter.decodeText(reader);
                    break;
                case "hdf5_output_param":
                    message.hdf5_output_param = caffe.HDF5OutputParameter.decodeText(reader);
                    break;
                case "hinge_loss_param":
                    message.hinge_loss_param = caffe.HingeLossParameter.decodeText(reader);
                    break;
                case "image_data_param":
                    message.image_data_param = caffe.ImageDataParameter.decodeText(reader);
                    break;
                case "infogain_loss_param":
                    message.infogain_loss_param = caffe.InfogainLossParameter.decodeText(reader);
                    break;
                case "inner_product_param":
                    message.inner_product_param = caffe.InnerProductParameter.decodeText(reader);
                    break;
                case "input_param":
                    message.input_param = caffe.InputParameter.decodeText(reader);
                    break;
                case "log_param":
                    message.log_param = caffe.LogParameter.decodeText(reader);
                    break;
                case "lrn_param":
                    message.lrn_param = caffe.LRNParameter.decodeText(reader);
                    break;
                case "memory_data_param":
                    message.memory_data_param = caffe.MemoryDataParameter.decodeText(reader);
                    break;
                case "mvn_param":
                    message.mvn_param = caffe.MVNParameter.decodeText(reader);
                    break;
                case "parameter_param":
                    message.parameter_param = caffe.ParameterParameter.decodeText(reader);
                    break;
                case "pooling_param":
                    message.pooling_param = caffe.PoolingParameter.decodeText(reader);
                    break;
                case "power_param":
                    message.power_param = caffe.PowerParameter.decodeText(reader);
                    break;
                case "prelu_param":
                    message.prelu_param = caffe.PReLUParameter.decodeText(reader);
                    break;
                case "python_param":
                    message.python_param = caffe.PythonParameter.decodeText(reader);
                    break;
                case "recurrent_param":
                    message.recurrent_param = caffe.RecurrentParameter.decodeText(reader);
                    break;
                case "reduction_param":
                    message.reduction_param = caffe.ReductionParameter.decodeText(reader);
                    break;
                case "relu_param":
                    message.relu_param = caffe.ReLUParameter.decodeText(reader);
                    break;
                case "reshape_param":
                    message.reshape_param = caffe.ReshapeParameter.decodeText(reader);
                    break;
                case "scale_param":
                    message.scale_param = caffe.ScaleParameter.decodeText(reader);
                    break;
                case "sigmoid_param":
                    message.sigmoid_param = caffe.SigmoidParameter.decodeText(reader);
                    break;
                case "softmax_param":
                    message.softmax_param = caffe.SoftmaxParameter.decodeText(reader);
                    break;
                case "spp_param":
                    message.spp_param = caffe.SPPParameter.decodeText(reader);
                    break;
                case "slice_param":
                    message.slice_param = caffe.SliceParameter.decodeText(reader);
                    break;
                case "swish_param":
                    message.swish_param = caffe.SwishParameter.decodeText(reader);
                    break;
                case "tanh_param":
                    message.tanh_param = caffe.TanHParameter.decodeText(reader);
                    break;
                case "threshold_param":
                    message.threshold_param = caffe.ThresholdParameter.decodeText(reader);
                    break;
                case "tile_param":
                    message.tile_param = caffe.TileParameter.decodeText(reader);
                    break;
                case "window_data_param":
                    message.window_data_param = caffe.WindowDataParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.LayerParameter.prototype.name = "";
caffe.LayerParameter.prototype.type = "";
caffe.LayerParameter.prototype.phase = 0;
caffe.LayerParameter.prototype.transform_param = null;
caffe.LayerParameter.prototype.loss_param = null;
caffe.LayerParameter.prototype.accuracy_param = null;
caffe.LayerParameter.prototype.argmax_param = null;
caffe.LayerParameter.prototype.batch_norm_param = null;
caffe.LayerParameter.prototype.bias_param = null;
caffe.LayerParameter.prototype.clip_param = null;
caffe.LayerParameter.prototype.concat_param = null;
caffe.LayerParameter.prototype.contrastive_loss_param = null;
caffe.LayerParameter.prototype.convolution_param = null;
caffe.LayerParameter.prototype.crop_param = null;
caffe.LayerParameter.prototype.data_param = null;
caffe.LayerParameter.prototype.dropout_param = null;
caffe.LayerParameter.prototype.dummy_data_param = null;
caffe.LayerParameter.prototype.eltwise_param = null;
caffe.LayerParameter.prototype.elu_param = null;
caffe.LayerParameter.prototype.embed_param = null;
caffe.LayerParameter.prototype.exp_param = null;
caffe.LayerParameter.prototype.flatten_param = null;
caffe.LayerParameter.prototype.hdf5_data_param = null;
caffe.LayerParameter.prototype.hdf5_output_param = null;
caffe.LayerParameter.prototype.hinge_loss_param = null;
caffe.LayerParameter.prototype.image_data_param = null;
caffe.LayerParameter.prototype.infogain_loss_param = null;
caffe.LayerParameter.prototype.inner_product_param = null;
caffe.LayerParameter.prototype.input_param = null;
caffe.LayerParameter.prototype.log_param = null;
caffe.LayerParameter.prototype.lrn_param = null;
caffe.LayerParameter.prototype.memory_data_param = null;
caffe.LayerParameter.prototype.mvn_param = null;
caffe.LayerParameter.prototype.parameter_param = null;
caffe.LayerParameter.prototype.pooling_param = null;
caffe.LayerParameter.prototype.power_param = null;
caffe.LayerParameter.prototype.prelu_param = null;
caffe.LayerParameter.prototype.python_param = null;
caffe.LayerParameter.prototype.recurrent_param = null;
caffe.LayerParameter.prototype.reduction_param = null;
caffe.LayerParameter.prototype.relu_param = null;
caffe.LayerParameter.prototype.reshape_param = null;
caffe.LayerParameter.prototype.scale_param = null;
caffe.LayerParameter.prototype.sigmoid_param = null;
caffe.LayerParameter.prototype.softmax_param = null;
caffe.LayerParameter.prototype.spp_param = null;
caffe.LayerParameter.prototype.slice_param = null;
caffe.LayerParameter.prototype.swish_param = null;
caffe.LayerParameter.prototype.tanh_param = null;
caffe.LayerParameter.prototype.threshold_param = null;
caffe.LayerParameter.prototype.tile_param = null;
caffe.LayerParameter.prototype.window_data_param = null;

caffe.TransformationParameter = class TransformationParameter {

    constructor() {
        this.mean_value = [];
    }

    static decode(reader, length) {
        const message = new caffe.TransformationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.TransformationParameter();
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

caffe.TransformationParameter.prototype.scale = 1;
caffe.TransformationParameter.prototype.mirror = false;
caffe.TransformationParameter.prototype.crop_size = 0;
caffe.TransformationParameter.prototype.mean_file = "";
caffe.TransformationParameter.prototype.force_color = false;
caffe.TransformationParameter.prototype.force_gray = false;

caffe.LossParameter = class LossParameter {

    static decode(reader, length) {
        const message = new caffe.LossParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.LossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ignore_label":
                    message.ignore_label = reader.int32();
                    break;
                case "normalization":
                    message.normalization = reader.enum(caffe.LossParameter.NormalizationMode);
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

caffe.LossParameter.prototype.ignore_label = 0;
caffe.LossParameter.prototype.normalization = 1;
caffe.LossParameter.prototype.normalize = false;

caffe.LossParameter.NormalizationMode = {
    "FULL": 0,
    "VALID": 1,
    "BATCH_SIZE": 2,
    "NONE": 3
};

caffe.AccuracyParameter = class AccuracyParameter {

    static decode(reader, length) {
        const message = new caffe.AccuracyParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.AccuracyParameter();
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

caffe.AccuracyParameter.prototype.top_k = 1;
caffe.AccuracyParameter.prototype.axis = 1;
caffe.AccuracyParameter.prototype.ignore_label = 0;

caffe.ArgMaxParameter = class ArgMaxParameter {

    static decode(reader, length) {
        const message = new caffe.ArgMaxParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ArgMaxParameter();
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

caffe.ArgMaxParameter.prototype.out_max_val = false;
caffe.ArgMaxParameter.prototype.top_k = 1;
caffe.ArgMaxParameter.prototype.axis = 0;

caffe.ClipParameter = class ClipParameter {

    static decode(reader, length) {
        const message = new caffe.ClipParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ClipParameter();
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

caffe.ClipParameter.prototype.min = 0;
caffe.ClipParameter.prototype.max = 0;

caffe.ConcatParameter = class ConcatParameter {

    static decode(reader, length) {
        const message = new caffe.ConcatParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ConcatParameter();
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

caffe.ConcatParameter.prototype.axis = 1;
caffe.ConcatParameter.prototype.concat_dim = 1;

caffe.BatchNormParameter = class BatchNormParameter {

    static decode(reader, length) {
        const message = new caffe.BatchNormParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.BatchNormParameter();
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

caffe.BatchNormParameter.prototype.use_global_stats = false;
caffe.BatchNormParameter.prototype.moving_average_fraction = 0.999;
caffe.BatchNormParameter.prototype.eps = 0.00001;

caffe.BiasParameter = class BiasParameter {

    static decode(reader, length) {
        const message = new caffe.BiasParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.BiasParameter();
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
                    message.filler = caffe.FillerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.BiasParameter.prototype.axis = 1;
caffe.BiasParameter.prototype.num_axes = 1;
caffe.BiasParameter.prototype.filler = null;

caffe.ContrastiveLossParameter = class ContrastiveLossParameter {

    static decode(reader, length) {
        const message = new caffe.ContrastiveLossParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ContrastiveLossParameter();
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

caffe.ContrastiveLossParameter.prototype.margin = 1;
caffe.ContrastiveLossParameter.prototype.legacy_version = false;

caffe.ConvolutionParameter = class ConvolutionParameter {

    constructor() {
        this.pad = [];
        this.kernel_size = [];
        this.stride = [];
        this.dilation = [];
    }

    static decode(reader, length) {
        const message = new caffe.ConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.weight_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.bias_filler = caffe.FillerParameter.decode(reader, reader.uint32());
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
        const message = new caffe.ConvolutionParameter();
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
                    message.weight_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "engine":
                    message.engine = reader.enum(caffe.ConvolutionParameter.Engine);
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

caffe.ConvolutionParameter.prototype.num_output = 0;
caffe.ConvolutionParameter.prototype.bias_term = true;
caffe.ConvolutionParameter.prototype.pad_h = 0;
caffe.ConvolutionParameter.prototype.pad_w = 0;
caffe.ConvolutionParameter.prototype.kernel_h = 0;
caffe.ConvolutionParameter.prototype.kernel_w = 0;
caffe.ConvolutionParameter.prototype.stride_h = 0;
caffe.ConvolutionParameter.prototype.stride_w = 0;
caffe.ConvolutionParameter.prototype.group = 1;
caffe.ConvolutionParameter.prototype.weight_filler = null;
caffe.ConvolutionParameter.prototype.bias_filler = null;
caffe.ConvolutionParameter.prototype.engine = 0;
caffe.ConvolutionParameter.prototype.axis = 1;
caffe.ConvolutionParameter.prototype.force_nd_im2col = false;

caffe.ConvolutionParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.CropParameter = class CropParameter {

    constructor() {
        this.offset = [];
    }

    static decode(reader, length) {
        const message = new caffe.CropParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.CropParameter();
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

caffe.CropParameter.prototype.axis = 2;

caffe.DataParameter = class DataParameter {

    static decode(reader, length) {
        const message = new caffe.DataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.DataParameter();
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
                    message.backend = reader.enum(caffe.DataParameter.DB);
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

caffe.DataParameter.prototype.source = "";
caffe.DataParameter.prototype.batch_size = 0;
caffe.DataParameter.prototype.rand_skip = 0;
caffe.DataParameter.prototype.backend = 0;
caffe.DataParameter.prototype.scale = 1;
caffe.DataParameter.prototype.mean_file = "";
caffe.DataParameter.prototype.crop_size = 0;
caffe.DataParameter.prototype.mirror = false;
caffe.DataParameter.prototype.force_encoded_color = false;
caffe.DataParameter.prototype.prefetch = 4;

caffe.DataParameter.DB = {
    "LEVELDB": 0,
    "LMDB": 1
};

caffe.DropoutParameter = class DropoutParameter {

    static decode(reader, length) {
        const message = new caffe.DropoutParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.DropoutParameter();
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

caffe.DropoutParameter.prototype.dropout_ratio = 0.5;

caffe.DummyDataParameter = class DummyDataParameter {

    constructor() {
        this.data_filler = [];
        this.shape = [];
        this.num = [];
        this.channels = [];
        this.height = [];
        this.width = [];
    }

    static decode(reader, length) {
        const message = new caffe.DummyDataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.data_filler.push(caffe.FillerParameter.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.shape.push(caffe.BlobShape.decode(reader, reader.uint32()));
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
        const message = new caffe.DummyDataParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "data_filler":
                    message.data_filler.push(caffe.FillerParameter.decodeText(reader));
                    break;
                case "shape":
                    message.shape.push(caffe.BlobShape.decodeText(reader));
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

caffe.EltwiseParameter = class EltwiseParameter {

    constructor() {
        this.coeff = [];
    }

    static decode(reader, length) {
        const message = new caffe.EltwiseParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.EltwiseParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "operation":
                    message.operation = reader.enum(caffe.EltwiseParameter.EltwiseOp);
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

caffe.EltwiseParameter.prototype.operation = 1;
caffe.EltwiseParameter.prototype.stable_prod_grad = true;

caffe.EltwiseParameter.EltwiseOp = {
    "PROD": 0,
    "SUM": 1,
    "MAX": 2
};

caffe.ELUParameter = class ELUParameter {

    static decode(reader, length) {
        const message = new caffe.ELUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ELUParameter();
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

caffe.ELUParameter.prototype.alpha = 1;

caffe.EmbedParameter = class EmbedParameter {

    static decode(reader, length) {
        const message = new caffe.EmbedParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.weight_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.bias_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.EmbedParameter();
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
                    message.weight_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.EmbedParameter.prototype.num_output = 0;
caffe.EmbedParameter.prototype.input_dim = 0;
caffe.EmbedParameter.prototype.bias_term = true;
caffe.EmbedParameter.prototype.weight_filler = null;
caffe.EmbedParameter.prototype.bias_filler = null;

caffe.ExpParameter = class ExpParameter {

    static decode(reader, length) {
        const message = new caffe.ExpParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ExpParameter();
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

caffe.ExpParameter.prototype.base = -1;
caffe.ExpParameter.prototype.scale = 1;
caffe.ExpParameter.prototype.shift = 0;

caffe.FlattenParameter = class FlattenParameter {

    static decode(reader, length) {
        const message = new caffe.FlattenParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.FlattenParameter();
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

caffe.FlattenParameter.prototype.axis = 1;
caffe.FlattenParameter.prototype.end_axis = -1;

caffe.HDF5DataParameter = class HDF5DataParameter {

    static decode(reader, length) {
        const message = new caffe.HDF5DataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.HDF5DataParameter();
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

caffe.HDF5DataParameter.prototype.source = "";
caffe.HDF5DataParameter.prototype.batch_size = 0;
caffe.HDF5DataParameter.prototype.shuffle = false;

caffe.HDF5OutputParameter = class HDF5OutputParameter {

    static decode(reader, length) {
        const message = new caffe.HDF5OutputParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.HDF5OutputParameter();
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

caffe.HDF5OutputParameter.prototype.file_name = "";

caffe.HingeLossParameter = class HingeLossParameter {

    static decode(reader, length) {
        const message = new caffe.HingeLossParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.HingeLossParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "norm":
                    message.norm = reader.enum(caffe.HingeLossParameter.Norm);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.HingeLossParameter.prototype.norm = 1;

caffe.HingeLossParameter.Norm = {
    "L1": 1,
    "L2": 2
};

caffe.ImageDataParameter = class ImageDataParameter {

    static decode(reader, length) {
        const message = new caffe.ImageDataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ImageDataParameter();
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

caffe.ImageDataParameter.prototype.source = "";
caffe.ImageDataParameter.prototype.batch_size = 1;
caffe.ImageDataParameter.prototype.rand_skip = 0;
caffe.ImageDataParameter.prototype.shuffle = false;
caffe.ImageDataParameter.prototype.new_height = 0;
caffe.ImageDataParameter.prototype.new_width = 0;
caffe.ImageDataParameter.prototype.is_color = true;
caffe.ImageDataParameter.prototype.scale = 1;
caffe.ImageDataParameter.prototype.mean_file = "";
caffe.ImageDataParameter.prototype.crop_size = 0;
caffe.ImageDataParameter.prototype.mirror = false;
caffe.ImageDataParameter.prototype.root_folder = "";

caffe.InfogainLossParameter = class InfogainLossParameter {

    static decode(reader, length) {
        const message = new caffe.InfogainLossParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.InfogainLossParameter();
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

caffe.InfogainLossParameter.prototype.source = "";
caffe.InfogainLossParameter.prototype.axis = 1;

caffe.InnerProductParameter = class InnerProductParameter {

    static decode(reader, length) {
        const message = new caffe.InnerProductParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.weight_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bias_filler = caffe.FillerParameter.decode(reader, reader.uint32());
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
        const message = new caffe.InnerProductParameter();
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
                    message.weight_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = caffe.FillerParameter.decodeText(reader);
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

caffe.InnerProductParameter.prototype.num_output = 0;
caffe.InnerProductParameter.prototype.bias_term = true;
caffe.InnerProductParameter.prototype.weight_filler = null;
caffe.InnerProductParameter.prototype.bias_filler = null;
caffe.InnerProductParameter.prototype.axis = 1;
caffe.InnerProductParameter.prototype.transpose = false;

caffe.InputParameter = class InputParameter {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new caffe.InputParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape.push(caffe.BlobShape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.InputParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape.push(caffe.BlobShape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.LogParameter = class LogParameter {

    static decode(reader, length) {
        const message = new caffe.LogParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.LogParameter();
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

caffe.LogParameter.prototype.base = -1;
caffe.LogParameter.prototype.scale = 1;
caffe.LogParameter.prototype.shift = 0;

caffe.LRNParameter = class LRNParameter {

    static decode(reader, length) {
        const message = new caffe.LRNParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.LRNParameter();
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
                    message.norm_region = reader.enum(caffe.LRNParameter.NormRegion);
                    break;
                case "k":
                    message.k = reader.float();
                    break;
                case "engine":
                    message.engine = reader.enum(caffe.LRNParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.LRNParameter.prototype.local_size = 5;
caffe.LRNParameter.prototype.alpha = 1;
caffe.LRNParameter.prototype.beta = 0.75;
caffe.LRNParameter.prototype.norm_region = 0;
caffe.LRNParameter.prototype.k = 1;
caffe.LRNParameter.prototype.engine = 0;

caffe.LRNParameter.NormRegion = {
    "ACROSS_CHANNELS": 0,
    "WITHIN_CHANNEL": 1
};

caffe.LRNParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.MemoryDataParameter = class MemoryDataParameter {

    static decode(reader, length) {
        const message = new caffe.MemoryDataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.MemoryDataParameter();
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

caffe.MemoryDataParameter.prototype.batch_size = 0;
caffe.MemoryDataParameter.prototype.channels = 0;
caffe.MemoryDataParameter.prototype.height = 0;
caffe.MemoryDataParameter.prototype.width = 0;

caffe.MVNParameter = class MVNParameter {

    static decode(reader, length) {
        const message = new caffe.MVNParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.MVNParameter();
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

caffe.MVNParameter.prototype.normalize_variance = true;
caffe.MVNParameter.prototype.across_channels = false;
caffe.MVNParameter.prototype.eps = 1e-9;

caffe.ParameterParameter = class ParameterParameter {

    static decode(reader, length) {
        const message = new caffe.ParameterParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = caffe.BlobShape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.ParameterParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = caffe.BlobShape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.ParameterParameter.prototype.shape = null;

caffe.PoolingParameter = class PoolingParameter {

    static decode(reader, length) {
        const message = new caffe.PoolingParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.PoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pool":
                    message.pool = reader.enum(caffe.PoolingParameter.PoolMethod);
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
                    message.engine = reader.enum(caffe.PoolingParameter.Engine);
                    break;
                case "global_pooling":
                    message.global_pooling = reader.bool();
                    break;
                case "round_mode":
                    message.round_mode = reader.enum(caffe.PoolingParameter.RoundMode);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.PoolingParameter.prototype.pool = 0;
caffe.PoolingParameter.prototype.pad = 0;
caffe.PoolingParameter.prototype.pad_h = 0;
caffe.PoolingParameter.prototype.pad_w = 0;
caffe.PoolingParameter.prototype.kernel_size = 0;
caffe.PoolingParameter.prototype.kernel_h = 0;
caffe.PoolingParameter.prototype.kernel_w = 0;
caffe.PoolingParameter.prototype.stride = 1;
caffe.PoolingParameter.prototype.stride_h = 0;
caffe.PoolingParameter.prototype.stride_w = 0;
caffe.PoolingParameter.prototype.engine = 0;
caffe.PoolingParameter.prototype.global_pooling = false;
caffe.PoolingParameter.prototype.round_mode = 0;

caffe.PoolingParameter.PoolMethod = {
    "MAX": 0,
    "AVE": 1,
    "STOCHASTIC": 2
};

caffe.PoolingParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.PoolingParameter.RoundMode = {
    "CEIL": 0,
    "FLOOR": 1
};

caffe.PowerParameter = class PowerParameter {

    static decode(reader, length) {
        const message = new caffe.PowerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.PowerParameter();
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

caffe.PowerParameter.prototype.power = 1;
caffe.PowerParameter.prototype.scale = 1;
caffe.PowerParameter.prototype.shift = 0;

caffe.PythonParameter = class PythonParameter {

    static decode(reader, length) {
        const message = new caffe.PythonParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.PythonParameter();
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

caffe.PythonParameter.prototype.module = "";
caffe.PythonParameter.prototype.layer = "";
caffe.PythonParameter.prototype.param_str = "";
caffe.PythonParameter.prototype.share_in_parallel = false;

caffe.RecurrentParameter = class RecurrentParameter {

    static decode(reader, length) {
        const message = new caffe.RecurrentParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_output = reader.uint32();
                    break;
                case 2:
                    message.weight_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.bias_filler = caffe.FillerParameter.decode(reader, reader.uint32());
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
        const message = new caffe.RecurrentParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_output":
                    message.num_output = reader.uint32();
                    break;
                case "weight_filler":
                    message.weight_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = caffe.FillerParameter.decodeText(reader);
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

caffe.RecurrentParameter.prototype.num_output = 0;
caffe.RecurrentParameter.prototype.weight_filler = null;
caffe.RecurrentParameter.prototype.bias_filler = null;
caffe.RecurrentParameter.prototype.debug_info = false;
caffe.RecurrentParameter.prototype.expose_hidden = false;

caffe.ReductionParameter = class ReductionParameter {

    static decode(reader, length) {
        const message = new caffe.ReductionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ReductionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "operation":
                    message.operation = reader.enum(caffe.ReductionParameter.ReductionOp);
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

caffe.ReductionParameter.prototype.operation = 1;
caffe.ReductionParameter.prototype.axis = 0;
caffe.ReductionParameter.prototype.coeff = 1;

caffe.ReductionParameter.ReductionOp = {
    "SUM": 1,
    "ASUM": 2,
    "SUMSQ": 3,
    "MEAN": 4
};

caffe.ReLUParameter = class ReLUParameter {

    static decode(reader, length) {
        const message = new caffe.ReLUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "negative_slope":
                    message.negative_slope = reader.float();
                    break;
                case "engine":
                    message.engine = reader.enum(caffe.ReLUParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.ReLUParameter.prototype.negative_slope = 0;
caffe.ReLUParameter.prototype.engine = 0;

caffe.ReLUParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.ReshapeParameter = class ReshapeParameter {

    static decode(reader, length) {
        const message = new caffe.ReshapeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = caffe.BlobShape.decode(reader, reader.uint32());
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
        const message = new caffe.ReshapeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = caffe.BlobShape.decodeText(reader);
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

caffe.ReshapeParameter.prototype.shape = null;
caffe.ReshapeParameter.prototype.axis = 0;
caffe.ReshapeParameter.prototype.num_axes = -1;

caffe.ScaleParameter = class ScaleParameter {

    static decode(reader, length) {
        const message = new caffe.ScaleParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.bias_term = reader.bool();
                    break;
                case 5:
                    message.bias_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.ScaleParameter();
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
                    message.filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_term":
                    message.bias_term = reader.bool();
                    break;
                case "bias_filler":
                    message.bias_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.ScaleParameter.prototype.axis = 1;
caffe.ScaleParameter.prototype.num_axes = 1;
caffe.ScaleParameter.prototype.filler = null;
caffe.ScaleParameter.prototype.bias_term = false;
caffe.ScaleParameter.prototype.bias_filler = null;

caffe.SigmoidParameter = class SigmoidParameter {

    static decode(reader, length) {
        const message = new caffe.SigmoidParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.SigmoidParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "engine":
                    message.engine = reader.enum(caffe.SigmoidParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.SigmoidParameter.prototype.engine = 0;

caffe.SigmoidParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.SliceParameter = class SliceParameter {

    constructor() {
        this.slice_point = [];
    }

    static decode(reader, length) {
        const message = new caffe.SliceParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.SliceParameter();
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

caffe.SliceParameter.prototype.axis = 1;
caffe.SliceParameter.prototype.slice_dim = 1;

caffe.SoftmaxParameter = class SoftmaxParameter {

    static decode(reader, length) {
        const message = new caffe.SoftmaxParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.SoftmaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "engine":
                    message.engine = reader.enum(caffe.SoftmaxParameter.Engine);
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

caffe.SoftmaxParameter.prototype.engine = 0;
caffe.SoftmaxParameter.prototype.axis = 1;

caffe.SoftmaxParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.SwishParameter = class SwishParameter {

    static decode(reader, length) {
        const message = new caffe.SwishParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.SwishParameter();
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

caffe.SwishParameter.prototype.beta = 1;

caffe.TanHParameter = class TanHParameter {

    static decode(reader, length) {
        const message = new caffe.TanHParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.TanHParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "engine":
                    message.engine = reader.enum(caffe.TanHParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.TanHParameter.prototype.engine = 0;

caffe.TanHParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.TileParameter = class TileParameter {

    static decode(reader, length) {
        const message = new caffe.TileParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.TileParameter();
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

caffe.TileParameter.prototype.axis = 1;
caffe.TileParameter.prototype.tiles = 0;

caffe.ThresholdParameter = class ThresholdParameter {

    static decode(reader, length) {
        const message = new caffe.ThresholdParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.ThresholdParameter();
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

caffe.ThresholdParameter.prototype.threshold = 0;

caffe.WindowDataParameter = class WindowDataParameter {

    static decode(reader, length) {
        const message = new caffe.WindowDataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.WindowDataParameter();
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

caffe.WindowDataParameter.prototype.source = "";
caffe.WindowDataParameter.prototype.scale = 1;
caffe.WindowDataParameter.prototype.mean_file = "";
caffe.WindowDataParameter.prototype.batch_size = 0;
caffe.WindowDataParameter.prototype.crop_size = 0;
caffe.WindowDataParameter.prototype.mirror = false;
caffe.WindowDataParameter.prototype.fg_threshold = 0.5;
caffe.WindowDataParameter.prototype.bg_threshold = 0.5;
caffe.WindowDataParameter.prototype.fg_fraction = 0.25;
caffe.WindowDataParameter.prototype.context_pad = 0;
caffe.WindowDataParameter.prototype.crop_mode = "warp";
caffe.WindowDataParameter.prototype.cache_images = false;
caffe.WindowDataParameter.prototype.root_folder = "";

caffe.SPPParameter = class SPPParameter {

    static decode(reader, length) {
        const message = new caffe.SPPParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new caffe.SPPParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pyramid_height":
                    message.pyramid_height = reader.uint32();
                    break;
                case "pool":
                    message.pool = reader.enum(caffe.SPPParameter.PoolMethod);
                    break;
                case "engine":
                    message.engine = reader.enum(caffe.SPPParameter.Engine);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.SPPParameter.prototype.pyramid_height = 0;
caffe.SPPParameter.prototype.pool = 0;
caffe.SPPParameter.prototype.engine = 0;

caffe.SPPParameter.PoolMethod = {
    "MAX": 0,
    "AVE": 1,
    "STOCHASTIC": 2
};

caffe.SPPParameter.Engine = {
    "DEFAULT": 0,
    "CAFFE": 1,
    "CUDNN": 2
};

caffe.V1LayerParameter = class V1LayerParameter {

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
        const message = new caffe.V1LayerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.include.push(caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 33:
                    message.exclude.push(caffe.NetStateRule.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.type = reader.int32();
                    break;
                case 6:
                    message.blobs.push(caffe.BlobProto.decode(reader, reader.uint32()));
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
                    message.accuracy_param = caffe.AccuracyParameter.decode(reader, reader.uint32());
                    break;
                case 23:
                    message.argmax_param = caffe.ArgMaxParameter.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.concat_param = caffe.ConcatParameter.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.contrastive_loss_param = caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.convolution_param = caffe.ConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.data_param = caffe.DataParameter.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.dropout_param = caffe.DropoutParameter.decode(reader, reader.uint32());
                    break;
                case 26:
                    message.dummy_data_param = caffe.DummyDataParameter.decode(reader, reader.uint32());
                    break;
                case 24:
                    message.eltwise_param = caffe.EltwiseParameter.decode(reader, reader.uint32());
                    break;
                case 41:
                    message.exp_param = caffe.ExpParameter.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.hdf5_data_param = caffe.HDF5DataParameter.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.hdf5_output_param = caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                    break;
                case 29:
                    message.hinge_loss_param = caffe.HingeLossParameter.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.image_data_param = caffe.ImageDataParameter.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.infogain_loss_param = caffe.InfogainLossParameter.decode(reader, reader.uint32());
                    break;
                case 17:
                    message.inner_product_param = caffe.InnerProductParameter.decode(reader, reader.uint32());
                    break;
                case 18:
                    message.lrn_param = caffe.LRNParameter.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.memory_data_param = caffe.MemoryDataParameter.decode(reader, reader.uint32());
                    break;
                case 34:
                    message.mvn_param = caffe.MVNParameter.decode(reader, reader.uint32());
                    break;
                case 19:
                    message.pooling_param = caffe.PoolingParameter.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.power_param = caffe.PowerParameter.decode(reader, reader.uint32());
                    break;
                case 30:
                    message.relu_param = caffe.ReLUParameter.decode(reader, reader.uint32());
                    break;
                case 38:
                    message.sigmoid_param = caffe.SigmoidParameter.decode(reader, reader.uint32());
                    break;
                case 39:
                    message.softmax_param = caffe.SoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.slice_param = caffe.SliceParameter.decode(reader, reader.uint32());
                    break;
                case 37:
                    message.tanh_param = caffe.TanHParameter.decode(reader, reader.uint32());
                    break;
                case 25:
                    message.threshold_param = caffe.ThresholdParameter.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.window_data_param = caffe.WindowDataParameter.decode(reader, reader.uint32());
                    break;
                case 36:
                    message.transform_param = caffe.TransformationParameter.decode(reader, reader.uint32());
                    break;
                case 42:
                    message.loss_param = caffe.LossParameter.decode(reader, reader.uint32());
                    break;
                case 1:
                    message.layer = caffe.V0LayerParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.V1LayerParameter();
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
                    message.include.push(caffe.NetStateRule.decodeText(reader));
                    break;
                case "exclude":
                    message.exclude.push(caffe.NetStateRule.decodeText(reader));
                    break;
                case "type":
                    message.type = reader.enum(caffe.V1LayerParameter.LayerType);
                    break;
                case "blobs":
                    message.blobs.push(caffe.BlobProto.decodeText(reader));
                    break;
                case "param":
                    reader.array(message.param, () => reader.string());
                    break;
                case "blob_share_mode":
                    reader.array(message.blob_share_mode, () => reader.enum(caffe.V1LayerParameter.DimCheckMode));
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
                    message.accuracy_param = caffe.AccuracyParameter.decodeText(reader);
                    break;
                case "argmax_param":
                    message.argmax_param = caffe.ArgMaxParameter.decodeText(reader);
                    break;
                case "concat_param":
                    message.concat_param = caffe.ConcatParameter.decodeText(reader);
                    break;
                case "contrastive_loss_param":
                    message.contrastive_loss_param = caffe.ContrastiveLossParameter.decodeText(reader);
                    break;
                case "convolution_param":
                    message.convolution_param = caffe.ConvolutionParameter.decodeText(reader);
                    break;
                case "data_param":
                    message.data_param = caffe.DataParameter.decodeText(reader);
                    break;
                case "dropout_param":
                    message.dropout_param = caffe.DropoutParameter.decodeText(reader);
                    break;
                case "dummy_data_param":
                    message.dummy_data_param = caffe.DummyDataParameter.decodeText(reader);
                    break;
                case "eltwise_param":
                    message.eltwise_param = caffe.EltwiseParameter.decodeText(reader);
                    break;
                case "exp_param":
                    message.exp_param = caffe.ExpParameter.decodeText(reader);
                    break;
                case "hdf5_data_param":
                    message.hdf5_data_param = caffe.HDF5DataParameter.decodeText(reader);
                    break;
                case "hdf5_output_param":
                    message.hdf5_output_param = caffe.HDF5OutputParameter.decodeText(reader);
                    break;
                case "hinge_loss_param":
                    message.hinge_loss_param = caffe.HingeLossParameter.decodeText(reader);
                    break;
                case "image_data_param":
                    message.image_data_param = caffe.ImageDataParameter.decodeText(reader);
                    break;
                case "infogain_loss_param":
                    message.infogain_loss_param = caffe.InfogainLossParameter.decodeText(reader);
                    break;
                case "inner_product_param":
                    message.inner_product_param = caffe.InnerProductParameter.decodeText(reader);
                    break;
                case "lrn_param":
                    message.lrn_param = caffe.LRNParameter.decodeText(reader);
                    break;
                case "memory_data_param":
                    message.memory_data_param = caffe.MemoryDataParameter.decodeText(reader);
                    break;
                case "mvn_param":
                    message.mvn_param = caffe.MVNParameter.decodeText(reader);
                    break;
                case "pooling_param":
                    message.pooling_param = caffe.PoolingParameter.decodeText(reader);
                    break;
                case "power_param":
                    message.power_param = caffe.PowerParameter.decodeText(reader);
                    break;
                case "relu_param":
                    message.relu_param = caffe.ReLUParameter.decodeText(reader);
                    break;
                case "sigmoid_param":
                    message.sigmoid_param = caffe.SigmoidParameter.decodeText(reader);
                    break;
                case "softmax_param":
                    message.softmax_param = caffe.SoftmaxParameter.decodeText(reader);
                    break;
                case "slice_param":
                    message.slice_param = caffe.SliceParameter.decodeText(reader);
                    break;
                case "tanh_param":
                    message.tanh_param = caffe.TanHParameter.decodeText(reader);
                    break;
                case "threshold_param":
                    message.threshold_param = caffe.ThresholdParameter.decodeText(reader);
                    break;
                case "window_data_param":
                    message.window_data_param = caffe.WindowDataParameter.decodeText(reader);
                    break;
                case "transform_param":
                    message.transform_param = caffe.TransformationParameter.decodeText(reader);
                    break;
                case "loss_param":
                    message.loss_param = caffe.LossParameter.decodeText(reader);
                    break;
                case "layer":
                    message.layer = caffe.V0LayerParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.V1LayerParameter.prototype.name = "";
caffe.V1LayerParameter.prototype.type = 0;
caffe.V1LayerParameter.prototype.accuracy_param = null;
caffe.V1LayerParameter.prototype.argmax_param = null;
caffe.V1LayerParameter.prototype.concat_param = null;
caffe.V1LayerParameter.prototype.contrastive_loss_param = null;
caffe.V1LayerParameter.prototype.convolution_param = null;
caffe.V1LayerParameter.prototype.data_param = null;
caffe.V1LayerParameter.prototype.dropout_param = null;
caffe.V1LayerParameter.prototype.dummy_data_param = null;
caffe.V1LayerParameter.prototype.eltwise_param = null;
caffe.V1LayerParameter.prototype.exp_param = null;
caffe.V1LayerParameter.prototype.hdf5_data_param = null;
caffe.V1LayerParameter.prototype.hdf5_output_param = null;
caffe.V1LayerParameter.prototype.hinge_loss_param = null;
caffe.V1LayerParameter.prototype.image_data_param = null;
caffe.V1LayerParameter.prototype.infogain_loss_param = null;
caffe.V1LayerParameter.prototype.inner_product_param = null;
caffe.V1LayerParameter.prototype.lrn_param = null;
caffe.V1LayerParameter.prototype.memory_data_param = null;
caffe.V1LayerParameter.prototype.mvn_param = null;
caffe.V1LayerParameter.prototype.pooling_param = null;
caffe.V1LayerParameter.prototype.power_param = null;
caffe.V1LayerParameter.prototype.relu_param = null;
caffe.V1LayerParameter.prototype.sigmoid_param = null;
caffe.V1LayerParameter.prototype.softmax_param = null;
caffe.V1LayerParameter.prototype.slice_param = null;
caffe.V1LayerParameter.prototype.tanh_param = null;
caffe.V1LayerParameter.prototype.threshold_param = null;
caffe.V1LayerParameter.prototype.window_data_param = null;
caffe.V1LayerParameter.prototype.transform_param = null;
caffe.V1LayerParameter.prototype.loss_param = null;
caffe.V1LayerParameter.prototype.layer = null;

caffe.V1LayerParameter.LayerType = {
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

caffe.V1LayerParameter.DimCheckMode = {
    "STRICT": 0,
    "PERMISSIVE": 1
};

caffe.V0LayerParameter = class V0LayerParameter {

    constructor() {
        this.blobs = [];
        this.blobs_lr = [];
        this.weight_decay = [];
    }

    static decode(reader, length) {
        const message = new caffe.V0LayerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.weight_filler = caffe.FillerParameter.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.bias_filler = caffe.FillerParameter.decode(reader, reader.uint32());
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
                    message.blobs.push(caffe.BlobProto.decode(reader, reader.uint32()));
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
                    message.hdf5_output_param = caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe.V0LayerParameter();
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
                    message.weight_filler = caffe.FillerParameter.decodeText(reader);
                    break;
                case "bias_filler":
                    message.bias_filler = caffe.FillerParameter.decodeText(reader);
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
                    message.pool = reader.enum(caffe.V0LayerParameter.PoolMethod);
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
                    message.blobs.push(caffe.BlobProto.decodeText(reader));
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
                    message.hdf5_output_param = caffe.HDF5OutputParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe.V0LayerParameter.prototype.name = "";
caffe.V0LayerParameter.prototype.type = "";
caffe.V0LayerParameter.prototype.num_output = 0;
caffe.V0LayerParameter.prototype.biasterm = true;
caffe.V0LayerParameter.prototype.weight_filler = null;
caffe.V0LayerParameter.prototype.bias_filler = null;
caffe.V0LayerParameter.prototype.pad = 0;
caffe.V0LayerParameter.prototype.kernelsize = 0;
caffe.V0LayerParameter.prototype.group = 1;
caffe.V0LayerParameter.prototype.stride = 1;
caffe.V0LayerParameter.prototype.pool = 0;
caffe.V0LayerParameter.prototype.dropout_ratio = 0.5;
caffe.V0LayerParameter.prototype.local_size = 5;
caffe.V0LayerParameter.prototype.alpha = 1;
caffe.V0LayerParameter.prototype.beta = 0.75;
caffe.V0LayerParameter.prototype.k = 1;
caffe.V0LayerParameter.prototype.source = "";
caffe.V0LayerParameter.prototype.scale = 1;
caffe.V0LayerParameter.prototype.meanfile = "";
caffe.V0LayerParameter.prototype.batchsize = 0;
caffe.V0LayerParameter.prototype.cropsize = 0;
caffe.V0LayerParameter.prototype.mirror = false;
caffe.V0LayerParameter.prototype.rand_skip = 0;
caffe.V0LayerParameter.prototype.det_fg_threshold = 0.5;
caffe.V0LayerParameter.prototype.det_bg_threshold = 0.5;
caffe.V0LayerParameter.prototype.det_fg_fraction = 0.25;
caffe.V0LayerParameter.prototype.det_context_pad = 0;
caffe.V0LayerParameter.prototype.det_crop_mode = "warp";
caffe.V0LayerParameter.prototype.new_num = 0;
caffe.V0LayerParameter.prototype.new_channels = 0;
caffe.V0LayerParameter.prototype.new_height = 0;
caffe.V0LayerParameter.prototype.new_width = 0;
caffe.V0LayerParameter.prototype.shuffle_images = false;
caffe.V0LayerParameter.prototype.concat_dim = 1;
caffe.V0LayerParameter.prototype.hdf5_output_param = null;

caffe.V0LayerParameter.PoolMethod = {
    "MAX": 0,
    "AVE": 1,
    "STOCHASTIC": 2
};

caffe.PReLUParameter = class PReLUParameter {

    static decode(reader, length) {
        const message = new caffe.PReLUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.filler = caffe.FillerParameter.decode(reader, reader.uint32());
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
        const message = new caffe.PReLUParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "filler":
                    message.filler = caffe.FillerParameter.decodeText(reader);
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

caffe.PReLUParameter.prototype.filler = null;
caffe.PReLUParameter.prototype.channel_shared = false;
