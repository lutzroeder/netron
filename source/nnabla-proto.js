var $root = protobuf.get('nnabla');

$root.nnabla = {};

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
        const message = new $root.nnabla.Shape();
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

$root.nnabla.Communicator = class Communicator {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Communicator();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.Context();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.backends.push(reader.string());
                    break;
                case 2:
                    message.array_class = reader.string();
                    break;
                case 3:
                    message.device_id = reader.string();
                    break;
                case 4:
                    message.backend = reader.string();
                    break;
                case 5:
                    message.compute_backend = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.Context();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "backends":
                    reader.array(message.backends, () => reader.string());
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
$root.nnabla.Context.prototype.backend = "";
$root.nnabla.Context.prototype.compute_backend = "";

$root.nnabla.NNablaProtoBuf = class NNablaProtoBuf {

    constructor() {
        this.network = [];
        this.parameter = [];
        this.dataset = [];
        this.optimizer = [];
        this.monitor = [];
        this.executor = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.NNablaProtoBuf();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.string();
                    break;
                case 2:
                    message.global_config = $root.nnabla.GlobalConfig.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.training_config = $root.nnabla.TrainingConfig.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.network.push($root.nnabla.Network.decode(reader, reader.uint32()));
                    break;
                case 200:
                    message.parameter.push($root.nnabla.Parameter.decode(reader, reader.uint32()));
                    break;
                case 300:
                    message.dataset.push($root.nnabla.Dataset.decode(reader, reader.uint32()));
                    break;
                case 400:
                    message.optimizer.push($root.nnabla.Optimizer.decode(reader, reader.uint32()));
                    break;
                case 500:
                    message.monitor.push($root.nnabla.Monitor.decode(reader, reader.uint32()));
                    break;
                case 600:
                    message.executor.push($root.nnabla.Executor.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.NNablaProtoBuf();
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
                case "parameter":
                    message.parameter.push($root.nnabla.Parameter.decodeText(reader));
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

$root.nnabla.NNablaProtoBuf.prototype.version = "";
$root.nnabla.NNablaProtoBuf.prototype.global_config = null;
$root.nnabla.NNablaProtoBuf.prototype.training_config = null;

$root.nnabla.GlobalConfig = class GlobalConfig {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GlobalConfig();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.default_context = $root.nnabla.Context.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.TrainingConfig = class TrainingConfig {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TrainingConfig();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.max_epoch = reader.int64();
                    break;
                case 2:
                    message.iter_per_epoch = reader.int64();
                    break;
                case 100:
                    message.save_best = reader.bool();
                    break;
                case 200:
                    message.monitor_interval = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.TrainingConfig.prototype.max_epoch = protobuf.Int64.create(0);
$root.nnabla.TrainingConfig.prototype.iter_per_epoch = protobuf.Int64.create(0);
$root.nnabla.TrainingConfig.prototype.save_best = false;
$root.nnabla.TrainingConfig.prototype.monitor_interval = protobuf.Int64.create(0);

$root.nnabla.Network = class Network {

    constructor() {
        this.repeat_info = [];
        this.variable = [];
        this["function"] = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Network();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 10:
                    message.batch_size = reader.int64();
                    break;
                case 11:
                    message.repeat_info.push($root.nnabla.RepeatInfo.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.variable.push($root.nnabla.Variable.decode(reader, reader.uint32()));
                    break;
                case 200:
                    message["function"].push($root.nnabla.Function.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    message["function"].push($root.nnabla.Function.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Network.prototype.name = "";
$root.nnabla.Network.prototype.batch_size = protobuf.Int64.create(0);

$root.nnabla.RepeatInfo = class RepeatInfo {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RepeatInfo();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.id = reader.string();
                    break;
                case 2:
                    message.times = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.RepeatInfo.prototype.times = protobuf.Int64.create(0);

$root.nnabla.RepeatParameter = class RepeatParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RepeatParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.repeat_id = reader.string();
                    break;
                case 2:
                    message.times = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.RepeatParameter.prototype.times = protobuf.Int64.create(0);

$root.nnabla.RecurrentParameter = class RecurrentParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RecurrentParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.repeat_id = reader.string();
                    break;
                case 2:
                    message.length = reader.int64();
                    break;
                case 3:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.RecurrentParameter.prototype.length = protobuf.Int64.create(0);
$root.nnabla.RecurrentParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.Variable = class Variable {

    constructor() {
        this.repeat_id = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Variable();
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
                    message.repeat_id.push(reader.string());
                    break;
                case 20:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.initializer = $root.nnabla.Initializer.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.repeat_id, () => reader.string());
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
$root.nnabla.Variable.prototype.initializer = null;

$root.nnabla.Initializer = class Initializer {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Initializer();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.type = reader.string();
                    break;
                case 10:
                    message.multiplier = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.Initializer.prototype.multiplier = 0;

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
                    message.data = reader.floats(message.data, tag);
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

    static decodeText(reader) {
        const message = new $root.nnabla.Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "shape":
                    message.shape = $root.nnabla.Shape.decodeText(reader);
                    break;
                case "data":
                    reader.array(message.data, () => reader.float());
                    break;
                case "need_grad":
                    message.need_grad = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.Parameter.prototype.variable_name = "";
$root.nnabla.Parameter.prototype.shape = null;
$root.nnabla.Parameter.prototype.need_grad = false;

$root.nnabla.Dataset = class Dataset {

    constructor() {
        this.variable = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Dataset();
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
                case 10:
                    message.uri = reader.string();
                    break;
                case 20:
                    message.batch_size = reader.int64();
                    break;
                case 30:
                    message.cache_dir = reader.string();
                    break;
                case 31:
                    message.overwrite_cache = reader.bool();
                    break;
                case 32:
                    message.create_cache_explicitly = reader.bool();
                    break;
                case 100:
                    message.shuffle = reader.bool();
                    break;
                case 101:
                    message.no_image_normalization = reader.bool();
                    break;
                case 200:
                    message.variable.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.variable, () => reader.string());
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
$root.nnabla.Dataset.prototype.batch_size = protobuf.Int64.create(0);
$root.nnabla.Dataset.prototype.cache_dir = "";
$root.nnabla.Dataset.prototype.overwrite_cache = false;
$root.nnabla.Dataset.prototype.create_cache_explicitly = false;
$root.nnabla.Dataset.prototype.shuffle = false;
$root.nnabla.Dataset.prototype.no_image_normalization = false;

$root.nnabla.Optimizer = class Optimizer {

    constructor() {
        this.dataset_name = [];
        this.data_variable = [];
        this.generator_variable = [];
        this.loss_variable = [];
        this.parameter_variable = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Optimizer();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 3:
                    message.order = reader.int64();
                    break;
                case 10:
                    message.network_name = reader.string();
                    break;
                case 20:
                    message.dataset_name.push(reader.string());
                    break;
                case 30:
                    message.solver = $root.nnabla.Solver.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.update_interval = reader.int64();
                    break;
                case 50:
                    message.data_variable.push($root.nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 60:
                    message.generator_variable.push($root.nnabla.GeneratorVariable.decode(reader, reader.uint32()));
                    break;
                case 70:
                    message.loss_variable.push($root.nnabla.LossVariable.decode(reader, reader.uint32()));
                    break;
                case 80:
                    message.parameter_variable.push($root.nnabla.ParameterVariable.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.start_iter = reader.int64();
                    break;
                case 101:
                    message.end_iter = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.dataset_name, () => reader.string());
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
$root.nnabla.Optimizer.prototype.order = protobuf.Int64.create(0);
$root.nnabla.Optimizer.prototype.network_name = "";
$root.nnabla.Optimizer.prototype.solver = null;
$root.nnabla.Optimizer.prototype.update_interval = protobuf.Int64.create(0);
$root.nnabla.Optimizer.prototype.start_iter = protobuf.Int64.create(0);
$root.nnabla.Optimizer.prototype.end_iter = protobuf.Int64.create(0);

$root.nnabla.SolverStateParameter = class SolverStateParameter {

    constructor() {
        this.data = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SolverStateParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.data = reader.floats(message.data, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.data, () => reader.float());
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

$root.nnabla.SolverState = class SolverState {

    constructor() {
        this.state_parameter = {};
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SolverState();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.t = reader.uint32();
                    break;
                case 2:
                    reader.entry(message.state_parameter, () => reader.string(), () => $root.nnabla.SolverStateParameter.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SolverState();
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

$root.nnabla.Solver = class Solver {

    constructor() {
        this.states = {};
    }

    get parameter() {
        $root.nnabla.Solver.parameterSet = $root.nnabla.Solver.parameterSet || new Set([ "sgd_param", "sgdw_param", "momentum_param", "lars_param", "nesterov_param", "adadelta_param", "adagrad_param", "adabelief_param", "rmsprop_param", "rmsprop_graves_param", "adam_param", "adamw_param", "adabound_param", "adamax_param", "amsgrad_param", "amsbound_param", "lamb_param"]);
        return Object.keys(this).find((key) => $root.nnabla.Solver.parameterSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Solver();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.type = reader.string();
                    break;
                case 10:
                    message.context = $root.nnabla.Context.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.weight_decay = reader.float();
                    break;
                case 40:
                    reader.entry(message.states, () => reader.string(), () => $root.nnabla.SolverState.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.sgd_param = $root.nnabla.SgdParameter.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.sgdw_param = $root.nnabla.SgdWParameter.decode(reader, reader.uint32());
                    break;
                case 102:
                    message.momentum_param = $root.nnabla.MomentumParameter.decode(reader, reader.uint32());
                    break;
                case 103:
                    message.lars_param = $root.nnabla.LarsParameter.decode(reader, reader.uint32());
                    break;
                case 104:
                    message.nesterov_param = $root.nnabla.NesterovParameter.decode(reader, reader.uint32());
                    break;
                case 105:
                    message.adadelta_param = $root.nnabla.AdadeltaParameter.decode(reader, reader.uint32());
                    break;
                case 106:
                    message.adagrad_param = $root.nnabla.AdagradParameter.decode(reader, reader.uint32());
                    break;
                case 107:
                    message.adabelief_param = $root.nnabla.AdaBeliefParameter.decode(reader, reader.uint32());
                    break;
                case 108:
                    message.rmsprop_param = $root.nnabla.RMSpropParameter.decode(reader, reader.uint32());
                    break;
                case 109:
                    message.rmsprop_graves_param = $root.nnabla.RMSpropGravesParameter.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.adam_param = $root.nnabla.AdamParameter.decode(reader, reader.uint32());
                    break;
                case 111:
                    message.adamw_param = $root.nnabla.AdamWParameter.decode(reader, reader.uint32());
                    break;
                case 112:
                    message.adabound_param = $root.nnabla.AdaBoundParameter.decode(reader, reader.uint32());
                    break;
                case 113:
                    message.adamax_param = $root.nnabla.AdamaxParameter.decode(reader, reader.uint32());
                    break;
                case 114:
                    message.amsgrad_param = $root.nnabla.AMSGRADParameter.decode(reader, reader.uint32());
                    break;
                case 115:
                    message.amsbound_param = $root.nnabla.AMSBoundParameter.decode(reader, reader.uint32());
                    break;
                case 116:
                    message.lamb_param = $root.nnabla.LambParameter.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.lr_scheduler_type = reader.string();
                    break;
                case 210:
                    message.polynomial_scheduler_param = $root.nnabla.PolynomialSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 211:
                    message.cosine_scheduler_param = $root.nnabla.CosineSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 212:
                    message.exponential_scheduler_param = $root.nnabla.ExponentialSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 213:
                    message.step_scheduler_param = $root.nnabla.StepSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 299:
                    message.custom_scheduler_param = $root.nnabla.CustomSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 300:
                    message.lr_warmup_scheduler_type = reader.string();
                    break;
                case 310:
                    message.linear_warmup_scheduler_param = $root.nnabla.LinearWarmupSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 30:
                    message.lr_decay = reader.float();
                    break;
                case 31:
                    message.lr_decay_interval = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                case "lamb_param":
                    message.lamb_param = $root.nnabla.LambParameter.decodeText(reader);
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
$root.nnabla.Solver.prototype.weight_decay = 0;
$root.nnabla.Solver.prototype.lr_scheduler_type = "";
$root.nnabla.Solver.prototype.polynomial_scheduler_param = null;
$root.nnabla.Solver.prototype.cosine_scheduler_param = null;
$root.nnabla.Solver.prototype.exponential_scheduler_param = null;
$root.nnabla.Solver.prototype.step_scheduler_param = null;
$root.nnabla.Solver.prototype.custom_scheduler_param = null;
$root.nnabla.Solver.prototype.lr_warmup_scheduler_type = "";
$root.nnabla.Solver.prototype.linear_warmup_scheduler_param = null;
$root.nnabla.Solver.prototype.lr_decay = 0;
$root.nnabla.Solver.prototype.lr_decay_interval = protobuf.Int64.create(0);

$root.nnabla.SgdParameter = class SgdParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SgdParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SgdParameter.prototype.lr = 0;

$root.nnabla.SgdWParameter = class SgdWParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SgdWParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.momentum = reader.float();
                    break;
                case 3:
                    message.wd = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SgdWParameter.prototype.lr = 0;
$root.nnabla.SgdWParameter.prototype.momentum = 0;
$root.nnabla.SgdWParameter.prototype.wd = 0;

$root.nnabla.MomentumParameter = class MomentumParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MomentumParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.momentum = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.MomentumParameter.prototype.lr = 0;
$root.nnabla.MomentumParameter.prototype.momentum = 0;

$root.nnabla.LarsParameter = class LarsParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LarsParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.momentum = reader.float();
                    break;
                case 3:
                    message.coefficient = reader.float();
                    break;
                case 4:
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

$root.nnabla.LarsParameter.prototype.lr = 0;
$root.nnabla.LarsParameter.prototype.momentum = 0;
$root.nnabla.LarsParameter.prototype.coefficient = 0;
$root.nnabla.LarsParameter.prototype.eps = 0;

$root.nnabla.NesterovParameter = class NesterovParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.NesterovParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.momentum = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.NesterovParameter.prototype.lr = 0;
$root.nnabla.NesterovParameter.prototype.momentum = 0;

$root.nnabla.AdadeltaParameter = class AdadeltaParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdadeltaParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.decay = reader.float();
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

$root.nnabla.AdadeltaParameter.prototype.lr = 0;
$root.nnabla.AdadeltaParameter.prototype.decay = 0;
$root.nnabla.AdadeltaParameter.prototype.eps = 0;

$root.nnabla.AdagradParameter = class AdagradParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdagradParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
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

$root.nnabla.AdagradParameter.prototype.lr = 0;
$root.nnabla.AdagradParameter.prototype.eps = 0;

$root.nnabla.AdaBeliefParameter = class AdaBeliefParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdaBeliefParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.wd = reader.float();
                    break;
                case 6:
                    message.amsgrad = reader.bool();
                    break;
                case 7:
                    message.weight_decouple = reader.bool();
                    break;
                case 8:
                    message.fixed_decay = reader.bool();
                    break;
                case 9:
                    message.rectify = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AdaBeliefParameter.prototype.alpha = 0;
$root.nnabla.AdaBeliefParameter.prototype.beta1 = 0;
$root.nnabla.AdaBeliefParameter.prototype.beta2 = 0;
$root.nnabla.AdaBeliefParameter.prototype.eps = 0;
$root.nnabla.AdaBeliefParameter.prototype.wd = 0;
$root.nnabla.AdaBeliefParameter.prototype.amsgrad = false;
$root.nnabla.AdaBeliefParameter.prototype.weight_decouple = false;
$root.nnabla.AdaBeliefParameter.prototype.fixed_decay = false;
$root.nnabla.AdaBeliefParameter.prototype.rectify = false;

$root.nnabla.RMSpropParameter = class RMSpropParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RMSpropParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.decay = reader.float();
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

$root.nnabla.RMSpropParameter.prototype.lr = 0;
$root.nnabla.RMSpropParameter.prototype.decay = 0;
$root.nnabla.RMSpropParameter.prototype.eps = 0;

$root.nnabla.RMSpropGravesParameter = class RMSpropGravesParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RMSpropGravesParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.decay = reader.float();
                    break;
                case 3:
                    message.momentum = reader.float();
                    break;
                case 4:
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

$root.nnabla.RMSpropGravesParameter.prototype.lr = 0;
$root.nnabla.RMSpropGravesParameter.prototype.decay = 0;
$root.nnabla.RMSpropGravesParameter.prototype.momentum = 0;
$root.nnabla.RMSpropGravesParameter.prototype.eps = 0;

$root.nnabla.AdamParameter = class AdamParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdamParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
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

$root.nnabla.AdamParameter.prototype.alpha = 0;
$root.nnabla.AdamParameter.prototype.beta1 = 0;
$root.nnabla.AdamParameter.prototype.beta2 = 0;
$root.nnabla.AdamParameter.prototype.eps = 0;

$root.nnabla.AdamWParameter = class AdamWParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdamWParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.wd = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AdamWParameter.prototype.alpha = 0;
$root.nnabla.AdamWParameter.prototype.beta1 = 0;
$root.nnabla.AdamWParameter.prototype.beta2 = 0;
$root.nnabla.AdamWParameter.prototype.eps = 0;
$root.nnabla.AdamWParameter.prototype.wd = 0;

$root.nnabla.AdaBoundParameter = class AdaBoundParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdaBoundParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.final_lr = reader.float();
                    break;
                case 6:
                    message.gamma = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AdaBoundParameter.prototype.alpha = 0;
$root.nnabla.AdaBoundParameter.prototype.beta1 = 0;
$root.nnabla.AdaBoundParameter.prototype.beta2 = 0;
$root.nnabla.AdaBoundParameter.prototype.eps = 0;
$root.nnabla.AdaBoundParameter.prototype.final_lr = 0;
$root.nnabla.AdaBoundParameter.prototype.gamma = 0;

$root.nnabla.AdamaxParameter = class AdamaxParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AdamaxParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
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

$root.nnabla.AdamaxParameter.prototype.alpha = 0;
$root.nnabla.AdamaxParameter.prototype.beta1 = 0;
$root.nnabla.AdamaxParameter.prototype.beta2 = 0;
$root.nnabla.AdamaxParameter.prototype.eps = 0;

$root.nnabla.AMSGRADParameter = class AMSGRADParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AMSGRADParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.bias_correction = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AMSGRADParameter.prototype.alpha = 0;
$root.nnabla.AMSGRADParameter.prototype.beta1 = 0;
$root.nnabla.AMSGRADParameter.prototype.beta2 = 0;
$root.nnabla.AMSGRADParameter.prototype.eps = 0;
$root.nnabla.AMSGRADParameter.prototype.bias_correction = false;

$root.nnabla.AMSBoundParameter = class AMSBoundParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AMSBoundParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.final_lr = reader.float();
                    break;
                case 6:
                    message.gamma = reader.float();
                    break;
                case 7:
                    message.bias_correction = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AMSBoundParameter.prototype.alpha = 0;
$root.nnabla.AMSBoundParameter.prototype.beta1 = 0;
$root.nnabla.AMSBoundParameter.prototype.beta2 = 0;
$root.nnabla.AMSBoundParameter.prototype.eps = 0;
$root.nnabla.AMSBoundParameter.prototype.final_lr = 0;
$root.nnabla.AMSBoundParameter.prototype.gamma = 0;
$root.nnabla.AMSBoundParameter.prototype.bias_correction = false;

$root.nnabla.LambParameter = class LambParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LambParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.eta = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                case 4:
                    message.gamma_l = reader.float();
                    break;
                case 5:
                    message.gamma_u = reader.float();
                    break;
                case 6:
                    message.eps = reader.float();
                    break;
                case 7:
                    message.bias_correction = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.LambParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "eta":
                    message.eta = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                case "gamma_l":
                    message.gamma_l = reader.float();
                    break;
                case "gamma_u":
                    message.gamma_u = reader.float();
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

$root.nnabla.LambParameter.prototype.eta = 0;
$root.nnabla.LambParameter.prototype.beta1 = 0;
$root.nnabla.LambParameter.prototype.beta2 = 0;
$root.nnabla.LambParameter.prototype.gamma_l = 0;
$root.nnabla.LambParameter.prototype.gamma_u = 0;
$root.nnabla.LambParameter.prototype.eps = 0;
$root.nnabla.LambParameter.prototype.bias_correction = false;

$root.nnabla.PolynomialSchedulerParameter = class PolynomialSchedulerParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.PolynomialSchedulerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.max_iter = reader.float();
                    break;
                case 2:
                    message.power = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.PolynomialSchedulerParameter.prototype.max_iter = 0;
$root.nnabla.PolynomialSchedulerParameter.prototype.power = 0;

$root.nnabla.CosineSchedulerParameter = class CosineSchedulerParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.CosineSchedulerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.max_iter = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CosineSchedulerParameter.prototype.max_iter = 0;

$root.nnabla.ExponentialSchedulerParameter = class ExponentialSchedulerParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ExponentialSchedulerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.gamma = reader.float();
                    break;
                case 2:
                    message.iter_interval = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ExponentialSchedulerParameter.prototype.gamma = 0;
$root.nnabla.ExponentialSchedulerParameter.prototype.iter_interval = protobuf.Int64.create(0);

$root.nnabla.StepSchedulerParameter = class StepSchedulerParameter {

    constructor() {
        this.iter_steps = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.StepSchedulerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.gamma = reader.float();
                    break;
                case 2:
                    message.iter_steps = reader.array(message.iter_steps, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.StepSchedulerParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "gamma":
                    message.gamma = reader.float();
                    break;
                case "iter_steps":
                    reader.array(message.iter_steps, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.StepSchedulerParameter.prototype.gamma = 0;

$root.nnabla.CustomSchedulerParameter = class CustomSchedulerParameter {

    constructor() {
        this.data_variable = [];
        this.output_variable = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.CustomSchedulerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.max_iter = reader.float();
                    break;
                case 10:
                    message.network_name = reader.string();
                    break;
                case 50:
                    message.data_variable.push($root.nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 80:
                    message.output_variable.push($root.nnabla.OutputVariable.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CustomSchedulerParameter.prototype.max_iter = 0;
$root.nnabla.CustomSchedulerParameter.prototype.network_name = "";

$root.nnabla.LinearWarmupSchedulerParameter = class LinearWarmupSchedulerParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LinearWarmupSchedulerParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.warmup_iter = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LinearWarmupSchedulerParameter.prototype.warmup_iter = protobuf.Int64.create(0);

$root.nnabla.DataVariable = class DataVariable {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.DataVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 3:
                    message.data_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.GeneratorVariable = class GeneratorVariable {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GeneratorVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.multiplier = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.GeneratorVariable.prototype.multiplier = 0;

$root.nnabla.LossVariable = class LossVariable {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LossVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ParameterVariable = class ParameterVariable {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ParameterVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 2:
                    message.learning_rate_multiplier = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.ParameterVariable.prototype.learning_rate_multiplier = 0;

$root.nnabla.Monitor = class Monitor {

    constructor() {
        this.dataset_name = [];
        this.data_variable = [];
        this.generator_variable = [];
        this.monitor_variable = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Monitor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 10:
                    message.network_name = reader.string();
                    break;
                case 20:
                    message.dataset_name.push(reader.string());
                    break;
                case 50:
                    message.data_variable.push($root.nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 60:
                    message.generator_variable.push($root.nnabla.GeneratorVariable.decode(reader, reader.uint32()));
                    break;
                case 70:
                    message.monitor_variable.push($root.nnabla.MonitorVariable.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.dataset_name, () => reader.string());
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

$root.nnabla.Monitor.prototype.name = "";
$root.nnabla.Monitor.prototype.network_name = "";

$root.nnabla.MonitorVariable = class MonitorVariable {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MonitorVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.data_name = reader.string();
                    break;
                case 100:
                    message.multiplier = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.MonitorVariable.prototype.multiplier = 0;

$root.nnabla.Executor = class Executor {

    constructor() {
        this.data_variable = [];
        this.generator_variable = [];
        this.loss_variable = [];
        this.output_variable = [];
        this.parameter_variable = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Executor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 10:
                    message.network_name = reader.string();
                    break;
                case 20:
                    message.num_evaluations = reader.int64();
                    break;
                case 21:
                    message.repeat_evaluation_type = reader.string();
                    break;
                case 30:
                    message.need_back_propagation = reader.bool();
                    break;
                case 50:
                    message.data_variable.push($root.nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 60:
                    message.generator_variable.push($root.nnabla.GeneratorVariable.decode(reader, reader.uint32()));
                    break;
                case 70:
                    message.loss_variable.push($root.nnabla.LossVariable.decode(reader, reader.uint32()));
                    break;
                case 80:
                    message.output_variable.push($root.nnabla.OutputVariable.decode(reader, reader.uint32()));
                    break;
                case 90:
                    message.parameter_variable.push($root.nnabla.ParameterVariable.decode(reader, reader.uint32()));
                    break;
                case 101:
                    message.no_image_normalization = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.Executor.prototype.num_evaluations = protobuf.Int64.create(0);
$root.nnabla.Executor.prototype.repeat_evaluation_type = "";
$root.nnabla.Executor.prototype.need_back_propagation = false;
$root.nnabla.Executor.prototype.no_image_normalization = false;

$root.nnabla.OutputVariable = class OutputVariable {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.OutputVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.data_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.Function = class Function {

    constructor() {
        this.repeat_id = [];
        this.input = [];
        this.output = [];
    }

    get parameter() {
        $root.nnabla.Function.parameterSet = $root.nnabla.Function.parameterSet || new Set([ "affine_param", "rnn_param", "lstm_param", "gru_param", "convolution_param", "fused_convolution_param", "depthwise_convolution_param", "deconvolution_param", "depthwise_deconvolution_param", "deformable_convolution_param", "max_pooling_param", "average_pooling_param", "sum_pooling_param", "unpooling_param", "roi_align_param", "relu_param", "leaky_relu_param", "softmax_param", "log_softmax_param", "elu_param", "selu_param", "crelu_param", "celu_param", "prelu_param", "softplus_param", "fused_batch_normalization_param", "batch_normalization_param", "group_normalization_param", "instance_normalization_param", "layer_normalization_param", "norm_normalization_param", "sync_batch_normalization_param", "tensor_normalization_param", "weight_normalization_param", "weight_standardization_param", "spectral_norm_param", "mean_subtraction_param", "clip_grad_by_norm_param", "sum_param", "cumsum_param", "mean_param", "max_param", "min_param", "norm_param", "prod_param", "cumprod_param", "add2_param", "bc_add2_param", "sub2_param", "mul2_param", "div2_param", "pow2_param", "add_scalar_param", "mul_scalar_param", "pow_scalar_param", "r_sub_scalar_param", "r_div_scalar_param", "r_pow_scalar_param", "sign_param", "minimum_scalar_param", "maximum_scalar_param", "searchsorted_param", "logical_and_scalar_param", "logical_or_scalar_param", "logical_xor_scalar_param", "equal_scalar_param", "not_equal_scalar_param", "greater_equal_scalar_param", "greater_scalar_param", "less_equal_scalar_param", "less_scalar_param", "reset_nan_param", "reset_inf_param", "constant_param", "arange_param", "linspace_param", "batch_matmul_param", "round_param", "ceil_param", "floor_param", "concatenate_param", "split_param", "stack_param", "slice_param", "pad_param", "transpose_param", "broadcast_param", "broadcast_to_param", "tile_param", "one_hot_param", "flip_param", "shift_param", "sort_param", "reshape_param", "shape_param", "meshgrid_param", "batch_cholesky_param", "gather_param", "scatter_nd_param", "scatter_add_param", "bool_fill_param", "pack_padded_sequence_param", "pad_packed_sequence_param", "interpolate_param", "fft_param", "ifft_param", "stft_param", "istft_param", "dropout_param", "top_k_data_param", "top_k_grad_param", "rand_param", "randint_param", "randn_param", "rand_binomial_param", "rand_beta_param", "rand_gamma_param", "random_choice_param", "random_crop_param", "random_flip_param", "random_shift_param", "random_erase_param", "image_augmentation_param", "softmax_cross_entropy_param", "categorical_cross_entropy_param", "huber_loss_param", "epsilon_insensitive_loss_param", "kl_multinomial_param", "affine_grid_param", "warp_by_grid_param", "binary_connect_affine_param", "binary_connect_convolution_param", "binary_weight_affine_param", "binary_weight_convolution_param", "inq_affine_param", "inq_convolution_param", "fixed_point_quantize_param", "min_max_quantize_param", "pow2_quantize_param", "prune_param", "quantize_linear_param", "top_n_error_param", "confusion_matrix_param", "vat_noise_param", "sink_param", "nms_detection2d_param", "max_pooling_backward_param", "patch_correlation_param"]);
        return Object.keys(this).find((key) => $root.nnabla.Function.parameterSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Function();
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
                    message.repeat_id.push(reader.string());
                    break;
                case 10:
                    message.context = $root.nnabla.Context.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.input.push(reader.string());
                    break;
                case 30:
                    message.output.push(reader.string());
                    break;
                case 1001:
                    message.affine_param = $root.nnabla.AffineParameter.decode(reader, reader.uint32());
                    break;
                case 1002:
                    message.rnn_param = $root.nnabla.RNNParameter.decode(reader, reader.uint32());
                    break;
                case 1003:
                    message.lstm_param = $root.nnabla.LSTMParameter.decode(reader, reader.uint32());
                    break;
                case 1004:
                    message.gru_param = $root.nnabla.GRUParameter.decode(reader, reader.uint32());
                    break;
                case 1005:
                    message.convolution_param = $root.nnabla.ConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1006:
                    message.fused_convolution_param = $root.nnabla.FusedConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1007:
                    message.depthwise_convolution_param = $root.nnabla.DepthwiseConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1008:
                    message.deconvolution_param = $root.nnabla.DeconvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1009:
                    message.depthwise_deconvolution_param = $root.nnabla.DepthwiseDeconvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1010:
                    message.deformable_convolution_param = $root.nnabla.DeformableConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1012:
                    message.max_pooling_param = $root.nnabla.MaxPoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1013:
                    message.average_pooling_param = $root.nnabla.AveragePoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1015:
                    message.sum_pooling_param = $root.nnabla.SumPoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1016:
                    message.unpooling_param = $root.nnabla.UnpoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1018:
                    message.roi_align_param = $root.nnabla.RoiAlignParameter.decode(reader, reader.uint32());
                    break;
                case 1022:
                    message.relu_param = $root.nnabla.ReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1023:
                    message.leaky_relu_param = $root.nnabla.LeakyReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1024:
                    message.softmax_param = $root.nnabla.SoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 1025:
                    message.log_softmax_param = $root.nnabla.LogSoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 1026:
                    message.elu_param = $root.nnabla.ELUParameter.decode(reader, reader.uint32());
                    break;
                case 1027:
                    message.selu_param = $root.nnabla.SELUParameter.decode(reader, reader.uint32());
                    break;
                case 1028:
                    message.crelu_param = $root.nnabla.CReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1029:
                    message.celu_param = $root.nnabla.CELUParameter.decode(reader, reader.uint32());
                    break;
                case 1030:
                    message.prelu_param = $root.nnabla.PReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1037:
                    message.softplus_param = $root.nnabla.SoftPlusParameter.decode(reader, reader.uint32());
                    break;
                case 1041:
                    message.fused_batch_normalization_param = $root.nnabla.FusedBatchNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1042:
                    message.batch_normalization_param = $root.nnabla.BatchNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1043:
                    message.group_normalization_param = $root.nnabla.GroupNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1044:
                    message.instance_normalization_param = $root.nnabla.InstanceNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1045:
                    message.layer_normalization_param = $root.nnabla.LayerNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1046:
                    message.norm_normalization_param = $root.nnabla.NormNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1047:
                    message.sync_batch_normalization_param = $root.nnabla.SyncBatchNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1048:
                    message.tensor_normalization_param = $root.nnabla.TensorNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1049:
                    message.weight_normalization_param = $root.nnabla.WeightNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1050:
                    message.weight_standardization_param = $root.nnabla.WeightStandardizationParameter.decode(reader, reader.uint32());
                    break;
                case 1051:
                    message.spectral_norm_param = $root.nnabla.SpectralNormParameter.decode(reader, reader.uint32());
                    break;
                case 1052:
                    message.mean_subtraction_param = $root.nnabla.MeanSubtractionParameter.decode(reader, reader.uint32());
                    break;
                case 1054:
                    message.clip_grad_by_norm_param = $root.nnabla.ClipGradByNormParameter.decode(reader, reader.uint32());
                    break;
                case 1055:
                    message.sum_param = $root.nnabla.SumParameter.decode(reader, reader.uint32());
                    break;
                case 1056:
                    message.cumsum_param = $root.nnabla.CumSumParameter.decode(reader, reader.uint32());
                    break;
                case 1057:
                    message.mean_param = $root.nnabla.MeanParameter.decode(reader, reader.uint32());
                    break;
                case 1058:
                    message.max_param = $root.nnabla.MaxParameter.decode(reader, reader.uint32());
                    break;
                case 1059:
                    message.min_param = $root.nnabla.MinParameter.decode(reader, reader.uint32());
                    break;
                case 1060:
                    message.norm_param = $root.nnabla.NormParameter.decode(reader, reader.uint32());
                    break;
                case 1061:
                    message.prod_param = $root.nnabla.ProdParameter.decode(reader, reader.uint32());
                    break;
                case 1062:
                    message.cumprod_param = $root.nnabla.CumProdParameter.decode(reader, reader.uint32());
                    break;
                case 1065:
                    message.add2_param = $root.nnabla.Add2Parameter.decode(reader, reader.uint32());
                    break;
                case 1067:
                    message.bc_add2_param = $root.nnabla.BcAdd2Parameter.decode(reader, reader.uint32());
                    break;
                case 1068:
                    message.sub2_param = $root.nnabla.Sub2Parameter.decode(reader, reader.uint32());
                    break;
                case 1069:
                    message.mul2_param = $root.nnabla.Mul2Parameter.decode(reader, reader.uint32());
                    break;
                case 1071:
                    message.div2_param = $root.nnabla.Div2Parameter.decode(reader, reader.uint32());
                    break;
                case 1072:
                    message.pow2_param = $root.nnabla.Pow2Parameter.decode(reader, reader.uint32());
                    break;
                case 1073:
                    message.add_scalar_param = $root.nnabla.AddScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1074:
                    message.mul_scalar_param = $root.nnabla.MulScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1075:
                    message.pow_scalar_param = $root.nnabla.PowScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1076:
                    message.r_sub_scalar_param = $root.nnabla.RSubScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1077:
                    message.r_div_scalar_param = $root.nnabla.RDivScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1078:
                    message.r_pow_scalar_param = $root.nnabla.RPowScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1079:
                    message.sign_param = $root.nnabla.SignParameter.decode(reader, reader.uint32());
                    break;
                case 1082:
                    message.minimum_scalar_param = $root.nnabla.MinimumScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1083:
                    message.maximum_scalar_param = $root.nnabla.MaximumScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1093:
                    message.searchsorted_param = $root.nnabla.SearchSortedParameter.decode(reader, reader.uint32());
                    break;
                case 1094:
                    message.logical_and_scalar_param = $root.nnabla.LogicalAndScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1095:
                    message.logical_or_scalar_param = $root.nnabla.LogicalOrScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1096:
                    message.logical_xor_scalar_param = $root.nnabla.LogicalXorScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1097:
                    message.equal_scalar_param = $root.nnabla.EqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1098:
                    message.not_equal_scalar_param = $root.nnabla.NotEqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1099:
                    message.greater_equal_scalar_param = $root.nnabla.GreaterEqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1100:
                    message.greater_scalar_param = $root.nnabla.GreaterScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1101:
                    message.less_equal_scalar_param = $root.nnabla.LessEqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1102:
                    message.less_scalar_param = $root.nnabla.LessScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1106:
                    message.reset_nan_param = $root.nnabla.ResetNaNParameter.decode(reader, reader.uint32());
                    break;
                case 1107:
                    message.reset_inf_param = $root.nnabla.ResetInfParameter.decode(reader, reader.uint32());
                    break;
                case 1109:
                    message.constant_param = $root.nnabla.ConstantParameter.decode(reader, reader.uint32());
                    break;
                case 1110:
                    message.arange_param = $root.nnabla.ArangeParameter.decode(reader, reader.uint32());
                    break;
                case 1111:
                    message.linspace_param = $root.nnabla.LinspaceParameter.decode(reader, reader.uint32());
                    break;
                case 1116:
                    message.batch_matmul_param = $root.nnabla.BatchMatmulParameter.decode(reader, reader.uint32());
                    break;
                case 1117:
                    message.round_param = $root.nnabla.RoundParameter.decode(reader, reader.uint32());
                    break;
                case 1118:
                    message.ceil_param = $root.nnabla.CeilParameter.decode(reader, reader.uint32());
                    break;
                case 1119:
                    message.floor_param = $root.nnabla.FloorParameter.decode(reader, reader.uint32());
                    break;
                case 1133:
                    message.concatenate_param = $root.nnabla.ConcatenateParameter.decode(reader, reader.uint32());
                    break;
                case 1134:
                    message.split_param = $root.nnabla.SplitParameter.decode(reader, reader.uint32());
                    break;
                case 1135:
                    message.stack_param = $root.nnabla.StackParameter.decode(reader, reader.uint32());
                    break;
                case 1136:
                    message.slice_param = $root.nnabla.SliceParameter.decode(reader, reader.uint32());
                    break;
                case 1137:
                    message.pad_param = $root.nnabla.PadParameter.decode(reader, reader.uint32());
                    break;
                case 1138:
                    message.transpose_param = $root.nnabla.TransposeParameter.decode(reader, reader.uint32());
                    break;
                case 1139:
                    message.broadcast_param = $root.nnabla.BroadcastParameter.decode(reader, reader.uint32());
                    break;
                case 1140:
                    message.broadcast_to_param = $root.nnabla.BroadcastToParameter.decode(reader, reader.uint32());
                    break;
                case 1141:
                    message.tile_param = $root.nnabla.TileParameter.decode(reader, reader.uint32());
                    break;
                case 1142:
                    message.one_hot_param = $root.nnabla.OneHotParameter.decode(reader, reader.uint32());
                    break;
                case 1143:
                    message.flip_param = $root.nnabla.FlipParameter.decode(reader, reader.uint32());
                    break;
                case 1144:
                    message.shift_param = $root.nnabla.ShiftParameter.decode(reader, reader.uint32());
                    break;
                case 1145:
                    message.sort_param = $root.nnabla.SortParameter.decode(reader, reader.uint32());
                    break;
                case 1146:
                    message.reshape_param = $root.nnabla.ReshapeParameter.decode(reader, reader.uint32());
                    break;
                case 1147:
                    message.shape_param = $root.nnabla.ShapeParameter.decode(reader, reader.uint32());
                    break;
                case 1150:
                    message.meshgrid_param = $root.nnabla.MeshgridParameter.decode(reader, reader.uint32());
                    break;
                case 1154:
                    message.batch_cholesky_param = $root.nnabla.BatchCholeskyParameter.decode(reader, reader.uint32());
                    break;
                case 1156:
                    message.gather_param = $root.nnabla.GatherParameter.decode(reader, reader.uint32());
                    break;
                case 1159:
                    message.scatter_nd_param = $root.nnabla.ScatterNdParameter.decode(reader, reader.uint32());
                    break;
                case 1160:
                    message.scatter_add_param = $root.nnabla.ScatterAddParameter.decode(reader, reader.uint32());
                    break;
                case 1162:
                    message.bool_fill_param = $root.nnabla.BoolFillParameter.decode(reader, reader.uint32());
                    break;
                case 1163:
                    message.pack_padded_sequence_param = $root.nnabla.PackPaddedSequenceParameter.decode(reader, reader.uint32());
                    break;
                case 1164:
                    message.pad_packed_sequence_param = $root.nnabla.PadPackedSequenceParameter.decode(reader, reader.uint32());
                    break;
                case 1165:
                    message.interpolate_param = $root.nnabla.InterpolateParameter.decode(reader, reader.uint32());
                    break;
                case 1166:
                    message.fft_param = $root.nnabla.FFTParameter.decode(reader, reader.uint32());
                    break;
                case 1167:
                    message.ifft_param = $root.nnabla.IFFTParameter.decode(reader, reader.uint32());
                    break;
                case 1168:
                    message.stft_param = $root.nnabla.STFTParameter.decode(reader, reader.uint32());
                    break;
                case 1169:
                    message.istft_param = $root.nnabla.ISTFTParameter.decode(reader, reader.uint32());
                    break;
                case 1170:
                    message.dropout_param = $root.nnabla.DropoutParameter.decode(reader, reader.uint32());
                    break;
                case 1171:
                    message.top_k_data_param = $root.nnabla.TopKDataParameter.decode(reader, reader.uint32());
                    break;
                case 1172:
                    message.top_k_grad_param = $root.nnabla.TopKGradParameter.decode(reader, reader.uint32());
                    break;
                case 1173:
                    message.rand_param = $root.nnabla.RandParameter.decode(reader, reader.uint32());
                    break;
                case 1174:
                    message.randint_param = $root.nnabla.RandintParameter.decode(reader, reader.uint32());
                    break;
                case 1175:
                    message.randn_param = $root.nnabla.RandnParameter.decode(reader, reader.uint32());
                    break;
                case 1176:
                    message.rand_binomial_param = $root.nnabla.RandBinomialParameter.decode(reader, reader.uint32());
                    break;
                case 1177:
                    message.rand_beta_param = $root.nnabla.RandBetaParameter.decode(reader, reader.uint32());
                    break;
                case 1178:
                    message.rand_gamma_param = $root.nnabla.RandGammaParameter.decode(reader, reader.uint32());
                    break;
                case 1179:
                    message.random_choice_param = $root.nnabla.RandomChoiceParameter.decode(reader, reader.uint32());
                    break;
                case 1180:
                    message.random_crop_param = $root.nnabla.RandomCropParameter.decode(reader, reader.uint32());
                    break;
                case 1181:
                    message.random_flip_param = $root.nnabla.RandomFlipParameter.decode(reader, reader.uint32());
                    break;
                case 1182:
                    message.random_shift_param = $root.nnabla.RandomShiftParameter.decode(reader, reader.uint32());
                    break;
                case 1183:
                    message.random_erase_param = $root.nnabla.RandomEraseParameter.decode(reader, reader.uint32());
                    break;
                case 1184:
                    message.image_augmentation_param = $root.nnabla.ImageAugmentationParameter.decode(reader, reader.uint32());
                    break;
                case 1187:
                    message.softmax_cross_entropy_param = $root.nnabla.SoftmaxCrossEntropyParameter.decode(reader, reader.uint32());
                    break;
                case 1188:
                    message.categorical_cross_entropy_param = $root.nnabla.CategoricalCrossEntropyParameter.decode(reader, reader.uint32());
                    break;
                case 1191:
                    message.huber_loss_param = $root.nnabla.HuberLossParameter.decode(reader, reader.uint32());
                    break;
                case 1192:
                    message.epsilon_insensitive_loss_param = $root.nnabla.EpsilonInsensitiveLossParameter.decode(reader, reader.uint32());
                    break;
                case 1193:
                    message.kl_multinomial_param = $root.nnabla.KLMultinomialParameter.decode(reader, reader.uint32());
                    break;
                case 1194:
                    message.affine_grid_param = $root.nnabla.AffineGridParameter.decode(reader, reader.uint32());
                    break;
                case 1195:
                    message.warp_by_grid_param = $root.nnabla.WarpByGridParameter.decode(reader, reader.uint32());
                    break;
                case 1199:
                    message.binary_connect_affine_param = $root.nnabla.BinaryConnectAffineParameter.decode(reader, reader.uint32());
                    break;
                case 1200:
                    message.binary_connect_convolution_param = $root.nnabla.BinaryConnectConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1201:
                    message.binary_weight_affine_param = $root.nnabla.BinaryWeightAffineParameter.decode(reader, reader.uint32());
                    break;
                case 1202:
                    message.binary_weight_convolution_param = $root.nnabla.BinaryWeightConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1203:
                    message.inq_affine_param = $root.nnabla.INQAffineParameter.decode(reader, reader.uint32());
                    break;
                case 1204:
                    message.inq_convolution_param = $root.nnabla.INQConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1205:
                    message.fixed_point_quantize_param = $root.nnabla.FixedPointQuantizeParameter.decode(reader, reader.uint32());
                    break;
                case 1206:
                    message.min_max_quantize_param = $root.nnabla.MinMaxQuantizeParameter.decode(reader, reader.uint32());
                    break;
                case 1207:
                    message.pow2_quantize_param = $root.nnabla.Pow2QuantizeParameter.decode(reader, reader.uint32());
                    break;
                case 1208:
                    message.prune_param = $root.nnabla.PruneParameter.decode(reader, reader.uint32());
                    break;
                case 1209:
                    message.quantize_linear_param = $root.nnabla.QuantizeLinearParameter.decode(reader, reader.uint32());
                    break;
                case 1211:
                    message.top_n_error_param = $root.nnabla.TopNErrorParameter.decode(reader, reader.uint32());
                    break;
                case 1213:
                    message.confusion_matrix_param = $root.nnabla.ConfusionMatrixParameter.decode(reader, reader.uint32());
                    break;
                case 1214:
                    message.vat_noise_param = $root.nnabla.VATNoiseParameter.decode(reader, reader.uint32());
                    break;
                case 1216:
                    message.sink_param = $root.nnabla.SinkParameter.decode(reader, reader.uint32());
                    break;
                case 1217:
                    message.nms_detection2d_param = $root.nnabla.NmsDetection2dParameter.decode(reader, reader.uint32());
                    break;
                case 1218:
                    message.max_pooling_backward_param = $root.nnabla.MaxPoolingBackwardParameter.decode(reader, reader.uint32());
                    break;
                case 1219:
                    message.patch_correlation_param = $root.nnabla.PatchCorrelationParameter.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.repeat_param = $root.nnabla.RepeatParameter.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.recurrent_param = $root.nnabla.RecurrentParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.repeat_id, () => reader.string());
                    break;
                case "context":
                    message.context = $root.nnabla.Context.decodeText(reader);
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
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
                case "max_pooling_param":
                    message.max_pooling_param = $root.nnabla.MaxPoolingParameter.decodeText(reader);
                    break;
                case "average_pooling_param":
                    message.average_pooling_param = $root.nnabla.AveragePoolingParameter.decodeText(reader);
                    break;
                case "sum_pooling_param":
                    message.sum_pooling_param = $root.nnabla.SumPoolingParameter.decodeText(reader);
                    break;
                case "unpooling_param":
                    message.unpooling_param = $root.nnabla.UnpoolingParameter.decodeText(reader);
                    break;
                case "roi_align_param":
                    message.roi_align_param = $root.nnabla.RoiAlignParameter.decodeText(reader);
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
                case "softplus_param":
                    message.softplus_param = $root.nnabla.SoftPlusParameter.decodeText(reader);
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
                case "add2_param":
                    message.add2_param = $root.nnabla.Add2Parameter.decodeText(reader);
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
                case "minimum_scalar_param":
                    message.minimum_scalar_param = $root.nnabla.MinimumScalarParameter.decodeText(reader);
                    break;
                case "maximum_scalar_param":
                    message.maximum_scalar_param = $root.nnabla.MaximumScalarParameter.decodeText(reader);
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
                case "reset_nan_param":
                    message.reset_nan_param = $root.nnabla.ResetNaNParameter.decodeText(reader);
                    break;
                case "reset_inf_param":
                    message.reset_inf_param = $root.nnabla.ResetInfParameter.decodeText(reader);
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
                case "shape_param":
                    message.shape_param = $root.nnabla.ShapeParameter.decodeText(reader);
                    break;
                case "meshgrid_param":
                    message.meshgrid_param = $root.nnabla.MeshgridParameter.decodeText(reader);
                    break;
                case "batch_cholesky_param":
                    message.batch_cholesky_param = $root.nnabla.BatchCholeskyParameter.decodeText(reader);
                    break;
                case "gather_param":
                    message.gather_param = $root.nnabla.GatherParameter.decodeText(reader);
                    break;
                case "scatter_nd_param":
                    message.scatter_nd_param = $root.nnabla.ScatterNdParameter.decodeText(reader);
                    break;
                case "scatter_add_param":
                    message.scatter_add_param = $root.nnabla.ScatterAddParameter.decodeText(reader);
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
                case "softmax_cross_entropy_param":
                    message.softmax_cross_entropy_param = $root.nnabla.SoftmaxCrossEntropyParameter.decodeText(reader);
                    break;
                case "categorical_cross_entropy_param":
                    message.categorical_cross_entropy_param = $root.nnabla.CategoricalCrossEntropyParameter.decodeText(reader);
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
                case "top_n_error_param":
                    message.top_n_error_param = $root.nnabla.TopNErrorParameter.decodeText(reader);
                    break;
                case "confusion_matrix_param":
                    message.confusion_matrix_param = $root.nnabla.ConfusionMatrixParameter.decodeText(reader);
                    break;
                case "vat_noise_param":
                    message.vat_noise_param = $root.nnabla.VATNoiseParameter.decodeText(reader);
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
$root.nnabla.Function.prototype.context = null;
$root.nnabla.Function.prototype.repeat_param = null;
$root.nnabla.Function.prototype.recurrent_param = null;

$root.nnabla.AffineParameter = class AffineParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AffineParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AffineParameter.prototype.base_axis = protobuf.Int64.create(0);

$root.nnabla.RNNParameter = class RNNParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RNNParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_layers = reader.int64();
                    break;
                case 2:
                    message.nonlinearity = reader.string();
                    break;
                case 3:
                    message.dropout = reader.float();
                    break;
                case 4:
                    message.bidirectional = reader.bool();
                    break;
                case 5:
                    message.training = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RNNParameter.prototype.num_layers = protobuf.Int64.create(0);
$root.nnabla.RNNParameter.prototype.nonlinearity = "";
$root.nnabla.RNNParameter.prototype.dropout = 0;
$root.nnabla.RNNParameter.prototype.bidirectional = false;
$root.nnabla.RNNParameter.prototype.training = false;

$root.nnabla.LSTMParameter = class LSTMParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LSTMParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_layers = reader.int64();
                    break;
                case 2:
                    message.dropout = reader.float();
                    break;
                case 3:
                    message.bidirectional = reader.bool();
                    break;
                case 4:
                    message.training = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LSTMParameter.prototype.num_layers = protobuf.Int64.create(0);
$root.nnabla.LSTMParameter.prototype.dropout = 0;
$root.nnabla.LSTMParameter.prototype.bidirectional = false;
$root.nnabla.LSTMParameter.prototype.training = false;

$root.nnabla.GRUParameter = class GRUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GRUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_layers = reader.int64();
                    break;
                case 2:
                    message.dropout = reader.float();
                    break;
                case 3:
                    message.bidirectional = reader.bool();
                    break;
                case 4:
                    message.training = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.GRUParameter.prototype.num_layers = protobuf.Int64.create(0);
$root.nnabla.GRUParameter.prototype.dropout = 0;
$root.nnabla.GRUParameter.prototype.bidirectional = false;
$root.nnabla.GRUParameter.prototype.training = false;

$root.nnabla.ConvolutionParameter = class ConvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.ConvolutionParameter.prototype.pad = null;
$root.nnabla.ConvolutionParameter.prototype.stride = null;
$root.nnabla.ConvolutionParameter.prototype.dilation = null;
$root.nnabla.ConvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.ConvolutionParameter.prototype.channel_last = false;

$root.nnabla.FusedConvolutionParameter = class FusedConvolutionParameter {

    constructor() {
        this.nonlinearity_args = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.FusedConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.channel_last = reader.bool();
                    break;
                case 7:
                    message.decay_rate = reader.float();
                    break;
                case 8:
                    message.eps = reader.float();
                    break;
                case 9:
                    message.batch_stat = reader.bool();
                    break;
                case 10:
                    message.nonlinearity = reader.string();
                    break;
                case 11:
                    message.nonlinearity_args = reader.floats(message.nonlinearity_args, tag);
                    break;
                case 12:
                    message.pad_mode = reader.string();
                    break;
                case 13:
                    message.constant_value = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.nonlinearity_args, () => reader.float());
                    break;
                case "pad_mode":
                    message.pad_mode = reader.string();
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

$root.nnabla.FusedConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.FusedConvolutionParameter.prototype.pad = null;
$root.nnabla.FusedConvolutionParameter.prototype.stride = null;
$root.nnabla.FusedConvolutionParameter.prototype.dilation = null;
$root.nnabla.FusedConvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.FusedConvolutionParameter.prototype.channel_last = false;
$root.nnabla.FusedConvolutionParameter.prototype.decay_rate = 0;
$root.nnabla.FusedConvolutionParameter.prototype.eps = 0;
$root.nnabla.FusedConvolutionParameter.prototype.batch_stat = false;
$root.nnabla.FusedConvolutionParameter.prototype.nonlinearity = "";
$root.nnabla.FusedConvolutionParameter.prototype.pad_mode = "";
$root.nnabla.FusedConvolutionParameter.prototype.constant_value = 0;

$root.nnabla.DepthwiseConvolutionParameter = class DepthwiseConvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.DepthwiseConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.multiplier = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.DepthwiseConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.DepthwiseConvolutionParameter.prototype.pad = null;
$root.nnabla.DepthwiseConvolutionParameter.prototype.stride = null;
$root.nnabla.DepthwiseConvolutionParameter.prototype.dilation = null;
$root.nnabla.DepthwiseConvolutionParameter.prototype.multiplier = protobuf.Int64.create(0);

$root.nnabla.DeconvolutionParameter = class DeconvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.DeconvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.channel_last = reader.bool();
                    break;
                case 7:
                    message.output_padding = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.DeconvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.DeconvolutionParameter.prototype.pad = null;
$root.nnabla.DeconvolutionParameter.prototype.stride = null;
$root.nnabla.DeconvolutionParameter.prototype.dilation = null;
$root.nnabla.DeconvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.DeconvolutionParameter.prototype.channel_last = false;
$root.nnabla.DeconvolutionParameter.prototype.output_padding = null;

$root.nnabla.DepthwiseDeconvolutionParameter = class DepthwiseDeconvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.DepthwiseDeconvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.divisor = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.DepthwiseDeconvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.pad = null;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.stride = null;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.dilation = null;
$root.nnabla.DepthwiseDeconvolutionParameter.prototype.divisor = protobuf.Int64.create(0);

$root.nnabla.DeformableConvolutionParameter = class DeformableConvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.DeformableConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.deformable_group = reader.int64();
                    break;
                case 7:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.DeformableConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.DeformableConvolutionParameter.prototype.pad = null;
$root.nnabla.DeformableConvolutionParameter.prototype.stride = null;
$root.nnabla.DeformableConvolutionParameter.prototype.dilation = null;
$root.nnabla.DeformableConvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.DeformableConvolutionParameter.prototype.deformable_group = protobuf.Int64.create(0);
$root.nnabla.DeformableConvolutionParameter.prototype.channel_last = false;

$root.nnabla.MaxPoolingParameter = class MaxPoolingParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MaxPoolingParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.MaxPoolingParameter.prototype.ignore_border = false;
$root.nnabla.MaxPoolingParameter.prototype.pad = null;
$root.nnabla.MaxPoolingParameter.prototype.channel_last = false;

$root.nnabla.AveragePoolingParameter = class AveragePoolingParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AveragePoolingParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.channel_last = reader.bool();
                    break;
                case 6:
                    message.including_pad = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.AveragePoolingParameter.prototype.ignore_border = false;
$root.nnabla.AveragePoolingParameter.prototype.pad = null;
$root.nnabla.AveragePoolingParameter.prototype.channel_last = false;
$root.nnabla.AveragePoolingParameter.prototype.including_pad = false;

$root.nnabla.SumPoolingParameter = class SumPoolingParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SumPoolingParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.SumPoolingParameter.prototype.ignore_border = false;
$root.nnabla.SumPoolingParameter.prototype.pad = null;
$root.nnabla.SumPoolingParameter.prototype.channel_last = false;

$root.nnabla.UnpoolingParameter = class UnpoolingParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.UnpoolingParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RoiAlignParameter = class RoiAlignParameter {

    constructor() {
        this.spatial_scale = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RoiAlignParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.output_size = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.spatial_scale = reader.floats(message.spatial_scale, tag);
                    break;
                case 3:
                    message.sampling_ratio = reader.int64();
                    break;
                case 4:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.spatial_scale, () => reader.float());
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
$root.nnabla.RoiAlignParameter.prototype.sampling_ratio = protobuf.Int64.create(0);
$root.nnabla.RoiAlignParameter.prototype.channel_last = false;

$root.nnabla.ReLUParameter = class ReLUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ReLUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.LeakyReLUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LeakyReLUParameter.prototype.alpha = 0;
$root.nnabla.LeakyReLUParameter.prototype.inplace = false;

$root.nnabla.SoftmaxParameter = class SoftmaxParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SoftmaxParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SoftmaxParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.LogSoftmaxParameter = class LogSoftmaxParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LogSoftmaxParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LogSoftmaxParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.ELUParameter = class ELUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ELUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ELUParameter.prototype.alpha = 0;

$root.nnabla.SELUParameter = class SELUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SELUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.scale = reader.double();
                    break;
                case 2:
                    message.alpha = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SELUParameter.prototype.scale = 0;
$root.nnabla.SELUParameter.prototype.alpha = 0;

$root.nnabla.CReLUParameter = class CReLUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.CReLUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CReLUParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.CELUParameter = class CELUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.CELUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.double();
                    break;
                case 2:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CELUParameter.prototype.alpha = 0;
$root.nnabla.CELUParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.PReLUParameter = class PReLUParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.PReLUParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.PReLUParameter.prototype.base_axis = protobuf.Int64.create(0);

$root.nnabla.SoftPlusParameter = class SoftPlusParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SoftPlusParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.beta = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SoftPlusParameter.prototype.beta = 0;

$root.nnabla.FusedBatchNormalizationParameter = class FusedBatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.FusedBatchNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.decay_rate = reader.float();
                    break;
                case 3:
                    message.eps = reader.float();
                    break;
                case 4:
                    message.batch_stat = reader.bool();
                    break;
                case 5:
                    message.nonlinearity = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.FusedBatchNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.FusedBatchNormalizationParameter.prototype.decay_rate = 0;
$root.nnabla.FusedBatchNormalizationParameter.prototype.eps = 0;
$root.nnabla.FusedBatchNormalizationParameter.prototype.batch_stat = false;
$root.nnabla.FusedBatchNormalizationParameter.prototype.nonlinearity = "";

$root.nnabla.BatchNormalizationParameter = class BatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BatchNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.decay_rate = reader.float();
                    break;
                case 3:
                    message.eps = reader.float();
                    break;
                case 4:
                    message.batch_stat = reader.bool();
                    break;
                case 5:
                    message.no_scale = reader.bool();
                    break;
                case 6:
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.BatchNormalizationParameter.prototype.decay_rate = 0;
$root.nnabla.BatchNormalizationParameter.prototype.eps = 0;
$root.nnabla.BatchNormalizationParameter.prototype.batch_stat = false;
$root.nnabla.BatchNormalizationParameter.prototype.no_scale = false;
$root.nnabla.BatchNormalizationParameter.prototype.no_bias = false;

$root.nnabla.GroupNormalizationParameter = class GroupNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GroupNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_groups = reader.int64();
                    break;
                case 2:
                    message.channel_axis = reader.int64();
                    break;
                case 3:
                    message.batch_axis = reader.array(message.batch_axis, () => reader.int64(), tag);
                    break;
                case 4:
                    message.eps = reader.float();
                    break;
                case 5:
                    message.no_scale = reader.bool();
                    break;
                case 6:
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.batch_axis, () => reader.int64());
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

$root.nnabla.GroupNormalizationParameter.prototype.num_groups = protobuf.Int64.create(0);
$root.nnabla.GroupNormalizationParameter.prototype.channel_axis = protobuf.Int64.create(0);
$root.nnabla.GroupNormalizationParameter.prototype.eps = 0;
$root.nnabla.GroupNormalizationParameter.prototype.no_scale = false;
$root.nnabla.GroupNormalizationParameter.prototype.no_bias = false;

$root.nnabla.InstanceNormalizationParameter = class InstanceNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.InstanceNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.channel_axis = reader.int64();
                    break;
                case 2:
                    message.batch_axis = reader.array(message.batch_axis, () => reader.int64(), tag);
                    break;
                case 3:
                    message.eps = reader.float();
                    break;
                case 4:
                    message.no_scale = reader.bool();
                    break;
                case 5:
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.batch_axis, () => reader.int64());
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

$root.nnabla.InstanceNormalizationParameter.prototype.channel_axis = protobuf.Int64.create(0);
$root.nnabla.InstanceNormalizationParameter.prototype.eps = 0;
$root.nnabla.InstanceNormalizationParameter.prototype.no_scale = false;
$root.nnabla.InstanceNormalizationParameter.prototype.no_bias = false;

$root.nnabla.LayerNormalizationParameter = class LayerNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LayerNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.batch_axis = reader.array(message.batch_axis, () => reader.int64(), tag);
                    break;
                case 2:
                    message.eps = reader.float();
                    break;
                case 3:
                    message.no_scale = reader.bool();
                    break;
                case 4:
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.LayerNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "batch_axis":
                    reader.array(message.batch_axis, () => reader.int64());
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

$root.nnabla.LayerNormalizationParameter.prototype.eps = 0;
$root.nnabla.LayerNormalizationParameter.prototype.no_scale = false;
$root.nnabla.LayerNormalizationParameter.prototype.no_bias = false;

$root.nnabla.NormNormalizationParameter = class NormNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.NormNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.p = reader.float();
                    break;
                case 2:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
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
        const message = new $root.nnabla.NormNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "p":
                    message.p = reader.float();
                    break;
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.NormNormalizationParameter.prototype.p = 0;
$root.nnabla.NormNormalizationParameter.prototype.eps = 0;

$root.nnabla.SyncBatchNormalizationParameter = class SyncBatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SyncBatchNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.comm = $root.nnabla.Communicator.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.group = reader.string();
                    break;
                case 3:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 4:
                    message.decay_rate = reader.float();
                    break;
                case 5:
                    message.eps = reader.float();
                    break;
                case 6:
                    message.batch_stat = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.SyncBatchNormalizationParameter.prototype.comm = null;
$root.nnabla.SyncBatchNormalizationParameter.prototype.group = "";
$root.nnabla.SyncBatchNormalizationParameter.prototype.decay_rate = 0;
$root.nnabla.SyncBatchNormalizationParameter.prototype.eps = 0;
$root.nnabla.SyncBatchNormalizationParameter.prototype.batch_stat = false;

$root.nnabla.TensorNormalizationParameter = class TensorNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TensorNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.eps = reader.float();
                    break;
                case 3:
                    message.no_scale = reader.bool();
                    break;
                case 4:
                    message.no_bias = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.TensorNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.TensorNormalizationParameter.prototype.eps = 0;
$root.nnabla.TensorNormalizationParameter.prototype.no_scale = false;
$root.nnabla.TensorNormalizationParameter.prototype.no_bias = false;

$root.nnabla.WeightNormalizationParameter = class WeightNormalizationParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.WeightNormalizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim = reader.int64();
                    break;
                case 2:
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

$root.nnabla.WeightNormalizationParameter.prototype.dim = protobuf.Int64.create(0);
$root.nnabla.WeightNormalizationParameter.prototype.eps = 0;

$root.nnabla.WeightStandardizationParameter = class WeightStandardizationParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.WeightStandardizationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.channel_axis = reader.int64();
                    break;
                case 2:
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

$root.nnabla.WeightStandardizationParameter.prototype.channel_axis = protobuf.Int64.create(0);
$root.nnabla.WeightStandardizationParameter.prototype.eps = 0;

$root.nnabla.SpectralNormParameter = class SpectralNormParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SpectralNormParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim = reader.int64();
                    break;
                case 2:
                    message.itr = reader.int64();
                    break;
                case 3:
                    message.eps = reader.float();
                    break;
                case 4:
                    message.test = reader.bool();
                    break;
                case 5:
                    message.output_u = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SpectralNormParameter.prototype.dim = protobuf.Int64.create(0);
$root.nnabla.SpectralNormParameter.prototype.itr = protobuf.Int64.create(0);
$root.nnabla.SpectralNormParameter.prototype.eps = 0;
$root.nnabla.SpectralNormParameter.prototype.test = false;
$root.nnabla.SpectralNormParameter.prototype.output_u = false;

$root.nnabla.MeanSubtractionParameter = class MeanSubtractionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MeanSubtractionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.update_running_mean = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.MeanSubtractionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.MeanSubtractionParameter.prototype.update_running_mean = false;

$root.nnabla.ClipGradByNormParameter = class ClipGradByNormParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ClipGradByNormParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.clip_norm = reader.float();
                    break;
                case 2:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.axes, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ClipGradByNormParameter.prototype.clip_norm = 0;

$root.nnabla.SumParameter = class SumParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SumParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SumParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.CumSumParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.exclusive = reader.bool();
                    break;
                case 3:
                    message.reverse = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CumSumParameter.prototype.axis = protobuf.Int64.create(0);
$root.nnabla.CumSumParameter.prototype.exclusive = false;
$root.nnabla.CumSumParameter.prototype.reverse = false;

$root.nnabla.MeanParameter = class MeanParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MeanParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.MeanParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.MaxParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keep_dims = reader.bool();
                    break;
                case 3:
                    message.with_index = reader.bool();
                    break;
                case 4:
                    message.only_index = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.MaxParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.MinParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keep_dims = reader.bool();
                    break;
                case 3:
                    message.with_index = reader.bool();
                    break;
                case 4:
                    message.only_index = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.MinParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.NormParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.p = reader.float();
                    break;
                case 2:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 3:
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.NormParameter.prototype.p = 0;
$root.nnabla.NormParameter.prototype.keep_dims = false;

$root.nnabla.ProdParameter = class ProdParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ProdParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.keep_dims = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ProdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.CumProdParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.exclusive = reader.bool();
                    break;
                case 3:
                    message.reverse = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CumProdParameter.prototype.axis = protobuf.Int64.create(0);
$root.nnabla.CumProdParameter.prototype.exclusive = false;
$root.nnabla.CumProdParameter.prototype.reverse = false;

$root.nnabla.Add2Parameter = class Add2Parameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Add2Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BcAdd2Parameter = class BcAdd2Parameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BcAdd2Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.Sub2Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.Mul2Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.Div2Parameter = class Div2Parameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Div2Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.Pow2Parameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.AddScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                case 2:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.AddScalarParameter.prototype.val = 0;
$root.nnabla.AddScalarParameter.prototype.inplace = false;

$root.nnabla.MulScalarParameter = class MulScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MulScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                case 2:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.MulScalarParameter.prototype.val = 0;
$root.nnabla.MulScalarParameter.prototype.inplace = false;

$root.nnabla.PowScalarParameter = class PowScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.PowScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                case 2:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.PowScalarParameter.prototype.val = 0;
$root.nnabla.PowScalarParameter.prototype.inplace = false;

$root.nnabla.RSubScalarParameter = class RSubScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RSubScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RSubScalarParameter.prototype.val = 0;

$root.nnabla.RDivScalarParameter = class RDivScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RDivScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RDivScalarParameter.prototype.val = 0;

$root.nnabla.RPowScalarParameter = class RPowScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RPowScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RPowScalarParameter.prototype.val = 0;

$root.nnabla.SignParameter = class SignParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SignParameter();
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

$root.nnabla.SignParameter.prototype.alpha = 0;

$root.nnabla.MinimumScalarParameter = class MinimumScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MinimumScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.MinimumScalarParameter.prototype.val = 0;

$root.nnabla.MaximumScalarParameter = class MaximumScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MaximumScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.MaximumScalarParameter.prototype.val = 0;

$root.nnabla.SearchSortedParameter = class SearchSortedParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SearchSortedParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.right = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.LogicalAndScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.LogicalOrScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.LogicalXorScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.EqualScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.EqualScalarParameter.prototype.val = 0;

$root.nnabla.NotEqualScalarParameter = class NotEqualScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.NotEqualScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.NotEqualScalarParameter.prototype.val = 0;

$root.nnabla.GreaterEqualScalarParameter = class GreaterEqualScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GreaterEqualScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.GreaterEqualScalarParameter.prototype.val = 0;

$root.nnabla.GreaterScalarParameter = class GreaterScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GreaterScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.GreaterScalarParameter.prototype.val = 0;

$root.nnabla.LessEqualScalarParameter = class LessEqualScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LessEqualScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LessEqualScalarParameter.prototype.val = 0;

$root.nnabla.LessScalarParameter = class LessScalarParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LessScalarParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LessScalarParameter.prototype.val = 0;

$root.nnabla.ResetNaNParameter = class ResetNaNParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ResetNaNParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ResetNaNParameter.prototype.val = 0;

$root.nnabla.ResetInfParameter = class ResetInfParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ResetInfParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ResetInfParameter.prototype.val = 0;

$root.nnabla.ConstantParameter = class ConstantParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ConstantParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.float();
                    break;
                case 2:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ConstantParameter.prototype.val = 0;
$root.nnabla.ConstantParameter.prototype.shape = null;

$root.nnabla.ArangeParameter = class ArangeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ArangeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start = reader.float();
                    break;
                case 2:
                    message.stop = reader.float();
                    break;
                case 3:
                    message.step = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ArangeParameter.prototype.start = 0;
$root.nnabla.ArangeParameter.prototype.stop = 0;
$root.nnabla.ArangeParameter.prototype.step = 0;

$root.nnabla.LinspaceParameter = class LinspaceParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.LinspaceParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start = reader.float();
                    break;
                case 2:
                    message.stop = reader.float();
                    break;
                case 3:
                    message.num = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.LinspaceParameter.prototype.start = 0;
$root.nnabla.LinspaceParameter.prototype.stop = 0;
$root.nnabla.LinspaceParameter.prototype.num = protobuf.Int64.create(0);

$root.nnabla.BatchMatmulParameter = class BatchMatmulParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BatchMatmulParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.transpose_a = reader.bool();
                    break;
                case 2:
                    message.transpose_b = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.RoundParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.CeilParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.FloorParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ConcatenateParameter = class ConcatenateParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ConcatenateParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ConcatenateParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.SplitParameter = class SplitParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SplitParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SplitParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.StackParameter = class StackParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.StackParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.StackParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.SliceParameter = class SliceParameter {

    constructor() {
        this.start = [];
        this.stop = [];
        this.step = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SliceParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start = reader.array(message.start, () => reader.int64(), tag);
                    break;
                case 2:
                    message.stop = reader.array(message.stop, () => reader.int64(), tag);
                    break;
                case 3:
                    message.step = reader.array(message.step, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.SliceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    reader.array(message.start, () => reader.int64());
                    break;
                case "stop":
                    reader.array(message.stop, () => reader.int64());
                    break;
                case "step":
                    reader.array(message.step, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.PadParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pad_width = reader.array(message.pad_width, () => reader.int64(), tag);
                    break;
                case 2:
                    message.mode = reader.string();
                    break;
                case 3:
                    message.constant_value = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.PadParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pad_width":
                    reader.array(message.pad_width, () => reader.int64());
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

$root.nnabla.PadParameter.prototype.mode = "";
$root.nnabla.PadParameter.prototype.constant_value = 0;

$root.nnabla.TransposeParameter = class TransposeParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TransposeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.TransposeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.BroadcastParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.BroadcastToParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BroadcastToParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.TileParameter = class TileParameter {

    constructor() {
        this.reps = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TileParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.reps = reader.array(message.reps, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.TileParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "reps":
                    reader.array(message.reps, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.OneHotParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.FlipParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.FlipParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.ShiftParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shifts = reader.array(message.shifts, () => reader.int64(), tag);
                    break;
                case 2:
                    message.border_mode = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ShiftParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shifts":
                    reader.array(message.shifts, () => reader.int64());
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

$root.nnabla.ShiftParameter.prototype.border_mode = "";

$root.nnabla.SortParameter = class SortParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SortParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.reverse = reader.bool();
                    break;
                case 3:
                    message.with_index = reader.bool();
                    break;
                case 4:
                    message.only_index = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SortParameter.prototype.axis = protobuf.Int64.create(0);
$root.nnabla.SortParameter.prototype.reverse = false;
$root.nnabla.SortParameter.prototype.with_index = false;
$root.nnabla.SortParameter.prototype.only_index = false;

$root.nnabla.ReshapeParameter = class ReshapeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ReshapeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.inplace = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.ReshapeParameter.prototype.inplace = false;

$root.nnabla.ShapeParameter = class ShapeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ShapeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start = reader.int64();
                    break;
                case 2:
                    message.end = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ShapeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    message.start = reader.int64();
                    break;
                case "end":
                    message.end = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.ShapeParameter.prototype.start = protobuf.Int64.create(0);
$root.nnabla.ShapeParameter.prototype.end = protobuf.Int64.create(0);

$root.nnabla.MeshgridParameter = class MeshgridParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MeshgridParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ij_indexing = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BatchCholeskyParameter = class BatchCholeskyParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BatchCholeskyParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.upper = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.BatchCholeskyParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "upper":
                    message.upper = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.BatchCholeskyParameter.prototype.upper = false;

$root.nnabla.GatherParameter = class GatherParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.GatherParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.batch_dims = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.GatherParameter.prototype.axis = protobuf.Int64.create(0);
$root.nnabla.GatherParameter.prototype.batch_dims = protobuf.Int64.create(0);

$root.nnabla.ScatterNdParameter = class ScatterNdParameter {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ScatterNdParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = reader.array(message.shape, () => reader.int64(), tag);
                    break;
                case 2:
                    message.add = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.ScatterNdParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    reader.array(message.shape, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.ScatterAddParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ScatterAddParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.BoolFillParameter = class BoolFillParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BoolFillParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BoolFillParameter.prototype.value = 0;

$root.nnabla.PackPaddedSequenceParameter = class PackPaddedSequenceParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.PackPaddedSequenceParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.batch_first = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

    static decode(reader, length) {
        const message = new $root.nnabla.PadPackedSequenceParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.batch_first = reader.bool();
                    break;
                case 2:
                    message.padding_value = reader.float();
                    break;
                case 3:
                    message.total_length = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.PadPackedSequenceParameter.prototype.padding_value = 0;
$root.nnabla.PadPackedSequenceParameter.prototype.total_length = protobuf.Int64.create(0);

$root.nnabla.InterpolateParameter = class InterpolateParameter {

    constructor() {
        this.output_size = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.InterpolateParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.output_size = reader.array(message.output_size, () => reader.int64(), tag);
                    break;
                case 2:
                    message.mode = reader.string();
                    break;
                case 3:
                    message.align_corners = reader.bool();
                    break;
                case 4:
                    message.half_pixel = reader.bool();
                    break;
                case 5:
                    message.half_pixel_for_nn = reader.bool();
                    break;
                case 6:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.InterpolateParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "output_size":
                    reader.array(message.output_size, () => reader.int64());
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
$root.nnabla.InterpolateParameter.prototype.align_corners = false;
$root.nnabla.InterpolateParameter.prototype.half_pixel = false;
$root.nnabla.InterpolateParameter.prototype.half_pixel_for_nn = false;
$root.nnabla.InterpolateParameter.prototype.channel_last = false;

$root.nnabla.FFTParameter = class FFTParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.FFTParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.signal_ndim = reader.int64();
                    break;
                case 2:
                    message.normalized = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.FFTParameter.prototype.signal_ndim = protobuf.Int64.create(0);
$root.nnabla.FFTParameter.prototype.normalized = false;

$root.nnabla.IFFTParameter = class IFFTParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.IFFTParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.signal_ndim = reader.int64();
                    break;
                case 2:
                    message.normalized = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.IFFTParameter.prototype.signal_ndim = protobuf.Int64.create(0);
$root.nnabla.IFFTParameter.prototype.normalized = false;

$root.nnabla.STFTParameter = class STFTParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.STFTParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.window_size = reader.int64();
                    break;
                case 2:
                    message.stride = reader.int64();
                    break;
                case 3:
                    message.fft_size = reader.int64();
                    break;
                case 4:
                    message.window_type = reader.string();
                    break;
                case 5:
                    message.center = reader.bool();
                    break;
                case 6:
                    message.pad_mode = reader.string();
                    break;
                case 7:
                    message.as_istft_backward = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.STFTParameter.prototype.window_size = protobuf.Int64.create(0);
$root.nnabla.STFTParameter.prototype.stride = protobuf.Int64.create(0);
$root.nnabla.STFTParameter.prototype.fft_size = protobuf.Int64.create(0);
$root.nnabla.STFTParameter.prototype.window_type = "";
$root.nnabla.STFTParameter.prototype.center = false;
$root.nnabla.STFTParameter.prototype.pad_mode = "";
$root.nnabla.STFTParameter.prototype.as_istft_backward = false;

$root.nnabla.ISTFTParameter = class ISTFTParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ISTFTParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.window_size = reader.int64();
                    break;
                case 2:
                    message.stride = reader.int64();
                    break;
                case 3:
                    message.fft_size = reader.int64();
                    break;
                case 4:
                    message.window_type = reader.string();
                    break;
                case 5:
                    message.center = reader.bool();
                    break;
                case 6:
                    message.pad_mode = reader.string();
                    break;
                case 7:
                    message.as_stft_backward = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ISTFTParameter.prototype.window_size = protobuf.Int64.create(0);
$root.nnabla.ISTFTParameter.prototype.stride = protobuf.Int64.create(0);
$root.nnabla.ISTFTParameter.prototype.fft_size = protobuf.Int64.create(0);
$root.nnabla.ISTFTParameter.prototype.window_type = "";
$root.nnabla.ISTFTParameter.prototype.center = false;
$root.nnabla.ISTFTParameter.prototype.pad_mode = "";
$root.nnabla.ISTFTParameter.prototype.as_stft_backward = false;

$root.nnabla.DropoutParameter = class DropoutParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.DropoutParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.p = reader.double();
                    break;
                case 2:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.DropoutParameter.prototype.p = 0;
$root.nnabla.DropoutParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.TopKDataParameter = class TopKDataParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TopKDataParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.int64();
                    break;
                case 2:
                    message.abs = reader.bool();
                    break;
                case 3:
                    message.reduce = reader.bool();
                    break;
                case 4:
                    message.base_axis = reader.int64();
                    break;
                case 5:
                    message.largest = reader.bool();
                    break;
                case 6:
                    message.with_index = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                case "largest":
                    message.largest = reader.bool();
                    break;
                case "with_index":
                    message.with_index = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.nnabla.TopKDataParameter.prototype.k = protobuf.Int64.create(0);
$root.nnabla.TopKDataParameter.prototype.abs = false;
$root.nnabla.TopKDataParameter.prototype.reduce = false;
$root.nnabla.TopKDataParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.TopKDataParameter.prototype.largest = false;
$root.nnabla.TopKDataParameter.prototype.with_index = false;

$root.nnabla.TopKGradParameter = class TopKGradParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TopKGradParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.int64();
                    break;
                case 2:
                    message.abs = reader.bool();
                    break;
                case 3:
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.TopKGradParameter.prototype.k = protobuf.Int64.create(0);
$root.nnabla.TopKGradParameter.prototype.abs = false;
$root.nnabla.TopKGradParameter.prototype.base_axis = protobuf.Int64.create(0);

$root.nnabla.RandParameter = class RandParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.low = reader.float();
                    break;
                case 2:
                    message.high = reader.float();
                    break;
                case 3:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RandParameter.prototype.low = 0;
$root.nnabla.RandParameter.prototype.high = 0;
$root.nnabla.RandParameter.prototype.shape = null;
$root.nnabla.RandParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandintParameter = class RandintParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandintParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.low = reader.int64();
                    break;
                case 2:
                    message.high = reader.int64();
                    break;
                case 3:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RandintParameter.prototype.low = protobuf.Int64.create(0);
$root.nnabla.RandintParameter.prototype.high = protobuf.Int64.create(0);
$root.nnabla.RandintParameter.prototype.shape = null;
$root.nnabla.RandintParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandnParameter = class RandnParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandnParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mu = reader.float();
                    break;
                case 2:
                    message.sigma = reader.float();
                    break;
                case 3:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RandnParameter.prototype.mu = 0;
$root.nnabla.RandnParameter.prototype.sigma = 0;
$root.nnabla.RandnParameter.prototype.shape = null;
$root.nnabla.RandnParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandBinomialParameter = class RandBinomialParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandBinomialParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.n = reader.int64();
                    break;
                case 2:
                    message.p = reader.float();
                    break;
                case 3:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RandBinomialParameter.prototype.n = protobuf.Int64.create(0);
$root.nnabla.RandBinomialParameter.prototype.p = 0;
$root.nnabla.RandBinomialParameter.prototype.shape = null;
$root.nnabla.RandBinomialParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandBetaParameter = class RandBetaParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandBetaParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alpha = reader.float();
                    break;
                case 2:
                    message.beta = reader.float();
                    break;
                case 3:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RandBetaParameter.prototype.alpha = 0;
$root.nnabla.RandBetaParameter.prototype.beta = 0;
$root.nnabla.RandBetaParameter.prototype.shape = null;
$root.nnabla.RandBetaParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandGammaParameter = class RandGammaParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandGammaParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.float();
                    break;
                case 2:
                    message.theta = reader.float();
                    break;
                case 3:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.RandGammaParameter.prototype.k = 0;
$root.nnabla.RandGammaParameter.prototype.theta = 0;
$root.nnabla.RandGammaParameter.prototype.shape = null;
$root.nnabla.RandGammaParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandomChoiceParameter = class RandomChoiceParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandomChoiceParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.replace = reader.bool();
                    break;
                case 3:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.RandomChoiceParameter.prototype.replace = false;
$root.nnabla.RandomChoiceParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandomCropParameter = class RandomCropParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandomCropParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.base_axis = reader.int64();
                    break;
                case 3:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.RandomCropParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.RandomCropParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandomFlipParameter = class RandomFlipParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandomFlipParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axes = reader.array(message.axes, () => reader.int64(), tag);
                    break;
                case 2:
                    message.base_axis = reader.int64();
                    break;
                case 3:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomFlipParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "axes":
                    reader.array(message.axes, () => reader.int64());
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

$root.nnabla.RandomFlipParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.RandomFlipParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandomShiftParameter = class RandomShiftParameter {

    constructor() {
        this.shifts = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandomShiftParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shifts = reader.array(message.shifts, () => reader.int64(), tag);
                    break;
                case 2:
                    message.border_mode = reader.string();
                    break;
                case 3:
                    message.constant_value = reader.float();
                    break;
                case 4:
                    message.base_axis = reader.int64();
                    break;
                case 5:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.RandomShiftParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shifts":
                    reader.array(message.shifts, () => reader.int64());
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

$root.nnabla.RandomShiftParameter.prototype.border_mode = "";
$root.nnabla.RandomShiftParameter.prototype.constant_value = 0;
$root.nnabla.RandomShiftParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.RandomShiftParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.RandomEraseParameter = class RandomEraseParameter {

    constructor() {
        this.area_ratios = [];
        this.aspect_ratios = [];
        this.replacements = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.RandomEraseParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.prob = reader.float();
                    break;
                case 2:
                    message.area_ratios = reader.floats(message.area_ratios, tag);
                    break;
                case 3:
                    message.aspect_ratios = reader.floats(message.aspect_ratios, tag);
                    break;
                case 4:
                    message.replacements = reader.floats(message.replacements, tag);
                    break;
                case 5:
                    message.n = reader.int64();
                    break;
                case 6:
                    message.share = reader.bool();
                    break;
                case 7:
                    message.inplace = reader.bool();
                    break;
                case 8:
                    message.base_axis = reader.int64();
                    break;
                case 9:
                    message.seed = reader.int64();
                    break;
                case 10:
                    message.channel_last = reader.bool();
                    break;
                case 11:
                    message.ste_fine_grained = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.area_ratios, () => reader.float());
                    break;
                case "aspect_ratios":
                    reader.array(message.aspect_ratios, () => reader.float());
                    break;
                case "replacements":
                    reader.array(message.replacements, () => reader.float());
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

$root.nnabla.RandomEraseParameter.prototype.prob = 0;
$root.nnabla.RandomEraseParameter.prototype.n = protobuf.Int64.create(0);
$root.nnabla.RandomEraseParameter.prototype.share = false;
$root.nnabla.RandomEraseParameter.prototype.inplace = false;
$root.nnabla.RandomEraseParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.RandomEraseParameter.prototype.seed = protobuf.Int64.create(0);
$root.nnabla.RandomEraseParameter.prototype.channel_last = false;
$root.nnabla.RandomEraseParameter.prototype.ste_fine_grained = false;

$root.nnabla.ImageAugmentationParameter = class ImageAugmentationParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ImageAugmentationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.min_scale = reader.float();
                    break;
                case 4:
                    message.max_scale = reader.float();
                    break;
                case 5:
                    message.angle = reader.float();
                    break;
                case 6:
                    message.aspect_ratio = reader.float();
                    break;
                case 7:
                    message.distortion = reader.float();
                    break;
                case 8:
                    message.flip_lr = reader.bool();
                    break;
                case 9:
                    message.flip_ud = reader.bool();
                    break;
                case 10:
                    message.brightness = reader.float();
                    break;
                case 11:
                    message.brightness_each = reader.bool();
                    break;
                case 12:
                    message.contrast = reader.float();
                    break;
                case 13:
                    message.contrast_center = reader.float();
                    break;
                case 14:
                    message.contrast_each = reader.bool();
                    break;
                case 15:
                    message.noise = reader.float();
                    break;
                case 16:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.ImageAugmentationParameter.prototype.min_scale = 0;
$root.nnabla.ImageAugmentationParameter.prototype.max_scale = 0;
$root.nnabla.ImageAugmentationParameter.prototype.angle = 0;
$root.nnabla.ImageAugmentationParameter.prototype.aspect_ratio = 0;
$root.nnabla.ImageAugmentationParameter.prototype.distortion = 0;
$root.nnabla.ImageAugmentationParameter.prototype.flip_lr = false;
$root.nnabla.ImageAugmentationParameter.prototype.flip_ud = false;
$root.nnabla.ImageAugmentationParameter.prototype.brightness = 0;
$root.nnabla.ImageAugmentationParameter.prototype.brightness_each = false;
$root.nnabla.ImageAugmentationParameter.prototype.contrast = 0;
$root.nnabla.ImageAugmentationParameter.prototype.contrast_center = 0;
$root.nnabla.ImageAugmentationParameter.prototype.contrast_each = false;
$root.nnabla.ImageAugmentationParameter.prototype.noise = 0;
$root.nnabla.ImageAugmentationParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.SoftmaxCrossEntropyParameter = class SoftmaxCrossEntropyParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SoftmaxCrossEntropyParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SoftmaxCrossEntropyParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.CategoricalCrossEntropyParameter = class CategoricalCrossEntropyParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.CategoricalCrossEntropyParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.CategoricalCrossEntropyParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.HuberLossParameter = class HuberLossParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.HuberLossParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.delta = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.HuberLossParameter.prototype.delta = 0;

$root.nnabla.EpsilonInsensitiveLossParameter = class EpsilonInsensitiveLossParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.EpsilonInsensitiveLossParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.epsilon = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.EpsilonInsensitiveLossParameter.prototype.epsilon = 0;

$root.nnabla.KLMultinomialParameter = class KLMultinomialParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.KLMultinomialParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.KLMultinomialParameter.prototype.base_axis = protobuf.Int64.create(0);

$root.nnabla.AffineGridParameter = class AffineGridParameter {

    constructor() {
        this.size = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.AffineGridParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.size = reader.array(message.size, () => reader.int64(), tag);
                    break;
                case 2:
                    message.align_corners = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.nnabla.AffineGridParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    reader.array(message.size, () => reader.int64());
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

    static decode(reader, length) {
        const message = new $root.nnabla.WarpByGridParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.mode = reader.string();
                    break;
                case 2:
                    message.padding_mode = reader.string();
                    break;
                case 3:
                    message.align_corners = reader.bool();
                    break;
                case 4:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.WarpByGridParameter.prototype.mode = "";
$root.nnabla.WarpByGridParameter.prototype.padding_mode = "";
$root.nnabla.WarpByGridParameter.prototype.align_corners = false;
$root.nnabla.WarpByGridParameter.prototype.channel_last = false;

$root.nnabla.BinaryConnectAffineParameter = class BinaryConnectAffineParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BinaryConnectAffineParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BinaryConnectAffineParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.BinaryConnectAffineParameter.prototype.quantize_zero_to = 0;

$root.nnabla.BinaryConnectConvolutionParameter = class BinaryConnectConvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BinaryConnectConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BinaryConnectConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.BinaryConnectConvolutionParameter.prototype.pad = null;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.stride = null;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.dilation = null;
$root.nnabla.BinaryConnectConvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.BinaryConnectConvolutionParameter.prototype.quantize_zero_to = 0;

$root.nnabla.BinaryWeightAffineParameter = class BinaryWeightAffineParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BinaryWeightAffineParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BinaryWeightAffineParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.BinaryWeightAffineParameter.prototype.quantize_zero_to = 0;

$root.nnabla.BinaryWeightConvolutionParameter = class BinaryWeightConvolutionParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.BinaryWeightConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.quantize_zero_to = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.BinaryWeightConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.BinaryWeightConvolutionParameter.prototype.pad = null;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.stride = null;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.dilation = null;
$root.nnabla.BinaryWeightConvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.BinaryWeightConvolutionParameter.prototype.quantize_zero_to = 0;

$root.nnabla.INQAffineParameter = class INQAffineParameter {

    constructor() {
        this.inq_iterations = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.INQAffineParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.num_bits = reader.int64();
                    break;
                case 3:
                    message.inq_iterations = reader.array(message.inq_iterations, () => reader.int64(), tag);
                    break;
                case 4:
                    message.selection_algorithm = reader.string();
                    break;
                case 5:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.inq_iterations, () => reader.int64());
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

$root.nnabla.INQAffineParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.INQAffineParameter.prototype.num_bits = protobuf.Int64.create(0);
$root.nnabla.INQAffineParameter.prototype.selection_algorithm = "";
$root.nnabla.INQAffineParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.INQConvolutionParameter = class INQConvolutionParameter {

    constructor() {
        this.inq_iterations = [];
    }

    static decode(reader, length) {
        const message = new $root.nnabla.INQConvolutionParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.num_bits = reader.int64();
                    break;
                case 7:
                    message.inq_iterations = reader.array(message.inq_iterations, () => reader.int64(), tag);
                    break;
                case 8:
                    message.selection_algorithm = reader.string();
                    break;
                case 9:
                    message.seed = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
                    reader.array(message.inq_iterations, () => reader.int64());
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

$root.nnabla.INQConvolutionParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.INQConvolutionParameter.prototype.pad = null;
$root.nnabla.INQConvolutionParameter.prototype.stride = null;
$root.nnabla.INQConvolutionParameter.prototype.dilation = null;
$root.nnabla.INQConvolutionParameter.prototype.group = protobuf.Int64.create(0);
$root.nnabla.INQConvolutionParameter.prototype.num_bits = protobuf.Int64.create(0);
$root.nnabla.INQConvolutionParameter.prototype.selection_algorithm = "";
$root.nnabla.INQConvolutionParameter.prototype.seed = protobuf.Int64.create(0);

$root.nnabla.FixedPointQuantizeParameter = class FixedPointQuantizeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.FixedPointQuantizeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sign = reader.bool();
                    break;
                case 2:
                    message.n = reader.int64();
                    break;
                case 3:
                    message.delta = reader.float();
                    break;
                case 4:
                    message.ste_fine_grained = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.FixedPointQuantizeParameter.prototype.sign = false;
$root.nnabla.FixedPointQuantizeParameter.prototype.n = protobuf.Int64.create(0);
$root.nnabla.FixedPointQuantizeParameter.prototype.delta = 0;
$root.nnabla.FixedPointQuantizeParameter.prototype.ste_fine_grained = false;

$root.nnabla.MinMaxQuantizeParameter = class MinMaxQuantizeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MinMaxQuantizeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.decay = reader.float();
                    break;
                case 2:
                    message.x_min_max = reader.bool();
                    break;
                case 3:
                    message.ema = reader.bool();
                    break;
                case 4:
                    message.ste_fine_grained = reader.bool();
                    break;
                case 5:
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

$root.nnabla.MinMaxQuantizeParameter.prototype.decay = 0;
$root.nnabla.MinMaxQuantizeParameter.prototype.x_min_max = false;
$root.nnabla.MinMaxQuantizeParameter.prototype.ema = false;
$root.nnabla.MinMaxQuantizeParameter.prototype.ste_fine_grained = false;
$root.nnabla.MinMaxQuantizeParameter.prototype.eps = 0;

$root.nnabla.Pow2QuantizeParameter = class Pow2QuantizeParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.Pow2QuantizeParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sign = reader.bool();
                    break;
                case 2:
                    message.with_zero = reader.bool();
                    break;
                case 3:
                    message.n = reader.int64();
                    break;
                case 4:
                    message.m = reader.int64();
                    break;
                case 5:
                    message.ste_fine_grained = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.Pow2QuantizeParameter.prototype.sign = false;
$root.nnabla.Pow2QuantizeParameter.prototype.with_zero = false;
$root.nnabla.Pow2QuantizeParameter.prototype.n = protobuf.Int64.create(0);
$root.nnabla.Pow2QuantizeParameter.prototype.m = protobuf.Int64.create(0);
$root.nnabla.Pow2QuantizeParameter.prototype.ste_fine_grained = false;

$root.nnabla.PruneParameter = class PruneParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.PruneParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.rate = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.PruneParameter.prototype.rate = 0;

$root.nnabla.QuantizeLinearParameter = class QuantizeLinearParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.QuantizeLinearParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.round_mode = reader.string();
                    break;
                case 2:
                    message.narrow_range = reader.bool();
                    break;
                case 3:
                    message.dtype = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.QuantizeLinearParameter.prototype.round_mode = "";
$root.nnabla.QuantizeLinearParameter.prototype.narrow_range = false;
$root.nnabla.QuantizeLinearParameter.prototype.dtype = protobuf.Int64.create(0);

$root.nnabla.TopNErrorParameter = class TopNErrorParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.TopNErrorParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                case 2:
                    message.n = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.TopNErrorParameter.prototype.axis = protobuf.Int64.create(0);
$root.nnabla.TopNErrorParameter.prototype.n = protobuf.Int64.create(0);

$root.nnabla.ConfusionMatrixParameter = class ConfusionMatrixParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.ConfusionMatrixParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.axis = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.ConfusionMatrixParameter.prototype.axis = protobuf.Int64.create(0);

$root.nnabla.VATNoiseParameter = class VATNoiseParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.VATNoiseParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
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

$root.nnabla.VATNoiseParameter.prototype.base_axis = protobuf.Int64.create(0);
$root.nnabla.VATNoiseParameter.prototype.eps = 0;

$root.nnabla.SinkParameter = class SinkParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.SinkParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.one_input_grad = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.SinkParameter.prototype.one_input_grad = false;

$root.nnabla.NmsDetection2dParameter = class NmsDetection2dParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.NmsDetection2dParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.thresh = reader.float();
                    break;
                case 2:
                    message.nms = reader.float();
                    break;
                case 3:
                    message.nms_per_class = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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

$root.nnabla.NmsDetection2dParameter.prototype.thresh = 0;
$root.nnabla.NmsDetection2dParameter.prototype.nms = 0;
$root.nnabla.NmsDetection2dParameter.prototype.nms_per_class = false;

$root.nnabla.MaxPoolingBackwardParameter = class MaxPoolingBackwardParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.MaxPoolingBackwardParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.channel_last = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
$root.nnabla.MaxPoolingBackwardParameter.prototype.ignore_border = false;
$root.nnabla.MaxPoolingBackwardParameter.prototype.pad = null;
$root.nnabla.MaxPoolingBackwardParameter.prototype.channel_last = false;

$root.nnabla.PatchCorrelationParameter = class PatchCorrelationParameter {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.nnabla.PatchCorrelationParameter();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.patch = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.shift = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.patch_step = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.shift_step = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.padding = $root.nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
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
