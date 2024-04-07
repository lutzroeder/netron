
export const nnabla = {};

nnabla.Shape = class Shape {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Shape();
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
        const message = new nnabla.Shape();
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

nnabla.Communicator = class Communicator {

    static decode(reader, length) {
        const message = new nnabla.Communicator();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Communicator();
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

nnabla.Context = class Context {

    constructor() {
        this.backends = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Context();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Context();
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

nnabla.Context.prototype.array_class = "";
nnabla.Context.prototype.device_id = "";
nnabla.Context.prototype.backend = "";
nnabla.Context.prototype.compute_backend = "";

nnabla.NNablaProtoBuf = class NNablaProtoBuf {

    constructor() {
        this.network = [];
        this.parameter = [];
        this.dataset = [];
        this.optimizer = [];
        this.monitor = [];
        this.executor = [];
    }

    static decode(reader, length) {
        const message = new nnabla.NNablaProtoBuf();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.string();
                    break;
                case 2:
                    message.global_config = nnabla.GlobalConfig.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.training_config = nnabla.TrainingConfig.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.network.push(nnabla.Network.decode(reader, reader.uint32()));
                    break;
                case 200:
                    message.parameter.push(nnabla.Parameter.decode(reader, reader.uint32()));
                    break;
                case 300:
                    message.dataset.push(nnabla.Dataset.decode(reader, reader.uint32()));
                    break;
                case 400:
                    message.optimizer.push(nnabla.Optimizer.decode(reader, reader.uint32()));
                    break;
                case 500:
                    message.monitor.push(nnabla.Monitor.decode(reader, reader.uint32()));
                    break;
                case 600:
                    message.executor.push(nnabla.Executor.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.NNablaProtoBuf();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.string();
                    break;
                case "global_config":
                    message.global_config = nnabla.GlobalConfig.decodeText(reader);
                    break;
                case "training_config":
                    message.training_config = nnabla.TrainingConfig.decodeText(reader);
                    break;
                case "network":
                    message.network.push(nnabla.Network.decodeText(reader));
                    break;
                case "parameter":
                    message.parameter.push(nnabla.Parameter.decodeText(reader));
                    break;
                case "dataset":
                    message.dataset.push(nnabla.Dataset.decodeText(reader));
                    break;
                case "optimizer":
                    message.optimizer.push(nnabla.Optimizer.decodeText(reader));
                    break;
                case "monitor":
                    message.monitor.push(nnabla.Monitor.decodeText(reader));
                    break;
                case "executor":
                    message.executor.push(nnabla.Executor.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.NNablaProtoBuf.prototype.version = "";
nnabla.NNablaProtoBuf.prototype.global_config = null;
nnabla.NNablaProtoBuf.prototype.training_config = null;

nnabla.GlobalConfig = class GlobalConfig {

    static decode(reader, length) {
        const message = new nnabla.GlobalConfig();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.default_context = nnabla.Context.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.GlobalConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "default_context":
                    message.default_context = nnabla.Context.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.GlobalConfig.prototype.default_context = null;

nnabla.TrainingConfig = class TrainingConfig {

    static decode(reader, length) {
        const message = new nnabla.TrainingConfig();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TrainingConfig();
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

nnabla.TrainingConfig.prototype.max_epoch = 0n;
nnabla.TrainingConfig.prototype.iter_per_epoch = 0n;
nnabla.TrainingConfig.prototype.save_best = false;
nnabla.TrainingConfig.prototype.monitor_interval = 0n;

nnabla.Network = class Network {

    constructor() {
        this.repeat_info = [];
        this.variable = [];
        this.function = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Network();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.repeat_info.push(nnabla.RepeatInfo.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.variable.push(nnabla.Variable.decode(reader, reader.uint32()));
                    break;
                case 200:
                    message.function.push(nnabla.Function.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.Network();
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
                    message.repeat_info.push(nnabla.RepeatInfo.decodeText(reader));
                    break;
                case "variable":
                    message.variable.push(nnabla.Variable.decodeText(reader));
                    break;
                case "function":
                    message.function.push(nnabla.Function.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.Network.prototype.name = "";
nnabla.Network.prototype.batch_size = 0n;

nnabla.RepeatInfo = class RepeatInfo {

    static decode(reader, length) {
        const message = new nnabla.RepeatInfo();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RepeatInfo();
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

nnabla.RepeatInfo.prototype.id = "";
nnabla.RepeatInfo.prototype.times = 0n;

nnabla.RepeatParameter = class RepeatParameter {

    static decode(reader, length) {
        const message = new nnabla.RepeatParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RepeatParameter();
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

nnabla.RepeatParameter.prototype.repeat_id = "";
nnabla.RepeatParameter.prototype.times = 0n;

nnabla.RecurrentParameter = class RecurrentParameter {

    static decode(reader, length) {
        const message = new nnabla.RecurrentParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RecurrentParameter();
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

nnabla.RecurrentParameter.prototype.repeat_id = "";
nnabla.RecurrentParameter.prototype.length = 0n;
nnabla.RecurrentParameter.prototype.axis = 0n;

nnabla.Variable = class Variable {

    constructor() {
        this.repeat_id = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Variable();
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
                    message.repeat_id.push(reader.string());
                    break;
                case 20:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.initializer = nnabla.Initializer.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.Variable();
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
                    message.shape = nnabla.Shape.decodeText(reader);
                    break;
                case "initializer":
                    message.initializer = nnabla.Initializer.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.Variable.prototype.name = "";
nnabla.Variable.prototype.type = "";
nnabla.Variable.prototype.shape = null;
nnabla.Variable.prototype.initializer = null;

nnabla.Initializer = class Initializer {

    static decode(reader, length) {
        const message = new nnabla.Initializer();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Initializer();
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

nnabla.Initializer.prototype.type = "";
nnabla.Initializer.prototype.multiplier = 0;

nnabla.Parameter = class Parameter {

    constructor() {
        this.data = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 20:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.Parameter.prototype.variable_name = "";
nnabla.Parameter.prototype.shape = null;
nnabla.Parameter.prototype.need_grad = false;

nnabla.Dataset = class Dataset {

    constructor() {
        this.variable = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Dataset();
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
        const message = new nnabla.Dataset();
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

nnabla.Dataset.prototype.name = "";
nnabla.Dataset.prototype.type = "";
nnabla.Dataset.prototype.uri = "";
nnabla.Dataset.prototype.batch_size = 0n;
nnabla.Dataset.prototype.cache_dir = "";
nnabla.Dataset.prototype.overwrite_cache = false;
nnabla.Dataset.prototype.create_cache_explicitly = false;
nnabla.Dataset.prototype.shuffle = false;
nnabla.Dataset.prototype.no_image_normalization = false;

nnabla.Optimizer = class Optimizer {

    constructor() {
        this.dataset_name = [];
        this.data_variable = [];
        this.generator_variable = [];
        this.loss_variable = [];
        this.parameter_variable = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Optimizer();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.solver = nnabla.Solver.decode(reader, reader.uint32());
                    break;
                case 40:
                    message.update_interval = reader.int64();
                    break;
                case 50:
                    message.data_variable.push(nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 60:
                    message.generator_variable.push(nnabla.GeneratorVariable.decode(reader, reader.uint32()));
                    break;
                case 70:
                    message.loss_variable.push(nnabla.LossVariable.decode(reader, reader.uint32()));
                    break;
                case 80:
                    message.parameter_variable.push(nnabla.ParameterVariable.decode(reader, reader.uint32()));
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
        const message = new nnabla.Optimizer();
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
                    message.solver = nnabla.Solver.decodeText(reader);
                    break;
                case "update_interval":
                    message.update_interval = reader.int64();
                    break;
                case "data_variable":
                    message.data_variable.push(nnabla.DataVariable.decodeText(reader));
                    break;
                case "generator_variable":
                    message.generator_variable.push(nnabla.GeneratorVariable.decodeText(reader));
                    break;
                case "loss_variable":
                    message.loss_variable.push(nnabla.LossVariable.decodeText(reader));
                    break;
                case "parameter_variable":
                    message.parameter_variable.push(nnabla.ParameterVariable.decodeText(reader));
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

nnabla.Optimizer.prototype.name = "";
nnabla.Optimizer.prototype.order = 0n;
nnabla.Optimizer.prototype.network_name = "";
nnabla.Optimizer.prototype.solver = null;
nnabla.Optimizer.prototype.update_interval = 0n;
nnabla.Optimizer.prototype.start_iter = 0n;
nnabla.Optimizer.prototype.end_iter = 0n;

nnabla.SolverStateParameter = class SolverStateParameter {

    constructor() {
        this.data = [];
    }

    static decode(reader, length) {
        const message = new nnabla.SolverStateParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 20:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.SolverStateParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.SolverStateParameter.prototype.shape = null;

nnabla.SolverState = class SolverState {

    constructor() {
        this.state_parameter = {};
    }

    static decode(reader, length) {
        const message = new nnabla.SolverState();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.t = reader.uint32();
                    break;
                case 2:
                    reader.entry(message.state_parameter, () => reader.string(), () => nnabla.SolverStateParameter.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.SolverState();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "t":
                    message.t = reader.uint32();
                    break;
                case "state_parameter":
                    reader.entry(message.state_parameter, () => reader.string(), () => nnabla.SolverStateParameter.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.SolverState.prototype.t = 0;

nnabla.Solver = class Solver {

    constructor() {
        this.states = {};
    }

    get parameter() {
        nnabla.Solver.parameterSet = nnabla.Solver.parameterSet || new Set(["sgd_param", "sgdw_param", "momentum_param", "lars_param", "nesterov_param", "adadelta_param", "adagrad_param", "adabelief_param", "rmsprop_param", "rmsprop_graves_param", "adam_param", "adamw_param", "adabound_param", "adamax_param", "amsgrad_param", "amsbound_param", "lamb_param", "lion_param"]);
        return Object.keys(this).find((key) => nnabla.Solver.parameterSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new nnabla.Solver();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.type = reader.string();
                    break;
                case 10:
                    message.context = nnabla.Context.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.weight_decay = reader.float();
                    break;
                case 40:
                    reader.entry(message.states, () => reader.string(), () => nnabla.SolverState.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.sgd_param = nnabla.SgdParameter.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.sgdw_param = nnabla.SgdWParameter.decode(reader, reader.uint32());
                    break;
                case 102:
                    message.momentum_param = nnabla.MomentumParameter.decode(reader, reader.uint32());
                    break;
                case 103:
                    message.lars_param = nnabla.LarsParameter.decode(reader, reader.uint32());
                    break;
                case 104:
                    message.nesterov_param = nnabla.NesterovParameter.decode(reader, reader.uint32());
                    break;
                case 105:
                    message.adadelta_param = nnabla.AdadeltaParameter.decode(reader, reader.uint32());
                    break;
                case 106:
                    message.adagrad_param = nnabla.AdagradParameter.decode(reader, reader.uint32());
                    break;
                case 107:
                    message.adabelief_param = nnabla.AdaBeliefParameter.decode(reader, reader.uint32());
                    break;
                case 108:
                    message.rmsprop_param = nnabla.RMSpropParameter.decode(reader, reader.uint32());
                    break;
                case 109:
                    message.rmsprop_graves_param = nnabla.RMSpropGravesParameter.decode(reader, reader.uint32());
                    break;
                case 110:
                    message.adam_param = nnabla.AdamParameter.decode(reader, reader.uint32());
                    break;
                case 111:
                    message.adamw_param = nnabla.AdamWParameter.decode(reader, reader.uint32());
                    break;
                case 112:
                    message.adabound_param = nnabla.AdaBoundParameter.decode(reader, reader.uint32());
                    break;
                case 113:
                    message.adamax_param = nnabla.AdamaxParameter.decode(reader, reader.uint32());
                    break;
                case 114:
                    message.amsgrad_param = nnabla.AMSGRADParameter.decode(reader, reader.uint32());
                    break;
                case 115:
                    message.amsbound_param = nnabla.AMSBoundParameter.decode(reader, reader.uint32());
                    break;
                case 116:
                    message.lamb_param = nnabla.LambParameter.decode(reader, reader.uint32());
                    break;
                case 117:
                    message.lion_param = nnabla.LionParameter.decode(reader, reader.uint32());
                    break;
                case 200:
                    message.lr_scheduler_type = reader.string();
                    break;
                case 210:
                    message.polynomial_scheduler_param = nnabla.PolynomialSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 211:
                    message.cosine_scheduler_param = nnabla.CosineSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 212:
                    message.exponential_scheduler_param = nnabla.ExponentialSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 213:
                    message.step_scheduler_param = nnabla.StepSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 299:
                    message.custom_scheduler_param = nnabla.CustomSchedulerParameter.decode(reader, reader.uint32());
                    break;
                case 300:
                    message.lr_warmup_scheduler_type = reader.string();
                    break;
                case 310:
                    message.linear_warmup_scheduler_param = nnabla.LinearWarmupSchedulerParameter.decode(reader, reader.uint32());
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
        const message = new nnabla.Solver();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "context":
                    message.context = nnabla.Context.decodeText(reader);
                    break;
                case "weight_decay":
                    message.weight_decay = reader.float();
                    break;
                case "states":
                    reader.entry(message.states, () => reader.string(), () => nnabla.SolverState.decodeText(reader));
                    break;
                case "sgd_param":
                    message.sgd_param = nnabla.SgdParameter.decodeText(reader);
                    break;
                case "sgdw_param":
                    message.sgdw_param = nnabla.SgdWParameter.decodeText(reader);
                    break;
                case "momentum_param":
                    message.momentum_param = nnabla.MomentumParameter.decodeText(reader);
                    break;
                case "lars_param":
                    message.lars_param = nnabla.LarsParameter.decodeText(reader);
                    break;
                case "nesterov_param":
                    message.nesterov_param = nnabla.NesterovParameter.decodeText(reader);
                    break;
                case "adadelta_param":
                    message.adadelta_param = nnabla.AdadeltaParameter.decodeText(reader);
                    break;
                case "adagrad_param":
                    message.adagrad_param = nnabla.AdagradParameter.decodeText(reader);
                    break;
                case "adabelief_param":
                    message.adabelief_param = nnabla.AdaBeliefParameter.decodeText(reader);
                    break;
                case "rmsprop_param":
                    message.rmsprop_param = nnabla.RMSpropParameter.decodeText(reader);
                    break;
                case "rmsprop_graves_param":
                    message.rmsprop_graves_param = nnabla.RMSpropGravesParameter.decodeText(reader);
                    break;
                case "adam_param":
                    message.adam_param = nnabla.AdamParameter.decodeText(reader);
                    break;
                case "adamw_param":
                    message.adamw_param = nnabla.AdamWParameter.decodeText(reader);
                    break;
                case "adabound_param":
                    message.adabound_param = nnabla.AdaBoundParameter.decodeText(reader);
                    break;
                case "adamax_param":
                    message.adamax_param = nnabla.AdamaxParameter.decodeText(reader);
                    break;
                case "amsgrad_param":
                    message.amsgrad_param = nnabla.AMSGRADParameter.decodeText(reader);
                    break;
                case "amsbound_param":
                    message.amsbound_param = nnabla.AMSBoundParameter.decodeText(reader);
                    break;
                case "lamb_param":
                    message.lamb_param = nnabla.LambParameter.decodeText(reader);
                    break;
                case "lion_param":
                    message.lion_param = nnabla.LionParameter.decodeText(reader);
                    break;
                case "lr_scheduler_type":
                    message.lr_scheduler_type = reader.string();
                    break;
                case "polynomial_scheduler_param":
                    message.polynomial_scheduler_param = nnabla.PolynomialSchedulerParameter.decodeText(reader);
                    break;
                case "cosine_scheduler_param":
                    message.cosine_scheduler_param = nnabla.CosineSchedulerParameter.decodeText(reader);
                    break;
                case "exponential_scheduler_param":
                    message.exponential_scheduler_param = nnabla.ExponentialSchedulerParameter.decodeText(reader);
                    break;
                case "step_scheduler_param":
                    message.step_scheduler_param = nnabla.StepSchedulerParameter.decodeText(reader);
                    break;
                case "custom_scheduler_param":
                    message.custom_scheduler_param = nnabla.CustomSchedulerParameter.decodeText(reader);
                    break;
                case "lr_warmup_scheduler_type":
                    message.lr_warmup_scheduler_type = reader.string();
                    break;
                case "linear_warmup_scheduler_param":
                    message.linear_warmup_scheduler_param = nnabla.LinearWarmupSchedulerParameter.decodeText(reader);
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

nnabla.Solver.prototype.type = "";
nnabla.Solver.prototype.context = null;
nnabla.Solver.prototype.weight_decay = 0;
nnabla.Solver.prototype.lr_scheduler_type = "";
nnabla.Solver.prototype.polynomial_scheduler_param = null;
nnabla.Solver.prototype.cosine_scheduler_param = null;
nnabla.Solver.prototype.exponential_scheduler_param = null;
nnabla.Solver.prototype.step_scheduler_param = null;
nnabla.Solver.prototype.custom_scheduler_param = null;
nnabla.Solver.prototype.lr_warmup_scheduler_type = "";
nnabla.Solver.prototype.linear_warmup_scheduler_param = null;
nnabla.Solver.prototype.lr_decay = 0;
nnabla.Solver.prototype.lr_decay_interval = 0n;

nnabla.SgdParameter = class SgdParameter {

    static decode(reader, length) {
        const message = new nnabla.SgdParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SgdParameter();
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

nnabla.SgdParameter.prototype.lr = 0;

nnabla.SgdWParameter = class SgdWParameter {

    static decode(reader, length) {
        const message = new nnabla.SgdWParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SgdWParameter();
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

nnabla.SgdWParameter.prototype.lr = 0;
nnabla.SgdWParameter.prototype.momentum = 0;
nnabla.SgdWParameter.prototype.wd = 0;

nnabla.MomentumParameter = class MomentumParameter {

    static decode(reader, length) {
        const message = new nnabla.MomentumParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MomentumParameter();
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

nnabla.MomentumParameter.prototype.lr = 0;
nnabla.MomentumParameter.prototype.momentum = 0;

nnabla.LarsParameter = class LarsParameter {

    static decode(reader, length) {
        const message = new nnabla.LarsParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LarsParameter();
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

nnabla.LarsParameter.prototype.lr = 0;
nnabla.LarsParameter.prototype.momentum = 0;
nnabla.LarsParameter.prototype.coefficient = 0;
nnabla.LarsParameter.prototype.eps = 0;

nnabla.NesterovParameter = class NesterovParameter {

    static decode(reader, length) {
        const message = new nnabla.NesterovParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.NesterovParameter();
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

nnabla.NesterovParameter.prototype.lr = 0;
nnabla.NesterovParameter.prototype.momentum = 0;

nnabla.AdadeltaParameter = class AdadeltaParameter {

    static decode(reader, length) {
        const message = new nnabla.AdadeltaParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdadeltaParameter();
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

nnabla.AdadeltaParameter.prototype.lr = 0;
nnabla.AdadeltaParameter.prototype.decay = 0;
nnabla.AdadeltaParameter.prototype.eps = 0;

nnabla.AdagradParameter = class AdagradParameter {

    static decode(reader, length) {
        const message = new nnabla.AdagradParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdagradParameter();
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

nnabla.AdagradParameter.prototype.lr = 0;
nnabla.AdagradParameter.prototype.eps = 0;

nnabla.AdaBeliefParameter = class AdaBeliefParameter {

    static decode(reader, length) {
        const message = new nnabla.AdaBeliefParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdaBeliefParameter();
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

nnabla.AdaBeliefParameter.prototype.alpha = 0;
nnabla.AdaBeliefParameter.prototype.beta1 = 0;
nnabla.AdaBeliefParameter.prototype.beta2 = 0;
nnabla.AdaBeliefParameter.prototype.eps = 0;
nnabla.AdaBeliefParameter.prototype.wd = 0;
nnabla.AdaBeliefParameter.prototype.amsgrad = false;
nnabla.AdaBeliefParameter.prototype.weight_decouple = false;
nnabla.AdaBeliefParameter.prototype.fixed_decay = false;
nnabla.AdaBeliefParameter.prototype.rectify = false;

nnabla.RMSpropParameter = class RMSpropParameter {

    static decode(reader, length) {
        const message = new nnabla.RMSpropParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RMSpropParameter();
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

nnabla.RMSpropParameter.prototype.lr = 0;
nnabla.RMSpropParameter.prototype.decay = 0;
nnabla.RMSpropParameter.prototype.eps = 0;

nnabla.RMSpropGravesParameter = class RMSpropGravesParameter {

    static decode(reader, length) {
        const message = new nnabla.RMSpropGravesParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RMSpropGravesParameter();
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

nnabla.RMSpropGravesParameter.prototype.lr = 0;
nnabla.RMSpropGravesParameter.prototype.decay = 0;
nnabla.RMSpropGravesParameter.prototype.momentum = 0;
nnabla.RMSpropGravesParameter.prototype.eps = 0;

nnabla.AdamParameter = class AdamParameter {

    static decode(reader, length) {
        const message = new nnabla.AdamParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdamParameter();
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

nnabla.AdamParameter.prototype.alpha = 0;
nnabla.AdamParameter.prototype.beta1 = 0;
nnabla.AdamParameter.prototype.beta2 = 0;
nnabla.AdamParameter.prototype.eps = 0;

nnabla.AdamWParameter = class AdamWParameter {

    static decode(reader, length) {
        const message = new nnabla.AdamWParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdamWParameter();
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

nnabla.AdamWParameter.prototype.alpha = 0;
nnabla.AdamWParameter.prototype.beta1 = 0;
nnabla.AdamWParameter.prototype.beta2 = 0;
nnabla.AdamWParameter.prototype.eps = 0;
nnabla.AdamWParameter.prototype.wd = 0;

nnabla.AdaBoundParameter = class AdaBoundParameter {

    static decode(reader, length) {
        const message = new nnabla.AdaBoundParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdaBoundParameter();
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

nnabla.AdaBoundParameter.prototype.alpha = 0;
nnabla.AdaBoundParameter.prototype.beta1 = 0;
nnabla.AdaBoundParameter.prototype.beta2 = 0;
nnabla.AdaBoundParameter.prototype.eps = 0;
nnabla.AdaBoundParameter.prototype.final_lr = 0;
nnabla.AdaBoundParameter.prototype.gamma = 0;

nnabla.AdamaxParameter = class AdamaxParameter {

    static decode(reader, length) {
        const message = new nnabla.AdamaxParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AdamaxParameter();
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

nnabla.AdamaxParameter.prototype.alpha = 0;
nnabla.AdamaxParameter.prototype.beta1 = 0;
nnabla.AdamaxParameter.prototype.beta2 = 0;
nnabla.AdamaxParameter.prototype.eps = 0;

nnabla.AMSGRADParameter = class AMSGRADParameter {

    static decode(reader, length) {
        const message = new nnabla.AMSGRADParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AMSGRADParameter();
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

nnabla.AMSGRADParameter.prototype.alpha = 0;
nnabla.AMSGRADParameter.prototype.beta1 = 0;
nnabla.AMSGRADParameter.prototype.beta2 = 0;
nnabla.AMSGRADParameter.prototype.eps = 0;
nnabla.AMSGRADParameter.prototype.bias_correction = false;

nnabla.AMSBoundParameter = class AMSBoundParameter {

    static decode(reader, length) {
        const message = new nnabla.AMSBoundParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AMSBoundParameter();
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

nnabla.AMSBoundParameter.prototype.alpha = 0;
nnabla.AMSBoundParameter.prototype.beta1 = 0;
nnabla.AMSBoundParameter.prototype.beta2 = 0;
nnabla.AMSBoundParameter.prototype.eps = 0;
nnabla.AMSBoundParameter.prototype.final_lr = 0;
nnabla.AMSBoundParameter.prototype.gamma = 0;
nnabla.AMSBoundParameter.prototype.bias_correction = false;

nnabla.LambParameter = class LambParameter {

    static decode(reader, length) {
        const message = new nnabla.LambParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LambParameter();
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

nnabla.LambParameter.prototype.eta = 0;
nnabla.LambParameter.prototype.beta1 = 0;
nnabla.LambParameter.prototype.beta2 = 0;
nnabla.LambParameter.prototype.gamma_l = 0;
nnabla.LambParameter.prototype.gamma_u = 0;
nnabla.LambParameter.prototype.eps = 0;
nnabla.LambParameter.prototype.bias_correction = false;

nnabla.LionParameter = class LionParameter {

    static decode(reader, length) {
        const message = new nnabla.LionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lr = reader.float();
                    break;
                case 2:
                    message.beta1 = reader.float();
                    break;
                case 3:
                    message.beta2 = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.LionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lr":
                    message.lr = reader.float();
                    break;
                case "beta1":
                    message.beta1 = reader.float();
                    break;
                case "beta2":
                    message.beta2 = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.LionParameter.prototype.lr = 0;
nnabla.LionParameter.prototype.beta1 = 0;
nnabla.LionParameter.prototype.beta2 = 0;

nnabla.PolynomialSchedulerParameter = class PolynomialSchedulerParameter {

    static decode(reader, length) {
        const message = new nnabla.PolynomialSchedulerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PolynomialSchedulerParameter();
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

nnabla.PolynomialSchedulerParameter.prototype.max_iter = 0;
nnabla.PolynomialSchedulerParameter.prototype.power = 0;

nnabla.CosineSchedulerParameter = class CosineSchedulerParameter {

    static decode(reader, length) {
        const message = new nnabla.CosineSchedulerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CosineSchedulerParameter();
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

nnabla.CosineSchedulerParameter.prototype.max_iter = 0;

nnabla.ExponentialSchedulerParameter = class ExponentialSchedulerParameter {

    static decode(reader, length) {
        const message = new nnabla.ExponentialSchedulerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ExponentialSchedulerParameter();
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

nnabla.ExponentialSchedulerParameter.prototype.gamma = 0;
nnabla.ExponentialSchedulerParameter.prototype.iter_interval = 0n;

nnabla.StepSchedulerParameter = class StepSchedulerParameter {

    constructor() {
        this.iter_steps = [];
    }

    static decode(reader, length) {
        const message = new nnabla.StepSchedulerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.StepSchedulerParameter();
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

nnabla.StepSchedulerParameter.prototype.gamma = 0;

nnabla.CustomSchedulerParameter = class CustomSchedulerParameter {

    constructor() {
        this.data_variable = [];
        this.output_variable = [];
    }

    static decode(reader, length) {
        const message = new nnabla.CustomSchedulerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.data_variable.push(nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 80:
                    message.output_variable.push(nnabla.OutputVariable.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.CustomSchedulerParameter();
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
                    message.data_variable.push(nnabla.DataVariable.decodeText(reader));
                    break;
                case "output_variable":
                    message.output_variable.push(nnabla.OutputVariable.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.CustomSchedulerParameter.prototype.max_iter = 0;
nnabla.CustomSchedulerParameter.prototype.network_name = "";

nnabla.LinearWarmupSchedulerParameter = class LinearWarmupSchedulerParameter {

    static decode(reader, length) {
        const message = new nnabla.LinearWarmupSchedulerParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LinearWarmupSchedulerParameter();
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

nnabla.LinearWarmupSchedulerParameter.prototype.warmup_iter = 0n;

nnabla.DataVariable = class DataVariable {

    static decode(reader, length) {
        const message = new nnabla.DataVariable();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.DataVariable();
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

nnabla.DataVariable.prototype.variable_name = "";
nnabla.DataVariable.prototype.data_name = "";

nnabla.GeneratorVariable = class GeneratorVariable {

    static decode(reader, length) {
        const message = new nnabla.GeneratorVariable();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.GeneratorVariable();
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

nnabla.GeneratorVariable.prototype.variable_name = "";
nnabla.GeneratorVariable.prototype.type = "";
nnabla.GeneratorVariable.prototype.multiplier = 0;

nnabla.LossVariable = class LossVariable {

    static decode(reader, length) {
        const message = new nnabla.LossVariable();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LossVariable();
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

nnabla.LossVariable.prototype.variable_name = "";

nnabla.ParameterVariable = class ParameterVariable {

    static decode(reader, length) {
        const message = new nnabla.ParameterVariable();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ParameterVariable();
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

nnabla.ParameterVariable.prototype.variable_name = "";
nnabla.ParameterVariable.prototype.learning_rate_multiplier = 0;

nnabla.Monitor = class Monitor {

    constructor() {
        this.dataset_name = [];
        this.data_variable = [];
        this.generator_variable = [];
        this.monitor_variable = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Monitor();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.data_variable.push(nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 60:
                    message.generator_variable.push(nnabla.GeneratorVariable.decode(reader, reader.uint32()));
                    break;
                case 70:
                    message.monitor_variable.push(nnabla.MonitorVariable.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.Monitor();
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
                    message.data_variable.push(nnabla.DataVariable.decodeText(reader));
                    break;
                case "generator_variable":
                    message.generator_variable.push(nnabla.GeneratorVariable.decodeText(reader));
                    break;
                case "monitor_variable":
                    message.monitor_variable.push(nnabla.MonitorVariable.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.Monitor.prototype.name = "";
nnabla.Monitor.prototype.network_name = "";

nnabla.MonitorVariable = class MonitorVariable {

    static decode(reader, length) {
        const message = new nnabla.MonitorVariable();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MonitorVariable();
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

nnabla.MonitorVariable.prototype.variable_name = "";
nnabla.MonitorVariable.prototype.type = "";
nnabla.MonitorVariable.prototype.data_name = "";
nnabla.MonitorVariable.prototype.multiplier = 0;

nnabla.Executor = class Executor {

    constructor() {
        this.data_variable = [];
        this.generator_variable = [];
        this.loss_variable = [];
        this.output_variable = [];
        this.parameter_variable = [];
    }

    static decode(reader, length) {
        const message = new nnabla.Executor();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.data_variable.push(nnabla.DataVariable.decode(reader, reader.uint32()));
                    break;
                case 60:
                    message.generator_variable.push(nnabla.GeneratorVariable.decode(reader, reader.uint32()));
                    break;
                case 70:
                    message.loss_variable.push(nnabla.LossVariable.decode(reader, reader.uint32()));
                    break;
                case 80:
                    message.output_variable.push(nnabla.OutputVariable.decode(reader, reader.uint32()));
                    break;
                case 90:
                    message.parameter_variable.push(nnabla.ParameterVariable.decode(reader, reader.uint32()));
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
        const message = new nnabla.Executor();
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
                    message.data_variable.push(nnabla.DataVariable.decodeText(reader));
                    break;
                case "generator_variable":
                    message.generator_variable.push(nnabla.GeneratorVariable.decodeText(reader));
                    break;
                case "loss_variable":
                    message.loss_variable.push(nnabla.LossVariable.decodeText(reader));
                    break;
                case "output_variable":
                    message.output_variable.push(nnabla.OutputVariable.decodeText(reader));
                    break;
                case "parameter_variable":
                    message.parameter_variable.push(nnabla.ParameterVariable.decodeText(reader));
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

nnabla.Executor.prototype.name = "";
nnabla.Executor.prototype.network_name = "";
nnabla.Executor.prototype.num_evaluations = 0n;
nnabla.Executor.prototype.repeat_evaluation_type = "";
nnabla.Executor.prototype.need_back_propagation = false;
nnabla.Executor.prototype.no_image_normalization = false;

nnabla.OutputVariable = class OutputVariable {

    static decode(reader, length) {
        const message = new nnabla.OutputVariable();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.OutputVariable();
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

nnabla.OutputVariable.prototype.variable_name = "";
nnabla.OutputVariable.prototype.type = "";
nnabla.OutputVariable.prototype.data_name = "";

nnabla.Function = class Function {

    constructor() {
        this.repeat_id = [];
        this.input = [];
        this.output = [];
    }

    get parameter() {
        nnabla.Function.parameterSet = nnabla.Function.parameterSet || new Set(["affine_param", "rnn_param", "lstm_param", "gru_param", "convolution_param", "fused_convolution_param", "depthwise_convolution_param", "deconvolution_param", "depthwise_deconvolution_param", "deformable_convolution_param", "max_pooling_param", "average_pooling_param", "sum_pooling_param", "unpooling_param", "roi_align_param", "relu_param", "leaky_relu_param", "softmax_param", "log_softmax_param", "elu_param", "selu_param", "crelu_param", "celu_param", "prelu_param", "softplus_param", "fused_batch_normalization_param", "batch_normalization_param", "group_normalization_param", "instance_normalization_param", "layer_normalization_param", "norm_normalization_param", "sync_batch_normalization_param", "tensor_normalization_param", "weight_normalization_param", "weight_standardization_param", "spectral_norm_param", "mean_subtraction_param", "clip_grad_by_norm_param", "sum_param", "cumsum_param", "mean_param", "max_param", "min_param", "norm_param", "prod_param", "cumprod_param", "add2_param", "bc_add2_param", "sub2_param", "mul2_param", "div2_param", "pow2_param", "add_scalar_param", "mul_scalar_param", "pow_scalar_param", "r_sub_scalar_param", "r_div_scalar_param", "r_pow_scalar_param", "sign_param", "minimum_scalar_param", "maximum_scalar_param", "searchsorted_param", "logical_and_scalar_param", "logical_or_scalar_param", "logical_xor_scalar_param", "equal_scalar_param", "not_equal_scalar_param", "greater_equal_scalar_param", "greater_scalar_param", "less_equal_scalar_param", "less_scalar_param", "reset_nan_param", "reset_inf_param", "constant_param", "arange_param", "linspace_param", "batch_matmul_param", "round_param", "ceil_param", "floor_param", "concatenate_param", "split_param", "stack_param", "slice_param", "pad_param", "transpose_param", "broadcast_param", "broadcast_to_param", "tile_param", "one_hot_param", "flip_param", "shift_param", "sort_param", "reshape_param", "shape_param", "trilu_param", "meshgrid_param", "batch_cholesky_param", "gather_param", "scatter_nd_param", "scatter_add_param", "bool_fill_param", "pack_padded_sequence_param", "pad_packed_sequence_param", "interpolate_param", "onnx_resize_param", "fft_param", "ifft_param", "stft_param", "istft_param", "dropout_param", "top_k_data_param", "top_k_grad_param", "rand_param", "randint_param", "randn_param", "rand_binomial_param", "rand_beta_param", "rand_gamma_param", "random_choice_param", "random_crop_param", "random_flip_param", "random_shift_param", "random_erase_param", "image_augmentation_param", "softmax_cross_entropy_param", "categorical_cross_entropy_param", "huber_loss_param", "epsilon_insensitive_loss_param", "kl_multinomial_param", "affine_grid_param", "warp_by_grid_param", "binary_connect_affine_param", "binary_connect_convolution_param", "binary_weight_affine_param", "binary_weight_convolution_param", "inq_affine_param", "inq_convolution_param", "fixed_point_quantize_param", "min_max_quantize_param", "pow2_quantize_param", "prune_param", "quantize_linear_param", "top_n_error_param", "confusion_matrix_param", "vat_noise_param", "sink_param", "nms_detection2d_param", "onnx_non_max_suppression_param", "max_pooling_backward_param", "patch_correlation_param", "unique_param", "eye_like_param", "mod2_param", "bit_shift_param", "einsum_param"]);
        return Object.keys(this).find((key) => nnabla.Function.parameterSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new nnabla.Function();
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
                    message.repeat_id.push(reader.string());
                    break;
                case 10:
                    message.context = nnabla.Context.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.input.push(reader.string());
                    break;
                case 30:
                    message.output.push(reader.string());
                    break;
                case 1001:
                    message.affine_param = nnabla.AffineParameter.decode(reader, reader.uint32());
                    break;
                case 1002:
                    message.rnn_param = nnabla.RNNParameter.decode(reader, reader.uint32());
                    break;
                case 1003:
                    message.lstm_param = nnabla.LSTMParameter.decode(reader, reader.uint32());
                    break;
                case 1004:
                    message.gru_param = nnabla.GRUParameter.decode(reader, reader.uint32());
                    break;
                case 1005:
                    message.convolution_param = nnabla.ConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1006:
                    message.fused_convolution_param = nnabla.FusedConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1007:
                    message.depthwise_convolution_param = nnabla.DepthwiseConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1008:
                    message.deconvolution_param = nnabla.DeconvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1009:
                    message.depthwise_deconvolution_param = nnabla.DepthwiseDeconvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1010:
                    message.deformable_convolution_param = nnabla.DeformableConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1012:
                    message.max_pooling_param = nnabla.MaxPoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1013:
                    message.average_pooling_param = nnabla.AveragePoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1015:
                    message.sum_pooling_param = nnabla.SumPoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1016:
                    message.unpooling_param = nnabla.UnpoolingParameter.decode(reader, reader.uint32());
                    break;
                case 1018:
                    message.roi_align_param = nnabla.RoiAlignParameter.decode(reader, reader.uint32());
                    break;
                case 1022:
                    message.relu_param = nnabla.ReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1023:
                    message.leaky_relu_param = nnabla.LeakyReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1024:
                    message.softmax_param = nnabla.SoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 1025:
                    message.log_softmax_param = nnabla.LogSoftmaxParameter.decode(reader, reader.uint32());
                    break;
                case 1026:
                    message.elu_param = nnabla.ELUParameter.decode(reader, reader.uint32());
                    break;
                case 1027:
                    message.selu_param = nnabla.SELUParameter.decode(reader, reader.uint32());
                    break;
                case 1028:
                    message.crelu_param = nnabla.CReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1029:
                    message.celu_param = nnabla.CELUParameter.decode(reader, reader.uint32());
                    break;
                case 1030:
                    message.prelu_param = nnabla.PReLUParameter.decode(reader, reader.uint32());
                    break;
                case 1037:
                    message.softplus_param = nnabla.SoftPlusParameter.decode(reader, reader.uint32());
                    break;
                case 1041:
                    message.fused_batch_normalization_param = nnabla.FusedBatchNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1042:
                    message.batch_normalization_param = nnabla.BatchNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1043:
                    message.group_normalization_param = nnabla.GroupNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1044:
                    message.instance_normalization_param = nnabla.InstanceNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1045:
                    message.layer_normalization_param = nnabla.LayerNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1046:
                    message.norm_normalization_param = nnabla.NormNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1047:
                    message.sync_batch_normalization_param = nnabla.SyncBatchNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1048:
                    message.tensor_normalization_param = nnabla.TensorNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1049:
                    message.weight_normalization_param = nnabla.WeightNormalizationParameter.decode(reader, reader.uint32());
                    break;
                case 1050:
                    message.weight_standardization_param = nnabla.WeightStandardizationParameter.decode(reader, reader.uint32());
                    break;
                case 1051:
                    message.spectral_norm_param = nnabla.SpectralNormParameter.decode(reader, reader.uint32());
                    break;
                case 1052:
                    message.mean_subtraction_param = nnabla.MeanSubtractionParameter.decode(reader, reader.uint32());
                    break;
                case 1054:
                    message.clip_grad_by_norm_param = nnabla.ClipGradByNormParameter.decode(reader, reader.uint32());
                    break;
                case 1055:
                    message.sum_param = nnabla.SumParameter.decode(reader, reader.uint32());
                    break;
                case 1056:
                    message.cumsum_param = nnabla.CumSumParameter.decode(reader, reader.uint32());
                    break;
                case 1057:
                    message.mean_param = nnabla.MeanParameter.decode(reader, reader.uint32());
                    break;
                case 1058:
                    message.max_param = nnabla.MaxParameter.decode(reader, reader.uint32());
                    break;
                case 1059:
                    message.min_param = nnabla.MinParameter.decode(reader, reader.uint32());
                    break;
                case 1060:
                    message.norm_param = nnabla.NormParameter.decode(reader, reader.uint32());
                    break;
                case 1061:
                    message.prod_param = nnabla.ProdParameter.decode(reader, reader.uint32());
                    break;
                case 1062:
                    message.cumprod_param = nnabla.CumProdParameter.decode(reader, reader.uint32());
                    break;
                case 1065:
                    message.add2_param = nnabla.Add2Parameter.decode(reader, reader.uint32());
                    break;
                case 1067:
                    message.bc_add2_param = nnabla.BcAdd2Parameter.decode(reader, reader.uint32());
                    break;
                case 1068:
                    message.sub2_param = nnabla.Sub2Parameter.decode(reader, reader.uint32());
                    break;
                case 1069:
                    message.mul2_param = nnabla.Mul2Parameter.decode(reader, reader.uint32());
                    break;
                case 1071:
                    message.div2_param = nnabla.Div2Parameter.decode(reader, reader.uint32());
                    break;
                case 1072:
                    message.pow2_param = nnabla.Pow2Parameter.decode(reader, reader.uint32());
                    break;
                case 1073:
                    message.add_scalar_param = nnabla.AddScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1074:
                    message.mul_scalar_param = nnabla.MulScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1075:
                    message.pow_scalar_param = nnabla.PowScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1076:
                    message.r_sub_scalar_param = nnabla.RSubScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1077:
                    message.r_div_scalar_param = nnabla.RDivScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1078:
                    message.r_pow_scalar_param = nnabla.RPowScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1079:
                    message.sign_param = nnabla.SignParameter.decode(reader, reader.uint32());
                    break;
                case 1082:
                    message.minimum_scalar_param = nnabla.MinimumScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1083:
                    message.maximum_scalar_param = nnabla.MaximumScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1093:
                    message.searchsorted_param = nnabla.SearchSortedParameter.decode(reader, reader.uint32());
                    break;
                case 1094:
                    message.logical_and_scalar_param = nnabla.LogicalAndScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1095:
                    message.logical_or_scalar_param = nnabla.LogicalOrScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1096:
                    message.logical_xor_scalar_param = nnabla.LogicalXorScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1097:
                    message.equal_scalar_param = nnabla.EqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1098:
                    message.not_equal_scalar_param = nnabla.NotEqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1099:
                    message.greater_equal_scalar_param = nnabla.GreaterEqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1100:
                    message.greater_scalar_param = nnabla.GreaterScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1101:
                    message.less_equal_scalar_param = nnabla.LessEqualScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1102:
                    message.less_scalar_param = nnabla.LessScalarParameter.decode(reader, reader.uint32());
                    break;
                case 1106:
                    message.reset_nan_param = nnabla.ResetNaNParameter.decode(reader, reader.uint32());
                    break;
                case 1107:
                    message.reset_inf_param = nnabla.ResetInfParameter.decode(reader, reader.uint32());
                    break;
                case 1109:
                    message.constant_param = nnabla.ConstantParameter.decode(reader, reader.uint32());
                    break;
                case 1110:
                    message.arange_param = nnabla.ArangeParameter.decode(reader, reader.uint32());
                    break;
                case 1111:
                    message.linspace_param = nnabla.LinspaceParameter.decode(reader, reader.uint32());
                    break;
                case 1116:
                    message.batch_matmul_param = nnabla.BatchMatmulParameter.decode(reader, reader.uint32());
                    break;
                case 1117:
                    message.round_param = nnabla.RoundParameter.decode(reader, reader.uint32());
                    break;
                case 1118:
                    message.ceil_param = nnabla.CeilParameter.decode(reader, reader.uint32());
                    break;
                case 1119:
                    message.floor_param = nnabla.FloorParameter.decode(reader, reader.uint32());
                    break;
                case 1133:
                    message.concatenate_param = nnabla.ConcatenateParameter.decode(reader, reader.uint32());
                    break;
                case 1134:
                    message.split_param = nnabla.SplitParameter.decode(reader, reader.uint32());
                    break;
                case 1135:
                    message.stack_param = nnabla.StackParameter.decode(reader, reader.uint32());
                    break;
                case 1136:
                    message.slice_param = nnabla.SliceParameter.decode(reader, reader.uint32());
                    break;
                case 1137:
                    message.pad_param = nnabla.PadParameter.decode(reader, reader.uint32());
                    break;
                case 1138:
                    message.transpose_param = nnabla.TransposeParameter.decode(reader, reader.uint32());
                    break;
                case 1139:
                    message.broadcast_param = nnabla.BroadcastParameter.decode(reader, reader.uint32());
                    break;
                case 1140:
                    message.broadcast_to_param = nnabla.BroadcastToParameter.decode(reader, reader.uint32());
                    break;
                case 1141:
                    message.tile_param = nnabla.TileParameter.decode(reader, reader.uint32());
                    break;
                case 1142:
                    message.one_hot_param = nnabla.OneHotParameter.decode(reader, reader.uint32());
                    break;
                case 1143:
                    message.flip_param = nnabla.FlipParameter.decode(reader, reader.uint32());
                    break;
                case 1144:
                    message.shift_param = nnabla.ShiftParameter.decode(reader, reader.uint32());
                    break;
                case 1145:
                    message.sort_param = nnabla.SortParameter.decode(reader, reader.uint32());
                    break;
                case 1146:
                    message.reshape_param = nnabla.ReshapeParameter.decode(reader, reader.uint32());
                    break;
                case 1147:
                    message.shape_param = nnabla.ShapeParameter.decode(reader, reader.uint32());
                    break;
                case 1150:
                    message.trilu_param = nnabla.TriluParameter.decode(reader, reader.uint32());
                    break;
                case 1151:
                    message.meshgrid_param = nnabla.MeshgridParameter.decode(reader, reader.uint32());
                    break;
                case 1155:
                    message.batch_cholesky_param = nnabla.BatchCholeskyParameter.decode(reader, reader.uint32());
                    break;
                case 1157:
                    message.gather_param = nnabla.GatherParameter.decode(reader, reader.uint32());
                    break;
                case 1160:
                    message.scatter_nd_param = nnabla.ScatterNdParameter.decode(reader, reader.uint32());
                    break;
                case 1161:
                    message.scatter_add_param = nnabla.ScatterAddParameter.decode(reader, reader.uint32());
                    break;
                case 1163:
                    message.bool_fill_param = nnabla.BoolFillParameter.decode(reader, reader.uint32());
                    break;
                case 1164:
                    message.pack_padded_sequence_param = nnabla.PackPaddedSequenceParameter.decode(reader, reader.uint32());
                    break;
                case 1165:
                    message.pad_packed_sequence_param = nnabla.PadPackedSequenceParameter.decode(reader, reader.uint32());
                    break;
                case 1167:
                    message.interpolate_param = nnabla.InterpolateParameter.decode(reader, reader.uint32());
                    break;
                case 1168:
                    message.onnx_resize_param = nnabla.ONNXResizeParameter.decode(reader, reader.uint32());
                    break;
                case 1169:
                    message.fft_param = nnabla.FFTParameter.decode(reader, reader.uint32());
                    break;
                case 1170:
                    message.ifft_param = nnabla.IFFTParameter.decode(reader, reader.uint32());
                    break;
                case 1171:
                    message.stft_param = nnabla.STFTParameter.decode(reader, reader.uint32());
                    break;
                case 1172:
                    message.istft_param = nnabla.ISTFTParameter.decode(reader, reader.uint32());
                    break;
                case 1173:
                    message.dropout_param = nnabla.DropoutParameter.decode(reader, reader.uint32());
                    break;
                case 1174:
                    message.top_k_data_param = nnabla.TopKDataParameter.decode(reader, reader.uint32());
                    break;
                case 1175:
                    message.top_k_grad_param = nnabla.TopKGradParameter.decode(reader, reader.uint32());
                    break;
                case 1176:
                    message.rand_param = nnabla.RandParameter.decode(reader, reader.uint32());
                    break;
                case 1177:
                    message.randint_param = nnabla.RandintParameter.decode(reader, reader.uint32());
                    break;
                case 1178:
                    message.randn_param = nnabla.RandnParameter.decode(reader, reader.uint32());
                    break;
                case 1179:
                    message.rand_binomial_param = nnabla.RandBinomialParameter.decode(reader, reader.uint32());
                    break;
                case 1180:
                    message.rand_beta_param = nnabla.RandBetaParameter.decode(reader, reader.uint32());
                    break;
                case 1181:
                    message.rand_gamma_param = nnabla.RandGammaParameter.decode(reader, reader.uint32());
                    break;
                case 1182:
                    message.random_choice_param = nnabla.RandomChoiceParameter.decode(reader, reader.uint32());
                    break;
                case 1183:
                    message.random_crop_param = nnabla.RandomCropParameter.decode(reader, reader.uint32());
                    break;
                case 1184:
                    message.random_flip_param = nnabla.RandomFlipParameter.decode(reader, reader.uint32());
                    break;
                case 1185:
                    message.random_shift_param = nnabla.RandomShiftParameter.decode(reader, reader.uint32());
                    break;
                case 1186:
                    message.random_erase_param = nnabla.RandomEraseParameter.decode(reader, reader.uint32());
                    break;
                case 1187:
                    message.image_augmentation_param = nnabla.ImageAugmentationParameter.decode(reader, reader.uint32());
                    break;
                case 1190:
                    message.softmax_cross_entropy_param = nnabla.SoftmaxCrossEntropyParameter.decode(reader, reader.uint32());
                    break;
                case 1191:
                    message.categorical_cross_entropy_param = nnabla.CategoricalCrossEntropyParameter.decode(reader, reader.uint32());
                    break;
                case 1194:
                    message.huber_loss_param = nnabla.HuberLossParameter.decode(reader, reader.uint32());
                    break;
                case 1195:
                    message.epsilon_insensitive_loss_param = nnabla.EpsilonInsensitiveLossParameter.decode(reader, reader.uint32());
                    break;
                case 1196:
                    message.kl_multinomial_param = nnabla.KLMultinomialParameter.decode(reader, reader.uint32());
                    break;
                case 1197:
                    message.affine_grid_param = nnabla.AffineGridParameter.decode(reader, reader.uint32());
                    break;
                case 1198:
                    message.warp_by_grid_param = nnabla.WarpByGridParameter.decode(reader, reader.uint32());
                    break;
                case 1202:
                    message.binary_connect_affine_param = nnabla.BinaryConnectAffineParameter.decode(reader, reader.uint32());
                    break;
                case 1203:
                    message.binary_connect_convolution_param = nnabla.BinaryConnectConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1204:
                    message.binary_weight_affine_param = nnabla.BinaryWeightAffineParameter.decode(reader, reader.uint32());
                    break;
                case 1205:
                    message.binary_weight_convolution_param = nnabla.BinaryWeightConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1206:
                    message.inq_affine_param = nnabla.INQAffineParameter.decode(reader, reader.uint32());
                    break;
                case 1207:
                    message.inq_convolution_param = nnabla.INQConvolutionParameter.decode(reader, reader.uint32());
                    break;
                case 1208:
                    message.fixed_point_quantize_param = nnabla.FixedPointQuantizeParameter.decode(reader, reader.uint32());
                    break;
                case 1209:
                    message.min_max_quantize_param = nnabla.MinMaxQuantizeParameter.decode(reader, reader.uint32());
                    break;
                case 1210:
                    message.pow2_quantize_param = nnabla.Pow2QuantizeParameter.decode(reader, reader.uint32());
                    break;
                case 1211:
                    message.prune_param = nnabla.PruneParameter.decode(reader, reader.uint32());
                    break;
                case 1212:
                    message.quantize_linear_param = nnabla.QuantizeLinearParameter.decode(reader, reader.uint32());
                    break;
                case 1214:
                    message.top_n_error_param = nnabla.TopNErrorParameter.decode(reader, reader.uint32());
                    break;
                case 1216:
                    message.confusion_matrix_param = nnabla.ConfusionMatrixParameter.decode(reader, reader.uint32());
                    break;
                case 1217:
                    message.vat_noise_param = nnabla.VATNoiseParameter.decode(reader, reader.uint32());
                    break;
                case 1219:
                    message.sink_param = nnabla.SinkParameter.decode(reader, reader.uint32());
                    break;
                case 1220:
                    message.nms_detection2d_param = nnabla.NmsDetection2dParameter.decode(reader, reader.uint32());
                    break;
                case 1221:
                    message.onnx_non_max_suppression_param = nnabla.ONNXNonMaxSuppressionParameter.decode(reader, reader.uint32());
                    break;
                case 1222:
                    message.max_pooling_backward_param = nnabla.MaxPoolingBackwardParameter.decode(reader, reader.uint32());
                    break;
                case 1223:
                    message.patch_correlation_param = nnabla.PatchCorrelationParameter.decode(reader, reader.uint32());
                    break;
                case 1224:
                    message.unique_param = nnabla.UniqueParameter.decode(reader, reader.uint32());
                    break;
                case 1225:
                    message.eye_like_param = nnabla.EyeLikeParameter.decode(reader, reader.uint32());
                    break;
                case 1226:
                    message.mod2_param = nnabla.Mod2Parameter.decode(reader, reader.uint32());
                    break;
                case 1227:
                    message.bit_shift_param = nnabla.BitShiftParameter.decode(reader, reader.uint32());
                    break;
                case 1228:
                    message.einsum_param = nnabla.EinsumParameter.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.repeat_param = nnabla.RepeatParameter.decode(reader, reader.uint32());
                    break;
                case 101:
                    message.recurrent_param = nnabla.RecurrentParameter.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.Function();
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
                    message.context = nnabla.Context.decodeText(reader);
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                case "affine_param":
                    message.affine_param = nnabla.AffineParameter.decodeText(reader);
                    break;
                case "rnn_param":
                    message.rnn_param = nnabla.RNNParameter.decodeText(reader);
                    break;
                case "lstm_param":
                    message.lstm_param = nnabla.LSTMParameter.decodeText(reader);
                    break;
                case "gru_param":
                    message.gru_param = nnabla.GRUParameter.decodeText(reader);
                    break;
                case "convolution_param":
                    message.convolution_param = nnabla.ConvolutionParameter.decodeText(reader);
                    break;
                case "fused_convolution_param":
                    message.fused_convolution_param = nnabla.FusedConvolutionParameter.decodeText(reader);
                    break;
                case "depthwise_convolution_param":
                    message.depthwise_convolution_param = nnabla.DepthwiseConvolutionParameter.decodeText(reader);
                    break;
                case "deconvolution_param":
                    message.deconvolution_param = nnabla.DeconvolutionParameter.decodeText(reader);
                    break;
                case "depthwise_deconvolution_param":
                    message.depthwise_deconvolution_param = nnabla.DepthwiseDeconvolutionParameter.decodeText(reader);
                    break;
                case "deformable_convolution_param":
                    message.deformable_convolution_param = nnabla.DeformableConvolutionParameter.decodeText(reader);
                    break;
                case "max_pooling_param":
                    message.max_pooling_param = nnabla.MaxPoolingParameter.decodeText(reader);
                    break;
                case "average_pooling_param":
                    message.average_pooling_param = nnabla.AveragePoolingParameter.decodeText(reader);
                    break;
                case "sum_pooling_param":
                    message.sum_pooling_param = nnabla.SumPoolingParameter.decodeText(reader);
                    break;
                case "unpooling_param":
                    message.unpooling_param = nnabla.UnpoolingParameter.decodeText(reader);
                    break;
                case "roi_align_param":
                    message.roi_align_param = nnabla.RoiAlignParameter.decodeText(reader);
                    break;
                case "relu_param":
                    message.relu_param = nnabla.ReLUParameter.decodeText(reader);
                    break;
                case "leaky_relu_param":
                    message.leaky_relu_param = nnabla.LeakyReLUParameter.decodeText(reader);
                    break;
                case "softmax_param":
                    message.softmax_param = nnabla.SoftmaxParameter.decodeText(reader);
                    break;
                case "log_softmax_param":
                    message.log_softmax_param = nnabla.LogSoftmaxParameter.decodeText(reader);
                    break;
                case "elu_param":
                    message.elu_param = nnabla.ELUParameter.decodeText(reader);
                    break;
                case "selu_param":
                    message.selu_param = nnabla.SELUParameter.decodeText(reader);
                    break;
                case "crelu_param":
                    message.crelu_param = nnabla.CReLUParameter.decodeText(reader);
                    break;
                case "celu_param":
                    message.celu_param = nnabla.CELUParameter.decodeText(reader);
                    break;
                case "prelu_param":
                    message.prelu_param = nnabla.PReLUParameter.decodeText(reader);
                    break;
                case "softplus_param":
                    message.softplus_param = nnabla.SoftPlusParameter.decodeText(reader);
                    break;
                case "fused_batch_normalization_param":
                    message.fused_batch_normalization_param = nnabla.FusedBatchNormalizationParameter.decodeText(reader);
                    break;
                case "batch_normalization_param":
                    message.batch_normalization_param = nnabla.BatchNormalizationParameter.decodeText(reader);
                    break;
                case "group_normalization_param":
                    message.group_normalization_param = nnabla.GroupNormalizationParameter.decodeText(reader);
                    break;
                case "instance_normalization_param":
                    message.instance_normalization_param = nnabla.InstanceNormalizationParameter.decodeText(reader);
                    break;
                case "layer_normalization_param":
                    message.layer_normalization_param = nnabla.LayerNormalizationParameter.decodeText(reader);
                    break;
                case "norm_normalization_param":
                    message.norm_normalization_param = nnabla.NormNormalizationParameter.decodeText(reader);
                    break;
                case "sync_batch_normalization_param":
                    message.sync_batch_normalization_param = nnabla.SyncBatchNormalizationParameter.decodeText(reader);
                    break;
                case "tensor_normalization_param":
                    message.tensor_normalization_param = nnabla.TensorNormalizationParameter.decodeText(reader);
                    break;
                case "weight_normalization_param":
                    message.weight_normalization_param = nnabla.WeightNormalizationParameter.decodeText(reader);
                    break;
                case "weight_standardization_param":
                    message.weight_standardization_param = nnabla.WeightStandardizationParameter.decodeText(reader);
                    break;
                case "spectral_norm_param":
                    message.spectral_norm_param = nnabla.SpectralNormParameter.decodeText(reader);
                    break;
                case "mean_subtraction_param":
                    message.mean_subtraction_param = nnabla.MeanSubtractionParameter.decodeText(reader);
                    break;
                case "clip_grad_by_norm_param":
                    message.clip_grad_by_norm_param = nnabla.ClipGradByNormParameter.decodeText(reader);
                    break;
                case "sum_param":
                    message.sum_param = nnabla.SumParameter.decodeText(reader);
                    break;
                case "cumsum_param":
                    message.cumsum_param = nnabla.CumSumParameter.decodeText(reader);
                    break;
                case "mean_param":
                    message.mean_param = nnabla.MeanParameter.decodeText(reader);
                    break;
                case "max_param":
                    message.max_param = nnabla.MaxParameter.decodeText(reader);
                    break;
                case "min_param":
                    message.min_param = nnabla.MinParameter.decodeText(reader);
                    break;
                case "norm_param":
                    message.norm_param = nnabla.NormParameter.decodeText(reader);
                    break;
                case "prod_param":
                    message.prod_param = nnabla.ProdParameter.decodeText(reader);
                    break;
                case "cumprod_param":
                    message.cumprod_param = nnabla.CumProdParameter.decodeText(reader);
                    break;
                case "add2_param":
                    message.add2_param = nnabla.Add2Parameter.decodeText(reader);
                    break;
                case "bc_add2_param":
                    message.bc_add2_param = nnabla.BcAdd2Parameter.decodeText(reader);
                    break;
                case "sub2_param":
                    message.sub2_param = nnabla.Sub2Parameter.decodeText(reader);
                    break;
                case "mul2_param":
                    message.mul2_param = nnabla.Mul2Parameter.decodeText(reader);
                    break;
                case "div2_param":
                    message.div2_param = nnabla.Div2Parameter.decodeText(reader);
                    break;
                case "pow2_param":
                    message.pow2_param = nnabla.Pow2Parameter.decodeText(reader);
                    break;
                case "add_scalar_param":
                    message.add_scalar_param = nnabla.AddScalarParameter.decodeText(reader);
                    break;
                case "mul_scalar_param":
                    message.mul_scalar_param = nnabla.MulScalarParameter.decodeText(reader);
                    break;
                case "pow_scalar_param":
                    message.pow_scalar_param = nnabla.PowScalarParameter.decodeText(reader);
                    break;
                case "r_sub_scalar_param":
                    message.r_sub_scalar_param = nnabla.RSubScalarParameter.decodeText(reader);
                    break;
                case "r_div_scalar_param":
                    message.r_div_scalar_param = nnabla.RDivScalarParameter.decodeText(reader);
                    break;
                case "r_pow_scalar_param":
                    message.r_pow_scalar_param = nnabla.RPowScalarParameter.decodeText(reader);
                    break;
                case "sign_param":
                    message.sign_param = nnabla.SignParameter.decodeText(reader);
                    break;
                case "minimum_scalar_param":
                    message.minimum_scalar_param = nnabla.MinimumScalarParameter.decodeText(reader);
                    break;
                case "maximum_scalar_param":
                    message.maximum_scalar_param = nnabla.MaximumScalarParameter.decodeText(reader);
                    break;
                case "searchsorted_param":
                    message.searchsorted_param = nnabla.SearchSortedParameter.decodeText(reader);
                    break;
                case "logical_and_scalar_param":
                    message.logical_and_scalar_param = nnabla.LogicalAndScalarParameter.decodeText(reader);
                    break;
                case "logical_or_scalar_param":
                    message.logical_or_scalar_param = nnabla.LogicalOrScalarParameter.decodeText(reader);
                    break;
                case "logical_xor_scalar_param":
                    message.logical_xor_scalar_param = nnabla.LogicalXorScalarParameter.decodeText(reader);
                    break;
                case "equal_scalar_param":
                    message.equal_scalar_param = nnabla.EqualScalarParameter.decodeText(reader);
                    break;
                case "not_equal_scalar_param":
                    message.not_equal_scalar_param = nnabla.NotEqualScalarParameter.decodeText(reader);
                    break;
                case "greater_equal_scalar_param":
                    message.greater_equal_scalar_param = nnabla.GreaterEqualScalarParameter.decodeText(reader);
                    break;
                case "greater_scalar_param":
                    message.greater_scalar_param = nnabla.GreaterScalarParameter.decodeText(reader);
                    break;
                case "less_equal_scalar_param":
                    message.less_equal_scalar_param = nnabla.LessEqualScalarParameter.decodeText(reader);
                    break;
                case "less_scalar_param":
                    message.less_scalar_param = nnabla.LessScalarParameter.decodeText(reader);
                    break;
                case "reset_nan_param":
                    message.reset_nan_param = nnabla.ResetNaNParameter.decodeText(reader);
                    break;
                case "reset_inf_param":
                    message.reset_inf_param = nnabla.ResetInfParameter.decodeText(reader);
                    break;
                case "constant_param":
                    message.constant_param = nnabla.ConstantParameter.decodeText(reader);
                    break;
                case "arange_param":
                    message.arange_param = nnabla.ArangeParameter.decodeText(reader);
                    break;
                case "linspace_param":
                    message.linspace_param = nnabla.LinspaceParameter.decodeText(reader);
                    break;
                case "batch_matmul_param":
                    message.batch_matmul_param = nnabla.BatchMatmulParameter.decodeText(reader);
                    break;
                case "round_param":
                    message.round_param = nnabla.RoundParameter.decodeText(reader);
                    break;
                case "ceil_param":
                    message.ceil_param = nnabla.CeilParameter.decodeText(reader);
                    break;
                case "floor_param":
                    message.floor_param = nnabla.FloorParameter.decodeText(reader);
                    break;
                case "concatenate_param":
                    message.concatenate_param = nnabla.ConcatenateParameter.decodeText(reader);
                    break;
                case "split_param":
                    message.split_param = nnabla.SplitParameter.decodeText(reader);
                    break;
                case "stack_param":
                    message.stack_param = nnabla.StackParameter.decodeText(reader);
                    break;
                case "slice_param":
                    message.slice_param = nnabla.SliceParameter.decodeText(reader);
                    break;
                case "pad_param":
                    message.pad_param = nnabla.PadParameter.decodeText(reader);
                    break;
                case "transpose_param":
                    message.transpose_param = nnabla.TransposeParameter.decodeText(reader);
                    break;
                case "broadcast_param":
                    message.broadcast_param = nnabla.BroadcastParameter.decodeText(reader);
                    break;
                case "broadcast_to_param":
                    message.broadcast_to_param = nnabla.BroadcastToParameter.decodeText(reader);
                    break;
                case "tile_param":
                    message.tile_param = nnabla.TileParameter.decodeText(reader);
                    break;
                case "one_hot_param":
                    message.one_hot_param = nnabla.OneHotParameter.decodeText(reader);
                    break;
                case "flip_param":
                    message.flip_param = nnabla.FlipParameter.decodeText(reader);
                    break;
                case "shift_param":
                    message.shift_param = nnabla.ShiftParameter.decodeText(reader);
                    break;
                case "sort_param":
                    message.sort_param = nnabla.SortParameter.decodeText(reader);
                    break;
                case "reshape_param":
                    message.reshape_param = nnabla.ReshapeParameter.decodeText(reader);
                    break;
                case "shape_param":
                    message.shape_param = nnabla.ShapeParameter.decodeText(reader);
                    break;
                case "trilu_param":
                    message.trilu_param = nnabla.TriluParameter.decodeText(reader);
                    break;
                case "meshgrid_param":
                    message.meshgrid_param = nnabla.MeshgridParameter.decodeText(reader);
                    break;
                case "batch_cholesky_param":
                    message.batch_cholesky_param = nnabla.BatchCholeskyParameter.decodeText(reader);
                    break;
                case "gather_param":
                    message.gather_param = nnabla.GatherParameter.decodeText(reader);
                    break;
                case "scatter_nd_param":
                    message.scatter_nd_param = nnabla.ScatterNdParameter.decodeText(reader);
                    break;
                case "scatter_add_param":
                    message.scatter_add_param = nnabla.ScatterAddParameter.decodeText(reader);
                    break;
                case "bool_fill_param":
                    message.bool_fill_param = nnabla.BoolFillParameter.decodeText(reader);
                    break;
                case "pack_padded_sequence_param":
                    message.pack_padded_sequence_param = nnabla.PackPaddedSequenceParameter.decodeText(reader);
                    break;
                case "pad_packed_sequence_param":
                    message.pad_packed_sequence_param = nnabla.PadPackedSequenceParameter.decodeText(reader);
                    break;
                case "interpolate_param":
                    message.interpolate_param = nnabla.InterpolateParameter.decodeText(reader);
                    break;
                case "onnx_resize_param":
                    message.onnx_resize_param = nnabla.ONNXResizeParameter.decodeText(reader);
                    break;
                case "fft_param":
                    message.fft_param = nnabla.FFTParameter.decodeText(reader);
                    break;
                case "ifft_param":
                    message.ifft_param = nnabla.IFFTParameter.decodeText(reader);
                    break;
                case "stft_param":
                    message.stft_param = nnabla.STFTParameter.decodeText(reader);
                    break;
                case "istft_param":
                    message.istft_param = nnabla.ISTFTParameter.decodeText(reader);
                    break;
                case "dropout_param":
                    message.dropout_param = nnabla.DropoutParameter.decodeText(reader);
                    break;
                case "top_k_data_param":
                    message.top_k_data_param = nnabla.TopKDataParameter.decodeText(reader);
                    break;
                case "top_k_grad_param":
                    message.top_k_grad_param = nnabla.TopKGradParameter.decodeText(reader);
                    break;
                case "rand_param":
                    message.rand_param = nnabla.RandParameter.decodeText(reader);
                    break;
                case "randint_param":
                    message.randint_param = nnabla.RandintParameter.decodeText(reader);
                    break;
                case "randn_param":
                    message.randn_param = nnabla.RandnParameter.decodeText(reader);
                    break;
                case "rand_binomial_param":
                    message.rand_binomial_param = nnabla.RandBinomialParameter.decodeText(reader);
                    break;
                case "rand_beta_param":
                    message.rand_beta_param = nnabla.RandBetaParameter.decodeText(reader);
                    break;
                case "rand_gamma_param":
                    message.rand_gamma_param = nnabla.RandGammaParameter.decodeText(reader);
                    break;
                case "random_choice_param":
                    message.random_choice_param = nnabla.RandomChoiceParameter.decodeText(reader);
                    break;
                case "random_crop_param":
                    message.random_crop_param = nnabla.RandomCropParameter.decodeText(reader);
                    break;
                case "random_flip_param":
                    message.random_flip_param = nnabla.RandomFlipParameter.decodeText(reader);
                    break;
                case "random_shift_param":
                    message.random_shift_param = nnabla.RandomShiftParameter.decodeText(reader);
                    break;
                case "random_erase_param":
                    message.random_erase_param = nnabla.RandomEraseParameter.decodeText(reader);
                    break;
                case "image_augmentation_param":
                    message.image_augmentation_param = nnabla.ImageAugmentationParameter.decodeText(reader);
                    break;
                case "softmax_cross_entropy_param":
                    message.softmax_cross_entropy_param = nnabla.SoftmaxCrossEntropyParameter.decodeText(reader);
                    break;
                case "categorical_cross_entropy_param":
                    message.categorical_cross_entropy_param = nnabla.CategoricalCrossEntropyParameter.decodeText(reader);
                    break;
                case "huber_loss_param":
                    message.huber_loss_param = nnabla.HuberLossParameter.decodeText(reader);
                    break;
                case "epsilon_insensitive_loss_param":
                    message.epsilon_insensitive_loss_param = nnabla.EpsilonInsensitiveLossParameter.decodeText(reader);
                    break;
                case "kl_multinomial_param":
                    message.kl_multinomial_param = nnabla.KLMultinomialParameter.decodeText(reader);
                    break;
                case "affine_grid_param":
                    message.affine_grid_param = nnabla.AffineGridParameter.decodeText(reader);
                    break;
                case "warp_by_grid_param":
                    message.warp_by_grid_param = nnabla.WarpByGridParameter.decodeText(reader);
                    break;
                case "binary_connect_affine_param":
                    message.binary_connect_affine_param = nnabla.BinaryConnectAffineParameter.decodeText(reader);
                    break;
                case "binary_connect_convolution_param":
                    message.binary_connect_convolution_param = nnabla.BinaryConnectConvolutionParameter.decodeText(reader);
                    break;
                case "binary_weight_affine_param":
                    message.binary_weight_affine_param = nnabla.BinaryWeightAffineParameter.decodeText(reader);
                    break;
                case "binary_weight_convolution_param":
                    message.binary_weight_convolution_param = nnabla.BinaryWeightConvolutionParameter.decodeText(reader);
                    break;
                case "inq_affine_param":
                    message.inq_affine_param = nnabla.INQAffineParameter.decodeText(reader);
                    break;
                case "inq_convolution_param":
                    message.inq_convolution_param = nnabla.INQConvolutionParameter.decodeText(reader);
                    break;
                case "fixed_point_quantize_param":
                    message.fixed_point_quantize_param = nnabla.FixedPointQuantizeParameter.decodeText(reader);
                    break;
                case "min_max_quantize_param":
                    message.min_max_quantize_param = nnabla.MinMaxQuantizeParameter.decodeText(reader);
                    break;
                case "pow2_quantize_param":
                    message.pow2_quantize_param = nnabla.Pow2QuantizeParameter.decodeText(reader);
                    break;
                case "prune_param":
                    message.prune_param = nnabla.PruneParameter.decodeText(reader);
                    break;
                case "quantize_linear_param":
                    message.quantize_linear_param = nnabla.QuantizeLinearParameter.decodeText(reader);
                    break;
                case "top_n_error_param":
                    message.top_n_error_param = nnabla.TopNErrorParameter.decodeText(reader);
                    break;
                case "confusion_matrix_param":
                    message.confusion_matrix_param = nnabla.ConfusionMatrixParameter.decodeText(reader);
                    break;
                case "vat_noise_param":
                    message.vat_noise_param = nnabla.VATNoiseParameter.decodeText(reader);
                    break;
                case "sink_param":
                    message.sink_param = nnabla.SinkParameter.decodeText(reader);
                    break;
                case "nms_detection2d_param":
                    message.nms_detection2d_param = nnabla.NmsDetection2dParameter.decodeText(reader);
                    break;
                case "onnx_non_max_suppression_param":
                    message.onnx_non_max_suppression_param = nnabla.ONNXNonMaxSuppressionParameter.decodeText(reader);
                    break;
                case "max_pooling_backward_param":
                    message.max_pooling_backward_param = nnabla.MaxPoolingBackwardParameter.decodeText(reader);
                    break;
                case "patch_correlation_param":
                    message.patch_correlation_param = nnabla.PatchCorrelationParameter.decodeText(reader);
                    break;
                case "unique_param":
                    message.unique_param = nnabla.UniqueParameter.decodeText(reader);
                    break;
                case "eye_like_param":
                    message.eye_like_param = nnabla.EyeLikeParameter.decodeText(reader);
                    break;
                case "mod2_param":
                    message.mod2_param = nnabla.Mod2Parameter.decodeText(reader);
                    break;
                case "bit_shift_param":
                    message.bit_shift_param = nnabla.BitShiftParameter.decodeText(reader);
                    break;
                case "einsum_param":
                    message.einsum_param = nnabla.EinsumParameter.decodeText(reader);
                    break;
                case "repeat_param":
                    message.repeat_param = nnabla.RepeatParameter.decodeText(reader);
                    break;
                case "recurrent_param":
                    message.recurrent_param = nnabla.RecurrentParameter.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.Function.prototype.name = "";
nnabla.Function.prototype.type = "";
nnabla.Function.prototype.context = null;
nnabla.Function.prototype.repeat_param = null;
nnabla.Function.prototype.recurrent_param = null;

nnabla.AffineParameter = class AffineParameter {

    static decode(reader, length) {
        const message = new nnabla.AffineParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AffineParameter();
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

nnabla.AffineParameter.prototype.base_axis = 0n;

nnabla.RNNParameter = class RNNParameter {

    static decode(reader, length) {
        const message = new nnabla.RNNParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RNNParameter();
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

nnabla.RNNParameter.prototype.num_layers = 0n;
nnabla.RNNParameter.prototype.nonlinearity = "";
nnabla.RNNParameter.prototype.dropout = 0;
nnabla.RNNParameter.prototype.bidirectional = false;
nnabla.RNNParameter.prototype.training = false;

nnabla.LSTMParameter = class LSTMParameter {

    static decode(reader, length) {
        const message = new nnabla.LSTMParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LSTMParameter();
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

nnabla.LSTMParameter.prototype.num_layers = 0n;
nnabla.LSTMParameter.prototype.dropout = 0;
nnabla.LSTMParameter.prototype.bidirectional = false;
nnabla.LSTMParameter.prototype.training = false;

nnabla.GRUParameter = class GRUParameter {

    static decode(reader, length) {
        const message = new nnabla.GRUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.GRUParameter();
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

nnabla.GRUParameter.prototype.num_layers = 0n;
nnabla.GRUParameter.prototype.dropout = 0;
nnabla.GRUParameter.prototype.bidirectional = false;
nnabla.GRUParameter.prototype.training = false;

nnabla.ConvolutionParameter = class ConvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.ConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.ConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.ConvolutionParameter.prototype.base_axis = 0n;
nnabla.ConvolutionParameter.prototype.pad = null;
nnabla.ConvolutionParameter.prototype.stride = null;
nnabla.ConvolutionParameter.prototype.dilation = null;
nnabla.ConvolutionParameter.prototype.group = 0n;
nnabla.ConvolutionParameter.prototype.channel_last = false;

nnabla.FusedConvolutionParameter = class FusedConvolutionParameter {

    constructor() {
        this.nonlinearity_args = [];
    }

    static decode(reader, length) {
        const message = new nnabla.FusedConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.FusedConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.FusedConvolutionParameter.prototype.base_axis = 0n;
nnabla.FusedConvolutionParameter.prototype.pad = null;
nnabla.FusedConvolutionParameter.prototype.stride = null;
nnabla.FusedConvolutionParameter.prototype.dilation = null;
nnabla.FusedConvolutionParameter.prototype.group = 0n;
nnabla.FusedConvolutionParameter.prototype.channel_last = false;
nnabla.FusedConvolutionParameter.prototype.decay_rate = 0;
nnabla.FusedConvolutionParameter.prototype.eps = 0;
nnabla.FusedConvolutionParameter.prototype.batch_stat = false;
nnabla.FusedConvolutionParameter.prototype.nonlinearity = "";
nnabla.FusedConvolutionParameter.prototype.pad_mode = "";
nnabla.FusedConvolutionParameter.prototype.constant_value = 0;

nnabla.DepthwiseConvolutionParameter = class DepthwiseConvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.DepthwiseConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.DepthwiseConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.DepthwiseConvolutionParameter.prototype.base_axis = 0n;
nnabla.DepthwiseConvolutionParameter.prototype.pad = null;
nnabla.DepthwiseConvolutionParameter.prototype.stride = null;
nnabla.DepthwiseConvolutionParameter.prototype.dilation = null;
nnabla.DepthwiseConvolutionParameter.prototype.multiplier = 0n;

nnabla.DeconvolutionParameter = class DeconvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.DeconvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.group = reader.int64();
                    break;
                case 6:
                    message.channel_last = reader.bool();
                    break;
                case 7:
                    message.output_padding = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.DeconvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
                    break;
                case "group":
                    message.group = reader.int64();
                    break;
                case "channel_last":
                    message.channel_last = reader.bool();
                    break;
                case "output_padding":
                    message.output_padding = nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.DeconvolutionParameter.prototype.base_axis = 0n;
nnabla.DeconvolutionParameter.prototype.pad = null;
nnabla.DeconvolutionParameter.prototype.stride = null;
nnabla.DeconvolutionParameter.prototype.dilation = null;
nnabla.DeconvolutionParameter.prototype.group = 0n;
nnabla.DeconvolutionParameter.prototype.channel_last = false;
nnabla.DeconvolutionParameter.prototype.output_padding = null;

nnabla.DepthwiseDeconvolutionParameter = class DepthwiseDeconvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.DepthwiseDeconvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.DepthwiseDeconvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.DepthwiseDeconvolutionParameter.prototype.base_axis = 0n;
nnabla.DepthwiseDeconvolutionParameter.prototype.pad = null;
nnabla.DepthwiseDeconvolutionParameter.prototype.stride = null;
nnabla.DepthwiseDeconvolutionParameter.prototype.dilation = null;
nnabla.DepthwiseDeconvolutionParameter.prototype.divisor = 0n;

nnabla.DeformableConvolutionParameter = class DeformableConvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.DeformableConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.DeformableConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.DeformableConvolutionParameter.prototype.base_axis = 0n;
nnabla.DeformableConvolutionParameter.prototype.pad = null;
nnabla.DeformableConvolutionParameter.prototype.stride = null;
nnabla.DeformableConvolutionParameter.prototype.dilation = null;
nnabla.DeformableConvolutionParameter.prototype.group = 0n;
nnabla.DeformableConvolutionParameter.prototype.deformable_group = 0n;
nnabla.DeformableConvolutionParameter.prototype.channel_last = false;

nnabla.MaxPoolingParameter = class MaxPoolingParameter {

    static decode(reader, length) {
        const message = new nnabla.MaxPoolingParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.MaxPoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
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

nnabla.MaxPoolingParameter.prototype.kernel = null;
nnabla.MaxPoolingParameter.prototype.stride = null;
nnabla.MaxPoolingParameter.prototype.ignore_border = false;
nnabla.MaxPoolingParameter.prototype.pad = null;
nnabla.MaxPoolingParameter.prototype.channel_last = false;

nnabla.AveragePoolingParameter = class AveragePoolingParameter {

    static decode(reader, length) {
        const message = new nnabla.AveragePoolingParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.AveragePoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
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

nnabla.AveragePoolingParameter.prototype.kernel = null;
nnabla.AveragePoolingParameter.prototype.stride = null;
nnabla.AveragePoolingParameter.prototype.ignore_border = false;
nnabla.AveragePoolingParameter.prototype.pad = null;
nnabla.AveragePoolingParameter.prototype.channel_last = false;
nnabla.AveragePoolingParameter.prototype.including_pad = false;

nnabla.SumPoolingParameter = class SumPoolingParameter {

    static decode(reader, length) {
        const message = new nnabla.SumPoolingParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.SumPoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
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

nnabla.SumPoolingParameter.prototype.kernel = null;
nnabla.SumPoolingParameter.prototype.stride = null;
nnabla.SumPoolingParameter.prototype.ignore_border = false;
nnabla.SumPoolingParameter.prototype.pad = null;
nnabla.SumPoolingParameter.prototype.channel_last = false;

nnabla.UnpoolingParameter = class UnpoolingParameter {

    static decode(reader, length) {
        const message = new nnabla.UnpoolingParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.UnpoolingParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = nnabla.Shape.decodeText(reader);
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

nnabla.UnpoolingParameter.prototype.kernel = null;
nnabla.UnpoolingParameter.prototype.channel_last = false;

nnabla.RoiAlignParameter = class RoiAlignParameter {

    constructor() {
        this.spatial_scale = [];
    }

    static decode(reader, length) {
        const message = new nnabla.RoiAlignParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.output_size = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RoiAlignParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "output_size":
                    message.output_size = nnabla.Shape.decodeText(reader);
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

nnabla.RoiAlignParameter.prototype.output_size = null;
nnabla.RoiAlignParameter.prototype.sampling_ratio = 0n;
nnabla.RoiAlignParameter.prototype.channel_last = false;

nnabla.ReLUParameter = class ReLUParameter {

    static decode(reader, length) {
        const message = new nnabla.ReLUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ReLUParameter();
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

nnabla.ReLUParameter.prototype.inplace = false;

nnabla.LeakyReLUParameter = class LeakyReLUParameter {

    static decode(reader, length) {
        const message = new nnabla.LeakyReLUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LeakyReLUParameter();
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

nnabla.LeakyReLUParameter.prototype.alpha = 0;
nnabla.LeakyReLUParameter.prototype.inplace = false;

nnabla.SoftmaxParameter = class SoftmaxParameter {

    static decode(reader, length) {
        const message = new nnabla.SoftmaxParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SoftmaxParameter();
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

nnabla.SoftmaxParameter.prototype.axis = 0n;

nnabla.LogSoftmaxParameter = class LogSoftmaxParameter {

    static decode(reader, length) {
        const message = new nnabla.LogSoftmaxParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LogSoftmaxParameter();
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

nnabla.LogSoftmaxParameter.prototype.axis = 0n;

nnabla.ELUParameter = class ELUParameter {

    static decode(reader, length) {
        const message = new nnabla.ELUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ELUParameter();
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

nnabla.ELUParameter.prototype.alpha = 0;

nnabla.SELUParameter = class SELUParameter {

    static decode(reader, length) {
        const message = new nnabla.SELUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SELUParameter();
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

nnabla.SELUParameter.prototype.scale = 0;
nnabla.SELUParameter.prototype.alpha = 0;

nnabla.CReLUParameter = class CReLUParameter {

    static decode(reader, length) {
        const message = new nnabla.CReLUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CReLUParameter();
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

nnabla.CReLUParameter.prototype.axis = 0n;

nnabla.CELUParameter = class CELUParameter {

    static decode(reader, length) {
        const message = new nnabla.CELUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CELUParameter();
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

nnabla.CELUParameter.prototype.alpha = 0;
nnabla.CELUParameter.prototype.axis = 0n;

nnabla.PReLUParameter = class PReLUParameter {

    static decode(reader, length) {
        const message = new nnabla.PReLUParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PReLUParameter();
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

nnabla.PReLUParameter.prototype.base_axis = 0n;

nnabla.SoftPlusParameter = class SoftPlusParameter {

    static decode(reader, length) {
        const message = new nnabla.SoftPlusParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SoftPlusParameter();
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

nnabla.SoftPlusParameter.prototype.beta = 0;

nnabla.FusedBatchNormalizationParameter = class FusedBatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.FusedBatchNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.FusedBatchNormalizationParameter();
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

nnabla.FusedBatchNormalizationParameter.prototype.decay_rate = 0;
nnabla.FusedBatchNormalizationParameter.prototype.eps = 0;
nnabla.FusedBatchNormalizationParameter.prototype.batch_stat = false;
nnabla.FusedBatchNormalizationParameter.prototype.nonlinearity = "";

nnabla.BatchNormalizationParameter = class BatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.BatchNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BatchNormalizationParameter();
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

nnabla.BatchNormalizationParameter.prototype.decay_rate = 0;
nnabla.BatchNormalizationParameter.prototype.eps = 0;
nnabla.BatchNormalizationParameter.prototype.batch_stat = false;
nnabla.BatchNormalizationParameter.prototype.no_scale = false;
nnabla.BatchNormalizationParameter.prototype.no_bias = false;

nnabla.GroupNormalizationParameter = class GroupNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decode(reader, length) {
        const message = new nnabla.GroupNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.GroupNormalizationParameter();
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

nnabla.GroupNormalizationParameter.prototype.num_groups = 0n;
nnabla.GroupNormalizationParameter.prototype.channel_axis = 0n;
nnabla.GroupNormalizationParameter.prototype.eps = 0;
nnabla.GroupNormalizationParameter.prototype.no_scale = false;
nnabla.GroupNormalizationParameter.prototype.no_bias = false;

nnabla.InstanceNormalizationParameter = class InstanceNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decode(reader, length) {
        const message = new nnabla.InstanceNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.InstanceNormalizationParameter();
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

nnabla.InstanceNormalizationParameter.prototype.channel_axis = 0n;
nnabla.InstanceNormalizationParameter.prototype.eps = 0;
nnabla.InstanceNormalizationParameter.prototype.no_scale = false;
nnabla.InstanceNormalizationParameter.prototype.no_bias = false;

nnabla.LayerNormalizationParameter = class LayerNormalizationParameter {

    constructor() {
        this.batch_axis = [];
    }

    static decode(reader, length) {
        const message = new nnabla.LayerNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LayerNormalizationParameter();
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

nnabla.LayerNormalizationParameter.prototype.eps = 0;
nnabla.LayerNormalizationParameter.prototype.no_scale = false;
nnabla.LayerNormalizationParameter.prototype.no_bias = false;

nnabla.NormNormalizationParameter = class NormNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.NormNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.NormNormalizationParameter();
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

nnabla.NormNormalizationParameter.prototype.p = 0;
nnabla.NormNormalizationParameter.prototype.eps = 0;

nnabla.SyncBatchNormalizationParameter = class SyncBatchNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.SyncBatchNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.comm = nnabla.Communicator.decode(reader, reader.uint32());
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
        const message = new nnabla.SyncBatchNormalizationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "comm":
                    message.comm = nnabla.Communicator.decodeText(reader);
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

nnabla.SyncBatchNormalizationParameter.prototype.comm = null;
nnabla.SyncBatchNormalizationParameter.prototype.group = "";
nnabla.SyncBatchNormalizationParameter.prototype.decay_rate = 0;
nnabla.SyncBatchNormalizationParameter.prototype.eps = 0;
nnabla.SyncBatchNormalizationParameter.prototype.batch_stat = false;

nnabla.TensorNormalizationParameter = class TensorNormalizationParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.TensorNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TensorNormalizationParameter();
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

nnabla.TensorNormalizationParameter.prototype.eps = 0;
nnabla.TensorNormalizationParameter.prototype.no_scale = false;
nnabla.TensorNormalizationParameter.prototype.no_bias = false;

nnabla.WeightNormalizationParameter = class WeightNormalizationParameter {

    static decode(reader, length) {
        const message = new nnabla.WeightNormalizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.WeightNormalizationParameter();
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

nnabla.WeightNormalizationParameter.prototype.dim = 0n;
nnabla.WeightNormalizationParameter.prototype.eps = 0;

nnabla.WeightStandardizationParameter = class WeightStandardizationParameter {

    static decode(reader, length) {
        const message = new nnabla.WeightStandardizationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.WeightStandardizationParameter();
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

nnabla.WeightStandardizationParameter.prototype.channel_axis = 0n;
nnabla.WeightStandardizationParameter.prototype.eps = 0;

nnabla.SpectralNormParameter = class SpectralNormParameter {

    static decode(reader, length) {
        const message = new nnabla.SpectralNormParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SpectralNormParameter();
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

nnabla.SpectralNormParameter.prototype.dim = 0n;
nnabla.SpectralNormParameter.prototype.itr = 0n;
nnabla.SpectralNormParameter.prototype.eps = 0;
nnabla.SpectralNormParameter.prototype.test = false;
nnabla.SpectralNormParameter.prototype.output_u = false;

nnabla.MeanSubtractionParameter = class MeanSubtractionParameter {

    static decode(reader, length) {
        const message = new nnabla.MeanSubtractionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MeanSubtractionParameter();
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

nnabla.MeanSubtractionParameter.prototype.base_axis = 0n;
nnabla.MeanSubtractionParameter.prototype.update_running_mean = false;

nnabla.ClipGradByNormParameter = class ClipGradByNormParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.ClipGradByNormParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ClipGradByNormParameter();
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

nnabla.ClipGradByNormParameter.prototype.clip_norm = 0;

nnabla.SumParameter = class SumParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.SumParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SumParameter();
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

nnabla.SumParameter.prototype.keep_dims = false;

nnabla.CumSumParameter = class CumSumParameter {

    static decode(reader, length) {
        const message = new nnabla.CumSumParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CumSumParameter();
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

nnabla.CumSumParameter.prototype.axis = 0n;
nnabla.CumSumParameter.prototype.exclusive = false;
nnabla.CumSumParameter.prototype.reverse = false;

nnabla.MeanParameter = class MeanParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.MeanParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MeanParameter();
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

nnabla.MeanParameter.prototype.keep_dims = false;

nnabla.MaxParameter = class MaxParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.MaxParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MaxParameter();
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

nnabla.MaxParameter.prototype.keep_dims = false;
nnabla.MaxParameter.prototype.with_index = false;
nnabla.MaxParameter.prototype.only_index = false;

nnabla.MinParameter = class MinParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.MinParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MinParameter();
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

nnabla.MinParameter.prototype.keep_dims = false;
nnabla.MinParameter.prototype.with_index = false;
nnabla.MinParameter.prototype.only_index = false;

nnabla.NormParameter = class NormParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.NormParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.NormParameter();
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

nnabla.NormParameter.prototype.p = 0;
nnabla.NormParameter.prototype.keep_dims = false;

nnabla.ProdParameter = class ProdParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.ProdParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ProdParameter();
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

nnabla.ProdParameter.prototype.keep_dims = false;

nnabla.CumProdParameter = class CumProdParameter {

    static decode(reader, length) {
        const message = new nnabla.CumProdParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CumProdParameter();
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

nnabla.CumProdParameter.prototype.axis = 0n;
nnabla.CumProdParameter.prototype.exclusive = false;
nnabla.CumProdParameter.prototype.reverse = false;

nnabla.Add2Parameter = class Add2Parameter {

    static decode(reader, length) {
        const message = new nnabla.Add2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Add2Parameter();
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

nnabla.Add2Parameter.prototype.inplace = false;

nnabla.BcAdd2Parameter = class BcAdd2Parameter {

    static decode(reader, length) {
        const message = new nnabla.BcAdd2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BcAdd2Parameter();
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

nnabla.BcAdd2Parameter.prototype.inplace = false;

nnabla.Sub2Parameter = class Sub2Parameter {

    static decode(reader, length) {
        const message = new nnabla.Sub2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Sub2Parameter();
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

nnabla.Sub2Parameter.prototype.inplace = false;

nnabla.Mul2Parameter = class Mul2Parameter {

    static decode(reader, length) {
        const message = new nnabla.Mul2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Mul2Parameter();
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

nnabla.Mul2Parameter.prototype.inplace = false;

nnabla.Div2Parameter = class Div2Parameter {

    static decode(reader, length) {
        const message = new nnabla.Div2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Div2Parameter();
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

nnabla.Div2Parameter.prototype.inplace = false;

nnabla.Pow2Parameter = class Pow2Parameter {

    static decode(reader, length) {
        const message = new nnabla.Pow2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Pow2Parameter();
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

nnabla.Pow2Parameter.prototype.inplace = false;

nnabla.AddScalarParameter = class AddScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.AddScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AddScalarParameter();
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

nnabla.AddScalarParameter.prototype.val = 0;
nnabla.AddScalarParameter.prototype.inplace = false;

nnabla.MulScalarParameter = class MulScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.MulScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MulScalarParameter();
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

nnabla.MulScalarParameter.prototype.val = 0;
nnabla.MulScalarParameter.prototype.inplace = false;

nnabla.PowScalarParameter = class PowScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.PowScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PowScalarParameter();
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

nnabla.PowScalarParameter.prototype.val = 0;
nnabla.PowScalarParameter.prototype.inplace = false;

nnabla.RSubScalarParameter = class RSubScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.RSubScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RSubScalarParameter();
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

nnabla.RSubScalarParameter.prototype.val = 0;

nnabla.RDivScalarParameter = class RDivScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.RDivScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RDivScalarParameter();
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

nnabla.RDivScalarParameter.prototype.val = 0;

nnabla.RPowScalarParameter = class RPowScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.RPowScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RPowScalarParameter();
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

nnabla.RPowScalarParameter.prototype.val = 0;

nnabla.SignParameter = class SignParameter {

    static decode(reader, length) {
        const message = new nnabla.SignParameter();
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
        const message = new nnabla.SignParameter();
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

nnabla.SignParameter.prototype.alpha = 0;

nnabla.MinimumScalarParameter = class MinimumScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.MinimumScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MinimumScalarParameter();
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

nnabla.MinimumScalarParameter.prototype.val = 0;

nnabla.MaximumScalarParameter = class MaximumScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.MaximumScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MaximumScalarParameter();
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

nnabla.MaximumScalarParameter.prototype.val = 0;

nnabla.SearchSortedParameter = class SearchSortedParameter {

    static decode(reader, length) {
        const message = new nnabla.SearchSortedParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SearchSortedParameter();
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

nnabla.SearchSortedParameter.prototype.right = false;

nnabla.LogicalAndScalarParameter = class LogicalAndScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.LogicalAndScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LogicalAndScalarParameter();
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

nnabla.LogicalAndScalarParameter.prototype.val = false;

nnabla.LogicalOrScalarParameter = class LogicalOrScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.LogicalOrScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LogicalOrScalarParameter();
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

nnabla.LogicalOrScalarParameter.prototype.val = false;

nnabla.LogicalXorScalarParameter = class LogicalXorScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.LogicalXorScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LogicalXorScalarParameter();
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

nnabla.LogicalXorScalarParameter.prototype.val = false;

nnabla.EqualScalarParameter = class EqualScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.EqualScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.EqualScalarParameter();
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

nnabla.EqualScalarParameter.prototype.val = 0;

nnabla.NotEqualScalarParameter = class NotEqualScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.NotEqualScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.NotEqualScalarParameter();
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

nnabla.NotEqualScalarParameter.prototype.val = 0;

nnabla.GreaterEqualScalarParameter = class GreaterEqualScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.GreaterEqualScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.GreaterEqualScalarParameter();
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

nnabla.GreaterEqualScalarParameter.prototype.val = 0;

nnabla.GreaterScalarParameter = class GreaterScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.GreaterScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.GreaterScalarParameter();
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

nnabla.GreaterScalarParameter.prototype.val = 0;

nnabla.LessEqualScalarParameter = class LessEqualScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.LessEqualScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LessEqualScalarParameter();
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

nnabla.LessEqualScalarParameter.prototype.val = 0;

nnabla.LessScalarParameter = class LessScalarParameter {

    static decode(reader, length) {
        const message = new nnabla.LessScalarParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LessScalarParameter();
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

nnabla.LessScalarParameter.prototype.val = 0;

nnabla.ResetNaNParameter = class ResetNaNParameter {

    static decode(reader, length) {
        const message = new nnabla.ResetNaNParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ResetNaNParameter();
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

nnabla.ResetNaNParameter.prototype.val = 0;

nnabla.ResetInfParameter = class ResetInfParameter {

    static decode(reader, length) {
        const message = new nnabla.ResetInfParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ResetInfParameter();
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

nnabla.ResetInfParameter.prototype.val = 0;

nnabla.ConstantParameter = class ConstantParameter {

    static decode(reader, length) {
        const message = new nnabla.ConstantParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val = reader.float();
                    break;
                case 2:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.ConstantParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val = reader.float();
                    break;
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.ConstantParameter.prototype.val = 0;
nnabla.ConstantParameter.prototype.shape = null;

nnabla.ArangeParameter = class ArangeParameter {

    static decode(reader, length) {
        const message = new nnabla.ArangeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ArangeParameter();
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

nnabla.ArangeParameter.prototype.start = 0;
nnabla.ArangeParameter.prototype.stop = 0;
nnabla.ArangeParameter.prototype.step = 0;

nnabla.LinspaceParameter = class LinspaceParameter {

    static decode(reader, length) {
        const message = new nnabla.LinspaceParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.LinspaceParameter();
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

nnabla.LinspaceParameter.prototype.start = 0;
nnabla.LinspaceParameter.prototype.stop = 0;
nnabla.LinspaceParameter.prototype.num = 0n;

nnabla.BatchMatmulParameter = class BatchMatmulParameter {

    static decode(reader, length) {
        const message = new nnabla.BatchMatmulParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BatchMatmulParameter();
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

nnabla.BatchMatmulParameter.prototype.transpose_a = false;
nnabla.BatchMatmulParameter.prototype.transpose_b = false;

nnabla.RoundParameter = class RoundParameter {

    static decode(reader, length) {
        const message = new nnabla.RoundParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RoundParameter();
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

nnabla.CeilParameter = class CeilParameter {

    static decode(reader, length) {
        const message = new nnabla.CeilParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CeilParameter();
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

nnabla.FloorParameter = class FloorParameter {

    static decode(reader, length) {
        const message = new nnabla.FloorParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.FloorParameter();
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

nnabla.ConcatenateParameter = class ConcatenateParameter {

    static decode(reader, length) {
        const message = new nnabla.ConcatenateParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ConcatenateParameter();
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

nnabla.ConcatenateParameter.prototype.axis = 0n;

nnabla.SplitParameter = class SplitParameter {

    static decode(reader, length) {
        const message = new nnabla.SplitParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SplitParameter();
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

nnabla.SplitParameter.prototype.axis = 0n;

nnabla.StackParameter = class StackParameter {

    static decode(reader, length) {
        const message = new nnabla.StackParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.StackParameter();
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

nnabla.StackParameter.prototype.axis = 0n;

nnabla.SliceParameter = class SliceParameter {

    constructor() {
        this.start = [];
        this.stop = [];
        this.step = [];
    }

    static decode(reader, length) {
        const message = new nnabla.SliceParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SliceParameter();
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

nnabla.PadParameter = class PadParameter {

    constructor() {
        this.pad_width = [];
    }

    static decode(reader, length) {
        const message = new nnabla.PadParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PadParameter();
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

nnabla.PadParameter.prototype.mode = "";
nnabla.PadParameter.prototype.constant_value = 0;

nnabla.TransposeParameter = class TransposeParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.TransposeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TransposeParameter();
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

nnabla.BroadcastParameter = class BroadcastParameter {

    static decode(reader, length) {
        const message = new nnabla.BroadcastParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.BroadcastParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.BroadcastParameter.prototype.shape = null;

nnabla.BroadcastToParameter = class BroadcastToParameter {

    static decode(reader, length) {
        const message = new nnabla.BroadcastToParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BroadcastToParameter();
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

nnabla.BroadcastToParameter.prototype.axis = 0n;

nnabla.TileParameter = class TileParameter {

    constructor() {
        this.reps = [];
    }

    static decode(reader, length) {
        const message = new nnabla.TileParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TileParameter();
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

nnabla.OneHotParameter = class OneHotParameter {

    static decode(reader, length) {
        const message = new nnabla.OneHotParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.OneHotParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.OneHotParameter.prototype.shape = null;

nnabla.FlipParameter = class FlipParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.FlipParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.FlipParameter();
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

nnabla.ShiftParameter = class ShiftParameter {

    constructor() {
        this.shifts = [];
    }

    static decode(reader, length) {
        const message = new nnabla.ShiftParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ShiftParameter();
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

nnabla.ShiftParameter.prototype.border_mode = "";

nnabla.SortParameter = class SortParameter {

    static decode(reader, length) {
        const message = new nnabla.SortParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SortParameter();
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

nnabla.SortParameter.prototype.axis = 0n;
nnabla.SortParameter.prototype.reverse = false;
nnabla.SortParameter.prototype.with_index = false;
nnabla.SortParameter.prototype.only_index = false;

nnabla.ReshapeParameter = class ReshapeParameter {

    static decode(reader, length) {
        const message = new nnabla.ReshapeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.ReshapeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.ReshapeParameter.prototype.shape = null;
nnabla.ReshapeParameter.prototype.inplace = false;

nnabla.ShapeParameter = class ShapeParameter {

    static decode(reader, length) {
        const message = new nnabla.ShapeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ShapeParameter();
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

nnabla.ShapeParameter.prototype.start = 0n;
nnabla.ShapeParameter.prototype.end = 0n;

nnabla.TriluParameter = class TriluParameter {

    static decode(reader, length) {
        const message = new nnabla.TriluParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.int64();
                    break;
                case 2:
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
        const message = new nnabla.TriluParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.int64();
                    break;
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

nnabla.TriluParameter.prototype.k = 0n;
nnabla.TriluParameter.prototype.upper = false;

nnabla.MeshgridParameter = class MeshgridParameter {

    static decode(reader, length) {
        const message = new nnabla.MeshgridParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MeshgridParameter();
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

nnabla.MeshgridParameter.prototype.ij_indexing = false;

nnabla.BatchCholeskyParameter = class BatchCholeskyParameter {

    static decode(reader, length) {
        const message = new nnabla.BatchCholeskyParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BatchCholeskyParameter();
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

nnabla.BatchCholeskyParameter.prototype.upper = false;

nnabla.GatherParameter = class GatherParameter {

    static decode(reader, length) {
        const message = new nnabla.GatherParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.GatherParameter();
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

nnabla.GatherParameter.prototype.axis = 0n;
nnabla.GatherParameter.prototype.batch_dims = 0n;

nnabla.ScatterNdParameter = class ScatterNdParameter {

    constructor() {
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new nnabla.ScatterNdParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ScatterNdParameter();
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

nnabla.ScatterNdParameter.prototype.add = false;

nnabla.ScatterAddParameter = class ScatterAddParameter {

    static decode(reader, length) {
        const message = new nnabla.ScatterAddParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ScatterAddParameter();
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

nnabla.ScatterAddParameter.prototype.axis = 0n;

nnabla.BoolFillParameter = class BoolFillParameter {

    static decode(reader, length) {
        const message = new nnabla.BoolFillParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BoolFillParameter();
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

nnabla.BoolFillParameter.prototype.value = 0;

nnabla.PackPaddedSequenceParameter = class PackPaddedSequenceParameter {

    static decode(reader, length) {
        const message = new nnabla.PackPaddedSequenceParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PackPaddedSequenceParameter();
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

nnabla.PackPaddedSequenceParameter.prototype.batch_first = false;

nnabla.PadPackedSequenceParameter = class PadPackedSequenceParameter {

    static decode(reader, length) {
        const message = new nnabla.PadPackedSequenceParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PadPackedSequenceParameter();
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

nnabla.PadPackedSequenceParameter.prototype.batch_first = false;
nnabla.PadPackedSequenceParameter.prototype.padding_value = 0;
nnabla.PadPackedSequenceParameter.prototype.total_length = 0n;

nnabla.InterpolateParameter = class InterpolateParameter {

    constructor() {
        this.output_size = [];
    }

    static decode(reader, length) {
        const message = new nnabla.InterpolateParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.InterpolateParameter();
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

nnabla.InterpolateParameter.prototype.mode = "";
nnabla.InterpolateParameter.prototype.align_corners = false;
nnabla.InterpolateParameter.prototype.half_pixel = false;
nnabla.InterpolateParameter.prototype.half_pixel_for_nn = false;
nnabla.InterpolateParameter.prototype.channel_last = false;

nnabla.ONNXResizeParameter = class ONNXResizeParameter {

    constructor() {
        this.roi = [];
        this.scales = [];
        this.sizes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.ONNXResizeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.roi = reader.floats(message.roi, tag);
                    break;
                case 2:
                    message.scales = reader.floats(message.scales, tag);
                    break;
                case 3:
                    message.sizes = reader.array(message.sizes, () => reader.int64(), tag);
                    break;
                case 4:
                    message.mode = reader.string();
                    break;
                case 5:
                    message.coordinate_transformation_mode = reader.string();
                    break;
                case 6:
                    message.cubic_coeff_a = reader.float();
                    break;
                case 7:
                    message.exclude_outside = reader.int64();
                    break;
                case 8:
                    message.extrapolation_value = reader.float();
                    break;
                case 9:
                    message.nearest_mode = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.ONNXResizeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "roi":
                    reader.array(message.roi, () => reader.float());
                    break;
                case "scales":
                    reader.array(message.scales, () => reader.float());
                    break;
                case "sizes":
                    reader.array(message.sizes, () => reader.int64());
                    break;
                case "mode":
                    message.mode = reader.string();
                    break;
                case "coordinate_transformation_mode":
                    message.coordinate_transformation_mode = reader.string();
                    break;
                case "cubic_coeff_a":
                    message.cubic_coeff_a = reader.float();
                    break;
                case "exclude_outside":
                    message.exclude_outside = reader.int64();
                    break;
                case "extrapolation_value":
                    message.extrapolation_value = reader.float();
                    break;
                case "nearest_mode":
                    message.nearest_mode = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.ONNXResizeParameter.prototype.mode = "";
nnabla.ONNXResizeParameter.prototype.coordinate_transformation_mode = "";
nnabla.ONNXResizeParameter.prototype.cubic_coeff_a = 0;
nnabla.ONNXResizeParameter.prototype.exclude_outside = 0n;
nnabla.ONNXResizeParameter.prototype.extrapolation_value = 0;
nnabla.ONNXResizeParameter.prototype.nearest_mode = "";

nnabla.FFTParameter = class FFTParameter {

    static decode(reader, length) {
        const message = new nnabla.FFTParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.FFTParameter();
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

nnabla.FFTParameter.prototype.signal_ndim = 0n;
nnabla.FFTParameter.prototype.normalized = false;

nnabla.IFFTParameter = class IFFTParameter {

    static decode(reader, length) {
        const message = new nnabla.IFFTParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.IFFTParameter();
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

nnabla.IFFTParameter.prototype.signal_ndim = 0n;
nnabla.IFFTParameter.prototype.normalized = false;

nnabla.STFTParameter = class STFTParameter {

    static decode(reader, length) {
        const message = new nnabla.STFTParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.STFTParameter();
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

nnabla.STFTParameter.prototype.window_size = 0n;
nnabla.STFTParameter.prototype.stride = 0n;
nnabla.STFTParameter.prototype.fft_size = 0n;
nnabla.STFTParameter.prototype.window_type = "";
nnabla.STFTParameter.prototype.center = false;
nnabla.STFTParameter.prototype.pad_mode = "";
nnabla.STFTParameter.prototype.as_istft_backward = false;

nnabla.ISTFTParameter = class ISTFTParameter {

    static decode(reader, length) {
        const message = new nnabla.ISTFTParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ISTFTParameter();
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

nnabla.ISTFTParameter.prototype.window_size = 0n;
nnabla.ISTFTParameter.prototype.stride = 0n;
nnabla.ISTFTParameter.prototype.fft_size = 0n;
nnabla.ISTFTParameter.prototype.window_type = "";
nnabla.ISTFTParameter.prototype.center = false;
nnabla.ISTFTParameter.prototype.pad_mode = "";
nnabla.ISTFTParameter.prototype.as_stft_backward = false;

nnabla.DropoutParameter = class DropoutParameter {

    static decode(reader, length) {
        const message = new nnabla.DropoutParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.DropoutParameter();
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

nnabla.DropoutParameter.prototype.p = 0;
nnabla.DropoutParameter.prototype.seed = 0n;

nnabla.TopKDataParameter = class TopKDataParameter {

    static decode(reader, length) {
        const message = new nnabla.TopKDataParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TopKDataParameter();
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

nnabla.TopKDataParameter.prototype.k = 0n;
nnabla.TopKDataParameter.prototype.abs = false;
nnabla.TopKDataParameter.prototype.reduce = false;
nnabla.TopKDataParameter.prototype.base_axis = 0n;
nnabla.TopKDataParameter.prototype.largest = false;
nnabla.TopKDataParameter.prototype.with_index = false;

nnabla.TopKGradParameter = class TopKGradParameter {

    static decode(reader, length) {
        const message = new nnabla.TopKGradParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TopKGradParameter();
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

nnabla.TopKGradParameter.prototype.k = 0n;
nnabla.TopKGradParameter.prototype.abs = false;
nnabla.TopKGradParameter.prototype.base_axis = 0n;

nnabla.RandParameter = class RandParameter {

    static decode(reader, length) {
        const message = new nnabla.RandParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandParameter();
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
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandParameter.prototype.low = 0;
nnabla.RandParameter.prototype.high = 0;
nnabla.RandParameter.prototype.shape = null;
nnabla.RandParameter.prototype.seed = 0n;

nnabla.RandintParameter = class RandintParameter {

    static decode(reader, length) {
        const message = new nnabla.RandintParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandintParameter();
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
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandintParameter.prototype.low = 0n;
nnabla.RandintParameter.prototype.high = 0n;
nnabla.RandintParameter.prototype.shape = null;
nnabla.RandintParameter.prototype.seed = 0n;

nnabla.RandnParameter = class RandnParameter {

    static decode(reader, length) {
        const message = new nnabla.RandnParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandnParameter();
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
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandnParameter.prototype.mu = 0;
nnabla.RandnParameter.prototype.sigma = 0;
nnabla.RandnParameter.prototype.shape = null;
nnabla.RandnParameter.prototype.seed = 0n;

nnabla.RandBinomialParameter = class RandBinomialParameter {

    static decode(reader, length) {
        const message = new nnabla.RandBinomialParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandBinomialParameter();
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
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandBinomialParameter.prototype.n = 0n;
nnabla.RandBinomialParameter.prototype.p = 0;
nnabla.RandBinomialParameter.prototype.shape = null;
nnabla.RandBinomialParameter.prototype.seed = 0n;

nnabla.RandBetaParameter = class RandBetaParameter {

    static decode(reader, length) {
        const message = new nnabla.RandBetaParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandBetaParameter();
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
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandBetaParameter.prototype.alpha = 0;
nnabla.RandBetaParameter.prototype.beta = 0;
nnabla.RandBetaParameter.prototype.shape = null;
nnabla.RandBetaParameter.prototype.seed = 0n;

nnabla.RandGammaParameter = class RandGammaParameter {

    static decode(reader, length) {
        const message = new nnabla.RandGammaParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandGammaParameter();
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
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandGammaParameter.prototype.k = 0;
nnabla.RandGammaParameter.prototype.theta = 0;
nnabla.RandGammaParameter.prototype.shape = null;
nnabla.RandGammaParameter.prototype.seed = 0n;

nnabla.RandomChoiceParameter = class RandomChoiceParameter {

    static decode(reader, length) {
        const message = new nnabla.RandomChoiceParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandomChoiceParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandomChoiceParameter.prototype.shape = null;
nnabla.RandomChoiceParameter.prototype.replace = false;
nnabla.RandomChoiceParameter.prototype.seed = 0n;

nnabla.RandomCropParameter = class RandomCropParameter {

    static decode(reader, length) {
        const message = new nnabla.RandomCropParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.RandomCropParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
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

nnabla.RandomCropParameter.prototype.shape = null;
nnabla.RandomCropParameter.prototype.base_axis = 0n;
nnabla.RandomCropParameter.prototype.seed = 0n;

nnabla.RandomFlipParameter = class RandomFlipParameter {

    constructor() {
        this.axes = [];
    }

    static decode(reader, length) {
        const message = new nnabla.RandomFlipParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RandomFlipParameter();
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

nnabla.RandomFlipParameter.prototype.base_axis = 0n;
nnabla.RandomFlipParameter.prototype.seed = 0n;

nnabla.RandomShiftParameter = class RandomShiftParameter {

    constructor() {
        this.shifts = [];
    }

    static decode(reader, length) {
        const message = new nnabla.RandomShiftParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RandomShiftParameter();
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

nnabla.RandomShiftParameter.prototype.border_mode = "";
nnabla.RandomShiftParameter.prototype.constant_value = 0;
nnabla.RandomShiftParameter.prototype.base_axis = 0n;
nnabla.RandomShiftParameter.prototype.seed = 0n;

nnabla.RandomEraseParameter = class RandomEraseParameter {

    constructor() {
        this.area_ratios = [];
        this.aspect_ratios = [];
        this.replacements = [];
    }

    static decode(reader, length) {
        const message = new nnabla.RandomEraseParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.RandomEraseParameter();
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

nnabla.RandomEraseParameter.prototype.prob = 0;
nnabla.RandomEraseParameter.prototype.n = 0n;
nnabla.RandomEraseParameter.prototype.share = false;
nnabla.RandomEraseParameter.prototype.inplace = false;
nnabla.RandomEraseParameter.prototype.base_axis = 0n;
nnabla.RandomEraseParameter.prototype.seed = 0n;
nnabla.RandomEraseParameter.prototype.channel_last = false;
nnabla.RandomEraseParameter.prototype.ste_fine_grained = false;

nnabla.ImageAugmentationParameter = class ImageAugmentationParameter {

    static decode(reader, length) {
        const message = new nnabla.ImageAugmentationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.ImageAugmentationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = nnabla.Shape.decodeText(reader);
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
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

nnabla.ImageAugmentationParameter.prototype.shape = null;
nnabla.ImageAugmentationParameter.prototype.pad = null;
nnabla.ImageAugmentationParameter.prototype.min_scale = 0;
nnabla.ImageAugmentationParameter.prototype.max_scale = 0;
nnabla.ImageAugmentationParameter.prototype.angle = 0;
nnabla.ImageAugmentationParameter.prototype.aspect_ratio = 0;
nnabla.ImageAugmentationParameter.prototype.distortion = 0;
nnabla.ImageAugmentationParameter.prototype.flip_lr = false;
nnabla.ImageAugmentationParameter.prototype.flip_ud = false;
nnabla.ImageAugmentationParameter.prototype.brightness = 0;
nnabla.ImageAugmentationParameter.prototype.brightness_each = false;
nnabla.ImageAugmentationParameter.prototype.contrast = 0;
nnabla.ImageAugmentationParameter.prototype.contrast_center = 0;
nnabla.ImageAugmentationParameter.prototype.contrast_each = false;
nnabla.ImageAugmentationParameter.prototype.noise = 0;
nnabla.ImageAugmentationParameter.prototype.seed = 0n;

nnabla.SoftmaxCrossEntropyParameter = class SoftmaxCrossEntropyParameter {

    static decode(reader, length) {
        const message = new nnabla.SoftmaxCrossEntropyParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SoftmaxCrossEntropyParameter();
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

nnabla.SoftmaxCrossEntropyParameter.prototype.axis = 0n;

nnabla.CategoricalCrossEntropyParameter = class CategoricalCrossEntropyParameter {

    static decode(reader, length) {
        const message = new nnabla.CategoricalCrossEntropyParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.CategoricalCrossEntropyParameter();
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

nnabla.CategoricalCrossEntropyParameter.prototype.axis = 0n;

nnabla.HuberLossParameter = class HuberLossParameter {

    static decode(reader, length) {
        const message = new nnabla.HuberLossParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.HuberLossParameter();
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

nnabla.HuberLossParameter.prototype.delta = 0;

nnabla.EpsilonInsensitiveLossParameter = class EpsilonInsensitiveLossParameter {

    static decode(reader, length) {
        const message = new nnabla.EpsilonInsensitiveLossParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.EpsilonInsensitiveLossParameter();
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

nnabla.EpsilonInsensitiveLossParameter.prototype.epsilon = 0;

nnabla.KLMultinomialParameter = class KLMultinomialParameter {

    static decode(reader, length) {
        const message = new nnabla.KLMultinomialParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.KLMultinomialParameter();
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

nnabla.KLMultinomialParameter.prototype.base_axis = 0n;

nnabla.AffineGridParameter = class AffineGridParameter {

    constructor() {
        this.size = [];
    }

    static decode(reader, length) {
        const message = new nnabla.AffineGridParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.AffineGridParameter();
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

nnabla.AffineGridParameter.prototype.align_corners = false;

nnabla.WarpByGridParameter = class WarpByGridParameter {

    static decode(reader, length) {
        const message = new nnabla.WarpByGridParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.WarpByGridParameter();
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

nnabla.WarpByGridParameter.prototype.mode = "";
nnabla.WarpByGridParameter.prototype.padding_mode = "";
nnabla.WarpByGridParameter.prototype.align_corners = false;
nnabla.WarpByGridParameter.prototype.channel_last = false;

nnabla.BinaryConnectAffineParameter = class BinaryConnectAffineParameter {

    static decode(reader, length) {
        const message = new nnabla.BinaryConnectAffineParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BinaryConnectAffineParameter();
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

nnabla.BinaryConnectAffineParameter.prototype.base_axis = 0n;
nnabla.BinaryConnectAffineParameter.prototype.quantize_zero_to = 0;

nnabla.BinaryConnectConvolutionParameter = class BinaryConnectConvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.BinaryConnectConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.BinaryConnectConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.BinaryConnectConvolutionParameter.prototype.base_axis = 0n;
nnabla.BinaryConnectConvolutionParameter.prototype.pad = null;
nnabla.BinaryConnectConvolutionParameter.prototype.stride = null;
nnabla.BinaryConnectConvolutionParameter.prototype.dilation = null;
nnabla.BinaryConnectConvolutionParameter.prototype.group = 0n;
nnabla.BinaryConnectConvolutionParameter.prototype.quantize_zero_to = 0;

nnabla.BinaryWeightAffineParameter = class BinaryWeightAffineParameter {

    static decode(reader, length) {
        const message = new nnabla.BinaryWeightAffineParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.BinaryWeightAffineParameter();
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

nnabla.BinaryWeightAffineParameter.prototype.base_axis = 0n;
nnabla.BinaryWeightAffineParameter.prototype.quantize_zero_to = 0;

nnabla.BinaryWeightConvolutionParameter = class BinaryWeightConvolutionParameter {

    static decode(reader, length) {
        const message = new nnabla.BinaryWeightConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.BinaryWeightConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.BinaryWeightConvolutionParameter.prototype.base_axis = 0n;
nnabla.BinaryWeightConvolutionParameter.prototype.pad = null;
nnabla.BinaryWeightConvolutionParameter.prototype.stride = null;
nnabla.BinaryWeightConvolutionParameter.prototype.dilation = null;
nnabla.BinaryWeightConvolutionParameter.prototype.group = 0n;
nnabla.BinaryWeightConvolutionParameter.prototype.quantize_zero_to = 0;

nnabla.INQAffineParameter = class INQAffineParameter {

    constructor() {
        this.inq_iterations = [];
    }

    static decode(reader, length) {
        const message = new nnabla.INQAffineParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.INQAffineParameter();
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

nnabla.INQAffineParameter.prototype.base_axis = 0n;
nnabla.INQAffineParameter.prototype.num_bits = 0n;
nnabla.INQAffineParameter.prototype.selection_algorithm = "";
nnabla.INQAffineParameter.prototype.seed = 0n;

nnabla.INQConvolutionParameter = class INQConvolutionParameter {

    constructor() {
        this.inq_iterations = [];
    }

    static decode(reader, length) {
        const message = new nnabla.INQConvolutionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.base_axis = reader.int64();
                    break;
                case 2:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dilation = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.INQConvolutionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "base_axis":
                    message.base_axis = reader.int64();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "dilation":
                    message.dilation = nnabla.Shape.decodeText(reader);
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

nnabla.INQConvolutionParameter.prototype.base_axis = 0n;
nnabla.INQConvolutionParameter.prototype.pad = null;
nnabla.INQConvolutionParameter.prototype.stride = null;
nnabla.INQConvolutionParameter.prototype.dilation = null;
nnabla.INQConvolutionParameter.prototype.group = 0n;
nnabla.INQConvolutionParameter.prototype.num_bits = 0n;
nnabla.INQConvolutionParameter.prototype.selection_algorithm = "";
nnabla.INQConvolutionParameter.prototype.seed = 0n;

nnabla.FixedPointQuantizeParameter = class FixedPointQuantizeParameter {

    static decode(reader, length) {
        const message = new nnabla.FixedPointQuantizeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.FixedPointQuantizeParameter();
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

nnabla.FixedPointQuantizeParameter.prototype.sign = false;
nnabla.FixedPointQuantizeParameter.prototype.n = 0n;
nnabla.FixedPointQuantizeParameter.prototype.delta = 0;
nnabla.FixedPointQuantizeParameter.prototype.ste_fine_grained = false;

nnabla.MinMaxQuantizeParameter = class MinMaxQuantizeParameter {

    static decode(reader, length) {
        const message = new nnabla.MinMaxQuantizeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.MinMaxQuantizeParameter();
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

nnabla.MinMaxQuantizeParameter.prototype.decay = 0;
nnabla.MinMaxQuantizeParameter.prototype.x_min_max = false;
nnabla.MinMaxQuantizeParameter.prototype.ema = false;
nnabla.MinMaxQuantizeParameter.prototype.ste_fine_grained = false;
nnabla.MinMaxQuantizeParameter.prototype.eps = 0;

nnabla.Pow2QuantizeParameter = class Pow2QuantizeParameter {

    static decode(reader, length) {
        const message = new nnabla.Pow2QuantizeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.Pow2QuantizeParameter();
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

nnabla.Pow2QuantizeParameter.prototype.sign = false;
nnabla.Pow2QuantizeParameter.prototype.with_zero = false;
nnabla.Pow2QuantizeParameter.prototype.n = 0n;
nnabla.Pow2QuantizeParameter.prototype.m = 0n;
nnabla.Pow2QuantizeParameter.prototype.ste_fine_grained = false;

nnabla.PruneParameter = class PruneParameter {

    static decode(reader, length) {
        const message = new nnabla.PruneParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.PruneParameter();
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

nnabla.PruneParameter.prototype.rate = 0;

nnabla.QuantizeLinearParameter = class QuantizeLinearParameter {

    static decode(reader, length) {
        const message = new nnabla.QuantizeLinearParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.QuantizeLinearParameter();
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

nnabla.QuantizeLinearParameter.prototype.round_mode = "";
nnabla.QuantizeLinearParameter.prototype.narrow_range = false;
nnabla.QuantizeLinearParameter.prototype.dtype = 0n;

nnabla.TopNErrorParameter = class TopNErrorParameter {

    static decode(reader, length) {
        const message = new nnabla.TopNErrorParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.TopNErrorParameter();
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

nnabla.TopNErrorParameter.prototype.axis = 0n;
nnabla.TopNErrorParameter.prototype.n = 0n;

nnabla.ConfusionMatrixParameter = class ConfusionMatrixParameter {

    static decode(reader, length) {
        const message = new nnabla.ConfusionMatrixParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.ConfusionMatrixParameter();
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

nnabla.ConfusionMatrixParameter.prototype.axis = 0n;

nnabla.VATNoiseParameter = class VATNoiseParameter {

    static decode(reader, length) {
        const message = new nnabla.VATNoiseParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.VATNoiseParameter();
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

nnabla.VATNoiseParameter.prototype.base_axis = 0n;
nnabla.VATNoiseParameter.prototype.eps = 0;

nnabla.SinkParameter = class SinkParameter {

    static decode(reader, length) {
        const message = new nnabla.SinkParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.SinkParameter();
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

nnabla.SinkParameter.prototype.one_input_grad = false;

nnabla.NmsDetection2dParameter = class NmsDetection2dParameter {

    static decode(reader, length) {
        const message = new nnabla.NmsDetection2dParameter();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new nnabla.NmsDetection2dParameter();
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

nnabla.NmsDetection2dParameter.prototype.thresh = 0;
nnabla.NmsDetection2dParameter.prototype.nms = 0;
nnabla.NmsDetection2dParameter.prototype.nms_per_class = false;

nnabla.ONNXNonMaxSuppressionParameter = class ONNXNonMaxSuppressionParameter {

    static decode(reader, length) {
        const message = new nnabla.ONNXNonMaxSuppressionParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.center_point_box = reader.int64();
                    break;
                case 2:
                    message.max_output_boxes_per_class = reader.int64();
                    break;
                case 3:
                    message.iou_threshold = reader.float();
                    break;
                case 4:
                    message.score_threshold = reader.float();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.ONNXNonMaxSuppressionParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "center_point_box":
                    message.center_point_box = reader.int64();
                    break;
                case "max_output_boxes_per_class":
                    message.max_output_boxes_per_class = reader.int64();
                    break;
                case "iou_threshold":
                    message.iou_threshold = reader.float();
                    break;
                case "score_threshold":
                    message.score_threshold = reader.float();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.ONNXNonMaxSuppressionParameter.prototype.center_point_box = 0n;
nnabla.ONNXNonMaxSuppressionParameter.prototype.max_output_boxes_per_class = 0n;
nnabla.ONNXNonMaxSuppressionParameter.prototype.iou_threshold = 0;
nnabla.ONNXNonMaxSuppressionParameter.prototype.score_threshold = 0;

nnabla.MaxPoolingBackwardParameter = class MaxPoolingBackwardParameter {

    static decode(reader, length) {
        const message = new nnabla.MaxPoolingBackwardParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.kernel = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.stride = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.ignore_border = reader.bool();
                    break;
                case 4:
                    message.pad = nnabla.Shape.decode(reader, reader.uint32());
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
        const message = new nnabla.MaxPoolingBackwardParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "kernel":
                    message.kernel = nnabla.Shape.decodeText(reader);
                    break;
                case "stride":
                    message.stride = nnabla.Shape.decodeText(reader);
                    break;
                case "ignore_border":
                    message.ignore_border = reader.bool();
                    break;
                case "pad":
                    message.pad = nnabla.Shape.decodeText(reader);
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

nnabla.MaxPoolingBackwardParameter.prototype.kernel = null;
nnabla.MaxPoolingBackwardParameter.prototype.stride = null;
nnabla.MaxPoolingBackwardParameter.prototype.ignore_border = false;
nnabla.MaxPoolingBackwardParameter.prototype.pad = null;
nnabla.MaxPoolingBackwardParameter.prototype.channel_last = false;

nnabla.PatchCorrelationParameter = class PatchCorrelationParameter {

    static decode(reader, length) {
        const message = new nnabla.PatchCorrelationParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.patch = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.shift = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.patch_step = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.shift_step = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.padding = nnabla.Shape.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.PatchCorrelationParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "patch":
                    message.patch = nnabla.Shape.decodeText(reader);
                    break;
                case "shift":
                    message.shift = nnabla.Shape.decodeText(reader);
                    break;
                case "patch_step":
                    message.patch_step = nnabla.Shape.decodeText(reader);
                    break;
                case "shift_step":
                    message.shift_step = nnabla.Shape.decodeText(reader);
                    break;
                case "padding":
                    message.padding = nnabla.Shape.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.PatchCorrelationParameter.prototype.patch = null;
nnabla.PatchCorrelationParameter.prototype.shift = null;
nnabla.PatchCorrelationParameter.prototype.patch_step = null;
nnabla.PatchCorrelationParameter.prototype.shift_step = null;
nnabla.PatchCorrelationParameter.prototype.padding = null;

nnabla.UniqueParameter = class UniqueParameter {

    static decode(reader, length) {
        const message = new nnabla.UniqueParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.flatten = reader.bool();
                    break;
                case 2:
                    message.axis = reader.int64();
                    break;
                case 3:
                    message.sorted = reader.bool();
                    break;
                case 4:
                    message.with_index = reader.bool();
                    break;
                case 5:
                    message.with_inverse = reader.bool();
                    break;
                case 6:
                    message.with_counts = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.UniqueParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "flatten":
                    message.flatten = reader.bool();
                    break;
                case "axis":
                    message.axis = reader.int64();
                    break;
                case "sorted":
                    message.sorted = reader.bool();
                    break;
                case "with_index":
                    message.with_index = reader.bool();
                    break;
                case "with_inverse":
                    message.with_inverse = reader.bool();
                    break;
                case "with_counts":
                    message.with_counts = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.UniqueParameter.prototype.flatten = false;
nnabla.UniqueParameter.prototype.axis = 0n;
nnabla.UniqueParameter.prototype.sorted = false;
nnabla.UniqueParameter.prototype.with_index = false;
nnabla.UniqueParameter.prototype.with_inverse = false;
nnabla.UniqueParameter.prototype.with_counts = false;

nnabla.EyeLikeParameter = class EyeLikeParameter {

    static decode(reader, length) {
        const message = new nnabla.EyeLikeParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.k = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.EyeLikeParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "k":
                    message.k = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.EyeLikeParameter.prototype.k = 0n;

nnabla.Mod2Parameter = class Mod2Parameter {

    static decode(reader, length) {
        const message = new nnabla.Mod2Parameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.fmod = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.Mod2Parameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "fmod":
                    message.fmod = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.Mod2Parameter.prototype.fmod = false;

nnabla.BitShiftParameter = class BitShiftParameter {

    static decode(reader, length) {
        const message = new nnabla.BitShiftParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.direction = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.BitShiftParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "direction":
                    message.direction = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.BitShiftParameter.prototype.direction = "";

nnabla.EinsumParameter = class EinsumParameter {

    static decode(reader, length) {
        const message = new nnabla.EinsumParameter();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.equation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new nnabla.EinsumParameter();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "equation":
                    message.equation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

nnabla.EinsumParameter.prototype.equation = "";
