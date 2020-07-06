(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('caffe');

    $root.caffe = (function() {

        const caffe = {};

        caffe.BlobShape = (function() {

            function BlobShape() {
                this.dim = [];
            }

            BlobShape.prototype.dim = [];

            BlobShape.decode = function (reader, length) {
                const message = new $root.caffe.BlobShape();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BlobShape.decodeText = function (reader) {
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
            };

            return BlobShape;
        })();

        caffe.BlobProto = (function() {

            function BlobProto() {
                this.data = [];
                this.diff = [];
                this.double_data = [];
                this.double_diff = [];
            }

            BlobProto.prototype.shape = null;
            BlobProto.prototype.data = [];
            BlobProto.prototype.diff = [];
            BlobProto.prototype.double_data = [];
            BlobProto.prototype.double_diff = [];
            BlobProto.prototype.num = 0;
            BlobProto.prototype.channels = 0;
            BlobProto.prototype.height = 0;
            BlobProto.prototype.width = 0;

            BlobProto.decode = function (reader, length) {
                const message = new $root.caffe.BlobProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BlobProto.decodeText = function (reader) {
                const message = new $root.caffe.BlobProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shape":
                            message.shape = $root.caffe.BlobShape.decodeText(reader, true);
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
            };

            return BlobProto;
        })();

        caffe.BlobProtoVector = (function() {

            function BlobProtoVector() {
                this.blobs = [];
            }

            BlobProtoVector.prototype.blobs = [];

            BlobProtoVector.decode = function (reader, length) {
                const message = new $root.caffe.BlobProtoVector();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BlobProtoVector.decodeText = function (reader) {
                const message = new $root.caffe.BlobProtoVector();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "blobs":
                            message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return BlobProtoVector;
        })();

        caffe.Datum = (function() {

            function Datum() {
                this.float_data = [];
            }

            Datum.prototype.channels = 0;
            Datum.prototype.height = 0;
            Datum.prototype.width = 0;
            Datum.prototype.data = new Uint8Array([]);
            Datum.prototype.label = 0;
            Datum.prototype.float_data = [];
            Datum.prototype.encoded = false;

            Datum.decode = function (reader, length) {
                const message = new $root.caffe.Datum();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            Datum.decodeText = function (reader) {
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
            };

            return Datum;
        })();

        caffe.FillerParameter = (function() {

            function FillerParameter() {
            }

            FillerParameter.prototype.type = "constant";
            FillerParameter.prototype.value = 0;
            FillerParameter.prototype.min = 0;
            FillerParameter.prototype.max = 1;
            FillerParameter.prototype.mean = 0;
            FillerParameter.prototype.std = 1;
            FillerParameter.prototype.sparse = -1;
            FillerParameter.prototype.variance_norm = 0;

            FillerParameter.decode = function (reader, length) {
                const message = new $root.caffe.FillerParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            FillerParameter.decodeText = function (reader) {
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
            };

            FillerParameter.VarianceNorm = (function() {
                const values = {};
                values["FAN_IN"] = 0;
                values["FAN_OUT"] = 1;
                values["AVERAGE"] = 2;
                return values;
            })();

            return FillerParameter;
        })();

        caffe.NetParameter = (function() {

            function NetParameter() {
                this.input = [];
                this.input_shape = [];
                this.input_dim = [];
                this.layer = [];
                this.layers = [];
            }

            NetParameter.prototype.name = "";
            NetParameter.prototype.input = [];
            NetParameter.prototype.input_shape = [];
            NetParameter.prototype.input_dim = [];
            NetParameter.prototype.force_backward = false;
            NetParameter.prototype.state = null;
            NetParameter.prototype.debug_info = false;
            NetParameter.prototype.layer = [];
            NetParameter.prototype.layers = [];

            NetParameter.decode = function (reader, length) {
                const message = new $root.caffe.NetParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            NetParameter.decodeText = function (reader) {
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
                            message.input_shape.push($root.caffe.BlobShape.decodeText(reader, true));
                            break;
                        case "input_dim":
                            reader.array(message.input_dim, () => reader.int32());
                            break;
                        case "force_backward":
                            message.force_backward = reader.bool();
                            break;
                        case "state":
                            message.state = $root.caffe.NetState.decodeText(reader, true);
                            break;
                        case "debug_info":
                            message.debug_info = reader.bool();
                            break;
                        case "layer":
                            message.layer.push($root.caffe.LayerParameter.decodeText(reader, true));
                            break;
                        case "layers":
                            message.layers.push($root.caffe.V1LayerParameter.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return NetParameter;
        })();

        caffe.SolverParameter = (function() {

            function SolverParameter() {
                this.test_net = [];
                this.test_net_param = [];
                this.test_state = [];
                this.test_iter = [];
                this.stepvalue = [];
                this.weights = [];
            }

            SolverParameter.prototype.net = "";
            SolverParameter.prototype.net_param = null;
            SolverParameter.prototype.train_net = "";
            SolverParameter.prototype.test_net = [];
            SolverParameter.prototype.train_net_param = null;
            SolverParameter.prototype.test_net_param = [];
            SolverParameter.prototype.train_state = null;
            SolverParameter.prototype.test_state = [];
            SolverParameter.prototype.test_iter = [];
            SolverParameter.prototype.test_interval = 0;
            SolverParameter.prototype.test_compute_loss = false;
            SolverParameter.prototype.test_initialization = true;
            SolverParameter.prototype.base_lr = 0;
            SolverParameter.prototype.display = 0;
            SolverParameter.prototype.average_loss = 1;
            SolverParameter.prototype.max_iter = 0;
            SolverParameter.prototype.iter_size = 1;
            SolverParameter.prototype.lr_policy = "";
            SolverParameter.prototype.gamma = 0;
            SolverParameter.prototype.power = 0;
            SolverParameter.prototype.momentum = 0;
            SolverParameter.prototype.weight_decay = 0;
            SolverParameter.prototype.regularization_type = "L2";
            SolverParameter.prototype.stepsize = 0;
            SolverParameter.prototype.stepvalue = [];
            SolverParameter.prototype.clip_gradients = -1;
            SolverParameter.prototype.snapshot = 0;
            SolverParameter.prototype.snapshot_prefix = "";
            SolverParameter.prototype.snapshot_diff = false;
            SolverParameter.prototype.snapshot_format = 1;
            SolverParameter.prototype.solver_mode = 1;
            SolverParameter.prototype.device_id = 0;
            SolverParameter.prototype.random_seed = $protobuf.Long ? $protobuf.Long.fromBits(-1, -1, false) : -1;
            SolverParameter.prototype.type = "SGD";
            SolverParameter.prototype.delta = 1e-8;
            SolverParameter.prototype.momentum2 = 0.999;
            SolverParameter.prototype.rms_decay = 0.99;
            SolverParameter.prototype.debug_info = false;
            SolverParameter.prototype.snapshot_after_train = true;
            SolverParameter.prototype.solver_type = 0;
            SolverParameter.prototype.layer_wise_reduce = true;
            SolverParameter.prototype.weights = [];

            SolverParameter.decode = function (reader, length) {
                const message = new $root.caffe.SolverParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SolverParameter.decodeText = function (reader) {
                const message = new $root.caffe.SolverParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "net":
                            message.net = reader.string();
                            break;
                        case "net_param":
                            message.net_param = $root.caffe.NetParameter.decodeText(reader, true);
                            break;
                        case "train_net":
                            message.train_net = reader.string();
                            break;
                        case "test_net":
                            reader.array(message.test_net, () => reader.string());
                            break;
                        case "train_net_param":
                            message.train_net_param = $root.caffe.NetParameter.decodeText(reader, true);
                            break;
                        case "test_net_param":
                            message.test_net_param.push($root.caffe.NetParameter.decodeText(reader, true));
                            break;
                        case "train_state":
                            message.train_state = $root.caffe.NetState.decodeText(reader, true);
                            break;
                        case "test_state":
                            message.test_state.push($root.caffe.NetState.decodeText(reader, true));
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
            };

            SolverParameter.SnapshotFormat = (function() {
                const values = {};
                values["HDF5"] = 0;
                values["BINARYPROTO"] = 1;
                return values;
            })();

            SolverParameter.SolverMode = (function() {
                const values = {};
                values["CPU"] = 0;
                values["GPU"] = 1;
                return values;
            })();

            SolverParameter.SolverType = (function() {
                const values = {};
                values["SGD"] = 0;
                values["NESTEROV"] = 1;
                values["ADAGRAD"] = 2;
                values["RMSPROP"] = 3;
                values["ADADELTA"] = 4;
                values["ADAM"] = 5;
                return values;
            })();

            return SolverParameter;
        })();

        caffe.SolverState = (function() {

            function SolverState() {
                this.history = [];
            }

            SolverState.prototype.iter = 0;
            SolverState.prototype.learned_net = "";
            SolverState.prototype.history = [];
            SolverState.prototype.current_step = 0;

            SolverState.decode = function (reader, length) {
                const message = new $root.caffe.SolverState();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SolverState.decodeText = function (reader) {
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
                            message.history.push($root.caffe.BlobProto.decodeText(reader, true));
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
            };

            return SolverState;
        })();

        caffe.Phase = (function() {
            const values = {};
            values["TRAIN"] = 0;
            values["TEST"] = 1;
            return values;
        })();

        caffe.NetState = (function() {

            function NetState() {
                this.stage = [];
            }

            NetState.prototype.phase = 1;
            NetState.prototype.level = 0;
            NetState.prototype.stage = [];

            NetState.decode = function (reader, length) {
                const message = new $root.caffe.NetState();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            NetState.decodeText = function (reader) {
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
            };

            return NetState;
        })();

        caffe.NetStateRule = (function() {

            function NetStateRule() {
                this.stage = [];
                this.not_stage = [];
            }

            NetStateRule.prototype.phase = 0;
            NetStateRule.prototype.min_level = 0;
            NetStateRule.prototype.max_level = 0;
            NetStateRule.prototype.stage = [];
            NetStateRule.prototype.not_stage = [];

            NetStateRule.decode = function (reader, length) {
                const message = new $root.caffe.NetStateRule();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            NetStateRule.decodeText = function (reader) {
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
            };

            return NetStateRule;
        })();

        caffe.ParamSpec = (function() {

            function ParamSpec() {
            }

            ParamSpec.prototype.name = "";
            ParamSpec.prototype.share_mode = 0;
            ParamSpec.prototype.lr_mult = 1;
            ParamSpec.prototype.decay_mult = 1;

            ParamSpec.decode = function (reader, length) {
                const message = new $root.caffe.ParamSpec();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ParamSpec.decodeText = function (reader) {
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
            };

            ParamSpec.DimCheckMode = (function() {
                const values = {};
                values["STRICT"] = 0;
                values["PERMISSIVE"] = 1;
                return values;
            })();

            return ParamSpec;
        })();

        caffe.LayerParameter = (function() {

            function LayerParameter() {
                this.bottom = [];
                this.top = [];
                this.loss_weight = [];
                this.param = [];
                this.blobs = [];
                this.propagate_down = [];
                this.include = [];
                this.exclude = [];
            }

            LayerParameter.prototype.name = "";
            LayerParameter.prototype.type = "";
            LayerParameter.prototype.bottom = [];
            LayerParameter.prototype.top = [];
            LayerParameter.prototype.phase = 0;
            LayerParameter.prototype.loss_weight = [];
            LayerParameter.prototype.param = [];
            LayerParameter.prototype.blobs = [];
            LayerParameter.prototype.propagate_down = [];
            LayerParameter.prototype.include = [];
            LayerParameter.prototype.exclude = [];
            LayerParameter.prototype.transform_param = null;
            LayerParameter.prototype.loss_param = null;
            LayerParameter.prototype.accuracy_param = null;
            LayerParameter.prototype.argmax_param = null;
            LayerParameter.prototype.batch_norm_param = null;
            LayerParameter.prototype.bias_param = null;
            LayerParameter.prototype.clip_param = null;
            LayerParameter.prototype.concat_param = null;
            LayerParameter.prototype.contrastive_loss_param = null;
            LayerParameter.prototype.convolution_param = null;
            LayerParameter.prototype.crop_param = null;
            LayerParameter.prototype.data_param = null;
            LayerParameter.prototype.dropout_param = null;
            LayerParameter.prototype.dummy_data_param = null;
            LayerParameter.prototype.eltwise_param = null;
            LayerParameter.prototype.elu_param = null;
            LayerParameter.prototype.embed_param = null;
            LayerParameter.prototype.exp_param = null;
            LayerParameter.prototype.flatten_param = null;
            LayerParameter.prototype.hdf5_data_param = null;
            LayerParameter.prototype.hdf5_output_param = null;
            LayerParameter.prototype.hinge_loss_param = null;
            LayerParameter.prototype.image_data_param = null;
            LayerParameter.prototype.infogain_loss_param = null;
            LayerParameter.prototype.inner_product_param = null;
            LayerParameter.prototype.input_param = null;
            LayerParameter.prototype.log_param = null;
            LayerParameter.prototype.lrn_param = null;
            LayerParameter.prototype.memory_data_param = null;
            LayerParameter.prototype.mvn_param = null;
            LayerParameter.prototype.parameter_param = null;
            LayerParameter.prototype.pooling_param = null;
            LayerParameter.prototype.power_param = null;
            LayerParameter.prototype.prelu_param = null;
            LayerParameter.prototype.python_param = null;
            LayerParameter.prototype.recurrent_param = null;
            LayerParameter.prototype.reduction_param = null;
            LayerParameter.prototype.relu_param = null;
            LayerParameter.prototype.reshape_param = null;
            LayerParameter.prototype.scale_param = null;
            LayerParameter.prototype.sigmoid_param = null;
            LayerParameter.prototype.softmax_param = null;
            LayerParameter.prototype.spp_param = null;
            LayerParameter.prototype.slice_param = null;
            LayerParameter.prototype.swish_param = null;
            LayerParameter.prototype.tanh_param = null;
            LayerParameter.prototype.threshold_param = null;
            LayerParameter.prototype.tile_param = null;
            LayerParameter.prototype.window_data_param = null;

            LayerParameter.decode = function (reader, length) {
                const message = new $root.caffe.LayerParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            LayerParameter.decodeText = function (reader) {
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
                            message.param.push($root.caffe.ParamSpec.decodeText(reader, true));
                            break;
                        case "blobs":
                            message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                            break;
                        case "propagate_down":
                            reader.array(message.propagate_down, () => reader.bool());
                            break;
                        case "include":
                            message.include.push($root.caffe.NetStateRule.decodeText(reader, true));
                            break;
                        case "exclude":
                            message.exclude.push($root.caffe.NetStateRule.decodeText(reader, true));
                            break;
                        case "transform_param":
                            message.transform_param = $root.caffe.TransformationParameter.decodeText(reader, true);
                            break;
                        case "loss_param":
                            message.loss_param = $root.caffe.LossParameter.decodeText(reader, true);
                            break;
                        case "accuracy_param":
                            message.accuracy_param = $root.caffe.AccuracyParameter.decodeText(reader, true);
                            break;
                        case "argmax_param":
                            message.argmax_param = $root.caffe.ArgMaxParameter.decodeText(reader, true);
                            break;
                        case "batch_norm_param":
                            message.batch_norm_param = $root.caffe.BatchNormParameter.decodeText(reader, true);
                            break;
                        case "bias_param":
                            message.bias_param = $root.caffe.BiasParameter.decodeText(reader, true);
                            break;
                        case "clip_param":
                            message.clip_param = $root.caffe.ClipParameter.decodeText(reader, true);
                            break;
                        case "concat_param":
                            message.concat_param = $root.caffe.ConcatParameter.decodeText(reader, true);
                            break;
                        case "contrastive_loss_param":
                            message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decodeText(reader, true);
                            break;
                        case "convolution_param":
                            message.convolution_param = $root.caffe.ConvolutionParameter.decodeText(reader, true);
                            break;
                        case "crop_param":
                            message.crop_param = $root.caffe.CropParameter.decodeText(reader, true);
                            break;
                        case "data_param":
                            message.data_param = $root.caffe.DataParameter.decodeText(reader, true);
                            break;
                        case "dropout_param":
                            message.dropout_param = $root.caffe.DropoutParameter.decodeText(reader, true);
                            break;
                        case "dummy_data_param":
                            message.dummy_data_param = $root.caffe.DummyDataParameter.decodeText(reader, true);
                            break;
                        case "eltwise_param":
                            message.eltwise_param = $root.caffe.EltwiseParameter.decodeText(reader, true);
                            break;
                        case "elu_param":
                            message.elu_param = $root.caffe.ELUParameter.decodeText(reader, true);
                            break;
                        case "embed_param":
                            message.embed_param = $root.caffe.EmbedParameter.decodeText(reader, true);
                            break;
                        case "exp_param":
                            message.exp_param = $root.caffe.ExpParameter.decodeText(reader, true);
                            break;
                        case "flatten_param":
                            message.flatten_param = $root.caffe.FlattenParameter.decodeText(reader, true);
                            break;
                        case "hdf5_data_param":
                            message.hdf5_data_param = $root.caffe.HDF5DataParameter.decodeText(reader, true);
                            break;
                        case "hdf5_output_param":
                            message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader, true);
                            break;
                        case "hinge_loss_param":
                            message.hinge_loss_param = $root.caffe.HingeLossParameter.decodeText(reader, true);
                            break;
                        case "image_data_param":
                            message.image_data_param = $root.caffe.ImageDataParameter.decodeText(reader, true);
                            break;
                        case "infogain_loss_param":
                            message.infogain_loss_param = $root.caffe.InfogainLossParameter.decodeText(reader, true);
                            break;
                        case "inner_product_param":
                            message.inner_product_param = $root.caffe.InnerProductParameter.decodeText(reader, true);
                            break;
                        case "input_param":
                            message.input_param = $root.caffe.InputParameter.decodeText(reader, true);
                            break;
                        case "log_param":
                            message.log_param = $root.caffe.LogParameter.decodeText(reader, true);
                            break;
                        case "lrn_param":
                            message.lrn_param = $root.caffe.LRNParameter.decodeText(reader, true);
                            break;
                        case "memory_data_param":
                            message.memory_data_param = $root.caffe.MemoryDataParameter.decodeText(reader, true);
                            break;
                        case "mvn_param":
                            message.mvn_param = $root.caffe.MVNParameter.decodeText(reader, true);
                            break;
                        case "parameter_param":
                            message.parameter_param = $root.caffe.ParameterParameter.decodeText(reader, true);
                            break;
                        case "pooling_param":
                            message.pooling_param = $root.caffe.PoolingParameter.decodeText(reader, true);
                            break;
                        case "power_param":
                            message.power_param = $root.caffe.PowerParameter.decodeText(reader, true);
                            break;
                        case "prelu_param":
                            message.prelu_param = $root.caffe.PReLUParameter.decodeText(reader, true);
                            break;
                        case "python_param":
                            message.python_param = $root.caffe.PythonParameter.decodeText(reader, true);
                            break;
                        case "recurrent_param":
                            message.recurrent_param = $root.caffe.RecurrentParameter.decodeText(reader, true);
                            break;
                        case "reduction_param":
                            message.reduction_param = $root.caffe.ReductionParameter.decodeText(reader, true);
                            break;
                        case "relu_param":
                            message.relu_param = $root.caffe.ReLUParameter.decodeText(reader, true);
                            break;
                        case "reshape_param":
                            message.reshape_param = $root.caffe.ReshapeParameter.decodeText(reader, true);
                            break;
                        case "scale_param":
                            message.scale_param = $root.caffe.ScaleParameter.decodeText(reader, true);
                            break;
                        case "sigmoid_param":
                            message.sigmoid_param = $root.caffe.SigmoidParameter.decodeText(reader, true);
                            break;
                        case "softmax_param":
                            message.softmax_param = $root.caffe.SoftmaxParameter.decodeText(reader, true);
                            break;
                        case "spp_param":
                            message.spp_param = $root.caffe.SPPParameter.decodeText(reader, true);
                            break;
                        case "slice_param":
                            message.slice_param = $root.caffe.SliceParameter.decodeText(reader, true);
                            break;
                        case "swish_param":
                            message.swish_param = $root.caffe.SwishParameter.decodeText(reader, true);
                            break;
                        case "tanh_param":
                            message.tanh_param = $root.caffe.TanHParameter.decodeText(reader, true);
                            break;
                        case "threshold_param":
                            message.threshold_param = $root.caffe.ThresholdParameter.decodeText(reader, true);
                            break;
                        case "tile_param":
                            message.tile_param = $root.caffe.TileParameter.decodeText(reader, true);
                            break;
                        case "window_data_param":
                            message.window_data_param = $root.caffe.WindowDataParameter.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return LayerParameter;
        })();

        caffe.TransformationParameter = (function() {

            function TransformationParameter() {
                this.mean_value = [];
            }

            TransformationParameter.prototype.scale = 1;
            TransformationParameter.prototype.mirror = false;
            TransformationParameter.prototype.crop_size = 0;
            TransformationParameter.prototype.mean_file = "";
            TransformationParameter.prototype.mean_value = [];
            TransformationParameter.prototype.force_color = false;
            TransformationParameter.prototype.force_gray = false;

            TransformationParameter.decode = function (reader, length) {
                const message = new $root.caffe.TransformationParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TransformationParameter.decodeText = function (reader) {
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
            };

            return TransformationParameter;
        })();

        caffe.LossParameter = (function() {

            function LossParameter() {
            }

            LossParameter.prototype.ignore_label = 0;
            LossParameter.prototype.normalization = 1;
            LossParameter.prototype.normalize = false;

            LossParameter.decode = function (reader, length) {
                const message = new $root.caffe.LossParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            LossParameter.decodeText = function (reader) {
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
            };

            LossParameter.NormalizationMode = (function() {
                const values = {};
                values["FULL"] = 0;
                values["VALID"] = 1;
                values["BATCH_SIZE"] = 2;
                values["NONE"] = 3;
                return values;
            })();

            return LossParameter;
        })();

        caffe.AccuracyParameter = (function() {

            function AccuracyParameter() {
            }

            AccuracyParameter.prototype.top_k = 1;
            AccuracyParameter.prototype.axis = 1;
            AccuracyParameter.prototype.ignore_label = 0;

            AccuracyParameter.decode = function (reader, length) {
                const message = new $root.caffe.AccuracyParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            AccuracyParameter.decodeText = function (reader) {
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
            };

            return AccuracyParameter;
        })();

        caffe.ArgMaxParameter = (function() {

            function ArgMaxParameter() {
            }

            ArgMaxParameter.prototype.out_max_val = false;
            ArgMaxParameter.prototype.top_k = 1;
            ArgMaxParameter.prototype.axis = 0;

            ArgMaxParameter.decode = function (reader, length) {
                const message = new $root.caffe.ArgMaxParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ArgMaxParameter.decodeText = function (reader) {
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
            };

            return ArgMaxParameter;
        })();

        caffe.ClipParameter = (function() {

            function ClipParameter() {
            }

            ClipParameter.prototype.min = 0;
            ClipParameter.prototype.max = 0;

            ClipParameter.decode = function (reader, length) {
                const message = new $root.caffe.ClipParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ClipParameter.decodeText = function (reader) {
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
            };

            return ClipParameter;
        })();

        caffe.ConcatParameter = (function() {

            function ConcatParameter() {
            }

            ConcatParameter.prototype.axis = 1;
            ConcatParameter.prototype.concat_dim = 1;

            ConcatParameter.decode = function (reader, length) {
                const message = new $root.caffe.ConcatParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ConcatParameter.decodeText = function (reader) {
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
            };

            return ConcatParameter;
        })();

        caffe.BatchNormParameter = (function() {

            function BatchNormParameter() {
            }

            BatchNormParameter.prototype.use_global_stats = false;
            BatchNormParameter.prototype.moving_average_fraction = 0.999;
            BatchNormParameter.prototype.eps = 0.00001;

            BatchNormParameter.decode = function (reader, length) {
                const message = new $root.caffe.BatchNormParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BatchNormParameter.decodeText = function (reader) {
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
            };

            return BatchNormParameter;
        })();

        caffe.BiasParameter = (function() {

            function BiasParameter() {
            }

            BiasParameter.prototype.axis = 1;
            BiasParameter.prototype.num_axes = 1;
            BiasParameter.prototype.filler = null;

            BiasParameter.decode = function (reader, length) {
                const message = new $root.caffe.BiasParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BiasParameter.decodeText = function (reader) {
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
                            message.filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return BiasParameter;
        })();

        caffe.ContrastiveLossParameter = (function() {

            function ContrastiveLossParameter() {
            }

            ContrastiveLossParameter.prototype.margin = 1;
            ContrastiveLossParameter.prototype.legacy_version = false;

            ContrastiveLossParameter.decode = function (reader, length) {
                const message = new $root.caffe.ContrastiveLossParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ContrastiveLossParameter.decodeText = function (reader) {
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
            };

            return ContrastiveLossParameter;
        })();

        caffe.ConvolutionParameter = (function() {

            function ConvolutionParameter() {
                this.pad = [];
                this.kernel_size = [];
                this.stride = [];
                this.dilation = [];
            }

            ConvolutionParameter.prototype.num_output = 0;
            ConvolutionParameter.prototype.bias_term = true;
            ConvolutionParameter.prototype.pad = [];
            ConvolutionParameter.prototype.kernel_size = [];
            ConvolutionParameter.prototype.stride = [];
            ConvolutionParameter.prototype.dilation = [];
            ConvolutionParameter.prototype.pad_h = 0;
            ConvolutionParameter.prototype.pad_w = 0;
            ConvolutionParameter.prototype.kernel_h = 0;
            ConvolutionParameter.prototype.kernel_w = 0;
            ConvolutionParameter.prototype.stride_h = 0;
            ConvolutionParameter.prototype.stride_w = 0;
            ConvolutionParameter.prototype.group = 1;
            ConvolutionParameter.prototype.weight_filler = null;
            ConvolutionParameter.prototype.bias_filler = null;
            ConvolutionParameter.prototype.engine = 0;
            ConvolutionParameter.prototype.axis = 1;
            ConvolutionParameter.prototype.force_nd_im2col = false;

            ConvolutionParameter.decode = function (reader, length) {
                const message = new $root.caffe.ConvolutionParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ConvolutionParameter.decodeText = function (reader) {
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
                            message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        case "bias_filler":
                            message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
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
            };

            ConvolutionParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return ConvolutionParameter;
        })();

        caffe.CropParameter = (function() {

            function CropParameter() {
                this.offset = [];
            }

            CropParameter.prototype.axis = 2;
            CropParameter.prototype.offset = [];

            CropParameter.decode = function (reader, length) {
                const message = new $root.caffe.CropParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            CropParameter.decodeText = function (reader) {
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
            };

            return CropParameter;
        })();

        caffe.DataParameter = (function() {

            function DataParameter() {
            }

            DataParameter.prototype.source = "";
            DataParameter.prototype.batch_size = 0;
            DataParameter.prototype.rand_skip = 0;
            DataParameter.prototype.backend = 0;
            DataParameter.prototype.scale = 1;
            DataParameter.prototype.mean_file = "";
            DataParameter.prototype.crop_size = 0;
            DataParameter.prototype.mirror = false;
            DataParameter.prototype.force_encoded_color = false;
            DataParameter.prototype.prefetch = 4;

            DataParameter.decode = function (reader, length) {
                const message = new $root.caffe.DataParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            DataParameter.decodeText = function (reader) {
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
            };

            DataParameter.DB = (function() {
                const values = {};
                values["LEVELDB"] = 0;
                values["LMDB"] = 1;
                return values;
            })();

            return DataParameter;
        })();

        caffe.DropoutParameter = (function() {

            function DropoutParameter() {
            }

            DropoutParameter.prototype.dropout_ratio = 0.5;

            DropoutParameter.decode = function (reader, length) {
                const message = new $root.caffe.DropoutParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            DropoutParameter.decodeText = function (reader) {
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
            };

            return DropoutParameter;
        })();

        caffe.DummyDataParameter = (function() {

            function DummyDataParameter() {
                this.data_filler = [];
                this.shape = [];
                this.num = [];
                this.channels = [];
                this.height = [];
                this.width = [];
            }

            DummyDataParameter.prototype.data_filler = [];
            DummyDataParameter.prototype.shape = [];
            DummyDataParameter.prototype.num = [];
            DummyDataParameter.prototype.channels = [];
            DummyDataParameter.prototype.height = [];
            DummyDataParameter.prototype.width = [];

            DummyDataParameter.decode = function (reader, length) {
                const message = new $root.caffe.DummyDataParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            DummyDataParameter.decodeText = function (reader) {
                const message = new $root.caffe.DummyDataParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "data_filler":
                            message.data_filler.push($root.caffe.FillerParameter.decodeText(reader, true));
                            break;
                        case "shape":
                            message.shape.push($root.caffe.BlobShape.decodeText(reader, true));
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
            };

            return DummyDataParameter;
        })();

        caffe.EltwiseParameter = (function() {

            function EltwiseParameter() {
                this.coeff = [];
            }

            EltwiseParameter.prototype.operation = 1;
            EltwiseParameter.prototype.coeff = [];
            EltwiseParameter.prototype.stable_prod_grad = true;

            EltwiseParameter.decode = function (reader, length) {
                const message = new $root.caffe.EltwiseParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            EltwiseParameter.decodeText = function (reader) {
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
            };

            EltwiseParameter.EltwiseOp = (function() {
                const values = {};
                values["PROD"] = 0;
                values["SUM"] = 1;
                values["MAX"] = 2;
                return values;
            })();

            return EltwiseParameter;
        })();

        caffe.ELUParameter = (function() {

            function ELUParameter() {
            }

            ELUParameter.prototype.alpha = 1;

            ELUParameter.decode = function (reader, length) {
                const message = new $root.caffe.ELUParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ELUParameter.decodeText = function (reader) {
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
            };

            return ELUParameter;
        })();

        caffe.EmbedParameter = (function() {

            function EmbedParameter() {
            }

            EmbedParameter.prototype.num_output = 0;
            EmbedParameter.prototype.input_dim = 0;
            EmbedParameter.prototype.bias_term = true;
            EmbedParameter.prototype.weight_filler = null;
            EmbedParameter.prototype.bias_filler = null;

            EmbedParameter.decode = function (reader, length) {
                const message = new $root.caffe.EmbedParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            EmbedParameter.decodeText = function (reader) {
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
                            message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        case "bias_filler":
                            message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return EmbedParameter;
        })();

        caffe.ExpParameter = (function() {

            function ExpParameter() {
            }

            ExpParameter.prototype.base = -1;
            ExpParameter.prototype.scale = 1;
            ExpParameter.prototype.shift = 0;

            ExpParameter.decode = function (reader, length) {
                const message = new $root.caffe.ExpParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ExpParameter.decodeText = function (reader) {
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
            };

            return ExpParameter;
        })();

        caffe.FlattenParameter = (function() {

            function FlattenParameter() {
            }

            FlattenParameter.prototype.axis = 1;
            FlattenParameter.prototype.end_axis = -1;

            FlattenParameter.decode = function (reader, length) {
                const message = new $root.caffe.FlattenParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            FlattenParameter.decodeText = function (reader) {
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
            };

            return FlattenParameter;
        })();

        caffe.HDF5DataParameter = (function() {

            function HDF5DataParameter() {
            }

            HDF5DataParameter.prototype.source = "";
            HDF5DataParameter.prototype.batch_size = 0;
            HDF5DataParameter.prototype.shuffle = false;

            HDF5DataParameter.decode = function (reader, length) {
                const message = new $root.caffe.HDF5DataParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            HDF5DataParameter.decodeText = function (reader) {
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
            };

            return HDF5DataParameter;
        })();

        caffe.HDF5OutputParameter = (function() {

            function HDF5OutputParameter() {
            }

            HDF5OutputParameter.prototype.file_name = "";

            HDF5OutputParameter.decode = function (reader, length) {
                const message = new $root.caffe.HDF5OutputParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            HDF5OutputParameter.decodeText = function (reader) {
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
            };

            return HDF5OutputParameter;
        })();

        caffe.HingeLossParameter = (function() {

            function HingeLossParameter() {
            }

            HingeLossParameter.prototype.norm = 1;

            HingeLossParameter.decode = function (reader, length) {
                const message = new $root.caffe.HingeLossParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            HingeLossParameter.decodeText = function (reader) {
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
            };

            HingeLossParameter.Norm = (function() {
                const values = {};
                values["L1"] = 1;
                values["L2"] = 2;
                return values;
            })();

            return HingeLossParameter;
        })();

        caffe.ImageDataParameter = (function() {

            function ImageDataParameter() {
            }

            ImageDataParameter.prototype.source = "";
            ImageDataParameter.prototype.batch_size = 1;
            ImageDataParameter.prototype.rand_skip = 0;
            ImageDataParameter.prototype.shuffle = false;
            ImageDataParameter.prototype.new_height = 0;
            ImageDataParameter.prototype.new_width = 0;
            ImageDataParameter.prototype.is_color = true;
            ImageDataParameter.prototype.scale = 1;
            ImageDataParameter.prototype.mean_file = "";
            ImageDataParameter.prototype.crop_size = 0;
            ImageDataParameter.prototype.mirror = false;
            ImageDataParameter.prototype.root_folder = "";

            ImageDataParameter.decode = function (reader, length) {
                const message = new $root.caffe.ImageDataParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ImageDataParameter.decodeText = function (reader) {
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
            };

            return ImageDataParameter;
        })();

        caffe.InfogainLossParameter = (function() {

            function InfogainLossParameter() {
            }

            InfogainLossParameter.prototype.source = "";
            InfogainLossParameter.prototype.axis = 1;

            InfogainLossParameter.decode = function (reader, length) {
                const message = new $root.caffe.InfogainLossParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            InfogainLossParameter.decodeText = function (reader) {
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
            };

            return InfogainLossParameter;
        })();

        caffe.InnerProductParameter = (function() {

            function InnerProductParameter() {
            }

            InnerProductParameter.prototype.num_output = 0;
            InnerProductParameter.prototype.bias_term = true;
            InnerProductParameter.prototype.weight_filler = null;
            InnerProductParameter.prototype.bias_filler = null;
            InnerProductParameter.prototype.axis = 1;
            InnerProductParameter.prototype.transpose = false;

            InnerProductParameter.decode = function (reader, length) {
                const message = new $root.caffe.InnerProductParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            InnerProductParameter.decodeText = function (reader) {
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
                            message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        case "bias_filler":
                            message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
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
            };

            return InnerProductParameter;
        })();

        caffe.InputParameter = (function() {

            function InputParameter() {
                this.shape = [];
            }

            InputParameter.prototype.shape = [];

            InputParameter.decode = function (reader, length) {
                const message = new $root.caffe.InputParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            InputParameter.decodeText = function (reader) {
                const message = new $root.caffe.InputParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shape":
                            message.shape.push($root.caffe.BlobShape.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return InputParameter;
        })();

        caffe.LogParameter = (function() {

            function LogParameter() {
            }

            LogParameter.prototype.base = -1;
            LogParameter.prototype.scale = 1;
            LogParameter.prototype.shift = 0;

            LogParameter.decode = function (reader, length) {
                const message = new $root.caffe.LogParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            LogParameter.decodeText = function (reader) {
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
            };

            return LogParameter;
        })();

        caffe.LRNParameter = (function() {

            function LRNParameter() {
            }

            LRNParameter.prototype.local_size = 5;
            LRNParameter.prototype.alpha = 1;
            LRNParameter.prototype.beta = 0.75;
            LRNParameter.prototype.norm_region = 0;
            LRNParameter.prototype.k = 1;
            LRNParameter.prototype.engine = 0;

            LRNParameter.decode = function (reader, length) {
                const message = new $root.caffe.LRNParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            LRNParameter.decodeText = function (reader) {
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
            };

            LRNParameter.NormRegion = (function() {
                const values = {};
                values["ACROSS_CHANNELS"] = 0;
                values["WITHIN_CHANNEL"] = 1;
                return values;
            })();

            LRNParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return LRNParameter;
        })();

        caffe.MemoryDataParameter = (function() {

            function MemoryDataParameter() {
            }

            MemoryDataParameter.prototype.batch_size = 0;
            MemoryDataParameter.prototype.channels = 0;
            MemoryDataParameter.prototype.height = 0;
            MemoryDataParameter.prototype.width = 0;

            MemoryDataParameter.decode = function (reader, length) {
                const message = new $root.caffe.MemoryDataParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            MemoryDataParameter.decodeText = function (reader) {
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
            };

            return MemoryDataParameter;
        })();

        caffe.MVNParameter = (function() {

            function MVNParameter() {
            }

            MVNParameter.prototype.normalize_variance = true;
            MVNParameter.prototype.across_channels = false;
            MVNParameter.prototype.eps = 1e-9;

            MVNParameter.decode = function (reader, length) {
                const message = new $root.caffe.MVNParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            MVNParameter.decodeText = function (reader) {
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
            };

            return MVNParameter;
        })();

        caffe.ParameterParameter = (function() {

            function ParameterParameter() {
            }

            ParameterParameter.prototype.shape = null;

            ParameterParameter.decode = function (reader, length) {
                const message = new $root.caffe.ParameterParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ParameterParameter.decodeText = function (reader) {
                const message = new $root.caffe.ParameterParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shape":
                            message.shape = $root.caffe.BlobShape.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return ParameterParameter;
        })();

        caffe.PoolingParameter = (function() {

            function PoolingParameter() {
            }

            PoolingParameter.prototype.pool = 0;
            PoolingParameter.prototype.pad = 0;
            PoolingParameter.prototype.pad_h = 0;
            PoolingParameter.prototype.pad_w = 0;
            PoolingParameter.prototype.kernel_size = 0;
            PoolingParameter.prototype.kernel_h = 0;
            PoolingParameter.prototype.kernel_w = 0;
            PoolingParameter.prototype.stride = 1;
            PoolingParameter.prototype.stride_h = 0;
            PoolingParameter.prototype.stride_w = 0;
            PoolingParameter.prototype.engine = 0;
            PoolingParameter.prototype.global_pooling = false;
            PoolingParameter.prototype.round_mode = 0;

            PoolingParameter.decode = function (reader, length) {
                const message = new $root.caffe.PoolingParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            PoolingParameter.decodeText = function (reader) {
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
            };

            PoolingParameter.PoolMethod = (function() {
                const values = {};
                values["MAX"] = 0;
                values["AVE"] = 1;
                values["STOCHASTIC"] = 2;
                return values;
            })();

            PoolingParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            PoolingParameter.RoundMode = (function() {
                const values = {};
                values["CEIL"] = 0;
                values["FLOOR"] = 1;
                return values;
            })();

            return PoolingParameter;
        })();

        caffe.PowerParameter = (function() {

            function PowerParameter() {
            }

            PowerParameter.prototype.power = 1;
            PowerParameter.prototype.scale = 1;
            PowerParameter.prototype.shift = 0;

            PowerParameter.decode = function (reader, length) {
                const message = new $root.caffe.PowerParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            PowerParameter.decodeText = function (reader) {
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
            };

            return PowerParameter;
        })();

        caffe.PythonParameter = (function() {

            function PythonParameter() {
            }

            PythonParameter.prototype.module = "";
            PythonParameter.prototype.layer = "";
            PythonParameter.prototype.param_str = "";
            PythonParameter.prototype.share_in_parallel = false;

            PythonParameter.decode = function (reader, length) {
                const message = new $root.caffe.PythonParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            PythonParameter.decodeText = function (reader) {
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
            };

            return PythonParameter;
        })();

        caffe.RecurrentParameter = (function() {

            function RecurrentParameter() {
            }

            RecurrentParameter.prototype.num_output = 0;
            RecurrentParameter.prototype.weight_filler = null;
            RecurrentParameter.prototype.bias_filler = null;
            RecurrentParameter.prototype.debug_info = false;
            RecurrentParameter.prototype.expose_hidden = false;

            RecurrentParameter.decode = function (reader, length) {
                const message = new $root.caffe.RecurrentParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            RecurrentParameter.decodeText = function (reader) {
                const message = new $root.caffe.RecurrentParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "num_output":
                            message.num_output = reader.uint32();
                            break;
                        case "weight_filler":
                            message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        case "bias_filler":
                            message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
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
            };

            return RecurrentParameter;
        })();

        caffe.ReductionParameter = (function() {

            function ReductionParameter() {
            }

            ReductionParameter.prototype.operation = 1;
            ReductionParameter.prototype.axis = 0;
            ReductionParameter.prototype.coeff = 1;

            ReductionParameter.decode = function (reader, length) {
                const message = new $root.caffe.ReductionParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ReductionParameter.decodeText = function (reader) {
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
            };

            ReductionParameter.ReductionOp = (function() {
                const values = {};
                values["SUM"] = 1;
                values["ASUM"] = 2;
                values["SUMSQ"] = 3;
                values["MEAN"] = 4;
                return values;
            })();

            return ReductionParameter;
        })();

        caffe.ReLUParameter = (function() {

            function ReLUParameter() {
            }

            ReLUParameter.prototype.negative_slope = 0;
            ReLUParameter.prototype.engine = 0;

            ReLUParameter.decode = function (reader, length) {
                const message = new $root.caffe.ReLUParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ReLUParameter.decodeText = function (reader) {
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
            };

            ReLUParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return ReLUParameter;
        })();

        caffe.ReshapeParameter = (function() {

            function ReshapeParameter() {
            }

            ReshapeParameter.prototype.shape = null;
            ReshapeParameter.prototype.axis = 0;
            ReshapeParameter.prototype.num_axes = -1;

            ReshapeParameter.decode = function (reader, length) {
                const message = new $root.caffe.ReshapeParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ReshapeParameter.decodeText = function (reader) {
                const message = new $root.caffe.ReshapeParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shape":
                            message.shape = $root.caffe.BlobShape.decodeText(reader, true);
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
            };

            return ReshapeParameter;
        })();

        caffe.ScaleParameter = (function() {

            function ScaleParameter() {
            }

            ScaleParameter.prototype.axis = 1;
            ScaleParameter.prototype.num_axes = 1;
            ScaleParameter.prototype.filler = null;
            ScaleParameter.prototype.bias_term = false;
            ScaleParameter.prototype.bias_filler = null;

            ScaleParameter.decode = function (reader, length) {
                const message = new $root.caffe.ScaleParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ScaleParameter.decodeText = function (reader) {
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
                            message.filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        case "bias_term":
                            message.bias_term = reader.bool();
                            break;
                        case "bias_filler":
                            message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return ScaleParameter;
        })();

        caffe.SigmoidParameter = (function() {

            function SigmoidParameter() {
            }

            SigmoidParameter.prototype.engine = 0;

            SigmoidParameter.decode = function (reader, length) {
                const message = new $root.caffe.SigmoidParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SigmoidParameter.decodeText = function (reader) {
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
            };

            SigmoidParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return SigmoidParameter;
        })();

        caffe.SliceParameter = (function() {

            function SliceParameter() {
                this.slice_point = [];
            }

            SliceParameter.prototype.axis = 1;
            SliceParameter.prototype.slice_point = [];
            SliceParameter.prototype.slice_dim = 1;

            SliceParameter.decode = function (reader, length) {
                const message = new $root.caffe.SliceParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SliceParameter.decodeText = function (reader) {
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
            };

            return SliceParameter;
        })();

        caffe.SoftmaxParameter = (function() {

            function SoftmaxParameter() {
            }

            SoftmaxParameter.prototype.engine = 0;
            SoftmaxParameter.prototype.axis = 1;

            SoftmaxParameter.decode = function (reader, length) {
                const message = new $root.caffe.SoftmaxParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SoftmaxParameter.decodeText = function (reader) {
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
            };

            SoftmaxParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return SoftmaxParameter;
        })();

        caffe.SwishParameter = (function() {

            function SwishParameter() {
            }

            SwishParameter.prototype.beta = 1;

            SwishParameter.decode = function (reader, length) {
                const message = new $root.caffe.SwishParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SwishParameter.decodeText = function (reader) {
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
            };

            return SwishParameter;
        })();

        caffe.TanHParameter = (function() {

            function TanHParameter() {
            }

            TanHParameter.prototype.engine = 0;

            TanHParameter.decode = function (reader, length) {
                const message = new $root.caffe.TanHParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TanHParameter.decodeText = function (reader) {
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
            };

            TanHParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return TanHParameter;
        })();

        caffe.TileParameter = (function() {

            function TileParameter() {
            }

            TileParameter.prototype.axis = 1;
            TileParameter.prototype.tiles = 0;

            TileParameter.decode = function (reader, length) {
                const message = new $root.caffe.TileParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TileParameter.decodeText = function (reader) {
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
            };

            return TileParameter;
        })();

        caffe.ThresholdParameter = (function() {

            function ThresholdParameter() {
            }

            ThresholdParameter.prototype.threshold = 0;

            ThresholdParameter.decode = function (reader, length) {
                const message = new $root.caffe.ThresholdParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ThresholdParameter.decodeText = function (reader) {
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
            };

            return ThresholdParameter;
        })();

        caffe.WindowDataParameter = (function() {

            function WindowDataParameter() {
            }

            WindowDataParameter.prototype.source = "";
            WindowDataParameter.prototype.scale = 1;
            WindowDataParameter.prototype.mean_file = "";
            WindowDataParameter.prototype.batch_size = 0;
            WindowDataParameter.prototype.crop_size = 0;
            WindowDataParameter.prototype.mirror = false;
            WindowDataParameter.prototype.fg_threshold = 0.5;
            WindowDataParameter.prototype.bg_threshold = 0.5;
            WindowDataParameter.prototype.fg_fraction = 0.25;
            WindowDataParameter.prototype.context_pad = 0;
            WindowDataParameter.prototype.crop_mode = "warp";
            WindowDataParameter.prototype.cache_images = false;
            WindowDataParameter.prototype.root_folder = "";

            WindowDataParameter.decode = function (reader, length) {
                const message = new $root.caffe.WindowDataParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            WindowDataParameter.decodeText = function (reader) {
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
            };

            return WindowDataParameter;
        })();

        caffe.SPPParameter = (function() {

            function SPPParameter() {
            }

            SPPParameter.prototype.pyramid_height = 0;
            SPPParameter.prototype.pool = 0;
            SPPParameter.prototype.engine = 0;

            SPPParameter.decode = function (reader, length) {
                const message = new $root.caffe.SPPParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SPPParameter.decodeText = function (reader) {
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
            };

            SPPParameter.PoolMethod = (function() {
                const values = {};
                values["MAX"] = 0;
                values["AVE"] = 1;
                values["STOCHASTIC"] = 2;
                return values;
            })();

            SPPParameter.Engine = (function() {
                const values = {};
                values["DEFAULT"] = 0;
                values["CAFFE"] = 1;
                values["CUDNN"] = 2;
                return values;
            })();

            return SPPParameter;
        })();

        caffe.V1LayerParameter = (function() {

            function V1LayerParameter() {
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

            V1LayerParameter.prototype.bottom = [];
            V1LayerParameter.prototype.top = [];
            V1LayerParameter.prototype.name = "";
            V1LayerParameter.prototype.include = [];
            V1LayerParameter.prototype.exclude = [];
            V1LayerParameter.prototype.type = 0;
            V1LayerParameter.prototype.blobs = [];
            V1LayerParameter.prototype.param = [];
            V1LayerParameter.prototype.blob_share_mode = [];
            V1LayerParameter.prototype.blobs_lr = [];
            V1LayerParameter.prototype.weight_decay = [];
            V1LayerParameter.prototype.loss_weight = [];
            V1LayerParameter.prototype.accuracy_param = null;
            V1LayerParameter.prototype.argmax_param = null;
            V1LayerParameter.prototype.concat_param = null;
            V1LayerParameter.prototype.contrastive_loss_param = null;
            V1LayerParameter.prototype.convolution_param = null;
            V1LayerParameter.prototype.data_param = null;
            V1LayerParameter.prototype.dropout_param = null;
            V1LayerParameter.prototype.dummy_data_param = null;
            V1LayerParameter.prototype.eltwise_param = null;
            V1LayerParameter.prototype.exp_param = null;
            V1LayerParameter.prototype.hdf5_data_param = null;
            V1LayerParameter.prototype.hdf5_output_param = null;
            V1LayerParameter.prototype.hinge_loss_param = null;
            V1LayerParameter.prototype.image_data_param = null;
            V1LayerParameter.prototype.infogain_loss_param = null;
            V1LayerParameter.prototype.inner_product_param = null;
            V1LayerParameter.prototype.lrn_param = null;
            V1LayerParameter.prototype.memory_data_param = null;
            V1LayerParameter.prototype.mvn_param = null;
            V1LayerParameter.prototype.pooling_param = null;
            V1LayerParameter.prototype.power_param = null;
            V1LayerParameter.prototype.relu_param = null;
            V1LayerParameter.prototype.sigmoid_param = null;
            V1LayerParameter.prototype.softmax_param = null;
            V1LayerParameter.prototype.slice_param = null;
            V1LayerParameter.prototype.tanh_param = null;
            V1LayerParameter.prototype.threshold_param = null;
            V1LayerParameter.prototype.window_data_param = null;
            V1LayerParameter.prototype.transform_param = null;
            V1LayerParameter.prototype.loss_param = null;
            V1LayerParameter.prototype.layer = null;

            V1LayerParameter.decode = function (reader, length) {
                const message = new $root.caffe.V1LayerParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            V1LayerParameter.decodeText = function (reader) {
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
                            message.include.push($root.caffe.NetStateRule.decodeText(reader, true));
                            break;
                        case "exclude":
                            message.exclude.push($root.caffe.NetStateRule.decodeText(reader, true));
                            break;
                        case "type":
                            message.type = reader.enum($root.caffe.V1LayerParameter.LayerType);
                            break;
                        case "blobs":
                            message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
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
                            message.accuracy_param = $root.caffe.AccuracyParameter.decodeText(reader, true);
                            break;
                        case "argmax_param":
                            message.argmax_param = $root.caffe.ArgMaxParameter.decodeText(reader, true);
                            break;
                        case "concat_param":
                            message.concat_param = $root.caffe.ConcatParameter.decodeText(reader, true);
                            break;
                        case "contrastive_loss_param":
                            message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decodeText(reader, true);
                            break;
                        case "convolution_param":
                            message.convolution_param = $root.caffe.ConvolutionParameter.decodeText(reader, true);
                            break;
                        case "data_param":
                            message.data_param = $root.caffe.DataParameter.decodeText(reader, true);
                            break;
                        case "dropout_param":
                            message.dropout_param = $root.caffe.DropoutParameter.decodeText(reader, true);
                            break;
                        case "dummy_data_param":
                            message.dummy_data_param = $root.caffe.DummyDataParameter.decodeText(reader, true);
                            break;
                        case "eltwise_param":
                            message.eltwise_param = $root.caffe.EltwiseParameter.decodeText(reader, true);
                            break;
                        case "exp_param":
                            message.exp_param = $root.caffe.ExpParameter.decodeText(reader, true);
                            break;
                        case "hdf5_data_param":
                            message.hdf5_data_param = $root.caffe.HDF5DataParameter.decodeText(reader, true);
                            break;
                        case "hdf5_output_param":
                            message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader, true);
                            break;
                        case "hinge_loss_param":
                            message.hinge_loss_param = $root.caffe.HingeLossParameter.decodeText(reader, true);
                            break;
                        case "image_data_param":
                            message.image_data_param = $root.caffe.ImageDataParameter.decodeText(reader, true);
                            break;
                        case "infogain_loss_param":
                            message.infogain_loss_param = $root.caffe.InfogainLossParameter.decodeText(reader, true);
                            break;
                        case "inner_product_param":
                            message.inner_product_param = $root.caffe.InnerProductParameter.decodeText(reader, true);
                            break;
                        case "lrn_param":
                            message.lrn_param = $root.caffe.LRNParameter.decodeText(reader, true);
                            break;
                        case "memory_data_param":
                            message.memory_data_param = $root.caffe.MemoryDataParameter.decodeText(reader, true);
                            break;
                        case "mvn_param":
                            message.mvn_param = $root.caffe.MVNParameter.decodeText(reader, true);
                            break;
                        case "pooling_param":
                            message.pooling_param = $root.caffe.PoolingParameter.decodeText(reader, true);
                            break;
                        case "power_param":
                            message.power_param = $root.caffe.PowerParameter.decodeText(reader, true);
                            break;
                        case "relu_param":
                            message.relu_param = $root.caffe.ReLUParameter.decodeText(reader, true);
                            break;
                        case "sigmoid_param":
                            message.sigmoid_param = $root.caffe.SigmoidParameter.decodeText(reader, true);
                            break;
                        case "softmax_param":
                            message.softmax_param = $root.caffe.SoftmaxParameter.decodeText(reader, true);
                            break;
                        case "slice_param":
                            message.slice_param = $root.caffe.SliceParameter.decodeText(reader, true);
                            break;
                        case "tanh_param":
                            message.tanh_param = $root.caffe.TanHParameter.decodeText(reader, true);
                            break;
                        case "threshold_param":
                            message.threshold_param = $root.caffe.ThresholdParameter.decodeText(reader, true);
                            break;
                        case "window_data_param":
                            message.window_data_param = $root.caffe.WindowDataParameter.decodeText(reader, true);
                            break;
                        case "transform_param":
                            message.transform_param = $root.caffe.TransformationParameter.decodeText(reader, true);
                            break;
                        case "loss_param":
                            message.loss_param = $root.caffe.LossParameter.decodeText(reader, true);
                            break;
                        case "layer":
                            message.layer = $root.caffe.V0LayerParameter.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            V1LayerParameter.LayerType = (function() {
                const values = {};
                values["NONE"] = 0;
                values["ABSVAL"] = 35;
                values["ACCURACY"] = 1;
                values["ARGMAX"] = 30;
                values["BNLL"] = 2;
                values["CONCAT"] = 3;
                values["CONTRASTIVE_LOSS"] = 37;
                values["CONVOLUTION"] = 4;
                values["DATA"] = 5;
                values["DECONVOLUTION"] = 39;
                values["DROPOUT"] = 6;
                values["DUMMY_DATA"] = 32;
                values["EUCLIDEAN_LOSS"] = 7;
                values["ELTWISE"] = 25;
                values["EXP"] = 38;
                values["FLATTEN"] = 8;
                values["HDF5_DATA"] = 9;
                values["HDF5_OUTPUT"] = 10;
                values["HINGE_LOSS"] = 28;
                values["IM2COL"] = 11;
                values["IMAGE_DATA"] = 12;
                values["INFOGAIN_LOSS"] = 13;
                values["INNER_PRODUCT"] = 14;
                values["LRN"] = 15;
                values["MEMORY_DATA"] = 29;
                values["MULTINOMIAL_LOGISTIC_LOSS"] = 16;
                values["MVN"] = 34;
                values["POOLING"] = 17;
                values["POWER"] = 26;
                values["RELU"] = 18;
                values["SIGMOID"] = 19;
                values["SIGMOID_CROSS_ENTROPY_LOSS"] = 27;
                values["SILENCE"] = 36;
                values["SOFTMAX"] = 20;
                values["SOFTMAX_LOSS"] = 21;
                values["SPLIT"] = 22;
                values["SLICE"] = 33;
                values["TANH"] = 23;
                values["WINDOW_DATA"] = 24;
                values["THRESHOLD"] = 31;
                return values;
            })();

            V1LayerParameter.DimCheckMode = (function() {
                const values = {};
                values["STRICT"] = 0;
                values["PERMISSIVE"] = 1;
                return values;
            })();

            return V1LayerParameter;
        })();

        caffe.V0LayerParameter = (function() {

            function V0LayerParameter() {
                this.blobs = [];
                this.blobs_lr = [];
                this.weight_decay = [];
            }

            V0LayerParameter.prototype.name = "";
            V0LayerParameter.prototype.type = "";
            V0LayerParameter.prototype.num_output = 0;
            V0LayerParameter.prototype.biasterm = true;
            V0LayerParameter.prototype.weight_filler = null;
            V0LayerParameter.prototype.bias_filler = null;
            V0LayerParameter.prototype.pad = 0;
            V0LayerParameter.prototype.kernelsize = 0;
            V0LayerParameter.prototype.group = 1;
            V0LayerParameter.prototype.stride = 1;
            V0LayerParameter.prototype.pool = 0;
            V0LayerParameter.prototype.dropout_ratio = 0.5;
            V0LayerParameter.prototype.local_size = 5;
            V0LayerParameter.prototype.alpha = 1;
            V0LayerParameter.prototype.beta = 0.75;
            V0LayerParameter.prototype.k = 1;
            V0LayerParameter.prototype.source = "";
            V0LayerParameter.prototype.scale = 1;
            V0LayerParameter.prototype.meanfile = "";
            V0LayerParameter.prototype.batchsize = 0;
            V0LayerParameter.prototype.cropsize = 0;
            V0LayerParameter.prototype.mirror = false;
            V0LayerParameter.prototype.blobs = [];
            V0LayerParameter.prototype.blobs_lr = [];
            V0LayerParameter.prototype.weight_decay = [];
            V0LayerParameter.prototype.rand_skip = 0;
            V0LayerParameter.prototype.det_fg_threshold = 0.5;
            V0LayerParameter.prototype.det_bg_threshold = 0.5;
            V0LayerParameter.prototype.det_fg_fraction = 0.25;
            V0LayerParameter.prototype.det_context_pad = 0;
            V0LayerParameter.prototype.det_crop_mode = "warp";
            V0LayerParameter.prototype.new_num = 0;
            V0LayerParameter.prototype.new_channels = 0;
            V0LayerParameter.prototype.new_height = 0;
            V0LayerParameter.prototype.new_width = 0;
            V0LayerParameter.prototype.shuffle_images = false;
            V0LayerParameter.prototype.concat_dim = 1;
            V0LayerParameter.prototype.hdf5_output_param = null;

            V0LayerParameter.decode = function (reader, length) {
                const message = new $root.caffe.V0LayerParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            V0LayerParameter.decodeText = function (reader) {
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
                            message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                            break;
                        case "bias_filler":
                            message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
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
                            message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
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
                            message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            V0LayerParameter.PoolMethod = (function() {
                const values = {};
                values["MAX"] = 0;
                values["AVE"] = 1;
                values["STOCHASTIC"] = 2;
                return values;
            })();

            return V0LayerParameter;
        })();

        caffe.PReLUParameter = (function() {

            function PReLUParameter() {
            }

            PReLUParameter.prototype.filler = null;
            PReLUParameter.prototype.channel_shared = false;

            PReLUParameter.decode = function (reader, length) {
                const message = new $root.caffe.PReLUParameter();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            PReLUParameter.decodeText = function (reader) {
                const message = new $root.caffe.PReLUParameter();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "filler":
                            message.filler = $root.caffe.FillerParameter.decodeText(reader, true);
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
            };

            return PReLUParameter;
        })();

        return caffe;
    })();
    return $root;
})(protobuf);
