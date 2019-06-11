/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.caffe || ($protobuf.roots.caffe = {});
    
    $root.caffe = (function() {
    
        var caffe = {};
    
        caffe.BlobShape = (function() {
    
            function BlobShape(properties) {
                this.dim = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobShape.prototype.dim = $util.emptyArray;
    
            BlobShape.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BlobShape();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dim.push(reader.int64());
                        } else
                            message.dim.push(reader.int64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BlobShape.decodeText = function decodeText(reader) {
                var message = new $root.caffe.BlobShape();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dim":
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dim.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.dim.push(reader.int64());
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
    
            function BlobProto(properties) {
                this.data = [];
                this.diff = [];
                this.double_data = [];
                this.double_diff = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobProto.prototype.shape = null;
            BlobProto.prototype.data = $util.emptyArray;
            BlobProto.prototype.diff = $util.emptyArray;
            BlobProto.prototype.double_data = $util.emptyArray;
            BlobProto.prototype.double_diff = $util.emptyArray;
            BlobProto.prototype.num = 0;
            BlobProto.prototype.channels = 0;
            BlobProto.prototype.height = 0;
            BlobProto.prototype.width = 0;
    
            BlobProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BlobProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 7:
                        message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                        break;
                    case 5:
                        if (!(message.data && message.data.length))
                            message.data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.data.length == 0 && (end2 - reader.pos) > 1048576) {
                                var dataLength = end2 - reader.pos;
                                var dataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, dataLength);
                                dataLength = dataLength >>> 2;
                                var data = new Float32Array(dataLength);
                                for (var i = 0; i < dataLength; i++) {
                                    data[i] = dataView.getFloat32(i << 2, true);
                                }
                                message.data = data;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.data.push(reader.float());
                            }
                        } else
                            message.data.push(reader.float());
                        break;
                    case 6:
                        if (!(message.diff && message.diff.length))
                            message.diff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.diff.push(reader.float());
                        } else
                            message.diff.push(reader.float());
                        break;
                    case 8:
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.double_data.push(reader.double());
                        } else
                            message.double_data.push(reader.double());
                        break;
                    case 9:
                        if (!(message.double_diff && message.double_diff.length))
                            message.double_diff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.double_diff.push(reader.double());
                        } else
                            message.double_diff.push(reader.double());
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
    
            BlobProto.decodeText = function decodeText(reader) {
                var message = new $root.caffe.BlobProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shape":
                        message.shape = $root.caffe.BlobShape.decodeText(reader, true);
                        break;
                    case "data":
                        if (!(message.data && message.data.length))
                            message.data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.data.push(reader.float());
                                reader.next();
                            }
                        else
                            message.data.push(reader.float());
                        break;
                    case "diff":
                        if (!(message.diff && message.diff.length))
                            message.diff = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.diff.push(reader.float());
                                reader.next();
                            }
                        else
                            message.diff.push(reader.float());
                        break;
                    case "double_data":
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.double_data.push(reader.double());
                                reader.next();
                            }
                        else
                            message.double_data.push(reader.double());
                        break;
                    case "double_diff":
                        if (!(message.double_diff && message.double_diff.length))
                            message.double_diff = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.double_diff.push(reader.double());
                                reader.next();
                            }
                        else
                            message.double_diff.push(reader.double());
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
    
            function BlobProtoVector(properties) {
                this.blobs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobProtoVector.prototype.blobs = $util.emptyArray;
    
            BlobProtoVector.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BlobProtoVector();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BlobProtoVector.decodeText = function decodeText(reader) {
                var message = new $root.caffe.BlobProtoVector();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
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
    
            function Datum(properties) {
                this.float_data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Datum.prototype.channels = 0;
            Datum.prototype.height = 0;
            Datum.prototype.width = 0;
            Datum.prototype.data = $util.newBuffer([]);
            Datum.prototype.label = 0;
            Datum.prototype.float_data = $util.emptyArray;
            Datum.prototype.encoded = false;
    
            Datum.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.Datum();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.float_data.push(reader.float());
                        } else
                            message.float_data.push(reader.float());
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
    
            Datum.decodeText = function decodeText(reader) {
                var message = new $root.caffe.Datum();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.float_data.push(reader.float());
                                reader.next();
                            }
                        else
                            message.float_data.push(reader.float());
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
    
            function FillerParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FillerParameter.prototype.type = "constant";
            FillerParameter.prototype.value = 0;
            FillerParameter.prototype.min = 0;
            FillerParameter.prototype.max = 1;
            FillerParameter.prototype.mean = 0;
            FillerParameter.prototype.std = 1;
            FillerParameter.prototype.sparse = -1;
            FillerParameter.prototype.variance_norm = 0;
    
            FillerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.FillerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            FillerParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.FillerParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "FAN_IN"] = 0;
                values[valuesById[1] = "FAN_OUT"] = 1;
                values[valuesById[2] = "AVERAGE"] = 2;
                return values;
            })();
    
            return FillerParameter;
        })();
    
        caffe.NetParameter = (function() {
    
            function NetParameter(properties) {
                this.input = [];
                this.input_shape = [];
                this.input_dim = [];
                this.layer = [];
                this.layers = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetParameter.prototype.name = "";
            NetParameter.prototype.input = $util.emptyArray;
            NetParameter.prototype.input_shape = $util.emptyArray;
            NetParameter.prototype.input_dim = $util.emptyArray;
            NetParameter.prototype.force_backward = false;
            NetParameter.prototype.state = null;
            NetParameter.prototype.debug_info = false;
            NetParameter.prototype.layer = $util.emptyArray;
            NetParameter.prototype.layers = $util.emptyArray;
    
            NetParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.NetParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 3:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 8:
                        if (!(message.input_shape && message.input_shape.length))
                            message.input_shape = [];
                        message.input_shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        if (!(message.input_dim && message.input_dim.length))
                            message.input_dim = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.input_dim.push(reader.int32());
                        } else
                            message.input_dim.push(reader.int32());
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
                        if (!(message.layer && message.layer.length))
                            message.layer = [];
                        message.layer.push($root.caffe.LayerParameter.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        if (!(message.layers && message.layers.length))
                            message.layers = [];
                        message.layers.push($root.caffe.V1LayerParameter.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.NetParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.input.push(reader.string());
                                reader.next();
                            }
                        else
                            message.input.push(reader.string());
                        break;
                    case "input_shape":
                        if (!(message.input_shape && message.input_shape.length))
                            message.input_shape = [];
                        message.input_shape.push($root.caffe.BlobShape.decodeText(reader, true));
                        break;
                    case "input_dim":
                        if (!(message.input_dim && message.input_dim.length))
                            message.input_dim = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.input_dim.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.input_dim.push(reader.int32());
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
                        if (!(message.layer && message.layer.length))
                            message.layer = [];
                        message.layer.push($root.caffe.LayerParameter.decodeText(reader, true));
                        break;
                    case "layers":
                        if (!(message.layers && message.layers.length))
                            message.layers = [];
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
    
            function SolverParameter(properties) {
                this.test_net = [];
                this.test_net_param = [];
                this.test_state = [];
                this.test_iter = [];
                this.stepvalue = [];
                this.weights = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SolverParameter.prototype.net = "";
            SolverParameter.prototype.net_param = null;
            SolverParameter.prototype.train_net = "";
            SolverParameter.prototype.test_net = $util.emptyArray;
            SolverParameter.prototype.train_net_param = null;
            SolverParameter.prototype.test_net_param = $util.emptyArray;
            SolverParameter.prototype.train_state = null;
            SolverParameter.prototype.test_state = $util.emptyArray;
            SolverParameter.prototype.test_iter = $util.emptyArray;
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
            SolverParameter.prototype.stepvalue = $util.emptyArray;
            SolverParameter.prototype.clip_gradients = -1;
            SolverParameter.prototype.snapshot = 0;
            SolverParameter.prototype.snapshot_prefix = "";
            SolverParameter.prototype.snapshot_diff = false;
            SolverParameter.prototype.snapshot_format = 1;
            SolverParameter.prototype.solver_mode = 1;
            SolverParameter.prototype.device_id = 0;
            SolverParameter.prototype.random_seed = $util.Long ? $util.Long.fromBits(-1,-1,false) : -1;
            SolverParameter.prototype.type = "SGD";
            SolverParameter.prototype.delta = 1e-8;
            SolverParameter.prototype.momentum2 = 0.999;
            SolverParameter.prototype.rms_decay = 0.99;
            SolverParameter.prototype.debug_info = false;
            SolverParameter.prototype.snapshot_after_train = true;
            SolverParameter.prototype.solver_type = 0;
            SolverParameter.prototype.layer_wise_reduce = true;
            SolverParameter.prototype.weights = $util.emptyArray;
    
            SolverParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SolverParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.test_net && message.test_net.length))
                            message.test_net = [];
                        message.test_net.push(reader.string());
                        break;
                    case 21:
                        message.train_net_param = $root.caffe.NetParameter.decode(reader, reader.uint32());
                        break;
                    case 22:
                        if (!(message.test_net_param && message.test_net_param.length))
                            message.test_net_param = [];
                        message.test_net_param.push($root.caffe.NetParameter.decode(reader, reader.uint32()));
                        break;
                    case 26:
                        message.train_state = $root.caffe.NetState.decode(reader, reader.uint32());
                        break;
                    case 27:
                        if (!(message.test_state && message.test_state.length))
                            message.test_state = [];
                        message.test_state.push($root.caffe.NetState.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.test_iter && message.test_iter.length))
                            message.test_iter = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.test_iter.push(reader.int32());
                        } else
                            message.test_iter.push(reader.int32());
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
                        if (!(message.stepvalue && message.stepvalue.length))
                            message.stepvalue = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.stepvalue.push(reader.int32());
                        } else
                            message.stepvalue.push(reader.int32());
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
                        if (!(message.weights && message.weights.length))
                            message.weights = [];
                        message.weights.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SolverParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SolverParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.test_net && message.test_net.length))
                            message.test_net = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.test_net.push(reader.string());
                                reader.next();
                            }
                        else
                            message.test_net.push(reader.string());
                        break;
                    case "train_net_param":
                        message.train_net_param = $root.caffe.NetParameter.decodeText(reader, true);
                        break;
                    case "test_net_param":
                        if (!(message.test_net_param && message.test_net_param.length))
                            message.test_net_param = [];
                        message.test_net_param.push($root.caffe.NetParameter.decodeText(reader, true));
                        break;
                    case "train_state":
                        message.train_state = $root.caffe.NetState.decodeText(reader, true);
                        break;
                    case "test_state":
                        if (!(message.test_state && message.test_state.length))
                            message.test_state = [];
                        message.test_state.push($root.caffe.NetState.decodeText(reader, true));
                        break;
                    case "test_iter":
                        if (!(message.test_iter && message.test_iter.length))
                            message.test_iter = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.test_iter.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.test_iter.push(reader.int32());
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
                        if (!(message.stepvalue && message.stepvalue.length))
                            message.stepvalue = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.stepvalue.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.stepvalue.push(reader.int32());
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
                        if (!(message.weights && message.weights.length))
                            message.weights = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.weights.push(reader.string());
                                reader.next();
                            }
                        else
                            message.weights.push(reader.string());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SolverParameter.SnapshotFormat = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "HDF5"] = 0;
                values[valuesById[1] = "BINARYPROTO"] = 1;
                return values;
            })();
    
            SolverParameter.SolverMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "CPU"] = 0;
                values[valuesById[1] = "GPU"] = 1;
                return values;
            })();
    
            SolverParameter.SolverType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "SGD"] = 0;
                values[valuesById[1] = "NESTEROV"] = 1;
                values[valuesById[2] = "ADAGRAD"] = 2;
                values[valuesById[3] = "RMSPROP"] = 3;
                values[valuesById[4] = "ADADELTA"] = 4;
                values[valuesById[5] = "ADAM"] = 5;
                return values;
            })();
    
            return SolverParameter;
        })();
    
        caffe.SolverState = (function() {
    
            function SolverState(properties) {
                this.history = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SolverState.prototype.iter = 0;
            SolverState.prototype.learned_net = "";
            SolverState.prototype.history = $util.emptyArray;
            SolverState.prototype.current_step = 0;
    
            SolverState.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SolverState();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.iter = reader.int32();
                        break;
                    case 2:
                        message.learned_net = reader.string();
                        break;
                    case 3:
                        if (!(message.history && message.history.length))
                            message.history = [];
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
    
            SolverState.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SolverState();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "iter":
                        message.iter = reader.int32();
                        break;
                    case "learned_net":
                        message.learned_net = reader.string();
                        break;
                    case "history":
                        if (!(message.history && message.history.length))
                            message.history = [];
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
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "TRAIN"] = 0;
            values[valuesById[1] = "TEST"] = 1;
            return values;
        })();
    
        caffe.NetState = (function() {
    
            function NetState(properties) {
                this.stage = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetState.prototype.phase = 1;
            NetState.prototype.level = 0;
            NetState.prototype.stage = $util.emptyArray;
    
            NetState.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.NetState();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.phase = reader.int32();
                        break;
                    case 2:
                        message.level = reader.int32();
                        break;
                    case 3:
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetState.decodeText = function decodeText(reader) {
                var message = new $root.caffe.NetState();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "phase":
                        message.phase = reader.enum($root.caffe.Phase);
                        break;
                    case "level":
                        message.level = reader.int32();
                        break;
                    case "stage":
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.stage.push(reader.string());
                                reader.next();
                            }
                        else
                            message.stage.push(reader.string());
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
    
            function NetStateRule(properties) {
                this.stage = [];
                this.not_stage = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetStateRule.prototype.phase = 0;
            NetStateRule.prototype.min_level = 0;
            NetStateRule.prototype.max_level = 0;
            NetStateRule.prototype.stage = $util.emptyArray;
            NetStateRule.prototype.not_stage = $util.emptyArray;
    
            NetStateRule.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.NetStateRule();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    case 5:
                        if (!(message.not_stage && message.not_stage.length))
                            message.not_stage = [];
                        message.not_stage.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetStateRule.decodeText = function decodeText(reader) {
                var message = new $root.caffe.NetStateRule();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.stage.push(reader.string());
                                reader.next();
                            }
                        else
                            message.stage.push(reader.string());
                        break;
                    case "not_stage":
                        if (!(message.not_stage && message.not_stage.length))
                            message.not_stage = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.not_stage.push(reader.string());
                                reader.next();
                            }
                        else
                            message.not_stage.push(reader.string());
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
    
            function ParamSpec(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ParamSpec.prototype.name = "";
            ParamSpec.prototype.share_mode = 0;
            ParamSpec.prototype.lr_mult = 1;
            ParamSpec.prototype.decay_mult = 1;
    
            ParamSpec.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ParamSpec();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ParamSpec.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ParamSpec();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "STRICT"] = 0;
                values[valuesById[1] = "PERMISSIVE"] = 1;
                return values;
            })();
    
            return ParamSpec;
        })();
    
        caffe.LayerParameter = (function() {
    
            function LayerParameter(properties) {
                this.bottom = [];
                this.top = [];
                this.loss_weight = [];
                this.param = [];
                this.blobs = [];
                this.propagate_down = [];
                this.include = [];
                this.exclude = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LayerParameter.prototype.name = "";
            LayerParameter.prototype.type = "";
            LayerParameter.prototype.bottom = $util.emptyArray;
            LayerParameter.prototype.top = $util.emptyArray;
            LayerParameter.prototype.phase = 0;
            LayerParameter.prototype.loss_weight = $util.emptyArray;
            LayerParameter.prototype.param = $util.emptyArray;
            LayerParameter.prototype.blobs = $util.emptyArray;
            LayerParameter.prototype.propagate_down = $util.emptyArray;
            LayerParameter.prototype.include = $util.emptyArray;
            LayerParameter.prototype.exclude = $util.emptyArray;
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
    
            LayerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LayerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.type = reader.string();
                        break;
                    case 3:
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        message.bottom.push(reader.string());
                        break;
                    case 4:
                        if (!(message.top && message.top.length))
                            message.top = [];
                        message.top.push(reader.string());
                        break;
                    case 10:
                        message.phase = reader.int32();
                        break;
                    case 5:
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.loss_weight.push(reader.float());
                        } else
                            message.loss_weight.push(reader.float());
                        break;
                    case 6:
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push($root.caffe.ParamSpec.decode(reader, reader.uint32()));
                        break;
                    case 7:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 11:
                        if (!(message.propagate_down && message.propagate_down.length))
                            message.propagate_down = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.propagate_down.push(reader.bool());
                        } else
                            message.propagate_down.push(reader.bool());
                        break;
                    case 8:
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 9:
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
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
    
            LayerParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.LayerParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "type":
                        message.type = reader.string();
                        break;
                    case "bottom":
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.bottom.push(reader.string());
                                reader.next();
                            }
                        else
                            message.bottom.push(reader.string());
                        break;
                    case "top":
                        if (!(message.top && message.top.length))
                            message.top = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.top.push(reader.string());
                                reader.next();
                            }
                        else
                            message.top.push(reader.string());
                        break;
                    case "phase":
                        message.phase = reader.enum($root.caffe.Phase);
                        break;
                    case "loss_weight":
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.loss_weight.push(reader.float());
                                reader.next();
                            }
                        else
                            message.loss_weight.push(reader.float());
                        break;
                    case "param":
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push($root.caffe.ParamSpec.decodeText(reader, true));
                        break;
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "propagate_down":
                        if (!(message.propagate_down && message.propagate_down.length))
                            message.propagate_down = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.propagate_down.push(reader.bool());
                                reader.next();
                            }
                        else
                            message.propagate_down.push(reader.bool());
                        break;
                    case "include":
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "exclude":
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
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
    
            function TransformationParameter(properties) {
                this.mean_value = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TransformationParameter.prototype.scale = 1;
            TransformationParameter.prototype.mirror = false;
            TransformationParameter.prototype.crop_size = 0;
            TransformationParameter.prototype.mean_file = "";
            TransformationParameter.prototype.mean_value = $util.emptyArray;
            TransformationParameter.prototype.force_color = false;
            TransformationParameter.prototype.force_gray = false;
    
            TransformationParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.TransformationParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.mean_value && message.mean_value.length))
                            message.mean_value = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.mean_value.push(reader.float());
                        } else
                            message.mean_value.push(reader.float());
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
    
            TransformationParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.TransformationParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.mean_value && message.mean_value.length))
                            message.mean_value = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.mean_value.push(reader.float());
                                reader.next();
                            }
                        else
                            message.mean_value.push(reader.float());
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
    
            function LossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LossParameter.prototype.ignore_label = 0;
            LossParameter.prototype.normalization = 1;
            LossParameter.prototype.normalize = false;
    
            LossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            LossParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.LossParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "FULL"] = 0;
                values[valuesById[1] = "VALID"] = 1;
                values[valuesById[2] = "BATCH_SIZE"] = 2;
                values[valuesById[3] = "NONE"] = 3;
                return values;
            })();
    
            return LossParameter;
        })();
    
        caffe.AccuracyParameter = (function() {
    
            function AccuracyParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            AccuracyParameter.prototype.top_k = 1;
            AccuracyParameter.prototype.axis = 1;
            AccuracyParameter.prototype.ignore_label = 0;
    
            AccuracyParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.AccuracyParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            AccuracyParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.AccuracyParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ArgMaxParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ArgMaxParameter.prototype.out_max_val = false;
            ArgMaxParameter.prototype.top_k = 1;
            ArgMaxParameter.prototype.axis = 0;
    
            ArgMaxParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ArgMaxParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ArgMaxParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ArgMaxParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ClipParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ClipParameter.prototype.min = 0;
            ClipParameter.prototype.max = 0;
    
            ClipParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ClipParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ClipParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ClipParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ConcatParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ConcatParameter.prototype.axis = 1;
            ConcatParameter.prototype.concat_dim = 1;
    
            ConcatParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ConcatParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ConcatParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ConcatParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function BatchNormParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BatchNormParameter.prototype.use_global_stats = false;
            BatchNormParameter.prototype.moving_average_fraction = 0.999;
            BatchNormParameter.prototype.eps = 0.00001;
    
            BatchNormParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BatchNormParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            BatchNormParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.BatchNormParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function BiasParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BiasParameter.prototype.axis = 1;
            BiasParameter.prototype.num_axes = 1;
            BiasParameter.prototype.filler = null;
    
            BiasParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BiasParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            BiasParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.BiasParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ContrastiveLossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ContrastiveLossParameter.prototype.margin = 1;
            ContrastiveLossParameter.prototype.legacy_version = false;
    
            ContrastiveLossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ContrastiveLossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ContrastiveLossParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ContrastiveLossParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ConvolutionParameter(properties) {
                this.pad = [];
                this.kernel_size = [];
                this.stride = [];
                this.dilation = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ConvolutionParameter.prototype.num_output = 0;
            ConvolutionParameter.prototype.bias_term = true;
            ConvolutionParameter.prototype.pad = $util.emptyArray;
            ConvolutionParameter.prototype.kernel_size = $util.emptyArray;
            ConvolutionParameter.prototype.stride = $util.emptyArray;
            ConvolutionParameter.prototype.dilation = $util.emptyArray;
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
    
            ConvolutionParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ConvolutionParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.num_output = reader.uint32();
                        break;
                    case 2:
                        message.bias_term = reader.bool();
                        break;
                    case 3:
                        if (!(message.pad && message.pad.length))
                            message.pad = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.pad.push(reader.uint32());
                        } else
                            message.pad.push(reader.uint32());
                        break;
                    case 4:
                        if (!(message.kernel_size && message.kernel_size.length))
                            message.kernel_size = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.kernel_size.push(reader.uint32());
                        } else
                            message.kernel_size.push(reader.uint32());
                        break;
                    case 6:
                        if (!(message.stride && message.stride.length))
                            message.stride = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.stride.push(reader.uint32());
                        } else
                            message.stride.push(reader.uint32());
                        break;
                    case 18:
                        if (!(message.dilation && message.dilation.length))
                            message.dilation = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dilation.push(reader.uint32());
                        } else
                            message.dilation.push(reader.uint32());
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
    
            ConvolutionParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ConvolutionParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "num_output":
                        message.num_output = reader.uint32();
                        break;
                    case "bias_term":
                        message.bias_term = reader.bool();
                        break;
                    case "pad":
                        if (!(message.pad && message.pad.length))
                            message.pad = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.pad.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.pad.push(reader.uint32());
                        break;
                    case "kernel_size":
                        if (!(message.kernel_size && message.kernel_size.length))
                            message.kernel_size = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.kernel_size.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.kernel_size.push(reader.uint32());
                        break;
                    case "stride":
                        if (!(message.stride && message.stride.length))
                            message.stride = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.stride.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.stride.push(reader.uint32());
                        break;
                    case "dilation":
                        if (!(message.dilation && message.dilation.length))
                            message.dilation = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dilation.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.dilation.push(reader.uint32());
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return ConvolutionParameter;
        })();
    
        caffe.CropParameter = (function() {
    
            function CropParameter(properties) {
                this.offset = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            CropParameter.prototype.axis = 2;
            CropParameter.prototype.offset = $util.emptyArray;
    
            CropParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.CropParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        if (!(message.offset && message.offset.length))
                            message.offset = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.offset.push(reader.uint32());
                        } else
                            message.offset.push(reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            CropParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.CropParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "offset":
                        if (!(message.offset && message.offset.length))
                            message.offset = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.offset.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.offset.push(reader.uint32());
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
    
            function DataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
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
    
            DataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.DataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            DataParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.DataParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "LEVELDB"] = 0;
                values[valuesById[1] = "LMDB"] = 1;
                return values;
            })();
    
            return DataParameter;
        })();
    
        caffe.DropoutParameter = (function() {
    
            function DropoutParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DropoutParameter.prototype.dropout_ratio = 0.5;
    
            DropoutParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.DropoutParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            DropoutParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.DropoutParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function DummyDataParameter(properties) {
                this.data_filler = [];
                this.shape = [];
                this.num = [];
                this.channels = [];
                this.height = [];
                this.width = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DummyDataParameter.prototype.data_filler = $util.emptyArray;
            DummyDataParameter.prototype.shape = $util.emptyArray;
            DummyDataParameter.prototype.num = $util.emptyArray;
            DummyDataParameter.prototype.channels = $util.emptyArray;
            DummyDataParameter.prototype.height = $util.emptyArray;
            DummyDataParameter.prototype.width = $util.emptyArray;
    
            DummyDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.DummyDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.data_filler && message.data_filler.length))
                            message.data_filler = [];
                        message.data_filler.push($root.caffe.FillerParameter.decode(reader, reader.uint32()));
                        break;
                    case 6:
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        if (!(message.num && message.num.length))
                            message.num = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.num.push(reader.uint32());
                        } else
                            message.num.push(reader.uint32());
                        break;
                    case 3:
                        if (!(message.channels && message.channels.length))
                            message.channels = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.channels.push(reader.uint32());
                        } else
                            message.channels.push(reader.uint32());
                        break;
                    case 4:
                        if (!(message.height && message.height.length))
                            message.height = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.height.push(reader.uint32());
                        } else
                            message.height.push(reader.uint32());
                        break;
                    case 5:
                        if (!(message.width && message.width.length))
                            message.width = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.width.push(reader.uint32());
                        } else
                            message.width.push(reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DummyDataParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.DummyDataParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "data_filler":
                        if (!(message.data_filler && message.data_filler.length))
                            message.data_filler = [];
                        message.data_filler.push($root.caffe.FillerParameter.decodeText(reader, true));
                        break;
                    case "shape":
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decodeText(reader, true));
                        break;
                    case "num":
                        if (!(message.num && message.num.length))
                            message.num = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.num.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.num.push(reader.uint32());
                        break;
                    case "channels":
                        if (!(message.channels && message.channels.length))
                            message.channels = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.channels.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.channels.push(reader.uint32());
                        break;
                    case "height":
                        if (!(message.height && message.height.length))
                            message.height = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.height.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.height.push(reader.uint32());
                        break;
                    case "width":
                        if (!(message.width && message.width.length))
                            message.width = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.width.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.width.push(reader.uint32());
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
    
            function EltwiseParameter(properties) {
                this.coeff = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            EltwiseParameter.prototype.operation = 1;
            EltwiseParameter.prototype.coeff = $util.emptyArray;
            EltwiseParameter.prototype.stable_prod_grad = true;
    
            EltwiseParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.EltwiseParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.operation = reader.int32();
                        break;
                    case 2:
                        if (!(message.coeff && message.coeff.length))
                            message.coeff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.coeff.push(reader.float());
                        } else
                            message.coeff.push(reader.float());
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
    
            EltwiseParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.EltwiseParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "operation":
                        message.operation = reader.enum($root.caffe.EltwiseParameter.EltwiseOp);
                        break;
                    case "coeff":
                        if (!(message.coeff && message.coeff.length))
                            message.coeff = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.coeff.push(reader.float());
                                reader.next();
                            }
                        else
                            message.coeff.push(reader.float());
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "PROD"] = 0;
                values[valuesById[1] = "SUM"] = 1;
                values[valuesById[2] = "MAX"] = 2;
                return values;
            })();
    
            return EltwiseParameter;
        })();
    
        caffe.ELUParameter = (function() {
    
            function ELUParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ELUParameter.prototype.alpha = 1;
    
            ELUParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ELUParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ELUParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ELUParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function EmbedParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            EmbedParameter.prototype.num_output = 0;
            EmbedParameter.prototype.input_dim = 0;
            EmbedParameter.prototype.bias_term = true;
            EmbedParameter.prototype.weight_filler = null;
            EmbedParameter.prototype.bias_filler = null;
    
            EmbedParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.EmbedParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            EmbedParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.EmbedParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ExpParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ExpParameter.prototype.base = -1;
            ExpParameter.prototype.scale = 1;
            ExpParameter.prototype.shift = 0;
    
            ExpParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ExpParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ExpParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ExpParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function FlattenParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FlattenParameter.prototype.axis = 1;
            FlattenParameter.prototype.end_axis = -1;
    
            FlattenParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.FlattenParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            FlattenParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.FlattenParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function HDF5DataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            HDF5DataParameter.prototype.source = "";
            HDF5DataParameter.prototype.batch_size = 0;
            HDF5DataParameter.prototype.shuffle = false;
    
            HDF5DataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.HDF5DataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            HDF5DataParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.HDF5DataParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function HDF5OutputParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            HDF5OutputParameter.prototype.file_name = "";
    
            HDF5OutputParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.HDF5OutputParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            HDF5OutputParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.HDF5OutputParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function HingeLossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            HingeLossParameter.prototype.norm = 1;
    
            HingeLossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.HingeLossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            HingeLossParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.HingeLossParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[1] = "L1"] = 1;
                values[valuesById[2] = "L2"] = 2;
                return values;
            })();
    
            return HingeLossParameter;
        })();
    
        caffe.ImageDataParameter = (function() {
    
            function ImageDataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
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
    
            ImageDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ImageDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ImageDataParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ImageDataParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function InfogainLossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            InfogainLossParameter.prototype.source = "";
            InfogainLossParameter.prototype.axis = 1;
    
            InfogainLossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.InfogainLossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            InfogainLossParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.InfogainLossParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function InnerProductParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            InnerProductParameter.prototype.num_output = 0;
            InnerProductParameter.prototype.bias_term = true;
            InnerProductParameter.prototype.weight_filler = null;
            InnerProductParameter.prototype.bias_filler = null;
            InnerProductParameter.prototype.axis = 1;
            InnerProductParameter.prototype.transpose = false;
    
            InnerProductParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.InnerProductParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            InnerProductParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.InnerProductParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function InputParameter(properties) {
                this.shape = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            InputParameter.prototype.shape = $util.emptyArray;
    
            InputParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.InputParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            InputParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.InputParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shape":
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
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
    
            function LogParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LogParameter.prototype.base = -1;
            LogParameter.prototype.scale = 1;
            LogParameter.prototype.shift = 0;
    
            LogParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LogParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            LogParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.LogParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function LRNParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LRNParameter.prototype.local_size = 5;
            LRNParameter.prototype.alpha = 1;
            LRNParameter.prototype.beta = 0.75;
            LRNParameter.prototype.norm_region = 0;
            LRNParameter.prototype.k = 1;
            LRNParameter.prototype.engine = 0;
    
            LRNParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LRNParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            LRNParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.LRNParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "ACROSS_CHANNELS"] = 0;
                values[valuesById[1] = "WITHIN_CHANNEL"] = 1;
                return values;
            })();
    
            LRNParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return LRNParameter;
        })();
    
        caffe.MemoryDataParameter = (function() {
    
            function MemoryDataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MemoryDataParameter.prototype.batch_size = 0;
            MemoryDataParameter.prototype.channels = 0;
            MemoryDataParameter.prototype.height = 0;
            MemoryDataParameter.prototype.width = 0;
    
            MemoryDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.MemoryDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            MemoryDataParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.MemoryDataParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function MVNParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MVNParameter.prototype.normalize_variance = true;
            MVNParameter.prototype.across_channels = false;
            MVNParameter.prototype.eps = 1e-9;
    
            MVNParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.MVNParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            MVNParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.MVNParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ParameterParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ParameterParameter.prototype.shape = null;
    
            ParameterParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ParameterParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ParameterParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ParameterParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function PoolingParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
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
    
            PoolingParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PoolingParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            PoolingParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.PoolingParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "MAX"] = 0;
                values[valuesById[1] = "AVE"] = 1;
                values[valuesById[2] = "STOCHASTIC"] = 2;
                return values;
            })();
    
            PoolingParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            PoolingParameter.RoundMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "CEIL"] = 0;
                values[valuesById[1] = "FLOOR"] = 1;
                return values;
            })();
    
            return PoolingParameter;
        })();
    
        caffe.PowerParameter = (function() {
    
            function PowerParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PowerParameter.prototype.power = 1;
            PowerParameter.prototype.scale = 1;
            PowerParameter.prototype.shift = 0;
    
            PowerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PowerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            PowerParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.PowerParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function PythonParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PythonParameter.prototype.module = "";
            PythonParameter.prototype.layer = "";
            PythonParameter.prototype.param_str = "";
            PythonParameter.prototype.share_in_parallel = false;
    
            PythonParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PythonParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            PythonParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.PythonParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function RecurrentParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            RecurrentParameter.prototype.num_output = 0;
            RecurrentParameter.prototype.weight_filler = null;
            RecurrentParameter.prototype.bias_filler = null;
            RecurrentParameter.prototype.debug_info = false;
            RecurrentParameter.prototype.expose_hidden = false;
    
            RecurrentParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.RecurrentParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            RecurrentParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.RecurrentParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ReductionParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ReductionParameter.prototype.operation = 1;
            ReductionParameter.prototype.axis = 0;
            ReductionParameter.prototype.coeff = 1;
    
            ReductionParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ReductionParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ReductionParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ReductionParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[1] = "SUM"] = 1;
                values[valuesById[2] = "ASUM"] = 2;
                values[valuesById[3] = "SUMSQ"] = 3;
                values[valuesById[4] = "MEAN"] = 4;
                return values;
            })();
    
            return ReductionParameter;
        })();
    
        caffe.ReLUParameter = (function() {
    
            function ReLUParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ReLUParameter.prototype.negative_slope = 0;
            ReLUParameter.prototype.engine = 0;
    
            ReLUParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ReLUParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ReLUParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ReLUParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return ReLUParameter;
        })();
    
        caffe.ReshapeParameter = (function() {
    
            function ReshapeParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ReshapeParameter.prototype.shape = null;
            ReshapeParameter.prototype.axis = 0;
            ReshapeParameter.prototype.num_axes = -1;
    
            ReshapeParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ReshapeParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ReshapeParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ReshapeParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ScaleParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ScaleParameter.prototype.axis = 1;
            ScaleParameter.prototype.num_axes = 1;
            ScaleParameter.prototype.filler = null;
            ScaleParameter.prototype.bias_term = false;
            ScaleParameter.prototype.bias_filler = null;
    
            ScaleParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ScaleParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ScaleParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ScaleParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SigmoidParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SigmoidParameter.prototype.engine = 0;
    
            SigmoidParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SigmoidParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SigmoidParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SigmoidParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return SigmoidParameter;
        })();
    
        caffe.SliceParameter = (function() {
    
            function SliceParameter(properties) {
                this.slice_point = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SliceParameter.prototype.axis = 1;
            SliceParameter.prototype.slice_point = $util.emptyArray;
            SliceParameter.prototype.slice_dim = 1;
    
            SliceParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SliceParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 3:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        if (!(message.slice_point && message.slice_point.length))
                            message.slice_point = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.slice_point.push(reader.uint32());
                        } else
                            message.slice_point.push(reader.uint32());
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
    
            SliceParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SliceParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "slice_point":
                        if (!(message.slice_point && message.slice_point.length))
                            message.slice_point = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.slice_point.push(reader.uint32());
                                reader.next();
                            }
                        else
                            message.slice_point.push(reader.uint32());
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
    
            function SoftmaxParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SoftmaxParameter.prototype.engine = 0;
            SoftmaxParameter.prototype.axis = 1;
    
            SoftmaxParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SoftmaxParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SoftmaxParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SoftmaxParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return SoftmaxParameter;
        })();
    
        caffe.SwishParameter = (function() {
    
            function SwishParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SwishParameter.prototype.beta = 1;
    
            SwishParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SwishParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SwishParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SwishParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function TanHParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TanHParameter.prototype.engine = 0;
    
            TanHParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.TanHParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            TanHParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.TanHParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return TanHParameter;
        })();
    
        caffe.TileParameter = (function() {
    
            function TileParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TileParameter.prototype.axis = 1;
            TileParameter.prototype.tiles = 0;
    
            TileParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.TileParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            TileParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.TileParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function ThresholdParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ThresholdParameter.prototype.threshold = 0;
    
            ThresholdParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ThresholdParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            ThresholdParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.ThresholdParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function WindowDataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
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
    
            WindowDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.WindowDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            WindowDataParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.WindowDataParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
    
            function SPPParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SPPParameter.prototype.pyramid_height = 0;
            SPPParameter.prototype.pool = 0;
            SPPParameter.prototype.engine = 0;
    
            SPPParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SPPParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            SPPParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.SPPParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "MAX"] = 0;
                values[valuesById[1] = "AVE"] = 1;
                values[valuesById[2] = "STOCHASTIC"] = 2;
                return values;
            })();
    
            SPPParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return SPPParameter;
        })();
    
        caffe.V1LayerParameter = (function() {
    
            function V1LayerParameter(properties) {
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
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            V1LayerParameter.prototype.bottom = $util.emptyArray;
            V1LayerParameter.prototype.top = $util.emptyArray;
            V1LayerParameter.prototype.name = "";
            V1LayerParameter.prototype.include = $util.emptyArray;
            V1LayerParameter.prototype.exclude = $util.emptyArray;
            V1LayerParameter.prototype.type = 0;
            V1LayerParameter.prototype.blobs = $util.emptyArray;
            V1LayerParameter.prototype.param = $util.emptyArray;
            V1LayerParameter.prototype.blob_share_mode = $util.emptyArray;
            V1LayerParameter.prototype.blobs_lr = $util.emptyArray;
            V1LayerParameter.prototype.weight_decay = $util.emptyArray;
            V1LayerParameter.prototype.loss_weight = $util.emptyArray;
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
    
            V1LayerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.V1LayerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 2:
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        message.bottom.push(reader.string());
                        break;
                    case 3:
                        if (!(message.top && message.top.length))
                            message.top = [];
                        message.top.push(reader.string());
                        break;
                    case 4:
                        message.name = reader.string();
                        break;
                    case 32:
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 33:
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
                        message.exclude.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 5:
                        message.type = reader.int32();
                        break;
                    case 6:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 1001:
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push(reader.string());
                        break;
                    case 1002:
                        if (!(message.blob_share_mode && message.blob_share_mode.length))
                            message.blob_share_mode = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blob_share_mode.push(reader.int32());
                        } else
                            message.blob_share_mode.push(reader.int32());
                        break;
                    case 7:
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobs_lr.push(reader.float());
                        } else
                            message.blobs_lr.push(reader.float());
                        break;
                    case 8:
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.weight_decay.push(reader.float());
                        } else
                            message.weight_decay.push(reader.float());
                        break;
                    case 35:
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.loss_weight.push(reader.float());
                        } else
                            message.loss_weight.push(reader.float());
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
    
            V1LayerParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.V1LayerParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "bottom":
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.bottom.push(reader.string());
                                reader.next();
                            }
                        else
                            message.bottom.push(reader.string());
                        break;
                    case "top":
                        if (!(message.top && message.top.length))
                            message.top = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.top.push(reader.string());
                                reader.next();
                            }
                        else
                            message.top.push(reader.string());
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "include":
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "exclude":
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
                        message.exclude.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "type":
                        message.type = reader.enum($root.caffe.V1LayerParameter.LayerType);
                        break;
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "param":
                        if (!(message.param && message.param.length))
                            message.param = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.param.push(reader.string());
                                reader.next();
                            }
                        else
                            message.param.push(reader.string());
                        break;
                    case "blob_share_mode":
                        if (!(message.blob_share_mode && message.blob_share_mode.length))
                            message.blob_share_mode = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.blob_share_mode.push(reader.enum($root.caffe.V1LayerParameter.DimCheckMode));
                                reader.next();
                            }
                        else
                            message.blob_share_mode.push(reader.enum($root.caffe.V1LayerParameter.DimCheckMode));
                        break;
                    case "blobs_lr":
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.blobs_lr.push(reader.float());
                                reader.next();
                            }
                        else
                            message.blobs_lr.push(reader.float());
                        break;
                    case "weight_decay":
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.weight_decay.push(reader.float());
                                reader.next();
                            }
                        else
                            message.weight_decay.push(reader.float());
                        break;
                    case "loss_weight":
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.loss_weight.push(reader.float());
                                reader.next();
                            }
                        else
                            message.loss_weight.push(reader.float());
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "NONE"] = 0;
                values[valuesById[35] = "ABSVAL"] = 35;
                values[valuesById[1] = "ACCURACY"] = 1;
                values[valuesById[30] = "ARGMAX"] = 30;
                values[valuesById[2] = "BNLL"] = 2;
                values[valuesById[3] = "CONCAT"] = 3;
                values[valuesById[37] = "CONTRASTIVE_LOSS"] = 37;
                values[valuesById[4] = "CONVOLUTION"] = 4;
                values[valuesById[5] = "DATA"] = 5;
                values[valuesById[39] = "DECONVOLUTION"] = 39;
                values[valuesById[6] = "DROPOUT"] = 6;
                values[valuesById[32] = "DUMMY_DATA"] = 32;
                values[valuesById[7] = "EUCLIDEAN_LOSS"] = 7;
                values[valuesById[25] = "ELTWISE"] = 25;
                values[valuesById[38] = "EXP"] = 38;
                values[valuesById[8] = "FLATTEN"] = 8;
                values[valuesById[9] = "HDF5_DATA"] = 9;
                values[valuesById[10] = "HDF5_OUTPUT"] = 10;
                values[valuesById[28] = "HINGE_LOSS"] = 28;
                values[valuesById[11] = "IM2COL"] = 11;
                values[valuesById[12] = "IMAGE_DATA"] = 12;
                values[valuesById[13] = "INFOGAIN_LOSS"] = 13;
                values[valuesById[14] = "INNER_PRODUCT"] = 14;
                values[valuesById[15] = "LRN"] = 15;
                values[valuesById[29] = "MEMORY_DATA"] = 29;
                values[valuesById[16] = "MULTINOMIAL_LOGISTIC_LOSS"] = 16;
                values[valuesById[34] = "MVN"] = 34;
                values[valuesById[17] = "POOLING"] = 17;
                values[valuesById[26] = "POWER"] = 26;
                values[valuesById[18] = "RELU"] = 18;
                values[valuesById[19] = "SIGMOID"] = 19;
                values[valuesById[27] = "SIGMOID_CROSS_ENTROPY_LOSS"] = 27;
                values[valuesById[36] = "SILENCE"] = 36;
                values[valuesById[20] = "SOFTMAX"] = 20;
                values[valuesById[21] = "SOFTMAX_LOSS"] = 21;
                values[valuesById[22] = "SPLIT"] = 22;
                values[valuesById[33] = "SLICE"] = 33;
                values[valuesById[23] = "TANH"] = 23;
                values[valuesById[24] = "WINDOW_DATA"] = 24;
                values[valuesById[31] = "THRESHOLD"] = 31;
                return values;
            })();
    
            V1LayerParameter.DimCheckMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "STRICT"] = 0;
                values[valuesById[1] = "PERMISSIVE"] = 1;
                return values;
            })();
    
            return V1LayerParameter;
        })();
    
        caffe.V0LayerParameter = (function() {
    
            function V0LayerParameter(properties) {
                this.blobs = [];
                this.blobs_lr = [];
                this.weight_decay = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
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
            V0LayerParameter.prototype.blobs = $util.emptyArray;
            V0LayerParameter.prototype.blobs_lr = $util.emptyArray;
            V0LayerParameter.prototype.weight_decay = $util.emptyArray;
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
    
            V0LayerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.V0LayerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 51:
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobs_lr.push(reader.float());
                        } else
                            message.blobs_lr.push(reader.float());
                        break;
                    case 52:
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.weight_decay.push(reader.float());
                        } else
                            message.weight_decay.push(reader.float());
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
    
            V0LayerParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.V0LayerParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "blobs_lr":
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.blobs_lr.push(reader.float());
                                reader.next();
                            }
                        else
                            message.blobs_lr.push(reader.float());
                        break;
                    case "weight_decay":
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.weight_decay.push(reader.float());
                                reader.next();
                            }
                        else
                            message.weight_decay.push(reader.float());
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
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "MAX"] = 0;
                values[valuesById[1] = "AVE"] = 1;
                values[valuesById[2] = "STOCHASTIC"] = 2;
                return values;
            })();
    
            return V0LayerParameter;
        })();
    
        caffe.PReLUParameter = (function() {
    
            function PReLUParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PReLUParameter.prototype.filler = null;
            PReLUParameter.prototype.channel_shared = false;
    
            PReLUParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PReLUParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
    
            PReLUParameter.decodeText = function decodeText(reader) {
                var message = new $root.caffe.PReLUParameter();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
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
