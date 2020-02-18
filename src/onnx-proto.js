/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.onnx || ($protobuf.roots.onnx = {});
    
    $root.onnx = (function() {
    
        var onnx = {};
    
        onnx.Version = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "_START_VERSION"] = 0;
            values[valuesById[1] = "IR_VERSION_2017_10_10"] = 1;
            values[valuesById[2] = "IR_VERSION_2017_10_30"] = 2;
            values[valuesById[3] = "IR_VERSION_2017_11_3"] = 3;
            values[valuesById[4] = "IR_VERSION_2019_1_22"] = 4;
            values[valuesById[5] = "IR_VERSION_2019_3_18"] = 5;
            values[valuesById[6] = "IR_VERSION_2019_9_19"] = 6;
            values[valuesById[7] = "IR_VERSION"] = 7;
            return values;
        })();
    
        onnx.AttributeProto = (function() {
    
            function AttributeProto(properties) {
                this.floats = [];
                this.ints = [];
                this.strings = [];
                this.tensors = [];
                this.graphs = [];
                this.sparse_tensors = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            AttributeProto.prototype.name = "";
            AttributeProto.prototype.ref_attr_name = "";
            AttributeProto.prototype.doc_string = "";
            AttributeProto.prototype.type = 0;
            AttributeProto.prototype.f = 0;
            AttributeProto.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            AttributeProto.prototype.s = $util.newBuffer([]);
            AttributeProto.prototype.t = null;
            AttributeProto.prototype.g = null;
            AttributeProto.prototype.sparse_tensor = null;
            AttributeProto.prototype.floats = $util.emptyArray;
            AttributeProto.prototype.ints = $util.emptyArray;
            AttributeProto.prototype.strings = $util.emptyArray;
            AttributeProto.prototype.tensors = $util.emptyArray;
            AttributeProto.prototype.graphs = $util.emptyArray;
            AttributeProto.prototype.sparse_tensors = $util.emptyArray;
    
            AttributeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.AttributeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 21:
                        message.ref_attr_name = reader.string();
                        break;
                    case 13:
                        message.doc_string = reader.string();
                        break;
                    case 20:
                        message.type = reader.int32();
                        break;
                    case 2:
                        message.f = reader.float();
                        break;
                    case 3:
                        message.i = reader.int64();
                        break;
                    case 4:
                        message.s = reader.bytes();
                        break;
                    case 5:
                        message.t = $root.onnx.TensorProto.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.g = $root.onnx.GraphProto.decode(reader, reader.uint32());
                        break;
                    case 22:
                        message.sparse_tensor = $root.onnx.SparseTensorProto.decode(reader, reader.uint32());
                        break;
                    case 7:
                        if (!(message.floats && message.floats.length))
                            message.floats = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.floats.push(reader.float());
                        } else
                            message.floats.push(reader.float());
                        break;
                    case 8:
                        if (!(message.ints && message.ints.length))
                            message.ints = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.ints.push(reader.int64());
                        } else
                            message.ints.push(reader.int64());
                        break;
                    case 9:
                        if (!(message.strings && message.strings.length))
                            message.strings = [];
                        message.strings.push(reader.bytes());
                        break;
                    case 10:
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
                        message.tensors.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                        break;
                    case 11:
                        if (!(message.graphs && message.graphs.length))
                            message.graphs = [];
                        message.graphs.push($root.onnx.GraphProto.decode(reader, reader.uint32()));
                        break;
                    case 23:
                        if (!(message.sparse_tensors && message.sparse_tensors.length))
                            message.sparse_tensors = [];
                        message.sparse_tensors.push($root.onnx.SparseTensorProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            AttributeProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.AttributeProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "ref_attr_name":
                        message.ref_attr_name = reader.string();
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "type":
                        message.type = reader.enum($root.onnx.AttributeProto.AttributeType);
                        break;
                    case "f":
                        message.f = reader.float();
                        break;
                    case "i":
                        message.i = reader.int64();
                        break;
                    case "s":
                        message.s = reader.bytes();
                        break;
                    case "t":
                        message.t = $root.onnx.TensorProto.decodeText(reader, true);
                        break;
                    case "g":
                        message.g = $root.onnx.GraphProto.decodeText(reader, true);
                        break;
                    case "sparse_tensor":
                        message.sparse_tensor = $root.onnx.SparseTensorProto.decodeText(reader, true);
                        break;
                    case "floats":
                        if (!(message.floats && message.floats.length))
                            message.floats = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.floats.push(reader.float());
                                reader.next();
                            }
                        else
                            message.floats.push(reader.float());
                        break;
                    case "ints":
                        if (!(message.ints && message.ints.length))
                            message.ints = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.ints.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.ints.push(reader.int64());
                        break;
                    case "strings":
                        if (!(message.strings && message.strings.length))
                            message.strings = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.strings.push(reader.bytes());
                                reader.next();
                            }
                        else
                            message.strings.push(reader.bytes());
                        break;
                    case "tensors":
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
                        message.tensors.push($root.onnx.TensorProto.decodeText(reader, true));
                        break;
                    case "graphs":
                        if (!(message.graphs && message.graphs.length))
                            message.graphs = [];
                        message.graphs.push($root.onnx.GraphProto.decodeText(reader, true));
                        break;
                    case "sparse_tensors":
                        if (!(message.sparse_tensors && message.sparse_tensors.length))
                            message.sparse_tensors = [];
                        message.sparse_tensors.push($root.onnx.SparseTensorProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            AttributeProto.AttributeType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNDEFINED"] = 0;
                values[valuesById[1] = "FLOAT"] = 1;
                values[valuesById[2] = "INT"] = 2;
                values[valuesById[3] = "STRING"] = 3;
                values[valuesById[4] = "TENSOR"] = 4;
                values[valuesById[5] = "GRAPH"] = 5;
                values[valuesById[11] = "SPARSE_TENSOR"] = 11;
                values[valuesById[6] = "FLOATS"] = 6;
                values[valuesById[7] = "INTS"] = 7;
                values[valuesById[8] = "STRINGS"] = 8;
                values[valuesById[9] = "TENSORS"] = 9;
                values[valuesById[10] = "GRAPHS"] = 10;
                values[valuesById[12] = "SPARSE_TENSORS"] = 12;
                return values;
            })();
    
            return AttributeProto;
        })();
    
        onnx.ValueInfoProto = (function() {
    
            function ValueInfoProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ValueInfoProto.prototype.name = "";
            ValueInfoProto.prototype.type = null;
            ValueInfoProto.prototype.doc_string = "";
    
            ValueInfoProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.ValueInfoProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ValueInfoProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.ValueInfoProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "type":
                        message.type = $root.onnx.TypeProto.decodeText(reader, true);
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ValueInfoProto;
        })();
    
        onnx.NodeProto = (function() {
    
            function NodeProto(properties) {
                this.input = [];
                this.output = [];
                this.attribute = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NodeProto.prototype.input = $util.emptyArray;
            NodeProto.prototype.output = $util.emptyArray;
            NodeProto.prototype.name = "";
            NodeProto.prototype.op_type = "";
            NodeProto.prototype.domain = "";
            NodeProto.prototype.attribute = $util.emptyArray;
            NodeProto.prototype.doc_string = "";
    
            NodeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.NodeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 2:
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push(reader.string());
                        break;
                    case 3:
                        message.name = reader.string();
                        break;
                    case 4:
                        message.op_type = reader.string();
                        break;
                    case 7:
                        message.domain = reader.string();
                        break;
                    case 5:
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push($root.onnx.AttributeProto.decode(reader, reader.uint32()));
                        break;
                    case 6:
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NodeProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.NodeProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
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
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.output.push(reader.string());
                                reader.next();
                            }
                        else
                            message.output.push(reader.string());
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "op_type":
                        message.op_type = reader.string();
                        break;
                    case "domain":
                        message.domain = reader.string();
                        break;
                    case "attribute":
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push($root.onnx.AttributeProto.decodeText(reader, true));
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return NodeProto;
        })();
    
        onnx.TrainingInfoProto = (function() {
    
            function TrainingInfoProto(properties) {
                this.initialization_binding = [];
                this.update_binding = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TrainingInfoProto.prototype.initialization = null;
            TrainingInfoProto.prototype.algorithm = null;
            TrainingInfoProto.prototype.initialization_binding = $util.emptyArray;
            TrainingInfoProto.prototype.update_binding = $util.emptyArray;
    
            TrainingInfoProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TrainingInfoProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.initialization = $root.onnx.GraphProto.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.algorithm = $root.onnx.GraphProto.decode(reader, reader.uint32());
                        break;
                    case 3:
                        if (!(message.initialization_binding && message.initialization_binding.length))
                            message.initialization_binding = [];
                        message.initialization_binding.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        if (!(message.update_binding && message.update_binding.length))
                            message.update_binding = [];
                        message.update_binding.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TrainingInfoProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.TrainingInfoProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "initialization":
                        message.initialization = $root.onnx.GraphProto.decodeText(reader, true);
                        break;
                    case "algorithm":
                        message.algorithm = $root.onnx.GraphProto.decodeText(reader, true);
                        break;
                    case "initialization_binding":
                        if (!(message.initialization_binding && message.initialization_binding.length))
                            message.initialization_binding = [];
                        message.initialization_binding.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                        break;
                    case "update_binding":
                        if (!(message.update_binding && message.update_binding.length))
                            message.update_binding = [];
                        message.update_binding.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return TrainingInfoProto;
        })();
    
        onnx.ModelProto = (function() {
    
            function ModelProto(properties) {
                this.opset_import = [];
                this.metadata_props = [];
                this.training_info = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ModelProto.prototype.ir_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ModelProto.prototype.opset_import = $util.emptyArray;
            ModelProto.prototype.producer_name = "";
            ModelProto.prototype.producer_version = "";
            ModelProto.prototype.domain = "";
            ModelProto.prototype.model_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ModelProto.prototype.doc_string = "";
            ModelProto.prototype.graph = null;
            ModelProto.prototype.metadata_props = $util.emptyArray;
            ModelProto.prototype.training_info = $util.emptyArray;
    
            ModelProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.ModelProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.ir_version = reader.int64();
                        break;
                    case 8:
                        if (!(message.opset_import && message.opset_import.length))
                            message.opset_import = [];
                        message.opset_import.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        message.producer_name = reader.string();
                        break;
                    case 3:
                        message.producer_version = reader.string();
                        break;
                    case 4:
                        message.domain = reader.string();
                        break;
                    case 5:
                        message.model_version = reader.int64();
                        break;
                    case 6:
                        message.doc_string = reader.string();
                        break;
                    case 7:
                        message.graph = $root.onnx.GraphProto.decode(reader, reader.uint32());
                        break;
                    case 14:
                        if (!(message.metadata_props && message.metadata_props.length))
                            message.metadata_props = [];
                        message.metadata_props.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    case 20:
                        if (!(message.training_info && message.training_info.length))
                            message.training_info = [];
                        message.training_info.push($root.onnx.TrainingInfoProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ModelProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.ModelProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "ir_version":
                        message.ir_version = reader.int64();
                        break;
                    case "opset_import":
                        if (!(message.opset_import && message.opset_import.length))
                            message.opset_import = [];
                        message.opset_import.push($root.onnx.OperatorSetIdProto.decodeText(reader, true));
                        break;
                    case "producer_name":
                        message.producer_name = reader.string();
                        break;
                    case "producer_version":
                        message.producer_version = reader.string();
                        break;
                    case "domain":
                        message.domain = reader.string();
                        break;
                    case "model_version":
                        message.model_version = reader.int64();
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "graph":
                        message.graph = $root.onnx.GraphProto.decodeText(reader, true);
                        break;
                    case "metadata_props":
                        if (!(message.metadata_props && message.metadata_props.length))
                            message.metadata_props = [];
                        message.metadata_props.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                        break;
                    case "training_info":
                        if (!(message.training_info && message.training_info.length))
                            message.training_info = [];
                        message.training_info.push($root.onnx.TrainingInfoProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ModelProto;
        })();
    
        onnx.StringStringEntryProto = (function() {
    
            function StringStringEntryProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            StringStringEntryProto.prototype.key = "";
            StringStringEntryProto.prototype.value = "";
    
            StringStringEntryProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.StringStringEntryProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.key = reader.string();
                        break;
                    case 2:
                        message.value = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            StringStringEntryProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.StringStringEntryProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "key":
                        message.key = reader.string();
                        break;
                    case "value":
                        message.value = reader.string();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return StringStringEntryProto;
        })();
    
        onnx.TensorAnnotation = (function() {
    
            function TensorAnnotation(properties) {
                this.quant_parameter_tensor_names = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorAnnotation.prototype.tensor_name = "";
            TensorAnnotation.prototype.quant_parameter_tensor_names = $util.emptyArray;
    
            TensorAnnotation.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorAnnotation();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.tensor_name = reader.string();
                        break;
                    case 2:
                        if (!(message.quant_parameter_tensor_names && message.quant_parameter_tensor_names.length))
                            message.quant_parameter_tensor_names = [];
                        message.quant_parameter_tensor_names.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorAnnotation.decodeText = function decodeText(reader) {
                var message = new $root.onnx.TensorAnnotation();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "tensor_name":
                        message.tensor_name = reader.string();
                        break;
                    case "quant_parameter_tensor_names":
                        if (!(message.quant_parameter_tensor_names && message.quant_parameter_tensor_names.length))
                            message.quant_parameter_tensor_names = [];
                        message.quant_parameter_tensor_names.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return TensorAnnotation;
        })();
    
        onnx.GraphProto = (function() {
    
            function GraphProto(properties) {
                this.node = [];
                this.initializer = [];
                this.sparse_initializer = [];
                this.input = [];
                this.output = [];
                this.value_info = [];
                this.quantization_annotation = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            GraphProto.prototype.node = $util.emptyArray;
            GraphProto.prototype.name = "";
            GraphProto.prototype.initializer = $util.emptyArray;
            GraphProto.prototype.sparse_initializer = $util.emptyArray;
            GraphProto.prototype.doc_string = "";
            GraphProto.prototype.input = $util.emptyArray;
            GraphProto.prototype.output = $util.emptyArray;
            GraphProto.prototype.value_info = $util.emptyArray;
            GraphProto.prototype.quantization_annotation = $util.emptyArray;
    
            GraphProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.GraphProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        message.name = reader.string();
                        break;
                    case 5:
                        if (!(message.initializer && message.initializer.length))
                            message.initializer = [];
                        message.initializer.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                        break;
                    case 15:
                        if (!(message.sparse_initializer && message.sparse_initializer.length))
                            message.sparse_initializer = [];
                        message.sparse_initializer.push($root.onnx.SparseTensorProto.decode(reader, reader.uint32()));
                        break;
                    case 10:
                        message.doc_string = reader.string();
                        break;
                    case 11:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    case 12:
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    case 13:
                        if (!(message.value_info && message.value_info.length))
                            message.value_info = [];
                        message.value_info.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    case 14:
                        if (!(message.quantization_annotation && message.quantization_annotation.length))
                            message.quantization_annotation = [];
                        message.quantization_annotation.push($root.onnx.TensorAnnotation.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            GraphProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.GraphProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "node":
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decodeText(reader, true));
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "initializer":
                        if (!(message.initializer && message.initializer.length))
                            message.initializer = [];
                        message.initializer.push($root.onnx.TensorProto.decodeText(reader, true));
                        break;
                    case "sparse_initializer":
                        if (!(message.sparse_initializer && message.sparse_initializer.length))
                            message.sparse_initializer = [];
                        message.sparse_initializer.push($root.onnx.SparseTensorProto.decodeText(reader, true));
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                        break;
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                        break;
                    case "value_info":
                        if (!(message.value_info && message.value_info.length))
                            message.value_info = [];
                        message.value_info.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                        break;
                    case "quantization_annotation":
                        if (!(message.quantization_annotation && message.quantization_annotation.length))
                            message.quantization_annotation = [];
                        message.quantization_annotation.push($root.onnx.TensorAnnotation.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return GraphProto;
        })();
    
        onnx.TensorProto = (function() {
    
            function TensorProto(properties) {
                this.dims = [];
                this.float_data = [];
                this.int32_data = [];
                this.string_data = [];
                this.int64_data = [];
                this.external_data = [];
                this.double_data = [];
                this.uint64_data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProto.prototype.dims = $util.emptyArray;
            TensorProto.prototype.data_type = 0;
            TensorProto.prototype.segment = null;
            TensorProto.prototype.float_data = $util.emptyArray;
            TensorProto.prototype.int32_data = $util.emptyArray;
            TensorProto.prototype.string_data = $util.emptyArray;
            TensorProto.prototype.int64_data = $util.emptyArray;
            TensorProto.prototype.name = "";
            TensorProto.prototype.doc_string = "";
            TensorProto.prototype.raw_data = $util.newBuffer([]);
            TensorProto.prototype.external_data = $util.emptyArray;
            TensorProto.prototype.data_location = 0;
            TensorProto.prototype.double_data = $util.emptyArray;
            TensorProto.prototype.uint64_data = $util.emptyArray;
    
            TensorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dims.push(reader.int64());
                        } else
                            message.dims.push(reader.int64());
                        break;
                    case 2:
                        message.data_type = reader.int32();
                        break;
                    case 3:
                        message.segment = $root.onnx.TensorProto.Segment.decode(reader, reader.uint32());
                        break;
                    case 4:
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.float_data.length == 0 && (end2 - reader.pos) > 1048576) {
                                var float_dataLength = end2 - reader.pos;
                                var float_dataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, float_dataLength);
                                float_dataLength = float_dataLength >>> 2;
                                var float_data = new Float32Array(float_dataLength);
                                for (var i = 0; i < float_dataLength; i++) {
                                    float_data[i] = float_dataView.getFloat32(i << 2, true);
                                }
                                message.float_data = float_data;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.float_data.push(reader.float());
                            }
                        } else
                            message.float_data.push(reader.float());
                        break;
                    case 5:
                        if (!(message.int32_data && message.int32_data.length))
                            message.int32_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int32_data.push(reader.int32());
                        } else
                            message.int32_data.push(reader.int32());
                        break;
                    case 6:
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        message.string_data.push(reader.bytes());
                        break;
                    case 7:
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64_data.push(reader.int64());
                        } else
                            message.int64_data.push(reader.int64());
                        break;
                    case 8:
                        message.name = reader.string();
                        break;
                    case 12:
                        message.doc_string = reader.string();
                        break;
                    case 9:
                        message.raw_data = reader.bytes();
                        break;
                    case 13:
                        if (!(message.external_data && message.external_data.length))
                            message.external_data = [];
                        message.external_data.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    case 14:
                        message.data_location = reader.int32();
                        break;
                    case 10:
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.double_data.length == 0 && (end2 - reader.pos) > 1048576) {
                                var double_dataLength = end2 - reader.pos;
                                var double_dataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, double_dataLength);
                                double_dataLength = double_dataLength >>> 3;
                                var double_data = new Float64Array(double_dataLength);
                                for (var i = 0; i < double_dataLength; i++) {
                                    double_data[i] = double_dataView.getFloat64(i << 3, true);
                                }
                                message.double_data = double_data;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.double_data.push(reader.double());
                            }
                        } else
                            message.double_data.push(reader.double());
                        break;
                    case 11:
                        if (!(message.uint64_data && message.uint64_data.length))
                            message.uint64_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint64_data.push(reader.uint64());
                        } else
                            message.uint64_data.push(reader.uint64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.TensorProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dims.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.dims.push(reader.int64());
                        break;
                    case "data_type":
                        message.data_type = reader.int32();
                        break;
                    case "segment":
                        message.segment = $root.onnx.TensorProto.Segment.decodeText(reader, true);
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
                    case "int32_data":
                        if (!(message.int32_data && message.int32_data.length))
                            message.int32_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.int32_data.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.int32_data.push(reader.int32());
                        break;
                    case "string_data":
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.string_data.push(reader.bytes());
                                reader.next();
                            }
                        else
                            message.string_data.push(reader.bytes());
                        break;
                    case "int64_data":
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.int64_data.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.int64_data.push(reader.int64());
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "raw_data":
                        message.raw_data = reader.bytes();
                        break;
                    case "external_data":
                        if (!(message.external_data && message.external_data.length))
                            message.external_data = [];
                        message.external_data.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                        break;
                    case "data_location":
                        message.data_location = reader.enum($root.onnx.TensorProto.DataLocation);
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
                    case "uint64_data":
                        if (!(message.uint64_data && message.uint64_data.length))
                            message.uint64_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.uint64_data.push(reader.uint64());
                                reader.next();
                            }
                        else
                            message.uint64_data.push(reader.uint64());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TensorProto.DataType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNDEFINED"] = 0;
                values[valuesById[1] = "FLOAT"] = 1;
                values[valuesById[2] = "UINT8"] = 2;
                values[valuesById[3] = "INT8"] = 3;
                values[valuesById[4] = "UINT16"] = 4;
                values[valuesById[5] = "INT16"] = 5;
                values[valuesById[6] = "INT32"] = 6;
                values[valuesById[7] = "INT64"] = 7;
                values[valuesById[8] = "STRING"] = 8;
                values[valuesById[9] = "BOOL"] = 9;
                values[valuesById[10] = "FLOAT16"] = 10;
                values[valuesById[11] = "DOUBLE"] = 11;
                values[valuesById[12] = "UINT32"] = 12;
                values[valuesById[13] = "UINT64"] = 13;
                values[valuesById[14] = "COMPLEX64"] = 14;
                values[valuesById[15] = "COMPLEX128"] = 15;
                values[valuesById[16] = "BFLOAT16"] = 16;
                return values;
            })();
    
            TensorProto.Segment = (function() {
    
                function Segment(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Segment.prototype.begin = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Segment.prototype.end = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                Segment.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorProto.Segment();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.begin = reader.int64();
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
                };
    
                Segment.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TensorProto.Segment();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "begin":
                            message.begin = reader.int64();
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
                };
    
                return Segment;
            })();
    
            TensorProto.DataLocation = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "EXTERNAL"] = 1;
                return values;
            })();
    
            return TensorProto;
        })();
    
        onnx.SparseTensorProto = (function() {
    
            function SparseTensorProto(properties) {
                this.dims = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SparseTensorProto.prototype.values = null;
            SparseTensorProto.prototype.indices = null;
            SparseTensorProto.prototype.dims = $util.emptyArray;
    
            SparseTensorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.SparseTensorProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.values = $root.onnx.TensorProto.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.indices = $root.onnx.TensorProto.decode(reader, reader.uint32());
                        break;
                    case 3:
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dims.push(reader.int64());
                        } else
                            message.dims.push(reader.int64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SparseTensorProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.SparseTensorProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "values":
                        message.values = $root.onnx.TensorProto.decodeText(reader, true);
                        break;
                    case "indices":
                        message.indices = $root.onnx.TensorProto.decodeText(reader, true);
                        break;
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dims.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.dims.push(reader.int64());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return SparseTensorProto;
        })();
    
        onnx.TensorShapeProto = (function() {
    
            function TensorShapeProto(properties) {
                this.dim = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorShapeProto.prototype.dim = $util.emptyArray;
    
            TensorShapeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorShapeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        message.dim.push($root.onnx.TensorShapeProto.Dimension.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorShapeProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.TensorShapeProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dim":
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        message.dim.push($root.onnx.TensorShapeProto.Dimension.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TensorShapeProto.Dimension = (function() {
    
                function Dimension(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Dimension.prototype.dim_value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Dimension.prototype.dim_param = "";
                Dimension.prototype.denotation = "";
    
                var $oneOfFields;
    
                Object.defineProperty(Dimension.prototype, "value", {
                    get: $util.oneOfGetter($oneOfFields = ["dim_value", "dim_param"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Dimension.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorShapeProto.Dimension();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.dim_value = reader.int64();
                            break;
                        case 2:
                            message.dim_param = reader.string();
                            break;
                        case 3:
                            message.denotation = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Dimension.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TensorShapeProto.Dimension();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "dim_value":
                            message.dim_value = reader.int64();
                            break;
                        case "dim_param":
                            message.dim_param = reader.string();
                            break;
                        case "denotation":
                            message.denotation = reader.string();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                return Dimension;
            })();
    
            return TensorShapeProto;
        })();
    
        onnx.TypeProto = (function() {
    
            function TypeProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TypeProto.prototype.tensor_type = null;
            TypeProto.prototype.sequence_type = null;
            TypeProto.prototype.map_type = null;
            TypeProto.prototype.sparse_tensor_type = null;
            TypeProto.prototype.opaque_type = null;
            TypeProto.prototype.denotation = "";
    
            var $oneOfFields;
    
            Object.defineProperty(TypeProto.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["tensor_type", "sequence_type", "map_type", "sparse_tensor_type", "opaque_type"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            TypeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.tensor_type = $root.onnx.TypeProto.Tensor.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.sequence_type = $root.onnx.TypeProto.Sequence.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.map_type = $root.onnx.TypeProto.Map.decode(reader, reader.uint32());
                        break;
                    case 8:
                        message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.opaque_type = $root.onnx.TypeProto.Opaque.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.denotation = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TypeProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.TypeProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "tensor_type":
                        message.tensor_type = $root.onnx.TypeProto.Tensor.decodeText(reader, true);
                        break;
                    case "sequence_type":
                        message.sequence_type = $root.onnx.TypeProto.Sequence.decodeText(reader, true);
                        break;
                    case "map_type":
                        message.map_type = $root.onnx.TypeProto.Map.decodeText(reader, true);
                        break;
                    case "sparse_tensor_type":
                        message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.decodeText(reader, true);
                        break;
                    case "opaque_type":
                        message.opaque_type = $root.onnx.TypeProto.Opaque.decodeText(reader, true);
                        break;
                    case "denotation":
                        message.denotation = reader.string();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TypeProto.Tensor = (function() {
    
                function Tensor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Tensor.prototype.elem_type = 0;
                Tensor.prototype.shape = null;
    
                Tensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Tensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elem_type = reader.int32();
                            break;
                        case 2:
                            message.shape = $root.onnx.TensorShapeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Tensor.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TypeProto.Tensor();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "elem_type":
                            message.elem_type = reader.int32();
                            break;
                        case "shape":
                            message.shape = $root.onnx.TensorShapeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                return Tensor;
            })();
    
            TypeProto.Sequence = (function() {
    
                function Sequence(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Sequence.prototype.elem_type = null;
    
                Sequence.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Sequence();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elem_type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Sequence.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TypeProto.Sequence();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "elem_type":
                            message.elem_type = $root.onnx.TypeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                return Sequence;
            })();
    
            TypeProto.Map = (function() {
    
                function Map(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Map.prototype.key_type = 0;
                Map.prototype.value_type = null;
    
                Map.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Map();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.key_type = reader.int32();
                            break;
                        case 2:
                            message.value_type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Map.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TypeProto.Map();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "key_type":
                            message.key_type = reader.int32();
                            break;
                        case "value_type":
                            message.value_type = $root.onnx.TypeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                return Map;
            })();
    
            TypeProto.SparseTensor = (function() {
    
                function SparseTensor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SparseTensor.prototype.elem_type = 0;
                SparseTensor.prototype.shape = null;
    
                SparseTensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.SparseTensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elem_type = reader.int32();
                            break;
                        case 2:
                            message.shape = $root.onnx.TensorShapeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                SparseTensor.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TypeProto.SparseTensor();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "elem_type":
                            message.elem_type = reader.int32();
                            break;
                        case "shape":
                            message.shape = $root.onnx.TensorShapeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                return SparseTensor;
            })();
    
            TypeProto.Opaque = (function() {
    
                function Opaque(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Opaque.prototype.domain = "";
                Opaque.prototype.name = "";
    
                Opaque.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Opaque();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.domain = reader.string();
                            break;
                        case 2:
                            message.name = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Opaque.decodeText = function decodeText(reader) {
                    var message = new $root.onnx.TypeProto.Opaque();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "domain":
                            message.domain = reader.string();
                            break;
                        case "name":
                            message.name = reader.string();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                return Opaque;
            })();
    
            return TypeProto;
        })();
    
        onnx.OperatorSetIdProto = (function() {
    
            function OperatorSetIdProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorSetIdProto.prototype.domain = "";
            OperatorSetIdProto.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            OperatorSetIdProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.OperatorSetIdProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.domain = reader.string();
                        break;
                    case 2:
                        message.version = reader.int64();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OperatorSetIdProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.OperatorSetIdProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "domain":
                        message.domain = reader.string();
                        break;
                    case "version":
                        message.version = reader.int64();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return OperatorSetIdProto;
        })();
    
        onnx.OperatorStatus = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "EXPERIMENTAL"] = 0;
            values[valuesById[1] = "STABLE"] = 1;
            return values;
        })();
    
        onnx.FunctionProto = (function() {
    
            function FunctionProto(properties) {
                this.input = [];
                this.output = [];
                this.attribute = [];
                this.node = [];
                this.opset_import = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FunctionProto.prototype.name = "";
            FunctionProto.prototype.since_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            FunctionProto.prototype.status = 0;
            FunctionProto.prototype.input = $util.emptyArray;
            FunctionProto.prototype.output = $util.emptyArray;
            FunctionProto.prototype.attribute = $util.emptyArray;
            FunctionProto.prototype.node = $util.emptyArray;
            FunctionProto.prototype.doc_string = "";
            FunctionProto.prototype.opset_import = $util.emptyArray;
    
            FunctionProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.FunctionProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.since_version = reader.int64();
                        break;
                    case 3:
                        message.status = reader.int32();
                        break;
                    case 4:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 5:
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push(reader.string());
                        break;
                    case 6:
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push(reader.string());
                        break;
                    case 7:
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                        break;
                    case 8:
                        message.doc_string = reader.string();
                        break;
                    case 9:
                        if (!(message.opset_import && message.opset_import.length))
                            message.opset_import = [];
                        message.opset_import.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FunctionProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.FunctionProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "since_version":
                        message.since_version = reader.int64();
                        break;
                    case "status":
                        message.status = reader.enum($root.onnx.OperatorStatus);
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
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.output.push(reader.string());
                                reader.next();
                            }
                        else
                            message.output.push(reader.string());
                        break;
                    case "attribute":
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.attribute.push(reader.string());
                                reader.next();
                            }
                        else
                            message.attribute.push(reader.string());
                        break;
                    case "node":
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decodeText(reader, true));
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "opset_import":
                        if (!(message.opset_import && message.opset_import.length))
                            message.opset_import = [];
                        message.opset_import.push($root.onnx.OperatorSetIdProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return FunctionProto;
        })();
    
        onnx.OperatorProto = (function() {
    
            function OperatorProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorProto.prototype.op_type = "";
            OperatorProto.prototype.since_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorProto.prototype.status = 0;
            OperatorProto.prototype.doc_string = "";
    
            OperatorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.OperatorProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.op_type = reader.string();
                        break;
                    case 2:
                        message.since_version = reader.int64();
                        break;
                    case 3:
                        message.status = reader.int32();
                        break;
                    case 10:
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OperatorProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.OperatorProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "op_type":
                        message.op_type = reader.string();
                        break;
                    case "since_version":
                        message.since_version = reader.int64();
                        break;
                    case "status":
                        message.status = reader.enum($root.onnx.OperatorStatus);
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return OperatorProto;
        })();
    
        onnx.OperatorSetProto = (function() {
    
            function OperatorSetProto(properties) {
                this.operator = [];
                this.functions = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorSetProto.prototype.magic = "";
            OperatorSetProto.prototype.ir_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorSetProto.prototype.ir_version_prerelease = "";
            OperatorSetProto.prototype.ir_build_metadata = "";
            OperatorSetProto.prototype.domain = "";
            OperatorSetProto.prototype.opset_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorSetProto.prototype.doc_string = "";
            OperatorSetProto.prototype.operator = $util.emptyArray;
            OperatorSetProto.prototype.functions = $util.emptyArray;
    
            OperatorSetProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.OperatorSetProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.magic = reader.string();
                        break;
                    case 2:
                        message.ir_version = reader.int64();
                        break;
                    case 3:
                        message.ir_version_prerelease = reader.string();
                        break;
                    case 7:
                        message.ir_build_metadata = reader.string();
                        break;
                    case 4:
                        message.domain = reader.string();
                        break;
                    case 5:
                        message.opset_version = reader.int64();
                        break;
                    case 6:
                        message.doc_string = reader.string();
                        break;
                    case 8:
                        if (!(message.operator && message.operator.length))
                            message.operator = [];
                        message.operator.push($root.onnx.OperatorProto.decode(reader, reader.uint32()));
                        break;
                    case 9:
                        if (!(message.functions && message.functions.length))
                            message.functions = [];
                        message.functions.push($root.onnx.FunctionProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OperatorSetProto.decodeText = function decodeText(reader) {
                var message = new $root.onnx.OperatorSetProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "magic":
                        message.magic = reader.string();
                        break;
                    case "ir_version":
                        message.ir_version = reader.int64();
                        break;
                    case "ir_version_prerelease":
                        message.ir_version_prerelease = reader.string();
                        break;
                    case "ir_build_metadata":
                        message.ir_build_metadata = reader.string();
                        break;
                    case "domain":
                        message.domain = reader.string();
                        break;
                    case "opset_version":
                        message.opset_version = reader.int64();
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "operator":
                        if (!(message.operator && message.operator.length))
                            message.operator = [];
                        message.operator.push($root.onnx.OperatorProto.decodeText(reader, true));
                        break;
                    case "functions":
                        if (!(message.functions && message.functions.length))
                            message.functions = [];
                        message.functions.push($root.onnx.FunctionProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return OperatorSetProto;
        })();
    
        return onnx;
    })();

    return $root;
})(protobuf);
