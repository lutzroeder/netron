(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('onnx');

    $root.onnx = (function() {

        const onnx = {};

        onnx.Version = (function() {
            const values = {};
            values["_START_VERSION"] = 0;
            values["IR_VERSION_2017_10_10"] = 1;
            values["IR_VERSION_2017_10_30"] = 2;
            values["IR_VERSION_2017_11_3"] = 3;
            values["IR_VERSION_2019_1_22"] = 4;
            values["IR_VERSION_2019_3_18"] = 5;
            values["IR_VERSION_2019_9_19"] = 6;
            values["IR_VERSION"] = 7;
            return values;
        })();

        onnx.AttributeProto = (function() {

            function AttributeProto() {
                this.floats = [];
                this.ints = [];
                this.strings = [];
                this.tensors = [];
                this.graphs = [];
                this.sparse_tensors = [];
            }

            AttributeProto.prototype.name = "";
            AttributeProto.prototype.ref_attr_name = "";
            AttributeProto.prototype.doc_string = "";
            AttributeProto.prototype.type = 0;
            AttributeProto.prototype.f = 0;
            AttributeProto.prototype.i = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            AttributeProto.prototype.s = new Uint8Array([]);
            AttributeProto.prototype.t = null;
            AttributeProto.prototype.g = null;
            AttributeProto.prototype.sparse_tensor = null;
            AttributeProto.prototype.floats = [];
            AttributeProto.prototype.ints = [];
            AttributeProto.prototype.strings = [];
            AttributeProto.prototype.tensors = [];
            AttributeProto.prototype.graphs = [];
            AttributeProto.prototype.sparse_tensors = [];

            AttributeProto.decode = function (reader, length) {
                const message = new $root.onnx.AttributeProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                            message.floats = reader.floats(message.floats, tag);
                            break;
                        case 8:
                            message.ints = reader.array(message.ints, () => reader.int64(), tag);
                            break;
                        case 9:
                            message.strings.push(reader.bytes());
                            break;
                        case 10:
                            message.tensors.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                            break;
                        case 11:
                            message.graphs.push($root.onnx.GraphProto.decode(reader, reader.uint32()));
                            break;
                        case 23:
                            message.sparse_tensors.push($root.onnx.SparseTensorProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            AttributeProto.decodeText = function (reader) {
                const message = new $root.onnx.AttributeProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                            reader.array(message.floats, () => reader.float());
                            break;
                        case "ints":
                            reader.array(message.ints, () => reader.int64());
                            break;
                        case "strings":
                            reader.array(message.strings, () => reader.bytes());
                            break;
                        case "tensors":
                            message.tensors.push($root.onnx.TensorProto.decodeText(reader, true));
                            break;
                        case "graphs":
                            message.graphs.push($root.onnx.GraphProto.decodeText(reader, true));
                            break;
                        case "sparse_tensors":
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
                const values = {};
                values["UNDEFINED"] = 0;
                values["FLOAT"] = 1;
                values["INT"] = 2;
                values["STRING"] = 3;
                values["TENSOR"] = 4;
                values["GRAPH"] = 5;
                values["SPARSE_TENSOR"] = 11;
                values["FLOATS"] = 6;
                values["INTS"] = 7;
                values["STRINGS"] = 8;
                values["TENSORS"] = 9;
                values["GRAPHS"] = 10;
                values["SPARSE_TENSORS"] = 12;
                return values;
            })();

            return AttributeProto;
        })();

        onnx.ValueInfoProto = (function() {

            function ValueInfoProto() {
            }

            ValueInfoProto.prototype.name = "";
            ValueInfoProto.prototype.type = null;
            ValueInfoProto.prototype.doc_string = "";

            ValueInfoProto.decode = function (reader, length) {
                const message = new $root.onnx.ValueInfoProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            ValueInfoProto.decodeText = function (reader) {
                const message = new $root.onnx.ValueInfoProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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

            function NodeProto() {
                this.input = [];
                this.output = [];
                this.attribute = [];
            }

            NodeProto.prototype.input = [];
            NodeProto.prototype.output = [];
            NodeProto.prototype.name = "";
            NodeProto.prototype.op_type = "";
            NodeProto.prototype.domain = "";
            NodeProto.prototype.attribute = [];
            NodeProto.prototype.doc_string = "";

            NodeProto.decode = function (reader, length) {
                const message = new $root.onnx.NodeProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.input.push(reader.string());
                            break;
                        case 2:
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

            NodeProto.decodeText = function (reader) {
                const message = new $root.onnx.NodeProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "input":
                            reader.array(message.input, () => reader.string());
                            break;
                        case "output":
                            reader.array(message.output, () => reader.string());
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

            function TrainingInfoProto() {
                this.initialization_binding = [];
                this.update_binding = [];
            }

            TrainingInfoProto.prototype.initialization = null;
            TrainingInfoProto.prototype.algorithm = null;
            TrainingInfoProto.prototype.initialization_binding = [];
            TrainingInfoProto.prototype.update_binding = [];

            TrainingInfoProto.decode = function (reader, length) {
                const message = new $root.onnx.TrainingInfoProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.initialization = $root.onnx.GraphProto.decode(reader, reader.uint32());
                            break;
                        case 2:
                            message.algorithm = $root.onnx.GraphProto.decode(reader, reader.uint32());
                            break;
                        case 3:
                            message.initialization_binding.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                            break;
                        case 4:
                            message.update_binding.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TrainingInfoProto.decodeText = function (reader) {
                const message = new $root.onnx.TrainingInfoProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "initialization":
                            message.initialization = $root.onnx.GraphProto.decodeText(reader, true);
                            break;
                        case "algorithm":
                            message.algorithm = $root.onnx.GraphProto.decodeText(reader, true);
                            break;
                        case "initialization_binding":
                            message.initialization_binding.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                            break;
                        case "update_binding":
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

            function ModelProto() {
                this.opset_import = [];
                this.metadata_props = [];
                this.training_info = [];
            }

            ModelProto.prototype.ir_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            ModelProto.prototype.opset_import = [];
            ModelProto.prototype.producer_name = "";
            ModelProto.prototype.producer_version = "";
            ModelProto.prototype.domain = "";
            ModelProto.prototype.model_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            ModelProto.prototype.doc_string = "";
            ModelProto.prototype.graph = null;
            ModelProto.prototype.metadata_props = [];
            ModelProto.prototype.training_info = [];

            ModelProto.decode = function (reader, length) {
                const message = new $root.onnx.ModelProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.ir_version = reader.int64();
                            break;
                        case 8:
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
                            message.metadata_props.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                            break;
                        case 20:
                            message.training_info.push($root.onnx.TrainingInfoProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ModelProto.decodeText = function (reader) {
                const message = new $root.onnx.ModelProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "ir_version":
                            message.ir_version = reader.int64();
                            break;
                        case "opset_import":
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
                            message.metadata_props.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                            break;
                        case "training_info":
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

            function StringStringEntryProto() {
            }

            StringStringEntryProto.prototype.key = "";
            StringStringEntryProto.prototype.value = "";

            StringStringEntryProto.decode = function (reader, length) {
                const message = new $root.onnx.StringStringEntryProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            StringStringEntryProto.decodeText = function (reader) {
                const message = new $root.onnx.StringStringEntryProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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

            function TensorAnnotation() {
                this.quant_parameter_tensor_names = [];
            }

            TensorAnnotation.prototype.tensor_name = "";
            TensorAnnotation.prototype.quant_parameter_tensor_names = [];

            TensorAnnotation.decode = function (reader, length) {
                const message = new $root.onnx.TensorAnnotation();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.tensor_name = reader.string();
                            break;
                        case 2:
                            message.quant_parameter_tensor_names.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorAnnotation.decodeText = function (reader) {
                const message = new $root.onnx.TensorAnnotation();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "tensor_name":
                            message.tensor_name = reader.string();
                            break;
                        case "quant_parameter_tensor_names":
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

            function GraphProto() {
                this.node = [];
                this.initializer = [];
                this.sparse_initializer = [];
                this.input = [];
                this.output = [];
                this.value_info = [];
                this.quantization_annotation = [];
            }

            GraphProto.prototype.node = [];
            GraphProto.prototype.name = "";
            GraphProto.prototype.initializer = [];
            GraphProto.prototype.sparse_initializer = [];
            GraphProto.prototype.doc_string = "";
            GraphProto.prototype.input = [];
            GraphProto.prototype.output = [];
            GraphProto.prototype.value_info = [];
            GraphProto.prototype.quantization_annotation = [];

            GraphProto.decode = function (reader, length) {
                const message = new $root.onnx.GraphProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            message.name = reader.string();
                            break;
                        case 5:
                            message.initializer.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                            break;
                        case 15:
                            message.sparse_initializer.push($root.onnx.SparseTensorProto.decode(reader, reader.uint32()));
                            break;
                        case 10:
                            message.doc_string = reader.string();
                            break;
                        case 11:
                            message.input.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                            break;
                        case 12:
                            message.output.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                            break;
                        case 13:
                            message.value_info.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                            break;
                        case 14:
                            message.quantization_annotation.push($root.onnx.TensorAnnotation.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            GraphProto.decodeText = function (reader) {
                const message = new $root.onnx.GraphProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "node":
                            message.node.push($root.onnx.NodeProto.decodeText(reader, true));
                            break;
                        case "name":
                            message.name = reader.string();
                            break;
                        case "initializer":
                            message.initializer.push($root.onnx.TensorProto.decodeText(reader, true));
                            break;
                        case "sparse_initializer":
                            message.sparse_initializer.push($root.onnx.SparseTensorProto.decodeText(reader, true));
                            break;
                        case "doc_string":
                            message.doc_string = reader.string();
                            break;
                        case "input":
                            message.input.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                            break;
                        case "output":
                            message.output.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                            break;
                        case "value_info":
                            message.value_info.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                            break;
                        case "quantization_annotation":
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

            function TensorProto() {
                this.dims = [];
                this.float_data = [];
                this.int32_data = [];
                this.string_data = [];
                this.int64_data = [];
                this.external_data = [];
                this.double_data = [];
                this.uint64_data = [];
            }

            TensorProto.prototype.dims = [];
            TensorProto.prototype.data_type = 0;
            TensorProto.prototype.segment = null;
            TensorProto.prototype.float_data = [];
            TensorProto.prototype.int32_data = [];
            TensorProto.prototype.string_data = [];
            TensorProto.prototype.int64_data = [];
            TensorProto.prototype.name = "";
            TensorProto.prototype.doc_string = "";
            TensorProto.prototype.raw_data = new Uint8Array([]);
            TensorProto.prototype.external_data = [];
            TensorProto.prototype.data_location = 0;
            TensorProto.prototype.double_data = [];
            TensorProto.prototype.uint64_data = [];

            TensorProto.decode = function (reader, length) {
                const message = new $root.onnx.TensorProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.dims = reader.array(message.dims, () => reader.int64(), tag);
                            break;
                        case 2:
                            message.data_type = reader.int32();
                            break;
                        case 3:
                            message.segment = $root.onnx.TensorProto.Segment.decode(reader, reader.uint32());
                            break;
                        case 4:
                            message.float_data = reader.floats(message.float_data, tag);
                            break;
                        case 5:
                            message.int32_data = reader.array(message.int32_data, () => reader.int32(), tag);
                            break;
                        case 6:
                            message.string_data.push(reader.bytes());
                            break;
                        case 7:
                            message.int64_data = reader.array(message.int64_data, () => reader.int64(), tag);
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
                            message.external_data.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                            break;
                        case 14:
                            message.data_location = reader.int32();
                            break;
                        case 10:
                            message.double_data = reader.doubles(message.double_data, tag);
                            break;
                        case 11:
                            message.uint64_data = reader.array(message.uint64_data, () => reader.uint64(), tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorProto.decodeText = function (reader) {
                const message = new $root.onnx.TensorProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dims":
                            reader.array(message.dims, () => reader.int64());
                            break;
                        case "data_type":
                            message.data_type = reader.int32();
                            break;
                        case "segment":
                            message.segment = $root.onnx.TensorProto.Segment.decodeText(reader, true);
                            break;
                        case "float_data":
                            reader.array(message.float_data, () => reader.float());
                            break;
                        case "int32_data":
                            reader.array(message.int32_data, () => reader.int32());
                            break;
                        case "string_data":
                            reader.array(message.string_data, () => reader.bytes());
                            break;
                        case "int64_data":
                            reader.array(message.int64_data, () => reader.int64());
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
                            message.external_data.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                            break;
                        case "data_location":
                            message.data_location = reader.enum($root.onnx.TensorProto.DataLocation);
                            break;
                        case "double_data":
                            reader.array(message.double_data, () => reader.double());
                            break;
                        case "uint64_data":
                            reader.array(message.uint64_data, () => reader.uint64());
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            TensorProto.DataType = (function() {
                const values = {};
                values["UNDEFINED"] = 0;
                values["FLOAT"] = 1;
                values["UINT8"] = 2;
                values["INT8"] = 3;
                values["UINT16"] = 4;
                values["INT16"] = 5;
                values["INT32"] = 6;
                values["INT64"] = 7;
                values["STRING"] = 8;
                values["BOOL"] = 9;
                values["FLOAT16"] = 10;
                values["DOUBLE"] = 11;
                values["UINT32"] = 12;
                values["UINT64"] = 13;
                values["COMPLEX64"] = 14;
                values["COMPLEX128"] = 15;
                values["BFLOAT16"] = 16;
                return values;
            })();

            TensorProto.Segment = (function() {

                function Segment() {
                }

                Segment.prototype.begin = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Segment.prototype.end = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                Segment.decode = function (reader, length) {
                    const message = new $root.onnx.TensorProto.Segment();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                Segment.decodeText = function (reader) {
                    const message = new $root.onnx.TensorProto.Segment();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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
                const values = {};
                values["DEFAULT"] = 0;
                values["EXTERNAL"] = 1;
                return values;
            })();

            return TensorProto;
        })();

        onnx.SparseTensorProto = (function() {

            function SparseTensorProto() {
                this.dims = [];
            }

            SparseTensorProto.prototype.values = null;
            SparseTensorProto.prototype.indices = null;
            SparseTensorProto.prototype.dims = [];

            SparseTensorProto.decode = function (reader, length) {
                const message = new $root.onnx.SparseTensorProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.values = $root.onnx.TensorProto.decode(reader, reader.uint32());
                            break;
                        case 2:
                            message.indices = $root.onnx.TensorProto.decode(reader, reader.uint32());
                            break;
                        case 3:
                            message.dims = reader.array(message.dims, () => reader.int64(), tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            SparseTensorProto.decodeText = function (reader) {
                const message = new $root.onnx.SparseTensorProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "values":
                            message.values = $root.onnx.TensorProto.decodeText(reader, true);
                            break;
                        case "indices":
                            message.indices = $root.onnx.TensorProto.decodeText(reader, true);
                            break;
                        case "dims":
                            reader.array(message.dims, () => reader.int64());
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

            function TensorShapeProto() {
                this.dim = [];
            }

            TensorShapeProto.prototype.dim = [];

            TensorShapeProto.decode = function (reader, length) {
                const message = new $root.onnx.TensorShapeProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.dim.push($root.onnx.TensorShapeProto.Dimension.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorShapeProto.decodeText = function (reader) {
                const message = new $root.onnx.TensorShapeProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dim":
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

                function Dimension() {
                }

                Dimension.prototype.dim_value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Dimension.prototype.dim_param = "";
                Dimension.prototype.denotation = "";

                const valueSet = new Set([ "dim_value", "dim_param"]);
                Object.defineProperty(Dimension.prototype, "value", {
                    get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
                });

                Dimension.decode = function (reader, length) {
                    const message = new $root.onnx.TensorShapeProto.Dimension();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                Dimension.decodeText = function (reader) {
                    const message = new $root.onnx.TensorShapeProto.Dimension();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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

            function TypeProto() {
            }

            TypeProto.prototype.tensor_type = null;
            TypeProto.prototype.sequence_type = null;
            TypeProto.prototype.map_type = null;
            TypeProto.prototype.sparse_tensor_type = null;
            TypeProto.prototype.opaque_type = null;
            TypeProto.prototype.denotation = "";

            const valueSet = new Set([ "tensor_type", "sequence_type", "map_type", "sparse_tensor_type", "opaque_type"]);
            Object.defineProperty(TypeProto.prototype, "value", {
                get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
            });

            TypeProto.decode = function (reader, length) {
                const message = new $root.onnx.TypeProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            TypeProto.decodeText = function (reader) {
                const message = new $root.onnx.TypeProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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

                function Tensor() {
                }

                Tensor.prototype.elem_type = 0;
                Tensor.prototype.shape = null;

                Tensor.decode = function (reader, length) {
                    const message = new $root.onnx.TypeProto.Tensor();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                Tensor.decodeText = function (reader) {
                    const message = new $root.onnx.TypeProto.Tensor();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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

                function Sequence() {
                }

                Sequence.prototype.elem_type = null;

                Sequence.decode = function (reader, length) {
                    const message = new $root.onnx.TypeProto.Sequence();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                Sequence.decodeText = function (reader) {
                    const message = new $root.onnx.TypeProto.Sequence();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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

                function Map() {
                }

                Map.prototype.key_type = 0;
                Map.prototype.value_type = null;

                Map.decode = function (reader, length) {
                    const message = new $root.onnx.TypeProto.Map();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                Map.decodeText = function (reader) {
                    const message = new $root.onnx.TypeProto.Map();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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

                function SparseTensor() {
                }

                SparseTensor.prototype.elem_type = 0;
                SparseTensor.prototype.shape = null;

                SparseTensor.decode = function (reader, length) {
                    const message = new $root.onnx.TypeProto.SparseTensor();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                SparseTensor.decodeText = function (reader) {
                    const message = new $root.onnx.TypeProto.SparseTensor();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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

                function Opaque() {
                }

                Opaque.prototype.domain = "";
                Opaque.prototype.name = "";

                Opaque.decode = function (reader, length) {
                    const message = new $root.onnx.TypeProto.Opaque();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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

                Opaque.decodeText = function (reader) {
                    const message = new $root.onnx.TypeProto.Opaque();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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

            function OperatorSetIdProto() {
            }

            OperatorSetIdProto.prototype.domain = "";
            OperatorSetIdProto.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

            OperatorSetIdProto.decode = function (reader, length) {
                const message = new $root.onnx.OperatorSetIdProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            OperatorSetIdProto.decodeText = function (reader) {
                const message = new $root.onnx.OperatorSetIdProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
            const values = {};
            values["EXPERIMENTAL"] = 0;
            values["STABLE"] = 1;
            return values;
        })();

        onnx.FunctionProto = (function() {

            function FunctionProto() {
                this.input = [];
                this.output = [];
                this.attribute = [];
                this.node = [];
                this.opset_import = [];
            }

            FunctionProto.prototype.name = "";
            FunctionProto.prototype.since_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            FunctionProto.prototype.status = 0;
            FunctionProto.prototype.input = [];
            FunctionProto.prototype.output = [];
            FunctionProto.prototype.attribute = [];
            FunctionProto.prototype.node = [];
            FunctionProto.prototype.doc_string = "";
            FunctionProto.prototype.opset_import = [];

            FunctionProto.decode = function (reader, length) {
                const message = new $root.onnx.FunctionProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                            message.input.push(reader.string());
                            break;
                        case 5:
                            message.output.push(reader.string());
                            break;
                        case 6:
                            message.attribute.push(reader.string());
                            break;
                        case 7:
                            message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                            break;
                        case 8:
                            message.doc_string = reader.string();
                            break;
                        case 9:
                            message.opset_import.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            FunctionProto.decodeText = function (reader) {
                const message = new $root.onnx.FunctionProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                            reader.array(message.input, () => reader.string());
                            break;
                        case "output":
                            reader.array(message.output, () => reader.string());
                            break;
                        case "attribute":
                            reader.array(message.attribute, () => reader.string());
                            break;
                        case "node":
                            message.node.push($root.onnx.NodeProto.decodeText(reader, true));
                            break;
                        case "doc_string":
                            message.doc_string = reader.string();
                            break;
                        case "opset_import":
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

            function OperatorProto() {
            }

            OperatorProto.prototype.op_type = "";
            OperatorProto.prototype.since_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            OperatorProto.prototype.status = 0;
            OperatorProto.prototype.doc_string = "";

            OperatorProto.decode = function (reader, length) {
                const message = new $root.onnx.OperatorProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            OperatorProto.decodeText = function (reader) {
                const message = new $root.onnx.OperatorProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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

            function OperatorSetProto() {
                this.operator = [];
                this.functions = [];
            }

            OperatorSetProto.prototype.magic = "";
            OperatorSetProto.prototype.ir_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            OperatorSetProto.prototype.ir_version_prerelease = "";
            OperatorSetProto.prototype.ir_build_metadata = "";
            OperatorSetProto.prototype.domain = "";
            OperatorSetProto.prototype.opset_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            OperatorSetProto.prototype.doc_string = "";
            OperatorSetProto.prototype.operator = [];
            OperatorSetProto.prototype.functions = [];

            OperatorSetProto.decode = function (reader, length) {
                const message = new $root.onnx.OperatorSetProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                            message.operator.push($root.onnx.OperatorProto.decode(reader, reader.uint32()));
                            break;
                        case 9:
                            message.functions.push($root.onnx.FunctionProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            OperatorSetProto.decodeText = function (reader) {
                const message = new $root.onnx.OperatorSetProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                            message.operator.push($root.onnx.OperatorProto.decodeText(reader, true));
                            break;
                        case "functions":
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
