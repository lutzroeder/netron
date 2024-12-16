
export const onnx = {};

onnx.Version = {
    "_START_VERSION": 0,
    "IR_VERSION_2017_10_10": 1,
    "IR_VERSION_2017_10_30": 2,
    "IR_VERSION_2017_11_3": 3,
    "IR_VERSION_2019_1_22": 4,
    "IR_VERSION_2019_3_18": 5,
    "IR_VERSION_2019_9_19": 6,
    "IR_VERSION_2020_5_8": 7,
    "IR_VERSION_2021_7_30": 8,
    "IR_VERSION_2023_5_5": 9,
    "IR_VERSION_2024_3_25": 10,
    "IR_VERSION": 11
};

onnx.AttributeProto = class AttributeProto {

    constructor() {
        this.floats = [];
        this.ints = [];
        this.strings = [];
        this.tensors = [];
        this.graphs = [];
        this.sparse_tensors = [];
        this.type_protos = [];
    }

    static decode(reader, length) {
        const message = new onnx.AttributeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.t = onnx.TensorProto.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.g = onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.sparse_tensor = onnx.SparseTensorProto.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.tp = onnx.TypeProto.decode(reader, reader.uint32());
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
                    message.tensors.push(onnx.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.graphs.push(onnx.GraphProto.decode(reader, reader.uint32()));
                    break;
                case 23:
                    message.sparse_tensors.push(onnx.SparseTensorProto.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.type_protos.push(onnx.TypeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.AttributeProto();
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
                    message.type = reader.enum(onnx.AttributeProto.AttributeType);
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
                    message.t = onnx.TensorProto.decodeText(reader);
                    break;
                case "g":
                    message.g = onnx.GraphProto.decodeText(reader);
                    break;
                case "sparse_tensor":
                    message.sparse_tensor = onnx.SparseTensorProto.decodeText(reader);
                    break;
                case "tp":
                    message.tp = onnx.TypeProto.decodeText(reader);
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
                    message.tensors.push(onnx.TensorProto.decodeText(reader));
                    break;
                case "graphs":
                    message.graphs.push(onnx.GraphProto.decodeText(reader));
                    break;
                case "sparse_tensors":
                    message.sparse_tensors.push(onnx.SparseTensorProto.decodeText(reader));
                    break;
                case "type_protos":
                    message.type_protos.push(onnx.TypeProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.AttributeProto();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.refAttrName !== undefined) {
            message.ref_attr_name = obj.refAttrName;
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.type !== undefined) {
            message.type = onnx.AttributeProto.AttributeType[obj.type];
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
            message.t = onnx.TensorProto.decodeJson(obj.t);
        }
        if (obj.g !== undefined) {
            message.g = onnx.GraphProto.decodeJson(obj.g);
        }
        if (obj.sparseTensor !== undefined) {
            message.sparse_tensor = onnx.SparseTensorProto.decodeJson(obj.sparseTensor);
        }
        if (obj.tp !== undefined) {
            message.tp = onnx.TypeProto.decodeJson(obj.tp);
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
            message.tensors = obj.tensors.map((obj) => onnx.TensorProto.decodeJson(obj));
        }
        if (obj.graphs !== undefined) {
            message.graphs = obj.graphs.map((obj) => onnx.GraphProto.decodeJson(obj));
        }
        if (obj.sparseTensors !== undefined) {
            message.sparse_tensors = obj.sparseTensors.map((obj) => onnx.SparseTensorProto.decodeJson(obj));
        }
        if (obj.typeProtos !== undefined) {
            message.type_protos = obj.typeProtos.map((obj) => onnx.TypeProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.AttributeProto.prototype.name = "";
onnx.AttributeProto.prototype.ref_attr_name = "";
onnx.AttributeProto.prototype.doc_string = "";
onnx.AttributeProto.prototype.type = 0;
onnx.AttributeProto.prototype.f = 0;
onnx.AttributeProto.prototype.i = 0n;
onnx.AttributeProto.prototype.s = new Uint8Array([]);
onnx.AttributeProto.prototype.t = null;
onnx.AttributeProto.prototype.g = null;
onnx.AttributeProto.prototype.sparse_tensor = null;
onnx.AttributeProto.prototype.tp = null;

onnx.AttributeProto.AttributeType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "INT": 2,
    "STRING": 3,
    "TENSOR": 4,
    "GRAPH": 5,
    "SPARSE_TENSOR": 11,
    "TYPE_PROTO": 13,
    "FLOATS": 6,
    "INTS": 7,
    "STRINGS": 8,
    "TENSORS": 9,
    "GRAPHS": 10,
    "SPARSE_TENSORS": 12,
    "TYPE_PROTOS": 14
};

onnx.ValueInfoProto = class ValueInfoProto {

    constructor() {
        this.metadata_props = [];
    }

    static decode(reader, length) {
        const message = new onnx.ValueInfoProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.doc_string = reader.string();
                    break;
                case 4:
                    message.metadata_props.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.ValueInfoProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = onnx.TypeProto.decodeText(reader);
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "metadata_props":
                    message.metadata_props.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.ValueInfoProto();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.type !== undefined) {
            message.type = onnx.TypeProto.decodeJson(obj.type);
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.metadataProps !== undefined) {
            message.metadata_props = obj.metadataProps.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.ValueInfoProto.prototype.name = "";
onnx.ValueInfoProto.prototype.type = null;
onnx.ValueInfoProto.prototype.doc_string = "";

onnx.NodeProto = class NodeProto {

    constructor() {
        this.input = [];
        this.output = [];
        this.attribute = [];
        this.metadata_props = [];
    }

    static decode(reader, length) {
        const message = new onnx.NodeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                case 8:
                    message.overload = reader.string();
                    break;
                case 5:
                    message.attribute.push(onnx.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                case 9:
                    message.metadata_props.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.NodeProto();
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
                case "overload":
                    message.overload = reader.string();
                    break;
                case "attribute":
                    message.attribute.push(onnx.AttributeProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "metadata_props":
                    message.metadata_props.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.NodeProto();
        if (obj.input !== undefined) {
            message.input = obj.input;
        }
        if (obj.output !== undefined) {
            message.output = obj.output;
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.opType !== undefined) {
            message.op_type = obj.opType;
        }
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.overload !== undefined) {
            message.overload = obj.overload;
        }
        if (obj.attribute !== undefined) {
            message.attribute = obj.attribute.map((obj) => onnx.AttributeProto.decodeJson(obj));
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.metadataProps !== undefined) {
            message.metadata_props = obj.metadataProps.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.NodeProto.prototype.name = "";
onnx.NodeProto.prototype.op_type = "";
onnx.NodeProto.prototype.domain = "";
onnx.NodeProto.prototype.overload = "";
onnx.NodeProto.prototype.doc_string = "";

onnx.TrainingInfoProto = class TrainingInfoProto {

    constructor() {
        this.initialization_binding = [];
        this.update_binding = [];
    }

    static decode(reader, length) {
        const message = new onnx.TrainingInfoProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.initialization = onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.algorithm = onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.initialization_binding.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.update_binding.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TrainingInfoProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "initialization":
                    message.initialization = onnx.GraphProto.decodeText(reader);
                    break;
                case "algorithm":
                    message.algorithm = onnx.GraphProto.decodeText(reader);
                    break;
                case "initialization_binding":
                    message.initialization_binding.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                case "update_binding":
                    message.update_binding.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TrainingInfoProto();
        if (obj.initialization !== undefined) {
            message.initialization = onnx.GraphProto.decodeJson(obj.initialization);
        }
        if (obj.algorithm !== undefined) {
            message.algorithm = onnx.GraphProto.decodeJson(obj.algorithm);
        }
        if (obj.initializationBinding !== undefined) {
            message.initialization_binding = obj.initializationBinding.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        if (obj.updateBinding !== undefined) {
            message.update_binding = obj.updateBinding.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.TrainingInfoProto.prototype.initialization = null;
onnx.TrainingInfoProto.prototype.algorithm = null;

onnx.ModelProto = class ModelProto {

    constructor() {
        this.opset_import = [];
        this.metadata_props = [];
        this.training_info = [];
        this.functions = [];
    }

    static decode(reader, length) {
        const message = new onnx.ModelProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ir_version = reader.int64();
                    break;
                case 8:
                    message.opset_import.push(onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
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
                    message.graph = onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.metadata_props.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                case 20:
                    message.training_info.push(onnx.TrainingInfoProto.decode(reader, reader.uint32()));
                    break;
                case 25:
                    message.functions.push(onnx.FunctionProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.ModelProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ir_version":
                    message.ir_version = reader.int64();
                    break;
                case "opset_import":
                    message.opset_import.push(onnx.OperatorSetIdProto.decodeText(reader));
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
                    message.graph = onnx.GraphProto.decodeText(reader);
                    break;
                case "metadata_props":
                    message.metadata_props.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                case "training_info":
                    message.training_info.push(onnx.TrainingInfoProto.decodeText(reader));
                    break;
                case "functions":
                    message.functions.push(onnx.FunctionProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.ModelProto();
        if (obj.irVersion !== undefined) {
            message.ir_version = BigInt(obj.irVersion);
        }
        if (obj.opsetImport !== undefined) {
            message.opset_import = obj.opsetImport.map((obj) => onnx.OperatorSetIdProto.decodeJson(obj));
        }
        if (obj.producerName !== undefined) {
            message.producer_name = obj.producerName;
        }
        if (obj.producerVersion !== undefined) {
            message.producer_version = obj.producerVersion;
        }
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.modelVersion !== undefined) {
            message.model_version = BigInt(obj.modelVersion);
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.graph !== undefined) {
            message.graph = onnx.GraphProto.decodeJson(obj.graph);
        }
        if (obj.metadataProps !== undefined) {
            message.metadata_props = obj.metadataProps.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        if (obj.trainingInfo !== undefined) {
            message.training_info = obj.trainingInfo.map((obj) => onnx.TrainingInfoProto.decodeJson(obj));
        }
        if (obj.functions !== undefined) {
            message.functions = obj.functions.map((obj) => onnx.FunctionProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.ModelProto.prototype.ir_version = 0n;
onnx.ModelProto.prototype.producer_name = "";
onnx.ModelProto.prototype.producer_version = "";
onnx.ModelProto.prototype.domain = "";
onnx.ModelProto.prototype.model_version = 0n;
onnx.ModelProto.prototype.doc_string = "";
onnx.ModelProto.prototype.graph = null;

onnx.StringStringEntryProto = class StringStringEntryProto {

    static decode(reader, length) {
        const message = new onnx.StringStringEntryProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new onnx.StringStringEntryProto();
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
    }

    static decodeJson(obj) {
        const message = new onnx.StringStringEntryProto();
        if (obj.key !== undefined) {
            message.key = obj.key;
        }
        if (obj.value !== undefined) {
            message.value = obj.value;
        }
        return message;
    }
};

onnx.StringStringEntryProto.prototype.key = "";
onnx.StringStringEntryProto.prototype.value = "";

onnx.TensorAnnotation = class TensorAnnotation {

    constructor() {
        this.quant_parameter_tensor_names = [];
    }

    static decode(reader, length) {
        const message = new onnx.TensorAnnotation();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_name = reader.string();
                    break;
                case 2:
                    message.quant_parameter_tensor_names.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TensorAnnotation();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_name":
                    message.tensor_name = reader.string();
                    break;
                case "quant_parameter_tensor_names":
                    message.quant_parameter_tensor_names.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TensorAnnotation();
        if (obj.tensorName !== undefined) {
            message.tensor_name = obj.tensorName;
        }
        if (obj.quantParameterTensorNames !== undefined) {
            message.quant_parameter_tensor_names = obj.quantParameterTensorNames.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.TensorAnnotation.prototype.tensor_name = "";

onnx.GraphProto = class GraphProto {

    constructor() {
        this.node = [];
        this.initializer = [];
        this.sparse_initializer = [];
        this.input = [];
        this.output = [];
        this.value_info = [];
        this.quantization_annotation = [];
        this.metadata_props = [];
    }

    static decode(reader, length) {
        const message = new onnx.GraphProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push(onnx.NodeProto.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 5:
                    message.initializer.push(onnx.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.sparse_initializer.push(onnx.SparseTensorProto.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.doc_string = reader.string();
                    break;
                case 11:
                    message.input.push(onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 12:
                    message.output.push(onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.value_info.push(onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 14:
                    message.quantization_annotation.push(onnx.TensorAnnotation.decode(reader, reader.uint32()));
                    break;
                case 16:
                    message.metadata_props.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.GraphProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node":
                    message.node.push(onnx.NodeProto.decodeText(reader));
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "initializer":
                    message.initializer.push(onnx.TensorProto.decodeText(reader));
                    break;
                case "sparse_initializer":
                    message.sparse_initializer.push(onnx.SparseTensorProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "input":
                    message.input.push(onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "output":
                    message.output.push(onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "value_info":
                    message.value_info.push(onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "quantization_annotation":
                    message.quantization_annotation.push(onnx.TensorAnnotation.decodeText(reader));
                    break;
                case "metadata_props":
                    message.metadata_props.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.GraphProto();
        if (obj.node !== undefined) {
            message.node = obj.node.map((obj) => onnx.NodeProto.decodeJson(obj));
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.initializer !== undefined) {
            message.initializer = obj.initializer.map((obj) => onnx.TensorProto.decodeJson(obj));
        }
        if (obj.sparseInitializer !== undefined) {
            message.sparse_initializer = obj.sparseInitializer.map((obj) => onnx.SparseTensorProto.decodeJson(obj));
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.input !== undefined) {
            message.input = obj.input.map((obj) => onnx.ValueInfoProto.decodeJson(obj));
        }
        if (obj.output !== undefined) {
            message.output = obj.output.map((obj) => onnx.ValueInfoProto.decodeJson(obj));
        }
        if (obj.valueInfo !== undefined) {
            message.value_info = obj.valueInfo.map((obj) => onnx.ValueInfoProto.decodeJson(obj));
        }
        if (obj.quantizationAnnotation !== undefined) {
            message.quantization_annotation = obj.quantizationAnnotation.map((obj) => onnx.TensorAnnotation.decodeJson(obj));
        }
        if (obj.metadataProps !== undefined) {
            message.metadata_props = obj.metadataProps.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.GraphProto.prototype.name = "";
onnx.GraphProto.prototype.doc_string = "";

onnx.TensorProto = class TensorProto {

    constructor() {
        this.dims = [];
        this.float_data = [];
        this.int32_data = [];
        this.string_data = [];
        this.int64_data = [];
        this.external_data = [];
        this.double_data = [];
        this.uint64_data = [];
        this.metadata_props = [];
    }

    static decode(reader, length) {
        const message = new onnx.TensorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dims = reader.array(message.dims, () => reader.int64(), tag);
                    break;
                case 2:
                    message.data_type = reader.int32();
                    break;
                case 3:
                    message.segment = onnx.TensorProto.Segment.decode(reader, reader.uint32());
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
                    message.external_data.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
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
                case 16:
                    message.metadata_props.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TensorProto();
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
                    message.segment = onnx.TensorProto.Segment.decodeText(reader);
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
                    message.external_data.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                case "data_location":
                    message.data_location = reader.enum(onnx.TensorProto.DataLocation);
                    break;
                case "double_data":
                    reader.array(message.double_data, () => reader.double());
                    break;
                case "uint64_data":
                    reader.array(message.uint64_data, () => reader.uint64());
                    break;
                case "metadata_props":
                    message.metadata_props.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TensorProto();
        if (obj.dims !== undefined) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        if (obj.dataType !== undefined) {
            message.data_type = Number(obj.dataType);
        }
        if (obj.segment !== undefined) {
            message.segment = onnx.TensorProto.Segment.decodeJson(obj.segment);
        }
        if (obj.floatData !== undefined) {
            message.float_data = obj.floatData.map((obj) => Number(obj));
        }
        if (obj.int32Data !== undefined) {
            message.int32_data = obj.int32Data.map((obj) => Number(obj));
        }
        if (obj.stringData !== undefined) {
            message.string_data = obj.stringData.map((obj) => new Uint8Array(atob(obj)));
        }
        if (obj.int64Data !== undefined) {
            message.int64_data = obj.int64Data.map((obj) => BigInt(obj));
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.rawData !== undefined) {
            message.raw_data = new Uint8Array(atob(obj.rawData));
        }
        if (obj.externalData !== undefined) {
            message.external_data = obj.externalData.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        if (obj.dataLocation !== undefined) {
            message.data_location = onnx.TensorProto.DataLocation[obj.dataLocation];
        }
        if (obj.doubleData !== undefined) {
            message.double_data = obj.doubleData.map((obj) => Number(obj));
        }
        if (obj.uint64Data !== undefined) {
            message.uint64_data = obj.uint64Data.map((obj) => BigInt(obj));
        }
        if (obj.metadataProps !== undefined) {
            message.metadata_props = obj.metadataProps.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.TensorProto.prototype.data_type = 0;
onnx.TensorProto.prototype.segment = null;
onnx.TensorProto.prototype.name = "";
onnx.TensorProto.prototype.doc_string = "";
onnx.TensorProto.prototype.raw_data = new Uint8Array([]);
onnx.TensorProto.prototype.data_location = 0;

onnx.TensorProto.DataType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "UINT8": 2,
    "INT8": 3,
    "UINT16": 4,
    "INT16": 5,
    "INT32": 6,
    "INT64": 7,
    "STRING": 8,
    "BOOL": 9,
    "FLOAT16": 10,
    "DOUBLE": 11,
    "UINT32": 12,
    "UINT64": 13,
    "COMPLEX64": 14,
    "COMPLEX128": 15,
    "BFLOAT16": 16,
    "FLOAT8E4M3FN": 17,
    "FLOAT8E4M3FNUZ": 18,
    "FLOAT8E5M2": 19,
    "FLOAT8E5M2FNUZ": 20,
    "UINT4": 21,
    "INT4": 22,
    "FLOAT4E2M1": 23
};

onnx.TensorProto.Segment = class Segment {

    static decode(reader, length) {
        const message = new onnx.TensorProto.Segment();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new onnx.TensorProto.Segment();
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
    }

    static decodeJson(obj) {
        const message = new onnx.TensorProto.Segment();
        if (obj.begin !== undefined) {
            message.begin = BigInt(obj.begin);
        }
        if (obj.end !== undefined) {
            message.end = BigInt(obj.end);
        }
        return message;
    }
};

onnx.TensorProto.Segment.prototype.begin = 0n;
onnx.TensorProto.Segment.prototype.end = 0n;

onnx.TensorProto.DataLocation = {
    "DEFAULT": 0,
    "EXTERNAL": 1
};

onnx.SparseTensorProto = class SparseTensorProto {

    constructor() {
        this.dims = [];
    }

    static decode(reader, length) {
        const message = new onnx.SparseTensorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = onnx.TensorProto.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.indices = onnx.TensorProto.decode(reader, reader.uint32());
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
    }

    static decodeText(reader) {
        const message = new onnx.SparseTensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values = onnx.TensorProto.decodeText(reader);
                    break;
                case "indices":
                    message.indices = onnx.TensorProto.decodeText(reader);
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
    }

    static decodeJson(obj) {
        const message = new onnx.SparseTensorProto();
        if (obj.values !== undefined) {
            message.values = onnx.TensorProto.decodeJson(obj.values);
        }
        if (obj.indices !== undefined) {
            message.indices = onnx.TensorProto.decodeJson(obj.indices);
        }
        if (obj.dims !== undefined) {
            message.dims = obj.dims.map((obj) => BigInt(obj));
        }
        return message;
    }
};

onnx.SparseTensorProto.prototype.values = null;
onnx.SparseTensorProto.prototype.indices = null;

onnx.TensorShapeProto = class TensorShapeProto {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new onnx.TensorShapeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim.push(onnx.TensorShapeProto.Dimension.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TensorShapeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim.push(onnx.TensorShapeProto.Dimension.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TensorShapeProto();
        if (obj.dim !== undefined) {
            message.dim = obj.dim.map((obj) => onnx.TensorShapeProto.Dimension.decodeJson(obj));
        }
        return message;
    }
};

onnx.TensorShapeProto.Dimension = class Dimension {

    get value() {
        onnx.TensorShapeProto.Dimension.valueSet = onnx.TensorShapeProto.Dimension.valueSet || new Set(["dim_value", "dim_param"]);
        return Object.keys(this).find((key) => onnx.TensorShapeProto.Dimension.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new onnx.TensorShapeProto.Dimension();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new onnx.TensorShapeProto.Dimension();
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
    }

    static decodeJson(obj) {
        const message = new onnx.TensorShapeProto.Dimension();
        if (obj.dimValue !== undefined) {
            message.dim_value = BigInt(obj.dimValue);
        }
        if (obj.dimParam !== undefined) {
            message.dim_param = obj.dimParam;
        }
        if (obj.denotation !== undefined) {
            message.denotation = obj.denotation;
        }
        return message;
    }
};

onnx.TensorShapeProto.Dimension.prototype.denotation = "";

onnx.TypeProto = class TypeProto {

    get value() {
        onnx.TypeProto.valueSet = onnx.TypeProto.valueSet || new Set(["tensor_type", "sequence_type", "map_type", "optional_type", "sparse_tensor_type", "opaque_type"]);
        return Object.keys(this).find((key) => onnx.TypeProto.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new onnx.TypeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_type = onnx.TypeProto.Tensor.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.sequence_type = onnx.TypeProto.Sequence.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.map_type = onnx.TypeProto.Map.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.optional_type = onnx.TypeProto.Optional.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.sparse_tensor_type = onnx.TypeProto.SparseTensor.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.opaque_type = onnx.TypeProto.Opaque.decode(reader, reader.uint32());
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
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_type":
                    message.tensor_type = onnx.TypeProto.Tensor.decodeText(reader);
                    break;
                case "sequence_type":
                    message.sequence_type = onnx.TypeProto.Sequence.decodeText(reader);
                    break;
                case "map_type":
                    message.map_type = onnx.TypeProto.Map.decodeText(reader);
                    break;
                case "optional_type":
                    message.optional_type = onnx.TypeProto.Optional.decodeText(reader);
                    break;
                case "sparse_tensor_type":
                    message.sparse_tensor_type = onnx.TypeProto.SparseTensor.decodeText(reader);
                    break;
                case "opaque_type":
                    message.opaque_type = onnx.TypeProto.Opaque.decodeText(reader);
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
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto();
        if (obj.tensorType !== undefined) {
            message.tensor_type = onnx.TypeProto.Tensor.decodeJson(obj.tensorType);
        }
        if (obj.sequenceType !== undefined) {
            message.sequence_type = onnx.TypeProto.Sequence.decodeJson(obj.sequenceType);
        }
        if (obj.mapType !== undefined) {
            message.map_type = onnx.TypeProto.Map.decodeJson(obj.mapType);
        }
        if (obj.optionalType !== undefined) {
            message.optional_type = onnx.TypeProto.Optional.decodeJson(obj.optionalType);
        }
        if (obj.sparseTensorType !== undefined) {
            message.sparse_tensor_type = onnx.TypeProto.SparseTensor.decodeJson(obj.sparseTensorType);
        }
        if (obj.opaqueType !== undefined) {
            message.opaque_type = onnx.TypeProto.Opaque.decodeJson(obj.opaqueType);
        }
        if (obj.denotation !== undefined) {
            message.denotation = obj.denotation;
        }
        return message;
    }
};

onnx.TypeProto.prototype.denotation = "";

onnx.TypeProto.Tensor = class Tensor {

    static decode(reader, length) {
        const message = new onnx.TypeProto.Tensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = reader.int32();
                    break;
                case 2:
                    message.shape = onnx.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto.Tensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = reader.int32();
                    break;
                case "shape":
                    message.shape = onnx.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto.Tensor();
        if (obj.elemType !== undefined) {
            message.elem_type = Number(obj.elemType);
        }
        if (obj.shape !== undefined) {
            message.shape = onnx.TensorShapeProto.decodeJson(obj.shape);
        }
        return message;
    }
};

onnx.TypeProto.Tensor.prototype.elem_type = 0;
onnx.TypeProto.Tensor.prototype.shape = null;

onnx.TypeProto.Sequence = class Sequence {

    static decode(reader, length) {
        const message = new onnx.TypeProto.Sequence();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto.Sequence();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = onnx.TypeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto.Sequence();
        if (obj.elemType !== undefined) {
            message.elem_type = onnx.TypeProto.decodeJson(obj.elemType);
        }
        return message;
    }
};

onnx.TypeProto.Sequence.prototype.elem_type = null;

onnx.TypeProto.Map = class Map {

    static decode(reader, length) {
        const message = new onnx.TypeProto.Map();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key_type = reader.int32();
                    break;
                case 2:
                    message.value_type = onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto.Map();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key_type":
                    message.key_type = reader.int32();
                    break;
                case "value_type":
                    message.value_type = onnx.TypeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto.Map();
        if (obj.keyType !== undefined) {
            message.key_type = Number(obj.keyType);
        }
        if (obj.valueType !== undefined) {
            message.value_type = onnx.TypeProto.decodeJson(obj.valueType);
        }
        return message;
    }
};

onnx.TypeProto.Map.prototype.key_type = 0;
onnx.TypeProto.Map.prototype.value_type = null;

onnx.TypeProto.Optional = class Optional {

    static decode(reader, length) {
        const message = new onnx.TypeProto.Optional();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto.Optional();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = onnx.TypeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto.Optional();
        if (obj.elemType !== undefined) {
            message.elem_type = onnx.TypeProto.decodeJson(obj.elemType);
        }
        return message;
    }
};

onnx.TypeProto.Optional.prototype.elem_type = null;

onnx.TypeProto.SparseTensor = class SparseTensor {

    static decode(reader, length) {
        const message = new onnx.TypeProto.SparseTensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = reader.int32();
                    break;
                case 2:
                    message.shape = onnx.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto.SparseTensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = reader.int32();
                    break;
                case "shape":
                    message.shape = onnx.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto.SparseTensor();
        if (obj.elemType !== undefined) {
            message.elem_type = Number(obj.elemType);
        }
        if (obj.shape !== undefined) {
            message.shape = onnx.TensorShapeProto.decodeJson(obj.shape);
        }
        return message;
    }
};

onnx.TypeProto.SparseTensor.prototype.elem_type = 0;
onnx.TypeProto.SparseTensor.prototype.shape = null;

onnx.TypeProto.Opaque = class Opaque {

    static decode(reader, length) {
        const message = new onnx.TypeProto.Opaque();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new onnx.TypeProto.Opaque();
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
    }

    static decodeJson(obj) {
        const message = new onnx.TypeProto.Opaque();
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        return message;
    }
};

onnx.TypeProto.Opaque.prototype.domain = "";
onnx.TypeProto.Opaque.prototype.name = "";

onnx.OperatorSetIdProto = class OperatorSetIdProto {

    static decode(reader, length) {
        const message = new onnx.OperatorSetIdProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new onnx.OperatorSetIdProto();
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
    }

    static decodeJson(obj) {
        const message = new onnx.OperatorSetIdProto();
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.version !== undefined) {
            message.version = BigInt(obj.version);
        }
        return message;
    }
};

onnx.OperatorSetIdProto.prototype.domain = "";
onnx.OperatorSetIdProto.prototype.version = 0n;

onnx.OperatorStatus = {
    "EXPERIMENTAL": 0,
    "STABLE": 1
};

onnx.FunctionProto = class FunctionProto {

    constructor() {
        this.input = [];
        this.output = [];
        this.attribute = [];
        this.attribute_proto = [];
        this.node = [];
        this.opset_import = [];
        this.value_info = [];
        this.metadata_props = [];
    }

    static decode(reader, length) {
        const message = new onnx.FunctionProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
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
                case 11:
                    message.attribute_proto.push(onnx.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.node.push(onnx.NodeProto.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.doc_string = reader.string();
                    break;
                case 9:
                    message.opset_import.push(onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.domain = reader.string();
                    break;
                case 13:
                    message.overload = reader.string();
                    break;
                case 12:
                    message.value_info.push(onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 14:
                    message.metadata_props.push(onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.FunctionProto();
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
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                case "attribute":
                    reader.array(message.attribute, () => reader.string());
                    break;
                case "attribute_proto":
                    message.attribute_proto.push(onnx.AttributeProto.decodeText(reader));
                    break;
                case "node":
                    message.node.push(onnx.NodeProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "opset_import":
                    message.opset_import.push(onnx.OperatorSetIdProto.decodeText(reader));
                    break;
                case "domain":
                    message.domain = reader.string();
                    break;
                case "overload":
                    message.overload = reader.string();
                    break;
                case "value_info":
                    message.value_info.push(onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "metadata_props":
                    message.metadata_props.push(onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.FunctionProto();
        if (obj.name !== undefined) {
            message.name = obj.name;
        }
        if (obj.input !== undefined) {
            message.input = obj.input;
        }
        if (obj.output !== undefined) {
            message.output = obj.output;
        }
        if (obj.attribute !== undefined) {
            message.attribute = obj.attribute;
        }
        if (obj.attributeProto !== undefined) {
            message.attribute_proto = obj.attributeProto.map((obj) => onnx.AttributeProto.decodeJson(obj));
        }
        if (obj.node !== undefined) {
            message.node = obj.node.map((obj) => onnx.NodeProto.decodeJson(obj));
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.opsetImport !== undefined) {
            message.opset_import = obj.opsetImport.map((obj) => onnx.OperatorSetIdProto.decodeJson(obj));
        }
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.overload !== undefined) {
            message.overload = obj.overload;
        }
        if (obj.valueInfo !== undefined) {
            message.value_info = obj.valueInfo.map((obj) => onnx.ValueInfoProto.decodeJson(obj));
        }
        if (obj.metadataProps !== undefined) {
            message.metadata_props = obj.metadataProps.map((obj) => onnx.StringStringEntryProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.FunctionProto.prototype.name = "";
onnx.FunctionProto.prototype.doc_string = "";
onnx.FunctionProto.prototype.domain = "";
onnx.FunctionProto.prototype.overload = "";

onnx.OperatorProto = class OperatorProto {

    static decode(reader, length) {
        const message = new onnx.OperatorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new onnx.OperatorProto();
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
                    message.status = reader.enum(onnx.OperatorStatus);
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
    }

    static decodeJson(obj) {
        const message = new onnx.OperatorProto();
        if (obj.opType !== undefined) {
            message.op_type = obj.opType;
        }
        if (obj.sinceVersion !== undefined) {
            message.since_version = BigInt(obj.sinceVersion);
        }
        if (obj.status !== undefined) {
            message.status = onnx.OperatorStatus[obj.status];
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        return message;
    }
};

onnx.OperatorProto.prototype.op_type = "";
onnx.OperatorProto.prototype.since_version = 0n;
onnx.OperatorProto.prototype.status = 0;
onnx.OperatorProto.prototype.doc_string = "";

onnx.OperatorSetProto = class OperatorSetProto {

    constructor() {
        this.operator = [];
        this.functions = [];
    }

    static decode(reader, length) {
        const message = new onnx.OperatorSetProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
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
                    message.operator.push(onnx.OperatorProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.functions.push(onnx.FunctionProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new onnx.OperatorSetProto();
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
                    message.operator.push(onnx.OperatorProto.decodeText(reader));
                    break;
                case "functions":
                    message.functions.push(onnx.FunctionProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new onnx.OperatorSetProto();
        if (obj.magic !== undefined) {
            message.magic = obj.magic;
        }
        if (obj.irVersion !== undefined) {
            message.ir_version = BigInt(obj.irVersion);
        }
        if (obj.irVersionPrerelease !== undefined) {
            message.ir_version_prerelease = obj.irVersionPrerelease;
        }
        if (obj.irBuildMetadata !== undefined) {
            message.ir_build_metadata = obj.irBuildMetadata;
        }
        if (obj.domain !== undefined) {
            message.domain = obj.domain;
        }
        if (obj.opsetVersion !== undefined) {
            message.opset_version = BigInt(obj.opsetVersion);
        }
        if (obj.docString !== undefined) {
            message.doc_string = obj.docString;
        }
        if (obj.operator !== undefined) {
            message.operator = obj.operator.map((obj) => onnx.OperatorProto.decodeJson(obj));
        }
        if (obj.functions !== undefined) {
            message.functions = obj.functions.map((obj) => onnx.FunctionProto.decodeJson(obj));
        }
        return message;
    }
};

onnx.OperatorSetProto.prototype.magic = "";
onnx.OperatorSetProto.prototype.ir_version = 0n;
onnx.OperatorSetProto.prototype.ir_version_prerelease = "";
onnx.OperatorSetProto.prototype.ir_build_metadata = "";
onnx.OperatorSetProto.prototype.domain = "";
onnx.OperatorSetProto.prototype.opset_version = 0n;
onnx.OperatorSetProto.prototype.doc_string = "";
