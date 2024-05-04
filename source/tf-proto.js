
export const tensorflow = {};
export const google = {};

tensorflow.SavedModel = class SavedModel {

    constructor() {
        this.meta_graphs = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedModel();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.saved_model_schema_version = reader.int64();
                    break;
                case 2:
                    message.meta_graphs.push(tensorflow.MetaGraphDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "saved_model_schema_version":
                    message.saved_model_schema_version = reader.int64();
                    break;
                case "meta_graphs":
                    message.meta_graphs.push(tensorflow.MetaGraphDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedModel.prototype.saved_model_schema_version = 0n;

tensorflow.MetaGraphDef = class MetaGraphDef {

    constructor() {
        this.collection_def = {};
        this.signature_def = {};
        this.asset_file_def = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.MetaGraphDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meta_info_def = tensorflow.MetaGraphDef.MetaInfoDef.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.graph_def = tensorflow.GraphDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.saver_def = tensorflow.SaverDef.decode(reader, reader.uint32());
                    break;
                case 4:
                    reader.entry(message.collection_def, () => reader.string(), () => tensorflow.CollectionDef.decode(reader, reader.uint32()));
                    break;
                case 5:
                    reader.entry(message.signature_def, () => reader.string(), () => tensorflow.SignatureDef.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.asset_file_def.push(tensorflow.AssetFileDef.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.object_graph_def = tensorflow.SavedObjectGraph.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.MetaGraphDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meta_info_def":
                    message.meta_info_def = tensorflow.MetaGraphDef.MetaInfoDef.decodeText(reader);
                    break;
                case "graph_def":
                    message.graph_def = tensorflow.GraphDef.decodeText(reader);
                    break;
                case "saver_def":
                    message.saver_def = tensorflow.SaverDef.decodeText(reader);
                    break;
                case "collection_def":
                    reader.entry(message.collection_def, () => reader.string(), () => tensorflow.CollectionDef.decodeText(reader));
                    break;
                case "signature_def":
                    reader.entry(message.signature_def, () => reader.string(), () => tensorflow.SignatureDef.decodeText(reader));
                    break;
                case "asset_file_def":
                    message.asset_file_def.push(tensorflow.AssetFileDef.decodeText(reader));
                    break;
                case "object_graph_def":
                    message.object_graph_def = tensorflow.SavedObjectGraph.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.MetaGraphDef.prototype.meta_info_def = null;
tensorflow.MetaGraphDef.prototype.graph_def = null;
tensorflow.MetaGraphDef.prototype.saver_def = null;
tensorflow.MetaGraphDef.prototype.object_graph_def = null;

tensorflow.MetaGraphDef.MetaInfoDef = class MetaInfoDef {

    constructor() {
        this.tags = [];
        this.function_aliases = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.MetaGraphDef.MetaInfoDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meta_graph_version = reader.string();
                    break;
                case 2:
                    message.stripped_op_list = tensorflow.OpList.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.any_info = google.protobuf.Any.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.tags.push(reader.string());
                    break;
                case 5:
                    message.tensorflow_version = reader.string();
                    break;
                case 6:
                    message.tensorflow_git_version = reader.string();
                    break;
                case 7:
                    message.stripped_default_attrs = reader.bool();
                    break;
                case 8:
                    reader.entry(message.function_aliases, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.MetaGraphDef.MetaInfoDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meta_graph_version":
                    message.meta_graph_version = reader.string();
                    break;
                case "stripped_op_list":
                    message.stripped_op_list = tensorflow.OpList.decodeText(reader);
                    break;
                case "any_info":
                    message.any_info = google.protobuf.Any.decodeText(reader);
                    break;
                case "tags":
                    reader.array(message.tags, () => reader.string());
                    break;
                case "tensorflow_version":
                    message.tensorflow_version = reader.string();
                    break;
                case "tensorflow_git_version":
                    message.tensorflow_git_version = reader.string();
                    break;
                case "stripped_default_attrs":
                    message.stripped_default_attrs = reader.bool();
                    break;
                case "function_aliases":
                    reader.entry(message.function_aliases, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.MetaGraphDef.MetaInfoDef.prototype.meta_graph_version = "";
tensorflow.MetaGraphDef.MetaInfoDef.prototype.stripped_op_list = null;
tensorflow.MetaGraphDef.MetaInfoDef.prototype.any_info = null;
tensorflow.MetaGraphDef.MetaInfoDef.prototype.tensorflow_version = "";
tensorflow.MetaGraphDef.MetaInfoDef.prototype.tensorflow_git_version = "";
tensorflow.MetaGraphDef.MetaInfoDef.prototype.stripped_default_attrs = false;

tensorflow.CollectionDef = class CollectionDef {

    get kind() {
        tensorflow.CollectionDef.kindSet = tensorflow.CollectionDef.kindSet || new Set(["node_list", "bytes_list", "int64_list", "float_list", "any_list"]);
        return Object.keys(this).find((key) => tensorflow.CollectionDef.kindSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.CollectionDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node_list = tensorflow.CollectionDef.NodeList.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.bytes_list = tensorflow.CollectionDef.BytesList.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.int64_list = tensorflow.CollectionDef.Int64List.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.float_list = tensorflow.CollectionDef.FloatList.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.any_list = tensorflow.CollectionDef.AnyList.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CollectionDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_list":
                    message.node_list = tensorflow.CollectionDef.NodeList.decodeText(reader);
                    break;
                case "bytes_list":
                    message.bytes_list = tensorflow.CollectionDef.BytesList.decodeText(reader);
                    break;
                case "int64_list":
                    message.int64_list = tensorflow.CollectionDef.Int64List.decodeText(reader);
                    break;
                case "float_list":
                    message.float_list = tensorflow.CollectionDef.FloatList.decodeText(reader);
                    break;
                case "any_list":
                    message.any_list = tensorflow.CollectionDef.AnyList.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CollectionDef.NodeList = class NodeList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CollectionDef.NodeList();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CollectionDef.NodeList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CollectionDef.BytesList = class BytesList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CollectionDef.BytesList();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push(reader.bytes());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CollectionDef.BytesList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.bytes());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CollectionDef.Int64List = class Int64List {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CollectionDef.Int64List();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CollectionDef.Int64List();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CollectionDef.FloatList = class FloatList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CollectionDef.FloatList();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.floats(message.value, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CollectionDef.FloatList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CollectionDef.AnyList = class AnyList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CollectionDef.AnyList();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push(google.protobuf.Any.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CollectionDef.AnyList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.anyarray(message.value, () => new google.protobuf.Any());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorInfo = class TensorInfo {

    get encoding() {
        tensorflow.TensorInfo.encodingSet = tensorflow.TensorInfo.encodingSet || new Set(["name", "coo_sparse", "composite_tensor"]);
        return Object.keys(this).find((key) => tensorflow.TensorInfo.encodingSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.TensorInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 4:
                    message.coo_sparse = tensorflow.TensorInfo.CooSparse.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.composite_tensor = tensorflow.TensorInfo.CompositeTensor.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.dtype = reader.int32();
                    break;
                case 3:
                    message.tensor_shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "coo_sparse":
                    message.coo_sparse = tensorflow.TensorInfo.CooSparse.decodeText(reader);
                    break;
                case "composite_tensor":
                    message.composite_tensor = tensorflow.TensorInfo.CompositeTensor.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "tensor_shape":
                    message.tensor_shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorInfo.prototype.dtype = 0;
tensorflow.TensorInfo.prototype.tensor_shape = null;

tensorflow.TensorInfo.CooSparse = class CooSparse {

    static decode(reader, length) {
        const message = new tensorflow.TensorInfo.CooSparse();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values_tensor_name = reader.string();
                    break;
                case 2:
                    message.indices_tensor_name = reader.string();
                    break;
                case 3:
                    message.dense_shape_tensor_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorInfo.CooSparse();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values_tensor_name":
                    message.values_tensor_name = reader.string();
                    break;
                case "indices_tensor_name":
                    message.indices_tensor_name = reader.string();
                    break;
                case "dense_shape_tensor_name":
                    message.dense_shape_tensor_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorInfo.CooSparse.prototype.values_tensor_name = "";
tensorflow.TensorInfo.CooSparse.prototype.indices_tensor_name = "";
tensorflow.TensorInfo.CooSparse.prototype.dense_shape_tensor_name = "";

tensorflow.TensorInfo.CompositeTensor = class CompositeTensor {

    constructor() {
        this.components = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TensorInfo.CompositeTensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_spec = tensorflow.TypeSpecProto.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.components.push(tensorflow.TensorInfo.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorInfo.CompositeTensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_spec":
                    message.type_spec = tensorflow.TypeSpecProto.decodeText(reader);
                    break;
                case "components":
                    message.components.push(tensorflow.TensorInfo.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorInfo.CompositeTensor.prototype.type_spec = null;

tensorflow.SignatureDef = class SignatureDef {

    constructor() {
        this.inputs = {};
        this.outputs = {};
        this.defaults = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.SignatureDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.inputs, () => reader.string(), () => tensorflow.TensorInfo.decode(reader, reader.uint32()));
                    break;
                case 2:
                    reader.entry(message.outputs, () => reader.string(), () => tensorflow.TensorInfo.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.method_name = reader.string();
                    break;
                case 4:
                    reader.entry(message.defaults, () => reader.string(), () => tensorflow.TensorProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SignatureDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputs":
                    reader.entry(message.inputs, () => reader.string(), () => tensorflow.TensorInfo.decodeText(reader));
                    break;
                case "outputs":
                    reader.entry(message.outputs, () => reader.string(), () => tensorflow.TensorInfo.decodeText(reader));
                    break;
                case "method_name":
                    message.method_name = reader.string();
                    break;
                case "defaults":
                    reader.entry(message.defaults, () => reader.string(), () => tensorflow.TensorProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SignatureDef.prototype.method_name = "";

tensorflow.AssetFileDef = class AssetFileDef {

    static decode(reader, length) {
        const message = new tensorflow.AssetFileDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_info = tensorflow.TensorInfo.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.filename = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AssetFileDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_info":
                    message.tensor_info = tensorflow.TensorInfo.decodeText(reader);
                    break;
                case "filename":
                    message.filename = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AssetFileDef.prototype.tensor_info = null;
tensorflow.AssetFileDef.prototype.filename = "";

tensorflow.GraphDef = class GraphDef {

    constructor() {
        this.node = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.GraphDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push(tensorflow.NodeDef.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.versions = tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.version = reader.int32();
                    break;
                case 2:
                    message.library = tensorflow.FunctionDefLibrary.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.debug_info = tensorflow.GraphDebugInfo.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GraphDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node":
                    message.node.push(tensorflow.NodeDef.decodeText(reader));
                    break;
                case "versions":
                    message.versions = tensorflow.VersionDef.decodeText(reader);
                    break;
                case "version":
                    message.version = reader.int32();
                    break;
                case "library":
                    message.library = tensorflow.FunctionDefLibrary.decodeText(reader);
                    break;
                case "debug_info":
                    message.debug_info = tensorflow.GraphDebugInfo.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GraphDef.prototype.versions = null;
tensorflow.GraphDef.prototype.version = 0;
tensorflow.GraphDef.prototype.library = null;
tensorflow.GraphDef.prototype.debug_info = null;

tensorflow.FunctionDefLibrary = class FunctionDefLibrary {

    constructor() {
        this.function = [];
        this.gradient = [];
        this.registered_gradients = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.FunctionDefLibrary();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.function.push(tensorflow.FunctionDef.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.gradient.push(tensorflow.GradientDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.registered_gradients.push(tensorflow.RegisteredGradient.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.FunctionDefLibrary();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "function":
                    message.function.push(tensorflow.FunctionDef.decodeText(reader));
                    break;
                case "gradient":
                    message.gradient.push(tensorflow.GradientDef.decodeText(reader));
                    break;
                case "registered_gradients":
                    message.registered_gradients.push(tensorflow.RegisteredGradient.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FunctionDef = class FunctionDef {

    constructor() {
        this.attr = {};
        this.arg_attr = {};
        this.resource_arg_unique_id = {};
        this.node_def = [];
        this.ret = {};
        this.control_ret = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.FunctionDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.signature = tensorflow.OpDef.decode(reader, reader.uint32());
                    break;
                case 5:
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 7:
                    reader.entry(message.arg_attr, () => reader.uint32(), () => tensorflow.FunctionDef.ArgAttrs.decode(reader, reader.uint32()));
                    break;
                case 8:
                    reader.entry(message.resource_arg_unique_id, () => reader.uint32(), () => reader.uint32());
                    break;
                case 3:
                    message.node_def.push(tensorflow.NodeDef.decode(reader, reader.uint32()));
                    break;
                case 4:
                    reader.entry(message.ret, () => reader.string(), () => reader.string());
                    break;
                case 6:
                    reader.entry(message.control_ret, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.FunctionDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "signature":
                    message.signature = tensorflow.OpDef.decodeText(reader);
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decodeText(reader));
                    break;
                case "arg_attr":
                    reader.entry(message.arg_attr, () => reader.uint32(), () => tensorflow.FunctionDef.ArgAttrs.decodeText(reader));
                    break;
                case "resource_arg_unique_id":
                    reader.entry(message.resource_arg_unique_id, () => reader.uint32(), () => reader.uint32());
                    break;
                case "node_def":
                    message.node_def.push(tensorflow.NodeDef.decodeText(reader));
                    break;
                case "ret":
                    reader.entry(message.ret, () => reader.string(), () => reader.string());
                    break;
                case "control_ret":
                    reader.entry(message.control_ret, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FunctionDef.prototype.signature = null;

tensorflow.FunctionDef.ArgAttrs = class ArgAttrs {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.FunctionDef.ArgAttrs();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.FunctionDef.ArgAttrs();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GradientDef = class GradientDef {

    static decode(reader, length) {
        const message = new tensorflow.GradientDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.function_name = reader.string();
                    break;
                case 2:
                    message.gradient_func = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GradientDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "function_name":
                    message.function_name = reader.string();
                    break;
                case "gradient_func":
                    message.gradient_func = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GradientDef.prototype.function_name = "";
tensorflow.GradientDef.prototype.gradient_func = "";

tensorflow.RegisteredGradient = class RegisteredGradient {

    static decode(reader, length) {
        const message = new tensorflow.RegisteredGradient();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.gradient_func = reader.string();
                    break;
                case 2:
                    message.registered_op_type = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RegisteredGradient();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "gradient_func":
                    message.gradient_func = reader.string();
                    break;
                case "registered_op_type":
                    message.registered_op_type = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RegisteredGradient.prototype.gradient_func = "";
tensorflow.RegisteredGradient.prototype.registered_op_type = "";

tensorflow.AttrValue = class AttrValue {

    get value() {
        tensorflow.AttrValue.valueSet = tensorflow.AttrValue.valueSet || new Set(["s", "i", "f", "b", "type", "shape", "tensor", "list", "func", "placeholder"]);
        return Object.keys(this).find((key) => tensorflow.AttrValue.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.AttrValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.s = reader.bytes();
                    break;
                case 3:
                    message.i = reader.int64();
                    break;
                case 4:
                    message.f = reader.float();
                    break;
                case 5:
                    message.b = reader.bool();
                    break;
                case 6:
                    message.type = reader.int32();
                    break;
                case 7:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.tensor = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                case 1:
                    message.list = tensorflow.AttrValue.ListValue.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.func = tensorflow.NameAttrList.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.placeholder = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AttrValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    message.s = reader.bytes();
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "f":
                    message.f = reader.float();
                    break;
                case "b":
                    message.b = reader.bool();
                    break;
                case "type":
                    message.type = reader.enum(tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "tensor":
                    message.tensor = tensorflow.TensorProto.decodeText(reader);
                    break;
                case "list":
                    message.list = tensorflow.AttrValue.ListValue.decodeText(reader);
                    break;
                case "func":
                    message.func = tensorflow.NameAttrList.decodeText(reader);
                    break;
                case "placeholder":
                    message.placeholder = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AttrValue.ListValue = class ListValue {

    constructor() {
        this.s = [];
        this.i = [];
        this.f = [];
        this.b = [];
        this.type = [];
        this.shape = [];
        this.tensor = [];
        this.func = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.AttrValue.ListValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.s.push(reader.bytes());
                    break;
                case 3:
                    message.i = reader.array(message.i, () => reader.int64(), tag);
                    break;
                case 4:
                    message.f = reader.floats(message.f, tag);
                    break;
                case 5:
                    message.b = reader.array(message.b, () => reader.bool(), tag);
                    break;
                case 6:
                    message.type = reader.array(message.type, () => reader.int32(), tag);
                    break;
                case 7:
                    message.shape.push(tensorflow.TensorShapeProto.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.tensor.push(tensorflow.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.func.push(tensorflow.NameAttrList.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AttrValue.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    reader.array(message.s, () => reader.bytes());
                    break;
                case "i":
                    reader.array(message.i, () => reader.int64());
                    break;
                case "f":
                    reader.array(message.f, () => reader.float());
                    break;
                case "b":
                    reader.array(message.b, () => reader.bool());
                    break;
                case "type":
                    reader.array(message.type, () => reader.enum(tensorflow.DataType));
                    break;
                case "shape":
                    message.shape.push(tensorflow.TensorShapeProto.decodeText(reader));
                    break;
                case "tensor":
                    message.tensor.push(tensorflow.TensorProto.decodeText(reader));
                    break;
                case "func":
                    message.func.push(tensorflow.NameAttrList.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NameAttrList = class NameAttrList {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.NameAttrList();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.NameAttrList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NameAttrList.prototype.name = "";

tensorflow.TensorProto = class TensorProto {

    constructor() {
        this.half_val = [];
        this.float_val = [];
        this.double_val = [];
        this.int_val = [];
        this.string_val = [];
        this.scomplex_val = [];
        this.int64_val = [];
        this.bool_val = [];
        this.dcomplex_val = [];
        this.resource_handle_val = [];
        this.variant_val = [];
        this.uint32_val = [];
        this.uint64_val = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TensorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.tensor_shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.version_number = reader.int32();
                    break;
                case 4:
                    message.tensor_content = reader.bytes();
                    break;
                case 13:
                    message.half_val = reader.array(message.half_val, () => reader.int32(), tag);
                    break;
                case 5:
                    message.float_val = reader.floats(message.float_val, tag);
                    break;
                case 6:
                    message.double_val = reader.doubles(message.double_val, tag);
                    break;
                case 7:
                    message.int_val = reader.array(message.int_val, () => reader.int32(), tag);
                    break;
                case 8:
                    message.string_val.push(reader.bytes());
                    break;
                case 9:
                    message.scomplex_val = reader.floats(message.scomplex_val, tag);
                    break;
                case 10:
                    message.int64_val = reader.array(message.int64_val, () => reader.int64(), tag);
                    break;
                case 11:
                    message.bool_val = reader.array(message.bool_val, () => reader.bool(), tag);
                    break;
                case 12:
                    message.dcomplex_val = reader.doubles(message.dcomplex_val, tag);
                    break;
                case 14:
                    message.resource_handle_val.push(tensorflow.ResourceHandleProto.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.variant_val.push(tensorflow.VariantTensorDataProto.decode(reader, reader.uint32()));
                    break;
                case 16:
                    message.uint32_val = reader.array(message.uint32_val, () => reader.uint32(), tag);
                    break;
                case 17:
                    message.uint64_val = reader.array(message.uint64_val, () => reader.uint64(), tag);
                    break;
                case 18:
                    message.float8_val = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "tensor_shape":
                    message.tensor_shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "version_number":
                    message.version_number = reader.int32();
                    break;
                case "tensor_content":
                    message.tensor_content = reader.bytes();
                    break;
                case "half_val":
                    reader.array(message.half_val, () => reader.int32());
                    break;
                case "float_val":
                    reader.array(message.float_val, () => reader.float());
                    break;
                case "double_val":
                    reader.array(message.double_val, () => reader.double());
                    break;
                case "int_val":
                    reader.array(message.int_val, () => reader.int32());
                    break;
                case "string_val":
                    reader.array(message.string_val, () => reader.bytes());
                    break;
                case "scomplex_val":
                    reader.array(message.scomplex_val, () => reader.float());
                    break;
                case "int64_val":
                    reader.array(message.int64_val, () => reader.int64());
                    break;
                case "bool_val":
                    reader.array(message.bool_val, () => reader.bool());
                    break;
                case "dcomplex_val":
                    reader.array(message.dcomplex_val, () => reader.double());
                    break;
                case "resource_handle_val":
                    message.resource_handle_val.push(tensorflow.ResourceHandleProto.decodeText(reader));
                    break;
                case "variant_val":
                    message.variant_val.push(tensorflow.VariantTensorDataProto.decodeText(reader));
                    break;
                case "uint32_val":
                    reader.array(message.uint32_val, () => reader.uint32());
                    break;
                case "uint64_val":
                    reader.array(message.uint64_val, () => reader.uint64());
                    break;
                case "float8_val":
                    message.float8_val = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorProto.prototype.dtype = 0;
tensorflow.TensorProto.prototype.tensor_shape = null;
tensorflow.TensorProto.prototype.version_number = 0;
tensorflow.TensorProto.prototype.tensor_content = new Uint8Array([]);
tensorflow.TensorProto.prototype.float8_val = new Uint8Array([]);

tensorflow.VariantTensorDataProto = class VariantTensorDataProto {

    constructor() {
        this.tensors = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.VariantTensorDataProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_name = reader.string();
                    break;
                case 2:
                    message.metadata = reader.bytes();
                    break;
                case 3:
                    message.tensors.push(tensorflow.TensorProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.VariantTensorDataProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_name":
                    message.type_name = reader.string();
                    break;
                case "metadata":
                    message.metadata = reader.bytes();
                    break;
                case "tensors":
                    message.tensors.push(tensorflow.TensorProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.VariantTensorDataProto.prototype.type_name = "";
tensorflow.VariantTensorDataProto.prototype.metadata = new Uint8Array([]);

tensorflow.ResourceHandleProto = class ResourceHandleProto {

    constructor() {
        this.dtypes_and_shapes = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.ResourceHandleProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.device = reader.string();
                    break;
                case 2:
                    message.container = reader.string();
                    break;
                case 3:
                    message.name = reader.string();
                    break;
                case 4:
                    message.hash_code = reader.uint64();
                    break;
                case 5:
                    message.maybe_type_name = reader.string();
                    break;
                case 6:
                    message.dtypes_and_shapes.push(tensorflow.ResourceHandleProto.DtypeAndShape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ResourceHandleProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "device":
                    message.device = reader.string();
                    break;
                case "container":
                    message.container = reader.string();
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "hash_code":
                    message.hash_code = reader.uint64();
                    break;
                case "maybe_type_name":
                    message.maybe_type_name = reader.string();
                    break;
                case "dtypes_and_shapes":
                    message.dtypes_and_shapes.push(tensorflow.ResourceHandleProto.DtypeAndShape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.ResourceHandleProto.prototype.device = "";
tensorflow.ResourceHandleProto.prototype.container = "";
tensorflow.ResourceHandleProto.prototype.name = "";
tensorflow.ResourceHandleProto.prototype.hash_code = 0n;
tensorflow.ResourceHandleProto.prototype.maybe_type_name = "";

tensorflow.ResourceHandleProto.DtypeAndShape = class DtypeAndShape {

    static decode(reader, length) {
        const message = new tensorflow.ResourceHandleProto.DtypeAndShape();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ResourceHandleProto.DtypeAndShape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.ResourceHandleProto.DtypeAndShape.prototype.dtype = 0;
tensorflow.ResourceHandleProto.DtypeAndShape.prototype.shape = null;

tensorflow.TensorShapeProto = class TensorShapeProto {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TensorShapeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.dim.push(tensorflow.TensorShapeProto.Dim.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.unknown_rank = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorShapeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim.push(tensorflow.TensorShapeProto.Dim.decodeText(reader));
                    break;
                case "unknown_rank":
                    message.unknown_rank = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorShapeProto.prototype.unknown_rank = false;

tensorflow.TensorShapeProto.Dim = class Dim {

    static decode(reader, length) {
        const message = new tensorflow.TensorShapeProto.Dim();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.size = reader.int64();
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
        const message = new tensorflow.TensorShapeProto.Dim();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    message.size = reader.int64();
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
};

tensorflow.TensorShapeProto.Dim.prototype.size = 0n;
tensorflow.TensorShapeProto.Dim.prototype.name = "";

tensorflow.DataType = {
    "DT_INVALID": 0,
    "DT_FLOAT": 1,
    "DT_DOUBLE": 2,
    "DT_INT32": 3,
    "DT_UINT8": 4,
    "DT_INT16": 5,
    "DT_INT8": 6,
    "DT_STRING": 7,
    "DT_COMPLEX64": 8,
    "DT_INT64": 9,
    "DT_BOOL": 10,
    "DT_QINT8": 11,
    "DT_QUINT8": 12,
    "DT_QINT32": 13,
    "DT_BFLOAT16": 14,
    "DT_QINT16": 15,
    "DT_QUINT16": 16,
    "DT_UINT16": 17,
    "DT_COMPLEX128": 18,
    "DT_HALF": 19,
    "DT_RESOURCE": 20,
    "DT_VARIANT": 21,
    "DT_UINT32": 22,
    "DT_UINT64": 23,
    "DT_FLOAT8_E5M2": 24,
    "DT_FLOAT8_E4M3FN": 25,
    "DT_INT4": 29,
    "DT_UINT4": 30,
    "DT_FLOAT_REF": 101,
    "DT_DOUBLE_REF": 102,
    "DT_INT32_REF": 103,
    "DT_UINT8_REF": 104,
    "DT_INT16_REF": 105,
    "DT_INT8_REF": 106,
    "DT_STRING_REF": 107,
    "DT_COMPLEX64_REF": 108,
    "DT_INT64_REF": 109,
    "DT_BOOL_REF": 110,
    "DT_QINT8_REF": 111,
    "DT_QUINT8_REF": 112,
    "DT_QINT32_REF": 113,
    "DT_BFLOAT16_REF": 114,
    "DT_QINT16_REF": 115,
    "DT_QUINT16_REF": 116,
    "DT_UINT16_REF": 117,
    "DT_COMPLEX128_REF": 118,
    "DT_HALF_REF": 119,
    "DT_RESOURCE_REF": 120,
    "DT_VARIANT_REF": 121,
    "DT_UINT32_REF": 122,
    "DT_UINT64_REF": 123,
    "DT_FLOAT8_E5M2_REF": 124,
    "DT_FLOAT8_E4M3FN_REF": 125,
    "DT_INT4_REF": 129,
    "DT_UINT4_REF": 130
};

tensorflow.SerializedDType = class SerializedDType {

    static decode(reader, length) {
        const message = new tensorflow.SerializedDType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.datatype = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SerializedDType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "datatype":
                    message.datatype = reader.enum(tensorflow.DataType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SerializedDType.prototype.datatype = 0;

tensorflow.NodeDef = class NodeDef {

    constructor() {
        this.input = [];
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.NodeDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.op = reader.string();
                    break;
                case 3:
                    message.input.push(reader.string());
                    break;
                case 4:
                    message.device = reader.string();
                    break;
                case 5:
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.experimental_debug_info = tensorflow.NodeDef.ExperimentalDebugInfo.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.experimental_type = tensorflow.FullTypeDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.NodeDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "op":
                    message.op = reader.string();
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "device":
                    message.device = reader.string();
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => tensorflow.AttrValue.decodeText(reader));
                    break;
                case "experimental_debug_info":
                    message.experimental_debug_info = tensorflow.NodeDef.ExperimentalDebugInfo.decodeText(reader);
                    break;
                case "experimental_type":
                    message.experimental_type = tensorflow.FullTypeDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NodeDef.prototype.name = "";
tensorflow.NodeDef.prototype.op = "";
tensorflow.NodeDef.prototype.device = "";
tensorflow.NodeDef.prototype.experimental_debug_info = null;
tensorflow.NodeDef.prototype.experimental_type = null;

tensorflow.NodeDef.ExperimentalDebugInfo = class ExperimentalDebugInfo {

    constructor() {
        this.original_node_names = [];
        this.original_func_names = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.NodeDef.ExperimentalDebugInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.original_node_names.push(reader.string());
                    break;
                case 2:
                    message.original_func_names.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.NodeDef.ExperimentalDebugInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "original_node_names":
                    reader.array(message.original_node_names, () => reader.string());
                    break;
                case "original_func_names":
                    reader.array(message.original_func_names, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FullTypeId = {
    "TFT_UNSET": 0,
    "TFT_VAR": 1,
    "TFT_ANY": 2,
    "TFT_PRODUCT": 3,
    "TFT_NAMED": 4,
    "TFT_FOR_EACH": 20,
    "TFT_CALLABLE": 100,
    "TFT_TENSOR": 1000,
    "TFT_ARRAY": 1001,
    "TFT_OPTIONAL": 1002,
    "TFT_LITERAL": 1003,
    "TFT_ENCODED": 1004,
    "TFT_SHAPE_TENSOR": 1005,
    "TFT_BOOL": 200,
    "TFT_UINT8": 201,
    "TFT_UINT16": 202,
    "TFT_UINT32": 203,
    "TFT_UINT64": 204,
    "TFT_INT8": 205,
    "TFT_INT16": 206,
    "TFT_INT32": 207,
    "TFT_INT64": 208,
    "TFT_HALF": 209,
    "TFT_FLOAT": 210,
    "TFT_DOUBLE": 211,
    "TFT_BFLOAT16": 215,
    "TFT_COMPLEX64": 212,
    "TFT_COMPLEX128": 213,
    "TFT_STRING": 214,
    "TFT_DATASET": 10102,
    "TFT_RAGGED": 10103,
    "TFT_ITERATOR": 10104,
    "TFT_MUTEX_LOCK": 10202,
    "TFT_LEGACY_VARIANT": 10203
};

tensorflow.FullTypeDef = class FullTypeDef {

    constructor() {
        this.args = [];
    }

    get attr() {
        tensorflow.FullTypeDef.attrSet = tensorflow.FullTypeDef.attrSet || new Set(["s", "i"]);
        return Object.keys(this).find((key) => tensorflow.FullTypeDef.attrSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.FullTypeDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_id = reader.int32();
                    break;
                case 2:
                    message.args.push(tensorflow.FullTypeDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.s = reader.string();
                    break;
                case 4:
                    message.i = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.FullTypeDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_id":
                    message.type_id = reader.enum(tensorflow.FullTypeId);
                    break;
                case "args":
                    message.args.push(tensorflow.FullTypeDef.decodeText(reader));
                    break;
                case "s":
                    message.s = reader.string();
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FullTypeDef.prototype.type_id = 0;

tensorflow.OpDef = class OpDef {

    constructor() {
        this.input_arg = [];
        this.output_arg = [];
        this.control_output = [];
        this.attr = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.OpDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.input_arg.push(tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.output_arg.push(tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                    break;
                case 20:
                    message.control_output.push(reader.string());
                    break;
                case 4:
                    message.attr.push(tensorflow.OpDef.AttrDef.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.deprecation = tensorflow.OpDeprecation.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.summary = reader.string();
                    break;
                case 6:
                    message.description = reader.string();
                    break;
                case 18:
                    message.is_commutative = reader.bool();
                    break;
                case 16:
                    message.is_aggregate = reader.bool();
                    break;
                case 17:
                    message.is_stateful = reader.bool();
                    break;
                case 19:
                    message.allows_uninitialized_input = reader.bool();
                    break;
                case 21:
                    message.is_distributed_communication = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.OpDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input_arg":
                    message.input_arg.push(tensorflow.OpDef.ArgDef.decodeText(reader));
                    break;
                case "output_arg":
                    message.output_arg.push(tensorflow.OpDef.ArgDef.decodeText(reader));
                    break;
                case "control_output":
                    reader.array(message.control_output, () => reader.string());
                    break;
                case "attr":
                    message.attr.push(tensorflow.OpDef.AttrDef.decodeText(reader));
                    break;
                case "deprecation":
                    message.deprecation = tensorflow.OpDeprecation.decodeText(reader);
                    break;
                case "summary":
                    message.summary = reader.string();
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                case "is_commutative":
                    message.is_commutative = reader.bool();
                    break;
                case "is_aggregate":
                    message.is_aggregate = reader.bool();
                    break;
                case "is_stateful":
                    message.is_stateful = reader.bool();
                    break;
                case "allows_uninitialized_input":
                    message.allows_uninitialized_input = reader.bool();
                    break;
                case "is_distributed_communication":
                    message.is_distributed_communication = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.OpDef.prototype.name = "";
tensorflow.OpDef.prototype.deprecation = null;
tensorflow.OpDef.prototype.summary = "";
tensorflow.OpDef.prototype.description = "";
tensorflow.OpDef.prototype.is_commutative = false;
tensorflow.OpDef.prototype.is_aggregate = false;
tensorflow.OpDef.prototype.is_stateful = false;
tensorflow.OpDef.prototype.allows_uninitialized_input = false;
tensorflow.OpDef.prototype.is_distributed_communication = false;

tensorflow.OpDef.ArgDef = class ArgDef {

    constructor() {
        this.handle_data = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.OpDef.ArgDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.description = reader.string();
                    break;
                case 3:
                    message.type = reader.int32();
                    break;
                case 4:
                    message.type_attr = reader.string();
                    break;
                case 5:
                    message.number_attr = reader.string();
                    break;
                case 6:
                    message.type_list_attr = reader.string();
                    break;
                case 7:
                    message.handle_data.push(tensorflow.ResourceHandleProto.DtypeAndShape.decode(reader, reader.uint32()));
                    break;
                case 16:
                    message.is_ref = reader.bool();
                    break;
                case 17:
                    message.experimental_full_type = tensorflow.FullTypeDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.OpDef.ArgDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                case "type":
                    message.type = reader.enum(tensorflow.DataType);
                    break;
                case "type_attr":
                    message.type_attr = reader.string();
                    break;
                case "number_attr":
                    message.number_attr = reader.string();
                    break;
                case "type_list_attr":
                    message.type_list_attr = reader.string();
                    break;
                case "handle_data":
                    message.handle_data.push(tensorflow.ResourceHandleProto.DtypeAndShape.decodeText(reader));
                    break;
                case "is_ref":
                    message.is_ref = reader.bool();
                    break;
                case "experimental_full_type":
                    message.experimental_full_type = tensorflow.FullTypeDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.OpDef.ArgDef.prototype.name = "";
tensorflow.OpDef.ArgDef.prototype.description = "";
tensorflow.OpDef.ArgDef.prototype.type = 0;
tensorflow.OpDef.ArgDef.prototype.type_attr = "";
tensorflow.OpDef.ArgDef.prototype.number_attr = "";
tensorflow.OpDef.ArgDef.prototype.type_list_attr = "";
tensorflow.OpDef.ArgDef.prototype.is_ref = false;
tensorflow.OpDef.ArgDef.prototype.experimental_full_type = null;

tensorflow.OpDef.AttrDef = class AttrDef {

    static decode(reader, length) {
        const message = new tensorflow.OpDef.AttrDef();
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
                    message.default_value = tensorflow.AttrValue.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.description = reader.string();
                    break;
                case 5:
                    message.has_minimum = reader.bool();
                    break;
                case 6:
                    message.minimum = reader.int64();
                    break;
                case 7:
                    message.allowed_values = tensorflow.AttrValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.OpDef.AttrDef();
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
                case "default_value":
                    message.default_value = tensorflow.AttrValue.decodeText(reader);
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                case "has_minimum":
                    message.has_minimum = reader.bool();
                    break;
                case "minimum":
                    message.minimum = reader.int64();
                    break;
                case "allowed_values":
                    message.allowed_values = tensorflow.AttrValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.OpDef.AttrDef.prototype.name = "";
tensorflow.OpDef.AttrDef.prototype.type = "";
tensorflow.OpDef.AttrDef.prototype.default_value = null;
tensorflow.OpDef.AttrDef.prototype.description = "";
tensorflow.OpDef.AttrDef.prototype.has_minimum = false;
tensorflow.OpDef.AttrDef.prototype.minimum = 0n;
tensorflow.OpDef.AttrDef.prototype.allowed_values = null;

tensorflow.OpDeprecation = class OpDeprecation {

    static decode(reader, length) {
        const message = new tensorflow.OpDeprecation();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int32();
                    break;
                case 2:
                    message.explanation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.OpDeprecation();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.int32();
                    break;
                case "explanation":
                    message.explanation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.OpDeprecation.prototype.version = 0;
tensorflow.OpDeprecation.prototype.explanation = "";

tensorflow.OpList = class OpList {

    constructor() {
        this.op = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.OpList();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op.push(tensorflow.OpDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.OpList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op":
                    message.op.push(tensorflow.OpDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GraphDebugInfo = class GraphDebugInfo {

    constructor() {
        this.files = [];
        this.frames_by_id = {};
        this.traces_by_id = {};
        this.traces = {};
        this.name_to_trace_id = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.GraphDebugInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.files.push(reader.string());
                    break;
                case 4:
                    reader.entry(message.frames_by_id, () => reader.fixed64(), () => tensorflow.GraphDebugInfo.FileLineCol.decode(reader, reader.uint32()));
                    break;
                case 6:
                    reader.entry(message.traces_by_id, () => reader.fixed64(), () => tensorflow.GraphDebugInfo.StackTrace.decode(reader, reader.uint32()));
                    break;
                case 2:
                    reader.entry(message.traces, () => reader.string(), () => tensorflow.GraphDebugInfo.StackTrace.decode(reader, reader.uint32()));
                    break;
                case 5:
                    reader.entry(message.name_to_trace_id, () => reader.string(), () => reader.fixed64());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GraphDebugInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "files":
                    reader.array(message.files, () => reader.string());
                    break;
                case "frames_by_id":
                    reader.entry(message.frames_by_id, () => reader.fixed64(), () => tensorflow.GraphDebugInfo.FileLineCol.decodeText(reader));
                    break;
                case "traces_by_id":
                    reader.entry(message.traces_by_id, () => reader.fixed64(), () => tensorflow.GraphDebugInfo.StackTrace.decodeText(reader));
                    break;
                case "traces":
                    reader.entry(message.traces, () => reader.string(), () => tensorflow.GraphDebugInfo.StackTrace.decodeText(reader));
                    break;
                case "name_to_trace_id":
                    reader.entry(message.name_to_trace_id, () => reader.string(), () => reader.fixed64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GraphDebugInfo.FileLineCol = class FileLineCol {

    static decode(reader, length) {
        const message = new tensorflow.GraphDebugInfo.FileLineCol();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.file_index = reader.int32();
                    break;
                case 2:
                    message.line = reader.int32();
                    break;
                case 3:
                    message.col = reader.int32();
                    break;
                case 4:
                    message.func = reader.string();
                    break;
                case 5:
                    message.code = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GraphDebugInfo.FileLineCol();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "file_index":
                    message.file_index = reader.int32();
                    break;
                case "line":
                    message.line = reader.int32();
                    break;
                case "col":
                    message.col = reader.int32();
                    break;
                case "func":
                    message.func = reader.string();
                    break;
                case "code":
                    message.code = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GraphDebugInfo.FileLineCol.prototype.file_index = 0;
tensorflow.GraphDebugInfo.FileLineCol.prototype.line = 0;
tensorflow.GraphDebugInfo.FileLineCol.prototype.col = 0;
tensorflow.GraphDebugInfo.FileLineCol.prototype.func = "";
tensorflow.GraphDebugInfo.FileLineCol.prototype.code = "";

tensorflow.GraphDebugInfo.StackTrace = class StackTrace {

    constructor() {
        this.file_line_cols = [];
        this.frame_id = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.GraphDebugInfo.StackTrace();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.file_line_cols.push(tensorflow.GraphDebugInfo.FileLineCol.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.frame_id = reader.array(message.frame_id, () => reader.fixed64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GraphDebugInfo.StackTrace();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "file_line_cols":
                    message.file_line_cols.push(tensorflow.GraphDebugInfo.FileLineCol.decodeText(reader));
                    break;
                case "frame_id":
                    reader.array(message.frame_id, () => reader.fixed64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.VersionDef = class VersionDef {

    constructor() {
        this.bad_consumers = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.VersionDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.producer = reader.int32();
                    break;
                case 2:
                    message.min_consumer = reader.int32();
                    break;
                case 3:
                    message.bad_consumers = reader.array(message.bad_consumers, () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.VersionDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "producer":
                    message.producer = reader.int32();
                    break;
                case "min_consumer":
                    message.min_consumer = reader.int32();
                    break;
                case "bad_consumers":
                    reader.array(message.bad_consumers, () => reader.int32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.VersionDef.prototype.producer = 0;
tensorflow.VersionDef.prototype.min_consumer = 0;

tensorflow.SavedObjectGraph = class SavedObjectGraph {

    constructor() {
        this.nodes = [];
        this.concrete_functions = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedObjectGraph();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push(tensorflow.SavedObject.decode(reader, reader.uint32()));
                    break;
                case 2:
                    reader.entry(message.concrete_functions, () => reader.string(), () => tensorflow.SavedConcreteFunction.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedObjectGraph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push(tensorflow.SavedObject.decodeText(reader));
                    break;
                case "concrete_functions":
                    reader.entry(message.concrete_functions, () => reader.string(), () => tensorflow.SavedConcreteFunction.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedObject = class SavedObject {

    constructor() {
        this.children = [];
        this.dependencies = [];
        this.slot_variables = [];
        this.saveable_objects = {};
    }

    get kind() {
        tensorflow.SavedObject.kindSet = tensorflow.SavedObject.kindSet || new Set(["user_object", "asset", "function", "variable", "bare_concrete_function", "constant", "resource", "captured_tensor"]);
        return Object.keys(this).find((key) => tensorflow.SavedObject.kindSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedObject();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.children.push(tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.dependencies.push(tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.slot_variables.push(tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.user_object = tensorflow.SavedUserObject.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.asset = tensorflow.SavedAsset.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.function = tensorflow.SavedFunction.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.variable = tensorflow.SavedVariable.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.bare_concrete_function = tensorflow.SavedBareConcreteFunction.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.constant = tensorflow.SavedConstant.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.resource = tensorflow.SavedResource.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.captured_tensor = tensorflow.CapturedTensor.decode(reader, reader.uint32());
                    break;
                case 11:
                    reader.entry(message.saveable_objects, () => reader.string(), () => tensorflow.SaveableObject.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.registered_name = reader.string();
                    break;
                case 14:
                    message.serialized_user_proto = google.protobuf.Any.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.registered_saver = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "children":
                    message.children.push(tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader));
                    break;
                case "dependencies":
                    message.dependencies.push(tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader));
                    break;
                case "slot_variables":
                    message.slot_variables.push(tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeText(reader));
                    break;
                case "user_object":
                    message.user_object = tensorflow.SavedUserObject.decodeText(reader);
                    break;
                case "asset":
                    message.asset = tensorflow.SavedAsset.decodeText(reader);
                    break;
                case "function":
                    message.function = tensorflow.SavedFunction.decodeText(reader);
                    break;
                case "variable":
                    message.variable = tensorflow.SavedVariable.decodeText(reader);
                    break;
                case "bare_concrete_function":
                    message.bare_concrete_function = tensorflow.SavedBareConcreteFunction.decodeText(reader);
                    break;
                case "constant":
                    message.constant = tensorflow.SavedConstant.decodeText(reader);
                    break;
                case "resource":
                    message.resource = tensorflow.SavedResource.decodeText(reader);
                    break;
                case "captured_tensor":
                    message.captured_tensor = tensorflow.CapturedTensor.decodeText(reader);
                    break;
                case "saveable_objects":
                    reader.entry(message.saveable_objects, () => reader.string(), () => tensorflow.SaveableObject.decodeText(reader));
                    break;
                case "registered_name":
                    message.registered_name = reader.string();
                    break;
                case "serialized_user_proto":
                    message.serialized_user_proto = google.protobuf.Any.decodeText(reader);
                    break;
                case "registered_saver":
                    message.registered_saver = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedObject.prototype.registered_name = "";
tensorflow.SavedObject.prototype.serialized_user_proto = null;
tensorflow.SavedObject.prototype.registered_saver = "";

tensorflow.SavedUserObject = class SavedUserObject {

    static decode(reader, length) {
        const message = new tensorflow.SavedUserObject();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.identifier = reader.string();
                    break;
                case 2:
                    message.version = tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.metadata = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedUserObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "identifier":
                    message.identifier = reader.string();
                    break;
                case "version":
                    message.version = tensorflow.VersionDef.decodeText(reader);
                    break;
                case "metadata":
                    message.metadata = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedUserObject.prototype.identifier = "";
tensorflow.SavedUserObject.prototype.version = null;
tensorflow.SavedUserObject.prototype.metadata = "";

tensorflow.SavedAsset = class SavedAsset {

    static decode(reader, length) {
        const message = new tensorflow.SavedAsset();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.asset_file_def_index = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedAsset();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "asset_file_def_index":
                    message.asset_file_def_index = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedAsset.prototype.asset_file_def_index = 0;

tensorflow.SavedFunction = class SavedFunction {

    constructor() {
        this.concrete_functions = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedFunction();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.concrete_functions.push(reader.string());
                    break;
                case 2:
                    message.function_spec = tensorflow.FunctionSpec.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedFunction();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "concrete_functions":
                    reader.array(message.concrete_functions, () => reader.string());
                    break;
                case "function_spec":
                    message.function_spec = tensorflow.FunctionSpec.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedFunction.prototype.function_spec = null;

tensorflow.CapturedTensor = class CapturedTensor {

    static decode(reader, length) {
        const message = new tensorflow.CapturedTensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.concrete_function = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CapturedTensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "concrete_function":
                    message.concrete_function = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CapturedTensor.prototype.name = "";
tensorflow.CapturedTensor.prototype.concrete_function = "";

tensorflow.SavedConcreteFunction = class SavedConcreteFunction {

    constructor() {
        this.bound_inputs = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedConcreteFunction();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.bound_inputs = reader.array(message.bound_inputs, () => reader.int32(), tag);
                    break;
                case 3:
                    message.canonicalized_input_signature = tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.output_signature = tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedConcreteFunction();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "bound_inputs":
                    reader.array(message.bound_inputs, () => reader.int32());
                    break;
                case "canonicalized_input_signature":
                    message.canonicalized_input_signature = tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "output_signature":
                    message.output_signature = tensorflow.StructuredValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedConcreteFunction.prototype.canonicalized_input_signature = null;
tensorflow.SavedConcreteFunction.prototype.output_signature = null;

tensorflow.SavedBareConcreteFunction = class SavedBareConcreteFunction {

    constructor() {
        this.argument_keywords = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedBareConcreteFunction();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.concrete_function_name = reader.string();
                    break;
                case 2:
                    message.argument_keywords.push(reader.string());
                    break;
                case 3:
                    message.allowed_positional_arguments = reader.int64();
                    break;
                case 4:
                    message.function_spec = tensorflow.FunctionSpec.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedBareConcreteFunction();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "concrete_function_name":
                    message.concrete_function_name = reader.string();
                    break;
                case "argument_keywords":
                    reader.array(message.argument_keywords, () => reader.string());
                    break;
                case "allowed_positional_arguments":
                    message.allowed_positional_arguments = reader.int64();
                    break;
                case "function_spec":
                    message.function_spec = tensorflow.FunctionSpec.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedBareConcreteFunction.prototype.concrete_function_name = "";
tensorflow.SavedBareConcreteFunction.prototype.allowed_positional_arguments = 0n;
tensorflow.SavedBareConcreteFunction.prototype.function_spec = null;

tensorflow.SavedConstant = class SavedConstant {

    static decode(reader, length) {
        const message = new tensorflow.SavedConstant();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.operation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedConstant();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "operation":
                    message.operation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedConstant.prototype.operation = "";

tensorflow.SavedVariable = class SavedVariable {

    constructor() {
        this.experimental_distributed_variable_components = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedVariable();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.trainable = reader.bool();
                    break;
                case 4:
                    message.synchronization = reader.int32();
                    break;
                case 5:
                    message.aggregation = reader.int32();
                    break;
                case 6:
                    message.name = reader.string();
                    break;
                case 7:
                    message.device = reader.string();
                    break;
                case 8:
                    message.experimental_distributed_variable_components.push(tensorflow.SavedVariable.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "trainable":
                    message.trainable = reader.bool();
                    break;
                case "synchronization":
                    message.synchronization = reader.enum(tensorflow.VariableSynchronization);
                    break;
                case "aggregation":
                    message.aggregation = reader.enum(tensorflow.VariableAggregation);
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "device":
                    message.device = reader.string();
                    break;
                case "experimental_distributed_variable_components":
                    message.experimental_distributed_variable_components.push(tensorflow.SavedVariable.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedVariable.prototype.dtype = 0;
tensorflow.SavedVariable.prototype.shape = null;
tensorflow.SavedVariable.prototype.trainable = false;
tensorflow.SavedVariable.prototype.synchronization = 0;
tensorflow.SavedVariable.prototype.aggregation = 0;
tensorflow.SavedVariable.prototype.name = "";
tensorflow.SavedVariable.prototype.device = "";

tensorflow.FunctionSpec = class FunctionSpec {

    static decode(reader, length) {
        const message = new tensorflow.FunctionSpec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.fullargspec = tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.is_method = reader.bool();
                    break;
                case 5:
                    message.input_signature = tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.jit_compile = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.FunctionSpec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "fullargspec":
                    message.fullargspec = tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "is_method":
                    message.is_method = reader.bool();
                    break;
                case "input_signature":
                    message.input_signature = tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "jit_compile":
                    message.jit_compile = reader.enum(tensorflow.FunctionSpec.JitCompile);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FunctionSpec.prototype.fullargspec = null;
tensorflow.FunctionSpec.prototype.is_method = false;
tensorflow.FunctionSpec.prototype.input_signature = null;
tensorflow.FunctionSpec.prototype.jit_compile = 0;

tensorflow.FunctionSpec.JitCompile = {
    "DEFAULT": 0,
    "ON": 1,
    "OFF": 2
};

tensorflow.SavedResource = class SavedResource {

    static decode(reader, length) {
        const message = new tensorflow.SavedResource();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.device = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedResource();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "device":
                    message.device = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedResource.prototype.device = "";

tensorflow.SaveableObject = class SaveableObject {

    static decode(reader, length) {
        const message = new tensorflow.SaveableObject();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.save_function = reader.int32();
                    break;
                case 3:
                    message.restore_function = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SaveableObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "save_function":
                    message.save_function = reader.int32();
                    break;
                case "restore_function":
                    message.restore_function = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SaveableObject.prototype.save_function = 0;
tensorflow.SaveableObject.prototype.restore_function = 0;

tensorflow.VariableSynchronization = {
    "VARIABLE_SYNCHRONIZATION_AUTO": 0,
    "VARIABLE_SYNCHRONIZATION_NONE": 1,
    "VARIABLE_SYNCHRONIZATION_ON_WRITE": 2,
    "VARIABLE_SYNCHRONIZATION_ON_READ": 3
};

tensorflow.VariableAggregation = {
    "VARIABLE_AGGREGATION_NONE": 0,
    "VARIABLE_AGGREGATION_SUM": 1,
    "VARIABLE_AGGREGATION_MEAN": 2,
    "VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA": 3
};

tensorflow.VariableDef = class VariableDef {

    static decode(reader, length) {
        const message = new tensorflow.VariableDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.variable_name = reader.string();
                    break;
                case 6:
                    message.initial_value_name = reader.string();
                    break;
                case 2:
                    message.initializer_name = reader.string();
                    break;
                case 3:
                    message.snapshot_name = reader.string();
                    break;
                case 4:
                    message.save_slice_info_def = tensorflow.SaveSliceInfoDef.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.is_resource = reader.bool();
                    break;
                case 7:
                    message.trainable = reader.bool();
                    break;
                case 8:
                    message.synchronization = reader.int32();
                    break;
                case 9:
                    message.aggregation = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.VariableDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "variable_name":
                    message.variable_name = reader.string();
                    break;
                case "initial_value_name":
                    message.initial_value_name = reader.string();
                    break;
                case "initializer_name":
                    message.initializer_name = reader.string();
                    break;
                case "snapshot_name":
                    message.snapshot_name = reader.string();
                    break;
                case "save_slice_info_def":
                    message.save_slice_info_def = tensorflow.SaveSliceInfoDef.decodeText(reader);
                    break;
                case "is_resource":
                    message.is_resource = reader.bool();
                    break;
                case "trainable":
                    message.trainable = reader.bool();
                    break;
                case "synchronization":
                    message.synchronization = reader.enum(tensorflow.VariableSynchronization);
                    break;
                case "aggregation":
                    message.aggregation = reader.enum(tensorflow.VariableAggregation);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.VariableDef.prototype.variable_name = "";
tensorflow.VariableDef.prototype.initial_value_name = "";
tensorflow.VariableDef.prototype.initializer_name = "";
tensorflow.VariableDef.prototype.snapshot_name = "";
tensorflow.VariableDef.prototype.save_slice_info_def = null;
tensorflow.VariableDef.prototype.is_resource = false;
tensorflow.VariableDef.prototype.trainable = false;
tensorflow.VariableDef.prototype.synchronization = 0;
tensorflow.VariableDef.prototype.aggregation = 0;

tensorflow.SaveSliceInfoDef = class SaveSliceInfoDef {

    constructor() {
        this.full_shape = [];
        this.var_offset = [];
        this.var_shape = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SaveSliceInfoDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.full_name = reader.string();
                    break;
                case 2:
                    message.full_shape = reader.array(message.full_shape, () => reader.int64(), tag);
                    break;
                case 3:
                    message.var_offset = reader.array(message.var_offset, () => reader.int64(), tag);
                    break;
                case 4:
                    message.var_shape = reader.array(message.var_shape, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SaveSliceInfoDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "full_name":
                    message.full_name = reader.string();
                    break;
                case "full_shape":
                    reader.array(message.full_shape, () => reader.int64());
                    break;
                case "var_offset":
                    reader.array(message.var_offset, () => reader.int64());
                    break;
                case "var_shape":
                    reader.array(message.var_shape, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SaveSliceInfoDef.prototype.full_name = "";

tensorflow.StructuredValue = class StructuredValue {

    get kind() {
        tensorflow.StructuredValue.kindSet = tensorflow.StructuredValue.kindSet || new Set(["none_value", "float64_value", "int64_value", "string_value", "bool_value", "tensor_shape_value", "tensor_dtype_value", "tensor_spec_value", "type_spec_value", "bounded_tensor_spec_value", "list_value", "tuple_value", "dict_value", "named_tuple_value", "tensor_value", "numpy_value"]);
        return Object.keys(this).find((key) => tensorflow.StructuredValue.kindSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.StructuredValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.none_value = tensorflow.NoneValue.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.float64_value = reader.double();
                    break;
                case 12:
                    message.int64_value = reader.sint64();
                    break;
                case 13:
                    message.string_value = reader.string();
                    break;
                case 14:
                    message.bool_value = reader.bool();
                    break;
                case 31:
                    message.tensor_shape_value = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.tensor_dtype_value = reader.int32();
                    break;
                case 33:
                    message.tensor_spec_value = tensorflow.TensorSpecProto.decode(reader, reader.uint32());
                    break;
                case 34:
                    message.type_spec_value = tensorflow.TypeSpecProto.decode(reader, reader.uint32());
                    break;
                case 35:
                    message.bounded_tensor_spec_value = tensorflow.BoundedTensorSpecProto.decode(reader, reader.uint32());
                    break;
                case 51:
                    message.list_value = tensorflow.ListValue.decode(reader, reader.uint32());
                    break;
                case 52:
                    message.tuple_value = tensorflow.TupleValue.decode(reader, reader.uint32());
                    break;
                case 53:
                    message.dict_value = tensorflow.DictValue.decode(reader, reader.uint32());
                    break;
                case 54:
                    message.named_tuple_value = tensorflow.NamedTupleValue.decode(reader, reader.uint32());
                    break;
                case 55:
                    message.tensor_value = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                case 56:
                    message.numpy_value = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.StructuredValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "none_value":
                    message.none_value = tensorflow.NoneValue.decodeText(reader);
                    break;
                case "float64_value":
                    message.float64_value = reader.double();
                    break;
                case "int64_value":
                    message.int64_value = reader.sint64();
                    break;
                case "string_value":
                    message.string_value = reader.string();
                    break;
                case "bool_value":
                    message.bool_value = reader.bool();
                    break;
                case "tensor_shape_value":
                    message.tensor_shape_value = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "tensor_dtype_value":
                    message.tensor_dtype_value = reader.enum(tensorflow.DataType);
                    break;
                case "tensor_spec_value":
                    message.tensor_spec_value = tensorflow.TensorSpecProto.decodeText(reader);
                    break;
                case "type_spec_value":
                    message.type_spec_value = tensorflow.TypeSpecProto.decodeText(reader);
                    break;
                case "bounded_tensor_spec_value":
                    message.bounded_tensor_spec_value = tensorflow.BoundedTensorSpecProto.decodeText(reader);
                    break;
                case "list_value":
                    message.list_value = tensorflow.ListValue.decodeText(reader);
                    break;
                case "tuple_value":
                    message.tuple_value = tensorflow.TupleValue.decodeText(reader);
                    break;
                case "dict_value":
                    message.dict_value = tensorflow.DictValue.decodeText(reader);
                    break;
                case "named_tuple_value":
                    message.named_tuple_value = tensorflow.NamedTupleValue.decodeText(reader);
                    break;
                case "tensor_value":
                    message.tensor_value = tensorflow.TensorProto.decodeText(reader);
                    break;
                case "numpy_value":
                    message.numpy_value = tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NoneValue = class NoneValue {

    static decode(reader, length) {
        const message = new tensorflow.NoneValue();
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
        const message = new tensorflow.NoneValue();
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

tensorflow.ListValue = class ListValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.ListValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push(tensorflow.StructuredValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push(tensorflow.StructuredValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TupleValue = class TupleValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TupleValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push(tensorflow.StructuredValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TupleValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push(tensorflow.StructuredValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.DictValue = class DictValue {

    constructor() {
        this.fields = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.DictValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.fields, () => reader.string(), () => tensorflow.StructuredValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.DictValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "fields":
                    reader.entry(message.fields, () => reader.string(), () => tensorflow.StructuredValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.PairValue = class PairValue {

    static decode(reader, length) {
        const message = new tensorflow.PairValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.PairValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = tensorflow.StructuredValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.PairValue.prototype.key = "";
tensorflow.PairValue.prototype.value = null;

tensorflow.NamedTupleValue = class NamedTupleValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.NamedTupleValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.values.push(tensorflow.PairValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.NamedTupleValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "values":
                    message.values.push(tensorflow.PairValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NamedTupleValue.prototype.name = "";

tensorflow.TensorSpecProto = class TensorSpecProto {

    static decode(reader, length) {
        const message = new tensorflow.TensorSpecProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.dtype = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorSpecProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorSpecProto.prototype.name = "";
tensorflow.TensorSpecProto.prototype.shape = null;
tensorflow.TensorSpecProto.prototype.dtype = 0;

tensorflow.BoundedTensorSpecProto = class BoundedTensorSpecProto {

    static decode(reader, length) {
        const message = new tensorflow.BoundedTensorSpecProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.dtype = reader.int32();
                    break;
                case 4:
                    message.minimum = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.maximum = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.BoundedTensorSpecProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "minimum":
                    message.minimum = tensorflow.TensorProto.decodeText(reader);
                    break;
                case "maximum":
                    message.maximum = tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.BoundedTensorSpecProto.prototype.name = "";
tensorflow.BoundedTensorSpecProto.prototype.shape = null;
tensorflow.BoundedTensorSpecProto.prototype.dtype = 0;
tensorflow.BoundedTensorSpecProto.prototype.minimum = null;
tensorflow.BoundedTensorSpecProto.prototype.maximum = null;

tensorflow.TypeSpecProto = class TypeSpecProto {

    static decode(reader, length) {
        const message = new tensorflow.TypeSpecProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_spec_class = reader.int32();
                    break;
                case 2:
                    message.type_state = tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.type_spec_class_name = reader.string();
                    break;
                case 4:
                    message.num_flat_components = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TypeSpecProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_spec_class":
                    message.type_spec_class = reader.enum(tensorflow.TypeSpecProto.TypeSpecClass);
                    break;
                case "type_state":
                    message.type_state = tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "type_spec_class_name":
                    message.type_spec_class_name = reader.string();
                    break;
                case "num_flat_components":
                    message.num_flat_components = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TypeSpecProto.prototype.type_spec_class = 0;
tensorflow.TypeSpecProto.prototype.type_state = null;
tensorflow.TypeSpecProto.prototype.type_spec_class_name = "";
tensorflow.TypeSpecProto.prototype.num_flat_components = 0;

tensorflow.TypeSpecProto.TypeSpecClass = {
    "UNKNOWN": 0,
    "SPARSE_TENSOR_SPEC": 1,
    "INDEXED_SLICES_SPEC": 2,
    "RAGGED_TENSOR_SPEC": 3,
    "TENSOR_ARRAY_SPEC": 4,
    "DATA_DATASET_SPEC": 5,
    "DATA_ITERATOR_SPEC": 6,
    "OPTIONAL_SPEC": 7,
    "PER_REPLICA_SPEC": 8,
    "VARIABLE_SPEC": 9,
    "ROW_PARTITION_SPEC": 10,
    "REGISTERED_TYPE_SPEC": 12,
    "EXTENSION_TYPE_SPEC": 13
};

tensorflow.TrackableObjectGraph = class TrackableObjectGraph {

    constructor() {
        this.nodes = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TrackableObjectGraph();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push(tensorflow.TrackableObjectGraph.TrackableObject.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TrackableObjectGraph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push(tensorflow.TrackableObjectGraph.TrackableObject.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TrackableObjectGraph.TrackableObject = class TrackableObject {

    constructor() {
        this.children = [];
        this.attributes = [];
        this.slot_variables = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.children.push(tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.attributes.push(tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.slot_variables.push(tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.registered_saver = tensorflow.RegisteredSaver.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.has_checkpoint_values = google.protobuf.BoolValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "children":
                    message.children.push(tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader));
                    break;
                case "attributes":
                    message.attributes.push(tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decodeText(reader));
                    break;
                case "slot_variables":
                    message.slot_variables.push(tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeText(reader));
                    break;
                case "registered_saver":
                    message.registered_saver = tensorflow.RegisteredSaver.decodeText(reader);
                    break;
                case "has_checkpoint_values":
                    message.has_checkpoint_values = google.protobuf.BoolValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TrackableObjectGraph.TrackableObject.prototype.registered_saver = null;
tensorflow.TrackableObjectGraph.TrackableObject.prototype.has_checkpoint_values = null;

tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference = class ObjectReference {

    static decode(reader, length) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node_id = reader.int32();
                    break;
                case 2:
                    message.local_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_id":
                    message.node_id = reader.int32();
                    break;
                case "local_name":
                    message.local_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.prototype.node_id = 0;
tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.prototype.local_name = "";

tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor = class SerializedTensor {

    static decode(reader, length) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.full_name = reader.string();
                    break;
                case 3:
                    message.checkpoint_key = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "full_name":
                    message.full_name = reader.string();
                    break;
                case "checkpoint_key":
                    message.checkpoint_key = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.name = "";
tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.full_name = "";
tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.checkpoint_key = "";

tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference = class SlotVariableReference {

    static decode(reader, length) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.original_variable_node_id = reader.int32();
                    break;
                case 2:
                    message.slot_name = reader.string();
                    break;
                case 3:
                    message.slot_variable_node_id = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "original_variable_node_id":
                    message.original_variable_node_id = reader.int32();
                    break;
                case "slot_name":
                    message.slot_name = reader.string();
                    break;
                case "slot_variable_node_id":
                    message.slot_variable_node_id = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.prototype.original_variable_node_id = 0;
tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.prototype.slot_name = "";
tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.prototype.slot_variable_node_id = 0;

tensorflow.RegisteredSaver = class RegisteredSaver {

    static decode(reader, length) {
        const message = new tensorflow.RegisteredSaver();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.object_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RegisteredSaver();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "object_name":
                    message.object_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RegisteredSaver.prototype.name = "";
tensorflow.RegisteredSaver.prototype.object_name = "";

tensorflow.SaverDef = class SaverDef {

    static decode(reader, length) {
        const message = new tensorflow.SaverDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.filename_tensor_name = reader.string();
                    break;
                case 2:
                    message.save_tensor_name = reader.string();
                    break;
                case 3:
                    message.restore_op_name = reader.string();
                    break;
                case 4:
                    message.max_to_keep = reader.int32();
                    break;
                case 5:
                    message.sharded = reader.bool();
                    break;
                case 6:
                    message.keep_checkpoint_every_n_hours = reader.float();
                    break;
                case 7:
                    message.version = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SaverDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "filename_tensor_name":
                    message.filename_tensor_name = reader.string();
                    break;
                case "save_tensor_name":
                    message.save_tensor_name = reader.string();
                    break;
                case "restore_op_name":
                    message.restore_op_name = reader.string();
                    break;
                case "max_to_keep":
                    message.max_to_keep = reader.int32();
                    break;
                case "sharded":
                    message.sharded = reader.bool();
                    break;
                case "keep_checkpoint_every_n_hours":
                    message.keep_checkpoint_every_n_hours = reader.float();
                    break;
                case "version":
                    message.version = reader.enum(tensorflow.SaverDef.CheckpointFormatVersion);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SaverDef.prototype.filename_tensor_name = "";
tensorflow.SaverDef.prototype.save_tensor_name = "";
tensorflow.SaverDef.prototype.restore_op_name = "";
tensorflow.SaverDef.prototype.max_to_keep = 0;
tensorflow.SaverDef.prototype.sharded = false;
tensorflow.SaverDef.prototype.keep_checkpoint_every_n_hours = 0;
tensorflow.SaverDef.prototype.version = 0;

tensorflow.SaverDef.CheckpointFormatVersion = {
    "LEGACY": 0,
    "V1": 1,
    "V2": 2
};

tensorflow.BundleHeaderProto = class BundleHeaderProto {

    static decode(reader, length) {
        const message = new tensorflow.BundleHeaderProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_shards = reader.int32();
                    break;
                case 2:
                    message.endianness = reader.int32();
                    break;
                case 3:
                    message.version = tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.BundleHeaderProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_shards":
                    message.num_shards = reader.int32();
                    break;
                case "endianness":
                    message.endianness = reader.enum(tensorflow.BundleHeaderProto.Endianness);
                    break;
                case "version":
                    message.version = tensorflow.VersionDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.BundleHeaderProto.prototype.num_shards = 0;
tensorflow.BundleHeaderProto.prototype.endianness = 0;
tensorflow.BundleHeaderProto.prototype.version = null;

tensorflow.BundleHeaderProto.Endianness = {
    "LITTLE": 0,
    "BIG": 1
};

tensorflow.BundleEntryProto = class BundleEntryProto {

    constructor() {
        this.slices = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.BundleEntryProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.shard_id = reader.int32();
                    break;
                case 4:
                    message.offset = reader.int64();
                    break;
                case 5:
                    message.size = reader.int64();
                    break;
                case 6:
                    message.crc32c = reader.fixed32();
                    break;
                case 7:
                    message.slices.push(tensorflow.TensorSliceProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.BundleEntryProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "shard_id":
                    message.shard_id = reader.int32();
                    break;
                case "offset":
                    message.offset = reader.int64();
                    break;
                case "size":
                    message.size = reader.int64();
                    break;
                case "crc32c":
                    message.crc32c = reader.fixed32();
                    break;
                case "slices":
                    message.slices.push(tensorflow.TensorSliceProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.BundleEntryProto.prototype.dtype = 0;
tensorflow.BundleEntryProto.prototype.shape = null;
tensorflow.BundleEntryProto.prototype.shard_id = 0;
tensorflow.BundleEntryProto.prototype.offset = 0n;
tensorflow.BundleEntryProto.prototype.size = 0n;
tensorflow.BundleEntryProto.prototype.crc32c = 0;

tensorflow.TensorSliceProto = class TensorSliceProto {

    constructor() {
        this.extent = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.TensorSliceProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.extent.push(tensorflow.TensorSliceProto.Extent.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorSliceProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "extent":
                    message.extent.push(tensorflow.TensorSliceProto.Extent.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorSliceProto.Extent = class Extent {

    get has_length() {
        tensorflow.TensorSliceProto.Extent.has_lengthSet = tensorflow.TensorSliceProto.Extent.has_lengthSet || new Set(["length"]);
        return Object.keys(this).find((key) => tensorflow.TensorSliceProto.Extent.has_lengthSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.TensorSliceProto.Extent();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.start = reader.int64();
                    break;
                case 2:
                    message.length = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorSliceProto.Extent();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    message.start = reader.int64();
                    break;
                case "length":
                    message.length = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorSliceProto.Extent.prototype.start = 0n;

tensorflow.SavedSliceMeta = class SavedSliceMeta {

    constructor() {
        this.slice = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedSliceMeta();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.type = reader.int32();
                    break;
                case 4:
                    message.slice.push(tensorflow.TensorSliceProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedSliceMeta();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "type":
                    message.type = reader.enum(tensorflow.DataType);
                    break;
                case "slice":
                    message.slice.push(tensorflow.TensorSliceProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedSliceMeta.prototype.name = "";
tensorflow.SavedSliceMeta.prototype.shape = null;
tensorflow.SavedSliceMeta.prototype.type = 0;

tensorflow.SavedTensorSliceMeta = class SavedTensorSliceMeta {

    constructor() {
        this.tensor = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.SavedTensorSliceMeta();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor.push(tensorflow.SavedSliceMeta.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.versions = tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedTensorSliceMeta();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor.push(tensorflow.SavedSliceMeta.decodeText(reader));
                    break;
                case "versions":
                    message.versions = tensorflow.VersionDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedTensorSliceMeta.prototype.versions = null;

tensorflow.SavedSlice = class SavedSlice {

    static decode(reader, length) {
        const message = new tensorflow.SavedSlice();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.slice = tensorflow.TensorSliceProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.data = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedSlice();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "slice":
                    message.slice = tensorflow.TensorSliceProto.decodeText(reader);
                    break;
                case "data":
                    message.data = tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedSlice.prototype.name = "";
tensorflow.SavedSlice.prototype.slice = null;
tensorflow.SavedSlice.prototype.data = null;

tensorflow.SavedTensorSlices = class SavedTensorSlices {

    static decode(reader, length) {
        const message = new tensorflow.SavedTensorSlices();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meta = tensorflow.SavedTensorSliceMeta.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.data = tensorflow.SavedSlice.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SavedTensorSlices();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meta":
                    message.meta = tensorflow.SavedTensorSliceMeta.decodeText(reader);
                    break;
                case "data":
                    message.data = tensorflow.SavedSlice.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SavedTensorSlices.prototype.meta = null;
tensorflow.SavedTensorSlices.prototype.data = null;

tensorflow.Event = class Event {

    get what() {
        tensorflow.Event.whatSet = tensorflow.Event.whatSet || new Set(["file_version", "graph_def", "summary", "log_message", "session_log", "tagged_run_metadata", "meta_graph_def"]);
        return Object.keys(this).find((key) => tensorflow.Event.whatSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.Event();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.wall_time = reader.double();
                    break;
                case 2:
                    message.step = reader.int64();
                    break;
                case 3:
                    message.file_version = reader.string();
                    break;
                case 4:
                    message.graph_def = reader.bytes();
                    break;
                case 5:
                    message.summary = tensorflow.Summary.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.log_message = tensorflow.LogMessage.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.session_log = tensorflow.SessionLog.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.tagged_run_metadata = tensorflow.TaggedRunMetadata.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.meta_graph_def = reader.bytes();
                    break;
                case 10:
                    message.source_metadata = tensorflow.SourceMetadata.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.Event();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "wall_time":
                    message.wall_time = reader.double();
                    break;
                case "step":
                    message.step = reader.int64();
                    break;
                case "file_version":
                    message.file_version = reader.string();
                    break;
                case "graph_def":
                    message.graph_def = reader.bytes();
                    break;
                case "summary":
                    message.summary = tensorflow.Summary.decodeText(reader);
                    break;
                case "log_message":
                    message.log_message = tensorflow.LogMessage.decodeText(reader);
                    break;
                case "session_log":
                    message.session_log = tensorflow.SessionLog.decodeText(reader);
                    break;
                case "tagged_run_metadata":
                    message.tagged_run_metadata = tensorflow.TaggedRunMetadata.decodeText(reader);
                    break;
                case "meta_graph_def":
                    message.meta_graph_def = reader.bytes();
                    break;
                case "source_metadata":
                    message.source_metadata = tensorflow.SourceMetadata.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.Event.prototype.wall_time = 0;
tensorflow.Event.prototype.step = 0n;
tensorflow.Event.prototype.source_metadata = null;

tensorflow.SourceMetadata = class SourceMetadata {

    static decode(reader, length) {
        const message = new tensorflow.SourceMetadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.writer = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SourceMetadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "writer":
                    message.writer = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SourceMetadata.prototype.writer = "";

tensorflow.LogMessage = class LogMessage {

    static decode(reader, length) {
        const message = new tensorflow.LogMessage();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.level = reader.int32();
                    break;
                case 2:
                    message.message = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.LogMessage();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "level":
                    message.level = reader.enum(tensorflow.LogMessage.Level);
                    break;
                case "message":
                    message.message = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.LogMessage.prototype.level = 0;
tensorflow.LogMessage.prototype.message = "";

tensorflow.LogMessage.Level = {
    "UNKNOWN": 0,
    "DEBUGGING": 10,
    "INFO": 20,
    "WARN": 30,
    "ERROR": 40,
    "FATAL": 50
};

tensorflow.SessionLog = class SessionLog {

    static decode(reader, length) {
        const message = new tensorflow.SessionLog();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.status = reader.int32();
                    break;
                case 2:
                    message.checkpoint_path = reader.string();
                    break;
                case 3:
                    message.msg = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SessionLog();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "status":
                    message.status = reader.enum(tensorflow.SessionLog.SessionStatus);
                    break;
                case "checkpoint_path":
                    message.checkpoint_path = reader.string();
                    break;
                case "msg":
                    message.msg = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SessionLog.prototype.status = 0;
tensorflow.SessionLog.prototype.checkpoint_path = "";
tensorflow.SessionLog.prototype.msg = "";

tensorflow.SessionLog.SessionStatus = {
    "STATUS_UNSPECIFIED": 0,
    "START": 1,
    "STOP": 2,
    "CHECKPOINT": 3
};

tensorflow.TaggedRunMetadata = class TaggedRunMetadata {

    static decode(reader, length) {
        const message = new tensorflow.TaggedRunMetadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tag = reader.string();
                    break;
                case 2:
                    message.run_metadata = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TaggedRunMetadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tag":
                    message.tag = reader.string();
                    break;
                case "run_metadata":
                    message.run_metadata = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TaggedRunMetadata.prototype.tag = "";
tensorflow.TaggedRunMetadata.prototype.run_metadata = new Uint8Array([]);

tensorflow.WorkerHealth = {
    "OK": 0,
    "RECEIVED_SHUTDOWN_SIGNAL": 1,
    "INTERNAL_ERROR": 2,
    "SHUTTING_DOWN": 3
};

tensorflow.WorkerShutdownMode = {
    "DEFAULT": 0,
    "NOT_CONFIGURED": 1,
    "WAIT_FOR_COORDINATOR": 2,
    "SHUTDOWN_AFTER_TIMEOUT": 3
};

tensorflow.WatchdogConfig = class WatchdogConfig {

    static decode(reader, length) {
        const message = new tensorflow.WatchdogConfig();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.timeout_ms = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.WatchdogConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "timeout_ms":
                    message.timeout_ms = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.WatchdogConfig.prototype.timeout_ms = 0n;

tensorflow.RequestedExitCode = class RequestedExitCode {

    static decode(reader, length) {
        const message = new tensorflow.RequestedExitCode();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.exit_code = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RequestedExitCode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "exit_code":
                    message.exit_code = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RequestedExitCode.prototype.exit_code = 0;

tensorflow.WorkerHeartbeatRequest = class WorkerHeartbeatRequest {

    static decode(reader, length) {
        const message = new tensorflow.WorkerHeartbeatRequest();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shutdown_mode = reader.int32();
                    break;
                case 2:
                    message.watchdog_config = tensorflow.WatchdogConfig.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.exit_code = tensorflow.RequestedExitCode.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.WorkerHeartbeatRequest();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shutdown_mode":
                    message.shutdown_mode = reader.enum(tensorflow.WorkerShutdownMode);
                    break;
                case "watchdog_config":
                    message.watchdog_config = tensorflow.WatchdogConfig.decodeText(reader);
                    break;
                case "exit_code":
                    message.exit_code = tensorflow.RequestedExitCode.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.WorkerHeartbeatRequest.prototype.shutdown_mode = 0;
tensorflow.WorkerHeartbeatRequest.prototype.watchdog_config = null;
tensorflow.WorkerHeartbeatRequest.prototype.exit_code = null;

tensorflow.WorkerHeartbeatResponse = class WorkerHeartbeatResponse {

    constructor() {
        this.worker_log = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.WorkerHeartbeatResponse();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.health_status = reader.int32();
                    break;
                case 2:
                    message.worker_log.push(tensorflow.Event.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.hostname = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.WorkerHeartbeatResponse();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "health_status":
                    message.health_status = reader.enum(tensorflow.WorkerHealth);
                    break;
                case "worker_log":
                    message.worker_log.push(tensorflow.Event.decodeText(reader));
                    break;
                case "hostname":
                    message.hostname = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.WorkerHeartbeatResponse.prototype.health_status = 0;
tensorflow.WorkerHeartbeatResponse.prototype.hostname = "";

tensorflow.SummaryDescription = class SummaryDescription {

    static decode(reader, length) {
        const message = new tensorflow.SummaryDescription();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_hint = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SummaryDescription();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_hint":
                    message.type_hint = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SummaryDescription.prototype.type_hint = "";

tensorflow.SummaryMetadata = class SummaryMetadata {

    static decode(reader, length) {
        const message = new tensorflow.SummaryMetadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.plugin_data = tensorflow.SummaryMetadata.PluginData.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.display_name = reader.string();
                    break;
                case 3:
                    message.summary_description = reader.string();
                    break;
                case 4:
                    message.data_class = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SummaryMetadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "plugin_data":
                    message.plugin_data = tensorflow.SummaryMetadata.PluginData.decodeText(reader);
                    break;
                case "display_name":
                    message.display_name = reader.string();
                    break;
                case "summary_description":
                    message.summary_description = reader.string();
                    break;
                case "data_class":
                    message.data_class = reader.enum(tensorflow.DataClass);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SummaryMetadata.prototype.plugin_data = null;
tensorflow.SummaryMetadata.prototype.display_name = "";
tensorflow.SummaryMetadata.prototype.summary_description = "";
tensorflow.SummaryMetadata.prototype.data_class = 0;

tensorflow.SummaryMetadata.PluginData = class PluginData {

    static decode(reader, length) {
        const message = new tensorflow.SummaryMetadata.PluginData();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.plugin_name = reader.string();
                    break;
                case 2:
                    message.content = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.SummaryMetadata.PluginData();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "plugin_name":
                    message.plugin_name = reader.string();
                    break;
                case "content":
                    message.content = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.SummaryMetadata.PluginData.prototype.plugin_name = "";
tensorflow.SummaryMetadata.PluginData.prototype.content = new Uint8Array([]);

tensorflow.DataClass = {
    "DATA_CLASS_UNKNOWN": 0,
    "DATA_CLASS_SCALAR": 1,
    "DATA_CLASS_TENSOR": 2,
    "DATA_CLASS_BLOB_SEQUENCE": 3
};

tensorflow.Summary = class Summary {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.Summary();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push(tensorflow.Summary.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.Summary();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value.push(tensorflow.Summary.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.Summary.Image = class Image {

    static decode(reader, length) {
        const message = new tensorflow.Summary.Image();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.height = reader.int32();
                    break;
                case 2:
                    message.width = reader.int32();
                    break;
                case 3:
                    message.colorspace = reader.int32();
                    break;
                case 4:
                    message.encoded_image_string = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.Summary.Image();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "height":
                    message.height = reader.int32();
                    break;
                case "width":
                    message.width = reader.int32();
                    break;
                case "colorspace":
                    message.colorspace = reader.int32();
                    break;
                case "encoded_image_string":
                    message.encoded_image_string = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.Summary.Image.prototype.height = 0;
tensorflow.Summary.Image.prototype.width = 0;
tensorflow.Summary.Image.prototype.colorspace = 0;
tensorflow.Summary.Image.prototype.encoded_image_string = new Uint8Array([]);

tensorflow.Summary.Audio = class Audio {

    static decode(reader, length) {
        const message = new tensorflow.Summary.Audio();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.sample_rate = reader.float();
                    break;
                case 2:
                    message.num_channels = reader.int64();
                    break;
                case 3:
                    message.length_frames = reader.int64();
                    break;
                case 4:
                    message.encoded_audio_string = reader.bytes();
                    break;
                case 5:
                    message.content_type = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.Summary.Audio();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sample_rate":
                    message.sample_rate = reader.float();
                    break;
                case "num_channels":
                    message.num_channels = reader.int64();
                    break;
                case "length_frames":
                    message.length_frames = reader.int64();
                    break;
                case "encoded_audio_string":
                    message.encoded_audio_string = reader.bytes();
                    break;
                case "content_type":
                    message.content_type = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.Summary.Audio.prototype.sample_rate = 0;
tensorflow.Summary.Audio.prototype.num_channels = 0n;
tensorflow.Summary.Audio.prototype.length_frames = 0n;
tensorflow.Summary.Audio.prototype.encoded_audio_string = new Uint8Array([]);
tensorflow.Summary.Audio.prototype.content_type = "";

tensorflow.Summary.Value = class Value {

    get value() {
        tensorflow.Summary.Value.valueSet = tensorflow.Summary.Value.valueSet || new Set(["simple_value", "obsolete_old_style_histogram", "image", "histo", "audio", "tensor"]);
        return Object.keys(this).find((key) => tensorflow.Summary.Value.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new tensorflow.Summary.Value();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 7:
                    message.node_name = reader.string();
                    break;
                case 1:
                    message.tag = reader.string();
                    break;
                case 9:
                    message.metadata = tensorflow.SummaryMetadata.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.simple_value = reader.float();
                    break;
                case 3:
                    message.obsolete_old_style_histogram = reader.bytes();
                    break;
                case 4:
                    message.image = tensorflow.Summary.Image.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.histo = tensorflow.HistogramProto.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.audio = tensorflow.Summary.Audio.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.tensor = tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.Summary.Value();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_name":
                    message.node_name = reader.string();
                    break;
                case "tag":
                    message.tag = reader.string();
                    break;
                case "metadata":
                    message.metadata = tensorflow.SummaryMetadata.decodeText(reader);
                    break;
                case "simple_value":
                    message.simple_value = reader.float();
                    break;
                case "obsolete_old_style_histogram":
                    message.obsolete_old_style_histogram = reader.bytes();
                    break;
                case "image":
                    message.image = tensorflow.Summary.Image.decodeText(reader);
                    break;
                case "histo":
                    message.histo = tensorflow.HistogramProto.decodeText(reader);
                    break;
                case "audio":
                    message.audio = tensorflow.Summary.Audio.decodeText(reader);
                    break;
                case "tensor":
                    message.tensor = tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.Summary.Value.prototype.node_name = "";
tensorflow.Summary.Value.prototype.tag = "";
tensorflow.Summary.Value.prototype.metadata = null;

tensorflow.HistogramProto = class HistogramProto {

    constructor() {
        this.bucket_limit = [];
        this.bucket = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.HistogramProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.min = reader.double();
                    break;
                case 2:
                    message.max = reader.double();
                    break;
                case 3:
                    message.num = reader.double();
                    break;
                case 4:
                    message.sum = reader.double();
                    break;
                case 5:
                    message.sum_squares = reader.double();
                    break;
                case 6:
                    message.bucket_limit = reader.doubles(message.bucket_limit, tag);
                    break;
                case 7:
                    message.bucket = reader.doubles(message.bucket, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.HistogramProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "min":
                    message.min = reader.double();
                    break;
                case "max":
                    message.max = reader.double();
                    break;
                case "num":
                    message.num = reader.double();
                    break;
                case "sum":
                    message.sum = reader.double();
                    break;
                case "sum_squares":
                    message.sum_squares = reader.double();
                    break;
                case "bucket_limit":
                    reader.array(message.bucket_limit, () => reader.double());
                    break;
                case "bucket":
                    reader.array(message.bucket, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.HistogramProto.prototype.min = 0;
tensorflow.HistogramProto.prototype.max = 0;
tensorflow.HistogramProto.prototype.num = 0;
tensorflow.HistogramProto.prototype.sum = 0;
tensorflow.HistogramProto.prototype.sum_squares = 0;

tensorflow.GPUOptions = class GPUOptions {

    static decode(reader, length) {
        const message = new tensorflow.GPUOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.per_process_gpu_memory_fraction = reader.double();
                    break;
                case 4:
                    message.allow_growth = reader.bool();
                    break;
                case 2:
                    message.allocator_type = reader.string();
                    break;
                case 3:
                    message.deferred_deletion_bytes = reader.int64();
                    break;
                case 5:
                    message.visible_device_list = reader.string();
                    break;
                case 6:
                    message.polling_active_delay_usecs = reader.int32();
                    break;
                case 7:
                    message.polling_inactive_delay_msecs = reader.int32();
                    break;
                case 8:
                    message.force_gpu_compatible = reader.bool();
                    break;
                case 9:
                    message.experimental = tensorflow.GPUOptions.Experimental.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GPUOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "per_process_gpu_memory_fraction":
                    message.per_process_gpu_memory_fraction = reader.double();
                    break;
                case "allow_growth":
                    message.allow_growth = reader.bool();
                    break;
                case "allocator_type":
                    message.allocator_type = reader.string();
                    break;
                case "deferred_deletion_bytes":
                    message.deferred_deletion_bytes = reader.int64();
                    break;
                case "visible_device_list":
                    message.visible_device_list = reader.string();
                    break;
                case "polling_active_delay_usecs":
                    message.polling_active_delay_usecs = reader.int32();
                    break;
                case "polling_inactive_delay_msecs":
                    message.polling_inactive_delay_msecs = reader.int32();
                    break;
                case "force_gpu_compatible":
                    message.force_gpu_compatible = reader.bool();
                    break;
                case "experimental":
                    message.experimental = tensorflow.GPUOptions.Experimental.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GPUOptions.prototype.per_process_gpu_memory_fraction = 0;
tensorflow.GPUOptions.prototype.allow_growth = false;
tensorflow.GPUOptions.prototype.allocator_type = "";
tensorflow.GPUOptions.prototype.deferred_deletion_bytes = 0n;
tensorflow.GPUOptions.prototype.visible_device_list = "";
tensorflow.GPUOptions.prototype.polling_active_delay_usecs = 0;
tensorflow.GPUOptions.prototype.polling_inactive_delay_msecs = 0;
tensorflow.GPUOptions.prototype.force_gpu_compatible = false;
tensorflow.GPUOptions.prototype.experimental = null;

tensorflow.GPUOptions.Experimental = class Experimental {

    constructor() {
        this.virtual_devices = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.GPUOptions.Experimental();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.virtual_devices.push(tensorflow.GPUOptions.Experimental.VirtualDevices.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.num_virtual_devices_per_gpu = reader.int32();
                    break;
                case 2:
                    message.use_unified_memory = reader.bool();
                    break;
                case 3:
                    message.num_dev_to_dev_copy_streams = reader.int32();
                    break;
                case 4:
                    message.collective_ring_order = reader.string();
                    break;
                case 5:
                    message.timestamped_allocator = reader.bool();
                    break;
                case 7:
                    message.kernel_tracker_max_interval = reader.int32();
                    break;
                case 8:
                    message.kernel_tracker_max_bytes = reader.int32();
                    break;
                case 9:
                    message.kernel_tracker_max_pending = reader.int32();
                    break;
                case 10:
                    message.internal_fragmentation_fraction = reader.double();
                    break;
                case 11:
                    message.use_cuda_malloc_async = reader.bool();
                    break;
                case 12:
                    message.disallow_retry_on_allocation_failure = reader.bool();
                    break;
                case 13:
                    message.gpu_host_mem_limit_in_mb = reader.float();
                    break;
                case 14:
                    message.gpu_host_mem_disallow_growth = reader.bool();
                    break;
                case 16:
                    message.gpu_system_memory_size_in_mb = reader.int32();
                    break;
                case 17:
                    message.populate_pjrt_gpu_client_creation_info = reader.bool();
                    break;
                case 18:
                    message.node_id = reader.int32();
                    break;
                case 19:
                    message.stream_merge_options = tensorflow.GPUOptions.Experimental.StreamMergeOptions.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GPUOptions.Experimental();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "virtual_devices":
                    message.virtual_devices.push(tensorflow.GPUOptions.Experimental.VirtualDevices.decodeText(reader));
                    break;
                case "num_virtual_devices_per_gpu":
                    message.num_virtual_devices_per_gpu = reader.int32();
                    break;
                case "use_unified_memory":
                    message.use_unified_memory = reader.bool();
                    break;
                case "num_dev_to_dev_copy_streams":
                    message.num_dev_to_dev_copy_streams = reader.int32();
                    break;
                case "collective_ring_order":
                    message.collective_ring_order = reader.string();
                    break;
                case "timestamped_allocator":
                    message.timestamped_allocator = reader.bool();
                    break;
                case "kernel_tracker_max_interval":
                    message.kernel_tracker_max_interval = reader.int32();
                    break;
                case "kernel_tracker_max_bytes":
                    message.kernel_tracker_max_bytes = reader.int32();
                    break;
                case "kernel_tracker_max_pending":
                    message.kernel_tracker_max_pending = reader.int32();
                    break;
                case "internal_fragmentation_fraction":
                    message.internal_fragmentation_fraction = reader.double();
                    break;
                case "use_cuda_malloc_async":
                    message.use_cuda_malloc_async = reader.bool();
                    break;
                case "disallow_retry_on_allocation_failure":
                    message.disallow_retry_on_allocation_failure = reader.bool();
                    break;
                case "gpu_host_mem_limit_in_mb":
                    message.gpu_host_mem_limit_in_mb = reader.float();
                    break;
                case "gpu_host_mem_disallow_growth":
                    message.gpu_host_mem_disallow_growth = reader.bool();
                    break;
                case "gpu_system_memory_size_in_mb":
                    message.gpu_system_memory_size_in_mb = reader.int32();
                    break;
                case "populate_pjrt_gpu_client_creation_info":
                    message.populate_pjrt_gpu_client_creation_info = reader.bool();
                    break;
                case "node_id":
                    message.node_id = reader.int32();
                    break;
                case "stream_merge_options":
                    message.stream_merge_options = tensorflow.GPUOptions.Experimental.StreamMergeOptions.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GPUOptions.Experimental.prototype.num_virtual_devices_per_gpu = 0;
tensorflow.GPUOptions.Experimental.prototype.use_unified_memory = false;
tensorflow.GPUOptions.Experimental.prototype.num_dev_to_dev_copy_streams = 0;
tensorflow.GPUOptions.Experimental.prototype.collective_ring_order = "";
tensorflow.GPUOptions.Experimental.prototype.timestamped_allocator = false;
tensorflow.GPUOptions.Experimental.prototype.kernel_tracker_max_interval = 0;
tensorflow.GPUOptions.Experimental.prototype.kernel_tracker_max_bytes = 0;
tensorflow.GPUOptions.Experimental.prototype.kernel_tracker_max_pending = 0;
tensorflow.GPUOptions.Experimental.prototype.internal_fragmentation_fraction = 0;
tensorflow.GPUOptions.Experimental.prototype.use_cuda_malloc_async = false;
tensorflow.GPUOptions.Experimental.prototype.disallow_retry_on_allocation_failure = false;
tensorflow.GPUOptions.Experimental.prototype.gpu_host_mem_limit_in_mb = 0;
tensorflow.GPUOptions.Experimental.prototype.gpu_host_mem_disallow_growth = false;
tensorflow.GPUOptions.Experimental.prototype.gpu_system_memory_size_in_mb = 0;
tensorflow.GPUOptions.Experimental.prototype.populate_pjrt_gpu_client_creation_info = false;
tensorflow.GPUOptions.Experimental.prototype.node_id = 0;
tensorflow.GPUOptions.Experimental.prototype.stream_merge_options = null;

tensorflow.GPUOptions.Experimental.VirtualDevices = class VirtualDevices {

    constructor() {
        this.memory_limit_mb = [];
        this.priority = [];
        this.device_ordinal = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.GPUOptions.Experimental.VirtualDevices();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.memory_limit_mb = reader.floats(message.memory_limit_mb, tag);
                    break;
                case 2:
                    message.priority = reader.array(message.priority, () => reader.int32(), tag);
                    break;
                case 3:
                    message.device_ordinal = reader.array(message.device_ordinal, () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GPUOptions.Experimental.VirtualDevices();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "memory_limit_mb":
                    reader.array(message.memory_limit_mb, () => reader.float());
                    break;
                case "priority":
                    reader.array(message.priority, () => reader.int32());
                    break;
                case "device_ordinal":
                    reader.array(message.device_ordinal, () => reader.int32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GPUOptions.Experimental.StreamMergeOptions = class StreamMergeOptions {

    static decode(reader, length) {
        const message = new tensorflow.GPUOptions.Experimental.StreamMergeOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.merge_host_to_device_stream = reader.bool();
                    break;
                case 2:
                    message.merge_device_to_host_stream = reader.bool();
                    break;
                case 3:
                    message.merge_device_to_device_stream = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GPUOptions.Experimental.StreamMergeOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "merge_host_to_device_stream":
                    message.merge_host_to_device_stream = reader.bool();
                    break;
                case "merge_device_to_host_stream":
                    message.merge_device_to_host_stream = reader.bool();
                    break;
                case "merge_device_to_device_stream":
                    message.merge_device_to_device_stream = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GPUOptions.Experimental.StreamMergeOptions.prototype.merge_host_to_device_stream = false;
tensorflow.GPUOptions.Experimental.StreamMergeOptions.prototype.merge_device_to_host_stream = false;
tensorflow.GPUOptions.Experimental.StreamMergeOptions.prototype.merge_device_to_device_stream = false;

tensorflow.OptimizerOptions = class OptimizerOptions {

    static decode(reader, length) {
        const message = new tensorflow.OptimizerOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.do_common_subexpression_elimination = reader.bool();
                    break;
                case 2:
                    message.do_constant_folding = reader.bool();
                    break;
                case 6:
                    message.max_folded_constant_in_bytes = reader.int64();
                    break;
                case 4:
                    message.do_function_inlining = reader.bool();
                    break;
                case 3:
                    message.opt_level = reader.int32();
                    break;
                case 5:
                    message.global_jit_level = reader.int32();
                    break;
                case 7:
                    message.cpu_global_jit = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.OptimizerOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "do_common_subexpression_elimination":
                    message.do_common_subexpression_elimination = reader.bool();
                    break;
                case "do_constant_folding":
                    message.do_constant_folding = reader.bool();
                    break;
                case "max_folded_constant_in_bytes":
                    message.max_folded_constant_in_bytes = reader.int64();
                    break;
                case "do_function_inlining":
                    message.do_function_inlining = reader.bool();
                    break;
                case "opt_level":
                    message.opt_level = reader.enum(tensorflow.OptimizerOptions.Level);
                    break;
                case "global_jit_level":
                    message.global_jit_level = reader.enum(tensorflow.OptimizerOptions.GlobalJitLevel);
                    break;
                case "cpu_global_jit":
                    message.cpu_global_jit = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.OptimizerOptions.prototype.do_common_subexpression_elimination = false;
tensorflow.OptimizerOptions.prototype.do_constant_folding = false;
tensorflow.OptimizerOptions.prototype.max_folded_constant_in_bytes = 0n;
tensorflow.OptimizerOptions.prototype.do_function_inlining = false;
tensorflow.OptimizerOptions.prototype.opt_level = 0;
tensorflow.OptimizerOptions.prototype.global_jit_level = 0;
tensorflow.OptimizerOptions.prototype.cpu_global_jit = false;

tensorflow.OptimizerOptions.Level = {
    "L1": 0,
    "L0": -1
};

tensorflow.OptimizerOptions.GlobalJitLevel = {
    "DEFAULT": 0,
    "OFF": -1,
    "ON_1": 1,
    "ON_2": 2
};

tensorflow.GraphOptions = class GraphOptions {

    static decode(reader, length) {
        const message = new tensorflow.GraphOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.enable_recv_scheduling = reader.bool();
                    break;
                case 3:
                    message.optimizer_options = tensorflow.OptimizerOptions.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.build_cost_model = reader.int64();
                    break;
                case 9:
                    message.build_cost_model_after = reader.int64();
                    break;
                case 5:
                    message.infer_shapes = reader.bool();
                    break;
                case 6:
                    message.place_pruned_graph = reader.bool();
                    break;
                case 7:
                    message.enable_bfloat16_sendrecv = reader.bool();
                    break;
                case 8:
                    message.timeline_step = reader.int32();
                    break;
                case 10:
                    message.rewrite_options = tensorflow.RewriterConfig.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.GraphOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "enable_recv_scheduling":
                    message.enable_recv_scheduling = reader.bool();
                    break;
                case "optimizer_options":
                    message.optimizer_options = tensorflow.OptimizerOptions.decodeText(reader);
                    break;
                case "build_cost_model":
                    message.build_cost_model = reader.int64();
                    break;
                case "build_cost_model_after":
                    message.build_cost_model_after = reader.int64();
                    break;
                case "infer_shapes":
                    message.infer_shapes = reader.bool();
                    break;
                case "place_pruned_graph":
                    message.place_pruned_graph = reader.bool();
                    break;
                case "enable_bfloat16_sendrecv":
                    message.enable_bfloat16_sendrecv = reader.bool();
                    break;
                case "timeline_step":
                    message.timeline_step = reader.int32();
                    break;
                case "rewrite_options":
                    message.rewrite_options = tensorflow.RewriterConfig.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.GraphOptions.prototype.enable_recv_scheduling = false;
tensorflow.GraphOptions.prototype.optimizer_options = null;
tensorflow.GraphOptions.prototype.build_cost_model = 0n;
tensorflow.GraphOptions.prototype.build_cost_model_after = 0n;
tensorflow.GraphOptions.prototype.infer_shapes = false;
tensorflow.GraphOptions.prototype.place_pruned_graph = false;
tensorflow.GraphOptions.prototype.enable_bfloat16_sendrecv = false;
tensorflow.GraphOptions.prototype.timeline_step = 0;
tensorflow.GraphOptions.prototype.rewrite_options = null;

tensorflow.ThreadPoolOptionProto = class ThreadPoolOptionProto {

    static decode(reader, length) {
        const message = new tensorflow.ThreadPoolOptionProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_threads = reader.int32();
                    break;
                case 2:
                    message.global_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ThreadPoolOptionProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_threads":
                    message.num_threads = reader.int32();
                    break;
                case "global_name":
                    message.global_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.ThreadPoolOptionProto.prototype.num_threads = 0;
tensorflow.ThreadPoolOptionProto.prototype.global_name = "";

tensorflow.SessionMetadata = class SessionMetadata {

    static decode(reader, length) {
        const message = new tensorflow.SessionMetadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
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
        const message = new tensorflow.SessionMetadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
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
};

tensorflow.SessionMetadata.prototype.name = "";
tensorflow.SessionMetadata.prototype.version = 0n;

tensorflow.ConfigProto = class ConfigProto {

    constructor() {
        this.device_count = {};
        this.session_inter_op_thread_pool = [];
        this.device_filters = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.ConfigProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.device_count, () => reader.string(), () => reader.int32());
                    break;
                case 2:
                    message.intra_op_parallelism_threads = reader.int32();
                    break;
                case 5:
                    message.inter_op_parallelism_threads = reader.int32();
                    break;
                case 9:
                    message.use_per_session_threads = reader.bool();
                    break;
                case 12:
                    message.session_inter_op_thread_pool.push(tensorflow.ThreadPoolOptionProto.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.placement_period = reader.int32();
                    break;
                case 4:
                    message.device_filters.push(reader.string());
                    break;
                case 6:
                    message.gpu_options = tensorflow.GPUOptions.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.allow_soft_placement = reader.bool();
                    break;
                case 8:
                    message.log_device_placement = reader.bool();
                    break;
                case 10:
                    message.graph_options = tensorflow.GraphOptions.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.operation_timeout_in_ms = reader.int64();
                    break;
                case 13:
                    message.rpc_options = tensorflow.RPCOptions.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.cluster_def = tensorflow.ClusterDef.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.isolate_session_state = reader.bool();
                    break;
                case 17:
                    message.share_cluster_devices_in_session = reader.bool();
                    break;
                case 16:
                    message.experimental = tensorflow.ConfigProto.Experimental.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ConfigProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "device_count":
                    reader.entry(message.device_count, () => reader.string(), () => reader.int32());
                    break;
                case "intra_op_parallelism_threads":
                    message.intra_op_parallelism_threads = reader.int32();
                    break;
                case "inter_op_parallelism_threads":
                    message.inter_op_parallelism_threads = reader.int32();
                    break;
                case "use_per_session_threads":
                    message.use_per_session_threads = reader.bool();
                    break;
                case "session_inter_op_thread_pool":
                    message.session_inter_op_thread_pool.push(tensorflow.ThreadPoolOptionProto.decodeText(reader));
                    break;
                case "placement_period":
                    message.placement_period = reader.int32();
                    break;
                case "device_filters":
                    reader.array(message.device_filters, () => reader.string());
                    break;
                case "gpu_options":
                    message.gpu_options = tensorflow.GPUOptions.decodeText(reader);
                    break;
                case "allow_soft_placement":
                    message.allow_soft_placement = reader.bool();
                    break;
                case "log_device_placement":
                    message.log_device_placement = reader.bool();
                    break;
                case "graph_options":
                    message.graph_options = tensorflow.GraphOptions.decodeText(reader);
                    break;
                case "operation_timeout_in_ms":
                    message.operation_timeout_in_ms = reader.int64();
                    break;
                case "rpc_options":
                    message.rpc_options = tensorflow.RPCOptions.decodeText(reader);
                    break;
                case "cluster_def":
                    message.cluster_def = tensorflow.ClusterDef.decodeText(reader);
                    break;
                case "isolate_session_state":
                    message.isolate_session_state = reader.bool();
                    break;
                case "share_cluster_devices_in_session":
                    message.share_cluster_devices_in_session = reader.bool();
                    break;
                case "experimental":
                    message.experimental = tensorflow.ConfigProto.Experimental.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.ConfigProto.prototype.intra_op_parallelism_threads = 0;
tensorflow.ConfigProto.prototype.inter_op_parallelism_threads = 0;
tensorflow.ConfigProto.prototype.use_per_session_threads = false;
tensorflow.ConfigProto.prototype.placement_period = 0;
tensorflow.ConfigProto.prototype.gpu_options = null;
tensorflow.ConfigProto.prototype.allow_soft_placement = false;
tensorflow.ConfigProto.prototype.log_device_placement = false;
tensorflow.ConfigProto.prototype.graph_options = null;
tensorflow.ConfigProto.prototype.operation_timeout_in_ms = 0n;
tensorflow.ConfigProto.prototype.rpc_options = null;
tensorflow.ConfigProto.prototype.cluster_def = null;
tensorflow.ConfigProto.prototype.isolate_session_state = false;
tensorflow.ConfigProto.prototype.share_cluster_devices_in_session = false;
tensorflow.ConfigProto.prototype.experimental = null;

tensorflow.ConfigProto.Experimental = class Experimental {

    static decode(reader, length) {
        const message = new tensorflow.ConfigProto.Experimental();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.collective_group_leader = reader.string();
                    break;
                case 3:
                    message.executor_type = reader.string();
                    break;
                case 4:
                    message.recv_buf_max_chunk = reader.int32();
                    break;
                case 5:
                    message.use_numa_affinity = reader.bool();
                    break;
                case 6:
                    message.collective_deterministic_sequential_execution = reader.bool();
                    break;
                case 7:
                    message.collective_nccl = reader.bool();
                    break;
                case 8:
                    message.share_session_state_in_clusterspec_propagation = reader.bool();
                    break;
                case 9:
                    message.disable_thread_spinning = reader.bool();
                    break;
                case 10:
                    message.share_cluster_devices_in_session = reader.bool();
                    break;
                case 11:
                    message.session_metadata = tensorflow.SessionMetadata.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.optimize_for_static_graph = reader.bool();
                    break;
                case 13:
                    message.enable_mlir_bridge = reader.bool();
                    break;
                case 17:
                    message.mlir_bridge_rollout = reader.int32();
                    break;
                case 16:
                    message.enable_mlir_graph_optimization = reader.bool();
                    break;
                case 14:
                    message.disable_output_partition_graphs = reader.bool();
                    break;
                case 15:
                    message.xla_fusion_autotuner_thresh = reader.int64();
                    break;
                case 18:
                    message.use_tfrt = reader.bool();
                    break;
                case 27:
                    message.enable_multi_host = reader.bool();
                    break;
                case 28:
                    message.backend_server_port = reader.int32();
                    break;
                case 29:
                    message.target_tpu = reader.bool();
                    break;
                case 30:
                    message.target_gpu = reader.bool();
                    break;
                case 31:
                    message.stream_merge_threshold = reader.int32();
                    break;
                case 21:
                    message.disable_functional_ops_lowering = reader.bool();
                    break;
                case 22:
                    message.xla_prefer_single_graph_cluster = reader.bool();
                    break;
                case 23:
                    message.coordination_config = tensorflow.CoordinationServiceConfig.decode(reader, reader.uint32());
                    break;
                case 24:
                    message.disable_optimize_for_static_graph = reader.bool();
                    break;
                case 26:
                    message.disable_eager_executor_streaming_enqueue = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ConfigProto.Experimental();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "collective_group_leader":
                    message.collective_group_leader = reader.string();
                    break;
                case "executor_type":
                    message.executor_type = reader.string();
                    break;
                case "recv_buf_max_chunk":
                    message.recv_buf_max_chunk = reader.int32();
                    break;
                case "use_numa_affinity":
                    message.use_numa_affinity = reader.bool();
                    break;
                case "collective_deterministic_sequential_execution":
                    message.collective_deterministic_sequential_execution = reader.bool();
                    break;
                case "collective_nccl":
                    message.collective_nccl = reader.bool();
                    break;
                case "share_session_state_in_clusterspec_propagation":
                    message.share_session_state_in_clusterspec_propagation = reader.bool();
                    break;
                case "disable_thread_spinning":
                    message.disable_thread_spinning = reader.bool();
                    break;
                case "share_cluster_devices_in_session":
                    message.share_cluster_devices_in_session = reader.bool();
                    break;
                case "session_metadata":
                    message.session_metadata = tensorflow.SessionMetadata.decodeText(reader);
                    break;
                case "optimize_for_static_graph":
                    message.optimize_for_static_graph = reader.bool();
                    break;
                case "enable_mlir_bridge":
                    message.enable_mlir_bridge = reader.bool();
                    break;
                case "mlir_bridge_rollout":
                    message.mlir_bridge_rollout = reader.enum(tensorflow.ConfigProto.Experimental.MlirBridgeRollout);
                    break;
                case "enable_mlir_graph_optimization":
                    message.enable_mlir_graph_optimization = reader.bool();
                    break;
                case "disable_output_partition_graphs":
                    message.disable_output_partition_graphs = reader.bool();
                    break;
                case "xla_fusion_autotuner_thresh":
                    message.xla_fusion_autotuner_thresh = reader.int64();
                    break;
                case "use_tfrt":
                    message.use_tfrt = reader.bool();
                    break;
                case "enable_multi_host":
                    message.enable_multi_host = reader.bool();
                    break;
                case "backend_server_port":
                    message.backend_server_port = reader.int32();
                    break;
                case "target_tpu":
                    message.target_tpu = reader.bool();
                    break;
                case "target_gpu":
                    message.target_gpu = reader.bool();
                    break;
                case "stream_merge_threshold":
                    message.stream_merge_threshold = reader.int32();
                    break;
                case "disable_functional_ops_lowering":
                    message.disable_functional_ops_lowering = reader.bool();
                    break;
                case "xla_prefer_single_graph_cluster":
                    message.xla_prefer_single_graph_cluster = reader.bool();
                    break;
                case "coordination_config":
                    message.coordination_config = tensorflow.CoordinationServiceConfig.decodeText(reader);
                    break;
                case "disable_optimize_for_static_graph":
                    message.disable_optimize_for_static_graph = reader.bool();
                    break;
                case "disable_eager_executor_streaming_enqueue":
                    message.disable_eager_executor_streaming_enqueue = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.ConfigProto.Experimental.prototype.collective_group_leader = "";
tensorflow.ConfigProto.Experimental.prototype.executor_type = "";
tensorflow.ConfigProto.Experimental.prototype.recv_buf_max_chunk = 0;
tensorflow.ConfigProto.Experimental.prototype.use_numa_affinity = false;
tensorflow.ConfigProto.Experimental.prototype.collective_deterministic_sequential_execution = false;
tensorflow.ConfigProto.Experimental.prototype.collective_nccl = false;
tensorflow.ConfigProto.Experimental.prototype.share_session_state_in_clusterspec_propagation = false;
tensorflow.ConfigProto.Experimental.prototype.disable_thread_spinning = false;
tensorflow.ConfigProto.Experimental.prototype.share_cluster_devices_in_session = false;
tensorflow.ConfigProto.Experimental.prototype.session_metadata = null;
tensorflow.ConfigProto.Experimental.prototype.optimize_for_static_graph = false;
tensorflow.ConfigProto.Experimental.prototype.enable_mlir_bridge = false;
tensorflow.ConfigProto.Experimental.prototype.mlir_bridge_rollout = 0;
tensorflow.ConfigProto.Experimental.prototype.enable_mlir_graph_optimization = false;
tensorflow.ConfigProto.Experimental.prototype.disable_output_partition_graphs = false;
tensorflow.ConfigProto.Experimental.prototype.xla_fusion_autotuner_thresh = 0n;
tensorflow.ConfigProto.Experimental.prototype.use_tfrt = false;
tensorflow.ConfigProto.Experimental.prototype.enable_multi_host = false;
tensorflow.ConfigProto.Experimental.prototype.backend_server_port = 0;
tensorflow.ConfigProto.Experimental.prototype.target_tpu = false;
tensorflow.ConfigProto.Experimental.prototype.target_gpu = false;
tensorflow.ConfigProto.Experimental.prototype.stream_merge_threshold = 0;
tensorflow.ConfigProto.Experimental.prototype.disable_functional_ops_lowering = false;
tensorflow.ConfigProto.Experimental.prototype.xla_prefer_single_graph_cluster = false;
tensorflow.ConfigProto.Experimental.prototype.coordination_config = null;
tensorflow.ConfigProto.Experimental.prototype.disable_optimize_for_static_graph = false;
tensorflow.ConfigProto.Experimental.prototype.disable_eager_executor_streaming_enqueue = false;

tensorflow.ConfigProto.Experimental.MlirBridgeRollout = {
    "MLIR_BRIDGE_ROLLOUT_UNSPECIFIED": 0,
    "MLIR_BRIDGE_ROLLOUT_ENABLED": 1,
    "MLIR_BRIDGE_ROLLOUT_DISABLED": 2
};

tensorflow.RunOptions = class RunOptions {

    static decode(reader, length) {
        const message = new tensorflow.RunOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.trace_level = reader.int32();
                    break;
                case 2:
                    message.timeout_in_ms = reader.int64();
                    break;
                case 3:
                    message.inter_op_thread_pool = reader.int32();
                    break;
                case 5:
                    message.output_partition_graphs = reader.bool();
                    break;
                case 6:
                    message.debug_options = tensorflow.DebugOptions.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.report_tensor_allocations_upon_oom = reader.bool();
                    break;
                case 8:
                    message.experimental = tensorflow.RunOptions.Experimental.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RunOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "trace_level":
                    message.trace_level = reader.enum(tensorflow.RunOptions.TraceLevel);
                    break;
                case "timeout_in_ms":
                    message.timeout_in_ms = reader.int64();
                    break;
                case "inter_op_thread_pool":
                    message.inter_op_thread_pool = reader.int32();
                    break;
                case "output_partition_graphs":
                    message.output_partition_graphs = reader.bool();
                    break;
                case "debug_options":
                    message.debug_options = tensorflow.DebugOptions.decodeText(reader);
                    break;
                case "report_tensor_allocations_upon_oom":
                    message.report_tensor_allocations_upon_oom = reader.bool();
                    break;
                case "experimental":
                    message.experimental = tensorflow.RunOptions.Experimental.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RunOptions.prototype.trace_level = 0;
tensorflow.RunOptions.prototype.timeout_in_ms = 0n;
tensorflow.RunOptions.prototype.inter_op_thread_pool = 0;
tensorflow.RunOptions.prototype.output_partition_graphs = false;
tensorflow.RunOptions.prototype.debug_options = null;
tensorflow.RunOptions.prototype.report_tensor_allocations_upon_oom = false;
tensorflow.RunOptions.prototype.experimental = null;

tensorflow.RunOptions.TraceLevel = {
    "NO_TRACE": 0,
    "SOFTWARE_TRACE": 1,
    "HARDWARE_TRACE": 2,
    "FULL_TRACE": 3
};

tensorflow.RunOptions.Experimental = class Experimental {

    static decode(reader, length) {
        const message = new tensorflow.RunOptions.Experimental();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.collective_graph_key = reader.int64();
                    break;
                case 2:
                    message.use_run_handler_pool = reader.bool();
                    break;
                case 3:
                    message.run_handler_pool_options = tensorflow.RunOptions.Experimental.RunHandlerPoolOptions.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RunOptions.Experimental();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "collective_graph_key":
                    message.collective_graph_key = reader.int64();
                    break;
                case "use_run_handler_pool":
                    message.use_run_handler_pool = reader.bool();
                    break;
                case "run_handler_pool_options":
                    message.run_handler_pool_options = tensorflow.RunOptions.Experimental.RunHandlerPoolOptions.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RunOptions.Experimental.prototype.collective_graph_key = 0n;
tensorflow.RunOptions.Experimental.prototype.use_run_handler_pool = false;
tensorflow.RunOptions.Experimental.prototype.run_handler_pool_options = null;

tensorflow.RunOptions.Experimental.RunHandlerPoolOptions = class RunHandlerPoolOptions {

    static decode(reader, length) {
        const message = new tensorflow.RunOptions.Experimental.RunHandlerPoolOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.priority = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RunOptions.Experimental.RunHandlerPoolOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "priority":
                    message.priority = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RunOptions.Experimental.RunHandlerPoolOptions.prototype.priority = 0n;

tensorflow.RunMetadata = class RunMetadata {

    constructor() {
        this.partition_graphs = [];
        this.function_graphs = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.RunMetadata();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.step_stats = tensorflow.StepStats.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.cost_graph = tensorflow.CostGraphDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.partition_graphs.push(tensorflow.GraphDef.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.function_graphs.push(tensorflow.RunMetadata.FunctionGraphs.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.session_metadata = tensorflow.SessionMetadata.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RunMetadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "step_stats":
                    message.step_stats = tensorflow.StepStats.decodeText(reader);
                    break;
                case "cost_graph":
                    message.cost_graph = tensorflow.CostGraphDef.decodeText(reader);
                    break;
                case "partition_graphs":
                    message.partition_graphs.push(tensorflow.GraphDef.decodeText(reader));
                    break;
                case "function_graphs":
                    message.function_graphs.push(tensorflow.RunMetadata.FunctionGraphs.decodeText(reader));
                    break;
                case "session_metadata":
                    message.session_metadata = tensorflow.SessionMetadata.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RunMetadata.prototype.step_stats = null;
tensorflow.RunMetadata.prototype.cost_graph = null;
tensorflow.RunMetadata.prototype.session_metadata = null;

tensorflow.RunMetadata.FunctionGraphs = class FunctionGraphs {

    constructor() {
        this.partition_graphs = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.RunMetadata.FunctionGraphs();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.partition_graphs.push(tensorflow.GraphDef.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.pre_optimization_graph = tensorflow.GraphDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.post_optimization_graph = tensorflow.GraphDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RunMetadata.FunctionGraphs();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "partition_graphs":
                    message.partition_graphs.push(tensorflow.GraphDef.decodeText(reader));
                    break;
                case "pre_optimization_graph":
                    message.pre_optimization_graph = tensorflow.GraphDef.decodeText(reader);
                    break;
                case "post_optimization_graph":
                    message.post_optimization_graph = tensorflow.GraphDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RunMetadata.FunctionGraphs.prototype.pre_optimization_graph = null;
tensorflow.RunMetadata.FunctionGraphs.prototype.post_optimization_graph = null;

tensorflow.TensorConnection = class TensorConnection {

    static decode(reader, length) {
        const message = new tensorflow.TensorConnection();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.from_tensor = reader.string();
                    break;
                case 2:
                    message.to_tensor = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorConnection();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "from_tensor":
                    message.from_tensor = reader.string();
                    break;
                case "to_tensor":
                    message.to_tensor = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorConnection.prototype.from_tensor = "";
tensorflow.TensorConnection.prototype.to_tensor = "";

tensorflow.CallableOptions = class CallableOptions {

    constructor() {
        this.feed = [];
        this.fetch = [];
        this.target = [];
        this.tensor_connection = [];
        this.feed_devices = {};
        this.fetch_devices = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.CallableOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.feed.push(reader.string());
                    break;
                case 2:
                    message.fetch.push(reader.string());
                    break;
                case 3:
                    message.target.push(reader.string());
                    break;
                case 4:
                    message.run_options = tensorflow.RunOptions.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.tensor_connection.push(tensorflow.TensorConnection.decode(reader, reader.uint32()));
                    break;
                case 6:
                    reader.entry(message.feed_devices, () => reader.string(), () => reader.string());
                    break;
                case 7:
                    reader.entry(message.fetch_devices, () => reader.string(), () => reader.string());
                    break;
                case 8:
                    message.fetch_skip_sync = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CallableOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "feed":
                    reader.array(message.feed, () => reader.string());
                    break;
                case "fetch":
                    reader.array(message.fetch, () => reader.string());
                    break;
                case "target":
                    reader.array(message.target, () => reader.string());
                    break;
                case "run_options":
                    message.run_options = tensorflow.RunOptions.decodeText(reader);
                    break;
                case "tensor_connection":
                    message.tensor_connection.push(tensorflow.TensorConnection.decodeText(reader));
                    break;
                case "feed_devices":
                    reader.entry(message.feed_devices, () => reader.string(), () => reader.string());
                    break;
                case "fetch_devices":
                    reader.entry(message.fetch_devices, () => reader.string(), () => reader.string());
                    break;
                case "fetch_skip_sync":
                    message.fetch_skip_sync = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CallableOptions.prototype.run_options = null;
tensorflow.CallableOptions.prototype.fetch_skip_sync = false;

tensorflow.CostGraphDef = class CostGraphDef {

    constructor() {
        this.node = [];
        this.cost = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CostGraphDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push(tensorflow.CostGraphDef.Node.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.cost.push(tensorflow.CostGraphDef.AggregatedCost.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CostGraphDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node":
                    message.node.push(tensorflow.CostGraphDef.Node.decodeText(reader));
                    break;
                case "cost":
                    message.cost.push(tensorflow.CostGraphDef.AggregatedCost.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CostGraphDef.Node = class Node {

    constructor() {
        this.input_info = [];
        this.output_info = [];
        this.control_input = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CostGraphDef.Node();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.device = reader.string();
                    break;
                case 3:
                    message.id = reader.int32();
                    break;
                case 4:
                    message.input_info.push(tensorflow.CostGraphDef.Node.InputInfo.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.output_info.push(tensorflow.CostGraphDef.Node.OutputInfo.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.temporary_memory_size = reader.int64();
                    break;
                case 12:
                    message.persistent_memory_size = reader.int64();
                    break;
                case 10:
                    message.host_temp_memory_size = reader.int64();
                    break;
                case 11:
                    message.device_temp_memory_size = reader.int64();
                    break;
                case 16:
                    message.device_persistent_memory_size = reader.int64();
                    break;
                case 9:
                    message.compute_cost = reader.int64();
                    break;
                case 14:
                    message.compute_time = reader.int64();
                    break;
                case 15:
                    message.memory_time = reader.int64();
                    break;
                case 7:
                    message.is_final = reader.bool();
                    break;
                case 8:
                    message.control_input = reader.array(message.control_input, () => reader.int32(), tag);
                    break;
                case 17:
                    message.inaccurate = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CostGraphDef.Node();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "device":
                    message.device = reader.string();
                    break;
                case "id":
                    message.id = reader.int32();
                    break;
                case "input_info":
                    message.input_info.push(tensorflow.CostGraphDef.Node.InputInfo.decodeText(reader));
                    break;
                case "output_info":
                    message.output_info.push(tensorflow.CostGraphDef.Node.OutputInfo.decodeText(reader));
                    break;
                case "temporary_memory_size":
                    message.temporary_memory_size = reader.int64();
                    break;
                case "persistent_memory_size":
                    message.persistent_memory_size = reader.int64();
                    break;
                case "host_temp_memory_size":
                    message.host_temp_memory_size = reader.int64();
                    break;
                case "device_temp_memory_size":
                    message.device_temp_memory_size = reader.int64();
                    break;
                case "device_persistent_memory_size":
                    message.device_persistent_memory_size = reader.int64();
                    break;
                case "compute_cost":
                    message.compute_cost = reader.int64();
                    break;
                case "compute_time":
                    message.compute_time = reader.int64();
                    break;
                case "memory_time":
                    message.memory_time = reader.int64();
                    break;
                case "is_final":
                    message.is_final = reader.bool();
                    break;
                case "control_input":
                    reader.array(message.control_input, () => reader.int32());
                    break;
                case "inaccurate":
                    message.inaccurate = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CostGraphDef.Node.prototype.name = "";
tensorflow.CostGraphDef.Node.prototype.device = "";
tensorflow.CostGraphDef.Node.prototype.id = 0;
tensorflow.CostGraphDef.Node.prototype.temporary_memory_size = 0n;
tensorflow.CostGraphDef.Node.prototype.persistent_memory_size = 0n;
tensorflow.CostGraphDef.Node.prototype.host_temp_memory_size = 0n;
tensorflow.CostGraphDef.Node.prototype.device_temp_memory_size = 0n;
tensorflow.CostGraphDef.Node.prototype.device_persistent_memory_size = 0n;
tensorflow.CostGraphDef.Node.prototype.compute_cost = 0n;
tensorflow.CostGraphDef.Node.prototype.compute_time = 0n;
tensorflow.CostGraphDef.Node.prototype.memory_time = 0n;
tensorflow.CostGraphDef.Node.prototype.is_final = false;
tensorflow.CostGraphDef.Node.prototype.inaccurate = false;

tensorflow.CostGraphDef.Node.InputInfo = class InputInfo {

    static decode(reader, length) {
        const message = new tensorflow.CostGraphDef.Node.InputInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.preceding_node = reader.int32();
                    break;
                case 2:
                    message.preceding_port = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CostGraphDef.Node.InputInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "preceding_node":
                    message.preceding_node = reader.int32();
                    break;
                case "preceding_port":
                    message.preceding_port = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CostGraphDef.Node.InputInfo.prototype.preceding_node = 0;
tensorflow.CostGraphDef.Node.InputInfo.prototype.preceding_port = 0;

tensorflow.CostGraphDef.Node.OutputInfo = class OutputInfo {

    static decode(reader, length) {
        const message = new tensorflow.CostGraphDef.Node.OutputInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.size = reader.int64();
                    break;
                case 2:
                    message.alias_input_port = reader.int64();
                    break;
                case 3:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.dtype = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CostGraphDef.Node.OutputInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    message.size = reader.int64();
                    break;
                case "alias_input_port":
                    message.alias_input_port = reader.int64();
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CostGraphDef.Node.OutputInfo.prototype.size = 0n;
tensorflow.CostGraphDef.Node.OutputInfo.prototype.alias_input_port = 0n;
tensorflow.CostGraphDef.Node.OutputInfo.prototype.shape = null;
tensorflow.CostGraphDef.Node.OutputInfo.prototype.dtype = 0;

tensorflow.CostGraphDef.AggregatedCost = class AggregatedCost {

    static decode(reader, length) {
        const message = new tensorflow.CostGraphDef.AggregatedCost();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.cost = reader.float();
                    break;
                case 2:
                    message.dimension = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CostGraphDef.AggregatedCost();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "cost":
                    message.cost = reader.float();
                    break;
                case "dimension":
                    message.dimension = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CostGraphDef.AggregatedCost.prototype.cost = 0;
tensorflow.CostGraphDef.AggregatedCost.prototype.dimension = "";

tensorflow.AllocationRecord = class AllocationRecord {

    static decode(reader, length) {
        const message = new tensorflow.AllocationRecord();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.alloc_micros = reader.int64();
                    break;
                case 2:
                    message.alloc_bytes = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AllocationRecord();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "alloc_micros":
                    message.alloc_micros = reader.int64();
                    break;
                case "alloc_bytes":
                    message.alloc_bytes = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AllocationRecord.prototype.alloc_micros = 0n;
tensorflow.AllocationRecord.prototype.alloc_bytes = 0n;

tensorflow.AllocatorMemoryUsed = class AllocatorMemoryUsed {

    constructor() {
        this.allocation_records = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.AllocatorMemoryUsed();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.allocator_name = reader.string();
                    break;
                case 2:
                    message.total_bytes = reader.int64();
                    break;
                case 3:
                    message.peak_bytes = reader.int64();
                    break;
                case 4:
                    message.live_bytes = reader.int64();
                    break;
                case 6:
                    message.allocation_records.push(tensorflow.AllocationRecord.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.allocator_bytes_in_use = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AllocatorMemoryUsed();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "allocator_name":
                    message.allocator_name = reader.string();
                    break;
                case "total_bytes":
                    message.total_bytes = reader.int64();
                    break;
                case "peak_bytes":
                    message.peak_bytes = reader.int64();
                    break;
                case "live_bytes":
                    message.live_bytes = reader.int64();
                    break;
                case "allocation_records":
                    message.allocation_records.push(tensorflow.AllocationRecord.decodeText(reader));
                    break;
                case "allocator_bytes_in_use":
                    message.allocator_bytes_in_use = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AllocatorMemoryUsed.prototype.allocator_name = "";
tensorflow.AllocatorMemoryUsed.prototype.total_bytes = 0n;
tensorflow.AllocatorMemoryUsed.prototype.peak_bytes = 0n;
tensorflow.AllocatorMemoryUsed.prototype.live_bytes = 0n;
tensorflow.AllocatorMemoryUsed.prototype.allocator_bytes_in_use = 0n;

tensorflow.NodeOutput = class NodeOutput {

    static decode(reader, length) {
        const message = new tensorflow.NodeOutput();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.slot = reader.int32();
                    break;
                case 3:
                    message.tensor_description = tensorflow.TensorDescription.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.NodeOutput();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "slot":
                    message.slot = reader.int32();
                    break;
                case "tensor_description":
                    message.tensor_description = tensorflow.TensorDescription.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NodeOutput.prototype.slot = 0;
tensorflow.NodeOutput.prototype.tensor_description = null;

tensorflow.MemoryStats = class MemoryStats {

    constructor() {
        this.persistent_tensor_alloc_ids = [];
        this.device_persistent_tensor_alloc_ids = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.MemoryStats();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.temp_memory_size = reader.int64();
                    break;
                case 3:
                    message.persistent_memory_size = reader.int64();
                    break;
                case 5:
                    message.persistent_tensor_alloc_ids = reader.array(message.persistent_tensor_alloc_ids, () => reader.int64(), tag);
                    break;
                case 2:
                    message.device_temp_memory_size = reader.int64();
                    break;
                case 4:
                    message.device_persistent_memory_size = reader.int64();
                    break;
                case 6:
                    message.device_persistent_tensor_alloc_ids = reader.array(message.device_persistent_tensor_alloc_ids, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.MemoryStats();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "temp_memory_size":
                    message.temp_memory_size = reader.int64();
                    break;
                case "persistent_memory_size":
                    message.persistent_memory_size = reader.int64();
                    break;
                case "persistent_tensor_alloc_ids":
                    reader.array(message.persistent_tensor_alloc_ids, () => reader.int64());
                    break;
                case "device_temp_memory_size":
                    message.device_temp_memory_size = reader.int64();
                    break;
                case "device_persistent_memory_size":
                    message.device_persistent_memory_size = reader.int64();
                    break;
                case "device_persistent_tensor_alloc_ids":
                    reader.array(message.device_persistent_tensor_alloc_ids, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.MemoryStats.prototype.temp_memory_size = 0n;
tensorflow.MemoryStats.prototype.persistent_memory_size = 0n;
tensorflow.MemoryStats.prototype.device_temp_memory_size = 0n;
tensorflow.MemoryStats.prototype.device_persistent_memory_size = 0n;

tensorflow.NodeExecStats = class NodeExecStats {

    constructor() {
        this.memory = [];
        this.output = [];
        this.referenced_tensor = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.NodeExecStats();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node_name = reader.string();
                    break;
                case 2:
                    message.all_start_micros = reader.int64();
                    break;
                case 3:
                    message.op_start_rel_micros = reader.int64();
                    break;
                case 4:
                    message.op_end_rel_micros = reader.int64();
                    break;
                case 5:
                    message.all_end_rel_micros = reader.int64();
                    break;
                case 6:
                    message.memory.push(tensorflow.AllocatorMemoryUsed.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.output.push(tensorflow.NodeOutput.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.timeline_label = reader.string();
                    break;
                case 9:
                    message.scheduled_micros = reader.int64();
                    break;
                case 10:
                    message.thread_id = reader.uint32();
                    break;
                case 11:
                    message.referenced_tensor.push(tensorflow.AllocationDescription.decode(reader, reader.uint32()));
                    break;
                case 12:
                    message.memory_stats = tensorflow.MemoryStats.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.all_start_nanos = reader.int64();
                    break;
                case 14:
                    message.op_start_rel_nanos = reader.int64();
                    break;
                case 15:
                    message.op_end_rel_nanos = reader.int64();
                    break;
                case 16:
                    message.all_end_rel_nanos = reader.int64();
                    break;
                case 17:
                    message.scheduled_nanos = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.NodeExecStats();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_name":
                    message.node_name = reader.string();
                    break;
                case "all_start_micros":
                    message.all_start_micros = reader.int64();
                    break;
                case "op_start_rel_micros":
                    message.op_start_rel_micros = reader.int64();
                    break;
                case "op_end_rel_micros":
                    message.op_end_rel_micros = reader.int64();
                    break;
                case "all_end_rel_micros":
                    message.all_end_rel_micros = reader.int64();
                    break;
                case "memory":
                    message.memory.push(tensorflow.AllocatorMemoryUsed.decodeText(reader));
                    break;
                case "output":
                    message.output.push(tensorflow.NodeOutput.decodeText(reader));
                    break;
                case "timeline_label":
                    message.timeline_label = reader.string();
                    break;
                case "scheduled_micros":
                    message.scheduled_micros = reader.int64();
                    break;
                case "thread_id":
                    message.thread_id = reader.uint32();
                    break;
                case "referenced_tensor":
                    message.referenced_tensor.push(tensorflow.AllocationDescription.decodeText(reader));
                    break;
                case "memory_stats":
                    message.memory_stats = tensorflow.MemoryStats.decodeText(reader);
                    break;
                case "all_start_nanos":
                    message.all_start_nanos = reader.int64();
                    break;
                case "op_start_rel_nanos":
                    message.op_start_rel_nanos = reader.int64();
                    break;
                case "op_end_rel_nanos":
                    message.op_end_rel_nanos = reader.int64();
                    break;
                case "all_end_rel_nanos":
                    message.all_end_rel_nanos = reader.int64();
                    break;
                case "scheduled_nanos":
                    message.scheduled_nanos = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.NodeExecStats.prototype.node_name = "";
tensorflow.NodeExecStats.prototype.all_start_micros = 0n;
tensorflow.NodeExecStats.prototype.op_start_rel_micros = 0n;
tensorflow.NodeExecStats.prototype.op_end_rel_micros = 0n;
tensorflow.NodeExecStats.prototype.all_end_rel_micros = 0n;
tensorflow.NodeExecStats.prototype.timeline_label = "";
tensorflow.NodeExecStats.prototype.scheduled_micros = 0n;
tensorflow.NodeExecStats.prototype.thread_id = 0;
tensorflow.NodeExecStats.prototype.memory_stats = null;
tensorflow.NodeExecStats.prototype.all_start_nanos = 0n;
tensorflow.NodeExecStats.prototype.op_start_rel_nanos = 0n;
tensorflow.NodeExecStats.prototype.op_end_rel_nanos = 0n;
tensorflow.NodeExecStats.prototype.all_end_rel_nanos = 0n;
tensorflow.NodeExecStats.prototype.scheduled_nanos = 0n;

tensorflow.DeviceStepStats = class DeviceStepStats {

    constructor() {
        this.node_stats = [];
        this.thread_names = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.DeviceStepStats();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.device = reader.string();
                    break;
                case 2:
                    message.node_stats.push(tensorflow.NodeExecStats.decode(reader, reader.uint32()));
                    break;
                case 3:
                    reader.entry(message.thread_names, () => reader.uint32(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.DeviceStepStats();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "device":
                    message.device = reader.string();
                    break;
                case "node_stats":
                    message.node_stats.push(tensorflow.NodeExecStats.decodeText(reader));
                    break;
                case "thread_names":
                    reader.entry(message.thread_names, () => reader.uint32(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.DeviceStepStats.prototype.device = "";

tensorflow.StepStats = class StepStats {

    constructor() {
        this.dev_stats = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.StepStats();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dev_stats.push(tensorflow.DeviceStepStats.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.StepStats();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dev_stats":
                    message.dev_stats.push(tensorflow.DeviceStepStats.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AllocationDescription = class AllocationDescription {

    static decode(reader, length) {
        const message = new tensorflow.AllocationDescription();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.requested_bytes = reader.int64();
                    break;
                case 2:
                    message.allocated_bytes = reader.int64();
                    break;
                case 3:
                    message.allocator_name = reader.string();
                    break;
                case 4:
                    message.allocation_id = reader.int64();
                    break;
                case 5:
                    message.has_single_reference = reader.bool();
                    break;
                case 6:
                    message.ptr = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AllocationDescription();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "requested_bytes":
                    message.requested_bytes = reader.int64();
                    break;
                case "allocated_bytes":
                    message.allocated_bytes = reader.int64();
                    break;
                case "allocator_name":
                    message.allocator_name = reader.string();
                    break;
                case "allocation_id":
                    message.allocation_id = reader.int64();
                    break;
                case "has_single_reference":
                    message.has_single_reference = reader.bool();
                    break;
                case "ptr":
                    message.ptr = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AllocationDescription.prototype.requested_bytes = 0n;
tensorflow.AllocationDescription.prototype.allocated_bytes = 0n;
tensorflow.AllocationDescription.prototype.allocator_name = "";
tensorflow.AllocationDescription.prototype.allocation_id = 0n;
tensorflow.AllocationDescription.prototype.has_single_reference = false;
tensorflow.AllocationDescription.prototype.ptr = 0n;

tensorflow.TensorDescription = class TensorDescription {

    static decode(reader, length) {
        const message = new tensorflow.TensorDescription();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.allocation_description = tensorflow.AllocationDescription.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.TensorDescription();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum(tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "allocation_description":
                    message.allocation_description = tensorflow.AllocationDescription.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.TensorDescription.prototype.dtype = 0;
tensorflow.TensorDescription.prototype.shape = null;
tensorflow.TensorDescription.prototype.allocation_description = null;

tensorflow.JobDef = class JobDef {

    constructor() {
        this.tasks = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.JobDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    reader.entry(message.tasks, () => reader.int32(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.JobDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "tasks":
                    reader.entry(message.tasks, () => reader.int32(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.JobDef.prototype.name = "";

tensorflow.ClusterDef = class ClusterDef {

    constructor() {
        this.job = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.ClusterDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.job.push(tensorflow.JobDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ClusterDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "job":
                    message.job.push(tensorflow.JobDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.DebugTensorWatch = class DebugTensorWatch {

    constructor() {
        this.debug_ops = [];
        this.debug_urls = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.DebugTensorWatch();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node_name = reader.string();
                    break;
                case 2:
                    message.output_slot = reader.int32();
                    break;
                case 3:
                    message.debug_ops.push(reader.string());
                    break;
                case 4:
                    message.debug_urls.push(reader.string());
                    break;
                case 5:
                    message.tolerate_debug_op_creation_failures = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.DebugTensorWatch();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_name":
                    message.node_name = reader.string();
                    break;
                case "output_slot":
                    message.output_slot = reader.int32();
                    break;
                case "debug_ops":
                    reader.array(message.debug_ops, () => reader.string());
                    break;
                case "debug_urls":
                    reader.array(message.debug_urls, () => reader.string());
                    break;
                case "tolerate_debug_op_creation_failures":
                    message.tolerate_debug_op_creation_failures = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.DebugTensorWatch.prototype.node_name = "";
tensorflow.DebugTensorWatch.prototype.output_slot = 0;
tensorflow.DebugTensorWatch.prototype.tolerate_debug_op_creation_failures = false;

tensorflow.DebugOptions = class DebugOptions {

    constructor() {
        this.debug_tensor_watch_opts = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.DebugOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 4:
                    message.debug_tensor_watch_opts.push(tensorflow.DebugTensorWatch.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.global_step = reader.int64();
                    break;
                case 11:
                    message.reset_disk_byte_usage = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.DebugOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "debug_tensor_watch_opts":
                    message.debug_tensor_watch_opts.push(tensorflow.DebugTensorWatch.decodeText(reader));
                    break;
                case "global_step":
                    message.global_step = reader.int64();
                    break;
                case "reset_disk_byte_usage":
                    message.reset_disk_byte_usage = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.DebugOptions.prototype.global_step = 0n;
tensorflow.DebugOptions.prototype.reset_disk_byte_usage = false;

tensorflow.DebuggedSourceFile = class DebuggedSourceFile {

    constructor() {
        this.lines = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.DebuggedSourceFile();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.host = reader.string();
                    break;
                case 2:
                    message.file_path = reader.string();
                    break;
                case 3:
                    message.last_modified = reader.int64();
                    break;
                case 4:
                    message.bytes = reader.int64();
                    break;
                case 5:
                    message.lines.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.DebuggedSourceFile();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "host":
                    message.host = reader.string();
                    break;
                case "file_path":
                    message.file_path = reader.string();
                    break;
                case "last_modified":
                    message.last_modified = reader.int64();
                    break;
                case "bytes":
                    message.bytes = reader.int64();
                    break;
                case "lines":
                    reader.array(message.lines, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.DebuggedSourceFile.prototype.host = "";
tensorflow.DebuggedSourceFile.prototype.file_path = "";
tensorflow.DebuggedSourceFile.prototype.last_modified = 0n;
tensorflow.DebuggedSourceFile.prototype.bytes = 0n;

tensorflow.DebuggedSourceFiles = class DebuggedSourceFiles {

    constructor() {
        this.source_files = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.DebuggedSourceFiles();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.source_files.push(tensorflow.DebuggedSourceFile.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.DebuggedSourceFiles();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "source_files":
                    message.source_files.push(tensorflow.DebuggedSourceFile.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AutoParallelOptions = class AutoParallelOptions {

    static decode(reader, length) {
        const message = new tensorflow.AutoParallelOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.enable = reader.bool();
                    break;
                case 2:
                    message.num_replicas = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.AutoParallelOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "enable":
                    message.enable = reader.bool();
                    break;
                case "num_replicas":
                    message.num_replicas = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.AutoParallelOptions.prototype.enable = false;
tensorflow.AutoParallelOptions.prototype.num_replicas = 0;

tensorflow.ScopedAllocatorOptions = class ScopedAllocatorOptions {

    constructor() {
        this.enable_op = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.ScopedAllocatorOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.enable_op.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.ScopedAllocatorOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "enable_op":
                    reader.array(message.enable_op, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RewriterConfig = class RewriterConfig {

    constructor() {
        this.optimizers = [];
        this.custom_optimizers = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.RewriterConfig();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 50:
                    message.cpu_layout_conversion = reader.int32();
                    break;
                case 1:
                    message.layout_optimizer = reader.int32();
                    break;
                case 3:
                    message.constant_folding = reader.int32();
                    break;
                case 13:
                    message.shape_optimization = reader.int32();
                    break;
                case 14:
                    message.remapping = reader.int32();
                    break;
                case 24:
                    message.common_subgraph_elimination = reader.int32();
                    break;
                case 7:
                    message.arithmetic_optimization = reader.int32();
                    break;
                case 8:
                    message.dependency_optimization = reader.int32();
                    break;
                case 9:
                    message.loop_optimization = reader.int32();
                    break;
                case 10:
                    message.function_optimization = reader.int32();
                    break;
                case 11:
                    message.debug_stripper = reader.int32();
                    break;
                case 2:
                    message.disable_model_pruning = reader.bool();
                    break;
                case 15:
                    message.scoped_allocator_optimization = reader.int32();
                    break;
                case 18:
                    message.pin_to_host_optimization = reader.int32();
                    break;
                case 22:
                    message.implementation_selector = reader.int32();
                    break;
                case 23:
                    message.auto_mixed_precision = reader.int32();
                    break;
                case 25:
                    message.auto_mixed_precision_mkl = reader.int32();
                    break;
                case 31:
                    message.auto_mixed_precision_onednn_bfloat16 = reader.int32();
                    break;
                case 29:
                    message.auto_mixed_precision_cpu = reader.int32();
                    break;
                case 19:
                    message.disable_meta_optimizer = reader.bool();
                    break;
                case 32:
                    message.disable_tfg_optimizer = reader.bool();
                    break;
                case 28:
                    message.use_plugin_optimizers = reader.int32();
                    break;
                case 30:
                    message.experimental_conditional_code_motion = reader.int32();
                    break;
                case 12:
                    message.meta_optimizer_iterations = reader.int32();
                    break;
                case 17:
                    message.min_graph_nodes = reader.int32();
                    break;
                case 26:
                    message.experimental_disable_compressed_tensor_optimization = reader.bool();
                    break;
                case 27:
                    message.experimental_disable_folding_quantization_emulation = reader.bool();
                    break;
                case 4:
                    message.memory_optimization = reader.int32();
                    break;
                case 6:
                    message.memory_optimizer_target_node_name_scope = reader.string();
                    break;
                case 20:
                    message.meta_optimizer_timeout_ms = reader.int64();
                    break;
                case 5:
                    message.auto_parallel = tensorflow.AutoParallelOptions.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.fail_on_optimizer_errors = reader.bool();
                    break;
                case 16:
                    message.scoped_allocator_opts = tensorflow.ScopedAllocatorOptions.decode(reader, reader.uint32());
                    break;
                case 100:
                    message.optimizers.push(reader.string());
                    break;
                case 200:
                    message.custom_optimizers.push(tensorflow.RewriterConfig.CustomGraphOptimizer.decode(reader, reader.uint32()));
                    break;
                case 300:
                    message.inter_optimizer_verifier_config = tensorflow.VerifierConfig.decode(reader, reader.uint32());
                    break;
                case 301:
                    message.post_optimization_verifier_config = tensorflow.VerifierConfig.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RewriterConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "cpu_layout_conversion":
                    message.cpu_layout_conversion = reader.enum(tensorflow.RewriterConfig.CpuLayout);
                    break;
                case "layout_optimizer":
                    message.layout_optimizer = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "constant_folding":
                    message.constant_folding = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "shape_optimization":
                    message.shape_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "remapping":
                    message.remapping = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "common_subgraph_elimination":
                    message.common_subgraph_elimination = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "arithmetic_optimization":
                    message.arithmetic_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "dependency_optimization":
                    message.dependency_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "loop_optimization":
                    message.loop_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "function_optimization":
                    message.function_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "debug_stripper":
                    message.debug_stripper = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "disable_model_pruning":
                    message.disable_model_pruning = reader.bool();
                    break;
                case "scoped_allocator_optimization":
                    message.scoped_allocator_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "pin_to_host_optimization":
                    message.pin_to_host_optimization = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "implementation_selector":
                    message.implementation_selector = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "auto_mixed_precision":
                    message.auto_mixed_precision = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "auto_mixed_precision_mkl":
                    message.auto_mixed_precision_mkl = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "auto_mixed_precision_onednn_bfloat16":
                    message.auto_mixed_precision_onednn_bfloat16 = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "auto_mixed_precision_cpu":
                    message.auto_mixed_precision_cpu = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "disable_meta_optimizer":
                    message.disable_meta_optimizer = reader.bool();
                    break;
                case "disable_tfg_optimizer":
                    message.disable_tfg_optimizer = reader.bool();
                    break;
                case "use_plugin_optimizers":
                    message.use_plugin_optimizers = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "experimental_conditional_code_motion":
                    message.experimental_conditional_code_motion = reader.enum(tensorflow.RewriterConfig.Toggle);
                    break;
                case "meta_optimizer_iterations":
                    message.meta_optimizer_iterations = reader.enum(tensorflow.RewriterConfig.NumIterationsType);
                    break;
                case "min_graph_nodes":
                    message.min_graph_nodes = reader.int32();
                    break;
                case "experimental_disable_compressed_tensor_optimization":
                    message.experimental_disable_compressed_tensor_optimization = reader.bool();
                    break;
                case "experimental_disable_folding_quantization_emulation":
                    message.experimental_disable_folding_quantization_emulation = reader.bool();
                    break;
                case "memory_optimization":
                    message.memory_optimization = reader.enum(tensorflow.RewriterConfig.MemOptType);
                    break;
                case "memory_optimizer_target_node_name_scope":
                    message.memory_optimizer_target_node_name_scope = reader.string();
                    break;
                case "meta_optimizer_timeout_ms":
                    message.meta_optimizer_timeout_ms = reader.int64();
                    break;
                case "auto_parallel":
                    message.auto_parallel = tensorflow.AutoParallelOptions.decodeText(reader);
                    break;
                case "fail_on_optimizer_errors":
                    message.fail_on_optimizer_errors = reader.bool();
                    break;
                case "scoped_allocator_opts":
                    message.scoped_allocator_opts = tensorflow.ScopedAllocatorOptions.decodeText(reader);
                    break;
                case "optimizers":
                    reader.array(message.optimizers, () => reader.string());
                    break;
                case "custom_optimizers":
                    message.custom_optimizers.push(tensorflow.RewriterConfig.CustomGraphOptimizer.decodeText(reader));
                    break;
                case "inter_optimizer_verifier_config":
                    message.inter_optimizer_verifier_config = tensorflow.VerifierConfig.decodeText(reader);
                    break;
                case "post_optimization_verifier_config":
                    message.post_optimization_verifier_config = tensorflow.VerifierConfig.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RewriterConfig.prototype.cpu_layout_conversion = 0;
tensorflow.RewriterConfig.prototype.layout_optimizer = 0;
tensorflow.RewriterConfig.prototype.constant_folding = 0;
tensorflow.RewriterConfig.prototype.shape_optimization = 0;
tensorflow.RewriterConfig.prototype.remapping = 0;
tensorflow.RewriterConfig.prototype.common_subgraph_elimination = 0;
tensorflow.RewriterConfig.prototype.arithmetic_optimization = 0;
tensorflow.RewriterConfig.prototype.dependency_optimization = 0;
tensorflow.RewriterConfig.prototype.loop_optimization = 0;
tensorflow.RewriterConfig.prototype.function_optimization = 0;
tensorflow.RewriterConfig.prototype.debug_stripper = 0;
tensorflow.RewriterConfig.prototype.disable_model_pruning = false;
tensorflow.RewriterConfig.prototype.scoped_allocator_optimization = 0;
tensorflow.RewriterConfig.prototype.pin_to_host_optimization = 0;
tensorflow.RewriterConfig.prototype.implementation_selector = 0;
tensorflow.RewriterConfig.prototype.auto_mixed_precision = 0;
tensorflow.RewriterConfig.prototype.auto_mixed_precision_mkl = 0;
tensorflow.RewriterConfig.prototype.auto_mixed_precision_onednn_bfloat16 = 0;
tensorflow.RewriterConfig.prototype.auto_mixed_precision_cpu = 0;
tensorflow.RewriterConfig.prototype.disable_meta_optimizer = false;
tensorflow.RewriterConfig.prototype.disable_tfg_optimizer = false;
tensorflow.RewriterConfig.prototype.use_plugin_optimizers = 0;
tensorflow.RewriterConfig.prototype.experimental_conditional_code_motion = 0;
tensorflow.RewriterConfig.prototype.meta_optimizer_iterations = 0;
tensorflow.RewriterConfig.prototype.min_graph_nodes = 0;
tensorflow.RewriterConfig.prototype.experimental_disable_compressed_tensor_optimization = false;
tensorflow.RewriterConfig.prototype.experimental_disable_folding_quantization_emulation = false;
tensorflow.RewriterConfig.prototype.memory_optimization = 0;
tensorflow.RewriterConfig.prototype.memory_optimizer_target_node_name_scope = "";
tensorflow.RewriterConfig.prototype.meta_optimizer_timeout_ms = 0n;
tensorflow.RewriterConfig.prototype.auto_parallel = null;
tensorflow.RewriterConfig.prototype.fail_on_optimizer_errors = false;
tensorflow.RewriterConfig.prototype.scoped_allocator_opts = null;
tensorflow.RewriterConfig.prototype.inter_optimizer_verifier_config = null;
tensorflow.RewriterConfig.prototype.post_optimization_verifier_config = null;

tensorflow.RewriterConfig.Toggle = {
    "DEFAULT": 0,
    "ON": 1,
    "OFF": 2,
    "AGGRESSIVE": 3,
    "EXPERIMENTAL_MLIR": 4,
    "EXPERIMENTAL_BOTH": 5
};

tensorflow.RewriterConfig.CpuLayout = {
    "NO_CONVERSION_ON_CPU": 0,
    "NCHW_TO_NHWC": 1,
    "NHWC_TO_NCHW": 2
};

tensorflow.RewriterConfig.NumIterationsType = {
    "DEFAULT_NUM_ITERS": 0,
    "ONE": 1,
    "TWO": 2
};

tensorflow.RewriterConfig.MemOptType = {
    "DEFAULT_MEM_OPT": 0,
    "NO_MEM_OPT": 1,
    "MANUAL": 2,
    "SWAPPING_HEURISTICS": 4,
    "RECOMPUTATION_HEURISTICS": 5,
    "SCHEDULING_HEURISTICS": 6,
    "HEURISTICS": 3
};

tensorflow.RewriterConfig.CustomGraphOptimizer = class CustomGraphOptimizer {

    constructor() {
        this.parameter_map = {};
    }

    static decode(reader, length) {
        const message = new tensorflow.RewriterConfig.CustomGraphOptimizer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    reader.entry(message.parameter_map, () => reader.string(), () => tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RewriterConfig.CustomGraphOptimizer();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "parameter_map":
                    reader.entry(message.parameter_map, () => reader.string(), () => tensorflow.AttrValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RewriterConfig.CustomGraphOptimizer.prototype.name = "";

tensorflow.VerifierConfig = class VerifierConfig {

    static decode(reader, length) {
        const message = new tensorflow.VerifierConfig();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.verification_timeout_in_ms = reader.int64();
                    break;
                case 2:
                    message.structure_verifier = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.VerifierConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "verification_timeout_in_ms":
                    message.verification_timeout_in_ms = reader.int64();
                    break;
                case "structure_verifier":
                    message.structure_verifier = reader.enum(tensorflow.VerifierConfig.Toggle);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.VerifierConfig.prototype.verification_timeout_in_ms = 0n;
tensorflow.VerifierConfig.prototype.structure_verifier = 0;

tensorflow.VerifierConfig.Toggle = {
    "DEFAULT": 0,
    "ON": 1,
    "OFF": 2
};

tensorflow.dummy = {};

tensorflow.RPCOptions = class RPCOptions {

    static decode(reader, length) {
        const message = new tensorflow.RPCOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.use_rpc_for_inprocess_master = reader.bool();
                    break;
                case 2:
                    message.compression_algorithm = reader.string();
                    break;
                case 3:
                    message.compression_level = reader.int32();
                    break;
                case 4:
                    message.cache_rpc_response = reader.bool();
                    break;
                case 5:
                    message.disable_session_connection_sharing = reader.bool();
                    break;
                case 6:
                    message.num_channels_per_target = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.RPCOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "use_rpc_for_inprocess_master":
                    message.use_rpc_for_inprocess_master = reader.bool();
                    break;
                case "compression_algorithm":
                    message.compression_algorithm = reader.string();
                    break;
                case "compression_level":
                    message.compression_level = reader.int32();
                    break;
                case "cache_rpc_response":
                    message.cache_rpc_response = reader.bool();
                    break;
                case "disable_session_connection_sharing":
                    message.disable_session_connection_sharing = reader.bool();
                    break;
                case "num_channels_per_target":
                    message.num_channels_per_target = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.RPCOptions.prototype.use_rpc_for_inprocess_master = false;
tensorflow.RPCOptions.prototype.compression_algorithm = "";
tensorflow.RPCOptions.prototype.compression_level = 0;
tensorflow.RPCOptions.prototype.cache_rpc_response = false;
tensorflow.RPCOptions.prototype.disable_session_connection_sharing = false;
tensorflow.RPCOptions.prototype.num_channels_per_target = 0;

tensorflow.CoordinatedJob = class CoordinatedJob {

    static decode(reader, length) {
        const message = new tensorflow.CoordinatedJob();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.num_tasks = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CoordinatedJob();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "num_tasks":
                    message.num_tasks = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CoordinatedJob.prototype.name = "";
tensorflow.CoordinatedJob.prototype.num_tasks = 0;

tensorflow.CoordinationServiceConfig = class CoordinationServiceConfig {

    constructor() {
        this.coordinated_job_list = [];
        this.recoverable_jobs = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.CoordinationServiceConfig();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.service_type = reader.string();
                    break;
                case 2:
                    message.service_leader = reader.string();
                    break;
                case 3:
                    message.enable_health_check = reader.bool();
                    break;
                case 4:
                    message.cluster_register_timeout_in_ms = reader.int64();
                    break;
                case 5:
                    message.heartbeat_timeout_in_ms = reader.int64();
                    break;
                case 10:
                    message.coordinated_job_list.push(tensorflow.CoordinatedJob.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.shutdown_barrier_timeout_in_ms = reader.int64();
                    break;
                case 8:
                    message.agent_destruction_without_shutdown = reader.bool();
                    break;
                case 9:
                    message.recoverable_jobs.push(reader.string());
                    break;
                case 11:
                    message.allow_new_incarnation_to_reconnect = reader.bool();
                    break;
                case 12:
                    message.force_disable = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.CoordinationServiceConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "service_type":
                    message.service_type = reader.string();
                    break;
                case "service_leader":
                    message.service_leader = reader.string();
                    break;
                case "enable_health_check":
                    message.enable_health_check = reader.bool();
                    break;
                case "cluster_register_timeout_in_ms":
                    message.cluster_register_timeout_in_ms = reader.int64();
                    break;
                case "heartbeat_timeout_in_ms":
                    message.heartbeat_timeout_in_ms = reader.int64();
                    break;
                case "coordinated_job_list":
                    message.coordinated_job_list.push(tensorflow.CoordinatedJob.decodeText(reader));
                    break;
                case "shutdown_barrier_timeout_in_ms":
                    message.shutdown_barrier_timeout_in_ms = reader.int64();
                    break;
                case "agent_destruction_without_shutdown":
                    message.agent_destruction_without_shutdown = reader.bool();
                    break;
                case "recoverable_jobs":
                    reader.array(message.recoverable_jobs, () => reader.string());
                    break;
                case "allow_new_incarnation_to_reconnect":
                    message.allow_new_incarnation_to_reconnect = reader.bool();
                    break;
                case "force_disable":
                    message.force_disable = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.CoordinationServiceConfig.prototype.service_type = "";
tensorflow.CoordinationServiceConfig.prototype.service_leader = "";
tensorflow.CoordinationServiceConfig.prototype.enable_health_check = false;
tensorflow.CoordinationServiceConfig.prototype.cluster_register_timeout_in_ms = 0n;
tensorflow.CoordinationServiceConfig.prototype.heartbeat_timeout_in_ms = 0n;
tensorflow.CoordinationServiceConfig.prototype.shutdown_barrier_timeout_in_ms = 0n;
tensorflow.CoordinationServiceConfig.prototype.agent_destruction_without_shutdown = false;
tensorflow.CoordinationServiceConfig.prototype.allow_new_incarnation_to_reconnect = false;
tensorflow.CoordinationServiceConfig.prototype.force_disable = false;

tensorflow.MemmappedFileSystemDirectoryElement = class MemmappedFileSystemDirectoryElement {

    static decode(reader, length) {
        const message = new tensorflow.MemmappedFileSystemDirectoryElement();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.offset = reader.uint64();
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 3:
                    message.length = reader.uint64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.MemmappedFileSystemDirectoryElement();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "offset":
                    message.offset = reader.uint64();
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "length":
                    message.length = reader.uint64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.MemmappedFileSystemDirectoryElement.prototype.offset = 0n;
tensorflow.MemmappedFileSystemDirectoryElement.prototype.name = "";
tensorflow.MemmappedFileSystemDirectoryElement.prototype.length = 0n;

tensorflow.MemmappedFileSystemDirectory = class MemmappedFileSystemDirectory {

    constructor() {
        this.element = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.MemmappedFileSystemDirectory();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.element.push(tensorflow.MemmappedFileSystemDirectoryElement.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.MemmappedFileSystemDirectory();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "element":
                    message.element.push(tensorflow.MemmappedFileSystemDirectoryElement.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FingerprintDef = class FingerprintDef {

    static decode(reader, length) {
        const message = new tensorflow.FingerprintDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.saved_model_checksum = reader.uint64();
                    break;
                case 2:
                    message.graph_def_program_hash = reader.uint64();
                    break;
                case 3:
                    message.signature_def_hash = reader.uint64();
                    break;
                case 4:
                    message.saved_object_graph_hash = reader.uint64();
                    break;
                case 5:
                    message.checkpoint_hash = reader.uint64();
                    break;
                case 6:
                    message.version = tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.FingerprintDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "saved_model_checksum":
                    message.saved_model_checksum = reader.uint64();
                    break;
                case "graph_def_program_hash":
                    message.graph_def_program_hash = reader.uint64();
                    break;
                case "signature_def_hash":
                    message.signature_def_hash = reader.uint64();
                    break;
                case "saved_object_graph_hash":
                    message.saved_object_graph_hash = reader.uint64();
                    break;
                case "checkpoint_hash":
                    message.checkpoint_hash = reader.uint64();
                    break;
                case "version":
                    message.version = tensorflow.VersionDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

tensorflow.FingerprintDef.prototype.saved_model_checksum = 0n;
tensorflow.FingerprintDef.prototype.graph_def_program_hash = 0n;
tensorflow.FingerprintDef.prototype.signature_def_hash = 0n;
tensorflow.FingerprintDef.prototype.saved_object_graph_hash = 0n;
tensorflow.FingerprintDef.prototype.checkpoint_hash = 0n;
tensorflow.FingerprintDef.prototype.version = null;

google.protobuf = {};

google.protobuf.Any = class Any {

    static decode(reader, length) {
        const message = new google.protobuf.Any();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_url = reader.string();
                    break;
                case 2:
                    message.value = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        return reader.any(() => new google.protobuf.Any());
    }
};

google.protobuf.Any.prototype.type_url = "";
google.protobuf.Any.prototype.value = new Uint8Array([]);

google.protobuf.BoolValue = class BoolValue {

    static decode(reader, length) {
        const message = new google.protobuf.BoolValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new google.protobuf.BoolValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

google.protobuf.BoolValue.prototype.value = false;
