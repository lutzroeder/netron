var $root = protobuf.get('tf');

$root.tensorflow = {};

$root.tensorflow.SavedModel = class SavedModel {

    constructor() {
        this.meta_graphs = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedModel();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.saved_model_schema_version = reader.int64();
                    break;
                case 2:
                    message.meta_graphs.push($root.tensorflow.MetaGraphDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedModel();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "saved_model_schema_version":
                    message.saved_model_schema_version = reader.integer();
                    break;
                case "meta_graphs":
                    message.meta_graphs.push($root.tensorflow.MetaGraphDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedModel.prototype.saved_model_schema_version = protobuf.Int64.create(0);

$root.tensorflow.MetaGraphDef = class MetaGraphDef {

    constructor() {
        this.collection_def = {};
        this.signature_def = {};
        this.asset_file_def = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.MetaGraphDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meta_info_def = $root.tensorflow.MetaGraphDef.MetaInfoDef.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.graph_def = $root.tensorflow.GraphDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.saver_def = $root.tensorflow.SaverDef.decode(reader, reader.uint32());
                    break;
                case 4:
                    reader.entry(message.collection_def, () => reader.string(), () => $root.tensorflow.CollectionDef.decode(reader, reader.uint32()));
                    break;
                case 5:
                    reader.entry(message.signature_def, () => reader.string(), () => $root.tensorflow.SignatureDef.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.asset_file_def.push($root.tensorflow.AssetFileDef.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.object_graph_def = $root.tensorflow.SavedObjectGraph.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.MetaGraphDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meta_info_def":
                    message.meta_info_def = $root.tensorflow.MetaGraphDef.MetaInfoDef.decodeText(reader);
                    break;
                case "graph_def":
                    message.graph_def = $root.tensorflow.GraphDef.decodeText(reader);
                    break;
                case "saver_def":
                    message.saver_def = $root.tensorflow.SaverDef.decodeText(reader);
                    break;
                case "collection_def":
                    reader.entry(message.collection_def, () => reader.string(), () => $root.tensorflow.CollectionDef.decodeText(reader));
                    break;
                case "signature_def":
                    reader.entry(message.signature_def, () => reader.string(), () => $root.tensorflow.SignatureDef.decodeText(reader));
                    break;
                case "asset_file_def":
                    message.asset_file_def.push($root.tensorflow.AssetFileDef.decodeText(reader));
                    break;
                case "object_graph_def":
                    message.object_graph_def = $root.tensorflow.SavedObjectGraph.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.MetaGraphDef.prototype.meta_info_def = null;
$root.tensorflow.MetaGraphDef.prototype.graph_def = null;
$root.tensorflow.MetaGraphDef.prototype.saver_def = null;
$root.tensorflow.MetaGraphDef.prototype.object_graph_def = null;

$root.tensorflow.MetaGraphDef.MetaInfoDef = class MetaInfoDef {

    constructor() {
        this.tags = [];
        this.function_aliases = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meta_graph_version = reader.string();
                    break;
                case 2:
                    message.stripped_op_list = $root.tensorflow.OpList.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.any_info = $root.google.protobuf.Any.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meta_graph_version":
                    message.meta_graph_version = reader.string();
                    break;
                case "stripped_op_list":
                    message.stripped_op_list = $root.tensorflow.OpList.decodeText(reader);
                    break;
                case "any_info":
                    message.any_info = $root.google.protobuf.Any.decodeText(reader);
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
                    message.stripped_default_attrs = reader.boolean();
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

$root.tensorflow.MetaGraphDef.MetaInfoDef.prototype.meta_graph_version = "";
$root.tensorflow.MetaGraphDef.MetaInfoDef.prototype.stripped_op_list = null;
$root.tensorflow.MetaGraphDef.MetaInfoDef.prototype.any_info = null;
$root.tensorflow.MetaGraphDef.MetaInfoDef.prototype.tensorflow_version = "";
$root.tensorflow.MetaGraphDef.MetaInfoDef.prototype.tensorflow_git_version = "";
$root.tensorflow.MetaGraphDef.MetaInfoDef.prototype.stripped_default_attrs = false;

$root.tensorflow.CollectionDef = class CollectionDef {

    constructor() {
    }

    get kind() {
        $root.tensorflow.CollectionDef.kindSet = $root.tensorflow.CollectionDef.kindSet || new Set([ "node_list", "bytes_list", "int64_list", "float_list", "any_list"]);
        return Object.keys(this).find((key) => $root.tensorflow.CollectionDef.kindSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.CollectionDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node_list = $root.tensorflow.CollectionDef.NodeList.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.bytes_list = $root.tensorflow.CollectionDef.BytesList.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.int64_list = $root.tensorflow.CollectionDef.Int64List.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.float_list = $root.tensorflow.CollectionDef.FloatList.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.any_list = $root.tensorflow.CollectionDef.AnyList.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.CollectionDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_list":
                    message.node_list = $root.tensorflow.CollectionDef.NodeList.decodeText(reader);
                    break;
                case "bytes_list":
                    message.bytes_list = $root.tensorflow.CollectionDef.BytesList.decodeText(reader);
                    break;
                case "int64_list":
                    message.int64_list = $root.tensorflow.CollectionDef.Int64List.decodeText(reader);
                    break;
                case "float_list":
                    message.float_list = $root.tensorflow.CollectionDef.FloatList.decodeText(reader);
                    break;
                case "any_list":
                    message.any_list = $root.tensorflow.CollectionDef.AnyList.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.CollectionDef.NodeList = class NodeList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.CollectionDef.NodeList();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.CollectionDef.NodeList();
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

$root.tensorflow.CollectionDef.BytesList = class BytesList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.CollectionDef.BytesList();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.CollectionDef.BytesList();
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

$root.tensorflow.CollectionDef.Int64List = class Int64List {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.CollectionDef.Int64List();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.CollectionDef.Int64List();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.integer());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.CollectionDef.FloatList = class FloatList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.CollectionDef.FloatList();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.CollectionDef.FloatList();
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

$root.tensorflow.CollectionDef.AnyList = class AnyList {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.CollectionDef.AnyList();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push($root.google.protobuf.Any.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.CollectionDef.AnyList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.anyarray(message.value, () => new $root.google.protobuf.Any());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorInfo = class TensorInfo {

    constructor() {
    }

    get encoding() {
        $root.tensorflow.TensorInfo.encodingSet = $root.tensorflow.TensorInfo.encodingSet || new Set([ "name", "coo_sparse", "composite_tensor"]);
        return Object.keys(this).find((key) => $root.tensorflow.TensorInfo.encodingSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorInfo();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 4:
                    message.coo_sparse = $root.tensorflow.TensorInfo.CooSparse.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.composite_tensor = $root.tensorflow.TensorInfo.CompositeTensor.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.dtype = reader.int32();
                    break;
                case 3:
                    message.tensor_shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TensorInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "coo_sparse":
                    message.coo_sparse = $root.tensorflow.TensorInfo.CooSparse.decodeText(reader);
                    break;
                case "composite_tensor":
                    message.composite_tensor = $root.tensorflow.TensorInfo.CompositeTensor.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                case "tensor_shape":
                    message.tensor_shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorInfo.prototype.dtype = 0;
$root.tensorflow.TensorInfo.prototype.tensor_shape = null;

$root.tensorflow.TensorInfo.CooSparse = class CooSparse {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorInfo.CooSparse();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.TensorInfo.CooSparse();
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

$root.tensorflow.TensorInfo.CooSparse.prototype.values_tensor_name = "";
$root.tensorflow.TensorInfo.CooSparse.prototype.indices_tensor_name = "";
$root.tensorflow.TensorInfo.CooSparse.prototype.dense_shape_tensor_name = "";

$root.tensorflow.TensorInfo.CompositeTensor = class CompositeTensor {

    constructor() {
        this.components = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorInfo.CompositeTensor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_spec = $root.tensorflow.TypeSpecProto.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.components.push($root.tensorflow.TensorInfo.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TensorInfo.CompositeTensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_spec":
                    message.type_spec = $root.tensorflow.TypeSpecProto.decodeText(reader);
                    break;
                case "components":
                    message.components.push($root.tensorflow.TensorInfo.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorInfo.CompositeTensor.prototype.type_spec = null;

$root.tensorflow.SignatureDef = class SignatureDef {

    constructor() {
        this.inputs = {};
        this.outputs = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SignatureDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.inputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decode(reader, reader.uint32()));
                    break;
                case 2:
                    reader.entry(message.outputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.method_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SignatureDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "inputs":
                    reader.entry(message.inputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decodeText(reader));
                    break;
                case "outputs":
                    reader.entry(message.outputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decodeText(reader));
                    break;
                case "method_name":
                    message.method_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SignatureDef.prototype.method_name = "";

$root.tensorflow.AssetFileDef = class AssetFileDef {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.AssetFileDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_info = $root.tensorflow.TensorInfo.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.AssetFileDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_info":
                    message.tensor_info = $root.tensorflow.TensorInfo.decodeText(reader);
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

$root.tensorflow.AssetFileDef.prototype.tensor_info = null;
$root.tensorflow.AssetFileDef.prototype.filename = "";

$root.tensorflow.GraphDef = class GraphDef {

    constructor() {
        this.node = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.GraphDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push($root.tensorflow.NodeDef.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.versions = $root.tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.version = reader.int32();
                    break;
                case 2:
                    message.library = $root.tensorflow.FunctionDefLibrary.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.GraphDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node":
                    message.node.push($root.tensorflow.NodeDef.decodeText(reader));
                    break;
                case "versions":
                    message.versions = $root.tensorflow.VersionDef.decodeText(reader);
                    break;
                case "version":
                    message.version = reader.integer();
                    break;
                case "library":
                    message.library = $root.tensorflow.FunctionDefLibrary.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.GraphDef.prototype.versions = null;
$root.tensorflow.GraphDef.prototype.version = 0;
$root.tensorflow.GraphDef.prototype.library = null;

$root.tensorflow.FunctionDefLibrary = class FunctionDefLibrary {

    constructor() {
        this["function"] = [];
        this.gradient = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.FunctionDefLibrary();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message["function"].push($root.tensorflow.FunctionDef.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.gradient.push($root.tensorflow.GradientDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.FunctionDefLibrary();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "function":
                    message["function"].push($root.tensorflow.FunctionDef.decodeText(reader));
                    break;
                case "gradient":
                    message.gradient.push($root.tensorflow.GradientDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.FunctionDef = class FunctionDef {

    constructor() {
        this.attr = {};
        this.arg_attr = {};
        this.resource_arg_unique_id = {};
        this.node_def = [];
        this.ret = {};
        this.control_ret = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.FunctionDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.signature = $root.tensorflow.OpDef.decode(reader, reader.uint32());
                    break;
                case 5:
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 7:
                    reader.entry(message.arg_attr, () => reader.uint32(), () => $root.tensorflow.FunctionDef.ArgAttrs.decode(reader, reader.uint32()));
                    break;
                case 8:
                    reader.entry(message.resource_arg_unique_id, () => reader.uint32(), () => reader.uint32());
                    break;
                case 3:
                    message.node_def.push($root.tensorflow.NodeDef.decode(reader, reader.uint32()));
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
        const message = new $root.tensorflow.FunctionDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "signature":
                    message.signature = $root.tensorflow.OpDef.decodeText(reader);
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader));
                    break;
                case "arg_attr":
                    reader.entry(message.arg_attr, () => reader.integer(), () => $root.tensorflow.FunctionDef.ArgAttrs.decodeText(reader));
                    break;
                case "resource_arg_unique_id":
                    reader.entry(message.resource_arg_unique_id, () => reader.integer(), () => reader.uint32());
                    break;
                case "node_def":
                    message.node_def.push($root.tensorflow.NodeDef.decodeText(reader));
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

$root.tensorflow.FunctionDef.prototype.signature = null;

$root.tensorflow.FunctionDef.ArgAttrs = class ArgAttrs {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.FunctionDef.ArgAttrs();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.FunctionDef.ArgAttrs();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.GradientDef = class GradientDef {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.GradientDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.GradientDef();
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

$root.tensorflow.GradientDef.prototype.function_name = "";
$root.tensorflow.GradientDef.prototype.gradient_func = "";

$root.tensorflow.AttrValue = class AttrValue {

    constructor() {
    }

    get value() {
        $root.tensorflow.AttrValue.valueSet = $root.tensorflow.AttrValue.valueSet || new Set([ "s", "i", "f", "b", "type", "shape", "tensor", "list", "func", "placeholder"]);
        return Object.keys(this).find((key) => $root.tensorflow.AttrValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.AttrValue();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.tensor = $root.tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                case 1:
                    message.list = $root.tensorflow.AttrValue.ListValue.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.func = $root.tensorflow.NameAttrList.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.AttrValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    message.s = reader.bytes();
                    break;
                case "i":
                    message.i = reader.integer();
                    break;
                case "f":
                    message.f = reader.float();
                    break;
                case "b":
                    message.b = reader.boolean();
                    break;
                case "type":
                    message.type = reader.enum($root.tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "tensor":
                    message.tensor = $root.tensorflow.TensorProto.decodeText(reader);
                    break;
                case "list":
                    message.list = $root.tensorflow.AttrValue.ListValue.decodeText(reader);
                    break;
                case "func":
                    message.func = $root.tensorflow.NameAttrList.decodeText(reader);
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

$root.tensorflow.AttrValue.ListValue = class ListValue {

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
        const message = new $root.tensorflow.AttrValue.ListValue();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.shape.push($root.tensorflow.TensorShapeProto.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.tensor.push($root.tensorflow.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.func.push($root.tensorflow.NameAttrList.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.AttrValue.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    reader.array(message.s, () => reader.bytes());
                    break;
                case "i":
                    reader.array(message.i, () => reader.integer());
                    break;
                case "f":
                    reader.array(message.f, () => reader.float());
                    break;
                case "b":
                    reader.array(message.b, () => reader.boolean());
                    break;
                case "type":
                    reader.array(message.type, () => reader.enum($root.tensorflow.DataType));
                    break;
                case "shape":
                    message.shape.push($root.tensorflow.TensorShapeProto.decodeText(reader));
                    break;
                case "tensor":
                    message.tensor.push($root.tensorflow.TensorProto.decodeText(reader));
                    break;
                case "func":
                    message.func.push($root.tensorflow.NameAttrList.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.NameAttrList = class NameAttrList {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.NameAttrList();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.NameAttrList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.NameAttrList.prototype.name = "";

$root.tensorflow.TensorProto = class TensorProto {

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
        const message = new $root.tensorflow.TensorProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.tensor_shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
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
                    message.resource_handle_val.push($root.tensorflow.ResourceHandleProto.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.variant_val.push($root.tensorflow.VariantTensorDataProto.decode(reader, reader.uint32()));
                    break;
                case 16:
                    message.uint32_val = reader.array(message.uint32_val, () => reader.uint32(), tag);
                    break;
                case 17:
                    message.uint64_val = reader.array(message.uint64_val, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                case "tensor_shape":
                    message.tensor_shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "version_number":
                    message.version_number = reader.integer();
                    break;
                case "tensor_content":
                    message.tensor_content = reader.bytes();
                    break;
                case "half_val":
                    reader.array(message.half_val, () => reader.integer());
                    break;
                case "float_val":
                    reader.array(message.float_val, () => reader.float());
                    break;
                case "double_val":
                    reader.array(message.double_val, () => reader.float());
                    break;
                case "int_val":
                    reader.array(message.int_val, () => reader.integer());
                    break;
                case "string_val":
                    reader.array(message.string_val, () => reader.bytes());
                    break;
                case "scomplex_val":
                    reader.array(message.scomplex_val, () => reader.float());
                    break;
                case "int64_val":
                    reader.array(message.int64_val, () => reader.integer());
                    break;
                case "bool_val":
                    reader.array(message.bool_val, () => reader.boolean());
                    break;
                case "dcomplex_val":
                    reader.array(message.dcomplex_val, () => reader.float());
                    break;
                case "resource_handle_val":
                    message.resource_handle_val.push($root.tensorflow.ResourceHandleProto.decodeText(reader));
                    break;
                case "variant_val":
                    message.variant_val.push($root.tensorflow.VariantTensorDataProto.decodeText(reader));
                    break;
                case "uint32_val":
                    reader.array(message.uint32_val, () => reader.integer());
                    break;
                case "uint64_val":
                    reader.array(message.uint64_val, () => reader.integer());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorProto.prototype.dtype = 0;
$root.tensorflow.TensorProto.prototype.tensor_shape = null;
$root.tensorflow.TensorProto.prototype.version_number = 0;
$root.tensorflow.TensorProto.prototype.tensor_content = new Uint8Array([]);

$root.tensorflow.VariantTensorDataProto = class VariantTensorDataProto {

    constructor() {
        this.tensors = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.VariantTensorDataProto();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.tensors.push($root.tensorflow.TensorProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.VariantTensorDataProto();
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
                    message.tensors.push($root.tensorflow.TensorProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.VariantTensorDataProto.prototype.type_name = "";
$root.tensorflow.VariantTensorDataProto.prototype.metadata = new Uint8Array([]);

$root.tensorflow.ResourceHandleProto = class ResourceHandleProto {

    constructor() {
        this.dtypes_and_shapes = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.ResourceHandleProto();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.dtypes_and_shapes.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.ResourceHandleProto();
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
                    message.hash_code = reader.integer();
                    break;
                case "maybe_type_name":
                    message.maybe_type_name = reader.string();
                    break;
                case "dtypes_and_shapes":
                    message.dtypes_and_shapes.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.ResourceHandleProto.prototype.device = "";
$root.tensorflow.ResourceHandleProto.prototype.container = "";
$root.tensorflow.ResourceHandleProto.prototype.name = "";
$root.tensorflow.ResourceHandleProto.prototype.hash_code = protobuf.Uint64.create(0);
$root.tensorflow.ResourceHandleProto.prototype.maybe_type_name = "";

$root.tensorflow.ResourceHandleProto.DtypeAndShape = class DtypeAndShape {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.ResourceHandleProto.DtypeAndShape();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.ResourceHandleProto.DtypeAndShape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.ResourceHandleProto.DtypeAndShape.prototype.dtype = 0;
$root.tensorflow.ResourceHandleProto.DtypeAndShape.prototype.shape = null;

$root.tensorflow.TensorShapeProto = class TensorShapeProto {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorShapeProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.dim.push($root.tensorflow.TensorShapeProto.Dim.decode(reader, reader.uint32()));
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
        const message = new $root.tensorflow.TensorShapeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim.push($root.tensorflow.TensorShapeProto.Dim.decodeText(reader));
                    break;
                case "unknown_rank":
                    message.unknown_rank = reader.boolean();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorShapeProto.prototype.unknown_rank = false;

$root.tensorflow.TensorShapeProto.Dim = class Dim {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorShapeProto.Dim();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.TensorShapeProto.Dim();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "size":
                    message.size = reader.integer();
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

$root.tensorflow.TensorShapeProto.Dim.prototype.size = protobuf.Int64.create(0);
$root.tensorflow.TensorShapeProto.Dim.prototype.name = "";

$root.tensorflow.DataType = {
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
    "DT_UINT64_REF": 123
};

$root.tensorflow.SpecializedType = {
    "ST_INVALID": 0,
    "ST_TENSOR_LIST": 1,
    "ST_OPTIONAL": 2
};

$root.tensorflow.NodeDef = class NodeDef {

    constructor() {
        this.input = [];
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.NodeDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.experimental_debug_info = $root.tensorflow.NodeDef.ExperimentalDebugInfo.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.NodeDef();
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
                    reader.entry(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader));
                    break;
                case "experimental_debug_info":
                    message.experimental_debug_info = $root.tensorflow.NodeDef.ExperimentalDebugInfo.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.NodeDef.prototype.name = "";
$root.tensorflow.NodeDef.prototype.op = "";
$root.tensorflow.NodeDef.prototype.device = "";
$root.tensorflow.NodeDef.prototype.experimental_debug_info = null;

$root.tensorflow.NodeDef.ExperimentalDebugInfo = class ExperimentalDebugInfo {

    constructor() {
        this.original_node_names = [];
        this.original_func_names = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.NodeDef.ExperimentalDebugInfo();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.NodeDef.ExperimentalDebugInfo();
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

$root.tensorflow.OpDef = class OpDef {

    constructor() {
        this.input_arg = [];
        this.output_arg = [];
        this.control_output = [];
        this.attr = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.OpDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.input_arg.push($root.tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.output_arg.push($root.tensorflow.OpDef.ArgDef.decode(reader, reader.uint32()));
                    break;
                case 20:
                    message.control_output.push(reader.string());
                    break;
                case 4:
                    message.attr.push($root.tensorflow.OpDef.AttrDef.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.deprecation = $root.tensorflow.OpDeprecation.decode(reader, reader.uint32());
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
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.OpDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input_arg":
                    message.input_arg.push($root.tensorflow.OpDef.ArgDef.decodeText(reader));
                    break;
                case "output_arg":
                    message.output_arg.push($root.tensorflow.OpDef.ArgDef.decodeText(reader));
                    break;
                case "control_output":
                    reader.array(message.control_output, () => reader.string());
                    break;
                case "attr":
                    message.attr.push($root.tensorflow.OpDef.AttrDef.decodeText(reader));
                    break;
                case "deprecation":
                    message.deprecation = $root.tensorflow.OpDeprecation.decodeText(reader);
                    break;
                case "summary":
                    message.summary = reader.string();
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                case "is_commutative":
                    message.is_commutative = reader.boolean();
                    break;
                case "is_aggregate":
                    message.is_aggregate = reader.boolean();
                    break;
                case "is_stateful":
                    message.is_stateful = reader.boolean();
                    break;
                case "allows_uninitialized_input":
                    message.allows_uninitialized_input = reader.boolean();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.OpDef.prototype.name = "";
$root.tensorflow.OpDef.prototype.deprecation = null;
$root.tensorflow.OpDef.prototype.summary = "";
$root.tensorflow.OpDef.prototype.description = "";
$root.tensorflow.OpDef.prototype.is_commutative = false;
$root.tensorflow.OpDef.prototype.is_aggregate = false;
$root.tensorflow.OpDef.prototype.is_stateful = false;
$root.tensorflow.OpDef.prototype.allows_uninitialized_input = false;

$root.tensorflow.OpDef.ArgDef = class ArgDef {

    constructor() {
        this.handle_data = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.OpDef.ArgDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.handle_data.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decode(reader, reader.uint32()));
                    break;
                case 16:
                    message.is_ref = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.OpDef.ArgDef();
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
                    message.type = reader.enum($root.tensorflow.DataType);
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
                    message.handle_data.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decodeText(reader));
                    break;
                case "is_ref":
                    message.is_ref = reader.boolean();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.OpDef.ArgDef.prototype.name = "";
$root.tensorflow.OpDef.ArgDef.prototype.description = "";
$root.tensorflow.OpDef.ArgDef.prototype.type = 0;
$root.tensorflow.OpDef.ArgDef.prototype.type_attr = "";
$root.tensorflow.OpDef.ArgDef.prototype.number_attr = "";
$root.tensorflow.OpDef.ArgDef.prototype.type_list_attr = "";
$root.tensorflow.OpDef.ArgDef.prototype.is_ref = false;

$root.tensorflow.OpDef.AttrDef = class AttrDef {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.OpDef.AttrDef();
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
                    message.default_value = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
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
                    message.allowed_values = $root.tensorflow.AttrValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.OpDef.AttrDef();
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
                    message.default_value = $root.tensorflow.AttrValue.decodeText(reader);
                    break;
                case "description":
                    message.description = reader.string();
                    break;
                case "has_minimum":
                    message.has_minimum = reader.boolean();
                    break;
                case "minimum":
                    message.minimum = reader.integer();
                    break;
                case "allowed_values":
                    message.allowed_values = $root.tensorflow.AttrValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.OpDef.AttrDef.prototype.name = "";
$root.tensorflow.OpDef.AttrDef.prototype.type = "";
$root.tensorflow.OpDef.AttrDef.prototype.default_value = null;
$root.tensorflow.OpDef.AttrDef.prototype.description = "";
$root.tensorflow.OpDef.AttrDef.prototype.has_minimum = false;
$root.tensorflow.OpDef.AttrDef.prototype.minimum = protobuf.Int64.create(0);
$root.tensorflow.OpDef.AttrDef.prototype.allowed_values = null;

$root.tensorflow.OpDeprecation = class OpDeprecation {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.OpDeprecation();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.OpDeprecation();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.integer();
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

$root.tensorflow.OpDeprecation.prototype.version = 0;
$root.tensorflow.OpDeprecation.prototype.explanation = "";

$root.tensorflow.OpList = class OpList {

    constructor() {
        this.op = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.OpList();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op.push($root.tensorflow.OpDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.OpList();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op":
                    message.op.push($root.tensorflow.OpDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.VersionDef = class VersionDef {

    constructor() {
        this.bad_consumers = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.VersionDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.VersionDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "producer":
                    message.producer = reader.integer();
                    break;
                case "min_consumer":
                    message.min_consumer = reader.integer();
                    break;
                case "bad_consumers":
                    reader.array(message.bad_consumers, () => reader.integer());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.VersionDef.prototype.producer = 0;
$root.tensorflow.VersionDef.prototype.min_consumer = 0;

$root.tensorflow.SavedObjectGraph = class SavedObjectGraph {

    constructor() {
        this.nodes = [];
        this.concrete_functions = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedObjectGraph();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push($root.tensorflow.SavedObject.decode(reader, reader.uint32()));
                    break;
                case 2:
                    reader.entry(message.concrete_functions, () => reader.string(), () => $root.tensorflow.SavedConcreteFunction.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedObjectGraph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push($root.tensorflow.SavedObject.decodeText(reader));
                    break;
                case "concrete_functions":
                    reader.entry(message.concrete_functions, () => reader.string(), () => $root.tensorflow.SavedConcreteFunction.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedObject = class SavedObject {

    constructor() {
        this.children = [];
        this.slot_variables = [];
        this.saveable_objects = {};
    }

    get kind() {
        $root.tensorflow.SavedObject.kindSet = $root.tensorflow.SavedObject.kindSet || new Set([ "user_object", "asset", "function", "variable", "bare_concrete_function", "constant", "resource"]);
        return Object.keys(this).find((key) => $root.tensorflow.SavedObject.kindSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedObject();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.user_object = $root.tensorflow.SavedUserObject.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.asset = $root.tensorflow.SavedAsset.decode(reader, reader.uint32());
                    break;
                case 6:
                    message["function"] = $root.tensorflow.SavedFunction.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.variable = $root.tensorflow.SavedVariable.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.bare_concrete_function = $root.tensorflow.SavedBareConcreteFunction.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.constant = $root.tensorflow.SavedConstant.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.resource = $root.tensorflow.SavedResource.decode(reader, reader.uint32());
                    break;
                case 11:
                    reader.entry(message.saveable_objects, () => reader.string(), () => $root.tensorflow.SaveableObject.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "children":
                    message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader));
                    break;
                case "slot_variables":
                    message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeText(reader));
                    break;
                case "user_object":
                    message.user_object = $root.tensorflow.SavedUserObject.decodeText(reader);
                    break;
                case "asset":
                    message.asset = $root.tensorflow.SavedAsset.decodeText(reader);
                    break;
                case "function":
                    message["function"] = $root.tensorflow.SavedFunction.decodeText(reader);
                    break;
                case "variable":
                    message.variable = $root.tensorflow.SavedVariable.decodeText(reader);
                    break;
                case "bare_concrete_function":
                    message.bare_concrete_function = $root.tensorflow.SavedBareConcreteFunction.decodeText(reader);
                    break;
                case "constant":
                    message.constant = $root.tensorflow.SavedConstant.decodeText(reader);
                    break;
                case "resource":
                    message.resource = $root.tensorflow.SavedResource.decodeText(reader);
                    break;
                case "saveable_objects":
                    reader.entry(message.saveable_objects, () => reader.string(), () => $root.tensorflow.SaveableObject.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedUserObject = class SavedUserObject {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedUserObject();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.identifier = reader.string();
                    break;
                case 2:
                    message.version = $root.tensorflow.VersionDef.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.SavedUserObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "identifier":
                    message.identifier = reader.string();
                    break;
                case "version":
                    message.version = $root.tensorflow.VersionDef.decodeText(reader);
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

$root.tensorflow.SavedUserObject.prototype.identifier = "";
$root.tensorflow.SavedUserObject.prototype.version = null;
$root.tensorflow.SavedUserObject.prototype.metadata = "";

$root.tensorflow.SavedAsset = class SavedAsset {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedAsset();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SavedAsset();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "asset_file_def_index":
                    message.asset_file_def_index = reader.integer();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedAsset.prototype.asset_file_def_index = 0;

$root.tensorflow.SavedFunction = class SavedFunction {

    constructor() {
        this.concrete_functions = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedFunction();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.concrete_functions.push(reader.string());
                    break;
                case 2:
                    message.function_spec = $root.tensorflow.FunctionSpec.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedFunction();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "concrete_functions":
                    reader.array(message.concrete_functions, () => reader.string());
                    break;
                case "function_spec":
                    message.function_spec = $root.tensorflow.FunctionSpec.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedFunction.prototype.function_spec = null;

$root.tensorflow.SavedConcreteFunction = class SavedConcreteFunction {

    constructor() {
        this.bound_inputs = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedConcreteFunction();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.bound_inputs = reader.array(message.bound_inputs, () => reader.int32(), tag);
                    break;
                case 3:
                    message.canonicalized_input_signature = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.output_signature = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedConcreteFunction();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "bound_inputs":
                    reader.array(message.bound_inputs, () => reader.integer());
                    break;
                case "canonicalized_input_signature":
                    message.canonicalized_input_signature = $root.tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "output_signature":
                    message.output_signature = $root.tensorflow.StructuredValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedConcreteFunction.prototype.canonicalized_input_signature = null;
$root.tensorflow.SavedConcreteFunction.prototype.output_signature = null;

$root.tensorflow.SavedBareConcreteFunction = class SavedBareConcreteFunction {

    constructor() {
        this.argument_keywords = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedBareConcreteFunction();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.function_spec = $root.tensorflow.FunctionSpec.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedBareConcreteFunction();
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
                    message.allowed_positional_arguments = reader.integer();
                    break;
                case "function_spec":
                    message.function_spec = $root.tensorflow.FunctionSpec.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedBareConcreteFunction.prototype.concrete_function_name = "";
$root.tensorflow.SavedBareConcreteFunction.prototype.allowed_positional_arguments = protobuf.Int64.create(0);
$root.tensorflow.SavedBareConcreteFunction.prototype.function_spec = null;

$root.tensorflow.SavedConstant = class SavedConstant {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedConstant();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SavedConstant();
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

$root.tensorflow.SavedConstant.prototype.operation = "";

$root.tensorflow.SavedVariable = class SavedVariable {

    constructor() {
        this.experimental_distributed_variable_components = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedVariable();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
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
                    message.experimental_distributed_variable_components.push($root.tensorflow.SavedVariable.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedVariable();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "trainable":
                    message.trainable = reader.boolean();
                    break;
                case "synchronization":
                    message.synchronization = reader.enum($root.tensorflow.VariableSynchronization);
                    break;
                case "aggregation":
                    message.aggregation = reader.enum($root.tensorflow.VariableAggregation);
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "device":
                    message.device = reader.string();
                    break;
                case "experimental_distributed_variable_components":
                    message.experimental_distributed_variable_components.push($root.tensorflow.SavedVariable.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedVariable.prototype.dtype = 0;
$root.tensorflow.SavedVariable.prototype.shape = null;
$root.tensorflow.SavedVariable.prototype.trainable = false;
$root.tensorflow.SavedVariable.prototype.synchronization = 0;
$root.tensorflow.SavedVariable.prototype.aggregation = 0;
$root.tensorflow.SavedVariable.prototype.name = "";
$root.tensorflow.SavedVariable.prototype.device = "";

$root.tensorflow.FunctionSpec = class FunctionSpec {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.FunctionSpec();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.fullargspec = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.is_method = reader.bool();
                    break;
                case 5:
                    message.input_signature = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.FunctionSpec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "fullargspec":
                    message.fullargspec = $root.tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "is_method":
                    message.is_method = reader.boolean();
                    break;
                case "input_signature":
                    message.input_signature = $root.tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "jit_compile":
                    message.jit_compile = reader.enum($root.tensorflow.FunctionSpec.JitCompile);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.FunctionSpec.prototype.fullargspec = null;
$root.tensorflow.FunctionSpec.prototype.is_method = false;
$root.tensorflow.FunctionSpec.prototype.input_signature = null;
$root.tensorflow.FunctionSpec.prototype.jit_compile = 0;

$root.tensorflow.FunctionSpec.JitCompile = {
    "DEFAULT": 0,
    "ON": 1,
    "OFF": 2
};

$root.tensorflow.SavedResource = class SavedResource {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedResource();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SavedResource();
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

$root.tensorflow.SavedResource.prototype.device = "";

$root.tensorflow.SaveableObject = class SaveableObject {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SaveableObject();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SaveableObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "save_function":
                    message.save_function = reader.integer();
                    break;
                case "restore_function":
                    message.restore_function = reader.integer();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SaveableObject.prototype.save_function = 0;
$root.tensorflow.SaveableObject.prototype.restore_function = 0;

$root.tensorflow.VariableSynchronization = {
    "VARIABLE_SYNCHRONIZATION_AUTO": 0,
    "VARIABLE_SYNCHRONIZATION_NONE": 1,
    "VARIABLE_SYNCHRONIZATION_ON_WRITE": 2,
    "VARIABLE_SYNCHRONIZATION_ON_READ": 3
};

$root.tensorflow.VariableAggregation = {
    "VARIABLE_AGGREGATION_NONE": 0,
    "VARIABLE_AGGREGATION_SUM": 1,
    "VARIABLE_AGGREGATION_MEAN": 2,
    "VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA": 3
};

$root.tensorflow.VariableDef = class VariableDef {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.VariableDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.save_slice_info_def = $root.tensorflow.SaveSliceInfoDef.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.VariableDef();
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
                    message.save_slice_info_def = $root.tensorflow.SaveSliceInfoDef.decodeText(reader);
                    break;
                case "is_resource":
                    message.is_resource = reader.boolean();
                    break;
                case "trainable":
                    message.trainable = reader.boolean();
                    break;
                case "synchronization":
                    message.synchronization = reader.enum($root.tensorflow.VariableSynchronization);
                    break;
                case "aggregation":
                    message.aggregation = reader.enum($root.tensorflow.VariableAggregation);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.VariableDef.prototype.variable_name = "";
$root.tensorflow.VariableDef.prototype.initial_value_name = "";
$root.tensorflow.VariableDef.prototype.initializer_name = "";
$root.tensorflow.VariableDef.prototype.snapshot_name = "";
$root.tensorflow.VariableDef.prototype.save_slice_info_def = null;
$root.tensorflow.VariableDef.prototype.is_resource = false;
$root.tensorflow.VariableDef.prototype.trainable = false;
$root.tensorflow.VariableDef.prototype.synchronization = 0;
$root.tensorflow.VariableDef.prototype.aggregation = 0;

$root.tensorflow.SaveSliceInfoDef = class SaveSliceInfoDef {

    constructor() {
        this.full_shape = [];
        this.var_offset = [];
        this.var_shape = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SaveSliceInfoDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SaveSliceInfoDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "full_name":
                    message.full_name = reader.string();
                    break;
                case "full_shape":
                    reader.array(message.full_shape, () => reader.integer());
                    break;
                case "var_offset":
                    reader.array(message.var_offset, () => reader.integer());
                    break;
                case "var_shape":
                    reader.array(message.var_shape, () => reader.integer());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SaveSliceInfoDef.prototype.full_name = "";

$root.tensorflow.StructuredValue = class StructuredValue {

    constructor() {
    }

    get kind() {
        $root.tensorflow.StructuredValue.kindSet = $root.tensorflow.StructuredValue.kindSet || new Set([ "none_value", "float64_value", "int64_value", "string_value", "bool_value", "tensor_shape_value", "tensor_dtype_value", "tensor_spec_value", "type_spec_value", "bounded_tensor_spec_value", "list_value", "tuple_value", "dict_value", "named_tuple_value"]);
        return Object.keys(this).find((key) => $root.tensorflow.StructuredValue.kindSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.StructuredValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.none_value = $root.tensorflow.NoneValue.decode(reader, reader.uint32());
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
                    message.tensor_shape_value = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.tensor_dtype_value = reader.int32();
                    break;
                case 33:
                    message.tensor_spec_value = $root.tensorflow.TensorSpecProto.decode(reader, reader.uint32());
                    break;
                case 34:
                    message.type_spec_value = $root.tensorflow.TypeSpecProto.decode(reader, reader.uint32());
                    break;
                case 35:
                    message.bounded_tensor_spec_value = $root.tensorflow.BoundedTensorSpecProto.decode(reader, reader.uint32());
                    break;
                case 51:
                    message.list_value = $root.tensorflow.ListValue.decode(reader, reader.uint32());
                    break;
                case 52:
                    message.tuple_value = $root.tensorflow.TupleValue.decode(reader, reader.uint32());
                    break;
                case 53:
                    message.dict_value = $root.tensorflow.DictValue.decode(reader, reader.uint32());
                    break;
                case 54:
                    message.named_tuple_value = $root.tensorflow.NamedTupleValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.StructuredValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "none_value":
                    message.none_value = $root.tensorflow.NoneValue.decodeText(reader);
                    break;
                case "float64_value":
                    message.float64_value = reader.float();
                    break;
                case "int64_value":
                    message.int64_value = reader.integer();
                    break;
                case "string_value":
                    message.string_value = reader.string();
                    break;
                case "bool_value":
                    message.bool_value = reader.boolean();
                    break;
                case "tensor_shape_value":
                    message.tensor_shape_value = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "tensor_dtype_value":
                    message.tensor_dtype_value = reader.enum($root.tensorflow.DataType);
                    break;
                case "tensor_spec_value":
                    message.tensor_spec_value = $root.tensorflow.TensorSpecProto.decodeText(reader);
                    break;
                case "type_spec_value":
                    message.type_spec_value = $root.tensorflow.TypeSpecProto.decodeText(reader);
                    break;
                case "bounded_tensor_spec_value":
                    message.bounded_tensor_spec_value = $root.tensorflow.BoundedTensorSpecProto.decodeText(reader);
                    break;
                case "list_value":
                    message.list_value = $root.tensorflow.ListValue.decodeText(reader);
                    break;
                case "tuple_value":
                    message.tuple_value = $root.tensorflow.TupleValue.decodeText(reader);
                    break;
                case "dict_value":
                    message.dict_value = $root.tensorflow.DictValue.decodeText(reader);
                    break;
                case "named_tuple_value":
                    message.named_tuple_value = $root.tensorflow.NamedTupleValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.NoneValue = class NoneValue {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.NoneValue();
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
        const message = new $root.tensorflow.NoneValue();
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

$root.tensorflow.ListValue = class ListValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.ListValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push($root.tensorflow.StructuredValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push($root.tensorflow.StructuredValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TupleValue = class TupleValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TupleValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values.push($root.tensorflow.StructuredValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TupleValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values.push($root.tensorflow.StructuredValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.DictValue = class DictValue {

    constructor() {
        this.fields = {};
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.DictValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.fields, () => reader.string(), () => $root.tensorflow.StructuredValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.DictValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "fields":
                    reader.entry(message.fields, () => reader.string(), () => $root.tensorflow.StructuredValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.PairValue = class PairValue {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.PairValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.PairValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = $root.tensorflow.StructuredValue.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.PairValue.prototype.key = "";
$root.tensorflow.PairValue.prototype.value = null;

$root.tensorflow.NamedTupleValue = class NamedTupleValue {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.NamedTupleValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.values.push($root.tensorflow.PairValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.NamedTupleValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "values":
                    message.values.push($root.tensorflow.PairValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.NamedTupleValue.prototype.name = "";

$root.tensorflow.TensorSpecProto = class TensorSpecProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorSpecProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.TensorSpecProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorSpecProto.prototype.name = "";
$root.tensorflow.TensorSpecProto.prototype.shape = null;
$root.tensorflow.TensorSpecProto.prototype.dtype = 0;

$root.tensorflow.BoundedTensorSpecProto = class BoundedTensorSpecProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.BoundedTensorSpecProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.dtype = reader.int32();
                    break;
                case 4:
                    message.minimum = $root.tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.maximum = $root.tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.BoundedTensorSpecProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                case "minimum":
                    message.minimum = $root.tensorflow.TensorProto.decodeText(reader);
                    break;
                case "maximum":
                    message.maximum = $root.tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.BoundedTensorSpecProto.prototype.name = "";
$root.tensorflow.BoundedTensorSpecProto.prototype.shape = null;
$root.tensorflow.BoundedTensorSpecProto.prototype.dtype = 0;
$root.tensorflow.BoundedTensorSpecProto.prototype.minimum = null;
$root.tensorflow.BoundedTensorSpecProto.prototype.maximum = null;

$root.tensorflow.TypeSpecProto = class TypeSpecProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TypeSpecProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type_spec_class = reader.int32();
                    break;
                case 2:
                    message.type_state = $root.tensorflow.StructuredValue.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.type_spec_class_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TypeSpecProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type_spec_class":
                    message.type_spec_class = reader.enum($root.tensorflow.TypeSpecProto.TypeSpecClass);
                    break;
                case "type_state":
                    message.type_state = $root.tensorflow.StructuredValue.decodeText(reader);
                    break;
                case "type_spec_class_name":
                    message.type_spec_class_name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TypeSpecProto.prototype.type_spec_class = 0;
$root.tensorflow.TypeSpecProto.prototype.type_state = null;
$root.tensorflow.TypeSpecProto.prototype.type_spec_class_name = "";

$root.tensorflow.TypeSpecProto.TypeSpecClass = {
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
    "NDARRAY_SPEC": 11
};

$root.tensorflow.TrackableObjectGraph = class TrackableObjectGraph {

    constructor() {
        this.nodes = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TrackableObjectGraph();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.nodes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TrackableObjectGraph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "nodes":
                    message.nodes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TrackableObjectGraph.TrackableObject = class TrackableObject {

    constructor() {
        this.children = [];
        this.attributes = [];
        this.slot_variables = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.attributes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "children":
                    message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader));
                    break;
                case "attributes":
                    message.attributes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decodeText(reader));
                    break;
                case "slot_variables":
                    message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference = class ObjectReference {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node_id":
                    message.node_id = reader.integer();
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

$root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.prototype.node_id = 0;
$root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.prototype.local_name = "";

$root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor = class SerializedTensor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                case 4:
                    message.optional_restore = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
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
                case "optional_restore":
                    message.optional_restore = reader.boolean();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.name = "";
$root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.full_name = "";
$root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.checkpoint_key = "";
$root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.prototype.optional_restore = false;

$root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference = class SlotVariableReference {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "original_variable_node_id":
                    message.original_variable_node_id = reader.integer();
                    break;
                case "slot_name":
                    message.slot_name = reader.string();
                    break;
                case "slot_variable_node_id":
                    message.slot_variable_node_id = reader.integer();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.prototype.original_variable_node_id = 0;
$root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.prototype.slot_name = "";
$root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.prototype.slot_variable_node_id = 0;

$root.tensorflow.SaverDef = class SaverDef {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SaverDef();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SaverDef();
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
                    message.max_to_keep = reader.integer();
                    break;
                case "sharded":
                    message.sharded = reader.boolean();
                    break;
                case "keep_checkpoint_every_n_hours":
                    message.keep_checkpoint_every_n_hours = reader.float();
                    break;
                case "version":
                    message.version = reader.enum($root.tensorflow.SaverDef.CheckpointFormatVersion);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SaverDef.prototype.filename_tensor_name = "";
$root.tensorflow.SaverDef.prototype.save_tensor_name = "";
$root.tensorflow.SaverDef.prototype.restore_op_name = "";
$root.tensorflow.SaverDef.prototype.max_to_keep = 0;
$root.tensorflow.SaverDef.prototype.sharded = false;
$root.tensorflow.SaverDef.prototype.keep_checkpoint_every_n_hours = 0;
$root.tensorflow.SaverDef.prototype.version = 0;

$root.tensorflow.SaverDef.CheckpointFormatVersion = {
    "LEGACY": 0,
    "V1": 1,
    "V2": 2
};

$root.tensorflow.BundleHeaderProto = class BundleHeaderProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.BundleHeaderProto();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.version = $root.tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.BundleHeaderProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_shards":
                    message.num_shards = reader.integer();
                    break;
                case "endianness":
                    message.endianness = reader.enum($root.tensorflow.BundleHeaderProto.Endianness);
                    break;
                case "version":
                    message.version = $root.tensorflow.VersionDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.BundleHeaderProto.prototype.num_shards = 0;
$root.tensorflow.BundleHeaderProto.prototype.endianness = 0;
$root.tensorflow.BundleHeaderProto.prototype.version = null;

$root.tensorflow.BundleHeaderProto.Endianness = {
    "LITTLE": 0,
    "BIG": 1
};

$root.tensorflow.BundleEntryProto = class BundleEntryProto {

    constructor() {
        this.slices = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.BundleEntryProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dtype = reader.int32();
                    break;
                case 2:
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
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
                    message.slices.push($root.tensorflow.TensorSliceProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.BundleEntryProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dtype":
                    message.dtype = reader.enum($root.tensorflow.DataType);
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "shard_id":
                    message.shard_id = reader.integer();
                    break;
                case "offset":
                    message.offset = reader.integer();
                    break;
                case "size":
                    message.size = reader.integer();
                    break;
                case "crc32c":
                    message.crc32c = reader.integer();
                    break;
                case "slices":
                    message.slices.push($root.tensorflow.TensorSliceProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.BundleEntryProto.prototype.dtype = 0;
$root.tensorflow.BundleEntryProto.prototype.shape = null;
$root.tensorflow.BundleEntryProto.prototype.shard_id = 0;
$root.tensorflow.BundleEntryProto.prototype.offset = protobuf.Int64.create(0);
$root.tensorflow.BundleEntryProto.prototype.size = protobuf.Int64.create(0);
$root.tensorflow.BundleEntryProto.prototype.crc32c = 0;

$root.tensorflow.TensorSliceProto = class TensorSliceProto {

    constructor() {
        this.extent = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorSliceProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.extent.push($root.tensorflow.TensorSliceProto.Extent.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.TensorSliceProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "extent":
                    message.extent.push($root.tensorflow.TensorSliceProto.Extent.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorSliceProto.Extent = class Extent {

    constructor() {
    }

    get has_length() {
        $root.tensorflow.TensorSliceProto.Extent.has_lengthSet = $root.tensorflow.TensorSliceProto.Extent.has_lengthSet || new Set([ "length"]);
        return Object.keys(this).find((key) => $root.tensorflow.TensorSliceProto.Extent.has_lengthSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TensorSliceProto.Extent();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.TensorSliceProto.Extent();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "start":
                    message.start = reader.integer();
                    break;
                case "length":
                    message.length = reader.integer();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.TensorSliceProto.Extent.prototype.start = protobuf.Int64.create(0);

$root.tensorflow.SavedSliceMeta = class SavedSliceMeta {

    constructor() {
        this.slice = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedSliceMeta();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.shape = $root.tensorflow.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.type = reader.int32();
                    break;
                case 4:
                    message.slice.push($root.tensorflow.TensorSliceProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedSliceMeta();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "shape":
                    message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader);
                    break;
                case "type":
                    message.type = reader.enum($root.tensorflow.DataType);
                    break;
                case "slice":
                    message.slice.push($root.tensorflow.TensorSliceProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedSliceMeta.prototype.name = "";
$root.tensorflow.SavedSliceMeta.prototype.shape = null;
$root.tensorflow.SavedSliceMeta.prototype.type = 0;

$root.tensorflow.SavedTensorSliceMeta = class SavedTensorSliceMeta {

    constructor() {
        this.tensor = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedTensorSliceMeta();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor.push($root.tensorflow.SavedSliceMeta.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.versions = $root.tensorflow.VersionDef.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedTensorSliceMeta();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor.push($root.tensorflow.SavedSliceMeta.decodeText(reader));
                    break;
                case "versions":
                    message.versions = $root.tensorflow.VersionDef.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedTensorSliceMeta.prototype.versions = null;

$root.tensorflow.SavedSlice = class SavedSlice {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedSlice();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.slice = $root.tensorflow.TensorSliceProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.data = $root.tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedSlice();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "slice":
                    message.slice = $root.tensorflow.TensorSliceProto.decodeText(reader);
                    break;
                case "data":
                    message.data = $root.tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedSlice.prototype.name = "";
$root.tensorflow.SavedSlice.prototype.slice = null;
$root.tensorflow.SavedSlice.prototype.data = null;

$root.tensorflow.SavedTensorSlices = class SavedTensorSlices {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SavedTensorSlices();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.meta = $root.tensorflow.SavedTensorSliceMeta.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.data = $root.tensorflow.SavedSlice.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.SavedTensorSlices();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "meta":
                    message.meta = $root.tensorflow.SavedTensorSliceMeta.decodeText(reader);
                    break;
                case "data":
                    message.data = $root.tensorflow.SavedSlice.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SavedTensorSlices.prototype.meta = null;
$root.tensorflow.SavedTensorSlices.prototype.data = null;

$root.tensorflow.Event = class Event {

    constructor() {
    }

    get what() {
        $root.tensorflow.Event.whatSet = $root.tensorflow.Event.whatSet || new Set([ "file_version", "graph_def", "summary", "log_message", "session_log", "tagged_run_metadata", "meta_graph_def"]);
        return Object.keys(this).find((key) => $root.tensorflow.Event.whatSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.Event();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.summary = $root.tensorflow.Summary.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.log_message = $root.tensorflow.LogMessage.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.session_log = $root.tensorflow.SessionLog.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.tagged_run_metadata = $root.tensorflow.TaggedRunMetadata.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.meta_graph_def = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.Event();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "wall_time":
                    message.wall_time = reader.float();
                    break;
                case "step":
                    message.step = reader.integer();
                    break;
                case "file_version":
                    message.file_version = reader.string();
                    break;
                case "graph_def":
                    message.graph_def = reader.bytes();
                    break;
                case "summary":
                    message.summary = $root.tensorflow.Summary.decodeText(reader);
                    break;
                case "log_message":
                    message.log_message = $root.tensorflow.LogMessage.decodeText(reader);
                    break;
                case "session_log":
                    message.session_log = $root.tensorflow.SessionLog.decodeText(reader);
                    break;
                case "tagged_run_metadata":
                    message.tagged_run_metadata = $root.tensorflow.TaggedRunMetadata.decodeText(reader);
                    break;
                case "meta_graph_def":
                    message.meta_graph_def = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.Event.prototype.wall_time = 0;
$root.tensorflow.Event.prototype.step = protobuf.Int64.create(0);

$root.tensorflow.LogMessage = class LogMessage {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.LogMessage();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.LogMessage();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "level":
                    message.level = reader.enum($root.tensorflow.LogMessage.Level);
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

$root.tensorflow.LogMessage.prototype.level = 0;
$root.tensorflow.LogMessage.prototype.message = "";

$root.tensorflow.LogMessage.Level = {
    "UNKNOWN": 0,
    "DEBUGGING": 10,
    "INFO": 20,
    "WARN": 30,
    "ERROR": 40,
    "FATAL": 50
};

$root.tensorflow.SessionLog = class SessionLog {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SessionLog();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SessionLog();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "status":
                    message.status = reader.enum($root.tensorflow.SessionLog.SessionStatus);
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

$root.tensorflow.SessionLog.prototype.status = 0;
$root.tensorflow.SessionLog.prototype.checkpoint_path = "";
$root.tensorflow.SessionLog.prototype.msg = "";

$root.tensorflow.SessionLog.SessionStatus = {
    "STATUS_UNSPECIFIED": 0,
    "START": 1,
    "STOP": 2,
    "CHECKPOINT": 3
};

$root.tensorflow.TaggedRunMetadata = class TaggedRunMetadata {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.TaggedRunMetadata();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.TaggedRunMetadata();
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

$root.tensorflow.TaggedRunMetadata.prototype.tag = "";
$root.tensorflow.TaggedRunMetadata.prototype.run_metadata = new Uint8Array([]);

$root.tensorflow.WorkerHealth = {
    "OK": 0,
    "RECEIVED_SHUTDOWN_SIGNAL": 1,
    "INTERNAL_ERROR": 2,
    "SHUTTING_DOWN": 3
};

$root.tensorflow.WorkerShutdownMode = {
    "DEFAULT": 0,
    "NOT_CONFIGURED": 1,
    "WAIT_FOR_COORDINATOR": 2,
    "SHUTDOWN_AFTER_TIMEOUT": 3
};

$root.tensorflow.WatchdogConfig = class WatchdogConfig {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.WatchdogConfig();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.WatchdogConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "timeout_ms":
                    message.timeout_ms = reader.integer();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.WatchdogConfig.prototype.timeout_ms = protobuf.Int64.create(0);

$root.tensorflow.RequestedExitCode = class RequestedExitCode {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.RequestedExitCode();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.RequestedExitCode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "exit_code":
                    message.exit_code = reader.integer();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.RequestedExitCode.prototype.exit_code = 0;

$root.tensorflow.WorkerHeartbeatRequest = class WorkerHeartbeatRequest {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.WorkerHeartbeatRequest();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shutdown_mode = reader.int32();
                    break;
                case 2:
                    message.watchdog_config = $root.tensorflow.WatchdogConfig.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.exit_code = $root.tensorflow.RequestedExitCode.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.WorkerHeartbeatRequest();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shutdown_mode":
                    message.shutdown_mode = reader.enum($root.tensorflow.WorkerShutdownMode);
                    break;
                case "watchdog_config":
                    message.watchdog_config = $root.tensorflow.WatchdogConfig.decodeText(reader);
                    break;
                case "exit_code":
                    message.exit_code = $root.tensorflow.RequestedExitCode.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.WorkerHeartbeatRequest.prototype.shutdown_mode = 0;
$root.tensorflow.WorkerHeartbeatRequest.prototype.watchdog_config = null;
$root.tensorflow.WorkerHeartbeatRequest.prototype.exit_code = null;

$root.tensorflow.WorkerHeartbeatResponse = class WorkerHeartbeatResponse {

    constructor() {
        this.worker_log = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.WorkerHeartbeatResponse();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.health_status = reader.int32();
                    break;
                case 2:
                    message.worker_log.push($root.tensorflow.Event.decode(reader, reader.uint32()));
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
        const message = new $root.tensorflow.WorkerHeartbeatResponse();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "health_status":
                    message.health_status = reader.enum($root.tensorflow.WorkerHealth);
                    break;
                case "worker_log":
                    message.worker_log.push($root.tensorflow.Event.decodeText(reader));
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

$root.tensorflow.WorkerHeartbeatResponse.prototype.health_status = 0;
$root.tensorflow.WorkerHeartbeatResponse.prototype.hostname = "";

$root.tensorflow.SummaryDescription = class SummaryDescription {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SummaryDescription();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SummaryDescription();
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

$root.tensorflow.SummaryDescription.prototype.type_hint = "";

$root.tensorflow.HistogramProto = class HistogramProto {

    constructor() {
        this.bucket_limit = [];
        this.bucket = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.HistogramProto();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.HistogramProto();
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
                case "num":
                    message.num = reader.float();
                    break;
                case "sum":
                    message.sum = reader.float();
                    break;
                case "sum_squares":
                    message.sum_squares = reader.float();
                    break;
                case "bucket_limit":
                    reader.array(message.bucket_limit, () => reader.float());
                    break;
                case "bucket":
                    reader.array(message.bucket, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.HistogramProto.prototype.min = 0;
$root.tensorflow.HistogramProto.prototype.max = 0;
$root.tensorflow.HistogramProto.prototype.num = 0;
$root.tensorflow.HistogramProto.prototype.sum = 0;
$root.tensorflow.HistogramProto.prototype.sum_squares = 0;

$root.tensorflow.SummaryMetadata = class SummaryMetadata {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SummaryMetadata();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.plugin_data = $root.tensorflow.SummaryMetadata.PluginData.decode(reader, reader.uint32());
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
        const message = new $root.tensorflow.SummaryMetadata();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "plugin_data":
                    message.plugin_data = $root.tensorflow.SummaryMetadata.PluginData.decodeText(reader);
                    break;
                case "display_name":
                    message.display_name = reader.string();
                    break;
                case "summary_description":
                    message.summary_description = reader.string();
                    break;
                case "data_class":
                    message.data_class = reader.enum($root.tensorflow.DataClass);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.SummaryMetadata.prototype.plugin_data = null;
$root.tensorflow.SummaryMetadata.prototype.display_name = "";
$root.tensorflow.SummaryMetadata.prototype.summary_description = "";
$root.tensorflow.SummaryMetadata.prototype.data_class = 0;

$root.tensorflow.SummaryMetadata.PluginData = class PluginData {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.SummaryMetadata.PluginData();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.SummaryMetadata.PluginData();
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

$root.tensorflow.SummaryMetadata.PluginData.prototype.plugin_name = "";
$root.tensorflow.SummaryMetadata.PluginData.prototype.content = new Uint8Array([]);

$root.tensorflow.DataClass = {
    "DATA_CLASS_UNKNOWN": 0,
    "DATA_CLASS_SCALAR": 1,
    "DATA_CLASS_TENSOR": 2,
    "DATA_CLASS_BLOB_SEQUENCE": 3
};

$root.tensorflow.Summary = class Summary {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.Summary();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push($root.tensorflow.Summary.Value.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.Summary();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value.push($root.tensorflow.Summary.Value.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.Summary.Image = class Image {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.Summary.Image();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.Summary.Image();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "height":
                    message.height = reader.integer();
                    break;
                case "width":
                    message.width = reader.integer();
                    break;
                case "colorspace":
                    message.colorspace = reader.integer();
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

$root.tensorflow.Summary.Image.prototype.height = 0;
$root.tensorflow.Summary.Image.prototype.width = 0;
$root.tensorflow.Summary.Image.prototype.colorspace = 0;
$root.tensorflow.Summary.Image.prototype.encoded_image_string = new Uint8Array([]);

$root.tensorflow.Summary.Audio = class Audio {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.Summary.Audio();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        const message = new $root.tensorflow.Summary.Audio();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "sample_rate":
                    message.sample_rate = reader.float();
                    break;
                case "num_channels":
                    message.num_channels = reader.integer();
                    break;
                case "length_frames":
                    message.length_frames = reader.integer();
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

$root.tensorflow.Summary.Audio.prototype.sample_rate = 0;
$root.tensorflow.Summary.Audio.prototype.num_channels = protobuf.Int64.create(0);
$root.tensorflow.Summary.Audio.prototype.length_frames = protobuf.Int64.create(0);
$root.tensorflow.Summary.Audio.prototype.encoded_audio_string = new Uint8Array([]);
$root.tensorflow.Summary.Audio.prototype.content_type = "";

$root.tensorflow.Summary.Value = class Value {

    constructor() {
    }

    get value() {
        $root.tensorflow.Summary.Value.valueSet = $root.tensorflow.Summary.Value.valueSet || new Set([ "simple_value", "obsolete_old_style_histogram", "image", "histo", "audio", "tensor"]);
        return Object.keys(this).find((key) => $root.tensorflow.Summary.Value.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.tensorflow.Summary.Value();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.metadata = $root.tensorflow.SummaryMetadata.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.simple_value = reader.float();
                    break;
                case 3:
                    message.obsolete_old_style_histogram = reader.bytes();
                    break;
                case 4:
                    message.image = $root.tensorflow.Summary.Image.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.histo = $root.tensorflow.HistogramProto.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.audio = $root.tensorflow.Summary.Audio.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.tensor = $root.tensorflow.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.tensorflow.Summary.Value();
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
                    message.metadata = $root.tensorflow.SummaryMetadata.decodeText(reader);
                    break;
                case "simple_value":
                    message.simple_value = reader.float();
                    break;
                case "obsolete_old_style_histogram":
                    message.obsolete_old_style_histogram = reader.bytes();
                    break;
                case "image":
                    message.image = $root.tensorflow.Summary.Image.decodeText(reader);
                    break;
                case "histo":
                    message.histo = $root.tensorflow.HistogramProto.decodeText(reader);
                    break;
                case "audio":
                    message.audio = $root.tensorflow.Summary.Audio.decodeText(reader);
                    break;
                case "tensor":
                    message.tensor = $root.tensorflow.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.tensorflow.Summary.Value.prototype.node_name = "";
$root.tensorflow.Summary.Value.prototype.tag = "";
$root.tensorflow.Summary.Value.prototype.metadata = null;

$root.google = {};

$root.google.protobuf = {};

$root.google.protobuf.Any = class Any {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.google.protobuf.Any();
        const end = length !== undefined ? reader.position + length : reader.length;
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
        return reader.any(() => new $root.google.protobuf.Any());
    }
};

$root.google.protobuf.Any.prototype.type_url = "";
$root.google.protobuf.Any.prototype.value = new Uint8Array([]);
