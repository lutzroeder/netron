
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedModel();
        if ('savedModelSchemaVersion' in obj) {
            message.saved_model_schema_version = BigInt(obj.savedModelSchemaVersion);
        }
        if ('metaGraphs' in obj) {
            message.meta_graphs = obj.metaGraphs.map((obj) => tensorflow.MetaGraphDef.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.MetaGraphDef();
        if ('metaInfoDef' in obj) {
            message.meta_info_def = tensorflow.MetaGraphDef.MetaInfoDef.decodeJson(obj.metaInfoDef);
        }
        if ('graphDef' in obj) {
            message.graph_def = tensorflow.GraphDef.decodeJson(obj.graphDef);
        }
        if ('saverDef' in obj) {
            message.saver_def = tensorflow.SaverDef.decodeJson(obj.saverDef);
        }
        if ('collectionDef' in obj) {
            for (const [key, value] of Object.entries(obj.collectionDef)) {
                message.collection_def[key] = tensorflow.CollectionDef.decodeJson(value);
            }
        }
        if ('signatureDef' in obj) {
            for (const [key, value] of Object.entries(obj.signatureDef)) {
                message.signature_def[key] = tensorflow.SignatureDef.decodeJson(value);
            }
        }
        if ('assetFileDef' in obj) {
            message.asset_file_def = obj.assetFileDef.map((obj) => tensorflow.AssetFileDef.decodeJson(obj));
        }
        if ('objectGraphDef' in obj) {
            message.object_graph_def = tensorflow.SavedObjectGraph.decodeJson(obj.objectGraphDef);
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

    static decodeJson(obj) {
        const message = new tensorflow.MetaGraphDef.MetaInfoDef();
        if ('metaGraphVersion' in obj) {
            message.meta_graph_version = obj.metaGraphVersion;
        }
        if ('strippedOpList' in obj) {
            message.stripped_op_list = tensorflow.OpList.decodeJson(obj.strippedOpList);
        }
        if ('anyInfo' in obj) {
            message.any_info = google.protobuf.Any.decodeJson(obj.anyInfo);
        }
        if ('tags' in obj) {
            message.tags = obj.tags;
        }
        if ('tensorflowVersion' in obj) {
            message.tensorflow_version = obj.tensorflowVersion;
        }
        if ('tensorflowGitVersion' in obj) {
            message.tensorflow_git_version = obj.tensorflowGitVersion;
        }
        if ('strippedDefaultAttrs' in obj) {
            message.stripped_default_attrs = obj.strippedDefaultAttrs;
        }
        if ('functionAliases' in obj) {
            for (const [key, value] of Object.entries(obj.functionAliases)) {
                message.function_aliases[key] = value;
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

    static decodeJson(obj) {
        const message = new tensorflow.CollectionDef();
        if ('nodeList' in obj) {
            message.node_list = tensorflow.CollectionDef.NodeList.decodeJson(obj.nodeList);
        }
        if ('bytesList' in obj) {
            message.bytes_list = tensorflow.CollectionDef.BytesList.decodeJson(obj.bytesList);
        }
        if ('int64List' in obj) {
            message.int64_list = tensorflow.CollectionDef.Int64List.decodeJson(obj.int64List);
        }
        if ('floatList' in obj) {
            message.float_list = tensorflow.CollectionDef.FloatList.decodeJson(obj.floatList);
        }
        if ('anyList' in obj) {
            message.any_list = tensorflow.CollectionDef.AnyList.decodeJson(obj.anyList);
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

    static decodeJson(obj) {
        const message = new tensorflow.CollectionDef.NodeList();
        if ('value' in obj) {
            message.value = obj.value;
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

    static decodeJson(obj) {
        const message = new tensorflow.CollectionDef.BytesList();
        if ('value' in obj) {
            message.value = obj.value.map((obj) => typeof obj === 'string' ? Uint8Array.from(atob(obj), (c) => c.charCodeAt(0)) : Uint8Array.from(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.CollectionDef.Int64List();
        if ('value' in obj) {
            message.value = obj.value.map((obj) => BigInt(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.CollectionDef.FloatList();
        if ('value' in obj) {
            message.value = obj.value.map((obj) => Number(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.CollectionDef.AnyList();
        if ('value' in obj) {
            message.value = obj.value.map((obj) => google.protobuf.Any.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorInfo();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('cooSparse' in obj) {
            message.coo_sparse = tensorflow.TensorInfo.CooSparse.decodeJson(obj.cooSparse);
        }
        if ('compositeTensor' in obj) {
            message.composite_tensor = tensorflow.TensorInfo.CompositeTensor.decodeJson(obj.compositeTensor);
        }
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('tensorShape' in obj) {
            message.tensor_shape = tensorflow.TensorShapeProto.decodeJson(obj.tensorShape);
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorInfo.CooSparse();
        if ('valuesTensorName' in obj) {
            message.values_tensor_name = obj.valuesTensorName;
        }
        if ('indicesTensorName' in obj) {
            message.indices_tensor_name = obj.indicesTensorName;
        }
        if ('denseShapeTensorName' in obj) {
            message.dense_shape_tensor_name = obj.denseShapeTensorName;
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorInfo.CompositeTensor();
        if ('typeSpec' in obj) {
            message.type_spec = tensorflow.TypeSpecProto.decodeJson(obj.typeSpec);
        }
        if ('components' in obj) {
            message.components = obj.components.map((obj) => tensorflow.TensorInfo.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.SignatureDef();
        if ('inputs' in obj) {
            for (const [key, value] of Object.entries(obj.inputs)) {
                message.inputs[key] = tensorflow.TensorInfo.decodeJson(value);
            }
        }
        if ('outputs' in obj) {
            for (const [key, value] of Object.entries(obj.outputs)) {
                message.outputs[key] = tensorflow.TensorInfo.decodeJson(value);
            }
        }
        if ('methodName' in obj) {
            message.method_name = obj.methodName;
        }
        if ('defaults' in obj) {
            for (const [key, value] of Object.entries(obj.defaults)) {
                message.defaults[key] = tensorflow.TensorProto.decodeJson(value);
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

    static decodeJson(obj) {
        const message = new tensorflow.AssetFileDef();
        if ('tensorInfo' in obj) {
            message.tensor_info = tensorflow.TensorInfo.decodeJson(obj.tensorInfo);
        }
        if ('filename' in obj) {
            message.filename = obj.filename;
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

    static decodeJson(obj) {
        const message = new tensorflow.GraphDef();
        if ('node' in obj) {
            message.node = obj.node.map((obj) => tensorflow.NodeDef.decodeJson(obj));
        }
        if ('versions' in obj) {
            message.versions = tensorflow.VersionDef.decodeJson(obj.versions);
        }
        if ('version' in obj) {
            message.version = Number(obj.version);
        }
        if ('library' in obj) {
            message.library = tensorflow.FunctionDefLibrary.decodeJson(obj.library);
        }
        if ('debugInfo' in obj) {
            message.debug_info = tensorflow.GraphDebugInfo.decodeJson(obj.debugInfo);
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

    static decodeJson(obj) {
        const message = new tensorflow.FunctionDefLibrary();
        if ('function' in obj) {
            message.function = obj.function.map((obj) => tensorflow.FunctionDef.decodeJson(obj));
        }
        if ('gradient' in obj) {
            message.gradient = obj.gradient.map((obj) => tensorflow.GradientDef.decodeJson(obj));
        }
        if ('registeredGradients' in obj) {
            message.registered_gradients = obj.registeredGradients.map((obj) => tensorflow.RegisteredGradient.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.FunctionDef();
        if ('signature' in obj) {
            message.signature = tensorflow.OpDef.decodeJson(obj.signature);
        }
        if ('attr' in obj) {
            for (const [key, value] of Object.entries(obj.attr)) {
                message.attr[key] = tensorflow.AttrValue.decodeJson(value);
            }
        }
        if ('argAttr' in obj) {
            for (const [key, value] of Object.entries(obj.argAttr)) {
                message.arg_attr[key] = tensorflow.FunctionDef.ArgAttrs.decodeJson(value);
            }
        }
        if ('resourceArgUniqueId' in obj) {
            for (const [key, value] of Object.entries(obj.resourceArgUniqueId)) {
                message.resource_arg_unique_id[key] = value;
            }
        }
        if ('nodeDef' in obj) {
            message.node_def = obj.nodeDef.map((obj) => tensorflow.NodeDef.decodeJson(obj));
        }
        if ('ret' in obj) {
            for (const [key, value] of Object.entries(obj.ret)) {
                message.ret[key] = value;
            }
        }
        if ('controlRet' in obj) {
            for (const [key, value] of Object.entries(obj.controlRet)) {
                message.control_ret[key] = value;
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

    static decodeJson(obj) {
        const message = new tensorflow.FunctionDef.ArgAttrs();
        if ('attr' in obj) {
            for (const [key, value] of Object.entries(obj.attr)) {
                message.attr[key] = tensorflow.AttrValue.decodeJson(value);
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

    static decodeJson(obj) {
        const message = new tensorflow.GradientDef();
        if ('functionName' in obj) {
            message.function_name = obj.functionName;
        }
        if ('gradientFunc' in obj) {
            message.gradient_func = obj.gradientFunc;
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

    static decodeJson(obj) {
        const message = new tensorflow.RegisteredGradient();
        if ('gradientFunc' in obj) {
            message.gradient_func = obj.gradientFunc;
        }
        if ('registeredOpType' in obj) {
            message.registered_op_type = obj.registeredOpType;
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

    static decodeJson(obj) {
        const message = new tensorflow.AttrValue();
        if ('s' in obj) {
            message.s = typeof source === 'string' ? Uint8Array.from(atob(obj.s), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.s);
        }
        if ('i' in obj) {
            message.i = BigInt(obj.i);
        }
        if ('f' in obj) {
            message.f = Number(obj.f);
        }
        if ('b' in obj) {
            message.b = obj.b;
        }
        if ('type' in obj) {
            message.type = typeof obj.type === 'string' ? tensorflow.DataType[obj.type] : obj.type;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('tensor' in obj) {
            message.tensor = tensorflow.TensorProto.decodeJson(obj.tensor);
        }
        if ('list' in obj) {
            message.list = tensorflow.AttrValue.ListValue.decodeJson(obj.list);
        }
        if ('func' in obj) {
            message.func = tensorflow.NameAttrList.decodeJson(obj.func);
        }
        if ('placeholder' in obj) {
            message.placeholder = obj.placeholder;
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

    static decodeJson(obj) {
        const message = new tensorflow.AttrValue.ListValue();
        if ('s' in obj) {
            message.s = obj.s.map((obj) => typeof obj === 'string' ? Uint8Array.from(atob(obj), (c) => c.charCodeAt(0)) : Uint8Array.from(obj));
        }
        if ('i' in obj) {
            message.i = obj.i.map((obj) => BigInt(obj));
        }
        if ('f' in obj) {
            message.f = obj.f.map((obj) => Number(obj));
        }
        if ('b' in obj) {
            message.b = obj.b;
        }
        if ('type' in obj) {
            message.type = obj.type.map((key) => typeof key === 'string' ? tensorflow.DataType[key] : key);
        }
        if ('shape' in obj) {
            message.shape = obj.shape.map((obj) => tensorflow.TensorShapeProto.decodeJson(obj));
        }
        if ('tensor' in obj) {
            message.tensor = obj.tensor.map((obj) => tensorflow.TensorProto.decodeJson(obj));
        }
        if ('func' in obj) {
            message.func = obj.func.map((obj) => tensorflow.NameAttrList.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.NameAttrList();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('attr' in obj) {
            for (const [key, value] of Object.entries(obj.attr)) {
                message.attr[key] = tensorflow.AttrValue.decodeJson(value);
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorProto();
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('tensorShape' in obj) {
            message.tensor_shape = tensorflow.TensorShapeProto.decodeJson(obj.tensorShape);
        }
        if ('versionNumber' in obj) {
            message.version_number = Number(obj.versionNumber);
        }
        if ('tensorContent' in obj) {
            message.tensor_content = typeof source === 'string' ? Uint8Array.from(atob(obj.tensorContent), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.tensorContent);
        }
        if ('halfVal' in obj) {
            message.half_val = obj.halfVal.map((obj) => Number(obj));
        }
        if ('floatVal' in obj) {
            message.float_val = obj.floatVal.map((obj) => Number(obj));
        }
        if ('doubleVal' in obj) {
            message.double_val = obj.doubleVal.map((obj) => Number(obj));
        }
        if ('intVal' in obj) {
            message.int_val = obj.intVal.map((obj) => Number(obj));
        }
        if ('stringVal' in obj) {
            message.string_val = obj.stringVal.map((obj) => typeof obj === 'string' ? Uint8Array.from(atob(obj), (c) => c.charCodeAt(0)) : Uint8Array.from(obj));
        }
        if ('scomplexVal' in obj) {
            message.scomplex_val = obj.scomplexVal.map((obj) => Number(obj));
        }
        if ('int64Val' in obj) {
            message.int64_val = obj.int64Val.map((obj) => BigInt(obj));
        }
        if ('boolVal' in obj) {
            message.bool_val = obj.boolVal;
        }
        if ('dcomplexVal' in obj) {
            message.dcomplex_val = obj.dcomplexVal.map((obj) => Number(obj));
        }
        if ('resourceHandleVal' in obj) {
            message.resource_handle_val = obj.resourceHandleVal.map((obj) => tensorflow.ResourceHandleProto.decodeJson(obj));
        }
        if ('variantVal' in obj) {
            message.variant_val = obj.variantVal.map((obj) => tensorflow.VariantTensorDataProto.decodeJson(obj));
        }
        if ('uint32Val' in obj) {
            message.uint32_val = obj.uint32Val.map((obj) => Number(obj));
        }
        if ('uint64Val' in obj) {
            message.uint64_val = obj.uint64Val.map((obj) => BigInt(obj));
        }
        if ('float8Val' in obj) {
            message.float8_val = typeof source === 'string' ? Uint8Array.from(atob(obj.float8Val), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.float8Val);
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

    static decodeJson(obj) {
        const message = new tensorflow.VariantTensorDataProto();
        if ('typeName' in obj) {
            message.type_name = obj.typeName;
        }
        if ('metadata' in obj) {
            message.metadata = typeof source === 'string' ? Uint8Array.from(atob(obj.metadata), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.metadata);
        }
        if ('tensors' in obj) {
            message.tensors = obj.tensors.map((obj) => tensorflow.TensorProto.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.ResourceHandleProto();
        if ('device' in obj) {
            message.device = obj.device;
        }
        if ('container' in obj) {
            message.container = obj.container;
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('hashCode' in obj) {
            message.hash_code = BigInt(obj.hashCode);
        }
        if ('maybeTypeName' in obj) {
            message.maybe_type_name = obj.maybeTypeName;
        }
        if ('dtypesAndShapes' in obj) {
            message.dtypes_and_shapes = obj.dtypesAndShapes.map((obj) => tensorflow.ResourceHandleProto.DtypeAndShape.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.ResourceHandleProto.DtypeAndShape();
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorShapeProto();
        if ('dim' in obj) {
            message.dim = obj.dim.map((obj) => tensorflow.TensorShapeProto.Dim.decodeJson(obj));
        }
        if ('unknownRank' in obj) {
            message.unknown_rank = obj.unknownRank;
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorShapeProto.Dim();
        if ('size' in obj) {
            message.size = BigInt(obj.size);
        }
        if ('name' in obj) {
            message.name = obj.name;
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
    "DT_FLOAT8_E4M3FNUZ": 26,
    "DT_FLOAT8_E4M3B11FNUZ": 27,
    "DT_FLOAT8_E5M2FNUZ": 28,
    "DT_INT4": 29,
    "DT_UINT4": 30,
    "DT_INT2": 31,
    "DT_UINT2": 32,
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
    "DT_FLOAT8_E4M3FNUZ_REF": 126,
    "DT_FLOAT8_E4M3B11FNUZ_REF": 127,
    "DT_FLOAT8_E5M2FNUZ_REF": 128,
    "DT_INT4_REF": 129,
    "DT_UINT4_REF": 130,
    "DT_INT2_REF": 131,
    "DT_UINT2_REF": 132
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

    static decodeJson(obj) {
        const message = new tensorflow.SerializedDType();
        if ('datatype' in obj) {
            message.datatype = typeof obj.datatype === 'string' ? tensorflow.DataType[obj.datatype] : obj.datatype;
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

    static decodeJson(obj) {
        const message = new tensorflow.NodeDef();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('op' in obj) {
            message.op = obj.op;
        }
        if ('input' in obj) {
            message.input = obj.input;
        }
        if ('device' in obj) {
            message.device = obj.device;
        }
        if ('attr' in obj) {
            for (const [key, value] of Object.entries(obj.attr)) {
                message.attr[key] = tensorflow.AttrValue.decodeJson(value);
            }
        }
        if ('experimentalDebugInfo' in obj) {
            message.experimental_debug_info = tensorflow.NodeDef.ExperimentalDebugInfo.decodeJson(obj.experimentalDebugInfo);
        }
        if ('experimentalType' in obj) {
            message.experimental_type = tensorflow.FullTypeDef.decodeJson(obj.experimentalType);
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

    static decodeJson(obj) {
        const message = new tensorflow.NodeDef.ExperimentalDebugInfo();
        if ('originalNodeNames' in obj) {
            message.original_node_names = obj.originalNodeNames;
        }
        if ('originalFuncNames' in obj) {
            message.original_func_names = obj.originalFuncNames;
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

    static decodeJson(obj) {
        const message = new tensorflow.FullTypeDef();
        if ('typeId' in obj) {
            message.type_id = typeof obj.typeId === 'string' ? tensorflow.FullTypeId[obj.typeId] : obj.typeId;
        }
        if ('args' in obj) {
            message.args = obj.args.map((obj) => tensorflow.FullTypeDef.decodeJson(obj));
        }
        if ('s' in obj) {
            message.s = obj.s;
        }
        if ('i' in obj) {
            message.i = BigInt(obj.i);
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

    static decodeJson(obj) {
        const message = new tensorflow.OpDef();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('inputArg' in obj) {
            message.input_arg = obj.inputArg.map((obj) => tensorflow.OpDef.ArgDef.decodeJson(obj));
        }
        if ('outputArg' in obj) {
            message.output_arg = obj.outputArg.map((obj) => tensorflow.OpDef.ArgDef.decodeJson(obj));
        }
        if ('controlOutput' in obj) {
            message.control_output = obj.controlOutput;
        }
        if ('attr' in obj) {
            message.attr = obj.attr.map((obj) => tensorflow.OpDef.AttrDef.decodeJson(obj));
        }
        if ('deprecation' in obj) {
            message.deprecation = tensorflow.OpDeprecation.decodeJson(obj.deprecation);
        }
        if ('summary' in obj) {
            message.summary = obj.summary;
        }
        if ('description' in obj) {
            message.description = obj.description;
        }
        if ('isCommutative' in obj) {
            message.is_commutative = obj.isCommutative;
        }
        if ('isAggregate' in obj) {
            message.is_aggregate = obj.isAggregate;
        }
        if ('isStateful' in obj) {
            message.is_stateful = obj.isStateful;
        }
        if ('allowsUninitializedInput' in obj) {
            message.allows_uninitialized_input = obj.allowsUninitializedInput;
        }
        if ('isDistributedCommunication' in obj) {
            message.is_distributed_communication = obj.isDistributedCommunication;
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

    static decodeJson(obj) {
        const message = new tensorflow.OpDef.ArgDef();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('description' in obj) {
            message.description = obj.description;
        }
        if ('type' in obj) {
            message.type = typeof obj.type === 'string' ? tensorflow.DataType[obj.type] : obj.type;
        }
        if ('typeAttr' in obj) {
            message.type_attr = obj.typeAttr;
        }
        if ('numberAttr' in obj) {
            message.number_attr = obj.numberAttr;
        }
        if ('typeListAttr' in obj) {
            message.type_list_attr = obj.typeListAttr;
        }
        if ('handleData' in obj) {
            message.handle_data = obj.handleData.map((obj) => tensorflow.ResourceHandleProto.DtypeAndShape.decodeJson(obj));
        }
        if ('isRef' in obj) {
            message.is_ref = obj.isRef;
        }
        if ('experimentalFullType' in obj) {
            message.experimental_full_type = tensorflow.FullTypeDef.decodeJson(obj.experimentalFullType);
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

    static decodeJson(obj) {
        const message = new tensorflow.OpDef.AttrDef();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('type' in obj) {
            message.type = obj.type;
        }
        if ('defaultValue' in obj) {
            message.default_value = tensorflow.AttrValue.decodeJson(obj.defaultValue);
        }
        if ('description' in obj) {
            message.description = obj.description;
        }
        if ('hasMinimum' in obj) {
            message.has_minimum = obj.hasMinimum;
        }
        if ('minimum' in obj) {
            message.minimum = BigInt(obj.minimum);
        }
        if ('allowedValues' in obj) {
            message.allowed_values = tensorflow.AttrValue.decodeJson(obj.allowedValues);
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

    static decodeJson(obj) {
        const message = new tensorflow.OpDeprecation();
        if ('version' in obj) {
            message.version = Number(obj.version);
        }
        if ('explanation' in obj) {
            message.explanation = obj.explanation;
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

    static decodeJson(obj) {
        const message = new tensorflow.OpList();
        if ('op' in obj) {
            message.op = obj.op.map((obj) => tensorflow.OpDef.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.GraphDebugInfo();
        if ('files' in obj) {
            message.files = obj.files;
        }
        if ('framesById' in obj) {
            for (const [key, value] of Object.entries(obj.framesById)) {
                message.frames_by_id[key] = tensorflow.GraphDebugInfo.FileLineCol.decodeJson(value);
            }
        }
        if ('tracesById' in obj) {
            for (const [key, value] of Object.entries(obj.tracesById)) {
                message.traces_by_id[key] = tensorflow.GraphDebugInfo.StackTrace.decodeJson(value);
            }
        }
        if ('traces' in obj) {
            for (const [key, value] of Object.entries(obj.traces)) {
                message.traces[key] = tensorflow.GraphDebugInfo.StackTrace.decodeJson(value);
            }
        }
        if ('nameToTraceId' in obj) {
            for (const [key, value] of Object.entries(obj.nameToTraceId)) {
                message.name_to_trace_id[key] = value;
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

    static decodeJson(obj) {
        const message = new tensorflow.GraphDebugInfo.FileLineCol();
        if ('fileIndex' in obj) {
            message.file_index = Number(obj.fileIndex);
        }
        if ('line' in obj) {
            message.line = Number(obj.line);
        }
        if ('col' in obj) {
            message.col = Number(obj.col);
        }
        if ('func' in obj) {
            message.func = obj.func;
        }
        if ('code' in obj) {
            message.code = obj.code;
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

    static decodeJson(obj) {
        const message = new tensorflow.GraphDebugInfo.StackTrace();
        if ('fileLineCols' in obj) {
            message.file_line_cols = obj.fileLineCols.map((obj) => tensorflow.GraphDebugInfo.FileLineCol.decodeJson(obj));
        }
        if ('frameId' in obj) {
            message.frame_id = obj.frameId.map((obj) => BigInt(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.VersionDef();
        if ('producer' in obj) {
            message.producer = Number(obj.producer);
        }
        if ('minConsumer' in obj) {
            message.min_consumer = Number(obj.minConsumer);
        }
        if ('badConsumers' in obj) {
            message.bad_consumers = obj.badConsumers.map((obj) => Number(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedObjectGraph();
        if ('nodes' in obj) {
            message.nodes = obj.nodes.map((obj) => tensorflow.SavedObject.decodeJson(obj));
        }
        if ('concreteFunctions' in obj) {
            for (const [key, value] of Object.entries(obj.concreteFunctions)) {
                message.concrete_functions[key] = tensorflow.SavedConcreteFunction.decodeJson(value);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedObject();
        if ('children' in obj) {
            message.children = obj.children.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeJson(obj));
        }
        if ('dependencies' in obj) {
            message.dependencies = obj.dependencies.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeJson(obj));
        }
        if ('slotVariables' in obj) {
            message.slot_variables = obj.slotVariables.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeJson(obj));
        }
        if ('userObject' in obj) {
            message.user_object = tensorflow.SavedUserObject.decodeJson(obj.userObject);
        }
        if ('asset' in obj) {
            message.asset = tensorflow.SavedAsset.decodeJson(obj.asset);
        }
        if ('function' in obj) {
            message.function = tensorflow.SavedFunction.decodeJson(obj.function);
        }
        if ('variable' in obj) {
            message.variable = tensorflow.SavedVariable.decodeJson(obj.variable);
        }
        if ('bareConcreteFunction' in obj) {
            message.bare_concrete_function = tensorflow.SavedBareConcreteFunction.decodeJson(obj.bareConcreteFunction);
        }
        if ('constant' in obj) {
            message.constant = tensorflow.SavedConstant.decodeJson(obj.constant);
        }
        if ('resource' in obj) {
            message.resource = tensorflow.SavedResource.decodeJson(obj.resource);
        }
        if ('capturedTensor' in obj) {
            message.captured_tensor = tensorflow.CapturedTensor.decodeJson(obj.capturedTensor);
        }
        if ('saveableObjects' in obj) {
            for (const [key, value] of Object.entries(obj.saveableObjects)) {
                message.saveable_objects[key] = tensorflow.SaveableObject.decodeJson(value);
            }
        }
        if ('registeredName' in obj) {
            message.registered_name = obj.registeredName;
        }
        if ('serializedUserProto' in obj) {
            message.serialized_user_proto = google.protobuf.Any.decodeJson(obj.serializedUserProto);
        }
        if ('registeredSaver' in obj) {
            message.registered_saver = obj.registeredSaver;
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedUserObject();
        if ('identifier' in obj) {
            message.identifier = obj.identifier;
        }
        if ('version' in obj) {
            message.version = tensorflow.VersionDef.decodeJson(obj.version);
        }
        if ('metadata' in obj) {
            message.metadata = obj.metadata;
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedAsset();
        if ('assetFileDefIndex' in obj) {
            message.asset_file_def_index = Number(obj.assetFileDefIndex);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedFunction();
        if ('concreteFunctions' in obj) {
            message.concrete_functions = obj.concreteFunctions;
        }
        if ('functionSpec' in obj) {
            message.function_spec = tensorflow.FunctionSpec.decodeJson(obj.functionSpec);
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

    static decodeJson(obj) {
        const message = new tensorflow.CapturedTensor();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('concreteFunction' in obj) {
            message.concrete_function = obj.concreteFunction;
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedConcreteFunction();
        if ('boundInputs' in obj) {
            message.bound_inputs = obj.boundInputs.map((obj) => Number(obj));
        }
        if ('canonicalizedInputSignature' in obj) {
            message.canonicalized_input_signature = tensorflow.StructuredValue.decodeJson(obj.canonicalizedInputSignature);
        }
        if ('outputSignature' in obj) {
            message.output_signature = tensorflow.StructuredValue.decodeJson(obj.outputSignature);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedBareConcreteFunction();
        if ('concreteFunctionName' in obj) {
            message.concrete_function_name = obj.concreteFunctionName;
        }
        if ('argumentKeywords' in obj) {
            message.argument_keywords = obj.argumentKeywords;
        }
        if ('allowedPositionalArguments' in obj) {
            message.allowed_positional_arguments = BigInt(obj.allowedPositionalArguments);
        }
        if ('functionSpec' in obj) {
            message.function_spec = tensorflow.FunctionSpec.decodeJson(obj.functionSpec);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedConstant();
        if ('operation' in obj) {
            message.operation = obj.operation;
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedVariable();
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('trainable' in obj) {
            message.trainable = obj.trainable;
        }
        if ('synchronization' in obj) {
            message.synchronization = typeof obj.synchronization === 'string' ? tensorflow.VariableSynchronization[obj.synchronization] : obj.synchronization;
        }
        if ('aggregation' in obj) {
            message.aggregation = typeof obj.aggregation === 'string' ? tensorflow.VariableAggregation[obj.aggregation] : obj.aggregation;
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('device' in obj) {
            message.device = obj.device;
        }
        if ('experimentalDistributedVariableComponents' in obj) {
            message.experimental_distributed_variable_components = obj.experimentalDistributedVariableComponents.map((obj) => tensorflow.SavedVariable.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.FunctionSpec();
        if ('fullargspec' in obj) {
            message.fullargspec = tensorflow.StructuredValue.decodeJson(obj.fullargspec);
        }
        if ('isMethod' in obj) {
            message.is_method = obj.isMethod;
        }
        if ('inputSignature' in obj) {
            message.input_signature = tensorflow.StructuredValue.decodeJson(obj.inputSignature);
        }
        if ('jitCompile' in obj) {
            message.jit_compile = typeof obj.jitCompile === 'string' ? tensorflow.FunctionSpec.JitCompile[obj.jitCompile] : obj.jitCompile;
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedResource();
        if ('device' in obj) {
            message.device = obj.device;
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

    static decodeJson(obj) {
        const message = new tensorflow.SaveableObject();
        if ('saveFunction' in obj) {
            message.save_function = Number(obj.saveFunction);
        }
        if ('restoreFunction' in obj) {
            message.restore_function = Number(obj.restoreFunction);
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

    static decodeJson(obj) {
        const message = new tensorflow.VariableDef();
        if ('variableName' in obj) {
            message.variable_name = obj.variableName;
        }
        if ('initialValueName' in obj) {
            message.initial_value_name = obj.initialValueName;
        }
        if ('initializerName' in obj) {
            message.initializer_name = obj.initializerName;
        }
        if ('snapshotName' in obj) {
            message.snapshot_name = obj.snapshotName;
        }
        if ('saveSliceInfoDef' in obj) {
            message.save_slice_info_def = tensorflow.SaveSliceInfoDef.decodeJson(obj.saveSliceInfoDef);
        }
        if ('isResource' in obj) {
            message.is_resource = obj.isResource;
        }
        if ('trainable' in obj) {
            message.trainable = obj.trainable;
        }
        if ('synchronization' in obj) {
            message.synchronization = typeof obj.synchronization === 'string' ? tensorflow.VariableSynchronization[obj.synchronization] : obj.synchronization;
        }
        if ('aggregation' in obj) {
            message.aggregation = typeof obj.aggregation === 'string' ? tensorflow.VariableAggregation[obj.aggregation] : obj.aggregation;
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

    static decodeJson(obj) {
        const message = new tensorflow.SaveSliceInfoDef();
        if ('fullName' in obj) {
            message.full_name = obj.fullName;
        }
        if ('fullShape' in obj) {
            message.full_shape = obj.fullShape.map((obj) => BigInt(obj));
        }
        if ('varOffset' in obj) {
            message.var_offset = obj.varOffset.map((obj) => BigInt(obj));
        }
        if ('varShape' in obj) {
            message.var_shape = obj.varShape.map((obj) => BigInt(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.StructuredValue();
        if ('noneValue' in obj) {
            message.none_value = tensorflow.NoneValue.decodeJson(obj.noneValue);
        }
        if ('float64Value' in obj) {
            message.float64_value = Number(obj.float64Value);
        }
        if ('int64Value' in obj) {
            message.int64_value = BigInt(obj.int64Value);
        }
        if ('stringValue' in obj) {
            message.string_value = obj.stringValue;
        }
        if ('boolValue' in obj) {
            message.bool_value = obj.boolValue;
        }
        if ('tensorShapeValue' in obj) {
            message.tensor_shape_value = tensorflow.TensorShapeProto.decodeJson(obj.tensorShapeValue);
        }
        if ('tensorDtypeValue' in obj) {
            message.tensor_dtype_value = typeof obj.tensorDtypeValue === 'string' ? tensorflow.DataType[obj.tensorDtypeValue] : obj.tensorDtypeValue;
        }
        if ('tensorSpecValue' in obj) {
            message.tensor_spec_value = tensorflow.TensorSpecProto.decodeJson(obj.tensorSpecValue);
        }
        if ('typeSpecValue' in obj) {
            message.type_spec_value = tensorflow.TypeSpecProto.decodeJson(obj.typeSpecValue);
        }
        if ('boundedTensorSpecValue' in obj) {
            message.bounded_tensor_spec_value = tensorflow.BoundedTensorSpecProto.decodeJson(obj.boundedTensorSpecValue);
        }
        if ('listValue' in obj) {
            message.list_value = tensorflow.ListValue.decodeJson(obj.listValue);
        }
        if ('tupleValue' in obj) {
            message.tuple_value = tensorflow.TupleValue.decodeJson(obj.tupleValue);
        }
        if ('dictValue' in obj) {
            message.dict_value = tensorflow.DictValue.decodeJson(obj.dictValue);
        }
        if ('namedTupleValue' in obj) {
            message.named_tuple_value = tensorflow.NamedTupleValue.decodeJson(obj.namedTupleValue);
        }
        if ('tensorValue' in obj) {
            message.tensor_value = tensorflow.TensorProto.decodeJson(obj.tensorValue);
        }
        if ('numpyValue' in obj) {
            message.numpy_value = tensorflow.TensorProto.decodeJson(obj.numpyValue);
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

    static decodeJson() {
        const message = new tensorflow.NoneValue();
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

    static decodeJson(obj) {
        const message = new tensorflow.ListValue();
        if ('values' in obj) {
            message.values = obj.values.map((obj) => tensorflow.StructuredValue.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.TupleValue();
        if ('values' in obj) {
            message.values = obj.values.map((obj) => tensorflow.StructuredValue.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.DictValue();
        if ('fields' in obj) {
            for (const [key, value] of Object.entries(obj.fields)) {
                message.fields[key] = tensorflow.StructuredValue.decodeJson(value);
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

    static decodeJson(obj) {
        const message = new tensorflow.PairValue();
        if ('key' in obj) {
            message.key = obj.key;
        }
        if ('value' in obj) {
            message.value = tensorflow.StructuredValue.decodeJson(obj.value);
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

    static decodeJson(obj) {
        const message = new tensorflow.NamedTupleValue();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('values' in obj) {
            message.values = obj.values.map((obj) => tensorflow.PairValue.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorSpecProto();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
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

    static decodeJson(obj) {
        const message = new tensorflow.BoundedTensorSpecProto();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('minimum' in obj) {
            message.minimum = tensorflow.TensorProto.decodeJson(obj.minimum);
        }
        if ('maximum' in obj) {
            message.maximum = tensorflow.TensorProto.decodeJson(obj.maximum);
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

    static decodeJson(obj) {
        const message = new tensorflow.TypeSpecProto();
        if ('typeSpecClass' in obj) {
            message.type_spec_class = typeof obj.typeSpecClass === 'string' ? tensorflow.TypeSpecProto.TypeSpecClass[obj.typeSpecClass] : obj.typeSpecClass;
        }
        if ('typeState' in obj) {
            message.type_state = tensorflow.StructuredValue.decodeJson(obj.typeState);
        }
        if ('typeSpecClassName' in obj) {
            message.type_spec_class_name = obj.typeSpecClassName;
        }
        if ('numFlatComponents' in obj) {
            message.num_flat_components = Number(obj.numFlatComponents);
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

    static decodeJson(obj) {
        const message = new tensorflow.TrackableObjectGraph();
        if ('nodes' in obj) {
            message.nodes = obj.nodes.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject();
        if ('children' in obj) {
            message.children = obj.children.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeJson(obj));
        }
        if ('attributes' in obj) {
            message.attributes = obj.attributes.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decodeJson(obj));
        }
        if ('slotVariables' in obj) {
            message.slot_variables = obj.slotVariables.map((obj) => tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeJson(obj));
        }
        if ('registeredSaver' in obj) {
            message.registered_saver = tensorflow.RegisteredSaver.decodeJson(obj.registeredSaver);
        }
        if ('hasCheckpointValues' in obj) {
            message.has_checkpoint_values = google.protobuf.BoolValue.decodeJson(obj.hasCheckpointValues);
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

    static decodeJson(obj) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
        if ('nodeId' in obj) {
            message.node_id = Number(obj.nodeId);
        }
        if ('localName' in obj) {
            message.local_name = obj.localName;
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

    static decodeJson(obj) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('fullName' in obj) {
            message.full_name = obj.fullName;
        }
        if ('checkpointKey' in obj) {
            message.checkpoint_key = obj.checkpointKey;
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

    static decodeJson(obj) {
        const message = new tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
        if ('originalVariableNodeId' in obj) {
            message.original_variable_node_id = Number(obj.originalVariableNodeId);
        }
        if ('slotName' in obj) {
            message.slot_name = obj.slotName;
        }
        if ('slotVariableNodeId' in obj) {
            message.slot_variable_node_id = Number(obj.slotVariableNodeId);
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

    static decodeJson(obj) {
        const message = new tensorflow.RegisteredSaver();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('objectName' in obj) {
            message.object_name = obj.objectName;
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

    static decodeJson(obj) {
        const message = new tensorflow.SaverDef();
        if ('filenameTensorName' in obj) {
            message.filename_tensor_name = obj.filenameTensorName;
        }
        if ('saveTensorName' in obj) {
            message.save_tensor_name = obj.saveTensorName;
        }
        if ('restoreOpName' in obj) {
            message.restore_op_name = obj.restoreOpName;
        }
        if ('maxToKeep' in obj) {
            message.max_to_keep = Number(obj.maxToKeep);
        }
        if ('sharded' in obj) {
            message.sharded = obj.sharded;
        }
        if ('keepCheckpointEveryNHours' in obj) {
            message.keep_checkpoint_every_n_hours = Number(obj.keepCheckpointEveryNHours);
        }
        if ('version' in obj) {
            message.version = typeof obj.version === 'string' ? tensorflow.SaverDef.CheckpointFormatVersion[obj.version] : obj.version;
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

    static decodeJson(obj) {
        const message = new tensorflow.BundleHeaderProto();
        if ('numShards' in obj) {
            message.num_shards = Number(obj.numShards);
        }
        if ('endianness' in obj) {
            message.endianness = typeof obj.endianness === 'string' ? tensorflow.BundleHeaderProto.Endianness[obj.endianness] : obj.endianness;
        }
        if ('version' in obj) {
            message.version = tensorflow.VersionDef.decodeJson(obj.version);
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

    static decodeJson(obj) {
        const message = new tensorflow.BundleEntryProto();
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('shardId' in obj) {
            message.shard_id = Number(obj.shardId);
        }
        if ('offset' in obj) {
            message.offset = BigInt(obj.offset);
        }
        if ('size' in obj) {
            message.size = BigInt(obj.size);
        }
        if ('crc32c' in obj) {
            message.crc32c = Number(obj.crc32c);
        }
        if ('slices' in obj) {
            message.slices = obj.slices.map((obj) => tensorflow.TensorSliceProto.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorSliceProto();
        if ('extent' in obj) {
            message.extent = obj.extent.map((obj) => tensorflow.TensorSliceProto.Extent.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorSliceProto.Extent();
        if ('start' in obj) {
            message.start = BigInt(obj.start);
        }
        if ('length' in obj) {
            message.length = BigInt(obj.length);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedSliceMeta();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('type' in obj) {
            message.type = typeof obj.type === 'string' ? tensorflow.DataType[obj.type] : obj.type;
        }
        if ('slice' in obj) {
            message.slice = obj.slice.map((obj) => tensorflow.TensorSliceProto.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedTensorSliceMeta();
        if ('tensor' in obj) {
            message.tensor = obj.tensor.map((obj) => tensorflow.SavedSliceMeta.decodeJson(obj));
        }
        if ('versions' in obj) {
            message.versions = tensorflow.VersionDef.decodeJson(obj.versions);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedSlice();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('slice' in obj) {
            message.slice = tensorflow.TensorSliceProto.decodeJson(obj.slice);
        }
        if ('data' in obj) {
            message.data = tensorflow.TensorProto.decodeJson(obj.data);
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

    static decodeJson(obj) {
        const message = new tensorflow.SavedTensorSlices();
        if ('meta' in obj) {
            message.meta = tensorflow.SavedTensorSliceMeta.decodeJson(obj.meta);
        }
        if ('data' in obj) {
            message.data = tensorflow.SavedSlice.decodeJson(obj.data);
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

    static decodeJson(obj) {
        const message = new tensorflow.Event();
        if ('wallTime' in obj) {
            message.wall_time = Number(obj.wallTime);
        }
        if ('step' in obj) {
            message.step = BigInt(obj.step);
        }
        if ('fileVersion' in obj) {
            message.file_version = obj.fileVersion;
        }
        if ('graphDef' in obj) {
            message.graph_def = typeof source === 'string' ? Uint8Array.from(atob(obj.graphDef), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.graphDef);
        }
        if ('summary' in obj) {
            message.summary = tensorflow.Summary.decodeJson(obj.summary);
        }
        if ('logMessage' in obj) {
            message.log_message = tensorflow.LogMessage.decodeJson(obj.logMessage);
        }
        if ('sessionLog' in obj) {
            message.session_log = tensorflow.SessionLog.decodeJson(obj.sessionLog);
        }
        if ('taggedRunMetadata' in obj) {
            message.tagged_run_metadata = tensorflow.TaggedRunMetadata.decodeJson(obj.taggedRunMetadata);
        }
        if ('metaGraphDef' in obj) {
            message.meta_graph_def = typeof source === 'string' ? Uint8Array.from(atob(obj.metaGraphDef), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.metaGraphDef);
        }
        if ('sourceMetadata' in obj) {
            message.source_metadata = tensorflow.SourceMetadata.decodeJson(obj.sourceMetadata);
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

    static decodeJson(obj) {
        const message = new tensorflow.SourceMetadata();
        if ('writer' in obj) {
            message.writer = obj.writer;
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

    static decodeJson(obj) {
        const message = new tensorflow.LogMessage();
        if ('level' in obj) {
            message.level = typeof obj.level === 'string' ? tensorflow.LogMessage.Level[obj.level] : obj.level;
        }
        if ('message' in obj) {
            message.message = obj.message;
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

    static decodeJson(obj) {
        const message = new tensorflow.SessionLog();
        if ('status' in obj) {
            message.status = typeof obj.status === 'string' ? tensorflow.SessionLog.SessionStatus[obj.status] : obj.status;
        }
        if ('checkpointPath' in obj) {
            message.checkpoint_path = obj.checkpointPath;
        }
        if ('msg' in obj) {
            message.msg = obj.msg;
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

    static decodeJson(obj) {
        const message = new tensorflow.TaggedRunMetadata();
        if ('tag' in obj) {
            message.tag = obj.tag;
        }
        if ('runMetadata' in obj) {
            message.run_metadata = typeof source === 'string' ? Uint8Array.from(atob(obj.runMetadata), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.runMetadata);
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

    static decodeJson(obj) {
        const message = new tensorflow.WatchdogConfig();
        if ('timeoutMs' in obj) {
            message.timeout_ms = BigInt(obj.timeoutMs);
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

    static decodeJson(obj) {
        const message = new tensorflow.RequestedExitCode();
        if ('exitCode' in obj) {
            message.exit_code = Number(obj.exitCode);
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

    static decodeJson(obj) {
        const message = new tensorflow.WorkerHeartbeatRequest();
        if ('shutdownMode' in obj) {
            message.shutdown_mode = typeof obj.shutdownMode === 'string' ? tensorflow.WorkerShutdownMode[obj.shutdownMode] : obj.shutdownMode;
        }
        if ('watchdogConfig' in obj) {
            message.watchdog_config = tensorflow.WatchdogConfig.decodeJson(obj.watchdogConfig);
        }
        if ('exitCode' in obj) {
            message.exit_code = tensorflow.RequestedExitCode.decodeJson(obj.exitCode);
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

    static decodeJson(obj) {
        const message = new tensorflow.WorkerHeartbeatResponse();
        if ('healthStatus' in obj) {
            message.health_status = typeof obj.healthStatus === 'string' ? tensorflow.WorkerHealth[obj.healthStatus] : obj.healthStatus;
        }
        if ('workerLog' in obj) {
            message.worker_log = obj.workerLog.map((obj) => tensorflow.Event.decodeJson(obj));
        }
        if ('hostname' in obj) {
            message.hostname = obj.hostname;
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

    static decodeJson(obj) {
        const message = new tensorflow.SummaryDescription();
        if ('typeHint' in obj) {
            message.type_hint = obj.typeHint;
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

    static decodeJson(obj) {
        const message = new tensorflow.SummaryMetadata();
        if ('pluginData' in obj) {
            message.plugin_data = tensorflow.SummaryMetadata.PluginData.decodeJson(obj.pluginData);
        }
        if ('displayName' in obj) {
            message.display_name = obj.displayName;
        }
        if ('summaryDescription' in obj) {
            message.summary_description = obj.summaryDescription;
        }
        if ('dataClass' in obj) {
            message.data_class = typeof obj.dataClass === 'string' ? tensorflow.DataClass[obj.dataClass] : obj.dataClass;
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

    static decodeJson(obj) {
        const message = new tensorflow.SummaryMetadata.PluginData();
        if ('pluginName' in obj) {
            message.plugin_name = obj.pluginName;
        }
        if ('content' in obj) {
            message.content = typeof source === 'string' ? Uint8Array.from(atob(obj.content), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.content);
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

    static decodeJson(obj) {
        const message = new tensorflow.Summary();
        if ('value' in obj) {
            message.value = obj.value.map((obj) => tensorflow.Summary.Value.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.Summary.Image();
        if ('height' in obj) {
            message.height = Number(obj.height);
        }
        if ('width' in obj) {
            message.width = Number(obj.width);
        }
        if ('colorspace' in obj) {
            message.colorspace = Number(obj.colorspace);
        }
        if ('encodedImageString' in obj) {
            message.encoded_image_string = typeof source === 'string' ? Uint8Array.from(atob(obj.encodedImageString), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.encodedImageString);
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

    static decodeJson(obj) {
        const message = new tensorflow.Summary.Audio();
        if ('sampleRate' in obj) {
            message.sample_rate = Number(obj.sampleRate);
        }
        if ('numChannels' in obj) {
            message.num_channels = BigInt(obj.numChannels);
        }
        if ('lengthFrames' in obj) {
            message.length_frames = BigInt(obj.lengthFrames);
        }
        if ('encodedAudioString' in obj) {
            message.encoded_audio_string = typeof source === 'string' ? Uint8Array.from(atob(obj.encodedAudioString), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.encodedAudioString);
        }
        if ('contentType' in obj) {
            message.content_type = obj.contentType;
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

    static decodeJson(obj) {
        const message = new tensorflow.Summary.Value();
        if ('nodeName' in obj) {
            message.node_name = obj.nodeName;
        }
        if ('tag' in obj) {
            message.tag = obj.tag;
        }
        if ('metadata' in obj) {
            message.metadata = tensorflow.SummaryMetadata.decodeJson(obj.metadata);
        }
        if ('simpleValue' in obj) {
            message.simple_value = Number(obj.simpleValue);
        }
        if ('obsoleteOldStyleHistogram' in obj) {
            message.obsolete_old_style_histogram = typeof source === 'string' ? Uint8Array.from(atob(obj.obsoleteOldStyleHistogram), (c) => c.charCodeAt(0)) : Uint8Array.from(obj.obsoleteOldStyleHistogram);
        }
        if ('image' in obj) {
            message.image = tensorflow.Summary.Image.decodeJson(obj.image);
        }
        if ('histo' in obj) {
            message.histo = tensorflow.HistogramProto.decodeJson(obj.histo);
        }
        if ('audio' in obj) {
            message.audio = tensorflow.Summary.Audio.decodeJson(obj.audio);
        }
        if ('tensor' in obj) {
            message.tensor = tensorflow.TensorProto.decodeJson(obj.tensor);
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

    static decodeJson(obj) {
        const message = new tensorflow.HistogramProto();
        if ('min' in obj) {
            message.min = Number(obj.min);
        }
        if ('max' in obj) {
            message.max = Number(obj.max);
        }
        if ('num' in obj) {
            message.num = Number(obj.num);
        }
        if ('sum' in obj) {
            message.sum = Number(obj.sum);
        }
        if ('sumSquares' in obj) {
            message.sum_squares = Number(obj.sumSquares);
        }
        if ('bucketLimit' in obj) {
            message.bucket_limit = obj.bucketLimit.map((obj) => Number(obj));
        }
        if ('bucket' in obj) {
            message.bucket = obj.bucket.map((obj) => Number(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.GPUOptions();
        if ('perProcessGpuMemoryFraction' in obj) {
            message.per_process_gpu_memory_fraction = Number(obj.perProcessGpuMemoryFraction);
        }
        if ('allowGrowth' in obj) {
            message.allow_growth = obj.allowGrowth;
        }
        if ('allocatorType' in obj) {
            message.allocator_type = obj.allocatorType;
        }
        if ('deferredDeletionBytes' in obj) {
            message.deferred_deletion_bytes = BigInt(obj.deferredDeletionBytes);
        }
        if ('visibleDeviceList' in obj) {
            message.visible_device_list = obj.visibleDeviceList;
        }
        if ('pollingActiveDelayUsecs' in obj) {
            message.polling_active_delay_usecs = Number(obj.pollingActiveDelayUsecs);
        }
        if ('pollingInactiveDelayMsecs' in obj) {
            message.polling_inactive_delay_msecs = Number(obj.pollingInactiveDelayMsecs);
        }
        if ('forceGpuCompatible' in obj) {
            message.force_gpu_compatible = obj.forceGpuCompatible;
        }
        if ('experimental' in obj) {
            message.experimental = tensorflow.GPUOptions.Experimental.decodeJson(obj.experimental);
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

    static decodeJson(obj) {
        const message = new tensorflow.GPUOptions.Experimental();
        if ('virtualDevices' in obj) {
            message.virtual_devices = obj.virtualDevices.map((obj) => tensorflow.GPUOptions.Experimental.VirtualDevices.decodeJson(obj));
        }
        if ('numVirtualDevicesPerGpu' in obj) {
            message.num_virtual_devices_per_gpu = Number(obj.numVirtualDevicesPerGpu);
        }
        if ('useUnifiedMemory' in obj) {
            message.use_unified_memory = obj.useUnifiedMemory;
        }
        if ('numDevToDevCopyStreams' in obj) {
            message.num_dev_to_dev_copy_streams = Number(obj.numDevToDevCopyStreams);
        }
        if ('collectiveRingOrder' in obj) {
            message.collective_ring_order = obj.collectiveRingOrder;
        }
        if ('timestampedAllocator' in obj) {
            message.timestamped_allocator = obj.timestampedAllocator;
        }
        if ('kernelTrackerMaxInterval' in obj) {
            message.kernel_tracker_max_interval = Number(obj.kernelTrackerMaxInterval);
        }
        if ('kernelTrackerMaxBytes' in obj) {
            message.kernel_tracker_max_bytes = Number(obj.kernelTrackerMaxBytes);
        }
        if ('kernelTrackerMaxPending' in obj) {
            message.kernel_tracker_max_pending = Number(obj.kernelTrackerMaxPending);
        }
        if ('internalFragmentationFraction' in obj) {
            message.internal_fragmentation_fraction = Number(obj.internalFragmentationFraction);
        }
        if ('useCudaMallocAsync' in obj) {
            message.use_cuda_malloc_async = obj.useCudaMallocAsync;
        }
        if ('disallowRetryOnAllocationFailure' in obj) {
            message.disallow_retry_on_allocation_failure = obj.disallowRetryOnAllocationFailure;
        }
        if ('gpuHostMemLimitInMb' in obj) {
            message.gpu_host_mem_limit_in_mb = Number(obj.gpuHostMemLimitInMb);
        }
        if ('gpuHostMemDisallowGrowth' in obj) {
            message.gpu_host_mem_disallow_growth = obj.gpuHostMemDisallowGrowth;
        }
        if ('gpuSystemMemorySizeInMb' in obj) {
            message.gpu_system_memory_size_in_mb = Number(obj.gpuSystemMemorySizeInMb);
        }
        if ('populatePjrtGpuClientCreationInfo' in obj) {
            message.populate_pjrt_gpu_client_creation_info = obj.populatePjrtGpuClientCreationInfo;
        }
        if ('nodeId' in obj) {
            message.node_id = Number(obj.nodeId);
        }
        if ('streamMergeOptions' in obj) {
            message.stream_merge_options = tensorflow.GPUOptions.Experimental.StreamMergeOptions.decodeJson(obj.streamMergeOptions);
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

    static decodeJson(obj) {
        const message = new tensorflow.GPUOptions.Experimental.VirtualDevices();
        if ('memoryLimitMb' in obj) {
            message.memory_limit_mb = obj.memoryLimitMb.map((obj) => Number(obj));
        }
        if ('priority' in obj) {
            message.priority = obj.priority.map((obj) => Number(obj));
        }
        if ('deviceOrdinal' in obj) {
            message.device_ordinal = obj.deviceOrdinal.map((obj) => Number(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.GPUOptions.Experimental.StreamMergeOptions();
        if ('mergeHostToDeviceStream' in obj) {
            message.merge_host_to_device_stream = obj.mergeHostToDeviceStream;
        }
        if ('mergeDeviceToHostStream' in obj) {
            message.merge_device_to_host_stream = obj.mergeDeviceToHostStream;
        }
        if ('mergeDeviceToDeviceStream' in obj) {
            message.merge_device_to_device_stream = obj.mergeDeviceToDeviceStream;
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

    static decodeJson(obj) {
        const message = new tensorflow.OptimizerOptions();
        if ('doCommonSubexpressionElimination' in obj) {
            message.do_common_subexpression_elimination = obj.doCommonSubexpressionElimination;
        }
        if ('doConstantFolding' in obj) {
            message.do_constant_folding = obj.doConstantFolding;
        }
        if ('maxFoldedConstantInBytes' in obj) {
            message.max_folded_constant_in_bytes = BigInt(obj.maxFoldedConstantInBytes);
        }
        if ('doFunctionInlining' in obj) {
            message.do_function_inlining = obj.doFunctionInlining;
        }
        if ('optLevel' in obj) {
            message.opt_level = typeof obj.optLevel === 'string' ? tensorflow.OptimizerOptions.Level[obj.optLevel] : obj.optLevel;
        }
        if ('globalJitLevel' in obj) {
            message.global_jit_level = typeof obj.globalJitLevel === 'string' ? tensorflow.OptimizerOptions.GlobalJitLevel[obj.globalJitLevel] : obj.globalJitLevel;
        }
        if ('cpuGlobalJit' in obj) {
            message.cpu_global_jit = obj.cpuGlobalJit;
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

    static decodeJson(obj) {
        const message = new tensorflow.GraphOptions();
        if ('enableRecvScheduling' in obj) {
            message.enable_recv_scheduling = obj.enableRecvScheduling;
        }
        if ('optimizerOptions' in obj) {
            message.optimizer_options = tensorflow.OptimizerOptions.decodeJson(obj.optimizerOptions);
        }
        if ('buildCostModel' in obj) {
            message.build_cost_model = BigInt(obj.buildCostModel);
        }
        if ('buildCostModelAfter' in obj) {
            message.build_cost_model_after = BigInt(obj.buildCostModelAfter);
        }
        if ('inferShapes' in obj) {
            message.infer_shapes = obj.inferShapes;
        }
        if ('placePrunedGraph' in obj) {
            message.place_pruned_graph = obj.placePrunedGraph;
        }
        if ('enableBfloat16Sendrecv' in obj) {
            message.enable_bfloat16_sendrecv = obj.enableBfloat16Sendrecv;
        }
        if ('timelineStep' in obj) {
            message.timeline_step = Number(obj.timelineStep);
        }
        if ('rewriteOptions' in obj) {
            message.rewrite_options = tensorflow.RewriterConfig.decodeJson(obj.rewriteOptions);
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

    static decodeJson(obj) {
        const message = new tensorflow.ThreadPoolOptionProto();
        if ('numThreads' in obj) {
            message.num_threads = Number(obj.numThreads);
        }
        if ('globalName' in obj) {
            message.global_name = obj.globalName;
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

    static decodeJson(obj) {
        const message = new tensorflow.SessionMetadata();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('version' in obj) {
            message.version = BigInt(obj.version);
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
                case 18:
                    message.pluggable_device_options = tensorflow.GPUOptions.decode(reader, reader.uint32());
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
                case "pluggable_device_options":
                    message.pluggable_device_options = tensorflow.GPUOptions.decodeText(reader);
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

    static decodeJson(obj) {
        const message = new tensorflow.ConfigProto();
        if ('deviceCount' in obj) {
            for (const [key, value] of Object.entries(obj.deviceCount)) {
                message.device_count[key] = value;
            }
        }
        if ('intraOpParallelismThreads' in obj) {
            message.intra_op_parallelism_threads = Number(obj.intraOpParallelismThreads);
        }
        if ('interOpParallelismThreads' in obj) {
            message.inter_op_parallelism_threads = Number(obj.interOpParallelismThreads);
        }
        if ('usePerSessionThreads' in obj) {
            message.use_per_session_threads = obj.usePerSessionThreads;
        }
        if ('sessionInterOpThreadPool' in obj) {
            message.session_inter_op_thread_pool = obj.sessionInterOpThreadPool.map((obj) => tensorflow.ThreadPoolOptionProto.decodeJson(obj));
        }
        if ('placementPeriod' in obj) {
            message.placement_period = Number(obj.placementPeriod);
        }
        if ('deviceFilters' in obj) {
            message.device_filters = obj.deviceFilters;
        }
        if ('gpuOptions' in obj) {
            message.gpu_options = tensorflow.GPUOptions.decodeJson(obj.gpuOptions);
        }
        if ('pluggableDeviceOptions' in obj) {
            message.pluggable_device_options = tensorflow.GPUOptions.decodeJson(obj.pluggableDeviceOptions);
        }
        if ('allowSoftPlacement' in obj) {
            message.allow_soft_placement = obj.allowSoftPlacement;
        }
        if ('logDevicePlacement' in obj) {
            message.log_device_placement = obj.logDevicePlacement;
        }
        if ('graphOptions' in obj) {
            message.graph_options = tensorflow.GraphOptions.decodeJson(obj.graphOptions);
        }
        if ('operationTimeoutInMs' in obj) {
            message.operation_timeout_in_ms = BigInt(obj.operationTimeoutInMs);
        }
        if ('rpcOptions' in obj) {
            message.rpc_options = tensorflow.RPCOptions.decodeJson(obj.rpcOptions);
        }
        if ('clusterDef' in obj) {
            message.cluster_def = tensorflow.ClusterDef.decodeJson(obj.clusterDef);
        }
        if ('isolateSessionState' in obj) {
            message.isolate_session_state = obj.isolateSessionState;
        }
        if ('shareClusterDevicesInSession' in obj) {
            message.share_cluster_devices_in_session = obj.shareClusterDevicesInSession;
        }
        if ('experimental' in obj) {
            message.experimental = tensorflow.ConfigProto.Experimental.decodeJson(obj.experimental);
        }
        return message;
    }
};

tensorflow.ConfigProto.prototype.intra_op_parallelism_threads = 0;
tensorflow.ConfigProto.prototype.inter_op_parallelism_threads = 0;
tensorflow.ConfigProto.prototype.use_per_session_threads = false;
tensorflow.ConfigProto.prototype.placement_period = 0;
tensorflow.ConfigProto.prototype.gpu_options = null;
tensorflow.ConfigProto.prototype.pluggable_device_options = null;
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
                case 32:
                    message.tfrt_use_ifrt = reader.bool();
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
                case 33:
                    message.finalize_function_library_runtime = reader.bool();
                    break;
                case 34:
                    message.finalize_resource_manager = reader.bool();
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
                case "tfrt_use_ifrt":
                    message.tfrt_use_ifrt = reader.bool();
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
                case "finalize_function_library_runtime":
                    message.finalize_function_library_runtime = reader.bool();
                    break;
                case "finalize_resource_manager":
                    message.finalize_resource_manager = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new tensorflow.ConfigProto.Experimental();
        if ('collectiveGroupLeader' in obj) {
            message.collective_group_leader = obj.collectiveGroupLeader;
        }
        if ('executorType' in obj) {
            message.executor_type = obj.executorType;
        }
        if ('recvBufMaxChunk' in obj) {
            message.recv_buf_max_chunk = Number(obj.recvBufMaxChunk);
        }
        if ('useNumaAffinity' in obj) {
            message.use_numa_affinity = obj.useNumaAffinity;
        }
        if ('collectiveDeterministicSequentialExecution' in obj) {
            message.collective_deterministic_sequential_execution = obj.collectiveDeterministicSequentialExecution;
        }
        if ('collectiveNccl' in obj) {
            message.collective_nccl = obj.collectiveNccl;
        }
        if ('shareSessionStateInClusterspecPropagation' in obj) {
            message.share_session_state_in_clusterspec_propagation = obj.shareSessionStateInClusterspecPropagation;
        }
        if ('disableThreadSpinning' in obj) {
            message.disable_thread_spinning = obj.disableThreadSpinning;
        }
        if ('shareClusterDevicesInSession' in obj) {
            message.share_cluster_devices_in_session = obj.shareClusterDevicesInSession;
        }
        if ('sessionMetadata' in obj) {
            message.session_metadata = tensorflow.SessionMetadata.decodeJson(obj.sessionMetadata);
        }
        if ('optimizeForStaticGraph' in obj) {
            message.optimize_for_static_graph = obj.optimizeForStaticGraph;
        }
        if ('enableMlirBridge' in obj) {
            message.enable_mlir_bridge = obj.enableMlirBridge;
        }
        if ('mlirBridgeRollout' in obj) {
            message.mlir_bridge_rollout = typeof obj.mlirBridgeRollout === 'string' ? tensorflow.ConfigProto.Experimental.MlirBridgeRollout[obj.mlirBridgeRollout] : obj.mlirBridgeRollout;
        }
        if ('enableMlirGraphOptimization' in obj) {
            message.enable_mlir_graph_optimization = obj.enableMlirGraphOptimization;
        }
        if ('disableOutputPartitionGraphs' in obj) {
            message.disable_output_partition_graphs = obj.disableOutputPartitionGraphs;
        }
        if ('xlaFusionAutotunerThresh' in obj) {
            message.xla_fusion_autotuner_thresh = BigInt(obj.xlaFusionAutotunerThresh);
        }
        if ('useTfrt' in obj) {
            message.use_tfrt = obj.useTfrt;
        }
        if ('enableMultiHost' in obj) {
            message.enable_multi_host = obj.enableMultiHost;
        }
        if ('tfrtUseIfrt' in obj) {
            message.tfrt_use_ifrt = obj.tfrtUseIfrt;
        }
        if ('backendServerPort' in obj) {
            message.backend_server_port = Number(obj.backendServerPort);
        }
        if ('targetTpu' in obj) {
            message.target_tpu = obj.targetTpu;
        }
        if ('targetGpu' in obj) {
            message.target_gpu = obj.targetGpu;
        }
        if ('streamMergeThreshold' in obj) {
            message.stream_merge_threshold = Number(obj.streamMergeThreshold);
        }
        if ('disableFunctionalOpsLowering' in obj) {
            message.disable_functional_ops_lowering = obj.disableFunctionalOpsLowering;
        }
        if ('xlaPreferSingleGraphCluster' in obj) {
            message.xla_prefer_single_graph_cluster = obj.xlaPreferSingleGraphCluster;
        }
        if ('coordinationConfig' in obj) {
            message.coordination_config = tensorflow.CoordinationServiceConfig.decodeJson(obj.coordinationConfig);
        }
        if ('disableOptimizeForStaticGraph' in obj) {
            message.disable_optimize_for_static_graph = obj.disableOptimizeForStaticGraph;
        }
        if ('disableEagerExecutorStreamingEnqueue' in obj) {
            message.disable_eager_executor_streaming_enqueue = obj.disableEagerExecutorStreamingEnqueue;
        }
        if ('finalizeFunctionLibraryRuntime' in obj) {
            message.finalize_function_library_runtime = obj.finalizeFunctionLibraryRuntime;
        }
        if ('finalizeResourceManager' in obj) {
            message.finalize_resource_manager = obj.finalizeResourceManager;
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
tensorflow.ConfigProto.Experimental.prototype.tfrt_use_ifrt = false;
tensorflow.ConfigProto.Experimental.prototype.backend_server_port = 0;
tensorflow.ConfigProto.Experimental.prototype.target_tpu = false;
tensorflow.ConfigProto.Experimental.prototype.target_gpu = false;
tensorflow.ConfigProto.Experimental.prototype.stream_merge_threshold = 0;
tensorflow.ConfigProto.Experimental.prototype.disable_functional_ops_lowering = false;
tensorflow.ConfigProto.Experimental.prototype.xla_prefer_single_graph_cluster = false;
tensorflow.ConfigProto.Experimental.prototype.coordination_config = null;
tensorflow.ConfigProto.Experimental.prototype.disable_optimize_for_static_graph = false;
tensorflow.ConfigProto.Experimental.prototype.disable_eager_executor_streaming_enqueue = false;
tensorflow.ConfigProto.Experimental.prototype.finalize_function_library_runtime = false;
tensorflow.ConfigProto.Experimental.prototype.finalize_resource_manager = false;

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

    static decodeJson(obj) {
        const message = new tensorflow.RunOptions();
        if ('traceLevel' in obj) {
            message.trace_level = typeof obj.traceLevel === 'string' ? tensorflow.RunOptions.TraceLevel[obj.traceLevel] : obj.traceLevel;
        }
        if ('timeoutInMs' in obj) {
            message.timeout_in_ms = BigInt(obj.timeoutInMs);
        }
        if ('interOpThreadPool' in obj) {
            message.inter_op_thread_pool = Number(obj.interOpThreadPool);
        }
        if ('outputPartitionGraphs' in obj) {
            message.output_partition_graphs = obj.outputPartitionGraphs;
        }
        if ('debugOptions' in obj) {
            message.debug_options = tensorflow.DebugOptions.decodeJson(obj.debugOptions);
        }
        if ('reportTensorAllocationsUponOom' in obj) {
            message.report_tensor_allocations_upon_oom = obj.reportTensorAllocationsUponOom;
        }
        if ('experimental' in obj) {
            message.experimental = tensorflow.RunOptions.Experimental.decodeJson(obj.experimental);
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

    static decodeJson(obj) {
        const message = new tensorflow.RunOptions.Experimental();
        if ('collectiveGraphKey' in obj) {
            message.collective_graph_key = BigInt(obj.collectiveGraphKey);
        }
        if ('useRunHandlerPool' in obj) {
            message.use_run_handler_pool = obj.useRunHandlerPool;
        }
        if ('runHandlerPoolOptions' in obj) {
            message.run_handler_pool_options = tensorflow.RunOptions.Experimental.RunHandlerPoolOptions.decodeJson(obj.runHandlerPoolOptions);
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

    static decodeJson(obj) {
        const message = new tensorflow.RunOptions.Experimental.RunHandlerPoolOptions();
        if ('priority' in obj) {
            message.priority = BigInt(obj.priority);
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

    static decodeJson(obj) {
        const message = new tensorflow.RunMetadata();
        if ('stepStats' in obj) {
            message.step_stats = tensorflow.StepStats.decodeJson(obj.stepStats);
        }
        if ('costGraph' in obj) {
            message.cost_graph = tensorflow.CostGraphDef.decodeJson(obj.costGraph);
        }
        if ('partitionGraphs' in obj) {
            message.partition_graphs = obj.partitionGraphs.map((obj) => tensorflow.GraphDef.decodeJson(obj));
        }
        if ('functionGraphs' in obj) {
            message.function_graphs = obj.functionGraphs.map((obj) => tensorflow.RunMetadata.FunctionGraphs.decodeJson(obj));
        }
        if ('sessionMetadata' in obj) {
            message.session_metadata = tensorflow.SessionMetadata.decodeJson(obj.sessionMetadata);
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

    static decodeJson(obj) {
        const message = new tensorflow.RunMetadata.FunctionGraphs();
        if ('partitionGraphs' in obj) {
            message.partition_graphs = obj.partitionGraphs.map((obj) => tensorflow.GraphDef.decodeJson(obj));
        }
        if ('preOptimizationGraph' in obj) {
            message.pre_optimization_graph = tensorflow.GraphDef.decodeJson(obj.preOptimizationGraph);
        }
        if ('postOptimizationGraph' in obj) {
            message.post_optimization_graph = tensorflow.GraphDef.decodeJson(obj.postOptimizationGraph);
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorConnection();
        if ('fromTensor' in obj) {
            message.from_tensor = obj.fromTensor;
        }
        if ('toTensor' in obj) {
            message.to_tensor = obj.toTensor;
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

    static decodeJson(obj) {
        const message = new tensorflow.CallableOptions();
        if ('feed' in obj) {
            message.feed = obj.feed;
        }
        if ('fetch' in obj) {
            message.fetch = obj.fetch;
        }
        if ('target' in obj) {
            message.target = obj.target;
        }
        if ('runOptions' in obj) {
            message.run_options = tensorflow.RunOptions.decodeJson(obj.runOptions);
        }
        if ('tensorConnection' in obj) {
            message.tensor_connection = obj.tensorConnection.map((obj) => tensorflow.TensorConnection.decodeJson(obj));
        }
        if ('feedDevices' in obj) {
            for (const [key, value] of Object.entries(obj.feedDevices)) {
                message.feed_devices[key] = value;
            }
        }
        if ('fetchDevices' in obj) {
            for (const [key, value] of Object.entries(obj.fetchDevices)) {
                message.fetch_devices[key] = value;
            }
        }
        if ('fetchSkipSync' in obj) {
            message.fetch_skip_sync = obj.fetchSkipSync;
        }
        return message;
    }
};

tensorflow.CallableOptions.prototype.run_options = null;
tensorflow.CallableOptions.prototype.fetch_skip_sync = false;

tensorflow.BatchingOptions = class BatchingOptions {

    constructor() {
        this.allowed_batch_sizes = [];
    }

    static decode(reader, length) {
        const message = new tensorflow.BatchingOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.num_batch_threads = reader.int32();
                    break;
                case 2:
                    message.max_batch_size = reader.int32();
                    break;
                case 3:
                    message.batch_timeout_micros = reader.int32();
                    break;
                case 4:
                    message.allowed_batch_sizes = reader.array(message.allowed_batch_sizes, () => reader.int32(), tag);
                    break;
                case 5:
                    message.max_enqueued_batches = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new tensorflow.BatchingOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "num_batch_threads":
                    message.num_batch_threads = reader.int32();
                    break;
                case "max_batch_size":
                    message.max_batch_size = reader.int32();
                    break;
                case "batch_timeout_micros":
                    message.batch_timeout_micros = reader.int32();
                    break;
                case "allowed_batch_sizes":
                    reader.array(message.allowed_batch_sizes, () => reader.int32());
                    break;
                case "max_enqueued_batches":
                    message.max_enqueued_batches = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new tensorflow.BatchingOptions();
        if ('numBatchThreads' in obj) {
            message.num_batch_threads = Number(obj.numBatchThreads);
        }
        if ('maxBatchSize' in obj) {
            message.max_batch_size = Number(obj.maxBatchSize);
        }
        if ('batchTimeoutMicros' in obj) {
            message.batch_timeout_micros = Number(obj.batchTimeoutMicros);
        }
        if ('allowedBatchSizes' in obj) {
            message.allowed_batch_sizes = obj.allowedBatchSizes.map((obj) => Number(obj));
        }
        if ('maxEnqueuedBatches' in obj) {
            message.max_enqueued_batches = Number(obj.maxEnqueuedBatches);
        }
        return message;
    }
};

tensorflow.BatchingOptions.prototype.num_batch_threads = 0;
tensorflow.BatchingOptions.prototype.max_batch_size = 0;
tensorflow.BatchingOptions.prototype.batch_timeout_micros = 0;
tensorflow.BatchingOptions.prototype.max_enqueued_batches = 0;

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

    static decodeJson(obj) {
        const message = new tensorflow.CoordinatedJob();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('numTasks' in obj) {
            message.num_tasks = Number(obj.numTasks);
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
                case 14:
                    message.cluster_register_with_barrier = reader.bool();
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
                case 13:
                    message.poll_for_error_from_service_at_startup = reader.bool();
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
                case "cluster_register_with_barrier":
                    message.cluster_register_with_barrier = reader.bool();
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
                case "poll_for_error_from_service_at_startup":
                    message.poll_for_error_from_service_at_startup = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }

    static decodeJson(obj) {
        const message = new tensorflow.CoordinationServiceConfig();
        if ('serviceType' in obj) {
            message.service_type = obj.serviceType;
        }
        if ('serviceLeader' in obj) {
            message.service_leader = obj.serviceLeader;
        }
        if ('enableHealthCheck' in obj) {
            message.enable_health_check = obj.enableHealthCheck;
        }
        if ('clusterRegisterTimeoutInMs' in obj) {
            message.cluster_register_timeout_in_ms = BigInt(obj.clusterRegisterTimeoutInMs);
        }
        if ('clusterRegisterWithBarrier' in obj) {
            message.cluster_register_with_barrier = obj.clusterRegisterWithBarrier;
        }
        if ('heartbeatTimeoutInMs' in obj) {
            message.heartbeat_timeout_in_ms = BigInt(obj.heartbeatTimeoutInMs);
        }
        if ('coordinatedJobList' in obj) {
            message.coordinated_job_list = obj.coordinatedJobList.map((obj) => tensorflow.CoordinatedJob.decodeJson(obj));
        }
        if ('shutdownBarrierTimeoutInMs' in obj) {
            message.shutdown_barrier_timeout_in_ms = BigInt(obj.shutdownBarrierTimeoutInMs);
        }
        if ('agentDestructionWithoutShutdown' in obj) {
            message.agent_destruction_without_shutdown = obj.agentDestructionWithoutShutdown;
        }
        if ('recoverableJobs' in obj) {
            message.recoverable_jobs = obj.recoverableJobs;
        }
        if ('allowNewIncarnationToReconnect' in obj) {
            message.allow_new_incarnation_to_reconnect = obj.allowNewIncarnationToReconnect;
        }
        if ('forceDisable' in obj) {
            message.force_disable = obj.forceDisable;
        }
        if ('pollForErrorFromServiceAtStartup' in obj) {
            message.poll_for_error_from_service_at_startup = obj.pollForErrorFromServiceAtStartup;
        }
        return message;
    }
};

tensorflow.CoordinationServiceConfig.prototype.service_type = "";
tensorflow.CoordinationServiceConfig.prototype.service_leader = "";
tensorflow.CoordinationServiceConfig.prototype.enable_health_check = false;
tensorflow.CoordinationServiceConfig.prototype.cluster_register_timeout_in_ms = 0n;
tensorflow.CoordinationServiceConfig.prototype.cluster_register_with_barrier = false;
tensorflow.CoordinationServiceConfig.prototype.heartbeat_timeout_in_ms = 0n;
tensorflow.CoordinationServiceConfig.prototype.shutdown_barrier_timeout_in_ms = 0n;
tensorflow.CoordinationServiceConfig.prototype.agent_destruction_without_shutdown = false;
tensorflow.CoordinationServiceConfig.prototype.allow_new_incarnation_to_reconnect = false;
tensorflow.CoordinationServiceConfig.prototype.force_disable = false;
tensorflow.CoordinationServiceConfig.prototype.poll_for_error_from_service_at_startup = false;

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

    static decodeJson(obj) {
        const message = new tensorflow.CostGraphDef();
        if ('node' in obj) {
            message.node = obj.node.map((obj) => tensorflow.CostGraphDef.Node.decodeJson(obj));
        }
        if ('cost' in obj) {
            message.cost = obj.cost.map((obj) => tensorflow.CostGraphDef.AggregatedCost.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.CostGraphDef.Node();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('device' in obj) {
            message.device = obj.device;
        }
        if ('id' in obj) {
            message.id = Number(obj.id);
        }
        if ('inputInfo' in obj) {
            message.input_info = obj.inputInfo.map((obj) => tensorflow.CostGraphDef.Node.InputInfo.decodeJson(obj));
        }
        if ('outputInfo' in obj) {
            message.output_info = obj.outputInfo.map((obj) => tensorflow.CostGraphDef.Node.OutputInfo.decodeJson(obj));
        }
        if ('temporaryMemorySize' in obj) {
            message.temporary_memory_size = BigInt(obj.temporaryMemorySize);
        }
        if ('persistentMemorySize' in obj) {
            message.persistent_memory_size = BigInt(obj.persistentMemorySize);
        }
        if ('hostTempMemorySize' in obj) {
            message.host_temp_memory_size = BigInt(obj.hostTempMemorySize);
        }
        if ('deviceTempMemorySize' in obj) {
            message.device_temp_memory_size = BigInt(obj.deviceTempMemorySize);
        }
        if ('devicePersistentMemorySize' in obj) {
            message.device_persistent_memory_size = BigInt(obj.devicePersistentMemorySize);
        }
        if ('computeCost' in obj) {
            message.compute_cost = BigInt(obj.computeCost);
        }
        if ('computeTime' in obj) {
            message.compute_time = BigInt(obj.computeTime);
        }
        if ('memoryTime' in obj) {
            message.memory_time = BigInt(obj.memoryTime);
        }
        if ('isFinal' in obj) {
            message.is_final = obj.isFinal;
        }
        if ('controlInput' in obj) {
            message.control_input = obj.controlInput.map((obj) => Number(obj));
        }
        if ('inaccurate' in obj) {
            message.inaccurate = obj.inaccurate;
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

    static decodeJson(obj) {
        const message = new tensorflow.CostGraphDef.Node.InputInfo();
        if ('precedingNode' in obj) {
            message.preceding_node = Number(obj.precedingNode);
        }
        if ('precedingPort' in obj) {
            message.preceding_port = Number(obj.precedingPort);
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

    static decodeJson(obj) {
        const message = new tensorflow.CostGraphDef.Node.OutputInfo();
        if ('size' in obj) {
            message.size = BigInt(obj.size);
        }
        if ('aliasInputPort' in obj) {
            message.alias_input_port = BigInt(obj.aliasInputPort);
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
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

    static decodeJson(obj) {
        const message = new tensorflow.CostGraphDef.AggregatedCost();
        if ('cost' in obj) {
            message.cost = Number(obj.cost);
        }
        if ('dimension' in obj) {
            message.dimension = obj.dimension;
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

    static decodeJson(obj) {
        const message = new tensorflow.AllocationRecord();
        if ('allocMicros' in obj) {
            message.alloc_micros = BigInt(obj.allocMicros);
        }
        if ('allocBytes' in obj) {
            message.alloc_bytes = BigInt(obj.allocBytes);
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

    static decodeJson(obj) {
        const message = new tensorflow.AllocatorMemoryUsed();
        if ('allocatorName' in obj) {
            message.allocator_name = obj.allocatorName;
        }
        if ('totalBytes' in obj) {
            message.total_bytes = BigInt(obj.totalBytes);
        }
        if ('peakBytes' in obj) {
            message.peak_bytes = BigInt(obj.peakBytes);
        }
        if ('liveBytes' in obj) {
            message.live_bytes = BigInt(obj.liveBytes);
        }
        if ('allocationRecords' in obj) {
            message.allocation_records = obj.allocationRecords.map((obj) => tensorflow.AllocationRecord.decodeJson(obj));
        }
        if ('allocatorBytesInUse' in obj) {
            message.allocator_bytes_in_use = BigInt(obj.allocatorBytesInUse);
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

    static decodeJson(obj) {
        const message = new tensorflow.NodeOutput();
        if ('slot' in obj) {
            message.slot = Number(obj.slot);
        }
        if ('tensorDescription' in obj) {
            message.tensor_description = tensorflow.TensorDescription.decodeJson(obj.tensorDescription);
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

    static decodeJson(obj) {
        const message = new tensorflow.MemoryStats();
        if ('tempMemorySize' in obj) {
            message.temp_memory_size = BigInt(obj.tempMemorySize);
        }
        if ('persistentMemorySize' in obj) {
            message.persistent_memory_size = BigInt(obj.persistentMemorySize);
        }
        if ('persistentTensorAllocIds' in obj) {
            message.persistent_tensor_alloc_ids = obj.persistentTensorAllocIds.map((obj) => BigInt(obj));
        }
        if ('deviceTempMemorySize' in obj) {
            message.device_temp_memory_size = BigInt(obj.deviceTempMemorySize);
        }
        if ('devicePersistentMemorySize' in obj) {
            message.device_persistent_memory_size = BigInt(obj.devicePersistentMemorySize);
        }
        if ('devicePersistentTensorAllocIds' in obj) {
            message.device_persistent_tensor_alloc_ids = obj.devicePersistentTensorAllocIds.map((obj) => BigInt(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.NodeExecStats();
        if ('nodeName' in obj) {
            message.node_name = obj.nodeName;
        }
        if ('allStartMicros' in obj) {
            message.all_start_micros = BigInt(obj.allStartMicros);
        }
        if ('opStartRelMicros' in obj) {
            message.op_start_rel_micros = BigInt(obj.opStartRelMicros);
        }
        if ('opEndRelMicros' in obj) {
            message.op_end_rel_micros = BigInt(obj.opEndRelMicros);
        }
        if ('allEndRelMicros' in obj) {
            message.all_end_rel_micros = BigInt(obj.allEndRelMicros);
        }
        if ('memory' in obj) {
            message.memory = obj.memory.map((obj) => tensorflow.AllocatorMemoryUsed.decodeJson(obj));
        }
        if ('output' in obj) {
            message.output = obj.output.map((obj) => tensorflow.NodeOutput.decodeJson(obj));
        }
        if ('timelineLabel' in obj) {
            message.timeline_label = obj.timelineLabel;
        }
        if ('scheduledMicros' in obj) {
            message.scheduled_micros = BigInt(obj.scheduledMicros);
        }
        if ('threadId' in obj) {
            message.thread_id = Number(obj.threadId);
        }
        if ('referencedTensor' in obj) {
            message.referenced_tensor = obj.referencedTensor.map((obj) => tensorflow.AllocationDescription.decodeJson(obj));
        }
        if ('memoryStats' in obj) {
            message.memory_stats = tensorflow.MemoryStats.decodeJson(obj.memoryStats);
        }
        if ('allStartNanos' in obj) {
            message.all_start_nanos = BigInt(obj.allStartNanos);
        }
        if ('opStartRelNanos' in obj) {
            message.op_start_rel_nanos = BigInt(obj.opStartRelNanos);
        }
        if ('opEndRelNanos' in obj) {
            message.op_end_rel_nanos = BigInt(obj.opEndRelNanos);
        }
        if ('allEndRelNanos' in obj) {
            message.all_end_rel_nanos = BigInt(obj.allEndRelNanos);
        }
        if ('scheduledNanos' in obj) {
            message.scheduled_nanos = BigInt(obj.scheduledNanos);
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

    static decodeJson(obj) {
        const message = new tensorflow.DeviceStepStats();
        if ('device' in obj) {
            message.device = obj.device;
        }
        if ('nodeStats' in obj) {
            message.node_stats = obj.nodeStats.map((obj) => tensorflow.NodeExecStats.decodeJson(obj));
        }
        if ('threadNames' in obj) {
            for (const [key, value] of Object.entries(obj.threadNames)) {
                message.thread_names[key] = value;
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

    static decodeJson(obj) {
        const message = new tensorflow.StepStats();
        if ('devStats' in obj) {
            message.dev_stats = obj.devStats.map((obj) => tensorflow.DeviceStepStats.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.AllocationDescription();
        if ('requestedBytes' in obj) {
            message.requested_bytes = BigInt(obj.requestedBytes);
        }
        if ('allocatedBytes' in obj) {
            message.allocated_bytes = BigInt(obj.allocatedBytes);
        }
        if ('allocatorName' in obj) {
            message.allocator_name = obj.allocatorName;
        }
        if ('allocationId' in obj) {
            message.allocation_id = BigInt(obj.allocationId);
        }
        if ('hasSingleReference' in obj) {
            message.has_single_reference = obj.hasSingleReference;
        }
        if ('ptr' in obj) {
            message.ptr = BigInt(obj.ptr);
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

    static decodeJson(obj) {
        const message = new tensorflow.TensorDescription();
        if ('dtype' in obj) {
            message.dtype = typeof obj.dtype === 'string' ? tensorflow.DataType[obj.dtype] : obj.dtype;
        }
        if ('shape' in obj) {
            message.shape = tensorflow.TensorShapeProto.decodeJson(obj.shape);
        }
        if ('allocationDescription' in obj) {
            message.allocation_description = tensorflow.AllocationDescription.decodeJson(obj.allocationDescription);
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

    static decodeJson(obj) {
        const message = new tensorflow.JobDef();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('tasks' in obj) {
            for (const [key, value] of Object.entries(obj.tasks)) {
                message.tasks[key] = value;
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

    static decodeJson(obj) {
        const message = new tensorflow.ClusterDef();
        if ('job' in obj) {
            message.job = obj.job.map((obj) => tensorflow.JobDef.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.DebugTensorWatch();
        if ('nodeName' in obj) {
            message.node_name = obj.nodeName;
        }
        if ('outputSlot' in obj) {
            message.output_slot = Number(obj.outputSlot);
        }
        if ('debugOps' in obj) {
            message.debug_ops = obj.debugOps;
        }
        if ('debugUrls' in obj) {
            message.debug_urls = obj.debugUrls;
        }
        if ('tolerateDebugOpCreationFailures' in obj) {
            message.tolerate_debug_op_creation_failures = obj.tolerateDebugOpCreationFailures;
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

    static decodeJson(obj) {
        const message = new tensorflow.DebugOptions();
        if ('debugTensorWatchOpts' in obj) {
            message.debug_tensor_watch_opts = obj.debugTensorWatchOpts.map((obj) => tensorflow.DebugTensorWatch.decodeJson(obj));
        }
        if ('globalStep' in obj) {
            message.global_step = BigInt(obj.globalStep);
        }
        if ('resetDiskByteUsage' in obj) {
            message.reset_disk_byte_usage = obj.resetDiskByteUsage;
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

    static decodeJson(obj) {
        const message = new tensorflow.DebuggedSourceFile();
        if ('host' in obj) {
            message.host = obj.host;
        }
        if ('filePath' in obj) {
            message.file_path = obj.filePath;
        }
        if ('lastModified' in obj) {
            message.last_modified = BigInt(obj.lastModified);
        }
        if ('bytes' in obj) {
            message.bytes = BigInt(obj.bytes);
        }
        if ('lines' in obj) {
            message.lines = obj.lines;
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

    static decodeJson(obj) {
        const message = new tensorflow.DebuggedSourceFiles();
        if ('sourceFiles' in obj) {
            message.source_files = obj.sourceFiles.map((obj) => tensorflow.DebuggedSourceFile.decodeJson(obj));
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

    static decodeJson(obj) {
        const message = new tensorflow.AutoParallelOptions();
        if ('enable' in obj) {
            message.enable = obj.enable;
        }
        if ('numReplicas' in obj) {
            message.num_replicas = Number(obj.numReplicas);
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

    static decodeJson(obj) {
        const message = new tensorflow.ScopedAllocatorOptions();
        if ('enableOp' in obj) {
            message.enable_op = obj.enableOp;
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

    static decodeJson(obj) {
        const message = new tensorflow.RewriterConfig();
        if ('cpuLayoutConversion' in obj) {
            message.cpu_layout_conversion = typeof obj.cpuLayoutConversion === 'string' ? tensorflow.RewriterConfig.CpuLayout[obj.cpuLayoutConversion] : obj.cpuLayoutConversion;
        }
        if ('layoutOptimizer' in obj) {
            message.layout_optimizer = typeof obj.layoutOptimizer === 'string' ? tensorflow.RewriterConfig.Toggle[obj.layoutOptimizer] : obj.layoutOptimizer;
        }
        if ('constantFolding' in obj) {
            message.constant_folding = typeof obj.constantFolding === 'string' ? tensorflow.RewriterConfig.Toggle[obj.constantFolding] : obj.constantFolding;
        }
        if ('shapeOptimization' in obj) {
            message.shape_optimization = typeof obj.shapeOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.shapeOptimization] : obj.shapeOptimization;
        }
        if ('remapping' in obj) {
            message.remapping = typeof obj.remapping === 'string' ? tensorflow.RewriterConfig.Toggle[obj.remapping] : obj.remapping;
        }
        if ('commonSubgraphElimination' in obj) {
            message.common_subgraph_elimination = typeof obj.commonSubgraphElimination === 'string' ? tensorflow.RewriterConfig.Toggle[obj.commonSubgraphElimination] : obj.commonSubgraphElimination;
        }
        if ('arithmeticOptimization' in obj) {
            message.arithmetic_optimization = typeof obj.arithmeticOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.arithmeticOptimization] : obj.arithmeticOptimization;
        }
        if ('dependencyOptimization' in obj) {
            message.dependency_optimization = typeof obj.dependencyOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.dependencyOptimization] : obj.dependencyOptimization;
        }
        if ('loopOptimization' in obj) {
            message.loop_optimization = typeof obj.loopOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.loopOptimization] : obj.loopOptimization;
        }
        if ('functionOptimization' in obj) {
            message.function_optimization = typeof obj.functionOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.functionOptimization] : obj.functionOptimization;
        }
        if ('debugStripper' in obj) {
            message.debug_stripper = typeof obj.debugStripper === 'string' ? tensorflow.RewriterConfig.Toggle[obj.debugStripper] : obj.debugStripper;
        }
        if ('disableModelPruning' in obj) {
            message.disable_model_pruning = obj.disableModelPruning;
        }
        if ('scopedAllocatorOptimization' in obj) {
            message.scoped_allocator_optimization = typeof obj.scopedAllocatorOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.scopedAllocatorOptimization] : obj.scopedAllocatorOptimization;
        }
        if ('pinToHostOptimization' in obj) {
            message.pin_to_host_optimization = typeof obj.pinToHostOptimization === 'string' ? tensorflow.RewriterConfig.Toggle[obj.pinToHostOptimization] : obj.pinToHostOptimization;
        }
        if ('implementationSelector' in obj) {
            message.implementation_selector = typeof obj.implementationSelector === 'string' ? tensorflow.RewriterConfig.Toggle[obj.implementationSelector] : obj.implementationSelector;
        }
        if ('autoMixedPrecision' in obj) {
            message.auto_mixed_precision = typeof obj.autoMixedPrecision === 'string' ? tensorflow.RewriterConfig.Toggle[obj.autoMixedPrecision] : obj.autoMixedPrecision;
        }
        if ('autoMixedPrecisionMkl' in obj) {
            message.auto_mixed_precision_mkl = typeof obj.autoMixedPrecisionMkl === 'string' ? tensorflow.RewriterConfig.Toggle[obj.autoMixedPrecisionMkl] : obj.autoMixedPrecisionMkl;
        }
        if ('autoMixedPrecisionOnednnBfloat16' in obj) {
            message.auto_mixed_precision_onednn_bfloat16 = typeof obj.autoMixedPrecisionOnednnBfloat16 === 'string' ? tensorflow.RewriterConfig.Toggle[obj.autoMixedPrecisionOnednnBfloat16] : obj.autoMixedPrecisionOnednnBfloat16;
        }
        if ('autoMixedPrecisionCpu' in obj) {
            message.auto_mixed_precision_cpu = typeof obj.autoMixedPrecisionCpu === 'string' ? tensorflow.RewriterConfig.Toggle[obj.autoMixedPrecisionCpu] : obj.autoMixedPrecisionCpu;
        }
        if ('disableMetaOptimizer' in obj) {
            message.disable_meta_optimizer = obj.disableMetaOptimizer;
        }
        if ('disableTfgOptimizer' in obj) {
            message.disable_tfg_optimizer = obj.disableTfgOptimizer;
        }
        if ('usePluginOptimizers' in obj) {
            message.use_plugin_optimizers = typeof obj.usePluginOptimizers === 'string' ? tensorflow.RewriterConfig.Toggle[obj.usePluginOptimizers] : obj.usePluginOptimizers;
        }
        if ('experimentalConditionalCodeMotion' in obj) {
            message.experimental_conditional_code_motion = typeof obj.experimentalConditionalCodeMotion === 'string' ? tensorflow.RewriterConfig.Toggle[obj.experimentalConditionalCodeMotion] : obj.experimentalConditionalCodeMotion;
        }
        if ('metaOptimizerIterations' in obj) {
            message.meta_optimizer_iterations = typeof obj.metaOptimizerIterations === 'string' ? tensorflow.RewriterConfig.NumIterationsType[obj.metaOptimizerIterations] : obj.metaOptimizerIterations;
        }
        if ('minGraphNodes' in obj) {
            message.min_graph_nodes = Number(obj.minGraphNodes);
        }
        if ('experimentalDisableCompressedTensorOptimization' in obj) {
            message.experimental_disable_compressed_tensor_optimization = obj.experimentalDisableCompressedTensorOptimization;
        }
        if ('experimentalDisableFoldingQuantizationEmulation' in obj) {
            message.experimental_disable_folding_quantization_emulation = obj.experimentalDisableFoldingQuantizationEmulation;
        }
        if ('memoryOptimization' in obj) {
            message.memory_optimization = typeof obj.memoryOptimization === 'string' ? tensorflow.RewriterConfig.MemOptType[obj.memoryOptimization] : obj.memoryOptimization;
        }
        if ('memoryOptimizerTargetNodeNameScope' in obj) {
            message.memory_optimizer_target_node_name_scope = obj.memoryOptimizerTargetNodeNameScope;
        }
        if ('metaOptimizerTimeoutMs' in obj) {
            message.meta_optimizer_timeout_ms = BigInt(obj.metaOptimizerTimeoutMs);
        }
        if ('autoParallel' in obj) {
            message.auto_parallel = tensorflow.AutoParallelOptions.decodeJson(obj.autoParallel);
        }
        if ('failOnOptimizerErrors' in obj) {
            message.fail_on_optimizer_errors = obj.failOnOptimizerErrors;
        }
        if ('scopedAllocatorOpts' in obj) {
            message.scoped_allocator_opts = tensorflow.ScopedAllocatorOptions.decodeJson(obj.scopedAllocatorOpts);
        }
        if ('optimizers' in obj) {
            message.optimizers = obj.optimizers;
        }
        if ('customOptimizers' in obj) {
            message.custom_optimizers = obj.customOptimizers.map((obj) => tensorflow.RewriterConfig.CustomGraphOptimizer.decodeJson(obj));
        }
        if ('interOptimizerVerifierConfig' in obj) {
            message.inter_optimizer_verifier_config = tensorflow.VerifierConfig.decodeJson(obj.interOptimizerVerifierConfig);
        }
        if ('postOptimizationVerifierConfig' in obj) {
            message.post_optimization_verifier_config = tensorflow.VerifierConfig.decodeJson(obj.postOptimizationVerifierConfig);
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

    static decodeJson(obj) {
        const message = new tensorflow.RewriterConfig.CustomGraphOptimizer();
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('parameterMap' in obj) {
            for (const [key, value] of Object.entries(obj.parameterMap)) {
                message.parameter_map[key] = tensorflow.AttrValue.decodeJson(value);
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

    static decodeJson(obj) {
        const message = new tensorflow.VerifierConfig();
        if ('verificationTimeoutInMs' in obj) {
            message.verification_timeout_in_ms = BigInt(obj.verificationTimeoutInMs);
        }
        if ('structureVerifier' in obj) {
            message.structure_verifier = typeof obj.structureVerifier === 'string' ? tensorflow.VerifierConfig.Toggle[obj.structureVerifier] : obj.structureVerifier;
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

    static decodeJson(obj) {
        const message = new tensorflow.RPCOptions();
        if ('useRpcForInprocessMaster' in obj) {
            message.use_rpc_for_inprocess_master = obj.useRpcForInprocessMaster;
        }
        if ('compressionAlgorithm' in obj) {
            message.compression_algorithm = obj.compressionAlgorithm;
        }
        if ('compressionLevel' in obj) {
            message.compression_level = Number(obj.compressionLevel);
        }
        if ('cacheRpcResponse' in obj) {
            message.cache_rpc_response = obj.cacheRpcResponse;
        }
        if ('disableSessionConnectionSharing' in obj) {
            message.disable_session_connection_sharing = obj.disableSessionConnectionSharing;
        }
        if ('numChannelsPerTarget' in obj) {
            message.num_channels_per_target = Number(obj.numChannelsPerTarget);
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

    static decodeJson(obj) {
        const message = new tensorflow.MemmappedFileSystemDirectoryElement();
        if ('offset' in obj) {
            message.offset = BigInt(obj.offset);
        }
        if ('name' in obj) {
            message.name = obj.name;
        }
        if ('length' in obj) {
            message.length = BigInt(obj.length);
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

    static decodeJson(obj) {
        const message = new tensorflow.MemmappedFileSystemDirectory();
        if ('element' in obj) {
            message.element = obj.element.map((obj) => tensorflow.MemmappedFileSystemDirectoryElement.decodeJson(obj));
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
                case 7:
                    message.uuid = reader.string();
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
                case "uuid":
                    message.uuid = reader.string();
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

    static decodeJson(obj) {
        const message = new tensorflow.FingerprintDef();
        if ('savedModelChecksum' in obj) {
            message.saved_model_checksum = BigInt(obj.savedModelChecksum);
        }
        if ('graphDefProgramHash' in obj) {
            message.graph_def_program_hash = BigInt(obj.graphDefProgramHash);
        }
        if ('signatureDefHash' in obj) {
            message.signature_def_hash = BigInt(obj.signatureDefHash);
        }
        if ('savedObjectGraphHash' in obj) {
            message.saved_object_graph_hash = BigInt(obj.savedObjectGraphHash);
        }
        if ('checkpointHash' in obj) {
            message.checkpoint_hash = BigInt(obj.checkpointHash);
        }
        if ('uuid' in obj) {
            message.uuid = obj.uuid;
        }
        if ('version' in obj) {
            message.version = tensorflow.VersionDef.decodeJson(obj.version);
        }
        return message;
    }
};

tensorflow.FingerprintDef.prototype.saved_model_checksum = 0n;
tensorflow.FingerprintDef.prototype.graph_def_program_hash = 0n;
tensorflow.FingerprintDef.prototype.signature_def_hash = 0n;
tensorflow.FingerprintDef.prototype.saved_object_graph_hash = 0n;
tensorflow.FingerprintDef.prototype.checkpoint_hash = 0n;
tensorflow.FingerprintDef.prototype.uuid = "";
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

    static decodeJson() {
        throw new Error('Any fields not implemented.');
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

    static decodeJson(obj) {
        const message = new google.protobuf.BoolValue();
        if ('value' in obj) {
            message.value = obj.value;
        }
        return message;
    }
};

google.protobuf.BoolValue.prototype.value = false;
