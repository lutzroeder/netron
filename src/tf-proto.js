(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('tf');

    $root.tensorflow = (function() {

        const tensorflow = {};

        tensorflow.SavedModel = (function() {

            function SavedModel() {
                this.meta_graphs = [];
            }

            SavedModel.prototype.saved_model_schema_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            SavedModel.prototype.meta_graphs = [];

            SavedModel.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedModel();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedModel.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedModel();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "saved_model_schema_version":
                            message.saved_model_schema_version = reader.int64();
                            break;
                        case "meta_graphs":
                            message.meta_graphs.push($root.tensorflow.MetaGraphDef.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedModel;
        })();

        tensorflow.MetaGraphDef = (function() {

            function MetaGraphDef() {
                this.collection_def = {};
                this.signature_def = {};
                this.asset_file_def = [];
            }

            MetaGraphDef.prototype.meta_info_def = null;
            MetaGraphDef.prototype.graph_def = null;
            MetaGraphDef.prototype.saver_def = null;
            MetaGraphDef.prototype.collection_def = {};
            MetaGraphDef.prototype.signature_def = {};
            MetaGraphDef.prototype.asset_file_def = [];
            MetaGraphDef.prototype.object_graph_def = null;

            MetaGraphDef.decode = function (reader, length) {
                const message = new $root.tensorflow.MetaGraphDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
                            reader.pair(message.collection_def, () => reader.string(), () => $root.tensorflow.CollectionDef.decode(reader, reader.uint32()));
                            break;
                        case 5:
                            reader.pair(message.signature_def, () => reader.string(), () => $root.tensorflow.SignatureDef.decode(reader, reader.uint32()));
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
            };

            MetaGraphDef.decodeText = function (reader) {
                const message = new $root.tensorflow.MetaGraphDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "meta_info_def":
                            message.meta_info_def = $root.tensorflow.MetaGraphDef.MetaInfoDef.decodeText(reader, true);
                            break;
                        case "graph_def":
                            message.graph_def = $root.tensorflow.GraphDef.decodeText(reader, true);
                            break;
                        case "saver_def":
                            message.saver_def = $root.tensorflow.SaverDef.decodeText(reader, true);
                            break;
                        case "collection_def":
                            reader.pair(message.collection_def, () => reader.string(), () => $root.tensorflow.CollectionDef.decodeText(reader, true));
                            break;
                        case "signature_def":
                            reader.pair(message.signature_def, () => reader.string(), () => $root.tensorflow.SignatureDef.decodeText(reader, true));
                            break;
                        case "asset_file_def":
                            message.asset_file_def.push($root.tensorflow.AssetFileDef.decodeText(reader, true));
                            break;
                        case "object_graph_def":
                            message.object_graph_def = $root.tensorflow.SavedObjectGraph.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            MetaGraphDef.MetaInfoDef = (function() {

                function MetaInfoDef() {
                    this.tags = [];
                    this.function_aliases = {};
                }

                MetaInfoDef.prototype.meta_graph_version = "";
                MetaInfoDef.prototype.stripped_op_list = null;
                MetaInfoDef.prototype.any_info = null;
                MetaInfoDef.prototype.tags = [];
                MetaInfoDef.prototype.tensorflow_version = "";
                MetaInfoDef.prototype.tensorflow_git_version = "";
                MetaInfoDef.prototype.stripped_default_attrs = false;
                MetaInfoDef.prototype.function_aliases = {};

                MetaInfoDef.decode = function (reader, length) {
                    const message = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                                reader.pair(message.function_aliases, () => reader.string(), () => reader.string());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                MetaInfoDef.decodeText = function (reader) {
                    const message = new $root.tensorflow.MetaGraphDef.MetaInfoDef();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "meta_graph_version":
                                message.meta_graph_version = reader.string();
                                break;
                            case "stripped_op_list":
                                message.stripped_op_list = $root.tensorflow.OpList.decodeText(reader, true);
                                break;
                            case "any_info":
                                message.any_info = $root.google.protobuf.Any.decodeText(reader, true);
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
                                reader.pair(message.function_aliases, () => reader.string(), () => reader.string());
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return MetaInfoDef;
            })();

            return MetaGraphDef;
        })();

        tensorflow.CollectionDef = (function() {

            function CollectionDef() {
            }

            CollectionDef.prototype.node_list = null;
            CollectionDef.prototype.bytes_list = null;
            CollectionDef.prototype.int64_list = null;
            CollectionDef.prototype.float_list = null;
            CollectionDef.prototype.any_list = null;

            const kindSet = new Set([ "node_list", "bytes_list", "int64_list", "float_list", "any_list"]);
            Object.defineProperty(CollectionDef.prototype, "kind", {
                get: function() { return Object.keys(this).find((key) => kindSet.has(key) && this[key] != null); }
            });

            CollectionDef.decode = function (reader, length) {
                const message = new $root.tensorflow.CollectionDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            CollectionDef.decodeText = function (reader) {
                const message = new $root.tensorflow.CollectionDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "node_list":
                            message.node_list = $root.tensorflow.CollectionDef.NodeList.decodeText(reader, true);
                            break;
                        case "bytes_list":
                            message.bytes_list = $root.tensorflow.CollectionDef.BytesList.decodeText(reader, true);
                            break;
                        case "int64_list":
                            message.int64_list = $root.tensorflow.CollectionDef.Int64List.decodeText(reader, true);
                            break;
                        case "float_list":
                            message.float_list = $root.tensorflow.CollectionDef.FloatList.decodeText(reader, true);
                            break;
                        case "any_list":
                            message.any_list = $root.tensorflow.CollectionDef.AnyList.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            CollectionDef.NodeList = (function() {

                function NodeList() {
                    this.value = [];
                }

                NodeList.prototype.value = [];

                NodeList.decode = function (reader, length) {
                    const message = new $root.tensorflow.CollectionDef.NodeList();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                NodeList.decodeText = function (reader) {
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
                };

                return NodeList;
            })();

            CollectionDef.BytesList = (function() {

                function BytesList() {
                    this.value = [];
                }

                BytesList.prototype.value = [];

                BytesList.decode = function (reader, length) {
                    const message = new $root.tensorflow.CollectionDef.BytesList();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                BytesList.decodeText = function (reader) {
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
                };

                return BytesList;
            })();

            CollectionDef.Int64List = (function() {

                function Int64List() {
                    this.value = [];
                }

                Int64List.prototype.value = [];

                Int64List.decode = function (reader, length) {
                    const message = new $root.tensorflow.CollectionDef.Int64List();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                Int64List.decodeText = function (reader) {
                    const message = new $root.tensorflow.CollectionDef.Int64List();
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
                };

                return Int64List;
            })();

            CollectionDef.FloatList = (function() {

                function FloatList() {
                    this.value = [];
                }

                FloatList.prototype.value = [];

                FloatList.decode = function (reader, length) {
                    const message = new $root.tensorflow.CollectionDef.FloatList();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                FloatList.decodeText = function (reader) {
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
                };

                return FloatList;
            })();

            CollectionDef.AnyList = (function() {

                function AnyList() {
                    this.value = [];
                }

                AnyList.prototype.value = [];

                AnyList.decode = function (reader, length) {
                    const message = new $root.tensorflow.CollectionDef.AnyList();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                AnyList.decodeText = function (reader) {
                    const message = new $root.tensorflow.CollectionDef.AnyList();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "value":
                                message.value.push($root.google.protobuf.Any.decodeText(reader, true));
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return AnyList;
            })();

            return CollectionDef;
        })();

        tensorflow.TensorInfo = (function() {

            function TensorInfo() {
            }

            TensorInfo.prototype.name = "";
            TensorInfo.prototype.coo_sparse = null;
            TensorInfo.prototype.composite_tensor = null;
            TensorInfo.prototype.dtype = 0;
            TensorInfo.prototype.tensor_shape = null;

            const encodingSet = new Set([ "name", "coo_sparse", "composite_tensor"]);
            Object.defineProperty(TensorInfo.prototype, "encoding", {
                get: function() { return Object.keys(this).find((key) => encodingSet.has(key) && this[key] != null); }
            });

            TensorInfo.decode = function (reader, length) {
                const message = new $root.tensorflow.TensorInfo();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TensorInfo.decodeText = function (reader) {
                const message = new $root.tensorflow.TensorInfo();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "coo_sparse":
                            message.coo_sparse = $root.tensorflow.TensorInfo.CooSparse.decodeText(reader, true);
                            break;
                        case "composite_tensor":
                            message.composite_tensor = $root.tensorflow.TensorInfo.CompositeTensor.decodeText(reader, true);
                            break;
                        case "dtype":
                            message.dtype = reader.enum($root.tensorflow.DataType);
                            break;
                        case "tensor_shape":
                            message.tensor_shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            TensorInfo.CooSparse = (function() {

                function CooSparse() {
                }

                CooSparse.prototype.values_tensor_name = "";
                CooSparse.prototype.indices_tensor_name = "";
                CooSparse.prototype.dense_shape_tensor_name = "";

                CooSparse.decode = function (reader, length) {
                    const message = new $root.tensorflow.TensorInfo.CooSparse();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                CooSparse.decodeText = function (reader) {
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
                };

                return CooSparse;
            })();

            TensorInfo.CompositeTensor = (function() {

                function CompositeTensor() {
                    this.components = [];
                }

                CompositeTensor.prototype.type_spec = null;
                CompositeTensor.prototype.components = [];

                CompositeTensor.decode = function (reader, length) {
                    const message = new $root.tensorflow.TensorInfo.CompositeTensor();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                CompositeTensor.decodeText = function (reader) {
                    const message = new $root.tensorflow.TensorInfo.CompositeTensor();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "type_spec":
                                message.type_spec = $root.tensorflow.TypeSpecProto.decodeText(reader, true);
                                break;
                            case "components":
                                message.components.push($root.tensorflow.TensorInfo.decodeText(reader, true));
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return CompositeTensor;
            })();

            return TensorInfo;
        })();

        tensorflow.SignatureDef = (function() {

            function SignatureDef() {
                this.inputs = {};
                this.outputs = {};
            }

            SignatureDef.prototype.inputs = {};
            SignatureDef.prototype.outputs = {};
            SignatureDef.prototype.method_name = "";

            SignatureDef.decode = function (reader, length) {
                const message = new $root.tensorflow.SignatureDef();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            reader.pair(message.inputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            reader.pair(message.outputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decode(reader, reader.uint32()));
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
            };

            SignatureDef.decodeText = function (reader) {
                const message = new $root.tensorflow.SignatureDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "inputs":
                            reader.pair(message.inputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decodeText(reader, true));
                            break;
                        case "outputs":
                            reader.pair(message.outputs, () => reader.string(), () => $root.tensorflow.TensorInfo.decodeText(reader, true));
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
            };

            return SignatureDef;
        })();

        tensorflow.AssetFileDef = (function() {

            function AssetFileDef() {
            }

            AssetFileDef.prototype.tensor_info = null;
            AssetFileDef.prototype.filename = "";

            AssetFileDef.decode = function (reader, length) {
                const message = new $root.tensorflow.AssetFileDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            AssetFileDef.decodeText = function (reader) {
                const message = new $root.tensorflow.AssetFileDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "tensor_info":
                            message.tensor_info = $root.tensorflow.TensorInfo.decodeText(reader, true);
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
            };

            return AssetFileDef;
        })();

        tensorflow.GraphDef = (function() {

            function GraphDef() {
                this.node = [];
            }

            GraphDef.prototype.node = [];
            GraphDef.prototype.versions = null;
            GraphDef.prototype.version = 0;
            GraphDef.prototype.library = null;

            GraphDef.decode = function (reader, length) {
                const message = new $root.tensorflow.GraphDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            GraphDef.decodeText = function (reader) {
                const message = new $root.tensorflow.GraphDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "node":
                            message.node.push($root.tensorflow.NodeDef.decodeText(reader, true));
                            break;
                        case "versions":
                            message.versions = $root.tensorflow.VersionDef.decodeText(reader, true);
                            break;
                        case "version":
                            message.version = reader.int32();
                            break;
                        case "library":
                            message.library = $root.tensorflow.FunctionDefLibrary.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return GraphDef;
        })();

        tensorflow.FunctionDefLibrary = (function() {

            function FunctionDefLibrary() {
                this["function"] = [];
                this.gradient = [];
            }

            FunctionDefLibrary.prototype["function"] = [];
            FunctionDefLibrary.prototype.gradient = [];

            FunctionDefLibrary.decode = function (reader, length) {
                const message = new $root.tensorflow.FunctionDefLibrary();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            FunctionDefLibrary.decodeText = function (reader) {
                const message = new $root.tensorflow.FunctionDefLibrary();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "function":
                            message["function"].push($root.tensorflow.FunctionDef.decodeText(reader, true));
                            break;
                        case "gradient":
                            message.gradient.push($root.tensorflow.GradientDef.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return FunctionDefLibrary;
        })();

        tensorflow.FunctionDef = (function() {

            function FunctionDef() {
                this.attr = {};
                this.arg_attr = {};
                this.resource_arg_unique_id = {};
                this.node_def = [];
                this.ret = {};
                this.control_ret = {};
            }

            FunctionDef.prototype.signature = null;
            FunctionDef.prototype.attr = {};
            FunctionDef.prototype.arg_attr = {};
            FunctionDef.prototype.resource_arg_unique_id = {};
            FunctionDef.prototype.node_def = [];
            FunctionDef.prototype.ret = {};
            FunctionDef.prototype.control_ret = {};

            FunctionDef.decode = function (reader, length) {
                const message = new $root.tensorflow.FunctionDef();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.signature = $root.tensorflow.OpDef.decode(reader, reader.uint32());
                            break;
                        case 5:
                            reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                            break;
                        case 7:
                            reader.pair(message.arg_attr, () => reader.uint32(), () => $root.tensorflow.FunctionDef.ArgAttrs.decode(reader, reader.uint32()));
                            break;
                        case 8:
                            reader.pair(message.resource_arg_unique_id, () => reader.uint32(), () => reader.uint32());
                            break;
                        case 3:
                            message.node_def.push($root.tensorflow.NodeDef.decode(reader, reader.uint32()));
                            break;
                        case 4:
                            reader.pair(message.ret, () => reader.string(), () => reader.string());
                            break;
                        case 6:
                            reader.pair(message.control_ret, () => reader.string(), () => reader.string());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            FunctionDef.decodeText = function (reader) {
                const message = new $root.tensorflow.FunctionDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "signature":
                            message.signature = $root.tensorflow.OpDef.decodeText(reader, true);
                            break;
                        case "attr":
                            reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader, true));
                            break;
                        case "arg_attr":
                            reader.pair(message.arg_attr, () => reader.uint32(), () => $root.tensorflow.FunctionDef.ArgAttrs.decodeText(reader, true));
                            break;
                        case "resource_arg_unique_id":
                            reader.pair(message.resource_arg_unique_id, () => reader.uint32(), () => reader.uint32());
                            break;
                        case "node_def":
                            message.node_def.push($root.tensorflow.NodeDef.decodeText(reader, true));
                            break;
                        case "ret":
                            reader.pair(message.ret, () => reader.string(), () => reader.string());
                            break;
                        case "control_ret":
                            reader.pair(message.control_ret, () => reader.string(), () => reader.string());
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            FunctionDef.ArgAttrs = (function() {

                function ArgAttrs() {
                    this.attr = {};
                }

                ArgAttrs.prototype.attr = {};

                ArgAttrs.decode = function (reader, length) {
                    const message = new $root.tensorflow.FunctionDef.ArgAttrs();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                ArgAttrs.decodeText = function (reader) {
                    const message = new $root.tensorflow.FunctionDef.ArgAttrs();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "attr":
                                reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader, true));
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ArgAttrs;
            })();

            return FunctionDef;
        })();

        tensorflow.GradientDef = (function() {

            function GradientDef() {
            }

            GradientDef.prototype.function_name = "";
            GradientDef.prototype.gradient_func = "";

            GradientDef.decode = function (reader, length) {
                const message = new $root.tensorflow.GradientDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            GradientDef.decodeText = function (reader) {
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
            };

            return GradientDef;
        })();

        tensorflow.AttrValue = (function() {

            function AttrValue() {
            }

            AttrValue.prototype.s = new Uint8Array([]);
            AttrValue.prototype.i = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            AttrValue.prototype.f = 0;
            AttrValue.prototype.b = false;
            AttrValue.prototype.type = 0;
            AttrValue.prototype.shape = null;
            AttrValue.prototype.tensor = null;
            AttrValue.prototype.list = null;
            AttrValue.prototype.func = null;
            AttrValue.prototype.placeholder = "";

            const valueSet = new Set([ "s", "i", "f", "b", "type", "shape", "tensor", "list", "func", "placeholder"]);
            Object.defineProperty(AttrValue.prototype, "value", {
                get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
            });

            AttrValue.decode = function (reader, length) {
                const message = new $root.tensorflow.AttrValue();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            AttrValue.decodeText = function (reader) {
                const message = new $root.tensorflow.AttrValue();
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
                            message.type = reader.enum($root.tensorflow.DataType);
                            break;
                        case "shape":
                            message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                            break;
                        case "tensor":
                            message.tensor = $root.tensorflow.TensorProto.decodeText(reader, true);
                            break;
                        case "list":
                            message.list = $root.tensorflow.AttrValue.ListValue.decodeText(reader, true);
                            break;
                        case "func":
                            message.func = $root.tensorflow.NameAttrList.decodeText(reader, true);
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
            };

            AttrValue.ListValue = (function() {

                function ListValue() {
                    this.s = [];
                    this.i = [];
                    this.f = [];
                    this.b = [];
                    this.type = [];
                    this.shape = [];
                    this.tensor = [];
                    this.func = [];
                }

                ListValue.prototype.s = [];
                ListValue.prototype.i = [];
                ListValue.prototype.f = [];
                ListValue.prototype.b = [];
                ListValue.prototype.type = [];
                ListValue.prototype.shape = [];
                ListValue.prototype.tensor = [];
                ListValue.prototype.func = [];

                ListValue.decode = function (reader, length) {
                    const message = new $root.tensorflow.AttrValue.ListValue();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                ListValue.decodeText = function (reader) {
                    const message = new $root.tensorflow.AttrValue.ListValue();
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
                                reader.array(message.type, () => reader.enum($root.tensorflow.DataType));
                                break;
                            case "shape":
                                message.shape.push($root.tensorflow.TensorShapeProto.decodeText(reader, true));
                                break;
                            case "tensor":
                                message.tensor.push($root.tensorflow.TensorProto.decodeText(reader, true));
                                break;
                            case "func":
                                message.func.push($root.tensorflow.NameAttrList.decodeText(reader, true));
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ListValue;
            })();

            return AttrValue;
        })();

        tensorflow.NameAttrList = (function() {

            function NameAttrList() {
                this.attr = {};
            }

            NameAttrList.prototype.name = "";
            NameAttrList.prototype.attr = {};

            NameAttrList.decode = function (reader, length) {
                const message = new $root.tensorflow.NameAttrList();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.name = reader.string();
                            break;
                        case 2:
                            reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            NameAttrList.decodeText = function (reader) {
                const message = new $root.tensorflow.NameAttrList();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "attr":
                            reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return NameAttrList;
        })();

        tensorflow.TensorProto = (function() {

            function TensorProto() {
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

            TensorProto.prototype.dtype = 0;
            TensorProto.prototype.tensor_shape = null;
            TensorProto.prototype.version_number = 0;
            TensorProto.prototype.tensor_content = new Uint8Array([]);
            TensorProto.prototype.half_val = [];
            TensorProto.prototype.float_val = [];
            TensorProto.prototype.double_val = [];
            TensorProto.prototype.int_val = [];
            TensorProto.prototype.string_val = [];
            TensorProto.prototype.scomplex_val = [];
            TensorProto.prototype.int64_val = [];
            TensorProto.prototype.bool_val = [];
            TensorProto.prototype.dcomplex_val = [];
            TensorProto.prototype.resource_handle_val = [];
            TensorProto.prototype.variant_val = [];
            TensorProto.prototype.uint32_val = [];
            TensorProto.prototype.uint64_val = [];

            TensorProto.decode = function (reader, length) {
                const message = new $root.tensorflow.TensorProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TensorProto.decodeText = function (reader) {
                const message = new $root.tensorflow.TensorProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dtype":
                            message.dtype = reader.enum($root.tensorflow.DataType);
                            break;
                        case "tensor_shape":
                            message.tensor_shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
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
                            message.resource_handle_val.push($root.tensorflow.ResourceHandleProto.decodeText(reader, true));
                            break;
                        case "variant_val":
                            message.variant_val.push($root.tensorflow.VariantTensorDataProto.decodeText(reader, true));
                            break;
                        case "uint32_val":
                            reader.array(message.uint32_val, () => reader.uint32());
                            break;
                        case "uint64_val":
                            reader.array(message.uint64_val, () => reader.uint64());
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return TensorProto;
        })();

        tensorflow.VariantTensorDataProto = (function() {

            function VariantTensorDataProto() {
                this.tensors = [];
            }

            VariantTensorDataProto.prototype.type_name = "";
            VariantTensorDataProto.prototype.metadata = new Uint8Array([]);
            VariantTensorDataProto.prototype.tensors = [];

            VariantTensorDataProto.decode = function (reader, length) {
                const message = new $root.tensorflow.VariantTensorDataProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            VariantTensorDataProto.decodeText = function (reader) {
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
                            message.tensors.push($root.tensorflow.TensorProto.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return VariantTensorDataProto;
        })();

        tensorflow.ResourceHandleProto = (function() {

            function ResourceHandleProto() {
                this.dtypes_and_shapes = [];
            }

            ResourceHandleProto.prototype.device = "";
            ResourceHandleProto.prototype.container = "";
            ResourceHandleProto.prototype.name = "";
            ResourceHandleProto.prototype.hash_code = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
            ResourceHandleProto.prototype.maybe_type_name = "";
            ResourceHandleProto.prototype.dtypes_and_shapes = [];

            ResourceHandleProto.decode = function (reader, length) {
                const message = new $root.tensorflow.ResourceHandleProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ResourceHandleProto.decodeText = function (reader) {
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
                            message.hash_code = reader.uint64();
                            break;
                        case "maybe_type_name":
                            message.maybe_type_name = reader.string();
                            break;
                        case "dtypes_and_shapes":
                            message.dtypes_and_shapes.push($root.tensorflow.ResourceHandleProto.DtypeAndShape.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            ResourceHandleProto.DtypeAndShape = (function() {

                function DtypeAndShape() {
                }

                DtypeAndShape.prototype.dtype = 0;
                DtypeAndShape.prototype.shape = null;

                DtypeAndShape.decode = function (reader, length) {
                    const message = new $root.tensorflow.ResourceHandleProto.DtypeAndShape();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                DtypeAndShape.decodeText = function (reader) {
                    const message = new $root.tensorflow.ResourceHandleProto.DtypeAndShape();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "dtype":
                                message.dtype = reader.enum($root.tensorflow.DataType);
                                break;
                            case "shape":
                                message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return DtypeAndShape;
            })();

            return ResourceHandleProto;
        })();

        tensorflow.TensorShapeProto = (function() {

            function TensorShapeProto() {
                this.dim = [];
            }

            TensorShapeProto.prototype.dim = [];
            TensorShapeProto.prototype.unknown_rank = false;

            TensorShapeProto.decode = function (reader, length) {
                const message = new $root.tensorflow.TensorShapeProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TensorShapeProto.decodeText = function (reader) {
                const message = new $root.tensorflow.TensorShapeProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dim":
                            message.dim.push($root.tensorflow.TensorShapeProto.Dim.decodeText(reader, true));
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
            };

            TensorShapeProto.Dim = (function() {

                function Dim() {
                }

                Dim.prototype.size = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Dim.prototype.name = "";

                Dim.decode = function (reader, length) {
                    const message = new $root.tensorflow.TensorShapeProto.Dim();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                Dim.decodeText = function (reader) {
                    const message = new $root.tensorflow.TensorShapeProto.Dim();
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
                };

                return Dim;
            })();

            return TensorShapeProto;
        })();

        tensorflow.DataType = (function() {
            const values = {};
            values["DT_INVALID"] = 0;
            values["DT_FLOAT"] = 1;
            values["DT_DOUBLE"] = 2;
            values["DT_INT32"] = 3;
            values["DT_UINT8"] = 4;
            values["DT_INT16"] = 5;
            values["DT_INT8"] = 6;
            values["DT_STRING"] = 7;
            values["DT_COMPLEX64"] = 8;
            values["DT_INT64"] = 9;
            values["DT_BOOL"] = 10;
            values["DT_QINT8"] = 11;
            values["DT_QUINT8"] = 12;
            values["DT_QINT32"] = 13;
            values["DT_BFLOAT16"] = 14;
            values["DT_QINT16"] = 15;
            values["DT_QUINT16"] = 16;
            values["DT_UINT16"] = 17;
            values["DT_COMPLEX128"] = 18;
            values["DT_HALF"] = 19;
            values["DT_RESOURCE"] = 20;
            values["DT_VARIANT"] = 21;
            values["DT_UINT32"] = 22;
            values["DT_UINT64"] = 23;
            values["DT_FLOAT_REF"] = 101;
            values["DT_DOUBLE_REF"] = 102;
            values["DT_INT32_REF"] = 103;
            values["DT_UINT8_REF"] = 104;
            values["DT_INT16_REF"] = 105;
            values["DT_INT8_REF"] = 106;
            values["DT_STRING_REF"] = 107;
            values["DT_COMPLEX64_REF"] = 108;
            values["DT_INT64_REF"] = 109;
            values["DT_BOOL_REF"] = 110;
            values["DT_QINT8_REF"] = 111;
            values["DT_QUINT8_REF"] = 112;
            values["DT_QINT32_REF"] = 113;
            values["DT_BFLOAT16_REF"] = 114;
            values["DT_QINT16_REF"] = 115;
            values["DT_QUINT16_REF"] = 116;
            values["DT_UINT16_REF"] = 117;
            values["DT_COMPLEX128_REF"] = 118;
            values["DT_HALF_REF"] = 119;
            values["DT_RESOURCE_REF"] = 120;
            values["DT_VARIANT_REF"] = 121;
            values["DT_UINT32_REF"] = 122;
            values["DT_UINT64_REF"] = 123;
            return values;
        })();

        tensorflow.NodeDef = (function() {

            function NodeDef() {
                this.input = [];
                this.attr = {};
            }

            NodeDef.prototype.name = "";
            NodeDef.prototype.op = "";
            NodeDef.prototype.input = [];
            NodeDef.prototype.device = "";
            NodeDef.prototype.attr = {};
            NodeDef.prototype.experimental_debug_info = null;

            NodeDef.decode = function (reader, length) {
                const message = new $root.tensorflow.NodeDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
                            reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decode(reader, reader.uint32()));
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
            };

            NodeDef.decodeText = function (reader) {
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
                            reader.pair(message.attr, () => reader.string(), () => $root.tensorflow.AttrValue.decodeText(reader, true));
                            break;
                        case "experimental_debug_info":
                            message.experimental_debug_info = $root.tensorflow.NodeDef.ExperimentalDebugInfo.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            NodeDef.ExperimentalDebugInfo = (function() {

                function ExperimentalDebugInfo() {
                    this.original_node_names = [];
                    this.original_func_names = [];
                }

                ExperimentalDebugInfo.prototype.original_node_names = [];
                ExperimentalDebugInfo.prototype.original_func_names = [];

                ExperimentalDebugInfo.decode = function (reader, length) {
                    const message = new $root.tensorflow.NodeDef.ExperimentalDebugInfo();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                ExperimentalDebugInfo.decodeText = function (reader) {
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
                };

                return ExperimentalDebugInfo;
            })();

            return NodeDef;
        })();

        tensorflow.OpDef = (function() {

            function OpDef() {
                this.input_arg = [];
                this.output_arg = [];
                this.control_output = [];
                this.attr = [];
            }

            OpDef.prototype.name = "";
            OpDef.prototype.input_arg = [];
            OpDef.prototype.output_arg = [];
            OpDef.prototype.control_output = [];
            OpDef.prototype.attr = [];
            OpDef.prototype.deprecation = null;
            OpDef.prototype.summary = "";
            OpDef.prototype.description = "";
            OpDef.prototype.is_commutative = false;
            OpDef.prototype.is_aggregate = false;
            OpDef.prototype.is_stateful = false;
            OpDef.prototype.allows_uninitialized_input = false;

            OpDef.decode = function (reader, length) {
                const message = new $root.tensorflow.OpDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            OpDef.decodeText = function (reader) {
                const message = new $root.tensorflow.OpDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "input_arg":
                            message.input_arg.push($root.tensorflow.OpDef.ArgDef.decodeText(reader, true));
                            break;
                        case "output_arg":
                            message.output_arg.push($root.tensorflow.OpDef.ArgDef.decodeText(reader, true));
                            break;
                        case "control_output":
                            reader.array(message.control_output, () => reader.string());
                            break;
                        case "attr":
                            message.attr.push($root.tensorflow.OpDef.AttrDef.decodeText(reader, true));
                            break;
                        case "deprecation":
                            message.deprecation = $root.tensorflow.OpDeprecation.decodeText(reader, true);
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
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            OpDef.ArgDef = (function() {

                function ArgDef() {
                }

                ArgDef.prototype.name = "";
                ArgDef.prototype.description = "";
                ArgDef.prototype.type = 0;
                ArgDef.prototype.type_attr = "";
                ArgDef.prototype.number_attr = "";
                ArgDef.prototype.type_list_attr = "";
                ArgDef.prototype.is_ref = false;

                ArgDef.decode = function (reader, length) {
                    const message = new $root.tensorflow.OpDef.ArgDef();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                            case 16:
                                message.is_ref = reader.bool();
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                ArgDef.decodeText = function (reader) {
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
                            case "is_ref":
                                message.is_ref = reader.bool();
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ArgDef;
            })();

            OpDef.AttrDef = (function() {

                function AttrDef() {
                }

                AttrDef.prototype.name = "";
                AttrDef.prototype.type = "";
                AttrDef.prototype.default_value = null;
                AttrDef.prototype.description = "";
                AttrDef.prototype.has_minimum = false;
                AttrDef.prototype.minimum = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                AttrDef.prototype.allowed_values = null;

                AttrDef.decode = function (reader, length) {
                    const message = new $root.tensorflow.OpDef.AttrDef();
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
                };

                AttrDef.decodeText = function (reader) {
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
                                message.default_value = $root.tensorflow.AttrValue.decodeText(reader, true);
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
                                message.allowed_values = $root.tensorflow.AttrValue.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return AttrDef;
            })();

            return OpDef;
        })();

        tensorflow.OpDeprecation = (function() {

            function OpDeprecation() {
            }

            OpDeprecation.prototype.version = 0;
            OpDeprecation.prototype.explanation = "";

            OpDeprecation.decode = function (reader, length) {
                const message = new $root.tensorflow.OpDeprecation();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            OpDeprecation.decodeText = function (reader) {
                const message = new $root.tensorflow.OpDeprecation();
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
            };

            return OpDeprecation;
        })();

        tensorflow.OpList = (function() {

            function OpList() {
                this.op = [];
            }

            OpList.prototype.op = [];

            OpList.decode = function (reader, length) {
                const message = new $root.tensorflow.OpList();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            OpList.decodeText = function (reader) {
                const message = new $root.tensorflow.OpList();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "op":
                            message.op.push($root.tensorflow.OpDef.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return OpList;
        })();

        tensorflow.VersionDef = (function() {

            function VersionDef() {
                this.bad_consumers = [];
            }

            VersionDef.prototype.producer = 0;
            VersionDef.prototype.min_consumer = 0;
            VersionDef.prototype.bad_consumers = [];

            VersionDef.decode = function (reader, length) {
                const message = new $root.tensorflow.VersionDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            VersionDef.decodeText = function (reader) {
                const message = new $root.tensorflow.VersionDef();
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
            };

            return VersionDef;
        })();

        tensorflow.SavedObjectGraph = (function() {

            function SavedObjectGraph() {
                this.nodes = [];
                this.concrete_functions = {};
            }

            SavedObjectGraph.prototype.nodes = [];
            SavedObjectGraph.prototype.concrete_functions = {};

            SavedObjectGraph.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedObjectGraph();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.nodes.push($root.tensorflow.SavedObject.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            reader.pair(message.concrete_functions, () => reader.string(), () => $root.tensorflow.SavedConcreteFunction.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            SavedObjectGraph.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedObjectGraph();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "nodes":
                            message.nodes.push($root.tensorflow.SavedObject.decodeText(reader, true));
                            break;
                        case "concrete_functions":
                            reader.pair(message.concrete_functions, () => reader.string(), () => $root.tensorflow.SavedConcreteFunction.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedObjectGraph;
        })();

        tensorflow.SavedObject = (function() {

            function SavedObject() {
                this.children = [];
                this.slot_variables = [];
                this.saveable_objects = {};
            }

            SavedObject.prototype.children = [];
            SavedObject.prototype.slot_variables = [];
            SavedObject.prototype.user_object = null;
            SavedObject.prototype.asset = null;
            SavedObject.prototype["function"] = null;
            SavedObject.prototype.variable = null;
            SavedObject.prototype.bare_concrete_function = null;
            SavedObject.prototype.constant = null;
            SavedObject.prototype.resource = null;
            SavedObject.prototype.saveable_objects = {};

            const kindSet = new Set([ "user_object", "asset", "function", "variable", "bare_concrete_function", "constant", "resource"]);
            Object.defineProperty(SavedObject.prototype, "kind", {
                get: function() { return Object.keys(this).find((key) => kindSet.has(key) && this[key] != null); }
            });

            SavedObject.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedObject();
                const end = reader.next(length);
                while (reader.end(end)) {
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
                            reader.pair(message.saveable_objects, () => reader.string(), () => $root.tensorflow.SaveableObject.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            SavedObject.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedObject();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "children":
                            message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader, true));
                            break;
                        case "slot_variables":
                            message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeText(reader, true));
                            break;
                        case "user_object":
                            message.user_object = $root.tensorflow.SavedUserObject.decodeText(reader, true);
                            break;
                        case "asset":
                            message.asset = $root.tensorflow.SavedAsset.decodeText(reader, true);
                            break;
                        case "function":
                            message["function"] = $root.tensorflow.SavedFunction.decodeText(reader, true);
                            break;
                        case "variable":
                            message.variable = $root.tensorflow.SavedVariable.decodeText(reader, true);
                            break;
                        case "bare_concrete_function":
                            message.bare_concrete_function = $root.tensorflow.SavedBareConcreteFunction.decodeText(reader, true);
                            break;
                        case "constant":
                            message.constant = $root.tensorflow.SavedConstant.decodeText(reader, true);
                            break;
                        case "resource":
                            message.resource = $root.tensorflow.SavedResource.decodeText(reader, true);
                            break;
                        case "saveable_objects":
                            reader.pair(message.saveable_objects, () => reader.string(), () => $root.tensorflow.SaveableObject.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedObject;
        })();

        tensorflow.SavedUserObject = (function() {

            function SavedUserObject() {
            }

            SavedUserObject.prototype.identifier = "";
            SavedUserObject.prototype.version = null;
            SavedUserObject.prototype.metadata = "";

            SavedUserObject.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedUserObject();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedUserObject.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedUserObject();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "identifier":
                            message.identifier = reader.string();
                            break;
                        case "version":
                            message.version = $root.tensorflow.VersionDef.decodeText(reader, true);
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
            };

            return SavedUserObject;
        })();

        tensorflow.SavedAsset = (function() {

            function SavedAsset() {
            }

            SavedAsset.prototype.asset_file_def_index = 0;

            SavedAsset.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedAsset();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedAsset.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedAsset();
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
            };

            return SavedAsset;
        })();

        tensorflow.SavedFunction = (function() {

            function SavedFunction() {
                this.concrete_functions = [];
            }

            SavedFunction.prototype.concrete_functions = [];
            SavedFunction.prototype.function_spec = null;

            SavedFunction.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedFunction();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedFunction.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedFunction();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "concrete_functions":
                            reader.array(message.concrete_functions, () => reader.string());
                            break;
                        case "function_spec":
                            message.function_spec = $root.tensorflow.FunctionSpec.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedFunction;
        })();

        tensorflow.SavedConcreteFunction = (function() {

            function SavedConcreteFunction() {
                this.bound_inputs = [];
            }

            SavedConcreteFunction.prototype.bound_inputs = [];
            SavedConcreteFunction.prototype.canonicalized_input_signature = null;
            SavedConcreteFunction.prototype.output_signature = null;

            SavedConcreteFunction.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedConcreteFunction();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedConcreteFunction.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedConcreteFunction();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "bound_inputs":
                            reader.array(message.bound_inputs, () => reader.int32());
                            break;
                        case "canonicalized_input_signature":
                            message.canonicalized_input_signature = $root.tensorflow.StructuredValue.decodeText(reader, true);
                            break;
                        case "output_signature":
                            message.output_signature = $root.tensorflow.StructuredValue.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedConcreteFunction;
        })();

        tensorflow.SavedBareConcreteFunction = (function() {

            function SavedBareConcreteFunction() {
                this.argument_keywords = [];
            }

            SavedBareConcreteFunction.prototype.concrete_function_name = "";
            SavedBareConcreteFunction.prototype.argument_keywords = [];
            SavedBareConcreteFunction.prototype.allowed_positional_arguments = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

            SavedBareConcreteFunction.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedBareConcreteFunction();
                const end = reader.next(length);
                while (reader.end(end)) {
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
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            SavedBareConcreteFunction.decodeText = function (reader) {
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
                            message.allowed_positional_arguments = reader.int64();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedBareConcreteFunction;
        })();

        tensorflow.SavedConstant = (function() {

            function SavedConstant() {
            }

            SavedConstant.prototype.operation = "";

            SavedConstant.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedConstant();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedConstant.decodeText = function (reader) {
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
            };

            return SavedConstant;
        })();

        tensorflow.SavedVariable = (function() {

            function SavedVariable() {
            }

            SavedVariable.prototype.dtype = 0;
            SavedVariable.prototype.shape = null;
            SavedVariable.prototype.trainable = false;
            SavedVariable.prototype.synchronization = 0;
            SavedVariable.prototype.aggregation = 0;
            SavedVariable.prototype.name = "";

            SavedVariable.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedVariable();
                const end = reader.next(length);
                while (reader.end(end)) {
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
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            SavedVariable.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedVariable();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dtype":
                            message.dtype = reader.enum($root.tensorflow.DataType);
                            break;
                        case "shape":
                            message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                            break;
                        case "trainable":
                            message.trainable = reader.bool();
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
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedVariable;
        })();

        tensorflow.FunctionSpec = (function() {

            function FunctionSpec() {
            }

            FunctionSpec.prototype.fullargspec = null;
            FunctionSpec.prototype.is_method = false;
            FunctionSpec.prototype.input_signature = null;

            FunctionSpec.decode = function (reader, length) {
                const message = new $root.tensorflow.FunctionSpec();
                const end = reader.next(length);
                while (reader.end(end)) {
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
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            FunctionSpec.decodeText = function (reader) {
                const message = new $root.tensorflow.FunctionSpec();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "fullargspec":
                            message.fullargspec = $root.tensorflow.StructuredValue.decodeText(reader, true);
                            break;
                        case "is_method":
                            message.is_method = reader.bool();
                            break;
                        case "input_signature":
                            message.input_signature = $root.tensorflow.StructuredValue.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return FunctionSpec;
        })();

        tensorflow.SavedResource = (function() {

            function SavedResource() {
            }

            SavedResource.prototype.device = "";

            SavedResource.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedResource();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedResource.decodeText = function (reader) {
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
            };

            return SavedResource;
        })();

        tensorflow.SaveableObject = (function() {

            function SaveableObject() {
            }

            SaveableObject.prototype.save_function = 0;
            SaveableObject.prototype.restore_function = 0;

            SaveableObject.decode = function (reader, length) {
                const message = new $root.tensorflow.SaveableObject();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SaveableObject.decodeText = function (reader) {
                const message = new $root.tensorflow.SaveableObject();
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
            };

            return SaveableObject;
        })();

        tensorflow.VariableSynchronization = (function() {
            const values = {};
            values["VARIABLE_SYNCHRONIZATION_AUTO"] = 0;
            values["VARIABLE_SYNCHRONIZATION_NONE"] = 1;
            values["VARIABLE_SYNCHRONIZATION_ON_WRITE"] = 2;
            values["VARIABLE_SYNCHRONIZATION_ON_READ"] = 3;
            return values;
        })();

        tensorflow.VariableAggregation = (function() {
            const values = {};
            values["VARIABLE_AGGREGATION_NONE"] = 0;
            values["VARIABLE_AGGREGATION_SUM"] = 1;
            values["VARIABLE_AGGREGATION_MEAN"] = 2;
            values["VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA"] = 3;
            return values;
        })();

        tensorflow.VariableDef = (function() {

            function VariableDef() {
            }

            VariableDef.prototype.variable_name = "";
            VariableDef.prototype.initial_value_name = "";
            VariableDef.prototype.initializer_name = "";
            VariableDef.prototype.snapshot_name = "";
            VariableDef.prototype.save_slice_info_def = null;
            VariableDef.prototype.is_resource = false;
            VariableDef.prototype.trainable = false;
            VariableDef.prototype.synchronization = 0;
            VariableDef.prototype.aggregation = 0;

            VariableDef.decode = function (reader, length) {
                const message = new $root.tensorflow.VariableDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            VariableDef.decodeText = function (reader) {
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
                            message.save_slice_info_def = $root.tensorflow.SaveSliceInfoDef.decodeText(reader, true);
                            break;
                        case "is_resource":
                            message.is_resource = reader.bool();
                            break;
                        case "trainable":
                            message.trainable = reader.bool();
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
            };

            return VariableDef;
        })();

        tensorflow.SaveSliceInfoDef = (function() {

            function SaveSliceInfoDef() {
                this.full_shape = [];
                this.var_offset = [];
                this.var_shape = [];
            }

            SaveSliceInfoDef.prototype.full_name = "";
            SaveSliceInfoDef.prototype.full_shape = [];
            SaveSliceInfoDef.prototype.var_offset = [];
            SaveSliceInfoDef.prototype.var_shape = [];

            SaveSliceInfoDef.decode = function (reader, length) {
                const message = new $root.tensorflow.SaveSliceInfoDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SaveSliceInfoDef.decodeText = function (reader) {
                const message = new $root.tensorflow.SaveSliceInfoDef();
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
            };

            return SaveSliceInfoDef;
        })();

        tensorflow.StructuredValue = (function() {

            function StructuredValue() {
            }

            StructuredValue.prototype.none_value = null;
            StructuredValue.prototype.float64_value = 0;
            StructuredValue.prototype.int64_value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            StructuredValue.prototype.string_value = "";
            StructuredValue.prototype.bool_value = false;
            StructuredValue.prototype.tensor_shape_value = null;
            StructuredValue.prototype.tensor_dtype_value = 0;
            StructuredValue.prototype.tensor_spec_value = null;
            StructuredValue.prototype.type_spec_value = null;
            StructuredValue.prototype.bounded_tensor_spec_value = null;
            StructuredValue.prototype.list_value = null;
            StructuredValue.prototype.tuple_value = null;
            StructuredValue.prototype.dict_value = null;
            StructuredValue.prototype.named_tuple_value = null;

            const kindSet = new Set([ "none_value", "float64_value", "int64_value", "string_value", "bool_value", "tensor_shape_value", "tensor_dtype_value", "tensor_spec_value", "type_spec_value", "bounded_tensor_spec_value", "list_value", "tuple_value", "dict_value", "named_tuple_value"]);
            Object.defineProperty(StructuredValue.prototype, "kind", {
                get: function() { return Object.keys(this).find((key) => kindSet.has(key) && this[key] != null); }
            });

            StructuredValue.decode = function (reader, length) {
                const message = new $root.tensorflow.StructuredValue();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            StructuredValue.decodeText = function (reader) {
                const message = new $root.tensorflow.StructuredValue();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "none_value":
                            message.none_value = $root.tensorflow.NoneValue.decodeText(reader, true);
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
                            message.tensor_shape_value = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                            break;
                        case "tensor_dtype_value":
                            message.tensor_dtype_value = reader.enum($root.tensorflow.DataType);
                            break;
                        case "tensor_spec_value":
                            message.tensor_spec_value = $root.tensorflow.TensorSpecProto.decodeText(reader, true);
                            break;
                        case "type_spec_value":
                            message.type_spec_value = $root.tensorflow.TypeSpecProto.decodeText(reader, true);
                            break;
                        case "bounded_tensor_spec_value":
                            message.bounded_tensor_spec_value = $root.tensorflow.BoundedTensorSpecProto.decodeText(reader, true);
                            break;
                        case "list_value":
                            message.list_value = $root.tensorflow.ListValue.decodeText(reader, true);
                            break;
                        case "tuple_value":
                            message.tuple_value = $root.tensorflow.TupleValue.decodeText(reader, true);
                            break;
                        case "dict_value":
                            message.dict_value = $root.tensorflow.DictValue.decodeText(reader, true);
                            break;
                        case "named_tuple_value":
                            message.named_tuple_value = $root.tensorflow.NamedTupleValue.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return StructuredValue;
        })();

        tensorflow.NoneValue = (function() {

            function NoneValue() {
            }

            NoneValue.decode = function (reader, length) {
                const message = new $root.tensorflow.NoneValue();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            NoneValue.decodeText = function (reader) {
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
            };

            return NoneValue;
        })();

        tensorflow.ListValue = (function() {

            function ListValue() {
                this.values = [];
            }

            ListValue.prototype.values = [];

            ListValue.decode = function (reader, length) {
                const message = new $root.tensorflow.ListValue();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            ListValue.decodeText = function (reader) {
                const message = new $root.tensorflow.ListValue();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "values":
                            message.values.push($root.tensorflow.StructuredValue.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return ListValue;
        })();

        tensorflow.TupleValue = (function() {

            function TupleValue() {
                this.values = [];
            }

            TupleValue.prototype.values = [];

            TupleValue.decode = function (reader, length) {
                const message = new $root.tensorflow.TupleValue();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TupleValue.decodeText = function (reader) {
                const message = new $root.tensorflow.TupleValue();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "values":
                            message.values.push($root.tensorflow.StructuredValue.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return TupleValue;
        })();

        tensorflow.DictValue = (function() {

            function DictValue() {
                this.fields = {};
            }

            DictValue.prototype.fields = {};

            DictValue.decode = function (reader, length) {
                const message = new $root.tensorflow.DictValue();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            reader.pair(message.fields, () => reader.string(), () => $root.tensorflow.StructuredValue.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            DictValue.decodeText = function (reader) {
                const message = new $root.tensorflow.DictValue();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "fields":
                            reader.pair(message.fields, () => reader.string(), () => $root.tensorflow.StructuredValue.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return DictValue;
        })();

        tensorflow.PairValue = (function() {

            function PairValue() {
            }

            PairValue.prototype.key = "";
            PairValue.prototype.value = null;

            PairValue.decode = function (reader, length) {
                const message = new $root.tensorflow.PairValue();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            PairValue.decodeText = function (reader) {
                const message = new $root.tensorflow.PairValue();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "key":
                            message.key = reader.string();
                            break;
                        case "value":
                            message.value = $root.tensorflow.StructuredValue.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return PairValue;
        })();

        tensorflow.NamedTupleValue = (function() {

            function NamedTupleValue() {
                this.values = [];
            }

            NamedTupleValue.prototype.name = "";
            NamedTupleValue.prototype.values = [];

            NamedTupleValue.decode = function (reader, length) {
                const message = new $root.tensorflow.NamedTupleValue();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            NamedTupleValue.decodeText = function (reader) {
                const message = new $root.tensorflow.NamedTupleValue();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "values":
                            message.values.push($root.tensorflow.PairValue.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return NamedTupleValue;
        })();

        tensorflow.TensorSpecProto = (function() {

            function TensorSpecProto() {
            }

            TensorSpecProto.prototype.name = "";
            TensorSpecProto.prototype.shape = null;
            TensorSpecProto.prototype.dtype = 0;

            TensorSpecProto.decode = function (reader, length) {
                const message = new $root.tensorflow.TensorSpecProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TensorSpecProto.decodeText = function (reader) {
                const message = new $root.tensorflow.TensorSpecProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "shape":
                            message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
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
            };

            return TensorSpecProto;
        })();

        tensorflow.BoundedTensorSpecProto = (function() {

            function BoundedTensorSpecProto() {
            }

            BoundedTensorSpecProto.prototype.name = "";
            BoundedTensorSpecProto.prototype.shape = null;
            BoundedTensorSpecProto.prototype.dtype = 0;
            BoundedTensorSpecProto.prototype.minimum = null;
            BoundedTensorSpecProto.prototype.maximum = null;

            BoundedTensorSpecProto.decode = function (reader, length) {
                const message = new $root.tensorflow.BoundedTensorSpecProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BoundedTensorSpecProto.decodeText = function (reader) {
                const message = new $root.tensorflow.BoundedTensorSpecProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "shape":
                            message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                            break;
                        case "dtype":
                            message.dtype = reader.enum($root.tensorflow.DataType);
                            break;
                        case "minimum":
                            message.minimum = $root.tensorflow.TensorProto.decodeText(reader, true);
                            break;
                        case "maximum":
                            message.maximum = $root.tensorflow.TensorProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return BoundedTensorSpecProto;
        })();

        tensorflow.TypeSpecProto = (function() {

            function TypeSpecProto() {
            }

            TypeSpecProto.prototype.type_spec_class = 0;
            TypeSpecProto.prototype.type_state = null;
            TypeSpecProto.prototype.type_spec_class_name = "";

            TypeSpecProto.decode = function (reader, length) {
                const message = new $root.tensorflow.TypeSpecProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TypeSpecProto.decodeText = function (reader) {
                const message = new $root.tensorflow.TypeSpecProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "type_spec_class":
                            message.type_spec_class = reader.enum($root.tensorflow.TypeSpecProto.TypeSpecClass);
                            break;
                        case "type_state":
                            message.type_state = $root.tensorflow.StructuredValue.decodeText(reader, true);
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
            };

            TypeSpecProto.TypeSpecClass = (function() {
                const values = {};
                values["UNKNOWN"] = 0;
                values["SPARSE_TENSOR_SPEC"] = 1;
                values["INDEXED_SLICES_SPEC"] = 2;
                values["RAGGED_TENSOR_SPEC"] = 3;
                values["TENSOR_ARRAY_SPEC"] = 4;
                values["DATA_DATASET_SPEC"] = 5;
                values["DATA_ITERATOR_SPEC"] = 6;
                values["OPTIONAL_SPEC"] = 7;
                values["PER_REPLICA_SPEC"] = 8;
                values["VARIABLE_SPEC"] = 9;
                values["ROW_PARTITION_SPEC"] = 10;
                return values;
            })();

            return TypeSpecProto;
        })();

        tensorflow.TrackableObjectGraph = (function() {

            function TrackableObjectGraph() {
                this.nodes = [];
            }

            TrackableObjectGraph.prototype.nodes = [];

            TrackableObjectGraph.decode = function (reader, length) {
                const message = new $root.tensorflow.TrackableObjectGraph();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TrackableObjectGraph.decodeText = function (reader) {
                const message = new $root.tensorflow.TrackableObjectGraph();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "nodes":
                            message.nodes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            TrackableObjectGraph.TrackableObject = (function() {

                function TrackableObject() {
                    this.children = [];
                    this.attributes = [];
                    this.slot_variables = [];
                }

                TrackableObject.prototype.children = [];
                TrackableObject.prototype.attributes = [];
                TrackableObject.prototype.slot_variables = [];

                TrackableObject.decode = function (reader, length) {
                    const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                TrackableObject.decodeText = function (reader) {
                    const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "children":
                                message.children.push($root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference.decodeText(reader, true));
                                break;
                            case "attributes":
                                message.attributes.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor.decodeText(reader, true));
                                break;
                            case "slot_variables":
                                message.slot_variables.push($root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference.decodeText(reader, true));
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                TrackableObject.ObjectReference = (function() {

                    function ObjectReference() {
                    }

                    ObjectReference.prototype.node_id = 0;
                    ObjectReference.prototype.local_name = "";

                    ObjectReference.decode = function (reader, length) {
                        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    ObjectReference.decodeText = function (reader) {
                        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference();
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
                    };

                    return ObjectReference;
                })();

                TrackableObject.SerializedTensor = (function() {

                    function SerializedTensor() {
                    }

                    SerializedTensor.prototype.name = "";
                    SerializedTensor.prototype.full_name = "";
                    SerializedTensor.prototype.checkpoint_key = "";
                    SerializedTensor.prototype.optional_restore = false;

                    SerializedTensor.decode = function (reader, length) {
                        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    SerializedTensor.decodeText = function (reader) {
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
                                    message.optional_restore = reader.bool();
                                    break;
                                default:
                                    reader.field(tag, message);
                                    break;
                            }
                        }
                        return message;
                    };

                    return SerializedTensor;
                })();

                TrackableObject.SlotVariableReference = (function() {

                    function SlotVariableReference() {
                    }

                    SlotVariableReference.prototype.original_variable_node_id = 0;
                    SlotVariableReference.prototype.slot_name = "";
                    SlotVariableReference.prototype.slot_variable_node_id = 0;

                    SlotVariableReference.decode = function (reader, length) {
                        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    SlotVariableReference.decodeText = function (reader) {
                        const message = new $root.tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference();
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
                    };

                    return SlotVariableReference;
                })();

                return TrackableObject;
            })();

            return TrackableObjectGraph;
        })();

        tensorflow.SaverDef = (function() {

            function SaverDef() {
            }

            SaverDef.prototype.filename_tensor_name = "";
            SaverDef.prototype.save_tensor_name = "";
            SaverDef.prototype.restore_op_name = "";
            SaverDef.prototype.max_to_keep = 0;
            SaverDef.prototype.sharded = false;
            SaverDef.prototype.keep_checkpoint_every_n_hours = 0;
            SaverDef.prototype.version = 0;

            SaverDef.decode = function (reader, length) {
                const message = new $root.tensorflow.SaverDef();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SaverDef.decodeText = function (reader) {
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
                            message.max_to_keep = reader.int32();
                            break;
                        case "sharded":
                            message.sharded = reader.bool();
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
            };

            SaverDef.CheckpointFormatVersion = (function() {
                const values = {};
                values["LEGACY"] = 0;
                values["V1"] = 1;
                values["V2"] = 2;
                return values;
            })();

            return SaverDef;
        })();

        tensorflow.BundleHeaderProto = (function() {

            function BundleHeaderProto() {
            }

            BundleHeaderProto.prototype.num_shards = 0;
            BundleHeaderProto.prototype.endianness = 0;
            BundleHeaderProto.prototype.version = null;

            BundleHeaderProto.decode = function (reader, length) {
                const message = new $root.tensorflow.BundleHeaderProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BundleHeaderProto.decodeText = function (reader) {
                const message = new $root.tensorflow.BundleHeaderProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "num_shards":
                            message.num_shards = reader.int32();
                            break;
                        case "endianness":
                            message.endianness = reader.enum($root.tensorflow.BundleHeaderProto.Endianness);
                            break;
                        case "version":
                            message.version = $root.tensorflow.VersionDef.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            BundleHeaderProto.Endianness = (function() {
                const values = {};
                values["LITTLE"] = 0;
                values["BIG"] = 1;
                return values;
            })();

            return BundleHeaderProto;
        })();

        tensorflow.BundleEntryProto = (function() {

            function BundleEntryProto() {
                this.slices = [];
            }

            BundleEntryProto.prototype.dtype = 0;
            BundleEntryProto.prototype.shape = null;
            BundleEntryProto.prototype.shard_id = 0;
            BundleEntryProto.prototype.offset = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            BundleEntryProto.prototype.size = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            BundleEntryProto.prototype.crc32c = 0;
            BundleEntryProto.prototype.slices = [];

            BundleEntryProto.decode = function (reader, length) {
                const message = new $root.tensorflow.BundleEntryProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            BundleEntryProto.decodeText = function (reader) {
                const message = new $root.tensorflow.BundleEntryProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dtype":
                            message.dtype = reader.enum($root.tensorflow.DataType);
                            break;
                        case "shape":
                            message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
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
                            message.slices.push($root.tensorflow.TensorSliceProto.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return BundleEntryProto;
        })();

        tensorflow.TensorSliceProto = (function() {

            function TensorSliceProto() {
                this.extent = [];
            }

            TensorSliceProto.prototype.extent = [];

            TensorSliceProto.decode = function (reader, length) {
                const message = new $root.tensorflow.TensorSliceProto();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            TensorSliceProto.decodeText = function (reader) {
                const message = new $root.tensorflow.TensorSliceProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "extent":
                            message.extent.push($root.tensorflow.TensorSliceProto.Extent.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            TensorSliceProto.Extent = (function() {

                function Extent() {
                }

                Extent.prototype.start = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Extent.prototype.length = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                const has_lengthSet = new Set([ "length"]);
                Object.defineProperty(Extent.prototype, "has_length", {
                    get: function() { return Object.keys(this).find((key) => has_lengthSet.has(key) && this[key] != null); }
                });

                Extent.decode = function (reader, length) {
                    const message = new $root.tensorflow.TensorSliceProto.Extent();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                Extent.decodeText = function (reader) {
                    const message = new $root.tensorflow.TensorSliceProto.Extent();
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
                };

                return Extent;
            })();

            return TensorSliceProto;
        })();

        tensorflow.SavedSliceMeta = (function() {

            function SavedSliceMeta() {
                this.slice = [];
            }

            SavedSliceMeta.prototype.name = "";
            SavedSliceMeta.prototype.shape = null;
            SavedSliceMeta.prototype.type = 0;
            SavedSliceMeta.prototype.slice = [];

            SavedSliceMeta.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedSliceMeta();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedSliceMeta.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedSliceMeta();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "shape":
                            message.shape = $root.tensorflow.TensorShapeProto.decodeText(reader, true);
                            break;
                        case "type":
                            message.type = reader.enum($root.tensorflow.DataType);
                            break;
                        case "slice":
                            message.slice.push($root.tensorflow.TensorSliceProto.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedSliceMeta;
        })();

        tensorflow.SavedTensorSliceMeta = (function() {

            function SavedTensorSliceMeta() {
                this.tensor = [];
            }

            SavedTensorSliceMeta.prototype.tensor = [];
            SavedTensorSliceMeta.prototype.versions = null;

            SavedTensorSliceMeta.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedTensorSliceMeta();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedTensorSliceMeta.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedTensorSliceMeta();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "tensor":
                            message.tensor.push($root.tensorflow.SavedSliceMeta.decodeText(reader, true));
                            break;
                        case "versions":
                            message.versions = $root.tensorflow.VersionDef.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedTensorSliceMeta;
        })();

        tensorflow.SavedSlice = (function() {

            function SavedSlice() {
            }

            SavedSlice.prototype.name = "";
            SavedSlice.prototype.slice = null;
            SavedSlice.prototype.data = null;

            SavedSlice.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedSlice();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedSlice.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedSlice();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "slice":
                            message.slice = $root.tensorflow.TensorSliceProto.decodeText(reader, true);
                            break;
                        case "data":
                            message.data = $root.tensorflow.TensorProto.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedSlice;
        })();

        tensorflow.SavedTensorSlices = (function() {

            function SavedTensorSlices() {
            }

            SavedTensorSlices.prototype.meta = null;
            SavedTensorSlices.prototype.data = null;

            SavedTensorSlices.decode = function (reader, length) {
                const message = new $root.tensorflow.SavedTensorSlices();
                const end = reader.next(length);
                while (reader.end(end)) {
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
            };

            SavedTensorSlices.decodeText = function (reader) {
                const message = new $root.tensorflow.SavedTensorSlices();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "meta":
                            message.meta = $root.tensorflow.SavedTensorSliceMeta.decodeText(reader, true);
                            break;
                        case "data":
                            message.data = $root.tensorflow.SavedSlice.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return SavedTensorSlices;
        })();

        return tensorflow;
    })();

    $root.google = (function() {

        const google = {};

        google.protobuf = (function() {

            const protobuf = {};

            protobuf.Any = (function() {

                function Any() {
                }

                Any.prototype.type_url = "";
                Any.prototype.value = new Uint8Array([]);

                Any.decode = function (reader, length) {
                    const message = new $root.google.protobuf.Any();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                Any.decodeText = function (reader) {
                    const message = new $root.google.protobuf.Any();
                    reader.start();
                    if (reader.any(message))
                        return message;
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "type_url":
                                message.type_url = reader.string();
                                break;
                            case "value":
                                message.value = reader.bytes();
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return Any;
            })();

            return protobuf;
        })();

        return google;
    })();
    return $root;
})(protobuf);
